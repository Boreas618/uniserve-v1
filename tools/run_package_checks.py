#!/usr/bin/env python3
"""Run every package's own check on Triton's interpreter and summarise the outcome.

    python3 tools/run_package_checks.py [--kernels crates/ops/kernels] [--jobs 6] [--verbose]

Each package under `crates/ops/kernels/<op>/<backend>/` carries a `test/` directory whose scripts
check that package's program against a reference. They are the operators' own evidence, and nothing
runs them on a schedule: this does, in one pass, device-free and in seconds, because the Triton checks
set `TRITON_INTERPRET` themselves or accept it from here.

A script is **run** only when it is a check: one with an `if __name__` block. A module a check
imports (`scenario.py` and the like) is not run on its own, and a check that needs a *module
directory* or a device on its command line is reported as **skipped with its reason**, which is what
its own argument parser says.

Outcomes:

- **passed**: exit status 0.
- **skipped**: the check refused before doing arithmetic — it wants a compiled module, a device, or an
  argument this runner does not invent. The first line of its refusal is printed, so a skip that is
  really a defect is visible rather than counted as success.
- **failed**: anything else. Exit status 1 if any check failed.

This is not a test and runs nothing a caller's suite would: it exercises the operators' kernels on the
interpreter, which is where a wrong index or a missing mask is cheapest to find.
"""

import argparse
import concurrent.futures
import os
import pathlib
import subprocess
import sys

#: First lines a check prints when it needs something this runner does not supply. Each is a refusal
#: by the check itself, not a failure of its arithmetic.
SKIP_MARKERS = (
    "needs a device",
    "requires a device",
    "the following arguments are required",
    "this check runs the kernel through Triton's interpreter",
    "ModuleNotFoundError",
    "No module named",
    "run this with a device",
    "needs the device",
    "needs a module directory",
    "module directory is required",
    "a module directory",
)

#: A check's own verdict on the arithmetic it did. A marker says a part of the check could not run
#: here; a verdict says a part that *did* run disagreed with its reference, and that outranks the
#: marker: `modulated_rms_norm` prints "…'s E4M3 output needs a device" for one output and then
#: `FAIL: hand-computed …` for the hand case, and it was reported as skipped — a real arithmetic
#: failure filed under a platform limitation.
FAILURE_VERDICTS = ("FAIL", "DIFFERS")

#: Checks that are kept as **byte-identical copies** in more than one package, because each copy's
#: `default_op()` names its own operator from the package it lives in. A fix landed in one copy and
#: not the other is invisible to every other instrument: the two `modulation_check.py` copies diverged
#: once, and the `gated_residual` copy then launched the modulated form's kernel against the gated
#: literals for a whole stretch. The sweep compares them before it runs anything.
SHARED_CHECKS = ("modulation_check.py",)

#: Where a launch's descriptor is built. A package whose operator no file here names **in code** is
#: a check over a kernel nothing can reach, and its green is not evidence about the model: five
#: packages of exactly this kind were deleted on 2026-11-04 after one of their checks cost an evening.
#: The search strips Rust comments first, because an operator's name in a doc comment is a comment
#: and not a launch — that distinction is what turned "nine deletions" into six.
CALLERS = ("crates/nn/src", "crates/models", "crates/executor/src")


def declared(repository):
    """Every operator `ops` declares, as `name -> Rust type`, read from its own `impl Op` blocks.

    The type is what a caller writes: a layer builds `WeightedRmsNorm`'s descriptor, and the operator
    NAME string appears nowhere at a launch site. So a census that greps names finds nothing and a
    census that greps types finds the callers — that was the one wrong turn this instrument had.
    """
    found = {}
    for source in sorted((repository / "crates/ops/src").rglob("*.rs")):
        text = source.read_text(errors="ignore")
        for block in text.split("impl Op for ")[1:]:
            name = block.split('const NAME: &\'static str = "')
            if len(name) < 2:
                continue
            found[name[1].split('"')[0]] = block.split(None, 1)[0].strip(" {")
    return found


def comments_stripped(text):
    """The file with its Rust comments removed, so a mention in prose does not count as a launch."""
    import re

    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def reachable(repository, operators):
    """The operators whose type at least one caller crate names in code."""
    bodies = []
    for where in CALLERS:
        for source in sorted((repository / where).rglob("*.rs")):
            bodies.append(comments_stripped(source.read_text(errors="ignore")))
    body = "\n".join(bodies)
    import re

    return {name for name, kind in operators.items() if re.search(rf"\b{kind}\b", body)}


def is_check(path):
    """Whether `path` is a check rather than a module a check imports."""
    text = path.read_text(errors="ignore")
    return "if __name__" in text


def packages(root):
    """Every package directory holding a package.toml, with its test scripts."""
    found = []
    for manifest in sorted(root.rglob("package.toml")):
        package = manifest.parent
        tests = package / "test"
        if not tests.is_dir():
            continue
        scripts = [script for script in sorted(tests.glob("*.py")) if is_check(script)]
        found.append((package, scripts))
    return found


def run_one(package, script, timeout):
    """Run one check in its own directory with the interpreter enabled."""
    environment = dict(os.environ)
    environment.setdefault("TRITON_INTERPRET", "1")
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            [sys.executable, script.name],
            cwd=str(script.parent),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ("timeout", f"{script.name}: no answer in {timeout} s", "")
    output = (completed.stdout + completed.stderr).strip().splitlines()
    tail = output[-1] if output else ""
    head = output[0] if output else ""
    if completed.returncode == 0:
        return ("passed", tail, "")
    verdict = next(
        (line for line in output if line.startswith(FAILURE_VERDICTS)), None
    )
    if verdict is not None:
        return ("failed", verdict, "")
    if any(marker in head or marker in tail for marker in SKIP_MARKERS):
        return ("skipped", head or tail, "")
    if completed.returncode < 0:
        # A check killed by a signal may be a check that dies when several interpreter runs
        # import the same libraries at once rather than one that is wrong. It is re-run alone
        # before it is reported, and it is reported under its own outcome either way: a check
        # whose verdict depends on how many ran beside it is a finding, not a pass.
        alone = run_one_alone(script, timeout)
        if alone.returncode == 0:
            return (
                "contended",
                f"died of signal {-completed.returncode} beside other checks; passes alone: "
                + (alone.stdout + alone.stderr).strip().splitlines()[-1:][0]
                if (alone.stdout + alone.stderr).strip()
                else "",
                completed.returncode,
            )
        return ("failed", tail or head, completed.returncode)
    return ("failed", tail or head, completed.returncode)


def run_one_alone(script, timeout):
    """Run one check again with nothing beside it, to separate contention from a defect."""
    environment = dict(os.environ)
    environment.setdefault("TRITON_INTERPRET", "1")
    environment.pop("PYTHONPATH", None)
    try:
        return subprocess.run(
            [sys.executable, script.name],
            cwd=str(script.parent),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 124, "", "no answer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", default="crates/ops/kernels", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=6, help="checks run at once")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per check")
    parser.add_argument("--verbose", action="store_true", help="name every check, not only outcomes")
    options = parser.parse_args()

    found = packages(options.kernels)
    diverged = []
    for name in SHARED_CHECKS:
        copies = sorted(script for _, scripts in found for script in scripts if script.name == name)
        if len(copies) > 1 and len({script.read_bytes() for script in copies}) > 1:
            diverged.append(
                f"{name}: {len(copies)} copies are not identical — "
                + ", ".join(str(script.parent.parent.parent.name) for script in copies)
            )
    jobs = [
        (package, script)
        for package, scripts in found
        for script in scripts
    ]
    outcomes = {
        "passed": [],
        "skipped": [],
        "failed": [],
        "timeout": [],
        "contended": [],
        "unreachable": [],
    }
    for line in diverged:
        outcomes["failed"].append((line.split(":")[0], line, 1))
    operators = declared(pathlib.Path("."))
    live = reachable(pathlib.Path("."), operators)
    # `package` is the backend directory (`kernels/<operator>/<backend>`), so the operator is the
    # first component below the kernels root — `package.name` would be `triton`, `fa4`, `cudnn`.
    seen = set()
    for package, _ in found:
        name = package.relative_to(options.kernels).parts[0]
        if name in seen:
            continue
        seen.add(name)
        if name in operators and name in live:
            continue
        reason = (
            f"{operators[name]}: no crate outside `ops` names it in code, so its checks prove "
            f"nothing about the model"
            if name in operators
            else "`ops` declares no operator of this name"
        )
        outcomes["unreachable"].append((name, reason, 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.jobs) as pool:
        futures = {
            pool.submit(run_one, package, script, options.timeout): (package, script)
            for package, script in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            package, script = futures[future]
            outcome, detail, code = future.result()
            name = f"{package.relative_to(options.kernels)}/{script.name}"
            outcomes[outcome].append((name, detail, code))

    checks = sum(len(scripts) for _, scripts in found)
    # A package directory is one backend of one operator, so the two counts differ where an operator
    # has more than one backend. Both are stated because a count is quoted without its unit, and an
    # operator with two backends read as two operators is a number nobody can reconcile later.
    operators_count = len({package.relative_to(options.kernels).parts[0] for package, _ in found})
    print(
        f"# {len(found)} packages ({operators_count} operators), {checks} checks: "
        f"{len(outcomes['passed'])} passed, "
        f"{len(outcomes['skipped'])} skipped, {len(outcomes['failed'])} failed, "
        f"{len(outcomes['timeout'])} timed out, {len(outcomes['contended'])} contended, "
        f"{len(outcomes['unreachable'])} unreachable"
    )
    for outcome in ("failed", "timeout", "contended", "unreachable", "skipped"):
        for name, detail, code in sorted(outcomes[outcome]):
            prefix = f" [{code}]" if code not in (0, "", None) else ""
            print(f"{outcome:8s} {name}{prefix} — {detail}")
    if options.verbose:
        for name, detail, _ in sorted(outcomes["passed"]):
            print(f"passed   {name} — {detail}")
    # `unreachable` does not fail the run: it says a package exists that no caller can launch, which
    # is a question for the operator set's owner, not a check that went wrong. What it must not do is
    # disappear, so it is in the count line, in the detail lines, and out of the passed count.
    return 1 if outcomes["failed"] or outcomes["timeout"] or outcomes["contended"] else 0


if __name__ == "__main__":
    sys.exit(main())
