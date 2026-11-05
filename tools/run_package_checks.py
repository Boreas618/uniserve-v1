#!/usr/bin/env python3
"""Run every package's own check on Triton's interpreter and summarise the outcome.

    python3 tools/run_package_checks.py [--kernels crates/ops/kernels] [--jobs 6] [--verbose]

Each package under `crates/ops/kernels/<op>/<backend>/` carries a `test/` directory whose scripts
check that package's program against a reference. They are the operators' own evidence, and nothing
runs them on a schedule: this does, in one pass, device-free and in seconds, because the Triton checks
set `TRITON_INTERPRET` themselves or accept it from here.

A script is **run** only when it is a check: one with an `if __name__` block. A module a check
imports (`scenario.py` and the like) is not run on its own.

**A tree check answers about the whole tree rather than about one package** (`TREE_CHECKS`): today
`tools/kernel_arguments.py --check`, which compares the three statements of every Triton kernel's
argument list — the kernel's `def`, the package's `spec`, and the operator's Rust `Arguments` chain.
It is one check in the count line like any other, and its verdict is `passed`, `failed` or
`unrunnable`; never `skipped`, because a disagreement it finds is a defect and an absence of the tool
is a gap in the harness. An instrument a run never calls is a gap this runner exists to name: nineteen
checks read as device-dependent for weeks because nothing called them with the argument they wanted.

**A check that refuses for a missing argument is called again with the arguments this runner can
answer** — today that is `--op`, whose value is the operator the package is for, since it is the
directory the check lives under. One that needs an argument nobody here can answer is **unrunnable**,
named with the argument: that is a gap in the instrument, not a property of the kernel. The runner
does not compile a module to feed one, because a check that loads a module launches it, so it is a
device check, and a device check is skipped here by design rather than compiled for and then skipped.

Outcomes:

- **passed**: exit status 0.
- **skipped**: the check refused before doing arithmetic for something this host does not have — a
  device, a compiled module, a library. The line it printed is shown, so a skip that is really a
  defect stays visible rather than counting as success.
- **unrunnable**: this runner cannot call the check at all; the line names the argument it would have
  needed. Reported, not failed, for the same reason `unreachable` is: nobody's kernel is wrong.
- **failed**: anything else. Exit status 1 if any check failed, timed out or was contended.

This is not a test and runs nothing a caller's suite would: it exercises the operators' kernels on the
interpreter, which is where a wrong index or a missing mask is cheapest to find.
"""

import argparse
import concurrent.futures
import os
import pathlib
import re
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

#: Arguments a check declares required that this runner can answer from where the check lives. A
#: required argument the runner cannot answer makes a check **unrunnable** rather than skipped: "wants
#: a device" is a statement about the host, and "the harness does not know how to call it" is a
#: statement about the harness — filing the second under the first is how nineteen checks read as
#: device-dependent for weeks (`usage: conv_check.py [-h] --op {conv1d,conv3d,conv_transpose1d}`).
DERIVABLE = {
    # `--op` names the operator the package is for, which is where the script lives.
    "--op": lambda operator, script: operator,
}

#: What argparse prints when it refuses for a missing argument, with the names that were required.
REQUIRED = re.compile(r"the following arguments are required: (.+)$", re.MULTILINE)

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

#: Checks that are not a package's own but the whole tree's, run by name from `tools/`. Each is one
#: check in the count line. They are listed here rather than discovered, because a tool that refuses
#: by not existing must still be one entry in the count: an instrument missing from the tree is a gap
#: to report, not a check that disappears from the arithmetic.
#:
#: `kernel_arguments.py` compares the three statements of every Triton kernel's argument list. Without
#: a store it reads each package's declared shapes against the operator's Rust chain, which needs no
#: device and no build; with `--fixture` and `--store` it compares the *built* cells of those keys,
#: whose `metadata.json` is what a launch checks. This runner builds no store of its own - a second
#: store for one run is two statements of one thing - so it passes on the one it is given and names
#: the pair it would need otherwise.
TREE_CHECKS = (("kernel_arguments.py", ("--check",)),)

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


def offers(script, flag):
    """Whether a check's own argument parser lists `flag`."""
    environment = dict(os.environ)
    environment.setdefault("TRITON_INTERPRET", "1")
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            [sys.executable, script.name, "--help"],
            cwd=str(script.parent),
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return flag in (completed.stdout + completed.stderr)


def run_one(package, script, timeout, operator=None, arguments=(), retried=False, cpu_only=False):
    """Run one check in its own directory with the interpreter enabled.

    A check that refuses for a missing argument is called once more with the arguments this runner
    can answer (`DERIVABLE`); one that needs an argument nobody here can answer is `unrunnable`.
    """
    environment = dict(os.environ)
    environment.setdefault("TRITON_INTERPRET", "1")
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            [sys.executable, script.name, *arguments],
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
    missing = REQUIRED.search("\n".join(output))
    if missing is not None:
        names = [one.strip() for one in missing.group(1).split(",") if one.strip()]
        unknown = [one for one in names if one not in DERIVABLE]
        if unknown:
            # A required name that is not a flag is a positional: either an input this runner does
            # not produce (a compiled module) or a choice among the check's own cases. Both are the
            # harness's gap and neither is answered by writing a value into the check's mouth.
            flags = [one for one in unknown if one.startswith("-")]
            positional = [one for one in unknown if not one.startswith("-")]
            needs = []
            if flags:
                needs.append(f"the arguments {', '.join(flags)}")
            if positional:
                needs.append(f"its input {', '.join(positional)}")
            return (
                "unrunnable",
                f"{script.name}: this runner supplies {' and '.join(needs)}, so the check is not "
                f"called and proves nothing here",
                2,
            )
        if not retried:
            supplied = []
            for one in names:
                supplied.extend((one, DERIVABLE[one](operator, script)))
            return run_one(package, script, timeout, operator, tuple(supplied), retried=True)
        # The second call refused for an argument this runner did supply, so the refusal is not one
        # the table can answer and the check stays uncalled rather than passed off as a skip.
        return (
            "unrunnable",
            f"{script.name}: still refusing for {', '.join(names)} after they were supplied",
            2,
        )
    if any(marker in head or marker in tail for marker in SKIP_MARKERS):
        # A check that offers `--cpu-only` can be held to the half of itself that does not need a
        # device, and that half is a real assertion: `gemm_check.py` refuses without a module and
        # then passes `--cpu-only` with four verdicts it had never been asked for here.
        if not cpu_only and offers(script, "--cpu-only"):
            again = run_one(
                package, script, timeout, operator,
                tuple(arguments) + ("--cpu-only",), retried=True, cpu_only=True,
            )
            if again[0] == "passed":
                return ("passed", f"host-only half: {again[1]}", 0)
            if again[0] == "failed":
                return again
            return ("skipped", f"{head or tail} (and --cpu-only: {again[1]})", "")
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


def tree_arguments(name, fixture=None, store=None, target="sm_100a"):
    """The arguments one tree check is called with.

    The store is one this runner is **given**, never one it builds: a second store for one run is two
    statements of one thing, and this runner is the place they would drift apart. Without a fixture
    `kernel_arguments.py` reads each package's declared shapes against the operator's Rust chain, which
    needs neither a device nor a build.
    """
    arguments = dict(TREE_CHECKS)[name]
    if name == "kernel_arguments.py" and fixture:
        arguments = (*arguments, "--fixture", fixture, "--store", store, "--target", target)
    return arguments


def run_tree_check(root, name, arguments, timeout):
    """Run one tree check and classify it the way a package check is classified.

    `passed` for exit 0 — the check may still report gaps, and its own line counts them; `failed` for
    exit 1 with a report, which is the check saying a shape disagrees, and the line names the cause;
    `unrunnable` for anything else, including the tool not being in this tree, because a check this
    runner cannot call is the harness's gap and not a verdict about a kernel.
    """
    script = pathlib.Path(root) / "tools" / name
    if not script.is_file():
        return (
            "unrunnable",
            f"tools/{name} is not in this tree, so nothing compares the kernels' argument lists",
            "",
        )
    try:
        completed = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ("timeout", f"tools/{name} did not answer in {timeout} s", "")
    output = (completed.stdout + completed.stderr).strip().splitlines()
    summary = output[0] if output else ""
    if completed.returncode == 0:
        return ("passed", summary, "")
    if completed.returncode == 1 and not any("Traceback (most recent call last)" in line for line in output):
        problems = [line.strip() for line in output if line.strip().startswith("problem:")]
        named = problems[0] if problems else summary
        extra = f" (+{len(problems) - 1} more cause(s))" if len(problems) > 1 else ""
        return ("failed", f"{named}{extra}", "")
    # A tool that ends in an exception did not compare anything, so its exit status is not a verdict
    # about a kernel: the last line of the traceback is the reason, and the outcome is the harness's
    # gap. Reading a crash as a disagreement would report a defect nobody can act on.
    reason = output[-1].strip() if output else "no output"
    return ("unrunnable", f"tools/{name} could not run here: {reason}", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", default="crates/ops/kernels", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=6, help="checks run at once")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per check")
    parser.add_argument("--verbose", action="store_true", help="name every check, not only outcomes")
    parser.add_argument(
        "--fixture",
        help="the key fixture a tree check compares built cells against (needs --store)",
    )
    parser.add_argument("--store", help="a store already built for --fixture")
    parser.add_argument("--target", default="sm_100a", help="the architecture --store was built for")
    options = parser.parse_args()
    if options.fixture and not options.store:
        parser.error("--fixture needs --store: the cells are what the comparison reads")

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
        "unrunnable": [],
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
            pool.submit(
                run_one, package, script, options.timeout,
                package.relative_to(options.kernels).parts[0],
            ): (package, script)
            for package, script in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            package, script = futures[future]
            outcome, detail, code = future.result()
            name = f"{package.relative_to(options.kernels)}/{script.name}"
            outcomes[outcome].append((name, detail, code))

    # A tree check stands outside `packages()`: there is no package directory under it, and its
    # subject is every package at once. It runs after the packages so its output reads beside theirs.
    for name, _ in TREE_CHECKS:
        arguments = tree_arguments(name, options.fixture, options.store, options.target)
        outcome, detail, code = run_tree_check(pathlib.Path("."), name, arguments, options.timeout)
        outcomes[outcome].append((f"tools/{name}", detail, code))

    checks = sum(len(scripts) for _, scripts in found) + len(TREE_CHECKS)
    # A package directory is one backend of one operator, so the two counts differ where an operator
    # has more than one backend. Both are stated because a count is quoted without its unit, and an
    # operator with two backends read as two operators is a number nobody can reconcile later.
    operators_count = len({package.relative_to(options.kernels).parts[0] for package, _ in found})
    parts = (
        ("passed", len(outcomes["passed"])),
        ("skipped", len(outcomes["skipped"])),
        ("failed", len(outcomes["failed"])),
        ("timed out", len(outcomes["timeout"])),
        ("contended", len(outcomes["contended"])),
        ("unrunnable", len(outcomes["unrunnable"])),
    )
    print(
        f"# {len(found)} packages ({operators_count} operators), {checks} checks: "
        + ", ".join(f"{count} {word}" for word, count in parts)
        + f"; {len(outcomes['unreachable'])} unreachable operator(s)"
    )
    # The arithmetic is printed because a count that does not close is a finding rather than a
    # rounding: every check has exactly one outcome, and `unreachable` is a statement about an
    # operator (a package no launch names) rather than about a check, so it is not in the sum.
    total = sum(count for _, count in parts)
    closing = " + ".join(f"{count} {word}" for word, count in parts) + f" = {total}"
    print(f"# {total} checks by outcome: {closing}" + ("" if total == checks else f" — NOT {checks}"))
    for outcome in ("failed", "timeout", "contended", "unreachable", "unrunnable", "skipped"):
        for name, detail, code in sorted(outcomes[outcome]):
            prefix = f" [{code}]" if code not in (0, "", None) else ""
            print(f"{outcome:8s} {name}{prefix} — {detail}")
    if options.verbose:
        for name, detail, _ in sorted(outcomes["passed"]):
            print(f"passed   {name} — {detail}")
    # `unreachable` does not fail the run: it says a package exists that no caller can launch, which
    # is a question for the operator set's owner, not a check that went wrong. What it must not do is
    # disappear, so it is in the count line, in the detail lines, and out of the passed count.
    # `unrunnable` is the same shape for the harness: a check this runner cannot call is a gap in the
    # instrument, named with the argument it would have needed, and not a verdict about the kernel.
    return 1 if outcomes["failed"] or outcomes["timeout"] or outcomes["contended"] else 0


if __name__ == "__main__":
    sys.exit(main())
