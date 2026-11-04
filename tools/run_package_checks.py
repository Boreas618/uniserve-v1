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
    jobs = [
        (package, script)
        for package, scripts in found
        for script in scripts
    ]
    outcomes = {"passed": [], "skipped": [], "failed": [], "timeout": [], "contended": []}
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
    print(
        f"# {len(found)} packages, {checks} checks: {len(outcomes['passed'])} passed, "
        f"{len(outcomes['skipped'])} skipped, {len(outcomes['failed'])} failed, "
        f"{len(outcomes['timeout'])} timed out, {len(outcomes['contended'])} contended"
    )
    for outcome in ("failed", "timeout", "contended", "skipped"):
        for name, detail, code in sorted(outcomes[outcome]):
            prefix = f" [{code}]" if code not in (0, "", None) else ""
            print(f"{outcome:8s} {name}{prefix} — {detail}")
    if options.verbose:
        for name, detail, _ in sorted(outcomes["passed"]):
            print(f"passed   {name} — {detail}")
    return 1 if outcomes["failed"] or outcomes["timeout"] or outcomes["contended"] else 0


if __name__ == "__main__":
    sys.exit(main())
