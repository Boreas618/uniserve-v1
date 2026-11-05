"""How the package sweep files a tree check's exit status.

`tools/run_package_checks.py` calls the instruments that answer about the whole tree -
`tools/kernel_arguments.py` is today's - and every one of them states its verdict through its exit
status. The mapping is the one place a harness can turn a defect into a silence: a check that ran and
disagreed must be `failed`, the loudest outcome; a check this runner could not call at all, because the
tool is not in the tree or ended in an exception, must be `unrunnable` with the reason, never `failed`
(which would report a kernel defect nobody can act on) and never `skipped` (the quietest outcome, and
the one nineteen checks sat under for weeks).
"""

import importlib.util
import pathlib
import textwrap

import pytest

pytestmark = pytest.mark.unit

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def sweep():
    """The sweep script, imported by path: it is a program under `tools/`, not a package."""
    path = REPOSITORY / "tools" / "run_package_checks.py"
    spec = importlib.util.spec_from_file_location("run_package_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(root: pathlib.Path, name: str, body: str) -> None:
    """Write a stand-in tool under `root/tools`, the only path the sweep looks in."""
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / name).write_text(textwrap.dedent(body))


def test_a_tool_out_of_the_tree_is_unrunnable_and_names_the_absence(sweep, tmp_path):
    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", ("--check",), 60)

    assert outcome == "unrunnable"
    assert "tools/kernel_arguments.py is not in this tree" in detail


def test_a_tool_that_agrees_passes(sweep, tmp_path):
    install(
        tmp_path,
        "kernel_arguments.py",
        """
        import sys
        print("83 operator(s) read, 0 shape(s) with a problem in 0 cause(s), 24 gap(s)")
        sys.exit(0)
        """,
    )

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", ("--check",), 60)

    assert outcome == "passed"
    assert "83 operator(s) read" in detail


def test_a_disagreement_fails_and_names_the_cause(sweep, tmp_path):
    install(
        tmp_path,
        "kernel_arguments.py",
        """
        import sys
        print("2 operator(s) read, 3 shape(s) with a problem in 2 cause(s), 0 gap(s)")
        print("  problem: rope_apply/triton {..}: rope.rs packs 7 word(s) and the signature states 6")
        print("  problem: weighted_rms_norm/triton {..}: packs 5 word(s) and the signature states 4")
        sys.exit(1)
        """,
    )

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", ("--check",), 60)

    assert outcome == "failed"
    assert "rope_apply" in detail
    assert "+1 more cause(s)" in detail


def test_a_tool_that_ends_in_an_exception_is_a_gap_in_the_harness(sweep, tmp_path):
    install(
        tmp_path,
        "kernel_arguments.py",
        """
        raise RuntimeError("no Triton in this interpreter")
        """,
    )

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", ("--check",), 60)

    assert outcome == "unrunnable"
    assert "RuntimeError: no Triton in this interpreter" in detail


def test_a_usage_refusal_is_unrunnable_and_quotes_the_refusal(sweep, tmp_path):
    install(
        tmp_path,
        "kernel_arguments.py",
        """
        import sys
        print("--fixture needs --store: the cells are what the comparison reads")
        sys.exit(2)
        """,
    )

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", ("--check", "x"), 60)

    assert outcome == "unrunnable"
    assert "--fixture needs --store" in detail


def test_the_store_is_the_one_the_runner_is_given(sweep):
    """A store is passed on, never built here: two stores for one run are two statements of one thing."""
    arguments = sweep.tree_arguments(
        "kernel_arguments.py", "notes/programs/qwen3-moe-keys.txt", "/tmp/store", "sm_100a"
    )

    assert arguments == (
        "--check",
        "--fixture",
        "notes/programs/qwen3-moe-keys.txt",
        "--store",
        "/tmp/store",
        "--target",
        "sm_100a",
    )


def test_without_a_fixture_the_check_reads_sources_and_needs_no_store(sweep):
    assert sweep.tree_arguments("kernel_arguments.py") == ("--check",)
