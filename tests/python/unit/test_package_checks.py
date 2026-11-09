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
    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", (("--check",),), 60)

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

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", (("--check",),), 60)

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

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", (("--check",),), 60)

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

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", (("--check",),), 60)

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

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", (("--check", "x"),), 60)

    assert outcome == "unrunnable"
    assert "--fixture needs --store" in detail


def test_each_invocation_carries_the_store_and_the_fixture_it_compares(sweep):
    """The store is passed on, never built here; every mode of the comparison reads the same one."""
    invocations = sweep.tree_arguments(
        "kernel_arguments.py", "keys.txt", "store", "sm_100a"
    )

    assert len(invocations) == 2
    assert any("--native" in invocation for invocation in invocations)
    for invocation in invocations:
        assert invocation[0] == "--check"
        assert invocation[invocation.index("--fixture") + 1] == "keys.txt"
        assert invocation[invocation.index("--store") + 1] == "store"
        assert invocation[invocation.index("--target") + 1] == "sm_100a"


def test_source_mode_is_never_run_because_it_reports_an_operators_arms_against_each_other(sweep):
    """The check runs against a built artifact or not at all: its source mode over-reports."""
    assert sweep.tree_needs("kernel_arguments.py") == ["--fixture", "--store"]
    assert sweep.tree_needs("kernel_arguments.py", "keys.txt") == ["--store"]
    assert sweep.tree_needs("kernel_arguments.py", "keys.txt", "store") == []


def test_an_empty_store_is_not_a_green(sweep, tmp_path):
    """A comparison against a store nothing filled compares nothing, and must not read as agreement."""
    assert not sweep.store_holds_cells(tmp_path)
    (tmp_path / "op" / "triton" / "1" / "sm_100a" / "key").mkdir(parents=True)
    assert not sweep.store_holds_cells(tmp_path)
    (tmp_path / "op" / "triton" / "1" / "sm_100a" / "key" / "metadata.json").write_text("{}")
    assert sweep.store_holds_cells(tmp_path)


def test_a_comparison_that_compared_nothing_is_not_a_pass(sweep, tmp_path):
    """Exit 0 over zero cells is the green that is worse than a refusal."""
    install(
        tmp_path,
        "kernel_arguments.py",
        """
        import sys
        print("0 cell(s) compared in order against the packages that wrote them")
        print("0 problem(s), 3 gap(s)")
        sys.exit(0)
        """,
    )

    outcome, detail, _ = sweep.run_tree_check(tmp_path, "kernel_arguments.py", (("--check",),), 60)

    assert outcome == "unrunnable"
    assert "compared no cell" in detail


def test_two_invocations_are_one_verdict_and_the_louder_mode_wins(sweep, tmp_path):
    """A tool whose subject needs two passes is not passed by the one that agreed."""
    install(
        tmp_path,
        "kernel_arguments.py",
        """
        import sys
        if "--native" in sys.argv:
            print("1 call structure(s) compared, 1 disagreement(s), 0 not compared")
            print("  problem: conv1d/cudnn {..}: the entry reads 11 fields and the Rust structure has 10")
            sys.exit(1)
        print("131 cell(s) compared in order against the packages that wrote them")
        print("0 problem(s), 0 gap(s)")
        sys.exit(0)
        """,
    )

    outcome, detail, _ = sweep.run_tree_check(
        tmp_path, "kernel_arguments.py", (("--check",), ("--check", "--native")), 60
    )

    assert outcome == "failed"
    assert "--check --native" in detail
    assert "conv1d" in detail


def test_both_invocations_clean_is_a_pass_with_each_detail(sweep, tmp_path):
    install(
        tmp_path,
        "kernel_arguments.py",
        """
        import sys
        if "--native" in sys.argv:
            print("21 call structure(s) compared, 0 disagreement(s), 0 not compared")
        else:
            print("131 cell(s) compared in order against the packages that wrote them")
        sys.exit(0)
        """,
    )

    outcome, detail, _ = sweep.run_tree_check(
        tmp_path, "kernel_arguments.py", (("--check",), ("--check", "--native")), 60
    )

    assert outcome == "passed"
    assert "131 cell(s) compared" in detail
    assert "21 call structure(s) compared" in detail
