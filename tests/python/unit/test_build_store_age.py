"""Whether a store's cells were built from the source this tree holds.

A store outlives the tree it was built from, and a key names the descriptor
rather than the source, so a cell built from an older package installs under a
key that still matches. Every cell `build_store.py` writes therefore records
the package and each source file's digest, and `verify` answers for a fixture
whether those cells are this tree's.

Three verdicts are the contract. A changed source refuses the cell by name and
says which file. A store that states no source anywhere installs as today,
because a Program Server's store states none and refusing it would refuse the
path that ships. A cell without a record, in a store that states them
elsewhere, cannot be told from a stale one, which is what a partly rebuilt
store is.
"""

import hashlib
import importlib.util
import json
import pathlib

import pytest

pytestmark = pytest.mark.unit

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]

MANIFEST = """op = "{op}"
backend = "{backend}"

[module]
form = "cubin"
entries = {{ launch = "uniserve_{op}" }}

[source]
kind = "triton"
files = ["src/kernel.py"]
script = "src/kernel.py"

[kernel]
target = "arch"
architectures = ["sm_100a"]
descriptors = {{ keys = ["dtype_bf16"] }}
"""

ROWS = [("add_rms_norm", "dtype_bf16", None)]


@pytest.fixture(scope="module")
def store():
    """`tools/build_store.py` by path: a program under `tools/`."""
    path = REPOSITORY / "tools" / "build_store.py"
    spec = importlib.util.spec_from_file_location("build_store", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_package(root, op, backend="triton"):
    """One package tree under `<root>/kernels`, and its path."""
    tree = root / "kernels" / op / backend
    (tree / "src").mkdir(parents=True)
    (tree / "package.toml").write_text(MANIFEST.format(op=op, backend=backend))
    (tree / "src" / "kernel.py").write_text("# the program\n")
    return tree


def record_of(package):
    """The record a build writes: the package and each source file's digest."""
    files = {}
    for path in sorted(package["root"].rglob("*")):
        if path.is_file():
            files[path.relative_to(package["root"]).as_posix()] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
    return {
        "package": package["root"].as_posix(),
        "target": "sm_100a",
        "files": files,
    }


def write_cell(store_root, package, record):
    """One committed cell, with the record a build writes or with none."""
    directory = (
        store_root
        / package["op"]
        / package["backend"]
        / "1"
        / "sm_100a"
        / "dtype_bf16"
    )
    directory.mkdir(parents=True)
    (directory / "module.cubin").write_bytes(b"")
    (directory / ".committed").write_text("")
    if record is not None:
        (directory / "source.json").write_text(json.dumps(record, indent=1))


def test_a_changed_source_refuses_the_cell_and_names_the_file(
    store, tmp_path, capsys
):
    tree = write_package(tmp_path, "add_rms_norm")
    table = store.load_packages(tmp_path / "kernels")
    write_cell(
        tmp_path / "store",
        table["add_rms_norm"][0],
        record_of(table["add_rms_norm"][0]),
    )
    (tree / "src" / "kernel.py").write_text(
        "# the program, with an argument's order moved\n"
    )

    verdict = store.verify(
        tmp_path / "store", ROWS, table, "sm_100a", 1, "rows"
    )

    printed = capsys.readouterr().out
    assert verdict == 1
    assert "stale\tadd_rms_norm\ttriton\tdtype_bf16:" in printed
    assert "src/kernel.py changed" in printed


def test_a_store_that_states_no_source_installs_as_today(
    store, tmp_path, capsys
):
    tree = write_package(tmp_path, "add_rms_norm")
    table = store.load_packages(tmp_path / "kernels")
    write_cell(tmp_path / "store", table["add_rms_norm"][0], None)
    (tree / "src" / "kernel.py").write_text(
        "# a source that moved since the build\n"
    )

    verdict = store.verify(
        tmp_path / "store", ROWS, table, "sm_100a", 1, "rows"
    )

    printed = capsys.readouterr().out
    assert verdict == 0
    assert "states no source digest" in printed


def test_a_cell_without_a_record_is_refused_where_the_store_states_one(
    store, tmp_path, capsys
):
    write_package(tmp_path, "add_rms_norm")
    write_package(tmp_path, "add_scale")
    table = store.load_packages(tmp_path / "kernels")
    write_cell(
        tmp_path / "store",
        table["add_rms_norm"][0],
        record_of(table["add_rms_norm"][0]),
    )
    write_cell(tmp_path / "store", table["add_scale"][0], None)
    rows = ROWS + [("add_scale", "dtype_bf16", None)]

    verdict = store.verify(
        tmp_path / "store", rows, table, "sm_100a", 1, "rows"
    )

    printed = capsys.readouterr().out
    assert verdict == 1
    assert (
        "stale\tadd_scale\ttriton\tdtype_bf16: the cell states no source digest"
        in printed
    )
    assert "stale\tadd_rms_norm" not in printed
