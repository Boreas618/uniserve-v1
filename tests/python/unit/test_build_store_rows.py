"""How a key fixture's rows are read, and what a row may state.

A fixture row is `op<TAB>descriptor`, and it may state a third field: the
backend the deployment launches that key on. The field exists because the
backend is not a property of the key - `paged_attention` declares `bf16`/`f16`
on `trtllm` and `f8e4m3` on `triton`, so a store built from a two-column fixture
chooses a backend by coverage, which is not necessarily the one the deployment
names. A row that states it is held to: the store builds that package or
refuses.
"""

import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.unit

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def store():
    """`tools/build_store.py`, imported by path.

    It is a program under `tools/`, not a package.
    """
    path = REPOSITORY / "tools" / "build_store.py"
    spec = importlib.util.spec_from_file_location("build_store", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(tmp_path, text):
    fixture = tmp_path / "keys.txt"
    fixture.write_text(text)
    return fixture


def test_a_row_may_state_the_backend_it_launches_on(store, tmp_path):
    fixture = write(
        tmp_path,
        "# a comment\n"
        "add_rms_norm\tdtype_bf16\n"
        "paged_attention\tphase_decode_kv_f8e4m3\ttriton\n",
    )

    assert store.read_rows(fixture) == [
        ("add_rms_norm", "dtype_bf16", None),
        ("paged_attention", "phase_decode_kv_f8e4m3", "triton"),
    ]


def test_a_row_of_neither_shape_is_refused_by_name(store, tmp_path):
    fixture = write(tmp_path, "add_rms_norm\n")

    with pytest.raises(SystemExit) as refused:
        store.read_rows(fixture)

    message = str(refused.value)
    assert "1 field(s)" in message
    assert "op<TAB>descriptor<TAB>backend" in message
