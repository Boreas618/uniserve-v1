"""Pinned sources: the declared pins, and a build's staged copy of one.

A pin of Python sources states which of its directories are packages, and a
build imports them from its own staged copy of the files under their own names
(``fetch.load``); a pin of C++ sources states the directories a compilation
includes (``fetch.includes``). A header set of a locked distribution is used
where the distribution installs it (``fetch.installed``). These tests stage
small trees of their own and read the packaged pins, so no pin is fetched.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from uniserve_build_server import fetch

#: The prefix of the packages these tests stage, which no other module uses.
PREFIX = "uniserve_staged_"


@pytest.fixture
def staged(tmp_path):
    """A staged tree whose packages name one another as top-level modules."""
    first = tmp_path / "vendor" / "src" / f"{PREFIX}first"
    second = tmp_path / "vendor" / "src" / f"{PREFIX}second"
    loose = tmp_path / "vendor" / "tools" / f"{PREFIX}loose"
    for directory in (first, second, loose):
        directory.mkdir(parents=True)

    # The first package imports a module of the second, which comes later.
    (first / "__init__.py").write_text(
        f"from {PREFIX}second.value import VALUE\n"
    )
    (second / "__init__.py").write_text("")
    (second / "value.py").write_text("VALUE = 7\n")
    # A directory the pin lists no `__init__.py` of: a namespace package.
    (loose / "tool.py").write_text("NAME = 'tool'\n")

    yield tmp_path

    for name in [name for name in sys.modules if name.startswith(PREFIX)]:
        del sys.modules[name]


def pin(*packages: str) -> fetch.Pin:
    return fetch.Pin(
        name="staged",
        sha256="0" * 64,
        origin=fetch.Distribution("staged"),
        packages=packages,
    )


def test_a_pins_packages_import_from_the_staged_copy(staged):
    path = list(sys.path)

    fetch.load(
        pin(
            f"vendor/src/{PREFIX}first",
            f"vendor/src/{PREFIX}second",
            f"vendor/tools/{PREFIX}loose",
        ),
        staged,
    )

    first = sys.modules[f"{PREFIX}first"]
    assert first.VALUE == 7
    assert first.__file__ == str(
        staged / "vendor" / "src" / f"{PREFIX}first" / "__init__.py"
    )
    tool = importlib.import_module(f"{PREFIX}loose.tool")
    assert tool.NAME == "tool"
    assert sys.path == path


def test_a_package_the_process_already_imported_is_refused(staged):
    (staged / "vendor" / "src" / "json").mkdir()

    with pytest.raises(fetch.FetchError, match="already imported"):
        fetch.load(pin("vendor/src/json"), staged)


@pytest.mark.parametrize(
    "declared",
    [one for one in fetch.pins().values() if one.manifest is None],
    ids=lambda one: one.name,
)
def test_a_packaged_manifest_is_the_pins_and_holds_its_include_directories(
    declared, tmp_path
):
    # The manifest is refused unless its digest is the one the pin names.
    files = fetch.manifest(declared, tmp_path)
    if not declared.include:
        return

    # Every include directory holds files of the pin.
    for directory in fetch.includes(declared, tmp_path):
        relative = directory.relative_to(tmp_path).as_posix()
        assert any(name.startswith(f"{relative}/") for name in files), relative


def test_a_pin_without_include_directories_is_refused(tmp_path):
    with pytest.raises(fetch.FetchError, match="no include directory"):
        fetch.includes(pin(), tmp_path)


def test_a_locked_distribution_is_used_where_it_installs_a_path():
    found = fetch.installed("urllib3", "urllib3/__init__.py")

    assert found.is_file()
    with pytest.raises(fetch.FetchError, match="installs no"):
        fetch.installed("urllib3", "urllib3/absent")
    with pytest.raises(fetch.FetchError, match="has no"):
        fetch.installed("uniserve-absent-distribution", "include")
