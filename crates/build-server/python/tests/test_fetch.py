"""Pinned sources: importing a pin's packages from a build's staged copy.

A pin of Python sources states which of its directories are packages, and a
build imports them from its own staged copy of the files under their own names
(``fetch.load``). These tests stage small trees of their own, so no pin is
fetched.
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
