#!/usr/bin/env python3
"""Pins the property list: each property's entry of properties.toml, but its
status, and the module that holds its formulas, by SHA-256, in pins.sha256.

    pins.py check     fails when a pinned entry or file differs from its pin
    pins.py pin ID    writes the pin of one property (a new one, or one changed on purpose)
"""
import hashlib
import os
import re
import sys

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "properties")
PINS = os.path.join(HERE, "pins.sha256")


def entries():
    """The entries of properties.toml: id -> (text of the entry, its file)."""
    text = open(os.path.join(HERE, "properties.toml")).read()
    found = {}
    for block in text.split("[[property]]")[1:]:
        block = "[[property]]" + block.rstrip() + "\n"
        name = re.search(r'^id = "(\w+)"', block, re.M).group(1)
        file = re.search(r'^file = "([\w.]+)"', block, re.M).group(1)
        found[name] = (block, file)
    return found


def digests(name):
    block, file = entries()[name]
    # A property's status records how far it was checked; it is not pinned.
    block = "".join(line for line in block.splitlines(True) if not line.startswith("status = "))
    return (hashlib.sha256(block.encode()).hexdigest(),
            hashlib.sha256(open(os.path.join(HERE, file), "rb").read()).hexdigest(), file)


def pinned():
    pins = {}
    if os.path.exists(PINS):
        for line in open(PINS):
            name, entry, file_digest, file = line.split()
            pins[name] = (entry, file_digest, file)
    return pins


def main():
    pins = pinned()
    if sys.argv[1:2] == ["pin"]:
        pins[sys.argv[2]] = digests(sys.argv[2])
        with open(PINS, "w") as out:
            for name in sorted(pins):
                out.write(" ".join((name,) + pins[name]) + "\n")
        return 0
    failed = False
    for name in sorted(entries()):
        if name not in pins:
            print(f"{name}: not pinned"); failed = True
        elif digests(name) != pins[name]:
            print(f"{name}: differs from its pin"); failed = True
    for name in sorted(set(pins) - set(entries())):
        print(f"{name}: pinned and no longer listed"); failed = True
    if not failed:
        print(f"{len(pins)} properties match their pins")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
