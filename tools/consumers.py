#!/usr/bin/env python3
"""List every place the tree names a file, so an absence claim is a list.

A search answers "where is this named" only if its output is read whole, and
six times in one lane it was not: a pattern piped through `head`, or a `grep`
whose include list was narrower than the tree, produced a short list that was
then reported as "nothing reads this". The defect is not the pattern - it is
that a truncated view and a complete one look the same.

This prints the whole answer instead: every line of every tracked file that
names the path, grouped by file, with the number of files and lines at the end,
and no option that shortens the list. A file named here is a consumer to check;
the command does not decide whether a hit is a read, a default, a doc reference
or a comment, because that judgement needs the line, and the line is printed.

Usage:
    tools/consumers.py <path within the tree> [--root <directory>]

Exit status: 0 when the path is named at least once, 1 when it is named
nowhere, which is the answer that has to be read as a list of zero rather than
as a search that found nothing.
"""

import argparse
import pathlib
import sys

# Directories that hold generated or machine-local state. A reference from
# inside one is not a consumer of the tree's files, and reading them would bury
# the answer under copies of it.
SKIPPED = {".git", ".local", "target", "artifacts", "__pycache__", ".venv"}


def text_files(root):
    """Every file under `root` that is not binary and not in a skipped tree."""
    for path in sorted(root.rglob("*")):
        if any(part in SKIPPED for part in path.parts):
            continue
        if not path.is_file():
            continue
        try:
            yield path, path.read_text()
        except (UnicodeDecodeError, OSError):
            continue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="the path whose references are wanted")
    parser.add_argument(
        "--root",
        default=".",
        help="the tree to search, default the working directory",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    needle = pathlib.PurePosixPath(args.path).name
    if not needle:
        parser.error("the path names no file")

    hits = []
    for path, text in text_files(root):
        for number, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                hits.append((path.relative_to(root), number, line.strip()))

    files = sorted({str(path) for path, _, _ in hits})
    for path in files:
        print(path)
        for hit_path, number, line in hits:
            if str(hit_path) == path:
                print(f"  {number}: {line}")

    print(f"{len(hits)} line(s) in {len(files)} file(s) name {needle}")
    if not hits:
        print(
            "named nowhere - the answer is this list of zero, not a search that"
            " found nothing"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
