"""Summarize profiling sessions' files or convert them to a Perfetto trace."""

import argparse
import json
import sys

from .perfetto import write
from .reader import ProfileError, ProfileFile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summary", help="What each file holds")
    summary.add_argument("files", nargs="+")
    perfetto = commands.add_parser(
        "perfetto", help="One trace of a session's files"
    )
    perfetto.add_argument("files", nargs="+")
    perfetto.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "summary":
            summaries = []
            for path in args.files:
                with ProfileFile(path) as profile:
                    summaries.append(profile.summary())
            print(json.dumps(summaries, indent=2))
        else:
            events = write(args.files, args.output)
            print(f"wrote {events} events to {args.output}")
    except ProfileError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
