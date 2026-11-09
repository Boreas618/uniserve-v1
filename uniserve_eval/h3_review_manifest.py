"""Write the review manifest for the H3 check set.

The design confirms H3 quality by a person watching and listening to the
results, with no metric, so every MP4 the check set produces is evidence only
once it is reviewed. This tool writes the page that makes review possible: per
cell, the task, the checkpoint, the prompt, the seed, the input files, and the
baseline MP4 the candidate output is to be compared with.

It is deliberately independent of the evaluation harness: it reads the profile,
the workload the profile names, and the run directory, and writes one Markdown
file beside the MP4s. Run it after a run finishes; it never touches a server.

    python -m uniserve_eval.h3_review_manifest \
        --config uniserve_eval/check_h3.toml \
        --candidate-results artifacts/minimax_h3/check/results

Each column is filled by the root it is given: `--results` for the earlier
run, `--candidate-results` for the run under review, and a root that is not
there is an error rather than an empty column. With both, the reviewer reads
the two MP4s side by side; with one, the page states that cell's run alone.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def reference_media(row: dict[str, object]) -> list[str]:
    """The media a row references, as `type: path`, with any keyframe anchor.

    A workload row states its inputs under `input_references`, one entry per
    input, each naming the media it reads; an anchor keyframe also names the
    frame it pins (`-1` is the last frame).
    """
    references = row.get("input_references") or []
    if not isinstance(references, list):
        return []
    found: list[str] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        media = reference.get("media")
        if not media:
            continue
        kind = reference.get("type", "input")
        anchor = reference.get("frame_index")
        found.append(
            f"{kind}: `{media}`"
            + (f" (keyframe at frame {anchor})" if anchor is not None else "")
        )
    return found


VIDEO_SUFFIXES = (".mp4", ".webm", ".mov")


def media_in(value: object) -> list[str]:
    """The media paths a workload field names, whichever shape it takes."""
    if isinstance(value, str):
        return (
            [value]
            if value.lower().endswith(VIDEO_SUFFIXES) or "/" in value
            else []
        )
    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            found.extend(media_in(item))
        return found
    return []


def measured_row(path: Path) -> dict[str, object]:
    """The row a cell measures: the workload's first row, which carries seed 0.

    The check workloads list the measured request first and its warmup second,
    so the first row is the request the cell's result is about.
    """
    with path.open() as workload:
        for line in workload:
            if line.strip():
                return json.loads(line)
    raise SystemExit(f"{path}: the workload is empty")


def cell_mp4s(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    )


def mp4_name(mp4s: list[Path], missing: str = "missing") -> str:
    """The MP4 a reviewer is meant to open, named as the run wrote it."""
    return f"`{mp4s[0].name}`" if mp4s else missing


def manifest(
    config: Path, baseline: Path | None, candidate: Path | None
) -> str:
    """The review page for whichever runs exist.

    The design confirms H3 by watching the candidate's six videos, so a
    candidate alone is a reviewable run; the baseline column is a convenience
    when its bundles are still on the host. At least one side must be given.
    """
    if baseline is None and candidate is None:
        raise ValueError(
            "a review page needs a run: --results or --candidate-results"
        )
    profile = tomllib.loads(config.read_text())
    points = profile["suites"]["check-h3"]["points"]
    servers = profile["servers"]
    benchmarks = profile["benchmarks"]

    lines = [
        "# H3 check set — review manifest",
        "",
        "The design confirms H3 quality by a person watching and listening to",
        "these results, with no metric. The automatic checks only establish",
        "that each MP4 is a valid video; they cannot say whether it shows what",
        "the prompt asks or whether it is as good as today's output. Both",
        "questions are answered per cell, by the two MP4s named there.",
        "",
        f"Baseline run: `{baseline}`"
        if baseline is not None
        else "Baseline run: not on this host — the baseline column is empty.",
    ]
    lines.append(
        f"Candidate run: `{candidate}`"
        if candidate is not None
        else "Candidate run: not measured — the candidate column is empty."
    )
    lines += [
        "",
        "| cell | task | checkpoint | seed |",
        "| --- | --- | --- | --- |",
    ]

    rows: list[
        tuple[str, str, str, dict[str, object], list[Path], list[Path]]
    ] = []
    for name in points:
        cell = benchmarks[name]
        server = servers[cell["server"]]
        # The cell's own provenance names the checkpoint it confirms; the
        # server states one only when it is a fixed local path. A server name
        # alone would tell the reviewer which deployment ran, not which
        # checkpoint the cell confirms.
        checkpoint = (
            (cell.get("provenance") or {}).get("checkpoint")
            or server.get("checkpoint")
            or server.get("model")
            or cell["server"]
        )
        row = measured_row(config.parent.parent / cell["dataset_path"])
        task = (cell.get("provenance") or {}).get("task", "")
        seed = row.get("seed", "")
        base_mp4s = cell_mp4s(baseline / name) if baseline is not None else []
        cand_mp4s = cell_mp4s(candidate / name) if candidate is not None else []
        rows.append((name, task, checkpoint, row, base_mp4s, cand_mp4s))
        lines.append(f"| `{name}` | {task} | `{checkpoint}` | {seed} |")

    lines.append("")
    for name, task, checkpoint, row, base_mp4s, cand_mp4s in rows:
        inputs = reference_media(row)
        base_text = mp4_name(
            base_mp4s, "not on this host" if baseline is None else "missing"
        )
        lines += [
            f"## `{name}`",
            "",
            f"- **Task**: {task}",
            f"- **Checkpoint**: `{checkpoint}`",
            f"- **Seed**: {row.get('seed', '')}",
            f"- **Request**: `{row.get('id', '')}`",
            f"- **Input files**: {', '.join(inputs) if inputs else 'none'}",
            f"- **Baseline MP4**: {base_text}",
            f"- **Candidate MP4**: {mp4_name(cand_mp4s, 'not measured')}",
            "",
            "Prompt:",
            "",
            "> " + " ".join(str(row.get("prompt", "")).split()),
            "",
            "Does it show what the prompt asks? Is it as good as baseline?",
            "",
        ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="the check set profile"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="the earlier run's directory, when its bundles exist",
    )
    parser.add_argument(
        "--candidate-results",
        type=Path,
        default=None,
        help="the candidate run directory, when one has been measured",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="the manifest path; defaults to <results>/review-manifest.md",
    )
    args = parser.parse_args()
    if args.results is None and args.candidate_results is None:
        parser.error("give --results, --candidate-results, or both")

    root = args.results or args.candidate_results
    output = args.output or root / "review-manifest.md"
    output.write_text(
        manifest(args.config, args.results, args.candidate_results)
    )
    print(output)


if __name__ == "__main__":
    main()
