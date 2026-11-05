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
        --results /workspace/uniserve-worker-artifacts/h3-check-baseline-3

The manifest states both columns for every cell. A baseline run fills the
baseline column; a candidate run fills the candidate column when it is given
`--candidate-results`, and the reviewer sees the two MP4s side by side.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def condition_media(row: dict[str, object]) -> list[str]:
    """The media a row conditions on, as `type: path (role)`.

    A workload row states its inputs under `conditions`, one entry per input,
    each naming the media it reads and the role it plays in the request.
    """
    conditions = row.get("conditions") or []
    if not isinstance(conditions, list):
        return []
    found: list[str] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        media = condition.get("media")
        if not media:
            continue
        kind = condition.get("type", "input")
        role = condition.get("role")
        found.append(f"{kind}: `{media}`" + (f" ({role})" if role else ""))
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


def manifest(config: Path, baseline: Path, candidate: Path | None) -> str:
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
        f"Baseline run: `{baseline}`",
    ]
    if candidate is not None:
        lines.append(f"Candidate run: `{candidate}`")
    else:
        lines.append(
            "Candidate run: not measured — the candidate column is empty."
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
        checkpoint = (
            server.get("checkpoint") or server.get("model") or cell["server"]
        )
        row = measured_row(config.parent.parent / cell["dataset_path"])
        task = cell.get("video", {}).get("task", "")
        seed = row.get("seed", "")
        base_mp4s = cell_mp4s(baseline / name)
        cand_mp4s = cell_mp4s(candidate / name) if candidate is not None else []
        rows.append((name, task, checkpoint, row, base_mp4s, cand_mp4s))
        lines.append(f"| `{name}` | {task} | `{checkpoint}` | {seed} |")

    lines.append("")
    for name, task, checkpoint, row, base_mp4s, cand_mp4s in rows:
        inputs = condition_media(row)
        lines += [
            f"## `{name}`",
            "",
            f"- **Task**: {task}",
            f"- **Checkpoint**: `{checkpoint}`",
            f"- **Seed**: {row.get('seed', '')}",
            f"- **Request**: `{row.get('id', '')}`",
            f"- **Input files**: {', '.join(inputs) if inputs else 'none'}",
            f"- **Baseline MP4**: {mp4_name(base_mp4s)}",
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
        "--results", type=Path, required=True, help="the baseline run directory"
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

    output = args.output or args.results / "review-manifest.md"
    output.write_text(
        manifest(args.config, args.results, args.candidate_results)
    )
    print(output)


if __name__ == "__main__":
    main()
