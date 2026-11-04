"""Loads GSM8K test questions with their reference answers, in file order.

GSM8K's complete test split is the fixed sample set that confirms
Qwen3-MoE quality (``specs/worker.md`` §10): 1319 rows read in file order,
without shuffling or filtering, so two runs of the same revision resolve the
same rows and their ``selected_rows`` identity is comparable. Each row
becomes one example whose prompt is the question verbatim and whose
``metadata['answer']`` is the reference answer; the task extracts the gold
value from it.

The source is ``openai/gsm8k``'s test Parquet file. A ``dataset_path`` names
a local JSON Lines or Parquet file, or a directory holding the split, and
only an unset path downloads the file from the Hub.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

GSM8K_REPO_ID = "openai/gsm8k"
# The test split's only file in the published repository.
GSM8K_FILENAME = "main/test-00000-of-00001.parquet"
# Names a directory given as `dataset_path` may hold the test split under, in
# the order they are tried: a JSON Lines conversion first, then the split's
# own Parquet file name with and without its `main/` directory.
_DIRECTORY_NAMES = (
    "test.jsonl",
    "main/test-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
    "test.parquet",
)


class GSM8KDataset(Dataset):
    """Adapts GSM8K questions and reference answers to benchmark examples.

    Rows keep the source's file order and are neither shuffled nor filtered.
    At most ``point.load.num_prompts`` rows are returned, so the complete
    1319-row split is the measured sample set and a smaller count takes the
    split's first rows; ``datasets.load_examples`` rejects a source that
    holds fewer.

    A row's example id is ``gsm8k-<position>`` with a zero-padded position,
    and its ``metadata`` holds the reference answer under ``answer``.
    """

    name: ClassVar[str] = "gsm8k"

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Return the split's rows in file order up to ``num_prompts``.

        Args:
            tokenizer: Unused; the questions and answers are text.

        Returns:
            Examples in file order, at most ``num_prompts`` of them.

        Raises:
            FileNotFoundError: If a named ``dataset_path`` or the split file
                in a named directory does not exist.
            ValueError: If ``dataset_path`` names a file that is neither
                Parquet nor JSON Lines, a line is not a JSON object, a row
                lacks a non-empty string ``question`` or ``answer``, or the
                downloaded file is not Parquet.
        """
        point = self.point
        path = _split_path(point.dataset_path, point.dataset_revision)
        rows = _read_rows(path)

        examples: list[Example] = []
        for position, row in enumerate(rows):
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(question, str) or not question:
                raise ValueError(
                    f"gsm8k row {position} has no question: {path}"
                )
            if not isinstance(answer, str) or not answer:
                raise ValueError(f"gsm8k row {position} has no answer: {path}")
            examples.append(
                Example(
                    id=f"gsm8k-{len(examples):05d}",
                    prompt=question,
                    metadata={"answer": answer},
                )
            )
            if len(examples) == point.load.num_prompts:
                break
        return examples


def _split_path(dataset_path: str | None, revision: str | None) -> Path:
    """Resolve the test split file, downloading it only when unset.

    A ``dataset_path`` that names a directory resolves to the first name of
    ``_DIRECTORY_NAMES`` it holds.

    Raises:
        FileNotFoundError: If a directory holds none of those names.
    """
    if dataset_path:
        path = Path(dataset_path)
        if not path.is_dir():
            return path
        for name in _DIRECTORY_NAMES:
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"{path} holds none of {', '.join(_DIRECTORY_NAMES)}"
        )

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=GSM8K_REPO_ID,
            filename=GSM8K_FILENAME,
            repo_type="dataset",
            revision=revision,
        )
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read object rows from a Parquet or JSON Lines file, in file order.

    ``pyarrow`` and ``datasets`` belong to the optional ``datasets`` extra
    of uniserve-eval and are imported only when a Parquet file is read.

    Raises:
        ValueError: If the suffix names no supported format or a JSON Lines
            line is not an object.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        parquet = importlib.import_module("pyarrow.parquet")
        return [dict(row) for row in parquet.read_table(path).to_pylist()]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"gsm8k JSON Lines row {line_no} must be an object: "
                        f"{path}"
                    )
                rows.append(value)
        return rows
    raise ValueError(
        f"gsm8k dataset_path must name a .parquet or .jsonl file, not {path}"
    )
