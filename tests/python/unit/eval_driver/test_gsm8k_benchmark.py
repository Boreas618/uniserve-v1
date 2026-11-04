"""Tests GSM8K's dataset, scoring protocol and confirmation profile.

Expected values come from the protocol the swarm map fixes: the complete
1319-row test split in file order, the strict ``#### `` extracted-answer
rule, accuracy over every measured row, and the three cells' settings.
"""

from __future__ import annotations

import json
from argparse import Namespace
from decimal import Decimal
from pathlib import Path

import huggingface_hub
import pytest

import uniserve_eval
from uniserve_eval import cli
from uniserve_eval.config import load_config
from uniserve_eval.datasets import load_examples
from uniserve_eval.datasets.gsm8k import (
    GSM8K_FILENAME,
    GSM8K_REPO_ID,
    GSM8KDataset,
)
from uniserve_eval.tasks.gsm8k import (
    SYSTEM_PROMPT,
    GSM8KTask,
    extract_answer,
    score_records,
)
from uniserve_eval.types import (
    BenchmarkPoint,
    Example,
    LoadConfig,
    MetricDefinition,
    RequestRecord,
    SamplingConfig,
    TaskName,
    selected_rows_identity,
)

pytestmark = pytest.mark.unit

# The profile the confirmation runs, beside the package.
PROFILE = Path(uniserve_eval.__file__).parent / "qwen3_moe.toml"

# The three confirmation cells, in the profile's suite order.
CELLS = (
    "qwen3-moe-gsm8k-bf16",
    "qwen3-moe-gsm8k-nvfp4",
    "qwen3-moe-gsm8k-bf16-fp8-kv",
)

# The split's row count: the denominator of every reported accuracy.
SPLIT_ROWS = 1319


@pytest.fixture(autouse=True)
def confirmation_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the checkpoint variables a confirmation run documents."""
    monkeypatch.setenv(
        "UNISERVE_QWEN3_MOE_MODEL", "/workspace/models/Qwen3-30B-A3B"
    )
    monkeypatch.setenv(
        "UNISERVE_QWEN3_MOE_NVFP4_MODEL",
        "/workspace/models/Qwen3-30B-A3B-NVFP4",
    )
    monkeypatch.delenv("UNISERVE_QWEN3_FP8_ATTENTION_BACKEND", raising=False)


def _point(
    dataset_path: str | None,
    num_prompts: int,
    *,
    sampling: SamplingConfig | None = None,
    dataset_revision: str | None = None,
) -> BenchmarkPoint:
    """Build a GSM8K point over a local split or the Hub default."""
    return BenchmarkPoint(
        name="qwen3-moe-gsm8k",
        server="server",
        task=TaskName.GSM8K,
        model="Qwen3-30B-A3B",
        dataset="gsm8k",
        dataset_path=dataset_path,
        dataset_revision=dataset_revision,
        metrics=(MetricDefinition(("accuracy",), "higher"),),
        load=LoadConfig(num_prompts=num_prompts),
        sampling=sampling or _sampling(),
    )


def _sampling() -> SamplingConfig:
    """Return the confirmation's fixed sampling settings."""
    return SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        sampling_seed=42,
        ignore_eos=False,
        max_tokens=1024,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _rows(count: int) -> list[dict[str, str]]:
    """Return GSM8K-shaped rows whose answers differ per position."""
    return [
        {
            "question": f"question number {index}",
            "answer": f"reasoning {index} #### {index}",
        }
        for index in range(count)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write rows as one JSON object per line, in order."""
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _record(
    index: int,
    text: str,
    *,
    success: bool = True,
    gold: str = "reasoning #### 18",
) -> RequestRecord:
    """Build a measured record carrying its row's gold answer."""
    record = RequestRecord(
        request_id=f"gsm8k-{index:05d}",
        task="gsm8k",
        success=success,
        generated_text=text,
    )
    record.example = {
        "id": f"gsm8k-{index:05d}",
        "prompt": f"question number {index}",
        "metadata": {"answer": gold},
    }
    return record


def test_the_split_is_read_in_file_order_with_its_answers(
    tmp_path: Path,
) -> None:
    """Rows keep the source order and carry their gold answer."""
    rows = _rows(4)
    path = _write_jsonl(tmp_path / "test.jsonl", rows)

    examples = GSM8KDataset(_point(str(path), num_prompts=4)).load()

    assert [example.id for example in examples] == [
        "gsm8k-00000",
        "gsm8k-00001",
        "gsm8k-00002",
        "gsm8k-00003",
    ]
    assert [example.prompt for example in examples] == [
        row["question"] for row in rows
    ]
    assert [example.metadata["answer"] for example in examples] == [
        row["answer"] for row in rows
    ]
    # File order is part of the identity: swapping two rows changes it.
    identity = selected_rows_identity(examples)
    swapped = [examples[1], examples[0], *examples[2:]]
    assert selected_rows_identity(swapped) != identity


def test_the_complete_split_is_1319_rows_in_file_order(
    tmp_path: Path,
) -> None:
    """The measured sample set is the split's first `num_prompts` rows."""
    rows = _rows(SPLIT_ROWS)
    path = _write_jsonl(tmp_path / "test.jsonl", rows)

    examples = GSM8KDataset(_point(str(path), SPLIT_ROWS)).load()

    assert len(examples) == SPLIT_ROWS
    assert (examples[0].id, examples[-1].id) == (
        "gsm8k-00000",
        "gsm8k-01318",
    )
    assert examples[0].prompt == rows[0]["question"]
    assert examples[-1].metadata["answer"] == rows[-1]["answer"]
    identity = selected_rows_identity(examples)
    assert identity["count"] == SPLIT_ROWS

    # A smaller count is the same rows' prefix, not a new sample.
    prefix = GSM8KDataset(_point(str(path), num_prompts=5)).load()
    assert [example.as_dict() for example in prefix] == [
        example.as_dict() for example in examples[:5]
    ]


def test_a_directory_resolves_the_test_split(tmp_path: Path) -> None:
    """A directory `dataset_path` names the split it holds."""
    _write_jsonl(tmp_path / "test.jsonl", _rows(2))

    examples = GSM8KDataset(_point(str(tmp_path), num_prompts=2)).load()

    assert [example.prompt for example in examples] == [
        "question number 0",
        "question number 1",
    ]


def test_an_empty_directory_names_no_split(tmp_path: Path) -> None:
    """A directory without a recognised split file is not guessed."""
    with pytest.raises(FileNotFoundError, match="test.jsonl"):
        GSM8KDataset(_point(str(tmp_path), num_prompts=1)).load()


def test_only_an_unset_path_downloads_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Hub copy is fetched at the point's revision only when needed."""
    import pyarrow
    import pyarrow.parquet

    rows = _rows(2)
    parquet = tmp_path / "test.parquet"
    pyarrow.parquet.write_table(
        pyarrow.table(
            {
                "question": [row["question"] for row in rows],
                "answer": [row["answer"] for row in rows],
            }
        ),
        parquet,
    )
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(parquet)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    point = _point(None, num_prompts=2, dataset_revision="740312add88f")

    examples = GSM8KDataset(point).load()

    assert calls == [
        {
            "repo_id": GSM8K_REPO_ID,
            "filename": GSM8K_FILENAME,
            "repo_type": "dataset",
            "revision": "740312add88f",
        }
    ]
    assert [example.prompt for example in examples] == [
        row["question"] for row in rows
    ]


def test_a_short_source_fails_the_row_count(tmp_path: Path) -> None:
    """A source with fewer rows than the point requires is rejected."""
    path = _write_jsonl(tmp_path / "test.jsonl", _rows(2))

    with pytest.raises(ValueError, match="exactly 3"):
        load_examples(_point(str(path), num_prompts=3))


def test_a_row_without_an_answer_is_rejected(tmp_path: Path) -> None:
    """A row that cannot be scored fails instead of entering the sample."""
    path = _write_jsonl(
        tmp_path / "test.jsonl",
        [{"question": "question", "answer": ""}],
    )

    with pytest.raises(ValueError, match="row 0 has no answer"):
        GSM8KDataset(_point(str(path), num_prompts=1)).load()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is #### 18", Decimal("18")),
        ("#### 1,250", Decimal("1250")),
        ("#### 18,250.5", Decimal("18250.5")),
        ("#### $18", Decimal("18")),
        ("#### (18)", Decimal("18")),
        ("#### **18**", Decimal("18")),
        ("#### 18.", Decimal("18")),
        ("#### -5", Decimal("-5")),
        ("#### -18.5", Decimal("-18.5")),
        ("#### +18", Decimal("18")),
        ("first #### 5 then #### 7", Decimal("7")),
        ("#### 18 and then #### 1,250", Decimal("1250")),
        ("####20", None),
        ("no marker at all", None),
        ("#### ", None),
        ("#### five", None),
        ("#### The answer is 18", None),
        ("#### 18 apples", None),
        ("#### 1/2", None),
        ("#### 1,25", None),
        ("#### 18 because 2+3", None),
    ],
)
def test_the_marker_and_value_rule_decides_parsing(
    text: str, expected: Decimal | None
) -> None:
    """The last ``#### `` marker's value parses; anything else is no answer."""
    assert extract_answer(text) == expected


def test_the_request_is_one_system_and_one_user_message() -> None:
    """Every measured request carries the fixed prompt and sampling fields."""
    task = GSM8KTask(_point(None, num_prompts=1))

    request = task.build_request(
        Example(id="gsm8k-00000", prompt="Natalia sold clips to 48 friends.")
    )

    assert request.stream is True
    assert request.endpoint == "/v1/chat/completions"
    assert request.payload == {
        "model": "Qwen3-30B-A3B",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Natalia sold clips to 48 friends.",
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_the_request_always_streams_and_honours_a_row_limit() -> None:
    """Streaming is fixed and a row may shorten the output limit."""
    sampling = SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        sampling_seed=42,
        ignore_eos=False,
        max_tokens=1024,
        stream=False,
    )
    task = GSM8KTask(_point(None, num_prompts=1, sampling=sampling))

    request = task.build_request(
        Example(id="row", prompt="question", max_tokens=16)
    )

    assert request.stream is True
    assert request.payload["stream"] is True
    assert request.payload["max_completion_tokens"] == 16


def test_accuracy_counts_every_measured_row() -> None:
    """Correct, unparseable and failed rows all stay in the denominator."""
    records = [
        _record(0, "#### 18"),
        _record(1, "#### 19"),
        _record(2, "#### 18", success=False),
        _record(3, "no marker"),
    ]

    score = score_records(records)
    assert (score.correct, score.total, score.unparseable) == (1, 4, 1)
    assert score.accuracy == 0.25

    validation = GSM8KTask(_point(None, num_prompts=4)).validate(records)
    assert validation.checks["all_requests_succeeded"] is False
    assert validation.checks["gold_answers"] is True
    assert validation.statistics["correct"] == 1
    assert validation.statistics["total"] == 4
    assert validation.statistics["unparseable_predictions"] == 1


def test_the_summary_metrics_are_accuracy_correct_and_total() -> None:
    """The run summary receives the three task-owned metrics."""
    records = [_record(0, "#### 18"), _record(1, "#### 19")]

    metrics = GSM8KTask(_point(None, num_prompts=2)).summary_metrics(records)

    assert metrics == {"accuracy": 0.5, "correct": 1, "total": 2}


def test_a_failed_request_is_incorrect_whatever_its_text() -> None:
    """A request that failed produced no canonical answer."""
    score = score_records([_record(0, "#### 18", success=False)])

    assert (score.correct, score.total, score.unparseable) == (0, 1, 0)


def test_answers_compare_numerically() -> None:
    """The comparison is numeric, not textual."""
    records = [
        _record(0, "#### 18.0", gold="reasoning #### 18"),
        _record(1, "#### $1,250", gold="reasoning #### 1250"),
    ]

    assert score_records(records).correct == 2


def test_a_gold_answer_that_does_not_parse_fails_validation() -> None:
    """A row whose reference answer cannot be scored is not a valid run."""
    task = GSM8KTask(_point(None, num_prompts=1))
    record = _record(0, "#### 18", gold="no marker")

    assert task.validate([record]).checks["gold_answers"] is False


def _configured_cell(name: str) -> tuple[BenchmarkPoint, list[str]]:
    """Return one cell's point and its server command."""
    config = load_config(PROFILE)
    point = config.benchmarks[name]
    return point, list(config.servers[point.server].command)


def test_each_cell_states_the_fixed_sample_set_and_sampling() -> None:
    """All three cells run the same rows with the same sampling settings."""
    for name in CELLS:
        point, _ = _configured_cell(name)
        assert point.task is TaskName.GSM8K
        assert point.dataset == "gsm8k"
        assert point.load.num_prompts == SPLIT_ROWS
        assert point.sampling == _sampling()
        assert [metric.as_dict() for metric in point.metrics] == [
            {"path": "accuracy", "direction": "higher"}
        ]

    provenances = {str(_configured_cell(name)[0].provenance) for name in CELLS}
    assert len(provenances) == len(CELLS)


def test_only_the_checkpoint_and_kv_cache_dtype_differ_between_cells() -> None:
    """One change separates the cells, so a comparison isolates it."""

    def normalized(command: list[str]) -> list[str]:
        values = [
            value
            for value in command
            if not value.startswith("attention_backend=")
        ]
        values[values.index("serve") + 1] = "<checkpoint>"
        return [
            "kv_cache_config.dtype=<dtype>"
            if value.startswith("kv_cache_config.dtype=")
            else value
            for value in values
        ]

    commands = [_configured_cell(name)[1] for name in CELLS]
    assert normalized(commands[0]) == normalized(commands[1])
    assert normalized(commands[0]) == normalized(commands[2])

    def model(command: list[str]) -> str:
        """Return the checkpoint path a server command serves."""
        return command[command.index("serve") + 1]

    assert model(commands[0]) == "/workspace/models/Qwen3-30B-A3B"
    assert model(commands[2]) == "/workspace/models/Qwen3-30B-A3B"
    assert model(commands[1]) == "/workspace/models/Qwen3-30B-A3B-NVFP4"
    assert "kv_cache_config.dtype=float8_e4m3fn" in commands[2]
    # Today's `auto` selection refuses per-block FP8 pages, so the FP8 cell
    # names the one provider that reads them; the other cells leave the
    # provider to the implementation.
    assert "attention_backend=torch" in commands[2]
    for command in commands[:2]:
        assert not any(
            value.startswith("attention_backend=") for value in command
        )
    for command in commands:
        # The checkpoint's own representation is used; no precision flag is
        # sent.
        assert not any(
            value.startswith(("dtype=", "quantization=")) for value in command
        )
        # `--executable` must replace the whole deployment, so the profile
        # cannot pin one implementation's worker binary.
        assert "--worker-executable" not in command


def test_a_host_may_point_the_cells_at_a_local_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset path downloads; a set one names the same split locally."""
    for name in CELLS:
        assert _configured_cell(name)[0].dataset_path == ""

    monkeypatch.setenv("UNISERVE_GSM8K_DATASET", "/data/gsm8k/test.parquet")

    for name in CELLS:
        assert (
            _configured_cell(name)[0].dataset_path == "/data/gsm8k/test.parquet"
        )


def test_the_profile_renders_every_point(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`plan` resolves each cell's server command and workload."""
    cli.plan(
        Namespace(config=PROFILE, selection="qwen3-moe-gsm8k", executable=None)
    )
    rendered = capsys.readouterr().out

    points = json.loads(rendered)
    assert [point["benchmark"] for point in points] == list(CELLS)
    for point in points:
        assert point["task"] == "gsm8k"
        assert point["num_prompts"] == SPLIT_ROWS
        (command,) = point["server_commands"]
        assert Path(command[0]).is_absolute()
        assert Path(point["server_working_directory"]).is_absolute()
        assert "${" not in rendered
