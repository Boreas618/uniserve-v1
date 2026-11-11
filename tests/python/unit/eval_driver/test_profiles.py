from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import uniserve_eval
from uniserve_eval.config import load_config
from uniserve_eval.datasets import load_examples

pytestmark = pytest.mark.unit


# Profiles shipped beside the evaluator; each runs from the repository root.
_SHIPPED = sorted(
    path
    for path in Path(uniserve_eval.__file__).parent.glob("*.toml")
    if path.name != "pyproject.toml"
)


@pytest.mark.parametrize("profile", _SHIPPED, ids=lambda path: path.name)
def test_shipped_profiles_name_deployments_and_workloads_that_exist(
    profile: Path,
) -> None:
    """Every deployment and request manifest a shipped point uses exists."""
    config = load_config(profile)

    for name, server in config.servers.items():
        if "--worker-groups" not in server.command:
            continue
        flag = server.command.index("--worker-groups")
        deployment = config.root / server.command[flag + 1]
        assert deployment.is_file(), f"{name} names {deployment}"
        groups = tomllib.loads(deployment.read_text())["worker_groups"]
        assert groups and all(group["blocks"] for group in groups)

    for name, point in config.benchmarks.items():
        manifests = (
            point.dataset_path,
            point.load.warmup_manifest,
            point.load.priming_manifest,
        )
        for manifest in filter(None, manifests):
            assert (config.root / manifest).is_file(), (
                f"{name} names {manifest}"
            )
        if point.dataset == "jsonl":
            rows = (config.root / point.dataset_path).read_text().splitlines()
            measured = sum(1 for row in rows if row.strip())
            assert measured >= point.load.num_prompts, name


def test_toml_rejects_an_unknown_benchmark_field(tmp_path: Path) -> None:
    config = tmp_path / "profiles.toml"
    config.write_text(
        """
[servers.local]
port = 8000
command = ["server"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
dataset = "sharegpt"
tokenizer = "model"
unexpected = true

[benchmarks.point.load]
num_prompts = 1

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_config(config)


def test_toml_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    config = tmp_path / "profiles.toml"
    config.write_text(
        """
artifct_root = "results"

[servers.local]
port = 8000
command = ["server"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifct_root"):
        load_config(config)


def test_toml_rejects_a_mistyped_load_value_as_a_schema_error(
    tmp_path: Path,
) -> None:
    config = tmp_path / "profiles.toml"
    config.write_text(
        """
[servers.local]
port = 8000
command = ["server"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
dataset = "sharegpt"
tokenizer = "model"

[benchmarks.point.load]
num_prompts = "5"

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"benchmarks\.point\.load"):
        load_config(config)


def test_a_profile_resolves_its_paths_against_the_root_it_states(
    tmp_path: Path,
) -> None:
    """A profile means the same thing wherever the driver is installed.

    Its executable, interpreter and working directory follow the tree the file
    names, so nothing depends on where this package's own source sits.
    """
    tree = tmp_path / "tree"
    (tree / "uniserve_eval").mkdir(parents=True)
    config_path = tree / "uniserve_eval" / "profiles.toml"
    config_path.write_text(
        """
root = ".."

[servers.local]
port = 8000
command = ["target/release/uniserve", "serve", "model"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
dataset = "sharegpt"
tokenizer = "model"

[benchmarks.point.load]
num_prompts = 1

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uniserve_eval.cli",
            "--config",
            str(config_path),
            "plan",
            "point",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    plan = json.loads(result.stdout)[0]

    (command,) = plan["server_commands"]
    assert command[0] == str(tree / "target/release/uniserve")
    assert plan["server_working_directory"] == str(tree)


def _text_profile(tmp_path: Path, sampling: str) -> Path:
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
[servers.server]
command = ["uniserve", "serve", "model"]
port = 8000

[benchmarks.point]
server = "server"
task = "text"
model = "model"
dataset = "jsonl"
dataset_path = "rows.jsonl"

[benchmarks.point.sampling]
{sampling}

[benchmarks.point.metrics]
output_throughput = "higher"
""",
        encoding="utf-8",
    )
    return profile


def test_server_sampling_resolves_to_a_request_without_controls(
    tmp_path: Path,
) -> None:
    config = load_config(
        _text_profile(tmp_path, "server_sampling = true\nmax_tokens = 256")
    )

    sampling = config.benchmarks["point"].workload_dict()["sampling"]

    assert sampling["temperature"] is None
    assert sampling["top_p"] is None
    assert sampling["ignore_eos"] is None
    assert sampling["max_tokens"] == 256
    assert "server_sampling" not in sampling


def test_server_sampling_keeps_a_request_seed(tmp_path: Path) -> None:
    config = load_config(
        _text_profile(tmp_path, "server_sampling = true\nsampling_seed = 42")
    )

    sampling = config.benchmarks["point"].workload_dict()["sampling"]

    assert sampling["sampling_seed"] == 42
    assert sampling["temperature"] is None


def test_server_sampling_rejects_a_token_control(tmp_path: Path) -> None:
    profile = _text_profile(
        tmp_path, "server_sampling = true\ntemperature = 0.0"
    )

    with pytest.raises(
        ValueError, match="token-sampling controls: temperature"
    ):
        load_config(profile)


def test_relative_paths_resolve_against_the_profile_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile's relative paths name files under its root, not the cwd."""
    tree = tmp_path / "tree"
    (tree / "profiles").mkdir(parents=True)
    (tree / "workloads").mkdir()
    (tree / "workloads" / "rows.jsonl").write_text(
        '{"id": "measured", "prompt": "hello"}\n', encoding="utf-8"
    )
    (tree / "workloads" / "warmup.jsonl").write_text(
        '{"id": "warmup", "prompt": "warm"}\n', encoding="utf-8"
    )
    profile = tree / "profiles" / "profile.toml"
    profile.write_text(
        """
root = ".."

[servers.local]
port = 8000
command = ["server"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
dataset = "jsonl"
dataset_path = "workloads/rows.jsonl"

[benchmarks.point.load]
num_prompts = 1
warmup_manifest = "workloads/warmup.jsonl"

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    # The driver runs from a directory that holds none of the files.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    point = load_config(profile).benchmarks["point"]

    rows, _ = load_examples(point)
    assert [row.id for row in rows] == ["measured"]
    assert point.load.warmup_manifest == str(tree / "workloads/warmup.jsonl")


def test_an_unset_path_and_an_unresolved_reference_stay_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a stated, resolved relative path is joined to the root."""
    monkeypatch.delenv("UNISERVE_TEST_UNSET_DATASET", raising=False)
    profile = tmp_path / "profile.toml"
    profile.write_text(
        """
[servers.local]
port = 8000
command = ["server"]

[benchmarks.unset]
server = "local"
task = "text"
model = "model"
dataset = "sharegpt"
tokenizer = "org/tokenizer"
dataset_path = "${UNISERVE_TEST_UNSET_DATASET:-}"

[benchmarks.unset.metrics]
output_throughput = "higher"

[benchmarks.unresolved]
server = "local"
task = "text"
model = "model"
dataset = "jsonl"
dataset_path = "${UNISERVE_TEST_UNSET_DATASET}/rows.jsonl"

[benchmarks.unresolved.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(profile)

    assert config.benchmarks["unset"].dataset_path == ""
    # A tokenizer the root does not hold is a Hub repository ID.
    assert config.benchmarks["unset"].tokenizer == "org/tokenizer"
    assert config.benchmarks["unresolved"].dataset_path == (
        "${UNISERVE_TEST_UNSET_DATASET}/rows.jsonl"
    )
