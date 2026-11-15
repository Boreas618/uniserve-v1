"""Loads and validates evaluator servers, benchmark points, and suites.

An evaluator profile is a TOML file with optional `root` (default `.`) and
`artifact_root` (where `run` writes without `--output-root`) keys and up to
three tables: `[servers.<name>]` describes how to launch a server,
`[benchmarks.<name>]` describes one measured workload against a named
server, with the changes a timeline load applies to it as
`[[benchmarks.<name>.changes]]`, and `[suites.<name>]`
lists benchmark names in execution order. `load_config` turns the file into
an `EvaluationConfig`, rejecting unknown keys at the top level and in every
server, benchmark, and suite table. The adapters that
`tasks.get_task` and `datasets.get_dataset` return validate the parts of a
benchmark that depend on its task and dataset.

String values in server and benchmark tables may reference the process
environment as `${NAME}` or `${NAME:-default}`; see `expand_environment`.

Every path a profile states is relative to its `root` unless absolute: the
server executable (`server_launch`), the artifact
root, and a benchmark's `dataset_path`, `load.warmup_manifest`,
`load.priming_manifest`, `video.media_dir` and `video.reference_root`
(`_resolve_paths`). The loaded `BenchmarkPoint` carries them as absolute
paths, so the driver reads and writes the same files from any working
directory.
"""

from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from .datasets import get_dataset
from .tasks import get_task
from .tasks.base import BenchmarkTask
from .types import (
    BenchmarkPoint,
    Change,
    ImageConfig,
    LoadConfig,
    MetricDefinition,
    MetricDirection,
    SamplingConfig,
    TaskName,
    VideoConfig,
)

#: The profiles this driver ships, which `--config` replaces.
DEFAULT_CONFIG = Path(__file__).resolve().parent / "profiles.toml"

# `${NAME}` and `${NAME:-default}`. The default is matched non-greedily, so it
# ends at the first `}` and cannot itself contain one.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_DEFAULT_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-(.*?)\}")

# Keys accepted at the top level of a profile.
_PROFILE_FIELDS = {"root", "artifact_root", "servers", "benchmarks", "suites"}

# Keys accepted in a `[benchmarks.<name>]` table. The nested load, sampling,
# image, and video tables accept exactly the fields of their dataclasses.
_ROOT_FIELDS = {
    "server",
    "task",
    "model",
    "dataset",
    "dataset_revision",
    "dataset_path",
    "tokenizer",
    "endpoint",
    "question",
    "load",
    "sampling",
    "image",
    "video",
    "metrics",
    "provenance",
    "changes",
}
_LOAD_FIELDS = set(LoadConfig.__dataclass_fields__)
_CHANGE_FIELDS = set(Change.__dataclass_fields__)
_SAMPLING_FIELDS = set(SamplingConfig.__dataclass_fields__)
_IMAGE_FIELDS = set(ImageConfig.__dataclass_fields__)
_VIDEO_FIELDS = set(VideoConfig.__dataclass_fields__)

# Request-level sampling controls a point with server-owned sampling may not
# set: every token-sampling parameter and ignore_eos. A request seed only
# selects the random stream the server's schedule draws from, so it stays
# available to servers that accept one.
_TOKEN_SAMPLING_CONTROLS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "ignore_eos",
}

# The settings dataclasses a benchmark table's nested tables construct.
_Settings = TypeVar(
    "_Settings", LoadConfig, SamplingConfig, ImageConfig, VideoConfig, Change
)


@dataclass(frozen=True)
class ServerProfile:
    """Describes one benchmark deployment and its launch environment.

    A deployment is either one server process, described by `command`,
    `host`, `port` and `environment`, or independent `replicas`, each a
    complete single-process profile. Replicas are separate engines on
    disjoint devices; the client routes each request to one of them.

    A server is ready once its listener accepts a connection and, when
    `ready_path` names one, that path answers HTTP 200: a server that listens
    before it finishes starting reports its readiness there.

    `admin_socket` is the Unix socket of a single-process server's admin
    listener, which a timeline load's changes call and whose event log it
    reads. A relative path is taken in the deployment's runtime directory,
    the `XDG_RUNTIME_DIR` the evaluator creates for it
    (`server.ManagedDeployment`); `uniserve serve` places its admin socket
    in an instance directory of its own unless `admin_config.admin_uds`
    names one, so a profile that names `admin_socket` states that socket to
    the server as `admin_config.admin_uds`.
    """

    name: str
    command: tuple[str, ...]
    host: str
    port: int
    environment: dict[str, str]
    replicas: tuple[ServerProfile, ...] = ()
    ready_path: str | None = None
    admin_socket: str | None = None

    @property
    def instances(self) -> tuple[ServerProfile, ...]:
        """Return the single-process profiles that form this deployment."""
        return self.replicas or (self,)

    @property
    def base_url(self) -> str:
        """Return the HTTP origin of a single-process profile."""
        if self.replicas:
            raise ValueError(f"server {self.name} has replica origins")
        return f"http://{self.host}:{self.port}"

    @property
    def base_urls(self) -> tuple[str, ...]:
        """Return every HTTP origin requests may be routed to."""
        return tuple(instance.base_url for instance in self.instances)


@dataclass(frozen=True)
class ServerLaunch:
    """Contains a resolved server command and its process context."""

    command: tuple[str, ...]
    working_directory: Path
    environment: dict[str, str]


@dataclass(frozen=True)
class SuiteProfile:
    """Names an ordered collection of benchmark points."""

    name: str
    points: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    """Contains the complete validated evaluator configuration."""

    #: Directory the profile's relative paths resolve against: the file's
    #: `root` key (default `.`), itself relative to the file.
    root: Path
    #: Default parent of per-point result directories; `artifact_root` in the
    #: file, relative to `root` unless absolute. `run --output-root` overrides
    #: it, and a relative override is taken from the current directory.
    artifact_root: Path | None
    servers: dict[str, ServerProfile]
    benchmarks: dict[str, BenchmarkPoint]
    suites: dict[str, SuiteProfile]

    def selected_points(self, selection: str) -> tuple[BenchmarkPoint, ...]:
        """Resolve a benchmark or suite name to its ordered points.

        A benchmark name takes precedence over a suite of the same name.

        Raises:
            ValueError: If `selection` names neither a benchmark nor a suite.
        """
        if selection in self.benchmarks:
            return (self.benchmarks[selection],)
        suite = self.suites.get(selection)
        if suite is None:
            known = ", ".join(sorted({*self.benchmarks, *self.suites}))
            raise ValueError(
                f"unknown benchmark or suite {selection!r}; known: {known}"
            )
        return tuple(self.benchmarks[name] for name in suite.points)


def load_config(path: Path = DEFAULT_CONFIG) -> EvaluationConfig:
    """Load and validate an evaluator TOML file.

    A profile's relative paths -- its executable, its interpreter, its
    deployment configuration, its artifact root -- are relative to the tree the
    file states as its `root`, itself relative to the file. Resolving them
    against the file keeps a profile's meaning independent of where the
    driver is installed.

    Environment references are expanded here, but an unset reference without
    a default stays in the value verbatim. `pipeline.setup.prepare_launch`
    rejects a server command or benchmark workload that still contains one
    before a run starts the server; server `environment` values are not
    checked.

    Validation stops at the first violation. Schema violations raise
    `ValueError` naming the offending table, including a load, sampling,
    image, or video value whose type or range its settings dataclass
    rejects; an unregistered task or dataset name raises `KeyError`.
    """
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    # A misspelled top-level key would otherwise leave its setting at the
    # default without notice.
    _reject_unknown(raw, _PROFILE_FIELDS, str(config_path))
    root = (config_path.parent / str(raw.get("root", "."))).resolve()

    # Benchmarks refer to servers and suites refer to benchmarks, so each
    # table is validated against the ones parsed before it.
    servers = {
        name: _server_profile(name, value)
        for name, value in _table(raw, "servers").items()
    }
    benchmarks = {
        name: _benchmark_point(name, value, servers, root)
        for name, value in _table(raw, "benchmarks").items()
    }
    _require_own_media(benchmarks)
    suites = {
        name: _suite_profile(name, value, benchmarks)
        for name, value in _table(raw, "suites").items()
    }
    artifact_root = None
    if "artifact_root" in raw:
        artifact_root = Path(str(raw["artifact_root"]))
        if not artifact_root.is_absolute():
            artifact_root = root / artifact_root
    return EvaluationConfig(
        root=root,
        artifact_root=artifact_root,
        servers=servers,
        benchmarks=benchmarks,
        suites=suites,
    )


def expand_environment(value: Any) -> Any:
    """Expand declared environment references recursively when values exist.

    `${NAME:-default}` becomes the value of `NAME` when it is set, including
    when it is set to an empty string, and `default` otherwise. `${NAME}`
    becomes the value of `NAME` when it is set and is otherwise left in place
    for `unresolved_environment` to report. Strings inside lists and dict
    values are expanded; dict keys are converted to strings but not expanded.
    Other values are returned unchanged.
    """
    if isinstance(value, str):
        expanded = _ENV_DEFAULT_REF.sub(
            lambda match: os.environ.get(match.group(1), match.group(2)), value
        )
        return _ENV_REF.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)),
            expanded,
        )
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): expand_environment(item) for key, item in value.items()
        }
    return value


def unresolved_environment(value: Any) -> tuple[str, ...]:
    """Return sorted environment names that remain referenced in a value."""
    names: set[str] = set()
    if isinstance(value, str):
        names.update(_ENV_REF.findall(value))
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.update(unresolved_environment(item))
    elif isinstance(value, dict):
        for item in value.values():
            names.update(unresolved_environment(item))
    return tuple(sorted(names))


def require_resolved(value: Any, *, context: str) -> None:
    """Reject a value that still contains environment references.

    Raises:
        ValueError: If `unresolved_environment` finds any name; the message
            starts with `context` and lists the names.
    """
    names = unresolved_environment(value)
    if names:
        raise ValueError(
            f"{context} has unresolved environment variables: "
            f"{', '.join(names)}"
        )


def server_launch(
    server: ServerProfile, executable: Path | None, root: Path
) -> ServerLaunch:
    """Resolve a server profile into an executable launch description.

    Args:
        server: Profile whose command is resolved.
        executable: Replacement for the profile's first command element
            (the server binary), or `None` to keep it. Either one resolves
            against `root` when relative.
        root: `EvaluationConfig.root`, the tree the profile's relative paths
            are written against.

    Returns:
        A launch whose executable is an absolute path, whose working
        directory is `root`, and whose environment is a copy of the
        profile's. Environment references are not checked here.
    """
    command = list(server.command)
    selected_executable = (
        executable if executable is not None else Path(command[0])
    )
    if not selected_executable.is_absolute():
        selected_executable = root / selected_executable
    command[0] = str(selected_executable.absolute())

    # A server runs in the tree its profile is written against, so a relative
    # path it was given -- a deployment configuration, a dataset -- resolves
    # the same way there as it reads in the file.
    return ServerLaunch(tuple(command), root, dict(server.environment))


def _server_profile(name: str, raw: Any) -> ServerProfile:
    """Validate and construct one server profile table.

    A table with `replicas` declares an array of single-process tables. Each
    replica inherits the table's `host` and `ready_path` unless it names its
    own, merges the table's `environment` under its own, and must not repeat
    a host and port.
    """
    value = expand_environment(_mapping(raw, f"servers.{name}"))
    _reject_unknown(
        value,
        {
            "command",
            "host",
            "port",
            "environment",
            "ready_path",
            "replicas",
            "admin_socket",
        },
        f"servers.{name}",
    )
    if "replicas" not in value:
        return _server_process(name, value)

    replicas_value = value["replicas"]
    if "command" in value or "port" in value or "admin_socket" in value:
        raise ValueError(
            f"servers.{name} declares replicas and cannot also declare "
            "a command, port or admin_socket"
        )
    if not isinstance(replicas_value, list) or len(replicas_value) < 2:
        raise ValueError(f"servers.{name}.replicas must list two or more")
    host = _server_host(value, f"servers.{name}")
    shared = _server_environment(value, f"servers.{name}")
    ready_path = _server_ready_path(value, f"servers.{name}")
    replicas = []
    for index, item in enumerate(replicas_value):
        context = f"servers.{name}.replicas[{index}]"
        item = _mapping(item, context)
        _reject_unknown(
            item,
            {"command", "host", "port", "environment", "ready_path"},
            context,
        )
        environment = {**shared, **_server_environment(item, context)}
        replicas.append(
            _server_process(
                f"{name}.replica-{index}",
                {
                    "host": host,
                    "ready_path": ready_path,
                    **item,
                    "environment": environment,
                },
                context,
            )
        )
    listeners = {(replica.host, replica.port) for replica in replicas}
    if len(listeners) != len(replicas):
        raise ValueError(f"servers.{name}.replicas must use distinct ports")
    first = replicas[0]
    return ServerProfile(
        name,
        first.command,
        host,
        first.port,
        shared,
        tuple(replicas),
        ready_path,
    )


def _server_process(
    name: str, value: dict[str, Any], context: str | None = None
) -> ServerProfile:
    """Validate the fields of one single-process server description."""
    context = context or f"servers.{name}"
    command = value.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise ValueError(f"{context}.command must be a non-empty string array")
    port = value.get("port")
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 < port < 65536
    ):
        raise ValueError(f"{context}.port must be a valid TCP port")
    admin_socket = value.get("admin_socket")
    if admin_socket is not None and (
        not isinstance(admin_socket, str) or not admin_socket
    ):
        raise ValueError(f"{context}.admin_socket must be a non-empty string")
    return ServerProfile(
        name,
        tuple(command),
        _server_host(value, context),
        port,
        _server_environment(value, context),
        ready_path=_server_ready_path(value, context),
        admin_socket=admin_socket,
    )


def _server_ready_path(value: dict[str, Any], context: str) -> str | None:
    """Return a table's validated readiness path, if it names one."""
    path = value.get("ready_path")
    if path is not None and (
        not isinstance(path, str) or not path.startswith("/")
    ):
        raise ValueError(f"{context}.ready_path must be an absolute URL path")
    return path


def _server_host(value: dict[str, Any], context: str) -> str:
    """Return a table's validated listener host."""
    host = value.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise ValueError(f"{context}.host must be a non-empty string")
    return host


def _server_environment(value: dict[str, Any], context: str) -> dict[str, str]:
    """Return a table's validated launch environment."""
    environment = value.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in environment.items()
    ):
        raise ValueError(f"{context}.environment must contain string values")
    # Each deployment runs under a runtime directory the evaluator creates
    # and removes (`server.ManagedDeployment`), so its host services end
    # with it.
    if "XDG_RUNTIME_DIR" in environment:
        raise ValueError(
            f"{context}.environment may not set XDG_RUNTIME_DIR: each "
            "deployment runs under a runtime directory of its own"
        )
    return dict(environment)


def _benchmark_point(
    name: str,
    raw: Any,
    servers: dict[str, ServerProfile],
    root: Path,
) -> BenchmarkPoint:
    """Validate and construct one task-aware benchmark point.

    `root` is the profile's `EvaluationConfig.root`, which the point's
    relative paths are resolved against (`_resolve_paths`).
    """
    # Resolve the benchmark's named server, task, dataset adapter, and model
    # before interpreting task-specific configuration tables.
    value = expand_environment(_mapping(raw, f"benchmarks.{name}"))
    _reject_unknown(value, _ROOT_FIELDS, f"benchmarks.{name}")
    value = _resolve_paths(value, root, f"benchmarks.{name}")
    server = value.get("server")
    if not isinstance(server, str) or server not in servers:
        raise ValueError(
            f"benchmarks.{name}.server must name a declared server"
        )
    task_name = value.get("task")
    if not isinstance(task_name, str):
        raise ValueError(f"benchmarks.{name}.task must be a string")
    task = get_task(task_name)
    dataset_name = value.get("dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(
            f"benchmarks.{name}.dataset must be a non-empty string"
        )
    dataset_cls = get_dataset(dataset_name)
    model = value.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"benchmarks.{name}.model must be a non-empty string")

    # Each nested parser owns its schema, while the task and dataset adapters
    # enforce modality and source requirements that cross table boundaries.
    metrics = _metrics(value.get("metrics"), f"benchmarks.{name}")
    load = _load_config(value.get("load"), f"benchmarks.{name}.load")
    changes = _changes(
        value.get("changes"), servers[server], load, f"benchmarks.{name}"
    )
    sampling = _sampling_config(
        value.get("sampling"), task, f"benchmarks.{name}.sampling"
    )
    image = _image_config(value.get("image"), task, f"benchmarks.{name}.image")
    video = _video_config(value.get("video"), f"benchmarks.{name}.video")
    endpoint = task.check_endpoint(value.get("endpoint"), f"benchmarks.{name}")
    question = task.check_question(value.get("question"), f"benchmarks.{name}")
    tokenizer = value.get("tokenizer")
    dataset_path = value.get("dataset_path")
    dataset_cls.check_point(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        context=f"benchmarks.{name}",
    )
    revision = value.get("dataset_revision")
    if revision is not None and not isinstance(revision, str):
        raise ValueError(f"benchmarks.{name}.dataset_revision must be a string")

    # Construct only after the server, task and dataset adapter it names
    # have accepted the point.
    task.check_image(image, f"benchmarks.{name}")
    task.check_video(video, f"benchmarks.{name}.video")
    return BenchmarkPoint(
        name=name,
        server=server,
        task=TaskName(task_name),
        model=model,
        dataset=dataset_name,
        metrics=metrics,
        load=load,
        sampling=sampling,
        image=image,
        video=video,
        dataset_revision=revision,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        endpoint=endpoint,
        question=question,
        provenance=_mapping(
            value.get("provenance", {}), f"benchmarks.{name}.provenance"
        ),
        changes=changes,
    )


# The path-valued fields of a benchmark table, by the nested table that holds
# them (`None` for the benchmark table itself).
_PATH_FIELDS: dict[str | None, tuple[str, ...]] = {
    None: ("dataset_path",),
    "load": ("warmup_manifest", "priming_manifest"),
    "video": ("media_dir", "reference_root"),
}


def _resolve_paths(
    value: dict[str, Any], root: Path, context: str
) -> dict[str, Any]:
    """Return a benchmark table whose path fields are absolute.

    A relative path in one of `_PATH_FIELDS` is joined to `root`. An empty
    string stays empty, since it states that the field is unset (as
    `${NAME:-}` does for an unset variable), and a value that still holds an
    unresolved `${NAME}` stays as written, so `plan` shows it and a run
    rejects it (`require_resolved`).

    `tokenizer` names a Hub repository or a local directory, which
    `transformers` tells apart by whether the directory exists. A relative
    `tokenizer` is resolved against `root` under that same rule: it becomes
    the absolute directory when `root` holds it, and stays a repository ID
    otherwise.

    Nested tables are copied before they are changed, so the parsed TOML
    document is left as it was.
    """
    value = dict(value)
    for table, fields in _PATH_FIELDS.items():
        if table is None:
            target = value
        elif table in value:
            target = _mapping(value[table], f"{context}.{table}")
            value[table] = target
        else:
            continue
        for field_name in fields:
            path = target.get(field_name)
            if isinstance(path, str) and path:
                target[field_name] = _rooted(path, root)

    tokenizer = value.get("tokenizer")
    if (
        isinstance(tokenizer, str)
        and tokenizer
        and not Path(tokenizer).is_absolute()
        and (root / tokenizer).is_dir()
    ):
        value["tokenizer"] = str(root / tokenizer)
    return value


def _require_own_media(benchmarks: dict[str, BenchmarkPoint]) -> None:
    """Reject two benchmarks that name the same `video.media_dir`.

    A point's media directory belongs to its run: the run requires it empty
    and a rerun moves it aside (`artifacts.supersede`), which would take a
    second point's videos with it.

    Raises:
        ValueError: Naming both benchmarks.
    """
    owners: dict[Path, str] = {}
    for name, point in benchmarks.items():
        if not point.video.media_dir:
            continue
        directory = Path(point.video.media_dir)
        if directory in owners:
            raise ValueError(
                f"benchmarks.{name}.video.media_dir is also the media "
                f"directory of benchmarks.{owners[directory]}"
            )
        owners[directory] = name


def _rooted(path: str, root: Path) -> str:
    """Return `path` joined to `root` unless it is absolute or unresolved."""
    if unresolved_environment(path) or Path(path).is_absolute():
        return path
    return str(root / path)


def _metrics(raw: Any, context: str) -> tuple[MetricDefinition, ...]:
    """Parse protected metric paths and optimization directions.

    Each key is a dotted path into the `metrics` mapping of the run summary
    and each value is `higher` or `lower`, the direction in which the metric
    improves. Run validation in `pipeline.report` requires every listed
    metric to be measured as a finite number; `pipeline.compare` reads the
    direction to tell a regression from an improvement.
    """
    table = _mapping(raw, f"{context}.metrics")
    metrics: list[MetricDefinition] = []
    for path, direction in table.items():
        if not isinstance(direction, str) or direction not in {
            "higher",
            "lower",
        }:
            raise ValueError(
                f"{context}.metrics.{path} must be 'higher' or 'lower'"
            )
        parts = tuple(str(path).split("."))
        if not parts or any(not part for part in parts):
            raise ValueError(f"{context} has an invalid metric path {path!r}")
        metrics.append(
            MetricDefinition(parts, cast(MetricDirection, direction))
        )
    return tuple(metrics)


def _load_config(raw: Any, context: str) -> LoadConfig:
    """Parse request arrival and concurrency settings."""
    if raw is None:
        return LoadConfig()
    value = _mapping(raw, context)
    _reject_unknown(value, _LOAD_FIELDS, context)
    # A bare TOML `inf` already parses as a float; the quoted string "inf" is
    # accepted as the same unbounded arrival rate.
    if "request_rate" in value and value["request_rate"] == "inf":
        value["request_rate"] = float("inf")
    return _build_settings(LoadConfig, value, context)


def _changes(
    raw: Any, server: ServerProfile, load: LoadConfig, context: str
) -> tuple[Change, ...]:
    """Parse a point's `[[changes]]`, which its timeline load applies.

    Names are unique and an `after` names a change declared before it, so
    the references form no cycle. A change that calls the admin listener
    or waits for its events requires the point's server to state
    `admin_socket`. A `cancel` disconnects streams of a closed loop
    (`request_rate = inf`), at most as many as it runs.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{context}.changes must be an array of tables")
    if load.duration_s is None:
        raise ValueError(
            f"{context}.changes apply on a timeline load and require "
            "load.duration_s"
        )

    changes: list[Change] = []
    for index, item in enumerate(raw):
        where = f"{context}.changes[{index}]"
        value = _mapping(item, where)
        _reject_unknown(value, _CHANGE_FIELDS, where)
        if "command" in value:
            command = value["command"]
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(argument, str) for argument in command)
            ):
                raise ValueError(
                    f"{where}.command must be a non-empty string array"
                )
            value["command"] = tuple(command)
        if "body" in value:
            value["body"] = _mapping(value["body"], f"{where}.body")
        change = _build_settings(Change, value, where)

        names = {earlier.name for earlier in changes}
        if change.name in names:
            raise ValueError(f"{where} repeats the name {change.name!r}")
        if change.after is not None and change.after not in names:
            raise ValueError(f"{where}.after must name an earlier change")
        if change.calls_admin and server.admin_socket is None:
            raise ValueError(
                f"{where} uses the admin listener, and servers."
                f"{server.name} states no admin_socket"
            )
        if change.cancel is not None and (
            not math.isinf(load.request_rate)
            or change.cancel > (load.max_concurrency or 0)
        ):
            raise ValueError(
                f"{where}.cancel disconnects at most the max_concurrency "
                "streams of a closed loop (request_rate = inf)"
            )
        changes.append(change)
    return tuple(changes)


def _sampling_config(
    raw: Any, task: type[BenchmarkTask], context: str
) -> SamplingConfig:
    """Parse sampling settings with task-specific streaming defaults.

    `server_sampling = true` declares that the server's configured sampling
    schedule governs every request, as on a block-diffusion server that
    refuses token-sampling controls. TOML has no null, so this key stands
    for setting `temperature`, `top_p`, and `ignore_eos` to `None`; the
    requests then carry no token-sampling control at all, and the table may
    not name any of them. `sampling_seed` stays allowed: a seed selects the
    random stream of the server's own schedule. The key itself is not a
    `SamplingConfig` field: the resolved workload shows the omitted controls
    as nulls.
    """
    value = dict(_mapping(raw, context)) if raw is not None else {}
    server_sampling = value.pop("server_sampling", False)
    if not isinstance(server_sampling, bool):
        raise ValueError(f"{context}.server_sampling must be a boolean")
    _reject_unknown(value, _SAMPLING_FIELDS, context)
    if server_sampling:
        named = sorted(_TOKEN_SAMPLING_CONTROLS & set(value))
        if named:
            raise ValueError(
                f"{context} declares server_sampling and sets token-sampling "
                f"controls: {', '.join(named)}"
            )
        value.update(temperature=None, top_p=None, ignore_eos=None)
    if "stream" not in value:
        value["stream"] = task.default_stream
    extra = value.get("extra_body", {})
    if extra and not isinstance(extra, dict):
        raise ValueError(f"{context}.extra_body must be a table")
    return _build_settings(SamplingConfig, value, context)


def _image_config(
    raw: Any, task: type[BenchmarkTask], context: str
) -> ImageConfig:
    """Parse image-generation settings accepted by the selected task."""
    if raw is None:
        return ImageConfig()
    if not task.accepts_image:
        raise ValueError(f"{context} is not valid for this task")
    value = _mapping(raw, context)
    _reject_unknown(value, _IMAGE_FIELDS, context)
    if "cfg_interval" in value:
        interval = value["cfg_interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{context}.cfg_interval must have two values")
        # TOML arrays arrive as lists; the dataclass field is a float pair.
        try:
            value["cfg_interval"] = (float(interval[0]), float(interval[1]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{context}.cfg_interval must have two numeric values"
            ) from error
    return _build_settings(ImageConfig, value, context)


def _video_config(raw: Any, context: str) -> VideoConfig:
    """Parse synchronous video-generation settings."""
    if raw is None:
        return VideoConfig()
    value = _mapping(raw, context)
    _reject_unknown(value, _VIDEO_FIELDS, context)
    extra = value.get("extra_params", {})
    if not isinstance(extra, dict):
        raise ValueError(f"{context}.extra_params must be a table")
    return _build_settings(VideoConfig, value, context)


def _build_settings(
    cls: type[_Settings], value: dict[str, Any], context: str
) -> _Settings:
    """Construct a settings dataclass and report a rejected value as schema.

    The dataclasses check their values in `__post_init__`: a value of the
    wrong type fails there with `TypeError`, for example when a string is
    compared with a number, and a violated constraint with `ValueError`.
    Neither names the profile table, so both are raised as `ValueError`
    prefixed with `context`. A mistyped value that no check touches is
    accepted as given.
    """
    try:
        return cls(**value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: {error}") from error


def _suite_profile(
    name: str,
    raw: Any,
    benchmarks: dict[str, BenchmarkPoint],
) -> SuiteProfile:
    """Validate an ordered suite against the declared benchmark points."""
    value = _mapping(raw, f"suites.{name}")
    _reject_unknown(value, {"points"}, f"suites.{name}")
    points = value.get("points")
    if (
        not isinstance(points, list)
        or not points
        or not all(isinstance(item, str) for item in points)
    ):
        raise ValueError(
            f"suites.{name}.points must be a non-empty string array"
        )
    if len(set(points)) != len(points):
        raise ValueError(f"suites.{name}.points must be unique")
    unknown = [point for point in points if point not in benchmarks]
    if unknown:
        raise ValueError(
            f"suites.{name} names unknown points: {', '.join(unknown)}"
        )
    return SuiteProfile(name, tuple(points))


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a named root table or an empty table when absent."""
    return _mapping(raw.get(key, {}), key)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    """Return a defensive copy of a required TOML mapping.

    The copy is shallow; callers may add or replace top-level keys without
    changing the parsed TOML document.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table")
    return dict(value)


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], context: str
) -> None:
    """Reject keys outside a table's public configuration schema."""
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")
