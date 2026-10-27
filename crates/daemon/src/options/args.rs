//! The command line that resolves a [`ServeConfig`].
//!
//! Options are read from three places, each overriding the last: the TOML
//! file `--config` names, the flags that name the most common options (each
//! a shortcut for one path of the tree, [`FLAG_PATHS`]), and repeatable
//! `--set PATH=VALUE` assignments that reach any option by its path, such as
//! `--set cuda_graph_config.policy=off`. The three are merged as TOML values
//! and deserialized once, so every option is checked by the same type
//! whichever place states it, and an error names the option's path.

use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;

use anyhow::{Context as _, anyhow};
use clap::Args;
use toml::{Table, Value};

use super::ServeConfig;

/// Each flag and the configuration path it sets.
pub const FLAG_PATHS: [(&str, &str); 15] = [
    ("MODEL", "model"),
    ("--host", "server_config.host"),
    ("--port", "server_config.port"),
    ("--served-model-name", "served_model_name"),
    ("--worker-groups", "worker_groups"),
    ("--tensor-parallel-size", "tensor_parallel_size"),
    ("--data-parallel-size", "data_parallel_size"),
    ("--max-running-requests", "max_running_requests"),
    ("--max-model-len", "max_model_len"),
    ("--mem-fraction-static", "mem_fraction_static"),
    ("--host-identity", "host_identity"),
    ("--nnodes", "nnodes"),
    ("--node-rank", "node_rank"),
    ("--dist-init-addr", "dist_init_addr"),
    ("--worker-executable", "worker_config.executable"),
];

/// The serving command line.
#[derive(Debug, Clone, Default, Args)]
pub struct ServeArgs {
    /// Local model directory or Hugging Face repository (`model`).
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,
    /// TCP address the server listens on (`server_config.host`).
    #[arg(long)]
    pub host: Option<String>,
    /// TCP port the server listens on (`server_config.port`).
    #[arg(long)]
    pub port: Option<u16>,
    /// The single model name the API serves (`served_model_name`).
    #[arg(long)]
    pub served_model_name: Option<String>,
    /// Deployment file placing worker groups and their components
    /// (`worker_groups`).
    #[arg(long, value_name = "FILE")]
    pub worker_groups: Option<PathBuf>,
    /// Tensor-parallel workers of each scheduler without a deployment file
    /// (`tensor_parallel_size`).
    #[arg(long)]
    pub tensor_parallel_size: Option<NonZeroUsize>,
    /// Schedulers (`data_parallel_size`).
    #[arg(long)]
    pub data_parallel_size: Option<NonZeroUsize>,
    /// Concurrently resident requests (`max_running_requests`).
    #[arg(long)]
    pub max_running_requests: Option<NonZeroUsize>,
    /// Longest sequence a request may reach, in tokens (`max_model_len`).
    #[arg(long)]
    pub max_model_len: Option<NonZeroU32>,
    /// Each rank's share of its device's memory, in (0, 1]
    /// (`mem_fraction_static`).
    #[arg(long)]
    pub mem_fraction_static: Option<f64>,
    /// This node's host name in a deployment of several nodes
    /// (`host_identity`).
    #[arg(long)]
    pub host_identity: Option<String>,
    /// Nodes of the deployment, each running one `uniserve serve`
    /// (`nnodes`).
    #[arg(long)]
    pub nnodes: Option<NonZeroU32>,
    /// This node's rank; the leader is rank 0 (`node_rank`).
    #[arg(long)]
    pub node_rank: Option<u32>,
    /// The leader's `host:port` the other nodes join at
    /// (`dist_init_addr`).
    #[arg(long, value_name = "HOST:PORT")]
    pub dist_init_addr: Option<String>,
    /// The `uniserve` executable that runs the workers
    /// (`worker_config.executable`).
    #[arg(long, value_name = "PATH")]
    pub worker_executable: Option<PathBuf>,
    /// TOML file of serving options; the flags and `--set` override it.
    /// `--print-config` prints every option it may state.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Set the option at PATH, such as `cuda_graph_config.policy=off` or
    /// `moe_config.exchange=deepep`; repeatable, and it overrides the flags
    /// and the file. VALUE is a TOML value (`2`, `true`, `[1024, 4096]`,
    /// `{ mode = "balanced" }`) or JSON, and otherwise text; a list may be
    /// comma-separated text. One invocation may not set both a table and an
    /// option inside it.
    #[arg(long = "set", value_name = "PATH=VALUE")]
    pub set: Vec<String>,
    /// Print the resolved configuration, defaults included, as TOML and
    /// exit.
    #[arg(long)]
    pub print_config: bool,
    /// Take a deployment over from the daemon at the other end of the
    /// handoff socket at this descriptor, as `POST /admin/restart` starts
    /// its successor (`daemon::restart`); every other argument is ignored.
    #[arg(long, hide = true)]
    pub takeover: Option<i32>,
}

/// The configuration paths an invocation stated: every table and option the
/// file, the flags and `--set` named, rather than left at its default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stated(BTreeSet<String>);

impl Stated {
    /// Collects the paths of every key of `tree`, tables and options alike.
    fn of(tree: &Table) -> Self {
        fn collect(table: &Table, prefix: &str, paths: &mut BTreeSet<String>) {
            for (key, value) in table {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if let Value::Table(table) = value {
                    collect(table, &path, paths);
                }
                paths.insert(path);
            }
        }

        let mut paths = BTreeSet::new();
        collect(tree, "", &mut paths);
        Self(paths)
    }

    /// Whether the option or table at `path` was stated.
    /// The stated paths, as a hot restart hands them to the successor.
    pub fn paths(&self) -> Vec<String> {
        self.0.iter().cloned().collect()
    }

    /// The statement of `paths`, as [`Stated::paths`] gave them.
    pub fn from_paths(paths: Vec<String>) -> Self {
        Self(paths.into_iter().collect())
    }

    pub fn contains(&self, path: &str) -> bool {
        self.0.contains(path)
    }

    /// Whether `path` or anything inside it was stated.
    pub fn contains_within(&self, path: &str) -> bool {
        let prefix = format!("{path}.");
        self.0
            .iter()
            .any(|stated| stated == path || stated.starts_with(&prefix))
    }
}

impl ServeArgs {
    /// Resolves the configuration: the `--config` file, overridden by the
    /// flags, overridden by `--set`, with every unstated option at its
    /// default and `server_config.api_key` falling back to
    /// `UNISERVE_API_KEY`. Returns it with the paths the invocation stated.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be read or is not TOML, a `--set`
    /// assignment is malformed or conflicts with another, a path crosses an
    /// option that is not a table, or the merged tree names an unknown
    /// option or a value its option does not take; the message names the
    /// option's path.
    pub fn resolve(&self) -> anyhow::Result<(ServeConfig, Stated)> {
        let mut tree = match &self.config {
            Some(path) => std::fs::read_to_string(path)
                .with_context(|| format!("reading --config {}", path.display()))?
                .parse::<Table>()
                .with_context(|| format!("parsing --config {}", path.display()))?,
            None => Table::new(),
        };
        for (path, value) in self.flag_values() {
            let path = path.split('.').map(str::to_owned).collect::<Vec<_>>();
            set(&mut tree, &path, value)?;
        }
        for (path, value) in assignments(&self.set)? {
            set(&mut tree, &path, value)?;
        }
        configured(tree)
    }

    /// The values the flags state, by configuration path.
    fn flag_values(&self) -> Vec<(&'static str, Value)> {
        let text = |value: &str| Value::String(value.to_owned());
        let path = |value: &PathBuf| Value::String(value.to_string_lossy().into_owned());
        let count = |value: u64| Value::Integer(i64::try_from(value).unwrap_or(i64::MAX));

        [
            ("model", self.model.as_deref().map(text)),
            ("server_config.host", self.host.as_deref().map(text)),
            (
                "server_config.port",
                self.port.map(|port| Value::Integer(i64::from(port))),
            ),
            (
                "served_model_name",
                self.served_model_name.as_deref().map(text),
            ),
            ("worker_groups", self.worker_groups.as_ref().map(path)),
            (
                "tensor_parallel_size",
                self.tensor_parallel_size
                    .map(|size| count(size.get() as u64)),
            ),
            (
                "data_parallel_size",
                self.data_parallel_size.map(|size| count(size.get() as u64)),
            ),
            (
                "max_running_requests",
                self.max_running_requests
                    .map(|requests| count(requests.get() as u64)),
            ),
            (
                "max_model_len",
                self.max_model_len
                    .map(|length| count(u64::from(length.get()))),
            ),
            (
                "mem_fraction_static",
                self.mem_fraction_static.map(Value::Float),
            ),
            ("host_identity", self.host_identity.as_deref().map(text)),
            (
                "nnodes",
                self.nnodes.map(|nodes| count(u64::from(nodes.get()))),
            ),
            (
                "node_rank",
                self.node_rank.map(|rank| count(u64::from(rank))),
            ),
            ("dist_init_addr", self.dist_init_addr.as_deref().map(text)),
            (
                "worker_config.executable",
                self.worker_executable.as_ref().map(path),
            ),
        ]
        .into_iter()
        .filter_map(|(path, value)| value.map(|value| (path, value)))
        .collect()
    }
}

/// The configuration of a TOML tree of options, with the paths it states.
fn configured(tree: Table) -> anyhow::Result<(ServeConfig, Stated)> {
    let stated = Stated::of(&tree);
    let mut config: ServeConfig =
        serde_path_to_error::deserialize(Value::Table(tree)).map_err(|error| {
            let path = error.path().to_string();
            let cause = error.into_inner();
            if path == "." {
                anyhow!("invalid configuration: {cause}")
            } else {
                anyhow!("invalid option `{path}`: {cause}")
            }
        })?;
    // The environment supplies the key the configuration leaves unset or
    // blank.
    config.server_config.api_key =
        super::configured_api_key(config.server_config.api_key.as_deref());
    Ok((config, stated))
}

/// Resolves a configuration from `text`, a TOML file's contents, overridden
/// by the `--set` assignments `sets`, as [`ServeArgs::resolve`] does for a
/// command line without flags.
///
/// # Errors
///
/// Fails as [`ServeArgs::resolve`] fails.
pub fn resolve_text(text: &str, sets: &[String]) -> anyhow::Result<(ServeConfig, Stated)> {
    let mut tree = text.parse::<Table>().context("parsing the configuration")?;
    for (path, value) in assignments(sets)? {
        set(&mut tree, &path, value)?;
    }
    configured(tree)
}

/// Prints a configuration as TOML.
///
/// # Errors
///
/// Fails when a value has no TOML form.
pub fn printed(config: &ServeConfig) -> anyhow::Result<String> {
    toml::to_string_pretty(config).context("printing the configuration")
}

/// `config` with the `--set` assignments `texts` applied over it, as
/// [`ServeArgs::resolve`] applies them over the file and the flags, and the
/// paths they assign. The API key is the one `config` holds.
///
/// # Errors
///
/// Fails as [`ServeArgs::resolve`] fails on an assignment: one that is
/// malformed or conflicts with another, a path that crosses an option that
/// is not a table, or a value its option does not take.
pub fn overridden(
    config: &ServeConfig,
    texts: &[String],
) -> anyhow::Result<(ServeConfig, Vec<String>)> {
    let mut tree = match Value::try_from(config).context("encoding the configuration")? {
        Value::Table(tree) => tree,
        _ => return Err(anyhow!("the configuration is not a table")),
    };
    let mut paths = Vec::new();
    for (path, value) in assignments(texts)? {
        paths.push(path.join("."));
        set(&mut tree, &path, value)?;
    }
    let mut overridden: ServeConfig = serde_path_to_error::deserialize(Value::Table(tree))
        .map_err(|error| {
            let path = error.path().to_string();
            anyhow!("invalid option `{path}`: {}", error.into_inner())
        })?;
    overridden.server_config.api_key = config.server_config.api_key.clone();
    Ok((overridden, paths))
}

/// Parses `--set` assignments into paths and values.
///
/// A later assignment of the same path replaces an earlier one; assigning
/// both a table and an option inside it is refused, since the result would
/// depend on their order.
fn assignments(texts: &[String]) -> anyhow::Result<Vec<(Vec<String>, Value)>> {
    let mut assigned: Vec<(Vec<String>, Value)> = Vec::new();
    for text in texts {
        let (path, value) = text
            .split_once('=')
            .with_context(|| format!("--set {text:?}: expected PATH=VALUE"))?;
        let path = path
            .trim()
            .split('.')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            path.iter().all(|key| {
                !key.is_empty()
                    && key
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            }),
            "--set {text:?}: a path is dot-separated option names"
        );
        let value = parsed_value(value).with_context(|| format!("--set {text:?}"))?;

        if let Some(conflict) = assigned.iter().find(|(other, _)| {
            let shared = other.len().min(path.len());
            other[..shared] == path[..shared] && *other != path
        }) {
            anyhow::bail!(
                "--set `{}` and `{}` cannot both be set: one contains the other",
                conflict.0.join("."),
                path.join(".")
            );
        }
        assigned.retain(|(other, _)| *other != path);
        assigned.push((path, value));
    }
    Ok(assigned)
}

/// Reads a `--set` value: a TOML value, else JSON, else the text itself.
fn parsed_value(text: &str) -> anyhow::Result<Value> {
    anyhow::ensure!(!text.trim().is_empty(), "the value is empty");

    // The text is a TOML value when it parses as the only key of a document;
    // text that adds keys of its own is not one value.
    if let Ok(document) = format!("value = {text}").parse::<Table>()
        && document.len() == 1
        && let Some(value) = document.get("value")
    {
        return Ok(value.clone());
    }
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(text)
        && let Ok(value) = Value::try_from(json)
    {
        return Ok(value);
    }
    Ok(Value::String(text.to_owned()))
}

/// Sets `value` at `path` of `tree`, creating the tables on the way.
fn set(tree: &mut Table, path: &[String], value: Value) -> anyhow::Result<()> {
    let Some((last, parents)) = path.split_last() else {
        anyhow::bail!("an empty option path");
    };
    let mut table = tree;
    for (index, key) in parents.iter().enumerate() {
        table = table
            .entry(key.clone())
            .or_insert_with(|| Value::Table(Table::new()))
            .as_table_mut()
            .with_context(|| {
                format!(
                    "cannot set `{}`: `{}` is not a table",
                    path.join("."),
                    path[..=index].join(".")
                )
            })?;
    }
    table.insert(last.clone(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use clap::{FromArgMatches, Parser};
    use proto::request::{ExpertExchange, GraphPolicy, SchedulingPolicy};

    use super::*;

    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        serve: ServeArgs,
    }

    fn resolved(args: &[&str]) -> anyhow::Result<(ServeConfig, Stated)> {
        let command = <Command as clap::CommandFactory>::command();
        let matches =
            command.try_get_matches_from(std::iter::once("serve").chain(args.iter().copied()))?;
        Command::from_arg_matches(&matches)?.serve.resolve()
    }

    fn written(name: &str, body: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "uniserve-options-{name}-{}-{:?}.toml",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, body).expect("write test file");
        path
    }

    /// The file states options by path, a flag overrides the file, and
    /// `--set` overrides both; everything unstated keeps its default.
    #[test]
    fn set_overrides_flags_which_override_the_file() {
        let file = written(
            "precedence",
            "model = \"/models/file\"\nmax_running_requests = 4\n\
             [cuda_graph_config]\npolicy = \"full\"\n\
             [scheduler_config]\npolicy = \"priority\"\n",
        );
        let file = file.to_str().unwrap();

        let (config, _) = resolved(&["--config", file]).unwrap();
        assert_eq!(config.model.as_deref(), Some("/models/file"));
        assert_eq!(config.max_running_requests.get(), 4);
        assert_eq!(config.cuda_graph_config.policy, GraphPolicy::Full);

        let (config, _) = resolved(&[
            "/models/flag",
            "--config",
            file,
            "--max-running-requests",
            "2",
            "--set",
            "max_running_requests=1",
            "--set",
            "cuda_graph_config.policy=off",
        ])
        .unwrap();
        assert_eq!(config.model.as_deref(), Some("/models/flag"));
        assert_eq!(config.max_running_requests.get(), 1);
        assert_eq!(config.cuda_graph_config.policy, GraphPolicy::Off);
        assert_eq!(config.scheduler_config.policy, SchedulingPolicy::Priority);
        assert_eq!(config.server_config.port, 8000);
    }

    /// A `--set` value is a TOML value, JSON, or text; lists also take
    /// comma-separated text, and durations and sizes take units.
    #[test]
    fn set_values_read_as_toml_json_or_text() {
        let (config, _) = resolved(&[
            "--set",
            "video_config.text_capacities=1024,10240",
            "--set",
            "video_config.aspect_ratios=[\"16:9\"]",
            "--set",
            "quantization_config={\"mode\":\"balanced\"}",
            "--set",
            "moe_config={ expert_parallel = true, exchange = \"deepep\" }",
            "--set",
            "server_config.request_timeout=2m",
            "--set",
            "flashinfer_config.workspace_size=1GiB",
            "--set",
            "readout_config.canvas=64",
            "--set",
            "server_config.host=0.0.0.0",
        ])
        .unwrap();
        assert_eq!(
            config
                .video_config
                .text_capacities
                .map(|sizes| sizes.iter().map(|size| size.get()).collect::<Vec<_>>()),
            Some(vec![1024, 10240])
        );
        assert_eq!(config.video_config.aspect_ratios, ["16:9"]);
        assert_eq!(config.quantization_config["mode"], "balanced");
        assert!(config.moe_config.expert_parallel);
        assert_eq!(config.moe_config.exchange, ExpertExchange::DeepEp);
        assert_eq!(
            config.server_config.request_timeout,
            Some(Duration::from_secs(120))
        );
        assert_eq!(config.flashinfer_config.workspace_size, 1 << 30);
        assert_eq!(
            config.readout_config.canvas,
            crate::options::CanvasMode::Fixed(64)
        );
        assert_eq!(config.server_config.host, "0.0.0.0");
    }

    /// An unknown option, a value its option does not take, a path through
    /// an option, and a table set beside an option inside it are refused,
    /// naming the path.
    #[test]
    fn malformed_options_are_refused_by_path() {
        for (args, named) in [
            (vec!["--set", "cuda_graph_config.polcy=off"], "polcy"),
            (
                vec!["--set", "max_running_requests=0"],
                "max_running_requests",
            ),
            (
                vec!["--set", "cuda_graph_config.policy=sometimes"],
                "cuda_graph_config.policy",
            ),
            (vec!["--set", "max_batch=4"], "max_batch"),
            (vec!["--set", "dtype.size=2"], "dtype"),
            (
                vec![
                    "--set",
                    "moe_config={ exchange = \"deepep\" }",
                    "--set",
                    "moe_config.microbatches=2",
                ],
                "moe_config",
            ),
            (vec!["--set", "no_value"], "PATH=VALUE"),
            (
                vec!["--set", "video_config..max_seconds=5"],
                "dot-separated",
            ),
        ] {
            let error = resolved(&args).unwrap_err();
            assert!(format!("{error:#}").contains(named), "{args:?}: {error:#}");
        }
    }

    /// The stated paths are what the invocation named, tables and options
    /// alike, and nothing it left at its default.
    #[test]
    fn stated_paths_are_what_was_named() {
        let (_, stated) =
            resolved(&["m", "--port", "9000", "--set", "video_config.max_seconds=5"]).unwrap();
        for path in [
            "model",
            "server_config",
            "server_config.port",
            "video_config.max_seconds",
        ] {
            assert!(stated.contains(path), "{path} was stated");
        }
        assert!(stated.contains_within("video_config"));
        assert!(!stated.contains("server_config.host"));
        assert!(!stated.contains_within("cuda_graph_config"));
    }

    /// Every flag is a shortcut for the path it names, and the printed
    /// configuration reads back as the same configuration.
    #[test]
    fn flags_set_their_paths_and_printing_round_trips() {
        let (config, stated) = resolved(&[
            "/models/m",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--served-model-name",
            "served",
            "--worker-groups",
            "deployment.toml",
            "--tensor-parallel-size",
            "2",
            "--data-parallel-size",
            "2",
            "--max-running-requests",
            "4",
            "--max-model-len",
            "4096",
            "--mem-fraction-static",
            "0.5",
            "--host-identity",
            "rank-0",
            "--nnodes",
            "2",
            "--node-rank",
            "0",
            "--dist-init-addr",
            "rank-0:5000",
            "--worker-executable",
            "/opt/env/bin/uniserve",
        ])
        .unwrap();
        for (_, path) in FLAG_PATHS {
            assert!(stated.contains(path), "{path} is set by its flag");
        }
        assert_eq!(
            config.worker_config.executable,
            PathBuf::from("/opt/env/bin/uniserve")
        );

        let text = printed(&config).unwrap();
        let reread: ServeConfig = toml::from_str(&text).unwrap();
        assert_eq!(reread, config, "{text}");
    }

    /// Resolves `args`, checks them, and derives the engine configuration
    /// the engine reads from the spec they state.
    fn lowered(args: &[&str]) -> anyhow::Result<(ServeConfig, engine::EngineConfig)> {
        let (config, stated) = resolved(args)?;
        config.validate(&stated)?;
        let mut spec = config.engine_spec()?;
        spec.model_facts = Some(Default::default());
        let engine = engine::EngineConfig::from_spec(&spec)?;
        Ok((config, engine))
    }

    /// Only the leader serves a model; every node of a deployment of
    /// several needs the leader's address, and its rank among them.
    #[test]
    fn nodes_other_than_the_leader_take_no_model() {
        let member = [
            "--nnodes",
            "2",
            "--node-rank",
            "1",
            "--dist-init-addr",
            "rank-0:5000",
        ];
        let (config, stated) = resolved(&member).unwrap();
        config.validate(&stated).unwrap();
        assert!(!config.leads());

        let (config, stated) = resolved(&["--nnodes", "2", "--node-rank", "1"]).unwrap();
        let error = config.validate(&stated).unwrap_err();
        assert!(error.to_string().contains("dist_init_addr"), "{error}");

        let outside = [
            "--nnodes",
            "2",
            "--node-rank",
            "2",
            "--dist-init-addr",
            "rank-0:5000",
        ];
        let (config, stated) = resolved(&outside).unwrap();
        let error = config.validate(&stated).unwrap_err();
        assert!(error.to_string().contains("node_rank"), "{error}");

        let leader = ["--nnodes", "2", "--dist-init-addr", "rank-0:5000"];
        let (config, stated) = resolved(&leader).unwrap();
        let error = config.validate(&stated).unwrap_err();
        assert!(error.to_string().contains("`model` is required"), "{error}");
    }

    /// The model comes from the command line or the file; a single node's
    /// daemon without one validates, and serves its admin API alone until a
    /// configuration names it.
    #[test]
    fn the_model_comes_from_the_command_line_or_the_file() {
        let (config, stated) = resolved(&[]).unwrap();
        assert!(config.model.is_none());
        config.validate(&stated).unwrap();
        let (config, engine) = lowered(&["/models/m"]).unwrap();
        assert_eq!(config.model(), "/models/m");
        assert_eq!(config.served_model_name(), "/models/m");
        assert_eq!(engine.worker_process.model, "/models/m");

        let file = written("model", "model = \"/models/file\"\n");
        let (config, _) = lowered(&["--config", file.to_str().unwrap()]).unwrap();
        assert_eq!(config.model(), "/models/file");
    }

    /// Without a deployment file each scheduler's group runs one component
    /// tensor-parallel over its workers; the placement shorthand is refused
    /// beside a file.
    #[test]
    fn the_shorthand_places_engine_cores_and_is_refused_beside_a_file() {
        let (_, engine) = lowered(&[
            "m",
            "--tensor-parallel-size",
            "2",
            "--data-parallel-size",
            "2",
            "--set",
            "placement_config.device=cpu",
        ])
        .unwrap();
        assert_eq!(engine.worker_groups.len(), 2);
        assert!(
            engine
                .worker_groups
                .iter()
                .all(|worker| worker.ranks.len() == 2)
        );

        let deployment = written(
            "deployment",
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cpu*2\"]\nqueue_depth = 2\n\
             [worker_groups.components.model]\ntensor_parallel_size = 2\n",
        );
        let path = deployment.to_str().unwrap();
        assert_eq!(
            lowered(&["m", "--worker-groups", path])
                .unwrap()
                .1
                .worker_groups
                .len(),
            1
        );
        for shorthand in [
            ["--tensor-parallel-size", "2"],
            ["--set", "placement_config.device=cpu"],
            ["--set", "placement_config.queue_depth=3"],
        ] {
            let mut args = vec!["m", "--worker-groups", path];
            args.extend(shorthand);
            let error = lowered(&args).unwrap_err();
            assert!(
                error.to_string().contains("deployment file"),
                "{shorthand:?}: {error}"
            );
        }
    }

    /// A deployment rank entry without a host runs on this process's host.
    #[test]
    fn a_deployment_places_unnamed_ranks_on_this_host() {
        let deployment = written(
            "engine-deployment",
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cpu\", \"rank-1/cpu\"]\nqueue_depth = 2\n\
             [worker_groups.components.model]\ntensor_parallel_size = 2\n",
        );
        let (_, engine) = lowered(&[
            "m",
            "--worker-groups",
            deployment.to_str().unwrap(),
            "--host-identity",
            "rank-0",
        ])
        .unwrap();
        let hosts = engine.worker_groups[0]
            .ranks
            .iter()
            .map(|rank| rank.host.as_str())
            .collect::<Vec<_>>();
        assert_eq!(hosts, ["rank-0", "rank-1"]);
        assert!(lowered(&["m", "--worker-groups", "/nonexistent/deployment.toml"]).is_err());
    }

    /// Capture lists and capacities are checked before any rank starts.
    #[test]
    fn capture_lists_are_strictly_increasing_and_shapes_distinct() {
        let (_, engine) = lowered(&[
            "m",
            "--set",
            "video_config.text_capacities=1024,4096",
            "--set",
            "cuda_graph_config.flow_shapes=1152x2048,2048x1152",
        ])
        .unwrap();
        let worker = &engine.worker_process;
        assert_eq!(worker.video_text_capacities, Some(vec![1024, 4096]));
        assert_eq!(
            worker.flow_graph_shapes,
            Some(vec![[1152, 2048], [2048, 1152]])
        );

        for refused in [
            "video_config.text_capacities=4096,1024",
            "cuda_graph_config.decode_batch_sizes=1,0",
            "cuda_graph_config.flow_shapes=1152x2048,1152x2048",
            "cuda_graph_config.flow_shapes=1152",
        ] {
            assert!(
                lowered(&["m", "--set", refused]).is_err(),
                "{refused} must be refused"
            );
        }
    }

    /// The precision policy, expert parallelism, prefix caching and the
    /// scheduler budgets reach the engine configuration as stated, and
    /// unstated budgets take the engine defaults.
    #[test]
    fn grouped_options_reach_the_engine() {
        let (_, engine) = lowered(&[
            "m",
            "--set",
            "quantization_config.mode=performance",
            "--set",
            "moe_config={ expert_parallel = true, exchange = \"dwdp\" }",
            "--set",
            "kv_cache_config.prefix_caching=false",
            "--set",
            "scheduler_config.chunked_prefill_size=512",
        ])
        .unwrap();
        let worker = &engine.worker_process;
        assert_eq!(worker.quantization_config["mode"], "performance");
        assert_eq!(engine.expert_parallel, Some(ExpertExchange::Dwdp));
        assert!(!engine.prefix_cache);
        assert_eq!(engine.chunked_prefill_size, 512);
        assert_eq!(
            engine.max_num_batched_tokens,
            engine::DEFAULT_MAX_NUM_BATCHED_TOKENS
        );
        assert_eq!(
            worker.max_batch_tokens as usize,
            engine::DEFAULT_MAX_NUM_BATCHED_TOKENS
        );
    }

    /// The server listens on a non-blank host or Unix socket path, port 0
    /// asks for an ephemeral port, and a video capacity lies within the
    /// durations the video API admits.
    #[test]
    fn listeners_and_video_capacities_are_checked() {
        assert!(lowered(&["m", "--port", "0"]).is_ok());
        assert!(lowered(&["m", "--host", " "]).is_err());
        assert!(lowered(&["m", "--set", "server_config.uds=\"\""]).is_err());
        for (seconds, valid) in [("4", true), ("15", true), ("3.9", false), ("15.5", false)] {
            let set = format!("video_config.max_seconds={seconds}");
            assert_eq!(lowered(&["m", "--set", &set]).is_ok(), valid, "{seconds} s");
        }
    }

    /// An option of another model family than the served one is refused
    /// rather than ignored; unstated defaults never are.
    #[test]
    fn options_of_another_family_are_refused() {
        use models::ModelDescription::{Bagel, DiffusionGemma, MiniMaxH3, Qwen3};

        let (_, defaults) = resolved(&["m"]).unwrap();
        for description in [Qwen3, Bagel, MiniMaxH3, DiffusionGemma] {
            assert!(defaults.refuse_foreign_options(description).is_ok());
        }

        let refused = |set: &str, description| {
            let (_, stated) = resolved(&["m", "--set", set]).unwrap();
            stated.refuse_foreign_options(description).is_err()
        };
        assert!(refused("readout_config.canvas=compact", Qwen3));
        assert!(!refused("readout_config.canvas=compact", DiffusionGemma));
        assert!(refused("video_config.max_seconds=5", DiffusionGemma));
        assert!(!refused("video_config.max_seconds=5", MiniMaxH3));
        // A video pipeline schedules no token batches and renders no chat.
        for token_option in [
            "max_num_batched_tokens=2048",
            "scheduler_config.chunked_prefill_size=512",
            "chat_config.reasoning_parser=none",
        ] {
            assert!(refused(token_option, MiniMaxH3), "{token_option}");
            assert!(!refused(token_option, Qwen3), "{token_option}");
        }
    }
}
