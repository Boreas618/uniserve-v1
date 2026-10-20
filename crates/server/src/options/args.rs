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
pub const FLAG_PATHS: [(&str, &str); 12] = [
    ("MODEL", "model"),
    ("--host", "server_config.host"),
    ("--port", "server_config.port"),
    ("--served-model-name", "served_model_name"),
    ("--workers", "workers"),
    ("--tensor-parallel-size", "tensor_parallel_size"),
    ("--data-parallel-size", "data_parallel_size"),
    ("--max-running-requests", "max_running_requests"),
    ("--max-model-len", "max_model_len"),
    ("--mem-fraction-static", "mem_fraction_static"),
    ("--host-identity", "host_identity"),
    ("--worker-python", "worker_config.python"),
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
    /// (`workers`).
    #[arg(long, value_name = "FILE")]
    pub workers: Option<PathBuf>,
    /// Tensor-parallel ranks of each replica without a deployment file
    /// (`tensor_parallel_size`).
    #[arg(long)]
    pub tensor_parallel_size: Option<NonZeroUsize>,
    /// Independent replicas (`data_parallel_size`).
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
    /// This process's host name in a multi-host deployment
    /// (`host_identity`).
    #[arg(long)]
    pub host_identity: Option<String>,
    /// Python interpreter that runs the worker ranks
    /// (`worker_config.python`).
    #[arg(long, value_name = "PATH")]
    pub worker_python: Option<PathBuf>,
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

    /// The top-level tables among `tables` that were stated.
    pub(super) fn tables(&self, tables: &[&str]) -> Vec<String> {
        tables
            .iter()
            .filter(|table| self.contains(table))
            .map(|table| (*table).to_owned())
            .collect()
    }
}

impl ServeArgs {
    /// Resolves the configuration: the `--config` file, overridden by the
    /// flags, overridden by `--set`, with every unstated option at its
    /// default. Returns it with the paths the invocation stated.
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

        let stated = Stated::of(&tree);
        let config: ServeConfig =
            serde_path_to_error::deserialize(Value::Table(tree)).map_err(|error| {
                let path = error.path().to_string();
                let cause = error.into_inner();
                if path == "." {
                    anyhow!("invalid configuration: {cause}")
                } else {
                    anyhow!("invalid option `{path}`: {cause}")
                }
            })?;
        Ok((config, stated))
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
            ("workers", self.workers.as_ref().map(path)),
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
                "worker_config.python",
                self.worker_python.as_ref().map(path),
            ),
        ]
        .into_iter()
        .filter_map(|(path, value)| value.map(|value| (path, value)))
        .collect()
    }
}

/// Prints a configuration as TOML.
///
/// # Errors
///
/// Fails when a value has no TOML form.
pub fn printed(config: &ServeConfig) -> anyhow::Result<String> {
    toml::to_string_pretty(config).context("printing the configuration")
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
        assert_eq!(
            config.cuda_graph_config.policy,
            uniserve_engine::GraphPolicy::Full
        );

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
        assert_eq!(
            config.cuda_graph_config.policy,
            uniserve_engine::GraphPolicy::Off
        );
        assert_eq!(
            config.scheduler_config.policy,
            uniserve_engine::SchedulingPolicy::Priority
        );
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
        assert_eq!(
            config.moe_config.exchange,
            uniserve_engine::ExpertExchange::DeepEp
        );
        assert_eq!(
            config.server_config.request_timeout,
            Some(Duration::from_secs(120))
        );
        assert_eq!(config.flashinfer_config.workspace_size, 1 << 30);
        assert_eq!(
            config.readout_config.canvas,
            crate::serving::systemone::CanvasMode::Fixed(64)
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
            "--workers",
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
            "--worker-python",
            "/usr/bin/python3",
            "--set",
            "lanes=[{ lane_id = \"decode\", sm_budget = 64, domains = [\"decode\"] }]",
        ])
        .unwrap();
        for (_, path) in FLAG_PATHS {
            assert!(stated.contains(path), "{path} is set by its flag");
        }
        assert_eq!(
            config.worker_config.python,
            PathBuf::from("/usr/bin/python3")
        );

        let text = printed(&config).unwrap();
        let reread: ServeConfig = toml::from_str(&text).unwrap();
        assert_eq!(reread, config, "{text}");
    }
}
