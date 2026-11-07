//! The command line that resolves a [`ServerArgs`].
//!
//! Options are read from three places, each overriding the last: the TOML
//! file `--config` names, the flags that name the most common options (each
//! the option of its own name, declared on the option's field of
//! [`ServerArgs`]), and repeatable
//! `--set PATH=VALUE` assignments that reach any option by its path, such as
//! `--set cuda_graph_config.policy=off`. The three are merged as TOML values
//! and deserialized once, so every option is checked by the same type
//! whichever place states it, and an error names the option's path.

use std::collections::BTreeSet;

use anyhow::{Context as _, anyhow};
use toml::{Table, Value};

use super::ServerArgs;

/// Which process the `--host` and `--port` flags state the listening address
/// of: the daemon's own server (`admin_config`) or the application's
/// (`server_config`, which `uniserve frontend` serves).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Listen {
    Daemon,
    Frontend,
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

impl ServerArgs {
    /// Resolves the configuration this command line states: the `--config` file, overridden by
    /// the flags the command line states, overridden by `--set`, with every unstated option at its
    /// default and `server_config.api_key` falling back to `UNISERVE_API_KEY`. `--host` and
    /// `--port` state `listen`'s address. Returns it with the paths the invocation stated.
    ///
    /// `matches` are the command line's own: a flag counts when the command line states it, which
    /// clap's value source tells apart from the default it parsed beside the stated ones.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be read or is not TOML, a `--set`
    /// assignment is malformed or conflicts with another, a path crosses an
    /// option that is not a table, or the merged tree names an unknown
    /// option or a value its option does not take; the message names the
    /// option's path.
    pub fn resolve(
        &self,
        matches: &clap::ArgMatches,
        listen: Listen,
    ) -> anyhow::Result<(ServerArgs, Stated)> {
        let mut tree = match &self.config_file {
            Some(path) => std::fs::read_to_string(path)
                .with_context(|| format!("reading --config {}", path.display()))?
                .parse::<Table>()
                .with_context(|| format!("parsing --config {}", path.display()))?,
            None => Table::new(),
        };
        // Each stated flag is the option of its own name, as this value prints it.
        let printed = Value::try_from(self).context("printing the command line's options")?;
        for id in matches.ids() {
            let id = id.as_str();
            if matches.value_source(id) != Some(clap::parser::ValueSource::CommandLine) {
                continue;
            }
            if let Some(value) = printed.get(id) {
                set(&mut tree, &[id.to_owned()], value.clone())?;
            }
        }
        let (host, port) = match listen {
            Listen::Daemon => ("admin_config", "admin_config"),
            Listen::Frontend => ("server_config", "server_config"),
        };
        let path = |table: &str, key: &str| vec![table.to_owned(), key.to_owned()];
        if let Some(value) = &self.host {
            set(&mut tree, &path(host, "host"), Value::String(value.clone()))?;
        }
        if let Some(value) = self.port {
            set(
                &mut tree,
                &path(port, "port"),
                Value::Integer(i64::from(value)),
            )?;
        }
        if let Some(value) = &self.worker_executable {
            let value = Value::String(value.to_string_lossy().into_owned());
            set(&mut tree, &path("worker_config", "executable"), value)?;
        }
        for (path, value) in assignments(&self.set)? {
            set(&mut tree, &path, value)?;
        }
        configured(tree)
    }
}

/// A TOML tree of options as `T`, refusing a key no option takes.
///
/// The configuration sections do not refuse unknown fields themselves: the same types travel to
/// the frontend and the engine processes, which may be of another build than this daemon
/// (`proto::config`). The file is where a key nothing reads is a mistake, so it is refused here,
/// by its path.
fn deserialize_options<T: serde::de::DeserializeOwned>(tree: Table) -> anyhow::Result<T> {
    let mut unknown = Vec::new();
    let mut record = |path: serde_ignored::Path<'_>| unknown.push(path.to_string());
    let parsed: Result<T, _> = serde_path_to_error::deserialize(serde_ignored::Deserializer::new(
        Value::Table(tree),
        &mut record,
    ));
    let parsed = parsed.map_err(|error| {
        let path = error.path().to_string();
        let cause = error.into_inner();
        if path == "." {
            anyhow!("invalid configuration: {cause}")
        } else {
            anyhow!("invalid option `{path}`: {cause}")
        }
    })?;
    if let Some(path) = unknown.first() {
        return Err(anyhow!("unknown option `{path}`: no option takes it"));
    }
    Ok(parsed)
}

/// The configuration of a TOML tree of options, with the paths it states.
fn configured(tree: Table) -> anyhow::Result<(ServerArgs, Stated)> {
    let stated = Stated::of(&tree);
    let mut config: ServerArgs = deserialize_options(tree)?;
    // A retired option's name is refused here, with the configuration, rather
    // than left unknown: an unknown field names the mistake but not the way
    // out, and a table nothing reads would be silent. The field that parses it
    // is never serialized, so a dump cannot carry it back.
    anyhow::ensure!(
        config.retired_quantization_config.is_none(),
        "`quantization_config` is retired: state the precision policy as `quantization` \
         (`--set quantization.mode=...`, or `[quantization]` in --config)"
    );
    // The environment supplies the key the configuration leaves unset or
    // blank.
    config.server_config.api_key =
        super::configured_api_key(config.server_config.api_key.as_deref());
    Ok((config, stated))
}

/// Resolves a configuration from `text`, a TOML file's contents, overridden
/// by the `--set` assignments `sets`, as [`ServerArgs::resolve`] does for a
/// command line without flags.
///
/// # Errors
///
/// Fails as [`ServerArgs::resolve`] fails.
pub fn resolve_text(text: &str, sets: &[String]) -> anyhow::Result<(ServerArgs, Stated)> {
    let mut tree = text.parse::<Table>().context("parsing the configuration")?;
    for (path, value) in assignments(sets)? {
        set(&mut tree, &path, value)?;
    }
    configured(tree)
}

/// Prints reloadable TOML containing the configuration's non-default options.
/// Defaults must remain unstated: model-family and rank-assignment validation
/// distinguish configured options from defaults belonging to other modes.
///
/// # Errors
///
/// Fails when a value has no TOML form.
pub fn printed(config: &ServerArgs) -> anyhow::Result<String> {
    fn omit_defaults(table: &mut Table, defaults: &Table) {
        table.retain(|key, value| {
            let Some(default) = defaults.get(key) else {
                return true;
            };
            if value == default {
                return false;
            }
            if let (Value::Table(values), Value::Table(defaults)) = (value, default) {
                omit_defaults(values, defaults);
                return !values.is_empty();
            }
            true
        });
    }

    let mut table = Table::try_from(config).context("encoding the configuration")?;
    let defaults = Table::try_from(ServerArgs::default()).context("encoding the defaults")?;
    omit_defaults(&mut table, &defaults);
    toml::to_string_pretty(&table).context("printing the configuration")
}

/// `config` with the `--set` assignments `texts` applied over it, as
/// [`ServerArgs::resolve`] applies them over the file and the flags, and the
/// paths they assign. The API key is the one `config` holds.
///
/// # Errors
///
/// Fails as [`ServerArgs::resolve`] fails on an assignment: one that is
/// malformed or conflicts with another, a path that crosses an option that
/// is not a table, or a value its option does not take.
pub fn overridden(
    config: &ServerArgs,
    texts: &[String],
) -> anyhow::Result<(ServerArgs, Vec<String>)> {
    let mut tree = match Value::try_from(config).context("encoding the configuration")? {
        Value::Table(tree) => tree,
        _ => return Err(anyhow!("the configuration is not a table")),
    };
    let mut paths = Vec::new();
    for (path, value) in assignments(texts)? {
        paths.push(path.join("."));
        set(&mut tree, &path, value)?;
    }
    let mut overridden: ServerArgs = deserialize_options(tree)?;
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
    use std::path::PathBuf;
    use std::time::Duration;

    use clap::{FromArgMatches, Parser};
    use proto::request::{ExpertExchange, GraphPolicy, SchedulingPolicy};

    use super::*;

    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        serve: ServerArgs,
    }

    fn resolved(args: &[&str]) -> anyhow::Result<(ServerArgs, Stated)> {
        let command = <Command as clap::CommandFactory>::command();
        let matches =
            command.try_get_matches_from(std::iter::once("serve").chain(args.iter().copied()))?;
        Command::from_arg_matches(&matches)?
            .serve
            .resolve(&matches, Listen::Daemon)
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
            "quantization={\"mode\":\"balanced\"}",
            "--set",
            "moe_config={ expert_parallel = true, exchange = \"deepep\" }",
            "--set",
            "server_config.request_timeout=2m",
            "--set",
            "flashinfer_config.workspace_size=1GiB",
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
        assert_eq!(config.quantization["mode"], "balanced");
        assert!(config.moe_config.expert_parallel);
        assert_eq!(config.moe_config.exchange, ExpertExchange::DeepEp);
        assert_eq!(
            config.server_config.request_timeout,
            Some(Duration::from_secs(120))
        );
        assert_eq!(config.flashinfer_config.workspace_size, 1 << 30);
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

    /// The precision policy's retired name is refused with the name to state
    /// instead, whether it arrives as a table, as a path into one, or in the
    /// configuration file.
    #[test]
    fn the_retired_quantization_option_is_refused_by_name() {
        for args in [
            vec!["m", "--set", "quantization_config.mode=performance"],
            vec![
                "m",
                "--set",
                "quantization_config={ mode = \"performance\" }",
            ],
        ] {
            let error = resolved(&args).unwrap_err();
            let message = format!("{error:#}");
            assert!(
                message.contains("`quantization_config` is retired"),
                "{args:?}: {message}"
            );
            assert!(
                message.contains("`--set quantization.mode=...`"),
                "{args:?}: {message}"
            );
        }
        let error = resolve_text("[quantization_config]\nmode = \"performance\"\n", &[])
            .expect_err("the retired name is refused in a configuration file too");
        assert!(
            format!("{error:#}").contains("`quantization_config` is retired"),
            "{error:#}"
        );
    }

    /// The stated paths are what the invocation named, tables and options
    /// alike, and nothing it left at its default.
    #[test]
    fn stated_paths_are_what_was_named() {
        let (_, stated) =
            resolved(&["m", "--port", "9000", "--set", "video_config.max_seconds=5"]).unwrap();
        for path in [
            "model",
            "admin_config",
            "admin_config.port",
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
        for path in [
            "model",
            "admin_config.host",
            "admin_config.port",
            "served_model_name",
            "worker_groups",
            "tensor_parallel_size",
            "data_parallel_size",
            "max_running_requests",
            "max_model_len",
            "mem_fraction_static",
            "host_identity",
            "nnodes",
            "node_rank",
            "dist_init_addr",
            "worker_config.executable",
        ] {
            assert!(stated.contains(path), "{path} is set by its flag");
        }
        assert_eq!(
            config.worker_config.executable,
            PathBuf::from("/opt/env/bin/uniserve")
        );

        let text = printed(&config).unwrap();
        let reread: ServerArgs = toml::from_str(&text).unwrap();
        assert_eq!(reread, config, "{text}");
    }

    #[test]
    fn exported_options_remain_valid_for_their_model_family() {
        use models::ModelDescription::{MiniMaxH3, Qwen3};

        for (description, setting) in [
            (Qwen3, "chat_config.reasoning_parser=none"),
            (MiniMaxH3, "video_config.max_seconds=5"),
        ] {
            let (config, stated) = resolved(&["m", "--set", setting]).unwrap();
            config.validate(&stated).unwrap();
            stated.refuse_foreign_options(description).unwrap();
            let (reread, stated) = resolve_text(&printed(&config).unwrap(), &[]).unwrap();
            assert_eq!(reread, config);
            reread.validate(&stated).unwrap();
            stated.refuse_foreign_options(description).unwrap();
        }
    }

    /// A token bound stated above the deployment's own graph catalog is refused
    /// by the option's name, before any weight loads: the engine admits no batch
    /// larger than its largest captured bucket, so the scheduling policy's
    /// declared limits would be refused at the first prepare instead
    /// (`crates/scheduling/src/limits.rs`). A bound at the ceiling is admitted,
    /// and a stated value is never clamped to fit.
    #[test]
    fn a_token_bound_above_the_captured_graphs_is_refused_by_name() {
        let catalog = "cuda_graph_config.prefill_token_sizes=[512]";

        let (config, stated) =
            resolved(&["m", "--set", catalog, "--set", "max_num_batched_tokens=512"]).unwrap();
        config
            .validate(&stated)
            .expect("a bound at the catalog's ceiling is admitted");

        // An unstated budget is the catalog's own largest shape, so the
        // deployment needs to state nothing to be consistent.
        let (config, stated) = resolved(&["m", "--set", catalog]).unwrap();
        config
            .validate(&stated)
            .expect("an unstated budget follows the catalog");
        assert_eq!(config.cuda_graph_config.tiers().tokens, Some(512));

        let (config, stated) = resolved(&[
            "m",
            "--set",
            catalog,
            "--set",
            "max_num_batched_tokens=1024",
        ])
        .unwrap();
        let error = config
            .validate(&stated)
            .expect_err("a budget above the catalog is refused");
        let message = format!("{error:#}");
        assert!(
            message.contains("`max_num_batched_tokens` is 1024"),
            "{message}"
        );
        assert!(
            message.contains("512 tokens the deployment's captured graphs"),
            "{message}"
        );

        // The batch of requests is the same rule against the largest decode
        // graph: a deployment that leaves it unstated follows the catalog, and
        // one that states more than it captured is refused by name.
        let decodes = "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]";
        let (config, stated) = resolved(&["m", "--set", decodes]).unwrap();
        config
            .validate(&stated)
            .expect("an unstated batch follows the catalog");
        assert_eq!(config.cuda_graph_config.tiers().rows, Some(8));

        let (config, stated) = resolved(&[
            "m",
            "--set",
            decodes,
            "--set",
            "scheduler_config.max_batch=16",
        ])
        .unwrap();
        let error = config
            .validate(&stated)
            .expect_err("a batch above the captured decode graphs is refused");
        let message = format!("{error:#}");
        assert!(
            message.contains("`scheduler_config.max_batch` is 16"),
            "{message}"
        );
        assert!(
            message.contains("8 requests the deployment's captured decode graphs"),
            "{message}"
        );

        // The prefill chunk is the same bound and is refused the same way.
        let (config, stated) = resolved(&[
            "m",
            "--set",
            catalog,
            "--set",
            "scheduler_config.chunked_prefill_size=1024",
        ])
        .unwrap();
        let error = config
            .validate(&stated)
            .expect_err("a chunk above the catalog is refused");
        let message = format!("{error:#}");
        assert!(
            message.contains("`scheduler_config.chunked_prefill_size` is 1024"),
            "{message}"
        );
    }

    /// Resolves `args`, checks them, and resolves the engine configuration
    /// they state.
    fn lowered(args: &[&str]) -> anyhow::Result<(ServerArgs, proto::config::EngineConfig)> {
        let (config, stated) = resolved(args)?;
        config.validate(&stated)?;
        let engine = config.engine_config()?;
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
        assert_eq!(engine.model.model, "/models/m");

        let file = written("model", "model = \"/models/file\"\n");
        let (config, _) = lowered(&["--config", file.to_str().unwrap()]).unwrap();
        assert_eq!(config.model(), "/models/file");
    }

    /// Without a deployment file each scheduler's group runs one component
    /// tensor-parallel over its workers; the rank layout shorthand is refused
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
            "device_config.device=cpu",
        ])
        .unwrap();
        assert_eq!(engine.parallel.worker_groups.len(), 2);
        assert!(
            engine
                .parallel
                .worker_groups
                .iter()
                .all(|worker| worker.ranks.len() == 2)
        );

        let deployment = written(
            "deployment",
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cpu*2\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.model]\ntensor_parallel_size = 2\n",
        );
        let path = deployment.to_str().unwrap();
        let (config, engine) = lowered(&["m", "--worker-groups", path]).unwrap();
        assert_eq!(engine.parallel.worker_groups.len(), 1);
        let (reread, stated) = resolve_text(&printed(&config).unwrap(), &[]).unwrap();
        assert_eq!(reread, config);
        reread.validate(&stated).unwrap();
        for shorthand in [
            ["--tensor-parallel-size", "2"],
            ["--set", "device_config.device=cpu"],
            ["--set", "device_config.queue_depth=3"],
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
             [worker_groups.blocks.model]\ntensor_parallel_size = 2\n",
        );
        let (_, engine) = lowered(&[
            "m",
            "--worker-groups",
            deployment.to_str().unwrap(),
            "--host-identity",
            "rank-0",
        ])
        .unwrap();
        let hosts = engine.parallel.worker_groups[0]
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
        let capacities = engine.video.text_capacities.unwrap();
        assert_eq!(
            capacities
                .iter()
                .map(|value| value.get())
                .collect::<Vec<_>>(),
            vec![1024, 4096]
        );
        let shapes = engine.cuda_graph.flow_shapes.unwrap();
        assert_eq!(
            shapes.iter().map(|shape| shape.0).collect::<Vec<_>>(),
            vec![[1152, 2048], [2048, 1152]]
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
    /// unstated budgets stay unstated for the engine to default.
    #[test]
    fn grouped_options_reach_the_engine() {
        let (_, engine) = lowered(&[
            "m",
            "--set",
            "quantization.mode=performance",
            "--set",
            "moe_config={ expert_parallel = true, exchange = \"dwdp\" }",
            "--set",
            "kv_cache_config.prefix_caching=false",
            "--set",
            "scheduler_config.chunked_prefill_size=512",
        ])
        .unwrap();
        assert_eq!(engine.model.quantization["mode"], "performance");
        assert!(engine.moe.expert_parallel);
        assert_eq!(engine.moe.exchange, ExpertExchange::Dwdp);
        assert!(!engine.kv_cache.prefix_caching);
        assert_eq!(
            engine.scheduler.chunked_prefill_size,
            std::num::NonZeroUsize::new(512)
        );
        assert_eq!(engine.max_num_batched_tokens, None);
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
        use models::ModelDescription::{MiniMaxH3, Qwen3};

        let (_, defaults) = resolved(&["m"]).unwrap();
        for description in [Qwen3, MiniMaxH3] {
            assert!(defaults.refuse_foreign_options(description).is_ok());
        }

        let refused = |set: &str, description| {
            let (_, stated) = resolved(&["m", "--set", set]).unwrap();
            stated.refuse_foreign_options(description).is_err()
        };
        assert!(refused("video_config.max_seconds=5", Qwen3));
        assert!(!refused("video_config.max_seconds=5", MiniMaxH3));
        assert!(refused("chat_config.reasoning_parser=none", MiniMaxH3));
        assert!(!refused("chat_config.reasoning_parser=none", Qwen3));
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
