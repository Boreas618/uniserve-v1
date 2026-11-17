//! The command line that resolves a [`ServerArgs`].
//!
//! Options are read from three places, each overriding the last: the TOML
//! file `--config` names, the flags that name the most common options (each
//! the option of its own name, declared on the option's field of
//! [`ServerArgs`]), and repeatable
//! `--set PATH=VALUE` assignments that reach any option by its path, such as
//! `--set video_config.max_seconds=10`. The three are merged as TOML values
//! and deserialized once, so every option is checked by the same type
//! whichever place states it, and an error names the option's path.

use std::collections::BTreeSet;

use anyhow::{Context as _, anyhow};
use toml::{Table, Value};

use super::ServerArgs;

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
    /// Resolves the configuration this command line states: the `--config` files, each
    /// overriding the ones before it, overridden by the flags the command line states,
    /// overridden by `--set`, with every unstated option at its default. `--host` and `--port`
    /// state the daemon's own address (`admin_config`), and `--source` the source tree of the
    /// host's Build Server (`build_config.source`), made absolute. Returns it with the paths the
    /// invocation stated.
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
    pub fn resolve(&self, matches: &clap::ArgMatches) -> anyhow::Result<(ServerArgs, Stated)> {
        let mut tree = Table::new();
        for path in &self.config_files {
            let file = std::fs::read_to_string(path)
                .with_context(|| format!("reading --config {}", path.display()))?
                .parse::<Table>()
                .with_context(|| format!("parsing --config {}", path.display()))?;
            merge(&mut tree, file);
        }
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
        let path = |table: &str, key: &str| vec![table.to_owned(), key.to_owned()];
        if let Some(value) = &self.host {
            set(
                &mut tree,
                &path("admin_config", "host"),
                Value::String(value.clone()),
            )?;
        }
        if let Some(value) = self.port {
            set(
                &mut tree,
                &path("admin_config", "port"),
                Value::Integer(i64::from(value)),
            )?;
        }
        if let Some(value) = &self.source {
            // Absolute, so the Build Server the daemon starts, and a daemon that succeeds this one,
            // read the tree this command line was started in.
            let source = std::path::absolute(value)
                .with_context(|| format!("resolving --source {}", value.display()))?;
            set(
                &mut tree,
                &path("build_config", "source"),
                Value::String(source.to_string_lossy().into_owned()),
            )?;
        }
        for (path, value) in assignments(&self.set)? {
            set(&mut tree, &path, value)?;
        }
        configured(tree)
    }
}

/// Merges the configuration file `over` into `tree`: a table merges key by key, and any other
/// value of `over`, an array of tables such as `worker_groups` included, replaces the value of
/// `tree` at its path.
fn merge(tree: &mut Table, over: Table) {
    for (key, value) in over {
        match (tree.get_mut(&key), value) {
            (Some(Value::Table(table)), Value::Table(over)) => merge(table, over),
            (_, value) => {
                tree.insert(key, value);
            }
        }
    }
}

/// A TOML tree of options as `T`; a key no option takes is refused by the option types, by its
/// path.
fn deserialize_options<T: serde::de::DeserializeOwned>(tree: Table) -> anyhow::Result<T> {
    serde_path_to_error::deserialize(Value::Table(tree)).map_err(|error| {
        let path = error.path().to_string();
        let cause = error.into_inner();
        if path == "." {
            anyhow!("invalid configuration: {cause}")
        } else {
            anyhow!("invalid option `{path}`: {cause}")
        }
    })
}

/// The configuration of a TOML tree of options, with the paths it states.
fn configured(tree: Table) -> anyhow::Result<(ServerArgs, Stated)> {
    let stated = Stated::of(&tree);
    let config: ServerArgs = deserialize_options(tree)?;
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

/// Reads a `--set` value: one TOML value, written as the configuration file writes it. A text
/// value is quoted (`"deepep"`); a number, boolean, array or inline table is not.
///
/// # Errors
///
/// Fails when the text is not exactly one TOML value, naming the quoted form an unquoted word
/// takes.
fn parsed_value(text: &str) -> anyhow::Result<Value> {
    // `toml::Value` parses one value expression: a text holding a second value or a key of its
    // own is refused rather than read in part.
    let text = text.trim();
    text.parse::<Value>().map_err(|error| {
        anyhow!(
            "the value is not one TOML value ({}); a text value is quoted, as in `PATH=\"{text}\"`",
            error.message().trim_end()
        )
    })
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
    use proto::config::{PrecisionMode, WeightFormat};

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
        Command::from_arg_matches(&matches)?.serve.resolve(&matches)
    }

    /// A scratch configuration file holding `body`, removed when the path drops.
    fn written(body: &str) -> tempfile::TempPath {
        let mut file = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("creating a scratch file");
        std::io::Write::write_all(&mut file, body.as_bytes()).expect("writing a scratch file");
        file.into_temp_path()
    }

    /// The file states options by path, a flag overrides the file, and
    /// `--set` overrides both; everything unstated keeps its default.
    #[test]
    fn set_overrides_flags_which_override_the_file() {
        let file = written(
            "model = \"org/file-model\"\nmax_running_requests = 4\n\
             [scheduler_config]\npolicy = \"priority\"\n",
        );
        let file = file.to_str().unwrap();

        let (config, _) = resolved(&["--config", file]).unwrap();
        assert_eq!(config.model.as_deref(), Some("org/file-model"));
        assert_eq!(config.max_running_requests.get(), 4);

        let (config, _) = resolved(&[
            "org/flag-model",
            "--config",
            file,
            "--max-running-requests",
            "2",
            "--set",
            "max_running_requests=1",
            "--set",
            "scheduler_config.policy=\"fcfs\"",
        ])
        .unwrap();
        assert_eq!(config.model.as_deref(), Some("org/flag-model"));
        assert_eq!(config.max_running_requests.get(), 1);
        assert_eq!(config.scheduler_config.policy, "fcfs");
        assert_eq!(config.admin_config.port, 0);
    }

    /// Several `--config` files merge in order: a later file's option replaces an earlier one's,
    /// tables merge key by key, and a later `[[worker_groups]]` list replaces the earlier list.
    #[test]
    fn later_configuration_files_override_earlier_ones() {
        let layout = written(
            "max_running_requests = 4\n[scheduler_config]\npolicy = \"priority\"\nmax_batch = 8\n\
             [[worker_groups]]\nid = \"first\"\nranks = [\"cpu\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.model]\n",
        );
        let overrides = written(
            "max_running_requests = 2\n[scheduler_config]\npolicy = \"fcfs\"\n\
             [[worker_groups]]\nid = \"second\"\nranks = [\"cpu\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.model]\n",
        );
        let (config, stated) = resolved(&[
            "m",
            "--config",
            layout.to_str().unwrap(),
            "--config",
            overrides.to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(config.max_running_requests.get(), 2);
        assert_eq!(config.scheduler_config.policy, "fcfs");
        assert_eq!(config.scheduler_config.max_batch.get(), 8);
        let groups = config.worker_groups.as_deref().unwrap_or_default();
        assert_eq!(
            groups
                .iter()
                .map(|group| group.id.0.as_str())
                .collect::<Vec<_>>(),
            ["second"]
        );
        assert!(stated.contains("worker_groups"));
    }

    /// A `--set` value is one TOML value, as the configuration file writes it: a list is an
    /// array, a table an inline table, and text, durations and sizes are quoted.
    #[test]
    fn set_values_read_as_toml() {
        let (config, _) = resolved(&[
            "--set",
            "video_config.text_capacities=[1024, 10240]",
            "--set",
            "video_config.sizes=[\"1344x768\"]",
            "--set",
            "quantization={ mode = \"balanced\", modules = { video_vae = \"bf16\" } }",
            "--set",
            "moe_config={ expert_parallel = true, exchange = \"deepep\" }",
            "--set",
            "server_config.request_timeout=\"2m\"",
            "--set",
            "admin_config.host=\"0.0.0.0\"",
        ])
        .unwrap();
        assert_eq!(
            config
                .video_config
                .text_capacities
                .map(|sizes| sizes.iter().map(|size| size.get()).collect::<Vec<_>>()),
            Some(vec![1024, 10240])
        );
        assert_eq!(config.video_config.sizes, Some(vec!["1344x768".to_owned()]));
        assert_eq!(config.quantization.mode, Some(PrecisionMode::Balanced));
        assert_eq!(
            config.quantization.modules,
            [("video_vae".to_owned(), WeightFormat::Bf16)].into()
        );
        assert!(config.moe_config.expert_parallel);
        assert_eq!(config.moe_config.exchange.as_deref(), Some("deepep"));
        assert_eq!(
            config.server_config.request_timeout,
            Some(Duration::from_secs(120))
        );
        assert_eq!(config.admin_config.host, "0.0.0.0");
    }

    /// A value that is not one TOML value is refused, naming its quoted form, rather than read
    /// as JSON or as text: an unquoted word, JSON's object syntax and two values.
    #[test]
    fn a_value_that_is_not_one_toml_value_is_refused() {
        for (value, named) in [
            ("dtype=bfloat16", "PATH=\"bfloat16\""),
            ("quantization={\"mode\":\"balanced\"}", "not one TOML value"),
            ("max_running_requests=1 2", "not one TOML value"),
            ("max_running_requests=", "not one TOML value"),
        ] {
            let error = resolved(&["--set", value]).unwrap_err();
            assert!(format!("{error:#}").contains(named), "{value}: {error:#}");
        }
    }

    /// An unknown option, a value its option does not take, a path through
    /// an option, and a table set beside an option inside it are refused,
    /// naming the path.
    #[test]
    fn malformed_options_are_refused_by_path() {
        for (args, named) in [
            (
                vec!["--set", "cuda_graph_config.decode_batch_szes=[1]"],
                "decode_batch_szes",
            ),
            (
                vec!["--set", "max_running_requests=0"],
                "max_running_requests",
            ),
            (
                vec!["--set", "scheduler_config.policy=[\"fcfs\"]"],
                "scheduler_config.policy",
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
            (
                vec!["--set", "quantization.mode=\"fastest\""],
                "quantization.mode",
            ),
            (
                vec!["--set", "quantization.modules.mlp=\"int4\""],
                "quantization.modules.mlp",
            ),
            (
                vec!["--set", "quantization.components={ mlp = \"fp8\" }"],
                "quantization.components",
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
            "admin_config",
            "admin_config.port",
            "video_config.max_seconds",
        ] {
            assert!(stated.contains(path), "{path} was stated");
        }
        assert!(stated.contains_within("video_config"));
        assert!(!stated.contains("admin_config.host"));
        assert!(!stated.contains_within("cuda_graph_config"));
    }

    /// Every flag is a shortcut for the path it names, and the printed
    /// configuration reads back as the same configuration.
    #[test]
    fn flags_set_their_paths_and_printing_round_trips() {
        let (config, stated) = resolved(&[
            "org/model",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--served-model-name",
            "served",
            "--tensor-parallel-size",
            "2",
            "--data-parallel-size",
            "2",
            "--max-running-requests",
            "4",
            "--max-model-len",
            "4096",
            "--lora",
            "org/adapter",
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
            "--source",
            "/srv/uniserve",
            "--set",
            "quantization={ mode = \"quality\", modules = { mlp = \"bf16\" } }",
        ])
        .unwrap();
        assert_eq!(
            config.build_config.source.as_deref(),
            Some(std::path::Path::new("/srv/uniserve"))
        );
        for path in [
            "model",
            "admin_config.host",
            "admin_config.port",
            "served_model_name",
            "tensor_parallel_size",
            "data_parallel_size",
            "max_running_requests",
            "max_model_len",
            "lora",
            "mem_fraction_static",
            "host_identity",
            "nnodes",
            "node_rank",
            "dist_init_addr",
            "build_config.source",
        ] {
            assert!(stated.contains(path), "{path} is set by its flag");
        }

        let text = printed(&config).unwrap();
        let reread: ServerArgs = toml::from_str(&text).unwrap();
        assert_eq!(reread, config, "{text}");
    }

    #[test]
    fn exported_options_remain_valid_for_their_model_family() {
        use crate::test_support::{CHAT, VIDEO};

        for (description, setting) in [
            (&CHAT, "chat_config.reasoning_parser=\"none\""),
            (&VIDEO, "video_config.max_seconds=5"),
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
        let (config, engine) = lowered(&["org/model"]).unwrap();
        assert_eq!(config.model(), "org/model");
        assert_eq!(config.served_model_name(), "org/model");
        assert_eq!(engine.model.model, "org/model");

        let file = written("model = \"org/file-model\"\n");
        let (config, _) = lowered(&["--config", file.to_str().unwrap()]).unwrap();
        assert_eq!(config.model(), "org/file-model");
    }

    /// Without `worker_groups` each scheduler's group runs one block
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
            "device_config.device=\"cpu\"",
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
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cpu*2\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.model]\ntensor_parallel_size = 2\n",
        );
        let path = deployment.to_str().unwrap();
        let (config, engine) = lowered(&["m", "--config", path]).unwrap();
        assert_eq!(engine.parallel.worker_groups.len(), 1);
        let (reread, stated) = resolve_text(&printed(&config).unwrap(), &[]).unwrap();
        assert_eq!(reread, config);
        reread.validate(&stated).unwrap();
        for shorthand in [
            ["--tensor-parallel-size", "2"],
            ["--set", "device_config.device=\"cpu\""],
            ["--set", "device_config.queue_depth=3"],
        ] {
            let mut args = vec!["m", "--config", path];
            args.extend(shorthand);
            let error = lowered(&args).unwrap_err();
            assert!(
                error.to_string().contains("deployment layout"),
                "{shorthand:?}: {error}"
            );
        }
    }

    /// A deployment rank entry without a host runs on this process's host.
    #[test]
    fn a_deployment_places_unnamed_ranks_on_this_host() {
        let deployment = written(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cpu\", \"rank-1/cpu\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.model]\ntensor_parallel_size = 2\n",
        );
        let (_, engine) = lowered(&[
            "m",
            "--config",
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
        let (config, _) = resolved(&["m", "--set", "worker_groups=[]"]).unwrap();
        assert!(
            config.engine_config().is_err(),
            "a layout names at least one group"
        );
    }

    /// Capture lists and capacities are checked before any rank starts.
    #[test]
    fn capture_lists_are_strictly_increasing() {
        let (_, engine) =
            lowered(&["m", "--set", "video_config.text_capacities=[1024, 4096]"]).unwrap();
        let capacities = engine.video.text_capacities.unwrap();
        assert_eq!(
            capacities
                .iter()
                .map(|value| value.get())
                .collect::<Vec<_>>(),
            vec![1024, 4096]
        );

        for refused in [
            "video_config.text_capacities=[4096, 1024]",
            "cuda_graph_config.decode_batch_sizes=[1, 0]",
            // A list has one form, the array: comma-separated text and a bare element are
            // refused.
            "video_config.text_capacities=1024,4096",
            "video_config.text_capacities=1024",
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
            "quantization.mode=\"performance\"",
            "--set",
            "moe_config={ expert_parallel = true, exchange = \"deepep\" }",
            "--set",
            "kv_cache_config.prefix_caching=false",
            "--set",
            "scheduler_config.chunked_prefill_size=512",
        ])
        .unwrap();
        assert_eq!(
            engine.model.quantization,
            proto::config::Quantization {
                mode: Some(PrecisionMode::Performance),
                modules: Default::default(),
            }
        );
        assert!(engine.moe.expert_parallel);
        assert_eq!(engine.moe.exchange.as_deref(), Some("deepep"));
        assert!(!engine.kv_cache.prefix_caching);
        assert_eq!(
            engine.scheduler.chunked_prefill_size,
            std::num::NonZeroUsize::new(512)
        );
        assert_eq!(engine.max_num_batched_tokens, None);
    }

    /// The server listens on a non-blank host or Unix socket path, port 0
    /// asks for an ephemeral port, and a video capacity is a positive
    /// duration; the model holds it to its own bounds when it binds.
    #[test]
    fn listeners_and_video_capacities_are_checked() {
        assert!(lowered(&["m", "--port", "0"]).is_ok());
        assert!(lowered(&["m", "--host", " "]).is_err());
        assert!(lowered(&["m", "--set", "admin_config.uds=\"\""]).is_err());
        assert!(lowered(&["m", "--set", "server_config.api_key=\"\""]).is_err());
        assert!(lowered(&["m", "--set", "server_config.api_key=\"secret\""]).is_ok());
        for (seconds, valid) in [("4", true), ("15.5", true), ("0", false), ("-1", false)] {
            let set = format!("video_config.max_seconds={seconds}");
            assert_eq!(lowered(&["m", "--set", &set]).is_ok(), valid, "{seconds} s");
        }
    }

    /// An option of another kind of model than the served one is refused
    /// rather than ignored; unstated defaults never are.
    #[test]
    fn options_of_another_family_are_refused() {
        use crate::test_support::{CHAT as Chat, VIDEO as Video};

        let (_, defaults) = resolved(&["m"]).unwrap();
        for description in [&Chat, &Video] {
            assert!(defaults.refuse_foreign_options(description).is_ok());
        }

        let refused = |set: &str, description| {
            let (_, stated) = resolved(&["m", "--set", set]).unwrap();
            stated.refuse_foreign_options(description).is_err()
        };
        assert!(refused("video_config.max_seconds=5", &Chat));
        assert!(!refused("video_config.max_seconds=5", &Video));
        assert!(refused("chat_config.reasoning_parser=\"none\"", &Video));
        assert!(!refused("chat_config.reasoning_parser=\"none\"", &Chat));
        // A video pipeline schedules no token batches and renders no chat.
        for token_option in [
            "max_num_batched_tokens=2048",
            "scheduler_config.chunked_prefill_size=512",
            "chat_config.reasoning_parser=\"none\"",
        ] {
            assert!(refused(token_option, &Video), "{token_option}");
            assert!(!refused(token_option, &Chat), "{token_option}");
        }
    }

    /// A deployment states the settings of its Storage Servers: its `storage_config`, with the
    /// Mooncake connector `kv_cache_config.mooncake` states, and a TCP endpoint on every
    /// interface when other nodes may join it and it states none.
    #[test]
    fn a_deployment_states_its_storage_servers_settings() {
        let (plain, _) = resolved(&["m"]).unwrap();
        assert_eq!(plain.storage_settings(), storage::Settings::default());

        let (config, _) = resolved(&[
            "m",
            "--dist-init-addr",
            "10.0.0.2:5000",
            "--set",
            "kv_cache_config.mooncake.master=\"10.0.0.1:50051\"",
            "--set",
            "kv_cache_config.mooncake.segment_gib=4",
            "--set",
            "storage_config.host_reserve_gib=4",
        ])
        .unwrap();
        let settings = config.storage_settings();
        let connector = settings.connector.unwrap();
        assert_eq!(connector.master.to_string(), "10.0.0.1:50051");
        assert_eq!(connector.segment_bytes(), 4 << 30);
        assert_eq!(settings.server.host_reserve_gib, 4);
        assert_eq!(
            settings.server.tcp.map(|tcp| tcp.to_string()).as_deref(),
            Some("0.0.0.0:0")
        );

        // A library location is no setting: Mooncake is opened from the loader's search path.
        assert!(
            resolved(&[
                "m",
                "--set",
                "kv_cache_config.mooncake={ master = \"10.0.0.1:50051\", library = \"/x.so\" }",
            ])
            .is_err()
        );
    }
}
