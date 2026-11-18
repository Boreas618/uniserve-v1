//! A policy package's `config.toml` as a [`Config`]: parsing it, the option paths it states, and
//! printing a configuration back as the TOML a package carries.
//!
//! The file is deserialized once, so every option is checked by its own type and an error names
//! the option's path. An option the file does not state takes its default.

use std::collections::BTreeSet;

use anyhow::{Context as _, anyhow};
use toml::{Table, Value};

use super::Config;

/// The configuration paths a package's `config.toml` states: every table and option it names,
/// rather than leaves at its default.
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

    /// The stated paths, as a hot restart hands them to the successor.
    pub fn paths(&self) -> Vec<String> {
        self.0.iter().cloned().collect()
    }

    /// The statement of `paths`, as [`Stated::paths`] gave them.
    pub fn from_paths(paths: Vec<String>) -> Self {
        Self(paths.into_iter().collect())
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

/// The configuration `text`, a package's `config.toml`, states, with the paths it states.
///
/// # Errors
///
/// Fails when `text` is not TOML, or names an unknown option or a value its option does not
/// take; the message names the option's path.
pub fn parse(text: &str) -> anyhow::Result<(Config, Stated)> {
    let tree = text
        .parse::<Table>()
        .context("parsing the package's config.toml")?;
    let stated = Stated::of(&tree);
    let config: Config = deserialize_options(tree)?;
    Ok((config, stated))
}

/// The configuration as the `config.toml` of a package: its non-default options, which read back
/// as the same configuration. Defaults stay unstated, so the options of another model family and
/// of another rank layout are not read as stated ([`Stated`]).
///
/// # Errors
///
/// Fails when a value has no TOML form.
pub fn printed(config: &Config) -> anyhow::Result<String> {
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
    let defaults = Table::try_from(Config::default()).context("encoding the defaults")?;
    omit_defaults(&mut table, &defaults);
    toml::to_string_pretty(&table).context("printing the configuration")
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use proto::config::{PrecisionMode, WeightFormat};

    use super::*;

    /// `config.toml` text: a model, then `body`.
    fn package(body: &str) -> String {
        format!("model = \"m\"\n{body}")
    }

    /// Parses and checks `text`, and resolves the engine configuration it states.
    fn lowered(text: &str) -> anyhow::Result<(Config, proto::config::EngineConfig)> {
        let (config, stated) = parse(text)?;
        config.validate(&stated)?;
        let engine = config.engine_config()?;
        Ok((config, engine))
    }

    /// Every option is read by its own type: lists are arrays, tables nest, and durations and
    /// sizes are text with units; everything unstated keeps its default.
    #[test]
    fn options_read_by_their_types() {
        let (config, _) = parse(&package(
            "[video_config]\ntext_capacities = [1024, 10240]\nsizes = [\"1344x768\"]\n\
             [quantization]\nmode = \"balanced\"\nmodules = { video_vae = \"bf16\" }\n\
             [moe_config]\nexpert_parallel = true\n\
             [kernels.expert_exchange]\n\"mode_dispatch_ranks_2\" = \"deepep@3\"\n\
             [server_config]\nrequest_timeout = \"2m\"\n\
             [admin_config]\nhost = \"0.0.0.0\"\n",
        ))
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
        assert_eq!(
            config.kernels["expert_exchange"]["mode_dispatch_ranks_2"],
            proto::module::Pin {
                backend: "deepep".to_owned(),
                revision: Some(3),
            }
        );
        assert_eq!(
            config.server_config.request_timeout,
            Some(Duration::from_secs(120))
        );
        assert_eq!(config.admin_config.host, "0.0.0.0");
        assert_eq!(config.admin_config.port, 0);
        assert_eq!(config.max_running_requests.get(), 128);
    }

    /// An unknown option and a value its option does not take are refused, naming the path.
    #[test]
    fn malformed_options_are_refused_by_path() {
        for (text, named) in [
            (
                "[cuda_graph_config]\ndecode_batch_szes = [1]\n",
                "decode_batch_szes",
            ),
            ("max_running_requests = 0\n", "max_running_requests"),
            (
                "[programs.chunk]\nsource = \"chunk\"\n",
                "programs.chunk.source",
            ),
            ("max_batch = 4\n", "max_batch"),
            ("[quantization]\nmode = \"fastest\"\n", "quantization.mode"),
            (
                "[quantization.modules]\nmlp = \"int4\"\n",
                "quantization.modules.mlp",
            ),
            ("host_identity = \"rank-0\"\n", "host_identity"),
            ("model = \n", "config.toml"),
        ] {
            let error = parse(text).unwrap_err();
            assert!(format!("{error:#}").contains(named), "{text}: {error:#}");
        }
    }

    /// The stated paths are what the file named, tables and options alike, and nothing it left
    /// at its default.
    #[test]
    fn stated_paths_are_what_the_file_names() {
        let (_, stated) = parse(&package(
            "[admin_config]\nport = 9000\n[video_config]\nmax_seconds = 5\n",
        ))
        .unwrap();
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

    /// A printed configuration reads back as the same configuration, and stays valid for the
    /// model family it configures.
    #[test]
    fn printing_round_trips() {
        use crate::test_support::{CHAT, VIDEO};

        let (config, _) = parse(&package(
            "served_model_name = \"served\"\nhosts = [\"rank-0\", \"rank-1\"]\n\
             tensor_parallel_size = 2\ndata_parallel_size = 2\nmax_model_len = 4096\n\
             mem_fraction_static = 0.5\n[build_config]\nsource = \"/srv/uniserve\"\n\
             [quantization]\nmode = \"quality\"\nmodules = { mlp = \"bf16\" }\n",
        ))
        .unwrap();
        let text = printed(&config).unwrap();
        let (reread, _) = parse(&text).unwrap();
        assert_eq!(reread, config, "{text}");

        for (description, body) in [
            (&CHAT, "[chat_config]\nreasoning_parser = \"none\"\n"),
            (&VIDEO, "[video_config]\nmax_seconds = 5\n"),
        ] {
            let (config, stated) = parse(&package(body)).unwrap();
            config.validate(&stated).unwrap();
            stated.refuse_foreign_options(description).unwrap();
            let (reread, stated) = parse(&printed(&config).unwrap()).unwrap();
            assert_eq!(reread, config);
            reread.validate(&stated).unwrap();
            stated.refuse_foreign_options(description).unwrap();
        }
    }

    /// A token bound stated above the deployment's own graph catalog is refused by the option's
    /// name, before any weight loads: the engine admits no batch larger than its largest
    /// captured bucket. A bound at the ceiling is admitted, and a stated value is never clamped
    /// to fit.
    #[test]
    fn a_token_bound_above_the_captured_graphs_is_refused_by_name() {
        let catalog = "[cuda_graph_config]\nprefill_token_sizes = [512]\n";
        let refusal = |text: String| {
            let (config, stated) = parse(&text).unwrap();
            format!("{:#}", config.validate(&stated).unwrap_err())
        };

        let (config, stated) = parse(&package(&format!(
            "max_num_batched_tokens = 512\n{catalog}"
        )))
        .unwrap();
        config
            .validate(&stated)
            .expect("a bound at the catalog's ceiling is admitted");
        let (config, stated) = parse(&package(catalog)).unwrap();
        config
            .validate(&stated)
            .expect("an unstated budget follows the catalog");
        assert_eq!(config.cuda_graph_config.tiers().tokens, Some(512));

        let message = refusal(package(&format!(
            "max_num_batched_tokens = 1024\n{catalog}"
        )));
        assert!(
            message.contains("`max_num_batched_tokens` is 1024"),
            "{message}"
        );
        assert!(
            message.contains("512 tokens the deployment's captured graphs"),
            "{message}"
        );

        // The batch of requests is the same rule against the largest decode graph.
        let decodes = "[cuda_graph_config]\ndecode_batch_sizes = [1, 2, 4, 8]\n";
        let (config, stated) = parse(&package(decodes)).unwrap();
        config
            .validate(&stated)
            .expect("an unstated batch follows the catalog");
        assert_eq!(config.cuda_graph_config.tiers().rows, Some(8));
        let message = refusal(package(&format!(
            "{decodes}[scheduler_config]\nmax_batch = 16\n"
        )));
        assert!(
            message.contains("`scheduler_config.max_batch` is 16"),
            "{message}"
        );

        // The prefill chunk is the same bound and is refused the same way.
        let message = refusal(package(&format!(
            "{catalog}[scheduler_config]\nchunked_prefill_size = 1024\n"
        )));
        assert!(
            message.contains("`scheduler_config.chunked_prefill_size` is 1024"),
            "{message}"
        );
    }

    /// A package names its model; one without is refused.
    #[test]
    fn a_package_names_its_model() {
        let (config, stated) = parse("").unwrap();
        let error = config.validate(&stated).unwrap_err();
        assert!(error.to_string().contains("names no model"), "{error}");
        let (config, engine) = lowered("model = \"org/model\"\n").unwrap();
        assert_eq!(config.model(), "org/model");
        assert_eq!(config.served_model_name(), "org/model");
        assert_eq!(engine.model.model, "org/model");
    }

    /// Without `worker_groups` each scheduler's group runs one block tensor-parallel over its
    /// workers; the rank layout shorthand is refused beside a deployment layout.
    #[test]
    fn the_shorthand_places_engine_cores_and_is_refused_beside_a_layout() {
        let (_, engine) = lowered(&package(
            "tensor_parallel_size = 2\ndata_parallel_size = 2\n[device_config]\ndevice = \"cpu\"\n",
        ))
        .unwrap();
        assert_eq!(engine.parallel.worker_groups.len(), 2);
        assert!(
            engine
                .parallel
                .worker_groups
                .iter()
                .all(|worker| worker.ranks.len() == 2)
        );

        let layout = "[[worker_groups]]\nid = \"model\"\nranks = [\"cpu*2\"]\nqueue_depth = 2\n\
                      [worker_groups.blocks.model]\ntensor_parallel_size = 2\n";
        let (config, engine) = lowered(&package(layout)).unwrap();
        assert_eq!(engine.parallel.worker_groups.len(), 1);
        let (reread, stated) = parse(&printed(&config).unwrap()).unwrap();
        assert_eq!(reread, config);
        reread.validate(&stated).unwrap();
        for shorthand in [
            "tensor_parallel_size = 2\n",
            "[device_config]\ndevice = \"cpu\"\n",
            "[device_config]\nqueue_depth = 3\n",
        ] {
            let error = lowered(&package(&format!("{shorthand}{layout}"))).unwrap_err();
            assert!(
                error.to_string().contains("deployment layout"),
                "{shorthand}: {error}"
            );
        }
    }

    /// A deployment rank entry without a host runs on the leader's host, the first of `hosts`.
    #[test]
    fn a_deployment_places_unnamed_ranks_on_the_leaders_host() {
        let (_, engine) = lowered(&package(
            "hosts = [\"rank-0\", \"rank-1\"]\n\
             [[worker_groups]]\nid = \"model\"\nranks = [\"cpu\", \"rank-1/cpu\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.model]\ntensor_parallel_size = 2\n",
        ))
        .unwrap();
        let hosts = engine.parallel.worker_groups[0]
            .ranks
            .iter()
            .map(|rank| rank.host.as_str())
            .collect::<Vec<_>>();
        assert_eq!(hosts, ["rank-0", "rank-1"]);
        let (config, _) = parse(&package("worker_groups = []\n")).unwrap();
        assert!(
            config.engine_config().is_err(),
            "a layout names at least one group"
        );
    }

    /// Capture lists and capacities are checked before any rank starts.
    #[test]
    fn capture_lists_are_strictly_increasing() {
        let (_, engine) =
            lowered(&package("[video_config]\ntext_capacities = [1024, 4096]\n")).unwrap();
        let capacities = engine.video.text_capacities.unwrap();
        assert_eq!(
            capacities
                .iter()
                .map(|value| value.get())
                .collect::<Vec<_>>(),
            vec![1024, 4096]
        );
        for refused in [
            "[video_config]\ntext_capacities = [4096, 1024]\n",
            "[cuda_graph_config]\ndecode_batch_sizes = [1, 0]\n",
            "[video_config]\ntext_capacities = \"1024,4096\"\n",
            "[video_config]\ntext_capacities = 1024\n",
        ] {
            assert!(lowered(&package(refused)).is_err(), "{refused} is refused");
        }
    }

    /// A change of the deployment groups' queue depths alone takes effect at the next step, and
    /// reaches the engines' step values by group; any other change of the groups is an epoch's.
    #[test]
    fn a_groups_queue_depth_takes_effect_at_the_next_step() {
        let groups = |depth: usize, ranks: &str| {
            package(&format!(
                "[[worker_groups]]\nid = \"model\"\nranks = [\"{ranks}\"]\nqueue_depth = {depth}\n\
                 [worker_groups.blocks.model]\ntensor_parallel_size = 2\n"
            ))
        };
        let (served, _) = lowered(&groups(2, "cuda:0-1")).unwrap();
        let (deeper, engine) = lowered(&groups(4, "cuda:0-1")).unwrap();
        let (moved, _) = lowered(&groups(2, "cuda:2-3")).unwrap();
        let effect = |target: &Config| {
            proto::reload::Effects::of(&proto::reload::changes(Some(&served), target)).heaviest()
        };
        assert_eq!(effect(&deeper), Some(proto::reload::Effect::Step));
        assert_eq!(effect(&moved), Some(proto::reload::Effect::Epoch));
        let tuning = proto::config::Tuning::of(&engine, 3);
        assert_eq!(tuning.queue_depths["model"].get(), 4);
    }

    /// The precision policy, expert parallelism, prefix caching and the scheduler budgets reach
    /// the engine configuration as stated, and unstated budgets stay unstated for the engine to
    /// default.
    #[test]
    fn grouped_options_reach_the_engine() {
        let (_, engine) = lowered(&package(
            "[quantization]\nmode = \"performance\"\n\
             [moe_config]\nexpert_parallel = true\n\
             [kv_cache_config]\nprefix_caching = false\n\
             [scheduler_config]\nchunked_prefill_size = 512\n",
        ))
        .unwrap();
        assert_eq!(
            engine.model.quantization,
            proto::config::Quantization {
                mode: Some(PrecisionMode::Performance),
                modules: Default::default(),
            }
        );
        assert!(engine.moe.expert_parallel);
        assert!(!engine.kv_cache.prefix_caching);
        assert_eq!(
            engine.scheduler.chunked_prefill_size,
            std::num::NonZeroUsize::new(512)
        );
        assert_eq!(engine.max_num_batched_tokens, None);
    }

    /// The server listens on a non-blank host or Unix socket path, port 0 asks for an
    /// ephemeral port, a deployment of several nodes names its join port, and a video capacity
    /// is a positive duration.
    #[test]
    fn listeners_and_video_capacities_are_checked() {
        assert!(lowered(&package("[admin_config]\nport = 0\n")).is_ok());
        assert!(lowered(&package("[admin_config]\nhost = \" \"\n")).is_err());
        assert!(lowered(&package("[admin_config]\nuds = \"\"\n")).is_err());
        assert!(lowered(&package("[server_config]\napi_key = \"\"\n")).is_err());
        assert!(lowered(&package("[server_config]\napi_key = \"secret\"\n")).is_ok());
        assert!(
            lowered(&package(
                "hosts = [\"a\", \"b\"]\n[admin_config]\njoin = 0\n"
            ))
            .is_err()
        );
        assert!(lowered(&package("hosts = [\"a\", \"a\"]\n")).is_err());
        for (seconds, valid) in [("4", true), ("15.5", true), ("0", false), ("-1", false)] {
            let body = format!("[video_config]\nmax_seconds = {seconds}\n");
            assert_eq!(lowered(&package(&body)).is_ok(), valid, "{seconds} s");
        }
    }

    /// An option of another kind of model than the served one is refused rather than ignored;
    /// unstated defaults never are.
    #[test]
    fn options_of_another_family_are_refused() {
        use crate::test_support::{CHAT as Chat, VIDEO as Video};

        let (_, defaults) = parse(&package("")).unwrap();
        for description in [&Chat, &Video] {
            assert!(defaults.refuse_foreign_options(description).is_ok());
        }
        let refused = |body: &str, description| {
            let (_, stated) = parse(&package(body)).unwrap();
            stated.refuse_foreign_options(description).is_err()
        };
        let video = "[video_config]\nmax_seconds = 5\n";
        let chat = "[chat_config]\nreasoning_parser = \"none\"\n";
        assert!(refused(video, &Chat));
        assert!(!refused(video, &Video));
        assert!(refused(chat, &Video));
        assert!(!refused(chat, &Chat));
        // A video pipeline schedules no token batches and renders no chat.
        for token_option in [
            "max_num_batched_tokens = 2048\n",
            "[scheduler_config]\nchunked_prefill_size = 512\n",
            chat,
        ] {
            assert!(refused(token_option, &Video), "{token_option}");
            assert!(!refused(token_option, &Chat), "{token_option}");
        }
    }

    /// A deployment states the settings of its Storage Servers: its `storage_config`, with the
    /// Mooncake connector `kv_cache_config.mooncake` states, and a TCP endpoint on every
    /// interface when it spans several nodes and states none.
    #[test]
    fn a_deployment_states_its_storage_servers_settings() {
        let (plain, _) = parse(&package("")).unwrap();
        assert_eq!(plain.storage_settings(), storage::Settings::default());

        let (config, _) = parse(&package(
            "hosts = [\"rank-0\", \"rank-1\"]\n\
             [kv_cache_config.mooncake]\nmaster = \"10.0.0.1:50051\"\nsegment_gib = 4\n\
             [storage_config]\nhost_reserve_gib = 4\n",
        ))
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
            parse(&package(
                "[kv_cache_config.mooncake]\nmaster = \"10.0.0.1:50051\"\nlibrary = \"/x.so\"\n"
            ))
            .is_err()
        );
    }

    /// A node finds itself in `hosts` by its host name; the first host leads and accepts the
    /// others at its join port, and a host the list does not name serves none of it.
    #[test]
    fn a_node_finds_its_place_in_the_hosts() {
        use super::super::Identity;

        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let leader = Identity::of(&hosts, "rank-0", 7000).unwrap();
        assert!(leader.leads());
        assert_eq!((leader.rank, leader.nodes), (0, 2));
        assert_eq!(
            leader.join.map(|join| join.to_string()).as_deref(),
            Some("rank-0:7000")
        );
        let member = Identity::of(&hosts, "rank-1", 7000).unwrap();
        assert!(!member.leads());
        assert_eq!((member.rank, member.host.as_str()), (1, "rank-1"));
        let error = Identity::of(&hosts, "rank-9", 7000).unwrap_err();
        assert!(error.to_string().contains("rank-9"), "{error}");

        let alone = Identity::of(&[], "anywhere", 7000).unwrap();
        assert!(alone.leads());
        assert_eq!((alone.nodes, alone.join), (1, None));
    }
}
