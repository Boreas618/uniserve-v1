//! The launch file a main process reads to learn the facts of the rank it runs.
//!
//! A main process that holds its own devices is started with one file that
//! states its launch: which model, which group of the world it is and at which
//! rank, the budgets its schedulers are built for, the cache's geometry, and
//! the model's own bounds. The file is the single source of those values - the
//! process derives nothing from its environment and the welcome states nothing
//! of them (`M1379`) - so [`read`] is the boot's first step and
//! [`RankFacts`](crate::RankFacts) is what it produces.
//!
//! # The keys
//!
//! Every key is a dotted path of the file's tables, and each is the same path
//! the deployment states the value under:
//!
//! | key | fact |
//! |---|---|
//! | `model` | the model's path or served name |
//! | `world` | the world the ranks execute |
//! | `worker_config.dtype` | the parameters' and activations' dtype |
//! | `worker_config.watchdog_timeout_ms` | how long a job may run without progress |
//! | `placement_config.group` | the worker group's identity |
//! | `placement_config.rank` | this process's rank within the group |
//! | `placement_config.hosts` | the group's hosts, in rank order |
//! | `placement_config.devices` | the group's devices, in rank order |
//! | `scheduling.max_running_requests` | the resident request slots |
//! | `scheduling.max_batch` | the calls one step may assemble |
//! | `scheduling.max_num_batched_tokens` | the tokens one step may carry |
//! | `kv_cache_config.block_size` | the KV cache's block size in tokens |
//! | `kv_cache_config.dtype` | the KV cache's dtype |
//! | `kv_cache_config.host_ratio` | the host cache's size over the device's |
//! | `kv_cache_config.host_bytes` | the host cache's bytes per host, if stated |
//! | `cuda_graphs.decode_batch_sizes` | the decode batch sizes to capture |
//! | `cuda_graphs.prefill_token_sizes` | the prefill token sizes to capture |
//! | `model_config.bos` | the tokenizer's beginning of sequence |
//! | `model_config.eos` | the tokenizer's end of sequence |
//! | `model_config.end_of_image` | the model's image end token |
//! | `model_config.max_model_len` | the model's context length |
//! | `model_config.generation_limits` | what one request may be given to generate |
//!
//! A key the file does not state is left `None` and nothing is guessed in its
//! place: [`EngineConfig::of`](crate::EngineConfig::of) refuses the missing fact
//! by its name and by where it comes from, which is what makes an incomplete
//! launch file a startup failure that names itself rather than a process that
//! runs on values nobody stated.
//!
//! Keys the file states and this reader does not know are ignored: a launch
//! file is the deployment's, and it states the world's placement and every
//! group's components for the ranks the daemon starts. The reader takes the
//! facts of this rank and leaves the rest to their owners.

use std::path::Path;

use anyhow::Context;
use serde::de::DeserializeOwned;

use crate::RankFacts;
use crate::kv_cache::HostCacheSize;

/// Reads the launch file at `path` and answers the facts it states.
pub fn read(path: impl AsRef<Path>) -> anyhow::Result<RankFacts> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("the launch file `{}` is readable", path.display()))?;
    parse(&text).with_context(|| format!("the launch file `{}`", path.display()))
}

/// Reads the launch file `text` states and answers the facts it states.
///
/// The file is TOML, the deployment's own notation: tables are the sections
/// whose dotted keys the table above lists, and a value's type is the fact's.
pub fn parse(text: &str) -> anyhow::Result<RankFacts> {
    let tree: toml::Table = text
        .parse()
        .context("the launch file is a TOML table of the deployment's own keys")?;

    let hosts: Vec<String> = value(&tree, "placement_config.hosts")?.unwrap_or_default();
    let devices: Vec<String> = value(&tree, "placement_config.devices")?.unwrap_or_default();
    if !hosts.is_empty() && !devices.is_empty() && hosts.len() != devices.len() {
        anyhow::bail!(
            "the launch file places {} hosts and {} devices: a group states one device per rank",
            hosts.len(),
            devices.len()
        );
    }
    let world_size = match (hosts.is_empty(), devices.is_empty()) {
        (false, _) => Some(hosts.len()),
        (true, false) => Some(devices.len()),
        (true, true) => None,
    };

    Ok(RankFacts {
        model: value(&tree, "model")?,
        dtype: value(&tree, "worker_config.dtype")?,
        world: value(&tree, "world")?,
        group: value(&tree, "placement_config.group")?,
        rank: value(&tree, "placement_config.rank")?,
        world_size,
        devices: (!devices.is_empty()).then_some(devices),
        hosts: (!hosts.is_empty()).then_some(hosts),
        // The groups this rank exchanges state with are the daemon's placement
        // edges: a rank's own file does not state them, and `EngineConfig::of`
        // derives the transfer edges between the ranks it does state.
        peers: None,
        max_running_requests: value(&tree, "scheduling.max_running_requests")?,
        max_batch: value(&tree, "scheduling.max_batch")?,
        max_num_batched_tokens: value(&tree, "scheduling.max_num_batched_tokens")?,
        block_size: value(&tree, "kv_cache_config.block_size")?,
        kv_cache_dtype: value(&tree, "kv_cache_config.dtype")?,
        host_cache: host_cache(&tree)?,
        graph_tiers: graph_tiers(&tree)?,
        bos: value(&tree, "model_config.bos")?,
        eos: value(&tree, "model_config.eos")?,
        end_of_image: value(&tree, "model_config.end_of_image")?,
        generation_limits: value(&tree, "model_config.generation_limits")?,
        max_model_len: value(&tree, "model_config.max_model_len")?,
        watchdog_timeout: value::<u64>(&tree, "worker_config.watchdog_timeout_ms")?
            .map(std::time::Duration::from_millis),
    })
}

/// The host cache the file states, or none when it states no part of it.
///
/// `host_ratio` alone is a host cache sized against the device cache; with
/// `host_bytes` the deployment states the size outright, and the ratio stands as
/// the share it is expected to be.
fn host_cache(tree: &toml::Table) -> anyhow::Result<Option<HostCacheSize>> {
    let ratio: Option<f64> = value(tree, "kv_cache_config.host_ratio")?;
    let host_bytes: Option<u64> = value(tree, "kv_cache_config.host_bytes")?;
    match (ratio, host_bytes) {
        (None, None) => Ok(None),
        (ratio, host_bytes) => Ok(Some(HostCacheSize {
            ratio: ratio.unwrap_or(HostCacheSize::default().ratio),
            host_bytes,
        })),
    }
}

/// The captured graph sizes, ascending and once each: the file states them per
/// phase, and a graph is a graph whatever it was captured for.
fn graph_tiers(tree: &toml::Table) -> anyhow::Result<Option<Vec<u32>>> {
    let decode: Option<Vec<u32>> = value(tree, "cuda_graphs.decode_batch_sizes")?;
    let prefill: Option<Vec<u32>> = value(tree, "cuda_graphs.prefill_token_sizes")?;
    if decode.is_none() && prefill.is_none() {
        return Ok(None);
    }
    let mut tiers: Vec<u32> = decode.into_iter().chain(prefill).flatten().collect();
    tiers.sort_unstable();
    tiers.dedup();
    Ok(Some(tiers))
}

/// The value `path` names in `tree`, or `None` when the file does not state it.
///
/// A key stated with the wrong type is an error that names the key: the file is
/// a launch's, and a value it states wrongly is not a value to fall back from.
fn value<T: DeserializeOwned>(tree: &toml::Table, path: &str) -> anyhow::Result<Option<T>> {
    let Some(found) = dotted(tree, path) else {
        return Ok(None);
    };
    found
        .clone()
        .try_into()
        .with_context(|| format!("the launch file's `{path}` is the value its kind states"))
        .map(Some)
}

/// The value at the dotted `path`, when every table on the way states one.
fn dotted<'a>(tree: &'a toml::Table, path: &str) -> Option<&'a toml::Value> {
    let mut parts = path.split('.');
    let mut value = tree.get(parts.next()?)?;
    for part in parts {
        value = value.as_table()?.get(part)?;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A launch file stating every key the reader knows, for one rank of a
    /// group of two.
    const STATED: &str = r#"
model = "meta-llama/Llama-3.1-8B"
world = 3

[worker_config]
dtype = "bfloat16"
watchdog_timeout_ms = 30000

[placement_config]
group = "model"
rank = 1
hosts = ["node-0", "node-1"]
devices = ["cuda:0", "cuda:0"]

[scheduling]
max_running_requests = 128
max_batch = 64
max_num_batched_tokens = 8192

[kv_cache_config]
block_size = 64
dtype = "bfloat16"
host_ratio = 2.0
host_bytes = 68719476736

[cuda_graphs]
decode_batch_sizes = [1, 8, 32]
prefill_token_sizes = [4, 32]

[model_config]
bos = 128000
eos = [128001, 128008]
end_of_image = 128257
max_model_len = 131072

[model_config.generation_limits]
max_latent_units = 1024
"#;

    #[test]
    fn a_launch_file_states_every_fact_its_rank_is_built_from() {
        let facts = parse(STATED).expect("the launch file parses");
        assert_eq!(facts.model.as_deref(), Some("meta-llama/Llama-3.1-8B"));
        assert_eq!(facts.dtype, Some(crate::ModelDtype::BFloat16));
        assert_eq!(facts.world, Some(3));
        assert_eq!(facts.group.as_deref(), Some("model"));
        assert_eq!(facts.rank, Some(1));
        assert_eq!(facts.world_size, Some(2), "the group's ranks are its hosts");
        assert_eq!(
            facts.hosts.as_deref(),
            Some(["node-0".to_owned(), "node-1".to_owned()].as_slice())
        );
        assert_eq!(facts.max_running_requests, Some(128));
        assert_eq!(facts.max_batch, Some(64));
        assert_eq!(facts.max_num_batched_tokens, Some(8192));
        assert_eq!(facts.block_size, Some(64));
        assert_eq!(facts.kv_cache_dtype, Some(ipc::PrefixCacheDtype::BFloat16));
        assert_eq!(
            facts.host_cache,
            Some(HostCacheSize {
                ratio: 2.0,
                host_bytes: Some(68_719_476_736),
            })
        );
        assert_eq!(
            facts.graph_tiers.as_deref(),
            Some([1, 4, 8, 32, 32].as_slice()),
            "both phases' sizes, ascending"
        );
        assert_eq!(facts.bos, Some(128_000));
        assert_eq!(facts.eos.as_deref(), Some([128_001, 128_008].as_slice()));
        assert_eq!(facts.end_of_image, Some(128_257));
        assert_eq!(facts.max_model_len, Some(131_072));
        assert_eq!(
            facts.generation_limits,
            Some(crate::GenerationLimits {
                max_latent_units: 1024,
                ..Default::default()
            })
        );
        assert_eq!(
            facts.watchdog_timeout,
            Some(std::time::Duration::from_millis(30_000))
        );
    }

    #[test]
    fn a_key_the_file_does_not_state_is_refused_by_the_derivation_and_not_guessed() {
        let facts = parse("model = \"stub\"\n[scheduling]\nmax_batch = 8\n")
            .expect("a partial launch file parses");
        assert_eq!(facts.model.as_deref(), Some("stub"));
        assert_eq!(facts.max_batch, Some(8));
        assert_eq!(facts.dtype, None, "nothing was guessed");

        // The derivation is where an unstated fact becomes a failure, and it
        // names the fact and the source that owed it.
        let error = crate::EngineConfig::of(&facts).expect_err("the facts are incomplete");
        let message = format!("{error}");
        assert!(
            message.contains("dtype") && message.contains("launch file"),
            "the refusal names the fact and its source: {message}"
        );
    }

    #[test]
    fn a_key_stated_with_another_type_is_an_error_that_names_it() {
        let error =
            parse("[scheduling]\nmax_batch = \"many\"\n").expect_err("a batch bound is a number");
        let message = format!("{error:#}");
        assert!(
            message.contains("scheduling.max_batch"),
            "the refusal names the key: {message}"
        );
    }

    #[test]
    fn a_group_states_one_device_per_rank() {
        let error =
            parse("[placement_config]\nhosts = [\"node-0\"]\ndevices = [\"cuda:0\", \"cuda:1\"]\n")
                .expect_err("a group's ranks have one device each");
        assert!(
            format!("{error}").contains("one device per rank"),
            "the refusal says what the placement must be: {error}"
        );
    }
}
