//! The pools a fake worker states, and what a handle over one is built with.
//!
//! The mechanism builds its whole view of a worker from [`Inventory`]: its ranks, their call kinds
//! and request rows, and the layout the deployment launched the pool with. A case states its pool
//! here and the handle derives the rest exactly as it does for a launch.

use ipc::execution::{Batches, Inventory, PoolLayout};
use ipc::{MainProcessEndpoint, MainProcessInfo};
use proto::config::{BlockConfig, EngineArgs};
use scheduler::SchedulerConfig;

/// One rank of a text pool on `host`: prefill and decode, eight request rows, the batch bounds a
/// launch states, and the paged KV pool every token-serving rank advertises.
pub fn rank(host: &str) -> MainProcessInfo {
    MainProcessInfo {
        endpoint: MainProcessEndpoint {
            host: host.to_owned(),
            address_space: format!("{host}-space"),
            incarnation: format!("{host}-rank"),
        },
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        request_slots: 8,
        max_batch_calls: 16,
        max_batch_tokens: 512,
        // The prefix cache is `MainProcessInfo::default`'s: a rank without one is refused every
        // token request at admission, which is the rule for a media-only rank and not this one.
        buffer_pool_bytes: 1 << 20,
        // A rank that samples states how many candidates it computes for a position, which
        // bounds what a request may ask (`GenerationLimits::max_logprobs`): the fake reports
        // every sampled token's log-probability, as a launch with `max_logprobs = 20` does.
        max_logprobs: 20,
        ..MainProcessInfo::default()
    }
}

/// A text pool named `group` of one [`rank`], one batch in flight per device, serving the default
/// block on that rank.
pub fn pool(group: &str) -> Inventory {
    Inventory {
        group: group.to_owned(),
        ranks: vec![rank("gpu-a")],
        layout: PoolLayout {
            queue_depth: 1,
            ranks: 1,
            components: std::collections::BTreeMap::from([(
                proto::config::DEFAULT_BLOCK.to_owned(),
                BlockConfig {
                    ranks: vec![0],
                    tensor_parallel_size: 1,
                    pipeline_parallel_size: 1,
                    sequence_parallel: Default::default(),
                    units_per_rank: 1,
                    weights: Default::default(),
                    distribution: None,
                },
            )]),
        },
        max_model_len: 4096,
    }
}

/// The launch arguments of a deployment named `name` on the CPU: one pool of one rank on this
/// host, named as its worker group ([`proto::config::DEFAULT_BLOCK`]), generation limits without a
/// model-imposed bound, the fake's end of text ([`crate::EOS`]), two batches in flight, and every
/// other section's default. Pair them with a fake over [`pool`] of `args.group`.
pub fn args(name: &str) -> EngineArgs {
    launch(name)
}

fn launch(model: impl Into<String>) -> EngineArgs {
    use std::num::NonZeroUsize;
    #[expect(
        clippy::expect_used,
        reason = "DEFAULT_MAX_RUNNING_REQUESTS is a positive constant"
    )]
    let max_running_requests =
        NonZeroUsize::new(proto::config::DEFAULT_MAX_RUNNING_REQUESTS).expect("a positive default");
    let group = proto::config::WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        1,
        2,
        proto::config::WorkerGroupConfig::single_component(proto::config::DEFAULT_BLOCK, 1),
    );
    let mut args = EngineArgs::new(proto::config::EngineConfig {
        executable: proto::BUILD.to_owned(),
        model: proto::config::ModelConfig {
            model: model.into(),
            base: None,
            lora_path: None,
            max_model_len: 0,
            dtype: scheduler::ModelDtype::BFloat16,
            quantization: Default::default(),
            generation_limits: proto::request::GenerationLimits {
                max_vae_grid_tokens: u32::MAX,
                max_vit_grid_tokens: u32::MAX,
                max_latent_feature_bytes: 256 << 20,
                max_vision_feature_bytes: 256 << 20,
                encoder_cache_entries: 256,
                max_logprobs: 20,
                ..Default::default()
            },
            bos: 0,
            // The end of text a fake samples (`crate::EOS`), which the scheduler must
            // recognize to finish a request.
            eos: vec![crate::EOS],
            canvas_sampling: None,
            prefill_outputs: true,
        },
        parallel: proto::config::ParallelConfig {
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size: NonZeroUsize::MIN,
            engine_host: "localhost".to_owned(),
            worker_groups: vec![group.clone()],
        },
        deployment: format!("deployment-{}", uuid::Uuid::new_v4().simple()),
        max_running_requests,
        max_num_batched_tokens: None,
        max_logprobs: 20,
        attention_backend: String::new(),
        prefill_attention_backend: None,
        decode_attention_backend: None,
        moe_runner_backend: None,
        device: Default::default(),
        scheduler: Default::default(),
        media: Default::default(),
        mem_fraction_static: 0.9,
        load: Default::default(),
        kv_cache: Default::default(),
        moe: Default::default(),
        cuda_graph: Default::default(),
        worker: Default::default(),
        speculative: Default::default(),
        video: Default::default(),
        profiling: Default::default(),
    });
    // A fake keeps no host tier: it has no Storage Server to keep one in.
    args.engine_config.kv_cache.host_tier = false;
    args.host = "localhost".to_owned();
    args.group = group.id.0.clone();
    #[expect(
        clippy::expect_used,
        reason = "the group is placed with one rank, which a u32 holds"
    )]
    let world_size = u32::try_from(group.ranks.len()).expect("one rank");
    args.world_size = world_size;
    args.queue_depth = 2;
    args
}

/// The mechanism's limits for `batches`: as many running requests as its ranks hold request rows.
///
/// `SchedulerHandle::over_batches` refuses a configuration that asks for more running requests
/// than the ranks hold rows, so the count is read from the pool rather than restated per case.
pub fn limits(batches: &dyn Batches) -> SchedulerConfig {
    let rows: usize = batches
        .inventory()
        .iter()
        .flat_map(|inventory| &inventory.ranks)
        .map(|rank| rank.request_slots as usize)
        .sum();
    SchedulerConfig {
        max_running_requests: rows.max(1),
        ..SchedulerConfig::default()
    }
}
