//! What a fake engine is launched with and what its worker groups state.
//!
//! [`args`] and [`launch`] are a deployment's launch arguments on the CPU, as the daemon writes
//! them; [`cache`] is the cache a fake engine's ranks hold and [`capacity`] what each of its
//! worker groups serves, from which the engine's schedulers build their whole view of the groups
//! (`engine::scheduler::Capacity`), as they do from a rank start's.
//!
//! [`rank`] and [`pool`] state the inventory of the batch-interface fake ([`crate::FakeBatches`]).

use std::num::NonZeroUsize;

use ipc::batch::{Phase, Shape, Variant};
use ipc::execution::{Inventory, PoolLayout};
use ipc::{Capacity, MainProcessEndpoint, MainProcessInfo};
use proto::config::{BlockConfig, EngineArgs, WorkerGroupConfig};
use proto::request::{ModelDtype, PrefixCacheDtype};
use worker::memory::cache::CacheLayout;

/// The tokens of a block of [`cache`].
pub const BLOCK_TOKENS: u32 = 64;

/// The most calls of a batch the buckets of [`capacity`] hold, and the most tokens of a prefill
/// batch.
const BATCH_CALLS: u32 = 16;
const BATCH_TOKENS: u32 = 512;

/// The longest sequence a fake engine's groups serve, in tokens.
const MAX_MODEL_LEN: u32 = 4096;

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
            stages: std::collections::BTreeMap::from([(
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
        max_model_len: MAX_MODEL_LEN,
    }
}

/// [`launch`] of one scheduler.
pub fn args(name: &str) -> EngineArgs {
    launch(name, 1)
}

/// The launch arguments of a deployment of model `name` on the CPU with `schedulers`
/// data-parallel schedulers, each over a worker group of one rank on a device of its own on this
/// host: a text model's generation limits without a model-imposed bound, sequences of up to 4096
/// tokens, the fake's end of text ([`crate::EOS`]), two batches in flight, the module set's `fcfs` policy, and
/// every other section's default. The epoch is the first, and the process its daemon's first
/// engine.
///
/// # Panics
///
/// For no scheduler.
pub fn launch(name: &str, schedulers: usize) -> EngineArgs {
    #[expect(
        clippy::expect_used,
        reason = "DEFAULT_MAX_RUNNING_REQUESTS is a positive constant"
    )]
    let max_running_requests =
        NonZeroUsize::new(proto::config::DEFAULT_MAX_RUNNING_REQUESTS).expect("a positive default");
    #[expect(clippy::expect_used, reason = "a caller's count of schedulers")]
    let data_parallel_size = NonZeroUsize::new(schedulers).expect("a deployment of a scheduler");

    // Each group's rank names a device of its own, so the groups' memories are distinct, as on a
    // host of one GPU per data-parallel scheduler.
    let mut groups = WorkerGroupConfig::data_parallel(
        &["localhost".to_owned()],
        "cpu",
        schedulers,
        1,
        2,
        WorkerGroupConfig::single_component(proto::config::DEFAULT_BLOCK, 1),
    );
    for (index, group) in groups.iter_mut().enumerate() {
        for rank in &mut group.ranks {
            rank.device = format!("cpu:{index}");
        }
    }

    // A fake runs no module: the set names the packages a launch file states.
    let package = |backend: &str| proto::module::PackageRevision {
        backend: backend.to_owned(),
        revision: 1,
        arguments: Default::default(),
    };
    let modules = proto::module::ModuleSet {
        engine: package("uniserve"),
        policy: package("fcfs"),
        kernels: Default::default(),
    };
    let config = proto::config::EngineConfig {
        executable: proto::BUILD.to_owned(),
        model: proto::config::ModelConfig {
            model: name.to_owned(),
            base: None,
            lora_path: None,
            max_model_len: MAX_MODEL_LEN,
            dtype: ModelDtype::BFloat16,
            quantization: Default::default(),
            // A text model's: it understands, and computes the candidates of every sampled
            // token's log-probability that `max_logprobs` names.
            generation_limits: proto::request::GenerationLimits {
                features: proto::request::GenerationFeatures::UNDERSTANDING,
                max_vae_grid_tokens: u32::MAX,
                max_vit_grid_tokens: u32::MAX,
                max_latent_feature_bytes: 256 << 20,
                max_vision_feature_bytes: 256 << 20,
                encoder_cache_entries: 256,
                max_logprobs: 20,
            },
            bos: 0,
            // The end of text a fake samples (`crate::EOS`), which the scheduler must
            // recognize to finish a request.
            eos: vec![crate::EOS],
        },
        parallel: proto::config::ParallelConfig {
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size,
            engine_host: "localhost".to_owned(),
            worker_groups: groups.clone(),
        },
        deployment: format!("deployment-{}", uuid::Uuid::new_v4().simple()),
        max_running_requests,
        max_num_batched_tokens: None,
        max_logprobs: 20,
        attention_backend: String::new(),
        prefill_attention_backend: None,
        decode_attention_backend: None,
        moe_runner_backend: None,
        // The ranks are no device's: the engine launches nothing on one, and warms nothing.
        device: proto::config::DeviceConfig {
            device: "cpu".to_owned(),
            ..Default::default()
        },
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
    };
    let mut args = EngineArgs::new(config, modules);
    // A fake keeps no host tier: it has no Storage Server to keep one in.
    args.engine_config.kv_cache.host_tier = false;
    args.epoch = 1;
    args.engine = 1;
    args.scheduler_node = true;
    args.host = "localhost".to_owned();
    args.group = groups[0].id.0.clone();
    args.world_size = 1;
    args.queue_depth = 2;
    args
}

/// The cache a fake engine's ranks hold: one dense attention group of [`BLOCK_TOKENS`]-token
/// blocks over one layer of one head. Its unit sizes are an accounting figure: no rank holds the
/// memory.
pub fn cache() -> CacheLayout {
    CacheLayout::dense(BLOCK_TOKENS, 1, 1, 64, PrefixCacheDtype::BFloat16)
}

/// What each worker group of a fake engine of `launch` serves over `cache`, as a rank start
/// states it of a text model's ranks (`engine::scheduler::Capacity`): prefill and decode, the
/// launch's request slots in each, the cache's groups seeded by the model's name as by the
/// weights that compute their pages, the launch's sequences and control tokens, and buckets of up
/// to 16 calls and of prefills of up to 512 tokens in every variant a token batch chooses among.
///
/// # Panics
///
/// For a launch of more request slots than the protocol addresses, or of no batch in flight.
pub fn capacity(launch: &EngineArgs, cache: &CacheLayout) -> Capacity {
    let config = &launch.engine_config;
    #[expect(clippy::expect_used, reason = "a fixture's launch states its slots")]
    let slots = u32::try_from(config.max_running_requests.get()).expect("slots a u32 holds");

    let served = [Phase::Prefill, Phase::Decode];
    let mut rows = [0; Phase::COUNT];
    let mut phases = 0;
    for phase in served {
        phases |= 1 << phase.index();
        rows[phase.index()] = slots;
    }

    let prefills = ipc::capacity::variants(Phase::Prefill)
        .iter()
        .map(|variant| Shape::prefill(BATCH_CALLS, BATCH_TOKENS, *variant));
    let decodes = [Variant::DecodeGreedy, Variant::DecodeSampled]
        .into_iter()
        .map(|variant| Shape::decode(BATCH_CALLS, variant));
    #[expect(
        clippy::expect_used,
        reason = "a fixture's launch has batches in flight"
    )]
    let window = worker::memory::division::call_window(
        launch.queue_depth.max(1) as usize,
        BATCH_CALLS as usize,
    )
    .expect("positive bounds");

    let seed = storage::key::page_seed(&config.model.model);
    Capacity {
        phases,
        slots,
        kv_cache: Some(config.kv_cache.clone()),
        cache_groups: cache
            .groups
            .iter()
            .map(|group| (group.kind, group.block_size))
            .collect(),
        seeds: cache.groups.iter().map(|_| seed).collect(),
        buckets: prefills.chain(decodes).collect(),
        rows,
        max_model_len: config.model.max_model_len,
        denoise_steps: 0,
        bos_token_id: config.model.bos,
        eos_token_ids: config.model.eos.clone(),
        feature_bytes: 0,
        media_bytes: Vec::new(),
        call_window: u32::try_from(window).unwrap_or(u32::MAX),
    }
}
