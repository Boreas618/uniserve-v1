//! Scheduler construction shared by the engine's integration tests.
//!
//! Serving states a deployment's model limits from the model's description;
//! these tests have no model, so a simulated worker's limits follow from the
//! call kinds it serves.

pub(crate) mod serving;
pub(crate) mod storage;
pub(crate) mod video;

pub(crate) use serving::{config, pool, Serving, SAMPLED};

/// Waits for `want` generated tokens, or for the request's terminal event.
///
/// The engine publishes generated tokens as `TextTokens` carrying the ids a
/// step produced; a single `TextToken` is the one-id form of the same event. A
/// wait that matches one variant alone reads a served request as a starved one
/// and burns its whole bound doing it, which is why both forms are matched
/// here, in the one place that knows the pair.
///
/// Returns the tokens collected and the terminal event when the request ended
/// first; reaching `want` tokens returns them with `None`. Panics by name if
/// neither happens within `deadline`.
pub(crate) fn wait_for_tokens(
    events: &mut engine::EventRx,
    want: usize,
    deadline: std::time::Duration,
) -> (Vec<u32>, Option<engine::Event>) {
    let end = std::time::Instant::now() + deadline;
    let mut tokens = Vec::new();
    while std::time::Instant::now() < end {
        match events.try_recv() {
            Ok(engine::Event::TextToken { id, .. }) => tokens.push(id),
            Ok(engine::Event::TextTokens { ids }) => tokens.extend(ids),
            Ok(event) if event.is_terminal() => return (tokens, Some(event)),
            Ok(_) => {}
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
        if tokens.len() >= want {
            return (tokens, None);
        }
    }
    panic!("the request did not reach {want} tokens within {deadline:?}: it produced {tokens:?}")
}

/// The digest of test image `id` (`ImageInput::digest`): its little-endian
/// bytes, then zeros, so that images of distinct ids differ.
pub(crate) fn image_digest(id: u64) -> [u8; 32] {
    let mut digest = [0; 32];
    digest[..8].copy_from_slice(&id.to_le_bytes());
    digest
}

use engine::{
    Executor, Parameters, PolicyProgram, Scheduler, SchedulerConfig, SimEngine, SpecialTokenIds,
};
use engine::{GenerationFeatures, GenerationLimits, ModelDtype};
use ipc::{CallKind, ForwardMode, MediaCall};

/// Builds a scheduler whose model limits follow from `executor`'s call kinds.
///
/// A token-denoising worker without text decode serves canvas readouts over
/// prompts that may carry encoded images; one that prepares latents without
/// decoding text serves video diffusion, which needs no generation feature.
/// Any other denoising, vision-encoding or latent-encoding worker serves
/// interleaved text and images without bound, and the rest serve text.
pub(crate) fn scheduler_with(
    executor: Box<dyn Executor>,
    special_tokens: SpecialTokenIds,
    config: SchedulerConfig,
) -> anyhow::Result<Scheduler> {
    scheduler_with_stating(executor, special_tokens, config, Parameters::default())
}

/// [`scheduler_with`], installing a policy that also states `stated`.
pub(crate) fn scheduler_with_stating(
    executor: Box<dyn Executor>,
    special_tokens: SpecialTokenIds,
    config: SchedulerConfig,
    stated: Parameters,
) -> anyhow::Result<Scheduler> {
    let info = executor.info().runtime_info()?;
    let calls = &info.supported_calls;
    let decodes = calls.contains(&CallKind::Forward(ForwardMode::Decode));
    let limits = if calls.contains(&CallKind::Forward(ForwardMode::TokenDenoising)) && !decodes {
        GenerationLimits {
            features: GenerationFeatures::TOKEN_DENOISING | GenerationFeatures::VISION_ENCODE,
            latent_downsample: 1,
            max_vit_grid_tokens: u32::MAX,
            max_vision_feature_bytes: 256 << 20,
            max_cfg_branches: 1,
            encoder_cache_entries: 256,
            ..Default::default()
        }
    } else if calls.contains(&CallKind::Media(MediaCall::LatentPreparation)) && !decodes {
        GenerationLimits {
            latent_downsample: 1,
            max_cfg_branches: 1,
            ..Default::default()
        }
    } else if calls.contains(&CallKind::Media(MediaCall::Denoising))
        || calls.contains(&CallKind::Media(MediaCall::VisionEncoding))
        || calls.contains(&CallKind::Media(MediaCall::LatentEncoding))
    {
        SimEngine::generation_limits()
    } else {
        GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING,
            latent_downsample: 1,
            max_cfg_branches: 1,
            ..Default::default()
        }
    };
    let mut scheduler = Scheduler::new(
        engine::Submissions::new(executor),
        special_tokens,
        config,
        ModelDtype::BFloat16,
        limits,
    )?;
    install_policy_stating(&mut scheduler, engine::BatchCapacity::of(&info), stated);
    Ok(scheduler)
}

/// The policy module every integration test's world serves with.
///
/// A deployed epoch takes its policy from the Program Server; a test world has
/// none, and a scheduler with no policy decides nothing — no plan, no
/// admission, no batch — so a world that submits a request must be given one.
/// The module is the `fcfs` program this tree builds for
/// `wasm32-unknown-unknown`, the one that reproduces the rules the tests were
/// written against: `engine::packaged_module` builds it on first use, and a
/// module named by `SCHED_POLICY_WASM` is read instead.
pub(crate) fn policy_module() -> Vec<u8> {
    engine::packaged_module()
}

/// Installs `fcfs` with the ceilings a world advertises and the parameters the
/// world states about itself.
///
/// A judgement the views cannot carry is the policy's, weighed from the
/// parameters the deployment states (design §7.7): a world that speculates
/// states its speedup and what a refilled token costs there, because the
/// mechanism's own measurement is not a view a policy reads.
pub(crate) fn install_policy_stating(
    scheduler: &mut Scheduler,
    capacity: engine::BatchCapacity,
    stated: Parameters,
) {
    let ceilings = engine::ceilings_of(capacity);
    let program = PolicyProgram {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: Parameters {
            chunked_prefill_size: Some(ceilings.chunked_prefill_size),
            max_queued_requests: ceilings.max_queued_requests,
            max_batch: Some(ceilings.max_batch),
            max_num_batched_tokens: Some(ceilings.max_num_batched_tokens),
            prefetch_min_tokens: Some(ceilings.prefetch_min_tokens),
            timeout: Some(ceilings.timeout),
            ..stated
        },
        stated: Default::default(),
    };
    let source = engine::PolicyBytes::new(policy_module());
    scheduler
        .install_policy(program, &source)
        .unwrap_or_else(|error| panic!("the fcfs policy is installed: {error}"));
}

/// Installs `fcfs` on a handle's world through the command surface the launch
/// path uses: prepared by a precopy, then served at the switchover (§7.3).
///
/// A handle's world serves nothing until a policy serves it — the epoch's
/// first policy is the daemon's to install, not the config's — so a test that
/// submits to a handle installs one the same way a launched epoch receives it.
pub(crate) fn install_policy_on(handle: &engine::SchedulerHandle, capacity: engine::BatchCapacity) {
    let ceilings = engine::ceilings_of(capacity);
    let program = PolicyProgram {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: Parameters {
            chunked_prefill_size: Some(ceilings.chunked_prefill_size),
            max_queued_requests: ceilings.max_queued_requests,
            max_batch: Some(ceilings.max_batch),
            max_num_batched_tokens: Some(ceilings.max_num_batched_tokens),
            prefetch_min_tokens: Some(ceilings.prefetch_min_tokens),
            timeout: Some(ceilings.timeout),
            ..Default::default()
        },
        stated: Default::default(),
    };
    let source: std::sync::Arc<dyn engine::PolicySource + Send + Sync> =
        std::sync::Arc::new(engine::PolicyBytes::new(policy_module()));
    handle
        .precopy_policy(program, source)
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the scheduler answers a precopy")
        .expect("the fcfs policy is precopied");
    handle
        .stage(engine::Stage::Switchover)
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the scheduler answers a switchover")
        .expect("the fcfs policy serves");
}

/// Builds a scheduler that batches at most `max_batch` calls per pass.
pub(crate) fn scheduler(
    executor: Box<dyn Executor>,
    special_tokens: SpecialTokenIds,
    max_batch: usize,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        executor,
        special_tokens,
        SchedulerConfig {
            max_batch,
            ..Default::default()
        },
    )
}

/// Builds a scheduler that batches at most `max_batch` calls per pass and
/// orders waiting requests by `policy`.
pub(crate) fn scheduler_with_depth(
    executor: Box<dyn Executor>,
    special_tokens: SpecialTokenIds,
    max_batch: usize,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        executor,
        special_tokens,
        SchedulerConfig {
            max_batch,
            ..Default::default()
        },
    )
}

/// The leader daemon this test process runs for the worker groups its tests
/// launch, as a deployment of one node whose host is `localhost`; returns
/// its process socket. The workers it starts are children of its service
/// thread and end with the test process.
pub(crate) fn daemon() -> String {
    static DAEMON: std::sync::OnceLock<(daemon::Daemon, String)> = std::sync::OnceLock::new();
    DAEMON
        .get_or_init(|| {
            let directory = tempfile::tempdir().unwrap().keep();
            let daemon = daemon::Daemon::start(daemon::DaemonOptions {
                nnodes: 1,
                node_rank: 0,
                host: "localhost".into(),
                dist_init_addr: None,
                socket: directory.join("daemon.sock"),
                executable: uniserve_executable(),
                join_timeout: daemon::JOIN_TIMEOUT,
            })
            .unwrap();
            let address = daemon.address().unwrap().to_owned();
            (daemon, address)
        })
        .1
        .clone()
}

/// The `uniserve` executable the workers run as, and that starts the node's
/// Storage Server: `UNISERVE_WORKER_EXECUTABLE`, or else the repository
/// environment's.
pub(crate) fn uniserve_executable() -> std::path::PathBuf {
    std::env::var_os("UNISERVE_WORKER_EXECUTABLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(".venv/bin/uniserve")
        })
}
