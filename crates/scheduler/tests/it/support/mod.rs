//! Scheduler construction shared by the engine's integration tests.
//!
//! Serving states a deployment's model limits from the model's description;
//! these tests have no model, so a simulated worker's limits follow from the
//! call kinds it serves.

pub(crate) mod serving;
pub(crate) mod storage;
pub(crate) mod video;

pub(crate) use serving::{SAMPLED, Serving, config, pool};

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
    events: &mut scheduler::EventRx,
    want: usize,
    deadline: std::time::Duration,
) -> (Vec<u32>, Option<scheduler::Event>) {
    let end = std::time::Instant::now() + deadline;
    let mut tokens = Vec::new();
    while std::time::Instant::now() < end {
        match events.try_recv() {
            Ok(scheduler::Event::TextToken { id, .. }) => tokens.push(id),
            Ok(scheduler::Event::TextTokens { ids }) => tokens.extend(ids),
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

use ipc::{CallKind, ForwardMode, MediaCall};
use scheduler::{GenerationFeatures, GenerationLimits, ModelDtype};
use scheduler::{
    Parameters, PolicyProgram, Scheduler, SchedulerConfig, SimEngine, SpecialTokenIds, Worker,
};

/// Builds a scheduler whose model limits follow from `worker`'s call kinds.
///
/// A token-denoising worker without text decode serves canvas readouts over
/// prompts that may carry encoded images; one that prepares latents without
/// decoding text serves video diffusion, which needs no generation feature.
/// Any other denoising, vision-encoding or latent-encoding worker serves
/// interleaved text and images without bound, and the rest serve text.
pub(crate) fn scheduler_with(
    worker: Box<dyn Worker>,
    special_tokens: SpecialTokenIds,
    config: SchedulerConfig,
) -> anyhow::Result<Scheduler> {
    scheduler_with_stating(worker, special_tokens, config, Parameters::default())
}

/// [`scheduler_with`], installing a policy that also states `stated`.
pub(crate) fn scheduler_with_stating(
    worker: Box<dyn Worker>,
    special_tokens: SpecialTokenIds,
    config: SchedulerConfig,
    stated: Parameters,
) -> anyhow::Result<Scheduler> {
    let info = worker.info().runtime_info()?;
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
        scheduler::Submissions::new(worker),
        special_tokens,
        config,
        ModelDtype::BFloat16,
        limits,
    )?;
    install_policy_stating(&mut scheduler, scheduler::BatchCapacity::of(&info), stated);
    Ok(scheduler)
}

/// The policy module every integration test's epoch serves with.
///
/// A deployed epoch takes its policy from the Program Server; a test epoch has
/// none, and a scheduler with no policy decides nothing — no plan, no
/// admission, no batch — so an epoch that submits a request must be given one.
/// The module is the `fcfs` program this tree builds for
/// `wasm32-unknown-unknown`, the one that reproduces the rules the tests were
/// written against: `scheduler::packaged_module` builds it on first use, and a
/// module named by `SCHED_POLICY_WASM` is read instead.
pub(crate) fn policy_module() -> Vec<u8> {
    scheduler::packaged_module()
}

/// Installs `fcfs` with the ceilings an epoch advertises and the parameters the
/// epoch states about itself.
///
/// A judgement the views cannot carry is the policy's, weighed from the
/// parameters the deployment states (design §7.7): an epoch that speculates
/// states its speedup and what a refilled token costs there, because the
/// mechanism's own measurement is not a view a policy reads.
pub(crate) fn install_policy_stating(
    scheduler: &mut Scheduler,
    capacity: scheduler::BatchCapacity,
    stated: Parameters,
) {
    let ceilings = scheduler::ceilings_of(capacity);
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
    let source = scheduler::PolicyBytes::new(policy_module());
    scheduler
        .install_policy(program, &source)
        .unwrap_or_else(|error| panic!("the fcfs policy is installed: {error}"));
}

/// Installs `fcfs` on a handle's epoch through the command surface the launch
/// path uses: prepared by a precopy, then served at the switchover (§7.3).
///
/// A handle's epoch serves nothing until a policy serves it — the epoch's
/// first policy is the daemon's to install, not the config's — so a test that
/// submits to a handle installs one the same way a launched epoch receives it.
pub(crate) fn install_policy_on(
    handle: &scheduler::SchedulerHandle,
    capacity: scheduler::BatchCapacity,
) {
    let ceilings = scheduler::ceilings_of(capacity);
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
    let source: std::sync::Arc<dyn scheduler::PolicySource + Send + Sync> =
        std::sync::Arc::new(scheduler::PolicyBytes::new(policy_module()));
    handle
        .precopy_policy(program, source)
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the scheduler answers a precopy")
        .expect("the fcfs policy is precopied");
    handle
        .stage(scheduler::Stage::Switchover)
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the scheduler answers a switchover")
        .expect("the fcfs policy serves");
}

/// Builds a scheduler that batches at most `max_batch` calls per pass.
pub(crate) fn scheduler(
    worker: Box<dyn Worker>,
    special_tokens: SpecialTokenIds,
    max_batch: usize,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        worker,
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
    worker: Box<dyn Worker>,
    special_tokens: SpecialTokenIds,
    max_batch: usize,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        worker,
        special_tokens,
        SchedulerConfig {
            max_batch,
            ..Default::default()
        },
    )
}
