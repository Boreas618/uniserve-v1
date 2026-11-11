//! Scheduler construction shared by the engine's integration tests.
//!
//! Serving states a deployment's model limits from the model's description;
//! these tests have no model, so a simulated worker's limits follow from the
//! call kinds it serves.

pub(crate) mod fake;
pub(crate) mod storage;
pub(crate) mod toy;
pub(crate) mod video;

pub(crate) use testkit::{SAMPLED, pool};

use proto::config::EngineArgs;
use testkit::{FakeBatches, Tokens};

/// A fake worker over `inventory` that samples [`SAMPLED`] plus each call's place in its batch:
/// every batch looks alike, so a case reads the mechanism's choices from the batch, not the token.
pub(crate) fn positional(inventory: Vec<ipc::execution::Inventory>) -> FakeBatches {
    FakeBatches::new(inventory).with_tokens(Tokens::Positional)
}

/// A fake worker over `inventory` whose sampled token follows each call's cache extent: the same
/// request continued from a successor samples what it would have sampled uninterrupted.
pub(crate) fn by_extent(inventory: Vec<ipc::execution::Inventory>) -> FakeBatches {
    FakeBatches::new(inventory).with_tokens(Tokens::Extent)
}

/// The arguments and limits of a handle over `pool` for a deployment named `name`: the
/// simulation's launch facts, with as many running requests as the pool holds rows.
pub(crate) fn config(pool: &FakeBatches, name: &str) -> (EngineArgs, SchedulerConfig) {
    (testkit::args(name), testkit::limits(pool))
}

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

use ipc::{CallKind, ForwardMode, MainProcessInfo, MediaCall};
use scheduler::{GenerationFeatures, GenerationLimits};
use scheduler::{Parameters, PolicyProgram, SchedulerConfig};
use testkit::EndOfText;

/// A rank of the scheduler tests' text deployment: every call kind, a buffer
/// pool and encoder cache large enough for any test's products, up to 1024
/// calls a batch and two unresolved calls a request slot, and the paged KV
/// pool of `MainProcessInfo::default` (64-token pages, one unit each).
pub(crate) fn text_rank() -> MainProcessInfo {
    MainProcessInfo {
        supported_calls: CallKind::ALL.to_vec(),
        buffer_pool_bytes: 257_u64 * (256 << 20),
        encoder_cache_entries: 256,
        encoder_entry_bytes: 256 << 20,
        max_logprobs: 20,
        max_batch_calls: 1024,
        max_unresolved_calls: 2,
        ..MainProcessInfo::default()
    }
}

/// The longest sequence the tests' text deployment is launched for.
pub(crate) const TEXT_MAX_MODEL_LEN: u32 = 32_768;

/// A fake worker of one pool named `model` (the pool `testkit::args` launches) of one `rank`, holding `queue_depth`
/// batches in flight, whose text ends at `end` (never, with `None`).
pub(crate) fn text_pool(
    rank: MainProcessInfo,
    queue_depth: usize,
    end: Option<EndOfText>,
) -> FakeBatches {
    let pool = ipc::execution::Inventory {
        group: "model".to_owned(),
        ranks: vec![rank],
        layout: ipc::execution::PoolLayout {
            queue_depth,
            ranks: 1,
            stages: std::collections::BTreeMap::new(),
        },
        max_model_len: TEXT_MAX_MODEL_LEN,
    };
    let batches = FakeBatches::new(vec![pool]);
    match end {
        Some(end) => batches.with_end(end),
        None => batches,
    }
}

/// The model limits a deployment whose workers serve `calls` states.
///
/// A token-denoising worker without text decode generates block-diffusion
/// canvases over prompts that may carry encoded images; one that prepares
/// latents without decoding text serves video diffusion, which needs no
/// generation feature. Any other vision- or latent-encoding worker reads
/// images without bound, and the rest serve text.
pub(crate) fn generation_limits(calls: &[CallKind]) -> GenerationLimits {
    let decodes = calls.contains(&CallKind::Forward(ForwardMode::Decode));
    if calls.contains(&CallKind::Forward(ForwardMode::TokenDenoising)) && !decodes {
        GenerationLimits {
            features: GenerationFeatures::TOKEN_DENOISING | GenerationFeatures::VISION_ENCODE,
            max_vit_grid_tokens: u32::MAX,
            max_vision_feature_bytes: 256 << 20,
            encoder_cache_entries: 256,
            max_logprobs: 20,
            ..Default::default()
        }
    } else if calls.contains(&CallKind::Media(MediaCall::LatentPreparation)) && !decodes {
        GenerationLimits::default()
    } else if calls.contains(&CallKind::Media(MediaCall::Denoising))
        || calls.contains(&CallKind::Media(MediaCall::VisionEncoding))
        || calls.contains(&CallKind::Media(MediaCall::LatentEncoding))
    {
        GenerationLimits {
            max_vae_grid_tokens: u32::MAX,
            max_vit_grid_tokens: u32::MAX,
            max_latent_feature_bytes: 256 << 20,
            max_vision_feature_bytes: 256 << 20,
            encoder_cache_entries: 256,
            max_logprobs: 20,
            ..Default::default()
        }
    } else {
        GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING,
            ..Default::default()
        }
    }
}

/// The policy module every integration test's epoch serves with.
///
/// A deployed epoch takes its policy from the Program Server; a test epoch has
/// none, and a scheduler with no policy decides nothing — no output, no
/// admission, no batch — so an epoch that submits a request must be given one.
/// The module is the `fcfs` program this tree builds for
/// `wasm32-unknown-unknown`, the one that reproduces the rules the tests were
/// written against: `testkit::packaged_module` builds it on first use, and a
/// module named by `SCHED_POLICY_WASM` is read instead.
pub(crate) fn policy_module() -> Vec<u8> {
    testkit::packaged_module()
}

/// The `fcfs` program with the ceilings an epoch advertises and the parameters
/// the epoch states about itself.
///
/// A judgement the views cannot carry is the policy's, weighed from the
/// parameters the deployment states: an epoch that speculates states its
/// speedup and what a refilled token costs there, because the mechanism's own
/// measurement is not a view a policy reads.
pub(crate) fn fcfs(capacity: scheduler::BatchCapacity, stated: Parameters) -> PolicyProgram {
    let ceilings = scheduler::ceilings_of(capacity);
    PolicyProgram {
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
    }
}

/// The source the tests' policy module is read from.
pub(crate) fn policy_source() -> std::sync::Arc<dyn scheduler::PolicySource + Send + Sync> {
    std::sync::Arc::new(scheduler::PolicyBytes::new(policy_module()))
}

/// The response to a policy command, which a scheduler gives within a minute.
pub(crate) fn applied<T>(
    response: crossbeam_channel::Receiver<Result<T, String>>,
    what: &str,
) -> T {
    response
        .recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("the scheduler responds {what}"))
        .unwrap_or_else(|error| panic!("{what} of the fcfs policy: {error}"))
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
    let program = fcfs(capacity, Parameters::default());
    applied(handle.precopy_policy(program, policy_source()), "a precopy");
    applied(handle.stage(scheduler::Stage::Switchover), "a switchover");
}
