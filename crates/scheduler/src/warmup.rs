//! Serving warmup through the scheduler.
//!
//! When the process opens admission, the scheduler admits real requests of the
//! shapes the workers prepared for, so the serving path of every captured row
//! count and diffusion LM step has run before the first client request that
//! needs it.
//!
//! It runs at the opening of admission, not at startup or at the epoch's first
//! switchover, because its requests are batches like a client's: the scheduler
//! plans them into the device's captured buckets, and the switchover only
//! queues those captures, which run behind it for as long as a model's bucket
//! catalogue takes. A batch planned before its bucket is captured finds no graph
//! to replay, which fails the worker. Admission opens once every capture has
//! run, and a scheduler that admits has a policy (one with none forms no batch),
//! so the owner of admission starts the warm-up then
//! ([`crate::SchedulerHandle::warm_serving_path`]) and does not wait for it; a
//! warm-up that fails is logged while serving continues — a device that cannot
//! run these shapes fails the first client request that needs them, in the
//! open. The scheduler and its allocators form every batch as they form a
//! client's; a rank states only the shapes
//! (`MainProcessInfo::diffusion_lm_step_rows`) and takes no part in forming
//! them. The ranks warmed their kernels and captured their graphs before
//! admission opened; these requests warm the serving path around them.
//!
//! The requests are ordinary generation requests with a one-token prompt:
//!
//! - a token request of two tokens, which runs a prefill and a decode that
//!   consumes the prefill's token;
//! - for each row count from `diffusion_lm_step_rows` down to one, that many
//!   diffusion LM requests of one block, submitted together: they prefill
//!   their prompts in one batch and then step together, so every step of
//!   the group runs with exactly that many rows. The one-row request
//!   generates a second block, so its committed block runs the prefill of a
//!   commit.
//!
//! No request reads or populates the prefix or encoder caches. A request the
//! deployment's generation limits would refuse a client is not one serving
//! can reach and is left out.

use std::time::Duration;

use crate::{CachePolicy, Event, FinishReason, GenerationRequest, MultimodalInputs, Request};
use anyhow::Context as _;
use ipc::SamplingParams;
use ipc::{CallKind, ForwardMode};

use crate::handle::EventRx;
use proto::config::EngineArgs;

/// Runs the warmup requests the workers' reports call for on `core`.
///
/// Returns once every request has finished. Fails, naming the group, when a
/// request is rejected, ends in an error, or the engine dies first.
///
/// Warmup runs only where the workers execute on CUDA devices: a CPU or stub
/// worker has no kernels or graphs to warm.
/// What a warm-up needs of a scheduler, so a thread of the engine's can warm a
/// rank without holding the handle the engine serves with.
pub(crate) struct Warm {
    /// The rank's reported capabilities, which state the shapes to warm.
    pub(crate) info: ipc::MainProcessInfo,
    /// The generation limits its requests are admitted under.
    pub(crate) generation_limits: crate::GenerationLimits,
    /// The scheduler's request ids, shared with the handle that owns them.
    pub(crate) next_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Set when the scheduler stops fatally; see [`Warm::is_dead`].
    pub(crate) dead: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The scheduler's first fatal error, if it has one.
    pub(crate) failure: std::sync::Arc<std::sync::OnceLock<String>>,
    /// The scheduler's command channel, which the requests are submitted on.
    pub(crate) commands: crate::CommandSender,
}

impl Warm {
    /// The rank's reported capabilities.
    fn info(&self) -> &ipc::MainProcessInfo {
        &self.info
    }

    /// The generation limits a request of the deployment's shapes is admitted
    /// under.
    fn generation_limits(&self) -> crate::GenerationLimits {
        self.generation_limits.clone()
    }

    /// A request id no client request takes: the warm-up's requests may still run when the
    /// first client requests arrive, and one id names one request of the scheduler.
    fn next_request_id(&self) -> ipc::RequestId {
        ipc::RequestId(
            self.next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Whether the scheduler stopped fatally, which ends a warm-up.
    fn is_dead(&self) -> bool {
        self.dead.load(std::sync::atomic::Ordering::SeqCst) || self.failure.get().is_some()
    }

    /// Submits the warm-up's requests together, as a client's arrival would.
    fn submit_all(&self, requests: Vec<Request>) -> Result<Vec<EventRx>, crate::SubmitError> {
        if self.is_dead() {
            return Err(crate::SubmitError::Dead);
        }
        self.commands.submit_all(requests)
    }
}

/// Runs the warm-up of `args`' deployment on a thread of its own ([`Warm`] is
/// what it holds), from the opening of admission
/// ([`crate::SchedulerHandle::warm_serving_path`]).
pub(crate) fn start(intent: Warm, args: EngineArgs) {
    let spawned = std::thread::Builder::new()
        .name("warmup".to_owned())
        .spawn(move || {
            if let Err(error) = warm(&intent, &args) {
                // Serving continues: the shapes this warms are the ones a client
                // request reaches anyway, where its failure is visible.
                tracing::warn!(%error, "the serving path's warm-up failed; the engine serves on");
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "the warm-up thread could not start; the engine serves on");
    }
}

/// One group of warm-up requests: the shapes one part of the serving path runs,
/// submitted together and awaited together.
pub(crate) struct Group {
    /// What the group warms, for the log: a token model's prefill and decode,
    /// a diffusion LM's row count.
    pub(crate) what: String,
    /// The requests, submitted in one command so the scheduler queues all of
    /// them before it forms its next batch.
    pub(crate) requests: Vec<GenerationRequest>,
}

/// The warm-up requests this deployment's ranks call for, in the order they run.
///
/// The set is kept apart from submitting it: the submission needs a policy —
/// a scheduler with no policy forms no batch — while the requests a
/// deployment's shapes call for can be stated, and checked, without one.
/// `limits` are the deployment's resolved generation limits, so a shape serving
/// could not reach is left out here as well.
pub(crate) fn groups(
    args: &EngineArgs,
    info: &ipc::MainProcessInfo,
    limits: &crate::GenerationLimits,
    mut next_id: impl FnMut() -> ipc::RequestId,
) -> anyhow::Result<Vec<Group>> {
    let facts = &args.engine_config.model;
    let supports = |kind| info.supported_calls.contains(&kind);
    // Keeps the requests clients could submit, naming what it leaves out.
    let servable = |what: &str, requests: Vec<GenerationRequest>| {
        // A deployment with block sampling generates every request in
        // canvases, so admission checks its canvas features.
        let canvases = facts.canvas_sampling.is_some();
        let (kept, refused): (Vec<_>, Vec<_>) = requests.into_iter().partition(|request| {
            let needs = if canvases {
                request.canvas_features()
            } else {
                request.required_features()
            };
            limits.features.contains(needs) && request.validate_resources(limits).is_ok()
        });
        if !refused.is_empty() {
            tracing::info!(
                "left out {} {what} warmup requests the deployment's generation limits refuse",
                refused.len()
            );
        }
        kept
    };
    let mut und = |tokens: usize| warmup_request(facts.bos, next_id(), tokens);

    let mut groups = Vec::new();

    // The token request: a prefill followed by a decode over its token. A
    // diffusion LM deployment, whose limits grant no understanding feature,
    // generates in blocks instead, which its groups below run.
    let decodes = supports(CallKind::Forward(ForwardMode::Prefill))
        && supports(CallKind::Forward(ForwardMode::Decode));
    if decodes {
        let requests = servable("token", vec![und(2)]);
        if !requests.is_empty() {
            groups.push(Group {
                what: "token".to_owned(),
                requests,
            });
        }
    }

    if info.diffusion_lm_step_rows > 0 {
        let sampling = facts.canvas_sampling.context(
            "workers report diffusion LM step rows but the deployment has no block sampling",
        )?;
        let length = sampling.canvas_length as usize;
        // Requests of different lengths stop stepping together once the
        // shortest finishes: each commits its block in its own batch, and
        // whether two of them step together again depends on timing. So each
        // row count is its own group of equal requests.
        for rows in (1..=info.diffusion_lm_step_rows as usize).rev() {
            let blocks = if rows == 1 { 2 } else { 1 };
            let requests = servable(
                "diffusion LM",
                // The scheduler generates every such request in canvases of
                // the deployment's block sampling.
                (0..rows).map(|_| und(blocks * length)).collect(),
            );
            if !requests.is_empty() {
                groups.push(Group {
                    what: "diffusion LM".to_owned(),
                    requests,
                });
            }
        }
    }

    Ok(groups)
}

pub(crate) fn warm(core: &Warm, args: &EngineArgs) -> anyhow::Result<()> {
    let on_cuda = args
        .engine_config
        .parallel
        .worker_groups
        .iter()
        .flat_map(|worker| &worker.ranks)
        .any(|rank| rank.device == "cuda" || rank.device.starts_with("cuda:"));
    if !on_cuda {
        return Ok(());
    }

    let info = core.info().clone();
    let limits = core.generation_limits();
    for group in groups(args, &info, &limits, || core.next_request_id())? {
        run(core, &group.what, group.requests)?;
        tracing::info!(warmup = %group.what, "completed a serving path warm-up group");
    }
    Ok(())
}

/// Builds one warmup generation request of `tokens` understanding tokens.
fn warmup_request(bos: u32, request_id: ipc::RequestId, tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id,
        prompt_token_ids: vec![bos],
        multimodal_inputs: MultimodalInputs::default(),
        sampling: SamplingParams {
            temperature: 0.0,
            ignore_eos: true,
            // A diffusion LM request needs a seed; the rest ignore it.
            seed: Some(0),
            ..SamplingParams::default()
        },
        max_und_tokens: tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy {
            read: false,
            write: false,
            isolation_key: None,
            state_seeds: Vec::new(),
            resume: None,
        },
        include_stop_token: false,
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// Submits `requests` together and waits for each to finish.
///
/// The requests reach the scheduler in one command, so it queues all of them
/// before it forms its next batch, as it would concurrent clients that all
/// arrived before that batch.
fn run(core: &Warm, group: &str, requests: Vec<GenerationRequest>) -> anyhow::Result<()> {
    let mut streams = core
        .submit_all(requests.into_iter().map(Request::from).collect())
        .map_err(|error| anyhow::anyhow!("{group} warmup was not accepted: {error}"))?;

    for stream in &mut streams {
        await_finish(core, group, stream)?;
    }
    Ok(())
}

/// Reads one request's events until its terminal event.
fn await_finish(core: &Warm, group: &str, stream: &mut EventRx) -> anyhow::Result<()> {
    loop {
        match stream.try_recv() {
            Ok(Event::Finished { reason, .. }) => {
                anyhow::ensure!(
                    !matches!(reason, FinishReason::Error | FinishReason::Aborted),
                    "{group} warmup finished with {reason:?}"
                );
                return Ok(());
            }
            Ok(Event::Rejected { message, .. }) => {
                anyhow::bail!("{group} warmup was rejected: {message}")
            }
            Ok(Event::Error { message }) => {
                anyhow::bail!("{group} warmup failed: {message}")
            }
            Ok(_) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                anyhow::ensure!(!core.is_dead(), "the engine died during {group} warmup");
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                anyhow::bail!("the {group} warmup stream closed before its request finished")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{GenerationFeatures, GenerationLimits};
    use ipc::CanvasSampling;

    use super::*;

    /// The launch arguments of a deployment whose ranks run on a CUDA device.
    fn cuda_args() -> EngineArgs {
        let mut args = testkit::args("warmup");
        args.engine_config.parallel.worker_groups[0].ranks[0].device = "cuda:0".to_owned();
        args
    }

    /// The warm-up's groups of a deployment whose rank reports `info`, admitted
    /// under `limits`.
    ///
    /// The requests are stated without being submitted: what a deployment's
    /// shapes ask for is checked here, and that they run when admission opens
    /// in the serving cases.
    fn groups_of(
        args: &EngineArgs,
        info: &ipc::MainProcessInfo,
        limits: &GenerationLimits,
    ) -> Vec<Group> {
        let mut id = 0;
        groups(args, info, limits, || {
            id += 1;
            ipc::RequestId(id)
        })
        .expect("the deployment's warm-up requests are stated")
    }

    #[test]
    fn a_token_deployment_warms_a_prefill_and_a_decode() {
        let args = cuda_args();
        let info = testkit::rank("localhost");
        let limits = GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING,
            ..args.engine_config.model.generation_limits.clone()
        };
        let groups = groups_of(&args, &info, &limits);
        assert_eq!(groups.len(), 1, "one group: the token runtime's");
        assert_eq!(groups[0].what, "token");
        // A one-token prompt and two generated tokens: the prefill, then a
        // decode that consumes its token.
        let [request] = &groups[0].requests[..] else {
            panic!("the token warm-up is one request");
        };
        assert_eq!(request.prompt_token_ids.len(), 1);
        assert_eq!(request.max_und_tokens, 2);
    }

    #[test]
    fn a_cpu_deployment_runs_no_warmup() {
        // `testkit::args` runs on CPU, where no serving path is captured: the
        // warm-up submits nothing, so its command channel is never read.
        let args = testkit::args("warmup");
        let (commands, _unread) = crossbeam_channel::unbounded();
        let intent = Warm {
            info: testkit::rank("localhost"),
            generation_limits: args.engine_config.model.generation_limits.clone(),
            next_id: Default::default(),
            dead: Default::default(),
            failure: Default::default(),
            commands: crate::CommandSender::new(commands),
        };
        warm(&intent, &args).expect("a CPU deployment has nothing to warm");
    }

    #[test]
    fn a_diffusion_lm_steps_at_every_row_count_and_commits_a_block() {
        let canvas = CanvasSampling {
            canvas_length: 16,
            max_steps: 2,
            entropy_bound: 0.1,
            t_min: 0.4,
            t_max: 0.8,
            confidence_threshold: 0.005,
            stability_threshold: 1,
        };
        // The rank prefills prompts and denoises blocks, three rows a step.
        let info = ipc::MainProcessInfo {
            supported_calls: vec![
                CallKind::Forward(ForwardMode::Prefill),
                CallKind::Forward(ForwardMode::TokenDenoising),
            ],
            diffusion_lm_step_rows: 3,
            ..testkit::rank("localhost")
        };
        let mut args = cuda_args();
        let model = &mut args.engine_config.model;
        model.generation_limits = GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING | GenerationFeatures::TOKEN_DENOISING,
            ..model.generation_limits.clone()
        };
        model.canvas_sampling = Some(canvas);
        let limits = model.generation_limits.clone();

        let groups = groups_of(&args, &info, &limits);
        let rows: Vec<usize> = groups
            .iter()
            .filter(|group| group.what == "diffusion LM")
            .map(|group| group.requests.len())
            .collect();
        assert_eq!(
            rows,
            [3, 2, 1],
            "every row count from the rank's largest down to one, as its own group"
        );
        // Each group asks for its own number of blocks; the one-row request
        // asks for two, so its committed block runs the prefill of a commit.
        let blocks: Vec<usize> = groups
            .iter()
            .filter(|group| group.what == "diffusion LM")
            .map(|group| group.requests[0].max_und_tokens / canvas.canvas_length as usize)
            .collect();
        assert_eq!(blocks, [1, 1, 2]);
    }
}
