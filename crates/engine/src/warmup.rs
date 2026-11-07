//! Serving warmup through the scheduler.
//!
//! Once the epoch's first policy serves, the engine admits real requests of the
//! shapes the executors prepared for, so the serving path of every captured row
//! count, guidance-branch count and diffusion LM step has run before the first
//! client request that needs it.
//!
//! It runs after the first policy, not at startup, because it is the serving
//! path that is warmed and the serving path is the policy's: a scheduler with no
//! policy forms no batch, so a warm-up at startup would wait forever for
//! requests nothing could decide. Startup therefore records the intent
//! ([`after_serving`]) and does not wait for it, and a warm-up that fails is
//! logged while serving continues — a device that cannot run these shapes fails
//! the first client request that needs them, in the open. The scheduler and its allocators form every batch
//! as they form a client's; a rank states only the shapes
//! (`MainProcessInfo::image_warmups`, `diffusion_lm_step_rows`) and takes no part
//! in forming them. The ranks warmed their kernels and captured their graphs
//! before they reported; these requests warm the serving path around them.
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
//!   commit;
//! - for each image shape, one image request of two denoising steps at the
//!   guidance scales that realize the shape's branch count, which also
//!   decodes its image. The scheduler lets one request at a time hold the
//!   denoiser, so serving never batches denoising calls and one request
//!   reaches every shape it can run.
//!
//! No request reads or populates the prefix or encoder caches. A request the
//! deployment's generation limits would refuse a client, such as a guidance
//! branch count above the model's bound or a raster above its latent bound,
//! is not one serving can reach and is left out.

use std::time::Duration;

use crate::{
    CachePolicy, Event, FeedbackNextToken, FinishReason, GenerationConstraint, GenerationRequest,
    ImageGenerationConfig, ImageTrigger, MultimodalInputs, Request,
};
use anyhow::Context as _;
use ipc::{CallKind, ForwardMode, MediaCall};
use ipc::{ImageParams, SamplingParams};

use crate::core::{EngineConfig, SchedulerHandle};
use crate::handle::EventRx;

/// Denoising steps of a warmup image: the fewest that run a step from a
/// prepared latent and a step from a denoised one.
const IMAGE_STEPS: u16 = 2;

/// Runs the warmup requests the executors' reports call for on `core`.
///
/// Returns once every request has finished. Fails, naming the group, when a
/// request is rejected, ends in an error, or the engine dies first.
///
/// Warmup runs only where the executors execute on CUDA devices: a CPU or stub
/// executor has no kernels or graphs to warm.
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

    /// A request id of this scheduler's, so warm-up requests are numbered with
    /// a client's.
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

/// Runs the warm-up once the epoch's first policy serves, on a thread of its own
/// ([`Warm`] is what it holds).
pub(crate) fn after_serving(core: &SchedulerHandle, config: &EngineConfig) {
    let intent = core.warmup();
    let serving = core.serving();
    let config = config.clone();
    let spawned = std::thread::Builder::new()
        .name("warmup".to_owned())
        .spawn(move || {
            wait_serving(serving);
            if let Err(error) = warm(&intent, &config) {
                // Serving continues: the shapes this warms are the ones a client
                // request reaches anyway, where its failure is visible.
                tracing::warn!(%error, "the serving path's warm-up failed; the engine serves on");
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "the warm-up thread could not start; the engine serves on");
    }
}

/// Blocks until a policy instance serves, given the scheduler's serving signal.
///
/// The signal is the scheduler's `watch`, polled: the waiters are ordinary
/// threads (the warm-up's), and a closed channel means the scheduler is gone.
pub(crate) fn wait_serving(mut serving: tokio::sync::watch::Receiver<bool>) {
    loop {
        if *serving.borrow_and_update() {
            return;
        }
        if serving.has_changed().is_err() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// One group of warm-up requests: the shapes one part of the serving path runs,
/// submitted together and awaited together.
pub(crate) struct Group {
    /// What the group warms, for the log: a token model's prefill and decode,
    /// a diffusion LM's row count, an image shape.
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
    config: &EngineConfig,
    info: &ipc::MainProcessInfo,
    limits: &crate::GenerationLimits,
    mut next_id: impl FnMut() -> ipc::RequestId,
) -> anyhow::Result<Vec<Group>> {
    let supports = |kind| info.supported_calls.contains(&kind);
    // Keeps the requests clients could submit, naming what it leaves out.
    let servable = |what: &str, requests: Vec<GenerationRequest>| {
        // A deployment with block sampling generates every request but a
        // readout in canvases, so admission checks its canvas features.
        let canvases = config.canvas_sampling.is_some();
        let (kept, refused): (Vec<_>, Vec<_>) = requests.into_iter().partition(|request| {
            let needs = if canvases && !request.is_readout() {
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
    let mut und = |tokens: usize| {
        warmup_request(
            config,
            next_id(),
            GenerationConstraint::UndOnly,
            tokens,
            ImageParams::default(),
            Vec::new(),
            0,
        )
    };

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
        let sampling = config.canvas_sampling.context(
            "executors report diffusion LM step rows but the deployment has no block sampling",
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

    let images = supports(CallKind::Media(MediaCall::LatentPreparation))
        && supports(CallKind::Media(MediaCall::Denoising));
    if images {
        for shape in &info.image_warmups {
            let (cfg_text_scale, cfg_img_scale) = ImageParams::guidance_scales(shape.branches)
                .with_context(|| {
                    format!(
                        "an image warmup shape has {} guidance branches",
                        shape.branches
                    )
                })?;
            let image = ImageParams {
                steps: IMAGE_STEPS,
                cfg_text_scale,
                cfg_img_scale,
                height: shape.height,
                width: shape.width,
                seed: Some(0),
                ..ImageParams::default()
            };
            let what = format!(
                "{}x{} image of {} branches",
                shape.width, shape.height, shape.branches
            );
            let requests = servable(
                &what,
                vec![warmup_request(
                    config,
                    next_id(),
                    GenerationConstraint::GenOnly,
                    0,
                    image,
                    shape.negative_token_ids.clone(),
                    1,
                )],
            );
            if !requests.is_empty() {
                groups.push(Group { what, requests });
            }
        }
    }
    Ok(groups)
}

pub(crate) fn warm(core: &Warm, config: &EngineConfig) -> anyhow::Result<()> {
    let on_cuda = config
        .worker_groups
        .iter()
        .flat_map(|worker| &worker.ranks)
        .any(|rank| rank.device == "cuda" || rank.device.starts_with("cuda:"));
    if !on_cuda {
        return Ok(());
    }

    let info = core.info().clone();
    let limits = core.generation_limits();
    for group in groups(config, &info, &limits, || core.next_request_id())? {
        run(core, &group.what, group.requests)?;
        tracing::info!(warmup = %group.what, "completed a serving path warm-up group");
    }
    Ok(())
}

/// Builds one warmup generation request.
///
/// `tokens` is the number of understanding tokens it generates; an image
/// request generates none and starts its image directly after the prompt.
/// `negative_token_ids` are the request's negative conditioning tokens: for
/// a guided image, the rank's framing of an empty negative prompt, so the
/// request prefills the guidance prefix a client's request without a
/// negative prompt does.
fn warmup_request(
    config: &EngineConfig,
    request_id: ipc::RequestId,
    constraint: GenerationConstraint,
    tokens: usize,
    image: ImageParams,
    negative_token_ids: Vec<u32>,
    images: u16,
) -> GenerationRequest {
    GenerationRequest {
        request_id,
        prompt_token_ids: vec![config.bos],
        negative_prompt_token_ids: negative_token_ids,
        multimodal_inputs: MultimodalInputs::default(),
        constraint,
        sampling: SamplingParams {
            temperature: 0.0,
            ignore_eos: true,
            // A diffusion LM request needs a seed; the rest ignore it.
            seed: Some(0),
            ..SamplingParams::default()
        },
        image,
        // The images the request asks for: one for an image shape, none for a
        // request that only generates tokens, since a text deployment has no
        // image path to run and no shape to read.
        max_images: images,
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
        image_generation: ImageGenerationConfig {
            trigger: ImageTrigger::Disabled,
            requires_text_for_image: false,
            feedback_source: None,
            feedback_encoders: Vec::new(),
            num_feedback_positions: 0,
            feedback_next_token: FeedbackNextToken::None,
            sample_feedback_continuation: false,
        },
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
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
    use ipc::ImageWarmup;

    use super::*;
    use crate::sim::{SimEngine, SimExecutor};

    /// The warm-up's groups on `sim`, which reports running on a CUDA device, in
    /// the deployment `deployment` adjusts.
    ///
    /// The requests are stated without being submitted: a scheduler with no
    /// policy forms no batch, so a warm-up only runs once the epoch's first
    /// policy serves, and what a deployment's shapes ask for is checked here.
    fn groups_of(sim: SimEngine, deployment: impl FnOnce(&mut EngineConfig)) -> Vec<Group> {
        let mut config = EngineConfig::sim("warmup");
        config.worker_groups[0].ranks[0].device = "cuda:0".to_owned();
        deployment(&mut config);
        let executor = SimExecutor::new(sim);
        let core = SchedulerHandle::with_executor(config.clone(), Box::new(executor)).unwrap();
        let mut id = 0;
        let groups = groups(&config, core.info(), &core.generation_limits(), || {
            id += 1;
            ipc::RequestId(id)
        })
        .expect("the deployment's warm-up requests are stated");
        core.shutdown();
        groups
    }

    fn shape(height: u32, width: u32, branches: u8, negative_token_ids: Vec<u32>) -> ImageWarmup {
        ImageWarmup {
            height,
            width,
            branches,
            negative_token_ids,
        }
    }

    /// A simulated rank that serves images, prepared for `shapes`.
    fn image_rank(shapes: Vec<ImageWarmup>) -> SimEngine {
        let mut sim = SimEngine::new();
        sim.mut_info_for_test().image_warmups = shapes;
        sim
    }

    #[test]
    fn a_token_deployment_warms_a_prefill_and_a_decode() {
        let groups = groups_of(SimEngine::new(), |_| {});
        assert_eq!(groups.len(), 1, "one group: the token runtime's");
        assert_eq!(groups[0].what, "token");
        // A one-token prompt and two generated tokens: the prefill, then a
        // decode that consumes its token.
        let [request] = &groups[0].requests[..] else {
            panic!("the token warm-up is one request");
        };
        assert_eq!(request.prompt_token_ids.len(), 1);
        assert_eq!(request.max_und_tokens, 2);
        assert_eq!(request.max_images, 0, "no image in a token request");
    }

    #[test]
    fn a_cpu_deployment_runs_no_warmup() {
        // `EngineConfig::sim` runs on CPU, where no serving path is captured.
        let config = EngineConfig::sim("warmup");
        let executor = SimExecutor::new(SimEngine::new());
        let core = SchedulerHandle::with_executor(config.clone(), Box::new(executor)).unwrap();
        let intent = core.warmup();
        let result = warm(&intent, &config);
        core.shutdown();
        result.expect("a CPU deployment has nothing to warm");
    }

    #[test]
    fn each_image_shape_runs_both_denoising_steps_and_decodes_its_image() {
        let groups = groups_of(
            image_rank(vec![
                shape(64, 64, 1, Vec::new()),
                shape(64, 128, 2, vec![11, 12]),
                shape(128, 64, 3, vec![11, 12]),
            ]),
            |_| {},
        );

        // The token runtime's group comes first; the shapes follow, one group
        // each, because the denoiser runs one request at a time.
        let images: Vec<&Group> = groups
            .iter()
            .filter(|group| group.what != "token")
            .collect();
        assert_eq!(images.len(), 3, "one group per shape");
        for group in &images {
            let [request] = &group.requests[..] else {
                panic!("an image warm-up group holds one request");
            };
            assert_eq!(request.image.steps, IMAGE_STEPS, "every denoising step");
            assert_eq!(request.max_images, 1, "and one decoded image");
        }
        let shapes: Vec<(u32, u32)> = images
            .iter()
            .map(|group| {
                let request = &group.requests[0];
                (request.image.height, request.image.width)
            })
            .collect();
        assert_eq!(
            shapes,
            [(64, 64), (64, 128), (128, 64)],
            "the rank's prepared shapes, each once, in the order it prepared them"
        );
    }

    #[test]
    fn a_guided_shape_admits_the_negative_tokens_its_rank_frames() {
        // The rank conditions the text-unconditional branch on exactly the
        // admitted negative tokens, and the engine sizes that prefix from them,
        // so the request carries the model's framing of an empty negative
        // prompt as a client's request does.
        let groups = groups_of(image_rank(vec![shape(64, 64, 2, vec![11, 12, 13])]), |_| {});
        let image = groups
            .iter()
            .find(|group| group.what == "64x64 image of 2 branches")
            .expect("the rank's guided shape is warmed");
        assert_eq!(
            image.requests[0].negative_prompt_token_ids,
            [11, 12, 13],
            "the shape's negative tokens travel in the request"
        );
    }

    #[test]
    fn a_shape_the_limits_refuse_is_not_warmed() {
        let groups = groups_of(
            image_rank(vec![
                shape(64, 64, 1, Vec::new()),
                shape(64, 64, 3, vec![11]),
            ]),
            // At most two guidance branches: no client can submit the
            // three-branch shape either.
            |config| config.generation_limits.max_cfg_branches = 2,
        );
        let images: Vec<&str> = groups
            .iter()
            .map(|group| group.what.as_str())
            .filter(|what| *what != "token")
            .collect();
        assert_eq!(
            images,
            ["64x64 image of 1 branches"],
            "the shape the deployment's limits refuse is left out"
        );
    }

    #[test]
    fn a_diffusion_lm_steps_at_every_row_count_and_commits_a_block() {
        let mut sim = SimEngine::new();
        sim.mut_info_for_test().supported_calls = vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::TokenDenoising),
        ];
        sim.mut_info_for_test().diffusion_lm_step_rows = 3;
        sim.set_queue_depth(8);
        sim.set_block_size(16);
        let canvas = CanvasSampling {
            canvas_length: 16,
            max_steps: 2,
            entropy_bound: 0.1,
            t_min: 0.4,
            t_max: 0.8,
            confidence_threshold: 0.005,
            stability_threshold: 1,
        };
        // The ranks run under the deployment's block sampling.
        sim.set_canvas_sampling(canvas);

        let groups = groups_of(sim, |config| {
            config.generation_limits = GenerationLimits {
                features: GenerationFeatures::UNDERSTANDING | GenerationFeatures::TOKEN_DENOISING,
                ..config.generation_limits.clone()
            };
            config.canvas_sampling = Some(canvas);
        });

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
        // Each group asks for its own number of blocks of the canvas; the
        // one-row request asks for two, so its committed block runs the prefill
        // of a commit.
        let blocks: Vec<usize> = groups
            .iter()
            .filter(|group| group.what == "diffusion LM")
            .map(|group| group.requests[0].max_und_tokens / canvas.canvas_length as usize)
            .collect();
        assert_eq!(blocks, [1, 1, 2]);
    }
}
