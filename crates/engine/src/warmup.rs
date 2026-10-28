//! Startup warmup through the scheduler.
//!
//! Once the workers have reported, the engine admits real requests of the
//! shapes they prepared for, so the serving path of every captured row
//! count, guidance-branch count and diffusion LM step has run before the
//! first client request. The scheduler and its allocators form every batch
//! as they form a client's; a rank states only the shapes
//! (`WorkerInfo::image_warmups`, `diffusion_lm_step_rows`) and takes no part
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

/// Runs the warmup requests the workers' reports call for on `core`.
///
/// Returns once every request has finished. Fails, naming the group, when a
/// request is rejected, ends in an error, or the engine dies first.
///
/// Warmup runs only where the workers execute on CUDA devices: a CPU or stub
/// worker has no kernels or graphs to warm.
pub(crate) fn warm(core: &SchedulerHandle, config: &EngineConfig) -> anyhow::Result<()> {
    let on_cuda = config
        .worker_groups
        .iter()
        .flat_map(|worker| &worker.ranks)
        .any(|rank| rank.device == "cuda" || rank.device.starts_with("cuda:"));
    if config.worker_process.stub || !on_cuda {
        return Ok(());
    }

    let info = core.info();
    let limits = core.generation_limits();
    let supports = |kind| info.supported_calls.contains(&kind);
    let request = |tokens: usize| {
        warmup_request(
            config,
            core.next_request_id(),
            GenerationConstraint::UndOnly,
            tokens,
            ImageParams::default(),
            Vec::new(),
        )
    };
    // Keeps the requests clients could submit, naming what it leaves out.
    let servable = |what: &str, requests: Vec<GenerationRequest>| {
        // A deployment with block sampling generates every request but a
        // readout in canvases, so admission checks its canvas features.
        let canvases = config.worker_process.canvas_sampling.is_some();
        let (kept, refused): (Vec<_>, Vec<_>) = requests.into_iter().partition(|request| {
            let needs = if canvases && !request.is_readout() {
                request.canvas_features()
            } else {
                request.required_features()
            };
            limits.features.contains(needs) && request.validate_resources(&limits).is_ok()
        });
        if !refused.is_empty() {
            tracing::info!(
                "left out {} {what} warmup requests the deployment's generation limits refuse",
                refused.len()
            );
        }
        kept
    };

    // The token request: a prefill followed by a decode over its token. A
    // diffusion LM deployment, whose limits grant no understanding feature,
    // generates in blocks instead, which its groups below run.
    let decodes = supports(CallKind::Forward(ForwardMode::Prefill))
        && supports(CallKind::Forward(ForwardMode::Decode));
    if decodes {
        let requests = servable("token", vec![request(2)]);
        if !requests.is_empty() {
            run(core, "token", requests)?;
            tracing::info!("completed token runtime warmup");
        }
    }

    if info.diffusion_lm_step_rows > 0 {
        let sampling = config.worker_process.canvas_sampling.context(
            "workers report diffusion LM step rows but the deployment has no block sampling",
        )?;
        let length = sampling.canvas_length as usize;
        // Requests of different lengths stop stepping together once the
        // shortest finishes: each commits its block in its own batch, and
        // whether two of them step together again depends on timing. So each
        // row count is its own group of equal requests.
        let mut warmed = false;
        for rows in (1..=info.diffusion_lm_step_rows as usize).rev() {
            let blocks = if rows == 1 { 2 } else { 1 };
            let requests = servable(
                "diffusion LM",
                // The scheduler generates every such request in canvases of
                // the deployment's block sampling.
                (0..rows).map(|_| request(blocks * length)).collect(),
            );
            if !requests.is_empty() {
                run(core, "diffusion LM", requests)?;
                warmed = true;
            }
        }
        if warmed {
            tracing::info!("completed diffusion LM generation warmup");
        }
    }

    let images = supports(CallKind::Media(MediaCall::LatentPreparation))
        && supports(CallKind::Media(MediaCall::Denoising));
    if images {
        let mut warmed = false;
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
                    core.next_request_id(),
                    GenerationConstraint::GenOnly,
                    0,
                    image,
                    shape.negative_token_ids.clone(),
                )],
            );
            if !requests.is_empty() {
                run(core, "image", requests)?;
                warmed = true;
            }
        }
        if warmed {
            tracing::info!("completed image runtime warmup");
        }
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
        // One generated image, the budget a client's image request states.
        max_images: 1,
        max_und_tokens: tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy {
            read: false,
            write: false,
            isolation_key: None,
            state_seed: None,
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
    }
}

/// Submits `requests` together and waits for each to finish.
///
/// The requests reach the scheduler in one command, so it queues all of them
/// before it forms its next batch, as it would concurrent clients that all
/// arrived before that batch.
fn run(
    core: &SchedulerHandle,
    group: &str,
    requests: Vec<GenerationRequest>,
) -> anyhow::Result<()> {
    let mut streams = core
        .submit_all(requests.into_iter().map(Request::from).collect())
        .map_err(|error| anyhow::anyhow!("{group} warmup was not accepted: {error}"))?;

    for stream in &mut streams {
        await_finish(core, group, stream)?;
    }
    Ok(())
}

/// Reads one request's events until its terminal event.
fn await_finish(core: &SchedulerHandle, group: &str, stream: &mut EventRx) -> anyhow::Result<()> {
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
    use std::collections::BTreeSet;

    use crate::{GenerationFeatures, GenerationLimits};
    use ipc::CanvasSampling;
    use ipc::{BatchCommand, ImageWarmup};

    use super::*;
    use crate::executor::ExecutionBatch;
    use crate::sim::{BatchEvent, SimEngine, SimExecutor};

    /// Runs `warm` on `sim`, reported to run on a CUDA device, in the
    /// deployment `deployment` adjusts, and returns every batch the executor
    /// accepted.
    fn warmed(sim: SimEngine, deployment: impl FnOnce(&mut EngineConfig)) -> Vec<ExecutionBatch> {
        let mut executor = SimExecutor::new(sim);
        let boundary = executor.observe();
        let mut config = EngineConfig::sim("warmup");
        config.worker_groups[0].ranks[0].device = "cuda:0".to_owned();
        deployment(&mut config);
        let core = SchedulerHandle::with_executor(config.clone(), Box::new(executor)).unwrap();

        let result = warm(&core, &config);
        core.shutdown();
        result.unwrap();
        boundary
            .try_iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(batch),
                BatchEvent::Resolved { .. } => None,
            })
            .collect()
    }

    /// The batches that ran calls of `kind`, by their call count.
    fn calls(batches: &[ExecutionBatch], kind: CallKind) -> Vec<usize> {
        batches
            .iter()
            .map(|batch| {
                batch
                    .requests
                    .iter()
                    .filter(|(call, _)| call.kind == kind)
                    .count()
            })
            .filter(|&count| count > 0)
            .collect()
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
    fn a_token_deployment_runs_a_prefill_and_a_decode() {
        let batches = warmed(SimEngine::new(), |_| {});
        assert_eq!(
            calls(&batches, CallKind::Forward(ForwardMode::Prefill)),
            [1]
        );
        assert_eq!(calls(&batches, CallKind::Forward(ForwardMode::Decode)), [1]);
        assert!(calls(&batches, CallKind::Media(MediaCall::Denoising)).is_empty());
    }

    #[test]
    fn a_cpu_deployment_runs_no_warmup() {
        let mut executor = SimExecutor::new(SimEngine::new());
        let boundary = executor.observe();
        let config = EngineConfig::sim("warmup");
        let core = SchedulerHandle::with_executor(config.clone(), Box::new(executor)).unwrap();

        let result = warm(&core, &config);
        core.shutdown();
        result.unwrap();
        assert_eq!(boundary.try_iter().count(), 0);
    }

    #[test]
    fn each_image_shape_runs_both_denoising_steps_and_decodes_its_image() {
        let sim = image_rank(vec![
            shape(64, 64, 1, Vec::new()),
            shape(64, 128, 2, vec![11, 12]),
            shape(128, 64, 3, vec![11, 12]),
        ]);
        let batches = warmed(sim, |_| {});

        // The denoiser runs one request at a time: one call per batch,
        // two steps and one decode per shape.
        assert_eq!(
            calls(&batches, CallKind::Media(MediaCall::Denoising)),
            [1; 6]
        );
        assert_eq!(
            calls(&batches, CallKind::Media(MediaCall::ImageDecoding)),
            [1; 3]
        );
    }

    #[test]
    fn a_guided_shape_admits_the_negative_tokens_its_rank_frames() {
        let sim = image_rank(vec![shape(64, 64, 2, vec![11, 12, 13])]);

        // The rank conditions the text-unconditional branch on exactly the
        // admitted negative tokens, and the engine sizes that prefix from
        // them, so the request carries the model's framing of an empty
        // negative prompt as a client's request does.
        let admitted = warmed(sim, |_| {})
            .iter()
            .flat_map(|batch| &batch.commands)
            .filter_map(|command| match command {
                BatchCommand::Start { request } if request.image.is_some() => request
                    .generation
                    .as_ref()
                    .map(|generation| generation.negative_token_ids.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(admitted, [vec![11, 12, 13]]);
    }

    #[test]
    fn a_shape_the_limits_refuse_is_not_warmed() {
        let sim = image_rank(vec![
            shape(64, 64, 1, Vec::new()),
            shape(64, 64, 3, vec![11]),
        ]);

        // At most two guidance branches: no client can submit the
        // three-branch shape either.
        let batches = warmed(sim, |config| config.generation_limits.max_cfg_branches = 2);
        assert_eq!(
            calls(&batches, CallKind::Media(MediaCall::Denoising)),
            [1; 2]
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
        // The ranks are launched with the deployment's block sampling.
        sim.set_canvas_sampling(canvas);

        let batches = warmed(sim, |config| {
            config.generation_limits = GenerationLimits {
                features: GenerationFeatures::UNDERSTANDING | GenerationFeatures::TOKEN_DENOISING,
                ..config.generation_limits.clone()
            };
            config.worker_process.canvas_sampling = Some(canvas);
        });
        let steps = calls(&batches, CallKind::Forward(ForwardMode::TokenDenoising));
        assert_eq!(
            steps.iter().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([1, 2, 3])
        );
        // Each group prefills its prompts together; the one-row request's
        // first block is committed by a prefill of its own.
        let prefills = calls(&batches, CallKind::Forward(ForwardMode::Prefill));
        assert!(prefills.contains(&3) && prefills.contains(&2));
        assert!(prefills.iter().filter(|&&rows| rows == 1).count() >= 2);
    }
}
