//! Demonstrates concurrent text and image requests through the simulated engine.
//!
//! The example exercises scheduling and lifecycle events without requiring a
//! model worker or GPU. Three text-only and two image-only requests share one
//! scheduler thread over `SimExecutor`; the run passes when every request
//! finishes within the deadline, each text request emits at least one text
//! token, and each image request emits at least one image. The process exits
//! with status 1 on failure, and reports an error when the simulation cannot
//! start or a request is refused.

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use engine::Event;
use engine::{
    CommandWaker, Executor, Scheduler, SchedulerConfig, SchedulerHandle, SimEngine, SimExecutor,
};
use engine::{
    FeedbackNextToken, FeedbackSource, GenerationConstraint, GenerationLimits, GenerationRequest,
    ImageEncoderInput, ImageGenerationConfig, ImageTrigger, ModelDtype,
};
use ipc::{ImageParams, RequestId, SamplingParams};

fn main() -> anyhow::Result<()> {
    let mut sim = SimEngine::new();
    // Two batches may be in flight at once, so a step never waits on the one
    // before it to be retired.
    sim.set_queue_depth(2);
    let executor = Box::new(SimExecutor::new(sim));
    let capacity = engine::BatchCapacity::of(&executor.info().runtime_info()?);
    let mut sched = Scheduler::new(
        engine::Submissions::new(executor),
        SimEngine::special_tokens(),
        SchedulerConfig {
            max_batch: 32,
            ..Default::default()
        },
        ModelDtype::BFloat16,
        SimEngine::generation_limits(),
    )?;

    // A scheduler with no policy decides nothing — no plan, no admission, no
    // batch — so a world that serves must be given the default one: `fcfs`
    // revision one, instantiated for the capacity this pool was built for.
    let (program, module) = engine::packaged_policy(capacity);
    sched
        .install_policy(program, &engine::PolicyBytes::new(module))
        .map_err(|error| anyhow::anyhow!("the fcfs policy is installed: {error}"))?;

    // Prefix reuse is a world setting (`EngineConfig::prefix_cache`), off in a
    // scheduler built by hand: the `reuse` scenario is about the pages a second
    // request reads, so it is turned on as a deployment turns it on.
    sched.set_prefix_reuse(true);

    // The scheduler runs on its own thread; this thread submits through the
    // handle and drains each request's event receiver, as a server would.
    let handle = SchedulerHandle::spawn(sched, "sim-model", CommandWaker::noop())?;

    let mut rxs: HashMap<RequestId, (String, engine::EventRx)> = HashMap::new();
    let mut next_id = 1u64;
    let mut mk = |constraint: GenerationConstraint, generated: Vec<u32>, prompt: Vec<u32>| {
        let id = RequestId(next_id);
        next_id += 1;
        let policy = ImageGenerationConfig {
            trigger: ImageTrigger::Token { token_id: 1000 },
            requires_text_for_image: false,
            feedback_source: Some(FeedbackSource::DeviceProduct),
            feedback_next_token: FeedbackNextToken::EndOfImage,
            num_feedback_positions: 2,
            feedback_encoders: vec![ImageEncoderInput {
                encoder: engine::ImageEncoder::Vision,
                num_cache_tokens: None,
                max_cache_tokens: None,
                height: None,
                width: None,
            }],
            sample_feedback_continuation: true,
        };
        let prompt_token_ids = prompt;

        let image = ImageParams {
            steps: 6,
            ..Default::default()
        };
        let cache = Default::default();
        // These limits only check the request locally before submission; the
        // scheduler resolves its limits separately against the executor.
        let limits = GenerationLimits {
            features: engine::GenerationFeatures::UNDERSTANDING
                | engine::GenerationFeatures::VISION_ENCODE
                | engine::GenerationFeatures::IMAGE_GENERATION,
            max_latent_units: 65_536,
            latent_downsample: 16,
            max_vae_grid_tokens: 65_536,
            max_vit_grid_tokens: 8_192,
            latent_bytes_per_unit: (256 << 20) / 65_536,
            max_latent_feature_bytes: 256 << 20,
            max_vision_feature_bytes: 256 << 20,
            commit_marker_tokens: 2,
            max_cfg_branches: 3,
            encoder_cache_entries: 256,
        };
        let request = GenerationRequest {
            request_id: id,
            prompt_token_ids,
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint,
            sampling: SamplingParams::default(),
            image,
            max_images: 1,
            max_und_tokens: 16,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache,
            image_generation: policy,
            readout: Vec::new(),
            generated_token_ids: generated,
            adapter: 0,
            generated_images: Vec::new(),
            trajectory: Default::default(),
        };
        request.validate_resources(&limits)?;
        anyhow::Ok((id, request))
    };

    // A scenario argument narrows the world to the kinds it names, so a
    // failure can be attributed to one kind of request rather than to the mix.
    let wanted = std::env::args().nth(1);
    let wants = |kind: &str| wanted.as_deref().is_none_or(|only| only == kind || only == "all");
    if wants("text") {
        for _ in 0..3 {
            let (id, request) = mk(GenerationConstraint::UndOnly, Vec::new(), vec![1, 2, 3, 4, 5])?;
            let rx = handle.submit(request)?;
            rxs.insert(id, ("text".to_string(), rx));
        }
    }
    if wants("image") {
        for _ in 0..2 {
            let (id, request) = mk(GenerationConstraint::GenOnly, Vec::new(), vec![1, 2, 3, 4, 5])?;
            let rx = handle.submit(request)?;
            rxs.insert(id, ("image".to_string(), rx));
        }
    }
    // A request that resumes a generation: its prompt continues the tokens it
    // carries, so its context is prefilled from `prompt + carried` and its
    // budget, penalties and stop rules continue from the carried token count.
    // Two requests over one 600-token prompt, the second submitted only after
    // the first finished: it should read the pages the first stored, up to the
    // last boundary that leaves a token to compute. Submitted together, the
    // second arrives before the first's pages are published and reads nothing,
    // which says nothing about reuse.
    // The reuse scenario is sequential, so it runs only when it is the one named:
    // `all` is the concurrent world.
    if wanted.as_deref() == Some("reuse") {
        let prompt: Vec<u32> = (0..600).map(|index| 2_000 + index % 500).collect();
        for id in 1..=2 {
            let (_, request) = mk(GenerationConstraint::UndOnly, Vec::new(), prompt.clone())?;
            let mut rx = handle.submit(request)?;
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            let mut finished = false;
            while !finished && std::time::Instant::now() < deadline {
                match rx.try_recv() {
                    Ok(event) if event.is_terminal() => finished = true,
                    Ok(_) => {}
                    Err(_) => thread::sleep(Duration::from_millis(1)),
                }
            }
            anyhow::ensure!(finished, "reuse request {id} did not finish");
        }
        let stats = handle.stats();
        println!(
            "prefix: query_tokens={} hit_tokens={} pages_stored={}",
            stats.prefix.query_tokens.load(std::sync::atomic::Ordering::Relaxed),
            stats.prefix.hit_tokens.load(std::sync::atomic::Ordering::Relaxed),
            stats.prefix_cache.pages_stored.load(std::sync::atomic::Ordering::Relaxed),
        );
        handle.shutdown();
        println!("\nRUST SIM CONTROL-PLANE: PASS");
        std::process::exit(0);
    }
    if wants("migrated") {
        let (id, request) = mk(GenerationConstraint::UndOnly, vec![1_500, 1_501], vec![1, 2, 3, 4, 5])?;
        let rx = handle.submit(request)?;
        rxs.insert(id, ("migrated".to_string(), rx));
    }

    // Poll every event receiver until each request has reported `Finished`
    // once, or the deadline passes. After the kind label, the tuple counts
    // text tokens and images and records whether `Finished` was seen.
    let mut done = 0usize;
    let total = rxs.len();
    anyhow::ensure!(total > 0, "the scenario names no request kind: {wanted:?}");
    let mut counts: HashMap<RequestId, (String, usize, usize, bool)> = HashMap::new();
    let mut watched = 0usize;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while done < total && std::time::Instant::now() < deadline {
        for (id, (kind, rx)) in rxs.iter_mut() {
            while let Ok(ev) = rx.try_recv() {
                watched += 1;
                let e = counts.entry(*id).or_insert((kind.clone(), 0, 0, false));
                match ev {
                    Event::TextToken { .. } => e.1 += 1,
                    Event::ImageDone { .. } => e.2 += 1,
                    Event::Finished { .. } if !e.3 => {
                        e.3 = true;
                        done += 1;
                    }
                    Event::Rejected { kind, message } => {
                        println!("req {id:?} rejected ({kind:?}): {message}");
                    }
                    Event::Error { message, .. } => println!("req {id:?} failed: {message}"),
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(2));
    }

    let stats = handle.stats();
    let general = &stats.general;
    println!(
        "events={watched} steps={} running={} in_flight={} dead={} failure={:?}",
        general.steps.load(std::sync::atomic::Ordering::Relaxed),
        general.running.load(std::sync::atomic::Ordering::Relaxed),
        general.in_flight.load(std::sync::atomic::Ordering::Relaxed),
        handle.is_dead(),
        handle.failure(),
    );
    let mut ok = done == total;
    for (id, (kind, toks, imgs, fin)) in &counts {
        println!(
            "req {:?} [{}]: text_tokens={} images={} finished={}",
            id, kind, toks, imgs, fin
        );
        if kind == "text" && *toks == 0 {
            ok = false;
        }
        if kind == "image" && *imgs == 0 {
            ok = false;
        }
        if kind == "migrated" && (*toks == 0 || !*fin) {
            ok = false;
        }
    }
    handle.shutdown();
    println!(
        "\nRUST SIM CONTROL-PLANE: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
