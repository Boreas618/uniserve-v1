//! Demonstrates concurrent text requests through the scheduler over a fake
//! worker.
//!
//! The example exercises scheduling and lifecycle events without requiring a
//! model worker or GPU. Three text requests share one scheduler thread over
//! the production submission door, with `testkit::FakeBatches` responding to its
//! batches; the run passes when every request finishes within the
//! deadline and each text request emits at least one text token. The process exits
//! with status 1 on failure, and reports an error when the simulation cannot
//! start or a request is refused.

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use ipc::{RequestId, SamplingParams};
use scheduler::Event;
use scheduler::{
    CommandWaker, Scheduler, SchedulerConfig, SchedulerHandle, SpecialTokenIds, Submissions,
    WorkerInfo,
};
use scheduler::{GenerationLimits, GenerationRequest};

fn main() -> anyhow::Result<()> {
    // One text pool of one rank. Two batches may be in flight at once, so a
    // step never waits on the one before it to be retired.
    let mut pool = testkit::pool("sim");
    pool.layout.queue_depth = 2;
    let batches: std::sync::Arc<dyn ipc::execution::Batches> =
        std::sync::Arc::new(testkit::FakeBatches::new(vec![pool]));
    let capacity = scheduler::BatchCapacity::of(
        &WorkerInfo::from_inventory(batches.inventory())?.runtime_info()?,
    );
    let mut sched = Scheduler::new(
        Submissions::over_batches(batches)?,
        // The fake samples no EOS: every token it samples lies in
        // `testkit::SAMPLED`'s window, so each request ends at its budget.
        SpecialTokenIds {
            bos: 151_644,
            eos: vec![151_643],
        },
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            ..Default::default()
        },
        GenerationLimits {
            features: scheduler::GenerationFeatures::UNDERSTANDING,
            ..Default::default()
        },
    )?;

    // Prefix reuse is an epoch setting (`EngineConfig::prefix_cache`), off in a
    // scheduler built by hand: the `reuse` scenario is about the pages a second
    // request reads, so it is turned on as a deployment turns it on.
    sched.set_prefix_reuse(true);

    // The scheduler runs on its own thread; this thread submits through the
    // handle and drains each request's event receiver, as a server would.
    let handle = SchedulerHandle::spawn(sched, "sim-model", CommandWaker::noop())?;

    // A scheduler with no policy decides nothing, so an epoch that serves is
    // given the default one as a launch gives it: `fcfs` revision one for the
    // capacity this pool was built for, prepared by a precopy and served from
    // the switchover.
    let (program, module) = testkit::packaged_policy(capacity);
    let responded = |response: crossbeam_channel::Receiver<Result<(), String>>| {
        response
            .recv()
            .map_err(|_| anyhow::anyhow!("the scheduler stopped"))?
            .map_err(|error| anyhow::anyhow!("the fcfs policy is installed: {error}"))
    };
    responded(handle.precopy_policy(
        program,
        std::sync::Arc::new(scheduler::PolicyBytes::new(module)),
    ))?;
    responded(handle.switchover_policy())?;

    let mut rxs: HashMap<RequestId, (String, scheduler::EventRx)> = HashMap::new();
    let mut next_id = 1u64;
    let mut mk = |generated: Vec<u32>, prompt: Vec<u32>| {
        let id = RequestId(next_id);
        next_id += 1;
        let prompt_token_ids = prompt;

        let cache = Default::default();
        // These limits only check the request locally before submission; the
        // scheduler resolves its limits separately against the worker.
        let limits = GenerationLimits {
            features: scheduler::GenerationFeatures::UNDERSTANDING
                | scheduler::GenerationFeatures::VISION_ENCODE,
            max_vae_grid_tokens: 65_536,
            max_vit_grid_tokens: 8_192,
            max_latent_feature_bytes: 256 << 20,
            max_vision_feature_bytes: 256 << 20,
            encoder_cache_entries: 256,
            max_logprobs: 20,
        };
        let request = GenerationRequest {
            request_id: id,
            prompt_token_ids,
            multimodal_inputs: Default::default(),
            sampling: SamplingParams::default(),
            max_und_tokens: 16,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache,
            generated_token_ids: generated,
            adapter: 0,
        };
        request.validate_resources(&limits)?;
        anyhow::Ok((id, request))
    };

    // A scenario argument narrows the epoch to the kinds it names, so a
    // failure can be attributed to one kind of request rather than to the mix.
    let wanted = std::env::args().nth(1);
    let wants = |kind: &str| {
        wanted
            .as_deref()
            .is_none_or(|only| only == kind || only == "all")
    };
    if wants("text") {
        for _ in 0..3 {
            let (id, request) = mk(Vec::new(), vec![1, 2, 3, 4, 5])?;
            let rx = handle.submit(request)?;
            rxs.insert(id, ("text".to_string(), rx));
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
    // `all` is the concurrent epoch.
    if wanted.as_deref() == Some("reuse") {
        let prompt: Vec<u32> = (0..600).map(|index| 2_000 + index % 500).collect();
        for id in 1..=2 {
            let (_, request) = mk(Vec::new(), prompt.clone())?;
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
            stats
                .prefix
                .query_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
            stats
                .prefix
                .hit_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
            stats
                .prefix_cache
                .pages_stored
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        handle.shutdown();
        println!("\nRUST CONTROL-PLANE: PASS");
        std::process::exit(0);
    }
    if wants("migrated") {
        let (id, request) = mk(vec![1_500, 1_501], vec![1, 2, 3, 4, 5])?;
        let rx = handle.submit(request)?;
        rxs.insert(id, ("migrated".to_string(), rx));
    }

    // Poll every event receiver until each request has reported `Finished`
    // once, or the deadline passes. After the kind label, the tuple counts
    // text tokens and records whether `Finished` was seen.
    let mut done = 0usize;
    let total = rxs.len();
    anyhow::ensure!(total > 0, "the scenario names no request kind: {wanted:?}");
    let mut counts: HashMap<RequestId, (String, usize, bool)> = HashMap::new();
    let mut watched = 0usize;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while done < total && std::time::Instant::now() < deadline {
        for (id, (kind, rx)) in rxs.iter_mut() {
            while let Ok(ev) = rx.try_recv() {
                watched += 1;
                let e = counts.entry(*id).or_insert((kind.clone(), 0, false));
                match ev {
                    Event::TextToken { .. } => e.1 += 1,
                    Event::TextTokens { ids } => e.1 += ids.len(),
                    Event::Finished { .. } if !e.2 => {
                        e.2 = true;
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
    for (id, (kind, toks, fin)) in &counts {
        println!(
            "req {:?} [{}]: text_tokens={} finished={}",
            id, kind, toks, fin
        );
        if kind == "text" && *toks == 0 {
            ok = false;
        }
        if kind == "migrated" && (*toks == 0 || !*fin) {
            ok = false;
        }
    }
    handle.shutdown();
    println!("\nRUST CONTROL-PLANE: {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
