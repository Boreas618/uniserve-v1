//! Giving back and restoring the KV cache's device memory: a shrink lets
//! the requests that hold the tail it withdraws run to their end (or
//! recomputes them past its bound) before the workers give the tail back, a
//! restore creates it again and a growth as much of it as the memory holds,
//! a sleeping scheduler admits nothing until it wakes, and a weight update
//! rewrites the weights while the workers sleep.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use engine::{
    BatchResult, CacheChange, CommandSender, Event, EventRx, ExecutionBatch, Executor,
    ExecutorError, ExecutorInfo, ExecutorSubmitError, SchedulerConfig, SimEngine, SimExecutor,
    WorkerGroupId,
};
use engine::{GenerationConstraint, GenerationRequest, ImageGenerationConfig};
use ipc::{BatchCommand, ImageParams, RequestId, SamplingParams};
use proto::request::MigrationCause;

use crate::support;

/// Segments of the simulated planes, and the units of each: a pool of 32
/// units, the sentinel included.
const SEGMENTS: u32 = 8;
const SEGMENT_UNITS: u32 = 4;
/// Tokens of one cache page.
const BLOCK: u32 = 16;

/// Device bytes of one unit across the simulated planes: a segment holds
/// 1000.
const UNIT_BYTES: u64 = 250;

/// What the simulated workers report and the hooks they ran.
#[derive(Default)]
struct Workers {
    /// Device bytes the worker holds beside the weights and the cache.
    process_bytes: u64,
    /// Free device bytes.
    free_bytes: u64,
    /// Segments the worker holds.
    held: u32,
    /// Device bytes of the head segment a successor creates.
    head_bytes: u64,
    /// The units a configured token capacity holds; `None` for a cache
    /// sized by memory.
    capacity_units: Option<u64>,
    /// Whether an expert worker, which holds no cache, reports beside the
    /// worker that holds it.
    experts: bool,
    /// Each hook run, with its payload.
    hooks: Vec<String>,
}

/// The simulator, whose workers answer the engine's own hooks as workers
/// with segmented planes do and whose results take `STEP` each.
struct Segmented {
    inner: SimExecutor,
    workers: Arc<Mutex<Workers>>,
}

/// How long each batch result takes.
const STEP: Duration = Duration::from_millis(10);

impl Executor for Segmented {
    fn info(&self) -> &ExecutorInfo {
        self.inner.info()
    }

    fn is_ready(&self, worker: &WorkerGroupId) -> bool {
        self.inner.is_ready(worker)
    }

    fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        self.inner.has_capacity(worker)
    }

    fn command_has_capacity(&self, command: &BatchCommand) -> bool {
        self.inner.command_has_capacity(command)
    }

    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        self.inner.submit(batch)
    }

    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, ExecutorError> {
        thread::sleep(STEP.min(timeout));
        self.inner.poll(timeout)
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.inner.close()
    }

    fn hook(&mut self, name: &str, payload: &str) -> Result<Vec<String>, ExecutorError> {
        let mut workers = self.workers.lock().unwrap();
        workers
            .hooks
            .push(format!("{name} {payload}").trim_end().to_owned());
        let result = match name {
            "uniserve.cache_memory" => serde_json::json!({
                "segments": (0..SEGMENTS)
                    .map(|ordinal| ipc::CacheSegment {
                        world: 1,
                        ordinal,
                        units: SEGMENT_UNITS,
                    })
                    .collect::<Vec<_>>(),
                "held": workers.held,
                "unit_bytes": UNIT_BYTES,
                "head_bytes": workers.head_bytes,
                "capacity_units": workers.capacity_units,
                "scratch_units": 0,
                "free_bytes": workers.free_bytes,
                "process_bytes": workers.process_bytes,
                "reserve_bytes": 0,
                "exclusive_bytes": 0,
            }),
            "uniserve.release_cache" => {
                let payload: serde_json::Value = serde_json::from_str(payload)?;
                workers.held = payload["from"].as_u64().unwrap() as u32;
                serde_json::Value::Null
            }
            "uniserve.restore_cache" => {
                let payload: serde_json::Value = if payload.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::from_str(payload)?
                };
                workers.held = match payload["segments"].as_u64() {
                    Some(count) => (workers.held + count as u32).min(SEGMENTS),
                    None => SEGMENTS,
                };
                serde_json::Value::Null
            }
            "uniserve.wake" => {
                workers.held = SEGMENTS;
                serde_json::Value::Null
            }
            "uniserve.sleep" => {
                workers.held = 0;
                serde_json::Value::Null
            }
            "uniserve.update_weights" => serde_json::json!({ "group": null, "page_contents": [] }),
            other => anyhow::bail!("no worker hook {other}"),
        };
        let mut results = vec![result.to_string()];
        if workers.experts && name == "uniserve.cache_memory" {
            results.push(
                serde_json::json!({
                    "free_bytes": 0,
                    "process_bytes": 1 << 30,
                })
                .to_string(),
            );
        }
        Ok(results)
    }
}

/// A scheduler over the simulator whose workers hold `process_bytes`
/// beside the cache, with segmented planes of 32 units of `BLOCK` tokens.
fn start(process_bytes: u64) -> (CommandSender, Arc<Mutex<Workers>>, thread::JoinHandle<()>) {
    let mut sim = SimEngine::new();
    sim.set_text_len(10_000);
    sim.set_num_units(SEGMENTS * SEGMENT_UNITS);
    sim.set_block_size(BLOCK);
    let workers = Arc::new(Mutex::new(Workers {
        process_bytes,
        free_bytes: 0,
        held: SEGMENTS,
        head_bytes: 0,
        capacity_units: None,
        experts: false,
        hooks: Vec::new(),
    }));
    let scheduler = support::scheduler_with(
        Box::new(Segmented {
            inner: SimExecutor::new(sim),
            workers: Arc::clone(&workers),
        }),
        SimEngine::special_tokens(),
        SchedulerConfig::default(),
    )
    .unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = thread::spawn(move || {
        let _ = scheduler.run(rx);
    });
    (CommandSender::new(tx), workers, thread)
}

/// A text request of `pages` cache pages of prompt that generates up to
/// `tokens` tokens.
fn request(id: u64, pages: u32, tokens: usize) -> GenerationRequest {
    request_of(id, (pages * BLOCK) as usize, tokens)
}

/// A text request of `prompt` prompt tokens that generates up to `tokens`
/// tokens.
fn request_of(id: u64, prompt: usize, tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![7; prompt],
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: ImageGenerationConfig::default(),
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// Waits up to `wait` for the next text token of `events`.
fn next_token(events: &mut EventRx, wait: Duration) -> Option<u32> {
    let deadline = Instant::now() + wait;
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => return Some(id),
            Ok(event) if event.is_terminal() => panic!("the request ended: {event:?}"),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    None
}

/// The terminal event of `events`, within ten seconds.
fn terminal(events: &mut EventRx) -> Option<Event> {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(event) if event.is_terminal() => return Some(event),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    None
}

/// The answer to a change of the cache's memory, within ten seconds.
fn resize(handle: &CommandSender, change: CacheChange) -> Result<Vec<String>, String> {
    handle
        .resize_cache(change)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers")
}

/// A shrink for a worker holding 3.5 segments' bytes beside the cache gives
/// back the last four segments (units 16..32): the request holding units
/// 1..=19, whose last page has room for every token it generates, runs on
/// to its end, and the workers give the tail back once it let the units go.
/// A request that fits the whole cache and not what is left waits until
/// the restore.
#[test]
fn a_shrink_lets_the_holders_of_the_tail_run_and_a_restore_returns_it() {
    let (handle, workers, thread) = start(3_500);
    let mut holder = handle
        .submit(request_of(1, 18 * BLOCK as usize + 6, 8))
        .unwrap();
    assert!(next_token(&mut holder, Duration::from_secs(10)).is_some());

    let shrunk = handle.resize_cache(CacheChange::Shrink {
        preempt_after: None,
    });
    assert!(
        matches!(terminal(&mut holder), Some(Event::Finished { .. })),
        "the request holding the tail runs to its end"
    );
    assert!(
        shrunk
            .recv_timeout(Duration::from_secs(10))
            .expect("the scheduler answers")
            .is_ok()
    );
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec![
            "uniserve.cache_memory".to_owned(),
            r#"uniserve.release_cache {"from":4}"#.to_owned(),
        ]
    );

    // Twenty pages fit the 31 units of the whole cache, not the 15 left.
    let mut waiting = handle.submit(request(2, 20, 4)).unwrap();
    assert_eq!(next_token(&mut waiting, Duration::from_millis(300)), None);
    assert!(resize(&handle, CacheChange::Restore).is_ok());
    assert_eq!(
        workers.lock().unwrap().hooks.last().map(String::as_str),
        Some("uniserve.restore_cache")
    );
    assert!(matches!(
        terminal(&mut waiting),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A cache of a configured token capacity makes room within it for a
/// successor's head segment beside the segments the successor inherits,
/// however much memory is free: a head of nine units withdraws the last
/// three segments of the 32-unit capacity. A cache sized by memory withdraws
/// nothing while the memory holds the successor.
#[test]
fn a_configured_capacity_withdraws_room_for_a_successors_head() {
    let (handle, workers, thread) = start(0);
    {
        let mut workers = workers.lock().unwrap();
        workers.free_bytes = 1 << 40;
        workers.head_bytes = 9 * UNIT_BYTES;
    }
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None
            }
        )
        .is_ok()
    );
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec!["uniserve.cache_memory".to_owned()]
    );

    workers.lock().unwrap().capacity_units = Some(u64::from(SEGMENTS * SEGMENT_UNITS));
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None
            }
        )
        .is_ok()
    );
    assert_eq!(
        workers.lock().unwrap().hooks[1..],
        [
            "uniserve.cache_memory".to_owned(),
            r#"uniserve.release_cache {"from":5}"#.to_owned(),
        ]
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// An expert worker's report, which holds no cache, leaves the shrink to
/// the worker that holds it: the configured capacity withdraws room for a
/// successor's head as it does without one.
#[test]
fn a_worker_without_a_cache_leaves_the_shrink_to_the_others() {
    let (handle, workers, thread) = start(0);
    {
        let mut workers = workers.lock().unwrap();
        workers.free_bytes = 1 << 40;
        workers.head_bytes = 9 * UNIT_BYTES;
        workers.capacity_units = Some(u64::from(SEGMENTS * SEGMENT_UNITS));
        workers.experts = true;
    }
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None
            }
        )
        .is_ok()
    );
    assert_eq!(
        workers.lock().unwrap().hooks,
        [
            "uniserve.cache_memory".to_owned(),
            r#"uniserve.release_cache {"from":5}"#.to_owned(),
        ]
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// A shrink past its bound recomputes the requests still holding the tail:
/// they end as migratable, and the workers give the tail back.
#[test]
fn a_shrink_past_its_bound_recomputes_the_holders_of_the_tail() {
    let (handle, workers, thread) = start(3_500);
    let mut holder = handle.submit(request(1, 20, 10_000)).unwrap();
    assert!(next_token(&mut holder, Duration::from_secs(10)).is_some());

    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: Some(Duration::from_millis(50)),
            },
        )
        .is_ok()
    );
    assert!(
        matches!(
            terminal(&mut holder),
            Some(Event::Migratable {
                cause: MigrationCause::Drain
            })
        ),
        "the request holding the tail is recomputed"
    );
    assert_eq!(
        workers.lock().unwrap().hooks.last().map(String::as_str),
        Some(r#"uniserve.release_cache {"from":4}"#)
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// A growth creates as many of the segments given back as the free memory
/// holds, and returns their units to allocation.
#[test]
fn a_growth_creates_the_segments_the_free_memory_holds() {
    let (handle, workers, thread) = start(3_500);
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None,
            },
        )
        .is_ok()
    );
    assert_eq!(workers.lock().unwrap().held, 4);

    // Two and a half segments' bytes free: two segments grow back.
    workers.lock().unwrap().free_bytes = 2_500;
    assert!(resize(&handle, CacheChange::Grow).is_ok());
    assert!(
        workers
            .lock()
            .unwrap()
            .hooks
            .iter()
            .any(|hook| hook == r#"uniserve.restore_cache {"segments":2}"#),
        "{:?}",
        workers.lock().unwrap().hooks
    );
    assert_eq!(workers.lock().unwrap().held, 6);
    // Twenty-two pages fit the 23 units of six segments.
    let mut served = handle.submit(request(1, 22, 1)).unwrap();
    assert!(matches!(
        terminal(&mut served),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A worker holding more beside the cache than the cache past its first
/// segment holds (7 segments of 1000 bytes) cannot get the room: the
/// shrink fails before it withdraws anything, and requests go on with the
/// whole cache.
#[test]
fn a_shrink_that_cannot_make_room_fails_and_keeps_the_cache() {
    let (handle, workers, thread) = start(7_500);
    let error = resize(
        &handle,
        CacheChange::Shrink {
            preempt_after: None,
        },
    )
    .unwrap_err();
    assert!(error.contains("cannot make room"), "{error}");
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec!["uniserve.cache_memory".to_owned()]
    );
    let mut served = handle.submit(request(1, 20, 4)).unwrap();
    assert!(matches!(
        terminal(&mut served),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A sleep recomputes the requests that run, gives the memory back, and
/// admits nothing until the wake.
#[test]
fn a_sleeping_scheduler_admits_nothing_until_it_wakes() {
    let (handle, workers, thread) = start(0);
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    assert!(resize(&handle, CacheChange::Sleep).is_ok());
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Migratable {
            cause: MigrationCause::Drain
        })
    ));
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec!["uniserve.sleep".to_owned()]
    );

    let mut waiting = handle.submit(request(2, 2, 4)).unwrap();
    assert_eq!(next_token(&mut waiting, Duration::from_millis(300)), None);
    assert!(resize(&handle, CacheChange::Wake).is_ok());
    assert_eq!(
        workers.lock().unwrap().hooks.last().map(String::as_str),
        Some("uniserve.wake")
    );
    assert!(matches!(
        terminal(&mut waiting),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A weight update recomputes the requests that run, and rewrites the
/// weights while the workers sleep, which they wake from after; requests
/// are served again.
#[test]
fn a_weight_update_runs_while_the_workers_sleep() {
    let (handle, workers, thread) = start(0);
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    let updated = handle
        .update_weights("/checkpoints/next".to_owned())
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(updated.is_ok(), "{updated:?}");
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Migratable {
            cause: MigrationCause::Drain
        })
    ));
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec![
            "uniserve.sleep".to_owned(),
            r#"uniserve.update_weights {"model_path":"/checkpoints/next"}"#.to_owned(),
            "uniserve.wake".to_owned(),
        ]
    );
    let mut served = handle.submit(request(2, 2, 4)).unwrap();
    assert!(matches!(
        terminal(&mut served),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}
