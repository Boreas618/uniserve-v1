//! Giving back and restoring the KV cache's device memory against the GPU's
//! memory ledger: a shrink keeps the segments the quota leaves the serving
//! world beside everything else the ledger books and what a successor adds,
//! lets the requests that hold the tail it withdraws run to their end (or
//! recomputes them past its bound) before the workers give the tail back,
//! and refuses a successor whose limit cannot hold the first segment; a
//! restore creates the tail again and a growth as much of it as the
//! ledger's room holds; a sleeping scheduler admits nothing until it wakes,
//! and a weight update rewrites the weights while the workers sleep.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use engine::{
    BatchResult, CacheChange, CommandSender, Event, EventRx, ExecutionBatch, Executor,
    ExecutorError, ExecutorInfo, ExecutorSubmitError, FinishReason, SchedulerConfig, SimEngine,
    SimExecutor, Successor, WeightsInFlight, WorkerGroupId,
};
use engine::{GenerationConstraint, GenerationRequest, ImageGenerationConfig};
use ipc::{BatchCommand, ImageParams, RequestId, SamplingParams};
use proto::request::MigrationCause;
use storage::{GpuLedger, Item, LedgerEntry, Scope, Segment};

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
const SEGMENT_BYTES: u64 = SEGMENT_UNITS as u64 * UNIT_BYTES;

/// The serving world of the simulated workers.
const WORLD: u64 = 1;

/// What the simulated workers report and the hooks they ran.
#[derive(Default)]
struct Workers {
    /// The quota of the worker's GPU.
    quota: u64,
    /// What the worker's process books beside its weights and its cache.
    process_bytes: u64,
    /// What the Transfer Server books on the GPU.
    transfer_bytes: u64,
    /// What the Transfer Server's context takes on the GPU while held,
    /// which the ledger reports booked or not.
    transfer_context: u64,
    /// What another world's process books on the GPU.
    other_world_bytes: u64,
    /// The serving world's weights on the GPU.
    weights_bytes: u64,
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
    /// What the expert worker's process books on a GPU of its own whose
    /// quota is 1000 bytes; `None` for an expert worker that reports no
    /// ledger.
    expert_process: Option<u64>,
    /// Each hook run, with its payload.
    hooks: Vec<String>,
    /// The graph share a successor's warmup takes on the GPU, and whether
    /// the worker's model can be captured at all.
    graph_share_bytes: u64,
    capturable: bool,
    /// The bandwidth from host memory to the GPU the Transfer Server
    /// measured, in bytes per second.
    host_link: Option<u64>,
}

impl Workers {
    /// The serving workers of a GPU whose quota their cache of eight
    /// segments and their process of `process_bytes` fill.
    fn filling(process_bytes: u64) -> Self {
        Self {
            quota: u64::from(SEGMENTS) * SEGMENT_BYTES + process_bytes,
            process_bytes,
            free_bytes: 1 << 40,
            held: SEGMENTS,
            ..Self::default()
        }
    }

    /// The deployment's ledger of the worker's GPU: each held segment of the
    /// serving world, its process, and the other world's and the Transfer
    /// Server's bookings.
    fn ledger(&self) -> GpuLedger {
        let scope = Scope {
            deployment: "model".into(),
            group: "model".into(),
            rank: 0,
        };
        let entry = |item, scope: Option<&Scope>, worlds: Vec<u64>, segment, bytes| LedgerEntry {
            item,
            scope: scope.cloned(),
            worlds,
            segment,
            bytes,
        };
        let mut entries: Vec<LedgerEntry> = (0..self.held)
            .map(|ordinal| {
                entry(
                    Item::CacheSegment,
                    Some(&scope),
                    vec![WORLD],
                    Some(Segment {
                        world: WORLD,
                        ordinal,
                    }),
                    SEGMENT_BYTES,
                )
            })
            .collect();
        entries.push(entry(
            Item::Process,
            Some(&scope),
            vec![WORLD],
            None,
            self.process_bytes,
        ));
        entries.push(entry(
            Item::TransferContext,
            None,
            Vec::new(),
            None,
            self.transfer_bytes,
        ));
        entries.push(entry(
            Item::Process,
            Some(&scope),
            vec![WORLD + 1],
            None,
            self.other_world_bytes,
        ));
        entries.push(entry(
            Item::Weights,
            Some(&scope),
            vec![WORLD],
            None,
            self.weights_bytes,
        ));
        entries.retain(|entry| entry.bytes > 0);
        GpuLedger {
            gpu: [7; 16],
            deployment: "model".into(),
            total: 1 << 40,
            quota: Some(self.quota),
            entries,
            host_link_bytes_per_second: self.host_link,
            transfer_context: self.transfer_context,
        }
    }
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
                        world: WORLD,
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
                "graph_share_bytes": workers.graph_share_bytes,
                "capturable": workers.capturable,
                "host": "node0",
                "scope": {"deployment": "model", "group": "model", "rank": 0},
                "world": WORLD,
                "ledger": workers.ledger(),
                "host_memory": {"available": 1u64 << 40, "reserve": 10u64 << 30, "pools_unallocated": 0},
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
            let ledger = workers.expert_process.map(|process| GpuLedger {
                gpu: [8; 16],
                deployment: "model".into(),
                total: 2_000,
                quota: Some(1_000),
                host_link_bytes_per_second: None,
                entries: vec![LedgerEntry {
                    item: Item::Process,
                    scope: Some(Scope {
                        deployment: "model".into(),
                        group: "experts".into(),
                        rank: 0,
                    }),
                    worlds: vec![WORLD],
                    segment: None,
                    bytes: process,
                }],
                transfer_context: 0,
            });
            // The device has free what its total leaves beside the process.
            let free = workers.expert_process.map_or(0, |process| 2_000 - process);
            results.push(
                serde_json::json!({
                    "free_bytes": free,
                    "host": "node1",
                    "scope": {"deployment": "model", "group": "experts", "rank": 0},
                    "world": WORLD,
                    "ledger": ledger,
                })
                .to_string(),
            );
        }
        Ok(results)
    }
}

/// A scheduler over the simulator whose `workers` serve a cache of
/// segmented planes of 32 units of `BLOCK` tokens.
fn start(workers: Workers) -> (CommandSender, Arc<Mutex<Workers>>, thread::JoinHandle<()>) {
    let mut sim = SimEngine::new();
    sim.set_text_len(10_000);
    sim.set_num_units(SEGMENTS * SEGMENT_UNITS);
    sim.set_block_size(BLOCK);
    let workers = Arc::new(Mutex::new(workers));
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

/// On a GPU whose quota the cache and a process of 3.5 segments' bytes
/// fill, a successor adds a process as large, which leaves the serving
/// world's cache 4.5 segments' bytes: the shrink gives back the last four
/// segments (units 16..32). The request holding units 1..=19, whose last
/// page has room for every token it generates, runs on to its end, and the
/// workers give the tail back once it let the units go. A request that fits
/// the whole cache and not what is left waits until the restore.
#[test]
fn a_shrink_lets_the_holders_of_the_tail_run_and_a_restore_returns_it() {
    let (handle, workers, thread) = start(Workers::filling(3_500));
    let mut holder = handle
        .submit(request_of(1, 18 * BLOCK as usize + 6, 8))
        .unwrap();
    assert!(next_token(&mut holder, Duration::from_secs(10)).is_some());

    let shrunk = handle.resize_cache(CacheChange::Shrink {
        preempt_after: None,
        successor: Successor::default(),
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
/// however much of the quota is left: a head of nine units withdraws the
/// last three segments of the 32-unit capacity. A cache sized by memory
/// withdraws nothing while the quota holds the successor.
#[test]
fn a_configured_capacity_withdraws_room_for_a_successors_head() {
    let (handle, workers, thread) = start(Workers {
        quota: 1 << 40,
        head_bytes: 9 * UNIT_BYTES,
        ..Workers::filling(0)
    });
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None,
                successor: Successor::default(),
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
                preempt_after: None,
                successor: Successor::default(),
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

/// A successor of another parallel configuration inherits none of the
/// cache's segments and lays its own cache out within its own capacity:
/// the configured capacity takes nothing back while the quota holds the
/// successor, so the requests running on the cache keep the units they
/// grow into.
#[test]
fn a_redistributing_successor_leaves_a_configured_capacity_whole() {
    let (handle, workers, thread) = start(Workers {
        quota: 1 << 40,
        head_bytes: 9 * UNIT_BYTES,
        capacity_units: Some(u64::from(SEGMENTS * SEGMENT_UNITS)),
        ..Workers::filling(0)
    });
    let redistributing = Successor {
        recomputes_cache: true,
        ..Successor::default()
    };
    let plans = resize(&handle, CacheChange::Plan(redistributing.clone())).unwrap();
    let plan: serde_json::Value = serde_json::from_str(&plans[0]).unwrap();
    assert_eq!(plan["segments_kept"], SEGMENTS);
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None,
                successor: redistributing,
            }
        )
        .is_ok()
    );
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec![
            "uniserve.cache_memory".to_owned(),
            "uniserve.cache_memory".to_owned()
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
    let (handle, workers, thread) = start(Workers {
        quota: 1 << 40,
        head_bytes: 9 * UNIT_BYTES,
        capacity_units: Some(u64::from(SEGMENTS * SEGMENT_UNITS)),
        experts: true,
        ..Workers::filling(0)
    });
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None,
                successor: Successor::default(),
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

/// A process outside the deployment may hold more of the device than the
/// quota leaves it: the successor's grant is no more than the device has
/// free, so the plan keeps no more of the serving world's cache than the
/// free storage with it leaves the successor's additions, and the shrink
/// gives back the rest though the quota would hold it.
#[test]
fn a_shrink_leaves_the_successor_what_the_device_has_free() {
    // The quota holds the whole cache beside the successor's head segment;
    // the device has 1500 bytes free.
    let (handle, workers, thread) = start(Workers {
        quota: u64::from(SEGMENTS) * SEGMENT_BYTES + 3 * SEGMENT_BYTES,
        head_bytes: 2 * SEGMENT_BYTES,
        free_bytes: 1_500,
        ..Workers::filling(0)
    });
    let plans = resize(&handle, CacheChange::Plan(Successor::default())).unwrap();
    let plan: serde_json::Value = serde_json::from_str(&plans[0]).unwrap();
    // 1500 free and the 8000 of the cache, less the 2000 of the head.
    assert_eq!(plan["kv_limit"], 7_500);
    assert_eq!(plan["free"], 1_500);
    assert_eq!(plan["segments_kept"], 7);

    let shrunk = resize(
        &handle,
        CacheChange::Shrink {
            preempt_after: None,
            successor: Successor::default(),
        },
    );
    assert!(shrunk.is_ok(), "{shrunk:?}");
    assert_eq!(
        workers.lock().unwrap().hooks.last().map(String::as_str),
        Some(r#"uniserve.release_cache {"from":7}"#)
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// A successor of another parallel configuration creates weight chunks on
/// the GPU beside those it maps: the serving world's limit leaves it them,
/// and the shrink gives back that much more of the tail. Bytes the
/// successor creates on another GPU leave this one's limit as it was.
#[test]
fn the_weights_a_successor_creates_leave_the_serving_world_less_cache() {
    let (handle, workers, thread) = start(Workers {
        quota: u64::from(SEGMENTS) * SEGMENT_BYTES + 3 * SEGMENT_BYTES,
        head_bytes: 2 * SEGMENT_BYTES,
        free_bytes: 1 << 40,
        ..Workers::filling(0)
    });
    let creating = |gpu: [u8; 16]| Successor {
        created_weights: engine::CreatedWeights::from([(gpu, 3 * SEGMENT_BYTES)]),
        ..Successor::default()
    };
    let plans = resize(&handle, CacheChange::Plan(creating([8; 16]))).unwrap();
    let plan: serde_json::Value = serde_json::from_str(&plans[0]).unwrap();
    assert_eq!(plan["successor"]["weights"], 0);
    assert_eq!(plan["segments_kept"], SEGMENTS);

    // Three segments' bytes of weights on the cache's GPU: the quota held
    // the cache, the head and one segment more, so the limit falls to six
    // segments and two go back.
    let plans = resize(&handle, CacheChange::Plan(creating([7; 16]))).unwrap();
    let plan: serde_json::Value = serde_json::from_str(&plans[0]).unwrap();
    assert_eq!(plan["successor"]["weights"], 3 * SEGMENT_BYTES);
    assert_eq!(plan["segments_kept"], SEGMENTS - 2);

    let shrunk = resize(
        &handle,
        CacheChange::Shrink {
            preempt_after: None,
            successor: creating([7; 16]),
        },
    );
    assert!(shrunk.is_ok(), "{shrunk:?}");
    assert_eq!(
        workers.lock().unwrap().hooks.last().map(String::as_str),
        Some(format!(r#"uniserve.release_cache {{"from":{}}}"#, SEGMENTS - 2).as_str())
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// The GPU of an expert worker, which holds no cache to give back, must
/// hold a successor's process beside the serving world's: a process of 400
/// bytes fits twice in a quota of 1000 and the plan reports that GPU, one
/// of 600 does not, and the shrink is refused before anything is withdrawn,
/// naming that GPU's items.
#[test]
fn a_gpu_without_a_cache_must_hold_the_successor_beside_the_serving_world() {
    let (handle, workers, thread) = start(Workers {
        quota: 1 << 40,
        experts: true,
        expert_process: Some(400),
        ..Workers::filling(0)
    });
    let plans = resize(&handle, CacheChange::Plan(Successor::default())).unwrap();
    assert_eq!(plans.len(), 2, "{plans:?}");
    let experts: serde_json::Value = serde_json::from_str(&plans[1]).unwrap();
    assert_eq!(
        (&experts["host"], &experts["kv_limit"], &experts["feasible"]),
        (
            &serde_json::json!("node1"),
            &serde_json::json!(200),
            &serde_json::json!(true)
        )
    );

    workers.lock().unwrap().expert_process = Some(600);
    let refused = resize(
        &handle,
        CacheChange::Shrink {
            preempt_after: None,
            successor: Successor::default(),
        },
    )
    .unwrap_err();
    assert!(
        refused.contains("host node1 GPU GPU-08080808") && refused.contains("KV cache limit -200"),
        "{refused}"
    );
    assert!(
        !workers
            .lock()
            .unwrap()
            .hooks
            .iter()
            .any(|hook| hook.starts_with("uniserve.release_cache")),
        "nothing is given back"
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// A shrink past its bound recomputes the requests still holding the tail:
/// they end as migratable, and the workers give the tail back.
#[test]
fn a_shrink_past_its_bound_recomputes_the_holders_of_the_tail() {
    let (handle, workers, thread) = start(Workers::filling(3_500));
    let mut holder = handle.submit(request(1, 20, 10_000)).unwrap();
    assert!(next_token(&mut holder, Duration::from_secs(10)).is_some());

    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: Some(Duration::from_millis(50)),
                successor: Successor::default(),
            },
        )
        .is_ok()
    );
    assert!(
        matches!(
            terminal(&mut holder),
            Some(Event::Migratable {
                cause: MigrationCause::Drain,
                ..
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

/// A growth creates as many of the segments given back as the ledger's
/// room within the quota holds, and returns their units to allocation.
#[test]
fn a_growth_creates_the_segments_the_ledgers_room_holds() {
    let (handle, workers, thread) = start(Workers::filling(3_500));
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None,
                successor: Successor::default(),
            },
        )
        .is_ok()
    );
    assert_eq!(workers.lock().unwrap().held, 4);

    // Another world's process of 1.5 segments' bytes leaves two and a half
    // segments' bytes of the quota: two segments grow back, although the
    // device has more free.
    workers.lock().unwrap().other_world_bytes = 1_500;
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

/// Room for a profiling session's device storage on a GPU whose quota the
/// cache and the process fill: 1500 bytes take the last two segments, which
/// the request holding units 1..=27 keeps until it ends, before the workers
/// give them back; once the session's storage is released, a growth
/// creates them again. A ledger with room withdraws nothing, and storage
/// that even every segment past the first does not make room for is
/// refused, naming the bytes, with the cache kept whole.
#[test]
fn room_for_a_sessions_storage_comes_from_the_tail_and_grows_back() {
    let (handle, workers, thread) = start(Workers::filling(3_500));
    let mut holder = handle
        .submit(request_of(1, 26 * BLOCK as usize + 6, 8))
        .unwrap();
    assert!(next_token(&mut holder, Duration::from_secs(10)).is_some());

    let made = handle.resize_cache(CacheChange::MakeRoom { bytes: 1_500 });
    assert!(
        matches!(terminal(&mut holder), Some(Event::Finished { .. })),
        "the request holding the tail runs to its end"
    );
    assert!(
        made.recv_timeout(Duration::from_secs(10))
            .expect("the scheduler answers")
            .is_ok()
    );
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec![
            "uniserve.cache_memory".to_owned(),
            r#"uniserve.release_cache {"from":6}"#.to_owned(),
        ]
    );

    // The room is there: nothing more is given back.
    assert!(resize(&handle, CacheChange::MakeRoom { bytes: 1_500 }).is_ok());
    assert_eq!(workers.lock().unwrap().held, 6);
    // The session books its storage, and releases it as it ends.
    workers.lock().unwrap().process_bytes = 5_000;
    assert!(resize(&handle, CacheChange::Grow).is_ok());
    assert_eq!(
        workers.lock().unwrap().held,
        6,
        "the session's storage holds the room"
    );
    workers.lock().unwrap().process_bytes = 3_500;
    assert!(resize(&handle, CacheChange::Grow).is_ok());
    assert_eq!(workers.lock().unwrap().held, SEGMENTS);

    let error = resize(&handle, CacheChange::MakeRoom { bytes: 7_500 }).unwrap_err();
    assert!(
        error.contains("7500 bytes of device storage for a profiling session do not fit"),
        "{error}"
    );
    assert!(
        error.contains("11500 of a quota of 11500 bytes booked"),
        "{error}"
    );
    assert_eq!(workers.lock().unwrap().held, SEGMENTS);
    let mut served = handle.submit(request(2, 30, 1)).unwrap();
    assert!(matches!(
        terminal(&mut served),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A successor whose process books more than the cache past its first
/// segment holds (7 segments of 1000 bytes) does not fit: the serving
/// world's limit of 500 bytes cannot hold its first segment. The shrink
/// fails before it withdraws anything, naming every item of the device and
/// its host, and requests go on with the whole cache.
#[test]
fn a_shrink_that_cannot_make_room_fails_and_keeps_the_cache() {
    let (handle, workers, thread) = start(Workers::filling(7_500));
    let error = resize(
        &handle,
        CacheChange::Shrink {
            preempt_after: None,
            successor: Successor::default(),
        },
    )
    .unwrap_err();
    assert!(error.contains("does not fit"), "{error}");
    assert!(
        error.contains("KV cache limit 500 bytes, 1000 of which it cannot give back"),
        "{error}"
    );
    assert!(
        error.contains("successor world: 0 weights it does not map, 0 graph pool, 7500 process"),
        "{error}"
    );
    assert!(
        error.contains("ledger: process of model/model/0 of world 1: 7500 bytes"),
        "{error}"
    );
    assert!(error.contains("host node0: admits"), "{error}");
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

/// A plan reports each device's memory beside a successor without
/// withdrawing anything: the Transfer Server's context lowers the serving
/// world's limit as every other item of the ledger does, and the shrink
/// then keeps the segments the plan named.
#[test]
fn a_plan_reports_each_devices_memory_and_withdraws_nothing() {
    // The Transfer Server holds no context on the GPU while no transfer
    // runs, so the ledger books none; the change's transfers will need it.
    let mut fill = Workers::filling(3_500);
    fill.transfer_context = 1_500;
    fill.quota += 1_500;
    let (handle, workers, thread) = start(fill);
    let plans = resize(&handle, CacheChange::Plan(Successor::default())).unwrap();
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec!["uniserve.cache_memory".to_owned()]
    );
    let [plan] = plans.as_slice() else {
        panic!("one plan per device: {plans:?}");
    };
    let plan: serde_json::Value = serde_json::from_str(plan).unwrap();
    // 13000 bytes of quota, less 3500 of the serving world's process, 1500
    // of the Transfer Server's context, not booked yet, and 3500 of the
    // successor's process.
    assert_eq!(plan["kv_limit"], 4_500);
    assert_eq!(plan["transfer"], 1_500);
    assert_eq!(plan["successor"]["process"], 3_500);
    assert_eq!(plan["segments_kept"], 4);
    assert_eq!(plan["feasible"], true);
    assert_eq!(plan["host"], "node0");

    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None,
                successor: Successor::default(),
            }
        )
        .is_ok()
    );
    assert_eq!(workers.lock().unwrap().held, 4);

    handle.shutdown();
    thread.join().unwrap();
}

/// A world on trial holds, beside the free units of its own segments, the
/// segments it has yet to create that the ledger has room for: while the
/// previous world's processes keep what they booked (it is parked), only
/// its own 15 free units, too few for a request recomputing 20; once the
/// promotion ends them (the previous world is kept as its weights), their
/// bookings are the successor's, whose four deferred segments then fit.
#[test]
fn a_promotion_counts_the_segments_the_previous_worlds_release_leaves_room_for() {
    let (handle, workers, thread) = start(Workers::filling(3_500));
    assert!(
        resize(
            &handle,
            CacheChange::Shrink {
                preempt_after: None,
                successor: Successor::default(),
            }
        )
        .is_ok()
    );
    // The previous world's process books the room the shrink left.
    workers.lock().unwrap().other_world_bytes = 4 * SEGMENT_BYTES;
    let parked = engine::ParkedCache {
        pages: Vec::new(),
        held_units: 20,
        requests: 1,
    };
    let refused = handle
        .promote(parked.clone(), true, false)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers")
        .expect_err("15 free units cannot hold 21");
    assert!(
        refused.contains("15 free units") && refused.contains("need 21"),
        "{refused}"
    );
    assert!(
        handle
            .promote(parked, true, true)
            .recv_timeout(Duration::from_secs(10))
            .expect("the scheduler answers")
            .is_ok()
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// A successor that loads its weights from the model's files adds as many
/// beside the serving world's, and one that maps none of the cache takes of
/// its own the units the requests in flight hold: the plan counts both, and
/// the shrink gives back the more of the cache. The plan also states the
/// KV cache the requests in flight hold, and how long a promotion that
/// moves it through host memory waits for it at the host link's measured
/// bandwidth.
#[test]
fn a_successor_of_other_weights_and_its_own_cache_needs_more_room() {
    // Bytes per second from host memory to the GPU.
    const HOST_LINK: u64 = 1_000;
    let (handle, _workers, thread) = start(Workers {
        weights_bytes: 2_000,
        quota: u64::from(SEGMENTS) * SEGMENT_BYTES + 2_000,
        host_link: Some(HOST_LINK),
        ..Workers::filling(0)
    });
    let plan_of = |successor| -> serde_json::Value {
        let plans = resize(&handle, CacheChange::Plan(successor)).unwrap();
        serde_json::from_str(&plans[0]).unwrap()
    };
    // Mapping the weights and the cache, it needs nothing beside them.
    let mapped = plan_of(Successor::default());
    assert_eq!(mapped["kv_limit"], 8_000);
    assert_eq!(
        (&mapped["in_flight_kv"], &mapped["promotion_stall_ms"]),
        (&serde_json::json!(0), &serde_json::json!(0))
    );

    let reloaded = plan_of(Successor {
        loads_weights: true,
        ..Successor::default()
    });
    assert_eq!(reloaded["successor"]["weights"], 2_000);
    assert_eq!(reloaded["kv_limit"], 6_000);
    assert_eq!(reloaded["segments_kept"], 6);

    // A request in flight holds units the successor computes again.
    let mut running = handle.submit(request(1, 8, 10_000)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());
    let recomputed = plan_of(Successor {
        loads_weights: true,
        recomputes_cache: true,
        ..Successor::default()
    });
    let in_flight = recomputed["successor"]["in_flight"].as_u64().unwrap();
    assert!(
        in_flight >= 8 * UNIT_BYTES && in_flight.is_multiple_of(UNIT_BYTES),
        "{recomputed}"
    );
    assert_eq!(
        recomputed["kv_limit"].as_i64().unwrap(),
        6_000 - in_flight as i64
    );
    assert_eq!(recomputed["in_flight_kv"], in_flight);
    assert_eq!(
        recomputed["promotion_stall_ms"],
        (in_flight * 1000).div_ceil(HOST_LINK)
    );
    // A successor that maps the cache takes none of its own, but the
    // requests in flight hold as much.
    let mapped = plan_of(Successor::default());
    assert_eq!(mapped["successor"]["in_flight"], 0);
    assert_eq!(mapped["in_flight_kv"], in_flight);

    handle.shutdown();
    thread.join().unwrap();
}

/// A parked world whose successor maps none of its cache gives back every
/// segment past its first, which a world that serves refuses; resumed at a
/// rollback, it creates them again.
#[test]
fn a_parked_world_evicts_its_cache_past_its_first_segment() {
    let (handle, workers, thread) = start(Workers::filling(0));
    assert!(resize(&handle, CacheChange::Evict).is_err());

    let parked = handle
        .park()
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(parked.is_ok(), "{parked:?}");
    let evicted = resize(&handle, CacheChange::Evict).unwrap();
    let evicted: serde_json::Value = serde_json::from_str(&evicted[0]).unwrap();
    assert_eq!(evicted["evicted_segments"], SEGMENTS - 1);
    assert_eq!(
        evicted["evicted_bytes"],
        u64::from(SEGMENTS - 1) * SEGMENT_BYTES
    );
    assert_eq!(workers.lock().unwrap().held, 1);

    // The rollback resumes the world, which creates its cache again and
    // serves a request on all of it.
    let resumed = handle
        .resume()
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(resumed.is_ok(), "{resumed:?}");
    assert!(resize(&handle, CacheChange::Restore).is_ok());
    assert_eq!(workers.lock().unwrap().held, SEGMENTS);
    let mut request = handle.submit(request(1, 20, 4)).unwrap();
    assert!(matches!(
        terminal(&mut request),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A successor's warmup takes a graph share only when its graph policy
/// captures and its model can be captured: the plan of one that serves
/// every call eagerly, or of a model that cannot be captured, leaves the
/// serving world's cache that share.
#[test]
fn a_successor_that_captures_no_graphs_takes_no_graph_share() {
    let (handle, _workers, thread) = start(Workers {
        quota: u64::from(SEGMENTS) * SEGMENT_BYTES + 2_000,
        graph_share_bytes: 2_000,
        capturable: true,
        ..Workers::filling(0)
    });
    let plan_of = |successor| -> serde_json::Value {
        let plans = resize(&handle, CacheChange::Plan(successor)).unwrap();
        serde_json::from_str(&plans[0]).unwrap()
    };
    let capturing = plan_of(Successor {
        captures_graphs: true,
        ..Successor::default()
    });
    assert_eq!(capturing["successor"]["graph_pool"], 2_000);
    assert_eq!(capturing["kv_limit"], 8_000);
    let eager = plan_of(Successor::default());
    assert_eq!(eager["successor"]["graph_pool"], 0);
    assert_eq!(eager["kv_limit"], 10_000);

    handle.shutdown();
    thread.join().unwrap();

    // A model that cannot be captured serves eagerly whatever the policy.
    let (handle, _workers, thread) = start(Workers {
        quota: u64::from(SEGMENTS) * SEGMENT_BYTES + 2_000,
        graph_share_bytes: 2_000,
        capturable: false,
        ..Workers::filling(0)
    });
    let plans = resize(
        &handle,
        CacheChange::Plan(Successor {
            captures_graphs: true,
            ..Successor::default()
        }),
    )
    .unwrap();
    let uncaptured: serde_json::Value = serde_json::from_str(&plans[0]).unwrap();
    assert_eq!(uncaptured["successor"]["graph_pool"], 0);

    handle.shutdown();
    thread.join().unwrap();
}

/// A sleep recomputes the requests that run, gives the memory back, and
/// admits nothing until the wake.
#[test]
fn a_sleeping_scheduler_admits_nothing_until_it_wakes() {
    let (handle, workers, thread) = start(Workers::filling(0));
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    assert!(resize(&handle, CacheChange::Sleep).is_ok());
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Migratable {
            cause: MigrationCause::Drain,
            ..
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

/// A weight update that recomputes the requests in flight ends those that
/// run as migratable, and rewrites the weights while the workers sleep,
/// which they wake from after; requests are served again.
#[test]
fn a_weight_update_recomputing_the_requests_in_flight_runs_while_the_workers_sleep() {
    let (handle, workers, thread) = start(Workers::filling(0));
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    let updated = handle
        .update_weights("/checkpoints/next".to_owned(), WeightsInFlight::Recompute)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(updated.is_ok(), "{updated:?}");
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Migratable {
            cause: MigrationCause::Drain,
            ..
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

/// A weight update that keeps the requests in flight rewrites the weights
/// beside their KV cache, and the request running continues to its end
/// without being handed back.
#[test]
fn a_weight_update_keeping_the_requests_in_flight_continues_them() {
    let (handle, workers, thread) = start(Workers::filling(0));
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    let updated = handle
        .update_weights("/checkpoints/next".to_owned(), WeightsInFlight::Keep)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(updated.is_ok(), "{updated:?}");
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec![
            r#"uniserve.update_weights {"model_path":"/checkpoints/next","keep":true}"#.to_owned()
        ]
    );
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Finished {
            reason: FinishReason::MaxTokens | FinishReason::Completed | FinishReason::Eos,
            ..
        })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A weight update that aborts the requests in flight ends both the request
/// running and the one queued, aborted.
#[test]
fn a_weight_update_aborting_the_requests_in_flight_ends_them() {
    let (handle, _workers, thread) = start(Workers::filling(0));
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    let updated = handle
        .update_weights("/checkpoints/next".to_owned(), WeightsInFlight::Abort)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(updated.is_ok(), "{updated:?}");
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Finished {
            reason: FinishReason::Aborted,
            ..
        })
    ));

    handle.shutdown();
    thread.join().unwrap();
}
