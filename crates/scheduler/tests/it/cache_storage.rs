//! Giving back and restoring the KV cache's device storage against the GPU's
//! storage ledger: a shrink keeps the segments the quota leaves the serving
//! epoch beside everything else the ledger books and what a successor adds,
//! lets the requests that hold the tail it withdraws run to their end (or
//! recomputes them past its bound, or once they cannot get a unit to
//! continue) before the workers give the tail back,
//! and refuses a successor whose limit cannot hold the first segment; a
//! restore creates the tail again and a growth as much of it as the
//! ledger's room holds; a sleeping scheduler admits nothing until it wakes,
//! and a weight update rewrites the weights while the workers sleep.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ipc::{BatchCommand, ImageParams, RequestId, SamplingParams};
use proto::request::MigrationCause;
use scheduler::{
    BatchResult, CacheChange, CommandSender, Event, EventRx, ExecutionBatch, FinishReason,
    SchedulerConfig, SimEngine, SimWorker, WeightsInFlight, Worker, WorkerError, WorkerGroupId,
    WorkerInfo, WorkerSubmitError,
};
use scheduler::{GenerationConstraint, GenerationRequest, ImageGenerationConfig};
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

/// The serving epoch of the simulated workers.
const EPOCH: u64 = 1;

/// What the simulated workers report and the hooks they ran.
#[derive(Default)]
struct SimWorkers {
    /// The quota of the worker's GPU.
    quota: u64,
    /// What the worker's process books beside its weights and its cache.
    process_bytes: u64,
    /// The serving epoch's captured graph allocations.
    graph_bytes: u64,
    /// What the Transfer Server books on the GPU.
    transfer_bytes: u64,
    /// What the Transfer Server's context takes on the GPU while held,
    /// which the ledger reports booked or not.
    transfer_context: u64,
    /// What another epoch's process books on the GPU.
    other_epoch_bytes: u64,
    /// The serving epoch's weights on the GPU.
    weights_bytes: u64,
    /// Free device bytes.
    free_bytes: u64,
    /// Segments the worker holds.
    held: u32,
    /// Device bytes of the head segment a successor creates.
    head_bytes: u64,
    /// The units a configured token capacity holds; `None` for a cache
    /// sized by storage.
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
    /// The bandwidth from host storage to the GPU the Transfer Server
    /// measured, in bytes per second.
    host_link: Option<u64>,
}

impl SimWorkers {
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
    /// serving epoch, its process, and the other epoch's and the Transfer
    /// Server's bookings.
    fn ledger(&self) -> GpuLedger {
        let scope = Scope {
            deployment: "model".into(),
            group: "model".into(),
            rank: 0,
        };
        let entry = |item, scope: Option<&Scope>, epochs: Vec<u64>, segment, bytes| LedgerEntry {
            item,
            scope: scope.cloned(),
            epochs,
            segment,
            bytes,
        };
        let mut entries: Vec<LedgerEntry> = (0..self.held)
            .map(|ordinal| {
                entry(
                    Item::CacheSegment,
                    Some(&scope),
                    vec![EPOCH],
                    Some(Segment {
                        epoch: EPOCH,
                        ordinal,
                    }),
                    SEGMENT_BYTES,
                )
            })
            .collect();
        entries.push(entry(
            Item::Process,
            Some(&scope),
            vec![EPOCH],
            None,
            self.process_bytes,
        ));
        entries.push(entry(
            Item::GraphPool,
            Some(&scope),
            vec![EPOCH],
            None,
            self.graph_bytes,
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
            vec![EPOCH + 1],
            None,
            self.other_epoch_bytes,
        ));
        entries.push(entry(
            Item::Weights,
            Some(&scope),
            vec![EPOCH],
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
    inner: SimWorker,
    workers: Arc<Mutex<SimWorkers>>,
}

/// How long each batch result takes.
const STEP: Duration = Duration::from_millis(10);

impl Worker for Segmented {
    fn info(&self) -> &WorkerInfo {
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

    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), WorkerSubmitError> {
        self.inner.submit(batch)
    }

    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, WorkerError> {
        thread::sleep(STEP.min(timeout));
        self.inner.poll(timeout)
    }

    fn close(&mut self) -> Result<(), WorkerError> {
        self.inner.close()
    }

    fn hook(&mut self, name: &str, payload: &str) -> Result<Vec<String>, WorkerError> {
        let mut workers = self.workers.lock().unwrap();
        workers
            .hooks
            .push(format!("{name} {payload}").trim_end().to_owned());
        let result = match name {
            "uniserve.cache_storage" => serde_json::json!({
                "segments": (0..SEGMENTS)
                    .map(|ordinal| ipc::CacheSegment {
                        epoch: EPOCH,
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
                "epoch": EPOCH,
                "ledger": workers.ledger(),
                "host_storage": {"available": 1u64 << 40, "reserve": 10u64 << 30, "unallocated": 0},
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
            "uniserve.adopt_cache" => serde_json::json!({ "inherited": 0 }),
            other => anyhow::bail!("no worker hook {other}"),
        };
        let mut results = vec![result.to_string()];
        if workers.experts && name == "uniserve.cache_storage" {
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
                    epochs: vec![EPOCH],
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
                    "epoch": EPOCH,
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
fn start(
    workers: SimWorkers,
) -> (
    CommandSender,
    Arc<Mutex<SimWorkers>>,
    thread::JoinHandle<()>,
) {
    start_split(workers, None)
}

/// [`start`], the workers reporting their planes' segments when `split`,
/// the first `inherited` of them past the head mapped from a predecessor.
fn start_split(
    workers: SimWorkers,
    split: Option<u32>,
) -> (
    CommandSender,
    Arc<Mutex<SimWorkers>>,
    thread::JoinHandle<()>,
) {
    let mut sim = SimEngine::new();
    sim.set_text_len(10_000);
    sim.set_num_units(SEGMENTS * SEGMENT_UNITS);
    sim.set_block_size(BLOCK);
    if let Some(inherited) = split {
        sim.set_segments(SEGMENTS, SEGMENT_UNITS, inherited);
    }
    let workers = Arc::new(Mutex::new(workers));
    let scheduler = support::scheduler_with(
        Box::new(Segmented {
            inner: SimWorker::new(sim),
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
        generated_images: Vec::new(),
        trajectory: Default::default(),
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

/// The answer to a change of the cache's storage, within ten seconds.
fn resize(handle: &CommandSender, change: CacheChange) -> Result<Vec<String>, String> {
    handle
        .resize_cache(change)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers")
}

/// A request holding the tail that cannot get a unit to continue is
/// recomputed at once, though the shrink has no bound: the request holding
/// units 1..=20 and more has none left to grow by once the shrink withdraws
/// units 16..32, so it ends as migratable, and the workers give the tail
/// back.
#[test]
fn room_recomputes_the_holders_of_the_tail_that_cannot_continue() {
    let (handle, workers, thread) = start(SimWorkers::filling(3_500));
    let mut holder = handle.submit(request(1, 20, 10_000)).unwrap();
    assert!(next_token(&mut holder, Duration::from_secs(10)).is_some());

    assert!(
        resize(
            &handle,
            CacheChange::MakeRoom {
                bytes: 3_500,
                devices: Default::default(),
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
        "the request that cannot continue is recomputed"
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
    let (handle, workers, thread) = start(SimWorkers::filling(3_500));
    assert!(
        resize(
            &handle,
            CacheChange::MakeRoom {
                bytes: 3_500,
                devices: Default::default(),
            },
        )
        .is_ok()
    );
    assert_eq!(workers.lock().unwrap().held, 4);

    // Another epoch's process of 1.5 segments' bytes leaves two and a half
    // segments' bytes of the quota: two segments grow back, although the
    // device has more free.
    workers.lock().unwrap().other_epoch_bytes = 1_500;
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
    let (handle, workers, thread) = start(SimWorkers::filling(3_500));
    let mut holder = handle
        .submit(request_of(1, 26 * BLOCK as usize + 6, 8))
        .unwrap();
    assert!(next_token(&mut holder, Duration::from_secs(10)).is_some());

    let made = handle.resize_cache(CacheChange::MakeRoom {
        bytes: 1_500,
        devices: Default::default(),
    });
    assert!(
        matches!(terminal(&mut holder), Some(Event::Finished { .. })),
        "the request holding the tail runs to its end"
    );
    let reports = made
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers")
        .unwrap();
    let plan: serde_json::Value = serde_json::from_str(&reports[0]).unwrap();
    assert_eq!(plan["segments_kept"], 6);

    // The room is there: nothing more is given back.
    assert!(
        resize(
            &handle,
            CacheChange::MakeRoom {
                bytes: 1_500,
                devices: Default::default()
            }
        )
        .is_ok()
    );
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

    let error = resize(
        &handle,
        CacheChange::MakeRoom {
            bytes: 7_500,
            devices: Default::default(),
        },
    )
    .unwrap_err();
    assert!(
        error.contains("7500 bytes of additional device storage do not fit"),
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

#[test]
fn room_accounts_for_each_devices_allocation_and_dedicated_experts() {
    let (handle, _workers, thread) = start(SimWorkers {
        experts: true,
        expert_process: Some(900),
        ..SimWorkers::filling(3_500)
    });
    let request = |expert_bytes| CacheChange::MakeRoom {
        bytes: 0,
        devices: [([7; 16], 1_500), ([8; 16], expert_bytes)].into(),
    };
    // A dedicated expert GPU cannot borrow the attention GPU's cache room.
    let error = resize(&handle, request(101)).unwrap_err();
    assert!(
        error.contains("900 of a quota of 1000 bytes booked"),
        "{error}"
    );
    let mut served = handle.submit(request_of(1, 8, 2)).unwrap();
    assert!(matches!(
        terminal(&mut served),
        Some(Event::Finished { .. })
    ));

    let reports = resize(&handle, request(100)).unwrap();
    let plans = reports
        .iter()
        .map(|report| serde_json::from_str::<serde_json::Value>(report).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(plans.len(), 2);
    assert!(plans.iter().all(|plan| plan["feasible"] == true));
    let attention = plans.iter().find(|plan| plan["bytes"] == 1_500).unwrap();
    assert_eq!(attention["segments_kept"], 6);
    let experts = plans.iter().find(|plan| plan["bytes"] == 100).unwrap();
    assert_eq!(experts["segments_held"], 0);
    let mut served = handle.submit(request_of(2, 8, 2)).unwrap();
    assert!(matches!(
        terminal(&mut served),
        Some(Event::Finished { .. })
    ));
    handle.shutdown();
    thread.join().unwrap();
}

/// Two requests that each grow past half the cache: once neither can get
/// the units its next call needs and nothing in flight frees any, the one
/// admitted last is recomputed (it ends as migratable after the tokens it
/// generated), and the other runs to its end on the units it let go.
#[test]
fn requests_that_wait_for_each_others_units_recompute_the_last_admitted() {
    let (handle, _workers, thread) = start(SimWorkers::filling(0));
    // Each takes 10 pages of prompt and 10 more of the 160 tokens it
    // generates: 40 pages of the 31 units past the sentinel.
    let mut first = handle.submit(request(1, 10, 10 * BLOCK as usize)).unwrap();
    let mut second = handle.submit(request(2, 10, 10 * BLOCK as usize)).unwrap();
    assert!(
        matches!(
            terminal(&mut second),
            Some(Event::Migratable {
                cause: MigrationCause::Drain,
                ..
            })
        ),
        "the request admitted last runs on or ends otherwise"
    );
    assert!(matches!(
        terminal(&mut first),
        Some(Event::Finished {
            completion_tokens: 160,
            ..
        })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A request alone whose sequence grows past the whole cache ends where the
/// cache is full, as it would at its token limit: giving up any request would
/// not let it grow, so it is not left to starve until the watchdog fails it.
#[test]
fn a_request_longer_than_the_cache_ends_at_the_length_the_cache_holds() {
    let (handle, _workers, thread) = start(SimWorkers::filling(0));
    // 10 pages of prompt and up to 40 more of the 640 tokens it may generate:
    // 50 pages, and the pool has 31 units past the sentinel.
    let mut long = handle.submit(request(1, 10, 40 * BLOCK as usize)).unwrap();
    match terminal(&mut long) {
        Some(Event::Finished {
            reason: FinishReason::MaxTokens,
            completion_tokens,
            ..
        }) => {
            // The 21 pages past the prompt hold what it generated.
            let held = 21 * BLOCK as usize;
            assert!(
                (held - BLOCK as usize..=held + 1).contains(&completion_tokens),
                "{completion_tokens} tokens"
            );
        }
        other => panic!("the request ends at the length the cache holds: {other:?}"),
    }

    handle.shutdown();
    thread.join().unwrap();
}

/// An epoch parked for a successor that maps its cache lends it the
/// segments past its first that hold no page a request in flight continues
/// from, and allocates none of their units until a resume takes them back:
/// a request that needs more than the units it kept waits for that resume.
#[test]
fn a_parked_epoch_lends_the_segments_no_request_in_flight_continues_from() {
    let (handle, _workers, thread) = start_split(SimWorkers::filling(0), Some(0));
    let mut running = handle.submit(request(1, 2, 10_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut generated = 0;
    while generated < 3 && Instant::now() < deadline {
        match running.try_recv() {
            Ok(Event::TextToken { .. }) => generated += 1,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    let parked = handle
        .park(true)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers")
        .expect("the scheduler parks");
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Migratable { .. })
    ));

    // The request in flight continues from the pages it published under
    // its private alias; every other segment past the first is lent.
    let continued: std::collections::BTreeSet<u32> = parked
        .pages
        .iter()
        .filter(|page| page.resume.is_some())
        .flat_map(|page| page.units.iter().copied())
        .collect();
    assert!(!continued.is_empty(), "{parked:?}");
    let lent: Vec<_> = (1..SEGMENTS)
        .map(|segment| segment * SEGMENT_UNITS..(segment + 1) * SEGMENT_UNITS)
        .filter(|units| !continued.iter().any(|unit| units.contains(unit)))
        .collect();
    assert!(!lent.is_empty());
    assert_eq!(parked.lent, lent);

    // The units kept: those past the sentinel less the lent ones. The waiting
    // request asks for one page more than the epoch keeps, and its prompt
    // shares no prefix with the request in flight: a request draws its token
    // before it needs a page for it, and a shared prefix is served from the
    // pages the request in flight publishes, so either would let it fit in
    // the units the epoch kept and the case would test nothing.
    let kept = SEGMENTS * SEGMENT_UNITS - 1 - lent.len() as u32 * SEGMENT_UNITS;
    let resumed = handle
        .resume(false)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(resumed.is_ok(), "{resumed:?}");
    let mut asking = request(2, kept + 1, 1);
    asking.prompt_token_ids = vec![9; asking.prompt_token_ids.len()];
    let mut waiting = handle.submit(asking).unwrap();
    thread::sleep(Duration::from_millis(500));
    assert!(
        std::iter::from_fn(|| waiting.try_recv().ok()).all(|event| !event.is_terminal()),
        "the request ended within the units the epoch kept"
    );
    let reclaimed = handle
        .resume(true)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(reclaimed.is_ok(), "{reclaimed:?}");
    assert!(matches!(
        terminal(&mut waiting),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A successor promoted with the segments its predecessor lent maps them
/// writable (`uniserve.adopt_cache`), counts their units in what its cache
/// holds on trial, and allocates them: the requests handed over fit, and a
/// request that needs its own free units and the lent ones runs.
#[test]
fn a_successor_allocates_the_segments_its_predecessor_lent() {
    // Segments 1 to 4 (units 4 to 19) map the predecessor's units 0 to 15.
    let (handle, workers, thread) = start_split(SimWorkers::filling(0), Some(4));
    let parked = scheduler::ParkedCache {
        pages: Vec::new(),
        lent: Vec::new(),
        held_units: 20,
        requests: 1,
        reserved_units: 0,
        reserving: 0,
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
    let lent = scheduler::ParkedCache {
        lent: vec![4..8, 12..16],
        ..parked
    };
    let promoted = handle
        .promote(lent, true, false)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(promoted.is_ok(), "{promoted:?}");
    assert!(
        workers
            .lock()
            .unwrap()
            .hooks
            .contains(&r#"uniserve.adopt_cache {"borrowed":[[8,12],[16,20]]}"#.to_owned()),
        "{:?}",
        workers.lock().unwrap().hooks
    );

    // Its own 15 free units and the 8 lent hold a request of 22 prompt
    // pages and the page of its token.
    let mut request = handle.submit(request(1, 22, 1)).unwrap();
    assert!(matches!(
        terminal(&mut request),
        Some(Event::Finished { .. })
    ));

    handle.shutdown();
    thread.join().unwrap();
}

/// A sleep recomputes the requests that run, gives the storage back, and
/// admits nothing until the wake.
#[test]
fn a_sleeping_scheduler_admits_nothing_until_it_wakes() {
    let (handle, workers, thread) = start(SimWorkers::filling(0));
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

/// A main process gives the cache's storage to a weight update's load before
/// it asks for the update: `abort` ends every request the scheduler holds,
/// the one running and the one queued, and the workers then sleep.
#[test]
fn an_aborting_sleep_ends_every_request_before_the_workers_sleep() {
    let (handle, workers, thread) = start(SimWorkers::filling(0));
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    assert!(resize(&handle, CacheChange::SleepAborting).is_ok());
    assert!(matches!(
        terminal(&mut running),
        Some(Event::Finished {
            reason: FinishReason::Aborted,
            ..
        })
    ));
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec!["uniserve.sleep".to_owned()]
    );

    // Asleep, the update needs no second sleep and no wake of its own: the
    // process that put the workers to sleep wakes them.
    let updated = handle
        .update_weights("/checkpoints/next".to_owned(), WeightsInFlight::Abort)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    assert!(updated.is_ok(), "{updated:?}");
    assert!(resize(&handle, CacheChange::Wake).is_ok());
    assert_eq!(
        workers.lock().unwrap().hooks,
        vec![
            "uniserve.sleep".to_owned(),
            r#"uniserve.update_weights {"model_path":"/checkpoints/next"}"#.to_owned(),
            "uniserve.wake".to_owned(),
        ]
    );

    handle.shutdown();
    thread.join().unwrap();
}

/// A held scheduler forms no batch and answers once those in flight completed.
/// Every request keeps its state through the hold, and generates on from the
/// token it reached once the hold ends.
#[test]
fn a_held_scheduler_keeps_its_requests_and_serves_on_after() {
    let (handle, _workers, thread) = start(SimWorkers::filling(0));
    let mut running = handle.submit(request(1, 2, 150)).unwrap();
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());

    assert!(resize(&handle, CacheChange::Hold { held: true }).is_ok());
    // What the batches in flight produced arrives, and then nothing does.
    while next_token(&mut running, Duration::from_millis(200)).is_some() {}
    assert_eq!(next_token(&mut running, Duration::from_millis(300)), None);
    let mut waiting = handle.submit(request(2, 2, 4)).unwrap();
    assert_eq!(next_token(&mut waiting, Duration::from_millis(300)), None);

    assert!(resize(&handle, CacheChange::Hold { held: false }).is_ok());
    assert!(next_token(&mut running, Duration::from_secs(10)).is_some());
    assert!(matches!(
        terminal(&mut waiting),
        Some(Event::Finished { .. })
    ));
    assert!(matches!(
        terminal(&mut running),
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
    let (handle, workers, thread) = start(SimWorkers::filling(0));
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
    let (handle, workers, thread) = start(SimWorkers::filling(0));
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
    let (handle, _workers, thread) = start(SimWorkers::filling(0));
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
