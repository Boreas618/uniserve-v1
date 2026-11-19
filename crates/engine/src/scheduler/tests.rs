//! The scheduler's end of its queue pair, over a real `storage::QueuePair` whose buffer pool end
//! the tests play by hand: they read the records the scheduler wrote and write completions.
//! The executor is a recording double of the [`Executor`] interface.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::batch::cache::CacheGroupKind;
use crate::batch::{Phase, Shape, Variant};
use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, Sender};
use proto::config::{
    BlockConfig, BlockDistribution, HostBlock, SchedulerConfig, Tuning, WorkerGroupConfig,
    WorkerRank,
};
use proto::request::{GenerationLimits, GenerationRequest};
use scheduling::limits::Parameters;
use scheduling::view::Snapshot;
use storage::runtime::{self, Class, Completion, GroupId, OwnerId, Region, Tier};
use storage::{Key, QueuePair};

use super::kv::{Cached, DEVICE};
use super::policy::ceilings;
use super::policy::tests::{Fixture, fixture, packaged, packaged_arguments};
use super::{
    Blocks, Capacity, Executor, Message, Output, Policy, Request, RequestStatus, Scheduler,
    WorkerGroup,
};

pub(super) const A: GroupId = GroupId(0);
pub(super) const B: GroupId = GroupId(1);

/// The class of KV blocks, one unit each, in a region of each group's own memory.
pub(super) const KV: u8 = 0;
/// A sized class both groups take from one region: a product one group writes and the other
/// reads.
pub(super) const PRODUCT: u8 = 1;

/// The owners of request 1: one on each group, in slot rows 3 and 4.
pub(super) const OWNER_A: OwnerId = OwnerId(10);
const OWNER_B: OwnerId = OwnerId(11);

/// What the executor was handed.
#[derive(Default)]
struct Recorded {
    stamps: Vec<(GroupId, runtime::Request)>,
    maps: Vec<(
        GroupId,
        u32,
        runtime::RequestId,
        runtime::Request,
        Completion,
    )>,
}

struct Recording(Arc<Mutex<Recorded>>);

impl Executor for Recording {
    fn submit(&self, _: crate::batch::Batch) {}

    fn map(
        &self,
        group: GroupId,
        slot: u32,
        id: runtime::RequestId,
        request: runtime::Request,
        completion: Completion,
    ) {
        let mut recorded = self.0.lock().unwrap();
        recorded.maps.push((group, slot, id, request, completion));
    }

    fn stamp(&self, group: GroupId, requests: Vec<runtime::Request>) {
        let mut recorded = self.0.lock().unwrap();
        recorded
            .stamps
            .extend(requests.into_iter().map(|request| (group, request)));
    }
}

/// A scheduler of two groups on its queue pair, and the buffer pool's end of that queue pair.
pub(super) struct Bench {
    pub(super) scheduler: Scheduler,
    pub(super) pool: QueuePair,
    executor: Arc<Mutex<Recorded>>,
}

/// The bench of [`built`] over groups A and B, both as [`capacity`] states them, once the
/// scheduler's startup `List` was answered with an empty `Listing`: each case starts with an
/// empty cache and nothing outstanding.
pub(super) fn bench(ring_bytes: usize) -> Bench {
    let groups = configs().into_iter().map(|config| (config, capacity()));
    let config = SchedulerConfig::default();
    let policy = (idle_policy(), Parameters::default());
    let Ok(mut bench) = built(ring_bytes, config, groups.collect(), policy) else {
        panic!("the startup checks pass the bench's groups");
    };

    bench.scheduler.post().unwrap();
    let taken = bench.take();
    let [(list, runtime::Request::List)] = taken[..] else {
        panic!("the scheduler starts with a `List`: {taken:?}");
    };
    bench.complete(list, &Completion::Listing { keys: Vec::new() });
    assert_eq!(bench.answers().len(), 1);
    bench
}

/// The configurations of groups A and B: one rank each, on one host.
fn configs() -> Vec<WorkerGroupConfig> {
    WorkerGroupConfig::data_parallel(
        &["localhost".to_owned()],
        proto::config::DeviceKind::Cpu,
        2,
        1,
        2,
        BTreeMap::new(),
    )
}

/// The session of the bench's scheduler over the first `count` of groups A and B: regions
/// numbered 4 to 7 (the KV regions of groups A and B, the product region both state, the slot
/// rows), class `KV` of one-unit blocks and the sized class `PRODUCT`, which is the class of
/// trajectories too, and the capacity words of region `n` at register `2 + 3 * (n - 4)`.
fn layouts(count: usize) -> (runtime::Open, runtime::Opened) {
    let region = |name: &str, units| Region {
        name: name.to_owned(),
        tier: Tier::Device,
        space: name.to_owned(),
        unit_bytes: 1 << 20,
        part_bytes: 1 << 20,
        reserved: 1,
        units,
        limit: units,
        segments: Vec::new(),
        memory: String::new(),
    };
    let group = |id: GroupId, kv: u8| runtime::Group {
        id,
        memory: format!("memory-{}", id.0),
        slots: 3,
        regions: vec![vec![kv], vec![2]],
        receives: Vec::new(),
        parts: Vec::new(),
    };
    let class = |units| Class {
        units,
        host: None,
        state_in_slot: false,
        tensors: [0; 32],
        encoding: [0; 32],
    };
    let layout = runtime::Open {
        version: runtime::VERSION,
        epoch: 1,
        trial: false,
        regions: vec![
            region("kv-a", 64),
            region("kv-b", 64),
            region("products", 64),
            region("slots", 16),
        ],
        classes: vec![class(1), class(0)],
        trajectory: PRODUCT,
        groups: vec![group(A, 0), group(B, 1)],
        schedulers: vec![[A, B][..count].to_vec()],
    };
    let opened = runtime::Opened {
        regions: vec![(4, 64), (5, 64), (6, 64), (7, 16)],
        segments: Vec::new(),
        limits: Vec::new(),
        registers: vec![runtime::Registers {
            completed: vec![0, 1],
            capacity: vec![(4, 2), (5, 5), (6, 8), (7, 11)],
        }],
        session: 0,
    };
    (layout, opened)
}

/// The session of the bench's scheduler over groups A and B, with its groups as [`capacity`]
/// states them.
pub(super) fn session() -> (runtime::Open, runtime::Opened, Vec<WorkerGroup>) {
    let (layout, opened) = layouts(2);
    let groups = configs()
        .into_iter()
        .zip(&layout.groups)
        .map(|(config, group)| WorkerGroup::new(config, capacity(), group.clone()).unwrap())
        .collect();
    (layout, opened, groups)
}

/// A scheduler of `config` over the first `groups.len()` of groups A and B, each with its
/// configuration and capacity, serving `policy` (a module and its package's arguments), on its
/// queue pair with rings of `ring_bytes`, and the buffer pool's end of that queue pair; or the
/// constructor's refusal. The session is [`layouts`]'.
fn built(
    ring_bytes: usize,
    config: SchedulerConfig,
    groups: Vec<(WorkerGroupConfig, Capacity)>,
    policy: (Arc<wasmtime::Module>, Parameters),
) -> Result<Bench, super::Error> {
    let (layout, opened) = layouts(groups.len());
    let pool = QueuePair::create(ring_bytes, 14).unwrap();
    let engine = QueuePair::open(pool.duplicate().unwrap(), true).unwrap();

    let executor = Arc::new(Mutex::new(Recorded::default()));
    let (output, _) = crossbeam_channel::unbounded();
    let scheduler = Scheduler::new(
        config,
        GenerationLimits::default(),
        &layout,
        &opened,
        0,
        engine,
        groups,
        policy.0,
        policy.1,
        Box::new(Recording(Arc::clone(&executor))),
        output,
        Arc::new(crate::batch::Generations::default()),
        crate::scheduler::tests::tuned(),
    )?;

    Ok(Bench {
        scheduler,
        pool,
        executor,
    })
}

pub(crate) fn capacity() -> Capacity {
    Capacity {
        phases: 0,
        slots: 16,
        kv_cache: None,
        cache_groups: Vec::new(),
        seeds: Vec::new(),
        buckets: Vec::new(),
        rows: [0; crate::batch::Phase::COUNT],
        max_model_len: 4096,
        denoise_steps: 0,
        bos_token_id: 0,
        eos_token_ids: Vec::new(),
        token_history: 0,
        feature_bytes: 0,
        encoder_identity: String::new(),
        latent_seed: 0,
        media_bytes: Vec::new(),
        pipeline: None,
        call_window: 2,
    }
}

/// Request 1, a token request of three prompt tokens waiting for its admission's answers,
/// attached to the groups `attachments` names, each with its owner and slot row.
pub(super) fn attached(attachments: &[(GroupId, OwnerId, u32)]) -> Request {
    let id = proto::request::RequestId(1);
    let input = GenerationRequest {
        trace: 0,
        request_id: id,
        prompt_token_ids: vec![1, 2, 3],
        multimodal_inputs: Default::default(),
        sampling: Default::default(),
        max_und_tokens: 4,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        generated_token_ids: Vec::new(),
        adapter: 0,
        hand_off: false,
        lend: None,
    };
    let now = Instant::now();
    Request {
        id,
        input: input.into(),
        arrival: now,
        status: RequestStatus::WaitingForRemoteKvs,
        placement: [None; crate::batch::Phase::COUNT],
        attachments: attachments
            .iter()
            .map(|&(group, owner, slot)| (group, owner, slot, false))
            .collect(),
        scheduled: [0; crate::batch::Phase::COUNT],
        completed: [0; crate::batch::Phase::COUNT],
        early: Vec::new(),
        blocks: vec![Blocks::default(); 2],
        queued: true,
        pending: 0,
        token_ids: vec![1, 2, 3],
        num_prompt_tokens: 3,
        num_computed_tokens: 0,
        num_cached_tokens: 0,
        num_host_tokens: 0,
        listing: 0,
        keys: Vec::new(),
        spec_token_ids: Vec::new(),
        num_draft_tokens: 0,
        num_encoded_inputs: 0,
        encodings: Vec::new(),
        unpinned: Vec::new(),
        crossings: Vec::new(),
        num_acknowledged_tokens: 0,
        terminal: None,
        event_credit: 64,
        event_window: 64,
        draft_limit: 0,
        last_progress: now,
        dirty: false,
        path: None,
    }
}

impl Bench {
    /// Request 1, attached to the groups `attachments` names, with live owners ([`attached`]).
    pub(super) fn attach(&mut self, attachments: &[(GroupId, OwnerId, u32)]) {
        let request = attached(attachments);
        let id = request.id;
        for &(_, owner, _) in attachments {
            self.scheduler.owners.insert(owner, id);
        }
        self.scheduler.requests.insert(id, request);
    }

    /// Every request the scheduler wrote that the buffer pool has not taken, with its number.
    pub(super) fn take(&mut self) -> Vec<(runtime::RequestId, runtime::Request)> {
        let mut record = Vec::new();
        let mut taken = Vec::new();
        while self.pool.next_request(&mut record).unwrap() {
            taken.push(runtime::Request::decode(&record).unwrap());
        }
        taken
    }

    /// Answers request `id` with `completion`.
    pub(super) fn complete(&mut self, id: runtime::RequestId, completion: &Completion) {
        let mut record = Vec::new();
        completion.encode(id, &mut record);
        assert!(self.pool.complete(&record).unwrap());
    }

    /// Every completion the scheduler applies, as it reads them.
    pub(super) fn answers(
        &mut self,
    ) -> Vec<(
        runtime::RequestId,
        Option<OwnerId>,
        runtime::Request,
        Completion,
    )> {
        let mut answers = Vec::new();
        while let Some(answer) = self.scheduler.next_completion().unwrap() {
            answers.push(answer);
        }
        answers
    }

    pub(super) fn stamps(&self) -> Vec<(GroupId, runtime::Request)> {
        std::mem::take(&mut self.executor.lock().unwrap().stamps)
    }

    fn maps(
        &self,
    ) -> Vec<(
        GroupId,
        u32,
        runtime::RequestId,
        runtime::Request,
        Completion,
    )> {
        std::mem::take(&mut self.executor.lock().unwrap().maps)
    }
}

pub(super) fn key(byte: u8) -> Key {
    Key([byte; 32])
}

fn allocate(owner: OwnerId, class: u8, at: u32, count: u32) -> runtime::Request {
    runtime::Request::Allocate {
        owner,
        class,
        at,
        count,
        units: if class == PRODUCT { 4 } else { 0 },
    }
}

fn release(owner: OwnerId, class: u8, at: u32, count: u32) -> runtime::Request {
    runtime::Request::Release {
        owner,
        class,
        at,
        count,
        after: Vec::new(),
    }
}

#[test]
fn each_request_gets_a_fresh_number_and_each_completion_finds_its_request_in_any_order() {
    let mut bench = bench(1024);
    let submitted = [
        runtime::Request::Evict { keys: vec![key(1)] },
        runtime::Request::Evict { keys: vec![key(2)] },
        runtime::Request::Victims {
            region: 4,
            keys: vec![key(3)],
        },
    ];
    for request in submitted.clone() {
        bench.scheduler.submit(request);
    }
    bench.scheduler.post().unwrap();

    // The buffer pool takes them in the order submitted, under rising numbers (BP §7).
    let taken = bench.take();
    let requests: Vec<_> = taken.iter().map(|(_, request)| request.clone()).collect();
    assert_eq!(requests, submitted);
    assert!(taken.windows(2).all(|pair| pair[0].0 < pair[1].0));

    // A later request's number is above every number before it.
    bench
        .scheduler
        .submit(runtime::Request::Evict { keys: vec![key(4)] });
    bench.scheduler.post().unwrap();
    let later = bench.take();
    assert_eq!(later.len(), 1);
    assert!(later[0].0 > taken[2].0);

    // Completions arrive in another order than their requests; each carries only its number.
    let answered = [
        (taken[2].0, Completion::Done),
        (later[0].0, Completion::Evicted { keys: vec![key(4)] }),
        (taken[0].0, Completion::Evicted { keys: vec![key(1)] }),
        (taken[1].0, Completion::Evicted { keys: vec![] }),
    ];
    for (id, completion) in &answered {
        bench.complete(*id, completion);
    }

    let everything: Vec<_> = taken.iter().chain(&later).cloned().collect();
    let expected: Vec<_> = answered
        .iter()
        .map(|(id, completion)| {
            let request = everything
                .iter()
                .find(|(number, _)| number == id)
                .map(|(_, request)| request.clone())
                .unwrap();
            (*id, None, request, completion.clone())
        })
        .collect();
    assert_eq!(bench.answers(), expected);
}

#[test]
fn a_full_submission_queue_holds_requests_and_merges_a_further_allocate_of_one_owner_and_class() {
    let mut bench = bench(128);
    bench.attach(&[(A, OWNER_A, 3)]);

    // Two records of 56 bytes leave 16 of the ring's 128: no room for an `Allocate` (40).
    for byte in [1, 2] {
        bench.scheduler.submit(runtime::Request::Victims {
            region: 4,
            keys: vec![key(byte)],
        });
    }
    bench.scheduler.submit(allocate(OWNER_A, KV, 0, 1));
    bench.scheduler.post().unwrap();

    // The next need of the same owner and class raises the held one's count.
    bench.scheduler.submit(allocate(OWNER_A, KV, 1, 1));
    bench.scheduler.post().unwrap();

    let taken = bench.take();
    assert_eq!(taken.len(), 2, "only the victims fit the submission queue");
    for (id, _) in &taken {
        bench.complete(*id, &Completion::Done);
    }
    assert_eq!(bench.answers().len(), 2);

    bench.scheduler.post().unwrap();
    let requests: Vec<_> = bench
        .take()
        .into_iter()
        .map(|(_, request)| request)
        .collect();
    assert_eq!(requests, [allocate(OWNER_A, KV, 0, 2)]);
}

#[test]
fn nothing_is_written_while_the_completion_queue_lacks_room_for_every_outstanding_answer() {
    let mut bench = bench(256);
    bench.attach(&[(A, OWNER_A, 3)]);

    // A `Load` of one key takes 72 bytes of the submission queue. Its answer may be a `Copy` of
    // that key with one run to read, one to write and one key evicted for it: a record of 108
    // bytes, 120 in the completion queue. Two answers fit its 256 bytes; a third does not.
    let load = |at| runtime::Request::Load {
        owner: OWNER_A,
        class: KV,
        at,
        keys: vec![key(at as u8)],
    };
    for at in 0..3 {
        bench.scheduler.submit(load(at));
    }
    bench.scheduler.post().unwrap();
    let taken = bench.take();
    assert_eq!(
        taken
            .iter()
            .map(|(_, request)| request.clone())
            .collect::<Vec<_>>(),
        [load(0), load(1)]
    );

    // Still no room while both are outstanding.
    bench.scheduler.post().unwrap();
    assert!(bench.take().is_empty());

    // Once an answer is read, the third is written.
    bench.complete(taken[0].0, &Completion::Full { region: 4 });
    assert_eq!(bench.answers().len(), 1);
    bench.scheduler.post().unwrap();
    let requests: Vec<_> = bench
        .take()
        .into_iter()
        .map(|(_, request)| request)
        .collect();
    assert_eq!(requests, [load(2)]);
}

#[test]
fn a_stamped_request_is_written_once_every_group_holding_its_entries_stamped_it() {
    let mut bench = bench(1024);
    bench.attach(&[(A, OWNER_A, 3), (B, OWNER_B, 4)]);

    // A product block both groups map: the writer's group and the reader's stamp its release.
    bench.scheduler.submit(release(OWNER_A, PRODUCT, 0, 1));
    let mut stamps = bench.stamps();
    stamps.sort_by_key(|(group, _)| *group);
    assert_eq!(
        stamps,
        [
            (A, release(OWNER_A, PRODUCT, 0, 1)),
            (B, release(OWNER_A, PRODUCT, 0, 1))
        ]
    );

    // A KV block lies in the memory of the owner's group alone.
    bench.scheduler.submit(release(OWNER_A, KV, 0, 1));
    assert_eq!(bench.stamps(), [(A, release(OWNER_A, KV, 0, 1))]);

    // A block allocated at the position the product release frees waits behind it.
    bench.scheduler.submit(allocate(OWNER_A, PRODUCT, 0, 1));
    bench.scheduler.post().unwrap();
    assert!(bench.take().is_empty());

    let stamped = |group: GroupId, sequence, class| runtime::Request::Release {
        owner: OWNER_A,
        class,
        at: 0,
        count: 1,
        after: vec![(group, sequence)],
    };
    bench.scheduler.stamped(vec![stamped(A, 7, PRODUCT)]);
    bench.scheduler.post().unwrap();
    assert!(bench.take().is_empty(), "group B has not stamped it");

    bench
        .scheduler
        .stamped(vec![stamped(A, 8, KV), stamped(B, 3, PRODUCT)]);
    bench.scheduler.post().unwrap();
    let requests: Vec<_> = bench
        .take()
        .into_iter()
        .map(|(_, request)| request)
        .collect();
    assert_eq!(
        requests,
        [
            runtime::Request::Release {
                owner: OWNER_A,
                class: KV,
                at: 0,
                count: 1,
                after: vec![(A, 8)],
            },
            runtime::Request::Release {
                owner: OWNER_A,
                class: PRODUCT,
                at: 0,
                count: 1,
                after: vec![(A, 7), (B, 3)],
            },
            allocate(OWNER_A, PRODUCT, 0, 1),
        ]
    );
}

#[test]
fn entries_a_release_handed_after_their_request_names_are_not_mapped() {
    let mut bench = bench(1024);
    bench.attach(&[(A, OWNER_A, 3)]);

    bench.scheduler.submit(allocate(OWNER_A, KV, 0, 2));
    bench.scheduler.submit(allocate(OWNER_A, PRODUCT, 0, 1));
    bench.scheduler.post().unwrap();
    let taken = bench.take();

    // Position 1 of the KV table is given up before the `Allocate` that fills it is answered.
    bench.scheduler.submit(release(OWNER_A, KV, 1, 1));
    let kv = Completion::Allocated {
        runs: vec![(4, 9, 1), (4, 10, 1)],
        evicted: Vec::new(),
    };
    let product = Completion::Allocated {
        runs: vec![(6, 0, 4)],
        evicted: Vec::new(),
    };
    bench.complete(taken[0].0, &kv);
    bench.complete(taken[1].0, &product);

    // The KV answer is dropped; the product's, whose entries the release does not name, is
    // handed to the owner's group under the owner's slot and applied.
    let answers = bench.answers();
    assert_eq!(
        answers,
        [(
            taken[1].0,
            Some(OWNER_A),
            allocate(OWNER_A, PRODUCT, 0, 1),
            product.clone()
        )]
    );
    assert_eq!(
        bench.maps(),
        [(A, 3, taken[1].0, allocate(OWNER_A, PRODUCT, 0, 1), product)]
    );
}

#[test]
fn a_completion_for_a_forgotten_owner_is_dropped_and_its_copy_given_up() {
    let mut bench = bench(1024);
    bench.attach(&[(A, OWNER_A, 0)]);

    let load = runtime::Request::Load {
        owner: OWNER_A,
        class: KV,
        at: 0,
        keys: vec![key(1)],
    };
    bench.scheduler.submit(runtime::Request::Attach {
        owner: OWNER_A,
        group: A,
    });
    bench.scheduler.submit(load);
    bench.scheduler.post().unwrap();
    let taken = bench.take();

    // The admission is returned: its owner is detached before its answers arrive.
    bench.scheduler.submit(runtime::Request::Detach {
        owner: OWNER_A,
        after: Vec::new(),
    });
    bench.stamps();

    bench.complete(taken[0].0, &Completion::Attached { slot: 3 });
    let copy = Completion::Copy {
        group: A,
        keys: vec![key(1)],
        from: vec![(9, 0, 1)],
        to: vec![(4, 2, 1)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    bench.complete(taken[1].0, &copy);

    assert!(bench.answers().is_empty());
    assert!(bench.maps().is_empty());
    // The copy no worker will run is given up through the executor.
    assert_eq!(
        bench.stamps(),
        [(
            A,
            runtime::Request::Finish {
                copy: taken[1].0,
                ok: false,
                after: Vec::new(),
            }
        )]
    );
}

#[test]
fn the_answer_to_a_copys_finish_is_mapped_and_applied_as_the_copys_result() {
    let mut bench = bench(1024);
    bench.attach(&[(A, OWNER_A, 3)]);

    let load = runtime::Request::Load {
        owner: OWNER_A,
        class: KV,
        at: 0,
        keys: vec![key(1)],
    };
    bench.scheduler.submit(load.clone());
    bench.scheduler.post().unwrap();
    let taken = bench.take();
    let [(id, _)] = taken[..] else {
        panic!("one request was written");
    };

    let copy = Completion::Copy {
        group: A,
        keys: vec![key(1)],
        from: vec![(9, 0, 1)],
        to: vec![(4, 2, 1)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    bench.complete(id, &copy);
    assert_eq!(
        bench.answers(),
        [(id, Some(OWNER_A), load.clone(), copy.clone())]
    );
    assert_eq!(bench.maps(), [(A, 3, id, load.clone(), copy)]);

    // The executor ran the copy and returns its `Finish`, which is written as it is.
    let finish = runtime::Request::Finish {
        copy: id,
        ok: true,
        after: vec![(A, 12)],
    };
    bench.scheduler.stamped(vec![finish.clone()]);
    bench.scheduler.post().unwrap();
    let taken = bench.take();
    let [(finished, ref written)] = taken[..] else {
        panic!("one request was written");
    };
    assert_eq!(*written, finish);

    // Its answer gives the loaded entries: handed to the copy's group with the `Finish`, and
    // applied as the result of the `Load`.
    bench.complete(finished, &Completion::Loaded);
    assert_eq!(
        bench.answers(),
        [(id, Some(OWNER_A), load, Completion::Loaded)]
    );
    assert_eq!(bench.maps(), [(A, 3, finished, finish, Completion::Loaded)]);
}

/// A `Finish` is written past held requests that wait for room in the completion queue, which
/// keep their order (SD §3.1): a `List` owes the whole ring and waits for every answer
/// outstanding, among them the copy whose `Finish` is held, which owes until that `Finish` is
/// answered. Once it is, the `List` is written, and the request behind it once the `Listing` is
/// read.
#[test]
fn a_finish_is_written_past_requests_that_wait_for_room() {
    let mut bench = bench(1024);
    bench.attach(&[(A, OWNER_A, 3)]);

    let load = runtime::Request::Load {
        owner: OWNER_A,
        class: KV,
        at: 0,
        keys: vec![key(1)],
    };
    bench.scheduler.submit(load);
    bench.scheduler.post().unwrap();
    let [(id, _)] = bench.take()[..] else {
        panic!("one request was written");
    };
    let copy = Completion::Copy {
        group: A,
        keys: vec![key(1)],
        from: vec![(9, 0, 1)],
        to: vec![(4, 2, 1)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    bench.complete(id, &copy);
    assert_eq!(bench.answers().len(), 1);

    // A `List` and an `Allocate` are held when the executor returns the copy's `Finish`.
    let finish = runtime::Request::Finish {
        copy: id,
        ok: true,
        after: vec![(A, 12)],
    };
    bench.scheduler.submit(runtime::Request::List);
    bench.scheduler.submit(allocate(OWNER_A, KV, 1, 1));
    bench.scheduler.stamped(vec![finish.clone()]);
    bench.scheduler.post().unwrap();
    let taken = bench.take();
    let [(finished, ref written)] = taken[..] else {
        panic!("the `Finish` alone was written: {taken:?}");
    };
    assert_eq!(*written, finish);

    bench.complete(finished, &Completion::Loaded);
    assert_eq!(bench.answers().len(), 1);
    bench.scheduler.post().unwrap();
    let taken = bench.take();
    let [(listed, runtime::Request::List)] = taken[..] else {
        panic!("the `List` alone was written: {taken:?}");
    };

    bench.complete(listed, &Completion::Listing { keys: Vec::new() });
    assert_eq!(bench.answers().len(), 1);
    bench.scheduler.post().unwrap();
    let written: Vec<runtime::Request> = bench
        .take()
        .into_iter()
        .map(|(_, request)| request)
        .collect();
    assert_eq!(written, [allocate(OWNER_A, KV, 1, 1)]);
}

#[test]
fn available_is_the_buffer_pools_word_less_the_units_of_allocates_not_yet_answered() {
    let mut bench = bench(1024);
    bench.attach(&[(A, OWNER_A, 3)]);

    // Region 4 (the KV region of group A): 50 units available, 3 retiring.
    bench.pool.register(2).store(50, Ordering::Release);
    bench.pool.register(3).store(3, Ordering::Release);
    // Region 6 (the products): 40 available.
    bench.pool.register(8).store(40, Ordering::Release);

    // Two KV blocks outstanding, and a held product block of 4 units.
    bench.scheduler.submit(allocate(OWNER_A, KV, 0, 2));
    bench.scheduler.post().unwrap();
    bench.scheduler.submit(allocate(OWNER_A, PRODUCT, 0, 1));
    bench.scheduler.read_capacity();

    assert_eq!(bench.scheduler.kv.available(0), 48);
    assert_eq!(bench.scheduler.kv.retiring(0), 3);
    assert_eq!(bench.scheduler.kv.available(2), 36);
}

/// A request marked since the last step is told to the policy at the next `dispatch`, however
/// often it was marked, and can be marked again once told: the packaged policy, which learns of
/// request 1 from nothing else, admits it.
#[test]
fn a_marked_request_is_told_at_the_next_dispatch() {
    let mut bench = bench(1024);
    bench.attach(&[]);
    let scheduler = &mut bench.scheduler;
    scheduler.policy = Policy::new(
        packaged("arrival"),
        packaged_arguments(""),
        Snapshot::default(),
        &ceilings([], &proto::config::PolicyBudget::default()).limits,
        &proto::config::PolicyBudget::default(),
    )
    .unwrap();
    let id = proto::request::RequestId(1);

    scheduler.mark(id);
    scheduler.mark(id);
    assert!(scheduler.requests[&id].dirty);

    let room = scheduler.room();
    let output = scheduler
        .policy
        .dispatch(&room, &mut scheduler.requests, &scheduler.groups)
        .unwrap();
    assert_eq!(output.admit, [scheduling::ids::RequestId(1)]);
    assert!(!scheduler.requests[&id].dirty);

    scheduler.mark(id);
    assert!(scheduler.requests[&id].dirty);
}

/// The room states the units of the regions the cache groups' blocks are taken from (the KV
/// regions of both groups, not the products), the free slots of the fullest serving group, the
/// policy's own budgets, and the batches in flight with those that may carry prefill work.
#[test]
fn the_room_states_the_cache_regions_the_free_slots_and_the_batches_in_flight() {
    let mut bench = bench(1024);
    let scheduler = &mut bench.scheduler;
    for group in &mut scheduler.groups {
        group.capacity.phases = 1 << Phase::Prefill.index();
    }
    scheduler.groups[0].num_requests = 10;
    scheduler.groups[1].num_requests = 4;
    scheduler.groups[1].in_flight = 2;
    scheduler.groups[1].occupied[Phase::Prefill.index()] = 1;
    *scheduler.kv.available_mut() = vec![50, 30, 20, 16];

    let room = scheduler.room();
    assert_eq!(room.kv_units, 80);
    assert_eq!(room.capacity_units, 128);
    assert_eq!(room.slots, 6);
    assert_eq!(room.token_budget, 8192);
    assert_eq!(room.call_budget, 128);
    assert_eq!(room.inflight.batches, 2);
    assert_eq!(room.inflight.prefill_batches, 1);
    assert_eq!(room.prefill_tokens, None);
}

/// A starting policy is told the waiting requests, every cached block of a cache group as a page
/// (evictable when no request holds it, on the host when no device copy is left), every other
/// cached block as an encoding, and the groups that serve phases with the phases they serve.
#[test]
fn the_snapshot_states_the_requests_the_cache_and_the_serving_groups() {
    let mut bench = bench(1024);
    bench.attach(&[]);
    let scheduler = &mut bench.scheduler;
    scheduler.waiting.push_back(proto::request::RequestId(1));
    scheduler.groups[1].capacity.phases =
        (1 << Phase::Prefill.index()) | (1 << Phase::Decode.index());

    let cached = |byte, class, tiers, pins| Cached {
        key: key(byte),
        class,
        tokens: 16,
        tiers,
        groups: 1,
        pins,
        last_used: 5,
        refused: None,
    };
    for block in [
        cached(1, KV, 0b001, 1),
        cached(2, KV, 0b010, 0),
        cached(3, PRODUCT, 0b001, 0),
    ] {
        scheduler.kv.insert(block);
    }

    let snapshot = scheduler.snapshot();
    let requests: Vec<u64> = snapshot.requests.iter().map(|view| view.id.0).collect();
    assert_eq!(requests, [1]);

    let mut pages: Vec<(u64, bool, bool)> = snapshot
        .pages
        .iter()
        .map(|page| (page.id.0, page.evictable, page.on_host))
        .collect();
    pages.sort_unstable();
    assert_eq!(
        pages,
        [
            (Cached::id(&key(1)), false, false),
            (Cached::id(&key(2)), true, true)
        ]
    );
    let encodings: Vec<u64> = snapshot.encodings.iter().map(|entry| entry.id.0).collect();
    assert_eq!(encodings, [Cached::id(&key(3))]);

    assert_eq!(snapshot.replicas.len(), 1);
    assert_eq!(snapshot.replicas[0].kinds, [Phase::Prefill, Phase::Decode]);
    let phases: Vec<(u32, Phase)> = snapshot
        .phases
        .iter()
        .map(|phase| (phase.id.0, phase.kind))
        .collect();
    assert_eq!(phases, [(3, Phase::Prefill), (4, Phase::Decode)]);
}

/// The scheduler's first request is `List` (SD §4.5): what the `Listing` states it knows, and its
/// next step installs those blocks, which no request holds, as the eviction order of their region.
#[test]
fn a_scheduler_starts_by_listing_its_sessions_cached_blocks_and_orders_them() {
    let groups = configs().into_iter().map(|config| (config, capacity()));
    let policy = (idle_policy(), Parameters::default());
    let Ok(mut bench) = built(1024, SchedulerConfig::default(), groups.collect(), policy) else {
        panic!("the startup checks pass the bench's groups");
    };

    bench.scheduler.step();
    let taken = bench.take();
    let [(list, runtime::Request::List)] = taken[..] else {
        panic!("the first request written is a `List`: {taken:?}");
    };

    // Two blocks of the KV class whose device copies lie in group A's memory (region 4).
    let block = |byte| (key(byte), KV, 16, DEVICE, 1 << A.0, 0);
    let keys = vec![block(1), block(2)];
    bench.complete(list, &Completion::Listing { keys });
    bench.scheduler.step();

    let requests: Vec<_> = bench
        .take()
        .into_iter()
        .map(|(_, request)| request)
        .collect();
    let victims = runtime::Request::Victims {
        region: 4,
        keys: vec![key(1), key(2)],
    };
    assert_eq!(requests, [victims]);
}

/// The constructor refuses a deployment its workers cannot serve (SD §7.1): no group, a group
/// of token sequences without a longest sequence, encoders of weights of different identities,
/// whose outputs no one key finds (SD §4.3), and a video decoder whose units no video encoder
/// reads on the hosts that decoded them. A KV cache of fewer slots than the stated running
/// requests runs as many requests as it holds slots.
#[test]
fn the_startup_checks_refuse_a_deployment_its_workers_cannot_serve() {
    let idle = || (idle_policy(), Parameters::default());
    let refusal = |groups| match built(1024, SchedulerConfig::default(), groups, idle()) {
        Err(super::Error::InvalidRequest(message)) => message,
        Err(other) => panic!("refused for another cause: {other}"),
        Ok(_) => panic!("the deployment passed the startup checks"),
    };
    let serving = |phase: Phase| Capacity {
        phases: 1 << phase.index(),
        ..capacity()
    };
    let [a, b] = <[WorkerGroupConfig; 2]>::try_from(configs()).unwrap();

    assert!(refusal(Vec::new()).contains("no worker group"));

    let unbounded = Capacity {
        max_model_len: 0,
        ..serving(Phase::Prefill)
    };
    assert!(refusal(vec![(a.clone(), unbounded)]).contains("longest sequence"));

    // Two groups of 16 slots for the 128 running requests of the default limits run 32.
    let cached = Capacity {
        cache_groups: vec![(CacheGroupKind::Full, 16)],
        ..serving(Phase::Prefill)
    };
    let groups = vec![(a.clone(), cached.clone()), (b.clone(), cached)];
    let Ok(bench) = built(1024, SchedulerConfig::default(), groups, idle()) else {
        panic!("a cache of fewer slots than running requests is served");
    };
    assert_eq!(bench.scheduler.max_running_requests(), 32);

    let encoding = |identity: &str| Capacity {
        encoder_identity: identity.to_owned(),
        ..serving(Phase::VisionEncode)
    };
    let differing = vec![
        (a.clone(), encoding("vision-weights-a")),
        (b.clone(), encoding("vision-weights-b")),
    ];
    assert!(refusal(differing).contains("compute their encoder outputs"));
    let alike = vec![
        (a.clone(), encoding("vision-weights-a")),
        (b.clone(), encoding("vision-weights-a")),
    ];
    assert!(built(1024, SchedulerConfig::default(), alike, idle()).is_ok());

    // Group A decodes one unit per round on its one rank, which lies on `localhost`; group B's
    // codec encodes it on its one rank, on `elsewhere` and then on `localhost`.
    let with = |mut config: WorkerGroupConfig, block: &str, host: &str| {
        let units = BlockConfig {
            distribution: Some(BlockDistribution::TemporalUnits),
            ..BlockConfig::parallel(vec![0])
        };
        config.blocks.insert(block.to_owned(), units);
        config.ranks = vec![WorkerRank {
            host: host.to_owned(),
            device: "cpu".parse().unwrap(),
        }];
        config
    };
    // Both groups serve parts of MiniMax-H3's pipeline, whose video decoder block they place.
    let media = |phase: Phase| Capacity {
        pipeline: Some(models::minimax_h3::family::pipeline()),
        ..serving(phase)
    };
    let decoder = (
        with(a, "video_decoder", "localhost"),
        media(Phase::VideoDecode),
    );
    let encoder = |host| {
        (
            with(b.clone(), HostBlock::VideoCodec.name(), host),
            media(Phase::VideoEncode),
        )
    };
    let crossing = vec![decoder.clone(), encoder("elsewhere")];
    assert!(refusal(crossing).contains("decodes video units"));
    let beside = vec![decoder, encoder("localhost")];
    assert!(built(1024, SchedulerConfig::default(), beside, idle()).is_ok());
}

/// A step's budget is the step values of the snapshot it began with, lowered to what the
/// workers captured: a step's tokens and its prefill chunk at most the widest prefill bucket's
/// tokens, its calls at most the largest bucket's rows; a value the snapshot leaves unstated is
/// the ceiling's. The batches of the step name the snapshot's revision (SD §7.1;
/// `specs/configuration.md` §7, step values).
#[test]
fn a_step_takes_the_step_values_it_began_with_within_the_workers_capture() {
    // Both groups prefill and captured one bucket of 8 rows and 512 tokens.
    let prefills = Capacity {
        phases: 1 << Phase::Prefill.index(),
        buckets: vec![Shape::prefill(8, 512, Variant::PrefillWrite)],
        ..capacity()
    };
    let groups = configs()
        .into_iter()
        .map(|config| (config, prefills.clone()));
    let policy = (packaged("arrival"), packaged_arguments(""));
    let Ok(mut bench) = built(1024, SchedulerConfig::default(), groups.collect(), policy) else {
        panic!("the startup checks pass both groups");
    };

    bench.scheduler.tune(|tuning| {
        tuning.revision = 7;
        tuning.max_batch = NonZeroUsize::new(4).unwrap();
        tuning.max_num_batched_tokens = NonZeroUsize::new(256);
        tuning.chunked_prefill_size = NonZeroUsize::new(64);
        tuning.max_queued_requests = NonZeroUsize::new(3);
        tuning.synthetic_acceptance_length = Some(2.5);
    });
    bench.scheduler.step();
    assert_eq!(
        bench.scheduler.budget,
        super::Budget {
            revision: 7,
            tokens: 256,
            calls: 4,
            chunk: 64,
            queued: Some(3),
            prefix_caching: true,
            host_tier: true,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            synthetic_acceptance_length: Some(2.5),
        }
    );

    // Values above the capture are lowered to it; unstated ones take the capture's.
    bench.scheduler.tune(|tuning| {
        tuning.revision = 8;
        tuning.max_batch = NonZeroUsize::new(9).unwrap();
        tuning.max_num_batched_tokens = None;
        tuning.chunked_prefill_size = NonZeroUsize::new(4096);
        tuning.max_queued_requests = None;
        tuning.synthetic_acceptance_length = None;
    });
    bench.scheduler.step();
    assert_eq!(
        bench.scheduler.budget,
        super::Budget {
            revision: 8,
            tokens: 512,
            calls: 8,
            chunk: 512,
            queued: None,
            prefix_caching: true,
            host_tier: true,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            synthetic_acceptance_length: None,
        }
    );
}

/// The step values a scheduler keeps beside the budget follow the snapshot from the next step
/// on: a group's queue depth and the call window it bounds, the statistics' report period, and
/// the copy budget and the cache switches the step's batches and blocks are given.
#[test]
fn a_step_takes_the_groups_depths_and_the_switches_of_its_snapshot() {
    let groups = configs().into_iter().map(|config| (config, capacity()));
    let policy = (packaged("arrival"), packaged_arguments(""));
    let Ok(mut bench) = built(1024, SchedulerConfig::default(), groups.collect(), policy) else {
        panic!("the startup checks pass both groups");
    };
    let named = bench.scheduler.groups[0].config.id.0.clone();
    let other = bench.scheduler.groups[1].config.queue_depth;

    bench.scheduler.tune(|tuning| {
        tuning.revision = 9;
        tuning
            .queue_depths
            .insert(named.clone(), NonZeroUsize::new(5).unwrap());
        tuning.report_period = std::time::Duration::from_millis(250);
        tuning.copy_budget = 4096;
        tuning.prefix_caching = false;
        tuning.host_tier = false;
    });
    bench.scheduler.step();
    let group = &bench.scheduler.groups[0];
    assert_eq!(group.config.queue_depth, 5);
    assert_eq!(group.capacity.call_window, 5);
    assert_eq!(bench.scheduler.groups[1].config.queue_depth, other);
    assert_eq!(
        bench.scheduler.config.report_period,
        std::time::Duration::from_millis(250)
    );
    let budget = bench.scheduler.budget;
    assert_eq!(budget.copy_budget, 4096);
    assert!(!budget.prefix_caching && !budget.host_tier);
}

/// A flush evicts every cached block the scheduler knows of by `Evict`, as many keys per record
/// as the submission queue takes (SD §2.4), whatever the requests in flight do.
#[test]
fn a_flush_evicts_every_cached_block_the_scheduler_knows_of() {
    // Rings of 2048 bytes: an `Evict` record holds at most (2040 - 13) / 32 = 63 keys. The
    // scheduler knows of 80 blocks on the host, more than one record, or one `Listing`, holds.
    let mut bench = bench(2048);
    let known: Vec<Key> = (0..80).map(key).collect();
    for &key in &known {
        let block = Cached {
            key,
            class: KV,
            tokens: 16,
            tiers: 0b010,
            groups: 0,
            pins: 0,
            last_used: 1,
            refused: None,
        };
        bench.scheduler.kv.insert(block);
    }

    let (reply, replied) = crossbeam_channel::bounded(1);
    bench.scheduler.apply(Message::Flush(reply));
    assert!(replied.try_recv().is_ok());

    // Each step writes what the queues have room for; the buffer pool evicts what it is named.
    let mut evicted = Vec::new();
    for _ in 0..4 {
        bench.scheduler.step();
        for (id, request) in bench.take() {
            let completion = match request {
                runtime::Request::Evict { keys } => {
                    assert!(keys.len() <= 63, "{} keys in one record", keys.len());
                    evicted.extend(keys.iter().copied());
                    Completion::Evicted { keys }
                }
                _ => Completion::Done,
            };
            bench.complete(id, &completion);
        }
    }
    evicted.sort_unstable();
    assert_eq!(evicted, known);
}

/// Runs the bench's scheduler on a thread of its own over `inbox`, its outputs sent to `output`;
/// the receiver returned answers once `run` has returned.
fn running(inbox: Receiver<Message>, output: Sender<Output>) -> Receiver<()> {
    let (ended, end) = crossbeam_channel::bounded(1);
    std::thread::spawn(move || {
        let mut bench = bench(1024);
        bench.scheduler.output = output;
        bench.scheduler.run(inbox);
        ended.send(()).unwrap();
    });
    end
}

/// The loop ends at `Shutdown`, and once no sender of its inbox is left, without a failure.
#[test]
fn the_loop_ends_at_shutdown_and_once_its_inbox_has_no_sender() {
    for shutdown in [true, false] {
        let (inbox, messages) = crossbeam_channel::unbounded();
        let (output, outputs) = crossbeam_channel::unbounded();
        let end = running(messages, output);

        if shutdown {
            inbox.send(Message::Shutdown).unwrap();
        } else {
            drop(inbox);
        }
        end.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(
            !outputs
                .try_iter()
                .any(|output| matches!(output, Output::Failed(_)))
        );
    }
}

/// A store to the host in flight when `Shutdown` arrives gets its grace time (BP §20): the
/// scheduler hands its copy to the workers, writes the copy's `Finish`, and stops once that is
/// answered with what was stored. A scheduler with nothing in flight stops at once.
#[test]
fn a_store_in_flight_at_shutdown_ends_before_the_scheduler_stops() {
    let mut idle = bench(1024);
    idle.scheduler.apply(Message::Shutdown);
    assert!(idle.scheduler.stopped);

    let mut bench = bench(1024);
    let store = runtime::Request::Store {
        group: A,
        keys: vec![key(1)],
        tier: Tier::Host,
    };
    bench.scheduler.submit(store.clone());
    bench.scheduler.post().unwrap();
    let [(id, _)] = bench.take()[..] else {
        panic!("the `Store` was written");
    };

    bench.scheduler.apply(Message::Shutdown);
    assert!(!bench.scheduler.stopped, "a store is in flight");

    // The buffer pool answers the store with its copy, which the workers are handed to run.
    let copy = Completion::Copy {
        group: A,
        keys: vec![key(1)],
        from: vec![(4, 2, 1)],
        to: vec![(9, 0, 1)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    bench.complete(id, &copy);
    bench.scheduler.step();
    assert!(
        bench
            .maps()
            .iter()
            .any(|(group, _, mapped, request, _)| *group == A
                && *mapped == id
                && *request == store),
        "the copy was handed to the workers"
    );
    assert!(!bench.scheduler.stopped, "the copy runs");

    // The workers ran it: its `Finish` is written, and its answer ends the store.
    let finish = runtime::Request::Finish {
        copy: id,
        ok: true,
        after: vec![(A, 1)],
    };
    bench.scheduler.stamped(vec![finish.clone()]);
    bench.scheduler.step();
    let taken = bench.take();
    let [(finished, ref written)] = taken[..] else {
        panic!("the `Finish` was written: {taken:?}");
    };
    assert_eq!(*written, finish);
    assert!(!bench.scheduler.stopped, "the `Finish` is unanswered");

    bench.complete(finished, &Completion::Stored { keys: vec![key(1)] });
    bench.scheduler.step();
    assert!(bench.scheduler.stopped);
}

/// The executor's failed job outside a batch stops the scheduler at once: `run` returns, and the
/// engine is told with `Output::Failed`, which carries the executor's error.
#[test]
fn a_failed_executor_job_stops_the_loop_and_reaches_the_engine() {
    let (inbox, messages) = crossbeam_channel::unbounded();
    let (output, outputs) = crossbeam_channel::unbounded();
    let end = running(messages, output);

    let failure = "worker group 1 did not apply and stamp requests";
    inbox
        .send(Message::Failed(super::Error::Executor(failure.to_owned())))
        .unwrap();
    end.recv_timeout(Duration::from_secs(10)).unwrap();

    let failed: Vec<String> = outputs
        .try_iter()
        .filter_map(|output| match output {
            Output::Failed(error) => Some(error.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(failed, [format!("an executor failed: {failure}")]);
}

/// A policy module that admits nothing, whose limits are the policy defaults: tests that drive
/// the queue pair, the views or the placement, which no policy decision reaches.
pub(crate) fn idle_policy() -> Arc<wasmtime::Module> {
    fixture(Fixture::Idle)
}

/// The step values of a scheduler of the tests, which no deployment changes: the default batch,
/// and the capture's tokens and the default chunk.
pub(crate) fn tuned() -> Arc<ArcSwap<Tuning>> {
    Arc::new(ArcSwap::from_pointee(Tuning {
        revision: 0,
        chunked_prefill_size: None,
        max_batch: NonZeroUsize::new(proto::config::DEFAULT_MAX_BATCH).unwrap(),
        max_num_batched_tokens: None,
        max_queued_requests: None,
        queue_depths: Default::default(),
        prefix_caching: true,
        host_tier: true,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        synthetic_acceptance_length: None,
        report_period: proto::config::DEFAULT_REPORT_PERIOD,
        watchdog_timeout: proto::config::DEFAULT_WATCHDOG_TIMEOUT,
        compare_timeout: proto::config::DEFAULT_COMPARE_TIMEOUT,
        nvtx_ranges: false,
        kv_events: Default::default(),
        roles: Vec::new(),
    }))
}

impl Scheduler {
    /// Replaces the step values the scheduler's engine shares by `change` of them, as the
    /// engine's `Tune` does, and takes them as the next step begins.
    pub(crate) fn tune(&mut self, change: impl FnOnce(&mut Tuning)) {
        let mut tuning = Tuning::clone(&self.tuning.load());
        change(&mut tuning);
        self.tuning.store(Arc::new(tuning));
        self.budget = super::Budget::of(&self.tuning.load(), &self.ceilings, &self.config);
    }
}

/// A kernel generation's graphs bound the steps from the next one on (`Message::SetGraphs`): the
/// group they name forms its batches within their shapes, a step's tokens and calls stay within
/// them, a verification carries their drafts at most, and a request may ask for their candidates
/// at a position; the serving graphs' limits, given back, bound the steps again.
#[test]
fn a_generation_s_graphs_bound_the_steps_from_the_next_one_on() {
    use proto::request::SamplingParams;

    let mut bench = bench(1024);
    let scheduler = &mut bench.scheduler;
    scheduler.groups[0].capacity.kv_cache = Some(proto::config::KvCacheConfig::default());
    scheduler.limits.features = proto::request::GenerationFeatures::all();
    scheduler.groups[0].capacity.buckets = vec![
        Shape::decode(8, Variant::DecodeGreedy),
        Shape::prefill(8, 512, Variant::PrefillGreedy),
    ];
    let serving = Arc::new(super::GraphLimits {
        config: SchedulerConfig::default(),
        buckets: vec![(A, scheduler.groups[0].capacity.buckets.clone())],
        phases: Vec::new(),
        max_logprobs: 0,
    });
    let asking = |logprobs: u32| {
        let sampling = SamplingParams {
            return_logprobs: true,
            n_logprobs: logprobs,
            ..SamplingParams::default()
        };
        proto::request::Request::from(GenerationRequest {
            trace: 0,
            request_id: proto::request::RequestId(7),
            prompt_token_ids: vec![1; 4],
            multimodal_inputs: Default::default(),
            sampling,
            max_und_tokens: 4,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            include_stop_token: false,
            generated_token_ids: Vec::new(),
            adapter: 0,
            hand_off: false,
            lend: None,
        })
    };
    let take = |scheduler: &mut Scheduler, limits: &Arc<super::GraphLimits>| {
        let (reply, replied) = crossbeam_channel::bounded(1);
        scheduler.apply(Message::SetGraphs(Arc::clone(limits), reply));
        replied.recv_timeout(Duration::from_secs(1)).unwrap();
    };
    assert!(scheduler.validate(&asking(3)).is_err());

    // The generation's graphs capture decodes of four rows and prefills of 128 tokens, with
    // verifications of two drafts and three candidates a position.
    let next = Arc::new(super::GraphLimits {
        config: SchedulerConfig {
            max_num_batched_tokens: 1024,
            draft_tokens: 2,
            ..SchedulerConfig::default()
        },
        buckets: vec![(
            A,
            vec![
                Shape::decode(4, Variant::DecodeGreedy),
                Shape::prefill(4, 128, Variant::PrefillGreedy),
            ],
        )],
        phases: Vec::new(),
        max_logprobs: 3,
    });
    take(scheduler, &next);
    assert_eq!(scheduler.validate(&asking(3)), Ok(()));
    assert!(scheduler.validate(&asking(4)).is_err());
    assert_eq!(scheduler.ceilings.tokens, 128);
    assert_eq!(scheduler.ceilings.rows, 4);
    assert_eq!(scheduler.config.max_num_batched_tokens, 128);
    assert_eq!(scheduler.config.draft_tokens, 2);
    assert_eq!(scheduler.groups[0].capacity.buckets, next.buckets[0].1);

    // The serving graphs' limits, given back, bound the steps as before.
    take(scheduler, &serving);
    assert!(scheduler.validate(&asking(3)).is_err());
    assert_eq!(scheduler.ceilings.tokens, 512);
    assert_eq!(scheduler.ceilings.rows, 8);
    assert_eq!(scheduler.config.draft_tokens, 0);
}
