//! A scheduler's executor over workers with no device: a worker group of one or two ranks, each
//! a worker over a runner that holds no model. The tests read what reaches the
//! scheduler's channel.

use std::sync::{Arc, Mutex};

use crate::batch::{
    Batch, BatchId, BatchOutput, CallOutput, CallStatus, Columns, Phase, Shape, Variant,
};
use crate::executor::Route;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::buckets::Bucket;
use crate::runner::graphs::Graphs;
use crate::runner::{Forward, Runner};
use crate::worker::Sequences;
use crate::worker::completion::Outcome;
use crossbeam_channel::{Receiver, Sender};
use ops::Generation;
use proto::request::RequestId;
use service::lock;
use storage::Key;
use storage::runtime::{self, Completion, GroupId, OwnerId, Request, Tier};
use tensor::Tensor;

use super::{Shadow, WorkerExecutor};
use crate::executor::Executor as _;
use crate::scheduler::Message;
use testkit::LONG;

pub(crate) const GROUP: GroupId = GroupId(3);

/// The regions every rank maps: the KV units on the device and the host pool of their copies.
const DEVICE: u8 = 0;
const HOST: u8 = 1;

/// The batch the runners fail.
const FAILING: u64 = 13;

/// A decode runner that holds no device and no model: every call of a batch samples the batch's
/// number, a copy is issued with the batch, and the batches it prepares paired forwards for are
/// recorded.
struct Answering {
    phases: Vec<Phase>,
    graphs: Graphs,
    shadowed: Arc<Mutex<Vec<u64>>>,
}

impl Forward for Answering {
    fn buckets(&self) -> Vec<Bucket> {
        Vec::new()
    }

    fn forward(&self, _: &Bucket) -> crate::worker::Result<Vec<Tensor>> {
        Err(crate::worker::Error::Unsupported(
            "the runner holds no model".to_owned(),
        ))
    }

    fn idle_inputs(&self, _: &Bucket, _: &mut [u8]) -> crate::worker::Result<()> {
        Ok(())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

impl Runner for Answering {
    fn phases(&self) -> &[Phase] {
        &self.phases
    }

    fn capturer(&self, graphs: Graphs) -> crate::worker::Result<Box<dyn Forward>> {
        Ok(Box::new(crate::runner::HostForward::new(graphs)))
    }

    fn prepare_inputs(&mut self, batch: &Batch, _: &BlockTables) -> crate::worker::Result<Bucket> {
        Ok(Bucket::decode(batch.shape.rows, batch.shape.variant))
    }

    fn prepare_shadow_inputs(
        &mut self,
        batch: &Batch,
        tables: &BlockTables,
    ) -> crate::worker::Result<Bucket> {
        lock(&self.shadowed).push(batch.id.0);
        self.prepare_inputs(batch, tables)
    }

    fn execute(&mut self, batch: &Batch, _: &BlockTables) -> crate::worker::Result<BatchOutput> {
        if batch.id.0 == FAILING {
            return Err(crate::worker::Error::Invalid(
                "the runner fails this batch".to_owned(),
            ));
        }

        Ok(BatchOutput {
            calls: batch.requests.iter().map(|_| sampled(batch.id.0)).collect(),
            ..BatchOutput::default()
        })
    }

    fn copy(&mut self, _: &[BlockCopy]) -> crate::worker::Result<()> {
        Ok(())
    }

    /// The test's runner holds the device and host regions every copy of these tests names.
    fn maps(&mut self, region: u8) -> bool {
        [DEVICE, HOST].contains(&region)
    }
}

/// A call's output that sampled `token`.
fn sampled(token: u64) -> CallOutput {
    CallOutput {
        status: CallStatus::Ok,
        error_code: None,
        error_message: None,
        tokens: vec![token as u32],
        step_index: 1,
        cache_computed_len: 1,
        cache_visible_len: 1,
        token_logprobs: Vec::new(),
        prompt_logprobs: Vec::new(),
        media_output: None,
        draft_token_ids: Vec::new(),
    }
}

/// A node's `ranks` ranks of `GROUP` serving generation 1, each a worker over one
/// [`Answering`] that records the batches it prepares paired forwards for into `shadowed`, and
/// the route over them.
pub(crate) fn node(
    ranks: usize,
    shadowed: &Arc<Mutex<Vec<u64>>>,
) -> (Arc<Route>, Vec<Arc<crate::worker::Worker>>) {
    let kernels = Arc::new(Generation::new(1, []));
    let sequences = Sequences::new(GROUP, ranks, Vec::new());

    let workers: Vec<Arc<crate::worker::Worker>> = (0..ranks)
        .map(|rank| {
            let runner = Answering {
                phases: vec![Phase::Decode],
                graphs: Graphs::meta(Arc::clone(&kernels), 0),
                shadowed: Arc::clone(shadowed),
            };
            let worker = crate::worker::Worker::spawn(
                None,
                vec![Box::new(runner)],
                Arc::clone(&kernels),
                // One unit of each region is 64 bytes: the tests move no bytes.
                BlockTables::new(GROUP, &[DEVICE, HOST], &[(DEVICE, 64), (HOST, 64)]),
                Arc::clone(&sequences),
                rank,
            );
            Arc::new(worker.unwrap())
        })
        .collect();

    let route = Route::new(vec![sequences], workers.clone(), Vec::new()).unwrap();
    (Arc::new(route), workers)
}

/// The executor of a scheduler of `GROUP` over `route`, with the shadow pass `shadow`,
/// and the scheduler's channel.
fn groups(route: &Arc<Route>, shadow: &Arc<Mutex<Shadow>>) -> (WorkerExecutor, Receiver<Message>) {
    let (inbox, messages) = crossbeam_channel::unbounded();
    let groups =
        WorkerExecutor::new(Arc::clone(route), &[GROUP], inbox, Arc::clone(shadow)).unwrap();
    (groups, messages)
}

/// The executor of a scheduler of `GROUP` over `ranks` new ranks, with no shadow pass
/// open, and the scheduler's channel.
fn served(ranks: usize) -> (WorkerExecutor, Receiver<Message>) {
    let (route, _) = node(ranks, &Arc::default());
    groups(&route, &Arc::default())
}

/// Everything the scheduler is ever sent: the executor is dropped, so its waiters have sent all
/// they will.
fn sent(groups: WorkerExecutor, messages: &Receiver<Message>) -> Vec<Message> {
    drop(groups);
    messages.try_iter().collect()
}

/// Batch `id` of `GROUP`: a greedy decode of `rows` requests.
fn decode(id: u64, rows: u32) -> Batch {
    Batch {
        id: BatchId(id),
        group: GROUP,
        phase: Phase::Decode,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: Shape::decode(rows, Variant::DecodeGreedy),
        requests: (0..u64::from(rows))
            .map(|row| RequestId(100 * id + row))
            .collect(),
        columns: Columns::default(),
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// Host batch `id` of `phase`, launched as `variant`, on `group`: one row of one request, which
/// names no rank and reads one result and writes another.
pub(crate) fn host_batch(id: u64, group: GroupId, phase: Phase, variant: Variant) -> Batch {
    let row = crate::batch::MediaRow {
        request: RequestId(9),
        slot: 1,
        conditioning: Vec::new(),
        media: Vec::new(),
        step: 0,
        steps: 1,
        seed: 0,
        width: 16,
        height: 16,
        frames: 17,
        rounds: 0,
        unit: 0,
        rank: None,
        image: None,
        results: vec![0, 1],
        reads: 1,
    };
    Batch {
        id: BatchId(id),
        group,
        phase,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: Shape {
            rows: 1,
            tokens: 0,
            variant,
            extent: crate::batch::Extent::None,
        },
        requests: vec![row.request],
        columns: Columns::default(),
        media: vec![row],
        new_slots: Vec::new(),
    }
}

/// Holds `rank` inside a control operation until the returned sender is dropped; returns once
/// the operation is in flight.
fn hold(rank: &crate::worker::Worker) -> Sender<()> {
    let (entered, in_flight) = crossbeam_channel::bounded(1);
    let (release, released) = crossbeam_channel::bounded::<()>(1);

    rank.control(
        "holding the rank",
        Box::new(move |_| {
            entered.send(()).unwrap();
            let _ = released.recv();
            Ok(())
        }),
    )
    .unwrap();
    in_flight.recv_timeout(LONG).unwrap();
    release
}

#[test]
fn every_batch_s_output_reaches_the_scheduler_once_as_output() {
    let (groups, messages) = served(2);

    let mut stray = decode(4, 1);
    stray.group = GroupId(9);
    let mut unserved = decode(2, 1);
    unserved.phase = Phase::Prefill;

    groups.submit(decode(1, 2));
    groups.submit(unserved);
    groups.submit(decode(FAILING, 1));
    groups.submit(stray);

    let sent = sent(groups, &messages);
    let [
        Message::Output(served, Ok(output)),
        Message::Output(refused, Err(refusal)),
        Message::Output(failed, Err(failure)),
        Message::Output(other, Err(foreign)),
    ] = sent.as_slice()
    else {
        panic!("one output per batch, and nothing else: {sent:?}");
    };

    // Both ranks of the group ran batch 1; the scheduler is given its output once, one call per
    // request.
    assert_eq!(served.id, BatchId(1));
    let tokens: Vec<&[u32]> = output.calls.iter().map(|call| &call.tokens[..]).collect();
    assert_eq!(tokens, [[1], [1]]);

    // A batch no rank serves, one the runners fail, and one of a group that is not the
    // scheduler's arrive as their errors.
    assert_eq!(refused.id, BatchId(2));
    assert!(refusal.to_string().contains("Prefill"), "{refusal}");
    assert_eq!(failed.id, BatchId(FAILING));
    assert!(
        failure.to_string().contains("the runner fails this batch"),
        "{failure}"
    );
    assert_eq!(other.id, BatchId(4));
    assert!(
        foreign.to_string().contains("not one of this scheduler's"),
        "{foreign}"
    );
}

#[test]
fn a_stamped_request_returns_as_submit_with_the_last_batch_its_group_launched() {
    let (groups, messages) = served(2);
    let release = |after| Request::Release {
        owner: OwnerId(1),
        class: 0,
        at: 0,
        count: 1,
        after,
    };

    // Owner 1 in slot 2 with blocks at positions 0 and 1: maps send nothing.
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: GROUP,
    };
    groups.map(
        GROUP,
        2,
        runtime::RequestId(1),
        attach,
        Completion::Attached { slot: 2 },
    );
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: 0,
        at: 0,
        count: 2,
        units: 0,
    };
    let allocated = Completion::Allocated {
        runs: vec![(DEVICE, 7, 1), (DEVICE, 9, 1)],
        evicted: Vec::new(),
    };
    groups.map(GROUP, 2, runtime::RequestId(2), allocate, allocated);

    groups.submit(decode(1, 1));
    groups.stamp(GROUP, vec![release(Vec::new())]);

    let sent = sent(groups, &messages);
    let [Message::Output(batch, Ok(_)), Message::Submit(stamped)] = sent.as_slice() else {
        panic!("the batch's output, then the stamped release: {sent:?}");
    };
    assert_eq!(batch.id, BatchId(1));

    // The group's first batch is number 1, and it was launched when the release was applied.
    assert_eq!(stamped, &[release(vec![(GROUP, 1)])]);
}

#[test]
fn a_copy_s_finish_arrives_just_before_the_output_of_the_batch_that_ran_it() {
    let (groups, messages) = served(2);

    // A store of a device block to the host, which every rank runs.
    let store = Request::Store {
        group: GROUP,
        keys: vec![Key([4; 32])],
        tier: Tier::Host,
    };
    let copy = Completion::Copy {
        group: GROUP,
        keys: vec![Key([4; 32])],
        from: vec![(DEVICE, 7, 1)],
        to: vec![(HOST, 3, 1)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    groups.map(GROUP, 0, runtime::RequestId(10), store, copy);
    groups.submit(decode(1, 1));
    groups.submit(decode(2, 1));

    let sent = sent(groups, &messages);
    let [
        Message::Submit(finished),
        Message::Output(first, Ok(_)),
        Message::Output(second, Ok(_)),
    ] = sent.as_slice()
    else {
        panic!("the copy's `Finish`, the first batch's output, the second's output: {sent:?}");
    };
    assert_eq!((first.id, second.id), (BatchId(1), BatchId(2)));

    // The copy ran with batch 1 on both ranks, and batch 1 is the group's stamp of its `Finish`.
    let finish = Request::Finish {
        copy: runtime::RequestId(10),
        ok: true,
        after: vec![(GROUP, 1)],
    };
    assert_eq!(finished, &[finish]);
}

#[test]
fn a_failed_map_reaches_the_scheduler_as_failed_in_its_place_among_the_groups_jobs() {
    let (groups, messages) = served(2);

    // An `Attached` mapped under another slot than the one it states, which the tables refuse.
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: GROUP,
    };
    groups.submit(decode(1, 1));
    groups.map(
        GROUP,
        2,
        runtime::RequestId(7),
        attach,
        Completion::Attached { slot: 5 },
    );
    groups.submit(decode(2, 1));

    let sent = sent(groups, &messages);
    let [
        Message::Output(before, Ok(_)),
        Message::Failed(failure),
        Message::Output(after, _),
    ] = sent.as_slice()
    else {
        panic!("the batch before the map, the map's failure, the batch after it: {sent:?}");
    };
    assert_eq!((before.id, after.id), (BatchId(1), BatchId(2)));

    let failure = failure.to_string();
    assert!(
        failure.contains("request 7") && failure.contains("slot 5"),
        "{failure}"
    );
}

#[test]
fn a_shadow_pass_shadows_each_shape_once_and_answers_once_its_shadow_batches_ended() {
    let shadowed = Arc::default();
    let shadow = Arc::default();
    let (route, ranks) = node(1, &shadowed);
    let (groups, messages) = groups(&route, &shadow);

    // Generation 2 is prepared beside the serving one.
    let prepared = ranks[0].prepare(Arc::new(Generation::new(2, []))).unwrap();
    assert!(matches!(
        prepared.wait_timeout(LONG),
        Some(Outcome::Success(_))
    ));

    // The rank is held so that no shadow batch ends before the last batch is handed: the pass
    // covers all three, and batch 2 has the shape batch 1 has.
    let (reply, replied) = crossbeam_channel::bounded(1);
    lock(&shadow).open(2, reply);
    let release = hold(&ranks[0]);
    groups.submit(decode(1, 1));
    groups.submit(decode(2, 1));
    groups.submit(decode(3, 2));
    drop(release);

    assert!(matches!(replied.recv_timeout(LONG), Ok(Ok(()))));
    assert_eq!(*lock(&shadowed), [1, 3]);

    // The pass closed with its answer: a batch of a new shape is not shadowed.
    groups.submit(decode(4, 4));

    // A pass of a generation the rank does not prepare fails with the rank's refusal.
    let (reply, replied) = crossbeam_channel::bounded(1);
    lock(&shadow).open(9, reply);
    groups.submit(decode(5, 1));
    let failure = replied.recv_timeout(LONG).unwrap().unwrap_err().to_string();
    assert!(failure.contains("generation 9"), "{failure}");

    // Every batch's output arrived once, shadowed or not.
    let outputs: Vec<u64> = sent(groups, &messages)
        .iter()
        .map(|message| match message {
            Message::Output(batch, Ok(_)) => batch.id.0,
            other => panic!("only outputs are sent: {other:?}"),
        })
        .collect();
    assert_eq!(outputs, [1, 2, 3, 4, 5]);
    assert_eq!(*lock(&shadowed), [1, 3]);
}

/// A peer's process none of whose ranks serves a video encode round takes the round's number
/// and passes it: ranks of other processes decoded its units. Host work the scheduler node
/// handed it to run once is refused, as nothing here runs it.
#[test]
fn a_peer_passes_a_video_encode_its_ranks_do_not_serve_and_refuses_host_work_it_must_run() {
    let (route, _) = node(1, &Arc::default());

    let encode = host_batch(1, GROUP, Phase::VideoEncode, Variant::VideoEncode);
    let submission = route.run(1, encode, false, false).unwrap();
    let Some(Outcome::Success((output, _))) = submission.wait_timeout(LONG) else {
        panic!("the encode was not passed");
    };
    assert!(output.calls.is_empty());
    assert_eq!(route.completed(GROUP).unwrap(), 1);

    let track = host_batch(2, GROUP, Phase::AudioEncode, Variant::AudioEncode);
    assert!(route.run(2, track, false, false).is_err());
}
