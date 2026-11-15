//! Workers without a device: runners over [`Graphs::meta`], which run the profile run, the
//! capture and the whole generation state machine with every device step left out, driven
//! through the worker's queue, and worker groups of such workers driven through
//! [`Route`] with the buffer pool's requests and completions.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::batch::{
    Batch, BatchId, BatchOutput, CallOutput, CallStatus, Columns, MediaRow, Phase,
    Shape as BatchShape, Variant,
};
use crossbeam_channel::{Receiver, Sender};
use nn::functional::{embedding, rms_norm};
use ops::{Generation, KernelEntry};
use proto::request::RequestId;
use storage::QueuePair;
use storage::runtime::{self, Completion, GroupId, OwnerId, Request, Run, Tier};
use tensor::{DType, Tensor};

use super::watchdog::Watchdog;
use super::{Sequences, Worker};
use crate::executor::Route;
use crate::executor::peers::{Liveness, Remote, Upstream, accept, register};
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::Runner;
use crate::runner::buckets::Bucket;
use crate::runner::graphs::Graphs;
use crate::worker::completion::{Outcome, Submission, Waker};
use crate::worker::{Error, Result};
use service::lock;
use testkit::LONG;
use testkit::serving::unlaunched_kernel;

const VOCABULARY: usize = 32;
const HIDDEN: usize = 16;

/// The bytes of a runner's input tensors: one column of token ids.
const INPUTS: u64 = 256;

/// The batch a runner refuses.
const REFUSED: u64 = 13;

/// The worker group of the tests' workers, and the regions its ranks map: the KV units on the
/// device and the host pool of their copies, and each of two ranks' product buffer.
const GROUP: GroupId = GroupId(3);
const DEVICE: u8 = 0;
const HOST: u8 = 1;
const PRODUCTS: [u8; 2] = [2, 3];

/// The bytes of a unit of every region, summed over the ranks, and the copy budget of the tests'
/// workers: three units.
const UNIT: u64 = 1024;
const UNIT_BYTES: [(u8, u64); 4] = [
    (DEVICE, UNIT),
    (HOST, UNIT),
    (PRODUCTS[0], UNIT),
    (PRODUCTS[1], UNIT),
];
const BUDGET: u64 = 3 * UNIT;

/// The phase the `index`-th runner of a test worker serves, and one no runner serves.
const PHASES: [Phase; 2] = [Phase::Decode, Phase::Prefill];
const UNSERVED: Phase = Phase::Denoise;

/// What the runners of one test did, in the order they did it.
type Log = Arc<Mutex<Vec<String>>>;

/// A runner of a model of two operators over meta weights. It states what it is asked to do in
/// the test's log, with the device block table entries of class 0 each slot its batch reads
/// holds, and returns one output per request of a batch.
struct LoggingRunner {
    name: &'static str,
    phases: Vec<Phase>,
    buckets: Vec<Bucket>,
    graphs: Graphs,
    table: Tensor,
    gain: Tensor,
    log: Log,
    /// The batches whose execution the test holds, with the channels it holds them by: the
    /// runner sends the batch's id once it executes it and waits for the other to close.
    held: Option<(Vec<u64>, Sender<u64>, Receiver<()>)>,
    /// What the worker last stated with `Runner::set_warming`.
    warming: AtomicBool,
    /// The regions whose memory the runner maps: every region of the tests' ranks, unless a
    /// test states fewer.
    regions: Vec<u8>,
    /// Whether the runner issued copies that no wait has covered yet.
    copying: bool,
    /// Where a runner that serves its batches as steps in flight hands the test each batch it
    /// launched with what wakes the worker for its landing, which the test calls when the
    /// batch's device work ends; `None` for a runner that serves a batch in one turn.
    in_flight: Option<Sender<(u64, Waker)>>,
    /// The batches launched and not landed.
    launched: Vec<u64>,
}

impl LoggingRunner {
    fn new(name: &'static str, phase: Phase, kernels: Arc<Generation>, log: &Log) -> Self {
        let weight = |address: u64, dims: &[usize]| {
            ops::retained(address, dims, DType::BF16, &ops::meta()).unwrap()
        };

        Self {
            name,
            phases: vec![phase],
            buckets: vec![
                Bucket::decode(8, Variant::DecodeGreedy),
                Bucket::decode(4, Variant::DecodeGreedy),
                Bucket::decode(2, Variant::DecodeGreedy),
            ],
            graphs: Graphs::meta(kernels, INPUTS),
            table: weight(0x1_0000, &[VOCABULARY, HIDDEN]),
            gain: weight(0x2_0000, &[HIDDEN]),
            log: Arc::clone(log),
            held: None,
            warming: AtomicBool::new(false),
            regions: vec![DEVICE, HOST, PRODUCTS[0], PRODUCTS[1]],
            copying: false,
            in_flight: None,
            launched: Vec::new(),
        }
    }

    fn state(&self, what: String) {
        lock(&self.log).push(format!("{}: {what}", self.name));
    }
}

impl Runner for LoggingRunner {
    fn phases(&self) -> &[Phase] {
        &self.phases
    }

    fn buckets(&self) -> Vec<Bucket> {
        self.buckets.clone()
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let ids = self.graphs.input(0, bucket.rows as usize, DType::U32)?;
        let hidden = embedding(&ids, &self.table, 0)?;
        Ok(vec![rms_norm(
            &hidden,
            &self.gain,
            1e-6,
            false,
            hidden.dtype(),
        )?])
    }

    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        // A capture fills the idle inputs of its bucket once, and is a warm-up's.
        if self.warming.load(Ordering::Acquire) {
            self.state(format!("capture {bucket}"));
        } else {
            self.state(format!("capture {bucket} outside a warm-up"));
        }
        columns.fill(0);
        Ok(())
    }

    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        let _ = tables;
        let bucket = self
            .graphs
            .select(|bucket| bucket.holds(&batch.shape))
            .ok_or_else(|| Error::Invalid(format!("no bucket holds {}", batch.shape)))?;

        self.graphs
            .prepare_inputs(&*self, &bucket, |_columns| Ok(()))?;
        Ok(bucket)
    }

    fn prepare_shadow_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        // The runner's forward writes no state, so a paired forward's inputs are a step's.
        self.prepare_inputs(batch, tables)
    }

    /// A runner with steps in flight launches every batch with rows and returns: the test ends
    /// its device work.
    fn start(&mut self, batch: &Batch, tables: &BlockTables, resume: Waker) -> Result<bool> {
        let _ = tables;
        let Some(in_flight) = &self.in_flight else {
            return Ok(false);
        };

        self.state(format!("launch {}", batch.id.0));
        self.launched.push(batch.id.0);
        in_flight.send((batch.id.0, resume)).unwrap();
        Ok(true)
    }

    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        // The landing of a batch `start` launched, once its device work ended.
        if let Some(at) = self.launched.iter().position(|id| *id == batch.id.0) {
            self.launched.remove(at);
            self.state(format!("land {}", batch.id.0));
            if batch.id.0 == REFUSED {
                return Err(Error::Invalid("the runner refuses this batch".into()));
            }
            return Ok(BatchOutput {
                calls: batch
                    .requests
                    .iter()
                    .map(|_| completed(batch.id.0))
                    .collect(),
                ..BatchOutput::default()
            });
        }

        if let Some((held, entered, released)) = &self.held
            && held.contains(&batch.id.0)
        {
            entered.send(batch.id.0).unwrap();
            let _ = released.recv();
        }
        if batch.id.0 == REFUSED {
            return Err(Error::Invalid("the runner refuses this batch".into()));
        }

        let warming = if self.warming.load(Ordering::Acquire) {
            " during a warm-up"
        } else {
            ""
        };
        let read: String = batch
            .columns
            .slots
            .iter()
            .map(|&slot| {
                let entries: Vec<(u32, Run)> = tables.entries(slot as u32, 0).collect();
                format!(" reading slot {slot} at {entries:?}")
            })
            .collect();
        self.state(format!(
            "execute {} at generation {}{read}{warming}",
            batch.id.0,
            self.graphs.generation()
        ));

        Ok(BatchOutput {
            calls: batch
                .requests
                .iter()
                .map(|_| completed(batch.id.0))
                .collect(),
            ..BatchOutput::default()
        })
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }

    fn set_warming(&self, warming: bool) {
        self.warming.store(warming, Ordering::Release);
    }

    fn weights_changed(&mut self) -> Result<()> {
        self.state("weights changed".to_owned());
        Ok(())
    }

    fn maps(&mut self, region: u8) -> bool {
        self.regions.contains(&region)
    }

    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        let copies: Vec<(u64, Vec<(Run, Run)>)> = copies
            .iter()
            .map(|copy| (copy.request.0, copy.runs.clone()))
            .collect();
        self.state(format!("copy {copies:?}"));
        self.copying = true;
        Ok(())
    }

    /// States the wait that covers the copies issued since the last one.
    fn wait_copies(&mut self) -> Result<()> {
        if std::mem::take(&mut self.copying) {
            self.state("wait for copies".to_owned());
        }
        Ok(())
    }
}

/// A call's output that sampled `token`.
fn completed(token: u64) -> CallOutput {
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

/// Generation 1, holding a kernel for every key a profile run of the test runner states.
fn serving() -> Arc<Generation> {
    let profiled = LoggingRunner::new(
        "profiled",
        Phase::Decode,
        Arc::new(Generation::new(0, [])),
        &Log::default(),
    );
    profiled.graphs.profile_run(&profiled).unwrap();
    let kernel = unlaunched_kernel();

    Arc::new(Generation::new(
        1,
        profiled
            .graphs
            .keys()
            .into_iter()
            .map(|(operator, key)| KernelEntry::keyed(operator, key, Arc::clone(&kernel))),
    ))
}

/// The generation `number` after `serving` in which every kernel of `operator` is another one.
fn replacing(serving: &Generation, number: u64, operator: &'static str) -> Arc<Generation> {
    let kernel = unlaunched_kernel();
    let keys: Vec<String> = serving.keys(operator).map(str::to_owned).collect();

    Arc::new(
        serving.replace(
            number,
            keys.into_iter()
                .map(|key| KernelEntry::keyed(operator, key, Arc::clone(&kernel))),
        ),
    )
}

/// The worker with no device of rank `rank` of `sequences`' group over `runners`, mapping
/// `regions`, with the tests' copy budget.
fn spawn(
    runners: Vec<Box<dyn Runner>>,
    kernels: &Arc<Generation>,
    sequences: &Arc<Sequences>,
    rank: usize,
    regions: &[u8],
) -> Worker {
    let tables = BlockTables::new(sequences.group(), regions, &UNIT_BYTES);
    Worker::spawn(
        None,
        runners,
        Arc::clone(kernels),
        tables,
        Arc::clone(sequences),
        rank,
        BUDGET,
    )
    .unwrap()
}

/// A worker with no device, the one rank of group `GROUP`, over one runner per name, the
/// `index`-th serving `PHASES[index]`, with the generation they serve.
fn worker(names: &[&'static str], log: &Log) -> (Worker, Arc<Generation>) {
    let kernels = serving();
    let runners = names
        .iter()
        .zip(PHASES)
        .map(|(name, phase)| {
            Box::new(LoggingRunner::new(name, phase, Arc::clone(&kernels), log)) as Box<dyn Runner>
        })
        .collect();

    let sequences = Sequences::new(GROUP, 1, Vec::new());
    let worker = spawn(runners, &kernels, &sequences, 0, &[DEVICE, HOST]);
    (worker, kernels)
}

/// A decode batch `id` of group `GROUP` and `phase`, of `rows` rows and one request.
fn batch(id: u64, phase: Phase, rows: u32) -> Batch {
    Batch {
        id: BatchId(id),
        group: GROUP,
        phase,
        generation: 1,
        shape: BatchShape::decode(rows, Variant::DecodeGreedy),
        requests: vec![RequestId(id)],
        columns: Columns::default(),
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// As [`batch`], of phase `Decode`, with a row reading `slot`.
fn reading(id: u64, slot: i32) -> Batch {
    let mut batch = batch(id, Phase::Decode, 1);
    batch.columns.slots = vec![slot];
    batch
}

/// Holds `worker` inside a control operation until the returned sender is dropped, and
/// returns once the operation is in flight, with its submission.
fn hold(worker: &Worker) -> (Sender<()>, Submission) {
    let (entered, in_flight) = crossbeam_channel::bounded(1);
    let (release, released) = crossbeam_channel::bounded::<()>(1);

    let submission = worker
        .control(
            "holding the worker for a test",
            Box::new(move |_| {
                entered.send(()).unwrap();
                let _ = released.recv();
                Ok(())
            }),
        )
        .unwrap();
    in_flight
        .recv_timeout(LONG)
        .expect("the worker takes the operation");
    (release, submission)
}

/// What `submission`, which succeeds, returned.
fn returned_by(submission: &Submission) -> (BatchOutput, Vec<Request>) {
    match submission.wait_timeout(LONG) {
        Some(Outcome::Success(returned)) => returned,
        Some(Outcome::Failed(error)) => panic!("the job failed: {error}"),
        Some(Outcome::Cancelled) => panic!("the job was cancelled"),
        None => panic!("the job did not complete"),
    }
}

/// The output of `submission`, which succeeds.
fn output_of(submission: &Submission) -> BatchOutput {
    returned_by(submission).0
}

/// The error of `submission`, which fails.
fn failure_of(submission: &Submission) -> String {
    match submission.wait_timeout(LONG) {
        Some(Outcome::Failed(error)) => error.to_string(),
        other => panic!("the job did not fail: {other:?}"),
    }
}

/// The generation each runner of `worker` serves, read between two jobs.
fn generations(worker: &Worker) -> Vec<u64> {
    worker
        .call("reading the generations the runners serve", |runners| {
            Ok(runners
                .iter()
                .map(|runner| runner.graphs().generation())
                .collect())
        })
        .unwrap()
}

fn entries(log: &Log) -> Vec<String> {
    lock(log).clone()
}

#[test]
fn jobs_run_in_submission_order_and_a_control_operation_runs_between_two_batches() {
    let log = Log::default();
    let (worker, _) = worker(&["a"], &log);

    let (release, held) = hold(&worker);
    let before = Instant::now();
    let first = worker.submit(1, batch(1, Phase::Decode, 2)).unwrap();
    let submitted = Instant::now();
    let control = worker
        .control(
            "recomputing what the runners derive from their weights",
            Box::new(|runners| {
                runners
                    .iter_mut()
                    .try_for_each(|runner| runner.weights_changed())
            }),
        )
        .unwrap();
    let second = worker.submit(2, batch(2, Phase::Decode, 2)).unwrap();

    // Nothing behind the job in flight has run.
    assert!(first.outcome().is_none());
    assert!(entries(&log).is_empty());

    let released = Instant::now();
    drop(release);
    output_of(&held);
    let output = output_of(&first);
    let completed = Instant::now();
    output_of(&control);
    output_of(&second);

    assert_eq!(
        entries(&log),
        [
            "a: execute 1 at generation 1",
            "a: weights changed",
            "a: execute 2 at generation 1",
        ]
    );

    // The batch's output is the runner's, with the time the batch waited in the queue: at
    // least the time it was held behind the operation, and no more than the time from before
    // its submission to its completion.
    assert_eq!(output.calls.len(), 1);
    assert_eq!(output.calls[0].tokens, [1]);
    let queued = u128::from(output.timing.queued_us);
    assert!(queued >= (released - submitted).as_micros(), "{queued}");
    assert!(queued <= (completed - before).as_micros(), "{queued}");

    assert_eq!(worker.counters().0, 2);
    worker.close().unwrap();
}

/// A worker of one rank over one runner `a` whose steps are in flight across two turns, the
/// group's sequences, and the runner's launches with what lands each.
fn in_flight(log: &Log) -> (Worker, Arc<Sequences>, Receiver<(u64, Waker)>) {
    let kernels = serving();
    let (launches, launched) = crossbeam_channel::unbounded();
    let mut runner = LoggingRunner::new("a", Phase::Decode, Arc::clone(&kernels), log);
    runner.in_flight = Some(launches);

    let sequences = Sequences::new(GROUP, 1, Vec::new());
    let worker = spawn(
        vec![Box::new(runner)],
        &kernels,
        &sequences,
        0,
        &[DEVICE, HOST],
    );
    (worker, sequences, launched)
}

#[test]
fn a_batch_submitted_while_the_one_before_is_in_flight_runs_and_both_end_in_order() {
    let log = Log::default();
    let (worker, sequences, launched) = in_flight(&log);

    let first = worker.submit(1, batch(1, Phase::Decode, 2)).unwrap();
    let second = worker.submit(2, batch(2, Phase::Decode, 2)).unwrap();

    // The second batch is launched while the first is in flight, and neither has ended.
    let (one, land_first) = launched.recv_timeout(LONG).unwrap();
    let (two, land_second) = launched.recv_timeout(LONG).unwrap();
    assert_eq!((one, two), (1, 2));
    assert!(first.outcome().is_none());
    assert!(second.outcome().is_none());
    assert_eq!(sequences.completed(), 0);

    // The first step's end completes its batch with its output, and the group's `completed`
    // reaches it and not the batch still in flight.
    land_first();
    assert_eq!(output_of(&first).calls[0].tokens, [1]);
    assert_eq!(sequences.completed(), 1);
    assert!(second.outcome().is_none());

    land_second();
    assert_eq!(output_of(&second).calls[0].tokens, [2]);
    assert_eq!(sequences.completed(), 2);

    assert_eq!(
        entries(&log),
        ["a: launch 1", "a: launch 2", "a: land 1", "a: land 2"]
    );
    worker.close().unwrap();
}

#[test]
fn a_step_that_fails_as_it_lands_fails_its_batch_and_the_one_behind_it_ends() {
    let log = Log::default();
    let (worker, sequences, launched) = in_flight(&log);

    let refused = worker.submit(1, batch(REFUSED, Phase::Decode, 2)).unwrap();
    let behind = worker
        .submit(2, batch(REFUSED + 1, Phase::Decode, 2))
        .unwrap();
    let (_, land_refused) = launched.recv_timeout(LONG).unwrap();
    let (_, land_behind) = launched.recv_timeout(LONG).unwrap();

    // The failed batch ends: the device has passed it, so `completed` reaches it.
    land_refused();
    assert!(failure_of(&refused).contains("refuses"));
    assert_eq!(sequences.completed(), 1);

    land_behind();
    assert_eq!(output_of(&behind).calls[0].tokens, [REFUSED as u32 + 1]);
    assert_eq!(sequences.completed(), 2);
    worker.close().unwrap();
}

#[test]
fn a_waiter_on_a_submission_is_woken_by_its_completion() {
    let log = Log::default();
    let (worker, _) = worker(&["a"], &log);

    let (release, held) = hold(&worker);
    let submission = worker.submit(1, batch(1, Phase::Decode, 2)).unwrap();

    let (woken, woke) = crossbeam_channel::bounded(1);
    let waiter = std::thread::spawn(move || {
        woken.send(submission.wait()).unwrap();
    });

    // The job in flight has not ended, so the batch behind it has not run and its waiter
    // sleeps.
    assert!(held.outcome().is_none());
    assert!(woke.try_recv().is_err());
    assert!(entries(&log).is_empty());

    drop(release);
    let status = woke
        .recv_timeout(LONG)
        .expect("the completion wakes the waiter");
    assert!(matches!(status, Outcome::Success(_)));
    waiter.join().unwrap();
}

#[test]
fn a_batch_goes_to_the_runner_of_its_phase_and_one_no_runner_serves_is_refused() {
    let log = Log::default();
    let (worker, _) = worker(&["a", "b"], &log);
    assert_eq!(worker.phases(), PHASES);

    output_of(&worker.submit(1, batch(1, Phase::Prefill, 2)).unwrap());
    output_of(&worker.submit(2, batch(2, Phase::Decode, 2)).unwrap());
    assert_eq!(
        entries(&log),
        [
            "b: execute 1 at generation 1",
            "a: execute 2 at generation 1",
        ]
    );

    // The refusal is made where the batch is submitted and names both sides.
    let refused = worker.submit(3, batch(3, UNSERVED, 2)).unwrap_err();
    assert!(matches!(refused, Error::Invalid(_)), "{refused}");
    let refused = refused.to_string();
    assert!(
        refused.contains("Denoise") && refused.contains("[Decode, Prefill]"),
        "{refused}"
    );

    // An expert join takes no number and is refused for a phase nothing here serves, and a
    // batch of any other variant is never taken as one.
    let mut join = batch(4, UNSERVED, 1);
    join.shape = BatchShape::decode(1, Variant::ExpertJoin);
    assert!(worker.join(join).is_err());
    assert!(worker.join(batch(5, Phase::Decode, 1)).is_err());
}

#[test]
fn warm_up_captures_every_bucket_largest_first_and_batches_run_between_captures() {
    let log = Log::default();
    let (worker, _) = worker(&["a", "b"], &log);

    // The warm-up and a batch are queued behind the job in flight, in that order.
    let (release, held) = hold(&worker);
    let warm_up = worker.warm_up().unwrap();
    let served = worker.submit(1, batch(1, Phase::Decode, 2)).unwrap();
    drop(release);

    output_of(&held);
    output_of(&served);
    output_of(&warm_up);

    // One bucket per turn: the batch is executed after the first capture, not after the last,
    // and every runner's forwards are a warm-up's until the last capture.
    assert_eq!(
        entries(&log),
        [
            "a: capture decode_greedy/8 rows/8 tokens",
            "a: execute 1 at generation 1 during a warm-up",
            "a: capture decode_greedy/4 rows/4 tokens",
            "a: capture decode_greedy/2 rows/2 tokens",
            "b: capture decode_greedy/8 rows/8 tokens",
            "b: capture decode_greedy/4 rows/4 tokens",
            "b: capture decode_greedy/2 rows/2 tokens",
        ]
    );

    // Every bucket of every runner has its graph, and a second warm-up captures nothing.
    let captured = worker
        .call("reading the captured buckets", |runners| {
            Ok(runners
                .iter()
                .map(|runner| runner.graphs().buckets().len())
                .collect::<Vec<_>>())
        })
        .unwrap();
    assert_eq!(captured, [3, 3]);

    output_of(&worker.warm_up().unwrap());
    assert_eq!(entries(&log).len(), 7);

    // The warm-up ended: a batch after it is no warm-up's.
    output_of(&worker.submit(2, batch(2, Phase::Decode, 2)).unwrap());
    assert_eq!(
        entries(&log).last().map(String::as_str),
        Some("a: execute 2 at generation 1")
    );
}

#[test]
fn a_generation_is_prepared_switched_to_and_committed_on_every_runner_between_batches() {
    let log = Log::default();
    let (worker, kernels) = worker(&["a", "b"], &log);
    output_of(&worker.warm_up().unwrap());

    // The preparation leaves the serving generation serving.
    let next = replacing(&kernels, 2, "rms_norm");
    output_of(&worker.prepare(next).unwrap());
    assert_eq!(generations(&worker), [1, 1]);
    assert!(worker.uncompared().is_empty());
    lock(&log).clear();

    // The switch takes effect between the batch submitted before it and the one after it.
    let (release, held) = hold(&worker);
    let before = worker.submit(1, batch(1, Phase::Decode, 2)).unwrap();
    let switch = worker.switch(2).unwrap();
    let after = worker.submit(2, batch(2, Phase::Prefill, 2)).unwrap();
    drop(release);
    for submission in [&held, &before, &switch, &after] {
        output_of(submission);
    }

    assert_eq!(
        entries(&log),
        [
            "a: execute 1 at generation 1",
            "b: execute 2 at generation 2",
        ]
    );
    assert_eq!(generations(&worker), [2, 2]);

    // The committed generation serves, and nothing of it is left to roll back.
    output_of(&worker.commit(2).unwrap());
    assert_eq!(generations(&worker), [2, 2]);
    let refused = failure_of(&worker.rollback(2).unwrap());
    assert!(refused.contains("rolling back generation 2"), "{refused}");
    assert_eq!(generations(&worker), [2, 2]);
}

#[test]
fn a_generation_is_rolled_back_prepared_or_on_trial() {
    let log = Log::default();
    let (worker, kernels) = worker(&["a", "b"], &log);
    output_of(&worker.warm_up().unwrap());

    // Prepared and never switched to: its graphs go, and the next update starts from the
    // serving generation.
    output_of(&worker.prepare(replacing(&kernels, 2, "rms_norm")).unwrap());
    output_of(&worker.rollback(2).unwrap());
    assert_eq!(generations(&worker), [1, 1]);
    let refused = failure_of(&worker.switch(2).unwrap());
    assert!(refused.contains("was not prepared"), "{refused}");
    assert_eq!(generations(&worker), [1, 1]);

    // On trial: the generation it replaced serves again.
    output_of(&worker.prepare(replacing(&kernels, 3, "rms_norm")).unwrap());
    output_of(&worker.switch(3).unwrap());
    assert_eq!(generations(&worker), [3, 3]);
    output_of(&worker.rollback(3).unwrap());
    assert_eq!(generations(&worker), [1, 1]);

    // A generation the worker holds nothing of is rolled back by doing nothing.
    output_of(&worker.rollback(9).unwrap());
    assert_eq!(generations(&worker), [1, 1]);
}

#[test]
fn a_switch_before_the_preparation_ended_switches_no_runner() {
    let log = Log::default();
    let (worker, kernels) = worker(&["a", "b"], &log);
    output_of(&worker.warm_up().unwrap());

    // The preparation's first turn runs, and the switch is taken before its buckets' turns.
    let (release, held) = hold(&worker);
    let prepare = worker.prepare(replacing(&kernels, 2, "rms_norm")).unwrap();
    let switch = worker.switch(2).unwrap();
    drop(release);
    output_of(&held);

    let refused = failure_of(&switch);
    assert!(refused.contains("is not prepared"), "{refused}");
    output_of(&prepare);
    assert_eq!(generations(&worker), [1, 1]);

    // A generation that lacks a kernel the runners launch is refused, and is not being
    // prepared afterwards.
    output_of(&worker.rollback(2).unwrap());
    let refused = failure_of(&worker.prepare(Arc::new(Generation::new(4, []))).unwrap());
    assert!(refused.contains("preparing generation 4"), "{refused}");
    output_of(&worker.prepare(replacing(&kernels, 5, "rms_norm")).unwrap());
}

/// A preparation examines its buckets between batches, whatever the generation changes: one
/// that keeps every graph (a policy's, which replaces no kernel) is still a profile run of each
/// bucket, and a batch submitted after it began is executed before it ends, at the generation
/// that serves.
#[test]
fn a_preparation_that_keeps_every_graph_runs_between_batches() {
    let log = Log::default();
    let (worker, kernels) = worker(&["a", "b"], &log);
    output_of(&worker.warm_up().unwrap());
    lock(&log).clear();

    // The worker takes the preparation, then the batch, then an operation the test holds.
    let (release, held) = hold(&worker);
    let prepare = worker.prepare(Arc::new(kernels.replace(2, []))).unwrap();
    let served = worker.submit(1, batch(1, Phase::Decode, 2)).unwrap();
    let (entered, behind) = crossbeam_channel::bounded(1);
    let (release_behind, released_behind) = crossbeam_channel::bounded::<()>(1);
    let behind_batch = worker
        .control(
            "holding the worker behind the batch",
            Box::new(move |_| {
                entered.send(()).unwrap();
                let _ = released_behind.recv();
                Ok(())
            }),
        )
        .unwrap();
    drop(release);
    output_of(&held);
    behind
        .recv_timeout(LONG)
        .expect("the worker takes the operation");

    // The batch was served while the preparation is under way.
    assert!(matches!(served.outcome(), Some(Outcome::Success(_))));
    assert!(prepare.outcome().is_none());
    assert_eq!(entries(&log), ["a: execute 1 at generation 1"]);

    // The preparation ends with every graph kept: it captured nothing, and it is switched to.
    drop(release_behind);
    output_of(&behind_batch);
    output_of(&prepare);
    output_of(&worker.switch(2).unwrap());
    assert_eq!(generations(&worker), [2, 2]);
    assert_eq!(entries(&log), ["a: execute 1 at generation 1"]);
}

#[test]
fn a_failed_job_states_the_work_in_flight_and_the_jobs_behind_it_run() {
    let log = Log::default();
    let (worker, _) = worker(&["a"], &log);

    let (release, held) = hold(&worker);
    let refused = worker.submit(1, batch(REFUSED, Phase::Decode, 2)).unwrap();
    let broken = worker
        .control(
            "giving the cache's segments back",
            Box::new(|_| Err(Error::Resource("the storage refused"))),
        )
        .unwrap();
    let served = worker.submit(2, batch(2, Phase::Decode, 2)).unwrap();
    drop(release);
    output_of(&held);

    // The error carries the job it was taken in.
    let error = failure_of(&refused);
    assert!(error.contains("the runner refuses this batch"), "{error}");
    assert!(
        error.contains(
            "in flight: executing batch 13 of phase Decode (decode_greedy/2 rows/2 tokens)"
        ),
        "{error}"
    );

    let error = failure_of(&broken);
    assert!(
        error.contains("the storage refused (in flight: giving the cache's segments back)"),
        "{error}"
    );

    output_of(&served);
    assert_eq!(worker.counters().0, 1);
}

#[test]
fn the_watchdog_reads_the_job_in_flight_longest_with_its_batch() {
    let log = Log::default();
    let kernels = serving();

    // The first worker's runner is held inside batch 5; the second worker is held inside a
    // control operation it took later.
    let (entered, in_flight) = crossbeam_channel::bounded(1);
    let (release_batch, released) = crossbeam_channel::bounded::<()>(1);
    let mut runner = LoggingRunner::new("a", Phase::Decode, Arc::clone(&kernels), &log);
    runner.held = Some((vec![5], entered, released));
    let sequences = Sequences::new(GROUP, 1, Vec::new());
    let first = spawn(
        vec![Box::new(runner)],
        &kernels,
        &sequences,
        0,
        &[DEVICE, HOST],
    );
    let (second, _) = worker(&["b"], &log);

    assert_eq!(first.in_flight(), None);
    let stuck = first.submit(1, batch(5, Phase::Decode, 2)).unwrap();
    assert_eq!(in_flight.recv_timeout(LONG), Ok(5));
    let (release, held) = hold(&second);

    let jobs = [first.in_flight(), second.in_flight()];
    let (Some((since, batch)), Some((later, None))) = (jobs[0], jobs[1]) else {
        panic!("both workers are inside a job: {jobs:?}");
    };
    assert_eq!(batch, Some(BatchId(5)));
    assert!(since <= later);
    assert_eq!(
        first.performing().as_deref(),
        Some("executing batch 5 of phase Decode (decode_greedy/2 rows/2 tokens)")
    );
    assert_eq!(
        second.performing().as_deref(),
        Some("holding the worker for a test")
    );

    // The control thread looks again when the job in flight longest reaches the stall bound, and
    // reports each job once it reached it, with its rank and batch.
    let timeout = Duration::from_secs(30);
    let mut watchdog = Watchdog::new(timeout).unwrap();
    assert_eq!(watchdog.deadline(&jobs), Some(since + timeout));
    assert!(watchdog.stalls(later, &jobs).is_empty());

    let stalls = watchdog.stalls(later + timeout, &jobs);
    assert_eq!(stalls.len(), 2);
    assert_eq!((stalls[0].0, stalls[0].1.batch_id), (0, Some(5)));
    assert_eq!((stalls[1].0, stalls[1].1.batch_id), (1, None));

    // Once a job ended nothing is in flight.
    drop(release_batch);
    drop(release);
    output_of(&stuck);
    output_of(&held);
    assert_eq!(first.in_flight(), None);
    assert_eq!(first.performing(), None);
    assert_eq!(
        watchdog.deadline(&[first.in_flight(), second.in_flight()]),
        None
    );
}

#[test]
fn a_closed_queue_cancels_its_queued_jobs_and_takes_no_other() {
    let log = Log::default();
    let (worker, _) = worker(&["a"], &log);

    let (release, held) = hold(&worker);
    let queued = worker.submit(1, batch(1, Phase::Decode, 2)).unwrap();
    worker.queue().close();

    // The queued job's waiter is woken with the cancellation; the job in flight ends.
    assert!(matches!(
        queued.wait_timeout(LONG),
        Some(Outcome::Cancelled)
    ));
    assert!(matches!(
        worker.submit(2, batch(2, Phase::Decode, 2)),
        Err(Error::State(_))
    ));

    drop(release);
    output_of(&held);
    assert!(entries(&log).is_empty());
    worker.close().unwrap();
}

#[test]
fn a_shadow_batch_compares_the_replaced_operators_its_bucket_launches() {
    let log = Log::default();
    let (worker, kernels) = worker(&["a"], &log);
    output_of(&worker.warm_up().unwrap());

    let mut sequence = 0;
    let mut shadow = |generation: u64| {
        let mut batch = batch(20, Phase::Decode, 3);
        batch.generation = generation;
        sequence += 1;
        worker.shadow(sequence, batch).unwrap()
    };

    // A generation nobody prepares has nothing to be compared with.
    let refused = failure_of(&shadow(2));
    assert!(refused.contains("is not preparing"), "{refused}");

    // The bucket of three rows is the one of four, which launches the replaced normalization
    // and not a replaced embedding: one paired forward, of that operator. Graphs with no
    // device run none, which is the failure the pair is named in.
    output_of(&worker.prepare(replacing(&kernels, 2, "rms_norm")).unwrap());
    let refused = failure_of(&shadow(2));
    assert!(
        refused.contains(
            "shadowing `rms_norm` of generation 2 in bucket decode_greedy/4 rows/4 tokens on \
             batch 20"
        ),
        "{refused}"
    );
    output_of(&worker.rollback(2).unwrap());

    // A generation that only adds a kernel replaces nothing the bucket launches: no paired
    // forward runs, and the update states what it has no comparison for.
    let adding = Arc::new(kernels.replace(
        3,
        [KernelEntry::keyed(
            "softmax",
            "dtype_f32".to_owned(),
            unlaunched_kernel(),
        )],
    ));
    output_of(&worker.prepare(adding).unwrap());
    output_of(&shadow(3));

    let uncompared = worker.uncompared();
    assert_eq!(uncompared.len(), 1, "{uncompared:?}");
    assert!(uncompared[0].contains("operator `softmax` is new in generation 3"));

    // The statement belongs to the update: it goes with the generation.
    output_of(&worker.rollback(3).unwrap());
    assert!(worker.uncompared().is_empty());
}

#[test]
fn a_worker_with_no_device_records_no_profiling_session() {
    let log = Log::default();
    let (worker, _) = worker(&["a"], &log);
    assert_eq!(worker.device(), None);
    assert!(worker.fault().is_none());

    let refused = worker
        .profile(crate::profiling::session::Change::Stop {
            session: 1,
            reason: "the test ends it".to_owned(),
            rank: 0,
        })
        .unwrap_err();
    assert!(refused.to_string().contains("no device"), "{refused}");
}

/// The buffer pool's end of a queue pair of `registers` registers, and the engine's mapping of
/// it that a group's sequences write the `completed` word into.
fn queue_pair(registers: usize) -> (QueuePair, QueuePair) {
    let pool = QueuePair::create(64, registers).unwrap();
    let engine = QueuePair::open(pool.duplicate().unwrap(), true).unwrap();
    (pool, engine)
}

/// The runners of `ranks` ranks of a group, one each, serving `Decode` and named after their
/// rank, with the generation they serve.
fn ranks(ranks: usize, log: &Log) -> (Vec<LoggingRunner>, Arc<Generation>) {
    const NAMES: [&str; 3] = ["rank-0", "rank-1", "rank-2"];
    let kernels = serving();

    let runners = NAMES[..ranks]
        .iter()
        .map(|name| LoggingRunner::new(name, Phase::Decode, Arc::clone(&kernels), log))
        .collect();
    (runners, kernels)
}

/// A group of this node's workers of `GROUP`, one per runner given, each mapping the device
/// and host regions, whose `completed` word is held by `registers`. Returns the route, the
/// workers and the group's sequences.
fn group(
    ranks: (Vec<LoggingRunner>, Arc<Generation>),
    registers: Vec<(QueuePair, usize)>,
) -> (Route, Vec<Arc<Worker>>, Arc<Sequences>) {
    let regions = vec![vec![DEVICE, HOST]; ranks.0.len()];
    mapping(ranks, registers, &regions)
}

/// As [`group`], with rank `i` mapping `regions[i]`.
fn mapping(
    (runners, kernels): (Vec<LoggingRunner>, Arc<Generation>),
    registers: Vec<(QueuePair, usize)>,
    regions: &[Vec<u8>],
) -> (Route, Vec<Arc<Worker>>, Arc<Sequences>) {
    let sequences = Sequences::new(GROUP, runners.len(), registers);
    let workers: Vec<Arc<Worker>> = runners
        .into_iter()
        .zip(regions)
        .enumerate()
        .map(|(rank, (runner, regions))| {
            Arc::new(spawn(
                vec![Box::new(runner)],
                &kernels,
                &sequences,
                rank,
                regions,
            ))
        })
        .collect();

    let route = Route::new(vec![Arc::clone(&sequences)], workers.clone(), Vec::new()).unwrap();
    (route, workers, sequences)
}

/// Maps `completion`, the answer to `request` numbered `id`, into every rank of `GROUP` under
/// `slot`, and waits for it.
fn map(route: &Route, slot: u32, id: u64, request: Request, completion: Completion) {
    returned_by(
        &route
            .map(GROUP, slot, runtime::RequestId(id), request, completion)
            .unwrap(),
    );
}

/// Attaches owner 1 in slot 2 and gives it blocks at positions 0 and 1 of class 0, on every
/// rank of `route`'s group.
fn attach_and_allocate(route: &Route) {
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: GROUP,
    };
    map(route, 2, 1, attach, Completion::Attached { slot: 2 });

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
    map(route, 2, 2, allocate, allocated);
}

#[test]
fn a_completion_mapped_before_a_batch_is_seen_by_the_batch_on_every_rank() {
    let log = Log::default();
    let (route, _, _) = group(ranks(2, &log), Vec::new());

    // The map is a job: a batch submitted after it sees its entries, on each rank.
    attach_and_allocate(&route);
    output_of(&route.submit(reading(1, 2)).unwrap());

    let mut executed = entries(&log);
    executed.sort();
    let seen = "execute 1 at generation 1 reading slot 2 at [(0, (0, 7, 1)), (1, (0, 9, 1))]";
    assert_eq!(
        executed,
        [format!("rank-0: {seen}"), format!("rank-1: {seen}")]
    );

    // A pair the tables refuse fails the map, and gives no rank anything: an owner given a
    // block at a position it holds.
    let again = Request::Allocate {
        owner: OwnerId(1),
        class: 0,
        at: 1,
        count: 1,
        units: 0,
    };
    let allocated = Completion::Allocated {
        runs: vec![(DEVICE, 11, 1)],
        evicted: Vec::new(),
    };
    let refused = route
        .map(GROUP, 2, runtime::RequestId(3), again, allocated)
        .unwrap();
    assert!(failure_of(&refused).contains("holds an entry"));

    output_of(&route.submit(reading(2, 2)).unwrap());
    assert!(
        entries(&log)
            .last()
            .unwrap()
            .ends_with("reading slot 2 at [(0, (0, 7, 1)), (1, (0, 9, 1))]")
    );

    // A group nothing here serves is refused by name.
    let stray = route.map(
        GroupId(9),
        2,
        runtime::RequestId(4),
        Request::List,
        Completion::Done,
    );
    assert!(stray.unwrap_err().to_string().contains("worker group 9"));
}

#[test]
fn a_stamped_release_removes_its_entries_and_carries_the_last_batch_launched_before_it() {
    let log = Log::default();
    let (route, workers, _) = group(ranks(2, &log), Vec::new());
    attach_and_allocate(&route);

    let release = |at| Request::Release {
        owner: OwnerId(1),
        class: 0,
        at,
        count: 1,
        after: Vec::new(),
    };
    let stamped = |at, sequence| Request::Release {
        owner: OwnerId(1),
        class: 0,
        at,
        count: 1,
        after: vec![(GROUP, sequence)],
    };

    // Before any batch, nothing a batch addresses can hold the entry: the stamp is 0.
    let (_, requests) = returned_by(&route.stamp(GROUP, vec![release(0)]).unwrap());
    assert_eq!(requests, [stamped(0, 0)]);

    // A stamp is the number of the last batch each rank launched when it applied the request:
    // the batch queued ahead of it on every rank, which has not run yet when it is submitted.
    let (first, held) = hold(&workers[0]);
    let (second, also_held) = hold(&workers[1]);
    let ahead = route.submit(reading(1, 2)).unwrap();
    let stamping = route.stamp(GROUP, vec![release(1)]).unwrap();
    assert!(stamping.outcome().is_none());
    drop((first, second));
    output_of(&held);
    output_of(&also_held);

    output_of(&ahead);
    let (_, requests) = returned_by(&stamping);
    assert_eq!(requests, [stamped(1, 1)]);

    // The entries left every rank's tables before the stamp was returned: the next batch reads
    // none, and the batch before the stamp read both.
    output_of(&route.submit(reading(2, 2)).unwrap());
    let read: Vec<String> = entries(&log)
        .into_iter()
        .filter(|line| line.starts_with("rank-0"))
        .collect();
    assert_eq!(
        read,
        [
            "rank-0: execute 1 at generation 1 reading slot 2 at [(1, (0, 9, 1))]",
            "rank-0: execute 2 at generation 1 reading slot 2 at []",
        ]
    );

    // Only a request the workers stamp is taken, and none is applied when one is refused.
    let attach = Request::Attach {
        owner: OwnerId(2),
        group: GROUP,
    };
    assert!(route.stamp(GROUP, vec![release(5), attach]).is_err());
}

#[test]
fn completed_is_the_smallest_ended_sequence_over_the_ranks_and_passes_no_batch_running() {
    let log = Log::default();
    let (pool, engine) = queue_pair(4);
    let (entered, in_flight) = crossbeam_channel::bounded(1);
    let (release, released) = crossbeam_channel::bounded::<()>(1);
    let (mut runners, kernels) = ranks(2, &log);
    runners[1].held = Some((vec![2], entered, released));
    let (route, _, sequences) = group((runners, kernels), vec![(engine, 2)]);
    let word = || pool.register(2).load(Ordering::Acquire);
    assert_eq!(word(), 0);

    output_of(&route.submit(reading(1, 2)).unwrap());
    assert_eq!(word(), 1);

    // Rank 1 is inside batch 2, which rank 0 has ended: the group's word stays at 1.
    let second = route.submit(reading(2, 2)).unwrap();
    assert_eq!(in_flight.recv_timeout(LONG), Ok(2));
    let third = route.submit(reading(3, 2)).unwrap();
    let deadline = Instant::now() + Duration::from_millis(200);
    while Instant::now() < deadline {
        assert_eq!(word(), 1, "completed passed a batch that has not ended");
        assert_eq!(sequences.completed(), 1);
    }

    // Once every rank ended the batches, the word covers them, before their outputs are read.
    drop(release);
    let (second, third) = (second.wait(), third.wait());
    assert!(matches!(second, Outcome::Success(_)) && matches!(third, Outcome::Success(_)));
    assert_eq!(word(), 3);
    assert_eq!(sequences.completed(), 3);
}

#[test]
fn a_copy_runs_once_with_the_next_batch_and_its_finish_carries_that_batch_s_stamp() {
    let log = Log::default();
    let (route, _, _) = group(ranks(2, &log), Vec::new());
    attach_and_allocate(&route);
    output_of(&route.submit(reading(1, 2)).unwrap());

    // A store of the block at position 0 to the host, which every rank runs.
    let store = Request::Store {
        group: GROUP,
        keys: vec![storage::Key([4; 32])],
        tier: Tier::Host,
    };
    let copy = Completion::Copy {
        group: GROUP,
        keys: vec![storage::Key([4; 32])],
        from: vec![(DEVICE, 7, 1)],
        to: vec![(HOST, 3, 1)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    map(&route, 0, 10, store, copy);

    // The next batch, number 2, runs it on each rank and returns its `Finish` with its stamp.
    let (_, requests) = returned_by(&route.submit(reading(2, 2)).unwrap());
    let finish = Request::Finish {
        copy: runtime::RequestId(10),
        ok: true,
        after: vec![(GROUP, 2)],
    };
    assert_eq!(requests, [finish]);

    // The batch after it runs no copy and returns no `Finish`.
    let (_, requests) = returned_by(&route.submit(reading(3, 2)).unwrap());
    assert!(requests.is_empty());

    let copies: Vec<String> = entries(&log)
        .into_iter()
        .filter(|line| line.contains("copy"))
        .collect();
    assert_eq!(copies.len(), 2, "{copies:?}");
    for rank in ["rank-0", "rank-1"] {
        assert!(
            copies.contains(&format!("{rank}: copy [(10, [((0, 7, 1), (1, 3, 1))])]")),
            "{copies:?}"
        );
    }
}

#[test]
fn a_copy_given_up_before_a_batch_started_it_never_runs_and_returns_no_finish() {
    let log = Log::default();
    let (route, _, _) = group(ranks(2, &log), Vec::new());
    attach_and_allocate(&route);

    // A load into position 2, given up before any batch took it.
    let load = Request::Load {
        owner: OwnerId(1),
        class: 0,
        at: 2,
        keys: vec![storage::Key([5; 32])],
    };
    let copy = Completion::Copy {
        group: GROUP,
        keys: vec![storage::Key([5; 32])],
        from: vec![(HOST, 0, 1)],
        to: vec![(DEVICE, 12, 1)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    map(&route, 2, 20, load, copy);

    let give_up = Request::Finish {
        copy: runtime::RequestId(20),
        ok: false,
        after: Vec::new(),
    };
    let (_, requests) = returned_by(&route.stamp(GROUP, vec![give_up]).unwrap());
    let stamped = Request::Finish {
        copy: runtime::RequestId(20),
        ok: false,
        after: vec![(GROUP, 0)],
    };
    assert_eq!(requests, [stamped]);

    // No batch runs it, none returns a `Finish` for it, and its position stays empty.
    for id in 1..=2 {
        let (_, requests) = returned_by(&route.submit(reading(id, 2)).unwrap());
        assert!(requests.is_empty());
    }
    let log = entries(&log);
    assert!(log.iter().all(|line| !line.contains("copy")), "{log:?}");
    assert!(
        log.last()
            .unwrap()
            .ends_with("reading slot 2 at [(0, (0, 7, 1)), (1, (0, 9, 1))]")
    );
}

/// Maps, on every rank of `route`'s group, the `Copy` that answers request `id`: a store to the
/// host of one block of `units` units in region `from`.
fn store(route: &Route, id: u64, from: u8, units: u32) {
    let key = storage::Key([id as u8; 32]);
    let first = 16 * id as u32;

    let store = Request::Store {
        group: GROUP,
        keys: vec![key],
        tier: Tier::Host,
    };
    let copy = Completion::Copy {
        group: GROUP,
        keys: vec![key],
        from: vec![(from, first, units)],
        to: vec![(HOST, first, units)],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    };
    map(route, 0, id, store, copy);
}

/// The `Finish` that reports each of `copies` run, stamped with the batch numbered `sequence`.
fn finished(copies: &[u64], sequence: u64) -> Vec<Request> {
    copies
        .iter()
        .map(|&copy| Request::Finish {
            copy: runtime::RequestId(copy),
            ok: true,
            after: vec![(GROUP, sequence)],
        })
        .collect()
}

/// The copies `rank` was handed, call by call, with the runs it ran of each.
fn copied(log: &Log, rank: &str) -> Vec<String> {
    entries(log)
        .into_iter()
        .filter_map(|line| {
            line.strip_prefix(&format!("{rank}: copy "))
                .map(str::to_owned)
        })
        .collect()
}

#[test]
fn a_batch_starts_queued_copies_in_their_order_within_the_copy_budget() {
    let log = Log::default();
    let (route, _, _) = group(ranks(1, &log), Vec::new());

    // Copies of 2, 2, 1, 5 and 1 units under a budget of 3 units.
    for (id, units) in [(10, 2), (11, 2), (12, 1), (13, 5), (14, 1)] {
        store(&route, id, DEVICE, units);
    }

    // The second copy would pass the budget, and the third, which would fit, does not pass it
    // in the queue; the copy larger than the budget runs, as the only copy of its batch.
    let started: [&[u64]; 5] = [&[10], &[11, 12], &[13], &[14], &[]];
    for (sequence, copies) in (1..).zip(started) {
        let (_, requests) = returned_by(&route.submit(reading(sequence, 2)).unwrap());
        assert_eq!(requests, finished(copies, sequence), "batch {sequence}");
    }

    assert_eq!(
        copied(&log, "rank-0"),
        [
            "[(10, [((0, 160, 2), (1, 160, 2))])]",
            "[(11, [((0, 176, 2), (1, 176, 2))]), (12, [((0, 192, 1), (1, 192, 1))])]",
            "[(13, [((0, 208, 5), (1, 208, 5))])]",
            "[(14, [((0, 224, 1), (1, 224, 1))])]",
        ]
    );
}

#[test]
fn every_rank_starts_the_same_copies_with_a_batch_whatever_runs_it_maps() {
    let log = Log::default();
    let regions = [
        vec![DEVICE, HOST, PRODUCTS[0]],
        vec![DEVICE, HOST, PRODUCTS[1]],
    ];
    let (route, _, _) = mapping(ranks(2, &log), Vec::new(), &regions);

    // A store of two units from rank 0's product buffer, which rank 1 has no part in, then a
    // store of two units every rank runs: together they pass the budget of three units on the
    // group, whatever each rank's own part of them.
    store(&route, 10, PRODUCTS[0], 2);
    store(&route, 11, DEVICE, 2);

    // Each batch's `Finish`es are joined over the ranks, which fails unless both started the
    // same copies with it.
    for (sequence, copies) in [(1, [10]), (2, [11])] {
        let (_, requests) = returned_by(&route.submit(reading(sequence, 2)).unwrap());
        assert_eq!(requests, finished(&copies, sequence), "batch {sequence}");
    }

    assert_eq!(
        copied(&log, "rank-0"),
        [
            "[(10, [((2, 160, 2), (1, 160, 2))])]",
            "[(11, [((0, 176, 2), (1, 176, 2))])]",
        ]
    );
    // Rank 1 holds no run of the first copy, which asks nothing of it.
    assert_eq!(
        copied(&log, "rank-1"),
        ["[(11, [((0, 176, 2), (1, 176, 2))])]"]
    );
}

/// The one rank of `GROUP` as a text rank holds it: a causal runner serving `Decode` over the KV
/// units and their host pool, and a vision runner serving `VisionEncode` over the product buffer
/// alone, in that order, with tables that map all three regions.
fn text_rank(log: &Log) -> Route {
    let kernels = serving();
    let mut causal = LoggingRunner::new("causal", Phase::Decode, Arc::clone(&kernels), log);
    causal.regions = vec![DEVICE, HOST];
    let mut vision = LoggingRunner::new("vision", Phase::VisionEncode, Arc::clone(&kernels), log);
    vision.regions = vec![PRODUCTS[0]];

    let sequences = Sequences::new(GROUP, 1, Vec::new());
    let worker = spawn(
        vec![Box::new(causal), Box::new(vision)],
        &kernels,
        &sequences,
        0,
        &[DEVICE, HOST, PRODUCTS[0]],
    );
    Route::new(vec![sequences], vec![Arc::new(worker)], Vec::new()).unwrap()
}

/// The `Finish` of `copy`, `ok` as stated, stamped with the batch numbered `sequence`.
fn finish(copy: u64, ok: bool, sequence: u64) -> Request {
    Request::Finish {
        copy: runtime::RequestId(copy),
        ok,
        after: vec![(GROUP, sequence)],
    }
}

#[test]
fn a_kv_copy_that_starts_with_a_vision_encode_is_run_by_the_causal_runner() {
    let log = Log::default();
    let route = text_rank(&log);

    // A store of a KV block to the host, and the vision encode it starts with.
    store(&route, 10, DEVICE, 1);
    let (_, requests) = returned_by(&route.submit(batch(1, Phase::VisionEncode, 1)).unwrap());
    assert_eq!(requests, [finish(10, true, 1)]);

    // The causal runner issued the copy, and the batch ended once its copies were waited for.
    assert_eq!(
        entries(&log),
        [
            "causal: copy [(10, [((0, 160, 1), (1, 160, 1))])]",
            "vision: execute 1 at generation 1",
            "causal: wait for copies",
        ]
    );
}

#[test]
fn a_copy_no_runner_maps_fails_and_the_copies_beside_it_run() {
    let log = Log::default();
    let route = text_rank(&log);

    // A store from the product buffer to the KV host pool, whose two regions no runner maps
    // together, queued ahead of a store of a KV block.
    store(&route, 10, PRODUCTS[0], 1);
    store(&route, 11, DEVICE, 1);
    let (_, requests) = returned_by(&route.submit(batch(1, Phase::Decode, 1)).unwrap());
    assert_eq!(requests, [finish(10, false, 1), finish(11, true, 1)]);

    // No runner was handed the first, and the batch's own runner ran the second.
    assert_eq!(
        entries(&log),
        [
            "causal: copy [(11, [((0, 176, 1), (1, 176, 1))])]",
            "causal: execute 1 at generation 1",
            "causal: wait for copies",
        ]
    );

    // The failed copy ended with its batch: no later batch starts it.
    let (_, requests) = returned_by(&route.submit(batch(2, Phase::VisionEncode, 1)).unwrap());
    assert!(requests.is_empty());
}

#[test]
fn a_batch_a_rank_runs_alone_is_passed_by_the_others_and_ends_on_every_rank() {
    let log = Log::default();
    let (pool, engine) = queue_pair(1);
    let (route, _, sequences) = group(ranks(3, &log), vec![(engine, 0)]);

    // A video decode round named for rank 1 of three.
    let mut alone = batch(1, Phase::Decode, 1);
    alone.media = vec![MediaRow {
        request: RequestId(1),
        slot: 1,
        conditioning: Vec::new(),
        step: 0,
        steps: 1,
        seed: 0,
        width: 832,
        height: 480,
        frames: 25,
        rounds: 1,
        unit: 0,
        rank: Some(1),
        image: None,
        results: Vec::new(),
        reads: 0,
    }];
    let output = output_of(&route.submit(alone.clone()).unwrap());
    assert_eq!(output.calls[0].tokens, [1]);
    assert_eq!(entries(&log), ["rank-1: execute 1 at generation 1"]);

    // Every rank took its number, so the group's word covers it.
    assert_eq!(sequences.completed(), 1);
    assert_eq!(pool.register(0).load(Ordering::Acquire), 1);

    // A rank that is not this node's is refused before any rank takes a number.
    alone.media[0].rank = Some(3);
    assert!(route.submit(alone).is_err());
    output_of(&route.submit(batch(2, Phase::Decode, 1)).unwrap());
    assert_eq!(sequences.completed(), 2);
}

#[test]
fn a_joined_submission_fails_with_the_rank_that_failed() {
    let log = Log::default();
    let (route, _, _) = group(ranks(2, &log), Vec::new());

    // Both ranks refuse the batch; the group's submission is the failure.
    let submission = route.submit(batch(REFUSED, Phase::Decode, 2)).unwrap();
    let refused = failure_of(&submission);
    assert!(
        refused.contains("the runner refuses this batch"),
        "{refused}"
    );

    // A phase no rank of the group serves is refused by name, and takes no number.
    let refused = route.submit(batch(4, UNSERVED, 2)).unwrap_err().to_string();
    assert!(
        refused.contains("Denoise") && refused.contains("[Decode, Decode]"),
        "{refused}"
    );
}

#[test]
fn a_batch_that_fails_reports_the_copies_that_started_with_it_failed() {
    let log = Log::default();
    let (pool, engine) = queue_pair(1);
    let (route, _, sequences) = group(ranks(2, &log), vec![(engine, 0)]);

    // A store every rank starts with the batch the runners refuse, the group's number 1.
    store(&route, 10, DEVICE, 1);
    let submission = route.submit(batch(REFUSED, Phase::Decode, 2)).unwrap();
    let Some(Outcome::Failed(fault)) = submission.wait_timeout(LONG) else {
        panic!("the refused batch fails");
    };
    assert!(
        fault.to_string().contains("the runner refuses this batch"),
        "{fault}"
    );

    // The failure carries the copy's `Finish`, failed and stamped with the batch, and it was
    // posted once every rank had ended the batch: the group's word covers it.
    let finished = fault.downcast_ref::<Error>().map(Error::finished);
    let failed = Request::Finish {
        copy: runtime::RequestId(10),
        ok: false,
        after: vec![(GROUP, 1)],
    };
    assert_eq!(finished, Some(&[failed][..]));
    assert_eq!(sequences.completed(), 1);
    assert_eq!(pool.register(0).load(Ordering::Acquire), 1);

    // The copy ended with the batch: no later batch starts it or reports it again.
    let (_, requests) = returned_by(&route.submit(reading(2, 2)).unwrap());
    assert!(requests.is_empty());
    assert_eq!(
        copied(&log, "rank-0"),
        ["[(10, [((0, 160, 1), (1, 160, 1))])]"]
    );

    // A failed batch that started no copy carries none.
    let submission = route.submit(batch(REFUSED, Phase::Decode, 2)).unwrap();
    let Some(Outcome::Failed(fault)) = submission.wait_timeout(LONG) else {
        panic!("the refused batch fails");
    };
    assert_eq!(
        fault.downcast_ref::<Error>().map(Error::finished),
        Some(&[][..])
    );
}

/// A batch numbered `id` of group `GROUP` with no rows: what a step sends a group whose worker
/// holds copies not yet finished when it formed no other batch for the group.
fn rowless(id: u64) -> Batch {
    Batch {
        requests: Vec::new(),
        ..batch(id, Phase::Decode, 0)
    }
}

#[test]
fn a_batch_with_no_rows_takes_its_number_runs_the_copies_and_no_forward() {
    let log = Log::default();
    let (pool, engine) = queue_pair(1);
    let (route, workers, sequences) = group(ranks(2, &log), vec![(engine, 0)]);

    // Two copies of two units under a budget of three: each batch with no rows starts one.
    store(&route, 10, DEVICE, 2);
    store(&route, 11, DEVICE, 2);

    for (sequence, copy) in [(1, 10), (2, 11)] {
        let (output, requests) = returned_by(&route.submit(rowless(sequence)).unwrap());
        assert!(output.calls.is_empty(), "batch {sequence}");
        assert_eq!(requests, finished(&[copy], sequence), "batch {sequence}");
        assert_eq!(sequences.completed(), sequence);
    }
    assert_eq!(pool.register(0).load(Ordering::Acquire), 2);
    assert_eq!(
        copied(&log, "rank-1"),
        [
            "[(10, [((0, 160, 2), (1, 160, 2))])]",
            "[(11, [((0, 176, 2), (1, 176, 2))])]",
        ]
    );

    // No forward ran: the workers spent no time in one.
    for worker in &workers {
        assert_eq!(worker.counters().1, 0);
    }

    // With nothing queued, a batch with no rows still takes its number and returns no `Finish`.
    let (output, requests) = returned_by(&route.submit(rowless(3)).unwrap());
    assert!(output.calls.is_empty() && requests.is_empty());
    assert_eq!(sequences.completed(), 3);
}

#[test]
fn a_group_s_ranks_on_another_host_take_its_jobs_and_state_what_they_ended() {
    let log = Log::default();
    let listener = service::Listener::bind(&"127.0.0.1:0".parse().unwrap()).unwrap();
    let address = listener.endpoint().to_string();

    // The peer: an engine process holding the group's one rank, serving the scheduler node's
    // jobs until the test closes the link.
    let (release, released) = crossbeam_channel::bounded::<()>(1);
    let peer = {
        let log = Arc::clone(&log);

        std::thread::spawn(move || {
            let channel = register(&address, 0, 7, LONG).unwrap();
            let (ranks, _, _) = group(ranks(1, &log), Vec::new());
            let liveness = Liveness::new(Arc::new(|| {}));
            // This test's scheduler node runs no operation of the data plane.
            let operate: crate::executor::peers::Operate =
                Box::new(|_| Err("this peer runs no operation".to_owned()));
            let upstream = Upstream::serve(channel, Arc::new(ranks), liveness, operate).unwrap();

            let _ = released.recv();
            upstream.close();
        })
    };

    // The scheduler node: it holds no rank of the group, only the link to the peer, whose
    // ended sequence its register holds.
    let (rank, channel) = accept(&listener, 1, 7, LONG).unwrap().pop().unwrap();
    let liveness = Liveness::new(Arc::new(|| {}));
    let remote = Remote::spawn(rank, channel, Arc::clone(&liveness)).unwrap();
    let (pool, engine) = queue_pair(1);
    let sequences = Sequences::new(GROUP, 1, vec![(engine, 0)]);
    let node = Route::new(
        vec![Arc::clone(&sequences)],
        Vec::new(),
        vec![(vec![GROUP], Arc::clone(&remote))],
    )
    .unwrap();

    // The completions and the batch reach the peer's rank in order, and the batch's output and
    // the peer's ended sequence come back.
    attach_and_allocate(&node);
    let output = output_of(&node.submit(reading(1, 2)).unwrap());
    assert_eq!(output.calls[0].tokens, [1]);
    assert_eq!(
        entries(&log),
        ["rank-0: execute 1 at generation 1 reading slot 2 at [(0, (0, 7, 1)), (1, (0, 9, 1))]"]
    );
    assert_eq!(pool.register(0).load(Ordering::Acquire), 1);

    // A stamp is the peer's.
    let release_entry = Request::Release {
        owner: OwnerId(1),
        class: 0,
        at: 0,
        count: 1,
        after: Vec::new(),
    };
    let (_, requests) = returned_by(&node.stamp(GROUP, vec![release_entry]).unwrap());
    assert_eq!(
        requests,
        [Request::Release {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            count: 1,
            after: vec![(GROUP, 1)],
        }]
    );

    // A batch the peer's ranks refuse fails the scheduler's submission with the peer's reason.
    let refused = failure_of(&node.submit(batch(REFUSED, Phase::Decode, 2)).unwrap());
    assert!(
        refused.contains("the runner refuses this batch"),
        "{refused}"
    );

    // A batch of a phase the peer's ranks do not serve, which the group's ranks of another
    // process would, takes its number there and answers no call.
    let passed = output_of(&node.submit(batch(3, UNSERVED, 2)).unwrap());
    assert!(passed.calls.is_empty());

    // A close the scheduler node asked for is not the loss of its peer.
    remote.close();
    drop(release);
    peer.join().unwrap();
    assert_eq!(liveness.lost(), None);
}
