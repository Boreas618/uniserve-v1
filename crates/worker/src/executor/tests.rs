//! Executors without a device: runners over [`Graphs::meta`], which run the profile run, the
//! capture and the whole generation state machine with every device step left out, driven
//! through the executor's queue.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use ipc::PhaseId;
use ipc::execution::{
    Batch, BatchColumns, BatchId, BatchOutput, Batches, CallId, CallOutput, Computation, Extent,
    Outcome, RowState, Shape as BatchShape, Submission, Variant,
};
use nn::functional::{embedding, rms_norm};
use ops::{Generation, Kernel, KernelEntry};
use platform::NativeLibrary;
use uniserve_tensor::{DType, Tensor};

use super::peers::{Liveness, Remote, Upstream, accept, register};
use super::watchdog::{CONTEXT_CHECK, Watchdog};
use super::{Executor, Executors, Job};
use crate::buckets::Bucket;
use crate::graphs::Graphs;
use crate::runner::Runner;
use crate::{Error, Result, lock};

/// A bound on the tests' waits that only a defect reaches.
const LONG: Duration = Duration::from_secs(30);

const VOCABULARY: usize = 32;
const HIDDEN: usize = 16;

/// The bytes of a runner's input tensors: one column of token ids.
const INPUTS: u64 = 256;

/// The batch a runner refuses.
const REFUSED: u64 = 13;

/// What the runners of one test did, in the order they did it.
type Log = Arc<Mutex<Vec<String>>>;

/// A runner of a model of two operators over meta weights. It states what it is asked to do in
/// the test's log and returns one output per call of a batch.
struct LoggingRunner {
    name: &'static str,
    phases: Vec<PhaseId>,
    buckets: Vec<Bucket>,
    graphs: Graphs,
    table: Tensor,
    gain: Tensor,
    log: Log,
    /// The batches whose execution the test holds, with the channels it holds them by: the
    /// runner sends the batch's id once it executes it and waits for the other to close.
    held: Option<(Vec<u64>, Sender<u64>, Receiver<()>)>,
}

impl LoggingRunner {
    fn new(name: &'static str, phase: u16, kernels: Arc<Generation>, log: &Log) -> Self {
        let weight = |address: u64, dims: &[usize]| {
            backend::retained(address, dims, DType::BF16, &backend::meta()).unwrap()
        };

        Self {
            name,
            phases: vec![PhaseId(phase)],
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
        }
    }

    fn state(&self, what: String) {
        lock(&self.log).push(format!("{}: {what}", self.name));
    }
}

impl Runner for LoggingRunner {
    fn phases(&self) -> &[PhaseId] {
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
        // A capture fills the idle inputs of its bucket once.
        self.state(format!("capture {bucket}"));
        columns.fill(0);
        Ok(())
    }

    fn prepare_inputs(&mut self, batch: &Batch) -> Result<Bucket> {
        let bucket = self
            .graphs
            .select(|bucket| bucket.holds(&batch.shape))
            .ok_or_else(|| Error::Invalid(format!("no bucket holds {}", batch.shape)))?;

        self.graphs
            .prepare_inputs(&*self, &bucket, |_columns| Ok(()))?;
        Ok(bucket)
    }

    fn execute(&mut self, batch: &Batch) -> Result<BatchOutput> {
        if let Some((held, entered, released)) = &self.held
            && held.contains(&batch.id.0)
        {
            entered.send(batch.id.0).unwrap();
            let _ = released.recv();
        }
        if batch.id.0 == REFUSED {
            return Err(Error::Invalid("the runner refuses this batch".into()));
        }

        let loads: Vec<u32> = batch.page_loads.iter().map(|copy| copy.rank).collect();
        self.state(format!(
            "execute {} at generation {} with the page loads of ranks {loads:?}",
            batch.id.0,
            self.graphs.generation()
        ));

        Ok(BatchOutput {
            calls: batch
                .calls
                .iter()
                .map(|call| {
                    CallOutput::completed(*call, vec![batch.id.0 as u32], RowState::default())
                })
                .collect(),
            ..BatchOutput::default()
        })
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }

    fn weights_changed(&mut self) -> Result<()> {
        self.state("weights changed".to_owned());
        Ok(())
    }
}

/// A kernel no test launches: an entry of the C library stands in for a package's.
fn kernel() -> Arc<Kernel> {
    let library = NativeLibrary::open("libc.so.6").unwrap();
    Arc::new(Kernel::load_native(library, "getpid").unwrap())
}

/// Generation 1, holding a kernel for every key a profile run of the test runner states.
fn serving() -> Arc<Generation> {
    let profiled = LoggingRunner::new(
        "profiled",
        0,
        Arc::new(Generation::new(0, [])),
        &Log::default(),
    );
    profiled.graphs.profile_run(&profiled).unwrap();
    let kernel = kernel();

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
    let kernel = kernel();
    let keys: Vec<String> = serving.keys(operator).map(str::to_owned).collect();

    Arc::new(
        serving.replace(
            number,
            keys.into_iter()
                .map(|key| KernelEntry::keyed(operator, key, Arc::clone(&kernel))),
        ),
    )
}

/// An executor with no device over one runner per name, the `index`-th serving phase `index`,
/// with the generation they serve.
fn executor(names: &[&'static str], log: &Log) -> (Executor, Arc<Generation>) {
    let kernels = serving();
    let runners = names
        .iter()
        .enumerate()
        .map(|(phase, name)| {
            Box::new(LoggingRunner::new(
                name,
                phase as u16,
                Arc::clone(&kernels),
                log,
            )) as Box<dyn Runner>
        })
        .collect();

    let executor = Executor::spawn(None, runners, Arc::clone(&kernels)).unwrap();
    (executor, kernels)
}

/// A decode batch `id` of `rows` rows and one call for `phase` of group `decode`.
fn batch(id: u64, phase: u16, rows: u32) -> Batch {
    Batch {
        id: BatchId(id),
        phase: PhaseId(phase),
        group: "decode".to_owned(),
        calls: vec![CallId(id)],
        shape: BatchShape::decode(rows, Variant::DecodeGreedy),
        columns: BatchColumns::default(),
        generation: 1,
        cache_units: Vec::new(),
        block_tables: Vec::new(),
        page_loads: Vec::new(),
        page_offloads: Vec::new(),
        latent_restores: Vec::new(),
        host_outputs: Vec::new(),
        images: Vec::new(),
        frees: Vec::new(),
        media: None,
    }
}

/// Holds `executor` inside a control operation until the returned sender is dropped, and
/// returns once the operation is in flight, with its submission.
fn hold(executor: &Executor) -> (Sender<()>, Submission) {
    let (entered, in_flight) = crossbeam_channel::bounded(1);
    let (release, released) = crossbeam_channel::bounded::<()>(1);

    let submission = executor
        .control(
            "holding the executor for a test",
            Box::new(move |_| {
                entered.send(()).unwrap();
                let _ = released.recv();
                Ok(())
            }),
        )
        .unwrap();
    in_flight
        .recv_timeout(LONG)
        .expect("the executor takes the operation");
    (release, submission)
}

/// The output of `submission`, which succeeds.
fn output_of(submission: &Submission) -> BatchOutput {
    match submission.wait_timeout(LONG) {
        Some(Outcome::Success(output)) => output,
        Some(Outcome::Failed(error)) => panic!("the job failed: {error}"),
        Some(Outcome::Cancelled) => panic!("the job was cancelled"),
        None => panic!("the job did not complete"),
    }
}

/// The error of `submission`, which fails.
fn failure_of(submission: &Submission) -> String {
    match submission.wait_timeout(LONG) {
        Some(Outcome::Failed(error)) => error.to_string(),
        other => panic!("the job did not fail: {other:?}"),
    }
}

/// The generation each runner of `executor` serves, read between two jobs.
fn generations(executor: &Executor) -> Vec<u64> {
    executor
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
    let (executor, _) = executor(&["a"], &log);

    let (release, held) = hold(&executor);
    let before = Instant::now();
    let first = executor.submit(batch(1, 0, 2)).unwrap();
    let submitted = Instant::now();
    let control = executor
        .control(
            "recomputing what the runners derive from their weights",
            Box::new(|runners| {
                runners
                    .iter_mut()
                    .try_for_each(|runner| runner.weights_changed())
            }),
        )
        .unwrap();
    let second = executor.submit(batch(2, 0, 2)).unwrap();

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
            "a: execute 1 at generation 1 with the page loads of ranks []",
            "a: weights changed",
            "a: execute 2 at generation 1 with the page loads of ranks []",
        ]
    );

    // The batch's output is the runner's, with the time the batch waited in the queue: at
    // least the time it was held behind the operation, and no more than the time from before
    // its submission to its completion.
    assert_eq!(output.calls.len(), 1);
    assert_eq!(output.calls[0].call, CallId(1));
    assert_eq!(output.calls[0].tokens, [1]);
    let queued = u128::from(output.timing.queued_us);
    assert!(queued >= (released - submitted).as_micros(), "{queued}");
    assert!(queued <= (completed - before).as_micros(), "{queued}");

    assert_eq!(executor.counters().0, 2);
    executor.close().unwrap();
}

#[test]
fn a_waiter_on_a_submission_is_woken_by_its_completion() {
    let log = Log::default();
    let (executor, _) = executor(&["a"], &log);

    let (release, held) = hold(&executor);
    let submission = executor.submit(batch(1, 0, 2)).unwrap();

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
    let (executor, _) = executor(&["a", "b"], &log);
    assert_eq!(executor.phases(), [PhaseId(0), PhaseId(1)]);

    output_of(&executor.submit(batch(1, 1, 2)).unwrap());
    output_of(&executor.submit(batch(2, 0, 2)).unwrap());
    assert_eq!(
        entries(&log),
        [
            "b: execute 1 at generation 1 with the page loads of ranks []",
            "a: execute 2 at generation 1 with the page loads of ranks []",
        ]
    );

    // The refusal is made where the batch is submitted and names both sides.
    let refused = executor.submit(batch(3, 7, 2)).unwrap_err();
    assert!(matches!(refused, Error::Invalid(_)), "{refused}");
    let refused = refused.to_string();
    assert!(
        refused.contains("phase 7") && refused.contains("[0, 1]"),
        "{refused}"
    );

    // So is a join a peer submits through the queue for a phase nothing here serves.
    assert!(
        executor
            .queue()
            .submit(Job::Execute(Box::new(batch(4, 9, 1))))
            .is_err()
    );
}

#[test]
fn warm_up_captures_every_bucket_largest_first_and_batches_run_between_captures() {
    let log = Log::default();
    let (executor, _) = executor(&["a", "b"], &log);

    // The warm-up and a batch are queued behind the job in flight, in that order.
    let (release, held) = hold(&executor);
    let warm_up = executor.warm_up().unwrap();
    let served = executor.submit(batch(1, 0, 2)).unwrap();
    drop(release);

    output_of(&held);
    output_of(&served);
    output_of(&warm_up);

    // One bucket per turn: the batch is executed after the first capture, not after the last.
    assert_eq!(
        entries(&log),
        [
            "a: capture decode_greedy/8 rows/8 tokens",
            "a: execute 1 at generation 1 with the page loads of ranks []",
            "a: capture decode_greedy/4 rows/4 tokens",
            "a: capture decode_greedy/2 rows/2 tokens",
            "b: capture decode_greedy/8 rows/8 tokens",
            "b: capture decode_greedy/4 rows/4 tokens",
            "b: capture decode_greedy/2 rows/2 tokens",
        ]
    );

    // Every bucket of every runner has its graph, and a second warm-up captures nothing.
    let captured = executor
        .call("reading the captured buckets", |runners| {
            Ok(runners
                .iter()
                .map(|runner| runner.graphs().buckets().len())
                .collect::<Vec<_>>())
        })
        .unwrap();
    assert_eq!(captured, [3, 3]);

    output_of(&executor.warm_up().unwrap());
    assert_eq!(entries(&log).len(), 7);
}

#[test]
fn a_generation_is_prepared_switched_to_and_committed_on_every_runner_between_batches() {
    let log = Log::default();
    let (executor, kernels) = executor(&["a", "b"], &log);
    output_of(&executor.warm_up().unwrap());

    // The preparation leaves the serving generation serving.
    let next = replacing(&kernels, 2, "rms_norm");
    output_of(&executor.prepare(next).unwrap());
    assert_eq!(generations(&executor), [1, 1]);
    assert!(executor.uncompared().is_empty());
    lock(&log).clear();

    // The switch takes effect between the batch submitted before it and the one after it.
    let (release, held) = hold(&executor);
    let before = executor.submit(batch(1, 0, 2)).unwrap();
    let switch = executor.switch(2).unwrap();
    let after = executor.submit(batch(2, 1, 2)).unwrap();
    drop(release);
    for submission in [&held, &before, &switch, &after] {
        output_of(submission);
    }

    assert_eq!(
        entries(&log),
        [
            "a: execute 1 at generation 1 with the page loads of ranks []",
            "b: execute 2 at generation 2 with the page loads of ranks []",
        ]
    );
    assert_eq!(generations(&executor), [2, 2]);

    // The committed generation serves, and nothing of it is left to roll back.
    output_of(&executor.commit(2).unwrap());
    assert_eq!(generations(&executor), [2, 2]);
    let refused = failure_of(&executor.rollback(2).unwrap());
    assert!(refused.contains("rolling back generation 2"), "{refused}");
    assert_eq!(generations(&executor), [2, 2]);
}

#[test]
fn a_generation_is_rolled_back_prepared_or_on_trial() {
    let log = Log::default();
    let (executor, kernels) = executor(&["a", "b"], &log);
    output_of(&executor.warm_up().unwrap());

    // Prepared and never switched to: its graphs go, and the next update starts from the
    // serving generation.
    output_of(
        &executor
            .prepare(replacing(&kernels, 2, "rms_norm"))
            .unwrap(),
    );
    output_of(&executor.rollback(2).unwrap());
    assert_eq!(generations(&executor), [1, 1]);
    let refused = failure_of(&executor.switch(2).unwrap());
    assert!(refused.contains("was not prepared"), "{refused}");
    assert_eq!(generations(&executor), [1, 1]);

    // On trial: the generation it replaced serves again.
    output_of(
        &executor
            .prepare(replacing(&kernels, 3, "rms_norm"))
            .unwrap(),
    );
    output_of(&executor.switch(3).unwrap());
    assert_eq!(generations(&executor), [3, 3]);
    output_of(&executor.rollback(3).unwrap());
    assert_eq!(generations(&executor), [1, 1]);

    // A generation the executor holds nothing of is rolled back by doing nothing.
    output_of(&executor.rollback(9).unwrap());
    assert_eq!(generations(&executor), [1, 1]);
}

#[test]
fn a_switch_before_the_preparation_ended_switches_no_runner() {
    let log = Log::default();
    let (executor, kernels) = executor(&["a", "b"], &log);
    output_of(&executor.warm_up().unwrap());

    // The preparation's first turn runs, and the switch is taken before its buckets' turns.
    let (release, held) = hold(&executor);
    let prepare = executor
        .prepare(replacing(&kernels, 2, "rms_norm"))
        .unwrap();
    let switch = executor.switch(2).unwrap();
    drop(release);
    output_of(&held);

    let refused = failure_of(&switch);
    assert!(refused.contains("is not prepared"), "{refused}");
    output_of(&prepare);
    assert_eq!(generations(&executor), [1, 1]);

    // A generation that lacks a kernel the runners launch is refused, and is not being
    // prepared afterwards.
    output_of(&executor.rollback(2).unwrap());
    let refused = failure_of(&executor.prepare(Arc::new(Generation::new(4, []))).unwrap());
    assert!(refused.contains("preparing generation 4"), "{refused}");
    output_of(
        &executor
            .prepare(replacing(&kernels, 5, "rms_norm"))
            .unwrap(),
    );
}

#[test]
fn a_failed_job_states_the_work_in_flight_and_the_jobs_behind_it_run() {
    let log = Log::default();
    let (executor, _) = executor(&["a"], &log);

    let (release, held) = hold(&executor);
    let refused = executor.submit(batch(REFUSED, 0, 2)).unwrap();
    let broken = executor
        .control(
            "giving the cache's segments back",
            Box::new(|_| Err(Error::Resource("the storage refused"))),
        )
        .unwrap();
    let served = executor.submit(batch(2, 0, 2)).unwrap();
    drop(release);
    output_of(&held);

    // The error carries the job it was taken in, and so does the submission's label.
    let error = failure_of(&refused);
    assert!(error.contains("the runner refuses this batch"), "{error}");
    assert!(
        error.contains("in flight: executing batch 13 of phase 0"),
        "{error}"
    );
    assert_eq!(
        refused.failure_label().as_deref(),
        Some("executing batch 13 of phase 0 (decode_greedy/2 rows/2 tokens)")
    );

    let error = failure_of(&broken);
    assert!(
        error.contains("the storage refused (in flight: giving the cache's segments back)"),
        "{error}"
    );

    // A failure classifies as the error it carries.
    let Some(Outcome::Failed(fault)) = broken.outcome() else {
        panic!("the operation failed");
    };
    let failure = crate::Failure::classify(Error::Shared(fault), None);
    assert_eq!(failure.code, crate::error::ErrorCode::ResourceError);

    output_of(&served);
    assert_eq!(executor.counters().0, 1);
}

#[test]
fn the_watchdog_reads_the_job_in_flight_longest_with_its_batch() {
    let log = Log::default();
    let kernels = serving();

    // The first executor's runner is held inside batch 5; the second executor is held inside a
    // control operation it took later.
    let (entered, in_flight) = crossbeam_channel::bounded(1);
    let (release_batch, released) = crossbeam_channel::bounded::<()>(1);
    let mut runner = LoggingRunner::new("a", 0, Arc::clone(&kernels), &log);
    runner.held = Some((vec![5], entered, released));
    let first = Executor::spawn(None, vec![Box::new(runner)], Arc::clone(&kernels)).unwrap();
    let (second, _) = executor(&["b"], &log);

    assert_eq!(first.in_flight(), None);
    let stuck = first.submit(batch(5, 0, 2)).unwrap();
    assert_eq!(in_flight.recv_timeout(LONG), Ok(5));
    let (release, held) = hold(&second);

    let jobs = [first.in_flight(), second.in_flight()];
    let (Some((since, batch)), Some((later, None))) = (jobs[0], jobs[1]) else {
        panic!("both executors are inside a job: {jobs:?}");
    };
    assert_eq!(batch, Some(BatchId(5)));
    assert!(since <= later);
    assert_eq!(
        first.performing().as_deref(),
        Some("executing batch 5 of phase 0 (decode_greedy/2 rows/2 tokens)")
    );
    assert_eq!(
        second.performing().as_deref(),
        Some("holding the executor for a test")
    );

    // The control thread looks again when the job in flight longest has run for a second, and
    // reports each job once it reached the stall bound, with its rank and batch.
    let mut watchdog = Watchdog::new(Duration::from_secs(30)).unwrap();
    assert_eq!(watchdog.deadline(&jobs), Some(since + CONTEXT_CHECK));
    assert!(watchdog.stalls(later, &jobs).is_empty());

    let stalls = watchdog.stalls(later + watchdog.stall_bound(), &jobs);
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
    let (executor, _) = executor(&["a"], &log);

    let (release, held) = hold(&executor);
    let queued = executor.submit(batch(1, 0, 2)).unwrap();
    executor.queue().close();

    // The queued job's waiter is woken with the cancellation; the job in flight ends.
    assert!(matches!(
        queued.wait_timeout(LONG),
        Some(Outcome::Cancelled)
    ));
    assert!(matches!(
        executor.submit(batch(2, 0, 2)),
        Err(Error::State(_))
    ));

    drop(release);
    output_of(&held);
    assert!(entries(&log).is_empty());
    executor.close().unwrap();
}

#[test]
fn a_shadow_batch_compares_the_replaced_operators_its_bucket_launches() {
    let log = Log::default();
    let (executor, kernels) = executor(&["a"], &log);
    output_of(&executor.warm_up().unwrap());

    let shadow = |generation: u64| {
        let mut batch = batch(20, 0, 3);
        batch.generation = generation;
        executor.shadow(batch).unwrap()
    };

    // A generation nobody prepares has nothing to be compared with.
    let refused = failure_of(&shadow(2));
    assert!(refused.contains("is not preparing"), "{refused}");

    // The bucket of three rows is the one of four, which launches the replaced normalization
    // and not a replaced embedding: one paired forward, of that operator. Graphs with no
    // device run none, which is the failure the pair is named in.
    output_of(
        &executor
            .prepare(replacing(&kernels, 2, "rms_norm"))
            .unwrap(),
    );
    let refused = failure_of(&shadow(2));
    assert!(
        refused.contains(
            "shadowing `rms_norm` of generation 2 in bucket decode_greedy/4 rows/4 tokens on \
             batch 20"
        ),
        "{refused}"
    );
    output_of(&executor.rollback(2).unwrap());

    // A generation that only adds a kernel replaces nothing the bucket launches: no paired
    // forward runs, and the update states what it has no comparison for.
    let adding = Arc::new(kernels.replace(
        3,
        [KernelEntry::keyed(
            "softmax",
            "dtype_f32".to_owned(),
            kernel(),
        )],
    ));
    output_of(&executor.prepare(adding).unwrap());
    output_of(&shadow(3));

    let uncompared = executor.uncompared();
    assert_eq!(uncompared.len(), 1, "{uncompared:?}");
    assert!(uncompared[0].contains("operator `softmax` is new in generation 3"));

    // The statement belongs to the update: it goes with the generation.
    output_of(&executor.rollback(3).unwrap());
    assert!(executor.uncompared().is_empty());
}

#[test]
fn an_executor_with_no_device_records_no_profiling_session() {
    let log = Log::default();
    let (executor, _) = executor(&["a"], &log);
    assert_eq!(executor.device(), None);
    assert!(matches!(executor.sticky_error(), Ok(None)));

    let refused = executor
        .profile(crate::profiling::session::Change::Stop {
            session: 1,
            reason: "the test ends it".to_owned(),
            rank: 0,
        })
        .unwrap_err();
    assert!(refused.to_string().contains("no device"), "{refused}");
}

/// Three executors of one node: two ranks of group `decode` and one of group `prefill`, each
/// with one runner serving phase 0.
fn node(log: &Log) -> Executors {
    let executors = [
        ("decode", "decode-0"),
        ("decode", "decode-1"),
        ("prefill", "prefill-0"),
    ]
    .into_iter()
    .map(|(group, name)| (group.to_owned(), Arc::new(executor(&[name], log).0)))
    .collect();

    Executors::new(executors, Vec::new(), Vec::new())
}

#[test]
fn a_batch_is_submitted_to_every_executor_of_its_group_and_joined() {
    let log = Log::default();
    let node = node(&log);
    assert_eq!(node.device_count(), 3);

    // Each rank of the group gets the batch, with the page copies that name it.
    let mut decode = batch(1, 0, 2);
    decode.page_loads = [1, 0, 1]
        .into_iter()
        .map(|rank| ipc::PageCopy {
            rank,
            pool: 0,
            group_id: 0,
            host_page: 0,
            units: Vec::new(),
            staged: false,
        })
        .collect();
    assert_eq!(node.devices(&decode).unwrap(), [0, 1]);

    let output = output_of(&Batches::submit(&node, decode).unwrap());
    assert_eq!(output.calls[0].call, CallId(1));

    let mut executed = entries(&log);
    executed.sort();
    assert_eq!(
        executed,
        [
            "decode-0: execute 1 at generation 1 with the page loads of ranks [0]",
            "decode-1: execute 1 at generation 1 with the page loads of ranks [1, 1]",
        ]
    );

    // Another group's batch goes to that group's executor alone.
    let mut prefill = batch(2, 0, 2);
    prefill.group = "prefill".to_owned();
    assert_eq!(node.devices(&prefill).unwrap(), [2]);
    output_of(&node.submit(prefill).unwrap());
    assert_eq!(
        entries(&log).last().map(String::as_str),
        Some("prefill-0: execute 2 at generation 1 with the page loads of ranks []")
    );

    // A group or a phase nothing here serves is refused by name.
    let mut stray = batch(3, 0, 2);
    stray.group = "encode".to_owned();
    let refused = node.submit(stray).unwrap_err().to_string();
    assert!(refused.contains("\"encode\""), "{refused}");

    let refused = node.submit(batch(4, 5, 2)).unwrap_err().to_string();
    assert!(
        refused.contains("phase 5") && refused.contains("[0, 0]"),
        "{refused}"
    );
}

#[test]
fn a_joined_submission_fails_with_the_rank_that_failed() {
    let log = Log::default();
    let node = node(&log);

    // Both ranks refuse the batch; the group's submission is the failure.
    let submission = node.submit(batch(REFUSED, 0, 2)).unwrap();
    let refused = failure_of(&submission);
    assert!(
        refused.contains("the runner refuses this batch"),
        "{refused}"
    );
}

#[test]
fn a_control_operation_is_submitted_to_every_executor_of_a_node() {
    let log = Log::default();
    let node = node(&log);

    let ranks = Arc::new(Mutex::new(Vec::new()));
    let submission = node
        .control("copying each rank's pages", |rank| {
            let ranks = Arc::clone(&ranks);
            Box::new(move |runners| {
                lock(&ranks).push((rank, runners.len()));
                Ok(())
            })
        })
        .unwrap();
    output_of(&submission);

    let mut ranks = lock(&ranks).clone();
    ranks.sort_unstable();
    assert_eq!(ranks, [(0, 1), (1, 1), (2, 1)]);
}

#[test]
fn the_media_query_is_served_from_the_buckets_the_runners_state() {
    let log = Log::default();
    let kernels = serving();

    let mut runner = LoggingRunner::new("media", 0, Arc::clone(&kernels), &log);
    runner.buckets = vec![
        Bucket::host(Variant::Mux),
        Bucket::clip(25, 480, 832),
        Bucket::clip(13, 480, 832),
    ];
    let executor = Executor::spawn(None, vec![Box::new(runner)], kernels).unwrap();
    let node = Executors::new(
        vec![("media".to_owned(), Arc::new(executor))],
        Vec::new(),
        Vec::new(),
    );

    let computation = |call, extent| Computation {
        call,
        rows: 1,
        tokens: 0,
        extent,
    };
    let clip = |frames| Extent::Clip {
        frames,
        height: 480,
        width: 832,
    };

    // A stated extent is served by the bucket of exactly that extent, or not at all.
    let decode = computation(ipc::MediaCall::VideoDecoding, clip(25));
    assert_eq!(
        node.media(PhaseId(0), &decode).unwrap(),
        Some(Bucket::clip(25, 480, 832))
    );
    let other = computation(ipc::MediaCall::VideoDecoding, clip(49));
    assert_eq!(Batches::media(&node, PhaseId(0), &other).unwrap(), None);

    // A host call's bucket holds every computation of its call; a call no bucket states is not
    // served, and a phase no executor serves is refused.
    let mux = computation(ipc::MediaCall::Muxing, Extent::None);
    assert_eq!(
        node.media(PhaseId(0), &mux).unwrap(),
        Some(Bucket::host(Variant::Mux))
    );
    let denoise = computation(ipc::MediaCall::Denoising, Extent::None);
    assert_eq!(node.media(PhaseId(0), &denoise).unwrap(), None);
    assert!(node.media(PhaseId(4), &mux).is_err());
}

#[test]
fn a_batch_of_ranks_on_another_host_runs_there_and_its_output_returns() {
    let log = Log::default();
    let listener = ipc::Listener::bind(&["127.0.0.1:0"]).unwrap();
    let address = listener.addresses()[0].clone();

    // The peer: an engine process holding the group's one rank, serving the scheduler node's
    // batches until the test closes the link.
    let (release, released) = crossbeam_channel::bounded::<()>(1);
    let peer = {
        let log = Arc::clone(&log);

        std::thread::spawn(move || {
            let channel = register(&address, 0, 7, LONG).unwrap();
            let ranks = Executors::new(
                vec![("decode".to_owned(), Arc::new(executor(&["peer"], &log).0))],
                Vec::new(),
                Vec::new(),
            );
            let waker = ipc::Waker::new().unwrap();
            let liveness = Liveness::new(waker.wake(), Duration::from_secs(3600));
            let upstream = Upstream::serve(
                channel,
                Arc::new(ranks),
                ipc::MainProcessInfo::default(),
                liveness,
            )
            .unwrap();

            let _ = released.recv();
            upstream.close();
        })
    };

    // The scheduler node: it holds no rank of the group, only the link to the peer.
    let (rank, channel) = accept(&listener, 1, 7, LONG).unwrap().pop().unwrap();
    let waker = ipc::Waker::new().unwrap();
    let liveness = Liveness::new(waker.wake(), Duration::from_secs(3600));
    let remote = Remote::spawn(rank, channel, Arc::clone(&liveness)).unwrap();
    let node = Executors::new(
        Vec::new(),
        vec![(vec!["decode".to_owned()], Arc::clone(&remote))],
        Vec::new(),
    );

    let sent = batch(1, 0, 2);
    assert_eq!(node.devices(&sent).unwrap(), [0]);
    let output = output_of(&node.submit(sent).unwrap());
    assert_eq!(output.calls[0].tokens, [1]);
    assert_eq!(
        entries(&log),
        ["peer: execute 1 at generation 1 with the page loads of ranks []"]
    );

    // A batch the peer's ranks refuse fails the scheduler's submission with the peer's reason.
    let refused = failure_of(&node.submit(batch(REFUSED, 0, 2)).unwrap());
    assert!(
        refused.contains("the runner refuses this batch"),
        "{refused}"
    );
    let refused = failure_of(&node.submit(batch(2, 6, 2)).unwrap());
    assert!(refused.contains("phase 6"), "{refused}");

    // A close the scheduler node asked for is not the loss of its peer.
    remote.close();
    drop(release);
    peer.join().unwrap();
    assert_eq!(liveness.lost(), None);
}

#[test]
#[ignore = "requires a GPU"]
fn an_executor_of_a_device_waits_for_events_recorded_on_its_streams() {
    let log = Log::default();
    let kernels = serving();
    let device = platform::DeviceId(0);
    let runner = LoggingRunner::new("a", 0, Arc::clone(&kernels), &log);
    let executor = Executor::spawn(Some(device), vec![Box::new(runner)], kernels).unwrap();

    // A storage owner's work is issued by the executor: an event recorded on a stream in one
    // job is waited for in a later one.
    let event = Arc::new(platform::Event::new(platform::EventFlags::default()));
    let recorded = Arc::clone(&event);
    let issued = executor
        .control(
            "recording an event behind a storage owner's copies",
            Box::new(move |_| {
                let stream = platform::Stream::new(device)?;
                recorded.record(&stream)?;
                Ok(())
            }),
        )
        .unwrap();
    let waited = executor.wait(vec![Arc::clone(&event)]).unwrap();

    output_of(&issued);
    output_of(&waited);
    assert!(event.query().unwrap());
    assert!(matches!(executor.sticky_error(), Ok(None)));

    // A batch's output carries what the rank observed, once something observes it.
    let output = output_of(&executor.submit(batch(1, 0, 2)).unwrap());
    assert_eq!(output.calls.len(), 1);
}
