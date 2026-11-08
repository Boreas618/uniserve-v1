//! The scheduler over the worker crate's batch path: `SchedulerHandle::over_batches`
//! builds the handle the launched path builds, over a pool this process did not
//! launch, and refuses what it cannot serve.
//!
//! The pool here answers no batch: these tests state the constructor's contract —
//! the capabilities come from what the worker reports, the command surface is
//! the launched path's, and a pool whose facts are absent is refused rather than
//! guessed.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use engine::{SchedulerConfig, SchedulerHandle};
use proto::config::EngineArgs;
use worker::batch::{Batch, Batches, Inventory, Submission};

/// How long a command answer may take.
///
/// Generous because the first one is the whole installation of a policy: the
/// tree's `fcfs` module is built for `wasm32-unknown-unknown` on first use and
/// then loaded and instantiated, on a machine that runs this suite's tests in
/// parallel. The bound is the harness's, not the mechanism's; a mechanism that
/// stopped answering would still fail here, only later.
const ANSWER: Duration = Duration::from_secs(60);

/// Blocks until a command answers, and answers its result.
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => panic!("the scheduler did not answer: {error}"),
    }
}

/// A pool that serves nothing and states what it is.
struct Pool {
    inventory: Vec<Inventory>,
    /// The batches the mechanism handed it, so a test can see none was needed.
    submitted: Mutex<Vec<Batch>>,
}

impl Pool {
    fn new(inventory: Vec<Inventory>) -> Self {
        Self {
            inventory,
            submitted: Mutex::new(Vec::new()),
        }
    }
}

impl Batches for Pool {
    fn submit(&self, batch: Batch) -> Result<Submission, worker::Error> {
        self.submitted
            .lock()
            .expect("the pool's batches")
            .push(batch);
        Err(worker::Error::Invalid(
            "this test's pool serves no batch".to_owned(),
        ))
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, worker::Error> {
        self.submitted
            .lock()
            .expect("the pool's batches")
            .push(batch);
        Err(worker::Error::Invalid(
            "this test's pool runs no shadow batch".to_owned(),
        ))
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

/// The arguments a main process holds, with the simulation's facts since these tests run no
/// device, and the mechanism's limits of its pool.
fn config() -> (EngineArgs, SchedulerConfig) {
    let limits = SchedulerConfig {
        max_running_requests: 4,
        ..SchedulerConfig::default()
    };
    (engine::sim_args("in-process"), limits)
}

#[test]
fn a_pool_the_process_did_not_launch_serves_the_launched_paths_commands() {
    let batch_sink = Arc::new(Pool::new(vec![crate::support::pool("model")]));
    let batches: Arc<dyn Batches> = batch_sink.clone();
    let handle = {
        let (args, limits) = config();
        SchedulerHandle::over_batches(&args, limits, batches)
    }
    .expect("the pool serves");

    // The command surface is the launched path's: the handle answers its own
    // commands over the same channel.
    let progress = handle
        .progress()
        .recv_timeout(ANSWER)
        .expect("the scheduler answers with its progress");
    assert_eq!(progress.step, 0, "no step ran: no policy serves yet");
    // Nothing serves yet: a switchover has no prepared policy to put in, and
    // the refusal says so rather than applying nothing silently.
    let refused = answer(handle.stage(engine::Stage::Switchover))
        .expect_err("no policy is prepared before the epoch's first precopy");
    assert!(refused.contains("without a policy prepared"), "{refused}");

    // A scheduler with no policy forms no batch: the pool was asked for none.
    assert!(
        batch_sink
            .submitted
            .lock()
            .expect("the pool's batches")
            .is_empty(),
        "no batch reached the worker before a policy serves"
    );
    handle.shutdown();
}

#[test]
fn a_launch_of_ranks_the_worker_does_not_serve_is_refused() {
    let (mut args, limits) = config();
    args.world_size = 2;
    let batches: Arc<dyn Batches> = Arc::new(Pool::new(vec![crate::support::pool("model")]));
    let refused = SchedulerHandle::over_batches(&args, limits, batches)
        .err()
        .expect("the launch and the worker state one pool");
    let refused = format!("{refused:#}");
    assert!(refused.contains("2 ranks"), "{refused}");
}

#[test]
fn a_pool_that_states_no_ranks_is_refused_by_name() {
    let empty = Inventory {
        group: "model".to_owned(),
        ranks: Vec::new(),
        ..crate::support::pool("model")
    };
    let batches: Arc<dyn Batches> = Arc::new(Pool::new(vec![empty]));
    let refused = {
        let (args, limits) = config();
        SchedulerHandle::over_batches(&args, limits, batches)
    }
    .err()
    .expect("a pool without ranks");
    let refused = format!("{refused:#}");
    assert!(refused.contains("no rank"), "{refused}");
}

#[test]
fn an_worker_that_states_no_pool_is_refused_rather_than_guessed() {
    let batches: Arc<dyn Batches> = Arc::new(Pool::new(Vec::new()));
    let refused = {
        let (args, limits) = config();
        SchedulerHandle::over_batches(&args, limits, batches)
    }
    .err()
    .expect("a worker states its pools");
    let refused = format!("{refused:#}");
    assert!(refused.contains("at least one pool"), "{refused}");
}
