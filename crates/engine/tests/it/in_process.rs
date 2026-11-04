//! The scheduler over the executor crate's batch path: `SchedulerHandle::over_batches`
//! builds the handle the launched path builds, over a pool this process did not
//! launch, and refuses what it cannot serve.
//!
//! The pool here answers no batch: these tests state the constructor's contract —
//! the capabilities come from what the executor reports, the command surface is
//! the launched path's, and a pool whose facts are absent is refused rather than
//! guessed.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use engine::{EngineConfig, SchedulerHandle};
use executor::batch::{Batch, Batches, Inventory, Placement, Submission};
use ipc::{ComponentConfig, WorkerEndpoint, WorkerInfo};

/// How long a command answer may take.
const ANSWER: Duration = Duration::from_secs(10);

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
    fn submit(&self, batch: Batch) -> Result<Submission, executor::Error> {
        self.submitted.lock().expect("the pool's batches").push(batch);
        Err(executor::Error::Invalid(
            "this test's pool serves no batch".to_owned(),
        ))
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, executor::Error> {
        self.submitted.lock().expect("the pool's batches").push(batch);
        Err(executor::Error::Invalid(
            "this test's pool runs no shadow batch".to_owned(),
        ))
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

/// One rank of a text pool: prefill and decode, eight request rows, a prefix
/// cache, and the batch bounds the launch stated.
fn rank(host: &str) -> WorkerInfo {
    WorkerInfo {
        endpoint: WorkerEndpoint {
            host: host.to_owned(),
            address_space: format!("{host}-space"),
            incarnation: format!("{host}-rank"),
        },
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        request_slots: 8,
        max_batch_calls: 16,
        max_batch_tokens: 512,
        prefix_cache: None,
        buffer_pool_bytes: 1 << 20,
        ..WorkerInfo::default()
    }
}

/// A text pool of one rank named `model`.
fn pool() -> Inventory {
    Inventory {
        group: "model".to_owned(),
        ranks: vec![rank("gpu-a")],
        placement: Placement {
            queue_depth: 1,
            ranks: 1,
            components: std::collections::BTreeMap::from([(
                ipc::DEFAULT_COMPONENT.to_owned(),
                ComponentConfig {
                    ranks: vec![0],
                    parallel_config: Default::default(),
                    units_per_rank: 1,
                    weights: Default::default(),
                    distribution: None,
                },
            )]),
        },
        max_model_len: 4096,
    }
}

/// The configuration a main process holds: the simulation's facts, since this
/// test runs no device.
fn config() -> EngineConfig {
    let mut config = EngineConfig::sim("in-process");
    config.max_running_requests = 4;
    config
}

#[test]
fn a_pool_the_process_did_not_launch_serves_the_launched_paths_commands() {
    let batch_sink = Arc::new(Pool::new(vec![pool()]));
    let batches: Arc<dyn Batches> = batch_sink.clone();
    let handle = SchedulerHandle::over_batches(config(), batches).expect("the pool serves");

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
        batch_sink.submitted.lock().expect("the pool's batches").is_empty(),
        "no batch reached the executor before a policy serves"
    );
    handle.shutdown();
}

#[test]
fn a_config_of_more_than_one_rank_is_not_one_handle() {
    let mut config = config();
    config.data_parallel_size = 2;
    let batches: Arc<dyn Batches> = Arc::new(Pool::new(vec![pool()]));
    let refused = SchedulerHandle::over_batches(config, batches)
        .err()
        .expect("one handle, one rank");
    let refused = format!("{refused:#}");
    assert!(refused.contains("2 ranks"), "{refused}");
}

#[test]
fn a_pool_that_states_no_ranks_is_refused_by_name() {
    let empty = Inventory {
        group: "model".to_owned(),
        ranks: Vec::new(),
        ..pool()
    };
    let batches: Arc<dyn Batches> = Arc::new(Pool::new(vec![empty]));
    let refused = SchedulerHandle::over_batches(config(), batches)
        .err()
        .expect("a pool without ranks");
    let refused = format!("{refused:#}");
    assert!(refused.contains("no rank"), "{refused}");
}

#[test]
fn an_executor_that_states_no_pool_is_refused_rather_than_guessed() {
    let batches: Arc<dyn Batches> = Arc::new(Pool::new(Vec::new()));
    let refused = SchedulerHandle::over_batches(config(), batches)
        .err()
        .expect("an executor states its pools");
    let refused = format!("{refused:#}");
    assert!(refused.contains("at least one pool"), "{refused}");
}
