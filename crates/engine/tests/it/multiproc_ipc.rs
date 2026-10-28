//! Multiprocess launch, request lifecycle, token execution, close and rank
//! recovery.
//!
//! These tests launch real worker processes (`uniserve worker` with the
//! weightless stub model on CPU) and drive them through the engine's
//! `WorkerGroup`, `WorkerExecutor` or `SchedulerHandle`, or through a raw
//! `Channel` when the worker's own protocol handling is under test.
//! Lifecycle batches (Start, Finish, Free) exercise registration, startup,
//! Info, admission, retirement, close and signals; token batches run the
//! stub's text runtime (prefill, decode, verification, sampling, logprobs and
//! device token relays). The executable comes from
//! `UNISERVE_WORKER_EXECUTABLE` or the repository environment's
//! `.venv/bin/uniserve`. Several tests replace that executable with a
//! small wrapper script that records a worker's process id, exits a chosen rank, starts a relaunched rank as a
//! stand-in that reports changed worker info, registers under another build
//! or for an earlier launch, or holds a rank's exit until its peers have
//! closed.
//!
//! Faults are injected at the OS boundary: `/proc/thread-self/children` and
//! `/proc/<pid>/cmdline` locate a rank process, and `PausedProcess` stops,
//! resumes or kills it with signals. These are Linux interfaces, and the file
//! compiles only on Linux.

#![cfg(target_os = "linux")]

use std::fs::remove_file;
use std::os::unix::fs::MetadataExt as _;
use std::os::unix::process::ExitStatusExt as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use anyhow::Context as _;
use engine::{
    CallReaders, EngineConfig, Executor, RankBuffer, SchedulerHandle, WorkerFailure, WorkerGroup,
    WorkerGroupConfig, WorkerProcessArgs,
};
use engine::{DiffusionRequest, Event, Request};
use ipc::{
    Batch, BatchCommand, BlockTable, Bounds, BufferAllocation, CacheUnitAllocation, Call,
    CallCoordinates, CallId, CallKind, CallStatus, DType, ErrorCode, ForwardBatch, ForwardMode,
    GenerationParams, LatentParams, LaunchDescriptor, MediaCall, NewRequest, RequestKey,
    ShapeBound, TensorRef, TransferMode,
};
use ipc::{DiffusionSamplingParams, RequestId, SamplingParams, UnitId};

const WORLD_SIZE: usize = 2;
const QUEUE_DEPTH: usize = 2;
/// The stub model's image start token, the successor of 1001.
const STUB_IMAGE_START: u32 = 151_670;
/// The stub model's token whose forward on a CPU rank never returns
/// (`uniserve.models.stub.STUB_STALL_TOKEN_ID`).
const STUB_STALL: u32 = 151_669;
/// Held across `SchedulerHandle::start_all` in
/// `unsupported_media_is_rejected_without_stopping_the_engine` and across the
/// rank-group launch in `a_lost_worker_relaunches_its_group_and_an_incomplete_launch_fails`,
/// so those two launches never overlap. Other tests launch without taking it.
static CHILD_LAUNCH_ENV_LOCK: Mutex<()> = Mutex::new(());

/// A lifecycle-only batch: `starts` are admitted, then `commands` apply in
/// order. Its collective sequence equals its id.
fn lifecycle_batch(batch_id: u64, starts: Vec<NewRequest>, commands: Vec<BatchCommand>) -> Batch {
    let mut batch = Batch::new(batch_id, starts, Vec::new()).with_commands(commands);
    batch.collective_seq = batch_id;
    batch
}

fn finish(request_key: RequestKey) -> BatchCommand {
    BatchCommand::Finish {
        request_key,
        retained_buffers: Vec::new(),
    }
}

/// The engine's end of one rank's channel, driven by a test: it starts a stub
/// rank, answers its registration, sends requests and takes each response by
/// the identity its request carried.
struct RawEngine {
    channel: ipc::Channel,
    /// Responses read while waiting for another one.
    inbox: std::collections::HashMap<u64, ipc::WorkerResponse>,
}

impl RawEngine {
    /// Starts the stub worker as a raw process, outside any group, as a
    /// member of launch 1, and answers its registration with `descriptor`.
    ///
    /// The engine listens at `address`, so the rank reaches it over a Unix
    /// socket (`unix:<path>`) or over TCP (`127.0.0.1:0`).
    fn launch(
        address: &str,
        descriptor: &LaunchDescriptor,
    ) -> anyhow::Result<(Self, std::process::Child)> {
        Self::launch_with(address, descriptor, std::process::Stdio::inherit())
    }

    /// [`Self::launch`] with the rank's log output (its standard error) sent
    /// to `log`.
    fn launch_with(
        address: &str,
        descriptor: &LaunchDescriptor,
        log: std::process::Stdio,
    ) -> anyhow::Result<(Self, std::process::Child)> {
        let listener = ipc::Listener::bind(&[address])?;
        let mut child = std::process::Command::new(worker_executable())
            .arg("worker")
            .args(["--engine", &listener.addresses()[0]])
            .args(["--worker-id", STUB_WORKER])
            .args(["--rank", &descriptor.rank.to_string()])
            .args(["--generation", "1"])
            .stderr(log)
            .spawn()?;
        let registered = (|| -> anyhow::Result<Self> {
            // A worker that fails before registering never connects, so the
            // wait for its registration is bounded.
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            let channel = loop {
                if let Some(channel) = listener.try_accept()? {
                    break channel;
                }
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "no rank connected within 60 seconds"
                );
                thread::sleep(Duration::from_millis(5));
            };
            let mut engine = Self {
                channel,
                inbox: std::collections::HashMap::new(),
            };

            let registration: ipc::Registration = engine.receive(Duration::from_secs(60))?;
            anyhow::ensure!(
                registration
                    == ipc::Registration::Worker {
                        worker_id: STUB_WORKER.into(),
                        rank: descriptor.rank,
                        generation: 1,
                        version: ipc::VERSION.into(),
                    },
                "the rank registered as {registration:?}"
            );
            engine.channel.send(descriptor)?;
            Ok(engine)
        })();
        match registered {
            Ok(engine) => Ok((engine, child)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(error)
            }
        }
    }

    /// Waits up to `timeout` for the next message of type `M`.
    fn receive<M: serde::de::DeserializeOwned>(&mut self, timeout: Duration) -> anyhow::Result<M> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(message) = self.channel.try_recv()? {
                return Ok(message);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "no message arrived within {timeout:?}"
            );
            self.channel.wait(remaining)?;
        }
    }

    /// Sends `request` and returns the identity its response will carry.
    fn send(&mut self, request: &ipc::WorkerRequest) -> anyhow::Result<u64> {
        self.channel.send(request)?;
        request
            .message_id()
            .context("a request states its identity")
    }

    /// Waits up to `timeout` for the response to `message_id`; a zero timeout
    /// takes it only if it has already arrived.
    fn response(
        &mut self,
        message_id: u64,
        timeout: Duration,
    ) -> anyhow::Result<Option<ipc::WorkerResponse>> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            while let Some(response) = self.channel.try_recv::<ipc::WorkerResponse>()? {
                self.inbox
                    .insert(response.message_id().unwrap_or_default(), response);
            }
            if let Some(response) = self.inbox.remove(&message_id) {
                return Ok(Some(response));
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            self.channel.wait(remaining)?;
        }
    }
}

/// Waits up to `timeout` for `child` to exit.
fn wait_exit(
    child: &mut std::process::Child,
    timeout: Duration,
) -> anyhow::Result<std::process::ExitStatus> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "worker did not exit within {timeout:?}"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn native_close_drains_accepted_results_on_each_launch() -> anyhow::Result<()> {
    use ipc::{WorkerRequest, WorkerResponse};

    // A graceful exit is observable as the next launch registering and
    // serving again. The rank reaches the engine over loopback TCP, as a rank
    // on another host would.
    for _ in 0..2 {
        let (mut client, mut child) = RawEngine::launch("127.0.0.1:0", &stub_launch_descriptor())?;
        let result = (|| -> anyhow::Result<()> {
            let request = |mut request: WorkerRequest, message_id| {
                request.set_message_id(Some(message_id));
                request
            };

            // The rank describes itself once warmup is done.
            let info = client.send(&request(WorkerRequest::info(), 1))?;
            let info = client
                .response(info, Duration::from_secs(60))?
                .context("Info was not answered")?;
            let WorkerResponse::Info { info, .. } = info else {
                anyhow::bail!("Info failed: {info:?}");
            };
            info.validate()?;
            anyhow::ensure!(
                info.prefix_cache.is_some(),
                "the stub reports no cache pool"
            );

            let first = text_admission(51, 1, 1)?;
            let first_key = first.request_key;
            let started = lifecycle_batch(1, vec![first], Vec::new());
            let initial = client.send(&request(WorkerRequest::submit(started.clone()), 2))?;
            let first = client
                .response(initial, Duration::from_secs(30))?
                .context("initial submission did not complete")?;
            let WorkerResponse::Result { result: first, .. } = first else {
                anyhow::bail!("initial submission failed: {first:?}");
            };
            anyhow::ensure!(first.batch_id == 1, "result names another batch");
            anyhow::ensure!(
                first.completions.is_empty(),
                "a lifecycle batch completed calls"
            );

            // A second request starts while the first finishes. The rank's own
            // admission answers the two resubmissions of batch 1: a batch id
            // that does not exceed the last admitted one is refused with
            // `InvalidDescriptor`. Close is sent behind all of them.
            let other =
                lifecycle_batch(2, vec![text_admission(52, 1, 2)?], vec![finish(first_key)]);
            let accepted = client.send(&request(WorkerRequest::submit(other), 3))?;
            let rejected = client.send(&request(WorkerRequest::submit(started.clone()), 4))?;
            let duplicate = client.send(&request(WorkerRequest::submit(started), 5))?;
            let close = client.send(&request(WorkerRequest::close(), 6))?;
            let closed = client
                .response(close, Duration::from_secs(30))?
                .context("Close did not drain accepted work")?;
            anyhow::ensure!(
                closed
                    == WorkerResponse::Ok {
                        message_id: Some(6)
                    },
                "Close failed: {closed:?}"
            );

            // Receiving Close first must still leave every accepted response available.
            let mut receive = |pending| -> anyhow::Result<WorkerResponse> {
                client
                    .response(pending, Duration::ZERO)?
                    .context("Close acknowledged before accepted response delivery")
            };
            let WorkerResponse::Result { result: other, .. } = receive(accepted)? else {
                anyhow::bail!("independent submission did not return a result");
            };
            anyhow::ensure!(other.batch_id == 2, "independent batch identity changed");
            for pending in [rejected, duplicate] {
                let WorkerResponse::Error { error, .. } = receive(pending)? else {
                    anyhow::bail!("duplicate submission was accepted");
                };
                anyhow::ensure!(
                    error.code == Some(ErrorCode::InvalidDescriptor),
                    "duplicate must be a protocol error: {error:?}"
                );
            }

            // A worker that acknowledged Close exits on its own with success.
            let status = wait_exit(&mut child, Duration::from_secs(30))?;
            anyhow::ensure!(status.success(), "worker shutdown failed: {status}");
            Ok(())
        })();
        // Clean up the external process even when an assertion or transport call fails.
        if result.is_err() {
            let _ = child.kill();
            let _ = child.wait();
        }
        result?;
    }
    Ok(())
}

#[test]
fn a_serving_worker_leaves_interrupt_and_termination_to_its_engine() -> anyhow::Result<()> {
    use ipc::{WorkerRequest, WorkerResponse};

    let directory = tempfile::tempdir()?;
    let address = format!("unix:{}", directory.path().join("engine.sock").display());
    let (mut client, mut child) = RawEngine::launch(&address, &stub_launch_descriptor())?;
    let result = (|| -> anyhow::Result<()> {
        let request = |mut request: WorkerRequest, message_id| {
            request.set_message_id(Some(message_id));
            request
        };
        let mut serve = |batch_id: u64, request_id: u64, slot: u32| -> anyhow::Result<()> {
            let batch = lifecycle_batch(
                batch_id,
                vec![text_admission(request_id, 1, slot)?],
                Vec::new(),
            );
            let pending = client.send(&request(WorkerRequest::submit(batch), batch_id))?;
            let response = client
                .response(pending, Duration::from_secs(60))?
                .context("submission did not complete")?;
            anyhow::ensure!(
                matches!(response, WorkerResponse::Result { .. }),
                "submission failed: {response:?}"
            );
            Ok(())
        };

        // A result means the rank serves: warmup has ended.
        serve(1, 61, 1)?;

        // A terminal's Ctrl-C and a service manager's stop reach every process
        // of the job; the rank keeps serving until its engine closes it.
        let pid = libc::pid_t::try_from(child.id())?;
        for signal in [libc::SIGINT, libc::SIGTERM] {
            // SAFETY: kill takes only integer arguments and touches no memory of this process.
            anyhow::ensure!(unsafe { libc::kill(pid, signal) } == 0);
        }
        serve(2, 62, 2)?;

        let close = client.send(&request(WorkerRequest::close(), 3))?;
        let closed = client
            .response(close, Duration::from_secs(30))?
            .context("Close was not acknowledged")?;
        anyhow::ensure!(
            closed
                == WorkerResponse::Ok {
                    message_id: Some(3)
                },
            "Close failed: {closed:?}"
        );
        let status = wait_exit(&mut child, Duration::from_secs(30))?;
        anyhow::ensure!(status.success(), "worker shutdown failed: {status}");
        Ok(())
    })();
    // Clean up the external process even when an assertion or transport call fails.
    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}

/// Before it serves, a rank stops on SIGINT with status 130 and dies of
/// SIGTERM, so an operator or a failed launch can stop a starting rank.
#[test]
fn a_starting_worker_stops_on_interrupt_and_dies_of_termination() -> anyhow::Result<()> {
    for (signal, expect_interrupt) in [(libc::SIGINT, true), (libc::SIGTERM, false)] {
        // Rank 1 of a two-rank group whose rank 0 never starts: the rank
        // registers, then waits in its group's rendezvous for a peer that
        // never answers, so it stays in startup until the signal.
        let rendezvous = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut descriptor = stub_launch_descriptor();
        descriptor.rank = 1;
        descriptor.local_rank = 1;
        descriptor.world_size = 2;
        descriptor.acknowledgment_slot = 1;
        descriptor.host_slots = vec![0, 1];
        descriptor.components = WorkerGroupConfig::single_component("model", 2);
        descriptor.rendezvous_address = Some(rendezvous.local_addr()?.to_string());
        let directory = tempfile::tempdir()?;
        let address = format!("unix:{}", directory.path().join("engine.sock").display());
        let (_engine, mut child) = RawEngine::launch(&address, &descriptor)?;
        let result = (|| -> anyhow::Result<()> {
            // The rank connects to the rendezvous once it has described its
            // model; it is then blocked waiting for the address table.
            rendezvous.set_nonblocking(false)?;
            let (_peer, _) = rendezvous.accept()?;
            let pid = libc::pid_t::try_from(child.id())?;
            // SAFETY: kill takes only integer arguments and touches no memory of this process.
            anyhow::ensure!(unsafe { libc::kill(pid, signal) } == 0);
            let status = wait_exit(&mut child, Duration::from_secs(30))?;
            if expect_interrupt {
                anyhow::ensure!(
                    status.code() == Some(130),
                    "SIGINT ended the rank with {status}"
                );
            } else {
                anyhow::ensure!(
                    status.signal() == Some(libc::SIGTERM),
                    "SIGTERM ended the rank with {status}"
                );
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = child.kill();
            let _ = child.wait();
        }
        result?;
    }
    Ok(())
}

/// The engine's connection is a rank's liveness in every state. A rank that is
/// still starting, here one waiting in its group's rendezvous for a peer that
/// never answers, exits when the engine closes the connection, and so does a
/// rank that is serving.
#[test]
fn a_worker_exits_when_its_engine_closes_the_connection() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let address = |name: &str| format!("unix:{}", directory.path().join(name).display());
    let exits_with_failure = |mut child: std::process::Child| -> anyhow::Result<()> {
        let status = wait_exit(&mut child, Duration::from_secs(30)).inspect_err(|_| {
            let _ = child.kill();
            let _ = child.wait();
        })?;
        anyhow::ensure!(
            status.code() == Some(1),
            "the rank ended with {status} after its engine closed"
        );
        Ok(())
    };

    // Starting: rank 1 of a two-rank group whose rank 0 never starts.
    let rendezvous = std::net::TcpListener::bind("127.0.0.1:0")?;
    let mut descriptor = stub_launch_descriptor();
    descriptor.rank = 1;
    descriptor.local_rank = 1;
    descriptor.world_size = 2;
    descriptor.acknowledgment_slot = 1;
    descriptor.host_slots = vec![0, 1];
    descriptor.components = WorkerGroupConfig::single_component("model", 2);
    descriptor.rendezvous_address = Some(rendezvous.local_addr()?.to_string());
    let (engine, child) = RawEngine::launch(&address("starting.sock"), &descriptor)?;
    let (_peer, _) = rendezvous.accept()?;
    drop(engine);
    exits_with_failure(child)?;

    // Serving: a rank that has answered Info waits for its next request.
    let (mut engine, child) =
        RawEngine::launch(&address("serving.sock"), &stub_launch_descriptor())?;
    let mut info = ipc::WorkerRequest::info();
    info.set_message_id(Some(1));
    let info = engine.send(&info)?;
    let answered = engine.response(info, Duration::from_secs(60))?;
    anyhow::ensure!(
        matches!(answered, Some(ipc::WorkerResponse::Info { .. })),
        "the rank did not become ready: {answered:?}"
    );
    drop(engine);
    exits_with_failure(child)
}

/// Runs one stub rank under `descriptor` that is sent a prefill of `token`,
/// then loses its engine once the forward was answered, or, for the stub's
/// stalling token, once the stalled job was reported. Returns the rank's
/// exit status, which it must reach within 30 seconds, and its log.
fn lose_engine(
    directory: &Path,
    descriptor: &LaunchDescriptor,
    name: &str,
    token: u32,
) -> anyhow::Result<(std::process::ExitStatus, String)> {
    use ipc::{WorkerRequest, WorkerResponse};

    let address = format!("unix:{}", directory.join(format!("{name}.sock")).display());
    let log_path = directory.join(format!("{name}.log"));
    let log = std::fs::File::create(&log_path)?;
    let (mut engine, mut child) = RawEngine::launch_with(&address, descriptor, log.into())?;
    let served = (|| -> anyhow::Result<()> {
        let admission = text_admission(81, 1, 1)?;
        let mut request = WorkerRequest::submit(token_batch(
            1,
            1,
            admission.request_key,
            Some(admission),
            CallId::new(1, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[token],
            UnitId(1),
            0,
        ));
        request.set_message_id(Some(1));
        engine.send(&request)?;
        // A stalled job's report answers no request, so it carries no
        // identity.
        let stalling = token == STUB_STALL;
        let answered = engine
            .response(if stalling { 0 } else { 1 }, Duration::from_secs(60))?
            .context("the rank neither answered nor reported its job")?;
        anyhow::ensure!(
            matches!(
                (&answered, stalling),
                (WorkerResponse::Result { .. }, false) | (WorkerResponse::Stalled { .. }, true)
            ),
            "the rank answered with {answered:?}"
        );
        Ok(())
    })();
    drop(engine);
    let status = wait_exit(&mut child, Duration::from_secs(30));
    if status.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    served?;
    Ok((status?, std::fs::read_to_string(log_path)?))
}

/// A serving rank whose engine is lost ends by what it has in flight. With
/// nothing in flight, as once its last batch was answered, it closes without
/// its peers and destroys its device contexts; with a job stuck in flight,
/// here a forward that never returns, it ends at once from its control
/// thread, since its builder thread waits in that job. Either way it exits
/// promptly with status 1 and names the path in its log.
#[test]
fn a_rank_without_its_engine_closes_when_idle_and_ends_at_once_when_stuck() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let descriptor = LaunchDescriptor {
        watchdog_timeout_ms: 1_000,
        ..stub_launch_descriptor()
    };

    let (status, log) = lose_engine(directory.path(), &descriptor, "idle", 7)?;
    anyhow::ensure!(
        status.code() == Some(1),
        "the idle rank ended with {status}"
    );
    anyhow::ensure!(
        log.contains("the engine's channel failed with nothing in flight"),
        "the idle rank did not close orphaned:\n{log}"
    );

    let (status, log) = lose_engine(directory.path(), &descriptor, "stuck", STUB_STALL)?;
    anyhow::ensure!(
        status.code() == Some(1),
        "the stuck rank ended with {status}"
    );
    anyhow::ensure!(
        log.contains("the engine's channel failed with work in flight"),
        "the stuck rank did not abandon its work:\n{log}"
    );
    Ok(())
}

/// A two-rank group reports its capacities in its Info, retires finished
/// requests, refuses a repeated batch, and once closed reports nothing and
/// refuses work.
#[test]
fn a_worker_group_reports_its_capacities_and_refuses_work_after_close() -> anyhow::Result<()> {
    let mut executor = spawn_worker_group()?;
    let info = executor.info().clone();
    assert_eq!(info.max_batch_calls, 256);
    assert_eq!(info.max_batch_tokens, 256);
    assert!(info.prefix_cache.is_some());

    // Admit two requests, then finish one while admitting a third into the
    // retired slot's neighbour; each batch returns one empty result.
    let first = text_admission(11, 1, 1)?;
    let first_key = first.request_key;
    let admitted = execute(
        &mut executor,
        lifecycle_batch(1, vec![first, text_admission(12, 1, 2)?], Vec::new()),
    )?;
    assert_eq!(admitted.batch_id, 1);
    assert!(admitted.results.is_empty());
    let retire = lifecycle_batch(2, vec![text_admission(13, 1, 3)?], vec![finish(first_key)]);
    let retired = execute(&mut executor, retire.clone())?;
    assert_eq!(retired.batch_id, 2);
    assert!(retired.done);

    // The finished epoch's slot is free again for a new request.
    let reused = execute(
        &mut executor,
        lifecycle_batch(3, vec![text_admission(14, 1, 1)?], Vec::new()),
    )?;
    assert_eq!(reused.batch_id, 3);

    // A repeated batch is refused.
    assert!(matches!(
        executor.submit_placed(retire),
        Err(engine::BatchSubmitError::Failed(_))
    ));

    // A closed group reports no results, refuses submissions, and tolerates a
    // second close.
    executor.close()?;
    assert!(!executor.is_ready());
    assert!(executor.poll_batch(Duration::ZERO)?.is_none());
    assert!(matches!(
        executor.submit_placed(lifecycle_batch(100, Vec::new(), vec![finish(first_key)]),),
        Err(engine::BatchSubmitError::Failed(_))
    ));
    executor.close()?;
    Ok(())
}

/// Kills rank 1 of a two-rank group while a batch is outstanding, and requires
/// the group to report its resident requests lost, relaunch every rank under a
/// new incarnation, and serve again. Then requires a group whose rank 1 exits
/// during startup to fail its launch.
#[test]
fn a_lost_worker_relaunches_its_group_and_an_incomplete_launch_fails() -> anyhow::Result<()> {
    // The group's name is this test's own, so its rank 1 is found among the
    // daemon's children by its group and rank.
    let mut args = worker_group_args();
    args.worker_id = "relaunched".into();
    let mut executor = WorkerGroup::spawn(args)?;
    let victim = crate::support::group_processes("relaunched")
        .into_iter()
        .find(|(_, rank, _)| *rank == 1)
        .map(|(pid, _, _)| pid)
        .context("rank child was not found")?;

    // Batch 1 admits requests 21 and 20 together; batch 2 finishes request 20,
    // so only 21 and 22 (admitted by batch 3) are resident when rank 1 dies.
    let initial_endpoint = executor.info().endpoint.clone();
    let finished = text_admission(20, 1, 2)?;
    let finished_key = finished.request_key;
    execute(
        &mut executor,
        lifecycle_batch(1, vec![text_admission(21, 1, 1)?, finished], Vec::new()),
    )?;
    execute(
        &mut executor,
        lifecycle_batch(2, Vec::new(), vec![finish(finished_key)]),
    )?;

    // Keep this rank from completing the batch before the test terminates it.
    let paused = PausedProcess::new(victim.try_into()?)?;
    executor.submit_placed(lifecycle_batch(
        3,
        vec![text_admission(22, 1, 2)?],
        Vec::new(),
    ))?;
    paused.terminate()?;
    let loss = executor
        .poll_batch(Duration::from_secs(30))
        .expect_err("rank loss must be reported");
    let loss = loss
        .downcast_ref::<WorkerFailure>()
        .context("expected scoped WorkerGroup loss")?;
    assert_eq!(loss.worker_id.0, "relaunched");
    assert_eq!(loss.endpoints.len(), WORLD_SIZE);
    assert!(loss.endpoints.contains(&initial_endpoint));
    let requests = loss
        .requests
        .iter()
        .map(|request| request.request_id.0)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(requests, [21, 22].into_iter().collect());

    // Recovery relaunches the complete group, so rank 0's endpoint also comes
    // back with a new incarnation and address space.
    let ready_deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !executor.is_ready() {
        anyhow::ensure!(
            std::time::Instant::now() < ready_deadline,
            "replacement did not become ready"
        );
        executor.poll_batch(Duration::from_millis(100))?;
    }
    let recovered = execute(
        &mut executor,
        lifecycle_batch(4, vec![text_admission(23, 1, 1)?], Vec::new()),
    )?;
    assert_eq!(recovered.batch_id, 4);
    assert_ne!(
        executor.info().endpoint.incarnation,
        initial_endpoint.incarnation
    );
    assert_ne!(
        executor.info().endpoint.address_space,
        initial_endpoint.address_space
    );
    executor.close()?;

    // A wrapper that exits rank 1 before it registers, while rank 0 starts
    // normally and waits for it in the group's rendezvous.
    use std::os::unix::fs::PermissionsExt as _;
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let worker = serde_json::to_string(&worker_executable())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nif sys.argv[sys.argv.index('--rank') + 1] == '1':\n    sys.exit(47)\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut args = worker_group_args();
    args.executable = wrapper.clone();
    let startup = WorkerGroup::spawn(args);
    remove_file(wrapper)?;
    let Err(error) = startup else {
        anyhow::bail!("an incomplete rank group became ready");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("47"),
        "the failure did not name the rank's exit: {message}"
    );
    Ok(())
}

/// A child rank process: its pid, `--rank` and `--generation`.
type LaunchedRank = (u32, u32, u64);

/// The processes of group `worker_id` the daemon started once `launch`
/// returns: each with the value of its `--rank` and `--generation`
/// arguments.
fn launched_ranks(
    worker_id: &str,
    launch: impl FnOnce() -> anyhow::Result<WorkerGroup>,
) -> anyhow::Result<(WorkerGroup, Vec<LaunchedRank>)> {
    let group = launch()?;
    Ok((group, crate::support::group_processes(worker_id)))
}

/// A group that keeps a standby is replaced by promoting it: the lost
/// group's resident requests fail, the standby's processes serve the next
/// batch at once, and a new standby starts beside them.
#[test]
fn a_lost_worker_group_is_replaced_by_its_standby() -> anyhow::Result<()> {
    let mut args = worker_group_args();
    args.keep_standby = true;
    args.worker_id = "promoted".into();
    let (mut executor, ranks) = launched_ranks("promoted", || WorkerGroup::spawn(args))?;
    // The serving ranks are the group's first launch, the standby's its
    // second.
    let rank_of = |rank: u32, generation: u64| {
        ranks
            .iter()
            .find(|(_, r, g)| *r == rank && *g == generation)
            .map(|(pid, _, _)| *pid)
            .with_context(|| format!("rank {rank} of launch {generation} was not started"))
    };
    let victim = rank_of(1, 1)?;
    let standby = rank_of(0, 2)?;
    let ready_deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !executor.standby_ready() {
        anyhow::ensure!(
            std::time::Instant::now() < ready_deadline,
            "the standby did not park"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    execute(
        &mut executor,
        lifecycle_batch(1, vec![text_admission(31, 1, 1)?], Vec::new()),
    )?;
    let paused = PausedProcess::new(victim.try_into()?)?;
    executor.submit_placed(lifecycle_batch(
        2,
        vec![text_admission(32, 1, 2)?],
        Vec::new(),
    ))?;
    paused.terminate()?;
    let loss = executor
        .poll_batch(Duration::from_secs(30))
        .expect_err("rank loss must be reported");
    let loss = loss
        .downcast_ref::<WorkerFailure>()
        .context("expected scoped WorkerGroup loss")?;
    let requests = loss
        .requests
        .iter()
        .map(|request| request.request_id.0)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(requests, [31, 32].into_iter().collect());

    // The promoted standby serves without a relaunch.
    assert!(executor.is_ready());
    let served = execute(
        &mut executor,
        lifecycle_batch(3, vec![text_admission(33, 1, 1)?], Vec::new()),
    )?;
    assert_eq!(served.batch_id, 3);
    assert!(
        executor
            .info()
            .endpoint
            .address_space
            .starts_with(&format!("{standby}:")),
        "rank 0 serves from {} rather than the standby {standby}",
        executor.info().endpoint.address_space
    );
    let ready_deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !executor.standby_ready() {
        anyhow::ensure!(
            std::time::Instant::now() < ready_deadline,
            "no new standby parked after the promotion"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    executor.close()?;
    Ok(())
}

/// A rank whose startup fails reports its own error to the engine, which
/// names the rank and returns that error: here a checkpoint that does not
/// exist.
#[test]
fn a_failed_startup_reaches_the_engine_with_the_workers_error() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let missing = directory.path().join("no-such-checkpoint");
    let mut args = single_worker_args("failing", QUEUE_DEPTH);
    args.stub = false;
    args.model = missing.display().to_string();
    let Err(error) = WorkerGroup::spawn(args) else {
        anyhow::bail!("a rank with a missing checkpoint started");
    };
    let message = format!("{error:#}");
    anyhow::ensure!(
        message.contains("rank 0 failed during startup") && message.contains("no-such-checkpoint"),
        "the engine did not return the rank's error: {message}"
    );
    Ok(())
}

/// A rank whose builder thread stays inside one job reports the job's
/// forward to its engine, unasked, once the job has run for the launch's
/// stall bound, and goes on serving its channel.
#[test]
fn a_stalled_job_is_reported_while_its_worker_serves_on() -> anyhow::Result<()> {
    use ipc::{ErrorCallIdentity, WorkerRequest, WorkerResponse};

    let directory = tempfile::tempdir()?;
    let address = format!("unix:{}", directory.path().join("engine.sock").display());
    let descriptor = LaunchDescriptor {
        watchdog_timeout_ms: 1_000,
        ..stub_launch_descriptor()
    };
    let (mut engine, mut child) = RawEngine::launch(&address, &descriptor)?;
    let result = (|| -> anyhow::Result<()> {
        let request = |mut request: WorkerRequest, message_id| {
            request.set_message_id(Some(message_id));
            request
        };

        // The prefill's forward embeds the stub's stalling token, so the
        // job that runs it never completes.
        let admission = text_admission(71, 1, 1)?;
        let call_id = CallId::new(1, 0);
        let stalling = token_batch(
            1,
            1,
            admission.request_key,
            Some(admission.clone()),
            call_id,
            CallKind::Forward(ForwardMode::Prefill),
            &[STUB_STALL],
            UnitId(1),
            0,
        );
        engine.send(&request(WorkerRequest::submit(stalling), 1))?;

        // The report answers no request, so it carries no identity.
        let report = engine
            .response(0, Duration::from_secs(60))?
            .context("the stalled job was not reported")?;
        let WorkerResponse::Stalled { stall } = report else {
            anyhow::bail!("the rank answered with {report:?}");
        };
        anyhow::ensure!(stall.batch_id == Some(1), "{stall:?}");
        anyhow::ensure!(stall.route.as_deref() == Some("prefill"), "{stall:?}");
        anyhow::ensure!(stall.method.is_some(), "{stall:?}");
        anyhow::ensure!(
            stall.calls
                == [ErrorCallIdentity {
                    request_key: admission.request_key,
                    call_id,
                }],
            "{stall:?}"
        );
        anyhow::ensure!(stall.running_ms >= 500, "{stall:?}");

        // The control thread still serves the channel.
        let info = engine.send(&request(WorkerRequest::info(), 2))?;
        let answered = engine.response(info, Duration::from_secs(10))?;
        anyhow::ensure!(
            matches!(answered, Some(WorkerResponse::Info { .. })),
            "the stalled rank did not answer Info: {answered:?}"
        );
        Ok(())
    })();
    // The rank's builder thread never returns from the job, so the process
    // is killed.
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// A group whose rank reports a stalled job fails the batch's requests with
/// an error naming the stalled builder thread and the job's forward, before
/// its own watchdog timeout, and is replaced by a group that serves.
#[test]
fn a_reported_stall_replaces_the_group_naming_the_forward() -> anyhow::Result<()> {
    let mut args = single_worker_args("stalling", QUEUE_DEPTH);
    // The rank reports a job that has run for half of this, 4 seconds.
    args.watchdog_timeout = Duration::from_secs(8);
    let mut group = WorkerGroup::spawn(args)?;

    let admission = text_admission(31, 1, 1)?;
    let stalling = token_batch(
        1,
        1,
        admission.request_key,
        Some(admission),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[STUB_STALL],
        UnitId(1),
        0,
    );
    group.submit_placed(stalling)?;
    let failure = group
        .poll_batch(Duration::from_secs(30))
        .expect_err("a stalled job must fail its group");
    let failure = failure
        .downcast_ref::<WorkerFailure>()
        .context("expected the group's replacement")?;
    // The rank's report failed the group, not the group's own deadline,
    // whose failure names neither the stalled thread nor the forward.
    for named in ["the builder thread stalled", "batch 1 (prefill", "31/"] {
        anyhow::ensure!(
            failure.message.contains(named),
            "the failure does not name {named:?}: {}",
            failure.message
        );
    }
    let requests = failure
        .requests
        .iter()
        .map(|request| request.request_id.0)
        .collect::<Vec<_>>();
    assert_eq!(requests, [31]);

    // The replacement serves the next request.
    let ready_deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !group.is_ready() {
        anyhow::ensure!(
            std::time::Instant::now() < ready_deadline,
            "replacement did not become ready"
        );
        group.poll_batch(Duration::from_millis(100))?;
    }
    let admission = text_admission(32, 1, 1)?;
    let served = execute(
        &mut group,
        token_batch(
            2,
            2,
            admission.request_key,
            Some(admission),
            CallId::new(2, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[7, 8],
            UnitId(1),
            0,
        ),
    )?;
    let record = &served.results[0].output;
    assert_eq!(record.status, CallStatus::Ok);
    // The stub's head maps 8 to 1000.
    assert_eq!(record.committed_tokens, vec![1000]);
    group.close()?;
    Ok(())
}

#[test]
fn replacement_rejects_changed_capacity() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // Killing rank zero relaunches the group. Once replacement.json exists
    // the wrapper starts each relaunched rank as a `media_worker` process,
    // which answers Info with that file's worker info, so the replacement
    // describes a different capacity; the engine must refuse it and stop
    // using the group.
    type Change = fn(&mut ipc::WorkerInfo);
    let changes: [(&str, Change); 2] = [
        ("batch tokens", |info| info.max_batch_tokens = 128),
        ("cache capacity", |info| {
            if let Some(prefix_cache) = info.prefix_cache.as_mut() {
                prefix_cache.num_units /= 2;
            }
        }),
    ];
    let executable = serde_json::to_string(&std::env::current_exe()?)?;
    for (change, apply) in changes {
        let directory = tempfile::tempdir()?;
        let wrapper = directory.path().join("worker");
        let worker = serde_json::to_string(&worker_executable())?;
        let state = serde_json::to_string(&directory.path())?;
        std::fs::write(
            &wrapper,
            format!(
                "#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\nstate = Path({state})\nrank = sys.argv[sys.argv.index('--rank') + 1]\n(state / (rank + '.pid')).write_text(str(os.getpid()))\nchange = state / 'replacement.json'\nif change.exists():\n    env = dict(os.environ, {MEDIA_WORKER_ARGS}=json.dumps(sys.argv[1:]), {MEDIA_WORKER_INFO}=str(change), {MEDIA_WORKER_CASE}='none', {MEDIA_WORKER_NAME}=str(state / 'media-name'))\n    os.execve({executable}, [{executable}, '--exact', 'multiproc_ipc::media_worker', '--ignored', '--nocapture', '--test-threads', '1'], env)\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n"
            ),
        )?;
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
        let mut args = worker_group_args();
        args.executable = wrapper;
        let mut worker = WorkerGroup::spawn(args)?;

        let mut changed = worker.info().clone();
        apply(&mut changed);
        anyhow::ensure!(
            &changed != worker.info(),
            "{change} did not change the info"
        );
        std::fs::write(
            directory.path().join("replacement.json"),
            serde_json::to_vec(&changed)?,
        )?;
        let pid =
            std::fs::read_to_string(directory.path().join("0.pid"))?.parse::<libc::pid_t>()?;
        // SAFETY: kill takes only integer arguments and touches no memory of this process.
        anyhow::ensure!(unsafe { libc::kill(pid, libc::SIGKILL) } == 0);

        let loss = worker
            .poll_batch(Duration::from_secs(60))
            .expect_err("rank loss must be reported");
        assert!(
            loss.downcast_ref::<WorkerFailure>()
                .is_some_and(|loss| !loss.endpoints.is_empty()),
            "unexpected loss for {change}: {loss:#}",
        );
        // The group's failure domain relaunches it on a thread of its own;
        // installing ranks that describe another capacity closes the group
        // with a failure that says why.
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        let failure = loop {
            match worker.poll_batch(Duration::from_millis(100)) {
                Ok(_) => anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "the relaunch for {change} was not refused"
                ),
                Err(failure) => break failure,
            }
        };
        assert!(
            format!("{failure:#}").contains("worker info changed"),
            "unexpected failure for {change}: {failure:#}",
        );
        assert!(
            !worker.is_ready(),
            "incompatible replacement accepted: {change}"
        );
        worker.close()?;
    }
    Ok(())
}

#[test]
fn closing_a_group_lets_every_worker_finish_a_collective_teardown() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // Each wrapper rank serves the stub worker as a child process (handing
    // it the inherited rendezvous socket) and, once
    // that worker has exited after Close, waits for every rank of its group
    // to have exited too before it records `<rank>.closed` and exits. Like
    // the release of the communicators a group's ranks share, that teardown
    // completes on any rank only once every rank has been asked to close.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let worker = serde_json::to_string(&worker_executable())?;
    let records = serde_json::to_string(&directory.path())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport subprocess, sys, time\nfrom pathlib import Path\nrecords = Path({records})\nrank = sys.argv[sys.argv.index('--rank') + 1]\nstatus = subprocess.call([{worker}, *sys.argv[1:]], close_fds=False)\n(records / (rank + '.served')).write_text(str(status))\ndeadline = time.monotonic() + 60\nwhile len(list(records.glob('*.served'))) < {WORLD_SIZE}:\n    if time.monotonic() > deadline:\n        sys.exit(3)\n    time.sleep(0.05)\n(records / (rank + '.closed')).write_text('')\nsys.exit(status)\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut args = worker_group_args();
    args.executable = wrapper;
    let mut worker = WorkerGroup::spawn(args)?;

    // Close returns once every local rank process has exited, so each rank
    // has either finished its teardown or been stopped short of it.
    worker.close()?;
    for rank in 0..WORLD_SIZE {
        assert!(
            directory.path().join(format!("{rank}.closed")).exists(),
            "rank {rank} exited without finishing its teardown"
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join(format!("{rank}.served")))?,
            "0",
            "rank {rank}'s worker did not shut down cleanly"
        );
    }
    Ok(())
}

#[test]
fn a_worker_group_samples_scores_and_retires_token_calls() -> anyhow::Result<()> {
    let mut executor = spawn_worker_group()?;
    let info = executor.info();
    assert_eq!(info.max_batch_calls, 256);
    assert_eq!(info.max_batch_tokens, 256);

    let mut admission = text_admission(11, 1, 1)?;
    let sampling = &mut admission.generation.as_mut().unwrap().sampling;
    sampling.return_logprobs = true;
    sampling.n_logprobs = 1;
    sampling.return_prompt_logprobs = true;
    sampling.n_prompt_logprobs = 1;
    let mut initial = token_batch(
        1,
        1,
        admission.request_key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        UnitId(1),
        0,
    );
    // The bound the scheduler's `logprob_result_bytes` would derive: the
    // generated token and the one scored prompt position each take four bytes
    // plus twelve per ranked entry, and each has two entries (the scored token
    // and one requested candidate).
    initial.calls[0].bounds.max_completion_bytes = 4 + 2 * 12 + 4 + 2 * 12;
    let first = execute(&mut executor, initial.clone())?;
    let first_record = &first.results[0].output;
    assert_eq!(first_record.status, CallStatus::Ok);
    // The stub's head maps 8 to 1000.
    assert_eq!(first_record.committed_tokens, vec![1000]);
    assert_eq!(first_record.token_logprobs.len(), 1);
    let score = first_record.token_logprobs[0][0].logprob;
    assert!(score.is_finite() && score <= 0.0);
    assert_eq!(
        first_record.token_logprobs[0][0].token_id,
        first_record.committed_tokens[0]
    );
    // The first chunk scores its second token only.
    assert_eq!(first_record.prompt_logprobs.len(), 1);
    assert_eq!(first_record.prompt_logprobs[0][0].token_id, 8);
    assert_eq!(first_record.cache_visible_len, 2);

    // A repeated or conflicting batch id is refused before submission.
    assert!(executor.submit_placed(initial).is_err());

    // A Finish for another epoch of the same request id leaves the resident
    // epoch in place: the continuation below still runs.
    let stale = BatchCommand::Finish {
        request_key: RequestKey::new(
            admission.request_key.engine_id,
            admission.request_key.request_id,
            admission.request_key.request_epoch + 1,
        ),
        retained_buffers: Vec::new(),
    };
    assert!(
        execute(&mut executor, command_batch(2, stale))?
            .results
            .is_empty()
    );

    let mut continuation = token_batch(
        3,
        2,
        admission.request_key,
        None,
        CallId::new(3, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[first_record.committed_tokens[0]],
        UnitId(1),
        2,
    );
    continuation.calls[0].bounds.max_completion_bytes = 4 + 2 * 12;
    let continued = execute(&mut executor, continuation)?;
    let record = &continued.results[0].output;
    assert_eq!(record.status, CallStatus::Ok);
    assert_eq!(record.committed_tokens, vec![1001]);
    assert_eq!(record.cache_visible_len, 3);
    assert_eq!(record.token_logprobs.len(), 1);
    assert_eq!(record.token_logprobs[0][0].token_id, 1001);
    assert!(record.prompt_logprobs.is_empty());

    // Retire the request in a batch that also admits and runs an
    // independent one; a later call on the retired request fails as an
    // invalid call.
    let independent = text_admission(12, 1, 2)?;
    let mut close_batch = token_batch(
        6,
        3,
        independent.request_key,
        Some(independent),
        CallId::new(6, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9, 10],
        UnitId(2),
        0,
    );
    close_batch.commands.push(BatchCommand::Finish {
        request_key: admission.request_key,
        retained_buffers: Vec::new(),
    });
    let result = execute(&mut executor, close_batch)?;
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].output.status, CallStatus::Ok);
    assert_eq!(result.results[0].output.committed_tokens, vec![1000]);
    let descendant = token_batch(
        8,
        4,
        admission.request_key,
        None,
        CallId::new(8, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[first_record.committed_tokens[0]],
        UnitId(1),
        2,
    );
    let closed = execute(&mut executor, descendant)?;
    let closed_record = &closed.results[0].output;
    assert_eq!(closed_record.status, CallStatus::Error);
    assert_eq!(closed_record.error_code, Some(ErrorCode::InvalidDescriptor));
    executor.close()?;
    Ok(())
}

/// The first token a fresh request samples after the prompt `[7, 8]` at
/// temperature 100 over the whole vocabulary: the stub's ±16 logits become
/// nearly uniform, so the token is a function of the request's draw.
fn sampled_first_token(
    executor: &mut WorkerGroup,
    batch_id: u64,
    request_id: u64,
    slot: u32,
    seed: Option<u64>,
) -> anyhow::Result<u32> {
    let mut admission = text_admission(request_id, 1, slot)?;
    let sampling = &mut admission.generation.as_mut().unwrap().sampling;
    sampling.temperature = 100.0;
    sampling.seed = seed;
    let batch = token_batch(
        batch_id,
        batch_id,
        admission.request_key,
        Some(admission),
        CallId::new(batch_id, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        UnitId(slot),
        0,
    );
    let result = execute(executor, batch)?;
    let record = &result.results[0].output;
    assert_eq!(record.status, CallStatus::Ok);
    Ok(record.committed_tokens[0])
}

#[test]
fn a_seeded_request_repeats_under_another_identity() -> anyhow::Result<()> {
    let mut executor = spawn_worker_group()?;

    // Two requests with one seed draw the same token; another seed, and
    // unseeded requests, draw their own.
    let first = sampled_first_token(&mut executor, 1, 21, 1, Some(17))?;
    let again = sampled_first_token(&mut executor, 2, 22, 2, Some(17))?;
    let other = sampled_first_token(&mut executor, 3, 23, 3, Some(18))?;
    let unseeded = sampled_first_token(&mut executor, 4, 24, 4, None)?;
    let unseeded_other = sampled_first_token(&mut executor, 5, 25, 5, None)?;

    assert_eq!(first, again);
    assert_ne!(first, other);
    assert_ne!(unseeded, unseeded_other);
    executor.close()?;
    Ok(())
}

/// The five top logprobs a fresh request reports for its first token after
/// the prompt `[7, 8]`, selected greedily under the nucleus `top_p`.
fn first_token_logprobs(
    executor: &mut WorkerGroup,
    batch_id: u64,
    request_id: u64,
    slot: u32,
    top_p: f32,
) -> anyhow::Result<Vec<ipc::TokenLogprob>> {
    let mut admission = text_admission(request_id, 1, slot)?;
    let sampling = &mut admission.generation.as_mut().unwrap().sampling;
    sampling.top_p = top_p;
    sampling.return_logprobs = true;
    sampling.n_logprobs = 5;
    let mut batch = token_batch(
        batch_id,
        batch_id,
        admission.request_key,
        Some(admission),
        CallId::new(batch_id, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        UnitId(slot),
        0,
    );
    batch.calls[0].bounds.max_completion_bytes = 4 + 6 * 12;
    let result = execute(executor, batch)?;
    let record = &result.results[0].output;
    assert_eq!(record.status, CallStatus::Ok);
    // The stub's head maps 8 to 1000.
    assert_eq!(record.committed_tokens, vec![1000]);
    Ok(record.token_logprobs[0].clone())
}

#[test]
fn logprobs_report_the_model_distribution_under_a_cutoff() -> anyhow::Result<()> {
    let mut executor = spawn_worker_group()?;

    // The stub's successor holds nearly all mass, so a nucleus of one half
    // keeps it alone; the five reported tokens are still the model's five
    // most likely, with the model's log probabilities.
    let cut = first_token_logprobs(&mut executor, 1, 31, 1, 0.5)?;
    let whole = first_token_logprobs(&mut executor, 2, 32, 2, 1.0)?;

    assert_eq!(cut.len(), 5);
    assert_eq!(cut[0].token_id, 1000);
    assert!(cut.iter().all(|entry| entry.logprob.is_finite()));
    assert_eq!(cut, whole);
    executor.close()?;
    Ok(())
}

#[test]
fn same_batch_successor_consumes_the_unobserved_device_token() -> anyhow::Result<()> {
    use engine::{ExecutionBatch, RequestPlacement, WorkerExecutor, WorkerGroupId};

    // Four calls of one request travel as four batches, all submitted before
    // any result is read, so each successor starts from its predecessor's
    // device product rather than from a host observation.
    let transfer = engine::TransferConfig::default();
    let mut args = worker_group_args();
    args.queue_depth = 4;
    let mut executor = WorkerExecutor::try_new(
        vec![(WorkerGroupId("worker".into()), WorkerGroup::spawn(args)?)],
        transfer,
    )?;
    let admission = text_admission(63, 1, 1)?;
    let key = admission.request_key;
    let first_id = CallId::new(9, 0);
    let second_id = CallId::new(10, 0);
    let verify_id = CallId::new(11, 0);
    let resumed_id = CallId::new(12, 0);
    let first = token_batch(
        9,
        1,
        key,
        Some(admission),
        first_id,
        CallKind::Forward(ForwardMode::Prefill),
        &[7],
        UnitId(1),
        0,
    );
    let mut successor = token_batch(
        10,
        2,
        key,
        None,
        second_id,
        CallKind::Forward(ForwardMode::Decode),
        &[0],
        UnitId(1),
        1,
    );
    // Reserve the next row's shape, but obtain its token only from the device
    // relay. No host result is read between the two logical call kinds.
    successor.calls[0].input_token_ids.clear();
    successor.calls[0].predicate = Some(first.calls[0].token_output.clone().unwrap());
    let mut verifier = token_batch(
        11,
        3,
        key,
        None,
        verify_id,
        CallKind::Forward(ForwardMode::Verify),
        &[0, 900, 901],
        UnitId(1),
        2,
    );
    verifier.calls[0].input_token_ids = vec![900, 901];
    verifier.calls[0].predicate = Some(successor.calls[0].token_output.clone().unwrap());
    // Five initialized cache positions are the verifier's maximum. Its first
    // rejected draft must leave only three positions visible to the next
    // decode, whose forward row attends to those three and its query.
    let mut resumed = token_batch(
        12,
        4,
        key,
        None,
        resumed_id,
        CallKind::Forward(ForwardMode::Decode),
        &[0],
        UnitId(1),
        3,
    );
    resumed.calls[0].input_token_ids.clear();
    resumed.calls[0].predicate = Some(verifier.calls[0].token_output.clone().unwrap());
    resumed.calls[0].coordinates.cache_computed_len = 5;
    for mut batch in [first, successor, verifier, resumed] {
        let batch_id = batch.batch_id;
        let commands = std::mem::take(&mut batch.commands);
        let call = batch.calls.remove(0);
        let placement = RequestPlacement {
            worker: WorkerGroupId("worker".into()),
            request_pool_idx: None,
            block_tables: batch.block_tables,
            new_cache_units: batch.new_cache_units,
            forward: batch.forward,
            latent: None,
            decode: None,
            buffers: on_every_rank(batch.buffer_allocations),
            readers: Default::default(),
            page_loads: Vec::new(),
        };
        executor.submit(ExecutionBatch::new(
            batch_id,
            vec![(call, placement)],
            commands,
            vec![],
        ))?;
    }
    let mut outputs = std::collections::BTreeMap::new();
    while outputs.len() < 4 {
        let result = poll_logical(&mut executor)?.context("device successor did not complete")?;
        for result in result.results {
            assert_eq!(result.output.status, CallStatus::Ok);
            assert!(
                outputs
                    .insert(result.output.call_id, result.output)
                    .is_none()
            );
        }
    }
    // The stub model's head maps 7 to 1000, 1000 to 1001, 1001 to its
    // image-start token 151_670, and that token to 1002. The verifier's first
    // draft (900) therefore disagrees with the target and only the target's
    // own token commits, leaving the prompt token, 1000 and 1001 visible.
    assert_eq!(
        outputs.keys().copied().collect::<Vec<_>>(),
        vec![first_id, second_id, verify_id, resumed_id]
    );
    assert_eq!(outputs[&first_id].committed_tokens, vec![1000]);
    assert_eq!(outputs[&second_id].committed_tokens, vec![1001]);
    assert_eq!(outputs[&second_id].cache_visible_len, 2);
    assert_eq!(outputs[&verify_id].committed_tokens, vec![151_670]);
    assert_eq!(outputs[&verify_id].cache_visible_len, 3);
    assert_eq!(outputs[&verify_id].cache_computed_len, 5);
    assert_eq!(outputs[&resumed_id].committed_tokens, vec![1002]);
    assert_eq!(outputs[&resumed_id].cache_visible_len, 4);
    executor.close()?;
    Ok(())
}

/// Waits up to 30 seconds in total for the executor's next partial or
/// terminal batch result, polling again after progress-only wakes.
fn poll_logical(executor: &mut impl Executor) -> anyhow::Result<Option<engine::BatchResult>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
            return Ok(None);
        };
        if let Some(result) = executor.poll(remaining)? {
            return Ok(Some(result));
        }
    }
}

/// Builds a batch with one token call of kind `mode` on the `model` component.
///
/// The call reads `tokens` after `prefix_length` cached positions and writes
/// its sampled token to an `I64` token output produced by `call_id`, with
/// generation `4 * call_id.batch_id + 1` (saturating). `admission`, when
/// given, becomes the batch's `Start` command and supplies the request pool
/// slot; without it the call targets slot 1. The block table maps the request
/// to the single page `unit`, which the batch allocates only when
/// `prefix_length` is zero; a continuation passes the same page and a nonzero
/// prefix.
#[expect(
    clippy::too_many_arguments,
    reason = "every field of the one-call batch a test sends"
)]
fn token_batch(
    batch_id: u64,
    collective_seq: u64,
    request_key: RequestKey,
    admission: Option<NewRequest>,
    call_id: CallId,
    mode: CallKind,
    tokens: &[u32],
    unit: UnitId,
    prefix_length: u32,
) -> Batch {
    let request_pool_idx = admission.as_ref().map_or(1, |value| value.request_pool_idx);
    let token_output = TensorRef {
        request_key,
        producer_call_id: call_id,
        output_index: 0,
        generation: (call_id.batch_id as u32)
            .saturating_mul(4)
            .saturating_add(1),
        dtype: DType::I64,
        shape_bound: ShapeBound::default(),
    };
    let mut call = Call::new(request_key, call_id, mode);
    // These requests hold no cached prefix beyond their own tokens, so the
    // logical sequence and the initialized cache advance together.
    call.coordinates = CallCoordinates {
        logical_position: prefix_length,
        cache_visible_len: prefix_length,
        cache_computed_len: prefix_length,
    };
    call.token_output = Some(token_output);
    call.input_token_ids = tokens.to_vec();
    call.component = "model".into();
    call.bounds = Bounds {
        max_tokens: tokens.len().max(1) as u32,
        ..Bounds::default()
    };
    let input_length = tokens.len() as u32;
    let mut batch = Batch::new(batch_id, admission.into_iter().collect(), vec![call]);
    batch.collective_seq = collective_seq;
    batch.block_tables = vec![BlockTable {
        request_pool_idx,
        group_id: 0,
        start_page: 0,
        unit_ids: vec![unit],
        allocated_tokens: prefix_length + input_length,
    }];
    batch.new_cache_units = if prefix_length == 0 {
        vec![CacheUnitAllocation {
            request_pool_idx,
            group_id: 0,
            unit_ids: vec![unit],
        }]
    } else {
        Vec::new()
    };
    batch.forward = ForwardBatch {
        call_indices: vec![0],
        request_pool_indices: vec![request_pool_idx],
        seq_lens: vec![prefix_length + input_length.max(1)],
        query_lens: vec![input_length.max(1)],
        write_cache: vec![true],
    };
    batch
}

fn command_batch(batch_id: u64, command: BatchCommand) -> Batch {
    Batch::new(batch_id, Vec::new(), Vec::new()).with_commands(vec![command])
}

/// Encodes one registration frame per rank of `args`'s group, as `stated`
/// gives it for each rank, in the hexadecimal form a wrapper script embeds.
fn registration_frames(
    args: &WorkerProcessArgs,
    stated: impl Fn(u32) -> ipc::Registration,
) -> anyhow::Result<Vec<String>> {
    (0..args.ranks.len())
        .map(|rank| {
            let frame = ipc::Frame::encode(&stated(rank as u32))?;
            Ok(frame
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>())
        })
        .collect()
}

/// The engine and its ranks must be one build, since the message types are the
/// only definition of their protocol and their encoding is positional. A rank
/// states its build when it registers, and a launch fails by naming a rank of
/// another one, including another build of the same package version.
#[test]
fn launch_refuses_a_worker_of_another_build() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // The stand-in registers as the rank it was started as, for the launch
    // it was started for, under a build of the engine's package version that
    // the engine is not, and then waits for the engine to close.
    let other = format!("{}+0123456789abcdef", env!("CARGO_PKG_VERSION"));
    anyhow::ensure!(
        other != ipc::VERSION,
        "the stand-in's build is the engine's"
    );
    let mut args = worker_group_args();
    let frames = registration_frames(&args, |rank| ipc::Registration::Worker {
        worker_id: args.worker_id.clone(),
        rank,
        generation: 1,
        version: other.clone(),
    })?;
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport socket, sys\nframes = {frames:?}\nrank = int(sys.argv[sys.argv.index('--rank') + 1])\nengine = sys.argv[sys.argv.index('--engine') + 1]\nconnection = socket.socket(socket.AF_UNIX)\nconnection.connect(engine.removeprefix('unix:'))\nconnection.sendall(bytes.fromhex(frames[rank]))\nconnection.settimeout(60)\nwhile connection.recv(4096):\n    pass\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;

    args.executable = wrapper;
    let reported = match WorkerGroup::spawn(args) {
        Ok(_) => anyhow::bail!("a rank of another build was accepted"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        reported.contains(&other) && reported.contains(ipc::VERSION),
        "launch failure did not name both builds: {reported}"
    );
    Ok(())
}

/// A group relaunched after it lost a rank takes only ranks of the relaunch.
/// A rank that registers for the launch the group lost is refused: the engine
/// closes its connection, the relaunch completes and the group serves again.
#[test]
fn a_relaunch_refuses_a_worker_of_the_launch_it_replaced() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // Each wrapper rank records its process id. When it is started for the
    // relaunch (launch 2), it first registers as the same rank of launch 1
    // on a connection of its own, waits until the engine closes that
    // connection, records `<rank>.refused`, and then runs the stub worker,
    // which registers for the relaunch. Had the engine taken the stale
    // registration, it would have answered it instead of closing, and the
    // relaunch would have failed on a rank registered twice.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let worker = serde_json::to_string(&worker_executable())?;
    let records = serde_json::to_string(&directory.path())?;
    let mut args = worker_group_args();
    let frames = registration_frames(&args, |rank| ipc::Registration::Worker {
        worker_id: args.worker_id.clone(),
        rank,
        generation: 1,
        version: ipc::VERSION.into(),
    })?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, socket, sys\nfrom pathlib import Path\nrecords = Path({records})\nframes = {frames:?}\nargument = lambda name: sys.argv[sys.argv.index(name) + 1]\nrank = argument('--rank')\n(records / (rank + '.pid')).write_text(str(os.getpid()))\nif argument('--generation') == '2':\n    stale = socket.socket(socket.AF_UNIX)\n    stale.connect(argument('--engine').removeprefix('unix:'))\n    stale.sendall(bytes.fromhex(frames[int(rank)]))\n    stale.settimeout(60)\n    if stale.recv(4096) == b'':\n        (records / (rank + '.refused')).write_text('')\n    stale.close()\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    args.executable = wrapper;
    let mut worker = WorkerGroup::spawn(args)?;
    let launched = worker.info().endpoint.incarnation.clone();

    // Losing rank 0 relaunches the group.
    let pid = std::fs::read_to_string(directory.path().join("0.pid"))?.parse::<libc::pid_t>()?;
    // SAFETY: kill takes only integer arguments and touches no memory of this process.
    anyhow::ensure!(unsafe { libc::kill(pid, libc::SIGKILL) } == 0);
    let failure = worker
        .poll_batch(Duration::from_secs(60))
        .expect_err("rank loss must be reported");
    // The relaunch runs on the failure domain's thread; a poll installs its
    // ranks once they are ready.
    let ready_deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !worker.is_ready() {
        anyhow::ensure!(
            std::time::Instant::now() < ready_deadline,
            "the relaunch did not recover the group: {failure:#}"
        );
        worker.poll_batch(Duration::from_millis(100))?;
    }
    assert_ne!(worker.info().endpoint.incarnation, launched);
    for rank in 0..WORLD_SIZE {
        assert!(
            directory.path().join(format!("{rank}.refused")).exists(),
            "rank {rank}'s registration for the replaced launch was not refused"
        );
    }
    let served = execute(
        &mut worker,
        lifecycle_batch(1, vec![text_admission(31, 1, 1)?], Vec::new()),
    )?;
    assert_eq!(served.batch_id, 1);
    worker.close()?;
    Ok(())
}

/// A group's relaunch runs on its failure domain's thread: while the
/// relaunched ranks start, the executor serves its other groups, and the
/// relaunched group serves again once its ranks are installed.
#[test]
fn a_relaunching_group_leaves_its_executor_serving_the_others() -> anyhow::Result<()> {
    use engine::{WorkerExecutor, WorkerGroupId};
    use std::os::unix::fs::PermissionsExt as _;

    // The wrapper records each launch's process id. A relaunched rank
    // (launch 2) waits for `release` before it starts, which holds the
    // relaunch in its ranks' registration until the test lets it go.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let worker = serde_json::to_string(&worker_executable())?;
    let state = serde_json::to_string(directory.path())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys, time\nfrom pathlib import Path\nstate = Path({state})\ngeneration = sys.argv[sys.argv.index('--generation') + 1]\n(state / (generation + '.pid')).write_text(str(os.getpid()))\nwhile generation == '2' and not (state / 'release').exists():\n    time.sleep(0.05)\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let relaunched = WorkerGroupId("relaunched".into());
    let serving = WorkerGroupId("serving".into());
    let spawn = |worker_id: &WorkerGroupId| -> anyhow::Result<WorkerGroup> {
        let mut args = single_worker_args(&worker_id.0, 1);
        if *worker_id == relaunched {
            args.executable = wrapper.clone();
        }
        WorkerGroup::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        vec![
            (relaunched.clone(), spawn(&relaunched)?),
            (serving.clone(), spawn(&serving)?),
        ],
        engine::TransferConfig::default(),
    )?;
    let prefill = |batch_id, request_id| -> anyhow::Result<Batch> {
        let admission = text_admission(request_id, 1, 1)?;
        Ok(token_batch(
            batch_id,
            batch_id,
            admission.request_key,
            Some(admission),
            CallId::new(batch_id, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[7],
            UnitId(1),
            0,
        ))
    };

    let pid = std::fs::read_to_string(directory.path().join("1.pid"))?.parse::<libc::pid_t>()?;
    // SAFETY: kill takes only integer arguments and touches no memory of this process.
    anyhow::ensure!(unsafe { libc::kill(pid, libc::SIGKILL) } == 0);
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let loss = loop {
        match executor.poll(Duration::from_millis(100)) {
            Ok(None) => anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "the loss was not reported"
            ),
            Ok(Some(result)) => anyhow::bail!("an idle group returned {result:?}"),
            Err(loss) => break loss,
        }
    };
    let loss = loss
        .downcast_ref::<WorkerFailure>()
        .context("expected the group's loss")?;
    assert_eq!(loss.worker_id, relaunched);

    // The relaunch waits for its held rank while the other group serves.
    executor.submit(place(prefill(1, 61)?, &serving.0, Vec::new()))?;
    let served = poll_logical(&mut executor)?.context("the other group did not serve")?;
    assert_eq!(served.batch_id, 1);
    assert_eq!(served.results[0].output.status, CallStatus::Ok);
    assert!(
        !executor.has_capacity(&relaunched),
        "the relaunching group accepts work"
    );

    std::fs::write(directory.path().join("release"), "")?;
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !executor.has_capacity(&relaunched) {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "the relaunched group did not become ready"
        );
        executor.poll(Duration::from_millis(100))?;
    }
    executor.submit(place(prefill(2, 62)?, &relaunched.0, Vec::new()))?;
    let recovered = poll_logical(&mut executor)?.context("the relaunched group did not serve")?;
    assert_eq!(recovered.batch_id, 2);
    assert_eq!(recovered.results[0].output.status, CallStatus::Ok);
    executor.close()?;
    Ok(())
}

#[test]
fn losing_every_worker_stops_request_admission() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let worker = serde_json::to_string(&worker_executable())?;
    let records = serde_json::to_string(directory.path())?;
    // Record process identities at the launch boundary; the numerical
    // worker and the engine both execute their normal public paths.
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nfrom pathlib import Path\nrank = sys.argv[sys.argv.index('--rank') + 1]\n(Path({records}) / (rank + '.pid')).write_text(str(os.getpid()))\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut config = EngineConfig::sim("stub");
    config.generation_limits = engine::GenerationLimits {
        features: engine::GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerGroupConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = worker_group_args();
    config.worker_process.executable = wrapper.clone();
    let engine = SchedulerHandle::start_all(config)?.remove(0);

    // Removing the launch executable makes recovery fail at the same OS
    // boundary as a missing deployment prerequisite, without changing IPC.
    remove_file(wrapper)?;
    let pid = std::fs::read_to_string(directory.path().join("0.pid"))?.parse::<libc::pid_t>()?;
    // SAFETY: kill takes only integer arguments and touches no memory of this process.
    anyhow::ensure!(unsafe { libc::kill(pid, libc::SIGKILL) } == 0);
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while !engine.is_dead() && std::time::Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    let dead = engine.is_dead();
    engine.shutdown();
    assert!(
        dead,
        "an engine without execution workers kept accepting work"
    );
    Ok(())
}

/// An engine whose only group is lost stays alive while the group's failure
/// domain relaunches it: a request submitted meanwhile waits for the
/// relaunched ranks and completes on them.
#[test]
fn an_engine_serves_once_its_only_group_is_relaunched() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // The wrapper records each launch's process ids by generation and rank.
    // A relaunched rank (launch 2) waits for `release` before it starts.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let worker = serde_json::to_string(&worker_executable())?;
    let state = serde_json::to_string(directory.path())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys, time\nfrom pathlib import Path\nstate = Path({state})\nargument = lambda name: sys.argv[sys.argv.index(name) + 1]\ngeneration = argument('--generation')\n(state / (generation + '-' + argument('--rank') + '.pid')).write_text(str(os.getpid()))\nwhile generation == '2' and not (state / 'release').exists():\n    time.sleep(0.05)\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut config = EngineConfig::sim("stub");
    config.generation_limits = engine::GenerationLimits {
        features: engine::GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerGroupConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = worker_group_args();
    config.worker_process.executable = wrapper;
    let engine = SchedulerHandle::start_all(config)?.remove(0);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;

    let result = (|| -> anyhow::Result<()> {
        let pid =
            std::fs::read_to_string(directory.path().join("1-0.pid"))?.parse::<libc::pid_t>()?;
        // SAFETY: kill takes only integer arguments and touches no memory of this process.
        anyhow::ensure!(unsafe { libc::kill(pid, libc::SIGKILL) } == 0);
        // The relaunch has started once its first rank recorded itself.
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !directory.path().join("2-0.pid").exists() {
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "the group was not relaunched"
            );
            thread::sleep(Duration::from_millis(20));
        }

        let mut events = engine.submit(Request::from(engine::GenerationRequest {
            request_id: RequestId(81),
            prompt_token_ids: vec![7, 8],
            negative_prompt_token_ids: Vec::new(),
            multimodal_inputs: Default::default(),
            constraint: engine::GenerationConstraint::Default,
            sampling: SamplingParams::default(),
            image: Default::default(),
            max_images: 1,
            max_und_tokens: 2,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            include_stop_token: false,
            image_generation: Default::default(),
            readout: Vec::new(),
            generated_token_ids: Vec::new(),
            adapter: 0,
        }))?;
        std::fs::write(directory.path().join("release"), "")?;
        runtime.block_on(async {
            loop {
                let event = tokio::time::timeout(Duration::from_secs(60), events.recv())
                    .await
                    .context("the request did not finish after the relaunch")?
                    .context("the request's events ended without a terminal event")?;
                match event {
                    Event::Finished {
                        completion_tokens, ..
                    } => {
                        anyhow::ensure!(completion_tokens == 2, "{completion_tokens} tokens");
                        return Ok(());
                    }
                    Event::Rejected { message, .. } | Event::Error { message } => {
                        anyhow::bail!("the request failed: {message}")
                    }
                    _ => {}
                }
            }
        })
    })();

    // The dead latch is set only when the scheduler thread exits with a fatal
    // status, so `shutdown` joins that thread before `is_dead` is read.
    engine.shutdown();
    result?;
    assert!(!engine.is_dead(), "the relaunch killed the engine");
    Ok(())
}

/// The engine configuration of a two-rank stub deployment named
/// `deployment`.
fn stub_engine_config(deployment: &str) -> EngineConfig {
    let mut config = EngineConfig::sim("stub");
    config.generation_limits = engine::GenerationLimits {
        features: engine::GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerGroupConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = worker_group_args();
    config.worker_process.deployment = deployment.to_owned();
    config
}

/// A greedy text request of the prompt `[7, 8, 9]` that generated
/// `generated` before it migrated and generates `max_und_tokens` tokens in
/// all: it ignores EOS, which the stub model reaches early.
fn greedy_text(id: u64, generated: Vec<u32>, max_und_tokens: usize) -> Request {
    Request::from(engine::GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![7, 8, 9],
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: Default::default(),
        constraint: engine::GenerationConstraint::UndOnly,
        sampling: SamplingParams {
            ignore_eos: true,
            ..Default::default()
        },
        image: Default::default(),
        max_images: 1,
        max_und_tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        image_generation: Default::default(),
        readout: Vec::new(),
        generated_token_ids: generated,
        adapter: 0,
    })
}

/// The tokens `events` delivers until its terminal event, and that event.
fn tokens_until_terminal(events: &mut engine::EventRx) -> anyhow::Result<(Vec<u32>, Event)> {
    let mut tokens = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "the request did not end"
        );
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => tokens.push(id),
            Ok(event) if event.is_terminal() => return Ok((tokens, event)),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(5)),
        }
    }
}

/// A stub model that also counts each request's tokens in a cache layer
/// kept per request slot (`uniserve.models.stub.TokenCount`, the test
/// implementation of such a cache layer type), whose forward fails when a
/// row continues from another count than its prefix. A prompt of three
/// chunks and its decode steps carry the count in the request's slot, and
/// each chunk checkpoints it at its last page boundary; a second request of
/// the same prompt reuses the prefix up to the last chunk's checkpoint,
/// resumes its count from it in another slot, and generates the same
/// tokens.
#[test]
fn a_cache_layer_kept_per_request_resumes_from_its_checkpoints() -> anyhow::Result<()> {
    let model = tempfile::tempdir()?;
    std::fs::write(model.path().join("stub.json"), r#"{"request_state": true}"#)?;
    let mut config = stub_engine_config("request-states");
    config.worker_process.model = model.path().to_string_lossy().into_owned();
    let engine = SchedulerHandle::start_all(config)?.remove(0);
    let stats = std::sync::Arc::clone(engine.stats());
    // 600 tokens: three chunks of the ranks' 256-token batches.
    let prompt: Vec<u32> = (0..600u32).map(|index| index % 47 + 5).collect();
    let result = (|| -> anyhow::Result<Vec<Vec<u32>>> {
        let mut generated = Vec::new();
        for id in [91, 92] {
            let mut events = engine.submit(Request::from(engine::GenerationRequest {
                request_id: RequestId(id),
                prompt_token_ids: prompt.clone(),
                negative_prompt_token_ids: Vec::new(),
                multimodal_inputs: Default::default(),
                constraint: engine::GenerationConstraint::UndOnly,
                sampling: SamplingParams {
                    ignore_eos: true,
                    ..Default::default()
                },
                image: Default::default(),
                max_images: 1,
                max_und_tokens: 8,
                stop_strings: Vec::new(),
                stop_token_ids: Vec::new(),
                priority: 0,
                cache: Default::default(),
                include_stop_token: false,
                image_generation: Default::default(),
                readout: Vec::new(),
                generated_token_ids: Vec::new(),
                adapter: 0,
            }))?;
            let (tokens, terminal) = tokens_until_terminal(&mut events)?;
            anyhow::ensure!(
                matches!(terminal, Event::Finished { .. }),
                "request {id} ended with {terminal:?}"
            );
            generated.push(tokens);
        }
        Ok(generated)
    })();
    engine.shutdown();
    let generated = result?;
    anyhow::ensure!(
        generated[0].len() == 8 && generated[0] == generated[1],
        "{generated:?}"
    );
    // The last chunk, of tokens 512 to 600, checkpointed its last page
    // boundary: the deepest one leaving a token to compute.
    let reused = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    anyhow::ensure!(
        reused > 512 && reused < 600,
        "the second request reused {reused} prompt tokens"
    );
    Ok(())
}

/// A deployment of a prefill group and a decode group (`WorkerGroupRole`)
/// prefills each request on the first, which exports the request's KV
/// cache once its prompt is complete, and decodes it on the second, which
/// installs that cache: requests generate what one model group generates,
/// though the prefill group serves no decode and the decode group no
/// prefill.
#[test]
fn a_prefill_group_hands_its_requests_to_a_decode_group() -> anyhow::Result<()> {
    let generate = |config: EngineConfig| -> anyhow::Result<Vec<Vec<u32>>> {
        let engine = SchedulerHandle::start_all(config)?.remove(0);
        let result = (|| -> anyhow::Result<Vec<Vec<u32>>> {
            // Two requests at once, one of a prompt of three chunks.
            let mut streams = Vec::new();
            for (id, prompt) in [
                (101_u64, vec![7, 8, 9]),
                (102, (0..600u32).map(|index| index % 47 + 5).collect()),
            ] {
                streams.push(engine.submit(Request::from(engine::GenerationRequest {
                    request_id: RequestId(id),
                    prompt_token_ids: prompt,
                    negative_prompt_token_ids: Vec::new(),
                    multimodal_inputs: Default::default(),
                    constraint: engine::GenerationConstraint::UndOnly,
                    sampling: SamplingParams {
                        ignore_eos: true,
                        ..Default::default()
                    },
                    image: Default::default(),
                    max_images: 1,
                    max_und_tokens: 6,
                    stop_strings: Vec::new(),
                    stop_token_ids: Vec::new(),
                    priority: 0,
                    cache: Default::default(),
                    include_stop_token: false,
                    image_generation: Default::default(),
                    readout: Vec::new(),
                    generated_token_ids: Vec::new(),
                    adapter: 0,
                }))?);
            }
            let mut generated = Vec::new();
            for mut events in streams {
                let (tokens, terminal) = tokens_until_terminal(&mut events)?;
                anyhow::ensure!(
                    matches!(terminal, Event::Finished { .. }),
                    "a request ended with {terminal:?}"
                );
                generated.push(tokens);
            }
            Ok(generated)
        })();
        engine.shutdown();
        result
    };

    let expected = generate(stub_engine_config("one-group"))?;
    let mut config = stub_engine_config("prefill-decode");
    let group = config.worker_groups.remove(0);
    config.worker_groups = [
        ("prefill", engine::WorkerGroupRole::Prefill),
        ("decode", engine::WorkerGroupRole::Decode),
    ]
    .into_iter()
    .map(|(id, role)| WorkerGroupConfig {
        id: engine::WorkerGroupId(id.into()),
        role,
        ..group.clone()
    })
    .collect();
    let handed = generate(config)?;
    anyhow::ensure!(
        expected.iter().all(|tokens| tokens.len() == 6) && handed == expected,
        "one group generated {expected:?}, the prefill and decode groups {handed:?}"
    );
    Ok(())
}

/// A world that succeeds a running one starts beside it under the same
/// claim, serves once the running world parked and it is promoted with the
/// running world's prefix cache index, commits once the running world
/// closed, and continues a request the running one generated part of: the
/// continuation generates what the uninterrupted request generates after
/// those tokens, and its usage is the uninterrupted request's.
#[test]
fn a_successor_world_continues_the_requests_of_the_world_it_replaces() -> anyhow::Result<()> {
    let deployment = format!("worlds-{}", std::process::id());
    let mut first = stub_engine_config(&deployment);
    first.world = 1;
    let running = SchedulerHandle::start_all(first)?.remove(0);
    let result =
        (|| -> anyhow::Result<(Vec<u32>, Event, SchedulerHandle)> {
            let (reference, finished) =
                tokens_until_terminal(&mut running.submit(greedy_text(91, Vec::new(), 8))?)?;
            anyhow::ensure!(
                matches!(finished, Event::Finished { .. }),
                "the reference did not finish: {finished:?}"
            );

            let caches = running
                .describe_cache()
                .recv()?
                .map_err(|error| anyhow::anyhow!(error))?;
            let mut config = stub_engine_config(&deployment);
            config.world = 2;
            config.predecessor = Some(
                caches
                    .into_iter()
                    .map(|cache| {
                        (
                            cache.worker_group,
                            ipc::PredecessorCache {
                                world: 1,
                                segments: cache.segments[..cache.held as usize].to_vec(),
                                num_units: cache.units,
                                scratch_units: cache.scratch,
                            },
                        )
                    })
                    .collect(),
            );
            let successor = SchedulerHandle::start_all(config)?.remove(0);
            let index = running
                .park()
                .recv()?
                .map_err(|error| anyhow::anyhow!(error))?;
            successor.promote(index, false, false)?;
            running.shutdown();
            successor
                .commit()
                .recv()?
                .map_err(|error| anyhow::anyhow!(error))?;

            let (continued, finished) = tokens_until_terminal(
                &mut successor.submit(greedy_text(92, reference[..3].to_vec(), 8))?,
            )?;
            let mut whole = reference[..3].to_vec();
            whole.extend(continued);
            anyhow::ensure!(whole == reference, "{whole:?} differs from {reference:?}");
            Ok((reference, finished, successor))
        })();
    running.shutdown();
    let (_reference, finished, standby) = result?;
    standby.shutdown();
    assert!(
        matches!(
            finished,
            Event::Finished {
                prompt_tokens: 3,
                completion_tokens: 8,
                ..
            }
        ),
        "{finished:?}"
    );
    Ok(())
}

/// The tokens of the request whose worker group is lost: enough that the
/// kill after its first tokens lands while it generates.
const LOST_GROUP_TOKENS: usize = 1_000;

/// A token request on a worker group that is lost leaves the engine as
/// migratable after the tokens it generated, and submitted again with them it
/// waits for the relaunched group and continues as the uninterrupted request
/// does.
#[test]
fn a_request_on_a_lost_group_migrates_and_continues_after_the_relaunch() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // The wrapper records each launch's process ids by generation and rank.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let worker = serde_json::to_string(&worker_executable())?;
    let state = serde_json::to_string(directory.path())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nfrom pathlib import Path\nargument = lambda name: sys.argv[sys.argv.index(name) + 1]\n(Path({state}) / (argument('--generation') + '-' + argument('--rank') + '.pid')).write_text(str(os.getpid()))\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut config = stub_engine_config(&format!("relaunch-{}", std::process::id()));
    config.worker_process.executable = wrapper;
    let engine = SchedulerHandle::start_all(config)?.remove(0);

    let result = (|| -> anyhow::Result<()> {
        let (reference, finished) = tokens_until_terminal(&mut engine.submit(greedy_text(
            101,
            Vec::new(),
            LOST_GROUP_TOKENS,
        ))?)?;
        anyhow::ensure!(
            matches!(finished, Event::Finished { .. }),
            "the reference did not finish: {finished:?}"
        );

        let mut events = engine.submit(greedy_text(102, Vec::new(), LOST_GROUP_TOKENS))?;
        let mut generated = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while generated.len() < 3 {
            anyhow::ensure!(std::time::Instant::now() < deadline, "no tokens");
            match events.try_recv() {
                Ok(Event::TextToken { id, .. }) => generated.push(id),
                Ok(event) if event.is_terminal() => anyhow::bail!("ended early: {event:?}"),
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        let pid =
            std::fs::read_to_string(directory.path().join("1-0.pid"))?.parse::<libc::pid_t>()?;
        // SAFETY: kill takes only integer arguments and touches no memory of this process.
        anyhow::ensure!(unsafe { libc::kill(pid, libc::SIGKILL) } == 0);
        let (tokens, terminal) = tokens_until_terminal(&mut events)?;
        generated.extend(tokens);
        anyhow::ensure!(
            matches!(
                terminal,
                Event::Migratable {
                    cause: proto::request::MigrationCause::Loss,
                    ..
                }
            ),
            "the lost request did not leave as migratable: {terminal:?}"
        );

        let (continued, finished) = tokens_until_terminal(&mut engine.submit(greedy_text(
            103,
            generated.clone(),
            LOST_GROUP_TOKENS,
        ))?)?;
        anyhow::ensure!(
            matches!(
                finished,
                Event::Finished {
                    prompt_tokens: 3,
                    completion_tokens,
                    ..
                } if completion_tokens == LOST_GROUP_TOKENS
            ),
            "the continuation did not finish: {finished:?}"
        );
        generated.extend(continued);
        anyhow::ensure!(
            generated == reference,
            "the continuation differs from the reference"
        );
        Ok(())
    })();
    engine.shutdown();
    result
}

/// Workers launched by the engine hold a request row for every running
/// request the engine is configured to keep resident, beyond the worker's
/// own default pool.
/// Launched ranks keep their host heap out of transparent huge pages: no
/// mapping of a rank process is advised for them (`MADV_HUGEPAGE`, VmFlags
/// `hg`), so khugepaged never collapses a serving rank's memory.
#[test]
fn launched_workers_hold_no_huge_page_advised_memory() -> anyhow::Result<()> {
    let mut args = worker_group_args();
    args.worker_id = "huge-pages".into();
    let worker = WorkerGroup::spawn(args)?;
    let ranks: Vec<u32> = crate::support::group_processes("huge-pages")
        .into_iter()
        .map(|(pid, _, _)| pid)
        .collect();
    anyhow::ensure!(ranks.len() == WORLD_SIZE, "found rank processes {ranks:?}");
    for pid in &ranks {
        let smaps = std::fs::read_to_string(format!("/proc/{pid}/smaps"))?;
        let advised = smaps
            .lines()
            .filter_map(|line| line.strip_prefix("VmFlags:"))
            .filter(|flags| flags.split_whitespace().any(|flag| flag == "hg"))
            .count();
        assert_eq!(
            advised, 0,
            "rank process {pid} holds {advised} mappings advised for huge pages"
        );
    }
    drop(worker);
    Ok(())
}

#[test]
fn launched_workers_hold_a_request_row_per_running_request() -> anyhow::Result<()> {
    const RUNNING: usize = 200;

    let mut config = EngineConfig::sim("stub");
    config.generation_limits = engine::GenerationLimits {
        features: engine::GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.max_running_requests = RUNNING;
    config.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerGroupConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = worker_group_args();
    let engine = {
        let _launch_guard = CHILD_LAUNCH_ENV_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("child-launch environment lock is poisoned"))?;
        SchedulerHandle::start_all(config)?.remove(0)
    };
    let request_rows = engine.info().request_slots as usize;
    engine.shutdown();

    assert!(
        request_rows >= RUNNING,
        "{request_rows} request rows cannot hold {RUNNING} running requests"
    );
    Ok(())
}

#[test]
fn unsupported_media_is_rejected_without_stopping_the_engine() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .try_init();
    let mut config = EngineConfig::sim("stub");
    config.generation_limits = engine::GenerationLimits {
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.max_batch = QUEUE_DEPTH * 8;
    config.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerGroupConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = worker_group_args();
    let engine = {
        let _launch_guard = CHILD_LAUNCH_ENV_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("child-launch environment lock is poisoned"))?;
        SchedulerHandle::start_all(config)?.remove(0)
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;

    // Each of these unsupported video requests must be rejected through its
    // own event stream, and the engine must stay alive.
    let result = (|| -> anyhow::Result<()> {
        let mut requests = Vec::with_capacity(2);
        for index in 0..2 {
            let request_id = RequestId(10_000 + index as u64);
            let prompt_token_ids = vec![100_000 + index as u32];
            let conditioning = models::video::Conditioning {
                task: models::video::VideoTask::T2va,
                text_tags: vec![1; prompt_token_ids.len()],
                conditions: Vec::new(),
            };
            let events = engine.submit(Request::Diffusion(DiffusionRequest {
                request_id,
                task: "t2va".to_owned(),
                prompt_token_ids,
                conditioning: conditioning.encode()?,
                conditions: Vec::new(),
                priority: 0,
                sampling: DiffusionSamplingParams {
                    num_frames: 22,
                    video_units: 3,
                    seed: index as u64,
                    width: 1344,
                    height: 768,
                },
                trajectory: Default::default(),
            }))?;
            requests.push((request_id, events));
        }
        for (request_id, mut events) in requests {
            let event = runtime
                .block_on(async {
                    tokio::time::timeout(Duration::from_secs(10), events.recv()).await
                })
                .with_context(|| format!("timed out waiting for media request {request_id:?}"))?;
            let event = event.ok_or_else(|| {
                anyhow::anyhow!("media event stream closed for request {request_id:?}")
            })?;
            assert!(
                matches!(event, Event::Rejected { .. }),
                "unsupported request {request_id:?} was not rejected: {event:?}"
            );
        }
        Ok(())
    })();

    // The dead latch is set only when the scheduler thread exits with a fatal
    // status, so `shutdown` joins that thread before `is_dead` is read.
    engine.shutdown();
    result?;
    assert!(!engine.is_dead(), "media rejection killed the engine");
    Ok(())
}

#[test]
fn placement_binds_ranks_to_the_engine_host_by_name() {
    let named = WorkerGroupConfig::placed(
        &["compute-0".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerGroupConfig::single_component("model", WORLD_SIZE),
    );
    assert!(
        named.ranks.iter().all(|rank| rank.host == "compute-0"),
        "the placement shorthand must place ranks on the host it is given"
    );

    // A rank placed on another host is started by that host's daemon, which
    // joined the leader. No daemon runs on that host here, so the launch
    // fails naming the host whose ranks cannot start rather than waiting for
    // them or starting them in the wrong place.
    let spawned = WorkerGroup::spawn(WorkerProcessArgs {
        executable: worker_executable(),
        daemon: crate::support::daemon(),
        model: String::new(),
        host: "compute-0".into(),
        ranks: WorkerGroupConfig::placed(
            &["compute-1".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            WorkerGroupConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: named.components.clone(),
        stub: true,
        ..WorkerProcessArgs::default()
    });
    let Err(error) = spawned else {
        panic!("a rank on a host no daemon runs on must not launch");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("compute-1"),
        "the refusal must name the host no daemon runs on: {message}"
    );
}

/// A process stopped with `SIGSTOP` until `resume` continues it or `terminate`
/// kills it.
///
/// Dropping a still-paused value sends `SIGCONT`, so a failing test does not
/// leave a stopped rank behind.
struct PausedProcess(Option<i32>);

impl PausedProcess {
    /// Stops the process and returns once every one of its threads has
    /// stopped.
    ///
    /// `kill` returns when the signal is queued. The kernel hands it to one
    /// thread, and the others keep running until that thread starts the group
    /// stop, so a rank whose channel is served off its main thread could
    /// still answer a request sent right after `kill`.
    fn new(pid: i32) -> anyhow::Result<Self> {
        anyhow::ensure!(pid > 0, "worker process id must be positive");
        anyhow::ensure!(
            // SAFETY: kill takes only integer arguments and touches no memory of this process.
            unsafe { libc::kill(pid, libc::SIGSTOP) } == 0,
            "failed to pause the selected worker: {}",
            std::io::Error::last_os_error()
        );
        let paused = Self(Some(pid));

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !all_threads_stopped(pid)? {
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "worker process {pid} did not stop"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(paused)
    }

    fn resume(&mut self) -> anyhow::Result<()> {
        if let Some(pid) = self.0 {
            anyhow::ensure!(
                // SAFETY: kill takes only integer arguments and touches no memory of this process.
                unsafe { libc::kill(pid, libc::SIGCONT) } == 0,
                "failed to resume the selected worker: {}",
                std::io::Error::last_os_error()
            );
            self.0 = None;
        }
        Ok(())
    }

    fn terminate(mut self) -> anyhow::Result<()> {
        let pid = self.0.context("selected worker is no longer paused")?;
        anyhow::ensure!(
            // SAFETY: kill takes only integer arguments and touches no memory of this process.
            unsafe { libc::kill(pid, libc::SIGKILL) } == 0,
            "failed to terminate the selected worker: {}",
            std::io::Error::last_os_error()
        );
        self.0 = None;
        Ok(())
    }
}

impl Drop for PausedProcess {
    fn drop(&mut self) {
        let _ = self.resume();
    }
}

/// Whether every thread of process `pid` is in the stopped state (`T` in
/// `/proc/<pid>/task/<tid>/stat`).
fn all_threads_stopped(pid: i32) -> anyhow::Result<bool> {
    for task in std::fs::read_dir(format!("/proc/{pid}/task"))? {
        // A thread that exited between the listing and the read has no file.
        let Ok(stat) = std::fs::read_to_string(task?.path().join("stat")) else {
            continue;
        };
        // The state follows the command name, which is parenthesized and may
        // itself contain spaces and parentheses.
        let state = stat
            .rsplit_once(')')
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .context("malformed task stat")?;
        if state != "T" {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Spawns the default two-rank stub group from `worker_group_args`.
fn spawn_worker_group() -> anyhow::Result<WorkerGroup> {
    WorkerGroup::spawn(worker_group_args())
}

/// Launch arguments for a two-rank stub group on the local host's CPU.
///
/// Both ranks hold the single `model` component, with `QUEUE_DEPTH` batches
/// in flight. Tests adjust the returned value (for example components, executable or transfer edges)
/// before spawning.
fn worker_group_args() -> WorkerProcessArgs {
    let executable = worker_executable();
    let config = WorkerProcessArgs {
        stub: true,
        // The stub rank encodes many small images, so it keeps a larger
        // encoder cache than a served model's default.
        encoder_cache_entries: 1024,
        ..WorkerProcessArgs::default()
    };
    WorkerProcessArgs {
        executable,
        daemon: crate::support::daemon(),
        model: String::new(),
        ranks: WorkerGroupConfig::placed(
            &["localhost".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            WorkerGroupConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: WorkerGroupConfig::single_component("model", WORLD_SIZE),
        queue_depth: QUEUE_DEPTH,
        max_total_tokens: Some(4096),
        block_size: Some(16),
        max_batch_calls: 256,
        max_batch_tokens: 256,
        attention_backend: "torch".into(),
        transfer: Default::default(),
        ..config
    }
}

/// Builds the launch descriptor for a weightless CPU worker used by IPC tests.
///
/// The worker id a directly driven stub rank registers with: its argv
/// states it, as a launch's does.
const STUB_WORKER: &str = "worker";

/// The Storage Server the directly driven stub workers keep their memory
/// at, served by a thread of this test process, and the claim its head,
/// this process, holds for their deployment.
fn stub_storage() -> ipc::StorageLaunch {
    static STORAGE: std::sync::OnceLock<(std::sync::Mutex<storage::Client>, ipc::StorageLaunch)> =
        std::sync::OnceLock::new();
    STORAGE
        .get_or_init(|| {
            let directory = tempfile::tempdir().unwrap().keep();
            let socket = directory.join("storage.sock");
            let (listener, lock) = storage::socket::bind(&socket).unwrap().unwrap();
            std::thread::spawn(move || {
                let _lock = lock;
                let (_keep, wake) = std::os::unix::net::UnixStream::pair().unwrap();
                let mut server = storage_server::Server::new(
                    listener,
                    storage_server::DEFAULT_POOL_BYTES,
                    0,
                    None,
                    None,
                )
                .unwrap();
                server
                    .run(std::os::fd::AsFd::as_fd(&wake), &|| false)
                    .unwrap();
            });
            let mut head = storage::Client::connect(
                &socket,
                storage::Role::Head {
                    deployment: STUB_WORKER.into(),
                    world: 0,
                },
            )
            .unwrap();
            let claim = head.claim(0, Vec::new(), Vec::new()).unwrap();
            let launch = ipc::StorageLaunch {
                socket: socket.to_string_lossy().into_owned(),
                deployment: STUB_WORKER.into(),
                claim: claim.0,
                host_kv_cache: false,
                tcp: None,
                peers: Vec::new(),
                world: 0,
                standby: false,
                standby_memory: proto::request::StandbyMemory::Auto,
            };
            (std::sync::Mutex::new(head), launch)
        })
        .1
        .clone()
}

/// Production launches receive the same typed descriptor from the engine;
/// these tests state one directly because they drive the worker process
/// without a group.
fn stub_launch_descriptor() -> LaunchDescriptor {
    LaunchDescriptor {
        storage: stub_storage(),
        acknowledgment_slot: 0,
        host_slots: vec![0],
        products_cross_hosts: false,
        queue_depth: 8,
        // The engine's default watchdog timeout.
        watchdog_timeout_ms: 300_000,
        model: String::new(),
        weights: None,
        host: "localhost".into(),
        device: "cpu".into(),
        rank: 0,
        local_rank: 0,
        world_size: 1,
        components: WorkerGroupConfig::single_component("model", 1),
        deployment_components: vec!["model".into()],
        transfer_backends: "local".into(),
        export_backends: "local".into(),
        rendezvous_address: None,
        rendezvous_listen_fd: None,
        expert_parallel: None,
        tensor_experts: false,
        role: "model".into(),
        expert_microbatches: 1,
        distributed_backend: None,
        mesh: None,
        no_model: true,
        load_format: "auto".into(),
        download_dir: None,
        load_threads: None,
        parallel_degrees: Vec::new(),
        model_dtype: "bfloat16".into(),
        quantization_config: serde_json::json!({}),
        prefix_cache_dtype: None,
        mem_fraction_static: 0.7,
        max_total_tokens: Some(4096),
        predecessor: None,
        segments: None,
        received_weights: None,
        attention_backend: "torch".into(),
        block_size: Some(16),
        max_batch_calls: 8,
        max_batch_tokens: 256,
        max_request_pool_size: 129,
        max_model_len: 8192,
        max_video_seconds: 15.0,
        max_condition_rows: None,
        // The stub rank encodes many small images, so it keeps a larger
        // encoder cache than a served model's default.
        encoder_cache_entries: 1024,
        ffmpeg: "ffmpeg".into(),
        min_video_seconds: None,
        graph_policy: "off".into(),
        decode_graph_batch_sizes: None,
        prefill_outputs: true,
        draft_tokens: 0,
        prefill_graph_token_sizes: None,
        flow_graph_shapes: None,
        video_text_capacities: None,
        video_resolutions: Vec::new(),
        video_aspect_ratios: Vec::new(),
        canvas_sampling: None,
        flashinfer_workspace_size: 512 << 20,
        flashinfer_use_tensor_core: None,
        flashinfer_decode_backend: "fa2".into(),
        flashinfer_prefill_backend: "auto".into(),
        flashinfer_decode_split_tile_size: None,
        flashinfer_prefill_split_tile_size: None,
        flashinfer_disable_split_kv: false,
        fence_probes: false,
        fence_components: Vec::new(),
        graph_pc_sampling: false,
    }
}

fn worker_executable() -> PathBuf {
    std::env::var_os("UNISERVE_WORKER_EXECUTABLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(".venv/bin/uniserve")
        })
}

/// Places a batch's buffer spans at the same offsets on every rank of the
/// groups these tests launch, which share one layout of their pools.
fn on_every_rank(allocations: Vec<BufferAllocation>) -> Vec<RankBuffer> {
    RankBuffer::on_every_rank(&allocations, WORLD_SIZE as u32)
}

/// Submission of a batch that carries its buffer spans, which a group takes
/// beside the batch as a rank-by-rank placement.
trait SubmitPlaced {
    fn submit_placed(&mut self, batch: Batch) -> Result<(), engine::BatchSubmitError>;
}

impl SubmitPlaced for WorkerGroup {
    fn submit_placed(&mut self, mut batch: Batch) -> Result<(), engine::BatchSubmitError> {
        let spans = on_every_rank(std::mem::take(&mut batch.buffer_allocations));
        self.submit_batch(batch, &CallReaders::new(), &spans)
    }
}

/// Submits one batch and waits up to 30 seconds for its result.
fn execute(executor: &mut WorkerGroup, batch: Batch) -> anyhow::Result<engine::WorkerResult> {
    executor.submit_placed(batch)?;
    executor
        .poll_batch(Duration::from_secs(30))?
        .ok_or_else(|| anyhow::anyhow!("submission did not complete"))
}

/// A greedy text admission for engine 1 that ignores EOS, starting at
/// position zero in request pool slot `request_pool_idx`.
fn text_admission(
    request_id: u64,
    request_epoch: u64,
    request_pool_idx: u32,
) -> anyhow::Result<NewRequest> {
    Ok(NewRequest::new(
        RequestKey::new(1, RequestId(request_id), request_epoch),
        request_pool_idx,
        Some(GenerationParams {
            sampling: SamplingParams {
                temperature: 0.0,
                ignore_eos: true,
                ..SamplingParams::default()
            },
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 0,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }),
        None,
        0,
    )?)
}

// Products and transfers: tensor products published by one component or
// worker and imported by another, cache exports, and the retirement of work
// that reads a failed or unreachable product.

#[test]
fn independent_components_complete_on_their_assigned_ranks() -> anyhow::Result<()> {
    let mut args = worker_group_args();
    // The language backbone and the patch encoder hold a rank each, which is
    // the arrangement a model with several components is served in.
    args.components = split_components();
    let mut worker = WorkerGroup::spawn(args)?;
    let supported = worker.info().supported_calls.clone();
    assert!(supported.contains(&CallKind::Forward(ForwardMode::Prefill)));
    assert!(supported.contains(&CallKind::Media(MediaCall::VisionEncoding)));
    assert_eq!(
        worker.info().media_components[&MediaCall::VisionEncoding],
        "vision_encoder"
    );

    let first = text_admission(51, 1, 1)?;
    let second = image_admission(52, 1, 2)?;
    let first_key = first.request_key;
    let second_key = second.request_key;
    let mut extend = token_batch(
        1,
        1,
        first_key,
        Some(first),
        CallId::new(2, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        UnitId(1),
        0,
    );
    let second_admission = token_batch(
        1,
        1,
        second_key,
        Some(second),
        CallId::new(3, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9, 10],
        UnitId(2),
        0,
    );
    // Both requests are admitted by one command-only batch carrying their
    // `Start` commands, so the component batches that follow carry only calls.
    let mut starts = std::mem::take(&mut extend.commands);
    starts.extend(second_admission.commands);
    let admission = execute(
        &mut worker,
        Batch::new(1, vec![], vec![]).with_commands(starts),
    )?;
    assert!(admission.results.is_empty());

    // A batch carries one call kind for one component, so each component's
    // work travels in its own batch and retires on the ranks that component
    // holds.
    extend.batch_id = 2;
    extend.collective_seq = 2;
    let extend_report = execute(&mut worker, extend)?;
    assert_eq!(extend_report.results.len(), 1);
    assert_eq!(extend_report.results[0].output.call_id, CallId::new(2, 0));
    assert_eq!(extend_report.results[0].output.status, CallStatus::Ok);
    assert_eq!(extend_report.results[0].output.committed_tokens.len(), 1);

    // The patch encoder's feature lands in a buffer the batch allocates at
    // the product's maximum size.
    let feature = feature_ref(second_key, CallId::new(3, 0), 1);
    let mut encode = Batch::new(3, Vec::new(), vec![encode_call(second_key, 3, &feature)]);
    encode.collective_seq = 3;
    encode.buffer_allocations.push(BufferAllocation {
        buffer: feature.buffer_id(),
        offset: 0,
        bytes: feature.max_bytes(),
    });
    let encode_report = execute(&mut worker, encode)?;
    assert_eq!(encode_report.results.len(), 1);
    assert_eq!(encode_report.results[0].output.call_id, CallId::new(3, 0));
    assert_eq!(encode_report.results[0].output.status, CallStatus::Ok);

    // Closing a request releases every participating rank's physical storage.
    worker.submit_placed(
        Batch::new(4, vec![], vec![]).with_commands(vec![finish(first_key), finish(second_key)]),
    )?;
    let report = worker
        .poll_batch(Duration::from_secs(5))?
        .context("request release did not retire")?;
    assert!(report.results.is_empty());
    worker.close()?;
    Ok(())
}

#[test]
fn components_transfer_published_values_within_one_worker() -> anyhow::Result<()> {
    use engine::{WorkerExecutor, WorkerGroupId};

    // Once with an explicit shared-storage edge from rank 1 (the backbone) to
    // rank 0 (the patch encoder), and once with only the edges
    // `TransferConfig::with_worker_defaults` derives for this worker.
    for transfer in [
        engine::TransferConfig::parse("worker:1->worker:0=posix_shm")?,
        engine::TransferConfig::default(),
    ] {
        let mut args = worker_group_args();
        args.components = split_components();
        let transfer = transfer.with_worker_defaults(&[WorkerGroupConfig {
            role: engine::WorkerGroupRole::Model,
            id: WorkerGroupId("worker".into()),
            ranks: args.ranks.clone(),
            components: args.components.clone(),
            queue_depth: args.queue_depth,
            mem_fraction_static: None,
        }])?;
        args.transfer = transfer.clone();
        let mut executor = WorkerExecutor::try_new(
            vec![(WorkerGroupId("worker".into()), WorkerGroup::spawn(args)?)],
            transfer,
        )?;
        let admission = text_admission(61, 1, 1)?;
        let key = admission.request_key;
        let source = token_batch(
            1,
            1,
            key,
            Some(admission),
            CallId::new(1, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[7],
            UnitId(1),
            0,
        );
        let value = source.calls[0].token_output.clone().unwrap();

        // The chain under test: the backbone's prefill token (call 1) is
        // republished by a tensor transfer on the backbone's rank (call 2) and
        // copied by a transfer on the patch encoder's rank (call 3). Each step
        // names a new producer and generation for its output.
        let export = TensorRef {
            producer_call_id: CallId::new(2, 0),
            generation: 2,
            ..value.clone()
        };
        let publish = token_transfer(key, CallId::new(2, 0), "model", &value, &export);
        let copy = TensorRef {
            producer_call_id: CallId::new(3, 0),
            generation: 3,
            ..export.clone()
        };
        let consume = token_transfer(key, CallId::new(3, 0), "vision_encoder", &export, &copy);

        // Reads partial results until the submitted batch is done, requiring
        // every call to succeed and never to complete twice.
        let mut completions = std::collections::BTreeMap::new();
        let mut drain = |executor: &mut WorkerExecutor| -> anyhow::Result<()> {
            loop {
                let result = poll_logical(executor)?.context("a submitted batch did not retire")?;
                for result in result.results {
                    assert_eq!(result.output.status, CallStatus::Ok);
                    assert!(
                        completions
                            .insert(result.output.call_id, result.output)
                            .is_none()
                    );
                }
                if result.done {
                    return Ok(());
                }
            }
        };

        // Each batch carries one call kind for one component, and a batch
        // that reads another's product is submitted only after the batch
        // producing it has been drained to done. Batch 2 reaches only rank 1
        // and batch 3 only rank 0 (a rank with neither calls nor commands in
        // a batch is skipped), so channel order alone does not order them.
        executor.submit(place(source, "worker", Vec::new()))?;
        drain(&mut executor)?;
        executor.submit(place(
            Batch::new(2, vec![], vec![publish]),
            "worker",
            Vec::new(),
        ))?;
        drain(&mut executor)?;
        executor.submit(place(
            Batch::new(3, vec![], vec![consume]),
            "worker",
            Vec::new(),
        ))?;
        drain(&mut executor)?;
        assert_eq!(
            completions.keys().copied().collect::<Vec<_>>(),
            vec![CallId::new(1, 0), CallId::new(2, 0), CallId::new(3, 0)]
        );
        assert_eq!(completions[&CallId::new(1, 0)].committed_tokens, vec![1000]);
        executor.close()?;
    }
    Ok(())
}

#[test]
fn failed_producer_retires_waiting_consumers_and_preserves_independent_work() -> anyhow::Result<()>
{
    use engine::{WorkerExecutor, WorkerGroupId};

    // The group's name is this test's own, so its rank 0 is found among the
    // daemon's children by its group and rank.
    let transfer = engine::TransferConfig::parse("failed-producer:1->failed-producer:0=posix_shm")?;
    let mut args = worker_group_args();
    args.worker_id = "failed-producer".into();
    args.queue_depth = 3;
    args.transfer = transfer.clone();
    args.components = split_components();
    let mut executor = WorkerExecutor::try_new(
        vec![(
            WorkerGroupId("failed-producer".into()),
            WorkerGroup::spawn(args)?,
        )],
        transfer,
    )?;
    // Delay the command-only rank until the producer's completion is
    // observed. Its later acknowledgment must retire the same batch exactly
    // once. Rank 0 holds only the patch encoder, so batch 1, whose call runs
    // on the backbone's rank 1, reaches it only as the request's `Start`
    // command.
    let rank = worker_pid("failed-producer", 0)?;
    let mut delayed_rank = PausedProcess::new(rank)?;
    let key = RequestKey::new(1, RequestId(71), 1);
    let admission = NewRequest::new_media(
        key,
        1,
        vec![7],
        DiffusionSamplingParams {
            num_frames: 22,
            video_units: 1,
            seed: 1000,
            width: 1344,
            height: 768,
        },
        ipc::VideoAdmission {
            conditioning: models::video::Conditioning {
                task: models::video::VideoTask::T2va,
                text_tags: vec![1],
                conditions: Vec::new(),
            }
            .encode()?,
        },
    )?;
    let value = scalar_product(key, CallId::new(1, 0), 1, DType::F32);
    // The component holding the language backbone does not serve vision
    // encoding, which the patch encoder's own component does. Registration
    // succeeds, then the call reports its actual execution error without a
    // product.
    let mut produce = Call::new(
        key,
        CallId::new(1, 0),
        CallKind::Media(MediaCall::VisionEncoding),
    );
    produce.component = "model".into();
    produce.outputs = vec![value.clone()];
    let mut source = Batch::new(1, vec![admission], vec![produce]);
    source.buffer_allocations.push(BufferAllocation {
        buffer: value.buffer_id(),
        offset: 0,
        bytes: value.max_bytes(),
    });
    let copied = TensorRef {
        producer_call_id: CallId::new(2, 0),
        generation: 2,
        ..value.clone()
    };
    let mut consumer = Batch::new(
        2,
        vec![],
        vec![product_transfer(
            key,
            CallId::new(2, 0),
            "vision_encoder",
            &value,
            &copied,
        )],
    );
    consumer.buffer_allocations.push(BufferAllocation {
        buffer: copied.buffer_id(),
        offset: 256,
        bytes: copied.max_bytes(),
    });
    let independent = text_admission(72, 1, 2)?;
    let independent_key = independent.request_key;
    let independent = token_batch(
        3,
        3,
        independent_key,
        Some(independent),
        CallId::new(3, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9],
        UnitId(2),
        0,
    );

    executor.submit(place(source, "failed-producer", Vec::new()))?;
    executor.submit(place(consumer, "failed-producer", Vec::new()))?;
    executor.submit(place(independent, "failed-producer", Vec::new()))?;

    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut failed = false;
    let mut source_returned = false;
    let mut source_done = false;
    let mut batch_done = false;
    let mut independent_returned = false;
    while std::time::Instant::now() < deadline
        && !(failed && source_returned && source_done && batch_done && independent_returned)
    {
        match executor.poll(Duration::from_millis(100)) {
            // The failed producer retires the consumer waiting on its product
            // and reports that product's buffer as incomplete. The group
            // retains its allocations, so no endpoint is invalidated.
            Err(error) => {
                let loss = error.downcast::<WorkerFailure>()?;
                assert!(
                    !failed,
                    "one producer failure must retire its dependents once"
                );
                assert!(loss.endpoints.is_empty());
                assert_eq!(loss.requests, vec![key]);
                assert_eq!(loss.retired, vec![(2, key, CallId::new(2, 0))]);
                assert!(loss.buffers.contains(&value.buffer_id()));
                failed = true;
            }
            Ok(Some(result)) => {
                source_done |= result.done && result.batch_id == 1;
                batch_done |= result.done && result.batch_id == 3;
                let done = result.done;
                for result in result.results {
                    match result.output.call_id {
                        CallId {
                            batch_id: 1,
                            request_index: 0,
                        } => {
                            assert_eq!(result.output.status, CallStatus::Error);
                            assert!(!source_returned);
                            assert!(!done, "command-only rank has not acknowledged yet");
                            source_returned = true;
                            delayed_rank.resume()?;
                        }
                        CallId {
                            batch_id: 3,
                            request_index: 0,
                        } => {
                            assert_eq!(result.output.status, CallStatus::Ok);
                            assert_eq!(result.output.committed_tokens, vec![1000]);
                            assert!(!independent_returned);
                            independent_returned = true;
                        }
                        kind => panic!("unexpected completed call {kind:?}"),
                    }
                }
            }
            Ok(None) => {}
        }
    }
    assert!(failed && source_returned && source_done && batch_done && independent_returned);
    executor.close()?;
    Ok(())
}

#[test]
fn input_no_edge_carries_fails_only_the_requests_reading_it() -> anyhow::Result<()> {
    use engine::{WorkerExecutor, WorkerGroupId};
    use ipc::{Locator, TensorExport, TensorTransfer, TransferHandle, TransferTransport};

    // Two one-worker groups whose products cross only over shared storage.
    // A product whose one location is in its producer's own address space
    // therefore has no way to the other group.
    let edges = "producer->consumer=posix_shm,consumer->producer=posix_shm";
    let spawn = |worker_id: &str| -> anyhow::Result<WorkerGroup> {
        let mut args = single_worker_args(worker_id, QUEUE_DEPTH);
        args.transfer = engine::TransferConfig::parse(edges)?;
        WorkerGroup::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        vec![
            (WorkerGroupId("producer".into()), spawn("producer")?),
            (WorkerGroupId("consumer".into()), spawn("consumer")?),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;
    let producer = executor
        .info()
        .workers
        .iter()
        .find(|(id, _)| id.0 == "producer")
        .map(|(_, info)| info.endpoint.clone())
        .context("producer group is not bound")?;

    let admission = text_admission(81, 1, 1)?;
    let key = admission.request_key;
    let value = scalar_product(key, CallId::new(1, 0), 1, DType::F32);
    // The producer's own address space is the product's only location.
    let local_only = TensorExport {
        product: value.clone(),
        value: TransferHandle::DeviceProduct {
            height: 0,
            width: 0,
            value_range: String::new(),
            tensor: TensorTransfer {
                shape: vec![1],
                locations: vec![Locator {
                    source: producer.clone(),
                    transport: TransferTransport::Local {
                        endpoint: producer.incarnation.clone(),
                        key: 1,
                    },
                    nbytes: 4,
                    dtype: "float32".to_owned(),
                    shape: vec![1],
                    offset: vec![0],
                    device: "cpu".to_owned(),
                }],
            },
        },
    };
    let copied = TensorRef {
        producer_call_id: CallId::new(2, 0),
        generation: 2,
        ..value.clone()
    };
    let mut refused = Batch::new(
        2,
        vec![admission],
        vec![product_transfer(
            key,
            CallId::new(2, 0),
            "vision_encoder",
            &value,
            &copied,
        )],
    );
    refused.buffer_allocations.push(BufferAllocation {
        buffer: copied.buffer_id(),
        offset: 0,
        bytes: copied.max_bytes(),
    });

    // The refusal fails the one request reading the product; the group
    // keeps its allocations and stays usable.
    executor.submit(place(refused, "consumer", vec![local_only]))?;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut failed = false;
    let mut retired = false;
    while std::time::Instant::now() < deadline && !(failed && retired) {
        match executor.poll(Duration::from_millis(100)) {
            Err(error) => {
                let loss = error.downcast::<WorkerFailure>()?;
                assert!(!failed, "one refusal must fail its request once");
                assert_eq!(loss.worker_id.0, "consumer");
                assert!(loss.endpoints.is_empty());
                assert_eq!(loss.requests, vec![key]);
                assert_eq!(loss.retired, vec![(2, key, CallId::new(2, 0))]);
                assert!(
                    loss.message
                        .contains("no location on a configured transfer edge"),
                    "{}",
                    loss.message
                );
                failed = true;
            }
            Ok(Some(result)) => {
                assert_eq!(result.batch_id, 2);
                assert!(result.results.is_empty(), "the refused call never ran");
                retired |= result.done;
            }
            Ok(None) => {}
        }
    }
    assert!(failed && retired);

    let independent = text_admission(82, 1, 2)?;
    let independent_key = independent.request_key;
    let independent = token_batch(
        3,
        3,
        independent_key,
        Some(independent),
        CallId::new(3, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9],
        UnitId(2),
        0,
    );
    executor.submit(place(independent, "consumer", Vec::new()))?;
    let completed = poll_logical(&mut executor)?.context("independent work did not complete")?;
    assert!(completed.done);
    assert_eq!(completed.results.len(), 1);
    assert_eq!(completed.results[0].output.status, CallStatus::Ok);
    assert_eq!(completed.results[0].output.committed_tokens, vec![1000]);
    executor.close()?;
    Ok(())
}

/// Independent groups accepting the same call must retain
/// separate capacity.
#[test]
fn independent_workers_preserve_capacity_retirement_and_failed_work() -> anyhow::Result<()> {
    use engine::{ExecutionBatch, ExecutorSubmitError, WorkerExecutor, WorkerGroupId};
    use std::os::unix::fs::PermissionsExt as _;

    // Control process scheduling at the OS boundary. This keeps the occupied
    // group deterministic without forging a product from an unbound
    // source. The wrapper writes encoder-0's rank pid, which `execv`
    // preserves, so the test can pause that rank.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let pid_file = directory.path().join("encoder-0.pid");
    let worker = serde_json::to_string(&worker_executable())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nwith open({}, 'w') as output:\n    output.write(str(os.getpid()))\nos.execv({worker}, [{worker}, *sys.argv[1:]])\n",
            serde_json::to_string(&pid_file)?,
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    // Each group is a one-worker group holding every stub component, with
    // its own queue depth; both route products to each other over shared
    // storage.
    let edges = "encoder-0->encoder-1=posix_shm,encoder-1->encoder-0=posix_shm";
    let spawn = |worker_id: &str, depth| -> anyhow::Result<WorkerGroup> {
        let mut args = single_worker_args(worker_id, depth);
        if worker_id == "encoder-0" {
            args.executable = wrapper.clone();
        }
        args.transfer = engine::TransferConfig::parse(edges)?;
        WorkerGroup::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        vec![
            (WorkerGroupId("encoder-0".into()), spawn("encoder-0", 1)?),
            (WorkerGroupId("encoder-1".into()), spawn("encoder-1", 2)?),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;
    let pid = std::fs::read_to_string(&pid_file)?.parse::<i32>()?;
    let mut paused = PausedProcess::new(pid)?;
    let admitted_batch = |admission: NewRequest, batch_id, request_index, page| -> Batch {
        token_batch(
            batch_id,
            batch_id,
            admission.request_key,
            Some(admission),
            CallId::new(batch_id, request_index),
            CallKind::Forward(ForwardMode::Prefill),
            &[7],
            UnitId(page),
            0,
        )
    };
    let make_batch = |batch_id, request_index, request_id, page| -> anyhow::Result<Batch> {
        Ok(admitted_batch(
            text_admission(request_id, 1, page)?,
            batch_id,
            request_index,
            page,
        ))
    };

    // Paused encoder-0 holds batch 1 in its only slot, so batch 2 for it must
    // be handed back, while encoder-1 still serves batch 3.
    executor.submit(place(make_batch(1, 0, 51, 1)?, "encoder-0", Vec::new()))?;
    let blocked = place(make_batch(2, 0, 52, 2)?, "encoder-0", Vec::new());
    let blocked = match executor.submit(blocked) {
        Err(ExecutorSubmitError::WouldBlock(batch)) => batch,
        other => anyhow::bail!("occupied group did not apply its capacity bound: {other:?}"),
    };
    let independent = make_batch(3, 0, 53, 1)?;
    let token = independent.calls[0].token_output.clone().unwrap();
    executor.submit(place(independent, "encoder-1", Vec::new()))?;
    let first = poll_logical(&mut executor)?.context("independent worker did not complete")?;
    assert_eq!(first.batch_id, 3);
    assert_eq!(first.results[0].output.status, CallStatus::Ok);
    paused.resume()?;
    let slow = poll_logical(&mut executor)?.context("released worker did not complete")?;
    assert_eq!(slow.batch_id, 1);
    assert_eq!(slow.results[0].output.status, CallStatus::Ok);
    executor.submit(*blocked)?;
    let resumed = poll_logical(&mut executor)?.context("capacity was not reusable")?;
    assert_eq!(resumed.batch_id, 2);
    assert_eq!(resumed.results[0].output.status, CallStatus::Ok);

    // A request-relay token product has no `BufferAllocation`, but its `Free`
    // must still reach the rank holding the published generation (encoder-1,
    // which ran batch 3).
    executor.submit(ExecutionBatch::new(
        4,
        Vec::new(),
        vec![BatchCommand::Free {
            buffer: token.buffer_id(),
        }],
        Vec::new(),
    ))?;
    let retired = poll_logical(&mut executor)?.context("product release did not complete")?;
    assert_eq!(retired.batch_id, 4);
    assert!(retired.results.is_empty());

    // The same request owns numerical state on one group and a
    // transported product on another; Finish must retire both physical
    // owners. Batches 5 to 8 prefill on encoder-0, publish the token there,
    // copy it to encoder-1, and encode an image on encoder-1, whose feature
    // the Finish retains.
    let shared = admitted_batch(image_admission(54, 1, 3)?, 5, 0, 3);
    let admission = shared.admissions().next().unwrap().clone();
    let key = admission.request_key;
    let source = shared.calls[0].token_output.clone().unwrap();
    executor.submit(place(shared, "encoder-0", Vec::new()))?;
    let produced = poll_logical(&mut executor)?.context("source did not complete")?;
    assert!(produced.done);
    let export = TensorRef {
        producer_call_id: CallId::new(6, 0),
        generation: 2,
        ..source.clone()
    };
    let mut publish = token_transfer(key, CallId::new(6, 0), "model", &source, &export);
    publish.coordinates = coordinates_after(&produced.results[0].output);
    executor.submit(place(
        Batch::new(6, Vec::new(), vec![publish]),
        "encoder-0",
        Vec::new(),
    ))?;
    let published = poll_logical(&mut executor)?.context("source export did not complete")?;
    assert!(published.done);
    assert_eq!(published.results[0].output.status, CallStatus::Ok);
    let copy = TensorRef {
        producer_call_id: CallId::new(7, 0),
        generation: 3,
        ..export.clone()
    };
    let transfer = token_transfer(key, CallId::new(7, 0), "model", &export, &copy);
    executor.submit(place(
        Batch::new(7, vec![admission.clone()], vec![transfer]),
        "encoder-1",
        Vec::new(),
    ))?;
    let copied = poll_logical(&mut executor)?.context("auxiliary transfer did not complete")?;
    assert!(copied.done);
    assert_eq!(copied.results[0].output.status, CallStatus::Ok);

    let feature = feature_ref(key, CallId::new(8, 0), 4);
    let mut encode = Batch::new(8, Vec::new(), vec![encode_call(key, 8, &feature)]);
    encode.buffer_allocations.push(BufferAllocation {
        buffer: feature.buffer_id(),
        offset: 0,
        bytes: 8192,
    });
    executor.submit(place(encode, "encoder-1", Vec::new()))?;
    let encoded = poll_logical(&mut executor)?.context("encoder product did not complete")?;
    assert_eq!(encoded.results[0].output.status, CallStatus::Ok);

    executor.submit(ExecutionBatch::new(
        9,
        Vec::new(),
        vec![BatchCommand::Finish {
            request_key: key,
            retained_buffers: vec![feature.buffer_id()],
        }],
        Vec::new(),
    ))?;
    let closed = poll_logical(&mut executor)?.context("shared request did not retire")?;
    assert!(closed.done);

    // Request pool slot 3 is free again on both groups.
    for (batch_id, request_id, worker) in [(10, 55, "encoder-0"), (11, 56, "encoder-1")] {
        executor.submit(place(
            make_batch(batch_id, 0, request_id, 3)?,
            worker,
            Vec::new(),
        ))?;
        let reused = poll_logical(&mut executor)?.context("retired slot was not reusable")?;
        assert!(reused.done);
        assert_eq!(reused.results[0].output.status, CallStatus::Ok);
    }

    // Closing one request leaves admitted work on both groups
    // independent. Batch 12 prefills request 57 on encoder-0 and request 61
    // on encoder-1, and finishes request 56, admitted on encoder-1 by batch
    // 11.
    let closing_key = RequestKey::new(1, RequestId(56), 1);
    let active_key = RequestKey::new(1, RequestId(61), 1);
    let mut mixed = place(make_batch(12, 0, 57, 4)?, "encoder-0", Vec::new());
    let affected = place(make_batch(12, 1, 61, 6)?, "encoder-1", Vec::new());
    mixed.requests.extend(affected.requests);
    mixed.commands.extend(affected.commands);
    mixed.input_transfers.extend(affected.input_transfers);
    mixed.commands.push(finish(closing_key));
    executor.submit(mixed)?;
    let mut completed = false;
    let mut tokens = 0;
    while !completed {
        let report = poll_logical(&mut executor)?.context("mixed closure did not complete")?;
        assert_eq!(report.batch_id, 12);
        for result in report.results {
            assert_eq!(result.output.status, CallStatus::Ok);
            tokens += result.output.committed_tokens.len();
        }
        if report.done {
            assert_eq!(report.command_results.len(), 1);
            assert_eq!(
                report.command_results[0].outcome,
                engine::CommandOutcome::Applied
            );
            completed = true;
        }
    }
    assert_eq!(tokens, 2);
    executor.submit(ExecutionBatch::new(
        13,
        Vec::new(),
        vec![finish(closing_key), finish(active_key)],
        Vec::new(),
    ))?;
    assert!(
        poll_logical(&mut executor)?
            .context("request close did not retire")?
            .done
    );
    executor.submit(place(make_batch(14, 0, 58, 3)?, "encoder-1", Vec::new()))?;
    let reused = poll_logical(&mut executor)?.context("closed request slot was not reusable")?;
    assert_eq!(reused.results[0].output.status, CallStatus::Ok);

    // The retained product's separate lifetime survives request slot reuse:
    // a later request copies the retained feature from encoder-1 into a new
    // feature on encoder-0.
    let next = text_admission(59, 1, 5)?;
    let mut retained = Call::new(
        next.request_key,
        CallId::new(15, 0),
        CallKind::Transfer(TransferMode::Tensor),
    );
    retained.vision_inputs = vec![ipc::VisionInput {
        offset: 0,
        feature: feature.clone(),
    }];
    let target = TensorRef {
        request_key: next.request_key,
        producer_call_id: CallId::new(15, 0),
        ..feature.clone()
    };
    retained.encoder_output = Some(target.clone());
    retained.bounds = Bounds {
        max_transfer_bytes: feature.max_bytes(),
        ..Bounds::default()
    };
    let mut retained = Batch::new(15, vec![next], vec![retained]);
    // The source product lives at offset zero on encoder-1. Its destination
    // address on encoder-0 is independent and must not collide with the new
    // output that this call writes at offset zero in encoder-0's address
    // space.
    retained.buffer_allocations.extend([
        BufferAllocation {
            buffer: target.buffer_id(),
            offset: 0,
            bytes: 8192,
        },
        BufferAllocation {
            buffer: feature.buffer_id(),
            offset: 8192,
            bytes: 8192,
        },
    ]);
    executor.submit(place(retained, "encoder-0", Vec::new()))?;
    let retained = poll_logical(&mut executor)?.context("retained product was not readable")?;
    assert_eq!(retained.results[0].output.status, CallStatus::Ok);
    executor.submit(place(make_batch(17, 0, 62, 6)?, "encoder-1", Vec::new()))?;
    let fresh =
        poll_logical(&mut executor)?.context("WorkerGroup did not accept work after closure")?;
    assert_eq!(fresh.results[0].output.status, CallStatus::Ok);
    executor.close()?;
    Ok(())
}

/// Publishes a prefilled request's prefix cache with a `TransferMode::CacheExport` call
/// on a two-rank text group and requires every published tensor to name a
/// location on each rank.
#[test]
fn a_cache_export_names_a_location_on_every_rank() -> anyhow::Result<()> {
    let mut executor = spawn_worker_group()?;
    let admission = text_admission(13, 1, 3)?;
    let request_key = admission.request_key;
    let mut initial = token_batch(
        9,
        5,
        request_key,
        Some(admission),
        CallId::new(9, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        UnitId(3),
        0,
    );
    initial.calls[0].token_output = None;
    let tables = initial.block_tables.clone();
    let first = execute(&mut executor, initial)?;
    assert_eq!(first.results[0].output.status, CallStatus::Ok);

    // The export locates the request's cache through the same block tables
    // the prefill wrote it with.
    let buffer = ipc::BufferId {
        owner: request_key,
        producer_call_id: CallId::new(10, 0),
        output_index: 0,
        generation: 2,
    };
    let mut call = Call::new(
        request_key,
        CallId::new(10, 0),
        CallKind::Transfer(TransferMode::CacheExport),
    );
    call.coordinates = coordinates_after(&first.results[0].output);
    call.cache_output = Some(buffer);
    call.bounds = Bounds {
        max_transfer_bytes: 4096,
        ..Bounds::default()
    };
    let mut publish = Batch::new(10, Vec::new(), vec![call]);
    publish.collective_seq = 6;
    publish.block_tables = tables;
    let report = execute(&mut executor, publish.clone())?;
    assert_eq!(report.results[0].output.status, CallStatus::Ok);
    let export = report.results[0]
        .output
        .cache_output
        .as_ref()
        .context("cache export has no physical locations")?;
    assert_eq!(export.source, buffer);
    assert_eq!(export.exported_extent, 2);
    // Every rank of the group contributes a location from its own
    // incarnation.
    for tensor in export.tensors() {
        let sources = tensor
            .locations
            .iter()
            .map(|location| &location.source.incarnation)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(sources.len(), WORLD_SIZE);
    }
    // A repeated batch id is refused.
    assert!(execute(&mut executor, publish).is_err());
    executor.close()?;
    Ok(())
}

/// Blocks a batch on an external shared-storage input that is not yet ready
/// and requires the group to hold it and the batch behind it, without
/// resubmission, until the publisher marks the segment ready. Then copies
/// the same external input into a device product through a tensor transfer.
#[test]
fn an_unpublished_external_input_holds_its_batch_until_ready() -> anyhow::Result<()> {
    let mut args = worker_group_args();
    // The segment stands in for a product that rank 0 of this worker
    // published over shared storage.
    args.transfer = engine::TransferConfig::parse("worker->worker=posix_shm")?;
    let mut executor = WorkerGroup::spawn(args)?;
    let publisher = executor.info().endpoint.clone();

    let slow_admission = text_admission(31, 1, 1)?;
    let mut slow = token_batch(
        1,
        1,
        slow_admission.request_key,
        Some(slow_admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[6],
        UnitId(1),
        0,
    );
    let predicate = TensorRef {
        request_key: slow_admission.request_key,
        producer_call_id: CallId::new(91, 0),
        output_index: 0,
        generation: 91,
        dtype: DType::U8,
        shape_bound: ShapeBound::default(),
    };
    // The slow call's predicate is the external segment, which stays pending
    // until `SlowShmExport::publish` runs.
    let export = SlowShmExport::start()?;
    slow.calls[0].predicate = Some(predicate.clone());
    slow.input_products.push(ipc::TensorExport {
        product: predicate,
        value: export.descriptor(&publisher)?,
    });

    let fast_admission = text_admission(32, 1, 2)?;
    let fast = token_batch(
        2,
        2,
        fast_admission.request_key,
        Some(fast_admission),
        CallId::new(2, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9],
        UnitId(2),
        0,
    );

    // Resubmitting batch 1 is refused because batch ids must increase; the
    // original submission stays in flight.
    executor.submit_placed(slow.clone())?;
    assert!(matches!(
        executor.submit_placed(slow),
        Err(engine::BatchSubmitError::Failed(_))
    ));
    executor.submit_placed(fast)?;
    // This worker's component spans both ranks, so its ranks stay inside the
    // same collective and hold one batch in flight. The blocked batch is at
    // the head of the channel, so nothing behind it runs until its
    // dependency is readable, and neither batch completes meanwhile.
    assert!(executor.poll_batch(Duration::from_millis(50))?.is_none());
    assert!(!export.published.load(Ordering::Acquire));

    // Waiting does not submit the slow batch again. Its original result
    // becomes available once the external publisher makes the dependency
    // readable, and the batch behind it follows in the order the channel
    // delivered them.
    export.publish()?;
    let first = executor
        .poll_batch(Duration::from_secs(30))?
        .context("blocked submission did not complete")?;
    assert_eq!(first.batch_id, 1);
    assert_eq!(first.results[0].output.status, CallStatus::Ok);
    let second = executor
        .poll_batch(Duration::from_secs(30))?
        .context("following submission did not complete")?;
    assert_eq!(second.batch_id, 2);
    assert_eq!(second.results[0].output.status, CallStatus::Ok);

    // A tensor transfer reads the now-ready segment as its input and must
    // publish its output as a device product with a location on every rank.
    let admission = text_admission(33, 1, 3)?;
    let input = TensorRef {
        request_key: admission.request_key,
        producer_call_id: CallId::new(92, 0),
        output_index: 0,
        generation: 92,
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![ipc::DimBound::Static(1)],
        },
    };
    let output = TensorRef {
        producer_call_id: CallId::new(3, 0),
        generation: 1,
        ..input.clone()
    };
    let call = product_transfer(
        admission.request_key,
        CallId::new(3, 0),
        "model",
        &input,
        &output,
    );
    let mut batch = Batch::new(3, vec![admission], vec![call]);
    batch.collective_seq = 3;
    batch.buffer_allocations = [(&input, 0), (&output, 256)]
        .into_iter()
        .map(|(product, offset)| BufferAllocation {
            buffer: product.buffer_id(),
            offset,
            bytes: product.max_bytes(),
        })
        .collect();
    batch.input_products.push(ipc::TensorExport {
        product: input,
        value: export.descriptor(&publisher)?,
    });
    executor.submit_placed(batch)?;
    let transferred = executor
        .poll_batch(Duration::from_secs(30))?
        .context("tensor input did not reach its component")?;
    assert_eq!(transferred.results[0].output.status, CallStatus::Ok);
    assert_eq!(transferred.products[0].product, output);
    let ipc::TransferHandle::DeviceProduct { tensor, .. } = &transferred.products[0].value else {
        anyhow::bail!("tensor output has no physical export");
    };
    assert_eq!(tensor.shape, vec![1]);
    assert_eq!(tensor.locations.len(), WORLD_SIZE);

    // The ranks only read the external segment; it survives their close and
    // is removed when `export` drops.
    executor.close()?;
    assert_eq!(export.file.metadata()?.nlink(), 1);
    Ok(())
}

/// A request prefilled on one worker continues on another: the source
/// publishes its visible cache extent, the target installs it into the pages
/// the scheduler assigned there, and the target's next decode reads it.
#[test]
fn a_request_continues_on_the_worker_that_installed_its_cache() -> anyhow::Result<()> {
    use engine::{WorkerExecutor, WorkerGroupId};

    let edges = "source->target=posix_shm";
    let spawn = |worker_id: &str| -> anyhow::Result<WorkerGroup> {
        let mut args = single_worker_args(worker_id, QUEUE_DEPTH);
        args.transfer = engine::TransferConfig::parse(edges)?;
        WorkerGroup::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        vec![
            (WorkerGroupId("source".into()), spawn("source")?),
            (WorkerGroupId("target".into()), spawn("target")?),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;
    let admission = text_admission(91, 1, 1)?;
    let key = admission.request_key;

    // The source prefills two prompt tokens and publishes their prefix cache.
    let prefill = token_batch(
        1,
        1,
        key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8],
        UnitId(1),
        0,
    );
    let tables = prefill.block_tables.clone();
    executor.submit(place(prefill, "source", Vec::new()))?;
    let prefilled = poll_logical(&mut executor)?.context("source prefill did not complete")?;
    assert_eq!(prefilled.results[0].output.status, CallStatus::Ok);
    let exported = ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(2, 0),
        output_index: 0,
        generation: 2,
    };
    let mut export = Call::new(
        key,
        CallId::new(2, 0),
        CallKind::Transfer(TransferMode::CacheExport),
    );
    export.coordinates = coordinates_after(&prefilled.results[0].output);
    export.cache_output = Some(exported);
    export.bounds.max_transfer_bytes = 4096;
    let mut publish = Batch::new(2, Vec::new(), vec![export]);
    publish.block_tables = tables;
    executor.submit(place(publish, "source", Vec::new()))?;
    let published = poll_logical(&mut executor)?.context("cache export did not complete")?;
    assert_eq!(published.results[0].output.status, CallStatus::Ok);
    let transfer = published.results[0]
        .output
        .cache_output
        .clone()
        .context("cache export has no transfer")?;
    assert_eq!((transfer.base_extent, transfer.exported_extent), (0, 2));

    // The target admits the request and installs the published extent into
    // its own page; the executor supplies the export to the install.
    let installed = ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(3, 0),
        output_index: 0,
        generation: 3,
    };
    let mut install = Call::new(
        key,
        CallId::new(3, 0),
        CallKind::Transfer(TransferMode::CacheInstall),
    );
    install.cache_input = Some(exported);
    install.cache_output = Some(installed);
    install.bounds.max_transfer_bytes = 4096;
    let mut target = Batch::new(3, vec![admission], vec![install]);
    target.block_tables = vec![BlockTable {
        request_pool_idx: 1,
        group_id: 0,
        start_page: 0,
        unit_ids: vec![UnitId(5)],
        allocated_tokens: 2,
    }];
    target.new_cache_units = vec![CacheUnitAllocation {
        request_pool_idx: 1,
        group_id: 0,
        unit_ids: vec![UnitId(5)],
    }];
    executor.submit(place(target, "target", Vec::new()))?;
    let installation = poll_logical(&mut executor)?.context("cache install did not complete")?;
    let record = &installation.results[0].output;
    assert_eq!(record.status, CallStatus::Ok);
    assert_eq!(
        (record.cache_visible_len, record.cache_computed_len),
        (2, 2)
    );

    // The target decodes the sampled token after the installed prefix.
    let mut decode = token_batch(
        4,
        4,
        key,
        None,
        CallId::new(4, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[1000],
        UnitId(5),
        2,
    );
    decode.calls[0].coordinates.logical_position = 2;
    executor.submit(place(decode, "target", Vec::new()))?;
    let decoded = poll_logical(&mut executor)?.context("target decode did not complete")?;
    let record = &decoded.results[0].output;
    assert_eq!(record.status, CallStatus::Ok);
    assert_eq!(record.committed_tokens, vec![1001]);
    assert_eq!(record.cache_visible_len, 3);

    executor.submit(engine::ExecutionBatch::new(
        5,
        Vec::new(),
        vec![finish(key)],
        Vec::new(),
    ))?;
    assert!(
        poll_logical(&mut executor)?
            .context("request close did not retire")?
            .done
    );
    executor.close()?;
    Ok(())
}

/// The calls of a generated image's trajectory, as the scheduler plans a
/// 32x32 image of `IMAGE_STEPS` steps without guidance for the stub: each
/// builder returns a batch of one call, named by its batch id, whose
/// outputs take the generations `generation` and `generation + 1`.
mod trajectory {
    use ipc::BufferId;

    use super::*;

    /// Solver steps of the generated image.
    pub(super) const IMAGE_STEPS: u32 = 2;
    const SIDE: u32 = 32;
    /// The stub's latent rows of a 32x32 image, and their bytes.
    const LATENT_UNITS: u32 = 4;
    const LATENT_BYTES: u64 = 6144;
    const PROMPT: [u32; 3] = [5, 6, 7];
    /// Cache tokens of the conditioning: the prompt and the token that
    /// closes it.
    const CONDITIONING: u32 = 4;

    /// A text request with image parameters in `slot`, ending at the image
    /// tokens.
    pub(super) fn admission(request_id: u64, slot: u32) -> anyhow::Result<NewRequest> {
        let generation = GenerationParams {
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            negative_token_ids: Vec::new(),
            finish_token_ids: vec![1000, 1007],
            initial_position: 0,
            generated_token_ids: Vec::new(),
            adapter: 0,
        };
        let image = ipc::ImageParams {
            steps: IMAGE_STEPS as u16,
            height: SIDE,
            width: SIDE,
            seed: Some(7),
            cfg_text_scale: 1.0,
            ..ipc::ImageParams::default()
        };
        Ok(NewRequest::new(
            RequestKey::new(1, RequestId(request_id), 1),
            slot,
            Some(generation),
            Some(image),
            0,
        )?)
    }

    fn output(key: RequestKey, batch: u64, index: u16, generation: u32, dtype: DType) -> TensorRef {
        TensorRef {
            request_key: key,
            producer_call_id: CallId::new(batch, 0),
            output_index: index,
            generation,
            dtype,
            shape_bound: ShapeBound::default(),
        }
    }

    fn latent(key: RequestKey, batch: u64, generation: u32) -> TensorRef {
        TensorRef {
            shape_bound: ShapeBound {
                dims: vec![ipc::DimBound::Device {
                    max: (LATENT_BYTES / 2) as u32,
                }],
            },
            ..output(key, batch, 0, generation, DType::BF16)
        }
    }

    /// The coordinates of a call after the closed conditioning.
    fn coordinates() -> CallCoordinates {
        CallCoordinates {
            logical_position: PROMPT.len() as u32,
            cache_visible_len: CONDITIONING,
            cache_computed_len: CONDITIONING,
        }
    }

    /// The latent page and raster a call on the trajectory in `page`
    /// addresses; the rank's latent pool keeps the trajectory's step.
    fn params(key: RequestKey, batch: u64, page: u32) -> LatentParams {
        LatentParams {
            request_key: key,
            call_id: CallId::new(batch, 0),
            page_table: vec![page],
            latent_units: LATENT_UNITS,
            height: SIDE,
            width: SIDE,
        }
    }

    fn single(batch_id: u64, call: Call) -> Batch {
        let mut batch = Batch::new(batch_id, Vec::new(), vec![call]);
        batch.collective_seq = batch_id;
        batch
    }

    /// Start `admission` and prefill its prompt into a page of `unit`.
    pub(super) fn prompt(batch_id: u64, admission: &NewRequest, unit: UnitId) -> Batch {
        let slot = admission.request_pool_idx;
        let mut call = Call::new(
            admission.request_key,
            CallId::new(batch_id, 0),
            CallKind::Forward(ForwardMode::Prefill),
        );
        call.component = "model".into();
        call.input_token_ids = PROMPT.to_vec();
        call.bounds = Bounds {
            max_tokens: PROMPT.len() as u32,
            ..Bounds::default()
        };
        let mut batch = single(batch_id, call);
        batch.commands = vec![BatchCommand::Start {
            request: Box::new(admission.clone()),
        }];
        batch.block_tables = vec![BlockTable {
            request_pool_idx: slot,
            group_id: 0,
            start_page: 0,
            unit_ids: vec![unit],
            allocated_tokens: 16,
        }];
        batch.new_cache_units = vec![CacheUnitAllocation {
            request_pool_idx: slot,
            group_id: 0,
            unit_ids: vec![unit],
        }];
        batch.forward = ForwardBatch {
            call_indices: vec![0],
            request_pool_indices: vec![slot],
            seq_lens: vec![PROMPT.len() as u32],
            query_lens: vec![PROMPT.len() as u32],
            write_cache: vec![true],
        };
        batch
    }

    /// Write the token that closes the conditioning into the cache after
    /// the prompt, sampling nothing.
    pub(super) fn close(batch_id: u64, key: RequestKey, slot: u32, generation: u32) -> Batch {
        let mut call = Call::new(
            key,
            CallId::new(batch_id, 0),
            CallKind::Forward(ForwardMode::Prefill),
        );
        call.component = "model".into();
        call.coordinates = CallCoordinates {
            logical_position: PROMPT.len() as u32,
            cache_visible_len: PROMPT.len() as u32,
            cache_computed_len: PROMPT.len() as u32,
        };
        call.input_token_ids = vec![0];
        call.completion_output = Some(output(key, batch_id, 0, generation, DType::U8));
        call.bounds.max_tokens = 1;
        let mut batch = single(batch_id, call);
        batch.forward = ForwardBatch {
            call_indices: vec![0],
            request_pool_indices: vec![slot],
            seq_lens: vec![CONDITIONING],
            query_lens: vec![1],
            write_cache: vec![true],
        };
        batch
    }

    /// Export the conditioning cache of `key` as buffer `(batch_id, 0)`.
    pub(super) fn conditioning(
        batch_id: u64,
        key: RequestKey,
        generation: u32,
    ) -> (BufferId, Batch) {
        let buffer = BufferId {
            owner: key,
            producer_call_id: CallId::new(batch_id, 0),
            output_index: 0,
            generation,
        };
        let mut call = Call::new(
            key,
            CallId::new(batch_id, 0),
            CallKind::Transfer(TransferMode::CacheExport),
        );
        call.component = "model".into();
        call.coordinates = coordinates();
        call.cache_output = Some(buffer);
        call.completion_output = Some(output(key, batch_id, 1, generation + 1, DType::U8));
        call.bounds = Bounds {
            max_transfer_bytes: 16,
            ..Bounds::default()
        };
        (buffer, single(batch_id, call))
    }

    /// Draw the seeded initial latent of `key` into latent page `page`,
    /// conditioned on `conditioning`.
    pub(super) fn prepare(
        batch_id: u64,
        key: RequestKey,
        conditioning: BufferId,
        page: u32,
        generation: u32,
    ) -> (TensorRef, Batch) {
        let initial = latent(key, batch_id, generation + 1);
        let mut call = Call::new(
            key,
            CallId::new(batch_id, 0),
            CallKind::Media(MediaCall::LatentPreparation),
        );
        call.component = "model".into();
        call.coordinates = coordinates();
        call.cache_input = Some(conditioning);
        call.latent_output = Some(initial.clone());
        call.completion_output = Some(output(key, batch_id, 1, generation, DType::U8));
        call.image_index = Some(1);
        call.bounds = Bounds {
            max_tokens: 1,
            ..Bounds::default()
        };
        let mut batch = single(batch_id, call);
        batch.latent_params = vec![params(key, batch_id, page)];
        (initial, batch)
    }

    /// Denoise the next step of the trajectory whose committed version is
    /// `input`, in latent page `page`.
    pub(super) fn denoise(
        batch_id: u64,
        slot: u32,
        conditioning: BufferId,
        input: &TensorRef,
        page: u32,
        generation: u32,
    ) -> (TensorRef, Batch) {
        let key = input.request_key;
        let next = latent(key, batch_id, generation + 1);
        let mut call = Call::new(
            key,
            CallId::new(batch_id, 0),
            CallKind::Media(MediaCall::Denoising),
        );
        call.component = "model".into();
        call.coordinates = coordinates();
        call.cache_input = Some(conditioning);
        call.latent_input = Some(input.clone());
        call.latent_output = Some(next.clone());
        call.completion_output = Some(output(key, batch_id, 1, generation, DType::U8));
        call.bounds = Bounds {
            max_tokens: 1,
            ..Bounds::default()
        };
        let mut batch = single(batch_id, call);
        // The conditioning, then the latent rows between their two commit
        // marker tokens; denoising writes no cache.
        batch.forward = ForwardBatch {
            call_indices: vec![0],
            request_pool_indices: vec![slot],
            seq_lens: vec![CONDITIONING + LATENT_UNITS + 2],
            query_lens: vec![LATENT_UNITS + 2],
            write_cache: vec![false],
        };
        batch.latent_params = vec![params(key, batch_id, page)];
        (next, batch)
    }

    /// Decode the finished trajectory `input` to an image.
    pub(super) fn decode(batch_id: u64, input: &TensorRef, page: u32, generation: u32) -> Batch {
        let key = input.request_key;
        let mut call = Call::new(
            key,
            CallId::new(batch_id, 0),
            CallKind::Media(MediaCall::ImageDecoding),
        );
        call.component = "model".into();
        call.coordinates = coordinates();
        call.latent_input = Some(input.clone());
        call.completion_output = Some(output(key, batch_id, 2, generation, DType::U8));
        // The base64 bound the scheduler derives for a 32x32 PNG.
        call.bounds = Bounds {
            max_tokens: 1,
            max_completion_bytes: 1_406_380,
            ..Bounds::default()
        };
        let mut batch = single(batch_id, call);
        batch.latent_params = vec![params(key, batch_id, page)];
        batch
    }
}

/// Run `batch` on `worker` and require its one call to succeed.
fn completed(
    executor: &mut engine::WorkerExecutor,
    batch: Batch,
    worker: &str,
) -> anyhow::Result<engine::BatchResult> {
    let id = batch.batch_id;
    let kind = batch.calls[0].kind;
    executor.submit(place(batch, worker, Vec::new()))?;
    let result = poll_logical(executor)?.with_context(|| format!("batch {id} did not complete"))?;
    anyhow::ensure!(
        result.batch_id == id,
        "batch {} completed instead of {id}",
        result.batch_id
    );
    let output = &result.results[0].output;
    anyhow::ensure!(
        output.status == CallStatus::Ok,
        "{kind:?} on {worker} ended {:?} ({:?})",
        output.status,
        output.error_code
    );
    Ok(result)
}

/// The encoded image a decoding batch's result carries.
fn image_bytes(result: &engine::BatchResult) -> anyhow::Result<Vec<u8>> {
    let media = result.results[0]
        .media
        .clone()
        .map_err(anyhow::Error::msg)?
        .context("the decoding produced no image")?;
    Ok(media.as_bytes().to_vec())
}

/// A trajectory one worker prepared reaches another worker as a latent
/// transfer; that worker opens the trajectory from it (a rank that did not
/// prepare a trajectory plans its guidance when its first denoising step
/// arrives) and denoises it to the image its own preparation of the same
/// seed gives.
///
/// Both workers publish every version they write, and the test, as the engine,
/// submits each batch only after the previous result and frees no version on
/// the way: a step that writes over the bank of a published version two steps
/// back runs without waiting for a Free.
#[test]
fn a_trajectory_prepared_on_another_worker_denoises_to_the_same_image() -> anyhow::Result<()> {
    use engine::{ExecutionBatch, WorkerExecutor, WorkerGroupId};
    use trajectory::{
        IMAGE_STEPS, admission, close, conditioning, decode, denoise, prepare, prompt,
    };

    // Each worker publishes its trajectory versions to the other over
    // shared storage on their host. Each holds every stub component and
    // knows the other's: the components a published version's denoising
    // may run on.
    let workers = ["prepare", "denoise"];
    let placement = single_worker_args("prepare", QUEUE_DEPTH);
    let mut transfer =
        engine::TransferConfig::parse("prepare->denoise=posix_shm,denoise->prepare=posix_shm")?;
    for worker in workers {
        let hosts = placement.ranks.iter().map(|rank| rank.host.clone());
        transfer
            .worker_ranks
            .insert(worker.into(), placement.ranks.len() as u32);
        transfer.worker_hosts.insert(worker.into(), hosts.collect());
    }
    let spawn = |worker_id: &str| -> anyhow::Result<WorkerGroup> {
        let mut args = single_worker_args(worker_id, QUEUE_DEPTH);
        args.transfer = transfer.clone();
        args.peers = workers
            .iter()
            .map(|worker| ((*worker).to_owned(), args.components.clone()))
            .collect();
        WorkerGroup::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        workers
            .iter()
            .map(|worker| Ok((WorkerGroupId((*worker).into()), spawn(worker)?)))
            .collect::<anyhow::Result<Vec<_>>>()?,
        transfer.clone(),
    )?;

    // The reference: request 2 is prepared, denoised and decoded on the
    // denoising worker, in slot 2, cache unit 2 and latent page 2.
    let reference = admission(2, 2)?;
    let key = reference.request_key;
    completed(&mut executor, prompt(1, &reference, UnitId(2)), "denoise")?;
    completed(&mut executor, close(2, key, 2, 1), "denoise")?;
    let (reference_conditioning, batch) = conditioning(3, key, 2);
    completed(&mut executor, batch, "denoise")?;
    let (mut version, batch) = prepare(4, key, reference_conditioning, 2, 4);
    completed(&mut executor, batch, "denoise")?;
    for step in 0..IMAGE_STEPS {
        let batch_id = 5 + u64::from(step);
        let (next, batch) = denoise(
            batch_id,
            2,
            reference_conditioning,
            &version,
            2,
            6 + 2 * step,
        );
        completed(&mut executor, batch, "denoise")?;
        version = next;
    }
    let expected = image_bytes(&completed(
        &mut executor,
        decode(7, &version, 2, 10),
        "denoise",
    )?)?;

    // Request 1 has the reference's seed, so its initial latent is the
    // reference's. Only the preparing worker draws it; both hold the
    // request's conditioning.
    let moved = admission(1, 1)?;
    let key = moved.request_key;
    completed(&mut executor, prompt(8, &moved, UnitId(1)), "prepare")?;
    completed(&mut executor, close(9, key, 1, 1), "prepare")?;
    let (prepared_conditioning, batch) = conditioning(10, key, 2);
    completed(&mut executor, batch, "prepare")?;
    let (mut version, batch) = prepare(11, key, prepared_conditioning, 1, 4);
    completed(&mut executor, batch, "prepare")?;

    completed(&mut executor, prompt(12, &moved, UnitId(1)), "denoise")?;
    completed(&mut executor, close(13, key, 1, 6), "denoise")?;
    let (denoising_conditioning, batch) = conditioning(14, key, 7);
    completed(&mut executor, batch, "denoise")?;

    // The first step reads the transferred version, which opens the
    // trajectory on the denoising worker; the second reads the version the
    // first wrote there.
    for step in 0..IMAGE_STEPS {
        let batch_id = 15 + u64::from(step);
        let (next, batch) = denoise(
            batch_id,
            1,
            denoising_conditioning,
            &version,
            1,
            9 + 2 * step,
        );
        completed(&mut executor, batch, "denoise")?;
        version = next;
    }
    let image = image_bytes(&completed(
        &mut executor,
        decode(17, &version, 1, 13),
        "denoise",
    )?)?;
    assert_eq!(
        image, expected,
        "the moved trajectory decoded another image"
    );

    for (batch_id, key) in [(18, key), (19, reference.request_key)] {
        executor.submit(ExecutionBatch::new(
            batch_id,
            Vec::new(),
            vec![finish(key)],
            Vec::new(),
        ))?;
        let retired = poll_logical(&mut executor)?.context("request close did not retire")?;
        assert!(retired.done);
    }
    executor.close()?;
    Ok(())
}

/// Device products cross worker processes over CUDA VMM with their values:
/// the source publishes two requests' transition flags (one false, one
/// true), and target calls gated by them run exactly when the imported flag
/// is true, whether they read the import directly or a copy a tensor
/// transfer made into the target's own relay.
#[test]
#[ignore = "requires a CUDA device"]
fn device_products_cross_workers_over_cuda_vmm() -> anyhow::Result<()> {
    use engine::{WorkerExecutor, WorkerGroupId};

    let retire = Retire::default();
    let edges = "source->target=cuda_vmm";
    let mut executor = WorkerExecutor::try_new(
        vec![
            (
                WorkerGroupId("source".into()),
                cuda_worker(&retire, "source", edges, None)?,
            ),
            (
                WorkerGroupId("target".into()),
                cuda_worker(&retire, "target", edges, None)?,
            ),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;

    // The stub's head maps 7 to 1000 and 1001 to its image start token,
    // which these calls declare a transition: request 95's flag is false,
    // request 96's true.
    let mut flags = Vec::new();
    for (batch_id, request_id, prompt) in [(1, 95, 7), (3, 96, 1001)] {
        let admission = text_admission(request_id, 1, request_id as u32 - 94)?;
        let mut prefill = token_batch(
            batch_id,
            batch_id,
            admission.request_key,
            Some(admission),
            CallId::new(batch_id, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[prompt],
            UnitId(request_id as u32 - 94),
            0,
        );
        let flag = scalar_flag(prefill.calls[0].request_key, CallId::new(batch_id, 0), 3);
        prefill.calls[0].transition_output = Some(flag.clone());
        prefill.calls[0].sampling_state = Some(ipc::SamplingState {
            transition_token_ids: vec![STUB_IMAGE_START],
            ..Default::default()
        });
        executor.submit(place(prefill, "source", Vec::new()))?;
        let prefilled = poll_logical(&mut executor)?.context("source prefill did not complete")?;
        assert_eq!(prefilled.results[0].output.status, CallStatus::Ok);

        // The source republishes the flag as a product other workers read;
        // the target imports it over the source's one edge.
        let export = TensorRef {
            producer_call_id: CallId::new(batch_id + 1, 0),
            generation: 2,
            ..flag.clone()
        };
        let publish = product_transfer(
            flag.request_key,
            CallId::new(batch_id + 1, 0),
            "model",
            &flag,
            &export,
        );
        let mut publish = Batch::new(batch_id + 1, Vec::new(), vec![publish]);
        publish.buffer_allocations.push(BufferAllocation {
            buffer: export.buffer_id(),
            offset: 256 * batch_id,
            bytes: export.max_bytes(),
        });
        executor.submit(place(publish, "source", Vec::new()))?;
        let published = poll_logical(&mut executor)?.context("flag export did not complete")?;
        assert_eq!(published.results[0].output.status, CallStatus::Ok);
        flags.push(export);
    }

    // The same requests on the target, gated directly by their imported
    // flags.
    for (batch_id, (request_id, flag, expected)) in [
        (95, &flags[0], CallStatus::Predicated),
        (96, &flags[1], CallStatus::Ok),
    ]
    .into_iter()
    .enumerate()
    {
        let admission = text_admission(request_id, 1, request_id as u32 - 94)?;
        let status = gated_prefill(
            &mut executor,
            21 + batch_id as u64,
            admission.request_key,
            Some(admission),
            flag,
            0,
        )?;
        assert_eq!(status, expected, "request {request_id}");
    }

    // The true flag copied into the target's relay of its request gates the
    // request's next call the same way.
    let owner = flags[1].request_key;
    let copy = TensorRef {
        producer_call_id: CallId::new(23, 0),
        generation: 3,
        ..flags[1].clone()
    };
    let transfer = product_transfer(owner, CallId::new(23, 0), "model", &flags[1], &copy);
    let mut transfer = Batch::new(23, Vec::new(), vec![transfer]);
    transfer.buffer_allocations.push(BufferAllocation {
        buffer: copy.buffer_id(),
        offset: 0,
        bytes: copy.max_bytes(),
    });
    executor.submit(place(transfer, "target", Vec::new()))?;
    let copied = poll_logical(&mut executor)?.context("target transfer did not complete")?;
    assert_eq!(copied.results[0].output.status, CallStatus::Ok);
    assert_eq!(
        gated_prefill(&mut executor, 24, owner, None, &copy, 1)?,
        CallStatus::Ok
    );
    executor.close()?;
    Ok(())
}

/// A request prefilled on a worker with a BF16 cache continues on a worker
/// with an FP16 cache: its cache crosses the processes over CUDA VMM and is
/// converted into the target's representation by the cache's conversion.
#[test]
#[ignore = "requires a CUDA device"]
fn a_converting_install_copies_the_cache_another_worker_exported() -> anyhow::Result<()> {
    use engine::{WorkerExecutor, WorkerGroupId};

    let retire = Retire::default();
    let edges = "source->target=cuda_vmm";
    let mut executor = WorkerExecutor::try_new(
        vec![
            (
                WorkerGroupId("source".into()),
                cuda_worker(&retire, "source", edges, None)?,
            ),
            (
                WorkerGroupId("target".into()),
                cuda_worker(
                    &retire,
                    "target",
                    edges,
                    Some(ipc::PrefixCacheDtype::Float16),
                )?,
            ),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;
    let admission = text_admission(97, 1, 1)?;
    let key = admission.request_key;
    let prefill = token_batch(
        1,
        1,
        key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8, 9],
        UnitId(1),
        0,
    );
    let tables = prefill.block_tables.clone();
    executor.submit(place(prefill, "source", Vec::new()))?;
    let prefilled = poll_logical(&mut executor)?.context("source prefill did not complete")?;
    let exported = ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(2, 0),
        output_index: 0,
        generation: 2,
    };
    let mut export = Call::new(
        key,
        CallId::new(2, 0),
        CallKind::Transfer(TransferMode::CacheExport),
    );
    export.coordinates = coordinates_after(&prefilled.results[0].output);
    export.cache_output = Some(exported);
    export.bounds.max_transfer_bytes = 1 << 20;
    let mut publish = Batch::new(2, Vec::new(), vec![export]);
    publish.block_tables = tables;
    executor.submit(place(publish, "source", Vec::new()))?;
    let published = poll_logical(&mut executor)?.context("cache export did not complete")?;
    assert_eq!(published.results[0].output.status, CallStatus::Ok);
    let transfer = published.results[0]
        .output
        .cache_output
        .clone()
        .context("cache export has no transfer")?;
    assert_eq!(transfer.exported_extent, 3);
    assert!(transfer.tensors().all(|tensor| {
        tensor
            .locations
            .iter()
            .any(|location| matches!(location.transport, ipc::TransferTransport::CudaVmm { .. }))
    }));

    let mut install = Call::new(
        key,
        CallId::new(3, 0),
        CallKind::Transfer(TransferMode::CacheInstall),
    );
    install.cache_input = Some(exported);
    install.cache_output = Some(ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(3, 0),
        output_index: 0,
        generation: 3,
    });
    install.bounds.max_transfer_bytes = 1 << 20;
    let mut target = Batch::new(3, vec![admission], vec![install]);
    target.block_tables = vec![BlockTable {
        request_pool_idx: 1,
        group_id: 0,
        start_page: 0,
        unit_ids: vec![UnitId(6)],
        allocated_tokens: 3,
    }];
    target.new_cache_units = vec![CacheUnitAllocation {
        request_pool_idx: 1,
        group_id: 0,
        unit_ids: vec![UnitId(6)],
    }];
    executor.submit(place(target, "target", Vec::new()))?;
    let installation = poll_logical(&mut executor)?.context("cache install did not complete")?;
    assert_eq!(installation.results[0].output.status, CallStatus::Ok);
    assert_eq!(installation.results[0].output.cache_visible_len, 3);

    let mut decode = token_batch(
        4,
        4,
        key,
        None,
        CallId::new(4, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[1000],
        UnitId(6),
        3,
    );
    decode.calls[0].coordinates.logical_position = 3;
    executor.submit(place(decode, "target", Vec::new()))?;
    let decoded = poll_logical(&mut executor)?.context("target decode did not complete")?;
    assert_eq!(decoded.results[0].output.status, CallStatus::Ok);
    assert_eq!(decoded.results[0].output.committed_tokens, vec![1001]);
    assert_eq!(decoded.results[0].output.cache_visible_len, 4);
    executor.close()?;
    Ok(())
}

/// A real model's request continues on another worker from the prefix cache it
/// installed and the token the source sampled: the target's greedy
/// continuation of a prompt equals the source's token for token, since the
/// installed pages hold the source's bytes. Cache and token cross over shared
/// memory and over CUDA VMM. Over CUDA VMM the source's pages are read where
/// they lie, and the token, a relay scalar outside exportable storage, is
/// copied into the source's export pool beside its cache pool of a served
/// model's scale. `UNISERVE_KV_INSTALL_MODEL` names the checkpoint and
/// `UNISERVE_KV_INSTALL_PROMPT` the prompt's token ids (a JSON list).
#[test]
#[ignore = "requires a CUDA device and the checkpoint UNISERVE_KV_INSTALL_MODEL names"]
fn a_model_continues_greedily_from_installed_cache() -> anyhow::Result<()> {
    // One mechanism at a time: each pair of workers holds two copies of the
    // model on the device.
    for edges in ["source->target=posix_shm", "source->target=cuda_vmm"] {
        continue_greedily_from_installed_cache(edges).with_context(|| edges.to_owned())?;
    }
    Ok(())
}

/// Prefill a prompt on the source, export its cache and its sampled token to
/// the target over `edges`, install both there, and compare both workers'
/// greedy continuations, the target's starting from the imported token.
fn continue_greedily_from_installed_cache(edges: &str) -> anyhow::Result<()> {
    use engine::{WorkerExecutor, WorkerGroupId};

    let model = std::env::var("UNISERVE_KV_INSTALL_MODEL")?;
    let prompt: Vec<u32> = serde_json::from_str(&std::env::var("UNISERVE_KV_INSTALL_PROMPT")?)?;
    let retire = Retire::default();
    let spawn = |worker_id: &str| -> anyhow::Result<WorkerGroup> {
        let mut args = single_worker_args(worker_id, QUEUE_DEPTH);
        for rank in &mut args.ranks {
            rank.device = "cuda:0".into();
        }
        // The model's own block size, attention selection and component.
        args.stub = false;
        args.model = model.clone();
        args.block_size = None;
        args.max_batch_tokens = 8192;
        args.attention_backend = "auto".into();
        args.components = WorkerGroupConfig::single_component("model", 1);
        args.transfer = engine::TransferConfig::parse(edges)?;
        retire.add(&args);
        WorkerGroup::spawn(args)
    };
    let mut executor = WorkerExecutor::try_new(
        vec![
            (WorkerGroupId("source".into()), spawn("source")?),
            (WorkerGroupId("target".into()), spawn("target")?),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;
    let admission = text_admission(101, 1, 1)?;
    let key = admission.request_key;
    let n = prompt.len() as u32;
    let prefill = token_batch(
        1,
        1,
        key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &prompt,
        UnitId(1),
        0,
    );
    let tables = prefill.block_tables.clone();
    executor.submit(place(prefill, "source", Vec::new()))?;
    let prefilled = poll_logical(&mut executor)?.context("source prefill did not complete")?;
    let first = prefilled.results[0].output.committed_tokens[0];

    let exported = ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(2, 0),
        output_index: 0,
        generation: 2,
    };
    let mut export = Call::new(
        key,
        CallId::new(2, 0),
        CallKind::Transfer(TransferMode::CacheExport),
    );
    export.coordinates = coordinates_after(&prefilled.results[0].output);
    export.cache_output = Some(exported);
    export.bounds.max_transfer_bytes = 1 << 30;
    let mut publish = Batch::new(2, Vec::new(), vec![export]);
    publish.block_tables = tables;
    executor.submit(place(publish, "source", Vec::new()))?;
    let published = poll_logical(&mut executor)?.context("cache export did not complete")?;
    assert_eq!(published.results[0].output.status, CallStatus::Ok);

    let mut install = Call::new(
        key,
        CallId::new(3, 0),
        CallKind::Transfer(TransferMode::CacheInstall),
    );
    install.cache_input = Some(exported);
    install.cache_output = Some(ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(3, 0),
        output_index: 0,
        generation: 3,
    });
    install.bounds.max_transfer_bytes = 1 << 30;
    let mut target = Batch::new(3, vec![admission], vec![install]);
    target.block_tables = vec![BlockTable {
        request_pool_idx: 1,
        group_id: 0,
        start_page: 0,
        unit_ids: vec![UnitId(7)],
        allocated_tokens: n,
    }];
    target.new_cache_units = vec![CacheUnitAllocation {
        request_pool_idx: 1,
        group_id: 0,
        unit_ids: vec![UnitId(7)],
    }];
    executor.submit(place(target, "target", Vec::new()))?;
    let installed = poll_logical(&mut executor)?.context("cache install did not complete")?;
    assert_eq!(installed.results[0].output.status, CallStatus::Ok);

    // Greedy continuations of the same prefix on both workers.
    let mut tokens = std::collections::BTreeMap::new();
    let mut last = None;
    for (worker, unit, base) in [("source", 1, 10), ("target", 7, 20)] {
        let mut token = first;
        let mut sequence = Vec::new();
        for step in 0..4_u32 {
            let batch_id = u64::from(base + step);
            let (decoded, sampled) =
                decode_step(&mut executor, worker, key, batch_id, token, unit, n + step)?;
            token = decoded.committed_tokens[0];
            sequence.push(token);
            if worker == "source" {
                last = Some((token, sampled));
            }
        }
        tokens.insert(worker, sequence);
    }
    let (last, sampled) = last.context("the source decoded")?;

    // The source publishes its last token and the target copies it into its
    // own relay; one more step on each worker continues from that token, the
    // target's reading it from the device.
    let published = TensorRef {
        producer_call_id: CallId::new(14, 0),
        generation: 14 * 4 + 1,
        ..sampled.clone()
    };
    let publish = token_transfer(key, CallId::new(14, 0), "model", &sampled, &published);
    executor.submit(place(
        Batch::new(14, Vec::new(), vec![publish]),
        "source",
        Vec::new(),
    ))?;
    let exported = poll_logical(&mut executor)?.context("token export did not complete")?;
    assert_eq!(exported.results[0].output.status, CallStatus::Ok);
    let imported = TensorRef {
        producer_call_id: CallId::new(24, 0),
        generation: 24 * 4 + 1,
        ..published.clone()
    };
    let copy = token_transfer(key, CallId::new(24, 0), "model", &published, &imported);
    executor.submit(place(
        Batch::new(24, Vec::new(), vec![copy]),
        "target",
        Vec::new(),
    ))?;
    let copied = poll_logical(&mut executor)?.context("token import did not complete")?;
    assert_eq!(copied.results[0].output.status, CallStatus::Ok);

    let position = n + 4;
    let (source, _) = decode_step(&mut executor, "source", key, 15, last, 1, position)?;
    let mut decode = decode_batch(key, 25, 0, 7, position);
    decode.calls[0].input_token_ids.clear();
    decode.calls[0].predicate = Some(imported);
    executor.submit(place(decode, "target", Vec::new()))?;
    let target = poll_logical(&mut executor)?.context("decode did not complete")?;
    assert_eq!(target.results[0].output.status, CallStatus::Ok);
    for (worker, output) in [("source", &source), ("target", &target.results[0].output)] {
        if let Some(sequence) = tokens.get_mut(worker) {
            sequence.push(output.committed_tokens[0]);
        }
    }
    assert_eq!(tokens["source"], tokens["target"]);
    executor.close()?;
    Ok(())
}

/// A one-token decode of `key` at `position` on `worker`, in batch and call
/// `batch_id`, over the request's unit `unit`.
fn decode_batch(key: RequestKey, batch_id: u64, token: u32, unit: u32, position: u32) -> Batch {
    let mut decode = token_batch(
        batch_id,
        batch_id,
        key,
        None,
        CallId::new(batch_id, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[token],
        UnitId(unit),
        position,
    );
    decode.calls[0].coordinates.logical_position = position;
    decode
}

/// Run [`decode_batch`] on `worker` and return its output and the token
/// product it published.
fn decode_step(
    executor: &mut engine::WorkerExecutor,
    worker: &str,
    key: RequestKey,
    batch_id: u64,
    token: u32,
    unit: u32,
    position: u32,
) -> anyhow::Result<(ipc::CallOutput, TensorRef)> {
    let decode = decode_batch(key, batch_id, token, unit, position);
    let sampled = decode.calls[0]
        .token_output
        .clone()
        .context("a decode samples a token")?;
    executor.submit(place(decode, worker, Vec::new()))?;
    let decoded = poll_logical(executor)?.context("decode did not complete")?;
    assert_eq!(decoded.results[0].output.status, CallStatus::Ok);
    Ok((decoded.results[0].output.clone(), sampled))
}

/// Launch arguments of a one-rank stub worker on the first visible CUDA
/// device, with a cache of `prefix_cache_dtype` (the model's dtype by default)
/// and the transfer `edges`.
fn cuda_worker(
    retire: &Retire,
    worker_id: &str,
    edges: &str,
    prefix_cache_dtype: Option<ipc::PrefixCacheDtype>,
) -> anyhow::Result<WorkerGroup> {
    let mut args = single_worker_args(worker_id, QUEUE_DEPTH);
    for rank in &mut args.ranks {
        rank.device = "cuda:0".into();
    }
    args.kv_cache_dtype = prefix_cache_dtype;
    args.transfer = engine::TransferConfig::parse(edges)?;
    retire.add(&args);
    WorkerGroup::spawn(args)
}

/// The deployments of a test's CUDA workers, retired at the host's Storage
/// Server once the test's workers are gone: their device memory outlives
/// them there, and no later launch claims a test's deployment again. A test
/// declares it before its workers, so that it drops after them.
#[derive(Default)]
struct Retire(std::sync::Mutex<Vec<String>>);

impl Retire {
    fn add(&self, args: &WorkerProcessArgs) {
        self.0.lock().unwrap().push(args.deployment.clone());
    }
}

impl Drop for Retire {
    fn drop(&mut self) {
        let socket = storage::socket::default_socket();
        for deployment in self.0.lock().unwrap().drain(..) {
            // The server notices the head's disconnect on its next turn.
            for _ in 0..500 {
                let retired = storage::Client::connect(&socket, storage::Role::Operator)
                    .and_then(|mut client| client.retire(deployment.clone()));
                if retired.is_ok() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// A one-element U8 flag that `producer` publishes as output `index`.
fn scalar_flag(request_key: RequestKey, producer: CallId, index: u16) -> TensorRef {
    TensorRef {
        request_key,
        producer_call_id: producer,
        output_index: index,
        generation: 1,
        dtype: DType::U8,
        shape_bound: ShapeBound::default(),
    }
}

/// Prefill one prompt token of `key` on the target after its first
/// `prefix` tokens, gated by `flag`, admitting the request with
/// `admission` when given; returns the call's status.
fn gated_prefill(
    executor: &mut engine::WorkerExecutor,
    batch_id: u64,
    key: RequestKey,
    admission: Option<NewRequest>,
    flag: &TensorRef,
    prefix: u32,
) -> anyhow::Result<CallStatus> {
    let slot = admission
        .as_ref()
        .map_or(2, |admission| admission.request_pool_idx);
    let mut prefill = token_batch(
        batch_id,
        batch_id,
        key,
        admission,
        CallId::new(batch_id, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7],
        UnitId(slot + 10),
        prefix,
    );
    prefill.calls[0].predicate = Some(flag.clone());
    prefill.block_tables[0].request_pool_idx = slot;
    prefill.forward.request_pool_indices = vec![slot];
    for allocation in &mut prefill.new_cache_units {
        allocation.request_pool_idx = slot;
    }
    executor.submit(place(prefill, "target", Vec::new()))?;
    let result = poll_logical(executor)?.context("gated prefill did not complete")?;
    let output = &result.results[0].output;
    if output.status == CallStatus::Ok {
        assert_eq!(output.committed_tokens, vec![1000]);
    }
    Ok(output.status)
}

/// Environment of a rank process `media_worker` serves: the launch arguments
/// (JSON), the worker info template (a path), the media case and the path
/// that receives the published media name.
const MEDIA_WORKER_ARGS: &str = "UNISERVE_TEST_MEDIA_WORKER_ARGS";
const MEDIA_WORKER_INFO: &str = "UNISERVE_TEST_MEDIA_WORKER_INFO";
const MEDIA_WORKER_CASE: &str = "UNISERVE_TEST_MEDIA_RESPONSE";
const MEDIA_WORKER_NAME: &str = "UNISERVE_TEST_MEDIA_NAME";
/// The bytes a `media_worker` process publishes.
const MEDIA_PAYLOAD: &[u8] = b"generated media content";

#[test]
fn media_storage_is_owned_through_worker_result_validation() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // The ranks are `media_worker` processes speaking the rank protocol. Each
    // answers Info with a real stub group's worker info, completes every
    // call it receives and attaches a shared-memory media payload to a
    // completion, selected by case: a valid payload from rank 0
    // ("retained"), a payload on a call identity the batch never carried
    // ("unknown-call"), a payload from rank 1 ("rank-output"), and a payload
    // declaring one byte more than it stores ("short-storage").
    let template = tempfile::tempdir()?;
    let info_path = template.path().join("info.json");
    {
        let mut worker = spawn_worker_group()?;
        std::fs::write(&info_path, serde_json::to_vec(worker.info())?)?;
        worker.close()?;
    }
    let executable = serde_json::to_string(&std::env::current_exe()?)?;
    for case in ["retained", "unknown-call", "rank-output", "short-storage"] {
        let directory = tempfile::tempdir()?;
        let wrapper = directory.path().join("worker");
        let name_path = directory.path().join("media-name");
        // `WorkerGroup` runs `<executable> <args>`, so the wrapper runs
        // this test binary's `media_worker` with those arguments.
        std::fs::write(
            &wrapper,
            format!(
                "#!/usr/bin/env python3\nimport json, os, sys\nenv = dict(os.environ, {MEDIA_WORKER_ARGS}=json.dumps(sys.argv[1:]), {MEDIA_WORKER_INFO}={info}, {MEDIA_WORKER_CASE}={case:?}, {MEDIA_WORKER_NAME}={name})\nos.execve({executable}, [{executable}, '--exact', 'multiproc_ipc::media_worker', '--ignored', '--nocapture', '--test-threads', '1'], env)\n",
                info = serde_json::to_string(&info_path)?,
                name = serde_json::to_string(&name_path)?,
            ),
        )?;
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
        let mut args = worker_group_args();
        args.executable = wrapper;
        let mut worker = WorkerGroup::spawn(args)?;
        let admission = text_admission(81, 1, 1)?;
        let batch = token_batch(
            1,
            1,
            admission.request_key,
            Some(admission),
            CallId::new(1, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[7, 8],
            UnitId(1),
            0,
        );
        worker.submit_placed(batch)?;

        // A correlation or rank-ownership failure may fail the poll and must
        // deliver no result. A short extent still delivers the call's
        // result, with its `media` failed.
        let rejected = matches!(case, "unknown-call" | "rank-output");
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut terminal = false;
        let mut results = Vec::new();
        while std::time::Instant::now() < deadline && !terminal {
            match worker.poll_batch(Duration::from_millis(100)) {
                Ok(Some(report)) => {
                    terminal = true;
                    results.extend(report.results);
                }
                Ok(None) => {}
                Err(error) => {
                    anyhow::ensure!(
                        rejected,
                        "unexpected media result failure in {case}: {error:#}"
                    );
                    terminal = true;
                }
            }
        }
        assert!(terminal, "media result did not retire in {case}");
        let published_name = std::fs::read_to_string(&name_path)?;
        // The receiver claims the POSIX name even when correlation, rank
        // ownership, or extent validation rejects the result. Retained
        // mappings keep the bytes.
        assert!(
            // SAFETY: the rank handed off an immutable, completed export.
            unsafe { ipc::SharedMedia::open(published_name.trim(), 1) }.is_err(),
            "unclaimed media in {case}"
        );
        if rejected {
            assert!(
                results.is_empty(),
                "invalid rank output was delivered in {case}"
            );
        } else {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].output.status, CallStatus::Ok);
            if case == "short-storage" {
                assert!(results[0].media.is_err());
            } else {
                let media = std::sync::Arc::clone(
                    results[0]
                        .media
                        .as_ref()
                        .map_err(|error| anyhow::anyhow!(error.clone()))?
                        .as_ref()
                        .context("media result has no storage")?,
                );
                // The claimed mapping outlives both the rank group and the
                // result that delivered it.
                worker.close()?;
                drop(results);
                assert_eq!(media.as_bytes(), MEDIA_PAYLOAD);
                continue;
            }
        }
        worker.close()?;
    }
    Ok(())
}

/// A rank process of `media_storage_is_owned_through_worker_result_validation`:
/// it registers its channel endpoint like a worker, answers Info with
/// the template worker info under its own identity, completes every call it
/// receives as the stub's prefill of a request's first prompt tokens would,
/// and attaches the case's media payload to a completion.
#[test]
#[ignore = "runs as a rank process of the media result test"]
fn media_worker() -> anyhow::Result<()> {
    use ipc::{
        ArtifactHandle, BatchOutput, CallOutput, Channel, LaunchDescriptor, MediaOutput,
        Registration, WorkerRequest, WorkerResponse,
    };

    let Ok(arguments) = std::env::var(MEDIA_WORKER_ARGS) else {
        return Ok(());
    };
    let arguments: Vec<String> = serde_json::from_str(&arguments)?;
    let argument = |name: &str| {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|index| arguments.get(index + 1))
            .with_context(|| format!("a rank launch names {name}"))
    };

    // Register with the engine as the rank this process was started as, and
    // take the launch descriptor it answers with.
    let mut server = Channel::connect(argument("--engine")?)?;
    server.send(&Registration::Worker {
        worker_id: argument("--worker-id")?.clone(),
        rank: argument("--rank")?.parse()?,
        generation: argument("--generation")?.parse()?,
        version: ipc::VERSION.into(),
    })?;
    let descriptor: LaunchDescriptor = loop {
        if let Some(descriptor) = server.try_recv()? {
            break descriptor;
        }
        server.wait(Duration::from_secs(60))?;
    };
    let case = std::env::var(MEDIA_WORKER_CASE)?;
    let mut info: ipc::WorkerInfo =
        serde_json::from_slice(&std::fs::read(std::env::var(MEDIA_WORKER_INFO)?)?)?;
    info.endpoint.address_space = format!("rust:{}", std::process::id());
    info.endpoint.incarnation = format!("media-rank-{}-{}", descriptor.rank, std::process::id());

    // Media attaches to rank 1's completion only in "rank-output".
    let media_rank = u32::from(case == "rank-output");
    loop {
        let request = loop {
            if let Some(request) = server.try_recv::<WorkerRequest>()? {
                break request;
            }
            server.wait(Duration::from_secs(1))?;
        };
        let response = match request {
            WorkerRequest::Info { message_id } => WorkerResponse::Info {
                message_id,
                info: info.clone(),
            },
            WorkerRequest::Close { message_id } => {
                server.send(&WorkerResponse::Ok { message_id })?;
                return Ok(());
            }
            WorkerRequest::Promote { message_id } => WorkerResponse::Ok { message_id },
            WorkerRequest::Hook { message_id, .. } => WorkerResponse::Error {
                message_id,
                error: ipc::WorkerResponseError {
                    message: "this worker runs no hooks".into(),
                    code: None,
                    fatal: false,
                    phase: None,
                    route: None,
                    calls: Vec::new(),
                },
            },
            WorkerRequest::Submit { message_id, batch } => {
                let mut completions: Vec<CallOutput> = batch
                    .calls
                    .iter()
                    .map(|call| {
                        let length = call.input_token_ids.len() as u32;
                        CallOutput {
                            request_key: call.request_key,
                            call_id: call.call_id,
                            status: CallStatus::Ok,
                            error_code: None,
                            error_message: None,
                            timing_counters: Default::default(),
                            cache_visible_len: call.coordinates.cache_visible_len + length,
                            cache_computed_len: call.coordinates.cache_computed_len + length,
                            step_index: 0,
                            committed_tokens: vec![1000],
                            token_logprobs: Vec::new(),
                            prompt_logprobs: Vec::new(),
                            candidate_logprobs: Vec::new(),
                            finish_flags: Default::default(),
                            media_output: None,
                            cache_output: None,
                            draft_token_ids: Vec::new(),
                        }
                    })
                    .collect();
                if descriptor.rank == media_rank
                    && let Some(completion) = completions.first_mut()
                {
                    let locator = ipc::MediaSource::publish(MEDIA_PAYLOAD)?.into_locator();
                    std::fs::write(std::env::var(MEDIA_WORKER_NAME)?, &locator.name)?;
                    completion.media_output = Some(MediaOutput {
                        handle: ArtifactHandle::PosixShm { name: locator.name },
                        bytes: locator.bytes + u64::from(case == "short-storage"),
                    });
                    if case == "unknown-call" {
                        completion.call_id.request_index = 1000;
                    }
                }
                WorkerResponse::Result {
                    message_id,
                    result: BatchOutput {
                        batch_id: batch.batch_id,
                        completions,
                        products: Vec::new(),
                        worker_exec_us: None,
                        forward_stats: None,
                    },
                }
            }
        };
        server.send(&response)?;
    }
}

/// An external input whose producer failed fails the calls that read it,
/// as a completed batch, and the group keeps serving.
#[test]
fn a_failed_external_input_fails_the_calls_reading_it() -> anyhow::Result<()> {
    let mut args = worker_group_args();
    // The segment stands in for a product that rank 0 of this worker
    // published over shared storage.
    args.transfer = engine::TransferConfig::parse("worker->worker=posix_shm")?;
    let mut executor = WorkerGroup::spawn(args)?;
    let publisher = executor.info().endpoint.clone();

    let admission = text_admission(34, 1, 1)?;
    let mut gated = token_batch(
        1,
        1,
        admission.request_key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[6],
        UnitId(1),
        0,
    );
    let predicate = TensorRef {
        request_key: admission.request_key,
        producer_call_id: CallId::new(91, 0),
        output_index: 0,
        generation: 91,
        dtype: DType::U8,
        shape_bound: ShapeBound::default(),
    };
    let export = SlowShmExport::start()?;
    gated.calls[0].predicate = Some(predicate.clone());
    gated.input_products.push(ipc::TensorExport {
        product: predicate,
        value: export.descriptor(&publisher)?,
    });
    executor.submit_placed(gated)?;
    export.fail()?;
    let failed = executor
        .poll_batch(Duration::from_secs(30))?
        .context("the gated batch did not complete")?;
    assert_eq!(failed.batch_id, 1);
    assert_eq!(failed.results[0].output.status, CallStatus::Error);
    assert!(failed.results[0].output.committed_tokens.is_empty());

    let independent = text_admission(35, 1, 2)?;
    let independent = token_batch(
        2,
        2,
        independent.request_key,
        Some(independent),
        CallId::new(2, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[9],
        UnitId(2),
        0,
    );
    let served = execute(&mut executor, independent)?;
    assert_eq!(served.results[0].output.status, CallStatus::Ok);
    assert_eq!(served.results[0].output.committed_tokens, vec![1000]);
    executor.close()?;
    Ok(())
}

/// The components the stub model declares, each on its own rank of a
/// two-rank group: the language backbone on rank 1 and the patch encoder on
/// rank 0.
fn split_components() -> std::collections::BTreeMap<String, ipc::ComponentConfig> {
    [
        (
            "model".into(),
            ipc::ComponentConfig::parallel(vec![1], Default::default()),
        ),
        (
            "vision_encoder".into(),
            ipc::ComponentConfig::parallel(vec![0], Default::default()),
        ),
    ]
    .into_iter()
    .collect()
}

/// Launch arguments for a one-worker stub group `worker_id` that holds
/// every stub component, with `queue_depth` batches in flight.
fn single_worker_args(worker_id: &str, queue_depth: usize) -> WorkerProcessArgs {
    let mut args = worker_group_args();
    let components = std::collections::BTreeMap::from([
        (
            "model".into(),
            ipc::ComponentConfig::parallel(vec![0], Default::default()),
        ),
        (
            "vision_encoder".into(),
            ipc::ComponentConfig::parallel(vec![0], Default::default()),
        ),
    ]);
    let binding =
        WorkerGroupConfig::placed(&["localhost".to_owned()], "cpu", 1, queue_depth, components);
    args.ranks = binding.ranks.clone();
    args.components = binding.components;
    args.queue_depth = queue_depth;
    args.worker_id = worker_id.into();
    args
}

/// The process id of rank `rank` of group `worker_id`, which the daemon
/// started.
fn worker_pid(worker_id: &str, rank: u32) -> anyhow::Result<i32> {
    crate::support::group_processes(worker_id)
        .into_iter()
        .find(|(_, launched, _)| *launched == rank)
        .map(|(pid, _, _)| pid as i32)
        .with_context(|| format!("rank {rank} of worker group {worker_id} was not found"))
}

/// Lifts a single-call worker batch into the logical form the executor
/// accepts, placing it on `worker` with the batch's tables, pages, buffers
/// and the external `inputs` it reads.
fn place(mut batch: Batch, worker: &str, inputs: Vec<ipc::TensorExport>) -> engine::ExecutionBatch {
    use engine::{ExecutionBatch, RequestPlacement, WorkerGroupId};

    assert_eq!(batch.calls.len(), 1);
    let call = batch.calls.remove(0);
    let mut inputs = inputs;
    inputs.extend(batch.input_products);
    ExecutionBatch::new(
        batch.batch_id,
        vec![(
            call,
            RequestPlacement {
                worker: WorkerGroupId(worker.into()),
                request_pool_idx: None,
                block_tables: batch.block_tables,
                new_cache_units: batch.new_cache_units,
                forward: batch.forward,
                latent: batch.latent_params.into_iter().next(),
                decode: batch.decode_ranges.into_iter().next(),
                buffers: on_every_rank(batch.buffer_allocations),
                readers: Default::default(),
                page_loads: Vec::new(),
            },
        )],
        batch.commands,
        inputs,
    )
}

/// A text admission that also declares one input image, as the engine does
/// for a request whose vision encoding call encodes an image it carries.
fn image_admission(
    request_id: u64,
    request_epoch: u64,
    request_pool_idx: u32,
) -> anyhow::Result<NewRequest> {
    let mut admission = text_admission(request_id, request_epoch, request_pool_idx)?;
    admission.input_images = 1;
    Ok(admission)
}

/// The coordinates a call entering after `output` executes at, for a request
/// whose completions advance its logical position with its visible cache.
fn coordinates_after(output: &ipc::CallOutput) -> CallCoordinates {
    CallCoordinates {
        logical_position: output.cache_visible_len,
        cache_visible_len: output.cache_visible_len,
        cache_computed_len: output.cache_computed_len,
    }
}

/// A one-element product of `dtype` that `producer` publishes.
fn scalar_product(
    request_key: RequestKey,
    producer: CallId,
    generation: u32,
    dtype: DType,
) -> TensorRef {
    TensorRef {
        request_key,
        producer_call_id: producer,
        output_index: 0,
        generation,
        dtype,
        shape_bound: ShapeBound {
            dims: vec![ipc::DimBound::Static(1)],
        },
    }
}

/// The BF16 patch-encoder feature `producer` publishes, of at most 4096
/// elements.
fn feature_ref(request_key: RequestKey, producer: CallId, generation: u32) -> TensorRef {
    TensorRef {
        request_key,
        producer_call_id: producer,
        output_index: 0,
        generation,
        dtype: DType::BF16,
        shape_bound: ShapeBound {
            dims: vec![ipc::DimBound::Device { max: 4096 }],
        },
    }
}

/// A vision encoding call of batch `batch_id` on the patch encoder that
/// encodes a base64 16x16 RGB PNG into `feature`.
fn encode_call(request_key: RequestKey, batch_id: u64, feature: &TensorRef) -> Call {
    const IMAGE: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGN0SGhgIAUwkaR6VMOohiGlAQCjvQFA6eri4wAAAABJRU5ErkJggg==";
    let mut call = Call::new(
        request_key,
        CallId::new(batch_id, 0),
        CallKind::Media(MediaCall::VisionEncoding),
    );
    call.component = "vision_encoder".into();
    call.encoder_output = Some(feature.clone());
    call.input_image = Some(IMAGE.into());
    call.bounds = Bounds {
        max_tokens: 64,
        ..Bounds::default()
    };
    call
}

/// A tensor transfer on `component` that copies the relayed token `input`
/// into the relayed token `output`.
fn token_transfer(
    request_key: RequestKey,
    call_id: CallId,
    component: &str,
    input: &TensorRef,
    output: &TensorRef,
) -> Call {
    let mut call = Call::new(
        request_key,
        call_id,
        CallKind::Transfer(TransferMode::Tensor),
    );
    call.component = component.into();
    call.token_input = Some(input.clone());
    call.token_output = Some(output.clone());
    call.bounds = Bounds {
        max_transfer_bytes: input.max_bytes(),
        ..Bounds::default()
    };
    call
}

/// A tensor transfer on `component` that copies the generic product
/// `input` into the generic product `output`.
fn product_transfer(
    request_key: RequestKey,
    call_id: CallId,
    component: &str,
    input: &TensorRef,
    output: &TensorRef,
) -> Call {
    let mut call = Call::new(
        request_key,
        call_id,
        CallKind::Transfer(TransferMode::Tensor),
    );
    call.component = component.into();
    call.inputs = vec![input.clone()];
    call.outputs = vec![output.clone()];
    call.bounds = Bounds {
        max_transfer_bytes: input.max_bytes(),
        ..Bounds::default()
    };
    call
}

/// An external shared-storage publisher the test controls.
///
/// The reader waits on a readiness word, written after the payload.
struct SlowShmExport {
    name: std::ffi::CString,
    endpoint: String,
    file: std::fs::File,
    published: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Byte layout of a shared-storage export's header: a 512-byte header whose
/// first word is the readiness state.
const SEGMENT_HEADER_BYTES: u64 = 512;
const SEGMENT_STATE_OFFSET: u64 = 0;
const SEGMENT_READY: u32 = 1;
const SEGMENT_FAILED: u32 = 2;

impl SlowShmExport {
    fn start() -> anyhow::Result<Self> {
        use std::os::fd::FromRawFd as _;

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let name = format!("uniserve-transfer-{}-{nonce}", std::process::id());
        let endpoint = format!("{name}-readers");
        let name = std::ffi::CString::new(format!("/{name}"))?;
        // SAFETY: name is NUL-terminated; a successful call returns a fresh fd.
        let descriptor = unsafe {
            libc::shm_open(
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
                0o600,
            )
        };
        if descriptor < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let export = Self {
            name,
            endpoint,
            // SAFETY: this File takes sole ownership of the descriptor.
            file: unsafe { std::fs::File::from_raw_fd(descriptor) },
            published: Default::default(),
        };
        // A one-byte payload after the header. The file starts zero-filled,
        // which leaves the readiness word pending and every acknowledgment
        // word unclaimed.
        export.file.set_len(SEGMENT_HEADER_BYTES + 1)?;
        Ok(export)
    }

    fn publish(&self) -> anyhow::Result<()> {
        use std::os::unix::fs::FileExt as _;

        // The payload lands before the readiness word; each write is a system
        // call, which orders them for a reader on another core.
        self.file.write_all_at(&[1], SEGMENT_HEADER_BYTES)?;
        self.file
            .write_all_at(&SEGMENT_READY.to_ne_bytes(), SEGMENT_STATE_OFFSET)?;
        self.published.store(true, Ordering::Release);
        Ok(())
    }

    /// Mark the segment failed: its producer will never write it.
    fn fail(&self) -> anyhow::Result<()> {
        use std::os::unix::fs::FileExt as _;

        self.file
            .write_all_at(&SEGMENT_FAILED.to_ne_bytes(), SEGMENT_STATE_OFFSET)?;
        Ok(())
    }

    /// Describes the segment as a product published by `source`, a rank the
    /// engine launched.
    fn descriptor(&self, source: &ipc::WorkerEndpoint) -> anyhow::Result<ipc::TransferHandle> {
        use ipc::{Locator, TensorTransfer, TransferHandle, TransferTransport};

        Ok(TransferHandle::DeviceProduct {
            height: 0,
            width: 0,
            value_range: String::new(),
            tensor: TensorTransfer {
                shape: vec![1],
                locations: vec![Locator {
                    source: source.clone(),
                    transport: TransferTransport::PosixShm {
                        endpoint: self.endpoint.clone(),
                        name: self.name.to_str()?.to_owned(),
                    },
                    nbytes: 1,
                    dtype: "uint8".to_owned(),
                    shape: vec![1],
                    offset: vec![0],
                    device: "cpu".to_owned(),
                }],
            },
        })
    }
}

impl Drop for SlowShmExport {
    fn drop(&mut self) {
        // SAFETY: the name remains valid and identifies this fixture's segment.
        unsafe {
            libc::shm_unlink(self.name.as_ptr());
        }
    }
}
