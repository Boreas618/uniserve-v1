//! Multiprocess launch, request lifecycle, close and rank recovery.
//!
//! These tests launch real rank processes (`uniserve_models.worker` with the
//! weightless stub model on CPU) and drive them through the engine's
//! `WorkerGroup` or `EngineCore`, or through a raw `ClientEndpoint` when the
//! worker's own protocol handling is under test. The batches carry lifecycle
//! commands (Start, Finish, Free) only: they exercise registration, startup,
//! Info, admission, retirement, close and signals, whatever numerical
//! runtimes the rank serves. The interpreter comes from
//! `UNISERVE_WORKER_PYTHON` or the repository's `.venv/bin/python`. Several
//! tests replace that interpreter with a small wrapper script that records a
//! rank's process id, exits a chosen rank, rewrites a relaunched rank's
//! descriptor, reports a registration with an unsupported channel transport,
//! or holds a rank's exit until its peers have closed.
//!
//! Faults are injected at the OS boundary: `/proc/thread-self/children` and
//! `/proc/<pid>/cmdline` locate a rank process, and `PausedProcess` stops,
//! resumes or kills it with signals. These are Linux interfaces, and the file
//! compiles only on Linux.

#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::remove_file;
use std::os::unix::process::ExitStatusExt as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use anyhow::Context as _;
use uniserve_core::{
    DiffusionRequest, DiffusionSamplingParams, EngineCoreOutput, Request, RequestId, RuntimeFamily,
    SamplingParams,
};
use uniserve_engine::{
    CallReaders, EngineConfig, EngineCore, WorkerConfig, WorkerFailure, WorkerGroup,
    WorkerProcessArgs,
};
use uniserve_ipc::{
    ArRequestParams, Batch, BatchCommand, LaunchDescriptor, NewRequest, RequestKey,
};

const WORLD_SIZE: usize = 2;
const QUEUE_DEPTH: usize = 2;
/// Held across `EngineCore::new` in
/// `unsupported_media_is_rejected_without_stopping_the_engine` and across the
/// rank-group launch in `a_lost_rank_relaunches_its_group_and_an_incomplete_launch_fails`,
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

/// Starts the stub worker with `descriptor` as a raw process, outside any
/// group.
fn spawn_raw_worker(
    descriptor: &LaunchDescriptor,
    directory: &Path,
) -> anyhow::Result<std::process::Child> {
    let path = directory.join("launch.json");
    std::fs::write(&path, descriptor.to_json()?)?;
    Ok(std::process::Command::new(worker_python())
        .args(["-m", "uniserve_models.worker"])
        .arg("--launch-descriptor")
        .arg(&path)
        .spawn()?)
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
    use uniserve_ipc::{ClientEndpoint, WorkerRequest, WorkerResponse};

    // Each launch names its own endpoint and reports it here, so a graceful
    // exit is observable as the next launch registering and serving again.
    for _ in 0..2 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let directory = tempfile::tempdir()?;
        let mut child = spawn_raw_worker(
            &stub_launch_descriptor(&listener.local_addr()?.to_string()),
            directory.path(),
        )?;
        let result = (|| -> anyhow::Result<()> {
            let client =
                ClientEndpoint::connect(&accept_reported_endpoint(&listener)?, 1 << 20, 8)?;
            let request = |mut request: WorkerRequest, message_id| {
                request.set_call_id(Some(message_id));
                request
            };

            // The rank describes itself once warmup is done.
            let info = client.send_request(&request(WorkerRequest::info(), 1))?;
            let info = client
                .recv_response_timeout(&info, Duration::from_secs(60))?
                .context("Info was not answered")?
                .decode_response()?;
            let WorkerResponse::Info { info, .. } = info else {
                anyhow::bail!("Info failed: {info:?}");
            };
            info.validate()?;
            anyhow::ensure!(info.queue_depth == 8, "queue depth changed");
            anyhow::ensure!(info.kv_cache.is_some(), "the stub reports no KV pool");

            let first = text_admission(51, 1, 1)?;
            let first_key = first.request_key;
            let started = lifecycle_batch(1, vec![first], Vec::new());
            let initial =
                client.send_request(&request(WorkerRequest::submit(started.clone()), 2))?;
            let first = client
                .recv_response_timeout(&initial, Duration::from_secs(30))?
                .context("initial submission did not complete")?
                .decode_response()?;
            let WorkerResponse::Result { result: first, .. } = first else {
                anyhow::bail!("initial submission failed: {first:?}");
            };
            anyhow::ensure!(first.batch_id == 1, "result names another batch");
            anyhow::ensure!(
                first.completions.is_empty(),
                "a lifecycle batch completed calls"
            );
            drop(initial);

            // A second request starts while the first finishes. The rank's own
            // admission answers the two resubmissions of batch 1: a batch id
            // that does not exceed the last admitted one is refused with
            // `InvalidDescriptor`. Close is sent behind all of them.
            let other =
                lifecycle_batch(2, vec![text_admission(52, 1, 2)?], vec![finish(first_key)]);
            let accepted = client.send_request(&request(WorkerRequest::submit(other), 3))?;
            let rejected =
                client.send_request(&request(WorkerRequest::submit(started.clone()), 4))?;
            let duplicate = client.send_request(&request(WorkerRequest::submit(started), 5))?;
            let close = client.send_request(&request(WorkerRequest::close(), 6))?;
            let closed = client
                .recv_response_timeout(&close, Duration::from_secs(30))?
                .context("Close did not drain accepted work")?
                .decode_response()?;
            anyhow::ensure!(
                closed
                    == WorkerResponse::Ok {
                        message_id: Some(6)
                    },
                "Close failed: {closed:?}"
            );

            // Receiving Close first must still leave every accepted response available.
            let receive = |pending| -> anyhow::Result<WorkerResponse> {
                Ok(client
                    .try_recv_response(pending)?
                    .context("Close acknowledged before accepted response delivery")?
                    .decode_response()?)
            };
            let WorkerResponse::Result { result: other, .. } = receive(&accepted)? else {
                anyhow::bail!("independent submission did not return a result");
            };
            anyhow::ensure!(other.batch_id == 2, "independent batch identity changed");
            for pending in [&rejected, &duplicate] {
                let WorkerResponse::Error { error, .. } = receive(pending)? else {
                    anyhow::bail!("duplicate submission was accepted");
                };
                anyhow::ensure!(
                    error.code.as_deref() == Some("InvalidDescriptor"),
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
fn a_serving_rank_leaves_interrupt_and_termination_to_its_head() -> anyhow::Result<()> {
    use uniserve_ipc::{ClientEndpoint, WorkerRequest, WorkerResponse};

    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let directory = tempfile::tempdir()?;
    let mut child = spawn_raw_worker(
        &stub_launch_descriptor(&listener.local_addr()?.to_string()),
        directory.path(),
    )?;
    let result = (|| -> anyhow::Result<()> {
        let client = ClientEndpoint::connect(&accept_reported_endpoint(&listener)?, 1 << 20, 8)?;
        let request = |mut request: WorkerRequest, message_id| {
            request.set_call_id(Some(message_id));
            request
        };
        let serve = |batch_id: u64, request_id: u64, slot: u32| -> anyhow::Result<()> {
            let batch = lifecycle_batch(
                batch_id,
                vec![text_admission(request_id, 1, slot)?],
                Vec::new(),
            );
            let pending = client.send_request(&request(WorkerRequest::submit(batch), batch_id))?;
            let response = client
                .recv_response_timeout(&pending, Duration::from_secs(60))?
                .context("submission did not complete")?
                .decode_response()?;
            anyhow::ensure!(
                matches!(response, WorkerResponse::Result { .. }),
                "submission failed: {response:?}"
            );
            Ok(())
        };

        // A result means the rank serves: warmup has ended.
        serve(1, 61, 1)?;

        // A terminal's Ctrl-C and a service manager's stop reach every process
        // of the job; the rank keeps serving until its head closes it.
        let pid = libc::pid_t::try_from(child.id())?;
        for signal in [libc::SIGINT, libc::SIGTERM] {
            anyhow::ensure!(unsafe { libc::kill(pid, signal) } == 0);
        }
        serve(2, 62, 2)?;

        let close = client.send_request(&request(WorkerRequest::close(), 3))?;
        let closed = client
            .recv_response_timeout(&close, Duration::from_secs(30))?
            .context("Close was not acknowledged")?
            .decode_response()?;
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
fn a_starting_rank_stops_on_interrupt_and_dies_of_termination() -> anyhow::Result<()> {
    for (signal, expect_interrupt) in [(libc::SIGINT, true), (libc::SIGTERM, false)] {
        // Rank 1 of a two-rank group whose rank 0 never starts: the rank
        // registers, then waits in its group's rendezvous for a peer that
        // never answers, so it stays in startup until the signal.
        let registration = std::net::TcpListener::bind("127.0.0.1:0")?;
        let rendezvous = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut descriptor = stub_launch_descriptor(&registration.local_addr()?.to_string());
        descriptor.rank = 1;
        descriptor.local_rank = 1;
        descriptor.world_size = 2;
        descriptor.acknowledgment_slot = 1;
        descriptor.host_slots = vec![0, 1];
        descriptor.components = WorkerConfig::single_component("model", 2);
        descriptor.rendezvous_address = Some(rendezvous.local_addr()?.to_string());
        let directory = tempfile::tempdir()?;
        let mut child = spawn_raw_worker(&descriptor, directory.path())?;
        let result = (|| -> anyhow::Result<()> {
            accept_reported_endpoint(&registration)?;
            // The rank connects to the rendezvous once it has described its
            // model; it is then blocked waiting for the address table.
            rendezvous.set_nonblocking(false)?;
            let (_peer, _) = rendezvous.accept()?;
            let pid = libc::pid_t::try_from(child.id())?;
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

/// A two-rank group reports its launch in its Info, retires finished
/// requests, refuses a repeated batch, and once closed reports nothing and
/// refuses work.
#[test]
fn a_rank_group_reports_its_launch_and_refuses_work_after_close() -> anyhow::Result<()> {
    let mut executor = spawn_rank_group()?;
    let info = executor.info().clone();
    assert_eq!(info.endpoint.rank, 0);
    assert_eq!(info.world_size, WORLD_SIZE as u32);
    assert_eq!(info.max_batch_calls, 256);
    assert_eq!(info.max_batch_tokens, 256);
    assert_eq!(info.queue_depth as usize, QUEUE_DEPTH);
    assert_eq!(info.model_dtype, "bfloat16");
    assert!(info.kv_cache.is_some());

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
        executor.submit_batch(retire, &CallReaders::new()),
        Err(uniserve_engine::BatchSubmitError::Failed(_))
    ));

    // A closed group reports no results, refuses submissions, and tolerates a
    // second close.
    executor.close()?;
    assert!(!executor.is_ready());
    assert!(executor.poll_batch(Duration::ZERO)?.is_none());
    assert!(matches!(
        executor.submit_batch(
            lifecycle_batch(100, Vec::new(), vec![finish(first_key)]),
            &CallReaders::new()
        ),
        Err(uniserve_engine::BatchSubmitError::Failed(_))
    ));
    executor.close()?;
    Ok(())
}

/// Kills rank 1 of a two-rank group while a batch is outstanding, and requires
/// the group to report its resident requests lost, relaunch every rank under a
/// new incarnation, and serve again. Then requires a group whose rank 1 exits
/// during startup to fail its launch.
#[test]
fn a_lost_rank_relaunches_its_group_and_an_incomplete_launch_fails() -> anyhow::Result<()> {
    // Under the launch lock, snapshot this thread's children, launch the
    // group, and take rank 1 from the children that were not there before.
    let launch_guard = CHILD_LAUNCH_ENV_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("child launch lock poisoned"))?;
    let child_ids = || -> anyhow::Result<Vec<u32>> {
        Ok(std::fs::read_to_string("/proc/thread-self/children")?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?)
    };
    let before = child_ids()?;
    let mut executor = spawn_rank_group()?;
    let victim = child_ids()?
        .into_iter()
        .find(|pid| {
            if before.contains(pid) {
                return false;
            }
            let args = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            let args = args.split(|byte| *byte == 0).collect::<Vec<_>>();
            args.windows(2)
                .any(|pair| pair == [b"--rank".as_slice(), b"1".as_slice()])
        })
        .context("rank child was not found")?;
    drop(launch_guard);

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
    executor.submit_batch(
        lifecycle_batch(3, vec![text_admission(22, 1, 2)?], Vec::new()),
        &CallReaders::new(),
    )?;
    paused.terminate()?;
    let loss = executor
        .poll_batch(Duration::from_secs(30))
        .expect_err("rank loss must be reported");
    let loss = loss
        .downcast_ref::<WorkerFailure>()
        .context("expected scoped WorkerGroup loss")?;
    assert_eq!(loss.worker_id.0, initial_endpoint.worker_id);
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
    assert_eq!(
        executor.info().endpoint.worker_id,
        initial_endpoint.worker_id
    );
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
    let wrapper = directory.path().join("python");
    let python = serde_json::to_string(&worker_python())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nif sys.argv[sys.argv.index('--rank') + 1] == '1':\n    sys.exit(47)\nos.execv({python}, [{python}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.python = wrapper.clone();
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

#[test]
fn replacement_rejects_changed_numerical_policy_and_capacity() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // Killing rank zero relaunches the group. The wrapper merges
    // replacement.json into each relaunched rank's launch descriptor, so the
    // replacement describes a different numerical policy or capacity; the
    // head must refuse it and stop using the group.
    for change in [
        serde_json::json!({"model_dtype": "float32"}),
        serde_json::json!({"max_batch_tokens": 128}),
        serde_json::json!({"kv_token_capacity": 2048}),
    ] {
        let directory = tempfile::tempdir()?;
        let wrapper = directory.path().join("python");
        let python = serde_json::to_string(&worker_python())?;
        let state = serde_json::to_string(&directory.path())?;
        std::fs::write(
            &wrapper,
            format!(
                "#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\nstate = Path({state})\nrank = sys.argv[sys.argv.index('--rank') + 1]\n(state / (rank + '.pid')).write_text(str(os.getpid()))\nchange = state / 'replacement.json'\nif change.exists():\n    path = sys.argv[sys.argv.index('--launch-descriptor') + 1]\n    descriptor = json.loads(Path(path).read_text())\n    descriptor.update(json.loads(change.read_text()))\n    Path(path).write_text(json.dumps(descriptor))\nos.execv({python}, [{python}, *sys.argv[1:]])\n"
            ),
        )?;
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
        let mut args = rank_group_args(1 << 20, 8 << 20);
        args.python = wrapper;
        let mut worker = WorkerGroup::spawn(args)?;

        std::fs::write(
            directory.path().join("replacement.json"),
            serde_json::to_vec(&change)?,
        )?;
        let pid =
            std::fs::read_to_string(directory.path().join("0.pid"))?.parse::<libc::pid_t>()?;
        anyhow::ensure!(unsafe { libc::kill(pid, libc::SIGKILL) } == 0);

        let failure = worker
            .poll_batch(Duration::from_secs(60))
            .expect_err("rank loss must report replacement rejection");
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
fn closing_a_group_lets_every_rank_finish_a_collective_teardown() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // Each wrapper rank serves the stub worker as a child process (handing
    // it the inherited rendezvous socket) and, once
    // that worker has exited after Close, waits for every rank of its group
    // to have exited too before it records `<rank>.closed` and exits. Like
    // the release of the communicators a group's ranks share, that teardown
    // completes on any rank only once every rank has been asked to close.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    let python = serde_json::to_string(&worker_python())?;
    let records = serde_json::to_string(&directory.path())?;
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport json, subprocess, sys, time\nfrom pathlib import Path\nrecords = Path({records})\nrank = sys.argv[sys.argv.index('--rank') + 1]\npath = sys.argv[sys.argv.index('--launch-descriptor') + 1]\nfd = json.loads(Path(path).read_text())['rendezvous_listen_fd']\nstatus = subprocess.call([{python}, *sys.argv[1:]], pass_fds=() if fd is None else (fd,))\n(records / (rank + '.served')).write_text(str(status))\ndeadline = time.monotonic() + 60\nwhile len(list(records.glob('*.served'))) < {WORLD_SIZE}:\n    if time.monotonic() > deadline:\n        sys.exit(3)\n    time.sleep(0.05)\n(records / (rank + '.closed')).write_text('')\nsys.exit(status)\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.python = wrapper;
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

/// Reads one rank's registration report and returns the endpoint it names.
///
/// Accepts a single connection on the registration listener and parses the
/// first line it sends as a JSON report. The accept has no timeout; each read
/// on the accepted connection times out after 60 seconds. Fails on an I/O or
/// JSON error, or when the report has no string `endpoint` field.
fn accept_reported_endpoint(listener: &std::net::TcpListener) -> anyhow::Result<String> {
    use std::io::BufRead as _;

    // A worker that fails before registering never connects, so the wait
    // for its registration is bounded like the read of its report.
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    listener.set_nonblocking(true)?;
    let stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "no rank registered within 60 seconds"
                );
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(error.into()),
        }
    };
    listener.set_nonblocking(false)?;
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let mut line = String::new();
    std::io::BufReader::new(stream).read_line(&mut line)?;
    let report: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(report["endpoint"]
        .as_str()
        .context("rank registration named no endpoint")?
        .to_owned())
}

#[test]
fn launch_refuses_a_rank_that_offers_an_unsupported_channel() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // A rank names its own endpoint, so the mechanism it offers is data the
    // head has to check rather than a value it chose.
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    std::fs::write(
        &wrapper,
        r#"#!/usr/bin/env python3
import json, socket, sys

descriptor = json.load(open(sys.argv[sys.argv.index("--launch-descriptor") + 1]))
host, _, port = descriptor["registration_address"].rpartition(":")
report = {
    "worker_id": descriptor["worker_id"],
    "rank": descriptor["rank"],
    "transport": "socket",
    "endpoint": "unreachable",
}
with socket.create_connection((host, int(port)), timeout=60) as connection:
    connection.sendall(json.dumps(report).encode() + b"\n")
"#,
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;

    let mut args = rank_group_args(1 << 20, 8 << 20);
    args.python = wrapper;
    let reported = match WorkerGroup::spawn(args) {
        Ok(_) => anyhow::bail!("an unsupported channel transport was accepted"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        reported.contains("unsupported channel transport socket"),
        "launch failure did not name the offered transport: {reported}"
    );
    Ok(())
}

#[test]
fn losing_every_worker_stops_request_admission() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("python");
    let python = serde_json::to_string(&worker_python())?;
    let records = serde_json::to_string(directory.path())?;
    // Record process identities at the launch boundary; the numerical
    // worker and the engine both execute their normal public paths.
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport os, sys\nfrom pathlib import Path\nrank = sys.argv[sys.argv.index('--rank') + 1]\n(Path({records}) / (rank + '.pid')).write_text(str(os.getpid()))\nos.execv({python}, [{python}, *sys.argv[1:]])\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
    let mut config = EngineConfig::sim("stub");
    config.runtime_family = RuntimeFamily::Ar;
    config.generation_limits = uniserve_core::GenerationLimits {
        features: uniserve_core::GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.workers = vec![WorkerConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = rank_group_args(128 << 10, 128 << 10);
    config.worker_process.python = wrapper.clone();
    let engine = EngineCore::new(config)?;

    // Removing the launch executable makes recovery fail at the same OS
    // boundary as a missing deployment prerequisite, without changing IPC.
    remove_file(wrapper)?;
    let pid = std::fs::read_to_string(directory.path().join("0.pid"))?.parse::<libc::pid_t>()?;
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

/// Workers launched by the engine hold a request row for every running
/// request the engine is configured to keep resident, beyond the worker's
/// own default pool.
/// Launched ranks keep their host heap out of transparent huge pages: no
/// mapping of a rank process is advised for them (`MADV_HUGEPAGE`, VmFlags
/// `hg`), so khugepaged never collapses a serving rank's memory.
#[test]
fn launched_ranks_hold_no_huge_page_advised_memory() -> anyhow::Result<()> {
    let worker = spawn_rank_group()?;
    // The ranks are this thread's children, each launched with `--rank`.
    let ranks: Vec<String> = std::fs::read_to_string("/proc/thread-self/children")?
        .split_whitespace()
        .filter(|pid| {
            std::fs::read(format!("/proc/{pid}/cmdline"))
                .is_ok_and(|args| args.split(|byte| *byte == 0).any(|arg| arg == b"--rank"))
        })
        .map(str::to_owned)
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
    config.runtime_family = RuntimeFamily::Ar;
    config.generation_limits = uniserve_core::GenerationLimits {
        features: uniserve_core::GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.max_num_seqs = RUNNING;
    config.workers = vec![WorkerConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = rank_group_args(128 << 10, 128 << 10);
    let engine = {
        let _launch_guard = CHILD_LAUNCH_ENV_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("child-launch environment lock is poisoned"))?;
        EngineCore::new(config)?
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
    config.runtime_family = RuntimeFamily::Diffusion;
    config.generation_limits = uniserve_core::GenerationLimits {
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..Default::default()
    };
    config.max_batch = QUEUE_DEPTH * 8;
    config.workers = vec![WorkerConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerConfig::single_component("model", WORLD_SIZE),
    )];
    config.worker_process = rank_group_args(128 << 10, 128 << 10);
    let engine = {
        let _launch_guard = CHILD_LAUNCH_ENV_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("child-launch environment lock is poisoned"))?;
        EngineCore::new(config)?
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
            let events = engine.submit(Request::Diffusion(DiffusionRequest {
                request_id,
                task: uniserve_core::VideoTask::T2va,
                text_tags: vec![1; prompt_token_ids.len()],
                prompt_token_ids,
                conditions: Vec::new(),
                media: Vec::new(),
                priority: 0,
                sampling: DiffusionSamplingParams {
                    num_frames: 22,
                    video_units: 3,
                    num_inference_steps: 4,
                    seed: index as u64,
                    width: 1344,
                    height: 768,
                },
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
                matches!(event, EngineCoreOutput::Rejected { .. }),
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
    let named = WorkerConfig::placed(
        &["compute-0".to_owned()],
        "cpu",
        WORLD_SIZE,
        2,
        WorkerConfig::single_component("model", WORLD_SIZE),
    );
    assert!(
        named.ranks.iter().all(|rank| rank.node == "compute-0"),
        "the placement shorthand must place ranks on the host it is given"
    );

    // A rank placed on another host is started by that host's launcher, which
    // connects to the head and presents the host it owns. No launcher presents
    // here, so the launch fails naming the host whose ranks never started
    // rather than waiting for them or starting them in the wrong place.
    let spawned = WorkerGroup::spawn(WorkerProcessArgs {
        python: worker_python(),
        model: String::new(),
        host: "compute-0".into(),
        ranks: WorkerConfig::placed(
            &["compute-1".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            WorkerConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: named.components.clone(),
        stub: true,
        launcher_timeout: std::time::Duration::from_secs(2),
        ..WorkerProcessArgs::default()
    });
    let Err(error) = spawned else {
        panic!("a rank whose host never presented a launcher must not launch");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("compute-1"),
        "the refusal must name the host whose launcher never presented: {message}"
    );
}

/// A process stopped with `SIGSTOP` until `resume` continues it or `terminate`
/// kills it.
///
/// Dropping a still-paused value sends `SIGCONT`, so a failing test does not
/// leave a stopped rank behind.
struct PausedProcess(Option<i32>);

impl PausedProcess {
    fn new(pid: i32) -> anyhow::Result<Self> {
        anyhow::ensure!(pid > 0, "worker process id must be positive");
        anyhow::ensure!(
            unsafe { libc::kill(pid, libc::SIGSTOP) } == 0,
            "failed to pause the selected worker: {}",
            std::io::Error::last_os_error()
        );
        Ok(Self(Some(pid)))
    }

    fn resume(&mut self) -> anyhow::Result<()> {
        if let Some(pid) = self.0 {
            anyhow::ensure!(
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

/// Spawns the default two-rank stub group from `rank_group_args`.
fn spawn_rank_group() -> anyhow::Result<WorkerGroup> {
    spawn_rank_group_with_capacities(1 << 20, 8 << 20)
}

fn spawn_rank_group_with_capacities(
    request_slot_capacity: usize,
    response_slot_capacity: usize,
) -> anyhow::Result<WorkerGroup> {
    WorkerGroup::spawn(rank_group_args(
        request_slot_capacity,
        response_slot_capacity,
    ))
}

/// Launch arguments for a two-rank stub group on the local host's CPU.
///
/// Both ranks hold the single `model` component, with `QUEUE_DEPTH` batches
/// in flight and the given channel slot capacities in bytes. Tests adjust the
/// returned value (for example components, interpreter or transfer edges)
/// before spawning.
fn rank_group_args(
    request_slot_capacity: usize,
    response_slot_capacity: usize,
) -> WorkerProcessArgs {
    let worker = worker_python();
    let config = WorkerProcessArgs {
        stub: true,
        prefill_cuda_graph: false,
        ..WorkerProcessArgs::default()
    };
    WorkerProcessArgs {
        python: worker,
        model: String::new(),
        ranks: WorkerConfig::placed(
            &["localhost".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            WorkerConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: WorkerConfig::single_component("model", WORLD_SIZE),
        queue_depth: QUEUE_DEPTH,
        req_slot_cap: request_slot_capacity,
        resp_slot_cap: response_slot_capacity,
        kv_token_capacity: Some(4096),
        block_size: Some(16),
        max_batch_calls: 256,
        max_batch_tokens: 256,
        attention_backend: uniserve_ipc::AttentionBackend::TorchSdpa,
        transfer: Default::default(),
        ..config
    }
}

/// Builds the launch descriptor for a weightless CPU worker used by IPC tests.
///
/// Production launches receive the same typed descriptor from the engine;
/// these tests state one directly because they drive the worker process
/// without a group.
fn stub_launch_descriptor(registration: &str) -> LaunchDescriptor {
    LaunchDescriptor {
        worker_id: "worker".into(),
        registration_address: registration.to_owned(),
        channel_transport: uniserve_ipc::SHARED_STORAGE_CHANNEL.into(),
        acknowledgment_slot: 0,
        host_slots: vec![0],
        products_cross_hosts: false,
        queue_depth: 8,
        ipc_payload_cap: 1 << 20,
        model: String::new(),
        base_model: None,
        device: "cpu".into(),
        rank: 0,
        local_rank: 0,
        world_size: 1,
        components: WorkerConfig::single_component("model", 1),
        deployment_components: vec!["model".into()],
        supported_calls: None,
        transfer_backends: "local".into(),
        export_backends: "local".into(),
        rendezvous_address: None,
        rendezvous_listen_fd: None,
        expert_parallel: None,
        role: "model".into(),
        expert_microbatches: 1,
        distributed_backend: None,
        mesh: None,
        lane: Vec::new(),
        no_model: true,
        allow_stub: true,
        load_format: "auto".into(),
        download_dir: None,
        load_threads: None,
        checksum_manifest: None,
        model_dtype: "bfloat16".into(),
        quantization_config: serde_json::json!({}),
        kv_cache_dtype: None,
        kv_memory_fraction: 0.7,
        kv_token_capacity: Some(4096),
        attention_backend: "torch_sdpa".into(),
        block_size: Some(16),
        max_batch_calls: 8,
        max_batch_tokens: 256,
        max_request_pool_size: 129,
        max_model_len: 8192,
        max_video_seconds: 15.0,
        max_condition_rows: 0,
        ffmpeg: "ffmpeg".into(),
        min_video_seconds: None,
        graph_policy: "off".into(),
        decode_graph_batch_sizes: None,
        prefill_cuda_graph: false,
        prefill_outputs: true,
        prefill_graph_token_sizes: None,
        flow_cuda_graph: false,
        flow_graph_batch_sizes: None,
        flow_graph_shapes: None,
        video_text_capacities: None,
        canvas_sampling: None,
        flashinfer_workspace_size: 512 << 20,
        flashinfer_use_tensor_core: None,
        flashinfer_decode_backend: "fa2".into(),
        flashinfer_prefill_backend: "auto".into(),
        flashinfer_decode_split_tile_size: None,
        flashinfer_prefill_split_tile_size: None,
        flashinfer_disable_split_kv: false,
    }
}

fn worker_python() -> PathBuf {
    std::env::var_os("UNISERVE_WORKER_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(".venv/bin/python")
        })
}

/// Submits one batch and waits up to 30 seconds for its result.
fn execute(
    executor: &mut WorkerGroup,
    batch: Batch,
) -> anyhow::Result<uniserve_engine::WorkerResult> {
    executor.submit_batch(batch, &CallReaders::new())?;
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
        Some(ArRequestParams {
            sampling: SamplingParams {
                temperature: 0.0,
                ignore_eos: true,
                ..SamplingParams::default()
            },
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 0,
            canvas: None,
        }),
        None,
        0,
    )?)
}
