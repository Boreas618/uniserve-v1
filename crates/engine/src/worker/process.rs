//! One rank's process and channel, from spawn to shutdown.
//!
//! `PendingWorker::spawn` derives the rank's launch descriptor and has the
//! daemon of the rank's node start the rank process (`processes`). The rank
//! connects to the engine and registers (`registry::WorkerRegistry`);
//! `PendingWorker::adopt` answers on that connection with the descriptor and
//! yields a `WorkerProcess`, which `WorkerGroup` (in `group`) drives:
//! `finish_startup` validates the ranks' startup reports, `submit_batch` and
//! `poll_batch` exchange batches and results, and `close_workers` or
//! `terminate` retires the rank.
//!
//! A rank's connection is its liveness: the engine treats a rank whose
//! connection has closed as dead, and a rank exits when its connection to
//! the engine closes. A rank that has not registered yet is known lost by
//! its daemon's exit report. A rank whose builder thread stays inside one
//! job reports it unasked (`WorkerResponse::Stalled`), and the engine fails
//! that rank as it fails a dead one. The daemon that started a rank reaps it and removes
//! the CUDA driver memory a rank that did not end in order left behind.
//!
//! The engine exchanges batch descriptors and bounded result values while
//! tensors, cache pages, and latent storage remain worker-resident.
//!
//! The defaults of the launch settings `WorkerProcessArgs` carries and the
//! launch descriptor built from them are defined in `descriptor`.

mod descriptor;

use crate::executor::WorkerResult;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::BatchSubmitError;
use crate::executor::WorkerExecError;
use anyhow::{Context, bail};
use ipc::Channel;
use ipc::{Batch, LaunchDescriptor, WorkerInfo, WorkerRequest, WorkerResponse};

use crate::worker::WorkerProcessArgs;
use crate::worker::processes::Processes;

use descriptor::worker_environment;

/// Maximum time `request_close` spends draining in-flight responses from a
/// still-alive worker before it sends the shutdown request. A hung (alive but
/// unresponsive) worker must not be able to block shutdown forever.
const WORKER_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the ranks closed together by `close_workers` have, from the last
/// shutdown request, to acknowledge it, finish their teardown and exit. A
/// local process still running afterwards is killed, which abandons whatever
/// its teardown had not yet released.
const WORKER_EXIT_GRACE: Duration = Duration::from_secs(30);
/// Longest wait between checks for the exit of a rank `close_workers` is
/// closing. A daemon reports an exit within its own 100-millisecond poll.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Interval between progress logs while a rank prepares during startup.
const STARTUP_LOG_INTERVAL: Duration = Duration::from_secs(30);
/// Longest wait between liveness checks (`WorkerProcess::check_worker`) while
/// waiting on a rank, which bounds how late a local rank's exit status is
/// read once its connection has closed.
const WORKER_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// One adopted rank: its channel, the report of its process's exit, and the
/// requests outstanding on it.
///
/// Dropping it closes the rank alone through `close_workers`.
pub(super) struct WorkerProcess {
    client: Channel,
    /// The rank's validated startup report; the default value until
    /// `finish_startup` stores it.
    info: WorkerInfo,
    /// The startup cancellation flag shared by the ranks launched together,
    /// which the teardown of any of them sets. `WorkerGroup` detaches it
    /// with `set_startup_cancel(None)` once the group is ready.
    startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The rank's process, which its node's daemon supervises.
    process: ExitWatch,
    /// Most requests outstanding at once on the rank.
    depth: usize,
    rank: u32,
    /// Submitted batches awaiting a response, by message identity.
    pending: HashMap<u64, OutstandingKind>,
    /// Routed results and errors, in arrival order, that `poll_batch` has
    /// not returned yet.
    ready: VecDeque<anyhow::Result<WorkerResult>>,
    /// Next message identity to assign. Identities start at 1 and are never
    /// zero, which is the identity a channel gives a response that states
    /// none.
    next_message_id: u64,
    /// How far the rank's retirement has progressed.
    retirement: Retirement,
    /// Whether the startup report has been asked for and has not arrived.
    starting: bool,
}

/// How far a rank's retirement has progressed.
enum Retirement {
    /// Neither `request_close` nor `terminate` has run.
    Serving,
    /// `request_close` has asked the rank to shut down, and `await_close`
    /// has not yet waited for it. `acknowledgment` is the shutdown request's
    /// message identity while its answer is outstanding; it is `None` when
    /// the rank had already exited or the request could not be sent.
    Closing { acknowledgment: Option<u64> },
    /// The rank was closed or terminated; nothing remains to do.
    Retired,
}

/// One launched rank between its start and its registration.
///
/// The rank's channel exists once the rank has connected and registered, so
/// everything the rank is answered with is retained here meanwhile. A launch
/// that fails stops its group's ranks through their daemons.
pub(super) struct PendingWorker {
    process: ExitWatch,
    rank: u32,
    /// Most requests outstanding at once on the rank.
    depth: usize,
    /// Moves by adoption alongside the process it cancels.
    startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The launch descriptor `adopt` answers the rank's registration with.
    descriptor: LaunchDescriptor,
}

/// A rank's process as its daemon reports it: whether it exited.
#[derive(Clone)]
pub(super) struct ExitWatch {
    processes: Processes,
    group: String,
    generation: u64,
    rank: u32,
}

impl ExitWatch {
    /// The rank's exit status, once its daemon reported it.
    fn exited(&self) -> Option<String> {
        crate::worker::processes::lock(&self.processes)
            .ok()?
            .exited(&self.group, self.generation, self.rank)
    }
}

/// What an outstanding request was sent for; only batch submissions are
/// tracked in `WorkerProcess::pending`.
enum OutstandingKind {
    Batch { batch_id: u64 },
}

/// Sockets a rank inherits from its daemon and serves collective stores on,
/// as reservations of its node.
#[derive(Default)]
pub(crate) struct WorkerSockets {
    /// The group's rendezvous store, served by the group's first rank.
    pub(crate) group: Option<(u64, i32)>,
    /// The expert-parallel world's store, served by that world's rank 0.
    pub(crate) experts: Option<(u64, i32)>,
}

impl PendingWorker {
    /// Has the daemon of the rank's node start one rank, which connects to
    /// the engine at `engine_address` and registers as a member of its group's
    /// launch `generation`.
    ///
    /// Nothing here waits for the rank. `rendezvous` is the group's
    /// collective store address, and `sockets` name the reservations of the
    /// rank's node whose sockets it inherits and serves stores on: the
    /// group's store when it is the group's first rank, its expert-parallel
    /// world's store when it is that world's rank 0. `startup_abort` is the
    /// startup cancellation flag shared by the ranks launched together.
    ///
    /// Fails when the rank's host-relative values or its descriptor cannot
    /// be built, or its daemon cannot start it.
    #[expect(
        clippy::too_many_arguments,
        reason = "one rank's launch: its arguments, sockets, daemons and generation"
    )]
    pub(crate) fn spawn(
        args: &WorkerProcessArgs,
        rank: u32,
        storage: ipc::StorageLaunch,
        rendezvous: Option<&str>,
        sockets: WorkerSockets,
        processes: &Processes,
        startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
        engine_address: &str,
        generation: u64,
    ) -> anyhow::Result<Self> {
        let depth = args.queue_depth.max(1);
        let world_size = args.world_size();

        // argv carries the address of the engine the worker connects to and
        // what its registration states: its worker group and its rank, and
        // the generation of the group's launch it belongs to. They are the
        // worker's only statement of its identity; the launch descriptor
        // repeats none of them.
        let arguments = vec![
            "worker".to_owned(),
            "--engine".to_owned(),
            engine_address.to_owned(),
            "--worker-id".to_owned(),
            args.worker_id.clone(),
            "--rank".to_owned(),
            rank.to_string(),
            "--generation".to_owned(),
            generation.to_string(),
        ];

        // Torch and the numerical libraries read the host-relative pair; the
        // global pair names the rank's place in the whole process world. The
        // daemon applies the variables over its own environment.
        let (local_rank, local_world_size) = args.host_slot(rank)?;
        let mut environment = std::collections::HashMap::from([
            ("RANK".to_owned(), rank.to_string()),
            ("WORLD_SIZE".to_owned(), world_size.to_string()),
            ("LOCAL_RANK".to_owned(), local_rank.to_string()),
            ("LOCAL_WORLD_SIZE".to_owned(), local_world_size.to_string()),
        ]);
        for (name, value) in worker_environment(|name| std::env::var_os(name)) {
            environment.insert(name.to_owned(), value.to_string_lossy().into_owned());
        }
        // A standby that may suspend its NCCL communicators while it parks
        // captures their collectives without registering graph buffers:
        // the registrations would not survive the suspension.
        if args.standby && args.standby_memory != proto::request::StandbyMemory::Resident {
            environment.insert("NCCL_GRAPH_REGISTER".to_owned(), "0".to_owned());
        }

        // One typed descriptor carries every launch value. The engine keeps
        // it and answers the rank's registration with it.
        let descriptor = args.launch_descriptor(
            rank,
            storage,
            rendezvous,
            sockets.group.map(|(_, descriptor)| descriptor),
            sockets.experts.map(|(_, descriptor)| descriptor),
        )?;

        // The worker runs the engine's build: the executable imports the
        // `uniserve` package of the Python environment it is installed in,
        // which holds the worker library of that build.
        let executable = args
            .executable
            .to_str()
            .context("the worker executable's path is not valid text")?
            .to_owned();
        crate::worker::processes::lock(processes)?.spawn(proto::process::Spawn {
            host: args.ranks[rank as usize].host.clone(),
            group: args.worker_id.clone(),
            rank,
            generation,
            executable,
            arguments,
            environment,
            inherit: sockets
                .group
                .into_iter()
                .chain(sockets.experts)
                .map(|(reservation, _)| reservation)
                .collect(),
        })?;

        Ok(Self {
            process: ExitWatch {
                processes: std::sync::Arc::clone(processes),
                group: args.worker_id.clone(),
                generation,
                rank,
            },
            rank,
            depth,
            startup_abort,
            descriptor,
        })
    }

    /// Fails by name when the rank exited before registering, and when its
    /// launch's startup was cancelled.
    ///
    /// `registry::collect` calls this, through the `alive` callback `adopt_workers`
    /// passes it, while the launch is incomplete.
    pub(crate) fn check_alive(&mut self) -> anyhow::Result<()> {
        if self
            .startup_abort
            .load(std::sync::atomic::Ordering::Acquire)
        {
            bail!("rank {} was cancelled before registering", self.rank);
        }
        match self.process.exited() {
            None => Ok(()),
            Some(status) => Err(anyhow::anyhow!(
                "rank {} exited with {status} before registering",
                self.rank
            )),
        }
    }

    /// Answers the rank's registration on `channel` with its launch
    /// descriptor and takes the connection as the rank's channel.
    ///
    /// The returned `WorkerProcess`'s `info` stays the default until
    /// `finish_startup` runs.
    pub(crate) fn adopt(self, mut channel: Channel) -> anyhow::Result<WorkerProcess> {
        let PendingWorker {
            process,
            rank,
            depth,
            startup_abort,
            descriptor,
        } = self;

        // A failed send closes the connection, which ends the rank.
        channel
            .send(&descriptor)
            .with_context(|| format!("sending rank {rank} its launch descriptor"))?;
        tracing::info!(rank, "a rank registered and received its launch descriptor");

        Ok(WorkerProcess {
            client: channel,
            info: WorkerInfo::default(),
            startup_cancel: Some(startup_abort),
            process,
            depth,
            rank,
            pending: HashMap::new(),
            ready: VecDeque::new(),
            next_message_id: 1,
            retirement: Retirement::Serving,
            starting: false,
        })
    }
}

impl WorkerProcess {
    /// Asks the rank for its startup report.
    ///
    /// The rank answers only once its model is built and warmed up;
    /// `poll_startup` takes the answer.
    pub(super) fn request_startup(&mut self) -> anyhow::Result<()> {
        let message_id = self.alloc_call_id();
        let mut request = WorkerRequest::info();
        request.set_message_id(Some(message_id));
        self.client.send(&request)?;
        self.starting = true;
        Ok(())
    }

    /// Takes the rank's startup report if it has arrived, as `info`. Returns
    /// whether the rank's startup is complete.
    ///
    /// Fails when the rank reports that its startup failed, and when its
    /// connection has closed, which is how a rank that died while starting
    /// is noticed.
    pub(super) fn poll_startup(&mut self) -> anyhow::Result<bool> {
        let response = self
            .client
            .try_recv::<WorkerResponse>()
            .with_context(|| format!("rank {} was lost during startup", self.rank))?;
        // A rank that is ready stays under watch until its peers are: the
        // receive above is what notices its connection closing.
        if !self.starting {
            return Ok(true);
        }
        let Some(response) = response else {
            return Ok(false);
        };
        let info = match response {
            WorkerResponse::Info { info, .. } => info,
            WorkerResponse::Error { error, .. } => {
                tracing::error!(
                    rank = self.rank,
                    "rank failed during startup: {}",
                    error.message
                );
                bail!(
                    "rank {} failed during startup: {}",
                    self.rank,
                    error.message
                )
            }
            WorkerResponse::Result { result, .. } => {
                // Receiving claims any media the result published, so the
                // shared-memory objects are released rather than left named.
                drop(WorkerResult::receive(result));
                bail!("unexpected startup result response");
            }
            other => bail!("unexpected startup response: {:?}", other.kind()),
        };

        self.check_worker("Worker startup")?;
        self.info = info;
        self.starting = false;
        tracing::info!(?self.info, "worker ready");
        Ok(true)
    }

    /// Asks a parked standby rank to promote (`WorkerRequest::Promote`): it
    /// takes write access to the exclusive storages its predecessor held,
    /// which the Storage Server shoots down if it still holds them, restores
    /// what it suspended while it parked, and serves. Returns the request's
    /// message, whose answer [`Self::await_promotion`] waits for. The ranks
    /// of one world restore their communication collectively, so every
    /// rank of it is asked before any answer is awaited ([`promote_all`]).
    pub(super) fn request_promotion(&mut self) -> anyhow::Result<u64> {
        let message_id = self.alloc_call_id();
        let mut request = WorkerRequest::promote();
        request.set_message_id(Some(message_id));
        self.client.send(&request)?;
        Ok(message_id)
    }

    /// Waits at most `timeout` for the rank's answer to the promotion
    /// `message_id`.
    ///
    /// Fails when the rank refuses, dies, or does not answer in time.
    pub(super) fn await_promotion(
        &mut self,
        message_id: u64,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            let response = self
                .client
                .try_recv::<WorkerResponse>()
                .with_context(|| format!("standby rank {} was lost", self.rank))?;
            match response {
                Some(WorkerResponse::Ok {
                    message_id: Some(id),
                }) if id == message_id => {
                    tracing::info!(rank = self.rank, "the standby rank was promoted");
                    return Ok(());
                }
                Some(WorkerResponse::Error { error, .. }) => {
                    bail!(
                        "standby rank {} was not promoted: {}",
                        self.rank,
                        error.message
                    )
                }
                Some(other) => bail!("unexpected promotion response: {:?}", other.kind()),
                None => {}
            }
            self.check_worker("standby promotion")?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "standby rank {} was not promoted within {timeout:?}",
                self.rank
            );
            let _ = self.client.wait(remaining.min(WORKER_CHECK_INTERVAL));
        }
    }

    /// Asks the rank to run the worker hook `name` with `payload`
    /// (`WorkerRequest::Hook`). Returns the request's message, whose answer
    /// [`Self::await_hook`] waits for. The engine asks only while the rank
    /// runs no batch, so no other answer is outstanding.
    pub(super) fn request_hook(&mut self, name: &str, payload: &str) -> anyhow::Result<u64> {
        let message_id = self.alloc_call_id();
        let mut request = WorkerRequest::hook(name, payload);
        request.set_message_id(Some(message_id));
        self.client.send(&request)?;
        Ok(message_id)
    }

    /// Waits at most `timeout` for the rank's answer to the hook
    /// `message_id`: the hook's JSON result.
    ///
    /// Fails when the hook fails, the rank dies, or it does not answer in
    /// time.
    pub(super) fn await_hook(
        &mut self,
        message_id: u64,
        timeout: Duration,
    ) -> anyhow::Result<String> {
        let deadline = Instant::now() + timeout;
        loop {
            let response = self
                .client
                .try_recv::<WorkerResponse>()
                .with_context(|| format!("rank {} was lost", self.rank))?;
            match response {
                Some(WorkerResponse::Hooked {
                    message_id: Some(id),
                    result,
                }) if id == message_id => return Ok(result),
                Some(WorkerResponse::Error { error, .. }) => {
                    bail!("rank {}: {}", self.rank, error.message)
                }
                Some(WorkerResponse::Stalled { stall }) => return Err(self.stalled(&stall)),
                Some(other) => bail!("unexpected hook response: {:?}", other.kind()),
                None => {}
            }
            self.check_worker("a worker hook")?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "rank {} did not run the hook within {timeout:?}",
                self.rank
            );
            let _ = self.client.wait(remaining.min(WORKER_CHECK_INTERVAL));
        }
    }

    /// The generation of the group's launch the rank belongs to.
    pub(super) fn generation(&self) -> u64 {
        self.process.generation
    }

    /// Allocates the next worker message identifier.
    fn alloc_call_id(&mut self) -> u64 {
        let id = self.next_message_id;
        self.next_message_id += 1;
        id
    }

    /// Replaces the startup cancellation flag `check_worker` consults.
    ///
    /// `WorkerGroup` passes `None` once the group is ready, so a later exit
    /// no longer reads as a cancelled startup on the surviving ranks.
    pub(crate) fn set_startup_cancel(
        &mut self,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) {
        self.startup_cancel = cancel;
    }

    /// Aborts peers when an unfinished deployment is being dismantled.
    /// Ready ranks have detached the flag, so ordinary shutdown is unaffected.
    pub(crate) fn cancel_startup(&self) {
        if let Some(cancel) = &self.startup_cancel {
            cancel.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    /// Terminates a failed or cancelled rank.
    ///
    /// Sends no shutdown request and discards outstanding and routed results.
    /// The rank's channel closes, which ends the rank; `WorkerGroup` also
    /// stops the group's processes through their daemons
    /// (`ProcessClient::stop`), which kill a rank that did not end.
    pub(crate) fn terminate(&mut self) {
        self.retirement = Retirement::Retired;
        self.client.shutdown();
        self.pending.clear();
        self.ready.clear();
    }

    /// Checks the rank's liveness and cancellation of an unfinished startup.
    ///
    /// Fails when the startup cancellation flag is set, or when the rank's
    /// daemon reported its exit.
    pub(super) fn check_worker(&mut self, context: &str) -> anyhow::Result<()> {
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            bail!("worker startup cancelled during {context}");
        }
        if let Some(status) = self.process.exited() {
            bail!("worker process exited during {context}: {status}");
        }
        Ok(())
    }

    /// Drains ready responses and routes them into `ready`, and returns how
    /// many were routed.
    ///
    /// A transport, execution, or routing error, or a stall report, stops
    /// the scan and is returned; responses routed before it stay in
    /// `ready`. The channel reports its close, which is the rank's death,
    /// as an error only after every response that arrived whole has been
    /// routed.
    fn drain_ready(&mut self) -> anyhow::Result<usize> {
        let mut drained = 0usize;
        while let Some(response) = self.client.try_recv::<WorkerResponse>()? {
            // A stall report answers no request: it fails the rank.
            if let WorkerResponse::Stalled { stall } = response {
                return Err(self.stalled(&stall));
            }
            // A response answers the request whose identity it states.
            let message_id = response
                .message_id()
                .context("worker response states no message identity")?;
            let kind = self.pending.remove(&message_id).with_context(|| {
                format!("worker response {message_id} answers no outstanding request")
            })?;
            self.route(kind, response)?;
            drained += 1;
        }
        Ok(drained)
    }

    /// Routes one batch response and appends its result to `ready`.
    ///
    /// The result is received, which claims any media it published
    /// (`WorkerResult::receive`). Returns an error and appends nothing when
    /// the worker answered with an
    /// execution error (as a `WorkerExecError`) or an unexpected response
    /// kind, a product locator names an endpoint other than this rank's
    /// reported one, or the result's batch identity differs from the
    /// submitted batch.
    fn route(&mut self, kind: OutstandingKind, response: WorkerResponse) -> anyhow::Result<()> {
        let OutstandingKind::Batch { batch_id } = kind;
        let report = match response {
            WorkerResponse::Result { result, .. } => Ok(WorkerResult::receive(result)),
            WorkerResponse::Error { error, .. } => Err(WorkerExecError {
                batch_id: Some(batch_id),
                fatal: error.fatal,
                code: error.code,
                message: error.message,
                phase: error.phase,
                route: error.route,
                calls: error.calls,
            }
            .into()),
            other => Err(anyhow::anyhow!(
                "unexpected execute response kind: {:?}",
                other.kind()
            )),
        };
        let report = report?;

        // A product must name this rank's reported endpoint, which identifies
        // the incarnation whose startup report `finish_startup` accepted.
        for product in &report.products {
            anyhow::ensure!(
                product
                    .value
                    .locators()
                    .all(|locator| locator.source == self.info.endpoint),
                "worker published a product from an unbound rank incarnation"
            );
        }
        anyhow::ensure!(
            report.batch_id == batch_id,
            "worker result batch id mismatch: expected {batch_id}, got {}",
            report.batch_id
        );
        self.ready.push_back(Ok(report));
        Ok(())
    }

    /// Logs a job the rank reports stalled and returns the failure of the
    /// rank, which names the forward the job belongs to.
    ///
    /// The rank's builder thread stays inside one job, so the rank cannot
    /// complete the batches behind it: the group fails it as it fails a
    /// lost rank.
    fn stalled(&self, stall: &ipc::Stall) -> anyhow::Error {
        let calls = stall
            .calls
            .iter()
            .map(|call| format!("{}/{:?}", call.request_key.request_id.0, call.call_id))
            .collect::<Vec<_>>();
        tracing::error!(
            rank = self.rank,
            batch_id = ?stall.batch_id,
            route = ?stall.route,
            method = ?stall.method,
            calls = ?calls,
            running_ms = stall.running_ms,
            "a rank reports its builder thread stalled inside one job"
        );

        let forward = match stall.batch_id {
            Some(batch_id) => format!(
                "a job of batch {batch_id} ({}, method {}, calls [{}])",
                stall.route.as_deref().unwrap_or("no calls"),
                stall.method.as_deref().unwrap_or("none"),
                calls.join(", "),
            ),
            None => format!(
                "a job of no batch (method {})",
                stall.method.as_deref().unwrap_or("none")
            ),
        };
        anyhow::anyhow!(
            "the builder thread stalled: it has performed {forward} for {:.1} s without completing it",
            Duration::from_millis(stall.running_ms).as_secs_f64()
        )
    }

    /// Returns the descriptors that signal this rank's progress: its
    /// connection, whose close is also the rank's death.
    pub(crate) fn progress_fds(&self) -> Vec<libc::pollfd> {
        self.client.poll_fds().to_vec()
    }
}

impl WorkerProcess {
    /// Returns metadata for the physical worker.
    pub(super) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Submits one physical batch and records its outstanding message identity.
    ///
    /// Hands the batch back as `WouldBlock` when `depth` requests are already
    /// outstanding; the batch is not recorded as pending then.
    /// Fails when draining earlier responses or the send fails.
    pub(super) fn submit_batch(&mut self, batch: Batch) -> Result<(), BatchSubmitError> {
        // Answered requests free their slots before the depth check.
        self.drain_ready().map_err(BatchSubmitError::Failed)?;
        if self.pending.len() >= self.depth {
            return Err(BatchSubmitError::WouldBlock(Box::new(batch)));
        }
        let batch_id = batch.batch_id;
        let message_id = self.alloc_call_id();
        let mut req = WorkerRequest::submit(batch);
        req.set_message_id(Some(message_id));
        self.client
            .send(&req)
            .map_err(|error| BatchSubmitError::Failed(error.into()))?;
        self.pending
            .insert(message_id, OutstandingKind::Batch { batch_id });
        Ok(())
    }

    /// Returns completed responses before failing outstanding work on disconnect.
    ///
    /// Returns the next entry of `ready`, waiting up to `timeout`, and
    /// `Ok(None)` when the timeout passes with nothing ready. An entry is a
    /// result or the error that draining or routing a response produced.
    /// Results already received are returned before a rank exit, startup
    /// cancellation, the rank's connection closing, or a failed wait fails
    /// the call.
    pub(super) fn poll_batch(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(result) = self.ready.pop_front() {
                return result.map(Some);
            }
            if let Err(error) = self.drain_ready() {
                // Complete replies remain ahead of a trailing transport or
                // decoding error in the same drain.
                self.ready.push_back(Err(error));
                continue;
            }
            if !self.ready.is_empty() {
                continue;
            }
            self.check_worker("executor poll")?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            self.client.wait(remaining.min(WORKER_CHECK_INTERVAL))?;
        }
    }

    /// Drains outstanding calls and asks the rank to shut down.
    ///
    /// A startup that was cancelled goes straight to `terminate`. Otherwise
    /// only a first call acts, and none after `terminate`. A live rank gets up
    /// to `WORKER_DRAIN_TIMEOUT` to answer outstanding batches and is then
    /// sent a shutdown request, without waiting for its answer; `await_close`
    /// completes the retirement. A failure to reach the rank is absorbed, and
    /// `await_close` then bounds the rank by its deadline alone.
    fn request_close(&mut self) {
        // A rank of this launch exited before the group became ready, so this
        // rank is terminated without draining or a shutdown request.
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            self.terminate();
            return;
        }
        if !matches!(self.retirement, Retirement::Serving) {
            return;
        }

        let mut acknowledgment = None;
        let exited = self.process.exited().is_some();
        if !exited {
            // Bound response draining so shutdown can advance to graceful termination
            // and, if necessary, forced process cleanup.
            let drain_deadline = Instant::now() + WORKER_DRAIN_TIMEOUT;
            while !self.pending.is_empty() {
                if Instant::now() >= drain_deadline {
                    tracing::warn!(
                        pending = self.pending.len(),
                        "worker did not drain in-flight responses before shutdown deadline"
                    );
                    break;
                }
                match self.drain_ready() {
                    Ok(0) => {
                        if self.check_worker("shutdown drain").is_err() {
                            break;
                        }
                        let remaining = drain_deadline.saturating_duration_since(Instant::now());
                        let _ = self.client.wait(remaining.min(WORKER_CHECK_INTERVAL));
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }

            // A rank still running after the drain is asked to shut down.
            if self.process.exited().is_none() {
                let message_id = self.alloc_call_id();
                let mut req = WorkerRequest::close();
                req.set_message_id(Some(message_id));
                acknowledgment = self.client.send(&req).ok().map(|()| message_id);
            }
        }
        self.retirement = Retirement::Closing { acknowledgment };
    }

    /// Waits until `deadline` for a rank `request_close` asked to shut down
    /// to exit, which its daemon reports.
    ///
    /// The shutdown request stays outstanding until the rank answers it, so
    /// the answer, which the rank sends before its teardown begins, always
    /// has a receiver. The rank is retired once its daemon reports its exit.
    /// Returns whether it was; the caller stops the group's processes still
    /// running at the deadline. Acts only on a rank that is closing.
    fn await_close(&mut self, deadline: Instant) -> bool {
        let Retirement::Closing { mut acknowledgment } =
            std::mem::replace(&mut self.retirement, Retirement::Retired)
        else {
            return true;
        };

        loop {
            // A failed receive means the rank's end of the channel is gone,
            // which answers the request as finally as a response does. A
            // response to anything else is a result nobody waits for any more.
            while let Some(message_id) = acknowledgment {
                match self.client.try_recv::<WorkerResponse>() {
                    Ok(Some(response)) if response.message_id() != Some(message_id) => {}
                    Ok(None) => break,
                    Ok(Some(_)) | Err(_) => acknowledgment = None,
                }
            }
            if self.process.exited().is_some() {
                return true;
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                tracing::warn!(
                    rank = self.rank,
                    "worker did not exit before the shutdown deadline; stopping it"
                );
                return false;
            }
            // A response ends the wait; a process exit does not, so the wait
            // is also bounded by the exit poll interval. A closed channel
            // fails the wait at once and still gets the interval, so the
            // loop never spins.
            let interval = remaining.min(EXIT_POLL_INTERVAL);
            if self.client.wait(interval).is_err() {
                std::thread::sleep(interval);
            }
        }
    }
}

/// Waits until every rank in `ranks` has reported that it is ready, and
/// checks each report against its launch.
///
/// A rank answers only once its model is built and warmed up, and ranks that
/// initialize together wait for one another, so the ranks are asked first
/// and then awaited together: the wait has no deadline, and ends by any
/// rank's failure. A rank that dies while starting closes its connection,
/// which fails the startup of all of them by that rank's name instead of
/// leaving its peers waiting for it.
pub(super) fn finish_startup(ranks: &mut [&mut WorkerProcess]) -> anyhow::Result<()> {
    for rank in ranks.iter_mut() {
        rank.request_startup()?;
    }

    let started = Instant::now();
    let mut last_log = started;
    let mut last_worker_check = started;
    loop {
        // Every rank is polled and parked on, including those already
        // ready, so that the close of any connection ends the wait.
        let mut ready = true;
        let mut descriptors = Vec::new();
        for rank in ranks.iter_mut() {
            ready &= rank.poll_startup()?;
            descriptors.extend(rank.progress_fds());
        }
        if ready {
            return Ok(());
        }

        if last_worker_check.elapsed() >= WORKER_CHECK_INTERVAL {
            for rank in ranks.iter_mut() {
                if let Err(lost) = rank.check_worker("Worker startup") {
                    // A rank that reported its failure before it exited is
                    // named by its own error, which its connection may still
                    // hold.
                    rank.poll_startup()?;
                    return Err(lost);
                }
            }
            last_worker_check = Instant::now();
        }
        if last_log.elapsed() >= STARTUP_LOG_INTERVAL {
            tracing::info!(
                elapsed_secs = started.elapsed().as_secs(),
                "workers still busy during startup"
            );
            last_log = Instant::now();
        }
        let until_check = WORKER_CHECK_INTERVAL.saturating_sub(last_worker_check.elapsed());
        let until_log = STARTUP_LOG_INTERVAL.saturating_sub(last_log.elapsed());
        crate::worker::park_descriptors(&descriptors, until_check.min(until_log))?;
    }
}

/// Retires the ranks of one worker group together, absorbing every failure.
///
/// A rank's orderly teardown retires the communicators and process groups it
/// shares with the group's other ranks (`Worker.close` on the Python side),
/// and that retirement completes on any rank only once every rank has
/// reached it. Every rank is therefore asked to shut down, after draining its
/// outstanding calls, before any is waited for, and all of them then share
/// one `WORKER_EXIT_GRACE` deadline. A rank still running then is stopped
/// with its group through its daemon (`stop`), which abandons whatever its
/// teardown had not yet released. Asking one rank at a time would leave the
/// first one waiting on peers that have not been asked, until it is killed
/// with its communicators unreleased.
pub(super) fn close_workers(ranks: &mut [WorkerProcess], stop: impl FnOnce()) {
    for rank in ranks.iter_mut() {
        rank.request_close();
    }
    let deadline = Instant::now() + WORKER_EXIT_GRACE;
    let mut exited = true;
    for rank in ranks.iter_mut() {
        exited &= rank.await_close(deadline);
    }
    if !exited {
        stop();
    }
}

impl Drop for WorkerProcess {
    /// Closes the rank as `close_workers` would close it alone; a rank still
    /// running at the deadline ends with its channel's close.
    fn drop(&mut self) {
        self.cancel_startup();
        close_workers(std::slice::from_mut(self), || {});
    }
}

/// Promotes the parked standby ranks `workers`: asks every rank, then waits
/// at most `timeout` for each answer, so that ranks restoring their world's
/// communication together are all promoting at once.
///
/// # Errors
///
/// Fails at the first rank that refuses, dies, or does not answer in time.
/// Runs the worker hook `name` on every rank in `workers`, asking each
/// before awaiting any (a hook may communicate between the ranks), and
/// returns each rank's result in order.
pub(in crate::worker) fn hook_all(
    workers: &mut [WorkerProcess],
    name: &str,
    payload: &str,
    timeout: Duration,
) -> anyhow::Result<Vec<String>> {
    let requests = workers
        .iter_mut()
        .map(|worker| worker.request_hook(name, payload))
        .collect::<anyhow::Result<Vec<_>>>()?;
    workers
        .iter_mut()
        .zip(requests)
        .map(|(worker, message_id)| worker.await_hook(message_id, timeout))
        .collect()
}

pub(in crate::worker) fn promote_all(
    workers: &mut [WorkerProcess],
    timeout: Duration,
) -> anyhow::Result<()> {
    let requests = workers
        .iter_mut()
        .map(WorkerProcess::request_promotion)
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (worker, message_id) in workers.iter_mut().zip(requests) {
        worker.await_promotion(message_id, timeout)?;
    }
    Ok(())
}
