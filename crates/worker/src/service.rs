//! Rank-channel admission, result delivery, stall reports, and orderly
//! worker shutdown.
//!
//! The control thread serves the channel here and never waits for a lane.
//! Each wait for a request or a completion ends, at the latest, when the
//! job in flight longest reaches the rank's stall bound: a lane that has
//! stayed inside one job that long, such as a driver call that does not
//! return, is reported to the engine ([`WorkerResponse::Stalled`]) with the
//! batch the job belongs to, and the channel goes on being served.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::{Channel, IpcError, RequestKind, WorkerInfo, WorkerRequest, WorkerResponse};

use crate::error::WorkerError;
use crate::executor::backend::WorkerBatch;
use crate::memory::retained::Retained;
use crate::{Batch, Error, Executor, Submission};

const EXPERT_POLL: Duration = Duration::from_micros(200);
// Keep a rank with open requests available for its engine's next batch before
// joining an empty expert step. The engine needs time to consume the result
// and schedule its next batch even when it still has admitted requests.
const OWN_STEP_WAIT: Duration = Duration::from_millis(8);

/// How a serving run ended other than by a failure of the rank's own.
pub(crate) enum Ending {
    /// The engine closed the rank, or a fatal response ended admission.
    Closed,
    /// The engine's channel failed (the error says how) while nothing was in
    /// flight: no admitted batch was unfinished and no job given to a lane
    /// or the builder thread was without its completion, so nothing the rank
    /// issued can still run on a device.
    Orphaned(WorkerError),
    /// The engine's channel failed (the error says how) while work was in
    /// flight. Nothing can deliver its results, and it may never complete:
    /// it may wait on a peer that stopped or abandoned its own work, and
    /// the builder thread may wait in it for good.
    Abandoned(WorkerError),
}

enum PendingResponse {
    Ready(Box<WorkerResponse>),
    Batch {
        message_id: Option<u64>,
        submission: Arc<Submission>,
    },
}

/// Borrows a worker's executor and channel for one synchronous serving run.
/// All channel waits run on the caller's thread; completion wakes interrupt
/// them without advancing execution on a callback thread.
/// What a parked standby worker released, which its promotion restores;
/// nothing for a standby that keeps its memory resident.
#[derive(Default)]
pub(crate) struct Suspension {
    /// The warmup work of the runtimes that run expert steps, when the
    /// expert transports freed workspaces that cannot keep their
    /// addresses: the promotion rebuilds them and captures those runtimes'
    /// graphs again.
    pub(crate) recapture: Vec<Box<dyn crate::warmup::WarmupHooks>>,
    /// The graph pools and scratch whose contents it moved to host memory.
    pub(crate) pools: Vec<Arc<crate::cuda_graph::GraphPool>>,
    /// The expert union whose transport workspaces it released.
    pub(crate) experts: Option<Arc<crate::parallel::ExpertUnion>>,
    /// The torch world whose NCCL communicators it suspended, with those of
    /// the runners' collectives.
    pub(crate) world: Option<crate::ffi::Handle>,
}

/// A standby worker parked until its promotion: its retained storages,
/// read-only, and what it released while parked.
pub(crate) struct Parked {
    pub(crate) retained: Arc<Retained>,
    pub(crate) suspension: Suspension,
}

impl Parked {
    /// Restore what the standby released, at its addresses, then take write
    /// access to the retained storages: the captured graphs run again.
    /// Restoring the expert transports is collective over the union: the
    /// engine asks every rank of the standby world to promote before it
    /// awaits any. Transports that freed their workspaces are rebuilt, and
    /// the graphs that run expert steps captured again, as warmup captured
    /// them: collectively, on the scratch units, with write access.
    fn promote(self, executor: &mut Executor) -> std::result::Result<(), WorkerError> {
        let Parked {
            retained,
            suspension:
                Suspension {
                    recapture,
                    pools,
                    experts,
                    world,
                },
        } = self;
        let backend = executor.backend();
        for pool in &pools {
            pool.restore()?;
        }
        if !pools.is_empty() {
            tracing::info!("the promoted standby worker restored its graph memory");
        }
        if let Some(world) = &world {
            for collectives in backend.collectives() {
                collectives.resume()?;
            }
            crate::ffi::Interpreter::import("uniserve.distributed.world.resume")?
                .invoke(&[world.view()])?;
            tracing::info!("the promoted standby worker resumed its communicators");
        }
        if let Some(experts) = &experts {
            experts.restore()?;
            tracing::info!("the promoted standby worker restored its expert transports");
        }
        retained.promote()?;
        if recapture.is_empty() {
            return Ok(());
        }
        // The union's only holder now is the backend.
        drop(experts);
        let backend = executor.backend_mut();
        if let Some(union) = backend.experts_mut()? {
            union.rebuild()?;
        }
        backend.reopen_expert_graphs()?;
        for mut hooks in recapture {
            hooks.capture(backend)?;
        }
        backend.seal_expert_graphs();
        tracing::info!("the promoted standby worker captured its expert graphs again");
        Ok(())
    }
}

pub(crate) struct Service<'a> {
    executor: &'a mut Executor,
    channel: &'a mut Channel,
    info: WorkerInfo,
    experts: bool,
    pending: VecDeque<PendingResponse>,
    closing: bool,
    shutdown: Option<WorkerResponse>,
    last_result: Instant,
    /// How long one job may run on a lane before the lane is reported.
    stall_bound: Duration,
    /// A stalled lane was reported. The engine fails the rank on the report,
    /// so it is made once.
    stalled: bool,
    /// The engine's channel failed: it closed, or a send or receive on it
    /// did.
    channel_failed: Cell<bool>,
    /// The Storage Server connection of a parked standby worker, which
    /// promotion takes write access through; `None` once the rank serves.
    standby: Option<Parked>,
}

impl<'a> Service<'a> {
    pub(crate) fn new(
        executor: &'a mut Executor,
        channel: &'a mut Channel,
        info: WorkerInfo,
        experts: bool,
        stall_bound: Duration,
        standby: Option<Parked>,
    ) -> Self {
        executor.set_completion_wake(Some(channel.wake()));
        Self {
            executor,
            channel,
            info,
            experts,
            pending: VecDeque::new(),
            closing: false,
            shutdown: None,
            last_result: Instant::now(),
            stall_bound,
            stalled: false,
            channel_failed: Cell::new(false),
            standby,
        }
    }

    /// Serve until the engine closes the rank or a fatal response ends
    /// admission, which is [`Ending::Closed`].
    ///
    /// Once the engine's channel fails nothing more can be admitted or
    /// delivered: the run ends [`Ending::Orphaned`] when no admitted batch
    /// is unfinished and the jobs in flight finish before the stall bound,
    /// and [`Ending::Abandoned`] otherwise.
    pub(crate) fn run(&mut self) -> Result<Ending, WorkerError> {
        let error = match self.serve() {
            Ok(()) => return Ok(Ending::Closed),
            Err(error) if self.channel_failed.get() => error,
            Err(error) => return Err(error),
        };
        if self.settled() {
            tracing::warn!(
                "the engine's channel failed with nothing in flight; the rank closes without its peers"
            );
            return Ok(Ending::Orphaned(error));
        }
        tracing::warn!(
            unfinished_batches = self.executor.has_work(),
            jobs_in_flight = self.executor.backend().oldest_job().is_some(),
            "the engine's channel failed with work in flight, which the rank abandons"
        );
        Ok(Ending::Abandoned(error))
    }

    /// Whether nothing the rank issued can still run on a device once its
    /// engine is gone: no admitted batch is unfinished, and every job given
    /// to a lane or the builder thread has completed.
    ///
    /// An answered batch can leave jobs in flight, such as the release of
    /// its storage, whose completions announce nothing. Nothing more is
    /// admitted, so those jobs are let finish: their completions are taken
    /// as they arrive, as serving would have, until none is in flight. A
    /// job still in flight when it reaches the stall bound is stuck, and
    /// the rank is not settled.
    fn settled(&mut self) -> bool {
        loop {
            if self.executor.has_work() || self.executor.advance().is_err() {
                return false;
            }
            if self.executor.has_work() {
                return false;
            }
            let Some((_, since)) = self.executor.backend().oldest_job() else {
                return true;
            };
            let Some(left) = self.stall_bound.checked_sub(since.elapsed()) else {
                return false;
            };
            if self.executor.backend().await_completion(left).is_err() {
                return false;
            }
        }
    }

    /// Drain accepted results before acknowledging Close. A fatal result stops
    /// admission too, while preserving responses already owed to the engine.
    fn serve(&mut self) -> Result<(), WorkerError> {
        loop {
            let advanced = self.executor.advance()?;
            if self.send_ready()? || advanced {
                continue;
            }

            if self.closing && self.pending.is_empty() && !self.executor.has_work() {
                if self.experts {
                    match self.executor.backend_mut().join_expert_step(true)? {
                        // Every peer has left the group.
                        Some((_, true)) => {}
                        Some((_, false)) => continue,
                        // The join is in flight: its completion wakes.
                        None => {
                            self.wait(EXPERT_POLL)?;
                            continue;
                        }
                    }
                }

                if let Some(response) = self.shutdown.take() {
                    self.respond(&response)?;
                }

                // The rank closes its connection after this, which discards
                // what is still queued: a fatal error or this acknowledgment
                // may wait behind a large result the engine is still reading.
                self.channel.flush().map_err(|e| self.channel_error(e))?;
                return Ok(());
            }

            if !self.closing && self.pending.len() < self.executor.capacity() {
                let request = self.channel.try_recv().map_err(|e| self.channel_error(e))?;
                if let Some(request) = request {
                    self.accept(request)?;
                    continue;
                }
            }

            // A parked standby joins no expert step: its world's ranks run no
            // collective between parking and the promotion but the one that
            // restores their transports (`Parked::promote`), which each
            // reaches from here in any order.
            let joins = self.experts && self.standby.is_none();
            if joins {
                if self.awaits_own_step()? {
                    self.wait(EXPERT_POLL)?;
                    continue;
                }

                // A step that ran may be followed by another at once; an
                // empty round, or a join in flight, waits below.
                if let Some((true, _)) = self.executor.backend_mut().join_expert_step(false)? {
                    continue;
                }
            }

            if !self.pending.is_empty() || self.executor.has_work() {
                // Remote acknowledgment words have no wake source. Poll only
                // while one is outstanding; ordinary work waits on its wake.
                let timeout = if joins {
                    EXPERT_POLL
                } else if self.executor.backend().awaiting_acknowledgment()? {
                    Duration::from_millis(1)
                } else {
                    Duration::from_secs(60)
                };
                self.wait(timeout)?;
            } else if joins {
                self.wait(EXPERT_POLL)?;
            } else {
                // Nothing is owed or running, so only a request or the engine's
                // close can end this wait.
                self.wait(Duration::from_secs(60))?;
            }
        }
    }

    fn awaits_own_step(&self) -> Result<bool, WorkerError> {
        Ok(self.executor.backend().has_open_requests()?
            && (!self.pending.is_empty()
                || self.executor.has_work()
                || self.last_result.elapsed() < OWN_STEP_WAIT))
    }

    fn accept(&mut self, request: WorkerRequest) -> Result<(), WorkerError> {
        let pending = match request {
            WorkerRequest::Close { message_id } => {
                self.closing = true;
                self.shutdown = Some(WorkerResponse::Ok { message_id });
                return Ok(());
            }
            WorkerRequest::Info { message_id } => {
                PendingResponse::Ready(Box::new(WorkerResponse::Info {
                    message_id,
                    info: self.info.clone(),
                }))
            }
            WorkerRequest::Promote { message_id } => {
                let promoted = match self.standby.take() {
                    Some(parked) => parked.promote(self.executor),
                    None => Err(WorkerError::from(Error::State(
                        "only a parked standby worker is promoted",
                    ))),
                };
                PendingResponse::Ready(Box::new(match promoted {
                    Ok(()) => WorkerResponse::Ok { message_id },
                    Err(error) => self.error_response(message_id, error)?,
                }))
            }
            WorkerRequest::Submit { message_id, .. } if self.standby.is_some() => {
                PendingResponse::Ready(Box::new(self.error_response(
                    message_id,
                    WorkerError::from(Error::State("a standby worker serves once it is promoted")),
                )?))
            }
            WorkerRequest::Submit { message_id, batch } => {
                let batch = Arc::from(batch);
                let data = WorkerBatch::new(Arc::clone(&batch));
                let admitted = self.executor.admit(Batch::from_plan(&batch, data), false);
                match admitted {
                    Ok(submission) => PendingResponse::Batch {
                        message_id,
                        submission,
                    },
                    Err(error) => {
                        PendingResponse::Ready(Box::new(self.error_response(message_id, error)?))
                    }
                }
            }
        };
        self.pending.push_back(pending);
        Ok(())
    }

    fn send_ready(&mut self) -> Result<bool, WorkerError> {
        // A pending transfer or host task must not hold up independent results.
        for index in 0..self.pending.len() {
            let mut media = Vec::new();
            let response = match &self.pending[index] {
                PendingResponse::Ready(response) => (**response).clone(),
                PendingResponse::Batch {
                    message_id,
                    submission,
                } => {
                    let message_id = *message_id;
                    let result = self.executor.poll(submission);
                    match result {
                        // The result is checked once, here, where the rank
                        // builds its response; the engine does not check it
                        // again. A result that breaks the contract is the
                        // rank's own defect, and its media names are dropped
                        // with it.
                        Ok(Some(result)) => match result.output.validate() {
                            Ok(()) => {
                                media = result.media;
                                WorkerResponse::Result {
                                    message_id,
                                    result: result.output,
                                }
                            }
                            Err(invalid) => self.error_response(
                                message_id,
                                WorkerError::classify(
                                    Error::Invariant(format!("batch result: {invalid}")),
                                    None,
                                ),
                            )?,
                        },
                        Ok(None) => continue,
                        Err(error) => self.error_response(message_id, error)?,
                    }
                }
            };

            self.pending.remove(index);
            self.last_result = Instant::now();
            self.respond(&response)?;
            // A failed response drops the still-owned names. A successful
            // send leaves them available for the engine's claim.
            for source in media {
                source.into_locator();
            }
            if matches!(response, WorkerResponse::Error { ref error, .. } if error.fatal) {
                self.closing = true;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn error_response(
        &self,
        message_id: Option<u64>,
        error: WorkerError,
    ) -> Result<WorkerResponse, WorkerError> {
        Ok(WorkerResponse::Error {
            message_id,
            error: self
                .executor
                .backend()
                .response_error(RequestKind::Submit, error)?,
        })
    }

    fn respond(&mut self, response: &WorkerResponse) -> Result<(), WorkerError> {
        self.channel
            .send(response)
            .map_err(|e| self.channel_error(e))
    }

    /// Report a lane stalled past the bound, then block until a request, a
    /// completion wake, the timeout or the moment the job in flight longest
    /// reaches the bound. Fails once the engine's connection has closed: it
    /// is the engine's liveness, and nothing this rank computes can be
    /// delivered after it.
    fn wait(&mut self, timeout: Duration) -> Result<(), WorkerError> {
        let timeout = match self.watch_lanes()? {
            Some(until) => timeout.min(until),
            None => timeout,
        };
        self.channel
            .wait(timeout)
            .map_err(|e| self.channel_error(e))
    }

    /// Report the job a lane has performed for at least the stall bound,
    /// once; otherwise return how long until the job in flight longest
    /// reaches the bound. The pass that waits has taken every completion
    /// posted before its `advance`, so a job still in flight then had not
    /// completed.
    fn watch_lanes(&mut self) -> Result<Option<Duration>, WorkerError> {
        if self.stalled {
            return Ok(None);
        }
        let Some((job, since)) = self.executor.backend().oldest_job() else {
            return Ok(None);
        };
        let running = since.elapsed();
        if running < self.stall_bound {
            return Ok(Some(self.stall_bound - running));
        }

        let batch = self.executor.batch_of(job);
        let stall = self.executor.backend().stall(job, running, batch);
        tracing::error!(
            lane = stall.lane.as_deref().unwrap_or("the builder thread"),
            batch_id = ?stall.batch_id,
            route = ?stall.route,
            method = ?stall.method,
            calls = stall.calls.len(),
            running_ms = stall.running_ms,
            "a lane has stayed inside one job past the stall bound of {:?}; reporting it to the engine",
            self.stall_bound,
        );
        self.stalled = true;
        self.respond(&WorkerResponse::Stalled { stall })?;
        Ok(None)
    }

    fn channel_error(&self, error: IpcError) -> WorkerError {
        self.channel_failed.set(true);
        WorkerError::classify(Error::transport_source("the engine channel", error), None)
    }
}

impl Drop for Service<'_> {
    fn drop(&mut self) {
        self.executor.set_completion_wake(None);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use ipc::{
        Batch as Plan, BatchCommand, Channel, GenerationParams, IpcError, Listener, NewRequest,
        RequestKey, WorkerRequest, WorkerResponse,
    };
    use ipc::{RequestId, SamplingParams};

    use super::{Ending, Service};
    use crate::error::WorkerError;
    use crate::executor::tests::rank;
    use crate::{Error, ErrorCode};

    type Result<T> = std::result::Result<T, WorkerError>;

    /// How long the engine waits for one response.
    const RESPONSE_WAIT: Duration = Duration::from_secs(30);

    fn transport(error: IpcError) -> WorkerError {
        Error::transport_source("the test channel", error).into()
    }

    /// The rank's end of a connection whose engine end runs `engine` on its
    /// own thread, and the engine's outcome once `serve` returned.
    fn connect<T: Send + 'static, S>(
        engine: impl FnOnce(Channel) -> Result<T> + Send + 'static,
        serve: impl FnOnce(&mut Channel) -> Result<S>,
    ) -> Result<(Result<S>, Result<T>)> {
        let listener = Listener::bind(&["127.0.0.1:0"]).map_err(transport)?;
        let address = listener.addresses()[0].clone();
        let engine =
            std::thread::spawn(move || engine(Channel::connect(&address).map_err(transport)?));

        let deadline = Instant::now() + RESPONSE_WAIT;
        let mut channel = loop {
            if let Some(channel) = listener.try_accept().map_err(transport)? {
                break channel;
            }
            if Instant::now() > deadline {
                return Err(Error::State("the engine did not connect").into());
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let served = serve(&mut channel);
        // The rank closes its end, as it does when it exits.
        drop(channel);
        let outcome = engine
            .join()
            .map_err(|_| WorkerError::from(Error::State("the engine thread panicked")))?;
        Ok((served, outcome))
    }

    /// The next response the engine receives, or `None` once the rank has
    /// closed the connection.
    fn receive(channel: &mut Channel) -> Result<Option<WorkerResponse>> {
        let deadline = Instant::now() + RESPONSE_WAIT;
        loop {
            match channel.try_recv() {
                Ok(Some(response)) => return Ok(Some(response)),
                Ok(None) => {}
                Err(IpcError::Transport(_)) => return Ok(None),
                Err(error) => return Err(transport(error)),
            }
            if Instant::now() > deadline {
                return Err(Error::State("the rank neither responded nor closed").into());
            }
            match channel.wait(Duration::from_millis(100)) {
                Ok(()) => {}
                Err(IpcError::Transport(_)) => return Ok(None),
                Err(error) => return Err(transport(error)),
            }
        }
    }

    /// Send `request` and receive the response the rank sends next.
    fn ask(channel: &mut Channel, request: WorkerRequest) -> Result<WorkerResponse> {
        channel.send(&request).map_err(transport)?;
        receive(channel)?
            .ok_or_else(|| Error::State("the rank closed the connection unanswered").into())
    }

    /// A Submit of a lifecycle batch starting request `id` in `slot`.
    fn submit(message_id: u64, batch_id: u64, id: u64, slot: u32) -> Result<WorkerRequest> {
        let generation = GenerationParams {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 0,
            generated_token_ids: Vec::new(),
        };
        let request = NewRequest::new(
            RequestKey::new(1, RequestId(id), 1),
            slot,
            Some(generation),
            None,
            0,
        )
        .map_err(|error| Error::Invalid(error.to_string()))?;
        let mut batch = Plan::new(batch_id, Vec::new(), Vec::new());
        batch.commands = vec![BatchCommand::Start {
            request: Box::new(request),
        }];
        Ok(WorkerRequest::Submit {
            message_id: Some(message_id),
            batch: Box::new(batch),
        })
    }

    /// Serve the connection with a rank without components that queues
    /// two batches. No job of these batches stays in flight, so the stall
    /// bound (here the default, half the engine's 300 s watchdog timeout)
    /// is never reached.
    fn serve(channel: &mut Channel) -> Result<Ending> {
        let mut executor = rank(2, false)?;
        let info = executor.backend().info().clone();
        Service::new(
            &mut executor,
            channel,
            info,
            false,
            Duration::from_secs(150),
            None,
        )
        .run()
    }

    #[test]
    fn the_service_answers_each_request_and_acknowledges_close() -> Result<()> {
        let engine = |mut channel: Channel| -> Result<()> {
            let info = ask(
                &mut channel,
                WorkerRequest::Info {
                    message_id: Some(1),
                },
            )?;
            assert!(matches!(
                info,
                WorkerResponse::Info {
                    message_id: Some(1),
                    ..
                }
            ));

            // An admitted batch is answered with its result.
            let result = ask(&mut channel, submit(2, 1, 7, 1)?)?;
            assert!(matches!(
                result,
                WorkerResponse::Result { message_id: Some(2), result } if result.batch_id == 1
            ));

            // A refused batch id and a failed batch are reported under their
            // messages; neither stops the rank.
            for (message_id, request) in [(3, submit(3, 1, 8, 2)?), (4, submit(4, 2, 8, 9)?)] {
                let error = ask(&mut channel, request)?;
                assert!(
                    matches!(
                        &error,
                        WorkerResponse::Error { message_id: Some(id), error }
                            if *id == message_id && !error.fatal
                    ),
                    "{error:?}"
                );
            }
            let info = ask(
                &mut channel,
                WorkerRequest::Info {
                    message_id: Some(5),
                },
            )?;
            assert!(matches!(
                info,
                WorkerResponse::Info {
                    message_id: Some(5),
                    ..
                }
            ));

            // Close is acknowledged, then the rank ends the connection.
            let closed = ask(
                &mut channel,
                WorkerRequest::Close {
                    message_id: Some(6),
                },
            )?;
            assert_eq!(
                closed,
                WorkerResponse::Ok {
                    message_id: Some(6)
                }
            );
            assert_eq!(receive(&mut channel)?, None);
            Ok(())
        };

        let (served, engine) = connect(engine, serve)?;
        engine?;
        assert!(matches!(served?, Ending::Closed));
        Ok(())
    }

    #[test]
    fn the_service_stops_once_the_engine_has_closed_its_connection() -> Result<()> {
        // The engine goes away without asking for anything, so nothing is in
        // flight: the rank is orphaned, and the failure names the channel.
        let (served, engine) = connect(
            |channel: Channel| {
                drop(channel);
                Ok(())
            },
            serve,
        )?;
        engine?;
        let Ending::Orphaned(error) = served? else {
            return Err(Error::State("the service did not end orphaned").into());
        };
        assert_eq!(error.code, ErrorCode::ComputeError);
        Ok(())
    }

    #[test]
    fn an_engine_that_closes_after_its_answer_orphans_the_worker() -> Result<()> {
        // The engine's batch was answered, so nothing is in flight when it
        // goes away.
        let engine = |mut channel: Channel| -> Result<()> {
            let result = ask(&mut channel, submit(1, 1, 7, 1)?)?;
            assert!(matches!(result, WorkerResponse::Result { .. }));
            Ok(())
        };
        let (served, engine) = connect(engine, serve)?;
        engine?;
        assert!(matches!(served?, Ending::Orphaned(_)));
        Ok(())
    }

    #[test]
    fn a_message_that_is_not_a_request_stops_the_service_unanswered() -> Result<()> {
        let engine = |mut channel: Channel| -> Result<bool> {
            // A frame whose message is no request. The rank ends the
            // connection without a response.
            channel.send(&true).map_err(transport)?;
            Ok(receive(&mut channel)?.is_none())
        };
        let (served, unanswered) = connect(engine, serve)?;
        assert!(matches!(served?, Ending::Orphaned(_)));
        assert!(unanswered?, "the rank answered a message it could not read");
        Ok(())
    }
}
