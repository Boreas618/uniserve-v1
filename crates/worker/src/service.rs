//! Rank-channel admission, result delivery, and orderly worker shutdown.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use uniserve_ipc::{Channel, IpcError, RequestKind, WorkerInfo, WorkerRequest, WorkerResponse};

use crate::error::WorkerError;
use crate::executor::backend::WorkerBatch;
use crate::{Batch, Error, Executor, Submission};

const EXPERT_POLL: Duration = Duration::from_micros(200);
// Keep a rank with open requests available for its engine's next batch before
// joining an empty expert step. The engine needs time to consume the result
// and schedule its next batch even when it still has admitted requests.
const OWN_STEP_WAIT: Duration = Duration::from_millis(8);

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
pub(crate) struct Service<'a> {
    executor: &'a mut Executor,
    channel: &'a mut Channel,
    info: WorkerInfo,
    experts: bool,
    pending: VecDeque<PendingResponse>,
    closing: bool,
    shutdown: Option<WorkerResponse>,
    last_result: Instant,
}

impl<'a> Service<'a> {
    pub(crate) fn new(
        executor: &'a mut Executor,
        channel: &'a mut Channel,
        info: WorkerInfo,
        experts: bool,
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
        }
    }

    /// Drain accepted results before acknowledging Close. A fatal result stops
    /// admission too, while preserving responses already owed to the engine.
    pub(crate) fn run(&mut self) -> Result<(), WorkerError> {
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
                // may wait behind a large result the head is still reading.
                self.channel.flush().map_err(|e| self.channel_error(e))?;
                return Ok(());
            }

            if !self.closing && self.pending.len() < self.info.queue_depth as usize {
                let request = self.channel.try_recv().map_err(|e| self.channel_error(e))?;
                if let Some(request) = request {
                    self.accept(request)?;
                    continue;
                }
            }

            if self.experts {
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
                let timeout = if self.experts {
                    EXPERT_POLL
                } else if self.executor.backend().awaiting_acknowledgment()? {
                    Duration::from_millis(1)
                } else {
                    Duration::from_secs(60)
                };
                self.wait(timeout)?;
            } else if self.experts {
                self.wait(EXPERT_POLL)?;
            } else {
                // Nothing is owed or running, so only a request or the head's
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
                        Ok(Some(result)) => {
                            media = result.media;
                            WorkerResponse::Result {
                                message_id,
                                result: result.output,
                            }
                        }
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

    /// Block until a request, a completion wake or the timeout. Fails once
    /// the head's connection has closed: it is the head's liveness, and
    /// nothing this rank computes can be delivered after it.
    fn wait(&mut self, timeout: Duration) -> Result<(), WorkerError> {
        self.channel
            .wait(timeout)
            .map_err(|e| self.channel_error(e))
    }

    fn channel_error(&self, error: IpcError) -> WorkerError {
        WorkerError::classify(Error::Transport(error.to_string()), None)
    }
}

impl Drop for Service<'_> {
    fn drop(&mut self) {
        self.executor.set_completion_wake(None);
    }
}
