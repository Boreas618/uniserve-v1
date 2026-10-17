//! Read tasks, physical credit returns and per-thread copy streams.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::read::NativeRead;
use super::{Notify, ReadReservation, TransferCapacity, notify};
use crate::host::{HostAction, HostLane, HostTask};
use crate::{Error, Outcome, Result};

/// A read action submitted through the common native host executor.
pub(crate) struct TransferRead {
    backend: Arc<NativeRead>,
}

impl HostAction for TransferRead {
    type Output = ();

    fn run(&self) -> Result<()> {
        if self.backend.with_ticket(|ticket| ticket.is_cancelled()) {
            return Err(Error::Resource("transfer read was cancelled"));
        }

        let _range = crate::profiling::range(c"uniserve.transfer", None);
        self.backend.read()
    }

    fn release(&self) -> Result<()> {
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        // Reads own their input accesses through read completion.
        Outcome::Success(())
    }
}

struct PoolState {
    error: Option<Arc<Error>>,
    unretired: Vec<Arc<NativeRead>>,
    wake: Option<Notify>,
}

/// Execute reads within the rank's shared byte and read budgets. Cancellation
/// withdraws queued tasks; running reads retain credits until physical access
/// ends. Late failures poison admission and unknown completion retains backing.
pub(crate) struct TransferPool {
    capacity: Arc<TransferCapacity>,
    lane: HostLane<TransferRead>,
    state: Mutex<PoolState>,
}

impl TransferPool {
    pub(crate) fn new(capacity: Arc<TransferCapacity>, workers: usize, name: &str) -> Result<Self> {
        let limit = capacity.ticket_capacity();
        Ok(Self {
            capacity,
            lane: HostLane::new(limit, workers.min(limit), name)?,
            state: Mutex::new(PoolState {
                error: None,
                unretired: Vec::new(),
                wake: None,
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn set_completion_wake(&self, wake: Option<Notify>) {
        let previous = std::mem::replace(&mut self.state().wake, wake);
        drop(previous);
    }

    pub(crate) fn completion_wake(&self) -> Option<Notify> {
        self.state().wake.clone()
    }

    /// Reserve the read's credits and host task before numerical preparation.
    /// The caller then records destination handoff and submits the task, or
    /// abandons it after preparation fails. Cancellation returns unused credits.
    pub(crate) fn reserve(
        self: &Arc<Self>,
        backend: NativeRead,
        nbytes: u64,
        reservation: Option<&ReadReservation>,
    ) -> Result<Arc<HostTask<TransferRead>>> {
        if self.error().is_some() {
            return Err(Error::Resource("transfer pool has failed"));
        }
        self.capacity.take_read(reservation)?;

        if let Err(error) = self.capacity.acquire(nbytes) {
            self.capacity.return_reads(1)?;
            return Err(error);
        }

        let backend = Arc::new(backend);
        let task = match self.lane.reserve() {
            Ok(task) => task,
            Err(error) => {
                self.return_capacity(nbytes)?;
                return Err(error);
            }
        };
        if let Err(error) = task.configure(TransferRead {
            backend: Arc::clone(&backend),
        }) {
            if let Err(cleanup) = task.cancel(true) {
                report(&cleanup);
            }
            self.return_capacity(nbytes)?;
            return Err(error);
        }

        let observed = Arc::downgrade(&task);
        let owner = Arc::clone(self);
        let immediate = task.completion.subscribe(Box::new(move || {
            // The worker or cancelling caller retains the task while notifying.
            if let Some(task) = observed.upgrade() {
                Self::finished(&owner.state, &owner.capacity, backend, nbytes, &task);
            }
        }));
        if let Some(callback) = immediate {
            callback();
        }

        Ok(task)
    }

    fn return_capacity(&self, nbytes: u64) -> Result<()> {
        self.capacity.release(nbytes)?;
        self.capacity.return_reads(1)
    }

    fn finished(
        state: &Mutex<PoolState>,
        capacity: &TransferCapacity,
        backend: Arc<NativeRead>,
        nbytes: u64,
        task: &HostTask<TransferRead>,
    ) {
        let mut error = match task.completion.outcome() {
            Some(Outcome::Success(_)) => None,
            Some(Outcome::Failed(error)) => Some(error),
            Some(Outcome::Cancelled) => Some(Arc::new(Error::Resource(
                "transfer read was cancelled before submission",
            ))),
            None => return,
        };
        let undrained = backend.with_ticket(|ticket| ticket.undrained());
        let release_error = if undrained {
            None
        } else {
            backend.release().err()
        };
        let retained = undrained || release_error.is_some();
        if let Some(cleanup) = release_error {
            if error.is_none() {
                error = Some(Arc::new(cleanup));
            } else {
                report(&cleanup);
            }
        }

        let (late, callbacks) = backend.with_ticket(|ticket| {
            error.as_ref().map_or((false, Vec::new()), |error| {
                ticket.fail_shared(Arc::clone(error))
            })
        });
        let wake = {
            let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(error) = error.as_ref().filter(|_| late || retained) {
                if state.error.is_none() {
                    state.error = Some(Arc::clone(error));
                }
                if retained {
                    state.unretired.push(Arc::clone(&backend));
                }
            }
            late.then(|| state.wake.clone()).flatten()
        };
        notify(callbacks);
        if let Some(wake) = wake {
            wake();
        }

        if !retained {
            let retired = capacity
                .release(nbytes)
                .and_then(|()| capacity.return_reads(1))
                .and_then(|()| backend.with_ticket(|ticket| ticket.retire()));
            match retired {
                Ok(callbacks) => notify(callbacks),
                Err(error) => report(&error),
            }
        }
    }

    pub(crate) fn error(&self) -> Option<Arc<Error>> {
        self.state().error.clone()
    }

    /// Join reads before destroying their streams. Failed physical retirement
    /// remains retained and its original error is returned to the caller.
    pub(crate) fn close(&self) -> Option<Arc<Error>> {
        let mut errors = self.lane.close().into_iter();
        let first = errors.next().map(Arc::new);
        let error = {
            let mut state = self.state();
            if state.error.is_none() {
                state.error = first.clone();
            }
            state.error.clone()
        };
        for cleanup in errors {
            report(&cleanup);
        }
        error
    }
}

impl Drop for TransferPool {
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion retains backing until process teardown"
    )]
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        for read in state.unretired.drain(..) {
            std::mem::forget(read);
        }
    }
}

/// Log a read cleanup failure that has no caller left to return it to.
fn report(error: &Error) {
    tracing::warn!(target: "uniserve_worker.transfer", "transfer read cleanup failed: {error}");
}
