//! Read tasks, their retirement after their device work, and physical
//! credit returns.

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};

use super::read::{NativeRead, PhysicalAccess};
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

/// A finished read whose submitted device work has yet to complete.
struct Retiring {
    backend: Arc<NativeRead>,
    nbytes: u64,
    /// The read task's failure, reported once its submitted work retired.
    error: Option<Arc<Error>>,
    access: PhysicalAccess,
}

/// The thread that retires reads once their device work completed, and
/// the channel that hands it the reads.
struct Retirer {
    reads: Sender<Retiring>,
    thread: JoinHandle<()>,
}

/// Execute reads within the rank's shared byte and read budgets. Cancellation
/// withdraws queued tasks; running reads retain credits until physical access
/// ends. Late failures poison admission and unknown completion retains backing.
///
/// A read task returns once its copy is submitted, so a transfer thread
/// moves on to the next read while the device copies. The pool's retirement
/// thread waits for each such read's device work in submission order and
/// then releases its source, returns its credits and retires its ticket.
pub(crate) struct TransferPool {
    capacity: Arc<TransferCapacity>,
    lane: HostLane<TransferRead>,
    state: Arc<Mutex<PoolState>>,
    retirer: Mutex<Option<Retirer>>,
}

impl TransferPool {
    pub(crate) fn new(capacity: Arc<TransferCapacity>, workers: usize, name: &str) -> Result<Self> {
        let limit = capacity.ticket_capacity();
        let lane = HostLane::new(limit, workers.min(limit), name)?;
        let state = Arc::new(Mutex::new(PoolState {
            error: None,
            unretired: Vec::new(),
            wake: None,
        }));

        let (reads, retiring) = mpsc::channel::<Retiring>();
        let retired = (Arc::clone(&state), Arc::clone(&capacity));
        let thread = thread::Builder::new()
            .name(format!("{name}-retire"))
            .spawn(move || {
                let (state, capacity) = retired;
                // Ends once the pool dropped its sender and every handed
                // read retired.
                for read in retiring {
                    Self::retire(&state, &capacity, read);
                }
            })
            .map_err(|error| Error::Invariant(format!("cannot start read retirement: {error}")))?;

        Ok(Self {
            capacity,
            lane,
            state,
            retirer: Mutex::new(Some(Retirer { reads, thread })),
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
                owner.finish(backend, nbytes, &task);
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

    /// Retire a finished read task: at once, or after the device work its
    /// copy submitted, on the retirement thread.
    fn finish(&self, backend: Arc<NativeRead>, nbytes: u64, task: &HostTask<TransferRead>) {
        let error = match task.completion.outcome() {
            Some(Outcome::Success(_)) => None,
            Some(Outcome::Failed(error)) => Some(error),
            Some(Outcome::Cancelled) => Some(Arc::new(Error::Resource(
                "transfer read was cancelled before submission",
            ))),
            None => return,
        };
        let Some(access) = backend.read.take_access() else {
            Self::finished(&self.state, &self.capacity, &backend, nbytes, error);
            return;
        };

        let handed = {
            let retirer = self.retirer.lock().unwrap_or_else(PoisonError::into_inner);
            let read = Retiring {
                backend,
                nbytes,
                error,
                access,
            };
            match retirer.as_ref() {
                Some(retirer) => retirer.reads.send(read).map_err(|refused| refused.0),
                None => Err(read),
            }
        };
        if let Err(read) = handed {
            // No retirement thread can wait for the work any more: keep it
            // and everything it addresses, as for an undrained copy.
            let Retiring {
                backend,
                nbytes,
                error,
                access,
            } = read;
            access.abandon(&backend);
            let error = error.unwrap_or_else(|| {
                Arc::new(Error::Resource(
                    "transfer read retired after its pool closed",
                ))
            });
            Self::finished(&self.state, &self.capacity, &backend, nbytes, Some(error));
        }
    }

    /// Wait for a read's device work, then retire it.
    fn retire(state: &Mutex<PoolState>, capacity: &TransferCapacity, read: Retiring) {
        let Retiring {
            backend,
            nbytes,
            error,
            access,
        } = read;
        let waited = access.retire(&backend);
        let error = match (error, waited) {
            (Some(error), Err(cleanup)) => {
                report(&cleanup);
                Some(error)
            }
            (None, Err(cleanup)) => Some(Arc::new(cleanup)),
            (error, Ok(())) => error,
        };
        Self::finished(state, capacity, &backend, nbytes, error);
    }

    fn finished(
        state: &Mutex<PoolState>,
        capacity: &TransferCapacity,
        backend: &Arc<NativeRead>,
        nbytes: u64,
        mut error: Option<Arc<Error>>,
    ) {
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
                    state.unretired.push(Arc::clone(backend));
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

    /// Join reads, then their retirements, before destroying their streams.
    /// Failed physical retirement remains retained and its original error is
    /// returned to the caller.
    pub(crate) fn close(&self) -> Option<Arc<Error>> {
        let mut errors = self.lane.close();
        // Every read task finished and handed its work over: the retirement
        // thread ends once it retired all of it.
        let retirer = self
            .retirer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let joined = retirer.and_then(|Retirer { reads, thread }| {
            drop(reads);
            thread
                .join()
                .err()
                .map(|_| Error::State("transfer read retirement panicked"))
        });
        errors.extend(joined);
        let mut errors = errors.into_iter();
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
        // Dropping the sender lets the retirement thread finish the reads
        // it holds and end; a destructor does not wait for it.
        drop(
            self.retirer
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
        let state = &mut *self.state.lock().unwrap_or_else(PoisonError::into_inner);
        for read in state.unretired.drain(..) {
            std::mem::forget(read);
        }
    }
}

/// Log a read cleanup failure that has no caller left to return it to.
fn report(error: &Error) {
    tracing::warn!(target: "uniserve_worker.transfer", "transfer read cleanup failed: {error}");
}
