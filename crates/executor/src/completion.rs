//! Shared operation results, waits, and completion observers.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{Error, Result, lock};

/// The operation's result. Physical storage requires a successful retirement
/// signal; a task result alone does not establish that a device has stopped.
#[derive(Debug)]
pub enum Outcome<T = ()> {
    Success(T),
    Failed(Arc<Error>),
    Cancelled,
}

impl<T: Clone> Clone for Outcome<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Success(value) => Self::Success(value.clone()),
            Self::Failed(error) => Self::Failed(Arc::clone(error)),
            Self::Cancelled => Self::Cancelled,
        }
    }
}

struct State<C, T> {
    outcome: Option<Outcome<T>>,
    callbacks: Vec<C>,
}

/// One completion shared by an operation and its consumers.
///
/// `C` is the observer type: a shared [`crate::transfer::Notify`] for
/// physical retirements, which one observer may watch together with other
/// signals, or a one-shot boxed closure for host task results. Completing an
/// operation returns its callbacks so the owner can dispatch them after
/// releasing its lock. `T` is the operation's value.
pub struct Completion<C, T = ()> {
    state: Mutex<State<C, T>>,
}

impl<C, T> Default for Completion<C, T> {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                outcome: None,
                callbacks: Vec::new(),
            }),
        }
    }
}

impl<C, T: Clone> Completion<C, T> {
    fn state(&self) -> MutexGuard<'_, State<C, T>> {
        // No user code mutates state under the lock. A panicking reference
        // visitor cannot leave a partially completed operation.
        lock(&self.state)
    }

    pub fn done(&self) -> bool {
        self.state().outcome.is_some()
    }

    pub fn succeeded(&self) -> bool {
        matches!(self.state().outcome, Some(Outcome::Success(_)))
    }

    pub(crate) fn outcome(&self) -> Option<Outcome<T>> {
        self.state().outcome.clone()
    }

    /// Register a callback, or return it for immediate dispatch if already done.
    pub(crate) fn subscribe(&self, callback: C) -> Option<C> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return Some(callback);
        }

        state.callbacks.push(callback);
        None
    }

    pub(crate) fn complete(&self, result: Result<T>) -> Result<Vec<C>> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return Err(Error::State("operation has already completed"));
        }

        state.outcome = Some(match result {
            Ok(value) => Outcome::Success(value),
            Err(error) => Outcome::Failed(Arc::new(error)),
        });
        Ok(std::mem::take(&mut state.callbacks))
    }

    /// Cancel an unresolved completion. This does not stop its producer.
    pub(crate) fn cancel(&self) -> (bool, Vec<C>) {
        let mut state = self.state();
        match state.outcome {
            Some(Outcome::Cancelled) => return (true, Vec::new()),
            Some(_) => return (false, Vec::new()),
            None => {}
        }

        state.outcome = Some(Outcome::Cancelled);
        (true, std::mem::take(&mut state.callbacks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{Notify, Retirement, notify};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    #[test]
    fn a_subscriber_racing_completion_is_notified_exactly_once() -> Result<()> {
        for _ in 0..64 {
            let completion = Retirement::default();
            let calls = Arc::new(AtomicUsize::new(0));
            let observer: Notify = {
                let calls = Arc::clone(&calls);
                Arc::new(move || {
                    calls.fetch_add(1, Ordering::Relaxed);
                })
            };

            thread::scope(|scope| -> Result<()> {
                let subscriber = scope.spawn(|| {
                    if let Some(callback) = completion.subscribe(observer) {
                        callback();
                    }
                });
                notify(completion.complete(Ok(()))?);
                subscriber
                    .join()
                    .map_err(|_| Error::State("subscriber panicked"))
            })?;

            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert!(completion.succeeded());
            assert!(!completion.cancel().0);
            assert!(completion.complete(Ok(())).is_err());
        }
        Ok(())
    }

    #[test]
    fn failure_and_cancellation_do_not_retire_storage() -> Result<()> {
        let failed = Retirement::default();
        assert!(!failed.done());
        failed.complete(Err(Error::transport("device completion unknown")))?;
        assert!(failed.done());
        assert!(!failed.succeeded());
        let Some(Outcome::Failed(error)) = failed.outcome() else {
            return Err(Error::State("the producer error was lost"));
        };
        assert!(error.to_string().contains("device completion unknown"));

        let cancelled = Retirement::default();
        assert!(cancelled.cancel().0);
        assert!(cancelled.done());
        assert!(!cancelled.succeeded());
        assert!(matches!(cancelled.outcome(), Some(Outcome::Cancelled)));
        assert!(cancelled.cancel().0);
        assert!(cancelled.complete(Ok(())).is_err());
        Ok(())
    }
}
