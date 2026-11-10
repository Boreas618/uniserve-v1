//! Shared operation results, waits, and completion observers.

use crate::Error;

/// How an operation of the worker ended, with the worker's own error when it failed.
///
/// Physical storage requires a successful retirement signal; a task result alone does not
/// establish that a device has stopped.
///
/// The variants are `ipc::execution::Outcome`'s: a submission's outcome is the same type over a
/// [`ipc::execution::Fault`], which the worker's error converts into where a submission completes.
pub type Outcome<T = ()> = ipc::execution::Outcome<T, Error>;

/// One completion shared by an operation and its consumers, failing with the worker's own error.
///
/// `C` is the observer type: a shared [`crate::transfer::Notify`] for physical retirements, which
/// one observer may watch together with other signals, or a one-shot boxed closure for host task
/// results. `T` is the operation's value. The type is `ipc::execution::Completion`, which a
/// submission is built on too.
pub type Completion<C, T = ()> = ipc::execution::Completion<C, T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;
    use crate::transfer::{Notify, Retirement, notify};
    use std::sync::Arc;
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
