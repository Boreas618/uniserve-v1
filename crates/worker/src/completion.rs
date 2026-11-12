//! How an operation of the worker ended, and the completion its waiters share.
//!
//! Both are `ipc::execution`'s types over the worker's own error: a host task of a
//! [`ThreadPool`](crate::thread_pool::ThreadPool) ends with an [`Outcome`] and is waited for
//! through a [`Completion`], as a submission is over an [`ipc::execution::Fault`].

use crate::Error;

/// How an operation of the worker ended, with the worker's own error when it failed.
///
/// Storage a device addressed is released only on a success: a failure or a cancellation does not
/// establish that the device stopped addressing it.
///
/// The variants are `ipc::execution::Outcome`'s: a submission ends with the same type over an
/// [`ipc::execution::Fault`], which the worker's error converts into where a submission completes.
pub type Outcome<T = ()> = ipc::execution::Outcome<T, Error>;

/// One completion shared by an operation and its waiters, failing with the worker's own error.
///
/// `C` is the observer type, such as an [`ipc::execution::Waker`] that signals a waiter, and `T`
/// the operation's value. The type is `ipc::execution::Completion`, which a submission is built
/// on too.
pub type Completion<C, T = ()> = ipc::execution::Completion<C, T, Error>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use ipc::execution::Waker;

    use super::*;
    use crate::Result;

    #[test]
    fn a_subscriber_racing_completion_is_notified_exactly_once() -> Result<()> {
        for _ in 0..64 {
            let completion = Completion::<Waker>::default();
            let calls = Arc::new(AtomicUsize::new(0));
            let observer: Waker = {
                let calls = Arc::clone(&calls);
                Arc::new(move || {
                    calls.fetch_add(1, Ordering::Relaxed);
                })
            };

            thread::scope(|scope| -> Result<()> {
                let subscriber = scope.spawn(|| {
                    if let Some(observer) = completion.subscribe(observer) {
                        observer();
                    }
                });
                for observer in completion.complete(Ok(()))? {
                    observer();
                }
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
    fn a_failure_and_a_cancellation_are_done_and_not_succeeded() -> Result<()> {
        let failed = Completion::<Waker>::default();
        assert!(!failed.done());
        failed.complete(Err(Error::transport("device completion unknown")))?;
        assert!(failed.done());
        assert!(!failed.succeeded());
        let Some(Outcome::Failed(error)) = failed.outcome() else {
            return Err(Error::State("the producer error was lost"));
        };
        assert!(error.to_string().contains("device completion unknown"));

        let cancelled = Completion::<Waker>::default();
        assert!(cancelled.cancel().0);
        assert!(cancelled.done());
        assert!(!cancelled.succeeded());
        assert!(matches!(cancelled.outcome(), Some(Outcome::Cancelled)));
        assert!(cancelled.cancel().0);
        assert!(cancelled.complete(Ok(())).is_err());
        Ok(())
    }
}
