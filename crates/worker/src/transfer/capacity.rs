//! A rank's shared transfer byte budget and read credits.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::{Notify, notify};
use crate::{Error, Result, lock};

struct CapacityState {
    used: u64,
    reads_free: usize,
    returns: u64,
    waiters: Vec<Notify>,
}

/// Shared byte reservations and physical read credits for one worker rank.
/// Waiters for returned credits are notified after the capacity lock is
/// released.
pub struct TransferCapacity {
    capacity: u64,
    ticket_capacity: usize,
    state: Mutex<CapacityState>,
}

impl TransferCapacity {
    pub fn new(capacity: u64, ticket_capacity: usize) -> Result<Self> {
        if capacity == 0 || ticket_capacity == 0 {
            return Err(Error::Invalid(
                "transfer byte and read capacities must be positive".into(),
            ));
        }

        Ok(Self {
            capacity,
            ticket_capacity,
            state: Mutex::new(CapacityState {
                used: 0,
                reads_free: ticket_capacity,
                returns: 0,
                waiters: Vec::new(),
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, CapacityState> {
        lock(&self.state)
    }

    pub(crate) fn ticket_capacity(&self) -> usize {
        self.ticket_capacity
    }

    pub fn used(&self) -> u64 {
        self.state().used
    }

    pub(crate) fn acquire(&self, amount: u64) -> Result<()> {
        let mut state = self.state();
        if amount > self.capacity - state.used {
            return Err(Error::Resource("transfer byte capacity is exhausted"));
        }

        state.used += amount;
        Ok(())
    }

    pub(crate) fn release(&self, amount: u64) -> Result<()> {
        let mut state = self.state();
        if amount > state.used {
            return Err(Error::State(
                "transfer byte release exceeds the live reservation",
            ));
        }

        state.used -= amount;
        Ok(())
    }

    /// Take the complete fan-out before starting any reads. Backpressure
    /// includes the return sequence used to subscribe without missing a wake.
    pub(crate) fn take_reads(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Err(Error::Invalid(
                "a read ticket reservation takes at least one".into(),
            ));
        }
        if count > self.ticket_capacity {
            return Err(Error::Unsupported(format!(
                "a fetch of {count} reads exceeds the rank's {} read tickets",
                self.ticket_capacity
            )));
        }

        let mut state = self.state();
        if count > state.reads_free {
            return Err(Error::ReadBackpressure {
                returns: state.returns,
            });
        }

        state.reads_free -= count;
        Ok(())
    }

    /// Admit one borrowed or copied read from the rank's shared reservation.
    pub(crate) fn take_read(self: &Arc<Self>, reservation: Option<&ReadReservation>) -> Result<()> {
        if let Some(reservation) = reservation {
            if !Arc::ptr_eq(reservation.capacity(), self) {
                return Err(Error::Invalid(
                    "read reservation belongs to another transfer capacity".into(),
                ));
            }
            reservation.use_read()
        } else {
            self.take_reads(1)
        }
    }

    /// Return credits only after a physical read retires, or if unused.
    pub(crate) fn return_reads(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let mut state = self.state();
        if count > self.ticket_capacity - state.reads_free {
            return Err(Error::State("read ticket return exceeds the tickets taken"));
        }

        state.reads_free += count;
        state.returns = state.returns.wrapping_add(1);
        let callbacks = std::mem::take(&mut state.waiters);
        drop(state);

        notify(callbacks);
        Ok(())
    }

    /// Register for a credit return, or return the callback for immediate
    /// dispatch if credits have already returned since the refused fetch.
    pub(crate) fn notify_reads_returned(&self, callback: Notify, after: u64) -> Option<Notify> {
        let mut state = self.state();
        if state.returns != after {
            return Some(callback);
        }

        state.waiters.push(callback);
        None
    }
}

/// Bytes of a rank's transfer budget held by one export. The bytes return
/// when the reservation is released or dropped: an attempt that fails on any
/// path gives them back, and an export's backing keeps them until it retires.
pub(crate) struct ByteReservation {
    capacity: Arc<TransferCapacity>,
    bytes: u64,
}

impl ByteReservation {
    pub(crate) fn new(capacity: &Arc<TransferCapacity>, bytes: u64) -> Result<Self> {
        capacity.acquire(bytes)?;
        Ok(Self {
            capacity: Arc::clone(capacity),
            bytes,
        })
    }

    /// Return the bytes, reporting an accounting failure to the caller.
    pub(crate) fn release(mut self) -> Result<()> {
        let bytes = std::mem::take(&mut self.bytes);
        self.capacity.release(bytes)
    }
}

impl Drop for ByteReservation {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        if let Err(error) = self.capacity.release(self.bytes) {
            tracing::warn!("failed to return transfer bytes: {error}");
        }
    }
}

/// A fetch's reserved read credits. Each submission takes one credit; close
/// and destruction return only those which have not been handed to a read.
pub struct ReadReservation {
    capacity: Arc<TransferCapacity>,
    unused: AtomicUsize,
}

impl ReadReservation {
    pub(crate) fn new(capacity: Arc<TransferCapacity>, count: usize) -> Result<Self> {
        capacity.take_reads(count)?;
        Ok(Self {
            capacity,
            unused: AtomicUsize::new(count),
        })
    }

    pub(crate) fn capacity(&self) -> &Arc<TransferCapacity> {
        &self.capacity
    }

    pub(crate) fn use_read(&self) -> Result<()> {
        self.unused
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |unused| {
                unused.checked_sub(1)
            })
            .map(|_| ())
            .map_err(|_| Error::State("read reservation has no ticket left"))
    }

    pub(crate) fn close(&self) -> Result<()> {
        // Only the counter crosses threads; read storage has its own fences.
        let unused = self.unused.swap(0, Ordering::Relaxed);
        self.capacity.return_reads(unused)
    }
}

impl Drop for ReadReservation {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::warn!("failed to return unused read credits: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Barrier, mpsc};
    use std::thread;

    use super::*;

    #[test]
    fn refused_reservations_leave_byte_and_read_budgets_available() -> Result<()> {
        let capacity = Arc::new(TransferCapacity::new(8, 2)?);
        capacity.acquire(4)?;
        assert!(matches!(
            capacity.acquire(u64::MAX),
            Err(Error::Resource(_))
        ));
        assert_eq!(capacity.used(), 4);
        capacity.acquire(4)?;
        assert!(capacity.release(9).is_err());
        assert_eq!(capacity.used(), 8);
        capacity.release(8)?;

        capacity.take_reads(1)?;
        assert!(matches!(
            ReadReservation::new(Arc::clone(&capacity), 2),
            Err(Error::ReadBackpressure { .. })
        ));
        assert!(matches!(
            ReadReservation::new(Arc::clone(&capacity), 3),
            Err(Error::Unsupported(_))
        ));

        // The refused fan-out must not consume its available first read.
        let remaining = ReadReservation::new(Arc::clone(&capacity), 1)?;
        remaining.use_read()?;
        assert!(remaining.use_read().is_err());
        remaining.close()?;
        capacity.return_reads(2)?;
        assert!(capacity.return_reads(1).is_err());
        capacity.take_reads(2)?;
        capacity.return_reads(2)?;

        let other = Arc::new(TransferCapacity::new(8, 2)?);
        let reservation = ReadReservation::new(Arc::clone(&capacity), 1)?;
        assert!(other.take_read(Some(&reservation)).is_err());
        capacity.take_read(Some(&reservation))?;
        capacity.return_reads(1)?;
        other.take_reads(2)?;
        other.return_reads(2)
    }

    #[test]
    fn dropping_unused_credits_wakes_reentrant_and_late_subscribers() -> Result<()> {
        let capacity = Arc::new(TransferCapacity::new(8, 2)?);
        let reservation = ReadReservation::new(Arc::clone(&capacity), 2)?;
        reservation.use_read()?;

        let Err(Error::ReadBackpressure { returns }) = capacity.take_reads(1) else {
            return Err(Error::State(
                "the reserved fan-out must occupy both read credits",
            ));
        };
        let (tx, rx) = mpsc::channel();
        let retry_capacity = Arc::clone(&capacity);
        // The subscriber reserves again from inside the notification.
        let retry: Notify = Arc::new(move || {
            let _ = tx.send(ReadReservation::new(Arc::clone(&retry_capacity), 1));
        });
        assert!(capacity.notify_reads_returned(retry, returns).is_none());

        thread::spawn(move || drop(reservation))
            .join()
            .map_err(|_| Error::State("reservation cleanup panicked"))?;
        let retry = rx
            .recv()
            .map_err(|_| Error::State("the returned credit woke no subscriber"))??;
        assert!(matches!(
            capacity.take_reads(1),
            Err(Error::ReadBackpressure { .. })
        ));

        // A subscriber arriving after the return is told to run at once.
        let late: Notify = Arc::new(|| {});
        assert!(capacity.notify_reads_returned(late, returns).is_some());

        retry.close()?;
        retry.close()?;
        // The submitted read still owns one credit after both reservations
        // close.
        assert!(matches!(
            capacity.take_reads(2),
            Err(Error::ReadBackpressure { .. })
        ));
        capacity.return_reads(1)?;
        capacity.take_reads(2)?;
        capacity.return_reads(2)
    }

    #[test]
    fn closing_a_reservation_racing_submission_returns_each_unused_credit_once() -> Result<()> {
        let capacity = Arc::new(TransferCapacity::new(8, 8)?);
        let reservation = ReadReservation::new(Arc::clone(&capacity), 8)?;
        let barrier = Barrier::new(9);

        let submitted = thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        reservation.use_read().is_ok()
                    })
                })
                .collect();

            barrier.wait();
            reservation.close()?;
            threads
                .into_iter()
                .map(|thread| {
                    thread
                        .join()
                        .map(usize::from)
                        .map_err(|_| Error::State("read submission panicked"))
                })
                .sum::<Result<usize>>()
        })?;

        capacity.return_reads(submitted)?;
        capacity.take_reads(8)?;
        capacity.return_reads(8)
    }
}
