//! A one-shot timer the server's loop waits on as it waits on its sockets and pidfds: a
//! `timerfd(2)` on the monotonic clock, readable once the time it was armed for has come. Each
//! waiter that waits for a time (a child's restart, a shoot-down's next question to NVML) owns
//! one, so the loop's `poll(2)` waits on descriptors alone and computes no timeout.

use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::time::Duration;

use rustix::time::{
    Itimerspec, TimerfdClockId, TimerfdFlags, TimerfdTimerFlags, Timespec, timerfd_create,
    timerfd_gettime, timerfd_settime,
};

/// A one-shot monotonic timer.
pub(crate) struct Timer(OwnedFd);

impl Timer {
    /// A timer that is not armed.
    ///
    /// # Errors
    ///
    /// Fails when the kernel cannot create the timer.
    pub(crate) fn new() -> io::Result<Self> {
        let flags = TimerfdFlags::CLOEXEC | TimerfdFlags::NONBLOCK;
        Ok(Self(timerfd_create(TimerfdClockId::Monotonic, flags)?))
    }

    /// Arms the timer to fire once `after` from now, replacing any earlier arming. A zero
    /// duration fires at once (the kernel reads a zero time as disarming, so it is one
    /// nanosecond).
    ///
    /// # Errors
    ///
    /// Fails when the duration does not fit the kernel's time.
    pub(crate) fn arm(&self, after: Duration) -> io::Result<()> {
        let after = after.max(Duration::from_nanos(1));
        let value = Timespec::try_from(after).map_err(|_| io::Error::other("a time too far"))?;
        let zero = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let spec = Itimerspec {
            it_interval: zero,
            it_value: value,
        };
        timerfd_settime(&self.0, TimerfdTimerFlags::empty(), &spec)?;
        Ok(())
    }

    /// Disarms the timer and forgets a firing not yet taken.
    pub(crate) fn disarm(&self) {
        let zero = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let spec = Itimerspec {
            it_interval: zero,
            it_value: zero,
        };
        // Disarming a timer the server created cannot fail.
        let _ = timerfd_settime(&self.0, TimerfdTimerFlags::empty(), &spec);
        let _ = self.fired();
    }

    /// Whether the timer fired since it was armed, taking the firing: it then reads as not
    /// fired until it is armed again and fires.
    pub(crate) fn fired(&self) -> bool {
        let mut expirations = [0_u8; 8];
        matches!(rustix::io::read(&self.0, &mut expirations), Ok(8))
    }

    /// The time left before the timer fires; `None` when it is not armed or already fired.
    pub(crate) fn remaining(&self) -> Option<Duration> {
        let spec = timerfd_gettime(&self.0).ok()?;
        let value = spec.it_value;
        let left = Duration::new(
            u64::try_from(value.tv_sec).ok()?,
            u32::try_from(value.tv_nsec).ok()?,
        );
        (!left.is_zero()).then_some(left)
    }
}

impl AsFd for Timer {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
