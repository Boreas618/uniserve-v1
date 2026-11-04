//! Waiting through a Program Server restart (design §7.5).
//!
//! The Program Server is not on the serving path: it restarts directly, and
//! nothing a request needs waits on it. Only an epoch's preparation does:
//! the leader's daemon `prepare`s the target epoch's programs and each main
//! process `resolve`s its modules ([`crate::epoch::FilledPrograms`]), and a
//! restart that lands between, or in the middle of, those calls must make
//! them wait rather than fail the preparation.
//!
//! A restart shows up as an I/O error on the socket: the path is refused
//! because the server has not bound it yet, the peer closes the connection
//! mid-answer, or the socket file is gone while it restarts.
//! [`through_restart`] re-runs the attempt, opening a fresh connection each
//! time — a restart ends the old one — until the server answers or the wait
//! is spent. Everything the server itself decides is returned at once: a
//! refusal, a module that cannot be built, an unknown revision. Retrying
//! those would only repeat the decision.
//!
//! The attempt must be idempotent: `prepare` selects the revision of every
//! program, and a program's revision names its content and is never
//! overwritten (§5.1), so a retried `prepare` chooses the same programs; a
//! retried `resolve` returns the module the store already holds or rebuilds
//! it. An attempt abandoned by a restart may have committed modules for its
//! programs; the retry finds them in the store. Progress an abandoned
//! attempt reported is not undone: a phase's progress report may name a step
//! twice.

use std::io::ErrorKind;
use std::num::NonZeroU32;
use std::time::Duration;

/// How long the wait tries, and how long it waits between attempts.
///
/// The total is `attempts × delay`: the default waits about thirty seconds,
/// which is longer than a Program Server restart takes and short enough that
/// a server that stays down fails the preparation promptly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wait {
    /// How many attempts the wait makes, the first one included.
    pub attempts: NonZeroU32,
    /// How long it waits between two attempts.
    pub delay: Duration,
}

impl Wait {
    /// The wait of `attempts` attempts, `delay` apart.
    pub fn new(attempts: u32, delay: Duration) -> Self {
        Self {
            attempts: NonZeroU32::new(attempts).unwrap_or(NonZeroU32::MIN),
            delay,
        }
    }

    /// How long a wait that never gets an answer spans.
    pub fn total(&self) -> Duration {
        self.delay.saturating_mul(self.attempts.get())
    }
}

impl Default for Wait {
    fn default() -> Self {
        Self::new(300, Duration::from_millis(100))
    }
}

/// Whether an error says the Program Server is not there now, which a restart
/// causes, and the wait retries.
///
/// These are the shapes a Unix-socket client sees while its peer restarts:
/// the path is not bound yet ([`ErrorKind::NotFound`]) or refuses the
/// connection ([`ErrorKind::ConnectionRefused`]), the peer closed or reset
/// the connection before answering ([`ErrorKind::UnexpectedEof`],
/// [`ErrorKind::BrokenPipe`], [`ErrorKind::ConnectionReset`],
/// [`ErrorKind::NotConnected`]).
///
/// A timeout is not one of them: the server is there and busy, and repeating
/// a `prepare` that is still building would add a second build, not wait for
/// the first.
pub fn away(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::NotFound
            | ErrorKind::ConnectionRefused
            | ErrorKind::ConnectionReset
            | ErrorKind::BrokenPipe
            | ErrorKind::UnexpectedEof
            | ErrorKind::NotConnected
    )
}

/// An error that says whether the Program Server was away when it happened.
pub trait ServerAway {
    /// Whether the error is one a restart causes, which the wait retries.
    fn server_away(&self) -> bool;
}

impl ServerAway for std::io::Error {
    fn server_away(&self) -> bool {
        away(self.kind())
    }
}

/// Runs `attempt` until the Program Server answers, waiting through a restart:
/// an error that says the server was away is retried with a fresh attempt
/// (the caller's attempt opens its own connection, because the restart ended
/// the previous one), and everything else is returned at once.
///
/// The wait makes at most [`Wait::attempts`] attempts, `wait.delay` apart. An
/// attempt in flight when the server goes away returns quickly with an I/O
/// error, so the wait does not stretch by the attempt's own duration. It
/// blocks between attempts: call it on a blocking thread, as the client's own
/// calls block on the socket.
///
/// # Errors
///
/// Returns the last error when the wait is spent, or the first error that is
/// not a restart: the server's own refusal or failure.
pub fn through_restart<T, E>(wait: Wait, mut attempt: impl FnMut() -> Result<T, E>) -> Result<T, E>
where
    E: ServerAway,
{
    let mut left = wait.attempts.get();
    loop {
        match attempt() {
            Ok(answer) => return Ok(answer),
            Err(error) if error.server_away() && left > 1 => {
                left -= 1;
                if !wait.delay.is_zero() {
                    std::thread::sleep(wait.delay);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io;

    /// A Program Server that is away for `away_calls` attempts, then answers,
    /// or keeps returning `error`.
    struct Stub {
        away_calls: u32,
        calls: Cell<u32>,
        error: Option<io::Error>,
    }

    impl Stub {
        fn restarting(away_calls: u32) -> Self {
            Self {
                away_calls,
                calls: Cell::new(0),
                error: None,
            }
        }

        fn failing(error: io::Error) -> Self {
            Self {
                away_calls: 0,
                calls: Cell::new(0),
                error: Some(error),
            }
        }

        fn attempt(&self) -> io::Result<u32> {
            self.calls.set(self.calls.get() + 1);
            if let Some(error) = &self.error {
                return Err(io::Error::new(error.kind(), error.to_string()));
            }
            if self.calls.get() <= self.away_calls {
                return Err(io::Error::from(io::ErrorKind::ConnectionRefused));
            }
            Ok(42)
        }
    }

    #[test]
    fn a_restart_is_waited_through_and_the_operation_completes_after_it() {
        let stub = Stub::restarting(2);
        let answer = through_restart(Wait::new(5, Duration::ZERO), || stub.attempt());
        assert_eq!(answer.ok(), Some(42));
        assert_eq!(stub.calls.get(), 3, "two restarts, then the answer");
    }

    #[test]
    fn a_server_that_does_not_come_back_fails_after_the_wait() {
        let stub = Stub::restarting(u32::MAX);
        let answer = through_restart(Wait::new(3, Duration::ZERO), || stub.attempt());
        assert!(answer.is_err());
        assert_eq!(stub.calls.get(), 3, "the wait is exactly what it says");
    }

    #[test]
    fn what_the_server_decided_is_returned_at_once() {
        let stub = Stub::failing(io::Error::new(
            io::ErrorKind::InvalidInput,
            "building a module: the source does not compile",
        ));
        let answer = through_restart(Wait::new(5, Duration::ZERO), || stub.attempt());
        assert!(answer.is_err());
        assert_eq!(stub.calls.get(), 1, "a refusal is not a restart");
    }

    #[test]
    fn every_shape_of_a_restart_is_waited_through() {
        for kind in [
            io::ErrorKind::NotFound,
            io::ErrorKind::ConnectionRefused,
            io::ErrorKind::ConnectionReset,
            io::ErrorKind::BrokenPipe,
            io::ErrorKind::UnexpectedEof,
            io::ErrorKind::NotConnected,
        ] {
            let stub = Stub::failing(io::Error::from(kind));
            let answer = through_restart(Wait::new(2, Duration::ZERO), || stub.attempt());
            assert!(answer.is_err());
            assert_eq!(stub.calls.get(), 2, "{kind:?} was not retried");
        }
        for kind in [
            io::ErrorKind::InvalidInput,
            io::ErrorKind::TimedOut,
            io::ErrorKind::PermissionDenied,
        ] {
            let stub = Stub::failing(io::Error::from(kind));
            let answer = through_restart(Wait::new(2, Duration::ZERO), || stub.attempt());
            assert!(answer.is_err());
            assert_eq!(stub.calls.get(), 1, "{kind:?} is not a restart");
        }
    }

    #[test]
    fn a_wait_of_no_attempts_still_makes_one() {
        assert_eq!(Wait::new(0, Duration::ZERO).attempts.get(), 1);
    }

    #[test]
    fn the_default_wait_spans_about_thirty_seconds() {
        let wait = Wait::default();
        assert_eq!(wait.total(), Duration::from_secs(30));
        assert!(
            !wait.delay.is_zero(),
            "the wait yields the machine between attempts"
        );
    }

    #[test]
    fn a_resolve_is_waited_through_like_a_prepare() {
        // The wait is over the attempt, not over one operation: a `resolve`
        // that meets a restart is retried the same way, and its answer's type
        // is whatever the caller's attempt produces.
        let stub = Stub::restarting(1);
        let module = through_restart(Wait::new(3, Duration::ZERO), || {
            stub.attempt().map(|answer| format!("module {answer}"))
        });
        assert_eq!(module.ok().as_deref(), Some("module 42"));
        assert_eq!(stub.calls.get(), 2);
    }
}
