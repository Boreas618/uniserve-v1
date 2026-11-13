//! Graph breaks: host work a forward states between two segments of its launches.
//!
//! A captured graph holds kernels and copies and nothing of the host. A forward whose work
//! between two of its launches is the host's (a distributed expert layer's copy of weights from
//! its peers) states that work with [`graph_break`]. What happens to it follows from the
//! context, as for every other operation:
//!
//! - a context that launches, on a stream that is not being captured, runs the work at once,
//!   after it issued the launches before the break;
//! - while execution captures the forward, the context's break taker
//!   ([`Context::set_breaks`](crate::Context::set_breaks)) receives the work: execution ends the
//!   segment it was capturing, keeps the work, and captures the launches after the break as the
//!   next segment. A replay then launches the first segment, runs the work on the host and
//!   launches the next. No host callback is placed on the stream or in a graph;
//! - a meta context runs nothing.
//!
//! While a launch table is being recorded, a break adds the entry `(BREAK, "", module path, 0)`,
//! so a profile run states where a capture of the same forward is segmented.

use std::error::Error as StdError;
use std::sync::Arc;

use platform::CaptureStatus;

use crate::{Error, Result, context, profiler};

/// The operator name a graph break has in a launch table.
pub const BREAK: &str = "graph_break";

/// The host work of one graph break. It is called on the thread that issues the forward, once
/// per eager forward and once per replay of a graph captured from it.
pub type HostWork =
    Arc<dyn Fn() -> std::result::Result<(), Box<dyn StdError + Send + Sync>> + Send + Sync>;

/// States `work` as the host work between the launches issued so far and those that follow, by
/// the context installed on this thread.
///
/// # Errors
///
/// [`Error::NoContext`]; [`Error::Break`] when the work fails, when the break taker refuses it,
/// or when the stream is being captured and the context holds no break taker;
/// [`Error::Platform`] when the stream cannot be queried.
pub fn graph_break(work: HostWork) -> Result<()> {
    let context = context::current("a graph break")?;

    let taker = {
        let mut state = context.state();
        let path = profiler::path(&state.paths);
        if let Some(launches) = state.launches.as_mut() {
            launches.push((BREAK, String::new(), path, 0));
        }
        state.breaks.clone()
    };

    // A meta forward computes nothing, on the device or on the host.
    let Some(stream) = context.stream() else {
        return Ok(());
    };

    match taker {
        Some(taker) => taker(work),
        None if stream.capture_status()? != CaptureStatus::None => Err(Error::Break(
            "the stream is being captured and nothing takes the forward's graph breaks".into(),
        )),
        None => work().map_err(Error::Break),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{BREAK, HostWork, graph_break};
    use crate::{Context, Error};

    /// Host work that counts its calls.
    fn counted(calls: &Arc<AtomicUsize>) -> HostWork {
        let calls = Arc::clone(calls);
        Arc::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }

    #[test]
    fn a_meta_context_records_the_break_and_runs_nothing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let context = Context::meta(None);
        context.record_launches();
        let _guard = context.enter();

        graph_break(counted(&calls)).unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            context.take_launches(),
            [(BREAK, String::new(), String::new(), 0)]
        );
    }

    #[test]
    fn a_break_with_no_context_installed_names_what_is_missing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let refused = graph_break(counted(&calls)).unwrap_err();

        assert!(matches!(
            refused,
            Error::NoContext {
                operation: "a graph break"
            }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}
