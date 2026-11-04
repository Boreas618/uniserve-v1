//! The main process's blocking thread pool (`specs/worker.md` §2.4).
//!
//! The main process does four kinds of blocking host work: loading modules,
//! reading transfers, reclaiming shared storage, and encoding, muxing and
//! reading media. None of them runs on a scheduler thread or on an executor
//! thread: those wait only for their own device work and their own queue
//! (§2.4), and a media encode that takes a second must not hold up the module
//! load of a program switch.
//!
//! Each kind therefore gets its own lane: a bounded pool ([`HostPool`]) with
//! its own queue, its own worker threads and its own admission bound. A lane is
//! created by the owner of its work, which names its action type; the
//! [`ThreadPool`] keeps the lanes so that the process can abort and close all
//! of them together, and owns nothing of any lane's action.
//!
//! Aborting stops admission without joining work that may wait on a failed
//! device or peer; closing lets the process exit once the lanes' threads ended.
//! A lane's inputs are retired by its pool, by the rules of
//! [`HostAction::input_outcome`]: a failed or cancelled physical producer keeps
//! its action until process exit, because a device work whose completion is
//! unknown may still address it.

use std::sync::{Arc, Mutex};

use crate::host::{HostAction, HostPool, Lane};
use crate::{Error, Result, lock};

/// The four kinds of blocking host work of §2.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockingKind {
    /// Resolving a program's modules at the Program Server and loading them
    /// into kernels.
    ModuleLoad,
    /// Reading a transfer the Storage Server sent.
    TransferRead,
    /// Reclaiming shared storage whose last device access completed.
    Reclaim,
    /// Encoding, muxing and reading media.
    Media,
}

impl BlockingKind {
    /// Every kind, in the order the process creates its lanes.
    pub const ALL: [Self; 4] = [
        Self::ModuleLoad,
        Self::TransferRead,
        Self::Reclaim,
        Self::Media,
    ];

    /// The lane's thread-name prefix and log name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModuleLoad => "module-load",
            Self::TransferRead => "transfer-read",
            Self::Reclaim => "reclaim",
            Self::Media => "media",
        }
    }
}

/// How many tasks a lane admits at once, and how many threads it runs them on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneLimits {
    /// Admitted tasks, running and queued together.
    pub capacity: usize,
    /// Worker threads, which must lie in `1..=capacity`.
    pub workers: usize,
}

/// The lanes' admission bounds, one per kind, from the build config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    lanes: [LaneLimits; 4],
}

impl Limits {
    /// The bounds of the four lanes, in [`BlockingKind::ALL`] order.
    pub fn new(lanes: [LaneLimits; 4]) -> Self {
        Self { lanes }
    }

    /// One lane's bounds.
    pub fn of(&self, kind: BlockingKind) -> LaneLimits {
        self.lanes[BlockingKind::ALL
            .iter()
            .position(|candidate| *candidate == kind)
            .unwrap_or_default()]
    }
}

/// What the pool keeps of a lane, with its action type erased.
trait LaneHandle: Send + Sync {
    fn abort(&self);
    fn close(&self) -> Vec<Error>;
}

impl<A: HostAction> LaneHandle for HostPool<A> {
    fn abort(&self) {
        HostPool::abort(self);
    }

    fn close(&self) -> Vec<Error> {
        HostPool::close(self)
    }
}

/// The main process's blocking pool: its lanes, their bounds, and their
/// orderly end.
pub struct ThreadPool {
    limits: Limits,
    lanes: Mutex<Vec<(BlockingKind, Arc<dyn LaneHandle>)>>,
}

impl ThreadPool {
    /// A pool with no lane yet; each owner creates its lane with [`Self::lane`].
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            lanes: Mutex::new(Vec::new()),
        }
    }

    /// Creates `kind`'s lane and returns it, ready to admit work.
    ///
    /// The lane is the caller's whole surface: `admit` queues an action and
    /// hands back the task whose outcome it waits on. One lane per kind: the
    /// second creation of a kind is refused.
    pub fn lane<A: HostAction>(&self, kind: BlockingKind) -> Result<Lane<A>> {
        let limits = self.limits.of(kind);
        let pool = Arc::new(HostPool::new(
            limits.capacity,
            limits.workers,
            &format!("blocking-{}", kind.as_str()),
        )?);
        {
            let mut lanes = lock(&self.lanes);
            if lanes.iter().any(|(existing, _)| *existing == kind) {
                return Err(Error::Invalid(format!(
                    "the {} lane was created twice",
                    kind.as_str()
                )));
            }
            let erased: Arc<dyn LaneHandle> = Arc::clone(&pool) as Arc<dyn LaneHandle>;
            lanes.push((kind, erased));
        }
        Ok(Lane::new(pool))
    }

    /// The kinds whose lane exists.
    pub fn kinds(&self) -> Vec<BlockingKind> {
        lock(&self.lanes).iter().map(|(kind, _)| *kind).collect()
    }

    /// Stops admission on every lane without joining work that may wait on a
    /// failed device or peer.
    pub fn abort(&self) {
        for (_, lane) in lock(&self.lanes).iter() {
            lane.abort();
        }
    }

    /// Closes every lane and returns the errors of those whose workers did not
    /// end cleanly.
    pub fn close(&self) -> Vec<Error> {
        let lanes = std::mem::take(&mut *lock(&self.lanes));
        let mut errors = Vec::new();
        for (_, lane) in lanes {
            errors.extend(lane.close());
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::completion::Outcome;

    /// A lane's task that counts its runs and holds an input.
    struct Probe {
        runs: Arc<AtomicUsize>,
        input: Arc<()>,
    }

    impl HostAction for Probe {
        type Output = ();

        fn run(&self) -> Result<()> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn release(&self) -> Result<()> {
            Ok(())
        }

        fn input_outcome(&self) -> Outcome {
            Outcome::Success(())
        }
    }

    fn limits() -> Limits {
        let lane = LaneLimits {
            capacity: 4,
            workers: 1,
        };
        Limits::new([lane; 4])
    }

    fn run_one(lane: &Lane<Probe>, runs: &Arc<AtomicUsize>) -> Result<()> {
        lane.admit(Probe {
            runs: Arc::clone(runs),
            input: Arc::new(()),
        })?;
        let deadline = Instant::now() + Duration::from_secs(10);
        while runs.load(Ordering::SeqCst) == 0 {
            assert!(Instant::now() < deadline, "the lane did not run its task");
            std::thread::yield_now();
        }
        Ok(())
    }

    #[test]
    fn a_lane_runs_the_work_its_owner_admits() -> Result<()> {
        let threads = ThreadPool::new(limits());
        let lane = threads.lane::<Probe>(BlockingKind::ModuleLoad)?;
        assert_eq!(threads.kinds(), vec![BlockingKind::ModuleLoad]);
        let runs = Arc::new(AtomicUsize::new(0));
        run_one(&lane, &runs)?;
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(threads.close().is_empty());
        Ok(())
    }

    #[test]
    fn a_kind_has_one_lane() -> Result<()> {
        let threads = ThreadPool::new(limits());
        let _lane = threads.lane::<Probe>(BlockingKind::Media)?;
        assert!(threads.lane::<Probe>(BlockingKind::Media).is_err());
        Ok(())
    }

    #[test]
    fn each_kind_gets_its_own_lane_and_limits() -> Result<()> {
        let threads = ThreadPool::new(limits());
        let _modules = threads.lane::<Probe>(BlockingKind::ModuleLoad)?;
        let _media = threads.lane::<Probe>(BlockingKind::Media)?;
        assert_eq!(
            threads.kinds(),
            vec![BlockingKind::ModuleLoad, BlockingKind::Media]
        );
        Ok(())
    }

    #[test]
    fn an_aborted_lane_admits_nothing_new() -> Result<()> {
        let threads = ThreadPool::new(limits());
        let lane = threads.lane::<Probe>(BlockingKind::Reclaim)?;
        threads.abort();
        assert!(
            lane.admit(Probe {
                runs: Arc::new(AtomicUsize::new(0)),
                input: Arc::new(()),
            })
            .is_err()
        );
        assert!(threads.close().is_empty());
        Ok(())
    }

    #[test]
    fn a_lane_bound_the_pool_refuses_is_an_error() {
        let lane = LaneLimits {
            capacity: 1,
            workers: 2,
        };
        let threads = ThreadPool::new(Limits::new([lane; 4]));
        assert!(threads.lane::<Probe>(BlockingKind::ModuleLoad).is_err());
    }
}
