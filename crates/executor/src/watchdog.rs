//! The main process's job watchdog (`specs/worker.md` §2.4).
//!
//! The control thread serves the channels and the commands and never waits for
//! a job. Each wait for a channel message ends, at the latest, when the executor
//! job in flight longest reaches the rank's stall bound: a job the executor
//! thread has stayed inside that long, such as a driver call that does not
//! return, is reported (`ipc::Stall`) with the batch it belongs to, and the
//! channel goes on being served from the report's failure on.
//!
//! A context an illegal memory access or a device-side assert left unusable
//! runs none of the work behind the failure, so the completions of the jobs in
//! flight never post. While a job has run for a second or more, the control
//! thread probes every device's context once a second; a context that reports a
//! sticky error fails the rank at once, with the batch it awaited, rather than
//! at the stall bound.
//!
//! Both are failures of the epoch: `§2.5` fails the rank's worker group with
//! them, and `§7.3` step 4 reverts an update on probation when one occurs. The
//! watchdog itself calls no driver function and owns no context: it decides
//! *when* the control thread looks and *which* job it reports, so that the rules
//! are testable without a device. The probe itself is the control thread's,
//! through [`crate::control`].

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ipc::Stall;

use crate::{Error, Result};

/// How long a job stays in flight before the control thread probes every
/// device's context, and how often it probes them while one does (§2.4).
pub const CONTEXT_PROBE: Duration = Duration::from_secs(1);

/// Half the configured watchdog timeout: a job that has run this long is
/// reported to the scheduler with its batch (§2.4).
pub fn stall_bound(watchdog_timeout: Duration) -> Duration {
    watchdog_timeout / 2
}

/// One executor's oldest unfinished job, as the control thread reads it on
/// each pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Running {
    /// The device executor's index, the rank the stall names.
    pub rank: usize,
    /// The batch the job belongs to: the one whose forward, input admission or
    /// output wait it is. `None` for a job of no batch, such as a lifecycle
    /// command's work.
    pub batch: Option<u64>,
    /// When the executor thread was given the job.
    pub since: Instant,
}

impl Running {
    /// How long the job has run at `now`.
    pub fn elapsed(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.since)
    }
}

/// The §2.4 watchdog: the stall bound, the once-per-job report, and the device
/// context probe schedule.
#[derive(Debug)]
pub struct Watchdog {
    /// Half `worker_config.watchdog_timeout`.
    stall_bound: Duration,
    /// When the control thread last probed the devices' contexts.
    probed_at: Option<Instant>,
    /// The jobs already reported, by rank, with their start: a job is reported
    /// once, and a new job of the same rank is a new report.
    reported: BTreeMap<usize, Instant>,
}

impl Watchdog {
    /// A watchdog for an executor whose jobs past `watchdog_timeout / 2` are
    /// reported.
    pub fn new(watchdog_timeout: Duration) -> Result<Self> {
        let stall_bound = stall_bound(watchdog_timeout);
        if stall_bound.is_zero() {
            return Err(Error::Invalid(
                "the watchdog timeout must leave a positive stall bound".into(),
            ));
        }
        Ok(Self {
            stall_bound,
            probed_at: None,
            reported: BTreeMap::new(),
        })
    }

    /// The bound a job may run before it is reported.
    pub fn stall_bound(&self) -> Duration {
        self.stall_bound
    }

    /// The instant the control thread must look again: the sooner of the oldest
    /// job reaching the stall bound and the next device context probe. `None`
    /// while no job runs, when the control thread may wait for its channels
    /// indefinitely.
    pub fn deadline(&self, oldest: Option<Running>) -> Option<Instant> {
        let oldest = oldest?;
        let stall_at = oldest.since + self.stall_bound;
        // The probe only starts once the job itself has run for a second, so a
        // job that has just started keeps the deadline at its stall bound.
        let probe_at = oldest
            .since
            .checked_add(CONTEXT_PROBE)
            .zip(self.probed_at.map(|at| at + CONTEXT_PROBE))
            .map(|(job_ready, probe_ready)| job_ready.max(probe_ready));
        Some(match probe_at {
            Some(probe_at) => stall_at.min(probe_at),
            None => stall_at,
        })
    }

    /// The jobs that have run past the stall bound and were not reported yet,
    /// each with the stall the control thread sends for it. Each job is
    /// reported once: the control thread reports it with its batch and goes on
    /// serving the channel.
    pub fn stalls(&mut self, now: Instant, running: &[Running]) -> Vec<(usize, Stall)> {
        let mut stalls = Vec::new();
        for job in running {
            if job.elapsed(now) < self.stall_bound {
                continue;
            }
            if self.reported.get(&job.rank) == Some(&job.since) {
                continue;
            }
            self.reported.insert(job.rank, job.since);
            stalls.push((
                job.rank,
                Stall {
                    batch_id: job.batch,
                    // The executor's job belongs to this step's batch; the
                    // method and the batch's calls are the executor's to state.
                    route: None,
                    method: None,
                    calls: Vec::new(),
                    running_ms: u64::try_from(job.elapsed(now).as_millis()).unwrap_or(u64::MAX),
                },
            ));
        }
        stalls
    }

    /// Whether this pass must probe the devices' contexts: while a job has run
    /// for a second or more, once a second.
    pub fn probe_due(&self, now: Instant, oldest: Option<Running>) -> bool {
        let Some(oldest) = oldest else {
            return false;
        };
        if oldest.elapsed(now) < CONTEXT_PROBE {
            return false;
        }
        match self.probed_at {
            None => true,
            Some(at) => now.saturating_duration_since(at) >= CONTEXT_PROBE,
        }
    }

    /// Records that the control thread probed the devices' contexts at `now`,
    /// whether or not one reported an error.
    pub fn probed(&mut self, now: Instant) {
        self.probed_at = Some(now);
    }

    /// Forgets the reports of ranks that no longer run a job, so that the map
    /// does not grow with every job a long-serving process performs.
    pub fn settled(&mut self, running: &[Running]) {
        self.reported
            .retain(|rank, since| running.iter().any(|job| job.rank == *rank && job.since == *since));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn watchdog() -> Watchdog {
        Watchdog::new(Duration::from_secs(30)).expect("a positive watchdog timeout")
    }

    fn never_shadowed(rank: usize, batch: u64, since: Instant) -> Running {
        Running {
            rank,
            batch: Some(batch),
            since,
        }
    }

    #[test]
    fn the_stall_bound_is_half_the_watchdog_timeout() {
        assert_eq!(stall_bound(Duration::from_secs(30)), Duration::from_secs(15));
        assert_eq!(stall_bound(Duration::from_secs(1)), Duration::from_millis(500));
    }

    #[test]
    fn a_zero_watchdog_timeout_is_refused() {
        assert!(Watchdog::new(Duration::ZERO).is_err());
        assert!(Watchdog::new(Duration::from_nanos(1)).is_err());
    }

    #[test]
    fn a_job_under_the_bound_is_not_reported() {
        let mut watchdog = watchdog();
        let now = Instant::now();
        let job = never_shadowed(0, 7, now - Duration::from_secs(14));
        assert!(watchdog.stalls(now, &[job]).is_empty());
    }

    #[test]
    fn a_job_past_the_bound_is_reported_once_with_its_batch() {
        let mut watchdog = watchdog();
        let now = Instant::now();
        let job = never_shadowed(2, 7, now - Duration::from_secs(16));
        let stalls = watchdog.stalls(now, &[job]);
        assert_eq!(stalls.len(), 1);
        assert_eq!(stalls[0].0, 2);
        assert_eq!(stalls[0].1.batch_id, Some(7));
        assert_eq!(stalls[0].1.running_ms, 16_000);
        assert!(watchdog.stalls(now, &[job]).is_empty());
    }

    #[test]
    fn a_new_job_of_a_reported_rank_is_reported_again() {
        let mut watchdog = watchdog();
        let now = Instant::now();
        let first = never_shadowed(0, 1, now - Duration::from_secs(16));
        assert_eq!(watchdog.stalls(now, &[first]).len(), 1);
        let next = never_shadowed(0, 2, now - Duration::from_secs(16));
        assert_eq!(watchdog.stalls(now, &[next]).len(), 1);
    }

    #[test]
    fn the_deadline_is_the_stall_bound_while_no_probe_ran() {
        let watchdog = watchdog();
        let now = Instant::now();
        let job = never_shadowed(0, 1, now);
        assert_eq!(watchdog.deadline(Some(job)), Some(now + Duration::from_secs(15)));
    }

    #[test]
    fn the_deadline_is_the_sooner_of_the_bound_and_the_probe() {
        let mut watchdog = watchdog();
        let now = Instant::now();
        let job = never_shadowed(0, 1, now - Duration::from_secs(10));
        watchdog.probed(now - Duration::from_millis(200));
        // The probe is 800 ms away; the stall bound is 5 s away.
        assert_eq!(
            watchdog.deadline(Some(job)),
            Some(now + Duration::from_millis(800))
        );
        watchdog.probed(now);
        // A fresh probe leaves the stall bound.
        assert_eq!(watchdog.deadline(Some(job)), Some(job.since + Duration::from_secs(15)));
    }

    #[test]
    fn nothing_runs_so_there_is_no_deadline() {
        assert_eq!(watchdog().deadline(None), None);
    }

    #[test]
    fn the_context_probe_waits_for_a_second_of_a_running_job() {
        let watchdog = watchdog();
        let now = Instant::now();
        let job = never_shadowed(0, 1, now - Duration::from_millis(999));
        assert!(!watchdog.probe_due(now, Some(job)));
        let job = never_shadowed(0, 1, now - Duration::from_secs(1));
        assert!(watchdog.probe_due(now, Some(job)));
        assert!(!watchdog.probe_due(now, None));
    }

    #[test]
    fn the_context_probe_runs_once_a_second() {
        let mut watchdog = watchdog();
        let now = Instant::now();
        let job = never_shadowed(0, 1, now - Duration::from_secs(5));
        watchdog.probed(now);
        assert!(!watchdog.probe_due(now + Duration::from_millis(999), Some(job)));
        assert!(watchdog.probe_due(now + Duration::from_secs(1), Some(job)));
    }

    #[test]
    fn settling_drops_the_reports_of_jobs_that_ended() {
        let mut watchdog = watchdog();
        let now = Instant::now();
        let job = never_shadowed(0, 1, now - Duration::from_secs(16));
        assert_eq!(watchdog.stalls(now, &[job]).len(), 1);
        watchdog.settled(&[]);
        assert_eq!(watchdog.stalls(now, &[job]).len(), 1);
    }
}
