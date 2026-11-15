//! The engine's job watchdog.
//!
//! The control thread serves its channels and never waits for a job. Each of its waits ends, at the
//! latest, when the job in flight longest reaches the stall bound (`worker_config.watchdog_timeout`):
//! a job a worker has stayed inside that long, such as a driver call that does not return or a
//! collective whose peer never arrives, is reported (`crate::executor::peers::messages::Stall`)
//! with the batch it belongs to, and the channels go on being served. A stall fails the epoch: the
//! rank's group is replaced, and an update on trial is rolled back.
//!
//! A context an illegal memory access or a device-side assert left unusable is not the
//! watchdog's: the worker of that device finds the sticky error at its own wait and states it as
//! its device's fault ([`Worker::on_fault`](super::Worker::on_fault)), which fails the epoch at
//! once.
//!
//! The watchdog itself calls no driver function and reads no worker: it is given what each worker
//! has in flight ([`Worker::in_flight`](super::Worker::in_flight)), one entry per rank of the
//! process, and decides when the control thread looks and which job it reports.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::batch::BatchId;
use crate::executor::peers::messages::Stall;

use crate::worker::{Error, Result};

/// The stall bound and the once-per-job report.
#[derive(Debug)]
pub(crate) struct Watchdog {
    /// `worker_config.watchdog_timeout`.
    stall_bound: Duration,
    /// The jobs already reported, by rank, with the instant they were taken: a job is reported
    /// once, and a later job of the same rank is another report.
    reported: BTreeMap<usize, Instant>,
}

impl Watchdog {
    /// A watchdog that reports a job in flight for `watchdog_timeout`.
    ///
    /// # Errors
    ///
    /// When the timeout is zero, which would report every job as it starts.
    pub(crate) fn new(watchdog_timeout: Duration) -> Result<Self> {
        if watchdog_timeout.is_zero() {
            return Err(Error::Invalid(
                "the watchdog timeout must be positive".into(),
            ));
        }

        Ok(Self {
            stall_bound: watchdog_timeout,
            reported: BTreeMap::new(),
        })
    }

    /// The instant the control thread must look again: when the job in flight longest reaches
    /// the stall bound. `None` while no job is in flight, when the control thread may wait for
    /// its channels indefinitely.
    ///
    /// `in_flight` holds, per rank of the process, when its worker took the job in flight and
    /// the job's batch.
    pub(crate) fn deadline(
        &self,
        in_flight: &[Option<(Instant, Option<BatchId>)>],
    ) -> Option<Instant> {
        Some(longest(in_flight)? + self.stall_bound)
    }

    /// The jobs in flight past the stall bound at `now` that were not reported yet, each as its
    /// rank and the stall the control thread sends for it.
    pub(crate) fn stalls(
        &mut self,
        now: Instant,
        in_flight: &[Option<(Instant, Option<BatchId>)>],
    ) -> Vec<(usize, Stall)> {
        let mut stalls = Vec::new();

        for (rank, job) in in_flight.iter().enumerate() {
            let Some((since, batch)) = *job else {
                continue;
            };
            let elapsed = now.saturating_duration_since(since);
            if elapsed < self.stall_bound || self.reported.get(&rank) == Some(&since) {
                continue;
            }

            self.reported.insert(rank, since);
            stalls.push((
                rank,
                Stall {
                    batch_id: batch.map(|batch| batch.0),
                    // The job belongs to this batch; its call kind and the runner method are the
                    // scheduler's to add from the batch it submitted.
                    route: None,
                    method: None,
                    running_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                },
            ));
        }
        stalls
    }

    /// Forgets the reports of the jobs that ended, so the reports do not grow
    /// with every job a long-serving process performs.
    pub(crate) fn settled(&mut self, in_flight: &[Option<(Instant, Option<BatchId>)>]) {
        self.reported.retain(|rank, since| {
            in_flight
                .get(*rank)
                .is_some_and(|job| job.is_some_and(|(taken, _)| taken == *since))
        });
    }
}

/// When the job in flight longest was taken.
fn longest(in_flight: &[Option<(Instant, Option<BatchId>)>]) -> Option<Instant> {
    in_flight.iter().flatten().map(|(since, _)| *since).min()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn watchdog() -> Watchdog {
        Watchdog::new(Duration::from_secs(15)).expect("a positive watchdog timeout")
    }

    /// A job of batch `batch` taken at `since`.
    fn job(batch: u64, since: Instant) -> Option<(Instant, Option<BatchId>)> {
        Some((since, Some(BatchId(batch))))
    }

    #[test]
    fn a_zero_watchdog_timeout_is_refused() {
        assert!(Watchdog::new(Duration::ZERO).is_err());
        assert!(Watchdog::new(Duration::from_nanos(1)).is_ok());
    }

    #[test]
    fn a_job_under_the_bound_is_not_reported() {
        let mut watchdog = watchdog();
        let now = Instant::now();

        let in_flight = [job(7, now - Duration::from_secs(14))];
        assert!(watchdog.stalls(now, &in_flight).is_empty());
    }

    #[test]
    fn a_job_past_the_bound_is_reported_once_with_its_rank_and_batch() {
        let mut watchdog = watchdog();
        let now = Instant::now();

        // Rank 1 is between two jobs; rank 2 has been inside one for sixteen seconds.
        let in_flight = [
            job(3, now - Duration::from_secs(2)),
            None,
            job(7, now - Duration::from_secs(16)),
        ];
        let stalls = watchdog.stalls(now, &in_flight);

        assert_eq!(stalls.len(), 1);
        assert_eq!(stalls[0].0, 2);
        assert_eq!(stalls[0].1.batch_id, Some(7));
        assert_eq!(stalls[0].1.running_ms, 16_000);
        assert!(watchdog.stalls(now, &in_flight).is_empty());
    }

    #[test]
    fn a_job_of_no_batch_is_reported_without_one() {
        let mut watchdog = watchdog();
        let now = Instant::now();

        let stalls = watchdog.stalls(now, &[Some((now - Duration::from_secs(20), None))]);
        assert_eq!(stalls.len(), 1);
        assert_eq!(stalls[0].1.batch_id, None);
    }

    #[test]
    fn a_later_job_of_a_reported_rank_is_reported_again() {
        let mut watchdog = watchdog();
        let now = Instant::now();

        let first = [job(1, now - Duration::from_secs(16))];
        assert_eq!(watchdog.stalls(now, &first).len(), 1);

        // A job taken at another instant is another job, and its stall another report.
        let later = [job(2, now - Duration::from_secs(15))];
        assert_eq!(watchdog.stalls(now, &later).len(), 1);
    }

    #[test]
    fn the_deadline_is_the_bound_of_the_job_in_flight_longest() {
        let watchdog = watchdog();
        let now = Instant::now();

        // Rank 2 took its job 400 ms before rank 0 did: its bound is reached first.
        let in_flight = [job(1, now), None, job(2, now - Duration::from_millis(400))];
        assert_eq!(
            watchdog.deadline(&in_flight),
            Some(now + Duration::from_millis(14_600))
        );
    }

    #[test]
    fn nothing_is_in_flight_so_there_is_no_deadline() {
        assert_eq!(watchdog().deadline(&[None, None]), None);
        assert_eq!(watchdog().deadline(&[]), None);
    }

    #[test]
    fn settling_drops_the_reports_of_jobs_that_ended() {
        let mut watchdog = watchdog();
        let now = Instant::now();
        let in_flight = [job(1, now - Duration::from_secs(16))];

        assert_eq!(watchdog.stalls(now, &in_flight).len(), 1);

        // While the job is in flight its report is kept, and it is not reported again.
        watchdog.settled(&in_flight);
        assert!(watchdog.stalls(now, &in_flight).is_empty());

        watchdog.settled(&[None]);
        assert_eq!(watchdog.stalls(now, &in_flight).len(), 1);
    }
}
