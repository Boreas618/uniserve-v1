//! Performing a job on the builder thread and posting its completion, behind
//! which the control thread's wake fires ([`Post`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use cuda::{DeviceGuard, DeviceId, Event, Stream};
use ipc::Wake;

use super::operation::{Completion, Failure, Job};
use crate::{Error, Result, lock};

/// Names a job from its submission to its completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct JobId(pub(super) u64);

thread_local! {
    /// The notification streams of the thread that performs jobs.
    static NOTIFICATIONS: RefCell<Notifications> = RefCell::default();
}

/// Perform `job` on the calling thread, the builder thread, and post its
/// completion through `post`: at once, or, for a job that names events
/// ([`Job::after`]), from the driver once they completed. The job and
/// everything it holds is gone before its submitter learns that it completed.
///
/// The driver calls in the job that make the thread wait for the device are
/// counted under `builder` (`crate::profiling::blocking`), not failed: the host
/// reads of predicates still wait for their producers. A serving call inside
/// the job fails on such a call (`crate::ffi::MethodCall::serving`); this count
/// includes it.
pub(super) fn perform(job: Job, post: Post) {
    let mut performing = Performing {
        job: Some(post.id),
        completions: post.completions.clone(),
    };
    let watched = cuda::cupti::blocking::Watched::enter();
    let (completion, after) = job.perform_operations();
    performing.job = None;
    crate::profiling::blocking::add("builder", &watched.calls());
    drop(watched);
    match after {
        Some((device, events)) if completion.failure.is_none() => {
            NOTIFICATIONS.with_borrow_mut(|notifications| {
                post.after(completion, device, &events, notifications);
            });
        }
        _ => post.now(completion),
    }
}

/// Posts the failure of the job the builder thread is performing if the thread
/// unwinds, so whoever waits for that job learns that it is gone.
struct Performing {
    job: Option<JobId>,
    completions: Sender<(JobId, Completion)>,
}

impl Drop for Performing {
    fn drop(&mut self) {
        if let Some(job) = self.job {
            let completion = Completion {
                failure: Some(Failure {
                    operation: 0,
                    error: Error::State("the builder thread exited inside a job"),
                }),
                ..Completion::default()
            };
            let _ = self.completions.send((job, completion));
        }
    }
}

/// The notification streams of the thread that performs jobs: one stream per
/// device that carries nothing but the notices of completed events, created
/// when the thread first orders a notice on the device. A notice waits for its
/// events there and never holds up the streams the jobs issue work on.
#[derive(Default)]
struct Notifications {
    streams: HashMap<DeviceId, Stream>,
}

impl Notifications {
    fn stream(&mut self, device: DeviceId) -> Result<&Stream> {
        match self.streams.entry(device) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let _device = DeviceGuard::new(device)?;
                Ok(entry.insert(Stream::new(device)?))
            }
        }
    }
}

/// Posts one job's completion and fires the control thread's wake.
pub(super) struct Post {
    id: JobId,
    completions: Sender<(JobId, Completion)>,
    wake: Option<Wake>,
}

impl Post {
    pub(super) fn new(
        id: JobId,
        completions: Sender<(JobId, Completion)>,
        wake: Option<Wake>,
    ) -> Self {
        Self {
            id,
            completions,
            wake,
        }
    }

    pub(super) fn now(self, completion: Completion) {
        // Without a receiver the rank is closing and nobody takes it.
        let _ = self.completions.send((self.id, completion));
        if let Some(wake) = &self.wake {
            wake.wake();
        }
    }

    /// Post `completion` from the driver once `events` completed: a host
    /// function on `device`'s notification stream, ordered behind them. It only
    /// sends and signals: it never calls CUDA or blocks, as host functions must
    /// not.
    fn after(
        self,
        completion: Completion,
        device: DeviceId,
        events: &[Arc<Event>],
        notifications: &mut Notifications,
    ) {
        // The completion stays reachable if the notice cannot be issued.
        let held = Arc::new(Mutex::new(Some(completion)));
        let take = |held: &Mutex<Option<Completion>>| lock(held).take();
        let issued = (|| -> Result<()> {
            let notification = notifications.stream(device)?;
            for event in events {
                notification.wait_event(event)?;
            }
            let (id, completions, wake) = (self.id, self.completions.clone(), self.wake.clone());
            let posted = Arc::clone(&held);
            notification.launch_host_fn(move || {
                if let Some(completion) = take(&posted) {
                    let _ = completions.send((id, completion));
                    if let Some(wake) = &wake {
                        wake.wake();
                    }
                }
            })?;
            Ok(())
        })();

        if let Err(error) = issued
            && let Some(mut completion) = take(&held)
        {
            completion.failure = Some(Failure {
                operation: 0,
                error,
            });
            self.now(completion);
        }
    }
}
