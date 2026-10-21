//! A lane's job queue: the submitting end the control thread holds
//! ([`JobQueue`]), the execution thread's end that performs each job in order
//! ([`Jobs`]) with the notification streams its completions wait on, and the
//! posting of a job's completion, behind which the control thread's wake fires
//! ([`Post`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender};
use cuda::{DeviceGuard, DeviceId, Event, Stream};
use ipc::Wake;

use super::job::{Completion, Failure, Job};
use crate::{Error, Result, lock};

/// Names a job from its submission to its completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct JobId(pub(super) u64);

/// What a lane's execution thread receives.
enum Message {
    /// A job, and whether the control thread's wake is fired behind its
    /// completion.
    Job(JobId, Job, bool),
    /// The wake of the control thread, fired behind every completion from
    /// now on; `None` stops it.
    Wake(Option<Wake>),
}

/// The submitting end of a lane's job queue. Only the control thread
/// submits.
#[derive(Clone)]
pub(crate) struct JobQueue {
    messages: Sender<Message>,
}

/// The execution thread's end of a lane's job queue.
pub(super) struct Jobs {
    messages: Receiver<Message>,
    /// Where the lanes of a rank post their completions.
    completions: Sender<(JobId, Completion)>,
}

/// Posts the failure of the job a lane is performing if the lane's thread
/// unwinds, so whoever waits for that job learns that the lane is gone.
struct Performing<'a> {
    job: Option<JobId>,
    completions: &'a Sender<(JobId, Completion)>,
}

impl Drop for Performing<'_> {
    fn drop(&mut self) {
        if let Some(job) = self.job {
            let completion = Completion {
                failure: Some(Failure {
                    operation: 0,
                    error: Error::State("the lane's execution thread has exited"),
                }),
                ..Completion::default()
            };
            let _ = self.completions.send((job, completion));
        }
    }
}

impl JobQueue {
    pub(super) fn new(completions: Sender<(JobId, Completion)>) -> (Self, Jobs) {
        let (messages, received) = crossbeam_channel::unbounded();
        (
            Self { messages },
            Jobs {
                messages: received,
                completions,
            },
        )
    }

    /// Queue `job` as `id` on the lane's execution thread, which posts its
    /// completion under that name and, with `wake`, fires the control
    /// thread's wake behind it.
    pub(super) fn start(&self, id: JobId, job: Job, wake: bool) -> Result<()> {
        self.send(Message::Job(id, job, wake))
    }

    pub(super) fn set_wake(&self, wake: Option<Wake>) -> Result<()> {
        self.send(Message::Wake(wake))
    }

    fn send(&self, message: Message) -> Result<()> {
        self.messages
            .send(message)
            .map_err(|_| Error::State("the lane's execution thread has exited"))
    }
}

impl Jobs {
    /// Perform jobs in arrival order on the calling thread, the lane's
    /// execution thread, until every submitting end is gone. Each
    /// completion is posted under its job's name, and the control thread's
    /// wake fired behind it; a job that names events ([`Job::after`]) has
    /// its completion posted by the driver once they completed.
    pub(super) fn run(self) {
        let mut wake = None;
        let mut notifications = Notifications::default();
        while let Ok(message) = self.messages.recv() {
            let (id, job, notify) = match message {
                Message::Job(id, job, notify) => (id, job, notify),
                Message::Wake(installed) => {
                    wake = installed;
                    continue;
                }
            };
            let mut performing = Performing {
                job: Some(id),
                completions: &self.completions,
            };
            // The job and everything it holds is gone before its submitter
            // learns that it completed.
            let (completion, after) = job.perform_operations();
            performing.job = None;

            let post = Post {
                id,
                completions: self.completions.clone(),
                wake: wake.clone().filter(|_| notify),
            };
            match after {
                Some((device, events)) if completion.failure.is_none() => {
                    post.after(completion, device, &events, &mut notifications);
                }
                _ => post.now(completion),
            }
        }
    }
}

/// The notification streams of a lane: one stream per device that carries
/// nothing but the notices of completed events, created by the lane's
/// execution thread when it first orders a notice on the device. A notice
/// waits for its events there and never holds up the lane's own stream.
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
    /// function on `device`'s notification stream, ordered behind them. It
    /// only sends and signals: it never calls CUDA or blocks, as host
    /// functions must not.
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
