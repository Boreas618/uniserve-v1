//! A profiling session on one rank: its window over the rank's steps, its
//! collectors and its file.
//!
//! The leader daemon starts a session on every worker of the world that
//! executes with the worker hook `uniserve.profile_start`, while the rank
//! runs no batch ([`Session::start`]). A rank the session does not select
//! answers so and records nothing. A selected rank prepares the collectors
//! ([`collectors`]) and creates its file ([`artifact`]); it arms the
//! collectors before the forward of the first step its window takes and
//! disarms them after the last, as jobs the step's forward waits for, so
//! that what the collectors record begins and ends at step boundaries:
//!
//! - A step is one batch's forward; its stage is its call kind.
//! - The window skips `start_step` steps of the selected stages, then takes
//!   `steps` of them, or every one until the session ends. Steps of the
//!   other stages are not recorded: the collectors are disarmed before such
//!   a step and armed again before the next selected one.
//! - Every step during the session is listed in the file (`steps`), with
//!   whether the window took it.
//!
//! A sampler thread drains the armed samplers every [`DRAIN_PERIOD`]. The
//! rank reports its progress with every batch result
//! (`ForwardStats::profile`): the steps its window took, the bytes of its
//! file, and that its window ended. The daemon ends the session with
//! `uniserve.profile_stop` ([`Session::stop`]) when every window ended, its
//! time passed, the operator ends it, its world changes, or the
//! quality-of-service guard ends it; the guard first releases collectors one
//! by one (`uniserve.profile_release`, [`Session::release`]).

pub(crate) mod artifact;
pub(crate) mod collectors;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cuda::DeviceId;
use proto::profile::{Collector, SessionStart};

use crate::job::{Completion, Job, JobId, JobQueue};
use crate::profiling::fences::{Fences, realtime};
use crate::stream::Issue;
use crate::{Error, Result, lock};
use artifact::{ClockRow, Message, Rows, StepRow, Writer};
use collectors::Collectors;

/// How often the sampler thread drains the armed samplers.
const DRAIN_PERIOD: Duration = Duration::from_millis(500);

/// Every how many recorded steps the range profiler's ranges are read.
const RANGE_CYCLE: u64 = 256;

/// A change of the collectors a step's forward waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Arm,
    Disarm,
    CycleRanges,
}

/// A session on a rank the session selected.
pub(crate) struct Session {
    id: u64,
    request: proto::profile::SessionRequest,
    writer: Option<Writer>,
    rows: Rows,
    collectors: Arc<Mutex<Collectors>>,
    /// The collectors running ([`collectors::bit`]), read at every step
    /// instead of the collectors' lock, which the sampler thread holds while
    /// it drains the samplers.
    running: Arc<AtomicU32>,
    /// Steps of the selected stages skipped before the window opened.
    skipped: u64,
    /// Steps the window took as they began, which bounds the window while
    /// earlier ones are still in flight.
    taken: u64,
    /// Steps the window took whose forward ended.
    recorded: u64,
    /// Whether the collectors are armed once the actions queued are done.
    intended: bool,
    /// Why the window ended, once it did.
    ended: Option<String>,
    pending: VecDeque<Action>,
    /// The action whose job is in flight.
    submitted: Option<Action>,
    /// The steps begun and not ended: stage, whether the window took it,
    /// when it began.
    open: HashMap<u64, (String, bool, u64)>,
    /// What the rank reported so far: steps, bytes, ended.
    reported: (u64, u64, bool),
    sampler: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
}

/// What a session's start or stop answers the daemon, as JSON.
pub(crate) type Answer = serde_json::Value;

impl Session {
    /// Start the session `start` on rank `rank`, whose CUDA device is
    /// `device` (none for a host rank): `None` with the answer when the
    /// session does not select the rank.
    ///
    /// # Errors
    ///
    /// Fails when the rank cannot create its file; collectors that cannot
    /// run are reported in the answer, and the session runs the others.
    pub(crate) fn start(
        start: SessionStart,
        rank: u32,
        device: Option<DeviceId>,
    ) -> Result<(Option<Self>, Answer)> {
        let SessionStart {
            session,
            world,
            request,
        } = start;
        request.validate().map_err(Error::Invalid)?;
        if !request.workers.is_empty() && !request.workers.contains(&rank) {
            return Ok((None, serde_json::json!({ "rank": rank, "selected": false })));
        }
        std::fs::create_dir_all(&request.output_dir).map_err(|error| {
            Error::transport_source(format!("creating {}", request.output_dir.display()), error)
        })?;
        let path = request
            .output_dir
            .join(format!("profile-{session}-world{world}-rank{rank}.sqlite"));
        let writer = Writer::create(
            path.clone(),
            metadata(session, world, rank, device, &request),
        )?;
        let rows = writer.rows();
        clocks(&rows, "start");
        if let Some(fences) = Fences::installed() {
            rows.send(Message::FenceProbes(
                fences
                    .instances()
                    .iter()
                    .enumerate()
                    .map(|(index, instance)| {
                        (
                            index as u32,
                            instance.path.clone(),
                            instance.component.clone(),
                        )
                    })
                    .collect(),
            ));
        }
        let (collectors, states) = Collectors::prepare(&request, device, rows.clone());
        let running = collectors.running();
        let collectors = Arc::new(Mutex::new(collectors));
        let sampler = sampler(Arc::clone(&collectors))?;
        let answer = serde_json::json!({
            "rank": rank,
            "selected": true,
            "path": path,
            "collectors": states,
        });
        tracing::info!(session, path = %path.display(), "a profiling session started");
        Ok((
            Some(Self {
                id: session,
                request,
                writer: Some(writer),
                rows,
                collectors,
                running,
                skipped: 0,
                taken: 0,
                recorded: 0,
                intended: false,
                ended: None,
                pending: VecDeque::new(),
                submitted: None,
                open: HashMap::new(),
                reported: (0, 0, false),
                sampler: Some(sampler),
            }),
            answer,
        ))
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Account the start of step `step` (a batch's forward) of `stage`:
    /// the window takes it or not, and the collectors are to be armed or
    /// disarmed before its forward accordingly ([`Session::next`]).
    pub(crate) fn begin(&mut self, step: u64, stage: &str) {
        let staged =
            self.request.stages.is_empty() || self.request.stages.iter().any(|s| s == stage);
        let mut taken = false;
        if self.ended.is_none() && staged {
            if self.skipped < self.request.start_step {
                self.skipped += 1;
            } else if self.request.steps.is_none_or(|steps| self.taken < steps) {
                taken = true;
                self.taken += 1;
            }
        }
        let collecting = self.running.load(Ordering::Acquire) != 0;
        if collecting && taken != self.intended {
            self.pending
                .push_back(if taken { Action::Arm } else { Action::Disarm });
            self.intended = taken;
        }
        let now = cuda::cupti::timestamp().unwrap_or(0);
        self.open.insert(step, (stage.to_owned(), taken, now));
    }

    /// Account the end of step `step`'s forward: a step the window took
    /// counts, and the window's last disarms the collectors.
    pub(crate) fn end(&mut self, step: u64) {
        let Some((stage, taken, begin)) = self.open.remove(&step) else {
            return;
        };
        self.rows.send(Message::Step(StepRow {
            step,
            stage,
            selected: taken,
            begin,
            end: cuda::cupti::timestamp().ok(),
        }));
        if !taken {
            return;
        }
        self.recorded += 1;
        if self
            .request
            .steps
            .is_some_and(|steps| self.recorded >= steps)
        {
            self.ended = Some("window".to_owned());
            if self.intended {
                self.pending.push_back(Action::Disarm);
                self.intended = false;
            }
        } else if self.recorded.is_multiple_of(RANGE_CYCLE)
            && self.running.load(Ordering::Acquire) & collectors::bit(Collector::RangeMetrics) != 0
        {
            self.pending.push_back(Action::CycleRanges);
        }
    }

    /// Submit the next change of the collectors as a job the current step's
    /// forward waits for; `None` when none is due or one is in flight.
    pub(crate) fn next(&mut self, jobs: &JobQueue) -> Option<JobId> {
        if self.submitted.is_some() {
            return None;
        }
        let action = self.pending.pop_front()?;
        let collectors = Arc::clone(&self.collectors);
        let job = Job::new();
        let issued = job.on(None).launch_host_fn(Box::new(move || {
            let mut collectors = lock(&collectors);
            match action {
                Action::Arm => collectors.arm(),
                Action::Disarm => collectors.disarm(),
                Action::CycleRanges => collectors.cycle_ranges(),
            }
        }));
        if let Err(error) = issued {
            tracing::error!("a profiling session's collectors were not switched: {error}");
            return None;
        }
        self.submitted = Some(action);
        Some(jobs.start(job, true))
    }

    /// Apply the completion of the job [`Session::next`] submitted.
    pub(crate) fn complete(&mut self, completion: Completion) {
        let Some(action) = self.submitted.take() else {
            return;
        };
        if let Err(failure) = completion.check() {
            tracing::error!(
                "a profiling session's collectors failed to {action:?}: {}",
                failure.error
            );
        }
    }

    /// What the rank reports since its last report: the steps its window
    /// took, the bytes its file grew by, and that its window ended (once).
    pub(crate) fn progress(&mut self) -> BTreeMap<String, u64> {
        let bytes = self.writer.as_ref().map_or(0, Writer::bytes);
        let mut progress = BTreeMap::new();
        let (steps, reported_bytes, ended) = self.reported;
        if self.recorded > steps {
            progress.insert(format!("{}/steps", self.id), self.recorded - steps);
        }
        if bytes > reported_bytes {
            progress.insert(format!("{}/bytes", self.id), bytes - reported_bytes);
        }
        let ending = self.ended.is_some()
            && !self.intended
            && self.pending.is_empty()
            && self.submitted.is_none();
        if ending && !ended {
            progress.insert(format!("{}/ended", self.id), 1);
        }
        self.reported = (
            self.recorded.max(steps),
            bytes.max(reported_bytes),
            ended || ending,
        );
        progress
    }

    /// Release `collector`, which the quality-of-service guard turned off
    /// for `reason`; answers whether the rank ran it.
    pub(crate) fn release(&mut self, collector: Collector, reason: &str) -> Answer {
        let released = lock(&self.collectors).release(collector, "released", reason);
        serde_json::json!({ "released": released })
    }

    /// End the session for `reason`: disarm and release the collectors,
    /// write what remains and close the file. Runs while the rank runs no
    /// batch.
    pub(crate) fn stop(mut self, reason: &str) -> Answer {
        if let Some((stop, thread)) = self.sampler.take() {
            drop(stop);
            let _ = thread.join();
        }
        lock(&self.collectors).release_all(reason);
        clocks(&self.rows, "end");
        self.rows.send(Message::Metadata(vec![
            ("ended".to_owned(), realtime().to_string()),
            ("end_reason".to_owned(), reason.to_owned()),
            ("steps_recorded".to_owned(), self.recorded.to_string()),
            (
                "window_ended".to_owned(),
                self.ended.clone().unwrap_or_default(),
            ),
        ]));
        // The writer finishes once every sender of rows is gone: the
        // collectors' (the sampler thread ended) and the session's.
        let Self {
            id,
            writer,
            rows,
            collectors,
            recorded,
            ..
        } = self;
        drop(collectors);
        drop(rows);
        let summary = match writer.map(Writer::finish) {
            Some(Ok(summary)) => serde_json::to_value(summary).unwrap_or_default(),
            Some(Err(error)) => serde_json::json!({ "error": error.to_string() }),
            None => serde_json::Value::Null,
        };
        tracing::info!(session = id, reason, "a profiling session ended");
        serde_json::json!({
            "session": id,
            "steps": recorded,
            "reason": reason,
            "file": summary,
        })
    }
}

/// The file's metadata at the session's start.
fn metadata(
    session: u64,
    world: u64,
    rank: u32,
    device: Option<DeviceId>,
    request: &proto::profile::SessionRequest,
) -> Vec<(String, String)> {
    let mut entries = vec![
        ("session".to_owned(), session.to_string()),
        ("world".to_owned(), world.to_string()),
        ("rank".to_owned(), rank.to_string()),
        ("pid".to_owned(), std::process::id().to_string()),
        ("started".to_owned(), realtime().to_string()),
        (
            "request".to_owned(),
            serde_json::to_string(request).unwrap_or_default(),
        ),
        ("build".to_owned(), proto::BUILD.to_owned()),
        (
            "cupti_library".to_owned(),
            cuda::cupti::library().to_owned(),
        ),
    ];
    if let Ok(version) = cuda::cupti::version() {
        entries.push(("cupti_api_version".to_owned(), version.to_string()));
    }
    if let Ok(host) = rustix::system::uname().nodename().to_str() {
        entries.push(("host".to_owned(), host.to_owned()));
    }
    if let Some(device) = device {
        entries.push(("device".to_owned(), device.0.to_string()));
        if let Ok(uuid) = device.uuid() {
            entries.push(("device_uuid".to_owned(), format!("{uuid}")));
        }
        if let Ok(chip) = cuda::cupti::metrics::chip_name(device.0 as usize) {
            entries.push(("chip".to_owned(), chip));
        }
    }
    entries
}

/// Sample the host's clocks, and calibrate the fence probes' clocks, into
/// the file's `clocks`.
fn clocks(rows: &Rows, source: &'static str) {
    let before = realtime();
    let cupti = cuda::cupti::timestamp().ok();
    let monotonic = crate::profiling::fences::monotonic();
    let after = realtime();
    rows.send(Message::Clock(ClockRow {
        source,
        device: None,
        gpu: None,
        realtime: before / 2 + after / 2,
        cupti,
        monotonic: Some(monotonic),
        uncertainty: (after - before) / 2,
    }));
    if let Some(fences) = Fences::installed() {
        for alignment in fences.calibrate() {
            rows.send(Message::Clock(ClockRow {
                source: "globaltimer",
                device: Some(alignment.device),
                gpu: Some(alignment.gpu),
                realtime: alignment.realtime,
                cupti: Some(alignment.cupti),
                monotonic: None,
                uncertainty: alignment.uncertainty,
            }));
        }
    }
}

/// The thread that drains the armed samplers every [`DRAIN_PERIOD`], until
/// its sender is dropped.
fn sampler(collectors: Arc<Mutex<Collectors>>) -> Result<(mpsc::Sender<()>, JoinHandle<()>)> {
    let (stop, stopped) = mpsc::channel::<()>();
    let thread = std::thread::Builder::new()
        .name("profile-sampler".into())
        .spawn(move || {
            while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(DRAIN_PERIOD) {
                lock(&collectors).drain();
            }
        })
        .map_err(|error| Error::transport_source("starting the profile sampler", error))?;
    Ok((stop, thread))
}
