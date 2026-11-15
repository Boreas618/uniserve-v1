//! A profiling session on one rank: its window over the rank's steps, its collectors and its file.
//!
//! The leader daemon starts a session on every rank of the epoch that executes with the operation
//! `uniserve.profile_start`, which reaches each rank's worker as a [`Change`] in its queue, so
//! it takes effect between two batches ([`Session::change`]). A rank the session does not select
//! responds so and records nothing. A selected rank prepares the collectors
//! ([`super::collectors`]) and creates its file ([`super::artifact`]); its worker arms the
//! collectors before the forward of the first step its window takes and disarms them after the
//! last, so that what the collectors record begins and ends at step boundaries:
//!
//! - A step is one batch's forward; its stage is its call kind.
//! - The window skips `start_step` steps of the selected stages, then takes `steps` of them, or
//!   every one until the session ends. Steps of the other stages are not recorded: the collectors
//!   are disarmed before such a step and armed again before the next selected one.
//! - Every step during the session is listed in the file (`steps`), with whether the window took
//!   it.
//!
//! A window is made of whole steps, so a worker whose rank holds a session runs its steps one
//! at a time: it tells the session each step's begin and end ([`Session::begin`],
//! [`Session::end`]) and, when either returns `true`, waits for the rank's streams and then
//! changes the collectors ([`Session::apply`]).
//!
//! A sampler thread drains the armed samplers every [`DRAIN_PERIOD`]. The rank reports its
//! progress with every batch result (`ForwardStats::profile`, [`Session::progress`]): the steps
//! its window took, the bytes of its file, and that its window ended. The daemon ends the session
//! with `uniserve.profile_stop` when every window ended, its time passed, the operator ends it,
//! its epoch changes, or the quality-of-service guard ends it; the guard first releases collectors
//! one by one (`uniserve.profile_release`).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::Sender;
use platform::DeviceId;
use proto::profile::{
    Collector, FileReport, RankRelease, RankStart, RankStop, SessionRequest, SessionStart,
};

use super::artifact::{ClockRow, Message, Writer, send};
use super::collectors::{self, Collectors};
use super::markers::{Markers, realtime};
use crate::worker::{Error, Result};
use service::lock;

/// How often the sampler thread drains the armed samplers.
pub(super) const DRAIN_PERIOD: Duration = Duration::from_millis(100);

/// A change of the collectors a step's forward waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Arm,
    Disarm,
    CycleRanges,
}

/// A change of a rank's session, as the daemon's operation reaches the rank's worker.
#[derive(Clone)]
pub(crate) enum Change {
    /// Start a session on the rank.
    Start { start: Box<SessionStart>, rank: u32 },
    /// End session `session` for `reason`.
    Stop {
        session: u64,
        reason: String,
        rank: u32,
    },
    /// Turn `collector` of session `session` off for `reason`.
    Release {
        session: u64,
        collector: Collector,
        reason: String,
        rank: u32,
    },
}

/// A rank's answer to a [`Change`]: its report of the start, the end or the release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProfileAnswer {
    Started(RankStart),
    Stopped(RankStop),
    Released(RankRelease),
}

impl ProfileAnswer {
    /// The report of a start.
    ///
    /// # Errors
    ///
    /// An answer to another change.
    pub(crate) fn started(self) -> Result<RankStart> {
        match self {
            Self::Started(started) => Ok(started),
            other => Err(Error::Invariant(format!(
                "a profiling start was answered with {other:?}"
            ))),
        }
    }

    /// The report of an end.
    ///
    /// # Errors
    ///
    /// An answer to another change.
    pub(crate) fn stopped(self) -> Result<RankStop> {
        match self {
            Self::Stopped(stopped) => Ok(stopped),
            other => Err(Error::Invariant(format!(
                "a profiling stop was answered with {other:?}"
            ))),
        }
    }

    /// The report of a release.
    ///
    /// # Errors
    ///
    /// An answer to another change.
    pub(crate) fn released(self) -> Result<RankRelease> {
        match self {
            Self::Released(released) => Ok(released),
            other => Err(Error::Invariant(format!(
                "a profiling release was answered with {other:?}"
            ))),
        }
    }
}

/// A session on a rank the session selected.
pub(crate) struct Session {
    id: u64,
    device: DeviceId,
    request: SessionRequest,
    writer: Option<Writer>,
    /// The file's writer, for the session's own rows.
    rows: Sender<Message>,
    collectors: Arc<Mutex<Collectors>>,
    /// The collectors running ([`collectors::bit`]), read at every step instead of the collectors'
    /// lock, which the sampler thread holds while it drains the samplers.
    running: Arc<AtomicU32>,
    /// Steps of the selected stages skipped before the window opened.
    skipped: u64,
    /// Steps the window took as they began.
    taken: u64,
    /// Steps the window took whose forward ended.
    recorded: u64,
    /// Whether the collectors are armed once the actions queued are done.
    intended: bool,
    /// Why the window ended, once it did.
    ended: Option<String>,
    pending: VecDeque<Action>,
    /// The steps begun and not ended: stage, whether the window took it, when it began.
    open: HashMap<u64, (String, bool, u64)>,
    /// What the rank reported so far: steps, bytes, ended.
    reported: (u64, u64, bool),
    sampler: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
}

impl Session {
    /// Performs `change` on `slot`, the session of the rank whose CUDA device is `device`, and
    /// returns the rank's answer to the daemon's operation: a start fills the slot when the
    /// session selects the rank, a stop of the session in the slot empties it, and a release
    /// turns one of its collectors off.
    ///
    /// The worker calls this between two batches, with nothing of the rank in flight: a start
    /// prepares collectors on the device and a stop releases them.
    ///
    /// # Errors
    ///
    /// Fails when a session already records the rank, the request is invalid, or the rank cannot
    /// create its file. A stop or a release that names another session than the slot's changes
    /// nothing and says so in its response.
    pub(crate) fn change(
        slot: &mut Option<Self>,
        change: Change,
        device: DeviceId,
    ) -> Result<ProfileAnswer> {
        match change {
            Change::Start { start, rank } => {
                if let Some(session) = slot {
                    return Err(Error::Invalid(format!(
                        "profiling session {} records the rank",
                        session.id
                    )));
                }
                let (session, response) = Self::start(*start, rank, device)?;
                *slot = session;
                Ok(ProfileAnswer::Started(response))
            }
            Change::Stop {
                session,
                reason,
                rank,
            } => Ok(ProfileAnswer::Stopped(
                match slot.take_if(|running| running.id == session) {
                    Some(running) => running.stop(&reason, rank),
                    None => RankStop::unselected(rank, session),
                },
            )),
            Change::Release {
                session,
                collector,
                reason,
                rank,
            } => {
                let released = match slot {
                    Some(running) if running.id == session => running.release(collector, &reason),
                    _ => false,
                };
                Ok(ProfileAnswer::Released(RankRelease { rank, released }))
            }
        }
    }

    /// Starts the session `start` on rank `rank`, whose CUDA device is `device`, and returns it
    /// with the rank's response: no session when it does not select the rank.
    ///
    /// # Errors
    ///
    /// Fails when the request is invalid or the rank cannot create its file; collectors that
    /// cannot run are reported in the response, and the session runs the others.
    fn start(
        start: SessionStart,
        rank: u32,
        device: DeviceId,
    ) -> Result<(Option<Self>, RankStart)> {
        let SessionStart {
            session,
            epoch,
            request,
        } = start;
        request.validate().map_err(Error::Invalid)?;
        if !request.workers.is_empty() && !request.workers.contains(&rank) {
            return Ok((None, RankStart::unselected(rank)));
        }

        std::fs::create_dir_all(&request.output_dir).map_err(|error| {
            Error::transport_source(format!("creating {}", request.output_dir.display()), error)
        })?;
        let path = request
            .output_dir
            .join(format!("profile-{session}-epoch{epoch}-rank{rank}.sqlite"));
        let writer = Writer::create(
            path.clone(),
            metadata(session, epoch, rank, device, &request),
        )?;
        let rows = writer.rows();
        clocks(&rows, device, "start");
        if let Some(markers) = Markers::installed() {
            send(&rows, Message::Instances(markers.instances()));
        }

        let (collectors, states) = Collectors::prepare(&request, Some(device), rows.clone());
        // What the collectors measured they hold on the device, which the rank's quota reserves.
        let device_storage = collectors.device_storage();
        let running = collectors.running();
        let collectors = Arc::new(Mutex::new(collectors));
        let sampler = sampler(Arc::clone(&collectors))?;
        let response = RankStart {
            rank,
            selected: true,
            refused: None,
            device_storage,
            path: Some(path.clone()),
            collectors: states,
        };
        tracing::info!(session, path = %path.display(), "a profiling session started");

        Ok((
            Some(Self {
                id: session,
                device,
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
                open: HashMap::new(),
                reported: (0, 0, false),
                sampler: Some(sampler),
            }),
            response,
        ))
    }

    /// Accounts the start of step `step` (a batch's id) of `stage` (its call kind), before the
    /// step's first launch: the window takes it or not, and the collectors are to be armed or
    /// disarmed before its forward accordingly. Returns whether [`Self::apply`] has work, which
    /// must run with nothing of the rank in flight.
    pub(crate) fn begin(&mut self, step: u64, stage: &str) -> bool {
        let selected =
            self.request.stages.is_empty() || self.request.stages.iter().any(|s| s == stage);
        let mut taken = false;
        if self.ended.is_none() && selected {
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
        let now = platform::cupti::timestamp().unwrap_or(0);
        self.open.insert(step, (stage.to_owned(), taken, now));
        !self.pending.is_empty()
    }

    /// Accounts the end of step `step`'s forward, once the device completed it: a step the window
    /// took counts, and the window's last disarms the collectors. Returns whether [`Self::apply`]
    /// has work.
    pub(crate) fn end(&mut self, step: u64) -> bool {
        let Some((stage, taken, begin)) = self.open.remove(&step) else {
            return false;
        };
        send(
            &self.rows,
            Message::Step {
                step,
                stage,
                selected: taken,
                begin,
                end: platform::cupti::timestamp().ok(),
            },
        );
        if !taken {
            return !self.pending.is_empty();
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
        } else if self
            .recorded
            .is_multiple_of(self.request.range_metrics.read_steps())
            && self.running.load(Ordering::Acquire) & collectors::bit(Collector::RangeMetrics) != 0
        {
            self.pending.push_back(Action::CycleRanges);
        }
        !self.pending.is_empty()
    }

    /// Performs the changes of the collectors that are due, on the rank's worker between two
    /// forwards: the caller waited for the rank's streams, so a collector starts or stops with
    /// nothing of the rank in flight and records whole steps.
    pub(crate) fn apply(&mut self) {
        let mut collectors = lock(&self.collectors);
        while let Some(action) = self.pending.pop_front() {
            match action {
                Action::Arm => collectors.arm(),
                Action::Disarm => collectors.disarm(),
                Action::CycleRanges => collectors.cycle_ranges(),
            }
        }
    }

    /// What the rank reports since its last report (`ForwardStats::profile`): the steps its window
    /// took (`{session}/steps`), the bytes its file grew by (`{session}/bytes`), and that its
    /// window ended (`{session}/ended`, once).
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
        let ending = self.ended.is_some() && !self.intended && self.pending.is_empty();
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

    /// Releases `collector`, which the quality-of-service guard turned off for `reason`; whether
    /// the rank ran it.
    fn release(&mut self, collector: Collector, reason: &str) -> bool {
        lock(&self.collectors).release(collector, "released", reason)
    }

    /// Ends the session for `reason` on rank `rank`: disarms and releases the collectors, writes
    /// what remains and closes the file. Runs while the rank runs no batch. Reports the steps
    /// recorded and what the file holds.
    fn stop(mut self, reason: &str, rank: u32) -> RankStop {
        if let Some((stop, thread)) = self.sampler.take() {
            drop(stop);
            let _ = thread.join();
        }
        lock(&self.collectors).release_all(reason);
        clocks(&self.rows, self.device, "end");
        send(
            &self.rows,
            Message::Metadata(vec![
                ("ended".to_owned(), realtime().to_string()),
                ("end_reason".to_owned(), reason.to_owned()),
                ("steps_recorded".to_owned(), self.recorded.to_string()),
                (
                    "window_ended".to_owned(),
                    self.ended.clone().unwrap_or_default(),
                ),
            ]),
        );

        // The writer finishes once every sender of rows is gone: the collectors' (the sampler
        // thread ended) and the session's.
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
        let file = writer.map(|writer| match writer.finish() {
            Ok(finished) => finished,
            Err(error) => FileReport::Failed {
                error: error.to_string(),
            },
        });
        tracing::info!(session = id, reason, "a profiling session ended");

        RankStop {
            rank,
            session: id,
            selected: true,
            steps: recorded,
            reason: reason.to_owned(),
            file,
        }
    }
}

/// The file's metadata at the session's start.
fn metadata(
    session: u64,
    epoch: u64,
    rank: u32,
    device: DeviceId,
    request: &SessionRequest,
) -> Vec<(String, String)> {
    let mut entries = vec![
        ("session".to_owned(), session.to_string()),
        ("epoch".to_owned(), epoch.to_string()),
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
            platform::cupti::library().to_owned(),
        ),
    ];
    if let Ok(version) = platform::cupti::version() {
        entries.push(("cupti_api_version".to_owned(), version.to_string()));
    }
    if let Ok(host) = rustix::system::uname().nodename().to_str() {
        entries.push(("host".to_owned(), host.to_owned()));
    }
    entries.push(("device".to_owned(), device.0.to_string()));
    if let Ok(uuid) = device.uuid() {
        entries.push(("device_uuid".to_owned(), format!("{uuid}")));
    }
    if let Ok(chip) = platform::cupti::metrics::chip_name(device.0 as usize) {
        entries.push(("chip".to_owned(), chip));
    }
    entries
}

/// Samples the host's clocks, and calibrates `device`'s range marker clock, into the file's
/// `clocks`.
fn clocks(rows: &Sender<Message>, device: DeviceId, source: &'static str) {
    let before = realtime();
    let cupti = platform::cupti::timestamp().ok();
    // The buffer pool's protocol clock, which the scheduler states its instants in: BPF's
    // `ktime_get_ns`.
    let monotonic = storage::runtime::monotonic_nanos();
    let after = realtime();
    send(
        rows,
        Message::Clock(ClockRow {
            source,
            device: None,
            gpu: None,
            realtime: before / 2 + after / 2,
            cupti,
            monotonic: Some(monotonic),
            uncertainty: (after - before) / 2,
        }),
    );

    if let Some(sample) = Markers::installed().and_then(|markers| markers.calibrate(device)) {
        send(rows, Message::Clock(sample));
    }
}

/// The thread that drains the armed samplers every [`DRAIN_PERIOD`], until its sender is dropped.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A request for the range timeline over every stage, of the ranks `workers`.
    fn request(workers: &[u32]) -> SessionRequest {
        serde_json::from_value(serde_json::json!({
            "collectors": ["fence_timeline"],
            "workers": workers,
            "output_dir": "/nonexistent/profiles",
        }))
        .unwrap()
    }

    #[test]
    fn a_rank_the_session_does_not_select_records_nothing() {
        let mut slot = None;
        let start = SessionStart {
            session: 7,
            epoch: 2,
            request: request(&[0]),
        };
        let response = Session::change(
            &mut slot,
            Change::Start {
                start: Box::new(start),
                rank: 1,
            },
            DeviceId(1),
        )
        .unwrap();

        assert_eq!(response, ProfileAnswer::Started(RankStart::unselected(1)));
        assert!(slot.is_none());
    }

    #[test]
    fn a_stop_or_release_of_a_session_the_rank_does_not_hold_changes_nothing() {
        let mut slot = None;

        let stopped = Session::change(
            &mut slot,
            Change::Stop {
                session: 7,
                reason: "operator".to_owned(),
                rank: 3,
            },
            DeviceId(3),
        )
        .unwrap();
        assert_eq!(stopped, ProfileAnswer::Stopped(RankStop::unselected(3, 7)));

        let released = Session::change(
            &mut slot,
            Change::Release {
                session: 7,
                collector: Collector::FenceTimeline,
                reason: "slowdown".to_owned(),
                rank: 3,
            },
            DeviceId(3),
        )
        .unwrap();
        assert_eq!(
            released,
            ProfileAnswer::Released(RankRelease {
                rank: 3,
                released: false
            })
        );
        assert!(slot.is_none());
    }
}
