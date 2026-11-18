//! Profiling sessions of the epoch that executes (`profiling_config.session`,
//! `docs/profiling.md`).
//!
//! A change of the deployment that states a session starts it on every worker of each
//! replica's executing epoch at the engine's next step boundary (the engine command
//! `StartProfile`), under one identity on every replica; it records within each worker's window
//! of steps and ends, with every worker's file written (`StopProfile`), when the first of these
//! happens:
//!
//! - every selected worker's window of steps ended, as the workers report
//!   with their batch results (`ForwardStats::profile`);
//! - its window of seconds passed;
//! - a change of the deployment states another session, or none;
//! - its epoch stops executing: an epoch change promotes a successor, or a
//!   trial rolls back. Neither carries a session into the next epoch;
//! - the quality-of-service guard ends it.
//!
//! At most one session runs, under one identity on every replica. Its task
//! decides as each statistics report of its replica arrives
//! ([`crate::observe::Observer::records`]), and at its window's seconds. The
//! guard measures the average step time of the session's epoch over the
//! request's `baseline_s` before the session started and over its last
//! `guard_window_s`. With `max_slowdown`, a step time past the baseline by more
//! than that fraction turns off the session's costliest collector still
//! running (`ReleaseProfile`, in `Collector::COSTLIEST_FIRST` order), at most
//! once a guard window so that each change is measured; a slowdown that
//! remains with no collector left ends the session. Every state the session enters, and every
//! action of its guard, is a `log` event of source `profile` (`crate::status::profile`).

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use proto::admin::Rates;
use proto::profile::{Collector, RankStart, RankStop, SessionRequest, SessionStart};
use proto::request::wire;
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::engine::client::{EngineClient, EngineState};
use crate::leader::{Replica, UpdateError};

/// Where a session is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    /// Its workers record.
    Running,
    /// Its workers write their files.
    Ending,
    /// It ended.
    Ended,
}

/// What the quality-of-service guard did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardActionKind {
    /// It turned a collector off.
    Release,
    /// It ended the session: no collector was left to turn off.
    End,
}

/// One action of the quality-of-service guard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GuardAction {
    /// When it acted, in seconds since the Unix epoch.
    #[serde(serialize_with = "unix_seconds")]
    pub time: SystemTime,
    pub action: GuardActionKind,
    pub collector: Option<Collector>,
    /// The slowdown that caused it: the session's step time over the
    /// baseline's, less one.
    pub slowdown: f64,
}

/// What the guard measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Overhead {
    /// The epoch's average step before the session, microseconds.
    pub baseline_step_us: f64,
    /// The session's average step over its last guard window.
    pub step_us: f64,
    /// `step_us / baseline_step_us - 1`.
    pub slowdown: f64,
}

/// One replica's session, running or the last that ended.
#[derive(Debug, Clone, Serialize)]
pub struct Session {
    pub id: u64,
    pub epoch: u64,
    pub state: SessionState,
    /// When it started, in seconds since the Unix epoch.
    #[serde(serialize_with = "unix_seconds")]
    pub started: SystemTime,
    /// When it ended, in seconds since the Unix epoch.
    #[serde(serialize_with = "optional_unix_seconds")]
    pub ended: Option<SystemTime>,
    /// Why it ended: `window`, `seconds`, `changed` (a change of the deployment stated another
    /// session or none, whoever wrote it), `epoch change`, `rollback`, `quality of service`, or a
    /// failure.
    pub end_reason: Option<String>,
    pub request: SessionRequest,
    /// Each worker's answer to the start: its rank, whether the session
    /// selects it, its file and its collectors' states.
    pub workers: Vec<RankStart>,
    /// Each worker's answer to the end: what its file holds.
    pub files: Vec<RankStop>,
    /// The steps the workers' windows took, summed over the workers.
    pub steps: u64,
    /// The bytes the workers' files hold, summed.
    pub bytes_written: u64,
    /// The workers whose window ended.
    pub windows_ended: u64,
    /// The collectors still running.
    pub collectors: Vec<Collector>,
    pub baseline: Option<Rates>,
    pub overhead: Option<Overhead>,
    pub guard: Vec<GuardAction>,
    /// The device storage the session holds on a selected worker's GPU, the largest any
    /// worker measured as the session started there (a range_metrics pass), which the workers
    /// reserve in the GPUs' quotas: the KV cache gave back tail segments for it where a quota
    /// lacked room, and grows back once the session ends.
    pub device_storage: u64,
}

/// A replica's sessions: the one that runs or the last that ended.
#[derive(Default)]
pub(crate) struct Profiles {
    session: std::sync::Mutex<Option<Session>>,
    /// When the guard last acted, for the running session.
    acted: std::sync::Mutex<Option<Instant>>,
    /// Cancelled when the running session ends, which ends its task.
    ending: std::sync::Mutex<CancellationToken>,
}

/// `time` in seconds since the Unix epoch.
///
/// # Errors
///
/// A time before the Unix epoch, which no clock of a serving host states.
fn unix_seconds<S: serde::Serializer>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error> {
    let seconds = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(serde::ser::Error::custom)?;
    serializer.serialize_f64(seconds.as_secs_f64())
}

/// [`unix_seconds`] of a time that may be unset.
fn optional_unix_seconds<S: serde::Serializer>(
    time: &Option<SystemTime>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match time {
        Some(time) => unix_seconds(time, serializer),
        None => serializer.serialize_none(),
    }
}

impl Replica {
    /// Makes the replica's session `target`'s (`profiling_config.session`) on the engine
    /// `client` serves with: ends the running session, every worker writing its file, and starts
    /// `target` under the deployment's next session identity, when it states one. The caller
    /// holds the replica while no engine replaces it.
    ///
    /// # Errors
    ///
    /// As [`Replica::start_profile`].
    pub(crate) async fn profile(
        self: &Arc<Self>,
        client: &EngineClient,
        target: Option<SessionRequest>,
    ) -> Result<(), UpdateError> {
        if let Some(id) = self.profile_running() {
            end(self, Some(client), id, "changed", true).await;
        }
        let Some(request) = target else {
            return Ok(());
        };
        let id = self
            .registry
            .upgrade()
            .map_or(1, |replicas| replicas.next_profile());
        self.start_profile(client, id, request).await.map(drop)
    }

    /// Starts session `id` of `request` on the executing epoch of `client`, and returns it.
    ///
    /// # Errors
    ///
    /// Refuses an invalid request, a baseline or guard window longer than the replica keeps
    /// its statistics for, and a session while one runs; fails when no engine serves or a
    /// worker cannot start it. A worker refuses what it cannot run: its GPU's quota cannot fit
    /// the session's storage, or a collector CUPTI cannot run beside its graphs' PC sampling.
    async fn start_profile(
        self: &Arc<Self>,
        client: &EngineClient,
        id: u64,
        request: SessionRequest,
    ) -> Result<Session, UpdateError> {
        request.validate().map_err(UpdateError::Refused)?;
        let baseline_span = Duration::from_secs_f64(request.baseline_s);
        let guard_window = Duration::from_secs_f64(request.guard_window_s);
        let kept = self.observer.retention();
        if baseline_span.max(guard_window) > kept {
            return Err(UpdateError::Refused(format!(
                "the replica keeps its statistics for {} s (update_config's baseline and \
                 rollback window), shorter than the session's baseline or guard window",
                kept.as_secs_f64()
            )));
        }
        if let Some(running) = self.profile_running() {
            return Err(UpdateError::Refused(format!(
                "profiling session {running} runs"
            )));
        }
        let epoch = client.epoch();
        let started_at = Instant::now();
        let baseline = self.observer.rates(
            epoch,
            started_at.checked_sub(baseline_span).unwrap_or(started_at),
            baseline_span,
        );
        let start = SessionStart {
            session: id,
            epoch,
            request: request.clone(),
        };
        // The device storage the collectors hold on each selected worker's GPU (a range_metrics
        // pass) has no size CUPTI states: each worker measures it as the session starts there,
        // and reserves it in its GPU's quota. A worker whose quota lacks the room refuses,
        // stating the bytes, and leaves the session to the workers that started it, which end
        // it; the KV cache then gives back tail segments for those bytes, as an epoch change's
        // preparation does, the requests holding them running on, and the session starts once
        // more, its storage now known. Any other refusal ends the start.
        let mut room = 0;
        let workers = loop {
            let workers = match client.start_profile(&start).await {
                Ok(workers) => workers,
                Err(error) => {
                    self.abandon_profile(id, room, client).await;
                    return Err(UpdateError::Failed(anyhow::anyhow!(
                        "profile_start: {error}"
                    )));
                }
            };
            let Some(refusal) = workers.iter().find_map(|worker| worker.refused.clone()) else {
                break workers;
            };
            self.abandon_profile(id, room, client).await;
            let needed = workers
                .iter()
                .filter(|worker| worker.refused.is_some())
                .map(|worker| worker.device_storage)
                .max()
                .unwrap_or(0);
            if room > 0 || needed == 0 {
                return Err(UpdateError::Refused(refusal));
            }
            client
                .reserve_storage(crate::engine::client::Room::Each(needed))
                .await
                .map_err(|error| {
                    UpdateError::Refused(format!(
                        "the session holds {needed} bytes of device storage on a selected \
                         worker's GPU (a range_metrics pass, as the worker measured it), for \
                         which the KV cache cannot make room: {error}"
                    ))
                })?;
            room = needed;
        };
        // What the selected workers measured their collectors hold, the largest of them.
        let device_storage = workers
            .iter()
            .map(|worker| worker.device_storage)
            .max()
            .unwrap_or(0)
            .max(room);
        // The collectors some selected worker runs.
        let collectors: BTreeSet<Collector> = workers
            .iter()
            .filter(|worker| worker.selected)
            .flat_map(|worker| {
                worker
                    .collectors
                    .iter()
                    .filter(|(_, state)| *state == "prepared")
                    .filter_map(|(name, _)| {
                        Collector::ALL
                            .into_iter()
                            .find(|collector| collector.as_str() == name)
                    })
            })
            .collect();
        let session = Session {
            id,
            epoch,
            state: SessionState::Running,
            started: SystemTime::now(),
            ended: None,
            end_reason: None,
            request,
            workers,
            files: Vec::new(),
            steps: 0,
            bytes_written: 0,
            windows_ended: 0,
            collectors: collectors.into_iter().collect(),
            baseline,
            overhead: None,
            guard: Vec::new(),
            device_storage,
        };
        let ending = CancellationToken::new();
        *service::lock(&self.profiles.session) = Some(session.clone());
        *service::lock(&self.profiles.acted) = None;
        *service::lock(&self.profiles.ending) = ending.clone();
        info!(session = id, epoch, "a profiling session started");
        crate::status::profile(&self.recorder, self.ordinal, &session);
        tokio::spawn(watch(Arc::clone(self), id, started_at, ending));
        Ok(session)
    }

    /// The running session's identity; `None` while none runs.
    pub(crate) fn profile_running(&self) -> Option<u64> {
        service::lock(&self.profiles.session)
            .as_ref()
            .filter(|session| session.state == SessionState::Running)
            .map(|session| session.id)
    }

    /// End session `id` on the workers that started it after another
    /// worker refused or failed to: the session never ran, and each worker
    /// that started it closes its file and releases its collectors and
    /// their reserved storage. The KV cache grows back into the room it made
    /// for the session's `device_storage`.
    async fn abandon_profile(&self, id: u64, device_storage: u64, client: &EngineClient) {
        if let Err(error) = client.stop_profile(id, "not started").await {
            warn!(
                session = id,
                "ending a session other workers did not start failed: {error}"
            );
        }
        if device_storage > 0 {
            grow_back(client, id).await;
        }
    }
}

impl Profiles {
    /// Read the workers' progress of `session` from the statistics they
    /// reported, and the guard's measure.
    fn progress(&self, session: &mut Session, observer: &crate::observe::Observer) {
        let progress = observer.profile(session.id);
        session.steps = progress.steps;
        session.bytes_written = progress.bytes;
        session.windows_ended = progress.windows_ended;
        let guard_window = Duration::from_secs_f64(session.request.guard_window_s);
        session.overhead = session.baseline.and_then(|baseline| {
            let since = Instant::now().checked_sub(guard_window)?;
            let current = observer.rates(session.epoch, since, guard_window)?;
            (baseline.step_us > 0.0).then(|| Overhead {
                baseline_step_us: baseline.step_us,
                step_us: current.step_us,
                slowdown: current.step_us / baseline.step_us - 1.0,
            })
        });
    }
}

/// Grow the engine's KV cache back into the room session `id` made for its
/// device storage, once the workers released it.
async fn grow_back(client: &EngineClient, id: u64) {
    match client.resize_cache(wire::CacheChange::Grow).await {
        Ok(()) => info!(
            session = id,
            "the KV cache grew back after the profiling session"
        ),
        Err(error) => warn!(
            session = id,
            "growing the KV cache back after the profiling session failed: {error}"
        ),
    }
}

/// End session `id` for `reason`, when it still runs: the workers of an epoch that still
/// serves write their files, and, with `grow`, the KV cache grows back into the room the
/// session made for its device storage; an epoch that stops executing (an epoch change's
/// promotion, a rollback) keeps its cache as the change left it, and its workers wrote what
/// they recorded as their epoch closed or keep it in their files' logs.
pub(crate) async fn end(
    replica: &crate::leader::Replica,
    client: Option<&EngineClient>,
    id: u64,
    reason: &str,
    grow: bool,
) {
    let (epoch, device_storage) = {
        let mut session = service::lock(&replica.profiles.session);
        let Some(running) = session
            .as_mut()
            .filter(|session| session.id == id && session.state == SessionState::Running)
        else {
            return;
        };
        replica.profiles.progress(running, &replica.observer);
        // Ending from here on: a second end finds it ending.
        running.state = SessionState::Ending;
        (running.epoch, running.device_storage)
    };
    service::lock(&replica.profiles.ending).cancel();
    let mut files = Vec::new();
    let mut failure = None;
    // Only an epoch that still serves writes its files: a failed one's
    // workers keep what they committed (the files' write-ahead logs). The
    // workers answer once their files are written, or their engine's end
    // ends the wait.
    if let Some(client) =
        client.filter(|client| client.epoch() == epoch && client.state() == EngineState::Serving)
    {
        match client.stop_profile(id, reason).await {
            Ok(stopped) => files = stopped,
            Err(error) => failure = Some(format!("the workers did not end it: {error}")),
        }
        if grow && device_storage > 0 {
            grow_back(client, id).await;
        }
    }
    let mut session = service::lock(&replica.profiles.session);
    if let Some(ended) = session.as_mut().filter(|session| session.id == id) {
        ended.state = SessionState::Ended;
        ended.ended = Some(SystemTime::now());
        ended.end_reason = Some(match &failure {
            Some(failure) => format!("{reason}; {failure}"),
            None => reason.to_owned(),
        });
        ended.files = files;
        ended.collectors.clear();
        if let Some(failure) = failure {
            warn!(session = id, %failure, "a profiling session ended without every file");
        }
        info!(session = id, reason, "a profiling session ended");
        crate::status::profile(&replica.recorder, replica.ordinal, ended);
    }
}

/// End the running session, if any, because its epoch stops executing:
/// before a successor's promotion parks it, or as a trial rolls back.
pub(crate) async fn end_for_epoch(
    replica: &crate::leader::Replica,
    client: Option<&EngineClient>,
    reason: &str,
) {
    if let Some(id) = replica.profile_running() {
        end(replica, client, id, reason, false).await;
    }
}

/// What the session's task does after a report of its replica arrived.
enum Next {
    /// The session goes on.
    Running,
    /// The session ends for the reason.
    End(&'static str),
}

/// The session's task: decides as each report of its replica arrives whether its windows
/// ended or its guard acts, ends it once its window of seconds passed, and stops once
/// something else ended it (`ending`).
async fn watch(replica: Arc<Replica>, id: u64, started: Instant, ending: CancellationToken) {
    let mut records = vec![replica.observer.records()];
    let seconds = service::lock(&replica.profiles.session)
        .as_ref()
        .and_then(|session| session.request.seconds)
        .map(|seconds| started + Duration::from_secs_f64(seconds));
    let expiry = async {
        match seconds {
            Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(expiry);
    loop {
        let reason = tokio::select! {
            () = ending.cancelled() => return,
            () = &mut expiry => "seconds",
            () = crate::observe::next_record(&mut records) => match guard(&replica, id, started).await {
                Next::Running => continue,
                Next::End(reason) => reason,
            },
        };
        let client = replica.hold.client();
        end(&replica, client.as_deref(), id, reason, true).await;
        return;
    }
}

/// Reads session `id`'s progress after a report: whether its windows ended, and the guard's
/// decision on its step time.
async fn guard(replica: &Replica, id: u64, started: Instant) -> Next {
    let (request, collectors, overhead) = {
        let mut session = service::lock(&replica.profiles.session);
        let Some(running) = session
            .as_mut()
            .filter(|session| session.id == id && session.state == SessionState::Running)
        else {
            return Next::Running;
        };
        replica.profiles.progress(running, &replica.observer);
        let selected = running
            .workers
            .iter()
            .filter(|worker| worker.selected)
            .count() as u64;
        if running.request.steps.is_some() && running.windows_ended >= selected {
            return Next::End("window");
        }
        (
            running.request.clone(),
            running.collectors.clone(),
            running.overhead,
        )
    };
    let (Some(max), Some(overhead)) = (request.max_slowdown, overhead) else {
        return Next::Running;
    };
    if overhead.slowdown <= max {
        return Next::Running;
    }
    let window = Duration::from_secs_f64(request.guard_window_s);
    let due = service::lock(&replica.profiles.acted).is_none_or(|acted| acted.elapsed() >= window)
        && started.elapsed() >= window;
    if !due {
        return Next::Running;
    }
    *service::lock(&replica.profiles.acted) = Some(Instant::now());
    let costliest = Collector::COSTLIEST_FIRST
        .into_iter()
        .find(|collector| collectors.contains(collector));
    let Some(collector) = costliest else {
        record(replica, id, GuardActionKind::End, None, overhead.slowdown);
        return Next::End("quality of service");
    };
    record(
        replica,
        id,
        GuardActionKind::Release,
        Some(collector),
        overhead.slowdown,
    );
    let reason = format!(
        "the step time rose {:.1}% over the baseline, past the {:.1}% allowed",
        overhead.slowdown * 100.0,
        max * 100.0
    );
    let released = match replica.hold.client() {
        Some(client) => client.release_profile(id, collector, &reason).await,
        None => Err("no engine serves".to_owned()),
    };
    if let Err(error) = released {
        warn!(session = id, %error, "the guard did not turn a collector off");
    }
    Next::Running
}

/// Record a guard action on session `id`, and stop counting `collector`
/// as running.
fn record(
    replica: &crate::leader::Replica,
    id: u64,
    action: GuardActionKind,
    collector: Option<Collector>,
    slowdown: f64,
) {
    let mut session = service::lock(&replica.profiles.session);
    if let Some(running) = session.as_mut().filter(|session| session.id == id) {
        warn!(
            session = id,
            ?action,
            collector = collector.map(Collector::as_str),
            slowdown,
            "the profiling guard acted"
        );
        running.guard.push(GuardAction {
            time: SystemTime::now(),
            action,
            collector,
            slowdown,
        });
        if let Some(collector) = collector {
            running.collectors.retain(|running| *running != collector);
        }
        crate::status::profile(&replica.recorder, replica.ordinal, running);
    }
}
