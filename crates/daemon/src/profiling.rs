//! Profiling sessions of the world that executes (`/admin/profile`,
//! `specs/online-operations.md` section 11).
//!
//! A session starts on every worker of the executing world at the engine's
//! next step boundary (the worker hook `uniserve.profile_start`), records
//! within each worker's window of steps (`docs/profiling.md`) and ends, with
//! every worker's file written (`uniserve.profile_stop`), when the first of
//! these happens:
//!
//! - every selected worker's window of steps ended, as the workers report
//!   with their batch results (`ForwardStats::profile`);
//! - its window of seconds passed;
//! - the operator ends it (`DELETE /admin/profile`);
//! - its world stops executing: a world change promotes a successor, or a
//!   trial rolls back. Neither carries a session into the next world;
//! - the quality-of-service guard ends it.
//!
//! At most one session runs. The guard measures the average step time of
//! the session's world over the [`BASELINE`] before the session started and
//! over the last [`GUARD_WINDOW`] of the session. With `max_slowdown`, a
//! step time past the baseline by more than that fraction turns off the
//! session's costliest collector still running (`uniserve.profile_release`,
//! in `Collector::COSTLIEST_FIRST` order), at most once a [`GUARD_WINDOW`] so
//! that each change is measured; a slowdown that remains with no collector
//! left ends the session. Every action is in the session's status.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use proto::profile::{Collector, SessionRequest, SessionStart};
use proto::request::wire;
use serde::Serialize;
use tracing::{info, warn};

use crate::engine::client::{EngineClient, EngineState};
use crate::leader::{Leader, Replica, UpdateError, locked};
use crate::worlds::Rates;

/// The span before a session's start its baseline step time averages.
pub const BASELINE: Duration = Duration::from_secs(60);

/// The span of a session the guard averages its step time over, and the
/// least time between two of its actions.
pub const GUARD_WINDOW: Duration = Duration::from_secs(10);

/// How often the session's task checks its window and its guard.
const POLL: Duration = Duration::from_millis(500);

/// How long ending a session waits for the workers to write their files.
const STOP_TIMEOUT: Duration = Duration::from_secs(120);

/// One action of the quality-of-service guard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GuardAction {
    /// Seconds since the Unix epoch.
    pub time: f64,
    /// `release` (a collector was turned off) or `end`.
    pub action: String,
    pub collector: Option<Collector>,
    /// The slowdown that caused it: the session's step time over the
    /// baseline's, less one.
    pub slowdown: f64,
}

/// What the guard measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Overhead {
    /// The world's average step before the session, microseconds.
    pub baseline_step_us: f64,
    /// The session's average step over its last [`GUARD_WINDOW`].
    pub step_us: f64,
    /// `step_us / baseline_step_us - 1`.
    pub slowdown: f64,
}

/// A session, running or the last that ended.
#[derive(Debug, Clone, Serialize)]
pub struct Session {
    pub id: u64,
    pub world: u64,
    /// `running` or `ended`.
    pub state: String,
    /// Seconds since the Unix epoch.
    pub started: f64,
    pub ended: Option<f64>,
    /// Why it ended: `window`, `seconds`, `operator`, `world change`,
    /// `rollback`, `quality of service`, or a failure.
    pub end_reason: Option<String>,
    pub request: SessionRequest,
    /// Each worker's answer to the start: its rank, whether the session
    /// selects it, its file and its collectors' states.
    pub workers: Vec<serde_json::Value>,
    /// Each worker's answer to the end: what its file holds.
    pub files: Vec<serde_json::Value>,
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
    /// The device storage the session holds on each selected worker's GPU
    /// (`SessionRequest::device_storage`), which the workers book in the
    /// GPUs' storage ledgers: the KV cache gave back tail segments for it
    /// where a ledger lacked room, and grows back once the session ends.
    pub device_storage: u64,
}

/// The deployment's sessions.
#[derive(Default)]
pub(crate) struct Profiles {
    session: std::sync::Mutex<Option<Session>>,
    next: std::sync::atomic::AtomicU64,
    /// When the guard last acted, for the running session.
    acted: std::sync::Mutex<Option<Instant>>,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// The JSON answers of every worker.
fn answers(results: &[String]) -> Vec<serde_json::Value> {
    results
        .iter()
        .map(|result| serde_json::from_str(result).unwrap_or(serde_json::Value::Null))
        .collect()
}

impl Leader {
    pub async fn start_profile(
        self: &Arc<Self>,
        request: SessionRequest,
    ) -> Result<serde_json::Value, UpdateError> {
        let mut sessions = Vec::new();
        for replica in self.replicas() {
            sessions.push(replica.start_profile(request.clone()).await?);
        }
        let mut report = sessions[0].clone();
        report["replicas"] = sessions.into();
        Ok(report)
    }

    pub fn profile_status(&self) -> Option<serde_json::Value> {
        let sessions: Vec<_> = self
            .replicas()
            .iter()
            .filter_map(|replica| replica.profile_status())
            .collect();
        let mut report = serde_json::to_value(sessions.first()?).ok()?;
        report["replicas"] = serde_json::to_value(sessions).ok()?;
        Some(report)
    }

    pub async fn stop_profile(&self, reason: &str) -> Result<serde_json::Value, UpdateError> {
        let mut reports = Vec::new();
        for replica in self.replicas() {
            if replica
                .profile_status()
                .is_some_and(|session| session.state == "running")
            {
                reports.push(replica.stop_profile(reason).await?);
            }
        }
        if reports.is_empty() {
            return Err(UpdateError::Refused("no profiling session runs".to_owned()));
        }
        let mut report = reports[0].clone();
        report["replicas"] = reports.into();
        Ok(report)
    }
}

impl Replica {
    /// Start a session of `request` on the executing world.
    ///
    /// # Errors
    ///
    /// Refuses an invalid request and a session while one runs; fails when
    /// no engine serves or a worker cannot start it.
    pub(crate) async fn start_profile(
        self: &Arc<Self>,
        request: SessionRequest,
    ) -> Result<serde_json::Value, UpdateError> {
        request.validate().map_err(UpdateError::Refused)?;
        // Ranks that keep PC sampling enabled for their graphs refuse the
        // collectors CUPTI cannot run beside it.
        let graph_pc_sampling = locked(&self.served)
            .as_ref()
            .and_then(|spec| spec.profiling.as_ref())
            .is_some_and(|profiling| profiling.graph_pc_sampling);
        if graph_pc_sampling && let Some(reason) = request.refused_beside_graph_pc_sampling() {
            return Err(UpdateError::Refused(reason));
        }
        let profiles = &self.profiles;
        if let Some(running) = locked(&profiles.session)
            .as_ref()
            .filter(|session| session.state == "running")
        {
            return Err(UpdateError::Refused(format!(
                "profiling session {} runs: end it first (DELETE /admin/profile)",
                running.id
            )));
        }
        let client = self.hold().client().ok_or(UpdateError::NoEngine)?;
        let world = client.world();
        let id = profiles
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let started_at = Instant::now();
        let baseline = self.observer.rates(
            world,
            started_at.checked_sub(BASELINE).unwrap_or(started_at),
            BASELINE,
        );
        let payload = serde_json::to_string(&SessionStart {
            session: id,
            world,
            request: request.clone(),
        })
        .map_err(|error| UpdateError::Failed(error.into()))?;
        // The device storage the collectors hold on each selected worker's
        // GPU must fit its storage ledger: where it lacks room, the KV cache
        // gives back tail segments first, as a world change's preparation
        // does, the requests holding them running on.
        let device_storage = request.device_storage();
        if device_storage > 0 {
            client.reserve_storage(device_storage, &Default::default()).await.map_err(|error| {
                UpdateError::Refused(format!(
                    "the session holds {device_storage} bytes of device storage on each selected \
                     worker's GPU (a range_metrics pass), for which the KV cache cannot make \
                     room: {error}"
                ))
            })?;
        }
        // A worker that cannot start the session, or whose GPU's storage
        // ledger cannot fit the device storage its collectors would hold,
        // leaves the session to the workers that started it, which end it.
        let results = match self.hook("uniserve.profile_start", &payload).await {
            Ok(results) => results,
            Err(error) => {
                self.abandon_profile(id, device_storage, &client).await;
                return Err(error);
            }
        };
        let workers = answers(&results);
        if let Some(refusal) = workers
            .iter()
            .find_map(|worker| worker["refused"].as_str().map(str::to_owned))
        {
            self.abandon_profile(id, device_storage, &client).await;
            return Err(UpdateError::Refused(refusal));
        }
        // The collectors some selected worker runs.
        let collectors: BTreeSet<Collector> = workers
            .iter()
            .filter(|worker| worker["selected"] == true)
            .flat_map(|worker| {
                worker["collectors"]
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter(|(_, state)| *state == "prepared")
                    .filter_map(|(name, _)| {
                        serde_json::from_value(serde_json::Value::String(name.clone())).ok()
                    })
            })
            .collect();
        let session = Session {
            id,
            world,
            state: "running".to_owned(),
            started: now(),
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
        let answer = serde_json::to_value(&session).unwrap_or_default();
        *locked(&profiles.session) = Some(session);
        *locked(&profiles.acted) = None;
        info!(session = id, world, "a profiling session started");
        tokio::spawn(watch(Arc::clone(self), id, started_at));
        Ok(answer)
    }

    /// The running session, or the last that ended; `None` before any.
    pub(crate) fn profile_status(&self) -> Option<Session> {
        let mut session = locked(&self.profiles.session).clone()?;
        if session.state == "running" {
            self.profiles.progress(&mut session, &self.observer);
        }
        Some(session)
    }

    /// End the running session for `reason`, every worker writing its file.
    ///
    /// # Errors
    ///
    /// Refuses when no session runs.
    pub(crate) async fn stop_profile(
        &self,
        reason: &str,
    ) -> Result<serde_json::Value, UpdateError> {
        let id = locked(&self.profiles.session)
            .as_ref()
            .filter(|session| session.state == "running")
            .map(|session| session.id)
            .ok_or_else(|| UpdateError::Refused("no profiling session runs".to_owned()))?;
        let client = self.hold().client();
        end(self, client.as_deref(), id, reason, true).await;
        Ok(serde_json::to_value(self.profile_status()).unwrap_or_default())
    }

    /// End session `id` on the workers that started it after another
    /// worker refused or failed to: the session never ran, and each worker
    /// that started it closes its file and releases its collectors and
    /// their booked storage. The KV cache grows back into the room it made
    /// for the session's `device_storage`.
    async fn abandon_profile(&self, id: u64, device_storage: u64, client: &EngineClient) {
        let payload = serde_json::json!({ "session": id, "reason": "not started" }).to_string();
        if let Err(error) = self.hook("uniserve.profile_stop", &payload).await {
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
    fn progress(&self, session: &mut Session, observer: &crate::worlds::Observer) {
        let (steps, bytes, ended) = observer.profile(session.id);
        session.steps = steps;
        session.bytes_written = bytes;
        session.windows_ended = ended;
        session.overhead = session.baseline.and_then(|baseline| {
            let since = Instant::now().checked_sub(GUARD_WINDOW)?;
            let current = observer.rates(session.world, since, GUARD_WINDOW)?;
            (baseline.step_us > 0.0).then(|| Overhead {
                baseline_step_us: baseline.step_us,
                step_us: current.step_us,
                slowdown: current.step_us / baseline.step_us - 1.0,
            })
        });
    }
}

/// End session `id` for `reason` on the workers of `client`, when it still
/// runs: every worker writes its file. A session whose world no longer
/// executes on `client` ends without its workers' answers, which wrote what
/// they recorded as their world closed or keep it in their files' logs.
/// Grow the engine's KV cache back into the room session `id` made for its
/// device storage, once the workers released it.
async fn grow_back(client: &EngineClient, id: u64) {
    match client.resize_cache(wire::CacheChange::Grow).await {
        Ok(_) => info!(
            session = id,
            "the KV cache grew back after the profiling session"
        ),
        Err(error) => warn!(
            session = id,
            "growing the KV cache back after the profiling session failed: {error}"
        ),
    }
}

/// End session `id` for `reason`: the workers of a world that still serves
/// write their files, and, with `grow`, the KV cache grows back into the
/// room the session made for its device storage; a world that stops
/// executing (a world change's promotion, a rollback) keeps its cache as
/// the change left it.
pub(crate) async fn end(
    replica: &crate::leader::Replica,
    client: Option<&EngineClient>,
    id: u64,
    reason: &str,
    grow: bool,
) {
    let (world, device_storage) = {
        let mut session = locked(&replica.profiles.session);
        let Some(running) = session
            .as_mut()
            .filter(|session| session.id == id && session.state == "running")
        else {
            return;
        };
        replica.profiles.progress(running, &replica.observer);
        // Ending from here on: a second end finds it ended.
        running.state = "ending".to_owned();
        (running.world, running.device_storage)
    };
    let mut files = Vec::new();
    let mut failure = None;
    // Only a world that still serves writes its files: a failed one's
    // workers keep what they committed (the files' write-ahead logs).
    if let Some(client) =
        client.filter(|client| client.world() == world && client.state() == EngineState::Ready)
    {
        let payload = serde_json::json!({ "session": id, "reason": reason }).to_string();
        match tokio::time::timeout(STOP_TIMEOUT, client.hook("uniserve.profile_stop", &payload))
            .await
        {
            Ok(Ok(results)) => files = answers(&results),
            Ok(Err(error)) => failure = Some(format!("the workers did not end it: {error}")),
            Err(_) => {
                failure = Some(format!(
                    "the workers did not end it within {STOP_TIMEOUT:?}"
                ))
            }
        }
        if grow && device_storage > 0 {
            grow_back(client, id).await;
        }
    }
    let mut session = locked(&replica.profiles.session);
    if let Some(ended) = session.as_mut().filter(|session| session.id == id) {
        ended.state = "ended".to_owned();
        ended.ended = Some(now());
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
    }
}

/// End the running session, if any, because its world stops executing:
/// before a successor's promotion parks it, or as a trial rolls back.
pub(crate) async fn end_for_world(
    replica: &crate::leader::Replica,
    client: Option<&EngineClient>,
    reason: &str,
) {
    let running = locked(&replica.profiles.session)
        .as_ref()
        .filter(|session| session.state == "running")
        .map(|session| session.id);
    if let Some(id) = running {
        end(replica, client, id, reason, false).await;
    }
}

/// The session's task: ends it when its windows end, its seconds pass or
/// its guard ends it, and runs the guard.
async fn watch(replica: Arc<Replica>, id: u64, started: Instant) {
    loop {
        tokio::time::sleep(POLL).await;
        let (request, collectors, overhead) = {
            let mut session = locked(&replica.profiles.session);
            let Some(running) = session
                .as_mut()
                .filter(|session| session.id == id && session.state == "running")
            else {
                return;
            };
            replica.profiles.progress(running, &replica.observer);
            let selected = running
                .workers
                .iter()
                .filter(|worker| worker["selected"] == true)
                .count() as u64;
            let ended = running.request.steps.is_some() && running.windows_ended >= selected;
            if ended {
                (None, Vec::new(), None)
            } else {
                (
                    Some(running.request.clone()),
                    running.collectors.clone(),
                    running.overhead,
                )
            }
        };
        let client = replica.hold.client();
        let Some(request) = request else {
            end(&replica, client.as_deref(), id, "window", true).await;
            return;
        };
        if request
            .seconds
            .is_some_and(|seconds| started.elapsed().as_secs_f64() >= seconds)
        {
            end(&replica, client.as_deref(), id, "seconds", true).await;
            return;
        }
        let (Some(max), Some(overhead)) = (request.max_slowdown, overhead) else {
            continue;
        };
        if overhead.slowdown <= max {
            continue;
        }
        let due = locked(&replica.profiles.acted)
            .is_none_or(|acted| acted.elapsed() >= GUARD_WINDOW)
            && started.elapsed() >= GUARD_WINDOW;
        if !due {
            continue;
        }
        *locked(&replica.profiles.acted) = Some(Instant::now());
        let costliest = Collector::COSTLIEST_FIRST
            .into_iter()
            .find(|collector| collectors.contains(collector));
        let Some(collector) = costliest else {
            record(&replica, id, "end", None, overhead.slowdown);
            end(&replica, client.as_deref(), id, "quality of service", true).await;
            return;
        };
        record(&replica, id, "release", Some(collector), overhead.slowdown);
        let payload = serde_json::json!({
            "session": id,
            "collector": collector,
            "reason": format!(
                "the step time rose {:.1}% over the baseline, past the {:.1}% allowed",
                overhead.slowdown * 100.0,
                max * 100.0
            ),
        })
        .to_string();
        let released = match &client {
            Some(client) => client.hook("uniserve.profile_release", &payload).await,
            None => Err("no engine serves".to_owned()),
        };
        if let Err(error) = released {
            warn!(session = id, %error, "the guard did not turn a collector off");
        }
    }
}

/// Record a guard action on session `id`, and stop counting `collector`
/// as running.
fn record(
    replica: &crate::leader::Replica,
    id: u64,
    action: &str,
    collector: Option<Collector>,
    slowdown: f64,
) {
    let mut session = locked(&replica.profiles.session);
    if let Some(running) = session.as_mut().filter(|session| session.id == id) {
        warn!(
            session = id,
            action,
            collector = collector.map(Collector::as_str),
            slowdown,
            "the profiling guard acted"
        );
        running.guard.push(GuardAction {
            time: now(),
            action: action.to_owned(),
            collector,
            slowdown,
        });
        if let Some(collector) = collector {
            running.collectors.retain(|running| *running != collector);
        }
    }
}
