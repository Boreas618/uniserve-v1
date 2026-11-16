//! A point's load: warm-up, then its measured requests at their arrivals, or a
//! timeline load whose changes act on the deployment while it runs.
//!
//! A load submits rows through a [`Submit`], which sends one row's request and
//! returns its closed record; `scheduled` is the arrival time on the run's
//! clock for a measured request and `None` for warm-up. Arrivals at a finite
//! rate are Poisson, their gaps drawn from NumPy's legacy generator seeded with
//! the load's seed, as the protocols this harness measures fixed them
//! ([`crate::pyrandom`]).

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::{Map, Value, json};
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinHandle;

use crate::pyrandom::Random;
use crate::record::{Clock, Example, Record};
use crate::spec::{Change, Load, SIGNALS};

/// Sends one row's request; `scheduled` is its arrival, `None` for warm-up.
pub type Submit =
    Arc<dyn Fn(Example, Option<f64>) -> Pin<Box<dyn Future<Output = Record> + Send>> + Send + Sync>;

/// A load's records and its measured window: wall-clock seconds bounding the
/// window (for aligning it with telemetry on that clock), and its duration on
/// the run's clock.
#[derive(Debug, Default)]
pub struct Outcome {
    pub warmup: Vec<Record>,
    pub measured: Vec<Record>,
    pub duration_s: f64,
    pub window_start: f64,
    pub window_end: f64,
}

/// A warm-up request failed: nothing was measured.
#[derive(Debug)]
pub struct WarmupFailure {
    pub records: Vec<Record>,
}

impl std::fmt::Display for WarmupFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let first = self.records.iter().find(|record| !record.success);
        write!(
            formatter,
            "Warmup failed -- check the benchmark arguments and server. First classifier: {:?}; status: {:?}; error: {:?}",
            first.map(|record| &record.classifier),
            first.and_then(|record| record.status_code),
            first.and_then(|record| record.error.as_ref()),
        )
    }
}

impl std::error::Error for WarmupFailure {}

/// `submit` within an in-flight limit; the arrival time is taken before the
/// limit, so time queued there is the gap between `scheduled` and the request's
/// start.
fn limited(submit: &Submit, max_concurrency: Option<usize>) -> Submit {
    let Some(limit) = max_concurrency.filter(|limit| *limit > 0) else {
        return Arc::clone(submit);
    };
    let semaphore = Arc::new(Semaphore::new(limit));
    let submit = Arc::clone(submit);
    Arc::new(move |row: Example, scheduled: Option<f64>| {
        let semaphore = Arc::clone(&semaphore);
        let submit = Arc::clone(&submit);
        Box::pin(async move {
            let _slot = semaphore.acquire_owned().await;
            submit(row, scheduled).await
        })
    })
}

/// The load's warm-up, sent together and checked: the rows of an explicit
/// corpus, else the first row `requests` times. `inspect` sees the records
/// before their success is checked.
async fn warm_up(
    rows: &[Example],
    submit: &Submit,
    requests: usize,
    corpus: Option<&[Example]>,
    inspect: &(dyn Fn(&mut [Record]) + Sync),
) -> Result<Vec<Record>> {
    let excluded: Vec<Example> = match corpus {
        Some(corpus) => corpus.to_vec(),
        None => std::iter::repeat_n(rows[0].clone(), requests).collect(),
    };
    if excluded.is_empty() {
        return Ok(Vec::new());
    }
    let tasks: Vec<JoinHandle<Record>> = excluded
        .into_iter()
        .map(|row| tokio::spawn(submit(row, None)))
        .collect();
    let mut records = Vec::new();
    for task in tasks {
        records.push(task.await?);
    }
    inspect(&mut records);
    if !records.iter().all(|record| record.success) {
        return Err(WarmupFailure { records }.into());
    }
    Ok(records)
}

/// The window the measured records span: from the first start to the last
/// close.
fn window(warmup: Vec<Record>, measured: Vec<Record>, clock_offset: f64) -> Result<Outcome> {
    if measured.is_empty() {
        return Ok(Outcome {
            warmup,
            ..Outcome::default()
        });
    }
    let mut last = f64::MIN;
    for record in &measured {
        last = last.max(
            record
                .final_event_time
                .context("request dispatch returned an unclosed record")?,
        );
    }
    let first = measured
        .iter()
        .map(|record| record.start_time)
        .fold(f64::MAX, f64::min);
    Ok(Outcome {
        warmup,
        measured,
        duration_s: last - first,
        window_start: clock_offset + first,
        window_end: clock_offset + last,
    })
}

/// The wall clock in seconds since the Unix epoch.
pub fn wall() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or_default()
}

/// What a load is given beside its rows.
pub struct Driver<'a> {
    pub load: &'a Load,
    pub submit: Submit,
    pub clock: Clock,
    /// The generator the arrival gaps are drawn from, seeded with the load's seed.
    pub arrivals: Random,
    pub warmup_requests: usize,
    pub warmup_rows: Option<Vec<Example>>,
    pub inspect: Inspect,
}

/// What a load lets see its warm-up records before their success is checked.
pub type Inspect = Box<dyn Fn(&mut [Record]) + Send + Sync>;

/// Runs the warm-up and the measured requests: every row once at its arrival,
/// within the concurrency limit, then waits for all of them. When every row
/// names a session, the sessions run at once, each sending its next row once its
/// previous one answered (a trace measures each session's cold first request,
/// so it sends no warm-up); a trace needs an unbounded rate.
///
/// # Errors
///
/// [`WarmupFailure`] when a warm-up request failed; a session trace at a finite
/// rate or with a row without a session.
pub async fn run(
    rows: &[Example],
    mut context: Driver<'_>,
    before_measure: impl Future<Output = ()>,
) -> Result<Outcome> {
    if rows.is_empty() {
        return Ok(Outcome::default());
    }
    let mut sessions: BTreeMap<String, Vec<(usize, Example)>> = BTreeMap::new();
    let traced = rows.iter().any(|row| row.session_id.is_some());
    if traced {
        ensure!(
            context.load.request_rate.is_infinite(),
            "closed-loop sessions require infinite request_rate"
        );
        for (index, row) in rows.iter().enumerate() {
            let session = row.session_id.clone().filter(|session| !session.is_empty());
            let session = session.context("closed-loop sessions require every session_id")?;
            sessions
                .entry(session)
                .or_default()
                .push((index, row.clone()));
        }
    }
    let limit = limited(&context.submit, context.load.max_concurrency);
    let warmup = warm_up(
        rows,
        &limit,
        if traced { 0 } else { context.warmup_requests },
        context.warmup_rows.as_deref(),
        &*context.inspect,
    )
    .await?;
    before_measure.await;
    let offset = wall() - context.clock.now();
    let clock = context.clock;
    let measured = if traced {
        let tasks: Vec<JoinHandle<Vec<(usize, Record)>>> = sessions
            .into_values()
            .map(|sequence| {
                let submit = Arc::clone(&context.submit);
                tokio::spawn(async move {
                    let mut completed = Vec::new();
                    for (index, row) in sequence {
                        // Arrival follows this session's previous answer.
                        completed.push((index, submit(row, Some(clock.now())).await));
                    }
                    completed
                })
            })
            .collect();
        let mut ordered = Vec::new();
        for task in tasks {
            ordered.extend(task.await?);
        }
        ordered.sort_by_key(|(index, _)| *index);
        ordered.into_iter().map(|(_, record)| record).collect()
    } else {
        let mut tasks = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            // A gap separates consecutive arrivals; none follows the last.
            if index > 0 && context.load.request_rate.is_finite() {
                let gap = context
                    .arrivals
                    .exponential(1.0 / context.load.request_rate);
                tokio::time::sleep(Duration::from_secs_f64(gap)).await;
            }
            tasks.push(tokio::spawn(limit(row.clone(), Some(clock.now()))));
        }
        let mut records = Vec::with_capacity(tasks.len());
        for task in tasks {
            records.push(task.await?);
        }
        records
    };
    window(warmup, measured, offset)
}

/// The phases in which an update has ended: a change waiting for another phase
/// of it waits no longer.
const ENDED_PHASES: [&str; 3] = ["committed", "rolled_back", "abandoned"];
/// Seconds one read of the event log waits for a new event, and between reads
/// while the listener does not answer.
const EVENT_WAIT_S: f64 = 10.0;
const EVENT_RETRY_S: f64 = 1.0;

/// What a timeline load acts on: the deployment's admin listener, its
/// processes, and the environment and directory a change's command runs in.
pub struct Deployment {
    pub admin_socket: Option<PathBuf>,
    pub pids: Arc<dyn Fn() -> Vec<u32> + Send + Sync>,
    pub working_directory: PathBuf,
    pub environment: BTreeMap<String, String>,
}

/// The timeline record a load fills as it proceeds (so a failed load leaves
/// what it observed): `load_start`, the run's clock at t = 0; `clock_offset`,
/// the wall clock less it; `issue_end_s`; per change its declaration,
/// `issued_s`, `ended_s` and what answered it; and every admin event with
/// `received_s` and, from its own time, `at_s`. Times ending in `_s` are
/// seconds from t = 0.
pub type Timeline = Arc<Mutex<Map<String, Value>>>;

#[expect(
    clippy::expect_used,
    reason = "a poisoned lock means a panic elsewhere, which ends the run"
)]
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().expect("the timeline's lock")
}

/// Runs the warm-up, then the timeline load and its changes.
///
/// Requests are issued from t = 0 until `duration_s`, on while a change has not
/// ended, and for `settle_s` after the last one ended; request `n` takes row
/// `n mod rows`. At an unbounded rate the load is a closed loop of
/// `max_concurrency` streams, each issuing its next request when its last one
/// ended; at a finite rate, arrivals are Poisson within the concurrency limit.
/// `cancelled` makes the record of a request a cancel change disconnected, from
/// its row and its submission time.
///
/// # Errors
///
/// [`WarmupFailure`], or the failure of a submission, after every request,
/// change and the event reader have been stopped.
pub async fn timeline(
    rows: &[Example],
    mut context: Driver<'_>,
    changes: &[Change],
    deployment: Option<Arc<Deployment>>,
    record: Timeline,
    cancelled: Arc<dyn Fn(Example, f64) -> Record + Send + Sync>,
    before_measure: impl Future<Output = ()>,
) -> Result<Outcome> {
    let load = context.load.clone();
    let duration = load
        .duration_s
        .context("a timeline load states load.duration_s")?;
    let limit = limited(&context.submit, load.max_concurrency);
    let warmup = warm_up(
        rows,
        &limit,
        context.warmup_requests,
        context.warmup_rows.as_deref(),
        &*context.inspect,
    )
    .await?;
    before_measure.await;

    // Before t = 0, so that the search is no part of the measured load.
    let targets: BTreeMap<String, Option<(u32, String)>> = changes
        .iter()
        .filter(|change| change.signal.is_some() && change.host.is_none())
        .map(|change| {
            let target = deployment
                .as_deref()
                .and_then(|deployment| signal_target(change, deployment).ok());
            (change.name.clone(), target)
        })
        .collect();
    let clock = context.clock;
    let start = clock.now();
    let offset = wall() - start;
    let since = move || clock.now() - start;
    {
        let mut timeline = lock(&record);
        timeline.insert("load_start".into(), start.into());
        timeline.insert("clock_offset".into(), offset.into());
        // A change's entry starts as the fields its declaration states.
        let entries: Vec<Value> = changes
            .iter()
            .map(|change| {
                let declared = serde_json::to_value(change).unwrap_or(Value::Null);
                let mut entry = Map::new();
                for (key, value) in declared.as_object().into_iter().flatten() {
                    let empty = value.is_null() || value.as_array().is_some_and(Vec::is_empty);
                    if !empty {
                        entry.insert(key.clone(), value.clone());
                    }
                }
                Value::Object(entry)
            })
            .collect();
        timeline.insert("changes".into(), Value::Array(entries));
        timeline.insert("events".into(), Value::Array(Vec::new()));
    }

    let state = Arc::new(State {
        pending: Mutex::new(changes.len()),
        last_end: Mutex::new(0.0),
        finished: changes
            .iter()
            .map(|change| (change.name.clone(), Arc::new(Notify::new())))
            .collect(),
        done: Mutex::new(BTreeSet::new()),
        arrived: Notify::new(),
        flight: Mutex::new(BTreeMap::new()),
        disconnected: Mutex::new(BTreeSet::new()),
        aborted: Mutex::new(BTreeSet::new()),
    });
    let issuing = {
        let state = Arc::clone(&state);
        let settle = load.settle_s;
        move || {
            let now = since();
            now < duration || *lock(&state.pending) > 0 || now < *lock(&state.last_end) + settle
        }
    };

    let mut background: Vec<JoinHandle<()>> = Vec::new();
    for (index, change) in changes.iter().enumerate() {
        let change = change.clone();
        let state = Arc::clone(&state);
        let record = Arc::clone(&record);
        let deployment = deployment.clone();
        let target = targets.get(&change.name).cloned().flatten();
        let timeout = load.request_timeout_s;
        let names: Vec<String> = changes.iter().map(|change| change.name.clone()).collect();
        background.push(tokio::spawn(async move {
            let outcome = apply(
                &change,
                &names,
                &state,
                &record,
                deployment.as_deref(),
                target,
                timeout,
                since,
            )
            .await;
            let ended = since();
            {
                let mut timeline = lock(&record);
                if let Some(entry) = entry(&mut timeline, index) {
                    match outcome {
                        Ok(answer) => entry.extend(answer),
                        Err(error) => {
                            entry.insert("error".into(), format!("{error:#}").into());
                        }
                    }
                    entry.insert("ended_s".into(), ended.into());
                }
            }
            *lock(&state.pending) -= 1;
            {
                let mut last = lock(&state.last_end);
                *last = last.max(ended);
            }
            lock(&state.done).insert(change.name.clone());
            state.finished[&change.name].notify_waiters();
        }));
    }
    if let Some(listener) = deployment
        .as_ref()
        .and_then(|deployment| deployment.admin_socket.clone())
    {
        let record = Arc::clone(&record);
        let state = Arc::clone(&state);
        background.push(tokio::spawn(async move {
            read_events(&listener, &record, start, offset, &state, clock).await;
        }));
    }

    let outputs: Arc<Mutex<BTreeMap<usize, Record>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let rows: Arc<Vec<Example>> = Arc::new(rows.to_vec());
    let mut requests: Vec<JoinHandle<()>> = Vec::new();
    let result: Result<()> = async {
        if load.request_rate.is_infinite() {
            let streams = load.max_concurrency.context("a closed loop's streams")?;
            for _ in 0..streams {
                let (submit, rows, outputs, counter, state, cancelled) = (
                    Arc::clone(&context.submit),
                    Arc::clone(&rows),
                    Arc::clone(&outputs),
                    Arc::clone(&counter),
                    Arc::clone(&state),
                    Arc::clone(&cancelled),
                );
                let issuing = issuing.clone();
                requests.push(tokio::spawn(async move {
                    while issuing() {
                        let index = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        let row = rows[index % rows.len()].clone();
                        let submitted = clock.now();
                        let request = tokio::spawn(submit(row.clone(), Some(submitted)));
                        lock(&state.flight).insert(index, request.abort_handle());
                        let output = match request.await {
                            Ok(record) => record,
                            // A request a cancel change disconnected ends with
                            // its cancelled record, and the stream goes on.
                            Err(error)
                                if error.is_cancelled()
                                    && lock(&state.disconnected).contains(&index) =>
                            {
                                lock(&state.aborted).insert(index);
                                cancelled(row, submitted)
                            }
                            Err(_) => {
                                lock(&state.flight).remove(&index);
                                return;
                            }
                        };
                        lock(&state.flight).remove(&index);
                        lock(&outputs).insert(index, output);
                    }
                }));
            }
        } else {
            while issuing() {
                let index = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let row = rows[index % rows.len()].clone();
                let limit = Arc::clone(&limit);
                let outputs = Arc::clone(&outputs);
                requests.push(tokio::spawn(async move {
                    let output = limit(row, Some(clock.now())).await;
                    lock(&outputs).insert(index, output);
                }));
                let gap = context.arrivals.exponential(1.0 / load.request_rate);
                tokio::time::sleep(Duration::from_secs_f64(gap)).await;
            }
        }
        for request in requests.drain(..) {
            request.await?;
        }
        Ok(())
    }
    .await;
    // Every change has ended once issuing stopped; the reader runs until the
    // last stream ended, and is stopped here on every path.
    for task in requests.iter().chain(&background) {
        task.abort();
    }
    for task in background {
        let _ = task.await;
    }
    result?;
    let last_end = *lock(&state.last_end);
    lock(&record).insert(
        "issue_end_s".into(),
        duration.max(last_end + load.settle_s).into(),
    );
    let measured: Vec<Record> = std::mem::take(&mut *lock(&outputs)).into_values().collect();
    window(warmup, measured, offset)
}

/// What the timeline's tasks share.
struct State {
    pending: Mutex<usize>,
    last_end: Mutex<f64>,
    finished: BTreeMap<String, Arc<Notify>>,
    done: Mutex<BTreeSet<String>>,
    /// Signalled when admin events arrive.
    arrived: Notify,
    /// A closed loop's requests in flight, by issue index.
    flight: Mutex<BTreeMap<usize, tokio::task::AbortHandle>>,
    /// The issue indices a cancel change disconnected.
    disconnected: Mutex<BTreeSet<usize>>,
    /// The disconnected indices whose requests the cancellation ended (a request
    /// may end on its own before its abort lands).
    aborted: Mutex<BTreeSet<usize>>,
}

fn entry(timeline: &mut Map<String, Value>, index: usize) -> Option<&mut Map<String, Value>> {
    timeline
        .get_mut("changes")
        .and_then(Value::as_array_mut)
        .and_then(|changes| changes.get_mut(index))
        .and_then(Value::as_object_mut)
}

/// Waits for a change's time, acts, and returns what answered it.
#[expect(
    clippy::too_many_arguments,
    reason = "the change's context, each one's own"
)]
async fn apply(
    change: &Change,
    names: &[String],
    state: &State,
    record: &Timeline,
    deployment: Option<&Deployment>,
    target: Option<(u32, String)>,
    timeout_s: f64,
    since: impl Fn() -> f64,
) -> Result<Map<String, Value>> {
    let at = match (&change.at_s, &change.after) {
        (Some(at), _) => *at,
        (None, Some(after)) => {
            let notified = state.finished[after].notified();
            if !lock(&state.done).contains(after) {
                notified.await;
            }
            let position = names
                .iter()
                .position(|name| name == after)
                .context("an earlier change")?;
            let reference = entry(&mut lock(record), position)
                .cloned()
                .unwrap_or_default();
            ensure!(!reference.contains_key("error"), "change {after} failed");
            match &change.phase {
                None => {
                    reference
                        .get("ended_s")
                        .and_then(Value::as_f64)
                        .unwrap_or_default()
                        + change.delay_s
                }
                Some(phase) => {
                    let issued = reference
                        .get("issued_s")
                        .and_then(Value::as_f64)
                        .unwrap_or_default();
                    // The event may never come; the wait is bounded as a request.
                    let reached = tokio::time::timeout(
                        Duration::from_secs_f64(timeout_s),
                        phase_reached(phase, issued, state, record, deployment),
                    )
                    .await
                    .map_err(|_| {
                        anyhow!("no event reported the phase {phase} within {timeout_s}s")
                    })??;
                    reached + change.delay_s
                }
            }
        }
        (None, None) => bail!("a change states at_s or after"),
    };
    tokio::time::sleep(Duration::from_secs_f64((at - since()).max(0.0))).await;
    let issued = since();
    if let Some(index) = names.iter().position(|name| name == &change.name)
        && let Some(entry) = entry(&mut lock(record), index)
    {
        entry.insert("issued_s".into(), issued.into());
    }
    if let Some(count) = change.cancel {
        return disconnect(count, &change.cancel_order, state).await;
    }
    act(change, deployment, timeout_s, target).await
}

/// The time of the first event from `after_s` on that reports `phase`.
async fn phase_reached(
    phase: &str,
    after_s: f64,
    state: &State,
    record: &Timeline,
    deployment: Option<&Deployment>,
) -> Result<f64> {
    ensure!(
        deployment.is_some_and(|deployment| deployment.admin_socket.is_some()),
        "no admin listener reports the update's phases"
    );
    loop {
        let arrived = state.arrived.notified();
        {
            let timeline = lock(record);
            for event in timeline
                .get("events")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let at = event
                    .get("at_s")
                    .and_then(Value::as_f64)
                    .unwrap_or(f64::NEG_INFINITY);
                if at < after_s {
                    continue;
                }
                let reached = event
                    .get("data")
                    .and_then(|data| data.get("phase"))
                    .and_then(Value::as_str);
                if reached == Some(phase) {
                    return Ok(at);
                }
                if let Some(reached) = reached.filter(|reached| ENDED_PHASES.contains(reached)) {
                    bail!("the update ended {reached} before {phase}");
                }
            }
        }
        arrived.await;
    }
}

/// Cancels `count` requests in flight at once, by issue order: aborting a
/// request's task closes its connection mid-stream, as a client that goes away
/// does.
async fn disconnect(count: usize, order: &str, state: &State) -> Result<Map<String, Value>> {
    let chosen: Vec<usize> = {
        let flight = lock(&state.flight);
        let live: Vec<usize> = flight
            .iter()
            .filter(|(_, task)| !task.is_finished())
            .map(|(index, _)| *index)
            .collect();
        ensure!(
            live.len() >= count,
            "{} streams are in flight, fewer than {count}",
            live.len()
        );
        if order == "oldest" {
            live[..count].to_vec()
        } else {
            live[live.len() - count..].to_vec()
        }
    };
    lock(&state.disconnected).extend(chosen.iter().copied());
    let handles: Vec<(usize, tokio::task::AbortHandle)> = {
        let flight = lock(&state.flight);
        chosen
            .iter()
            .filter_map(|index| Some((*index, flight.get(index)?.clone())))
            .collect()
    };
    for (_, handle) in &handles {
        handle.abort();
    }
    // The change ends once every one of them ended and its stream took its end.
    while handles.iter().any(|(_, handle)| !handle.is_finished())
        || chosen
            .iter()
            .any(|index| lock(&state.flight).contains_key(index))
    {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let ended: Vec<usize> = {
        let aborted = lock(&state.aborted);
        chosen
            .iter()
            .copied()
            .filter(|index| aborted.contains(index))
            .collect()
    };
    let mut answer = Map::new();
    if ended.len() != count {
        answer.insert(
            "error".into(),
            format!(
                "{} requests ended before their cancellation",
                count - ended.len()
            )
            .into(),
        );
    }
    answer.insert("cancelled".into(), json!(ended));
    Ok(answer)
}

/// A change's action: an admin call, a signal, or a command.
async fn act(
    change: &Change,
    deployment: Option<&Deployment>,
    timeout_s: f64,
    target: Option<(u32, String)>,
) -> Result<Map<String, Value>> {
    if let Some(host) = &change.host {
        return signal_on_host(change, host, timeout_s).await;
    }
    let deployment = deployment.context("no deployment to change")?;
    if let (Some(method), Some(path)) = (&change.method, &change.path) {
        let listener = deployment
            .admin_socket
            .as_ref()
            .context("the deployment states no admin listener")?;
        let body = change.body.as_ref().map(serde_json::to_vec).transpose()?;
        let (status, text) = tokio::time::timeout(
            Duration::from_secs_f64(timeout_s),
            crate::admin::request(listener, method, path, body.as_deref()),
        )
        .await
        .map_err(|_| anyhow!("the admin call outlived {timeout_s}s"))??;
        let mut answer = Map::new();
        answer.insert("status".into(), status.into());
        answer.insert(
            "answer".into(),
            serde_json::from_str(&text).unwrap_or(Value::String(text)),
        );
        if !(200..300).contains(&status) {
            answer.insert("error".into(), format!("HTTP {status}").into());
        }
        return Ok(answer);
    }
    if let Some(signal) = &change.signal {
        let target = match target {
            Some((pid, line)) if command_line(pid).as_deref() == Some(line.as_str()) => (pid, line),
            _ => signal_target(change, deployment)?,
        };
        let number = SIGNALS
            .iter()
            .find(|(name, _)| name == signal)
            .map(|(_, number)| *number)
            .context("a known signal")?;
        send_signal(target.0, number)?;
        let mut answer = Map::new();
        answer.insert("pid".into(), target.0.into());
        answer.insert("cmdline".into(), target.1.into());
        return Ok(answer);
    }
    // A command runs as the deployment's processes do: in their directory, with
    // their environment.
    let mut command = tokio::process::Command::new(&change.command[0]);
    command
        .args(&change.command[1..])
        .current_dir(&deployment.working_directory)
        .envs(&deployment.environment)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs_f64(timeout_s), command.output())
        .await
        .map_err(|_| anyhow!("the command outlived {timeout_s}s"))??;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    let mut answer = Map::new();
    answer.insert("returncode".into(), output.status.code().into());
    answer.insert("output".into(), text.into());
    if !output.status.success() {
        answer.insert(
            "error".into(),
            format!("exit status {:?}", output.status.code()).into(),
        );
    }
    Ok(answer)
}

/// Sends a signal to a process.
fn send_signal(pid: u32, signal: i32) -> Result<()> {
    let pid = i32::try_from(pid)?;
    // SAFETY: kill(2) takes a pid and a signal number and touches no memory of
    // this process.
    let status = unsafe { libc::kill(pid, signal) };
    ensure!(
        status == 0,
        "signalling {pid}: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

/// Signals the one process on the change's host that its pattern matches,
/// through this program on that host (`uniserve-eval signal-process`) over ssh in
/// batch mode, which never prompts. The pattern travels on its standard input, so
/// no command line on that host carries it.
async fn signal_on_host(change: &Change, host: &str, timeout_s: f64) -> Result<Map<String, Value>> {
    use tokio::io::AsyncWriteExt;
    let mut child = tokio::process::Command::new("ssh")
        .args([
            "-o",
            "BatchMode=yes",
            host,
            "--",
            "uniserve-eval",
            "signal-process",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("starting ssh")?;
    let request = serde_json::to_vec(&json!({"process": change.process, "signal": change.signal}))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&request).await?;
    }
    let output = tokio::time::timeout(Duration::from_secs_f64(timeout_s), child.wait_with_output())
        .await
        .map_err(|_| anyhow!("ssh {host} outlived {timeout_s}s"))??;
    ensure!(
        output.status.success(),
        "ssh {host} exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let matches: Vec<(u32, String)> = serde_json::from_slice(&output.stdout)?;
    ensure!(
        matches.len() == 1,
        "{} processes on {host} match {:?}: {matches:?}",
        matches.len(),
        change.process
    );
    let mut answer = Map::new();
    answer.insert("host".into(), host.into());
    answer.insert("pid".into(), matches[0].0.into());
    answer.insert("cmdline".into(), matches[0].1.clone().into());
    Ok(answer)
}

/// `uniserve-eval signal-process`: reads `{"process": PATTERN, "signal": NAME}`
/// on standard input, signals the one process of this user, other than itself
/// and its parent, whose command line (its arguments joined by spaces) the
/// pattern matches, and prints every match as JSON.
///
/// # Errors
///
/// When the request does not parse or the signal fails.
pub fn signal_process() -> Result<()> {
    let request: Value = serde_json::from_reader(std::io::stdin())?;
    let pattern = regex::Regex::new(
        request
            .get("process")
            .and_then(Value::as_str)
            .context("a process pattern")?,
    )?;
    let name = request
        .get("signal")
        .and_then(Value::as_str)
        .context("a signal")?;
    let number = SIGNALS
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, number)| *number)
        .context("a known signal")?;
    let own = std::process::id();
    // SAFETY: getppid(2) and getuid(2) take no argument and cannot fail.
    let (parent, user) = unsafe { (libc::getppid() as u32, libc::getuid()) };
    let mut matches: Vec<(u32, String)> = Vec::new();
    for entry in std::fs::read_dir("/proc")?.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == own || pid == parent {
            continue;
        }
        let Ok(metadata) = std::fs::metadata(entry.path()) else {
            continue;
        };
        if std::os::unix::fs::MetadataExt::uid(&metadata) != user {
            continue;
        }
        if let Some(line) =
            command_line(pid).filter(|line| !line.is_empty() && pattern.is_match(line))
        {
            matches.push((pid, line));
        }
    }
    if matches.len() == 1 {
        send_signal(matches[0].0, number)?;
    }
    println!("{}", serde_json::to_string(&matches)?);
    Ok(())
}

/// A process's arguments joined by spaces, while it runs.
fn command_line(pid: u32) -> Option<String> {
    let arguments = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let trimmed = arguments.strip_suffix(b"\0").unwrap_or(&arguments);
    Some(
        String::from_utf8_lossy(
            &trimmed
                .iter()
                .map(|byte| if *byte == 0 { b' ' } else { *byte })
                .collect::<Vec<_>>(),
        )
        .into_owned(),
    )
}

/// The one process of the deployment a signal's pattern matches.
fn signal_target(change: &Change, deployment: &Deployment) -> Result<(u32, String)> {
    let pattern = regex::Regex::new(change.process.as_deref().context("a process pattern")?)?;
    let matches: Vec<(u32, String)> = (deployment.pids)()
        .into_iter()
        .filter_map(|pid| Some((pid, command_line(pid)?)))
        .filter(|(_, line)| pattern.is_match(line))
        .collect();
    ensure!(
        matches.len() == 1,
        "{} processes of the deployment match {:?}: {matches:?}",
        matches.len(),
        change.process
    );
    matches.into_iter().next().context("one match")
}

/// Reads the admin event log into the timeline until stopped: every event the
/// log keeps, from its first, `at_s` placing an event's own time on the load's
/// clock. While the listener does not answer, the first error of each outage is
/// recorded and the read retried every second.
async fn read_events(
    listener: &Path,
    record: &Timeline,
    start: f64,
    offset: f64,
    state: &State,
    clock: Clock,
) {
    let mut after: u64 = 0;
    let mut failing = false;
    loop {
        let path = format!("/admin/events?after={after}&wait={EVENT_WAIT_S}");
        let page = tokio::time::timeout(
            Duration::from_secs_f64(EVENT_WAIT_S + 10.0),
            crate::admin::request(listener, "GET", &path, None),
        )
        .await
        .map_err(|_| anyhow!("the event read timed out"))
        .and_then(|result| result)
        .and_then(|(status, text)| {
            ensure!((200..300).contains(&status), "HTTP {status}");
            Ok(serde_json::from_str::<Value>(&text)?)
        });
        let page = match page {
            Ok(page) => page,
            Err(error) => {
                if !failing {
                    let received = clock.now() - start;
                    if let Some(events) =
                        lock(record).get_mut("events").and_then(Value::as_array_mut)
                    {
                        events.push(json!({"error": format!("{error:#}"), "received_s": received}));
                    }
                }
                failing = true;
                tokio::time::sleep(Duration::from_secs_f64(EVENT_RETRY_S)).await;
                continue;
            }
        };
        failing = false;
        let received = clock.now() - start;
        {
            let mut timeline = lock(record);
            if let Some(events) = timeline.get_mut("events").and_then(Value::as_array_mut) {
                for event in page
                    .get("events")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let mut event = event.as_object().cloned().unwrap_or_default();
                    event.insert("received_s".into(), received.into());
                    if let Some(time) = event.get("time").and_then(Value::as_f64) {
                        event.insert("at_s".into(), (time - offset - start).into());
                    }
                    events.push(Value::Object(event));
                }
            }
        }
        after = after.max(page.get("next").and_then(Value::as_u64).unwrap_or(after));
        state.arrived.notify_waiters();
    }
}
