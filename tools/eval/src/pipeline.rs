//! One measurement point, from its rows to its result bundle.
//!
//! A run fixes its task and rows before the measured window, warms up and primes
//! outside it, measures, inspects media after it, and writes the bundle:
//! `examples.jsonl`, `client.json`, the record streams (`warmup_requests`,
//! `priming_requests`, `requests` and `gpu_samples`, all `.jsonl`), the media
//! indices, `server_metrics.json`, a timeline load's `timeline.json` and
//! `scrapes.jsonl`, `summary.json`, `summary.md`, and `run.json`, the lifecycle
//! record written last as the bundle's commit marker. A run that fails still
//! writes its streams and a failed `run.json`.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::bundle::Bundle;
use crate::load::{self, Deployment};
use crate::pyrandom::Random;
use crate::record::{Clock, Example, Record, TaskRequest, hex, rows_identity};
use crate::spec::{Direction, Metric, Point};
use crate::task::{Task, TaskName};

/// Sends each request to the replica with the fewest in flight, ties in rotation
/// after the last choice (an idle deployment receives requests round-robin): the
/// least-outstanding-requests policy of common HTTP load balancers.
pub struct Router {
    pub base_urls: Vec<String>,
    state: Mutex<(Vec<usize>, usize)>,
}

#[expect(
    clippy::expect_used,
    reason = "a poisoned lock means a panic elsewhere, which ends the run"
)]
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().expect("the run's lock")
}

impl Router {
    pub fn new(base_urls: Vec<String>) -> Self {
        let count = base_urls.len();
        Self {
            base_urls,
            state: Mutex::new((vec![0; count], count.saturating_sub(1))),
        }
    }

    /// The chosen origin's index, counted busy until [`Router::release`].
    fn route(&self) -> usize {
        let mut state = lock(&self.state);
        let count = state.0.len();
        let start = state.1;
        let index = (1..=count)
            .map(|offset| (start + offset) % count)
            .min_by_key(|candidate| state.0[*candidate])
            .unwrap_or(0);
        state.1 = index;
        state.0[index] += 1;
        index
    }

    fn release(&self, index: usize) {
        lock(&self.state).0[index] -= 1;
    }
}

/// The output length a row's request asks for: the row's, else the point's
/// `max_tokens`, else 0 (no limit).
fn requested_output_len(point: &Point, example: &Example) -> u64 {
    example
        .output_len
        .or(point.sampling.max_tokens)
        .unwrap_or(0)
}

/// The lifecycle record of the run's current state.
fn run_state(
    status: &str,
    point: &Point,
    started_at: f64,
    launch: &Value,
    selected: Option<&Value>,
    completed: Option<(f64, bool)>,
) -> Value {
    let mut state = json!({
        "status": status,
        "benchmark": point.name,
        "workload": point.workload(),
        "started_at": started_at,
        "launch": launch,
    });
    if let Some(selected) = selected {
        state["selected_rows"] = selected.clone();
    }
    if let Some((completed_at, valid)) = completed {
        state["completed_at"] = completed_at.into();
        state["valid"] = valid.into();
    }
    state
}

/// What a run measures against.
pub struct Target<'a> {
    pub base_urls: Vec<String>,
    pub launch: Value,
    pub timeout: Option<Duration>,
    pub deployment: Option<Arc<Deployment>>,
    pub output: &'a Path,
}

/// Runs one point and writes its bundle; returns its summary. A point whose
/// validation fails returns normally with `validation.valid` false; one with
/// changes is valid only if every change applied (`changes_applied`).
///
/// # Errors
///
/// When the bundle or the media directory is not empty (nothing is written), or
/// the run fails, after the records collected so far and a failed `run.json`
/// are written.
pub async fn run_point(point: &Point, target: Target<'_>) -> Result<Value> {
    let output = target.output;
    ensure!(
        !std::fs::read_dir(output).is_ok_and(|mut entries| entries.next().is_some()),
        "result directory is not empty: {}",
        output.display()
    );
    // Measured videos are named by request id, which every run of the point
    // shares: a run that found another's would keep a video it did not produce.
    if let Some(media) = point
        .video
        .media_dir
        .as_deref()
        .filter(|media| !media.is_empty())
    {
        ensure!(
            !std::fs::read_dir(media).is_ok_and(|mut entries| entries.next().is_some()),
            "media directory is not empty: {media}"
        );
    }
    let bundle = Bundle::create(output)?;
    let started_at = load::wall();
    for stream in [
        "warmup_requests.jsonl",
        "priming_requests.jsonl",
        "requests.jsonl",
        "gpu_samples.jsonl",
    ] {
        bundle.write_jsonl(stream, &[])?;
    }
    bundle.write_json(
        "run.json",
        &run_state("preparing", point, started_at, &target.launch, None, None),
    )?;
    let state = Arc::new(Mutex::new(Collected::default()));
    let result = measure(point, &target, &bundle, started_at, &state).await;
    match result {
        Ok(summary) => Ok(summary),
        Err(error) => {
            let collected = std::mem::take(&mut *lock(&state));
            let mut failure = run_state(
                "failed",
                point,
                started_at,
                &target.launch,
                collected.selected.as_ref(),
                Some((load::wall(), false)),
            );
            failure["error"] = json!({"type": error_type(&error), "message": format!("{error:#}")});
            if let Some(gpu) = &collected.gpu_summary {
                failure["gpu_storage"] = gpu.clone();
            }
            // Persisting the streams may fail again when persisting them was the
            // failure: that second error is recorded, and the first propagates.
            let persisted = (|| -> Result<()> {
                bundle.write_jsonl("gpu_samples.jsonl", &collected.gpu_samples)?;
                write_records(
                    &bundle,
                    "warmup_requests.jsonl",
                    &mut lock(&collected.warmup),
                    None,
                )?;
                write_records(
                    &bundle,
                    "priming_requests.jsonl",
                    &mut lock(&collected.priming),
                    None,
                )?;
                write_records(
                    &bundle,
                    "requests.jsonl",
                    &mut lock(&collected.measured),
                    point.video.media_dir.as_deref(),
                )?;
                if point.load.timeline() {
                    bundle.write_json(
                        "timeline.json",
                        &Value::Object(lock(&collected.timeline).clone()),
                    )?;
                }
                if !collected.scrapes.is_empty() {
                    bundle.write_jsonl("scrapes.jsonl", &collected.scrapes)?;
                }
                Ok(())
            })();
            if let Err(artifact) = persisted {
                failure["artifact_error"] =
                    json!({"type": error_type(&artifact), "message": format!("{artifact:#}")});
            }
            bundle.write_json("run.json", &failure)?;
            Err(error)
        }
    }
}

fn error_type(error: &anyhow::Error) -> &'static str {
    if error.downcast_ref::<load::WarmupFailure>().is_some() {
        "WarmupFailure"
    } else {
        "Error"
    }
}

/// What a run has collected, which a failure persists.
#[derive(Default)]
struct Collected {
    selected: Option<Value>,
    warmup: Arc<Mutex<Vec<Record>>>,
    priming: Arc<Mutex<Vec<Record>>>,
    measured: Arc<Mutex<Vec<Record>>>,
    timeline: load::Timeline,
    scrapes: Vec<Value>,
    gpu_samples: Vec<Value>,
    gpu_summary: Option<Value>,
}

/// The excluded rows of a manifest, every one in its frozen order.
fn manifest_rows(point: &Point, path: Option<&str>) -> Result<Option<Vec<Example>>> {
    let Some(path) = path.filter(|path| !path.is_empty()) else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let count = text.lines().filter(|line| !line.trim().is_empty()).count();
    let mut excluded = point.clone();
    excluded.dataset = "jsonl".into();
    excluded.dataset_path = Some(path.to_owned());
    excluded.load.num_prompts = count;
    Ok(Some(crate::dataset::load(&excluded)?))
}

#[expect(clippy::too_many_lines, reason = "one run's stages, in order")]
async fn measure(
    point: &Point,
    target: &Target<'_>,
    bundle: &Bundle,
    started_at: f64,
    collected: &Arc<Mutex<Collected>>,
) -> Result<Value> {
    let timeout = target
        .timeout
        .unwrap_or(Duration::from_secs_f64(point.load.request_timeout_s));
    let router = Arc::new(Router::new(
        target
            .base_urls
            .iter()
            .map(|url| url.trim_end_matches('/').to_owned())
            .collect(),
    ));
    ensure!(
        !router.base_urls.is_empty(),
        "a deployment requires at least one origin"
    );

    // The task and the rows are fixed before the measured window opens.
    let task = Task::new(point);
    let rows = crate::dataset::load(point)?;
    let mut ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    ensure!(
        ids.len() == rows.len(),
        "request IDs must be unique within a manifest"
    );
    let warmup_rows = manifest_rows(point, point.load.warmup_manifest.as_deref())?;
    let priming_rows = manifest_rows(point, point.load.priming_manifest.as_deref())?;
    let tokenizer = match crate::dataset::Kind::of(&point.dataset)? {
        crate::dataset::Kind::ShareGpt | crate::dataset::Kind::FixedLength => {
            Some(Arc::new(crate::dataset::tokenizer(
                point
                    .tokenizer
                    .as_deref()
                    .filter(|name| !name.is_empty())
                    .unwrap_or(&point.model),
            )?))
        }
        _ => None,
    };
    // A video row's request reads its media now, outside every slot.
    let mut requests: std::collections::HashMap<String, TaskRequest> =
        std::collections::HashMap::new();
    if point.task == TaskName::Video {
        for row in rows
            .iter()
            .chain(warmup_rows.iter().flatten())
            .chain(priming_rows.iter().flatten())
        {
            requests.insert(row.id.clone(), task.build_request(row)?);
        }
    }
    let selected = rows_identity(&rows);
    lock(collected).selected = Some(selected.clone());
    bundle.write_jsonl(
        "examples.jsonl",
        &rows.iter().map(Example::as_value).collect::<Vec<_>>(),
    )?;
    bundle.write_json(
        "client.json",
        &json!({
            "request_timeout_s": timeout.as_secs_f64(),
            "connection_limits": Value::Null,
            "keepalive_expiry_s": Value::Null,
            "media_buffer": "host_storage",
            "warmup_rows": rows_identity(warmup_rows.as_deref().unwrap_or_default()),
            "priming_rows": rows_identity(priming_rows.as_deref().unwrap_or_default()),
        }),
    )?;
    bundle.write_json(
        "run.json",
        &run_state(
            "running",
            point,
            started_at,
            &target.launch,
            Some(&selected),
            None,
        ),
    )?;

    // One client, an unbounded pool: the only client-side limit on requests in
    // flight is the load's max_concurrency.
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(usize::MAX)
        .build()?;
    let clock = Clock::new();
    let (warmup_records, priming_records, measured_records) = {
        let collected = lock(collected);
        (
            Arc::clone(&collected.warmup),
            Arc::clone(&collected.priming),
            Arc::clone(&collected.measured),
        )
    };
    let priming_phase = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shared_point = Arc::new(point.clone());
    let shared_requests = Arc::new(requests);
    let submit: load::Submit = {
        let (client, router, point, requests) = (
            client.clone(),
            Arc::clone(&router),
            Arc::clone(&shared_point),
            Arc::clone(&shared_requests),
        );
        let (warmup_records, priming_records, measured_records, priming_phase) = (
            Arc::clone(&warmup_records),
            Arc::clone(&priming_records),
            Arc::clone(&measured_records),
            Arc::clone(&priming_phase),
        );
        Arc::new(move |example: Example, scheduled: Option<f64>| {
            let (client, router, point, requests) = (
                client.clone(),
                Arc::clone(&router),
                Arc::clone(&point),
                Arc::clone(&requests),
            );
            let (warmup_records, priming_records, measured_records, priming_phase) = (
                Arc::clone(&warmup_records),
                Arc::clone(&priming_records),
                Arc::clone(&measured_records),
                Arc::clone(&priming_phase),
            );
            Box::pin(async move {
                let task = Task::new(&point);
                let request = match requests.get(&example.id) {
                    Some(request) => Ok(request.clone()),
                    None => task.build_request(&example),
                };
                // Routing happens inside the slot: a request waits on its replica
                // as on any queueing server.
                let index = router.route();
                let mut record = match request {
                    Ok(request) => {
                        crate::transport::send(
                            crate::transport::Send {
                                client: &client,
                                clock: &clock,
                                base_url: &router.base_urls[index],
                                request_id: example.id.clone(),
                                task: point.task.as_str(),
                                prompt_len: example.prompt_len.unwrap_or(0),
                                output_len: requested_output_len(&point, &example),
                                scheduled_time: scheduled,
                                timeout,
                            },
                            &request,
                        )
                        .await
                    }
                    Err(error) => {
                        let mut record = Record::new(example.id.clone(), point.task.as_str());
                        record.begin(
                            &clock,
                            &point.endpoint,
                            scheduled,
                            requested_output_len(&point, &example),
                        );
                        record.mark_transport_failure(&clock, &error);
                        record
                    }
                };
                router.release(index);
                record.example = Some(example.as_value());
                // Records are kept as they finish, so a failed load leaves them.
                let stream = if scheduled.is_none() {
                    &warmup_records
                } else if priming_phase.load(std::sync::atomic::Ordering::SeqCst) {
                    &priming_records
                } else {
                    &measured_records
                };
                lock(stream).push(record.clone());
                record
            })
        })
    };
    let cancelled: Arc<dyn Fn(Example, f64) -> Record + Send + Sync> = {
        let (point, measured_records) = (Arc::clone(&shared_point), Arc::clone(&measured_records));
        Arc::new(move |example: Example, submitted: f64| {
            // The record of a request a cancel disconnected holds its row and
            // times; what its stream had received is not kept.
            let mut record = Record::new(example.id.clone(), point.task.as_str());
            record.endpoint = point.endpoint.clone();
            record.scheduled_time = Some(submitted);
            record.start_time = submitted;
            record.requested_output_len = requested_output_len(&point, &example);
            record.mark_cancelled(&clock);
            record.example = Some(example.as_value());
            lock(&measured_records).push(record.clone());
            record
        })
    };
    let inspect: load::Inspect = {
        let point = Arc::clone(&shared_point);
        Box::new(move |records: &mut [Record]| {
            if point.task == TaskName::Video {
                let task = Task::new(&point);
                for record in records {
                    task.inspect_output(record);
                }
            }
        })
    };

    let metrics_before: Arc<Mutex<Map<String, Value>>> = Arc::new(Mutex::new(Map::new()));
    let scraper_slot: Arc<Mutex<Option<crate::telemetry::Scraper>>> = Arc::new(Mutex::new(None));
    let snapshot = {
        let (client, router, metrics_before, scraper_slot) = (
            client.clone(),
            Arc::clone(&router),
            Arc::clone(&metrics_before),
            Arc::clone(&scraper_slot),
        );
        let timeline = point.load.timeline();
        async move {
            for origin in &router.base_urls {
                let text = crate::telemetry::scrape(&client, origin).await;
                lock(&metrics_before).insert(origin.clone(), text.into());
            }
            // A timeline load also scrapes every replica each second, from its
            // start until its last stream ended.
            if timeline {
                *lock(&scraper_slot) = Some(crate::telemetry::Scraper::start(
                    client.clone(),
                    router.base_urls.clone(),
                    clock,
                ));
            }
        }
    };

    // Sampling spans the excluded requests and the measured window.
    let mut sampler = crate::telemetry::GpuSampler::start();
    let mut arrivals = Random::numpy(u32::try_from(point.load.seed).context("a 32-bit load seed")?);
    let outcome: Result<load::Outcome> = async {
        let mut warmup_corpus = warmup_rows.clone();
        if let Some(priming) = &priming_rows {
            priming_phase.store(true, std::sync::atomic::Ordering::SeqCst);
            // The shape warm-up precedes the queue priming; priming has no pause
            // after it, and its inspection waits for the measurement.
            let primed = load::run(
                priming,
                load::Driver {
                    load: &crate::spec::Load {
                        request_rate: f64::INFINITY,
                        ..point.load.clone()
                    },
                    submit: Arc::clone(&submit),
                    clock,
                    arrivals: Random::numpy(0),
                    warmup_requests: point.load.warmup_requests,
                    warmup_rows: warmup_corpus.take(),
                    inspect: Box::new(|records: &mut [Record]| {
                        let _ = records;
                    }),
                },
                async {},
            )
            .await?;
            priming_phase.store(false, std::sync::atomic::Ordering::SeqCst);
            *lock(&warmup_records) = primed.warmup;
            *lock(&priming_records) = primed.measured;
            if !lock(&priming_records).iter().all(|record| record.success) {
                bail!(load::WarmupFailure {
                    records: lock(&priming_records).clone()
                });
            }
            warmup_corpus = Some(Vec::new());
        }
        let context = load::Driver {
            load: &point.load,
            submit: Arc::clone(&submit),
            clock,
            arrivals: std::mem::replace(&mut arrivals, Random::numpy(0)),
            warmup_requests: if priming_rows.is_some() {
                0
            } else {
                point.load.warmup_requests
            },
            warmup_rows: warmup_corpus,
            inspect,
        };
        if point.load.timeline() {
            let timeline = Arc::clone(&lock(collected).timeline);
            load::timeline(
                &rows,
                context,
                &point.changes,
                target.deployment.clone(),
                timeline,
                Arc::clone(&cancelled),
                snapshot,
            )
            .await
        } else {
            load::run(&rows, context, snapshot).await
        }
    }
    .await;
    sampler.stop();
    let scraper = lock(&scraper_slot).take();
    if let Some(scraper) = scraper {
        lock(collected).scrapes = scraper.stop().await;
    }
    {
        let mut collected = lock(collected);
        collected.gpu_samples = sampler.records();
        collected.gpu_summary = sampler.summary();
    }
    let outcome = outcome?;

    // The load's records, in submission order, replace those collected as
    // they finished.
    if priming_rows.is_none() {
        *lock(&warmup_records) = outcome.warmup;
    }
    let mut records = outcome.measured;
    {
        let task = Task::new(point);
        if point.task == TaskName::Video {
            for record in records.iter_mut().chain(lock(&priming_records).iter_mut()) {
                task.inspect_output(record);
            }
        }
    }
    // Counters and provenance after the window.
    let mut metrics_after = Map::new();
    for origin in &router.base_urls {
        metrics_after.insert(
            origin.clone(),
            crate::telemetry::scrape(&client, origin).await.into(),
        );
    }
    let server_version = version(&client, &router.base_urls[0]).await;

    // A request a cancel change disconnected is the change's doing: counted
    // apart, left out of the metrics and the validation.
    let served: Vec<&Record> = records
        .iter()
        .filter(|record| record.classifier != crate::record::CANCELLED)
        .collect();
    let mut summary = build_summary(
        point,
        &router.base_urls,
        &served,
        outcome.duration_s,
        &task,
        &selected,
        tokenizer.as_deref(),
        server_version.as_ref(),
        &target.launch,
    )?;
    if !point.changes.is_empty() {
        summary["cancelled_count"] = (records.len() - served.len()).into();
    }
    let timeline = lock(&lock(collected).timeline).clone();
    {
        let checks = summary["validation"]["checks"]
            .as_object_mut()
            .context("the summary's checks")?;
        let priming = lock(&priming_records);
        if !priming.is_empty() {
            checks.insert(
                "priming_valid".into(),
                priming.iter().all(|record| record.success).into(),
            );
        }
        // A run of changes that did not all apply is not a run of its point.
        if !point.changes.is_empty() {
            let applied = timeline
                .get("changes")
                .and_then(Value::as_array)
                .is_some_and(|changes| {
                    changes.iter().all(|entry| {
                        entry.get("issued_s").is_some() && entry.get("error").is_none()
                    })
                });
            checks.insert("changes_applied".into(), applied.into());
        }
        let valid = checks.values().all(|ok| ok.as_bool() == Some(true));
        summary["validation"]["valid"] = valid.into();
    }
    if let Some(gpu) = sampler.summary() {
        summary["gpu_storage"] = gpu;
    }
    let retained: usize = lock(&warmup_records)
        .iter()
        .chain(lock(&priming_records).iter())
        .chain(records.iter())
        .map(|record| record.video_body.as_ref().map_or(0, Vec::len))
        .sum();
    summary["client"] = json!({"peak_rss_bytes": peak_rss(), "retained_body_bytes": retained});
    // The window's utilization of the GPUs the server was launched on, and the
    // counters the window advanced.
    summary["gpu_utilization"] = sampler
        .window_utilization(
            outcome.window_start,
            outcome.window_end,
            launch_gpus(&target.launch).as_deref(),
        )
        .unwrap_or(Value::Null);
    let before = lock(&metrics_before).clone();
    summary["server_metrics"] = Value::Object(
        router
            .base_urls
            .iter()
            .map(|origin| {
                (
                    origin.clone(),
                    crate::telemetry::window_summary(
                        before.get(origin).and_then(Value::as_str),
                        metrics_after.get(origin).and_then(Value::as_str),
                    ),
                )
            })
            .collect(),
    );

    // The lifecycle record follows every result artifact: the bundle's commit.
    write_records(
        bundle,
        "warmup_requests.jsonl",
        &mut lock(&warmup_records),
        None,
    )?;
    write_records(
        bundle,
        "priming_requests.jsonl",
        &mut lock(&priming_records),
        None,
    )?;
    write_records(
        bundle,
        "requests.jsonl",
        &mut records,
        point.video.media_dir.as_deref(),
    )?;
    bundle.write_jsonl("gpu_samples.jsonl", &sampler.records())?;
    bundle.write_json(
        "server_metrics.json",
        &json!({"before": before, "after": metrics_after}),
    )?;
    if point.load.timeline() {
        bundle.write_json("timeline.json", &Value::Object(timeline))?;
        let scrapes = std::mem::take(&mut lock(collected).scrapes);
        bundle.write_jsonl("scrapes.jsonl", &scrapes)?;
    }
    bundle.write_json("summary.json", &summary)?;
    bundle.write_text("summary.md", &render_markdown(&summary))?;
    let valid = summary["validation"]["valid"].as_bool().unwrap_or(false);
    bundle.write_json(
        "run.json",
        &run_state(
            "completed",
            point,
            started_at,
            &target.launch,
            Some(&selected),
            Some((load::wall(), valid)),
        ),
    )?;
    Ok(summary)
}

/// The process's peak resident memory in bytes.
fn peak_rss() -> u64 {
    // SAFETY: getrusage(2) fills the structure it is given.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: as above, on a valid pointer.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) };
    u64::try_from(usage.ru_maxrss).unwrap_or(0) * 1024
}

/// The GPU indices a launch selected with `CUDA_VISIBLE_DEVICES`: `None` for a
/// UUID selector or none, which selects every sampled GPU.
fn launch_gpus(launch: &Value) -> Option<Vec<u32>> {
    let selector = launch
        .get("environment")?
        .get("CUDA_VISIBLE_DEVICES")?
        .as_str()?;
    if selector.trim().is_empty() {
        return None;
    }
    selector
        .split(',')
        .map(|item| item.trim().parse().ok())
        .collect()
}

/// The server's `GET /version` object, when it serves one.
async fn version(client: &reqwest::Client, base_url: &str) -> Option<Value> {
    let response = client
        .get(format!("{}/version", base_url.trim_end_matches('/')))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .ok()?;
    if response.status().as_u16() != 200 {
        return None;
    }
    let value: Value = response.json().await.ok()?;
    value.is_object().then_some(value)
}

/// Writes a stream's media before the stream that names it: each image sample
/// under `samples/`, each video body at `<media_dir>/<request id>.mp4` or, without
/// a media directory, content-addressed under `samples/`. `requests.jsonl`
/// carries its media index and, for video, its validity beside it.
fn write_records(
    bundle: &Bundle,
    name: &str,
    records: &mut [Record],
    media_dir: Option<&str>,
) -> Result<()> {
    for record in records.iter_mut() {
        for image in &record.decoded_images {
            bundle.write_image(image)?;
        }
        if let Some(body) = &record.video_body {
            let checksum = hex(&Sha256::digest(body));
            let path = match media_dir.filter(|media| !media.is_empty()) {
                Some(media) => {
                    let id = &record.request_id;
                    ensure!(
                        Path::new(id)
                            .file_name()
                            .is_some_and(|file| file == id.as_str())
                            && id != "."
                            && id != "..",
                        "media request ID must be a single file name"
                    );
                    Path::new(media).join(format!("{id}.mp4"))
                }
                None => bundle
                    .directory
                    .join("samples")
                    .join(format!("{checksum}.mp4")),
            };
            Bundle::write_original(&path, body)?;
            record.original_output = Some(json!({
                "path": path.display().to_string(),
                "sha256": checksum,
                "byte_size": body.len(),
                "mime": record.video_mime,
            }));
        }
    }
    bundle.write_jsonl(
        name,
        &records.iter().map(Record::to_value).collect::<Vec<_>>(),
    )?;
    if name == "requests.jsonl" {
        let run = bundle.directory.join("run.json");
        let timing = bundle.directory.join("requests.jsonl");
        bundle.write_jsonl(
            "media_index.jsonl",
            &records
                .iter()
                .map(|record| {
                    json!({
                        "request_id": record.request_id,
                        "success": record.success,
                        "classifier": record.classifier,
                        "original_output": record.original_output,
                        "media_checks": record.media_checks,
                        "example": record.example,
                        "run": run.display().to_string(),
                        "timing_records": timing.display().to_string(),
                    })
                })
                .collect::<Vec<_>>(),
        )?;
        bundle.write_jsonl(
            "media_validity.jsonl",
            &records
                .iter()
                .filter(|record| record.task == "video")
                .map(|record| {
                    json!({
                        "request_id": record.request_id,
                        "valid": record.success,
                        "classifier": record.classifier,
                        "checks": record.media_checks,
                        "video": record.decoded_video.as_ref().map(crate::record::DecodedVideo::metadata),
                    })
                })
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}

/// The value a dotted metric path names, when it is a finite number.
pub fn metric_value(metrics: &Value, metric: &Metric) -> Option<f64> {
    let mut value = metrics;
    for part in &metric.path {
        value = value.as_object()?.get(part)?;
    }
    if value.is_boolean() {
        return None;
    }
    value.as_f64().filter(|number| number.is_finite())
}

/// The summary: the workload's identity, the metrics with the task's own, the
/// validation (the task's checks and one per protected metric, which any
/// measured finite value passes), warnings and provenance.
///
/// # Errors
///
/// When a task check's name repeats another's.
#[expect(
    clippy::too_many_arguments,
    reason = "the summary's sources, each one's own"
)]
pub fn build_summary(
    point: &Point,
    base_urls: &[String],
    records: &[&Record],
    duration_s: f64,
    task: &Task<'_>,
    selected: &Value,
    tokenizer: Option<&tokenizers::Tokenizer>,
    server_version: Option<&Value>,
    launch: &Value,
) -> Result<Value> {
    let mut metrics = crate::metrics::summarize(records, duration_s, tokenizer);
    metrics.extend(task.summary_metrics(records));
    let metrics = Value::Object(metrics);
    let mut checks = crate::record::Validation::default();
    for metric in &point.metrics {
        checks.checks.insert(
            format!("metric:{}", metric.name()),
            metric_value(&metrics, metric).is_some(),
        );
    }
    let validation = task.validate(records)?.merged(checks)?;
    // Request warnings deduplicated and sorted, then the provenance warnings.
    let mut warnings: Vec<String> = records
        .iter()
        .flat_map(|record| record.warnings.iter().cloned())
        .collect();
    warnings.sort();
    warnings.dedup();
    if server_version.is_none() {
        warnings.push("server_version_unavailable".into());
    }
    if launch.get("dirty").and_then(Value::as_bool) == Some(true) {
        warnings.push("dirty_workspace".into());
    }
    let mut classifiers: Map<String, Value> = Map::new();
    for record in records {
        let count = classifiers
            .get(&record.classifier)
            .and_then(Value::as_u64)
            .unwrap_or(0);
        classifiers.insert(record.classifier.clone(), (count + 1).into());
    }
    Ok(json!({
        "status": "completed",
        "benchmark": point.name,
        "task": point.task.as_str(),
        "model": point.model,
        "dataset": point.dataset,
        "endpoint": point.endpoint,
        "base_urls": base_urls,
        "workload": point.workload(),
        "selected_rows": selected,
        "elapsed_s": duration_s,
        "request_count": records.len(),
        "ok_count": records.iter().filter(|record| record.success).count(),
        "failed_count": records.iter().filter(|record| !record.success).count(),
        "classifiers": classifiers,
        "metrics": metrics,
        "metric_definitions": point.metrics.iter().map(|metric| json!({"path": metric.name(), "direction": metric.direction})).collect::<Vec<_>>(),
        "validation": validation.to_value(),
        "warnings": warnings,
        "server_version": server_version,
        "launch": launch,
    }))
}

/// A number with six significant digits, as C's `%g` writes it.
pub fn general(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    if !value.is_finite() {
        return value.to_string();
    }
    let exponent = value.abs().log10().floor() as i32;
    let trim = |text: String| -> String {
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text
        }
    };
    if (-4..6).contains(&exponent) {
        let decimals = usize::try_from(5 - exponent).unwrap_or(0);
        trim(format!("{value:.decimals$}"))
    } else {
        let text = format!("{value:.5e}");
        let (mantissa, power) = text.split_once('e').unwrap_or((&text, "0"));
        let power: i32 = power.parse().unwrap_or(0);
        format!(
            "{}e{}{:02}",
            trim(mantissa.to_owned()),
            if power < 0 { '-' } else { '+' },
            power.abs()
        )
    }
}

/// The protected metrics and the validation, as Markdown.
pub fn render_markdown(summary: &Value) -> String {
    let valid = summary["validation"]["valid"].as_bool() == Some(true);
    let mut lines = vec![
        format!("# {}", summary["benchmark"].as_str().unwrap_or_default()),
        String::new(),
        format!(
            "- requests: {}/{} successful",
            summary["ok_count"].as_u64().unwrap_or(0),
            summary["request_count"].as_u64().unwrap_or(0)
        ),
        format!(
            "- elapsed: {:.3} s",
            summary["elapsed_s"].as_f64().unwrap_or(0.0)
        ),
        format!("- validation: {}", if valid { "pass" } else { "fail" }),
        String::new(),
        "| Metric | Value | Direction |".into(),
        "| --- | ---: | --- |".into(),
    ];
    for definition in summary["metric_definitions"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let path = definition["path"].as_str().unwrap_or_default();
        let direction = definition["direction"].as_str().unwrap_or_default();
        let metric = Metric {
            path: path.split('.').map(str::to_owned).collect(),
            direction: if direction == "higher" {
                Direction::Higher
            } else {
                Direction::Lower
            },
        };
        let rendered =
            metric_value(&summary["metrics"], &metric).map_or_else(|| "n/a".to_owned(), general);
        lines.push(format!("| `{path}` | {rendered} | {direction} |"));
    }
    let failed: Vec<String> = summary["validation"]["checks"]
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, ok)| ok.as_bool() != Some(true))
        .map(|(name, _)| format!("`{name}`"))
        .collect();
    if !failed.is_empty() {
        lines.push(String::new());
        lines.push(format!("Failed checks: {}.", failed.join(", ")));
    }
    if let Some(warnings) = summary["warnings"]
        .as_array()
        .filter(|warnings| !warnings.is_empty())
    {
        lines.push(String::new());
        lines.push(format!(
            "Warnings: {}.",
            warnings
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `%g`'s six significant digits: trailing zeros dropped, exponents past the
    /// fixed range.
    #[test]
    fn numbers_render_as_general_format() {
        assert_eq!(general(8426.35), "8426.35");
        assert_eq!(general(40.8318012), "40.8318");
        assert_eq!(general(0.5), "0.5");
        assert_eq!(general(1234567.0), "1.23457e+06");
        assert_eq!(general(0.00001234), "1.234e-05");
        assert_eq!(general(100.0), "100");
    }
}
