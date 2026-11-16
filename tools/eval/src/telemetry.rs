//! What the servers and the GPUs report while a point runs.
//!
//! A server's Prometheus counters are scraped just before the measured window
//! opens and after it closes, and, for a timeline load, every second of it;
//! the window's summary is how far each counter advanced (series ending in
//! `_total`, `_sum` or `_count`; a gauge's difference between two instants means
//! nothing). The GPUs are sampled through `nvidia-smi`, which reports every GPU
//! it sees and device-wide memory, so its readings include processes other than
//! the server under test: memory in MiB, utilization in percent.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::record::Clock;

/// A server's `/metrics` exposition, or `None` when it serves none (a non-200
/// status or any error): a server without metrics still completes its point.
pub async fn scrape(client: &reqwest::Client, base_url: &str) -> Option<String> {
    let response = client
        .get(format!("{}/metrics", base_url.trim_end_matches('/')))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .ok()?;
    if response.status().as_u16() != 200 {
        return None;
    }
    response.text().await.ok()
}

/// An exposition's samples by `name{labels}` as the server rendered it; comment
/// lines, lines without a numeric value and trailing timestamps are left out.
pub fn samples(text: &str) -> BTreeMap<String, f64> {
    let mut samples = BTreeMap::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Label values may hold spaces: a series' key ends at the closing brace.
        let (key, rest) = if line.contains('{') {
            let Some(close) = line.rfind('}') else {
                continue;
            };
            (&line[..=close], &line[close + 1..])
        } else {
            line.split_once(' ').unwrap_or((line, ""))
        };
        if let Some(value) = rest
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<f64>().ok())
        {
            samples.insert(key.to_owned(), value);
        }
    }
    samples
}

/// How far each counter advanced between two scrapes; a series absent from the
/// first started at zero, and one that did not advance (or by a non-finite
/// amount) is left out.
pub fn counter_deltas(before: &str, after: &str) -> Map<String, Value> {
    let start = samples(before);
    samples(after)
        .into_iter()
        .filter(|(key, _)| {
            let name = key.split('{').next().unwrap_or_default();
            ["_total", "_sum", "_count"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
        })
        .filter_map(|(key, value)| {
            let delta = value - start.get(&key).copied().unwrap_or(0.0);
            (delta != 0.0 && delta.is_finite()).then(|| (key, Value::from(delta)))
        })
        .collect()
}

/// The window's counter deltas; unavailable when either scrape failed.
pub fn window_summary(before: Option<&str>, after: Option<&str>) -> Value {
    match (before, after) {
        (Some(before), Some(after)) => {
            json!({"available": true, "counter_deltas": counter_deltas(before, after)})
        }
        _ => json!({"available": false, "counter_deltas": {}}),
    }
}

/// Scrapes a fixed set of origins once a second, on the grid of its first tick
/// (a late wake-up shifts no later tick), until stopped.
pub struct Scraper {
    entries: Arc<Mutex<Vec<Value>>>,
    ticker: tokio::task::JoinHandle<()>,
    scrapes: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

#[expect(
    clippy::expect_used,
    reason = "a poisoned lock means a panic elsewhere, which ends the run"
)]
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().expect("the telemetry's lock")
}

impl Scraper {
    /// Takes the first tick now.
    pub fn start(client: reqwest::Client, origins: Vec<String>, clock: Clock) -> Self {
        let entries = Arc::new(Mutex::new(Vec::new()));
        let scrapes: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let ticker = {
            let (entries, scrapes) = (Arc::clone(&entries), Arc::clone(&scrapes));
            tokio::spawn(async move {
                let origin_time = clock.now();
                let mut tick = 0u64;
                loop {
                    for origin in &origins {
                        let (client, entries, origin) = (
                            client.clone(),
                            Arc::clone(&entries),
                            origin.trim_end_matches('/').to_owned(),
                        );
                        lock(&scrapes).push(tokio::spawn(async move {
                            let sent = clock.now();
                            let text = scrape(&client, &origin).await;
                            lock(&entries).push(json!({
                                "origin": origin,
                                "tick": tick,
                                "sent": sent,
                                "received": clock.now(),
                                "text": text,
                            }));
                        }));
                    }
                    let next = origin_time + (tick + 1) as f64;
                    tokio::time::sleep(Duration::from_secs_f64((next - clock.now()).max(0.0)))
                        .await;
                    tick += 1;
                }
            })
        };
        Self {
            entries,
            ticker,
            scrapes,
        }
    }

    /// Stops ticking and waits for the scrapes in flight (each ends within its
    /// own timeout): the entries hold one for every tick taken.
    pub async fn stop(self) -> Vec<Value> {
        self.ticker.abort();
        let pending: Vec<_> = std::mem::take(&mut *lock(&self.scrapes));
        for scrape in pending {
            let _ = scrape.await;
        }
        std::mem::take(&mut *lock(&self.entries))
    }
}

/// One GPU's reading.
#[derive(Debug, Clone, Copy)]
struct Reading {
    index: u32,
    memory_used_mib: u64,
    utilization_gpu_pct: u64,
}

/// One reading per GPU, in `nvidia-smi`'s order; none when it fails.
fn read_gpus() -> Vec<Reading> {
    let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            let [index, memory, utilization] = parts.as_slice() else {
                return None;
            };
            Some(Reading {
                index: index.parse().ok()?,
                memory_used_mib: memory.parse().ok()?,
                utilization_gpu_pct: utilization.parse().ok()?,
            })
        })
        .collect()
}

/// The samples taken: each one's wall-clock time and every GPU's reading.
type Samples = Arc<Mutex<Vec<(f64, Vec<Reading>)>>>;

/// Samples the GPUs every half second on a thread of its own until stopped.
pub struct GpuSampler {
    samples: Samples,
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// The sampling interval in seconds.
const GPU_INTERVAL_S: f64 = 0.5;

impl GpuSampler {
    /// Starts sampling; without `nvidia-smi` it samples nothing.
    pub fn start() -> Self {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let thread = {
            let (samples, stop) = (Arc::clone(&samples), Arc::clone(&stop));
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                    let readings = read_gpus();
                    if !readings.is_empty() {
                        lock(&samples).push((crate::load::wall(), readings));
                    }
                    std::thread::sleep(Duration::from_secs_f64(GPU_INTERVAL_S));
                }
            })
        };
        Self {
            samples,
            stop,
            thread: Some(thread),
        }
    }

    /// Stops sampling and joins the thread.
    pub fn stop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    /// The raw samples, one line each: the wall-clock time and every GPU's reading.
    pub fn records(&self) -> Vec<Value> {
        lock(&self.samples)
            .iter()
            .map(|(time, readings)| {
                json!({
                    "time": time,
                    "gpus": readings.iter().map(|reading| json!({
                        "index": reading.index,
                        "memory_used_mib": reading.memory_used_mib,
                        "utilization_gpu_pct": reading.utilization_gpu_pct,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect()
    }

    /// The peaks over the whole sampling span (per GPU position, their total,
    /// the largest single GPU) and the mean of the per-sample utilization summed
    /// over the GPUs, which exceeds 100 on a host of several; `None` without a
    /// sample.
    pub fn summary(&self) -> Option<Value> {
        let samples = lock(&self.samples);
        if samples.is_empty() {
            return None;
        }
        let mut peak_memory: Vec<u64> = Vec::new();
        let mut peak_utilization: Vec<u64> = Vec::new();
        let (mut peak_total, mut utilization_total) = (0u64, 0u64);
        for (_, readings) in samples.iter() {
            peak_memory.resize(peak_memory.len().max(readings.len()), 0);
            peak_utilization.resize(peak_utilization.len().max(readings.len()), 0);
            for (position, reading) in readings.iter().enumerate() {
                peak_memory[position] = peak_memory[position].max(reading.memory_used_mib);
                peak_utilization[position] =
                    peak_utilization[position].max(reading.utilization_gpu_pct);
            }
            peak_total =
                peak_total.max(readings.iter().map(|reading| reading.memory_used_mib).sum());
            utilization_total += readings
                .iter()
                .map(|reading| reading.utilization_gpu_pct)
                .sum::<u64>();
        }
        Some(json!({
            "peak_per_gpu_mib": peak_memory,
            "peak_total_mib": peak_total,
            "peak_single_gpu_mib": peak_memory.iter().copied().max().unwrap_or(0),
            "peak_utilization_gpu_pct": peak_utilization,
            "mean_total_utilization_gpu_pct": utilization_total as f64 / samples.len() as f64,
            "samples": samples.len(),
            "interval_s": GPU_INTERVAL_S,
        }))
    }

    /// Each selected GPU's utilization over a wall-clock window (both ends
    /// included): its sample count, mean and median, keyed by index; `None` when
    /// no sample fell in it. `indices` `None` takes every GPU.
    pub fn window_utilization(
        &self,
        start: f64,
        end: f64,
        indices: Option<&[u32]>,
    ) -> Option<Value> {
        let mut per_gpu: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
        for (time, readings) in lock(&self.samples).iter() {
            if *time < start || *time > end {
                continue;
            }
            for reading in readings {
                if indices.is_none_or(|indices| indices.contains(&reading.index)) {
                    per_gpu
                        .entry(reading.index)
                        .or_default()
                        .push(reading.utilization_gpu_pct);
                }
            }
        }
        if per_gpu.is_empty() {
            return None;
        }
        let gpus: Map<String, Value> = per_gpu
            .into_iter()
            .map(|(index, mut values)| {
                values.sort_unstable();
                let middle = values.len() / 2;
                let median = if values.len() % 2 == 0 {
                    (values[middle - 1] + values[middle]) as f64 / 2.0
                } else {
                    values[middle] as f64
                };
                (
                    index.to_string(),
                    json!({
                        "samples": values.len(),
                        "mean_pct": values.iter().sum::<u64>() as f64 / values.len() as f64,
                        "p50_pct": median,
                    }),
                )
            })
            .collect();
        Some(json!({"interval_s": GPU_INTERVAL_S, "gpus": gpus}))
    }
}

impl Drop for GpuSampler {
    fn drop(&mut self) {
        self.stop();
    }
}
