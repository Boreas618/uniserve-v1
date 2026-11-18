//! What the servers and the GPUs report while a point runs.
//!
//! A server's observations are its deployment's observation store, read
//! through its admin socket with `/admin/query` ([`crate::admin`]). Every
//! counter's total (series ending in `_total`, and a histogram's observation
//! count and sum as `_count` and `_sum`) is read just before the measured window
//! opens and after it closes; the window's summary is how far each advanced (a
//! gauge's difference between two instants means nothing). The store sums a
//! counter's increases over its whole retained timeline once each, so a total
//! read twice differs by what was recorded between the reads. A timeline load
//! also keeps every counter and gauge row the store bucketed from its start on.
//! The GPUs are sampled through `nvidia-smi`, which reports every GPU it sees and
//! device-wide memory, so its readings include processes other than the server
//! under test: memory in MiB, utilization in percent.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use serde_json::{Map, Value, json};

/// Every counter's total and every histogram's observation count and sum, by
/// series ([`series`]).
const TOTALS: &str = "SELECT name, labels, sum(value) AS total, \
     sum(histogram['count']) AS count, sum(histogram['sum']) AS sum \
     FROM stats WHERE ends_with(name, '_total') OR histogram IS NOT NULL \
     GROUP BY name, labels";

/// A series' key: its metric's name and its labels sorted by key, as the
/// Prometheus text format writes a series (`name{key="value",...}`; a series
/// without labels is its name alone).
pub fn series(name: &str, labels: &Map<String, Value>) -> String {
    if labels.is_empty() {
        return name.to_owned();
    }
    let mut pairs: Vec<(&String, String)> = labels
        .iter()
        .map(|(key, value)| {
            (
                key,
                value
                    .as_str()
                    .map_or_else(|| value.to_string(), str::to_owned),
            )
        })
        .collect();
    pairs.sort();
    let inner: Vec<String> = pairs
        .into_iter()
        .map(|(key, value)| format!("{key}={}", Value::String(value)))
        .collect();
    format!("{name}{{{}}}", inner.join(","))
}

/// The metric name of a series key.
pub fn metric(series: &str) -> &str {
    series.split('{').next().unwrap_or_default()
}

/// The totals the store at `socket` holds now, by series: each counter's
/// (`name`), each histogram's observation count (`name_count`) and sum
/// (`name_sum`).
///
/// # Errors
///
/// When the query fails.
pub async fn totals(socket: &Path) -> Result<Map<String, Value>> {
    let mut totals = Map::new();
    for row in crate::admin::query(socket, TOTALS).await? {
        let name = row["name"].as_str().unwrap_or_default();
        let labels = row["labels"].as_object().cloned().unwrap_or_default();
        if let (Some(count), Some(sum)) = (row["count"].as_f64(), row["sum"].as_f64()) {
            totals.insert(series(&format!("{name}_count"), &labels), count.into());
            totals.insert(series(&format!("{name}_sum"), &labels), sum.into());
        } else if let Some(total) = row["total"].as_f64() {
            totals.insert(series(name, &labels), total.into());
        }
    }
    Ok(totals)
}

/// How far each total advanced between two readings; a series absent from the
/// first started at zero, and one that did not advance (or by a non-finite
/// amount) is left out.
pub fn counter_deltas(
    before: &Map<String, Value>,
    after: &Map<String, Value>,
) -> Map<String, Value> {
    after
        .iter()
        .filter_map(|(key, value)| {
            let delta = value.as_f64()? - before.get(key).and_then(Value::as_f64).unwrap_or(0.0);
            (delta != 0.0 && delta.is_finite()).then(|| (key.clone(), Value::from(delta)))
        })
        .collect()
}

/// The window's counter deltas; unavailable when either reading failed (or
/// the server has no admin socket).
pub fn window_summary(
    before: Option<&Map<String, Value>>,
    after: Option<&Map<String, Value>>,
) -> Value {
    match (before, after) {
        (Some(before), Some(after)) => {
            json!({"available": true, "counter_deltas": counter_deltas(before, after)})
        }
        _ => json!({"available": false, "counter_deltas": {}}),
    }
}

/// The counter and gauge rows the store at `socket` bucketed from `from_ns`
/// (nanoseconds since the Unix epoch, on the host's clock) on, in time order:
/// each row's bucket start (`time`, nanoseconds), metric, labels, value (a
/// counter's increase within the bucket, a gauge's latest reading in it) and
/// configuration revision.
///
/// # Errors
///
/// When the query fails.
pub async fn buckets(socket: &Path, from_ns: i64) -> Result<Vec<Value>> {
    let sql = format!(
        "SELECT CAST(time AS BIGINT) AS time, name, labels, value, revision FROM stats \
         WHERE histogram IS NULL AND CAST(time AS BIGINT) >= {from_ns} ORDER BY time, name"
    );
    crate::admin::query(socket, &sql).await
}

#[expect(
    clippy::expect_used,
    reason = "a poisoned lock means a panic elsewhere, which ends the run"
)]
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().expect("the telemetry's lock")
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
