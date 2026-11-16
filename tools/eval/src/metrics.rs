//! Throughput and latency metrics of a run's measured records.
//!
//! Failed requests stay validation inputs but enter no service-rate or latency
//! population; token metrics further require stream timing. The measured window
//! is the one denominator of every throughput and of `concurrency`.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::record::Record;

/// A percentile by linear interpolation between order statistics (NumPy's
/// default); zero for an empty population.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let (low, high) = (rank.floor() as usize, rank.ceil() as usize);
    sorted[low] + (sorted[high] - sorted[low]) * (rank - low as f64)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// The population standard deviation; zero when empty.
fn std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = mean(values);
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn extreme(values: &[f64], max: bool) -> f64 {
    values
        .iter()
        .copied()
        .reduce(if max { f64::max } else { f64::min })
        .unwrap_or(0.0)
}

/// A population's count, moments, extrema and percentiles, every statistic but
/// the count multiplied by `scale` (seconds to milliseconds); zeros when empty.
pub fn distribution(values: &[f64], scale: f64) -> Value {
    json!({
        "count": values.len(),
        "mean": mean(values) * scale,
        "std": std(values) * scale,
        "min": extreme(values, false) * scale,
        "p50": percentile(values, 50.0) * scale,
        "p90": percentile(values, 90.0) * scale,
        "p95": percentile(values, 95.0) * scale,
        "p99": percentile(values, 99.0) * scale,
        "max": extreme(values, true) * scale,
    })
}

/// The run's metrics over `duration_s`; `tokenizer` adds retokenized output
/// counts beside the server's. Media keys appear only for modalities the
/// successful records produced.
pub fn summarize(
    records: &[&Record],
    duration_s: f64,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Map<String, Value> {
    let successful: Vec<&Record> = records
        .iter()
        .copied()
        .filter(|record| record.success)
        .collect();
    let timed: Vec<&Record> = successful
        .iter()
        .copied()
        .filter(|record| record.token_timing_available)
        .collect();
    let total_input: u64 = successful.iter().map(|record| record.prompt_len).sum();
    let total_output: u64 = successful.iter().map(|record| record.output_len).sum();
    let ttfts: Vec<f64> = timed.iter().map(|record| record.ttft).collect();
    // TPOT spreads the time after the first token over the remaining output
    // tokens, so it includes any delay between the last text event and the
    // final event that closes the latency.
    let tpots: Vec<f64> = timed
        .iter()
        .filter(|record| record.output_len > 1)
        .map(|record| (record.latency - record.ttft) / (record.output_len - 1) as f64)
        .collect();
    let itls: Vec<f64> = timed
        .iter()
        .flat_map(|record| record.itl.iter().copied())
        .collect();
    let e2e: Vec<f64> = successful.iter().map(|record| record.latency).collect();
    let duration = duration_s.max(1e-9);
    let (peak_tokens, peak_requests) = peak_per_second(&timed);
    let optional_ms = |values: &[f64], value: f64| {
        if values.is_empty() {
            Value::Null
        } else {
            (value * 1000.0).into()
        }
    };
    let mut summary = Map::new();
    let mut put = |key: &str, value: Value| {
        summary.insert(key.to_owned(), value);
    };
    put("completed", successful.len().into());
    put("completed_requests", successful.len().into());
    put("total_input_tokens", total_input.into());
    put("total_output_tokens", total_output.into());
    put(
        "request_throughput",
        (successful.len() as f64 / duration).into(),
    );
    put("input_throughput", (total_input as f64 / duration).into());
    put("output_throughput", (total_output as f64 / duration).into());
    put(
        "total_throughput",
        ((total_input + total_output) as f64 / duration).into(),
    );
    put("mean_e2e_latency_ms", (mean(&e2e) * 1000.0).into());
    put("e2e_latency_ms", distribution(&e2e, 1000.0));
    put("mean_ttft_ms", optional_ms(&ttfts, mean(&ttfts)));
    put("ttft_ms", distribution(&ttfts, 1000.0));
    put("mean_tpot_ms", optional_ms(&tpots, mean(&tpots)));
    put("tpot_ms", distribution(&tpots, 1000.0));
    put("mean_itl_ms", optional_ms(&itls, mean(&itls)));
    put("itl_ms", distribution(&itls, 1000.0));
    put("max_itl_ms", optional_ms(&itls, extreme(&itls, true)));
    // The time-averaged number of successful requests in flight.
    put("concurrency", (e2e.iter().sum::<f64>() / duration).into());
    put("max_output_tokens_per_s", peak_tokens.into());
    put("max_concurrent_requests", peak_requests.into());
    put(
        "token_timing_available",
        (!timed.is_empty() && timed.len() == successful.len()).into(),
    );
    put("token_timing_request_count", timed.len().into());

    if let Some(tokenizer) = tokenizer {
        let retokenized: usize = successful
            .iter()
            .map(|record| {
                tokenizer
                    .encode(record.generated_text.as_str(), false)
                    .map_or(0, |encoding| encoding.len())
            })
            .sum();
        put("total_output_tokens_retokenized", retokenized.into());
        put(
            "output_throughput_retokenized",
            (retokenized as f64 / duration).into(),
        );
    }

    let total_images: usize = successful.iter().map(|record| record.images).sum();
    if total_images > 0 {
        let mut latencies: Vec<f64> = successful
            .iter()
            .flat_map(|record| record.image_latencies.iter().copied())
            .collect();
        // Without per-image latencies, each image takes its request's latency.
        if latencies.is_empty() {
            latencies = successful
                .iter()
                .flat_map(|record| std::iter::repeat_n(record.latency, record.images))
                .collect();
        }
        put("completed_images", total_images.into());
        put("images_per_second", (total_images as f64 / duration).into());
        put(
            "images_per_minute",
            (60.0 * total_images as f64 / duration).into(),
        );
        put("image_latency_ms", distribution(&latencies, 1000.0));
        let first: Vec<f64> = successful
            .iter()
            .filter_map(|record| record.first_image_latency)
            .collect();
        if !first.is_empty() {
            put("time_to_first_image_ms", distribution(&first, 1000.0));
        }
    }

    if records.iter().any(|record| record.task == "video") {
        let videos: Vec<&Record> = successful
            .iter()
            .copied()
            .filter(|record| record.decoded_video.is_some())
            .collect();
        let bytes: usize = videos
            .iter()
            .filter_map(|record| record.decoded_video.as_ref())
            .map(|video| video.data.len())
            .sum();
        put("completed_videos", videos.len().into());
        put("videos_per_second", (videos.len() as f64 / duration).into());
        put("media_bytes_per_second", (bytes as f64 / duration).into());
        let latencies: Vec<f64> = videos.iter().map(|record| record.latency).collect();
        put("video_latency_ms", distribution(&latencies, 1000.0));
        let inference: Vec<f64> = successful
            .iter()
            .filter_map(|record| record.server_inference_s)
            .collect();
        if !inference.is_empty() {
            put("server_inference_ms", distribution(&inference, 1000.0));
        }
        let mut stages: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for record in &successful {
            for (name, seconds) in &record.server_stage_s {
                stages.entry(name).or_default().push(*seconds);
            }
        }
        if !stages.is_empty() {
            put(
                "server_stage_ms",
                Value::Object(
                    stages
                        .into_iter()
                        .map(|(name, values)| (name.to_owned(), distribution(&values, 1000.0)))
                        .collect(),
                ),
            );
        }
        let peaks: Vec<f64> = successful
            .iter()
            .filter_map(|record| record.server_peak_storage_mib)
            .collect();
        if !peaks.is_empty() {
            put("server_peak_storage_mib", extreme(&peaks, true).into());
        }
    }
    summary
}

/// The peak one-second token completions and overlapping requests, over
/// one-second buckets from the earliest start: each stamped text event counts
/// as one token at its arrival, and a request counts in every bucket its span
/// touches. The two peaks may come from different buckets.
fn peak_per_second(records: &[&Record]) -> (f64, u64) {
    if records.is_empty() {
        return (0.0, 0);
    }
    let start = records
        .iter()
        .map(|record| record.start_time)
        .fold(f64::MAX, f64::min);
    let end = records
        .iter()
        .map(|record| record.start_time + record.latency)
        .fold(f64::MIN, f64::max);
    let buckets = (end - start).ceil() as usize + 1;
    let mut tokens = vec![0.0f64; buckets];
    let mut requests = vec![0u64; buckets];
    for record in records {
        for time in &record.text_times {
            let index = (time - start).trunc();
            if index >= 0.0 && (index as usize) < buckets {
                tokens[index as usize] += 1.0;
            }
        }
        let first = (record.start_time - start) as usize;
        let last = (record.start_time + record.latency - start) as usize;
        for bucket in requests
            .iter_mut()
            .take((last + 1).min(buckets))
            .skip(first)
        {
            *bucket += 1;
        }
    }
    (
        tokens.into_iter().fold(0.0, f64::max),
        requests.into_iter().max().unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NumPy's `percentile` of `[1, 2, 3, 4]`: 2.5, 3.7 and 3.97.
    #[test]
    fn percentiles_interpolate_linearly() {
        let values = [4.0, 1.0, 3.0, 2.0];
        assert!((percentile(&values, 50.0) - 2.5).abs() < 1e-12);
        assert!((percentile(&values, 90.0) - 3.7).abs() < 1e-12);
        assert!((percentile(&values, 99.0) - 3.97).abs() < 1e-12);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }
}
