//! The pass of a timeline run in which a process was stopped (`check-stop`):
//! case E2 stops the host's Storage Server with `SIGSTOP` and continues it with
//! `SIGCONT`, against a control of the same load without the signals.
//!
//! A stop window runs from a `SIGSTOP` change's signal to the next `SIGCONT` to
//! the same process; the control's window is the same interval of its own load.
//! A token arrives with its text-bearing event, and a gap is the interval between
//! two consecutive tokens of one stream. The items:
//!
//! 1. no failed request;
//! 2. the decode step time is unchanged while stopped, per window: the gaps
//!    whose later token arrived in it have their p50 and p99 within the control's
//!    minimum to maximum, and its longest silence (its longest part in which no
//!    stream received a token) is at most that maximum: a step held up by the
//!    stopped process shows there even when none of its gaps ends in the window;
//! 3. calls are left out only after the headroom, and scheduled again: between
//!    two consecutive scrapes the left-out counter rises only by calls a window
//!    explains, and for a window only when some stream received at least the
//!    headroom's tokens in it. A call left out is reported at most one statistics
//!    period later and read by the next scrape, so window `[a, b]` explains a
//!    rise between scrapes `k - 1` and `k` when `k` was answered at or after `a`
//!    and `k - 1` sent by `b + 2` periods; and every request in flight at a
//!    window's end completed.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};

use crate::check::{Request, bundle, item, metric_total, request_fault, verdict};
use crate::metrics::percentile;

/// The scheduler's statistics period in seconds: each report counts the calls
/// left out since the previous one.
pub const REPORT_PERIOD_S: f64 = 1.0;

/// The daemon's counter of calls left out of a step for late blocks.
pub const COUNTER: &str = "uniserve:scheduler_calls_without_blocks_total";

/// A record's token arrivals, and its stream's end (its last token without a
/// final event), from the load's start.
struct Timed<'a> {
    request: &'a Request,
    text_s: Vec<f64>,
    end_s: Option<f64>,
}

fn timed<'a>(records: &'a [Request], start: f64) -> Vec<Timed<'a>> {
    records
        .iter()
        .map(|request| {
            let text_s: Vec<f64> = request
                .raw
                .get("text_times")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_f64)
                .map(|time| time - start)
                .collect();
            let end_s = request
                .raw
                .get("final_event_time")
                .and_then(Value::as_f64)
                .map(|time| time - start)
                .or_else(|| text_s.last().copied());
            Timed {
                request,
                text_s,
                end_s,
            }
        })
        .collect()
}

/// The run's stop windows.
fn windows(timeline: &Map<String, Value>) -> Result<Vec<(f64, f64)>> {
    let changes: Vec<&Map<String, Value>> = timeline
        .get("changes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .collect();
    for change in &changes {
        ensure!(
            change.get("error").is_none() && change.contains_key("issued_s"),
            "change {} did not apply: {}",
            change
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            change
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("not issued")
        );
    }
    let mut signals: Vec<&Map<String, Value>> = changes
        .into_iter()
        .filter(|change| change.contains_key("signal"))
        .collect();
    signals.sort_by(|a, b| {
        a["issued_s"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&b["issued_s"].as_f64().unwrap_or(0.0))
    });
    let mut windows = Vec::new();
    for (index, stop) in signals.iter().enumerate() {
        if stop["signal"] != "SIGSTOP" {
            continue;
        }
        let continued = signals[index + 1..]
            .iter()
            .find(|change| change["signal"] == "SIGCONT" && change.get("pid") == stop.get("pid"))
            .with_context(|| {
                format!(
                    "{} is not continued by a SIGCONT",
                    stop["name"].as_str().unwrap_or_default()
                )
            })?;
        windows.push((
            stop["issued_s"].as_f64().unwrap_or(0.0),
            continued["issued_s"].as_f64().unwrap_or(0.0),
        ));
    }
    ensure!(!windows.is_empty(), "the run stopped no process");
    Ok(windows)
}

/// The gaps whose later token arrived in the window.
fn gaps(records: &[Timed<'_>], start: f64, end: f64) -> Vec<f64> {
    records
        .iter()
        .flat_map(|record| record.text_s.windows(2))
        .filter(|pair| (start..=end).contains(&pair[1]))
        .map(|pair| pair[1] - pair[0])
        .collect()
}

/// The longest part of the window in which no token arrived.
fn silence(records: &[Timed<'_>], start: f64, end: f64) -> f64 {
    let mut edges: Vec<f64> = records
        .iter()
        .flat_map(|record| record.text_s.iter().copied())
        .filter(|time| (start..=end).contains(time))
        .collect();
    edges.sort_by(f64::total_cmp);
    edges.insert(0, start);
    edges.push(end);
    edges
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f64::MIN, f64::max)
}

fn most_tokens(records: &[Timed<'_>], start: f64, end: f64) -> usize {
    records
        .iter()
        .map(|record| {
            record
                .text_s
                .iter()
                .filter(|time| (start..=end).contains(*time))
                .count()
        })
        .max()
        .unwrap_or(0)
}

/// The requests in flight at a window's end (a token of each before it, its
/// stream not ended by it), how many failed, and the longest time from the end to
/// the next token of one of them.
fn in_flight(records: &[Timed<'_>], end: f64) -> (usize, usize, Option<f64>) {
    let waiting: Vec<&Timed<'_>> = records
        .iter()
        .filter(|record| {
            record.text_s.first().is_some_and(|first| *first < end)
                && record.end_s.is_some_and(|done| done > end)
        })
        .collect();
    let failed = waiting
        .iter()
        .filter(|record| request_fault(record.request).is_some())
        .count();
    let longest = waiting
        .iter()
        .filter_map(|record| {
            record
                .text_s
                .iter()
                .copied()
                .filter(|time| *time > end)
                .reduce(f64::min)
        })
        .map(|next| (next - end) * 1000.0)
        .reduce(f64::max);
    (waiting.len(), failed, longest)
}

/// A set of gaps' count, extremes and percentiles, in milliseconds.
fn spread(gaps: &[f64]) -> Value {
    if gaps.is_empty() {
        return json!({"count": 0, "min_ms": null, "p50_ms": null, "p99_ms": null, "max_ms": null});
    }
    json!({
        "count": gaps.len(),
        "min_ms": gaps.iter().copied().fold(f64::MAX, f64::min) * 1000.0,
        "p50_ms": percentile(gaps, 50.0) * 1000.0,
        "p99_ms": percentile(gaps, 99.0) * 1000.0,
        "max_ms": gaps.iter().copied().fold(f64::MIN, f64::max) * 1000.0,
    })
}

/// `check-stop`'s report of `run` against `control`: `pass`, the windows, each
/// item, and the measures; with the Storage Server's listing after the load, its
/// check failures, and with the replay's output, its first line.
///
/// # Errors
///
/// When a file cannot be read, a bundle is not of a timeline load, a change
/// failed, or the run has no stop window.
pub fn stop_report(
    run: &Path,
    control: &Path,
    headroom_tokens: usize,
    counter: &str,
    listing: Option<&Path>,
    replay: Option<&Path>,
) -> Result<Value> {
    let (run_records, timeline) = bundle(run)?;
    let (control_records, control_timeline) = bundle(control)?;
    let records = timed(&run_records, timeline["load_start"].as_f64().unwrap_or(0.0));
    let controls = timed(
        &control_records,
        control_timeline["load_start"].as_f64().unwrap_or(0.0),
    );
    let windows = windows(&timeline)?;
    let mut items = Map::new();
    items.insert(
        "no_failed_request".into(),
        Value::Object(item(&run_records.iter().collect::<Vec<_>>(), request_fault)),
    );
    let mut measured = Vec::new();
    for &(start, end) in &windows {
        let run_gaps = gaps(&records, start, end);
        let control_gaps = gaps(&controls, start, end);
        let low = control_gaps.iter().copied().reduce(f64::min);
        let high = control_gaps.iter().copied().reduce(f64::max);
        let (p50, p99) = if run_gaps.is_empty() {
            (None, None)
        } else {
            (
                Some(percentile(&run_gaps, 50.0)),
                Some(percentile(&run_gaps, 99.0)),
            )
        };
        let quiet = silence(&records, start, end);
        let within = match (low, high, p50, p99) {
            (Some(low), Some(high), Some(p50), Some(p99)) => {
                (low..=high).contains(&p50) && (low..=high).contains(&p99) && quiet <= high
            }
            _ => false,
        };
        let (count, failed, longest) = in_flight(&records, end);
        measured.push(json!({
            "window_s": [start, end],
            "duration_ms": (end - start) * 1000.0,
            "pass": within,
            "run": spread(&run_gaps),
            "control": spread(&control_gaps),
            "silence_ms": quiet * 1000.0,
            "control_silence_ms": silence(&controls, start, end) * 1000.0,
            "most_tokens": most_tokens(&records, start, end),
            "in_flight": count,
            "in_flight_failed": failed,
            "longest_wait_ms": longest,
        }));
    }
    items.insert(
        "step_time_unchanged".into(),
        json!({"pass": measured.iter().all(|window| window["pass"] == true), "windows": measured}),
    );
    items.insert(
        "left_out_after_headroom".into(),
        left_out(run, &timeline, &measured, counter, headroom_tokens)?,
    );
    let pass = items.values().all(|item| item["pass"] == true);
    let mut report = json!({
        "run": run.display().to_string(),
        "control": control.display().to_string(),
        "pass": pass,
        "windows_s": windows.iter().map(|(start, end)| json!([start, end])).collect::<Vec<_>>(),
        "headroom_tokens": headroom_tokens,
        "counter": counter,
        "items": items,
        "measures": {
            "run": measures(&records, &windows),
            "control": measures(&controls, &windows),
        },
    });
    if let Some(listing) = listing {
        let text = String::from_utf8_lossy(&std::fs::read(listing)?).into_owned();
        let pattern = regex::Regex::new(r"(\d+) check failures")?;
        report["check_failures"] = pattern
            .captures_iter(&text)
            .filter_map(|captures| captures[1].parse::<u64>().ok())
            .collect::<Vec<_>>()
            .into();
    }
    if let Some(replay) = replay {
        let text = String::from_utf8_lossy(&std::fs::read(replay)?).into_owned();
        report["replay"] = text.lines().next().unwrap_or_default().into();
    }
    Ok(report)
}

/// Item 3, from the run's scrapes and the windows.
fn left_out(
    run: &Path,
    timeline: &Map<String, Value>,
    windows: &[Value],
    counter: &str,
    headroom: usize,
) -> Result<Value> {
    let start = timeline["load_start"].as_f64().unwrap_or(0.0);
    let text = std::fs::read_to_string(run.join("scrapes.jsonl"))?;
    let mut scrapes: Vec<Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    scrapes.sort_by_key(|scrape| scrape["tick"].as_u64().unwrap_or(0));
    let mut readings: indexmap::IndexMap<String, Vec<(f64, f64, f64)>> = indexmap::IndexMap::new();
    for scrape in &scrapes {
        if let Some(value) = metric_total(scrape["text"].as_str(), counter) {
            readings
                .entry(scrape["origin"].as_str().unwrap_or_default().to_owned())
                .or_default()
                .push((
                    scrape["sent"].as_f64().unwrap_or(0.0) - start,
                    scrape["received"].as_f64().unwrap_or(0.0) - start,
                    value,
                ));
        }
    }
    let (mut unexplained, mut before_headroom) = (Vec::new(), Vec::new());
    let mut rises = vec![0.0f64; windows.len()];
    let mut total = 0.0;
    for (origin, series) in &readings {
        total += series[series.len() - 1].2 - series[0].2;
        for pair in series.windows(2) {
            let (previous, current) = (pair[0], pair[1]);
            let rise = current.2 - previous.2;
            if rise <= 0.0 {
                continue;
            }
            let place = format!(
                "{origin}: +{} between the scrapes sent at {:.3} s and answered at {:.3} s",
                crate::pipeline::general(rise),
                previous.0,
                current.1
            );
            let explaining = windows.iter().position(|window| {
                current.1 >= window["window_s"][0].as_f64().unwrap_or(0.0)
                    && previous.0
                        <= window["window_s"][1].as_f64().unwrap_or(0.0) + 2.0 * REPORT_PERIOD_S
            });
            let Some(index) = explaining else {
                unexplained.push(place);
                continue;
            };
            rises[index] += rise;
            if windows[index]["most_tokens"].as_u64().unwrap_or(0) < headroom as u64 {
                before_headroom.push(place);
            }
        }
    }
    let in_flight: u64 = windows
        .iter()
        .map(|window| window["in_flight"].as_u64().unwrap_or(0))
        .sum();
    let in_flight_failed: u64 = windows
        .iter()
        .map(|window| window["in_flight_failed"].as_u64().unwrap_or(0))
        .sum();
    let mut faults: Vec<String> = unexplained
        .iter()
        .chain(&before_headroom)
        .cloned()
        .collect();
    if readings.is_empty() {
        faults.push(format!("no scrape reports {counter}"));
    }
    Ok(json!({
        "pass": faults.is_empty() && in_flight_failed == 0,
        "scrapes": scrapes.len(),
        "total_rise": total,
        "rises_by_window": rises,
        "unexplained": unexplained,
        "before_headroom": before_headroom,
        "in_flight": in_flight,
        "in_flight_failed": in_flight_failed,
        "first": faults.first(),
    }))
}

/// One run's measures over the stop windows.
fn measures(records: &[Timed<'_>], windows: &[(f64, f64)]) -> Value {
    let longest = records
        .iter()
        .flat_map(|record| record.text_s.windows(2))
        .map(|pair| pair[1] - pair[0])
        .fold(0.0, f64::max);
    let per_window: Vec<Value> = windows
        .iter()
        .map(|&(start, end)| {
            let submitted: Vec<&Timed<'_>> = records
                .iter()
                .filter(|record| (start..=end).contains(&record.request.submit_s))
                .collect();
            let ttfts: Vec<f64> = submitted
                .iter()
                .filter_map(|record| Some(record.request.first_s? - record.request.send_s))
                .collect();
            let at = |p: f64| (!ttfts.is_empty()).then(|| percentile(&ttfts, p) * 1000.0);
            json!({"submitted": submitted.len(), "ttft_p50_ms": at(50.0), "ttft_p99_ms": at(99.0)})
        })
        .collect();
    json!({"max_gap_ms": longest * 1000.0, "requests": records.len(), "windows": per_window})
}

fn ms(value: &Value) -> String {
    value
        .as_f64()
        .map_or_else(|| "none".into(), |value| format!("{value:.3}"))
}

fn spread_text(spread: &Value) -> String {
    let mut parts = vec![spread["count"].to_string()];
    parts.extend(
        ["min_ms", "p50_ms", "p99_ms", "max_ms"]
            .iter()
            .map(|key| ms(&spread[*key])),
    );
    parts.join(", ")
}

/// A `check-stop` report as the run's `result.md`.
pub fn render(report: &Value) -> String {
    let run = report["run"].as_str().unwrap_or_default();
    let name = Path::new(run)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut lines = vec![
        format!("# E2, the process stopped: {name}"),
        String::new(),
        format!(
            "Run `{run}` against the control `{}`. Verdict: **{}**.",
            report["control"].as_str().unwrap_or_default(),
            verdict(report["pass"] == true)
        ),
        String::new(),
        "| Item | Verdict | Detail |".into(),
        "| --- | --- | --- |".into(),
    ];
    let items = &report["items"];
    for (key, title) in [
        ("no_failed_request", "1. No failed request"),
        (
            "step_time_unchanged",
            "2. Decode step time within the control's",
        ),
        (
            "left_out_after_headroom",
            "3. Left out only after the headroom",
        ),
    ] {
        let item = &items[key];
        let detail = match key {
            "no_failed_request" => {
                let mut detail = format!("{} of {} requests fail", item["count"], item["of"]);
                if let Some(first) = item["first"].as_str() {
                    detail.push_str(&format!("; first: {first}"));
                }
                detail
            }
            "step_time_unchanged" => item["windows"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|window| {
                    format!(
                        "window at {:.3} s: p50 {}, p99 {}, silence {:.3} ms, against {} to {}",
                        window["window_s"][0].as_f64().unwrap_or(0.0),
                        ms(&window["run"]["p50_ms"]),
                        ms(&window["run"]["p99_ms"]),
                        window["silence_ms"].as_f64().unwrap_or(0.0),
                        ms(&window["control"]["min_ms"]),
                        ms(&window["control"]["max_ms"])
                    )
                })
                .collect::<Vec<_>>()
                .join("; "),
            _ => {
                let mut detail = format!(
                    "{} rises no window explains, {} for a window shorter than the headroom; {} of {} requests in flight at a window's end failed",
                    item["unexplained"].as_array().map_or(0, Vec::len),
                    item["before_headroom"].as_array().map_or(0, Vec::len),
                    item["in_flight_failed"],
                    item["in_flight"]
                );
                if let Some(first) = item["first"].as_str() {
                    detail.push_str(&format!("; first: {first}"));
                }
                detail
            }
        };
        lines.push(format!(
            "| {title} | {} | {detail} |",
            verdict(item["pass"] == true)
        ));
    }
    lines.extend([
        String::new(),
        "Per window (times in seconds from the load's start, gaps in ms):".into(),
        String::new(),
        "| Window | Duration (ms) | Run gaps: n, min, p50, p99, max | Control gaps: n, min, p50, p99, max | Silence run / control (ms) | Most tokens of a stream | In flight at end | Longest wait after end (ms) | Counter rise explained |".into(),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |".into(),
    ]);
    let left = &items["left_out_after_headroom"];
    let rises = left["rises_by_window"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (window, rise) in items["step_time_unchanged"]["windows"]
        .as_array()
        .into_iter()
        .flatten()
        .zip(&rises)
    {
        lines.push(format!(
            "| {:.3} to {:.3} | {:.1} | {} | {} | {:.3} / {:.3} | {} | {} | {} | {} |",
            window["window_s"][0].as_f64().unwrap_or(0.0),
            window["window_s"][1].as_f64().unwrap_or(0.0),
            window["duration_ms"].as_f64().unwrap_or(0.0),
            spread_text(&window["run"]),
            spread_text(&window["control"]),
            window["silence_ms"].as_f64().unwrap_or(0.0),
            window["control_silence_ms"].as_f64().unwrap_or(0.0),
            window["most_tokens"],
            window["in_flight"],
            ms(&window["longest_wait_ms"]),
            crate::pipeline::general(rise.as_f64().unwrap_or(0.0)),
        ));
    }
    let (run_measures, control_measures) =
        (&report["measures"]["run"], &report["measures"]["control"]);
    lines.extend([
        String::new(),
        format!(
            "Counter `{}`: rose by {} over the run, in {} scrapes; headroom {} tokens.",
            report["counter"].as_str().unwrap_or_default(),
            crate::pipeline::general(left["total_rise"].as_f64().unwrap_or(0.0)),
            left["scrapes"],
            report["headroom_tokens"]
        ),
        String::new(),
        "| Measure (no pass line) | Run | Control |".into(),
        "| --- | --- | --- |".into(),
        format!(
            "| Longest gap between two tokens of any stream (ms) | {:.3} | {:.3} |",
            run_measures["max_gap_ms"].as_f64().unwrap_or(0.0),
            control_measures["max_gap_ms"].as_f64().unwrap_or(0.0)
        ),
    ]);
    let pairs = run_measures["windows"]
        .as_array()
        .into_iter()
        .flatten()
        .zip(control_measures["windows"].as_array().into_iter().flatten());
    for (index, (mine, theirs)) in pairs.enumerate() {
        for (key, title) in [("ttft_p50_ms", "TTFT p50"), ("ttft_p99_ms", "TTFT p99")] {
            lines.push(format!(
                "| {title}, requests submitted in window {} (ms) | {} | {} |",
                index + 1,
                ms(&mine[key]),
                ms(&theirs[key])
            ));
        }
        lines.push(format!(
            "| Requests submitted in window {} | {} | {} |",
            index + 1,
            mine["submitted"],
            theirs["submitted"]
        ));
    }
    lines.push(format!(
        "| Requests | {} | {} |",
        run_measures["requests"], control_measures["requests"]
    ));
    if let Some(failures) = report.get("check_failures") {
        lines.extend([
            String::new(),
            format!("Storage Server check failures (its listing after the load): {failures}."),
        ]);
    }
    if let Some(replay) = report.get("replay").and_then(Value::as_str) {
        lines.extend([String::new(), format!("Trace replay: `{replay}`.")]);
    }
    lines.join("\n") + "\n"
}
