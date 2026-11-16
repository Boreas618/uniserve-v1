//! The pass of a timeline run with changes against its control (`check-change`),
//! and of a run of faults against its rows served alone (`check-faults`).
//!
//! `check-change` reads two completed bundles of timeline points: the run, whose
//! changes acted on the deployment, and its control, the same load and timeline
//! without them. Its items:
//!
//! 1. no failed request: each answered 200 and ended at `length` after the server
//!    counted exactly its requested tokens;
//! 2. no closed connection: none ended in a transport failure or in a stream
//!    without its terminal event;
//! 3. no token lost or repeated: each stream carried exactly its requested tokens,
//!    agreeing by the near-tie rule ([`near_tie_mismatch`]) with the first
//!    successful output of its row in the control of a configuration it may have
//!    ended on;
//! 4. new requests admitted throughout: each request submitted in the change's
//!    window had its first token by the window's end plus the control's
//!    99th-percentile time to first token;
//! 5. the buffer pool's invariants held: the check-failure counter reads 0 in
//!    every scrape after the window, and the replay verdict is `pass`;
//! 6. with the switch and a block size named (case E4): the target's first
//!    admission of each prompt the predecessor served before the switch reports
//!    as cached at least the prompt's whole blocks less its last token.
//!
//! The change's window runs from the first change's issue to the latest of every
//! change's end and every admin event recorded from that issue on. A request is
//! submitted when the load issued it, has its first token with the first chunk
//! that carried a token id, and ends with its last event; one a cancel change
//! disconnected is counted and left out of every item. A request ended on the
//! configuration that served when it was submitted, unless a change to another
//! overlapped its life: an epoch's target admits every request from its
//! promotion while the predecessor goes on with those it holds, so a request in
//! flight across the switch ended on either side, which the records do not tell
//! apart; it agrees when it agrees with a control of a configuration it may have
//! ended on.
//!
//! `check-faults` (case E6) holds every request not cancelled to its row's tokens
//! served alone, and item 5.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};

use crate::metrics::percentile;
use crate::record::CANCELLED;

/// One request of a bundle, timed from its load's start.
#[derive(Debug, Clone)]
pub struct Request {
    pub index: usize,
    pub row: String,
    pub raw: Map<String, Value>,
    pub submit_s: f64,
    pub send_s: f64,
    pub token_s: Vec<f64>,
    pub first_s: Option<f64>,
    pub end_s: f64,
}

impl Request {
    fn field(&self, key: &str) -> &Value {
        self.raw.get(key).unwrap_or(&Value::Null)
    }

    fn classifier(&self) -> &str {
        self.field("classifier").as_str().unwrap_or_default()
    }

    fn success(&self) -> bool {
        self.field("success").as_bool() == Some(true)
    }

    fn token_ids(&self) -> &[Value] {
        self.field("token_ids")
            .as_array()
            .map_or(&[], Vec::as_slice)
    }

    fn requested(&self) -> u64 {
        self.field("requested_output_len").as_u64().unwrap_or(0)
    }

    /// The token ids of the candidates at a position.
    fn candidates(&self, position: usize) -> Vec<&Value> {
        self.field("top_logprobs")
            .as_array()
            .and_then(|positions| positions.get(position))
            .and_then(Value::as_array)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter_map(|pair| pair.as_array()?.first())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// A bundle's records, timed from `start` (the run's clock at the load's start).
pub fn records(path: &Path, start: f64) -> Result<Vec<Request>> {
    let text = std::fs::read_to_string(path.join("requests.jsonl"))
        .with_context(|| format!("reading {}", path.display()))?;
    let mut requests = Vec::new();
    for (index, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let raw: Map<String, Value> = serde_json::from_str(line)?;
        let number = |key: &str| raw.get(key).and_then(Value::as_f64);
        let sent = number("client_send_time").context("a record's client_send_time")?;
        let submitted = number("scheduled_time").unwrap_or(sent);
        let tokens: Vec<f64> = raw
            .get("token_times")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_f64)
            .map(|time| time - start)
            .collect();
        let first = tokens
            .first()
            .copied()
            .or_else(|| number("first_text_time").map(|time| time - start));
        let end = number("final_event_time")
            .map(|time| time - start)
            .unwrap_or_else(|| tokens.iter().copied().fold(submitted - start, f64::max));
        let row = raw
            .get("example")
            .and_then(|example| example.get("id"))
            .and_then(Value::as_str)
            .context("a record's example id")?
            .to_owned();
        requests.push(Request {
            index,
            row,
            submit_s: submitted - start,
            send_s: sent - start,
            token_s: tokens,
            first_s: first,
            end_s: end,
            raw,
        });
    }
    Ok(requests)
}

/// A timeline bundle's records and its timeline.
pub(crate) fn bundle(path: &Path) -> Result<(Vec<Request>, Map<String, Value>)> {
    let timeline: Map<String, Value> =
        serde_json::from_slice(&std::fs::read(path.join("timeline.json"))?)?;
    let start = timeline
        .get("load_start")
        .and_then(Value::as_f64)
        .with_context(|| format!("{} is not a bundle of a timeline load", path.display()))?;
    Ok((records(path, start)?, timeline))
}

fn changes_of(timeline: &Map<String, Value>) -> Vec<&Map<String, Value>> {
    timeline
        .get("changes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .collect()
}

fn number(entry: &Map<String, Value>, key: &str) -> Option<f64> {
    entry.get(key).and_then(Value::as_f64)
}

/// The change's window, in seconds from the load's start.
fn window(timeline: &Map<String, Value>) -> Result<(f64, f64)> {
    let changes = changes_of(timeline);
    let start = changes
        .iter()
        .filter_map(|change| number(change, "issued_s"))
        .reduce(f64::min)
        .context("the run issued no change")?;
    let mut end = changes
        .iter()
        .filter_map(|change| number(change, "ended_s"))
        .fold(f64::MIN, f64::max);
    for event in timeline
        .get("events")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(at) = event.get("at_s").and_then(Value::as_f64)
            && at >= start
        {
            end = end.max(at);
        }
    }
    Ok((start, end))
}

/// An item: `fault` (a record's failure, or `None`) applied to every record.
pub(crate) fn item(
    records: &[&Request],
    fault: impl Fn(&Request) -> Option<String>,
) -> Map<String, Value> {
    let failing: Vec<(&Request, String)> = records
        .iter()
        .filter_map(|record| Some((*record, fault(record)?)))
        .collect();
    let first = failing.first().map(|(record, reason)| {
        format!(
            "request {} (row {}, submitted at {:.3} s): {reason}",
            record.index, record.row, record.submit_s
        )
    });
    let mut item = Map::new();
    item.insert("pass".into(), failing.is_empty().into());
    item.insert("count".into(), failing.len().into());
    item.insert("of".into(), records.len().into());
    item.insert("first".into(), first.into());
    item
}

/// How a request failed: anything but 200, `length` and the server's count of
/// exactly the requested tokens.
pub(crate) fn request_fault(record: &Request) -> Option<String> {
    let counted = (record.field("output_len_source").as_str() == Some("server_usage"))
        .then(|| record.field("output_len").as_u64())
        .flatten();
    let requested = record.requested();
    if record.success()
        && record.field("status_code").as_u64() == Some(200)
        && record.field("finish_reason").as_str() == Some("length")
        && requested > 0
        && counted == Some(requested)
    {
        return None;
    }
    Some(format!(
        "{}, status {}, finish {}, {} tokens counted",
        record.classifier(),
        display(record.field("status_code")),
        display(record.field("finish_reason")),
        counted.map_or_else(|| "None".to_owned(), |count| count.to_string()),
    ))
}

/// A JSON value as Python renders it in text: `None` for null.
fn display(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Where two greedy outputs of one prompt disagree, if they do: vLLM's
/// `check_logprobs_close` with `near_tie_tol = 0`. They agree when their ids are
/// equal up to the first position where they differ and, there, each side's
/// token is among the other's candidates; later positions, and positions past
/// the shorter output, are not compared.
pub fn near_tie_mismatch(first: &Request, second: &Request) -> Option<usize> {
    for (position, (one, other)) in first.token_ids().iter().zip(second.token_ids()).enumerate() {
        if one == other {
            continue;
        }
        let near = second.candidates(position).contains(&one)
            && first.candidates(position).contains(&other);
        return if near { None } else { Some(position) };
    }
    None
}

/// How a request's tokens fail every one of `references`: it must stream
/// exactly its requested tokens, which must agree with its row's output in one.
fn token_fault(
    record: &Request,
    references: &[(String, &BTreeMap<String, Request>)],
) -> Option<String> {
    let (count, wanted) = (record.token_ids().len() as u64, record.requested());
    if count != wanted {
        return Some(format!("{count} tokens streamed of {wanted}"));
    }
    let mut faults = Vec::new();
    for (label, outputs) in references {
        match outputs.get(&record.row) {
            None => faults.push(format!("{label} has no output of its row")),
            Some(reference) => {
                // Agreeing with one reference is no fault: `None` returns then.
                let position = near_tie_mismatch(record, reference)?;
                faults.push(format!("disagrees with {label} at token {position}"));
            }
        }
    }
    Some(faults.join("; "))
}

/// Each row's first successful output that carried tokens.
fn first_outputs(records: &[Request]) -> BTreeMap<String, Request> {
    let mut outputs = BTreeMap::new();
    for record in records {
        if record.success() && !record.token_ids().is_empty() {
            outputs
                .entry(record.row.clone())
                .or_insert_with(|| record.clone());
        }
    }
    outputs
}

/// The applied changes to another configuration, by their end: each its issue,
/// its end and the configuration it moved to. A failed change moves nothing.
fn moves(timeline: &Map<String, Value>) -> Vec<(f64, f64, String)> {
    let mut moves: Vec<(f64, f64, String)> = changes_of(timeline)
        .into_iter()
        .filter(|change| change.get("error").is_none())
        .filter_map(|change| {
            let name = change
                .get("configuration")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())?;
            Some((
                number(change, "issued_s")?,
                number(change, "ended_s")?,
                name.to_owned(),
            ))
        })
        .collect();
    moves.sort_by(|a, b| a.1.total_cmp(&b.1));
    moves
}

/// The configurations a request may have ended on: the one serving when it was
/// submitted (the target of the last move ended by then; `None`, the run's own,
/// before any), then each target of a move that overlapped its life.
fn ended_on(record: &Request, moves: &[(f64, f64, String)]) -> Vec<Option<String>> {
    let mut serving = None;
    for (_, end, name) in moves {
        if *end <= record.submit_s {
            serving = Some(name.clone());
        }
    }
    let mut names = vec![serving];
    for (issued, end, name) in moves {
        let candidate = Some(name.clone());
        if *end > record.submit_s && *issued < record.end_s && !names.contains(&candidate) {
            names.push(candidate);
        }
    }
    names
}

fn control_label(configuration: Option<&str>) -> String {
    configuration.map_or_else(
        || "the control".into(),
        |name| format!("the control of {name}"),
    )
}

/// The sum of a counter's series in a scrape, if it has one.
pub(crate) fn metric_total(text: Option<&str>, name: &str) -> Option<f64> {
    let samples = crate::telemetry::samples(text?);
    let values: Vec<f64> = samples
        .iter()
        .filter(|(key, _)| key.split('{').next() == Some(name))
        .map(|(_, value)| *value)
        .collect();
    (!values.is_empty()).then(|| values.iter().sum())
}

/// Item 5: the check-failure count 0 in every scrape after the window (one at
/// least must report it), and the replay's `pass`.
fn invariants(run: &Path, check_metric: &str, replay: &Path) -> Result<Map<String, Value>> {
    let metrics: Value = serde_json::from_slice(&std::fs::read(run.join("server_metrics.json"))?)?;
    let failures: Map<String, Value> = metrics["after"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(origin, text)| {
            (
                origin.clone(),
                metric_total(text.as_str(), check_metric).into(),
            )
        })
        .collect();
    let verdict: Value = serde_json::from_slice(
        &std::fs::read(replay).with_context(|| format!("reading {}", replay.display()))?,
    )?;
    let verdict = verdict.get("verdict").cloned().unwrap_or(Value::Null);
    let mut item = Map::new();
    item.insert(
        "pass".into(),
        (!failures.is_empty()
            && failures.values().all(|count| count.as_f64() == Some(0.0))
            && verdict == "pass")
            .into(),
    );
    item.insert("check_failures".into(), Value::Object(failures));
    item.insert("replay".into(), verdict);
    Ok(item)
}

/// Item 6, case E4's: what the target found of the predecessor's. The
/// predecessor served a prompt when a request of its row received a token
/// before the switch was issued; the target serves from the switch's end until
/// the next change's issue, and its first request of each such row must report
/// as cached at least `prompt_len - 1` rounded down to whole blocks (an
/// admission computes a prompt's last token whatever is cached). The item fails
/// when no such admission happened: it has no evidence.
fn cached_across_switch(
    records: &[&Request],
    timeline: &Map<String, Value>,
    name: &str,
    block_size: u64,
) -> Result<Map<String, Value>> {
    let changes: Vec<(&str, f64, f64)> = changes_of(timeline)
        .into_iter()
        .filter_map(|change| {
            Some((
                change.get("name")?.as_str()?,
                number(change, "issued_s")?,
                number(change, "ended_s")?,
            ))
        })
        .collect();
    let Some(&(_, switch_issued, switch_ended)) =
        changes.iter().find(|(change, _, _)| *change == name)
    else {
        bail!("the run applied no change {name:?}");
    };
    let target_end = changes
        .iter()
        .map(|(_, issued, _)| *issued)
        .filter(|issued| *issued > switch_issued)
        .fold(f64::INFINITY, f64::min);
    let served: BTreeSet<&str> = records
        .iter()
        .filter(|record| record.first_s.is_some_and(|first| first < switch_issued))
        .map(|record| record.row.as_str())
        .collect();
    let mut by_submission: Vec<&&Request> = records.iter().collect();
    by_submission.sort_by(|a, b| a.submit_s.total_cmp(&b.submit_s));
    let mut firsts: BTreeMap<&str, &Request> = BTreeMap::new();
    for record in by_submission {
        if served.contains(record.row.as_str())
            && switch_ended <= record.submit_s
            && record.submit_s < target_end
        {
            firsts.entry(record.row.as_str()).or_insert(record);
        }
    }
    let mut admissions: Vec<&Request> = firsts.into_values().collect();
    admissions.sort_by_key(|record| record.index);
    let mut item = item(&admissions, |record| {
        let prompt = record.field("prompt_len").as_u64().unwrap_or(0);
        let whole = prompt.saturating_sub(1) / block_size * block_size;
        let cached = record.field("cached_prompt_tokens").as_u64();
        if cached.is_some_and(|cached| cached >= whole) {
            return None;
        }
        Some(format!(
            "{} prompt tokens cached, below the {whole} of its blocks",
            cached.map_or_else(|| "None".to_owned(), |cached| cached.to_string())
        ))
    });
    item.insert("served".into(), served.len().into());
    if admissions.is_empty() {
        item.insert("pass".into(), false.into());
        item.insert(
            "first".into(),
            "no prompt the predecessor served was admitted on the target".into(),
        );
    }
    Ok(item)
}

/// The measures reported beside the items, of one run over `window`.
fn measures(records: &[&Request], (start, end): (f64, f64)) -> Value {
    let (mut gaps, mut window_gaps) = (vec![0.0f64], vec![0.0f64]);
    for record in records {
        for pair in record.token_s.windows(2) {
            gaps.push(pair[1] - pair[0]);
            if pair[1] >= start && pair[0] <= end {
                window_gaps.push(pair[1] - pair[0]);
            }
        }
    }
    let submitted: Vec<&&Request> = records
        .iter()
        .filter(|record| (start..=end).contains(&record.submit_s))
        .collect();
    let ttfts: Vec<f64> = submitted
        .iter()
        .filter_map(|record| Some(record.first_s? - record.send_s))
        .collect();
    json!({
        "max_gap_ms": gaps.iter().copied().fold(0.0, f64::max) * 1000.0,
        "max_gap_window_ms": window_gaps.iter().copied().fold(0.0, f64::max) * 1000.0,
        "ttft_p50_ms": percentile(&ttfts, 50.0) * 1000.0,
        "ttft_p99_ms": percentile(&ttfts, 99.0) * 1000.0,
        "window_requests": submitted.len(),
        "requests": records.len(),
    })
}

/// Each change with its times and any failure.
fn change_list(timeline: &Map<String, Value>) -> Value {
    Value::Array(
        changes_of(timeline)
            .into_iter()
            .map(|change| {
                json!({
                    "name": change.get("name"),
                    "issued_s": change.get("issued_s"),
                    "ended_s": change.get("ended_s"),
                    "error": change.get("error"),
                })
            })
            .collect(),
    )
}

/// `check-change`'s report of `run` against `control`: `pass`, the window, each
/// item, the cancelled requests, the measures and the changes. `configurations`
/// names, per configuration a change moves to, its control; `switch` with
/// `block_size` adds item 6.
///
/// # Errors
///
/// When a file cannot be read, a bundle is not of a timeline load, the run
/// issued no change, or a change moves to a configuration without a control.
pub fn change_report(
    run: &Path,
    control: &Path,
    check_metric: &str,
    replay: &Path,
    configurations: &BTreeMap<String, PathBuf>,
    switch: Option<(&str, u64)>,
) -> Result<Value> {
    let (all, timeline) = bundle(run)?;
    let (controls, _) = bundle(control)?;
    let (start, end) = window(&timeline)?;
    let cancelled = all
        .iter()
        .filter(|record| record.classifier() == CANCELLED)
        .count();
    let requests: Vec<&Request> = all
        .iter()
        .filter(|record| record.classifier() != CANCELLED)
        .collect();
    let mut items = Map::new();
    items.insert(
        "no_failed_request".into(),
        Value::Object(item(&requests, request_fault)),
    );
    items.insert(
        "no_closed_connection".into(),
        Value::Object(item(&requests, |record| {
            ["transport_failure", "response_missing_terminal"]
                .contains(&record.classifier())
                .then(|| {
                    format!(
                        "{}: {}",
                        record.classifier(),
                        display(record.field("error"))
                    )
                })
        })),
    );
    // Per configuration (`None` for the run's own), each row's reference.
    let mut references: BTreeMap<Option<String>, BTreeMap<String, Request>> = BTreeMap::new();
    references.insert(None, first_outputs(&controls));
    for (name, path) in configurations {
        references.insert(Some(name.clone()), first_outputs(&bundle(path)?.0));
    }
    let moves = moves(&timeline);
    let unknown: BTreeSet<&String> = moves
        .iter()
        .map(|(_, _, name)| name)
        .filter(|name| !references.contains_key(&Some((*name).clone())))
        .collect();
    ensure!(
        unknown.is_empty(),
        "the run's changes move the deployment to {unknown:?}, of which no control is given"
    );
    items.insert(
        "no_token_lost_or_repeated".into(),
        Value::Object(item(&requests, |record| {
            let labelled: Vec<(String, &BTreeMap<String, Request>)> = ended_on(record, &moves)
                .into_iter()
                .map(|name| (control_label(name.as_deref()), &references[&name]))
                .collect();
            token_fault(record, &labelled)
        })),
    );
    let ttfts: Vec<f64> = controls
        .iter()
        .filter(|record| record.success())
        .filter_map(|record| Some(record.first_s? - record.send_s))
        .collect();
    let deadline = end + percentile(&ttfts, 99.0);
    let submitted: Vec<&Request> = requests
        .iter()
        .copied()
        .filter(|record| (start..=end).contains(&record.submit_s))
        .collect();
    let mut admitted = item(&submitted, |record| {
        if record.first_s.is_some_and(|first| first <= deadline) {
            return None;
        }
        Some(format!("first token at {}", seconds(record.first_s)))
    });
    admitted.insert("deadline_s".into(), deadline.into());
    items.insert("admitted_throughout".into(), Value::Object(admitted));
    items.insert(
        "invariants_hold".into(),
        Value::Object(invariants(run, check_metric, replay)?),
    );
    if let Some((name, block_size)) = switch {
        items.insert(
            "cached_across_switch".into(),
            Value::Object(cached_across_switch(
                &requests, &timeline, name, block_size,
            )?),
        );
    }
    let pass = items
        .values()
        .all(|item| item["pass"].as_bool() == Some(true));
    Ok(json!({
        "run": run.display().to_string(),
        "control": control.display().to_string(),
        "controls": configurations.iter().map(|(name, path)| (name.clone(), Value::String(path.display().to_string()))).collect::<Map<_, _>>(),
        "pass": pass,
        "window_s": [start, end],
        "cancelled": cancelled,
        "items": items,
        "measures": {
            "run": measures(&requests, (start, end)),
            "control": measures(&controls.iter().collect::<Vec<_>>(), (start, end)),
        },
        "changes": change_list(&timeline),
    }))
}

/// `check-faults`' report of `run` against its rows served alone (a completed
/// bundle of `max_concurrency = 1` and no `duration_s`).
///
/// # Errors
///
/// As [`change_report`], and when `reference` is not of rows served alone.
pub fn fault_report(
    run: &Path,
    reference: &Path,
    check_metric: &str,
    replay: &Path,
) -> Result<Value> {
    let (all, timeline) = bundle(run)?;
    let state: Value = serde_json::from_slice(&std::fs::read(reference.join("run.json"))?)?;
    let load = &state["workload"]["load"];
    ensure!(
        state["status"] == "completed"
            && load["max_concurrency"] == 1
            && load["duration_s"].is_null(),
        "{} is not a completed bundle of rows served alone (max_concurrency = 1, no duration_s)",
        reference.display()
    );
    let alone = first_outputs(&records(reference, 0.0)?);
    let cancelled = all
        .iter()
        .filter(|record| record.classifier() == CANCELLED)
        .count();
    let requests: Vec<&Request> = all
        .iter()
        .filter(|record| record.classifier() != CANCELLED)
        .collect();
    let labelled = [("the row served alone".to_owned(), &alone)];
    let mut items = Map::new();
    items.insert(
        "same_tokens_as_alone".into(),
        Value::Object(item(&requests, |record| {
            request_fault(record).or_else(|| token_fault(record, &labelled))
        })),
    );
    items.insert(
        "invariants_hold".into(),
        Value::Object(invariants(run, check_metric, replay)?),
    );
    let pass = items
        .values()
        .all(|item| item["pass"].as_bool() == Some(true));
    Ok(json!({
        "run": run.display().to_string(),
        "reference": reference.display().to_string(),
        "pass": pass,
        "cancelled": cancelled,
        "items": items,
        "changes": change_list(&timeline),
    }))
}

fn seconds(value: Option<f64>) -> String {
    value.map_or_else(|| "never".into(), |value| format!("{value:.3} s"))
}

pub(crate) fn verdict(pass: bool) -> &'static str {
    if pass { "pass" } else { "FAIL" }
}

/// The table of a report's items, in the order of `titles`.
fn item_rows(items: &Value, titles: &[(&str, &str)]) -> Vec<String> {
    let mut lines = vec![
        "| Item | Verdict | Detail |".to_owned(),
        "| --- | --- | --- |".to_owned(),
    ];
    for (key, title) in titles {
        let Some(item) = items.get(*key) else {
            continue;
        };
        let mut detail = if *key == "invariants_hold" {
            format!(
                "check failures {}; replay {}",
                item["check_failures"],
                display(&item["replay"])
            )
        } else {
            let mut detail = format!("{} of {} requests fail", item["count"], item["of"]);
            if let Some(first) = item["first"].as_str().filter(|first| !first.is_empty()) {
                detail.push_str(&format!("; first: {first}"));
            }
            detail
        };
        if *key == "admitted_throughout" {
            detail.push_str(&format!(
                "; deadline {:.3} s",
                item["deadline_s"].as_f64().unwrap_or(0.0)
            ));
        }
        if *key == "cached_across_switch" {
            detail.push_str(&format!(
                "; {} prompts served before the switch",
                item["served"]
            ));
        }
        lines.push(format!(
            "| {title} | {} | {detail} |",
            verdict(item["pass"].as_bool() == Some(true))
        ));
    }
    lines
}

fn change_lines(changes: &Value) -> Vec<String> {
    let mut lines = vec![
        "Changes, in seconds from the load's start:".to_owned(),
        String::new(),
    ];
    for change in changes.as_array().into_iter().flatten() {
        let mut line = format!(
            "- `{}`: issued {}, ended {}",
            change["name"].as_str().unwrap_or_default(),
            seconds(change["issued_s"].as_f64()),
            seconds(change["ended_s"].as_f64())
        );
        if let Some(error) = change["error"].as_str().filter(|error| !error.is_empty()) {
            line.push_str(&format!("; failed: {error}"));
        }
        lines.push(line);
    }
    lines
}

/// A `check-change` report as the run's `result.md`.
pub fn render_change(report: &Value) -> String {
    let start = report["window_s"][0].as_f64().unwrap_or(0.0);
    let end = report["window_s"][1].as_f64().unwrap_or(0.0);
    let run = report["run"].as_str().unwrap_or_default();
    let name = Path::new(run)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut lines = vec![
        format!("# Pass under change: {name}"),
        String::new(),
        format!(
            "Run `{run}` against the control `{}`. The change's window runs from {start:.3} s to {end:.3} s after the load's start. Verdict: **{}**.",
            report["control"].as_str().unwrap_or_default(),
            verdict(report["pass"].as_bool() == Some(true))
        ),
        String::new(),
    ];
    for (configuration, path) in report["controls"].as_object().into_iter().flatten() {
        lines.push(format!(
            "- The control of configuration `{configuration}`: `{}`.",
            path.as_str().unwrap_or_default()
        ));
    }
    lines.push(format!(
        "- Requests a change cancelled, left out of the items: {}.",
        report["cancelled"]
    ));
    lines.push(String::new());
    lines.extend(item_rows(
        &report["items"],
        &[
            ("no_failed_request", "1. No failed request"),
            ("no_closed_connection", "2. No closed connection"),
            ("no_token_lost_or_repeated", "3. No token lost or repeated"),
            ("admitted_throughout", "4. New requests admitted throughout"),
            ("invariants_hold", "5. The buffer pool's invariants hold"),
            (
                "cached_across_switch",
                "6. Prompts cached across the switch",
            ),
        ],
    ));
    let (run_measures, control_measures) =
        (&report["measures"]["run"], &report["measures"]["control"]);
    lines.extend([
        String::new(),
        "| Measure (no pass line) | Run | Control |".into(),
        "| --- | --- | --- |".into(),
    ]);
    for (key, title) in [
        (
            "max_gap_ms",
            "Longest gap between two tokens of any stream (ms)",
        ),
        ("max_gap_window_ms", "Longest gap meeting the window (ms)"),
        (
            "ttft_p50_ms",
            "TTFT p50, requests submitted in the window (ms)",
        ),
        (
            "ttft_p99_ms",
            "TTFT p99, requests submitted in the window (ms)",
        ),
    ] {
        lines.push(format!(
            "| {title} | {:.3} | {:.3} |",
            run_measures[key].as_f64().unwrap_or(0.0),
            control_measures[key].as_f64().unwrap_or(0.0)
        ));
    }
    for (key, title) in [
        ("window_requests", "Requests submitted in the window"),
        ("requests", "Requests"),
    ] {
        lines.push(format!(
            "| {title} | {} | {} |",
            run_measures[key], control_measures[key]
        ));
    }
    lines.push(String::new());
    lines.extend(change_lines(&report["changes"]));
    lines.join("\n") + "\n"
}

/// A `check-faults` report as the run's `result.md`.
pub fn render_faults(report: &Value) -> String {
    let run = report["run"].as_str().unwrap_or_default();
    let name = Path::new(run)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut lines = vec![
        format!("# Faults under load: {name}"),
        String::new(),
        format!(
            "Run `{run}` against its rows served alone `{}`. Verdict: **{}**.",
            report["reference"].as_str().unwrap_or_default(),
            verdict(report["pass"].as_bool() == Some(true))
        ),
        String::new(),
        format!(
            "- Requests a change cancelled, left out of the items: {}.",
            report["cancelled"]
        ),
        String::new(),
    ];
    lines.extend(item_rows(
        &report["items"],
        &[
            (
                "same_tokens_as_alone",
                "1. Every request not cancelled gives its row's tokens served alone",
            ),
            ("invariants_hold", "2. The buffer pool's invariants hold"),
        ],
    ));
    lines.push(String::new());
    lines.extend(change_lines(&report["changes"]));
    lines.join("\n") + "\n"
}
