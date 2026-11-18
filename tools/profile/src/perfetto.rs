//! A session's files as one trace in the Trace Event Format, which the Perfetto
//! UI opens: a process per worker; a track per CUDA stream (kernels, copies,
//! fills, graphs), per host thread (API calls, NVTX ranges and marks), for the
//! steps, and per device for the fence probes; and the PM samples as counters.
//! Times start at the earliest event, in microseconds.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::session::{ProfileFile, STEP_BEGIN};

const STREAM: i64 = 1 << 32;
const STEPS: i64 = 2 << 32;
const FENCE: i64 = 3 << 32;
/// `CUpti_ActivityFlag` of an NVTX mark.
const INSTANT: i64 = 1;
const START: i64 = 2;
const END: i64 = 4;

fn number(row: &Map<String, Value>, key: &str) -> i64 {
    row.get(key).and_then(Value::as_i64).unwrap_or(0)
}

pub(crate) fn slice(pid: i64, tid: i64, name: &str, start: i64, end: i64, args: Value) -> Value {
    json!({"name": name, "ph": "X", "pid": pid, "tid": tid, "ts": start, "dur": (end - start).max(0), "args": args})
}

pub(crate) fn track(pid: i64, tid: i64, name: &str) -> Value {
    json!({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": name}})
}

/// The trace of the files at `paths`.
///
/// # Errors
///
/// When a file is not a session's.
pub fn convert(paths: &[PathBuf]) -> Result<Value> {
    let mut events = Vec::new();
    for profile in crate::session::open_all(paths)? {
        events.extend(file_events(&profile)?);
    }
    Ok(document(events))
}

/// The trace of `events`, whose `ts` and `dur` are nanoseconds: times are
/// moved to start at the earliest event and stated in microseconds, the
/// format's unit.
pub(crate) fn document(mut events: Vec<Value>) -> Value {
    let origin = events
        .iter()
        .filter_map(|event| event.get("ts").and_then(Value::as_i64))
        .min()
        .unwrap_or(0);
    for event in &mut events {
        if let Some(ts) = event.get("ts").and_then(Value::as_i64) {
            event["ts"] = ((ts - origin) as f64 / 1000.0).into();
            if let Some(duration) = event.get("dur").and_then(Value::as_i64) {
                event["dur"] = (duration as f64 / 1000.0).into();
            }
        }
    }
    json!({"traceEvents": events, "displayTimeUnit": "ns"})
}

#[expect(clippy::too_many_lines, reason = "one track per table, in order")]
fn file_events(profile: &ProfileFile) -> Result<Vec<Value>> {
    let pid = profile.rank();
    let meta = |key: &str| {
        profile
            .metadata
            .get(key)
            .cloned()
            .unwrap_or_else(|| "None".into())
    };
    let mut events = vec![json!({
        "name": "process_name",
        "ph": "M",
        "pid": pid,
        "args": {"name": format!("rank {pid} (epoch {}, device {}, session {})", meta("epoch"), meta("device"), meta("session"))},
    })];
    let mut streams = BTreeSet::new();
    let mut threads = BTreeSet::new();
    let mut device_slice = |events: &mut Vec<Value>,
                            row: &Map<String, Value>,
                            name: String,
                            mut args: Map<String, Value>| {
        let stream = number(row, "stream");
        streams.insert(stream);
        args.insert(
            "correlation".into(),
            row.get("correlation").cloned().unwrap_or(Value::Null),
        );
        events.push(slice(
            pid,
            STREAM + stream,
            &name,
            number(row, "start"),
            number(row, "end"),
            Value::Object(args),
        ));
    };
    let field = |row: &Map<String, Value>, key: &str| row.get(key).cloned().unwrap_or(Value::Null);
    for row in profile.rows("kernel", None)? {
        let mut args = Map::new();
        args.insert(
            "grid".into(),
            json!([
                field(&row, "grid_x"),
                field(&row, "grid_y"),
                field(&row, "grid_z")
            ]),
        );
        args.insert(
            "block".into(),
            json!([
                field(&row, "block_x"),
                field(&row, "block_y"),
                field(&row, "block_z")
            ]),
        );
        args.insert("graph".into(), field(&row, "graph"));
        args.insert("graph_node".into(), field(&row, "graph_node"));
        device_slice(&mut events, &row, profile.name(&field(&row, "name")), args);
    }
    for row in profile.rows("copy", None)? {
        let mut args = Map::new();
        args.insert("bytes".into(), field(&row, "bytes"));
        args.insert("kind".into(), field(&row, "copy_kind"));
        device_slice(&mut events, &row, "copy".into(), args);
    }
    for row in profile.rows("fill", None)? {
        let mut args = Map::new();
        args.insert("bytes".into(), field(&row, "bytes"));
        args.insert("value".into(), field(&row, "value"));
        device_slice(&mut events, &row, "fill".into(), args);
    }
    for row in profile.rows("graph", None)? {
        let mut args = Map::new();
        args.insert("graph".into(), field(&row, "graph"));
        device_slice(
            &mut events,
            &row,
            format!("graph {}", field(&row, "graph")),
            args,
        );
    }
    for row in profile.rows("api", None)? {
        let thread = number(&row, "thread");
        threads.insert(thread);
        events.push(slice(
            pid,
            thread,
            &profile.name(&field(&row, "name")),
            number(&row, "start"),
            number(&row, "end"),
            json!({"correlation": field(&row, "correlation"), "result": field(&row, "result")}),
        ));
    }
    // NVTX ranges pair their start and end marks by id; an instantaneous mark is
    // an instant event.
    let mut starts: BTreeMap<i64, Map<String, Value>> = BTreeMap::new();
    for row in profile.rows("marker", Some("time"))? {
        let flags = number(&row, "flags");
        if flags & START != 0 {
            starts.insert(number(&row, "id"), row);
        } else if flags & END != 0 && starts.contains_key(&number(&row, "id")) {
            let Some(start) = starts.remove(&number(&row, "id")) else {
                continue;
            };
            let thread = number(&start, "object_1");
            threads.insert(thread);
            events.push(slice(
                pid,
                thread,
                &profile.name(&field(&start, "name")),
                number(&start, "time"),
                number(&row, "time"),
                json!({"domain": profile.name(&field(&start, "domain"))}),
            ));
        } else if flags & INSTANT != 0 {
            let thread = number(&row, "object_1");
            threads.insert(thread);
            events.push(json!({
                "name": profile.name(&field(&row, "name")),
                "ph": "i",
                "s": "t",
                "pid": pid,
                "tid": thread,
                "ts": number(&row, "time"),
            }));
        }
    }
    if profile.count("steps")? > 0 {
        events.push(track(pid, STEPS, "steps"));
    }
    for row in profile.rows("steps", None)? {
        if row.get("end").is_none_or(Value::is_null) {
            continue;
        }
        events.push(slice(
            pid,
            STEPS,
            &format!(
                "{} {}",
                row.get("stage").and_then(Value::as_str).unwrap_or_default(),
                field(&row, "step")
            ),
            number(&row, "begin"),
            number(&row, "end"),
            json!({"selected": number(&row, "selected") != 0}),
        ));
    }
    let mut devices = BTreeSet::new();
    for interval in profile.fence_intervals()? {
        let device = number(&interval, "device");
        let Some(offset) = profile.offset("globaltimer", Some(device))? else {
            continue;
        };
        devices.insert(device);
        events.push(slice(
            pid,
            FENCE + device,
            interval
                .get("module")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            number(&interval, "start") + offset.nanoseconds,
            number(&interval, "end") + offset.nanoseconds,
            json!({"path": interval.get("path"), "step": interval.get("step")}),
        ));
    }
    // The steps' spans between their markers.
    let mut begins: BTreeMap<(i64, i64), i64> = BTreeMap::new();
    for row in profile.rows("fence", Some("device, time"))? {
        let (device, probe) = (number(&row, "device"), number(&row, "probe"));
        if probe < STEP_BEGIN {
            continue;
        }
        let Some(offset) = profile.offset("globaltimer", Some(device))? else {
            continue;
        };
        let key = (device, number(&row, "step"));
        if probe == STEP_BEGIN {
            begins.insert(key, number(&row, "time"));
        } else if let Some(begin) = begins.remove(&key) {
            devices.insert(device);
            events.push(slice(
                pid,
                FENCE + device,
                &format!("step {}", key.1),
                begin + offset.nanoseconds,
                number(&row, "time") + offset.nanoseconds,
                json!({}),
            ));
        }
    }
    for row in profile.rows("pm_sample", Some("sample"))? {
        events.push(json!({
            "name": row.get("metric"),
            "ph": "C",
            "pid": pid,
            "ts": number(&row, "start"),
            "args": {"value": row.get("value")},
        }));
    }
    events.extend(
        streams
            .iter()
            .map(|stream| track(pid, STREAM + stream, &format!("stream {stream}"))),
    );
    events.extend(
        threads
            .iter()
            .map(|thread| track(pid, *thread, &format!("thread {thread}"))),
    );
    events.extend(devices.iter().map(|device| {
        track(
            pid,
            FENCE + device,
            &format!("fence probes (device {device})"),
        )
    }));
    Ok(events)
}
