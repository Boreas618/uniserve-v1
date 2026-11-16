//! Advisory execution hints from an Nsight Systems SQLite export: the CUDA
//! calls made inside synchronous NVTX ranges of a name prefix, with their GPU
//! copies, call stacks and the GIL's ownership around them.
//!
//! Hints are candidates for inspection, not errors: no duration threshold proves
//! a wait unnecessary (even a short one can prevent schedule-ahead, and a long one
//! may belong to a result consumer or the start), so the scope, the stack and the
//! workload decide. A missing activity is reported as a coverage gap, never as a
//! clean result. Times are nanoseconds on the capture's timeline; global thread
//! ids pack the process above their low 24 bits.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

/// The rules, by name.
pub const RULES: [&str; 5] = [
    "host-sync",
    "gil-held-wait",
    "sync-copy",
    "pageable-copy",
    "device-roundtrip",
];

/// The CUDA host waits.
const HOST_WAITS: [&str; 6] = [
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
];

/// The low bits of a global thread id that name the thread within its process.
const THREAD_BITS: i64 = 0xFF_FFFF;

/// A synchronous NVTX range.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Scope {
    pub name: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub global_tid: i64,
    pub batch_id: Option<i64>,
}

/// A device copy correlated with its host call.
#[derive(Debug, Clone, Serialize)]
pub struct Copy {
    pub start_ns: i64,
    pub end_ns: i64,
    pub device: i64,
    pub stream: i64,
    pub bytes: i64,
    pub direction: String,
    pub source_storage: String,
    pub destination_storage: String,
}

/// One CUDA API call, with its stack, its copies and the GIL around it.
#[derive(Debug, Clone, Serialize)]
pub struct Event {
    pub name: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub global_tid: i64,
    pub correlation_id: i64,
    pub scope: Scope,
    pub stack: Vec<String>,
    pub copies: Vec<Copy>,
    pub gil: Vec<Scope>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hint {
    pub rule: String,
    pub observation: String,
    pub suggestion: String,
    pub evidence: Vec<Event>,
}

#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub hints: Vec<Hint>,
    pub coverage: BTreeMap<String, i64>,
    pub notes: Vec<String>,
}

/// A CUDA symbol without its ABI and per-thread-default-stream suffixes.
fn api_name(name: &str) -> String {
    let mut name = name;
    loop {
        let trimmed = ["_ptsz", "_ptds"]
            .iter()
            .find_map(|suffix| name.strip_suffix(suffix))
            .or_else(|| {
                let (head, tail) = name.rsplit_once("_v")?;
                (!tail.is_empty() && tail.bytes().all(|byte| byte.is_ascii_digit())).then_some(head)
            });
        match trimmed {
            Some(trimmed) => name = trimmed,
            None => return name.to_owned(),
        }
    }
}

/// Whether a call is a CUDA host wait.
pub fn is_host_wait(name: &str) -> bool {
    HOST_WAITS.contains(&api_name(name).as_str())
}

/// Applies the rules, keeping the events as evidence.
pub fn inspect(events: &[Event]) -> Vec<Hint> {
    let hint = |rule: &str, observation: String, suggestion: &str, evidence: Vec<Event>| Hint {
        rule: rule.to_owned(),
        observation,
        suggestion: suggestion.to_owned(),
        evidence,
    };
    let mut hints = Vec::new();
    let mut readbacks: BTreeMap<(Scope, i64), &Event> = BTreeMap::new();
    for event in events {
        let name = api_name(&event.name);
        if HOST_WAITS.contains(&name.as_str()) {
            hints.push(hint(
                "host-sync",
                format!("Host synchronization API {} inside {}.", event.name, event.scope.name),
                "Consider querying completion or using a stream dependency to keep submitting ready work. Result \
                 consumption, startup, shutdown and cleanup may require this wait.",
                vec![event.clone()],
            ));
        }
        let holds: Vec<&Scope> = event
            .gil
            .iter()
            .filter(|state| state.name == "Holding GIL")
            .collect();
        if !holds.is_empty() {
            let held: i64 = holds
                .iter()
                .map(|state| event.end_ns.min(state.end_ns) - event.start_ns.max(state.start_ns))
                .sum();
            let waiters: BTreeSet<i64> = event
                .gil
                .iter()
                .filter(|state| state.name == "Waiting for GIL")
                .map(|state| state.global_tid)
                .collect();
            let contention = if waiters.is_empty() {
                " No overlapping GIL waiter was captured.".to_owned()
            } else {
                format!(
                    " {} other thread(s) in this process were waiting for the GIL during that overlap.",
                    waiters.len()
                )
            };
            hints.push(hint(
                "gil-held-wait",
                format!("{} overlapped GIL ownership on its thread for {held} ns.{contention}", event.name),
                "Inspect the native binding or resource destructor. A necessary GPU wait can still release the GIL. \
                 The overlap alone does not prove deadlock or a latency bottleneck; interpreter identity is not in \
                 this trace.",
                vec![event.clone()],
            ));
        }
        for copy in &event.copies {
            let host_device =
                copy.direction == "Host-to-Device" || copy.direction == "Device-to-Host";
            if name.contains("Memcpy") && !name.contains("Async") && host_device {
                hints.push(hint(
                    "sync-copy",
                    format!("Synchronous host/device copy: {}.", event.name),
                    "Consider an asynchronous copy with pinned storage and a completion dependency. Immediate host \
                     consumption or startup can justify a synchronous copy.",
                    vec![event.clone()],
                ));
            }
            if copy.source_storage == "Pageable" || copy.destination_storage == "Pageable" {
                hints.push(hint(
                    "pageable-copy",
                    format!(
                        "{}: {} bytes through pageable host storage. An Async API may block or stage the copy.",
                        copy.direction, copy.bytes
                    ),
                    "Consider reusing pinned host buffers on the execution path. One-time loading and host-only copies \
                     may not justify pinned storage.",
                    vec![event.clone()],
                ));
            }
            let key = (event.scope.clone(), copy.device);
            if copy.direction == "Device-to-Host" {
                readbacks.insert(key, event);
            } else if copy.direction == "Host-to-Device" {
                let Some(previous) = readbacks.remove(&key) else {
                    continue;
                };
                let ready = previous
                    .copies
                    .iter()
                    .map(|copy| copy.end_ns)
                    .max()
                    .unwrap_or(i64::MIN);
                if ready > event.start_ns {
                    continue;
                }
                hints.push(hint(
                    "device-roundtrip",
                    "A D-to-H copy completed before an H-to-D submission on the same thread, device and execution scope."
                        .into(),
                    "Check whether intermediate values or decisions can stay on the device. The trace does not establish \
                     that these copies carry the same data; independent inputs or CPU computation can explain this \
                     ordering.",
                    vec![previous.clone(), event.clone()],
                ));
            }
        }
    }
    hints
}

/// Inspects the calls inside synchronous ranges named with `prefix`, reading the
/// export without changing it.
///
/// # Errors
///
/// When the export cannot be read; SQL and schema errors propagate.
pub fn analyze(path: &Path, prefix: &str) -> Result<Report> {
    let db =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("opening {}", path.display()))?;
    let mut report = Report::default();
    let tables: BTreeSet<String> = {
        let mut statement = db.prepare("SELECT name FROM sqlite_master")?;
        let names = statement.query_map([], |row| row.get::<_, String>(0))?;
        names.collect::<Result<_, _>>()?
    };
    let missing: Vec<&str> = ["CUPTI_ACTIVITY_KIND_RUNTIME", "NVTX_EVENTS"]
        .into_iter()
        .filter(|table| !tables.contains(*table))
        .collect();
    if !missing.is_empty() {
        report
            .notes
            .push(format!("Missing trace activities: {}", missing.join(", ")));
        return Ok(report);
    }
    let scopes = scopes(&db, prefix)?;
    let named = scopes
        .values()
        .flatten()
        .filter(|scope| scope.name.starts_with(prefix))
        .count();
    let batched = scopes
        .values()
        .flatten()
        .filter(|scope| scope.batch_id.is_some())
        .count();
    report.coverage.insert("scopes".into(), named as i64);
    report
        .coverage
        .insert("batch_scopes".into(), batched as i64);
    if named == 0 {
        report
            .notes
            .push(format!("No closed NVTX ranges match {prefix:?}."));
    } else if batched == 0 {
        report.notes.push(
            "No native batch stages recorded; preparation and completion coverage is unknown."
                .into(),
        );
    }
    let events = calls(&db, &tables, &scopes, prefix, &mut report)?;
    let events = copies(&db, &tables, events, &mut report)?;
    let events = gil(&db, events, &mut report)?;
    report.hints = inspect(&events);
    report.notes.push(
        "Hints are candidates for inspection, not errors. Trace timing includes profiler overhead; it is not a speedup \
         measurement."
            .into(),
    );
    Ok(report)
}

/// The closed synchronous ranges named with `prefix` or of the workers' batches
/// (whose 64-bit payload is the batch number), by thread, in start order.
fn scopes(db: &rusqlite::Connection, prefix: &str) -> Result<BTreeMap<i64, Vec<Scope>>> {
    let mut statement = db.prepare(
        "SELECT n.start, n.end, n.globalTid, n.uint64Value, coalesce(n.text, s.value) AS name \
         FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId \
         WHERE n.end IS NOT NULL AND n.end >= n.start \
           AND (n.endGlobalTid IS NULL OR n.endGlobalTid = n.globalTid) \
           AND (instr(coalesce(n.text, s.value), ?1) = 1 OR instr(coalesce(n.text, s.value), 'uniserve.worker.') = 1) \
         ORDER BY n.start, n.end DESC",
    )?;
    let rows = statement.query_map([prefix], |row| {
        let name: String = row.get(4)?;
        let batch: Option<i64> = row.get(3)?;
        Ok(Scope {
            batch_id: if name.starts_with("uniserve.worker.") {
                batch
            } else {
                None
            },
            name,
            start_ns: row.get(0)?,
            end_ns: row.get(1)?,
            global_tid: row.get(2)?,
        })
    })?;
    let mut scopes: BTreeMap<i64, Vec<Scope>> = BTreeMap::new();
    for scope in rows {
        let scope = scope?;
        scopes.entry(scope.global_tid).or_default().push(scope);
    }
    Ok(scopes)
}

/// The Runtime and Driver calls inside a selected range, each in its innermost
/// one, which takes the batch of the innermost enclosing batch range; backtrace
/// records repeat a call's interval and correlation and give its stack.
fn calls(
    db: &rusqlite::Connection,
    tables: &BTreeSet<String>,
    scopes: &BTreeMap<i64, Vec<Scope>>,
    prefix: &str,
    report: &mut Report,
) -> Result<Vec<Event>> {
    let classes: BTreeMap<String, i64> = {
        let mut statement = db.prepare("SELECT id, name FROM ENUM_NSYS_EVENT_CLASS")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, i64>(0)?))
        })?;
        rows.collect::<Result<_, _>>()?
    };
    let class = |name: &str| {
        classes
            .get(name)
            .copied()
            .with_context(|| format!("the event class {name}"))
    };
    let (runtime, driver, backtrace) = (
        class("TRACE_PROCESS_EVENT_CUDA_RUNTIME")?,
        class("TRACE_PROCESS_EVENT_CUDA_DRIVER")?,
        class("TRACE_PROCESS_EVENT_CUDABACKTRACE")?,
    );
    let stacks: BTreeMap<(i64, i64), i64> = {
        let mut statement = db.prepare(
            "SELECT globalTid, correlationId, callchainId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE eventClass = ?1",
        )?;
        let rows = statement.query_map([backtrace], |row| {
            Ok((
                (row.get::<_, i64>(0)?, row.get::<_, i64>(1)?),
                row.get::<_, Option<i64>>(2)?,
            ))
        })?;
        rows.filter_map(|row| row.ok().and_then(|(key, chain)| Some((key, chain?))))
            .collect()
    };
    for kind in ["runtime_calls", "driver_calls", "scoped_calls"] {
        report.coverage.insert(kind.into(), 0);
    }
    let mut statement = db.prepare(
        "SELECT a.start, a.end, a.globalTid, a.correlationId, a.callchainId, a.eventClass, s.value AS name \
         FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id = a.nameId \
         WHERE a.eventClass IN (?1, ?2) ORDER BY a.start, a.end DESC",
    )?;
    let rows = statement.query_map([runtime, driver], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, i64>(5)?,
            row.get::<_, String>(6)?,
        ))
    })?;
    let mut positions: BTreeMap<i64, usize> = BTreeMap::new();
    let mut active: BTreeMap<i64, Vec<Scope>> = BTreeMap::new();
    let mut stack_cache: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    let mut events = Vec::new();
    let empty = Vec::new();
    for row in rows {
        let (start, end, tid, correlation, chain, event_class, name) = row?;
        let kind = if event_class == runtime {
            "runtime_calls"
        } else {
            "driver_calls"
        };
        *report.coverage.entry(kind.into()).or_default() += 1;
        let candidates = scopes.get(&tid).unwrap_or(&empty);
        let position = positions.entry(tid).or_default();
        let open = active.entry(tid).or_default();
        while *position < candidates.len() && candidates[*position].start_ns <= start {
            open.push(candidates[*position].clone());
            *position += 1;
        }
        open.retain(|scope| scope.end_ns >= start);
        let enclosing: Vec<&Scope> = open.iter().filter(|scope| scope.end_ns >= end).collect();
        let Some(innermost) = enclosing
            .iter()
            .rev()
            .find(|scope| scope.name.starts_with(prefix))
        else {
            continue;
        };
        let mut scope = (*innermost).clone();
        if scope.batch_id.is_none() {
            scope.batch_id = enclosing.iter().rev().find_map(|scope| scope.batch_id);
        }
        let chain = chain.or_else(|| stacks.get(&(tid, correlation)).copied());
        let stack = match chain {
            Some(chain) if tables.contains("CUDA_CALLCHAINS") => {
                if let std::collections::btree_map::Entry::Vacant(entry) = stack_cache.entry(chain)
                {
                    entry.insert(stack_of(db, chain)?);
                }
                stack_cache[&chain].clone()
            }
            _ => Vec::new(),
        };
        events.push(Event {
            name,
            start_ns: start,
            end_ns: end,
            global_tid: tid,
            correlation_id: correlation,
            scope,
            stack,
            copies: Vec::new(),
            gil: Vec::new(),
        });
    }
    report
        .coverage
        .insert("scoped_calls".into(), events.len() as i64);
    report.coverage.insert(
        "scoped_calls_with_stack".into(),
        events
            .iter()
            .filter(|event| !event.stack.is_empty())
            .count() as i64,
    );
    for kind in ["runtime_calls", "driver_calls"] {
        if report.coverage.get(kind) == Some(&0) {
            report
                .notes
                .push(format!("No {kind} recorded; this API coverage is unknown."));
        }
    }
    if stack_cache.is_empty() {
        report
            .notes
            .push("No CUDA call stacks in selected scopes.".into());
    }
    Ok(events)
}

fn stack_of(db: &rusqlite::Connection, chain: i64) -> Result<Vec<String>> {
    let mut statement = db.prepare(
        "SELECT s.value, m.value FROM CUDA_CALLCHAINS c LEFT JOIN StringIds s ON s.id = c.symbol \
         LEFT JOIN StringIds m ON m.id = c.module WHERE c.id = ?1 ORDER BY c.stackDepth",
    )?;
    let frames = statement.query_map([chain], |row| {
        let symbol: Option<String> = row.get(0)?;
        let module: Option<String> = row.get(1)?;
        Ok(format!(
            "{} ({})",
            symbol.unwrap_or_else(|| "<unresolved>".into()),
            module.unwrap_or_else(|| "?".into())
        ))
    })?;
    Ok(frames.collect::<Result<_, _>>()?)
}

/// Each call's GPU copies, joined by process and correlation id (correlation
/// ids are process-local).
fn copies(
    db: &rusqlite::Connection,
    tables: &BTreeSet<String>,
    mut events: Vec<Event>,
    report: &mut Report,
) -> Result<Vec<Event>> {
    if !tables.contains("CUPTI_ACTIVITY_KIND_MEMCPY") {
        report
            .notes
            .push("No GPU copy activities; copy rules have no evidence.".into());
        return Ok(events);
    }
    let keys: BTreeSet<(i64, i64)> = events
        .iter()
        .map(|event| (event.global_tid & !THREAD_BITS, event.correlation_id))
        .collect();
    let mut statement = db.prepare(
        "SELECT c.start, c.end, c.deviceId, c.streamId, c.bytes, d.label, s.label, t.label, c.globalPid, c.correlationId \
         FROM CUPTI_ACTIVITY_KIND_MEMCPY c \
         LEFT JOIN ENUM_CUDA_MEMCPY_OPER d ON d.id = c.copyKind \
         LEFT JOIN ENUM_CUDA_MEM_KIND s ON s.id = c.srcKind \
         LEFT JOIN ENUM_CUDA_MEM_KIND t ON t.id = c.dstKind ORDER BY c.start",
    )?;
    let rows = statement.query_map([], |row| {
        let label = |index: usize| -> rusqlite::Result<String> {
            Ok(row
                .get::<_, Option<String>>(index)?
                .unwrap_or_else(|| "Unknown".into()))
        };
        Ok((
            (row.get::<_, i64>(8)?, row.get::<_, i64>(9)?),
            Copy {
                start_ns: row.get(0)?,
                end_ns: row.get(1)?,
                device: row.get(2)?,
                stream: row.get(3)?,
                bytes: row.get(4)?,
                direction: label(5)?,
                source_storage: label(6)?,
                destination_storage: label(7)?,
            },
        ))
    })?;
    let mut by_call: BTreeMap<(i64, i64), Vec<Copy>> = BTreeMap::new();
    for row in rows {
        let (key, copy) = row?;
        if keys.contains(&key) {
            by_call.entry(key).or_default().push(copy);
        }
    }
    report.coverage.insert(
        "scoped_copies".into(),
        by_call.values().map(Vec::len).sum::<usize>() as i64,
    );
    for event in &mut events {
        event.copies = by_call
            .get(&(event.global_tid & !THREAD_BITS, event.correlation_id))
            .cloned()
            .unwrap_or_default();
    }
    Ok(events)
}

/// The GIL around each host wait: Nsight records ownership and contention as
/// synchronous ranges of its own GIL domain (domain ids are process-local, so an
/// application range of the same label is not taken for them). A wait keeps the
/// holds on its thread that overlap it, and the waits of other threads of its
/// process that overlap both.
fn gil(db: &rusqlite::Connection, events: Vec<Event>, report: &mut Report) -> Result<Vec<Event>> {
    let domains: BTreeSet<(i64, i64)> = {
        let mut statement = db.prepare(
            "SELECT n.globalTid, n.domainId FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId \
             WHERE n.eventType = 75 AND coalesce(n.text, s.value) = 'GIL Trace'",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, i64>(0)? & !THREAD_BITS, row.get::<_, i64>(1)?))
        })?;
        rows.collect::<Result<_, _>>()?
    };
    for kind in ["gil_holds", "gil_waits", "gil_open_ranges"] {
        report.coverage.insert(kind.into(), 0);
    }
    if domains.is_empty() {
        report.notes.push(
            "No Python GIL trace; GIL ownership during waits is unknown. Capture with --trace=cuda,nvtx,python-gil.".into(),
        );
        return Ok(events);
    }
    let mut statement = db.prepare(
        "SELECT n.start, n.end, n.globalTid, n.domainId, coalesce(n.text, s.value) \
         FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId \
         WHERE n.eventType = 59 AND coalesce(n.text, s.value) IN ('Holding GIL', 'Waiting for GIL') ORDER BY n.start",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, Option<i64>>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;
    let mut states = Vec::new();
    for row in rows {
        let (start, end, tid, domain, name) = row?;
        if !domains.contains(&(tid & !THREAD_BITS, domain)) {
            continue;
        }
        let Some(end) = end else {
            *report.coverage.entry("gil_open_ranges".into()).or_default() += 1;
            continue;
        };
        let kind = if name == "Holding GIL" {
            "gil_holds"
        } else {
            "gil_waits"
        };
        *report.coverage.entry(kind.into()).or_default() += 1;
        states.push(Scope {
            name,
            start_ns: start,
            end_ns: end,
            global_tid: tid,
            batch_id: None,
        });
    }
    if report.coverage["gil_open_ranges"] > 0 {
        report
            .notes
            .push("Unclosed GIL ranges were omitted; capture boundaries may hide ownership or contention.".into());
    }
    if report.coverage["gil_holds"] == 0 {
        report
            .notes
            .push("No closed GIL ownership intervals were captured.".into());
    }
    // Both lists are in time order: one pass through the GIL timeline keeps the
    // intervals that can overlap the current or a nested wait.
    let mut position = 0;
    let mut active: Vec<Scope> = Vec::new();
    let mut result = Vec::with_capacity(events.len());
    for mut event in events {
        if !is_host_wait(&event.name) {
            result.push(event);
            continue;
        }
        while position < states.len() && states[position].start_ns < event.end_ns {
            if states[position].end_ns > event.start_ns {
                active.push(states[position].clone());
            }
            position += 1;
        }
        active.retain(|state| state.end_ns > event.start_ns);
        let holds: Vec<Scope> = active
            .iter()
            .filter(|state| {
                state.name == "Holding GIL"
                    && state.global_tid == event.global_tid
                    && state.start_ns < event.end_ns
            })
            .cloned()
            .collect();
        let waiters: Vec<Scope> = active
            .iter()
            .filter(|state| {
                state.name == "Waiting for GIL"
                    && state.global_tid != event.global_tid
                    && state.global_tid & !THREAD_BITS == event.global_tid & !THREAD_BITS
                    && holds.iter().any(|hold| {
                        event.start_ns.max(hold.start_ns).max(state.start_ns)
                            < event.end_ns.min(hold.end_ns).min(state.end_ns)
                    })
            })
            .cloned()
            .collect();
        event.gil = holds.into_iter().chain(waiters).collect();
        result.push(event);
    }
    Ok(result)
}
