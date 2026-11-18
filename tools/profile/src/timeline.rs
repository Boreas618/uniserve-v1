//! A deployment's `span` events as one trace in the Trace Event Format, which
//! the Perfetto UI opens (`docs/profiling.md`).
//!
//! The engine-side timeline writes every host stage and every timed kernel as
//! a row of the observation store's `events` table with `kind = 'span'`:
//! labels `replica`, `process` (`daemon`, `scheduler`, `worker`, `storage`)
//! and, for a worker, `rank`; fields `start` and `end` (nanoseconds since the
//! Unix epoch on the host clock), `request` (a request's id) or `batch` (a
//! batch id), and `stage` (a host stage's name) or `kernel` (the kernel's
//! `op` and `descriptor` are labels). A row that names both a request and a
//! batch is the scheduler's membership record of the request in the batch.
//!
//! The trace has a process per place a span ran (replica, process, rank),
//! ordered daemon, scheduler, worker, storage. In each, the spans of one
//! request share a track and the spans of batches share tracks of their
//! own; a span that overlaps an earlier one of its track without nesting in
//! it takes another track of the same name. A membership record is a slice
//! on its request's track, and a flow arrow joins it to the first span of
//! its batch in every process, so a request's batches and their kernels are
//! one click away. Times start at the earliest span, in microseconds.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::admin::Admin;
use crate::perfetto::{document, slice, track};

/// The labels that place a span's process; every other label is an argument
/// of its slice.
const PLACE: [&str; 3] = ["replica", "process", "rank"];

/// The fields that time a span; every other field is an argument of its
/// slice.
const TIME: [&str; 2] = ["start", "end"];

/// One `span` row as `/admin/query` renders it: its labels and fields, each a
/// string.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Span {
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
}

/// The spans a trace covers. Every stated condition holds for each span;
/// none stated selects every span the store retains.
#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// One request's spans, and the spans of every batch it was a member of.
    pub request: Option<String>,
    /// One batch's spans, its membership records among them.
    pub batch: Option<String>,
    /// Spans that end at or after this time, in nanoseconds since the Unix
    /// epoch.
    pub from: Option<i64>,
    /// Spans that start at or before this time, in nanoseconds since the
    /// Unix epoch.
    pub to: Option<i64>,
    /// Spans that end within this span before the daemon's clock.
    pub last: Option<Duration>,
}

impl Filter {
    /// The SQL statement over `events` that selects the filter's spans.
    #[must_use]
    pub fn statement(&self) -> String {
        let mut conditions = vec!["kind = 'span'".to_owned()];

        if let Some(request) = &self.request {
            // The request's own spans and its membership records, and the
            // batch spans of its batches; the membership records of the other
            // requests in those batches stay out.
            let id = literal(request);
            conditions.push(format!(
                "(fields['request'] = {id} OR (fields['request'] IS NULL AND fields['batch'] IN \
                 (SELECT fields['batch'] FROM events WHERE kind = 'span' \
                 AND fields['request'] = {id} AND fields['batch'] IS NOT NULL)))"
            ));
        }
        if let Some(batch) = &self.batch {
            conditions.push(format!("fields['batch'] = {}", literal(batch)));
        }
        if let Some(from) = self.from {
            conditions.push(format!("CAST(fields['end'] AS BIGINT) >= {from}"));
        }
        if let Some(to) = self.to {
            conditions.push(format!("CAST(fields['start'] AS BIGINT) <= {to}"));
        }
        if let Some(last) = self.last {
            conditions.push(format!(
                "CAST(fields['end'] AS BIGINT) >= CAST(now() AS BIGINT) - {}",
                last.as_nanos()
            ));
        }

        format!(
            "SELECT labels, fields FROM events WHERE {}",
            conditions.join(" AND ")
        )
    }
}

/// `text` as an SQL string literal.
fn literal(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// The spans `filter` selects from the daemon `admin` reaches.
///
/// # Errors
///
/// When the daemon cannot be reached or refuses the statement.
pub fn read(admin: &Admin, filter: &Filter) -> Result<Vec<Span>> {
    admin.query(&filter.statement())
}

/// A label's value as a sort key: numbers in numeric order, after the
/// values that are not numbers and the absent one.
type Ordinal = (Option<i64>, Option<String>);

fn ordinal(value: Option<&String>) -> Ordinal {
    (value.and_then(|text| text.parse().ok()), value.cloned())
}

/// Where a span ran: one process of the trace.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Place {
    replica: Ordinal,
    /// The process's position in a request's path, then its name.
    process: (u8, String),
    rank: Ordinal,
}

impl Place {
    fn of(span: &Span) -> Result<Self> {
        let process = span
            .labels
            .get("process")
            .ok_or_else(|| anyhow!("the span states no `process` label"))?;
        let position = match process.as_str() {
            "daemon" => 0,
            "scheduler" => 1,
            "worker" => 2,
            "storage" => 3,
            _ => 4,
        };
        Ok(Self {
            replica: ordinal(span.labels.get("replica")),
            process: (position, process.clone()),
            rank: ordinal(span.labels.get("rank")),
        })
    }

    /// The process's name in the trace, such as `worker (replica 0, rank 3)`.
    fn name(&self) -> String {
        let mut place = Vec::new();
        if let Some(replica) = &self.replica.1 {
            place.push(format!("replica {replica}"));
        }
        if let Some(rank) = &self.rank.1 {
            place.push(format!("rank {rank}"));
        }
        if place.is_empty() {
            self.process.1.clone()
        } else {
            format!("{} ({})", self.process.1, place.join(", "))
        }
    }
}

/// The tracks of a process a span takes: its request's, or the batches'.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Owner {
    Request(String),
    Batches,
    Unowned,
}

impl Owner {
    fn name(&self) -> String {
        match self {
            Self::Request(id) => format!("request {id}"),
            Self::Batches => "batches".to_owned(),
            Self::Unowned => "spans".to_owned(),
        }
    }
}

/// A span read for the trace.
struct Entry {
    place: Place,
    owner: Owner,
    /// Nanoseconds since the Unix epoch.
    start: i64,
    end: i64,
    name: String,
    batch: Option<String>,
    /// Whether the span is a membership record of its request in its batch.
    membership: bool,
    args: Map<String, Value>,
}

impl Entry {
    fn of(span: &Span) -> Result<Self> {
        let time = |key: &str| -> Result<i64> {
            let text = span
                .fields
                .get(key)
                .ok_or_else(|| anyhow!("the span states no `{key}` field"))?;
            text.parse()
                .with_context(|| format!("the span's `{key}` is `{text}`, not nanoseconds"))
        };

        let request = span.fields.get("request");
        let batch = span.fields.get("batch").cloned();
        let membership = request.is_some() && batch.is_some();
        let owner = match (request, &batch) {
            (Some(id), _) => Owner::Request(id.clone()),
            (None, Some(_)) => Owner::Batches,
            (None, None) => Owner::Unowned,
        };

        let name = span
            .fields
            .get("stage")
            .or_else(|| span.fields.get("kernel"))
            .cloned()
            .or_else(|| batch.as_ref().map(|id| format!("batch {id}")))
            .unwrap_or_else(|| "span".to_owned());

        let labels = span
            .labels
            .iter()
            .filter(|(key, _)| !PLACE.contains(&key.as_str()));
        let fields = span
            .fields
            .iter()
            .filter(|(key, _)| !TIME.contains(&key.as_str()));
        let args = labels
            .chain(fields)
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect();

        Ok(Self {
            place: Place::of(span)?,
            owner,
            start: time("start")?,
            end: time("end")?,
            name,
            batch,
            membership,
            args,
        })
    }
}

/// The lane of a track group a slice takes: the first whose open slices
/// either ended by its start or enclose it, so each lane nests properly.
/// Slices arrive by start, the longer first among equal starts; each lane is
/// the stack of its open slices' ends.
fn lane(lanes: &mut Vec<Vec<i64>>, start: i64, end: i64) -> usize {
    for (index, open) in lanes.iter_mut().enumerate() {
        while open.last().is_some_and(|&top| top <= start) {
            open.pop();
        }
        if open.last().is_none_or(|&top| end <= top) {
            open.push(end);
            return index;
        }
    }
    lanes.push(vec![end]);
    lanes.len() - 1
}

/// The trace of `spans`.
///
/// # Errors
///
/// When a span states no `process` label, or no `start` or `end` in
/// nanoseconds.
pub fn convert(spans: &[Span]) -> Result<Value> {
    let entries: Vec<Entry> = spans
        .iter()
        .enumerate()
        .map(|(index, span)| Entry::of(span).with_context(|| format!("span {index}")))
        .collect::<Result<_>>()?;

    let places: BTreeSet<&Place> = entries.iter().map(|entry| &entry.place).collect();
    let pids: BTreeMap<&Place, i64> = places.into_iter().zip(1..).collect();

    let mut events = Vec::new();
    for (place, &pid) in &pids {
        events.push(
            json!({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": place.name()}}),
        );
        events.push(json!({"name": "process_sort_index", "ph": "M", "pid": pid, "args": {"sort_index": pid}}));
    }

    // Each span's track: the groups are laid out in order, a process's
    // tracks numbered as they are first taken.
    let mut groups: BTreeMap<(&Place, &Owner), Vec<usize>> = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        groups
            .entry((&entry.place, &entry.owner))
            .or_default()
            .push(index);
    }
    let mut tracks = vec![(0, 0); entries.len()];
    let mut next: BTreeMap<&Place, i64> = BTreeMap::new();
    for ((place, owner), mut members) in groups {
        let pid = pids[place];
        members.sort_by_key(|&index| (entries[index].start, Reverse(entries[index].end)));

        let mut lanes = Vec::new();
        let mut tids = Vec::new();
        for index in members {
            let entry = &entries[index];
            let lane = lane(&mut lanes, entry.start, entry.end);
            if lane == tids.len() {
                let tid = next.entry(place).or_insert(0);
                *tid += 1;
                tids.push(*tid);

                let name = match lane {
                    0 => owner.name(),
                    _ => format!("{} #{}", owner.name(), lane + 1),
                };
                events.push(track(pid, *tid, &name));
                events.push(json!({"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": *tid, "args": {"sort_index": *tid}}));
            }
            tracks[index] = (pid, tids[lane]);
        }
    }

    for (entry, &(pid, tid)) in entries.iter().zip(&tracks) {
        events.push(slice(
            pid,
            tid,
            &entry.name,
            entry.start,
            entry.end,
            Value::Object(entry.args.clone()),
        ));
    }

    // The first span of each batch in each process, which a membership
    // record's arrows point at.
    let mut firsts: BTreeMap<&str, BTreeMap<i64, usize>> = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        let (Owner::Batches, Some(batch)) = (&entry.owner, &entry.batch) else {
            continue;
        };
        let first = firsts
            .entry(batch)
            .or_default()
            .entry(tracks[index].0)
            .or_insert(index);
        if entry.start < entries[*first].start {
            *first = index;
        }
    }

    // Flow events bind to the slice enclosing their time on their track: the
    // membership record at its start, the batch's span at its start.
    let mut flow = 0;
    for (index, entry) in entries.iter().enumerate() {
        let (true, Some(batch)) = (entry.membership, &entry.batch) else {
            continue;
        };
        let (pid, tid) = tracks[index];
        for &target in firsts
            .get(batch.as_str())
            .into_iter()
            .flat_map(BTreeMap::values)
        {
            flow += 1;
            let (target_pid, target_tid) = tracks[target];
            let name = format!("batch {batch}");
            events.push(json!({"name": name, "cat": "batch", "ph": "s", "id": flow, "pid": pid, "tid": tid, "ts": entry.start}));
            events.push(json!({"name": name, "cat": "batch", "ph": "f", "bp": "e", "id": flow, "pid": target_pid, "tid": target_tid, "ts": entries[target].start}));
        }
    }

    Ok(document(events))
}
