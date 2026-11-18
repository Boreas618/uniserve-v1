//! The storage of the `events` table: the most recent events in the order
//! they were appended.

use std::collections::VecDeque;
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, SystemTime};

use super::Labels;
use super::stats::Limits;

/// The fields of an event, by name.
pub type Fields = serde_json::Map<String, serde_json::Value>;

/// What an event records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A finished request: its arrival, prompt, prefix hit and output
    /// lengths, time to first token, time per output token and status.
    Request,
    /// The outcome of a configuration revision: its writer, changed
    /// options, mode of taking effect, result and duration.
    Revision,
    /// An engine process starting or exiting.
    Process,
    /// An interval with a start and an end: a stage of a request, or a
    /// kernel's execution.
    Span,
    /// A log line: a program's log, a rejected output, a stage of an apply.
    Log,
}

impl Kind {
    /// The kind's name in the `kind` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Revision => "revision",
            Self::Process => "process",
            Self::Span => "span",
            Self::Log => "log",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Kind {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Ok(match name {
            "request" => Self::Request,
            "revision" => Self::Revision,
            "process" => Self::Process,
            "span" => Self::Span,
            "log" => Self::Log,
            _ => return Err(format!("unknown event kind {name:?}")),
        })
    }
}

/// One row of the `events` table.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// The event's position in append order, from 1: the `id` of its SSE
    /// message, which `Last-Event-ID` resumes after.
    pub id: u64,
    /// When the store recorded the event. Times the event itself carries,
    /// such as a span's start and end, are fields.
    pub time: SystemTime,
    pub kind: Kind,
    pub labels: Labels,
    pub fields: Fields,
    /// The configuration revision the event happened under.
    pub revision: u64,
}

/// The retained events, oldest first, with consecutive ids.
#[derive(Debug)]
pub(crate) struct Table {
    rows: VecDeque<Event>,
    next: u64,
}

impl Default for Table {
    fn default() -> Self {
        Self {
            rows: VecDeque::new(),
            next: 1,
        }
    }
}

impl Table {
    /// Appends an event recorded at `now` and returns its id.
    pub(crate) fn append(
        &mut self,
        now: SystemTime,
        kind: Kind,
        labels: Labels,
        fields: Fields,
        revision: u64,
        limits: &Limits,
    ) -> u64 {
        let id = self.next;
        self.next += 1;
        self.rows.push_back(Event {
            id,
            time: now,
            kind,
            labels,
            fields,
            revision,
        });
        self.prune(now, limits);

        id
    }

    /// Drops the events older than the retention, and the oldest beyond the
    /// capacity.
    pub(crate) fn prune(&mut self, now: SystemTime, limits: &Limits) {
        let keep = Duration::from_nanos(u64::try_from(limits.events_keep).unwrap_or(0));
        let cutoff = now.checked_sub(keep).unwrap_or(SystemTime::UNIX_EPOCH);

        while self.rows.len() > limits.capacity {
            self.rows.pop_front();
        }
        while self.rows.front().is_some_and(|event| event.time < cutoff) {
            self.rows.pop_front();
        }
    }

    /// The retained events after `id`, oldest first.
    pub(crate) fn after(&self, id: u64) -> impl DoubleEndedIterator<Item = &Event> {
        let first = self.rows.front().map_or(self.next, |event| event.id);
        let skip =
            usize::try_from(id.saturating_add(1).saturating_sub(first)).unwrap_or(usize::MAX);

        self.rows.range(skip.min(self.rows.len())..)
    }

    /// The id of the last appended event; 0 before the first.
    pub(crate) fn last(&self) -> u64 {
        self.next - 1
    }
}
