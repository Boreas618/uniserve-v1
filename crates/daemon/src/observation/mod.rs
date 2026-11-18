//! The observation store of a deployment: the leader daemon's two tables,
//! queried with SQL through Apache DataFusion.
//!
//! - `stats`: one row is the value of one metric, under one label set and
//!   one configuration revision, in one time bucket. Columns: `time` (the
//!   bucket's start), `name`, `labels`, `value`, `histogram`, `revision`.
//!   `value` is a counter's increase within the bucket, a gauge's latest
//!   reading, or a histogram's mean observation; `histogram` holds a
//!   histogram's distribution (count, sum, extremes and buckets) and is
//!   null for counters and gauges.
//! - `events`: one row is one thing that happened. Columns: `time`, `kind`
//!   (`request`, `revision`, `process`, `span`, `log`), `labels`, `fields`,
//!   `revision`. A field is a string; a field written as another JSON value
//!   is its JSON text.
//!
//! Metric names follow the Prometheus conventions: the `uniserve_` prefix,
//! seconds and bytes, counters ending in `_total`; colons are reserved for
//! recording rules.
//!
//! Writers are the deployment's [`record::Recorder`] (engine reports, finished
//! requests, the HTTP layer, engine processes and the revision served), the
//! apply path and the program host (`Store::add`, `set`, `observe`, `merge`,
//! `append`).
//! Readers are `/admin/query` (`Store::query` with `Rows::json` or
//! `Rows::ipc`, and `Store::subscribe` for an SSE stream of `events`), and
//! control programs (`Store::stats`, `Store::events`).
//!
//! `stats` keeps fine buckets (1 s, an hour by default) and coarse buckets
//! (1 min, seven days by default), and past both a summary per series and
//! revision that stays until the revision is forgotten (`Store::forget`).
//! The table reads them as one timeline without overlap, so an aggregate
//! over it counts every write once. `events` keeps the most recent events
//! (an hour by default, at most `Retention::capacity`).

mod events;
pub(crate) mod export;
pub(crate) mod log;
pub(crate) mod record;
pub(crate) mod spans;
mod stats;
pub mod table;
mod view;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use bytes::Bytes;
use datafusion::error::DataFusionError;
use datafusion::execution::context::{SQLOptions, SessionConfig, SessionContext};
use futures::stream::BoxStream;
use tokio::sync::watch;

pub use self::events::{Event, Fields, Kind};
pub use self::stats::Instrument;
pub use self::view::{Events, Series, Stats, Window};
pub use proto::histogram::Histogram;

/// The labels of a row, by key: its place in the deployment (replica,
/// scheduler, group, rank, device, stream, module, operator, descriptor
/// key), its phase and bucket.
pub type Labels = BTreeMap<String, String>;

/// The label of an event's writer: the start, the operator, the Dynamo
/// sidecar or a program (`Events::writer`).
pub const WRITER: &str = "writer";

/// The most events one SSE message is computed over, which bounds the
/// memory of a subscription that resumes far behind.
const MESSAGE: usize = 4096;

/// The partitions a query runs on. The tables are copies of in-memory rows,
/// so a few partitions saturate a query without taking the daemon's cores.
const PARTITIONS: usize = 4;

/// The source of the times the store records and reads windows against.
pub trait Clock: Send + Sync {
    fn now(&self) -> SystemTime;
}

/// The host's wall clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct System;

impl Clock for System {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

/// The width of a tier's buckets and how long they are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tier {
    pub width: Duration,
    pub keep: Duration,
}

/// How long the store keeps its rows.
///
/// The coarse width is a multiple of the fine width, and the coarse tier
/// is kept at least one coarse bucket longer than the fine tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retention {
    pub fine: Tier,
    pub coarse: Tier,
    /// How long events are kept.
    pub events: Duration,
    /// The most events kept.
    pub capacity: usize,
}

impl Default for Retention {
    fn default() -> Self {
        Self {
            fine: Tier {
                width: Duration::from_secs(1),
                keep: Duration::from_secs(3600),
            },
            coarse: Tier {
                width: Duration::from_secs(60),
                keep: Duration::from_secs(7 * 24 * 3600),
            },
            events: Duration::from_secs(3600),
            capacity: 1 << 20,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("metric name {0:?} is not `uniserve_` followed by letters, digits and underscores")]
    Name(String),
    #[error("metric {name} is a {existing}, not a {requested}")]
    Instrument {
        name: String,
        existing: Instrument,
        requested: Instrument,
    },
    #[error("value {value} of metric {name} is not finite, or a negative counter increment")]
    Value { name: String, value: f64 },
    #[error("retention: {0}")]
    Retention(&'static str),
    #[error(transparent)]
    Query(Box<DataFusionError>),
    #[error(transparent)]
    Encode(Box<ArrowError>),
}

impl From<DataFusionError> for Error {
    fn from(error: DataFusionError) -> Self {
        Self::Query(Box::new(error))
    }
}

impl From<ArrowError> for Error {
    fn from(error: ArrowError) -> Self {
        Self::Encode(Box::new(error))
    }
}

/// Nanoseconds since the Unix epoch; earlier times are the epoch.
fn nanos(time: SystemTime) -> i64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| {
            i64::try_from(since.as_nanos()).unwrap_or(i64::MAX)
        })
}

/// The observation store: a handle that clones to share one store.
#[derive(Clone)]
pub struct Store {
    inner: Arc<Inner>,
}

struct Inner {
    clock: Arc<dyn Clock>,
    limits: stats::Limits,
    stats: Mutex<stats::Table>,
    events: Mutex<events::Table>,

    /// The id of the last appended event, which subscriptions wait on.
    appended: watch::Sender<u64>,
    /// What reads the deployment's spans for a query of `span` events ([`spans`]).
    spans: Mutex<Option<Arc<dyn spans::Source>>>,
}

impl fmt::Debug for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Store")
            .field("limits", &self.inner.limits)
            .finish_non_exhaustive()
    }
}

impl Store {
    /// An empty store reading time from `clock`; fails if the retention is
    /// inconsistent (see `Retention`).
    pub fn new(clock: Arc<dyn Clock>, retention: Retention) -> Result<Self, Error> {
        let limits = stats::Limits::new(&retention)?;

        Ok(Self {
            inner: Arc::new(Inner {
                clock,
                limits,
                stats: Mutex::new(stats::Table::new(limits)),
                events: Mutex::default(),
                appended: watch::Sender::new(0),
                spans: Mutex::new(None),
            }),
        })
    }

    /// The store's current time.
    pub fn now(&self) -> SystemTime {
        self.inner.clock.now()
    }

    fn record(
        &self,
        name: &str,
        labels: &Labels,
        revision: u64,
        instrument: Instrument,
        value: stats::Value,
    ) -> Result<(), Error> {
        let valid = name.strip_prefix("uniserve_").is_some_and(|rest| {
            !rest.is_empty()
                && rest
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        });
        if !valid {
            return Err(Error::Name(name.to_owned()));
        }

        let now = nanos(self.now());
        let mut table = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        table.record(name, labels, revision, instrument, &value, now)
    }

    fn check(name: &str, value: f64) -> Result<(), Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(Error::Value {
                name: name.to_owned(),
                value,
            })
        }
    }

    /// Adds `delta` (finite, not negative) to a counter.
    pub fn add(&self, name: &str, labels: &Labels, revision: u64, delta: f64) -> Result<(), Error> {
        Self::check(name, delta)?;
        if delta < 0.0 {
            return Err(Error::Value {
                name: name.to_owned(),
                value: delta,
            });
        }

        self.record(
            name,
            labels,
            revision,
            Instrument::Counter,
            stats::Value::Number(delta),
        )
    }

    /// Records a gauge's reading.
    pub fn set(&self, name: &str, labels: &Labels, revision: u64, value: f64) -> Result<(), Error> {
        Self::check(name, value)?;
        self.record(
            name,
            labels,
            revision,
            Instrument::Gauge,
            stats::Value::Number(value),
        )
    }

    /// Records one observation of a histogram.
    pub fn observe(
        &self,
        name: &str,
        labels: &Labels,
        revision: u64,
        value: f64,
    ) -> Result<(), Error> {
        Self::check(name, value)?;

        let mut histogram = Histogram::new();
        histogram.observe(value);
        self.record(
            name,
            labels,
            revision,
            Instrument::Histogram,
            stats::Value::Histogram(histogram),
        )
    }

    /// Records the observations of a histogram a report aggregated. An
    /// empty histogram records nothing.
    pub fn merge(
        &self,
        name: &str,
        labels: &Labels,
        revision: u64,
        histogram: &Histogram,
    ) -> Result<(), Error> {
        Self::check(name, histogram.sum())?;
        if histogram.count() == 0 {
            return Ok(());
        }

        self.record(
            name,
            labels,
            revision,
            Instrument::Histogram,
            stats::Value::Histogram(histogram.clone()),
        )
    }

    /// Appends an event recorded now and returns its id; subscriptions
    /// receive it.
    pub fn append(&self, kind: Kind, labels: Labels, fields: Fields, revision: u64) -> u64 {
        let now = self.now();
        let id = {
            let mut table = self
                .inner
                .events
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            table.append(now, kind, labels, fields, revision, &self.inner.limits)
        };
        self.inner.appended.send_replace(id);

        id
    }

    /// Drops the per-revision summaries of `revision`, once the revision
    /// itself is no longer kept.
    pub fn forget(&self, revision: u64) {
        self.inner
            .stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .forget(revision);
    }

    /// Has a query of `span` events read them through `source` from now on: the serving
    /// deployment's processes, or none.
    pub(crate) fn read_spans(&self, source: Option<Arc<dyn spans::Source>>) {
        *self
            .inner
            .spans
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = source;
    }

    /// The spans the source reads, as events; none without a source.
    async fn spans(&self) -> Vec<Event> {
        let source = self
            .inner
            .spans
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        match source {
            Some(source) => source.read().await,
            None => Vec::new(),
        }
    }

    /// The `stats` view of control programs.
    pub fn stats(&self) -> Stats {
        Stats::new(self.clone())
    }

    /// The `events` view of control programs.
    pub fn events(&self) -> Events {
        Events::new(self.clone())
    }

    /// The width of the fine buckets, in nanoseconds.
    pub(crate) fn fine_width(&self) -> i64 {
        self.inner.limits.fine_width
    }

    /// The fine buckets of every series that lie within `[from, to)` (nanoseconds since the Unix
    /// epoch): what an export of the window sends.
    pub(crate) fn points(&self, from: i64, to: i64) -> Vec<stats::Point> {
        let now = nanos(self.now());
        self.read_stats(now, |table| table.fine(from, to))
    }

    /// The events appended after `after`, oldest first, at most `limit` of them.
    pub(crate) fn events_after(&self, after: u64, limit: usize) -> Vec<Event> {
        self.read_events(|table| table.after(after).take(limit).cloned().collect())
    }

    /// Ages the `stats` table to `now` and reads it.
    fn read_stats<T>(&self, now: i64, read: impl FnOnce(&stats::Table) -> T) -> T {
        let mut table = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        table.prune(now);
        read(&table)
    }

    /// Ages the `events` table to now and reads it.
    fn read_events<T>(&self, read: impl FnOnce(&events::Table) -> T) -> T {
        let now = self.now();
        let mut table = self
            .inner
            .events
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        table.prune(now, &self.inner.limits);
        read(&table)
    }

    /// The `stats` rows of the metrics `names` selects (all when `None`).
    fn stats_batch(&self, names: Option<&[String]>) -> Result<RecordBatch, ArrowError> {
        let now = nanos(self.now());
        self.read_stats(now, |table| {
            let mut rows = Vec::new();
            table.rows(names, now, |row| rows.push(row));
            table::stats(&rows)
        })
    }

    /// The `events` rows of the kinds `kinds` selects (all when `None`): the table's, and the
    /// spans the deployment's processes keep when `span` is among them.
    async fn events_batch(&self, kinds: Option<&[String]>) -> Result<RecordBatch, ArrowError> {
        let admits =
            |kind: &str| kinds.is_none_or(|kinds| kinds.iter().any(|admitted| admitted == kind));
        let spans = if admits(Kind::Span.as_str()) {
            self.spans().await
        } else {
            Vec::new()
        };
        self.read_events(|table| {
            let held = table.after(0).filter(|event| admits(event.kind.as_str()));
            table::events(held.chain(&spans))
        })
    }

    fn session() -> SessionContext {
        let config = SessionConfig::new()
            .with_information_schema(true)
            .with_target_partitions(PARTITIONS);
        SessionContext::new_with_config(config)
    }

    /// A DataFusion session with the `stats` and `events` tables, and the
    /// information schema (`SHOW TABLES`, `SHOW COLUMNS FROM stats`).
    pub fn context(&self) -> Result<SessionContext, Error> {
        let context = Self::session();
        context.register_table("stats", Arc::new(table::Stats::new(self.clone())))?;
        context.register_table("events", Arc::new(table::Events::new(self.clone())))?;

        Ok(context)
    }

    /// Runs one read-only SQL query; statements that define, change or
    /// write anything are rejected.
    pub async fn query(&self, sql: &str) -> Result<Rows, Error> {
        run(&self.context()?, sql).await
    }

    /// Opens an SSE stream of a query over `events`: each message is the
    /// query's rows over the events appended since the previous message,
    /// as a JSON array, with the id of the last of those events. It starts
    /// after event `after` (a reconnecting client's `Last-Event-ID`), or
    /// from the oldest retained event; deltas without rows send nothing. An
    /// id beyond the last appended event was issued by an earlier store
    /// (event ids start over with the leader), so the stream starts from
    /// the oldest retained event.
    ///
    /// The query sees only `events`. A query that does not plan fails here;
    /// one that fails over a delta ends the stream with the error. The
    /// stream ends when the store is dropped.
    pub async fn subscribe(
        &self,
        sql: &str,
        after: Option<u64>,
    ) -> Result<BoxStream<'static, Result<Bytes, Error>>, Error> {
        let empty = table::events(std::iter::empty())?;
        delta(empty)?.sql_with_options(sql, options()).await?;

        let last = self.read_events(events::Table::last);
        let subscription = Subscription {
            store: self.clone(),
            sql: sql.to_owned(),
            cursor: after.filter(|&after| after <= last).unwrap_or(0),
            appended: self.inner.appended.subscribe(),
        };
        Ok(Box::pin(futures::stream::unfold(
            Some(subscription),
            |subscription| async move {
                let mut subscription = subscription?;
                match subscription.next().await {
                    Some(Ok(message)) => Some((Ok(message), Some(subscription))),
                    Some(Err(error)) => Some((Err(error), None)),
                    None => None,
                }
            },
        )))
    }
}

/// The options of `/admin/query`: queries only.
fn options() -> SQLOptions {
    SQLOptions::new()
        .with_allow_ddl(false)
        .with_allow_dml(false)
        .with_allow_statements(false)
}

async fn run(context: &SessionContext, sql: &str) -> Result<Rows, Error> {
    let frame = context.sql_with_options(sql, options()).await?;
    let schema = Arc::clone(frame.schema().inner());
    let batches = frame.collect().await?;

    Ok(Rows { schema, batches })
}

/// A session whose `events` table is one batch of events.
fn delta(batch: RecordBatch) -> Result<SessionContext, Error> {
    let context = Store::session();
    context.register_table("events", Arc::new(table::Events::batch(batch)))?;

    Ok(context)
}

/// The state of one SSE stream.
struct Subscription {
    store: Store,
    sql: String,
    /// The id of the last event a message has covered.
    cursor: u64,
    appended: watch::Receiver<u64>,
}

impl Subscription {
    /// The next message; `None` once the store is gone.
    async fn next(&mut self) -> Option<Result<Bytes, Error>> {
        loop {
            // Marking the current id seen before reading makes an append
            // after the read wake `changed`.
            self.appended.borrow_and_update();

            let cursor = self.cursor;
            let (batch, last) = self.store.read_events(|table| {
                let mut last = None;
                let events = table
                    .after(cursor)
                    .take(MESSAGE)
                    .inspect(|event| last = Some(event.id));
                (table::events(events), last)
            });

            let Some(last) = last else {
                self.appended.changed().await.ok()?;
                continue;
            };
            self.cursor = last;

            let rows = match batch {
                Ok(batch) => match delta(batch) {
                    Ok(context) => run(&context, &self.sql).await,
                    Err(error) => Err(error),
                },
                Err(error) => Err(error.into()),
            };
            match rows {
                Ok(rows) if rows.is_empty() => continue,
                Ok(rows) => return Some(rows.json().map(|data| message(last, &data))),
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

/// One SSE message: the id, then the data, a `data:` line per line.
fn message(id: u64, data: &[u8]) -> Bytes {
    let mut message = format!("id: {id}\n").into_bytes();
    for line in data.split(|&byte| byte == b'\n') {
        message.extend_from_slice(b"data: ");
        message.extend_from_slice(line);
        message.push(b'\n');
    }
    message.push(b'\n');

    Bytes::from(message)
}

/// The result of a query.
#[derive(Clone, Debug)]
pub struct Rows {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

impl Rows {
    pub fn is_empty(&self) -> bool {
        self.batches.iter().all(|batch| batch.num_rows() == 0)
    }

    /// The rows as a JSON array of objects, a key per column; nulls are
    /// written, maps are objects, times are RFC 3339 strings.
    pub fn json(&self) -> Result<Bytes, Error> {
        let mut writer = arrow::json::WriterBuilder::new()
            .with_explicit_nulls(true)
            .build::<_, arrow::json::writer::JsonArray>(Vec::new());
        for batch in &self.batches {
            writer.write(batch)?;
        }
        writer.finish()?;

        let json = writer.into_inner();
        // The writer writes nothing for no rows.
        Ok(if json.is_empty() {
            Bytes::from_static(b"[]")
        } else {
            Bytes::from(json)
        })
    }

    /// The rows as an Arrow IPC stream.
    pub fn ipc(&self) -> Result<Bytes, Error> {
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(Vec::new(), &self.schema)?;
        for batch in &self.batches {
            writer.write(batch)?;
        }
        writer.finish()?;

        Ok(Bytes::from(writer.into_inner()?))
    }
}

/// The response format of `/admin/query`, by the request's `Accept`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `application/json` (`Rows::json`), the default.
    Json,
    /// `application/vnd.apache.arrow.stream` (`Rows::ipc`).
    Ipc,
    /// `text/event-stream` (`Store::subscribe`).
    Sse,
}

impl Format {
    /// The format an `Accept` header prefers: of the media ranges it names,
    /// the known one of the highest quality, the earliest among equals;
    /// JSON when it names none, or is absent.
    pub fn negotiate(accept: Option<&str>) -> Self {
        let mut best = (Self::Json, 0.0);
        for range in accept.unwrap_or_default().split(',') {
            let mut parameters = range.split(';').map(str::trim);
            let format = match parameters.next().map(str::to_ascii_lowercase).as_deref() {
                Some("application/json") => Self::Json,
                Some("application/vnd.apache.arrow.stream") => Self::Ipc,
                Some("text/event-stream") => Self::Sse,
                _ => continue,
            };
            let quality = parameters
                .filter_map(|parameter| parameter.strip_prefix("q="))
                .find_map(|quality| quality.parse::<f64>().ok())
                .unwrap_or(1.0);
            if quality > best.1 {
                best = (format, quality);
            }
        }

        best.0
    }

    /// The format's media type, the response's `Content-Type`.
    pub fn media(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Ipc => "application/vnd.apache.arrow.stream",
            Self::Sse => "text/event-stream",
        }
    }
}
