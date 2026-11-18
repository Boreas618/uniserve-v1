//! The `span` events of the `events` table: the stages and timed kernels of every request, which
//! stay in the rings of the processes that recorded them (`service::span`) until a query reads
//! them.
//!
//! A query whose `kind` condition admits `span` (or that states none) has the store read them
//! through its [`Source`], which the leader sets while its deployment serves: every replica's
//! engine reads its processes' rings and every node's daemon its own and its Storage Server's
//! ([`Leader::spans`](crate::leader::Leader::spans)), and each span becomes one event
//! ([`event`]). Only the spans that ended within `profiling_config.span_window` before the query
//! are read. The rows are the query's own: nothing is appended to the table, so an SSE stream
//! of `events` carries none.
//!
//! A span's labels are where it ran: `process` (`daemon`, `scheduler`, `worker`, `storage`), the
//! `replica` of an engine's, the `host` of a daemon's or a Storage Server's, a scheduler's
//! `scheduler`, a worker's `group` and `rank`, and a kernel's `op`, `descriptor` and `module`.
//! Its fields are `start` and `end` (nanoseconds since the Unix epoch on the host clock), the
//! `request` (the trace id the daemon gave the request) or the `batch` (`<epoch>.<batch id>`,
//! which names one batch in the deployment), a stage's `stage` or a kernel's `kernel` (its
//! operator) with the `layers` it stands for, and the `bytes` a stage moved.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use futures::future::BoxFuture;
use service::span::{Label, Name, Span};

use super::{Event, Fields, Kind, Labels};

/// What reads a deployment's spans from its processes' rings.
pub(crate) trait Source: Send + Sync {
    /// The spans the deployment's processes keep that ended within its window, as events.
    fn read(&self) -> BoxFuture<'static, Vec<Event>>;
}

/// Where a process that recorded spans runs, as the reader that gathered them knows it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Place<'a> {
    /// The replica whose engine recorded them.
    pub(crate) replica: Option<usize>,
    /// The epoch that engine executes, which qualifies its batch ids.
    pub(crate) epoch: Option<u64>,
    /// The node of a daemon's or a Storage Server's spans.
    pub(crate) host: Option<&'a str>,
}

/// The event of `span`, which a process at `place` recorded, under configuration revision
/// `revision`. Its time is the span's start.
pub(crate) fn event(span: &Span, place: Place<'_>, revision: u64) -> Event {
    let mut labels = Labels::new();
    labels.insert("process".to_owned(), span.process.name().to_owned());
    if let Some(replica) = place.replica {
        labels.insert("replica".to_owned(), replica.to_string());
    }
    if let Some(host) = place.host {
        labels.insert("host".to_owned(), host.to_owned());
    }
    match span.label {
        Label::None => {}
        Label::Worker { group, rank } => {
            labels.insert("group".to_owned(), group.to_string());
            labels.insert("rank".to_owned(), rank.to_string());
        }
        Label::Scheduler(scheduler) => {
            labels.insert("scheduler".to_owned(), scheduler.to_string());
        }
    }

    let mut fields = Fields::new();
    fields.insert("start".to_owned(), span.start.to_string().into());
    fields.insert("end".to_owned(), span.end.to_string().into());
    if let Some(request) = span.request {
        fields.insert("request".to_owned(), request.to_string().into());
    }
    if let Some(batch) = span.batch {
        let batch = match place.epoch {
            Some(epoch) => format!("{epoch}.{batch}"),
            None => batch.to_string(),
        };
        fields.insert("batch".to_owned(), batch.into());
    }
    match &span.name {
        Name::Stage(stage) => {
            fields.insert("stage".to_owned(), stage.clone().into());
        }
        Name::Kernel(kernel) => {
            labels.insert("op".to_owned(), kernel.op.clone());
            labels.insert("descriptor".to_owned(), kernel.descriptor.clone());
            labels.insert("module".to_owned(), kernel.module.clone());
            fields.insert("kernel".to_owned(), kernel.op.clone().into());
            fields.insert("layers".to_owned(), kernel.layers.to_string().into());
        }
    }
    if let Some(bytes) = span.bytes {
        fields.insert("bytes".to_owned(), bytes.to_string().into());
    }

    Event {
        id: 0,
        time: SystemTime::UNIX_EPOCH + Duration::from_nanos(span.start),
        kind: Kind::Span,
        labels,
        fields,
        revision,
    }
}

/// The start of `window` before `now`, in nanoseconds since the Unix epoch: a read of spans
/// leaves out each span that ended before it.
pub(crate) fn since(now: SystemTime, window: Duration) -> u64 {
    now.checked_sub(window)
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
        })
}

/// The trace id of the next request the daemon receives: the daemon's start second in the high
/// 24 of the 48 bits a span names, and a count in the low 24, so the ids of one daemon's life
/// are distinct and apart from those an earlier daemon gave.
pub(crate) fn next_trace() -> u64 {
    static NEXT: LazyLock<AtomicU64> = LazyLock::new(|| {
        let seconds = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        AtomicU64::new(((seconds & 0xff_ffff) << 24) | 1)
    });
    NEXT.fetch_add(1, Ordering::Relaxed) % service::span::IDS
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use service::span::{Kernel, Process};

    use super::super::{Retention, Store, System};
    use super::*;

    fn span(process: Process, label: Label, name: Name, start: u64) -> Span {
        Span {
            start,
            end: start + 10,
            process,
            label,
            name,
            request: None,
            batch: None,
            bytes: None,
        }
    }

    fn stage(name: &str) -> Name {
        Name::Stage(name.to_owned())
    }

    /// A worker's kernel span is labelled with its replica, group, rank, operator, descriptor
    /// and module, and its batch is qualified by its epoch; a Storage Server's names its host
    /// and states its bytes.
    #[test]
    fn a_span_is_labelled_by_where_it_ran() {
        let kernel = Span {
            batch: Some(41),
            ..span(
                Process::Worker,
                Label::Worker { group: 1, rank: 3 },
                Name::Kernel(Kernel {
                    op: "gemm".to_owned(),
                    descriptor: "bf16_4096".to_owned(),
                    module: "model.layers.0.mlp".to_owned(),
                    layers: 48,
                }),
                1_000,
            )
        };
        let place = Place {
            replica: Some(2),
            epoch: Some(7),
            host: None,
        };
        let event = event(&kernel, place, 5);

        let label = |key: &str| event.labels.get(key).map(String::as_str);
        assert_eq!(label("process"), Some("worker"));
        assert_eq!(label("replica"), Some("2"));
        assert_eq!((label("group"), label("rank")), (Some("1"), Some("3")));
        assert_eq!(
            (label("op"), label("descriptor"), label("module")),
            (Some("gemm"), Some("bf16_4096"), Some("model.layers.0.mlp"))
        );
        let field = |key: &str| event.fields.get(key).and_then(|value| value.as_str());
        assert_eq!((field("start"), field("end")), (Some("1000"), Some("1010")));
        assert_eq!(field("batch"), Some("7.41"));
        assert_eq!(
            (field("kernel"), field("layers")),
            (Some("gemm"), Some("48"))
        );
        assert_eq!(field("request"), None);
        assert_eq!(event.kind, Kind::Span);
        assert_eq!(event.revision, 5);

        let handoff = Span {
            request: Some(9),
            bytes: Some(4096),
            ..span(Process::Storage, Label::None, stage("handoff"), 2_000)
        };
        let place = Place {
            host: Some("node-1"),
            ..Place::default()
        };
        let event = super::event(&handoff, place, 5);
        assert_eq!(event.labels.get("host").map(String::as_str), Some("node-1"));
        assert_eq!(event.fields.get("bytes"), Some(&"4096".into()));
        assert_eq!(event.fields.get("request"), Some(&"9".into()));
        assert_eq!(event.fields.get("stage"), Some(&"handoff".into()));
    }

    /// A read's window starts its length before now, and at the Unix epoch when it is longer
    /// than the time since.
    #[test]
    fn a_window_starts_its_length_before_now() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        assert_eq!(since(now, Duration::from_secs(10)), 90_000_000_000);
        assert_eq!(since(now, Duration::from_secs(1_000)), 0);
    }

    /// The spans a source reads, as the leader's does.
    struct Fixed {
        events: Vec<Event>,
        reads: AtomicUsize,
    }

    impl Source for Fixed {
        fn read(&self) -> BoxFuture<'static, Vec<Event>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let events = self.events.clone();
            Box::pin(async move { events })
        }
    }

    /// A query of `span` events reads them through the store's source beside the rows it
    /// holds; a query of another kind reads none. The timeline's selection of a request (its
    /// own spans, its membership records and the spans of its batches) selects them by SQL.
    #[tokio::test]
    async fn a_span_query_reads_the_deployment_s_spans_by_sql() {
        let store = Store::new(Arc::new(System), Retention::default()).unwrap();
        store.append(Kind::Log, Labels::new(), Fields::new(), 1);

        let at = |process, label, name, start| span(process, label, name, start);
        let worker = Label::Worker { group: 0, rank: 0 };
        let spans = [
            Span {
                request: Some(9),
                ..at(Process::Daemon, Label::None, stage("tokenize"), 100)
            },
            Span {
                request: Some(9),
                batch: Some(41),
                ..at(Process::Scheduler, Label::Scheduler(0), stage("batch"), 200)
            },
            Span {
                request: Some(8),
                batch: Some(41),
                ..at(Process::Scheduler, Label::Scheduler(0), stage("batch"), 200)
            },
            Span {
                batch: Some(41),
                ..at(Process::Worker, worker, stage("launch"), 300)
            },
            Span {
                batch: Some(42),
                ..at(Process::Worker, worker, stage("launch"), 400)
            },
        ];
        let place = Place {
            replica: Some(0),
            epoch: Some(3),
            host: None,
        };
        let source = Arc::new(Fixed {
            events: spans.iter().map(|span| event(span, place, 1)).collect(),
            reads: AtomicUsize::new(0),
        });
        store.read_spans(Some(Arc::clone(&source) as Arc<dyn Source>));

        let rows = |sql: &'static str| {
            let store = store.clone();
            async move {
                let rows = store.query(sql).await.unwrap();
                let json: serde_json::Value =
                    serde_json::from_slice(&rows.json().unwrap()).unwrap();
                json.as_array().unwrap().clone()
            }
        };
        assert_eq!(
            rows("SELECT * FROM events WHERE kind = 'log'").await.len(),
            1
        );
        assert_eq!(source.reads.load(Ordering::Relaxed), 0);
        assert_eq!(rows("SELECT * FROM events").await.len(), 6);

        let request = rows(
            "SELECT fields['stage'] AS stage, fields['batch'] AS batch FROM events \
             WHERE kind = 'span' AND (fields['request'] = '9' OR (fields['request'] IS NULL \
             AND fields['batch'] IN (SELECT fields['batch'] FROM events WHERE kind = 'span' \
             AND fields['request'] = '9' AND fields['batch'] IS NOT NULL))) \
             ORDER BY CAST(fields['start'] AS BIGINT)",
        )
        .await;
        let read: Vec<(String, Option<String>)> = request
            .iter()
            .map(|row| {
                (
                    row["stage"].as_str().unwrap().to_owned(),
                    row["batch"].as_str().map(str::to_owned),
                )
            })
            .collect();
        assert_eq!(
            read,
            [
                ("tokenize".to_owned(), None),
                ("batch".to_owned(), Some("3.41".to_owned())),
                ("launch".to_owned(), Some("3.41".to_owned())),
            ]
        );

        // Without a source, nothing is read.
        store.read_spans(None);
        assert_eq!(
            rows("SELECT * FROM events WHERE kind = 'span'").await.len(),
            0
        );
    }
}
