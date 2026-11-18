//! The behavior of the observation store through its public interface,
//! against a clock the tests set.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float64Type, UInt64Type};
use futures::StreamExt;

use super::*;

/// A clock that stands still until a test moves it.
struct Manual(Mutex<SystemTime>);

impl Clock for Manual {
    fn now(&self) -> SystemTime {
        *self.0.lock().unwrap()
    }
}

/// A store whose clock starts at the Unix epoch plus a whole day, so
/// every bucket boundary is a round number of seconds.
fn store(retention: Retention) -> (Store, Arc<Manual>) {
    let clock = Arc::new(Manual(Mutex::new(
        SystemTime::UNIX_EPOCH + Duration::from_secs(86_400),
    )));
    let store = Store::new(Arc::clone(&clock) as Arc<dyn Clock>, retention).unwrap();
    (store, clock)
}

/// Sets the clock to `seconds` after the start.
fn at(clock: &Manual, seconds: f64) {
    *clock.0.lock().unwrap() =
        SystemTime::UNIX_EPOCH + Duration::from_secs(86_400) + Duration::from_secs_f64(seconds);
}

fn labels(pairs: &[(&str, &str)]) -> Labels {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn fields(pairs: &[(&str, serde_json::Value)]) -> Fields {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// A short retention: 1 s buckets kept 10 s, 5 s buckets kept 30 s,
/// events kept 20 s and at most 4 of them.
fn short() -> Retention {
    Retention {
        fine: Tier {
            width: Duration::from_secs(1),
            keep: Duration::from_secs(10),
        },
        coarse: Tier {
            width: Duration::from_secs(5),
            keep: Duration::from_secs(30),
        },
        events: Duration::from_secs(20),
        capacity: 4,
    }
}

/// The rows of one batch-concatenated result as JSON values.
async fn json(store: &Store, sql: &str) -> serde_json::Value {
    let rows = store.query(sql).await.unwrap();
    serde_json::from_slice(&rows.json().unwrap()).unwrap()
}

#[tokio::test]
async fn design_queries_read_written_rows() {
    let (store, clock) = store(Retention::default());
    let kernels = "uniserve_kernel_launches_total";
    let gemm = labels(&[("op", "gemm"), ("descriptor", "bf16")]);
    let norm = labels(&[("op", "rms_norm"), ("descriptor", "4096")]);

    at(&clock, 0.0);
    store.add(kernels, &gemm, 41, 96.0).unwrap();
    store.add(kernels, &norm, 41, 48.0).unwrap();
    at(&clock, 1.0);
    store.add(kernels, &gemm, 42, 96.0).unwrap();
    store.add(kernels, &norm, 42, 48.0).unwrap();

    // One observation per second, so each row's mean is the observation.
    let step = "uniserve_step_device_seconds";
    let decode = labels(&[("phase", "decode")]);
    let prefill = labels(&[("phase", "prefill")]);
    for (second, seconds) in [0.010, 0.030, 0.020, 0.050, 0.040].into_iter().enumerate() {
        at(&clock, 10.0 + second as f64);
        store.observe(step, &decode, 42, seconds).unwrap();
        store.observe(step, &prefill, 42, 1.0).unwrap();
    }
    at(&clock, 20.0);
    store.observe(step, &decode, 41, 9.0).unwrap();

    let inventory = json(
        &store,
        "SELECT labels['op'], labels['descriptor'], sum(value) AS launches \
         FROM stats WHERE name = 'uniserve_kernel_launches_total' \
         GROUP BY 1, 2 ORDER BY launches DESC",
    )
    .await;
    let launches: Vec<f64> = inventory
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["launches"].as_f64().unwrap())
        .collect();
    assert_eq!(launches, [192.0, 96.0]);
    let op = inventory[0]
        .as_object()
        .unwrap()
        .iter()
        .find(|(column, _)| column.contains("op"))
        .unwrap()
        .1;
    assert_eq!(op, "gemm");

    let median = store
        .query(
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS median FROM stats \
             WHERE name = 'uniserve_step_device_seconds' AND labels['phase'] = 'decode' AND revision = 42",
        )
        .await
        .unwrap();
    let median = median.batches[0]
        .column(0)
        .as_primitive::<Float64Type>()
        .value(0);
    assert!((median - 0.030).abs() < 1e-12, "{median}");

    // A histogram row holds its distribution as well.
    let rows = json(
        &store,
        "SELECT histogram FROM stats WHERE name = 'uniserve_step_device_seconds' AND revision = 41",
    )
    .await;
    assert_eq!(rows[0]["histogram"]["count"], 1);
    assert_eq!(rows[0]["histogram"]["max"], 9.0);
    let rows = json(
        &store,
        "SELECT histogram FROM stats WHERE name = 'uniserve_kernel_launches_total' LIMIT 1",
    )
    .await;
    assert!(rows[0]["histogram"].is_null());
}

#[tokio::test]
async fn buckets_fold_writes_and_age_into_coarse_buckets() {
    let (store, clock) = store(short());
    let name = "uniserve_requests_total";
    let gauge = "uniserve_queue_requests";
    let none = Labels::new();

    for (second, delta) in [(0.2, 1.0), (0.7, 1.0), (1.5, 1.0), (3.2, 2.0)] {
        at(&clock, second);
        store.add(name, &none, 1, delta).unwrap();
        store.set(gauge, &none, 1, delta * 10.0).unwrap();
    }

    let sql = "SELECT CAST(time AS BIGINT) / 1000000000 - 86400 AS second, value FROM stats \
               WHERE name = '{}' ORDER BY time";
    let series = |rows: serde_json::Value| -> Vec<(i64, f64)> {
        rows.as_array()
            .unwrap()
            .iter()
            .map(|row| {
                (
                    row["second"].as_i64().unwrap(),
                    row["value"].as_f64().unwrap(),
                )
            })
            .collect()
    };

    // Fine buckets: writes within one second fold, a counter adding and a
    // gauge keeping its last reading.
    at(&clock, 4.0);
    assert_eq!(
        series(json(&store, &sql.replace("{}", name)).await),
        [(0, 2.0), (1, 1.0), (3, 2.0)]
    );
    assert_eq!(
        series(json(&store, &sql.replace("{}", gauge)).await),
        [(0, 10.0), (1, 10.0), (3, 20.0)]
    );

    // Once the fine tier no longer shows them, the seconds read as one
    // coarse bucket, and the sum over the table is unchanged.
    at(&clock, 14.0);
    assert_eq!(
        series(json(&store, &sql.replace("{}", name)).await),
        [(0, 5.0)]
    );
    assert_eq!(
        series(json(&store, &sql.replace("{}", gauge)).await),
        [(0, 20.0)]
    );

    // A write now lands in the fine tier after the coarse bucket.
    store.add(name, &none, 1, 1.0).unwrap();
    assert_eq!(
        series(json(&store, &sql.replace("{}", name)).await),
        [(0, 5.0), (14, 1.0)]
    );
}

#[tokio::test]
async fn retention_retires_old_buckets_and_drops_old_events() {
    let (store, clock) = store(short());
    let name = "uniserve_requests_total";
    let none = Labels::new();

    at(&clock, 0.0);
    store.add(name, &none, 7, 3.0).unwrap();
    at(&clock, 6.0);
    store.add(name, &none, 7, 4.0).unwrap();
    store.add(name, &none, 8, 1.0).unwrap();

    // Past the coarse retention, a series' buckets fold into its revision
    // summary, which starts where its first bucket did.
    at(&clock, 60.0);
    let rows = json(&store, "SELECT revision, value, CAST(time AS BIGINT) / 1000000000 - 86400 AS second FROM stats ORDER BY revision").await;
    assert_eq!(
        rows,
        serde_json::json!([{"revision": 7, "value": 7.0, "second": 0}, {"revision": 8, "value": 1.0, "second": 5}])
    );

    // Forgetting a revision drops its summaries.
    store.forget(7);
    let rows = json(&store, "SELECT revision FROM stats").await;
    assert_eq!(rows, serde_json::json!([{"revision": 8}]));

    // Events: kept 20 s, and at most 4.
    for second in 0..6 {
        at(&clock, 100.0 + f64::from(second) * 5.0);
        store.append(Kind::Log, Labels::new(), fields(&[("n", second.into())]), 8);
    }
    let rows = json(&store, "SELECT fields['n'] AS n FROM events").await;
    assert_eq!(
        rows,
        serde_json::json!([{"n": "2"}, {"n": "3"}, {"n": "4"}, {"n": "5"}])
    );
    at(&clock, 136.0);
    let rows = json(&store, "SELECT fields['n'] AS n FROM events").await;
    assert_eq!(rows, serde_json::json!([{"n": "4"}, {"n": "5"}]));
}

#[tokio::test]
async fn json_and_arrow_encode_the_same_rows() {
    let (store, clock) = store(Retention::default());
    at(&clock, 2.5);
    store.append(
        Kind::Request,
        labels(&[("replica", "0")]),
        fields(&[
            ("ttft", serde_json::json!(0.125)),
            ("status", "stop".into()),
        ]),
        3,
    );

    let rows = store.query("SELECT * FROM events").await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&rows.json().unwrap()).unwrap();
    assert_eq!(
        json,
        serde_json::json!([{
            "time": "1970-01-02T00:00:02.500Z",
            "kind": "request",
            "labels": {"replica": "0"},
            "fields": {"ttft": "0.125", "status": "stop"},
            "revision": 3,
        }])
    );

    let ipc = rows.ipc().unwrap();
    let reader =
        arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(ipc.to_vec()), None)
            .unwrap();
    assert_eq!(reader.schema(), *table::EVENTS);
    let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
    assert_eq!(batches, rows.batches);

    let empty = store
        .query("SELECT * FROM events WHERE kind = 'span'")
        .await
        .unwrap();
    assert_eq!(&empty.json().unwrap()[..], b"[]");
}

#[tokio::test]
async fn subscription_replays_after_an_id_then_follows() {
    let (store, clock) = store(Retention::default());
    at(&clock, 0.0);
    for text in ["a", "b", "c"] {
        store.append(
            Kind::Log,
            Labels::new(),
            fields(&[("text", text.into())]),
            1,
        );
    }
    store.append(Kind::Span, Labels::new(), Fields::new(), 1);

    let sql = "SELECT fields['text'] AS text FROM events WHERE kind = 'log'";
    let mut stream = store.subscribe(sql, Some(1)).await.unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(
        &first[..],
        b"id: 4\ndata: [{\"text\":\"b\"},{\"text\":\"c\"}]\n\n"
    );

    // A span matches nothing and sends nothing; the next log does.
    let writer = store.clone();
    let append = tokio::spawn(async move {
        writer.append(Kind::Span, Labels::new(), Fields::new(), 1);
        writer.append(Kind::Log, Labels::new(), fields(&[("text", "d".into())]), 1);
    });
    let next = stream.next().await.unwrap().unwrap();
    append.await.unwrap();
    assert_eq!(&next[..], b"id: 6\ndata: [{\"text\":\"d\"}]\n\n");

    // An id from an earlier store replays everything retained.
    let mut stream = store.subscribe(sql, Some(100)).await.unwrap();
    let replay = stream.next().await.unwrap().unwrap();
    assert!(
        replay.starts_with(b"id: 6\ndata: [{\"text\":\"a\"}"),
        "{replay:?}"
    );

    // A query that does not plan fails before streaming.
    assert!(store.subscribe("SELECT * FROM stats", None).await.is_err());
}

#[tokio::test]
async fn windows_read_quantiles_rates_and_latest_values() {
    let (store, clock) = store(Retention::default());
    let tpot = "uniserve_tpot_seconds";
    let none = Labels::new();

    // Outside the window: a large observation that would move the p99.
    at(&clock, 0.0);
    store.observe(tpot, &none, 1, 5.0).unwrap();

    // Within it: 1..=100 ms over ten seconds, half from a reported histogram.
    let mut reported = Histogram::new();
    for millis in 1..=100 {
        at(&clock, 20.0 + f64::from(millis % 10));
        if millis % 2 == 0 {
            reported.observe(f64::from(millis) / 1000.0);
        } else {
            store
                .observe(tpot, &none, 1, f64::from(millis) / 1000.0)
                .unwrap();
        }
    }
    store.merge(tpot, &none, 1, &reported).unwrap();

    at(&clock, 30.0);
    let window = store.stats().series(tpot).window(Duration::from_secs(15));
    let p99 = window.p99().unwrap();
    let p50 = window.p50().unwrap();
    assert!((p99 - 0.099).abs() / 0.099 < 0.09, "{p99}");
    assert!((p50 - 0.050).abs() / 0.050 < 0.09, "{p50}");
    assert!((window.mean().unwrap() - 0.0505).abs() < 1e-9);
    assert!((window.rate().unwrap() - 100.0 / 15.0).abs() < 1e-9);
    assert_eq!(
        store
            .stats()
            .series(tpot)
            .window(Duration::from_secs(60))
            .quantile(1.0),
        Some(5.0)
    );

    // Counters: the increase and its rate; a gauge: the total of each
    // label set's latest reading, or one label set's.
    let tokens = "uniserve_generation_tokens_total";
    let pages = "uniserve_kv_pages";
    for second in 0..10 {
        at(&clock, 100.0 + f64::from(second));
        for replica in ["0", "1"] {
            let replica = labels(&[("replica", replica)]);
            store.add(tokens, &replica, 2, 50.0).unwrap();
            store.set(pages, &replica, 2, f64::from(second)).unwrap();
        }
    }
    let window = store.stats().series(tokens).window(Duration::from_secs(10));
    assert_eq!(window.sum(), Some(1000.0));
    assert_eq!(window.rate(), Some(100.0));
    assert_eq!(window.last(), Some(100.0));
    let replica = store
        .stats()
        .series(tokens)
        .label("replica", "1")
        .revision(2)
        .window(Duration::from_secs(10));
    assert_eq!(replica.sum(), Some(500.0));
    assert_eq!(
        store
            .stats()
            .series(tokens)
            .revision(3)
            .window(Duration::from_secs(10))
            .sum(),
        None
    );

    let window = store.stats().series(pages).window(Duration::from_secs(10));
    assert_eq!(window.last(), Some(18.0));
    assert_eq!(window.rate(), Some(2.0));
    assert_eq!(
        store
            .stats()
            .series(pages)
            .label("replica", "0")
            .window(Duration::from_secs(10))
            .last(),
        Some(9.0)
    );
    assert_eq!(
        store
            .stats()
            .series("uniserve_absent")
            .window(Duration::from_secs(10))
            .p99(),
        None
    );
}

/// A selection split by a label is one window per value, each combining the series that carry
/// it as a whole selection combines its series; a series without the label is in none, and a
/// value with no bucket in the window is absent.
#[tokio::test]
async fn a_selection_split_by_a_label_combines_each_values_series() {
    let (store, clock) = store(Retention::default());
    let tokens = "uniserve_expert_tokens_total";
    let pages = "uniserve_kv_pages";

    // Old tokens of expert 9, outside the window.
    at(&clock, 0.0);
    let old = labels(&[("layer", "l0"), ("expert", "9"), ("scheduler", "0")]);
    store.add(tokens, &old, 1, 1000.0).unwrap();

    for second in 0..10 {
        at(&clock, 100.0 + f64::from(second));
        for scheduler in ["0", "1"] {
            for (expert, count) in [("0", 3.0), ("1", 1.0)] {
                let at = labels(&[
                    ("layer", "l0"),
                    ("expert", expert),
                    ("scheduler", scheduler),
                ]);
                store.add(tokens, &at, 1, count).unwrap();
            }
            let at = labels(&[("replica", scheduler)]);
            store.set(pages, &at, 1, f64::from(second)).unwrap();
        }
        store
            .add(tokens, &labels(&[("layer", "l0")]), 1, 7.0)
            .unwrap();
    }

    let span = Duration::from_secs(10);
    let experts = store
        .stats()
        .series(tokens)
        .label("layer", "l0")
        .by(span, "expert");
    let sums: Vec<(&str, Option<f64>)> = experts
        .iter()
        .map(|(expert, window)| (expert.as_str(), window.sum()))
        .collect();
    assert_eq!(sums, [("0", Some(60.0)), ("1", Some(20.0))]);

    // A gauge's latest value per group is the total of its series' latest readings.
    let replicas = store.stats().series(pages).by(span, "replica");
    assert_eq!(replicas["0"].last(), Some(9.0));
    assert_eq!(replicas["1"].last(), Some(9.0));
    assert!(
        store
            .stats()
            .series("uniserve_absent")
            .by(span, "replica")
            .is_empty()
    );
}

#[tokio::test]
async fn event_views_select_by_kind_and_writer() {
    let (store, clock) = store(Retention::default());
    at(&clock, 0.0);
    store.append(
        Kind::Revision,
        labels(&[(WRITER, "chunk")]),
        fields(&[("result", "committed".into())]),
        2,
    );
    at(&clock, 5.0);
    store.append(
        Kind::Revision,
        labels(&[(WRITER, "operator")]),
        fields(&[("result", "rolled_back".into())]),
        3,
    );
    store.append(Kind::Log, labels(&[(WRITER, "chunk")]), Fields::new(), 3);

    let revisions = store.events().kind(Kind::Revision);
    assert_eq!(revisions.last().unwrap().fields["result"], "rolled_back");
    assert_eq!(
        revisions.clone().writer("chunk").last().unwrap().revision,
        2
    );
    assert_eq!(
        store.events().writer("chunk").last().unwrap().kind,
        Kind::Log
    );
    assert_eq!(revisions.window(Duration::from_secs(2)).len(), 1);
    assert!(store.events().kind(Kind::Process).last().is_none());
}

#[test]
fn writes_keep_names_and_instruments_consistent() {
    let (store, _) = store(Retention::default());
    let none = Labels::new();

    assert!(matches!(
        store.add("requests_total", &none, 1, 1.0),
        Err(Error::Name(_))
    ));
    assert!(matches!(
        store.add("uniserve:requests_total", &none, 1, 1.0),
        Err(Error::Name(_))
    ));
    assert!(matches!(
        store.add("uniserve_requests_total", &none, 1, -1.0),
        Err(Error::Value { .. })
    ));
    assert!(matches!(
        store.set("uniserve_queue_requests", &none, 1, f64::NAN),
        Err(Error::Value { .. })
    ));
    store.add("uniserve_requests_total", &none, 1, 1.0).unwrap();
    assert!(matches!(
        store.set("uniserve_requests_total", &none, 1, 1.0),
        Err(Error::Instrument { .. })
    ));

    assert!(matches!(
        Store::new(
            Arc::new(System),
            Retention {
                coarse: Tier {
                    width: Duration::from_secs(60),
                    keep: Duration::from_secs(3600),
                },
                ..Retention::default()
            }
        ),
        Err(Error::Retention(_))
    ));
}

#[test]
fn accept_selects_the_response_format() {
    assert_eq!(Format::negotiate(None), Format::Json);
    assert_eq!(Format::negotiate(Some("*/*")), Format::Json);
    assert_eq!(
        Format::negotiate(Some("application/vnd.apache.arrow.stream")),
        Format::Ipc
    );
    assert_eq!(Format::negotiate(Some("text/event-stream")), Format::Sse);
    assert_eq!(
        Format::negotiate(Some("application/json;q=0.5, text/event-stream")),
        Format::Sse
    );
    assert_eq!(
        Format::negotiate(Some("text/html, application/json")),
        Format::Json
    );
}

/// The `histogram` column carries every bucket of a row's distribution.
#[tokio::test]
async fn histogram_column_holds_the_buckets() {
    let (store, clock) = store(Retention::default());
    at(&clock, 0.0);
    for value in [0.5, 0.5, 2.0] {
        store
            .observe("uniserve_ttft_seconds", &Labels::new(), 1, value)
            .unwrap();
    }

    let rows = store
        .query("SELECT value, histogram FROM stats")
        .await
        .unwrap();
    let batch = &rows.batches[0];
    assert!((batch.column(0).as_primitive::<Float64Type>().value(0) - 1.0).abs() < 1e-12);

    let histogram = batch.column(1).as_struct();
    assert_eq!(
        histogram
            .column_by_name("count")
            .unwrap()
            .as_primitive::<UInt64Type>()
            .value(0),
        3
    );
    let buckets = histogram
        .column_by_name("buckets")
        .unwrap()
        .as_list::<i32>()
        .value(0);
    let buckets = buckets.as_struct();
    let counts = buckets
        .column_by_name("count")
        .unwrap()
        .as_primitive::<UInt64Type>();
    assert_eq!(counts.values().to_vec(), [2, 1]);
    let uppers = buckets
        .column_by_name("upper")
        .unwrap()
        .as_primitive::<Float64Type>();
    assert!(uppers.value(0) >= 0.5 && uppers.value(1) >= 2.0 && uppers.len() == 2);
}
