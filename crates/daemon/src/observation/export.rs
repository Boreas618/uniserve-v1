//! The push of the observation store's two tables to an OpenTelemetry Collector
//! (`export_config`).
//!
//! The admin API is a Unix socket only its owner reaches, so an outside store does not pull; the
//! leader pushes instead, over OTLP/HTTP with Protobuf bodies: every `period` it sends the `stats`
//! fine buckets completed since the last push the collector accepted to `/v1/metrics`, and the
//! `events` appended since then to `/v1/logs`. A push the collector refuses or that does not
//! reach it is sent again at the next period with what came after it, for as long as the fine
//! tier and the event log keep it.
//!
//! The mapping keeps each row as it is:
//!
//! - a counter's bucket is a delta `Sum` data point over the bucket's interval, monotonic;
//! - a gauge's bucket is a `Gauge` data point at the bucket's end, the latest reading in it;
//! - a histogram's bucket is a delta `ExponentialHistogram` data point at scale 3, which is the
//!   store's own bucket scheme, so no observation moves between buckets;
//! - every event is a log record at the time the store recorded it, named by its kind, whose
//!   attributes are its labels and fields.
//!
//! A row's labels are its data point's attributes, beside `revision`, the configuration revision
//! it was measured under; the resource names the service (`uniserve`) and the deployment.

use std::time::Duration;

use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value};
use opentelemetry_proto::tonic::logs::v1::{LogRecord, ResourceLogs, ScopeLogs};
use opentelemetry_proto::tonic::metrics::v1::{
    AggregationTemporality, ExponentialHistogram, ExponentialHistogramDataPoint, Gauge, Metric,
    NumberDataPoint, ResourceMetrics, ScopeMetrics, Sum, exponential_histogram_data_point, metric,
    number_data_point,
};
use opentelemetry_proto::tonic::resource::v1::Resource;
use prost::Message as _;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use super::stats::{Instrument, Point, Value};
use super::{Event, Labels, Store};
use crate::config::ExportConfig;

/// The most events one push carries: a push that resumes far behind sends the rest in the next.
const EVENTS: usize = 65_536;

/// How long one request to the collector may take before the push counts as failed.
const TIMEOUT: Duration = Duration::from_secs(30);

/// Pushes `store`'s tables as `config` states until `shutdown` is cancelled; returns at once when
/// it names no endpoint. `deployment` names the resource the rows come from.
pub(crate) async fn run(
    store: Store,
    config: ExportConfig,
    deployment: String,
    shutdown: CancellationToken,
) {
    let Some(endpoint) = config.endpoint else {
        return;
    };
    let endpoint = endpoint.trim_end_matches('/').to_owned();
    let client = match reqwest::Client::builder().timeout(TIMEOUT).build() {
        Ok(client) => client,
        Err(error) => {
            warn!("the observation push cannot make its HTTP client: {error}");
            return;
        }
    };

    // The collector has been sent nothing yet: the first push sends everything the store keeps.
    let mut cursor = Cursor::default();
    let mut ticks = tokio::time::interval(config.period);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick fires at once: nothing completed yet.
    ticks.tick().await;
    loop {
        tokio::select! {
            _ = ticks.tick() => {}
            () = shutdown.cancelled() => return,
        }
        let batch = cursor.next(&store);
        if let Err(error) = push(&client, &endpoint, &deployment, &batch).await {
            warn!(%endpoint, "the observation push failed and is sent again: {error:#}");
            continue;
        }
        cursor.advance(&batch);
    }
}

/// Where the push stands: the start of the first fine bucket not yet accepted, and the last
/// event accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cursor {
    from: i64,
    after: u64,
}

/// What one push sends: the fine buckets of `[from, to)` and the events after the cursor's.
struct Batch {
    to: i64,
    points: Vec<Point>,
    events: Vec<Event>,
}

impl Cursor {
    /// What the next push sends: every fine bucket completed since `from`, and the events since
    /// `after`.
    fn next(&self, store: &Store) -> Batch {
        let now = super::nanos(store.now());
        let to = floor(now, store.fine_width());
        Batch {
            to,
            points: store.points(self.from, to),
            events: store.events_after(self.after, EVENTS),
        }
    }

    /// Moves past what `batch` sent, once the collector accepted it.
    fn advance(&mut self, batch: &Batch) {
        self.from = batch.to.max(self.from);
        if let Some(last) = batch.events.last() {
            self.after = last.id;
        }
    }
}

fn floor(time: i64, width: i64) -> i64 {
    time - time.rem_euclid(width)
}

/// Sends `batch` to the collector at `endpoint`: its metrics, then its events.
async fn push(
    client: &reqwest::Client,
    endpoint: &str,
    deployment: &str,
    batch: &Batch,
) -> anyhow::Result<()> {
    if !batch.points.is_empty() {
        let body = metrics(deployment, &batch.points, batch.to).encode_to_vec();
        send(client, &format!("{endpoint}/v1/metrics"), body).await?;
    }
    if !batch.events.is_empty() {
        let body = logs(deployment, &batch.events).encode_to_vec();
        send(client, &format!("{endpoint}/v1/logs"), body).await?;
    }
    Ok(())
}

async fn send(client: &reqwest::Client, url: &str, body: Vec<u8>) -> anyhow::Result<()> {
    let response = client
        .post(url)
        .header("content-type", "application/x-protobuf")
        .body(body)
        .send()
        .await?;
    let status = response.status();
    anyhow::ensure!(status.is_success(), "{url} answered {status}");
    Ok(())
}

/// The resource every row comes from.
fn resource(deployment: &str) -> Resource {
    Resource {
        attributes: vec![
            text("service.name", "uniserve"),
            text("service.version", env!("CARGO_PKG_VERSION")),
            text("uniserve.deployment", deployment),
        ],
        ..Resource::default()
    }
}

fn scope() -> InstrumentationScope {
    InstrumentationScope {
        name: "uniserve".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
        ..InstrumentationScope::default()
    }
}

fn text(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.to_owned())),
        }),
        ..KeyValue::default()
    }
}

/// A row's attributes: its labels and its revision.
fn attributes(labels: &Labels, revision: u64) -> Vec<KeyValue> {
    let mut attributes: Vec<KeyValue> =
        labels.iter().map(|(key, value)| text(key, value)).collect();
    attributes.push(KeyValue {
        key: "revision".to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::IntValue(
                i64::try_from(revision).unwrap_or(i64::MAX),
            )),
        }),
        ..KeyValue::default()
    });
    attributes
}

/// Nanoseconds since the Unix epoch as OTLP's unsigned timestamps state them.
fn unix(nanos: i64) -> u64 {
    u64::try_from(nanos).unwrap_or(0)
}

/// The metrics request of `points`, one metric per name, each bucket ending one fine width after
/// its start and at most at `to`.
fn metrics(deployment: &str, points: &[Point], to: i64) -> ExportMetricsServiceRequest {
    let mut names: std::collections::BTreeMap<&str, Vec<&Point>> = Default::default();
    for point in points {
        names.entry(point.name.as_str()).or_default().push(point);
    }
    // A bucket's width: the distance to the next start, the window's end for the last.
    let width = |point: &Point| {
        points
            .iter()
            .filter(|other| other.start > point.start)
            .map(|other| other.start - point.start)
            .min()
            .unwrap_or(to - point.start)
    };

    let metrics = names
        .into_iter()
        .map(|(name, points)| {
            let instrument = points[0].instrument;
            let data = match instrument {
                Instrument::Counter => metric::Data::Sum(Sum {
                    data_points: points
                        .iter()
                        .map(|point| number(point, width(point)))
                        .collect(),
                    aggregation_temporality: AggregationTemporality::Delta as i32,
                    is_monotonic: true,
                }),
                Instrument::Gauge => metric::Data::Gauge(Gauge {
                    data_points: points
                        .iter()
                        .map(|point| number(point, width(point)))
                        .collect(),
                }),
                Instrument::Histogram => metric::Data::ExponentialHistogram(ExponentialHistogram {
                    data_points: points
                        .iter()
                        .filter_map(|point| exponential(point, width(point)))
                        .collect(),
                    aggregation_temporality: AggregationTemporality::Delta as i32,
                }),
            };
            Metric {
                name: name.to_owned(),
                data: Some(data),
                ..Metric::default()
            }
        })
        .collect();

    ExportMetricsServiceRequest {
        resource_metrics: vec![ResourceMetrics {
            resource: Some(resource(deployment)),
            scope_metrics: vec![ScopeMetrics {
                scope: Some(scope()),
                metrics,
                ..ScopeMetrics::default()
            }],
            ..ResourceMetrics::default()
        }],
    }
}

fn number(point: &Point, width: i64) -> NumberDataPoint {
    NumberDataPoint {
        attributes: attributes(&point.labels, point.revision),
        start_time_unix_nano: unix(point.start),
        time_unix_nano: unix(point.start + width),
        value: Some(number_data_point::Value::AsDouble(point.value.number())),
        ..NumberDataPoint::default()
    }
}

fn exponential(point: &Point, width: i64) -> Option<ExponentialHistogramDataPoint> {
    let Value::Histogram(histogram) = &point.value else {
        return None;
    };
    let view = histogram.exponential();
    let buckets = |(offset, counts): (i32, Vec<u64>)| exponential_histogram_data_point::Buckets {
        offset,
        bucket_counts: counts,
    };
    Some(ExponentialHistogramDataPoint {
        attributes: attributes(&point.labels, point.revision),
        start_time_unix_nano: unix(point.start),
        time_unix_nano: unix(point.start + width),
        count: histogram.count(),
        sum: Some(histogram.sum()),
        scale: proto::histogram::EXPONENTIAL_SCALE,
        zero_count: view.zero_count,
        positive: Some(buckets(view.positive)),
        negative: Some(buckets(view.negative)),
        min: histogram.min(),
        max: histogram.max(),
        ..ExponentialHistogramDataPoint::default()
    })
}

/// The logs request of `events`.
fn logs(deployment: &str, events: &[Event]) -> ExportLogsServiceRequest {
    let records = events
        .iter()
        .map(|event| {
            let time = unix(super::nanos(event.time));
            let mut attributes = attributes(&event.labels, event.revision);
            attributes.extend(event.fields.iter().map(|(key, value)| {
                // A field is a string; another JSON value is its JSON text, as the table holds it.
                let value = match value {
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                text(key, &value)
            }));
            LogRecord {
                time_unix_nano: time,
                observed_time_unix_nano: time,
                event_name: event.kind.as_str().to_owned(),
                body: Some(AnyValue {
                    value: Some(any_value::Value::StringValue(
                        event.kind.as_str().to_owned(),
                    )),
                }),
                attributes,
                ..LogRecord::default()
            }
        })
        .collect();

    ExportLogsServiceRequest {
        resource_logs: vec![ResourceLogs {
            resource: Some(resource(deployment)),
            scope_logs: vec![ScopeLogs {
                scope: Some(scope()),
                log_records: records,
                ..ScopeLogs::default()
            }],
            ..ResourceLogs::default()
        }],
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime};

    use axum::Router;
    use axum::body::Bytes;
    use axum::routing::post;

    use super::*;
    use crate::observation::{Clock, Kind, Retention};

    /// A clock the test moves.
    struct Manual(Mutex<SystemTime>);

    impl Clock for Manual {
        fn now(&self) -> SystemTime {
            *self.0.lock().unwrap()
        }
    }

    /// One fine bucket's counter, gauge and histogram, and one event, become one push: a delta sum
    /// over the bucket, a gauge at its end, an exponential histogram at scale 3 with the same
    /// observations, and a log record named by the event's kind; nothing is sent twice.
    #[tokio::test]
    async fn a_push_sends_each_completed_bucket_and_event_once() {
        let start = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let clock = Arc::new(Manual(Mutex::new(start)));
        let store = Store::new(clock.clone(), Retention::default()).unwrap();
        let mut cursor = Cursor::default();

        let labels: Labels = [("replica".to_owned(), "0".to_owned())].into();
        store
            .add("uniserve_requests_total", &labels, 4, 3.0)
            .unwrap();
        store
            .set("uniserve_scheduler_running_requests", &labels, 4, 2.0)
            .unwrap();
        store
            .observe("uniserve_step_device_seconds", &labels, 4, 0.5)
            .unwrap();
        store.append(
            Kind::Request,
            labels.clone(),
            [("outcome".to_owned(), "stop".into())]
                .into_iter()
                .collect(),
            4,
        );

        // The bucket is still open: nothing completed, the event goes.
        let open = cursor.next(&store);
        assert!(open.points.is_empty());
        assert_eq!(open.events.len(), 1);
        cursor.advance(&open);

        *clock.0.lock().unwrap() = start + Duration::from_millis(1_500);
        let batch = cursor.next(&store);
        assert_eq!(batch.points.len(), 3);
        assert!(batch.events.is_empty());

        let request = metrics("sim", &batch.points, batch.to);
        let metrics = &request.resource_metrics[0].scope_metrics[0].metrics;
        let by_name = |name: &str| {
            metrics
                .iter()
                .find(|metric| metric.name == name)
                .and_then(|metric| metric.data.clone())
                .unwrap()
        };
        let bucket = unix(super::super::nanos(start));
        let second = 1_000_000_000;
        match by_name("uniserve_requests_total") {
            metric::Data::Sum(sum) => {
                assert!(sum.is_monotonic);
                assert_eq!(
                    sum.aggregation_temporality,
                    AggregationTemporality::Delta as i32
                );
                let point = &sum.data_points[0];
                assert_eq!(point.value, Some(number_data_point::Value::AsDouble(3.0)));
                assert_eq!(
                    (point.start_time_unix_nano, point.time_unix_nano),
                    (bucket, bucket + second)
                );
                assert!(point.attributes.contains(&text("replica", "0")));
                assert!(
                    point
                        .attributes
                        .iter()
                        .any(|attribute| attribute.key == "revision")
                );
            }
            other => panic!("a counter is a sum, not {other:?}"),
        }
        assert!(matches!(
            by_name("uniserve_scheduler_running_requests"),
            metric::Data::Gauge(_)
        ));
        match by_name("uniserve_step_device_seconds") {
            metric::Data::ExponentialHistogram(histogram) => {
                let point = &histogram.data_points[0];
                assert_eq!((point.count, point.sum, point.scale), (1, Some(0.5), 3));
                // 0.5 = 2^(-8/8) lies in (2^(-9/8), 2^(-8/8)], index -9.
                let positive = point.positive.as_ref().unwrap();
                assert_eq!(
                    (positive.offset, positive.bucket_counts.clone()),
                    (-9, vec![1])
                );
            }
            other => panic!("a histogram is an exponential histogram, not {other:?}"),
        }

        let logs = logs("sim", &open.events);
        let record = &logs.resource_logs[0].scope_logs[0].log_records[0];
        assert_eq!(record.event_name, "request");
        assert!(record.attributes.contains(&text("outcome", "stop")));

        cursor.advance(&batch);
        assert!(cursor.next(&store).points.is_empty());
    }

    /// The exporter posts both requests to the collector as Protobuf, and posts again what a
    /// refused push did not deliver.
    #[tokio::test]
    async fn the_collector_receives_protobuf_and_a_refused_push_is_sent_again() {
        let received: Arc<Mutex<Vec<(String, Bytes)>>> = Arc::default();
        let refuse = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let app = {
            let (received, refuse) = (Arc::clone(&received), Arc::clone(&refuse));
            let handler = move |path: &'static str| {
                let (received, refuse) = (Arc::clone(&received), Arc::clone(&refuse));
                move |body: Bytes| async move {
                    if refuse.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        return axum::http::StatusCode::SERVICE_UNAVAILABLE;
                    }
                    received.lock().unwrap().push((path.to_owned(), body));
                    axum::http::StatusCode::OK
                }
            };
            Router::new()
                .route("/v1/metrics", post(handler("metrics")))
                .route("/v1/logs", post(handler("logs")))
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });

        let store = Store::new(Arc::new(crate::observation::System), Retention::default()).unwrap();
        let shutdown = CancellationToken::new();
        let task = tokio::spawn(run(
            store.clone(),
            ExportConfig {
                endpoint: Some(format!("http://{address}/")),
                period: Duration::from_millis(100),
            },
            "sim".to_owned(),
            shutdown.clone(),
        ));
        store.append(Kind::Log, Labels::new(), Default::default(), 1);

        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        while received.lock().unwrap().is_empty() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the collector received nothing"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        shutdown.cancel();
        task.await.unwrap();

        let (path, body) = received.lock().unwrap()[0].clone();
        assert_eq!(path, "logs");
        let request = ExportLogsServiceRequest::decode(body).unwrap();
        assert_eq!(
            request.resource_logs[0].scope_logs[0].log_records[0].event_name,
            "log"
        );
    }
}
