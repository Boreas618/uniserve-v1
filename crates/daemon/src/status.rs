//! A deployment's status as rows of its observation store: what `/admin/query` reads of what the
//! deployment executes, owns and did.
//!
//! Transitions are `log` events, each labelled by its `source` and carrying its human line as the
//! `message` field:
//!
//! - `node`: the node's deployment comes up, serves, comes down, or failed to (fields `state`,
//!   `deployment`, `error`);
//! - `engine`: a replica's engine changed its serving state (label `replica`; fields `from`, `to`,
//!   `reason`);
//! - `update`: an update entered a phase (label `update`; the fields of its report,
//!   [`proto::admin::UpdateReport`]);
//! - `profile`: a replica's profiling session entered a state (labels `replica`, `session`; the
//!   fields of the session).
//!
//! What stands at a moment is a gauge the leader samples every [`PERIOD`] ([`watch`]): the
//! requests the daemon holds and each replica's in flight, each replica's state, epoch, serving
//! generation and standby, each epoch's state, and each node's GPU quotas, reservations and host
//! storage as its Storage Server reports them. A gauge's latest bucket holds what stands now; a
//! series that ends (a reservation released, an epoch retired) is absent from it.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use proto::admin::{NodeUsage, ServingState, UpdateReport};
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::leader::Leader;
use crate::observation::record::Recorder;
use crate::observation::{Fields, Kind, Labels};

/// How often the leader samples its gauges: the store's fine bucket width, so every bucket holds
/// a reading.
pub(crate) const PERIOD: Duration = Duration::from_secs(1);

/// The label of a `log` event's writer.
const SOURCE: &str = "source";

/// The fields `value` serializes to, a value that is not a map being no field: a field with no
/// value (`null`) is absent, so a query reads it as `NULL`.
fn fields_of(value: &impl Serialize) -> Fields {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::Object(mut fields)) => {
            fields.retain(|_, value| !value.is_null());
            fields
        }
        _ => Fields::new(),
    }
}

/// Appends a `log` event of `source` with `labels` and `fields` beside its `message`, under the
/// revision the deployment serves.
fn log(
    recorder: &Recorder,
    source: &str,
    mut labels: Labels,
    mut fields: Fields,
    message: impl Into<String>,
) {
    labels.insert(SOURCE.to_owned(), source.to_owned());
    fields.insert("message".to_owned(), message.into().into());
    recorder
        .store()
        .append(Kind::Log, labels, fields, recorder.revision());
}

/// Where a node's deployment is in its life, as its `node` events state it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeState {
    Starting,
    Serving,
    Stopping,
    Idle,
}

impl NodeState {
    fn name(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Serving => "serving",
            Self::Stopping => "stopping",
            Self::Idle => "idle",
        }
    }
}

/// Records that the node's deployment `deployment` entered `state`, and why it failed when
/// `error` states it.
pub(crate) fn node(recorder: &Recorder, state: NodeState, deployment: &str, error: Option<&str>) {
    let mut fields = Fields::new();
    fields.insert("state".to_owned(), state.name().into());
    fields.insert("deployment".to_owned(), deployment.into());
    let message = match (state, error) {
        (NodeState::Starting, _) => format!("{deployment} comes up"),
        (NodeState::Serving, _) => format!("serving {deployment}"),
        (NodeState::Stopping, _) => format!("{deployment} comes down"),
        (NodeState::Idle, None) => format!("{deployment} came down"),
        (NodeState::Idle, Some(error)) => {
            fields.insert("error".to_owned(), error.into());
            format!("{deployment} failed: {error}")
        }
    };
    log(recorder, "node", Labels::new(), fields, message);
}

/// Records that replica `replica`'s engine went from `from` to `to`, for `reason` when stated.
fn engine(
    recorder: &Recorder,
    replica: usize,
    from: ServingState,
    to: ServingState,
    reason: Option<&str>,
) {
    let labels = [("replica".to_owned(), replica.to_string())].into();
    let mut fields = Fields::new();
    fields.insert("from".to_owned(), from.name().into());
    fields.insert("to".to_owned(), to.name().into());
    let suffix = reason
        .map(|reason| format!(": {reason}"))
        .unwrap_or_default();
    if let Some(reason) = reason {
        fields.insert("reason".to_owned(), reason.into());
    }
    let message = format!("replica {replica}'s engine went from {from} to {to}{suffix}");
    log(recorder, "engine", labels, fields, message);
}

/// Records that an update entered the phase its `report` states.
fn update(recorder: &Recorder, report: &UpdateReport) {
    let labels = [("update".to_owned(), report.update.to_string())].into();
    let message = format!("update {}: {}", report.update, report.phase);
    log(recorder, "update", labels, fields_of(report), message);
}

/// Records that replica `replica`'s profiling session `session` entered the state it states.
pub(crate) fn profile(recorder: &Recorder, replica: usize, session: &crate::profiling::Session) {
    let labels = [
        ("replica".to_owned(), replica.to_string()),
        ("session".to_owned(), session.id.to_string()),
    ]
    .into();
    let message = match &session.end_reason {
        Some(reason) => format!("profiling session {} ended: {reason}", session.id),
        None => format!(
            "profiling session {} is {}",
            session.id,
            name_of(&session.state)
        ),
    };
    log(recorder, "profile", labels, fields_of(session), message);
}

/// Records the transitions of `leader` and samples its gauges until `stop` is cancelled: every
/// phase its updates enter, every change of a replica's serving state, and the gauges every
/// [`PERIOD`].
pub(crate) fn watch(leader: &Arc<Leader>, stop: &CancellationToken) {
    let recorder = leader.recorder().clone();
    if let Some(mut transitions) = leader.updates.transitions() {
        let recorder = recorder.clone();
        tokio::spawn(async move {
            while let Some(transition) = transitions.recv().await {
                update(&recorder, &transition.report);
            }
        });
    }

    // The changes are taken before the states they follow are read, so none is missed.
    let mut changes = leader.status_changes();
    let mut previous = states(leader);
    let weak = Arc::downgrade(leader);
    let stopped = stop.clone();
    let transitions = recorder.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                () = stopped.cancelled() => return,
                () = changes.changed() => {}
            }
            let Some(leader) = weak.upgrade() else {
                return;
            };
            let current = states(&leader);
            for (replica, (state, reason)) in current.iter().enumerate() {
                let before = previous.get(replica).map(|(state, _)| *state);
                if before != Some(*state) {
                    let from = before.unwrap_or(ServingState::Starting);
                    engine(&transitions, replica, from, *state, reason.as_deref());
                }
            }
            previous = current;
        }
    });

    // The gauges are sampled on a thread of their own: a node's storage is read over the engine
    // control, whose exchanges block, and the sampling's period is the host's clock, not the
    // runtime's.
    let weak = Arc::downgrade(leader);
    let stopped = stop.clone();
    let sampling = std::thread::Builder::new()
        .name("status".to_owned())
        .spawn(move || {
            while !stopped.is_cancelled() {
                let Some(leader) = weak.upgrade() else {
                    return;
                };
                sample(&leader);
                drop(leader);
                std::thread::sleep(PERIOD);
            }
        });
    if let Err(error) = sampling {
        warn!("the status gauges are not sampled: no thread started: {error}");
    }
}

/// Each replica's serving state, and why when it states a reason.
fn states(leader: &Leader) -> Vec<(ServingState, Option<String>)> {
    leader
        .replicas()
        .iter()
        .map(|replica| {
            let status = replica.status();
            (status.engine, status.reason)
        })
        .collect()
}

/// Writes one reading of every gauge of `leader`.
fn sample(leader: &Leader) {
    let recorder = leader.recorder();
    let revision = recorder.revision();
    let store = recorder.store();
    let set = |name: &str, labels: &Labels, value: f64| {
        if let Err(error) = store.set(name, labels, revision, value) {
            warn!(metric = name, "a status gauge was not recorded: {error}");
        }
    };

    set(
        "uniserve_requests_held",
        &Labels::new(),
        leader.held() as f64,
    );
    for (ordinal, replica) in leader.replicas().iter().enumerate() {
        let status = replica.status();
        let at: Labels = [("replica".to_owned(), ordinal.to_string())].into();
        set("uniserve_requests_in_flight", &at, status.in_flight as f64);
        set("uniserve_replica_epoch", &at, status.epoch as f64);
        let mut state = at.clone();
        state.insert("state".to_owned(), status.engine.name().to_owned());
        set("uniserve_replica_state", &state, 1.0);
        if let Some(generation) = status.generation {
            set("uniserve_replica_generation", &at, generation as f64);
        }
        for (member, generation) in status.standby.iter().enumerate() {
            let mut labels = at.clone();
            labels.insert("member".to_owned(), member.to_string());
            set(
                "uniserve_replica_standby_generation",
                &labels,
                *generation as f64,
            );
        }
    }
    for epoch in leader.updates.epochs().statuses() {
        let labels: Labels = [
            ("epoch".to_owned(), epoch.epoch.to_string()),
            ("state".to_owned(), name_of(&epoch.status)),
        ]
        .into();
        set("uniserve_epoch_state", &labels, 1.0);
    }

    let Some(control) = leader
        .replicas()
        .iter()
        .find_map(|replica| replica.control.get().cloned())
    else {
        return;
    };
    match control.storage_reports() {
        Ok(nodes) => {
            for node in &nodes {
                storage(node, &set);
            }
        }
        Err(error) => warn!("the nodes' storage was not read: {error:#}"),
    }
}

/// A unit variant's name as it serializes, or its tag for a variant that carries data.
fn name_of(value: &impl Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => name,
        Ok(serde_json::Value::Object(map)) => map.keys().next().cloned().unwrap_or_default(),
        _ => String::new(),
    }
}

/// The gauges of one node's storage: each GPU's total, the deployment's quota limit and every
/// reservation, and the host storage.
fn storage(node: &NodeUsage, set: &impl Fn(&str, &Labels, f64)) {
    let host: Labels = [("host".to_owned(), node.host.clone())].into();
    for quota in &node.quotas {
        let mut gpu = host.clone();
        gpu.insert(
            "gpu".to_owned(),
            proto::admin::GpuUuid(quota.gpu).to_string(),
        );
        gpu.insert(
            "deployment".to_owned(),
            quota.deployment.clone().unwrap_or_default(),
        );
        set("uniserve_gpu_total_bytes", &gpu, quota.total as f64);
        if let Some(limit) = quota.limit {
            set("uniserve_gpu_quota_bytes", &gpu, limit as f64);
        }
        // Reservations of one item, scope and epochs add into one series.
        let mut reserved: BTreeMap<Labels, u64> = BTreeMap::new();
        for reservation in &quota.reservations {
            let mut labels = gpu.clone();
            labels.insert("item".to_owned(), name_of(&reservation.item));
            labels.insert(
                "scope".to_owned(),
                reservation.scope.as_ref().map(name_of).unwrap_or_default(),
            );
            labels.insert(
                "epochs".to_owned(),
                reservation
                    .epochs
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            *reserved.entry(labels).or_default() += reservation.bytes;
        }
        for (labels, bytes) in &reserved {
            set("uniserve_gpu_reserved_bytes", labels, *bytes as f64);
        }
    }
    let host_storage = &node.host_storage;
    for (name, bytes) in [
        ("uniserve_host_available_bytes", host_storage.available),
        ("uniserve_host_reserve_bytes", host_storage.reserve),
        ("uniserve_host_unallocated_bytes", host_storage.unallocated),
    ] {
        set(name, &host, bytes as f64);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    /// The rows of `sql` over `recorder`'s store, as `/admin/query` answers them.
    async fn rows(recorder: &Recorder, sql: &str) -> Value {
        let rows = recorder.store().query(sql).await.unwrap().json().unwrap();
        serde_json::from_slice(&rows).unwrap()
    }

    /// A node's deployment's life is its `node` events, the error of one that failed among
    /// them, each with its line.
    #[tokio::test]
    async fn a_nodes_deployment_is_its_node_events() {
        let recorder = crate::test_support::recorder();
        node(&recorder, NodeState::Starting, "model", None);
        node(
            &recorder,
            NodeState::Idle,
            "model",
            Some("it did not come up: no device"),
        );
        assert_eq!(
            rows(
                &recorder,
                "SELECT fields['state'] AS state, fields['error'] AS error, \
                 fields['message'] AS message FROM events \
                 WHERE kind = 'log' AND labels['source'] = 'node' ORDER BY time",
            )
            .await,
            json!([
                {"state": "starting", "error": null, "message": "model comes up"},
                {
                    "state": "idle",
                    "error": "it did not come up: no device",
                    "message": "model failed: it did not come up: no device",
                },
            ])
        );
    }

    /// One sample states what the deployment's replica executes: its serving state, its epoch
    /// and the requests held and in flight. The test engine has no engine control, so no node's
    /// storage is read.
    #[tokio::test]
    async fn a_sample_states_each_replicas_state_and_the_requests_held() {
        let leader = Arc::new(crate::test_support::served_engine(
            crate::test_support::CHAT,
        ));
        sample(&leader);
        let gauges = rows(
            leader.recorder(),
            "SELECT name, labels, value FROM stats WHERE name IN \
             ('uniserve_requests_held', 'uniserve_requests_in_flight', 'uniserve_replica_state') \
             ORDER BY name",
        )
        .await;
        assert_eq!(
            gauges,
            json!([
                {"name": "uniserve_replica_state", "labels": {"replica": "0", "state": "serving"}, "value": 1.0},
                {"name": "uniserve_requests_held", "labels": {}, "value": 0.0},
                {"name": "uniserve_requests_in_flight", "labels": {"replica": "0"}, "value": 0.0},
            ])
        );
    }

    /// A node's storage report becomes one gauge per GPU total, quota and reservation series,
    /// reservations of one item, scope and epochs adding into one series, and one per host
    /// storage figure.
    #[test]
    fn a_storage_report_becomes_gauges() {
        use storage::protocol::{HostStorage, Item, Quota, Reservation};

        let reservation = |epochs: Vec<u64>, bytes| Reservation {
            item: Item::Weights,
            scope: None,
            epochs,
            segment: None,
            bytes,
        };
        let node = NodeUsage {
            host: "rank-0".to_owned(),
            quotas: vec![Quota {
                gpu: [0; 16],
                deployment: Some("model".to_owned()),
                total: 100,
                limit: Some(80),
                reservations: vec![
                    reservation(vec![1, 2], 10),
                    reservation(vec![1, 2], 5),
                    reservation(vec![2], 7),
                ],
                host_link_bytes_per_second: None,
                transfer_context: 0,
            }],
            host_storage: HostStorage {
                available: 1000,
                reserve: 100,
                unallocated: 50,
            },
            tasks: Vec::new(),
        };
        let written = std::sync::Mutex::new(Vec::new());
        storage(&node, &|name: &str, labels: &Labels, value: f64| {
            let epochs = labels.get("epochs").cloned().unwrap_or_default();
            service::lock(&written).push((name.to_owned(), epochs, value));
        });
        let written = written.into_inner().unwrap();
        let expected = [
            ("uniserve_gpu_total_bytes", "", 100.0),
            ("uniserve_gpu_quota_bytes", "", 80.0),
            ("uniserve_gpu_reserved_bytes", "1,2", 15.0),
            ("uniserve_gpu_reserved_bytes", "2", 7.0),
            ("uniserve_host_available_bytes", "", 1000.0),
            ("uniserve_host_reserve_bytes", "", 100.0),
            ("uniserve_host_unallocated_bytes", "", 50.0),
        ]
        .map(|(name, epochs, value)| (name.to_owned(), epochs.to_owned(), value));
        assert_eq!(written, expected);
    }
}
