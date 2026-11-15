//! The node's event log (`GET /admin/events`): what changed, in order, read by long poll.

use serde::{Deserialize, Serialize};

use super::operation::Operation;
use super::status::ServingState;
use super::update::UpdateReport;

/// The query of `GET /admin/events`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EventsQuery {
    /// The last sequence number the caller read; 0 for every event kept.
    pub after: u64,
    /// Seconds to wait for an event when none is newer than `after`; the node waits at most
    /// `admin_config.event_log.longest_wait`.
    pub wait: f64,
}

/// What `GET /admin/events` answers: the events after the query's `after`, the sequence number
/// to read after next, and whether events after `after` already left the log, in which case
/// the reader reads the status again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventsPage {
    pub events: Vec<Event>,
    /// The `after` of the next read: the last event's sequence number, or the query's `after`
    /// when none is newer.
    pub next: u64,
    pub gap: bool,
}

/// One entry of the event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Its sequence number, from 1, increasing by one across the node's deployments.
    pub seq: u64,
    /// Seconds since the Unix epoch.
    pub time: f64,
    pub message: String,
    /// The event's category (`kind`) and what it carries beyond its message (`data`).
    #[serde(flatten)]
    pub body: EventBody,
}

/// An event's category (`kind`) and what it carries (`data`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum EventBody {
    /// An operation started or ended: the operation as it stands then.
    Operation(Operation),
    /// The serving engine's lifecycle state changed, and why when it states a reason.
    Engine {
        from: ServingState,
        to: ServingState,
        reason: Option<String>,
    },
    /// The node or its deployment changed: a deployment comes up, serves, holds or serves new
    /// requests, comes down, or failed to.
    Deployment,
    /// An update entered a phase: the update as it stood then (its `phase` among it).
    Update(Box<UpdateReport>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admin::{OperationKind, OperationState};

    /// An event's category and data read back as the daemon wrote them, the data of a category
    /// that carries none included.
    #[test]
    fn events_read_back_with_their_category_and_data() {
        let operation = Operation {
            id: 3,
            kind: OperationKind::Config,
            state: OperationState::Failed,
            started: 1.0,
            ended: Some(2.0),
            error: Some("it failed".to_owned()),
        };
        let events = [
            EventBody::Operation(operation),
            EventBody::Engine {
                from: ServingState::Serving,
                to: ServingState::Failed,
                reason: Some("a device fault".to_owned()),
            },
            EventBody::Deployment,
        ];
        for (seq, body) in (1..).zip(events) {
            let event = Event {
                seq,
                time: 4.0,
                message: "what happened".to_owned(),
                body,
            };
            let encoded = serde_json::to_value(&event).unwrap();
            assert_eq!(serde_json::from_value::<Event>(encoded).unwrap(), event);
        }
        let page: EventsPage = serde_json::from_str(
            r#"{"events": [{"seq": 1, "time": 1.0, "message": "m", "kind": "deployment"},
                {"seq": 2, "time": 2.0, "message": "m", "kind": "engine",
                 "data": {"from": "starting", "to": "serving", "reason": null}}],
               "next": 2, "gap": false}"#,
        )
        .unwrap();
        assert_eq!(page.events[0].body, EventBody::Deployment);
        assert!(matches!(
            page.events[1].body,
            EventBody::Engine {
                to: ServingState::Serving,
                ..
            }
        ));
    }
}
