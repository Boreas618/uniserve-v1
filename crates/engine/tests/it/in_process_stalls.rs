//! A job that never finishes is reported the way a rank's stalled builder
//! thread is: as a failure naming the forward it was running and the requests
//! whose calls it held.
//!
//! The launched path states this with a worker process whose stub forward
//! embeds a stalling token; the mechanism that names the forward is the
//! engine's own, so the in-process case states the same failure from the
//! simulator and asserts the same three facts.

use std::time::Duration;

use engine::{Executor, SimEngine, SimExecutor, WorkerFailure, WorkerGroupId};
use ipc::{CallId, ErrorCallIdentity, RequestId, RequestKey, Stall};

/// The key a request submitted by `id` carries before it is admitted.
fn request_key(id: u64) -> RequestKey {
    RequestKey {
        engine_id: 0,
        request_id: RequestId(id),
        request_epoch: 0,
    }
}

/// A stall is reported as a worker failure that names the forward the job
/// belongs to, the batch and the call, and it carries the request whose work
/// is gone.
#[test]
fn a_reported_stall_names_the_forward_and_the_requests_it_held() {
    let mut executor = SimExecutor::new(SimEngine::new());
    let reports = executor.loss();
    reports.stall(
        WorkerGroupId("sim".to_owned()),
        &Stall {
            batch_id: Some(1),
            route: Some("prefill".to_owned()),
            method: Some("decode".to_owned()),
            calls: vec![ErrorCallIdentity {
                request_key: request_key(31),
                call_id: CallId::new(1, 0),
            }],
            running_ms: 4_000,
        },
    );

    let error = executor
        .poll(Duration::from_secs(1))
        .expect_err("a stalled job fails its group");
    let failure = error
        .downcast_ref::<WorkerFailure>()
        .expect("a stall is reported as a worker failure");
    // The rank's report failed the group, not the group's own deadline: the
    // name of the stalled forward is what the deadline's failure lacks.
    for named in ["the builder thread stalled", "batch 1 (prefill", "31/"] {
        assert!(
            failure.message.contains(named),
            "the failure does not name {named:?}: {}",
            failure.message
        );
    }
    let requests = failure
        .requests
        .iter()
        .map(|request| request.request_id.0)
        .collect::<Vec<_>>();
    assert_eq!(requests, [31], "the failure names the requests it held");
    // The group the job belonged to fails with it: no other group serves.
    assert!(
        !executor.is_ready(&WorkerGroupId("sim".to_owned())),
        "the group whose job stalled stayed ready"
    );
}
