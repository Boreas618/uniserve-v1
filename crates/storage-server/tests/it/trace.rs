//! The buffer pool's trace (`specs/buffer-pool.md` §18): a server whose
//! pools are watched records every input its buffer pool acts on and every
//! completion it produces, and a new pool driven by the recorded inputs
//! produces the same completions.

use ::storage::runtime::{Completion, RequestId};
use storage_server::buffer_pool::replay::replay;
use storage_server::buffer_pool::trace::files;

use crate::server::Running;
use crate::sessions::serve_a_session;

/// The integration test of the buffer pool's sessions, recorded, replays with every completion
/// equal, and the replay catches a completion altered in the run's records.
#[test]
fn a_recorded_run_replays_with_every_completion_equal_and_an_altered_one_is_caught() {
    let directory = tempfile::tempdir().unwrap();
    let trace = directory.path().to_owned();
    {
        let recorded = trace.clone();
        let server = Running::configured(move |server| server.watch_pools(Some(recorded)));
        serve_a_session(&server);
    }
    let (inputs, completions) = files(&trace, "model");

    // Every request the scenario submitted on its two schedulers' queue pairs is an input of the
    // trace: seven asked or submitted and one written to the queue pair by hand.
    let replayed = replay(&inputs, None).unwrap();
    assert_eq!(replayed.requests, 8, "{replayed:?}");

    // The first scheduler's first answer, its owner's slot row, altered.
    let record = |slot| {
        let mut record = Vec::new();
        Completion::Attached { slot }.encode(RequestId(1), &mut record);
        record
    };
    let (ran, altered) = (record(1), record(2));
    let mut bytes = std::fs::read(&completions).unwrap();
    let at = bytes
        .windows(ran.len())
        .position(|window| window == ran)
        .unwrap();
    bytes[at..at + ran.len()].copy_from_slice(&altered);
    std::fs::write(&completions, bytes).unwrap();

    let difference = replay(&inputs, None).unwrap_err();
    assert!(
        difference.contains("the run completed Attached { slot: 2 }")
            && difference.contains("the replay Attached { slot: 1 }"),
        "{difference}"
    );
}
