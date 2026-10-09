//! Wire forms of the head-to-launcher protocol.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use uniserve_ipc::launcher::{Instruction, RankExit, Report, Reservation};

/// A launcher's two reports share one connection and carry no tag: the head
/// tells them apart by their fields.
#[test]
fn reports_decode_by_their_fields() {
    let reservation: Report = serde_json::from_str(r#"{"worker_id":"model","port":4100}"#).unwrap();
    assert_eq!(
        reservation,
        Report::Reservation(Reservation {
            worker_id: "model".into(),
            port: 4100,
        })
    );

    let exit: Report = serde_json::from_str(
        r#"{"worker_id":"model","generation":2,"rank":5,"status":"exit status: 1"}"#,
    )
    .unwrap();
    assert_eq!(
        exit,
        Report::Exit(RankExit {
            worker_id: "model".into(),
            generation: 2,
            rank: 5,
            status: "exit status: 1".into(),
        })
    );

    assert!(serde_json::from_str::<Report>(r#"{"worker_id":"model"}"#).is_err());
}

/// Instructions are objects keyed by their snake-case name; one without
/// fields is that name alone.
#[test]
fn instructions_are_keyed_by_name() {
    let stop = Instruction::Stop {
        worker_id: "model".into(),
    };
    assert_eq!(
        serde_json::to_value(&stop).unwrap(),
        serde_json::json!({"stop": {"worker_id": "model"}})
    );
    assert_eq!(
        serde_json::to_value(Instruction::Terminate).unwrap(),
        serde_json::json!("terminate")
    );
}
