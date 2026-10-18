//! The launcher's registration, instructions and reports cross a channel as
//! frames.

#![cfg(test)]

use std::collections::HashMap;

use uniserve_ipc::launcher::{Instruction, RankExit, RankSpawn, Report, Reservation};
use uniserve_ipc::{Frame, Registration, VERSION};

/// Returns `message` as a receiver decodes it from the frame a sender made.
fn round_trip<M: serde::Serialize + serde::de::DeserializeOwned>(message: &M) -> M {
    Frame::encode(message).unwrap().decode().unwrap()
}

#[test]
fn every_instruction_round_trips_through_a_frame() {
    let spawn = Instruction::Spawn(Box::new(RankSpawn {
        rank: 5,
        worker_id: "model".into(),
        generation: 2,
        python: "python3".into(),
        head: "10.0.0.1:4100".into(),
        environment: HashMap::from([("RANK".to_owned(), "5".to_owned())]),
    }));
    let reserve = Instruction::Reserve {
        worker_id: "model".into(),
    };
    let stop = Instruction::Stop {
        worker_id: "model".into(),
    };
    for instruction in [spawn, reserve, stop, Instruction::Terminate] {
        assert_eq!(round_trip(&instruction), instruction);
    }
}

/// A launcher's two reports share one connection; the head must tell a
/// reservation from an exit whatever their fields hold.
#[test]
fn every_report_round_trips_through_a_frame() {
    let reservation = Report::Reservation(Reservation {
        port: 4100,
        listen_fd: 7,
    });
    let exit = Report::Exit(RankExit {
        worker_id: "model".into(),
        generation: 2,
        rank: 5,
        status: "exit status: 1".into(),
    });
    for report in [reservation, exit] {
        assert_eq!(round_trip(&report), report);
    }
}

#[test]
fn a_rank_and_a_launcher_register_with_the_same_first_frame() {
    let rank = Registration::Rank {
        worker_id: "model".into(),
        rank: 5,
        generation: 2,
        version: VERSION.into(),
    };
    let launcher = Registration::Launcher {
        host: "rank-1".into(),
        version: VERSION.into(),
    };
    for registration in [rank, launcher] {
        assert_eq!(round_trip(&registration), registration);
        assert_eq!(registration.version(), VERSION);
    }
}
