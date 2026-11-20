//! The start's asks for room, relayed through the control thread.
//!
//! The start runs on a thread of its own ([`super::Start`]) and waits where it asks for room in
//! its GPUs' quotas (`crate::memory::room`); the control thread owns the process interface, so an
//! ask reaches it as [`Input::RoomWanted`], which it sends to the daemon as `RoomWanted` and
//! answers when the daemon's `Room` names it ([`super::MainProcess::want_room`]).

use crossbeam_channel::Sender;

use super::Input;
use crate::memory::room::{DeviceRoom, RoomAsk};

/// The start's [`RoomAsk`] over the control thread's inbox.
pub(super) struct Relay {
    inbox: Sender<Input>,
}

impl Relay {
    /// The asks of a start whose control thread reads `inbox`.
    pub(super) fn new(inbox: Sender<Input>) -> Self {
        Self { inbox }
    }
}

impl RoomAsk for Relay {
    /// Hands the ask to the control thread and waits for the daemon's answer.
    fn ask(&self, devices: &[DeviceRoom]) -> Result<(), String> {
        if devices.is_empty() {
            return Ok(());
        }
        let (answer, answered) = crossbeam_channel::bounded(1);
        self.inbox
            .send(Input::RoomWanted(devices.to_vec(), answer))
            .map_err(|_| "the control thread ended before the ask for room".to_owned())?;
        answered.recv().map_err(|_| {
            "the control thread ended before the daemon answered the ask for room".to_owned()
        })?
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    use proto::process::{Room, RoomWanted, Upstream, report, upstream};
    use service::Connection;

    use super::super::MainProcess;

    /// A GPU's identity, as an ask names it.
    const GPU: [u8; 16] = [7; 16];

    /// The control thread of an engine whose process interface agreed on `version`, and the
    /// daemon's end of that interface.
    fn main_process(version: u32) -> (MainProcess, Connection, Connection) {
        let (engine_requests, daemon_requests) = UnixStream::pair().unwrap();
        let (engine_process, daemon_process) = UnixStream::pair().unwrap();
        let main = MainProcess::new(
            Connection::new(engine_requests).unwrap(),
            std::num::NonZeroU32::MIN,
            proto::Agreed(version),
            Connection::new(engine_process).unwrap(),
            testkit::serving::launch(),
        )
        .unwrap();
        (
            main,
            Connection::new(daemon_requests).unwrap(),
            Connection::new(daemon_process).unwrap(),
        )
    }

    /// The asks for room the daemon read on the process interface.
    fn asks(daemon: &mut Connection) -> Vec<RoomWanted> {
        let mut asks = Vec::new();
        while let Some(upstream) = daemon.recv::<Upstream>(Duration::from_millis(200)).unwrap() {
            if let Some(upstream::Kind::Report(proto::process::Report {
                kind: Some(report::Kind::RoomWanted(wanted)),
            })) = upstream.kind
            {
                asks.push(wanted);
            }
        }
        asks
    }

    /// A start's ask reaches the daemon as `RoomWanted`, numbered, and waits until the daemon's
    /// `Room` of its number answers it: made, or refused with the reason the start then fails
    /// with. An answer that names no ask is dropped.
    #[test]
    fn an_ask_waits_for_the_daemons_room_of_its_number() {
        let (mut main, _requests, mut daemon) = main_process(proto::ROOM);
        let (first, made) = crossbeam_channel::bounded(1);
        let (second, refused) = crossbeam_channel::bounded(1);
        main.want_room(vec![(GPU, 8 << 30)], first);
        main.want_room(vec![(GPU, 1 << 30)], second);

        let asks = asks(&mut daemon);
        assert_eq!(
            asks.iter()
                .map(|wanted| (
                    wanted.id,
                    wanted
                        .devices
                        .iter()
                        .map(|device| (device.gpu.clone(), device.bytes))
                        .collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>(),
            [
                (1, vec![(GPU.to_vec(), 8 << 30)]),
                (2, vec![(GPU.to_vec(), 1 << 30)])
            ]
        );
        assert!(made.try_recv().is_err(), "the start waits for the answer");

        main.room_made(Room {
            id: 9,
            refused: String::new(),
        });
        main.room_made(Room {
            id: 2,
            refused: "even the tail leaves too little room".to_owned(),
        });
        main.room_made(Room {
            id: 1,
            refused: String::new(),
        });
        assert_eq!(made.try_recv().unwrap(), Ok(()));
        assert_eq!(
            refused.try_recv().unwrap(),
            Err("even the tail leaves too little room".to_owned())
        );
    }

    /// A daemon of a version before `proto::ROOM` would leave an ask unanswered: the start is
    /// told at once that no room is made, and nothing is sent.
    #[test]
    fn a_daemon_of_an_earlier_version_is_asked_for_no_room() {
        let (mut main, _requests, mut daemon) = main_process(proto::ROOM - 1);
        let (answer, answered) = crossbeam_channel::bounded(1);
        main.want_room(vec![(GPU, 8 << 30)], answer);

        let refusal = answered.try_recv().unwrap().unwrap_err();
        assert!(refusal.contains("makes no room"), "{refusal}");
        assert!(asks(&mut daemon).is_empty());
    }
}
