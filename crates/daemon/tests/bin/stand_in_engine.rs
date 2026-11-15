//! A stand-in for an engine process, which the daemon's supervision tests have a node start.
//!
//! A node starts it exactly as it starts an engine: it is the executable of the engine package a
//! launch's module set names, resolved at the node's Build Server (the tests' stand-in Build
//! Server answers every engine package with this binary), and its command line is an engine's:
//! `--launch FD` (`proto::LAUNCH_OPTION`, the inherited sealed launch), the daemon's log level
//! options and `--lifeline FD` (`proto::LIFELINE_OPTION`). What it does is what the engine
//! package its launch names (`modules.engine.backend`) stands for:
//!
//! - `fails`: exits with status 1, as an engine that fails at its start.
//! - `succeeds`: exits with status 0.
//! - `follows-lifeline`: runs until its lifeline ends, which the kernel delivers once every write
//!   end of the pipe closed (its daemon's end), or until it is killed.
//! - `serves-store`: accepts one peer on the store socket it inherited at the descriptor its
//!   launch's world states (`world.inherited.store`), as a group's first rank serves the group's
//!   collective store, answers it `served <engine>` with the engine number its launch states, and
//!   exits with status 0.

use std::io::{Read as _, Write as _};
use std::os::fd::{FromRawFd as _, OwnedFd, RawFd};

/// The value of option `name` on this process's command line, as a descriptor number.
fn descriptor(arguments: &[String], name: &str) -> RawFd {
    let position = arguments
        .iter()
        .position(|argument| argument == name)
        .unwrap_or_else(|| panic!("the command line names no {name}"));
    arguments[position + 1]
        .parse()
        .unwrap_or_else(|error| panic!("{name} names no descriptor: {error}"))
}

/// Takes the inherited descriptor `descriptor`.
fn inherited(descriptor: RawFd) -> OwnedFd {
    // SAFETY: the node that started this process passed the descriptor for this option, and
    // nothing else in the process owns it.
    unsafe { OwnedFd::from_raw_fd(descriptor) }
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let mut text = String::new();
    std::fs::File::from(inherited(descriptor(&arguments, proto::LAUNCH_OPTION)))
        .read_to_string(&mut text)
        .expect("the launch reads");
    let launch: proto::config::EngineArgs =
        serde_json::from_str(&text).expect("the launch is an engine's");
    let lifeline = inherited(descriptor(&arguments, proto::LIFELINE_OPTION));

    match launch.modules.engine.backend.as_str() {
        "fails" => std::process::exit(1),
        "succeeds" => {}
        "follows-lifeline" => {
            // The read returns at the pipe's end: no byte is ever written to it.
            let _ = std::fs::File::from(lifeline).read(&mut [0]);
        }
        "serves-store" => {
            let inherited = launch
                .world
                .and_then(|world| world.inherited)
                .expect("the launch states the sockets the process inherited");
            let store = std::net::TcpListener::from(self::inherited(inherited.store));
            let (mut peer, _) = store.accept().expect("a peer of the store");
            peer.write_all(format!("served {}\n", launch.engine).as_bytes())
                .expect("the answer is written");
        }
        other => panic!("no stand-in engine package {other}"),
    }
}
