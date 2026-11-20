//! A stand-in for an engine process, which the daemon's supervision tests have a node start.
//!
//! A node starts it exactly as it starts an engine: it is the executable of the engine package a
//! launch's module set names, resolved at the node's Build Server (the tests' stand-in Build
//! Server answers every engine package with this binary), and its command line is an engine's:
//! `--launch FD` (`proto::LAUNCH_OPTION`, the inherited sealed launch), the daemon's log level
//! options and `--lifeline FD` (`proto::LIFELINE_OPTION`). What it does is what the engine
//! package its launch names (`modules.engine.backend`) stands for:
//!
//! - `fails`: exits with status 1 and the error [`FAILURE`], as an engine that fails at its start;
//!   like the engine, it states the error it ends with as its exit reason
//!   (`service::exit_reason`), which the node reads into its exit report.
//! - `succeeds`: exits with status 0.
//! - `follows-lifeline`: runs until its lifeline ends, which the kernel delivers once every write
//!   end of the pipe closed (its daemon's end), or until it is killed.
//! - `serves-store`: accepts one peer on the store socket it inherited at the descriptor its
//!   launch's world states (`world.inherited.store`), as a group's first rank serves the group's
//!   collective store, answers it `served <engine>` with the engine number its launch states, and
//!   exits with status 0.

use std::io::{Read as _, Write as _};
use std::os::fd::{FromRawFd as _, OwnedFd, RawFd};

use anyhow::Context as _;

/// The error the `fails` package ends with, as an engine that cannot read its launch does: the
/// tests' `package::FAILURE`.
const FAILURE: &str = "reading the launch at descriptor 3: unknown field `boundary_interval`";

/// The value of option `name` on this process's command line, as a descriptor number.
fn descriptor(arguments: &[String], name: &str) -> anyhow::Result<RawFd> {
    let position = arguments
        .iter()
        .position(|argument| argument == name)
        .with_context(|| format!("the command line names no {name}"))?;
    arguments
        .get(position + 1)
        .with_context(|| format!("{name} names no descriptor"))?
        .parse()
        .with_context(|| format!("{name} names no descriptor"))
}

/// Takes the inherited descriptor `descriptor`.
fn inherited(descriptor: RawFd) -> OwnedFd {
    // SAFETY: the node that started this process passed the descriptor for this option, and
    // nothing else in the process owns it.
    unsafe { OwnedFd::from_raw_fd(descriptor) }
}

fn main() -> anyhow::Result<()> {
    service::exit_reason::take();
    let ended = act();
    if let Err(error) = &ended {
        service::exit_reason::state(&format!("{error:#}"));
    }
    ended
}

/// What the engine package the launch names stands for.
fn act() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let mut text = String::new();
    std::fs::File::from(inherited(descriptor(&arguments, proto::LAUNCH_OPTION)?))
        .read_to_string(&mut text)
        .context("reading the launch")?;
    let launch: proto::config::EngineArgs =
        serde_json::from_str(&text).context("reading the launch as an engine's")?;
    let lifeline = inherited(descriptor(&arguments, proto::LIFELINE_OPTION)?);

    match launch.modules.engine.backend.as_str() {
        "fails" => anyhow::bail!(FAILURE),
        "succeeds" => Ok(()),
        "follows-lifeline" => {
            // The read returns at the pipe's end: no byte is ever written to it.
            std::fs::File::from(lifeline)
                .read(&mut [0])
                .context("following the lifeline")?;
            Ok(())
        }
        "serves-store" => {
            let inherited = launch
                .world
                .and_then(|world| world.inherited)
                .context("the launch states no socket the process inherited")?;
            let store = std::net::TcpListener::from(self::inherited(inherited.store));
            let (mut peer, _) = store.accept().context("accepting the store's peer")?;
            peer.write_all(format!("served {}\n", launch.engine).as_bytes())
                .context("answering the store's peer")
        }
        other => anyhow::bail!("no stand-in engine package {other}"),
    }
}
