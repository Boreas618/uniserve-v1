//! The daemon's process interface as one public type: the commands an epoch's
//! main process receives (§7.3) and the reports it sends back.
//!
//! The serving path of a world change — `Precopy`, `Switchover`, `Commit`,
//! `Revert`, `Purge` — and the lifecycle commands `Park`, `Serve`, `Drain`
//! arrive here, on the connection the daemon reserved for this main process,
//! and not on the request connection the front door submits through. One
//! connection serves the whole deployment, so one link reads it: a second
//! [`ProcessLink::connect`] in the same process is refused.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;

use crate::worker::processes::{self, Processes};

/// How long the reader waits for one message before it looks again at its stop
/// flag. Short, because a report needs the same connection.
const READ_INTERVAL: Duration = Duration::from_millis(10);

/// The daemon's process interface: the commands for this main process, and its
/// reports back.
///
/// The link reads the connection on a thread of its own for as long as it is
/// open, so commands reach [`Self::commands`] without a request being in
/// flight. Dropping it stops that thread; [`Self::close`] also closes the
/// connection, which is what tells the daemon this main process has gone.
pub struct ProcessLink {
    /// The deployment's connection, shared with whatever reports through it.
    processes: Processes,
    /// The commands of this main process, filed as they arrive.
    commands: crossbeam_channel::Receiver<proto::process::Command>,
    /// Set when the reader should stop.
    stop: Arc<AtomicBool>,
    /// The thread reading the interface.
    reader: Option<std::thread::JoinHandle<()>>,
}

impl ProcessLink {
    /// Connects to the leader daemon's process interface at `address` as the
    /// main process of `deployment`, and starts reading its commands.
    ///
    /// # Errors
    ///
    /// Fails when nothing listens there, when the daemon refuses the engine, or
    /// when this process already has a link.
    pub fn connect(address: &str, deployment: &str) -> anyhow::Result<Self> {
        let processes = processes::ProcessClient::connect(address, deployment)?;
        let commands = processes::take_commands().context(
            "this process already reads its daemon's process interface: one link per process",
        )?;
        let stop = Arc::new(AtomicBool::new(false));
        let reading = Arc::clone(&processes);
        let halted = Arc::clone(&stop);
        let reader = std::thread::Builder::new()
            .name("process-link".to_owned())
            .spawn(move || {
                while !halted.load(Ordering::Acquire) {
                    let outcome = processes::lock(&reading)
                        .and_then(|mut interface| interface.poll(READ_INTERVAL));
                    if let Err(error) = outcome {
                        if halted.load(Ordering::Acquire) {
                            return;
                        }
                        tracing::info!(%error, "the process interface stopped");
                        return;
                    }
                }
            })
            .context("starting the process interface's reader")?;
        Ok(Self {
            processes,
            commands,
            stop,
            reader: Some(reader),
        })
    }

    /// The commands this main process received, in the order the daemon sent
    /// them. `Park`, `Serve`, `Drain` and `Stop` end or start the process;
    /// `Precopy`, `Switchover`, `Commit`, `Revert` and `Purge` shape its policy
    /// generation (§7.3).
    pub fn commands(&self) -> crossbeam_channel::Receiver<proto::process::Command> {
        self.commands.clone()
    }

    /// Reports this main process's state to the daemon, which answers nothing
    /// (§7.3).
    ///
    /// # Errors
    ///
    /// Fails when the connection failed.
    pub fn report(&self, report: proto::process::Report) -> anyhow::Result<()> {
        processes::lock(&self.processes)?.report(report)
    }

    /// Stops reading and closes the connection, so the daemon sees this main
    /// process leave.
    pub fn close(mut self) {
        self.stop_reading();
        if let Ok(mut interface) = processes::lock(&self.processes) {
            interface.close();
        }
    }

    /// Stops the reader thread, once.
    fn stop_reading(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for ProcessLink {
    fn drop(&mut self) {
        // The connection outlives the link (the process may report through it
        // afterwards), but nothing should keep reading it for a link that is
        // gone.
        self.stop_reading();
    }
}
