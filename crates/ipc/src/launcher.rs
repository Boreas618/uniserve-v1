//! The messages between the head's engine and a host's launcher.
//!
//! The engine starts no process on another host. Each host of a deployment
//! runs one `uniserve-launcher` launcher, which connects to the head and keeps
//! that connection for the deployment's lifetime. The connection carries
//! frames in both directions:
//!
//! - the launcher sends its [`Registration`](crate::Registration) once, then
//!   [`Report`]s: a [`Reservation`] answering each [`Instruction::Reserve`],
//!   the claim answering [`Instruction::Claim`], and a [`WorkerExit`] for
//!   each worker it still supervises that exits, which excludes workers an
//!   [`Instruction::Stop`] ended;
//! - the head sends [`Instruction`]s.
//!
//! Each launch of a worker group has a generation, which every [`WorkerSpawn`]
//! names: the launcher starts the rank with it, the rank states it in its
//! registration, and the launcher echoes it in that rank's [`WorkerExit`]. The
//! head ends a group's launch with a stop before it relaunches the group
//! under the next generation, so an exit the launcher reported just before it
//! read the stop is recognized as the stopped launch's rather than the
//! relaunch's.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One instruction the head sends a launcher.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Instruction {
    /// Start one rank, as a member of its group's current generation. The
    /// rank connects to the head and receives its launch descriptor there.
    Spawn(Box<WorkerSpawn>),
    /// Bind and hold a collective store socket for one worker group, which
    /// the launcher hands to that group's first rank when it spawns it. The
    /// launcher answers with a [`Reservation`].
    Reserve { worker_id: String },
    /// Stop every rank of one worker group the launcher owns, before the
    /// group is relaunched under its next generation. The stopped ranks'
    /// exits are not reported.
    Stop { worker_id: String },
    /// Claim the deployment at the host's Storage Server on the head's
    /// behalf, starting the server when the host has none, and hold the
    /// claim while the head's connection lasts. The launcher answers with
    /// [`Report::Claim`].
    Claim(StorageClaim),
    /// Stop every rank the launcher owns and end supervision.
    Terminate,
}

/// Everything a launcher needs to start one rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerSpawn {
    /// Rank within its worker group's process world, not host-relative.
    pub rank: u32,
    /// Worker group the rank belongs to.
    pub worker_id: String,
    /// The generation of the group's launch the rank belongs to, which the
    /// rank states when it registers and its exit report carries back.
    pub generation: u64,
    /// The `uniserve-worker` executable that runs the worker.
    pub executable: String,
    /// The head's address as this host reaches it, which the rank connects
    /// to and registers at.
    pub head: String,
    /// The variables the head's spawn command for this rank sets, applied
    /// over the environment the launcher inherited. A variable the head's
    /// command removes arrives with an empty value and is set empty rather
    /// than removed.
    pub environment: HashMap<String, String>,
}

/// What a launcher claims at its host's Storage Server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageClaim {
    /// The deployment: the served model name.
    pub deployment: String,
    /// The `(worker group, rank)` scopes the deployment places on the host;
    /// the server releases every other scope of the deployment.
    pub scopes: Vec<(String, u32)>,
    /// The `uniserve-storage-server` executable that starts the host's
    /// server when it has none.
    pub server: String,
}

/// One message a launcher sends after its registration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Report {
    Reservation(Reservation),
    Exit(WorkerExit),
    /// The host's Storage Server and the claim held there, or why the claim
    /// failed.
    Claim(Result<crate::StorageLaunch, String>),
}

/// A launcher's answer to the [`Instruction::Reserve`] the head sent last.
///
/// The head keeps at most one reservation outstanding per launcher and reads
/// its answer before it sends that launcher anything else, so the answer
/// names no group: it is the reserved group's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reservation {
    /// Port the launcher bound on its host for the group's collective store.
    pub port: u16,
    /// The descriptor number the launcher holds the bound socket at. The
    /// group's first rank inherits the socket at the same number, which the
    /// head names in that rank's launch descriptor.
    pub listen_fd: i32,
}

/// What a launcher says when one of its ranks exits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerExit {
    /// Worker group of the process that exited.
    pub worker_id: String,
    /// The group generation the rank's spawn named.
    pub generation: u64,
    /// Rank of the process that exited within its group.
    pub rank: u32,
    /// Exit status text, or the reason the status could not be read.
    pub status: String,
}
