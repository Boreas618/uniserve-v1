//! Bounded host work: the blocking pool the main process runs its non-device
//! work on (`specs/worker.md` §2.4).
//!
//! Module loading, transfer reads, shared-storage reclamation and the media
//! codecs run here, each on its own kind of work, so that no device stream and
//! no scheduler thread ever waits on host I/O.

mod pool;

pub use crate::completion::Outcome;
pub use pool::{HostAction, HostPool, HostTask};
