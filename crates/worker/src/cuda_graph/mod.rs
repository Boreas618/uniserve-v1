//! The device memory of the CUDA graphs the executor captures and replays.
//!
//! The executor (Python) captures every graph it serves (`uniserve.cuda_graph`)
//! into allocator pools the worker creates for it ([`reserve_pools`]): each
//! pool allocates from device memory Rust owns ([`memory`]), which keeps its
//! addresses for as long as any graph captured in it lives, and is registered
//! with [`GraphStorage`] for budget accounting ([`GraphPool`]). [`GraphMode`]
//! names how a forward ran in the statistics the worker reports.

mod memory;
mod pool;

pub(crate) use pool::{GraphPool, GraphStorage, graph_storage_budget_bytes, reserve_pools};

/// How one numerical invocation ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GraphMode {
    Eager,
    Capture,
    Replay,
}

impl GraphMode {
    /// The mode the executor reports by `code` (`uniserve.executor`'s
    /// `GraphMode`): 0 eager, 1 replay, 2 capture; `None` for any other
    /// code.
    pub(crate) const fn from_code(code: i64) -> Option<Self> {
        match code {
            0 => Some(Self::Eager),
            1 => Some(Self::Replay),
            2 => Some(Self::Capture),
            _ => None,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Capture => "graph_capture",
            Self::Replay => "graph_replay",
        }
    }
}
