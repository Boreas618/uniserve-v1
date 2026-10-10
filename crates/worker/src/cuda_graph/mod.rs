//! CUDA graphs: captured forwards, their launch and their policy.
//!
//! Rust decides which shapes are captured, in which order, into which
//! allocator pool, and which captured graph a forward runs. Python records
//! a graph (`uniserve_models.interface.graphs`): a capture function runs one
//! bound step under `torch.cuda.CUDAGraph` in the pool Rust passes and
//! returns the graph with the raw executable of each of its segments. Rust
//! launches those executables itself on the forward's stream
//! ([`CUDAGraph::launch`]) and keeps the Python graph only to release it.
//!
//! The pieces are independent of any runner:
//!
//! - [`CUDAGraph`]: one captured forward;
//! - [`bucket`]: the smallest configured size that holds a live count;
//! - [`capture_order`]: shapes largest footprint first, so later captures
//!   reuse the pool blocks and workspaces of earlier ones;
//! - [`reserve_pools`]: one allocator pool per device for an owner, or the
//!   pools of an owner it shares a stream with, registered with
//!   [`GraphStorage`] for budget accounting;
//! - [`GraphTable`]: the graphs of one runner, keyed by the runner's own
//!   graph key and an optional expert capacity.
//!
//! A table is filled at startup and sealed with the rank's graph storage;
//! afterwards a configured shape without a resident graph is an error, never
//! a silent eager fallback, because every rank of a group must make the
//! same collective calls in the same order.

mod pool;
mod table;

use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_cuda::{GraphExec, Stream};

use crate::ffi::Handle;
use crate::{Error, Result};

pub use pool::{GraphPool, GraphPools, GraphStorage, graph_storage_budget_bytes, reserve_pools};
pub use table::{Capacity, GraphCounts, GraphTable, Lookup};

/// How one numerical invocation ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphMode {
    Eager,
    Capture,
    Replay,
}

impl GraphMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Capture => "graph_capture",
            Self::Replay => "graph_replay",
        }
    }
}

/// One captured forward.
///
/// A graph is an ordered list of segments, each one CUDA graph executable,
/// with a graph break between two segments: work a graph cannot record. The
/// Python graph owns the executables, the memory their nodes address and
/// the breaks it recorded.
#[derive(Clone)]
pub struct CUDAGraph {
    /// One executable per segment, in launch order.
    segments: Arc<[GraphExec]>,
    /// Graph breaks the Python graph performs between its segments.
    breaks: usize,
    /// The Python graph, passed back to release it and to a binding that
    /// replays the graph itself.
    graph: Handle,
}

impl CUDAGraph {
    /// Take the result of a capture function: the array
    /// `[graph, executables, breaks]` of the Python graph, the raw
    /// `CUgraphExec` handle of each segment in launch order, and the number
    /// of graph breaks the Python graph performs between them.
    pub fn captured(result: Any) -> Result<Self> {
        let malformed =
            || Error::Interface("a capture returns [graph, executables, breaks]".into());
        let fields = tvm_ffi::Array::<Any>::try_from(result).map_err(|_| malformed())?;
        if fields.len() != 3 {
            return Err(malformed());
        }
        let field = |index: usize| fields.get(index).map_err(|_| malformed());
        let executables = crate::ffi::integers(field(1)?).map_err(|_| malformed())?;
        let breaks = i64::try_from(field(2)?).map_err(|_| malformed())?;
        if executables.is_empty() || executables.contains(&0) || breaks < 0 {
            return Err(malformed());
        }

        let segments = executables
            .into_iter()
            // SAFETY: the Python graph in `graph` owns the executables and
            // the memory they address. It is released only through this
            // value's handle, after the streams it was launched on drained.
            .map(|raw| unsafe { GraphExec::borrowed(raw as usize) })
            .collect();
        Ok(Self {
            segments,
            breaks: breaks as usize,
            graph: Handle::new(field(0)?)?,
        })
    }

    /// The Python graph.
    pub fn handle(&self) -> &Handle {
        &self.graph
    }

    /// Submit one execution of the forward to `stream` without waiting for
    /// it. The inputs the graph reads were written on `stream` before, and
    /// its outputs are read on `stream` after.
    ///
    /// A graph with breaks is not launched here: its breaks are Python
    /// calls, which its binding makes between the segments it replays.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        if self.breaks > 0 {
            return Err(Error::State(
                "a CUDA graph with Python graph breaks is replayed by its binding",
            ));
        }
        for segment in self.segments.iter() {
            segment.launch(stream)?;
        }
        Ok(())
    }
}

/// The smallest configured size holding `count`, or `None` when every size
/// is smaller. `sizes` need not be sorted.
pub fn bucket(sizes: &[usize], count: usize) -> Option<usize> {
    sizes.iter().copied().filter(|&size| size >= count).min()
}

/// Order shapes for capture: largest `footprint` first, ties in their given
/// order, duplicates removed. Pool blocks freed by a large capture are
/// reused by the smaller ones that follow.
pub fn capture_order<T, F>(shapes: impl IntoIterator<Item = T>, footprint: F) -> Vec<T>
where
    T: PartialEq,
    F: Fn(&T) -> (usize, usize),
{
    let mut ordered: Vec<T> = Vec::new();
    for shape in shapes {
        if !ordered.contains(&shape) {
            ordered.push(shape);
        }
    }
    // A stable sort keeps the configured order among equal footprints.
    ordered.sort_by_key(|shape| std::cmp::Reverse(footprint(shape)));
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_and_capture_order_follow_configured_sizes() {
        assert_eq!(bucket(&[8, 1, 32, 4], 3), Some(4));
        assert_eq!(bucket(&[8, 1, 32, 4], 32), Some(32));
        assert_eq!(bucket(&[8, 1, 32, 4], 33), None);

        // (rows, tokens): largest product first, then tokens; duplicates
        // removed; equal footprints keep their configured order.
        let order = capture_order([(2, 16), (4, 64), (8, 8), (2, 16), (4, 16)], |&(r, t)| {
            (r * t, t)
        });
        assert_eq!(order, vec![(4, 64), (4, 16), (8, 8), (2, 16)]);
    }

    #[test]
    fn a_capture_result_names_its_graph_and_one_executable_per_segment() -> Result<()> {
        let result = |executables: &[i64], breaks: i64| {
            Any::from(crate::ffi::list([
                Any::from(7_i64),
                Any::from(crate::ffi::integer_list(executables.iter().copied())),
                Any::from(breaks),
            ]))
        };

        let graph = CUDAGraph::captured(result(&[0x1000, 0x2000], 1))?;
        assert_eq!(i64::try_from(graph.handle().to_any()).ok(), Some(7));
        assert_eq!(
            graph
                .segments
                .iter()
                .map(GraphExec::raw)
                .collect::<Vec<_>>(),
            vec![0x1000, 0x2000]
        );

        // No segment, a null executable and a value of another shape are
        // not capture results.
        assert!(CUDAGraph::captured(result(&[], 0)).is_err());
        assert!(CUDAGraph::captured(result(&[0], 0)).is_err());
        assert!(CUDAGraph::captured(Any::from(7_i64)).is_err());
        Ok(())
    }
}
