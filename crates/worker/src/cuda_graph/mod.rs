//! CUDA graphs: captured forwards, their launch and their policy.
//!
//! Rust decides which shapes are captured, in which order, into which
//! allocator pool, and which captured graph a forward runs. Python records
//! a graph (`uniserve_models.interface.graphs`): a capture function runs one
//! bound step under `torch.cuda.CUDAGraph` in the pool Rust passes and
//! returns the graph with the raw executable of each of its segments. Rust
//! launches those executables itself on the forward's stream
//! ([`CUDAGraph::launch`]), performing the graph breaks it noted during the
//! capture between them, and keeps the Python graph only to release it.
//!
//! The pieces are independent of any runner:
//!
//! - [`CUDAGraph`]: one captured forward;
//! - [`GraphBreaks`]: the graph breaks noted while a forward is captured;
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
mod shape_key;
mod table;

use std::sync::{Arc, Mutex, PoisonError};

use tvm_ffi::Any;
use uniserve_cuda::{GraphExec, Stream};

use crate::ffi::Handle;
use crate::{Error, Result};

pub use pool::{GraphPool, GraphPools, GraphStorage, graph_storage_budget_bytes, reserve_pools};
pub(crate) use shape_key::{PrefillShape, ShapeKey, TextShapes, prefill_shapes};
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

/// Work a graph cannot record, performed between two of its segments at
/// every launch.
pub(crate) type GraphBreak = Arc<dyn Fn() -> Result<()> + Send + Sync>;

/// The graph breaks noted while a forward is captured.
///
/// A capture ends a segment and begins the next where its forward reaches
/// work a graph cannot record; the owner of that work notes it here, and
/// whoever captures the forward takes the notes for its graph
/// ([`CUDAGraph::with_breaks`]).
#[derive(Clone, Default)]
pub(crate) struct GraphBreaks(Arc<Mutex<Vec<GraphBreak>>>);

impl GraphBreaks {
    /// Note the break between the segment that just ended and the next.
    pub(crate) fn note(&self, work: GraphBreak) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(work);
    }

    /// The breaks noted since the last call, in capture order.
    pub(crate) fn take(&self) -> Vec<GraphBreak> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// One captured forward.
///
/// A graph is an ordered list of segments, each one CUDA graph executable,
/// with a graph break between two segments: work a graph cannot record. The
/// Python graph owns the executables and the memory their nodes address.
#[derive(Clone)]
pub struct CUDAGraph {
    /// One executable per segment, in launch order.
    segments: Arc<[GraphExec]>,
    /// The work between segments `i` and `i + 1`, for a graph whose breaks
    /// Rust noted at capture.
    breaks: Arc<[GraphBreak]>,
    /// The Python graph, passed back to release it and to a binding that
    /// replays the graph itself. Shared, so that a launch which holds the
    /// graph for a job gives up no Python reference when it ends.
    graph: Arc<Handle>,
}

impl CUDAGraph {
    /// Take the result of a capture function: the array
    /// `[graph, executables]` of the Python graph and the raw `CUgraphExec`
    /// handle of each of its segments in launch order.
    pub fn captured(result: Any) -> Result<Self> {
        let malformed = || Error::Interface("a capture returns [graph, executables]".into());
        let fields = tvm_ffi::Array::<Any>::try_from(result).map_err(|_| malformed())?;
        let field = |index: usize| fields.get(index).map_err(|_| malformed());
        let executables = crate::ffi::integers(field(1)?).map_err(|_| malformed())?;

        let segments = executables
            .into_iter()
            // SAFETY: the Python graph in `graph` owns the executables and
            // the memory they address. It is released only through this
            // value's handle, after the streams it was launched on drained.
            .map(|raw| unsafe { GraphExec::borrowed(raw as usize) })
            .collect();
        Ok(Self {
            segments,
            breaks: Arc::new([]),
            graph: Arc::new(Handle::new(field(0)?)?),
        })
    }

    /// The graph with the breaks noted during its capture, one between
    /// every two segments.
    pub(crate) fn with_breaks(mut self, breaks: Vec<GraphBreak>) -> Self {
        self.breaks = breaks.into();
        self
    }

    /// The graph's segments: one more than its graph breaks.
    pub(crate) fn segments(&self) -> usize {
        self.segments.len()
    }

    /// The Python graph.
    pub fn handle(&self) -> &Handle {
        &self.graph
    }

    /// Submit one execution of the forward to `stream` without waiting for
    /// it: every segment in order, and after each segment the break that
    /// follows it. The inputs the graph reads were written on `stream`
    /// before, and its outputs are read on `stream` after.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        for (index, segment) in self.segments.iter().enumerate() {
            segment.launch(stream)?;
            if let Some(work) = self.breaks.get(index) {
                work()?;
            }
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
        let result = |executables: &[i64]| {
            Any::from(crate::ffi::list([
                Any::from(7_i64),
                Any::from(crate::ffi::integer_list(executables.iter().copied())),
            ]))
        };

        let graph = CUDAGraph::captured(result(&[0x1000, 0x2000]))?;
        assert_eq!(i64::try_from(graph.handle().to_any()).ok(), Some(7));
        assert_eq!(
            graph
                .segments
                .iter()
                .map(GraphExec::raw)
                .collect::<Vec<_>>(),
            vec![0x1000, 0x2000]
        );

        // A value of another shape is not a capture result.
        assert!(CUDAGraph::captured(Any::from(7_i64)).is_err());
        Ok(())
    }
}
