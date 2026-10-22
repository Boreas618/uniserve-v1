//! CUDA graphs: captured forwards, their launch and their policy.
//!
//! Rust decides which shapes are captured, in which order, into which
//! allocator pool, and which captured graph a forward runs. Python records
//! a graph (`uniserve.cuda_graph`): a capture runs one bound step under
//! `torch.cuda.CUDAGraph` in the pool Rust passes and returns the captured
//! graph of each of its segments. Rust instantiates each segment into an
//! executable it owns ([`cuda::GraphExec`]), has Python release its graph
//! objects, launches the executables on the forward's stream
//! ([`CUDAGraph::launch`]), performing the graph breaks it noted during the
//! capture between them, and destroys them when the graph is released.
//! The memory the graphs address is Rust's as well ([`memory`]): the pools'
//! device memory, which keeps its addresses while any graph lives.
//!
//! The pieces are independent of any runner:
//!
//! - [`CUDAGraph`]: one captured forward;
//! - [`BreakOp`]: the native work a graph break performs;
//! - [`GraphBreaks`]: the graph breaks noted while a forward is captured;
//! - [`bucket`]: the smallest configured size that holds a live count;
//! - [`capture_order`]: shapes largest footprint first, so later captures
//!   reuse the pool blocks and workspaces of earlier ones;
//! - [`reserve_pools`]: one allocator pool per device for an owner, or the
//!   pools of an owner it shares a stream with, registered with
//!   [`GraphStorage`] for budget accounting, each backed by Rust's device
//!   memory ([`GraphPool`]);
//! - [`GraphTable`]: the graphs of one runner, keyed by the runner's own
//!   graph key and an optional expert capacity.
//!
//! A table is filled at startup and sealed with the rank's graph storage;
//! afterwards a configured shape without a resident graph is an error, never
//! a silent eager fallback, because every rank of a group must make the
//! same collective calls in the same order.

mod memory;
mod pool;
mod shape_key;
mod table;

use std::sync::{Arc, Mutex};

use cuda::{DeviceId, GraphExec, Stream};
use tvm_ffi::Any;

use crate::ffi::Handle;
use crate::{Error, Result, lock};

pub use pool::GraphPool;
pub(crate) use pool::{GraphStorage, graph_storage_budget_bytes, reserve_pools};
pub(crate) use shape_key::{
    PrefillShape, PrefillVariant, ShapeKey, TextForward, TextShapes, prefill_shapes, task_buckets,
};
pub(crate) use table::{GraphTable, Lookup};

/// How one numerical invocation ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GraphMode {
    Eager,
    Capture,
    Replay,
}

impl GraphMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Capture => "graph_capture",
            Self::Replay => "graph_replay",
        }
    }
}

/// Work a graph cannot record, performed natively between two of its
/// segments at every launch.
///
/// Every graph break is one of these operations: capture ends a segment
/// where the forward reaches the operation, notes it and begins the next
/// segment, and the execution thread that launches the graph performs it
/// after the segment before it, on the thread's own stream order. The
/// operation addresses only memory its owner keeps for the graph's life.
#[derive(Clone)]
pub(crate) enum BreakOp {
    /// Copy expert layer `layer`'s weight shards from the peers' memory
    /// into the layer's slot on the prefetch stream (weight data
    /// parallelism), ordered after the slot's previous reads.
    WeightCopy {
        weights: Arc<crate::parallel::ExpertWeights>,
        layer: usize,
    },
}

impl BreakOp {
    /// Perform the operation between two segments of a launch.
    pub(crate) fn perform(&self) -> Result<()> {
        match self {
            Self::WeightCopy { weights, layer } => weights.copy(*layer),
        }
    }
}

/// The graph breaks noted while a forward is captured.
///
/// A capture ends a segment and begins the next where its forward reaches
/// work a graph cannot record; the owner of that work notes it here, and
/// whoever captures the forward takes the notes for its graph
/// ([`CUDAGraph::with_breaks`]).
#[derive(Clone, Default)]
pub(crate) struct GraphBreaks(Arc<Mutex<Vec<BreakOp>>>);

impl GraphBreaks {
    /// Note the break between the segment that just ended and the next.
    pub(crate) fn note(&self, work: BreakOp) {
        lock(&self.0).push(work);
    }

    /// The breaks noted since the last call, in capture order.
    pub(crate) fn take(&self) -> Vec<BreakOp> {
        std::mem::take(&mut *lock(&self.0))
    }
}

/// One captured forward.
///
/// A graph is an ordered list of segments, each one CUDA graph executable
/// Rust owns, with a graph break between two segments: work a graph cannot
/// record. The executables address memory of the pool the graph was
/// captured in, and of the buffers its step reads and writes, which their
/// owners keep until the graph is released.
#[derive(Clone)]
pub(crate) struct CUDAGraph {
    /// One executable per segment, in launch order. The last clone of the
    /// graph to drop destroys them; it drops after the streams the graph was
    /// launched on drained.
    segments: Arc<[GraphExec]>,
    /// The work between segments `i` and `i + 1`, for a graph whose breaks
    /// Rust noted at capture.
    breaks: Arc<[BreakOp]>,
    /// The Python graph, which keeps the step's output alive and is passed
    /// back to release it. Shared, so that a launch which holds the graph
    /// for a job gives up no Python reference when it ends.
    graph: Arc<Handle>,
}

impl CUDAGraph {
    /// Take the result of a capture on `device`: the array `[graph,
    /// graphs]` of the Python graph and the raw `CUgraph` of each of its
    /// segments in launch order, which this instantiates. The caller then
    /// has Python release the captured graphs (`handover`).
    pub(crate) fn captured(result: Any, device: DeviceId) -> Result<Self> {
        let malformed = || Error::Interface("a capture returns [graph, graphs]".into());
        let fields = tvm_ffi::Array::<Any>::try_from(result).map_err(|_| malformed())?;
        let field = |index: usize| fields.get(index).map_err(|_| malformed());
        let captured = crate::ffi::integers(field(1)?).map_err(|_| malformed())?;
        let graph = Arc::new(Handle::new(field(0)?)?);

        let segments = captured
            .into_iter()
            // SAFETY: each raw graph is a live graph the capture ended on
            // `device`, which Python keeps until the handover that follows.
            .map(|raw| unsafe { GraphExec::instantiate(device, raw as usize) })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            segments: segments.into(),
            breaks: Arc::new([]),
            graph,
        })
    }

    /// The graph with the breaks noted during its capture, one between
    /// every two segments.
    pub(crate) fn with_breaks(mut self, breaks: Vec<BreakOp>) -> Self {
        self.breaks = breaks.into();
        self
    }

    /// The graph's segments: one more than its graph breaks.
    pub(crate) fn segments(&self) -> usize {
        self.segments.len()
    }

    /// The Python graph.
    pub(crate) fn handle(&self) -> &Handle {
        &self.graph
    }

    /// Submit one execution of the forward to `stream` without waiting for
    /// it: every segment in order, and after each segment the break that
    /// follows it. The inputs the graph reads were written on `stream`
    /// before, and its outputs are read on `stream` after.
    pub(crate) fn launch(&self, stream: &Stream) -> Result<()> {
        for (index, segment) in self.segments.iter().enumerate() {
            segment.launch(stream)?;
            if let Some(work) = self.breaks.get(index) {
                work.perform()?;
            }
        }
        Ok(())
    }
}

/// The smallest configured size holding `count`, or `None` when every size
/// is smaller. `sizes` need not be sorted.
pub(crate) fn bucket(sizes: &[usize], count: usize) -> Option<usize> {
    sizes.iter().copied().filter(|&size| size >= count).min()
}

/// Order shapes for capture: largest `footprint` first, ties in their given
/// order, duplicates removed. Pool blocks freed by a large capture are
/// reused by the smaller ones that follow.
pub(crate) fn capture_order<T, F>(shapes: impl IntoIterator<Item = T>, footprint: F) -> Vec<T>
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
    fn a_value_of_another_shape_is_no_capture_result() {
        assert!(CUDAGraph::captured(Any::from(7_i64), DeviceId(0)).is_err());
    }
}
