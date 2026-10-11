//! Captured expert participation without tokens.
//!
//! A rank that is not part of an agreed expert step still executes every
//! expert layer of it with no tokens of its own (a join), so its peers'
//! tokens routed to its experts are computed and returned. [`JoinGraphs`]
//! holds one CUDA graph of that layer sequence per step capacity; a join
//! launches the graph of the agreed capacity instead of running the layers
//! eagerly. All ranks warm and capture the same capacities in the same
//! order, largest first, because each warmup is a collective expert step.

use std::collections::BTreeMap;

use tvm_ffi::Any;
use uniserve_cuda::Stream;

use crate::cuda_graph::CUDAGraph;
use crate::ffi::{Function, Handle};
use crate::{Error, Result};

/// The shared graph interface (`uniserve_models.interface.graphs`).
#[derive(Clone)]
pub struct GraphFunctions {
    /// `graphs.begin_capture(pool, stream, device) -> capture`.
    pub begin: Function,
    /// `graphs.end_capture(capture) -> [graph, executables]`.
    pub end: Function,
    /// `graphs.release(graph)`.
    pub release: Function,
}

impl GraphFunctions {
    pub fn from_interface(interface: &crate::model::interface::Interface) -> Result<Self> {
        Ok(Self {
            begin: interface.function("graphs.begin_capture")?,
            end: interface.function("graphs.end_capture")?,
            release: interface.function("graphs.release")?,
        })
    }

    /// Capture whatever `body` submits on `stream` (and on streams forked
    /// from it and joined back before it returns) into one graph in `pool`.
    /// A failed body still ends the capture, discarding the partial graph.
    pub fn capture(
        &self,
        pool: &Handle,
        stream: usize,
        device: i32,
        body: impl FnOnce() -> Result<()>,
    ) -> Result<CUDAGraph> {
        let capture = Handle::new(self.begin.call_values(&[
            pool.to_any(),
            Any::from(stream as i64),
            Any::from(i64::from(device)),
        ])?)?;
        let ran = body();
        let ended = self
            .end
            .call(&[capture.view()])
            .and_then(CUDAGraph::captured);
        match ran {
            Ok(()) => ended,
            Err(error) => {
                if let Ok(graph) = ended {
                    let _ = self.release.invoke(&[graph.handle().view()]);
                }
                Err(error)
            }
        }
    }
}

/// One captured join per step capacity.
pub struct JoinGraphs {
    graphs: BTreeMap<usize, CUDAGraph>,
    functions: GraphFunctions,
}

impl JoinGraphs {
    /// Capture `join(capacity)` for every capacity, largest first, in
    /// `pool` on `stream`. With `warm`, each capacity first joins eagerly:
    /// that is a collective step, so every rank of the union warms the same
    /// capacities at the same point of its startup. Capture itself issues
    /// no communication.
    pub fn capture(
        capacities: &[usize],
        warm: bool,
        pool: &Handle,
        stream: usize,
        device: i32,
        join: &dyn Fn(usize) -> Result<()>,
        functions: &GraphFunctions,
    ) -> Result<Self> {
        let mut graphs = Self {
            graphs: BTreeMap::new(),
            functions: functions.clone(),
        };
        let mut ordered = capacities.to_vec();
        ordered.sort_unstable_by(|left, right| right.cmp(left));
        ordered.dedup();
        for capacity in ordered {
            let captured = (|| {
                if warm {
                    join(capacity)?;
                }
                functions.capture(pool, stream, device, || join(capacity))
            })();
            match captured {
                Ok(graph) => {
                    graphs.graphs.insert(capacity, graph);
                }
                Err(error) => {
                    let _ = graphs.close(false);
                    return Err(error);
                }
            }
        }
        Ok(graphs)
    }

    /// Join a step of `capacity` tokens per rank on `stream`.
    pub fn launch(&self, capacity: usize, stream: &Stream) -> Result<()> {
        self.graph(capacity)?.launch(stream)
    }

    /// The captured join of `capacity`, which another thread can launch.
    pub fn graph(&self, capacity: usize) -> Result<&CUDAGraph> {
        self.graphs.get(&capacity).ok_or_else(|| {
            Error::Invalid(format!(
                "no captured expert join serves step capacity {capacity}"
            ))
        })
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Release the graphs after every stream they were launched on drained. An
    /// aborted close keeps them (and the storage they address) until exit.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = std::mem::take(&mut self.graphs);
        if aborted {
            std::mem::forget(graphs);
            return Ok(());
        }
        let mut failure = None;
        for graph in graphs.into_values().rev() {
            if let Err(error) = self.functions.release.invoke(&[graph.handle().view()]) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}
