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

use super::expert_union::ExpertUnion;
use super::microbatches::{MicrobatchCall, turn};
use crate::cuda_graph::CUDAGraph;
use crate::ffi::{DeferredCall, Function, Handle};
use crate::profiling::range;
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
        self.capture_with_peers(pool, &[], stream, device, body)
    }

    /// [`Self::capture`] of work whose modules also run on other devices
    /// (a generation tower): `peers` pairs each such device with the pool
    /// receiving the work's allocations there.
    pub fn capture_with_peers(
        &self,
        pool: &Handle,
        peers: &[(i32, Handle)],
        stream: usize,
        device: i32,
        body: impl FnOnce() -> Result<()>,
    ) -> Result<CUDAGraph> {
        let peers = crate::ffi::list(peers.iter().map(|(device, pool)| {
            Any::from(crate::ffi::list([
                Any::from(i64::from(*device)),
                pool.to_any(),
            ]))
        }));
        let capture = Handle::new(self.begin.call_values(&[
            pool.to_any(),
            Any::from(stream as i64),
            Any::from(i64::from(device)),
            Any::from(peers),
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

impl ExpertUnion {
    /// Whether every microbatch has its prepared join.
    pub(crate) fn has_joins(&self) -> bool {
        self.microbatches.iter().all(|peer| peer.prepared.is_some())
    }

    /// Whether captured joins have their graph pools.
    pub(crate) fn has_pools(&self) -> bool {
        !self.pools.is_empty()
    }

    /// Use `pools`, one per microbatch, for captured joins.
    pub(crate) fn set_pools(&mut self, pools: Vec<Handle>) {
        self.pools = pools;
    }

    /// Whether this rank's forwards may replay their captured graphs.
    ///
    /// A replayed forward submits its whole step in one launch, without
    /// passing the host turn, and its thread then waits in the driver for
    /// that stream to finish before it can launch more: the driver holds
    /// about a thousand unfinished launches per stream and blocks the next
    /// one. A rank that joins eagerly submits its microbatches in rotation
    /// behind kernels that wait for this rank's other microbatches, reaches
    /// the same limit and keeps its own turn. Neither rank then submits what
    /// the other waits for. A replayed join submits each microbatch whole
    /// and waits on no host turn, so forwards replay only once the union's
    /// joins are captured; until then they run eagerly, in the same rotation
    /// as every rank's joins. Without microbatch threads no turn is held.
    pub(crate) fn replays(&self) -> bool {
        self.lanes.is_none() || self.joins.is_some()
    }

    /// Join one step of `capacity` with every microbatch: replay the
    /// captured joins, or join eagerly before they exist.
    pub(crate) fn join(&self, capacity: usize) -> Result<()> {
        let _range = range(c"uniserve.expert.join", None);
        if let (Some(joins), Some(_)) = (&self.joins, &self.graphs) {
            return joins.launch(capacity, &self.microbatches[0].stream);
        }
        self.join_eager(capacity)
    }

    /// Join one step of `capacity` eagerly with every microbatch, in
    /// rotation.
    pub(crate) fn join_eager(&self, capacity: usize) -> Result<()> {
        match &self.lanes {
            Some(lanes) => {
                let calls = (0..self.microbatches.len())
                    .map(|peer| self.eager_join(peer, capacity).map(turn))
                    .collect::<Result<Vec<_>>>()?;
                lanes
                    .run(&self.microbatches[0].stream, &self.streams(), calls)
                    .map(drop)
            }
            None => self.eager_join(0, capacity)?(),
        }
    }

    /// The join of microbatch `peer` within a populated step: its captured
    /// graph, or its eager join before capture.
    pub(super) fn peer_join(&self, peer: usize, capacity: usize) -> Result<MicrobatchCall<()>> {
        let state = &self.microbatches[peer];
        if let Some(joins) = &state.graphs {
            let graph = joins.graph(capacity)?.clone();
            let stream = state.stream.clone();
            return Ok(Box::new(move || graph.launch(&stream)));
        }
        let join = self.eager_join(peer, capacity)?;
        Ok(if self.lanes.is_some() {
            turn(join)
        } else {
            join
        })
    }

    /// The eager join of microbatch `peer`: the library's `join` over its
    /// context owner.
    fn eager_join(&self, peer: usize, capacity: usize) -> Result<MicrobatchCall<()>> {
        let prepared = self.microbatches[peer]
            .prepared
            .as_ref()
            .ok_or(Error::State("an expert microbatch has no prepared join"))?;
        let call = DeferredCall::new(
            self.functions.join.clone(),
            vec![prepared.to_any(), Any::from(capacity as i64)],
        );
        Ok(Box::new(move || call.invoke()))
    }

    /// Capture the joins every rank replays outside its own steps: one
    /// graph per capacity of every microbatch's join on the first stream,
    /// warmed eagerly first (a collective step on every rank). Source ranks
    /// with microbatches also capture each microbatch's join alone in its
    /// own pool, which an empty microbatch of a populated step replays.
    pub(crate) fn capture_joins(&mut self) -> Result<()> {
        let Some(graphs) = self.graphs.clone() else {
            return Ok(());
        };
        let pools = self.pools.clone();
        if pools.len() != self.microbatches.len() {
            return Err(Error::Invariant(
                "expert joins need one graph pool per microbatch".into(),
            ));
        }
        let capacities = self.parallel().capacities().to_vec();
        let primary = JoinGraphs::capture(
            &capacities,
            true,
            &pools[0],
            self.microbatches[0].stream.raw(),
            self.device,
            &|capacity| self.join_eager(capacity),
            &graphs,
        )?;
        self.joins = Some(primary);
        if self.microbatches.len() > 1 && !self.dedicated {
            for (peer, pool) in pools.iter().enumerate() {
                let captured = JoinGraphs::capture(
                    &capacities,
                    false,
                    pool,
                    self.microbatches[peer].stream.raw(),
                    self.device,
                    &|capacity| self.eager_join(peer, capacity)?(),
                    &graphs,
                )?;
                self.microbatches[peer].graphs = Some(captured);
            }
        }
        let captured: usize = self.joins.iter().map(JoinGraphs::len).sum::<usize>()
            + self
                .microbatches
                .iter()
                .filter_map(|peer| peer.graphs.as_ref())
                .map(JoinGraphs::len)
                .sum::<usize>();
        tracing::info!(
            target: "uniserve_worker.warmup",
            "captured {captured} expert join graphs for capacities {capacities:?}"
        );
        Ok(())
    }
}
