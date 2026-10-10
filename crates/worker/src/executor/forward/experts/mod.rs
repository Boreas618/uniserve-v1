//! The expert-parallel runtime of a rank.
//!
//! Ranks of an expert union execute every expert step together: the ranks
//! whose forward the step runs send their tokens, every other rank joins
//! each expert layer with no tokens of its own. [`ExpertRuntime`] owns this
//! rank's side of that protocol:
//!
//! - the union's step agreement ([`ExpertParallel`]) and, during startup,
//!   the warmup announcements dedicated expert ranks follow;
//! - one token transport per microbatch (opaque numerical objects, created
//!   collectively through `experts.create_exchange`), each with its own
//!   stream;
//! - microbatch threads and their rotation ([`MicrobatchLanes`]);
//! - joins: per microbatch, the bound `join(capacity)` of its binding, and
//!   their captured graphs ([`JoinGraphs`]).
//!
//! A model rank's text runtime runs its forwards as expert steps through
//! [`ExpertRuntime::capacity`] and [`ExpertRuntime::run`]; between its own
//! steps, and on dedicated expert ranks always, the service loop joins the
//! union's steps ([`ExpertRuntime::idle`]).

mod start;
pub(crate) mod warmup;
mod weights;

pub(crate) use start::ExpertStart;
pub(crate) use weights::WeightRuntime;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tvm_ffi::Any;
use uniserve_cuda::Stream;

use super::super::backend::WorkerBackend;
use crate::error::WorkerError;
use crate::ffi::{DeferredCall, Function, Handle};
use crate::model::bind::ComponentBinding;
use crate::parallel::joins::{GraphFunctions, JoinGraphs};
use crate::parallel::{ExpertParallel, MicrobatchCall, MicrobatchLanes};
use crate::profiling::range;
use crate::{Error, Result};

/// The numerical interface entries of the experts area.
#[derive(Clone)]
pub(crate) struct ExpertFunctions {
    pub(crate) layers: Function,
    pub(crate) symmetric_bytes: Function,
    pub(crate) create: Function,
    pub(crate) bind_microbatches: Function,
    pub(crate) bind: Function,
    pub(crate) warm_thread: Function,
    pub(crate) close: Function,
}

impl ExpertFunctions {
    pub(crate) fn from_interface(interface: &crate::model::interface::Interface) -> Result<Self> {
        Ok(Self {
            layers: interface.function("experts.layers")?,
            symmetric_bytes: interface.function("experts.symmetric_bytes")?,
            create: interface.function("experts.create_exchange")?,
            bind_microbatches: interface.function("experts.bind_microbatches")?,
            bind: interface.function("experts.bind")?,
            warm_thread: interface.function("experts.warm_thread")?,
            close: interface.function("experts.close_exchange")?,
        })
    }
}

/// One microbatch: its transport, stream and joins.
struct Peer {
    /// The `experts.create_exchange` transport.
    transport: Handle,
    /// The microbatch's stream: the execution lane's stream for the first
    /// microbatch, a fork of it for the others. The rank's lanes close the
    /// streams after this runtime.
    stream: Stream,
    /// The bound `join(capacity)` of this microbatch's binding.
    join: Option<Function>,
    /// This microbatch's captured joins, so an empty microbatch of a
    /// populated step replays them on its own stream.
    graphs: Option<JoinGraphs>,
    /// Peer-mapped storage of split MegaMoE, retained until the transport
    /// closed.
    symmetric: Option<crate::transfer::PeerStorage>,
}

/// The rank's expert-parallel runtime.
pub(crate) struct ExpertRuntime {
    parallel: Arc<ExpertParallel>,
    functions: ExpertFunctions,
    graphs: Option<GraphFunctions>,
    device: i32,
    /// This rank holds only routed experts and runs no forward of its own.
    dedicated: bool,
    peers: Vec<Peer>,
    /// Microbatch threads, with two or more microbatches.
    lanes: Option<MicrobatchLanes>,
    /// Captured joins of every microbatch, on the first stream.
    joins: Option<JoinGraphs>,
    /// Startup warmup: source ranks announce each eager expert step
    /// instead of agreeing on it.
    warming: AtomicBool,
    /// The component bindings of a dedicated expert rank.
    bindings: Vec<ComponentBinding>,
    /// One graph pool per microbatch for captured joins.
    pools: Vec<Handle>,
    /// The layers that exchange tokens, compared across the union.
    layers: Vec<crate::parallel::experts::ExpertLayer>,
    /// The yield hook given to every transport, retained for its lifetime.
    _hooks: Option<Any>,
    closed: bool,
}

// SAFETY: the retained hook value is an immutable reference-counted TVM-FFI
// function object with atomic counts, callable from any thread; every
// other field is Send. The runtime is used by the executor thread, and its
// lanes' threads only receive owned calls.
unsafe impl Send for ExpertRuntime {}

impl ExpertRuntime {
    pub(crate) fn parallel(&self) -> &Arc<ExpertParallel> {
        &self.parallel
    }

    pub(crate) fn dedicated(&self) -> bool {
        self.dedicated
    }

    /// Microbatches; one without microbatch execution.
    pub(crate) fn microbatches(&self) -> usize {
        self.peers.len()
    }

    /// The transport of microbatch `peer`.
    pub(crate) fn transport(&self, peer: usize) -> &Handle {
        &self.peers[peer].transport
    }

    /// The stream of microbatch `peer`.
    pub(crate) fn stream(&self, peer: usize) -> &Stream {
        &self.peers[peer].stream
    }

    /// Install every microbatch's bound join, in microbatch order.
    pub(crate) fn set_joins(&mut self, joins: Vec<Function>) -> Result<()> {
        if joins.len() != self.peers.len() {
            return Err(Error::Invariant(
                "every microbatch binds one expert join".into(),
            ));
        }
        for (peer, join) in self.peers.iter_mut().zip(joins) {
            peer.join = Some(join);
        }
        Ok(())
    }

    /// Whether every microbatch's join is bound.
    pub(crate) fn has_joins(&self) -> bool {
        self.peers.iter().all(|peer| peer.join.is_some())
    }

    /// Whether captured joins have their graph pools.
    pub(crate) fn has_pools(&self) -> bool {
        !self.pools.is_empty()
    }

    /// Use `pools`, one per microbatch, for captured joins.
    pub(crate) fn set_pools(&mut self, pools: Vec<Handle>) {
        self.pools = pools;
    }

    /// Whether startup still announces warmup steps.
    pub(crate) fn warming(&self) -> bool {
        self.warming.load(Ordering::Acquire)
    }

    /// End warmup announcements: steps from now on are agreed.
    pub(crate) fn serve(&self) {
        self.warming.store(false, Ordering::Release);
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

    /// Order every microbatch stream after `parent`'s submitted work, before
    /// the caller stages microbatch inputs.
    pub(crate) fn fork(&self, parent: &Stream) -> Result<()> {
        if self.peers.len() < 2 {
            return Ok(());
        }
        let event = uniserve_cuda::Event::new(uniserve_cuda::EventFlags::default());
        event.record(parent)?;
        for peer in &self.peers[1..] {
            peer.stream.wait_event(&event)?;
        }
        Ok(())
    }

    /// Order `parent` after every microbatch stream's submitted work.
    pub(crate) fn join_streams(&self, parent: &Stream) -> Result<()> {
        for peer in &self.peers[1..] {
            let event = uniserve_cuda::Event::new(uniserve_cuda::EventFlags::default());
            event.record(&peer.stream)?;
            parent.wait_event(&event)?;
        }
        Ok(())
    }

    /// The per-rank capacity of the expert step that runs this rank's
    /// forward of `tokens` tokens and call `kind`.
    ///
    /// While serving, the ranks agree on it; until this rank's forward is
    /// selected, it joins every other selected step (replaying its captured
    /// joins). During startup warmup, the forward runs at `forced` or the
    /// smallest capacity holding its tokens, announced to dedicated expert
    /// ranks.
    pub(crate) fn capacity(
        &self,
        tokens: usize,
        kind: i64,
        forced: Option<usize>,
    ) -> Result<usize> {
        if self.warming() {
            let capacity = match forced {
                Some(capacity) => capacity,
                None => self.parallel.capacity_for(tokens)?,
            };
            return self.parallel.warmup(capacity);
        }
        if forced.is_some() {
            return Err(Error::State(
                "a forced expert capacity runs only during warmup",
            ));
        }
        loop {
            let agreement = self.parallel.agree(tokens, kind, false)?;
            if agreement.capacity == 0 {
                // No complete source group is ready yet; ask again.
                continue;
            }
            if agreement.active {
                return Ok(agreement.capacity);
            }
            self.join(agreement.capacity)?;
        }
    }

    /// Run one expert step of `capacity`: every populated microbatch's call
    /// and every empty microbatch's join. With microbatches, the calls run
    /// in rotation on the microbatch threads, and `parent` waits on every
    /// microbatch stream before this returns.
    pub(crate) fn run(
        &self,
        parent: &Stream,
        calls: Vec<Option<DeferredCall>>,
        capacity: usize,
    ) -> Result<()> {
        if calls.len() != self.peers.len() {
            return Err(Error::Invariant(
                "an expert step has one call per microbatch".into(),
            ));
        }
        let Some(lanes) = &self.lanes else {
            return match calls.into_iter().next().flatten() {
                Some(call) => call.invoke(),
                None => self.peer_join(0, capacity)?(),
            };
        };
        let calls = calls
            .into_iter()
            .enumerate()
            .map(|(peer, call)| match call {
                Some(call) => Ok(Box::new(move || call.invoke()) as MicrobatchCall<()>),
                None => self.peer_join(peer, capacity),
            })
            .collect::<Result<Vec<_>>>()?;
        lanes.run(parent, &self.streams(), calls).map(drop)
    }

    /// Join one step of `capacity` with every microbatch: replay the
    /// captured joins, or join eagerly before they exist.
    pub(crate) fn join(&self, capacity: usize) -> Result<()> {
        let _range = range(c"uniserve.expert.join", None);
        if let (Some(joins), Some(_)) = (&self.joins, &self.graphs) {
            return joins.launch(capacity, &self.peers[0].stream);
        }
        self.join_eager(capacity)
    }

    /// Join one step of `capacity` eagerly with every microbatch, in
    /// rotation.
    pub(crate) fn join_eager(&self, capacity: usize) -> Result<()> {
        match &self.lanes {
            Some(lanes) => {
                let calls = (0..self.peers.len())
                    .map(|peer| self.eager_join(peer, capacity))
                    .collect::<Result<Vec<_>>>()?;
                lanes
                    .run(&self.peers[0].stream, &self.streams(), calls)
                    .map(drop)
            }
            None => self.eager_join(0, capacity)?(),
        }
    }

    /// Participate in the union's next step without a forward of this
    /// rank's own. Returns (ran a step, every rank left the union).
    pub(crate) fn idle(&self, leaving: bool) -> Result<(bool, bool)> {
        let agreement = self.parallel.agree(0, 0, leaving)?;
        if agreement.capacity == 0 {
            return Ok((false, agreement.released));
        }
        self.join(agreement.capacity)?;
        Ok((true, agreement.released))
    }

    /// The join of microbatch `peer` within a populated step: its captured
    /// graph, or its eager join before capture.
    fn peer_join(&self, peer: usize, capacity: usize) -> Result<MicrobatchCall<()>> {
        let state = &self.peers[peer];
        if let Some(joins) = &state.graphs {
            let graph = joins.graph(capacity)?.clone();
            let stream = state.stream.clone();
            return Ok(Box::new(move || graph.launch(&stream)));
        }
        self.eager_join(peer, capacity)
    }

    fn eager_join(&self, peer: usize, capacity: usize) -> Result<MicrobatchCall<()>> {
        let join = self.peers[peer]
            .join
            .clone()
            .ok_or(Error::State("an expert microbatch has no bound join"))?;
        let call = DeferredCall::new(join, vec![Any::from(capacity as i64)]);
        Ok(Box::new(move || call.invoke()))
    }

    fn streams(&self) -> Vec<Stream> {
        self.peers.iter().map(|peer| peer.stream.clone()).collect()
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
        if pools.len() != self.peers.len() {
            return Err(Error::Invariant(
                "expert joins need one graph pool per microbatch".into(),
            ));
        }
        let capacities = self.parallel.capacities().to_vec();
        let primary = JoinGraphs::capture(
            &capacities,
            true,
            &pools[0],
            self.peers[0].stream.raw(),
            self.device,
            &|capacity| self.join_eager(capacity),
            &graphs,
        )?;
        self.joins = Some(primary);
        if self.peers.len() > 1 && !self.dedicated {
            for (peer, pool) in pools.iter().enumerate() {
                let join = self.eager_join_function(peer)?;
                let captured = JoinGraphs::capture(
                    &capacities,
                    false,
                    pool,
                    self.peers[peer].stream.raw(),
                    self.device,
                    &|capacity| join.invoke(&[(&Any::from(capacity as i64)).into()]),
                    &graphs,
                )?;
                self.peers[peer].graphs = Some(captured);
            }
        }
        let captured: usize = self.joins.iter().map(JoinGraphs::len).sum::<usize>()
            + self
                .peers
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

    fn eager_join_function(&self, peer: usize) -> Result<Function> {
        self.peers[peer]
            .join
            .clone()
            .ok_or(Error::State("an expert microbatch has no bound join"))
    }

    /// Release joins, threads, bindings and transports after every graph
    /// and binding using them retired. Transport close is collective over
    /// the union; an aborted close never waits on peers and keeps every
    /// resource until the process exits.
    #[allow(clippy::mem_forget)]
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        let mut failure = None;
        let mut record = |result: Result<()>| {
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        };
        if let Some(mut joins) = self.joins.take() {
            record(joins.close(aborted));
        }
        for peer in &mut self.peers {
            if let Some(mut graphs) = peer.graphs.take() {
                record(graphs.close(aborted));
            }
        }
        if let Some(lanes) = &mut self.lanes {
            if aborted {
                // A thread may be blocked on a failed peer.
                std::mem::forget(self.lanes.take());
            } else {
                record(lanes.close());
            }
        }
        for binding in &mut self.bindings {
            record(binding.close(aborted));
        }
        let peers = std::mem::take(&mut self.peers);
        if aborted {
            std::mem::forget(peers);
            return failure.map_or(Ok(()), Err);
        }
        for mut peer in peers {
            record(self.functions.close.invoke(&[peer.transport.view()]));
            // Peers no longer address the storage once every transport
            // passed its closing barrier.
            drop(peer.symmetric.take());
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for ExpertRuntime {
    fn drop(&mut self) {
        // A runtime dropped without a close belongs to a failed rank: peers
        // may be inside a collective, so nothing waits.
        let _ = self.close(true);
    }
}

/// Join the union's next step from the service loop. Returns (ran a step,
/// every member left the union).
pub(crate) fn join_step(
    backend: &WorkerBackend,
    leaving: bool,
) -> std::result::Result<(bool, bool), WorkerError> {
    let experts = backend.experts().ok_or_else(|| {
        WorkerError::classify(
            Error::State("expert-step participation requires an expert runtime"),
            None,
        )
    })?;
    experts
        .idle(leaving)
        .map_err(|error| WorkerError::classify(error, Some("expert step")))
}
