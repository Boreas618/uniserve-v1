//! `ExpertUnion`: this rank's member of an expert union.
//!
//! Ranks of an expert union execute every expert step together: the ranks
//! whose forward the step runs send their tokens, every other rank joins
//! each expert layer with no tokens of its own. The member holds this rank's
//! side of that protocol, each part with its owner in this module tree:
//!
//! - the union's step agreement ([`ExpertParallel`]) and, during startup,
//!   the warmup announcements expert-only ranks follow (here);
//! - one expert exchange per microbatch (an opaque library object, created
//!   collectively by `uniserve.distributed.experts.create_exchange`),
//!   each with its own stream, and the microbatch threads that rotate eager
//!   forwards (`microbatches.rs`);
//! - joins: per microbatch the owner of an execution context, whose
//!   expert layers a join runs with no tokens, and their captured graphs
//!   (`joins.rs`).
//!
//! A runner's forwards are expert steps of its lane's jobs, which the
//! execution thread runs through [`ExpertUnion::capacity`] and
//! [`ExpertUnion::run`]; between its own steps, and on expert-only ranks
//! always, the service loop submits a job that joins the union's next step
//! ([`ExpertUnion::idle`]).
//!
//! Every member constructs at the same point of its startup, after loading
//! and before capacity planning, so the exchanges' device workspaces are
//! resident when the rank measures its free memory:
//!
//! 1. `layers` describes the layers that exchange tokens;
//! 2. [`ExpertParallel::configure`] exchanges source memberships and, with
//!    expert-only ranks, the sources' token bound;
//! 3. per microbatch (forked streams beyond the first): split MegaMoE's
//!    peer-mapped storage, then `create_exchange`, collectively;
//! 4. microbatch threads, each warmed with its thread-local handles.
//!
//! Runners follow: one is constructed per microbatch with its exchange and
//! handed over for joins ([`ExpertUnion::bind_joins`]); an expert-only rank
//! constructs its expert layers itself ([`ExpertUnion::bind_experts`]).
//!
//! Warmup reaches the same collectives in the same order on every member:
//! [`WarmupHooks::prepare`] compares plans before any collective step;
//! [`WarmupHooks::capture`], after the runners captured their graphs (each
//! capture's eager warmup an expert step the sources announced), lets
//! expert-only ranks follow the announcements until the sources announce
//! zero, then captures every rank's joins, warming each capacity, largest
//! first, as a collective step. Steps are agreed from then on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tvm_ffi::Any;
use uniserve_cuda::{DeviceId, Stream};

use super::experts::{ExpertLayer, ExpertPlan, host_bootstrap};
use super::joins::{GraphFunctions, JoinGraphs};
use super::world::World;
use super::{ExpertParallel, ExpertTransport, MicrobatchLanes, yield_microbatch};
use crate::config::{Device, WorkerLaunch};
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::ffi::{Backing, Function, Handle, Interpreter, integer_list};
use crate::lane::Job;
use crate::runner::scratch::Scratches;
use crate::stream::Streams;
use crate::transfer::PeerStorage;
use crate::warmup::WarmupHooks;
use crate::worker::LoadedModel;
use crate::{Error, Result, lock};

/// The library module of expert exchanges.
const MODULE: &str = "uniserve.distributed.experts";

/// The library class over the worker's host collectives.
const HOST_BOOTSTRAP: &str = "uniserve.backends.experts.HostBootstrap";

/// The library functions of expert exchanges, reached by import path.
#[derive(Clone)]
pub(super) struct ExpertFunctions {
    layers: Function,
    symmetric_bytes: Function,
    create: Function,
    bind_microbatches: Function,
    /// `join(owner, capacity)`: one expert step this rank joins with no
    /// tokens, on the stream of the owner's execution context.
    pub(super) join: Function,
    warm_thread: Function,
    close: Function,
    /// `ExpertLayers.close(layers)`.
    close_layers: Function,
}

impl ExpertFunctions {
    fn import() -> Result<Self> {
        let function = |name: &str| Interpreter::import(&format!("{MODULE}.{name}"));
        Ok(Self {
            layers: function("layers")?,
            symmetric_bytes: function("symmetric_bytes")?,
            create: function("create_exchange")?,
            bind_microbatches: function("bind_microbatches")?,
            join: function("join")?,
            warm_thread: function("warm_thread")?,
            close: function("close_exchange")?,
            close_layers: function("ExpertLayers.close")?,
        })
    }
}

/// One microbatch: its exchange, stream and joins.
pub(super) struct Microbatch {
    /// The exchange `create_exchange` returned.
    pub(super) exchange: Handle,
    /// The microbatch's stream: the execution lane's stream for the first
    /// microbatch, a fork of it for the others. The rank's lanes close the
    /// streams after this member.
    pub(super) stream: Stream,
    /// The owner of the execution context whose expert layers this
    /// microbatch's joins run: a runner or an expert-only rank's
    /// `ExpertLayers`.
    pub(super) prepared: Option<Handle>,
    /// This microbatch's captured joins, so an empty microbatch of a
    /// populated step launches them on its own stream. Set once, by the
    /// warmup job.
    pub(super) graphs: OnceLock<JoinGraphs>,
    /// Peer-mapped storage of split MegaMoE, retained until the exchange
    /// closed.
    symmetric: Option<PeerStorage>,
}

/// This rank's member of an expert union.
pub(crate) struct ExpertUnion {
    parallel: Arc<ExpertParallel>,
    pub(super) functions: ExpertFunctions,
    pub(super) graphs: Option<GraphFunctions>,
    pub(super) device: DeviceId,
    /// This rank holds only routed experts and runs no forward of its own.
    pub(super) dedicated: bool,
    pub(super) microbatches: Vec<Microbatch>,
    /// Microbatch threads, with two or more microbatches.
    pub(super) lanes: Option<MicrobatchLanes>,
    /// Captured joins of every microbatch, on the first stream. Set once,
    /// by the warmup job.
    pub(super) joins: OnceLock<JoinGraphs>,
    /// Startup warmup: source ranks announce each eager expert step
    /// instead of agreeing on it.
    warming: AtomicBool,
    /// The `ExpertLayers` of an expert-only rank, one per microbatch.
    layers_owners: Vec<Handle>,
    /// One graph pool per microbatch for captured joins.
    pub(super) pools: Vec<Handle>,
    /// The layers that exchange tokens, compared across the union.
    layers: Vec<ExpertLayer>,
    /// The yield hook given to every exchange, retained for its lifetime.
    _yield_turn: Option<Handle>,
    closed: bool,
}

// A job that runs an expert step holds the member (`Arc`) while the lane's
// execution thread performs it. What the two threads reach through it:
//
// - facts fixed at startup, which both read: the capacities, the streams
//   and exchanges of the microbatches and their pools;
// - the captured joins: set once by the warmup job, on the execution
//   thread, and read by it afterwards;
// - the step agreement (`ExpertParallel`: its selection state and the host
//   group, each behind a lock) and the microbatch rotation: used by the
//   execution thread only, in jobs (warmup forwards, the end of warmup,
//   steps and joins). The control thread binds the member before the first
//   job and closes it after the last, when no job is in flight
//   (`WorkerBackend::experts_mut` refuses while one is);
// - the warmup flag: written by the control thread between jobs, read by
//   both.
//
// So the mutable parts have one user at a time, handed over with the job,
// and the type needs no `unsafe` to be shared.

impl ExpertUnion {
    pub(crate) fn parallel(&self) -> &Arc<ExpertParallel> {
        &self.parallel
    }

    /// Whether startup still announces warmup steps.
    pub(crate) fn warming(&self) -> bool {
        self.warming.load(Ordering::Acquire)
    }

    /// End warmup announcements: steps from now on are agreed.
    pub(crate) fn serve(&self) {
        self.warming.store(false, Ordering::Release);
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

    /// Release joins, threads, bindings and transports after every graph
    /// and binding using them retired. Transport close is collective over
    /// the union; an aborted close never waits on peers and keeps every
    /// resource until the process exits.
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
        for peer in &mut self.microbatches {
            if let Some(mut graphs) = peer.graphs.take() {
                record(graphs.close(aborted));
            }
        }
        if let Some(lanes) = &mut self.lanes {
            if aborted {
                // A thread may be blocked on a failed peer.
                crate::leak(self.lanes.take());
            } else {
                record(lanes.close());
            }
        }
        // An aborted close keeps the layers' contexts: device work may
        // still address their workspace.
        if !aborted {
            for owner in std::mem::take(&mut self.layers_owners) {
                record(self.functions.close_layers.invoke(&[owner.view()]));
            }
        }
        let peers = std::mem::take(&mut self.microbatches);
        if aborted {
            crate::leak(peers);
            return failure.map_or(Ok(()), Err);
        }
        for mut peer in peers {
            record(self.functions.close.invoke(&[peer.exchange.view()]));
            // Peers no longer address the storage once every transport
            // passed its closing barrier.
            drop(peer.symmetric.take());
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for ExpertUnion {
    fn drop(&mut self) {
        // A runtime dropped without a close belongs to a failed rank: peers
        // may be inside a collective, so nothing waits.
        let _ = self.close(true);
    }
}

/// What constructing the member borrows from the rank.
pub(crate) struct ExpertStart<'a> {
    pub(crate) launch: &'a WorkerLaunch,
    pub(crate) world: &'a World,
    pub(crate) model: &'a LoadedModel,
    pub(crate) streams: &'a mut Streams,
    /// Tokens one of this rank's forwards sends per expert step at most,
    /// graph padding included. Dedicated expert ranks receive the sources'
    /// bound instead.
    pub(crate) max_tokens: usize,
    /// The host's name, which identifies descriptor-shared peer storage.
    pub(crate) node: &'a str,
}

/// The stream the rank's forwards (and the first microbatch) run on: its
/// only execution lane, or the device stream.
fn primary_stream(streams: &Streams, device: DeviceId) -> Result<Stream> {
    let stream = match streams.lanes() {
        [] => streams
            .stream_for(
                uniserve_ipc::CallKind::Forward(uniserve_ipc::ForwardMode::Prefill),
                device,
            )
            .ok_or(Error::State("expert parallelism requires a CUDA stream"))?,
        [lane] => &lane.stream,
        _ => {
            return Err(Error::Unsupported(
                "expert parallelism runs on exactly one execution lane".into(),
            ));
        }
    };
    Ok(stream.stream()?.clone())
}

impl ExpertUnion {
    /// Construct this rank's member of its expert union, collectively.
    /// `None` without token exchange: no union, or weight prefetch.
    pub(crate) fn start(start: ExpertStart<'_>) -> Result<Option<Self>> {
        let Some(union) = &start.launch.expert_parallel else {
            return Ok(None);
        };
        let Some(transport) = ExpertTransport::parse(&union.exchange)? else {
            return Ok(None);
        };
        let config = &start.launch.execution;
        let dedicated = config.role == "experts";
        let disaggregated = union.disaggregated();
        let Device::Cuda(device) = start.launch.device() else {
            return Err(Error::Unsupported(
                "expert parallelism requires a CUDA device".into(),
            ));
        };
        let functions = ExpertFunctions::import()?;

        // `num_experts, top_k, hidden_size` of each layer, back to back.
        let sizes = functions.layers.call_values(&[
            start.model.handle().to_any(),
            Any::from(i64::from(disaggregated)),
        ])?;
        let sizes = crate::ffi::integers(sizes)
            .map_err(|error| Error::numerical(&format!("{MODULE}.layers"), &error))?;
        let layers: Vec<ExpertLayer> = sizes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[num_experts, top_k, hidden_size]| ExpertLayer {
                num_experts: num_experts as usize,
                top_k: top_k as usize,
                hidden_size: hidden_size as usize,
            })
            .collect();
        let microbatches = config.expert_microbatches;
        if layers.is_empty() {
            return Err(Error::Unsupported(
                "expert parallelism requires routed expert layers that exchange tokens".into(),
            ));
        }
        if layers.iter().any(|layer| *layer != layers[0]) {
            return Err(Error::Unsupported(
                "expert-parallel layers share one expert count, top-k and hidden size".into(),
            ));
        }

        // The union's members, sharing the startup connections.
        let union_group = start
            .world
            .union
            .as_ref()
            .ok_or(Error::State("an expert union requires its host group"))?;
        let members: Vec<usize> = (0..union_group.size()).collect();
        let group = union_group.subgroup(&members)?;
        // Expert steps wait on peers without a deadline: a peer may still
        // compile kernels before its first announcement or plan comparison,
        // and an idle union agrees indefinitely. A failed peer closes its
        // connections, which fails the waiting collective.
        group.set_timeout(None)?;
        let host = Arc::new(Mutex::new(group));
        let sources: Vec<usize> = if dedicated {
            Vec::new()
        } else {
            let first = union.rank - config.rank;
            (first..first + config.world_size).collect()
        };
        let parallel = Arc::new(ExpertParallel::configure(
            Arc::clone(&host),
            union.attention_ranks,
            &sources,
            start.max_tokens,
            transport,
        )?);
        let max_tokens = parallel.max_tokens();
        let (rank, size) = parallel.member();

        let mut streams = vec![primary_stream(start.streams, device)?];
        for _ in 1..microbatches {
            let fork = start.streams.fork_microbatch()?;
            streams.push(fork.stream()?.clone());
        }
        let yield_turn = (microbatches > 1).then(|| {
            Any::from(tvm_ffi::Function::from_packed(|_| {
                yield_microbatch().map_err(|error| {
                    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
                })?;
                Ok(Any::default())
            }))
        });
        // The library's host collectives over the union, which every
        // microbatch's exchange bootstraps its communicators with.
        let bootstrap = Interpreter::import(HOST_BOOTSTRAP)?.call_values(&host_bootstrap(&host))?;

        let mut runtime = Self {
            parallel,
            functions: functions.clone(),
            graphs: if config.graph_policy == "off" {
                None
            } else {
                Some(GraphFunctions::import()?)
            },
            device,
            dedicated,
            microbatches: Vec::with_capacity(microbatches),
            lanes: None,
            joins: OnceLock::new(),
            warming: AtomicBool::new(true),
            layers_owners: Vec::new(),
            pools: Vec::new(),
            layers,
            _yield_turn: yield_turn.clone().map(Handle::new).transpose()?,
            closed: false,
        };
        for stream in streams {
            let bytes: i64 = functions.symmetric_bytes.call_as(&[
                start.model.handle().view(),
                (&Any::from(tvm_ffi::String::from(transport.as_str()))).into(),
                (&Any::from(size as i64)).into(),
                (&Any::from(union.attention_ranks as i64)).into(),
                (&Any::from(max_tokens as i64)).into(),
            ])?;
            let symmetric = if bytes > 0 {
                Some(PeerStorage::allocate(
                    device,
                    bytes as usize,
                    start.node,
                    &lock(&host),
                )?)
            } else {
                None
            };
            // This rank's view of the mapped storage and every member's
            // address of its share; neither without mapped storage.
            let (view, pointers) = match &symmetric {
                Some(storage) => {
                    let view = crate::storage::memory::StorageView::whole(
                        Arc::clone(storage.local()) as Arc<dyn crate::ffi::Backing>,
                        crate::ffi::DType::I8,
                    )?;
                    let base = storage.peers().address();
                    let pointers = (0..size)
                        .map(|member| (base + (member * storage.bytes_each()) as u64) as i64);
                    (Some(view), integer_list(pointers))
                }
                None => (None, integer_list([])),
            };
            // The view's wrapping and the call take one handoff.
            let created = Interpreter::execute(|| -> Result<Any> {
                let storage = match &view {
                    Some(view) => view.handle()?.to_any(),
                    None => Any::default(),
                };
                functions.create.call_values(&[
                    start.model.handle().to_any(),
                    Any::from(tvm_ffi::String::from(transport.as_str())),
                    Any::from(rank as i64),
                    Any::from(size as i64),
                    Any::from(union.attention_ranks as i64),
                    bootstrap.clone(),
                    Any::from(max_tokens as i64),
                    Any::from(i64::from(device.0)),
                    Any::from(stream.raw() as i64),
                    yield_turn.clone().unwrap_or_default(),
                    storage,
                    Any::from(pointers),
                ])
            })?;
            runtime.microbatches.push(Microbatch {
                exchange: Handle::new(created)?,
                stream,
                prepared: None,
                graphs: OnceLock::new(),
                symmetric,
            });
        }
        if microbatches > 1 {
            let lanes = MicrobatchLanes::new(microbatches)?;
            let warm = functions.warm_thread.clone();
            // Thread-local numerical state belongs to the thread an eager
            // forward runs on, so each microbatch thread makes this call
            // itself.
            lanes.each(Arc::new(move |_| {
                crate::ffi::cooperative(|| warm.invoke(&[(&Any::from(i64::from(device.0))).into()]))
            }))?;
            runtime.lanes = Some(lanes);
        }
        tracing::info!(
            "expert union member {rank} of {size}: {} transport over {} layers, {microbatches} microbatches, {max_tokens} tokens per rank and step",
            transport.as_str(),
            runtime.layers.len(),
        );
        Ok(Some(runtime))
    }

    /// Take the context owner of every microbatch (`prepared`, in
    /// microbatch order) for its joins, then bind the exchanges' shared
    /// microbatch plan.
    pub(crate) fn bind_joins(&mut self, prepared: &[&Handle]) -> Result<()> {
        if prepared.len() != self.microbatches.len() {
            return Err(Error::Invariant(
                "every microbatch joins through one execution context".into(),
            ));
        }
        for (microbatch, &prepared) in self.microbatches.iter_mut().zip(prepared) {
            microbatch.prepared = Some(prepared.clone());
        }
        let exchanges = crate::ffi::list(
            self.microbatches
                .iter()
                .map(|microbatch| microbatch.exchange.to_any()),
        );
        self.functions
            .bind_microbatches
            .invoke(&[(&Any::from(exchanges)).into()])
    }

    /// Construct an expert-only rank's layers once per microbatch
    /// (`ExpertLayers`), each on its stream with its exchange and the
    /// stream's scratch, prepared for the sources' token bound.
    pub(crate) fn bind_experts(
        &mut self,
        model: &LoadedModel,
        scratches: &mut Scratches,
    ) -> Result<()> {
        let construct = Interpreter::import(&format!("{MODULE}.ExpertLayers"))?;
        for peer in &self.microbatches {
            let stream = peer.stream.raw();
            let scratch = scratches.get(stream, Some(self.device))?;
            self.layers_owners.push(Handle::new(construct.call_values(&[
                model.handle().to_any(),
                Any::from(stream as i64),
                scratch.to_any(),
                peer.exchange.to_any(),
                Any::from(self.parallel.max_tokens() as i64),
            ])?)?);
        }
        let owners = self.layers_owners.clone();
        self.bind_joins(&owners.iter().collect::<Vec<_>>())
    }

    /// Compare every rank's plan before warmup enters collectives. `calls`
    /// is a source rank's warmup and capture plan.
    pub(crate) fn compare_plans(&self, calls: Option<serde_json::Value>) -> Result<()> {
        self.parallel.compare_plans(&ExpertPlan {
            capacities: self.parallel.capacities().to_vec(),
            microbatches: self.microbatches.len(),
            layers: self.layers.clone(),
            calls,
        })
    }
}

/// The expert union's warmup work.
pub(crate) struct ExpertWarmup;

fn failed(phase: &'static str) -> impl Fn(Error) -> WorkerError {
    move |error| WorkerError::classify(error, Some(phase)).with_phase("warmup")
}

impl WarmupHooks for ExpertWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let text = backend.text_mut().map(|text| text.capture_plan());
        let diffusion_lm: Vec<_> = backend
            .diffusion_lm_mut()
            .iter()
            .map(|runner| runner.capture_plan())
            .collect();
        let calls = (text.is_some() || !diffusion_lm.is_empty())
            .then(|| serde_json::json!({"text": text, "diffusion_lm": diffusion_lm}));
        let Some(experts) = backend.experts() else {
            return Ok(());
        };
        experts
            .compare_plans(calls)
            .map_err(failed("expert plan comparison"))
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let Some(experts) = backend.experts().cloned() else {
            return Ok(());
        };

        // The end of the union's warmup is a job of the lane its forwards
        // and joins run on: it announces, joins and captures on that
        // lane's stream.
        let job = Job::new();
        job.expert_warmup(experts);
        let completion = backend
            .job_queues()
            .block_on(
                uniserve_ipc::CallKind::Forward(uniserve_ipc::ForwardMode::Decode),
                job,
                crate::lane::BLOCK_ON_TIMEOUT,
            )
            .map_err(failed("expert warmup"))?;
        completion
            .check()
            .map(drop)
            .map_err(|failure| failed("expert warmup")(failure.error))
    }
}

impl ExpertUnion {
    /// End the union's warmup on this rank: follow the announced warmup
    /// steps, start serving, and capture the joins every rank replays
    /// outside its own steps. Performed once, by the warmup job, on the
    /// lane's execution thread.
    pub(crate) fn finish_warmup(&self) -> Result<()> {
        self.follow()?;
        self.serve();
        self.capture_joins()
    }

    /// Dedicated expert ranks execute every announced warmup step; sources
    /// end their announcements with zero.
    fn follow(&self) -> Result<()> {
        let parallel = &self.parallel;
        if parallel.attention_ranks() == 0 {
            return Ok(());
        }
        if !self.dedicated {
            parallel.warmup(0)?;
            return Ok(());
        }
        let mut steps = 0;
        loop {
            let capacity = parallel.warmup(0)?;
            if capacity == 0 {
                break;
            }
            self.join_eager(capacity)?;
            steps += 1;
        }
        tracing::info!("joined {steps} announced expert warmup steps");
        Ok(())
    }
}
