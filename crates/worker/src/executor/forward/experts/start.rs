//! Construction of the expert runtime: step protocol, transports, threads
//! and bindings.
//!
//! Every member of the union constructs at the same point of its startup,
//! after loading and before capacity planning, so the transports' device
//! workspaces are resident when the rank measures its free memory:
//!
//! 1. `experts.layers` describes the layers that exchange tokens;
//! 2. [`ExpertParallel::configure`] exchanges source memberships and, with
//!    dedicated expert ranks, the sources' token bound;
//! 3. per microbatch (forked streams beyond the first): split MegaMoE's
//!    peer-mapped storage, then `experts.create_exchange`, collectively;
//! 4. microbatch threads, each warmed with its thread-local handles.
//!
//! Bindings follow: a model rank's text runtime prepares its microbatch
//! bindings with the transports and hands their joins over
//! ([`ExpertRuntime::bind_joins`]); a dedicated expert rank binds its
//! expert layers itself ([`ExpertRuntime::bind_experts`]).

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use tvm_ffi::Any;
use uniserve_cuda::{DeviceId, Stream};

use super::{ExpertFunctions, ExpertRuntime, Peer};
use crate::config::{Device, WorkerLaunch};
use crate::ffi::{Backing, Fields, Function, Functions, Handle, Record, integer_list};
use crate::lane::Lanes;
use crate::model::LoadedModel;
use crate::model::bind::{self, Backends, BindingRequest, StreamWorkspaces, TextSize};
use crate::model::interface::Interface;
use crate::parallel::experts::{ExpertLayer, ExpertPlan, host_functions};
use crate::parallel::joins::GraphFunctions;
use crate::parallel::world::World;
use crate::parallel::{ExpertParallel, ExpertTransport, MicrobatchLanes, yield_microbatch};
use crate::transfer::PeerStorage;
use crate::{Error, Result};

/// What constructing the expert runtime borrows from the rank.
pub(crate) struct ExpertStart<'a> {
    pub(crate) interface: &'a Interface,
    pub(crate) launch: &'a WorkerLaunch,
    pub(crate) world: &'a World,
    pub(crate) model: &'a LoadedModel,
    pub(crate) lanes: &'a mut Lanes,
    /// Tokens one of this rank's forwards sends per expert step at most,
    /// graph padding included. Dedicated expert ranks receive the sources'
    /// bound instead.
    pub(crate) max_tokens: usize,
    /// The host's name, which identifies descriptor-shared peer storage.
    pub(crate) node: &'a str,
}

/// The stream the rank's forwards (and the first microbatch) run on: its
/// only execution lane, or the device stream.
fn primary_stream(lanes: &Lanes, device: i32) -> Result<Stream> {
    let stream = match lanes.lanes() {
        [] => lanes
            .stream_for(
                uniserve_ipc::CallKind::Forward(uniserve_ipc::ForwardMode::Prefill),
                DeviceId(device),
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

impl ExpertRuntime {
    /// Construct the runtime of a rank in an expert union, collectively.
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
        let functions = ExpertFunctions::from_interface(start.interface)?;

        let layers: tvm_ffi::Array<Any> = functions.layers.call_as(&[
            start.model.handle().view(),
            (&Any::from(i64::from(disaggregated))).into(),
        ])?;
        let layers = layers
            .iter()
            .enumerate()
            .map(|(index, layer)| Record::new(format!("experts.layers[{index}]"), layer))
            .collect::<Result<Vec<_>>>()?;
        let microbatches = config.expert_microbatches;
        if layers.is_empty() {
            return Err(Error::Unsupported(
                "expert parallelism requires routed expert layers that exchange tokens".into(),
            ));
        }
        let layers = layers
            .iter()
            .map(ExpertLayer::from_record)
            .collect::<Result<Vec<_>>>()?;
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

        let mut streams = vec![primary_stream(start.lanes, device)?];
        for _ in 1..microbatches {
            let fork = start.lanes.fork_microbatch()?;
            streams.push(fork.stream()?.clone());
        }
        let hooks = (microbatches > 1).then(|| {
            Any::from(tvm_ffi::Function::from_packed(|_| {
                yield_microbatch().map_err(|error| {
                    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
                })?;
                Ok(Any::default())
            }))
        });
        let group = Fields::new()
            .with("rank", rank as i64)
            .with("size", size as i64)
            .with("attention_ranks", union.attention_ranks as i64)
            .into_any();
        let hook_map = Fields::new().optional("yield", hooks.clone()).into_any();
        let bootstrap = host_functions(&host);

        let mut runtime = Self {
            parallel,
            functions: functions.clone(),
            graphs: if config.graph_policy == "off" {
                None
            } else {
                Some(GraphFunctions::from_interface(start.interface)?)
            },
            device,
            dedicated,
            peers: Vec::with_capacity(microbatches),
            lanes: None,
            joins: None,
            warming: AtomicBool::new(true),
            bindings: Vec::new(),
            pools: Vec::new(),
            layers,
            _hooks: hooks,
            closed: false,
        };
        for stream in streams {
            let bytes: i64 = functions.symmetric_bytes.call_as(&[
                start.model.handle().view(),
                (&Any::from(tvm_ffi::String::from(transport.as_str()))).into(),
                (&group).into(),
                (&Any::from(max_tokens as i64)).into(),
            ])?;
            let symmetric = if bytes > 0 {
                Some(PeerStorage::allocate(
                    device,
                    bytes as usize,
                    start.node,
                    &*host
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                )?)
            } else {
                None
            };
            let storage = symmetric
                .as_ref()
                .map(|storage| -> Result<Any> {
                    let view = crate::storage::memory::StorageView::whole(
                        Arc::clone(storage.local()) as Arc<dyn crate::ffi::Backing>,
                        crate::ffi::DType::I8,
                    )?;
                    let base = storage.peers().address();
                    let pointers = (0..size)
                        .map(|member| (base + (member * storage.bytes_each()) as u64) as i64);
                    Ok(Fields::new()
                        .with("storage", view.to_tensor()?)
                        .with("pointers", integer_list(pointers))
                        .into_any())
                })
                .transpose()?;
            let created = functions.create.call_values(&[
                start.model.handle().to_any(),
                Any::from(tvm_ffi::String::from(transport.as_str())),
                group.clone(),
                bootstrap.clone(),
                Any::from(max_tokens as i64),
                Any::from(i64::from(device)),
                Any::from(stream.raw() as i64),
                hook_map.clone(),
                storage.unwrap_or_default(),
            ])?;
            runtime.peers.push(Peer {
                transport: Handle::new(created)?,
                stream,
                join: None,
                graphs: None,
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
                crate::ffi::cooperative(|| warm.invoke(&[(&Any::from(i64::from(device))).into()]))
            }))?;
            runtime.lanes = Some(lanes);
        }
        tracing::info!(
            target: "uniserve_worker.launch",
            "expert union member {rank} of {size}: {} transport over {} layers, {microbatches} microbatches, {max_tokens} tokens per rank and step",
            transport.as_str(),
            runtime.layers.len(),
        );
        Ok(Some(runtime))
    }

    /// Bind the joins of every microbatch's prepared binding (`prepared`,
    /// in microbatch order), then the transports' shared microbatch plan.
    pub(crate) fn bind_joins(&mut self, prepared: &[&Handle]) -> Result<()> {
        let joins = prepared
            .iter()
            .map(|handle| {
                Functions::new("experts.bind", self.functions.bind.call(&[handle.view()])?)?
                    .get("join")
            })
            .collect::<Result<Vec<Function>>>()?;
        self.set_joins(joins)?;
        let transports = crate::ffi::list(self.peers.iter().map(|peer| peer.transport.to_any()));
        self.functions
            .bind_microbatches
            .invoke(&[(&Any::from(transports)).into()])
    }

    /// Bind a dedicated expert rank's layers once per microbatch, each on
    /// its stream with its transport, over Rust workspaces sized for the
    /// sources' token bound.
    pub(crate) fn bind_experts(
        &mut self,
        interface: &Interface,
        model: &LoadedModel,
        workspaces: &mut StreamWorkspaces,
    ) -> Result<()> {
        let mut prepared = Vec::new();
        for peer in &self.peers {
            prepared.push(bind::prepare(
                interface,
                model.handle(),
                BindingRequest {
                    component: "experts".into(),
                    path: String::new(),
                    device: Device::Cuda(self.device),
                    stream: peer.stream.raw(),
                    size: Some(TextSize {
                        tokens: self.parallel.max_tokens(),
                        rows: 1,
                    }),
                    backends: Backends::default(),
                    derive_host_lengths: false,
                    experts: Some(peer.transport.clone()),
                    weights: None,
                    cache: None,
                },
            )?);
        }
        // Expert layers declare no symmetric workspace.
        self.bindings = bind::bind(interface, workspaces, prepared, None)?;
        let handles: Vec<Handle> = self
            .bindings
            .iter()
            .map(|binding| binding.prepared().clone())
            .collect();
        self.bind_joins(&handles.iter().collect::<Vec<_>>())
    }

    /// Compare every rank's plan before warmup enters collectives. `calls`
    /// is a source rank's warmup and capture plan.
    pub(crate) fn compare_plans(&self, calls: Option<serde_json::Value>) -> Result<()> {
        self.parallel.compare_plans(&ExpertPlan {
            capacities: self.parallel.capacities().to_vec(),
            microbatches: self.peers.len(),
            layers: self.layers.clone(),
            calls,
        })
    }
}
