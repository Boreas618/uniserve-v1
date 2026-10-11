//! Startup of the diffusion LM runner: bounds, banks, buffers and the
//! Python half.
//!
//! [`Bounds`] derives every bound of the runner from the worker
//! configuration, the prefix cache plan and the model's facts, once for
//! capacity planning (bytes to reserve beside the prefix cache pool) and
//! again for allocation, so both agree. [`DiffusionLMRunner::start`] then
//! creates one runner per diffusion LM component placed on the rank: it
//! allocates the runner's buffers, banks and graph pool and constructs the
//! Python runner on the lane's stream, which prepares the model's execution
//! context over the stream's scratch, the prefix cache pool and the model's
//! stream collectives.

use uniserve_core::CanvasSampling;
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{CallKind, ForwardMode};

use super::buffers::{BankLayout, Banks, DenoiseLayout, DenoiseViews, Dimensions};
use super::facts::Facts;
use super::shapes::{SLOT_BUCKETS, Shapes, chunk_rows};
use super::{CANDIDATES, DiffusionLMRunner, Functions, RUNNER};
use crate::config::{Device, WorkerConfig};
use crate::cuda_graph::{GraphStorage, GraphTable, reserve_pools};
use crate::ffi::Handle;
use crate::lane::Lanes;
use crate::model::LoadedModel;
use crate::model::bind;
use crate::model::components::RunnerKind;
use crate::model::interface::Interface;
use crate::model::registry::Components;
use crate::parallel::{DwdpManager, ExpertUnion};
use crate::runner::input_buffers::InputBuffers;
use crate::runner::methods::{Argument, Methods};
use crate::runner::scratch::Scratches;
use crate::storage::{GroupShape, KVCache, KVPlan, OutputPool};
use crate::{Error, Result};

/// The call kind the runner serves.
pub(crate) const KIND: CallKind = CallKind::Forward(ForwardMode::TokenDenoising);

/// Every bound of the runner.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Bounds {
    /// Blocks a forward may carry: the graph buckets' limit.
    capacity: usize,
    /// Packed sequences of one forward: an `attend` bucket's padding
    /// doubles its blocks.
    rows: usize,
    tokens: usize,
    groups: Vec<GroupShape>,
    /// Tokens per page and whether pages retire behind a window, per
    /// numerical block table.
    page_tokens: Vec<i64>,
    windowed: Vec<i64>,
    /// Page columns every table's rows hold.
    widths: Vec<usize>,
    /// Startup captures graphs.
    graphs: bool,
    /// Denoising rows and sampler chunk rows of a generating rank.
    denoise: Option<(usize, usize)>,
}

impl Bounds {
    /// Derive the bounds. `units` is the prefix cache pool's unit count,
    /// sentinel included, once fitted; before fitting, the pool bounds
    /// nothing.
    fn new(
        config: &WorkerConfig,
        request_slots: usize,
        cache: &KVPlan,
        units: Option<u32>,
        facts: &Facts,
        cuda: bool,
    ) -> Result<Self> {
        let mut max_rows = config.max_batch_calls.min(request_slots.max(1));
        let mut tokens = config.max_batch_tokens;
        for lane in &config.lanes {
            max_rows = max_rows.min(lane.max_batch_calls.unwrap_or(max_rows));
            tokens = tokens.min(lane.max_batch_tokens.unwrap_or(tokens));
        }
        let length = facts.sequence_length;
        let groups = cache.group_shapes()?;
        let row_units: usize = groups
            .iter()
            .map(|group| group.units_per_page as usize)
            .sum();
        // Startup reads a one-page prefix per synthetic block, so the pool
        // bounds the blocks a captured bucket may hold.
        let pool_rows = units.map_or(usize::MAX, |units| {
            (units.saturating_sub(1) as usize) / row_units.max(1)
        });
        let graphs = cuda && config.graph_policy != "off";
        let capacity = max_rows.min(pool_rows).min(tokens / length);
        if capacity == 0 {
            return Err(Error::Unsupported(format!(
                "no block of {length} tokens fits a batch of {tokens} tokens and a pool of {pool_rows} blocks"
            )));
        }
        let rows = if graphs {
            (2 * capacity).max(max_rows)
        } else {
            max_rows
        };

        let mut page_tokens = Vec::new();
        let mut windowed = Vec::new();
        let mut widths = Vec::new();
        for group in &groups {
            let mut width = group.table_width(config.max_sequence_tokens as u64, tokens as u64);
            if let Some(units) = units {
                // A table cannot select more pages than the pool holds.
                width = width.min(
                    (u64::from(units.saturating_sub(1)) / u64::from(group.units_per_page)).max(1),
                );
            }
            for _ in 0..group.units_per_page {
                page_tokens.push(i64::from(group.page_tokens));
                windowed.push(i64::from(group.window.is_some()));
                widths.push(width as usize);
            }
        }

        let denoise = (facts.generates && config.canvas_sampling.is_some()).then(|| {
            let rows = if graphs {
                capacity
            } else {
                max_rows.min(tokens / length)
            }
            .min(request_slots);
            (rows, chunk_rows(length, facts.vocab_size, rows))
        });
        Ok(Self {
            capacity,
            rows,
            tokens,
            groups,
            page_tokens,
            windowed,
            widths,
            graphs,
            denoise,
        })
    }

    /// The banks of a generating rank serving `sampling`.
    fn banks(
        &self,
        facts: &Facts,
        request_slots: usize,
        sampling: &CanvasSampling,
    ) -> Option<BankLayout> {
        self.denoise.map(|_| BankLayout {
            slots: request_slots,
            length: facts.sequence_length,
            depth: sampling.stability_threshold as usize,
            hidden: facts.hidden_size,
            dtype: facts.dtype,
        })
    }

    /// The dimensions of the runner's input buffers.
    fn dimensions(&self, facts: &Facts, banks: Option<BankLayout>) -> Dimensions {
        Dimensions {
            rows: self.rows,
            tokens: self.tokens,
            tables: self.widths.len(),
            width: self.widths.iter().copied().max().unwrap_or(1).max(1),
            slot_rows: SLOT_BUCKETS[SLOT_BUCKETS.len() - 1],
            candidates: CANDIDATES,
            attend_widths: facts.attend_widths.clone(),
            attend_dtype: facts.attend_dtype,
            denoise: self
                .denoise
                .zip(banks)
                .map(|((rows, _), banks)| DenoiseLayout { rows, banks }),
        }
    }
}

/// The stream serving the runner's calls on `device`: the configured lane
/// serving token denoising, or the device stream.
fn stream(lanes: &Lanes, device: Device) -> Result<Option<Stream>> {
    let Some(index) = device.cuda_index() else {
        return Ok(None);
    };
    let mut serving = lanes.lanes().iter().filter(|lane| lane.serves(KIND));
    let stream = match (serving.next(), serving.next()) {
        (Some(lane), None) => lane.stream.stream()?.clone(),
        (None, _) if lanes.lanes().is_empty() => lanes
            .stream_for(KIND, DeviceId(index))
            .ok_or(Error::State("diffusion LM calls have no execution stream"))?
            .stream()?
            .clone(),
        _ => {
            return Err(Error::Invalid(format!(
                "computation {} is served by one lane",
                KIND.as_str()
            )));
        }
    };
    Ok(Some(stream))
}

/// What startup borrows from the rank.
pub(crate) struct Startup<'a> {
    pub(crate) interface: &'a Interface,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) request_slots: usize,
    pub(crate) queue_depth: usize,
    /// Bytes of one IPC message, which bounds a batch's completion values.
    pub(crate) max_payload_bytes: usize,
    pub(crate) model: &'a LoadedModel,
    /// Routed experts use an expert exchange, and the attention ranks of a
    /// disaggregated expert union (zero when every rank holds its experts).
    pub(crate) experts: (bool, usize),
    pub(crate) lanes: &'a Lanes,
    pub(crate) kv: Option<&'a KVCache>,
    pub(crate) components: &'a mut Components,
    /// The shared work areas of the rank's streams.
    pub(crate) scratches: &'a mut Scratches,
    pub(crate) graphs: &'a mut GraphStorage<String, Handle>,
    /// The device's startup stream, which zeroes new storage.
    pub(crate) preparation: Option<&'a Stream>,
    /// The worker's ranks, across which symmetric workspaces are mapped.
    pub(crate) peers: Option<&'a bind::PeerGroup<'a>>,
    /// The expert union member, when forwards are expert steps: the component
    /// exchanges through its transport on its stream.
    pub(crate) expert_runtime: Option<&'a mut ExpertUnion>,
    /// The distributed expert weights, with DWDP.
    pub(crate) weights: Option<&'a DwdpManager>,
}

/// The diffusion LM components placed on this rank: name, module path and
/// device of each.
fn placed(components: &Components) -> Result<Vec<(String, String, Device)>> {
    components
        .for_runner(RunnerKind::DiffusionLM)
        .filter(|placed| !placed.host() && placed.kinds.contains(&KIND))
        .map(|placed| {
            let forward = placed
                .calls
                .iter()
                .find(|call| {
                    call.runner() == Some(RunnerKind::DiffusionLM) && call.method == "forward"
                })
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "diffusion LM component {} declares no forward",
                        placed.name
                    ))
                })?;
            Ok((placed.name.clone(), forward.path.clone(), placed.device))
        })
        .collect()
}

/// One component between its preparation and its runner.
struct Planned {
    component: String,
    path: String,
    device: Device,
    cuda: Option<i32>,
    facts: Facts,
    bounds: Bounds,
    sampling: Option<CanvasSampling>,
    banks: Option<BankLayout>,
    stream: Option<Stream>,
}

impl DiffusionLMRunner {
    /// Create one runner per diffusion LM component placed on this rank.
    pub(crate) fn start(startup: Startup<'_>) -> Result<Vec<Self>> {
        let placed = placed(startup.components)?;
        if placed.is_empty() {
            return Ok(Vec::new());
        }
        let kv = startup
            .kv
            .ok_or_else(|| Error::Invalid("a diffusion LM reads a prefix cache".into()))?;
        let config = startup.config;
        let interface = startup.interface;

        // The attention layers' view of the pool.
        let cache = kv.pool(interface)?;

        let mut experts = startup.expert_runtime;
        if let Some(experts) = experts.as_ref()
            && experts.microbatches() > 1
        {
            return Err(Error::Unsupported(
                "diffusion LMs run expert steps without microbatches".into(),
            ));
        }

        let mut planned = Vec::new();
        for (component, path, device) in placed {
            let cuda = device.cuda_index();
            let facts = Facts::read(
                interface,
                startup.model.handle(),
                &path,
                startup.experts.0,
                startup.experts.1,
            )?;
            let bounds = Bounds::new(
                config,
                startup.request_slots,
                kv.plan(),
                Some(kv.planes().num_units()),
                &facts,
                cuda.is_some(),
            )?;

            // The served sampling generates the model's blocks.
            let sampling = match (&config.canvas_sampling, bounds.denoise) {
                (Some(sampling), Some(_)) => {
                    if sampling.canvas_length as usize != facts.sequence_length {
                        return Err(Error::Invalid(format!(
                            "the served canvas length {} is not the model's canvas length {}",
                            sampling.canvas_length, facts.sequence_length
                        )));
                    }
                    Some(*sampling)
                }
                _ => None,
            };
            let banks = sampling
                .as_ref()
                .and_then(|sampling| bounds.banks(&facts, startup.request_slots, sampling));

            let stream = stream(startup.lanes, device)?;
            if let Some(experts) = experts.as_ref()
                && stream.as_ref().map(Stream::raw) != Some(experts.stream(0).raw())
            {
                return Err(Error::Unsupported(
                    "expert parallelism runs every forward on one execution lane".into(),
                ));
            }
            planned.push(Planned {
                component,
                path,
                device,
                cuda,
                facts,
                bounds,
                sampling,
                banks,
                stream,
            });
        }

        let mut runners = Vec::new();
        for (index, plan) in planned.into_iter().enumerate() {
            let Planned {
                component,
                path,
                device,
                cuda,
                facts,
                bounds,
                sampling,
                banks,
                stream,
            } = plan;
            // The model's collectives on the lane's stream, which its
            // execution context is constructed with.
            let collectives = match startup.peers {
                Some(peers) => {
                    let owner = stream
                        .as_ref()
                        .and_then(|stream| startup.lanes.owner(stream.raw()));
                    peers.stream_collectives(&component, device, owner)?
                }
                None => None,
            };
            let handle = stream.as_ref().map_or(0, Stream::raw);
            let scratch = startup.scratches.get(interface, handle, cuda)?;
            let optional = |handle: Option<Handle>| handle.map_or(Argument::None, Argument::Handle);

            let dimensions = bounds.dimensions(&facts, banks);
            let banks = banks
                .map(|layout| Banks::allocate(layout, cuda, startup.preparation))
                .transpose()?;
            let buffers = InputBuffers::allocate(dimensions, cuda, startup.preparation)?;
            let state = DenoiseViews::of(&buffers)?;
            let chunk_rows = bounds.denoise.map_or(0, |(_, chunk_rows)| chunk_rows);
            let mut arguments = vec![
                Argument::Handle(startup.model.handle().clone()),
                Argument::Text(path),
                handle.into(),
                Argument::Handle(scratch),
                Argument::Handle(cache.clone()),
                optional(experts.as_ref().map(|experts| experts.transport(0).clone())),
                optional(startup.weights.map(|weights| weights.prefetch().clone())),
                // The functions of each communicator's collectives on the
                // stream, by communicator name.
                match &collectives {
                    Some(collectives) => {
                        Argument::Handle(Handle::new(collectives.natives(interface)?)?)
                    }
                    None => Argument::None,
                },
                bounds.tokens.into(),
                bounds.rows.into(),
                bounds.page_tokens.as_slice().into(),
                bounds.windowed.as_slice().into(),
                chunk_rows.into(),
            ];
            if let (Some(sampling), Some(banks)) = (&sampling, &banks) {
                arguments.extend([
                    Argument::Integer(i64::from(sampling.max_steps)),
                    Argument::Float(f64::from(sampling.entropy_bound)),
                    Argument::Float(f64::from(sampling.t_min)),
                    Argument::Float(f64::from(sampling.t_max)),
                    Argument::Float(f64::from(sampling.confidence_threshold)),
                    Argument::Integer(i64::from(sampling.stability_threshold)),
                ]);
                for (_, view) in banks.views() {
                    arguments.push(view.into());
                }
            }
            let methods = Methods::construct(interface, RUNNER, arguments)?;
            let functions = Functions::resolve(interface, &methods)?;
            // Idle ranks join expert steps through the model's expert
            // layers; a text binding may already provide them.
            if let Some(experts) = experts.as_deref_mut()
                && !experts.has_joins()
            {
                experts.bind_joins(&[methods.runner()])?;
            }

            let owner = format!("diffusion_lm:{component}");
            let pool = match (cuda, bounds.graphs) {
                (Some(ordinal), true) => {
                    let key = buffers.column("input_ids")?.address() as usize;
                    reserve_pools(
                        startup.graphs,
                        &interface.function("graphs.pool")?,
                        key,
                        owner.clone(),
                        &[ordinal],
                        None,
                    )?
                    .first()
                    .map(|pool| pool.value.clone())
                }
                _ => None,
            };
            if let Some(experts) = experts.as_deref_mut()
                && !experts.has_pools()
                && let Some(pool) = &pool
            {
                experts.set_pools(vec![pool.clone()]);
            }
            let shapes = Shapes::new(
                facts.sequence_length,
                bounds.capacity,
                sampling.map(|_| startup.request_slots),
            )?;

            // One lease per batch in flight. A lease holds every step's
            // outcome and tokens, or the candidate values of readouts, whose
            // four bytes each the batch's result message bounds.
            let max_calls = config.max_batch_calls.max(1);
            let words = (max_calls * (1 + facts.sequence_length))
                .max(startup.max_payload_bytes.div_ceil(4));
            let outputs = OutputPool::new(cuda, startup.queue_depth.max(1) + 1, words);
            runners.push(Self {
                index,
                graphs: GraphTable::new(owner.clone(), methods.graphs.release.clone()),
                breaks: startup.weights.map(|weights| weights.breaks().clone()),
                segments: 0,
                owner,
                component,
                methods,
                functions,
                facts,
                buffers,
                state,
                banks,
                sampling,
                chunk_rows,
                shapes,
                graph_widths: bounds.widths,
                groups: bounds.groups,
                pool,
                stream,
                plans_attention: true,
                collectives: collectives.into_iter().collect(),
                outputs,
            });
        }
        if let Some(stream) = startup.preparation {
            stream.synchronize()?;
        }
        Ok(runners)
    }
}

/// Device bytes the runners take beside the prefix cache pool for the
/// diffusion LM components of `components`: each one's buffers, a
/// generating rank's banks and the sampler workspace its Python half
/// allocates. `experts` is as in [`Startup`].
pub(crate) fn reserved_bytes(
    interface: &Interface,
    model: &LoadedModel,
    components: &Components,
    experts: (bool, usize),
    config: &WorkerConfig,
    request_slots: usize,
    cache: &KVPlan,
) -> Result<u64> {
    let cuda = config.device.cuda_index().is_some();
    let mut bytes = 0;
    for (_, path, _) in placed(components)? {
        let facts = Facts::read(interface, model.handle(), &path, experts.0, experts.1)?;
        let bounds = Bounds::new(config, request_slots, cache, None, &facts, cuda)?;
        let banks = config
            .canvas_sampling
            .as_ref()
            .and_then(|sampling| bounds.banks(&facts, request_slots, sampling));
        let workspace = match (bounds.denoise, banks) {
            (Some((_, chunk_rows)), Some(_)) => {
                facts.workspace_bytes(interface, chunk_rows, cuda)?
            }
            _ => 0,
        };
        let resident = banks.map(|banks| banks.bytes()).transpose()?.unwrap_or(0);
        let dimensions = bounds.dimensions(&facts, banks);
        bytes += InputBuffers::bytes(&dimensions) + resident + workspace;
    }
    Ok(bytes)
}
