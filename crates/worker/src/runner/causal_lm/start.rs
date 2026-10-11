//! Startup of a rank's causal LM runners: bounds, bindings and owned storage.
//!
//! [`TextBounds`] derives every text bound from the worker configuration and
//! the KV plan, once for capacity planning (bytes to reserve beside the KV
//! pool, graph-bounded call limits) and again for allocation, so both agree.
//! [`CausalLMRunners::start`] then creates one runner per execution stream
//! serving text call kinds: it allocates the runner's columns and graph
//! pool and constructs the Python runner, whose execution context borrows
//! the stream's scratch and reads the K/V pool.

use std::collections::HashMap;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{CallKind, ForwardMode};

use super::CausalLMRunners;
use crate::config::{Device, PREFILL_ROW_BUCKETS, WorkerConfig};
use crate::cuda_graph::GraphStorage;
use crate::cuda_graph::reserve_pools;
use crate::cuda_graph::{PrefillShape, TextShapes, prefill_shapes};
use crate::executor::forward::launch::StreamOrder;
use crate::ffi::{DType, Handle};
use crate::lane::Lanes;
use crate::model::LoadedModel;
use crate::model::components::RunnerKind;
use crate::model::interface::Interface;
use crate::model::registry::Components;
use crate::request::decode_state::DecodeState;
use crate::runner::causal_lm::{
    CausalLMRunner, Construction, Dimensions, RunnerParts, TableShape, methods, model_dtype,
};
use crate::runner::input_buffers::InputBuffers;
use crate::runner::scratch::Scratches;
use crate::storage::{KVCache, KVPlan, OutputPool};
use crate::{Error, Result};

use crate::parallel::{DwdpManager, ExpertUnion, PeerGroup};

/// The text call kinds a runner may serve.
pub(crate) const TEXT_KINDS: [CallKind; 3] = [
    CallKind::Forward(ForwardMode::Prefill),
    CallKind::Forward(ForwardMode::Decode),
    CallKind::Forward(ForwardMode::Verify),
];

/// Every bound of the text runners, derived from configuration and the KV
/// plan.
#[derive(Debug, Clone)]
pub(crate) struct TextBounds {
    /// Rows a batch may carry.
    pub(crate) max_rows: usize,
    /// Row capacity of the columns: batch rows or a prefill bucket.
    pub(crate) rows: usize,
    pub(crate) tokens: usize,
    pub(crate) tables: Vec<TableShape>,
    /// Pages per table every graph addresses.
    pub(crate) widths: Vec<i64>,
    pub(crate) decode: Vec<usize>,
    pub(crate) prefill: Vec<PrefillShape>,
}

impl TextBounds {
    /// Derive the bounds. `units` is the KV pool's unit count, sentinel
    /// included, once fitted; before fitting, graph shapes are not bounded
    /// by the pool.
    pub(crate) fn new(
        config: &WorkerConfig,
        request_slots: usize,
        cache: &KVPlan,
        units: Option<u32>,
        cuda: bool,
    ) -> Result<Self> {
        let mut max_rows = config.max_batch_calls.min(request_slots.max(1));
        let mut tokens = config.max_batch_tokens;
        for lane in &config.lanes {
            max_rows = max_rows.min(lane.max_batch_calls.unwrap_or(max_rows));
            tokens = tokens.min(lane.max_batch_tokens.unwrap_or(tokens));
        }
        let shapes = cache.group_shapes()?;
        let mut tables = Vec::new();
        let mut widths = Vec::new();
        for shape in &shapes {
            let mut width = shape.table_width(config.max_sequence_tokens as u64, tokens as u64);
            if let Some(units) = units {
                // A table cannot select more pages than the pool holds.
                width = width.min(
                    (u64::from(units.saturating_sub(1)) / u64::from(shape.units_per_page)).max(1),
                );
            }
            for _ in 0..shape.units_per_page {
                tables.push(TableShape {
                    page_tokens: shape.page_tokens,
                    windowed: shape.window.is_some(),
                });
                widths.push(width as i64);
            }
        }

        let graphs = cuda && config.graph_policy != "off";
        let row_units: usize = shapes
            .iter()
            .map(|shape| shape.units_per_page as usize)
            .sum();
        let decode: Vec<usize> = if graphs {
            config
                .decode_graph_batch_sizes
                .iter()
                .copied()
                .filter(|&rows| {
                    rows > 0
                        && rows <= max_rows
                        && units.is_none_or(|units| rows * row_units < units as usize)
                })
                .collect()
        } else {
            Vec::new()
        };
        let pages: Vec<(usize, usize)> = shapes
            .iter()
            .map(|shape| (shape.page_tokens as usize, shape.units_per_page as usize))
            .collect();
        let prefill = if graphs && config.prefill_cuda_graph {
            let pool_rows = units.map_or(max_rows, |units| {
                max_rows.min((units.saturating_sub(1) as usize) / row_units.max(1))
            });
            prefill_shapes(
                &config.prefill_graph_token_sizes,
                PREFILL_ROW_BUCKETS,
                pool_rows,
                tokens,
                &[(Some(true), false)],
                config.prefill_outputs,
                units.map(|units| (pages.as_slice(), units.saturating_sub(1) as usize)),
            )
        } else {
            Vec::new()
        };
        let rows = prefill
            .iter()
            .map(|shape| shape.row_bucket)
            .chain(decode.iter().copied())
            .fold(max_rows, usize::max);
        let tokens = decode.iter().copied().fold(tokens, usize::max);
        Ok(Self {
            max_rows,
            rows,
            tokens,
            tables,
            widths,
            decode,
            prefill,
        })
    }

    /// The column dimensions of a runner over `vocab` tokens.
    /// Pages a row selects from one block table, at most.
    fn width(&self) -> usize {
        self.widths.iter().copied().max().unwrap_or(1).max(1) as usize
    }

    pub(crate) fn dimensions(
        &self,
        vocab: usize,
        local_vocab: usize,
        hidden: usize,
        dtype: DType,
    ) -> Dimensions {
        Dimensions {
            rows: self.rows,
            tokens: self.tokens,
            tables: self.tables.len(),
            width: self.width(),
            local_vocab,
            vocab,
            hidden,
            embeddings: false,
            dtype,
        }
    }

    /// Calls a decode and a prefill batch may carry: the largest captured
    /// buckets (each prefill bucket keeps one padding sequence). Zero leaves
    /// a kind bounded by the batch call limit alone.
    pub(crate) fn graph_calls(&self) -> (u32, u32) {
        let prefill = self
            .prefill
            .iter()
            .map(|shape| shape.row_bucket - 1)
            .max()
            .map_or(0, |rows| rows.min(self.max_rows));
        let decode = self.decode.iter().copied().max().unwrap_or(0);
        (prefill as u32, decode as u32)
    }
}

/// Tokens one text forward sends to each expert layer at most: the batch
/// token bound or the largest decode graph bucket, whichever is larger.
/// The runners' bounds never exceed it, before or after the KV pool fits.
pub(crate) fn expert_tokens(config: &WorkerConfig) -> usize {
    let mut max_rows = config.max_batch_calls;
    let mut tokens = config.max_batch_tokens;
    for lane in &config.lanes {
        max_rows = max_rows.min(lane.max_batch_calls.unwrap_or(max_rows));
        tokens = tokens.min(lane.max_batch_tokens.unwrap_or(tokens));
    }
    let graphs = config.device.is_cuda() && config.graph_policy != "off";
    if !graphs {
        return tokens;
    }
    config
        .decode_graph_batch_sizes
        .iter()
        .copied()
        .filter(|&rows| rows > 0 && rows <= max_rows)
        .fold(tokens, usize::max)
}

/// The streams serving text calls on `device`, with the text kinds each
/// serves: every configured lane with a text kind, or the device stream.
fn streams(
    lanes: &Lanes,
    device: Device,
    kinds: &[CallKind],
) -> Result<Vec<(Option<Stream>, Vec<CallKind>)>> {
    let Some(index) = device.cuda_index() else {
        return Ok(vec![(None, kinds.to_vec())]);
    };
    let mut result: Vec<(Option<Stream>, Vec<CallKind>)> = Vec::new();
    for lane in lanes.lanes() {
        let served: Vec<CallKind> = kinds
            .iter()
            .copied()
            .filter(|kind| lane.serves(*kind))
            .collect();
        if !served.is_empty() {
            let stream = lane.stream.stream()?.clone();
            result.push((Some(stream), served));
        }
    }
    if lanes.lanes().is_empty() {
        let stream = lanes
            .stream_for(kinds[0], DeviceId(index))
            .ok_or(Error::State("text calls have no execution stream"))?;
        let stream = stream.stream()?.clone();
        result.push((Some(stream), kinds.to_vec()));
    }
    Ok(result)
}

/// One runner to construct: a causal LM on one stream.
struct Target {
    component: String,
    /// The causal LM's module path in the model.
    path: String,
    device: Device,
    stream: Option<Stream>,
    /// The text call kinds the runner's lane serves.
    served: Vec<CallKind>,
    /// The expert exchange of the runner's microbatch.
    exchange: Option<Handle>,
}

/// What text startup borrows from the rank.
pub(crate) struct Startup<'a> {
    pub(crate) interface: &'a Interface,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) request_slots: usize,
    pub(crate) queue_depth: usize,
    pub(crate) max_payload_bytes: usize,
    pub(crate) model: &'a LoadedModel,
    /// The model declares an image policy.
    pub(crate) image_processor: bool,
    pub(crate) lanes: &'a Lanes,
    pub(crate) kv: Option<&'a KVCache>,
    pub(crate) components: &'a Components,
    /// The shared scratch of each stream the runners' contexts borrow.
    pub(crate) scratches: &'a mut Scratches,
    pub(crate) graphs: &'a mut GraphStorage<String, Handle>,
    /// The device's startup stream for resets.
    pub(crate) preparation: Option<&'a Stream>,
    /// The expert union, when forwards are expert steps: each lane gets
    /// one runner per microbatch, on its stream and with its exchange.
    pub(crate) experts: Option<&'a mut ExpertUnion>,
    /// The worker's ranks and the mesh groups the model communicates over.
    pub(crate) peers: Option<&'a PeerGroup<'a>>,
    /// The distributed expert weights, with DWDP.
    pub(crate) weights: Option<&'a DwdpManager>,
}

impl CausalLMRunners {
    /// Create the runners of every text component on this rank, or `None`
    /// on a rank without one.
    pub(crate) fn start(startup: Startup<'_>) -> Result<Option<Self>> {
        let texts: Vec<(String, String, Device, Vec<CallKind>)> = startup
            .components
            .for_runner(RunnerKind::Text)
            .filter(|placed| !placed.host())
            .filter_map(|placed| {
                let path = placed
                    .calls
                    .iter()
                    .find(|call| {
                        call.runner() == Some(RunnerKind::Text) && call.method == "forward"
                    })?
                    .path
                    .clone();
                let kinds: Vec<CallKind> = TEXT_KINDS
                    .into_iter()
                    .filter(|kind| placed.kinds.contains(kind))
                    .collect();
                Some((placed.name.clone(), path, placed.device, kinds))
            })
            .collect();
        if texts.is_empty() {
            return Ok(None);
        }
        let kv = startup
            .kv
            .ok_or_else(|| Error::Invalid("a causal language model requires a KV cache".into()))?;
        let config = startup.config;
        let device = texts[0].2;
        let cuda = device.cuda_index();
        let bounds = TextBounds::new(
            config,
            startup.request_slots,
            kv.plan(),
            Some(kv.planes().num_units()),
            cuda.is_some(),
        )?;
        if let Some(experts) = startup.experts.as_ref()
            && bounds.tokens > experts.parallel().max_tokens()
        {
            return Err(Error::Invariant(format!(
                "text invocations of {} tokens exceed the expert exchange's {} tokens per rank",
                bounds.tokens,
                experts.parallel().max_tokens()
            )));
        }
        let vocab = methods::vocab_size(startup.interface, startup.model.handle())?;
        let dtype = model_dtype(&config.model_dtype)?;

        // The attention layers' view of the pool, bound once.
        let pool = kv.pool(startup.interface)?;

        let state = DecodeState::allocate(
            startup.request_slots,
            vocab,
            dtype,
            kv.storage().verified_lengths().clone(),
            cuda,
            startup.preparation,
        )?;

        // One runner per lane serving text kinds. With experts, each lane
        // has one runner per microbatch: the first on the lane's stream, the
        // others on the microbatch streams, each with its own exchange.
        let mut experts = startup.experts;
        let mut targets = Vec::new();
        let mut groups = Vec::new();
        for (component, path, device, kinds) in &texts {
            for (stream, served) in streams(startup.lanes, *device, kinds)? {
                let mut peers = Vec::new();
                let microbatches = experts.as_ref().map_or(1, |experts| experts.microbatches());
                for microbatch in 0..microbatches {
                    let (stream, exchange) = match experts.as_ref() {
                        Some(experts) => {
                            if let Some(lane) = &stream
                                && microbatch == 0
                                && lane.raw() != experts.stream(0).raw()
                            {
                                return Err(Error::Unsupported(
                                    "expert parallelism runs every forward on one execution lane"
                                        .into(),
                                ));
                            }
                            let peer = experts.stream(microbatch).clone();
                            (Some(peer), Some(experts.transport(microbatch).clone()))
                        }
                        None => (stream.clone(), None),
                    };
                    peers.push(targets.len());
                    targets.push(Target {
                        component: component.clone(),
                        path: path.clone(),
                        device: *device,
                        stream,
                        served: served.clone(),
                        exchange,
                    });
                }
                groups.push(peers);
            }
        }
        if experts.is_some() && groups.len() != 1 {
            return Err(Error::Unsupported(format!(
                "expert parallelism serves one text lane; {} lanes serve text",
                groups.len()
            )));
        }

        let table_layout: Vec<i64> = bounds
            .tables
            .iter()
            .map(|table| i64::from(table.page_tokens))
            .collect();
        let windowed: Vec<i64> = bounds
            .tables
            .iter()
            .map(|table| i64::from(table.windowed))
            .collect();
        let bind_units = startup.interface.function("cache.bind_units")?;
        let create_pool = startup.interface.function("graphs.pool")?;
        let mut runners = Vec::new();
        let mut columns = Vec::new();
        let mut routes = HashMap::new();
        let mut shared: HashMap<usize, usize> = HashMap::new();
        let mut collectives = Vec::new();
        let mut labels = Vec::new();
        for target in targets {
            let Target {
                component,
                path,
                device,
                stream,
                served,
                exchange,
            } = target;
            let ordinal = device.cuda_index();
            let handle = stream.as_ref().map_or(0, Stream::raw);
            // The model's collectives on the runner's stream, which its
            // execution context routes its communicators to.
            let bound = match startup.peers {
                Some(peers) => {
                    let owner = stream
                        .as_ref()
                        .and_then(|stream| startup.lanes.owner(stream.raw()));
                    peers.stream_collectives(&component, device, owner)?
                }
                None => None,
            };
            let slot_tables = kv.storage().views();
            let slot_table = |name: &str| {
                slot_tables
                    .iter()
                    .find(|(table, _)| table == name)
                    .map(|(_, view)| view)
                    .ok_or_else(|| Error::Invariant(format!("block tables have no {name}")))
            };
            let methods = methods::construct(
                startup.interface,
                &Construction {
                    loaded: startup.model.handle(),
                    path: &path,
                    device: ordinal,
                    stream: handle,
                    scratch: startup.scratches.get(startup.interface, handle, ordinal)?,
                    cache: &pool,
                    experts: exchange,
                    weights: startup.weights.map(|weights| weights.prefetch().clone()),
                    collectives: bound
                        .as_ref()
                        .map(|bound| bound.natives(startup.interface).and_then(Handle::new))
                        .transpose()?,
                    rows: bounds.rows,
                    tokens: bounds.tokens,
                    width: bounds.width(),
                    page_tokens: &table_layout,
                    windowed: &windowed,
                    state: [
                        slot_table("unit_tables")?,
                        slot_table("start_pages")?,
                        slot_table("table_shapes")?,
                        state.verified_lengths(),
                        state.future_input_tokens(),
                        state.logical_lengths(),
                        state.predicates(),
                        state.sampling_positions(),
                        state.penalty_counts(),
                        state.prompt_logits(),
                    ],
                },
            )?;

            // The runner's vocabulary shard sizes its logits columns.
            let functions = methods::Functions::resolve(startup.interface, &methods)?;
            let shard = methods::vocab(startup.interface, &methods)?;
            let hidden = methods::hidden_size(startup.interface, startup.model.handle())?;
            let plans_attention = methods::plans_attention(startup.interface, &methods)?;
            if shard.size != vocab {
                return Err(Error::Interface(format!(
                    "the text runner's vocabulary ({}) differs from the model's ({vocab})",
                    shard.size
                )));
            }
            let mut dimensions = bounds.dimensions(vocab, shard.width(), hidden, dtype);
            // Prompts of a model with an image processor place vision
            // features in place of token embeddings.
            dimensions.embeddings = startup.image_processor;
            let buffers = InputBuffers::allocate(dimensions, ordinal, startup.preparation)?;

            let pool = match (
                ordinal,
                bounds.decode.is_empty() && bounds.prefill.is_empty(),
            ) {
                (Some(ordinal), false) => {
                    let key = buffers.column("logits")?.address() as usize;
                    let share = shared.get(&handle).copied();
                    let pools = reserve_pools(
                        startup.graphs,
                        &create_pool,
                        key,
                        format!("text:{component}"),
                        &[ordinal],
                        share,
                    )?;
                    shared.entry(handle).or_insert(key);
                    pools.first().map(|pool| pool.value.clone())
                }
                _ => None,
            };
            columns.push(kv.bind_units(&bind_units, stream.as_ref())?);
            let group = groups
                .iter()
                .position(|peers| peers.contains(&runners.len()))
                .ok_or_else(|| Error::Invariant("a text runner belongs to no group".into()))?;
            if groups[group][0] == runners.len() {
                for &kind in &served {
                    if routes.insert((component.clone(), kind), group).is_some() {
                        return Err(Error::Invalid(format!(
                            "computation {component}.{} has multiple lane bindings",
                            kind.as_str()
                        )));
                    }
                }
            }
            collectives.extend(bound);
            // The runner's kernel-table label: its component and module path.
            let label = if path.is_empty() {
                component.clone()
            } else {
                format!("{component}.{path}")
            };
            labels.push((label, device.to_string()));
            runners.push(CausalLMRunner::new(RunnerParts {
                component,
                device: ordinal,
                stream,
                methods,
                functions,
                plans_attention,
                breaks: startup.weights.map(|weights| weights.breaks().clone()),
                buffers,
                tables: bounds.tables.clone(),
                widths: bounds.widths.clone(),
                shapes: TextShapes::new(bounds.decode.clone(), bounds.prefill.clone()),
                pool,
            }));
        }
        // Idle ranks join expert steps through the runners' contexts, one
        // per microbatch.
        if let Some(experts) = experts.as_deref_mut() {
            let owners: Vec<&Handle> = groups[0]
                .iter()
                .map(|&runner| runners[runner].methods.runner())
                .collect();
            experts.bind_joins(&owners)?;
        }

        // Completion fields plus payload words per call, as the engine
        // bounds each call's completion bytes.
        let max_calls = config.max_batch_calls.max(1);
        let words = max_calls * (4 + startup.max_payload_bytes.div_ceil(4));
        let outputs = OutputPool::new(cuda, startup.queue_depth.max(1) * max_calls, words);
        if let Some(stream) = startup.preparation {
            stream.synchronize()?;
        }
        if let Some(experts) = experts {
            let pools: Vec<Handle> = groups[0]
                .iter()
                .filter_map(|&runner| runners[runner].pool().cloned())
                .collect();
            if !pools.is_empty() {
                experts.set_pools(pools);
            }
        }
        Ok(Some(Self {
            runners,
            labels,
            groups,
            collectives,
            columns,
            routes,
            state,
            outputs,
            vocab,
            order: StreamOrder::default(),
        }))
    }
}

/// Device bytes the causal LM runners allocate beside the KV pool.
pub(crate) fn reserved_bytes(
    interface: &Interface,
    model: &LoadedModel,
    config: &WorkerConfig,
    request_slots: usize,
    cache: &KVPlan,
    image_processor: bool,
    streams: usize,
) -> Result<u64> {
    // A rank without a causal language model reports no vocabulary.
    let vocab = methods::vocab_size(interface, model.handle())?;
    if vocab == 0 {
        return Ok(0);
    }
    let hidden = methods::hidden_size(interface, model.handle())?;
    let cuda = config.device.is_cuda();
    let bounds = TextBounds::new(config, request_slots, cache, None, cuda)?;
    let dtype = model_dtype(&config.model_dtype)?;
    let mut dimensions = bounds.dimensions(vocab, vocab, hidden, dtype);
    dimensions.embeddings = image_processor;
    Ok(InputBuffers::bytes(&dimensions) * streams.max(1) as u64
        + DecodeState::bytes(request_slots, vocab, dtype))
}
