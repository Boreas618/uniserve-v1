//! Startup of the text runtime: bounds, bindings and owned storage.
//!
//! [`TextBounds`] derives every text bound from the worker configuration and
//! the KV plan, once for capacity planning (bytes to reserve beside the KV
//! pool, graph-bounded call limits) and again for allocation, so both agree.
//! [`TextRuntime::start`] then creates one runner per execution stream
//! serving text call kinds: it prepares and binds the causal LM's component
//! (operators over Rust workspaces and the K/V pool), allocates the runner's
//! columns and graph pool, and binds the text steps over them.

use std::collections::HashMap;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{CallKind, ForwardMode};

use super::TextRuntime;
use crate::config::{Device, PREFILL_ROW_BUCKETS, WorkerConfig};
use crate::executor::forward::launch::StreamOrder;
use crate::ffi::{DType, Handle};
use crate::model::bind::{self, Backends, BindingRequest, StreamWorkspaces, TextSize};
use crate::model::components::RunnerKind;
use crate::model::interface::Interface;
use crate::model::registry::Components;
use crate::model::text::{self, TextModel};
use crate::model::{LoadedModel, ModelFacts};
use crate::request::decode_state::DecodeState;
use crate::runner::graphs::reserve_pools;
use crate::runner::text::{
    Dimensions, RunnerParts, TableShape, TextBuffers, TextRunner, model_dtype,
};
use crate::runner::{GraphStorage, PrefillShape, TextShapes, prefill_shapes};
use crate::storage::{KVCache, KVPlan, OutputPool};
use crate::worker::Lanes;
use crate::{Error, Result};

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
            width: self.widths.iter().copied().max().unwrap_or(1).max(1) as usize,
            local_vocab,
            vocab,
            hidden,
            hidden_output: self.prefill.iter().any(|shape| shape.outputs),
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
            .filter(|kind| lane.config.call_kinds.contains(kind))
            .collect();
        if !served.is_empty() {
            // SAFETY: the lanes own the stream and close after the runtime.
            let stream = unsafe { Stream::borrowed(DeviceId(index), lane.stream.handle()?) }?;
            result.push((Some(stream), served));
        }
    }
    if lanes.lanes().is_empty() {
        let stream = lanes
            .stream_for(kinds[0], DeviceId(index))
            .ok_or(Error::State("text calls have no execution stream"))?;
        // SAFETY: as above.
        let stream = unsafe { Stream::borrowed(DeviceId(index), stream.handle()?) }?;
        result.push((Some(stream), kinds.to_vec()));
    }
    Ok(result)
}

/// What text startup borrows from the rank.
pub(crate) struct Startup<'a> {
    pub(crate) interface: &'a Interface,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) request_slots: usize,
    pub(crate) queue_depth: usize,
    pub(crate) max_payload_bytes: usize,
    pub(crate) model: &'a LoadedModel,
    pub(crate) facts: &'a ModelFacts,
    pub(crate) lanes: &'a Lanes,
    pub(crate) kv: Option<&'a KVCache>,
    pub(crate) components: &'a mut Components,
    pub(crate) workspaces: &'a mut StreamWorkspaces,
    pub(crate) graphs: &'a mut GraphStorage<String, Handle>,
    /// The device's startup stream for resets.
    pub(crate) preparation: Option<&'a Stream>,
}

impl TextRuntime {
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
        let vocab = startup.facts.vocab_size.ok_or_else(|| {
            Error::Interface("a causal language model reports no vocabulary".into())
        })?;
        let dtype = model_dtype(&config.model_dtype)?;

        // The attention layers' view of the pool, bound once.
        let views = crate::ffi::Fields::new();
        let views = kv
            .views()?
            .into_iter()
            .try_fold(views, |fields, (name, view)| -> Result<_> {
                Ok(fields.with(&name, view.to_tensor()?))
            })?
            .into_any();
        let pool = Handle::new(
            startup
                .interface
                .function("cache.pool")?
                .call_values(&[tvm_ffi::Any::from(kv.layout()?), views])?,
        );

        let state = DecodeState::allocate(
            startup.request_slots,
            vocab,
            dtype,
            kv.storage().verified_lengths().clone(),
            cuda,
            startup.preparation,
        )?;

        // Prepare every binding before binding any, so shared workspaces
        // hold the largest of them.
        let mut prepared = Vec::new();
        let mut targets = Vec::new();
        for (component, path, device, kinds) in &texts {
            for (stream, served) in streams(startup.lanes, *device, kinds)? {
                let handle = stream.as_ref().map_or(0, Stream::raw);
                prepared.push(bind::prepare(
                    startup.interface,
                    startup.model.handle(),
                    BindingRequest {
                        component: component.clone(),
                        path: path.clone(),
                        device: *device,
                        stream: handle,
                        size: Some(TextSize {
                            tokens: bounds.tokens,
                            rows: bounds.rows,
                        }),
                        // The loaded model resolved its attention backend.
                        backends: Backends::default(),
                        derive_host_lengths: false,
                        experts: None,
                        weights: None,
                        cache: Some(pool.clone()),
                    },
                )?);
                targets.push((component.clone(), *device, stream, served));
            }
        }
        let bindings = bind::bind(startup.interface, startup.workspaces, prepared)?;

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
        for ((component, device, stream, served), binding) in targets.into_iter().zip(&bindings) {
            let ordinal = device.cuda_index();
            // The binding's vocabulary shard sizes its logits columns.
            let (shard, hidden) = text::describe(startup.interface, binding.prepared())?;
            if shard.size != vocab {
                return Err(Error::Interface(format!(
                    "the text binding's vocabulary ({}) differs from the model's ({vocab})",
                    shard.size
                )));
            }
            let mut dimensions = bounds.dimensions(vocab, shard.width(), hidden, dtype);
            // Prompts of a model with an image processor place vision
            // features in place of token embeddings.
            dimensions.embeddings = startup.facts.image_processor;
            let buffers = TextBuffers::allocate(dimensions, ordinal, startup.preparation)?;
            let mut views = buffers.views();
            views.extend(
                kv.storage()
                    .views()
                    .into_iter()
                    .filter(|(name, _)| name != "verified_lengths" && name != "alloced_lens")
                    .map(|(name, view)| (format!("slot.{name}"), view)),
            );
            views.extend(state.views());
            let handle = stream.as_ref().map_or(0, Stream::raw);
            let model = TextModel::bind(
                startup.interface,
                binding.prepared(),
                handle,
                &views,
                &table_layout,
                &windowed,
                shard,
                hidden,
            )?;

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
            for &kind in &served {
                if routes
                    .insert((component.clone(), kind), runners.len())
                    .is_some()
                {
                    return Err(Error::Invalid(format!(
                        "computation {component}.{} has multiple lane bindings",
                        kind.as_str()
                    )));
                }
            }
            runners.push(TextRunner::new(RunnerParts {
                component,
                device: ordinal,
                stream,
                model,
                buffers,
                tables: bounds.tables.clone(),
                widths: bounds.widths.clone(),
                shapes: TextShapes::new(bounds.decode.clone(), bounds.prefill.clone()),
                pool,
            }));
        }
        startup.components.add_bindings(bindings);

        // Completion fields plus payload words per call, as the engine
        // bounds each call's completion bytes.
        let max_calls = config.max_batch_calls.max(1);
        let words = max_calls * (4 + startup.max_payload_bytes.div_ceil(4));
        let outputs = OutputPool::new(cuda, startup.queue_depth.max(1) * max_calls, words);
        if let Some(stream) = startup.preparation {
            stream.synchronize()?;
        }
        Ok(Some(Self {
            runners,
            columns,
            routes,
            state,
            outputs,
            vocab,
            order: StreamOrder::default(),
        }))
    }
}

/// Device bytes the text runtime allocates beside the KV pool.
pub(crate) fn reserved_bytes(
    config: &WorkerConfig,
    request_slots: usize,
    cache: &KVPlan,
    facts: &ModelFacts,
    streams: usize,
) -> Result<u64> {
    let Some(vocab) = facts.vocab_size else {
        return Ok(0);
    };
    let cuda = config.primary_device().is_cuda();
    let bounds = TextBounds::new(config, request_slots, cache, None, cuda)?;
    let dtype = model_dtype(&config.model_dtype)?;
    let mut dimensions = bounds.dimensions(vocab, vocab, facts.hidden_size.unwrap_or(0), dtype);
    dimensions.embeddings = facts.image_processor;
    Ok(
        dimensions.bytes() * streams.max(1) as u64
            + DecodeState::bytes(request_slots, vocab, dtype),
    )
}
