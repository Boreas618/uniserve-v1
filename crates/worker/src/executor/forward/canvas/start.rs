//! Startup of the canvas runtime: bounds, banks, bindings and runners.
//!
//! [`CanvasBounds`] derives every canvas bound from the worker configuration,
//! the KV plan and the denoiser's facts, once for capacity planning (bytes
//! to reserve beside the KV pool) and again for allocation, so both agree.
//! [`CanvasRuntime::start`] then creates one runner per execution stream
//! serving token-denoising calls: it prepares and binds the token
//! denoiser's component (operators over Rust workspaces and the K/V pool),
//! allocates the runner's buffers and graph pool, and binds the canvas
//! functions over them. A generating rank also allocates the resident
//! banks every runner shares.

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_core::CanvasSampling;
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{CallKind, ForwardMode};

use super::CanvasRuntime;
use crate::config::{Device, WorkerConfig};
use crate::executor::forward::launch::StreamOrder;
use crate::ffi::Handle;
use crate::model::LoadedModel;
use crate::model::bind::{self, Backends, BindingRequest, StreamWorkspaces, TextSize};
use crate::model::canvas::{CanvasFacts, CanvasModel, StepOptions, TableLayout, scratch_bytes};
use crate::model::components::RunnerKind;
use crate::model::interface::Interface;
use crate::model::registry::Components;
use crate::runner::GraphStorage;
use crate::runner::canvas::{
    CANDIDATES, CanvasLane, CanvasRunner, CanvasShapes, SLOT_BUCKETS, step_rows,
};
use crate::runner::graphs::reserve_pools;
use crate::storage::canvas::{BankLayout, CanvasBanks, CanvasBuffers, CanvasLayout, StepLayout};
use crate::storage::{GroupShape, KVCache, KVPlan, OutputPool};
use crate::worker::Lanes;
use crate::{Error, Result};

/// The call kind every canvas runner serves.
pub(crate) const CANVAS_KIND: CallKind = CallKind::Forward(ForwardMode::TokenDenoising);

/// Every bound of the canvas runners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanvasBounds {
    /// Canvases a call may carry: the graph buckets' limit.
    pub(crate) capacity: usize,
    /// Packed sequences of one invocation: a readout bucket's padding
    /// doubles its canvases.
    pub(crate) rows: usize,
    pub(crate) tokens: usize,
    pub(crate) groups: Vec<GroupShape>,
    pub(crate) tables: Vec<TableLayout>,
    /// Page columns every table's rows hold.
    pub(crate) widths: Vec<usize>,
    /// Startup captures graphs.
    pub(crate) graphs: bool,
    /// Step rows and sampler chunk of a generating rank.
    pub(crate) step: Option<(usize, usize)>,
}

impl CanvasBounds {
    /// Derive the bounds. `units` is the KV pool's unit count, sentinel
    /// included, once fitted; before fitting, the pool bounds nothing.
    pub(crate) fn new(
        config: &WorkerConfig,
        request_slots: usize,
        cache: &KVPlan,
        units: Option<u32>,
        facts: &CanvasFacts,
        cuda: bool,
    ) -> Result<Self> {
        let mut max_rows = config.max_batch_calls.min(request_slots.max(1));
        let mut tokens = config.max_batch_tokens;
        for lane in &config.lanes {
            max_rows = max_rows.min(lane.max_batch_calls.unwrap_or(max_rows));
            tokens = tokens.min(lane.max_batch_tokens.unwrap_or(tokens));
        }
        let length = facts.canvas_length;
        let groups = cache.group_shapes()?;
        let row_units: usize = groups
            .iter()
            .map(|group| group.units_per_page as usize)
            .sum();
        // Startup reads a one-page prefix per synthetic canvas, so the pool
        // bounds the canvases a captured bucket may hold.
        let pool_rows = units.map_or(usize::MAX, |units| {
            (units.saturating_sub(1) as usize) / row_units.max(1)
        });
        let graphs = cuda && config.graph_policy != "off";
        let capacity = max_rows.min(pool_rows).min(tokens / length);
        if capacity == 0 {
            return Err(Error::Unsupported(format!(
                "no canvas of {length} tokens fits a batch of {tokens} tokens and a pool of {pool_rows} canvases"
            )));
        }
        let rows = if graphs {
            (2 * capacity).max(max_rows)
        } else {
            max_rows
        };

        let mut tables = Vec::new();
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
                tables.push(TableLayout {
                    page_tokens: group.page_tokens,
                    windowed: group.window.is_some(),
                });
                widths.push(width as usize);
            }
        }

        let step = (facts.generates && config.canvas_sampling.is_some()).then(|| {
            let rows = if graphs {
                capacity
            } else {
                max_rows.min(tokens / length)
            }
            .min(request_slots);
            (rows, step_rows(length, facts.vocab_size, rows))
        });
        Ok(Self {
            capacity,
            rows,
            tokens,
            groups,
            tables,
            widths,
            graphs,
            step,
        })
    }

    /// The banks of a generating rank serving `sampling`.
    pub(crate) fn banks(
        &self,
        facts: &CanvasFacts,
        request_slots: usize,
        sampling: &CanvasSampling,
    ) -> Option<BankLayout> {
        self.step.map(|_| BankLayout {
            slots: request_slots,
            length: facts.canvas_length,
            depth: sampling.stability_threshold as usize,
            hidden: facts.hidden_size,
            dtype: facts.dtype,
        })
    }

    /// One lane's buffers. `scratch` is the sampler chunk's product scratch.
    pub(crate) fn layout(
        &self,
        facts: &CanvasFacts,
        banks: Option<BankLayout>,
        scratch: usize,
    ) -> CanvasLayout {
        CanvasLayout {
            rows: self.rows,
            tokens: self.tokens,
            tables: self.tables.len(),
            width: self.widths.iter().copied().max().unwrap_or(1).max(1),
            tail_rows: SLOT_BUCKETS[SLOT_BUCKETS.len() - 1],
            candidates: CANDIDATES,
            attend_widths: facts.attend_widths.clone(),
            attend_dtype: facts.attend_dtype,
            step: self
                .step
                .zip(banks)
                .map(|((rows, chunk_rows), banks)| StepLayout {
                    rows,
                    banks,
                    vocab: facts.vocab_size,
                    chunk_rows,
                    scratch,
                }),
        }
    }
}

/// The streams serving canvas calls on `device`: every configured lane
/// serving token denoising, or the device stream.
fn streams(lanes: &Lanes, device: Device) -> Result<Vec<Option<Stream>>> {
    let Some(index) = device.cuda_index() else {
        return Ok(vec![None]);
    };
    let mut result = Vec::new();
    for lane in lanes.lanes() {
        if lane.config.call_kinds.contains(&CANVAS_KIND) {
            // SAFETY: the lanes own the stream and close after the runtime.
            result.push(Some(unsafe {
                Stream::borrowed(DeviceId(index), lane.stream.handle()?)
            }?));
        }
    }
    if lanes.lanes().is_empty() {
        let stream = lanes
            .stream_for(CANVAS_KIND, DeviceId(index))
            .ok_or(Error::State("canvas calls have no execution stream"))?;
        // SAFETY: as above.
        result.push(Some(unsafe {
            Stream::borrowed(DeviceId(index), stream.handle()?)
        }?));
    }
    Ok(result)
}

/// What canvas startup borrows from the rank.
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
    pub(crate) workspaces: &'a mut StreamWorkspaces,
    pub(crate) graphs: &'a mut GraphStorage<String, Handle>,
    /// The device's startup stream, which zeroes new storage.
    pub(crate) preparation: Option<&'a Stream>,
}

/// The token denoiser components placed on this rank: name, module path
/// and device.
fn denoisers(components: &Components) -> Vec<(String, String, Device)> {
    components
        .for_runner(RunnerKind::Canvas)
        .filter(|placed| !placed.host() && placed.kinds.contains(&CANVAS_KIND))
        .filter_map(|placed| {
            let path = placed
                .calls
                .iter()
                .find(|call| call.runner() == Some(RunnerKind::Canvas) && call.method == "forward")?
                .path
                .clone();
            Some((placed.name.clone(), path, placed.device))
        })
        .collect()
}

impl CanvasRuntime {
    /// Create the runners of every token denoiser on this rank, or `None`
    /// on a rank without one.
    pub(crate) fn start(startup: Startup<'_>) -> Result<Option<Self>> {
        let denoisers = denoisers(startup.components);
        let Some((_, path, device)) = denoisers.first().cloned() else {
            return Ok(None);
        };
        let kv = startup
            .kv
            .ok_or_else(|| Error::Invalid("a token denoiser reads a KV cache".into()))?;
        let config = startup.config;
        let cuda = device.cuda_index();
        let facts = Arc::new(CanvasFacts::describe(
            startup.interface,
            startup.model.handle(),
            &path,
            startup.experts.0,
            startup.experts.1,
        )?);
        let bounds = CanvasBounds::new(
            config,
            startup.request_slots,
            kv.plan(),
            Some(kv.planes().num_units()),
            &facts,
            cuda.is_some(),
        )?;

        // The served sampling must generate the model's canvases.
        let sampling = match (&config.canvas_sampling, bounds.step) {
            (Some(sampling), Some(_)) => {
                if sampling.canvas_length as usize != facts.canvas_length {
                    return Err(Error::Invalid(format!(
                        "the served canvas length {} is not the model's canvas length {}",
                        sampling.canvas_length, facts.canvas_length
                    )));
                }
                Some(*sampling)
            }
            _ => None,
        };
        let bank_layout = sampling
            .as_ref()
            .and_then(|sampling| bounds.banks(&facts, startup.request_slots, sampling));
        let banks = bank_layout
            .map(|layout| CanvasBanks::allocate(layout, cuda, startup.preparation).map(Arc::new))
            .transpose()?;
        let scratch = match bounds.step {
            Some((_, chunk_rows)) => scratch_bytes(
                startup.interface,
                chunk_rows * facts.canvas_length,
                facts.hidden_size,
                cuda,
            )?,
            None => 0,
        };
        let layout = bounds.layout(&facts, bank_layout, scratch);

        // The attention layers' view of the pool, bound once.
        let views = kv
            .views()?
            .into_iter()
            .try_fold(
                crate::ffi::Fields::new(),
                |fields, (name, view)| -> Result<_> { Ok(fields.with(&name, view.to_tensor()?)) },
            )?
            .into_any();
        let pool = Handle::new(
            startup
                .interface
                .function("cache.pool")?
                .call_values(&[tvm_ffi::Any::from(kv.layout()?), views])?,
        );

        // Prepare every binding before binding any, so shared workspaces
        // hold the largest of them.
        let mut prepared = Vec::new();
        let mut targets = Vec::new();
        for (component, path, device) in &denoisers {
            for stream in streams(startup.lanes, *device)? {
                prepared.push(bind::prepare(
                    startup.interface,
                    startup.model.handle(),
                    BindingRequest {
                        component: component.clone(),
                        path: path.clone(),
                        device: *device,
                        stream: stream.as_ref().map_or(0, Stream::raw),
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
                targets.push((component.clone(), *device, stream));
            }
        }
        let bindings = bind::bind(startup.interface, startup.workspaces, prepared)?;

        let create_pool = startup.interface.function("graphs.pool")?;
        let mut runners = Vec::new();
        let mut routes = HashMap::new();
        for ((component, device, stream), binding) in targets.into_iter().zip(&bindings) {
            let ordinal = device.cuda_index();
            let buffers = CanvasBuffers::allocate(layout.clone(), ordinal, startup.preparation)?;
            let mut views = buffers.views().to_vec();
            if let Some(banks) = &banks {
                views.extend(banks.views().iter().cloned());
            }
            let model = CanvasModel::bind(
                startup.interface,
                binding.prepared(),
                &views,
                &bounds.tables,
                ordinal,
                sampling.map(|sampling| StepOptions {
                    sampling,
                    step_rows: bounds.step.map_or(1, |(_, chunk_rows)| chunk_rows),
                }),
            )?;
            let owner = format!("canvas:{component}");
            let pool = match (ordinal, bounds.graphs) {
                (Some(ordinal), true) => {
                    let key = buffers.view("input_ids")?.address() as usize;
                    reserve_pools(
                        startup.graphs,
                        &create_pool,
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
            if routes.insert(component.clone(), runners.len()).is_some() {
                return Err(Error::Invalid(format!(
                    "computation {component}.{} has multiple lane bindings",
                    CANVAS_KIND.as_str()
                )));
            }
            runners.push(CanvasRunner::new(CanvasLane {
                owner,
                model,
                facts: Arc::clone(&facts),
                buffers,
                banks: banks.clone(),
                sampling,
                shapes: CanvasShapes::new(
                    facts.canvas_length,
                    bounds.capacity,
                    sampling.map(|_| startup.request_slots),
                )?,
                graph_widths: bounds.widths.clone(),
                groups: bounds.groups.clone(),
                pool,
                stream,
            }));
        }
        startup.components.add_bindings(bindings);

        // One lease per batch in flight. A lease holds every step's outcome
        // and canvas, or the candidate values of readouts, whose four bytes
        // each the batch's result message bounds.
        let max_calls = config.max_batch_calls.max(1);
        let words =
            (max_calls * (1 + facts.canvas_length)).max(startup.max_payload_bytes.div_ceil(4));
        let outputs = OutputPool::new(cuda, startup.queue_depth.max(1) + 1, words);
        if let Some(stream) = startup.preparation {
            stream.synchronize()?;
        }
        Ok(Some(Self {
            runners,
            routes,
            sampling,
            outputs,
            order: StreamOrder::default(),
        }))
    }
}

/// Device bytes the canvas runtime allocates beside the KV pool for the
/// token denoisers of `components`: every lane's buffers and a generating
/// rank's banks. `experts` is as in [`Startup`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn reserved_bytes(
    interface: &Interface,
    model: &LoadedModel,
    components: &Components,
    experts: (bool, usize),
    config: &WorkerConfig,
    request_slots: usize,
    cache: &KVPlan,
    lanes: usize,
) -> Result<u64> {
    let Some((_, path, _)) = denoisers(components).into_iter().next() else {
        return Ok(0);
    };
    let facts = CanvasFacts::describe(interface, model.handle(), &path, experts.0, experts.1)?;
    let facts = &facts;
    let cuda = config.primary_device().cuda_index();
    let bounds = CanvasBounds::new(config, request_slots, cache, None, facts, cuda.is_some())?;
    let banks = config
        .canvas_sampling
        .as_ref()
        .and_then(|sampling| bounds.banks(facts, request_slots, sampling));
    let scratch = match bounds.step {
        Some((_, chunk_rows)) => scratch_bytes(
            interface,
            chunk_rows * facts.canvas_length,
            facts.hidden_size,
            cuda,
        )?,
        None => 0,
    };
    let lane = bounds.layout(facts, banks, scratch).bytes();
    let resident = banks.map(|banks| banks.bytes()).transpose()?.unwrap_or(0);
    Ok(lane * lanes.max(1) as u64 + resident)
}
