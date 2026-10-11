//! KV-conditioned image denoising (`ImageDenoiser` models: BAGEL,
//! SenseNova-U1).
//!
//! An [`ImageDiffusion`] runner serves one bound image denoiser. A denoising
//! step evaluates, for every image of the batch, one row per guidance
//! branch: the image's framed latent sequence attending (non-causally, read
//! only) to the KV prefix its branch conditions on. Rust owns the rows'
//! geometry: positions ([`ImageDenoiserLayout::positions`]), timesteps,
//! attention pages and the graph choice; the Python half
//! (`uniserve_models.runner.denoiser.DenoiserRunner`) gathers the rows'
//! attention metadata on the device, evaluates the network into its
//! predictions (`denoise`) and, per image, combines its branch predictions
//! with the trajectory's guidance and advances its samples in place
//! (`step`).
//!
//! Input tensors of a forward: `positions [3, T]` int64, `timesteps [R, 1]`
//! float32 and the packed row columns of [`AttentionColumns`], where `R`
//! rows of at most `T` tokens predict at most `P` latent rows; `noise`
//! holds one image's drawn initial noise. Samples are the latent pool's
//! step workspace, into which the caller gathers each trajectory's
//! committed pages.
//!
//! Graphs: each captured `denoise` evaluates a fixed set of rows of one
//! image size at the configured attention widths ([`DiffusionShape`]): the
//! rows' sample offsets fix where it reads latents, and rows pack their
//! tokens and predictions in order. Startup captures the configured shapes
//! largest first; serving launches an exact shape natively (after
//! `plan_denoise` when an attention operator plans its launch on the host)
//! and runs any other on the Python thread. Solver updates run on the
//! Python thread.

use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::ImageParams;
use uniserve_cuda::Stream;

use crate::config::Device;
use crate::cuda_graph::GraphPool;
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::ffi::{DType, Function, Handle, Region, integer_list, list};
use crate::model::bind::ComponentBinding;
use crate::model::image::ImageDenoiserLayout;
use crate::model::interface::Interface;
use crate::parallel::joins::GraphFunctions;
use crate::request::diffusion::{BranchSet, GuidancePlan};
use crate::runner::attention::{AttentionCapacity, AttentionColumns, AttentionGeometry};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{AttentionRow, BlockTables, GroupTable};
use crate::{Error, Result};

/// The geometry of a captured denoising step: its rows of one image size,
/// by the first sample row each reads (rows of one image share it).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiffusionShape {
    pub height: u32,
    pub width: u32,
    pub samples: Vec<u32>,
}

impl DiffusionShape {
    /// The shape `rows` evaluate, when they share one image size.
    pub fn of(rows: &[DenoiseRow]) -> Option<Self> {
        let first = rows.first()?;
        rows.iter()
            .all(|row| (row.height, row.width) == (first.height, first.width))
            .then(|| Self {
                height: first.height,
                width: first.width,
                samples: rows.iter().map(|row| row.sample).collect(),
            })
    }
}

/// One denoising row: a guidance branch of one image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseRow {
    /// The image's first latent row in the samples and its row count.
    pub sample: u32,
    pub units: u32,
    pub height: u32,
    pub width: u32,
    /// The request slot holding the branch's KV prefix and its visible
    /// tokens.
    pub slot: u32,
    pub prefix: u32,
    /// The temporal position of the image sequence.
    pub temporal: u64,
    /// The model timestep of the step.
    pub timestep: f32,
}

/// A latent image inserted as conditioning: its K/V is written after the
/// slot's visible tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionRow {
    /// The latent feature product (`[units, width]`).
    pub features: Region,
    pub height: u32,
    pub width: u32,
    pub slot: u32,
    pub prefix: u32,
    /// The language position the image's opening framing token takes.
    pub position: u64,
}

/// One image's solver update: its samples, slot, step and branch
/// predictions.
#[derive(Clone, Debug)]
pub struct IntegrateRow {
    pub sample: u32,
    pub units: u32,
    /// The request slot whose timestep the solver reads.
    pub slot: u32,
    pub step: u32,
    /// The first prediction row of the image's branches, which follow
    /// each other in evaluation order.
    pub prediction: u32,
    pub branches: BranchSet,
    pub plan: Handle,
}

/// The rows' predictions of one evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Predicted {
    pub mode: GraphMode,
    /// Each row's first prediction row.
    pub offsets: Vec<u32>,
}

/// Capacity of a runner's columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffusionCapacity {
    /// Rows of one evaluation.
    pub rows: usize,
    /// Tokens of one evaluation (framed sequences).
    pub tokens: usize,
    /// Predicted latent rows of one evaluation.
    pub units: usize,
    /// The attention tables and their widest page width.
    pub tables: usize,
    pub width: usize,
    /// Each table's page width in captured graphs.
    pub graph_widths: Vec<usize>,
}

/// The resident tensors a runner is constructed over.
pub struct DiffusionStorage {
    /// The latent pool's step workspace `[units, width]`, on the
    /// denoiser's device.
    pub samples: StorageView,
    /// The latent pool's per-slot timesteps `[slots + 1, 1]` float32.
    pub slot_timesteps: StorageView,
    /// The denoiser device's product arena (`uint8`).
    pub products: StorageView,
    /// The KV cache's slot tables (`unit_tables`, `start_pages`,
    /// `table_shapes`), and each numerical block table's page tokens and
    /// whether it is windowed.
    pub slot_tables: Vec<(String, StorageView)>,
    pub page_tokens: Vec<i64>,
    pub windowed: Vec<i64>,
}

const RUNNER: &str = "uniserve_models.runner.denoiser.DenoiserRunner";

/// The Python half and its methods, each called with the runner first.
struct Methods {
    runner: Handle,
    plan: Function,
    draw_noise: Function,
    prepare_latents: Function,
    denoise: Function,
    plan_denoise: Function,
    encode_conditions: Function,
    step: Function,
    /// Whether an attention operator plans each launch on the host.
    plans_attention: bool,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
}

/// The runner of one bound image denoiser.
pub struct ImageDiffusion {
    owner: String,
    layout: ImageDenoiserLayout,
    device: Device,
    stream: Option<Stream>,
    methods: Methods,
    capacity: DiffusionCapacity,
    attention: AttentionColumns,
    positions: StorageView,
    timesteps: StorageView,
    /// One image's initial noise, `[P, width]` in the samples' dtype: the
    /// library's seeded draw fills it and `prepare_latents` reads it.
    noise: StorageView,
    graphs: GraphTable<DiffusionShape>,
    capturing: bool,
}

fn allocate(
    device: Device,
    dtype: DType,
    shape: &[i64],
    stream: Option<&Stream>,
) -> Result<StorageView> {
    let bytes = shape.iter().product::<i64>().max(1) as usize * dtype.bytes();
    let memory: Arc<dyn crate::ffi::Backing> = match device {
        Device::Cuda(index) => Memory::device(index, bytes)?,
        Device::Cpu => Memory::host(bytes),
    };
    let view = StorageView::contiguous(memory, 0, dtype, shape)?;
    view.zero(stream)?;
    Ok(view)
}

fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain integers or floats reinterpreted as their bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl ImageDiffusion {
    /// Allocate the runner's columns and bind the denoiser at `path` of
    /// `binding`. `stream` borrows the binding's stream, on which every
    /// step and sample update runs.
    pub fn bind(
        interface: &Interface,
        binding: &ComponentBinding,
        path: &str,
        layout: ImageDenoiserLayout,
        capacity: DiffusionCapacity,
        storage: DiffusionStorage,
        stream: Option<Stream>,
    ) -> Result<Self> {
        let owner = if path.is_empty() {
            binding.component.clone()
        } else {
            format!("{}.{path}", binding.component)
        };
        let device = binding.device;
        if device.is_cuda() != stream.is_some() {
            return Err(Error::Invariant(
                "an image denoiser runs on streams exactly on CUDA devices".into(),
            ));
        }
        if capacity.graph_widths.len() != capacity.tables {
            return Err(Error::Invariant(
                "graph widths must name every attention table".into(),
            ));
        }
        let width = i64::from(layout.width());
        let positions = allocate(
            device,
            DType::I64,
            &[3, capacity.tokens as i64],
            stream.as_ref(),
        )?;
        let timesteps = allocate(
            device,
            DType::F32,
            &[capacity.rows as i64, 1],
            stream.as_ref(),
        )?;
        let noise = allocate(
            device,
            storage.samples.dtype(),
            &[capacity.units as i64, width],
            stream.as_ref(),
        )?;
        let attention = AttentionColumns::allocate(
            AttentionCapacity {
                rows: capacity.rows,
                tokens: capacity.tokens,
                tables: capacity.tables,
                width: capacity.width,
            },
            device.cuda_index(),
            stream.as_ref(),
        )?;

        let slot_table = |name: &str| -> Result<Any> {
            let (_, view) = storage
                .slot_tables
                .iter()
                .find(|(table, _)| table == name)
                .ok_or_else(|| Error::Invariant(format!("the KV cache has no {name}")))?;
            Ok(Any::from(view.to_tensor()?))
        };
        let method = |name: &str| interface.import(&format!("{RUNNER}.{name}"));
        let runner = Handle::new(method("bound")?.call_values(&[
            binding.prepared().to_any(),
            Any::from(tvm_ffi::String::from(path)),
            Any::from(capacity.rows as i64),
            Any::from(capacity.tokens as i64),
            Any::from(capacity.units as i64),
            Any::from(capacity.width as i64),
            Any::from(integer_list(storage.page_tokens.iter().copied())),
            Any::from(integer_list(storage.windowed.iter().copied())),
            Any::from(storage.samples.to_tensor()?),
            Any::from(storage.slot_timesteps.to_tensor()?),
            Any::from(storage.products.to_tensor()?),
            slot_table("unit_tables")?,
            slot_table("start_pages")?,
            slot_table("table_shapes")?,
        ])?)?;
        let plans_attention = method("plans_attention")?.call_as::<bool>(&[runner.view()])?;
        let capture = GraphFunctions::from_interface(interface)?;
        Ok(Self {
            graphs: GraphTable::new(owner.clone(), capture.release.clone()),
            methods: Methods {
                runner,
                plan: method("plan")?,
                draw_noise: method("draw_noise")?,
                prepare_latents: method("prepare_latents")?,
                denoise: method("denoise")?,
                plan_denoise: method("plan_denoise")?,
                encode_conditions: method("encode_conditions")?,
                step: method("step")?,
                plans_attention,
                capture,
            },
            owner,
            layout,
            device,
            stream,
            capacity,
            attention,
            positions,
            timesteps,
            noise,
            capturing: false,
        })
    }

    pub fn layout(&self) -> &ImageDenoiserLayout {
        &self.layout
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The binding's stream; `None` on the host.
    pub fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    pub fn capacity(&self) -> &DiffusionCapacity {
        &self.capacity
    }

    /// The guidance plan of an admitted image's trajectory.
    pub fn plan(&self, image: &ImageParams) -> Result<GuidancePlan> {
        let value: tvm_ffi::Array<Any> = self.methods.plan.call_as(&[
            self.methods.runner.view(),
            (&Any::from(i64::from(image.steps))).into(),
            (&Any::from(f64::from(image.timestep_shift))).into(),
            (&Any::from(f64::from(image.cfg_text_scale))).into(),
            (&Any::from(f64::from(image.cfg_img_scale))).into(),
            (&Any::from(f64::from(image.cfg_interval.0))).into(),
            (&Any::from(f64::from(image.cfg_interval.1))).into(),
            (&Any::from(tvm_ffi::String::from(image.cfg_renorm_type.as_str()))).into(),
            (&Any::from(f64::from(image.cfg_renorm_min))).into(),
        ])?;
        let name = self.methods.plan.name();
        let item = |index: usize| {
            value
                .get(index)
                .map_err(|error| Error::numerical(name, &error))
        };
        if value.len() != 3 {
            return Err(Error::Interface(format!(
                "{name} returns [plan, timesteps, branches]"
            )));
        }
        let timesteps: tvm_ffi::Array<Any> =
            tvm_ffi::Array::try_from(item(1)?).map_err(|error| Error::numerical(name, &error))?;
        let timesteps = timesteps
            .iter()
            .map(|value| f64::try_from(value).map(|value| value as f32))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| Error::numerical(name, &error))?;
        let branches = crate::ffi::integers(item(2)?)
            .map_err(|error| Error::numerical(name, &error))?
            .into_iter()
            .map(|mask| {
                u8::try_from(mask)
                    .map_err(|_| Error::Interface(format!("{name} reports an invalid branch mask")))
                    .and_then(BranchSet::from_mask)
            })
            .collect::<Result<Vec<_>>>()?;
        if branches.len() != usize::from(image.steps) || timesteps.len() < branches.len() {
            return Err(Error::Interface(format!(
                "{name} plans {} steps for a {}-step image",
                branches.len(),
                image.steps
            )));
        }
        Ok(GuidancePlan {
            handle: Handle::new(item(0)?)?,
            timesteps,
            branches,
        })
    }

    /// Draw image `index`'s seeded initial noise with the library's draw,
    /// then zero `padded` sample rows from `sample` and write the image's
    /// initial latent into the first `units` of them, on the sample stream.
    #[expect(clippy::too_many_arguments, reason = "the trajectory's coordinates")]
    pub fn initial_latent(
        &self,
        sample: u32,
        units: u32,
        padded: u32,
        height: u32,
        width: u32,
        seed: u64,
        index: u32,
    ) -> Result<()> {
        let noise = Any::from(self.noise.to_tensor()?);
        self.methods.draw_noise.call_values(&[
            self.methods.runner.to_any(),
            // The seed's bits; the Python half reads them as unsigned.
            Any::from(seed as i64),
            Any::from(i64::from(index)),
            Any::from(i64::from(height)),
            Any::from(i64::from(width)),
            noise.clone(),
        ])?;
        self.methods.prepare_latents.call_values(&[
            self.methods.runner.to_any(),
            noise,
            Any::from(i64::from(sample)),
            Any::from(i64::from(units)),
            Any::from(i64::from(padded)),
            Any::from(i64::from(height)),
            Any::from(i64::from(width)),
        ])?;
        Ok(())
    }

    /// Write each row's positions and timestep into the columns; returns
    /// the rows' token count and prediction offsets.
    fn stage_rows(
        &self,
        rows: impl Iterator<Item = (u32, u32, u32, [Vec<i64>; 3], f32)>,
    ) -> Result<(usize, Vec<u32>, Vec<[i64; 7]>)> {
        let tokens_capacity = self.capacity.tokens;
        let mut axes: [Vec<i64>; 3] = Default::default();
        let mut times = Vec::new();
        let mut offsets = Vec::new();
        let mut described = Vec::new();
        let mut predicted = 0_u32;
        for (sample, height, width, positions, timestep) in rows {
            let units = self.layout.latent_rows(height, width)?;
            let token = axes[0].len();
            let count = positions[0].len();
            for (axis, values) in axes.iter_mut().zip(positions) {
                axis.extend(values);
            }
            described.push([
                i64::from(sample),
                i64::from(units),
                i64::from(height),
                i64::from(width),
                token as i64,
                count as i64,
                i64::from(predicted),
            ]);
            offsets.push(predicted);
            times.push(timestep);
            predicted += units;
        }
        let tokens = axes[0].len();
        if described.len() > self.capacity.rows
            || tokens > tokens_capacity
            || predicted as usize > self.capacity.units
        {
            return Err(Error::Invalid(format!(
                "{}: {} image rows of {tokens} tokens exceed the denoiser's columns",
                self.owner,
                described.len()
            )));
        }
        for (axis, values) in axes.iter().enumerate() {
            self.positions
                .select(0, axis as i64)?
                .upload(bytes(values), self.stream.as_ref())?;
        }
        self.timesteps.upload(bytes(&times), self.stream.as_ref())?;
        Ok((tokens, offsets, described))
    }

    /// The arguments of the Python half's `denoise`, `plan_denoise` and
    /// `encode_conditions` for staged rows and their attention geometry.
    fn arguments(&self, rows: &[[i64; 7]], geometry: &AttentionGeometry) -> Result<Vec<Any>> {
        let integers = |values: &[i64]| Any::from(integer_list(values.iter().copied()));
        Ok(vec![
            self.methods.runner.to_any(),
            Any::from(self.attention.column("row_columns")?.to_tensor()?),
            Any::from(self.positions.to_tensor()?),
            Any::from(self.timesteps.to_tensor()?),
            Any::from(integer_list(rows.iter().flatten().copied())),
            integers(&geometry.queries),
            integers(&geometry.prefixes),
            integers(&geometry.starts),
            integers(&geometry.widths),
        ])
    }

    /// Evaluate the denoising rows of one step. Rows of one image share
    /// samples; `check` validates attention reads against the cache.
    pub fn predict(
        &mut self,
        rows: &[DenoiseRow],
        tables: &BlockTables,
        check: impl FnMut(&GroupTable, u32, u32) -> Result<()>,
    ) -> Result<Predicted> {
        let positions = rows
            .iter()
            .map(|row| self.layout.positions(row.height, row.width, row.temporal))
            .collect::<Result<Vec<_>>>()?;
        let (_, offsets, described) =
            self.stage_rows(rows.iter().zip(positions).map(|(row, positions)| {
                (row.sample, row.height, row.width, positions, row.timestep)
            }))?;
        let attention: Vec<AttentionRow> = rows
            .iter()
            .zip(&described)
            .map(|(row, described)| AttentionRow {
                slot: row.slot,
                prefix: row.prefix,
                query: described[5] as u32,
                write: false,
            })
            .collect();

        // A captured shape is launched when the rows match it exactly.
        let captured = match DiffusionShape::of(rows) {
            Some(shape) if self.capturing && self.graphs.contains(&shape, None) => Some(shape),
            _ => None,
        };
        let widths = captured
            .as_ref()
            .map(|_| self.capacity.graph_widths.clone());
        let geometry = self.attention.stage(
            &attention,
            tables,
            widths.as_deref(),
            check,
            self.stream.as_ref(),
        )?;
        let arguments = self.arguments(&described, &geometry)?;
        let graph = match (&captured, &self.stream) {
            (Some(shape), Some(stream)) => match self.graphs.lookup(shape, None)? {
                Lookup::Replay(graph) => Some((graph, stream)),
                Lookup::Capture => None,
            },
            _ => None,
        };
        let mode = match graph {
            Some((graph, stream)) => {
                if self.methods.plans_attention {
                    self.methods.plan_denoise.call_values(&arguments)?;
                }
                graph.launch(stream)?;
                GraphMode::Replay
            }
            None => {
                self.methods.denoise.call_values(&arguments)?;
                GraphMode::Eager
            }
        };
        self.graphs.record(mode);
        Ok(Predicted { mode, offsets })
    }

    /// Write the K/V of latent images inserted as conditioning.
    pub fn condition(
        &mut self,
        rows: &[ConditionRow],
        tables: &BlockTables,
        check: impl FnMut(&GroupTable, u32, u32) -> Result<()>,
    ) -> Result<()> {
        let positions = rows
            .iter()
            .map(|row| {
                self.layout
                    .conditioning_positions(row.height, row.width, row.position)
            })
            .collect::<Result<Vec<_>>>()?;
        for row in rows {
            let units = self.layout.latent_rows(row.height, row.width)?;
            if row.features.shape() != [i64::from(units), i64::from(self.layout.width())] {
                return Err(Error::Invalid(
                    "state latent does not match the declared image dimensions".into(),
                ));
            }
        }
        let (_, _, described) = self.stage_rows(
            rows.iter()
                .zip(positions)
                .map(|(row, positions)| (0, row.height, row.width, positions, 0.0)),
        )?;
        let attention: Vec<AttentionRow> = rows
            .iter()
            .zip(&described)
            .map(|(row, described)| AttentionRow {
                slot: row.slot,
                prefix: row.prefix,
                query: described[5] as u32,
                write: true,
            })
            .collect();
        let geometry =
            self.attention
                .stage(&attention, tables, None, check, self.stream.as_ref())?;
        let mut arguments = self.arguments(&described, &geometry)?;
        arguments.push(Any::from(list(
            rows.iter().map(|row| row.features.to_any()),
        )));
        self.methods
            .encode_conditions
            .call_values(&arguments)
            .map(drop)
    }

    /// Combine each image's branch predictions and advance its samples by
    /// one solver step, on the sample stream.
    pub fn integrate(&self, rows: &[IntegrateRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut arguments = vec![
            self.methods.runner.to_any(),
            Any::from(integer_list(rows.iter().flat_map(|row| {
                [
                    i64::from(row.sample),
                    i64::from(row.units),
                    i64::from(row.slot),
                    i64::from(row.step),
                    i64::from(row.prediction),
                    i64::from(row.branches.mask()),
                ]
            }))),
        ];
        arguments.extend(rows.iter().map(|row| row.plan.to_any()));
        self.methods.step.call_values(&arguments).map(drop)
    }

    /// Capture the shape of `rows` after an eager evaluation of them (its
    /// warmup), which gathered the geometry the graph reads. `pools` holds
    /// the owner's allocator pool on the runner's device and one on each
    /// other device the network's modules run on (a generation tower).
    /// `check` runs after the capture (the graph storage budget); a failed
    /// check releases it.
    pub fn capture(
        &mut self,
        rows: &[DenoiseRow],
        tables: &BlockTables,
        pools: &[Arc<GraphPool<Handle>>],
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let shape = DiffusionShape::of(rows).ok_or_else(|| {
            Error::Invalid("a captured denoising step evaluates rows of one image size".into())
        })?;
        if self.graphs.contains(&shape, None) {
            return Ok(());
        }
        let positions = rows
            .iter()
            .map(|row| self.layout.positions(row.height, row.width, row.temporal))
            .collect::<Result<Vec<_>>>()?;
        let (_, _, described) =
            self.stage_rows(rows.iter().zip(positions).map(|(row, positions)| {
                (row.sample, row.height, row.width, positions, row.timestep)
            }))?;
        let attention: Vec<AttentionRow> = rows
            .iter()
            .zip(&described)
            .map(|(row, described)| AttentionRow {
                slot: row.slot,
                prefix: row.prefix,
                query: described[5] as u32,
                write: false,
            })
            .collect();
        let geometry: AttentionGeometry = self.attention.stage(
            &attention,
            tables,
            Some(&self.capacity.graph_widths),
            |_, _, _| Ok(()),
            self.stream.as_ref(),
        )?;
        // The warmup evaluation, then the capture of the same forward.
        let arguments = self.arguments(&described, &geometry)?;
        self.methods.denoise.call_values(&arguments)?;
        let (Some(ordinal), Some(stream)) = (self.device.cuda_index(), &self.stream) else {
            return Err(Error::Invariant(
                "a denoiser captures on its CUDA stream".into(),
            ));
        };
        let pool = pools
            .iter()
            .find(|pool| pool.device == ordinal)
            .ok_or_else(|| Error::Invariant("a denoiser graph has no pool on its device".into()))?;
        let peers: Vec<(i32, Handle)> = pools
            .iter()
            .filter(|pool| pool.device != ordinal)
            .map(|pool| (pool.device, pool.value.clone()))
            .collect();
        let capture = &self.methods.capture;
        let graph =
            capture.capture_with_peers(&pool.value, &peers, stream.raw(), ordinal, || {
                self.methods.denoise.call_values(&arguments).map(drop)
            })?;
        if let Err(error) = check() {
            if let Err(cleanup) = capture.release.invoke(&[graph.handle().view()]) {
                tracing::warn!(target: "uniserve_worker.graphs", "{}: releasing a refused graph failed: {cleanup}", self.owner);
            }
            return Err(error);
        }
        self.graphs.insert(shape, None, graph)?;
        self.capturing = true;
        Ok(())
    }

    /// Graphs captured so far.
    pub fn captured(&self) -> usize {
        self.graphs.len()
    }

    /// The graph pool's identity: the address of the runner's positions
    /// column, which no other runner shares.
    pub fn graph_key(&self) -> usize {
        self.positions.address() as usize
    }

    /// Refuse further captures: serving replays what startup captured.
    pub fn seal(&mut self) {
        self.graphs.seal();
    }

    /// Release the runner's graphs; the caller drained its streams.
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        self.graphs.close(aborted)
    }
}
