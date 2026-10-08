//! KV-conditioned image denoising (`ImageDenoiser` models: BAGEL,
//! SenseNova-U1).
//!
//! An [`ImageDiffusion`] runner serves one bound image denoiser. A denoising
//! step evaluates, for every image of the batch, one row per guidance
//! branch: the image's framed latent sequence attending (non-causally, read
//! only) to the KV prefix its branch conditions on. Rust owns the rows'
//! geometry: positions ([`ImageDenoiserLayout::positions`]), timesteps,
//! attention pages and the graph choice; the binding
//! (`uniserve_models.interface.denoiser`) evaluates the network into the
//! fixed prediction column and, per image, combines its branch predictions
//! with the trajectory's guidance and advances its samples in place
//! (`integrate`).
//!
//! Columns (bound once): `positions [3, T]` int64, `timesteps [R, 1]`
//! float32, `predictions [P, width]` in the prediction dtype, `step [1]`
//! int64 zero and the attention columns ([`AttentionColumns`]), where `R`
//! rows of at most `T` tokens predict at most `P` latent rows. Samples are
//! the latent pool's step workspace, into which the caller gathers each
//! trajectory's committed pages.
//!
//! Graphs: each captured step evaluates a fixed number of rows of one image
//! size at the configured attention widths ([`DiffusionShape`]); its host
//! prefix lengths are rebound per replay. Startup captures the configured
//! shapes largest first; serving replays an exact shape and runs any other
//! eagerly. Solver updates always run eagerly.

use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::ImageParams;
use uniserve_cuda::Stream;

use crate::config::Device;
use crate::ffi::{DType, Fields, Function, Functions, Handle, Region, integer_list, list};
use crate::model::bind::ComponentBinding;
use crate::model::image::ImageDenoiserLayout;
use crate::model::interface::Interface;
use crate::request::diffusion::{BranchSet, GuidancePlan};
use crate::runner::GraphPool;
use crate::runner::attention::{AttentionCapacity, AttentionColumns, AttentionGeometry};
use crate::runner::graphs::{GraphMode, GraphTable, Lookup};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{AttentionRow, BlockTables, GroupTable};
use crate::{Error, Result};

/// The geometry of a captured denoising step: its rows of one image size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DiffusionShape {
    pub rows: usize,
    pub height: u32,
    pub width: u32,
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

/// What a runner binds beside its own columns.
pub struct DiffusionStorage {
    /// The latent pool's step workspace `[units, width]`.
    pub samples: StorageView,
    /// The samples the network reads, on the denoiser's device: the step
    /// workspace, or a mirror of it.
    pub inputs: StorageView,
    /// The latent pool's per-slot timesteps `[slots + 1, 1]` float32.
    pub slot_timesteps: StorageView,
    /// The denoiser device's product arena (`uint8`).
    pub products: StorageView,
    /// The KV cache's slot tables (`slot.unit_tables`, ...) and layout.
    pub slot_tables: Vec<(String, StorageView)>,
    pub layout: tvm_ffi::Map<tvm_ffi::String, Any>,
}

struct Steps {
    plan: Function,
    initial_latent: Function,
    predict: Function,
    condition: Function,
    capture: Function,
    integrate: Function,
}

/// The runner of one bound image denoiser.
pub struct ImageDiffusion {
    owner: String,
    layout: ImageDenoiserLayout,
    device: Device,
    stream: Option<Stream>,
    sample_stream: Option<Stream>,
    steps: Steps,
    capacity: DiffusionCapacity,
    attention: AttentionColumns,
    positions: StorageView,
    timesteps: StorageView,
    graphs: GraphTable<DiffusionShape>,
    /// Releases a captured graph the storage check refused.
    release: Function,
    capturing: bool,
    /// Buffers bound to the binding, retained for its lifetime.
    _buffers: Vec<StorageView>,
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
    /// `binding`. `stream` borrows the binding's stream and
    /// `sample_stream` the latent pool's (the same stream when the pool
    /// shares the denoiser's device).
    #[expect(
        clippy::too_many_arguments,
        reason = "the binding's complete resources"
    )]
    pub fn bind(
        interface: &Interface,
        binding: &ComponentBinding,
        loaded: &Handle,
        path: &str,
        layout: ImageDenoiserLayout,
        capacity: DiffusionCapacity,
        storage: DiffusionStorage,
        streams: (Option<Stream>, Option<Stream>),
    ) -> Result<Self> {
        let owner = if path.is_empty() {
            binding.component.clone()
        } else {
            format!("{}.{path}", binding.component)
        };
        let device = binding.device;
        let (stream, sample_stream) = streams;
        if device.is_cuda() != stream.is_some() || stream.is_some() != sample_stream.is_some() {
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
        let predictions = allocate(
            device,
            layout.prediction_dtype,
            &[capacity.units as i64, width],
            stream.as_ref(),
        )?;
        let step = allocate(device, DType::I64, &[1], stream.as_ref())?;
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

        let own = [
            ("positions", positions.clone()),
            ("timesteps", timesteps.clone()),
            ("predictions", predictions),
            ("step", step),
            ("samples", storage.samples),
            ("inputs", storage.inputs),
            ("slot_timesteps", storage.slot_timesteps),
            ("products", storage.products),
        ];
        let mut tensors = Fields::new();
        let mut buffers = Vec::new();
        for (name, view) in own {
            tensors = tensors.with(name, view.to_tensor()?);
            buffers.push(view);
        }
        for (name, view) in storage.slot_tables {
            tensors = tensors.with(&format!("slot.{name}"), view.to_tensor()?);
            buffers.push(view);
        }
        let tensors = attention.tensors(tensors)?;
        let function = interface.function("denoiser.bind")?;
        let handle =
            |stream: &Option<Stream>| stream.as_ref().map_or(0, |stream| stream.raw() as i64);
        let functions = Functions::new(
            function.name(),
            function.call_values(&[
                binding.prepared().to_any(),
                loaded.to_any(),
                Any::from(tvm_ffi::String::from(path)),
                tensors.into_any(),
                Any::from(storage.layout),
                Any::from(handle(&sample_stream)),
            ])?,
        )?;
        let steps = Steps {
            plan: functions.get("plan")?,
            initial_latent: functions.get("initial_latent")?,
            predict: functions.get("predict")?,
            condition: functions.get("condition")?,
            capture: functions.get("capture")?,
            integrate: functions.get("integrate")?,
        };
        let release = functions.get("release")?;
        Ok(Self {
            graphs: GraphTable::new(owner.clone(), release.clone()),
            release,
            owner,
            layout,
            device,
            stream,
            sample_stream,
            steps,
            capacity,
            attention,
            positions,
            timesteps,
            capturing: false,
            _buffers: buffers,
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

    /// The latent pool's stream; `None` on the host.
    pub fn sample_stream(&self) -> Option<&Stream> {
        self.sample_stream.as_ref()
    }

    pub fn capacity(&self) -> &DiffusionCapacity {
        &self.capacity
    }

    /// The guidance plan of an admitted image's trajectory.
    pub fn plan(&self, image: &ImageParams) -> Result<GuidancePlan> {
        let value: tvm_ffi::Array<Any> = self.steps.plan.call_as(&[
            (&Any::from(i64::from(image.steps))).into(),
            (&Any::from(f64::from(image.timestep_shift))).into(),
            (&Any::from(f64::from(image.cfg_text_scale))).into(),
            (&Any::from(f64::from(image.cfg_img_scale))).into(),
            (&Any::from(f64::from(image.cfg_interval.0))).into(),
            (&Any::from(f64::from(image.cfg_interval.1))).into(),
            (&Any::from(tvm_ffi::String::from(image.cfg_renorm_type.as_str()))).into(),
            (&Any::from(f64::from(image.cfg_renorm_min))).into(),
        ])?;
        let name = self.steps.plan.name();
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
            handle: Handle::new(item(0)?),
            timesteps,
            branches,
        })
    }

    /// Zero `padded` sample rows from `sample` and draw image `index`'s
    /// seeded initial latent into the first `units` of them, on the sample
    /// stream.
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
        self.steps.initial_latent.invoke(&[
            (&Any::from(i64::from(sample))).into(),
            (&Any::from(i64::from(units))).into(),
            (&Any::from(i64::from(padded))).into(),
            (&Any::from(i64::from(height))).into(),
            (&Any::from(i64::from(width))).into(),
            (&Any::from(seed as i64)).into(),
            (&Any::from(i64::from(index))).into(),
        ])
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

    fn described(rows: &[[i64; 7]]) -> Any {
        Any::from(list(
            rows.iter()
                .map(|row| Any::from(integer_list(row.iter().copied()))),
        ))
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

        // One image size per evaluation can replay a captured shape.
        let shape = rows.first().map(|first| DiffusionShape {
            rows: rows.len(),
            height: first.height,
            width: first.width,
        });
        let uniform = rows.iter().all(|row| {
            Some((row.height, row.width)) == shape.map(|shape| (shape.height, shape.width))
        });
        let graph = match shape {
            Some(shape) if self.capturing && uniform && self.graphs.contains(&shape, None) => {
                match self.graphs.lookup(&shape, None)? {
                    Lookup::Replay(graph) => Some(graph.clone()),
                    Lookup::Capture => None,
                }
            }
            _ => None,
        };
        let widths = graph.as_ref().map(|_| self.capacity.graph_widths.clone());
        let geometry = self.attention.stage(
            &attention,
            tables,
            widths.as_deref(),
            check,
            self.stream.as_ref(),
        )?;
        self.steps.predict.invoke(&[
            (&graph.as_ref().map_or_else(Any::default, Handle::to_any)).into(),
            (&Self::described(&described)).into(),
            (&geometry.to_any()).into(),
        ])?;
        let mode = if graph.is_some() {
            GraphMode::Replay
        } else {
            GraphMode::Eager
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
        self.steps.condition.invoke(&[
            (&Self::described(&described)).into(),
            (&geometry.to_any()).into(),
            (&Any::from(list(rows.iter().map(|row| row.features.to_any())))).into(),
        ])
    }

    /// Combine each image's branch predictions and advance its samples by
    /// one solver step, on the sample stream.
    pub fn integrate(&self, rows: &[IntegrateRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let described = Any::from(list(rows.iter().map(|row| {
            Any::from(integer_list([
                i64::from(row.sample),
                i64::from(row.units),
                i64::from(row.slot),
                i64::from(row.step),
                i64::from(row.prediction),
                i64::from(row.branches.mask()),
            ]))
        })));
        let plans = Any::from(list(rows.iter().map(|row| row.plan.to_any())));
        self.steps
            .integrate
            .invoke(&[(&described).into(), (&plans).into()])
    }

    /// Capture `shape` after an eager evaluation of `rows` (its warmup),
    /// which gathered the geometry the graph reads. `check` runs after the
    /// capture (the graph storage budget); a failed check releases it.
    pub fn capture(
        &mut self,
        shape: DiffusionShape,
        rows: &[DenoiseRow],
        tables: &BlockTables,
        pool: &Arc<GraphPool<Handle>>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
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
        // The warmup evaluation, then the capture of the same geometry.
        self.steps.predict.invoke(&[
            (&Any::default()).into(),
            (&Self::described(&described)).into(),
            (&geometry.to_any()).into(),
        ])?;
        let graph = Handle::new(self.steps.capture.call_values(&[
            Self::described(&described),
            geometry.to_any(),
            pool.value.to_any(),
        ])?);
        if let Err(error) = check() {
            if let Err(cleanup) = self.release.invoke(&[graph.view()]) {
                tracing::warn!(target: "uniserve_worker.graphs", "{}: releasing a refused graph failed: {cleanup}", self.owner);
            }
            return Err(error);
        }
        self.graphs.insert(shape, None, graph)?;
        self.capturing = true;
        Ok(())
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
