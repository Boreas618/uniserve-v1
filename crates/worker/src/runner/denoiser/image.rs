//! Cache-conditioned image denoising (`ImageDenoiser` models: BAGEL,
//! SenseNova-U1): the image family of [`DenoiserRunner`].
//!
//! An image denoiser's runner serves one bound image denoiser
//! ([`ImageDenoiser`] holds the state only this family has). Its forward
//! is driven per row with the trajectory's guidance plan (`plan`,
//! `predict`, `integrate`), where the video family's is driven per layout
//! with slot state. A denoising step evaluates, for every image of the
//! batch, one row per guidance branch: the image's framed latent sequence
//! attending (non-causally, read only) to the cache prefix its branch
//! conditions on. Rust owns the rows'
//! geometry: positions ([`ImageDenoiserLayout::positions`]), timesteps,
//! attention pages and the graph choice; the Python half
//! (`uniserve.runner.denoiser.DenoiserRunner`) gathers the rows'
//! attention metadata on the device, evaluates the network into its
//! predictions (`denoise`) and, per image, combines its branch predictions
//! with the trajectory's guidance and advances its samples in place
//! (`step`).
//!
//! Input tensors of a forward: `positions [3, T]` int64, `timesteps [R, 1]`
//! float32 and the packed row columns of [`AttentionMetadata`], where `R`
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

use std::collections::HashMap;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::ImageParams;
use uniserve_cuda::Stream;
use uniserve_ipc::{DrawLayout, ForwardStats, LatentParams, RequestKey};

use crate::config::Device;
use crate::cuda_graph::GraphPool;
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::executor::image::ImageParts;
use crate::executor::image::{ImageBatch, Planned, Progress, Resume};
use crate::executor::transfer::latents;
use crate::ffi::{DType, Function, Handle, Interpreter, integer_list};
use crate::lane::{Completion, Job, Value};
use crate::parallel::ExpertUnion;
use crate::parallel::joins::GraphFunctions;
use crate::request::diffusion::{BranchSet, BranchSource, GuidancePlan, Prefix};
use crate::request::diffusion::{Coordinates, ForwardSlot, ImageTrajectory};
use crate::runner::ImageSetup;
use crate::runner::attention::{AttentionCapacity, AttentionMetadata};
use crate::runner::causal_lm::{CausalLMRunners, Step};
use crate::runner::denoiser::image_model::ImageDenoiserLayout;
use crate::runner::denoiser::{DENOISER, DenoiserRunner, Family, Methods};
use crate::runner::methods::{Argument, MethodCall};
use crate::storage::AttentionRow;
use crate::storage::LatentUpdate;
use crate::storage::PrefixCache;
use crate::storage::Scratch;
use crate::storage::memory::{Memory, StorageView};
use crate::stream::Issue;
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
    /// The request slot holding the branch's cache prefix and its visible
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
#[derive(Clone, Debug)]
pub struct ConditionRow {
    /// The latent feature product (`[units, width]`).
    pub features: StorageView,
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
    /// The image parameters its schedules and guidance are made from.
    pub params: Arc<ImageParams>,
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
    /// The prefix cache's slot tables (`unit_tables`, `start_pages`,
    /// `table_shapes`), and each numerical block table's page tokens and
    /// whether it is windowed.
    pub slot_tables: Vec<(String, StorageView)>,
    pub page_tokens: Vec<i64>,
    pub windowed: Vec<i64>,
}

/// An image denoiser's state: the columns its forwards stage, their
/// attention metadata and the graphs of its denoising steps.
pub(crate) struct ImageDenoiser {
    /// `DenoiserRunner.plan`, `plan_denoise` and `step`, which only an
    /// image denoiser has.
    plan: Function,
    plan_denoise: Function,
    step: Function,
    /// Whether an attention operator plans each launch on the host.
    plans_attention: bool,
    layout: ImageDenoiserLayout,
    capacity: DiffusionCapacity,
    attention: AttentionMetadata,
    positions: StorageView,
    timesteps: StorageView,
    /// One image's initial noise, `[P, width]` in the samples' dtype: the
    /// library's seeded draw fills it and `prepare_latents` reads it.
    noise: StorageView,
    pub(super) graphs: GraphTable<DiffusionShape>,
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

impl DenoiserRunner {
    /// The runner of an image model's denoiser: allocate its columns and
    /// construct the Python half of the denoiser `setup` names over the
    /// prefix cache `cache`. Every step and sample update runs on the
    /// setup's stream.
    pub fn image(
        setup: &ImageSetup<'_>,
        cache: &Handle,
        layout: ImageDenoiserLayout,
        capacity: DiffusionCapacity,
        storage: DiffusionStorage,
    ) -> Result<Self> {
        let owner = setup.owner();
        let (device, stream) = (setup.device, setup.stream.clone());
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
        let attention = AttentionMetadata::allocate(
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
                .ok_or_else(|| Error::Invariant(format!("the prefix cache has no {name}")))?;
            Ok(view.handle()?.to_any())
        };
        let method = |name: &str| Interpreter::import(&format!("{DENOISER}.{name}"));
        let mut arguments = setup.arguments();
        arguments.extend([
            cache.to_any(),
            Any::from(capacity.rows as i64),
            Any::from(capacity.tokens as i64),
            Any::from(capacity.units as i64),
            Any::from(capacity.width as i64),
            Any::from(integer_list(storage.page_tokens.iter().copied())),
            Any::from(integer_list(storage.windowed.iter().copied())),
            storage.samples.handle()?.to_any(),
            storage.slot_timesteps.handle()?.to_any(),
            slot_table("unit_tables")?,
            slot_table("start_pages")?,
            slot_table("table_shapes")?,
        ]);
        let runner = Handle::new(method("image")?.call_values(&arguments)?)?;
        let plans_attention = method("plans_attention")?.call_as::<bool>(&[runner.view()])?;
        let capture = GraphFunctions::import()?;
        Ok(Self {
            family: Family::Image(Box::new(ImageDenoiser {
                plan: method("plan")?,
                plan_denoise: method("plan_denoise")?,
                step: method("step")?,
                plans_attention,
                layout,
                capacity,
                attention,
                positions,
                timesteps,
                noise,
                graphs: GraphTable::new(owner.clone(), capture.release.clone()),
                capturing: false,
            })),
            methods: Methods::import(runner)?,
            capture,
            owner,
            device,
            stream,
        })
    }

    /// An image denoiser's layout.
    pub fn layout(&self) -> Result<&ImageDenoiserLayout> {
        Ok(&self.family.image()?.layout)
    }

    /// An image denoiser's column capacity.
    pub fn capacity(&self) -> Result<&DiffusionCapacity> {
        Ok(&self.family.image()?.capacity)
    }

    /// Record the call that makes the guidance plan of an admitted image's
    /// trajectory. The plan is a host computation of the library, so it is
    /// an operation of the job like the forward's other eager calls; the
    /// job's completion carries what it returned ([`Self::planned`]).
    pub(crate) fn plan(&self, image: &ImageParams, job: &Job) -> Result<()> {
        job.call_values(MethodCall::new(
            &self.family.image()?.plan,
            vec![
                Argument::Handle(self.methods.runner.clone()),
                Argument::Integer(i64::from(image.steps)),
                Argument::Float(f64::from(image.timestep_shift)),
                Argument::Float(f64::from(image.cfg_text_scale)),
                Argument::Float(f64::from(image.cfg_img_scale)),
                Argument::Float(f64::from(image.cfg_interval.0)),
                Argument::Float(f64::from(image.cfg_interval.1)),
                Argument::Text(image.cfg_renorm_type.as_str().to_owned()),
                Argument::Float(f64::from(image.cfg_renorm_min)),
            ],
        ));
        Ok(())
    }

    /// The guidance plan of an image of `params` from what its plan call
    /// returned: the model timestep of each solver step and the branch mask
    /// of each step.
    pub(crate) fn planned(
        &self,
        params: Arc<ImageParams>,
        values: Vec<Value>,
    ) -> Result<GuidancePlan> {
        let name = self.family.image()?.plan.name();
        let steps = usize::from(params.steps);
        let Ok([timesteps, Value::Integers(masks)]) = <[Value; 2]>::try_from(values) else {
            return Err(Error::Interface(format!(
                "{name} returns [timesteps, branches]"
            )));
        };
        let timesteps: Vec<f32> = match timesteps {
            Value::Floats(times) => times.into_iter().map(|time| time as f32).collect(),
            Value::Integers(times) => times.into_iter().map(|time| time as f32).collect(),
        };
        let branches = masks
            .into_iter()
            .map(|mask| {
                u8::try_from(mask)
                    .map_err(|_| Error::Interface(format!("{name} reports an invalid branch mask")))
                    .and_then(BranchSet::from_mask)
            })
            .collect::<Result<Vec<_>>>()?;
        if branches.len() != steps || timesteps.len() < branches.len() {
            return Err(Error::Interface(format!(
                "{name} plans {} steps for a {steps}-step image",
                branches.len(),
            )));
        }
        Ok(GuidancePlan {
            params,
            timesteps,
            branches,
        })
    }

    /// Draw image `index`'s seeded initial noise with the library's draw,
    /// then zero `padded` sample rows from `sample` and write the image's
    /// initial latent into the first `units` of them, on the sample stream.
    #[expect(clippy::too_many_arguments, reason = "the trajectory's coordinates")]
    pub(crate) fn initial_latent(
        &self,
        sample: u32,
        units: u32,
        padded: u32,
        height: u32,
        width: u32,
        seed: u64,
        index: u32,
        job: &Job,
    ) -> Result<()> {
        let runner = Argument::Handle(self.methods.runner.clone());
        let noise = Argument::Tensor(self.family.image()?.noise.clone());
        job.call(MethodCall::new(
            &self.methods.draw_noise,
            vec![
                runner.clone(),
                // The seed's bits; the Python half reads them as unsigned.
                Argument::Integer(seed as i64),
                Argument::Integer(i64::from(index)),
                Argument::Integer(i64::from(height)),
                Argument::Integer(i64::from(width)),
                noise.clone(),
            ],
        ));
        job.call(MethodCall::new(
            &self.methods.prepare_latents,
            vec![
                runner,
                noise,
                Argument::Integer(i64::from(sample)),
                Argument::Integer(i64::from(units)),
                Argument::Integer(i64::from(padded)),
                Argument::Integer(i64::from(height)),
                Argument::Integer(i64::from(width)),
            ],
        ));
        Ok(())
    }

    /// Evaluate the denoising rows of one step. Rows of one image share
    /// samples; their attention reads `prefix_cache`'s pages.
    pub(crate) fn predict(
        &mut self,
        rows: &[DenoiseRow],
        prefix_cache: &mut PrefixCache,
        job: &Job,
    ) -> Result<Predicted> {
        let stream = self.stream.clone();
        let issue = job.on(stream.as_ref());
        let image = self.family.image_mut()?;
        let positions = rows
            .iter()
            .map(|row| image.layout.positions(row.height, row.width, row.temporal))
            .collect::<Result<Vec<_>>>()?;
        let (_, offsets, described) = image.stage_rows(
            &self.owner,
            rows.iter().zip(positions).map(|(row, positions)| {
                (row.sample, row.height, row.width, positions, row.timestep)
            }),
            issue,
        )?;
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
            Some(shape) if image.capturing && image.graphs.contains(&shape, None) => Some(shape),
            _ => None,
        };
        let widths = captured
            .as_ref()
            .map(|_| image.capacity.graph_widths.clone());
        image
            .attention
            .stage(&attention, prefix_cache, widths.as_deref(), issue)?;
        let arguments = image.arguments(&self.methods.runner, &described);
        let graph = match (&captured, &stream) {
            (Some(shape), Some(_)) => match image.graphs.lookup(shape, None)? {
                Lookup::Replay(graph) => Some(graph),
                Lookup::Capture => None,
            },
            _ => None,
        };
        let mode = match graph {
            Some(graph) => {
                if image.plans_attention {
                    job.call(MethodCall::new(&image.plan_denoise, arguments));
                }
                issue.launch(graph)?;
                GraphMode::Replay
            }
            None => {
                job.call(MethodCall::new(&self.methods.denoise, arguments));
                GraphMode::Eager
            }
        };
        image.graphs.record(mode);
        Ok(Predicted { mode, offsets })
    }

    /// Write the K/V of latent images inserted as conditioning into
    /// `prefix_cache`, every write checked against its live accesses.
    pub(crate) fn condition_latents(
        &mut self,
        rows: &[ConditionRow],
        prefix_cache: &mut PrefixCache,
        job: &Job,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let issue = job.on(stream.as_ref());
        let image = self.family.image_mut()?;
        let positions = rows
            .iter()
            .map(|row| {
                image
                    .layout
                    .conditioning_positions(row.height, row.width, row.position)
            })
            .collect::<Result<Vec<_>>>()?;
        for row in rows {
            let units = image.layout.latent_rows(row.height, row.width)?;
            if row.features.shape() != [i64::from(units), i64::from(image.layout.width())] {
                return Err(Error::Invalid(
                    "state latent does not match the declared image dimensions".into(),
                ));
            }
        }
        let (_, _, described) = image.stage_rows(
            &self.owner,
            rows.iter()
                .zip(positions)
                .map(|(row, positions)| (0, row.height, row.width, positions, 0.0)),
            issue,
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
        image
            .attention
            .stage(&attention, prefix_cache, None, issue)?;
        let mut arguments = image.arguments(&self.methods.runner, &described);
        // Each row's features follow the forward's tensors as a tensor of
        // their own.
        arguments.extend(
            rows.iter()
                .map(|row| Argument::Tensor(row.features.clone())),
        );
        job.call(MethodCall::new(&self.methods.encode_conditions, arguments));
        Ok(())
    }

    /// Combine each image's branch predictions and advance its samples by
    /// one solver step, on the sample stream.
    pub(crate) fn integrate(&self, rows: &[IntegrateRow], job: &Job) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut arguments = vec![
            Argument::Handle(self.methods.runner.clone()),
            Argument::Integers(
                rows.iter()
                    .flat_map(|row| {
                        [
                            i64::from(row.sample),
                            i64::from(row.units),
                            i64::from(row.slot),
                            i64::from(row.step),
                            i64::from(row.prediction),
                            i64::from(row.branches.mask()),
                        ]
                    })
                    .collect(),
            ),
        ];
        // Each image's parameters, from which the solver update makes its
        // schedules and guidance as the plan call did.
        arguments.push(Argument::Floats(
            rows.iter()
                .flat_map(|row| {
                    let params = &row.params;
                    [
                        f64::from(params.steps),
                        f64::from(params.timestep_shift),
                        f64::from(params.cfg_text_scale),
                        f64::from(params.cfg_img_scale),
                        f64::from(params.cfg_interval.0),
                        f64::from(params.cfg_interval.1),
                        f64::from(params.cfg_renorm_min),
                    ]
                })
                .collect(),
        ));
        arguments.push(Argument::Texts(
            rows.iter()
                .map(|row| row.params.cfg_renorm_type.as_str().to_owned())
                .collect(),
        ));
        job.call(MethodCall::new(&self.family.image()?.step, arguments));
        Ok(())
    }

    /// Capture the denoising step that evaluates `rows` after an eager
    /// evaluation of them (its warmup), which gathered the geometry the
    /// graph reads. `pools` holds the owner's allocator pool on the
    /// runner's device and one on each other device the network's modules
    /// run on (a generation tower). `check` runs after the capture (the
    /// graph storage budget); a failed check releases it.
    pub fn capture_step(
        &mut self,
        rows: &[DenoiseRow],
        prefix_cache: &mut PrefixCache,
        pools: &[Arc<GraphPool<Handle>>],
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let image = self.family.image_mut()?;
        let shape = DiffusionShape::of(rows).ok_or_else(|| {
            Error::Invalid("a captured denoising step evaluates rows of one image size".into())
        })?;
        if image.graphs.contains(&shape, None) {
            return Ok(());
        }
        let positions = rows
            .iter()
            .map(|row| image.layout.positions(row.height, row.width, row.temporal))
            .collect::<Result<Vec<_>>>()?;
        let (_, _, described) = image.stage_rows(
            &self.owner,
            rows.iter().zip(positions).map(|(row, positions)| {
                (row.sample, row.height, row.width, positions, row.timestep)
            }),
            self.stream.as_ref(),
        )?;
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
        image.attention.stage(
            &attention,
            prefix_cache,
            Some(&image.capacity.graph_widths),
            self.stream.as_ref(),
        )?;
        // The warmup evaluation, then the capture of the same forward.
        let arguments = image.arguments(&self.methods.runner, &described);
        MethodCall::new(&self.methods.denoise, arguments.clone()).invoke()?;
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
        let capture = &self.capture;
        let denoise = MethodCall::new(&self.methods.denoise, arguments);
        let graph =
            capture.capture_with_peers(&pool.value, &peers, stream.raw(), ordinal, || {
                denoise.invoke().map(drop)
            })?;
        if let Err(error) = check() {
            if let Err(cleanup) = capture.release.invoke(&[graph.handle().view()]) {
                tracing::warn!(target: "uniserve_worker.graphs", "{}: releasing a refused graph failed: {cleanup}", self.owner);
            }
            return Err(error);
        }
        image.graphs.insert(shape, None, graph)?;
        image.capturing = true;
        Ok(())
    }

    /// The graph pool's identity: the address of an image denoiser's
    /// positions column, which no other runner shares.
    pub fn graph_key(&self) -> Result<usize> {
        Ok(self.family.image()?.positions.address() as usize)
    }
}

impl ImageDenoiser {
    /// Write each row's positions and timestep into the columns; returns
    /// the rows' token count and prediction offsets. `owner` names the
    /// runner in errors.
    fn stage_rows(
        &self,
        owner: &str,
        rows: impl Iterator<Item = (u32, u32, u32, [Vec<i64>; 3], f32)>,
        stream: impl Issue,
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
                "{owner}: {} image rows of {tokens} tokens exceed the denoiser's columns",
                described.len()
            )));
        }
        for (axis, values) in axes.iter().enumerate() {
            self.positions
                .select(0, axis as i64)?
                .upload(bytes(values), stream)?;
        }
        self.timesteps.upload(bytes(&times), stream)?;
        Ok((tokens, offsets, described))
    }

    /// The arguments of the Python half `runner`'s `denoise`,
    /// `plan_denoise` and `encode_conditions` for staged rows and their
    /// staged attention metadata.
    fn arguments(&self, runner: &Handle, rows: &[[i64; 7]]) -> Vec<Argument> {
        let attention = &self.attention;
        vec![
            Argument::Handle(runner.clone()),
            Argument::Tensor(attention.row_columns().clone()),
            Argument::Tensor(self.positions.clone()),
            Argument::Tensor(self.timesteps.clone()),
            Argument::Integers(rows.iter().flatten().copied().collect()),
            Argument::Integers(attention.queries.clone()),
            Argument::Integers(attention.prefixes.clone()),
            Argument::Integers(attention.starts.clone()),
            Argument::Integers(attention.widths.clone()),
        ]
    }
}

// Image latent preparation and cache-conditioned denoising batches
// (`LatentPreparation`, `Denoising`).
//
// A trajectory starts with latent preparation: its guidance plan opens, its
// seeded initial latent is drawn into the step workspace and scattered into
// bank 1 of its pages. Each denoising call then advances it by an interval
// of solver steps: the committed bank is gathered into the step workspace;
// every step fills the slot's timestep, resolves the guidance branches it
// evaluates (prefilling missing guidance prefixes through the text
// image), predicts every branch of every image (one numerical call per
// image size) and integrates each image's predictions; the interval's
// result is scattered into the other bank. Commits become visible when the
// batch commits.

/// The denoiser serving a batch, with its device and stream.
pub(crate) fn image_placement(
    image: &ImageParts<'_>,
) -> Result<(Option<i32>, Option<Stream>, usize)> {
    let runner = image.denoiser().ok_or_else(|| {
        Error::Unsupported("image trajectories are served by an image denoiser".into())
    })?;
    let stream = runner.stream().cloned();
    Ok((runner.device().cuda_index(), stream, 0))
}

/// The denoising rows of one image size: the size, its rows and, per
/// stepping image, its position among the batch's trajectories and its
/// first row.
type SizeGroup = ((u32, u32), Vec<DenoiseRow>, Vec<(usize, usize)>);

/// The stepping trajectories of one solver step: each one's position among
/// the batch's trajectories, its step, its guidance branches and its
/// timestep.
type Stepping = Vec<(usize, u32, BranchSet, f32)>;

/// The guidance prefixes a step misses: the position of the trajectory, the
/// slot the prefix is written to and its tokens.
type Fills = Vec<(usize, u32, Arc<[u32]>)>;

/// What a denoising keeps between its solver steps.
pub(crate) struct Denoising {
    /// Each active call's trajectory and admitted image, with its placed
    /// samples in `scratch`.
    trajectories: Vec<(Trajectory, ImageParams)>,
    scratch: Vec<Scratch>,
    /// The forward rows of each call, by call index.
    descriptors: HashMap<usize, Vec<ForwardSlot>>,
    layout: ImageDenoiserLayout,
    component: String,
    stats: ForwardStats,
    /// Solver steps of the longest interval, and the next one to record.
    intervals: u32,
    offset: u32,
}

/// A denoising between two of its jobs: the job on the lane ends with the
/// text runner's evaluation of the guidance prefixes step `state.offset`
/// misses.
pub(crate) struct DenoiseResume {
    state: Denoising,
    stepping: Stepping,
    fills: Fills,
    /// The evaluation, completed from the job's completion.
    step: Step,
}

/// A denoising between two of its jobs: the job on the lane ends with the
/// plan calls of the trajectories the denoising opens (one this rank did
/// not prepare, or one resumed at another size). Their plans say which
/// branches each step evaluates, so the steps are recorded once they are
/// known.
///
/// FIXME: no test and no reference request reaches this boundary. A rank
/// opens the trajectories it denoises when it prepares their latents; one
/// it did not prepare arrives only as a transferred trajectory, which the
/// stub rank's tests and the image harness do not build.
pub(crate) struct PlanResume {
    /// Each active call's trajectory, admitted image and conditioning
    /// coordinates.
    trajectories: Vec<(Trajectory, ImageParams, Coordinates)>,
    /// The forward rows of each call, by call index.
    descriptors: HashMap<usize, Vec<ForwardSlot>>,
    /// The trajectories to open, in the order their plan calls were
    /// recorded.
    plans: Vec<Planned>,
}

/// One active trajectory call of a batch.
struct Trajectory {
    index: usize,
    slot: usize,
    params: LatentParams,
    steps: u32,
}

impl ImageBatch<'_> {
    /// The latent allocation of call `index`.
    fn latent_params(&self, index: usize) -> Result<LatentParams> {
        let call = &self.plan.calls[index];
        self.plan
            .latent_params
            .iter()
            .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
            .cloned()
            .ok_or_else(|| Error::Invalid("trajectory call has no latent allocation".into()))
    }

    /// Validate call `index`'s allocation against its admitted image and
    /// return the admission.
    fn trajectory(&self, index: usize) -> Result<(Trajectory, uniserve_core::ImageParams)> {
        let call = &self.plan.calls[index];
        let request = self.requests.get(call.request_key.request_id.0)?;
        let image =
            request.admission().image.clone().ok_or_else(|| {
                Error::Invalid("image trajectory requires image parameters".into())
            })?;
        let params = self.latent_params(index)?;
        let layout = *self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
            .layout()?;
        if (params.height, params.width) != (image.height, image.width)
            || params.latent_units != layout.latent_rows(params.height, params.width)?
        {
            return Err(Error::Invalid(
                "latent allocation disagrees with the admitted image".into(),
            ));
        }
        Ok((
            Trajectory {
                index,
                slot: self.batch.calls[index].slot,
                params,
                steps: u32::from(image.steps),
            },
            image,
        ))
    }

    /// The request's own conditioning prefix cache: the export it published and its
    /// visible extent.
    fn conditioning(&mut self, index: usize) -> Result<Coordinates> {
        let call = &self.plan.calls[index];
        let buffer = call.cache_input.ok_or_else(|| {
            Error::Invalid("image trajectory requires its conditioning prefix cache".into())
        })?;
        let slot = self.batch.calls[index].slot as u32;
        let visible = self.batch.calls[index].output.progress.cache_visible_len;
        let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
            Error::Unsupported("image trajectories require a prefix cache".into())
        })?;
        let coordinates = prefix_cache.tables().coordinates(slot, visible)?;
        prefix_cache.validate_conditioning(call.request_key, buffer, slot, visible)?;
        Ok(coordinates.into())
    }

    /// Export the version call `index` just wrote into `bank` (at solver
    /// `step`) to other ranks and workers, when the rank publishes there
    /// and reports the component's results. Readers on this rank use the
    /// committed trajectory directly.
    fn publish_latent(
        &mut self,
        index: usize,
        bank: u8,
        step: u32,
        params: &LatentParams,
        stream: impl Issue,
    ) -> Result<()> {
        let call = &self.plan.calls[index];
        let Some(product) = &call.latent_output else {
            return Ok(());
        };
        if !self.batch.reports || !self.storage.transports.exports_remotely() {
            return Ok(());
        }
        let (pool, storage) = self.parts.latent_pool()?;
        let export = latents::issue_export(
            pool,
            storage,
            &self.storage.transports,
            call,
            product,
            self.batch.calls[index].slot,
            params,
            bank,
            step,
            stream,
        )?;
        // The batch's commit locates it once this work was performed.
        self.latent_exports.push(export.buffer());
        self.exports.push((index, export));
        Ok(())
    }

    /// Open the batch's use of the latent pool, behind its previous user
    /// in what `stream` issues, and place its trajectories.
    fn place(&mut self, trajectories: &[Trajectory], stream: impl Issue) -> Result<Vec<Scratch>> {
        let latents = self.parts.latents()?;
        latents.enter(stream)?;
        self.latents_entered = true;
        let params: Vec<&LatentParams> = trajectories.iter().map(|value| &value.params).collect();
        latents.place(&params)
    }

    /// Record into `job` the calls a trajectory of request `key` opens
    /// with: its guidance plan, and the prefix of each source one of its
    /// guidance branches may condition on. The model frames and tokenizes
    /// a prefix; admitted negative tokens are the negative prefix as they
    /// are. [`Self::planned_trajectory`] opens the trajectory from the
    /// job's completion.
    fn plan_trajectory(
        &self,
        slot: usize,
        key: RequestKey,
        size: (u32, u32),
        image: &ImageParams,
        job: &Job,
    ) -> Result<Planned> {
        let denoiser = self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?;
        denoiser.plan(image, job)?;

        let negative = self
            .requests
            .get(key.request_id.0)?
            .admission()
            .ar
            .as_ref()
            .map(|ar| ar.negative_token_ids.clone())
            .unwrap_or_default();
        let prompt = image.image_prompts.first().cloned().unwrap_or_default();
        let mut prefixes: Vec<(BranchSource, Option<Prefix>)> = Vec::new();
        for source in [
            BranchSource::Conditioning,
            BranchSource::NegativeOrStart,
            denoiser.layout()?.image_unconditional,
        ] {
            if prefixes.iter().any(|(known, _)| *known == source) {
                continue;
            }
            let given =
                (source == BranchSource::NegativeOrStart && !negative.is_empty()).then(|| Prefix {
                    tokens: Arc::from(&negative[..]),
                    reuses_conditioning: false,
                });
            if given.is_none() {
                self.parts
                    .policy
                    .ask_prefix(source, &prompt, &image.negative_prompt, job);
            }
            prefixes.push((source, given));
        }
        Ok(Planned {
            slot,
            key,
            size,
            params: Arc::new(image.clone()),
            prefixes,
        })
    }

    /// The trajectory `planned` opens and its slot, from the `completion`
    /// of the job that made its calls.
    pub(crate) fn planned_trajectory(
        &self,
        planned: Planned,
        completion: &mut Completion,
    ) -> Result<(usize, ImageTrajectory)> {
        let Planned {
            slot,
            key,
            size,
            params,
            prefixes,
        } = planned;
        let denoiser = self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?;
        let plan = denoiser.planned(params, completion.values()?)?;
        let prefixes = prefixes
            .into_iter()
            .map(|(source, given)| {
                let prefix = match given {
                    Some(prefix) => prefix,
                    None => self.parts.policy.prefix(&completion.returned()?)?,
                };
                Ok((source, prefix))
            })
            .collect::<Result<_>>()?;
        Ok((slot, ImageTrajectory::new(key, size, plan, prefixes)))
    }

    /// Draw each trajectory's seeded initial latent and stage it as its
    /// first committed version.
    pub(crate) fn prepare_latents(&mut self) -> Result<ForwardStats> {
        // The forward records into the batch's job, taken for its length so
        // that the batch's owners stay borrowable beside it.
        let job = std::mem::take(&mut self.job);
        let stats = self.record_latents(&job);
        self.job = job;
        stats
    }

    fn record_latents(&mut self, job: &Job) -> Result<ForwardStats> {
        let stream = self.batch.stream.clone();
        let issue = job.on(stream.as_ref());
        let mut trajectories = Vec::new();
        for index in self.active() {
            let call = &self.plan.calls[index];
            let (trajectory, image) = self.trajectory(index)?;
            if trajectory.params.start_step != 0 || trajectory.params.step_count != 0 {
                return Err(Error::Invalid(
                    "latent preparation assigns no denoising steps".into(),
                ));
            }
            if self.batch.calls[index].output.progress.flow_step != 0 {
                return Err(Error::Invalid(
                    "latent preparation requires a fresh trajectory".into(),
                ));
            }
            let rng = call
                .rng
                .ok_or_else(|| Error::Invalid("latent preparation requires an rng".into()))?;
            if rng.draw_layout != DrawLayout::FlowNoise
                || rng.seed != image.seed.unwrap_or(0)
                || rng.semantic_index_base == 0
            {
                return Err(Error::Invalid(
                    "latent preparation rng disagrees with the admitted image".into(),
                ));
            }
            if call
                .latent_output
                .as_ref()
                .is_none_or(|output| output.generation == 0)
            {
                return Err(Error::Invalid(
                    "latent preparation requires a latent output".into(),
                ));
            }
            self.conditioning(index)?;
            trajectories.push((trajectory, image, rng));
        }
        if trajectories.is_empty() {
            return Ok(ForwardStats::default());
        }
        let scratch = {
            let values: Vec<Trajectory> = trajectories
                .iter()
                .map(|(value, ..)| Trajectory {
                    index: value.index,
                    slot: value.slot,
                    params: value.params.clone(),
                    steps: value.steps,
                })
                .collect();
            self.place(&values, issue)?
        };

        for ((trajectory, image, rng), scratch) in trajectories.iter().zip(&scratch) {
            let call = &self.plan.calls[trajectory.index];
            let denoiser = self.parts.denoiser().ok_or_else(|| {
                Error::Unsupported("image trajectories require a denoiser".into())
            })?;
            let planned = self.plan_trajectory(
                trajectory.slot,
                call.request_key,
                (trajectory.params.height, trajectory.params.width),
                image,
                job,
            )?;
            let latents = self.parts.latents()?;
            let page_units = latents.pool.page_units();
            latents
                .pool
                .require_initial(trajectory.slot, &scratch.pages)?;
            let semantic = u32::try_from(rng.semantic_index_base)
                .map_err(|_| Error::Invalid("image index exceeds its range".into()))?;
            denoiser.initial_latent(
                scratch.sample,
                scratch.units,
                scratch.padded(page_units),
                trajectory.params.height,
                trajectory.params.width,
                rng.seed,
                semantic,
                job,
            )?;
            latents.scatter(1, scratch, issue)?;
            self.publish_latent(trajectory.index, 1, 0, &trajectory.params, issue)?;
            // The trajectory opens with the batch's commit, once its
            // calls were made.
            self.plans.push(planned);
            self.updates.push((
                trajectory.index,
                LatentUpdate::for_call(trajectory.slot, call, &trajectory.params)?,
            ));
        }
        Ok(ForwardStats::default())
    }

    /// Install the block tables of every slot the active calls' forward
    /// rows address, reset their new units and retain the extents the rows
    /// read and write until the batch's device work retires. Rows on slots
    /// other than a call's own hold its guidance prefixes.
    fn reserve_rows(
        &mut self,
        active: &[usize],
        stream: impl Issue,
    ) -> Result<HashMap<usize, Vec<ForwardSlot>>> {
        let plan = self.plan;
        let forward = &plan.forward;
        let mut descriptors: HashMap<usize, Vec<ForwardSlot>> = HashMap::new();
        for (row, &call) in forward.call_indices.iter().enumerate() {
            let index = call as usize;
            if index >= plan.calls.len() {
                return Err(Error::Invalid("forward row names no call".into()));
            }
            if !active.contains(&index) {
                continue;
            }
            descriptors.entry(index).or_default().push(ForwardSlot {
                slot: forward.request_pool_indices[row],
                seq_len: forward.seq_lens[row],
                query_len: forward.query_lens[row],
            });
        }
        let slots: std::collections::HashSet<u32> = descriptors
            .values()
            .flatten()
            .map(|row| row.slot)
            .chain(
                active
                    .iter()
                    .map(|&index| self.batch.calls[index].slot as u32),
            )
            .collect();
        let units = self.parts.denoisers.units.as_ref();
        let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
            Error::Unsupported("image trajectories require a prefix cache".into())
        })?;
        let assignments: Vec<_> = plan
            .block_tables
            .iter()
            .filter(|table| slots.contains(&table.request_pool_idx))
            .cloned()
            .collect();
        prefix_cache.install(&assignments, stream)?;
        let new_units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| slots.contains(&allocation.request_pool_idx))
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        if !new_units.is_empty() {
            let column = units.ok_or_else(|| {
                Error::Unsupported("image trajectories reset cache units on their stream".into())
            })?;
            prefix_cache.recycle(&new_units, column, stream)?;
        }

        for (row, &call) in forward.call_indices.iter().enumerate() {
            let index = call as usize;
            if !active.contains(&index) {
                continue;
            }
            let key = plan.calls[index].request_key;
            let slot = forward.request_pool_indices[row];
            let own = self.batch.calls[index].slot as u32;
            if slot != own {
                let prefix = forward.seq_lens[row] - forward.query_lens[row];
                if prefix > prefix_cache.tables().allocated_length(slot) {
                    return Err(Error::Invalid(
                        "forward row exceeds alternative-prefix capacity".into(),
                    ));
                }
                prefix_cache.retain_prefix(key, slot);
            }
            // Readers retain their prefix, writers also their query tokens.
            let length = u64::from(if forward.write_cache[row] {
                forward.seq_lens[row]
            } else {
                forward.seq_lens[row] - forward.query_lens[row]
            });
            prefix_cache.retain(key, slot, length, &self.batch.retirement)?;
        }
        Ok(descriptors)
    }

    /// Advance every active trajectory by its assigned interval. Missing
    /// guidance prefixes are evaluated by `text` first, as expert steps on a
    /// rank with `experts`.
    pub(crate) fn denoise(
        &mut self,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        // The forward records into the batch's job, taken for its length so
        // that the batch's owners stay borrowable beside it.
        let job = std::mem::take(&mut self.job);
        let progress = self.begin_denoise(&job, text, experts);
        self.job = job;
        progress
    }

    /// Continue the denoising from the `completion` of the job that ended
    /// with an evaluation of guidance prefixes.
    pub(crate) fn resume_denoise(
        &mut self,
        resume: DenoiseResume,
        completion: &mut Completion,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let job = std::mem::take(&mut self.job);
        let DenoiseResume {
            state,
            stepping,
            fills,
            step,
        } = resume;
        let progress = self.denoise_steps(
            &job,
            state,
            Some((stepping, fills, step, completion)),
            text,
            experts,
        );
        self.job = job;
        progress
    }

    /// Continue the denoising from the `completion` of the job that ended
    /// with the plan calls of the trajectories it opens.
    pub(crate) fn resume_planned(
        &mut self,
        resume: PlanResume,
        completion: &mut Completion,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let PlanResume {
            trajectories,
            descriptors,
            plans,
        } = resume;
        for planned in plans {
            let (slot, trajectory) = self.planned_trajectory(planned, completion)?;
            self.parts.trajectories.open(slot, trajectory);
        }

        let job = std::mem::take(&mut self.job);
        let progress = self.gather_denoise(&job, trajectories, descriptors, text, experts);
        self.job = job;
        progress
    }

    /// Record what precedes the solver steps: the wait on the text
    /// runners' order, the reservation of the rows' slots, the resumption
    /// of each trajectory and the gather of its committed version. The
    /// denoising is one job unless it opens a trajectory, whose plan call
    /// then ends the job, or a step misses guidance prefixes: the job then
    /// ends with the text runner's evaluation of them, which is completed
    /// before the step is recorded.
    fn begin_denoise(
        &mut self,
        job: &Job,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let plan = self.plan;
        let active = self.active();
        if active.is_empty() {
            return Ok(Progress::done(ForwardStats::default()));
        }
        let stream = self.batch.stream.clone();
        if let Some(text) = text.as_deref() {
            // Conditioning K/V and earlier prefixes were written by text
            // batches, possibly on another stream.
            text.order().wait(job.on(stream.as_ref()))?;
        }
        let descriptors = self.reserve_rows(&active, job.on(stream.as_ref()))?;

        // Validate every call and find the trajectories to open.
        let mut trajectories = Vec::with_capacity(active.len());
        let mut plans = Vec::new();
        for &index in &active {
            let call = &plan.calls[index];
            let (trajectory, image) = self.trajectory(index)?;
            let params = &trajectory.params;
            let generations = (
                call.latent_input.as_ref().map(|input| input.generation),
                call.latent_output.as_ref().map(|output| output.generation),
            );
            match generations {
                (Some(input), Some(output)) if input > 0 && output > 0 && input != output => {}
                _ => {
                    return Err(Error::Invalid(
                        "denoising requires distinct latent input and output generations".into(),
                    ));
                }
            }
            if call.rng.is_some() {
                return Err(Error::Invalid("denoising draws no random values".into()));
            }
            if params.step_count == 0
                || params.start_step + params.step_count > trajectory.steps
                || params.step_count > call.bounds.max_tokens
            {
                return Err(Error::Invalid(
                    "denoising interval lies outside the admitted schedule".into(),
                ));
            }
            let conditioning = self.conditioning(index)?;
            let size = (params.height, params.width);
            let key = call.request_key;
            let reopen = self
                .parts
                .trajectories
                .get(trajectory.slot, key)
                .is_none_or(|open| open.size != size);
            if reopen {
                plans.push(self.plan_trajectory(trajectory.slot, key, size, &image, job)?);
            }
            trajectories.push((trajectory, image, conditioning));
        }

        // The steps of an opened trajectory are recorded from its plan: the
        // job ends with the plan calls.
        if !plans.is_empty() {
            return Ok(Progress::Boundary(Resume::Plan(Box::new(PlanResume {
                trajectories,
                descriptors,
                plans,
            }))));
        }
        self.gather_denoise(job, trajectories, descriptors, text, experts)
    }

    /// Record the resumption of each trajectory, whose guidance plan is
    /// open, the gather of its committed version and the solver steps.
    fn gather_denoise(
        &mut self,
        job: &Job,
        trajectories: Vec<(Trajectory, ImageParams, Coordinates)>,
        descriptors: HashMap<usize, Vec<ForwardSlot>>,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let plan = self.plan;
        let stream = self.batch.stream.clone();
        let trajectories: Vec<(Trajectory, ImageParams)> = trajectories
            .into_iter()
            .map(|(trajectory, image, conditioning)| {
                let key = plan.calls[trajectory.index].request_key;
                if let Some(open) = self.parts.trajectories.get(trajectory.slot, key) {
                    open.begin_call(conditioning);
                }
                (trajectory, image)
            })
            .collect();
        let scratch = {
            let values: Vec<Trajectory> = trajectories
                .iter()
                .map(|(value, _)| Trajectory {
                    index: value.index,
                    slot: value.slot,
                    params: value.params.clone(),
                    steps: value.steps,
                })
                .collect();
            self.place(&values, job.on(stream.as_ref()))?
        };

        // Gather each committed version into the step workspace.
        {
            let latents = self.parts.latents()?;
            for ((trajectory, _), scratch) in trajectories.iter().zip(&scratch) {
                let call = &plan.calls[trajectory.index];
                let params = &trajectory.params;
                let generation = call
                    .latent_input
                    .as_ref()
                    .map_or(0, |input| i64::from(input.generation));
                let bank = latents.pool.current_bank(
                    trajectory.slot,
                    i64::from(params.start_step),
                    generation,
                    i64::from(params.latent_units),
                    i64::from(params.height),
                    i64::from(params.width),
                    &scratch.pages,
                )?;
                latents.gather(bank, scratch, job.on(stream.as_ref()))?;
            }
        }

        let layout = *self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
            .layout()?;
        let component = trajectories
            .first()
            .map(|(trajectory, _)| plan.calls[trajectory.index].component.clone())
            .unwrap_or_default();
        let intervals = trajectories
            .iter()
            .map(|(trajectory, _)| trajectory.params.step_count)
            .max()
            .unwrap_or(0);
        self.denoise_steps(
            job,
            Denoising {
                trajectories,
                scratch,
                descriptors,
                layout,
                component,
                stats: ForwardStats::default(),
                intervals,
                offset: 0,
            },
            None,
            text,
            experts,
        )
    }

    /// Record the solver steps from `state.offset` on, then the release of
    /// finished trajectories' guidance prefixes and the record on the text
    /// runners' order. `resumed` is the step whose guidance prefixes the
    /// last job evaluated, with that job's completion.
    fn denoise_steps(
        &mut self,
        job: &Job,
        state: Denoising,
        mut resumed: Option<(Stepping, Fills, Step, &mut Completion)>,
        mut text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let plan = self.plan;
        let stream = self.batch.stream.clone();
        let Denoising {
            trajectories,
            scratch,
            descriptors,
            layout,
            component,
            mut stats,
            intervals,
            offset: first,
        } = state;
        for offset in first..intervals {
            if let Some((stepping, fills, step, completion)) = resumed.take() {
                // The last job evaluated this step's guidance prefixes.
                let text = text.as_deref_mut().ok_or_else(|| {
                    Error::Unsupported("guidance prefixes require the text image".into())
                })?;
                text.complete_prefill(step, completion)?;
                text.order().wait(job.on(stream.as_ref()))?;
                for (position, slot, tokens) in &fills {
                    let (trajectory, _) = &trajectories[*position];
                    let key = plan.calls[trajectory.index].request_key;
                    if let Some(open) = self.parts.trajectories.get(trajectory.slot, key) {
                        open.prefilled(*slot, tokens.len() as u64);
                    }
                }
                self.denoise_step(job, &trajectories, &scratch, &stepping, &mut stats, offset)?;
                continue;
            }

            // Timesteps and guidance branches of this step.
            let mut stepping: Stepping = Vec::new();
            let mut fills: Fills = Vec::new();
            for (position, ((trajectory, _), _)) in trajectories.iter().zip(&scratch).enumerate() {
                let params = &trajectory.params;
                if offset >= params.step_count {
                    continue;
                }
                let step = params.start_step + offset;
                let key = plan.calls[trajectory.index].request_key;
                let call_rows = descriptors
                    .get(&trajectory.index)
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                let prefix_cache = self.storage.prefix_cache.as_ref().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a prefix cache".into())
                })?;
                let tables = prefix_cache.tables();
                let open = self
                    .parts
                    .trajectories
                    .get(trajectory.slot, key)
                    .ok_or_else(|| Error::Invariant("an image trajectory closed".into()))?;
                let (set, time) = open.plan.step(step as usize)?;
                let filled = open.resolve_branches(
                    set,
                    layout.image_unconditional,
                    call_rows,
                    |slot, visible| tables.coordinates(slot, visible).map(Coordinates::from),
                )?;
                fills.extend(
                    filled
                        .into_iter()
                        .map(|fill| (position, fill.slot, fill.tokens)),
                );
                self.parts.latents()?.fill_timestep(
                    trajectory.slot as u32,
                    time,
                    job.on(stream.as_ref()),
                )?;
                stepping.push((position, step, set, time));
            }

            // Missing guidance prefixes are written before the step reads
            // them.
            if !fills.is_empty() {
                let text = text.as_deref_mut().ok_or_else(|| {
                    Error::Unsupported("guidance prefixes require the text image".into())
                })?;
                let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a prefix cache".into())
                })?;
                let prefixes: Vec<(u32, &[u32])> = fills
                    .iter()
                    .map(|(_, slot, tokens)| (*slot, &tokens[..]))
                    .collect();
                // The text runner's evaluation of the prefixes is recorded
                // into this forward's job, behind the block table installs
                // it reads; the job ends with it, because the evaluation
                // is completed from the job that performed it, and the
                // step follows in the next job.
                let step = text.prefill_cache(experts, &component, &prefixes, prefix_cache, job)?;
                return Ok(Progress::Boundary(Resume::Denoise(Box::new(
                    DenoiseResume {
                        state: Denoising {
                            trajectories,
                            scratch,
                            descriptors,
                            layout,
                            component,
                            stats,
                            intervals,
                            offset,
                        },
                        stepping,
                        fills,
                        step,
                    },
                ))));
            }
            self.denoise_step(job, &trajectories, &scratch, &stepping, &mut stats, offset)?;
        }

        // A finished trajectory's guidance prefixes retire with it.
        for (trajectory, _) in &trajectories {
            let params = &trajectory.params;
            if params.start_step + params.step_count < trajectory.steps {
                continue;
            }
            let own = trajectory.slot as u32;
            let mut slots: Vec<u32> = descriptors
                .get(&trajectory.index)
                .map(|rows| {
                    rows.iter()
                        .map(|row| row.slot)
                        .filter(|&slot| slot != own)
                        .collect()
                })
                .unwrap_or_default();
            slots.sort_unstable();
            slots.dedup();
            if slots.is_empty() {
                continue;
            }
            let key = plan.calls[trajectory.index].request_key;
            let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a prefix cache".into())
            })?;
            prefix_cache.release_prefixes(key, &slots, job.on(stream.as_ref()))?;
        }
        if let Some(text) = text {
            text.order_mut().record(job.on(stream.as_ref()))?;
        }
        Ok(Progress::done(stats))
    }

    /// Record one solver step of the stepping trajectories: the rows of
    /// their guidance branches, grouped by image size, each group's
    /// prediction and integration, and for an interval that ends at this
    /// step the scatter of its successor into the other bank with its
    /// export.
    fn denoise_step(
        &mut self,
        job: &Job,
        trajectories: &[(Trajectory, ImageParams)],
        scratch: &[Scratch],
        stepping: &Stepping,
        stats: &mut ForwardStats,
        offset: u32,
    ) -> Result<()> {
        let plan = self.plan;
        let stream = self.batch.stream.clone();
        // One denoising row per branch of every stepping image; images
        // of one size share a numerical call.
        let mut groups: Vec<SizeGroup> = Vec::new();
        for &(position, _, set, time) in stepping {
            let (trajectory, _) = &trajectories[position];
            let key = plan.calls[trajectory.index].request_key;
            let logical = self.batch.calls[trajectory.index]
                .output
                .progress
                .logical_position;
            let open = self
                .parts
                .trajectories
                .get(trajectory.slot, key)
                .ok_or_else(|| Error::Invariant("an image trajectory closed".into()))?;
            let rows = open.branch_rows(set, logical)?;
            let size = (trajectory.params.height, trajectory.params.width);
            let group = match groups.iter().position(|(value, ..)| *value == size) {
                Some(group) => group,
                None => {
                    groups.push((size, Vec::new(), Vec::new()));
                    groups.len() - 1
                }
            };
            let (_, values, owners) = &mut groups[group];
            owners.push((position, values.len()));
            for (_, coordinates, temporal) in rows {
                values.push(DenoiseRow {
                    sample: scratch[position].sample,
                    units: scratch[position].units,
                    height: size.0,
                    width: size.1,
                    slot: coordinates.slot,
                    prefix: u32::try_from(coordinates.visible).map_err(|_| {
                        Error::Invalid("guidance prefix exceeds the token range".into())
                    })?,
                    temporal,
                    timestep: time,
                });
            }
        }
        for (_, rows, owners) in &groups {
            let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a prefix cache".into())
            })?;
            let denoiser = self.parts.denoiser_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a denoiser".into())
            })?;
            let predicted = denoiser.predict(rows, prefix_cache, job)?;
            let mode = predicted.mode.as_str();
            *stats.mode_counts.entry(mode.into()).or_default() += 1;
            if predicted.mode == GraphMode::Replay {
                stats.cuda_graph_replays += 1;
            }
            let mut updates = Vec::with_capacity(owners.len());
            for &(position, first) in owners {
                let (trajectory, _) = &trajectories[position];
                let &(_, step, set, _) = stepping
                    .iter()
                    .find(|(value, ..)| *value == position)
                    .ok_or_else(|| Error::Invariant("a stepping image lost its step".into()))?;
                let key = plan.calls[trajectory.index].request_key;
                let open = self
                    .parts
                    .trajectories
                    .get(trajectory.slot, key)
                    .ok_or_else(|| Error::Invariant("an image trajectory closed".into()))?;
                updates.push(IntegrateRow {
                    sample: scratch[position].sample,
                    units: scratch[position].units,
                    slot: trajectory.slot as u32,
                    step,
                    prediction: predicted.offsets[first],
                    branches: set,
                    params: Arc::clone(&open.plan.params),
                });
            }
            self.parts
                .denoiser()
                .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
                .integrate(&updates, job)?;
        }

        // Intervals ending at this step publish their successor into
        // the other bank.
        for ((trajectory, _), scratch) in trajectories.iter().zip(scratch) {
            if offset + 1 != trajectory.params.step_count {
                continue;
            }
            let latents = self.parts.latents()?;
            let bank = latents.pool.next_bank(trajectory.slot);
            latents.pool.require_writable(bank, &scratch.pages)?;
            latents.scatter(bank, scratch, job.on(stream.as_ref()))?;
            let params = &trajectory.params;
            self.publish_latent(
                trajectory.index,
                bank,
                params.start_step + params.step_count,
                params,
                job.on(stream.as_ref()),
            )?;
            let call = &plan.calls[trajectory.index];
            self.updates.push((
                trajectory.index,
                LatentUpdate::for_call(trajectory.slot, call, &trajectory.params)?,
            ));
        }
        Ok(())
    }
}
