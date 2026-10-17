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

mod denoising;

use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::ImageParams;
use uniserve_cuda::{DeviceId, Stream};

use crate::config::Device;
use crate::cuda_graph::GraphPool;
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::ffi::{DType, Function, Handle, Interpreter, integer_list};
use crate::lane::{Capture, Completion, Job, Value};
use crate::parallel::joins::GraphFunctions;
use crate::request::diffusion::{BranchSet, GuidancePlan};
use crate::runner::ImageSetup;
use crate::runner::attention::{AttentionCapacity, AttentionMetadata};
use crate::runner::denoiser::image_model::ImageDenoiserLayout;
use crate::runner::denoiser::{DENOISER, DenoiserRunner, Family, Methods};
use crate::runner::methods::{Argument, MethodCall};
use crate::storage::AttentionRow;
use crate::storage::PrefixCache;
use crate::storage::memory::{Memory, StorageView};
use crate::stream::Issue;
use crate::{Error, Result};

pub(crate) use denoising::{DenoiseResume, PlanResume, image_placement};

/// The geometry of a captured denoising step, which fixes the first sample
/// row each of its rows reads: `images` images of one size, each evaluating
/// `branches` consecutive guidance rows over its latents, which start at
/// sample row `image * stride` of the step workspace (the placement startup
/// captures and serving uses). Each such sequence of first sample rows has
/// exactly one shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DiffusionShape {
    pub height: u32,
    pub width: u32,
    pub images: u32,
    pub branches: u32,
    /// Sample rows from one image's first latent row to the next image's;
    /// zero for one image.
    pub stride: u32,
}

impl DiffusionShape {
    /// The shape `rows` evaluate, when they are rows of one image size in
    /// that placement. Other rows have no shape: no graph evaluates them.
    pub(crate) fn of(rows: &[DenoiseRow]) -> Option<Self> {
        let first = rows.first()?;
        // The first image's rows share its first sample row, zero; the next
        // image starts at one stride.
        let branches = rows
            .iter()
            .take_while(|row| row.sample == first.sample)
            .count();
        let stride = rows.get(branches).map_or(0, |row| row.sample);
        let placed = first.sample == 0
            && rows.len().is_multiple_of(branches)
            && rows.iter().enumerate().all(|(index, row)| {
                (row.height, row.width) == (first.height, first.width)
                    && u64::from(row.sample) == (index / branches) as u64 * u64::from(stride)
            });
        placed.then(|| Self {
            height: first.height,
            width: first.width,
            images: (rows.len() / branches) as u32,
            branches: branches as u32,
            stride,
        })
    }
}

/// One denoising row: a guidance branch of one image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DenoiseRow {
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
pub(crate) struct ConditionRow {
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
pub(crate) struct IntegrateRow {
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
pub(crate) struct Predicted {
    pub mode: GraphMode,
    /// Each row's first prediction row.
    pub offsets: Vec<u32>,
}

/// Capacity of a runner's columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiffusionCapacity {
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
pub(crate) struct DiffusionStorage {
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
    // SAFETY: a new allocation the denoiser keeps until it closes after
    // its stream drained; its launches run behind the fill on that stream.
    unsafe { view.zero(stream) }?;
    Ok(view)
}

impl DenoiserRunner {
    /// The runner of an image model's denoiser: allocate its columns and
    /// construct the Python half of the denoiser `setup` names over the
    /// prefix cache `cache`. Every step and sample update runs on the
    /// setup's stream.
    pub(crate) fn image(
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
            device.cuda(),
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
    pub(crate) fn layout(&self) -> Result<&ImageDenoiserLayout> {
        Ok(&self.family.image()?.layout)
    }

    /// An image denoiser's column capacity.
    pub(crate) fn capacity(&self) -> Result<&DiffusionCapacity> {
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
    ///
    /// Startup's capture passes the pools of the step's graph: the one on
    /// the runner's device and one on each other device the network's
    /// modules run on (a generation tower). The eager evaluation then
    /// warms the step, gathering the geometry the graph reads, and the
    /// step is captured behind it; the job's completion carries the graph
    /// ([`Self::complete_step`]). A capture is counted when its graph is
    /// stored.
    pub(crate) fn predict(
        &mut self,
        rows: &[DenoiseRow],
        prefix_cache: &mut PrefixCache,
        capture: Option<&[Arc<GraphPool>]>,
        job: &Job,
    ) -> Result<Predicted> {
        let stream = self.stream.clone();
        let issue = job.on(stream.as_ref());
        // The pools a capture records into: the device's, and each peer's.
        let capture = match capture {
            None => None,
            Some(pools) => {
                let (Some(ordinal), Some(stream)) = (self.device.cuda(), &stream) else {
                    return Err(Error::Invariant(
                        "a denoiser captures on its CUDA stream".into(),
                    ));
                };
                let pool = pools
                    .iter()
                    .find(|pool| pool.device == ordinal)
                    .ok_or_else(|| {
                        Error::Invariant("a denoiser graph has no pool on its device".into())
                    })?;
                let peers: Vec<(DeviceId, Handle)> = pools
                    .iter()
                    .filter(|pool| pool.device != ordinal)
                    .map(|pool| (pool.device, pool.value.clone()))
                    .collect();
                Some((pool.value.clone(), peers, stream.raw(), ordinal))
            }
        };
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

        // A captured shape is launched when the rows match it exactly. A
        // captured or capturing step reads its attention at the graphs'
        // widths.
        let captured = match DiffusionShape::of(rows) {
            Some(shape) if image.capturing && image.graphs.contains(&shape, None) => Some(shape),
            _ => None,
        };
        let widths =
            (captured.is_some() || capture.is_some()).then(|| image.capacity.graph_widths.clone());
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
        let mode = match (graph, capture) {
            (Some(graph), _) => {
                if image.plans_attention {
                    job.call(MethodCall::new(&image.plan_denoise, arguments));
                }
                issue.launch(graph)?;
                GraphMode::Replay
            }
            (None, Some((pool, peers, stream, device))) => {
                // The eager evaluation warms the capture, which records the
                // same call without executing it.
                job.call(MethodCall::new(&self.methods.denoise, arguments.clone()));
                job.capture(Capture {
                    functions: self.capture.clone(),
                    pool,
                    peers,
                    stream,
                    device,
                    method: MethodCall::new(&self.methods.denoise, arguments),
                    breaks: None,
                });
                GraphMode::Capture
            }
            (None, None) => {
                job.call(MethodCall::new(&self.methods.denoise, arguments));
                GraphMode::Eager
            }
        };
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

    /// Record into `job` the capture of the denoising step that evaluates
    /// `rows`, behind an eager evaluation of them (its warmup), both as
    /// serving records the step ([`Self::predict`]). `pools` holds the
    /// owner's allocator pool on the runner's device and one on each other
    /// device the network's modules run on (a generation tower). Returns
    /// the step's shape, whose graph [`Self::complete_step`] stores from
    /// the job's completion; `None` for a shape captured already, which
    /// records nothing.
    pub(crate) fn capture_step(
        &mut self,
        rows: &[DenoiseRow],
        prefix_cache: &mut PrefixCache,
        pools: &[Arc<GraphPool>],
        job: &Job,
    ) -> Result<Option<DiffusionShape>> {
        let shape = DiffusionShape::of(rows).ok_or_else(|| {
            Error::Invalid(
                "a captured denoising step evaluates rows of one image size in its placement"
                    .into(),
            )
        })?;
        if self.family.image()?.graphs.contains(&shape, None) {
            return Ok(None);
        }
        self.predict(rows, prefix_cache, Some(pools), job)?;
        Ok(Some(shape))
    }

    /// Store the graph the job of [`Self::capture_step`] captured for
    /// `shape`. `check` runs first (the graph storage budget); a failed
    /// check releases the graph. Serving then launches the captured steps.
    pub(crate) fn complete_step(
        &mut self,
        shape: DiffusionShape,
        completion: &mut Completion,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let graph = completion.captured()?;
        if let Err(error) = check() {
            if let Err(cleanup) = self.capture.release.invoke(&[graph.handle().view()]) {
                tracing::warn!(
                    "{}: releasing a refused graph failed: {cleanup}",
                    self.owner
                );
            }
            return Err(error);
        }
        let image = self.family.image_mut()?;
        image.graphs.insert(shape, None, graph)?;
        image.capturing = true;
        Ok(())
    }

    /// The graph pool's identity: the address of an image denoiser's
    /// positions column, which no other runner shares.
    pub(crate) fn graph_key(&self) -> Result<usize> {
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
        // SAFETY: the denoiser's position and timestep inputs, kept until it
        // closes after its stream drained; the launches reading them run on
        // `stream`, the earlier steps' before these writes and this step's
        // after them.
        unsafe {
            for (axis, values) in axes.iter().enumerate() {
                self.positions
                    .select(0, axis as i64)?
                    .upload(bytemuck::cast_slice(values), stream)?;
            }
            self.timesteps
                .upload(bytemuck::cast_slice(&times), stream)?;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows of 32x32 images reading these first sample rows.
    fn rows(samples: &[u32]) -> Vec<DenoiseRow> {
        samples
            .iter()
            .map(|&sample| DenoiseRow {
                sample,
                units: 4,
                height: 32,
                width: 32,
                slot: 1,
                prefix: 8,
                temporal: 8,
                timestep: 0.5,
            })
            .collect()
    }

    fn shape(images: u32, branches: u32, stride: u32) -> Option<DiffusionShape> {
        Some(DiffusionShape {
            height: 32,
            width: 32,
            images,
            branches,
            stride,
        })
    }

    #[test]
    fn a_step_has_a_shape_only_in_the_captured_placement() {
        // Two images of three branches, four sample rows apart; one image.
        assert_eq!(
            DiffusionShape::of(&rows(&[0, 0, 0, 4, 4, 4])),
            shape(2, 3, 4)
        );
        assert_eq!(DiffusionShape::of(&rows(&[0, 0])), shape(1, 2, 0));

        // Unequal branch counts, unequal strides and a later start read
        // other samples than any captured step.
        assert_eq!(DiffusionShape::of(&rows(&[0, 0, 4])), None);
        assert_eq!(DiffusionShape::of(&rows(&[0, 4, 12])), None);
        assert_eq!(DiffusionShape::of(&rows(&[4, 4])), None);

        // Rows of two image sizes are no captured step either.
        let mut mixed = rows(&[0, 4]);
        mixed[1].height = 64;
        assert_eq!(DiffusionShape::of(&mixed), None);
    }
}
