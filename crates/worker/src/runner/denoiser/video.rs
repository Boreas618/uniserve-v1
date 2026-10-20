//! The video denoiser: a runner's bound buffers, latent pool and layouts
//! ([`VideoDenoiser`]), the text conditioner of the pipeline stage that holds
//! it, and the runner methods that prepare layouts and requests, retain a
//! request's conditioning, advance its trajectory by one solver step, and warm
//! and capture the steps of startup layouts.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::Canvas;
use uniserve_cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use uniserve_ipc::{Call, LatentParams};

use super::{
    DENOISER, DenoiserRunner, ENCODER, Family, LayoutFields, Methods, SampleLayout, SamplePages,
    SampleRows, StateEnvelope, VideoModel, size_values,
};
use crate::config::Device;
use crate::cuda_graph::GraphPool;
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::ffi::{Backing, DType, Function, Handle, Interpreter, integer_list};
use crate::lane::{Capture, Completion, Job};
use crate::parallel::joins::GraphFunctions;
use crate::request::video::{VideoLayout, VideoSize};
use crate::runner::methods::{Argument, MethodCall};
use crate::storage::memory::{Storage, StorageView};
use crate::storage::{LatentPlan, RequestSlots, TrajectoryVersion};
use crate::stream::{Fill, Issue};
use crate::{Error, LatentPool, LatentStorage, LatentUpdate, Result};

/// The text conditioner of a pipeline stage that holds it: one Python
/// `EncoderRunner` per prompt capacity over the same input and output.
pub(super) struct Conditioner {
    /// `[capacity, width]` prompt features, zero past the prompt.
    input: StorageView,
    /// `[1]` int32 prompt length.
    lengths: StorageView,
    /// `[capacity, conditioning width]` BF16 refined rows.
    output: StorageView,
    /// The runner of each prompt capacity.
    pub(super) runners: BTreeMap<u32, Handle>,
    encode: Function,
    pub(super) kernels: Function,
    pub(super) close: Function,
    pub(super) graphs: GraphTable<u32>,
}

impl Conditioner {
    /// The arguments of `EncoderRunner.encode` at `capacity`.
    fn arguments(&self, capacity: u32) -> Result<Vec<Argument>> {
        let runner = self.runners.get(&capacity).ok_or_else(|| {
            Error::Invariant(format!("no text conditioner serves capacity {capacity}"))
        })?;
        Ok(vec![
            Argument::Handle(runner.clone()),
            Argument::Tensor(self.input.prefix(i64::from(capacity))?),
            Argument::Tensor(self.lengths.clone()),
            Argument::Tensor(self.output.prefix(i64::from(capacity))?),
        ])
    }
}

/// How a prepared layout serves steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Prepared,
    Warmed,
}

struct Layout {
    key: VideoLayout,
    /// The layout's prepared constants, as the Python half returned them.
    handle: Handle,
    fields: LayoutFields,
    samples: SampleLayout,
    /// The tensors `DenoiserRunner.denoise` takes for the layout, every
    /// state field in state order: a modality's local sample in the
    /// scratch, a table's fixed state buffer.
    tensors: Vec<StorageView>,
    /// State tables copied from the slot's rows before each step, as
    /// `(field, bytes)`.
    copies: Vec<(usize, u64)>,
    phase: Phase,
    startup: bool,
    /// Live requests evaluating in the layout.
    references: usize,
}

/// The transient layouts a new one displaced ([`DenoiserRunner::ask_layout`]):
/// their constants are released when this is dropped, behind the job that
/// drained the stream.
pub(crate) struct Evicted {
    /// Held, never read, until the value is dropped.
    _layouts: Vec<Layout>,
}

/// A video denoiser's state: its bound buffers, latent pool and layouts.
pub(crate) struct VideoDenoiser {
    /// `DenoiserRunner.prepare_layout`, which only a video denoiser has.
    prepare_layout: Function,
    /// `[elements]` FP32 scratch one request's local samples occupy during
    /// a step.
    samples: StorageView,
    /// `[1]` int64 evaluation index.
    step: StorageView,
    /// Fixed flat state buffers, in the envelope's device field order.
    state: Vec<StorageView>,
    banks: RequestSlots,
    host: RequestSlots,
    envelope: StateEnvelope,
    pub(super) conditioner: Option<Conditioner>,
    pages: SamplePages,
    storage: LatentStorage,
    /// The requests' trajectories. A video trajectory stays on its rank:
    /// its final latents leave as tensor products.
    pool: LatentPool,
    layouts: Vec<Option<Layout>>,
    ids: HashMap<VideoLayout, usize>,
    /// Transient layout ids, least recently used first.
    transient: VecDeque<usize>,
    transient_capacity: usize,
    pub(super) graphs: GraphTable<usize>,
    /// The graph pool captures allocate in; `None` serves steps eagerly.
    pool_handle: Option<Arc<GraphPool>>,
}

/// The text conditioner a denoising rank holds.
pub(crate) struct Conditioning {
    /// The conditioner module's path in the model.
    pub(crate) path: String,
    /// Prompt capacities the conditioner serves.
    pub(crate) capacities: Vec<u32>,
    /// Columns and dtype of the prompt features it refines.
    pub(crate) width: u64,
    pub(crate) dtype: DType,
}

/// What [`DenoiserRunner::video`] is built from.
pub(crate) struct DenoiserSetup<'a> {
    pub(crate) loaded: &'a Handle,
    pub(crate) component: &'a str,
    pub(crate) stream: Stream,
    pub(crate) device: DeviceId,
    pub(crate) maximum: VideoLayout,
    pub(crate) envelope: StateEnvelope,
    pub(crate) pages: SamplePages,
    /// The sample pool the rank's capacity planned.
    pub(crate) latent: LatentPlan,
    pub(crate) slots: u32,
    /// `None` without a conditioner on this rank.
    pub(crate) conditioning: Option<Conditioning>,
    pub(crate) conditioning_width: u64,
    pub(crate) attention: String,
    /// The stream's shared work areas, a Python `Scratch`.
    pub(crate) scratch: Handle,
    pub(crate) graph_pool: Option<Arc<GraphPool>>,
    pub(crate) transient_layouts: usize,
}

fn bytes_of(view: &StorageView) -> u64 {
    view.numel() * view.dtype().bytes() as u64
}

impl DenoiserRunner {
    /// The runner of a video model's denoiser: allocate its buffers and
    /// latent pool and bind them.
    pub(crate) fn video(setup: DenoiserSetup<'_>) -> Result<Self> {
        let device = setup.device;
        let memory = |bytes: u64| Storage::device(device, bytes.max(1) as usize);
        let samples = StorageView::contiguous(
            memory(setup.envelope.samples * 4)? as Arc<dyn Backing>,
            0,
            DType::F32,
            &[setup.envelope.samples as i64],
        )?;
        let step = StorageView::contiguous(memory(8)? as Arc<dyn Backing>, 0, DType::I64, &[1])?;
        let state = setup
            .envelope
            .device
            .iter()
            .map(|field| {
                StorageView::contiguous(
                    memory(field.shape[0] as u64 * field.dtype.bytes() as u64)? as Arc<dyn Backing>,
                    0,
                    field.dtype,
                    &field.shape,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let banks = RequestSlots::allocate(&setup.envelope.device, setup.slots, Some(device))?;
        let host = RequestSlots::allocate(&setup.envelope.host, setup.slots, Some(device))?;
        let latent = setup.latent;
        let storage = LatentStorage::allocate(&latent, Some(device), Some(&setup.stream))?;
        let pool = LatentPool::new(
            latent.request_pool_size,
            latent.num_pages,
            latent.page_units,
        )?;

        let text = |value: &str| Any::from(tvm_ffi::String::from(value));
        let stream = Any::from(setup.stream.raw() as i64);
        let runner = Interpreter::import(DENOISER)?.call_values(&[
            setup.loaded.to_any(),
            text(setup.component),
            stream.clone(),
            text(&setup.attention),
            setup.scratch.to_any(),
            setup.maximum.descriptor(),
        ])?;
        let methods = Methods::import(Handle::new(runner)?)?;
        let prepare_layout = Interpreter::import(&format!("{DENOISER}.prepare_layout"))?;
        let capture = GraphFunctions::import()?;

        let conditioner = match setup.conditioning {
            Some(conditioning) => {
                let Conditioning {
                    path,
                    mut capacities,
                    width,
                    dtype,
                } = conditioning;
                let rows = i64::from(capacities.iter().copied().max().unwrap_or(0));
                let construct = Interpreter::import(ENCODER)?;
                // The largest capacity is prepared first, so it sizes the
                // workspace the capacities share.
                capacities.sort_unstable_by(|left, right| right.cmp(left));
                let mut runners = BTreeMap::new();
                for capacity in capacities {
                    let runner = construct.call_values(&[
                        setup.loaded.to_any(),
                        text(&path),
                        stream.clone(),
                        text(&setup.attention),
                        setup.scratch.to_any(),
                        Any::from(integer_list([i64::from(capacity), width as i64])),
                    ])?;
                    runners.insert(capacity, Handle::new(runner)?);
                }
                Some(Conditioner {
                    input: StorageView::contiguous(
                        memory(rows as u64 * width * dtype.bytes() as u64)? as Arc<dyn Backing>,
                        0,
                        dtype,
                        &[rows, width as i64],
                    )?,
                    lengths: StorageView::contiguous(
                        memory(4)? as Arc<dyn Backing>,
                        0,
                        DType::I32,
                        &[1],
                    )?,
                    output: StorageView::contiguous(
                        memory(rows as u64 * setup.conditioning_width * 2)? as Arc<dyn Backing>,
                        0,
                        DType::BF16,
                        &[rows, setup.conditioning_width as i64],
                    )?,
                    runners,
                    encode: Interpreter::import(&format!("{ENCODER}.encode"))?,
                    kernels: Interpreter::import(&format!("{ENCODER}.kernels"))?,
                    close: Interpreter::import(&format!("{ENCODER}.close"))?,
                    graphs: GraphTable::new(
                        format!("{}.conditioner", setup.component),
                        capture.release.clone(),
                    ),
                })
            }
            None => None,
        };
        let graphs = GraphTable::new(
            format!("{}.denoiser", setup.component),
            capture.release.clone(),
        );
        Ok(Self {
            owner: setup.component.to_owned(),
            device: Device::Cuda(device),
            stream: Some(setup.stream),
            methods,
            capture,
            family: Family::Video(Box::new(VideoDenoiser {
                prepare_layout,
                samples,
                step,
                state,
                banks,
                host,
                envelope: setup.envelope,
                conditioner,
                pages: setup.pages,
                storage,
                pool,
                layouts: Vec::new(),
                ids: HashMap::new(),
                transient: VecDeque::new(),
                transient_capacity: setup.transient_layouts.max(1),
                graphs,
                pool_handle: setup.graph_pool,
            })),
        })
    }
}

impl DenoiserRunner {
    /// Prepare startup layout `key` under a new id: read its fields and
    /// prepare its constants, on the calling thread. Startup layouts are
    /// never released.
    fn prepare(&mut self, model: &VideoModel, key: VideoLayout) -> Result<usize> {
        let fields = model.layout_fields(&key)?;
        let video = self.family.video_mut()?;
        let handle = Handle::new(
            video
                .prepare_layout
                .call_values(&[self.methods.runner.to_any(), key.descriptor()])?,
        )?;
        video.install(key, fields, handle, true)
    }

    /// Prepare a startup layout once; the largest is prepared first so it
    /// sizes the shared workspace.
    pub(crate) fn prepare_startup(
        &mut self,
        model: &VideoModel,
        key: VideoLayout,
    ) -> Result<usize> {
        let prepared = self.family.video()?.ids.get(&key).copied();
        match prepared {
            Some(id) => Ok(id),
            None => self.prepare(model, key),
        }
    }

    /// Whether layout `key` is prepared.
    pub(crate) fn prepared(&self, key: &VideoLayout) -> bool {
        self.family
            .video()
            .is_ok_and(|video| video.ids.contains_key(key))
    }

    /// Hold prepared layout `key` for one more request and return its id;
    /// `None` when the layout is not prepared ([`Self::ask_layout`]).
    pub(crate) fn acquire(&mut self, key: VideoLayout) -> Result<Option<usize>> {
        let video = self.family.video_mut()?;
        let Some(&id) = video.ids.get(&key) else {
            return Ok(None);
        };
        if let Some(position) = video.transient.iter().position(|&entry| entry == id) {
            video.transient.remove(position);
            video.transient.push_back(id);
        }
        video.layout_mut(id)?.references += 1;
        Ok(Some(id))
    }

    /// Record into `job`, a job of the runner's lane, the preparation of
    /// transient layout `key`: the model's fields of the layout and the
    /// constants the library prepares for it on the lane's stream.
    /// [`Self::accept_layout`] keeps the layout from the job's completion.
    ///
    /// Unused transient layouts beyond the runner's capacity give way.
    /// Their constants must outlive their last step, so the job drains
    /// the stream first; the returned layouts are dropped once it
    /// completed.
    pub(crate) fn ask_layout(
        &mut self,
        model: &VideoModel,
        key: VideoLayout,
        job: &Job,
    ) -> Result<Evicted> {
        let (_, stream) = self.cuda()?;
        let video = self.family.video_mut()?;
        let mut evicted = Vec::new();
        while video.transient.len() >= video.transient_capacity {
            let Some(position) = video.transient.iter().position(|&entry| {
                video
                    .layout(entry)
                    .is_ok_and(|layout| layout.references == 0)
            }) else {
                break;
            };
            let Some(id) = video.transient.remove(position) else {
                break;
            };
            if let Some(layout) = video.layouts.get_mut(id).and_then(Option::take) {
                video.ids.remove(&layout.key);
                evicted.push(layout);
            }
        }
        if !evicted.is_empty() {
            job.on(Some(&stream)).synchronize()?;
        }

        model.ask_layout_fields(&key, job);
        job.call_handle(MethodCall::new(
            &video.prepare_layout,
            vec![
                Argument::Handle(self.methods.runner.clone()),
                Argument::Integers(key.values().to_vec()),
            ],
        ));
        Ok(Evicted { _layouts: evicted })
    }

    /// Keep transient layout `key` as the job of [`Self::ask_layout`]
    /// prepared it.
    pub(crate) fn accept_layout(
        &mut self,
        model: &VideoModel,
        key: VideoLayout,
        completion: &mut Completion,
    ) -> Result<()> {
        let fields = model.read_layout_fields(completion)?;
        let handle = completion.handle()?;
        let video = self.family.video_mut()?;
        let id = video.install(key, fields, handle, false)?;
        video.transient.push_back(id);
        Ok(())
    }

    /// End a request's use of its layout.
    pub(crate) fn release(&mut self, id: usize) -> Result<()> {
        let layout = self.family.video_mut()?.layout_mut(id)?;
        layout.references = layout
            .references
            .checked_sub(1)
            .ok_or_else(|| Error::Invariant("video layout reference underflow".into()))?;
        Ok(())
    }

    /// The layout a prepared id evaluates.
    pub(crate) fn key(&self, id: usize) -> Result<VideoLayout> {
        Ok(self.family.video()?.layout(id)?.key)
    }

    /// Draw a request's noise from `seed` and fill its slot's host sources
    /// for its layout: the library's draw fills the slot's pinned noise
    /// rows, and `DenoiserRunner.prepare_latents` turns them into the
    /// request's initial samples and state tables. Host work only, recorded
    /// into `job` so that it precedes the copies of the slot's sources.
    pub(crate) fn prepare_latents(
        &self,
        id: usize,
        slot: usize,
        size: &VideoSize,
        seed: u64,
        job: &Job,
    ) -> Result<()> {
        let video = self.family.video()?;
        let layout = video.layout(id)?;
        let host = |name: String| -> Result<Argument> {
            Ok(Argument::Tensor(video.host.slot(&name, slot as u32)?))
        };
        let noise = layout
            .fields
            .samples
            .iter()
            .map(|(name, _)| host(format!("{name}_noise")))
            .collect::<Result<Vec<_>>>()?;

        // The seed's bits; the Python half reads them as unsigned.
        let runner = Argument::Handle(self.methods.runner.clone());
        let mut draw = vec![
            runner.clone(),
            Argument::Integer(seed as i64),
            Argument::Handle(layout.handle.clone()),
        ];
        draw.extend(size_values(size));
        draw.push(host("condition_noise".into())?);
        draw.extend(noise.iter().cloned());
        job.call(MethodCall::new(&self.methods.draw_noise, draw));

        let mut prepare = vec![runner, Argument::Handle(layout.handle.clone())];
        prepare.extend(size_values(size));
        prepare.extend(noise);
        for field in &layout.fields.state {
            prepare.push(host(format!("{}_source", field.name))?);
        }
        job.call(MethodCall::new(&self.methods.prepare_latents, prepare));
        Ok(())
    }

    /// Copy a request's prepared host sources into its slot: each
    /// modality's sample into bank 1 of its pages and each table into its
    /// slot rows. The host task that filled them has completed. `canvas` is
    /// the request's admitted canvas, the raster its samples decode to.
    pub(crate) fn initialize(
        &mut self,
        id: usize,
        slot: usize,
        call: &Call,
        canvas: Canvas,
        job: &Job,
    ) -> Result<LatentUpdate> {
        let (_, stream) = self.cuda()?;
        let video = self.family.video_mut()?;
        let params = video.trajectory(slot, call, canvas);
        let pages = video.slot_pages(&params)?;
        video.pool.require_initial(slot, &pages)?;
        let layout = video.layout(id)?;
        let first = pages[0] as u64;
        let base = video.bank_address(1, first)?;
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (index, (name, _)) in layout.fields.samples.iter().enumerate() {
            let source = video.host.slot(&format!("{name}_source"), slot as u32)?;
            copies.push(
                base + layout.samples.offsets[index] * 4,
                source.address(),
                (layout.samples.elements[index] * 4) as usize,
            );
        }
        for field in layout.fields.state.iter().filter(|field| !field.sample) {
            let source = video
                .host
                .slot(&format!("{}_source", field.name), slot as u32)?;
            let target = video.banks.slot(&field.name, slot as u32)?;
            copies.push(target.address(), source.address(), field.bytes() as usize);
        }
        // SAFETY: every destination lies in this slot's pages or rows, which
        // no other request owns; the pinned sources stay allocated for the
        // runner's lifetime and are rewritten only by the slot's next request.
        unsafe { job.on(Some(&stream)).copy(&copies) }?;
        LatentUpdate::for_call(slot, call, &params, 0)
    }

    /// The step index of `slot`'s committed trajectory, from the latent
    /// pool: the step its next denoising call advances.
    pub(crate) fn step_index(&self, slot: usize) -> Result<i64> {
        Ok(self.family.video()?.pool.step_index(slot))
    }

    /// Whether this rank retains a request's conditioning: whether it holds
    /// the text conditioner. A model declares the conditioner on the
    /// pipeline stage whose forward reads the retained conditioning (the
    /// first); a later stage receives that stage's hidden rows, the refined
    /// prompt among them, through the forward's pipeline communication, so
    /// it neither refines the prompt nor supplies conditioning to its steps.
    pub(crate) fn conditions(&self) -> bool {
        matches!(&self.family, Family::Video(video) if video.conditioner.is_some())
    }

    /// Refine a request's prompt features and retain them in its slot's
    /// conditioning: `features` holds the prompt's `count` rows; the rows of
    /// the layout's conditioning past the prompt become zero. Only a rank
    /// that [`Self::conditions`] refines.
    pub(crate) fn condition(
        &mut self,
        id: usize,
        slot: usize,
        features: &StorageView,
        count: u64,
        job: &Job,
    ) -> Result<GraphMode> {
        let (_, stream) = self.cuda()?;
        let video = self.family.video_mut()?;
        let layout = video.layout(id)?;
        let (rows, width) = layout.fields.text_condition;
        let target = video.banks.slot("text_condition", slot as u32)?;
        let Some(buffers) = &video.conditioner else {
            return Err(Error::Invariant(format!(
                "{}: a rank without the text conditioner refines no prompt",
                self.owner
            )));
        };
        let capacity = buffers
            .runners
            .keys()
            .copied()
            .find(|&capacity| u64::from(capacity) >= count)
            .ok_or_else(|| {
                Error::Invalid("prompt exceeds the configured text capacities".into())
            })?;
        let feature_width = buffers.input.shape()[1] as u64;
        let element = buffers.input.dtype().bytes() as u64;
        if features.dtype() != buffers.input.dtype()
            || features.shape() != [count as i64, feature_width as i64]
            || !features.is_contiguous()
        {
            return Err(Error::Invalid(
                "prompt features disagree with the conditioner's input".into(),
            ));
        }
        let row_bytes = feature_width * element;
        let output_row = width * 2;
        let (input, lengths, output) = (
            buffers.input.address(),
            buffers.lengths.address(),
            buffers.output.address(),
        );
        let issue = job.on(Some(&stream));
        let mut prompt = CopyBatch::new(SourceOrder::Stream);
        prompt.push(input, features.address(), (count * row_bytes) as usize);
        // SAFETY: the conditioner buffers and the slot's rows are owned by
        // this runner and the features are leased until the batch commits;
        // every range lies inside them and the stream orders the copies
        // before and after the conditioner.
        unsafe {
            issue.copy(&prompt)?;
            issue.fill(Fill {
                address: input + count * row_bytes,
                value: 0,
                width: 1,
                count: ((u64::from(capacity) - count) * row_bytes) as usize,
            })?;
            issue.fill(Fill {
                address: lengths,
                value: count as u32,
                width: 4,
                count: 1,
            })?;
        }
        let mode = video.run_conditioner(&stream, capacity, None, job)?;

        let mut refined = CopyBatch::new(SourceOrder::Stream);
        refined.push(target.address(), output, (count * output_row) as usize);
        // SAFETY: as above.
        unsafe {
            issue.copy(&refined)?;
            issue.fill(Fill {
                address: target.address() + count * output_row,
                value: 0,
                width: 1,
                count: ((rows - count) * output_row) as usize,
            })?;
        }
        Ok(mode)
    }

    /// Write a request's encoded conditions into its slot's conditioning,
    /// after the prompt's rows. `latents` are the condition latents in
    /// request order, a condition's visual rows before its audio rows, each
    /// as the row tensors that together hold its rows in order.
    pub(crate) fn encode_conditions(
        &self,
        id: usize,
        slot: usize,
        size: &VideoSize,
        latents: &[Vec<StorageView>],
        job: &Job,
    ) -> Result<()> {
        let video = self.family.video()?;
        let layout = video.layout(id)?;
        let mut arguments = vec![
            Argument::Handle(self.methods.runner.clone()),
            Argument::Handle(layout.handle.clone()),
        ];
        arguments.extend(size_values(size));
        arguments.push(Argument::Tensor(
            video.host.slot("condition_noise", slot as u32)?,
        ));
        arguments.push(Argument::Tensor(
            video.banks.slot("text_condition", slot as u32)?,
        ));
        arguments.push(Argument::Integers(
            latents.iter().map(|parts| parts.len() as i64).collect(),
        ));
        for part in latents.iter().flatten() {
            arguments.push(Argument::Tensor(part.clone()));
        }
        job.call(MethodCall::new(&self.methods.encode_conditions, arguments));
        Ok(())
    }

    /// Advance a request's trajectory by one solver step, from the step the
    /// latent pool committed, and return its commit. `canvas` is the
    /// request's admitted canvas.
    pub(crate) fn denoise(
        &mut self,
        id: usize,
        slot: usize,
        call: &Call,
        canvas: Canvas,
        job: &Job,
    ) -> Result<(LatentUpdate, GraphMode)> {
        let (_, stream) = self.cuda()?;
        let video = self.family.video_mut()?;
        let params = video.trajectory(slot, call, canvas);
        let pages = video.slot_pages(&params)?;
        let step_index = video.pool.step_index(slot);
        let evaluation = u32::try_from(step_index)
            .map_err(|_| Error::Invariant("a committed latent step exceeds u32".into()))?;
        let trajectory = TrajectoryVersion {
            step_index,
            generation: step_index + 1,
            units: i64::from(params.latent_units),
            height: i64::from(params.height),
            width: i64::from(params.width),
        };
        let read = video.pool.current_bank(slot, &trajectory, &pages)?;
        let write = 1 - read;
        video.pool.require_writable(write, &pages)?;
        let first = pages[0] as u64;
        let (copies, layout_pages) = {
            let layout = video.layout(id)?;
            (layout.copies.clone(), layout.samples.pages)
        };
        let page_bytes = video.pages.page_units * 4;

        let mut gather = CopyBatch::new(SourceOrder::Stream);
        gather.push(
            video.samples.address(),
            video.bank_address(read, first)?,
            (layout_pages * page_bytes) as usize,
        );
        for &(index, bytes) in &copies {
            let name = &video.envelope.device[index].name;
            gather.push(
                video.state[index].address(),
                video.banks.slot(name, slot as u32)?.address(),
                bytes as usize,
            );
        }
        let issue = job.on(Some(&stream));
        // SAFETY: the scratch, state buffers, slot rows and pages are owned by
        // this runner; steps of the runner are ordered on its stream.
        unsafe {
            issue.copy(&gather)?;
            issue.fill(Fill {
                address: video.step.address(),
                value: evaluation,
                width: 4,
                count: 1,
            })?;
            issue.fill(Fill {
                address: video.step.address() + 4,
                value: 0,
                width: 4,
                count: 1,
            })?;
        }

        let mode = video.run_step(&self.methods, &stream, id, None, job)?;

        let mut scatter = CopyBatch::new(SourceOrder::Stream);
        scatter.push(
            video.bank_address(write, first)?,
            video.samples.address(),
            (layout_pages * page_bytes) as usize,
        );
        // SAFETY: as above; the scatter follows the step on the stream.
        unsafe { issue.copy(&scatter) }?;
        Ok((
            LatentUpdate::for_call(slot, call, &params, step_index)?,
            mode,
        ))
    }

    /// The final step's local sample of each modality, as `[rows, width]`
    /// views of the scratch (`None` for a modality this rank holds no rows
    /// of), in modality order.
    pub(crate) fn final_samples(&self, id: usize) -> Result<Vec<Option<StorageView>>> {
        let video = self.family.video()?;
        let layout = video.layout(id)?;
        layout
            .fields
            .samples
            .iter()
            .enumerate()
            .map(|(index, (_, rows))| {
                let local = rows.local.1 - rows.local.0;
                if local == 0 {
                    return Ok(None);
                }
                let offset = video.samples.offset() + (layout.samples.offsets[index] * 4) as usize;
                StorageView::contiguous(
                    Arc::clone(video.samples.memory()),
                    offset,
                    DType::F32,
                    &[local as i64, rows.width as i64],
                )
                .map(Some)
            })
            .collect()
    }

    /// The local rows of each modality's complete sample in a layout.
    pub(crate) fn sample_rows(&self, id: usize) -> Result<Vec<SampleRows>> {
        Ok(self
            .family
            .video()?
            .layout(id)?
            .fields
            .samples
            .iter()
            .map(|(_, rows)| *rows)
            .collect())
    }

    /// Validate trajectory commits before any becomes visible.
    pub(crate) fn validate(&self, updates: &[LatentUpdate]) -> Result<()> {
        self.family.video()?.pool.validate_updates(updates)
    }

    /// Make validated commits visible; only a video denoiser's commits
    /// validate ([`Self::validate`]).
    pub(crate) fn apply(&mut self, updates: &[LatentUpdate]) {
        if let Family::Video(video) = &mut self.family {
            drop(video.pool.apply_updates(updates));
        }
    }

    /// Release retired requests' pages.
    pub(crate) fn release_slots(&mut self, slots: &[usize]) -> Result<()> {
        let video = self.family.video_mut()?;
        let slots: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
        drop(video.pool.release_slots(&slots)?);
        video.pool.reap();
        Ok(())
    }

    /// Record into `job` one placeholder step of startup layout `id` on
    /// slot `slot`, whose host sources the job fills before
    /// ([`Self::prepare_latents`]), to prepare its kernels and plans before
    /// any capture: the copies of the sources into the step's scratch and
    /// state buffers, the zero conditioning and evaluation index, the step
    /// as serving records it, then the wait for the stream, after which
    /// the slot's sources may be rewritten. The scratch's values are
    /// discarded.
    pub(crate) fn warm(&mut self, id: usize, slot: usize, job: &Job) -> Result<()> {
        let (_, stream) = self.cuda()?;
        let issue = job.on(Some(&stream));
        let video = self.family.video_mut()?;
        let layout = video.layout(id)?;
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (index, (name, _)) in layout.fields.samples.iter().enumerate() {
            let source = video.host.slot(&format!("{name}_source"), slot as u32)?;
            copies.push(
                video.samples.address() + layout.samples.offsets[index] * 4,
                source.address(),
                (layout.samples.elements[index] * 4) as usize,
            );
        }
        for field in layout.fields.state.iter().filter(|field| !field.sample) {
            let source = video
                .host
                .slot(&format!("{}_source", field.name), slot as u32)?;
            let index = video
                .envelope
                .device
                .iter()
                .position(|candidate| candidate.name == field.name)
                .ok_or_else(|| Error::Invariant("layout field left the envelope".into()))?;
            copies.push(
                video.state[index].address(),
                source.address(),
                field.bytes() as usize,
            );
        }
        let text = video
            .envelope
            .device
            .iter()
            .position(|field| field.name == "text_condition")
            .ok_or_else(|| Error::Invariant("the envelope lacks text conditioning".into()))?;
        let text_bytes = bytes_of(&video.state[text]);
        // SAFETY: the scratch and state buffers are owned by this runner;
        // the pinned sources are rewritten only by a later job, which the
        // wait below orders behind these copies.
        unsafe {
            issue.copy(&copies)?;
            issue.fill(Fill {
                address: video.state[text].address(),
                value: 0,
                width: 1,
                count: text_bytes as usize,
            })?;
            issue.fill(Fill {
                address: video.step.address(),
                value: 0,
                width: 4,
                count: 2,
            })?;
        }
        // No layout is captured before every one is warmed: the step is the
        // eager call.
        video.run_step(&self.methods, &stream, id, None, job)?;
        issue.synchronize()?;
        // The job is performed before any capture is recorded.
        video.layout_mut(id)?.phase = Phase::Warmed;
        Ok(())
    }

    /// Record into `job` the capture of a warmed startup layout's step into
    /// the runner's graph pool; whether it records one: a runner without a
    /// graph pool serves steps eagerly, and a captured layout is not
    /// captured again. [`Self::complete_capture`] stores the graph from the
    /// job's completion.
    pub(crate) fn capture(&mut self, id: usize, job: &Job) -> Result<bool> {
        let (ordinal, stream) = self.cuda()?;
        let video = self.family.video_mut()?;
        if video.pool_handle.is_none() {
            return Ok(false);
        }
        let layout = video.layout(id)?;
        if !layout.startup || layout.phase != Phase::Warmed {
            return Err(Error::Invariant(
                "only warmed startup layouts are captured".into(),
            ));
        }
        if video.graphs.contains(&id, None) {
            return Ok(false);
        }
        let mode = video.run_step(
            &self.methods,
            &stream,
            id,
            Some((&self.capture, ordinal)),
            job,
        )?;
        Ok(mode == GraphMode::Capture)
    }

    /// Store the step graph the job of [`Self::capture`] captured for
    /// layout `id`.
    pub(crate) fn complete_capture(
        &mut self,
        id: usize,
        completion: &mut Completion,
    ) -> Result<()> {
        let video = self.family.video_mut()?;
        video.graphs.insert(id, None, completion.captured()?)?;
        Ok(())
    }

    /// Record into `job` the conditioner's warmup at every capacity
    /// (largest first) on zero features, each forward as serving records
    /// it, then the wait for the stream; with `capture`, then the capture
    /// of each capacity. Returns the captured capacities, in order, whose
    /// graphs [`Self::complete_conditioner`] stores from the job's
    /// completion.
    pub(crate) fn warm_conditioner(&mut self, capture: bool, job: &Job) -> Result<Vec<u32>> {
        let (ordinal, stream) = self.cuda()?;
        let issue = job.on(Some(&stream));
        let video = self.family.video_mut()?;
        let Some(conditioner) = &video.conditioner else {
            return Ok(Vec::new());
        };
        let (input, input_bytes, lengths) = (
            conditioner.input.address(),
            bytes_of(&conditioner.input),
            conditioner.lengths.address(),
        );
        let capacities: Vec<u32> = conditioner.runners.keys().rev().copied().collect();
        // SAFETY: the conditioner's buffers belong to this runner.
        unsafe {
            issue.fill(Fill {
                address: input,
                value: 0,
                width: 1,
                count: input_bytes as usize,
            })?;
        }
        for &capacity in &capacities {
            // SAFETY: as above.
            unsafe {
                issue.fill(Fill {
                    address: lengths,
                    value: capacity,
                    width: 4,
                    count: 1,
                })?;
            }
            // No capacity is captured before every one is warmed: the
            // forward is the eager call.
            video.run_conditioner(&stream, capacity, None, job)?;
        }
        issue.synchronize()?;
        if !capture || video.pool_handle.is_none() {
            return Ok(Vec::new());
        }
        for &capacity in &capacities {
            video.run_conditioner(&stream, capacity, Some((&self.capture, ordinal)), job)?;
        }
        Ok(capacities)
    }

    /// Store the conditioner graphs the job of [`Self::warm_conditioner`]
    /// captured for `capacities`, in their order.
    pub(crate) fn complete_conditioner(
        &mut self,
        capacities: Vec<u32>,
        completion: &mut Completion,
    ) -> Result<()> {
        if capacities.is_empty() {
            return Ok(());
        }
        let conditioner = self
            .family
            .video_mut()?
            .conditioner
            .as_mut()
            .ok_or_else(|| Error::Invariant("a captured conditioner is gone".into()))?;
        for capacity in capacities {
            conditioner
                .graphs
                .insert(capacity, None, completion.captured()?)?;
        }
        Ok(())
    }
}

impl VideoDenoiser {
    /// Keep layout `key`, whose constants `handle` holds, under a new id.
    fn install(
        &mut self,
        key: VideoLayout,
        fields: LayoutFields,
        handle: Handle,
        startup: bool,
    ) -> Result<usize> {
        let samples = SampleLayout::new(&fields, self.pages.page_units)?;
        if samples.pages > self.pages.pages {
            return Err(Error::Invalid(
                "a video layout's samples exceed the request's latent pages".into(),
            ));
        }
        let mut copies = Vec::new();
        // A step reads the slot's retained conditioning only where the
        // conditioner retains it (`DenoiserRunner::conditions`); on a later
        // pipeline stage the step's conditioning buffer keeps the zeros its
        // warmup wrote, and its forward reads none of it.
        let retained = self.conditioner.as_ref().map(|_| {
            (
                "text_condition".to_owned(),
                fields.text_condition.0 * fields.text_condition.1 * 2,
            )
        });
        let tables = fields
            .state
            .iter()
            .filter(|field| !field.sample)
            .map(|field| (field.name.clone(), field.bytes()))
            .chain(retained);
        for (name, bytes) in tables {
            let index = self
                .envelope
                .device
                .iter()
                .position(|field| field.name == name)
                .ok_or_else(|| {
                    Error::Invalid(format!("layout state field {name} exceeds the slot state"))
                })?;
            if bytes > bytes_of(&self.state[index]) {
                return Err(Error::Invalid(format!(
                    "layout state field {name} exceeds its slot capacity"
                )));
            }
            copies.push((index, bytes));
        }
        let id = match self.layouts.iter().position(Option::is_none) {
            Some(free) => free,
            None => {
                self.layouts.push(None);
                self.layouts.len() - 1
            }
        };
        let mut tensors = Vec::with_capacity(fields.state.len());
        let mut modality = 0;
        for field in &fields.state {
            if field.sample {
                tensors.push(StorageView::contiguous(
                    Arc::clone(self.samples.memory()),
                    self.samples.offset() + (samples.offsets[modality] * 4) as usize,
                    DType::F32,
                    &[samples.elements[modality] as i64],
                )?);
                modality += 1;
            } else {
                let index = self
                    .envelope
                    .device
                    .iter()
                    .position(|candidate| candidate.name == field.name)
                    .ok_or_else(|| Error::Invariant("layout field left the envelope".into()))?;
                tensors.push(self.state[index].clone());
            }
        }
        self.layouts[id] = Some(Layout {
            key,
            handle,
            fields,
            samples,
            tensors,
            copies,
            phase: Phase::Prepared,
            startup,
            references: 0,
        });
        self.ids.insert(key, id);
        Ok(id)
    }

    fn layout(&self, id: usize) -> Result<&Layout> {
        self.layouts
            .get(id)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Invariant(format!("video layout {id} is not prepared")))
    }

    fn layout_mut(&mut self, id: usize) -> Result<&mut Layout> {
        self.layouts
            .get_mut(id)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Invariant(format!("video layout {id} is not prepared")))
    }

    /// The pages, latent units and raster of `slot`'s trajectory. The pool
    /// gives every request slot the same run of consecutive pages after the
    /// padding page ([`SamplePages::for_slot`]), so the rank names them
    /// itself; the raster is the request's admitted canvas.
    fn trajectory(&self, slot: usize, call: &Call, canvas: Canvas) -> LatentParams {
        LatentParams {
            request_key: call.request_key,
            call_id: call.call_id,
            page_table: self
                .pages
                .for_slot(slot as u64)
                .map(|page| page as u32)
                .collect(),
            latent_units: (self.pages.pages * self.pages.page_units) as u32,
            height: canvas.height,
            width: canvas.width,
        }
    }

    /// The pages of a trajectory, checked against the pool's bounds.
    fn slot_pages(&self, params: &LatentParams) -> Result<Vec<usize>> {
        self.pool.validate_pages(
            &params
                .page_table
                .iter()
                .map(|&page| i64::from(page))
                .collect::<Vec<_>>(),
            i64::from(params.latent_units),
        )
    }

    /// Bytes of a slot's bank pages from `first` page for `pages` pages.
    fn bank_address(&self, bank: u8, first: u64) -> Result<u64> {
        Ok(self.storage.bank_view(bank, first as u32, 1)?.address())
    }

    /// Record the conditioner's forward at `capacity` on `stream`: the
    /// launch of its graph or the eager call. With `capture` (startup's
    /// capture: the capture functions and the CUDA ordinal), a capacity
    /// not captured yet is captured into the graph pool instead of called;
    /// the job's completion carries the graph. A capture is counted when
    /// its graph is stored.
    fn run_conditioner(
        &mut self,
        stream: &Stream,
        capacity: u32,
        capture: Option<(&GraphFunctions, DeviceId)>,
        job: &Job,
    ) -> Result<GraphMode> {
        let pool = self.pool_handle.as_ref();
        let conditioner = self
            .conditioner
            .as_mut()
            .ok_or_else(|| Error::Unsupported("this rank holds no text conditioner".into()))?;
        // Without a graph pool every capacity runs eagerly.
        let lookup = if pool.is_some() {
            conditioner.graphs.lookup(&capacity, None)?
        } else {
            Lookup::Capture
        };
        let mode = match (lookup, capture, pool) {
            (Lookup::Replay(graph), ..) => {
                job.on(Some(stream)).launch(graph)?;
                GraphMode::Replay
            }
            (Lookup::Capture, Some((functions, device)), Some(pool)) => {
                job.capture(Capture {
                    functions: functions.clone(),
                    pool: pool.value.clone(),
                    peers: Vec::new(),
                    stream: stream.raw(),
                    device,
                    method: MethodCall::new(&conditioner.encode, conditioner.arguments(capacity)?),
                    breaks: None,
                });
                GraphMode::Capture
            }
            (Lookup::Capture, ..) => {
                job.call(MethodCall::new(
                    &conditioner.encode,
                    conditioner.arguments(capacity)?,
                ));
                GraphMode::Eager
            }
        };
        Ok(mode)
    }

    /// The arguments of `DenoiserRunner.denoise` for a layout, the Python
    /// half `methods.runner` first.
    fn step_arguments(&self, methods: &Methods, id: usize) -> Result<Vec<Argument>> {
        let layout = self.layout(id)?;
        let text = self
            .envelope
            .device
            .iter()
            .position(|field| field.name == "text_condition")
            .ok_or_else(|| Error::Invariant("the envelope lacks text conditioning".into()))?;
        let mut arguments = vec![
            Argument::Handle(methods.runner.clone()),
            Argument::Handle(layout.handle.clone()),
            Argument::Tensor(self.step.clone()),
            Argument::Tensor(self.state[text].clone()),
        ];
        arguments.extend(layout.tensors.iter().cloned().map(Argument::Tensor));
        Ok(arguments)
    }

    /// Record a solver step of layout `id` on `stream`: the launch of its
    /// graph or the eager call of `methods.denoise`. With `capture`
    /// (startup's capture: the capture functions and the CUDA ordinal), a
    /// layout not captured yet is captured into the graph pool instead of
    /// called; the job's completion carries the graph. A capture is
    /// counted when its graph is stored.
    fn run_step(
        &mut self,
        methods: &Methods,
        stream: &Stream,
        id: usize,
        capture: Option<(&GraphFunctions, DeviceId)>,
        job: &Job,
    ) -> Result<GraphMode> {
        let mode = match (self.graphs.lookup(&id, None), capture, &self.pool_handle) {
            (Ok(Lookup::Replay(graph)), ..) => {
                job.on(Some(stream)).launch(graph)?;
                GraphMode::Replay
            }
            (Ok(Lookup::Capture), Some((functions, device)), Some(pool)) => {
                job.capture(Capture {
                    functions: functions.clone(),
                    pool: pool.value.clone(),
                    peers: Vec::new(),
                    stream: stream.raw(),
                    device,
                    method: MethodCall::new(&methods.denoise, self.step_arguments(methods, id)?),
                    breaks: None,
                });
                GraphMode::Capture
            }
            // Transient layouts and uncaptured startup layouts run eagerly.
            _ => {
                job.call(MethodCall::new(
                    &methods.denoise,
                    self.step_arguments(methods, id)?,
                ));
                GraphMode::Eager
            }
        };
        Ok(mode)
    }
}
