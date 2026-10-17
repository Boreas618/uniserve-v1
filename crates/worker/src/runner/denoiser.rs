//! The denoiser runner, and a video model's sizing.
//!
//! [`DenoiserRunner`] is the Rust half of
//! `uniserve.runner.denoiser.DenoiserRunner` for one denoiser of a model:
//! a video model's denoiser ([`VideoDenoiser`], below) or an image model's
//! cache-conditioned image denoiser ([`ImageDenoiser`], in [`image`]). The
//! two families share the runner's owner, device, stream, the Python
//! half with the methods both call (`draw_noise`, `prepare_latents`,
//! `encode_conditions`, `denoise`, `kernels`, `close`), the graph capture
//! functions and the runner's lifecycle (`seal`, `kernels`, `close`); they
//! differ in the state their forwards read and write. [`VideoModel`] asks
//! the sizing methods of the Python runners what admission, capacity
//! planning and the runners need.
//!
//! A video request's samples live in the latent pool: two banks of pages
//! whose slot owns `pages` consecutive pages after page 0. A solver step
//! gathers the committed bank's pages into the runner's sample scratch,
//! copies the request's state (its slot rows of every state table and, on
//! the pipeline stage holding the text conditioner, of the retained
//! conditioning) into the fixed state buffers, writes the
//! evaluation index, launches the layout's captured graph (or runs
//! `DenoiserRunner.denoise` of `uniserve.runner.denoiser` on the
//! Python thread) and scatters the scratch into the other bank. Every one of
//! those buffers is allocated here; the Python half receives the tensors of
//! a forward as its arguments.
//!
//! Layouts ([`VideoLayout`]) fix the shapes a step evaluates. Startup
//! layouts are prepared, warmed and captured before serving and stay
//! resident; a conditioned request whose layout was not prepared at startup
//! prepares a transient layout, evaluated eagerly, which the least recently
//! used transient layout makes room for once no request uses it.

pub(crate) mod image;
pub(crate) mod image_model;
pub(crate) mod runners;
mod start;
pub(crate) mod warmup;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::{Canvas, ConditionTiles, VideoCondition};
use uniserve_cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use uniserve_ipc::{
    Batch as BatchPlan, Call, DimBound, LatentParams, OutputInfo, RasterAxes, ShapeBound,
    VideoDenoiserInfo,
};

use crate::config::Device;
use crate::config::placement::{Capability, Declarations};
use crate::cuda_graph::GraphPool;
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::executor::products::{OutputShape, Placement, bind_outputs};
use crate::executor::video::MediaBatch;
use crate::executor::video::{Parts, latent_params, product_name, stream_of};
use crate::ffi::{Backing, DType, Function, Handle, Interpreter, integer_list, integers};
use crate::lane::{Capture, Completion, Job};
use crate::parallel::joins::GraphFunctions;
use crate::profiling::kernels::KernelRecord;
use crate::request::video::{VideoLayout, VideoSize};
use crate::runner::denoiser::image::ImageDenoiser;
use crate::runner::methods::{Argument, MethodCall, kernels_call, list_kernels};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{RequestSlots, SlotField};
use crate::stream::{Fill, Issue};
use crate::{Error, LatentPool, LatentStorage, LatentUpdate, Result};

/// A request's samples occupy at most this many pages, which bounds the
/// copies a step's gather and scatter make.
const REQUEST_PAGES: u64 = 8;

/// Each modality's sample starts on this element boundary.
const SAMPLE_ALIGNMENT: u64 = 256;

fn aligned(elements: u64) -> u64 {
    elements.div_ceil(SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT
}

/// The latent pages of one request slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SamplePages {
    /// Elements of one page.
    pub(crate) page_units: u64,
    /// Pages a request slot owns.
    pub(crate) pages: u64,
}

impl SamplePages {
    /// Pages holding the complete samples of every `largest` layout: the
    /// global extents bound every sequence-parallel shard, even when a
    /// shorter prompt moves more media rows onto one rank.
    pub(crate) fn plan(model: &VideoModel, largest: &[VideoLayout]) -> Result<Self> {
        let mut capacity = 0;
        for layout in largest {
            let mut total = 0;
            for modality in &model.facts.modalities {
                total += aligned(model.latent_elements(modality, layout)?);
            }
            capacity = capacity.max(total);
        }
        if capacity == 0 {
            return Err(Error::Interface(
                "the video denoiser describes empty samples".into(),
            ));
        }
        let page_units = aligned(capacity.div_ceil(REQUEST_PAGES));
        Ok(Self {
            page_units,
            pages: capacity.div_ceil(page_units),
        })
    }

    /// Page zero is padding; every slot owns a consecutive run after it.
    pub(crate) fn for_slot(&self, slot: u64) -> Range<u64> {
        1 + (slot - 1) * self.pages..1 + slot * self.pages
    }

    /// Pages of a pool of `slots` slots, the padding page included.
    pub(crate) fn pool_pages(&self, slots: u64) -> u64 {
        slots * self.pages + 1
    }
}

/// Where a layout's local samples sit in a slot's pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SampleLayout {
    /// Element offset of each modality's local sample, in modality order.
    pub(crate) offsets: Vec<u64>,
    /// Elements of each modality's local sample.
    pub(crate) elements: Vec<u64>,
    /// Pages the samples occupy.
    pub(crate) pages: u64,
}

impl SampleLayout {
    pub(crate) fn new(fields: &LayoutFields, page_units: u64) -> Result<Self> {
        let mut offsets = Vec::new();
        let mut elements = Vec::new();
        let mut cursor = 0;
        for (name, _) in &fields.samples {
            let field = fields
                .state
                .iter()
                .find(|field| field.name == *name && field.sample)
                .ok_or_else(|| Error::Interface(format!("layout state lacks sample {name}")))?;
            offsets.push(cursor);
            elements.push(field.elements);
            cursor += aligned(field.elements);
        }
        Ok(Self {
            offsets,
            elements,
            pages: cursor.div_ceil(page_units),
        })
    }
}

/// The capacity of every field across the layouts a rank admits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateEnvelope {
    /// Device fields of a request slot: every state table and the retained
    /// conditioning, as flat elements.
    pub(crate) device: Vec<SlotField>,
    /// Pinned host fields of a request slot: every state field's source,
    /// each modality's native draw and the condition draws.
    pub(crate) host: Vec<SlotField>,
    /// Sample elements the scratch holds: a slot's pages, which hold the
    /// complete samples and so any rank's share of them.
    pub(crate) samples: u64,
}

fn grow(
    fields: &mut Vec<SlotField>,
    name: String,
    dtype: DType,
    elements: u64,
    host: bool,
) -> Result<()> {
    let elements =
        i64::try_from(elements).map_err(|_| Error::Interface("state field overflows".into()))?;
    match fields.iter_mut().find(|field| field.name == name) {
        Some(field) if field.dtype != dtype => Err(Error::Interface(format!(
            "state field {name} changes its dtype across layouts"
        ))),
        Some(field) => {
            field.shape[0] = field.shape[0].max(elements);
            Ok(())
        }
        None => {
            fields.push(SlotField {
                name,
                dtype,
                shape: vec![elements],
                host,
            });
            Ok(())
        }
    }
}

impl StateEnvelope {
    /// The envelope of `layouts`' fields; the text conditioning is BF16.
    /// Every layout's local samples must fit a slot's `pages`.
    pub(crate) fn new(layouts: &[LayoutFields], pages: SamplePages) -> Result<Self> {
        let mut device = Vec::new();
        let mut host = Vec::new();
        for fields in layouts {
            for field in &fields.state {
                let elements = if field.sample {
                    // A rank's rows of a sample move with the rows before
                    // them in the sequence: the prompt's and conditions' of
                    // the request's own layout, which no startup layout
                    // need match. The complete sample bounds every share.
                    fields
                        .samples
                        .iter()
                        .find(|(name, _)| *name == field.name)
                        .map(|(_, sample)| sample.rows * sample.width)
                        .ok_or_else(|| {
                            Error::Interface(format!(
                                "sample field {} has no sample rows",
                                field.name
                            ))
                        })?
                } else {
                    grow(
                        &mut device,
                        field.name.clone(),
                        field.dtype,
                        field.elements,
                        false,
                    )?;
                    field.elements
                };
                grow(
                    &mut host,
                    format!("{}_source", field.name),
                    field.dtype,
                    elements,
                    true,
                )?;
            }
            for (name, elements) in &fields.noise {
                grow(
                    &mut host,
                    format!("{name}_noise"),
                    DType::F32,
                    *elements,
                    true,
                )?;
            }
            grow(
                &mut host,
                "condition_noise".into(),
                DType::F32,
                fields.condition_noise.max(1),
                true,
            )?;
            let (rows, width) = fields.text_condition;
            grow(
                &mut device,
                "text_condition".into(),
                DType::BF16,
                rows * width,
                false,
            )?;
            if SampleLayout::new(fields, pages.page_units)?.pages > pages.pages {
                return Err(Error::Invariant(
                    "a video layout's samples exceed the request's latent pages".into(),
                ));
            }
        }
        Ok(Self {
            device,
            host,
            samples: pages.pages * pages.page_units,
        })
    }

    /// Device bytes of one request slot's state.
    pub(crate) fn slot_bytes(&self) -> u64 {
        self.device
            .iter()
            .map(|field| {
                field.shape.iter().product::<i64>().max(0) as u64 * field.dtype.bytes() as u64
            })
            .sum()
    }
}

/// The text conditioner of a pipeline stage that holds it: one Python
/// `EncoderRunner` per prompt capacity over the same input and output.
struct Conditioner {
    /// `[capacity, width]` prompt features, zero past the prompt.
    input: StorageView,
    /// `[1]` int32 prompt length.
    lengths: StorageView,
    /// `[capacity, conditioning width]` BF16 refined rows.
    output: StorageView,
    /// The runner of each prompt capacity.
    runners: BTreeMap<u32, Handle>,
    encode: Function,
    kernels: Function,
    close: Function,
    graphs: GraphTable<u32>,
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

/// The Python half: a `uniserve.runner.denoiser.DenoiserRunner` and the
/// methods both families call, each with the runner first.
struct Methods {
    runner: Handle,
    draw_noise: Function,
    prepare_latents: Function,
    encode_conditions: Function,
    denoise: Function,
    kernels: Function,
    close: Function,
}

impl Methods {
    /// The methods of the Python half `runner`.
    fn import(runner: Handle) -> Result<Self> {
        let method = |name: &str| Interpreter::import(&format!("{DENOISER}.{name}"));
        Ok(Self {
            runner,
            draw_noise: method("draw_noise")?,
            prepare_latents: method("prepare_latents")?,
            encode_conditions: method("encode_conditions")?,
            denoise: method("denoise")?,
            kernels: method("kernels")?,
            close: method("close")?,
        })
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

/// The runner of one denoiser of a model.
pub(crate) struct DenoiserRunner {
    /// The component the runner serves, naming it in messages.
    owner: String,
    device: Device,
    /// The stream every forward and sample update runs on; `None` for an
    /// image denoiser on the host. A video denoiser computes on CUDA.
    stream: Option<Stream>,
    methods: Methods,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
    family: Family,
}

/// The denoiser family a runner serves, with the state of its forwards.
enum Family {
    Video(Box<VideoDenoiser>),
    Image(Box<ImageDenoiser>),
}

impl Family {
    fn video(&self) -> Result<&VideoDenoiser> {
        match self {
            Self::Video(video) => Ok(video),
            Self::Image(_) => Err(Error::Invariant("the denoiser denoises no video".into())),
        }
    }

    fn video_mut(&mut self) -> Result<&mut VideoDenoiser> {
        match self {
            Self::Video(video) => Ok(video),
            Self::Image(_) => Err(Error::Invariant("the denoiser denoises no video".into())),
        }
    }

    fn image(&self) -> Result<&ImageDenoiser> {
        match self {
            Self::Image(image) => Ok(image),
            Self::Video(_) => Err(Error::Invariant("the denoiser denoises no images".into())),
        }
    }

    fn image_mut(&mut self) -> Result<&mut ImageDenoiser> {
        match self {
            Self::Image(image) => Ok(image),
            Self::Video(_) => Err(Error::Invariant("the denoiser denoises no images".into())),
        }
    }
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
    conditioner: Option<Conditioner>,
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
    graphs: GraphTable<usize>,
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
        let memory = |bytes: u64| Memory::device(device, bytes.max(1) as usize);
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
        let num_pages = setup.pages.pool_pages(u64::from(setup.slots));
        let storage = LatentStorage::allocate(
            setup.slots,
            num_pages as u32,
            setup.pages.page_units as u32,
            1,
            DType::F32,
            false,
            Some(device),
            Some(&setup.stream),
        )?;
        let pool = LatentPool::new(
            setup.slots as usize,
            num_pages as usize,
            setup.pages.page_units as usize,
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

    /// The device the runner computes on.
    pub(crate) fn device(&self) -> Device {
        self.device
    }

    /// The runner's stream; `None` for an image denoiser on the host.
    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// The CUDA device and stream of a video denoiser, which always
    /// computes on CUDA.
    fn cuda(&self) -> Result<(DeviceId, Stream)> {
        match (self.device, &self.stream) {
            (Device::Cuda(device), Some(stream)) => Ok((device, stream.clone())),
            _ => Err(Error::Invariant(format!(
                "{}: a video denoiser computes on a CUDA stream",
                self.owner
            ))),
        }
    }

    /// Graphs of the denoiser's steps captured so far.
    pub(crate) fn captured(&self) -> usize {
        match &self.family {
            Family::Video(video) => video.graphs.len(),
            Family::Image(image) => image.graphs.len(),
        }
    }

    /// Refuse further captures: serving replays what startup captured.
    pub(crate) fn seal(&mut self) {
        match &mut self.family {
            Family::Video(video) => {
                video.graphs.seal();
                if let Some(conditioner) = &mut video.conditioner {
                    conditioner.graphs.seal();
                }
            }
            Family::Image(image) => image.graphs.seal(),
        }
    }

    /// The calls that list the kernels the operators of the denoiser's
    /// Python runners selected: the denoiser's, then a video denoiser's
    /// conditioner's at each capacity.
    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        let mut calls = vec![kernels_call(&self.methods.kernels, &self.methods.runner)];
        if let Family::Video(video) = &self.family
            && let Some(conditioner) = &video.conditioner
        {
            for runner in conditioner.runners.values() {
                calls.push(kernels_call(&conditioner.kernels, runner));
            }
        }
        calls
    }

    /// The kernels the operators of the denoiser's Python runners selected.
    pub(crate) fn kernels(&self) -> Result<Vec<KernelRecord>> {
        list_kernels(self.kernel_calls())
    }

    /// Release graphs and the Python halves; the caller drained the
    /// runner's streams. An aborted close keeps everything device work may
    /// still address until the process exits. Every release is attempted;
    /// the first failure is returned.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut failure = None;
        let mut keep = |result: Result<()>| {
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        };
        let flag = Any::from(aborted);
        match &mut self.family {
            Family::Video(video) => {
                keep(video.graphs.close(aborted));
                if let Some(conditioner) = &mut video.conditioner {
                    keep(conditioner.graphs.close(aborted));
                    for runner in conditioner.runners.values() {
                        keep(conditioner.close.invoke(&[runner.view(), (&flag).into()]));
                    }
                }
            }
            Family::Image(image) => keep(image.graphs.close(aborted)),
        }
        keep(
            self.methods
                .close
                .invoke(&[self.methods.runner.view(), (&flag).into()]),
        );
        failure.map_or(Ok(()), Err)
    }

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
    /// slot rows. The host task that filled them has completed.
    pub(crate) fn initialize(
        &mut self,
        id: usize,
        slot: usize,
        call: &Call,
        params: &LatentParams,
        job: &Job,
    ) -> Result<LatentUpdate> {
        let (_, stream) = self.cuda()?;
        let video = self.family.video_mut()?;
        let pages = video.slot_pages(slot, params)?;
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
        LatentUpdate::for_call(slot, call, params)
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

    /// Advance a request's trajectory by one solver step and return its
    /// commit. `outputs` receive the final step's local samples, one view
    /// per modality.
    pub(crate) fn denoise(
        &mut self,
        id: usize,
        slot: usize,
        call: &Call,
        params: &LatentParams,
        job: &Job,
    ) -> Result<(LatentUpdate, GraphMode)> {
        if params.step_count != 1 {
            return Err(Error::Invalid(
                "a video denoising call advances exactly one solver step".into(),
            ));
        }
        let (_, stream) = self.cuda()?;
        let video = self.family.video_mut()?;
        let pages = video.slot_pages(slot, params)?;
        let step = i64::from(params.start_step);
        let read = video.pool.current_bank(
            slot,
            step,
            step + 1,
            i64::from(params.latent_units),
            i64::from(params.height),
            i64::from(params.width),
            &pages,
        )?;
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
                value: params.start_step,
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
        Ok((LatentUpdate::for_call(slot, call, params)?, mode))
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

    /// The pages of `slot` as the scheduler must assign them.
    fn slot_pages(&self, slot: usize, params: &LatentParams) -> Result<Vec<usize>> {
        let expected = self.pages.for_slot(slot as u64);
        if params.page_table.len() as u64 != expected.end - expected.start
            || params
                .page_table
                .iter()
                .zip(expected)
                .any(|(&page, wanted)| u64::from(page) != wanted)
        {
            return Err(Error::Invalid(
                "video latent pages are not the request slot's pages".into(),
            ));
        }
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

/// Initialize a request's trajectory: its prepared sources, its refined
/// prompt where this rank retains conditioning
/// ([`DenoiserRunner::conditions`]) and its encoded conditions.
pub(crate) fn prepare(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let params = latent_params(plan, call)?;
    let (id, seed, conditioned, size) = {
        let request = parts.video.request(slot, call.request_key)?;
        let id = request
            .layout
            .ok_or_else(|| Error::Invariant("a denoising request has no layout".into()))?;
        let size = request
            .admission
            .as_ref()
            .map(|admission| admission.size.clone())
            .ok_or_else(|| Error::Invariant("a denoising request has no size".into()))?;
        (id, request.seed, !request.conditions.is_empty(), size)
    };
    if call.inputs.is_empty() || (call.inputs.len() > 1) != conditioned {
        return Err(Error::Invalid(
            "video preparation reads its conditioning and its conditions".into(),
        ));
    }
    let refines = parts
        .denoisers
        .video
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?
        .runner
        .conditions();
    let store = parts.owners().store;
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    // A pipeline stage without the conditioner leaves the prompt features
    // unread: its forward receives the refined prompt among the preceding
    // stage's hidden rows.
    let features = refines
        .then(|| batch.read(store, &call.inputs[0], call.call_id, issue))
        .transpose()?;
    let conditions: Vec<Arc<crate::ProductRead>> = call.inputs[1..]
        .iter()
        .map(|input| batch.read(store, input, call.call_id, issue))
        .collect::<Result<_>>()?;
    let denoiser = parts
        .denoisers
        .video
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
    // The request's noise and the host sources its slot is initialized from.
    denoiser
        .runner
        .prepare_latents(id, slot, &size, seed, job)?;
    let update = denoiser.runner.initialize(id, slot, call, params, job)?;
    if let Some(features) = features {
        let value = features.value();
        let count = u64::from(size.tokens);
        if value.shape().first().copied() != Some(count as i64) {
            return Err(Error::Invalid(
                "prompt features disagree with the request's prompt".into(),
            ));
        }
        let mode = denoiser.runner.condition(id, slot, value, count, job)?;
        batch.record(mode);
    }
    if conditioned {
        let request = parts.video.request(slot, call.request_key)?;
        let latents = condition_latents(&request.conditions, &conditions)?;
        let denoiser = parts
            .denoisers
            .video
            .as_ref()
            .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
        denoiser
            .runner
            .encode_conditions(id, slot, &size, &latents, job)?;
    }
    batch.batch.calls[index].forward.latent = Some(update);
    Ok(())
}

/// Rows of one condition read: `(read, first row, rows)`.
type RowSpan = (usize, u64, u64);

/// Locate every condition latent in latent preparation's condition reads.
///
/// `conditions` are each condition's `(visual rows, audio rows)` in request
/// order. The reads are the visual rounds' rows in round order (`rounds`
/// holds each round's row count), then, when any condition carries audio,
/// one read of every track's rows (`tracks`). The visual rounds hold the
/// conditions' visual rows back to back, and a round ends wherever its
/// units do, so a condition's rows may continue into the next round.
///
/// Returns the latents in request order, a condition's visual rows before
/// its audio rows, each as the spans that hold its rows in row order.
fn condition_spans(
    conditions: &[(u64, u64)],
    rounds: &[u64],
    tracks: Option<u64>,
) -> Result<Vec<Vec<RowSpan>>> {
    let mismatch =
        || Error::Invalid("condition latents disagree with the request's planned rows".into());
    let visual: u64 = conditions.iter().map(|rows| rows.0).sum();
    let audio: u64 = conditions.iter().map(|rows| rows.1).sum();
    if rounds.iter().sum::<u64>() != visual || tracks.unwrap_or(0) != audio {
        return Err(mismatch());
    }
    // The audio read follows the visual rounds.
    let track_read = rounds.len();
    let mut result = Vec::new();
    let (mut round, mut row, mut audio_row) = (0usize, 0u64, 0u64);
    for &(visual_rows, audio_rows) in conditions {
        if visual_rows > 0 {
            let mut spans = Vec::new();
            let mut needed = visual_rows;
            while needed > 0 {
                let available = *rounds.get(round).ok_or_else(mismatch)? - row;
                let taken = available.min(needed);
                if taken > 0 {
                    spans.push((round, row, taken));
                }
                needed -= taken;
                row += taken;
                if taken == available {
                    round += 1;
                    row = 0;
                }
            }
            result.push(spans);
        }
        if audio_rows > 0 {
            result.push(vec![(track_read, audio_row, audio_rows)]);
            audio_row += audio_rows;
        }
    }
    Ok(result)
}

/// The condition latents of latent preparation's condition `reads`, each a
/// list of row tensors that together hold its rows (see
/// [`condition_spans`]).
fn condition_latents(
    conditions: &[VideoCondition],
    reads: &[Arc<crate::ProductRead>],
) -> Result<Vec<Vec<StorageView>>> {
    let audio = conditions.iter().any(|condition| condition.audio_rows > 0);
    let (rounds, tracks) = match (audio, reads.split_last()) {
        (true, Some((tracks, rounds))) => (rounds, Some(tracks)),
        (true, None) => {
            return Err(Error::Invalid(
                "condition latents lack their audio rows".into(),
            ));
        }
        (false, _) => (reads, None),
    };
    let rows = |read: &Arc<crate::ProductRead>| {
        read.value().shape().first().copied().unwrap_or(0).max(0) as u64
    };
    let planned: Vec<(u64, u64)> = conditions
        .iter()
        .map(|condition| (condition.video_rows(), u64::from(condition.audio_rows)))
        .collect();
    let round_rows: Vec<u64> = rounds.iter().map(rows).collect();
    condition_spans(&planned, &round_rows, tracks.map(rows))?
        .into_iter()
        .map(|spans| {
            spans
                .into_iter()
                .map(|(read, first, count)| {
                    reads[read].value().narrow(0, first as i64, count as i64)
                })
                .collect()
        })
        .collect()
}

/// Advance a request's trajectory by one step; the final step publishes the
/// rank's rows of each modality's latents.
pub(crate) fn denoise(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let params = latent_params(plan, call)?;
    let id = parts
        .video
        .request(slot, call.request_key)?
        .layout
        .ok_or_else(|| Error::Invariant("a denoising request has no layout".into()))?;
    let num_steps = parts.video.model.facts.num_steps;
    let names: Vec<String> = call
        .outputs
        .iter()
        .map(|output| product_name(parts.info, call, output).map(str::to_owned))
        .collect::<Result<_>>()?;
    let denoiser = parts
        .denoisers
        .video
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
    let (update, mode) = denoiser.runner.denoise(id, slot, call, params, job)?;
    batch.record(mode);
    batch.batch.calls[index].forward.latent = Some(update);
    if call.outputs.is_empty() {
        return Ok(());
    }
    if params.start_step + params.step_count != num_steps {
        return Err(Error::Invalid(
            "final latent products require completed denoising".into(),
        ));
    }
    let samples = denoiser.runner.final_samples(id)?;
    let rows = denoiser.runner.sample_rows(id)?;
    let modalities = parts.video.model.facts.modalities.clone();
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    for (output, name) in call.outputs.iter().zip(&names) {
        let modality = modalities
            .iter()
            .position(|modality| format!("{modality}_latents") == *name)
            .ok_or_else(|| Error::Invalid(format!("the denoiser publishes no {name}")))?;
        let (Some(sample), sample_rows) = (&samples[modality], rows[modality]) else {
            // This rank holds none of the modality's rows.
            continue;
        };
        let shape = vec![sample_rows.rows, sample_rows.width];
        let region = Some(vec![
            sample_rows.local.0..sample_rows.local.1,
            0..sample_rows.width,
        ]);
        let bound = bind_outputs(
            owners.store,
            &[OutputShape {
                product: output.clone(),
                shape,
                region,
            }],
            batch.target(),
            placement,
        )?;
        let write = bound
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("a latent product was not bound".into()))?;
        batch.batch.writes.push(write.clone());
        let issue = job.on(Some(&stream));
        owners.store.write(&write, sample, issue, None)?;
        batch.batch.calls[index]
            .forward
            .export(owners, call, output, &write, false, issue)?;
    }
    Ok(())
}

const DENOISER: &str = "uniserve.runner.denoiser.DenoiserRunner";
const DECODER: &str = "uniserve.runner.decoder.DecoderRunner";
const ENCODER: &str = "uniserve.runner.encoder.EncoderRunner";
const OUTPUTS: &str = "uniserve.runner.denoiser.Outputs";

/// What the model reports about its video timeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VideoFacts {
    /// Sample modalities in the denoiser's order (`video`, `audio`).
    pub(crate) modalities: Vec<String>,
    pub(crate) num_steps: u32,
    pub(crate) tasks: Vec<String>,
    pub(crate) video_shift: f64,
    pub(crate) audio_shift: f64,
    /// Canvases the denoiser offers: a deployment prepares a selection of
    /// them, or all of them without one, and admits only those.
    pub(crate) canvases: Vec<Canvas>,
    pub(crate) max_sequence_rows: Option<u32>,
    pub(crate) condition_tiles: Option<ConditionTiles>,
    pub(crate) sample_dtype: DType,
    pub(crate) text_condition_width: u32,
    /// Output video frames per second.
    pub(crate) frame_rate: u32,
    /// Output audio samples per second.
    pub(crate) sample_rate: u32,
}

/// One state field of a layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateField {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    /// Elements of the field on this rank.
    pub(crate) elements: u64,
    /// A sample modality, stored in latent pages rather than slot banks.
    pub(crate) sample: bool,
}

impl StateField {
    pub(crate) fn bytes(&self) -> u64 {
        self.elements * self.dtype.bytes() as u64
    }
}

/// A modality's sample in one layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SampleRows {
    /// Rows of the complete sample across all ranks.
    pub(crate) rows: u64,
    pub(crate) width: u64,
    /// This rank's rows.
    pub(crate) local: (u64, u64),
}

/// The request state of one layout on this rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayoutFields {
    pub(crate) state: Vec<StateField>,
    /// Sample rows by modality, in modality order.
    pub(crate) samples: Vec<(String, SampleRows)>,
    /// Elements of each modality's native draw, in modality order.
    pub(crate) noise: Vec<(String, u64)>,
    /// FP32 elements of the condition draws.
    pub(crate) condition_noise: u64,
    /// `[rows, width]` of the retained conditioning.
    pub(crate) text_condition: (u64, u64),
    /// Elements of the layout's workspace buffers.
    pub(crate) workspace: u64,
}

impl LayoutFields {
    /// Elements of the state and workspace the denoiser holds for the
    /// layout on this rank: what the layout takes of the storage every
    /// layout shares.
    pub(crate) fn storage(&self) -> u64 {
        self.state.iter().map(|field| field.elements).sum::<u64>() + self.workspace
    }
}

/// What the model's layout functions answer about one layout, as
/// [`VideoModel::fields`] reads them.
struct LayoutAnswers {
    /// The state fields, in state order, and each one's dtype and elements.
    names: Vec<String>,
    dtypes: Vec<String>,
    elements: Vec<i64>,
    /// Each modality's `[rows, width, start, stop]` sample rows.
    rows: Vec<i64>,
    /// Elements of each modality's native draw.
    draws: Vec<i64>,
    condition_noise: i64,
    text_rows: i64,
    workspace: i64,
}

/// The sizing methods, each called with the loaded model first.
struct SizingMethods {
    legal_frame_count: Function,
    layout_size: Function,
    holds: Function,
    condition_layout: Function,
    latent_shape: Function,
    state_names: Function,
    state_dtypes: Function,
    state_elements: Function,
    sample_rows: Function,
    noise_elements: Function,
    condition_noise_elements: Function,
    text_condition_rows: Function,
    workspace_elements: Function,
    products: Function,
    reader_products: Function,
    /// The column methods of the `Outputs` the two product methods return:
    /// owners, names, dtypes, ranks, extents, variable and raster axes.
    output_columns: [Function; 7],
    unit_frames: Function,
    segment: Function,
    window_rows: Function,
    overlap_elements: Function,
    overlap_dtype: Function,
    track_samples: Function,
    audio_units: Function,
    audio_latent_frames: Function,
}

/// A video model's facts and sizing rules.
///
/// FIXME: admission and planning size a request before any runner of its
/// layout exists, so the facts record and the sizing functions are held
/// here apart from `DenoiserRunner`; they go with the fact records.
pub(crate) struct VideoModel {
    /// The component of the deployment's video denoiser.
    pub(crate) denoiser: String,
    pub(crate) facts: VideoFacts,
    loaded: Handle,
    methods: SizingMethods,
}

/// The video denoiser component a deployment places: the unique declared
/// video denoiser among `deployment` components (or the only one declared).
pub(crate) fn placed_denoiser(
    declarations: &Declarations,
    deployment: &[String],
) -> Result<String> {
    let declared: Vec<&str> = declarations
        .iter()
        .filter(|(_, calls)| calls.iter().any(|call| call.is(Capability::VideoDenoiser)))
        .map(|(name, _)| name)
        .collect();
    if let [only] = declared[..] {
        return Ok(only.to_owned());
    }
    let placed: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|name| deployment.iter().any(|component| component == name))
        .collect();
    match placed[..] {
        [one] => Ok(one.to_owned()),
        _ => Err(Error::Unsupported(format!(
            "a deployment places exactly one of the model's video denoisers {declared:?}; it places {}",
            placed.len()
        ))),
    }
}

fn narrow(value: i64, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Interface(format!("the video model reports an invalid {what}")))
}

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}

fn strings(function: &Function, value: Any) -> Result<Vec<String>> {
    let values: tvm_ffi::Array<Any> = value
        .try_into()
        .map_err(|error| Error::numerical(function.name(), &error))?;
    values
        .iter()
        .map(|value| {
            tvm_ffi::String::try_from(value)
                .map(|value| value.as_str().to_owned())
                .map_err(|error| Error::numerical(function.name(), &error))
        })
        .collect()
}

fn layout_arguments(layout: &VideoLayout) -> Any {
    layout.descriptor()
}

impl VideoModel {
    /// Resolve the sizing methods and read the facts of `loaded` for the
    /// `denoiser` component.
    pub(crate) fn new(loaded: &Handle, denoiser: String) -> Result<Self> {
        let of_denoiser = |name: &str| Interpreter::import(&format!("{DENOISER}.{name}"));
        let of_decoder = |name: &str| Interpreter::import(&format!("{DECODER}.{name}"));
        let methods = SizingMethods {
            legal_frame_count: of_denoiser("legal_frame_count")?,
            layout_size: of_denoiser("layout_size")?,
            holds: of_denoiser("holds")?,
            condition_layout: of_denoiser("condition_layout")?,
            latent_shape: of_denoiser("latent_shape")?,
            state_names: of_denoiser("state_names")?,
            state_dtypes: of_denoiser("state_dtypes")?,
            state_elements: of_denoiser("state_elements")?,
            sample_rows: of_denoiser("sample_rows")?,
            noise_elements: of_denoiser("noise_elements")?,
            condition_noise_elements: of_denoiser("condition_noise_elements")?,
            text_condition_rows: of_denoiser("text_condition_rows")?,
            workspace_elements: of_denoiser("workspace_elements")?,
            products: of_denoiser("products")?,
            reader_products: of_denoiser("reader_products")?,
            output_columns: [
                "owners",
                "names",
                "dtypes",
                "ranks",
                "extents",
                "variable_axes",
                "raster_axes",
            ]
            .map(|column| Interpreter::import(&format!("{OUTPUTS}.{column}")))
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::Invariant("the outputs have seven columns".into()))?,
            unit_frames: of_decoder("unit_frames")?,
            segment: of_decoder("segment")?,
            window_rows: of_decoder("window_rows")?,
            overlap_elements: of_decoder("overlap_elements")?,
            overlap_dtype: of_decoder("overlap_dtype")?,
            track_samples: of_decoder("track_samples")?,
            audio_units: of_decoder("audio_units")?,
            audio_latent_frames: of_decoder("audio_latent_frames")?,
        };

        // A fact of the denoiser component, read once.
        let fact = |name: &str| -> Result<(Function, Any)> {
            let function = of_denoiser(name)?;
            let value = function.call_values(&[loaded.to_any(), text(&denoiser)])?;
            Ok((function, value))
        };
        let integer = |name: &str| -> Result<i64> {
            let (function, value) = fact(name)?;
            i64::try_from(value).map_err(|error| Error::numerical(function.name(), &error))
        };
        let list = |name: &str| -> Result<Vec<i64>> {
            let (function, value) = fact(name)?;
            integers(value).map_err(|error| Error::numerical(function.name(), &error))
        };
        let names = |name: &str| -> Result<Vec<String>> {
            let (function, value) = fact(name)?;
            strings(&function, value)
        };
        // A fact of the decoders, which the model holds one of each.
        let rate = |name: &str| -> Result<u32> {
            let function = of_decoder(name)?;
            let value: i64 = function.call_as(&[loaded.view()])?;
            narrow(value, name)
        };

        let canvases = list("canvases")?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[width, height]| {
                Ok(Canvas {
                    width: narrow(width, "canvas")?,
                    height: narrow(height, "canvas")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let condition_tiles = match list("condition_tiles")?[..] {
            [] => None,
            [rows, frames, height, width] => Some(ConditionTiles {
                rows: narrow(rows, "tile")?,
                video: [
                    narrow(frames, "tile")?,
                    narrow(height, "tile")?,
                    narrow(width, "tile")?,
                ],
            }),
            _ => {
                return Err(Error::Interface(
                    "condition tiles are [rows, t, h, w]".into(),
                ));
            }
        };
        let shifts = {
            let (function, value) = fact("schedule_shifts")?;
            let values: tvm_ffi::Array<Any> = value
                .try_into()
                .map_err(|error| Error::numerical(function.name(), &error))?;
            values
                .iter()
                .map(|value| {
                    f64::try_from(value).map_err(|error| Error::numerical(function.name(), &error))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let [video_shift, audio_shift] = shifts[..] else {
            return Err(Error::Interface(
                "schedule shifts are [video, audio]".into(),
            ));
        };
        let sample_dtype = {
            let (function, value) = fact("sample_dtype")?;
            let name = tvm_ffi::String::try_from(value)
                .map_err(|error| Error::numerical(function.name(), &error))?;
            DType::parse(name.as_str())?
        };
        let rows = integer("max_sequence_rows")?;
        let facts = VideoFacts {
            modalities: names("modalities")?,
            num_steps: narrow(integer("num_steps")?, "step count")?,
            tasks: names("tasks")?,
            video_shift,
            audio_shift,
            canvases,
            max_sequence_rows: (rows > 0).then_some(rows as u32),
            condition_tiles,
            sample_dtype,
            text_condition_width: narrow(integer("text_condition_width")?, "width")?,
            frame_rate: rate("frame_rate")?,
            sample_rate: rate("sample_rate")?,
        };
        Ok(Self {
            denoiser,
            facts,
            loaded: loaded.clone(),
            methods,
        })
    }

    /// Call a sizing method of the denoiser component.
    fn denoiser(&self, method: &Function, arguments: Vec<Any>) -> Result<Any> {
        let mut all = vec![self.loaded.to_any(), text(&self.denoiser)];
        all.extend(arguments);
        method.call_values(&all)
    }

    /// Call a sizing method of the decoders.
    fn decoder(&self, method: &Function, arguments: Vec<Any>) -> Result<Any> {
        let mut all = vec![self.loaded.to_any()];
        all.extend(arguments);
        method.call_values(&all)
    }

    fn integers(&self, method: &Function, value: Any) -> Result<Vec<i64>> {
        integers(value).map_err(|error| Error::numerical(method.name(), &error))
    }

    fn integer(&self, method: &Function, value: Any) -> Result<i64> {
        i64::try_from(value).map_err(|error| Error::numerical(method.name(), &error))
    }

    /// The tasks of the denoiser this worker executes, in its order.
    pub(crate) fn executed_tasks(&self) -> Vec<String> {
        self.facts
            .tasks
            .iter()
            .filter(|task| matches!(task.as_str(), "t2va" | "fl2va" | "ref2va"))
            .cloned()
            .collect()
    }

    /// Whether any executed task takes conditions.
    pub(crate) fn conditions(&self) -> bool {
        self.executed_tasks().iter().any(|task| task != "t2va")
    }

    /// The handshake's description of the denoiser on a deployment that
    /// prepares and admits `canvases`.
    pub(crate) fn denoiser_info(&self, canvases: &[Canvas]) -> Result<VideoDenoiserInfo> {
        let tasks = self.executed_tasks();
        if tasks.is_empty() {
            return Err(Error::Unsupported(format!(
                "the placed video denoiser serves {}, none of which this worker executes",
                self.facts.tasks.join(", ")
            )));
        }
        Ok(VideoDenoiserInfo {
            tasks,
            schedule_points: self.facts.num_steps + 1,
            video_shift: self.facts.video_shift,
            audio_shift: self.facts.audio_shift,
            canvases: canvases.to_vec(),
            max_sequence_rows: self.facts.max_sequence_rows,
            condition_tiles: self.facts.condition_tiles,
        })
    }

    pub(crate) fn legal_frame_count(&self, frames: u32) -> Result<u32> {
        let method = &self.methods.legal_frame_count;
        let value = self.denoiser(method, vec![Any::from(i64::from(frames))])?;
        narrow(self.integer(method, value)?, "frame count")
    }

    /// The smallest layout holding `size`.
    pub(crate) fn layout_size(&self, size: &VideoSize) -> Result<VideoLayout> {
        let value = self
            .denoiser(&self.methods.layout_size, size_arguments(size))
            .map_err(invalid_size)?;
        layout_of(value)
    }

    /// A sizing method of the denoiser component as a call for a job to
    /// make.
    fn asked(&self, method: &Function, arguments: Vec<Argument>) -> MethodCall {
        let mut all = vec![
            Argument::Handle(self.loaded.clone()),
            Argument::Text(self.denoiser.clone()),
        ];
        all.extend(arguments);
        MethodCall::new(method, all)
    }

    /// Record the call asking the smallest layout holding `size`
    /// ([`Self::layout_size`] as a call of `job`): its completion carries
    /// the layout's extents.
    pub(crate) fn ask_layout_size(&self, size: &VideoSize, job: &Job) {
        let arguments = size_values(size).into();
        job.call_returning(self.asked(&self.methods.layout_size, arguments));
    }

    /// Record the call asking whether `layout` holds `size`: the job's
    /// completion carries one for a layout that does, zero otherwise. A size
    /// the model rejects fails the job with the model's error
    /// ([`invalid_size`]).
    pub(crate) fn ask_holds(&self, layout: &VideoLayout, size: &VideoSize, job: &Job) {
        let mut arguments = vec![Argument::Integers(layout.values().to_vec())];
        arguments.extend(size_values(size));
        job.call_returning(self.asked(&self.methods.holds, arguments));
    }

    /// `layout` widened so its condition region holds `rows`.
    pub(crate) fn condition_layout(&self, layout: &VideoLayout, rows: u32) -> Result<VideoLayout> {
        layout_of(self.denoiser(
            &self.methods.condition_layout,
            vec![layout_arguments(layout), Any::from(i64::from(rows))],
        )?)
    }

    /// A modality's complete sample shape in `layout`.
    pub(crate) fn latent_shape(&self, modality: &str, layout: &VideoLayout) -> Result<Vec<u64>> {
        let method = &self.methods.latent_shape;
        let value = self.denoiser(method, vec![text(modality), layout_arguments(layout)])?;
        Ok(self
            .integers(method, value)?
            .into_iter()
            .map(|extent| extent.max(0) as u64)
            .collect())
    }

    /// Elements of a modality's complete sample in `layout`.
    pub(crate) fn latent_elements(&self, modality: &str, layout: &VideoLayout) -> Result<u64> {
        Ok(self.latent_shape(modality, layout)?.iter().product())
    }

    /// The request state of `layout` on this rank, asked on the calling
    /// thread (startup).
    pub(crate) fn layout_fields(&self, layout: &VideoLayout) -> Result<LayoutFields> {
        let methods = &self.methods;
        let of_layout = |method: &Function| self.denoiser(method, vec![layout_arguments(layout)]);
        let counts =
            |method: &Function| -> Result<Vec<i64>> { self.integers(method, of_layout(method)?) };
        let count = |method: &Function| -> Result<i64> { self.integer(method, of_layout(method)?) };

        self.fields(LayoutAnswers {
            names: strings(&methods.state_names, of_layout(&methods.state_names)?)?,
            dtypes: strings(&methods.state_dtypes, of_layout(&methods.state_dtypes)?)?,
            elements: counts(&methods.state_elements)?,
            rows: counts(&methods.sample_rows)?,
            draws: counts(&methods.noise_elements)?,
            condition_noise: count(&methods.condition_noise_elements)?,
            text_rows: count(&methods.text_condition_rows)?,
            workspace: count(&methods.workspace_elements)?,
        })
    }

    /// Record into `job` the calls that ask the request state of `layout`
    /// on this rank; [`Self::read_layout_fields`] reads the job's
    /// completion.
    pub(crate) fn ask_layout_fields(&self, layout: &VideoLayout, job: &Job) {
        let methods = &self.methods;
        let of_layout = |method: &Function| {
            self.asked(method, vec![Argument::Integers(layout.values().to_vec())])
        };
        job.call_texts(of_layout(&methods.state_names));
        job.call_texts(of_layout(&methods.state_dtypes));
        for method in [
            &methods.state_elements,
            &methods.sample_rows,
            &methods.noise_elements,
            &methods.condition_noise_elements,
            &methods.text_condition_rows,
            &methods.workspace_elements,
        ] {
            job.call_returning(of_layout(method));
        }
    }

    /// The request state of a layout from the completion of the job that
    /// asked it ([`Self::ask_layout_fields`]).
    pub(crate) fn read_layout_fields(&self, completion: &mut Completion) -> Result<LayoutFields> {
        let count = |values: Vec<i64>| {
            values.first().copied().ok_or_else(|| {
                Error::Interface("a layout sizing function returned no value".into())
            })
        };
        self.fields(LayoutAnswers {
            names: completion.texts()?,
            dtypes: completion.texts()?,
            elements: completion.returned()?,
            rows: completion.returned()?,
            draws: completion.returned()?,
            condition_noise: count(completion.returned()?)?,
            text_rows: count(completion.returned()?)?,
            workspace: count(completion.returned()?)?,
        })
    }

    /// A layout's request state from the model's answers.
    fn fields(&self, answers: LayoutAnswers) -> Result<LayoutFields> {
        let extent = |value: i64| value.max(0) as u64;
        let LayoutAnswers {
            names,
            dtypes,
            elements,
            rows,
            draws,
            condition_noise,
            text_rows,
            workspace,
        } = answers;
        let state = names
            .into_iter()
            .zip(dtypes)
            .zip(elements)
            .map(|((name, dtype), elements)| {
                Ok(StateField {
                    sample: self.facts.modalities.contains(&name),
                    name,
                    dtype: DType::parse(&dtype)?,
                    elements: extent(elements),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let rows: Vec<u64> = rows.into_iter().map(extent).collect();
        let samples = self
            .facts
            .modalities
            .iter()
            .zip(rows.as_chunks::<4>().0)
            .map(|(name, &[rows, width, start, stop])| {
                (
                    name.clone(),
                    SampleRows {
                        rows,
                        width,
                        local: (start, stop),
                    },
                )
            })
            .collect();
        let noise = self
            .facts
            .modalities
            .iter()
            .cloned()
            .zip(draws.into_iter().map(extent))
            .collect();
        Ok(LayoutFields {
            state,
            samples,
            noise,
            condition_noise: extent(condition_noise),
            text_condition: (
                extent(text_rows),
                u64::from(self.facts.text_condition_width),
            ),
            workspace: extent(workspace),
        })
    }

    /// The `(frames, canvas)` segment unit `unit` of a video decodes to.
    pub(crate) fn segment(&self, frames: u32, canvas: Canvas, unit: u32) -> Result<(u32, Canvas)> {
        let method = &self.methods.segment;
        let value = self.decoder(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
                Any::from(i64::from(unit)),
            ],
        )?;
        let values = self.integers(method, value)?;
        let [frames, width, height] = values[..] else {
            return Err(Error::Interface(
                "a segment is [frames, width, height]".into(),
            ));
        };
        Ok((
            narrow(frames, "segment")?,
            Canvas {
                width: narrow(width, "segment")?,
                height: narrow(height, "segment")?,
            },
        ))
    }

    /// Latent rows the window of unit `unit` of a `frames`-frame video at
    /// `canvas` reads.
    pub(crate) fn window_rows(&self, frames: u32, canvas: Canvas, unit: u32) -> Result<u64> {
        let method = &self.methods.window_rows;
        let value = self.decoder(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
                Any::from(i64::from(unit)),
            ],
        )?;
        Ok(self.integers(method, value)?.len() as u64)
    }

    /// The overlap a unit of `canvas` carries, as `(elements, dtype)`.
    pub(crate) fn overlap(&self, canvas: Canvas) -> Result<(u64, DType)> {
        let arguments = || {
            vec![
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
            ]
        };
        let elements = &self.methods.overlap_elements;
        let count = self.integer(elements, self.decoder(elements, arguments())?)?;
        let dtype = &self.methods.overlap_dtype;
        let name = tvm_ffi::String::try_from(self.decoder(dtype, arguments())?)
            .map_err(|error| Error::numerical(dtype.name(), &error))?;
        Ok((count.max(0) as u64, DType::parse(name.as_str())?))
    }

    /// Frames of each media unit of a video of `frames` frames.
    pub(crate) fn unit_frames(&self, frames: u32) -> Result<Vec<u32>> {
        let method = &self.methods.unit_frames;
        let value = self.decoder(method, vec![Any::from(i64::from(frames))])?;
        self.integers(method, value)?
            .into_iter()
            .map(|count| narrow(count, "unit frame count"))
            .collect()
    }

    /// Samples of the audio track generated with a video of `frames`.
    pub(crate) fn track_samples(&self, frames: u32) -> Result<u64> {
        let method = &self.methods.track_samples;
        let value = self.decoder(method, vec![Any::from(i64::from(frames))])?;
        Ok(self.integer(method, value)?.max(0) as u64)
    }

    /// Each of `units` audio units' samples of a `samples`-sample track.
    pub(crate) fn audio_units(&self, samples: u64, units: u32) -> Result<Vec<Range<u64>>> {
        let method = &self.methods.audio_units;
        let value = self.decoder(
            method,
            vec![Any::from(samples as i64), Any::from(i64::from(units))],
        )?;
        Ok(self
            .integers(method, value)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[start, stop]| start.max(0) as u64..stop.max(0) as u64)
            .collect())
    }

    /// Latent frames of a `samples`-sample track.
    pub(crate) fn audio_latent_frames(&self, samples: u64) -> Result<u32> {
        let method = &self.methods.audio_latent_frames;
        let value = self.decoder(method, vec![Any::from(samples as i64)])?;
        narrow(self.integer(method, value)?, "latent frame count")
    }

    /// The products each model component publishes, bounded for any request
    /// of at most `frames` frames, `tokens` prompt tokens and `rows`
    /// condition rows at any of `canvases`.
    pub(crate) fn outputs(
        &self,
        frames: u32,
        tokens: u32,
        rows: u32,
        dtype: &str,
        canvases: &[Canvas],
    ) -> Result<BTreeMap<String, Vec<OutputInfo>>> {
        let method = &self.methods.products;
        let outputs = self.denoiser(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(tokens)),
                Any::from(i64::from(rows)),
                text(dtype),
                Any::from(integer_list(canvases.iter().flat_map(|canvas| {
                    [i64::from(canvas.width), i64::from(canvas.height)]
                }))),
            ],
        )?;
        let mut result: BTreeMap<String, Vec<OutputInfo>> = BTreeMap::new();
        for (component, output) in self.output_infos(outputs)? {
            result.entry(component).or_default().push(output);
        }
        Ok(result)
    }

    /// The media reader's condition products.
    pub(crate) fn reader_outputs(
        &self,
        frames: u32,
        tokens: u32,
        rows: u32,
    ) -> Result<Vec<OutputInfo>> {
        let method = &self.methods.reader_products;
        let value = self.denoiser(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(tokens)),
                Any::from(i64::from(rows)),
            ],
        )?;
        Ok(self
            .output_infos(value)?
            .into_iter()
            .map(|(_, output)| output)
            .collect())
    }

    /// Read the outputs a product method returned (`Outputs`), column by
    /// column, as each output's component and bound.
    fn output_infos(&self, outputs: Any) -> Result<Vec<(String, OutputInfo)>> {
        let [owners, names, dtypes, ranks, extents, variable, raster] =
            &self.methods.output_columns;
        let strings = |column: &Function| -> Result<Vec<String>> {
            let values: tvm_ffi::Array<tvm_ffi::String> = column.call_as(&[(&outputs).into()])?;
            Ok(values
                .iter()
                .map(|value| value.as_str().to_owned())
                .collect())
        };
        let numbers = |column: &Function| -> Result<Vec<i64>> {
            self.integers(column, column.call_values(std::slice::from_ref(&outputs))?)
        };
        let (owners, names, dtypes) = (strings(owners)?, strings(names)?, strings(dtypes)?);
        let (ranks, extents) = (numbers(ranks)?, numbers(extents)?);
        let (variable, raster) = (numbers(variable)?, numbers(raster)?);
        let mut extents = extents.into_iter();
        let mut result = Vec::with_capacity(names.len());
        for (index, name) in names.into_iter().enumerate() {
            let component = &owners[index];
            let extent = |value: i64| {
                u32::try_from(value).map_err(|_| {
                    Error::Unsupported(format!("result {component}.{name} has an invalid extent"))
                })
            };
            let dtype = DType::from_name(&dtypes[index])
                .and_then(|dtype| uniserve_ipc::DType::try_from(dtype).ok())
                .ok_or_else(|| {
                    Error::Unsupported(format!("result {component}.{name} has no protocol dtype"))
                })?;
            let dims = extents
                .by_ref()
                .take(ranks[index] as usize)
                .enumerate()
                .map(|(axis, size)| {
                    Ok(if variable[index] == axis as i64 {
                        DimBound::Device { max: extent(size)? }
                    } else {
                        DimBound::Static(extent(size)?)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // A negative axis stands for an output that is not laid out at
            // the canvas.
            let raster_axes = match (raster[2 * index], raster[2 * index + 1]) {
                (height, width) if height >= 0 && width >= 0 => Some(RasterAxes {
                    height: extent(height)?,
                    width: extent(width)?,
                }),
                _ => None,
            };
            result.push((
                component.clone(),
                OutputInfo {
                    name,
                    dtype,
                    shape_bound: ShapeBound { dims },
                    raster_axes,
                },
            ));
        }
        Ok(result)
    }
}

pub(crate) fn size_arguments(size: &VideoSize) -> Vec<Any> {
    vec![
        size.descriptor(),
        size.conditions_descriptor(),
        size.spans_descriptor(),
    ]
}

/// A size as the arguments of a recorded call: what [`size_arguments`]
/// passes, each list as its integers.
fn size_values(size: &VideoSize) -> [Argument; 3] {
    [
        Argument::Integers(size.values()),
        Argument::Integers(size.condition_values()),
        Argument::Integers(size.span_values()),
    ]
}

fn layout_of(value: Any) -> Result<VideoLayout> {
    let values = integers(value).map_err(|error| Error::numerical(DENOISER, &error))?;
    VideoLayout::from_values(&values)
}

/// A size the model rejects is the request's error, not the worker's.
pub(crate) fn invalid_size(error: Error) -> Error {
    match error {
        Error::Numerical { kind, message, .. } if kind == "InputError" || kind == "ValueError" => {
            Error::Invalid(message)
        }
        error => error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, dtype: DType, shape: &[i64], sample: bool) -> StateField {
        StateField {
            name: name.into(),
            dtype,
            elements: shape.iter().product::<i64>() as u64,
            sample,
        }
    }

    /// One layout's fields on a rank that holds every sample row but the
    /// video's last `shared` rows: a video and an audio sample, a timestep
    /// table and `rows` prompt rows of width 4.
    fn shared_layout(video: i64, audio: i64, table: i64, rows: u64, shared: i64) -> LayoutFields {
        let sample = |rows: i64, local: i64| SampleRows {
            rows: rows as u64,
            width: 8,
            local: (0, local as u64),
        };
        LayoutFields {
            state: vec![
                field("video", DType::F32, &[video - shared, 8], true),
                field("audio", DType::F32, &[audio, 8], true),
                field("tables", DType::I64, &[table], false),
            ],
            samples: vec![
                ("video".into(), sample(video, video - shared)),
                ("audio".into(), sample(audio, audio)),
            ],
            noise: vec![
                ("video".into(), (video * 8) as u64),
                ("audio".into(), (audio * 8) as u64),
            ],
            condition_noise: 0,
            text_condition: (rows, 4),
            workspace: 0,
        }
    }

    /// One layout's fields on a rank that holds every sample row.
    fn layout(video: i64, audio: i64, table: i64, rows: u64) -> LayoutFields {
        shared_layout(video, audio, table, rows, 0)
    }

    #[test]
    fn samples_start_on_aligned_offsets_within_their_pages() {
        // 40 video rows of 8 elements take 320 elements, aligned to 512.
        let samples = SampleLayout::new(&layout(40, 3, 5, 16), 256).unwrap();
        assert_eq!(samples.offsets, vec![0, 512]);
        assert_eq!(samples.elements, vec![320, 24]);
        assert_eq!(samples.pages, 3);
        assert!(
            SampleLayout::new(
                &LayoutFields {
                    samples: vec![(
                        "depth".into(),
                        SampleRows {
                            rows: 1,
                            width: 1,
                            local: (0, 1)
                        }
                    )],
                    ..layout(1, 1, 1, 1)
                },
                256,
            )
            .is_err()
        );
    }

    #[test]
    fn the_envelope_holds_every_layout_field_at_its_largest() {
        let pages = |pages| SamplePages {
            page_units: 256,
            pages,
        };
        let envelope =
            StateEnvelope::new(&[layout(40, 3, 5, 16), layout(8, 30, 9, 4)], pages(4)).unwrap();
        let size = |fields: &[SlotField], name: &str| {
            fields
                .iter()
                .find(|field| field.name == name)
                .map(|field| field.shape[0])
        };
        // Samples live in latent pages; only tables and the conditioning
        // are slot state on the device.
        assert_eq!(size(&envelope.device, "video"), None);
        assert_eq!(size(&envelope.device, "tables"), Some(9));
        assert_eq!(size(&envelope.device, "text_condition"), Some(64));
        assert_eq!(envelope.slot_bytes(), 9 * 8 + 64 * 2);
        // Host sources cover every field; noise and condition draws are FP32.
        assert_eq!(size(&envelope.host, "video_source"), Some(320));
        assert_eq!(size(&envelope.host, "audio_noise"), Some(240));
        assert_eq!(size(&envelope.host, "condition_noise"), Some(1));
        // The scratch holds a slot's pages, any rank's share of a sample.
        assert_eq!(envelope.samples, 4 * 256);
        // A layout whose samples exceed a slot's pages is refused.
        assert!(StateEnvelope::new(&[layout(40, 3, 5, 16)], pages(2)).is_err());

        // A rank's share of a sample grows and shrinks with the request's
        // prompt and conditions, so its source holds the complete sample
        // wherever the prepared layouts leave the rank fewer rows.
        let envelope = StateEnvelope::new(&[shared_layout(40, 3, 5, 16, 30)], pages(4)).unwrap();
        assert_eq!(size(&envelope.host, "video_source"), Some(320));

        let mut changed = layout(8, 30, 9, 4);
        changed.state[2].dtype = DType::I32;
        assert!(StateEnvelope::new(&[layout(40, 3, 5, 16), changed], pages(4)).is_err());
    }

    #[test]
    fn slots_own_consecutive_pages_after_the_padding_page() {
        let pages = SamplePages {
            page_units: 256,
            pages: 3,
        };
        assert_eq!(pages.for_slot(1), 1..4);
        assert_eq!(pages.for_slot(2), 4..7);
        assert_eq!(pages.pool_pages(2), 7);
    }

    #[test]
    fn condition_latents_follow_their_rows_across_encoding_rounds() {
        // A keyframe of 4 rows, a reference video of 10 rows with 3 audio
        // rows and a voice of 2 audio rows; the encoder published the
        // visual rows in rounds of 6, 5 and 3 rows.
        let conditions = [(4, 0), (10, 3), (0, 2)];
        let latents = condition_spans(&conditions, &[6, 5, 3], Some(5)).unwrap();
        assert_eq!(
            latents,
            vec![
                vec![(0, 0, 4)],
                // The video continues through the second round into the third.
                vec![(0, 4, 2), (1, 0, 5), (2, 0, 3)],
                // The audio read follows the three visual rounds.
                vec![(3, 0, 3)],
                vec![(3, 3, 2)],
            ]
        );

        // Reads that hold other rows than the request planned are refused.
        assert!(condition_spans(&conditions, &[6, 5], Some(5)).is_err());
        assert!(condition_spans(&conditions, &[6, 5, 3], Some(4)).is_err());
        assert!(condition_spans(&conditions, &[6, 5, 3], None).is_err());
        // Without audio, every read is a visual round.
        assert_eq!(
            condition_spans(&[(4, 0)], &[4], None).unwrap(),
            vec![vec![(0, 0, 4)]]
        );
    }
}
