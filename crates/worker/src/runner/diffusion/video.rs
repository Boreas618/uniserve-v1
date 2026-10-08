//! Video denoising over Rust-owned buffers.
//!
//! A video request's samples live in the latent pool: two banks of pages
//! whose slot owns `pages` consecutive pages after page 0. A solver step
//! gathers the committed bank's pages into the runner's sample scratch,
//! copies the request's state (its slot rows of every state table and of
//! the retained conditioning) into the fixed state buffers, writes the
//! evaluation index, runs the bound step (`video.bind_denoiser`'s `step`, or
//! the layout's captured graph) and scatters the scratch into the other
//! bank. Every one of those buffers is allocated here and bound to Python
//! once; a step crosses the interface with a layout id alone.
//!
//! Layouts ([`VideoLayout`]) fix the shapes a step evaluates. Startup
//! layouts are prepared, warmed and captured before serving and stay
//! resident; a conditioned request whose layout was not prepared at startup
//! prepares a transient layout, evaluated eagerly, which the least recently
//! used transient layout makes room for once no request uses it.

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use tvm_ffi::{Any, AnyView};
use uniserve_cuda::{CopyBatch, SourceOrder, Stream, memcpy_async, memset_d8, memset_d32};
use uniserve_ipc::{Call, LatentParams};

use crate::ffi::{Backing, DType, Fields, Function, Functions, Handle, integer_list, list};
use crate::model::video::{LayoutFields, VideoModel};
use crate::request::video::VideoLayout;
use crate::runner::GraphPool;
use crate::runner::graphs::{GraphMode, GraphTable, Lookup};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{LatentExport, LatentImport};
use crate::storage::{RequestSlots, SlotField};
use crate::transfer::Retirement;
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
            elements.push(field.elements());
            cursor += aligned(field.elements());
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
                if !field.sample {
                    grow(
                        &mut device,
                        field.name.clone(),
                        field.dtype,
                        field.elements(),
                        false,
                    )?;
                }
                grow(
                    &mut host,
                    format!("{}_source", field.name),
                    field.dtype,
                    field.elements(),
                    true,
                )?;
            }
            for (name, shape) in &fields.noise {
                let elements = shape.iter().map(|&extent| extent.max(0) as u64).product();
                grow(
                    &mut host,
                    format!("{name}_noise"),
                    DType::F32,
                    elements,
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

/// The text conditioner's input, length and output, bound once.
pub(crate) struct ConditionerBuffers {
    /// `[capacity, width]` prompt features, zero past the prompt.
    pub(crate) input: StorageView,
    /// `[1]` int32 prompt length.
    pub(crate) lengths: StorageView,
    /// `[capacity, conditioning width]` BF16 refined rows.
    pub(crate) output: StorageView,
}

/// The step functions of a bound denoiser.
struct Steps {
    prepare_layout: Function,
    release_layout: Function,
    step: Function,
    capture: Function,
    replay: Function,
    release: Function,
    prepare_request: Function,
    encode_conditions: Function,
    prepare_conditioner: Option<Function>,
    condition: Option<Function>,
    capture_conditioner: Option<Function>,
    kernels: Function,
    close: Function,
}

/// How a prepared layout serves steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Prepared,
    Warmed,
}

struct Layout {
    key: VideoLayout,
    fields: LayoutFields,
    samples: SampleLayout,
    /// State tables copied from the slot's rows before each step, as
    /// `(field, bytes)`.
    copies: Vec<(usize, u64)>,
    phase: Phase,
    startup: bool,
    /// Live requests evaluating in the layout.
    references: usize,
}

/// The latent pool of a denoising rank: pages imported from and exported
/// to other ranks retire through the transfer layer's retirements.
type Trajectories =
    LatentPool<Arc<LatentImport<Arc<Retirement>>>, Arc<LatentExport<Arc<Retirement>>>>;

/// One rank's video denoiser: its bound steps, buffers, latent pool and
/// layouts.
pub(crate) struct VideoDenoiser {
    steps: Steps,
    stream: Stream,
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
    conditioner: Option<ConditionerBuffers>,
    pages: SamplePages,
    storage: LatentStorage,
    pool: Trajectories,
    layouts: Vec<Option<Layout>>,
    ids: HashMap<VideoLayout, usize>,
    /// Transient layout ids, least recently used first.
    transient: VecDeque<usize>,
    transient_capacity: usize,
    graphs: GraphTable<usize>,
    conditioner_graphs: GraphTable<u32>,
    /// The graph pool captures allocate in; `None` serves steps eagerly.
    pool_handle: Option<Arc<GraphPool<Handle>>>,
    /// Prompt capacities the conditioner serves.
    text_capacities: Vec<u32>,
}

/// What [`VideoDenoiser::new`] binds.
pub(crate) struct DenoiserSetup<'a> {
    pub(crate) bind: &'a Function,
    pub(crate) loaded: &'a Handle,
    pub(crate) component: &'a str,
    pub(crate) stream: Stream,
    pub(crate) device: i32,
    pub(crate) maximum: VideoLayout,
    pub(crate) envelope: StateEnvelope,
    pub(crate) pages: SamplePages,
    pub(crate) slots: u32,
    /// Prompt capacities the conditioner serves, and its features' shape
    /// at the largest: `(rows, width, dtype)`; `None` without a conditioner
    /// on this rank.
    pub(crate) conditioning: Option<(Vec<u32>, u64, DType)>,
    pub(crate) conditioning_width: u64,
    pub(crate) attention: String,
    /// The worker's workspace allocation callback.
    pub(crate) allocate: tvm_ffi::Function,
    pub(crate) graph_pool: Option<Arc<GraphPool<Handle>>>,
    pub(crate) transient_layouts: usize,
}

fn tensors(fields: impl IntoIterator<Item = (String, StorageView)>) -> Result<Any> {
    let mut map = Fields::new();
    for (name, view) in fields {
        map = map.with(&name, view.to_tensor()?);
    }
    Ok(map.into_any())
}

fn bytes_of(view: &StorageView) -> u64 {
    view.numel() * view.dtype().bytes() as u64
}

impl VideoDenoiser {
    /// Allocate the runner's buffers and latent pool and bind them.
    pub(crate) fn new(setup: DenoiserSetup<'_>) -> Result<Self> {
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
        let conditioner = match &setup.conditioning {
            Some((capacities, width, dtype)) => {
                let rows = i64::from(capacities.iter().copied().max().unwrap_or(0));
                Some(ConditionerBuffers {
                    input: StorageView::contiguous(
                        memory(rows as u64 * width * dtype.bytes() as u64)? as Arc<dyn Backing>,
                        0,
                        *dtype,
                        &[rows, *width as i64],
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
                })
            }
            None => None,
        };
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

        let names = |fields: &[SlotField]| {
            fields
                .iter()
                .map(|field| field.name.clone())
                .collect::<Vec<_>>()
        };
        let state_names = names(&setup.envelope.device);
        let mut request = Fields::new()
            .text("component", setup.component)
            .with("stream", setup.stream.raw() as i64)
            .text("attention", &setup.attention)
            .with("allocate", setup.allocate.clone())
            .with("maximum", setup.maximum.descriptor())
            .with("samples", samples.to_tensor()?)
            .with("step", step.to_tensor()?)
            .with(
                "state",
                tensors(state_names.iter().cloned().zip(state.iter().cloned()))?,
            )
            .with(
                "banks",
                tensors(banks.views().into_iter().map(|(name, view)| {
                    let rows = view.shape()[0];
                    (
                        name,
                        view.reshape(view.dtype(), &[rows, -1]).unwrap_or(view),
                    )
                }))?,
            )
            .with("host", tensors(host.views())?);
        if let Some(buffers) = &conditioner {
            request = request.with(
                "conditioner",
                Fields::new()
                    .with("input", buffers.input.to_tensor()?)
                    .with("lengths", buffers.lengths.to_tensor()?)
                    .with("output", buffers.output.to_tensor()?)
                    .into_any(),
            );
        }
        let functions = Functions::new(
            "video.bind_denoiser",
            setup
                .bind
                .call(&[setup.loaded.view(), AnyView::from(&request.into_any())])?,
        )?;
        let steps = Steps {
            prepare_layout: functions.get("prepare_layout")?,
            release_layout: functions.get("release_layout")?,
            step: functions.get("step")?,
            capture: functions.get("capture")?,
            replay: functions.get("replay")?,
            release: functions.get("release")?,
            prepare_request: functions.get("prepare_request")?,
            encode_conditions: functions.get("encode_conditions")?,
            prepare_conditioner: functions.optional("prepare_conditioner")?,
            condition: functions.optional("condition")?,
            capture_conditioner: functions.optional("capture_conditioner")?,
            kernels: functions.get("kernels")?,
            close: functions.get("close")?,
        };
        if conditioner.is_some() != steps.condition.is_some() {
            return Err(Error::Interface(
                "the denoiser binding's conditioner does not match its buffers".into(),
            ));
        }
        let graphs = GraphTable::new(
            format!("{}.denoiser", setup.component),
            steps.release.clone(),
        );
        let conditioner_graphs = GraphTable::new(
            format!("{}.conditioner", setup.component),
            steps.release.clone(),
        );
        Ok(Self {
            steps,
            stream: setup.stream,
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
            conditioner_graphs,
            pool_handle: setup.graph_pool,
            text_capacities: setup
                .conditioning
                .map(|(capacities, _, _)| capacities)
                .unwrap_or_default(),
        })
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Prepare `key` under a new id. Startup layouts are never released.
    fn prepare(&mut self, model: &VideoModel, key: VideoLayout, startup: bool) -> Result<usize> {
        let fields = model.layout_fields(&key)?;
        let samples = SampleLayout::new(&fields, self.pages.page_units)?;
        if samples.pages > self.pages.pages {
            return Err(Error::Invalid(
                "a video layout's samples exceed the request's latent pages".into(),
            ));
        }
        let mut copies = Vec::new();
        let tables = fields
            .state
            .iter()
            .filter(|field| !field.sample)
            .map(|field| (field.name.clone(), field.bytes()))
            .chain(std::iter::once((
                "text_condition".to_owned(),
                fields.text_condition.0 * fields.text_condition.1 * 2,
            )));
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
        let offsets = integer_list(samples.offsets.iter().map(|&offset| offset as i64));
        self.steps.prepare_layout.invoke(&[
            AnyView::from(&Any::from(id as i64)),
            AnyView::from(&key.descriptor()),
            AnyView::from(&Any::from(offsets)),
        ])?;
        self.layouts[id] = Some(Layout {
            key,
            fields,
            samples,
            copies,
            phase: Phase::Prepared,
            startup,
            references: 0,
        });
        self.ids.insert(key, id);
        Ok(id)
    }

    /// Prepare a startup layout once; the largest is prepared first so it
    /// sizes the shared workspace.
    pub(crate) fn prepare_startup(
        &mut self,
        model: &VideoModel,
        key: VideoLayout,
    ) -> Result<usize> {
        match self.ids.get(&key) {
            Some(&id) => Ok(id),
            None => self.prepare(model, key, true),
        }
    }

    /// The id of the layout a request evaluates in, preparing a transient
    /// layout (and releasing an unused one) when it was not prepared.
    pub(crate) fn acquire(&mut self, model: &VideoModel, key: VideoLayout) -> Result<usize> {
        let id = match self.ids.get(&key) {
            Some(&id) => {
                if let Some(position) = self.transient.iter().position(|&entry| entry == id) {
                    self.transient.remove(position);
                    self.transient.push_back(id);
                }
                id
            }
            None => {
                while self.transient.len() >= self.transient_capacity {
                    let Some(position) = self.transient.iter().position(|&entry| {
                        self.layout(entry)
                            .is_ok_and(|layout| layout.references == 0)
                    }) else {
                        break;
                    };
                    let Some(evicted) = self.transient.remove(position) else {
                        break;
                    };
                    self.release_layout(evicted)?;
                }
                let id = self.prepare(model, key, false)?;
                self.transient.push_back(id);
                id
            }
        };
        self.layout_mut(id)?.references += 1;
        Ok(id)
    }

    /// End a request's use of its layout.
    pub(crate) fn release(&mut self, id: usize) -> Result<()> {
        let layout = self.layout_mut(id)?;
        layout.references = layout
            .references
            .checked_sub(1)
            .ok_or_else(|| Error::Invariant("video layout reference underflow".into()))?;
        Ok(())
    }

    fn release_layout(&mut self, id: usize) -> Result<()> {
        let Some(layout) = self.layouts.get_mut(id).and_then(Option::take) else {
            return Ok(());
        };
        // The binding's views must outlive the layout's last step. Evicting
        // a transient layout is rare, so the stream drains here.
        self.stream.synchronize()?;
        self.ids.remove(&layout.key);
        self.steps
            .release_layout
            .invoke(&[AnyView::from(&Any::from(id as i64))])
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

    /// The layout a prepared id evaluates.
    pub(crate) fn key(&self, id: usize) -> Result<VideoLayout> {
        Ok(self.layout(id)?.key)
    }

    /// The function a host task calls to prepare a request: it draws the
    /// request's noise and fills its slot's host sources for its layout.
    pub(crate) fn preparation(&self) -> Function {
        self.steps.prepare_request.clone()
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

    /// Copy a request's prepared host sources into its slot: each
    /// modality's sample into bank 1 of its pages and each table into its
    /// slot rows. The host task that filled them has completed.
    pub(crate) fn initialize(
        &mut self,
        id: usize,
        slot: usize,
        call: &Call,
        params: &LatentParams,
    ) -> Result<LatentUpdate> {
        let pages = self.slot_pages(slot, params)?;
        self.pool.require_initial(slot, &pages)?;
        let layout = self.layout(id)?;
        let first = pages[0] as u64;
        let base = self.bank_address(1, first)?;
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (index, (name, _)) in layout.fields.samples.iter().enumerate() {
            let source = self.host.slot(&format!("{name}_source"), slot as u32)?;
            copies.push(
                base + layout.samples.offsets[index] * 4,
                source.address(),
                (layout.samples.elements[index] * 4) as usize,
            );
        }
        for field in layout.fields.state.iter().filter(|field| !field.sample) {
            let source = self
                .host
                .slot(&format!("{}_source", field.name), slot as u32)?;
            let target = self.banks.slot(&field.name, slot as u32)?;
            copies.push(target.address(), source.address(), field.bytes() as usize);
        }
        // SAFETY: every destination lies in this slot's pages or rows, which
        // no other request owns; the pinned sources stay allocated for the
        // runner's lifetime and are rewritten only by the slot's next request.
        unsafe { copies.submit(&self.stream) }?;
        LatentUpdate::for_call(slot, call, params)
    }

    /// Refine a request's prompt features and retain them in its slot's
    /// conditioning: `features` holds the prompt's `count` rows; the rows of
    /// the layout's conditioning past the prompt become zero.
    pub(crate) fn condition(
        &mut self,
        id: usize,
        slot: usize,
        features: &StorageView,
        count: u64,
    ) -> Result<GraphMode> {
        let layout = self.layout(id)?;
        let (rows, width) = layout.fields.text_condition;
        let target = self.banks.slot("text_condition", slot as u32)?;
        let Some(buffers) = &self.conditioner else {
            return Err(Error::Unsupported(
                "this pipeline stage holds no text conditioner".into(),
            ));
        };
        let capacity = self
            .text_capacities
            .iter()
            .copied()
            .filter(|&capacity| u64::from(capacity) >= count)
            .min()
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
        // SAFETY: the conditioner buffers and the slot's rows are owned by
        // this runner; every range lies inside them and the stream orders
        // the copies before and after the conditioner.
        unsafe {
            memcpy_async(
                input,
                features.address(),
                (count * row_bytes) as usize,
                &self.stream,
            )?;
            memset_d8(
                input + count * row_bytes,
                0,
                ((u64::from(capacity) - count) * row_bytes) as usize,
                &self.stream,
            )?;
            memset_d32(lengths, count as u32, 1, &self.stream)?;
        }
        let mode = self.run_conditioner(capacity)?;
        // SAFETY: as above.
        unsafe {
            memcpy_async(
                target.address(),
                output,
                (count * output_row) as usize,
                &self.stream,
            )?;
            memset_d8(
                target.address() + count * output_row,
                0,
                ((rows - count) * output_row) as usize,
                &self.stream,
            )?;
        }
        Ok(mode)
    }

    fn run_conditioner(&mut self, capacity: u32) -> Result<GraphMode> {
        let condition = self
            .steps
            .condition
            .clone()
            .ok_or_else(|| Error::Unsupported("this rank holds no text conditioner".into()))?;
        let argument = Any::from(i64::from(capacity));
        // Without a graph pool every capacity runs eagerly.
        let lookup = match self.pool_handle {
            Some(_) => self.conditioner_graphs.lookup(&capacity, None)?,
            None => Lookup::Capture,
        };
        let mode = match lookup {
            Lookup::Replay(graph) => {
                self.steps.replay.invoke(&[graph.view()])?;
                GraphMode::Replay
            }
            Lookup::Capture => {
                condition.invoke(&[AnyView::from(&argument)])?;
                GraphMode::Eager
            }
        };
        self.conditioner_graphs.record(mode);
        Ok(mode)
    }

    /// Write a request's encoded conditions into its slot's conditioning,
    /// after the prompt's rows.
    pub(crate) fn encode_conditions(
        &self,
        id: usize,
        slot: usize,
        arguments: &[Any],
        latents: Vec<Any>,
    ) -> Result<()> {
        let mut values = vec![Any::from(slot as i64), Any::from(id as i64)];
        values.extend(arguments.iter().cloned());
        values.push(Any::from(list(latents)));
        self.steps.encode_conditions.call_values(&values).map(drop)
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
    ) -> Result<(LatentUpdate, GraphMode)> {
        if params.step_count != 1 {
            return Err(Error::Invalid(
                "a video denoising call advances exactly one solver step".into(),
            ));
        }
        let pages = self.slot_pages(slot, params)?;
        let step = i64::from(params.start_step);
        let read = self.pool.current_bank(
            slot,
            step,
            step + 1,
            i64::from(params.latent_units),
            i64::from(params.height),
            i64::from(params.width),
            &pages,
        )?;
        let write = 1 - read;
        self.pool.require_writable(write, &pages)?;
        let first = pages[0] as u64;
        let (copies, layout_pages) = {
            let layout = self.layout(id)?;
            (layout.copies.clone(), layout.samples.pages)
        };
        let page_bytes = self.pages.page_units * 4;

        let mut gather = CopyBatch::new(SourceOrder::Stream);
        gather.push(
            self.samples.address(),
            self.bank_address(read, first)?,
            (layout_pages * page_bytes) as usize,
        );
        for &(index, bytes) in &copies {
            let name = &self.envelope.device[index].name;
            gather.push(
                self.state[index].address(),
                self.banks.slot(name, slot as u32)?.address(),
                bytes as usize,
            );
        }
        // SAFETY: the scratch, state buffers, slot rows and pages are owned by
        // this runner; steps of the runner are ordered on its stream.
        unsafe {
            gather.submit(&self.stream)?;
            memset_d32(self.step.address(), params.start_step, 1, &self.stream)?;
            memset_d32(self.step.address() + 4, 0, 1, &self.stream)?;
        }

        let mode = self.run_step(id)?;

        // SAFETY: as above; the scatter follows the step on the stream.
        unsafe {
            memcpy_async(
                self.bank_address(write, first)?,
                self.samples.address(),
                (layout_pages * page_bytes) as usize,
                &self.stream,
            )?;
        }
        Ok((LatentUpdate::for_call(slot, call, params)?, mode))
    }

    fn run_step(&mut self, id: usize) -> Result<GraphMode> {
        let argument = Any::from(id as i64);
        let mode = match self.graphs.lookup(&id, None) {
            Ok(Lookup::Replay(graph)) => {
                self.steps.replay.invoke(&[graph.view()])?;
                GraphMode::Replay
            }
            // Transient layouts and uncaptured startup layouts run eagerly.
            _ => {
                self.steps.step.invoke(&[AnyView::from(&argument)])?;
                GraphMode::Eager
            }
        };
        self.graphs.record(mode);
        Ok(mode)
    }

    /// The final step's local sample of each modality, as `[rows, width]`
    /// views of the scratch (`None` for a modality this rank holds no rows
    /// of), in modality order.
    pub(crate) fn final_samples(&self, id: usize) -> Result<Vec<Option<StorageView>>> {
        let layout = self.layout(id)?;
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
                let offset = self.samples.offset() + (layout.samples.offsets[index] * 4) as usize;
                StorageView::contiguous(
                    Arc::clone(self.samples.memory()),
                    offset,
                    DType::F32,
                    &[local as i64, rows.width as i64],
                )
                .map(Some)
            })
            .collect()
    }

    /// The local rows of each modality's complete sample in a layout.
    pub(crate) fn sample_rows(&self, id: usize) -> Result<Vec<crate::model::video::SampleRows>> {
        Ok(self
            .layout(id)?
            .fields
            .samples
            .iter()
            .map(|(_, rows)| *rows)
            .collect())
    }

    /// Validate trajectory commits before any becomes visible.
    pub(crate) fn validate(&self, updates: &[LatentUpdate]) -> Result<()> {
        self.pool.validate_updates(updates)
    }

    /// Make validated commits visible.
    pub(crate) fn apply(&mut self, updates: &[LatentUpdate]) {
        drop(self.pool.apply_updates(updates));
    }

    /// Release retired requests' pages.
    pub(crate) fn release_slots(&mut self, slots: &[usize]) -> Result<()> {
        let slots: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
        drop(self.pool.release_slots(&slots)?);
        self.pool.reap();
        Ok(())
    }

    /// Run one placeholder step of a startup layout on slot `slot`, whose
    /// host sources a preparation filled, to prepare its kernels and plans
    /// before any capture. The scratch's values are discarded.
    pub(crate) fn warm(&mut self, id: usize, slot: usize) -> Result<()> {
        let layout = self.layout(id)?;
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (index, (name, _)) in layout.fields.samples.iter().enumerate() {
            let source = self.host.slot(&format!("{name}_source"), slot as u32)?;
            copies.push(
                self.samples.address() + layout.samples.offsets[index] * 4,
                source.address(),
                (layout.samples.elements[index] * 4) as usize,
            );
        }
        for field in layout.fields.state.iter().filter(|field| !field.sample) {
            let source = self
                .host
                .slot(&format!("{}_source", field.name), slot as u32)?;
            let index = self
                .envelope
                .device
                .iter()
                .position(|candidate| candidate.name == field.name)
                .ok_or_else(|| Error::Invariant("layout field left the envelope".into()))?;
            copies.push(
                self.state[index].address(),
                source.address(),
                field.bytes() as usize,
            );
        }
        let text = self
            .envelope
            .device
            .iter()
            .position(|field| field.name == "text_condition")
            .ok_or_else(|| Error::Invariant("the envelope lacks text conditioning".into()))?;
        let text_bytes = bytes_of(&self.state[text]);
        // SAFETY: the scratch and state buffers are owned by this runner; the
        // stream is drained below before the slot's sources are reused.
        unsafe {
            copies.submit(&self.stream)?;
            memset_d8(
                self.state[text].address(),
                0,
                text_bytes as usize,
                &self.stream,
            )?;
            memset_d32(self.step.address(), 0, 2, &self.stream)?;
        }
        self.steps
            .step
            .invoke(&[AnyView::from(&Any::from(id as i64))])?;
        self.layout_mut(id)?.phase = Phase::Warmed;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Capture a warmed startup layout's step into the runner's graph pool.
    pub(crate) fn capture(&mut self, id: usize) -> Result<bool> {
        let Some(pool) = &self.pool_handle else {
            return Ok(false);
        };
        let layout = self.layout(id)?;
        if !layout.startup || layout.phase != Phase::Warmed {
            return Err(Error::Invariant(
                "only warmed startup layouts are captured".into(),
            ));
        }
        if self.graphs.contains(&id, None) {
            return Ok(false);
        }
        let graph = Handle::new(
            self.steps
                .capture
                .call(&[AnyView::from(&Any::from(id as i64)), pool.value.view()])?,
        );
        self.graphs.insert(id, None, graph)?;
        self.graphs.record(GraphMode::Capture);
        Ok(true)
    }

    /// Prepare the conditioner for every prompt capacity, largest first.
    pub(crate) fn prepare_conditioner(&mut self) -> Result<()> {
        let Some(prepare) = self.steps.prepare_conditioner.clone() else {
            return Ok(());
        };
        let mut capacities = self.text_capacities.clone();
        capacities.sort_unstable_by(|left, right| right.cmp(left));
        for capacity in capacities {
            prepare.invoke(&[AnyView::from(&Any::from(i64::from(capacity)))])?;
        }
        Ok(())
    }

    /// Warm the conditioner at every capacity (largest first) on zero
    /// features, then capture each.
    pub(crate) fn warm_conditioner(&mut self, capture: bool) -> Result<()> {
        let Some(buffers) = &self.conditioner else {
            return Ok(());
        };
        let input_bytes = bytes_of(&buffers.input);
        // SAFETY: the conditioner's buffers belong to this runner.
        unsafe {
            memset_d8(
                buffers.input.address(),
                0,
                input_bytes as usize,
                &self.stream,
            )?;
        }
        let mut capacities = self.text_capacities.clone();
        capacities.sort_unstable_by(|left, right| right.cmp(left));
        for &capacity in &capacities {
            // SAFETY: as above.
            unsafe { memset_d32(buffers.lengths.address(), capacity, 1, &self.stream)? };
            if let Some(condition) = &self.steps.condition {
                condition.invoke(&[AnyView::from(&Any::from(i64::from(capacity)))])?;
            }
        }
        self.stream.synchronize()?;
        if !capture {
            return Ok(());
        }
        let (Some(pool), Some(function)) = (&self.pool_handle, &self.steps.capture_conditioner)
        else {
            return Ok(());
        };
        for capacity in capacities {
            let graph = Handle::new(function.call(&[
                AnyView::from(&Any::from(i64::from(capacity))),
                pool.value.view(),
            ])?);
            self.conditioner_graphs.insert(capacity, None, graph)?;
        }
        Ok(())
    }

    /// Refuse further captures.
    pub(crate) fn seal(&mut self) {
        self.graphs.seal();
        self.conditioner_graphs.seal();
    }

    /// The kernels the binding selected.
    pub(crate) fn kernels(&self) -> &Function {
        &self.steps.kernels
    }

    /// Release graphs and the binding. An aborted close keeps everything
    /// device work may still address until the process exits.
    #[allow(clippy::mem_forget)]
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut failure = None;
        for result in [
            self.graphs.close(aborted),
            self.conditioner_graphs.close(aborted),
        ] {
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        }
        if let Err(error) = self
            .steps
            .close
            .invoke(&[AnyView::from(&Any::from(aborted))])
        {
            failure.get_or_insert(error);
        }
        failure.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::video::{SampleRows, StateField};

    fn field(name: &str, dtype: DType, shape: &[i64], sample: bool) -> StateField {
        StateField {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            sample,
        }
    }

    /// One layout's fields: a video and an audio sample, a timestep table
    /// and `rows` prompt rows of width 4.
    fn layout(video: i64, audio: i64, table: i64, rows: u64) -> LayoutFields {
        let sample = |rows: i64| SampleRows {
            rows: rows as u64,
            width: 8,
            local: (0, rows as u64),
        };
        LayoutFields {
            state: vec![
                field("video", DType::F32, &[video, 8], true),
                field("audio", DType::F32, &[audio, 8], true),
                field("tables", DType::I64, &[table], false),
            ],
            samples: vec![
                ("video".into(), sample(video)),
                ("audio".into(), sample(audio)),
            ],
            noise: vec![
                ("video".into(), vec![1, video, 8]),
                ("audio".into(), vec![1, audio, 8]),
            ],
            condition_noise: 0,
            text_condition: (rows, 4),
        }
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
}
