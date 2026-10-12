//! The rank's cross-call tensor store over Rust-owned storage.
//!
//! [`TensorStore`] owns every value a call publishes for later calls on this
//! rank or on other ranks (a *product*): its placement, its write state, its
//! read leases, its imports from other ranks and its exports to them.
//!
//! - **Placement.** A product with a scheduler buffer allocation is
//!   *persistent*: a range of the device's [`BufferPool`] arena, which is an
//!   exportable allocation so CUDA VMM exports stay in place. A scalar
//!   product without one lives in a *relay lane*: one element of a
//!   per-(device, dtype, field) arena of `(request_capacity + 1) *
//!   relay_depth` elements, indexed by request slot and lane.
//! - **Writes.** Producers write the bound view (see
//!   [`TensorStore::producer_view`]) and declare it produced, or hand a
//!   value that is copied natively into it. Writes never convert dtypes: a
//!   producer writes the store's dtype. A producer fence (an event on the
//!   explicit producer stream) orders consumers after the write.
//! - **Reads.** Consumers lease committed products; their consuming stream
//!   waits on the producer fence, and completing the lease records a reader
//!   fence on that stream.
//! - **Retirement.** Logical release is not reuse: a buffer's storage
//!   returns only after release, with no reader, every transfer and export
//!   successfully retired, and every fence complete. Failed or unknown
//!   completions keep the storage forever.

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use uniserve_core::CallId;
use uniserve_cuda::{CopyBatch, Event, SourceOrder, Stream};
use uniserve_ipc::{
    BufferAllocation, BufferId, Call, CallKind, DimBound, FeatureKind, MediaCall, RequestKey,
    TensorExport, TensorRef, TensorTransfer, TransferHandle, subtract_region,
};

use super::buffer::BufferPool;
use super::memory::{Memory, StorageView, device_name, parse_dtype};
use super::tensor::{
    Backing, Buffer, Fence, StoreState, TensorImport, TensorRead, WriteState, buffer_key,
};
use crate::ffi::{DType, Device};
use crate::stream::{Fill, Issue};
use crate::transfer::exports::{self, Location};
use crate::transfer::{ReadRef, Retirement, Spans, Transports, copy_planner, fetch};
use crate::{Error, EventPool, Result};

fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    // No user code runs under these locks; a panic cannot leave a partial
    // update that later readers would misinterpret.
    value.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The numerical interval of an image product's pixel values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueRange {
    /// `[0, 1]`.
    Unit,
    /// `[-1, 1]`.
    SignedUnit,
}

impl ValueRange {
    fn name(self) -> &'static str {
        match self {
            Self::Unit => "unit",
            Self::SignedUnit => "signed_unit",
        }
    }
}

/// What a product's values represent, beyond its dtype and shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metadata {
    /// An immutable image; zero height and width mean undeclared dimensions.
    Image {
        height: u32,
        width: u32,
        value_range: Option<ValueRange>,
    },
    /// An encoder feature with its spatial dimensions.
    Feature { height: u32, width: u32 },
}

impl Metadata {
    fn validate(self) -> Result<()> {
        match self {
            Self::Image { height, width, .. } if (height == 0) != (width == 0) => Err(
                Error::Invalid("device-product image dimensions must be complete".into()),
            ),
            Self::Feature { height, width } if height == 0 || width == 0 => Err(Error::Invalid(
                "encoder feature dimensions must be positive".into(),
            )),
            _ => Ok(()),
        }
    }

    fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Image { height, width, .. } | Self::Feature { height, width } => (height, width),
        }
    }
}

/// A scalar written into a one-element product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Int(i64),
    Float(f64),
}

/// Scheduler placement of a group of outputs. Maps are keyed by buffer id.
#[derive(Clone, Copy, Default)]
pub struct OutputPlacement<'a> {
    /// Request slots of the owning requests, for relay outputs.
    pub request_slots: Option<&'a HashMap<RequestKey, usize>>,
    /// Buffer-pool allocations of persistent outputs.
    pub allocations: Option<&'a HashMap<BufferId, BufferAllocation>>,
    /// The logical region this rank stores of a sharded output.
    pub regions: Option<&'a HashMap<BufferId, Vec<Range<u64>>>>,
    /// Actual logical shapes overriding the reference's bound.
    pub shapes: Option<&'a HashMap<BufferId, Vec<u64>>>,
}

struct BufferViews {
    reference: TensorRef,
    /// The bound view: the logical tensor, or this rank's region of it.
    view: StorageView,
    /// This rank's region of a sharded product, in logical coordinates.
    region: Option<Vec<Range<u64>>>,
    metadata: Option<Metadata>,
    /// The whole reserved storage (the logical shape when the allocation
    /// covers it), which imports widen into.
    storage: StorageView,
}

struct StoredBuffer {
    state: Mutex<Buffer<ReadRef, Arc<Retirement>>>,
    views: Mutex<BufferViews>,
}

/// A handle to one stored product. Read leases and imports retain the same
/// buffer through logical release and storage reuse.
#[derive(Clone)]
pub struct ProductBuffer(Arc<StoredBuffer>);

impl Deref for ProductBuffer {
    type Target = Mutex<Buffer<ReadRef, Arc<Retirement>>>;

    fn deref(&self) -> &Self::Target {
        &self.0.state
    }
}

impl ProductBuffer {
    pub fn id(&self) -> BufferId {
        lock(self).id
    }

    pub fn reference(&self) -> TensorRef {
        lock(&self.0.views).reference.clone()
    }

    /// The bound view producers write.
    pub fn view(&self) -> StorageView {
        lock(&self.0.views).view.clone()
    }

    pub fn region(&self) -> Option<Vec<Range<u64>>> {
        lock(&self.0.views).region.clone()
    }

    pub fn metadata(&self) -> Option<Metadata> {
        lock(&self.0.views).metadata
    }

    /// A write deferred to a later call and not yet released.
    pub fn deferred(&self) -> bool {
        let buffer = lock(self);
        buffer.state == WriteState::Deferred && !buffer.released
    }

    fn views(&self) -> MutexGuard<'_, BufferViews> {
        lock(&self.0.views)
    }
}

struct StoredImport {
    state: Mutex<TensorImport<ProductBuffer, ReadRef>>,
    destination: StorageView,
    metadata: Option<Metadata>,
}

/// Missing coverage of one product materialized once and shared by its
/// concurrent readers.
#[derive(Clone)]
struct ImportHandle(Arc<StoredImport>);

impl Deref for ImportHandle {
    type Target = Mutex<TensorImport<ProductBuffer, ReadRef>>;

    fn deref(&self) -> &Self::Target {
        &self.0.state
    }
}

/// One consumer's lease of a product. The value view stays stable even when
/// another import widens the product's resident coverage.
pub struct ProductRead {
    value: StorageView,
    region: Option<Vec<Range<u64>>>,
    metadata: Option<Metadata>,
    lease: Mutex<TensorRead<ProductBuffer, ImportHandle>>,
}

impl ProductRead {
    pub fn value(&self) -> &StorageView {
        &self.value
    }

    pub fn region(&self) -> Option<&[Range<u64>]> {
        self.region.as_deref()
    }

    pub fn metadata(&self) -> Option<Metadata> {
        self.metadata
    }

    pub fn complete(&self) -> bool {
        lock(&self.lease).complete
    }

    /// The transport reads materializing an import (empty for an ordinary
    /// lease); callers wait for them before `TensorStore::wait_import`.
    pub fn reads(&self) -> Vec<Arc<crate::transfer::Read>> {
        let imported = lock(&self.lease).imported.clone();
        imported.map_or_else(Vec::new, |imported| {
            lock(&imported)
                .tickets
                .iter()
                .map(|ticket| Arc::clone(ticket.read()))
                .collect()
        })
    }
}

type State = StoreState<ProductBuffer, ImportHandle, StorageView>;

/// The element type of a protocol product dtype.
fn product_dtype(dtype: uniserve_ipc::DType) -> Result<DType> {
    parse_dtype(dtype.storage_name())
}

/// Static extents and the device-actual axis at its bound; a scalar product
/// is stored as shape `[1]`.
fn device_shape(reference: &TensorRef) -> Vec<u64> {
    let dims: Vec<u64> = reference
        .shape_bound
        .dims
        .iter()
        .map(|dim| match dim {
            DimBound::Static(extent) => u64::from(*extent),
            DimBound::Device { max } => u64::from(*max),
        })
        .collect();
    if dims.is_empty() { vec![1] } else { dims }
}

fn elements(shape: &[u64]) -> Result<u64> {
    shape
        .iter()
        .try_fold(1_u64, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| Error::Invalid("tensor extent overflows".into()))
}

fn shape_bytes(shape: &[u64], dtype: DType) -> Result<u64> {
    elements(shape)?
        .checked_mul(dtype.bytes() as u64)
        .ok_or_else(|| Error::Invalid("buffer tensor shape has an invalid byte extent".into()))
}

fn full(shape: &[u64]) -> Vec<Range<u64>> {
    shape.iter().map(|&extent| 0..extent).collect()
}

fn region_shape(region: &[Range<u64>]) -> Vec<u64> {
    region.iter().map(|range| range.end - range.start).collect()
}

fn within(region: &[Range<u64>], shape: &[u64]) -> bool {
    region.len() == shape.len()
        && region
            .iter()
            .zip(shape)
            .all(|(range, &extent)| range.start <= range.end && range.end <= extent)
}

fn signed(shape: &[u64]) -> Vec<i64> {
    shape.iter().map(|&extent| extent as i64).collect()
}

/// The published `extent` elements of `view` as `shape`, preserving the
/// strides of an exact shard.
fn value_view(view: &StorageView, shape: &[u64], extent: u64) -> Result<StorageView> {
    if view.shape() == signed(shape).as_slice() {
        return Ok(view.clone());
    }
    if !view.is_contiguous() || extent > view.numel() || elements(shape)? != extent {
        return Err(Error::Invariant(
            "a product value exceeds its contiguous storage".into(),
        ));
    }
    StorageView::contiguous(
        Arc::clone(view.memory()),
        view.offset(),
        view.dtype(),
        &signed(shape),
    )
}

/// A one-element value's bits in `dtype` and their width in bytes.
fn scalar_bits(value: Scalar, dtype: DType) -> Result<(u64, usize)> {
    let float = match value {
        Scalar::Int(value) => value as f64,
        Scalar::Float(value) => value,
    };
    let integer = match value {
        Scalar::Int(value) => value,
        // Float fills of integer storage truncate toward zero.
        Scalar::Float(value) => value as i64,
    };
    Ok(match dtype {
        DType::Bool => (u64::from(float != 0.0), 1),
        DType::U8 => (u64::from(integer as u8), 1),
        DType::I8 => (u64::from(integer as i8 as u8), 1),
        DType::I16 => (u64::from(integer as i16 as u16), 2),
        DType::I32 => (u64::from(integer as i32 as u32), 4),
        DType::U32 => (u64::from(integer as u32), 4),
        DType::I64 => (integer as u64, 8),
        DType::F16 => (u64::from(f16_bits(float as f32)), 2),
        DType::BF16 => (u64::from(bf16_bits(float as f32)), 2),
        DType::F32 => (u64::from((float as f32).to_bits()), 4),
        DType::F64 => (float.to_bits(), 8),
        DType::F8E4M3 | DType::F8E5M2 => {
            return Err(Error::Unsupported(
                "scalar products are not stored as FP8".into(),
            ));
        }
    })
}

/// IEEE binary16 bits of `value`, rounded to nearest even.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let biased = exponent - 127 + 15;
    if biased >= 0x1f {
        return sign | 0x7c00;
    }
    if biased <= 0 {
        if biased < -10 {
            return sign;
        }
        // Subnormal: the value in units of 2^-24, rounded to nearest even.
        let significand = mantissa | 0x80_0000;
        let shift = (14 - biased) as u32;
        let truncated = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = truncated
            + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 == 1));
        return sign | rounded as u16;
    }
    let truncated = ((biased as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    // A carry out of the mantissa correctly increments the exponent, up to
    // infinity.
    let rounded =
        truncated + u32::from(remainder > 0x1000 || (remainder == 0x1000 && truncated & 1 == 1));
    sign | rounded as u16
}

/// bfloat16 bits of `value`, rounded to nearest even.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// The rank's cross-call tensor store.
pub struct TensorStore {
    max_feature_bytes: usize,
    devices: Vec<Device>,
    buffer_pool: Mutex<BufferPool<Arc<Memory>>>,
    events: Arc<EventPool>,
    /// Committed export locations per local product, revoked on release.
    exports: Mutex<HashMap<BufferId, Vec<Location>>>,
    state: Mutex<State>,
}

impl TensorStore {
    /// A store of at most `capacity` persistent non-feature products per
    /// device and `byte_capacity` bytes of relay arenas, with features of at
    /// most `max_feature_bytes` on `devices`, `request_capacity` request
    /// slots of `relay_depth` relay lanes each (both zero without relays).
    #[expect(
        clippy::too_many_arguments,
        reason = "the store's complete startup facts"
    )]
    pub fn new(
        capacity: usize,
        byte_capacity: usize,
        max_feature_bytes: usize,
        devices: Vec<Device>,
        request_capacity: usize,
        relay_depth: usize,
        buffer_pool: BufferPool<Arc<Memory>>,
        events: Arc<EventPool>,
    ) -> Result<Self> {
        if max_feature_bytes == 0 || byte_capacity == 0 {
            return Err(Error::Invalid("tensor store capacities are invalid".into()));
        }
        if (request_capacity == 0) != (relay_depth == 0) {
            return Err(Error::Invalid(
                "request-relay dimensions must be complete".into(),
            ));
        }
        Ok(Self {
            max_feature_bytes,
            devices,
            buffer_pool: Mutex::new(buffer_pool),
            events,
            exports: Mutex::new(HashMap::new()),
            state: Mutex::new(State::new(
                capacity,
                byte_capacity,
                request_capacity,
                relay_depth,
            )),
        })
    }

    pub fn max_feature_bytes(&self) -> usize {
        self.max_feature_bytes
    }

    pub fn capacity(&self) -> usize {
        lock(&self.state).capacity
    }

    pub fn byte_capacity(&self) -> usize {
        lock(&self.state).byte_capacity
    }

    pub fn request_capacity(&self) -> usize {
        lock(&self.state).request_capacity
    }

    pub fn relay_depth(&self) -> usize {
        lock(&self.state).relay_depth
    }

    /// Return retired buffers' pool ranges and event references.
    fn reclaim(&self, state: &mut State) -> Result<()> {
        let retired = state.reclaim()?;
        let mut released: HashMap<usize, (Arc<Event>, usize)> = HashMap::new();
        for (buffer, backing) in retired {
            if let Backing::Persistent(binding) = backing {
                lock(&self.buffer_pool).release(&binding)?;
            }
            for event in lock(&buffer).events() {
                released
                    .entry(Arc::as_ptr(event) as usize)
                    .or_insert_with(|| (Arc::clone(event), 0))
                    .1 += 1;
            }
        }
        for (event, count) in released.into_values() {
            self.events.release(&event, count)?;
        }
        Ok(())
    }

    fn release_backing(&self, state: &mut State, buffer: &ProductBuffer) -> Result<()> {
        if let Backing::Persistent(binding) = state.release_backing(buffer)? {
            lock(&self.buffer_pool).release(&binding)?;
        }
        Ok(())
    }

    /// A fence after the producer's writes on `stream`, retained `count`
    /// times (one reference per buffer it fences). Host values need none.
    fn producer_fence(
        &self,
        device: Device,
        stream: impl Issue,
        count: usize,
    ) -> Result<Option<Fence>> {
        let Device::Cuda(index) = device else {
            return Ok(None);
        };
        let producer = stream.stream().ok_or_else(|| {
            Error::Invalid("a device product write requires its producer stream".into())
        })?;
        if producer.device().0 != index {
            return Err(Error::Invalid(
                "a producer stream is on another device than its product".into(),
            ));
        }
        let event = self.events.fence(stream, false)?;
        if count > 1 {
            self.events.retain(&event, index, count - 1)?;
        }
        Ok(Some(Fence {
            event,
            stream: producer.raw(),
        }))
    }

    /// Order `stream` after the buffer's producer, once per fence.
    fn wait_producer(
        buffer: &Buffer<ReadRef, Arc<Retirement>>,
        stream: impl Issue,
        waited: &mut HashSet<usize>,
    ) -> Result<()> {
        if let Some(fence) = &buffer.producer {
            let consumer = stream.stream().ok_or_else(|| {
                Error::Invalid("device product consumption requires its consumer stream".into())
            })?;
            if waited.insert(Arc::as_ptr(&fence.event) as usize) && consumer.raw() != fence.stream {
                stream.wait(&fence.event)?;
            }
        }
        Ok(())
    }

    /// Wake the service when a released buffer's fences complete. `stream`
    /// is what records the fences that are not recorded yet.
    fn wake_retirement(&self, buffer: &ProductBuffer, stream: impl Issue) -> Result<()> {
        let events: Vec<_> = lock(buffer).events().cloned().collect();
        for event in events {
            if !event.query()? {
                stream.wake(&self.events, &event)?;
            }
        }
        Ok(())
    }

    /// Wake on retirement and cancel the imports of a never-produced buffer.
    fn retire_buffer(&self, buffer: &ProductBuffer) -> Result<()> {
        self.wake_retirement(buffer, None::<&Stream>)?;
        let transfers: Vec<_> = {
            let value = lock(buffer);
            if value.state.produced() {
                Vec::new()
            } else {
                value
                    .transfers
                    .iter()
                    .map(|read| Arc::clone(read.read()))
                    .collect()
            }
        };
        for read in transfers {
            read.cancel()?;
        }
        Ok(())
    }

    fn require_reference(state: &State, reference: &TensorRef) -> Result<ProductBuffer> {
        let buffer = state.require_reference(reference.buffer_id())?.clone();
        if buffer.views().reference != *reference {
            return Err(Error::Invalid(
                "stale device-product logical generation".into(),
            ));
        }
        Ok(buffer)
    }

    /// Bind each group of outputs atomically; a failure releases every group
    /// bound by this call.
    pub fn bind_output_groups(
        &self,
        groups: &[Vec<(TensorRef, Device)>],
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<Vec<ProductBuffer>>> {
        let mut state = lock(&self.state);
        let mut outputs = Vec::new();
        let mut acquired = Vec::new();
        for group in groups.iter().filter(|group| !group.is_empty()) {
            match self.bind_group(&mut state, group, placement) {
                Ok(buffers) => {
                    acquired.extend(buffers.iter().cloned());
                    outputs.push(buffers);
                }
                Err(error) => {
                    self.rollback(&mut state, acquired)?;
                    return Err(error);
                }
            }
        }
        Ok(outputs)
    }

    /// Bind one group of outputs: persistent outputs into their pool
    /// allocations, or scalar outputs into relay lanes.
    pub fn bind_outputs(
        &self,
        bindings: &[(TensorRef, Device)],
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<ProductBuffer>> {
        let mut state = lock(&self.state);
        self.bind_group(&mut state, bindings, placement)
    }

    /// Reserve encoder features in their pool allocations.
    pub fn reserve_features(
        &self,
        bindings: &[(TensorRef, Device)],
        allocations: &HashMap<BufferId, BufferAllocation>,
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<ProductBuffer>> {
        let mut state = lock(&self.state);
        self.bind_persistent(&mut state, bindings, allocations, placement, true)
    }

    fn logical_shape(reference: &TensorRef, placement: OutputPlacement<'_>) -> Vec<u64> {
        placement
            .shapes
            .and_then(|shapes| shapes.get(&reference.buffer_id()))
            .cloned()
            .unwrap_or_else(|| device_shape(reference))
    }

    fn bind_group(
        &self,
        state: &mut State,
        bindings: &[(TensorRef, Device)],
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<ProductBuffer>> {
        if bindings.is_empty() {
            return Ok(Vec::new());
        }
        let mut persistent = 0;
        for (reference, _) in bindings {
            let shape = Self::logical_shape(reference, placement);
            if !reference.shape_bound.contains_shape(&shape) {
                return Err(Error::Invalid(
                    "tensor binding shape disagrees with its logical bounds".into(),
                ));
            }
            if let Some(region) = placement
                .regions
                .and_then(|regions| regions.get(&reference.buffer_id()))
                && !within(region, &shape)
            {
                return Err(Error::Invalid(
                    "tensor binding region disagrees with its logical bounds".into(),
                ));
            }
            if placement
                .allocations
                .is_some_and(|allocations| allocations.contains_key(&reference.buffer_id()))
            {
                persistent += 1;
            }
        }
        if persistent > 0 {
            if persistent != bindings.len() {
                return Err(Error::Invalid(
                    "persistent-buffer bindings cannot share a generic group".into(),
                ));
            }
            let allocations = placement.allocations.ok_or_else(|| {
                Error::Invariant("persistent buffers lost their placement".into())
            })?;
            return self.bind_persistent(state, bindings, allocations, placement, false);
        }
        if let Some(request_slots) = placement.request_slots {
            return self.bind_relays(state, bindings, request_slots);
        }
        Err(Error::Invalid(
            "tensor output requires a buffer allocation or request relay slot".into(),
        ))
    }

    fn bind_persistent(
        &self,
        state: &mut State,
        bindings: &[(TensorRef, Device)],
        allocations: &HashMap<BufferId, BufferAllocation>,
        placement: OutputPlacement<'_>,
        feature: bool,
    ) -> Result<Vec<ProductBuffer>> {
        self.reclaim(state)?;
        let mut native = Vec::new();
        let mut candidates = Vec::new();
        for (reference, device) in bindings {
            let id = reference.buffer_id();
            let allocation = *allocations.get(&id).ok_or_else(|| {
                Error::Invalid("persistent output has no buffer allocation".into())
            })?;
            if feature {
                if !matches!(
                    reference.dtype,
                    uniserve_ipc::DType::F16 | uniserve_ipc::DType::BF16 | uniserve_ipc::DType::F32
                ) {
                    return Err(Error::Invalid(
                        "encoder feature dtype is unsupported".into(),
                    ));
                }
                let bytes = reference.max_bytes();
                if bytes > self.max_feature_bytes as u64 {
                    return Err(Error::Resource(
                        "encoder feature exceeds the fixed feature byte capacity",
                    ));
                }
                if !self.devices.contains(device) {
                    return Err(Error::Invalid(
                        "encoder feature names an undeclared device".into(),
                    ));
                }
            }
            native.push((id, device_name(*device)));
            candidates.push((reference, *device, allocation));
        }
        state.validate_bindings(&native, feature)?;

        let mut buffers = Vec::new();
        let result = (|| -> Result<()> {
            for (reference, device, allocation) in candidates {
                let dtype = product_dtype(reference.dtype)?;
                let shape = Self::logical_shape(reference, placement);
                let region = placement
                    .regions
                    .and_then(|regions| regions.get(&reference.buffer_id()))
                    .filter(|region| **region != full(&shape))
                    .cloned();
                let local = region
                    .as_deref()
                    .map_or_else(|| shape.clone(), region_shape);
                // An allocation covering the whole logical tensor stores it at
                // logical coordinates, so imports can widen the shard later.
                let full_storage =
                    region.is_some() && allocation.bytes >= shape_bytes(&shape, dtype)?;
                let storage_shape = if full_storage { &shape } else { &local };

                let (binding, arena) = {
                    let mut pool = lock(&self.buffer_pool);
                    let (binding, arena) = pool.bind(
                        allocation,
                        &device_name(device),
                        shape_bytes(storage_shape, dtype)?,
                        dtype.bytes() as u64,
                    )?;
                    (binding, Arc::clone(arena))
                };
                let views = (|| -> Result<(StorageView, StorageView)> {
                    let storage = StorageView::contiguous(
                        arena as Arc<dyn crate::ffi::Backing>,
                        binding.physical_offset() as usize,
                        dtype,
                        &signed(storage_shape),
                    )?;
                    let view = match (&region, full_storage) {
                        (Some(region), true) => storage.region(region)?,
                        _ => storage.clone(),
                    };
                    Ok((view, storage))
                })();
                let (view, storage) = match views {
                    Ok(views) => views,
                    Err(error) => {
                        lock(&self.buffer_pool).release(&binding)?;
                        return Err(error);
                    }
                };
                let buffer = create_buffer(
                    reference,
                    device,
                    view,
                    storage,
                    shape,
                    region,
                    feature,
                    Backing::Persistent(binding),
                );
                state
                    .buffers
                    .insert(buffer_key(reference.buffer_id()), buffer.clone());
                buffers.push(buffer);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.rollback(state, buffers)?;
            return Err(error);
        }
        Ok(buffers)
    }

    fn bind_relays(
        &self,
        state: &mut State,
        bindings: &[(TensorRef, Device)],
        request_slots: &HashMap<RequestKey, usize>,
    ) -> Result<Vec<ProductBuffer>> {
        if state.request_capacity == 0 || state.relay_depth == 0 {
            return Err(Error::Resource("worker has no request-relay arena"));
        }
        self.reclaim(state)?;
        let mut seen = HashSet::new();
        let mut fields = HashMap::<((String, usize, RequestKey, CallId), DType), usize>::new();
        let mut buffers = Vec::new();
        let result = (|| -> Result<()> {
            for (reference, device) in bindings {
                let id = reference.buffer_id();
                let key = buffer_key(id);
                if !seen.insert(key) {
                    return Err(Error::Invalid(
                        "request-relay registration repeats an output identity".into(),
                    ));
                }
                if let Some(existing) = state.buffers.get(&key) {
                    let message = if lock(existing).state != WriteState::Committed {
                        "request-relay output already has a candidate"
                    } else if existing.views().reference != *reference {
                        "stale request-relay logical generation"
                    } else {
                        "request-relay output is already registered"
                    };
                    return Err(Error::Invalid(message.into()));
                }
                if id.generation == 0 {
                    return Err(Error::Invalid(
                        "request-relay registration requires a positive logical generation".into(),
                    ));
                }
                let slot = request_slots
                    .get(&reference.request_key)
                    .copied()
                    .unwrap_or(0);
                let shape = device_shape(reference);
                if slot == 0 || slot > state.request_capacity || shape.iter().any(|&dim| dim != 1) {
                    return Err(Error::Invalid(
                        "request-relay output has an invalid slot or scalar shape".into(),
                    ));
                }

                let name = device_name(*device);
                let dtype = product_dtype(reference.dtype)?;
                let call = (name.clone(), slot, id.owner, id.producer_call_id);
                let field = fields.entry((call, dtype)).or_default();
                let field_index = *field;
                *field += 1;
                let lane = state.relay_lane(id, &name, slot)?;
                let dtype_key = crate::storage::memory::dtype_name(dtype).to_owned();
                let relay = ((name.clone(), slot, lane), dtype_key.clone(), field_index);
                let view = match state.views.get(&relay) {
                    Some(view) => view.clone(),
                    None => {
                        let arena_key = (name, dtype_key, field_index);
                        if !state.arenas.contains_key(&arena_key) {
                            let count = state
                                .request_capacity
                                .checked_add(1)
                                .and_then(|rows| rows.checked_mul(state.relay_depth))
                                .ok_or(Error::Resource("request-relay byte capacity overflows"))?;
                            let bytes = count * dtype.bytes();
                            let projected = state.arena_capacity(bytes)?;
                            let memory = match device {
                                Device::Cuda(index) => Memory::device(*index, bytes)?,
                                Device::Cpu | Device::CudaHost => Memory::host(bytes),
                            };
                            let arena = StorageView::contiguous(
                                memory as Arc<dyn crate::ffi::Backing>,
                                0,
                                dtype,
                                &[count as i64],
                            )?;
                            state.arenas.insert(arena_key.clone(), arena);
                            state.allocated_bytes = projected;
                        }
                        let arena = state.arenas.get(&arena_key).ok_or_else(|| {
                            Error::Invariant("request-relay arena disappeared".into())
                        })?;
                        let view = arena.narrow(0, (slot * state.relay_depth + lane) as i64, 1)?;
                        state.views.insert(relay.clone(), view.clone());
                        view
                    }
                };
                let buffer = create_buffer(
                    reference,
                    *device,
                    view.clone(),
                    view,
                    vec![1],
                    None,
                    false,
                    Backing::Relay(relay.clone()),
                );
                state.bind_relay(id, &relay)?;
                state.buffers.insert(key, buffer.clone());
                buffers.push(buffer);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.rollback(state, buffers)?;
            return Err(error);
        }
        Ok(buffers)
    }

    fn rollback(&self, state: &mut State, buffers: Vec<ProductBuffer>) -> Result<()> {
        for buffer in buffers.into_iter().rev() {
            state.buffers.remove(&buffer_key(buffer.id()));
            self.release_backing(state, &buffer)?;
        }
        Ok(())
    }

    /// The bound view a producer writes before declaring it produced.
    pub fn producer_view(&self, buffer: &ProductBuffer) -> Result<StorageView> {
        let state = lock(&self.state);
        state.require_buffer(buffer)?;
        if lock(buffer).state.produced() {
            return Err(Error::Invariant(
                "device product was published more than once".into(),
            ));
        }
        Ok(buffer.view())
    }

    /// Publish a value the producer wrote into the bound view: `shape` is
    /// the value's actual shape, fenced on the producer `stream`.
    pub fn produced(
        &self,
        buffer: &ProductBuffer,
        shape: &[u64],
        stream: Option<&Stream>,
        metadata: Option<Metadata>,
    ) -> Result<StorageView> {
        self.events.reap()?;
        let state = lock(&self.state);
        state.require_buffer(buffer)?;
        let view = buffer.view();
        let value = if buffer.region().is_some() {
            view
        } else {
            value_view(&view, shape, elements(shape)?)?
        };
        self.write_value(buffer, &value, stream, metadata)
    }

    /// Copy `value` into the product's bound view (no copy when `value` is
    /// that view) on the producer `stream`, then fence it.
    pub fn write(
        &self,
        buffer: &ProductBuffer,
        value: &StorageView,
        stream: Option<&Stream>,
        metadata: Option<Metadata>,
    ) -> Result<StorageView> {
        self.events.reap()?;
        let state = lock(&self.state);
        state.require_buffer(buffer)?;
        self.write_value(buffer, value, stream, metadata)
    }

    fn write_value(
        &self,
        buffer: &ProductBuffer,
        value: &StorageView,
        stream: impl Issue,
        metadata: Option<Metadata>,
    ) -> Result<StorageView> {
        if let Some(metadata) = metadata {
            metadata.validate()?;
        }
        let feature = {
            let state = lock(buffer);
            if state.state.produced() {
                return Err(Error::Invariant(
                    "device product was published more than once".into(),
                ));
            }
            state.feature
        };
        let (reference, target, region) = {
            let views = buffer.views();
            (
                views.reference.clone(),
                views.view.clone(),
                views.region.clone(),
            )
        };
        match metadata {
            Some(Metadata::Feature { .. }) if !feature => {
                return Err(Error::Invalid(
                    "feature metadata requires feature admission".into(),
                ));
            }
            Some(Metadata::Feature { .. }) => {}
            _ if feature => {
                return Err(Error::Invalid(
                    "encoder feature requires spatial metadata".into(),
                ));
            }
            _ => {}
        }
        // Producers write the store's dtype: a feature's cast belongs to the
        // producing computation.
        if value.dtype() != target.dtype() {
            return Err(Error::Invalid(
                "tensor export changes its declared dtype".into(),
            ));
        }
        let shape: Vec<u64> = value.shape().iter().map(|&extent| extent as u64).collect();
        let matches = match &region {
            Some(region) => shape == region_shape(region),
            None => reference.shape_bound.contains_shape(&shape),
        };
        if !matches {
            return Err(Error::Invalid(format!(
                "tensor export of shape {shape:?} changes the declared shape of output {} of {:?}",
                reference.output_index, reference.producer_call_id
            )));
        }
        let extent = value.numel();
        if extent > target.numel() {
            return Err(Error::Invariant(
                "device product exceeds its registered shape bound".into(),
            ));
        }
        let view = match region {
            Some(_) => target,
            None => value_view(&target, &shape, extent)?,
        };

        // An exact alias needs no copy; noncontiguous shard strides stay intact.
        if view.address() != value.address() || view.strides() != value.strides() {
            let destination = Spans::single(view.clone())?;
            let source = Spans::single(value.clone())?;
            match (view.device(), value.device()) {
                (Device::Cuda(device), _) | (_, Device::Cuda(device)) => {
                    if stream.stream().is_none() {
                        return Err(Error::Invalid(
                            "a device product write requires its producer stream".into(),
                        ));
                    }
                    let mut plan = copy_planner(Some(device), SourceOrder::Stream)?;
                    destination.plan_copy(&source, &mut plan)?;
                    // SAFETY: the destination is reserved for this product and
                    // the producer keeps its value until its stream passes.
                    unsafe { stream.copy_strided(&plan) }?;
                }
                _ => {
                    let mut plan = copy_planner(None, SourceOrder::Stream)?;
                    destination.plan_copy(&source, &mut plan)?;
                    // SAFETY: host operands, copied in the issuer's order; the
                    // destination has no reader yet.
                    unsafe { stream.host().copy_strided(&plan) }?;
                }
            }
        }

        let fence = self.producer_fence(view.device(), stream, 1)?;
        lock(buffer).produced(
            shape.iter().map(|&extent| extent as usize).collect(),
            extent as usize,
            fence,
        );
        buffer.views().metadata = metadata;
        Ok(view)
    }

    /// Copy one scalar per product from a contiguous source column of the
    /// products' dtype, in one batched copy on the producer `stream`.
    pub fn write_scalars(
        &self,
        writes: &[ProductBuffer],
        values: &StorageView,
        stream: impl Issue,
    ) -> Result<Vec<StorageView>> {
        self.events.reap()?;
        let state = lock(&self.state);
        let mut targets = Vec::with_capacity(writes.len());
        let mut seen = HashSet::new();
        for write in writes {
            state.require_buffer(write)?;
            if lock(write).state.produced() || !seen.insert(Arc::as_ptr(&write.0) as usize) {
                return Err(Error::Invariant(
                    "device product was published more than once".into(),
                ));
            }
            targets.push(write.view());
        }
        let Some(first) = targets.first() else {
            return Ok(targets);
        };
        if values.numel() != targets.len() as u64 || !values.is_contiguous() {
            return Err(Error::Invalid(
                "batched device-product export requires one contiguous scalar per output".into(),
            ));
        }
        if targets.iter().any(|target| {
            target.numel() != 1
                || target.device() != first.device()
                || target.dtype() != first.dtype()
        }) || values.dtype() != first.dtype()
        {
            return Err(Error::Invalid(
                "batched device-product export spans incompatible storage".into(),
            ));
        }

        let width = first.dtype().bytes();
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for (index, target) in targets.iter().enumerate() {
            batch.push(
                target.address(),
                values.address() + (index * width) as u64,
                width,
            );
        }
        match first.device() {
            Device::Cuda(_) => {
                if stream.stream().is_none() {
                    return Err(Error::Invalid(
                        "a device product write requires its producer stream".into(),
                    ));
                }
                // SAFETY: each destination is one reserved element; the
                // producer keeps its column until its stream passes the copy.
                unsafe { stream.copy(&batch) }?;
            }
            // SAFETY: host elements of reserved products, copied in the
            // producer's order.
            _ => unsafe { stream.host().copy(&batch) }?,
        }

        let fence = self.producer_fence(first.device(), stream, writes.len())?;
        for write in writes {
            lock(write).produced(vec![1], 1, fence.clone());
        }
        Ok(targets)
    }

    /// Fill a one-element product with `value` (its bit pattern in the
    /// product's dtype) on the producer `stream`.
    pub fn write_scalar(
        &self,
        buffer: &ProductBuffer,
        value: Scalar,
        stream: impl Issue,
    ) -> Result<StorageView> {
        self.events.reap()?;
        let state = lock(&self.state);
        state.require_buffer(buffer)?;
        if lock(buffer).state.produced() {
            return Err(Error::Invariant(
                "device product was published more than once".into(),
            ));
        }
        let view = value_view(&buffer.view(), &[1], 1)?;
        let (bits, width) = scalar_bits(value, view.dtype())?;
        let address = view.address();
        // An eight-byte element is two words, low word first.
        let words = [(address, bits as u32), (address + 4, (bits >> 32) as u32)];
        let fills = words[..width.div_ceil(4)]
            .iter()
            .map(|&(address, value)| Fill {
                address,
                value,
                width: width.min(4),
                count: 1,
            });
        match view.device() {
            Device::Cuda(_) => {
                if stream.stream().is_none() {
                    return Err(Error::Invalid(
                        "a device product write requires its producer stream".into(),
                    ));
                }
                for fill in fills {
                    // SAFETY: one reserved element of `width` bytes on the
                    // stream's device.
                    unsafe { stream.fill(fill) }?;
                }
            }
            _ => {
                for fill in fills {
                    // SAFETY: one reserved host element of `width` bytes.
                    unsafe { stream.host().fill(fill) }?;
                }
            }
        }
        let fence = self.producer_fence(view.device(), stream, 1)?;
        lock(buffer).produced(vec![1], 1, fence);
        Ok(view)
    }

    /// Lease committed products for consumer calls. The whole batch is
    /// resolved before any lease is taken. `stream` (required for device
    /// products) waits on each producer fence once.
    pub fn consume_batch(
        &self,
        requests: &[(TensorRef, CallId, Option<Device>)],
        device: Option<Device>,
        stream: impl Issue,
    ) -> Result<Vec<Arc<ProductRead>>> {
        let state = lock(&self.state);
        let mut resolved = Vec::with_capacity(requests.len());
        let mut waited = HashSet::new();
        for (reference, consumer, requested) in requests {
            let buffer = Self::require_reference(&state, reference)?;
            let value = lock(&buffer);
            if value.released {
                return Err(Error::Invalid(
                    "device product was consumed after logical release".into(),
                ));
            }
            if let (Some(shared), Some(requested)) = (device, requested)
                && shared != *requested
            {
                return Err(Error::Invalid(
                    "device-product batch names conflicting consumer devices".into(),
                ));
            }
            if let Some(target) = device.or(*requested)
                && device_name(target) != value.device
            {
                return Err(Error::Invalid(
                    "device product consumer names a different device".into(),
                ));
            }
            Self::wait_producer(&value, stream, &mut waited)?;
            let shape: Vec<u64> = value
                .value_shape
                .iter()
                .map(|&extent| extent as u64)
                .collect();
            let extent = value.extent as u64;
            drop(value);
            let views = buffer.views();
            let read = Arc::new(ProductRead {
                value: value_view(&views.view, &shape, extent)?,
                region: views.region.clone(),
                metadata: views.metadata,
                lease: Mutex::new(TensorRead {
                    buffer: buffer.clone(),
                    consumer: Some(*consumer),
                    imported: None,
                    complete: false,
                }),
            });
            drop(views);
            resolved.push((buffer, read));
        }
        for (buffer, _) in &resolved {
            lock(buffer).readers += 1;
        }
        Ok(resolved.into_iter().map(|(_, read)| read).collect())
    }

    /// Lease one committed product.
    pub fn consume(
        &self,
        reference: &TensorRef,
        consumer: CallId,
        device: Option<Device>,
        stream: impl Issue,
    ) -> Result<Arc<ProductRead>> {
        let mut reads =
            self.consume_batch(&[(reference.clone(), consumer, device)], None, stream)?;
        reads
            .pop()
            .ok_or_else(|| Error::Invariant("a consumed product produced no lease".into()))
    }

    /// End leases after fencing their consuming `stream`. A consumer's own
    /// output fence on that stream (`after_writes`) serves as the reader
    /// fence; otherwise one is recorded.
    pub fn complete_reads(
        &self,
        reads: &[Arc<ProductRead>],
        stream: impl Issue,
        after_writes: &[ProductBuffer],
    ) -> Result<()> {
        self.events.reap()?;
        let mut state = lock(&self.state);
        let mut seen = HashSet::new();
        let pending: Vec<_> = reads
            .iter()
            .filter(|read| !read.complete() && seen.insert(Arc::as_ptr(read) as usize))
            .collect();
        if pending.is_empty() {
            return Ok(());
        }
        for read in &pending {
            let buffer = lock(&read.lease).buffer.clone();
            state.require_buffer(&buffer)?;
            if let (Some(consumer), Device::Cuda(device)) = (stream.stream(), read.value.device())
                && consumer.device().0 != device
            {
                return Err(Error::Invariant(
                    "device-product reader completed on a different device".into(),
                ));
            }
        }

        let mut write_fences = HashMap::new();
        for write in after_writes {
            state.require_buffer(write)?;
            let value = lock(write);
            if let Some(fence) = &value.producer {
                write_fences
                    .entry((
                        value.id.producer_call_id,
                        value.device.clone(),
                        fence.stream,
                    ))
                    .or_insert_with(|| Arc::clone(&fence.event));
            }
        }

        let mut recorded = HashMap::<i32, Arc<Event>>::new();
        for read in &pending {
            let Device::Cuda(device) = read.value.device() else {
                continue;
            };
            let raw = stream.stream().map(Stream::raw).ok_or_else(|| {
                Error::Invalid("device product reads complete on their consumer stream".into())
            })?;
            let (buffer, consumer) = {
                let lease = lock(&read.lease);
                (lease.buffer.clone(), lease.consumer)
            };
            let name = lock(&buffer).device.clone();
            let event = consumer
                .and_then(|consumer| write_fences.get(&(consumer, name, raw)))
                .cloned();
            let event = match event {
                Some(event) => event,
                None => match recorded.get(&device) {
                    Some(event) => Arc::clone(event),
                    None => {
                        let event = self.events.acquire(device, false, false)?;
                        self.events.record(&event, stream)?;
                        recorded.insert(device, Arc::clone(&event));
                        event
                    }
                },
            };
            if lock(&buffer).retain_reader(&event) {
                self.events.retain(&event, device, 1)?;
            }
        }

        let mut closing = Vec::new();
        let mut released = Vec::new();
        for read in pending {
            let (buffer, completed) = {
                let mut lease = lock(&read.lease);
                let buffer = lease.buffer.clone();
                (buffer, lease.complete())
            };
            if let Some(imported) = completed {
                state.imports.remove(&buffer_key(buffer.id()));
                let import = lock(&imported);
                closing.extend(
                    import
                        .tickets
                        .iter()
                        .map(|ticket| (Arc::clone(ticket.read()), !import.committed)),
                );
            }
            if lock(&buffer).released {
                released.push(buffer);
            }
        }
        drop(state);

        // Read observers may reenter the store; storage stays retained by
        // its native dependencies while they run.
        for (read, cancel) in closing {
            if cancel {
                read.cancel()?;
            }
            read.close()?;
        }
        for buffer in released {
            self.wake_retirement(&buffer, stream)?;
        }
        let mut state = lock(&self.state);
        self.reclaim(&mut state)
    }

    /// Materialize `tensor` (a product published on other ranks) on
    /// `device`: share a pending import of the same product, widen a
    /// committed shard by its missing regions, or reserve storage and read
    /// the whole tensor. Reads are planned against the whole destination and
    /// submitted as one reservation, ordered after `stream`. A failed
    /// submission retains any in-flight destination.
    #[expect(clippy::too_many_arguments, reason = "one import's complete inputs")]
    pub fn import_tensor(
        &self,
        reference: &TensorRef,
        tensor: &TensorTransfer,
        device: Device,
        transports: &Transports,
        placement: OutputPlacement<'_>,
        metadata: Option<Metadata>,
        stream: Option<&Stream>,
    ) -> Result<Arc<ProductRead>> {
        let id = reference.buffer_id();
        let key = buffer_key(id);
        let shape = tensor.shape.clone();
        let mut state = lock(&self.state);

        if let Some(pending) = state.imports.get(&key).cloned() {
            let mut value = lock(&pending);
            let buffer = value.buffer.clone();
            if buffer.views().reference != *reference
                || pending.0.destination.device() != device
                || pending.0.destination.shape() != signed(&shape).as_slice()
                || pending.0.metadata != metadata
            {
                return Err(Error::Invalid(
                    "product import conflicts with pending materialization".into(),
                ));
            }
            if lock(&buffer).released {
                return Err(Error::Invalid(
                    "product import requires a live published generation".into(),
                ));
            }
            value.share();
            drop(value);
            return Ok(import_read(&pending));
        }

        let existing = state
            .buffers
            .get(&key)
            .filter(|buffer| lock(buffer).state == WriteState::Committed)
            .cloned();
        let all = full(&shape);
        let (buffer, destination, missing) = match existing {
            Some(_) => {
                let buffer = Self::require_reference(&state, reference)?;
                let value = lock(&buffer);
                let views = buffer.views();
                if value.released {
                    return Err(Error::Invalid(
                        "product import requires a live published generation".into(),
                    ));
                }
                if device_name(device) != value.device || views.metadata != metadata {
                    return Err(Error::Invalid(
                        "product import conflicts with resident ownership".into(),
                    ));
                }
                let (destination, missing) = match &views.region {
                    None => {
                        if value
                            .value_shape
                            .iter()
                            .map(|&extent| extent as u64)
                            .ne(shape.iter().copied())
                        {
                            return Err(Error::Invalid(
                                "product import changes resident tensor shape".into(),
                            ));
                        }
                        (
                            value_view(&views.view, &shape, value.extent as u64)?,
                            Vec::new(),
                        )
                    }
                    Some(region) => {
                        if !matches!(value.backing, Some(Backing::Persistent(_)))
                            || views.storage.shape() != signed(&shape).as_slice()
                        {
                            return Err(Error::Invalid(
                                "product import exceeds its reserved logical storage".into(),
                            ));
                        }
                        if value.transfers.iter().any(|read| !read.succeeded()) {
                            return Err(Error::Resource(
                                "product storage has pending physical reads",
                            ));
                        }
                        let mut missing = Vec::new();
                        subtract_region(&all, region, &mut missing);
                        (views.storage.clone(), missing)
                    }
                };
                drop(views);
                drop(value);
                (buffer, destination, missing)
            }
            None => {
                let shapes = HashMap::from([(id, shape.clone())]);
                let placement = OutputPlacement {
                    shapes: Some(&shapes),
                    ..placement
                };
                let bound = [(reference.clone(), device)];
                let buffers = if matches!(metadata, Some(Metadata::Feature { .. })) {
                    let allocations = placement.allocations.ok_or_else(|| {
                        Error::Invalid("persistent output has no buffer allocation".into())
                    })?;
                    self.bind_persistent(&mut state, &bound, allocations, placement, true)?
                } else {
                    self.bind_group(&mut state, &bound, placement)?
                };
                let buffer = buffers.into_iter().next().ok_or_else(|| {
                    Error::Invariant("import destination was not reserved".into())
                })?;
                let destination = value_view(&buffer.view(), &shape, elements(&shape)?)?;
                (buffer, destination, vec![all])
            }
        };

        let source_dtype = tensor
            .locations
            .first()
            .ok_or_else(|| Error::Invalid("tensor transfer has no locations".into()))
            .and_then(|location| parse_dtype(&location.dtype))?;
        if destination.dtype() != source_dtype {
            if existing.is_none() {
                lock(&buffer).released = true;
                self.reclaim(&mut state)?;
            }
            return Err(Error::Invalid(
                "product import changes resident tensor dtype".into(),
            ));
        }

        lock(&buffer).readers += 1;
        let mut tickets = Vec::new();
        let submitted = (|| -> Result<()> {
            let mut locators = vec![None; tensor.locations.len()];
            let mut reads = Vec::new();
            for region in &missing {
                let target = Spans::single(destination.region(region)?)?;
                reads.extend(fetch::plan(
                    tensor,
                    &target,
                    transports,
                    region,
                    &mut locators,
                )?);
            }
            fetch::submit(&reads, stream, |read| {
                tickets.push(ReadRef::new(read));
                Ok(())
            })?;
            Ok(())
        })();
        let reads: Vec<_> = tickets
            .iter()
            .map(|ticket| Arc::clone(ticket.read()))
            .collect();
        lock(&buffer)
            .transfers
            .extend(tickets.iter().map(|ticket| ReadRef::new(ticket.read())));
        if let Err(error) = submitted {
            {
                let mut value = lock(&buffer);
                value.readers -= 1;
                if existing.is_none() {
                    value.released = true;
                }
            }
            drop(state);
            for read in reads {
                read.close()?;
            }
            let mut state = lock(&self.state);
            self.reclaim(&mut state)?;
            return Err(error);
        }

        let committed = existing.is_some() && buffer.region().is_none();
        let imported = ImportHandle(Arc::new(StoredImport {
            state: Mutex::new(TensorImport::new(buffer, tickets, committed)),
            destination,
            metadata,
        }));
        let read = import_read(&imported);
        state.imports.insert(key, imported);
        Ok(read)
    }

    fn imported(read: &ProductRead) -> Result<ImportHandle> {
        let lease = lock(&read.lease);
        if lease.complete {
            return Err(Error::Invariant(
                "closed or ordinary tensor read is not an import".into(),
            ));
        }
        lease.imported.clone().ok_or_else(|| {
            Error::Invariant("closed or ordinary tensor read is not an import".into())
        })
    }

    /// Order `stream` after an import's reads (which must be ready) and its
    /// buffer's producer, propagating read failures.
    pub fn wait_import(&self, read: &ProductRead, stream: impl Issue) -> Result<()> {
        let state = lock(&self.state);
        Self::wait_imported(&state, read, stream)
    }

    fn wait_imported(state: &State, read: &ProductRead, stream: impl Issue) -> Result<()> {
        let imported = Self::imported(read)?;
        let imported = lock(&imported);
        state.require_buffer(&imported.buffer)?;
        for ticket in &imported.tickets {
            ticket.read().result(stream)?;
        }
        let buffer = lock(&imported.buffer);
        Self::wait_producer(&buffer, stream, &mut HashSet::new())
    }

    /// Publish an import's materialized value on `stream`: a new product is
    /// produced and committed; a widened shard gets one fence over the
    /// union of its regions and becomes the whole logical tensor.
    pub fn complete_import(&self, read: &ProductRead, stream: impl Issue) -> Result<()> {
        self.events.reap()?;
        let mut state = lock(&self.state);
        Self::wait_imported(&state, read, stream)?;
        let imported = Self::imported(read)?;
        let (buffer, committed) = {
            let value = lock(&imported);
            (value.buffer.clone(), value.committed)
        };
        if committed {
            return Ok(());
        }
        let destination = imported.0.destination.clone();
        if !lock(&buffer).state.produced() {
            self.write_value(&buffer, &destination, stream, imported.0.metadata)?;
            state.commit_writes(std::slice::from_ref(&buffer))?;
        } else {
            // Existing leases retain their original shard views. A new fence
            // covers the union, including every newly fetched region.
            let fence = self.producer_fence(destination.device(), stream, 1)?;
            let previous = lock(&buffer)
                .producer
                .as_ref()
                .map(|fence| Arc::clone(&fence.event));
            if let Some(previous) = previous {
                let owner = buffer.clone();
                self.events.defer(
                    vec![previous],
                    Box::new(move || {
                        drop(owner);
                        Ok(())
                    }),
                )?;
            }
            {
                let mut views = buffer.views();
                views.view = destination.clone();
                views.region = None;
            }
            let mut value = lock(&buffer);
            value.value_shape = destination
                .shape()
                .iter()
                .map(|&extent| extent as usize)
                .collect();
            value.extent = destination.numel() as usize;
            value.producer = fence;
        }
        lock(&imported).committed = true;
        Ok(())
    }

    /// Release the outputs of finished calls.
    pub fn release_calls(&self, calls: &[(RequestKey, CallId)]) {
        lock(&self.state).release_calls(calls.iter().copied());
    }

    /// Revoke the exports of released buffers and release the buffers.
    pub fn release_buffers(&self, buffers: &HashSet<BufferId>) -> Result<()> {
        let revoked: Vec<_> = {
            let mut exports = lock(&self.exports);
            buffers
                .iter()
                .filter_map(|buffer| exports.remove(buffer))
                .flatten()
                .collect()
        };
        exports::release(&revoked)?;
        let released: Vec<_> = {
            let mut state = lock(&self.state);
            let keys = state.release_buffers(buffers);
            keys.iter()
                .filter_map(|key| state.buffers.get(key).cloned())
                .collect()
        };
        for buffer in released {
            self.retire_buffer(&buffer)?;
        }
        let mut state = lock(&self.state);
        self.reclaim(&mut state)
    }

    /// Release retired requests' products, except retained persistent ones,
    /// and revoke every export location registered for their buffers
    /// (products and published KV extents).
    pub fn release_requests(
        &self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()> {
        let revoked: Vec<_> = {
            let mut exports = lock(&self.exports);
            let buffers: Vec<BufferId> = exports
                .keys()
                .copied()
                .filter(|buffer| requests.contains(&buffer.owner) && !retained.contains(buffer))
                .collect();
            buffers
                .iter()
                .filter_map(|buffer| exports.remove(buffer))
                .flatten()
                .collect()
        };
        exports::release(&revoked)?;
        let released: Vec<_> = {
            let mut state = lock(&self.state);
            let keys = state.release_requests(requests, retained)?;
            keys.iter()
                .filter_map(|key| state.buffers.get(key).cloned())
                .collect()
        };
        for buffer in released {
            self.retire_buffer(&buffer)?;
        }
        let mut state = lock(&self.state);
        self.reclaim(&mut state)
    }

    /// Whether every selected buffer has retired. A read or export whose
    /// physical completion failed or is unknown is an error.
    pub fn retirement_ready(
        &self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        let selected = |id: &BufferId| {
            buffers.contains(id) || (requests.contains(&id.owner) && !retained.contains(id))
        };
        let mut state = lock(&self.state);
        for buffer in state.buffers.values() {
            let value = lock(buffer);
            if selected(&value.id) {
                for read in &value.transfers {
                    read.read().retirement_ready()?;
                }
                for export in &value.exports {
                    if let Some(crate::Outcome::Failed(error)) = export.outcome() {
                        return Err(Error::Transport(format!(
                            "product export retirement failed: {error}"
                        )));
                    }
                }
            }
        }
        self.reclaim(&mut state)?;
        Ok(!state
            .buffers
            .values()
            .any(|buffer| selected(&lock(buffer).id)))
    }

    /// Defer a write to a later call.
    pub fn defer_write(&self, write: &ProductBuffer) -> Result<()> {
        let state = lock(&self.state);
        state.require_buffer(write)?;
        lock(write).defer()
    }

    pub fn validate_writes(&self, writes: &[ProductBuffer]) -> Result<()> {
        lock(&self.state).validate_writes(writes)
    }

    /// Make produced writes discoverable; deferred writes stay pending.
    pub fn commit_writes(&self, writes: &[ProductBuffer]) -> Result<()> {
        lock(&self.state).commit_writes(writes)
    }

    /// Release the storage of writes whose call failed.
    pub fn abandon_writes(&self, writes: &[ProductBuffer]) -> Result<()> {
        let mut state = lock(&self.state);
        state.abandon_writes(writes);
        self.reclaim(&mut state)
    }

    /// Retain an export's physical retirement on its source buffer.
    pub fn retain_export(&self, write: &ProductBuffer, retirement: Arc<Retirement>) -> Result<()> {
        let state = lock(&self.state);
        state.require_buffer(write)?;
        lock(write).retain_export(retirement);
        Ok(())
    }

    /// Retain a read writing into a not-yet-produced buffer.
    pub fn retain_transfer(
        &self,
        write: &ProductBuffer,
        read: &Arc<crate::transfer::Read>,
    ) -> Result<()> {
        let state = lock(&self.state);
        state.require_buffer(write)?;
        lock(write).retain_transfer(ReadRef::new(read))
    }

    /// Check that `buffer` registers no export locations other than
    /// `locations`.
    pub fn validate_exports(&self, buffer: BufferId, locations: &[Location]) -> Result<()> {
        match lock(&self.exports).get(&buffer) {
            Some(existing) if !same_locations(existing, locations) => Err(Error::Invalid(
                "buffer already has different export locations".into(),
            )),
            _ => Ok(()),
        }
    }

    /// Record a committed product's export locations, revoked when the
    /// buffer is released. A buffer keeps one set of locations.
    pub fn register_exports(&self, buffer: BufferId, locations: Vec<Location>) -> Result<()> {
        let mut exports = lock(&self.exports);
        if let Some(existing) = exports.get(&buffer) {
            if !same_locations(existing, &locations) {
                return Err(Error::Invalid(
                    "buffer already has different export locations".into(),
                ));
            }
            return Ok(());
        }
        exports.insert(buffer, locations);
        Ok(())
    }

    /// Bytes of relay arenas allocated on `device`.
    pub fn resident_bytes(&self, device: Device) -> usize {
        let name = device_name(device);
        let state = lock(&self.state);
        let mut allocations = HashMap::new();
        for ((owner, _, _), arena) in &state.arenas {
            if *owner == name {
                allocations.insert(arena.memory().address(), arena.memory().len());
            }
        }
        allocations.values().sum()
    }

    /// Export a produced write through the transports reaching the call's
    /// consumers: its whole value, or each of `regions` (in value
    /// coordinates). Each export's retirement is retained on the buffer. On
    /// failure every location accepted by this call is revoked; on success
    /// the caller owns revocation of the returned locations until commit.
    #[expect(
        clippy::too_many_arguments,
        reason = "one product export's complete inputs"
    )]
    pub fn export_buffer(
        &self,
        call: &Call,
        product: &TensorRef,
        write: &ProductBuffer,
        transports: &Transports,
        host: bool,
        regions: Option<&[Vec<Range<u64>>]>,
        stream: Option<&Stream>,
    ) -> Result<(TensorExport, Vec<Location>)> {
        let (feature, logical_shape, value_shape, extent) = {
            let value = lock(write);
            (
                value.feature,
                value
                    .logical_shape
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect::<Vec<_>>(),
                value
                    .value_shape
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect::<Vec<_>>(),
                value.extent as u64,
            )
        };
        let (view, region, metadata) = {
            let views = write.views();
            (views.view.clone(), views.region.clone(), views.metadata)
        };
        let value = value_view(&view, &value_shape, extent)?;

        let (height, width) = metadata.map_or((0, 0), Metadata::dimensions);
        let value_range = match metadata {
            Some(Metadata::Image {
                value_range: Some(range),
                ..
            }) => range.name().to_owned(),
            _ => String::new(),
        };
        let source_kind = if feature {
            if call.code == CallKind::Media(MediaCall::VisionEncoding)
                || !call.vision_inputs.is_empty()
            {
                Some(FeatureKind::Vision)
            } else if call.code == CallKind::Media(MediaCall::LatentEncoding)
                || call.latent_feature_input.is_some()
            {
                Some(FeatureKind::Latent)
            } else {
                return Err(Error::Invalid(
                    "encoder transfer has no feature source".into(),
                ));
            }
        } else {
            if height == 0 && !value_range.is_empty() {
                return Err(Error::Invalid(
                    "non-image tensor carries an image range".into(),
                ));
            }
            None
        };

        // A shard exports its logical shape and the bound region's offset.
        let (shape, origin) = match &region {
            Some(region) => (
                logical_shape,
                region.iter().map(|range| range.start).collect(),
            ),
            None => (value_shape.clone(), vec![0; value_shape.len()]),
        };
        if !product.shape_bound.contains_shape(&shape) {
            return Err(Error::Invalid(
                "product transfer changes its declared representation".into(),
            ));
        }

        let consumers: Vec<usize> = call
            .consumer_slots
            .iter()
            .map(|&slot| slot as usize)
            .collect();
        let mut locations: Vec<Location> = Vec::new();
        let result = (|| -> Result<()> {
            let views: Vec<(StorageView, Vec<u64>)> = match regions {
                Some(regions) => regions
                    .iter()
                    .map(|requested| {
                        if !within(requested, &value_shape) {
                            return Err(Error::Invalid(
                                "consumer region exceeds the logical tensor".into(),
                            ));
                        }
                        let offset = origin
                            .iter()
                            .zip(requested)
                            .map(|(&base, range)| base + range.start)
                            .collect();
                        Ok((value.region(requested)?, offset))
                    })
                    .collect::<Result<_>>()?,
                None => vec![(value.clone(), origin.clone())],
            };
            for (view, offset) in views {
                let accepted = exports::export(
                    transports,
                    &Spans::single(view)?,
                    Some(&offset),
                    &consumers,
                    host,
                    stream,
                    |retirement| self.retain_export(write, retirement),
                )?;
                locations.extend(accepted);
            }
            Ok(())
        })();
        if let Err(error) = result {
            if let Err(cleanup) = exports::release(&locations) {
                tracing::warn!(target: "uniserve_worker.storage", "failed to revoke a failed product export: {cleanup}");
            }
            return Err(error);
        }

        let tensor = TensorTransfer {
            shape,
            locations: locations
                .iter()
                .map(|(_, locator)| locator.clone())
                .collect(),
        };
        let value = match source_kind {
            Some(payload_kind) => TransferHandle::Encoder {
                height,
                width,
                payload_kind,
                tensor,
            },
            None => TransferHandle::DeviceProduct {
                height,
                width,
                value_range,
                tensor,
            },
        };
        Ok((
            TensorExport {
                product: product.clone(),
                value,
            },
            locations,
        ))
    }

    /// Revoke every registered export location (shutdown); the transports
    /// retire them once their readers finished.
    pub fn revoke_exports(&self) -> Result<()> {
        let revoked: Vec<_> = lock(&self.exports)
            .drain()
            .flat_map(|(_, locations)| locations)
            .collect();
        exports::release(&revoked)
    }

    /// Forget every product after physical accesses drained. Existing reads
    /// retain their views.
    pub fn close(&self) {
        lock(&self.exports).clear();
        let retired = lock(&self.state).close();
        drop(retired);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "a buffer's complete initial views"
)]
fn create_buffer(
    reference: &TensorRef,
    device: Device,
    view: StorageView,
    storage: StorageView,
    logical_shape: Vec<u64>,
    region: Option<Vec<Range<u64>>>,
    feature: bool,
    backing: Backing,
) -> ProductBuffer {
    ProductBuffer(Arc::new(StoredBuffer {
        state: Mutex::new(Buffer::new(
            reference.buffer_id(),
            device_name(device),
            logical_shape
                .iter()
                .map(|&extent| extent as usize)
                .collect(),
            feature,
            backing,
        )),
        views: Mutex::new(BufferViews {
            reference: reference.clone(),
            view,
            region,
            metadata: None,
            storage,
        }),
    }))
}

fn same_locations(left: &[Location], right: &[Location]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.1 == right.1)
}

fn import_read(imported: &ImportHandle) -> Arc<ProductRead> {
    let buffer = lock(imported).buffer.clone();
    Arc::new(ProductRead {
        value: imported.0.destination.clone(),
        region: None,
        metadata: imported.0.metadata,
        lease: Mutex::new(TensorRead {
            buffer,
            consumer: None,
            imported: Some(imported.clone()),
            complete: false,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_bits_round_to_nearest_even() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.5), 0xc100);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1.0e6), 0x7c00);
        assert_eq!(f16_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits(-3.0), 0xc040);
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);
    }

    #[test]
    fn scalar_bits_follow_the_storage_dtype() -> Result<()> {
        assert_eq!(scalar_bits(Scalar::Int(-1), DType::I32)?, (0xffff_ffff, 4));
        assert_eq!(scalar_bits(Scalar::Float(2.9), DType::I64)?, (2, 8));
        assert_eq!(scalar_bits(Scalar::Int(1), DType::F32)?, (0x3f80_0000, 4));
        assert!(scalar_bits(Scalar::Int(1), DType::F8E4M3).is_err());
        Ok(())
    }
}
