//! A product's buffer: its views, metadata and shape.

use super::*;
use crate::ffi::{DType, MemoryKind};
use crate::storage::memory::StorageView;
use crate::transfer::{ReadRef, Retirement};
use crate::{BufferBinding, Error, Result};
use std::collections::HashMap;
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex, MutexGuard};
use uniserve_cuda::Event;
use uniserve_ipc::{BufferId, DimBound, TensorRef};

/// The numerical interval of an image product's pixel values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueRange {
    /// `[0, 1]`.
    Unit,
    /// `[-1, 1]`.
    SignedUnit,
}

impl ValueRange {
    pub(super) fn name(self) -> &'static str {
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
    pub(super) fn validate(self) -> Result<()> {
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

    pub(super) fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Image { height, width, .. } | Self::Feature { height, width } => (height, width),
        }
    }
}

pub(super) struct BufferViews {
    pub(super) reference: TensorRef,
    /// The bound view: the logical tensor, or this rank's region of it.
    pub(super) view: StorageView,
    /// This rank's region of a sharded product, in logical coordinates.
    pub(super) region: Option<Vec<Range<u64>>>,
    pub(super) metadata: Option<Metadata>,
    /// The whole reserved storage (the logical shape when the allocation
    /// covers it), which imports widen into.
    pub(super) storage: StorageView,
}

pub(super) struct StoredBuffer {
    pub(super) state: Mutex<Buffer>,
    pub(super) views: Mutex<BufferViews>,
}

/// A handle to one stored product. Read leases and imports retain the same
/// buffer through logical release and storage reuse.
#[derive(Clone)]
pub struct ProductBuffer(pub(super) Arc<StoredBuffer>);

impl Deref for ProductBuffer {
    type Target = Mutex<Buffer>;

    fn deref(&self) -> &Self::Target {
        &self.0.state
    }
}

impl ProductBuffer {
    pub(crate) fn id(&self) -> BufferId {
        lock(self).id
    }

    /// The bound view producers write.
    pub fn view(&self) -> StorageView {
        lock(&self.0.views).view.clone()
    }

    pub(crate) fn region(&self) -> Option<Vec<Range<u64>>> {
        lock(&self.0.views).region.clone()
    }

    pub(crate) fn metadata(&self) -> Option<Metadata> {
        lock(&self.0.views).metadata
    }

    pub(super) fn views(&self) -> MutexGuard<'_, BufferViews> {
        lock(&self.0.views)
    }
}

/// The element type of a protocol product dtype.
pub(super) fn product_dtype(dtype: uniserve_ipc::DType) -> Result<DType> {
    Ok(DType::from(dtype))
}

/// Static extents and the device-actual axis at its bound; a scalar product
/// is stored as shape `[1]`.
pub(super) fn device_shape(reference: &TensorRef) -> Vec<u64> {
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

pub(super) fn elements(shape: &[u64]) -> Result<u64> {
    shape
        .iter()
        .try_fold(1_u64, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| Error::Invalid("tensor extent overflows".into()))
}

pub(super) fn shape_bytes(shape: &[u64], dtype: DType) -> Result<u64> {
    elements(shape)?
        .checked_mul(dtype.bytes() as u64)
        .ok_or_else(|| Error::Invalid("buffer tensor shape has an invalid byte extent".into()))
}

pub(super) fn full(shape: &[u64]) -> Vec<Range<u64>> {
    shape.iter().map(|&extent| 0..extent).collect()
}

pub(super) fn region_shape(region: &[Range<u64>]) -> Vec<u64> {
    region.iter().map(|range| range.end - range.start).collect()
}

pub(super) fn within(region: &[Range<u64>], shape: &[u64]) -> bool {
    region.len() == shape.len()
        && region
            .iter()
            .zip(shape)
            .all(|(range, &extent)| range.start <= range.end && range.end <= extent)
}

pub(super) fn signed(shape: &[u64]) -> Vec<i64> {
    shape.iter().map(|&extent| extent as i64).collect()
}

/// The published `extent` elements of `view` as `shape`, preserving the
/// strides of an exact shard.
pub(super) fn value_view(view: &StorageView, shape: &[u64], extent: u64) -> Result<StorageView> {
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

#[expect(
    clippy::too_many_arguments,
    reason = "a buffer's complete initial views"
)]
pub(super) fn create_buffer(
    reference: &TensorRef,
    device: MemoryKind,
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
            device,
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

pub(crate) fn buffer_key(id: BufferId) -> BufferKey {
    (id.owner, id.producer_call_id, id.output_index)
}

pub(crate) enum Backing {
    Persistent(Arc<BufferBinding>),
    Relay(RelayKey),
}

/// A value's event and the stream on which its producer will record it.
#[derive(Clone)]
pub(crate) struct Fence {
    pub event: Arc<Event>,
    pub stream: usize,
}

/// One allocation-backed value. The owning store protects mutations, while
/// read handles retain the same buffer through logical release and reuse.
/// The reads writing it and the exports reading it retain it until their
/// physical retirement.
pub struct Buffer {
    pub(crate) id: BufferId,
    pub(crate) device: MemoryKind,
    pub(crate) logical_shape: Vec<usize>,
    pub(crate) feature: bool,
    pub(crate) backing: Option<Backing>,
    pub(crate) state: WriteState,
    pub(crate) extent: usize,
    pub(crate) value_shape: Vec<usize>,
    pub(crate) producer: Option<Fence>,
    pub(crate) readers: usize,
    pub(crate) reader_events: Vec<Arc<Event>>,
    pub(crate) transfers: Vec<ReadRef>,
    pub(crate) exports: Vec<Arc<Retirement>>,
    pub(crate) released: bool,
}

impl Buffer {
    pub(crate) fn new(
        id: BufferId,
        device: MemoryKind,
        logical_shape: Vec<usize>,
        feature: bool,
        backing: Backing,
    ) -> Self {
        Self {
            id,
            device,
            logical_shape,
            feature,
            backing: Some(backing),
            state: WriteState::Reserved,
            extent: 0,
            value_shape: Vec::new(),
            producer: None,
            readers: 0,
            reader_events: Vec::new(),
            transfers: Vec::new(),
            exports: Vec::new(),
            released: false,
        }
    }

    pub(crate) fn require_candidate(&self) -> Result<()> {
        if self.backing.is_none() || self.state == WriteState::Committed {
            return Err(Error::Invariant(
                "device-product candidate is not live".into(),
            ));
        }
        if self.state == WriteState::Reserved {
            return Err(Error::Invariant(
                "a write is validated before its producer published it".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn defer(&mut self) -> Result<()> {
        if self.state.produced() {
            return Err(Error::Invariant(
                "a published write cannot be deferred".into(),
            ));
        }
        self.state = WriteState::Deferred;
        Ok(())
    }

    pub(crate) fn produced(&mut self, shape: Vec<usize>, extent: usize, fence: Option<Fence>) {
        self.value_shape = shape;
        self.extent = extent;
        self.producer = fence;
        self.state = WriteState::Produced;
    }

    pub(crate) fn retain_reader(&mut self, event: &Arc<Event>) -> bool {
        if self
            .reader_events
            .iter()
            .any(|current| Arc::ptr_eq(current, event))
        {
            return false;
        }
        self.reader_events.push(Arc::clone(event));
        true
    }

    pub(crate) fn events(&self) -> impl Iterator<Item = &Arc<Event>> {
        self.producer
            .iter()
            .map(|fence| &fence.event)
            .chain(&self.reader_events)
    }

    /// Cancellation is not physical completion. A failed transfer or export
    /// retains its backing until the worker's failure shutdown drains access.
    pub(crate) fn access_retired(&self, queried: &mut HashMap<usize, bool>) -> Result<bool> {
        if !self.released
            || self.readers != 0
            || self.transfers.iter().any(|transfer| !transfer.succeeded())
            || self.exports.iter().any(|export| !export.succeeded())
        {
            return Ok(false);
        }

        for event in self.events() {
            let ready = match queried.get(&(Arc::as_ptr(event) as usize)) {
                Some(&ready) => ready,
                None => {
                    let ready = event.query()?;
                    queried.insert(Arc::as_ptr(event) as usize, ready);
                    ready
                }
            };
            if !ready {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(crate) fn retain_export(&mut self, retirement: Arc<Retirement>) {
        self.exports.retain(|export| !export.succeeded());
        self.exports.push(retirement);
    }
}
