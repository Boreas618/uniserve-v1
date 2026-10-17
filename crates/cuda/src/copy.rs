//! Stream-ordered copies and fills.
//!
//! Addresses are unified virtual addresses: device memory, pinned or
//! registered host memory, and (for [`SourceOrder::DuringCall`] sources)
//! ordinary host memory. Every submission is `unsafe` because the caller keeps
//! each addressed range valid, and not concurrently written by other work,
//! until the stream reaches the copy.
//!
//! # Mapped memory on SM-partition streams
//!
//! `cuMemcpyBatchAsync_v2` and `cuMemcpy3DBatchAsync_v2` fault inside the
//! driver (a segmentation fault, observed on driver 580.82.07) when the
//! stream belongs to a green context and an operand lies in memory mapped
//! with `cuMemMap`: an exportable allocation, an import or an alias. On the
//! same streams `cuMemcpyAsync` and `cuMemcpy3DAsync` serve those operands,
//! and the batched calls serve every other operand. [`CopyBatch::submit`] and
//! [`PitchedBatch::submit`] therefore send each entry with a mapped operand
//! in a call of its own when the stream runs in an SM partition, and all
//! other entries in one batched call.
//!
//! The unbatched calls read a page-locked host source or a device source in
//! stream order, and consume only an ordinary (pageable) host source before
//! they return. An entry of a [`SourceOrder::DuringCall`] batch that is sent
//! unbatched with a source of the first kind makes `submit` wait for the
//! stream before returning, so the caller may still reuse every source
//! immediately.

mod pitched;
mod strided;

use std::ffi::c_void;

use crate::driver::{
    CU_MEMORYTYPE_HOST, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CUDA_ERROR_INVALID_VALUE, MemLocation,
    MemcpyAttributes, driver,
};
use crate::{Error, Result, Stream, vmm};

pub use pitched::{Pitched, PitchedBatch};
pub use strided::{Strided, StridedCopy};

/// When a batched copy reads its sources (`CUmemcpySrcAccessOrder`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceOrder {
    /// Sources are read in stream order, after preceding work on the stream.
    Stream,
    /// Sources are read before the submitting call returns, so a host source
    /// may be reused immediately after `submit` (for small, ephemeral
    /// payloads such as block-table updates).
    DuringCall,
}

impl SourceOrder {
    fn raw(self) -> i32 {
        match self {
            Self::Stream => 1,
            Self::DuringCall => 2,
        }
    }
}

/// Rejects streams on which the batch APIs are not used: the legacy NULL
/// stream (not allowed by the API) and capturing streams (capture of batch
/// copies is undocumented, so storage copies never rely on it).
fn require_batch_stream(stream: &Stream, call: &'static str) -> Result<()> {
    if stream.raw() == 0 {
        return Err(Error::invalid(format!(
            "{call} does not accept the legacy NULL stream"
        )));
    }
    stream.require_eager(call)
}

/// Whether the batched calls cannot take a copy between `destination` and
/// `source` on a stream that runs in an SM partition (see the module
/// documentation).
fn mapped(destination: u64, source: u64) -> bool {
    vmm::is_mapped(destination) || vmm::is_mapped(source)
}

/// Where the driver places an operand address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Residence {
    Device,
    /// Page-locked or registered host memory.
    PageLocked,
    /// Host memory the driver does not track.
    Pageable,
}

impl Residence {
    /// Classifies `address` with the stream's context current.
    fn of(address: u64) -> Result<Self> {
        if vmm::is_mapped(address) {
            return Ok(Self::Device);
        }
        let driver = driver()?;
        let mut kind = 0_u32;
        // SAFETY: the driver writes one `CUmemorytype` for this attribute.
        let status = unsafe {
            (driver.pointer_get_attribute)(
                (&raw mut kind).cast::<c_void>(),
                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                address,
            )
        };
        // An address outside unified memory is ordinary host memory.
        if status == CUDA_ERROR_INVALID_VALUE {
            return Ok(Self::Pageable);
        }
        driver.check(status, "cuPointerGetAttribute")?;

        Ok(if kind == CU_MEMORYTYPE_HOST {
            Self::PageLocked
        } else {
            Self::Device
        })
    }

    /// Whether an unbatched copy reads this source before its call returns.
    /// Other sources are read in stream order.
    fn read_during_call(self) -> bool {
        self == Self::Pageable
    }
}

/// Copies `bytes` from `source` to `destination` after preceding work on
/// `stream` (`cuMemcpyAsync`; direction inferred from unified addresses).
///
/// # Safety
/// Both ranges stay valid until the copy completes, and a host range that is
/// not page-locked makes the call synchronous with respect to the host.
pub unsafe fn memcpy_async(
    destination: u64,
    source: u64,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    let _scope = stream.enter()?;
    let driver = driver()?;
    // SAFETY: forwarded from the caller.
    driver.check(
        unsafe { (driver.memcpy_async)(destination, source, bytes, stream.handle()) },
        "cuMemcpyAsync",
    )
}

macro_rules! memset {
    ($(#[$doc:meta])* $name:ident, $field:ident, $call:literal, $value:ty) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The destination holds `count` aligned values on the stream's device
        /// and stays valid until the fill completes.
        pub unsafe fn $name(
            destination: u64,
            value: $value,
            count: usize,
            stream: &Stream,
        ) -> Result<()> {
            let _scope = stream.enter()?;
            let driver = driver()?;
            driver.check(
                // SAFETY: forwarded from the caller, who keeps the destination valid
                // until the fill completes.
                unsafe { (driver.$field)(destination, value, count, stream.handle()) },
                $call,
            )
        }
    };
}

memset!(
    /// Fills `count` bytes with `value` (`cuMemsetD8Async`).
    memset_d8, memset_d8_async, "cuMemsetD8Async", u8
);
memset!(
    /// Fills `count` 16-bit words with `value` (`cuMemsetD16Async`).
    memset_d16, memset_d16_async, "cuMemsetD16Async", u16
);
memset!(
    /// Fills `count` 32-bit words with `value` (`cuMemsetD32Async`).
    memset_d32, memset_d32_async, "cuMemsetD32Async", u32
);

/// Fills `height` rows of `width` bytes, `pitch` bytes apart
/// (`cuMemsetD2D8Async`).
///
/// # Safety
/// Every row lies in device memory valid until the fill completes.
pub unsafe fn memset_2d_d8(
    destination: u64,
    pitch: usize,
    value: u8,
    width: usize,
    height: usize,
    stream: &Stream,
) -> Result<()> {
    let _scope = stream.enter()?;
    let driver = driver()?;
    // SAFETY: forwarded from the caller.
    driver.check(
        unsafe {
            (driver.memset_d2d8_async)(destination, pitch, value, width, height, stream.handle())
        },
        "cuMemsetD2D8Async",
    )
}

/// Fills `height` rows of `width` 32-bit words, `pitch` bytes apart
/// (`cuMemsetD2D32Async`).
///
/// # Safety
/// Every row lies in 4-byte aligned device memory valid until the fill
/// completes.
pub unsafe fn memset_2d_d32(
    destination: u64,
    pitch: usize,
    value: u32,
    width: usize,
    height: usize,
    stream: &Stream,
) -> Result<()> {
    let _scope = stream.enter()?;
    let driver = driver()?;
    // SAFETY: forwarded from the caller.
    driver.check(
        unsafe {
            (driver.memset_d2d32_async)(destination, pitch, value, width, height, stream.handle())
        },
        "cuMemsetD2D32Async",
    )
}

/// Contiguous copies submitted with one `cuMemcpyBatchAsync` call, except
/// entries with a mapped operand on an SM-partition stream, which are
/// submitted one `cuMemcpyAsync` call each (see the module documentation).
///
/// Copies in one batch are unordered with respect to each other: a batch must
/// never contain overlapping destinations, or a destination that another
/// entry reads. The batch keeps its entries after submission, so a fixed
/// batch can be resubmitted.
#[derive(Clone, Debug)]
pub struct CopyBatch {
    destinations: Vec<u64>,
    sources: Vec<u64>,
    sizes: Vec<usize>,
    order: SourceOrder,
}

impl CopyBatch {
    pub fn new(order: SourceOrder) -> Self {
        Self {
            destinations: Vec::new(),
            sources: Vec::new(),
            sizes: Vec::new(),
            order,
        }
    }

    /// Adds one copy. Empty copies are ignored (the API rejects zero sizes).
    pub fn push(&mut self, destination: u64, source: u64, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.destinations.push(destination);
        self.sources.push(source);
        self.sizes.push(bytes);
    }

    /// When the batch's sources are read.
    pub fn order(&self) -> SourceOrder {
        self.order
    }

    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    pub fn clear(&mut self) {
        self.destinations.clear();
        self.sources.clear();
        self.sizes.clear();
    }

    /// The `(destination, source, bytes)` entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, u64, usize)> + '_ {
        self.destinations
            .iter()
            .zip(&self.sources)
            .zip(&self.sizes)
            .map(|((destination, source), bytes)| (*destination, *source, *bytes))
    }

    /// Submits every entry on `stream`; an empty batch submits nothing.
    ///
    /// With [`SourceOrder::DuringCall`] every source has been read when this
    /// returns. For an entry submitted unbatched whose source is page-locked
    /// host memory or device memory, that requires waiting for the stream.
    ///
    /// Fails with [`Error::Capturing`] on a capturing stream and with
    /// [`Error::Symbol`] on a driver older than CUDA 12.8.
    ///
    /// # Safety
    /// Every range stays valid until the stream reaches the copies; with
    /// [`SourceOrder::Stream`] that includes the sources.
    pub unsafe fn submit(&self, stream: &Stream) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        require_batch_stream(stream, BATCH_CALL)?;
        let _scope = stream.enter()?;

        let operands = || self.destinations.iter().zip(&self.sources);
        if !stream.is_partitioned()? || !operands().any(|(&to, &from)| mapped(to, from)) {
            // SAFETY: forwarded from the caller.
            return unsafe {
                submit_batch(
                    &self.destinations,
                    &self.sources,
                    &self.sizes,
                    self.order,
                    stream,
                )
            };
        }

        let driver = driver()?;
        let mut batched = Self::new(self.order);
        let mut wait = false;
        for (destination, source, bytes) in self.iter() {
            if !mapped(destination, source) {
                batched.push(destination, source, bytes);
                continue;
            }
            driver.check(
                // SAFETY: forwarded from the caller, who keeps both ranges valid until
                // the stream reaches the copy.
                unsafe { (driver.memcpy_async)(destination, source, bytes, stream.handle()) },
                "cuMemcpyAsync",
            )?;
            if self.order == SourceOrder::DuringCall {
                wait |= !Residence::of(source)?.read_during_call();
            }
        }
        if !batched.is_empty() {
            // SAFETY: forwarded from the caller.
            unsafe {
                submit_batch(
                    &batched.destinations,
                    &batched.sources,
                    &batched.sizes,
                    self.order,
                    stream,
                )
            }?;
        }
        if wait {
            stream.synchronize()?;
        }
        Ok(())
    }
}

const BATCH_CALL: &str = "cuMemcpyBatchAsync_v2";

/// Submits parallel arrays of contiguous copies with one batched call.
///
/// # Safety
/// As for [`CopyBatch::submit`]; the stream's context is current and no
/// entry needs an unbatched call.
unsafe fn submit_batch(
    destinations: &[u64],
    sources: &[u64],
    sizes: &[usize],
    order: SourceOrder,
    stream: &Stream,
) -> Result<()> {
    let driver = driver()?;
    let copy = driver.memcpy_batch_async.clone()?;

    // One attribute applies to every entry (attrsIdxs[0] must be 0).
    let mut attributes = MemcpyAttributes {
        source_order: order.raw(),
        source_hint: MemLocation::default(),
        destination_hint: MemLocation::default(),
        flags: 0,
    };
    let mut first = 0_usize;

    // SAFETY: the driver reads the host arrays during the call and neither
    // writes nor retains them; the caller guarantees the device ranges.
    driver.check(
        unsafe {
            copy(
                destinations.as_ptr().cast_mut(),
                sources.as_ptr().cast_mut(),
                sizes.as_ptr().cast_mut(),
                sizes.len(),
                &mut attributes,
                &mut first,
                1,
                stream.handle(),
            )
        },
        BATCH_CALL,
    )
}
