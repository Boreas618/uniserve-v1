//! Stream-ordered copies and fills.
//!
//! Addresses are unified virtual addresses: device memory, pinned or
//! registered host memory, and (for [`SourceOrder::DuringCall`] sources)
//! ordinary host memory. Every submission is `unsafe` because the caller keeps
//! each addressed range valid, and not concurrently written by other work,
//! until the stream reaches the copy.

use crate::driver::{
    CU_MEMCPY_OPERAND_TYPE_POINTER, Handle, MemLocation, Memcpy3DBatchOp, Memcpy3DOperand,
    MemcpyAttributes, driver,
};
use crate::{DeviceAttribute, DeviceId, Error, Result, Stream};

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
            // SAFETY: forwarded from the caller.
            driver.check(
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

/// Contiguous copies submitted with one `cuMemcpyBatchAsync` call.
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
        let call = "cuMemcpyBatchAsync_v2";
        let driver = driver()?;
        let copy = driver.memcpy_batch_async.clone()?;
        require_batch_stream(stream, call)?;
        let _scope = stream.enter()?;

        // One attribute applies to every entry (attrsIdxs[0] must be 0).
        let mut attributes = MemcpyAttributes {
            source_order: self.order.raw(),
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
                    self.destinations.as_ptr().cast_mut(),
                    self.sources.as_ptr().cast_mut(),
                    self.sizes.as_ptr().cast_mut(),
                    self.sizes.len(),
                    &mut attributes,
                    &mut first,
                    1,
                    stream.handle(),
                )
            },
            call,
        )
    }
}

/// One pitched copy of `depth` layers of `height` rows of `width` bytes.
///
/// Row `y` of layer `z` starts at `base + (z * layer + y) * pitch` in each
/// operand: `pitch` is the byte distance between rows and `layer` the number
/// of rows between layer starts. With `height == 1` the pitch is unused, and
/// with `depth == 1` the layer is unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pitched {
    pub destination: u64,
    pub destination_pitch: usize,
    pub destination_layer: usize,
    pub source: u64,
    pub source_pitch: usize,
    pub source_layer: usize,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl Pitched {
    fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Err(Error::invalid("pitched copy extents must be positive"));
        }
        if self.height > 1
            && (self.destination_pitch < self.width || self.source_pitch < self.width)
        {
            return Err(Error::invalid(format!(
                "pitched copy rows of {} bytes overlap with pitches {} and {}",
                self.width, self.destination_pitch, self.source_pitch
            )));
        }
        if self.depth > 1
            && (self.destination_layer < self.height || self.source_layer < self.height)
        {
            return Err(Error::invalid(format!(
                "pitched copy layers of {} rows overlap with layer heights {} and {}",
                self.height, self.destination_layer, self.source_layer
            )));
        }
        Ok(())
    }

    fn operand(&self, pointer: u64, pitch: usize, layer: usize) -> Memcpy3DOperand {
        // For pointer operands the driver measures rows in bytes (the element
        // of an untyped pointer is one byte) and layers in rows.
        Memcpy3DOperand {
            kind: CU_MEMCPY_OPERAND_TYPE_POINTER,
            pointer,
            row_length: if self.height > 1 { pitch } else { self.width },
            layer_height: if self.depth > 1 { layer } else { self.height },
            hint: MemLocation::default(),
        }
    }
}

/// Pitched copies submitted with one `cuMemcpy3DBatchAsync` call. The same
/// ordering rules as [`CopyBatch`] apply.
#[derive(Clone, Debug)]
pub struct PitchedBatch {
    copies: Vec<Pitched>,
    order: SourceOrder,
}

impl PitchedBatch {
    pub fn new(order: SourceOrder) -> Self {
        Self {
            copies: Vec::new(),
            order,
        }
    }

    /// Adds one copy after checking that its rows and layers do not overlap.
    pub fn push(&mut self, copy: Pitched) -> Result<()> {
        copy.validate()?;
        self.copies.push(copy);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.copies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    pub fn clear(&mut self) {
        self.copies.clear();
    }

    pub fn copies(&self) -> &[Pitched] {
        &self.copies
    }

    /// Submits every copy on `stream`; an empty batch submits nothing.
    /// Fails like [`CopyBatch::submit`].
    ///
    /// # Safety
    /// Every addressed row stays valid until the stream reaches the copies.
    pub unsafe fn submit(&self, stream: &Stream) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let call = "cuMemcpy3DBatchAsync_v2";
        let driver = driver()?;
        let copy = driver.memcpy_3d_batch_async.clone()?;
        require_batch_stream(stream, call)?;
        let _scope = stream.enter()?;

        let mut operations: Vec<Memcpy3DBatchOp> = self
            .copies
            .iter()
            .map(|copy| Memcpy3DBatchOp {
                source: copy.operand(copy.source, copy.source_pitch, copy.source_layer),
                destination: copy.operand(
                    copy.destination,
                    copy.destination_pitch,
                    copy.destination_layer,
                ),
                extent: [copy.width, copy.height, copy.depth],
                source_order: self.order.raw(),
                flags: 0,
            })
            .collect();

        // SAFETY: the driver reads the operation array during the call only;
        // the caller guarantees the addressed rows.
        driver.check(
            unsafe {
                copy(
                    operations.len(),
                    operations.as_mut_ptr(),
                    0,
                    stream.handle() as Handle,
                )
            },
            call,
        )
    }
}

/// A strided copy operand: `shape[i]` elements along axis `i`, `strides[i]`
/// elements apart (row-major or any other layout; strides are non-negative).
#[derive(Clone, Copy, Debug)]
pub struct Strided<'a> {
    pub address: u64,
    pub element_bytes: usize,
    pub shape: &'a [u64],
    pub strides: &'a [i64],
}

/// One axis after planning: element count and byte strides in both operands.
#[derive(Clone, Copy, Debug)]
struct Axis {
    extent: u64,
    destination: u64,
    source: u64,
}

/// Plans exact copies between two equally shaped strided operands and
/// submits them as at most one [`CopyBatch`] and one [`PitchedBatch`].
///
/// The planner (generalizing a strided host/device DMA to any direction):
///
/// 1. drops unit axes and merges adjacent axes contiguous in both operands;
/// 2. grows the contiguous run (`width` bytes) by absorbing every axis whose
///    stride equals the run in both operands;
/// 3. selects as `height` the longest remaining axis whose byte strides are
///    at least `width` and at most the device's maximum pitch in both;
/// 4. selects as `depth` the longest remaining axis whose byte strides are a
///    whole number of rows, at least `height`, in both operands;
/// 5. enumerates every remaining axis, one copy per coordinate.
///
/// Copies with one row and one layer go to the contiguous batch, the others to
/// the pitched batch. The caller ensures the destination does not overlap
/// itself or the source.
#[derive(Clone, Debug)]
pub struct StridedCopy {
    linear: CopyBatch,
    pitched: PitchedBatch,
    max_pitch: u64,
}

impl StridedCopy {
    /// A planner for pitches up to `max_pitch` bytes.
    pub fn new(order: SourceOrder, max_pitch: usize) -> Self {
        Self {
            linear: CopyBatch::new(order),
            pitched: PitchedBatch::new(order),
            max_pitch: max_pitch as u64,
        }
    }

    /// A planner using `device`'s [`DeviceAttribute::MaxPitch`].
    pub fn for_device(device: DeviceId, order: SourceOrder) -> Result<Self> {
        let max_pitch = device.attribute(DeviceAttribute::MaxPitch)?;
        let max_pitch = usize::try_from(max_pitch)
            .map_err(|_| Error::invalid(format!("{device} reports a negative maximum pitch")))?;
        Ok(Self::new(order, max_pitch))
    }

    pub fn linear(&self) -> &CopyBatch {
        &self.linear
    }

    pub fn pitched(&self) -> &PitchedBatch {
        &self.pitched
    }

    pub fn is_empty(&self) -> bool {
        self.linear.is_empty() && self.pitched.is_empty()
    }

    pub fn clear(&mut self) {
        self.linear.clear();
        self.pitched.clear();
    }

    /// Appends the copies of `source` into `destination`. On error nothing is
    /// appended.
    pub fn plan(&mut self, destination: &Strided<'_>, source: &Strided<'_>) -> Result<()> {
        let mut axes = axes(destination, source)?;
        if axes.iter().any(|axis| axis.extent == 0) {
            return Ok(());
        }
        axes.retain(|axis| axis.extent != 1);
        let mut axes = coalesce(axes);

        // Contiguous run in both operands.
        let mut width = destination.element_bytes as u64;
        while let Some(index) = axes
            .iter()
            .position(|axis| axis.destination == width && axis.source == width)
        {
            width = mul(width, axes.remove(index).extent)?;
        }

        let rows = longest(&axes, |axis| {
            axis.destination >= width
                && axis.source >= width
                && axis.destination <= self.max_pitch
                && axis.source <= self.max_pitch
        })
        .map(|index| axes.remove(index));

        let layers = rows.and_then(|rows| {
            longest(&axes, |axis| {
                whole_rows(axis.destination, rows.destination, rows.extent)
                    && whole_rows(axis.source, rows.source, rows.extent)
            })
            .map(|index| axes.remove(index))
        });

        // Build every copy before appending any, so a failure appends nothing.
        let mut planned = Vec::new();
        let mut coordinates = vec![0_u64; axes.len()];
        loop {
            let mut destination_offset = 0_u64;
            let mut source_offset = 0_u64;
            for (axis, coordinate) in axes.iter().zip(&coordinates) {
                destination_offset = add(destination_offset, mul(*coordinate, axis.destination)?)?;
                source_offset = add(source_offset, mul(*coordinate, axis.source)?)?;
            }

            let (height, destination_pitch, source_pitch) = match rows {
                Some(rows) => (rows.extent, rows.destination, rows.source),
                None => (1, width, width),
            };
            let (depth, destination_layer, source_layer) = match layers {
                Some(layers) => (
                    layers.extent,
                    layers.destination / destination_pitch,
                    layers.source / source_pitch,
                ),
                None => (1, height, height),
            };
            planned.push(Pitched {
                destination: add(destination.address, destination_offset)?,
                destination_pitch: size(destination_pitch)?,
                destination_layer: size(destination_layer)?,
                source: add(source.address, source_offset)?,
                source_pitch: size(source_pitch)?,
                source_layer: size(source_layer)?,
                width: size(width)?,
                height: size(height)?,
                depth: size(depth)?,
            });

            // Odometer over the enumerated axes, first axis fastest: advance
            // the first axis that does not wrap; stop when every axis wraps.
            let advanced = coordinates.iter_mut().zip(&axes).any(|(coordinate, axis)| {
                *coordinate += 1;
                if *coordinate < axis.extent {
                    return true;
                }
                *coordinate = 0;
                false
            });
            if !advanced {
                break;
            }
        }

        for copy in planned {
            if copy.height == 1 && copy.depth == 1 {
                self.linear.push(copy.destination, copy.source, copy.width);
            } else {
                self.pitched.push(copy)?;
            }
        }
        Ok(())
    }

    /// Submits the planned copies on `stream`. Fails like
    /// [`CopyBatch::submit`].
    ///
    /// # Safety
    /// Both operands stay valid until the stream reaches the copies.
    pub unsafe fn submit(&self, stream: &Stream) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe {
            self.linear.submit(stream)?;
            self.pitched.submit(stream)
        }
    }
}

/// Validates the operands and converts strides to bytes.
fn axes(destination: &Strided<'_>, source: &Strided<'_>) -> Result<Vec<Axis>> {
    if destination.shape != source.shape {
        return Err(Error::invalid(format!(
            "strided copy shapes differ: {:?} and {:?}",
            destination.shape, source.shape
        )));
    }
    if destination.element_bytes != source.element_bytes || destination.element_bytes == 0 {
        return Err(Error::invalid(
            "strided copy operands need one positive element size",
        ));
    }
    if destination.strides.len() != destination.shape.len()
        || source.strides.len() != source.shape.len()
    {
        return Err(Error::invalid("each axis needs one stride per operand"));
    }

    let element = destination.element_bytes as u64;
    let stride = |value: i64| -> Result<u64> {
        let value = u64::try_from(value)
            .map_err(|_| Error::invalid("strided copies need non-negative strides"))?;
        mul(value, element)
    };

    destination
        .shape
        .iter()
        .zip(destination.strides.iter().zip(source.strides))
        .map(|(&extent, (&destination, &source))| {
            let axis = Axis {
                extent,
                destination: stride(destination)?,
                source: stride(source)?,
            };
            if axis.extent > 1 && axis.destination == 0 {
                return Err(Error::invalid(
                    "a destination axis with zero stride would write one element repeatedly",
                ));
            }
            Ok(axis)
        })
        .collect()
}

/// Merges each outer axis into the adjacent inner axis when both operands
/// address them as one longer axis.
fn coalesce(axes: Vec<Axis>) -> Vec<Axis> {
    let mut merged: Vec<Axis> = Vec::with_capacity(axes.len());
    for inner in axes {
        if let Some(outer) = merged.last_mut()
            && Some(outer.destination) == inner.destination.checked_mul(inner.extent)
            && Some(outer.source) == inner.source.checked_mul(inner.extent)
            && let Some(extent) = outer.extent.checked_mul(inner.extent)
        {
            *outer = Axis { extent, ..inner };
            continue;
        }
        merged.push(inner);
    }
    merged
}

/// The index of the longest axis satisfying `eligible`.
fn longest(axes: &[Axis], eligible: impl Fn(&Axis) -> bool) -> Option<usize> {
    axes.iter()
        .enumerate()
        .filter(|(_, axis)| eligible(axis))
        .max_by_key(|(_, axis)| axis.extent)
        .map(|(index, _)| index)
}

/// Whether `stride` is a whole number of `pitch`-byte rows, at least `height`.
fn whole_rows(stride: u64, pitch: u64, height: u64) -> bool {
    pitch > 0 && stride.is_multiple_of(pitch) && stride / pitch >= height
}

fn mul(left: u64, right: u64) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| Error::invalid("strided copy offsets overflow"))
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| Error::invalid("strided copy offsets overflow"))
}

fn size(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid("strided copy extent exceeds usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host model of the planned copies over byte arrays based at `BASE`.
    const BASE: u64 = 0x10_0000;

    fn execute(plan: &StridedCopy, destination: &mut [u8], source: &[u8]) {
        let offset = |address: u64| (address - BASE) as usize;
        for (to, from, bytes) in plan.linear().iter() {
            destination[offset(to)..offset(to) + bytes]
                .copy_from_slice(&source[offset(from)..offset(from) + bytes]);
        }
        for copy in plan.pitched().copies() {
            for z in 0..copy.depth {
                for y in 0..copy.height {
                    let to = offset(copy.destination)
                        + (z * copy.destination_layer + y) * copy.destination_pitch;
                    let from =
                        offset(copy.source) + (z * copy.source_layer + y) * copy.source_pitch;
                    destination[to..to + copy.width]
                        .copy_from_slice(&source[from..from + copy.width]);
                }
            }
        }
    }

    /// Element-by-element reference copy.
    fn reference(
        shape: &[u64],
        element: usize,
        destination_strides: &[i64],
        source_strides: &[i64],
        destination: &mut [u8],
        source: &[u8],
    ) {
        let count: u64 = shape.iter().product();
        for linear in 0..count {
            let (mut rest, mut to, mut from) = (linear, 0, 0);
            for axis in (0..shape.len()).rev() {
                let coordinate = (rest % shape[axis]) as usize;
                rest /= shape[axis];
                to += coordinate * destination_strides[axis] as usize * element;
                from += coordinate * source_strides[axis] as usize * element;
            }
            destination[to..to + element].copy_from_slice(&source[from..from + element]);
        }
    }

    fn extent(shape: &[u64], strides: &[i64], element: usize) -> usize {
        if shape.contains(&0) {
            return element;
        }
        shape
            .iter()
            .zip(strides)
            .map(|(extent, stride)| (*extent as usize - 1) * *stride as usize * element)
            .sum::<usize>()
            + element
    }

    /// Plans, executes on the host model, and compares with the reference.
    fn check(
        shape: &[u64],
        element: usize,
        destination_strides: &[i64],
        source_strides: &[i64],
        max_pitch: usize,
    ) -> StridedCopy {
        let source: Vec<u8> = (0..extent(shape, source_strides, element))
            .map(|index| (index * 7 + 3) as u8)
            .collect();
        let destination_bytes = extent(shape, destination_strides, element);
        let mut planned = vec![0_u8; destination_bytes];
        let mut expected = vec![0_u8; destination_bytes];

        let mut plan = StridedCopy::new(SourceOrder::Stream, max_pitch);
        plan.plan(
            &Strided {
                address: BASE,
                element_bytes: element,
                shape,
                strides: destination_strides,
            },
            &Strided {
                address: BASE,
                element_bytes: element,
                shape,
                strides: source_strides,
            },
        )
        .unwrap();

        execute(&plan, &mut planned, &source);
        reference(
            shape,
            element,
            destination_strides,
            source_strides,
            &mut expected,
            &source,
        );
        assert_eq!(planned, expected);
        plan
    }

    #[test]
    fn contiguous_operands_copy_as_one_run() {
        let plan = check(&[4, 3, 8], 2, &[24, 8, 1], &[24, 8, 1], 1 << 20);

        assert_eq!(
            plan.linear().iter().collect::<Vec<_>>(),
            [(BASE, BASE, 192)]
        );
        assert!(plan.pitched().is_empty());
    }

    #[test]
    fn row_slices_copy_as_one_pitched_copy() {
        // Rows of 8 elements out of a 16-wide source into a dense destination.
        let plan = check(&[5, 8], 4, &[8, 1], &[16, 1], 1 << 20);

        assert!(plan.linear().is_empty());
        assert_eq!(
            plan.pitched().copies(),
            [Pitched {
                destination: BASE,
                destination_pitch: 32,
                destination_layer: 5,
                source: BASE,
                source_pitch: 64,
                source_layer: 5,
                width: 32,
                height: 5,
                depth: 1,
            }]
        );
    }

    #[test]
    fn layered_operands_copy_as_one_three_dimensional_copy() {
        // [layer, row, column] with padded rows and padded layers.
        let plan = check(&[3, 4, 6], 2, &[40, 8, 1], &[60, 12, 1], 1 << 20);

        assert!(plan.linear().is_empty());
        let copies = plan.pitched().copies();
        assert_eq!(copies.len(), 1);
        assert_eq!(
            (copies[0].width, copies[0].height, copies[0].depth),
            (12, 4, 3)
        );
    }

    #[test]
    fn transposed_operands_copy_element_by_element_when_not_mergeable() {
        // A transpose has no contiguous run longer than one element; rows of
        // one element with a pitch still cover one axis per copy.
        check(&[3, 5], 4, &[5, 1], &[1, 3], 1 << 20);
    }

    #[test]
    fn pitches_above_the_device_limit_are_enumerated() {
        let plan = check(&[6, 16], 1, &[16, 1], &[4096, 1], 1024);

        assert!(plan.pitched().is_empty());
        assert_eq!(plan.linear().len(), 6);
    }

    #[test]
    fn outer_axes_enumerate_copies() {
        // Four independent blocks whose layer stride is not a whole number of
        // source rows.
        check(&[4, 3, 5], 1, &[15, 5, 1], &[77, 9, 1], 1 << 20);
    }

    #[test]
    fn empty_shapes_plan_nothing() {
        let plan = check(&[0, 4], 4, &[4, 1], &[4, 1], 1 << 20);

        assert!(plan.is_empty());
    }

    #[test]
    fn invalid_operands_are_rejected_without_planning() {
        let mut plan = StridedCopy::new(SourceOrder::Stream, 1 << 20);
        let shape = [4_u64, 4];
        let source = Strided {
            address: BASE,
            element_bytes: 4,
            shape: &shape,
            strides: &[4, 1],
        };

        let broadcast = Strided {
            strides: &[0, 1],
            ..source
        };
        assert!(matches!(
            plan.plan(&broadcast, &source),
            Err(Error::InvalidArgument(_))
        ));

        let other_shape = [2_u64, 8];
        let reshaped = Strided {
            shape: &other_shape,
            strides: &[8, 1],
            ..source
        };
        assert!(matches!(
            plan.plan(&reshaped, &source),
            Err(Error::InvalidArgument(_))
        ));

        let negative = Strided {
            strides: &[-4, 1],
            ..source
        };
        assert!(matches!(
            plan.plan(&source, &negative),
            Err(Error::InvalidArgument(_))
        ));
        assert!(plan.is_empty());
    }

    #[test]
    fn pitched_copies_reject_overlapping_rows() {
        let mut batch = PitchedBatch::new(SourceOrder::Stream);
        let copy = Pitched {
            destination: BASE,
            destination_pitch: 8,
            destination_layer: 1,
            source: BASE,
            source_pitch: 16,
            source_layer: 1,
            width: 16,
            height: 2,
            depth: 1,
        };

        assert!(matches!(batch.push(copy), Err(Error::InvalidArgument(_))));
        assert!(batch.is_empty());
    }

    #[test]
    fn copy_batches_skip_empty_copies() {
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        batch.push(BASE, BASE + 64, 0);
        batch.push(BASE + 128, BASE, 32);

        assert_eq!(batch.iter().collect::<Vec<_>>(), [(BASE + 128, BASE, 32)]);
    }
}
