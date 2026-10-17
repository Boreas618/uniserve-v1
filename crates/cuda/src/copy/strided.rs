//! Strided copies: a planner that lowers a copy between two equally shaped
//! strided operands to contiguous and pitched copies.

use super::{CopyBatch, Pitched, PitchedBatch, SourceOrder};
use crate::{DeviceAttribute, DeviceId, Error, Result, Stream};

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
