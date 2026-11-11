//! What the tensor primitives share: the rules their strided operands meet, the axes a program
//! traverses, and the launch words that carry extents and strides.
//!
//! The primitives — `elementwise_unary`, `elementwise_binary`, `elementwise_where`, `cast`,
//! `copy_strided`, `reduce`, `index` and `fill` — are the operators a tensor library's device
//! backend computes with. Each reads its inputs through strides: element `[i0, .., ir-1]` of an
//! operand is `Σ iₖ·strideₖ` elements from its address, a zero stride repeats an axis (which is
//! how a broadcast operand is stated), and an output is contiguous unless its operator says
//! otherwise. Extents and strides are words of the launch and never of the key: a program is
//! compiled for a number of axes and serves every shape that has them.
//!
//! ## The axes a program traverses
//!
//! A run of adjacent axes that every operand of a call steps through as one — the outer stride is
//! the inner extent times the inner stride, in each operand — is one axis as far as the address
//! arithmetic goes, and an axis of one element is none at all. [`merge_axes`] folds such runs, and
//! the number of axes left is the `rank` a primitive's descriptor states: a call over contiguous
//! operands of any shape is rank 1, a bias broadcast over the leading axes is rank 2, and only a
//! permuted operand needs as many axes as it has. Each primitive's `Descriptor::of` states the
//! descriptor of a call and its `check` refuses any other, so one call has one key.
//!
//! ## The words
//!
//! A program takes [`MAX_RANK`] strides per strided operand and `MAX_RANK − 1` extents per index
//! space, right-aligned: axis `k` of `rank` axes is slot `MAX_RANK − rank + k`, and the slots
//! before it are unit extents and zero strides that a program compiled for `rank` never reads.
//! The extent of slot 0 does not travel: only an outermost axis can sit there, and a row-major
//! index needs no modulus on its outermost axis.

use crate::kernel::Arguments;
use crate::platform::Stream;
use crate::tensor::{DType, MAX_RANK, Operand, Shape, Strides};
use crate::{Error, Result, Twins};

/// The element types the primitives compute with.
pub(crate) const ELEMENTS: [DType; 9] = [
    DType::U8,
    DType::U32,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

/// The element types an index or a condition is read in.
pub(crate) const INDICES: [DType; 3] = [DType::U8, DType::U32, DType::I64];

/// Check that a descriptor's `field` names one of `allowed`.
pub(crate) fn require_element(
    operator: &'static str,
    field: &str,
    dtype: DType,
    allowed: &[DType],
) -> Result<()> {
    if allowed.contains(&dtype) {
        return Ok(());
    }
    Err(Error::argument(
        operator,
        format!("the descriptor's {field} is {dtype}, which is none of {allowed:?}"),
    ))
}

/// Check that a descriptor's block is a power of two, as the grid and the program both read it.
pub(crate) fn require_block(operator: &'static str, block: u32) -> Result<()> {
    if block.is_power_of_two() {
        return Ok(());
    }
    Err(Error::argument(
        operator,
        format!("a block of {block} elements is not a power of two"),
    ))
}

/// Check that the operand `name` holds `dtype` elements.
pub(crate) fn require_dtype(
    operator: &'static str,
    name: &str,
    operand: &Operand,
    dtype: DType,
) -> Result<()> {
    if operand.dtype == dtype {
        return Ok(());
    }
    Err(Error::argument(
        operator,
        format!("expected {dtype} {name}, found {}", operand.dtype),
    ))
}

/// Check that the operand `name` has the extents of `shape`.
pub(crate) fn require_shape(
    operator: &'static str,
    name: &str,
    operand: &Operand,
    shape: Shape,
) -> Result<()> {
    if operand.shape.matches(&shape) && operand.strides.rank() == shape.rank() {
        return Ok(());
    }
    Err(Error::argument(
        operator,
        format!(
            "expected {name} of shape {:?}, found {:?} with strides {:?}",
            shape, operand.shape, operand.strides
        ),
    ))
}

/// Check that the operand `name` has an element to compute and that its strides fit the signed
/// 64-bit words a program reads them as.
///
/// A call over no element launches nothing: the grid would be empty, and a launch that is not
/// issued is not a node of a captured graph either. Its caller handles it without an operator.
pub(crate) fn require_elements(
    operator: &'static str,
    name: &str,
    operand: &Operand,
) -> Result<()> {
    if operand.numel() == 0 {
        return Err(Error::argument(
            operator,
            format!(
                "{name} of shape {:?} has no element, so there is nothing to launch",
                operand.shape
            ),
        ));
    }
    require_words(operator, name, operand)
}

/// Check that the strides of the operand `name` fit the signed 64-bit words a program reads them
/// as.
pub(crate) fn require_words(operator: &'static str, name: &str, operand: &Operand) -> Result<()> {
    if let Some(stride) = operand
        .strides
        .as_slice()
        .iter()
        .find(|&&stride| i64::try_from(stride).is_err())
    {
        return Err(Error::argument(
            operator,
            format!("a stride of {stride} elements in {name} does not fit a launch word"),
        ));
    }
    Ok(())
}

/// Whether an operand is contiguous: its elements follow each other in row-major order from its
/// address.
///
/// An axis of one element may carry any stride, since no step is ever taken along it.
pub(crate) fn is_contiguous(operand: &Operand) -> bool {
    let mut expected = 1;
    for axis in (0..operand.shape.rank()).rev() {
        let extent = operand.shape.dim(axis);
        if extent != 1 && operand.strides.get(axis) != expected {
            return false;
        }
        expected *= extent;
    }
    true
}

/// Check that the operand `name` is contiguous ([`is_contiguous`]).
pub(crate) fn require_contiguous(
    operator: &'static str,
    name: &str,
    operand: &Operand,
) -> Result<()> {
    if is_contiguous(operand) {
        return Ok(());
    }
    Err(Error::argument(
        operator,
        format!(
            "{name} must be contiguous: its strides are {:?} over {:?}",
            operand.strides, operand.shape
        ),
    ))
}

/// Check that the operand `name` is a row-major window: a destination whose elements keep their
/// row-major order and never share an address.
///
/// Each axis steps past everything the axes inside it reach, which is what a contiguous tensor
/// does and what a block of a larger contiguous tensor does with the larger tensor's strides. A
/// permuted destination and a broadcast one are refused: the first has no caller, and the second
/// would be written once per repetition.
pub(crate) fn require_window(operator: &'static str, name: &str, operand: &Operand) -> Result<()> {
    // The largest offset the axes inside the current one reach.
    let mut reach = 0_u64;
    for axis in (0..operand.shape.rank()).rev() {
        let extent = operand.shape.dim(axis);
        if extent <= 1 {
            continue;
        }
        let stride = operand.strides.get(axis);
        if stride <= reach {
            return Err(Error::argument(
                operator,
                format!(
                    "axis {axis} of {name} advances by {stride} elements while the axes inside it \
                     reach {reach}: its strides {:?} over {:?} are no row-major window",
                    operand.strides, operand.shape
                ),
            ));
        }
        reach = reach.saturating_add((extent - 1).saturating_mul(stride));
    }
    Ok(())
}

/// Check that a descriptor states the axes its call traverses.
pub(crate) fn require_rank(
    operator: &'static str,
    field: &str,
    stated: u8,
    traversed: u8,
) -> Result<()> {
    if stated == traversed {
        return Ok(());
    }
    Err(Error::argument(
        operator,
        format!("the operands traverse {traversed} axes and the descriptor's {field} is {stated}"),
    ))
}

/// The elements from an operand's address to its last element, inclusive: the storage a strided
/// operand reaches.
pub(crate) fn span(operand: &Operand) -> u64 {
    let last = operand
        .shape
        .dims()
        .iter()
        .zip(operand.strides.as_slice())
        .map(|(&extent, &stride)| extent.saturating_sub(1).saturating_mul(stride))
        .fold(0_u64, u64::saturating_add);
    last.saturating_add(1)
}

/// The shadow buffer of a destination that is a row-major window, for an operator's `shadow`.
///
/// A contiguous destination is written whole, so its shadow buffer needs none of its bytes. A
/// window leaves the elements between its own untouched: its shadow buffer is the storage the
/// window reaches, with that storage's bytes copied in, and the window keeps its strides inside
/// it. The two sides of the comparison then differ only where the operator wrote.
pub(crate) fn shadow_window(destination: Operand, twins: &mut Twins, stream: &Stream) -> Operand {
    if is_contiguous(&destination) {
        return twins.shadow_of(destination);
    }

    let reached = Operand::new(
        destination.ptr,
        destination.dtype,
        Shape::new(&[span(&destination)]),
    );
    destination.at(twins.in_place(reached, stream).ptr)
}

/// The operands with every run of adjacent axes they all step through as one merged into one axis.
///
/// The operands share one shape, which is the rule of every elementwise call. Two adjacent axes
/// merge when each operand's outer stride is the inner extent times its inner stride, and an axis
/// of one element merges with its neighbour whatever stride it carries. An operand of no axis
/// comes back as one axis of one element. The result names the same elements at the same
/// addresses in the same row-major order.
///
/// Operands of differing shapes come back as they are, for the operator's `check` to refuse.
pub(crate) fn merge_axes<const N: usize>(operands: [Operand; N]) -> [Operand; N] {
    let Some(first) = operands.first() else {
        return operands;
    };
    let shape = first.shape;
    let alike = operands
        .iter()
        .all(|operand| operand.shape.matches(&shape) && operand.strides.rank() == shape.rank());
    if !alike {
        return operands;
    }

    // Built from the last axis outward, so slot 0 is the innermost run. A slot never written is
    // the one axis of an operand that has none: one element, no step.
    let mut extents = [1_u64; MAX_RANK];
    let mut strides = [[0_u64; MAX_RANK]; N];
    let mut axes = 0;

    for axis in (0..shape.rank()).rev() {
        let extent = shape.dim(axis);

        if axes > 0 {
            let inner = axes - 1;
            let inner_extent = extents[inner];
            let together = extent == 1
                || inner_extent == 1
                || operands.iter().zip(&strides).all(|(operand, merged)| {
                    inner_extent.checked_mul(merged[inner]) == Some(operand.strides.get(axis))
                });

            if together {
                if inner_extent == 1 {
                    // A run of one element takes no step of its own: it takes this axis's.
                    for (operand, merged) in operands.iter().zip(&mut strides) {
                        merged[inner] = operand.strides.get(axis);
                    }
                }
                extents[inner] = extent.saturating_mul(inner_extent);
                continue;
            }
        }

        extents[axes] = extent;
        for (operand, merged) in operands.iter().zip(&mut strides) {
            merged[axes] = operand.strides.get(axis);
        }
        axes += 1;
    }
    let axes = axes.max(1);

    let mut dims = [0_u64; MAX_RANK];
    for (axis, dim) in dims[..axes].iter_mut().enumerate() {
        *dim = extents[axes - 1 - axis];
    }
    let shape = Shape::new(&dims[..axes]);

    std::array::from_fn(|operand| {
        let mut steps = [0_u64; MAX_RANK];
        for (axis, step) in steps[..axes].iter_mut().enumerate() {
            *step = strides[operand][axes - 1 - axis];
        }
        Operand::with_strides(
            operands[operand].ptr,
            operands[operand].dtype,
            shape,
            Strides::new(&steps[..axes]),
        )
    })
}

/// The number of axes a program traverses for `operands`: what is left once [`merge_axes`] has
/// folded them.
pub(crate) fn rank<const N: usize>(operands: [Operand; N]) -> u8 {
    merge_axes(operands)
        .first()
        .map_or(1, |operand| operand.shape.rank() as u8)
}

/// The view of `operand` along the axes `keep` selects, at index zero of every other axis.
pub(crate) fn select_axes(operand: &Operand, keep: impl Fn(usize) -> bool) -> Operand {
    let mut dims = [0_u64; MAX_RANK];
    let mut strides = [0_u64; MAX_RANK];
    let mut axes = 0;

    for axis in 0..operand.shape.rank() {
        if keep(axis) {
            dims[axes] = operand.shape.dim(axis);
            strides[axes] = operand.strides.get(axis);
            axes += 1;
        }
    }

    Operand::with_strides(
        operand.ptr,
        operand.dtype,
        Shape::new(&dims[..axes]),
        Strides::new(&strides[..axes]),
    )
}

/// Pack the extents of an index space: the words of slots 1 to `MAX_RANK − 1`, right-aligned,
/// unit before the first axis.
pub(crate) fn pack_extents(arguments: &mut Arguments, shape: Shape) {
    let pad = MAX_RANK - shape.rank();
    for word in 1..MAX_RANK {
        arguments.u64(if word < pad { 1 } else { shape.dim(word - pad) });
    }
}

/// Pack an operand's strides: `MAX_RANK` words, right-aligned, zero before the first axis.
pub(crate) fn pack_strides(arguments: &mut Arguments, strides: Strides) {
    let pad = MAX_RANK - strides.rank();
    for word in 0..MAX_RANK {
        arguments.u64(if word < pad {
            0
        } else {
            strides.get(word - pad)
        });
    }
}
