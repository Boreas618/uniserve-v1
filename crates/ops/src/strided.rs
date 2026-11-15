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
//! A program takes `slots` strides per strided operand (`s0`, `s1`, ...) and `slots − 1` extents
//! per index space (`d1`, ...), right-aligned: axis `k` of `rank` axes is slot
//! `slots − rank + k`, and the slots before it are unit extents and zero strides that a program
//! compiled for `rank` never reads. The extent of slot 0 does not travel: only an outermost axis
//! can sit there, and a row-major index needs no modulus on its outermost axis.
//!
//! The slots are the kernel's own parameters, so a launch counts them in the signature of the
//! module it launches ([`Slots::of`]). An operand takes as many axes as a tensor does, and the
//! folding brings a call down to the axes its address arithmetic needs; which ranks a package
//! compiles is its manifest's statement (the descriptor's `rank` dimension), so a call that still
//! traverses more axes than a package covers names a key no package builds.

use tensor::nditer::merge_axes as fold;
use tensor::{DType, Layout, Shape};

use crate::kernel::{Arguments, Kernel};
use crate::operand::Operand;
use crate::{Error, Result, Twins, abi};

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

/// The element types the primitives move without computing: `fill`, `copy_strided`,
/// `elementwise_where`, and the `gather` and `scatter_set` of `index`.
///
/// They are the types of [`ELEMENTS`] and `f8e4m3`, whose elements travel as the byte each one
/// is: a program addresses the operand as `u8` and stores the bits it loaded, so an FP8 tensor
/// is zeroed, copied, concatenated, gathered and selected with no conversion to round. No
/// primitive computes with an FP8 element; arithmetic on one is a fused operator's.
///
/// The NVFP4 element (`f4`) is not among them. Two of its elements share a byte, so no strided
/// program can address one: a caller that moves NVFP4 storage states it as the `u8` tensor of
/// its packed bytes, and the operators that decode it name its packed layout themselves.
pub(crate) const MOVED: [DType; 10] = [
    DType::U8,
    DType::U32,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::F8E4M3,
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
        format!(
            "the descriptor's {field} is {}, which is none of {allowed:?}",
            abi::name(dtype)
        ),
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
        format!(
            "expected {} {name}, found {}",
            abi::name(dtype),
            abi::name(operand.dtype)
        ),
    ))
}

/// Check that the operand `name` has the extents `dims`.
pub(crate) fn require_shape(
    operator: &'static str,
    name: &str,
    operand: &Operand,
    dims: &[usize],
) -> Result<()> {
    if operand.dims() == dims {
        return Ok(());
    }
    Err(Error::argument(
        operator,
        format!(
            "expected {name} of shape {dims:?}, found {:?} with strides {:?}",
            operand.dims(),
            operand.strides()
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
                operand.dims()
            ),
        ));
    }
    require_words(operator, name, operand)
}

/// Check that the strides of the operand `name` fit the signed 64-bit words a program reads them
/// as.
pub(crate) fn require_words(operator: &'static str, name: &str, operand: &Operand) -> Result<()> {
    if let Some(stride) = operand
        .strides()
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
    for axis in (0..operand.rank()).rev() {
        let extent = operand.dim(axis);
        if extent != 1 && operand.stride(axis) != expected {
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
            operand.strides(),
            operand.dims()
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
    for axis in (0..operand.rank()).rev() {
        let extent = operand.dim(axis);
        if extent <= 1 {
            continue;
        }
        let stride = operand.stride(axis);
        if stride <= reach {
            return Err(Error::argument(
                operator,
                format!(
                    "axis {axis} of {name} advances by {stride} elements while the axes inside it \
                     reach {reach}: its strides {:?} over {:?} are no row-major window",
                    operand.strides(),
                    operand.dims()
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
        .dims()
        .iter()
        .zip(operand.strides())
        .map(|(&extent, &stride)| {
            (extent as u64)
                .saturating_sub(1)
                .saturating_mul(stride as u64)
        })
        .fold(0_u64, u64::saturating_add);
    last.saturating_add(1)
}

/// The shadow buffer of a destination that is a row-major window, for an operator's `shadow`.
///
/// A contiguous destination is written whole, so its shadow buffer needs none of its bytes. A
/// window leaves the elements between its own untouched: its shadow buffer is the storage the
/// window reaches, with that storage's bytes copied in, and the window keeps its strides inside
/// it. The two sides of the comparison then differ only where the operator wrote.
pub(crate) fn shadow_window(destination: Operand, twins: &mut Twins) -> Operand {
    if is_contiguous(&destination) {
        return twins.shadow_of(destination);
    }

    let reached = Operand::new(destination.ptr, destination.dtype, &[span(&destination)]);
    destination.at(twins.in_place(reached).ptr)
}

/// The operands with every run of adjacent axes they all step through as one merged into one axis,
/// by the tensor library's folding ([`merge_axes`](tensor::nditer::merge_axes)).
///
/// The operands share one shape, which is the rule of every elementwise call. The result names
/// the same elements at the same addresses in the same row-major order, with at least one axis.
///
/// Operands of differing shapes come back as they are, for the operator's `check` to refuse.
pub(crate) fn merge_axes<const N: usize>(operands: [&Operand; N]) -> [Operand; N] {
    let Some(first) = operands.first() else {
        return operands.map(Operand::clone);
    };
    let dims = first.dims();
    if !operands.iter().all(|operand| operand.dims() == dims) {
        return operands.map(Operand::clone);
    }

    let (extents, strides) = fold(dims, operands.map(Operand::strides));
    let shape = Shape::from(extents);

    std::array::from_fn(|operand| {
        let layout = Layout::new(shape.clone(), strides[operand].clone(), 0);
        Operand::with_layout(operands[operand].ptr, operands[operand].dtype, &layout)
    })
}

/// The number of axes a program traverses for `operands`: what is left once [`merge_axes`] has
/// folded them.
pub(crate) fn rank<const N: usize>(operands: [&Operand; N]) -> u8 {
    merge_axes(operands)
        .first()
        .map_or(1, |operand| operand.rank() as u8)
}

/// The view of `operand` along the axes `keep` selects, at index zero of every other axis.
pub(crate) fn select_axes(operand: &Operand, keep: impl Fn(usize) -> bool) -> Operand {
    let (dims, strides): (Vec<usize>, Vec<usize>) = (0..operand.rank())
        .filter(|&axis| keep(axis))
        .map(|axis| (operand.dims()[axis], operand.strides()[axis]))
        .unzip();

    let layout = Layout::new(Shape::from(dims), strides, 0);
    Operand::with_layout(operand.ptr, operand.dtype, &layout)
}

/// The axes a primitive's module addresses: the stride words it takes per strided operand.
///
/// A primitive's kernel states its slots as parameters (`s0` to `s5` of an operand, `d1` to `d5`
/// of an index space), so a launch counts them in the signature of the module it launches and packs
/// that many words; the names it packs are then checked by the same signature.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Slots {
    operator: &'static str,
    count: usize,
}

impl Slots {
    /// The slots of `kernel`'s module: the stride words `<prefix>0`, `<prefix>1`, ... its
    /// signature takes for one strided operand.
    ///
    /// # Errors
    ///
    /// [`Error::Metadata`] when the kernel is no cubin or its signature takes no such word.
    pub(crate) fn of(operator: &'static str, kernel: &Kernel, prefix: &str) -> Result<Self> {
        let metadata = kernel.metadata().ok_or_else(|| Error::KernelForm {
            operator,
            kernel: kernel.name().to_owned(),
            expected: "cubin",
            actual: "native",
        })?;
        let count = metadata
            .signature
            .iter()
            .filter(|parameter| {
                parameter
                    .name
                    .strip_prefix(prefix)
                    .is_some_and(|slot| slot.parse::<usize>().is_ok())
            })
            .count();
        if count == 0 {
            return Err(Error::Metadata(format!(
                "`{}` takes no stride word `{prefix}0`",
                metadata.name
            )));
        }
        Ok(Self { operator, count })
    }

    /// The unit or zero words before an operand's first axis, right-aligning its `rank` axes in
    /// the slots.
    fn pad(self, operand: &Operand) -> Result<usize> {
        self.count.checked_sub(operand.rank()).ok_or_else(|| {
            Error::argument(
                self.operator,
                format!(
                    "the operands traverse {} axes and the module addresses {}",
                    operand.rank(),
                    self.count
                ),
            )
        })
    }

    /// Pack the extents of an index space: the words of slots 1 to `slots − 1`, right-aligned,
    /// unit before the first axis, each named `<prefix><slot>` (`d1` to `d5`).
    ///
    /// # Errors
    ///
    /// An argument error when the operand has more axes than the module's slots.
    pub(crate) fn extents(
        self,
        arguments: &mut Arguments,
        prefix: &str,
        operand: &Operand,
    ) -> Result<()> {
        let pad = self.pad(operand)?;
        for slot in 1..self.count {
            arguments.u64(
                format!("{prefix}{slot}"),
                if slot < pad {
                    1
                } else {
                    operand.dim(slot - pad)
                },
            );
        }
        Ok(())
    }

    /// Pack an operand's strides: one word per slot, right-aligned, zero before the first axis,
    /// each named `<prefix><slot>` (`s0` to `s5`).
    ///
    /// # Errors
    ///
    /// An argument error when the operand has more axes than the module's slots.
    pub(crate) fn strides(
        self,
        arguments: &mut Arguments,
        prefix: &str,
        operand: &Operand,
    ) -> Result<()> {
        let pad = self.pad(operand)?;
        for slot in 0..self.count {
            arguments.u64(
                format!("{prefix}{slot}"),
                if slot < pad {
                    0
                } else {
                    operand.stride(slot - pad)
                },
            );
        }
        Ok(())
    }
}
