//! Expert routing: the router's softmax top-k, and the logical-to-physical expert map.
//!
//! Both operators are single cubins whose launch shapes follow from the token count, so this module
//! computes their grids; everything a module bakes in — the score dtype, the expert count, the top-k
//! and whether the selected weights renormalize — is in the descriptor.
//!
//! # `topk_softmax`
//!
//! One warp routes one token: it takes the row's maximum, sums `expf(score − max)` over the warp by
//! xor butterflies, keeps the `top` largest probabilities in descending order with equal ones in
//! ascending expert order, and divides them by their sum clamped below by `FLT_EPSILON` when
//! [`Descriptor::renormalize`] holds; an optional per-expert scale multiplies the result in FP32.
//! The arithmetic and its order are those of the existing CUDA kernel
//! (`uniserve_kernels/src/uniserve_kernels/csrc/topk_softmax.cu:52-197`), which in turn reproduces
//! PyTorch's warp softmax and row-sum composition.
//!
//! The bound between two implementations: the probabilities are a scale of positive terms, so the
//! weights carry the `rounding` bound with the reduction length of the softmax, `experts`, and the
//! FP32 accumulation and output; the expert ids are exact — the descending-then-ascending order is
//! part of the contract, not an implementation detail, so two correct implementations name the same
//! experts in the same slots.
//!
//! # `expert_gather`
//!
//! `out[i] = map[ids[i]]` for every flattened `(token, route)` entry: the expert-parallel rank's
//! logical ids become indices into the physical expert weights. It moves integers, so two
//! implementations agree bit for bit.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::row_stride;

/// The warps one thread block of `topk_softmax` holds.
///
/// One warp routes one token, so a block covers [`TOKENS_PER_BLOCK`] tokens; the package compiles the
/// kernel with exactly this many warps and the launch checks it.
pub const WARPS_PER_BLOCK: u64 = 4;

/// The tokens one `topk_softmax` thread block routes.
pub const TOKENS_PER_BLOCK: u64 = WARPS_PER_BLOCK;

/// The largest expert count one warp holds.
pub const MAX_EXPERTS: u32 = 256;

/// The largest top-k the renormalizing butterfly holds.
pub const MAX_TOP: u32 = 8;

/// One compilation of `topk_softmax`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TopkDescriptor {
    /// Element type of the router scores.
    pub scores: DType,
    /// Experts in the router's vocabulary: the softmax reduction length.
    pub experts: u32,
    /// Experts each token routes to.
    pub top: u32,
    /// Divide the selected probabilities by their sum.
    pub renormalize: bool,
    /// Element type of the optional per-expert scale vector.
    pub scale: Option<DType>,
}

impl fmt::Display for TopkDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scale = match self.scale {
            Some(dtype) => dtype.to_string(),
            None => "none".to_owned(),
        };
        Key::new()
            .field("scores", self.scores)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("renormalize", self.renormalize)
            .variant("scale", &scale)
            .finish(formatter)
    }
}

/// The arguments of one routing call.
pub struct TopkArgs {
    /// `[tokens, experts]` router scores, unit stride on the expert axis.
    pub scores: TensorView,
    /// `[experts]` per-expert scale multiplied into the weights, when the descriptor has one.
    pub scale: Option<TensorView>,
    /// `[tokens, top]` i32 expert ids, written.
    pub ids: TensorView,
    /// `[tokens, top]` FP32 route weights, written.
    pub weights: TensorView,
}

/// The router's softmax top-k.
pub struct TopkSoftmax;

impl TopkSoftmax {
    /// The storage kind the reference kernel's parameter takes.
    const fn score_kind(dtype: DType) -> i32 {
        match dtype {
            DType::F32 => 0,
            DType::BF16 => 1,
            DType::F16 => 2,
            _ => -1,
        }
    }
}

impl Op for TopkSoftmax {
    const NAME: &'static str = "topk_softmax";

    type Descriptor = TopkDescriptor;
    type Args<'a> = TopkArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let tokens = args.scores.rows();

        if !matches!(descriptor.scores, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::Unsupported {
                operator: Self::NAME,
                problem: format!("scores of {} are not routeable", descriptor.scores),
            });
        }
        if descriptor.experts == 0 || descriptor.experts > MAX_EXPERTS {
            return Err(Error::Unsupported {
                operator: Self::NAME,
                problem: format!(
                    "{} experts are outside 1..={MAX_EXPERTS}",
                    descriptor.experts
                ),
            });
        }
        if descriptor.top == 0 || descriptor.top > MAX_TOP || u64::from(descriptor.top) > experts {
            return Err(Error::Unsupported {
                operator: Self::NAME,
                problem: format!("top {} is outside 1..={MAX_TOP}", descriptor.top),
            });
        }

        args.scores
            .require_shape(Self::NAME, Shape::new(&[tokens, experts]))?;
        args.scores
            .require_row(Self::NAME, descriptor.scores, experts)?;
        args.ids
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        args.weights
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        if args.ids.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 ids, found {}", args.ids.dtype),
            ));
        }
        if args.weights.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected FP32 weights, found {}", args.weights.dtype),
            ));
        }

        let (scale, scale_kind) = match (descriptor.scale, args.scale) {
            (Some(dtype), Some(scale)) => {
                scale.require_shape(Self::NAME, Shape::new(&[experts]))?;
                if scale.dtype != dtype {
                    return Err(Error::argument(
                        Self::NAME,
                        format!("expected {dtype} scales, found {}", scale.dtype),
                    ));
                }
                (Some(scale.ptr), Self::score_kind(dtype))
            }
            (None, None) => (None, -1),
            (Some(_), None) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor declares a per-expert scale and the call passed none",
                ));
            }
            (None, Some(_)) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the call passed a per-expert scale the descriptor does not declare",
                ));
            }
        };

        cubin_metadata(kernel, Self::NAME, WARPS_PER_BLOCK)?;
        let grid = LaunchGrid::covering(tokens, TOKENS_PER_BLOCK)?;
        let mut arguments = Arguments::new();
        arguments
            .view(args.scores)
            .i32(Self::score_kind(descriptor.scores))
            .i64(to_i64(Self::NAME, row_stride(&args.scores))?)
            .i32(to_i32(Self::NAME, tokens)?)
            .i32(to_i32(Self::NAME, experts)?)
            .i32(to_i32(Self::NAME, top)?)
            .i32(i32::from(descriptor.renormalize));
        match scale {
            Some(pointer) => {
                arguments.pointer(pointer).i32(scale_kind);
            }
            None => {
                arguments.u64(0).i32(scale_kind);
            }
        }
        arguments.view(args.ids).view(args.weights);
        kernel.launch(grid, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        TopkArgs {
            ids: twins.shadow_of(args.ids),
            weights: twins.shadow_of(args.weights),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.experts);
        let accumulate = DType::F32;
        // The softmax's maximum, its `expf` (one ulp of FP32), the butterfly sum and the divide cost
        // at most a few `u` on top of the reduction's own `γₙ`; the ids are exact.
        let rounding = Tolerance::rounding(reduction, accumulate, DType::F32);
        Tolerance::new(
            rounding.atol,
            (f64::from(rounding.rtol) + 4.0 * accumulate.unit_roundoff()) as f32,
        )
    }
}

/// The arguments of one expert-map call.
pub struct GatherArgs {
    /// `[tokens, top]` i32 logical expert ids.
    pub ids: TensorView,
    /// `[experts]` i32 physical expert index of each logical one.
    pub map: TensorView,
    /// `[tokens, top]` i32 physical expert ids, written.
    pub out: TensorView,
}

/// The logical-to-physical expert map of an expert-parallel placement.
pub struct ExpertGather;

impl Op for ExpertGather {
    const NAME: &'static str = "expert_gather";

    /// What one compilation covers: the expert vocabulary and the top-k it maps.
    type Descriptor = TopkDescriptor;

    type Args<'a> = GatherArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let tokens = args.ids.rows();

        args.ids
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        args.map.require_shape(Self::NAME, Shape::new(&[experts]))?;
        for view in [&args.ids, &args.out, &args.map] {
            if view.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected i32 entries, found {}", view.dtype),
                ));
            }
        }

        let threads = u64::from(cubin_metadata(kernel, Self::NAME, 1)?.threads());
        let grid = LaunchGrid::covering(args.ids.numel(), threads)?;
        let mut arguments = Arguments::new();
        arguments
            .view(args.ids)
            .view(args.map)
            .view(args.out)
            .i64(to_i64(Self::NAME, args.ids.numel())?);
        kernel.launch(grid, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        GatherArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

/// The launch metadata of a cubin kernel of `operator`, checking that its block holds `warps` warps.
///
/// An operator whose grid counts tokens per block depends on the package compiling the block it
/// expects; a module with another block width would route a different number of tokens per block,
/// so it fails here instead of launching.
fn cubin_metadata<'k>(
    kernel: &'k Kernel,
    operator: &'static str,
    warps: u64,
) -> Result<&'k crate::kernel::CubinMetadata> {
    let metadata = kernel.metadata().ok_or_else(|| Error::KernelForm {
        operator,
        kernel: kernel.name().to_owned(),
        expected: "cubin",
        actual: "native",
    })?;
    if u64::from(metadata.num_warps) != warps {
        return Err(Error::Metadata(format!(
            "{operator} launches blocks of {warps} warps and the module declares {}",
            metadata.num_warps
        )));
    }
    Ok(metadata)
}

fn to_i64(operator: &'static str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument(operator, "an extent above i64::MAX"))
}

fn to_i32(operator: &'static str, value: u64) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::argument(operator, "an extent above i32::MAX"))
}
