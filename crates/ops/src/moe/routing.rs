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
//! `Descriptor::renormalize` holds; an optional per-expert scale multiplies the result in FP32.
//! With a map, each route's id is written as the weight row the map names: the selection is over
//! logical ids, so which experts a token picks, and the order of equal scores, do not depend on
//! where the experts lie (`docs/admin.md`, Expert redistribution). The map is a launch value, a
//! null pointer without one, so a captured graph reads whatever the table at its address says.
//! The arithmetic and its order are those of PyTorch's warp softmax and row-sum composition.
//!
//! The bound between two implementations: the probabilities are a scale of positive terms, so the
//! weights carry the `rounding` bound with the reduction length of the softmax, `experts`, and the
//! FP32 accumulation and output; the expert ids are exact — the descending-then-ascending order is
//! part of the operator's rule, not an implementation detail, so two correct implementations name the same
//! experts in the same slots.
//!
//! The widest router a kernel serves is its package's: the expert counts and top-k widths its
//! manifest's descriptor ranges admit. A descriptor outside them is no key the package covers, so
//! the Build Server refuses it when the deployment's modules are prepared, naming the key. One
//! warp routes one token, so a block routes as many tokens as the module's metadata states warps,
//! which is the grid this module computes.

use std::fmt;

use platform::DevicePtr;
use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

use super::row_stride;

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
            Some(dtype) => abi::name(dtype).to_owned(),
            None => "none".to_owned(),
        };
        Key::new()
            .field("scores", abi::name(self.scores))
            .field("experts", self.experts)
            .field("top", self.top)
            .field("renormalize", self.renormalize)
            .variant("scale", &scale)
            .finish(formatter)
    }
}

/// The arguments of one routing call.
#[derive(Clone)]
pub struct TopkArgs {
    /// `[tokens, experts]` router scores, unit stride on the expert axis.
    pub scores: Operand,
    /// `[experts]` per-expert scale multiplied into the weights, when the descriptor has one.
    pub scale: Option<Operand>,
    /// `[tokens, top]` u32 expert ids, written.
    pub ids: Operand,
    /// `[tokens, top]` FP32 route weights, written.
    pub weights: Operand,
    /// `[experts]` u32 weight row of each logical expert, when the block's experts are not in
    /// logical order: `ids` then holds rows.
    pub map: Option<Operand>,
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
    const NAME: &'static str = proto::module::operator::TOPK_SOFTMAX;

    type Descriptor = TopkDescriptor;
    type Args<'a> = TopkArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let tokens = args.scores.rows();

        if !matches!(descriptor.scores, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::Unsupported {
                operation: Self::NAME,
                reason: format!(
                    "scores of {} are not routeable",
                    abi::name(descriptor.scores)
                ),
            });
        }
        if descriptor.experts == 0 {
            return Err(Error::Unsupported {
                operation: Self::NAME,
                reason: "a router selects over at least one expert".to_owned(),
            });
        }
        if descriptor.top == 0 || u64::from(descriptor.top) > experts {
            return Err(Error::Unsupported {
                operation: Self::NAME,
                reason: format!(
                    "top {} is outside 1..={} experts",
                    descriptor.top, descriptor.experts
                ),
            });
        }

        args.scores.require_shape(Self::NAME, &[tokens, experts])?;
        args.scores
            .require_row(Self::NAME, descriptor.scores, experts)?;
        args.ids.require_shape(Self::NAME, &[tokens, top])?;
        args.weights.require_shape(Self::NAME, &[tokens, top])?;
        if args.ids.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected u32 ids, found {}", abi::name(args.ids.dtype)),
            ));
        }
        if args.weights.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected FP32 weights, found {}",
                    abi::name(args.weights.dtype)
                ),
            ));
        }

        match (descriptor.scale, &args.scale) {
            (Some(dtype), Some(scale)) => {
                scale.require_shape(Self::NAME, &[experts])?;
                if scale.dtype != dtype {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "expected {} scales, found {}",
                            abi::name(dtype),
                            abi::name(scale.dtype)
                        ),
                    ));
                }
            }
            (None, None) => {}
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
        }

        if let Some(map) = &args.map {
            map.require_shape(Self::NAME, &[experts])?;
            if map.dtype != DType::U32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected a u32 expert map, found {}", abi::name(map.dtype)),
                ));
            }
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let tokens = args.scores.rows();
        // `check` holds the scale to the descriptor, so a declared scale is always passed.
        let (scale, scale_kind) = match (descriptor.scale, args.scale) {
            (Some(dtype), Some(scale)) => (Some(scale.ptr), Self::score_kind(dtype)),
            _ => (None, -1),
        };

        // One warp routes one token, so a block routes as many tokens as it holds warps.
        let metadata = kernel.metadata().ok_or_else(|| Error::KernelForm {
            operator: Self::NAME,
            kernel: kernel.name().to_owned(),
            expected: "cubin",
            actual: "native",
        })?;
        let grid = LaunchGrid::covering(tokens, u64::from(metadata.num_warps))?;
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("scores", &args.scores)
            .i32("score_kind", Self::score_kind(descriptor.scores))
            .i64("row_stride", to_i64(Self::NAME, row_stride(&args.scores))?)
            .i32("rows", to_i32(Self::NAME, tokens)?)
            .i32("experts", to_i32(Self::NAME, experts)?)
            .i32("top", to_i32(Self::NAME, top)?)
            .i32("renormalize", i32::from(descriptor.renormalize));
        match scale {
            Some(pointer) => {
                arguments
                    .pointer("scale", pointer)
                    .i32("scale_kind", scale_kind);
            }
            None => {
                // The module's parameter is a device address whatever the scale kind says: an
                // absent scale is a **null pointer** with the kind telling the program not to read
                // it, not a scalar zero. The driver takes one machine word per parameter and the
                // module declares a pointer there, so a scalar in that slot is refused before the
                // launch — and if it were not, the program would read address zero.
                arguments
                    .pointer("scale", DevicePtr::from_address(0))
                    .i32("scale_kind", scale_kind);
            }
        }
        arguments
            .view("ids", &args.ids)
            .view("weights", &args.weights);
        arguments.pointer(
            "map",
            args.map.map_or(DevicePtr::from_address(0), |map| map.ptr),
        );
        kernel.launch(grid, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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
            (f64::from(rounding.rtol) + 4.0 * tolerance::unit_roundoff(accumulate)) as f32,
        )
    }
}

fn to_i64(operator: &'static str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument(operator, "an extent above i64::MAX"))
}

fn to_i32(operator: &'static str, value: u64) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::argument(operator, "an extent above i32::MAX"))
}
