//! Combining the routes of each token into its output row.
//!
//! After the two grouped GEMMs the routes sit at expanded rows `t · top + k`, each holding that
//! route's expert product unweighted. `moe_unpermute` evaluates
//!
//! ```text
//! out[t] = Σ_{k visible} scales[t, k] · routes[t · top + k]
//! ```
//!
//! in FP32 and in route order, rounded once to the output dtype; a route the sort left out of this
//! rank carries `-1` in the map and contributes nothing, which is how an expert-parallel rank's
//! partial sum is expressed. The order is fixed, so repeated calls of the same inputs agree bit for
//! bit — the reason the two-stage form is used instead of an atomic add.
//!
//! The program is a native library (FlashInfer's `moe_unpermute`, with the host-side launch
//! configuration of its JIT module), so [`Op::launch`] computes no grid. It is exact in its
//! indexing and rounds once: the bound is the `top`-term reduction with the output's own rounding,
//! stated as [`Tolerance::dot`] because the products carry the routes' signs.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::{address, to_i64};

/// Which row convention the combine's input buffer is in.
///
/// The two products write different conventions and the combine's module is compiled for one of
/// them, so the choice is a descriptor field - one key per convention - rather than a constant in
/// the entry: `route_gemm`'s dense path writes one row per **expanded** route (`token · top + route`),
/// its encoded path writes one row per **permuted** row, and a module compiled for the wrong one
/// indexes a buffer with numbers that mean something else.
///
/// The call carries its number (`uniserve_unpermute_layout` in `uniserve_moe_unpermute.h`).
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Layout {
    /// One row per permuted row; `expanded_to_permuted` names the row of each `(token, route)`.
    Permuted = 0,
    /// One row per expanded route, in `token · top + route` order; the map is not read.
    Expanded = 1,
}

impl fmt::Display for Layout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Permuted => "permuted",
            Self::Expanded => "expanded",
        })
    }
}

/// One compilation of `moe_unpermute`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// The row convention the input buffer is in.
    pub layout: Layout,
    /// Hidden elements of a token.
    pub hidden: u32,
    /// Routes per token.
    pub top: u32,
    /// Experts the routing vocabulary holds, for validation only.
    pub experts: u32,
    /// Element type of the route rows.
    pub input: DType,
    /// Element type of the route weights.
    pub scales: DType,
    /// Element type of the combined rows.
    pub output: DType,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("layout", self.layout)
            .field("hidden", self.hidden)
            .field("top", self.top)
            .field("experts", self.experts)
            .field("input", abi::name(self.input))
            .field("scales", abi::name(self.scales))
            .field("output", abi::name(self.output))
            .finish(formatter)
    }
}

/// The arguments of one combine.
#[derive(Clone)]
pub struct UnpermuteArgs {
    /// `[rows, hidden]` every route's expert product, in the descriptor's [`Layout`]: permuted (one
    /// row per permuted row, found through `expanded_to_permuted`) or expanded (`token · top + route`).
    pub routes: Operand,
    /// `[tokens, top]` u32: the permuted row of each `(token, route)` pair, `-1` when absent.
    ///
    /// The module reads `routes` as the **permuted** buffer through this map. The dense `route_gemm`
    /// writes the other convention - one row per expanded route - so a dense deployment's combine
    /// needs a module compiled for expanded input, and the layout is not yet a descriptor field.
    pub expanded_to_permuted: Operand,
    /// `[tokens, top]` route weights, in the descriptor's scale dtype.
    pub scales: Operand,
    /// `[tokens, hidden]` combined rows, written.
    pub out: Operand,
}

/// The call structure the package's entry receives, declared to the packages as
/// `uniserve_unpermute_call` (`uniserve_moe_unpermute.h`, generated with its layout asserted:
/// [`abi`]).
#[repr(C)]
#[derive(abi::Layout)]
pub struct Call {
    pub routes: *const c_void,
    pub out: *mut c_void,
    pub expanded_to_permuted: *const i32,
    pub scales: *const c_void,
    pub tokens: i64,
    pub top: i64,
    pub hidden: i64,
    pub routes_rows: i64,
    pub input_dtype: i64,
    pub scales_dtype: i64,
    pub output_dtype: i64,
    /// The input buffer's row convention, the [`Layout`]'s number.
    pub layout: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// Combining the routes of each token.
pub struct MoeUnpermute;

impl Op for MoeUnpermute {
    const NAME: &'static str = proto::module::operator::MOE_UNPERMUTE;

    type Descriptor = Descriptor;
    type Args<'a> = UnpermuteArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let top = u64::from(descriptor.top);
        let tokens = args.out.rows();
        let routes = args.routes.rows();

        args.routes.require_shape(Self::NAME, &[routes, hidden])?;
        args.routes
            .require_row(Self::NAME, descriptor.input, hidden)?;
        args.out.require_shape(Self::NAME, &[tokens, hidden])?;
        args.out
            .require_row(Self::NAME, descriptor.output, hidden)?;
        args.expanded_to_permuted
            .require_shape(Self::NAME, &[tokens, top])?;
        args.scales.require_shape(Self::NAME, &[tokens, top])?;
        if args.expanded_to_permuted.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u32 route maps, found {}",
                    abi::name(args.expanded_to_permuted.dtype)
                ),
            ));
        }
        if args.scales.dtype != descriptor.scales {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} route weights, found {}",
                    abi::name(descriptor.scales),
                    abi::name(args.scales.dtype)
                ),
            ));
        }
        if routes < tokens * top {
            return Err(Error::argument(
                Self::NAME,
                format!("{routes} route rows cannot hold {} routes", tokens * top),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let top = u64::from(descriptor.top);
        let tokens = args.out.rows();
        let routes = args.routes.rows();

        let mut call = Call {
            routes: address(args.routes.ptr) as usize as *const c_void,
            out: address(args.out.ptr) as usize as *mut c_void,
            expanded_to_permuted: address(args.expanded_to_permuted.ptr) as usize as *const i32,
            scales: address(args.scales.ptr) as usize as *const c_void,
            tokens: to_i64(Self::NAME, tokens)?,
            top: to_i64(Self::NAME, top)?,
            hidden: to_i64(Self::NAME, hidden)?,
            routes_rows: to_i64(Self::NAME, routes)?,
            input_dtype: abi::code(descriptor.input)?,
            scales_dtype: abi::code(descriptor.scales)?,
            output_dtype: abi::code(descriptor.output)?,
            layout: descriptor.layout as i64,
            error: [0; MESSAGE],
        };
        // SAFETY: `Call` is the call structure this package's entry declares: the declaration is
        // generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        UnpermuteArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// A token's `top` routed rows are summed into one, so the reduction length is `top`.
    ///
    /// Each route's row is already the expert's own product; unpermuting adds `top` of them per token
    /// in FP32 and rounds once into the output's dtype, which is `dot(top, F32, output)` and nothing
    /// else — no activation touches the sum, so no envelope term is needed.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::dot(u64::from(descriptor.top), DType::F32, descriptor.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bound_is_the_route_reduction() {
        let descriptor = Descriptor {
            layout: Layout::Permuted,
            hidden: 2048,
            top: 8,
            experts: 128,
            input: DType::BF16,
            scales: DType::F32,
            output: DType::BF16,
        };
        let tolerance = MoeUnpermute::tolerance(&descriptor);
        assert!(tolerance.rtol > 0.0, "the output rounding must be covered");
        assert!(tolerance.allows(1.0, 1.0));
        assert!(!tolerance.allows(0.0, 1.0));
    }
}
