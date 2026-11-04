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
//! bit — the reason the two-stage form is used instead of an atomic add
//! (`uniserve/backends/moe/cutedsl.py:32-39`).
//!
//! The program is a native library (FlashInfer's `moe_unpermute`, with the host-side launch
//! configuration of its JIT module), so [`Op::launch`] computes no grid. It is exact in its
//! indexing and rounds once: the bound is the `top`-term reduction with the output's own rounding,
//! stated as [`Tolerance::dot`] because the products carry the routes' signs.

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::{address, dtype_code, to_i64};

/// One compilation of `moe_unpermute`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
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
            .field("hidden", self.hidden)
            .field("top", self.top)
            .field("experts", self.experts)
            .field("input", self.input)
            .field("scales", self.scales)
            .field("output", self.output)
            .finish(formatter)
    }
}

/// The arguments of one combine.
#[derive(Clone, Copy)]
pub struct UnpermuteArgs {
    /// `[tokens · top, hidden]` every route's expert product.
    pub routes: TensorView,
    /// `[tokens, top]` i32: expanded route of each `(token, route)` pair, `-1` when absent.
    pub expanded_to_permuted: TensorView,
    /// `[tokens, top]` route weights, in the descriptor's scale dtype.
    pub scales: TensorView,
    /// `[tokens, hidden]` combined rows, written.
    pub out: TensorView,
}

/// The call structure the package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/moe_unpermute/<backend>/include/uniserve_moe_unpermute.h`.
#[repr(C)]
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
    pub error: [u8; 512],
}

/// The size and field offsets the package's header must mirror, in bytes.
pub const UNPERMUTE_CALL_LAYOUT: CallLayout = CallLayout {
    size: 600,
    routes: 0,
    out: 8,
    expanded_to_permuted: 16,
    scales: 24,
    tokens: 32,
    top: 40,
    hidden: 48,
    routes_rows: 56,
    input_dtype: 64,
    scales_dtype: 72,
    output_dtype: 80,
    error: 88,
};

/// The layout of [`Call`], for the header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct CallLayout {
    pub size: usize,
    pub routes: usize,
    pub out: usize,
    pub expanded_to_permuted: usize,
    pub scales: usize,
    pub tokens: usize,
    pub top: usize,
    pub hidden: usize,
    pub routes_rows: usize,
    pub input_dtype: usize,
    pub scales_dtype: usize,
    pub output_dtype: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<Call>() == UNPERMUTE_CALL_LAYOUT.size,
    "the moe_unpermute call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, tokens) == UNPERMUTE_CALL_LAYOUT.tokens,
    "the moe_unpermute call structure's extents moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, error) == UNPERMUTE_CALL_LAYOUT.error,
    "the moe_unpermute call structure's `error` moved without its header"
);

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// Combining the routes of each token.
pub struct MoeUnpermute;

impl Op for MoeUnpermute {
    const NAME: &'static str = "moe_unpermute";

    type Descriptor = Descriptor;
    type Args<'a> = UnpermuteArgs;

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

        args.routes
            .require_shape(Self::NAME, Shape::new(&[routes, hidden]))?;
        args.routes
            .require_row(Self::NAME, descriptor.input, hidden)?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
        args.out
            .require_row(Self::NAME, descriptor.output, hidden)?;
        args.expanded_to_permuted
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        args.scales
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        if args.expanded_to_permuted.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected i32 route maps, found {}",
                    args.expanded_to_permuted.dtype
                ),
            ));
        }
        if args.scales.dtype != descriptor.scales {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} route weights, found {}",
                    descriptor.scales, args.scales.dtype
                ),
            ));
        }
        if routes < tokens * top {
            return Err(Error::argument(
                Self::NAME,
                format!("{routes} route rows cannot hold {} routes", tokens * top),
            ));
        }

        let mut call = Call {
            routes: address(args.routes.ptr) as usize as *const c_void,
            out: address(args.out.ptr) as usize as *mut c_void,
            expanded_to_permuted: address(args.expanded_to_permuted.ptr) as usize as *const i32,
            scales: address(args.scales.ptr) as usize as *const c_void,
            tokens: to_i64(Self::NAME, tokens)?,
            top: to_i64(Self::NAME, top)?,
            hidden: to_i64(Self::NAME, hidden)?,
            routes_rows: to_i64(Self::NAME, routes)?,
            input_dtype: dtype_code(descriptor.input),
            scales_dtype: dtype_code(descriptor.scales),
            output_dtype: dtype_code(descriptor.output),
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        UnpermuteArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::dot(u64::from(descriptor.top), DType::F32, descriptor.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_layout_is_the_one_the_header_declares() {
        let layout = UNPERMUTE_CALL_LAYOUT;
        assert_eq!(std::mem::size_of::<Call>(), layout.size);
        assert_eq!(std::mem::offset_of!(Call, routes), layout.routes);
        assert_eq!(std::mem::offset_of!(Call, out), layout.out);
        assert_eq!(
            std::mem::offset_of!(Call, expanded_to_permuted),
            layout.expanded_to_permuted
        );
        assert_eq!(std::mem::offset_of!(Call, scales), layout.scales);
        assert_eq!(std::mem::offset_of!(Call, tokens), layout.tokens);
        assert_eq!(std::mem::offset_of!(Call, top), layout.top);
        assert_eq!(std::mem::offset_of!(Call, hidden), layout.hidden);
        assert_eq!(std::mem::offset_of!(Call, routes_rows), layout.routes_rows);
        assert_eq!(std::mem::offset_of!(Call, input_dtype), layout.input_dtype);
        assert_eq!(
            std::mem::offset_of!(Call, scales_dtype),
            layout.scales_dtype
        );
        assert_eq!(
            std::mem::offset_of!(Call, output_dtype),
            layout.output_dtype
        );
        assert_eq!(std::mem::offset_of!(Call, error), layout.error);
    }

    #[test]
    fn the_bound_is_the_route_reduction() {
        let descriptor = Descriptor {
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
