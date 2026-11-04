//! The two grouped expert GEMMs of a sparse layer.
//!
//! Every route of every token is one row of both products; the sort's metadata says which expert
//! owns each tile of rows, so one launch multiplies all experts' weights without grouping the rows
//! physically:
//!
//! ```text
//! gather_gemm (FC1)   hidden [T, H] · weights [E, 2I, H]
//!                     out[M_perm, I] = act(gate rows) · up rows, per row's expert
//! route_gemm  (FC2)   rows [M_perm, I] · weights [E, H, I]
//!                     out[T · top, H], route k of token t at row t · top + k, unweighted
//! ```
//!
//! Each expert's FC1 weight holds its `I` up rows followed by its `I` gate rows (Hugging Face stores
//! gate then up; the stacked `up_gate` stores up then gate, `uniserve/nn/moe.py`), and the gated
//! product is `act(gate) · up` with the activation the descriptor names.
//!
//! Both programs are CuTe DSL native libraries — tiles, cluster shapes and the persistent scheduler
//! are the package's, and its tuning data (which tactic for which token count) is the package's too
//! (specs/worker.md 5.4) — so [`Op::launch`] computes no grid. The runtime arguments are the ones
//! the BF16 wrappers pass today
//! (`uniserve_kernels/src/uniserve_kernels/experts/__init__.py:117-211`).
//!
//! # The bound
//!
//! A row of FC1 or FC2 is a dot product of length `H` or `I` accumulated in FP32 and rounded once to
//! BF16. Two correct implementations therefore differ by at most `2γₙ Σ|aᵢbᵢ| + u_out|result|`
//! (specs/worker.md 7.3, Higham §3.1); with a sum of mixed-sign products the result can cancel, so
//! [`Tolerance::dot`] states the bound relative to the result and [`ACTIVATION_ENVELOPE`] gives the
//! absolute term for rows that cancel harder, over the data this operator sees.

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::{address, dtype_code, to_i64};

/// The largest magnitude one activation element of a grouped GEMM is written for.
///
/// The rows are normalized hidden states (RMSNorm output, weight near one) for FC1, and gated
/// products of them for FC2; an element stays below 8 in the models served here. The envelope is a
/// declaration about that data, not something the descriptor can state.
pub const ACTIVATION_ENVELOPE: f32 = 8.0;

/// The gated activation a FC1 applies.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Activation {
    /// `silu(gate) · up`.
    Silu,
    /// `gelu_tanh(gate) · up`.
    GeluTanh,
}

impl fmt::Display for Activation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Silu => "silu",
            Self::GeluTanh => "gelu_tanh",
        })
    }
}

/// One compilation of `gather_gemm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatherGemmDescriptor {
    /// Hidden elements of a token: the contraction length.
    pub hidden: u32,
    /// Expert width: the output width of FC1.
    pub intermediate: u32,
    /// Experts the weight stacks.
    pub experts: u32,
    /// Routes per token.
    pub top: u32,
    /// The gated activation.
    pub activation: Activation,
    /// Element type of the activations, the weights and the output.
    pub dtype: DType,
    /// The MMA tile the package compiled for: rows by columns.
    pub tile: (u32, u32),
    /// The cluster shape the package compiled for: rows by columns.
    pub cluster: (u32, u32),
}

impl fmt::Display for GatherGemmDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("activation", self.activation)
            .field("dtype", self.dtype)
            .field("tile", format_args!("{}x{}", self.tile.0, self.tile.1))
            .field(
                "cluster",
                format_args!("{}x{}", self.cluster.0, self.cluster.1),
            )
            .finish(formatter)
    }
}

/// One compilation of `route_gemm`.
///
/// The same fields as [`GatherGemmDescriptor`] less the activation: FC2 has none (FC1 applies it
/// inside its gated product), and a descriptor carries only what its operator reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RouteGemmDescriptor {
    /// Hidden elements of a token: FC2's output width.
    pub hidden: u32,
    /// Expert width: FC2's contraction length.
    pub intermediate: u32,
    /// Experts the weight stacks.
    pub experts: u32,
    /// Routes per token.
    pub top: u32,
    /// Element type of the rows, the weights and the output.
    pub dtype: DType,
    /// The MMA tile the package compiled for: rows by columns.
    pub tile: (u32, u32),
    /// The cluster shape the package compiled for: rows by columns.
    pub cluster: (u32, u32),
}

impl fmt::Display for RouteGemmDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("dtype", self.dtype)
            .field("tile", format_args!("{}x{}", self.tile.0, self.tile.1))
            .field(
                "cluster",
                format_args!("{}x{}", self.cluster.0, self.cluster.1),
            )
            .finish(formatter)
    }
}

/// The arguments of one FC1 launch.
#[derive(Clone, Copy)]
pub struct GatherGemmArgs {
    /// `[tokens, hidden]` hidden states, row-major.
    pub hidden: TensorView,
    /// `[experts, 2 · intermediate, hidden]` expert weights.
    pub weights: TensorView,
    /// `[tokens, top]` i32 route ids — the source of the expanded route count.
    pub ids: TensorView,
    /// `[max_tiles]` i32 expert of each tile.
    pub tile_to_expert: TensorView,
    /// `[max_tiles]` i32 live rows of each tile.
    pub tile_to_limit: TensorView,
    /// `[max_permuted_rows]` i32 expanded route of each permuted row.
    pub permuted_to_expanded: TensorView,
    /// `[1]` i32 live tiles.
    pub num_non_exiting_tiles: TensorView,
    /// `[max_permuted_rows, intermediate]` gated products, written.
    pub out: TensorView,
}

/// The call structure the FC1 package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/gather_gemm/<backend>/include/uniserve_gather_gemm.h`.
#[repr(C)]
pub struct GatherCall {
    pub hidden: *const c_void,
    pub weights: *const c_void,
    pub out: *mut c_void,
    pub tile_to_expert: *const i32,
    pub tile_to_limit: *const i32,
    pub permuted_to_expanded: *const i32,
    pub num_non_exiting_tiles: *const i32,
    pub tokens: i64,
    pub permuted_rows: i64,
    pub width: i64,
    pub hidden_size: i64,
    pub experts: i64,
    pub top: i64,
    pub dtype: i64,
    pub error: [u8; 512],
}

/// The layout of [`GatherCall`].
pub const GATHER_CALL_LAYOUT: CallLayout = CallLayout {
    size: 624,
    hidden: 0,
    weights: 8,
    out: 16,
    tile_to_expert: 24,
    tile_to_limit: 32,
    permuted_to_expanded: 40,
    num_non_exiting_tiles: 48,
    tokens: 56,
    permuted_rows: 64,
    width: 72,
    hidden_size: 80,
    experts: 88,
    top: 96,
    dtype: 104,
    error: 112,
};

/// The layout of a grouped-GEMM call, for the headers' mirrors and their tests.
#[derive(Debug, PartialEq, Eq)]
pub struct CallLayout {
    pub size: usize,
    pub hidden: usize,
    pub weights: usize,
    pub out: usize,
    pub tile_to_expert: usize,
    pub tile_to_limit: usize,
    pub permuted_to_expanded: usize,
    pub num_non_exiting_tiles: usize,
    pub tokens: usize,
    pub permuted_rows: usize,
    pub width: usize,
    pub hidden_size: usize,
    pub experts: usize,
    pub top: usize,
    pub dtype: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<GatherCall>() == GATHER_CALL_LAYOUT.size,
    "the gather_gemm call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(GatherCall, error) == GATHER_CALL_LAYOUT.error,
    "the gather_gemm call structure's `error` moved without its header"
);

impl NativeCall for GatherCall {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// The first matrix of a routed expert.
pub struct GatherGemm;

impl Op for GatherGemm {
    const NAME: &'static str = "gather_gemm";

    type Descriptor = GatherGemmDescriptor;
    type Args<'a> = GatherGemmArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let tokens = args.hidden.rows();
        let permuted_rows = args.out.rows();

        args.hidden
            .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
        args.hidden
            .require_row(Self::NAME, descriptor.dtype, hidden)?;
        args.weights
            .require_shape(Self::NAME, Shape::new(&[experts, 2 * intermediate, hidden]))?;
        args.weights
            .require_row(Self::NAME, descriptor.dtype, hidden)?;
        args.ids
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        if args.ids.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 route ids, found {}", args.ids.dtype),
            ));
        }
        for view in [
            &args.tile_to_expert,
            &args.tile_to_limit,
            &args.permuted_to_expanded,
            &args.num_non_exiting_tiles,
        ] {
            if view.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected i32 sort metadata, found {}", view.dtype),
                ));
            }
            view.require_row(Self::NAME, DType::I32, 1)?;
        }
        // The sort pads its permuted rows up to whole tiles, so the map holds at least one row per
        // route and exactly the rows the product writes.
        let permuted = args.permuted_to_expanded.numel();
        if permuted < tokens * top {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{permuted} permuted rows cannot hold {} routes",
                    tokens * top
                ),
            ));
        }
        if permuted != args.out.rows() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the map names {permuted} rows and the output has {}",
                    args.out.rows()
                ),
            ));
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[permuted, intermediate]))?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;

        let mut call = GatherCall {
            hidden: address(args.hidden.ptr) as usize as *const c_void,
            weights: address(args.weights.ptr) as usize as *const c_void,
            out: address(args.out.ptr) as usize as *mut c_void,
            tile_to_expert: address(args.tile_to_expert.ptr) as usize as *const i32,
            tile_to_limit: address(args.tile_to_limit.ptr) as usize as *const i32,
            permuted_to_expanded: address(args.permuted_to_expanded.ptr) as usize as *const i32,
            num_non_exiting_tiles: address(args.num_non_exiting_tiles.ptr) as usize as *const i32,
            tokens: to_i64(Self::NAME, tokens)?,
            permuted_rows: to_i64(Self::NAME, permuted_rows)?,
            width: to_i64(Self::NAME, 2 * intermediate)?,
            hidden_size: to_i64(Self::NAME, hidden)?,
            experts: to_i64(Self::NAME, experts)?,
            top: to_i64(Self::NAME, top)?,
            dtype: dtype_code(descriptor.dtype),
            error: [0; 512],
        };
        // SAFETY: `GatherCall` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        GatherGemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.hidden);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(ACTIVATION_ENVELOPE * ACTIVATION_ENVELOPE * 2.0)) as f32;
        Tolerance::dot(reduction, accumulate, descriptor.dtype).with_atol(atol)
    }
}

/// The second matrix of a routed expert.
pub struct RouteGemm;

impl Op for RouteGemm {
    const NAME: &'static str = "route_gemm";

    type Descriptor = RouteGemmDescriptor;
    type Args<'a> = RouteGemmArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let rows = args.rows.rows();
        let tokens = args.ids.rows();
        let expanded = tokens * top;

        args.rows
            .require_shape(Self::NAME, Shape::new(&[rows, intermediate]))?;
        args.rows
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        args.weights
            .require_shape(Self::NAME, Shape::new(&[experts, hidden, intermediate]))?;
        args.weights
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        args.ids
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        if args.ids.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 route ids, found {}", args.ids.dtype),
            ));
        }
        for view in [
            &args.tile_to_expert,
            &args.tile_to_limit,
            &args.permuted_to_expanded,
            &args.num_non_exiting_tiles,
        ] {
            if view.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected i32 sort metadata, found {}", view.dtype),
                ));
            }
            view.require_row(Self::NAME, DType::I32, 1)?;
        }
        if rows != args.permuted_to_expanded.numel() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the rows are {} permuted rows and the map names {}",
                    rows,
                    args.permuted_to_expanded.numel()
                ),
            ));
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[expanded, hidden]))?;
        args.out.require_row(Self::NAME, descriptor.dtype, hidden)?;

        let mut call = RouteCall {
            rows: address(args.rows.ptr) as usize as *const c_void,
            weights: address(args.weights.ptr) as usize as *const c_void,
            out: address(args.out.ptr) as usize as *mut c_void,
            tile_to_expert: address(args.tile_to_expert.ptr) as usize as *const i32,
            tile_to_limit: address(args.tile_to_limit.ptr) as usize as *const i32,
            permuted_to_expanded: address(args.permuted_to_expanded.ptr) as usize as *const i32,
            num_non_exiting_tiles: address(args.num_non_exiting_tiles.ptr) as usize as *const i32,
            permuted_rows: to_i64(Self::NAME, rows)?,
            width: to_i64(Self::NAME, hidden)?,
            intermediate: to_i64(Self::NAME, intermediate)?,
            experts: to_i64(Self::NAME, experts)?,
            expanded_rows: to_i64(Self::NAME, expanded)?,
            dtype: dtype_code(descriptor.dtype),
            error: [0; 512],
        };
        // SAFETY: `RouteCall` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        RouteGemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.intermediate);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(ACTIVATION_ENVELOPE * ACTIVATION_ENVELOPE * 2.0)) as f32;
        Tolerance::dot(reduction, accumulate, descriptor.dtype).with_atol(atol)
    }
}

/// The arguments of one FC2 launch.
#[derive(Clone, Copy)]
pub struct RouteGemmArgs {
    /// `[max_permuted_rows, intermediate]` FC1's gated products.
    pub rows: TensorView,
    /// `[experts, hidden, intermediate]` expert weights.
    pub weights: TensorView,
    /// `[tokens, top]` i32 route ids.
    pub ids: TensorView,
    /// `[max_tiles]` i32 expert of each tile.
    pub tile_to_expert: TensorView,
    /// `[max_tiles]` i32 live rows of each tile.
    pub tile_to_limit: TensorView,
    /// `[max_permuted_rows]` i32 expanded route of each permuted row.
    pub permuted_to_expanded: TensorView,
    /// `[1]` i32 live tiles.
    pub num_non_exiting_tiles: TensorView,
    /// `[tokens · top, hidden]` every route's expert product, written unweighted.
    pub out: TensorView,
}

/// The call structure the FC2 package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/route_gemm/<backend>/include/uniserve_route_gemm.h`.
#[repr(C)]
pub struct RouteCall {
    pub rows: *const c_void,
    pub weights: *const c_void,
    pub out: *mut c_void,
    pub tile_to_expert: *const i32,
    pub tile_to_limit: *const i32,
    pub permuted_to_expanded: *const i32,
    pub num_non_exiting_tiles: *const i32,
    pub permuted_rows: i64,
    pub width: i64,
    pub intermediate: i64,
    pub experts: i64,
    pub expanded_rows: i64,
    pub dtype: i64,
    pub error: [u8; 512],
}

/// The layout of [`RouteCall`], for its header's mirror and its test.
pub const ROUTE_CALL_LAYOUT: RouteCallLayout = RouteCallLayout {
    size: 616,
    rows: 0,
    weights: 8,
    out: 16,
    tile_to_expert: 24,
    tile_to_limit: 32,
    permuted_to_expanded: 40,
    num_non_exiting_tiles: 48,
    permuted_rows: 56,
    width: 64,
    intermediate: 72,
    experts: 80,
    expanded_rows: 88,
    dtype: 96,
    error: 104,
};

/// The layout of [`RouteCall`], for the header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct RouteCallLayout {
    pub size: usize,
    pub rows: usize,
    pub weights: usize,
    pub out: usize,
    pub tile_to_expert: usize,
    pub tile_to_limit: usize,
    pub permuted_to_expanded: usize,
    pub num_non_exiting_tiles: usize,
    pub permuted_rows: usize,
    pub width: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub expanded_rows: usize,
    pub dtype: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<RouteCall>() == ROUTE_CALL_LAYOUT.size,
    "the route_gemm call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(RouteCall, error) == ROUTE_CALL_LAYOUT.error,
    "the route_gemm call structure's `error` moved without its header"
);

impl NativeCall for RouteCall {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}
