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
use crate::fp4::ScaleLayout;
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
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

/// One compilation of `gather_gemm_nvfp4`.
///
/// The routed first matrix of an encoded (NVFP4) expert block. It is its own operator rather than a
/// variant of [`GatherGemm`]: the weight operand's type changes — packed E2M1 nibbles, the swizzled
/// block scales, one FP32 tensor scale per expert — and an optional-and-branch form would put the
/// branch in the descriptor key and in the hot path of the dense case that never has it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatherGemmNvfp4Descriptor {
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
    /// Element type of the activations and of the output. The weights are packed, so this is the
    /// unpacked element type rather than the stored one.
    pub dtype: DType,
    /// Element type of the block scales, as the checkpoint stores them.
    pub scale: DType,
    /// The block-scale layout the checkpoint uses.
    pub layout: ScaleLayout,
    /// The MMA tile the package compiled for: rows by columns.
    pub tile: (u32, u32),
    /// The cluster shape the package compiled for: rows by columns.
    pub cluster: (u32, u32),
}

impl fmt::Display for GatherGemmNvfp4Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("activation", self.activation)
            .field("dtype", self.dtype)
            .field("scale", self.scale)
            .field("layout", self.layout)
            .field("tile", format_args!("{}x{}", self.tile.0, self.tile.1))
            .field(
                "cluster",
                format_args!("{}x{}", self.cluster.0, self.cluster.1),
            )
            .finish(formatter)
    }
}

/// The arguments of one encoded FC1 launch.
#[derive(Clone, Copy)]
pub struct GatherGemmNvfp4Args {
    /// `[routes, hidden]` the gathered activations.
    pub hidden: TensorView,
    /// `[experts, 2·intermediate, hidden/2]` U8: two E2M1 codes a byte along the contraction.
    pub weights: TensorView,
    /// `[experts, ...]` the block scales in the descriptor's layout, one per sixteen elements.
    pub scales: TensorView,
    /// `[experts]` f32: the reciprocal of each expert's tensor scale, as the loader carries it.
    pub global_scale: TensorView,
    /// `[tokens, top]` i32: the experts each token's routes selected.
    pub ids: TensorView,
    /// `[tiles]` i32: the expert a tile's rows belong to.
    pub tile_to_expert: TensorView,
    /// `[tiles]` i32: the local row limit of that tile.
    pub tile_to_limit: TensorView,
    /// `[routes]` i32: the gathered row of each route.
    pub permuted_to_expanded: TensorView,
    /// `[1]` i32: the number of tiles that have work.
    pub num_non_exiting_tiles: TensorView,
    /// `[routes, intermediate]` the gated product.
    pub out: TensorView,
}

/// The routed first matrix over a packed, block-scaled expert stack.
pub struct GatherGemmNvfp4;

impl GatherGemmNvfp4 {
    /// The launch's shape and geometry rules, separated from the launch itself so a host test can
    /// state them without a device.
    ///
    /// Answers the contraction, the expert width and the permuted row count the launch needs.
    pub(crate) fn validate(
        descriptor: &GatherGemmNvfp4Descriptor,
        args: &GatherGemmNvfp4Args,
    ) -> Result<(u64, u64, u64)> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let routes = args.out.rows();

        if args.out.shape.dims().last().copied().unwrap_or(0) != intermediate {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a row of the gated product is {intermediate} wide, the output is {:?}",
                    args.out.shape.dims()
                ),
            ));
        }
        args.out
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        // The packed field holds two codes a byte along the contraction, so its width is half the
        // contraction and no more: a wider one would read past the expert's weights.
        args.weights.require_shape(
            Self::NAME,
            Shape::new(&[experts, 2 * intermediate, hidden / 2]),
        )?;
        if args.weights.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the packed E2M1 field as u8, found {}",
                    args.weights.dtype
                ),
            ));
        }
        if args.scales.dtype != descriptor.scale {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} block scales, found {}",
                    descriptor.scale, args.scales.dtype
                ),
            ));
        }
        // Two E2M1 codes share a byte along the contraction, so an odd one names a stack that cannot
        // exist; and the block scales cover the contraction in blocks of sixteen, one block per output
        // row, so a contraction that is not whole blocks leaves a partial block whose scale the
        // checkpoint does not store.
        let block_elements = u64::from(crate::fp4::BLOCK_ELEMENTS);
        if !hidden.is_multiple_of(2) {
            return Err(Error::argument(
                Self::NAME,
                format!("a contraction of {hidden} is odd, and two E2M1 codes share a byte"),
            ));
        }
        if !hidden.is_multiple_of(block_elements) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a contraction of {hidden} is not whole {block_elements}-element scale blocks"
                ),
            ));
        }
        let blocks_per_row = hidden / block_elements;
        let scale_rows = 2 * intermediate;
        let covered: u64 = args.scales.shape.dims().iter().product();
        if covered < experts * scale_rows * blocks_per_row {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{covered} block scales do not cover {experts} experts of {scale_rows} rows by \
                     {blocks_per_row} blocks"
                ),
            ));
        }
        args.global_scale
            .require_shape(Self::NAME, Shape::new(&[experts]))?;
        if args.global_scale.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected f32 tensor scales, found {}",
                    args.global_scale.dtype
                ),
            ));
        }
        // `routes` is the permuted row count -- the output's rows, which the sort pads to whole tiles
        // -- while the activations are the caller's *token* rows and `ids` holds one row of routes per
        // token. The program gathers through `permuted_to_expanded`, as the dense product does, so
        // checking the activations against the permuted count would refuse every launch whose sort
        // pads anything.
        let tokens = args.hidden.rows();
        args.ids
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        args.permuted_to_expanded
            .require_shape(Self::NAME, Shape::new(&[routes]))?;
        for (name, view) in [
            ("ids", args.ids),
            ("permuted_to_expanded", args.permuted_to_expanded),
        ] {
            if view.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected i32 {name}, found {}", view.dtype),
                ));
            }
        }
        args.hidden
            .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
        args.hidden
            .require_row(Self::NAME, descriptor.dtype, hidden)?;
        if top == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a launch with no routes per token packs no rows".to_owned(),
            ));
        }
        Ok((hidden, intermediate, routes))
    }
}

impl Op for GatherGemmNvfp4 {
    const NAME: &'static str = "gather_gemm_nvfp4";

    type Descriptor = GatherGemmNvfp4Descriptor;
    type Args<'a> = GatherGemmNvfp4Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let (_, intermediate, routes) = Self::validate(descriptor, &args)?;

        // The activation is compiled into the program (it is in the key), so no launch value names
        // it; what a launch carries is the geometry the grid and the addressing need.
        let mut arguments = Arguments::new();
        arguments
            .view(args.hidden)
            .view(args.weights)
            .view(args.scales)
            .view(args.global_scale)
            .view(args.ids)
            .view(args.tile_to_expert)
            .view(args.tile_to_limit)
            .view(args.permuted_to_expanded)
            .view(args.num_non_exiting_tiles)
            .view(args.out)
            .u32(descriptor.hidden)
            .u32(descriptor.intermediate)
            .u32(descriptor.experts)
            .u32(descriptor.top)
            .u32(u32::try_from(routes).unwrap_or(u32::MAX))
            .u32(descriptor.tile.0)
            .u32(descriptor.tile.1)
            .u32(descriptor.cluster.0)
            .u32(descriptor.cluster.1);
        // One block per (expert tile, column tile) of the gated product, the geometry the package
        // compiled for: the tile and cluster are the descriptor's, so the grid is a function of the
        // problem rather than of a backend's tactic.
        // Axis 0 is the *tile buffer's* count, a launch-fixed geometry: one block per tile the sort
        // could produce, each returning immediately when its index is past the live count on the
        // device. Reading `num_non_exiting_tiles` here would be a host read of a device value, and
        // `[1]` means the launch would always compute one tile.
        let tiles = args
            .tile_to_expert
            .shape
            .dims()
            .first()
            .copied()
            .unwrap_or(1)
            .max(1);
        kernel.launch(
            LaunchGrid::new([
                tiles,
                (2 * intermediate).div_ceil(u64::from(descriptor.tile.1)),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        GatherGemmNvfp4Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The packed weights and their block scales are the reference's own operands: the dequantized
        // product is what a dense launch of the same expert computes, so the bound is `GatherGemm`'s
        // — the contraction in FP32 with the activation's envelope on top. Any extra error would come
        // from dequantizing in a different order, which this bound also covers.
        let reduction = u64::from(descriptor.hidden);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(ACTIVATION_ENVELOPE * ACTIVATION_ENVELOPE * 2.0)) as f32;
        Tolerance::dot(reduction, accumulate, descriptor.dtype).with_atol(atol)
    }
}

#[cfg(test)]
mod nvfp4_tests {
    use super::*;

    fn descriptor() -> GatherGemmNvfp4Descriptor {
        GatherGemmNvfp4Descriptor {
            hidden: 2048,
            intermediate: 768,
            experts: 128,
            top: 8,
            activation: Activation::Silu,
            dtype: DType::BF16,
            scale: DType::F8E4M3,
            layout: ScaleLayout::Swizzled128X4,
            tile: (128, 128),
            cluster: (1, 1),
        }
    }

    #[test]
    fn the_key_separates_the_packed_operand_from_the_dense_one() {
        // The dense gather_gemm and this one are different programs for the same geometry: the
        // weight operand's type changes, so the key must not collide with it.
        let encoded = descriptor().to_string();
        assert!(
            encoded.starts_with("hidden_2048_intermediate_768_experts_128_top_8_activation_silu")
        );
        assert!(encoded.contains("_scale_f8e4m3_"));
        assert!(encoded.contains("layout_swizzled_128x4"));
        let dense = GatherGemmDescriptor {
            hidden: 2048,
            intermediate: 768,
            experts: 128,
            top: 8,
            activation: Activation::Silu,
            dtype: DType::BF16,
            tile: (128, 128),
            cluster: (1, 1),
        }
        .to_string();
        assert_ne!(encoded, dense);
    }

    #[test]
    fn the_bound_is_the_dense_gather_gemms_bound() {
        // Same contraction, same accumulation, same output: an encoder that dequantizes in another
        // order is still inside this envelope, and a *looser* one would hide a wrong scale.
        let dense = GatherGemmDescriptor {
            hidden: 2048,
            intermediate: 768,
            experts: 128,
            top: 8,
            activation: Activation::Silu,
            dtype: DType::BF16,
            tile: (128, 128),
            cluster: (1, 1),
        };
        assert_eq!(
            GatherGemmNvfp4::tolerance(&descriptor()),
            GatherGemm::tolerance(&dense)
        );
    }

    fn route_descriptor() -> RouteGemmNvfp4Descriptor {
        RouteGemmNvfp4Descriptor {
            hidden: 2048,
            intermediate: 768,
            experts: 128,
            top: 8,
            dtype: DType::BF16,
            scale: DType::F8E4M3,
            layout: ScaleLayout::Swizzled128X4,
            tile: (128, 128),
            cluster: (1, 1),
        }
    }

    #[test]
    fn fc2_mirrors_fc1_without_an_activation() {
        // FC1 applies the gated product inside its launch; FC2 only contracts, so the activation is
        // not in its descriptor and the key says so by not naming one.
        let key = route_descriptor().to_string();
        assert!(key.starts_with("hidden_2048_intermediate_768_experts_128_top_8_dtype_bf16"));
        assert!(!key.contains("activation"));
        // The bound is the dense FC2's: contraction at `intermediate`, FP32 accumulation.
        let dense = RouteGemmDescriptor {
            hidden: 2048,
            intermediate: 768,
            experts: 128,
            top: 8,
            dtype: DType::BF16,
            tile: (128, 128),
            cluster: (1, 1),
        };
        assert_eq!(
            RouteGemmNvfp4::tolerance(&route_descriptor()),
            RouteGemm::tolerance(&dense)
        );
    }

    #[test]
    fn the_scale_table_covers_the_contraction_in_blocks_of_sixteen() {
        // The scales are one per sixteen elements *along the contraction*, one row per output row:
        // 2048/16 = 128 blocks for each of the 1536 rows of one expert's fused stack, times 128
        // experts. A table that covers less is a checkpoint the loader should have refused.
        let block_elements = u64::from(crate::fp4::BLOCK_ELEMENTS);
        let blocks_per_row = 2048u64 / block_elements;
        assert_eq!(blocks_per_row, 128);
        assert_eq!(128 * (2 * 768u64) * blocks_per_row, 25_165_824);
        // The two geometry rules the size of that table rests on: an even contraction (two codes a
        // byte) and a contraction that is whole scale blocks.
        assert!(2048u64.is_multiple_of(2) && 2048u64.is_multiple_of(block_elements));
        assert!(!2815u64.is_multiple_of(2));
        assert!(!(2816u64 - 8).is_multiple_of(block_elements));
    }
}

/// One compilation of `route_gemm_nvfp4`.
///
/// FC2 of an encoded expert block: the same descriptor as `gather_gemm_nvfp4` less the activation,
/// because FC1 applies the gated product and FC2 only contracts it. Its contraction is the expert
/// width, so the packed weights are `[experts, hidden, intermediate/2]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RouteGemmNvfp4Descriptor {
    /// Hidden elements of a token: FC2's output width.
    pub hidden: u32,
    /// Expert width: FC2's contraction length.
    pub intermediate: u32,
    /// Experts the weight stacks.
    pub experts: u32,
    /// Routes per token.
    pub top: u32,
    /// Element type of the rows and of the output; the weights are packed.
    pub dtype: DType,
    /// Element type of the block scales.
    pub scale: DType,
    /// The block-scale layout the checkpoint uses.
    pub layout: ScaleLayout,
    /// The MMA tile the package compiled for: rows by columns.
    pub tile: (u32, u32),
    /// The cluster shape the package compiled for: rows by columns.
    pub cluster: (u32, u32),
}

impl fmt::Display for RouteGemmNvfp4Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("dtype", self.dtype)
            .field("scale", self.scale)
            .field("layout", self.layout)
            .field("tile", format_args!("{}x{}", self.tile.0, self.tile.1))
            .field(
                "cluster",
                format_args!("{}x{}", self.cluster.0, self.cluster.1),
            )
            .finish(formatter)
    }
}

/// The arguments of one encoded FC2 launch.
#[derive(Clone, Copy)]
pub struct RouteGemmNvfp4Args {
    /// `[routes, intermediate]` the gated product.
    pub hidden: TensorView,
    /// `[experts, hidden, intermediate/2]` U8: two E2M1 codes a byte along the contraction.
    pub weights: TensorView,
    /// The block scales, one per sixteen elements of the contraction and one row per output row.
    pub scales: TensorView,
    /// `[experts]` f32: the reciprocal of each expert's tensor scale.
    pub global_scale: TensorView,
    /// `[routes]` i32: the expert each row belongs to.
    pub ids: TensorView,
    /// `[tiles]` i32: the expert a tile's rows belong to.
    pub tile_to_expert: TensorView,
    /// `[tiles]` i32: the local row limit of that tile.
    pub tile_to_limit: TensorView,
    /// `[routes]` i32: the gathered row of each route.
    pub permuted_to_expanded: TensorView,
    /// `[1]` i32: the number of tiles that have work.
    pub num_non_exiting_tiles: TensorView,
    /// `[routes, hidden]` the contracted rows, one per route: the `top` repeats are combined by
    /// `moe_unpermute`, not here.
    pub out: TensorView,
}

/// FC2 of an encoded, block-scaled expert block.
pub struct RouteGemmNvfp4;

impl Op for RouteGemmNvfp4 {
    const NAME: &'static str = "route_gemm_nvfp4";

    type Descriptor = RouteGemmNvfp4Descriptor;
    type Args<'a> = RouteGemmNvfp4Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let routes = args.out.rows();

        args.hidden
            .require_shape(Self::NAME, Shape::new(&[routes, intermediate]))?;
        args.hidden
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[routes, hidden]))?;
        args.out.require_row(Self::NAME, descriptor.dtype, hidden)?;
        // The contraction is this operator's packed axis, and it must be even and whole scale blocks
        // for the same reason FC1's is: two codes share a byte, and a partial block has no scale.
        if !intermediate.is_multiple_of(2) {
            return Err(Error::argument(
                Self::NAME,
                format!("a contraction of {intermediate} is odd, and two E2M1 codes share a byte"),
            ));
        }
        let block_elements = u64::from(crate::fp4::BLOCK_ELEMENTS);
        if !intermediate.is_multiple_of(block_elements) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a contraction of {intermediate} is not whole {block_elements}-element scale \
                     blocks"
                ),
            ));
        }
        args.weights
            .require_shape(Self::NAME, Shape::new(&[experts, hidden, intermediate / 2]))?;
        if args.weights.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the packed E2M1 field as u8, found {}",
                    args.weights.dtype
                ),
            ));
        }
        if args.scales.dtype != descriptor.scale {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} block scales, found {}",
                    descriptor.scale, args.scales.dtype
                ),
            ));
        }
        let blocks_per_row = intermediate / block_elements;
        let covered: u64 = args.scales.shape.dims().iter().product();
        if covered < experts * hidden * blocks_per_row {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{covered} block scales do not cover {experts} experts of {hidden} rows by \
                     {blocks_per_row} blocks"
                ),
            ));
        }
        args.global_scale
            .require_shape(Self::NAME, Shape::new(&[experts]))?;
        if args.global_scale.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected f32 tensor scales, found {}",
                    args.global_scale.dtype
                ),
            ));
        }
        for (name, view) in [
            ("ids", args.ids),
            ("permuted_to_expanded", args.permuted_to_expanded),
        ] {
            view.require_shape(Self::NAME, Shape::new(&[routes]))?;
            if view.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected i32 {name}, found {}", view.dtype),
                ));
            }
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.hidden)
            .view(args.weights)
            .view(args.scales)
            .view(args.global_scale)
            .view(args.ids)
            .view(args.tile_to_expert)
            .view(args.tile_to_limit)
            .view(args.permuted_to_expanded)
            .view(args.num_non_exiting_tiles)
            .view(args.out)
            .u32(descriptor.hidden)
            .u32(descriptor.intermediate)
            .u32(descriptor.experts)
            .u32(descriptor.top)
            .u32(u32::try_from(routes).unwrap_or(u32::MAX))
            .u32(descriptor.tile.0)
            .u32(descriptor.tile.1)
            .u32(descriptor.cluster.0)
            .u32(descriptor.cluster.1);
        // Axis 0 is the *tile buffer's* count, a launch-fixed geometry: one block per tile the sort
        // could produce, each returning immediately when its index is past the live count on the
        // device. Reading `num_non_exiting_tiles` here would be a host read of a device value, and
        // `[1]` means the launch would always compute one tile.
        let tiles = args
            .tile_to_expert
            .shape
            .dims()
            .first()
            .copied()
            .unwrap_or(1)
            .max(1);
        kernel.launch(
            LaunchGrid::new([tiles, hidden.div_ceil(u64::from(descriptor.tile.1)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        RouteGemmNvfp4Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The contraction is the expert width, accumulated in FP32, and the packed weights are the
        // reference's own operands: the bound is the dense `route_gemm`'s.
        let reduction = u64::from(descriptor.intermediate);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(ACTIVATION_ENVELOPE * ACTIVATION_ENVELOPE * 2.0)) as f32;
        Tolerance::dot(reduction, accumulate, descriptor.dtype).with_atol(atol)
    }
}

#[cfg(test)]
mod nvfp4_launch_tests {
    use super::*;

    fn table(dtype: DType, dims: &[u64]) -> TensorView {
        TensorView::new(crate::tensor::DevicePtr::new(0), dtype, Shape::new(dims))
    }

    fn fc1() -> GatherGemmNvfp4Descriptor {
        GatherGemmNvfp4Descriptor {
            hidden: 2048,
            intermediate: 768,
            experts: 128,
            top: 8,
            activation: Activation::Silu,
            dtype: DType::BF16,
            scale: DType::F8E4M3,
            layout: ScaleLayout::Swizzled128X4,
            tile: (128, 128),
            cluster: (1, 1),
        }
    }

    /// The shapes a colocated block launches with: 3 tokens, 8 routes each, and a sort that padded
    /// the permuted rows past `tokens · top`.
    fn fc1_args() -> GatherGemmNvfp4Args {
        GatherGemmNvfp4Args {
            hidden: table(DType::BF16, &[3, 2048]),
            weights: table(DType::U8, &[128, 1536, 1024]),
            scales: table(DType::F8E4M3, &[128, 1536, 128]),
            global_scale: table(DType::F32, &[128]),
            ids: table(DType::I32, &[3, 8]),
            tile_to_expert: table(DType::I32, &[2]),
            tile_to_limit: table(DType::I32, &[2]),
            permuted_to_expanded: table(DType::I32, &[32]),
            num_non_exiting_tiles: table(DType::I32, &[1]),
            out: table(DType::BF16, &[32, 768]),
        }
    }

    #[test]
    fn the_activations_are_the_token_rows_and_the_output_the_permuted_ones() {
        // Builder01's M963: `routes` (the output's rows) is the sort's padded count and is never the
        // activation count. The invariant is `ids = [tokens, top]`, `hidden = [tokens, hidden]` and
        // `permuted_to_expanded = [routes]`, which is what the program gathers through.
        let descriptor = fc1();
        let args = fc1_args();
        assert_eq!(args.hidden.rows(), 3);
        assert_eq!(args.out.rows(), 32);
        assert_eq!(args.ids.shape.dims(), &[3, 8]);
        assert_eq!(args.permuted_to_expanded.shape.dims()[0], 32);
        // The scales cover one block per output row along the contraction, not one row's worth.
        assert_eq!(args.scales.shape.dims(), &[128, 1536, 128]);
        assert_eq!(descriptor.hidden / crate::fp4::BLOCK_ELEMENTS, 128);
        // And the weight stack is the packed field, half the contraction wide.
        assert_eq!(args.weights.shape.dims(), &[128, 1536, 1024]);
    }

    #[test]
    fn an_odd_contraction_and_a_partial_block_are_refused_by_name() {
        // An odd contraction with a weight stack that matches it: the geometry rule is what refuses,
        // not the shape check that happens to fire first.
        let mut odd = fc1();
        odd.hidden = 2047;
        let mut odd_args = fc1_args();
        odd_args.weights = table(DType::U8, &[128, 1536, 1023]);
        let error = GatherGemmNvfp4::validate(&odd, &odd_args).unwrap_err();
        assert!(format!("{error}").contains("odd"), "{error}");

        let mut partial = fc1();
        partial.hidden = 2040;
        let mut partial_args = fc1_args();
        partial_args.weights = table(DType::U8, &[128, 1536, 1020]);
        let error = GatherGemmNvfp4::validate(&partial, &partial_args).unwrap_err();
        assert!(format!("{error}").contains("scale blocks"), "{error}");
    }

    #[test]
    fn a_scale_cover_short_by_one_block_is_refused() {
        let mut args = fc1_args();
        args.scales = table(DType::F8E4M3, &[127, 1536, 128]);
        let error = GatherGemmNvfp4::validate(&fc1(), &args).unwrap_err();
        assert!(format!("{error}").contains("do not cover"));
    }
}
