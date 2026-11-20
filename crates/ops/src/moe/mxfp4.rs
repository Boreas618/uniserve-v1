//! The routed products of an MXFP4 expert block: MXFP8 activations times MXFP4 expert weights.
//!
//! ```text
//! gather_gemm_mxfp4   MXFP8 hidden rows, MXFP4 [up; gate] stack -> the gated product, MXFP8
//!                     out[p] = mxfp8(silu(min(gate, L)) · clamp(up, ±L))     (L: the clamp)
//! route_gemm_mxfp4    MXFP8 gated rows, MXFP4 down stack         -> down · gated, BF16
//! ```
//!
//! Both are trtllm-gen's routed batched products (the `trtllm-gen-bmm` artifact's MxE4m3 ×
//! MxE2m1 kernels), launched through FlashInfer's runners as the NVFP4 forms are
//! ([`super::grouped`]), over the same sort and the same tiles. The first product gathers each
//! permuted row's token itself, fuses the gated SwiGLU with its clamp into its epilogue, and
//! encodes its output into MXFP8 there: per 32 elements of a row, an E4M3 code under a UE8M0
//! scale chosen by the OCP rule ([`crate::mx::ScaleRule::Ocp`]), the scales in the 128 × 4 atom
//! the second product reads. The second writes BF16, one
//! row per route, unweighted; the combine weighs and sums them.
//!
//! The weights are the products' arrangement of the checkpoint's: each matrix's E2M1 codes two a
//! byte along the contraction with its rows shuffled ([`GatherGemmMxfp4Descriptor::WEIGHT_VALUES`],
//! [`RouteGemmMxfp4Descriptor::WEIGHT_VALUES`]) and its UE8M0 scales, one per 32 elements, moved
//! with their rows and swizzled ([`WEIGHT_SCALE_LAYOUT`]).
//!
//! The artifact's MxE4m3 × MxE2m1 kernels of 128-token tiles are two-block clusters
//! ([`CLUSTER`]); the descriptors state the tile and the cluster the module's cubins are, as the
//! NVFP4 descriptors do.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::fp4::ValueLayout;
use crate::kernel::{Kernel, NativeCall};
use crate::mx::{BLOCK, ScaleLayout};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

use super::grouped::{check_workspace, packed_workspace};
use super::{TILE, address, to_i64};

/// The thread-block cluster the MXFP4 products are compiled for, rows by columns.
pub const CLUSTER: (u32, u32) = (2, 1);

/// The arrangement of an MXFP4 stack's scales both products read: each matrix's
/// `[rows, columns / 32]` table, rows moved with the codes' rows, in the 128 × 4 atom.
pub const WEIGHT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// The arrangement of the gated rows' scales between the two products: what the first one's
/// epilogue writes and the second one reads.
pub const GATED_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// The arrangement of the hidden rows' scales the first product gathers: the rows' own table.
pub const TOKEN_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Linear;

/// One compilation of `gather_gemm_mxfp4`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatherGemmMxfp4Descriptor {
    /// Hidden elements of a token: the contraction length.
    pub hidden: u32,
    /// Expert width: one half of the gated stack's rows, and the product's output width.
    pub intermediate: u32,
    /// Experts the weight stacks.
    pub experts: u32,
    /// Routes per token.
    pub top: u32,
    /// The MMA tile the package compiled for: rows by columns.
    pub tile: (u32, u32),
    /// The cluster shape the package compiled for: rows by columns.
    pub cluster: (u32, u32),
}

impl GatherGemmMxfp4Descriptor {
    /// The arrangement of the `[up; gate]` stack's packed rows the gated product reads: the two
    /// halves interleaved row by row (up first), then shuffled inside the blocks of one tile.
    pub const WEIGHT_VALUES: ValueLayout = ValueLayout::InterleavedShuffled {
        epilogue_tile_m: TILE.0,
    };

    /// The compilation of the gated product of `top` routes over `experts` MXFP4 experts of
    /// `intermediate` width from `hidden`-wide tokens, at the products' tile and cluster.
    pub const fn new(hidden: u32, intermediate: u32, experts: u32, top: u32) -> Self {
        Self {
            hidden,
            intermediate,
            experts,
            top,
            tile: TILE,
            cluster: CLUSTER,
        }
    }

    /// The bytes of workspace a launch of `routes` permuted rows takes: the package's sizing over
    /// the gated product's `2 · intermediate` output columns.
    pub fn workspace_bytes(&self, routes: u64) -> u64 {
        packed_workspace(
            routes,
            2 * u64::from(self.intermediate),
            self.tile,
            self.cluster,
        )
    }
}

/// The key a routed MXFP4 product's descriptor prints.
fn key(
    formatter: &mut fmt::Formatter<'_>,
    widths: (u32, u32, u32, u32),
    tile: (u32, u32),
    cluster: (u32, u32),
) -> fmt::Result {
    let (hidden, intermediate, experts, top) = widths;
    Key::new()
        .field("hidden", hidden)
        .field("intermediate", intermediate)
        .field("experts", experts)
        .field("top", top)
        .field("tile", format_args!("{}x{}", tile.0, tile.1))
        .field("cluster", format_args!("{}x{}", cluster.0, cluster.1))
        .finish(formatter)
}

impl fmt::Display for GatherGemmMxfp4Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        key(
            formatter,
            (self.hidden, self.intermediate, self.experts, self.top),
            self.tile,
            self.cluster,
        )
    }
}

/// The arguments of one `gather_gemm_mxfp4` launch.
#[derive(Clone)]
pub struct GatherGemmMxfp4Args {
    /// `[tokens, hidden]` E4M3: the hidden rows' MXFP8 codes, token-major (the product gathers).
    pub hidden: Operand,
    /// `[tokens, hidden / 32]` U8: their UE8M0 scales, in [`TOKEN_SCALE_LAYOUT`].
    pub hidden_scale: Operand,
    /// `[experts, 2 · intermediate, hidden / 2]` U8: two E2M1 codes a byte, rows in
    /// [`GatherGemmMxfp4Descriptor::WEIGHT_VALUES`].
    pub weights: Operand,
    /// `[experts, ...]` U8: each expert's UE8M0 scales in [`WEIGHT_SCALE_LAYOUT`].
    pub scales: Operand,
    /// `[experts]` F32: each expert's clamp `L` of the gated product.
    pub limit: Operand,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: Operand,
    /// `[tiles]` u32: the permuted row one past each tile's last live row.
    pub tile_to_limit: Operand,
    /// `[routes]` u32: the token row each permuted row gathers.
    pub permuted_to_token: Operand,
    /// `[1]` u32: the tiles that hold routes.
    pub num_non_exiting_tiles: Operand,
    /// `[routes, intermediate]` E4M3: the gated rows' codes.
    pub out: Operand,
    /// The gated rows' UE8M0 scales over `[routes, intermediate]` in [`GATED_SCALE_LAYOUT`].
    pub out_scale: Operand,
    /// `[bytes]` U8: the program's scratch, sized by the caller.
    pub workspace: Operand,
}

/// One MXFP4 FC1 launch, as the trtllm-gen batched-GEMM entry reads it, declared to the packages
/// as `uniserve_gather_gemm_mxfp4_call` (`uniserve_gather_gemm_mxfp4.h`, generated with its layout
/// asserted: [`abi`]).
#[repr(C)]
#[derive(abi::Layout)]
pub struct GatherMxfp4Call {
    /// `[tokens, hidden]` E4M3: the hidden rows' codes, token-major.
    pub hidden: *const c_void,
    /// `[tokens, hidden / 32]` U8: their scales, linear.
    pub hidden_scale: *const c_void,
    /// `[experts, 2·intermediate, hidden/2]` U8: two E2M1 codes a byte.
    pub weights: *const c_void,
    /// The weights' scales, one per 32 elements of the contraction, swizzled.
    pub scales: *const c_void,
    /// `[experts]` f32: the gated product's clamp.
    pub limit: *const c_void,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: *const i32,
    /// `[tiles]` u32: the local row limit of that tile.
    pub tile_to_limit: *const i32,
    /// `[routes]` u32: the token row each permuted row gathers.
    pub permuted_to_token: *const i32,
    /// `[1]` u32: the number of tiles that have work.
    pub num_non_exiting_tiles: *const i32,
    /// `[routes, intermediate]` E4M3: the gated rows' codes.
    pub out: *mut c_void,
    /// The gated rows' scales, swizzled.
    pub out_scale: *mut c_void,
    /// The scratch the program needs, sized by the caller; the entry refuses a short one.
    pub workspace: *mut c_void,
    /// Bytes of `workspace`.
    pub workspace_bytes: i64,
    /// Rows of the activations: the token count.
    pub tokens: i64,
    /// Rows of the gated product: the permuted row count, padded to whole tiles.
    pub routes: i64,
    /// The contraction length.
    pub hidden_size: i64,
    /// The expert width of one half of the gated product.
    pub intermediate: i64,
    /// Experts the weight stack holds.
    pub experts: i64,
    /// Routes per token.
    pub top: i64,
    /// The compiled MMA tile, rows by columns.
    pub tile_m: i64,
    pub tile_n: i64,
    /// The compiled cluster, rows by columns.
    pub cluster_m: i64,
    pub cluster_n: i64,
    /// A NUL-terminated refusal, when the entry returns nonzero.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for GatherMxfp4Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// Checks `operand` is a contiguous `dims` of `dtype`, naming `what`.
fn require(
    operator: &'static str,
    what: &str,
    operand: &Operand,
    dims: &[u64],
    dtype: DType,
) -> Result<()> {
    operand.require_shape(operator, dims)?;
    if operand.dtype != dtype || !operand.is_row_major() {
        return Err(Error::argument(
            operator,
            format!(
                "the {what} are contiguous {}, not {} {:?}",
                abi::name(dtype),
                abi::name(operand.dtype),
                operand.dims()
            ),
        ));
    }
    Ok(())
}

/// Checks a routed product's tile maps: `[tiles]` u32 twice, and the live tile count.
fn require_tiles(operator: &'static str, maps: [&Operand; 3]) -> Result<()> {
    let [tile_to_expert, tile_to_limit, live] = maps;
    let tiles = tile_to_expert.numel();
    for (what, operand, count) in [
        ("tile experts", tile_to_expert, tiles),
        ("tile limits", tile_to_limit, tiles),
        ("live tile count", live, 1),
    ] {
        require(operator, what, operand, &[count], DType::U32)?;
    }
    Ok(())
}

/// The extents of an MXFP4 stack's scales: `experts` tables of `rows` by `columns` elements.
fn stack_scales(experts: u64, rows: u64, columns: u64) -> [u64; 3] {
    let [padded_rows, padded_blocks] = WEIGHT_SCALE_LAYOUT.shape(rows, columns);
    [experts, padded_rows, padded_blocks]
}

/// Refuses widths a routed MXFP4 product cannot compute: whole scale blocks of 32 elements on both
/// contractions, whole 128-row tiles of the weights' rows.
fn require_widths(operator: &'static str, hidden: u64, intermediate: u64) -> Result<()> {
    let block = u64::from(BLOCK);
    if hidden == 0
        || intermediate == 0
        || !hidden.is_multiple_of(block)
        || !intermediate.is_multiple_of(block)
        || !(2 * intermediate).is_multiple_of(u64::from(TILE.0))
        || !hidden.is_multiple_of(u64::from(TILE.0))
    {
        return Err(Error::argument(
            operator,
            format!(
                "hidden {hidden} and intermediate {intermediate}: both contractions are whole \
                 {block}-element scale blocks and the weights' rows whole {}-row tiles",
                TILE.0
            ),
        ));
    }
    Ok(())
}

/// The gated MXFP4 product: the routed FC1 of an MXFP4 expert block; see the module
/// documentation.
pub struct GatherGemmMxfp4;

impl Op for GatherGemmMxfp4 {
    const NAME: &'static str = proto::module::operator::GATHER_GEMM_MXFP4;

    type Descriptor = GatherGemmMxfp4Descriptor;
    type Args<'a> = GatherGemmMxfp4Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        require_widths(Self::NAME, hidden, intermediate)?;
        if descriptor.top == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a launch with no routes per token packs no rows".to_owned(),
            ));
        }

        let tokens = args.hidden.rows();
        let routes = args.out.rows();
        require(
            Self::NAME,
            "hidden codes",
            &args.hidden,
            &[tokens, hidden],
            DType::F8E4M3,
        )?;
        let token_scales = TOKEN_SCALE_LAYOUT.shape(tokens, hidden);
        require(
            Self::NAME,
            "hidden scales",
            &args.hidden_scale,
            &token_scales,
            DType::U8,
        )?;
        require(
            Self::NAME,
            "weights",
            &args.weights,
            &[experts, 2 * intermediate, hidden / 2],
            DType::U8,
        )?;
        require(
            Self::NAME,
            "weight scales",
            &args.scales,
            &stack_scales(experts, 2 * intermediate, hidden),
            DType::U8,
        )?;
        require(
            Self::NAME,
            "clamp limits",
            &args.limit,
            &[experts],
            DType::F32,
        )?;
        require_tiles(
            Self::NAME,
            [
                &args.tile_to_expert,
                &args.tile_to_limit,
                &args.num_non_exiting_tiles,
            ],
        )?;
        require(
            Self::NAME,
            "permuted tokens",
            &args.permuted_to_token,
            &[routes],
            DType::U32,
        )?;
        require(
            Self::NAME,
            "gated codes",
            &args.out,
            &[routes, intermediate],
            DType::F8E4M3,
        )?;
        let gated_scales = GATED_SCALE_LAYOUT.shape(routes, intermediate);
        require(
            Self::NAME,
            "gated scales",
            &args.out_scale,
            &gated_scales,
            DType::U8,
        )?;
        check_workspace(Self::NAME, &args.workspace)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let pointer = |operand: &Operand| address(operand.ptr) as usize as *const c_void;
        let table = |operand: &Operand| address(operand.ptr) as usize as *const i32;

        let mut call = GatherMxfp4Call {
            hidden: pointer(&args.hidden),
            hidden_scale: pointer(&args.hidden_scale),
            weights: pointer(&args.weights),
            scales: pointer(&args.scales),
            limit: pointer(&args.limit),
            tile_to_expert: table(&args.tile_to_expert),
            tile_to_limit: table(&args.tile_to_limit),
            permuted_to_token: table(&args.permuted_to_token),
            num_non_exiting_tiles: table(&args.num_non_exiting_tiles),
            out: address(args.out.ptr) as usize as *mut c_void,
            out_scale: address(args.out_scale.ptr) as usize as *mut c_void,
            workspace: address(args.workspace.ptr) as usize as *mut c_void,
            workspace_bytes: to_i64(Self::NAME, args.workspace.numel())?,
            tokens: to_i64(Self::NAME, args.hidden.rows())?,
            routes: to_i64(Self::NAME, args.out.rows())?,
            hidden_size: i64::from(descriptor.hidden),
            intermediate: i64::from(descriptor.intermediate),
            experts: i64::from(descriptor.experts),
            top: i64::from(descriptor.top),
            tile_m: i64::from(descriptor.tile.0),
            tile_n: i64::from(descriptor.tile.1),
            cluster_m: i64::from(descriptor.cluster.0),
            cluster_n: i64::from(descriptor.cluster.1),
            error: [0; MESSAGE],
        };
        // SAFETY: `GatherMxfp4Call` is the structure this package's entry declares: the
        // declaration is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        GatherGemmMxfp4Args {
            out: twins.shadow_of(args.out),
            out_scale: twins.shadow_of(args.out_scale),
            ..args
        }
    }

    /// The contraction's bound and the encoding of the result.
    ///
    /// The codes' products under power-of-two scales are exact in FP32, so two implementations
    /// differ first by the association of the `hidden` products of each half; the gated product
    /// is then encoded into E4M3 under a scale each 32 elements of a row measure, which can put
    /// two implementations' sums on either side of a rounding boundary: a code one step apart,
    /// `2⁻³` of the value at most. Both are relative to the output's magnitude, which the
    /// activation envelope ([`tolerance::VALUE_ENVELOPE`]) bounds the gate and the up halves by.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.hidden);
        let accumulate = DType::F32;
        let gamma = tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(tolerance::VALUE_ENVELOPE * tolerance::VALUE_ENVELOPE * 2.0))
                as f32;
        let dot = Tolerance::dot(reduction, accumulate, DType::F32);
        Tolerance::new(atol, dot.rtol + 2.0_f32.powi(-3))
    }
}

/// One compilation of `route_gemm_mxfp4`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RouteGemmMxfp4Descriptor {
    /// The output width.
    pub hidden: u32,
    /// The contraction: an expert's width.
    pub intermediate: u32,
    /// Experts the weight stacks.
    pub experts: u32,
    /// Routes per token.
    pub top: u32,
    /// The MMA tile the package compiled for: rows by columns.
    pub tile: (u32, u32),
    /// The cluster shape the package compiled for: rows by columns.
    pub cluster: (u32, u32),
}

impl RouteGemmMxfp4Descriptor {
    /// The arrangement of the down stack's packed rows the product reads: shuffled inside the
    /// blocks of one tile.
    pub const WEIGHT_VALUES: ValueLayout = ValueLayout::Shuffled {
        epilogue_tile_m: TILE.0,
    };

    /// The compilation of the second product of `top` routes over `experts` MXFP4 experts of
    /// `intermediate` width into `hidden`-wide BF16 rows, at the products' tile and cluster.
    pub const fn new(hidden: u32, intermediate: u32, experts: u32, top: u32) -> Self {
        Self {
            hidden,
            intermediate,
            experts,
            top,
            tile: TILE,
            cluster: CLUSTER,
        }
    }

    /// The bytes of workspace a launch of `routes` permuted rows takes.
    pub fn workspace_bytes(&self, routes: u64) -> u64 {
        packed_workspace(routes, u64::from(self.hidden), self.tile, self.cluster)
    }
}

impl fmt::Display for RouteGemmMxfp4Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        key(
            formatter,
            (self.hidden, self.intermediate, self.experts, self.top),
            self.tile,
            self.cluster,
        )
    }
}

/// The arguments of one `route_gemm_mxfp4` launch.
#[derive(Clone)]
pub struct RouteGemmMxfp4Args {
    /// `[routes, intermediate]` E4M3: the gated rows' codes the first product wrote.
    pub hidden: Operand,
    /// Their UE8M0 scales over `[routes, intermediate]` in [`GATED_SCALE_LAYOUT`].
    pub hidden_scale: Operand,
    /// `[experts, hidden, intermediate / 2]` U8: two E2M1 codes a byte, rows in
    /// [`RouteGemmMxfp4Descriptor::WEIGHT_VALUES`].
    pub weights: Operand,
    /// `[experts, ...]` U8: each expert's UE8M0 scales in [`WEIGHT_SCALE_LAYOUT`].
    pub scales: Operand,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: Operand,
    /// `[tiles]` u32: the permuted row one past each tile's last live row.
    pub tile_to_limit: Operand,
    /// `[routes]` u32: the route of each permuted row.
    pub permuted_to_expanded: Operand,
    /// `[1]` u32: the tiles that hold routes.
    pub num_non_exiting_tiles: Operand,
    /// `[routes, hidden]` BF16: each route's product, unweighted.
    pub out: Operand,
    /// `[bytes]` U8: the program's scratch, sized by the caller.
    pub workspace: Operand,
}

/// One MXFP4 FC2 launch, as the trtllm-gen batched-GEMM entry reads it, declared to the packages
/// as `uniserve_route_gemm_mxfp4_call` (`uniserve_route_gemm_mxfp4.h`, generated with its layout
/// asserted: [`abi`]).
#[repr(C)]
#[derive(abi::Layout)]
pub struct RouteMxfp4Call {
    /// `[routes, intermediate]` E4M3: the gated rows' codes.
    pub hidden: *const c_void,
    /// The gated rows' scales, swizzled.
    pub hidden_scale: *const c_void,
    /// `[experts, hidden, intermediate/2]` U8: two E2M1 codes a byte.
    pub weights: *const c_void,
    /// The weights' scales, one per 32 elements of the contraction, swizzled.
    pub scales: *const c_void,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: *const i32,
    /// `[tiles]` u32: the local row limit of that tile.
    pub tile_to_limit: *const i32,
    /// `[routes]` u32: the route of each permuted row.
    pub permuted_to_expanded: *const i32,
    /// `[1]` u32: the number of tiles that have work.
    pub num_non_exiting_tiles: *const i32,
    /// `[routes, hidden]` BF16: the contracted rows.
    pub out: *mut c_void,
    /// The scratch the program needs, sized by the caller; the entry refuses a short one.
    pub workspace: *mut c_void,
    /// Bytes of `workspace`.
    pub workspace_bytes: i64,
    /// Rows of the gated product and of the output: the padded route count.
    pub routes: i64,
    /// The output width.
    pub hidden_size: i64,
    /// The contraction length: one expert's intermediate width.
    pub intermediate: i64,
    /// Experts the weight stack holds.
    pub experts: i64,
    /// Routes per token.
    pub top: i64,
    /// The compiled MMA tile, rows by columns.
    pub tile_m: i64,
    pub tile_n: i64,
    /// The compiled cluster, rows by columns.
    pub cluster_m: i64,
    pub cluster_n: i64,
    /// A NUL-terminated refusal, when the entry returns nonzero.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for RouteMxfp4Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// The second MXFP4 product: the routed FC2 of an MXFP4 expert block; see the module
/// documentation.
pub struct RouteGemmMxfp4;

impl Op for RouteGemmMxfp4 {
    const NAME: &'static str = proto::module::operator::ROUTE_GEMM_MXFP4;

    type Descriptor = RouteGemmMxfp4Descriptor;
    type Args<'a> = RouteGemmMxfp4Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        require_widths(Self::NAME, hidden, intermediate)?;
        if descriptor.top == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a launch with no routes per token packs no rows".to_owned(),
            ));
        }

        let routes = args.out.rows();
        require(
            Self::NAME,
            "gated codes",
            &args.hidden,
            &[routes, intermediate],
            DType::F8E4M3,
        )?;
        let gated_scales = GATED_SCALE_LAYOUT.shape(routes, intermediate);
        require(
            Self::NAME,
            "gated scales",
            &args.hidden_scale,
            &gated_scales,
            DType::U8,
        )?;
        require(
            Self::NAME,
            "weights",
            &args.weights,
            &[experts, hidden, intermediate / 2],
            DType::U8,
        )?;
        require(
            Self::NAME,
            "weight scales",
            &args.scales,
            &stack_scales(experts, hidden, intermediate),
            DType::U8,
        )?;
        require_tiles(
            Self::NAME,
            [
                &args.tile_to_expert,
                &args.tile_to_limit,
                &args.num_non_exiting_tiles,
            ],
        )?;
        require(
            Self::NAME,
            "permuted routes",
            &args.permuted_to_expanded,
            &[routes],
            DType::U32,
        )?;
        require(
            Self::NAME,
            "products",
            &args.out,
            &[routes, hidden],
            DType::BF16,
        )?;
        check_workspace(Self::NAME, &args.workspace)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let pointer = |operand: &Operand| address(operand.ptr) as usize as *const c_void;
        let table = |operand: &Operand| address(operand.ptr) as usize as *const i32;

        let mut call = RouteMxfp4Call {
            hidden: pointer(&args.hidden),
            hidden_scale: pointer(&args.hidden_scale),
            weights: pointer(&args.weights),
            scales: pointer(&args.scales),
            tile_to_expert: table(&args.tile_to_expert),
            tile_to_limit: table(&args.tile_to_limit),
            permuted_to_expanded: table(&args.permuted_to_expanded),
            num_non_exiting_tiles: table(&args.num_non_exiting_tiles),
            out: address(args.out.ptr) as usize as *mut c_void,
            workspace: address(args.workspace.ptr) as usize as *mut c_void,
            workspace_bytes: to_i64(Self::NAME, args.workspace.numel())?,
            routes: to_i64(Self::NAME, args.out.rows())?,
            hidden_size: i64::from(descriptor.hidden),
            intermediate: i64::from(descriptor.intermediate),
            experts: i64::from(descriptor.experts),
            top: i64::from(descriptor.top),
            tile_m: i64::from(descriptor.tile.0),
            tile_n: i64::from(descriptor.tile.1),
            cluster_m: i64::from(descriptor.cluster.0),
            cluster_n: i64::from(descriptor.cluster.1),
            error: [0; MESSAGE],
        };
        // SAFETY: `RouteMxfp4Call` is the structure this package's entry declares: the
        // declaration is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        RouteGemmMxfp4Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// The contraction's bound over exact products of codes under power-of-two scales, rounded
    /// once into BF16, its absolute term the gated envelope's.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.intermediate);
        let accumulate = DType::F32;
        let gamma = tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(tolerance::VALUE_ENVELOPE * tolerance::VALUE_ENVELOPE * 2.0))
                as f32;
        Tolerance::dot(reduction, accumulate, DType::BF16).with_atol(atol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek-V4.1-Flash's routed experts: the keys state the widths, the tile and the cluster,
    /// and the stacks' scale tables are whole 128 × 4 atoms of 32-element blocks.
    #[test]
    fn the_keys_and_tables_are_deepseeks_experts() {
        let gated = GatherGemmMxfp4Descriptor::new(5120, 2304, 384, 6);
        assert_eq!(
            gated.to_string(),
            "hidden_5120_intermediate_2304_experts_384_top_6_tile_128x128_cluster_2x1"
        );
        let down = RouteGemmMxfp4Descriptor::new(5120, 2304, 384, 6);
        assert_eq!(down.to_string(), gated.to_string());

        assert_eq!(stack_scales(384, 2 * 2304, 5120), [384, 4608, 160]);
        // 2304 / 32 = 72 blocks of the down contraction, already whole groups of four.
        assert_eq!(stack_scales(384, 5120, 2304), [384, 5120, 72]);
        assert!(require_widths("test", 5120, 2304).is_ok());
        assert!(require_widths("test", 5120, 2300).is_err());
    }
}
