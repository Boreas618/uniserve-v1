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
//!                     out[M_perm, H], row p the product of permuted row p, unweighted
//!                     (route k of token t at row expanded_to_permuted[t · top + k])
//! ```
//!
//! Each expert's FC1 weight holds its `I` up rows followed by its `I` gate rows (Hugging Face stores
//! gate then up; the stacked `up_gate` stores up then gate), and the gated
//! product is `act(gate) · up` with the activation the descriptor names.
//!
//! Both programs are CuTe DSL native libraries — tiles, cluster shapes and the persistent scheduler
//! are the package's, and its tuning data (which tactic for which token count) is the package's too
//! — so [`Op::launch`] computes no grid. The persistent grid is bounded by the multiprocessors of
//! the device the launch runs on, a launch value rather than a compiled one (the device code is the
//! same on every part of a target), so the call states it: [`Op::launch`] reads it from the
//! stream's device (`platform::DeviceId::multiprocessors`, a driver attribute that touches neither
//! the stream nor a context) and the entry queries nothing, which keeps it safe to call while the
//! stream captures.
//!
//! # The bound
//!
//! A row of FC1 or FC2 is a dot product of length `H` or `I` accumulated in FP32 and rounded once to
//! BF16. Two correct implementations therefore differ by at most `2γₙ Σ|aᵢbᵢ| + u_out|result|`
//! (Higham §3.1); with a sum of mixed-sign products the result can cancel, so
//! [`Tolerance::dot`] states the bound relative to the result and the activation envelope
//! ([`tolerance::VALUE_ENVELOPE`]) gives the absolute term for rows that cancel harder.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::activation::Activation;
use crate::descriptor::Key;
use crate::fp4::{ScaleLayout, ValueLayout};
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

use super::{CLUSTER, TILE, address, to_i64};

/// Refuses the exact GELU: a FC1 gates with `silu` or `gelu_tanh`, the two its kernels compile.
fn require_gate(operator: &'static str, activation: Activation) -> Result<()> {
    match activation {
        Activation::Silu | Activation::GeluTanh => Ok(()),
        Activation::Gelu | Activation::Sigmoid => Err(Error::Shape(format!(
            "{operator}: a grouped expert product gates with silu or gelu_tanh, not {activation}"
        ))),
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

impl GatherGemmDescriptor {
    /// The compilation of the gated product of `top` routes over `experts` experts of
    /// `intermediate` gated width from `hidden`-wide `dtype` rows, gated by `activation`, at the
    /// products' tile and cluster.
    pub fn new(
        hidden: u32,
        intermediate: u32,
        experts: u32,
        top: u32,
        activation: Activation,
        dtype: DType,
    ) -> Self {
        Self {
            hidden,
            intermediate,
            experts,
            top,
            activation,
            dtype,
            tile: TILE,
            cluster: CLUSTER,
        }
    }
}

impl fmt::Display for GatherGemmDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("activation", self.activation)
            .field("dtype", abi::name(self.dtype))
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

impl RouteGemmDescriptor {
    /// The compilation of the second product of `top` routes over `experts` experts from
    /// `intermediate`-wide `dtype` rows into `hidden` columns, at the products' tile and
    /// cluster.
    pub fn new(hidden: u32, intermediate: u32, experts: u32, top: u32, dtype: DType) -> Self {
        Self {
            hidden,
            intermediate,
            experts,
            top,
            dtype,
            tile: TILE,
            cluster: CLUSTER,
        }
    }
}

impl fmt::Display for RouteGemmDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("dtype", abi::name(self.dtype))
            .field("tile", format_args!("{}x{}", self.tile.0, self.tile.1))
            .field(
                "cluster",
                format_args!("{}x{}", self.cluster.0, self.cluster.1),
            )
            .finish(formatter)
    }
}

/// The arguments of one FC1 launch.
#[derive(Clone)]
pub struct GatherGemmArgs {
    /// `[tokens, hidden]` hidden states, row-major.
    pub hidden: Operand,
    /// `[experts, 2 · intermediate, hidden]` expert weights.
    pub weights: Operand,
    /// `[tokens, top]` u32 route ids — the source of the expanded route count.
    pub ids: Operand,
    /// `[tokens, top]` u32: the permuted row of each `(token, route)` pair, as `moe_sort` writes it
    /// (the same map `moe_unpermute` reads). It is the consumer's lookup, not this product's: the
    /// product writes row `p` of `out` at the sort's permuted row `p`, so the map is what lets the
    /// reader find a route afterwards.
    pub expanded_to_permuted: Operand,
    /// `[max_tiles]` u32 expert of each tile.
    pub tile_to_expert: Operand,
    /// `[max_tiles]` u32 live rows of each tile.
    pub tile_to_limit: Operand,
    /// `[max_permuted_rows]` u32 expanded route of each permuted row.
    pub permuted_to_expanded: Operand,
    /// `[1]` u32 live tiles.
    pub num_non_exiting_tiles: Operand,
    /// `[max_permuted_rows, intermediate]` gated products, written.
    pub out: Operand,
}

/// The call structure the FC1 package's entry receives, declared to the packages as
/// `uniserve_gather_call` (`uniserve_gather_gemm.h`, generated with its layout asserted: [`abi`]).
#[repr(C)]
#[derive(abi::Layout)]
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
    /// The multiprocessors of the device the launch runs on: the bound of the persistent grid,
    /// which keeps at most one CTA on each.
    pub multiprocessors: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for GatherCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// The first matrix of a routed expert.
pub struct GatherGemm;

impl Op for GatherGemm {
    const NAME: &'static str = proto::module::operator::GATHER_GEMM;

    type Descriptor = GatherGemmDescriptor;
    type Args<'a> = GatherGemmArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        require_gate(Self::NAME, descriptor.activation)?;
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let tokens = args.hidden.rows();

        args.hidden.require_shape(Self::NAME, &[tokens, hidden])?;
        args.hidden
            .require_row(Self::NAME, descriptor.dtype, hidden)?;
        args.weights
            .require_shape(Self::NAME, &[experts, 2 * intermediate, hidden])?;
        args.weights
            .require_row(Self::NAME, descriptor.dtype, hidden)?;
        // `ids` states the expanded route count and `expanded_to_permuted` is that same
        // `(token, route)` space as the consumer sees it: the sort writes both, and a map of another
        // shape is one a layer looks routes up in past its end. The hidden states name `tokens`, so
        // all three agree on the route count or the launch refuses.
        for (what, view) in [
            ("route ids", &args.ids),
            ("the route map", &args.expanded_to_permuted),
        ] {
            view.require_shape(Self::NAME, &[tokens, top])?;
            if view.dtype != DType::U32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected u32 {what}, found {}", abi::name(view.dtype)),
                ));
            }
        }
        for view in [
            &args.tile_to_expert,
            &args.tile_to_limit,
            &args.permuted_to_expanded,
            &args.num_non_exiting_tiles,
        ] {
            if view.dtype != DType::U32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected u32 sort metadata, found {}",
                        abi::name(view.dtype)
                    ),
                ));
            }
            super::require_u32_vector(Self::NAME, view)?;
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
            .require_shape(Self::NAME, &[permuted, intermediate])?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        Ok(())
    }

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
            dtype: abi::code(descriptor.dtype)?,
            multiprocessors: i64::from(stream.device().multiprocessors()?),
            error: [0; MESSAGE],
        };
        // SAFETY: `GatherCall` is the call structure this package's entry declares: the declaration
        // is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        GatherGemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// The standard bound for the gated product, written against `hidden` and FP32 accumulation.
    ///
    /// One output element is `Σ_{j<hidden} x_j · w_j` over the contraction (a dense row of the
    /// expert's up/gate matrix gathered into place — the gather moves rows, not terms), so the
    /// reduction length is `hidden`, a descriptor field, and `γₙ` is the reduction's factor.
    ///
    /// The `atol` is the term the reduction does not cover: **the activation's envelope**. This
    /// operator is the first half of the gated product, so its output is what `silu` and the
    /// multiply then amplify, and the reference compares the *gated* result.
    /// [`tolerance::VALUE_ENVELOPE`] is the bound on that factor and `2 · γₙ · envelope² · 2` is
    /// what survives it — the same sentence for `route_gemm`, whose input is already one.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.hidden);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(tolerance::VALUE_ENVELOPE * tolerance::VALUE_ENVELOPE * 2.0))
                as f32;
        Tolerance::dot(reduction, accumulate, descriptor.dtype).with_atol(atol)
    }
}

/// The second matrix of a routed expert.
pub struct RouteGemm;

impl Op for RouteGemm {
    const NAME: &'static str = proto::module::operator::ROUTE_GEMM;

    type Descriptor = RouteGemmDescriptor;
    type Args<'a> = RouteGemmArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let rows = args.rows.rows();
        let tokens = args.ids.rows();

        args.rows.require_shape(Self::NAME, &[rows, intermediate])?;
        args.rows
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        args.weights
            .require_shape(Self::NAME, &[experts, hidden, intermediate])?;
        args.weights
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        args.ids.require_shape(Self::NAME, &[tokens, top])?;
        if args.ids.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u32 route ids, found {}",
                    abi::name(args.ids.dtype)
                ),
            ));
        }
        for view in [
            &args.tile_to_expert,
            &args.tile_to_limit,
            &args.permuted_to_expanded,
            &args.num_non_exiting_tiles,
        ] {
            if view.dtype != DType::U32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected u32 sort metadata, found {}",
                        abi::name(view.dtype)
                    ),
                ));
            }
            super::require_u32_vector(Self::NAME, view)?;
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
        args.out.require_row(Self::NAME, descriptor.dtype, hidden)?;
        // The product writes one row per **permuted** row — the height of `rows`, not the count of
        // live expanded routes — because the combine indexes this buffer in the permuted row space.
        // A buffer shorter than that is an overflow into whatever the arena placed next, and the
        // entry cannot see it: it knows the height it was given and writes the map's rows.
        if args.out.rows() < rows {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the output holds {} rows and this launch writes {rows} permuted rows",
                    args.out.rows()
                ),
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
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let rows = args.rows.rows();

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
            dtype: abi::code(descriptor.dtype)?,
            multiprocessors: i64::from(stream.device().multiprocessors()?),
            error: [0; MESSAGE],
        };
        // SAFETY: `RouteCall` is the call structure this package's entry declares: the declaration
        // is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        RouteGemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// `route_gemm`'s bound: the contraction is the expert's own width, `intermediate`.
    ///
    /// One output element is `Σ_{j<intermediate} g(x)_j · w_j`: `intermediate` products in FP32, so
    /// `rtol = γₙ + u` with `n = intermediate`, a descriptor field. The `atol` is again the gated
    /// activation's envelope — this operator has no activation of its own, but its *input* is one,
    /// and its rounding relative to the activation's bound is what the reference comparison sees.
    /// A dense launch of the same expert differs only in how the row was found, never in these terms.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.intermediate);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(tolerance::VALUE_ENVELOPE * tolerance::VALUE_ENVELOPE * 2.0))
                as f32;
        Tolerance::dot(reduction, accumulate, descriptor.dtype).with_atol(atol)
    }
}

/// The arguments of one FC2 launch.
#[derive(Clone)]
pub struct RouteGemmArgs {
    /// `[max_permuted_rows, intermediate]` FC1's gated products.
    pub rows: Operand,
    /// `[experts, hidden, intermediate]` expert weights.
    pub weights: Operand,
    /// `[tokens, top]` u32 route ids.
    pub ids: Operand,
    /// `[max_tiles]` u32 expert of each tile.
    pub tile_to_expert: Operand,
    /// `[max_tiles]` u32 live rows of each tile.
    pub tile_to_limit: Operand,
    /// `[max_permuted_rows]` u32 expanded route of each permuted row.
    pub permuted_to_expanded: Operand,
    /// `[1]` u32 live tiles.
    pub num_non_exiting_tiles: Operand,
    /// `[max_permuted_rows, hidden]` every permuted row's expert product, written unweighted: row
    /// `p` of `out` is the product of the sort's permuted row `p`, which is also `out`'s row count
    /// here and the height the launch checks.
    pub out: Operand,
}

/// The call structure the FC2 package's entry receives, declared to the packages as
/// `uniserve_route_call` (`uniserve_route_gemm.h`, generated with its layout asserted: [`abi`]).
#[repr(C)]
#[derive(abi::Layout)]
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
    pub dtype: i64,
    /// The multiprocessors of the device the launch runs on: the bound of the persistent grid,
    /// which keeps at most one CTA on each.
    pub multiprocessors: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for RouteCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// The block-scale layout both routed NVFP4 products read their weights' scales in.
///
/// The packages compile their kernels for this layout of the weight operand (their manifests'
/// `mSfLayoutA`), while the tokens' layout is the descriptor's [`GatherGemmNvfp4Descriptor::layout`];
/// a caller transforms every checkpoint's weight scales into this layout before it binds them.
pub const WEIGHT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

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
    /// Element type of the output — the type the accumulator rounds to. The weights *and* the
    /// activations are packed E2M1, so neither is this type; it is the epilogue's.
    pub dtype: DType,
    /// Element type of the block scales, as the checkpoint stores them: `F8_E4M3` for a ModelOpt
    /// export, `U8` for a shard that already holds the codes. One byte either way, and never
    /// assumed — the loader passes the shard's own type.
    pub scale: DType,
    /// The block-scale layout the checkpoint uses.
    pub layout: ScaleLayout,
    /// The MMA tile the package compiled for: rows by columns.
    pub tile: (u32, u32),
    /// The cluster shape the package compiled for: rows by columns.
    pub cluster: (u32, u32),
}

/// The expert width both encoded products compute over for `intermediate` columns of a
/// checkpoint's experts: padded to whole MMA row tiles of the gated product, whose stack holds
/// `2 · intermediate` rows, and to whole swizzled scale atoms of the second product's
/// contraction. A loader pads a rank's share of the experts to this width with zero codes and
/// zero scales, which contribute nothing.
pub const fn packed_width(intermediate: u32) -> u32 {
    let rows = TILE.0 / 2;
    let atoms = crate::fp4::SWIZZLE_COLUMNS as u32 * crate::fp4::BLOCK_ELEMENTS;

    // The least common multiple of the two granules, both powers of two in every package.
    let (mut a, mut b) = (rows, atoms);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    intermediate.next_multiple_of(rows / a * atoms)
}

/// The bytes an encoded product's entry writes before the package's workspace: a header that
/// holds the padded route count.
const ENTRY_PREAMBLE: u64 = 16;

/// The workspace an encoded product of `routes` permuted rows and `width` output columns at
/// `tile` and `cluster` asks for, in bytes: the package's own sizing of its row-maximum planes
/// and tile bars, each padded to a kibibyte, its scheduler's counter, and the entry's preamble
/// before them.
///
/// The size follows from a bucket's worst-case route count, so every replay of a captured bucket
/// takes the same bytes.
pub(super) fn packed_workspace(
    routes: u64,
    width: u64,
    tile: (u32, u32),
    cluster: (u32, u32),
) -> u64 {
    let kibibytes = |bytes: u64| bytes.div_ceil(1024) * 1024;
    let token_tile = u64::from(tile.0) * u64::from(cluster.0);
    let padded = routes.max(1).div_ceil(token_tile) * token_tile;

    // One FP32 row maximum per 128 elements of the padded tile grid.
    let row_maxima = width * padded / 128 * 4;
    // One half-word of tile bars per (token tile, width tile) pair.
    let bars = (padded / token_tile) * width.div_ceil(u64::from(tile.1)) / 2 * 4;

    ENTRY_PREAMBLE + kibibytes(row_maxima) + kibibytes(bars) + 4
}

impl GatherGemmNvfp4Descriptor {
    /// The arrangement of the `[up; gate]` stack's packed rows the gated product reads: the two
    /// halves interleaved row by row, then shuffled inside the blocks of one tile
    /// (`mUseShuffledMatrix`).
    pub const WEIGHT_VALUES: ValueLayout = ValueLayout::InterleavedShuffled {
        epilogue_tile_m: TILE.0,
    };

    /// The arrangement of the encoded tokens' block scales the product reads: it gathers the
    /// tokens it routes, in the linear form. The descriptor's `layout`.
    pub const TOKEN_SCALES: ScaleLayout = ScaleLayout::Linear;

    /// The compilation of the gated product of `top` routes over `experts` packed experts of
    /// `intermediate` gated width from `hidden`-wide tokens into `dtype` rows, gated by
    /// `activation`, over block scales of type `scale`: tokens in [`Self::TOKEN_SCALES`], at
    /// the products' tile and cluster.
    pub fn new(
        hidden: u32,
        intermediate: u32,
        experts: u32,
        top: u32,
        activation: Activation,
        dtype: DType,
        scale: DType,
    ) -> Self {
        Self {
            hidden,
            intermediate,
            experts,
            top,
            activation,
            dtype,
            scale,
            layout: Self::TOKEN_SCALES,
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

impl fmt::Display for GatherGemmNvfp4Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("activation", self.activation)
            .field("dtype", abi::name(self.dtype))
            .field("scale", abi::name(self.scale))
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
#[derive(Clone)]
pub struct GatherGemmNvfp4Args {
    /// `[tokens, hidden/2]` U8: the activations, encoded — two E2M1 codes a byte along the
    /// contraction, exactly what `nvfp4_quantize` produces. **Token-major**: the product gathers its
    /// rows itself, through the route map the cta maps and `permuted_to_token` describe, which is
    /// what every kernel in the artifact with an E2M1 weight and a BF16 output does (their routing
    /// implementation is `Ldgsts` or `Tma`, never `NoRoute`). A route-major buffer here would be
    /// read as tokens and gathered a second time.
    pub hidden: Operand,
    /// `[tokens, hidden/16]` the activations' block scales, one per sixteen elements of the
    /// contraction, of the descriptor's `scale` type in its `layout`.
    pub hidden_scale: Operand,
    /// `[tokens]` f32: the encoder's per-token scale factor. The artifact's FP4 kernels carry one
    /// beside the block scales (`Bb16tokFp32`), and it is the product's `perTokenScales` argument:
    /// a caller that passes the block scales alone leaves the kernel reading a null table.
    pub hidden_token_scale: Operand,
    /// `[experts, 2·intermediate, hidden/2]` U8: two E2M1 codes a byte along the contraction.
    pub weights: Operand,
    /// `[experts, ...]` the block scales in the descriptor's layout, one per sixteen elements.
    pub scales: Operand,
    /// `[2, experts]` f32, contiguous: each expert's tensor scales, one **per projection** in the
    /// fused reading order — row 0 is the value half (up), row 1 the gate half. A
    /// checkpoint carries two separate per-expert scalar tables, and the product's kernel reads each
    /// as its own array of `experts` floats, so the two halves are two *rows* here and not an
    /// interleaved `[experts, 2]`: an interleaved table cannot be handed to a reader that indexes
    /// `array[expert]`.
    pub global_scale: Operand,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: Operand,
    /// `[tiles]` u32: the local row limit of that tile.
    pub tile_to_limit: Operand,
    /// `[routes]` u32: the token row each permuted row gathers, `-1` on a tile's padding. The
    /// product indexes the token-major activations with it, so an expanded route here would read
    /// past the last token.
    pub permuted_to_token: Operand,
    /// `[1]` u32: the number of tiles that have work.
    pub num_non_exiting_tiles: Operand,
    /// `[routes, intermediate]` the gated product.
    pub out: Operand,
    /// `[bytes]` U8: scratch the program needs for its own bookkeeping. The size its tactic needs is
    /// the artifact's own requirement, not something `ops` can restate; a view shorter than that is
    /// refused by name at the point of use, never grown — a step that allocates is a host allocation inside a
    /// captured graph, and the address the pool baked is the wrong one the moment the step grows.
    pub workspace: Operand,
}

/// Check the scratch view both encoded products carry.
///
/// A workspace is bytes the program uses for its own bookkeeping, and the size the tactic needs is the
/// artifact's own requirement rather than anything `ops` can restate — so this checks only what is a
/// property of the operand itself: plain bytes, and at least one of them. The floor is the
/// *artifact's*, checked at
/// the point of use by the entry, where a short buffer is a named refusal; it is never grown, because
/// the step runs inside a captured graph and an allocation there lands at an address the pool did not
/// bake into the capture.
pub(super) fn check_workspace(operator: &'static str, workspace: &Operand) -> Result<()> {
    workspace.require_rank(operator, 1)?;
    if workspace.dtype != DType::U8 {
        return Err(Error::argument(
            operator,
            format!(
                "expected u8 workspace bytes, found {}",
                abi::name(workspace.dtype)
            ),
        ));
    }
    if workspace.numel() == 0 {
        return Err(Error::argument(
            operator,
            "a workspace of no bytes: the size the tactic needs is the caller's to provide",
        ));
    }
    Ok(())
}

/// One encoded FC1 launch, as the trtllm-gen batched-GEMM entry reads it, declared to the packages
/// as `uniserve_gather_gemm_nvfp4_call` (`uniserve_gather_gemm_nvfp4.h`, generated with its layout
/// asserted: [`abi`]).
///
/// The call is a plain structure of pointers and extents rather than a packed argument list, because
/// a native module's entry takes one: `uniserve_gather_gemm_nvfp4(call, stream)`. The descriptors'
/// choices that are *compiled* into the program — the tile, the cluster, the activation, the block
/// scale layout — travel as well, since the entry checks the call against the module it was built as
/// and refuses a mismatch by name rather than launching a kernel with another tactic's assumptions.
#[repr(C)]
#[derive(abi::Layout)]
pub struct GatherNvfp4Call {
    /// `[tokens, hidden/2]` U8: the activations, two E2M1 codes a byte, token-major.
    pub hidden: *const c_void,
    /// The activations' block scales.
    pub hidden_scale: *const c_void,
    /// `[tokens]` f32: the encoder's per-token scale factor.
    pub hidden_token_scale: *const c_void,
    /// `[experts, 2·intermediate, hidden/2]` U8: two E2M1 codes a byte.
    pub weights: *const c_void,
    /// The block scales, one per sixteen elements of the contraction.
    pub scales: *const c_void,
    /// `[experts]` f32: the reciprocals of the experts' tensor scales for the value (up) projection.
    pub global_scale: *const c_void,
    /// `[experts]` f32: the same for the gate projection.
    pub global_scale_gate: *const c_void,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: *const i32,
    /// `[tiles]` u32: the local row limit of that tile.
    pub tile_to_limit: *const i32,
    /// `[routes]` u32: the gathered row of each route.
    pub permuted_to_token: *const i32,
    /// `[1]` u32: the number of tiles that have work.
    pub num_non_exiting_tiles: *const i32,
    /// `[routes, intermediate]`: the gated product.
    pub out: *mut c_void,
    /// The scratch the program may use; the entry grows it itself when it is short.
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
    /// The element type of the activations and the output, as `abi::code` numbers it.
    pub dtype: i64,
    /// A NUL-terminated refusal, when the entry returns nonzero.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for GatherNvfp4Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// One encoded FC2 launch, as the trtllm-gen batched-GEMM entry reads it, declared to the packages
/// as `uniserve_route_gemm_nvfp4_call` (`uniserve_route_gemm_nvfp4.h`, generated with its layout
/// asserted: [`abi`]).
#[repr(C)]
#[derive(abi::Layout)]
pub struct RouteNvfp4Call {
    /// `[routes, intermediate/2]` U8: the gated rows FC1 produced, encoded.
    pub hidden: *const c_void,
    /// The rows' block scales.
    pub hidden_scale: *const c_void,
    /// `[routes]` f32: the encoder's per-token scale factor.
    pub hidden_token_scale: *const c_void,
    /// `[experts, hidden, intermediate/2]` U8: two E2M1 codes a byte.
    pub weights: *const c_void,
    /// The block scales, one per sixteen elements of the contraction.
    pub scales: *const c_void,
    /// `[experts, 1]` f32: each expert's tensor scale, as the checkpoint states it — a factor the
    /// product multiplies, not its reciprocal.
    pub global_scale: *const c_void,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: *const i32,
    /// `[tiles]` u32: the local row limit of that tile.
    pub tile_to_limit: *const i32,
    /// `[routes]` u32: the gathered row of each route.
    pub permuted_to_expanded: *const i32,
    /// `[1]` u32: the number of tiles that have work.
    pub num_non_exiting_tiles: *const i32,
    /// `[routes, hidden]`: the contracted rows.
    pub out: *mut c_void,
    /// The scratch the program needs, sized by the caller; the entry refuses a short one and never
    /// grows it.
    pub workspace: *mut c_void,
    /// Bytes of `workspace`.
    pub workspace_bytes: i64,
    /// Rows of the gated product and of the output: the padded route count.
    pub routes: i64,
    /// The output width, as the descriptor and the runner name it.
    pub hidden_size: i64,
    /// The contraction length: one expert's intermediate width.
    pub intermediate: i64,
    /// Experts the weight stack holds.
    pub experts: i64,
    /// Routes per token; the artifact's configuration list is indexed by it.
    pub top: i64,
    /// The compiled MMA tile, rows by columns.
    pub tile_m: i64,
    pub tile_n: i64,
    /// The compiled cluster, rows by columns.
    pub cluster_m: i64,
    pub cluster_n: i64,
    /// The element type of the activations and the output, as `abi::code` numbers it.
    pub dtype: i64,
    /// A NUL-terminated refusal, when the entry returns nonzero.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for RouteNvfp4Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// The routed first matrix over a packed, block-scaled expert stack.
pub struct GatherGemmNvfp4;

impl Op for GatherGemmNvfp4 {
    const NAME: &'static str = proto::module::operator::GATHER_GEMM_NVFP4;

    type Descriptor = GatherGemmNvfp4Descriptor;
    type Args<'a> = GatherGemmNvfp4Args;

    /// The launch's shape and geometry rules, and the workspace both encoded products carry.
    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        require_gate(Self::NAME, descriptor.activation)?;
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let top = u64::from(descriptor.top);
        let routes = args.out.rows();
        // The activation operands are token-major: the product gathers its rows itself.
        let tokens = args.hidden.rows();

        if args.out.last() != intermediate {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a row of the gated product is {intermediate} wide, the output is {:?}",
                    args.out.dims()
                ),
            ));
        }
        args.out
            .require_row(Self::NAME, descriptor.dtype, intermediate)?;
        // The activations are encoded the same way the weights are: two codes a byte along the
        // contraction. A BF16 row here would be read as packed codes and produce a product of half
        // the contraction, so the operand's width is the encoding's and is refused when it is not.
        args.hidden
            .require_shape(Self::NAME, &[tokens, hidden / 2])?;
        if args.hidden.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the gathered activations as packed u8, found {}: the product's kernels \
                     take an E2M1 activation, and `nvfp4_quantize` is the program that encodes one",
                    abi::name(args.hidden.dtype)
                ),
            ));
        }
        // The activation's block scales are the encoder's output, whose element type is the
        // encoder's declaration; `descriptor.scale` is the *weight* stack's, which is the
        // checkpoint's. `require_byte_scale` states the property both share.
        crate::fp4::require_byte_scale(Self::NAME, &args.hidden_scale, args.hidden_scale.last())?;
        args.hidden_token_scale
            .require_shape(Self::NAME, &[tokens])?;
        if args.hidden_token_scale.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected one f32 per-token scale, found {}: the artifact's FP4 kernels read the \
                     encoder's per-token factor beside the block scales",
                    abi::name(args.hidden_token_scale.dtype)
                ),
            ));
        }
        // The scales cover the contraction in blocks of sixteen, one per *token*: the product gathers
        // rows, so the table is indexed by the same token rows the activations are.
        let activation_blocks = hidden.div_ceil(u64::from(crate::fp4::BLOCK_ELEMENTS));
        let covered = args.hidden_scale.numel();
        if covered < tokens * activation_blocks {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{covered} activation block scales cover fewer than {tokens} rows of \
                     {activation_blocks} blocks"
                ),
            ));
        }
        // The packed field holds two codes a byte along the contraction, so its width is half the
        // contraction and no more: a wider one would read past the expert's weights.
        args.weights
            .require_shape(Self::NAME, &[experts, 2 * intermediate, hidden / 2])?;
        if args.weights.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the packed E2M1 field as u8, found {}",
                    abi::name(args.weights.dtype)
                ),
            ));
        }
        if args.scales.dtype != descriptor.scale {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} block scales, found {}",
                    abi::name(descriptor.scale),
                    abi::name(args.scales.dtype)
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
        let covered = args.scales.numel();
        if covered < experts * scale_rows * blocks_per_row {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{covered} block scales do not cover {experts} experts of {scale_rows} rows by \
                     {blocks_per_row} blocks"
                ),
            ));
        }
        args.global_scale.require_shape(Self::NAME, &[2, experts])?;
        if args.global_scale.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected f32 tensor scales, found {}",
                    abi::name(args.global_scale.dtype)
                ),
            ));
        }
        // The kernel reads each projection's table as `experts` consecutive floats, so the two rows
        // must be packed: a `[2, experts]` view with any other step would hand it one projection's
        // scales for both.
        if args.global_scale.stride(0) != experts || args.global_scale.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the tensor scales step {}/{}, and the product reads each projection's table as \
                     {experts} consecutive f32: the operand must be packed `[2, {experts}]`",
                    args.global_scale.stride(0),
                    args.global_scale.last_stride()
                ),
            ));
        }
        // `routes` is the permuted row count -- the output's rows, which the sort pads to whole tiles
        // -- while the activations are the caller's *token* rows. The product gathers through
        // `permuted_to_token`, so checking the activations against the permuted count would
        // refuse every launch whose sort pads anything. Neither artifact launch reads a per-route
        // expert id: the routing is the sort's, and the product takes the tile maps the sort
        // computed.
        args.permuted_to_token
            .require_shape(Self::NAME, &[routes])?;
        if args.permuted_to_token.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u32 permuted rows, found {}",
                    abi::name(args.permuted_to_token.dtype)
                ),
            ));
        }
        // The activation's width is the packed one stated above - `hidden / 2` bytes over the
        // contraction - and this product has no other: a second statement of `[tokens, hidden]` here
        // is the *dense* product's and refused every encoded launch by naming the width the caller
        // had already correctly halved.
        if top == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a launch with no routes per token packs no rows".to_owned(),
            ));
        }
        check_workspace(Self::NAME, &args.workspace)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let routes = args.out.rows();
        let tokens = args.hidden.rows();

        let mut call = GatherNvfp4Call {
            hidden: address(args.hidden.ptr) as usize as *const c_void,
            hidden_scale: address(args.hidden_scale.ptr) as usize as *const c_void,
            hidden_token_scale: address(args.hidden_token_scale.ptr) as usize as *const c_void,
            weights: address(args.weights.ptr) as usize as *const c_void,
            scales: address(args.scales.ptr) as usize as *const c_void,
            global_scale: address(args.global_scale.ptr) as usize as *const c_void,
            // The gate half's row: the two tables are consecutive `experts` floats, so the second
            // row is the first address plus the row stride the operand is validated for.
            global_scale_gate: (address(args.global_scale.ptr) + 4 * args.global_scale.stride(0))
                as usize as *const c_void,
            tile_to_expert: address(args.tile_to_expert.ptr) as usize as *const i32,
            tile_to_limit: address(args.tile_to_limit.ptr) as usize as *const i32,
            permuted_to_token: address(args.permuted_to_token.ptr) as usize as *const i32,
            num_non_exiting_tiles: address(args.num_non_exiting_tiles.ptr) as usize as *const i32,
            out: address(args.out.ptr) as usize as *mut c_void,
            workspace: address(args.workspace.ptr) as usize as *mut c_void,
            workspace_bytes: to_i64(Self::NAME, args.workspace.numel())?,
            tokens: to_i64(Self::NAME, tokens)?,
            routes: to_i64(Self::NAME, routes)?,
            hidden_size: to_i64(Self::NAME, hidden)?,
            intermediate: to_i64(Self::NAME, intermediate)?,
            experts: to_i64(Self::NAME, u64::from(descriptor.experts))?,
            top: to_i64(Self::NAME, u64::from(descriptor.top))?,
            tile_m: i64::from(descriptor.tile.0),
            tile_n: i64::from(descriptor.tile.1),
            cluster_m: i64::from(descriptor.cluster.0),
            cluster_n: i64::from(descriptor.cluster.1),
            dtype: abi::code(descriptor.dtype)?,
            error: [0; MESSAGE],
        };
        // SAFETY: `GatherNvfp4Call` is the structure this package's entry declares: the declaration
        // is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        GatherGemmNvfp4Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// `gather_gemm`'s bound exactly, because the dequantization is exact where it happens.
    ///
    /// The packed operand is E2M1 codes plus one E4M3 scale per sixteen elements *along the
    /// contraction*; the program decodes a code (exact: E2M1 is a subset of FP32) and multiplies it
    /// by its block's scale (exact: E4M3 into FP32), then sums the products in FP32 — each product is the
    /// reference's own `w · s` product, so the sum has the same reduction length (`hidden`), the same
    /// accumulation dtype, and the same bound. The one per-expert tensor scale is a *factor*, applied
    /// once, and contributes one rounding that `rtol`'s `u` already covers.
    ///
    /// The bound is therefore **not looser than the dense form's**: an implementation that instead
    /// scaled the accumulated row by the reciprocal would add one rounding per output element, which
    /// this bound would catch — the order is part of what the tolerance asserts, not an aside.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.hidden);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(tolerance::VALUE_ENVELOPE * tolerance::VALUE_ENVELOPE * 2.0))
                as f32;
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
        // FC1 applies the gated product inside its launch; FC2 only multiplies, so the activation is
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

    /// A share of 160 columns, a rank's quarter of 640, is computed as 192: three 128-row tiles of
    /// the gated stack and three scale atoms of the second contraction.
    #[test]
    fn a_share_is_padded_to_whole_tiles_and_scale_atoms() {
        assert_eq!(packed_width(160), 192);
        assert_eq!(packed_width(192), 192);
        assert_eq!(packed_width(768), 768);
        assert_eq!(packed_width(1), 64);
    }

    /// The workspaces follow the package's arithmetic over a bucket's route count: 15 tokens over
    /// 128 experts are 120 tiles of 128 permuted rows, for which the gated product (1536 columns)
    /// asks 16 + 740 356 bytes and the second (2048 columns) 16 + 987 140; a bucket of a single
    /// tile still carries the entry's preamble.
    #[test]
    fn a_workspace_is_sized_from_the_route_count() {
        let (gated, second) = (descriptor(), route_descriptor());

        assert_eq!(gated.workspace_bytes(15_360), 740_372);
        assert_eq!(second.workspace_bytes(15_360), 987_156);
        assert_eq!(gated.workspace_bytes(8), 7_188);
        assert_eq!(second.workspace_bytes(8), 9_236);

        // The size grows with the routes and stays far below a fixed floor of mebibytes.
        assert!(second.workspace_bytes(4096) > second.workspace_bytes(8));
        assert!(second.workspace_bytes(4096) < 16 << 20);
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
/// because FC1 applies the gated product and FC2 only multiplies it. Its contraction is the expert
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
    /// Element type of the output; the weights and the rows are both packed E2M1.
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

impl RouteGemmNvfp4Descriptor {
    /// The arrangement of the `down` stack's packed rows the product reads: shuffled inside the
    /// blocks of one tile (`mUseShuffledMatrix`).
    pub const WEIGHT_VALUES: ValueLayout = ValueLayout::Shuffled {
        epilogue_tile_m: TILE.0,
    };

    /// The arrangement of the encoded rows' block scales the product reads: the descriptor's
    /// `layout`.
    pub const TOKEN_SCALES: ScaleLayout = ScaleLayout::Swizzled128X4;

    /// The compilation of the second product of `top` routes over `experts` packed experts
    /// from `intermediate`-wide gated rows into `hidden` columns of `dtype`, over block scales
    /// of type `scale`: rows in [`Self::TOKEN_SCALES`], at the products' tile and cluster.
    pub fn new(
        hidden: u32,
        intermediate: u32,
        experts: u32,
        top: u32,
        dtype: DType,
        scale: DType,
    ) -> Self {
        Self {
            hidden,
            intermediate,
            experts,
            top,
            dtype,
            scale,
            layout: Self::TOKEN_SCALES,
            tile: TILE,
            cluster: CLUSTER,
        }
    }

    /// The bytes of workspace a launch of `routes` permuted rows takes: the package's sizing over
    /// the product's `hidden` output columns.
    pub fn workspace_bytes(&self, routes: u64) -> u64 {
        packed_workspace(routes, u64::from(self.hidden), self.tile, self.cluster)
    }
}

impl fmt::Display for RouteGemmNvfp4Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("experts", self.experts)
            .field("top", self.top)
            .field("dtype", abi::name(self.dtype))
            .field("scale", abi::name(self.scale))
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
#[derive(Clone)]
pub struct RouteGemmNvfp4Args {
    /// `[routes, intermediate/2]` U8: the gated rows, encoded as `nvfp4_quantize` produces them —
    /// two E2M1 codes a byte along the contraction. FC1's epilogue is a BF16 row, so the encoding
    /// between the two products is a program of its own, not a wider operand of this one.
    pub hidden: Operand,
    /// The rows' block scales, one per sixteen elements of the contraction, of the descriptor's
    /// `scale` type in its `layout`.
    pub hidden_scale: Operand,
    /// `[routes]` f32: the encoder's per-token scale factor, beside the block scales.
    pub hidden_token_scale: Operand,
    /// `[experts, hidden, intermediate/2]` U8: two E2M1 codes a byte along the contraction.
    pub weights: Operand,
    /// The block scales, one per sixteen elements of the contraction and one row per output row.
    pub scales: Operand,
    /// `[experts]` f32: each expert's tensor scale, one **per projection** — this product fuses a
    /// single projection, so the table is one array of `experts` floats, the shape a reader that
    /// indexes `array[expert]` takes. The values are the checkpoint's scales, factors the product
    /// multiplies, not their reciprocals: the operand's own reconstruction already carries the
    /// encoder's factor, so a reciprocal here would divide the scale out twice.
    pub global_scale: Operand,
    /// `[tiles]` u32: the expert a tile's rows belong to.
    pub tile_to_expert: Operand,
    /// `[tiles]` u32: the local row limit of that tile.
    pub tile_to_limit: Operand,
    /// `[routes]` u32: the gathered row of each route.
    pub permuted_to_expanded: Operand,
    /// `[1]` u32: the number of tiles that have work.
    pub num_non_exiting_tiles: Operand,
    /// `[routes, hidden]` the contracted rows, one per route: the `top` repeats are combined by
    /// `moe_unpermute`, not here.
    pub out: Operand,
    /// `[bytes]` U8: scratch for the program's own bookkeeping, as FC1's is: the artifact's size or
    /// more, refused at the point of use when it is short.
    pub workspace: Operand,
}

/// FC2 of an encoded, block-scaled expert block.
pub struct RouteGemmNvfp4;

impl Op for RouteGemmNvfp4 {
    const NAME: &'static str = proto::module::operator::ROUTE_GEMM_NVFP4;

    type Descriptor = RouteGemmNvfp4Descriptor;
    type Args<'a> = RouteGemmNvfp4Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let intermediate = u64::from(descriptor.intermediate);
        let experts = u64::from(descriptor.experts);
        let routes = args.out.rows();

        args.out.require_shape(Self::NAME, &[routes, hidden])?;
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
        args.hidden
            .require_shape(Self::NAME, &[routes, intermediate / 2])?;
        if args.hidden.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the gated rows as packed u8, found {}: FC1's epilogue emits \
                     {}, and `nvfp4_quantize` is the program that encodes it",
                    abi::name(args.hidden.dtype),
                    abi::name(descriptor.dtype)
                ),
            ));
        }
        // The activation's block scales are the encoder's output, whose element type is the
        // encoder's declaration; `descriptor.scale` is the *weight* stack's, which is the
        // checkpoint's. `require_byte_scale` states the property both share.
        crate::fp4::require_byte_scale(Self::NAME, &args.hidden_scale, args.hidden_scale.last())?;
        args.hidden_token_scale
            .require_shape(Self::NAME, &[routes])?;
        if args.hidden_token_scale.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected one f32 per-token scale per route, found {}",
                    abi::name(args.hidden_token_scale.dtype)
                ),
            ));
        }
        let row_blocks = intermediate.div_ceil(u64::from(crate::fp4::BLOCK_ELEMENTS));
        let covered = args.hidden_scale.numel();
        if covered < routes * row_blocks {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{covered} activation block scales cover fewer than {routes} rows of \
                     {row_blocks} blocks"
                ),
            ));
        }
        args.weights
            .require_shape(Self::NAME, &[experts, hidden, intermediate / 2])?;
        if args.weights.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the packed E2M1 field as u8, found {}",
                    abi::name(args.weights.dtype)
                ),
            ));
        }
        if args.scales.dtype != descriptor.scale {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} block scales, found {}",
                    abi::name(descriptor.scale),
                    abi::name(args.scales.dtype)
                ),
            ));
        }
        let blocks_per_row = intermediate / block_elements;
        let covered = args.scales.numel();
        if covered < experts * hidden * blocks_per_row {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{covered} block scales do not cover {experts} experts of {hidden} rows by \
                     {blocks_per_row} blocks"
                ),
            ));
        }
        // The table is one packed f32 per expert, and the loader states it as a row of `experts`
        // (`[1, experts]`, stride `experts`), which is the form the gather product beside this one
        // reads. A rank-one `[experts]` has stride 1 there, so demanding that shape refuses the
        // table the model actually holds: this product reads the row, and the two agree.
        if args.global_scale.stride(0) != experts || args.global_scale.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the tensor scales must be one packed f32 per expert, addressed by a row \
                     stride of {experts}; the table's strides are {} and {}",
                    args.global_scale.stride(0),
                    args.global_scale.last_stride()
                ),
            ));
        }
        if args.global_scale.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected f32 tensor scales, found {}",
                    abi::name(args.global_scale.dtype)
                ),
            ));
        }
        args.permuted_to_expanded
            .require_shape(Self::NAME, &[routes])?;
        if args.permuted_to_expanded.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u32 permuted rows, found {}",
                    abi::name(args.permuted_to_expanded.dtype)
                ),
            ));
        }
        check_workspace(Self::NAME, &args.workspace)?;
        Ok(())
    }

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

        let mut call = RouteNvfp4Call {
            hidden: address(args.hidden.ptr) as usize as *const c_void,
            hidden_scale: address(args.hidden_scale.ptr) as usize as *const c_void,
            hidden_token_scale: address(args.hidden_token_scale.ptr) as usize as *const c_void,
            weights: address(args.weights.ptr) as usize as *const c_void,
            scales: address(args.scales.ptr) as usize as *const c_void,
            global_scale: address(args.global_scale.ptr) as usize as *const c_void,
            tile_to_expert: address(args.tile_to_expert.ptr) as usize as *const i32,
            tile_to_limit: address(args.tile_to_limit.ptr) as usize as *const i32,
            permuted_to_expanded: address(args.permuted_to_expanded.ptr) as usize as *const i32,
            num_non_exiting_tiles: address(args.num_non_exiting_tiles.ptr) as usize as *const i32,
            out: address(args.out.ptr) as usize as *mut c_void,
            workspace: address(args.workspace.ptr) as usize as *mut c_void,
            workspace_bytes: to_i64(Self::NAME, args.workspace.numel())?,
            routes: to_i64(Self::NAME, routes)?,
            hidden_size: to_i64(Self::NAME, hidden)?,
            intermediate: to_i64(Self::NAME, intermediate)?,
            experts: to_i64(Self::NAME, experts)?,
            top: i64::from(descriptor.top),
            tile_m: i64::from(descriptor.tile.0),
            tile_n: i64::from(descriptor.tile.1),
            cluster_m: i64::from(descriptor.cluster.0),
            cluster_n: i64::from(descriptor.cluster.1),
            dtype: abi::code(descriptor.dtype)?,
            error: [0; MESSAGE],
        };
        // SAFETY: `RouteNvfp4Call` is the structure this package's entry declares: the declaration
        // is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        RouteGemmNvfp4Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// `route_gemm`'s bound, and for the same reason as FC1's pair: decode, scale per block, reduce
    /// in FP32 — never scale the accumulated row, which would add a rounding the dense bound does not
    /// allow. The contraction is `intermediate`, the accumulation FP32, and the `atol` is the gated
    /// activation's envelope, since this operator's input is one.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.intermediate);
        let accumulate = DType::F32;
        let gamma = crate::tolerance::gamma(reduction, accumulate);
        let atol =
            (2.0 * gamma * f64::from(tolerance::VALUE_ENVELOPE * tolerance::VALUE_ENVELOPE * 2.0))
                as f32;
        Tolerance::dot(reduction, accumulate, descriptor.dtype).with_atol(atol)
    }
}

#[cfg(test)]
mod nvfp4_launch_tests {
    use super::*;

    fn table(dtype: DType, dims: &[u64]) -> Operand {
        Operand::new(platform::DevicePtr::from_address(0), dtype, dims)
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
            hidden: table(DType::U8, &[3, 1024]),
            hidden_scale: table(DType::F8E4M3, &[3, 128]),
            hidden_token_scale: table(DType::F32, &[3]),
            weights: table(DType::U8, &[128, 1536, 1024]),
            scales: table(DType::F8E4M3, &[128, 1536, 128]),
            global_scale: table(DType::F32, &[2, 128]),
            tile_to_expert: table(DType::U32, &[2]),
            tile_to_limit: table(DType::U32, &[2]),
            permuted_to_token: table(DType::U32, &[32]),
            num_non_exiting_tiles: table(DType::U32, &[1]),
            out: table(DType::BF16, &[32, 768]),
            workspace: table(DType::U8, &[4096]),
        }
    }

    #[test]
    fn the_activations_are_the_token_rows_and_the_output_the_permuted_ones() {
        // Builder01's M963: `routes` (the output's rows) is the sort's padded count and is never the
        // activation count. The invariant is `hidden = [tokens, hidden]` and
        // `permuted_to_token = [routes]`, which is what the program gathers through.
        let descriptor = fc1();
        let args = fc1_args();
        assert_eq!(args.hidden.rows(), 3);
        assert_eq!(args.out.rows(), 32);
        assert_eq!(args.permuted_to_token.dims()[0], 32);
        // The scales cover one block per output row along the contraction, not one row's worth.
        assert_eq!(args.scales.dims(), &[128, 1536, 128]);
        assert_eq!(descriptor.hidden / crate::fp4::BLOCK_ELEMENTS, 128);
        // And the weight stack is the packed field, half the contraction wide.
        assert_eq!(args.weights.dims(), &[128, 1536, 1024]);
    }

    #[test]
    fn an_odd_contraction_and_a_partial_block_are_refused_by_name() {
        // An odd contraction with a weight stack that matches it: the geometry rule is what refuses,
        // not the shape check that happens to fire first.
        let mut odd = fc1();
        odd.hidden = 2047;
        let mut odd_args = fc1_args();
        odd_args.weights = table(DType::U8, &[128, 1536, 1023]);
        // The packed activations shrink with the contraction (`hidden / 2`), so the descriptor's odd
        // width and the operand it is refused for agree; otherwise the shape check fires first and
        // the test would be green about the wrong rule.
        odd_args.hidden = table(DType::U8, &[3, 1023]);
        let error = <GatherGemmNvfp4 as Op>::check(&odd, &odd_args).unwrap_err();
        assert!(format!("{error}").contains("odd"), "{error}");

        let mut partial = fc1();
        partial.hidden = 2040;
        let mut partial_args = fc1_args();
        partial_args.weights = table(DType::U8, &[128, 1536, 1020]);
        // The same rule as the odd case: the packed activations shrink with the contraction, so a
        // stale width here would be refused by the shape check instead of by the block rule, and the
        // test would be green about the wrong refusal.
        partial_args.hidden = table(DType::U8, &[3, 1020]);
        let error = <GatherGemmNvfp4 as Op>::check(&partial, &partial_args).unwrap_err();
        assert!(format!("{error}").contains("scale blocks"), "{error}");
    }

    #[test]
    fn a_scale_cover_short_by_one_block_is_refused() {
        let mut args = fc1_args();
        args.scales = table(DType::F8E4M3, &[127, 1536, 128]);
        let error = <GatherGemmNvfp4 as Op>::check(&fc1(), &args).unwrap_err();
        assert!(format!("{error}").contains("do not cover"));
    }
}
