//! Expert-parallel exchange: moving routes to the ranks that hold their experts.
//!
//! A rank routes every token to `top` logical experts, of which only its
//! `[local_offset, local_offset + local_experts)` window is resident. The exchange carries each
//! route's payload — the hidden state, the expert id and the route weight — to the rank that holds
//! the expert (dispatch), and returns each token's partial expert sums to the rank that owns the
//! token (combine). Each backend (`alltoall`, `deepep`, `megamoe`, `deepgemm`) is a native library,
//! because the transport's workspace and its handshake are the package's host code; the Build
//! Server selects the one of each key, unless a deployment's `[kernels]` table pins it.
//! A backend whose one launch runs the whole expert layer states so in [`Info::fused`] and takes
//! [`Mode::Fused`] instead of the two directions.
//!
//! # What a fused launch runs
//!
//! A fused launch runs the layer's routed experts in the representation its descriptor names
//! ([`Descriptor::weights`]), and beside them the shared experts it states ([`Descriptor::shared`]),
//! each expert's gated activation clamped at [`Descriptor::limit`].
//!
//! - [`ExpertWeights::Nvfp4`]: calibrated NVFP4 stacks with three `[local]` FP32 tables, over rows
//!   the layer encoded into the transport's `packed` views before the launch; each route's output
//!   is weighed after its down product.
//! - [`ExpertWeights::Mxfp4`]: MXFP4 stacks under UE8M0 scales, and MXFP8 shared experts, over the
//!   BF16 rows of `state`, which the launch encodes into MXFP8 itself. A rank's output row is
//!
//! ```text
//! x̂    = mxfp8(x)                            per 32 elements: 2^s ≥ max(amax, 10⁻⁴) · fl(1/448)
//! g, u = bf16(W_gate,e x̂), bf16(W_up,e x̂)    each an FP32 contraction rounded once
//! a    = silu(min(g, L)) · clamp(u, ±L) · w   in FP32; w the route's weight, 1 for a shared one
//! â    = mxfp8_exact(a)                      per 32 elements: the least 2^s with amax ≤ 448 · 2^s
//! o_e  = bf16(W_down,e â)
//! y[t] = bf16(Σ_k o_ids[t,k] + Σ_s o_s)       the routes' and shared experts' sum in FP32
//! ```
//!
//!   The rows' encoding is `mxfp8_quantize`'s (the reference's `act_quant`), and the gated rows'
//!   covering rule is exact (its scale at least `2⁻²²`), so it differs from it only where
//!   `amax · fl(1/448)` rounds across a power of two. The route weight multiplies the gated row before its encoding, as the
//!   reference weighs it; a row past the live count is computed as a zero row and routed nowhere.
//!
//! # Who owns the transport
//!
//! The transport is a device-resident workspace shared between the ranks of one expert group. It is
//! an execution resource, so the worker creates it through the package's own lifecycle entries
//! ([`Transport`]), keeps it across a standby's suspend and restore, and sets its borrowed handle
//! ([`ExchangeTransport`]) on the execution context of the forwards that exchange; the context
//! states it in each launch, so a model never holds it. Before the transport exists, the package
//! states what it will take of the device ([`Transport::footprint`]), which the worker divides the
//! device's memory with. `ops` launches over a handle and never sizes one.
//!
//! # The step's capacity
//!
//! The ranks agree before each step on how many rows a rank sends at most. The value reaches the
//! package through [`Transport::step`], which writes it to device memory ahead of the replay, so the
//! kernels read it at run time: one captured graph serves every capacity, and the views keep the
//! static extents of the transport's largest step.
//!
//! # The bound
//!
//! Dispatch only moves rows: it is exact. Combine sums at most one partial per rank in FP32 and
//! rounds once, so two implementations differ by the `ranks`-term reduction and the output's
//! rounding, stated as [`Tolerance::dot`].

use std::ffi::{c_char, c_void};
use std::fmt;
use std::sync::Arc;

use platform::DevicePtr;
use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::fp4::{ScaleLayout, ValueLayout};
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::{address, to_i64};

/// Which direction of the exchange one module performs.
///
/// The call carries its number (`uniserve_exchange` in `uniserve_expert_exchange.h`).
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// Send every route to the rank holding its expert.
    Dispatch = 0,
    /// Return every token's partial expert sums to the rank holding the token.
    Combine = 1,
    /// Dispatch, run this layer's experts and combine, in one launch.
    Fused = 2,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Dispatch => "dispatch",
            Self::Combine => "combine",
            Self::Fused => "fused",
        })
    }
}

/// The representation of the routed experts a fused launch runs ([`Descriptor::weights`]).
///
/// The lifecycle call carries its number (`uniserve_exchange_weights` in
/// `uniserve_expert_exchange.h`).
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ExpertWeights {
    /// The launch runs no experts: a dispatch or a combine, whatever the layer's experts are.
    None = 0,
    /// Calibrated NVFP4 stacks and their FP32 tables, over rows the layer encoded.
    Nvfp4 = 1,
    /// MXFP4 stacks under UE8M0 scales, over BF16 rows the launch encodes into MXFP8.
    Mxfp4 = 2,
}

impl fmt::Display for ExpertWeights {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "none",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp4 => "mxfp4",
        })
    }
}

/// What a fused launch runs beside the exchange: the routed experts' representation, the shared
/// experts it runs in the same launch, and the clamp of every expert's gated activation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FusedExperts {
    /// The routed experts' representation.
    pub weights: ExpertWeights,
    /// Shared experts of the routed experts' width the launch runs beside them, every token
    /// through each; zero for none.
    pub shared: u32,
    /// The clamp `L` of the gated activation, `act(min(gate, L)) · clamp(up, ±L)`, as the bit
    /// pattern of its `f32`; zero for an unclamped one.
    pub limit: crate::descriptor::Float,
}

impl FusedExperts {
    /// What a launch that runs no experts states: a dispatch's or a combine's.
    pub const NONE: Self = Self {
        weights: ExpertWeights::None,
        shared: 0,
        limit: crate::descriptor::Float(0),
    };

    /// What a fused launch over calibrated NVFP4 experts runs: no shared expert, no clamp.
    pub const NVFP4: Self = Self {
        weights: ExpertWeights::Nvfp4,
        shared: 0,
        limit: crate::descriptor::Float(0),
    };

    /// The clamp as the `f32` it is the pattern of.
    pub fn clamp(&self) -> f32 {
        f32::from_bits(self.limit.0 as u32)
    }
}

/// The transport an exchange runs over: a borrowed, non-owning handle.
///
/// The package's lifecycle entries create it on the device and keep it for the life of the expert
/// group; the worker owns that lifetime and passes the handle here. A handle is only valid on the
/// device and with the group it was created for, and only between its creation and its release.
#[derive(Clone, Copy)]
pub struct ExchangeTransport {
    handle: *mut c_void,
}

// SAFETY: the handle is an opaque address the package reads on the thread that launches, and nothing
// here dereferences it; a forward that carries it may be built on one thread and used on another.
unsafe impl Send for ExchangeTransport {}
// SAFETY: as above: sharing the address reads nothing through it.
unsafe impl Sync for ExchangeTransport {}

impl ExchangeTransport {
    /// Borrow the package's transport from its opaque handle.
    ///
    /// # Safety
    ///
    /// `handle` must be the value the package's creation entry returned for this device and group,
    /// and must stay valid for every launch this transport is passed to.
    pub const unsafe fn from_handle(handle: *mut c_void) -> Self {
        Self { handle }
    }

    /// The opaque handle the package reads.
    pub fn handle(self) -> *mut c_void {
        self.handle
    }
}

impl fmt::Debug for ExchangeTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ExchangeTransport")
            .field(&format_args!("0x{:x}", self.handle as usize))
            .finish()
    }
}

/// One compilation of `expert_exchange`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// The direction this module performs.
    pub mode: Mode,
    /// Hidden elements of one row.
    pub hidden: u32,
    /// An expert's gated width, which a backend that runs the experts compiles for.
    pub intermediate: u32,
    /// Routes per token.
    pub top: u32,
    /// Experts in the router's vocabulary.
    pub experts: u32,
    /// Rows one rank may hold, the transport's capacity.
    pub max_tokens: u32,
    /// Ranks of the expert group whose transport this module drives.
    pub ranks: u32,
    /// Leading ranks of the group that hold no experts.
    pub attention: u32,
    /// Multiprocessors of the device the exchange runs on (`platform::DeviceId::multiprocessors`).
    /// A backend whose kernels spread over the whole device compiles its grid for them: a
    /// persistent grid that synchronizes grid-wide needs every block resident, so it is sized to
    /// the device's count.
    pub multiprocessors: u32,
    /// Element type of the states.
    pub dtype: DType,
    /// The routed experts' representation a fused launch runs; [`ExpertWeights::None`] for a
    /// dispatch or a combine, which run none.
    pub weights: ExpertWeights,
    /// Shared experts a fused launch runs beside the routed ones; zero for none, and for a
    /// dispatch or a combine.
    pub shared: u32,
    /// The clamp of the gated activation of every expert a fused launch runs, as its bit pattern;
    /// zero for an unclamped one, and for a dispatch or a combine.
    pub limit: crate::descriptor::Float,
}

impl Descriptor {
    /// The arrangement of the `[up; gate]` stack's packed rows a fused launch reads
    /// ([`FusedLayer::up_gate`]): each thirty-two-row block leads with the gate rows.
    pub const FUSED_UP_GATE_VALUES: ValueLayout = ValueLayout::GateFirst { rows: 16 };

    /// The arrangement of the `down` stack's packed rows a fused launch reads
    /// ([`FusedLayer::down`]): the checkpoint's own.
    pub const FUSED_DOWN_VALUES: ValueLayout = ValueLayout::Plain;

    /// The arrangement of both stacks' block scales a fused launch reads.
    pub const FUSED_SCALES: ScaleLayout = ScaleLayout::Swizzled128X4;

    /// The arrangement of the exchanged tokens' block scales a fused launch reads.
    pub const FUSED_TOKEN_SCALES: ScaleLayout = ScaleLayout::Linear;

    /// The arrangement of an MXFP4 fused launch's `[up; gate]` stacks' rows, routed and shared
    /// ([`FusedLayer::up_gate`], [`FusedLayer::shared_up_gate`]): each sixteen-row block leads
    /// with eight gate rows, then the same eight rows of up.
    pub const MXFP4_UP_GATE_VALUES: ValueLayout = ValueLayout::GateFirst { rows: 8 };

    /// The arrangement of an MXFP4 fused launch's down stacks' rows: the checkpoint's own.
    pub const MXFP4_DOWN_VALUES: ValueLayout = ValueLayout::Plain;

    /// The arrangement of every weight scale an MXFP4 fused launch reads: DeepGEMM's packed
    /// MN-major words with the rows of each 128-row group in the order its tensor-core copy reads
    /// them.
    pub const MXFP4_SCALES: crate::mx::ScaleLayout = crate::mx::ScaleLayout::PackedMnMajorUtccp;

    /// What this descriptor's launch runs beside the exchange.
    pub const fn fused(&self) -> FusedExperts {
        FusedExperts {
            weights: self.weights,
            shared: self.shared,
            limit: self.limit,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("mode", self.mode)
            .field("hidden", self.hidden)
            .field("intermediate", self.intermediate)
            .field("top", self.top)
            .field("experts", self.experts)
            .field("max_tokens", self.max_tokens)
            .field("ranks", self.ranks)
            .field("attention", self.attention)
            .field("multiprocessors", self.multiprocessors)
            .field("dtype", abi::name(self.dtype))
            .field("weights", self.weights)
            .field("shared", self.shared)
            .field("limit", self.limit)
            .finish(formatter)
    }
}

/// The arguments of one exchange launch.
///
/// At [`Mode::Dispatch`] the rank sends its rows in `state`, `ids` and `weights`; what it received
/// stands in the transport's views afterwards. At [`Mode::Combine`] the rows its experts computed
/// stand in the transport's combined view, and the package writes the rank's own rows into `out`.
/// At [`Mode::Fused`] one launch reads `ids` and `weights` and writes `out`, with the layer's
/// expert operands in `fused`; it reads the rows in `state` when it runs MXFP4 experts, and from
/// the transport's packed views the layer encoded them into when it runs NVFP4 ones. Views a mode
/// does not read are empty (`[0]`-element) views.
#[derive(Clone)]
pub struct ExchangeArgs {
    /// The group's transport, created and owned by the worker through the package and stated by
    /// the execution context at the launch. `None` in a call that is checked before any
    /// transport exists (a profile run); a launch refuses it.
    pub transport: Option<ExchangeTransport>,
    /// Rows of this launch's extent: the captured bucket's, of which `live` are the batch's.
    pub tokens: u32,
    /// Rows per rank the step this launch belongs to is shaped for, the capacity class of its
    /// graph: the most any rank sends, which a fused launch tiles its experts' rows for. Every
    /// rank of a step states the same, at least `tokens`.
    pub capacity: u32,
    /// `[tokens, hidden]` states to send.
    pub state: Operand,
    /// `[tokens, top]` u32 expert ids.
    pub ids: Operand,
    /// `[tokens, top]` FP32 route weights.
    pub weights: Operand,
    /// `[1]` u32 on the device: the rows of `tokens` the batch holds, or `None` for all of them.
    pub live: Option<Operand>,
    /// `[tokens, hidden]` combined rows, written.
    pub out: Operand,
    /// The layer's expert operands, for a fused launch.
    pub fused: Option<FusedLayer>,
    /// Ranks of the expert group.
    pub ranks: u32,
}

/// What a fused launch reads of one layer: its experts, as the loader laid them out, declared to
/// the packages as `uniserve_fused_layer`.
///
/// Each routed weight is `[local experts, rows, columns / 2]` packed values, read K-major, with its
/// block scales in a table of `scale_bytes` per expert. Every address is a device address the
/// layer's weights hold for the life of the model.
///
/// - NVFP4 ([`ExpertWeights::Nvfp4`]): the stacks in [`Descriptor::FUSED_UP_GATE_VALUES`] and
///   [`Descriptor::FUSED_DOWN_VALUES`], E4M3 block scales in [`Descriptor::FUSED_SCALES`], the
///   three tables; no shared expert.
/// - MXFP4 ([`ExpertWeights::Mxfp4`]): the stacks in [`Descriptor::MXFP4_UP_GATE_VALUES`] and
///   [`Descriptor::MXFP4_DOWN_VALUES`], their UE8M0 scales in [`Descriptor::MXFP4_SCALES`]
///   (`[local, columns / 128, rows]` words), no table; and the [`Descriptor::shared`] shared
///   experts as one MXFP8 matrix pair: `[2 · shared · intermediate, hidden]` and
///   `[hidden, shared · intermediate]` E4M3 codes, row-major, in the same row arrangements, with
///   their scales in the same layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, abi::Layout)]
pub struct FusedLayer {
    /// `[local, 2 * intermediate, hidden / 2]` packed values.
    pub up_gate: *const c_void,
    /// The first projection's block scales.
    pub up_gate_scales: *const c_void,
    /// `[local, hidden, intermediate / 2]` packed values.
    pub down: *const c_void,
    /// The second projection's block scales.
    pub down_scales: *const c_void,
    /// `[local]` f32 that dequantizes the first product.
    pub up_gate_alpha: *const f32,
    /// `[local]` f32 that dequantizes the second product.
    pub down_alpha: *const f32,
    /// `[local]` f32 that encodes the gated rows.
    pub gated_scale: *const f32,
    /// Bytes of one expert's block scales, first projection.
    pub up_gate_scale_bytes: i64,
    /// Bytes of one expert's block scales, second projection.
    pub down_scale_bytes: i64,
    /// MXFP4: the shared experts' `[up; gate]` E4M3 codes, or null without one.
    pub shared_up_gate: *const c_void,
    /// MXFP4: their scales.
    pub shared_up_gate_scales: *const c_void,
    /// MXFP4: the shared experts' down E4M3 codes, or null without one.
    pub shared_down: *const c_void,
    /// MXFP4: their scales.
    pub shared_down_scales: *const c_void,
}

/// One launch of the exchange, declared to the packages as `uniserve_exchange_call`.
///
/// The structures of `expert_exchange` are generated into `uniserve_expert_exchange.h` with their
/// layouts asserted ([`abi`]); every backend includes that one header, so the executor drives any
/// backend through the same eight entries. Dispatch sends this rank's rows and leaves what the rank
/// received in the transport's views. Combine returns the rows the rank's experts computed, which
/// it wrote into the transport's `combined` view, and sums each token's partial results into `out`.
#[repr(C)]
#[derive(abi::Layout)]
pub struct Call {
    pub transport: *mut c_void,
    /// Dispatch, and a fused MXFP4 launch: `[tokens, hidden]` rows to send.
    pub states: *const c_void,
    /// Dispatch: `[tokens, top]` expert ids.
    pub ids: *const i32,
    /// Dispatch: `[tokens, top]` route weights.
    pub weights: *const f32,
    /// `[1]` device count of live rows, or null for all `tokens`.
    pub live: *const i32,
    /// Combine: `[tokens, hidden]` summed rows, written.
    pub out: *mut c_void,
    /// Fused: this layer's expert operands.
    pub layer: *const FusedLayer,
    /// The [`Mode`], by its number.
    pub mode: i64,
    /// Rows of this launch's extent, the captured bucket's.
    pub tokens: i64,
    /// Rows per rank the launch's step is shaped for ([`ExchangeArgs::capacity`]).
    pub capacity: i64,
    pub top: i64,
    pub hidden: i64,
    pub experts: i64,
    pub max_tokens: i64,
    pub ranks: i64,
    pub dtype: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// The worker's host channel to the group, as a package sees it, declared to the packages as
/// `UniserveGroup`.
///
/// One instance per rank and group, built by the worker and alive for the transport's whole life.
/// A package may cache neither the pointer nor a copy of what it points at, and must never free
/// it. Every function is collective over the group and is called from a lifecycle entry only,
/// never from a launch. The ranks of a host are threads of one process, so a function blocks its
/// caller's thread and no other's.
#[repr(C)]
#[derive(abi::Layout)]
pub struct HostGroup {
    /// What the functions below take as their first argument.
    pub handle: *mut c_void,
    /// The group's `ncclComm_t`, for a package that drives NCCL itself, or null.
    pub communicator: *mut c_void,
    /// Ranks of the group.
    pub ranks: i64,
    /// This rank's index in it.
    pub rank: i64,
    /// The CUDA device this rank's transport lives on.
    pub device: i64,
    /// Gathers each rank's `bytes` bytes of `mine` into `peers`, `[ranks][bytes]` in rank order:
    /// peer handles and addresses travel through this.
    pub all_gather: unsafe extern "C" fn(
        handle: *mut c_void,
        mine: *const c_void,
        peers: *mut c_void,
        bytes: i64,
    ) -> i32,
    /// Returns once every rank of the group has called it.
    pub barrier: unsafe extern "C" fn(handle: *mut c_void) -> i32,
    /// Why the last call failed; empty after a successful one.
    pub error: [c_char; MESSAGE],
}

/// What the worker states about a transport it creates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TransportShape {
    /// Ranks of the group.
    pub ranks: u32,
    /// This rank's index in it.
    pub rank: u32,
    /// Experts in the router's vocabulary.
    pub experts: u32,
    /// Hidden elements of one row.
    pub hidden: u32,
    /// An expert's gated width, for a backend that runs the experts.
    pub intermediate: u32,
    /// Rows one rank sends in one step, at most.
    pub max_tokens: u32,
    /// Routes per token.
    pub top: u32,
    /// Element type of the states.
    pub dtype: DType,
    /// The CUDA device of this rank.
    pub device: i32,
    /// Multiprocessors of that device ([`Descriptor::multiprocessors`]).
    pub multiprocessors: u32,
    /// Leading ranks that hold no experts; zero when every rank holds its share.
    pub attention_ranks: u32,
    /// What the layer's fused launch runs beside the exchange; [`FusedExperts::NONE`] for an
    /// exchange of two directions, whose launches run no experts.
    pub fused: FusedExperts,
}

impl TransportShape {
    /// The descriptor of the module that drives a transport of this shape in `mode`: the module
    /// whose `create` makes the transport, and whose launches run over it.
    #[must_use]
    pub const fn descriptor(&self, mode: Mode) -> Descriptor {
        let fused = match mode {
            Mode::Fused => self.fused,
            Mode::Dispatch | Mode::Combine => FusedExperts::NONE,
        };
        Descriptor {
            mode,
            hidden: self.hidden,
            intermediate: self.intermediate,
            top: self.top,
            experts: self.experts,
            max_tokens: self.max_tokens,
            ranks: self.ranks,
            attention: self.attention_ranks,
            multiprocessors: self.multiprocessors,
            dtype: self.dtype,
            weights: fused.weights,
            shared: fused.shared,
            limit: fused.limit,
        }
    }

    /// The lifecycle call of a transport of this shape, before it exists, over `group`.
    fn lifecycle(&self, group: *const HostGroup) -> Result<Lifecycle> {
        Ok(Lifecycle {
            transport: std::ptr::null_mut(),
            group,
            ranks: i64::from(self.ranks),
            rank: i64::from(self.rank),
            experts: i64::from(self.experts),
            hidden: i64::from(self.hidden),
            max_tokens: i64::from(self.max_tokens),
            top: i64::from(self.top),
            dtype: abi::code(self.dtype)?,
            device: i64::from(self.device),
            intermediate: i64::from(self.intermediate),
            capacity: 0,
            attention_ranks: i64::from(self.attention_ranks),
            weights: self.fused.weights as i64,
            shared: i64::from(self.fused.shared),
            limit: self.fused.clamp(),
            error: [0; MESSAGE],
        })
    }
}

/// The transport's lifecycle call, as the executor drives it, declared to the packages as
/// `uniserve_exchange_lifecycle`.
#[repr(C)]
#[derive(abi::Layout)]
pub struct Lifecycle {
    pub transport: *mut c_void,
    /// The executor's channel to the group, alive for the whole package life.
    pub group: *const HostGroup,
    pub ranks: i64,
    pub rank: i64,
    /// Experts in the router's vocabulary.
    pub experts: i64,
    pub hidden: i64,
    /// Rows one rank sends in one step, at most.
    pub max_tokens: i64,
    pub top: i64,
    pub dtype: i64,
    pub device: i64,
    /// An expert's gated width, for a backend that runs the experts.
    pub intermediate: i64,
    /// Step entry: the rows per rank the group agreed for this step.
    pub capacity: i64,
    /// Leading ranks that hold no experts; zero when every rank holds some.
    pub attention_ranks: i64,
    /// What the layer's fused launch runs ([`TransportShape::fused`]): the routed experts'
    /// representation by its [`ExpertWeights`] number, the shared experts and the clamp (zero
    /// for none), which a fused package lays its transport out for.
    pub weights: i64,
    pub shared: i64,
    pub limit: f32,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

/// The views a transport holds, asked for once per transport and again after a restore, whose
/// addresses it must agree with; declared to the packages as `uniserve_exchange_info`.
#[repr(C)]
#[derive(abi::Layout)]
pub struct Info {
    pub transport: *mut c_void,
    /// `[ranks * capacity, hidden]` rows received, in the activation's type.
    pub states: *mut c_void,
    /// `[ranks * capacity, top]` i32.
    pub ids: *mut c_void,
    /// `[ranks * capacity, top]` f32.
    pub weights: *mut c_void,
    /// `[ranks * capacity, hidden]` rows the experts computed.
    pub combined: *mut c_void,
    /// `[1]` i32: how many leading rows of those views this step holds.
    pub live: *mut c_void,
    /// Fused: `[capacity, hidden / 2]` the encoded rows a rank sends.
    pub packed: *mut c_void,
    /// Fused: `[capacity, scale columns]` their block scales.
    pub packed_scales: *mut c_void,
    pub ranks: i64,
    pub rank: i64,
    pub capacity: i64,
    pub hidden: i64,
    pub top: i64,
    pub dtype: i64,
    pub device: i64,
    /// Nonzero when one launch runs the whole expert layer.
    pub fused: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

type LifecycleEntry = unsafe extern "C" fn(*mut Lifecycle, *mut c_void) -> i32;
type InfoEntry = unsafe extern "C" fn(*mut Info, *mut c_void) -> i32;
type FootprintEntry = unsafe extern "C" fn(*mut Lifecycle, *mut i64) -> i32;

impl TransportShape {
    /// The element type and extents of each region a transport of this shape lays out, in the
    /// order [`TransportViews`] names them: `states`, `ids`, `weights`, `combined`, `live`, `packed`
    /// and `packed_scales`.
    pub fn regions(&self) -> [(DType, Vec<u64>); 7] {
        let max_tokens = u64::from(self.max_tokens);
        let rows = u64::from(self.ranks) * max_tokens;
        let hidden = u64::from(self.hidden);
        let top = u64::from(self.top);
        [
            (self.dtype, vec![rows, hidden]),
            (DType::U32, vec![rows, top]),
            (DType::F32, vec![rows, top]),
            (self.dtype, vec![rows, hidden]),
            (DType::U32, vec![1]),
            (DType::U8, vec![max_tokens, hidden / 2]),
            (DType::F8E4M3, vec![max_tokens, hidden / 16]),
        ]
    }
}

/// The views of a transport, as the layer reads them.
#[derive(Clone, Debug)]
pub struct TransportViews {
    /// The transport's handle.
    pub transport: ExchangeTransport,
    /// `[ranks * capacity, hidden]`: the rows this rank received.
    pub states: Operand,
    /// `[ranks * capacity, top]` u32: their expert ids.
    pub ids: Operand,
    /// `[ranks * capacity, top]` f32: their route weights.
    pub weights: Operand,
    /// `[ranks * capacity, hidden]`: where the rank's experts leave their sums.
    pub combined: Operand,
    /// `[1]` u32: the leading rows of those views this step holds.
    pub live: Operand,
    /// `[capacity, hidden / 2]` u8: where a fused launch reads this rank's encoded rows.
    pub packed: Operand,
    /// `[capacity, hidden / 16]` block scales of those rows.
    pub packed_scales: Operand,
    /// Whether one launch runs the whole expert layer.
    pub fused: bool,
}

/// A transport the worker owns: what the package's `create` returned, and the entries that drive
/// it through its life.
pub struct Transport {
    /// Keeps the module loaded while the handle lives.
    _kernel: Arc<Kernel>,
    call: Lifecycle,
    step: LifecycleEntry,
    info: InfoEntry,
    shape: TransportShape,
}

// SAFETY: a transport is created and driven on its device's thread, and moves there with its runner.
unsafe impl Send for Transport {}

impl Transport {
    /// Calls the package's `create` entry, collectively over the group.
    ///
    /// `group` must stay alive until [`Self::release`] returns, because the lifecycle entries reach
    /// the group through it. Every rank of the group calls this at the same point of its startup,
    /// each on its own thread, because `create` blocks until the ranks have met.
    pub fn create(
        kernel: Arc<Kernel>,
        group: *const HostGroup,
        shape: TransportShape,
        stream: &Stream,
    ) -> Result<Self> {
        // SAFETY: each symbol is the package's lifecycle or info entry, which takes the structure
        // its type names and the stream, as every native entry does (`kernel`).
        let (create, step, info) = unsafe {
            (
                kernel.native_symbol::<LifecycleEntry>("uniserve_expert_exchange_create")?,
                kernel.native_symbol::<LifecycleEntry>("uniserve_expert_exchange_step")?,
                kernel.native_symbol::<InfoEntry>("uniserve_expert_exchange_info")?,
            )
        };
        let mut call = shape.lifecycle(group)?;
        let handle = stream.raw() as *mut std::ffi::c_void;
        // The entry allocates the transport's device memory and exchanges its handles on the
        // device the calling thread holds current, as the CUDA runtime it calls reads it
        // (`cudaGetDevice`), so the stream's context is current for the call: the thread that
        // creates a rank's transport may hold no context, or another device's.
        let status = stream
            .with_context(|| {
                // SAFETY: `call` has the layout the header declares, and `group` is the caller's
                // to keep.
                unsafe { create(&mut call, handle) }
            })
            .map_err(Error::Platform)?;
        check(status, "create", &call.error)?;
        Ok(Self {
            _kernel: kernel,
            call,
            step,
            info,
            shape,
        })
    }

    /// The device bytes a transport of `shape` takes once `kernel`'s package creates it, as the
    /// package states them: what a rank divides its device's memory with before the transport
    /// exists. `kernel` is the module of the shape's descriptor in the mode that creates the
    /// transport ([`TransportShape::descriptor`]).
    ///
    /// The entry issues no device work and reaches no peer, so it runs on any thread.
    ///
    /// # Errors
    ///
    /// A module that exports no footprint entry, the package's refusal of the shape, or a
    /// negative statement.
    pub fn footprint(kernel: &Kernel, shape: &TransportShape) -> Result<u64> {
        // SAFETY: the symbol is the package's entry of the signature the header declares for it.
        let footprint = unsafe {
            kernel.native_symbol::<FootprintEntry>("uniserve_expert_exchange_footprint")?
        };
        let mut call = shape.lifecycle(std::ptr::null())?;
        let mut bytes = 0_i64;
        // SAFETY: `call` has the layout the header declares, and the entry writes one `i64`.
        let status = unsafe { footprint(&mut call, &mut bytes) };
        check(status, "footprint", &call.error)?;
        u64::try_from(bytes).map_err(|_| Error::Native {
            entry: "uniserve_expert_exchange_footprint".to_owned(),
            status,
            message: Some(format!("the package stated {bytes} bytes")),
        })
    }

    /// The handle a launch borrows.
    pub fn handle(&self) -> ExchangeTransport {
        // SAFETY: the value `create` returned, alive until `release`.
        unsafe { ExchangeTransport::from_handle(self.call.transport) }
    }

    /// The transport's views, as the package laid them out.
    pub fn views(&self) -> Result<TransportViews> {
        let mut info = Info {
            transport: self.call.transport,
            states: std::ptr::null_mut(),
            ids: std::ptr::null_mut(),
            weights: std::ptr::null_mut(),
            combined: std::ptr::null_mut(),
            live: std::ptr::null_mut(),
            packed: std::ptr::null_mut(),
            packed_scales: std::ptr::null_mut(),
            ranks: 0,
            rank: 0,
            capacity: 0,
            hidden: 0,
            top: 0,
            dtype: 0,
            device: 0,
            fused: 0,
            error: [0; MESSAGE],
        };
        // SAFETY: `info` has the layout the header declares; this entry issues no device work.
        let status = unsafe { (self.info)(&mut info, std::ptr::null_mut()) };
        check(status, "info", &info.error)?;
        let [states, ids, weights, combined, live, packed, packed_scales] = self.shape.regions();
        let view = |pointer: *mut c_void, (dtype, dims): (DType, Vec<u64>)| {
            Operand::new(DevicePtr::from_address(pointer as u64), dtype, &dims)
        };
        Ok(TransportViews {
            transport: self.handle(),
            states: view(info.states, states),
            ids: view(info.ids, ids),
            weights: view(info.weights, weights),
            combined: view(info.combined, combined),
            live: view(info.live, live),
            packed: view(info.packed, packed),
            packed_scales: view(info.packed_scales, packed_scales),
            fused: info.fused != 0,
        })
    }

    /// Writes the step's agreed capacity on `stream`, ahead of the step's first launch.
    pub fn step(&mut self, capacity: u32, stream: &Stream) -> Result<()> {
        self.call.capacity = i64::from(capacity);
        let handle = stream.raw() as *mut std::ffi::c_void;
        // The entry issues its write on the device the calling thread holds current, as `create`
        // does, so the stream's context is current for the call.
        let (step, call) = (self.step, &mut self.call);
        let status = stream
            .with_context(|| {
                // SAFETY: the transport is live and the call has the header's layout.
                unsafe { step(call, handle) }
            })
            .map_err(Error::Platform)?;
        check(status, "step", &self.call.error)
    }
}

/// A lifecycle entry's status, with the message it left.
fn check(status: i32, entry: &str, error: &[c_char; MESSAGE]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    Err(Error::Native {
        entry: format!("uniserve_expert_exchange_{entry}"),
        status,
        message: abi::message(error).map(str::to_owned),
    })
}

/// The expert-parallel exchange.
pub struct ExpertExchange;

impl Op for ExpertExchange {
    const NAME: &'static str = proto::module::operator::EXPERT_EXCHANGE;

    type Descriptor = Descriptor;
    type Args<'a> = ExchangeArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let top = u64::from(descriptor.top);
        let max_tokens = u64::from(descriptor.max_tokens);
        let tokens = u64::from(args.tokens);

        // The transport a module drives is built for one group size; a launch over another group
        // would read and write peer workspace the transport does not have.
        if descriptor.ranks != args.ranks {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the module is compiled for {} ranks, the launch passes {}",
                    descriptor.ranks, args.ranks
                ),
            ));
        }
        if tokens > max_tokens {
            return Err(Error::argument(
                Self::NAME,
                format!("{tokens} rows exceed the transport's {max_tokens}"),
            ));
        }
        if let Some(live) = &args.live {
            live.require_shape(Self::NAME, &[1])?;
            if live.dtype != DType::U32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected a u32 live count, found {}", abi::name(live.dtype)),
                ));
            }
        }
        if args.capacity < args.tokens || u64::from(args.capacity) > max_tokens {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a step of {} rows a rank holds this launch's {tokens} rows and fits the \
                     transport's {max_tokens}",
                    args.capacity
                ),
            ));
        }
        let fused_rows =
            descriptor.mode == Mode::Fused && descriptor.weights == ExpertWeights::Mxfp4;
        if (descriptor.mode == Mode::Fused) == (descriptor.weights == ExpertWeights::None) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a fused launch runs NVFP4 or MXFP4 experts and the two directions run none; \
                     a {} launch states {} experts",
                    descriptor.mode, descriptor.weights
                ),
            ));
        }
        if matches!(descriptor.mode, Mode::Dispatch | Mode::Fused) {
            let reads_rows = descriptor.mode == Mode::Dispatch || fused_rows;
            if reads_rows {
                args.state.require_shape(Self::NAME, &[tokens, hidden])?;
            }
            args.ids.require_shape(Self::NAME, &[tokens, top])?;
            args.weights.require_shape(Self::NAME, &[tokens, top])?;
            if tokens != 0 && reads_rows {
                args.state
                    .require_row(Self::NAME, descriptor.dtype, hidden)?;
            }
            if args.ids.dtype != DType::U32 || args.weights.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    "a dispatch sends u32 expert ids and FP32 route weights",
                ));
            }
        }
        if matches!(descriptor.mode, Mode::Combine | Mode::Fused) {
            args.out.require_shape(Self::NAME, &[tokens, hidden])?;
            if tokens != 0 {
                args.out.require_row(Self::NAME, descriptor.dtype, hidden)?;
            }
        }
        let fused = args.fused;
        if (descriptor.mode == Mode::Fused) != fused.is_some() {
            return Err(Error::argument(
                Self::NAME,
                "a fused launch states its layer's operands, and no other launch does",
            ));
        }
        if let Some(layer) = fused {
            let shared = !layer.shared_up_gate.is_null() && !layer.shared_down.is_null();
            if shared != (descriptor.shared > 0)
                || (descriptor.weights == ExpertWeights::Nvfp4 && descriptor.shared > 0)
            {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the launch runs {} shared experts beside its {} ones (NVFP4 runs none), \
                         and its layer states {} shared weights",
                        descriptor.shared,
                        descriptor.weights,
                        if shared { "" } else { "no" }
                    ),
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
        // The transport is the group's launch resource, like the stream and the kernel: a profile
        // run checks the call before any transport exists, so its presence is a launch rule and
        // not one of `check`'s argument rules.
        let transport = args
            .transport
            .filter(|transport| !transport.handle().is_null())
            .ok_or_else(|| {
                Error::argument(
                    Self::NAME,
                    "the call passed no transport; the exchange runs over the group's workspace",
                )
            })?;
        let hidden = u64::from(descriptor.hidden);
        let top = u64::from(descriptor.top);
        let experts = u64::from(descriptor.experts);
        let max_tokens = u64::from(descriptor.max_tokens);
        let ranks = u64::from(args.ranks);
        let tokens = u64::from(args.tokens);
        let fused = args.fused;

        let mut call = Call {
            transport: transport.handle(),
            states: address(args.state.ptr) as usize as *const c_void,
            ids: address(args.ids.ptr) as usize as *const i32,
            weights: address(args.weights.ptr) as usize as *const f32,
            live: args.live.map_or(std::ptr::null(), |live| {
                address(live.ptr) as usize as *const i32
            }),
            out: address(args.out.ptr) as usize as *mut c_void,
            layer: fused.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
            mode: descriptor.mode as i64,
            tokens: to_i64(Self::NAME, tokens)?,
            capacity: to_i64(Self::NAME, u64::from(args.capacity))?,
            top: to_i64(Self::NAME, top)?,
            hidden: to_i64(Self::NAME, hidden)?,
            experts: to_i64(Self::NAME, experts)?,
            max_tokens: to_i64(Self::NAME, max_tokens)?,
            ranks: to_i64(Self::NAME, ranks)?,
            dtype: abi::code(descriptor.dtype)?,
            error: [0; MESSAGE],
        };
        // SAFETY: `Call` is the call structure this package's entry declares: the declaration is
        // generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        // A dispatch writes only the peer slots inside the transport, which the caller does not see
        // as a tensor, so nothing it returns is compared; a combine writes its own rows.
        if args.out.numel() == 0 {
            return args;
        }
        ExchangeArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        match descriptor.mode {
            // A dispatch only moves rows: two implementations agree bit for bit.
            Mode::Dispatch => Tolerance::exact(),
            // A token's partial expert sums arrive from the ranks that hold its experts, so the
            // combine's reduction length is the group size.
            Mode::Combine => {
                Tolerance::dot(u64::from(descriptor.ranks), DType::F32, descriptor.dtype)
            }
            // A fused launch computes the experts too. Over NVFP4 experts the bound is the
            // combine's, the reduction of a token's partial sums over the group.
            Mode::Fused if descriptor.weights != ExpertWeights::Mxfp4 => {
                Tolerance::dot(u64::from(descriptor.ranks), DType::F32, descriptor.dtype)
            }
            // Over MXFP4 experts: the first product's contraction rounded to BF16, a gated row's
            // code one E4M3 step apart (twice its unit roundoff), the second product's
            // contraction rounded to BF16, and the FP32 sum of a token's routes and shared
            // experts rounded once into the states' type; each a sum that cancels, so the bound
            // is relative to the output's largest magnitude.
            Mode::Fused => {
                let terms = u64::from(descriptor.top + descriptor.shared);
                let first = Tolerance::dot(u64::from(descriptor.hidden), DType::F32, DType::BF16);
                let gated = 2.0 * crate::tolerance::unit_roundoff(DType::F8E4M3) as f32;
                let second =
                    Tolerance::dot(u64::from(descriptor.intermediate), DType::F32, DType::BF16);
                let sum = Tolerance::dot(terms.max(1), DType::F32, descriptor.dtype);
                Tolerance {
                    atol: sum.atol,
                    rtol: first.rtol + gated + second.rtol + sum.rtol,
                    cancels: true,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// The multiprocessors of the device the tests' descriptors state.
    const MULTIPROCESSORS: u32 = 148;

    /// A dispatch whose views meet its descriptor passes `check` before any transport exists, as
    /// a profile run calls it; the transport is required only by a launch.
    #[test]
    fn a_dispatch_without_a_transport_passes_its_check() {
        let descriptor = Descriptor {
            mode: Mode::Dispatch,
            hidden: 64,
            intermediate: 128,
            top: 2,
            experts: 8,
            max_tokens: 16,
            ranks: 2,
            attention: 0,
            multiprocessors: MULTIPROCESSORS,
            dtype: DType::BF16,
            weights: ExpertWeights::None,
            shared: 0,
            limit: 0.0_f32.into(),
        };
        let view = |dtype, dims: &[u64]| {
            Operand::new(platform::DevicePtr::from_address(0x1000), dtype, dims)
        };
        let args = ExchangeArgs {
            transport: None,
            tokens: 4,
            capacity: 4,
            state: view(DType::BF16, &[4, 64]),
            ids: view(DType::U32, &[4, 2]),
            weights: view(DType::F32, &[4, 2]),
            live: None,
            out: view(DType::BF16, &[4, 64]),
            fused: None,
            ranks: 2,
        };
        ExpertExchange::check(&descriptor, &args).unwrap();

        // The argument rules still hold without a transport: a group of another size is refused,
        // and so is a step that does not hold the launch's rows.
        let other = ExchangeArgs {
            ranks: 4,
            ..args.clone()
        };
        assert!(ExpertExchange::check(&descriptor, &other).is_err());
        let short = ExchangeArgs {
            capacity: 3,
            ..args
        };
        assert!(ExpertExchange::check(&descriptor, &short).is_err());
    }

    /// A fused MXFP4 launch reads the rows it encodes, and states its shared expert's weights
    /// exactly when its descriptor runs one.
    #[test]
    fn a_fused_mxfp4_launch_reads_its_rows_and_states_its_shared_expert() {
        let descriptor = Descriptor {
            mode: Mode::Fused,
            hidden: 5120,
            intermediate: 2304,
            top: 6,
            experts: 384,
            max_tokens: 64,
            ranks: 4,
            attention: 0,
            multiprocessors: 152,
            dtype: DType::BF16,
            weights: ExpertWeights::Mxfp4,
            shared: 1,
            limit: 10.0_f32.into(),
        };
        assert_eq!(
            descriptor.to_string(),
            "mode_fused_hidden_5120_intermediate_2304_top_6_experts_384_max_tokens_64_ranks_4_\
             attention_0_multiprocessors_152_dtype_bf16_weights_mxfp4_shared_1_limit_0x41200000"
        );
        let view = |dtype, dims: &[u64]| {
            Operand::new(platform::DevicePtr::from_address(0x1000), dtype, dims)
        };
        let some = 0x2000 as *const c_void;
        let layer = FusedLayer {
            up_gate: some,
            up_gate_scales: some,
            down: some,
            down_scales: some,
            up_gate_alpha: std::ptr::null(),
            down_alpha: std::ptr::null(),
            gated_scale: std::ptr::null(),
            up_gate_scale_bytes: 4608 * 40 * 4,
            down_scale_bytes: 5120 * 18 * 4,
            shared_up_gate: some,
            shared_up_gate_scales: some,
            shared_down: some,
            shared_down_scales: some,
        };
        let args = ExchangeArgs {
            transport: None,
            tokens: 8,
            capacity: 16,
            state: view(DType::BF16, &[8, 5120]),
            ids: view(DType::U32, &[8, 6]),
            weights: view(DType::F32, &[8, 6]),
            live: None,
            out: view(DType::BF16, &[8, 5120]),
            fused: Some(layer),
            ranks: 4,
        };
        ExpertExchange::check(&descriptor, &args).unwrap();

        // The rows it encodes are part of the call.
        let unread = ExchangeArgs {
            state: Operand::unread(),
            ..args.clone()
        };
        assert!(ExpertExchange::check(&descriptor, &unread).is_err());

        // A layer without its shared expert's weights is refused.
        let unshared = ExchangeArgs {
            fused: Some(FusedLayer {
                shared_up_gate: std::ptr::null(),
                shared_down: std::ptr::null(),
                ..layer
            }),
            ..args
        };
        assert!(ExpertExchange::check(&descriptor, &unshared).is_err());
    }

    use super::*;

    #[test]
    fn dispatch_is_exact_and_combine_carries_the_rank_sum() {
        let dispatch = Descriptor {
            mode: Mode::Dispatch,
            hidden: 2048,
            intermediate: 768,
            top: 8,
            experts: 128,
            max_tokens: 8192,
            ranks: 4,
            attention: 0,
            multiprocessors: MULTIPROCESSORS,
            dtype: DType::BF16,
            weights: ExpertWeights::None,
            shared: 0,
            limit: 0.0_f32.into(),
        };
        assert_eq!(
            ExpertExchange::tolerance(&dispatch),
            Tolerance::exact(),
            "a dispatch only moves rows"
        );
        let combine = Descriptor {
            mode: Mode::Combine,
            ..dispatch
        };
        assert_eq!(
            ExpertExchange::tolerance(&combine),
            Tolerance::dot(4, DType::F32, DType::BF16),
            "the combine sums one term per rank of the group"
        );
    }

    #[test]
    fn the_group_size_is_part_of_the_module_key() {
        let combine = Descriptor {
            mode: Mode::Combine,
            hidden: 2048,
            intermediate: 768,
            top: 8,
            experts: 128,
            max_tokens: 8192,
            ranks: 4,
            attention: 0,
            multiprocessors: MULTIPROCESSORS,
            dtype: DType::BF16,
            weights: ExpertWeights::None,
            shared: 0,
            limit: 0.0_f32.into(),
        };
        assert_eq!(
            combine.to_string(),
            "mode_combine_hidden_2048_intermediate_768_top_8_experts_128_max_tokens_8192_ranks_4_attention_0_\
             multiprocessors_148_dtype_bf16_weights_none_shared_0_limit_0x0"
        );
        let wider = Descriptor {
            ranks: 8,
            ..combine
        };
        assert_ne!(wider.to_string(), combine.to_string());
    }
}
