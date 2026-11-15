//! Expert-parallel exchange: moving routes to the ranks that hold their experts.
//!
//! A rank routes every token to `top` logical experts, of which only its
//! `[local_offset, local_offset + local_experts)` window is resident. The exchange carries each
//! route's payload — the hidden state, the expert id and the route weight — to the rank that holds
//! the expert (dispatch), and returns each token's partial expert sums to the rank that owns the
//! token (combine). `moe_config.exchange` names the backend (`alltoall`, `deepep`, `megamoe`), each a
//! native library because the transport's workspace and its handshake are the package's host code.
//! A backend whose one launch runs the whole expert layer states so in [`Info::fused`] and takes
//! [`Mode::Fused`] instead of the two directions.
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
    /// Multiprocessors of the device the exchange runs on ([`multiprocessors`]). A backend whose
    /// kernels spread over the whole device compiles its grid for them: a persistent grid that
    /// synchronizes grid-wide needs every block resident, so it is sized to the device's count.
    pub multiprocessors: u32,
    /// Element type of the states.
    pub dtype: DType,
}

impl Descriptor {
    /// The arrangement of the `[up; gate]` stack's packed rows a fused launch reads
    /// ([`FusedLayer::up_gate`]): each thirty-two-row block leads with the gate rows.
    pub const FUSED_UP_GATE_VALUES: ValueLayout = ValueLayout::GateFirst;

    /// The arrangement of the `down` stack's packed rows a fused launch reads
    /// ([`FusedLayer::down`]): the checkpoint's own.
    pub const FUSED_DOWN_VALUES: ValueLayout = ValueLayout::Plain;

    /// The arrangement of both stacks' block scales a fused launch reads.
    pub const FUSED_SCALES: ScaleLayout = ScaleLayout::Swizzled128X4;

    /// The arrangement of the exchanged tokens' block scales a fused launch reads.
    pub const FUSED_TOKEN_SCALES: ScaleLayout = ScaleLayout::Linear;
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
            .finish(formatter)
    }
}

/// The multiprocessors of `device` (`platform::DeviceAttribute::MultiprocessorCount`), as an
/// exchange descriptor and a transport shape state them.
///
/// # Errors
///
/// [`Error::Platform`] when the driver cannot report the device's count, and
/// [`Error::Unsupported`] for a count that is not positive.
pub fn multiprocessors(device: platform::DeviceId) -> Result<u32> {
    let count = device.attribute(platform::DeviceAttribute::MultiprocessorCount)?;
    u32::try_from(count)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| Error::Unsupported {
            operation: ExpertExchange::NAME,
            reason: format!("device {} reports {count} multiprocessors", device.0),
        })
}

/// The arguments of one exchange launch.
///
/// At [`Mode::Dispatch`] the rank sends its rows in `state`, `ids` and `weights`; what it received
/// stands in the transport's views afterwards. At [`Mode::Combine`] the rows its experts computed
/// stand in the transport's combined view, and the package writes the rank's own rows into `out`.
/// At [`Mode::Fused`] one launch reads `state`, `ids` and `weights` and writes `out`, with the
/// layer's expert operands in `fused`. Views a mode does not read are empty (`[0]`-element) views.
#[derive(Clone)]
pub struct ExchangeArgs {
    /// The group's transport, created and owned by the worker through the package and stated by
    /// the execution context at the launch. `None` in a call that is checked before any
    /// transport exists (a profile run); a launch refuses it.
    pub transport: Option<ExchangeTransport>,
    /// Rows of this launch's extent: the captured bucket's, of which `live` are the batch's.
    pub tokens: u32,
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
/// Each weight is `[local experts, rows, columns / 2]` packed values, read K-major, with its block
/// scales in a table of `scale_bytes` per expert. Every address is a device address the layer's
/// weights hold for the life of the model.
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
    /// Dispatch: `[tokens, hidden]` rows to send.
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
    /// Multiprocessors of that device ([`multiprocessors`]).
    pub multiprocessors: u32,
    /// Leading ranks that hold no experts; zero when every rank holds its share.
    pub attention_ranks: u32,
}

impl TransportShape {
    /// The descriptor of the module that drives a transport of this shape in `mode`: the module
    /// whose `create` makes the transport, and whose launches run over it.
    #[must_use]
    pub const fn descriptor(&self, mode: Mode) -> Descriptor {
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
        if matches!(descriptor.mode, Mode::Dispatch | Mode::Fused) {
            if descriptor.mode == Mode::Dispatch {
                args.state.require_shape(Self::NAME, &[tokens, hidden])?;
            }
            args.ids.require_shape(Self::NAME, &[tokens, top])?;
            args.weights.require_shape(Self::NAME, &[tokens, top])?;
            if tokens != 0 && descriptor.mode == Mode::Dispatch {
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
            Mode::Combine | Mode::Fused => {
                Tolerance::dot(u64::from(descriptor.ranks), DType::F32, descriptor.dtype)
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
        };
        let view = |dtype, dims: &[u64]| {
            Operand::new(platform::DevicePtr::from_address(0x1000), dtype, dims)
        };
        let args = ExchangeArgs {
            transport: None,
            tokens: 4,
            state: view(DType::BF16, &[4, 64]),
            ids: view(DType::U32, &[4, 2]),
            weights: view(DType::F32, &[4, 2]),
            live: None,
            out: view(DType::BF16, &[4, 64]),
            fused: None,
            ranks: 2,
        };
        ExpertExchange::check(&descriptor, &args).unwrap();

        // The argument rules still hold without a transport: a group of another size is refused.
        let other = ExchangeArgs { ranks: 4, ..args };
        assert!(ExpertExchange::check(&descriptor, &other).is_err());
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
        };
        assert_eq!(
            combine.to_string(),
            "mode_combine_hidden_2048_intermediate_768_top_8_experts_128_max_tokens_8192_ranks_4_attention_0_\
             multiprocessors_148_dtype_bf16"
        );
        let wider = Descriptor {
            ranks: 8,
            ..combine
        };
        assert_ne!(wider.to_string(), combine.to_string());
    }
}
