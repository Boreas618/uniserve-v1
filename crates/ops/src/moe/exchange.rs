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
//! ([`Transport`]), keeps it across a standby's suspend and restore, and hands it to this operator as
//! a borrowed handle ([`ExchangeTransport`]); `ops` launches over a handle and never sizes one.
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

use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, DevicePtr, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::{address, dtype_code, to_i64};

/// Which direction of the exchange one module performs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// Send every route to the rank holding its expert.
    Dispatch,
    /// Return every token's partial expert sums to the rank holding the token.
    Combine,
    /// Dispatch, run this layer's experts and combine, in one launch.
    Fused,
}

impl Mode {
    /// The number the package's header gives this direction.
    const fn number(self) -> i64 {
        match self {
            Self::Dispatch => 0,
            Self::Combine => 1,
            Self::Fused => 2,
        }
    }
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
    /// Element type of the states.
    pub dtype: DType,
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
            .field("dtype", self.dtype)
            .finish(formatter)
    }
}

/// The arguments of one exchange launch.
///
/// At [`Mode::Dispatch`] the rank sends its rows in `state`, `ids` and `weights`; what it received
/// stands in the transport's views afterwards. At [`Mode::Combine`] the rows its experts computed
/// stand in the transport's combined view, and the package writes the rank's own rows into `out`.
/// At [`Mode::Fused`] one launch reads `state`, `ids` and `weights` and writes `out`, with the
/// layer's expert operands in `fused`. Views a mode does not read are empty (`[0]`-element) views.
#[derive(Clone, Copy)]
pub struct ExchangeArgs {
    /// The group's transport, created and owned by the worker through the package.
    pub transport: ExchangeTransport,
    /// Rows of this launch's extent: the captured bucket's, of which `live` are the batch's.
    pub tokens: u32,
    /// `[tokens, hidden]` states to send.
    pub state: TensorView,
    /// `[tokens, top]` i32 expert ids.
    pub ids: TensorView,
    /// `[tokens, top]` FP32 route weights.
    pub weights: TensorView,
    /// `[1]` i32 on the device: the rows of `tokens` the batch holds, or `None` for all of them.
    pub live: Option<TensorView>,
    /// `[tokens, hidden]` combined rows, written.
    pub out: TensorView,
    /// The layer's expert operands, for a fused launch.
    pub fused: Option<FusedLayer>,
    /// Ranks of the expert group.
    pub ranks: u32,
}

/// What a fused launch reads of one layer: its experts, as the loader laid them out.
///
/// Mirrored by hand in the package's header as `uniserve_fused_layer`. Every address is a device
/// address the layer's weights hold for the life of the model.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FusedLayer {
    /// `[local, 2 * intermediate, hidden / 2]` packed values.
    pub up_gate: u64,
    /// The first projection's block scales.
    pub up_gate_scales: u64,
    /// `[local, hidden, intermediate / 2]` packed values.
    pub down: u64,
    /// The second projection's block scales.
    pub down_scales: u64,
    /// `[local]` f32 that dequantizes the first product.
    pub up_gate_alpha: u64,
    /// `[local]` f32 that dequantizes the second product.
    pub down_alpha: u64,
    /// `[local]` f32 that encodes the gated rows.
    pub gated_scale: u64,
    /// Bytes of one expert's block scales, first projection.
    pub up_gate_scale_bytes: i64,
    /// Bytes of one expert's block scales, second projection.
    pub down_scale_bytes: i64,
}

/// The call structure the package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/expert_exchange/<backend>/include/uniserve_expert_exchange.h`.
#[repr(C)]
pub struct Call {
    pub transport: *mut c_void,
    pub states: *const c_void,
    pub ids: *const i32,
    pub weights: *const f32,
    pub live: *const i32,
    pub out: *mut c_void,
    pub layer: *const FusedLayer,
    pub mode: i64,
    pub tokens: i64,
    pub top: i64,
    pub hidden: i64,
    pub experts: i64,
    pub max_tokens: i64,
    pub ranks: i64,
    pub dtype: i64,
    pub error: [u8; 512],
}

/// The size and field offsets the header must mirror, in bytes.
pub const CALL_LAYOUT: CallLayout = CallLayout {
    size: 632,
    error: 120,
};

/// The layout of one exchange structure, for its header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct CallLayout {
    pub size: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<Call>() == CALL_LAYOUT.size,
    "the expert_exchange call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, error) == CALL_LAYOUT.error,
    "the expert_exchange call structure's `error` moved without its header"
);

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        message(&self.error)
    }
}

fn message(error: &[u8; 512]) -> Option<&str> {
    let end = error.iter().position(|&byte| byte == 0)?;
    std::str::from_utf8(&error[..end]).ok()
}

/// The worker's host channel to the group, as a package sees it.
///
/// One instance per rank and group, built by the worker and alive for the transport's whole life.
/// Mirrored by hand in the same header as `UniserveGroup`.
#[repr(C)]
pub struct Group {
    /// What the functions below take as their first argument.
    pub handle: *mut c_void,
    /// The group's `ncclComm_t`, for a package that drives NCCL itself, or null.
    pub communicator: *mut c_void,
    pub ranks: i64,
    pub rank: i64,
    pub device: i64,
    /// Gathers each rank's `bytes` bytes of `mine` into `peers`, `[ranks][bytes]` in rank order.
    pub all_gather: unsafe extern "C" fn(
        handle: *mut c_void,
        mine: *const c_void,
        peers: *mut c_void,
        bytes: i64,
    ) -> i32,
    /// Returns once every rank of the group has called it.
    pub barrier: unsafe extern "C" fn(handle: *mut c_void) -> i32,
    pub error: [u8; 512],
}

/// The size and field offsets the group's header must mirror, in bytes.
pub const GROUP_LAYOUT: CallLayout = CallLayout {
    size: 568,
    error: 56,
};

const _: () = assert!(std::mem::size_of::<Group>() == GROUP_LAYOUT.size);
const _: () = assert!(std::mem::offset_of!(Group, error) == GROUP_LAYOUT.error);

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
    /// Leading ranks that hold no experts; zero when every rank holds its share.
    pub attention_ranks: u32,
}

impl TransportShape {
    /// What a fused backend holds beside its rows: its counters and its window's granule.
    const FUSED_FIXED_BYTES: u64 = 64 << 20;
    /// What any other backend holds beside its rows. A device communicator takes this: creating
    /// deepep's took about 1.6 GB of the device with 32 MB of rows, and alltoall takes none of it,
    /// but the process learns which backend it runs only when the epoch's programs arrive.
    const FIXED_BYTES: u64 = 2 << 30;

    /// The device bytes a rank's transport holds at most, for the capacity plan.
    ///
    /// The plan fits the cache before the process holds a program, so this cannot ask a package:
    /// it is a bound the worker compares with what the transport took when it is created. A
    /// backend whose experts run in the exchange (`fused`) holds its staged rows; any other holds
    /// the rows received and computed beside a window of the rows in flight.
    #[must_use]
    pub fn device_bytes(&self, fused: bool) -> u64 {
        let ranks = u64::from(self.ranks);
        let tokens = u64::from(self.max_tokens);
        let hidden = u64::from(self.hidden);
        let top = u64::from(self.top);
        let rows = ranks * tokens;
        if fused {
            // Every route of every rank may land here, with a tile of padding per local expert.
            // The kernel stages each one encoded, gated, and as a row of the combine's wire type.
            let local = u64::from(self.experts)
                / u64::from(self.ranks.saturating_sub(self.attention_ranks).max(1));
            let pool = rows * top + local * 64;
            let staged = pool * (3 * hidden + u64::from(self.intermediate));
            // Each route's partial row comes back in the shared window with its scales, and the
            // rows this rank sends sit beside them, encoded.
            let shared = tokens * top * hidden * 9 / 4;
            let sent = tokens * (hidden / 2 + hidden / 16 + top * 12);
            staged + shared + sent + Self::FUSED_FIXED_BYTES
        } else {
            let element = self.dtype.bytes().unwrap_or(2);
            let received = rows * (2 * hidden * element + top * 24);
            let window = (rows + ranks) * (hidden * element + 256);
            received + window + Self::FIXED_BYTES
        }
    }
}

/// The transport's lifecycle call, as the header declares it.
#[repr(C)]
pub struct Lifecycle {
    pub transport: *mut c_void,
    pub group: *const Group,
    pub ranks: i64,
    pub rank: i64,
    pub experts: i64,
    pub hidden: i64,
    pub max_tokens: i64,
    pub top: i64,
    pub dtype: i64,
    pub device: i64,
    pub intermediate: i64,
    pub capacity: i64,
    pub attention_ranks: i64,
    pub error: [u8; 512],
}

/// The size and field offsets the lifecycle's header must mirror, in bytes.
pub const LIFECYCLE_LAYOUT: CallLayout = CallLayout {
    size: 616,
    error: 104,
};

const _: () = assert!(std::mem::size_of::<Lifecycle>() == LIFECYCLE_LAYOUT.size);
const _: () = assert!(std::mem::offset_of!(Lifecycle, error) == LIFECYCLE_LAYOUT.error);

/// The views a transport holds, as the header declares them.
#[repr(C)]
pub struct Info {
    pub transport: *mut c_void,
    pub states: *mut c_void,
    pub ids: *mut c_void,
    pub weights: *mut c_void,
    pub combined: *mut c_void,
    pub live: *mut c_void,
    pub packed: *mut c_void,
    pub packed_scales: *mut c_void,
    pub ranks: i64,
    pub rank: i64,
    pub capacity: i64,
    pub hidden: i64,
    pub top: i64,
    pub dtype: i64,
    pub device: i64,
    pub fused: i64,
    pub error: [u8; 512],
}

/// The size and field offsets the info's header must mirror, in bytes.
pub const INFO_LAYOUT: CallLayout = CallLayout {
    size: 640,
    error: 128,
};

const _: () = assert!(std::mem::size_of::<Info>() == INFO_LAYOUT.size);
const _: () = assert!(std::mem::offset_of!(Info, error) == INFO_LAYOUT.error);

type LifecycleEntry = unsafe extern "C" fn(*mut Lifecycle, *mut c_void) -> i32;
type InfoEntry = unsafe extern "C" fn(*mut Info, *mut c_void) -> i32;

/// The views of a transport, as the layer reads them.
#[derive(Clone, Copy, Debug)]
pub struct TransportViews {
    /// The transport's handle.
    pub transport: ExchangeTransport,
    /// `[ranks * capacity, hidden]`: the rows this rank received.
    pub states: TensorView,
    /// `[ranks * capacity, top]` i32: their expert ids.
    pub ids: TensorView,
    /// `[ranks * capacity, top]` f32: their route weights.
    pub weights: TensorView,
    /// `[ranks * capacity, hidden]`: where the rank's experts leave their sums.
    pub combined: TensorView,
    /// `[1]` i32: the leading rows of those views this step holds.
    pub live: TensorView,
    /// `[capacity, hidden / 2]` u8: where a fused launch reads this rank's encoded rows.
    pub packed: TensorView,
    /// `[capacity, hidden / 16]` block scales of those rows.
    pub packed_scales: TensorView,
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
        group: *const Group,
        shape: TransportShape,
        stream: &Stream,
    ) -> Result<Self> {
        // SAFETY: each symbol is the package's entry of the signature the header declares for it.
        let (create, step, info) = unsafe {
            (
                kernel.native_symbol::<LifecycleEntry>("uniserve_expert_exchange_create")?,
                kernel.native_symbol::<LifecycleEntry>("uniserve_expert_exchange_step")?,
                kernel.native_symbol::<InfoEntry>("uniserve_expert_exchange_info")?,
            )
        };
        let mut call = Lifecycle {
            transport: std::ptr::null_mut(),
            group,
            ranks: i64::from(shape.ranks),
            rank: i64::from(shape.rank),
            experts: i64::from(shape.experts),
            hidden: i64::from(shape.hidden),
            max_tokens: i64::from(shape.max_tokens),
            top: i64::from(shape.top),
            dtype: dtype_code(shape.dtype),
            device: i64::from(shape.device),
            intermediate: i64::from(shape.intermediate),
            capacity: 0,
            attention_ranks: i64::from(shape.attention_ranks),
            error: [0; 512],
        };
        let handle = crate::platform::stream_handle(stream);
        // SAFETY: `call` has the layout the header declares, and `group` is the caller's to keep.
        let status = unsafe { create(&mut call, handle) };
        check(status, "create", &call.error)?;
        Ok(Self {
            _kernel: kernel,
            call,
            step,
            info,
            shape,
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
            error: [0; 512],
        };
        // SAFETY: `info` has the layout the header declares; this entry issues no device work.
        let status = unsafe { (self.info)(&mut info, std::ptr::null_mut()) };
        check(status, "info", &info.error)?;
        let shape = self.shape;
        let max_tokens = u64::from(shape.max_tokens);
        let rows = u64::from(shape.ranks) * max_tokens;
        let hidden = u64::from(shape.hidden);
        let top = u64::from(shape.top);
        let view = |pointer: *mut c_void, dtype: DType, dims: &[u64]| {
            TensorView::new(DevicePtr::new(pointer as u64), dtype, Shape::new(dims))
        };
        Ok(TransportViews {
            transport: self.handle(),
            states: view(info.states, shape.dtype, &[rows, hidden]),
            ids: view(info.ids, DType::I32, &[rows, top]),
            weights: view(info.weights, DType::F32, &[rows, top]),
            combined: view(info.combined, shape.dtype, &[rows, hidden]),
            live: view(info.live, DType::I32, &[1]),
            packed: view(info.packed, DType::U8, &[max_tokens, hidden / 2]),
            packed_scales: view(
                info.packed_scales,
                DType::F8E4M3,
                &[max_tokens, hidden / 16],
            ),
            fused: info.fused != 0,
        })
    }

    /// Writes the step's agreed capacity on `stream`, ahead of the step's first launch.
    pub fn step(&mut self, capacity: u32, stream: &Stream) -> Result<()> {
        self.call.capacity = i64::from(capacity);
        let handle = crate::platform::stream_handle(stream);
        // SAFETY: the transport is live and the call has the header's layout.
        let status = unsafe { (self.step)(&mut self.call, handle) };
        check(status, "step", &self.call.error)
    }



}

/// A lifecycle entry's status, with the message it left.
fn check(status: i32, entry: &str, error: &[u8; 512]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    Err(Error::Native {
        entry: format!("uniserve_expert_exchange_{entry}"),
        status,
        message: message(error).map(str::to_owned),
    })
}

/// The expert-parallel exchange.
pub struct ExpertExchange;

impl Op for ExpertExchange {
    const NAME: &'static str = "expert_exchange";

    type Descriptor = Descriptor;
    type Args<'a> = ExchangeArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let hidden = u64::from(descriptor.hidden);
        let top = u64::from(descriptor.top);
        let experts = u64::from(descriptor.experts);
        let max_tokens = u64::from(descriptor.max_tokens);
        let ranks = u64::from(args.ranks);
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
        if args.transport.handle().is_null() {
            return Err(Error::argument(
                Self::NAME,
                "the call passed no transport; the exchange runs over the group's workspace",
            ));
        }
        if tokens > max_tokens {
            return Err(Error::argument(
                Self::NAME,
                format!("{tokens} rows exceed the transport's {max_tokens}"),
            ));
        }
        if let Some(live) = args.live {
            live.require_shape(Self::NAME, Shape::new(&[1]))?;
            if live.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected an i32 live count, found {}", live.dtype),
                ));
            }
        }
        if matches!(descriptor.mode, Mode::Dispatch | Mode::Fused) {
            if descriptor.mode == Mode::Dispatch {
                args.state
                    .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
            }
            args.ids
                .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
            args.weights
                .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
            if tokens != 0 && descriptor.mode == Mode::Dispatch {
                args.state
                    .require_row(Self::NAME, descriptor.dtype, hidden)?;
            }
            if args.ids.dtype != DType::I32 || args.weights.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    "a dispatch sends i32 expert ids and FP32 route weights",
                ));
            }
        }
        if matches!(descriptor.mode, Mode::Combine | Mode::Fused) {
            args.out
                .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
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
        let mut call = Call {
            transport: args.transport.handle(),
            states: address(args.state.ptr) as usize as *const c_void,
            ids: address(args.ids.ptr) as usize as *const i32,
            weights: address(args.weights.ptr) as usize as *const f32,
            live: args.live.map_or(std::ptr::null(), |live| {
                address(live.ptr) as usize as *const i32
            }),
            out: address(args.out.ptr) as usize as *mut c_void,
            layer: fused.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
            mode: descriptor.mode.number(),
            tokens: to_i64(Self::NAME, tokens)?,
            top: to_i64(Self::NAME, top)?,
            hidden: to_i64(Self::NAME, hidden)?,
            experts: to_i64(Self::NAME, experts)?,
            max_tokens: to_i64(Self::NAME, max_tokens)?,
            ranks: to_i64(Self::NAME, ranks)?,
            dtype: dtype_code(descriptor.dtype),
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
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
    /// A shape of the deployment the bound was measured on.
    fn measured(max_tokens: u32, ranks: u32, attention_ranks: u32) -> TransportShape {
        TransportShape {
            ranks,
            rank: 0,
            experts: 128,
            hidden: 2048,
            intermediate: 768,
            max_tokens,
            top: 8,
            dtype: DType::BF16,
            device: 0,
            attention_ranks,
        }
    }

    #[test]
    fn the_transport_bound_holds_what_the_backends_were_measured_to_take() {
        // The fused kernel's local and shared workspaces as it states them, by rows and ranks.
        for (tokens, ranks, local, shared) in [
            (128, 2, 10_877_312_u64, 4_720_256_u64),
            (512, 2, 71_539_712, 18_876_032),
            (4096, 2, 405_283_200, 150_996_608),
            (16384, 2, 1_549_547_904, 603_981_440),
            (512, 8, 196_771_328, 18_875_648),
            (16384, 8, 6_108_804_608, 603_981_056),
        ] {
            let bound = measured(tokens, ranks, 0).device_bytes(true);
            assert!(
                bound >= local + shared,
                "{tokens} rows of {ranks} ranks: {bound}"
            );
        }
        // Creating deepep's transport took 1.65 GB of the device at 512 rows over three ranks.
        assert!(measured(512, 3, 1).device_bytes(false) >= 1_654_652_928);
    }

    use super::*;

    #[test]
    fn call_layouts_are_the_ones_the_header_declares() {
        assert_eq!(std::mem::size_of::<Call>(), CALL_LAYOUT.size);
        assert_eq!(std::mem::offset_of!(Call, error), CALL_LAYOUT.error);
        assert_eq!(std::mem::offset_of!(Call, transport), 0);
    }

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
            dtype: DType::BF16,
        };
        assert_eq!(
            combine.to_string(),
            "mode_combine_hidden_2048_intermediate_768_top_8_experts_128_max_tokens_8192_ranks_4_attention_0_dtype_bf16"
        );
        let wider = Descriptor {
            ranks: 8,
            ..combine
        };
        assert_ne!(wider.to_string(), combine.to_string());
    }

    /// Every package header that serves this operator states the layout this crate passes.
    ///
    /// The mirror is written by hand on both sides, so a field that moves on one side fails on the
    /// other: the module's entry reads these offsets, and a mismatch is a wrong number rather than a
    /// refusal.
    #[test]
    fn the_headers_declare_these_layouts() {
        for (structure, layout) in [
            ("uniserve_exchange_call", &CALL_LAYOUT),
            ("UniserveGroup", &GROUP_LAYOUT),
            ("uniserve_exchange_lifecycle", &LIFECYCLE_LAYOUT),
            ("uniserve_exchange_info", &INFO_LAYOUT),
        ] {
            crate::call_layout_check::check(
                "expert_exchange",
                structure,
                &[("error", layout.error)],
                layout.size,
            );
        }
    }
}
