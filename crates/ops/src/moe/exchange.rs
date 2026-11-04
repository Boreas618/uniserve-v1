//! Expert-parallel exchange: moving routes to the ranks that hold their experts.
//!
//! A rank routes every token to `top` logical experts, of which only its
//! `[local_offset, local_offset + local_experts)` window is resident. The exchange carries each
//! route's payload — the hidden state, the expert id and the route weight — to the rank that holds
//! the expert (dispatch), and returns each token's partial expert sums to the rank that owns the
//! token (combine). Both directions are one launch; `moe_config.exchange` names the backend
//! (`alltoall`, `deepep`, `megamoe`), each a native library because the transport's workspace, its
//! handshake and its per-step capacity are the package's host code
//! (`uniserve/backends/experts/{alltoall,deepep,megamoe}.py`).
//!
//! # Who owns the transport
//!
//! The transport is a device-resident workspace shared between the ranks of one expert group
//! (MNNVL peer mappings for `alltoall`). It is an execution resource, so the executor creates it
//! through the package's own lifecycle entries, keeps it across a standby's suspend and restore, and
//! hands it to this operator as a borrowed handle ([`ExchangeTransport`]); `ops` never creates,
//! sizes or destroys one. The per-step capacity — how many rows each rank may send — is agreed each
//! step by the executor's host group (`uniserve/executor/experts/agreement.py` today) and reaches
//! the package in the call structure.
//!
//! # The bound
//!
//! Dispatch only moves rows: it is exact. Combine sums at most one partial per rank in FP32 and
//! rounds once, so two implementations differ by the `ranks`-term reduction and the output's
//! rounding, stated as [`Tolerance::dot`].

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
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
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Dispatch => "dispatch",
            Self::Combine => "combine",
        })
    }
}

/// The transport an exchange runs over: a borrowed, non-owning handle.
///
/// The package's lifecycle entries create it on the device and keep it for the life of the expert
/// group; the executor owns that lifetime and passes the handle here. A handle is only valid on the
/// device and with the group it was created for, and only between its creation and its release.
#[derive(Clone, Copy)]
pub struct ExchangeTransport {
    handle: *mut c_void,
}

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
    /// Routes per token.
    pub top: u32,
    /// Experts in the router's vocabulary.
    pub experts: u32,
    /// Rows one rank may hold, the transport's capacity.
    pub max_tokens: u32,
    /// Ranks of the expert group whose transport this module drives.
    pub ranks: u32,
    /// Element type of the states.
    pub dtype: DType,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("mode", self.mode)
            .field("hidden", self.hidden)
            .field("top", self.top)
            .field("experts", self.experts)
            .field("max_tokens", self.max_tokens)
            .field("ranks", self.ranks)
            .field("dtype", self.dtype)
            .finish(formatter)
    }
}

/// The arguments of one exchange launch.
///
/// At [`Mode::Dispatch`] the rank sends its live rows in `state`, `ids` and `weights`, and the
/// package fills the peer slots and their counts. At [`Mode::Combine`] it passes the partials it
/// received in `state` and the counts of them in `counts`, and the package writes its own rows into
/// `out`. Views a mode does not read are empty (`[0]`-element) views.
pub struct ExchangeArgs {
    /// The group's transport, created and owned by the executor through the package.
    pub transport: ExchangeTransport,
    /// Live rows this rank sends at dispatch, or the capacity it expects at combine.
    pub tokens: u32,
    /// Dispatch: `[tokens, hidden]` states to send. Combine: `[max_tokens, hidden]` received partials.
    pub state: TensorView,
    /// Dispatch: `[tokens, top]` i32 expert ids. Empty at combine.
    pub ids: TensorView,
    /// Dispatch: `[tokens, top]` FP32 route weights. Empty at combine.
    pub weights: TensorView,
    /// Dispatch: empty. Combine: `[tokens, hidden]` combined rows, written.
    pub out: TensorView,
    /// `[ranks]` i32: rows each peer sent (dispatch, written) or received (combine, read).
    pub counts: TensorView,
    /// Ranks of the expert group.
    pub ranks: u32,
}

/// The call structure the dispatch package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/expert_exchange/<backend>/include/uniserve_expert_exchange.h`.
#[repr(C)]
pub struct DispatchCall {
    pub transport: *mut c_void,
    pub states: *const c_void,
    pub ids: *const i32,
    pub weights: *const f32,
    pub counts: *mut i32,
    pub tokens: i64,
    pub top: i64,
    pub hidden: i64,
    pub experts: i64,
    pub max_tokens: i64,
    pub ranks: i64,
    pub dtype: i64,
    pub error: [u8; 512],
}

/// The call structure the combine package's entry receives.
///
/// Mirrored by hand in the same header.
#[repr(C)]
pub struct CombineCall {
    pub transport: *mut c_void,
    pub partials: *const c_void,
    pub counts: *const i32,
    pub out: *mut c_void,
    pub tokens: i64,
    pub top: i64,
    pub hidden: i64,
    pub experts: i64,
    pub max_tokens: i64,
    pub ranks: i64,
    pub dtype: i64,
    pub error: [u8; 512],
}

/// The size and field offsets the dispatch header must mirror, in bytes.
pub const DISPATCH_CALL_LAYOUT: CallLayout = CallLayout {
    size: 608,
    error: 96,
};

/// The size and field offsets the combine header must mirror, in bytes.
pub const COMBINE_CALL_LAYOUT: CallLayout = CallLayout {
    size: 600,
    error: 88,
};

/// The layout of one exchange call structure, for its header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct CallLayout {
    pub size: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<DispatchCall>() == DISPATCH_CALL_LAYOUT.size,
    "the expert_exchange dispatch structure's size changed without its header"
);
const _: () = assert!(
    std::mem::size_of::<CombineCall>() == COMBINE_CALL_LAYOUT.size,
    "the expert_exchange combine structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(DispatchCall, error) == DISPATCH_CALL_LAYOUT.error,
    "the expert_exchange dispatch structure's `error` moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(CombineCall, error) == COMBINE_CALL_LAYOUT.error,
    "the expert_exchange combine structure's `error` moved without its header"
);

impl NativeCall for DispatchCall {
    fn message(&self) -> Option<&str> {
        message(&self.error)
    }
}

impl NativeCall for CombineCall {
    fn message(&self) -> Option<&str> {
        message(&self.error)
    }
}

fn message(error: &[u8; 512]) -> Option<&str> {
    let end = error.iter().position(|&byte| byte == 0)?;
    std::str::from_utf8(&error[..end]).ok()
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
        args.counts
            .require_shape(Self::NAME, Shape::new(&[ranks]))?;
        if args.counts.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 counts, found {}", args.counts.dtype),
            ));
        }
        if args.transport.handle().is_null() {
            return Err(Error::argument(
                Self::NAME,
                "the call passed no transport; the exchange runs over the group's workspace",
            ));
        }

        match descriptor.mode {
            Mode::Dispatch => {
                let tokens = u64::from(args.tokens);
                if tokens > max_tokens {
                    return Err(Error::argument(
                        Self::NAME,
                        format!("{tokens} rows exceed the transport's {max_tokens}"),
                    ));
                }
                args.state
                    .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
                args.state
                    .require_row(Self::NAME, descriptor.dtype, hidden)?;
                args.ids
                    .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
                args.weights
                    .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
                if args.ids.dtype != DType::I32 || args.weights.dtype != DType::F32 {
                    return Err(Error::argument(
                        Self::NAME,
                        "a dispatch sends i32 expert ids and FP32 route weights",
                    ));
                }

                let mut call = DispatchCall {
                    transport: args.transport.handle(),
                    states: address(args.state.ptr) as usize as *const c_void,
                    ids: address(args.ids.ptr) as usize as *const i32,
                    weights: address(args.weights.ptr) as usize as *const f32,
                    counts: address(args.counts.ptr) as usize as *mut i32,
                    tokens: to_i64(Self::NAME, tokens)?,
                    top: to_i64(Self::NAME, top)?,
                    hidden: to_i64(Self::NAME, hidden)?,
                    experts: to_i64(Self::NAME, experts)?,
                    max_tokens: to_i64(Self::NAME, max_tokens)?,
                    ranks: to_i64(Self::NAME, ranks)?,
                    dtype: dtype_code(descriptor.dtype),
                    error: [0; 512],
                };
                // SAFETY: `DispatchCall` is the call structure this package's entry declares, and
                // its layout is pinned by the assertions above and by the test below.
                unsafe { kernel.native(&mut call, stream) }
            }
            Mode::Combine => {
                let tokens = u64::from(args.tokens);
                args.state
                    .require_shape(Self::NAME, Shape::new(&[max_tokens, hidden]))?;
                args.state
                    .require_row(Self::NAME, descriptor.dtype, hidden)?;
                args.out
                    .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
                args.out.require_row(Self::NAME, descriptor.dtype, hidden)?;

                let mut call = CombineCall {
                    transport: args.transport.handle(),
                    partials: address(args.state.ptr) as usize as *const c_void,
                    counts: address(args.counts.ptr) as usize as *const i32,
                    out: address(args.out.ptr) as usize as *mut c_void,
                    tokens: to_i64(Self::NAME, tokens)?,
                    top: to_i64(Self::NAME, top)?,
                    hidden: to_i64(Self::NAME, hidden)?,
                    experts: to_i64(Self::NAME, experts)?,
                    max_tokens: to_i64(Self::NAME, max_tokens)?,
                    ranks: to_i64(Self::NAME, ranks)?,
                    dtype: dtype_code(descriptor.dtype),
                    error: [0; 512],
                };
                // SAFETY: `CombineCall` is the call structure this package's entry declares, and its
                // layout is pinned by the assertions above and by the test below.
                unsafe { kernel.native(&mut call, stream) }
            }
        }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
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
            Mode::Combine => {
                Tolerance::dot(u64::from(descriptor.ranks), DType::F32, descriptor.dtype)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_layouts_are_the_ones_the_header_declares() {
        let dispatch = DISPATCH_CALL_LAYOUT;
        assert_eq!(std::mem::size_of::<DispatchCall>(), dispatch.size);
        assert_eq!(std::mem::offset_of!(DispatchCall, error), dispatch.error);
        let combine = COMBINE_CALL_LAYOUT;
        assert_eq!(std::mem::size_of::<CombineCall>(), combine.size);
        assert_eq!(std::mem::offset_of!(CombineCall, error), combine.error);
        assert_eq!(std::mem::offset_of!(DispatchCall, transport), 0);
        assert_eq!(std::mem::offset_of!(CombineCall, transport), 0);
    }

    #[test]
    fn dispatch_is_exact_and_combine_carries_the_rank_sum() {
        let dispatch = Descriptor {
            mode: Mode::Dispatch,
            hidden: 2048,
            top: 8,
            experts: 128,
            max_tokens: 8192,
            ranks: 4,
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
            top: 8,
            experts: 128,
            max_tokens: 8192,
            ranks: 4,
            dtype: DType::BF16,
        };
        assert_eq!(
            combine.to_string(),
            "mode_combine_hidden_2048_top_8_experts_128_max_tokens_8192_ranks_4_dtype_bf16"
        );
        let wider = Descriptor {
            ranks: 8,
            ..combine
        };
        assert_ne!(wider.to_string(), combine.to_string());
    }
}
