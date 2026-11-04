//! Paged attention over a prefix cache.
//!
//! One operator for every attention a model computes over paged keys and values — a prefill of
//! packed query rows and a decode of one query per row — with the phase in the descriptor, because
//! the two phases select different kernels, grids and cubins inside the package
//! (specs/worker.md 5.1, 5.2, 9).
//!
//! ```text
//! query      [tokens, q_heads, head_dim]      one dtype, row-major
//! key        [pages, page_size, kv_heads, head_dim]   pages of one cache
//! value      [pages, page_size, kv_heads, head_dim]
//! page_table [batch, max_pages] i32           each row's pages, from column 0
//! seq_lens   [batch] i32                      keys visible to the row's last query
//! cum_seq_lens_q [batch + 1] i32              prefill only: query offsets of the packed rows
//! out        [tokens, q_heads, head_dim]
//! ```
//!
//! The cache is the layout `kv_write` produces (see [`crate::kv`]): `[pages, page_size, kv_heads,
//! head_dim]`, read here through strides, so a plan that stores keys and values in one allocation at
//! two offsets, or interleaves pages, needs no copy. Keys and values may address the same pages as
//! one view of twice the extent.
//!
//! # The mathematics
//!
//! For query row `t` with head `h`, with `s_j = scale · q_t,h · k_{j,h}` over the visible keys `j`:
//!
//! ```text
//! p_j   = exp(s_j − max_j s_j)
//! out_t,h = Σ_j p_j v_{j,h} / Σ_j p_j
//! ```
//!
//! accumulated in FP32 and rounded once to the output dtype. The visibility is the causal mask (a
//! key `j` is visible to a query at position `i` when `j ≤ i`) plus, when a history window is
//! launched, `j ≥ i − window_left`, and the dense mask when the descriptor declares it.
//!
//! # The bound between two implementations
//!
//! The output is a convex combination of the values it reads, so `Σ_j |p_j v_j| = S · V` with `S` the
//! softmax normalizer and `V = max_j |v_j|`, and the standard bound (Higham, *Accuracy and Stability
//! of Numerical Algorithms*, 2nd ed., §3.1) gives, for FP32 accumulation with unit round-off `u` and
//! `n` visible keys,
//!
//! ```text
//! |out − out_exact| ≤ (γₙ + u) V + u_out |out|,   γₙ = n·u / (1 − n·u)
//! ```
//!
//! (the `u` term is the division, and the softmax's own `expf` error is a few more `u`, far below
//! `γₙ` and `u_out`). Two implementations therefore differ by at most
//! `2(γₙ + u) V + 2 u_out |out|`. The `V` term does not scale with the result — a row whose output
//! cancels can be smaller than `V` — so it enters the tolerance as an absolute term over the element
//! envelope [`VALUE_ENVELOPE`] documented below, and the result-scaled part as `rtol`. The reduction
//! length `n` is the descriptor's [`Descriptor::max_kv_len`], the largest key count a module serves,
//! which bounds every launch: a longer reduction would only widen the bound.
//!
//! # The package
//!
//! The program is a native library: trtllm-gen's host code selects and loads its cubins, builds its
//! TMA descriptors, carves its workspace and computes its own grid and cluster (specs/worker.md 9,
//! `uniserve/backends/attention/trtllm.py:79-91, 163-205`), so [`Op::launch`] computes no grid and
//! hands the library [`Call`]. FlashAttention-4's CuTe DSL package implements the same operation and
//! the same call structure, and is chosen by `attention_backend` / `decode_attention_backend` /
//! `prefill_attention_backend`.
//!
//! The workspace is the program's `workspace_size` argument (512 MiB for trtllm-gen), a transient
//! region the executor allocates from its scratch pool and zeroes before its first use: trtllm-gen
//! keeps its multi-CTA reduction counters there across launches.

pub mod dense;
pub mod sparse;

pub use dense::{DenseArgs, DenseAttention, DenseCall, DenseDescriptor};
pub use sparse::{SparseArgs, SparseAttention, SparseCall, SparseDescriptor};

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::{self, Stream};
use crate::tensor::{DType, DevicePtr, Shape, TensorView};
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The largest value magnitude the attention bound's absolute term is written for.
///
/// Attention reads the value projection of a normalized residual stream: in the models served here
/// (BF16 hidden states after an RMSNorm whose weight is near one) an element of `v` stays below 8 in
/// magnitude. The bound's `V` term is a declaration about that data — it cannot be derived from the
/// descriptor — and rows that exceed the envelope are outside what a data-free tolerance can cover.
pub const VALUE_ENVELOPE: f32 = 8.0;

/// Which half of a batch's attention work a module serves.
///
/// Prefill packs several query rows per sequence; decode has one query per row. The two select
/// different kernels (a context kernel against a generation kernel) and different grids, so a
/// module is compiled for one of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    /// Packed query rows with `cum_seq_lens_q` offsets.
    Prefill,
    /// One query row per sequence, aligned to the end of its keys.
    Decode,
}

impl fmt::Display for Phase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        })
    }
}

/// The mask a module is compiled for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mask {
    /// A query at position `i` sees keys `0 ..= i` (and, with a window, `i − window_left ..= i`).
    Causal,
    /// Every query sees every visible key; a history window is not expressible with it.
    Dense,
}

impl fmt::Display for Mask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Causal => "causal",
            Self::Dense => "dense",
        })
    }
}

/// One compilation of `paged_attention`.
///
/// Everything here is a value a program bakes into its module: the element types, the head and page
/// geometry and the mask variant. What a launch may change without a new module — the batch, the
/// lengths, the scale, the window, the strides — travels in [`Args`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Which phase this module serves.
    pub phase: Phase,
    /// Element type of the queries.
    pub query: DType,
    /// Element type of the cached keys and values.
    pub key_value: DType,
    /// Element type of the output.
    pub output: DType,
    /// Elements per query and key head.
    pub head_dim: u32,
    /// Tokens per cache page.
    pub page_size: u32,
    /// Query heads this rank holds.
    pub q_heads: u32,
    /// Key/value heads this rank holds.
    pub kv_heads: u32,
    /// The mask variant.
    pub mask: Mask,
    /// Whether a history window may be launched.
    pub windowed: bool,
    /// The longest key reduction this module serves; the reduction length of the tolerance bound.
    pub max_kv_len: u32,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("phase", self.phase)
            .field("dtype", self.query)
            .field("kv", self.key_value)
            .field("out", self.output)
            .field("head_dim", self.head_dim)
            .field("page", self.page_size)
            .field("q_heads", self.q_heads)
            .field("kv_heads", self.kv_heads)
            .field("mask", self.mask)
            .field("window", self.windowed)
            .field("len", self.max_kv_len)
            .finish(formatter)
    }
}

/// The arguments of one attention call.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[tokens, q_heads, head_dim]` queries, row-major on the last axis.
    pub query: TensorView,
    /// `[pages, page_size, kv_heads, head_dim]` cached keys.
    pub key: TensorView,
    /// `[pages, page_size, kv_heads, head_dim]` cached values.
    pub value: TensorView,
    /// `[pages, 1, 1, 1]` f32: each page's key scale, required for a quantized `kv` and absent
    /// otherwise. The page holds `key / scale`, and the kernel multiplies it back in FP32.
    pub key_scale: Option<TensorView>,
    /// `[pages, 1, 1, 1]` f32: each page's value scale, as `key_scale`.
    pub value_scale: Option<TensorView>,
    /// `[batch, max_pages]` i32: each row's pages.
    pub page_table: TensorView,
    /// `[batch]` i32: keys visible to the row's last query.
    pub seq_lens: TensorView,
    /// `[batch + 1]` i32: packed query offsets; empty at decode.
    pub cum_seq_lens_q: TensorView,
    /// `[tokens, q_heads, head_dim]` output.
    pub out: TensorView,
    /// `[bytes]` u8 transient workspace of the program, zeroed before its first use.
    pub workspace: TensorView,
    /// Softmax scale applied to `q · k` before the exponential.
    pub scale: f32,
    /// Inclusive history window in keys; `-1` for none.
    pub window_left: i32,
    /// Whether a prefill applies the causal mask; decode is causal by construction.
    pub causal: bool,
}

/// The call structure a package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/paged_attention/<backend>/include/uniserve_paged_attention.h`;
/// the layout is pinned by [`PAGED_ATTENTION_CALL_LAYOUT`], the assertions below and the test beside
/// this module. A package caches what it derives from these fields per (device, function) — the SM
/// count, the loaded cubins, the runner — so nothing here is a device property.
#[repr(C)]
pub struct Call {
    /// `[tokens, q_heads, head_dim]`.
    pub out: *mut c_void,
    /// `[tokens, q_heads, head_dim]`.
    pub query: *const c_void,
    /// `[pages, page_size, kv_heads, head_dim]`.
    pub key: *const c_void,
    /// `[pages, page_size, kv_heads, head_dim]`.
    pub value: *const c_void,
    /// `[pages, 1, 1, 1]` f32 key scales; null for an unquantized cache.
    pub key_scale: *const c_void,
    /// `[pages, 1, 1, 1]` f32 value scales; null for an unquantized cache.
    pub value_scale: *const c_void,
    /// The program's transient workspace.
    pub workspace: *mut c_void,
    /// `[batch, max_pages]` i32.
    pub page_table: *const i32,
    /// `[batch]` i32.
    pub seq_lens: *const i32,
    /// `[batch + 1]` i32; null at decode.
    pub cum_seq_lens_q: *const i32,
    /// Rows of the page table.
    pub batch: i64,
    /// Query rows.
    pub tokens: i64,
    /// Longest row's query count in this call; 1 at decode.
    pub max_q_len: i64,
    /// Page table width times page size: the longest possible key count.
    pub max_kv_len: i64,
    /// Pages the pool holds.
    pub num_pages: i64,
    /// Tokens per page.
    pub page_size: i64,
    /// Query heads.
    pub q_heads: i64,
    /// Key/value heads.
    pub kv_heads: i64,
    /// Elements per head.
    pub head_dim: i64,
    /// Elements between consecutive query rows.
    pub q_stride_tokens: i64,
    /// Elements between consecutive query heads.
    pub q_stride_heads: i64,
    /// Elements between consecutive pages.
    pub kv_stride_page: i64,
    /// Elements between consecutive tokens inside a page.
    pub kv_stride_tokens: i64,
    /// Elements between consecutive key/value heads.
    pub kv_stride_heads: i64,
    /// Elements between consecutive output rows.
    pub o_stride_tokens: i64,
    /// Elements between consecutive output heads.
    pub o_stride_heads: i64,
    /// The softmax scale.
    pub scale: f64,
    /// Inclusive history window; -1 for none.
    pub window_left: i64,
    /// 1 when the causal mask applies.
    pub causal: i64,
    /// Bytes of the workspace.
    pub workspace_bytes: i64,
    /// `DType`'s ABI number of the queries.
    pub dtype_query: i64,
    /// `DType`'s ABI number of the cached keys and values.
    pub dtype_key_value: i64,
    /// `DType`'s ABI number of the output.
    pub dtype_output: i64,
    /// The phase: 0 prefill, 1 decode.
    pub phase: i64,
    /// The entry's message on a nonzero return.
    pub error: [u8; 512],
}

/// The size and field offsets the package's header must mirror, in bytes.
pub const PAGED_ATTENTION_CALL_LAYOUT: CallLayout = CallLayout {
    size: 784,
    out: 0,
    query: 8,
    key: 16,
    value: 24,
    key_scale: 32,
    value_scale: 40,
    workspace: 48,
    page_table: 56,
    seq_lens: 64,
    cum_seq_lens_q: 72,
    batch: 80,
    tokens: 88,
    max_q_len: 96,
    max_kv_len: 104,
    num_pages: 112,
    page_size: 120,
    q_heads: 128,
    kv_heads: 136,
    head_dim: 144,
    q_stride_tokens: 152,
    q_stride_heads: 160,
    kv_stride_page: 168,
    kv_stride_tokens: 176,
    kv_stride_heads: 184,
    o_stride_tokens: 192,
    o_stride_heads: 200,
    scale: 208,
    window_left: 216,
    causal: 224,
    workspace_bytes: 232,
    dtype_query: 240,
    dtype_key_value: 248,
    dtype_output: 256,
    phase: 264,
    error: 272,
};

/// The layout of [`Call`], for the header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct CallLayout {
    pub size: usize,
    pub out: usize,
    pub query: usize,
    pub key: usize,
    pub value: usize,
    pub key_scale: usize,
    pub value_scale: usize,
    pub workspace: usize,
    pub page_table: usize,
    pub seq_lens: usize,
    pub cum_seq_lens_q: usize,
    pub batch: usize,
    pub tokens: usize,
    pub max_q_len: usize,
    pub max_kv_len: usize,
    pub num_pages: usize,
    pub page_size: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_stride_tokens: usize,
    pub q_stride_heads: usize,
    pub kv_stride_page: usize,
    pub kv_stride_tokens: usize,
    pub kv_stride_heads: usize,
    pub o_stride_tokens: usize,
    pub o_stride_heads: usize,
    pub scale: usize,
    pub window_left: usize,
    pub causal: usize,
    pub workspace_bytes: usize,
    pub dtype_query: usize,
    pub dtype_key_value: usize,
    pub dtype_output: usize,
    pub phase: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<Call>() == PAGED_ATTENTION_CALL_LAYOUT.size,
    "the paged_attention call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, page_table) == PAGED_ATTENTION_CALL_LAYOUT.page_table,
    "the paged_attention call structure's device pointers moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, batch) == PAGED_ATTENTION_CALL_LAYOUT.batch,
    "the paged_attention call structure's geometry moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, q_stride_tokens) == PAGED_ATTENTION_CALL_LAYOUT.q_stride_tokens,
    "the paged_attention call structure's query strides moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, scale) == PAGED_ATTENTION_CALL_LAYOUT.scale,
    "the paged_attention call structure's scalars moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, dtype_query) == PAGED_ATTENTION_CALL_LAYOUT.dtype_query,
    "the paged_attention call structure's dtypes moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, error) == PAGED_ATTENTION_CALL_LAYOUT.error,
    "the paged_attention call structure's `error` moved without its header"
);

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// Paged attention.
pub struct PagedAttention;

impl Op for PagedAttention {
    const NAME: &'static str = "paged_attention";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let head_dim = u64::from(descriptor.head_dim);
        let q_heads = u64::from(descriptor.q_heads);
        let kv_heads = u64::from(descriptor.kv_heads);
        let page_size = u64::from(descriptor.page_size);

        let tokens = args.query.rows();
        args.query
            .require_shape(Self::NAME, Shape::new(&[tokens, q_heads, head_dim]))?;
        args.query
            .require_row(Self::NAME, descriptor.query, head_dim)?;
        if args.query.strides.rank() >= 2 && args.query.strides.get(1) != head_dim {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {head_dim} elements between query heads, found {}",
                    args.query.strides.get(1)
                ),
            ));
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[tokens, q_heads, head_dim]))?;
        args.out
            .require_row(Self::NAME, descriptor.output, head_dim)?;

        for cache in [&args.key, &args.value] {
            cache.require_rank(Self::NAME, 4)?;
            if cache.dtype != descriptor.key_value {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected {} cached keys and values, found {}",
                        descriptor.key_value, cache.dtype
                    ),
                ));
            }
            if cache.shape.dim(1) != page_size
                || cache.shape.dim(2) != kv_heads
                || cache.shape.dim(3) != head_dim
            {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected pages of [page_size, kv_heads, head_dim] = {page_size}x{kv_heads}x{head_dim}, found {:?}",
                        cache.shape.dims()
                    ),
                ));
            }
            if cache.strides.last() != 1 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected a unit stride on the cache's last axis, found {}",
                        cache.strides.last()
                    ),
                ));
            }
        }

        args.page_table.require_rank(Self::NAME, 2)?;
        if args.page_table.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 pages, found {}", args.page_table.dtype),
            ));
        }
        let batch = args.page_table.shape.dim(0);
        let max_pages = args.page_table.shape.dim(1);
        if args.page_table.strides.last() != 1 {
            return Err(Error::argument(
                Self::NAME,
                "expected a unit stride on the page table's last axis",
            ));
        }

        args.seq_lens
            .require_shape(Self::NAME, Shape::new(&[batch]))?;
        if args.seq_lens.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 lengths, found {}", args.seq_lens.dtype),
            ));
        }

        let max_kv_len = max_pages * page_size;
        if max_kv_len > u64::from(descriptor.max_kv_len) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the module serves reductions of at most {}, the page table can address {max_kv_len} keys",
                    descriptor.max_kv_len
                ),
            ));
        }

        let cum_seq_lens_q = match descriptor.phase {
            Phase::Prefill => {
                if args.cum_seq_lens_q.shape.rank() != 1 || args.cum_seq_lens_q.dtype != DType::I32
                {
                    return Err(Error::argument(
                        Self::NAME,
                        "a prefill passes its packed query offsets as an i32 [batch + 1] view",
                    ));
                }
                if batch > 0 && args.cum_seq_lens_q.numel() != batch + 1 {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "expected {} query offsets, found {}",
                            batch + 1,
                            args.cum_seq_lens_q.numel()
                        ),
                    ));
                }
                address(args.cum_seq_lens_q.ptr) as usize as *const i32
            }
            Phase::Decode => {
                // A decode has one query per row, so the offsets are the row ends and the package
                // derives nothing from them; an empty view is passed as null.
                if args.cum_seq_lens_q.numel() == 0 {
                    std::ptr::null()
                } else {
                    address(args.cum_seq_lens_q.ptr) as usize as *const i32
                }
            }
        };

        // A quantized cache holds each page divided by its own scale; the option is the cache's, so
        // the descriptor's dtype decides whether the call carries them.
        let (key_scale, value_scale) = match (
            quantized(descriptor.key_value),
            args.key_scale,
            args.value_scale,
        ) {
            (false, None, None) => (std::ptr::null(), std::ptr::null()),
            (true, Some(key), Some(value)) => {
                for (name, view) in [("key_scale", key), ("value_scale", value)] {
                    view.require_shape(Self::NAME, Shape::new(&[args.key.shape.dim(0), 1, 1, 1]))?;
                    if view.dtype != DType::F32 {
                        return Err(Error::argument(
                            Self::NAME,
                            format!("expected an f32 {name}, found {}", view.dtype),
                        ));
                    }
                }
                (
                    address(key.ptr) as usize as *const c_void,
                    address(value.ptr) as usize as *const c_void,
                )
            }
            (true, _, _) => {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "a {} cache is read through one scale per page, and the call passes fewer",
                        descriptor.key_value
                    ),
                ));
            }
            (false, _, _) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the call passes page scales for an unquantized cache",
                ));
            }
        };

        let causal = match (descriptor.mask, args.causal) {
            (Mask::Causal, true) => 1_i64,
            (Mask::Dense, false) => 0_i64,
            (Mask::Causal, false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the module was compiled for the causal mask and the call is dense",
                ));
            }
            (Mask::Dense, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the module was compiled for the dense mask and the call is causal",
                ));
            }
        };

        if args.window_left >= 0 && !descriptor.windowed {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the call asks for a history window of {} and the module has none",
                    args.window_left
                ),
            ));
        }
        if descriptor.mask == Mask::Dense && args.window_left >= 0 {
            return Err(Error::argument(
                Self::NAME,
                "a history window is expressible only with the causal mask",
            ));
        }

        if args.workspace.dtype != DType::U8 || args.workspace.numel() == 0 {
            return Err(Error::argument(
                Self::NAME,
                "the call passes its program's workspace as a non-empty u8 view",
            ));
        }

        let max_q_len = match descriptor.phase {
            Phase::Prefill => {
                // The longest row is only known from the offsets on the device; the module pads to
                // the packed token count, which bounds every row.
                tokens
            }
            Phase::Decode => 1,
        };

        let mut call = Call {
            out: address(args.out.ptr) as usize as *mut c_void,
            query: address(args.query.ptr) as usize as *const c_void,
            key: address(args.key.ptr) as usize as *const c_void,
            value: address(args.value.ptr) as usize as *const c_void,
            key_scale,
            value_scale,
            workspace: address(args.workspace.ptr) as usize as *mut c_void,
            page_table: address(args.page_table.ptr) as usize as *const i32,
            seq_lens: address(args.seq_lens.ptr) as usize as *const i32,
            cum_seq_lens_q,
            batch: to_i64(Self::NAME, batch)?,
            tokens: to_i64(Self::NAME, tokens)?,
            max_q_len: to_i64(Self::NAME, max_q_len)?,
            max_kv_len: to_i64(Self::NAME, max_kv_len)?,
            num_pages: to_i64(Self::NAME, args.key.shape.dim(0))?,
            page_size: to_i64(Self::NAME, page_size)?,
            q_heads: to_i64(Self::NAME, q_heads)?,
            kv_heads: to_i64(Self::NAME, kv_heads)?,
            head_dim: to_i64(Self::NAME, head_dim)?,
            q_stride_tokens: to_i64(Self::NAME, row_stride(&args.query))?,
            q_stride_heads: to_i64(Self::NAME, head_stride(&args.query))?,
            kv_stride_page: to_i64(Self::NAME, args.key.strides.get(0))?,
            kv_stride_tokens: to_i64(Self::NAME, args.key.strides.get(1))?,
            kv_stride_heads: to_i64(Self::NAME, args.key.strides.get(2))?,
            o_stride_tokens: to_i64(Self::NAME, row_stride(&args.out))?,
            o_stride_heads: to_i64(Self::NAME, head_stride(&args.out))?,
            scale: f64::from(args.scale),
            window_left: i64::from(args.window_left),
            causal,
            workspace_bytes: to_i64(Self::NAME, args.workspace.numel())?,
            dtype_query: descriptor.query.as_i64(),
            dtype_key_value: descriptor.key_value.as_i64(),
            dtype_output: descriptor.output.as_i64(),
            phase: match descriptor.phase {
                Phase::Prefill => 0,
                Phase::Decode => 1,
            },
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.max_kv_len);
        let accumulate = DType::F32;
        let gamma = tolerance::gamma(reduction, accumulate);
        let value_term = 2.0 * gamma + accumulate.unit_roundoff();
        Tolerance::rounding(reduction, accumulate, descriptor.output)
            .with_atol((value_term * f64::from(VALUE_ENVELOPE)) as f32)
    }
}

/// Whether the cache's pages hold quantized elements that are read through a per-page scale.
fn quantized(dtype: DType) -> bool {
    matches!(dtype, DType::F8E4M3 | DType::F4E2M1)
}

/// The elements between consecutive rows of a view.
pub(super) fn row_stride(view: &TensorView) -> u64 {
    if view.strides.rank() >= 2 {
        view.strides.get(view.strides.rank() - 2)
    } else {
        0
    }
}

/// The elements between consecutive heads of a `[rows, heads, dim]` view.
pub(super) fn head_stride(view: &TensorView) -> u64 {
    if view.strides.rank() >= 2 {
        view.strides.get(view.strides.rank() - 2 + 1)
    } else {
        0
    }
}

pub(super) fn address(pointer: DevicePtr) -> u64 {
    platform::address(pointer)
}

/// An extent as the ABI's `i64`, or an argument error naming `operator`.
pub(super) fn to_i64(operator: &'static str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument(operator, "an extent above i64::MAX"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A BF16 causal decode module of Qwen3-30B-A3B at TP=8, page size 64.
    fn decode() -> Descriptor {
        Descriptor {
            phase: Phase::Decode,
            query: DType::BF16,
            key_value: DType::BF16,
            output: DType::BF16,
            head_dim: 128,
            page_size: 64,
            q_heads: 4,
            kv_heads: 1,
            mask: Mask::Causal,
            windowed: false,
            max_kv_len: 40960,
        }
    }

    #[test]
    fn key_names_every_compiled_value() {
        assert_eq!(
            decode().to_string(),
            "phase_decode_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_4_kv_heads_1_mask_causal_window_false_len_40960"
        );
    }

    /// A quantized cache is another module of the same operator, and the dtype is what tells a
    /// launch to expect the per-page scales.
    #[test]
    fn a_quantized_cache_is_another_module() {
        let mut fp8 = decode();
        fp8.key_value = DType::F8E4M3;
        assert_ne!(decode().to_string(), fp8.to_string());
        assert_eq!(
            fp8.to_string(),
            "phase_decode_dtype_bf16_kv_f8e4m3_out_bf16_head_dim_128_page_64_q_heads_4_kv_heads_1_mask_causal_window_false_len_40960"
        );
        assert!(quantized(fp8.key_value));
        assert!(!quantized(decode().key_value));
        assert!(!quantized(DType::F32));
    }

    #[test]
    fn key_separates_page_sizes_and_phases() {
        let mut prefill = decode();
        prefill.phase = Phase::Prefill;
        prefill.page_size = 16;
        assert_ne!(decode().to_string(), prefill.to_string());
        assert!(prefill.to_string().starts_with("phase_prefill_dtype_bf16"));
        assert!(prefill.to_string().contains("_page_16_"));
    }

    /// The header the packages mirror must carry these offsets; the constants beside the structure
    /// are checked against the compiler here.
    #[test]
    fn call_layout_is_the_one_the_header_declares() {
        let layout = PAGED_ATTENTION_CALL_LAYOUT;
        assert_eq!(std::mem::size_of::<Call>(), layout.size);
        assert_eq!(std::mem::align_of::<Call>(), 8);
        assert_eq!(std::mem::offset_of!(Call, out), layout.out);
        assert_eq!(std::mem::offset_of!(Call, query), layout.query);
        assert_eq!(std::mem::offset_of!(Call, key), layout.key);
        assert_eq!(std::mem::offset_of!(Call, value), layout.value);
        assert_eq!(std::mem::offset_of!(Call, key_scale), layout.key_scale);
        assert_eq!(std::mem::offset_of!(Call, value_scale), layout.value_scale);
        assert_eq!(std::mem::offset_of!(Call, workspace), layout.workspace);
        assert_eq!(std::mem::offset_of!(Call, page_table), layout.page_table);
        assert_eq!(std::mem::offset_of!(Call, seq_lens), layout.seq_lens);
        assert_eq!(
            std::mem::offset_of!(Call, cum_seq_lens_q),
            layout.cum_seq_lens_q
        );
        assert_eq!(std::mem::offset_of!(Call, batch), layout.batch);
        assert_eq!(std::mem::offset_of!(Call, tokens), layout.tokens);
        assert_eq!(std::mem::offset_of!(Call, max_q_len), layout.max_q_len);
        assert_eq!(std::mem::offset_of!(Call, max_kv_len), layout.max_kv_len);
        assert_eq!(std::mem::offset_of!(Call, num_pages), layout.num_pages);
        assert_eq!(std::mem::offset_of!(Call, page_size), layout.page_size);
        assert_eq!(std::mem::offset_of!(Call, q_heads), layout.q_heads);
        assert_eq!(std::mem::offset_of!(Call, kv_heads), layout.kv_heads);
        assert_eq!(std::mem::offset_of!(Call, head_dim), layout.head_dim);
        assert_eq!(
            std::mem::offset_of!(Call, q_stride_tokens),
            layout.q_stride_tokens
        );
        assert_eq!(
            std::mem::offset_of!(Call, q_stride_heads),
            layout.q_stride_heads
        );
        assert_eq!(
            std::mem::offset_of!(Call, kv_stride_page),
            layout.kv_stride_page
        );
        assert_eq!(
            std::mem::offset_of!(Call, kv_stride_tokens),
            layout.kv_stride_tokens
        );
        assert_eq!(
            std::mem::offset_of!(Call, kv_stride_heads),
            layout.kv_stride_heads
        );
        assert_eq!(
            std::mem::offset_of!(Call, o_stride_tokens),
            layout.o_stride_tokens
        );
        assert_eq!(
            std::mem::offset_of!(Call, o_stride_heads),
            layout.o_stride_heads
        );
        assert_eq!(std::mem::offset_of!(Call, scale), layout.scale);
        assert_eq!(std::mem::offset_of!(Call, window_left), layout.window_left);
        assert_eq!(std::mem::offset_of!(Call, causal), layout.causal);
        assert_eq!(
            std::mem::offset_of!(Call, workspace_bytes),
            layout.workspace_bytes
        );
        assert_eq!(std::mem::offset_of!(Call, dtype_query), layout.dtype_query);
        assert_eq!(
            std::mem::offset_of!(Call, dtype_key_value),
            layout.dtype_key_value
        );
        assert_eq!(
            std::mem::offset_of!(Call, dtype_output),
            layout.dtype_output
        );
        assert_eq!(std::mem::offset_of!(Call, phase), layout.phase);
        assert_eq!(std::mem::offset_of!(Call, error), layout.error);
    }

    /// The bound grows with the reduction length and leaves room for a corrected implementation but
    /// not for a different mask: at 40960 keys it is under one percent plus the envelope.
    #[test]
    fn tolerance_follows_the_reduction_length() {
        let mut short = decode();
        short.max_kv_len = 1024;
        let short = PagedAttention::tolerance(&short);
        let long = PagedAttention::tolerance(&decode());
        assert!(short.rtol > 0.0 && short.rtol < long.rtol);
        assert!(long.rtol < 0.01, "rtol {} is above one percent", long.rtol);
        assert!(
            long.atol > 0.0 && long.atol <= VALUE_ENVELOPE,
            "atol {}",
            long.atol
        );
        // Two implementations of the same kernel agree exactly, so a zero difference is allowed.
        assert!(long.allows(1.0, 1.0));
        // A dropped history window changes the output by far more than the bound.
        assert!(!long.allows(0.0, 1.0));
    }
}
