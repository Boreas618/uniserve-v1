//! Paged attention over a prefix cache.
//!
//! One operator for every attention a model computes over paged keys and values — a prefill of
//! packed query rows and a decode of one query per row — with the phase in the descriptor, because
//! the two phases select different kernels, grids and cubins inside the package.
//!
//! ```text
//! query      [tokens, q_heads, head_dim]      one dtype, row-major
//! key        [pages, page_size, kv_heads, head_dim]   pages of one cache
//! value      [pages, page_size, kv_heads, head_dim]
//! page_table [batch, max_pages] u32           each row's pages, from column 0
//! seq_lens   [batch] u32                      keys visible to the row's last query
//! cum_seq_lens_q [batch + 1] u32              prefill only: query offsets of the packed rows
//! out        [tokens, q_heads, head_dim]
//! ```
//!
//! The cache is the layout `kv_write` produces (see [`crate::kv`]): `[pages, page_size, kv_heads,
//! head_dim]`, read here through strides, so a cache that stores keys and values in one allocation at
//! two offsets, or interleaves pages, needs no copy. Keys and values may address the same pages as
//! one view of twice the extent.
//!
//! `page_table`'s entries are indices below `pages`, and nothing here checks that: the kernel
//! dereferences every entry it reads into the cache with no bound, so the invariant belongs to the
//! table's producer — the allocator that attaches a page — and a check on this side would have to
//! read device memory to say anything. `max_pages` bounds the *columns* a row may use, not the
//! values in them.
//!
//! `seq_lens` is the second half of the same statement, and it is the one a length carries: a row's
//! `seq_lens` entry is at most `max_pages · page_size`, which is the descriptor's
//! [`Descriptor::max_kv_len`]. The program derives the pages it walks from that entry
//! (`ceil(len / page_size)`), so a longer length reads table columns past the row's width — another
//! row's entries, or past the table — and then dereferences whatever they hold as page ids. A caller
//! that states a longer length has also voided the tolerance this operator declares, because
//! [`Descriptor::max_kv_len`] is the reduction length that bound is derived from.
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
//! envelope [`tolerance::VALUE_ENVELOPE`], and the result-scaled part as `rtol`. The reduction
//! length `n` is the descriptor's [`Descriptor::max_kv_len`], the largest key count a module serves,
//! which bounds every launch: a longer reduction would only widen the bound.
//!
//! # The package
//!
//! The program is a native library: trtllm-gen's host code selects and loads its cubins, builds its
//! TMA descriptors, carves its workspace and computes its own grid and cluster, so [`Op::launch`]
//! computes no grid and hands the library [`Call`]. FlashAttention-4's CuTe DSL package implements
//! the same operation and the same call structure, and a deployment's `[kernels]` table pins
//! either for a descriptor key. This crate's Triton package serves the caches the vendor programs
//! do not read, an E4M3 page and a dense page of another type than the queries, with the
//! arguments its programs declare instead of [`Call`], and a grid this launch computes in the
//! query tiles the loaded module was compiled for (its `BLOCK_Q`); a key no pin names goes to the
//! package of highest priority that covers it (`cover` at the Build Server).
//!
//! The workspace is the program's `workspace_size` argument (512 MiB for trtllm-gen), a transient
//! region the worker allocates from its scratch pool and zeroes before its first use: trtllm-gen
//! keeps its multi-CTA reduction counters there across launches.

pub mod dense;
pub mod indexed;
pub mod sparse;

pub use dense::{DenseArgs, DenseAttention, DenseCall, DenseDescriptor};
pub use indexed::{
    IndexScores, IndexScoresArgs, IndexScoresDescriptor, IndexedAttention, IndexedAttentionArgs,
    IndexedAttentionDescriptor,
};
pub use sparse::{SparseArgs, SparseAttention, SparseCall, SparseDescriptor};

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::{DeviceId, DevicePtr, Stream};
use tensor::{DType, Device};

use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The compile-time definition of the Triton prefill program that states the query tokens one
/// program holds: `BLOCK_Q` consecutive tokens of one batch row, with the query heads that share
/// one key-value head. The package derives it from its tile and the head group; the launch
/// counts its grid in such tiles, read from the module it launches.
const PREFILL_QUERY_TILE: &str = "BLOCK_Q";

/// The bytes of one arrival counter of a decode's multi-block reduction.
const COUNTER_BYTES: u64 = 4;

/// The counters a decode's workspace holds come in whole groups of this many, as the kernel's
/// semaphores expect.
const COUNTER_GROUP: u64 = 8;

/// The bytes at the head of a paged attention's workspace that hold a decode's arrival counters
/// on `device`, for `rows` rows of `q_heads` query heads: one counter per (row, query head) pair
/// and at least one per multiprocessor of the device (`platform::DeviceId::multiprocessors`, the
/// count the expert exchange's descriptor states of a rank's device), in whole groups of
/// [`COUNTER_GROUP`], as the package's entry carves its workspace. The meta device launches no
/// decode, so its counters are the (row, head) pairs alone.
///
/// A prefill's split-key scratch starts at the head of the same workspace, so a forward clears
/// these bytes once before its first decode attention.
///
/// # Errors
///
/// [`Error::ForeignTensor`] for a device of another backend, and the driver's failure to state
/// the multiprocessor count.
pub fn arrival_counter_bytes(device: &Device, rows: u64, q_heads: u64) -> Result<u64> {
    let multiprocessors = match crate::device::gpu(device, "an attention workspace")? {
        Some(ordinal) => {
            let id = DeviceId(i32::try_from(ordinal).map_err(|_| {
                Error::Shape(format!("GPU {ordinal} is outside the driver's ordinals"))
            })?);
            u64::from(id.multiprocessors()?)
        }
        None => 0,
    };

    Ok(counter_bytes(
        rows.saturating_mul(q_heads).max(multiprocessors),
    ))
}

/// The bytes `slots` arrival counters take at the head of a decode's workspace: whole groups of
/// [`COUNTER_GROUP`] counters of [`COUNTER_BYTES`] each.
pub const fn counter_bytes(slots: u64) -> u64 {
    slots.next_multiple_of(COUNTER_GROUP) * COUNTER_BYTES
}

/// Which half of a batch's attention work a module serves.
///
/// Prefill packs several query rows per sequence; decode has one query per row. The two select
/// different kernels (a context kernel against a generation kernel) and different grids, so a
/// module is compiled for one of them.
///
/// The call carries its number (`uniserve_phase` in `uniserve_paged_attention.h`).
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    /// Packed query rows with `cum_seq_lens_q` offsets.
    Prefill = 0,
    /// One query row per sequence, aligned to the end of its keys.
    Decode = 1,
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
///
/// A `dense_attention` module that compiles its mask in names it by its enumerator in
/// `uniserve_dense_attention.h` (`uniserve_mask`).
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mask {
    /// A query at position `i` sees keys `0 ..= i` (and, with a window, `i − window_left ..= i`).
    Causal = 1,
    /// Every query sees every visible key; a history window is not expressible with it.
    Dense = 0,
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
            .field("dtype", abi::name(self.query))
            .field("kv", abi::name(self.key_value))
            .field("out", abi::name(self.output))
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
#[derive(Clone)]
pub struct Args {
    /// `[tokens, q_heads, head_dim]` queries, row-major on the last axis.
    pub query: Operand,
    /// `[pages, page_size, kv_heads, head_dim]` cached keys.
    pub key: Operand,
    /// `[pages, page_size, kv_heads, head_dim]` cached values.
    pub value: Operand,
    /// `[pages, 1, 1, 1]` f32: each page's key scale, required for a quantized `kv` and absent
    /// otherwise. The page holds `key / scale`, and the kernel multiplies it back in FP32.
    pub key_scale: Option<Operand>,
    /// `[pages, 1, 1, 1]` f32: each page's value scale, as `key_scale`.
    pub value_scale: Option<Operand>,
    /// `[batch, max_pages]` u32: each row's pages.
    pub page_table: Operand,
    /// `[batch]` u32: keys visible to the row's last query.
    pub seq_lens: Operand,
    /// `[batch + 1]` u32: packed query offsets; empty at decode.
    pub cum_seq_lens_q: Operand,
    /// `[tokens, q_heads, head_dim]` output.
    pub out: Operand,
    /// `[bytes]` u8 transient workspace of the program, zeroed before its first use.
    pub workspace: Operand,
    /// Softmax scale applied to `q · k` before the exponential.
    pub scale: f32,
    /// Inclusive history window in keys; `-1` for none.
    pub window_left: i32,
    /// Whether a prefill applies the causal mask; decode is causal by construction.
    pub causal: bool,
}

/// The call structure a package's entry receives, declared to the packages as
/// `uniserve_paged_attention_call` (`uniserve_paged_attention.h`, generated with its layout
/// asserted: [`abi`]).
///
/// A package caches what it derives from these fields per (device, function) — the SM count, the
/// loaded cubins, the runner — so nothing here is a device property.
#[repr(C)]
#[derive(abi::Layout)]
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
    /// The [`Phase`], by its number.
    pub phase: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// Paged attention.
pub struct PagedAttention;

impl Op for PagedAttention {
    const NAME: &'static str = proto::module::operator::PAGED_ATTENTION;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let head_dim = u64::from(descriptor.head_dim);
        let q_heads = u64::from(descriptor.q_heads);
        let kv_heads = u64::from(descriptor.kv_heads);
        let page_size = u64::from(descriptor.page_size);

        // The token count is the first extent of a `[tokens, heads, head_dim]` query; `rows()` is
        // the product of every extent but the last and would count tokens times heads, refusing a
        // launch whose own views agree (the defect `qk_norm_rope` had).
        let tokens = args.query.dim(0);
        args.query
            .require_shape(Self::NAME, &[tokens, q_heads, head_dim])?;
        args.query
            .require_row(Self::NAME, descriptor.query, head_dim)?;
        if args.query.rank() >= 2 && args.query.stride(1) != head_dim {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {head_dim} elements between query heads, found {}",
                    args.query.stride(1)
                ),
            ));
        }
        args.out
            .require_shape(Self::NAME, &[tokens, q_heads, head_dim])?;
        args.out
            .require_row(Self::NAME, descriptor.output, head_dim)?;

        for cache in [&args.key, &args.value] {
            cache.require_rank(Self::NAME, 4)?;
            if cache.dtype != descriptor.key_value {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected {} cached keys and values, found {}",
                        abi::name(descriptor.key_value),
                        abi::name(cache.dtype)
                    ),
                ));
            }
            if cache.dim(1) != page_size || cache.dim(2) != kv_heads || cache.dim(3) != head_dim {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected pages of [page_size, kv_heads, head_dim] = {page_size}x{kv_heads}x{head_dim}, found {:?}",
                        cache.dims()
                    ),
                ));
            }
            if cache.last_stride() != 1 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected a unit stride on the cache's last axis, found {}",
                        cache.last_stride()
                    ),
                ));
            }
        }

        for (name, view) in [("query", &args.query), ("out", &args.out)] {
            check_vector_alignment(Self::NAME, name, view)?;
        }
        check_vector_alignment(Self::NAME, "key", &args.key)?;
        check_vector_alignment(Self::NAME, "value", &args.value)?;

        args.page_table.require_rank(Self::NAME, 2)?;
        if args.page_table.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u32 pages, found {}",
                    abi::name(args.page_table.dtype)
                ),
            ));
        }
        let batch = args.page_table.dim(0);
        let max_pages = args.page_table.dim(1);
        if args.page_table.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                "expected a unit stride on the page table's last axis",
            ));
        }

        args.seq_lens.require_shape(Self::NAME, &[batch])?;
        if args.seq_lens.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u32 lengths, found {}",
                    abi::name(args.seq_lens.dtype)
                ),
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

        match descriptor.phase {
            Phase::Prefill => {
                if args.cum_seq_lens_q.rank() != 1 || args.cum_seq_lens_q.dtype != DType::U32 {
                    return Err(Error::argument(
                        Self::NAME,
                        "a prefill passes its packed query offsets as a u32 [batch + 1] view",
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
            }
            Phase::Decode => {
                // A decode has one query per row, and the kernel indexes the page table and the
                // sequence lengths by that row: a call whose query or output is shorter than the
                // page table's rows is a row the kernel reads and writes past, which is an illegal
                // address rather than a wrong number. The invariant was stated and not enforced.
                for (name, view) in [("query", &args.query), ("out", &args.out)] {
                    if view.dim(0) != batch {
                        return Err(Error::argument(
                            Self::NAME,
                            format!(
                                "a decode has one {name} row per page-table row: {batch} rows of \
                                 pages, {} rows of {name}",
                                view.dim(0)
                            ),
                        ));
                    }
                }
            }
        }

        // A quantized cache holds each page divided by its own scale; the option is the cache's, so
        // the descriptor's dtype decides whether the call carries them.
        match (
            quantized(descriptor.key_value),
            &args.key_scale,
            &args.value_scale,
        ) {
            (false, None, None) => {}
            (true, Some(key), Some(value)) => {
                for (name, view) in [("key_scale", key), ("value_scale", value)] {
                    view.require_shape(Self::NAME, &[args.key.dim(0), 1, 1, 1])?;
                    if view.dtype != DType::F32 {
                        return Err(Error::argument(
                            Self::NAME,
                            format!("expected an f32 {name}, found {}", abi::name(view.dtype)),
                        ));
                    }
                }
            }
            (true, _, _) => {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "a {} cache is read through one scale per page, and the call passes fewer",
                        abi::name(descriptor.key_value)
                    ),
                ));
            }
            (false, _, _) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the call passes page scales for an unquantized cache",
                ));
            }
        }

        match (descriptor.mask, args.causal) {
            (Mask::Causal, true) | (Mask::Dense, false) => {}
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
        }

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

        // The compiled programs load their operands in 16-byte units and assert it: the cute
        // kernel's `tensorptr<bf16, gmem, align16>` and trtllm-gen's vectorized loads both refuse a
        // pointer that is not 16-byte aligned or a step that is not a multiple of 8 elements of a
        // 2-byte type. A host-side layout defect that breaks either surfaces here, by name,
        // rather than as a warp fault inside a kernel with no operand named in it.
        let element_bytes = abi::bytes(descriptor.query).unwrap_or(2).max(2);
        let unit = 16 / element_bytes.max(1);
        for (name, view) in [
            ("query", &args.query),
            ("key", &args.key),
            ("value", &args.value),
            ("out", &args.out),
        ] {
            if !address(view.ptr).is_multiple_of(16) {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the compiled program loads {name} in 16-byte units; its first address \
                         {:#x} is not a multiple of 16",
                        address(view.ptr)
                    ),
                ));
            }
            for axis in 0..view.rank() {
                let step = view.stride(axis);
                if axis + 1 != view.rank() && !step.is_multiple_of(unit) {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "the compiled program steps {name} axis {axis} in 16-byte units; its \
                             stride {step} is not a multiple of {unit} elements"
                        ),
                    ));
                }
            }
        }
        for (name, view) in [
            ("page_table", &args.page_table),
            ("seq_lens", &args.seq_lens),
        ] {
            if !address(view.ptr).is_multiple_of(16) {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the compiled program reads {name} in 16-byte units; its first address \
                         {:#x} is not a multiple of 16",
                        address(view.ptr)
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
        let head_dim = u64::from(descriptor.head_dim);
        let q_heads = u64::from(descriptor.q_heads);
        let kv_heads = u64::from(descriptor.kv_heads);
        let page_size = u64::from(descriptor.page_size);
        // The first extent of the `[tokens, heads, head_dim]` query.
        let tokens = args.query.dim(0);
        let batch = args.page_table.dim(0);
        let max_pages = args.page_table.dim(1);
        let max_kv_len = max_pages * page_size;

        // `check` holds the offsets to the phase. A decode's are the row ends and the package
        // derives nothing from them; an empty view is passed as null.
        let cum_seq_lens_q = match descriptor.phase {
            Phase::Prefill => address(args.cum_seq_lens_q.ptr) as usize as *const i32,
            Phase::Decode => {
                if args.cum_seq_lens_q.numel() == 0 {
                    std::ptr::null()
                } else {
                    address(args.cum_seq_lens_q.ptr) as usize as *const i32
                }
            }
        };

        // `check` holds the page scales to the cache: a quantized cache passes both and any other
        // passes none.
        let (key_scale, value_scale) = match (&args.key_scale, &args.value_scale) {
            (Some(key), Some(value)) => (
                address(key.ptr) as usize as *const c_void,
                address(value.ptr) as usize as *const c_void,
            ),
            _ => (std::ptr::null(), std::ptr::null()),
        };

        // `check` holds the call's mask to the module's.
        let causal = i64::from(args.causal);

        let max_q_len = match descriptor.phase {
            Phase::Prefill => {
                // The longest row is only known from the offsets on the device; the module pads to
                // the packed token count, which bounds every row.
                tokens
            }
            Phase::Decode => 1,
        };

        // A Triton program is this crate's own kernel and takes the arguments its package declares;
        // a native module keeps the `Call` structure above. It is one operator served by two kinds
        // of program, which is why the descriptor names no backend and the store decides.
        if kernel.is_cubin() {
            return Self::launch_program(
                kernel,
                descriptor,
                &args,
                PagedLaunch {
                    batch,
                    q_heads,
                    kv_heads,
                    page_size,
                    tokens,
                    max_q_len,
                },
                stream,
            );
        }

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
            num_pages: to_i64(Self::NAME, args.key.dim(0))?,
            page_size: to_i64(Self::NAME, page_size)?,
            q_heads: to_i64(Self::NAME, q_heads)?,
            kv_heads: to_i64(Self::NAME, kv_heads)?,
            head_dim: to_i64(Self::NAME, head_dim)?,
            q_stride_tokens: to_i64(Self::NAME, row_stride(&args.query))?,
            q_stride_heads: to_i64(Self::NAME, head_stride(&args.query))?,
            kv_stride_page: to_i64(Self::NAME, args.key.stride(0))?,
            kv_stride_tokens: to_i64(Self::NAME, args.key.stride(1))?,
            kv_stride_heads: to_i64(Self::NAME, args.key.stride(2))?,
            o_stride_tokens: to_i64(Self::NAME, row_stride(&args.out))?,
            o_stride_heads: to_i64(Self::NAME, head_stride(&args.out))?,
            scale: f64::from(args.scale),
            window_left: i64::from(args.window_left),
            causal,
            workspace_bytes: to_i64(Self::NAME, args.workspace.numel())?,
            dtype_query: abi::code(descriptor.query)?,
            dtype_key_value: abi::code(descriptor.key_value)?,
            dtype_output: abi::code(descriptor.output)?,
            phase: descriptor.phase as i64,
            error: [0; MESSAGE],
        };
        // SAFETY: `Call` is the call structure this package's entry declares: the declaration is
        // generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// The same bound as `dense_attention`, for the same reason and with the same declared bound.
    ///
    /// Paging changes *where* the `max_kv_len` products of one row come from — pages addressed by the
    /// scheduler instead of a contiguous run — and not how many there are, nor the dtype they
    /// accumulate in (FP32). So the reduction length is `max_kv_len` (a descriptor field here: the
    /// deployment's cache length, known at selection time), `rtol = γₙ + u` for the reduction and the
    /// write, and `atol` is the normalized-weights' value term `(2γₙ + u) · max|v|`.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.max_kv_len);
        let accumulate = DType::F32;
        let gamma = tolerance::gamma(reduction, accumulate);
        let value_term = 2.0 * gamma + tolerance::unit_roundoff(accumulate);
        Tolerance::rounding(reduction, accumulate, descriptor.output)
            .with_atol((value_term * f64::from(tolerance::VALUE_ENVELOPE)) as f32)
    }
}

/// The numbers one paged launch is sized by: its rows, its heads, its page and its longest query.
///
/// They are launch-time, not descriptor-time — two launches of one descriptor can differ in every
/// field — so they travel as one value. Six adjacent `u64` arguments are transposable by a call site
/// and the compiler says nothing: `q_heads` and `kv_heads` swapped is a launch of the wrong grid with
/// both values in range, which is the defect class this crate keeps removing by making it
/// unrepresentable rather than by reading more carefully.
struct PagedLaunch {
    batch: u64,
    q_heads: u64,
    kv_heads: u64,
    page_size: u64,
    tokens: u64,
    max_q_len: u64,
}

/// One word of the Triton programs' signatures, in the order the programs declare them.
///
/// The launch packs these in [`words`]' order, each named for the parameter it fills
/// ([`PagedWord::name`]), so the module's own signature refuses a launch whose list it does not
/// take. The list exists because the prefill kernel takes its offsets immediately after
/// `seq_lens` while the decode kernel has none, the one position that is easy to pack at the end.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PagedWord {
    Out,
    Query,
    Key,
    Value,
    KeyScale,
    ValueScale,
    PageTable,
    SeqLens,
    /// The packed query offsets, prefill only.
    CumSeqLensQ,
    QueryRow,
    QueryHead,
    QueryElement,
    KeyPage,
    KeyToken,
    KeyHead,
    KeyElement,
    ValuePage,
    ValueToken,
    ValueHead,
    ValueElement,
    OutRow,
    OutHead,
    OutElement,
    /// The page table's width: the most pages any row may name.
    MaxPages,
    PageSize,
    Scale,
    /// The longest query row, prefill only.
    MaxQueryRow,
    /// Whether the batch carries offsets, prefill only.
    Packed,
    /// The launch's query rows, prefill only: the tokens past the last batch row's end up to this
    /// count are padding, which the program writes as zero.
    Tokens,
    /// The page table's rows, prefill only: the batch rows a query tile is searched among.
    Rows,
}

impl PagedWord {
    /// The name the Triton programs declare this word under.
    fn name(self) -> &'static str {
        match self {
            Self::Out => "out_ptr",
            Self::Query => "query_ptr",
            Self::Key => "key_ptr",
            Self::Value => "value_ptr",
            Self::KeyScale => "key_scale_ptr",
            Self::ValueScale => "value_scale_ptr",
            Self::PageTable => "page_table_ptr",
            Self::SeqLens => "seq_lens_ptr",
            Self::CumSeqLensQ => "cum_seq_lens_q_ptr",
            Self::QueryRow => "q_stride_tokens",
            Self::QueryHead => "q_stride_heads",
            Self::QueryElement => "q_stride_dim",
            Self::KeyPage => "k_stride_pages",
            Self::KeyToken => "k_stride_tokens",
            Self::KeyHead => "k_stride_heads",
            Self::KeyElement => "k_stride_dim",
            Self::ValuePage => "v_stride_pages",
            Self::ValueToken => "v_stride_tokens",
            Self::ValueHead => "v_stride_heads",
            Self::ValueElement => "v_stride_dim",
            Self::OutRow => "o_stride_tokens",
            Self::OutHead => "o_stride_heads",
            Self::OutElement => "o_stride_dim",
            Self::MaxPages => "max_pages",
            Self::PageSize => "page_size",
            Self::Scale => "softmax_scale",
            Self::MaxQueryRow => "max_q_len",
            Self::Packed => "packed",
            Self::Tokens => "tokens",
            Self::Rows => "rows",
        }
    }
}

/// The words one phase's launch packs, in the program's own order.
fn words(phase: Phase) -> Vec<PagedWord> {
    let mut words = vec![
        PagedWord::Out,
        PagedWord::Query,
        PagedWord::Key,
        PagedWord::Value,
        PagedWord::KeyScale,
        PagedWord::ValueScale,
        PagedWord::PageTable,
        PagedWord::SeqLens,
    ];
    // The program's own order: a packed prefill's row offsets come straight after the sequence
    // lengths, before any stride. Pushing them at the end packed every later value one slot early -
    // `q_stride_tokens` received `q_stride_heads`, and the program dereferenced a stride as the
    // `cum_seq_lens_q` pointer, which is a device fault rather than a wrong number.
    if phase == Phase::Prefill {
        words.push(PagedWord::CumSeqLensQ);
    }
    words.extend([
        PagedWord::QueryRow,
        PagedWord::QueryHead,
        PagedWord::QueryElement,
        PagedWord::KeyPage,
        PagedWord::KeyToken,
        PagedWord::KeyHead,
        PagedWord::KeyElement,
        PagedWord::ValuePage,
        PagedWord::ValueToken,
        PagedWord::ValueHead,
        PagedWord::ValueElement,
        PagedWord::OutRow,
        PagedWord::OutHead,
        PagedWord::OutElement,
        PagedWord::MaxPages,
        PagedWord::PageSize,
        PagedWord::Scale,
    ]);
    if phase == Phase::Prefill {
        // The prefill program's own words follow the scale (the parameter list of
        // `triton/src/paged_attention.py`'s prefill program: the strides, the extents, the scale,
        // then the row bound, the pack flag, the token count and the row count).
        words.extend([
            PagedWord::MaxQueryRow,
            PagedWord::Packed,
            PagedWord::Tokens,
            PagedWord::Rows,
        ]);
    }
    words
}

impl PagedAttention {
    /// Launch this crate's own Triton program for a page the vendor programs do not read: an
    /// `E4M3` page, or a dense page (BF16, F16 or F32) of another type than the queries.
    ///
    /// The vendor libraries refuse a plane carrying an f32 page scale, by name, and trtllm-gen's
    /// artifact holds dense kernels only for a cache of the queries' own type, so those caches
    /// need a program of ours. It widens each element to f32 as it reads it, multiplying an
    /// `E4M3` page's scale back, and is handed null scale pointers, never read, for a dense page.
    /// **It serves both phases from two kernels of one module**: decode walks one program per
    /// (query row, query head) along the row's page table with a running maximum, normaliser and
    /// accumulator; prefill runs one program per tile of a batch row's query tokens and one
    /// key-value head ([`PREFILL_QUERY_TILE`]), places each row's queries at the end of its keys,
    /// masks each at its own key, and writes zero to the tokens past the last row's end.
    ///
    /// The refusals here are the operator's own conditions stated where the program's compilation
    /// makes them true: a cache of a type the program does not read, an E4M3 cache read without its
    /// scale, a window this program was not compiled for. Each names what it refused, so a store
    /// that resolved the wrong program is a launch error rather than plausible attention.
    #[allow(clippy::too_many_arguments)]
    fn launch_program(
        kernel: &Kernel,
        descriptor: &Descriptor,
        args: &Args,
        launch: PagedLaunch,
        stream: &Stream,
    ) -> Result<()> {
        // The page scales: an E4M3 page's own, and the null address for a dense page, whose
        // program never reads them (`QUANTIZED` is compiled from the descriptor's `kv`).
        let null = DevicePtr::from_address(0);
        let (key_scale, value_scale) = match descriptor.key_value {
            DType::F8E4M3 => {
                let (Some(key), Some(value)) = (&args.key_scale, &args.value_scale) else {
                    return Err(Error::argument(
                        Self::NAME,
                        "an E4M3 page is read through one scale per page, and the call passes none"
                            .to_owned(),
                    ));
                };
                (key.ptr, value.ptr)
            }
            DType::BF16 | DType::F16 | DType::F32 => (null, null),
            other => {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the Triton program reads an E4M3, BF16, F16 or F32 page, and the \
                         descriptor's cache is {}",
                        abi::name(other)
                    ),
                ));
            }
        };
        // `-1` is the encoding for "no window", which is what a program compiled without one
        // needs; a window is set when the value is not negative. A check against zero refuses the
        // no-window call, which is every causal deployment's.
        if args.window_left >= 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the Triton program is compiled without an attention window; the call asks for                      a {} token window",
                    args.window_left
                ),
            ));
        }
        if !launch.q_heads.is_multiple_of(launch.kv_heads) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} query heads do not divide into {} key-value heads",
                    launch.q_heads, launch.kv_heads
                ),
            ));
        }

        // The program's words, in the order its signature declares them and each named for its
        // parameter: the module refuses a list it does not take, and lays each value out in the
        // parameter's 32-bit type, refusing a stride or an extent the type does not hold rather
        // than truncating it.
        let packed = args.cum_seq_lens_q.numel() > 0;
        let mut arguments = Arguments::new(Self::NAME);
        for parameter in words(descriptor.phase) {
            let name = parameter.name();
            let _ = match parameter {
                PagedWord::Out => arguments.view(name, &args.out),
                PagedWord::Query => arguments.view(name, &args.query),
                PagedWord::Key => arguments.view(name, &args.key),
                PagedWord::Value => arguments.view(name, &args.value),
                PagedWord::KeyScale => arguments.pointer(name, key_scale),
                PagedWord::ValueScale => arguments.pointer(name, value_scale),
                PagedWord::PageTable => arguments.view(name, &args.page_table),
                PagedWord::SeqLens => arguments.view(name, &args.seq_lens),
                // Passed whether or not the batch is packed: `packed` is what the program branches
                // on, and an unpacked call leaves the pointer unread.
                PagedWord::CumSeqLensQ => arguments.view(name, &args.cum_seq_lens_q),
                PagedWord::QueryRow => arguments.u64(name, args.query.stride(0)),
                PagedWord::QueryHead => arguments.u64(name, args.query.stride(1)),
                PagedWord::QueryElement => arguments.u64(name, args.query.stride(2)),
                PagedWord::KeyPage => arguments.u64(name, args.key.stride(0)),
                PagedWord::KeyToken => arguments.u64(name, args.key.stride(1)),
                PagedWord::KeyHead => arguments.u64(name, args.key.stride(2)),
                PagedWord::KeyElement => arguments.u64(name, args.key.stride(3)),
                PagedWord::ValuePage => arguments.u64(name, args.value.stride(0)),
                PagedWord::ValueToken => arguments.u64(name, args.value.stride(1)),
                PagedWord::ValueHead => arguments.u64(name, args.value.stride(2)),
                PagedWord::ValueElement => arguments.u64(name, args.value.stride(3)),
                PagedWord::OutRow => arguments.u64(name, args.out.stride(0)),
                PagedWord::OutHead => arguments.u64(name, args.out.stride(1)),
                PagedWord::OutElement => arguments.u64(name, args.out.stride(2)),
                PagedWord::MaxPages => arguments.u64(name, args.page_table.dim(1)),
                PagedWord::PageSize => arguments.u64(name, launch.page_size),
                PagedWord::Scale => arguments.f32(name, args.scale),
                PagedWord::MaxQueryRow => arguments.u64(name, launch.max_q_len),
                PagedWord::Packed => arguments.flag(name, packed),
                PagedWord::Tokens => arguments.u64(name, launch.tokens),
                PagedWord::Rows => arguments.u64(name, launch.batch),
            };
        }

        let grid = match descriptor.phase {
            // One program per (query row, query head): a decode's table row is one query token.
            Phase::Decode => LaunchGrid::new([launch.batch, launch.q_heads, 1])?,
            // One program per query tile and key-value head. The rows' lengths are only known
            // from the offsets on the device, so the first axis counts the tiles any split of
            // `tokens` among `batch` rows can need, with the padding past the last row's
            // (`cdiv(tokens, BLOCK_Q) + batch`); a program whose number no row's tile takes
            // returns at once. The count grows with the tokens and the rows, not their product.
            Phase::Prefill => {
                let tile = kernel.cubin(Self::NAME)?.integer(PREFILL_QUERY_TILE)?;
                LaunchGrid::new([
                    launch.tokens.div_ceil(tile) + launch.batch,
                    launch.kv_heads,
                    1,
                ])?
            }
        };
        kernel.launch(grid, &mut arguments, stream)
    }
}

/// Whether the cache's pages hold quantized elements that are read through a per-page scale.
fn quantized(dtype: DType) -> bool {
    matches!(dtype, DType::F8E4M3 | DType::F4)
}

/// The elements between consecutive rows of a `[rows, heads, dim]` view.
///
/// Every attention program takes a row's stride and a head's stride as separate launch values, so an
/// axis is named here by its distance from the last one — element last, head second to last, row
/// third to last — and never by its index in a rank-2 view. The distinction is not cosmetic: reading
/// the head axis as the row axis states `head_dim` where the program expects `heads * head_dim`, and
/// one element where it expects `head_dim`, which walks every head of a row two bytes into the row
/// and breaks the sixteen-byte vector loads the sm100 programs are compiled for.
pub(super) fn row_stride(view: &Operand) -> u64 {
    axis_stride(view, 2)
}

/// The elements between consecutive heads of a `[rows, heads, dim]` view.
pub(super) fn head_stride(view: &Operand) -> u64 {
    axis_stride(view, 1)
}

/// The stride of the axis `from_last` positions from the last, or `0` when the view has no such axis.
fn axis_stride(view: &Operand, from_last: usize) -> u64 {
    let rank = view.rank();
    match rank.checked_sub(from_last + 1) {
        Some(axis) => view.stride(axis),
        None => 0,
    }
}

/// The alignment every attention program states for its tensor operands.
///
/// The tensor-core programs move rows with sixteen-byte vector loads and 128-byte-swizzled TMA
/// boxes (`gmemalign16`) and take `i64` strides, so an operand's base and every axis they walk must
/// fall on a whole sixteen-byte unit; the dense, paged and sparse attention operators all hold their
/// operands to it.
pub(super) const ATTENTION_ALIGNMENT: u64 = 16;

/// Refuse an operand the tensor-core programs cannot address.
///
/// Their descriptors state a sixteen-byte-aligned tensor pointer (`gmemalign16`) and take the strides
/// of the axes they walk, so every address they form must be a whole number of sixteen-byte units:
/// the base, and each stride of an axis that is not the vectorized last one. A view carved at an odd
/// element offset, or with a stride that is not a whole unit, faults inside the kernel with no word
/// about which operand — a warp misaligned address and a program counter — so it is refused here
/// instead, by name.
pub(super) fn check_vector_alignment(
    operator: &'static str,
    name: &str,
    view: &Operand,
) -> Result<()> {
    // A four-bit element type packs two elements a byte, so "sixteen bytes" is not a whole number of
    // its elements and the packages that read one state their own alignment.
    let Some(bytes) = abi::bytes(view.dtype).filter(|bytes| *bytes > 0) else {
        return Ok(());
    };
    if !ATTENTION_ALIGNMENT.is_multiple_of(bytes) {
        return Ok(());
    }
    let unit = ATTENTION_ALIGNMENT / bytes;
    let address = view.ptr.address();
    if !address.is_multiple_of(ATTENTION_ALIGNMENT) {
        return Err(Error::argument(
            operator,
            format!(
                "the {name} starts at {address:#x}, which is not {} bytes aligned",
                ATTENTION_ALIGNMENT
            ),
        ));
    }
    let rank = view.rank();
    for axis in 0..rank.saturating_sub(1) {
        let stride = view.stride(axis);
        if !stride.is_multiple_of(unit) {
            return Err(Error::argument(
                operator,
                format!(
                    "the {name} has a stride of {stride} elements on axis {axis}, which is not a \
                     whole number of {unit}-element ({}-byte) units",
                    ATTENTION_ALIGNMENT
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn address(pointer: DevicePtr) -> u64 {
    pointer.address()
}

/// An extent as the ABI's `i64`, or an argument error naming `operator`.
pub(super) fn to_i64(operator: &'static str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument(operator, "an extent above i64::MAX"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A BF16 causal decode module of 32 query and 4 KV heads of 128 at TP=8, page size 64.
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
            long.atol > 0.0 && long.atol <= tolerance::VALUE_ENVELOPE,
            "atol {}",
            long.atol
        );
        // Two implementations of the same kernel agree exactly, so a zero difference is allowed.
        assert!(long.allows(1.0, 1.0));
        // A dropped history window changes the output by far more than the bound.
        assert!(!long.allows(0.0, 1.0));
    }

    /// A `[tokens, heads, dim]` view states its row stride and its head stride, not the two strides
    /// nearest the last axis: 32 heads of 128 elements put the next token 4096 elements on, and the
    /// next head 128. Reading the head axis as the row axis is what the launch cannot recover from,
    /// because the program then walks heads two bytes apart and its vector loads fault.
    #[test]
    fn a_rows_strides_are_the_rows_and_not_the_head() {
        let view = Operand::new(
            DevicePtr::from_address(0x1_0000),
            DType::BF16,
            &[15, 32, 128],
        );
        assert_eq!(row_stride(&view), 4096);
        assert_eq!(head_stride(&view), 128);
    }

    /// The eighteen-byte and four-byte cases: a view whose base or whose walked stride is not a whole
    /// sixteen-byte unit is refused before the launch, naming the operand.
    #[test]
    fn a_misaligned_operand_is_refused_by_name() {
        let ok = Operand::new(
            DevicePtr::from_address(0x1_0000),
            DType::BF16,
            &[15, 32, 128],
        );
        check_vector_alignment("test", "query", &ok).unwrap();

        let odd_base = Operand::new(
            DevicePtr::from_address(0x1_0002),
            DType::BF16,
            &[15, 32, 128],
        );
        let refusal = check_vector_alignment("test", "query", &odd_base).unwrap_err();
        assert!(
            format!("{refusal}").contains("16 bytes aligned"),
            "{refusal}"
        );

        // A row of 100 elements of bf16 is 200 bytes: every second row starts four bytes off.
        let narrow = Operand::new(DevicePtr::from_address(0x1_0000), DType::BF16, &[4, 100]);
        let refusal = check_vector_alignment("test", "key", &narrow).unwrap_err();
        assert!(format!("{refusal}").contains("stride of 100"), "{refusal}");
    }
}
