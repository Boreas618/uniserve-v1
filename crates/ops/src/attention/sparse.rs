//! Video Sparse Attention: complete block-sparse attention over selected key blocks.
//!
//! H3's video backbone does not attend densely: every query tile keeps a small set of `tile`-row
//! key blocks, chosen per head and per query tile by a score the caller computes. This operator
//! evaluates exactly the selected blocks and nothing else, so its cost follows the selection rather
//! than the sequence length (specs/worker.md 9, "VSA SM100 64 行").
//!
//! ```text
//! query   [query_rows, heads, head_dim]          rows are whole tiles
//! key     [key_rows, heads, head_dim]            rows are whole blocks
//! value   [key_rows, heads, head_dim]
//! out     [query_rows, heads, head_dim]
//! indices [heads, query_tiles, selected] i32     global key-block ids
//! counts  [heads, query_tiles] i32               how many of each row are read
//! valid   [key_blocks] i32                       valid rows of each block, 0..=tile
//! lse     [heads, query_rows] f32                when the descriptor asks for it
//! ```
//!
//! For query row `t` of head `h` inside query tile `q`, with `s_j = scale · q_t,h · k_{j,h}` over the
//! keys `j` of the blocks `indices[h, q, 0..counts[h, q]]` that are visible (a key row past its
//! block's `valid` entry is not):
//!
//! ```text
//! p_j     = exp(s_j − max_j s_j)
//! out_t,h = Σ_j p_j v_{j,h} / Σ_j p_j
//! lse_t,h = ln Σ_j exp(s_j)        (FP32, natural logarithm)
//! ```
//!
//! accumulated in FP32 and rounded once to the output dtype. A query tile that selects no visible
//! key writes zeros (and a negative-infinite `lse`), which is what a layout's unused capacity looks
//! like; the kernel reads the selection from the device, so one captured call serves any selection
//! of the same shape.
//!
//! # The bound between two implementations
//!
//! The output is a convex combination of the values it reads, so `Σ_j |p_j v_j| ≤ max_j |v_j|` and
//! the standard bound (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1)
//! gives, for FP32 accumulation with unit round-off `u` and `n` visible keys,
//!
//! ```text
//! |out − out_exact| ≤ (γₙ + u) V + u_out |out|,   γₙ = n·u / (1 − n·u)
//! ```
//!
//! so two implementations differ by at most `2(γₙ + u) V + 2 u_out |out|`. The `V` term does not
//! scale with the result, so it enters the tolerance as an absolute term over the declared element
//! envelope [`VALUE_ENVELOPE`], and the reduction length is the descriptor's
//! [`SparseDescriptor::keys`]: the largest key extent a module serves, which bounds every launch
//! because a query attends at most every key row.
//!
//! # The package
//!
//! A native library per backend (`sm100`, `sm90`, `sm120`). The backend owns the whole host side —
//! its cubin or CuTe module builds, the TMA tensor maps over the strided `[rows, heads, head_dim]`
//! views, the dynamic shared-memory size, the cluster launch, the packed divisors it divides work
//! items by, and the specialization it picks for a shape — so [`Op::launch`] computes no grid and
//! hands the library [`SparseCall`]. The SM100 package builds its 64-row and 128-row kernels as two
//! modules and the descriptor's `block` chooses between them; a package that compiles several
//! specializations picks inside the call, by the descriptor and the launch extents.
//!
//! `key` and `value` may use independent strides: the kernels address rows and heads through the
//! strides they are given, so a fused K/V projection that writes both into one allocation needs no
//! copy.

use std::ffi::c_void;
use std::fmt;

use super::{address, head_stride, row_stride, to_i64};
use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The largest value magnitude the sparse attention bound's absolute term is written for.
///
/// VSA reads the value projection of a normalized residual stream, as paged attention does; an
/// element stays below 8 in the models served here (BF16 serving). The envelope is a declaration
/// about that data, not something the descriptor can state.
pub const VALUE_ENVELOPE: f32 = 8.0;

/// The smallest alignment a query, key, value or output base address must have.
///
/// The kernels move rows with 128-byte-swizzled TMA boxes, which requires the tensor base to be
/// 16-byte aligned and the row and head strides to be whole 8-element groups.
pub const BASE_ALIGNMENT: u64 = 16;

/// One compilation of `sparse_attention`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SparseDescriptor {
    /// Rows of one query tile and of one key block.
    pub block: u32,
    /// Element type of the queries, keys and values.
    pub dtype: DType,
    /// Element type of the output.
    pub output: DType,
    /// Elements per head.
    pub head_dim: u32,
    /// Whether the call writes the natural-log softmax normalizer.
    pub lse: bool,
    /// The largest key extent (rows) the module serves.
    pub keys: u32,
}

impl fmt::Display for SparseDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("block", self.block)
            .field("dtype", self.dtype)
            .field("output", self.output)
            .field("head_dim", self.head_dim)
            .field("lse", self.lse)
            .field("keys", self.keys)
            .finish(formatter)
    }
}

/// The arguments of one block-sparse attention call.
pub struct SparseArgs {
    /// `[query_rows, heads, head_dim]` queries; rows are whole tiles.
    pub query: TensorView,
    /// `[key_rows, heads, head_dim]` keys; rows are whole blocks.
    pub key: TensorView,
    /// `[key_rows, heads, head_dim]` values.
    pub value: TensorView,
    /// `[heads, query_tiles, selected]` i32: the key blocks each query tile reads first.
    pub indices: TensorView,
    /// `[heads, query_tiles]` i32: how many of each row's indices are read.
    pub counts: TensorView,
    /// `[key_blocks]` i32: valid rows of each key block.
    pub valid: TensorView,
    /// `[heads, query_rows]` f32: the softmax normalizer, when the descriptor asks for it.
    pub lse: Option<TensorView>,
    /// `[query_rows, heads, head_dim]` output.
    pub out: TensorView,
    /// Softmax scale applied to `q · k` before the exponential.
    pub scale: f32,
}

/// The call structure a package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/sparse_attention/<backend>/include/uniserve_sparse_attention.h`;
/// the layout is pinned by [`SPARSE_CALL_LAYOUT`], the assertions below and the test beside them.
#[repr(C)]
#[derive(Debug)]
pub struct SparseCall {
    /// `[query_rows, heads, head_dim]` queries.
    pub query: *const c_void,
    /// `[key_rows, heads, head_dim]` keys.
    pub key: *const c_void,
    /// `[key_rows, heads, head_dim]` values.
    pub value: *const c_void,
    /// `[query_rows, heads, head_dim]` output.
    pub out: *mut c_void,
    /// `[heads, query_rows]` f32 normalizer, or null when the descriptor has none.
    pub lse: *mut c_void,
    /// `[heads, query_tiles, selected]` i32 key-block ids.
    pub indices: *const c_void,
    /// `[heads, query_tiles]` i32 read counts.
    pub counts: *const c_void,
    /// `[key_blocks]` i32 valid rows of each block.
    pub valid: *const c_void,
    /// Heads of the call.
    pub heads: i64,
    /// Query rows: whole tiles.
    pub query_rows: i64,
    /// Key rows: whole blocks.
    pub key_rows: i64,
    /// Elements between consecutive query rows.
    pub q_row_stride: i64,
    /// Elements between consecutive query heads.
    pub q_head_stride: i64,
    /// Elements between consecutive output rows.
    pub o_row_stride: i64,
    /// Elements between consecutive output heads.
    pub o_head_stride: i64,
    /// Elements between consecutive key rows.
    pub k_row_stride: i64,
    /// Elements between consecutive key heads.
    pub k_head_stride: i64,
    /// Elements between consecutive value rows.
    pub v_row_stride: i64,
    /// Elements between consecutive value heads.
    pub v_head_stride: i64,
    /// The widest selection a query tile has: the `indices` last extent.
    pub selected: i64,
    /// Elements per head.
    pub head_dim: i64,
    /// `DType`'s ABI number of the queries, keys and values.
    pub dtype: i64,
    /// `DType`'s ABI number of the output.
    pub output_dtype: i64,
    /// The softmax scale.
    pub scale: f32,
    /// The entry's message on a nonzero return.
    pub error: [u8; 512],
}

/// The size and field offsets the package's header must mirror, in bytes.
pub const SPARSE_CALL_LAYOUT: SparseCallLayout = SparseCallLayout {
    size: 704,
    query: 0,
    key: 8,
    value: 16,
    out: 24,
    lse: 32,
    indices: 40,
    counts: 48,
    valid: 56,
    heads: 64,
    query_rows: 72,
    key_rows: 80,
    q_row_stride: 88,
    q_head_stride: 96,
    o_row_stride: 104,
    o_head_stride: 112,
    k_row_stride: 120,
    k_head_stride: 128,
    v_row_stride: 136,
    v_head_stride: 144,
    selected: 152,
    head_dim: 160,
    dtype: 168,
    output_dtype: 176,
    scale: 184,
    error: 188,
};

/// The layout of [`SparseCall`], for the header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct SparseCallLayout {
    pub size: usize,
    pub query: usize,
    pub key: usize,
    pub value: usize,
    pub out: usize,
    pub lse: usize,
    pub indices: usize,
    pub counts: usize,
    pub valid: usize,
    pub heads: usize,
    pub query_rows: usize,
    pub key_rows: usize,
    pub q_row_stride: usize,
    pub q_head_stride: usize,
    pub o_row_stride: usize,
    pub o_head_stride: usize,
    pub k_row_stride: usize,
    pub k_head_stride: usize,
    pub v_row_stride: usize,
    pub v_head_stride: usize,
    pub selected: usize,
    pub head_dim: usize,
    pub dtype: usize,
    pub output_dtype: usize,
    pub scale: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<SparseCall>() == SPARSE_CALL_LAYOUT.size,
    "the sparse_attention call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(SparseCall, valid) == SPARSE_CALL_LAYOUT.valid,
    "the sparse_attention call structure's device pointers moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(SparseCall, heads) == SPARSE_CALL_LAYOUT.heads,
    "the sparse_attention call structure's geometry moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(SparseCall, selected) == SPARSE_CALL_LAYOUT.selected,
    "the sparse_attention call structure's selection moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(SparseCall, scale) == SPARSE_CALL_LAYOUT.scale,
    "the sparse_attention call structure's scalars moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(SparseCall, error) == SPARSE_CALL_LAYOUT.error,
    "the sparse_attention call structure's `error` moved without its header"
);

impl NativeCall for SparseCall {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// Block-sparse attention over selected key blocks.
pub struct SparseAttention;

impl Op for SparseAttention {
    const NAME: &'static str = "sparse_attention";

    type Descriptor = SparseDescriptor;
    type Args<'a> = SparseArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let block = u64::from(descriptor.block);
        let head_dim = u64::from(descriptor.head_dim);
        let operator = Self::NAME;

        args.query.require_rank(operator, 3)?;
        args.query
            .require_row(operator, descriptor.dtype, head_dim)?;

        let heads = args.query.shape.dim(1);
        let query_rows = args.query.shape.dim(0);
        let key_rows = args.key.shape.dim(0);

        if query_rows == 0 || key_rows == 0 {
            return Err(Error::argument(
                operator,
                "a call reads at least one query tile and one key block",
            ));
        }
        if heads == 0 {
            return Err(Error::argument(operator, "a call has at least one head"));
        }
        // Tiles are whole: a partial tile is the caller's padding, addressed by `valid`.
        if query_rows % block != 0 || key_rows % block != 0 {
            return Err(Error::argument(
                operator,
                format!(
                    "query rows ({query_rows}) and key rows ({key_rows}) must be whole {block}-row \
                     tiles"
                ),
            ));
        }
        // `keys` is what the module's bound is written for; a longer key extent is another module.
        if key_rows > u64::from(descriptor.keys) {
            return Err(Error::argument(
                operator,
                format!(
                    "the module serves {0} key rows, the call passes {key_rows}",
                    descriptor.keys
                ),
            ));
        }
        args.key.require_rank(operator, 3)?;
        args.key.require_row(operator, descriptor.dtype, head_dim)?;
        args.value.require_rank(operator, 3)?;
        args.value
            .require_row(operator, descriptor.dtype, head_dim)?;
        args.out.require_rank(operator, 3)?;
        args.out
            .require_row(operator, descriptor.output, head_dim)?;
        for (name, view) in [("key", &args.key), ("value", &args.value)] {
            if view.shape.dim(1) != heads || view.shape.dim(0) != key_rows {
                return Err(Error::argument(
                    operator,
                    format!(
                        "expected {name} of shape [{key_rows}, {heads}, {head_dim}], found {:?}",
                        view.shape.dims()
                    ),
                ));
            }
        }
        if !args.out.shape.matches(&args.query.shape) {
            return Err(Error::argument(
                operator,
                format!(
                    "the output takes the query's shape {:?}, found {:?}",
                    args.query.shape.dims(),
                    args.out.shape.dims()
                ),
            ));
        }
        if args.out.ptr == args.query.ptr || args.out.ptr == args.key.ptr {
            return Err(Error::argument(
                operator,
                "the output must not alias the query, key or value",
            ));
        }
        // The staged metadata tables: one int32 per entry, contiguous, read on the device.
        let query_tiles = query_rows / block;
        args.indices.require_rank(operator, 3)?;
        if args.indices.dtype != DType::I32
            || args.indices.shape.dim(0) != heads
            || args.indices.shape.dim(1) != query_tiles
            || args.indices.shape.dim(2) == 0
        {
            return Err(Error::argument(
                operator,
                format!(
                    "expected i32 indices [{heads}, {query_tiles}, selected] with selected > 0, \
                     found {} {:?}",
                    args.indices.dtype,
                    args.indices.shape.dims()
                ),
            ));
        }
        args.counts
            .require_shape(operator, Shape::new(&[heads, query_tiles]))?;
        if args.counts.dtype != DType::I32 {
            return Err(Error::argument(
                operator,
                format!("expected i32 counts, found {}", args.counts.dtype),
            ));
        }
        args.valid
            .require_shape(operator, Shape::new(&[key_rows / block]))?;
        if args.valid.dtype != DType::I32 {
            return Err(Error::argument(
                operator,
                format!("expected i32 valid sizes, found {}", args.valid.dtype),
            ));
        }
        match (descriptor.lse, args.lse) {
            (true, Some(lse)) => {
                lse.require_shape(operator, Shape::new(&[heads, query_rows]))?;
                if lse.dtype != DType::F32 {
                    return Err(Error::argument(
                        operator,
                        format!("expected an f32 normalizer, found {}", lse.dtype),
                    ));
                }
            }
            (false, None) => {}
            (true, None) => {
                return Err(Error::argument(
                    operator,
                    "the module writes a normalizer and the call passes none",
                ));
            }
            (false, Some(_)) => {
                return Err(Error::argument(
                    operator,
                    "the call passes a normalizer the module does not write",
                ));
            }
        }
        if !args.scale.is_finite() || args.scale <= 0.0 {
            return Err(Error::argument(
                operator,
                format!("expected a positive finite scale, found {}", args.scale),
            ));
        }
        // 128-byte-swizzled TMA boxes: 16-byte base alignment and whole 8-element stride groups.
        for (name, view) in [
            ("query", &args.query),
            ("key", &args.key),
            ("value", &args.value),
            ("out", &args.out),
        ] {
            if !address(view.ptr).is_multiple_of(BASE_ALIGNMENT) {
                return Err(Error::argument(
                    operator,
                    format!(
                        "{name} must be {BASE_ALIGNMENT}-byte aligned, found {}",
                        address(view.ptr)
                    ),
                ));
            }
            for axis in [0, 1] {
                if !view.strides.get(axis).is_multiple_of(8) {
                    return Err(Error::argument(
                        operator,
                        format!(
                            "{name} axis {axis} stride {} is not a whole 8-element group",
                            view.strides.get(axis)
                        ),
                    ));
                }
            }
        }

        let mut call = SparseCall {
            query: address(args.query.ptr) as usize as *const c_void,
            key: address(args.key.ptr) as usize as *const c_void,
            value: address(args.value.ptr) as usize as *const c_void,
            out: address(args.out.ptr) as usize as *mut c_void,
            lse: match args.lse {
                Some(lse) => address(lse.ptr) as usize as *mut c_void,
                None => std::ptr::null_mut(),
            },
            indices: address(args.indices.ptr) as usize as *const c_void,
            counts: address(args.counts.ptr) as usize as *const c_void,
            valid: address(args.valid.ptr) as usize as *const c_void,
            heads: to_i64(operator, heads)?,
            query_rows: to_i64(operator, query_rows)?,
            key_rows: to_i64(operator, key_rows)?,
            q_row_stride: to_i64(operator, row_stride(&args.query))?,
            q_head_stride: to_i64(operator, head_stride(&args.query))?,
            o_row_stride: to_i64(operator, row_stride(&args.out))?,
            o_head_stride: to_i64(operator, head_stride(&args.out))?,
            k_row_stride: to_i64(operator, row_stride(&args.key))?,
            k_head_stride: to_i64(operator, head_stride(&args.key))?,
            v_row_stride: to_i64(operator, row_stride(&args.value))?,
            v_head_stride: to_i64(operator, head_stride(&args.value))?,
            selected: to_i64(operator, args.indices.shape.dim(2))?,
            head_dim: to_i64(operator, head_dim)?,
            dtype: descriptor.dtype.as_i64(),
            output_dtype: descriptor.output.as_i64(),
            scale: args.scale,
            error: [0; 512],
        };
        // SAFETY: `SparseCall` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        SparseArgs {
            out: twins.shadow_of(args.out),
            lse: args.lse.map(|lse| twins.shadow_of(lse)),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.keys);
        let accumulate = DType::F32;
        let gamma = tolerance::gamma(reduction, accumulate);
        let value_term = 2.0 * gamma + accumulate.unit_roundoff();
        Tolerance::rounding(reduction, accumulate, descriptor.output)
            .with_atol((value_term * f64::from(VALUE_ENVELOPE)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SM100 64-row build H3's video backbone runs, BF16 out, with the normalizer.
    fn block64() -> SparseDescriptor {
        SparseDescriptor {
            block: 64,
            dtype: DType::BF16,
            output: DType::BF16,
            head_dim: 128,
            lse: true,
            keys: 65536,
        }
    }

    #[test]
    fn key_names_every_compiled_value() {
        assert_eq!(
            block64().to_string(),
            "block_64_dtype_bf16_output_bf16_head_dim_128_lse_true_keys_65536"
        );
    }

    #[test]
    fn key_separates_the_block_sizes_and_the_normalizer() {
        let mut block128 = block64();
        block128.block = 128;
        block128.lse = false;
        assert_ne!(block64().to_string(), block128.to_string());
        assert!(block128.to_string().starts_with("block_128_dtype_bf16"));
        assert!(block128.to_string().contains("_lse_false_"));
    }

    /// The header the packages mirror must carry these offsets; the constants beside the structure
    /// are checked against the compiler here.
    #[test]
    fn call_layout_is_the_one_the_header_declares() {
        let layout = SPARSE_CALL_LAYOUT;
        assert_eq!(std::mem::size_of::<SparseCall>(), layout.size);
        assert_eq!(std::mem::align_of::<SparseCall>(), 8);
        assert_eq!(std::mem::offset_of!(SparseCall, query), layout.query);
        assert_eq!(std::mem::offset_of!(SparseCall, key), layout.key);
        assert_eq!(std::mem::offset_of!(SparseCall, value), layout.value);
        assert_eq!(std::mem::offset_of!(SparseCall, out), layout.out);
        assert_eq!(std::mem::offset_of!(SparseCall, lse), layout.lse);
        assert_eq!(std::mem::offset_of!(SparseCall, indices), layout.indices);
        assert_eq!(std::mem::offset_of!(SparseCall, counts), layout.counts);
        assert_eq!(std::mem::offset_of!(SparseCall, valid), layout.valid);
        assert_eq!(std::mem::offset_of!(SparseCall, heads), layout.heads);
        assert_eq!(
            std::mem::offset_of!(SparseCall, query_rows),
            layout.query_rows
        );
        assert_eq!(std::mem::offset_of!(SparseCall, key_rows), layout.key_rows);
        assert_eq!(
            std::mem::offset_of!(SparseCall, q_row_stride),
            layout.q_row_stride
        );
        assert_eq!(
            std::mem::offset_of!(SparseCall, q_head_stride),
            layout.q_head_stride
        );
        assert_eq!(
            std::mem::offset_of!(SparseCall, o_row_stride),
            layout.o_row_stride
        );
        assert_eq!(
            std::mem::offset_of!(SparseCall, o_head_stride),
            layout.o_head_stride
        );
        assert_eq!(
            std::mem::offset_of!(SparseCall, k_row_stride),
            layout.k_row_stride
        );
        assert_eq!(
            std::mem::offset_of!(SparseCall, k_head_stride),
            layout.k_head_stride
        );
        assert_eq!(
            std::mem::offset_of!(SparseCall, v_row_stride),
            layout.v_row_stride
        );
        assert_eq!(
            std::mem::offset_of!(SparseCall, v_head_stride),
            layout.v_head_stride
        );
        assert_eq!(std::mem::offset_of!(SparseCall, selected), layout.selected);
        assert_eq!(std::mem::offset_of!(SparseCall, head_dim), layout.head_dim);
        assert_eq!(std::mem::offset_of!(SparseCall, dtype), layout.dtype);
        assert_eq!(
            std::mem::offset_of!(SparseCall, output_dtype),
            layout.output_dtype
        );
        assert_eq!(std::mem::offset_of!(SparseCall, scale), layout.scale);
        assert_eq!(std::mem::offset_of!(SparseCall, error), layout.error);
    }

    /// The bound is the standard one for a convex combination: relative to the result, plus the
    /// envelope term for a row whose output is smaller than the values it reads.
    #[test]
    fn the_bound_follows_the_key_extent() {
        let near = SparseAttention::tolerance(&SparseDescriptor {
            keys: 8192,
            ..block64()
        });
        let far = SparseAttention::tolerance(&block64());
        assert!(
            far.rtol >= near.rtol,
            "a longer key extent cannot tighten the bound"
        );
        assert!(far.atol >= near.atol);
        // A BF16 output's own rounding dominates: u_out = 2^-8 / 2.
        assert!(far.rtol > 0.0019 && far.rtol < 0.01, "rtol = {}", far.rtol);
        assert!(far.atol > 0.0 && far.atol < 0.1, "atol = {}", far.atol);
        assert!(
            far.allows(1.0, 1.002),
            "a 2e-3 deviation at |x| = 1 is inside BF16's rounding"
        );
        assert!(!far.allows(1.0, 1.5));
    }
}
