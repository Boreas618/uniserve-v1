//! Video Sparse Attention: complete block-sparse attention over selected key blocks.
//!
//! A video transformer need not attend densely: every query tile keeps a small set of `tile`-row
//! key blocks, chosen per head and per query tile by a score the caller computes. This operator
//! evaluates exactly the selected blocks and nothing else, so its cost follows the selection rather
//! than the sequence length.
//!
//! ```text
//! query   [query_rows, heads, head_dim]          rows are whole tiles
//! key     [key_rows, heads, head_dim]            rows are whole blocks
//! value   [key_rows, heads, head_dim]
//! out     [query_rows, heads, head_dim]
//! indices [heads, query_tiles, selected] u32     global key-block ids
//! counts  [heads, query_tiles] u32               how many of each row are read
//! valid   [key_blocks] u32                       valid rows of each block, 0..=tile
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
//! envelope [`tolerance::VALUE_ENVELOPE`], and the reduction length is the descriptor's
//! [`SparseDescriptor::keys`]: the largest key extent a module serves, which bounds every launch
//! because a query attends at most every key row.
//!
//! # The package
//!
//! A native library per backend (`tcgen05`, `wgmma`, `flashinfer`). The backend owns the whole host side —
//! its cubin or CuTe module builds, the TMA tensor maps over the strided `[rows, heads, head_dim]`
//! views, the dynamic shared-memory size, the cluster launch, the packed divisors it divides work
//! items by, and the specialization it picks for a shape — so [`Op::launch`] computes no grid and
//! hands the library [`SparseCall`]. The `tcgen05` package builds its 64-row and 128-row kernels as two
//! modules and the descriptor's `block` chooses between them; a package that compiles several
//! specializations picks inside the call, by the descriptor and the launch extents.
//!
//! `key` and `value` may use independent strides: the kernels address rows and heads through the
//! strides they are given, so a fused K/V projection that writes both into one allocation needs no
//! copy.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;

use super::{address, check_vector_alignment, head_stride, row_stride, to_i64};
use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};
use tensor::DType;

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
            .field("dtype", abi::name(self.dtype))
            .field("output", abi::name(self.output))
            .field("head_dim", self.head_dim)
            .field("lse", self.lse)
            .finish(formatter)
    }
}

/// The arguments of one block-sparse attention call.
#[derive(Clone)]
pub struct SparseArgs {
    /// `[query_rows, heads, head_dim]` queries; rows are whole tiles.
    pub query: Operand,
    /// `[key_rows, heads, head_dim]` keys; rows are whole blocks.
    pub key: Operand,
    /// `[key_rows, heads, head_dim]` values.
    pub value: Operand,
    /// `[heads, query_tiles, selected]` u32: the key blocks each query tile reads first.
    pub indices: Operand,
    /// `[heads, query_tiles]` u32: how many of each row's indices are read.
    pub counts: Operand,
    /// `[key_blocks]` u32: valid rows of each key block.
    pub valid: Operand,
    /// `[heads, query_rows]` f32: the softmax normalizer, when the descriptor asks for it.
    pub lse: Option<Operand>,
    /// `[query_rows, heads, head_dim]` output.
    pub out: Operand,
    /// Softmax scale applied to `q · k` before the exponential.
    pub scale: f32,
}

/// The call structure a package's entry receives, declared to the packages as
/// `uniserve_sparse_attention_call` (`uniserve_sparse_attention.h`, generated with its layout
/// asserted: [`abi`]).
///
/// One block-sparse attention launch over selected key blocks: rows are whole tiles of the
/// descriptor's block, which is the module's compilation, so a call whose rows do not divide it
/// is refused.
#[repr(C)]
#[derive(Debug, abi::Layout)]
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
    /// `[heads, query_tiles, selected]` u32 key-block ids.
    pub indices: *const c_void,
    /// `[heads, query_tiles]` u32 read counts.
    pub counts: *const c_void,
    /// `[key_blocks]` u32 valid rows of each block.
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
    pub error: [c_char; MESSAGE],
}

impl NativeCall for SparseCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// Block-sparse attention over selected key blocks.
pub struct SparseAttention;

impl Op for SparseAttention {
    const NAME: &'static str = proto::module::operator::SPARSE_ATTENTION;

    type Descriptor = SparseDescriptor;
    type Args<'a> = SparseArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let block = u64::from(descriptor.block);
        let head_dim = u64::from(descriptor.head_dim);
        let operator = Self::NAME;

        args.query.require_rank(operator, 3)?;
        args.query
            .require_row(operator, descriptor.dtype, head_dim)?;

        let heads = args.query.dim(1);
        let query_rows = args.query.dim(0);
        let key_rows = args.key.dim(0);

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
        // The declared bound; the program itself reads the extent from the key view.
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
            if view.dim(1) != heads || view.dim(0) != key_rows {
                return Err(Error::argument(
                    operator,
                    format!(
                        "expected {name} of shape [{key_rows}, {heads}, {head_dim}], found {:?}",
                        view.dims()
                    ),
                ));
            }
        }
        if args.out.dims() != args.query.dims() {
            return Err(Error::argument(
                operator,
                format!(
                    "the output takes the query's shape {:?}, found {:?}",
                    args.query.dims(),
                    args.out.dims()
                ),
            ));
        }
        if args.out.ptr == args.query.ptr || args.out.ptr == args.key.ptr {
            return Err(Error::argument(
                operator,
                "the output must not alias the query, key or value",
            ));
        }
        // The selection's tables: one 32-bit unsigned entry each, contiguous, read on the device.
        let query_tiles = query_rows / block;
        args.indices.require_rank(operator, 3)?;
        if args.indices.dtype != DType::U32
            || args.indices.dim(0) != heads
            || args.indices.dim(1) != query_tiles
            || args.indices.dim(2) == 0
        {
            return Err(Error::argument(
                operator,
                format!(
                    "expected u32 indices [{heads}, {query_tiles}, selected] with selected > 0, \
                     found {} {:?}",
                    abi::name(args.indices.dtype),
                    args.indices.dims()
                ),
            ));
        }
        args.counts.require_shape(operator, &[heads, query_tiles])?;
        if args.counts.dtype != DType::U32 {
            return Err(Error::argument(
                operator,
                format!(
                    "expected u32 counts, found {}",
                    abi::name(args.counts.dtype)
                ),
            ));
        }
        args.valid.require_shape(operator, &[key_rows / block])?;
        if args.valid.dtype != DType::U32 {
            return Err(Error::argument(
                operator,
                format!(
                    "expected u32 valid sizes, found {}",
                    abi::name(args.valid.dtype)
                ),
            ));
        }
        match (descriptor.lse, &args.lse) {
            (true, Some(lse)) => {
                lse.require_shape(operator, &[heads, query_rows])?;
                if lse.dtype != DType::F32 {
                    return Err(Error::argument(
                        operator,
                        format!("expected an f32 normalizer, found {}", abi::name(lse.dtype)),
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
        // The kernels move rows with 128-byte-swizzled TMA boxes: every attention program's
        // operand alignment.
        for (name, view) in [
            ("query", &args.query),
            ("key", &args.key),
            ("value", &args.value),
            ("out", &args.out),
        ] {
            check_vector_alignment(operator, name, view)?;
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
        let operator = Self::NAME;
        let heads = args.query.dim(1);
        let query_rows = args.query.dim(0);
        let key_rows = args.key.dim(0);

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
            selected: to_i64(operator, args.indices.dim(2))?,
            head_dim: to_i64(operator, head_dim)?,
            dtype: abi::code(descriptor.dtype)?,
            output_dtype: abi::code(descriptor.output)?,
            scale: args.scale,
            error: [0; MESSAGE],
        };
        // SAFETY: `SparseCall` is the call structure this package's entry declares: the declaration
        // is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        SparseArgs {
            out: twins.shadow_of(args.out),
            lse: args.lse.map(|lse| twins.shadow_of(lse)),
            ..args
        }
    }

    /// The standard bound written against the **declared key bound**, which is conservative.
    ///
    /// A row sums `Σ_j w_j · v_j` over the rows of the tiles its selection keeps, plus the dense
    /// prefix tiles: at most `keys` products, each an FP32 dot of `head_dim` against a value, so
    /// `γₙ` with `n = keys` is the reduction's factor. The true reduction is smaller — `Σ over
    /// selected tiles of their valid rows` — and the excess is bounded by the ratio `keys / Σ rows`,
    /// which the descriptor cannot know because the selection is the layout's. Conservatism is the
    /// right side to err on: a bound tighter than the arithmetic would fail a correct kernel.
    ///
    /// `keys` is a declared bound outside the key (the VSA's per-layout tile counts), so one program
    /// serves every segment that fits it and the bound is written for the largest.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.keys);
        let accumulate = DType::F32;
        let gamma = tolerance::gamma(reduction, accumulate);
        let value_term = 2.0 * gamma + tolerance::unit_roundoff(accumulate);
        Tolerance::rounding(reduction, accumulate, descriptor.output)
            .with_atol((value_term * f64::from(tolerance::VALUE_ENVELOPE)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SM100 64-row build of 128-wide heads, BF16 out, with the normalizer.
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
            "block_64_dtype_bf16_output_bf16_head_dim_128_lse_true"
        );
    }

    #[test]
    fn key_separates_the_block_sizes_and_the_normalizer() {
        let mut block128 = block64();
        block128.block = 128;
        block128.lse = false;
        assert_ne!(block64().to_string(), block128.to_string());
        assert!(block128.to_string().starts_with("block_128_dtype_bf16"));
        assert!(block128.to_string().ends_with("_lse_false"));
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
        // u_out of BF16 alone is 2^-8; the 65536-key accumulation adds 2*gamma on top.
        assert!(far.rtol > 0.003 && far.rtol < 0.05, "rtol = {}", far.rtol);
        assert!(far.atol > 0.0 && far.atol < 0.1, "atol = {}", far.atol);
        assert!(
            far.allows(1.0, 1.002),
            "a 2e-3 deviation at |x| = 1 is inside BF16's rounding"
        );
        assert!(!far.allows(1.0, 1.5));
    }
}
