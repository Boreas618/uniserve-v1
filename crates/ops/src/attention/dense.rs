//! Dense attention over packed sequences: no cache, no page table.
//!
//! The attention of everything H3 does without a prefix cache: the language encoder over packed
//! text and vision rows, the video transformer's cross-attention onto the encoded conditions, and
//! the audio encoder, which runs in FP32 (specs/worker.md 9). Each batch's rows are a contiguous run
//! of the packed tensors, named by the offsets the scheduler writes:
//!
//! ```text
//! query       [q_tokens, q_heads, head_dim]
//! key         [kv_tokens, kv_heads, head_dim]
//! value       [kv_tokens, kv_heads, head_dim]
//! out         [q_tokens, q_heads, head_dim]
//! cu_seq_lens_q [batch + 1] i32        row offsets of the queries of each sequence
//! cu_seq_lens_k [batch + 1] i32        row offsets of the keys of each sequence
//! ```
//!
//! For query row `i` of sequence `b`, head `h`, with `s_j = scale · q_i,h · k_{j,h}` over the keys
//! `j` of `b` that are visible to it:
//!
//! ```text
//! p_j     = exp(s_j − max_j s_j)
//! out_i,h = Σ_j p_j v_{j,h} / Σ_j p_j
//! ```
//!
//! accumulated in FP32 and rounded once to the output dtype. Visibility is the causal mask — key
//! `j` is visible to the query at row `i` of the sequence when `j − start_k ≤ i − start_q` — plus,
//! when a history window is launched, `j − start_k ≥ i − start_q − window_left`, or the dense mask
//! when the descriptor declares it.
//!
//! # The bound between two implementations
//!
//! As for [`crate::attention::PagedAttention`]: the output is a convex combination of the values it
//! reads, so the standard bound (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//! §3.1) gives `|out − out_exact| ≤ (γₙ + u) V + u_out |out|` for `n` visible keys, and two
//! implementations differ by twice that. The `V` term enters as an absolute term over the declared
//! envelope [`VALUE_ENVELOPE`]; the reduction length is the descriptor's
//! [`DenseDescriptor::max_kv_len`], the longest sequence a module serves, which bounds every launch.
//!
//! # The package
//!
//! A native library per backend. `fa4` is FlashAttention-4's CuTe package: tiles, the split-K and
//! scheduler heuristics, the compiled specializations and their choice, and the output allocation
//! are its host work (specs/worker.md 9, `flash_attn/cute/interface.py:276-522, 715-730`), so
//! [`Op::launch`] computes no grid. `f32` is the FP32 path — cuBLASLt SGEMM and a softmax kernel
//! inside the package — for the audio encoder and the sparse compression branch, which hold their
//! activations in FP32. Both take the same call structure and the same descriptor fields.

use std::ffi::c_void;
use std::fmt;

use super::{Mask, address, head_stride, row_stride, to_i64};
use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The largest value magnitude the attention bound's absolute term is written for.
///
/// Dense attention reads the value projection of a normalized residual stream; an element stays
/// below 8 in the models served here. The envelope is a declaration about that data, not something
/// the descriptor can state.
pub const VALUE_ENVELOPE: f32 = 8.0;

/// One compilation of `dense_attention`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DenseDescriptor {
    /// Element type of the queries.
    pub query: DType,
    /// Element type of the keys and values.
    pub key_value: DType,
    /// Element type of the output.
    pub output: DType,
    /// Elements per head.
    pub head_dim: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads: `q_heads / kv_heads` queries share one, as grouped-query attention has it.
    pub kv_heads: u32,
    /// The mask the module applies.
    pub mask: Mask,
    /// Whether a history window is applied with the causal mask.
    pub windowed: bool,
    /// The longest sequence (keys) the module serves.
    pub max_kv_len: u32,
}

impl fmt::Display for DenseDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.query)
            .field("kv", self.key_value)
            .field("out", self.output)
            .field("head_dim", self.head_dim)
            .field("q_heads", self.q_heads)
            .field("kv_heads", self.kv_heads)
            .field("mask", self.mask)
            .field("window", self.windowed)
            .field("len", self.max_kv_len)
            .finish(formatter)
    }
}

/// The arguments of one dense attention call.
pub struct DenseArgs {
    /// `[q_tokens, q_heads, head_dim]` queries, packed by sequence.
    pub query: TensorView,
    /// `[kv_tokens, kv_heads, head_dim]` keys, packed by sequence.
    pub key: TensorView,
    /// `[kv_tokens, kv_heads, head_dim]` values.
    pub value: TensorView,
    /// `[batch + 1]` i32: the first query row of each sequence, then `q_tokens`.
    pub cu_seq_lens_q: TensorView,
    /// `[batch + 1]` i32: the first key row of each sequence, then `kv_tokens`.
    pub cu_seq_lens_k: TensorView,
    /// `[q_tokens, q_heads, head_dim]` output.
    pub out: TensorView,
    /// Softmax scale applied to `q · k` before the exponential.
    pub scale: f32,
    /// Inclusive history window in keys; `-1` for none.
    pub window_left: i32,
    /// Whether the module applies the causal mask the descriptor names.
    pub causal: bool,
}

/// The call structure a package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/dense_attention/<backend>/include/uniserve_dense_attention.h`;
/// the layout is pinned by [`DENSE_ATTENTION_CALL_LAYOUT`], the assertions below and the test beside
/// them.
#[repr(C)]
#[derive(Debug)]
pub struct DenseCall {
    /// `[q_tokens, q_heads, head_dim]` output.
    pub out: *mut c_void,
    /// `[q_tokens, q_heads, head_dim]` queries.
    pub query: *const c_void,
    /// `[kv_tokens, kv_heads, head_dim]` keys.
    pub key: *const c_void,
    /// `[kv_tokens, kv_heads, head_dim]` values.
    pub value: *const c_void,
    /// `[batch + 1]` i32 query offsets.
    pub cu_seq_lens_q: *const i32,
    /// `[batch + 1]` i32 key offsets.
    pub cu_seq_lens_k: *const i32,
    /// Sequences in the call.
    pub batch: i64,
    /// Packed query rows.
    pub q_tokens: i64,
    /// Packed key rows.
    pub kv_tokens: i64,
    /// Rows of the longest query sequence: the module pads to at least this.
    pub max_q_len: i64,
    /// Rows of the longest key sequence, which the descriptor bounds.
    pub max_kv_len: i64,
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
    /// Elements between consecutive key rows.
    pub kv_stride_tokens: i64,
    /// Elements between consecutive key heads.
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
    /// `DType`'s ABI number of the queries.
    pub dtype_query: i64,
    /// `DType`'s ABI number of the keys and values.
    pub dtype_key_value: i64,
    /// `DType`'s ABI number of the output.
    pub dtype_output: i64,
    /// The entry's message on a nonzero return.
    pub error: [u8; 512],
}

/// The size and field offsets the package's header must mirror, in bytes.
pub const DENSE_ATTENTION_CALL_LAYOUT: DenseCallLayout = DenseCallLayout {
    size: 720,
    out: 0,
    query: 8,
    key: 16,
    value: 24,
    cu_seq_lens_q: 32,
    cu_seq_lens_k: 40,
    batch: 48,
    q_tokens: 56,
    kv_tokens: 64,
    max_q_len: 72,
    max_kv_len: 80,
    q_heads: 88,
    kv_heads: 96,
    head_dim: 104,
    q_stride_tokens: 112,
    q_stride_heads: 120,
    kv_stride_tokens: 128,
    kv_stride_heads: 136,
    o_stride_tokens: 144,
    o_stride_heads: 152,
    scale: 160,
    window_left: 168,
    causal: 176,
    dtype_query: 184,
    dtype_key_value: 192,
    dtype_output: 200,
    error: 208,
};

/// The layout of [`DenseCall`], for the header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct DenseCallLayout {
    pub size: usize,
    pub out: usize,
    pub query: usize,
    pub key: usize,
    pub value: usize,
    pub cu_seq_lens_q: usize,
    pub cu_seq_lens_k: usize,
    pub batch: usize,
    pub q_tokens: usize,
    pub kv_tokens: usize,
    pub max_q_len: usize,
    pub max_kv_len: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_stride_tokens: usize,
    pub q_stride_heads: usize,
    pub kv_stride_tokens: usize,
    pub kv_stride_heads: usize,
    pub o_stride_tokens: usize,
    pub o_stride_heads: usize,
    pub scale: usize,
    pub window_left: usize,
    pub causal: usize,
    pub dtype_query: usize,
    pub dtype_key_value: usize,
    pub dtype_output: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<DenseCall>() == DENSE_ATTENTION_CALL_LAYOUT.size,
    "the dense_attention call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(DenseCall, cu_seq_lens_q) == DENSE_ATTENTION_CALL_LAYOUT.cu_seq_lens_q,
    "the dense_attention call structure's pointers moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(DenseCall, batch) == DENSE_ATTENTION_CALL_LAYOUT.batch,
    "the dense_attention call structure's geometry moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(DenseCall, q_stride_tokens) == DENSE_ATTENTION_CALL_LAYOUT.q_stride_tokens,
    "the dense_attention call structure's strides moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(DenseCall, scale) == DENSE_ATTENTION_CALL_LAYOUT.scale,
    "the dense_attention call structure's scalars moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(DenseCall, dtype_query) == DENSE_ATTENTION_CALL_LAYOUT.dtype_query,
    "the dense_attention call structure's dtypes moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(DenseCall, error) == DENSE_ATTENTION_CALL_LAYOUT.error,
    "the dense_attention call structure's `error` moved without its header"
);

impl NativeCall for DenseCall {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// Dense attention over packed sequences.
pub struct DenseAttention;

impl Op for DenseAttention {
    const NAME: &'static str = "dense_attention";

    type Descriptor = DenseDescriptor;
    type Args<'a> = DenseArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let operator = Self::NAME;
        let head_dim = u64::from(descriptor.head_dim);
        let q_heads = u64::from(descriptor.q_heads);
        let kv_heads = u64::from(descriptor.kv_heads);

        args.query.require_rank(operator, 3)?;
        args.query
            .require_row(operator, descriptor.query, head_dim)?;
        args.key.require_rank(operator, 3)?;
        args.key
            .require_row(operator, descriptor.key_value, head_dim)?;
        args.value.require_rank(operator, 3)?;
        args.value
            .require_row(operator, descriptor.key_value, head_dim)?;
        args.out.require_rank(operator, 3)?;
        args.out
            .require_row(operator, descriptor.output, head_dim)?;

        let q_tokens = args.query.shape.dim(0);
        let kv_tokens = args.key.shape.dim(0);
        if q_tokens == 0 || kv_tokens == 0 {
            return Err(Error::argument(
                operator,
                "a call attends from at least one query row to at least one key row",
            ));
        }
        if args.query.shape.dim(1) != q_heads || args.key.shape.dim(1) != kv_heads {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {q_heads} query heads and {kv_heads} key heads, found {} and {}",
                    args.query.shape.dim(1),
                    args.key.shape.dim(1)
                ),
            ));
        }
        if kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(Error::argument(
                operator,
                format!("{q_heads} query heads do not divide over {kv_heads} key heads"),
            ));
        }
        if !args.key.shape.matches(&args.value.shape) {
            return Err(Error::argument(
                operator,
                format!(
                    "keys and values take one shape, found {:?} and {:?}",
                    args.key.shape.dims(),
                    args.value.shape.dims()
                ),
            ));
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
        // The scheduler writes the offsets; the module pads its schedule to the longest row, which
        // the descriptor bounds.
        let batch = args
            .cu_seq_lens_q
            .shape
            .dims()
            .first()
            .copied()
            .unwrap_or(0)
            .saturating_sub(1);
        args.cu_seq_lens_q
            .require_shape(operator, Shape::new(&[batch + 1]))?;
        args.cu_seq_lens_k
            .require_shape(operator, Shape::new(&[batch + 1]))?;
        for (name, view) in [
            ("cu_seq_lens_q", &args.cu_seq_lens_q),
            ("cu_seq_lens_k", &args.cu_seq_lens_k),
        ] {
            if view.dtype != DType::I32 {
                return Err(Error::argument(
                    operator,
                    format!("expected i32 {name}, found {}", view.dtype),
                ));
            }
        }
        if batch == 0 {
            return Err(Error::argument(
                operator,
                "expected at least one sequence: cu_seq_lens_q holds batch + 1 offsets",
            ));
        }
        if kv_tokens > u64::from(descriptor.max_kv_len) {
            return Err(Error::argument(
                operator,
                format!(
                    "the module serves sequences of {} keys, the call packs {kv_tokens}",
                    descriptor.max_kv_len
                ),
            ));
        }
        let causal = match (descriptor.mask, args.causal) {
            (Mask::Causal, true) | (Mask::Dense, false) => args.causal,
            _ => {
                return Err(Error::argument(
                    operator,
                    format!(
                        "the module's {:?} mask does not match a launch with causal = {}",
                        descriptor.mask, args.causal
                    ),
                ));
            }
        };
        if args.window_left >= 0 && !causal {
            return Err(Error::argument(
                operator,
                "a history window is expressible only with the causal mask",
            ));
        }
        if !args.scale.is_finite() || args.scale == 0.0 {
            return Err(Error::argument(
                operator,
                format!("expected a finite nonzero scale, found {}", args.scale),
            ));
        }

        let mut call = DenseCall {
            out: address(args.out.ptr) as usize as *mut c_void,
            query: address(args.query.ptr) as usize as *const c_void,
            key: address(args.key.ptr) as usize as *const c_void,
            value: address(args.value.ptr) as usize as *const c_void,
            cu_seq_lens_q: address(args.cu_seq_lens_q.ptr) as usize as *const i32,
            cu_seq_lens_k: address(args.cu_seq_lens_k.ptr) as usize as *const i32,
            batch: to_i64(operator, batch)?,
            q_tokens: to_i64(operator, q_tokens)?,
            kv_tokens: to_i64(operator, kv_tokens)?,
            // The longest row is only known from the offsets on the device; the packed token count
            // bounds it, and the module's schedule only pads.
            max_q_len: to_i64(operator, q_tokens)?,
            max_kv_len: to_i64(operator, u64::from(descriptor.max_kv_len))?,
            q_heads: to_i64(operator, q_heads)?,
            kv_heads: to_i64(operator, kv_heads)?,
            head_dim: to_i64(operator, head_dim)?,
            q_stride_tokens: to_i64(operator, row_stride(&args.query))?,
            q_stride_heads: to_i64(operator, head_stride(&args.query))?,
            kv_stride_tokens: to_i64(operator, row_stride(&args.key))?,
            kv_stride_heads: to_i64(operator, head_stride(&args.key))?,
            o_stride_tokens: to_i64(operator, row_stride(&args.out))?,
            o_stride_heads: to_i64(operator, head_stride(&args.out))?,
            scale: f64::from(args.scale),
            window_left: i64::from(args.window_left),
            causal: i64::from(causal),
            dtype_query: descriptor.query.as_i64(),
            dtype_key_value: descriptor.key_value.as_i64(),
            dtype_output: descriptor.output.as_i64(),
            error: [0; 512],
        };
        // SAFETY: `DenseCall` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        DenseArgs {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The FP32 audio encoder's attention: one head group, dense mask, 4096 rows.
    fn fp32() -> DenseDescriptor {
        DenseDescriptor {
            query: DType::F32,
            key_value: DType::F32,
            output: DType::F32,
            head_dim: 64,
            q_heads: 8,
            kv_heads: 8,
            mask: Mask::Dense,
            windowed: false,
            max_kv_len: 4096,
        }
    }

    /// The video transformer's cross-attention onto the encoded conditions.
    fn cross() -> DenseDescriptor {
        DenseDescriptor {
            query: DType::BF16,
            key_value: DType::BF16,
            output: DType::BF16,
            head_dim: 128,
            q_heads: 24,
            kv_heads: 8,
            mask: Mask::Dense,
            windowed: false,
            max_kv_len: 8192,
        }
    }

    #[test]
    fn key_names_every_compiled_value() {
        assert_eq!(
            cross().to_string(),
            "dtype_bf16_kv_bf16_out_bf16_head_dim_128_q_heads_24_kv_heads_8_mask_dense_window_false_len_8192"
        );
        assert_eq!(
            fp32().to_string(),
            "dtype_f32_kv_f32_out_f32_head_dim_64_q_heads_8_kv_heads_8_mask_dense_window_false_len_4096"
        );
    }

    #[test]
    fn the_fp32_path_is_another_module_of_the_same_operator() {
        let mut causal = cross();
        causal.mask = Mask::Causal;
        assert_ne!(cross().to_string(), causal.to_string());
        assert_eq!(DenseAttention::NAME, "dense_attention");
    }

    #[test]
    fn call_layout_is_the_one_the_header_declares() {
        let layout = DENSE_ATTENTION_CALL_LAYOUT;
        assert_eq!(std::mem::size_of::<DenseCall>(), layout.size);
        assert_eq!(std::mem::align_of::<DenseCall>(), 8);
        assert_eq!(std::mem::offset_of!(DenseCall, out), layout.out);
        assert_eq!(std::mem::offset_of!(DenseCall, query), layout.query);
        assert_eq!(std::mem::offset_of!(DenseCall, key), layout.key);
        assert_eq!(std::mem::offset_of!(DenseCall, value), layout.value);
        assert_eq!(
            std::mem::offset_of!(DenseCall, cu_seq_lens_q),
            layout.cu_seq_lens_q
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, cu_seq_lens_k),
            layout.cu_seq_lens_k
        );
        assert_eq!(std::mem::offset_of!(DenseCall, batch), layout.batch);
        assert_eq!(std::mem::offset_of!(DenseCall, q_tokens), layout.q_tokens);
        assert_eq!(std::mem::offset_of!(DenseCall, kv_tokens), layout.kv_tokens);
        assert_eq!(std::mem::offset_of!(DenseCall, max_q_len), layout.max_q_len);
        assert_eq!(
            std::mem::offset_of!(DenseCall, max_kv_len),
            layout.max_kv_len
        );
        assert_eq!(std::mem::offset_of!(DenseCall, q_heads), layout.q_heads);
        assert_eq!(std::mem::offset_of!(DenseCall, kv_heads), layout.kv_heads);
        assert_eq!(std::mem::offset_of!(DenseCall, head_dim), layout.head_dim);
        assert_eq!(
            std::mem::offset_of!(DenseCall, q_stride_tokens),
            layout.q_stride_tokens
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, q_stride_heads),
            layout.q_stride_heads
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, kv_stride_tokens),
            layout.kv_stride_tokens
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, kv_stride_heads),
            layout.kv_stride_heads
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, o_stride_tokens),
            layout.o_stride_tokens
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, o_stride_heads),
            layout.o_stride_heads
        );
        assert_eq!(std::mem::offset_of!(DenseCall, scale), layout.scale);
        assert_eq!(
            std::mem::offset_of!(DenseCall, window_left),
            layout.window_left
        );
        assert_eq!(std::mem::offset_of!(DenseCall, causal), layout.causal);
        assert_eq!(
            std::mem::offset_of!(DenseCall, dtype_query),
            layout.dtype_query
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, dtype_key_value),
            layout.dtype_key_value
        );
        assert_eq!(
            std::mem::offset_of!(DenseCall, dtype_output),
            layout.dtype_output
        );
        assert_eq!(std::mem::offset_of!(DenseCall, error), layout.error);
    }

    #[test]
    fn the_bound_follows_the_sequence_length() {
        let short = DenseAttention::tolerance(&DenseDescriptor {
            max_kv_len: 512,
            ..cross()
        });
        let long = DenseAttention::tolerance(&cross());
        assert!(
            long.rtol >= short.rtol,
            "a longer sequence cannot tighten the bound"
        );
        assert!(
            long.allows(1.0, 1.004),
            "a 4e-3 deviation at |x| = 1 is inside BF16's rounding"
        );
        assert!(!long.allows(1.0, 1.5));
        // FP32 out leaves only the accumulation terms: 2*gamma(4096) + u_f32.
        let exact = DenseAttention::tolerance(&fp32());
        assert!(
            exact.rtol > 1.0e-4 && exact.rtol < 1.0e-3,
            "rtol = {}",
            exact.rtol
        );
    }
}
