//! Dense attention over packed sequences: no cache, no page table.
//!
//! The attention of everything that runs without a prefix cache: an encoder over packed rows, a
//! transformer's cross-attention onto encoded conditions, and attention in FP32. Each batch's rows
//! are a contiguous run of the packed tensors, named by the offsets the scheduler writes:
//!
//! ```text
//! query       [q_tokens, q_heads, head_dim]
//! key         [kv_tokens, kv_heads, head_dim]
//! value       [kv_tokens, kv_heads, head_dim]
//! out         [q_tokens, q_heads, head_dim]
//! cu_seq_lens_q [batch + 1] u32        row offsets of the queries of each sequence
//!
//! ## One sequence without offsets
//!
//! `DenseDescriptor::cu_seq_lens` says whether a launch carries offsets, and the arguments carry
//! `Some`/`None` accordingly; a launch that disagrees with its descriptor is refused in both
//! directions. Without offsets the call is one sequence: `batch` is one, and the program attends
//! `[0, q_tokens)` against `[0, kv_tokens)` without reading a table. It is a separate compilation,
//! so a program never branches on a missing port, and no launch is ever handed a null view.
//! cu_seq_lens_k [batch + 1] u32        row offsets of the keys of each sequence
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
//! envelope [`tolerance::VALUE_ENVELOPE`]; the reduction length is the descriptor's
//! [`DenseDescriptor::max_kv_len`], the longest key extent a launch may declare. It is a bound and
//! not a compiled value, so it is absent from the key: a bidirectional attention may run over a
//! `padded_tokens` bucket that varies per layout, and naming it would compile a program per bucket.
//!
//! # The package
//!
//! A native library per backend. `fa4` is FlashAttention-4's forward pass, which its package
//! compiles with the tile, query stages and scheduler its tuning data states for the descriptor;
//! the grid is the exported launcher's host work, so [`Op::launch`] computes none. `f32` is the
//! FP32 path — cuBLASLt SGEMM and a softmax kernel
//! inside the package — for the audio encoder and the sparse compression branch, which hold their
//! activations in FP32. Both take the same call structure and the same descriptor fields.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;

use super::{Mask, address, head_stride, row_stride, to_i64};
use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};
use tensor::DType;

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
    /// The longest key extent a launch may declare: a bound the caller supplies per layout, **not**
    /// a compiled value.
    ///
    /// It is deliberately absent from the key. A full bidirectional attention may run over
    /// `padded_tokens`, a bucket value that varies per layout and may reach 174,598, so naming it
    /// in the key would compile a program per bucket. The program reads the extent from the key
    /// view; this bound only refuses a launch beyond what the caller declared and gives the
    /// tolerance its reduction length.
    pub max_kv_len: u32,
    /// Whether a launch carries `[batch + 1]` sequence offsets.
    ///
    /// `true` is the scheduled batch path, and the key every existing package has. `false` is one
    /// sequence with no offsets at all: the program attends `[0, q_tokens)` against
    /// `[0, kv_tokens)` and never reads a table. The two are separate compilations, so no program
    /// branches on a missing pointer.
    pub cu_seq_lens: bool,
}

impl fmt::Display for DenseDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", abi::name(self.query))
            .field("kv", abi::name(self.key_value))
            .field("out", abi::name(self.output))
            .field("head_dim", self.head_dim)
            .field("q_heads", self.q_heads)
            .field("kv_heads", self.kv_heads)
            .field("mask", self.mask)
            .field("window", self.windowed);
        // Named only for the form without offsets, so every key the scheduled batch path compiles
        // stays byte-identical: one sequence is a new variant, not a renamed old one.
        let key = if self.cu_seq_lens {
            key
        } else {
            key.field("cu_seq_lens", false)
        };
        key.finish(formatter)
    }
}

/// The arguments of one dense attention call.
#[derive(Clone)]
pub struct DenseArgs {
    /// `[q_tokens, q_heads, head_dim]` queries, packed by sequence.
    pub query: Operand,
    /// `[kv_tokens, kv_heads, head_dim]` keys, packed by sequence.
    pub key: Operand,
    /// `[kv_tokens, kv_heads, head_dim]` values.
    pub value: Operand,
    /// `[batch + 1]` u32: the first query row of each sequence, then `q_tokens`.
    ///
    /// Both offsets are `None` exactly when the descriptor says the launch has none. One absent and
    /// one given is refused: they describe one sequence layout together.
    pub cu_seq_lens_q: Option<Operand>,
    /// `[batch + 1]` u32: the first key row of each sequence, then `kv_tokens`.
    pub cu_seq_lens_k: Option<Operand>,
    /// `[q_tokens, q_heads, head_dim]` output.
    pub out: Operand,
    /// Softmax scale applied to `q · k` before the exponential.
    pub scale: f32,
    /// Inclusive history window in keys; `-1` for none.
    pub window_left: i32,
    /// Whether the module applies the causal mask the descriptor names.
    pub causal: bool,
    /// The longest query sequence, when the caller knows it from the offsets it wrote.
    ///
    /// The module sizes its schedule by this, so a launch of many equal sequences that left it to the
    /// packed token count would ask for that many times the programs and retire the surplus empty.
    /// `None` is the bound the offsets alone give: every token in one sequence.
    pub longest: Option<u64>,
}

/// The call structure a package's entry receives, declared to the packages as
/// `uniserve_dense_attention_call` (`uniserve_dense_attention.h`, generated with its layout
/// asserted: [`abi`]).
///
/// Every pointer is a device address. `cu_seq_lens_q` and `cu_seq_lens_k` are the two offset
/// tables the scheduler writes, read on the device, so an entry never copies them to the host and a
/// captured call stays capturable.
#[repr(C)]
#[derive(Debug, abi::Layout)]
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
    pub error: [c_char; MESSAGE],
}

impl NativeCall for DenseCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// Dense attention over packed sequences.
pub struct DenseAttention;

impl Op for DenseAttention {
    const NAME: &'static str = proto::module::operator::DENSE_ATTENTION;

    type Descriptor = DenseDescriptor;
    type Args<'a> = DenseArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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

        let q_tokens = args.query.dim(0);
        let kv_tokens = args.key.dim(0);
        if q_tokens == 0 || kv_tokens == 0 {
            return Err(Error::argument(
                operator,
                "a call attends from at least one query row to at least one key row",
            ));
        }
        if args.query.dim(1) != q_heads || args.key.dim(1) != kv_heads {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {q_heads} query heads and {kv_heads} key heads, found {} and {}",
                    args.query.dim(1),
                    args.key.dim(1)
                ),
            ));
        }
        if kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(Error::argument(
                operator,
                format!("{q_heads} query heads do not divide over {kv_heads} key heads"),
            ));
        }
        if args.key.dims() != args.value.dims() {
            return Err(Error::argument(
                operator,
                format!(
                    "keys and values take one shape, found {:?} and {:?}",
                    args.key.dims(),
                    args.value.dims()
                ),
            ));
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
        // The scheduler writes the offsets; the module pads its schedule to the longest row, which
        // the descriptor bounds. Two absent tables are the one-sequence form: every query row
        // attends the whole key range, which is exactly `[0, q_tokens)` and `[0, kv_tokens)`.
        sequence_layout(
            operator,
            descriptor,
            args.cu_seq_lens_q.as_ref(),
            args.cu_seq_lens_k.as_ref(),
        )?;
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
        Ok(())
    }

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
        let q_tokens = args.query.dim(0);
        let kv_tokens = args.key.dim(0);
        // The query offsets hold `batch + 1` entries; without offsets the packing is one sequence.
        let batch = args
            .cu_seq_lens_q
            .as_ref()
            .map_or(1, |offsets| offsets.dim(0) - 1);
        // The descriptor's mask and the call's flag agree.
        let causal = args.causal;

        let mut call = DenseCall {
            out: address(args.out.ptr) as usize as *mut c_void,
            query: address(args.query.ptr) as usize as *const c_void,
            key: address(args.key.ptr) as usize as *const c_void,
            value: address(args.value.ptr) as usize as *const c_void,
            // Absent offsets are null because the no-offsets form's program has no ports for them;
            // its entry reads `batch == 1` as `[0, q_tokens)` against `[0, kv_tokens)`.
            cu_seq_lens_q: args
                .cu_seq_lens_q
                .as_ref()
                .map_or(std::ptr::null(), |q| address(q.ptr) as usize as *const i32),
            cu_seq_lens_k: args
                .cu_seq_lens_k
                .as_ref()
                .map_or(std::ptr::null(), |k| address(k.ptr) as usize as *const i32),
            batch: to_i64(operator, batch)?,
            q_tokens: to_i64(operator, q_tokens)?,
            kv_tokens: to_i64(operator, kv_tokens)?,
            // The longest row is only known from the offsets on the device unless the caller stated
            // it; the packed token count bounds it, and the module's schedule only pads.
            max_q_len: to_i64(operator, args.longest.unwrap_or(q_tokens))?,
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
            dtype_query: abi::code(descriptor.query)?,
            dtype_key_value: abi::code(descriptor.key_value)?,
            dtype_output: abi::code(descriptor.output)?,
            error: [0; MESSAGE],
        };
        // SAFETY: `DenseCall` is the call structure this package's entry declares: the declaration
        // is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        DenseArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// The standard bound of 7.3 for a softmax-weighted sum of `max_kv_len` products.
    ///
    /// Each output element is `Σ_j w_j · v_j` over the visible keys: `max_kv_len` products of the
    /// query/key dot (itself a reduction of `head_dim` terms) against the value, accumulated in FP32,
    /// so `γₙ` is the reduction's own factor and `rtol = γₙ + u` covers it. The extra `atol` is the
    /// **value term** the reduction length does not cover: the weights are normalized, so the sum is
    /// bounded by `max|v|`, and forming it costs `2γₙ + u` relative to that bound.
    ///
    /// `max_kv_len` is a declared bound that is deliberately **not** in the key (a layout's padded
    /// token bucket), so the bound is written for the widest bucket the module serves: it is never
    /// tighter than the true reduction, which is the accepted deviation for a variable extent.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.max_kv_len);
        let accumulate = DType::F32;
        let gamma = tolerance::gamma(reduction, accumulate);
        let value_term = 2.0 * gamma + tolerance::unit_roundoff(accumulate);
        Tolerance::rounding(reduction, accumulate, descriptor.output)
            .with_atol((value_term * f64::from(tolerance::VALUE_ENVELOPE)) as f32)
    }
}

/// The batch a launch attends, from its offsets and its descriptor's declared form.
///
/// Two given offsets must be `[batch + 1]` u32 tables of the same length — the *values* are the
/// scheduler's, and the two tables need not end at the same row: a cross-attention has
/// `q_tokens > kv_tokens` with no offsets and one padded sequence on each side. Both absent is one
/// sequence. A launch that disagrees with the descriptor, or gives one table and not the other, is
/// refused by name.
fn sequence_layout(
    operator: &'static str,
    descriptor: &DenseDescriptor,
    queries: Option<&Operand>,
    keys: Option<&Operand>,
) -> Result<u64> {
    let offsets = match (queries, keys) {
        (Some(q), Some(k)) => Some((q, k)),
        (None, None) => None,
        _ => {
            return Err(Error::argument(
                operator,
                "the two sequence offsets are given together or both absent; one alone defines no \
                 sequence layout"
                    .to_owned(),
            ));
        }
    };
    if offsets.is_some() != descriptor.cu_seq_lens {
        return Err(Error::argument(
            operator,
            if descriptor.cu_seq_lens {
                "the module was compiled with sequence offsets and the launch has none"
            } else {
                "the module was compiled without sequence offsets and the launch has them"
            }
            .to_owned(),
        ));
    }
    let Some((q, k)) = offsets else {
        // One sequence: the whole packing, which for a cross-attention is a query range wider than
        // the key range.
        return Ok(1);
    };
    let batch = q
        .dims()
        .first()
        .map_or(0, |&entries| entries as u64)
        .saturating_sub(1);
    q.require_shape(operator, &[batch + 1])?;
    k.require_shape(operator, &[batch + 1])?;
    for (name, view) in [("cu_seq_lens_q", q), ("cu_seq_lens_k", k)] {
        if view.dtype != DType::U32 {
            return Err(Error::argument(
                operator,
                format!("expected u32 {name}, found {}", abi::name(view.dtype)),
            ));
        }
    }
    if batch == 0 {
        return Err(Error::argument(
            operator,
            "expected at least one sequence: cu_seq_lens_q holds batch + 1 offsets",
        ));
    }
    Ok(batch)
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
            cu_seq_lens: true,
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
            cu_seq_lens: true,
        }
    }

    /// A view the validation reads without touching: an address of zero is never dereferenced here.
    fn table(extent: u64) -> Operand {
        Operand::new(platform::DevicePtr::from_address(0), DType::U32, &[extent])
    }

    #[test]
    fn a_cross_attention_with_no_offsets_accepts_more_queries_than_keys() {
        // An unmasked cross-attention's whole visibility rule: every query sees keys `[0, used)`.
        // Eight queries against five keys, one sequence each, no offsets — the descriptor does not
        // change and no mask is added.
        let unweighted = DenseDescriptor {
            cu_seq_lens: false,
            ..cross()
        };
        assert_eq!(
            sequence_layout("dense_attention", &unweighted, None, None).unwrap(),
            1
        );
        // The batch path is unchanged: two tables of `batch + 1` offsets.
        assert_eq!(
            sequence_layout(
                "dense_attention",
                &cross(),
                Some(&table(3)),
                Some(&table(3))
            )
            .unwrap(),
            2
        );
        // A launch that disagrees with its descriptor is refused in both directions, and one table
        // without the other defines no layout.
        assert!(sequence_layout("dense_attention", &cross(), None, None).is_err());
        assert!(
            sequence_layout(
                "dense_attention",
                &unweighted,
                Some(&table(3)),
                Some(&table(3))
            )
            .is_err()
        );
        assert!(sequence_layout("dense_attention", &cross(), Some(&table(3)), None).is_err());
    }

    #[test]
    fn key_names_every_compiled_value() {
        assert_eq!(
            cross().to_string(),
            "dtype_bf16_kv_bf16_out_bf16_head_dim_128_q_heads_24_kv_heads_8_mask_dense_window_false"
        );
        assert_eq!(
            fp32().to_string(),
            "dtype_f32_kv_f32_out_f32_head_dim_64_q_heads_8_kv_heads_8_mask_dense_window_false"
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
