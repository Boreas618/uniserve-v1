//! Hashed n-gram memory: the table rows a token's n-grams name, and the gate that adds what they
//! recall into the streams of a residual.
//!
//! A token at position `p` and the `N − 1` before it (its context, oldest first) form one
//! n-gram of each length `2..=N`, ending at the token. Each length is hashed into `H` heads, each
//! head a range of its own of one table:
//!
//! ```text
//! t_i      = normalize[context[N − 1 − i]]                 i tokens back, its own first
//! rolling  = t_0 · μ_0;   rolling ← rolling ⊕ (t_i · μ_i)  for i = 1..N−1, 64-bit
//! id[i−1, h] = rolling mod P[i−1, h] + O[i−1, h]           the (i + 1)-gram's row in head h
//! ```
//!
//! `normalize` maps a vocabulary id onto a smaller vocabulary in which ids that read alike are one,
//! `μ` are odd multipliers bounded so no product overflows, `P` distinct primes, one per head of
//! each length, and `O` their cumulative offsets, so every head owns a disjoint range of the
//! table ([`NgramHash`]). A position before the request's first token is the vocabulary id the
//! model fills its context with, which normalizes as the reference's padding does.
//!
//! The rows recalled are projected (outside these operators) into one key per stream and one
//! value, and each stream adds the value under a gate of how well its key agrees with it
//! ([`NgramGate`]):
//!
//! ```text
//! dot_k  = Σ_c h_k[c] · q_k[c] · κ_k[c] · key_k[c] · rsqrt(mean(h_k²) + ε) · rsqrt(mean(key_k²) + ε) / √d
//! g_k    = sigmoid(copysign(√max(|dot_k|, 10⁻⁶), dot_k))
//! h'_k   = round(h_k + g_k · value)
//! ```
//!
//! with `q` and `κ` the gate's two learned weights of every stream, read only as their product,
//! and every product rounded before it is summed, as the reference's separate operations round
//! them.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{Key, row_tile};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// Tokens one program of [`NgramHash`] hashes.
const HASH_TOKENS: u32 = 64;

/// The widest stream [`NgramGate`] reduces in one tile, in elements.
pub const MAX_WIDTH: u32 = 32768;

/// One compilation of `ngram_hash`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HashDescriptor {
    /// Element type of the context's token ids.
    pub ids: DType,
    /// Tokens of a context, `N`: the longest n-gram's length.
    pub order: u32,
    /// Heads of each n-gram length, `H`.
    pub heads: u32,
    /// Tokens one program hashes.
    pub block: u32,
}

impl HashDescriptor {
    /// The compilation that hashes contexts of `order` `ids` token ids into `heads` heads of each
    /// n-gram length from 2 to `order`.
    pub fn new(ids: DType, order: u32, heads: u32) -> Self {
        Self {
            ids,
            order,
            heads,
            block: HASH_TOKENS,
        }
    }

    /// The table rows each token names: `(order − 1) · heads`.
    pub const fn columns(&self) -> u32 {
        self.order.saturating_sub(1) * self.heads
    }
}

impl fmt::Display for HashDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("ids", abi::name(self.ids))
            .field("order", self.order)
            .field("heads", self.heads)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `ngram_hash`.
#[derive(Clone)]
pub struct HashArgs {
    /// `[tokens, order]`, contiguous: each token's context, the `order − 1` vocabulary ids before
    /// it, oldest first, then its own.
    pub context: Operand,
    /// `[vocabulary]` I32: each vocabulary id's normalized id. An id at or past the end reads the
    /// last entry.
    pub normalize: Operand,
    /// `[order]` I64: the multiplier of the token `i` back, the token's own first.
    pub multipliers: Operand,
    /// `[order − 1, heads]` I64: the prime modulus of each head of each n-gram length, the
    /// bigram's first.
    pub moduli: Operand,
    /// `[order − 1, heads]` I64: the first table row of each head's range.
    pub offsets: Operand,
    /// `[tokens, (order − 1) · heads]` I64, contiguous: receives the table rows.
    pub out: Operand,
}

/// The table rows of every n-gram that ends at a token (module documentation): exact integer
/// arithmetic, one program per block of tokens.
pub struct NgramHash;

impl Op for NgramHash {
    const NAME: &'static str = proto::module::operator::NGRAM_HASH;

    type Descriptor = HashDescriptor;
    type Args<'a> = HashArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let HashDescriptor {
            ids, order, heads, ..
        } = *descriptor;
        if !matches!(ids, DType::U32 | DType::I32 | DType::I64) {
            return Err(Error::argument(
                Self::NAME,
                format!("context ids of {}", abi::name(ids)),
            ));
        }
        if order < 2 || heads == 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "contexts of {order} tokens hashed into {heads} heads: at least a bigram and \
                     one head"
                ),
            ));
        }

        args.context.require_rank(Self::NAME, 2)?;
        let tokens = args.context.dim(0);
        let (n, h) = (u64::from(order), u64::from(heads));
        let named = |name: &'static str| {
            move |error: Error| Error::argument(Self::NAME, format!("{name}: {error}"))
        };
        args.context
            .require_shape(Self::NAME, &[tokens, n])
            .map_err(named("context"))?;
        args.context
            .require_row(Self::NAME, ids, n)
            .map_err(named("context"))?;
        args.normalize.require_rank(Self::NAME, 1)?;
        args.normalize
            .require_row(Self::NAME, DType::I32, args.normalize.dim(0))
            .map_err(named("normalization"))?;
        if args.normalize.dim(0) == 0 {
            return Err(Error::argument(
                Self::NAME,
                "an empty normalization table".to_owned(),
            ));
        }
        args.multipliers
            .require_shape(Self::NAME, &[n])
            .map_err(named("multipliers"))?;
        args.multipliers
            .require_row(Self::NAME, DType::I64, n)
            .map_err(named("multipliers"))?;
        for (name, view) in [("moduli", &args.moduli), ("offsets", &args.offsets)] {
            view.require_shape(Self::NAME, &[n - 1, h])
                .map_err(named(name))?;
            view.require_row(Self::NAME, DType::I64, h)
                .map_err(named(name))?;
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} are not contiguous"),
                ));
            }
        }
        let columns = u64::from(descriptor.columns());
        args.out
            .require_shape(Self::NAME, &[tokens, columns])
            .map_err(named("rows"))?;
        args.out
            .require_row(Self::NAME, DType::I64, columns)
            .map_err(named("rows"))?;
        for (name, view) in [("context", &args.context), ("rows", &args.out)] {
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} are not contiguous"),
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
        let tokens = args.context.dim(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("context_ptr", &args.context)
            .view("normalize_ptr", &args.normalize)
            .view("multipliers_ptr", &args.multipliers)
            .view("moduli_ptr", &args.moduli)
            .view("offsets_ptr", &args.offsets)
            .view("out_ptr", &args.out)
            .u64("tokens", tokens)
            .u64("vocabulary", args.normalize.dim(0));
        kernel.launch(
            LaunchGrid::covering(tokens, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        HashArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Integer arithmetic: two implementations name the same rows.
        Tolerance::exact()
    }
}

/// One compilation of `ngram_gate`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GateDescriptor {
    /// Element type of the streams, the projected rows and the gate's weights.
    pub dtype: DType,
    /// Streams of the residual.
    pub streams: u32,
    /// Elements of one stream.
    pub hidden: u32,
    /// The lanes one program loads: `hidden` rounded up to a power of two.
    pub tile: u32,
}

impl GateDescriptor {
    /// The compilation that gates a value into `streams` streams of `hidden` `dtype` elements.
    pub fn new(dtype: DType, streams: u32, hidden: u32) -> Self {
        Self {
            dtype,
            streams,
            hidden,
            tile: row_tile(hidden),
        }
    }
}

impl fmt::Display for GateDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("streams", self.streams)
            .field("hidden", self.hidden)
            .field("tile", self.tile)
            .finish(formatter)
    }
}

/// The arguments of `ngram_gate`.
#[derive(Clone)]
pub struct GateArgs {
    /// `[tokens, streams, hidden]`, contiguous: the streams.
    pub streams: Operand,
    /// `[tokens, (streams + 1) · hidden]`, rows of unit stride: each token's key of every stream,
    /// then its value.
    pub projected: Operand,
    /// `[streams, hidden]`, contiguous: the gate's query weight.
    pub query: Operand,
    /// `[streams, hidden]`, contiguous: the gate's key weight.
    pub key: Operand,
    /// `[tokens, streams, hidden]`, contiguous: receives the gated streams.
    pub out: Operand,
    /// The epsilon added to both mean squares.
    pub eps: f32,
    /// The agreement's scale, `1/√hidden` as FP32.
    pub scale: f32,
}

/// Each stream plus the recalled value under the gate of its key's agreement with the stream
/// (module documentation): the three reductions of a stream in FP32, one program per
/// (token, stream).
pub struct NgramGate;

impl Op for NgramGate {
    const NAME: &'static str = proto::module::operator::NGRAM_GATE;

    type Descriptor = GateDescriptor;
    type Args<'a> = GateArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let GateDescriptor {
            dtype,
            streams,
            hidden,
            tile,
        } = *descriptor;
        if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
            return Err(Error::argument(
                Self::NAME,
                format!("streams of {} elements", abi::name(dtype)),
            ));
        }
        if streams == 0 || hidden == 0 || hidden > MAX_WIDTH {
            return Err(Error::argument(
                Self::NAME,
                format!("{streams} streams of {hidden} elements, 1..={MAX_WIDTH} each"),
            ));
        }
        if !tile.is_power_of_two() || tile < hidden {
            return Err(Error::argument(
                Self::NAME,
                format!("a tile of {tile} lanes does not cover a stream of {hidden} elements"),
            ));
        }

        args.streams.require_rank(Self::NAME, 3)?;
        let tokens = args.streams.dim(0);
        let (n, d) = (u64::from(streams), u64::from(hidden));
        let named = |name: &'static str| {
            move |error: Error| Error::argument(Self::NAME, format!("{name}: {error}"))
        };
        for (name, view, dims) in [
            ("streams", &args.streams, [tokens, n, d].as_slice()),
            ("gated streams", &args.out, &[tokens, n, d]),
            ("query weight", &args.query, &[n, d]),
            ("key weight", &args.key, &[n, d]),
        ] {
            view.require_shape(Self::NAME, dims).map_err(named(name))?;
            view.require_row(Self::NAME, dtype, d)
                .map_err(named(name))?;
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} are not contiguous"),
                ));
            }
        }
        args.projected
            .require_shape(Self::NAME, &[tokens, (n + 1) * d])
            .map_err(named("projected rows"))?;
        args.projected
            .require_row(Self::NAME, dtype, (n + 1) * d)
            .map_err(named("projected rows"))?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.streams.dim(0);
        // A token's projected row may sit in a wider buffer: the kernel steps by the view's
        // stride.
        let stride = args.projected.stride(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("streams_ptr", &args.streams)
            .view("projected_ptr", &args.projected)
            .view("query_ptr", &args.query)
            .view("key_ptr", &args.key)
            .view("out_ptr", &args.out)
            .u64("projected_stride", stride)
            .f32("eps", args.eps)
            .f32("scale", args.scale);
        kernel.launch(
            LaunchGrid::new([tokens, u64::from(descriptor.streams), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        GateArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The agreement is a `d`-term FP32 sum of mixed signs over unit-RMS rows: two
        // implementations differ by `2γ_d√d` of it at most. The signed square root is
        // Hölder-continuous (`|√a − √b| ≤ √|a − b|`) and the sigmoid 1/4-Lipschitz, so the gate
        // differs by `√(2γ_d√d)/4`, which the value carries into the sum with the stream before
        // the one rounding; the sum may cancel.
        let agreement = Tolerance::dot(u64::from(descriptor.hidden), DType::F32, DType::F32);
        Tolerance {
            atol: tolerance::underflow(descriptor.dtype) as f32,
            rtol: f64::from(agreement.rtol).sqrt() as f32 / 4.0
                + 2.0 * tolerance::unit_roundoff(descriptor.dtype) as f32,
            cancels: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use platform::DevicePtr;

    use super::*;

    fn view(dtype: DType, dims: &[u64]) -> Operand {
        Operand::new(DevicePtr::from_address(0x1000), dtype, dims)
    }

    #[test]
    fn the_keys_name_every_compiled_value() {
        assert_eq!(
            HashDescriptor::new(DType::U32, 4, 8).to_string(),
            "ids_u32_order_4_heads_8_block_64"
        );
        assert_eq!(
            GateDescriptor::new(DType::BF16, 4, 5120).to_string(),
            "dtype_bf16_streams_4_hidden_5120_tile_8192"
        );
    }

    #[test]
    fn a_token_names_one_row_per_head_of_every_length() {
        let descriptor = HashDescriptor::new(DType::U32, 4, 8);
        let args = HashArgs {
            context: view(DType::U32, &[5, 4]),
            normalize: view(DType::I32, &[129_280]),
            multipliers: view(DType::I64, &[4]),
            moduli: view(DType::I64, &[3, 8]),
            offsets: view(DType::I64, &[3, 8]),
            out: view(DType::I64, &[5, 24]),
        };
        NgramHash::check(&descriptor, &args).unwrap();

        let narrow = HashArgs {
            out: view(DType::I64, &[5, 16]),
            ..args
        };
        assert!(NgramHash::check(&descriptor, &narrow).is_err());
    }

    #[test]
    fn the_projected_rows_hold_a_key_per_stream_and_a_value() {
        let descriptor = GateDescriptor::new(DType::BF16, 4, 5120);
        let args = |width: u64| GateArgs {
            streams: view(DType::BF16, &[3, 4, 5120]),
            projected: view(DType::BF16, &[3, width]),
            query: view(DType::BF16, &[4, 5120]),
            key: view(DType::BF16, &[4, 5120]),
            out: view(DType::BF16, &[3, 4, 5120]),
            eps: 1e-20,
            scale: 5120_f64.powf(-0.5) as f32,
        };
        NgramGate::check(&descriptor, &args(5 * 5120)).unwrap();
        assert!(NgramGate::check(&descriptor, &args(4 * 5120)).is_err());
    }
}
