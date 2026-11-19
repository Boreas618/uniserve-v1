//! The compression of a request's tokens into latents of one entry per `ratio` tokens.
//!
//! A compressing attention layer keeps one latent per group of `ratio` consecutive tokens beside
//! its window. Each token projects its row into a value `kv` and a gate `score` of `width`
//! channels (one `[tokens, 2 · width]` FP32 projection, the values first), and a group's latent is
//! the per-channel softmax-weighted sum of its tokens' values, normalized:
//!
//! ```text
//! g          = p / ratio                         the group of the token at position p
//! pooled[c]  = Σ_m softmax_m(score[m, c]) · kv[m, c]      m over the group's ratio tokens, FP32
//! latent     = rms_norm(bf16(pooled), gain, eps)          statistics in FP32, rounded to BF16
//! ```
//!
//! computed by the token that completes its group (`(p + 1) mod ratio = 0`), which is where the
//! reference's `Compressor` pools (`inference/model.py`). A group's earlier tokens are the
//! completing token's predecessors in the batch when the batch holds them, and otherwise rows of
//! the request's **ring**: `[slots, ring, 2 · width]` FP32, the projection of position `q` held in
//! row `q mod ring` of the request's slot. This operator only reads the ring; the rows a later
//! step reads are placed into it after the forward has read them (the `ring_rows` it answers),
//! so no launch reads a row another program of it writes. A ring of `ratio + drafts` rows keeps
//! every row a verified prefix of drafts may resume from.
//!
//! A state row's tokens are consecutive positions of one request, the last one `seq_lens − 1`:
//! a prefill's chunk, or the tokens of a verification.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// One compilation of `latent_compress`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LatentCompressDescriptor {
    /// Tokens of one group; at least two (a ratio of one is a normalization alone).
    pub ratio: u32,
    /// Channels of a latent.
    pub width: u32,
    /// Rows of a request's ring.
    pub ring: u32,
    /// Element type of the normalization's gain.
    pub gain: DType,
}

impl fmt::Display for LatentCompressDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("ratio", self.ratio)
            .field("width", self.width)
            .field("ring", self.ring)
            .field("gain", abi::name(self.gain))
            .finish(formatter)
    }
}

/// The arguments of `latent_compress`.
#[derive(Clone)]
pub struct LatentCompressArgs {
    /// `[tokens, 2 · width]` F32, unit stride along a row: each token's values, then its gates.
    pub projected: Operand,
    /// `[width]`: the normalization's gain.
    pub weight: Operand,
    /// `[slots, ring, 2 · width]` F32, contiguous: each request's ring, read only.
    pub ring: Operand,
    /// `[rows]` U32: each state row's request slot; `NO_SLOT` for a padding row.
    pub slots: Operand,
    /// `[rows + 1]` U32: the packed offsets of each state row's tokens.
    pub query_offsets: Operand,
    /// `[rows]` U32: the tokens each state row has seen, its new ones included.
    pub seq_lens: Operand,
    /// `[tokens]` I32: each token's position.
    pub positions: Operand,
    /// `[tokens, width]` BF16, contiguous: receives the latent of the group each token
    /// completes; the row of a token that completes none is not written.
    pub latent: Operand,
    /// `[tokens]` I32: receives the entry each token completes, `p / ratio`, or `-1`.
    pub entries: Operand,
    /// `[tokens]` I32: receives the ring row (`slot · ring + p mod ring`) each token's projection
    /// is kept in for a later step, or `-1`: the state row's last `ring` tokens are kept.
    pub ring_rows: Operand,
    /// The normalization's epsilon.
    pub eps: f32,
}

/// Softmax-gated pooling of `ratio` tokens into one normalized latent; see the module
/// documentation.
pub struct LatentCompress;

impl Op for LatentCompress {
    const NAME: &'static str = proto::module::operator::LATENT_COMPRESS;

    type Descriptor = LatentCompressDescriptor;
    type Args<'a> = LatentCompressArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        if d.ratio < 2
            || d.width == 0
            || d.ring < d.ratio
            || !matches!(d.gain, DType::BF16 | DType::F32)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "groups of {} tokens of {} channels over a ring of {} with a {} gain: a ratio \
                     of at least two, a ring that holds a group",
                    d.ratio,
                    d.width,
                    d.ring,
                    abi::name(d.gain)
                ),
            ));
        }
        let width = u64::from(d.width);

        args.projected.require_rank(Self::NAME, 2)?;
        let tokens = args.projected.dim(0);
        args.projected
            .require_shape(Self::NAME, &[tokens, 2 * width])
            .and_then(|()| {
                args.projected
                    .require_row(Self::NAME, DType::F32, 2 * width)
            })
            .map_err(|error| Error::argument(Self::NAME, format!("the projection: {error}")))?;
        if args.projected.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                "the projection is not contiguous along a row".to_owned(),
            ));
        }
        args.weight
            .require_shape(Self::NAME, &[width])
            .and_then(|()| args.weight.require_row(Self::NAME, d.gain, width))
            .map_err(|error| Error::argument(Self::NAME, format!("the gain: {error}")))?;

        args.ring.require_rank(Self::NAME, 3)?;
        args.ring
            .require_shape(
                Self::NAME,
                &[args.ring.dim(0), u64::from(d.ring), 2 * width],
            )
            .and_then(|()| args.ring.require_row(Self::NAME, DType::F32, 2 * width))
            .map_err(|error| Error::argument(Self::NAME, format!("the ring: {error}")))?;
        if !args.ring.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the ring is not contiguous".to_owned(),
            ));
        }

        let rows = args.slots.dim(0);
        for (name, column, dtype, extent) in [
            ("slots", &args.slots, DType::U32, rows),
            ("query offsets", &args.query_offsets, DType::U32, rows + 1),
            ("sequence lengths", &args.seq_lens, DType::U32, rows),
            ("positions", &args.positions, DType::I32, tokens),
            ("entries", &args.entries, DType::I32, tokens),
            ("ring rows", &args.ring_rows, DType::I32, tokens),
        ] {
            column
                .require_shape(Self::NAME, &[extent])
                .and_then(|()| column.require_row(Self::NAME, dtype, extent))
                .map_err(|error| Error::argument(Self::NAME, format!("the {name}: {error}")))?;
        }
        args.latent
            .require_shape(Self::NAME, &[tokens, width])
            .and_then(|()| args.latent.require_row(Self::NAME, DType::BF16, width))
            .map_err(|error| Error::argument(Self::NAME, format!("the latent: {error}")))?;
        if !args.latent.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the latent is not contiguous".to_owned(),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.projected.dim(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("projected", &args.projected)
            .view("weight", &args.weight)
            .view("ring", &args.ring)
            .view("slots", &args.slots)
            .view("query_offsets", &args.query_offsets)
            .view("seq_lens", &args.seq_lens)
            .view("positions", &args.positions)
            .view("latent", &args.latent)
            .view("entries", &args.entries)
            .view("ring_rows", &args.ring_rows)
            .u64("tokens", tokens)
            .u64("rows", args.slots.dim(0))
            .u64("projected_stride", args.projected.stride(0))
            .f32("eps", args.eps);
        kernel.launch(LaunchGrid::linear(tokens)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        LatentCompressArgs {
            latent: twins.shadow_of(args.latent),
            entries: twins.shadow_of(args.entries),
            ring_rows: twins.shadow_of(args.ring_rows),
            ..args
        }
    }

    /// The pooling is a convex combination of `ratio` FP32 values, the normalization a reduction
    /// over `width` squares; both round once into BF16 (the pooled row and the latent), so the
    /// bound is two BF16 roundings over the normalization's reduction.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let rounded = Tolerance::rounding(u64::from(descriptor.width), DType::F32, DType::BF16);
        Tolerance {
            rtol: rounded.rtol + tolerance::unit_roundoff(DType::BF16) as f32,
            ..rounded
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_states_the_group_and_the_ring() {
        let compress = LatentCompressDescriptor {
            ratio: 2,
            width: 512,
            ring: 8,
            gain: DType::BF16,
        };
        assert_eq!(compress.to_string(), "ratio_2_width_512_ring_8_gain_bf16");
    }
}
