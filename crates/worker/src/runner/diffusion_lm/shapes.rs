//! Graph shapes of the diffusion LM runner: row, length and slot buckets
//! and their padding.
//!
//! An `attend` graph holds `rows * length` tokens in up to `2 * rows`
//! sequences of at most `length` tokens each: the live blocks, then
//! padding sequences of `length` tokens that make up the tokens the live
//! blocks leave (the last one shorter), then empty sequences. Padding
//! sequences read no prefix. Shorter blocks launch a shorter length
//! bucket, so they do not evaluate a full block of padding per row. A
//! `denoise` graph holds `rows` full blocks, one sequence each; a padding
//! row starts a block in the sentinel slot zero. A `readout` graph holds up
//! to 256 slot rows; a forward with more slots launches one per chunk.

use crate::cuda_graph::bucket;
use crate::{Error, Result};

/// Row buckets. After four rows, each step adds at most half the
/// previous count; the capacity itself is added too, so padding never
/// exceeds one third of a bucket's rows.
pub(crate) const ROW_BUCKETS: [usize; 14] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];

/// Slot rows of the `readout` graphs; forwards are chunked at the largest.
pub(crate) const SLOT_BUCKETS: [usize; 3] = [64, 128, 256];

/// The shortest readout length bucket; buckets double up to the block.
const SHORTEST_LENGTH: usize = 16;

/// The bytes of transient sampler logits and weights one step chunk may
/// hold: FP32 logits and BF16 weights per vocabulary entry and position.
const STEP_CHUNK_BYTES: usize = 512 << 20;

/// What identifies a captured graph of the runner besides its expert
/// capacity: the method and the scalar arguments that fix its shape. The
/// tensors behind every argument are the runner's own buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ShapeKey {
    /// `attend` over `rows` blocks of up to `length` tokens.
    Attend { rows: usize, length: usize },
    /// `denoise` of `rows` blocks; `first` when every row is at step zero,
    /// which skips the self-conditioning signal.
    Denoise { rows: usize, first: bool },
    /// `readout` of `slots` slot rows.
    Readout { slots: usize },
}

/// Blocks a sampler chunk projects and steps at once: as many as keep the
/// chunk's transient logits and weights within [`STEP_CHUNK_BYTES`], at
/// least one, at most `max_rows`.
pub(crate) fn chunk_rows(sequence_length: usize, vocab_size: usize, max_rows: usize) -> usize {
    let per_row = sequence_length.saturating_mul(vocab_size).saturating_mul(6);
    (STEP_CHUNK_BYTES / per_row.max(1)).min(max_rows).max(1)
}

/// The graph buckets of one runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Shapes {
    sequence_length: usize,
    /// Blocks one call may hold.
    capacity: usize,
    rows: Vec<usize>,
    lengths: Vec<usize>,
}

impl Shapes {
    /// Buckets for calls of at most `capacity` blocks of `sequence_length`
    /// tokens. A generating rank also never steps more blocks than it has
    /// request slots (`slots`).
    pub(crate) fn new(
        sequence_length: usize,
        capacity: usize,
        slots: Option<usize>,
    ) -> Result<Self> {
        if sequence_length == 0 || capacity == 0 {
            return Err(Error::Invalid(
                "diffusion LM graphs need a positive block length and capacity".into(),
            ));
        }
        let limit = slots.map_or(capacity, |slots| capacity.min(slots));
        let mut rows: Vec<usize> = ROW_BUCKETS
            .into_iter()
            .filter(|&rows| rows < limit)
            .collect();
        rows.push(limit);
        let mut lengths = Vec::new();
        let mut length = SHORTEST_LENGTH;
        while length < sequence_length {
            lengths.push(length);
            length *= 2;
        }
        lengths.push(sequence_length);
        Ok(Self {
            sequence_length,
            capacity: limit,
            rows,
            lengths,
        })
    }

    /// Blocks one call may hold.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Row buckets, ascending.
    pub(crate) fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// Readout length buckets, ascending; the last is the block length.
    pub(crate) fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// The `attend` bucket of `rows` blocks whose longest has `longest`
    /// tokens, or None when more blocks than the largest row bucket holds
    /// arrive: the engine bounds a readout's rows by its pass and token
    /// budget, so short blocks can outnumber the buckets, and such a call
    /// runs eagerly. A block longer than the model's fits no bucket and no
    /// buffer, which is an error.
    pub(crate) fn attend(&self, rows: usize, longest: usize) -> Result<Option<(usize, usize)>> {
        let length = bucket(&self.lengths, longest).ok_or_else(|| {
            Error::Invalid(format!(
                "a block of {longest} tokens exceeds the model's {} tokens",
                self.sequence_length
            ))
        })?;
        Ok(bucket(&self.rows, rows).map(|rows| (rows, length)))
    }

    /// The `denoise` bucket of `rows` blocks, or None when more blocks than
    /// the largest row bucket holds arrive, which run eagerly.
    pub(crate) fn denoise(&self, rows: usize) -> Option<usize> {
        bucket(&self.rows, rows)
    }
}

/// The query tokens of the padding sequences that widen a readout of
/// `live` blocks holding `tokens` tokens to a bucket of `rows` blocks
/// of `length` tokens: whole `length` sequences, then the remainder, then
/// empty sequences up to `2 * rows` sequences in all.
pub(crate) fn attend_padding(
    live: usize,
    tokens: usize,
    rows: usize,
    length: usize,
) -> Result<Vec<usize>> {
    let total = rows * length;
    if live > rows || tokens > total {
        return Err(Error::Invariant(format!(
            "a readout of {live} blocks and {tokens} tokens exceeds its bucket of {rows} blocks of {length} tokens"
        )));
    }
    let remaining = total - tokens;
    let mut padding = vec![length; remaining / length];
    if !remaining.is_multiple_of(length) {
        padding.push(remaining % length);
    }
    padding.resize(2 * rows - live, 0);
    Ok(padding)
}

/// Slot chunks of a readout: `(first slot, live slots, bucket)`.
pub(crate) fn slot_chunks(slots: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let largest = SLOT_BUCKETS[SLOT_BUCKETS.len() - 1];
    (0..slots).step_by(largest).map(move |start| {
        let live = (slots - start).min(largest);
        let bucket = bucket(&SLOT_BUCKETS, live).unwrap_or(largest);
        (start, live, bucket)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_follow_the_capacity_and_the_block_length() -> Result<()> {
        let shapes = Shapes::new(256, 10, None)?;
        assert_eq!(shapes.rows(), &[1, 2, 3, 4, 6, 8, 10]);
        assert_eq!(shapes.lengths(), &[16, 32, 64, 128, 256]);
        assert_eq!(shapes.attend(5, 40)?, Some((6, 64)));
        assert_eq!(shapes.attend(10, 256)?, Some((10, 256)));
        // More blocks than the largest bucket run eagerly; a block longer
        // than the model's is refused.
        assert_eq!(shapes.attend(11, 16)?, None);
        assert!(shapes.attend(1, 257).is_err());
        assert_eq!(shapes.denoise(7), Some(8));
        assert_eq!(shapes.denoise(11), None);

        // Generating ranks never step more blocks than their slots.
        let shapes = Shapes::new(48, 200, Some(3))?;
        assert_eq!(shapes.rows(), &[1, 2, 3]);
        assert_eq!(shapes.lengths(), &[16, 32, 48]);
        assert_eq!(shapes.capacity(), 3);
        assert!(Shapes::new(48, 0, None).is_err());
        Ok(())
    }

    #[test]
    fn attend_padding_fills_the_bucket_with_bounded_sequences() -> Result<()> {
        // Two blocks of 16 and 9 tokens in a bucket of 3 x 16 tokens:
        // one more full sequence, the 7-token remainder, then empty ones.
        assert_eq!(attend_padding(2, 25, 3, 16)?, vec![16, 7, 0, 0]);
        // A full bucket pads with empty sequences only.
        assert_eq!(attend_padding(2, 32, 2, 16)?, vec![0, 0]);
        // Padding never exceeds the sequence count the buffers hold.
        let padding = attend_padding(1, 1, 4, 16)?;
        assert_eq!(padding.len(), 7);
        assert_eq!(padding.iter().sum::<usize>() + 1, 64);
        assert!(padding.iter().all(|&tokens| tokens <= 16));
        assert!(attend_padding(3, 10, 2, 16).is_err());
        Ok(())
    }

    #[test]
    fn readouts_chunk_slots_at_the_largest_bucket() {
        let chunks: Vec<_> = slot_chunks(600).collect();
        assert_eq!(chunks, vec![(0, 256, 256), (256, 256, 256), (512, 88, 128)]);
        assert_eq!(slot_chunks(3).collect::<Vec<_>>(), vec![(0, 3, 64)]);
        assert_eq!(slot_chunks(0).count(), 0);
    }

    #[test]
    fn step_chunks_bound_their_transient_logits() {
        // DiffusionGemma: one block of 256 x 262144 entries per chunk.
        assert_eq!(chunk_rows(256, 262_144, 64), 1);
        assert_eq!(chunk_rows(16, 64, 4), 4);
        assert_eq!(chunk_rows(16, 64, 0), 1);
    }
}
