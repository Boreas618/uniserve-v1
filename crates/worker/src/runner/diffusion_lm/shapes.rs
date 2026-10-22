//! Graph shapes of the diffusion LM runner: row, token and slot buckets
//! and their padding.
//!
//! An `attend` graph holds `tokens` tokens in up to `2 * rows` sequences of
//! at most a block's tokens each: the live blocks, then padding sequences
//! of a block's tokens that make up the tokens the live blocks leave (the
//! last one shorter), then empty sequences. Padding sequences read no
//! prefix. A row bucket's token buckets are its rows times each readout
//! length bucket, up to the batch's token bound, so a pass of short blocks
//! pads few tokens, and every pass the engine forms (at most its rows and
//! its token budget, each block at most the model's) fits a bucket. A
//! `denoise` graph holds `rows` full blocks, one sequence each; a padding
//! row starts a block in the sentinel slot zero. A `finish` graph completes
//! the final layer at a bucket of slot rows, once per forward; a `readout`
//! graph projects up to 256 finished rows and reads their candidates, and a
//! forward with more slots launches one per chunk.

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
    /// `attend` over up to `2 * rows` sequences holding `tokens` tokens.
    Attend { rows: usize, tokens: usize },
    /// `denoise` of `rows` blocks; `first` when every row is at step zero,
    /// which skips the self-conditioning signal.
    Denoise { rows: usize, first: bool },
    /// `finish` of `slots` slot rows.
    Finish { slots: usize },
    /// `readout` of `slots` finished rows.
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
    /// Blocks one denoising step may hold.
    capacity: usize,
    /// `denoise` row buckets, ascending; the last is the capacity.
    rows: Vec<usize>,
    /// `attend` buckets: each row bucket, ascending, with its token
    /// buckets, ascending.
    attend: Vec<(usize, Vec<usize>)>,
    /// `finish` slot buckets, ascending; the last holds the slots of every
    /// forward.
    finish: Vec<usize>,
}

impl Shapes {
    /// Buckets for a runner whose denoising steps hold at most `capacity`
    /// blocks of `sequence_length` tokens and whose readouts hold at most
    /// `readout_rows` blocks and `tokens` tokens. A generating rank also
    /// never steps more blocks than it has request slots (`slots`).
    pub(crate) fn new(
        sequence_length: usize,
        capacity: usize,
        readout_rows: usize,
        tokens: usize,
        slots: Option<usize>,
    ) -> Result<Self> {
        if sequence_length == 0 || capacity == 0 || readout_rows == 0 || tokens == 0 {
            return Err(Error::Invalid(
                "diffusion LM graphs need a positive block length, capacity and token bound".into(),
            ));
        }
        let limit = slots.map_or(capacity, |slots| capacity.min(slots));
        let rows = buckets_to(&ROW_BUCKETS, limit);

        let mut lengths = Vec::new();
        let mut length = SHORTEST_LENGTH;
        while length < sequence_length {
            lengths.push(length);
            length *= 2;
        }
        lengths.push(sequence_length);

        // A row bucket holds its rows times each length bucket, within the
        // token bound; when its full blocks exceed the bound, the bound
        // itself is its largest bucket.
        let attend = buckets_to(&ROW_BUCKETS, readout_rows.min(tokens))
            .into_iter()
            .map(|rows| {
                let mut sizes: Vec<usize> = lengths
                    .iter()
                    .map(|&length| rows * length)
                    .filter(|&size| size <= tokens)
                    .collect();
                if rows * sequence_length > tokens && sizes.last() != Some(&tokens) {
                    sizes.push(tokens);
                }
                (rows, sizes)
            })
            .collect();

        // A forward's slots are some of its tokens.
        let mut finish = Vec::new();
        let mut slots = SLOT_BUCKETS[0];
        while slots < tokens {
            finish.push(slots);
            slots *= 2;
        }
        finish.push(tokens);

        Ok(Self {
            sequence_length,
            capacity: limit,
            rows,
            attend,
            finish,
        })
    }

    /// Blocks one denoising step may hold.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// `denoise` row buckets, ascending.
    pub(crate) fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// `attend` buckets `(rows, tokens)`, row-major, ascending.
    pub(crate) fn attend_buckets(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.attend
            .iter()
            .flat_map(|(rows, sizes)| sizes.iter().map(move |&tokens| (*rows, tokens)))
    }

    /// `finish` slot buckets, ascending.
    pub(crate) fn finish_buckets(&self) -> &[usize] {
        &self.finish
    }

    /// The `attend` bucket `(rows, tokens)` of `rows` blocks holding
    /// `tokens` tokens, the longest `longest`: the smallest row bucket
    /// holding the rows, then its smallest token bucket holding the tokens.
    /// A block longer than the model's, or a pass beyond the readout bounds
    /// the buckets were built for, fits no bucket and is refused.
    pub(crate) fn attend(
        &self,
        rows: usize,
        tokens: usize,
        longest: usize,
    ) -> Result<(usize, usize)> {
        if longest > self.sequence_length {
            return Err(Error::Invalid(format!(
                "a block of {longest} tokens exceeds the model's {} tokens",
                self.sequence_length
            )));
        }
        self.attend
            .iter()
            .find(|(bucket, _)| *bucket >= rows)
            .and_then(|(bucket, sizes)| {
                sizes
                    .iter()
                    .find(|&&size| size >= tokens)
                    .map(|&size| (*bucket, size))
            })
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "no captured readout bucket holds {rows} blocks of {tokens} tokens"
                ))
            })
    }

    /// The `denoise` bucket of `rows` blocks; more blocks than a step may
    /// hold are refused.
    pub(crate) fn denoise(&self, rows: usize) -> Result<usize> {
        bucket(&self.rows, rows).ok_or_else(|| {
            Error::Invalid(format!(
                "a denoising step of {rows} blocks exceeds the {} a step holds",
                self.capacity
            ))
        })
    }

    /// The `finish` bucket of `slots` slot rows.
    pub(crate) fn finish(&self, slots: usize) -> Result<usize> {
        bucket(&self.finish, slots).ok_or_else(|| {
            Error::Invalid(format!(
                "a readout of {slots} slots exceeds the {} a forward holds",
                self.finish.last().copied().unwrap_or(0)
            ))
        })
    }
}

/// The buckets of `catalog` below `limit`, then `limit` itself.
fn buckets_to(catalog: &[usize], limit: usize) -> Vec<usize> {
    let mut buckets: Vec<usize> = catalog
        .iter()
        .copied()
        .filter(|&bucket| bucket < limit)
        .collect();
    buckets.push(limit);
    buckets
}

/// The query tokens of the padding sequences that widen a readout of
/// `live` blocks holding `tokens` tokens to a bucket of `rows` sequences
/// and `total` tokens: sequences of `length` tokens, then the remainder,
/// then empty sequences up to `2 * rows` sequences in all.
pub(crate) fn attend_padding(
    live: usize,
    tokens: usize,
    rows: usize,
    total: usize,
    length: usize,
) -> Result<Vec<usize>> {
    if live > rows || tokens > total || total > rows * length {
        return Err(Error::Invariant(format!(
            "a readout of {live} blocks and {tokens} tokens exceeds its bucket of {rows} \
             blocks and {total} tokens"
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
        let shapes = Shapes::new(256, 10, 10, 2560, None)?;
        assert_eq!(shapes.rows(), &[1, 2, 3, 4, 6, 8, 10]);
        assert_eq!(shapes.attend(5, 200, 40)?, (6, 384));
        assert_eq!(shapes.attend(10, 2560, 256)?, (10, 2560));
        // A block longer than the model's is refused, and so is a pass
        // beyond the readout bounds.
        assert!(shapes.attend(1, 257, 257).is_err());
        assert!(shapes.attend(11, 16, 16).is_err());
        assert_eq!(shapes.denoise(7)?, 8);
        assert!(shapes.denoise(11).is_err());

        // Generating ranks never step more blocks than their slots.
        let shapes = Shapes::new(48, 200, 200, 9600, Some(3))?;
        assert_eq!(shapes.rows(), &[1, 2, 3]);
        assert_eq!(shapes.capacity(), 3);
        assert!(Shapes::new(48, 0, 1, 48, None).is_err());
        Ok(())
    }

    /// Every pass of at most the readout rows and the token bound, each
    /// block at most the model's, selects an `attend` bucket that holds it
    /// and the padding sequences the buffers hold.
    #[test]
    fn every_readout_within_the_bounds_selects_a_bucket() -> Result<()> {
        let (length, readout_rows, tokens) = (64, 20, 256);
        let shapes = Shapes::new(length, tokens / length, readout_rows, tokens, None)?;
        for rows in 1..=readout_rows {
            for total in rows..=tokens {
                // One block as long as the model's when the tokens allow,
                // the others sharing the rest.
                let longest = (total - (rows - 1)).min(length);
                if longest + (rows - 1) * length < total {
                    continue;
                }
                let (bucket, size) = shapes.attend(rows, total, longest)?;
                assert!(bucket >= rows && size >= total && size <= tokens);
                let padding = attend_padding(rows, total, bucket, size, length)?;
                assert_eq!(rows + padding.len(), 2 * bucket);
                assert_eq!(padding.iter().sum::<usize>() + total, size);
                assert!(padding.iter().all(|&count| count <= length));
            }
        }
        Ok(())
    }

    #[test]
    fn short_blocks_launch_few_padding_tokens() -> Result<()> {
        // A pass of twelve 16-token blocks within a 4096-token bound.
        let shapes = Shapes::new(256, 16, 128, 4096, None)?;
        assert_eq!(shapes.attend(12, 192, 16)?, (12, 192));
        // Beyond the full blocks the bound holds, the bound is the bucket.
        assert_eq!(shapes.attend(100, 3000, 256)?, (128, 4096));
        assert!(shapes.attend(129, 129, 1).is_err());
        Ok(())
    }

    #[test]
    fn attend_padding_fills_the_bucket_with_bounded_sequences() -> Result<()> {
        // Two blocks of 16 and 9 tokens in a bucket of 3 sequences and 48
        // tokens: one more full sequence, the 7-token remainder, then
        // empty ones.
        assert_eq!(attend_padding(2, 25, 3, 48, 16)?, vec![16, 7, 0, 0]);
        // A full bucket pads with empty sequences only.
        assert_eq!(attend_padding(2, 32, 2, 32, 16)?, vec![0, 0]);
        assert!(attend_padding(3, 10, 2, 32, 16).is_err());
        assert!(attend_padding(1, 10, 2, 40, 16).is_err());
        Ok(())
    }

    #[test]
    fn finish_buckets_reach_the_token_bound() -> Result<()> {
        let shapes = Shapes::new(256, 16, 128, 4096, None)?;
        assert_eq!(
            shapes.finish_buckets(),
            &[64, 128, 256, 512, 1024, 2048, 4096]
        );
        assert_eq!(shapes.finish(3)?, 64);
        assert_eq!(shapes.finish(4096)?, 4096);
        assert!(shapes.finish(4097).is_err());
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
