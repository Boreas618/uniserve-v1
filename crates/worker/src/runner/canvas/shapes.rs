//! Canvas graph shapes: row and length buckets, slot buckets and padding.
//!
//! A readout graph holds `rows * length` tokens in up to `2 * rows`
//! sequences of at most `length` tokens each: the live canvases, then
//! padding sequences of `length` tokens that make up the tokens the live
//! canvases leave (the last one shorter), then empty sequences. Padding
//! sequences read no prefix. Shorter canvases replay a shorter length
//! bucket, so they do not evaluate a full canvas of padding per row. A step
//! graph holds `rows` full canvases, one sequence each; a padding row starts
//! a canvas in the sentinel slot zero. Readout tails hold up to 256 slot
//! rows; a call with more slots replays them in chunks.

use super::super::graphs::bucket;
use crate::{Error, Result};

/// Canvas row buckets. After four rows, each step adds at most half the
/// previous count; the capacity itself is added too, so padding never
/// exceeds one third of a bucket's rows.
pub(crate) const ROW_BUCKETS: [usize; 14] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];

/// Slot rows of the readout tail graphs; calls are chunked at the largest.
pub(crate) const SLOT_BUCKETS: [usize; 3] = [64, 128, 256];

/// The shortest readout length bucket; buckets double up to the canvas.
const SHORTEST_LENGTH: usize = 16;

/// The bytes of transient sampler logits and weights one step chunk may
/// hold: FP32 logits and BF16 weights per vocabulary entry and position.
const STEP_CHUNK_BYTES: usize = 512 << 20;

/// What a captured canvas graph computes, by shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CanvasGraphKey {
    /// The readout pass of `rows` canvases of up to `length` tokens.
    Readout { rows: usize, length: usize },
    /// One denoising step of `rows` canvases; `first` when every row is at
    /// step zero, which skips the self-conditioning signal.
    Step { rows: usize, first: bool },
    /// The readout tail of `slots` slot rows.
    Tail { slots: usize },
}

impl CanvasGraphKey {
    /// Tokens the graph evaluates through the expert layers, which sizes the
    /// expert capacity it is captured for.
    pub(crate) fn tokens(self, canvas_length: usize) -> usize {
        match self {
            Self::Readout { rows, length } => rows * length,
            Self::Step { rows, .. } => rows * canvas_length,
            Self::Tail { slots } => slots,
        }
    }
}

/// Canvases a sampler chunk projects and steps at once: as many as keep the
/// chunk's transient logits and weights within [`STEP_CHUNK_BYTES`], at
/// least one, at most `max_rows`.
pub(crate) fn step_rows(canvas_length: usize, vocab_size: usize, max_rows: usize) -> usize {
    let per_row = canvas_length.saturating_mul(vocab_size).saturating_mul(6);
    (STEP_CHUNK_BYTES / per_row.max(1)).min(max_rows).max(1)
}

/// The graph buckets of one canvas lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanvasShapes {
    canvas_length: usize,
    /// Canvases one call may hold.
    capacity: usize,
    rows: Vec<usize>,
    lengths: Vec<usize>,
}

impl CanvasShapes {
    /// Buckets for calls of at most `capacity` canvases of `canvas_length`
    /// tokens. A generating rank also never steps more canvases than it has
    /// request slots (`slots`).
    pub(crate) fn new(canvas_length: usize, capacity: usize, slots: Option<usize>) -> Result<Self> {
        if canvas_length == 0 || capacity == 0 {
            return Err(Error::Invalid(
                "canvas graphs need a positive canvas length and capacity".into(),
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
        while length < canvas_length {
            lengths.push(length);
            length *= 2;
        }
        lengths.push(canvas_length);
        Ok(Self {
            canvas_length,
            capacity: limit,
            rows,
            lengths,
        })
    }

    /// Canvases one call may hold.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Row buckets, ascending.
    pub(crate) fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// Readout length buckets, ascending; the last is the canvas length.
    pub(crate) fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// The readout bucket of `rows` canvases whose longest has `longest`
    /// tokens.
    pub(crate) fn readout(&self, rows: usize, longest: usize) -> Result<(usize, usize)> {
        let length = bucket(&self.lengths, longest);
        match (bucket(&self.rows, rows), length) {
            (Some(rows), Some(length)) => Ok((rows, length)),
            _ => Err(Error::Invalid(format!(
                "no canvas graph holds {rows} canvases of up to {longest} tokens; capacity is {} canvases of {} tokens",
                self.capacity, self.canvas_length
            ))),
        }
    }

    /// The step bucket of `rows` canvases.
    pub(crate) fn step(&self, rows: usize) -> Result<usize> {
        bucket(&self.rows, rows).ok_or_else(|| {
            Error::Invalid(format!(
                "no canvas graph holds {rows} canvas steps; capacity is {}",
                self.capacity
            ))
        })
    }
}

/// The query tokens of the padding sequences that widen a readout of
/// `live` canvases holding `tokens` tokens to a bucket of `rows` canvases
/// of `length` tokens: whole `length` sequences, then the remainder, then
/// empty sequences up to `2 * rows` sequences in all.
pub(crate) fn readout_padding(
    live: usize,
    tokens: usize,
    rows: usize,
    length: usize,
) -> Result<Vec<usize>> {
    let total = rows * length;
    if live > rows || tokens > total {
        return Err(Error::Invariant(format!(
            "a readout of {live} canvases and {tokens} tokens exceeds its bucket of {rows} canvases of {length} tokens"
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

/// Slot chunks of a readout tail: `(first slot, live slots, bucket)`.
pub(crate) fn tail_chunks(slots: usize) -> impl Iterator<Item = (usize, usize, usize)> {
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
    fn buckets_follow_the_capacity_and_the_canvas_length() -> Result<()> {
        let shapes = CanvasShapes::new(256, 10, None)?;
        assert_eq!(shapes.rows(), &[1, 2, 3, 4, 6, 8, 10]);
        assert_eq!(shapes.lengths(), &[16, 32, 64, 128, 256]);
        assert_eq!(shapes.readout(5, 40)?, (6, 64));
        assert_eq!(shapes.readout(10, 256)?, (10, 256));
        assert!(shapes.readout(11, 16).is_err());
        assert!(shapes.readout(1, 257).is_err());
        assert_eq!(shapes.step(7)?, 8);

        // Generating ranks never step more canvases than their slots.
        let shapes = CanvasShapes::new(48, 200, Some(3))?;
        assert_eq!(shapes.rows(), &[1, 2, 3]);
        assert_eq!(shapes.lengths(), &[16, 32, 48]);
        assert_eq!(shapes.capacity(), 3);
        assert!(CanvasShapes::new(48, 0, None).is_err());
        Ok(())
    }

    #[test]
    fn readout_padding_fills_the_bucket_with_bounded_sequences() -> Result<()> {
        // Two canvases of 16 and 9 tokens in a bucket of 3 x 16 tokens:
        // one more full sequence, the 7-token remainder, then empty ones.
        assert_eq!(readout_padding(2, 25, 3, 16)?, vec![16, 7, 0, 0]);
        // A full bucket pads with empty sequences only.
        assert_eq!(readout_padding(2, 32, 2, 16)?, vec![0, 0]);
        // Padding never exceeds the sequence count the buffers hold.
        let padding = readout_padding(1, 1, 4, 16)?;
        assert_eq!(padding.len(), 7);
        assert_eq!(padding.iter().sum::<usize>() + 1, 64);
        assert!(padding.iter().all(|&tokens| tokens <= 16));
        assert!(readout_padding(3, 10, 2, 16).is_err());
        Ok(())
    }

    #[test]
    fn tails_chunk_slots_at_the_largest_bucket() {
        let chunks: Vec<_> = tail_chunks(600).collect();
        assert_eq!(chunks, vec![(0, 256, 256), (256, 256, 256), (512, 88, 128)]);
        assert_eq!(tail_chunks(3).collect::<Vec<_>>(), vec![(0, 3, 64)]);
        assert_eq!(tail_chunks(0).count(), 0);
    }

    #[test]
    fn step_chunks_bound_their_transient_logits() {
        // DiffusionGemma: one canvas of 256 x 262144 entries per chunk.
        assert_eq!(step_rows(256, 262_144, 64), 1);
        assert_eq!(step_rows(16, 64, 4), 4);
        assert_eq!(step_rows(16, 64, 0), 1);
        assert_eq!(
            CanvasGraphKey::Readout {
                rows: 3,
                length: 32
            }
            .tokens(256),
            96
        );
        assert_eq!(
            CanvasGraphKey::Step {
                rows: 3,
                first: true
            }
            .tokens(256),
            768
        );
        assert_eq!(CanvasGraphKey::Tail { slots: 64 }.tokens(256), 64);
    }
}
