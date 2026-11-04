//! Bucket choice: which captured size a forward is padded into.
//!
//! A captured graph addresses fixed buffers, so a forward may only be served by a graph captured
//! for a size that holds it (specs/worker.md 4.5: "each batch planned into the smallest captured
//! bucket"). This module is the arithmetic of that sentence and nothing else: a [`Shape`] asks for
//! rows and tokens of one kind of work, a [`Bucket`] is a captured size, and [`Buckets`] answers
//! which captured bucket serves a shape, and in which order the buckets are captured.
//!
//! The ordering rules come from the design:
//!
//! - the smallest captured bucket that holds a forward serves it, never a larger one, and a
//!   forward no bucket holds is an error, never an eager fallback (4.5, `specs/replaceable-processes.md`
//!   §9.2);
//! - captures run largest first, so the smaller captures reuse the graph pool's blocks
//!   (`specs/replaceable-processes.md` §9.3).
//!
//! Everything here is host arithmetic on plain integers, so it is testable without a device and
//! runs on the scheduler's critical path without allocating: [`Buckets::select`] walks a sorted
//! vector and [`Bucket::holds`] compares three pairs of integers.

use std::collections::BTreeSet;

use crate::error::{Error, Result};
use std::fmt;

/// The kind of forward a captured graph serves.
///
/// A bucket only ever holds a forward of its own variant: a decode row cannot be served by a
/// prefill graph, and a prefill that projects logits has different outputs from one that only
/// writes the KV cache.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Variant {
    /// Decode rows drawn from the slot state, sampled greedily.
    DecodeGreedy,
    /// Decode rows drawn from the slot state, sampled with the bucket's sampler graph.
    DecodeSampled,
    /// Prefill rows of host tokens writing the KV cache, keeping no hidden state.
    PrefillWrite,
    /// Prefill rows of host tokens that also project each row's last hidden state.
    PrefillOutput,
}

impl Variant {
    /// The variant's name, as a capture report prints it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DecodeGreedy => "decode_greedy",
            Self::DecodeSampled => "decode_sampled",
            Self::PrefillWrite => "prefill_write",
            Self::PrefillOutput => "prefill_output",
        }
    }

    /// Whether a bucket of this variant samples a token for each of its sampling rows (4.4).
    ///
    /// A prefill bucket samples only on the batch that ends a prompt, which the mechanism says by
    /// putting rows in `logits_rows`; the bucket still owns its sampler region and its nodes, so a
    /// last chunk samples from the graph like a decode step.
    pub const fn samples(self) -> bool {
        true
    }

    /// Whether a forward of this variant draws its inputs from the slot state one token per row.
    pub const fn is_decode(self) -> bool {
        matches!(self, Self::DecodeGreedy | Self::DecodeSampled)
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// One forward's size: what it asks a captured graph to hold.
///
/// `tokens` is the number of tokens the forward writes to the KV cache and `rows` the number of
/// sequences it evaluates. A decode forward evaluates one token per row, so its `tokens == rows`;
/// a prefill evaluates a chunk of host tokens, which may be more than its rows.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Shape {
    pub rows: u32,
    pub tokens: u32,
    pub variant: Variant,
}

impl Shape {
    /// A decode forward of `rows` rows.
    pub const fn decode(rows: u32, variant: Variant) -> Self {
        Self {
            rows,
            tokens: rows,
            variant,
        }
    }

    /// A prefill forward of `rows` rows holding `tokens` tokens.
    pub const fn prefill(rows: u32, tokens: u32, variant: Variant) -> Self {
        Self {
            rows,
            tokens,
            variant,
        }
    }

    /// The smallest capture size this forward can be served at: itself.
    pub const fn bucket(self) -> Bucket {
        Bucket {
            rows: self.rows,
            tokens: self.tokens,
            variant: self.variant,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}/{} rows/{} tokens",
            self.variant, self.rows, self.tokens
        )
    }
}

/// A captured size: the rows and tokens a bucket's graph holds, and what it computes.
///
/// Buckets are ordered by size, variant first, then rows, then tokens, so that the smallest
/// bucket holding a shape is the first one an ascending walk finds and the largest bucket is the
/// last. The ordering is what the capture order and the reclaim order both read.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Bucket {
    pub variant: Variant,
    pub rows: u32,
    pub tokens: u32,
}

impl Bucket {
    /// A decode bucket of `rows` rows.
    pub const fn decode(rows: u32, variant: Variant) -> Self {
        Self {
            variant,
            rows,
            tokens: rows,
        }
    }

    /// A prefill bucket of `rows` rows holding `tokens` tokens.
    pub const fn prefill(rows: u32, tokens: u32, variant: Variant) -> Self {
        Self {
            variant,
            rows,
            tokens,
        }
    }

    /// The shape a forward of this bucket has when the bucket is exactly full.
    pub const fn shape(self) -> Shape {
        Shape {
            rows: self.rows,
            tokens: self.tokens,
            variant: self.variant,
        }
    }

    /// Whether a forward of `shape` is served by this bucket: the same variant, and room for its
    /// rows and tokens.
    pub const fn holds(self, shape: &Shape) -> bool {
        self.variant as u8 == shape.variant as u8
            && self.rows >= shape.rows
            && self.tokens >= shape.tokens
    }
}

impl fmt::Display for Bucket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.shape().fmt(formatter)
    }
}

/// The decode row counts captured when the build config names none: dense small batches kept exact,
/// then strides of eight, because a bucket is captured for every size in the list.
pub fn decode_catalog() -> Vec<u32> {
    (1..=32).chain((40..=128).step_by(8)).collect()
}

/// The prefill token counts captured when the build config names none: fine near one prompt's first
/// chunk, coarser as chunks grow, which is where a bucket's cost is amortised.
pub fn prefill_catalog() -> Vec<u32> {
    (4..=32)
        .step_by(4)
        .chain((48..=256).step_by(16))
        .chain((288..=512).step_by(32))
        .chain((576..=4096).step_by(64))
        .chain((4608..=16384).step_by(512))
        .collect()
}

/// The media flow shapes `(height, width)` captured when the build config names none.
pub const FLOW_CATALOG: [(u32, u32); 2] = [(1152, 2048), (2048, 1152)];

/// The buckets one runner captures, in the order it captures them.
///
/// The set is fixed when the executor is bound, from the build config's graph sizes
/// (`cuda_graph_config`): a bucket the set does not hold is a configuration error found at
/// startup, not a scheduling surprise at the first odd-sized batch. The vector is kept sorted
/// ascending and walked backwards for capture, so both hot paths are a single scan.
#[derive(Clone, Default, Debug)]
pub struct Buckets {
    /// Every bucket, ascending by size.
    ascending: Vec<Bucket>,
}

impl Buckets {
    /// The bucket set holding exactly `buckets`, each once.
    pub fn new(buckets: impl IntoIterator<Item = Bucket>) -> Self {
        let set: BTreeSet<Bucket> = buckets.into_iter().collect();
        Self {
            ascending: set.into_iter().collect(),
        }
    }

    /// The bucket set of a build config's decode row sizes and prefill row and token sizes.
    ///
    /// A prefill bucket is captured for every token size with every row size, without ever
    /// asking for more than `max_rows` rows or `max_tokens` tokens. Decode sizes are their own
    /// buckets, because a decode forward's row count is its token count.
    pub fn causal_lm(rows: &[u32], tokens: &[u32], sampled: bool) -> Self {
        let mut buckets = Vec::new();
        for &rows in rows {
            buckets.push(Bucket::decode(rows, Variant::DecodeGreedy));
            if sampled {
                buckets.push(Bucket::decode(rows, Variant::DecodeSampled));
            }
            for &count in tokens {
                // A prefill chunk never holds fewer tokens than rows: a row holds at least one.
                for variant in [Variant::PrefillWrite, Variant::PrefillOutput] {
                    buckets.push(Bucket::prefill(rows, count.max(rows), variant));
                }
            }
        }
        Self::new(buckets)
    }

    /// The bucket set of a batch plan that admits `rows` rows and `tokens` tokens, from the sizes
    /// the deployment configured and the catalog where it configured none (4.5).
    ///
    /// A bucket is captured for each size and never for a size the batch plan cannot admit, so a
    /// configured size above the bound is dropped rather than captured and never used. `sampled`
    /// also captures the sampling variants, which a deployment that samples needs and one that
    /// serves logits only does not pay for. A plan whose sizes all exceed the bound is refused here,
    /// at startup, rather than at the first batch of an unusual shape.
    pub fn for_batch(
        decode: Option<&[u32]>,
        prefill: Option<&[u32]>,
        rows: u32,
        tokens: u32,
        sampled: bool,
    ) -> Result<Self> {
        let mut decode: Vec<u32> = match decode {
            Some(sizes) => sizes.to_vec(),
            None => decode_catalog(),
        };
        let mut prefill: Vec<u32> = match prefill {
            Some(sizes) => sizes.to_vec(),
            None => prefill_catalog(),
        };
        decode.retain(|size| *size != 0 && *size <= rows);
        prefill.retain(|size| *size != 0 && *size <= tokens);
        if decode.is_empty() && prefill.is_empty() {
            return Err(Error::Invalid(format!(
                "no captured bucket fits a batch of {rows} rows and {tokens} tokens"
            )));
        }
        Ok(Self::causal_lm(&decode, &prefill, sampled))
    }

    /// The default catalog of a causal-language rank: the deployment's graph options, or the
    /// rank's own sizes where an option leaves them unset.
    ///
    /// The sizes are the ones the reference deployment captures: every decode batch size from one
    /// row to 32 and then in eights to 128, and prefill chunks of 4 tokens up to 32, 16 up to 256,
    /// 32 up to 512, 64 up to 4096 and 512 up to 16384. They are captured every warmup, so a
    /// deployment that serves shorter prompts states its own and captures less.
    ///
    /// `sampled` captures a sampled decode bucket beside every greedy one: a request that samples
    /// needs one, and the mechanism cannot serve it from a greedy graph.
    pub fn default_causal_lm(rows: Option<&[u32]>, tokens: Option<&[u32]>, sampled: bool) -> Self {
        const DECODE: [u32; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        const DECODE_STEP: [u32; 12] = [40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128];
        let default_rows: Vec<u32> = DECODE.into_iter().chain(DECODE_STEP).collect();
        let default_tokens: Vec<u32> = (4..=32)
            .step_by(4)
            .chain((48..=256).step_by(16))
            .chain((288..=512).step_by(32))
            .chain((576..=4096).step_by(64))
            .chain((4608..=16384).step_by(512))
            .collect();
        Self::causal_lm(
            rows.unwrap_or(&default_rows),
            tokens.unwrap_or(&default_tokens),
            sampled,
        )
    }

    /// Every bucket, ascending by size.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Bucket> + ExactSizeIterator {
        self.ascending.iter()
    }

    /// How many buckets the set holds.
    pub fn len(&self) -> usize {
        self.ascending.len()
    }

    /// Whether the set holds no bucket.
    pub fn is_empty(&self) -> bool {
        self.ascending.is_empty()
    }

    /// The smallest bucket that holds `shape`, if the set has one.
    pub fn select(&self, shape: &Shape) -> Option<Bucket> {
        self.ascending.iter().copied().find(|bucket| bucket.holds(shape))
    }

    /// The buckets in capture order: largest first, so smaller captures reuse the graph pool's
    /// blocks of the larger ones.
    pub fn capture_order(&self) -> impl Iterator<Item = &Bucket> {
        self.ascending.iter().rev()
    }

    /// The largest bucket of each variant, which a plan of that variant's static activations is
    /// sized for.
    pub fn largest(&self, variant: Variant) -> Option<Bucket> {
        self.ascending
            .iter()
            .rev()
            .copied()
            .find(|bucket| bucket.variant == variant)
    }

    /// Whether the set holds `bucket`.
    pub fn contains(&self, bucket: &Bucket) -> bool {
        self.ascending.binary_search(bucket).is_ok()
    }

    /// The bytes the largest bucket of each variant can need, given `bytes` as the footprint of
    /// one full bucket: what an arena must be able to map before it captures the set.
    pub fn largest_footprint(&self) -> Option<Bucket> {
        self.ascending.last().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A forward is served by the smallest bucket that holds it, and never by a bucket of another
    /// variant.
    #[test]
    fn smallest_holding_bucket_serves_a_forward() {
        let buckets = Buckets::causal_lm(&[1, 8, 32], &[128, 512], false);

        let decode = Shape::decode(5, Variant::DecodeGreedy);
        assert_eq!(
            buckets.select(&decode),
            Some(Bucket::decode(8, Variant::DecodeGreedy))
        );

        let prefill = Shape::prefill(5, 200, Variant::PrefillOutput);
        assert_eq!(
            buckets.select(&prefill),
            Some(Bucket::prefill(8, 512, Variant::PrefillOutput))
        );

        // The same rows and tokens in another variant is another graph.
        let write = Shape::prefill(5, 200, Variant::PrefillWrite);
        assert_eq!(
            buckets.select(&write),
            Some(Bucket::prefill(8, 512, Variant::PrefillWrite))
        );

        // A forward larger than every bucket of its variant has no graph; the caller reports it
        // rather than serving it eagerly.
        let too_large = Shape::decode(33, Variant::DecodeGreedy);
        assert_eq!(buckets.select(&too_large), None);

        // A sampled decode is not served by a greedy bucket.
        let sampled = Shape::decode(5, Variant::DecodeSampled);
        assert_eq!(buckets.select(&sampled), None);
    }

    /// Captures run largest first.
    #[test]
    fn capture_order_is_largest_first() {
        let buckets = Buckets::causal_lm(&[1, 8], &[64], false);
        let order: Vec<Bucket> = buckets.capture_order().copied().collect();
        assert_eq!(order.first(), buckets.largest_footprint().as_ref());
        assert_eq!(order.len(), buckets.len());
        assert!(
            order.windows(2).all(|pair| pair[0] >= pair[1]),
            "capture order is not descending: {order:?}"
        );
        // The same bucket twice is one capture.
        let repeated = Buckets::new([Bucket::decode(4, Variant::DecodeGreedy); 3]);
        assert_eq!(repeated.len(), 1);
    }

    /// A prefill bucket never holds fewer tokens than rows, and a decode bucket's tokens are its
    /// rows.
    #[test]
    fn bucket_sizes_are_consistent() {
        let buckets = Buckets::causal_lm(&[16], &[4], false);
        for bucket in buckets.iter() {
            if bucket.variant.is_decode() {
                assert_eq!(bucket.tokens, bucket.rows);
            } else {
                assert!(bucket.tokens >= bucket.rows);
            }
            assert!(bucket.holds(&bucket.shape()));
            assert!(bucket.holds(&Shape {
                rows: 1,
                tokens: 1,
                variant: bucket.variant
            }));
        }
    }

    /// The catalog is clamped to what the batch plan admits, so nothing is captured that no batch
    /// can fill, and a batch of the bound's own shape is still served.
    #[test]
    fn the_catalog_is_clamped_to_the_batch_bound() {
        let buckets = Buckets::for_batch(None, None, 8, 64, false).expect("a plan within the catalog");
        for bucket in buckets.iter() {
            assert!(bucket.rows <= 8, "{bucket} is wider than the batch bound");
            assert!(bucket.tokens <= 64, "{bucket} is longer than the batch bound");
        }
        let full = Shape {
            rows: 8,
            tokens: 64,
            variant: Variant::PrefillWrite,
        };
        assert!(buckets.select(&full).is_some(), "the bound's own shape is served");
        // Every decode row up to the bound is served by its own bucket.
        for rows in 1..=8 {
            let shape = Shape {
                rows,
                tokens: rows,
                variant: Variant::DecodeGreedy,
            };
            assert_eq!(
                buckets.select(&shape).map(|bucket| bucket.rows),
                Some(rows),
                "a decode of {rows} rows is padded into a larger bucket"
            );
        }
    }

    /// A size the deployment configured is used as it stands, and one above the bound is not
    /// captured: capturing it would reserve a graph no batch can fill.
    #[test]
    fn a_configured_size_above_the_bound_is_not_captured() {
        let buckets = Buckets::for_batch(Some(&[4, 1024]), Some(&[512]), 8, 64, false)
            .expect("the sizes below the bound fit");
        assert!(buckets.contains(&Bucket::decode(4, Variant::DecodeGreedy)));
        assert!(!buckets.contains(&Bucket::decode(1024, Variant::DecodeGreedy)));
        assert!(
            buckets
                .iter()
                .all(|bucket| bucket.variant.is_decode() || bucket.tokens <= 4),
            "a prefill size above the token bound was captured"
        );
    }

    /// Sizes that all exceed the bound are refused at startup rather than failed in service.
    #[test]
    fn a_plan_no_size_fits_is_refused() {
        assert!(Buckets::for_batch(Some(&[256]), Some(&[4096]), 8, 64, false).is_err());
        // One size within the bound is enough for the plan to be usable.
        assert!(Buckets::for_batch(Some(&[8, 256]), Some(&[4096]), 8, 64, false).is_ok());
    }

    /// The sampling variants are captured only for a deployment that samples.
    #[test]
    fn sampling_variants_follow_the_deployment() {
        let rows = [4_u32];
        let greedy = Buckets::for_batch(Some(&rows), None, 8, 64, false).expect("a plan");
        let sampled = Buckets::for_batch(Some(&rows), None, 8, 64, true).expect("a plan");
        assert!(!greedy
            .iter()
            .any(|bucket| bucket.variant == Variant::DecodeSampled));
        assert!(sampled
            .iter()
            .any(|bucket| bucket.variant == Variant::DecodeSampled));
        // Both sets serve the same decode shape, greedily.
        let shape = Shape {
            rows: 4,
            tokens: 4,
            variant: Variant::DecodeGreedy,
        };
        assert_eq!(greedy.select(&shape), sampled.select(&shape));
    }
}
