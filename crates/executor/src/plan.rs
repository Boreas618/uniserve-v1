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
}
