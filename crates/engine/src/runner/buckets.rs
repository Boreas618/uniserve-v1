//! Bucket choice: which captured size a forward is padded into.
//!
//! A captured graph addresses fixed buffers, so a forward may only be served by a graph captured
//! for a size that holds it, and each batch runs in the smallest captured bucket that holds it.
//! This module is the arithmetic of that rule and nothing else: a [`Shape`] states the rows and
//! tokens of one kind of work and, for a media computation, the extent it is captured for, a
//! [`Bucket`] is a captured size, and [`Buckets`] selects the captured bucket that serves a shape
//! and states the order in which the buckets are captured.
//!
//! The ordering rules:
//!
//! - the smallest captured bucket that holds a forward serves it, never a larger one, and a
//!   forward no bucket holds is an error, never an eager fallback;
//! - captures run largest first, so the smaller captures reuse the allocator blocks of the
//!   larger ones.
//!
//! Everything here is host arithmetic on plain integers, so it is testable without a device and
//! runs on the scheduler's critical path without allocating: [`Buckets::select`] scans a sorted
//! vector and [`Bucket::holds`] compares a few integers.

use std::collections::BTreeSet;

use std::fmt;

use crate::batch::{Extent, Shape, Variant};

use crate::worker::error::{Error, Result};

/// A captured size: the rows and tokens a bucket's graph holds, and what it computes.
///
/// Buckets are ordered by size, variant first, then rows, then tokens, so that the smallest
/// bucket holding a shape is the first one an ascending scan finds and the largest bucket is the
/// last. The ordering is what the capture order and the preemption order both read.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Bucket {
    pub variant: Variant,
    pub rows: u32,
    pub tokens: u32,
    /// A media computation's extent, and [`Extent::None`] for the four causal-LM variants.
    pub extent: Extent,
    /// The request this bucket serves as `(frames, width, height)`, when the extent alone does not
    /// say: a layout states latents, and a request states output frames and a resolution.
    pub serves: Option<(u32, u32, u32)>,
    /// The rows per rank an expert step of this bucket's graph holds, or zero outside an expert
    /// union. A union's ranks agree a step's rows before they launch, and the graph is the one whose
    /// rows hold them: the expert work of a decode step is then shaped for the rows a decode step
    /// exchanges, not for the largest prefill any rank could send.
    pub capacity: u32,
}

impl Bucket {
    /// The bucket of a rank's join: the expert layers of one step, with no batch of its own.
    pub(crate) const fn expert_join() -> Self {
        Self::decode(1, Variant::ExpertJoin)
    }

    /// This bucket, captured for an expert step of `capacity` rows per rank.
    #[must_use]
    pub(crate) const fn at_capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// A decode bucket of `rows` rows.
    pub const fn decode(rows: u32, variant: Variant) -> Self {
        Self {
            variant,
            rows,
            tokens: rows,
            extent: Extent::None,
            serves: None,
            capacity: 0,
        }
    }

    /// A verification bucket of `rows` verifications of at most `drafts` drafts each: a row is
    /// a committed token followed by its drafts, so the bucket holds `rows · (drafts + 1)`
    /// tokens. `variant` is [`Variant::VerifyGreedy`] or [`Variant::VerifySampled`].
    pub(crate) const fn verify(rows: u32, drafts: u32, variant: Variant) -> Self {
        Self::prefill(rows, rows * (drafts + 1), variant)
    }

    /// The drafter's bucket behind a forward of `rows` rows holding `tokens` tokens: its
    /// entries over those tokens, then the drafts each row proposes.
    pub(crate) const fn draft(rows: u32, tokens: u32) -> Self {
        Self::prefill(rows, tokens, Variant::Draft)
    }

    /// A prefill or encode bucket of `rows` rows holding `tokens` tokens.
    ///
    /// An encode is the one media computation whose size is a row count and a token count: a
    /// prompt's text rows and the vision tower's tokens, so it is the one whose bucket needs no
    /// extent of its own.
    pub(crate) const fn prefill(rows: u32, tokens: u32, variant: Variant) -> Self {
        Self {
            variant,
            rows,
            tokens,
            extent: Extent::None,
            serves: None,
            capacity: 0,
        }
    }

    /// A denoising bucket: one step over `rows` requests of one `layout`.
    ///
    /// The layout is the extent because it *is* the size of the work: a step's velocity is one
    /// tensor of the layout's shape, and two layouts of the same row count are different graphs.
    pub(crate) const fn denoise(rows: u32, layout: proto::video::Layout) -> Self {
        Self {
            variant: Variant::Denoise,
            rows,
            tokens: 0,
            extent: Extent::Latents(layout),
            serves: None,
            capacity: 0,
        }
    }

    /// An audio decode bucket: one track of one layout's audio, decoded to interleaved PCM.
    ///
    /// The layout is the extent for the reason a denoise step's is: the track's latent shape, and
    /// the window it decodes to, are the model's own arithmetic, and there is no raster here for a
    /// clip-shaped extent to name - a track-only decode has no clip at all.
    pub(crate) const fn audio(rows: u32, layout: proto::video::Layout) -> Self {
        Self {
            variant: Variant::AudioDecode,
            rows,
            tokens: 0,
            extent: Extent::Latents(layout),
            serves: None,
            capacity: 0,
        }
    }

    /// Whether a forward of `shape` is served by this bucket: the same variant, room for its
    /// rows and tokens, and, for a media computation, the extent the bucket was captured for.
    ///
    /// This is the one statement of the rule, which the selection of a bucket
    /// ([`Buckets::select`]) and the worker's routing of a batch to its runner both read.
    ///
    /// A stated extent is matched exactly; `Extent::None` on a media computation's side is a
    /// caller that does not know the shape, and the bucket's own extent then decides; and a host
    /// bucket holds every computation of its variant, whatever its rows, tokens or resolution,
    /// because it captures no graph and so no size narrows it. A token forward states no extent
    /// and is held by a bucket of its variant, which states none either.
    pub fn holds(&self, shape: &Shape) -> bool {
        if self.variant != shape.variant {
            return false;
        }
        // A host call's bucket states which device serves the call and no bound at all: the stage
        // serves a batch of however many requests of that call arrived, one after the other, and
        // `Bucket::host`'s one row is the shape of one request rather than a capacity, so a
        // batch's rows are not compared against it.
        if self.variant.host() {
            return true;
        }
        self.rows >= shape.rows
            && self.tokens >= shape.tokens
            && match shape.extent {
                Extent::None => true,
                Extent::Request {
                    frames,
                    width,
                    height,
                    text_rows,
                    condition_rows,
                } => self.serves_request(frames, width, height, text_rows, condition_rows),
                stated => self.extent == stated,
            }
    }

    /// Whether this bucket serves a request of `frames` frames at `width` by `height`.
    ///
    /// A clip's bucket is the resolution's whatever the timeline, a track's is the timeline's
    /// whatever the resolution, and a step's is both, in a layout whose prompt rows and condition
    /// rows hold the request's. An encode's states neither.
    fn serves_request(
        &self,
        frames: u32,
        width: u32,
        height: u32,
        text_rows: u32,
        condition_rows: u32,
    ) -> bool {
        let same = |served: (u32, u32, u32)| served == (frames, width, height);
        match (self.variant, self.extent) {
            (
                Variant::Decode,
                Extent::Clip {
                    width: columns,
                    height: rows,
                    ..
                },
            ) => (columns, rows) == (width, height),
            (Variant::AudioDecode, _) => self.serves.is_none_or(|served| served.0 == frames),
            (_, Extent::Latents(layout)) => {
                self.serves.is_none_or(same)
                    && layout.text_rows >= text_rows
                    && layout.condition_rows >= condition_rows
                    && (layout.condition_rows > 0) == (condition_rows > 0)
            }
            _ => self.serves.is_none_or(same),
        }
    }

    /// The same bucket, stating the request it serves.
    pub(crate) const fn serving(mut self, frames: u32, width: u32, height: u32) -> Self {
        self.serves = Some((frames, width, height));
        self
    }

    /// A host call's bucket: a computation no graph is captured for.
    ///
    /// The three host calls of a media output are served by the engine on the host, so their
    /// runner states them without capturing anything: the bucket says which device serves the
    /// call, and one row is the shape of one request rather than a bound, which is why
    /// [`Bucket::holds`] measures no computation against it. It carries no extent
    /// because there is no resolution to state: an audio track's samples come from the decoder's
    /// window and a mux reads what the encodes already wrote.
    pub(crate) const fn host(variant: Variant) -> Self {
        Self {
            variant,
            rows: 1,
            tokens: 0,
            extent: Extent::None,
            serves: None,
            capacity: 0,
        }
    }

    /// A decode bucket: one clip of `frames` raster frames at `height` by `width`.
    pub(crate) const fn clip(frames: u32, height: u32, width: u32) -> Self {
        Self {
            variant: Variant::Decode,
            rows: 1,
            tokens: 0,
            extent: Extent::Clip {
                frames,
                height,
                width,
            },
            serves: None,
            capacity: 0,
        }
    }

    /// Whether this bucket is a causal-LM forward, and the refusal when it is not.
    ///
    /// The two families are refused by name rather than served from one another's graphs: a
    /// causal-LM runner given a media bucket would replay a graph shaped for a latent layout over
    /// token rows, and a mistaken bucket set would read as wrong tokens rather than as the mistake
    /// it is.
    pub(crate) fn causal_lm(self) -> Result<()> {
        if self.extent == Extent::None && !self.variant.media() {
            return Ok(());
        }
        Err(Error::Invariant(format!(
            "bucket {self} is a {} computation of extent {:?}, and a causal-LM runner serves none \
             of it",
            self.variant.name(),
            self.extent
        )))
    }

    /// The shape a forward of this bucket has when the bucket is exactly full.
    pub(crate) const fn shape(self) -> Shape {
        Shape {
            rows: self.rows,
            tokens: self.tokens,
            variant: self.variant,
            extent: self.extent,
        }
    }
}

impl fmt::Display for Bucket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.extent {
            Extent::None if self.capacity > 0 => {
                write!(formatter, "{}/{} expert rows", self.shape(), self.capacity)
            }
            Extent::None => self.shape().fmt(formatter),
            Extent::Request {
                frames,
                width,
                height,
                ..
            } => write!(
                formatter,
                "{}/{} rows/a request of {frames} frames at {width}x{height}",
                self.variant, self.rows
            ),
            Extent::Latents(layout) => write!(
                formatter,
                "{}/{} rows/{} latents {}x{}x{}{} and {} condition rows",
                self.variant,
                self.rows,
                layout.video.frames,
                layout.video.height,
                layout.video.width,
                layout.video.channels,
                if layout.audio.is_some() {
                    " with audio"
                } else {
                    ""
                },
                layout.condition_rows
            ),
            Extent::Clip {
                frames,
                height,
                width,
            } => write!(
                formatter,
                "{}/{} clip {frames}x{height}x{width}",
                self.variant, self.rows
            ),
        }
    }
}

/// The decode row counts captured when the build config names none: dense small batches kept exact,
/// then strides of eight, because a bucket is captured for every size in the list.
pub(crate) fn default_decode_rows() -> Vec<u32> {
    (1..=32).chain((40..=128).step_by(8)).collect()
}

/// The prefill token counts captured when the build config names none: fine near one prompt's first
/// chunk, coarser as chunks grow, which is where a bucket's cost is amortised.
pub(crate) fn default_prefill_tokens() -> Vec<u32> {
    (4..=32)
        .step_by(4)
        .chain((48..=256).step_by(16))
        .chain((288..=512).step_by(32))
        .chain((576..=4096).step_by(64))
        .chain((4608..=16384).step_by(512))
        .collect()
}

/// The row buckets of prefill graphs for a batch of at most `rows` rows.
///
/// Powers of two from eight, then `rows`, so that every batch the rank forms fits one and a bucket
/// at most doubles its rows. The last power of two below `rows` merges into it rather than leaving
/// a bucket that holds few row counts.
pub(crate) fn prefill_row_buckets(rows: u32) -> Vec<u32> {
    let mut buckets = Vec::new();
    let mut size = 8;
    while 2 * size <= rows {
        buckets.push(size);
        size *= 2;
    }
    if rows > 0 {
        buckets.push(rows);
    }
    buckets
}

/// The buckets one runner captures, in the order it captures them.
///
/// The set is fixed when its runner is built, from the build config's graph sizes
/// (`cuda_graph_config`): a bucket the set does not hold is a configuration error found at
/// startup, not a scheduling surprise at the first odd-sized batch. The vector is kept sorted
/// ascending and scanned backwards for capture, so both hot paths are a single scan.
#[derive(Clone, Default, Debug)]
pub struct Buckets {
    /// Every bucket, ascending by size.
    ascending: Vec<Bucket>,
}

impl Buckets {
    /// The bucket set holding exactly `buckets`, each once.
    pub(crate) fn new(buckets: impl IntoIterator<Item = Bucket>) -> Self {
        let set: BTreeSet<Bucket> = buckets.into_iter().collect();
        Self {
            ascending: set.into_iter().collect(),
        }
    }

    /// The bucket set of a build config's decode row sizes and prefill row and token sizes.
    ///
    /// Decode sizes are their own buckets, because a decode forward's row count is its token
    /// count. A prefill bucket holds any batch of fewer rows, whose rows past its own are empty
    /// sequences, so its rows are the coarse [`prefill_row_buckets`] of the largest size: every
    /// decode size against every token size would be thousands of graphs that serve no batch
    /// better. A prefill that scores its prompt ([`Variant::PrefillScored`]) is captured at every
    /// prefill size as well, so a request that asks for prompt logprobs is chunked and batched as
    /// any other.
    pub fn causal_lm(rows: &[u32], tokens: &[u32], sampled: bool) -> Self {
        let mut buckets = Vec::new();
        for &rows in rows {
            buckets.push(Bucket::decode(rows, Variant::DecodeGreedy));
            if sampled {
                buckets.push(Bucket::decode(rows, Variant::DecodeSampled));
            }
        }
        for bucket in prefill_row_buckets(rows.iter().copied().max().unwrap_or(0)) {
            for &count in tokens {
                // A prefill chunk never holds fewer tokens than rows: a row holds at least one.
                for variant in [
                    Variant::PrefillWrite,
                    Variant::PrefillOutput,
                    Variant::PrefillGreedy,
                    Variant::PrefillScored,
                ] {
                    buckets.push(Bucket::prefill(bucket, count.max(bucket), variant));
                }
            }
        }
        Self::new(buckets)
    }

    /// The bucket set of a rank that admits batches of `rows` rows and `tokens` tokens, from the
    /// sizes the deployment configured and the default sizes where it configured none.
    ///
    /// A bucket is captured for each size and never for a size the batch bound cannot admit, so a
    /// configured size above the bound is dropped rather than captured and never used. `sampled`
    /// also captures the sampling variants, which a deployment that samples needs and one that
    /// serves logits only does not pay for. Sizes that all exceed the bound are refused here, at
    /// startup, rather than at the first batch of an unusual shape.
    pub(crate) fn for_batch(
        decode: Option<&[u32]>,
        prefill: Option<&[u32]>,
        rows: u32,
        tokens: u32,
        sampled: bool,
    ) -> Result<Self> {
        let mut decode: Vec<u32> = match decode {
            Some(sizes) => sizes.to_vec(),
            None => default_decode_rows(),
        };
        let mut prefill: Vec<u32> = match prefill {
            Some(sizes) => sizes.to_vec(),
            None => default_prefill_tokens(),
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

    /// The bucket set of an epoch that fixes no batch bound of its own: the configured sizes, or the
    /// default sizes where the config names none, and no size dropped.
    ///
    /// This is the set an epoch captures; the bound it implies - the largest decode rows and the
    /// largest prefill tokens - is what capacity fitting and the scheduler admit, so the buckets
    /// come first and the batch bound follows them rather than the other way round.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
    pub(crate) fn default_causal_lm(
        decode: Option<&[u32]>,
        prefill: Option<&[u32]>,
        sampled: bool,
    ) -> Self {
        let decode = decode.map_or_else(default_decode_rows, <[u32]>::to_vec);
        let prefill = prefill.map_or_else(default_prefill_tokens, <[u32]>::to_vec);
        // A zero size names no bucket; a config that states one is not refused, the size is
        // dropped, since the rest of the set still serves.
        let decode: Vec<u32> = decode.into_iter().filter(|size| *size != 0).collect();
        let prefill: Vec<u32> = prefill.into_iter().filter(|size| *size != 0).collect();
        Self::causal_lm(&decode, &prefill, sampled)
    }

    /// The set of a deployment that speculates with its model's drafter, verifying at most
    /// `drafts` drafts a row: each decode bucket becomes the verification bucket of its rows
    /// ([`Bucket::verify`]), because a speculating request's every decode step is a
    /// verification, and every prefill and verification size gains the drafter's bucket
    /// ([`Bucket::draft`]), which is issued behind the forward it follows. A set that
    /// verifies no draft is returned as it is.
    #[must_use]
    pub(crate) fn with_drafts(&self, drafts: u32) -> Self {
        if drafts == 0 {
            return self.clone();
        }

        let mut buckets = Vec::with_capacity(2 * self.ascending.len());
        for bucket in &self.ascending {
            let target = match bucket.variant {
                Variant::DecodeGreedy => Bucket::verify(bucket.rows, drafts, Variant::VerifyGreedy),
                Variant::DecodeSampled => {
                    Bucket::verify(bucket.rows, drafts, Variant::VerifySampled)
                }
                _ => *bucket,
            };
            buckets.push(target);

            // A media computation and an expert join have no token the drafter reads.
            let drafted = target.extent == Extent::None
                && !target.variant.media()
                && target.variant != Variant::ExpertJoin;
            if drafted {
                buckets.push(Bucket::draft(target.rows, target.tokens));
            }
        }
        Self::new(buckets)
    }

    /// The largest decode row count and prefill token count the set captures: **the batch bound the
    /// scheduler may admit**, not an estimate of one.
    ///
    /// A rank that serves captured graphs only is never given a batch larger than its largest
    /// bucket, so the bucket set is the rank's admission bound and the two must not drift: whatever
    /// states a batch capacity for this rank (what the rank reports to the engine) states these
    /// figures, not the build config's, which may be larger. [`Self::require`] is the other half - a
    /// shape that exceeds the bound is refused naming the bound and the shape, so a scheduler bug is
    /// a named refusal rather than a batch no graph can serve.
    pub(crate) fn bounds(&self) -> (u32, u32) {
        let mut rows = 0;
        let mut tokens = 0;
        for bucket in self.iter() {
            // A media computation is admitted by its extent as well as its size, and its rows
            // and tokens are not a language-model batch size: counting one here would let a decode
            // clip's single row or a denoising step's zero tokens move the admission bound a
            // scheduler states for the token path. **A host call's bucket states no extent either**
            // - it captures no graph - so the variant is what keeps its one row out of the bound.
            if bucket.extent != Extent::None || bucket.variant.host() {
                continue;
            }
            rows = rows.max(bucket.rows);
            tokens = tokens.max(bucket.tokens);
        }
        (rows, tokens)
    }

    /// Every bucket, ascending by size.
    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = &Bucket> + ExactSizeIterator {
        self.ascending.iter()
    }

    /// How many buckets the set holds.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
    pub(crate) fn len(&self) -> usize {
        self.ascending.len()
    }

    /// Whether the set holds no bucket.
    pub(crate) fn is_empty(&self) -> bool {
        self.ascending.is_empty()
    }

    /// The smallest bucket that holds `shape`, if the set has one ([`Bucket::holds`]).
    ///
    /// A media computation is selected by what it is: the variant says which kind of graph it
    /// needs, and rows and tokens are room to spare. **An extent a caller states narrows the
    /// search; one it cannot state does not.** A caller that states a `Latents` layout or a
    /// `Clip` is matched to the graph captured for exactly that, so a 25-frame step cannot run in
    /// a 13-frame graph; `Extent::None` is the caller saying it does not know, and the bucket's
    /// own extent, returned with it, decides what the runner does with the batch. No bucket
    /// serving a computation is the worker stating it does not serve it.
    pub(crate) fn select(&self, shape: &Shape) -> Option<Bucket> {
        self.ascending
            .iter()
            .copied()
            .find(|bucket| bucket.holds(shape))
    }

    /// `bucket`'s graph for an expert step of `capacity` rows per rank: the smallest of its
    /// captures whose rows hold them.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
    pub(crate) fn at_capacity(&self, bucket: Bucket, capacity: u32) -> Option<Bucket> {
        self.ascending.iter().copied().find(|candidate| {
            candidate.at_capacity(0) == bucket.at_capacity(0) && candidate.capacity >= capacity
        })
    }

    /// The smallest captured bucket that holds `shape`, or a refusal naming the shape.
    ///
    /// A batch runs in the smallest captured bucket that holds it, because a larger bucket still
    /// produces correct tokens and only costs throughput while a smaller one would corrupt them. A
    /// shape no bucket holds is refused where it is selected - by name, with the shape and the
    /// variant - rather than clamped down to a bucket that does not hold it, or grown into one that
    /// was never captured.
    pub(crate) fn require(&self, shape: &Shape) -> Result<Bucket> {
        self.select(shape).ok_or_else(|| {
            let (rows, tokens) = self.bounds();
            Error::Invalid(format!(
                "no captured bucket holds {shape}: this rank's warmed set tops at {rows} rows and \
                 {tokens} tokens, and a batch above that bound must be admitted as smaller ones - \
                 no capture is made in the serving path"
            ))
        })
    }

    /// The buckets in capture order: largest first, so smaller captures reuse the allocator blocks
    /// of the larger ones.
    pub(crate) fn capture_order(&self) -> impl Iterator<Item = &Bucket> {
        self.ascending.iter().rev()
    }

    /// The largest bucket of `variant`, which that variant's static activations are sized for.
    #[expect(dead_code, reason = "no caller in the engine")]
    pub(crate) fn largest(&self, variant: Variant) -> Option<Bucket> {
        self.ascending
            .iter()
            .rev()
            .copied()
            .find(|bucket| bucket.variant == variant)
    }

    /// Whether the set holds `bucket`.
    pub(crate) fn contains(&self, bucket: &Bucket) -> bool {
        self.ascending.binary_search(bucket).is_ok()
    }

    /// The largest bucket of the set: the one whose footprint an allocator must be able to hold
    /// before the set is captured.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
    pub(crate) fn largest_footprint(&self) -> Option<Bucket> {
        self.ascending.last().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request finds the bucket of its own timeline, resolution and prompt, and never a longer
    /// one.
    #[test]
    fn a_request_selects_the_bucket_of_its_own_timeline() {
        let layout = |frames: u32, text_rows: u32| proto::video::Layout {
            video: proto::video::LatentShape {
                channels: 24,
                frames,
                height: 48,
                width: 84,
            },
            audio: None,
            text_rows,
            condition_rows: 0,
            condition_elements: 0,
        };
        let short = Bucket::denoise(1, layout(37, 1024)).serving(124, 1344, 768);
        let wide = Bucket::denoise(1, layout(37, 2048)).serving(124, 1344, 768);
        let long = Bucket::denoise(1, layout(57, 1024)).serving(192, 1344, 768);
        let buckets = Buckets::new([short, wide, long]);
        let request = |frames, text_rows| Shape {
            variant: Variant::Denoise,
            rows: 1,
            tokens: 0,
            extent: Extent::Request {
                frames,
                width: 1344,
                height: 768,
                text_rows,
                condition_rows: 0,
            },
        };
        assert_eq!(buckets.select(&request(124, 537)), Some(short));
        assert_eq!(buckets.select(&request(124, 1500)), Some(wide));
        assert_eq!(buckets.select(&request(192, 537)), Some(long));
        assert_eq!(buckets.select(&request(141, 537)), None);
    }

    /// A request without conditions finds the layout that holds none, and one with conditions the
    /// smallest layout that holds them, because a layout's condition rows are real rows of every
    /// step. Conditions no layout holds find none.
    #[test]
    fn a_request_selects_the_layout_of_its_own_conditions() {
        let layout = |condition_rows: u32| proto::video::Layout {
            video: proto::video::LatentShape {
                channels: 24,
                frames: 37,
                height: 48,
                width: 84,
            },
            audio: None,
            text_rows: 1024,
            condition_rows,
            condition_elements: 0,
        };
        let plain = Bucket::denoise(1, layout(0)).serving(124, 1344, 768);
        let low = Bucket::denoise(1, layout(2048)).serving(124, 1344, 768);
        let high = Bucket::denoise(1, layout(8192)).serving(124, 1344, 768);
        let buckets = Buckets::new([high, low, plain]);
        let request = |condition_rows| Shape {
            variant: Variant::Denoise,
            rows: 1,
            tokens: 0,
            extent: Extent::Request {
                frames: 124,
                width: 1344,
                height: 768,
                text_rows: 537,
                condition_rows,
            },
        };
        assert_eq!(buckets.select(&request(0)), Some(plain));
        assert_eq!(buckets.select(&request(1024)), Some(low));
        assert_eq!(buckets.select(&request(2048)), Some(low));
        assert_eq!(buckets.select(&request(2176)), Some(high));
        assert_eq!(buckets.select(&request(8192)), Some(high));
        assert_eq!(buckets.select(&request(8320)), None);
    }

    /// A host call's bucket serves its call whatever the batch holds, because it states no bound.
    ///
    /// The three host calls of a media output are served by the engine on the host, one request
    /// after another, so a batch of two of them is the same call twice and the same bucket serves
    /// both: its one row is the shape of one request and not a capacity.
    #[test]
    fn a_host_bucket_serves_every_request_of_its_call() {
        let buckets = Buckets::new([
            Bucket::host(Variant::VideoEncode),
            Bucket::host(Variant::AudioEncode),
            Bucket::host(Variant::Mux),
            Bucket::clip(48, 768, 1344),
        ]);

        for variant in [Variant::VideoEncode, Variant::AudioEncode, Variant::Mux] {
            for rows in [1, 2, 4] {
                let batch = Shape {
                    variant,
                    rows,
                    tokens: 0,
                    extent: Extent::None,
                };
                assert_eq!(
                    buckets.select(&batch),
                    Some(Bucket::host(variant)),
                    "a {variant} batch of {rows} request(s)"
                );
            }
        }

        // A device computation is still measured against its bucket: two clips are not one capture.
        let two_clips = Shape {
            variant: Variant::Decode,
            rows: 2,
            tokens: 0,
            extent: Extent::Clip {
                frames: 48,
                height: 768,
                width: 1344,
            },
        };
        assert_eq!(buckets.select(&two_clips), None);
    }

    /// The worker states which media computations its buckets serve, and the scheduler forms a
    /// batch only where the selection is a bucket: a denoising step is served only by a graph
    /// captured for its own latent layout, a decode only by one captured for its own clip, and a
    /// call no variant states - audio decoding, a mux - is served nowhere rather than by the nearest
    /// graph.
    #[test]
    fn a_media_computation_selects_only_a_bucket_captured_for_it() {
        let layout = proto::video::Layout {
            video: proto::video::LatentShape {
                channels: 16,
                frames: 13,
                height: 60,
                width: 34,
            },
            audio: None,
            text_rows: 256,
            condition_rows: 0,
            condition_elements: 0,
        };
        let buckets = Buckets::new([
            Bucket::denoise(1, layout),
            Bucket::clip(48, 768, 1344),
            Bucket::prefill(1, 512, Variant::Encode),
        ]);

        let denoise = Shape {
            variant: Variant::Denoise,
            rows: 1,
            tokens: 0,
            extent: Extent::Latents(layout),
        };
        assert_eq!(buckets.select(&denoise), Some(Bucket::denoise(1, layout)));

        // One frame more is another graph, and no capture is made in the serving path.
        let other = Shape {
            extent: Extent::Latents(proto::video::Layout {
                video: proto::video::LatentShape {
                    channels: 16,
                    frames: 25,
                    height: 60,
                    width: 34,
                },
                ..layout
            }),
            ..denoise
        };
        assert_eq!(buckets.select(&other), None);

        // The engine forms every media computation with `Extent::None`: a step's layout is the
        // model's and a clip is the request's, and the scheduler knows neither until the runner that
        // states the bucket reports it. The bucket's own extent decides then.
        let from_engine = Shape {
            extent: Extent::None,
            ..denoise
        };
        assert_eq!(
            buckets.select(&from_engine),
            Some(Bucket::denoise(1, layout)),
            "a computation that cannot state its extent is served by the bucket that states one"
        );

        let clip = Shape {
            variant: Variant::Decode,
            rows: 1,
            tokens: 0,
            extent: Extent::Clip {
                frames: 48,
                height: 768,
                width: 1344,
            },
        };
        assert_eq!(buckets.select(&clip), Some(Bucket::clip(48, 768, 1344)));

        // The same for a decode: the engine states `Extent::None` and the clip's bucket serves it.
        let clip_from_engine = Shape {
            extent: Extent::None,
            ..clip
        };
        assert_eq!(
            buckets.select(&clip_from_engine),
            Some(Bucket::clip(48, 768, 1344))
        );

        // An encode is sized by rows and tokens like a prefill, so it is the one media computation
        // whose extent says nothing.
        let encode = Shape {
            variant: Variant::Encode,
            rows: 1,
            tokens: 300,
            extent: Extent::None,
        };
        assert_eq!(
            buckets.select(&encode),
            Some(Bucket::prefill(1, 512, Variant::Encode))
        );

        // The admission bound is the token path's: a media bucket's rows and tokens are not a
        // language-model batch size and must not move it - and a host call's bucket states no
        // extent at all, so the variant is what keeps its one row out of the bound too.
        assert_eq!(buckets.bounds(), (1, 512));
        assert_eq!(
            Buckets::new([
                Bucket::host(Variant::VideoEncode),
                Bucket::host(Variant::Mux)
            ])
            .bounds(),
            (0, 0),
            "a set of host calls states no token admission bound"
        );

        // A track-only decode is a variant of its own whose whole extent is the layout: a `t2va`
        // clip carries a soundtrack, and the decode of the track has no raster to name. Two
        // resolutions of one duration collapse onto one track, which is why the bucket is built
        // from the layout rather than from the clip the denoiser took it from.
        let track = Bucket::audio(
            1,
            proto::video::Layout {
                audio: Some(proto::video::LatentShape {
                    channels: 8,
                    frames: 3,
                    height: 2,
                    width: 4,
                }),
                ..layout
            },
        );
        let tracks = Buckets::new([track]);
        let decode = Shape {
            variant: Variant::AudioDecode,
            rows: 1,
            tokens: 0,
            extent: Extent::None,
        };
        assert_eq!(
            tracks.select(&decode),
            Some(track),
            "a track's decode is served by the bucket that states its layout"
        );

        // The clip's decode is not served by the track's bucket, though both name a layout in some
        // form: the variant is what keeps them apart, and that is the whole reason the track could
        // reuse `Extent::Latents` instead of putting a second shape in the extent.
        let clip_decode = Shape {
            variant: Variant::Decode,
            ..decode
        };
        assert_eq!(tracks.select(&clip_decode), None);

        // The three host calls of a media output are computations of their own - the engine
        // encodes and muxes them on the host - and no device bucket serves one: they state a variant
        // so a batch reaches the runner that does, and they select no graph.
        for variant in [Variant::VideoEncode, Variant::AudioEncode, Variant::Mux] {
            let host = Shape {
                variant,
                rows: 1,
                tokens: 0,
                extent: Extent::None,
            };
            assert!(variant.media());
            assert_eq!(tracks.select(&host), None);
        }

        // A host bucket captures no graph, so no resolution narrows it: a video encode states the
        // clip it works at and the stage that serves it states no extent at all. A bucket captured
        // for a clip still holds that one and no other, which is the half of the rule that keeps a
        // 25-frame step out of a 13-frame graph.
        let host = Buckets::new([Bucket::host(Variant::VideoEncode)]);
        let clip_encode = Shape {
            variant: Variant::VideoEncode,
            rows: 1,
            tokens: 0,
            extent: Extent::Clip {
                frames: 17,
                height: 2,
                width: 2,
            },
        };
        assert_eq!(
            host.select(&clip_encode),
            Some(Bucket::host(Variant::VideoEncode)),
            "a call performed on the host is served whatever resolution it names"
        );
        assert_eq!(
            buckets.select(&Shape {
                extent: Extent::Clip {
                    frames: 25,
                    height: 768,
                    width: 1344,
                },
                ..clip
            }),
            None,
            "a captured bucket holds the clip it was captured for and no other"
        );

        // A media read is host work: a bucket set that states no host bucket for it serves it
        // nowhere, and the scheduler reads that as the capability it is.
        let read = Shape {
            variant: Variant::MediaRead,
            rows: 1,
            tokens: 0,
            extent: Extent::None,
        };
        assert_eq!(buckets.select(&read), None);
        assert_eq!(tracks.select(&read), None);
    }

    /// The two families are refused by name rather than served from one another's graphs: a media
    /// bucket states an extent a causal-LM forward has no use for, and a media variant is never a
    /// causal-LM variant - a bucket set that mixed them must fail where the batch selects its
    /// bucket, not serve tokens shaped by a graph captured for a latent layout.
    #[test]
    fn a_media_bucket_is_sized_by_its_extent_and_refused_by_a_causal_lm_runner() {
        let layout = proto::video::Layout {
            video: proto::video::LatentShape {
                channels: 16,
                frames: 13,
                height: 60,
                width: 34,
            },
            audio: None,
            text_rows: 256,
            condition_rows: 0,
            condition_elements: 0,
        };
        let denoise = Bucket::denoise(1, layout);
        assert_eq!(denoise.variant, Variant::Denoise);
        assert_eq!(denoise.extent, Extent::Latents(layout));
        assert!(Variant::Denoise.media());

        let clip = Bucket::clip(48, 768, 1344);
        assert_eq!(clip.variant, Variant::Decode);
        assert_eq!(clip.rows, 1, "one clip per call is the bound");
        assert!(Variant::Decode.media());

        for bucket in [denoise, clip] {
            let refusal = bucket
                .causal_lm()
                .expect_err("a causal-LM runner serves no media bucket");
            let message = refusal.to_string();
            assert!(
                message.contains(bucket.variant.name()),
                "the refusal names the computation: {message}"
            );
        }

        let decode = Bucket::decode(8, Variant::DecodeGreedy);
        assert_eq!(decode.extent, Extent::None);
        assert!(!Variant::DecodeGreedy.media());
        decode
            .causal_lm()
            .expect("a decode bucket is a causal-LM bucket");

        // Two layouts of one row count are two graphs, and the message states the layout rather
        // than a token count nobody holds.
        let other = Bucket::denoise(
            1,
            proto::video::Layout {
                video: proto::video::LatentShape {
                    channels: 16,
                    frames: 25,
                    height: 60,
                    width: 34,
                },
                ..layout
            },
        );
        assert_ne!(other, denoise);
        assert!(
            other.to_string().contains("25"),
            "the display states the layout: {other}"
        );
    }

    /// Every shape a scheduler can submit selects the **smallest** captured bucket that holds it,
    /// derived independently by brute force over the captured set: a larger bucket still produces
    /// correct tokens and costs throughput, a smaller one would corrupt them.
    ///
    /// The shapes are the ones a batch states, which is what makes the bucket set able to hold them
    /// at all: a decode forward's row count is its token count, and a prefill chunk holds at least
    /// as many tokens as rows. A shape outside both domains is the other test's subject - it is
    /// refused by name rather than selected, and expecting a bucket for it here would assert a
    /// coverage the bucket set does not claim.
    #[test]
    fn every_shape_selects_the_smallest_bucket_that_holds_it() {
        let buckets = Buckets::causal_lm(&[1, 8, 32], &[128, 512], true);
        let captured: Vec<Bucket> = buckets.iter().copied().collect();
        assert!(!captured.is_empty());
        let mut selected = 0;
        for &variant in &[
            Variant::DecodeGreedy,
            Variant::DecodeSampled,
            Variant::PrefillWrite,
            Variant::PrefillOutput,
        ] {
            let decode = matches!(variant, Variant::DecodeGreedy | Variant::DecodeSampled);
            for rows in 1..=32u32 {
                let counts: Box<dyn Iterator<Item = u32>> = if decode {
                    Box::new(std::iter::once(rows))
                } else {
                    Box::new(rows..=512)
                };
                for tokens in counts {
                    let shape = Shape {
                        rows,
                        tokens,
                        variant,
                        extent: Extent::None,
                    };
                    // Brute force: the smallest holder by rows and then tokens, computed from the
                    // captured set rather than taken from the code under test.
                    let expected = captured
                        .iter()
                        .copied()
                        .filter(|bucket| bucket.holds(&shape))
                        .min_by_key(|bucket| (bucket.rows, bucket.tokens));
                    let Some(expected) = expected else {
                        assert_eq!(
                            buckets.select(&shape),
                            None,
                            "{shape} has no holder and must not select a bucket"
                        );
                        continue;
                    };
                    selected += 1;
                    assert_eq!(
                        buckets.select(&shape),
                        Some(expected),
                        "{shape} must select its smallest holder"
                    );
                    // Whatever it picks does hold the shape, and nothing smaller does.
                    for bucket in &captured {
                        if bucket.holds(&shape) {
                            assert!(
                                (bucket.rows, bucket.tokens) >= (expected.rows, expected.tokens),
                                "{bucket} is smaller than {expected} for {shape}"
                            );
                        }
                    }
                }
            }
        }
        assert!(selected > 0, "no shape selected a bucket at all");
    }

    /// A shape the bucket set does not cover is refused by name - with the shape and its variant -
    /// and never clamped into a bucket that does not hold it or grown into one that was not
    /// captured.
    #[test]
    fn an_uncovered_shape_is_refused_by_name() {
        let buckets = Buckets::causal_lm(&[1, 8], &[128], false);
        // Beyond the largest captured token count of a prefill variant.
        let long = Shape::prefill(1, 129, Variant::PrefillOutput);
        let error = buckets
            .require(&long)
            .expect_err("no bucket holds 129 tokens");
        let message = error.to_string();
        assert!(message.contains(&long.to_string()), "{message}");
        assert!(message.contains("no captured bucket"), "{message}");
        // Beyond the largest captured row count of a decode variant.
        let wide = Shape::decode(9, Variant::DecodeGreedy);
        assert!(buckets.select(&wide).is_none(), "no bucket holds 9 rows");
        assert!(buckets.require(&wide).is_err());
        // A variant the deployment never captured: a greedy-only build has no sampled bucket.
        let sampled = Shape::decode(1, Variant::DecodeSampled);
        assert!(buckets.select(&sampled).is_none());
        assert!(buckets.require(&sampled).is_err());
        // The refusal names the bound as well as the shape, so the shape a scheduler admitted and
        // the bound it exceeded are both in the one line a deployment's reader has.
        let (rows, tokens) = buckets.bounds();
        assert!(
            message.contains(&format!("{rows} rows"))
                && message.contains(&format!("{tokens} tokens")),
            "the refusal does not state this rank's bound: {message}"
        );
    }

    /// The bound a rank reports as its batch capacity is this set's `bounds()`, so no batch the
    /// scheduler admits is larger than the largest warmed bucket - and a build config asking for
    /// more is clamped by the default sizes rather than honoured.
    ///
    /// This is the invariant that keeps a serving batch from reaching a bucket nobody captured:
    /// `require` would refuse it, and a refusal in the serving path ends the epoch, so the bound
    /// must never be stated above what is warmed.
    #[test]
    fn the_admission_bound_is_the_largest_warmed_bucket_and_never_the_build_configs() {
        // A deployment that asks for far more than the default sizes hold.
        let buckets =
            Buckets::for_batch(None, None, 256, 1_000_000, true).expect("the default sizes");
        let (rows, tokens) = buckets.bounds();
        assert!(
            (rows, tokens) == (128, 16384),
            "the default sizes top at {rows} rows, {tokens} tokens"
        );

        // Every shape inside the bound selects a bucket, at both edges and along each axis: the
        // bound is only useful if it is covered, and a bucket set with a hole inside its own bound
        // would be exactly the drift this test exists to catch.
        for variant in [
            Variant::DecodeGreedy,
            Variant::DecodeSampled,
            Variant::PrefillWrite,
            Variant::PrefillOutput,
        ] {
            if variant.is_decode() {
                for count in 1..=rows {
                    let shape = Shape::decode(count, variant);
                    assert!(
                        buckets.select(&shape).is_some(),
                        "{shape} is inside the bound"
                    );
                }
            } else {
                for count in 1..=tokens {
                    let shape = Shape::prefill(1, count, variant);
                    assert!(
                        buckets.select(&shape).is_some(),
                        "{shape} is inside the bound"
                    );
                }
                // The widest prefill the bound admits: every row count with the largest chunk.
                for row in 1..=rows {
                    let shape = Shape::prefill(row, tokens, variant);
                    assert!(
                        buckets.select(&shape).is_some(),
                        "{shape} is inside the bound"
                    );
                }
            }
            // One past the bound is refused by name, never clamped into the largest bucket.
            let over = match variant.is_decode() {
                true => Shape::decode(rows + 1, variant),
                false => Shape::prefill(1, tokens + 1, variant),
            };
            assert!(buckets.require(&over).is_err(), "{over} is past the bound");
        }
    }

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
                variant: bucket.variant,
                extent: Extent::None,
            }));
        }
    }

    /// One capture per bucket, and a bucket per (size, computation) pair the batch bound admits:
    /// the first generation's capture set is the bucket set, so its size follows from the
    /// configured sizes and nothing else.
    ///
    /// The figure matters because the first generation has no serving generation to compare
    /// against - every operator of every bucket is new - so this count is the work it does before it
    /// serves a request, and no pool accounts for the one executable each capture holds.
    #[test]
    fn the_capture_set_is_the_bucket_set() {
        // The e2e deployment's own sizes: 4 decode sizes and one prefill size, sampling on.
        let buckets = Buckets::for_batch(Some(&[1, 2, 4, 8]), Some(&[512]), 256, 40960, true)
            .expect("the deployment's sizes");
        // 4 sizes x (greedy + sampled) decode, and one row bucket x 1 token count x (write, drawn
        // output, greedy output and scored) prefill: a prefill bucket holds any batch of fewer rows.
        assert_eq!(buckets.len(), 12, "12 buckets, so 12 captures");

        // With no configured size the default sizes are the set, and every size they state is
        // captured with each computation it admits. The count is derived from them here rather than
        // written as a figure: a chunk never holds fewer tokens than its rows, so two default sizes
        // can name one bucket and the set holds it once.
        let defaults =
            Buckets::for_batch(None, None, u32::MAX, u32::MAX, true).expect("the default sizes");
        let (rows, tokens) = (default_decode_rows(), default_prefill_tokens());
        let mut stated: std::collections::BTreeSet<Bucket> = std::collections::BTreeSet::new();
        for &row in &rows {
            stated.insert(Bucket::decode(row, Variant::DecodeGreedy));
            stated.insert(Bucket::decode(row, Variant::DecodeSampled));
        }
        let largest = rows.iter().copied().max().unwrap_or(0);
        for row in prefill_row_buckets(largest) {
            for &count in &tokens {
                for variant in [
                    Variant::PrefillWrite,
                    Variant::PrefillOutput,
                    Variant::PrefillGreedy,
                    Variant::PrefillScored,
                ] {
                    stated.insert(Bucket::prefill(row, count.max(row), variant));
                }
            }
        }
        assert_eq!(
            defaults.len(),
            stated.len(),
            "one bucket per (variant, rows, tokens) the default sizes state"
        );
        for bucket in &stated {
            assert!(
                defaults.contains(bucket),
                "{bucket} is one of the default buckets"
            );
        }
    }

    /// The default sizes are clamped to what the batch bound admits, so nothing is captured that no
    /// batch can fill, and a batch of the bound's own shape is still served.
    #[test]
    fn the_default_sizes_are_clamped_to_the_batch_bound() {
        let buckets =
            Buckets::for_batch(None, None, 8, 64, false).expect("default sizes within the bound");
        for bucket in buckets.iter() {
            assert!(bucket.rows <= 8, "{bucket} is wider than the batch bound");
            assert!(
                bucket.tokens <= 64,
                "{bucket} is longer than the batch bound"
            );
        }
        let full = Shape {
            rows: 8,
            tokens: 64,
            variant: Variant::PrefillWrite,
            extent: Extent::None,
        };
        assert!(
            buckets.select(&full).is_some(),
            "the bound's own shape is served"
        );
        // Every decode row up to the bound is served by its own bucket.
        for rows in 1..=8 {
            let shape = Shape {
                rows,
                tokens: rows,
                variant: Variant::DecodeGreedy,
                extent: Extent::None,
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
    fn sizes_that_all_exceed_the_bound_are_refused() {
        assert!(Buckets::for_batch(Some(&[256]), Some(&[4096]), 8, 64, false).is_err());
        // One size within the bound is enough for the set to be usable.
        assert!(Buckets::for_batch(Some(&[8, 256]), Some(&[4096]), 8, 64, false).is_ok());
    }

    /// The sampling variants are captured only for a deployment that samples.
    #[test]
    fn sampling_variants_follow_the_deployment() {
        let rows = [4_u32];
        let greedy = Buckets::for_batch(Some(&rows), None, 8, 64, false).expect("a bucket set");
        let sampled = Buckets::for_batch(Some(&rows), None, 8, 64, true).expect("a bucket set");
        assert!(
            !greedy
                .iter()
                .any(|bucket| bucket.variant == Variant::DecodeSampled)
        );
        assert!(
            sampled
                .iter()
                .any(|bucket| bucket.variant == Variant::DecodeSampled)
        );
        // Both sets serve the same decode shape, greedily.
        let shape = Shape {
            rows: 4,
            tokens: 4,
            variant: Variant::DecodeGreedy,
            extent: Extent::None,
        };
        assert_eq!(greedy.select(&shape), sampled.select(&shape));
    }

    /// The set an epoch captures implies the batch bound it admits: the largest decode rows and the
    /// largest prefill tokens, which is what capacity fitting and the scheduler read.
    #[test]
    fn the_bounds_are_what_the_set_captures() {
        let buckets = Buckets::default_causal_lm(Some(&[1, 8]), Some(&[0, 64, 256]), true);
        assert_eq!(buckets.bounds(), (8, 256));
        // A zero size names no bucket, and an epoch that configures no size takes the default sizes.
        assert!(!buckets.contains(&Bucket::prefill(8, 0, Variant::PrefillWrite)));
        let defaults = Buckets::default_causal_lm(None, None, false);
        let (rows, tokens) = defaults.bounds();
        assert_eq!(
            rows,
            default_decode_rows()
                .into_iter()
                .max()
                .expect("default sizes")
        );
        assert_eq!(
            tokens,
            default_prefill_tokens()
                .into_iter()
                .max()
                .expect("default sizes")
        );
    }
}
