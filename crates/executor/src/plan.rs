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
    /// One encode of prompt rows: the text model's and the vision tower's outputs, kept as the
    /// conditioning the denoiser reads.
    Encode,
    /// One denoising step of one latent layout: the diffusion model's velocity, applied to the
    /// latents in place (a request's steps are one batch each, never a loop inside a batch).
    Denoise,
    /// One clip decoded from latents into a raster.
    Decode,
}

impl Variant {
    /// The variant's name, as a capture report prints it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DecodeGreedy => "decode_greedy",
            Self::DecodeSampled => "decode_sampled",
            Self::PrefillWrite => "prefill_write",
            Self::PrefillOutput => "prefill_output",
            Self::Encode => "encode",
            Self::Denoise => "denoise",
            Self::Decode => "decode",
        }
    }

    /// Whether this variant is a media computation rather than a causal-LM forward.
    ///
    /// The two families never share a bucket: a media bucket carries an [`Extent`] the language
    /// model has no use for, and a causal-LM runner refuses one by name rather than serving it
    /// from a graph shaped for another computation.
    pub const fn media(self) -> bool {
        matches!(self, Self::Encode | Self::Denoise | Self::Decode)
    }

    /// The computation a media call is, and `None` for a call no variant states yet.
    ///
    /// The three encoders are one computation because they write one thing - the conditioning the
    /// denoiser reads - and differ only in which tower produced it; the three decoders likewise
    /// produce one raster for one muxer. A call with no variant is not served, which the mechanism
    /// reads as a capability the executor does not state rather than as a batch to send anyway.
    pub const fn of_media(call: ipc::MediaCall) -> Option<Self> {
        match call {
            ipc::MediaCall::TextEncoding
            | ipc::MediaCall::VisionEncoding
            | ipc::MediaCall::LatentEncoding => Some(Self::Encode),
            ipc::MediaCall::Denoising => Some(Self::Denoise),
            ipc::MediaCall::ImageDecoding | ipc::MediaCall::VideoDecoding => Some(Self::Decode),
            ipc::MediaCall::MediaReading
            | ipc::MediaCall::LatentPreparation
            | ipc::MediaCall::AudioDecoding
            | ipc::MediaCall::VideoEncoding
            | ipc::MediaCall::AudioEncoding
            | ipc::MediaCall::Muxing => None,
        }
    }

    /// Whether a bucket of this variant picks each row's token by taking the largest candidate
    /// rather than by drawing one (4.4's greedy path).
    ///
    /// A greedy request states temperature zero, whose shaping divides by one, so the row's largest
    /// candidate *is* its token. No draw can produce it - the draw runs in token-id order - so the
    /// two are separate launch sequences and a bucket is homogeneous in which one it takes.
    pub const fn greedy(self) -> bool {
        matches!(self, Self::DecodeGreedy)
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

    /// A prefill or encode forward of `rows` rows holding `tokens` tokens.
    pub const fn prefill(rows: u32, tokens: u32, variant: Variant) -> Self {
        Self {
            rows,
            tokens,
            variant,
        }
    }

    /// The smallest capture size this forward can be served at: itself.
    ///
    /// A `Shape` states a causal-LM forward - rows and tokens - so its bucket holds no media
    /// extent; a media computation is sized by [`Bucket::denoise`] or [`Bucket::clip`], which have
    /// no `Shape` and are planned rather than admitted as rows.
    pub const fn bucket(self) -> Bucket {
        Bucket {
            rows: self.rows,
            tokens: self.tokens,
            variant: self.variant,
            extent: Extent::None,
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

/// One media computation as the mechanism states it, so that a plan can say whether it serves it.
///
/// The three fields are what a media graph's capture depends on: which computation it is, how many
/// requests it holds, and - for a decode or a denoising step - the shape its kernels are captured
/// for. An encode states rows and tokens like a prefill, so its extent is [`Extent::None`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Computation {
    /// The media call the mechanism is about to run.
    pub call: ipc::MediaCall,
    /// The requests in the batch.
    pub rows: u32,
    /// An encode's tokens; zero for a denoising step and a decode, which are sized by their extent.
    pub tokens: u32,
    /// The latent layout of a denoising step, the raster of a decode, and [`Extent::None`] for an
    /// encode.
    pub extent: Extent,
}

impl Computation {
    /// The variant that computes this call, or `None` when no variant states it.
    pub const fn variant(self) -> Option<Variant> {
        Variant::of_media(self.call)
    }

    /// Whether this is a computation the executor serves at all, before any plan is consulted.
    pub const fn stated(self) -> bool {
        self.variant().is_some()
    }
}

/// What a media bucket's computation is sized by, where a causal-LM bucket is sized by tokens
/// alone.
///
/// The language model's four variants leave it `None`, so their buckets are `(variant, rows,
/// tokens)` exactly as before and no causal-LM path reads this field. A media bucket states the
/// shape its graph is captured for: a denoising step is one latent layout, a decode is one clip.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Extent {
    /// No media extent: a causal-LM bucket, sized by its tokens.
    None,
    /// One denoising step's latent layout (`nn::capability::Layout`).
    Latents(nn::capability::Layout),
    /// One decoded clip's raster.
    Clip {
        /// Frames in the clip.
        frames: u32,
        /// Raster rows of a frame.
        height: u32,
        /// Raster columns of a frame.
        width: u32,
    },
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
    /// A media computation's extent, and [`Extent::None`] for the four causal-LM variants.
    pub extent: Extent,
}

impl Bucket {
    /// A decode bucket of `rows` rows.
    pub const fn decode(rows: u32, variant: Variant) -> Self {
        Self {
            variant,
            rows,
            tokens: rows,
            extent: Extent::None,
        }
    }

    /// A prefill or encode bucket of `rows` rows holding `tokens` tokens.
    ///
    /// An encode is the one media computation whose size is a row count and a token count: a
    /// prompt's text rows and the vision tower's tokens, so it is the one whose bucket needs no
    /// extent of its own.
    pub const fn prefill(rows: u32, tokens: u32, variant: Variant) -> Self {
        Self {
            variant,
            rows,
            tokens,
            extent: Extent::None,
        }
    }

    /// A denoising bucket: one step over `rows` requests of one `layout`.
    ///
    /// The layout is the extent because it *is* the size of the work: a step's velocity is one
    /// tensor per canvas, and two layouts of the same row count are different graphs.
    pub const fn denoise(rows: u32, layout: nn::capability::Layout) -> Self {
        Self {
            variant: Variant::Denoise,
            rows,
            tokens: 0,
            extent: Extent::Latents(layout),
        }
    }

    /// A decode bucket: one clip of `frames` raster frames at `height` by `width`.
    pub const fn clip(frames: u32, height: u32, width: u32) -> Self {
        Self {
            variant: Variant::Decode,
            rows: 1,
            tokens: 0,
            extent: Extent::Clip {
                frames,
                height,
                width,
            },
        }
    }

    /// Whether this bucket is a causal-LM forward, and the refusal when it is not.
    ///
    /// The two families are refused by name rather than served from one another's graphs: a
    /// causal-LM runner that was handed a media bucket would replay a graph shaped for a latent
    /// canvas over token rows, and a catalogue typo would read as wrong tokens rather than as the
    /// mistake it is.
    pub fn causal_lm(self) -> Result<()> {
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
        // A media computation is planned by its extent rather than by a `Shape`, and a graph
        // captured for one canvas holds no other, so no shape is served by a media bucket.
        matches!(self.extent, Extent::None)
            && self.variant as u8 == shape.variant as u8
            && self.rows >= shape.rows
            && self.tokens >= shape.tokens
    }
}

impl fmt::Display for Bucket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.extent {
            Extent::None => self.shape().fmt(formatter),
            Extent::Latents(layout) => write!(
                formatter,
                "{}/{} rows/{} latents {}x{}x{}{}",
                self.variant,
                self.rows,
                layout.video.frames,
                layout.video.height,
                layout.video.width,
                layout.video.channels,
                if layout.audio.is_some() { " with audio" } else { "" }
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

    /// The bucket set of an epoch that fixes no batch bound of its own: the configured sizes, or the
    /// catalog where the config names none, and no size dropped.
    ///
    /// This is the set an epoch captures; the bound it implies - the largest decode rows and the
    /// largest prefill tokens - is what the capacity plan and the scheduler admit, so the buckets
    /// come first and the batch bound follows them rather than the other way round.
    pub fn default_causal_lm(decode: Option<&[u32]>, prefill: Option<&[u32]>, sampled: bool) -> Self {
        let decode = decode.map_or_else(decode_catalog, <[u32]>::to_vec);
        let prefill = prefill.map_or_else(prefill_catalog, <[u32]>::to_vec);
        // A zero size names no bucket; a plan that states one is not a plan to refuse, only a size
        // to drop, since the rest of the set still serves.
        let decode: Vec<u32> = decode.into_iter().filter(|size| *size != 0).collect();
        let prefill: Vec<u32> = prefill.into_iter().filter(|size| *size != 0).collect();
        Self::causal_lm(&decode, &prefill, sampled)
    }

    /// The largest decode row count and prefill token count the set captures: **the batch bound the
    /// mechanism may admit**, not an estimate of one.
    ///
    /// A rank that serves captured graphs only is never handed a batch larger than its largest
    /// bucket, so the catalogue is the rank's admission bound and the two must not drift: whatever
    /// states a batch capacity for this rank (the epoch plan's rank facts) states these figures, not
    /// the build config's, which may be larger. [`Self::require`] is the other half - a shape that
    /// exceeds the bound is refused naming the bound and the shape, so a mechanism bug is a named
    /// refusal rather than a batch no graph can serve.
    pub fn bounds(&self) -> (u32, u32) {
        let mut rows = 0;
        let mut tokens = 0;
        for bucket in self.iter() {
            // A media computation is admitted by its own plan (`select_media`), and its rows and
            // tokens are not a language-model batch size: counting one here would let a decode
            // clip's single row or a denoising step's zero tokens move the admission bound a
            // scheduler states for the token path.
            if bucket.extent != Extent::None {
                continue;
            }
            rows = rows.max(bucket.rows);
            tokens = tokens.max(bucket.tokens);
        }
        (rows, tokens)
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

    /// The smallest captured bucket that holds `shape`, or a refusal naming the shape.
    ///
    /// 4.5's planning rule is this one lookup: a batch is planned into the smallest captured bucket
    /// that holds it, because a larger bucket still produces correct tokens and only costs
    /// throughput while a smaller one would corrupt them. A shape no bucket holds is refused where
    /// it is planned - by name, with the shape and the variant - rather than clamped down to a
    /// bucket that does not hold it, or grown into one that was never captured.
    pub fn require(&self, shape: &Shape) -> Result<Bucket> {
        self.select(shape).ok_or_else(|| {
            let (rows, tokens) = self.bounds();
            Error::Invalid(format!(
                "no captured bucket holds {shape}: this rank's warmed set tops at {rows} rows and \
                 {tokens} tokens, and a batch above that bound must be admitted as smaller ones - \
                 no capture is made in the serving path"
            ))
        })
    }

    /// The smallest captured bucket that serves a media computation, if the set has one.
    ///
    /// A media computation is planned by what it is, not by a `Shape`: a denoising step is served
    /// only by a graph captured for its own latent layout and a decode only by one captured for its
    /// own clip, so the extent is matched exactly while rows (and an encode's tokens) are room to
    /// spare. No bucket serving the computation is the executor stating it does not serve it.
    pub fn select_media(&self, computation: &Computation) -> Option<Bucket> {
        let variant = computation.variant()?;
        self.ascending.iter().copied().find(|bucket| {
            bucket.variant == variant
                && bucket.extent == computation.extent
                && bucket.rows >= computation.rows
                && bucket.tokens >= computation.tokens
        })
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

    /// The executor states which media computations its plan serves, and the mechanism forms a
    /// batch only where the answer is a bucket: a denoising step is served only by a graph captured
    /// for its own latent layout, a decode only by one captured for its own clip, and a call no
    /// variant states - audio decoding, a mux - is served nowhere rather than by the nearest graph.
    #[test]
    fn a_media_computation_is_planned_only_into_a_bucket_captured_for_it() {
        let layout = nn::capability::Layout {
            video: nn::capability::LatentShape {
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
        let plan = Buckets::new([
            Bucket::denoise(1, layout),
            Bucket::clip(48, 768, 1344),
            Bucket::prefill(1, 512, Variant::Encode),
        ]);

        let denoise = Computation {
            call: ipc::MediaCall::Denoising,
            rows: 1,
            tokens: 0,
            extent: Extent::Latents(layout),
        };
        assert_eq!(denoise.variant(), Some(Variant::Denoise));
        assert_eq!(plan.select_media(&denoise), Some(Bucket::denoise(1, layout)));

        // One frame more is another graph, and no capture is made in the serving path.
        let other = Computation {
            extent: Extent::Latents(nn::capability::Layout {
                video: nn::capability::LatentShape {
                    channels: 16,
                    frames: 25,
                    height: 60,
                    width: 34,
                },
                ..layout
            }),
            ..denoise
        };
        assert_eq!(plan.select_media(&other), None);

        let clip = Computation {
            call: ipc::MediaCall::VideoDecoding,
            rows: 1,
            tokens: 0,
            extent: Extent::Clip {
                frames: 48,
                height: 768,
                width: 1344,
            },
        };
        assert_eq!(clip.variant(), Some(Variant::Decode));
        assert_eq!(plan.select_media(&clip), Some(Bucket::clip(48, 768, 1344)));

        // An encode is sized by rows and tokens like a prefill, so it is the one media computation
        // whose extent says nothing.
        let encode = Computation {
            call: ipc::MediaCall::TextEncoding,
            rows: 1,
            tokens: 300,
            extent: Extent::None,
        };
        assert_eq!(encode.variant(), Some(Variant::Encode));
        assert_eq!(
            plan.select_media(&encode),
            Some(Bucket::prefill(1, 512, Variant::Encode))
        );

        // The admission bound is the token path's: a media bucket's rows and tokens are not a
        // language-model batch size and must not move it.
        assert_eq!(plan.bounds(), (1, 512));

        // A call no variant states is served nowhere, and the mechanism reads that as the
        // capability it is.
        for call in [ipc::MediaCall::AudioDecoding, ipc::MediaCall::Muxing] {
            let computation = Computation {
                call,
                rows: 1,
                tokens: 0,
                extent: Extent::None,
            };
            assert!(!computation.stated(), "{call:?} states no variant");
            assert_eq!(plan.select_media(&computation), None);
        }
    }

    /// The two families are refused by name rather than served from one another's graphs: a media
    /// bucket states an extent a causal-LM forward has no use for, and a media variant is never a
    /// causal-LM variant - a catalogue that mixed them must fail where the batch is planned, not
    /// serve tokens shaped by a graph captured for a canvas.
    #[test]
    fn a_media_bucket_is_sized_by_its_extent_and_refused_by_a_causal_lm_runner() {
        let layout = nn::capability::Layout {
            video: nn::capability::LatentShape {
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
        decode.causal_lm().expect("a decode bucket is a causal-LM bucket");

        // Two layouts of one row count are two graphs, and the message states the canvas rather
        // than a token count nobody holds.
        let other = Bucket::denoise(
            1,
            nn::capability::Layout {
                video: nn::capability::LatentShape {
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
            "the display states the canvas: {other}"
        );
    }

    /// Every shape a scheduler can hand in is planned into the **smallest** captured bucket that
    /// holds it, derived independently by brute force over the captured set: a larger bucket still
    /// produces correct tokens and costs throughput, a smaller one would corrupt them.
    ///
    /// The shapes are the ones a batch hands in, which is what makes the catalog able to hold them
    /// at all: a decode forward's row count is its token count, and a prefill chunk holds at least
    /// as many tokens as rows. A shape outside both domains is the other test's subject - it is
    /// refused by name rather than planned, and asking this one to plan it would assert a coverage
    /// the catalog does not claim.
    #[test]
    fn every_shape_is_planned_into_the_smallest_bucket_that_holds_it() {
        let buckets = Buckets::causal_lm(&[1, 8, 32], &[128, 512], true);
        let captured: Vec<Bucket> = buckets.iter().copied().collect();
        assert!(!captured.is_empty());
        let mut planned = 0;
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
                            "{shape} has no holder and must not be planned"
                        );
                        continue;
                    };
                    planned += 1;
                    assert_eq!(
                        buckets.select(&shape),
                        Some(expected),
                        "{shape} must be planned into its smallest holder"
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
        assert!(planned > 0, "no shape was planned at all");
    }

    /// A shape the plan does not cover is refused by name - with the shape and its variant - and
    /// never clamped into a bucket that does not hold it or grown into one that was not captured.
    #[test]
    fn an_uncovered_shape_is_refused_by_name() {
        let buckets = Buckets::causal_lm(&[1, 8], &[128], false);
        // Beyond the largest captured token count of a prefill variant.
        let long = Shape::prefill(1, 129, Variant::PrefillOutput);
        let error = buckets.require(&long).expect_err("no bucket holds 129 tokens");
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
        // The refusal names the bound as well as the shape, so the shape a mechanism admitted and
        // the bound it exceeded are both in the one line a deployment's reader has.
        let (rows, tokens) = buckets.bounds();
        assert!(
            message.contains(&format!("{rows} rows")) && message.contains(&format!("{tokens} tokens")),
            "the refusal does not state this rank's bound: {message}"
        );
    }

    /// The bound the epoch plan states as a rank's batch capacity is this set's `bounds()`, so no
    /// batch the mechanism admits is larger than the largest warmed bucket - and a build config
    /// asking for more is clamped by the catalogue rather than honoured.
    ///
    /// This is the invariant that keeps a serving batch from reaching a bucket nobody captured:
    /// `require` would refuse it, and a refusal in the serving path ends the epoch, so the bound
    /// must never be stated above what is warmed.
    #[test]
    fn the_admission_bound_is_the_largest_warmed_bucket_and_never_the_build_configs() {
        // A deployment that asks for far more than the default catalogue holds.
        let buckets =
            Buckets::for_batch(None, None, 256, 1_000_000, true).expect("the default catalogue");
        let (rows, tokens) = buckets.bounds();
        assert!((rows, tokens) == (128, 16384), "the catalogue tops at {rows} rows, {tokens} tokens");

        // Every shape inside the bound is planned, at both edges and along each axis: the bound is
        // only useful if it is covered, and a catalogue with a hole inside its own bound would be
        // exactly the drift this test exists to catch.
        for variant in [
            Variant::DecodeGreedy,
            Variant::DecodeSampled,
            Variant::PrefillWrite,
            Variant::PrefillOutput,
        ] {
            if variant.is_decode() {
                for count in 1..=rows {
                    let shape = Shape::decode(count, variant);
                    assert!(buckets.select(&shape).is_some(), "{shape} is inside the bound");
                }
            } else {
                for count in 1..=tokens {
                    let shape = Shape::prefill(1, count, variant);
                    assert!(buckets.select(&shape).is_some(), "{shape} is inside the bound");
                }
                // The widest prefill the bound admits: every row count with the largest chunk.
                for row in 1..=rows {
                    let shape = Shape::prefill(row, tokens, variant);
                    assert!(buckets.select(&shape).is_some(), "{shape} is inside the bound");
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
            }));
        }
    }

    /// One capture per bucket, and a bucket per (size, computation) pair the plan admits: the
    /// first generation's capture set is the bucket set, so its size follows from the configured
    /// sizes and nothing else.
    ///
    /// The figure matters because the first generation has no serving generation to compare
    /// against - every operator of every bucket is new - so this count is the work it does before it
    /// serves a request, and one executable per capture is not booked by any pool.
    #[test]
    fn the_capture_set_is_the_bucket_set() {
        // The e2e deployment's own sizes: 4 decode sizes and one prefill size, sampling on.
        let buckets = Buckets::for_batch(Some(&[1, 2, 4, 8]), Some(&[512]), 256, 40960, true)
            .expect("the deployment's sizes");
        // 4 sizes x (greedy + sampled) decode, and 4 sizes x 1 token count x (write + output)
        // prefill: a prefill bucket is captured per row size, because rows bound the packing.
        assert_eq!(buckets.len(), 16, "16 buckets, so 16 captures");

        // The catalogues are the default when a deployment names no size, and the plan is their
        // cross product: 44 decode sizes and 110 prefill token counts give 44 x (2 + 220).
        let catalog = Buckets::for_batch(None, None, u32::MAX, u32::MAX, true)
            .expect("the catalogues");
        assert_eq!(catalog.len(), 44 * (2 + 110 * 2), "the catalog's cross product");
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

    /// The set an epoch captures implies the batch bound it admits: the largest decode rows and the
    /// largest prefill tokens, which is what the capacity plan and the scheduler read.
    #[test]
    fn the_bounds_are_what_the_set_captures() {
        let buckets = Buckets::default_causal_lm(Some(&[1, 8]), Some(&[0, 64, 256]), true);
        assert_eq!(buckets.bounds(), (8, 256));
        // A zero size names no bucket, and an epoch that configures no size takes the catalog.
        assert!(!buckets.contains(&Bucket::prefill(8, 0, Variant::PrefillWrite)));
        let catalog = Buckets::default_causal_lm(None, None, false);
        let (rows, tokens) = catalog.bounds();
        assert_eq!(rows, decode_catalog().into_iter().max().expect("a catalog"));
        assert_eq!(tokens, prefill_catalog().into_iter().max().expect("a catalog"));
    }
}
