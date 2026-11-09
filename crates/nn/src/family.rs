//! Model families: what a concrete model supplies to execution.
//!
//! A model crate exports one [`Family`] value, and the executable registers it; execution resolves a
//! checkpoint to the family that claims its declared class and from then on dispatches by
//! capability alone. The two kinds of family differ in what they compute:
//!
//! - a [`TextFamily`] loads a causal language model ([`TextModel`]), which execution serves with
//!   prefill and decode forwards over a paged key/value cache;
//! - a [`VideoFamily`] loads a video generation pipeline ([`VideoModel`]): a text encoder, a
//!   denoiser, decoders and the media format of its conditions, which execution serves as the
//!   calls of one diffusion request.
//!
//! The contract is numerical. A family reads its checkpoint and states its mathematical facts
//! (widths, cache shape, frame rule, the requests its graphs are captured for); it receives the
//! deployment's mathematical partition ([`Split`], [`SequenceSplit`]) and the deployment's serving
//! bounds ([`VideoServing`]), and it holds no runtime object: execution owns the devices, streams,
//! memory, graphs and scheduling.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::capability::{CausalLm, Decoder, Denoiser, Encoder, MediaFormat, WarmRequest};
use crate::{Checkpoint, DType, Result, WeightLoader};

/// One model family a build serves.
#[derive(Clone, Copy)]
pub enum Family {
    /// A causal language model family.
    Text(&'static dyn TextFamily),
    /// A video generation pipeline family.
    Video(&'static dyn VideoFamily),
}

impl Family {
    /// The model classes a checkpoint of this family declares: a transformers checkpoint's
    /// `config.json` `architectures[0]`, or a diffusers pipeline's `_class_name`.
    pub fn classes(&self) -> &'static [&'static str] {
        match self {
            Self::Text(family) => family.classes(),
            Self::Video(family) => family.classes(),
        }
    }
}

impl std::fmt::Debug for Family {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Self::Text(_) => "Text",
            Self::Video(_) => "Video",
        };
        write!(formatter, "Family::{kind}({:?})", self.classes())
    }
}

/// One rank's place in a split of `size` ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// Ranks the split spans; one is no split.
    pub size: u32,
    /// This rank's position in it, below `size`.
    pub rank: u32,
}

impl Split {
    /// No split: one rank holds the whole.
    pub const WHOLE: Self = Self { size: 1, rank: 0 };
}

/// A causal language model family.
pub trait TextFamily: Send + Sync {
    /// The `architectures` classes this family loads.
    fn classes(&self) -> &'static [&'static str];

    /// Reads the checkpoint at `checkpoint`'s configuration.
    ///
    /// # Errors
    ///
    /// Fails when the configuration cannot be read or states a model this family cannot build.
    fn open(&self, checkpoint: &Path) -> Result<Box<dyn TextCheckpoint>>;
}

/// A causal language model checkpoint, read but not loaded.
pub trait TextCheckpoint: Send + Sync {
    /// The routed expert layers by module path, each with its expert count; empty for a dense
    /// model. An expert redistribution orders these layers' experts ([`ExpertOrder`]).
    fn expert_layers(&self) -> BTreeMap<String, u32>;

    /// The special token whose text ends a document in this family's vocabulary, such as
    /// `<|endoftext|>`, when the family has one: execution reads its id from the checkpoint's
    /// tokenizer, begins sequences with it when the checkpoint states no beginning token, and
    /// stops generation on it.
    fn end_of_text(&self) -> Option<&'static str>;

    /// Loads the model's share of the checkpoint for `placement`, its weights reached through
    /// `loader`, its routed experts in the slots `experts` states.
    ///
    /// # Errors
    ///
    /// Fails when the placement does not divide the model or a weight cannot be loaded.
    fn load(
        &self,
        loader: &mut WeightLoader<'_>,
        placement: &TextPlacement,
        experts: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>>;
}

/// The expert slots of each routed layer, by module path: entry `i` is the checkpoint expert held
/// in slot `i`. A layer it does not name keeps the checkpoint's order.
pub type ExpertOrder = BTreeMap<String, Vec<u32>>;

/// How a deployment splits a causal language model, and this rank's place in the split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextPlacement {
    /// The tensor-parallel split of every layer.
    pub tensor: Split,
    /// The expert-parallel split of the routed experts.
    pub experts: Split,
    /// The expert exchange of a deployment whose experts are split; `None` without one.
    pub exchange: Option<Exchange>,
}

/// The exchange that carries a step's rows between the members of an expert split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exchange {
    /// Rows one rank's exchange carries at most: the load-time bound its step buffers are shaped
    /// by.
    pub capacity: u64,
    /// Leading members of the exchange that hold no experts: the attention ranks of a deployment
    /// whose experts live on dedicated ranks.
    pub attention_ranks: u32,
}

/// A loaded causal language model, as execution plans and serves it.
pub trait TextModel: CausalLm + Send + Sync {
    /// The width of a hidden row.
    fn hidden(&self) -> u32;

    /// The vocabulary a logit row spans.
    fn vocabulary(&self) -> u32;

    /// The element type of its activations.
    fn dtype(&self) -> DType;

    /// The key/value cache its layers read, as this rank holds it.
    fn cache(&self) -> CacheShape;

    /// Its routed experts, when it has any.
    fn experts(&self) -> Option<ExpertShape>;
}

/// The paged key/value cache a causal language model reads, as one rank holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheShape {
    /// Tokens one page holds.
    pub page_tokens: u32,
    /// Layers that store keys and values.
    pub layers: u32,
    /// Key/value heads each layer stores on this rank.
    pub kv_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
}

/// The routed experts of a causal language model, as an expert exchange carries them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertShape {
    /// Routed experts of each layer.
    pub experts: u32,
    /// The width of one expert's intermediate projection.
    pub intermediate: u32,
    /// Experts each row is routed to.
    pub top: u32,
    /// Whether the experts are read packed, so a step encodes its activations before the exchange.
    pub encoded: bool,
}

/// A video generation pipeline family.
pub trait VideoFamily: Send + Sync {
    /// The diffusers pipeline classes this family loads.
    fn classes(&self) -> &'static [&'static str];

    /// Reads the checkpoint whose directories `roots` hold, first root first: a checkpoint and
    /// the base release it pins, when it pins one.
    ///
    /// # Errors
    ///
    /// Fails when the configuration cannot be read or states a pipeline this family cannot build.
    fn open(&self, roots: &[PathBuf]) -> Result<Box<dyn VideoCheckpoint>>;
}

/// A video generation checkpoint, read but not loaded.
pub trait VideoCheckpoint: Send + Sync {
    /// The deployment components that hold a denoiser network, in the names a deployment file
    /// places them under; a checkpoint may hold several networks and a deployment places one.
    fn denoisers(&self) -> &'static [&'static str];

    /// The deployment component that holds the text encoder.
    fn text_encoder(&self) -> &'static str;

    /// Loads the pipeline's share of `checkpoint`, the checkpoint opened over the roots
    /// [`VideoFamily::open`] read, for `placement`, its weights reached through `loader`.
    ///
    /// # Errors
    ///
    /// Fails when the placement does not divide the pipeline, places no network the checkpoint
    /// holds, or a weight cannot be loaded.
    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &VideoPlacement,
    ) -> Result<Arc<dyn VideoModel>>;
}

/// How a deployment splits a video pipeline and sizes it, and this rank's place in the split.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoPlacement {
    /// The components the deployment places on this rank's group; `None` places the whole
    /// pipeline.
    pub components: Option<BTreeSet<String>>,
    /// The denoiser's sequence split.
    pub sequence: SequenceSplit,
    /// The text encoder's tensor split.
    pub tensor: Split,
    /// The element type the deployment computes in.
    pub dtype: DType,
    /// What the deployment serves.
    pub serving: VideoServing,
}

/// What a video deployment serves: the bounds its requests are held to and its graphs are
/// captured for.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoServing {
    /// The output rasters it serves, `(width, height)` in pixels.
    pub sizes: Vec<(u32, u32)>,
    /// The longest output it serves, in seconds.
    pub max_seconds: f64,
    /// The prompt capacities it captures, strictly increasing: a request rides the smallest that
    /// holds its prompt.
    pub text_capacities: Vec<u32>,
    /// The denoiser rows one request's conditions may take; the model's whole condition capacity
    /// when `None`.
    pub condition_rows: Option<u32>,
}

/// A sequence split: the ranks a denoiser's sequence is exchanged over, and this rank's place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceSplit {
    /// Ranks the sequence is split over, and this rank's position.
    pub split: Split,
    /// How a member's merged head projections reach it.
    pub projection: Projection,
}

impl SequenceSplit {
    /// No split: the sequence is this rank's own.
    pub const WHOLE: Self = Self {
        split: Split::WHOLE,
        projection: Projection::Exchange,
    };
}

/// How the members of a sequence split reach their merged head projections, which decides the
/// weights a member holds and the communication its attention runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// Every member gathers every member's rows and projects its own head shard with
    /// head-partitioned weights: one all-gather of the normalized rows per layer.
    Gather,
    /// Every member projects its own rows for all heads with whole weights and sends each member
    /// its block: one all-to-all per projection.
    Exchange,
}

/// A loaded video generation pipeline, as execution plans and serves it.
///
/// Each block is absent when this rank does not place it. Execution binds one runner per block
/// present and states to its scheduler the calls those runners serve.
pub trait VideoModel: Send + Sync {
    /// The text encoder: a prompt, and the vision content it presents, to conditioning rows.
    fn text_encoder(&self) -> Option<Arc<dyn Encoder + Send + Sync>>;

    /// The width of the conditioning rows the text encoder writes.
    fn text_width(&self) -> u32;

    /// The denoiser network this rank places.
    fn denoiser(&self) -> Option<Arc<dyn Denoiser + Send + Sync>>;

    /// The deployment component the placed denoiser is, one of
    /// [`VideoCheckpoint::denoisers`].
    fn denoiser_component(&self) -> &'static str;

    /// The element type the denoiser's trajectory and noise are kept in.
    fn sample_type(&self) -> DType;

    /// The video decoder: latent frames to a clip of pixels.
    fn video_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>>;

    /// The video decoder's second entry point, which splices decoded clips into one timeline;
    /// `None` publishes each decoded clip as it is.
    fn splicer(&self) -> Option<Arc<dyn Decoder + Send + Sync>>;

    /// The audio decoder: latent frames to a soundtrack; `None` for a pipeline without sound.
    fn audio_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>>;

    /// The soundtrack's sample rate, in samples per second.
    fn audio_rate(&self) -> u32;

    /// The output's frame rate, in frames per second.
    fn frame_rate(&self) -> u32;

    /// The frames one decode call produces: the video decoder's window.
    fn decode_window(&self) -> u32;

    /// How the pipeline reads a request's condition media.
    ///
    /// # Errors
    ///
    /// Fails when the format cannot be built for the deployment's element type.
    fn format(&self) -> Result<Arc<dyn MediaFormat>>;

    /// The frames of the least legal timeline that lasts at least `seconds`.
    ///
    /// # Errors
    ///
    /// Fails when `seconds` is no positive duration or shorter than the shortest timeline.
    fn timeline(&self, seconds: f64) -> Result<u32>;

    /// The frames of each decode unit of a `frames`-frame timeline, in order.
    ///
    /// # Errors
    ///
    /// Fails when `frames` is no legal timeline.
    fn video_units(&self, frames: u32) -> Result<Vec<u32>>;

    /// The extents one request's conditions may reach on this rank.
    fn condition_bounds(&self) -> ConditionBounds;

    /// The requests a deployment serving `serving` captures its denoiser's graphs for: one per
    /// distinct layout a request of it may take.
    ///
    /// # Errors
    ///
    /// Fails when `serving` states a size, duration or capacity the pipeline does not serve.
    fn candidates(&self, serving: &VideoServing) -> Result<Vec<WarmRequest>>;

    /// What the placed denoiser serves under `serving`, as the deployment's request processing
    /// plans requests against it.
    ///
    /// # Errors
    ///
    /// Fails when `serving` states a size the pipeline does not serve.
    fn description(&self, serving: &VideoServing) -> Result<Description>;
}

/// The extents one request's conditions may reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConditionBounds {
    /// Denoiser rows its conditions may take; zero when this rank encodes no condition.
    pub rows: u32,
    /// Decoded pixels of its condition media.
    pub pixels: u32,
    /// Decoded samples of its condition tracks.
    pub samples: u32,
    /// Vision patches of one prompt token that presents vision content.
    pub patches_per_token: u32,
}

/// What a placed denoiser serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Description {
    /// Whether its requests may carry conditions.
    pub takes_conditions: bool,
    /// The denoiser evaluations of every request's fixed schedule.
    pub denoise_steps: u32,
    /// The rest, as the family's own request processing reads it; execution reads none of it.
    pub details: Vec<u8>,
}
