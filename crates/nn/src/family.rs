//! Model families: what a concrete model supplies to execution and to request processing.
//!
//! A model crate exports one [`Family`] value, and the executable registers it once: execution
//! resolves a checkpoint to the family that claims its declared class ([`Registry`]) and from
//! then on dispatches by capability alone ([`Family::model`]); request processing reads only the
//! family's request side ([`Family::request`], a [`processing::Family`]) from a registry built
//! of the same list, and links no numerical crate. The two kinds of model differ in what they
//! compute:
//!
//! - a [`TextFamily`] loads a causal language model ([`TextModel`]), which execution serves with
//!   prefill and decode forwards over a paged key/value cache;
//! - a [`VideoFamily`] loads a video generation pipeline ([`VideoModel`]): a text encoder, a
//!   denoiser, decoders and the media format of its conditions, which execution serves as the
//!   calls of one diffusion request.
//!
//! The interface is numerical. A family reads its checkpoint and states its mathematical facts
//! (widths, cache shape, frame rule, the requests its graphs are captured for); it receives the
//! deployment's mathematical partition ([`Split`], [`SequenceSplit`]) and the deployment's
//! serving bounds ([`VideoServing`]), and it holds no runtime object: execution owns the devices,
//! streams, memory, graphs and scheduling.
//!
//! ## One shape for every family
//!
//! Execution opens a [`Checkpoint`] and gives it to the family: `open` reads and normalizes the
//! configuration once, from the checkpoint's own sidecars, into the opened checkpoint value that
//! holds it; `load` builds the deployment's share from that configuration, the same checkpoint
//! and a loader, and takes no path. Nothing between the two reads a configuration file again.
//! The loader decides what a load is: over a store of a device it loads the model execution
//! serves, over [`CpuWeights`](crate::CpuWeights) the model a host check computes, and a meta
//! loader the model of a profile run, by the same `load`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use uniserve_tensor::DType;

use crate::capability::{
    CausalLm, ConditionShape, Decoder, Denoiser, Drafter, Encoder, MediaFormat, Splicer,
    VideoRequest, Vision,
};
use crate::collective::Split;
use crate::{Checkpoint, Result, WeightLoader};

/// One model family a build serves: what execution loads of its checkpoints, beside how they
/// declare it and what request processing serves ([`processing::Family`]).
#[derive(Clone, Copy)]
pub struct Family {
    /// How the family's checkpoints declare it and what request processing serves of it.
    pub request: processing::Family,
    /// What execution loads of a checkpoint.
    pub model: Model,
}

impl Family {
    /// The family's stable identifier, as the daemon reports it.
    pub fn id(&self) -> &'static str {
        self.request.id
    }
}

/// The model a family's checkpoints load to.
#[derive(Clone, Copy)]
pub enum Model {
    /// A causal language model.
    Text(&'static dyn TextFamily),
    /// A video generation pipeline.
    Video(&'static dyn VideoFamily),
}

impl std::fmt::Debug for Family {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.request.fmt(formatter)
    }
}

impl PartialEq for Family {
    /// Families are one value each: equal when they are the same family.
    fn eq(&self, other: &Self) -> bool {
        self.request == other.request
    }
}

impl Eq for Family {}

/// The families an executable loads, each claiming the `model_type`s and classes its
/// checkpoints declare.
///
/// The claims are request processing's registry ([`processing::Registry`]) of the same families,
/// so both sides resolve a checkpoint by one rule.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    families: Vec<Family>,
    claims: processing::Registry,
}

impl Registry {
    /// A registry of `families`.
    ///
    /// # Errors
    ///
    /// Fails, naming the class, when two families claim one `model_type` or class: a checkpoint
    /// of it would resolve to either.
    pub fn new(families: impl IntoIterator<Item = Family>) -> Result<Self> {
        let families: Vec<Family> = families.into_iter().collect();
        let claims = processing::Registry::new(families.iter().map(|family| family.request))
            .map_err(|error| crate::Error::Unsupported(error.to_string()))?;
        Ok(Self { families, claims })
    }

    /// The family a checkpoint declares by its class: a transformers checkpoint's
    /// `architectures[0]`, or a diffusers pipeline's `_class_name`.
    pub fn by_class(&self, class: &str) -> Option<Family> {
        let claimed = self.claims.by_class(class)?;
        self.families
            .iter()
            .copied()
            .find(|family| family.request == claimed)
    }

    /// Every family, in registration order.
    pub fn families(&self) -> &[Family] {
        &self.families
    }
}

/// One stage of a video pipeline, by the name a deployment file places it under and execution
/// reports it by.
///
/// The names are `model_index.json`'s, shared by every video family; which checkpoint
/// directories a stage reads is the family's own mapping.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Stage {
    /// The text encoder and its vision tower.
    TextEncoder,
    /// The base denoiser network.
    Transformer,
    /// The reference-conditioned denoiser network.
    TransformerRef,
    /// The video VAE decoder.
    VideoDecoder,
    /// The audio VAE decoder.
    AudioDecoder,
    /// The condition encoders a conditioned task reads its media with.
    LatentEncoder,
    /// The video postprocessor: the clip splice, which holds no weights.
    VideoPostprocessor,
}

impl Stage {
    /// Every stage a deployment may place, in deployment order.
    pub const ALL: [Stage; 7] = [
        Stage::TextEncoder,
        Stage::Transformer,
        Stage::TransformerRef,
        Stage::VideoDecoder,
        Stage::AudioDecoder,
        Stage::LatentEncoder,
        Stage::VideoPostprocessor,
    ];

    /// This stage's bit in [`Stages`].
    const fn bit(self) -> u8 {
        match self {
            Self::TextEncoder => 0,
            Self::Transformer => 1,
            Self::TransformerRef => 2,
            Self::VideoDecoder => 3,
            Self::AudioDecoder => 4,
            Self::LatentEncoder => 5,
            Self::VideoPostprocessor => 6,
        }
    }

    /// The stage's name in a deployment file and in what execution reports.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TextEncoder => "text_encoder",
            Self::Transformer => "transformer",
            Self::TransformerRef => "transformer_ref",
            Self::VideoDecoder => "video_decoder",
            Self::AudioDecoder => "audio_decoder",
            Self::LatentEncoder => "latent_encoder",
            Self::VideoPostprocessor => "video_postprocessor",
        }
    }
}

/// A set of pipeline stages: the stages one rank places.
///
/// A deployment places a handful of stages out of seven, so the set is a bit per stage and
/// `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Stages(u8);

impl Stages {
    /// Every stage a deployment may place.
    pub const ALL: Stages = Stages((1 << Stage::ALL.len()) - 1);

    /// No stage.
    pub const NONE: Stages = Stages(0);

    /// The set of `stages`.
    pub fn of(stages: impl IntoIterator<Item = Stage>) -> Self {
        stages.into_iter().fold(Self::NONE, Self::with)
    }

    /// The set with `stage` added.
    pub const fn with(self, stage: Stage) -> Self {
        Self(self.0 | 1 << stage.bit())
    }

    /// The set without `stage`.
    pub const fn without(self, stage: Stage) -> Self {
        Self(self.0 & !(1 << stage.bit()))
    }

    /// Whether `stage` is placed.
    pub const fn contains(self, stage: Stage) -> bool {
        self.0 & (1 << stage.bit()) != 0
    }

    /// The stages placed, in deployment order.
    pub fn iter(self) -> impl Iterator<Item = Stage> {
        Stage::ALL
            .into_iter()
            .filter(move |stage| self.contains(*stage))
    }
}

impl FromIterator<Stage> for Stages {
    fn from_iter<I: IntoIterator<Item = Stage>>(stages: I) -> Self {
        Self::of(stages)
    }
}

/// A causal language model family.
pub trait TextFamily: Send + Sync {
    /// Reads and normalizes `checkpoint`'s configuration, once, from its sidecars.
    ///
    /// # Errors
    ///
    /// Fails when the configuration cannot be read or states a model this family cannot build.
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn TextCheckpoint>>;
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

    /// Loads the model's share of `checkpoint`, the checkpoint [`TextFamily::open`] read, for
    /// `placement`, its weights reached through `loader` (whose root builder the model's modules
    /// load from), its routed experts in the slots `experts` states.
    ///
    /// # Errors
    ///
    /// Fails when the placement does not divide the model or a weight cannot be loaded.
    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &TextPlacement,
        experts: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>>;
}

/// The expert slots of each routed layer, by module path: entry `i` is the checkpoint expert
/// held in slot `i`. A layer it does not name keeps the checkpoint's order.
pub type ExpertOrder = BTreeMap<String, Vec<u32>>;

/// How a deployment splits a causal language model, and this rank's place in the split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextPlacement {
    /// The tensor-parallel split of every layer.
    pub tensor: Split,
    /// The expert-parallel split of the routed experts.
    pub experts: Split,
    /// Whether the deployment speculates with the checkpoint's own drafter, so its layers load
    /// beside the model's ([`TextModel::drafter`]); a deployment that does not leaves them unread.
    pub drafter: bool,
    /// The tokens a step of the deployment drafts beyond the one it commits, zero for a
    /// deployment that does not speculate: a model whose state fields hold a window per request
    /// ([`crate::capability::VerifySlots`]) sizes them by it.
    pub draft_tokens: u32,
}

/// A loaded causal language model, as execution sizes and serves it.
pub trait TextModel: CausalLm + Send + Sync {
    /// The width of a hidden row.
    fn hidden(&self) -> u32;

    /// The vocabulary a logit row spans.
    fn vocabulary(&self) -> u32;

    /// The element type of its activations.
    fn dtype(&self) -> DType;

    /// The key/value cache its layers read, as this rank holds it.
    fn cache(&self) -> CacheShape;

    /// The recurrent state its layers keep per request slot, as this rank holds it; `None` for a
    /// model whose every layer attends over the key/value cache.
    fn state(&self) -> Option<StateShape>;

    /// The recurrent state of a model whose layers do not all keep the same fields: one entry
    /// per run of layers that do, in the order [`crate::capability::RecurrentState::live`]
    /// holds their buffers (every layer of the first entry, then of the second). The one entry
    /// of [`Self::state`] for every other model.
    fn state_layers(&self) -> Vec<StateShape> {
        self.state().into_iter().collect()
    }

    /// The width of the rows [`CausalLm::forward`] returns, which [`CausalLm::compute_logits`]
    /// and the drafter read: [`Self::hidden`], or the residual streams' joint width for a model
    /// that carries several and reads them out at its head.
    fn stream_width(&self) -> u32 {
        self.hidden()
    }

    /// Its routed experts, when it has any.
    fn experts(&self) -> Option<ExpertShape>;

    /// The drafter its checkpoint carries for speculative decoding (multi-token prediction
    /// layers), when the model loaded one; `None` for a model that drafts nothing.
    fn drafter(&self) -> Option<&dyn Drafter>;

    /// Its vision tower, when the checkpoint holds one and the model serves it: the model then
    /// takes image rows in a prefill ([`crate::capability::TokenBatch::rope_positions`]).
    fn vision(&self) -> Option<&dyn Vision> {
        None
    }
}

/// The recurrent state a causal language model keeps per request slot, as one rank holds it:
/// every recurrent layer keeps the same fields, each a `[slots, …]` buffer execution allocates
/// ([`crate::capability::RecurrentState`]).
///
/// A layer's state after token `t` summarizes every token up to `t`, so a request that resumes
/// from a cached prefix resumes from the state its prefix's last token left; the prefix cache
/// keeps such checkpoints of every field
/// ([`crate::capability::RecurrentState::checkpoints`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateShape {
    /// Layers that keep a recurrent state.
    pub layers: u32,
    /// The fields one layer keeps, in the order its buffers are lent to it.
    pub fields: Vec<StateField>,
}

/// One field of a recurrent layer's state: its element type and its extents after the slot axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateField {
    /// What the field holds, for diagnostics and the cache's field names.
    pub name: &'static str,
    /// The element type of its buffer.
    pub dtype: DType,
    /// Its extents after the slot axis.
    pub dims: Vec<usize>,
}

impl StateField {
    /// The bytes one slot of the field occupies.
    pub fn bytes(&self) -> u64 {
        let elements: usize = self.dims.iter().product();
        elements as u64 * ops::abi::bytes(self.dtype).unwrap_or(0)
    }
}

impl StateShape {
    /// The bytes one slot of one layer's fields occupies.
    pub fn layer_bytes(&self) -> u64 {
        self.fields.iter().map(StateField::bytes).sum()
    }
}

/// The paged key/value cache a causal language model reads, as one rank holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheShape {
    /// Layers that store keys and values.
    pub layers: u32,
    /// Key/value heads each layer stores on this rank.
    pub kv_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// The index keys each layer stores beside its keys and values
    /// ([`crate::capability::KvPlanes::index_keys`]), for a model whose attention reads the
    /// tokens an indexer selects; `None` otherwise.
    pub index_keys: Option<IndexKeys>,
}

/// The index-key plane of an attention layer: one key per `ratio` consecutive tokens of a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexKeys {
    /// Elements of one index key.
    pub width: u32,
    /// Tokens one key stands for; it divides the page size.
    pub ratio: u32,
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
    /// Reads and normalizes `checkpoint`'s configuration, once, from its sidecars.
    ///
    /// `checkpoint` may span several roots, an export first and the base release it continues
    /// after it ([`Checkpoint::open_roots`]); each sidecar is read from the first root that holds
    /// it.
    ///
    /// # Errors
    ///
    /// Fails when the configuration cannot be read or states a pipeline this family cannot build.
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn VideoCheckpoint>>;

    /// The stage of `checkpoint` whose tensors `adapter` edits, or `None` for a flat
    /// checkpoint's own shard set: where the loading boundary fuses it
    /// ([`Checkpoint::attach`]) before [`VideoFamily::open`] reads the adapted checkpoint.
    ///
    /// Recognizing an adapter is the family's checkpoint mapping: a pipeline may hold several
    /// networks of one architecture, and only the family knows which one an adapter was trained
    /// from.
    ///
    /// # Errors
    ///
    /// Fails when the family does not recognize `adapter` as an adapter of its checkpoints,
    /// which is every adapter for a family that serves none.
    fn adapter_stage(
        &self,
        checkpoint: &Checkpoint,
        adapter: &crate::Adapter,
    ) -> Result<Option<&'static str>> {
        let _ = checkpoint;
        Err(crate::Error::Unsupported(format!(
            "{} is not an adapter of a checkpoint this model family serves",
            adapter.path().display()
        )))
    }
}

/// A video generation checkpoint, read but not loaded.
pub trait VideoCheckpoint: Send + Sync {
    /// The deployment stages that hold a denoiser network, in the names a deployment file
    /// places them under; a checkpoint may hold several networks and a deployment places one.
    fn denoisers(&self) -> &'static [&'static str];

    /// The deployment stage that holds the text encoder.
    fn text_encoder(&self) -> &'static str;

    /// Loads the pipeline's share of `checkpoint`, the checkpoint [`VideoFamily::open`] read, for
    /// `placement`, its weights reached through `loader`.
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
    /// The stages the deployment places on this rank's group; `None` places the whole
    /// pipeline.
    pub stages: Option<BTreeSet<String>>,
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
    /// The output resolutions it serves, `(width, height)` in pixels.
    pub sizes: Vec<(u32, u32)>,
    /// The longest output it serves, in seconds.
    pub max_seconds: f64,
    /// The prompt capacities it captures, strictly increasing, the largest the longest prompt it
    /// admits (`max_model_len`): a request rides the smallest that holds its prompt.
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

/// A loaded video generation pipeline, as execution sizes and serves it.
///
/// Each stage is absent when this rank does not place it. Execution binds one runner per stage
/// present and states to its scheduler the calls those runners serve.
pub trait VideoModel: Send + Sync {
    /// The text encoder: a prompt, and the vision content it presents, to conditioning rows.
    fn text_encoder(&self) -> Option<Arc<dyn Encoder + Send + Sync>>;

    /// The width of the conditioning rows the text encoder writes.
    fn text_width(&self) -> u32;

    /// The denoiser network this rank places.
    fn denoiser(&self) -> Option<Arc<dyn Denoiser + Send + Sync>>;

    /// The deployment stage the placed denoiser is, one of
    /// [`VideoCheckpoint::denoisers`].
    fn denoiser_stage(&self) -> &'static str;

    /// The element type the denoiser's trajectory and noise are kept in.
    fn sample_type(&self) -> DType;

    /// The video decoder: latent frames to a clip of pixels.
    fn video_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>>;

    /// The step that splices the video decoder's clips into one timeline; `None` publishes each
    /// decoded clip as it is.
    fn splicer(&self) -> Option<Arc<dyn Splicer + Send + Sync>>;

    /// The audio decoder: latent frames to a soundtrack; `None` for a pipeline without sound.
    fn audio_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>>;

    /// The soundtrack's sample rate, in samples per second.
    fn audio_rate(&self) -> u32;

    /// The output's frame rate, in frames per second.
    fn frame_rate(&self) -> u32;

    /// The frames one decode call produces: the video decoder's window.
    fn decode_window(&self) -> u32;

    /// How the pipeline's encoders read a request's condition media, which execution lays out.
    ///
    /// # Errors
    ///
    /// Fails when the configuration states no format for the deployment's element type.
    fn format(&self) -> Result<MediaFormat>;

    /// The frames of the least legal timeline that lasts at least `seconds`; a duration shorter
    /// than the shortest timeline lasts the shortest.
    ///
    /// # Errors
    ///
    /// Fails when `seconds` is no positive duration.
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
    /// distinct layout a request of it may take, each with the condition sets a request of the
    /// layout may open with.
    ///
    /// A capture happens before the first request of a shape, so a model states one request per
    /// layout its deployment serves and the bucket is derived from it
    /// ([`Denoiser::layout`]). The second value lists the condition sets, each in request order
    /// as the request presents them: a request's opening encodes its own conditions eagerly,
    /// outside any captured graph, so the profile run opens the layout once per set to measure
    /// that encode and state its kernels. They are not the request's own conditions, which it
    /// states none of; the list is empty for a layout whose requests state no condition, or
    /// whose conditions follow the client's media and nothing the deployment serves.
    ///
    /// # Errors
    ///
    /// Fails when `serving` states a size, duration or capacity the pipeline does not serve.
    fn candidates(
        &self,
        serving: &VideoServing,
    ) -> Result<Vec<(VideoRequest, Vec<Vec<ConditionShape>>)>>;

    /// The prompts with vision blocks a deployment serving `serving` admits whose grids the
    /// deployment fixes: one per distinct set of grids, each as a request of a served resolution
    /// presents it, as the patch grids (`[time, rows, columns]`) of its vision blocks in
    /// presentation order and the vision placeholder tokens those blocks present.
    ///
    /// A prompt that presents vision blocks is encoded eagerly over its own layout and not by a
    /// captured graph, in the same stream region the captured encodes take their memory from,
    /// so the text encoder's profile run encodes each beside the captured encodes and the
    /// region's peak holds the eager vision encode as well. A prompt holding the blocks is at
    /// least their tokens long, so it rides only an encode bucket of at least that many tokens.
    ///
    /// Empty for a deployment whose tasks present no vision block.
    ///
    /// # Errors
    ///
    /// Fails when `serving` states a size the pipeline does not serve, or when a task presents
    /// vision blocks and the checkpoint states no rule to size them.
    fn warm_prompts(&self, serving: &VideoServing) -> Result<Vec<(Vec<[u32; 3]>, u32)>>;

    /// What the placed denoiser serves under `serving`, as the deployment's request processing
    /// admits requests against it.
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

#[cfg(test)]
mod tests {
    use processing::{ChatProfile, Serving};

    use super::*;

    /// A model that opens nothing: the registry reads only a family's claims.
    struct Unopened;

    impl TextFamily for Unopened {
        fn open(&self, _: &Checkpoint) -> Result<Box<dyn TextCheckpoint>> {
            Err(crate::Error::Unsupported("no checkpoint".to_owned()))
        }
    }

    const CHAT: Family = Family {
        request: processing::Family {
            id: "chat",
            model_types: &["chat_model"],
            classes: &["ChatForCausalLM"],
            serving: Serving::Chat(ChatProfile {
                reasoning: None,
                tool_calls: None,
                images: None,
            }),
        },
        model: Model::Text(&Unopened),
    };

    /// A checkpoint resolves to the numerical family whose request side claims its class.
    #[test]
    fn a_family_is_found_by_its_declared_class() {
        let registry = Registry::new([CHAT]).unwrap();
        assert_eq!(registry.by_class("ChatForCausalLM"), Some(CHAT));
        assert_eq!(registry.by_class("chat_model"), None);
    }

    /// Two families that claim one class are refused, as request processing refuses them.
    #[test]
    fn two_families_may_not_claim_one_class() {
        let second = Family {
            request: processing::Family {
                id: "second",
                ..CHAT.request
            },
            ..CHAT
        };
        let error = Registry::new([CHAT, second]).unwrap_err();
        assert!(error.to_string().contains("chat_model"), "{error}");
    }
}
