//! The stages of a MiniMax-H3 pipeline a deployment places, by the names its deployment file
//! places them under and execution reports them by: the blocks the family's pipeline declares
//! ([`crate::minimax_h3::family::pipeline`]).
//!
//! The names follow the pipeline index's components where a stage loads one (`text_encoder`,
//! `transformer`, `transformer_ref`); the decoders, the condition encoders and the clip splice
//! are stages of their own over the VAEs' components.

/// One stage of the pipeline.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Stage {
    /// The text encoder and its vision tower.
    TextEncoder,
    /// The base denoiser network, of `t2va` and `fl2va`.
    Transformer,
    /// The reference-conditioned denoiser network, of `ref2va`.
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

/// A set of the pipeline's stages: the stages one rank places.
///
/// A deployment places a handful of stages out of seven, so the set is a bit per stage and
/// `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Stages(u8);

impl Stages {
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
