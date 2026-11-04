//! The loading boundary of a deployment: which blocks a rank places and the precision choices of
//! the load.
//!
//! An H3 checkpoint is a diffusers pipeline directory whose components are its subdirectories, and
//! `nn::Checkpoint` learns such a root: `checkpoint.component("transformer")` gives one
//! component's shard set with each shard's own identity, and `WeightLoader::reborrow` gives one
//! loader per component over the same store and precision policy (specs/worker.md 8). So this
//! module names components and their directories; the shard sets are the checkpoint's, not the
//! selection's.
//!
//! ## Component names
//!
//! The names here are the deployment files' component names, which are also the block names the
//! executor addresses: `text_encoder`, `transformer`, `transformer_ref`, `video_decoder`,
//! `audio_decoder` and `latent_encoder`. The last owns both condition encoders, which live in the
//! two VAE directories.
//!
//! ## Precision
//!
//! With `dtype` and `quantization` unset, as the design's default `dtype = "auto"` requires, each
//! weight keeps the representation the checkpoint stores it in: the DiT and the text encoder are
//! BF16 and the video VAE decoder is FP16, and each layer declares the precision it computes in
//! (specs/worker.md 1, 9). An explicit choice converts every weight once, here, and never again.

use nn::DType;
use nn::weights::Quantization;

/// One block a deployment places, by the name its deployment file gives it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Component {
    /// The Qwen3-VL text encoder and its vision tower.
    TextEncoder,
    /// The base DiT partition: `t2va` and `fl2va`.
    Transformer,
    /// The reference-conditioned DiT partition: `ref2va`.
    TransformerRef,
    /// The video VAE decoder and its RGB reconstruction.
    VideoDecoder,
    /// The audio VAE decoder, BigVGAN.
    AudioDecoder,
    /// Both condition encoders, which `ref2va` uses.
    LatentEncoder,
}

impl Component {
    /// Every component a deployment may place.
    pub const ALL: [Component; 6] = [
        Component::TextEncoder,
        Component::Transformer,
        Component::TransformerRef,
        Component::VideoDecoder,
        Component::AudioDecoder,
        Component::LatentEncoder,
    ];

    /// The component's name in a deployment file and in the executor's block addressing.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TextEncoder => "text_encoder",
            Self::Transformer => "transformer",
            Self::TransformerRef => "transformer_ref",
            Self::VideoDecoder => "video_decoder",
            Self::AudioDecoder => "audio_decoder",
            Self::LatentEncoder => "latent_encoder",
        }
    }

    /// The checkpoint directories this component's weights come from.
    ///
    /// A VAE directory holds both halves of its VAE, so a decoder and the encoder beside it read
    /// the same directory and each loads its own half.
    pub const fn directories(self) -> &'static [&'static str] {
        match self {
            Self::TextEncoder => &["text_encoder"],
            Self::Transformer => &["transformer"],
            Self::TransformerRef => &["transformer_ref"],
            Self::VideoDecoder => &["vae"],
            Self::AudioDecoder => &["audio_vae"],
            Self::LatentEncoder => &["vae", "audio_vae"],
        }
    }

    /// The component a deployment file names, when it names one of these.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|component| component.name() == name)
    }
}

/// The choices a load takes besides the checkpoint and the partition.
pub struct LoadOptions {
    /// The components this rank places, in deployment order.
    ///
    /// A base release holds both DiT partitions and a deployment places one; a placement that
    /// names a component the checkpoint does not hold is refused rather than skipped, so a
    /// misspelt deployment fails at load.
    pub blocks: Vec<Component>,
    /// An explicit element type for every weight, or `None` to keep the checkpoint's own.
    pub dtype: Option<DType>,
    /// An explicit quantization, or `None` to keep the checkpoint's own representation.
    pub quantization: Option<Quantization>,
}

impl LoadOptions {
    /// The options of a deployment that places `blocks` and converts nothing.
    pub fn new(blocks: Vec<Component>) -> Self {
        Self {
            blocks,
            dtype: None,
            quantization: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_component_is_named_as_its_deployment_file_names_it() {
        for component in Component::ALL {
            assert_eq!(Component::named(component.name()), Some(component));
            assert!(!component.directories().is_empty());
        }
        assert_eq!(Component::named("video_encoder"), None);
    }

    #[test]
    fn a_shared_vae_directory_is_read_by_both_halves() {
        assert_eq!(Component::VideoDecoder.directories(), ["vae"]);
        assert_eq!(Component::AudioDecoder.directories(), ["audio_vae"]);
        assert_eq!(Component::LatentEncoder.directories(), ["vae", "audio_vae"]);
        assert_eq!(Component::TransformerRef.directories(), ["transformer_ref"]);
    }
}
