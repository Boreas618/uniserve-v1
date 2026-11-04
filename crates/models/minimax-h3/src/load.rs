//! The loading boundary of a deployment: which blocks a rank places, the precision choices of the
//! load, and the per-component checkpoints those blocks read.
//!
//! An H3 checkpoint is a diffusers pipeline directory: each component is its own shard set under
//! its own subdirectory, and a VAE directory holds both halves of one VAE. So the load opens one
//! checkpoint per directory rather than one for the whole pipeline, and each block's weights come
//! from the directory its component reads.
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

use std::collections::BTreeMap;
use std::path::Path;

use nn::weights::{Quantization, WeightLoader};
use nn::{DType, RetainedWeights};

use crate::{Error, Result};

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

/// One open checkpoint per component directory a rank's blocks read.
///
/// Opening a checkpoint maps its shards read-only; it allocates no device storage. A directory two
/// blocks share - a VAE's two halves, or the two condition encoders - is opened once.
pub struct Components {
    checkpoints: BTreeMap<&'static str, nn::Checkpoint>,
}

impl Components {
    /// Open every directory the placed components read.
    pub fn open(root: &Path, blocks: &[Component]) -> Result<Self> {
        let mut checkpoints = BTreeMap::new();
        for component in blocks {
            for directory in component.directories() {
                if checkpoints.contains_key(directory) {
                    continue;
                }
                let checkpoint = nn::Checkpoint::open(&root.join(directory)).map_err(|error| {
                    Error::checkpoint(format!(
                        "the {directory} component of the checkpoint at {} could not be \
                             opened: {error}",
                        root.display()
                    ))
                })?;
                checkpoints.insert(*directory, checkpoint);
            }
        }
        Ok(Self { checkpoints })
    }

    /// The checkpoint of the component directory named `directory`.
    pub fn checkpoint(&self, directory: &str) -> Result<&nn::Checkpoint> {
        self.checkpoints.get(directory).ok_or_else(|| {
            Error::checkpoint(format!(
                "the load opened no {directory} checkpoint; the placed blocks read {}",
                self.checkpoints
                    .keys()
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
    }

    /// A loader over one component's checkpoint that publishes to `store`.
    ///
    /// The precision choices travel with it, so every component of one load converts the same way
    /// and each weight is converted once (specs/worker.md 1).
    pub fn loader<'a>(
        &'a self,
        directory: &str,
        store: &'a mut dyn RetainedWeights,
        options: &LoadOptions,
    ) -> Result<WeightLoader<'a>> {
        Ok(WeightLoader::new(
            self.checkpoint(directory)?,
            store,
            options.dtype,
            options.quantization,
        ))
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
    fn a_shared_vae_directory_is_named_once_per_half() {
        assert_eq!(Component::VideoDecoder.directories(), ["vae"]);
        assert_eq!(Component::AudioDecoder.directories(), ["audio_vae"]);
        assert_eq!(Component::LatentEncoder.directories(), ["vae", "audio_vae"]);
        assert_eq!(Component::TransformerRef.directories(), ["transformer_ref"]);
    }

    #[test]
    fn a_placement_opens_each_directory_once() {
        let root = tempfile::tempdir().expect("tempdir");
        // A component directory with one shard of one two-element tensor.
        let write_shard = |directory: &str| {
            let path = root.path().join(directory);
            std::fs::create_dir_all(&path).expect("create");
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&0_u64.to_le_bytes());
            // A minimal safetensors header: one BF16 tensor of two elements.
            let header = r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0_u8; 4]);
            std::fs::write(path.join("model.safetensors"), bytes).expect("write");
        };
        for directory in ["vae", "audio_vae"] {
            write_shard(directory);
        }
        let components = Components::open(
            root.path(),
            &[
                Component::VideoDecoder,
                Component::AudioDecoder,
                Component::LatentEncoder,
            ],
        )
        .expect("the two VAE directories");
        assert!(components.checkpoint("vae").is_ok());
        assert!(components.checkpoint("audio_vae").is_ok());
        let error = components
            .checkpoint("transformer")
            .err()
            .expect("no DiT was placed");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("transformer")),
            "{error}"
        );

        let error = Components::open(root.path(), &[Component::Transformer])
            .err()
            .expect("the DiT directory is missing");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("transformer")),
            "{error}"
        );
    }
}
