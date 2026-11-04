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

use std::path::PathBuf;

use nn::DType;
use nn::weights::Quantization;

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

    /// The activation element type this block computes in when a deployment states none.
    ///
    /// The design's default `dtype = "auto"` converts nothing at load, so each block computes in
    /// the checkpoint's own representation: BF16 for the text encoder and the DiT partitions,
    /// FP16 for the video VAE decoder (`specs/worker.md` 1; the `quality` preset of
    /// `uniserve/models/minimax_h3/precision.py`, which is that representation). The design names
    /// no representation for the audio VAE or the condition encoders, so they take BF16 like the
    /// blocks beside them rather than a conversion the design did not ask for.
    pub const fn activation(self) -> DType {
        match self {
            Self::VideoDecoder => DType::F16,
            Self::TextEncoder
            | Self::Transformer
            | Self::TransformerRef
            | Self::AudioDecoder
            | Self::LatentEncoder => DType::BF16,
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
    /// The base pipeline of an export that holds only the components it changed.
    ///
    /// `FastVideo-FastH3-Omni-8-Step-V1` states a hub revision of the base release and carries
    /// only its reference DiT and its schedules; a load of it reads every other component from the
    /// base pipeline's directory. `None` for a root that holds everything.
    pub base: Option<PathBuf>,
}

impl LoadOptions {
    /// The options of a deployment that places `blocks` and converts nothing.
    pub fn new(blocks: Vec<Component>) -> Self {
        Self {
            blocks,
            dtype: None,
            quantization: None,
            base: None,
        }
    }

    /// The activation element type of `component`, the deployment's choice or the design's.
    pub fn activation(&self, component: Component) -> DType {
        self.dtype.unwrap_or_else(|| component.activation())
    }

    /// The checkpoint `directory`'s weights come from, in the root or in the base pipeline.
    ///
    /// An export holds only what it changed and states the base release it continues, so a
    /// directory the root omits is read from `base`. A directory neither holds is refused, naming
    /// both, rather than skipped: a load that silently omitted a component would fail much later,
    /// on the device.
    pub fn source<'a>(
        &self,
        root: &'a nn::Checkpoint,
        base: Option<&'a nn::Checkpoint>,
        directory: &str,
    ) -> Result<&'a nn::Checkpoint> {
        if let Ok(component) = root.component(directory) {
            return Ok(component);
        }
        match base.and_then(|base| base.component(directory).ok()) {
            Some(component) => Ok(component),
            None => Err(Error::checkpoint(format!(
                "the checkpoint holds no {directory} component and {}",
                match base {
                    Some(base) => {
                        let held = base.components().collect::<Vec<_>>().join(", ");
                        format!("its base pipeline holds none either (it holds {held})")
                    }
                    None => "no base pipeline was given to read it from".to_owned(),
                }
            ))),
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

    /// A pipeline root of `components` directories, each with one BF16 shard of one tensor.
    fn pipeline(components: &[&str]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("tempdir");
        for directory in components {
            let path = root.path().join(directory);
            std::fs::create_dir_all(&path).expect("create");
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&0_u64.to_le_bytes());
            let header = r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0_u8; 4]);
            std::fs::write(path.join("model.safetensors"), bytes).expect("write");
        }
        root
    }

    #[test]
    fn the_designs_own_representation_is_each_blocks_activation_type() {
        let options = LoadOptions::new(Component::ALL.to_vec());
        // `dtype = "auto"`: no conversion, so the blocks compute where the checkpoint stores.
        assert_eq!(Component::VideoDecoder.activation(), DType::F16);
        for component in Component::ALL {
            if component != Component::VideoDecoder {
                assert_eq!(component.activation(), DType::BF16, "{}", component.name());
            }
        }
        assert_eq!(options.activation(Component::VideoDecoder), DType::F16);
        let explicit = LoadOptions {
            dtype: Some(DType::F32),
            ..LoadOptions::new(vec![Component::VideoDecoder])
        };
        assert_eq!(explicit.activation(Component::VideoDecoder), DType::F32);
    }

    #[test]
    fn an_export_reads_what_it_omits_from_its_base_pipeline() {
        let export = pipeline(&["transformer_ref", "scheduler"]);
        let pipeline_root = pipeline(&[
            "text_encoder",
            "transformer",
            "transformer_ref",
            "vae",
            "audio_vae",
        ]);
        let root = nn::Checkpoint::open(export.path()).expect("the export");
        let base = nn::Checkpoint::open(pipeline_root.path()).expect("the base pipeline");
        let options = LoadOptions {
            base: Some(pipeline_root.path().to_path_buf()),
            ..LoadOptions::new(Component::ALL.to_vec())
        };

        // What the export holds is read from it, what it omits from the base.
        assert!(std::ptr::eq(
            options
                .source(&root, Some(&base), "transformer_ref")
                .expect("the export holds its own reference DiT"),
            root.component("transformer_ref").expect("a component"),
        ));
        assert!(std::ptr::eq(
            options
                .source(&root, Some(&base), "vae")
                .expect("the base holds the video VAE"),
            base.component("vae").expect("a component"),
        ));

        // A directory neither holds is refused, not skipped.
        let empty = LoadOptions::new(vec![Component::AudioDecoder]);
        let error = empty
            .source(&root, None, "audio_vae")
            .err()
            .expect("no audio VAE in the export");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("audio_vae")),
            "{error}"
        );
    }
}
