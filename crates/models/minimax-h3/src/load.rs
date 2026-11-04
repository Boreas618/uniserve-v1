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

use std::path::{Path, PathBuf};

use nn::DType;
use nn::weights::Quantization;
use nn::{Block, BlockName};

use crate::checkpoint::{Dit, Layout, Sidecars};
use crate::config::Config;
use crate::denoiser::{Arrangement, Denoiser};
use crate::text_encoder::TextEncoder;
use crate::{Error, Partition, Result};

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

/// The largest request a deployment admits, as the counts that size its DiT.
///
/// The attention's scratch and the per-layer buffers of a DiT are allocated once at load, for the
/// largest layout the deployment will serve, so these are capacities and not the request's own
/// numbers: a request larger than them is refused rather than overrunning a table
/// (specs/worker.md 9, `uniserve/models/minimax_h3/denoiser.py`). They belong to the loading
/// boundary because serving capacity is the caller's decision, not the architecture's
/// (`AGENTS.md`, Model Library Boundary).
///
/// `text_rows` and `condition_rows` are counted in the attention kind's capacity unit: 64 rows
/// under dense attention, one VSA tile under sparse attention. The released checkpoint's
/// deployment is text 2 units (128 rows) and conditions 3 units (192 rows): nine images, three
/// videos and two keyframes at one unit each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capacity {
    /// Text rows every layout of this deployment holds, in capacity units.
    pub text_rows: u32,
    /// Condition rows it holds, in capacity units.
    pub condition_rows: u32,
    /// The largest frame count it serves.
    ///
    /// A layout's row count grows with the frames, so this is the frame count that sizes the
    /// buffers; it is not aligned here, because `align_num_frames` lowers it to the legal
    /// timeline the checkpoint serves.
    pub frames: u32,
}

impl Capacity {
    /// The padded row count of the largest layout this capacity can reach.
    ///
    /// Every canvas the DiT offers is packed at [`Capacity::frames`] and the largest padded row
    /// count wins; that is the row bound the attention and the modulation tables are allocated
    /// for. The packing's own rules refuse a canvas or a frame count they cannot serve, so a
    /// capacity the checkpoint cannot reach is reported here rather than on the device.
    pub fn rows(&self, config: &crate::config::DenoiserConfig) -> Result<u32> {
        let unit = config.attention.capacity_rows();
        let text = self.text_rows.checked_mul(unit).ok_or_else(|| {
            Error::config(format!(
                "a text capacity of {} units overflows at {unit} rows a unit",
                self.text_rows
            ))
        })?;
        let condition = self.condition_rows.checked_mul(unit).ok_or_else(|| {
            Error::config(format!(
                "a condition capacity of {} units overflows at {unit} rows a unit",
                self.condition_rows
            ))
        })?;
        let frames = crate::packing::align_num_frames(self.frames);
        let mut largest = 0;
        for canvas in config.offered_canvases() {
            let packing = crate::packing::dense_packing(frames, canvas, text, condition, unit)?;
            largest = largest.max(packing.padded_tokens);
        }
        if largest == 0 {
            return Err(Error::config(
                "the DiT offers no canvas, so no layout can be sized".to_owned(),
            ));
        }
        Ok(largest)
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
    /// The capacity a DiT is sized for, or `None` for a deployment that places no DiT.
    pub capacity: Option<Capacity>,
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
            capacity: None,
            dtype: None,
            quantization: None,
            base: None,
        }
    }

    /// The same options, with the capacity a DiT is sized for.
    pub fn with_capacity(mut self, capacity: Capacity) -> Self {
        self.capacity = Some(capacity);
        self
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

/// Load the deployment this rank places.
///
/// `root` is a pipeline directory, `partition` this rank's place in it, and `options` the components
/// the rank places and their capacities. The configuration is read once, here, from the
/// checkpoint's own sidecars, and each block is then built from its directories' shards with the
/// representation `options` states: with `dtype` unset nothing is converted, so the DiT and the text
/// encoder stay BF16 and the video VAE decoder FP16 (specs/worker.md 1).
///
/// A block whose construction has not landed is refused with [`Error::Unsupported`], naming it,
/// rather than omitted: a deployment whose file places a block this build cannot construct must
/// fail at load, not on the device.
pub fn load(
    root: &Path,
    partition: Partition,
    loader: &mut nn::weights::WeightLoader<'_>,
    options: LoadOptions,
) -> Result<nn::Model> {
    // The checkpoint learns the pipeline root, so a directory of shards that states it is found
    // from the root the caller names.
    let checkpoint = nn::Checkpoint::open(root)?;
    let base = match &options.base {
        Some(path) => Some(nn::Checkpoint::open(path)?),
        None => None,
    };
    let base_root = base.as_ref().map(nn::Checkpoint::root);

    let layout = Layout::detect(checkpoint.root())?;
    let sidecars = Sidecars::read(checkpoint.root(), base_root, &layout)?;
    // A normalized configuration is checked against itself and the timeline as it is built.
    let config = Config::normalize(&layout, &sidecars)?;
    partition.validate(&config)?;

    let shard = u32::try_from(partition.ulysses_degree).map_err(|_| {
        Error::config(format!(
            "a Ulysses degree of {} rows does not fit a row count",
            partition.ulysses_degree
        ))
    })?;

    let mut blocks = Vec::with_capacity(options.blocks.len());
    for &component in &options.blocks {
        let block = match component {
            Component::TextEncoder => {
                let source = options.source(&checkpoint, base.as_ref(), "text_encoder")?;
                let dtype = options.activation(component);
                let mut shards = loader.reborrow(source)?;
                let encoder =
                    TextEncoder::load(&mut shards, &config.text_encoder, &partition, dtype)?;
                Block::Encoder(Box::new(encoder))
            }
            Component::Transformer | Component::TransformerRef => {
                let dit = match component {
                    Component::Transformer => Dit::Base,
                    _ => Dit::Reference,
                };
                let denoiser = config.denoisers.get(&dit).ok_or_else(|| {
                    Error::checkpoint(format!(
                        "the checkpoint states no {} DiT, so the {} block cannot be placed (it                          states {})",
                        dit.directory(),
                        component.name(),
                        config
                            .denoisers
                            .keys()
                            .map(|dit| dit.directory())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?;
                let capacity = options.capacity.ok_or_else(|| {
                    Error::config(format!(
                        "a deployment that places {} is sized for the largest request it serves:                          its text rows, condition rows and layout rows are owed, and none were                          stated",
                        component.name()
                    ))
                })?;
                let arrangement =
                    Arrangement::new(denoiser, capacity.text_rows, capacity.condition_rows, shard)?;
                // The attention's scratch is allocated for the largest layout this capacity can
                // reach, so the row bound is computed from the canvases the DiT offers.
                let rows = capacity.rows(denoiser)?;
                let source = options.source(&checkpoint, base.as_ref(), dit.directory())?;
                let mut shards = loader.reborrow(source)?;
                let denoiser =
                    Denoiser::load(denoiser.clone(), arrangement, rows, &partition, &mut shards)?;
                Block::Denoiser(Box::new(denoiser))
            }
            Component::VideoDecoder | Component::AudioDecoder | Component::LatentEncoder => {
                // The decoders and the condition encoders become capability blocks as their
                // layers land; until then a deployment that places one is refused here rather
                // than served by a half-built block.
                return Err(Error::Unsupported(format!(
                    "the {} block has no construction yet, so it cannot be placed",
                    component.name()
                )));
            }
        };
        blocks.push((BlockName::new(component.name()), block));
    }

    Ok(nn::Model::new(blocks))
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
            let header = r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0_u8; 4]);
            std::fs::write(path.join("model.safetensors"), bytes).expect("write");
        }
        root
    }

    #[test]
    fn the_row_bound_is_the_largest_layout_the_deployments_capacity_reaches() {
        use crate::config::tests_support::released_config;
        let config = released_config();
        let (_, denoiser) = config
            .only_denoiser()
            .expect("the released checkpoint's DiT");
        let unit = denoiser.attention.capacity_rows();
        let capacity = Capacity {
            text_rows: 2,
            condition_rows: 3,
            frames: 124,
        };

        // The 5 s 16:9 layout needs 6913 rows; the bound is that or more, in whole units.
        let rows = capacity.rows(denoiser).expect("a row bound");
        assert!(rows >= 6913, "{rows}");
        assert_eq!(
            rows % unit,
            0,
            "{rows} is not a whole number of {unit}-row units"
        );

        // The bound grows with the frames served, so a longer admission is a larger allocation.
        let longer = Capacity {
            frames: 362,
            ..capacity
        }
        .rows(denoiser)
        .expect("a row bound");
        assert!(longer > rows, "{longer} is not larger than {rows}");
    }

    #[test]
    fn a_capacity_sizes_a_dit_and_is_owed_by_a_deployment_that_places_one() {
        let options = LoadOptions::new(vec![Component::Transformer]);
        assert!(options.capacity.is_none());
        let sized = LoadOptions::new(vec![Component::Transformer]).with_capacity(Capacity {
            text_rows: 2,
            condition_rows: 3,
            frames: 124,
        });
        assert!(sized.capacity.is_some());
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
