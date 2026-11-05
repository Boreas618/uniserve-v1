//! The mathematical partition of a MiniMax-H3 deployment.
//!
//! A deployment fixes how the model is split. The denoiser is sharded over the packed sequence by
//! Ulysses sequence parallelism: every rank holds a contiguous slice of the sequence's tokens and
//! all of one rank's heads, and an all-to-all turns one into the other around each attention. The
//! text encoder is sharded over the vocabulary, the attention heads and every dense projection by
//! tensor parallelism. The values here are a *mathematical* description the layers apply; they are
//! not a runner, a communicator, a stream or a placement (`AGENTS.md`, Model Library Boundary).
//!
//! A sequence rank's own token slice and the order ranks are gathered in are the runner's business:
//! the model is told how many ranks the exchange runs over, and the layer that shards a tensor
//! reads only the degree and the rank's position in it.

use nn::{DType, VideoTask};

use crate::checkpoint::TASK_DITS;
use crate::denoiser::Arrangement;
use crate::{Error, Result, config::Config};

/// The text rows the released checkpoint's dense deployment holds: two 64-row units.
pub const RELEASED_TEXT_ROWS: u32 = 128;

/// The condition rows it holds: three 64-row units, nine images, three videos and two keyframes.
pub const RELEASED_CONDITION_ROWS: u32 = 192;

/// The condition rows a deployment must reserve to serve `task` at the largest canvas it offers.
///
/// `RELEASED_CONDITION_ROWS` is the checkpoint's own table - nine images, three videos and two
/// keyframes of 64-row units - and it is a statement about the **reference's** raster, not about this
/// model's: an H3 keyframe is a whole generated frame, so its rows are the canvas's patch rows, and
/// two of them at 1344x768 are 2016 against the table's 192. A deployment sized by the table stops
/// every conditioned request at the packing's capacity check (`the model packed N condition rows and
/// the layout reserves 192`).
///
/// So the reservation is derived from what the deployment can be asked for: the largest canvas it
/// offers, the keyframes the task anchors, and a reference clip's soundtrack. Returning the rows the
/// task needs is what lets a caller with one fixed case size a stage to that case rather than to the
/// deployment's worst.
pub fn condition_rows(config: &Config, task: VideoTask) -> Result<u32> {
    let dit = TASK_DITS
        .iter()
        .find_map(|(served, dit)| (*served == task).then_some(*dit))
        .ok_or_else(|| Error::config(format!("no MiniMax-H3 deployment serves {task:?}")))?;
    let denoiser = config
        .denoisers
        .get(&dit)
        .ok_or_else(|| Error::config(format!("the checkpoint holds no {} DiT", dit.directory())))?;
    // The frame's rows at the largest canvas the deployment offers: a keyframe is fitted to the
    // generated canvas, so this is its size whatever the request asks for.
    let frame_rows = denoiser
        .offered_canvases()
        .iter()
        .map(|canvas| crate::packing::latent_raster(canvas).map(|raster| raster.patch_rows()))
        .collect::<Result<Vec<u32>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let rows = match task {
        VideoTask::TextToVideoAudio => 0,
        VideoTask::FirstLastToVideoAudio => 2 * frame_rows,
        VideoTask::ReferenceToVideoAudio => {
            // A reference **with a soundtrack is a clip** (`an H3 image reference carries no audio`),
            // so it is `CLIP_FRAMES` pixel frames and the packing's own rule reduces them to latent
            // frames - not one frame with a track beside it. One latent frame fewer is a capacity
            // that refuses the request it was derived for.
            let clip = crate::encoding::Clip::of(&config.video_vae)?;
            let latents = clip.latent_frames(crate::CLIP_FRAMES)?;
            // The track is one clip long, and the packing counts it in latent steps of stereo rows.
            let samples = crate::CLIP_FRAMES * crate::AUDIO_SAMPLE_RATE / crate::FPS;
            latents * frame_rows + crate::AUDIO_CHANNELS * samples.div_ceil(crate::AUDIO_HOP)
        }
    };
    capacity(config, dit, rows)
}

/// The smallest whole number of the deployment's capacity units that holds `rows` condition rows.
///
/// An arrangement's capacity is stated in units (`attention.capacity_rows()`, 64 for the dense
/// paths), so a caller that sizes a stage from the rows its case asks for has to round up: 2016 rows
/// of two keyframes at 1344x768 is 31.5 units, and a capacity is not a fraction of a unit.
pub fn capacity(config: &Config, dit: crate::checkpoint::Dit, rows: u32) -> Result<u32> {
    let denoiser = config
        .denoisers
        .get(&dit)
        .ok_or_else(|| Error::config(format!("the checkpoint holds no {} DiT", dit.directory())))?;
    let unit = denoiser.attention.capacity_rows();
    if rows == 0 {
        // A task with no conditions still reserves one unit: the packing's text region and the zero
        // row are placed against a capacity, and zero is refused by the arrangement.
        return Ok(unit);
    }
    rows.div_ceil(unit)
        .checked_mul(unit)
        .ok_or_else(|| Error::config(format!("a capacity of {rows} rows overflows")))
}

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
    /// The video postprocessor: the clip splice. Holds no weights, so it names no checkpoint directory and
    /// its block is a [`nn::Block::Decoder`] whose operation is the splice rather than a decode.
    VideoPostprocessor,
}

impl Component {
    /// This component's bit in [`Components`].
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

    /// Every component a deployment may place.
    pub const ALL: [Component; 7] = [
        Component::TextEncoder,
        Component::Transformer,
        Component::TransformerRef,
        Component::VideoDecoder,
        Component::AudioDecoder,
        Component::LatentEncoder,
        Component::VideoPostprocessor,
    ];

    /// The component's name in the checkpoint and in a deployment file.
    ///
    /// This is the checkpoint's own naming - a directory under the pipeline root - not the name the
    /// runner addresses the built block by; that is [`Component::block`].
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
            // The postprocessor holds no weights: its arithmetic is the splice, and the constants it reads
            // are the ImageNet means and standard deviations the reference hard-codes.
            Self::VideoPostprocessor => &[],
        }
    }

    /// The element type this block is declared to compute in when a deployment states none.
    ///
    /// The design's default `dtype = "auto"` converts nothing at load, so each block keeps the
    /// checkpoint's own representation: BF16 for the text encoder and the DiT partitions, FP16 for
    /// the video VAE decoder (`specs/worker.md` 1; the `quality` preset of
    /// `uniserve/models/minimax_h3/precision.py`, which is that representation). This is the type a
    /// block whose constructor takes one is *converted to*; a constructor that takes none keeps the
    /// stored representation, so this value is informational for those blocks. The design names no
    /// representation for the audio VAE or the condition encoders, so they are declared BF16 like
    /// the blocks beside them and no conversion is made for them.
    pub const fn activation(self) -> DType {
        match self {
            // The postprocessor reads and writes the video decoder's rasters, so it holds their element type.
            Self::VideoDecoder | Self::VideoPostprocessor => DType::F16,
            Self::TextEncoder
            | Self::Transformer
            | Self::TransformerRef
            | Self::AudioDecoder
            | Self::LatentEncoder => DType::BF16,
        }
    }

    /// The name the runner addresses this block by.
    ///
    /// Blocks are addressed by the capability their runner serves, not by the checkpoint's
    /// directories: both DiT partitions are the `Denoiser` block, so a rank that places either
    /// serves the same media call. The engine's media calls name `denoiser`, `text_encoder`,
    /// `video_decoder`, `audio_decoder` and `latent_encoder`, which is this list.
    pub const fn block(self) -> &'static str {
        match self {
            Self::TextEncoder => "text_encoder",
            Self::Transformer | Self::TransformerRef => "denoiser",
            Self::VideoDecoder => "video_decoder",
            Self::AudioDecoder => "audio_decoder",
            Self::LatentEncoder => "latent_encoder",
            Self::VideoPostprocessor => "video_postprocessor",
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
/// `text_rows` and `condition_rows` are **rows**, and each is a whole number of the attention
/// kind's capacity unit - 64 rows under dense attention, one VSA tile under sparse attention -
/// which is also what `Denoiser::load`'s `Arrangement` takes. The released checkpoint's dense
/// deployment is 128 text rows and 192 condition rows (two and three units: nine images, three
/// videos and two keyframes at one unit each); under a 128-row tile the same deployment needs
/// capacities in whole 128-row tiles instead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capacity {
    /// Text rows every layout of this deployment holds, a whole number of capacity units.
    pub text_rows: u32,
    /// Condition rows it holds, a whole number of capacity units.
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
        let text = self.text_rows;
        let condition = self.condition_rows;
        let frames = crate::packing::align_num_frames(self.frames);
        let mut largest = 0;
        for canvas in config.offered_canvases() {
            let packing = crate::packing::dense_packing(frames, canvas, text, condition, unit)?;
            largest = largest.max(packing.padded_tokens);
            // A sparse deployment packs by video tiles as well, and its segments pad the audio
            // and the video up to whole tiles; the bound is the larger of the two plans at the
            // same admission, so neither can overrun the scratch.
            if let crate::config::AttentionKind::Sparse { tile, .. } = config.attention {
                let segments =
                    crate::packing::segment_packing(frames, canvas, text, condition, tile, tile)?;
                largest = largest.max(segments.extent.padded_tokens);
            }
        }
        if largest == 0 {
            return Err(Error::config(
                "the DiT offers no canvas, so no layout can be sized".to_owned(),
            ));
        }
        Ok(largest)
    }
}

/// The set of blocks one rank places.
///
/// A deployment places a handful of blocks out of six, so the set is a bit per block: it is `Copy`,
/// which keeps [`Partition`] `Copy` and lets a layer hold the partition it was built for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Components(u8);

impl Components {
    /// Every block a deployment may place.
    pub const ALL: Components = Components((1 << Component::ALL.len()) - 1);

    /// No block.
    pub const NONE: Components = Components(0);

    /// The element type of the checkpoint's stored representation for the block, per the design's
    /// `auto` default: the video VAE decoder is FP16 as stored, the rest BF16.
    pub fn of(components: impl IntoIterator<Item = Component>) -> Self {
        let mut set = Self::NONE;
        for component in components {
            set = set.with(component);
        }
        set
    }

    /// The set with `component` added.
    pub const fn with(self, component: Component) -> Self {
        Self(self.0 | 1 << component.bit())
    }

    /// The set without `component`.
    pub const fn without(self, component: Component) -> Self {
        Self(self.0 & !(1 << component.bit()))
    }

    /// Whether `component` is placed.
    pub const fn contains(self, component: Component) -> bool {
        self.0 & (1 << component.bit()) != 0
    }

    /// Whether no block is placed.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The blocks placed, in deployment order.
    pub fn iter(self) -> impl Iterator<Item = Component> {
        Component::ALL
            .into_iter()
            .filter(move |component| self.contains(*component))
    }
}

impl FromIterator<Component> for Components {
    fn from_iter<I: IntoIterator<Item = Component>>(components: I) -> Self {
        Self::of(components)
    }
}

/// How one deployment splits the model, and which of its blocks this rank places.
///
/// The partition is a deployment fact the epoch hands the model: the degrees and the rank's place
/// in them, the blocks the rank holds, and the row arrangement the blocks are sized for. It is
/// still only a mathematical description - the shard sets, the communicator, the buffers and the
/// streams are the runner's (`AGENTS.md`, Model Library Boundary).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Partition {
    /// Ulysses sequence-parallel degree of the denoiser: the heads each rank attends and the
    /// ranks the all-to-all runs over.
    pub ulysses_degree: usize,
    /// Tensor-parallel degree of the text encoder: vocabulary, attention heads and dense
    /// projections are sharded over it.
    pub tensor_parallel: usize,
    /// This rank's position in the text encoder's tensor-parallel group.
    pub tensor_rank: usize,
    /// This rank's position in the **denoiser's** Ulysses group.
    ///
    /// A second, independent rank: the text encoder's tensor shard and the denoiser's sequence shard
    /// are different groups, and a deployment may shard one and not the other. The denoiser's rank
    /// decides which slice of the initial noise stream it draws and which shard of the packed rows it
    /// writes, so a deployment that split the sequence and left this at zero would have every rank
    /// draw and write the same shard - a whole video, self-consistent, and not the one the reference
    /// makes.
    pub sequence_rank: usize,
    /// The blocks this rank places.
    ///
    /// A base release holds both DiT partitions and a deployment places one; a rank that places
    /// none holds nothing. The set is the deployment file's, not the checkpoint's.
    pub blocks: Components,
    /// Text rows every layout this deployment serves holds.
    ///
    /// A whole number of the attention kind's capacity units - 64 rows under dense attention, one
    /// VSA tile under sparse attention - which is the rule the denoiser's `Arrangement` states.
    /// The released checkpoint's dense deployment is 128 text rows (two units).
    pub text_rows: u32,
    /// Condition rows it holds, a whole number of the same units.
    ///
    /// The released checkpoint's dense deployment is 192 rows (three units: nine images, three
    /// videos and two keyframes at one unit each). Under a 128-row tile the capacities must be
    /// whole 128-row tiles instead, and a `ref2va` deployment's are set by the reference videos'
    /// rows, not by the prompt.
    pub condition_rows: u32,
}

impl Partition {
    /// An unsharded deployment: one rank holds every block, sized for the released checkpoint.
    pub fn single() -> Self {
        Self {
            ulysses_degree: 1,
            tensor_parallel: 1,
            tensor_rank: 0,
            sequence_rank: 0,
            blocks: Components::ALL.without(Component::TransformerRef),
            text_rows: RELEASED_TEXT_ROWS,
            condition_rows: RELEASED_CONDITION_ROWS,
        }
    }

    /// The same partition, placed at `rank` of the denoiser's sequence group.
    ///
    /// A builder rather than another positional `usize` on [`Self::new`] and [`Self::serving`]: those
    /// already take the text encoder's degree and rank, and two adjacent integers of the same type
    /// with different meanings is how a deployment ends up sharded one way and drawing the other.
    pub fn at_sequence_rank(self, rank: usize) -> Self {
        Self {
            sequence_rank: rank,
            ..self
        }
    }

    /// The blocks a deployment serving `task` places over a checkpoint holding `config`.
    ///
    /// A server places **one** DiT partition, so the task decides which: `t2va` and `fl2va` run the
    /// base `transformer`, `ref2va` the `transformer_ref`. A checkpoint that holds both therefore
    /// serves the tasks of one partition per deployment, not all three.
    ///
    /// The decoders are placed for every task, because every task returns video and audio. The
    /// condition encoders are placed for the tasks that carry conditions - `fl2va`'s keyframes and
    /// `ref2va`'s references and keyframes - and not for `t2va`, whose request has none.
    /// `sequence_rank` is this rank's place in the denoiser's Ulysses group; it is not the text
    /// encoder's (`tensor_rank`), and a deployment must state both.
    pub fn serving(
        config: &Config,
        task: VideoTask,
        ulysses_degree: usize,
        tensor_parallel: usize,
        tensor_rank: usize,
        sequence_rank: usize,
    ) -> Result<Self> {
        let dit = TASK_DITS
            .iter()
            .find_map(|(served, dit)| (*served == task).then_some(*dit))
            .ok_or_else(|| Error::config(format!("no MiniMax-H3 deployment serves {task:?}")))?;
        if !config.denoisers.contains_key(&dit) {
            return Err(Error::config(format!(
                "the checkpoint holds no {} DiT, so it cannot serve {task:?} (it holds {})",
                dit.directory(),
                config
                    .denoisers
                    .keys()
                    .map(|dit| dit.directory())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let mut blocks = Components::of([
            Component::TextEncoder,
            Component::VideoDecoder,
            Component::AudioDecoder,
        ])
        .with(match dit {
            crate::checkpoint::Dit::Base => Component::Transformer,
            crate::checkpoint::Dit::Reference => Component::TransformerRef,
        });
        if matches!(
            task,
            VideoTask::FirstLastToVideoAudio | VideoTask::ReferenceToVideoAudio
        ) {
            blocks = blocks.with(Component::LatentEncoder);
        }
        let partition = Self::new(
            config,
            ulysses_degree,
            tensor_parallel,
            tensor_rank,
            sequence_rank,
        )?
        .placing(blocks)
        .sized(RELEASED_TEXT_ROWS, condition_rows(config, task)?);
        partition.validate(config)?;
        Ok(partition)
    }

    /// The same partition, placing `blocks`.
    ///
    /// A deployment that serves `ref2va` places [`Component::TransformerRef`] and drops
    /// [`Component::Transformer`]: the two are one block, and the task decides which checkpoint's
    /// weights fill it.
    pub const fn placing(mut self, blocks: Components) -> Self {
        self.blocks = blocks;
        self
    }

    /// The same partition, sized for `text_rows` text rows and `condition_rows` condition rows.
    pub fn sized(mut self, text_rows: u32, condition_rows: u32) -> Self {
        self.text_rows = text_rows;
        self.condition_rows = condition_rows;
        self
    }

    /// The row arrangement this partition gives a denoiser of `config`.
    ///
    /// The `shard` an arrangement carries is the Ulysses degree: one exchange rank's slice of a
    /// layout's rows, and the alignment the packing pads to.
    pub fn arrangement(&self, config: &crate::config::DenoiserConfig) -> Result<Arrangement> {
        let shard = u32::try_from(self.ulysses_degree).map_err(|_| {
            Error::config(format!(
                "a Ulysses degree of {} rows does not fit a row count",
                self.ulysses_degree
            ))
        })?;
        Arrangement::new(config, self.text_rows, self.condition_rows, shard)
    }

    /// A deployment's partition, checked against the checkpoint's head counts.
    ///
    /// `tensor_rank` and `sequence_rank` are two different groups' positions, and both are stated: a
    /// deployment that sharded the denoiser's sequence and named only the text encoder's rank would
    /// have every rank draw the same noise.
    pub fn new(
        config: &Config,
        ulysses_degree: usize,
        tensor_parallel: usize,
        tensor_rank: usize,
        sequence_rank: usize,
    ) -> Result<Self> {
        let partition = Self {
            ulysses_degree,
            tensor_parallel,
            tensor_rank,
            sequence_rank,
            blocks: Components::ALL.without(Component::TransformerRef),
            text_rows: RELEASED_TEXT_ROWS,
            condition_rows: RELEASED_CONDITION_ROWS,
        };
        partition.validate(config)?;
        Ok(partition)
    }

    /// Check every degree and rank against the model's dimensions.
    ///
    /// The denoiser's Ulysses degree splits the attention heads between the ranks of the
    /// exchange, so it must divide every DiT partition's head count and the group must be a power
    /// of two, which the exchange layer also refuses. The text encoder's tensor degree splits the
    /// query heads and the key/value heads, so it must divide both.
    pub fn validate(&self, config: &Config) -> Result<()> {
        if self.ulysses_degree == 0 || self.tensor_parallel == 0 {
            return Err(Error::config(
                "partition degrees must be positive".to_owned(),
            ));
        }
        if self.sequence_rank >= self.ulysses_degree {
            return Err(Error::config(format!(
                "the denoiser's rank {} is outside its Ulysses degree {}",
                self.sequence_rank, self.ulysses_degree
            )));
        }
        if !self.ulysses_degree.is_power_of_two() {
            return Err(Error::config(format!(
                "the Ulysses degree {} is not a power of two",
                self.ulysses_degree
            )));
        }
        // Both DiT partitions are the `denoiser` block - they are the same capability over two
        // checkpoints, and the task picks one - so a rank that held both would register it twice.
        if self.blocks.contains(Component::Transformer)
            && self.blocks.contains(Component::TransformerRef)
        {
            return Err(Error::config(
                "a rank cannot hold both the base and the reference denoiser: they are one block"
                    .to_owned(),
            ));
        }
        // The two condition encoders are the `Denoiser` block's own half - it runs them in
        // `prepare_conditioning`, once per request - so a rank that claims their directories must
        // hold a DiT to run them, and a partition that claimed the VAEs with no denoiser would load
        // weights nothing could use.
        if self.blocks.contains(Component::LatentEncoder)
            && !self.blocks.contains(Component::Transformer)
            && !self.blocks.contains(Component::TransformerRef)
        {
            return Err(Error::config(
                "a rank cannot hold the condition encoders without a denoiser: they are its \
                 conditioning half"
                    .to_owned(),
            ));
        }
        if self.text_rows == 0 || self.condition_rows == 0 {
            return Err(Error::config(
                "a deployment's text and condition row capacities are positive".to_owned(),
            ));
        }
        if self.tensor_rank >= self.tensor_parallel {
            return Err(Error::config(format!(
                "tensor rank {} is outside a degree of {}",
                self.tensor_rank, self.tensor_parallel
            )));
        }
        for (dit, denoiser) in &config.denoisers {
            let heads = denoiser.transformer.num_attention_heads as usize;
            if !heads.is_multiple_of(self.ulysses_degree) {
                return Err(Error::config(format!(
                    "the Ulysses degree {} does not divide the {} attention heads of {}",
                    self.ulysses_degree,
                    heads,
                    dit.directory()
                )));
            }
        }
        let text = &config.text_encoder;
        if !(text.num_attention_heads as usize).is_multiple_of(self.tensor_parallel) {
            return Err(Error::config(format!(
                "the tensor-parallel degree {} does not divide the text encoder's {} query heads",
                self.tensor_parallel, text.num_attention_heads
            )));
        }
        if !(text.num_key_value_heads as usize).is_multiple_of(self.tensor_parallel) {
            return Err(Error::config(format!(
                "the tensor-parallel degree {} does not divide the text encoder's {} key/value \
                 heads",
                self.tensor_parallel, text.num_key_value_heads
            )));
        }
        Ok(())
    }

    /// The attention heads one rank of the denoiser's exchange attends.
    pub const fn denoiser_heads(&self, heads: u32) -> u32 {
        heads / self.ulysses_degree as u32
    }

    /// The query heads one tensor-parallel rank of the text encoder holds.
    pub const fn text_query_heads(&self, config: &Config) -> u32 {
        config.text_encoder.num_attention_heads / self.tensor_parallel as u32
    }

    /// The key/value heads one tensor-parallel rank of the text encoder holds.
    pub const fn text_key_value_heads(&self, config: &Config) -> u32 {
        config.text_encoder.num_key_value_heads / self.tensor_parallel as u32
    }

    /// The vocabulary rows one tensor-parallel rank of the text encoder holds, as the half-open
    /// range `[first, stop)`.
    ///
    /// The rows are split into `tensor_parallel` contiguous shards, the last taking the remainder;
    /// the head pads its shard to a multiple of the descriptor's row alignment, and the gathered
    /// logits drop the padding.
    pub fn vocabulary_shard(&self, vocab_size: u32) -> (u32, u32) {
        let degree = self.tensor_parallel as u32;
        let rows = vocab_size.div_ceil(degree);
        let first = rows * self.tensor_rank as u32;
        (first, (first + rows).min(vocab_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released checkpoint's dimensions.
    fn config() -> Config {
        crate::config::tests_support::released_config()
    }

    #[test]
    fn a_deployments_row_bound_follows_its_condition_capacity() {
        // `Layouts::rows_bound` is the number every class-crossing row is keyed at, and the condition
        // capacity is part of the packed row count. So the three tasks' bounds must differ by exactly
        // their capacities and nothing else: a bound that stopped following the capacity is a store
        // keyed for a layout the deployment never serves, and it agrees with a walk that reads the
        // same stale number - which is why this is asserted at the layout rule rather than at a flag.
        let config = config();
        let denoiser = config
            .denoisers
            .get(&crate::checkpoint::Dit::Base)
            .expect("a base DiT");
        let bound = |task| -> u32 {
            let partition = Partition::serving(&config, task, 1, 1, 0, 0).expect("a partition");
            let arrangement = partition.arrangement(denoiser).expect("an arrangement");
            crate::denoiser::Layouts::new(denoiser.clone(), arrangement)
                .expect("a layout rule")
                .rows_bound()
                .expect("a bound")
        };
        let unconditioned = bound(VideoTask::TextToVideoAudio);
        let keyframes = bound(VideoTask::FirstLastToVideoAudio);
        let reference = bound(VideoTask::ReferenceToVideoAudio);
        let capacity = |task| condition_rows(&config, task).expect("a capacity");
        assert_eq!(
            keyframes - unconditioned,
            capacity(VideoTask::FirstLastToVideoAudio) - capacity(VideoTask::TextToVideoAudio),
            "the keyframe deployment's bound moves by its capacity and by nothing else"
        );
        assert_eq!(
            reference - keyframes,
            capacity(VideoTask::ReferenceToVideoAudio) - capacity(VideoTask::FirstLastToVideoAudio),
            "a reference clip's bound moves by the extra rows its clip reserves"
        );
    }

    #[test]
    fn a_conditioned_deployment_reserves_its_own_keyframes_not_the_checkpoints_table() {
        // The released table's 192 rows describe the reference's 64-row units. An H3 keyframe is a
        // whole generated frame: 1008 patch rows at 1344x768, so two of them are 2016 and the
        // capacity rounds up to the arrangement's 64-row unit. A deployment that reserved the table
        // stops every conditioned request at the packing's capacity check.
        let config = config();
        assert_eq!(
            condition_rows(&config, VideoTask::FirstLastToVideoAudio).expect("a capacity"),
            2048
        );
        // A reference **with a soundtrack is a clip**: `CLIP_FRAMES` pixel frames, which the
        // packing reduces to two latent frames at 1008 patch rows each, plus the soundtrack of one
        // clip (29 latent steps of two channels) - 2 x 1008 + 58 = 2074, in whole units.
        assert_eq!(
            condition_rows(&config, VideoTask::ReferenceToVideoAudio).expect("a capacity"),
            2112
        );
        // An unconditioned task still reserves one unit, because an arrangement refuses zero.
        assert_eq!(
            condition_rows(&config, VideoTask::TextToVideoAudio).expect("a capacity"),
            64
        );
        let serving = Partition::serving(&config, VideoTask::FirstLastToVideoAudio, 1, 1, 0, 0)
            .expect("a serving partition");
        assert_eq!(serving.condition_rows, 2048);
    }

    #[test]
    fn an_unsharded_partition_holds_the_whole_model() {
        let config = config();
        let partition = Partition::single();
        partition.validate(&config).expect("one rank");
        assert_eq!(partition.denoiser_heads(56), 56);
        assert_eq!(partition.text_query_heads(&config), 64);
        assert_eq!(partition.text_key_value_heads(&config), 8);
        assert_eq!(partition.vocabulary_shard(151_936), (0, 151_936));
        // One rank holds every component of a base deployment, and one denoiser: the task's. A
        // `ref2va` deployment places `TransformerRef` instead.
        assert!(partition.blocks.contains(Component::Transformer));
        assert!(!partition.blocks.contains(Component::TransformerRef));
        assert_eq!(partition.blocks.iter().count(), Component::ALL.len() - 1);
    }

    #[test]
    fn four_way_ulysses_and_text_tensor_parallelism_split_the_heads() {
        let config = config();
        let partition = Partition::new(&config, 4, 4, 2, 0).expect("four ranks");
        assert_eq!(partition.denoiser_heads(56), 14);
        assert_eq!(partition.text_query_heads(&config), 16);
        assert_eq!(partition.text_key_value_heads(&config), 2);
    }

    #[test]
    fn eight_way_ulysses_splits_fifty_six_heads_into_seven() {
        let config = config();
        let partition = Partition::new(&config, 8, 8, 0, 0).expect("eight ranks");
        assert_eq!(partition.denoiser_heads(56), 7);
        assert_eq!(partition.tensor_rank, 0);
    }

    #[test]
    fn the_condition_encoders_cannot_be_placed_without_a_denoiser() {
        // They are the denoiser's conditioning half, so a rank that claims their two checkpoint
        // directories must hold the DiT that runs them; otherwise it loads weights nothing uses.
        let config = config();
        let mut alone = Partition::single();
        alone.blocks = Components::of([Component::LatentEncoder]);
        let error = alone.validate(&config).expect_err("no denoiser");
        assert!(format!("{error}").contains("conditioning half"), "{error}");

        // With a denoiser beside them the same partition validates, which is the placement the
        // deployment actually uses.
        let mut together = alone;
        together.blocks = together.blocks.with(Component::Transformer);
        together.validate(&config).expect("a denoiser beside them");
    }

    #[test]
    fn a_partition_that_does_not_divide_the_heads_is_refused() {
        let config = config();
        let error = Partition::new(&config, 3, 4, 0, 0).expect_err("three is not a power of two");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("power of two")),
            "{error}"
        );
        let error = Partition::new(&config, 5, 1, 0, 0).expect_err("five does not divide 56");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("Ulysses")),
            "{error}"
        );
        let error =
            Partition::new(&config, 1, 3, 0, 0).expect_err("three does not divide 64 heads");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("query heads")),
            "{error}"
        );
        let error = Partition::new(&config, 1, 1, 1, 0).expect_err("rank outside the degree");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("outside")),
            "{error}"
        );
    }

    #[test]
    fn the_vocabulary_shard_gives_the_last_rank_the_remainder() {
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 3,
            tensor_rank: 2,
            ..Partition::single()
        };
        assert_eq!(partition.vocabulary_shard(151_936), (101_292, 151_936));
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 2,
            tensor_rank: 0,
            ..Partition::single()
        };
        assert_eq!(partition.vocabulary_shard(151_936), (0, 75_968));
        // A vocabulary the degree does not divide leaves the last shard short.
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 4,
            tensor_rank: 3,
            ..Partition::single()
        };
        assert_eq!(partition.vocabulary_shard(10), (9, 10));
    }

    #[test]
    fn a_block_is_named_for_the_capability_its_runner_serves() {
        // Both DiT partitions are the denoiser: a rank places one of them, and the runner's media
        // call looks the capability up, not the checkpoint's directory.
        assert_eq!(Component::Transformer.block(), "denoiser");
        assert_eq!(Component::TransformerRef.block(), "denoiser");
        assert_ne!(
            Component::Transformer.name(),
            Component::TransformerRef.name(),
            "the two partitions stay distinct components of one checkpoint"
        );
        for component in Component::ALL {
            assert!(!component.block().is_empty());
            assert!(!component.name().is_empty());
        }
    }

    #[test]
    fn a_rank_holds_one_denoiser() {
        let config = crate::config::tests_support::released_config();
        let both = Partition {
            blocks: Components::of([Component::Transformer, Component::TransformerRef])
                .with(Component::TextEncoder),
            ..Partition::single()
        };
        let refused = both.validate(&config).expect_err("two denoisers");
        assert!(refused.to_string().contains("one block"), "{refused}");
        let one = Partition {
            blocks: Components::of([Component::Transformer]).with(Component::TextEncoder),
            ..Partition::single()
        };
        one.validate(&config).expect("one denoiser");
    }

    #[test]
    fn the_sequence_rank_is_the_denoisers_and_not_the_encoders() {
        let config = config();
        // One tensor shard over two sequence shards: every rank's `tensor_rank` is 0, and the
        // partition still tells the two ranks apart - which is what the denoiser's draw offsets and
        // row shards are counted from.
        let first = Partition::serving(&config, VideoTask::TextToVideoAudio, 2, 1, 0, 0)
            .expect("two ranks");
        let second = first.at_sequence_rank(1);
        assert_eq!(first.tensor_rank, second.tensor_rank);
        assert_eq!(first.ulysses_degree, second.ulysses_degree);
        assert_ne!(first.sequence_rank, second.sequence_rank);
        first.validate(&config).expect("the first rank");
        second.validate(&config).expect("the second rank");

        // A rank outside its own degree is refused rather than drawn as another rank's.
        let outside = first.at_sequence_rank(2);
        let refused = outside.validate(&config).expect_err("outside the degree");
        assert!(refused.to_string().contains("denoiser's rank"), "{refused}");
    }

    #[test]
    fn a_deployment_places_one_denoiser_and_the_encoders_its_task_needs() {
        let config = config();

        // `t2va` carries no conditions, so it needs no condition encoder.
        let t2va =
            Partition::serving(&config, VideoTask::TextToVideoAudio, 1, 1, 0, 0).expect("t2va");
        assert_eq!(
            t2va.blocks,
            Components::of([
                Component::TextEncoder,
                Component::Transformer,
                Component::VideoDecoder,
                Component::AudioDecoder,
            ])
        );

        // A keyframe is a condition unit, so `fl2va` runs the encoders; it shares `t2va`'s DiT.
        let fl2va = Partition::serving(&config, VideoTask::FirstLastToVideoAudio, 1, 1, 0, 0)
            .expect("fl2va");
        assert!(fl2va.blocks.contains(Component::LatentEncoder));
        assert!(fl2va.blocks.contains(Component::Transformer));
        assert!(!fl2va.blocks.contains(Component::TransformerRef));

        // `ref2va` is the reference partition, and its references and keyframes need the encoders.
        let ref2va = Partition::serving(&config, VideoTask::ReferenceToVideoAudio, 1, 1, 0, 0)
            .expect("ref2va");
        assert!(ref2va.blocks.contains(Component::TransformerRef));
        assert!(!ref2va.blocks.contains(Component::Transformer));
        assert!(ref2va.blocks.contains(Component::LatentEncoder));
    }

    #[test]
    fn a_deployment_a_checkpoint_cannot_fill_is_refused() {
        // An OmniRef export holds only the reference partition: it serves `ref2va` and nothing else.
        let mut config = config();
        config
            .denoisers
            .remove(&crate::checkpoint::Dit::Base)
            .expect("the base partition");
        let error = Partition::serving(&config, VideoTask::TextToVideoAudio, 1, 1, 0, 0)
            .expect_err("no transformer");
        assert!(error.to_string().contains("transformer"), "{error}");
        Partition::serving(&config, VideoTask::ReferenceToVideoAudio, 1, 1, 0, 0)
            .expect("the reference partition is held");
    }
}
