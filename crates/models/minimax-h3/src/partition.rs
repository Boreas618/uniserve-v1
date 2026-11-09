//! The mathematical partition of a MiniMax-H3 deployment.
//!
//! A deployment fixes how the model is split. The denoiser is sharded over the packed sequence by
//! Ulysses sequence parallelism: every rank holds a contiguous slice of the sequence's tokens and
//! all of one rank's heads, and an all-to-all turns one into the other around each attention. The
//! text encoder is sharded over the vocabulary, the attention heads and every dense projection by
//! tensor parallelism. The values here are a *mathematical* description the layers apply; they are
//! not a runner, a communicator, a stream or a rank layout (`AGENTS.md`, Model Library Boundary).
//!
//! A sequence rank's own token slice and the order ranks are gathered in are the runner's business:
//! the model is told how many ranks the exchange runs over, and the layer that shards a tensor
//! reads only the degree and the rank's position in it.

use crate::VideoTask;
use nn::family::{Component, Components, Projection, Split};

use crate::checkpoint::TASK_DITS;
use crate::packing::{Condition, Frame, Role, Tiling};
use crate::{Error, Result, config::Config};

/// The condition rows a deployment must reserve to serve `task` at the largest canvas it offers.
///
/// The release's own table (192 rows) is the checkpoint's - nine images, three videos and two
/// keyframes of 64-row units - and it is a statement about the **reference's** raster, not about this
/// model's: an H3 keyframe is a whole generated frame, so its rows are the canvas's patch rows, and
/// two of them at 1344x768 are 2016 against the table's 192. A deployment sized by the table stops
/// every conditioned request at the packing's capacity check (`the model packed N condition rows and
/// the layout reserves 192`).
///
/// So the reservation is derived from what the deployment can be asked for: the keyframes the task
/// anchors, or a reference clip and its soundtrack, at whichever canvas the checkpoint generates
/// they take most rows of, counted row for row by the packing's own count
/// ([`crate::packing::Tiling::condition_rows`]). Returning the rows the task needs is what lets a
/// caller with one fixed case size a stage to that case rather than to the deployment's worst.
pub fn condition_rows(config: &Config, task: VideoTask) -> Result<u32> {
    let dit = dit_of(task)?;
    if !config.denoisers.contains_key(&dit) {
        return Err(Error::config(format!(
            "the checkpoint holds no {} DiT",
            dit.directory()
        )));
    }
    let mut rows = 0;
    for (_, canvas) in crate::config::named_canvases()? {
        // A keyframe is fitted to the generated canvas, so its raster is the canvas's whatever
        // the request asks for.
        let keyframe = |role| Condition {
            role,
            frame: Some(Frame {
                canvas,
                num_frames: 1,
            }),
            audio_samples: 0,
        };
        let conditions = match task {
            VideoTask::TextToVideoAudio => Vec::new(),
            VideoTask::FirstLastToVideoAudio => {
                vec![keyframe(Role::FirstFrame), keyframe(Role::LastFrame)]
            }
            // A reference **with a soundtrack is a clip** (`an H3 image reference carries no
            // audio`), so it is one clip of the frame rule's pixel frames, which the packing
            // reduces to latent frames - not one frame with a track beside it - and its track is one
            // clip long. One latent frame fewer is a capacity that refuses the request it was
            // derived for.
            VideoTask::ReferenceToVideoAudio => vec![Condition {
                role: Role::Reference,
                frame: Some(Frame {
                    canvas,
                    num_frames: crate::FRAMES.clip_frames,
                }),
                audio_samples: crate::FRAMES.clip_frames * crate::AUDIO_SAMPLE_RATE / crate::FPS,
            }],
        };
        let segments = crate::packing::condition_segments(&conditions, &canvas)?;
        rows = rows.max(Tiling::ROWS.condition_rows(&segments));
    }
    capacity(config, dit, rows)
}

/// The smallest whole number of the deployment's capacity units that holds `rows` condition rows.
///
/// A capacity is stated in units (`attention.capacity_rows()`, 64 for the dense
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
        // row are placed against a capacity, and a layout rule refuses zero.
        return Ok(unit);
    }
    rows.div_ceil(unit)
        .checked_mul(unit)
        .ok_or_else(|| Error::config(format!("a capacity of {rows} rows overflows")))
}

/// The checkpoint directories `component`'s weights come from.
///
/// A VAE directory holds both halves of its VAE, so a decoder and the encoder beside it read the
/// same directory and each loads its own half. The postprocessor holds no weights: its arithmetic is
/// the splice.
pub const fn directories(component: Component) -> &'static [&'static str] {
    match component {
        Component::TextEncoder => &["text_encoder"],
        Component::Transformer => {
            const BASE: [&str; 1] = [crate::checkpoint::Dit::Base.directory()];
            &BASE
        }
        Component::TransformerRef => {
            const REFERENCE: [&str; 1] = [crate::checkpoint::Dit::Reference.directory()];
            &REFERENCE
        }
        Component::VideoDecoder => &["vae"],
        Component::AudioDecoder => &["audio_vae"],
        Component::LatentEncoder => &["vae", "audio_vae"],
        Component::VideoPostprocessor => &[],
    }
}

/// The DiT partition that serves `task`.
fn dit_of(task: VideoTask) -> Result<crate::checkpoint::Dit> {
    TASK_DITS
        .iter()
        .find_map(|(served, dit)| (*served == task).then_some(*dit))
        .ok_or_else(|| Error::config(format!("no MiniMax-H3 deployment serves {task:?}")))
}

/// The blocks a deployment serving `tasks` places over a checkpoint holding `config`.
///
/// A server places **one** DiT partition, so the tasks must share one: `t2va` and `fl2va` run the
/// base `transformer`, `ref2va` the `transformer_ref`. The decoders and the postprocessor are
/// placed for every task, because every task returns video and audio; the condition encoders for
/// the tasks that carry conditions - `fl2va`'s keyframes and `ref2va`'s references and keyframes -
/// and not for `t2va`, whose request has none.
///
/// # Errors
///
/// No task, a task the checkpoint does not state or whose DiT it does not hold, or tasks of both
/// DiT partitions: the two are one block, and a rank that held both would register it twice.
pub fn blocks(config: &Config, tasks: &[VideoTask]) -> Result<Components> {
    // **The checkpoint is the authority on which tasks exist.** A checkpoint that does not state
    // a task cannot serve it, whatever a deployment asks for.
    let stated = config.tasks();
    let mut blocks = Components::of([
        Component::TextEncoder,
        Component::VideoDecoder,
        Component::AudioDecoder,
        Component::VideoPostprocessor,
    ]);
    if tasks.is_empty() {
        return Err(Error::config(
            "a deployment serves at least one task".to_owned(),
        ));
    }
    for task in tasks {
        if !stated.contains(task) {
            return Err(Error::config(format!(
                "this checkpoint states {} and not {task:?}",
                stated
                    .iter()
                    .map(|task| format!("{task:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let dit = dit_of(*task)?;
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
        blocks = blocks.with(match dit {
            crate::checkpoint::Dit::Base => Component::Transformer,
            crate::checkpoint::Dit::Reference => Component::TransformerRef,
        });
        if matches!(
            task,
            VideoTask::FirstLastToVideoAudio | VideoTask::ReferenceToVideoAudio
        ) {
            blocks = blocks.with(Component::LatentEncoder);
        }
    }
    // Both DiT partitions are the `denoiser` block - they are the same capability over two
    // checkpoints, and the task picks one - so a rank that held both would register it twice.
    if blocks.contains(Component::Transformer) && blocks.contains(Component::TransformerRef) {
        return Err(Error::config(
            "a rank cannot hold both the base and the reference denoiser: they are one block"
                .to_owned(),
        ));
    }
    Ok(blocks)
}

/// How one deployment splits the model: the degrees and this rank's place in them.
///
/// The partition is the mathematical description the layers apply. Which blocks a rank places
/// ([`blocks`], `nn::family::Components`) and the row capacities its layouts hold
/// (`crate::denoiser::Capacity`, from the deployment's `VideoServing`) are the deployment's and
/// travel beside it; the shard sets, the communicator, the buffers and the streams are the
/// runner's (`AGENTS.md`, Model Library Boundary).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Partition {
    /// Ulysses sequence-parallel degree of the denoiser: the heads each rank attends and the
    /// ranks the all-to-all runs over.
    pub ulysses_degree: usize,
    /// How a Ulysses member's head projections reach it, which decides the weights it holds. A group
    /// of one has no exchange and does not read this.
    pub projection: Projection,
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
}

impl Partition {
    /// An unsharded deployment: one rank holds the whole sequence and the whole text encoder.
    pub fn single() -> Self {
        Self {
            ulysses_degree: 1,
            projection: Projection::Exchange,
            tensor_parallel: 1,
            tensor_rank: 0,
            sequence_rank: 0,
        }
    }

    /// The same partition at `ulysses_degree` and `tensor_parallel`, with every other field as it
    /// stands.
    ///
    /// The two degrees are what a program key can depend on: a sparse attention states `heads` as the
    /// heads one exchange rank attends (`heads / degree`), and the text encoder's degree divides its
    /// query heads. A walk that cannot state them enumerates one deployment's rows and calls them the
    /// model's - which is why this is a builder on a stated partition rather than a constant.
    pub const fn sharded(mut self, ulysses_degree: usize, tensor_parallel: usize) -> Self {
        self.ulysses_degree = ulysses_degree;
        self.tensor_parallel = tensor_parallel;
        self
    }

    /// The same partition, reaching its heads by `projection`.
    pub const fn projecting(mut self, projection: Projection) -> Self {
        self.projection = projection;
        self
    }

    /// The same partition, placed at `rank` of the denoiser's sequence group.
    ///
    /// A builder rather than another positional `usize` on [`Self::new`]: it already takes the text
    /// encoder's degree and rank, and two adjacent integers of the same type
    /// with different meanings is how a deployment ends up sharded one way and drawing the other.
    pub fn at_sequence_rank(self, rank: usize) -> Self {
        Self {
            sequence_rank: rank,
            ..self
        }
    }

    /// A deployment's partition, checked against the checkpoint's head counts.
    ///
    /// `tensor_rank` and `sequence_rank` are two different groups' positions, and both are stated: a
    /// deployment that sharded the denoiser's sequence and named only the text encoder's rank would
    /// have every rank draw the same noise.
    ///
    /// **A partition built here is an exchange one**, because this constructor takes the degree and
    /// holds no projection; a deployment that has a projection to state adds it with
    /// [`Self::projecting`].
    pub fn new(
        config: &Config,
        ulysses_degree: usize,
        tensor_parallel: usize,
        tensor_rank: usize,
        sequence_rank: usize,
    ) -> Result<Self> {
        let partition = Self {
            ulysses_degree,
            projection: Projection::Exchange,
            tensor_parallel,
            tensor_rank,
            sequence_rank,
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
        config.text_encoder.text.local_heads(self.text_split()?)?;
        Ok(())
    }

    /// The attention heads one rank of the denoiser's exchange attends.
    pub const fn denoiser_heads(&self, heads: u32) -> u32 {
        heads / self.ulysses_degree as u32
    }

    /// The denoiser's sequence split: the Ulysses degree and this rank's place in it.
    pub fn sequence(&self) -> Result<Split> {
        let count = |value: usize, what: &str| {
            u32::try_from(value)
                .map_err(|_| Error::config(format!("a sequence-parallel {what} wider than a u32")))
        };
        Ok(Split {
            size: count(self.ulysses_degree, "degree")?,
            rank: count(self.sequence_rank, "rank")?,
        })
    }

    /// The text encoder's tensor-parallel split: this rank's place in its group.
    pub fn text_split(&self) -> Result<Split> {
        let count = |value: usize, what: &str| {
            u32::try_from(value)
                .map_err(|_| Error::config(format!("a tensor-parallel {what} wider than a u32")))
        };
        Ok(Split {
            size: count(self.tensor_parallel, "degree")?,
            rank: count(self.tensor_rank, "rank")?,
        })
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
            let capacity = crate::denoiser::Capacity {
                text_rows: 128,
                condition_rows: condition_rows(&config, task).expect("a capacity"),
            };
            crate::denoiser::Layouts::new(denoiser.clone(), capacity, Split::WHOLE)
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
        // capacity rounds up to the attention's 64-row unit. A deployment that reserved the table
        // stops every conditioned request at the packing's capacity check.
        let config = config();
        assert_eq!(
            condition_rows(&config, VideoTask::FirstLastToVideoAudio).expect("a capacity"),
            2048
        );
        // A reference **with a soundtrack is a clip**: a clip's pixel frames, which the
        // packing reduces to two latent frames at 1008 patch rows each, plus the soundtrack of one
        // clip (29 latent steps of two channels) - 2 x 1008 + 58 = 2074, in whole units.
        assert_eq!(
            condition_rows(&config, VideoTask::ReferenceToVideoAudio).expect("a capacity"),
            2112
        );
        // An unconditioned task still reserves one unit, because a layout rule refuses zero.
        assert_eq!(
            condition_rows(&config, VideoTask::TextToVideoAudio).expect("a capacity"),
            64
        );
    }

    #[test]
    fn an_unsharded_partition_holds_the_whole_model() {
        let config = config();
        let partition = Partition::single();
        partition.validate(&config).expect("one rank");
        assert_eq!(partition.denoiser_heads(56), 56);
        let split = partition.text_split().expect("a split");
        let heads = config
            .text_encoder
            .text
            .local_heads(split)
            .expect("the heads");
        assert_eq!(heads, (64, 8));
    }

    #[test]
    fn four_way_ulysses_and_text_tensor_parallelism_split_the_heads() {
        let config = config();
        let partition = Partition::new(&config, 4, 4, 2, 0).expect("four ranks");
        assert_eq!(partition.denoiser_heads(56), 14);
        let split = partition.text_split().expect("a split");
        let heads = config
            .text_encoder
            .text
            .local_heads(split)
            .expect("the heads");
        assert_eq!(heads, (16, 2));
    }

    #[test]
    fn eight_way_ulysses_splits_fifty_six_heads_into_seven() {
        let config = config();
        let partition = Partition::new(&config, 8, 8, 0, 0).expect("eight ranks");
        assert_eq!(partition.denoiser_heads(56), 7);
        assert_eq!(partition.tensor_rank, 0);
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
        // The text encoder's stack states its own divisibility rule.
        assert!(
            error
                .to_string()
                .contains("does not divide 64 attention heads"),
            "{error}"
        );
        let error = Partition::new(&config, 1, 1, 1, 0).expect_err("rank outside the degree");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("outside")),
            "{error}"
        );
    }

    #[test]
    fn a_rank_holds_one_denoiser() {
        // `t2va` runs on the base network and `ref2va` on the reference one: one rank serving both
        // would hold the denoiser block twice.
        let refused = blocks(
            &config(),
            &[
                VideoTask::TextToVideoAudio,
                VideoTask::ReferenceToVideoAudio,
            ],
        )
        .expect_err("two denoisers");
        assert!(refused.to_string().contains("one block"), "{refused}");
        blocks(
            &config(),
            &[
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
            ],
        )
        .expect("two tasks of one network");
    }

    #[test]
    fn the_sequence_rank_is_the_denoisers_and_not_the_encoders() {
        let config = config();
        // One tensor shard over two sequence shards: every rank's `tensor_rank` is 0, and the
        // partition still tells the two ranks apart - which is what the denoiser's draw offsets and
        // row shards are counted from.
        let first = Partition::new(&config, 2, 1, 0, 0)
            .map(|partition| partition.projecting(Projection::Exchange))
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
    fn a_task_the_checkpoint_does_not_state_is_refused_by_name() {
        // The checkpoint is the authority on which tasks exist. A deployment asking for one its
        // checkpoint does not state is refused when its blocks are derived, naming the tasks the
        // checkpoint does state - rather than loading the other task's network under this name.
        let mut config = config();
        let base = config
            .denoisers
            .get_mut(&crate::checkpoint::Dit::Base)
            .expect("a base DiT");
        base.tasks
            .retain(|task| *task != VideoTask::FirstLastToVideoAudio);
        let refused = blocks(&config, &[VideoTask::FirstLastToVideoAudio])
            .expect_err("a task the checkpoint does not state");
        let said = refused.to_string();
        assert!(said.contains("states"), "{said}");
        assert!(said.contains("FirstLastToVideoAudio"), "{said}");

        // The task the base DiT still states is served, so the refusal is the statement and not the
        // derivation refusing everything after one edit.
        blocks(&config, &[VideoTask::TextToVideoAudio]).expect("the task the checkpoint states");
    }

    #[test]
    fn a_partition_carries_the_projection_its_deployment_states() {
        // The mode decides what a rank holds: the gather holds its own heads' rows of the attention
        // projections and the exchange holds every member's, so the partition keeps the statement
        // for the layers to read.
        let config = config();
        let gathered = Partition::new(&config, 2, 1, 0, 0)
            .map(|partition| partition.projecting(Projection::Gather))
            .expect("a gathering deployment");
        assert_eq!(gathered.projection, Projection::Gather);

        // One rank has no exchange and no head shard, so both modes are the identity and the mode is
        // not consulted: a one-rank deployment is accepted whichever it states, which is what a
        // config stating no sequence parallelism gets from the protocol's default.
        let one = Partition::new(&config, 1, 1, 0, 0)
            .map(|partition| partition.projecting(Projection::Gather))
            .expect("one rank, either mode");
        assert_eq!(one.ulysses_degree, 1);
    }

    #[test]
    fn a_sparse_deployment_serves_a_ulysses_group() {
        // A sharded sparse deployment attends every row for its own heads, so each member composes
        // its own heads' compression and the partition places it like a dense one.
        let mut config = config();
        config
            .denoisers
            .get_mut(&crate::checkpoint::Dit::Base)
            .expect("a base DiT")
            .attention =
            crate::config::AttentionKind::sparse(128, 0.8, None).expect("a sparse kind");
        let two = Partition::new(&config, 2, 1, 0, 0)
            .map(|partition| partition.projecting(Projection::Exchange))
            .expect("a sharded sparse deployment");
        assert_eq!(two.ulysses_degree, 2);

        // One rank serves the sparse network, which is the deployment the tile bound is stated for.
        let one = Partition::new(&config, 1, 1, 0, 0).expect("one rank serves a sparse network");
        assert_eq!(one.ulysses_degree, 1);
    }

    #[test]
    fn an_exchanging_group_divides_the_heads_it_states() {
        // The degree is what a rank's head shard is counted from, so an accepted group must leave the
        // DiT's heads divisible by it - the state a later attention layer reads rather than rechecks.
        let config = config();
        let four = Partition::new(&config, 4, 1, 0, 0)
            .map(|partition| partition.projecting(Projection::Exchange))
            .expect("four exchanging ranks");
        assert_eq!(four.ulysses_degree, 4);
        let denoiser = config
            .denoisers
            .get(&crate::checkpoint::Dit::Base)
            .expect("a base DiT");
        assert_eq!(
            four.denoiser_heads(denoiser.transformer.num_attention_heads) * 4,
            denoiser.transformer.num_attention_heads
        );
    }

    #[test]
    fn a_deployment_places_one_denoiser_and_the_encoders_its_task_needs() {
        let config = config();
        let placed = |task| blocks(&config, &[task]).expect("a served task");

        // `t2va` carries no conditions, so it needs no condition encoder. Every task places the
        // postprocessor, the video decoder's clip splice.
        assert_eq!(
            placed(VideoTask::TextToVideoAudio),
            Components::of([
                Component::TextEncoder,
                Component::Transformer,
                Component::VideoDecoder,
                Component::AudioDecoder,
                Component::VideoPostprocessor,
            ])
        );

        // A keyframe is a condition unit, so `fl2va` runs the encoders; it shares `t2va`'s DiT.
        let fl2va = placed(VideoTask::FirstLastToVideoAudio);
        assert!(fl2va.contains(Component::LatentEncoder));
        assert!(fl2va.contains(Component::Transformer));
        assert!(!fl2va.contains(Component::TransformerRef));

        // `ref2va` is the reference partition, and its references and keyframes need the encoders.
        let ref2va = placed(VideoTask::ReferenceToVideoAudio);
        assert!(ref2va.contains(Component::TransformerRef));
        assert!(!ref2va.contains(Component::Transformer));
        assert!(ref2va.contains(Component::LatentEncoder));
    }

    #[test]
    fn a_deployment_a_checkpoint_cannot_fill_is_refused() {
        // An OmniRef export holds only the reference partition: it serves `ref2va` and nothing else.
        let mut config = config();
        config
            .denoisers
            .remove(&crate::checkpoint::Dit::Base)
            .expect("the base partition");
        // A task is stated by the partitions the export holds (`Config::tasks`), so removing the
        // base partition removes `t2va` from the checkpoint's vocabulary and the refusal names the
        // task it cannot serve.
        let error = blocks(&config, &[VideoTask::TextToVideoAudio]).expect_err("no transformer");
        assert!(error.to_string().contains("TextToVideoAudio"), "{error}");
        blocks(&config, &[VideoTask::ReferenceToVideoAudio])
            .expect("the reference partition is held");
    }
}
