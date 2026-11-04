//! The diffusion transformer's configuration, and the schedule and attention a checkpoint states
//! for it.
//!
//! A checkpoint holds one or two DiTs (`transformer`, `transformer_ref`). Its `config.json` states
//! the network's dimensions; the schedule and the attention are the base release's uniform grid
//! and dense attention, or a FastH3 student's distilled rungs or parallel-decoding blocks and its
//! sparse attention, which its `fastvideo_inference.json` contract states
//! ([`crate::checkpoint::Contract`]).
//!
//! The values here are the *statement*: how many sigma points, which rungs, which tile rows. The
//! numerical schedule the solver steps through - the FP32 sigmas and network times - is computed
//! from it by the schedule module, and the solver updates latents with it.

use nn::VideoTask;
use serde_json::{Map, Value};

use super::canvas::{Canvas, DMD_CANVASES, NAMED_CANVASES};
use super::field;
use crate::{Error, Result};

/// The video noise level shift and the audio one, as a schedule states them.
///
/// The two modalities run on their own noise levels over the same step count: video and audio
/// must evaluate the same number of times, which the schedule builder checks.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Shifts {
    /// The video schedule's shift `s` in `sigma = s u / (1 + (s - 1) u)`.
    pub video: f32,
    /// The audio schedule's shift.
    pub audio: f32,
}

/// The sigma grid a checkpoint states.
#[derive(Clone, PartialEq, Debug)]
pub enum ScheduleGrid {
    /// The released uniform grid: `points` sigma points of a descending `linspace(1, 0, points)`,
    /// each shifted by its modality's shift, deduplicated, of which the network evaluates every
    /// point but the clean endpoint.
    Uniform {
        /// Sigma points, the clean endpoint included.
        points: u32,
    },
    /// A DMD student's trained rungs: unshifted noise levels of the training schedule, shifted
    /// once per modality, with the clean endpoint after the last rung.
    Dmd {
        /// The rungs, strictly decreasing and within the training schedule.
        rungs: Vec<u32>,
        /// The training schedule's step count the rungs are stated on.
        num_train_timesteps: f32,
    },
    /// A parallel-decoding student's block grid: one network evaluation per block of the fine
    /// grid, each block's heads fused into one prediction.
    Blocks {
        /// The fine grid's interval count, which is the DiT's output head count.
        intervals: u32,
        /// Fine-grid indices of the block boundaries, strictly increasing from zero to
        /// `intervals`.
        nodes: Vec<u32>,
        /// The fixed point of both modality shifts, in `(0, 1]`.
        max_t: f32,
    },
}

/// The schedule a checkpoint states: its grid and its two modality shifts.
#[derive(Clone, PartialEq, Debug)]
pub struct ScheduleSpec {
    /// The grid the checkpoint states.
    pub grid: ScheduleGrid,
    /// The video and audio shifts.
    pub shifts: Shifts,
}

impl ScheduleGrid {
    /// The grid's name, as a checkpoint states it: `uniform`, `dmd` or `blocks`.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Uniform { .. } => "uniform",
            Self::Dmd { .. } => "dmd",
            Self::Blocks { .. } => "blocks",
        }
    }
}

impl ScheduleSpec {
    /// The released uniform grid.
    pub fn uniform(points: u32, shifts: Shifts) -> Result<Self> {
        if points < 2 {
            return Err(Error::config(format!(
                "an H3 uniform grid needs at least two sigma points, got {points}"
            )));
        }
        Ok(Self {
            grid: ScheduleGrid::Uniform { points },
            shifts,
        })
    }

    /// A DMD student's ladder of rungs on its training schedule.
    pub fn dmd(rungs: Vec<u32>, num_train_timesteps: f32, shifts: Shifts) -> Result<Self> {
        if rungs.is_empty() {
            return Err(Error::config(
                "an H3 DMD ladder needs at least one rung".to_owned(),
            ));
        }
        if !num_train_timesteps.is_finite() || num_train_timesteps <= 0.0 {
            return Err(Error::config(format!(
                "an H3 DMD training schedule must be finite and positive, got \
                 {num_train_timesteps}"
            )));
        }
        if rungs
            .iter()
            .any(|rung| *rung == 0 || *rung as f32 > num_train_timesteps)
        {
            return Err(Error::config(format!(
                "H3 DMD rungs must lie within the {num_train_timesteps}-step training schedule"
            )));
        }
        if rungs.windows(2).any(|pair| pair[0] <= pair[1]) {
            return Err(Error::config(
                "H3 DMD rungs must strictly decrease".to_owned(),
            ));
        }
        Ok(Self {
            grid: ScheduleGrid::Dmd {
                rungs,
                num_train_timesteps,
            },
            shifts,
        })
    }

    /// A parallel-decoding student's block grid.
    pub fn blocks(intervals: u32, nodes: Vec<u32>, max_t: f32, shifts: Shifts) -> Result<Self> {
        if intervals < 2 {
            return Err(Error::config(format!(
                "an H3 parallel-decoding grid needs at least two intervals, got {intervals}"
            )));
        }
        if nodes.len() < 2 || nodes[0] != 0 || nodes.last() != Some(&intervals) {
            return Err(Error::config(format!(
                "H3 parallel-decoding nodes must run from 0 to the {intervals} intervals"
            )));
        }
        if nodes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::config(
                "H3 parallel-decoding nodes must strictly increase".to_owned(),
            ));
        }
        if !max_t.is_finite() || max_t <= 0.0 || max_t > 1.0 {
            return Err(Error::config(format!(
                "H3 parallel-decoding grid_max_t must lie in (0, 1], got {max_t}"
            )));
        }
        Ok(Self {
            grid: ScheduleGrid::Blocks {
                intervals,
                nodes,
                max_t,
            },
            shifts,
        })
    }

    /// The number of network evaluations a step loop runs.
    ///
    /// The uniform grid evaluates its sigma points but the clean endpoint; both students evaluate
    /// one step per stated rung or block. The layer still reads the actual count from the computed
    /// schedule, which deduplicates collapsed FP32 values; this is the count a checkpoint states.
    pub fn steps(&self) -> u32 {
        match &self.grid {
            ScheduleGrid::Uniform { points } => points - 1,
            ScheduleGrid::Dmd { rungs, .. } => rungs.len() as u32,
            ScheduleGrid::Blocks { nodes, .. } => (nodes.len() - 1) as u32,
        }
    }
}

/// How the DiT attends over the packed sequence.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AttentionKind {
    /// Full bidirectional attention over every packed row.
    Dense,
    /// VSA sparse attention over spatiotemporal tiles of the generated video.
    Sparse {
        /// Rows of one tile, one video tile shape of the segment packing.
        tile: u32,
        /// The fraction of key tiles a generated-video query tile does not read.
        sparsity: f64,
        /// The fraction of its own key tiles a reference-video query keeps, when the network has
        /// reference videos.
        reference_keep: Option<f64>,
    },
}

/// The rows of one sparse attention tile, per tile shape: `tile -> (video frames, rows, columns)`.
pub const SPARSE_TILE_ROWS: [(u32, (u32, u32, u32)); 2] = [(64, (4, 4, 4)), (128, (4, 4, 8))];

impl AttentionKind {
    /// Dense attention, as the base release attends.
    pub const DENSE: Self = Self::Dense;

    /// Sparse attention over `tile`-row tiles.
    pub fn sparse(tile: u32, sparsity: f64, reference_keep: Option<f64>) -> Result<Self> {
        if !SPARSE_TILE_ROWS.iter().any(|(rows, _)| *rows == tile) {
            return Err(Error::config(format!(
                "H3 sparse attention tiles hold {} rows, got {tile}",
                SPARSE_TILE_ROWS
                    .iter()
                    .map(|(rows, _)| rows.to_string())
                    .collect::<Vec<_>>()
                    .join(" or ")
            )));
        }
        if !sparsity.is_finite() || !(0.0..1.0).contains(&sparsity) {
            return Err(Error::config(format!(
                "H3 VSA sparsity must lie in [0, 1), got {sparsity}"
            )));
        }
        if let Some(keep) = reference_keep
            && (!keep.is_finite() || !(0.0 < keep && keep <= 1.0))
        {
            return Err(Error::config(format!(
                "H3 VSA reference keep rate must lie in (0, 1], got {keep}"
            )));
        }
        Ok(Self::Sparse {
            tile,
            sparsity,
            reference_keep,
        })
    }

    /// Whether this network attends densely, which the packing must know when it lays rows out.
    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }

    /// The rows of one text and condition capacity unit of a layout: a sparse tile, or the dense
    /// packing's 64 rows.
    pub const fn capacity_rows(&self) -> u32 {
        match self {
            Self::Dense => 64,
            Self::Sparse { tile, .. } => *tile,
        }
    }
}

/// The dimensions of one diffusion transformer.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TransformerConfig {
    /// The transformer's width.
    pub hidden_size: u32,
    /// Attention heads.
    pub num_attention_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Dilated transformer layers.
    pub num_hidden_layers: u32,
    /// Layers of the token refiner that conditions the prompt.
    pub num_refiner_layers: u32,
    /// The feed-forward width.
    pub intermediate_size: u32,
    /// Latent channels of the video stream.
    pub video_channels: u32,
    /// Latent channels of the audio stream.
    pub audio_channels: u32,
    /// Width of the text features the DiT conditions on.
    pub text_dim: u32,
    /// Width of the timestep frequency embedding.
    pub frequency_dim: u32,
    /// Hidden width of the timestep embedding.
    pub time_hidden_dim: u32,
    /// Width of the modulation rows the timestep embedding produces.
    pub time_dim: u32,
    /// Frequency count of one rotary axis.
    pub rope_frequency_dim: u32,
    /// The rotary base.
    pub rope_theta: f32,
    /// The normalization epsilon.
    pub norm_eps: f32,
    /// The epsilon of the query/key normalization inside attention.
    pub qk_norm_eps: f32,
    /// Output heads: 1 for an ordinary DiT, the fine-grid interval count for a parallel-decoding
    /// student, whose output projections hold one head-major prediction per interval.
    pub output_heads: u32,
}

impl TransformerConfig {
    /// Read a DiT's `config.json`.
    ///
    /// `pdd_steps`, when the checkpoint states it, is the parallel-decoding student's interval
    /// count and so its output head count; an ordinary DiT has one output head.
    pub fn parse(values: &Map<String, Value>, owner: &str) -> Result<Self> {
        let output_heads = match field::optional_index(values, "pdd_steps", owner)? {
            None => 1,
            Some(0) => {
                return Err(Error::config(format!(
                    "{owner}.pdd_steps must be a positive integer"
                )));
            }
            Some(heads) if heads < 2 => {
                return Err(Error::config(
                    "unsupported MiniMax-H3 checkpoint: pdd_steps above one is what makes a \
                     parallel-decoding student, and it must state more than one interval"
                        .to_owned(),
                ));
            }
            Some(heads) => heads,
        };
        // The packing and the reconstruction hard-code the patch: 16x spatial and 4x temporal
        // compression applied to one latent frame at a time.
        let patch = field::counts(values, "patch_size", owner)?;
        if patch != [1, 2, 2] {
            return Err(Error::config(format!(
                "MiniMax-H3 transformer patch_size must be [1, 2, 2], got {patch:?}"
            )));
        }
        let norm_eps = field::positive(values, "norm_eps", owner)?;
        let final_norm_eps = field::positive(values, "final_norm_eps", owner)?;
        if final_norm_eps != norm_eps {
            return Err(Error::config(
                "MiniMax-H3 transformer final_norm_eps must equal norm_eps".to_owned(),
            ));
        }
        let config = Self {
            hidden_size: field::count(values, "hidden_size", owner)?,
            num_attention_heads: field::count(values, "num_attention_heads", owner)?,
            head_dim: field::count(values, "attention_head_dim", owner)?,
            num_hidden_layers: field::count(values, "num_layers", owner)?,
            num_refiner_layers: field::count(values, "num_refiner_layers", owner)?,
            intermediate_size: field::count(values, "ffn_dim", owner)?,
            video_channels: field::count(values, "in_channels", owner)?,
            audio_channels: field::count(values, "audio_in_channels", owner)?,
            text_dim: field::count(values, "text_dim", owner)?,
            frequency_dim: field::count(values, "freq_dim", owner)?,
            time_hidden_dim: field::count(values, "time_embed_hidden_dim", owner)?,
            time_dim: field::count(values, "time_embed_dim", owner)?,
            rope_frequency_dim: field::count(values, "rope_freq_dim", owner)?,
            rope_theta: field::positive(values, "rope_theta", owner)? as f32,
            norm_eps: norm_eps as f32,
            qk_norm_eps: field::positive(values, "qk_norm_eps", owner)? as f32,
            output_heads,
        };
        // Each of the three rotary axes (time, height, width) rotates 2 * rope_frequency_dim
        // channels of a head; the rest stay unrotated.
        if !config.rope_frequency_dim.is_multiple_of(2)
            || config.rope_frequency_dim * 6 > config.head_dim
        {
            return Err(Error::config(format!(
                "H3 transformer rope_freq_dim must be even and rotate at most a head's width, \
                 got {} of {}",
                config.rope_frequency_dim, config.head_dim
            )));
        }
        Ok(config)
    }

    /// The projection width of one attention head times the head count.
    pub const fn attention_width(&self) -> u32 {
        self.num_attention_heads * self.head_dim
    }
}

/// One DiT of the checkpoint: its network, its schedule, its attention and the tasks it serves.
#[derive(Clone, PartialEq, Debug)]
pub struct DenoiserConfig {
    /// The network's dimensions.
    pub transformer: TransformerConfig,
    /// The schedule the checkpoint states for it.
    pub schedule: ScheduleSpec,
    /// How it attends.
    pub attention: AttentionKind,
    /// The tasks it serves, in canonical order.
    pub tasks: Vec<VideoTask>,
    /// The canvases it generates, or `None` when it generates the canvas rule's canvas of every
    /// named ratio at the class a request states.
    ///
    /// A DMD student generates the twelve buckets it was trained on; the base release and an
    /// OmniRef student follow the canvas rule.
    pub canvases: Option<&'static [Canvas]>,
}

impl DenoiserConfig {
    /// The canvases a deployment may prepare layouts for.
    ///
    /// A checkpoint that generates only some canvases offers those; one that follows the canvas
    /// rule offers every named aspect ratio's canvas at the class's short edge.
    pub fn offered_canvases(&self) -> &'static [Canvas] {
        self.canvases.unwrap_or(&NAMED_CANVASES)
    }

    /// One DiT with the schedule and attention the checkpoint states for it.
    pub fn new(
        transformer: TransformerConfig,
        schedule: ScheduleSpec,
        attention: AttentionKind,
        tasks: Vec<VideoTask>,
    ) -> Result<Self> {
        if tasks.is_empty() {
            return Err(Error::config(
                "an H3 denoiser serves at least one task".to_owned(),
            ));
        }
        if tasks
            .iter()
            .enumerate()
            .any(|(index, task)| tasks[..index].contains(task))
        {
            return Err(Error::config(
                "an H3 denoiser's tasks must be distinct".to_owned(),
            ));
        }
        // A parallel-decoding student has one output head per fine-grid interval, and only one
        // does: the two statements must agree.
        let intervals = match &schedule.grid {
            ScheduleGrid::Blocks { intervals, .. } => Some(*intervals),
            _ => None,
        };
        if intervals != (transformer.output_heads > 1).then_some(transformer.output_heads) {
            return Err(Error::config(format!(
                "H3 output heads must equal the parallel-decoding interval count, and only a \
                 parallel-decoding student has more than one: {} heads, {intervals:?} intervals",
                transformer.output_heads
            )));
        }
        let canvases =
            matches!(schedule.grid, ScheduleGrid::Dmd { .. }).then_some(&DMD_CANVASES[..]);
        Ok(Self {
            transformer,
            schedule,
            attention,
            tasks,
            canvases,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transformer() -> Map<String, Value> {
        let values: Value = serde_json::from_str(
            r#"{
                "num_attention_heads": 56,
                "attention_head_dim": 128,
                "hidden_size": 5376,
                "num_layers": 50,
                "num_refiner_layers": 2,
                "ffn_dim": 14336,
                "in_channels": 24,
                "audio_in_channels": 32,
                "patch_size": [1, 2, 2],
                "text_dim": 5120,
                "freq_dim": 256,
                "time_embed_hidden_dim": 5376,
                "time_embed_dim": 2688,
                "rope_freq_dim": 16,
                "rope_theta": 10000.0,
                "norm_eps": 1e-05,
                "qk_norm_eps": 1e-05,
                "final_norm_eps": 1e-05
            }"#,
        )
        .expect("the fixture is JSON");
        values.as_object().expect("an object").clone()
    }

    fn shifts() -> Shifts {
        Shifts {
            video: 12.0,
            audio: 3.0,
        }
    }

    #[test]
    fn a_transformer_reads_the_released_dimensions() {
        let config = TransformerConfig::parse(&transformer(), "test transformer").expect("parsed");
        assert_eq!(config.hidden_size, 5376);
        assert_eq!(config.num_hidden_layers, 50);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.attention_width(), 7168);
        assert_eq!(config.output_heads, 1);
        assert_eq!(config.rope_theta, 10000.0);
    }

    #[test]
    fn a_transformer_refuses_an_unsupported_patch_or_a_mismatched_final_norm() {
        let mut patched = transformer();
        patched.insert("patch_size".into(), serde_json::json!([1, 4, 4]));
        assert!(TransformerConfig::parse(&patched, "test").is_err());

        let mut norm = transformer();
        norm.insert("final_norm_eps".into(), serde_json::json!(1e-6));
        let error = TransformerConfig::parse(&norm, "test").expect_err("mismatched epsilons");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("final_norm_eps")),
            "{error}"
        );
    }

    #[test]
    fn a_transformer_refuses_rotary_dimensions_that_do_not_fit_a_head() {
        let mut values = transformer();
        values.insert("rope_freq_dim".into(), serde_json::json!(32));
        assert!(TransformerConfig::parse(&values, "test").is_err());
        values.insert("rope_freq_dim".into(), serde_json::json!(15));
        assert!(TransformerConfig::parse(&values, "test").is_err());
    }

    #[test]
    fn a_parallel_decoding_transformer_states_its_output_heads() {
        let mut values = transformer();
        values.insert("pdd_steps".into(), serde_json::json!(32));
        let config = TransformerConfig::parse(&values, "test").expect("parsed");
        assert_eq!(config.output_heads, 32);
    }

    #[test]
    fn a_dmd_ladder_must_strictly_decrease_within_the_training_schedule() {
        let spec = ScheduleSpec::dmd(
            vec![999, 874, 749, 624, 500, 375, 250, 125],
            super::super::NUM_TRAIN_TIMESTEPS,
            shifts(),
        )
        .expect("the released ladder");
        assert_eq!(spec.steps(), 8);

        for rungs in [vec![999, 999], vec![125, 250], vec![1001, 500], vec![0]] {
            assert!(
                ScheduleSpec::dmd(rungs.clone(), super::super::NUM_TRAIN_TIMESTEPS, shifts())
                    .is_err(),
                "{rungs:?} is not a strict descending ladder within the training schedule"
            );
        }
    }

    #[test]
    fn a_block_grid_runs_from_zero_to_its_interval_count() {
        let spec = ScheduleSpec::blocks(32, vec![0, 4, 8, 12, 16, 20, 24, 28, 32], 0.999, shifts())
            .expect("the OmniRef grid");
        assert_eq!(spec.steps(), 8);

        assert!(ScheduleSpec::blocks(32, vec![0, 8, 32], 0.999, shifts()).is_ok());
        for nodes in [vec![0], vec![4, 32], vec![0, 8, 8, 32], vec![0, 8, 16]] {
            assert!(
                ScheduleSpec::blocks(32, nodes.clone(), 0.999, shifts()).is_err(),
                "{nodes:?} is not a block grid"
            );
        }
        assert!(ScheduleSpec::blocks(32, vec![0, 32], 0.0, shifts()).is_err());
        assert!(ScheduleSpec::blocks(32, vec![0, 32], 1.5, shifts()).is_err());
    }

    #[test]
    fn a_uniform_grid_needs_two_points() {
        assert!(ScheduleSpec::uniform(50, shifts()).is_ok());
        assert_eq!(
            ScheduleSpec::uniform(50, shifts()).expect("grid").steps(),
            49
        );
        assert!(ScheduleSpec::uniform(1, shifts()).is_err());
    }

    #[test]
    fn a_dmd_denoiser_generates_the_twelve_trained_canvases_and_others_the_rule() {
        let transformer = TransformerConfig::parse(&transformer(), "test").expect("parsed");
        let dmd = DenoiserConfig::new(
            transformer,
            ScheduleSpec::dmd(vec![999, 125], super::super::NUM_TRAIN_TIMESTEPS, shifts())
                .expect("ladder"),
            AttentionKind::DENSE,
            vec![
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
            ],
        )
        .expect("a DMD denoiser");
        assert_eq!(dmd.canvases, Some(&DMD_CANVASES[..]));
    }

    #[test]
    fn a_denoiser_refuses_tasks_that_repeat_and_heads_that_disagree_with_its_grid() {
        let transformer = TransformerConfig::parse(&transformer(), "test").expect("parsed");
        let error = DenoiserConfig::new(
            transformer,
            ScheduleSpec::uniform(50, shifts()).expect("grid"),
            AttentionKind::DENSE,
            vec![VideoTask::TextToVideoAudio, VideoTask::TextToVideoAudio],
        )
        .expect_err("a repeated task is refused");
        assert!(matches!(error, Error::Config(ref problem) if problem.contains("distinct")));

        // A parallel-decoding grid without the student's output heads, and one whose interval
        // count differs from them, are both refused.
        let error = DenoiserConfig::new(
            transformer,
            ScheduleSpec::blocks(32, vec![0, 32], 0.999, shifts()).expect("grid"),
            AttentionKind::DENSE,
            vec![VideoTask::ReferenceToVideoAudio],
        )
        .expect_err("an ordinary DiT has one head");
        assert!(matches!(error, Error::Config(ref problem) if problem.contains("output heads")));
    }

    #[test]
    fn sparse_attention_keeps_only_the_tile_shapes_and_ranges_it_serves() {
        assert!(AttentionKind::sparse(64, 0.8, None).is_ok());
        assert!(AttentionKind::sparse(128, 0.9, Some(0.1)).is_ok());
        assert!(AttentionKind::sparse(96, 0.5, None).is_err());
        assert!(AttentionKind::sparse(64, 1.0, None).is_err());
        assert!(AttentionKind::sparse(64, -0.1, None).is_err());
        assert!(AttentionKind::sparse(128, 0.5, Some(0.0)).is_err());
        assert_eq!(AttentionKind::DENSE.capacity_rows(), 64);
        assert_eq!(
            AttentionKind::sparse(128, 0.5, None)
                .expect("sparse")
                .capacity_rows(),
            128
        );
    }
}
