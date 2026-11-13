//! The transformer's timestep conditioning: the levels a step's rows are conditioned on, and
//! the projections that turn each level into a block's modulation.
//!
//! Every block is conditioned on the step's noise levels through six affine vectors per row
//! (shift, scale and gate of the attention, then of the feed-forward), and the final norm
//! through two. The network computes them with one shared timestep embedding and one
//! projection per block:
//!
//! ```text
//! t_emb   = time_embedder(sinusoidal(level))                    FP32
//! block   = adaln_proj.linear(silu(t_emb))    [18 · hidden]     the stack's type
//! output  = norm_out.linear(silu(t_emb))      [2 · hidden]
//! ```
//!
//! (`refs/vllm-omni/…/minimax_h3/minimax_h3_transformer.py:366-407`,
//! `minimax_h3_blocks.py:710-745`). The schedule is fixed, so the products are computed once
//! for the schedule's distinct levels into an [`nn::modules::TimestepTable`] and gathered per
//! step, as the reference derives its AdaLN tables at load.
//!
//! ## The row layout
//!
//! A step reads one modulation row per `(group, tag)`: the group is the level the row is
//! conditioned on ([`crate::minimax_h3::packing`]'s `GROUP_*`) and the tag the modality it belongs to
//! (`TAG_*`). A block's table row holds the three tags' six vectors each, so the gathered
//! `[groups, 18 · hidden]` rows read as `[3 · groups, 6 · hidden]` and a packed row's
//! modulation row is `3 · group + tag`.

use nn::modules::{
    Linear, LinearConfig, Parallel, TimestepEmbedding, TimestepEmbeddingConfig, TimestepTable,
};
use nn::{DType, Device, Tensor, VarBuilder, functional};

use crate::minimax_h3::config::{ScheduleConfig, TransformerConfig};
use crate::minimax_h3::{Error, Result};

/// The level a visual condition is anchored at, `t · clean + (1 − t) · noise`, and the level
/// its timestep group sits at until the generated video is cleaner than it
/// (`refs/vllm-omni/…/minimax_h3/pipeline_minimax_h3.py:233,394`).
pub const VISUAL_CONDITION_TIME: f32 = 0.999;

/// The level an audio reference sits at: its latent unchanged, so nothing is mixed in
/// (`pipeline_minimax_h3.py:234,395`).
pub const AUDIO_CONDITION_TIME: f32 = 1.0;

/// The modulation columns of one block's table row: three tags of six vectors, in hidden
/// widths.
pub const BLOCK_COLUMNS: u32 = 18;

/// The modulation columns of the final norm's table row: a shift and a scale, in hidden
/// widths.
pub const OUTPUT_COLUMNS: u32 = 2;

/// The timestep groups a step reads: the generated video and audio, and for a `conditioned`
/// network also a visual condition's level and an audio reference's.
pub const fn groups(conditioned: bool) -> usize {
    if conditioned { 4 } else { 2 }
}

/// The level of every group of every step of `schedule`: `[steps, groups]` FP32 on the CPU
/// device, in the order of the `GROUP_*` constants.
///
/// A level is the network time `1 − sigma`, in FP32. The generated video and audio have their
/// own level per step; a visual condition's is the video's floored at
/// [`VISUAL_CONDITION_TIME`], and an audio reference is clean at every step. Levels that
/// coincide share one table entry.
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn levels(schedule: &ScheduleConfig, conditioned: bool) -> Result<Tensor> {
    let steps = schedule.num_steps() as usize;
    let time = |sigmas: &[f32]| -> Result<Tensor> {
        Ok(Tensor::new(&sigmas[..steps], &Device::Cpu)?.affine(-1.0, 1.0)?)
    };
    let (video, audio) = (time(&schedule.video_sigmas)?, time(&schedule.audio_sigmas)?);

    let mut columns = vec![video.clone(), audio];
    if conditioned {
        columns.push(video.maximum(f64::from(VISUAL_CONDITION_TIME))?);
        columns.push(Tensor::full(AUDIO_CONDITION_TIME, steps, &Device::Cpu)?);
    }

    Ok(Tensor::stack(&columns, 1)?)
}

/// The projections the modulation tables are computed with: the shared timestep embedding, one
/// modulation projection per block and the final norm's.
pub struct Modulation {
    /// `time_embedder`: FP32 at every preset.
    time: TimestepEmbedding,
    /// One `transformer_blocks.<i>.adaln_proj.linear` per block, `[18 · hidden, time_dim]`.
    blocks: Vec<Linear>,
    /// `norm_out.linear`, `[2 · hidden, time_dim]`.
    output: Linear,
}

impl Modulation {
    /// The projections the checkpoint at `vb` states.
    ///
    /// The element type of each is the reference's: the timestep pair is FP32 and the
    /// modulation projections are the stack's, and the activation crosses between them once
    /// (`minimax_h3_blocks.py:713-716`).
    ///
    /// # Errors
    ///
    /// A weight the loader refuses.
    pub fn load(config: &TransformerConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let projection = |vb: VarBuilder<'_, '_>, columns: u32| {
            Linear::load(
                &LinearConfig {
                    inputs: config.time_dim,
                    outputs: columns * config.hidden_size,
                    bias: true,
                    parallel: Parallel::None,
                },
                vb,
            )
        };

        Ok(Self {
            time: TimestepEmbedding::load(
                &TimestepEmbeddingConfig {
                    frequency_dim: config.frequency_dim,
                    hidden: config.time_hidden_dim,
                    out: config.time_dim,
                },
                vb.pp("time_embedder").to_dtype(DType::F32),
            )?,
            blocks: (0..config.num_hidden_layers)
                .map(|index| {
                    let block = vb.pp(crate::minimax_h3::transformer::BLOCKS).pp(index);
                    projection(block.pp("adaln_proj").pp("linear"), BLOCK_COLUMNS)
                })
                .collect::<nn::Result<Vec<_>>>()?,
            output: projection(vb.pp("norm_out").pp("linear"), OUTPUT_COLUMNS)?,
        })
    }

    /// The element type the modulation projections compute in: the tables' type.
    pub fn dtype(&self) -> DType {
        self.output.weight().dtype()
    }

    /// Fills `table`'s block and output tables from its embedding, `[entries, frequency_dim]`
    /// FP32: each block's `[entries, 18 · hidden]` and the final norm's `[entries, 2 · hidden]`.
    ///
    /// The timestep pair computes in FP32 and the projections in their weights' type: the
    /// activation `silu(t_emb)` rounds into it once.
    ///
    /// # Errors
    ///
    /// Tables of another element type than the projections', or what a module refuses.
    pub fn fill(&self, table: &TimestepTable) -> Result<()> {
        let dtype = self.dtype();
        if table.output_table().dtype() != dtype {
            return Err(Error::config(format!(
                "the modulation tables are {:?} and the projections' weights {dtype:?}",
                table.output_table().dtype()
            )));
        }

        let time = self.time.forward(table.frequencies())?;
        let activated = functional::silu(&time)?.to_dtype(dtype)?;

        for (index, block) in self.blocks.iter().enumerate() {
            table
                .layer_table(index)?
                .slice_set(&block.forward(&activated)?, 0, 0)?;
        }
        Ok(table
            .output_table()
            .slice_set(&self.output.forward(&activated)?, 0, 0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_generated_step_reads_the_video_and_audio_levels() {
        let schedule = ScheduleConfig::uniform(5, 1.0, 1.0).unwrap();
        let levels = levels(&schedule, false).unwrap();

        // One row per evaluation: the clean endpoint is no evaluation.
        assert_eq!(levels.dims(), [4, groups(false)]);
        assert_eq!(
            levels.to_vec2::<f32>().unwrap(),
            [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]
        );
    }

    #[test]
    fn a_conditioned_step_reads_four_levels() {
        // A visual condition sits at the generated video's level until that level is cleaner
        // than its anchor, and an audio reference is clean at every step.
        let schedule = ScheduleConfig::ladder(&[1000, 500, 1], 1000.0, 1.0, 1.0).unwrap();
        let levels = levels(&schedule, true).unwrap().to_vec2::<f32>().unwrap();

        assert_eq!(levels[0], [0.0, 0.0, VISUAL_CONDITION_TIME, 1.0]);
        assert_eq!(levels[1], [0.5, 0.5, VISUAL_CONDITION_TIME, 1.0]);
        assert_eq!(levels[2][0], 1.0 - 0.001_f32);
        assert_eq!(levels[2][2], VISUAL_CONDITION_TIME.max(1.0 - 0.001_f32));
    }
}
