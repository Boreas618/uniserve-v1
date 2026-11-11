//! The diffusion transformer's timestep conditioning: the H3 row layout of its modulation and the
//! projections that compute it.
//!
//! Every layer is conditioned on the step's noise levels through six affine vectors per token -
//! shift, scale and gate for the attention block and for the feed-forward block - and the final
//! normalization carries two more. The released network computes them with one shared timestep
//! embedding and one projection per layer, and the schedule is fixed, so the products are
//! precomputed once for the schedule's distinct timesteps in an [`nn::diffusion::TimestepTable`].
//! The table is computed at load on the device: one `Linear(frequency_dim → time_hidden_dim)`, a
//! SiLU, one `Linear(time_hidden_dim → time_dim)` and then, per layer, one `Linear(time_dim → 18 ·
//! hidden_size)` - about 260 MMAC per layer per distinct timestep, which no host-side boundary can
//! absorb (`crate::Solver`'s note has the arithmetic).
//!
//! ## The row layout
//!
//! A step reads one modulation row per `(group, tag)`: the group is the timestep the token is
//! conditioned on (the generated video's, the generated audio's, a visual condition's, an audio
//! reference's) and the tag is the modality the token belongs to (video 0, text 1, audio 2). A
//! layer's table row holds the three tags' six vectors each, so the gathered `[groups, 18 · hidden]`
//! rows read as `[3 · groups, 6 · hidden]` and a token's row is `3 · group + tag`. The worker's
//! per-request `modulation_indices` table holds that value for every packed row, and the layers of
//! [`crate::transformer`] index with it.

use nn::diffusion::TimestepTable;
use nn::layers::{Bias, Linear, Parallel};
use nn::timestep::{TimestepEmbedding, TimestepSpec};
use nn::weights::WeightLoader;
use nn::{DType, Forward};

use crate::{Error, Result};

/// The six affine vectors one layer reads per token: shift, scale and gate of the attention block,
/// then of the feed-forward block.
pub const VECTORS: usize = 6;

/// The three token tags a timestep group's vectors are stored for, in row order.
pub const TAGS: u32 = 3;

/// How many timestep groups a step reads: the generated video and audio for a text-only denoiser
/// (`t2va`), and for a `conditioned` one (`fl2va`, `ref2va`) also a visual condition's clean level
/// and an audio reference's clean level.
pub const fn groups(conditioned: bool) -> u32 {
    if conditioned { 4 } else { 2 }
}

/// The affine row a token selects inside one step's modulation matrix: `3 · group + tag`.
///
/// This is the value the worker writes into the per-request `modulation_indices` table, and the
/// only request-dependent input of the conditioning.
pub const fn row(group: u32, tag: u32) -> u32 {
    TAGS * group + tag
}

/// The noise level of every group of every step, step-major: the levels the timestep table holds.
///
/// The generated video and audio have their own level per step; the visual condition's level is
/// the video's floored at the clean level the condition is anchored at
/// ([`crate::denoiser::VISUAL_CONDITION_TIME`]), and an audio reference is clean at the top of the
/// schedule. Levels that coincide share one table entry
/// ([`nn::diffusion::TimestepTable::entries`]), which is where the table's saving comes from - the
/// schedule's points repeat across the two modalities and the condition groups.
pub fn levels(video: &[f32], audio: &[f32], conditioned: bool) -> Result<Vec<f32>> {
    if video.is_empty() || video.len() != audio.len() {
        return Err(Error::checkpoint(format!(
            "{} video and {} audio noise levels are not one schedule's",
            video.len(),
            audio.len()
        )));
    }
    let mut levels = Vec::with_capacity(video.len() * groups(conditioned) as usize);
    for (video, audio) in video.iter().zip(audio) {
        levels.extend([*video, *audio]);
        if conditioned {
            // A visual condition sits at the generated video's level until that level is cleaner
            // than its anchor, and an audio reference is always clean.
            levels.extend([
                video.max(crate::denoiser::VISUAL_CONDITION_TIME),
                crate::denoiser::AUDIO_CONDITION_TIME,
            ]);
        }
    }
    Ok(levels)
}

/// The projections the AdaLN tables are computed with: the shared timestep embedding, one
/// modulation projection per layer and the final norm's.
pub struct Projections {
    /// `time_embedder`: `linear_1` `[time_hidden, frequency]` and `linear_2` `[time_dim,
    /// time_hidden]`, FP32, with their biases.
    time: TimestepEmbedding,
    /// One `adaln_proj.linear` per layer: `[18 * hidden, time_dim]`, with its bias.
    layers: Vec<Linear>,
    /// `norm_out.linear`: `[2 * hidden, time_dim]`, with its bias.
    output: Linear,
    frequency_dim: u32,
}

impl Projections {
    /// Read the cache's projections out of a DiT partition's checkpoint.
    pub fn load(
        config: &crate::config::TransformerConfig,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        // **The element type of each tensor is pinned by the reference, not by the deployment.**
        // Its loading table fixes the timestep pair at FP32 and the modulation projections at BF16
        // whatever the preset, and it casts the activation to the weight's
        // type before each product. Our product has one element type for both
        // operands, so the activation and the tables must agree with the weights on both sides - and
        // the run's first stop was exactly this: an FP32 activation against whatever type the
        // deployment's loader happened to publish.
        //
        // So the timestep pair is pinned FP32 and the modulation projections take the deployment's
        // type (`None`: the loader's own policy), which is the stack's - the pair differ exactly as
        // the reference's do. `compute` then crosses between them **once**, narrowing the timestep
        // activation into the modulation type; the tables follow the projections, so a block's
        // modulation and its activation are one type, and at a deployment that states FP32 the two
        // halves coincide and the crossing is skipped.
        let hidden = config.hidden_size;
        let rows = VECTORS as u32 * hidden * TAGS;
        // A modulation projection, read at the deployment's type.
        let projection = |loader: &mut WeightLoader<'_>, name: &str, outputs: u32| {
            Linear::load(
                &mut loader.scope(name),
                config.time_dim,
                outputs,
                Bias::Present,
                Parallel::None,
            )
        };
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                projection(
                    loader,
                    &format!("transformer_blocks.{index}.adaln_proj.linear"),
                    rows,
                )
            })
            .collect::<nn::Result<Vec<_>>>()?;
        Ok(Self {
            time: TimestepEmbedding::load(
                &mut loader.scope("time_embedder").pinned(DType::F32),
                TimestepSpec {
                    frequency_dim: config.frequency_dim,
                    hidden: config.time_hidden_dim,
                    out: config.time_dim,
                },
            )?,
            layers,
            output: projection(loader, "norm_out.linear", 2 * hidden)?,
            frequency_dim: config.frequency_dim,
        })
    }

    /// Fill `table`'s block and output tables from its embedding, `[entries, frequency_dim]` FP32.
    ///
    /// Each block table is `[entries, 18 * hidden]` and the output table `[entries, 2 * hidden]`:
    /// the tables every step gathers rows from. They are the table's writable constants, written
    /// once after the load ([`nn::capability::Denoiser::initialize`]); the levels they are computed
    /// at are the schedule's, which do not change per request.
    ///
    /// The timestep pair computes in FP32 and the modulation projections in their weights' type,
    /// which is the tables' type: the activation crosses between them once, as the reference's
    /// `F.linear(activated_timesteps.to(weight.dtype), weight, bias)` does.
    pub fn compute(&self, f: &Forward, table: &TimestepTable) -> Result<()> {
        let embedding = table.frequencies();
        if embedding.shape.rank() != 2 || embedding.shape.last() != u64::from(self.frequency_dim) {
            return Err(Error::config(format!(
                "an AdaLN embedding of {:?} is not [entries, {}]",
                embedding.shape.dims(),
                self.frequency_dim
            )));
        }
        let time_type = self.time.dtype();
        if time_type != DType::F32 || embedding.dtype != time_type {
            return Err(Error::config(format!(
                "the AdaLN timestep pair computes in {time_type} over a {} embedding; the \
                 reference pins both to FP32",
                embedding.dtype
            )));
        }
        let dtype = self.dtype();
        let output_table = table.output_table();
        if output_table.dtype != dtype {
            return Err(Error::config(format!(
                "the AdaLN tables are {} and the projections' weights are {dtype}",
                output_table.dtype
            )));
        }

        // `silu(time_embedder(embedding))`, narrowed once into the projections' type.
        let time = self.time.forward(f, embedding)?;
        let activated = nn::layers::Silu.forward(f, time)?;
        let narrow = if dtype == time_type {
            activated
        } else {
            nn::layers::Cast.forward(f, activated, dtype)?
        };

        // Each projection writes its table: computed into the arena, then stored into the
        // table's own storage (`inout ← 1 · product`).
        for (index, layer) in self.layers.iter().enumerate() {
            let product = layer.forward(f, narrow)?;
            nn::layers::Axpby.forward(f, table.layer_table(index)?, product, 1.0, 0.0)?;
        }
        let product = self.output.forward(f, narrow)?;
        Ok(nn::layers::Axpby.forward(f, output_table, product, 1.0, 0.0)?)
    }

    /// The element type the modulation projections read: the tables' type, and the cache's.
    pub fn dtype(&self) -> DType {
        self.layers[0].weight().dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_row_is_the_group_and_tag() {
        // Six vectors per group, three tags of two rows each: video, text, audio.
        assert_eq!(row(0, 0), 0);
        assert_eq!(row(0, 1), 1);
        assert_eq!(row(0, 2), 2);
        assert_eq!(row(1, 0), 3);
        assert_eq!(row(3, 2), 11);
        // The layer reshapes [groups, 18 · hidden] to [3 · groups, 6 · hidden], so a step of four
        // groups holds twelve rows of six vectors.
        assert_eq!(VECTORS, 6);
        assert_eq!(groups(true) * TAGS, 12);
    }

    #[test]
    fn a_generated_step_reads_the_video_and_audio_levels() {
        let levels = levels(&[0.9, 0.5], &[0.8, 0.4], false).unwrap();
        assert_eq!(levels, vec![0.9, 0.8, 0.5, 0.4]);
        let (times, table) = nn::diffusion::TimestepTable::entries(&levels, groups(false)).unwrap();
        assert_eq!(times, vec![0.9, 0.8, 0.5, 0.4]);
        assert_eq!(table, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_conditioned_step_reads_four_levels() {
        // The visual condition is the video's level floored at the clean anchor, and the audio
        // reference is clean: at the first step both are 1.0, at the second they separate.
        let levels = levels(&[1.0, 0.5], &[1.0, 0.5], true).unwrap();
        let (times, table) = nn::diffusion::TimestepTable::entries(&levels, groups(true)).unwrap();
        assert_eq!(times, vec![1.0, 0.5, 0.999]);
        assert_eq!(table, vec![0, 0, 0, 0, 1, 1, 2, 0]);
    }

    #[test]
    fn a_schedule_of_unequal_modalities_is_refused() {
        assert!(levels(&[1.0, 0.5], &[1.0], false).is_err());
        assert!(levels(&[], &[], false).is_err());
    }

    #[test]
    fn the_groups_follow_the_tasks() {
        // Only a text-to-video request skips the condition levels.
        assert_eq!(groups(false), 2);
        assert_eq!(groups(true), 4);
    }
}
