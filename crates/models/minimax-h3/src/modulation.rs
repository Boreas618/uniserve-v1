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

use nn::Shape;
use nn::diffusion::TimestepTable;
use nn::layers::{Bias, Linear, Parallel};
use nn::timestep::{TimestepEmbedding, TimestepSpec};
use nn::weights::WeightLoader;
use nn::{DType, Forward, TensorView};

use crate::{Error, Result};

/// The six affine vectors one layer reads per token: shift, scale and gate of the attention block,
/// then of the feed-forward block.
pub const VECTORS: usize = 6;

/// The three token tags a timestep group's vectors are stored for, in row order.
pub const TAGS: u32 = 3;

/// The timestep groups a step reads: the generated video and audio for a text-only denoiser, and the
/// visual-condition and audio-reference levels as well for a conditioned one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Groups {
    /// `t2va`: the generated video and audio.
    Generated,
    /// `fl2va` and `ref2va`: the generated video and audio, a visual condition's clean level and an
    /// audio reference's clean level.
    Conditioned,
}

impl Groups {
    /// The groups the task set reads, in row order.
    pub fn of(conditioned: bool) -> Self {
        if conditioned {
            Self::Conditioned
        } else {
            Self::Generated
        }
    }

    /// How many timestep groups a step reads.
    pub const fn count(self) -> u32 {
        match self {
            Self::Generated => 2,
            Self::Conditioned => 4,
        }
    }
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
pub fn levels(video: &[f32], audio: &[f32], groups: Groups) -> Result<Vec<f32>> {
    if video.is_empty() || video.len() != audio.len() {
        return Err(Error::checkpoint(format!(
            "{} video and {} audio noise levels are not one schedule's",
            video.len(),
            audio.len()
        )));
    }
    let mut levels = Vec::with_capacity(video.len() * groups.count() as usize);
    for (video, audio) in video.iter().zip(audio) {
        levels.extend([*video, *audio]);
        if groups == Groups::Conditioned {
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

pub struct Projections {
    /// `time_embedder`: `linear_1` `[time_hidden, frequency]` and `linear_2` `[time_dim,
    /// time_hidden]`, FP32, with their biases.
    time: TimestepEmbedding,
    /// One `adaln_proj.linear` per layer: `[18 * hidden, time_dim]`, with its bias.
    layers: Vec<Linear>,
    /// `norm_out.linear`: `[2 * hidden, time_dim]`, with its bias.
    output: Linear,
    frequency_dim: u32,
    time_dim: u32,
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
            time_dim: config.time_dim,
        })
    }

    /// Fill `table`'s block and output tables from its embedding, `[entries, frequency_dim]` FP32.
    ///
    /// Each block table is `[entries, 18 * hidden]` and the output table `[entries, 2 * hidden]`:
    /// the tables a forward then gathers rows from. The launch is captured with every other, so a
    /// table is computed once after loading and never again - the times it is computed at are the
    /// schedule's, which do not change per request.
    ///
    /// The weights are the checkpoint's `[out, in]` matrices, so the geometry is `NxK`.
    pub fn compute(&self, f: &Forward, table: &TimestepTable) -> Result<()> {
        let embedding = table.frequencies();
        let layers_out: &[TensorView] = &table.layer_tables();
        let output_out = table.output_table();
        if layers_out.len() != self.layers.len() {
            return Err(Error::checkpoint(format!(
                "{} AdaLN layer views for {} layers",
                layers_out.len(),
                self.layers.len()
            )));
        }
        let groups = embedding.shape.dim(0);
        if embedding.shape.rank() != 2 || embedding.shape.last() != u64::from(self.frequency_dim) {
            return Err(Error::config(format!(
                "an AdaLN embedding of {:?} is not [groups, {}]",
                embedding.shape.dims(),
                self.frequency_dim
            )));
        }
        let arena = f.arena();
        // **The activation is cast to the weight's type, and the tables take the projection's type.**
        // The reference writes both halves of that in one line - `F.linear(activated_timesteps.to(
        // weight.dtype), weight, bias)` with the tables allocated as `projected.new_empty(...)`
        // - so at the released BF16 stack the tables are BF16, and at
        // a deployment whose weights are FP32 they are FP32. Reading the tables' type from `out`
        // instead and leaving the weights at the checkpoint's type is what a device run found: the
        // first projection of the forward asked cuBLASLt for an FP32 activation against a BF16
        // weight, a pair no algorithm exists for (`input_f32_weight_bf16_...` refused at its plan).
        //
        // The two halves of the family have different types, and the reference says so: the timestep
        // pair is FP32 at every preset, the modulation projections take the
        // deployment's. So the activation the second pair leaves is FP32 and the modulation
        // projections read the deployment's type - **one narrowing between them**, which is the
        // reference's `.to(weight.dtype)` and the only crossing in this cache. At an FP32
        // deployment the two types coincide and the narrowing is skipped.
        let time_type = self.time.dtype();
        let dtype = self.layers[0].weight().dtype;
        if time_type != DType::F32 {
            return Err(Error::config(format!(
                "the AdaLN timestep pair is {time_type} and the reference pins it to FP32 at every \
                 preset"
            )));
        }
        if output_out.dtype != dtype {
            return Err(Error::config(format!(
                "the AdaLN tables are {} and the projections' weights are {dtype}: the reference \
                 takes the tables' type from the projection, so a view of another type is a table \
                 no launch of this cache can write",
                output_out.dtype
            )));
        }
        for table in layers_out {
            if table.dtype != dtype {
                return Err(Error::config(format!(
                    "an AdaLN layer table is {} and the projections' weights are {dtype}",
                    table.dtype
                )));
            }
        }
        if embedding.dtype != time_type {
            return Err(Error::config(format!(
                "the AdaLN embedding is {} and the timestep pair computes in {time_type}",
                embedding.dtype
            )));
        }
        let time = arena.tensor(time_type, Shape::new(&[groups, u64::from(self.time_dim)]))?;
        self.time.forward(f, embedding, time)?;
        // The projections read `silu(embedding)`: the
        // pair's output is activated once more before any modulation reads it.
        let activated_time = arena.tensor(time_type, time.shape)?;
        nn::layers::Silu.forward(f, time, activated_time)?;
        let time = activated_time;
        // The one crossing: the activation the timestep pair built, rounded into the type the
        // modulation projections read: the reference's
        // `.to(weight.dtype)`.
        let narrow = if dtype == time_type {
            time
        } else {
            let out = arena.tensor(dtype, time.shape)?;
            nn::layers::Cast.forward(f, time, out)?;
            out
        };
        for (layer, out) in self.layers.iter().zip(layers_out) {
            layer.forward(f, narrow, *out)?;
        }
        Ok(self.output.forward(f, narrow, output_out)?)
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
        assert_eq!(Groups::Conditioned.count() * TAGS, 12);
    }

    #[test]
    fn a_generated_step_reads_the_video_and_audio_levels() {
        let levels = levels(&[0.9, 0.5], &[0.8, 0.4], Groups::Generated).unwrap();
        assert_eq!(levels, vec![0.9, 0.8, 0.5, 0.4]);
        let (times, table) = TimestepTable::entries(&levels, Groups::Generated.count()).unwrap();
        assert_eq!(times, vec![0.9, 0.8, 0.5, 0.4]);
        assert_eq!(table, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_conditioned_step_reads_four_levels() {
        // The visual condition is the video's level floored at the clean anchor, and the audio
        // reference is clean: at the first step both are 1.0, at the second they separate.
        let levels = levels(&[1.0, 0.5], &[1.0, 0.5], Groups::Conditioned).unwrap();
        let (times, table) = TimestepTable::entries(&levels, Groups::Conditioned.count()).unwrap();
        assert_eq!(times, vec![1.0, 0.5, 0.999]);
        assert_eq!(table, vec![0, 0, 0, 0, 1, 1, 2, 0]);
    }

    #[test]
    fn a_schedule_of_unequal_modalities_is_refused() {
        assert!(levels(&[1.0, 0.5], &[1.0], Groups::Generated).is_err());
        assert!(levels(&[], &[], Groups::Generated).is_err());
    }

    #[test]
    fn the_groups_follow_the_tasks() {
        // Only a text-to-video request skips the condition levels.
        assert_eq!(Groups::of(false), Groups::Generated);
        assert_eq!(Groups::of(true), Groups::Conditioned);
        assert_eq!(Groups::Generated.count(), 2);
        assert_eq!(Groups::Conditioned.count(), 4);
    }
}
