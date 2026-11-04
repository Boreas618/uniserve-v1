//! The diffusion transformer's timestep conditioning: the AdaLN cache and the final output norm.
//!
//! Every layer is conditioned on the step's noise levels through six affine vectors per token -
//! shift, scale and gate for the attention block and for the feed-forward block - and the final
//! normalization carries two more. The released network computes them with one shared timestep
//! embedding and one projection per layer, and the schedule is fixed, so the products are
//! precomputed once for the schedule's distinct timesteps (the reference's `AdaLNCache`,
//! `uniserve/models/minimax_h3/transformer.py:380-402` and
//! `uniserve/models/minimax_h3/modulation.py`). The cache is computed at load on the device: one
//! `Linear(frequency_dim → time_hidden_dim)`, a SiLU, one `Linear(time_hidden_dim → time_dim)` and
//! then, per layer, one `Linear(time_dim → 18 · hidden_size)` - about 260 MMAC per layer per
//! distinct timestep, which no host-side boundary can absorb (`crate::Solver`'s note has the
//! arithmetic).
//!
//! ## What a forward reads
//!
//! Three tables, all of them constants of the checkpoint and the schedule:
//!
//! ```text
//! layers   [layers, entries, 18 · hidden]  BF16   one projection per layer per distinct timestep
//! output   [entries, 2 · hidden]           BF16   the final normalization's shift and scale
//! entries  [steps, groups]                 i32    the entry each (step, group) reads
//! ```
//!
//! A step is a **device** value: the executor writes the step's index into a `[1]` i32 table, the
//! cache gathers that row of `entries` to get `[groups]` entry ids, and each layer gathers its own
//! rows by those ids. One captured graph therefore serves every step of the schedule.
//!
//! Within a step's rows, a token's row is `3 · group + tag`: the group is the timestep the token is
//! conditioned on (the generated video's, the generated audio's, a visual condition's, an audio
//! reference's) and the tag is the modality the token belongs to (video 0, text 1, audio 2). The
//! executor's per-request `modulation_indices` table holds that value for every packed row, and the
//! layers of [`crate::transformer`] index with it.

use nn::layers::gather_rows;
use nn::weights::Weight;
use nn::{Forward, TensorView};

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
/// This is the value the executor writes into the per-request `modulation_indices` table, and the
/// only request-dependent input of the conditioning.
pub const fn row(group: u32, tag: u32) -> u32 {
    TAGS * group + tag
}

/// The AdaLN cache: the per-layer modulation tables a step gathers from.
pub struct AdaLnCache {
    layers: Vec<Weight>,
    output: Weight,
    entries: Weight,
    groups: Groups,
    hidden_size: u32,
}

impl AdaLnCache {
    /// The cache over `layers`' tables, the final normalization's, and the entry table.
    ///
    /// `layers` holds one `[entries, 18 · hidden_size]` table per transformer layer, `output` the
    /// `[entries, 2 · hidden_size]` table of the final normalization, and `entries` the
    /// `[steps, groups]` i32 table of the entry each group of each step reads.
    pub fn new(
        layers: Vec<Weight>,
        output: Weight,
        entries: Weight,
        groups: Groups,
        hidden_size: u32,
    ) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error::checkpoint(
                "the H3 AdaLN cache needs one table per transformer layer".to_owned(),
            ));
        }
        let width = u64::from(hidden_size) * VECTORS as u64 * u64::from(TAGS);
        for (index, table) in layers.iter().enumerate() {
            if table.shape().rank() != 2 || table.shape().last() != width {
                return Err(Error::checkpoint(format!(
                    "H3 AdaLN layer {index} is [entries, {width}], not {:?}",
                    table.shape().dims()
                )));
            }
        }
        if output.shape().rank() != 2 || output.shape().last() != 2 * u64::from(hidden_size) {
            return Err(Error::checkpoint(format!(
                "H3 AdaLN output table is [entries, {}], not {:?}",
                2 * u64::from(hidden_size),
                output.shape().dims()
            )));
        }
        if entries.shape().rank() != 2 || entries.shape().last() != u64::from(groups.count()) {
            return Err(Error::checkpoint(format!(
                "H3 AdaLN entry table is [steps, {}], not {:?}",
                groups.count(),
                entries.shape().dims()
            )));
        }
        Ok(Self {
            layers,
            output,
            entries,
            groups,
            hidden_size,
        })
    }

    /// How many transformer layers the cache covers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the cache covers no layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The timestep groups a step reads.
    pub const fn groups(&self) -> Groups {
        self.groups
    }

    /// The element type of the modulation tables.
    pub fn dtype(&self) -> nn::DType {
        self.layers[0].dtype()
    }

    /// The entry ids one step reads: `[groups]` i32, gathered from the entry table's `step` row.
    ///
    /// `step` is `[1]` i32 - the step index, which the executor writes - and `out` a `[1, groups]`
    /// i32 scratch view it overwrites with the entries.
    pub fn entries_of(&self, f: &Forward, step: TensorView, out: TensorView) -> Result<()> {
        Ok(gather_rows(f, self.entries.view(), step, out)?)
    }

    /// The modulation rows of layer `layer` at these entries: `[groups, 18 · hidden_size]`.
    ///
    /// `entries` is the `[1, groups]` i32 view [`AdaLnCache::entries_of`] wrote; `out` is the
    /// layer's `[groups, 18 · hidden_size]` view, into which the layer's six vectors per tag land.
    pub fn layer(
        &self,
        f: &Forward,
        layer: usize,
        entries: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let table = self.layers.get(layer).ok_or_else(|| {
            Error::checkpoint(format!(
                "H3 AdaLN cache covers {} layers, not {layer}",
                self.layers.len()
            ))
        })?;
        Ok(gather_rows(f, table.view(), entries, out)?)
    }

    /// The final normalization's modulation rows at these entries: `[groups, 2 · hidden_size]`.
    pub fn output(&self, f: &Forward, entries: TensorView, out: TensorView) -> Result<()> {
        Ok(gather_rows(f, self.output.view(), entries, out)?)
    }

    /// Elements one layer's gathered modulation holds.
    pub fn layer_elements(&self) -> u64 {
        u64::from(self.groups.count()) * u64::from(self.hidden_size) * VECTORS as u64 * u64::from(TAGS)
    }

    /// Elements the gathered final modulation holds.
    pub fn output_elements(&self) -> u64 {
        u64::from(self.groups.count()) * 2 * u64::from(self.hidden_size)
    }

    /// Bytes the per-forward scratch needs: the entry ids and the largest gathered table.
    pub fn scratch_bytes(&self) -> u64 {
        let elements = self.layer_elements().max(self.output_elements());
        let width = elements * nn::DType::BF16.bytes().unwrap_or(2);
        width + u64::from(self.groups.count()) * 4
    }
}


/// The distinct noise levels the cache holds and the entry each step's group reads.
///
/// The cache is indexed by *value*, not by step: the generated video and audio have their own
/// noise level per step, the visual condition's level is the video's floored at the clean level the
/// reference anchors it at (`uniserve/models/minimax_h3/denoiser.py`, the modulation table), and an
/// audio reference is clean at the top of the schedule. Steps whose levels coincide share one entry,
/// which is where the cache's saving comes from - the schedule's points repeat across the two
/// modalities and the condition groups.
///
/// The result is `(times, entries)`: one FP32 network time per entry, and one entry index per
/// `(step, group)` in step-major order, ready to be published as the `[steps, groups]` table.
pub fn entries(
    video: &[f32],
    audio: &[f32],
    groups: Groups,
) -> Result<(Vec<f32>, Vec<i32>)> {
    if video.is_empty() || video.len() != audio.len() {
        return Err(Error::checkpoint(format!(
            "{} video and {} audio noise levels are not one schedule's",
            video.len(),
            audio.len()
        )));
    }
    /// The clean level a visual condition is anchored at: the last sigma point of the schedule.
    const CLEAN: f32 = 0.999;
    let mut times: Vec<f32> = Vec::new();
    let mut table = Vec::with_capacity(video.len() * groups.count() as usize);
    for (video, audio) in video.iter().zip(audio) {
        let mut levels = vec![*video, *audio];
        if groups == Groups::Conditioned {
            // A visual condition sits at the generated video's level until that level is cleaner
            // than its anchor, and an audio reference is always clean.
            levels.push(video.max(CLEAN));
            levels.push(1.0);
        }
        for level in levels {
            let position = match times.iter().position(|other| other.to_bits() == level.to_bits()) {
                Some(position) => position,
                None => {
                    times.push(level);
                    times.len() - 1
                }
            };
            table.push(i32::try_from(position).map_err(|_| {
                Error::checkpoint("an AdaLN cache of more than i32::MAX entries".to_owned())
            })?);
        }
    }
    Ok((times, table))
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
    fn the_entries_are_the_distinct_noise_levels() {
        // Two steps, one modality pair, the generated groups only: four levels, no sharing.
        let (times, table) = entries(&[0.9, 0.5], &[0.8, 0.4], Groups::Generated).unwrap();
        assert_eq!(times, vec![0.9, 0.8, 0.5, 0.4]);
        assert_eq!(table, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_shared_level_is_one_entry() {
        // The same pair of levels in both steps, as the last step of a schedule: two entries.
        let (times, table) = entries(&[1.0, 1.0], &[1.0, 1.0], Groups::Generated).unwrap();
        assert_eq!(times, vec![1.0]);
        assert_eq!(table, vec![0, 0, 0, 0]);
    }

    #[test]
    fn a_conditioned_step_reads_four_levels() {
        // The visual condition is the video's level floored at the clean anchor, and the audio
        // reference is clean: at the first step both are 1.0, at the second they separate.
        let (times, table) = entries(&[1.0, 0.5], &[1.0, 0.5], Groups::Conditioned).unwrap();
        assert_eq!(times, vec![1.0, 0.5, 0.999]);
        assert_eq!(table, vec![0, 0, 0, 0, 1, 1, 2, 0]);
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
