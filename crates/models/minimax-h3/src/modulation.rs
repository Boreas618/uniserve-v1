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
//! A step is a **device** value: the worker writes the step's index into a `[1]` i32 table, the
//! cache gathers that row of `entries` to get `[groups]` entry ids, and each layer gathers its own
//! rows by those ids. One captured graph therefore serves every step of the schedule.
//!
//! Within a step's rows, a token's row is `3 · group + tag`: the group is the timestep the token is
//! conditioned on (the generated video's, the generated audio's, a visual condition's, an audio
//! reference's) and the tag is the modality the token belongs to (video 0, text 1, audio 2). The
//! worker's per-request `modulation_indices` table holds that value for every packed row, and the
//! layers of [`crate::transformer`] index with it.

use nn::Shape;
use nn::layers::gather_rows;
use nn::weights::{Weight, WeightLoader, WeightRequest};
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
#[cfg(test)]
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

/// A `[1, count]` index list as the flat `[count]` view the gather operator takes.
///
/// The operator's indices are one axis (`ops::logits::GatherArgs::rows`) while its output must be
/// two - `[count, columns]` - because it derives the row count from `out`. `entries_of` writes
/// `[1, groups]` to satisfy the output's shape, so the same bytes reach the layer and output
/// gathers as a two-axis view: the stack's tenth launch refused exactly that (`expected 1 axes,
/// found 2`). The arrangement is stated once, here, rather than at each of the three gathers.
fn flat_rows(rows: TensorView) -> TensorView {
    rows.reshape(Shape::new(&[rows.shape.numel()]))
}

impl AdaLnCache {
    /// Every checkpoint tensor the cache is built from, as complete names.
    ///
    /// The cache's tables are derived at load from these projections; the names are the ones
    /// `AdaLnCache::load` requests. `norm_out.linear` is the output table, which is a different
    /// tensor from the stack's own `norm_out.norm.weight`.
    pub fn keys(config: &crate::config::TransformerConfig) -> Vec<String> {
        let mut keys = Vec::new();
        for name in [
            "time_embedder.linear_1",
            "time_embedder.linear_2",
            "norm_out.linear",
        ] {
            keys.push(format!("{name}.weight"));
            keys.push(format!("{name}.bias"));
        }
        for index in 0..config.num_hidden_layers {
            keys.push(format!(
                "transformer_blocks.{index}.adaln_proj.linear.weight"
            ));
            keys.push(format!("transformer_blocks.{index}.adaln_proj.linear.bias"));
        }
        keys
    }

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
        _time_dim: u32,
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

    /// The timestep groups a step reads.
    pub const fn groups(&self) -> Groups {
        self.groups
    }

    /// The entry ids one step reads: `[1, groups]` i32, gathered from the entry table's `step` row.
    ///
    /// `step` is `[1]` i32 - the step index, which the worker writes - and `out` a `[1, groups]`
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
        Ok(gather_rows(f, table.view(), flat_rows(entries), out)?)
    }

    /// The final normalization's modulation rows at these entries: `[groups, 2 · hidden_size]`.
    pub fn output(&self, f: &Forward, entries: TensorView, out: TensorView) -> Result<()> {
        Ok(gather_rows(f, self.output.view(), flat_rows(entries), out)?)
    }

    /// The columns one entry's modulation table holds: the three interval tags of six vectors.
    ///
    /// A gather's output is `[entries, this]` - the shape the operator derives from the table -
    /// while a block cuts one interval at a time out of the same bytes.
    pub fn layer_width(&self) -> u64 {
        u64::from(TAGS) * VECTORS as u64 * u64::from(self.hidden_size)
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
pub fn entries(video: &[f32], audio: &[f32], groups: Groups) -> Result<(Vec<f32>, Vec<i32>)> {
    if video.is_empty() || video.len() != audio.len() {
        return Err(Error::checkpoint(format!(
            "{} video and {} audio noise levels are not one schedule's",
            video.len(),
            audio.len()
        )));
    }
    let mut times: Vec<f32> = Vec::new();
    let mut table = Vec::with_capacity(video.len() * groups.count() as usize);
    for (video, audio) in video.iter().zip(audio) {
        let mut levels = vec![*video, *audio];
        if groups == Groups::Conditioned {
            // A visual condition sits at the generated video's level until that level is cleaner
            // than its anchor, and an audio reference is always clean.
            levels.push(video.max(crate::denoiser::VISUAL_CONDITION_TIME));
            levels.push(crate::denoiser::AUDIO_CONDITION_TIME);
        }
        for level in levels {
            let position = match times
                .iter()
                .position(|other| other.to_bits() == level.to_bits())
            {
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

/// The checkpoint projections the cache is built from, read once at load.
///
/// The reference builds the cache with one shared pair of FP32 linear maps over the sinusoidal
/// timestep embedding and one projection per layer, then releases them
/// (`uniserve/models/minimax_h3/modulation.py`, `TimestepEmbedding`). Nothing runs them per
/// evaluation, so they are held only until `Projections::fill` has produced the tables once.
pub struct Projections {
    /// `time_embedder.linear_1`: `[time_hidden, frequency]`, FP32, with its bias.
    first: (Weight, Weight),
    /// `time_embedder.linear_2`: `[time_dim, time_hidden]`, FP32, with its bias.
    second: (Weight, Weight),
    /// One `adaln_proj.linear` per layer: `[18 * hidden, time_dim]`, with its bias.
    layers: Vec<(Weight, Weight)>,
    /// `norm_out.linear`: `[2 * hidden, time_dim]`, with its bias.
    output: (Weight, Weight),
    frequency_dim: u32,
    time_hidden_dim: u32,
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
        // whatever the preset (`weights.py:186-224`), and it casts the activation to the weight's
        // type before each product (`adaln_cache.py:128`). Our product has one element type for both
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
        let hidden = u64::from(config.hidden_size);
        let time_dim = u64::from(config.time_dim);
        let time_hidden = u64::from(config.time_hidden_dim);
        let frequency = u64::from(config.frequency_dim);
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                let prefix = format!("transformer_blocks.{index}.adaln_proj.linear");
                Ok((
                    matrix(
                        loader,
                        &format!("{prefix}.weight"),
                        VECTORS as u64 * hidden * u64::from(TAGS),
                        time_dim,
                        None,
                    )?,
                    vector(
                        loader,
                        &format!("{prefix}.bias"),
                        VECTORS as u64 * hidden * u64::from(TAGS),
                        None,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            first: (
                matrix(
                    loader,
                    "time_embedder.linear_1.weight",
                    time_hidden,
                    frequency,
                    Some(DType::F32),
                )?,
                vector(
                    loader,
                    "time_embedder.linear_1.bias",
                    time_hidden,
                    Some(DType::F32),
                )?,
            ),
            second: (
                matrix(
                    loader,
                    "time_embedder.linear_2.weight",
                    time_dim,
                    time_hidden,
                    Some(DType::F32),
                )?,
                vector(
                    loader,
                    "time_embedder.linear_2.bias",
                    time_dim,
                    Some(DType::F32),
                )?,
            ),
            layers,
            output: (
                matrix(loader, "norm_out.linear.weight", 2 * hidden, time_dim, None)?,
                vector(loader, "norm_out.linear.bias", 2 * hidden, None)?,
            ),
            frequency_dim: config.frequency_dim,
            time_hidden_dim: config.time_hidden_dim,
            time_dim: config.time_dim,
        })
    }

    /// Run the cache's projections over one embedding, `[groups, frequency_dim]` FP32.
    ///
    /// `layers_out` holds one view per transformer layer, `[groups, 18 * hidden]` each, and
    /// `output_out` is `[groups, 2 * hidden]`: the tables a forward then gathers rows from. The
    /// launch is captured with every other, so a table is computed once after loading and never
    /// again - the times it is computed at are the schedule's, which do not change per request.
    ///
    /// The weights are the checkpoint's `[out, in]` matrices, so the geometry is `NxK`.
    pub fn compute(
        &self,
        f: &Forward,
        embedding: TensorView,
        layers_out: &[TensorView],
        output_out: TensorView,
    ) -> Result<()> {
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
        // (`adaln_cache.py:128, 141-145`) - so at the released BF16 stack the tables are BF16, and at
        // a deployment whose weights are FP32 they are FP32. Reading the tables' type from `out`
        // instead and leaving the weights at the checkpoint's type is what a device run found: the
        // first projection of the forward asked cuBLASLt for an FP32 activation against a BF16
        // weight, a pair no algorithm exists for (`input_f32_weight_bf16_...` refused at its plan).
        //
        // The two halves of the family have different types, and the reference says so: the timestep
        // pair is FP32 at every preset (`weights.py:186-192`), the modulation projections take the
        // deployment's. So the activation the second pair leaves is FP32 and the modulation
        // projections read the deployment's type - **one narrowing between them**, which is the
        // reference's `.to(weight.dtype)` and the only crossing in this cache. At an FP32
        // deployment the two types coincide and the narrowing is skipped.
        let time_type = self.first.0.dtype();
        let dtype = self.layers[0].0.dtype();
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
        let hidden = arena.tensor(
            time_type,
            Shape::new(&[groups, u64::from(self.time_hidden_dim)]),
        )?;
        let activated = arena.tensor(
            time_type,
            Shape::new(&[groups, u64::from(self.time_hidden_dim)]),
        )?;
        let time = arena.tensor(time_type, Shape::new(&[groups, u64::from(self.time_dim)]))?;
        let linear = |weights: &(Weight, Weight)| {
            nn::layers::Linear::new(weights.0, Some(weights.1), nn::layers::WeightOrder::NxK)
        };
        linear(&self.first)?.forward(f, embedding, hidden)?;
        nn::layers::Silu.forward(f, hidden, activated)?;
        linear(&self.second)?.forward(f, activated, time)?;
        // The reference feeds the projections `silu(embedding)` (`weights.py`, `activated`): the
        // pair's output is activated once more before any modulation reads it.
        let activated_time = arena.tensor(time_type, time.shape)?;
        nn::layers::Silu.forward(f, time, activated_time)?;
        let time = activated_time;
        // The one crossing: the activation the timestep pair built, rounded into the type the
        // modulation projections read. `target <- 1 * source + 0 * target` is the reference's
        // `.to(weight.dtype)`, and the zero beta means the target is never read.
        let narrow = if dtype == time_type {
            time
        } else {
            let out = arena.tensor(dtype, time.shape)?;
            nn::layers::ScaledAdd.forward(f, out, time, 1.0, 0.0)?;
            out
        };
        for (weights, out) in self.layers.iter().zip(layers_out) {
            linear(weights)?.forward(f, narrow, *out)?;
        }
        Ok(linear(&self.output)?.forward(f, narrow, output_out)?)
    }

    /// The sinusoidal embedding a step's network times need: `[groups, frequency_dim]` FP32.
    ///
    /// Half a pairwise spectrum over the timestep, `sin` on the even columns and `cos` on the odd
    /// ones, at the frequencies `theta_high^(-2i / frequency_dim)`: the released recipe, written
    /// here because it is the only part of the cache that is arithmetic on a host value.
    pub fn frequencies(&self, times: &[f32]) -> Result<Vec<u8>> {
        embedding(times, self.frequency_dim)
    }

    /// The element type the modulation projections read: the tables' type, and the cache's.
    pub fn dtype(&self) -> DType {
        self.layers[0].0.dtype()
    }

    /// How many layers the projections cover.
    pub fn layers(&self) -> usize {
        self.layers.len()
    }
}

/// The sinusoidal embedding of each entry time, `[entries, frequency_dim]` FP32.
///
/// The reference's GLIDE/DiT order (`uniserve/nn/timestep.py`, `timestep_embedding`): the first half
/// of a row is a cosine per frequency and the second half is the sine of the same angle, so row
/// `r`, element `i` is `cos(t_r · 10000^(-i/half))` for `i < half` and the sine for `i >= half`.
///
/// **The order is not cosmetic.** The shared FP32 projection reads this vector as 256 unrelated
/// inputs, so a row that interleaved sine and cosine would be a different modulation under the same
/// schedule - and nothing downstream can see it: the products are finite, the widths match, and the
/// shape of the failure is a slightly different picture. The angle is computed in f64 and stored
/// once as f32, which is the reference's own rounding of `exp`/`cos` in float32 to within an ulp.
fn embedding(times: &[f32], frequency_dim: u32) -> Result<Vec<u8>> {
    if times.is_empty() {
        return Err(Error::checkpoint(
            "a timestep embedding needs at least one time".to_owned(),
        ));
    }
    if frequency_dim == 0 || !frequency_dim.is_multiple_of(2) {
        return Err(Error::checkpoint(format!(
            "a {frequency_dim}-wide timestep embedding has no cosine half"
        )));
    }
    let half = (frequency_dim / 2) as usize;
    let mut bytes = Vec::with_capacity(times.len() * frequency_dim as usize * 4);
    for time in times {
        let angles: Vec<f64> = (0..half)
            .map(|index| {
                let exponent = 2.0 * index as f64 / f64::from(frequency_dim);
                f64::from(*time) * 10_000_f64.powf(-exponent)
            })
            .collect();
        for angle in &angles {
            bytes.extend_from_slice(&(angle.cos() as f32).to_le_bytes());
        }
        for angle in &angles {
            bytes.extend_from_slice(&(angle.sin() as f32).to_le_bytes());
        }
    }
    Ok(bytes)
}

/// A `[rows, columns]` checkpoint weight.
fn matrix(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
    dtype: Option<DType>,
) -> Result<Weight> {
    let mut request = WeightRequest::region(name, 0, Shape::new(&[rows, columns]));
    if let Some(dtype) = dtype {
        request = request.pinned(dtype);
    }
    Ok(loader.load(&request)?)
}

fn vector(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    dtype: Option<DType>,
) -> Result<Weight> {
    let mut request = WeightRequest::region(name, 0, Shape::new(&[rows]));
    if let Some(dtype) = dtype {
        request = request.pinned(dtype);
    }
    Ok(loader.load(&request)?)
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
    fn the_timestep_embedding_is_every_cosine_then_every_sine() {
        // The reference's feature vector is `cat((cos, sin))` over the frequency ladder, not an
        // interleaving of the two: the shared projection reads 256 unrelated inputs, so a permuted
        // row is a different modulation for the same schedule and every downstream number is
        // finite, well-shaped and wrong.
        let dim = 8;
        let time = 0.25f32;
        let bytes = embedding(&[time], dim).expect("an embedding");
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        assert_eq!(values.len(), dim as usize);
        let half = dim as usize / 2;
        for index in 0..half {
            let angle = f64::from(time) * 10_000_f64.powf(-(index as f64) / half as f64);
            assert!(
                (values[index] - angle.cos() as f32).abs() < 1e-6,
                "element {index} is a cosine"
            );
            assert!(
                (values[half + index] - angle.sin() as f32).abs() < 1e-6,
                "element {} is the sine of the same angle",
                half + index
            );
        }
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
