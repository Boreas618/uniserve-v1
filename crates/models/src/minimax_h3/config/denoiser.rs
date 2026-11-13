//! The denoiser's configuration: one diffusion transformer's widths, the schedule the
//! checkpoint states for it, and its sparse attention when it has one.
//!
//! A checkpoint holds one or two networks (`transformer`, `transformer_ref`). A network's
//! `config.json` states its widths; its schedule and attention are the base release's uniform
//! grid and dense attention, or a FastH3 student's distilled rungs or parallel-decoding blocks
//! and its video sparse attention, which the export's `fastvideo_inference.json` states.
//!
//! ## The schedule
//!
//! A schedule is normalized at the loading boundary into the sigma points each modality's
//! trajectory visits ([`ScheduleConfig`]), by the released formulas in the released precision:
//!
//! ```text
//! uniform grid   nn::diffusion::uniform_sigmas  (FP32 shift, consecutive equal points collapse)
//! DMD ladder     nn::diffusion::ladder_sigmas   (FP64 shift, one point per rung, then 0)
//! block grid     u = linspace(max_t, 0, intervals + 1) in FP64;
//!                sigma_j = f_s(u[node_j]) rounded once to FP32
//! fixed point    f_s(u) = s u M / (u (s − 1) + M),  M = max_t, which maps M to itself
//! ```
//!
//! The block grid is a parallel-decoding student's (`refs/vllm-omni/…/minimax_h3/pdd.py:20-24`):
//! one network evaluation per block of a fine grid, whose stacked output heads are fused with
//! the block's normalized integration weights ([`ScheduleConfig::head_weights`],
//! `pdd.py:32-39`).

use nn::Step;
use nn::config::Fields;
use nn::diffusion::{ladder_sigmas, linspace, steps, uniform_sigmas};
use nn::quantization::Declaration;

use crate::minimax_h3::{Error, Result, VideoTask};

/// The rows of one text or condition capacity unit of a dense layout: the reference's packed
/// row alignment (`refs/vllm-omni/…/minimax_h3/packed_sequence.py:45`).
const DENSE_CAPACITY_UNIT: u32 = 64;

/// The `(time, height, width)` tokens of one sparse attention tile, per tile row count
/// (`refs/vllm-omni/…/minimax_h3/attention/fastvideo_h3.py:40,93`).
const TILE_SHAPES: [(u32, [u32; 3]); 2] = [(64, [4, 4, 4]), (128, [4, 4, 8])];

/// The widths of one diffusion transformer, as its `config.json` states them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TransformerConfig {
    /// The transformer's width.
    pub hidden_size: u32,
    /// Attention heads.
    pub num_attention_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Transformer blocks.
    pub num_hidden_layers: u32,
    /// Blocks of the token refiner that conditions the prompt.
    pub num_refiner_layers: u32,
    /// The feed-forward width.
    pub intermediate_size: u32,
    /// Latent channels of the video stream.
    pub video_channels: u32,
    /// Latent channels of the audio stream.
    pub audio_channels: u32,
    /// Width of the text features the network conditions on.
    pub text_dim: u32,
    /// Width of the timestep's sinusoidal embedding.
    pub frequency_dim: u32,
    /// Hidden width of the timestep embedding.
    pub time_hidden_dim: u32,
    /// Width of the timestep embedding the modulation projections read.
    pub time_dim: u32,
    /// Frequencies of one rotary axis.
    pub rope_frequency_dim: u32,
    /// The rotary base.
    pub rope_theta: f32,
    /// The variance floor of the block norms and of the final norm.
    pub norm_eps: f32,
    /// The variance floor of the per-head query and key normalization.
    pub qk_norm_eps: f32,
    /// Output heads: 1 for an ordinary network, the fine grid's interval count for a
    /// parallel-decoding student, whose output projections stack one head per interval.
    pub output_heads: u32,
}

impl TransformerConfig {
    /// Reads a network's `config.json`.
    ///
    /// `pdd_steps`, when the checkpoint states it, is the parallel-decoding student's interval
    /// count and so its output head count; an ordinary network has one output head.
    ///
    /// # Errors
    ///
    /// A field that is absent or of another type, a patch other than `[1, 2, 2]`, a final norm
    /// floor that differs from the blocks', or rotary frequencies that do not fit a head.
    pub fn parse(values: &Fields<'_>) -> Result<Self> {
        let owner = values.owner();

        let output_heads = match values.optional_index("pdd_steps")? {
            None => 1,
            Some(heads) if heads < 2 => {
                return Err(Error::config(format!(
                    "{owner}.pdd_steps is the interval count of a parallel-decoding student and \
                     states more than one interval, got {heads}"
                )));
            }
            Some(heads) => heads,
        };

        // The packing and the latent reconstruction are written for one patch: a `2 x 2` block
        // of one latent frame.
        let patch = values.counts("patch_size")?;
        if patch != [1, 2, 2] {
            return Err(Error::config(format!(
                "{owner}.patch_size must be [1, 2, 2], got {patch:?}"
            )));
        }

        let norm_eps = values.positive("norm_eps")?;
        if values.positive("final_norm_eps")? != norm_eps {
            return Err(Error::config(format!(
                "{owner}.final_norm_eps must equal norm_eps"
            )));
        }

        let config = Self {
            hidden_size: values.count("hidden_size")?,
            num_attention_heads: values.count("num_attention_heads")?,
            head_dim: values.count("attention_head_dim")?,
            num_hidden_layers: values.count("num_layers")?,
            num_refiner_layers: values.count("num_refiner_layers")?,
            intermediate_size: values.count("ffn_dim")?,
            video_channels: values.count("in_channels")?,
            audio_channels: values.count("audio_in_channels")?,
            text_dim: values.count("text_dim")?,
            frequency_dim: values.count("freq_dim")?,
            time_hidden_dim: values.count("time_embed_hidden_dim")?,
            time_dim: values.count("time_embed_dim")?,
            rope_frequency_dim: values.count("rope_freq_dim")?,
            rope_theta: values.positive("rope_theta")? as f32,
            norm_eps: norm_eps as f32,
            qk_norm_eps: values.positive("qk_norm_eps")? as f32,
            output_heads,
        };

        // Each of the three rotary axes (time, height, width) rotates `2 · rope_frequency_dim`
        // channels of a head; the rest pass through.
        if !config.rope_frequency_dim.is_multiple_of(2)
            || config.rope_frequency_dim * 6 > config.head_dim
        {
            return Err(Error::config(format!(
                "{owner}.rope_freq_dim must be even and rotate at most a head's width, got {} \
                 of {}",
                config.rope_frequency_dim, config.head_dim
            )));
        }
        Ok(config)
    }

    /// The channels of a head the rotary embedding rotates: `2 · rope_frequency_dim` on each of
    /// the three axes (`refs/vllm-omni/…/minimax_h3/minimax_h3_blocks.py:225`).
    pub const fn rotated(&self) -> u32 {
        6 * self.rope_frequency_dim
    }

    /// The width of one video patch row: the latent channels of a `1 x 2 x 2` patch.
    pub const fn patch_width(&self) -> u32 {
        4 * self.video_channels
    }
}

/// The video sparse attention (VSA) of a FastH3 student, as its export states it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VsaConfig {
    /// Rows of one tile: 64 (`4 x 4 x 4` tokens) or 128 (`4 x 4 x 8`).
    pub tile_rows: u32,
    /// The fraction of its key tiles a generated-video query tile does not read. It stays FP64
    /// because `ceil((1 − sparsity) · tiles)` is a tile count.
    pub sparsity: f64,
    /// The fraction of its own key tiles a reference-video query tile keeps, when the export
    /// states one; a reference keeps the generated video's rate otherwise
    /// (`refs/vllm-omni/…/minimax_h3/attention/fastvideo_h3.py:220-223`).
    pub reference_keep: Option<f64>,
}

impl VsaConfig {
    /// Sparse attention over `tile_rows`-row tiles.
    ///
    /// # Errors
    ///
    /// A tile the video grid does not hold, a sparsity outside `[0, 1)` or a keep rate outside
    /// `(0, 1]`.
    pub fn new(tile_rows: u32, sparsity: f64, reference_keep: Option<f64>) -> Result<Self> {
        if !TILE_SHAPES.iter().any(|(rows, _)| *rows == tile_rows) {
            return Err(Error::config(format!(
                "an H3 sparse attention tile holds 64 or 128 rows, got {tile_rows}"
            )));
        }
        if !sparsity.is_finite() || !(0.0..1.0).contains(&sparsity) {
            return Err(Error::config(format!(
                "the VSA sparsity must lie in [0, 1), got {sparsity}"
            )));
        }
        if let Some(keep) = reference_keep
            && (!keep.is_finite() || !(0.0 < keep && keep <= 1.0))
        {
            return Err(Error::config(format!(
                "the VSA reference keep rate must lie in (0, 1], got {keep}"
            )));
        }

        Ok(Self {
            tile_rows,
            sparsity,
            reference_keep,
        })
    }

    /// The `[time, height, width]` tokens of one video tile.
    pub fn tile_shape(&self) -> [u32; 3] {
        TILE_SHAPES
            .iter()
            .find_map(|(rows, shape)| (*rows == self.tile_rows).then_some(*shape))
            .unwrap_or([1, 1, self.tile_rows])
    }

    /// The tiles of a segment of `tiles` tiles a query keeps at the keep rate `rate`:
    /// `ceil(rate · tiles)`, at least one and at most every tile, in FP64
    /// (`refs/vllm-omni/…/minimax_h3/attention/fastvideo_h3.py:144`).
    pub fn kept_tiles(rate: f64, tiles: u32) -> u32 {
        ((rate * f64::from(tiles)).ceil() as u32).clamp(1, tiles.max(1))
    }

    /// The keep rates of a reference video's tiles and of the generated video's, in that order.
    ///
    /// The reference rate is formed as `1 − (1 − keep)`, the rounding the released routing
    /// feeds its `ceil` (`refs/vllm-omni/…/minimax_h3/attention/fastvideo_h3.py:219-222`).
    pub fn keep_rates(&self) -> (f64, f64) {
        let target = 1.0 - self.sparsity;
        let reference = self
            .reference_keep
            .map_or(target, |keep| 1.0 - (1.0 - keep));

        (reference, target)
    }
}

/// The schedule a checkpoint states for a network, normalized into the sigma points each
/// modality's trajectory visits.
///
/// Both modalities advance on one step counter, so their point counts agree; each sequence
/// decreases to the clean endpoint `0`, and a schedule of `n` points takes `n − 1` steps.
#[derive(Clone, PartialEq, Debug)]
pub struct ScheduleConfig {
    /// The video schedule's shift `s` in `sigma = s u / (1 + (s − 1) u)`.
    pub video_shift: f32,
    /// The audio schedule's shift.
    pub audio_shift: f32,
    /// The video trajectory's sigma points.
    pub video_sigmas: Vec<f32>,
    /// The audio trajectory's sigma points.
    pub audio_sigmas: Vec<f32>,
    /// A parallel-decoding student's block boundaries on its fine grid, strictly increasing
    /// from zero to the interval count; empty for a network with one head per evaluation.
    pub nodes: Vec<u32>,
    /// The fixed point of both shifts on a parallel-decoding student's fine grid, in `(0, 1]`;
    /// one where there is no such grid.
    pub max_t: f32,
}

impl ScheduleConfig {
    /// The released uniform grid of `points` sigma points, the clean endpoint included.
    ///
    /// # Errors
    ///
    /// Fewer than two points, or two modalities whose deduplicated grids differ in length.
    pub fn uniform(points: u32, video_shift: f32, audio_shift: f32) -> Result<Self> {
        if points < 2 {
            return Err(Error::config(format!(
                "an H3 uniform grid needs at least two sigma points, got {points}"
            )));
        }

        Self {
            video_shift,
            audio_shift,
            video_sigmas: uniform_sigmas(points, video_shift),
            audio_sigmas: uniform_sigmas(points, audio_shift),
            nodes: Vec::new(),
            max_t: 1.0,
        }
        .aligned()
    }

    /// A DMD student's ladder: `rungs` unshifted levels of a `train_steps`-step training
    /// schedule, each one evaluation.
    ///
    /// # Errors
    ///
    /// No rung, a training schedule that is not finite and positive, or rungs that do not
    /// strictly decrease within it.
    pub fn ladder(
        rungs: &[u32],
        train_steps: f32,
        video_shift: f32,
        audio_shift: f32,
    ) -> Result<Self> {
        if rungs.is_empty() {
            return Err(Error::config(
                "an H3 DMD ladder needs at least one rung".to_owned(),
            ));
        }
        if !train_steps.is_finite() || train_steps <= 0.0 {
            return Err(Error::config(format!(
                "an H3 DMD training schedule must be finite and positive, got {train_steps}"
            )));
        }
        if rungs
            .iter()
            .any(|rung| *rung == 0 || *rung as f32 > train_steps)
        {
            return Err(Error::config(format!(
                "H3 DMD rungs must lie within the {train_steps}-step training schedule"
            )));
        }
        if rungs.windows(2).any(|pair| pair[0] <= pair[1]) {
            return Err(Error::config(
                "H3 DMD rungs must strictly decrease".to_owned(),
            ));
        }
        let train_steps = f64::from(train_steps);

        Self {
            video_shift,
            audio_shift,
            video_sigmas: ladder_sigmas(rungs, train_steps, video_shift),
            audio_sigmas: ladder_sigmas(rungs, train_steps, audio_shift),
            nodes: Vec::new(),
            max_t: 1.0,
        }
        .aligned()
    }

    /// A parallel-decoding student's block grid: one evaluation per block of a fine grid of
    /// `intervals` intervals, the blocks bounded by `nodes`.
    ///
    /// # Errors
    ///
    /// Fewer than two intervals, nodes that do not strictly increase from zero to the interval
    /// count, or a fixed point outside `(0, 1]`.
    pub fn blocks(
        intervals: u32,
        nodes: Vec<u32>,
        max_t: f32,
        video_shift: f32,
        audio_shift: f32,
    ) -> Result<Self> {
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

        // The node sigmas: the fixed-point shift of the fine grid's node levels, in FP64, each
        // rounded once (`pdd.py:20-24`).
        let grid = linspace(f64::from(max_t), 0.0, intervals as usize + 1);
        let sigmas = |shift: f32| -> Vec<f32> {
            let (shift, max_t) = (f64::from(shift), f64::from(max_t));

            nodes
                .iter()
                .map(|node| {
                    let level = grid[*node as usize];
                    (shift * level * max_t / (level * (shift - 1.0) + max_t)) as f32
                })
                .collect()
        };

        Self {
            video_shift,
            audio_shift,
            video_sigmas: sigmas(video_shift),
            audio_sigmas: sigmas(audio_shift),
            nodes,
            max_t,
        }
        .aligned()
    }

    /// This schedule, checked for one step count over both modalities: one runner advances both
    /// latents with one step counter.
    fn aligned(self) -> Result<Self> {
        if self.video_sigmas.len() != self.audio_sigmas.len() {
            return Err(Error::checkpoint(format!(
                "the H3 video and audio schedules must align: {} video sigma points against {} \
                 audio sigma points",
                self.video_sigmas.len(),
                self.audio_sigmas.len()
            )));
        }
        Ok(self)
    }

    /// The network evaluations of the schedule: one fewer than its sigma points.
    pub fn num_steps(&self) -> u32 {
        self.video_sigmas.len().saturating_sub(1) as u32
    }

    /// The steps a runner executes, on the video ladder: evaluation `i` runs from `sigma[i]` to
    /// `sigma[i + 1]`. The audio ladder shares the index.
    pub fn steps(&self) -> Vec<Step> {
        steps(&self.video_sigmas)
    }

    /// The levels `(video, audio)` a fresh draw is scaled to before the first evaluation, when
    /// the schedule scales it.
    ///
    /// Only a parallel-decoding schedule scales its draw
    /// (`refs/vllm-omni/…/minimax_h3/pipeline_minimax_h3.py:2269-2270`); the released and the
    /// distilled grids start from unit noise, though a distilled grid's first sigma is below
    /// one too, so the grid's kind decides and not its first number.
    pub fn start_sigmas(&self) -> Option<(f32, f32)> {
        if self.nodes.is_empty() {
            return None;
        }

        Some((*self.video_sigmas.first()?, *self.audio_sigmas.first()?))
    }

    /// The fusion weights of a parallel-decoding student's stacked output heads under `shift`:
    /// `[blocks, intervals]` row-major in FP32, row `b` holding block `b`'s normalized
    /// integration weights at its own fine intervals and zeros elsewhere.
    ///
    /// A block's weights are the fine grid's sigma increments `f_s(u[j + 1]) − f_s(u[j])` over
    /// its intervals, in the cancellation-free form `s M² (b − a) / (D(a) D(b))` with
    /// `D(u) = M + (s − 1) u`, normalized by their sum in FP64 and rounded once
    /// (`refs/vllm-omni/…/minimax_h3/pdd.py:32-39`). Their sum is the block's node-sigma
    /// increment, so an ordinary Euler step between the node sigmas applies the fused head.
    ///
    /// # Errors
    ///
    /// A schedule with one head per evaluation, or a block whose increments sum to zero.
    pub fn head_weights(&self, shift: f32) -> Result<Vec<f32>> {
        let Some(intervals) = self.nodes.last().copied() else {
            return Err(Error::checkpoint(
                "an H3 schedule with one output head per evaluation fuses no heads".to_owned(),
            ));
        };
        let (shift, max_t) = (f64::from(shift), f64::from(self.max_t));
        let grid = linspace(max_t, 0.0, intervals as usize + 1);

        let increments: Vec<f64> = grid
            .windows(2)
            .map(|pair| {
                let (start, end) = (pair[0], pair[1]);

                shift * max_t * max_t * (end - start)
                    / ((max_t + (shift - 1.0) * start) * (max_t + (shift - 1.0) * end))
            })
            .collect();

        let blocks = self.nodes.len() - 1;
        let mut weights = vec![0.0_f32; blocks * intervals as usize];
        for (block, span) in self.nodes.windows(2).enumerate() {
            let (start, end) = (span[0] as usize, span[1] as usize);
            let total: f64 = increments[start..end].iter().sum();
            if !total.is_finite() || total == 0.0 {
                return Err(Error::checkpoint(format!(
                    "the integration weights of H3 parallel-decoding block {block} sum to {total}"
                )));
            }

            for interval in start..end {
                weights[block * intervals as usize + interval] =
                    (increments[interval] / total) as f32;
            }
        }
        Ok(weights)
    }
}

/// One network of the checkpoint: its widths, its schedule, its attention and the tasks it
/// serves.
#[derive(Clone, PartialEq, Debug)]
pub struct DenoiserConfig {
    /// The network's widths.
    pub transformer: TransformerConfig,
    /// The schedule the checkpoint states for it.
    pub schedule: ScheduleConfig,
    /// Its video sparse attention; `None` attends densely.
    pub vsa: Option<VsaConfig>,
    /// The tasks it serves, in canonical order.
    pub tasks: Vec<VideoTask>,
    /// How the network's stage stores its weights: dense, or the packed projections a ModelOpt
    /// export's manifest calibrates.
    pub quantization: Declaration,
}

impl DenoiserConfig {
    /// One network with the schedule and attention the checkpoint states for it, its weights
    /// dense until the checkpoint's reader states a calibration.
    ///
    /// # Errors
    ///
    /// No task, a repeated task, or output heads that disagree with the schedule: only a
    /// parallel-decoding schedule has more than one head, and its interval count is the head
    /// count.
    pub fn new(
        transformer: TransformerConfig,
        schedule: ScheduleConfig,
        vsa: Option<VsaConfig>,
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

        let intervals = schedule.nodes.last().copied();
        if intervals != (transformer.output_heads > 1).then_some(transformer.output_heads) {
            return Err(Error::config(format!(
                "H3 output heads must equal the parallel-decoding interval count, and only a \
                 parallel-decoding student has more than one: {} heads, {intervals:?} intervals",
                transformer.output_heads
            )));
        }

        Ok(Self {
            transformer,
            schedule,
            vsa,
            tasks,
            quantization: Declaration::dense(),
        })
    }

    /// The rows of one text or condition capacity unit of a layout: a sparse tile, or the dense
    /// layout's 64 rows.
    pub fn capacity_unit(&self) -> u32 {
        self.vsa.map_or(DENSE_CAPACITY_UNIT, |vsa| vsa.tile_rows)
    }

    /// Whether any task this network serves carries conditions, so its steps read the two
    /// condition timestep groups beside the generated video's and audio's.
    pub fn conditioned(&self) -> bool {
        self.tasks.iter().any(|task| task.conditioned())
    }
}

#[cfg(test)]
mod tests {
    use nn::diffusion::time_shift;
    use serde_json::{Map, Value};

    use super::*;

    /// A transformer config parsed from `values`, as the loading boundary reads one.
    fn parse(values: &Map<String, Value>) -> Result<TransformerConfig> {
        TransformerConfig::parse(&Fields::of(
            "transformer/config.json",
            &Value::Object(values.clone()),
        )?)
    }

    fn transformer() -> Map<String, Value> {
        crate::minimax_h3::config::tests_support::transformer()
            .as_object()
            .unwrap()
            .clone()
    }

    /// FastVideo-FastH3-8-Step-V2's ladder: eight rungs, video shift 10, audio shift 3.
    fn ladder() -> ScheduleConfig {
        ScheduleConfig::ladder(&[999, 874, 749, 624, 500, 375, 250, 125], 1000.0, 10.0, 3.0)
            .unwrap()
    }

    /// FastVideo-FastH3-Omni-8-Step-V1's grid: 32 intervals, nine nodes, a fixed point of 0.999.
    fn blocks() -> ScheduleConfig {
        ScheduleConfig::blocks(32, (0..=32).step_by(4).collect(), 0.999, 12.0, 3.0).unwrap()
    }

    #[test]
    fn a_transformer_reads_the_released_dimensions() {
        let config = parse(&transformer()).unwrap();

        assert_eq!(config.hidden_size, 5376);
        assert_eq!(config.num_hidden_layers, 50);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.output_heads, 1);
        assert_eq!(config.rope_theta, 10000.0);
        assert_eq!(config.rotated(), 96);
        assert_eq!(config.patch_width(), 96);
    }

    #[test]
    fn a_transformer_refuses_another_patch_or_a_mismatched_final_norm() {
        let mut patched = transformer();
        patched.insert("patch_size".into(), serde_json::json!([1, 4, 4]));
        assert!(parse(&patched).is_err());

        let mut norm = transformer();
        norm.insert("final_norm_eps".into(), serde_json::json!(1e-6));
        let error = parse(&norm).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("final_norm_eps")),
            "{error}"
        );
    }

    #[test]
    fn a_transformer_refuses_rotary_dimensions_that_do_not_fit_a_head() {
        let mut values = transformer();
        values.insert("rope_freq_dim".into(), serde_json::json!(32));
        assert!(parse(&values).is_err());

        values.insert("rope_freq_dim".into(), serde_json::json!(15));
        assert!(parse(&values).is_err());
    }

    #[test]
    fn a_parallel_decoding_transformer_states_its_output_heads() {
        let mut values = transformer();
        values.insert("pdd_steps".into(), serde_json::json!(32));
        assert_eq!(parse(&values).unwrap().output_heads, 32);

        values.insert("pdd_steps".into(), serde_json::json!(1));
        assert!(parse(&values).is_err());
    }

    #[test]
    fn the_base_grid_has_fifty_sigma_points_and_a_unit_start() {
        let schedule = ScheduleConfig::uniform(50, 12.0, 3.0).unwrap();

        assert_eq!(schedule.video_sigmas.len(), 50);
        assert_eq!(schedule.video_sigmas[0], 1.0);
        assert_eq!(schedule.video_sigmas[49], 0.0);
        assert_eq!(schedule.num_steps(), 49);
        assert_eq!(schedule.start_sigmas(), None);

        // The endpoints are every shift's; a stronger shift keeps the trajectory nearer the
        // start.
        assert_eq!(schedule.video_sigmas[0], schedule.audio_sigmas[0]);
        assert!(schedule.video_sigmas[1] > schedule.audio_sigmas[1]);

        assert!(ScheduleConfig::uniform(1, 12.0, 3.0).is_err());
    }

    #[test]
    fn a_strong_shift_collapses_both_modalities_to_the_same_points() {
        let schedule = ScheduleConfig::uniform(4, 1e30, 1e30).unwrap();

        assert_eq!(schedule.video_sigmas, vec![1.0, 0.0]);
        assert_eq!(schedule.num_steps(), 1);
    }

    #[test]
    fn a_ladder_must_strictly_decrease_within_the_training_schedule() {
        let schedule = ladder();

        assert_eq!(schedule.num_steps(), 8);
        assert_eq!(schedule.video_sigmas.len(), 9);
        assert_eq!(schedule.video_sigmas[8], 0.0);
        assert_eq!(
            schedule.video_sigmas[0],
            time_shift(999.0_f64 / 1000.0, 10.0) as f32
        );

        // A distilled grid's first sigma is below one, and its draw is still not scaled.
        assert!(schedule.video_sigmas[0] < 1.0);
        assert_eq!(schedule.start_sigmas(), None);

        for rungs in [vec![999, 999], vec![125, 250], vec![1001, 500], vec![0]] {
            assert!(
                ScheduleConfig::ladder(&rungs, 1000.0, 12.0, 3.0).is_err(),
                "{rungs:?} is not a strict descending ladder within the training schedule"
            );
        }
    }

    #[test]
    fn a_ladder_keeps_one_evaluation_per_rung_though_two_round_together() {
        let schedule =
            ScheduleConfig::ladder(&[1_000_000_000, 999_999_999], 1e9, 12.0, 3.0).unwrap();

        assert_eq!(schedule.video_sigmas.len(), 3);
        assert_eq!(schedule.video_sigmas[0], schedule.video_sigmas[1]);
        assert_eq!(schedule.num_steps(), 2);
    }

    #[test]
    fn a_block_grid_runs_from_zero_to_its_interval_count() {
        let schedule = blocks();
        assert_eq!(schedule.num_steps(), 8);

        // The shift's fixed point is `max_t`, so the first node is one level for both
        // modalities, and it is the level the draw is scaled to.
        let (video, audio) = schedule.start_sigmas().unwrap();
        assert!((video - 0.999).abs() < 1e-6);
        assert!((audio - 0.999).abs() < 1e-6);

        assert!(ScheduleConfig::blocks(32, vec![0, 8, 32], 0.999, 12.0, 3.0).is_ok());
        for nodes in [vec![0], vec![4, 32], vec![0, 8, 8, 32], vec![0, 8, 16]] {
            assert!(
                ScheduleConfig::blocks(32, nodes.clone(), 0.999, 12.0, 3.0).is_err(),
                "{nodes:?} is not a block grid"
            );
        }
        assert!(ScheduleConfig::blocks(32, vec![0, 32], 0.0, 12.0, 3.0).is_err());
        assert!(ScheduleConfig::blocks(32, vec![0, 32], 1.5, 12.0, 3.0).is_err());
    }

    #[test]
    fn sigma_decreases_from_the_start_to_the_clean_endpoint() {
        for schedule in [
            ScheduleConfig::uniform(50, 12.0, 3.0).unwrap(),
            ladder(),
            blocks(),
        ] {
            for sigmas in [&schedule.video_sigmas, &schedule.audio_sigmas] {
                assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
                assert_eq!(sigmas.last(), Some(&0.0));
            }
        }
    }

    #[test]
    fn the_steps_carry_each_evaluation_interval_of_the_video_ladder() {
        let schedule = ScheduleConfig::uniform(50, 12.0, 3.0).unwrap();
        let steps = schedule.steps();

        assert_eq!(steps.len(), 49);
        for (index, step) in steps.iter().enumerate() {
            assert_eq!((step.index, step.count), (index as u32, 49));
            assert_eq!(step.sigma, schedule.video_sigmas[index]);
            assert_eq!(step.sigma_next, schedule.video_sigmas[index + 1]);
        }
    }

    #[test]
    fn a_blocks_head_weights_cover_its_own_intervals_and_sum_to_one() {
        let schedule = blocks();

        for shift in [12.0, 3.0] {
            let weights = schedule.head_weights(shift).unwrap();
            assert_eq!(weights.len(), 8 * 32);

            for block in 0..8 {
                let row = &weights[block * 32..(block + 1) * 32];
                let span = block * 4..(block + 1) * 4;

                // Four heads of the block's own fine intervals, and no other head.
                assert!(
                    row.iter()
                        .enumerate()
                        .all(|(interval, weight)| span.contains(&interval) == (*weight > 0.0))
                );
                let total: f32 = row.iter().sum();
                assert!((total - 1.0).abs() < 1e-6, "block {block} sums to {total}");
            }
        }

        assert!(ladder().head_weights(12.0).is_err());
    }

    #[test]
    fn a_denoiser_refuses_tasks_that_repeat_and_heads_that_disagree_with_its_grid() {
        let transformer = parse(&transformer()).unwrap();
        let uniform = ScheduleConfig::uniform(50, 12.0, 3.0).unwrap();

        let error = DenoiserConfig::new(
            transformer,
            uniform,
            None,
            vec![VideoTask::TextToVideoAudio, VideoTask::TextToVideoAudio],
        )
        .unwrap_err();
        assert!(matches!(error, Error::Config(ref problem) if problem.contains("distinct")));

        // A parallel-decoding grid without the student's stacked heads is refused.
        let error = DenoiserConfig::new(
            transformer,
            ScheduleConfig::blocks(32, vec![0, 32], 0.999, 12.0, 3.0).unwrap(),
            None,
            vec![VideoTask::ReferenceToVideoAudio],
        )
        .unwrap_err();
        assert!(matches!(error, Error::Config(ref problem) if problem.contains("output heads")));
    }

    #[test]
    fn sparse_attention_keeps_only_the_tiles_and_rates_it_serves() {
        assert!(VsaConfig::new(64, 0.8, None).is_ok());
        assert!(VsaConfig::new(96, 0.5, None).is_err());
        assert!(VsaConfig::new(64, 1.0, None).is_err());
        assert!(VsaConfig::new(64, -0.1, None).is_err());
        assert!(VsaConfig::new(128, 0.5, Some(0.0)).is_err());

        let tiled = VsaConfig::new(128, 0.9, Some(0.25)).unwrap();
        assert_eq!(tiled.tile_shape(), [4, 4, 8]);
        assert_eq!(
            VsaConfig::new(64, 0.9, None).unwrap().tile_shape(),
            [4, 4, 4]
        );

        // `ceil(rate · tiles)`, at least one tile and at most every tile.
        assert_eq!(VsaConfig::kept_tiles(0.1, 1782), 179);
        assert_eq!(VsaConfig::kept_tiles(0.0, 10), 1);
        assert_eq!(VsaConfig::kept_tiles(1.0, 10), 10);

        // A reference keeps its own rate when the export states one, and the generated
        // video's otherwise.
        let (reference, target) = tiled.keep_rates();
        assert_eq!(reference, 1.0 - (1.0 - 0.25));
        assert_eq!(target, 1.0 - 0.9);
        assert_eq!(
            VsaConfig::new(64, 0.9, None).unwrap().keep_rates().0,
            1.0 - 0.9
        );
    }
}
