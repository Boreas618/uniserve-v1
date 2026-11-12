//! Diffusion timestep conditioning: the MLP that projects a time's sinusoidal embedding, and a
//! fixed schedule's modulation precomputed at each distinct noise level.

use uniserve_tensor::{DType, Device, Module, Tensor};

use crate::diffusion::sinusoidal;
use crate::error::{Error, Result};
use crate::functional;
use crate::modules::linear::{Linear, LinearConfig, Parallel};
use crate::var_builder::VarBuilder;

/// The widths of a [`TimestepEmbedding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestepEmbeddingConfig {
    /// The sinusoidal embedding's width.
    pub frequency_dim: u32,
    /// The hidden width between the two projections.
    pub hidden: u32,
    /// The output width.
    pub out: u32,
}

/// `linear_2(silu(linear_1(x)))` over a sinusoidal embedding: the `Linear → SiLU → Linear` pair
/// a DiT applies to it (`TimestepEmbedding` of diffusers, `TimestepEmbedder` of the multimodal
/// references).
pub struct TimestepEmbedding {
    first: Linear,
    second: Linear,
}

impl TimestepEmbedding {
    /// The pair the module at `vb` states as `linear_1` and `linear_2`, each with its bias.
    ///
    /// A model that keeps the embedding in a representation of its own pins `vb`
    /// ([`VarBuilder::to_dtype`]).
    ///
    /// # Errors
    ///
    /// A projection the loader refuses.
    pub fn load(config: &TimestepEmbeddingConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let projection = |name: &str, inputs: u32, outputs: u32| {
            Linear::load(
                &LinearConfig {
                    inputs,
                    outputs,
                    bias: true,
                    parallel: Parallel::None,
                },
                vb.pp(name),
            )
        };

        Ok(Self {
            first: projection("linear_1", config.frequency_dim, config.hidden)?,
            second: projection("linear_2", config.hidden, config.out)?,
        })
    }

    /// The element type the pair computes in: its weights'.
    pub fn dtype(&self) -> DType {
        self.first.weight().dtype()
    }

    /// `linear_2(silu(linear_1(embedding)))` of `embedding`, `[rows, frequency_dim]`:
    /// `[rows, out]` in the embedding's element type.
    ///
    /// # Errors
    ///
    /// Rows a projection refuses.
    pub fn forward(&self, embedding: &Tensor) -> Result<Tensor> {
        let hidden = self.first.forward(embedding)?;
        self.second.forward(&functional::silu(&hidden)?)
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, embedding: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, embedding)?)
    }
}

/// The widths of a [`TimestepTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestepTableConfig {
    /// Width of the sinusoidal embedding the levels are evaluated at.
    pub frequency_dim: u32,
    /// The blocks the table holds one modulation table for.
    pub layers: u32,
    /// Columns of one block's modulation row.
    pub layer_columns: u32,
    /// Columns of the final norm's modulation row.
    pub output_columns: u32,
    /// Element type of the modulation tables.
    pub dtype: DType,
}

/// A diffusion stack's timestep conditioning, precomputed at each distinct noise level of a
/// fixed schedule and gathered per step.
///
/// A step conditions its rows on several timestep groups (a modality's level, a condition's
/// level), and every block reads one modulation row per group. The schedule is fixed, so each
/// block's modulation projection is evaluated once per **distinct** level, and a step gathers
/// its groups' rows by entry index:
///
/// ```text
/// frequencies  [entries, frequency_dim]           FP32   the sinusoidal embedding of each level
/// entries      [steps, groups]                    U32    the entry each (step, group) reads
/// layers       [layers, entries, layer_columns]          each block's modulation at each entry
/// output       [entries, output_columns]                 the final norm's modulation at each entry
/// ```
///
/// `frequencies` and `entries` are constants of the schedule, computed at load. The two
/// modulation tables are zeros the load reserves writable: the model fills them once after
/// loading from `frequencies` through its own projections (`Denoiser::initialize`), and every
/// step reads them. A step's index is moved to the device as data, so one captured forward
/// serves every step.
pub struct TimestepTable {
    frequencies: Tensor,
    entries: Tensor,
    /// `[layers, entries, layer_columns]`: each block's modulation at each entry.
    layers: Tensor,
    /// `[entries, output_columns]`: the final norm's modulation at each entry.
    output: Tensor,
}

impl TimestepTable {
    /// The table of `levels` under `vb`: `[steps, groups]` FP32 on the CPU device, the level
    /// each group of each step is conditioned on.
    ///
    /// Levels are compared bitwise, and an entry is the position of its level's first
    /// occurrence, so the entries follow the schedule's order.
    ///
    /// # Errors
    ///
    /// Levels that are not `[steps, groups]` with a step and a group, an embedding width
    /// [`sinusoidal`] refuses, or a constant the loader refuses.
    pub fn load(
        config: &TimestepTableConfig,
        levels: &Tensor,
        vb: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let &[steps, groups] = levels.dims() else {
            return Err(Error::shape(
                "TimestepTable",
                format!("the levels are [steps, groups], not {:?}", levels.dims()),
            ));
        };
        if steps == 0 || groups == 0 {
            return Err(Error::shape(
                "TimestepTable",
                format!("{steps} steps of {groups} groups hold no level"),
            ));
        }

        // The distinct levels in the order the schedule first reaches them, and the entry of
        // each (step, group).
        let mut times: Vec<f32> = Vec::new();
        let mut entries: Vec<u32> = Vec::with_capacity(steps * groups);
        for level in levels.flatten_all()?.to_vec1::<f32>()? {
            let known = times
                .iter()
                .position(|time| time.to_bits() == level.to_bits());
            let entry = known.unwrap_or_else(|| {
                times.push(level);
                times.len() - 1
            });

            entries.push(entry as u32);
        }
        let distinct = times.len();

        let embedding = sinusoidal(
            &Tensor::from_vec(times, distinct, &Device::Cpu)?,
            config.frequency_dim,
        )?;
        let entries = Tensor::from_vec(entries, (steps, groups), &Device::Cpu)?;

        Ok(Self {
            frequencies: vb.constant_from("frequencies", &embedding)?,
            entries: vb.constant_from("entries", &entries)?,
            layers: vb.writable_zeros(
                "layers",
                config.dtype,
                (
                    config.layers as usize,
                    distinct,
                    config.layer_columns as usize,
                ),
            )?,
            output: vb.writable_zeros(
                "output",
                config.dtype,
                (distinct, config.output_columns as usize),
            )?,
        })
    }

    /// The groups one step reads.
    pub fn groups(&self) -> usize {
        self.entries.dims()[1]
    }

    /// The steps the entry table covers.
    pub fn steps(&self) -> usize {
        self.entries.dims()[0]
    }

    /// `[entries, frequency_dim]` FP32: the embedding the model's projections read.
    pub fn frequencies(&self) -> &Tensor {
        &self.frequencies
    }

    /// The entries step `step` reads: `[groups]` `U32`, on the tables' device.
    ///
    /// The step is a host value of the forward; its index is moved to the device and the row
    /// gathered there.
    ///
    /// # Errors
    ///
    /// A step the schedule does not have, or the gather's refusal.
    pub fn entries_of(&self, step: u32) -> Result<Tensor> {
        if step as usize >= self.steps() {
            return Err(Error::shape(
                "TimestepTable",
                format!("step {step} of a schedule of {}", self.steps()),
            ));
        }
        let index = Tensor::new(&[step], &Device::Cpu)?.to_device(self.entries.device())?;

        Ok(self.entries.index_select(&index, 0)?.flatten_all()?)
    }

    /// Block `layer`'s `[entries, layer_columns]` table, which the model writes once after the
    /// load and every step gathers from.
    ///
    /// # Errors
    ///
    /// A block the table does not hold.
    pub fn layer_table(&self, layer: usize) -> Result<Tensor> {
        let count = self.layers.dims()[0];
        if layer >= count {
            return Err(Error::shape(
                "TimestepTable",
                format!("{count} block tables, not block {layer}"),
            ));
        }
        Ok(self.layers.get(layer)?)
    }

    /// The final norm's `[entries, output_columns]` table, written once after the load.
    pub fn output_table(&self) -> &Tensor {
        &self.output
    }

    /// Block `layer`'s modulation rows at `entries` ([`Self::entries_of`]):
    /// `[groups, layer_columns]`.
    ///
    /// # Errors
    ///
    /// A block the table does not hold, or the gather's refusal.
    pub fn layer(&self, layer: usize, entries: &Tensor) -> Result<Tensor> {
        Ok(self.layer_table(layer)?.index_select(entries, 0)?)
    }

    /// The final norm's modulation rows at `entries`: `[groups, output_columns]`.
    ///
    /// # Errors
    ///
    /// The gather's refusal.
    pub fn output(&self, entries: &Tensor) -> Result<Tensor> {
        Ok(self.output.index_select(entries, 0)?)
    }
}
