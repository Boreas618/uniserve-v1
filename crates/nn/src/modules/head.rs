//! The vocabulary head: project the rows a batch asks logits for onto the vocabulary.
//!
//! The head is a column-parallel projection across the tensor-parallel ranks: each rank holds
//! its [`vocabulary_shard`] of the vocabulary rows and projects into its shard, the ranks
//! all-gather the shards, and the padding the shards hold is removed. Selecting the rows is the
//! caller's (a row gather before the head), so the head reads `[rows, hidden]`.

use tensor::{Module, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{self, WeightOrder, vocabulary_shard};
use crate::modules::embedding::Embedding;
use crate::var_builder::VarBuilder;

/// The rows a shard's projection is padded to: whole eight-row tiles of the product.
const PADDING: u32 = 8;

/// What a checkpoint states of an [`LmHead`], and this rank's share of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LmHeadConfig {
    /// The vocabulary a logit row spans.
    pub vocabulary: u32,
    /// The width of the hidden rows it projects.
    pub hidden: u32,
    /// The tensor-parallel split of the vocabulary.
    pub split: Split,
}

/// A vocabulary-parallel output projection.
pub struct LmHead {
    /// This rank's shard of the vocabulary rows, `[rows, hidden]`.
    weight: Tensor,
    vocabulary: u32,
}

impl LmHead {
    /// The head module at `vb`: its `weight` `[vocabulary, hidden]`, of which `config.split`'s
    /// rank reads its [`vocabulary_shard`].
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the table, or the rank's shard is not a whole number of tiles:
    /// the checkpoint's rows are contiguous, so a shard's padded rows would be another rank's.
    pub fn load(config: &LmHeadConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (first, rows) = vocabulary_shard(config.vocabulary, config.split);
        let weight = vb.get_rows(
            (rows as usize, config.hidden as usize),
            "weight",
            first as usize,
        )?;

        Self::over(weight, config.vocabulary)
    }

    /// The head that shares `embedding`'s table: a checkpoint that ties its word embeddings.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], for the embedding's shard.
    pub fn tied(embedding: &Embedding) -> Result<Self> {
        Self::over(embedding.weight().clone(), embedding.vocabulary())
    }

    /// The head over `weight`, a shard of the rows of a `vocabulary`-row table.
    fn over(weight: Tensor, vocabulary: u32) -> Result<Self> {
        let rows = weight.dims().first().copied().unwrap_or(0) as u32;
        let padded = rows.next_multiple_of(PADDING);

        if padded != rows {
            return Err(Error::Unsupported(format!(
                "LmHead: a vocabulary shard of {rows} rows needs {padded} padded rows; the \
                 checkpoint's rows are contiguous, so a rank whose shard is not its padded \
                 width is not loadable"
            )));
        }
        Ok(Self { weight, vocabulary })
    }

    /// The width of this rank's shard: its vocabulary rows, a whole number of tiles.
    pub fn padded(&self) -> u32 {
        self.weight.dims()[0] as u32
    }

    /// The logits of `hidden` `[rows, hidden]`, the rows the batch asks logits for:
    /// `[rows, vocabulary]` in FP32, the ranks' shards gathered.
    ///
    /// The projection is the same dense product as any other linear projection, with the
    /// vocabulary shard as its output width; when the forward's group has more than one rank
    /// the shards are all-gathered. Logits are FP32 because the sampler reads FP32: the unpad
    /// widens the model's product once.
    ///
    /// # Errors
    ///
    /// When `hidden` does not fit the head or an operator refuses.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let ranks = collective::split().map_or(1, |(_, ranks)| ranks);
        let projected = functional::linear(hidden, &self.weight, None, WeightOrder::NxK)?;

        let gathered = if ranks > 1 {
            let (rows, width) = (projected.dims()[0], projected.dims()[1]);
            let gathered = ops::empty((rows, width * ranks as usize), projected.dtype())?;

            collective::all_gather(&projected, &gathered)?;
            gathered
        } else {
            projected
        };

        // The gather lays the ranks' shards one after another, so a row's vocabulary is spread
        // over every shard, and the unpad reads each column from the shard that holds it.
        let ranks = u32::try_from(ranks)
            .map_err(|_| Error::shape("LmHead", format!("{ranks} ranks exceed u32::MAX")))?;
        functional::vocab_unpad(&gathered, self.vocabulary, ranks)
    }
}

impl Module for LmHead {
    fn forward(&self, hidden: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, hidden)?)
    }
}
