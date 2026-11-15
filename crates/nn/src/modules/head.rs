//! The vocabulary head: project the rows a batch asks logits for onto the vocabulary.
//!
//! The head is a column-parallel projection across the tensor-parallel ranks: each rank holds
//! its [`vocabulary_shard`] of the vocabulary rows and projects into its shard, the ranks
//! all-gather the shards, and the gathered columns are read back into vocabulary order.
//! Selecting the rows is the caller's (a row gather before the head), so the head reads
//! `[rows, hidden]`.
//!
//! The dense product takes any output width, so a shard is its vocabulary rows as the
//! checkpoint holds them. The gather lays equal blocks side by side, one per rank, so a split
//! head holds `ceil(vocabulary / ranks)` rows on every rank: a last shard that is shorter has no
//! rows in the checkpoint to fill its block with, and such a split is refused at load.

use tensor::{Module, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{self, WeightOrder, vocabulary_shard};
use crate::modules::embedding::Embedding;
use crate::var_builder::VarBuilder;

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
    /// The ranks the vocabulary is split over, one for a whole head. The forward's group may
    /// split something else (a pipeline's stages, a sequence's members), so the head gathers by
    /// its own split and never by the group's.
    shards: u32,
}

impl LmHead {
    /// The head module at `vb`: its `weight` `[vocabulary, hidden]`, of which `config.split`'s
    /// rank reads its [`vocabulary_shard`].
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the table, or the split leaves the rank a shorter shard than
    /// its peers (module documentation).
    pub fn load(config: &LmHeadConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (first, rows) = vocabulary_shard(config.vocabulary, config.split);
        let weight = vb.get_rows(
            (rows as usize, config.hidden as usize),
            "weight",
            first as usize,
        )?;

        Self::over(weight, config.vocabulary, config.split.size())
    }

    /// The head that shares `embedding`'s table: a checkpoint that ties its word embeddings.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], for the embedding's shard.
    pub fn tied(embedding: &Embedding) -> Result<Self> {
        Self::over(
            embedding.weight().clone(),
            embedding.vocabulary(),
            embedding.shards(),
        )
    }

    /// The head over `weight`, one shard of the rows of a `vocabulary`-row table split over
    /// `shards` ranks.
    fn over(weight: Tensor, vocabulary: u32, shards: u32) -> Result<Self> {
        let shards = shards.max(1);
        let rows = weight.dims().first().copied().unwrap_or(0);
        let block = vocabulary.div_ceil(shards) as usize;

        // The gather's blocks are equal, and a shorter shard has no checkpoint rows to fill its
        // block with.
        if shards > 1 && rows != block {
            return Err(Error::Unsupported(format!(
                "LmHead: a vocabulary shard of {rows} rows beside peers of {block}; the \
                 checkpoint's rows are contiguous, so a split whose last shard is shorter than \
                 its peers is not loadable"
            )));
        }
        Ok(Self {
            weight,
            vocabulary,
            shards,
        })
    }

    /// The logits of `hidden` `[rows, hidden]`, the rows the batch asks logits for:
    /// `[rows, vocabulary]` in FP32, the ranks' shards gathered.
    ///
    /// The projection is the same dense product as any other linear projection, with the
    /// vocabulary shard as its output width; a head split over several ranks all-gathers the
    /// shards over the forward's group, which is then the vocabulary's. Logits are FP32 because
    /// the sampler reads FP32: the unpad widens the model's product once.
    ///
    /// # Errors
    ///
    /// When `hidden` does not fit the head, an operator refuses, or a split head runs in a
    /// forward whose group is not its split.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let ranks = u64::from(self.shards);
        if ranks > 1 && collective::split().map(|(_, members)| members) != Some(ranks) {
            return Err(Error::shape(
                "LmHead",
                format!(
                    "a head split over {ranks} ranks runs in a forward whose group is {:?}",
                    collective::split()
                ),
            ));
        }
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
