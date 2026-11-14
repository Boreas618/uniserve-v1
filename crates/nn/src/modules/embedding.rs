//! The vocabulary embedding, sharded across the ranks of a tensor-parallel group.
//!
//! Each rank holds its [`vocabulary_shard`] of the table. A token outside this rank's shard
//! embeds to zeros here, every rank contributes its own rows, and the module all-reduces the
//! result when the table is sharded: the sum over the ranks is the token's embedding.

use tensor::{Module, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{self, vocabulary_shard};
use crate::var_builder::VarBuilder;

/// What a checkpoint states of an [`Embedding`], and this rank's share of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EmbeddingConfig {
    /// The rows of the whole table: the vocabulary.
    pub vocabulary: u32,
    /// The width of one row.
    pub hidden: u32,
    /// The tensor-parallel split of the vocabulary.
    pub split: Split,
}

/// A vocabulary-sharded embedding table.
pub struct Embedding {
    /// This rank's rows of the table, `[rows, hidden]`.
    weight: Tensor,
    /// The vocabulary id the shard opens at.
    first: u32,
    vocabulary: u32,
    /// The ranks the table is split over, one for a whole table.
    shards: u32,
}

impl Embedding {
    /// The embedding module at `vb`: its `weight` `[vocabulary, hidden]`, of which
    /// `config.split`'s rank reads its [`vocabulary_shard`].
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the table or holds another extent.
    pub fn load(config: &EmbeddingConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (first, rows) = vocabulary_shard(config.vocabulary, config.split);

        // A rank's rows start at its first id: read from the table's start, every rank past the
        // first would hold the first shard's rows under its own ids.
        let weight = vb.get_rows(
            (rows as usize, config.hidden as usize),
            "weight",
            first as usize,
        )?;

        Ok(Self {
            weight,
            first,
            vocabulary: config.vocabulary,
            shards: config.split.size.max(1),
        })
    }

    /// The table this rank holds, `[rows, hidden]`.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// This rank's shard: `(first id, rows)`.
    pub fn shard(&self) -> (u32, u32) {
        (self.first, self.weight.dims()[0] as u32)
    }

    /// The vocabulary the table spans.
    pub const fn vocabulary(&self) -> u32 {
        self.vocabulary
    }

    /// The ranks the table is split over, one for a whole table.
    pub const fn shards(&self) -> u32 {
        self.shards
    }

    /// Whether the table is sharded across ranks.
    pub fn is_sharded(&self) -> bool {
        self.shard().1 != self.vocabulary
    }

    /// The embedding rows of `ids`, one column of `U32` token ids: `[ids, hidden]` in the table's
    /// element type, all-reduced across the shard's ranks.
    ///
    /// # Errors
    ///
    /// Ids that are not one column, or a sharded table in a forward that has no group.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let out = functional::embedding(ids, &self.weight, self.first)?;

        if self.is_sharded() {
            if collective::split().is_none() {
                return Err(Error::Unsupported(
                    "Embedding: a vocabulary-sharded table needs the group its ranks sum over"
                        .to_owned(),
                ));
            }
            collective::all_reduce(&out)?;
        }
        Ok(out)
    }
}

impl Module for Embedding {
    fn forward(&self, ids: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, ids)?)
    }
}
