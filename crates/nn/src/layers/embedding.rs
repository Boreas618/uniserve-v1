//! The vocabulary-parallel embedding: a masked gather of the vocabulary shard this rank holds.
//!
//! The vocabulary is sharded across tensor-parallel ranks by [`vocabulary_shard`]. A token outside
//! this rank's shard is written as zeros by the operator, every rank contributes its own rows, and
//! the layer all-reduces the result when it is sharded: the sum over ranks is the token's
//! embedding.

use ops::Operand;
use ops::embedding::{self, Descriptor, Embedding as EmbeddingOp};
use uniserve_tensor::DType;

use crate::error::{Error, Result};
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::launch::{EMBEDDING_TOKENS, PDL};
use crate::weights::{Scope, Weight};

/// The rows of a `vocabulary`-row table `split`'s rank holds, as `(first id, rows)`.
///
/// Every rank holds `ceil(vocabulary / size)` contiguous rows, and the last ones the remainder, so
/// a rank past the vocabulary's end holds none. The embedding and the vocabulary head shard by this
/// one rule, so a token's row and its logit column are on the same rank.
pub fn vocabulary_shard(vocabulary: u32, split: Split) -> (u32, u32) {
    let per_rank = vocabulary.div_ceil(split.size.max(1));
    let first = split.rank.saturating_mul(per_rank).min(vocabulary);
    (first, per_rank.min(vocabulary - first))
}

/// A vocabulary-sharded embedding table.
pub struct VocabParallelEmbedding {
    weight: Weight,
    first: u32,
    rows: u32,
    hidden: u32,
    vocabulary: u32,
}

impl VocabParallelEmbedding {
    /// The embedding module at `scope`: its `weight` `[vocabulary, hidden]`, of which `split`'s
    /// rank reads its [`vocabulary_shard`].
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the table or holds another extent.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        vocabulary: u32,
        hidden: u32,
        split: Split,
    ) -> Result<Self> {
        let (first, rows) = vocabulary_shard(vocabulary, split);
        // A rank's rows start at its first id: read from the table's start, every rank past the
        // first would hold the first shard's rows under its own ids.
        let weight = scope.rows(
            "weight",
            u64::from(first),
            &[u64::from(rows), u64::from(hidden)],
        )?;
        Ok(Self {
            weight,
            first,
            rows,
            hidden,
            vocabulary,
        })
    }

    /// The table this rank holds, `[rows, hidden]`.
    pub fn weight(&self) -> Weight {
        self.weight
    }

    /// This rank's shard: `(first id, rows)`.
    pub const fn shard(&self) -> (u32, u32) {
        (self.first, self.rows)
    }

    /// The vocabulary the table spans.
    pub const fn vocabulary(&self) -> u32 {
        self.vocabulary
    }

    /// Whether the table is sharded across ranks.
    pub const fn is_sharded(&self) -> bool {
        self.rows != self.vocabulary
    }

    /// The descriptor [`Self::forward`] launches with, for an output of `out` and ids of `ids`.
    ///
    /// The token count is not in it — the ids' length is a launch value; `rows` is this rank's
    /// shard, `hidden` its width and `first` the vocabulary id the shard opens at.
    fn descriptor(&self, out: DType, ids: DType) -> Descriptor {
        Descriptor {
            dtype: out,
            ids,
            hidden: self.hidden,
            first: self.first,
            rows: self.rows,
            block: EMBEDDING_TOKENS,
            pdl: PDL,
        }
    }

    /// The embedding rows of `ids`, `[ids, hidden]` in the table's element type, all-reduced across
    /// the shard's ranks.
    ///
    /// # Errors
    ///
    /// Ids that are not one column, an arena that cannot hold the rows, or a sharded table on a
    /// forward with no group.
    pub fn forward(&self, f: &Forward, ids: Operand) -> Result<Operand> {
        let hidden = u64::from(self.hidden);
        if ids.shape.rank() != 1 {
            return Err(Error::shape(
                "VocabParallelEmbedding",
                format!("ids are a column, not {:?}", ids.shape.dims()),
            ));
        }
        let out = f.empty(self.weight.dtype(), ops::Shape::new(&[ids.numel(), hidden]))?;
        let descriptor = self.descriptor(out.dtype, ids.dtype);
        let args = embedding::Args {
            ids,
            weight: self.weight.view(),
            out,
        };

        f.operate::<EmbeddingOp>(&descriptor, args)?;
        if self.is_sharded() {
            if f.split().is_none() {
                return Err(Error::Unsupported(
                    "VocabParallelEmbedding: a vocabulary-sharded table needs the group its ranks \
                     sum over"
                        .to_owned(),
                ));
            }
            f.all_reduce(out)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shards_tile_the_vocabulary_in_order() {
        for (vocabulary, size) in [(151_936_u32, 4_u32), (151_937, 4), (30, 2), (7, 4), (5, 1)] {
            let mut next = 0;
            for rank in 0..size {
                let (first, rows) = vocabulary_shard(vocabulary, Split { size, rank });
                assert_eq!(first, next.min(vocabulary), "{vocabulary} over {size}");
                assert!(rows <= vocabulary.div_ceil(size));
                next = first + rows;
            }
            assert_eq!(
                next, vocabulary,
                "{vocabulary} over {size} covers every row"
            );
        }
        // The last rank of an uneven vocabulary keeps the remainder.
        assert_eq!(
            vocabulary_shard(151_937, Split { size: 4, rank: 3 }),
            (113_955, 37_982)
        );
        // A rank past the end of a small vocabulary holds no row.
        assert_eq!(vocabulary_shard(5, Split { size: 4, rank: 3 }), (5, 0));
    }
}
