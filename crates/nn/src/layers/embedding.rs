//! The vocabulary embedding: a masked gather of the vocabulary shard this rank holds.
//!
//! The vocabulary is sharded across tensor-parallel ranks. A token outside this rank's shard is
//! written as zeros by the operator, every rank contributes its own rows, and the layer
//! all-reduces the result when it has a group (specs/worker.md 9, "词嵌入"): the sum over ranks is
//! the token's embedding.

use ops::embedding::{self, Descriptor, Embedding as EmbeddingOp};
use ops::{Op, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::{EMBEDDING_TOKENS, PDL};
use crate::weights::Weight;

/// A vocabulary-sharded embedding table.
pub struct Embedding {
    weight: Weight,
    first: u32,
    rows: u32,
    hidden: u32,
    vocabulary: u32,
}

impl Embedding {
    /// The table of `shape` `[rows, hidden]` starting at vocabulary id `first`, out of `vocabulary`.
    ///
    /// `rows` is this rank's shard; a rank that holds the whole vocabulary states `first = 0` and
    /// `rows = vocabulary`, and no collective follows its lookups.
    pub fn new(weight: Weight, first: u32, rows: u32, vocabulary: u32) -> Result<Self> {
        if weight.shape().rank() != 2 {
            return Err(Error::shape(
                "Embedding",
                format!("a table is a matrix, not {:?}", weight.shape().dims()),
            ));
        }
        if weight.shape().dim(0) != u64::from(rows) {
            return Err(Error::shape(
                "Embedding",
                format!("{} rows for a shard of {rows}", weight.shape().dim(0)),
            ));
        }
        let hidden = u32::try_from(weight.shape().dim(1)).map_err(|_| {
            Error::shape("Embedding", "a vocabulary width above u32::MAX".to_owned())
        })?;
        let shard_end = first.checked_add(rows).ok_or_else(|| {
            Error::shape(
                "Embedding",
                "the shard's vocabulary range overflows".to_owned(),
            )
        })?;
        if shard_end > vocabulary {
            return Err(Error::shape(
                "Embedding",
                format!("a shard ending at {shard_end} is past the vocabulary's {vocabulary}"),
            ));
        }
        Ok(Self {
            weight,
            first,
            rows,
            hidden,
            vocabulary,
        })
    }

    /// The width of one embedding row.
    pub const fn hidden(&self) -> u32 {
        self.hidden
    }

    /// The vocabulary size of the whole model.
    pub const fn vocabulary(&self) -> u32 {
        self.vocabulary
    }

    /// Whether the table is sharded across ranks.
    pub const fn is_sharded(&self) -> bool {
        self.rows != self.vocabulary
    }

    /// `out = embed(ids)`, all-reduced across the shard's ranks.
    pub fn forward(&self, f: &Forward, ids: TensorView, out: TensorView) -> Result<()> {
        let hidden = u64::from(self.hidden);
        if ids.shape.rank() != 1 {
            return Err(Error::shape(
                "Embedding",
                format!("ids are a column, not {:?}", ids.shape.dims()),
            ));
        }
        out.require_row("Embedding", out.dtype, hidden)
            .map_err(Error::from)?;
        if out.rows() != ids.numel() {
            return Err(Error::shape(
                "Embedding",
                format!("{} ids but {} embedding rows", ids.numel(), out.rows()),
            ));
        }
        let descriptor = Descriptor {
            dtype: out.dtype,
            ids: ids.dtype,
            hidden: self.hidden,
            first: self.first,
            rows: self.rows,
            block: EMBEDDING_TOKENS,
            pdl: PDL,
        };
        let args = embedding::Args {
            ids,
            weight: self.weight.view(),
            out,
        };
        let kernel = f.kernels().op::<EmbeddingOp>(&descriptor)?;
        EmbeddingOp::launch(kernel, &descriptor, args, f.stream)?;
        if self.is_sharded() {
            let group = f.group.ok_or_else(|| {
                Error::Unsupported(
                    "Embedding: a vocabulary-sharded table needs the group its ranks sum over"
                        .to_owned(),
                )
            })?;
            group.all_reduce_sum(out, f.stream)?;
        }
        Ok(())
    }
}
