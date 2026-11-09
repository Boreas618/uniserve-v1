//! The vocabulary head: project selected hidden rows onto the vocabulary.
//!
//! The head is a column-parallel projection across the tensor-parallel ranks: each rank holds its
//! [`vocabulary_shard`] of the vocabulary rows, projects into its slice padded to a whole number
//! of eight-row tiles, the ranks all-gather the slices, and the padding is removed. The head owns
//! the shard rule, the padding and the buffers its projection and gather write, which it reserves
//! from the forward's arena. The row selection itself is the worker's: it gathers the rows whose
//! logits the batch asks for before calling the head, so the head reads a `[rows, hidden]` view and a
//! graph replays over a fixed row count.

use ops::logits::{self, UnpadDescriptor, VocabUnpad};
use ops::{DType, Shape, TensorView};

use crate::error::{Error, Result};
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::Linear;
use crate::layers::embedding::{VocabParallelEmbedding, vocabulary_shard};
use crate::weights::{Scope, Weight};

/// A vocabulary-parallel output projection.
pub struct ParallelLMHead {
    weight: Weight,
    vocabulary: u32,
    padded: u32,
}

/// Elements one `logits_gather` / `logits_unpad` program handles.
const BLOCK: u32 = 1024;

/// The rows a shard's projection is padded to: whole eight-row tiles of the product.
const PADDING: u32 = 8;

impl ParallelLMHead {
    /// The head module at `scope`: its `weight` `[vocabulary, hidden]`, of which `split`'s rank
    /// reads its [`vocabulary_shard`].
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the table, or the rank's shard is not a whole number of tiles: the
    /// checkpoint's rows are contiguous, so a shard's padded rows would be another rank's.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        vocabulary: u32,
        hidden: u32,
        split: Split,
    ) -> Result<Self> {
        let (first, rows) = vocabulary_shard(vocabulary, split);
        let weight = scope.rows(
            "weight",
            u64::from(first),
            &[u64::from(rows), u64::from(hidden)],
        )?;
        Self::over(weight, rows, vocabulary)
    }

    /// The head that shares `embedding`'s table: a checkpoint that ties its word embeddings.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], for the embedding's shard.
    pub fn tied(embedding: &VocabParallelEmbedding) -> Result<Self> {
        let (_, rows) = embedding.shard();
        Self::over(embedding.weight(), rows, embedding.vocabulary())
    }

    /// The head over `weight`, a shard of `rows` vocabulary rows.
    fn over(weight: Weight, rows: u32, vocabulary: u32) -> Result<Self> {
        let padded = rows.next_multiple_of(PADDING);
        if padded != rows {
            return Err(Error::Unsupported(format!(
                "ParallelLMHead: a vocabulary shard of {rows} rows needs {padded} padded rows; the \
                 checkpoint's rows are contiguous, so a rank whose shard is not its padded width is \
                 not loadable"
            )));
        }
        Ok(Self {
            weight,
            vocabulary,
            padded,
        })
    }

    /// The width of this rank's padded shard.
    pub const fn padded(&self) -> u32 {
        self.padded
    }

    /// Project `hidden` into `out`, gathering the ranks' shards.
    ///
    /// `hidden` is `[rows, hidden]` (the rows the batch asks logits for) and `out` is
    /// `[rows, vocabulary]`. The head reserves this rank's `[rows, padded]` projection and, when
    /// the forward's group has more than one rank, the `[rows, padded × ranks]` gather from the
    /// forward's arena, in that order.
    ///
    /// # Errors
    ///
    /// When a view does not fit the head or an operator refuses.
    pub fn forward(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()> {
        let rows = hidden.rows();
        if out.shape.rank() != 2
            || out.shape.dim(0) != rows
            || out.shape.dim(1) != u64::from(self.vocabulary)
        {
            return Err(Error::shape(
                "ParallelLMHead",
                format!(
                    "out is {:?}, [{rows}, {vocabulary}] expected",
                    out.shape.dims(),
                    vocabulary = self.vocabulary
                ),
            ));
        }
        let ranks = f.split().map_or(1, |split| split.ranks);
        let arena = f.arena();
        let projected = arena.tensor(hidden.dtype, Shape::new(&[rows, u64::from(self.padded)]))?;
        // The projection itself: the same dense operator as any other linear layer, with the
        // vocabulary shard as its output width.
        let linear = Linear::new(self.weight, None, ops::gemm::WeightOrder::NxK)?;
        linear.forward(f, hidden, projected)?;

        let gathered = if ranks > 1 {
            let gathered = arena.tensor(
                hidden.dtype,
                Shape::new(&[rows, u64::from(self.padded) * ranks]),
            )?;
            f.all_gather(projected, gathered)?;
            gathered
        } else {
            projected
        };

        // Remove the padding and order the columns: the gather lays the ranks' `[rows, padded]`
        // shards one after another, so a row's vocabulary is spread over every shard, and the
        // unpad reads each column from the shard that holds it.
        let width = u32::try_from(gathered.shape.dim(1)).map_err(|_| {
            Error::shape(
                "ParallelLMHead",
                "a gathered width above u32::MAX".to_owned(),
            )
        })?;
        let descriptor = self.unpad_descriptor(gathered.dtype, out.dtype, width);
        let args = logits::UnpadArgs {
            gathered,
            out,
            ranks: u32::try_from(ranks).map_err(|_| {
                Error::shape("ParallelLMHead", format!("{ranks} ranks exceed u32::MAX"))
            })?,
        };

        f.operate::<VocabUnpad>(&descriptor, args)
    }
    /// The descriptor the vocabulary unpad launches with, for a gathered matrix of `gathered_width`
    /// columns: the gathered logits' element type, the type the sampler's candidate buffer carries,
    /// the vocabulary, the gathered width, the block and PDL.
    ///
    /// The gathered width is `padded × ranks`, so it is a launch value that follows the deployment's
    /// rank count rather than a property of the head; the vocabulary and the block are the head's.
    ///
    /// The two element types are separate because the head's product is in the model's dtype and the
    /// sampler reads FP32; [`ops::logits::VocabUnpad`] widens here, once.
    fn unpad_descriptor(
        &self,
        dtype: DType,
        output: DType,
        gathered_width: u32,
    ) -> UnpadDescriptor {
        UnpadDescriptor {
            dtype,
            output,
            vocabulary: self.vocabulary,
            padded: gathered_width,
            block: BLOCK,
            pdl: crate::layers::launch::PDL,
        }
    }
}
