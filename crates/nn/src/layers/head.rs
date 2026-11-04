//! The vocabulary head: project selected hidden rows onto the vocabulary.
//!
//! The head is a column-parallel projection across the tensor-parallel ranks: each rank holds a
//! contiguous slice of the vocabulary rows, projects into its slice, the ranks all-gather the
//! slices, and the padded tail the sharding added is removed (specs/worker.md 9, "lm_head 与
//! logits"; the current `VocabParallelHead`). The row selection itself is the executor's: it
//! gathers the rows whose logits the batch asks for before calling the head, so the head reads a
//! `[rows, hidden]` view and a graph replays over a fixed row count.

use ops::logits::{self, GatherDescriptor, LogitsGather, UnpadDescriptor, VocabUnpad};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::Linear;
use crate::weights::Weight;

/// A vocabulary-parallel output projection.
pub struct ParallelLMHead {
    weight: Weight,
    first: u32,
    local: u32,
    vocabulary: u32,
    padded: u32,
}

/// Elements one `logits_gather` / `logits_unpad` program handles.
pub const BLOCK: u32 = 1024;

impl ParallelLMHead {
    /// The head whose rank holds `local` vocabulary rows starting at id `first`, padded to `padded`.
    ///
    /// `padded` is the shard's width after the projection: the local vocabulary rounded up to the
    /// matmul's alignment, so a rank computes a whole number of tiles. The gathered width is
    /// `padded × ranks`.
    pub fn new(
        weight: Weight,
        first: u32,
        local: u32,
        vocabulary: u32,
        padded: u32,
    ) -> Result<Self> {
        if weight.shape().rank() != 2 || weight.shape().dim(0) != u64::from(padded) {
            return Err(Error::shape(
                "ParallelLMHead",
                format!(
                    "the weight is {:?}, [{padded}, hidden] expected for the padded shard",
                    weight.shape().dims()
                ),
            ));
        }
        if padded < local {
            return Err(Error::shape(
                "ParallelLMHead",
                format!("a shard padded to {padded} is narrower than its {local} rows"),
            ));
        }
        if first.checked_add(local).is_none_or(|end| end > vocabulary) {
            return Err(Error::shape(
                "ParallelLMHead",
                format!("a shard at {first} of {local} rows is past the vocabulary {vocabulary}"),
            ));
        }
        Ok(Self {
            weight,
            first,
            local,
            vocabulary,
            padded,
        })
    }

    /// The width of this rank's padded shard.
    pub const fn padded(&self) -> u32 {
        self.padded
    }

    /// The vocabulary size of the whole model.
    pub const fn vocabulary(&self) -> u32 {
        self.vocabulary
    }

    /// Project `hidden` into `out`, gathering the ranks' shards.
    ///
    /// `hidden` is `[rows, hidden]` (the rows the batch asks logits for), `projected` is this rank's
    /// `[rows, padded]` output, `gathered` is the all-gather's `[rows, padded × ranks]` buffer (only
    /// when the group has more than one rank), and `out` is `[rows, vocabulary]`.
    pub fn forward(
        &self,
        f: &Forward,
        hidden: TensorView,
        projected: TensorView,
        gathered: Option<TensorView>,
        out: TensorView,
    ) -> Result<()> {
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
        if projected.shape.rank() != 2 || projected.shape.dim(0) != rows {
            return Err(Error::shape(
                "ParallelLMHead",
                format!(
                    "the shard output is {:?}, [{rows}, {padded}] expected",
                    projected.shape.dims(),
                    padded = self.padded
                ),
            ));
        }
        // The projection itself: the same dense operator as any other linear layer, with the
        // vocabulary shard as its output width.
        let linear = Linear::new(self.weight, None, ops::gemm::WeightOrder::NxK)?;
        linear.forward(f, hidden, projected)?;

        let group = f.group;
        let ranks = group.map_or(1, crate::Group::ranks);
        let gathered = if ranks > 1 {
            let gathered = gathered.ok_or_else(|| {
                Error::shape(
                    "ParallelLMHead",
                    format!(
                        "a gather over {ranks} ranks needs a {rows} × {width} buffer",
                        width = u64::from(self.padded) * ranks
                    ),
                )
            })?;
            let expected = u64::from(self.padded) * ranks;
            if gathered.shape.rank() != 2
                || gathered.shape.dim(0) != rows
                || gathered.shape.dim(1) != expected
            {
                return Err(Error::shape(
                    "ParallelLMHead",
                    format!(
                        "the gather buffer is {:?}, [{rows}, {expected}] expected",
                        gathered.shape.dims()
                    ),
                ));
            }
            group
                .ok_or_else(|| {
                    Error::Unsupported(
                        "ParallelLMHead: a sharded head needs the group its ranks gather over"
                            .to_owned(),
                    )
                })?
                .all_gather(projected, gathered, f.stream)?;
            gathered
        } else {
            projected
        };

        // Remove the padding: the gathered width is `padded × ranks`, of which only the first
        // `vocabulary` columns are real. The shards are contiguous, so the first `vocabulary`
        // columns of the gathered matrix are exactly the vocabulary, in order.
        let width = u32::try_from(gathered.shape.dim(1)).map_err(|_| {
            Error::shape(
                "ParallelLMHead",
                "a gathered width above u32::MAX".to_owned(),
            )
        })?;
        let descriptor = self.unpad_descriptor(gathered.dtype, out.dtype, width);
        let args = logits::UnpadArgs { gathered, out };

        f.operate::<VocabUnpad>(&descriptor, args)
    }

    /// The descriptor the vocabulary shard's projection launches with.
    ///
    /// The head projects through an ordinary dense product whose output width is the rank's shard,
    /// so its descriptor is a [`Linear`]'s: this answers that layer's descriptor for the element
    /// types a deployment names, without the caller building the layer. It is fallible where
    /// [`Linear::new`] is, which for the head's own two-dimensional weight is never.
    pub fn projection_descriptor(
        &self,
        input: DType,
        output: DType,
    ) -> Result<ops::gemm::Descriptor> {
        Ok(Linear::descriptor_of(
            self.weight.shape().last() as u32,
            self.weight.dtype(),
            ops::gemm::WeightOrder::NxK,
            false,
            input,
            output,
        ))
    }

    /// The descriptor the vocabulary unpad launches with, for a gathered matrix of `gathered_width`
    /// columns.
    ///
    /// The gathered width is `padded × ranks`, so it is a launch value that follows the deployment's
    /// rank count rather than a property of the head; the vocabulary and the block are the head's.
    /// The descriptor the unpadding launch uses: the gathered logits' element type, the type the
    /// sampler's candidate buffer carries, the vocabulary, the gathered width, the block and PDL.
    ///
    /// The two element types are separate because the head's product is in the model's dtype and the
    /// sampler reads FP32; [`ops::logits::VocabUnpad`] widens here, once.
    pub fn unpad_descriptor(
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

    /// The unpad's descriptor, from the vocabulary's size alone.
    ///
    /// The gathered width is `padded × ranks` — a launch value that follows the deployment's rank
    /// count — so it is an argument; the vocabulary and the block are the checkpoint's and the
    /// program's.
    pub fn unpad_descriptor_of(
        vocabulary: u32,
        dtype: DType,
        output: DType,
        gathered_width: u32,
    ) -> UnpadDescriptor {
        UnpadDescriptor {
            dtype,
            output,
            vocabulary,
            padded: gathered_width,
            block: 1024,
            pdl: crate::layers::launch::PDL,
        }
    }

    /// The offset of this rank's shard in the gathered vocabulary; 0 for an unsharded head.
    pub const fn first(&self) -> u32 {
        self.first
    }

    /// The rows this rank's projection computes before padding.
    pub const fn local(&self) -> u32 {
        self.local
    }
}

/// Gather `rows` of `hidden` into a `[rows, hidden]` view, for the head.
///
/// The executor calls this when it assembles a batch's logits requests: the operator moves whole
/// rows, so two implementations agree bit for bit.
pub fn gather_rows(
    f: &Forward,
    hidden: TensorView,
    rows: TensorView,
    out: TensorView,
) -> Result<()> {
    let columns = hidden.last();
    let count = rows.numel();
    // The operator's indices are one axis and its output two, because it derives the row count from
    // `out`. A caller holding `[1, count]` - the shape an output of one row needs - must flatten it
    // itself: the same bytes as a two-axis view reach the kernel as two axes and are refused there
    // (`expected 1 axes, found 2`), which names neither the caller nor the fix.
    if rows.shape.rank() != 1 {
        return Err(Error::shape(
            "gather_rows",
            format!(
                "rows is {:?}, and the operator takes a flat [count] index view",
                rows.shape.dims()
            ),
        ));
    }
    if out.shape.rank() != 2 || out.shape.dim(0) != count || out.shape.dim(1) != columns {
        return Err(Error::shape(
            "gather_rows",
            format!(
                "out is {:?}, [{count}, {columns}] expected",
                out.shape.dims()
            ),
        ));
    }
    let descriptor = GatherDescriptor {
        dtype: hidden.dtype,
        columns: u32::try_from(columns)
            .map_err(|_| Error::shape("gather_rows", "a hidden width above u32::MAX".to_owned()))?,
        block: BLOCK,
        pdl: crate::layers::launch::PDL,
    };
    let args = logits::GatherArgs { hidden, rows, out };

    f.operate::<LogitsGather>(&descriptor, args)
}
