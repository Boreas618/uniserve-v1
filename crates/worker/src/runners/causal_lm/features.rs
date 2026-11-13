//! The feature buffer: the rows of a prefill's image blocks, which its forward takes in place of
//! the embeddings of their placeholder tokens, and the rows' rotary coordinates.
//!
//! An image block of a prefill (`ipc::batch::VisionRows`) names the encoder output its rows hold
//! by its position in the encoder-output table of the request whose row holds the block; the
//! rank's tables hold that output's block, which holds the image's features and their rotary
//! coordinates from position 0 (`runners::vision`, module documentation). Before the replay the
//! runner copies each block's features and coordinates into this buffer at the block's token
//! rows and marks those rows, on its stream, and the forward of a prefill bucket reads
//!
//! ```text
//! embedded[t]          = marked[t] ? features[t] : embed_input_ids(ids)[t]
//! rope_positions[a, t] = positions[t] + (marked[t] ? coordinates[t, a] : 0)
//! ```
//!
//! An image row's position is the logical position its block starts at, so the coordinates
//! move there. The buffer lies at fixed addresses, so a captured prefill reads the images of
//! whatever batch it replays for:
//!
//! | Buffer | Type and extent |
//! | --- | --- |
//! | features | `[tokens, hidden]` in the model's activation type |
//! | coordinates | `[tokens, axes]` I32 |
//! | marks | `[tokens, 1]` U8, one on an image row |
//!
//! `tokens` are the largest bucket's. The marks start zero, and a step after one that marked
//! rows clears them first: a text prefill reads no image of an earlier one.

use std::sync::Arc;

use ipc::batch::{Batch, Rows, VisionRows};
use nn::family::TextModel;
use platform::Stream;
use uniserve_tensor::{DType, Tensor};

use crate::memory::allocation::Storage;
use crate::memory::buffers::BufferMemory;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runners::{DeviceCopy, address, element_bytes, issue};
use crate::{Error, Result};

/// The alignment of each buffer in the feature buffer's memory.
const ALIGNMENT: u64 = backend::allocator::ALIGNMENT;

/// The runner's feature buffer (module documentation).
pub(super) struct Features {
    /// The rank's media memory, where the encoder outputs lie.
    memory: Arc<BufferMemory>,
    /// The class of encoder outputs.
    class: u8,
    features: Tensor,
    coordinates: Tensor,
    marks: Tensor,
    /// The device memory of the three; `None` for a runner whose graphs hold no device, whose
    /// buffers are meta tensors at their offsets from 0.
    storage: Option<Arc<Storage>>,
    /// The tokens, the width of a feature row and the rotary axes.
    extents: (usize, usize, usize),
    /// Whether the marks hold rows an earlier step set.
    marked: bool,
    /// Whether copies of encoder outputs were issued that no wait has covered yet.
    copied: bool,
}

/// The byte offsets of the features, the coordinates and the marks of `tokens` rows of a model
/// of `hidden`-wide rows of `dtype` over `axes` rotary axes, and the bytes of the three.
fn offsets(tokens: usize, hidden: usize, dtype: DType, axes: usize) -> Result<([u64; 3], u64)> {
    let element = element_bytes(dtype)?;

    let mut offsets = [0; 3];
    let mut bytes = 0;
    for (offset, length) in offsets
        .iter_mut()
        .zip([hidden as u64 * element, axes as u64 * 4, 1])
    {
        *offset = bytes;
        bytes += (tokens as u64 * length).next_multiple_of(ALIGNMENT);
    }
    Ok((offsets, bytes))
}

/// The slot of the call whose row holds every token of `block`.
fn holder(rows: &Rows, block: &VisionRows) -> Result<u32> {
    let (first, end) = (
        i64::from(block.token),
        i64::from(block.token) + i64::from(block.rows),
    );

    rows.query_offsets
        .windows(2)
        .position(|row| i64::from(row[0]) <= first && first < end && end <= i64::from(row[1]))
        .and_then(|row| {
            rows.calls
                .iter()
                .find(|call| call.rows.contains(&(row as u32)))
        })
        .map(|call| call.slot)
        .ok_or_else(|| {
            Error::Invalid(format!(
                "an image block of tokens {first}..{end} lies in no one row of a call"
            ))
        })
}

impl Features {
    /// The bytes of device memory the feature buffer of `model` takes for buckets of at most
    /// `tokens` tokens; zero for a model without a vision tower.
    pub(super) fn bytes(model: &dyn TextModel, tokens: usize) -> Result<u64> {
        match model.vision() {
            Some(vision) => Ok(offsets(
                tokens,
                model.hidden() as usize,
                model.dtype(),
                vision.position_axes() as usize,
            )?
            .1),
            None => Ok(0),
        }
    }

    /// The feature buffer of `model` for buckets of at most `tokens` tokens, on the device of
    /// `stream`, the runner's, with its marks zeroed there, or of meta tensors for `None`. The
    /// encoder outputs are blocks of class `class` in `memory`.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model without a vision tower; the driver's refusal of the
    /// allocation or of the fill.
    pub(super) fn new(
        model: &dyn TextModel,
        tokens: usize,
        stream: Option<&Stream>,
        memory: Arc<BufferMemory>,
        class: u8,
    ) -> Result<Self> {
        let vision = model.vision().ok_or_else(|| {
            Error::Invalid("a feature buffer lent to a runner of a model without vision".into())
        })?;
        let (hidden, dtype) = (model.hidden() as usize, model.dtype());
        let axes = vision.position_axes() as usize;
        let ([features, coordinates, marks], bytes) = offsets(tokens, hidden, dtype, axes)?;

        let storage = match stream {
            Some(stream) => {
                let bytes = usize::try_from(bytes)
                    .map_err(|_| Error::Resource("a feature buffer exceeds the address space"))?;
                let storage = Storage::linear(stream.device(), bytes)?;

                // SAFETY: the marks lie inside the allocation this value keeps.
                unsafe { platform::memset_d8(storage.address() + marks, 0, tokens, stream) }?;
                Some(storage)
            }
            None => None,
        };

        // Each buffer over the memory, or a meta tensor at its offset from 0.
        let buffer = |offset: u64, shape: (usize, usize), dtype: DType| -> Result<Tensor> {
            match &storage {
                // SAFETY: this value owns the storage and outlives the work on the tensors: the
                // runner that holds it holds the graphs captured over them, and only the copies
                // and fills `Features::issue` makes and those graphs' kernels touch it.
                Some(storage) => unsafe { storage.external(offset as usize, shape, dtype) },
                // SAFETY: a meta tensor's address is never read or written.
                None => Ok(unsafe { backend::external(offset, shape, dtype, &backend::meta()) }?),
            }
        };

        Ok(Self {
            memory,
            class,
            features: buffer(features, (tokens, hidden), dtype)?,
            coordinates: buffer(coordinates, (tokens, axes), DType::I32)?,
            marks: buffer(marks, (tokens, 1), DType::U8)?,
            storage,
            extents: (tokens, hidden, axes),
            marked: false,
            copied: false,
        })
    }

    /// The copies of `batch`'s image blocks into the buffer, through `tables`: each block's
    /// features and then its coordinates, from the encoder output of the request whose row
    /// holds it. Nothing is issued.
    ///
    /// # Errors
    ///
    /// A block that lies in no one row of a call or past the buffer, a position the tables hold
    /// no entry for, or an encoder output smaller than the block's rows.
    pub(super) fn copies(&self, batch: &Batch, tables: &BlockTables) -> Result<Vec<DeviceCopy>> {
        let (_, hidden, axes) = self.extents;
        let dtype = self.features.dtype();
        let row_bytes = hidden as u64 * element_bytes(dtype)?;

        let mut copies = Vec::with_capacity(2 * batch.columns.vision.len());
        for block in &batch.columns.vision {
            let slot = holder(&batch.columns.rows, block)?;
            let (token, rows) = (block.token as usize, block.rows as usize);
            let output = (slot, self.class, block.feature);

            let features = self
                .memory
                .lend(tables, output, false, 0, &[rows, hidden], dtype)?;
            let coordinates = self.memory.lend(
                tables,
                output,
                false,
                rows as u64 * row_bytes,
                &[rows, axes],
                DType::I32,
            )?;

            copies.push((
                address(&features)?,
                address(&self.features.narrow(0, token, rows)?)?,
                rows as u64 * row_bytes,
            ));
            copies.push((
                address(&coordinates)?,
                address(&self.coordinates.narrow(0, token, rows)?)?,
                (rows * axes * 4) as u64,
            ));
        }
        Ok(copies)
    }

    /// Issues `copies`, those of `batch`'s image blocks ([`Features::copies`]), on `stream`,
    /// ahead of the batch's replay, and marks the blocks' rows, after clearing the rows an
    /// earlier step marked. With no stream nothing is issued.
    ///
    /// # Errors
    ///
    /// The driver's refusal; the marks are then cleared by the next step.
    pub(super) fn issue(
        &mut self,
        batch: &Batch,
        copies: &[DeviceCopy],
        stream: Option<&Stream>,
    ) -> Result<()> {
        let (Some(stream), Some(_)) = (stream, &self.storage) else {
            return Ok(());
        };
        let marks = address(&self.marks)?;
        let tokens = self.extents.0;

        // SAFETY: the marks and the destinations of the copies lie in this value's storage, and
        // each source in an encoder output the buffer pool holds until the batch has ended; the
        // replay that reads the buffer follows on the same stream.
        unsafe {
            if self.marked {
                platform::memset_d8(marks, 0, tokens, stream)?;
            }
            self.marked = !batch.columns.vision.is_empty();

            issue(copies, Some(stream))?;
            for block in &batch.columns.vision {
                platform::memset_d8(
                    marks + u64::from(block.token),
                    1,
                    block.rows as usize,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// The embeddings of a prefill of `tokens` tokens, `embedded`, with the marked rows' taken
    /// from the buffer.
    pub(super) fn merge(&self, embedded: &Tensor, tokens: usize) -> Result<Tensor> {
        let marks = self
            .marks
            .narrow(0, 0, tokens)?
            .broadcast_as(embedded.shape())?;

        Ok(marks.where_cond(&self.features.narrow(0, 0, tokens)?, embedded)?)
    }

    /// The `[axes, tokens]` I32 rotary coordinates of a prefill of `tokens` tokens at
    /// `positions`, `[tokens]` I32: each token's position on every axis, moved by the buffer's
    /// coordinates on a marked row.
    pub(super) fn rope(&self, positions: &Tensor, tokens: usize) -> Result<Tensor> {
        let shape = (self.extents.2, tokens);
        let marks = self.marks.narrow(0, 0, tokens)?.t()?.broadcast_as(shape)?;
        let text = positions.reshape((1, tokens))?.broadcast_as(shape)?;

        let image = self.coordinates.narrow(0, 0, tokens)?.t()?.add(&text)?;
        Ok(marks.where_cond(&image, &text)?)
    }

    /// Issues, through the rank's media memory on `stream`, those of `copies` that move encoder
    /// outputs between the product buffer and the host encoder pool, and returns the others.
    ///
    /// # Errors
    ///
    /// As `BufferMemory::copy`.
    pub(super) fn copy(
        &mut self,
        copies: &[BlockCopy],
        stream: Option<&Stream>,
    ) -> Result<Vec<BlockCopy>> {
        let (media, others): (Vec<BlockCopy>, Vec<BlockCopy>) =
            copies.iter().cloned().partition(|copy| {
                !copy.runs.is_empty()
                    && copy
                        .runs
                        .iter()
                        .all(|(from, to)| self.memory.holds(from.0) && self.memory.holds(to.0))
            });

        if !media.is_empty() {
            self.copied = true;
            self.memory.copy(&media, stream)?;
        }
        Ok(others)
    }

    /// Waits on `stream` for the copies [`Features::copy`] issued, when it issued any since the
    /// last wait.
    ///
    /// # Errors
    ///
    /// The driver's refusal.
    pub(super) fn settle(&mut self, stream: Option<&Stream>) -> Result<()> {
        if std::mem::take(&mut self.copied)
            && let Some(stream) = stream
        {
            stream.synchronize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ipc::batch::{Batch, BatchId, Columns, Phase, Row, Rows, Shape, Variant, VisionRows};
    use storage::runtime::GroupId;
    use uniserve_tensor::DType;

    use super::Features;
    use crate::memory::buffers::BufferMemory;
    use crate::runners::address;
    use crate::runners::vision::tests::{
        ENCODER, HIDDEN, PRODUCTS, SLOTS, Seeing, encoded, tables,
    };

    /// The tokens the buffer holds, and the bytes of a feature row and of a row's coordinates.
    const TOKENS: usize = 16;
    const ROW: usize = 2 * HIDDEN;
    const COORDINATES: usize = 3 * 4;

    /// A prefill of two calls, slot 1's of tokens 0..5 and slot 3's of tokens 5..12, whose
    /// second holds an image block of four rows from token `token`, its encoder output at
    /// position 1.
    fn prefill(token: u32) -> Batch {
        let call = |slot: u32, row: u32| Row {
            slot,
            kind: Phase::Prefill,
            cache_group: 0,
            rows: row..row + 1,
        };

        Batch {
            id: BatchId(5),
            group: GroupId(0),
            phase: Phase::Prefill,
            generation: 0,
            shape: Shape::prefill(2, 12, Variant::PrefillWrite),
            requests: Vec::new(),
            columns: Columns {
                rows: Rows {
                    calls: vec![call(1, 0), call(3, 1)],
                    positions: (0..12).collect(),
                    query_offsets: vec![0, 5, 12],
                    ..Rows::default()
                },
                vision: vec![VisionRows {
                    token,
                    rows: 4,
                    start: 3,
                    feature: 1,
                }],
                ..Columns::default()
            },
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    #[test]
    fn an_image_row_reads_the_encoder_output_its_table_holds() {
        let memory = BufferMemory::meta(0, (PRODUCTS, 64), (SLOTS, 4), 0, &[]).unwrap();
        let features = Features::new(&Seeing, TOKENS, None, Arc::new(memory), ENCODER).unwrap();

        // Slot 3's encoder output at position 1 is the block at unit 10, slot 1's another.
        let mut tables = tables(&[1, 3]);
        encoded(&mut tables, 3, 1, (PRODUCTS, 10, 1));
        encoded(&mut tables, 1, 1, (PRODUCTS, 20, 1));

        // Over host mirrors of the product buffer and the feature buffer, the block's features
        // land at its rows of the features and its coordinates, after them in the block, at
        // its rows of the coordinates.
        let products: Vec<u8> = (0..64 * 256).map(|byte| (byte % 251) as u8).collect();
        let bytes = Features::bytes(&Seeing, TOKENS).unwrap() as usize;
        let mut buffer = vec![0_u8; bytes];
        for (from, to, count) in features.copies(&prefill(7), &tables).unwrap() {
            let (from, to, count) = (from as usize, to as usize, count as usize);
            buffer[to..to + count].copy_from_slice(&products[from..from + count]);
        }

        let block = 10 * 256;
        assert_eq!(buffer[7 * ROW..11 * ROW], products[block..block + 4 * ROW]);
        let coordinates = address(&features.coordinates).unwrap() as usize + 7 * COORDINATES;
        let stored = block + 4 * ROW;
        assert_eq!(
            buffer[coordinates..coordinates + 4 * COORDINATES],
            products[stored..stored + 4 * COORDINATES]
        );
        assert!(buffer[..7 * ROW].iter().all(|byte| *byte == 0));

        // A block that crosses from one call's row into another's, or of a request whose table
        // holds no encoder output there, is refused.
        assert!(features.copies(&prefill(3), &tables).is_err());
        let mut held = self::tables(&[1, 3]);
        encoded(&mut held, 1, 1, (PRODUCTS, 20, 1));
        assert!(features.copies(&prefill(7), &held).is_err());
    }

    #[test]
    fn a_prefill_reads_its_embeddings_and_rotary_coordinates_through_the_marks() {
        let memory = BufferMemory::meta(0, (PRODUCTS, 64), (SLOTS, 4), 0, &[]).unwrap();
        let features = Features::new(&Seeing, TOKENS, None, Arc::new(memory), ENCODER).unwrap();
        let meta = backend::meta();
        let tensor = |shape: (usize, usize), dtype| {
            // SAFETY: a meta tensor's address is never read.
            unsafe { backend::external(0, shape, dtype, &meta) }.unwrap()
        };

        let context = backend::Context::meta(None);
        let _guard = context.enter();
        let embedded = tensor((12, HIDDEN), DType::BF16);
        let positions = tensor((12, 1), DType::I32).flatten_all().unwrap();

        let merged = features.merge(&embedded, 12).unwrap();
        assert_eq!(
            (merged.dims(), merged.dtype()),
            (&[12, HIDDEN][..], DType::BF16)
        );
        let rope = features.rope(&positions, 12).unwrap();
        assert_eq!((rope.dims(), rope.dtype()), (&[3, 12][..], DType::I32));
    }
}
