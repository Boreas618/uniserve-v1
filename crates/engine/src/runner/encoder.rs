//! `EncoderRunner`: the runner of a text encoder (`nn::capability::Encoder`), which encodes a
//! video request's prompt into its conditioning.
//!
//! A text encoding batch carries one row of token columns per request, its prompt, and one
//! media row per request naming its results as positions in the request's product table: the
//! products the call reads, then the conditioning it writes. Once the encode has run, each
//! prompt's rows of its `[tokens, width]` output are copied into the conditioning block the
//! rank's tables hold at that position, of the conditioning's class. The block's one run lies in
//! the product buffer of the group's first text encoder rank ([`EncoderConfig::holder`]): every
//! rank of that rank's process writes the same rows there, and a rank of another process, which
//! holds no entry of the block, writes none.
//!
//! ## Captured and eager encodes
//!
//! A prompt of text alone is encoded by the captured graph of the smallest bucket that holds its
//! batch, one bucket per `(rows, prompt capacity)` the rank states. A prompt that presents vision
//! blocks (its conditioning's `vision` grids) is encoded alone and eagerly over its own grids,
//! which no graph is captured for: its patch rows are the vision pixels product its row reads
//! first, lent in place. The eager encode takes its memory from the runner's region, over the
//! inputs of the bucket that holds the prompt.
//!
//! The runner states its eager buckets ([`Runner::eager_buckets`]): the extent of each one-row
//! bucket that a prompt of [`EncoderConfig::warm`] fits, under [`Variant::ConditionEncode`]. A
//! profile run of one encodes every such prompt that fits it, its ids padded to the bucket's
//! tokens, over patch rows of its grids, so the runner's region and its input buffer hold the
//! eager vision encodes of every prompt the deployment admits and every generation resolves
//! their kernels.
//!
//! An encode reads no cache and samples no token: its batch's slot mapping writes nothing, and
//! its output is the one tensor its forward returns.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use crate::batch::{Batch, BatchOutput, Phase, TimingCounters, Variant};
use nn::capability::{Encoder, PatchFormat};
use nn::{NO_SLOT, TokenBatch};
use platform::Stream;
use tensor::{DType, Tensor};

use super::{DeviceCopy, address, element_bytes, issue, performed, settle};
use crate::memory::buffers::BufferMemory;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::Runner;
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::graphs::Graphs;
use crate::worker::{Error, Result};
use ops::allocator::ALIGNMENT;

/// The phase an encoder runner serves.
const PHASES: &[Phase] = &[Phase::TextEncode];

/// The input columns, in the order they lie in the input block, each of 4-byte elements:
/// `[tokens]` U32 ids, `[tokens]` I32 positions, `[tokens]` U32 slot mapping, `[rows]` U32
/// sequence lengths, `[rows + 1]` U32 query offsets, `[rows]` U32 logits rows.
const IDS: usize = 0;
const POSITIONS: usize = 1;
const SLOT_MAPPING: usize = 2;
const SEQ_LENS: usize = 3;
const QUERY_OFFSETS: usize = 4;
const LOGITS_ROWS: usize = 5;

/// What a deployment fixes of an encoder runner. Immutable once the runner is built.
#[derive(Clone, Debug)]
pub(crate) struct EncoderConfig {
    /// The buckets the runner captures, `Bucket::prefill(rows, tokens, Variant::Encode)` each:
    /// one per `(rows, prompt capacity)` the rank states.
    pub buckets: Buckets,
    /// The width of the conditioning's rows: an encode writes `[prompt tokens, width]`.
    pub width: u32,
    /// The conditioning's element type.
    pub dtype: DType,
    /// How the vision pixels product lays out a vision block's patch rows
    /// (`nn::capability::MediaFormat::patches`); `None` for an encoder that takes no image.
    pub patches: Option<PatchFormat>,
    /// The class of the conditioning product, which a row writes.
    pub conditioning: u8,
    /// The rank index (the deployment's rank, as the rank's media memory numbers ranks) of the
    /// group's first text encoder rank, whose product buffer holds an encode's conditioning in
    /// one run. Every text encoder rank of its process computes the same rows and writes them
    /// there; a rank of another process writes none, and a crossing brings the block to its
    /// process once the encode's output was applied.
    pub holder: u32,
    /// The class of the vision pixels product, which a prompt with vision blocks reads.
    pub pixels: u8,
    /// The prompts with vision blocks its eager encodes are profiled over
    /// ([`crate::runner::captures::warm_prompts`]): those the deployment admits whose grids it
    /// fixes, and those that bound the presentations whose grids follow the client's media;
    /// each prompt's grids and the token ids that present them. Empty for an encoder that takes
    /// no image.
    pub warm: Vec<(Vec<[u32; 3]>, Vec<u32>)>,
}

/// The runner of a text encoder (module documentation).
pub(crate) struct EncoderRunner {
    model: Arc<dyn Encoder + Send + Sync>,
    config: EncoderConfig,
    graphs: Graphs,
    /// The stream the graphs replay on; `None` for graphs that hold no device.
    stream: Option<Stream>,
    /// The rank's media memory, where the products lie.
    memory: Arc<BufferMemory>,
    /// Each input column's byte offset in the input block.
    offsets: [u64; 6],
    /// The prompt ids of the step being prepared, padded to its bucket's tokens: what the
    /// encode computes its host values from (`Encoder::encode`).
    prompt: RefCell<Vec<u32>>,
    /// The vision blocks of the prompt being encoded eagerly: each block's grid and its patch
    /// rows in the vision pixels product.
    images: RefCell<Vec<([u32; 3], Tensor)>>,
}

/// The elements of each input column for `rows` rows and `tokens` tokens.
const fn extents(rows: usize, tokens: usize) -> [usize; 6] {
    [tokens, tokens, tokens, rows, rows + 1, rows]
}

/// Each input column's byte offset for buckets of at most `rows` rows and `tokens` tokens, and
/// the bytes of the input block.
fn offsets(rows: usize, tokens: usize) -> ([u64; 6], u64) {
    let mut offsets = [0; 6];
    let mut bytes = 0;
    for (offset, extent) in offsets.iter_mut().zip(extents(rows, tokens)) {
        *offset = bytes;
        bytes += (extent as u64 * 4).next_multiple_of(ALIGNMENT);
    }
    (offsets, bytes)
}

/// The largest rows and tokens of `buckets`, which the input columns are sized for.
fn bounds(buckets: &Buckets) -> (usize, usize) {
    buckets.iter().fold((0, 0), |(rows, tokens), bucket| {
        (
            rows.max(bucket.rows as usize),
            tokens.max(bucket.tokens as usize),
        )
    })
}

/// The patch grids of the vision blocks a video request's `conditioning` presents, in
/// presentation order: one per condition with a vision view.
fn vision_grids(conditioning: &[u8]) -> Result<Vec<[u32; 3]>> {
    if conditioning.is_empty() {
        return Ok(Vec::new());
    }
    let decoded = proto::video::Conditions::decode(conditioning).map_err(|error| {
        Error::Invalid(format!(
            "a media request's conditioning does not decode: {error}"
        ))
    })?;

    Ok(decoded
        .conditions
        .iter()
        .filter_map(|condition| condition.vision.as_ref())
        .map(|vision| [vision.grid.t, vision.grid.h, vision.grid.w])
        .collect())
}

impl EncoderRunner {
    /// The bytes of the runner's input tensors for `config`: the `inputs` its [`Graphs`] are
    /// made with.
    pub(crate) fn input_bytes(config: &EncoderConfig) -> u64 {
        let (rows, tokens) = bounds(&config.buckets);
        offsets(rows, tokens).1
    }

    /// A runner of `model` under `config` over `graphs`, made for
    /// [`EncoderRunner::input_bytes`] bytes of input tensors, which replay on `stream` (`None`
    /// for graphs that hold no device). It reads and writes products in `memory`, the rank's
    /// media memory.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket or a bucket that is no encode, a stream
    /// without a device or the reverse, or graphs that hold fewer bytes than the input tensors.
    pub(crate) fn new(
        model: Arc<dyn Encoder + Send + Sync>,
        config: EncoderConfig,
        graphs: Graphs,
        stream: Option<Stream>,
        memory: Arc<BufferMemory>,
    ) -> Result<Self> {
        let encodes = config
            .buckets
            .iter()
            .all(|bucket| bucket.variant == Variant::Encode && bucket.rows > 0);
        if config.buckets.is_empty() || !encodes {
            return Err(Error::Invalid(
                "an encoder runner captures encode buckets of at least one row, and at least one"
                    .into(),
            ));
        }
        if graphs.enter().is_ok() != stream.is_some() {
            return Err(Error::Invalid(
                "an encoder runner takes the stream of its graphs, and none for graphs that hold \
                 no device"
                    .into(),
            ));
        }

        let (rows, tokens) = bounds(&config.buckets);
        let (offsets, bytes) = offsets(rows, tokens);
        if graphs.input(0, bytes as usize, DType::U8).is_err() {
            return Err(Error::Invalid(format!(
                "the graphs hold fewer than the {bytes} bytes of the runner's input tensors"
            )));
        }

        Ok(Self {
            model,
            config,
            graphs,
            stream,
            memory,
            offsets,
            prompt: RefCell::new(Vec::new()),
            images: RefCell::new(Vec::new()),
        })
    }

    /// The input column `column`, its first `elements` elements of `dtype`.
    fn column(&self, column: usize, elements: usize, dtype: DType) -> Result<Tensor> {
        self.graphs.input(self.offsets[column], elements, dtype)
    }

    /// The prompts of the warm set whose ids `bucket`'s tokens hold.
    fn warm_fitting(&self, bucket: &Bucket) -> Vec<&(Vec<[u32; 3]>, Vec<u32>)> {
        self.config
            .warm
            .iter()
            .filter(|(_, ids)| ids.len() <= bucket.tokens as usize)
            .collect()
    }

    /// Patch rows of `grids`, one `[patches, width]` tensor per grid in the vision pixels
    /// product's format, from the context installed on this thread; their values are
    /// unspecified, as a profile run reads none.
    fn patch_rows(&self, grids: &[[u32; 3]]) -> Result<Vec<Tensor>> {
        let format = self.config.patches.ok_or_else(|| {
            Error::Invariant("an encoder that takes no image profiles no vision prompt".into())
        })?;

        grids
            .iter()
            .map(|grid| {
                let patches: usize = grid.iter().map(|extent| *extent as usize).product();
                let shape = (patches, format.width() as usize);
                Ok(ops::empty(shape, format.dtype)?)
            })
            .collect()
    }

    /// Encodes `prompt`, its ids `ids` on the device and its packed rows `batch`, with the
    /// vision blocks of `grids` whose patch rows are `pixels`, into a new output of its layout's
    /// rows.
    fn encode(
        &self,
        prompt: &[u32],
        (ids, batch): (&Tensor, &TokenBatch),
        grids: &[[u32; 3]],
        pixels: &[Tensor],
    ) -> Result<Tensor> {
        let layout = self.model.layout(batch, grids);
        let shape = (layout.rows as usize, self.config.width as usize);

        // SAFETY: the encode writes every row of its layout before anything reads them.
        let out = unsafe { Tensor::empty(shape, self.config.dtype, &self.graphs.device()) }?;
        self.model
            .encode(prompt, ids, batch, pixels, &layout, &out)?;
        Ok(out)
    }

    /// Writes `values`, padded with `padding` to `elements`, into column `column` of `block`.
    fn write(
        &self,
        block: &mut [u8],
        column: usize,
        values: impl Iterator<Item = u32>,
        (elements, padding): (usize, u32),
    ) -> Result<()> {
        let start = self.offsets[column] as usize;
        let bytes = block
            .get_mut(start..start + 4 * elements)
            .ok_or_else(|| Error::Invariant("an input column lies past the input block".into()))?;

        let values = values.chain(std::iter::repeat(padding));
        for (entry, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *entry = value.to_le_bytes();
        }
        Ok(())
    }

    /// Checks `batch` and returns the bucket that serves it and the copies of each prompt's rows
    /// of the encode's output into its conditioning block, each from a byte offset of the
    /// output. A prompt with vision blocks is encoded eagerly in the extent of the bucket that
    /// holds it, under a key of [`Variant::ConditionEncode`] no graph is captured for, so its
    /// inputs are prepared without a pass for a graph's copies; its blocks are staged for its
    /// forward. Nothing is issued: everything that can refuse the batch is here.
    ///
    /// # Errors
    ///
    /// A batch no captured bucket holds, a media row that does not name one conditioning it
    /// writes after the products it reads, a result the tables hold no block for, or a writable
    /// one where the row writes, a block smaller than the rows written to it, and vision blocks
    /// in a batch of more than one prompt or for an encoder that takes no image.
    fn place(&self, batch: &Batch, tables: &BlockTables) -> Result<(Bucket, Vec<DeviceCopy>)> {
        batch.check()?;
        let rows = &batch.columns.rows;
        if rows.rows() != batch.media.len() {
            return Err(Error::Invalid(format!(
                "batch {} encodes {} prompts for {} media rows",
                batch.id.0,
                rows.rows(),
                batch.media.len()
            )));
        }
        let bucket = self
            .graphs
            .select(|bucket| bucket.holds(&batch.shape))
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "batch {} of {} holds in no captured encode bucket",
                    batch.id.0, batch.shape
                ))
            })?;

        let (width, dtype) = (self.config.width as usize, self.config.dtype);
        let row_bytes = width as u64 * element_bytes(dtype)?;
        let mut copies = Vec::with_capacity(batch.media.len());
        let mut images = Vec::new();
        let writes = self.memory.reaches(self.config.holder);

        for (index, row) in batch.media.iter().enumerate() {
            let reads = row.reads as usize;
            let (first, end) = (rows.query_offsets[index], rows.query_offsets[index + 1]);
            let (Some(&written), true, Ok(first), Ok(end)) = (
                row.results.get(reads),
                row.results.len() == reads + 1,
                u32::try_from(first),
                u32::try_from(end),
            ) else {
                return Err(Error::Invalid(format!(
                    "request {} encodes rows {first}..{end} and names results {:?} reading {}: \
                     an encode writes one conditioning after what it reads",
                    row.request.0, row.results, row.reads
                )));
            };

            let tokens = end.saturating_sub(first) as usize;
            if writes {
                let block = self.memory.lend(
                    tables,
                    (row.slot, self.config.conditioning, written),
                    true,
                    0,
                    &[tokens, width],
                    dtype,
                )?;
                copies.push((
                    u64::from(first) * row_bytes,
                    address(&block)?,
                    tokens as u64 * row_bytes,
                ));
            }

            // A prompt that presents vision blocks reads their patch rows first.
            let grids = vision_grids(&row.conditioning)?;
            if grids.is_empty() {
                continue;
            }
            let (Some(format), 1, Some(&read)) = (
                self.config.patches,
                batch.media.len(),
                row.results[..reads].first(),
            ) else {
                return Err(Error::Invalid(format!(
                    "request {} presents vision blocks in a batch of {} prompts, to an encoder \
                     that {}, reading {} results: a prompt with vision blocks is encoded alone \
                     and reads its patch rows",
                    row.request.0,
                    batch.media.len(),
                    if self.config.patches.is_some() {
                        "takes images"
                    } else {
                        "takes none"
                    },
                    row.reads
                )));
            };

            let patches = |grid: &[u32; 3]| grid.iter().map(|extent| *extent as usize).product();
            let total: usize = grids.iter().map(patches).sum();
            let patch_rows = self.memory.lend(
                tables,
                (row.slot, self.config.pixels, read),
                false,
                0,
                &[total, format.width() as usize],
                format.dtype,
            )?;

            let mut start = 0;
            for grid in grids {
                let count = patches(&grid);
                images.push((grid, patch_rows.narrow(0, start, count)?));
                start += count;
            }
        }

        let bucket = match images.is_empty() {
            true => bucket,
            false => Bucket::prefill(bucket.rows, bucket.tokens, Variant::ConditionEncode),
        };
        *self.images.borrow_mut() = images;
        Ok((bucket, copies))
    }

    /// Writes `batch`'s columns, padded to `bucket`, into the input block, with the host values
    /// the bucket's forward computes.
    fn fill(&self, batch: &Batch, bucket: &Bucket) -> Result<()> {
        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
        let columns = &batch.columns;
        let last = columns.rows.query_offsets.last().copied().unwrap_or(0);

        // The ids are the vocabulary's own indices, which the columns carry as I32.
        let ids: Vec<u32> = columns.input_ids.iter().map(|id| *id as u32).collect();
        let mut prompt = ids.clone();
        prompt.resize(tokens, 0);
        *self.prompt.borrow_mut() = prompt;

        let as_u32 = |values: &[i32]| values.iter().map(|value| *value as u32).collect::<Vec<_>>();
        self.graphs.prepare_inputs(self, bucket, |block| {
            self.write(block, IDS, ids.into_iter(), (tokens, 0))?;
            self.write(
                block,
                POSITIONS,
                as_u32(&columns.rows.positions).into_iter(),
                (tokens, 0),
            )?;
            self.write(block, SLOT_MAPPING, std::iter::empty(), (tokens, NO_SLOT))?;
            self.write(
                block,
                SEQ_LENS,
                as_u32(&columns.rows.seq_lens).into_iter(),
                (rows, 0),
            )?;
            // A padding row is empty: its offsets repeat the last.
            self.write(
                block,
                QUERY_OFFSETS,
                as_u32(&columns.rows.query_offsets).into_iter(),
                (rows + 1, last as u32),
            )?;
            self.write(
                block,
                LOGITS_ROWS,
                as_u32(&columns.rows.logits_rows).into_iter(),
                (rows, 0),
            )
        })
    }

    /// Serves `batch` over `tables`: everything of [`Runner::execute`] but waiting for the
    /// device when the step fails.
    fn serve(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let started = Instant::now();

        // A batch that names no request carries copies alone.
        if batch.requests.is_empty() {
            return Ok(BatchOutput::default());
        }
        let (bucket, copies) = self.place(batch, tables)?;
        self.fill(batch, &bucket)?;

        let launched = Instant::now();
        let outputs = match bucket.variant {
            Variant::ConditionEncode => self.graphs.eager(&*self, &bucket)?,
            _ => self.graphs.replay(&bucket)?,
        };
        let output = outputs.first().ok_or_else(|| {
            Error::Invariant("an encode returns the conditioning it computed".into())
        })?;

        // Each prompt's rows, behind the encode on its stream.
        let base = address(output)?;
        let copies: Vec<DeviceCopy> = copies
            .into_iter()
            .map(|(offset, to, bytes)| (base + offset, to, bytes))
            .collect();
        // SAFETY: the output lies in the runner's region until its next forward, which this
        // stream issues after the copies, and each destination is its row's writable
        // conditioning block, which the buffer pool holds until the batch has ended.
        unsafe { issue(&copies, self.stream.as_ref()) }?;
        settle(self.stream.as_ref())?;

        Ok(BatchOutput {
            calls: vec![performed(); batch.requests.len()],
            timing: TimingCounters {
                host_us: launched.duration_since(started).as_micros() as u64,
                device_us: launched.elapsed().as_micros() as u64,
                ..TimingCounters::default()
            },
            forward_stats: None,
        })
    }
}

impl Runner for EncoderRunner {
    fn phases(&self) -> &[Phase] {
        PHASES
    }

    fn buckets(&self) -> Vec<Bucket> {
        self.config.buckets.capture_order().copied().collect()
    }

    /// The extent of each one-row bucket a prompt with vision blocks the deployment admits fits,
    /// as such a prompt is encoded alone in it (module documentation).
    fn eager_buckets(&self) -> Vec<Bucket> {
        if self.config.patches.is_none() {
            return Vec::new();
        }

        self.config
            .buckets
            .capture_order()
            .filter(|bucket| bucket.rows == 1 && !self.warm_fitting(bucket).is_empty())
            .map(|bucket| Bucket::prefill(1, bucket.tokens, Variant::ConditionEncode))
            .collect()
    }

    /// The encode of the prompt the runner prepared, padded to `bucket`, over the input
    /// columns, with the vision blocks of a prompt encoded eagerly: `[rows, width]`, one output
    /// row per row of the encode's layout.
    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
        let [
            _,
            positions,
            slot_mapping,
            seq_lens,
            query_offsets,
            logits_rows,
        ] = extents(rows, tokens);

        // A profile run computes over a prompt of its bucket's tokens like any other.
        let prompt = {
            let staged = self.prompt.borrow();
            match staged.len() == tokens {
                true => staged.clone(),
                false => vec![0; tokens],
            }
        };

        let batch = TokenBatch {
            positions: self.column(POSITIONS, positions, DType::I32)?,
            slot_mapping: self.column(SLOT_MAPPING, slot_mapping, DType::U32)?,
            seq_lens: self.column(SEQ_LENS, seq_lens, DType::U32)?,
            query_offsets: self.column(QUERY_OFFSETS, query_offsets, DType::U32)?,
            logits_rows: self.column(LOGITS_ROWS, logits_rows, DType::U32)?,
            phase: Phase::Prefill,
            // An encode reads no cache, and a row attends over its own tokens, which a bucket
            // holds no more than.
            page_size: 0,
            num_tokens: bucket.tokens,
            max_kv_len: bucket.tokens,
            draft_lens: None,
            rope_positions: None,
            live_tokens: None,
            token_ids: None,
            token_history: None,
        };

        let ids = self.column(IDS, tokens, DType::U32)?;
        let images = self.images.borrow();

        // A profile run of an eager bucket encodes every prompt of the warm set that fits it,
        // over patch rows of its grids.
        if bucket.variant == Variant::ConditionEncode && images.is_empty() {
            let mut out = Vec::new();
            for (grids, warm) in self.warm_fitting(bucket) {
                let mut prompt = warm.clone();
                prompt.resize(tokens, 0);
                let pixels = self.patch_rows(grids)?;
                out = vec![self.encode(&prompt, (&ids, &batch), grids, &pixels)?];
            }
            return Ok(out);
        }

        let grids: Vec<[u32; 3]> = images.iter().map(|(grid, _)| *grid).collect();
        let pixels: Vec<Tensor> = images.iter().map(|(_, pixels)| pixels.clone()).collect();
        Ok(vec![self.encode(
            &prompt,
            (&ids, &batch),
            &grids,
            &pixels,
        )?])
    }

    /// A prompt of zeros in one empty row: an encode that writes no request's conditioning.
    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        if !self.config.buckets.contains(bucket) {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is none the runner states"
            )));
        }
        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);

        *self.prompt.borrow_mut() = vec![0; tokens];
        self.images.borrow_mut().clear();
        for (column, elements) in extents(rows, tokens).into_iter().enumerate() {
            let padding = if column == SLOT_MAPPING { NO_SLOT } else { 0 };
            self.write(columns, column, std::iter::empty(), (elements, padding))?;
        }
        Ok(())
    }

    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        let (bucket, ..) = self.place(batch, tables)?;
        self.fill(batch, &bucket)?;
        Ok(bucket)
    }

    /// Encodes every prompt of `batch` and writes its rows into the conditioning block its row
    /// names (module documentation). It returns once the device has completed the encode, the
    /// copies of its rows and the copies [`Runner::copy`] issued.
    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let served = self.serve(batch, tables);
        self.images.borrow_mut().clear();

        // A step that failed still ends with what it issued.
        let ended = match served {
            Ok(_) => Ok(()),
            Err(_) => settle(self.stream.as_ref()),
        };
        let output = served?;
        ended?;
        Ok(output)
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }

    /// Issues the copies of products that start with the runner's next batch through the rank's
    /// media memory (`BufferMemory::copy`), on the runner's stream ahead of the batch, whose end
    /// covers them. A copy of other memory is refused.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        self.memory.copy(copies, self.stream.as_ref())
    }

    /// The rank's media memory: its product buffer and its parts of host regions
    /// (`BufferMemory::holds`).
    fn maps(&mut self, region: u8) -> bool {
        self.memory.holds(region)
    }

    /// The copies were issued on the runner's stream, which this waits for.
    fn wait_copies(&mut self) -> Result<()> {
        settle(self.stream.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::batch::{Batch, BatchId, Columns, Extent, MediaRow, Phase, Rows, Shape, Variant};
    use nn::TokenBatch;
    use nn::capability::{Encoder, EncoderLayout, PatchFormat, PixelNorm};
    use ops::Generation;
    use proto::request::RequestId;
    use storage::Key;
    use storage::runtime::{Completion, GroupId, OwnerId, Request};
    use tensor::{DType, Tensor};

    use super::{EncoderConfig, EncoderRunner};
    use crate::memory::buffers::BufferMemory;
    use crate::memory::tables::BlockTables;
    use crate::runner::Runner;
    use crate::runner::buckets::{Bucket, Buckets};
    use crate::runner::graphs::Graphs;

    /// The slot region and the product buffer, and the classes of the conditioning and the
    /// vision pixels.
    const SLOTS: u8 = 0;
    const PRODUCTS: u8 = 2;
    const CONDITIONING: u8 = 4;
    const PIXELS: u8 = 5;
    const GROUP: GroupId = GroupId(0);

    /// The bytes of a conditioning row: four BF16 elements.
    const ROW: usize = 8;

    /// An encoder whose encode writes nothing: the tests read where its output goes.
    struct Silent;

    impl Encoder for Silent {
        fn layout(&self, batch: &TokenBatch, images: &[[u32; 3]]) -> EncoderLayout {
            EncoderLayout {
                rows: batch.num_tokens,
                tokens: batch.num_tokens,
                images: images.to_vec(),
            }
        }

        fn encode(
            &self,
            prompt: &[u32],
            ids: &Tensor,
            _: &TokenBatch,
            _: &[Tensor],
            layout: &EncoderLayout,
            out: &Tensor,
        ) -> nn::Result<()> {
            assert_eq!(prompt.len(), ids.elem_count());
            assert_eq!(out.dims(), [layout.rows as usize, 4]);
            Ok(())
        }
    }

    /// A runner of rank 0 over graphs that hold no device with an encode bucket of one row of
    /// eight tokens and one of two rows of sixteen, both captured, writing conditionings in the
    /// product buffer of rank `holder`.
    fn runner(holder: u32) -> EncoderRunner {
        let config = EncoderConfig {
            buckets: Buckets::new([
                Bucket::prefill(1, 8, Variant::Encode),
                Bucket::prefill(2, 16, Variant::Encode),
            ]),
            width: 4,
            dtype: DType::BF16,
            patches: None,
            conditioning: CONDITIONING,
            holder,
            pixels: PIXELS,
            warm: Vec::new(),
        };
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            EncoderRunner::input_bytes(&config),
        );
        let memory = BufferMemory::meta(0, (PRODUCTS, 64), (SLOTS, 4), 0, &[]).unwrap();
        let runner =
            EncoderRunner::new(Arc::new(Silent), config, graphs, None, Arc::new(memory)).unwrap();

        runner.graphs().profile_run(&runner).unwrap();
        for bucket in runner.buckets() {
            runner.graphs().capture(&runner, &bucket).unwrap();
        }
        runner
    }

    /// Tables in which the requests on slots 1 and 2 hold their conditioning at position 3:
    /// one unit from unit 0 for the first, from unit 5 for the second, the second's read-only
    /// when `pinned`.
    fn tables(pinned: bool) -> BlockTables {
        let mut tables =
            BlockTables::new(GROUP, &[SLOTS, PRODUCTS], &[(SLOTS, 256), (PRODUCTS, 256)]);
        for (slot, run) in [(1, (PRODUCTS, 0, 1)), (2, (PRODUCTS, 5, 1))] {
            let owner = OwnerId(u64::from(slot));
            let attach = Request::Attach {
                owner,
                group: GROUP,
            };
            tables
                .map(
                    slot,
                    storage::runtime::RequestId(0),
                    &attach,
                    &Completion::Attached { slot },
                )
                .unwrap();

            let held: (Request, Completion) = match pinned && slot == 2 {
                true => (
                    Request::Pin {
                        owner,
                        class: CONDITIONING,
                        at: 3,
                        keys: vec![Key([2; 32])],
                    },
                    Completion::Pinned {
                        blocks: vec![vec![run]],
                        host: 0,
                    },
                ),
                false => (
                    Request::Allocate {
                        owner,
                        class: CONDITIONING,
                        at: 3,
                        count: 1,
                        units: 1,
                    },
                    Completion::Allocated {
                        runs: vec![run],
                        evicted: vec![],
                    },
                ),
            };
            tables
                .map(slot, storage::runtime::RequestId(1), &held.0, &held.1)
                .unwrap();
        }
        tables
    }

    /// A text encoding of a prompt of five tokens on slot 1 and one of three on slot 2, each
    /// writing its conditioning at position `written`.
    fn encoding(written: u32) -> Batch {
        let row = |request: u64, slot: u32| MediaRow {
            request: RequestId(request),
            slot,
            conditioning: Vec::new(),
            media: Vec::new(),
            step: 0,
            steps: 0,
            seed: 0,
            width: 0,
            height: 0,
            frames: 0,
            rounds: 0,
            unit: 0,
            rank: None,
            image: None,
            results: vec![written],
            reads: 0,
        };

        Batch {
            id: BatchId(3),
            group: GROUP,
            phase: Phase::TextEncode,
            kernels: 0,
            policy: 0,
            revision: 0,
            shape: Shape {
                rows: 2,
                tokens: 8,
                variant: Variant::Encode,
                extent: Extent::None,
            },
            requests: vec![RequestId(10), RequestId(11)],
            columns: Columns {
                rows: Rows {
                    positions: vec![0, 1, 2, 3, 4, 0, 1, 2],
                    seq_lens: vec![5, 3],
                    query_offsets: vec![0, 5, 8],
                    logits_rows: vec![4, 7],
                    ..Rows::default()
                },
                input_ids: vec![7; 8],
                ..Columns::default()
            },
            media: vec![row(10, 1), row(11, 2)],
            new_slots: Vec::new(),
        }
    }

    #[test]
    fn a_batch_s_prompts_are_encoded_into_the_conditioning_blocks_its_rows_name() {
        let mut runner = runner(0);
        let (tables, batch) = (tables(false), encoding(3));

        // The two prompts ride the two-row bucket, captured.
        let (bucket, copies) = runner.place(&batch, &tables).unwrap();
        assert_eq!(bucket, Bucket::prefill(2, 16, Variant::Encode));
        assert_eq!(runner.prepare_inputs(&batch, &tables).unwrap(), bucket);

        // Over host mirrors of the encode's output and the product buffer, each prompt's rows
        // land in its own block.
        let output: Vec<u8> = (0..16 * ROW).map(|byte| byte as u8).collect();
        let mut products = vec![0_u8; 64 * 256];
        for (from, to, bytes) in copies {
            let (from, to, bytes) = (from as usize, to as usize, bytes as usize);
            products[to..to + bytes].copy_from_slice(&output[from..from + bytes]);
        }
        assert_eq!(products[..5 * ROW], output[..5 * ROW]);
        assert_eq!(
            products[5 * 256..5 * 256 + 3 * ROW],
            output[5 * ROW..8 * ROW]
        );
        assert!(products[5 * ROW..5 * 256].iter().all(|byte| *byte == 0));

        // A result the tables hold no block at, and a block the request may only read, refuse
        // the batch.
        assert!(runner.place(&encoding(4), &tables).is_err());
        assert!(runner.place(&batch, &self::tables(true)).is_err());
    }

    /// A text encoder rank of another process than the group's first, whose product buffer holds
    /// the conditioning, takes part in the encode and writes no conditioning: its tables hold no
    /// entry of the block, which a crossing brings to its process afterwards.
    #[test]
    fn a_rank_of_another_process_than_the_conditioning_s_encodes_and_writes_none() {
        let runner = runner(4);
        let mut tables =
            BlockTables::new(GROUP, &[SLOTS, PRODUCTS], &[(SLOTS, 256), (PRODUCTS, 256)]);
        for slot in [1, 2] {
            let attach = Request::Attach {
                owner: OwnerId(u64::from(slot)),
                group: GROUP,
            };
            let attached = Completion::Attached { slot };
            let id = storage::runtime::RequestId(0);
            tables.map(slot, id, &attach, &attached).unwrap();
        }

        let (bucket, copies) = runner.place(&encoding(3), &tables).unwrap();
        assert_eq!(bucket, Bucket::prefill(2, 16, Variant::Encode));
        assert!(copies.is_empty());
    }

    /// An encoder that records the grids and the prompt length of each encode it is given.
    #[derive(Default)]
    struct Recording(Mutex<Vec<(Vec<[u32; 3]>, usize)>>);

    impl Encoder for Recording {
        fn layout(&self, batch: &TokenBatch, images: &[[u32; 3]]) -> EncoderLayout {
            Silent.layout(batch, images)
        }

        fn encode(
            &self,
            prompt: &[u32],
            _: &Tensor,
            _: &TokenBatch,
            images: &[Tensor],
            layout: &EncoderLayout,
            _: &Tensor,
        ) -> nn::Result<()> {
            // Each grid's patch rows, twelve elements wide.
            for (grid, pixels) in layout.images.iter().zip(images) {
                let patches: u32 = grid.iter().product();
                assert_eq!(pixels.dims(), [patches as usize, 12]);
            }
            assert_eq!(images.len(), layout.images.len());

            self.0
                .lock()
                .unwrap()
                .push((layout.images.clone(), prompt.len()));
            Ok(())
        }
    }

    #[test]
    fn a_profile_run_encodes_each_admitted_vision_prompt_in_every_bucket_it_rides() {
        // Patch rows of three channels of 2x2 pixels, merged 2x2: twelve elements a row, and a
        // 4x4 grid merges to four tokens, which a prompt presents between two markers.
        let patches = PatchFormat {
            side: 2,
            temporal: 1,
            merge: 2,
            channels: 3,
            norm: PixelNorm {
                mean: [0.5; 3],
                std: [0.5; 3],
            },
            dtype: DType::BF16,
        };
        let block = [0, 9, 9, 9, 9, 0];
        let warm = vec![
            (vec![[1, 4, 4]], block.to_vec()),
            (vec![[1, 4, 4]; 2], block.repeat(2)),
        ];
        let config = EncoderConfig {
            buckets: Buckets::new([
                Bucket::prefill(1, 8, Variant::Encode),
                Bucket::prefill(2, 8, Variant::Encode),
                Bucket::prefill(1, 16, Variant::Encode),
                Bucket::prefill(2, 16, Variant::Encode),
            ]),
            width: 4,
            dtype: DType::BF16,
            patches: Some(patches),
            conditioning: CONDITIONING,
            holder: 0,
            pixels: PIXELS,
            warm,
        };
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            EncoderRunner::input_bytes(&config),
        );
        let memory = BufferMemory::meta(0, (PRODUCTS, 64), (SLOTS, 4), 0, &[]).unwrap();
        let model = Arc::new(Recording::default());
        let runner = EncoderRunner::new(
            Arc::clone(&model) as _,
            config,
            graphs,
            None,
            Arc::new(memory),
        )
        .unwrap();

        // A prompt with vision blocks is encoded alone: the one-row buckets its ids fit, the one
        // keyframe's of eight tokens and both prompts' of sixteen.
        let eager = runner.eager_buckets();
        assert_eq!(
            eager,
            [
                Bucket::prefill(1, 16, Variant::ConditionEncode),
                Bucket::prefill(1, 8, Variant::ConditionEncode),
            ]
        );

        // The profile run encodes each such prompt in each bucket it rides, padded to the
        // bucket's tokens, beside the captured buckets' text encodes.
        runner.graphs().profile_run(&runner).unwrap();
        let mut vision: Vec<(Vec<[u32; 3]>, usize)> = model
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|(grids, _)| !grids.is_empty())
            .cloned()
            .collect();
        vision.sort();
        assert_eq!(
            vision,
            [
                (vec![[1, 4, 4]], 8),
                (vec![[1, 4, 4]], 16),
                (vec![[1, 4, 4]; 2], 16),
            ]
        );
        for bucket in &eager {
            assert!(runner.graphs().launch_table(bucket).is_some());
        }

        // An encoder that takes no image states no eager bucket.
        assert!(self::runner(0).eager_buckets().is_empty());
    }
}
