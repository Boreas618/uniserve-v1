//! `VisionRunner`: the vision tower of a causal language model on its text rank, which encodes
//! a token request's image into the encoder output its prefills read.
//!
//! A chat image's encoding is one media row of a `VisionEncode` batch: the image's bytes and,
//! as its one result, the position of the request's encoder output in its encoder-output table.
//! The runner sizes the image by the model's processor rule ([`Vision::image_grid`]), decodes,
//! resizes and lays it out as the tower's patch rows on the host, directly into the input
//! block, and encodes it on the device. The output is copied into the block the rank's tables
//! hold at that position, of the encoder-output class:
//!
//! | Bytes from the block's first | Hold |
//! | --- | --- |
//! | `rows · hidden · element` | `[rows, hidden]` features in the model's activation type |
//! | `rows · axes · 4`, after them | `[rows, axes]` I32 rotary coordinates of the block from position 0 |
//!
//! [`VisionRunner::feature_bytes`] is the bytes of one row of it, which the rank states as
//! `Capacity::feature_bytes`. The coordinates travel with the features because a prefill knows
//! of an image block only its rows and the logical position it starts at, never the image's
//! grid, and an output found cached was encoded for another request. A prefill moves them by
//! the block's start on every axis ([`CausalLmRunner::set_features`]): the runner checks at
//! construction that the model's coordinates from position `p` are those from 0 plus `p`.
//!
//! ## The buckets
//!
//! A forward's bucket is its grid's patch rows. The runner encodes images of at most `rows`
//! merged rows, which its rank states when it builds it: the most the tower sizes
//! ([`Vision::max_image_rows`]), within the bound an admitted image's block has
//! (`ModelRunner::image_rows` of the rank's start: the grid tokens and feature bytes the engine
//! reports, which one prefill writes whole). The input block holds the patch rows of an image of
//! `rows` rows; an image of more is refused, alone.
//!
//! The processor sizes an image to any grid of whole merge blocks, and [`Vision`] enumerates
//! none of them, so the runner states its buckets by their extent: the grid of `rows` rows and
//! one grid per doubling of the merged rows below it, each the most nearly square grid of its
//! rows. Warm-up captures each of them ([`Runner::buckets`]); an image of one of those grids
//! replays its graph, and every other grid runs eagerly over the same input block and region,
//! under the bucket of its patch rows. A forward's launches and memory follow its patch rows, so
//! the profile run of the captured grids sizes the runner's region for every grid an admitted
//! image can take and states the kernels an eager forward launches; no graph or region is held
//! for a grid no admitted image reaches.
//!
//! [`CausalLmRunner::set_features`]: super::CausalLmRunner::set_features

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use crate::batch::{Batch, BatchOutput, MediaRow, Phase, TimingCounters, Variant};
use nn::capability::{PatchFormat, Vision};
use nn::family::TextModel;
use platform::Stream;
use tensor::{DType, Tensor};

use super::{DeviceCopy, address, conditions, element_bytes, issue, performed, refused, settle};
use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::Runner;
use crate::runner::buckets::Bucket;
use crate::runner::graphs::Graphs;
use crate::worker::{Error, Result};

/// The phase a vision runner serves.
const PHASES: &[Phase] = &[Phase::VisionEncode];

/// The bytes of one rotary coordinate of an encoder output: an I32.
const COORDINATE_BYTES: u64 = 4;

/// The encoder outputs of the largest image a rank encodes that its region of encoder outputs
/// holds at once ([`VisionRunner::region_units`]): as many images as one request reads whole, or
/// the images of several requests. An output of a smaller image takes fewer units, so more of
/// them fit.
const ENTRIES: u64 = 8;

/// The runner of a causal language model's vision tower (module documentation).
pub(crate) struct VisionRunner {
    model: Arc<dyn TextModel>,
    graphs: Graphs,
    /// The stream the graphs replay on; `None` for graphs that hold no device.
    stream: Option<Stream>,
    /// The rank's media memory, where the encoder outputs lie.
    memory: Arc<BufferMemory>,
    /// The class of encoder outputs, in whose table a row names its output's position.
    class: u8,
    /// The most merged rows of an image the runner encodes (module documentation).
    rows: u32,
    /// The captured buckets, largest first, each with the grid its graph computes over: the
    /// grid of `rows` rows first.
    grids: Vec<(Bucket, [u32; 3])>,
    /// The grid of the forward being prepared or run: the largest captured grid but while an
    /// image or another captured grid is prepared.
    grid: Cell<[u32; 3]>,
    /// The threads an image's conversion into patch rows shares its work over: the process's
    /// host-thread budget (`worker_config.host_threads`).
    threads: NonZeroUsize,
}

/// One image of a batch, read far enough to know its block: its encoded bytes, its displayed
/// `(width, height)`, its patch grid and the address of the encoder output it is written to.
type Sized = (Arc<[u8]>, (u32, u32), [u32; 3], u64);

impl VisionRunner {
    /// The bytes of the input tensors of a runner of `model` that encodes images of at most
    /// `rows` rows: the patch rows of its largest image.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model without a vision tower, or no rows.
    pub(crate) fn input_bytes(model: &dyn TextModel, rows: u32) -> Result<u64> {
        let vision = tower(model)?;
        patch_bytes(&vision.patch_format(), grid(vision, bound(vision, rows)?)?)
    }

    /// The bytes of one row of the encoder output a runner of `model` writes (module
    /// documentation): what the rank states as `Capacity::feature_bytes`, and zero for a model
    /// without a vision tower, which encodes nothing.
    pub(crate) fn feature_bytes(model: &dyn TextModel) -> u32 {
        let Some(vision) = model.vision() else {
            return 0;
        };
        let element = ops::abi::bytes(model.dtype()).unwrap_or(0);
        let bytes = u64::from(model.hidden()) * element
            + u64::from(vision.position_axes()) * COORDINATE_BYTES;

        u32::try_from(bytes).unwrap_or(u32::MAX)
    }

    /// The bytes of the encoder output of the largest image a runner of `model` that encodes
    /// images of at most `rows` rows writes: its rows of [`VisionRunner::feature_bytes`] each.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model without a vision tower, or no rows.
    pub(crate) fn output_bytes(model: &dyn TextModel, rows: u32) -> Result<u64> {
        let rows = bound(tower(model)?, rows)?;
        Ok(u64::from(rows) * u64::from(Self::feature_bytes(model)))
    }

    /// The units of [`UNIT_BYTES`] bytes the region of encoder outputs of a rank that encodes
    /// `model`'s images of at most `rows` rows holds on that rank: [`ENTRIES`] outputs of the
    /// largest image, each from a unit's first byte.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model without a vision tower, no rows, or a region of more units
    /// than a session numbers.
    pub(crate) fn region_units(model: &dyn TextModel, rows: u32) -> Result<u32> {
        let units = Self::output_bytes(model, rows)?.div_ceil(UNIT_BYTES) * ENTRIES;
        u32::try_from(units).map_err(|_| {
            Error::Invalid(format!(
                "a region of {units} units of encoder outputs is more than a session numbers"
            ))
        })
    }

    /// The buckets a runner of `model` that encodes images of at most `rows` rows captures,
    /// largest first ([`Runner::buckets`]): the shapes its group states for its image encodings.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model without a vision tower, or no rows.
    pub(crate) fn captured(model: &dyn TextModel, rows: u32) -> Result<Vec<Bucket>> {
        let grids = ladder(tower(model)?, rows)?;
        Ok(grids.into_iter().map(|(bucket, _)| bucket).collect())
    }

    /// A runner of `model`'s vision tower that encodes images of at most `rows` rows (the
    /// tower's largest when it sizes fewer) over `graphs`, made for
    /// [`VisionRunner::input_bytes`] bytes of input tensors, which replay on `stream` (`None`
    /// for graphs that hold no device). It writes encoder outputs of class `class` in `memory`,
    /// the rank's media memory, and shares an image's conversion into patch rows over `threads`
    /// threads.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model without a vision tower, no rows, a stream without a device
    /// or the reverse, graphs that hold fewer bytes than the input tensors, or a tower whose
    /// rotary coordinates do not move with the position a block starts at.
    pub(crate) fn new(
        model: Arc<dyn TextModel>,
        graphs: Graphs,
        stream: Option<Stream>,
        memory: Arc<BufferMemory>,
        (class, rows): (u8, u32),
        threads: NonZeroUsize,
    ) -> Result<Self> {
        let vision = tower(&*model)?;
        let rows = bound(vision, rows)?;
        let grids = ladder(vision, rows)?;
        let largest = grid(vision, rows)?;

        if graphs.enter().is_ok() != stream.is_some() {
            return Err(Error::Invalid(
                "a vision runner takes the stream of its graphs, and none for graphs that hold \
                 no device"
                    .into(),
            ));
        }
        let bytes = patch_bytes(&vision.patch_format(), largest)?;
        if graphs.input(0, bytes as usize, DType::U8).is_err() {
            return Err(Error::Invalid(format!(
                "the graphs hold fewer than the {bytes} bytes of the largest image's patch rows"
            )));
        }

        // A prefill moves a cached block's coordinates by its start (module documentation).
        let from = |start| -> Result<Vec<i32>> {
            Ok(vision
                .image_positions(largest, start)?
                .flatten_all()?
                .to_vec1()?)
        };
        let (zero, one) = (from(0)?, from(1)?);
        if zero.len() != one.len() || zero.iter().zip(&one).any(|(at, moved)| moved - at != 1) {
            return Err(Error::Invalid(
                "the vision tower's rotary coordinates of a block do not move with the position \
                 it starts at"
                    .into(),
            ));
        }

        Ok(Self {
            model,
            graphs,
            stream,
            memory,
            class,
            rows,
            grids,
            grid: Cell::new(largest),
            threads,
        })
    }

    /// The model's vision tower.
    fn vision(&self) -> Result<&dyn Vision> {
        tower(&*self.model)
    }

    /// The largest grid the runner captures: the grid of its `rows` rows.
    fn largest(&self) -> [u32; 3] {
        self.grids.first().map_or([0; 3], |(_, grid)| *grid)
    }

    /// The captured grid of `bucket`; `None` for a bucket the runner captures no graph of.
    fn captured_grid(&self, bucket: &Bucket) -> Option<[u32; 3]> {
        self.grids
            .iter()
            .find(|(captured, _)| captured == bucket)
            .map(|(_, grid)| *grid)
    }

    /// The grid a forward of `bucket` computes over: the grid of the image being encoded when
    /// the bucket is its, otherwise the grid the runner captures the bucket over.
    ///
    /// # Errors
    ///
    /// A bucket that is neither.
    fn grid_of(&self, bucket: &Bucket) -> Result<[u32; 3]> {
        let prepared = self.grid.get();
        if bucket_of(prepared)? == *bucket {
            return Ok(prepared);
        }

        self.captured_grid(bucket).ok_or_else(|| {
            Error::Invariant(format!(
                "bucket {bucket} is neither the one of the image grid {prepared:?} being \
                     encoded nor one the runner states"
            ))
        })
    }

    /// Sizes the image of `row` and finds the block it is written to in `tables`: `Ok(Err(..))`
    /// names an image the request sent that cannot be read or sized, or of more rows than the
    /// runner encodes, which fails its call alone.
    ///
    /// # Errors
    ///
    /// A row that is no image encoding of one result, or whose result the tables hold no
    /// writable block of the image's bytes for.
    fn size(
        &self,
        row: &MediaRow,
        tables: &BlockTables,
    ) -> Result<std::result::Result<Sized, String>> {
        let (Some(image), [position], 0) = (&row.image, row.results.as_slice(), row.reads) else {
            return Err(Error::Invalid(format!(
                "request {} is encoded by the vision runner with no image, or with results other \
                 than its encoder output",
                row.request.0
            )));
        };

        let read = || -> std::result::Result<_, String> {
            let bytes = Arc::clone(image);

            // The displayed size, orientation applied, as the reader decodes the pixels and
            // the frontend sized the image's block.
            let facts = media::probe::probe_image(&bytes)
                .map_err(|error| format!("reading an input image: {error}"))?;
            let grid = self
                .vision()
                .and_then(|vision| Ok(vision.image_grid(facts.height, facts.width)?))
                .map_err(|error| format!("sizing an input image: {error}"))?;

            Ok((bytes, (facts.width, facts.height), grid))
        };
        let (bytes, size, grid) = match read() {
            Ok(read) => read,
            Err(refusal) => return Ok(Err(refusal)),
        };

        let rows = self.vision()?.image_rows(grid)?;
        if rows > self.rows {
            return Ok(Err(format!(
                "an input image of {rows} rows, beyond the {} rows of an image this rank encodes",
                self.rows
            )));
        }
        let block = u64::from(rows) * u64::from(Self::feature_bytes(&*self.model));
        let output = self.memory.lend(
            tables,
            (row.slot, self.class, *position),
            true,
            0,
            &[block as usize],
            DType::U8,
        )?;
        Ok(Ok((bytes, size, grid, address(&output)?)))
    }

    /// Decodes a sized image and lays its patch rows out into the input block of its bucket,
    /// which it returns; `Ok(Err(..))` names an image that cannot be decoded, which fails its
    /// call alone.
    fn prepare(
        &self,
        (bytes, (width, height), grid, _): &Sized,
    ) -> Result<std::result::Result<Bucket, String>> {
        let mut rgb = vec![0_u8; *width as usize * *height as usize * 3];
        let transform = media::read::ImageTransform {
            resized: (*width, *height),
            window: (0, 0, *width, *height),
        };
        if let Err(error) = media::read::read_image(bytes, &transform, &mut rgb) {
            return Ok(Err(format!("decoding an input image: {error}")));
        }

        let format = self.vision()?.patch_format();
        let patches = patch_bytes(&format, *grid)? as usize;
        let bucket = bucket_of(*grid)?;

        self.grid.set(*grid);
        self.graphs.prepare_inputs(self, &bucket, |block| {
            let out = block.get_mut(..patches).ok_or_else(|| {
                Error::Invariant("an image's patch rows exceed the input block".into())
            })?;
            conditions::pack_block(
                &format,
                &rgb,
                1,
                (*height, *width),
                *grid,
                out,
                self.threads,
            )
        })?;
        Ok(Ok(bucket))
    }

    /// Encodes a prepared image: the graph captured over its grid is replayed, or its forward
    /// run eagerly, and its features and coordinates are copied into its block behind it.
    fn encode(&self, bucket: &Bucket, (_, _, grid, block): &Sized) -> Result<()> {
        let outputs = if self.captured_grid(bucket) == Some(*grid) {
            self.graphs.replay(bucket)?
        } else {
            self.graphs.eager(self, bucket)?
        };
        let [features, coordinates] = outputs.as_slice() else {
            return Err(Error::Invariant(
                "a vision forward returns its features and their coordinates".into(),
            ));
        };

        let features_bytes = (features.elem_count() as u64) * element_bytes(self.model.dtype())?;
        let coordinates_bytes = coordinates.elem_count() as u64 * COORDINATE_BYTES;
        let copies: [DeviceCopy; 2] = [
            (address(features)?, *block, features_bytes),
            (
                address(coordinates)?,
                block + features_bytes,
                coordinates_bytes,
            ),
        ];

        // SAFETY: the outputs lie in the runner's region until its next forward, which this
        // stream issues after the copies, and the block is the row's writable encoder output,
        // which the buffer pool holds until the batch has ended.
        unsafe { issue(&copies, self.stream.as_ref()) }
    }

    /// Serves `batch` over `tables`: every image's block is found first, then each image is
    /// read and encoded in turn, the host reading an image while the device encodes the one
    /// before it. It waits for the device but when the step fails.
    fn serve(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let started = Instant::now();
        if batch.requests.is_empty() {
            return Ok(BatchOutput::default());
        }
        batch.check()?;

        let sized = batch
            .media
            .iter()
            .map(|row| self.size(row, tables))
            .collect::<Result<Vec<_>>>()?;

        let mut calls = Vec::with_capacity(sized.len());
        let mut launched = None;
        for image in sized {
            let prepared = match image {
                Ok(image) => self.prepare(&image)?.map(|bucket| (bucket, image)),
                Err(refusal) => Err(refusal),
            };
            calls.push(match prepared {
                Ok((bucket, image)) => {
                    launched.get_or_insert_with(Instant::now);
                    self.encode(&bucket, &image)?;
                    performed()
                }
                Err(refusal) => refused(refusal),
            });
        }
        settle(self.stream.as_ref())?;

        // The device's time runs from the first launch to the end of the last copy.
        let launched = launched.unwrap_or_else(Instant::now);
        Ok(BatchOutput {
            calls,
            timing: TimingCounters {
                host_us: launched.duration_since(started).as_micros() as u64,
                device_us: launched.elapsed().as_micros() as u64,
                ..TimingCounters::default()
            },
            forward_stats: None,
        })
    }
}

impl Runner for VisionRunner {
    fn phases(&self) -> &[Phase] {
        PHASES
    }

    /// The buckets of the grids the runner captures (module documentation), largest first.
    fn buckets(&self) -> Vec<Bucket> {
        self.grids.iter().map(|(bucket, _)| *bucket).collect()
    }

    /// The tower over the patch rows at the head of the input block, for the grid of the image
    /// the runner prepared, or for the grid it states the bucket by: `[rows, hidden]` features,
    /// then their `[rows, axes]` I32 rotary coordinates from position 0.
    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let vision = self.vision()?;
        let grid = self.grid_of(bucket)?;

        let format = vision.patch_format();
        let pixels = self.graphs.input(
            0,
            (bucket.tokens as usize, format.width() as usize),
            format.dtype,
        )?;
        let rows = vision.image_rows(grid)? as usize;
        let device = self.graphs.device();

        // SAFETY: the tower writes every element of the features before anything reads them.
        let features = unsafe {
            Tensor::empty(
                (rows, self.model.hidden() as usize),
                self.model.dtype(),
                &device,
            )
        }?;
        vision.encode_image(grid, &pixels, &features)?;

        let coordinates = vision
            .image_positions(grid, 0)?
            .t()?
            .contiguous()?
            .to_device(&device)?;
        Ok(vec![features, coordinates])
    }

    /// The bucket's captured grid over patch rows of zeros, which encode to rows no request
    /// reads.
    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        let grid = self.captured_grid(bucket).ok_or_else(|| {
            Error::Invariant(format!("bucket {bucket} is none the runner captures"))
        })?;
        self.grid.set(grid);
        columns.fill(0);
        Ok(())
    }

    /// Prepares the encode of a batch of one image, whose block `tables` hold.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`]; a batch of other than one image, or one the request sent
    /// that cannot be read.
    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        let [row] = batch.media.as_slice() else {
            return Err(Error::Invalid(format!(
                "batch {} prepares {} images, and a vision runner prepares one",
                batch.id.0,
                batch.media.len()
            )));
        };
        let image = self.size(row, tables)?.map_err(Error::Invalid)?;

        self.prepare(&image)?.map_err(Error::Invalid)
    }

    /// Encodes every image of `batch` into the encoder output its row names (module
    /// documentation). An image the request sent that cannot be read or sized fails its call
    /// with the reason; the others are encoded.
    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let served = self.serve(batch, tables);
        self.grid.set(self.largest());

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

    /// The rank's media memory, where the encoder outputs lie: its product buffer and its parts
    /// of host regions (`BufferMemory::holds`).
    fn maps(&mut self, region: u8) -> bool {
        self.memory.holds(region)
    }

    /// Issues the copies of encoder outputs between the product buffer and the host encoder
    /// pool through the rank's media memory (`BufferMemory::copy`), on the runner's stream ahead
    /// of the batch, whose end covers them.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        self.memory.copy(copies, self.stream.as_ref())
    }

    /// The copies were issued on the runner's stream, which this waits for.
    fn wait_copies(&mut self) -> Result<()> {
        settle(self.stream.as_ref())
    }
}

/// `model`'s vision tower.
fn tower(model: &dyn TextModel) -> Result<&dyn Vision> {
    model.vision().ok_or_else(|| {
        Error::Invalid("a vision runner was given a model without a vision tower".into())
    })
}

/// The most rows of an image a runner of `vision` encodes when its rank admits images of at
/// most `rows` rows: the fewer of those and the tower's largest ([`Vision::max_image_rows`]).
///
/// # Errors
///
/// [`Error::Invalid`] for no rows: a rank that admits no image builds no vision runner.
fn bound(vision: &dyn Vision, rows: u32) -> Result<u32> {
    match rows.min(vision.max_image_rows()) {
        0 => Err(Error::Invalid(
            "a vision runner encodes images of some rows, and this one of none".into(),
        )),
        rows => Ok(rows),
    }
}

/// The buckets a runner of `vision` that encodes images of at most `rows` rows captures, largest
/// first, each with its grid: the grid of the bound's rows ([`bound`]), then one per doubling of
/// the merged rows below it, from one row.
fn ladder(vision: &dyn Vision, rows: u32) -> Result<Vec<(Bucket, [u32; 3])>> {
    let rows = bound(vision, rows)?;
    let below = std::iter::successors(Some(1_u32), |rows| rows.checked_mul(2))
        .take_while(|below| *below < rows);

    let mut grids = Vec::new();
    for rows in std::iter::once(rows).chain(below.collect::<Vec<_>>().into_iter().rev()) {
        let grid = grid(vision, rows)?;
        grids.push((bucket_of(grid)?, grid));
    }
    Ok(grids)
}

/// The bucket a forward of an image of `grid` runs as: its patch rows, one image of them.
fn bucket_of(grid: [u32; 3]) -> Result<Bucket> {
    let patches = grid
        .iter()
        .try_fold(1_u32, |product, extent| product.checked_mul(*extent))
        .ok_or_else(|| Error::Invalid(format!("an image grid {grid:?} overflows")))?;

    Ok(Bucket::prefill(1, patches, Variant::ConditionEncode))
}

/// The grid the runner states an image of `rows` merged rows by: the one the tower captures an
/// encode of that many rows over ([`Vision::bucket_grid`]), which must encode to `rows` rows.
fn grid(vision: &dyn Vision, rows: u32) -> Result<[u32; 3]> {
    let grid = vision.bucket_grid(rows)?;
    if rows == 0 || vision.image_rows(grid)? != rows {
        return Err(Error::Invalid(format!(
            "the vision tower sizes no image of {rows} rows to a grid of whole merge blocks"
        )));
    }
    Ok(grid)
}

/// The bytes of the patch rows of an image of `grid` in `format`.
fn patch_bytes(format: &PatchFormat, grid: [u32; 3]) -> Result<u64> {
    let patches: u64 = grid.iter().map(|extent| u64::from(*extent)).product();
    Ok(patches * u64::from(format.width()) * element_bytes(format.dtype)?)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use crate::batch::{MediaRow, Variant};
    use nn::capability::{CausalLm, PatchFormat, PixelNorm, Vision};
    use nn::family::{CacheShape, ExpertShape, StateShape, TextModel, TokenHistory};
    use nn::{Drafter, ExpertExchange, KvCache, RecurrentState, TokenBatch};
    use ops::Generation;
    use proto::request::RequestId as CallRequest;
    use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId, Run};
    use tensor::{DType, Device, Tensor};

    use super::VisionRunner;
    use crate::memory::buffers::BufferMemory;
    use crate::memory::tables::BlockTables;
    use crate::runner::Runner;
    use crate::runner::buckets::Bucket;
    use crate::runner::graphs::Graphs;

    /// The width of a hidden row, the rows of the largest image and the patch rows' side.
    pub(crate) const HIDDEN: usize = 8;
    const MAX_ROWS: u32 = 6;
    const SIDE: u32 = 2;
    const MERGE: u32 = 2;

    /// The slot region and the product buffer the tables name, and the encoder-output class.
    pub(crate) const SLOTS: u8 = 0;
    pub(crate) const PRODUCTS: u8 = 2;
    pub(crate) const ENCODER: u8 = 3;
    const GROUP: GroupId = GroupId(0);

    /// A causal language model with a vision tower: rows of [`HIDDEN`] BF16 over three rotary
    /// axes, and patch rows of `SIDE × SIDE` pixels merged `MERGE × MERGE`. Its forwards are
    /// never run; its tower writes nothing.
    pub(crate) struct Seeing;

    impl CausalLm for Seeing {
        fn embed_input_ids(&self, _: &Tensor) -> nn::Result<Tensor> {
            unreachable!("the tests run no forward of the language model")
        }

        fn forward(
            &self,
            _: &TokenBatch,
            _: &[KvCache],
            _: Option<&RecurrentState>,
            _: Option<&ExpertExchange>,
            _: &Tensor,
        ) -> nn::Result<Tensor> {
            unreachable!("the tests run no forward of the language model")
        }

        fn compute_logits(&self, _: &Tensor) -> nn::Result<Tensor> {
            unreachable!("the tests run no forward of the language model")
        }
    }

    impl TextModel for Seeing {
        fn hidden(&self) -> u32 {
            HIDDEN as u32
        }

        fn vocabulary(&self) -> u32 {
            16
        }

        fn dtype(&self) -> DType {
            DType::BF16
        }

        fn cache(&self) -> CacheShape {
            CacheShape {
                layers: 1,
                kv_heads: 1,
                head_dim: 8,
                index_keys: None,
            }
        }

        fn state_layers(&self) -> Vec<StateShape> {
            Vec::new()
        }

        fn experts(&self) -> Option<ExpertShape> {
            None
        }

        fn drafter(&self) -> Option<&dyn Drafter> {
            None
        }

        fn vision(&self) -> Option<&dyn Vision> {
            Some(self)
        }

        fn token_history(&self) -> Option<TokenHistory> {
            None
        }
    }

    impl Vision for Seeing {
        /// Whole merge blocks over the image, rounded up.
        fn image_grid(&self, height: u32, width: u32) -> nn::Result<[u32; 3]> {
            let block = SIDE * MERGE;
            Ok([
                1,
                height.div_ceil(block) * MERGE,
                width.div_ceil(block) * MERGE,
            ])
        }

        fn patch_format(&self) -> PatchFormat {
            PatchFormat {
                side: SIDE,
                temporal: 1,
                merge: MERGE,
                channels: 3,
                norm: PixelNorm {
                    mean: [0.5; 3],
                    std: [0.5; 3],
                },
                dtype: DType::BF16,
            }
        }

        fn image_rows(&self, grid: [u32; 3]) -> nn::Result<u32> {
            Ok(grid[0] * grid[1] * grid[2] / (MERGE * MERGE))
        }

        fn max_image_rows(&self) -> u32 {
            MAX_ROWS
        }

        /// One time step, its sides the most nearly equal whole merge blocks.
        fn bucket_grid(&self, rows: u32) -> nn::Result<[u32; 3]> {
            let short = (1..=rows.isqrt())
                .rev()
                .find(|side| rows.is_multiple_of(*side))
                .ok_or_else(|| nn::Error::Unsupported(format!("no grid of {rows} rows")))?;
            Ok([1, short * MERGE, rows / short * MERGE])
        }

        fn position_axes(&self) -> u32 {
            3
        }

        /// Row `k` of a block `columns` merged rows wide at `start + (0, k / columns, k %
        /// columns)`.
        fn image_positions(&self, grid: [u32; 3], start: u32) -> nn::Result<Tensor> {
            let (rows, columns) = (self.image_rows(grid)?, grid[2] / MERGE);
            let mut coordinates = Vec::new();
            for axis in 0..3 {
                coordinates.extend((0..rows).map(|row| {
                    let along = [0, row / columns, row % columns][axis];
                    (start + along) as i32
                }));
            }
            Ok(Tensor::from_vec(
                coordinates,
                (3, rows as usize),
                &Device::Cpu,
            )?)
        }

        fn encode_image(&self, _: [u32; 3], _: &Tensor, _: &Tensor) -> nn::Result<()> {
            Ok(())
        }
    }

    /// The tables of one rank with an owner attached at each of `slots`.
    pub(crate) fn tables(slots: &[u32]) -> BlockTables {
        let mut tables =
            BlockTables::new(GROUP, &[SLOTS, PRODUCTS], &[(SLOTS, 256), (PRODUCTS, 256)]);
        for &slot in slots {
            let attach = Request::Attach {
                owner: OwnerId(u64::from(slot)),
                group: GROUP,
            };
            tables
                .map(slot, RequestId(0), &attach, &Completion::Attached { slot })
                .unwrap();
        }
        tables
    }

    /// Gives `slot`'s owner the encoder output at `position`, the run `run`.
    pub(crate) fn encoded(tables: &mut BlockTables, slot: u32, position: u32, run: Run) {
        let allocate = Request::Allocate {
            owner: OwnerId(u64::from(slot)),
            class: ENCODER,
            at: position,
            count: 1,
            units: run.2,
        };
        let allocated = Completion::Allocated {
            runs: vec![run],
            evicted: vec![],
        };
        tables
            .map(slot, RequestId(1), &allocate, &allocated)
            .unwrap();
    }

    /// A `width × height` 24-bit BMP of distinct pixels.
    fn bmp(width: u32, height: u32) -> Vec<u8> {
        let stride = (width * 3).next_multiple_of(4);
        let pixels = stride * height;

        let mut file = b"BM".to_vec();
        for value in [54 + pixels, 0, 54, 40] {
            file.extend(value.to_le_bytes());
        }
        file.extend(width.to_le_bytes());
        file.extend(height.to_le_bytes());
        file.extend(1_u16.to_le_bytes());
        file.extend(24_u16.to_le_bytes());
        for value in [0, pixels, 2835, 2835, 0, 0] {
            file.extend(value.to_le_bytes());
        }
        file.extend((0..pixels).map(|byte| byte as u8));
        file
    }

    /// The row that encodes `image` for the request on slot 1 into its encoder output at
    /// position 2.
    fn row(image: &[u8]) -> MediaRow {
        MediaRow {
            request: CallRequest(1),
            slot: 1,
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
            image: Some(Arc::from(image)),
            results: vec![2],
            reads: 0,
        }
    }

    /// A runner of [`Seeing`] that encodes images of at most `rows` rows, over meta graphs and
    /// a product buffer of 64 units.
    fn runner(rows: u32) -> VisionRunner {
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            VisionRunner::input_bytes(&Seeing, rows).unwrap(),
        );
        let memory = BufferMemory::meta(0, (PRODUCTS, 64), (SLOTS, 4), 0, &[]).unwrap();
        VisionRunner::new(
            Arc::new(Seeing),
            graphs,
            None,
            Arc::new(memory),
            (ENCODER, rows),
            NonZeroUsize::MIN,
        )
        .unwrap()
    }

    #[test]
    fn an_image_is_encoded_into_the_block_its_row_names_and_one_unread_fails_alone() {
        let runner = runner(MAX_ROWS);

        // The runner's buckets are profiled and captured.
        runner.graphs().profile_run(&runner).unwrap();
        for bucket in runner.buckets() {
            runner.graphs().capture(&runner, &bucket).unwrap();
        }
        let mut captured = runner.buckets();
        captured.sort();
        assert_eq!(runner.graphs().buckets(), captured);

        // A row is `HIDDEN` BF16 features and three I32 coordinates.
        assert_eq!(VisionRunner::feature_bytes(&Seeing), 16 + 12);

        // An 8 × 4 image is two merged rows of a 2 × 4 grid: 56 bytes of its block at unit 3.
        let mut tables = tables(&[1]);
        encoded(&mut tables, 1, 2, (PRODUCTS, 3, 1));
        let image = runner.size(&row(&bmp(8, 4)), &tables).unwrap().unwrap();
        assert_eq!((image.1, image.2, image.3), ((8, 4), [1, 2, 4], 3 * 256));
        let bucket = runner.prepare(&image).unwrap().unwrap();
        assert_eq!(bucket.tokens, 8);

        // Bytes that are no image, or an image of more rows than the runner encodes, fail their
        // call; a result the tables hold no block for fails the batch.
        assert!(runner.size(&row(b"no image"), &tables).unwrap().is_err());
        assert!(runner.size(&row(&bmp(200, 200)), &tables).unwrap().is_err());
        assert!(runner.size(&row(&bmp(8, 4)), &self::tables(&[1])).is_err());
    }

    #[test]
    fn the_runner_captures_a_grid_per_doubling_of_rows_within_the_rows_it_encodes() {
        let encode = |patches| Bucket::prefill(1, patches, Variant::ConditionEncode);

        // Images of the tower's six rows at most: the grid of six rows, then those of four, two
        // and one merged rows, each `MERGE × MERGE` patches a row: 2 × 3, 2 × 2, 1 × 2 and 1 × 1
        // merge blocks. Every one is captured and none runs as an eager bucket of its own.
        let tower = runner(MAX_ROWS);
        assert_eq!(
            tower.buckets(),
            [encode(24), encode(16), encode(8), encode(4)]
        );
        assert!(tower.eager_buckets().is_empty());
        tower.graphs().profile_run(&tower).unwrap();
        for bucket in tower.buckets() {
            assert!(tower.graphs().launch_table(&bucket).is_some(), "{bucket}");
        }
        assert_eq!(
            VisionRunner::captured(&Seeing, MAX_ROWS).unwrap(),
            tower.buckets()
        );

        // A rank that admits images of four rows at most captures no grid above them, holds
        // fewer bytes of patch rows, and refuses an image of six rows alone.
        let admitted = runner(4);
        assert_eq!(admitted.buckets(), [encode(16), encode(8), encode(4)]);
        assert!(
            VisionRunner::input_bytes(&Seeing, 4).unwrap()
                < VisionRunner::input_bytes(&Seeing, MAX_ROWS).unwrap()
        );
        let mut tables = tables(&[1]);
        encoded(&mut tables, 1, 2, (PRODUCTS, 3, 1));
        let six = admitted.size(&row(&bmp(12, 8)), &tables).unwrap();
        assert!(six.is_err());

        // A bound above the tower's largest image is the tower's; none is no runner.
        assert_eq!(runner(100).buckets(), tower.buckets());
        assert!(VisionRunner::input_bytes(&Seeing, 0).is_err());

        // The region of encoder outputs holds eight outputs of the largest admitted image, each
        // from a unit's first byte: four rows of 28 bytes take one 256-byte unit.
        assert_eq!(VisionRunner::region_units(&Seeing, 4).unwrap(), 8);
        assert_eq!(VisionRunner::output_bytes(&Seeing, 4).unwrap(), 4 * 28);

        // An 8 × 4 image, two merged rows, is prepared as the bucket of its patch rows.
        let image = tower.size(&row(&bmp(8, 4)), &tables).unwrap().unwrap();
        assert_eq!(tower.prepare(&image).unwrap().unwrap(), encode(8));
    }
}
