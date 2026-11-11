//! A causal language model's image input on its rank: the vision encodings of token requests and
//! the feature products their prefills read.
//!
//! A token request's image reaches the rank twice. Its vision encoding (`ipc::MediaCall::
//! VisionEncoding`) names the image's bytes and the feature product it writes
//! (`ipc::execution::ImageEncoding`): the runner reads the image's size, sizes it by the model's
//! processor rule ([`nn::capability::Vision::image_grid`]), and a host lane decodes, resizes and
//! lays its pixels out as patch rows ([`ImageRead`]) while the device thread serves other work. The
//! device thread then encodes them into the product's rows, which a slot of the rank's feature
//! storage holds ([`Features`]); an encoding the host encoder pool already holds is copied in from
//! its pages instead, and one the engine reserved pages for is copied out to them. A later prefill
//! names the product in its columns (`ipc::execution::VisionRows`), and the runner writes its rows
//! into the prefill's hidden rows before the stack runs. The engine frees a product once its last
//! reader completed (`ipc::execution::Batch::frees`).
//!
//! Every slot holds the largest image the processor sizes, which is what the engine's buffer pool
//! books each encoding at, so the engine never asks for more encodings at once than the slots hold.

use std::collections::HashMap;
use std::sync::Arc;

use nn::capability::{PatchFormat, Vision};
use ops::Operand;
use uniserve_tensor::DType;
use platform::DeviceId;

use crate::host::{HostAction, Lane, Outcome};
use crate::memory::encoders::EncoderHostPool;
use crate::{Error, Result};

/// The encodings the rank's feature storage holds at once: as many images as one request reads
/// whole, or the images of several requests.
pub const FEATURE_ENTRIES: u32 = 8;

/// The device bytes the feature slots of a model take: [`FEATURE_ENTRIES`] encodings of `rows`
/// rows of `row_bytes` each.
pub fn feature_bytes(rows: u32, row_bytes: u64) -> u64 {
    Features::bytes(rows, row_bytes)
}

/// One image's pixels laid out as the tower reads them, on a host lane.
pub struct ImageRead {
    /// The image's encoded bytes.
    bytes: Vec<u8>,
    /// Its displayed size, `(width, height)`.
    size: (u32, u32),
    /// The patch grid the processor sized it to.
    grid: [u32; 3],
    /// How the tower reads the patches.
    format: PatchFormat,
}

impl HostAction for ImageRead {
    /// The patch rows, `[patches, width]` in the format's element type.
    type Output = Vec<u8>;

    fn run(&self) -> Result<Self::Output> {
        let (width, height) = self.size;
        let mut rgb = vec![0_u8; width as usize * height as usize * 3];
        media::read::read_image(
            &self.bytes,
            &media::read::ImageTransform {
                resized: (width, height),
                window: (0, 0, width, height),
            },
            &mut rgb,
        )
        .map_err(|error| Error::Invalid(format!("decoding an input image: {error}")))?;
        let element = ops::abi::bytes(self.format.dtype).unwrap_or(0) as usize;
        let patches: usize = self.grid.iter().map(|extent| *extent as usize).product();
        let mut out = vec![0_u8; patches * self.format.width() as usize * element];
        super::media::pack_block(&self.format, &rgb, 1, (height, width), self.grid, &mut out)?;
        Ok(out)
    }

    /// Nothing to release: the bytes are the action's own.
    fn release(&self) -> Result<()> {
        Ok(())
    }

    /// The bytes were decoded from the call's own payload, which no device work produces.
    fn input_outcome(&self) -> Outcome {
        Outcome::Success(())
    }
}

/// One feature product the storage holds.
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// Its slot.
    slot: u32,
    /// The rows it holds.
    rows: u32,
    /// The image's patch grid, which places its rows' rotary coordinates.
    grid: [u32; 3],
}

/// The rank's feature storage: slots of the largest encoding each, carved from the resident pool.
pub(crate) struct Features {
    storage: platform::Buffer,
    /// The bytes of one feature row.
    row_bytes: u64,
    /// The rows one slot holds.
    slot_rows: u32,
    /// The products held, by their identity.
    entries: HashMap<ipc::BufferId, Entry>,
    /// The slots no product holds.
    free: Vec<u32>,
}

impl Features {
    /// The bytes [`FEATURE_ENTRIES`] slots of `slot_rows` rows of `row_bytes` take.
    pub(crate) fn bytes(slot_rows: u32, row_bytes: u64) -> u64 {
        u64::from(FEATURE_ENTRIES) * u64::from(slot_rows) * row_bytes
    }

    /// The storage over `storage`, of [`Self::bytes`] bytes.
    fn new(storage: platform::Buffer, slot_rows: u32, row_bytes: u64) -> Self {
        Self {
            storage,
            row_bytes,
            slot_rows,
            entries: HashMap::new(),
            free: (0..FEATURE_ENTRIES).rev().collect(),
        }
    }

    /// The `[rows, hidden]` view of slot `slot`.
    fn view(&self, slot: u32, rows: u32, dtype: DType, hidden: u32) -> Operand {
        let address =
            self.storage.address() + u64::from(slot) * u64::from(self.slot_rows) * self.row_bytes;
        Operand::new(
            ops::DevicePtr::new(address),
            dtype,
            ops::Shape::new(&[u64::from(rows), u64::from(hidden)]),
        )
    }

    /// Takes a slot for `product` of `rows` rows of an image of `grid`.
    fn insert(&mut self, product: ipc::BufferId, rows: u32, grid: [u32; 3]) -> Result<u32> {
        if rows > self.slot_rows {
            return Err(Error::Invalid(format!(
                "an image of {rows} feature rows exceeds the {} a slot holds",
                self.slot_rows
            )));
        }
        if self.entries.contains_key(&product) {
            return Err(Error::Invariant(format!(
                "feature product {product:?} was written twice"
            )));
        }
        let slot = self.free.pop().ok_or_else(|| {
            Error::Invariant(format!(
                "every one of the {FEATURE_ENTRIES} feature slots holds a product: the engine \
                 books no more encodings than the slots hold"
            ))
        })?;
        self.entries.insert(product, Entry { slot, rows, grid });
        Ok(slot)
    }

    /// Releases `product`'s slot; a product this rank never held is no fault, since the engine
    /// frees a product on every rank of the batch's group.
    fn free(&mut self, product: &ipc::BufferId) {
        if let Some(entry) = self.entries.remove(product) {
            self.free.push(entry.slot);
        }
    }
}

/// One vision encoding staged at submit, which the device thread performs.
struct Staged {
    request: ipc::RequestKey,
    output: ipc::BufferId,
    grid: [u32; 3],
    rows: u32,
    /// The host lane's read of the image's patches; `None` for an encoding the host encoder pool
    /// holds, which is copied in instead.
    read: Option<Arc<crate::host::HostTask<ImageRead>>>,
    /// Where the encoding lies in the host encoder pool, when the engine found it or reserved
    /// pages for it.
    host: Option<ipc::HostOutput>,
}

/// One image block of the staged prefill, resolved against the storage.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Block {
    /// The batch's token index of its first row.
    pub(crate) token: u32,
    /// The feature rows.
    pub(crate) features: Operand,
    /// The image's patch grid.
    pub(crate) grid: [u32; 3],
    /// The logical position of its first row.
    pub(crate) start: u32,
}

/// The image input of a causal language model's runner.
pub(crate) struct Images {
    device: DeviceId,
    /// How the tower reads an image's pixels.
    format: PatchFormat,
    /// The activation type and width of a feature row.
    dtype: DType,
    hidden: u32,
    /// The rows of the largest image the processor sizes: one slot's rows.
    slot_rows: u32,
    /// The slots, once the device's pools are attached.
    features: Option<Features>,
    /// The host lane that reads images.
    lane: Lane<ImageRead>,
    /// The host encoder pool, and whether this rank writes the encodings it computes to it.
    pool: Option<(Arc<EncoderHostPool>, bool)>,
    /// The encodings of the batch submitted last, by call.
    staged: Vec<Staged>,
    /// The generation the batch submitted last names.
    generation: u64,
    /// Pinned host buffers the device reads in place, kept until the event behind their reader.
    retained: Vec<(platform::Event, Vec<platform::PinnedBuffer>)>,
}

impl Images {
    /// The image input of a model whose tower reads `format` and writes rows of `hidden` in
    /// `dtype`, `slot_rows` for the largest image, its images read on `lane`.
    pub(crate) fn new(
        device: DeviceId,
        format: PatchFormat,
        dtype: DType,
        hidden: u32,
        slot_rows: u32,
        lane: Lane<ImageRead>,
    ) -> Self {
        Self {
            device,
            format,
            dtype,
            hidden,
            slot_rows,
            features: None,
            lane,
            pool: None,
            staged: Vec::new(),
            generation: 0,
            retained: Vec::new(),
        }
    }

    /// The generation the batch submitted last names.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The bytes of one feature row.
    pub(crate) fn row_bytes(&self) -> u64 {
        u64::from(self.hidden) * ops::abi::bytes(self.dtype).unwrap_or(0)
    }

    /// The rows of the largest image: one slot's rows.
    pub(crate) fn slot_rows(&self) -> u32 {
        self.slot_rows
    }

    /// The patch rows of the largest image, which the recorded walk encodes.
    pub(crate) fn largest_patches(&self) -> u32 {
        self.slot_rows * self.format.merge * self.format.merge
    }

    /// The tower's patch format.
    pub(crate) fn format(&self) -> PatchFormat {
        self.format
    }

    /// Copies encodings from and to `pool`'s pages, writing those this rank computes when
    /// `writes`.
    pub(crate) fn with_pool(&mut self, pool: Arc<EncoderHostPool>, writes: bool) {
        self.pool = Some((pool, writes));
    }

    /// Carves the slots from the device's resident pool.
    pub(crate) fn bind(&mut self, pools: &mut crate::arena::Pools) -> Result<()> {
        if self.features.is_some() {
            return Ok(());
        }
        let bytes = Features::bytes(self.slot_rows, self.row_bytes());
        let storage = pools.allocate(platform::PoolKind::Resident, bytes)?;
        self.features = Some(Features::new(storage, self.slot_rows, self.row_bytes()));
        Ok(())
    }

    fn features(&self) -> Result<&Features> {
        self.features.as_ref().ok_or(Error::State(
            "the feature slots are carved when the pools are attached",
        ))
    }

    /// Releases the products a batch frees.
    pub(crate) fn free(&mut self, products: &[ipc::BufferId]) {
        if let Some(features) = self.features.as_mut() {
            for product in products {
                features.free(product);
            }
        }
    }

    /// Stages the vision encodings of `batch`, one per call in call order, starting each image's
    /// read on the host lane.
    pub(crate) fn stage(
        &mut self,
        vision: &dyn Vision,
        batch: &ipc::execution::Batch,
    ) -> Result<()> {
        use base64::Engine as _;

        if batch.images.len() != batch.calls.len() {
            return Err(Error::Invalid(format!(
                "a vision encoding batch of {} calls names {} images",
                batch.calls.len(),
                batch.images.len()
            )));
        }
        self.staged.clear();
        self.generation = batch.generation;
        for encoding in &batch.images {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoding.image.as_bytes())
                .map_err(|error| Error::Invalid(format!("an input image's base64: {error}")))?;
            // The displayed size, orientation applied, as the reader decodes the pixels and the
            // frontend sized the image's block.
            let facts = media::probe::probe_image(&bytes)
                .map_err(|error| Error::Invalid(format!("reading an input image: {error}")))?;
            let grid = vision
                .image_grid(facts.height, facts.width)
                .map_err(Error::from)?;
            let rows = vision.image_rows(grid).map_err(Error::from)?;
            let host = batch
                .host_outputs
                .iter()
                .find(|(request, _)| *request == encoding.request)
                .map(|(_, output)| output.clone());
            let import = host.as_ref().is_some_and(|output| output.import) && self.pool.is_some();
            let read = if import {
                None
            } else {
                Some(self.lane.admit(ImageRead {
                    bytes,
                    size: (facts.width, facts.height),
                    grid,
                    format: self.format,
                })?)
            };
            self.staged.push(Staged {
                request: encoding.request,
                output: encoding.output,
                grid,
                rows,
                read,
                host,
            });
        }
        Ok(())
    }

    /// Performs the staged encoding of call `index` on `stream` with `kernels`: the read's patches
    /// encoded into the product's slot, or the stored encoding copied in. The tower's grid values
    /// are uploaded through a slot of `staging`, the runner's staging regions.
    pub(crate) fn encode(
        &mut self,
        vision: &dyn Vision,
        index: usize,
        stream: &platform::Stream,
        (arena, staging): (&nn::Arena, &crate::staging::StagingRegion),
        kernels: &Arc<ops::Generation>,
        group: Option<&nn::Group>,
    ) -> Result<()> {
        // Pinned buffers whose reader completed are freed; the stream is never waited for here.
        self.retained
            .retain(|(read, _)| !matches!(read.query(), Ok(true)));
        let staged = self.staged.get(index).ok_or_else(|| {
            Error::Invariant(format!("no vision encoding was staged for call {index}"))
        })?;
        let (dtype, hidden) = (self.dtype, self.hidden);
        let features = self.features.as_mut().ok_or(Error::State(
            "the feature slots are carved when the pools are attached",
        ))?;
        let slot = features.insert(staged.output, staged.rows, staged.grid)?;
        let out = features.view(slot, staged.rows, dtype, hidden);
        let Some(read) = staged.read.as_ref() else {
            // The host encoder pool holds the encoding.
            return self.copy_host(index, out, true, stream);
        };
        let patches = match read.wait() {
            Outcome::Success(patches) => patches,
            Outcome::Failed(error) => {
                return Err(Error::Invalid(format!(
                    "request {}'s input image could not be read: {error}",
                    staged.request.request_id.0
                )));
            }
            Outcome::Cancelled => {
                return Err(Error::State(
                    "an input image's read was withdrawn before it ran",
                ));
            }
        };
        let grid = staged.grid;
        let mut held = Vec::new();
        let pixels = self.pinned(&patches, &mut held)?;
        let patch_rows: u64 = grid.iter().map(|extent| u64::from(*extent)).product();
        let pixels = Operand::new(
            ops::DevicePtr::new(pixels),
            self.format.dtype,
            ops::Shape::new(&[patch_rows, u64::from(self.format.width())]),
        );
        let slot = staging.slot()?;
        let forward = super::denoiser::grouped(
            nn::Forward::new(stream, kernels, arena).with_staging(slot),
            group,
        )?;
        vision
            .encode_image(&forward, grid, pixels, out)
            .map_err(Error::from)?;
        staging.release(stream)?;
        let read = platform::Event::new(platform::EventFlags::default());
        read.record(stream)?;
        self.retained.push((read, held));
        self.copy_host(index, out, false, stream)
    }

    /// Copies `bytes` into a pinned host buffer pushed onto `held`, answering its address.
    fn pinned(&self, bytes: &[u8], held: &mut Vec<platform::PinnedBuffer>) -> Result<u64> {
        let mut buffer = platform::PinnedBuffer::new(self.device, bytes.len().max(1))?;
        // SAFETY: the buffer was allocated above and no device work reads it yet.
        unsafe {
            buffer.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        }
        let address = buffer.address();
        held.push(buffer);
        Ok(address)
    }

    /// Copies between staged call `index`'s feature rows `features` and its stored encoding's
    /// pages: from the pages when `import`, to them otherwise when this rank writes encodings.
    /// The copies have completed when this returns, before the call's result lets the engine
    /// release or commit the pages.
    fn copy_host(
        &self,
        index: usize,
        features: Operand,
        import: bool,
        stream: &platform::Stream,
    ) -> Result<()> {
        let (Some((pool, writes)), Some(output)) = (
            self.pool.as_ref(),
            self.staged
                .get(index)
                .and_then(|staged| staged.host.as_ref()),
        ) else {
            return if import {
                Err(Error::Invariant(
                    "an encoding read from the host encoder pool names no pool".into(),
                ))
            } else {
                Ok(())
            };
        };
        if output.import != import || (!import && !*writes) {
            return Ok(());
        }
        if output.pool != pool.pool() || output.row_bytes != self.row_bytes() {
            return Err(Error::Invalid(format!(
                "a host output of pool {} with rows of {} bytes reached a rank of pool {} with \
                 rows of {}",
                output.pool,
                output.row_bytes,
                pool.pool(),
                self.row_bytes()
            )));
        }
        let rows = u64::from(output.rows[1] - output.rows[0]).min(features.rows());
        let mut copied = false;
        for copy in crate::memory::encoders::page_copies(
            output.rows[0],
            0..rows,
            output.row_bytes,
            pool.page_bytes(),
        ) {
            // An export leaves a page the pool holds already as it is.
            let Some(slot) = output.slots.get(copy.page).copied().flatten() else {
                continue;
            };
            let page = pool.view(slot)?.address() + copy.offset;
            let row = features.ptr.address() + copy.output;
            let (destination, source) = if import { (row, page) } else { (page, row) };
            // SAFETY: the pages stay pinned or reserved until this call's result, the copies
            // complete before this returns, and the feature slot is this product's alone.
            unsafe { platform::memcpy_async(destination, source, copy.bytes as usize, stream) }?;
            copied = true;
        }
        if copied {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// The image blocks of a prefill's `rows`, resolved to the products' slots.
    ///
    /// # Errors
    ///
    /// A block of a product this rank does not hold, or of other rows than its product's.
    pub(crate) fn blocks(&self, rows: &[ipc::execution::VisionRows]) -> Result<Vec<Block>> {
        let features = self.features()?;
        rows.iter()
            .map(|block| {
                let entry = features.entries.get(&block.feature).ok_or_else(|| {
                    Error::Invalid(format!(
                        "a prefill reads feature product {:?}, which this rank does not hold",
                        block.feature
                    ))
                })?;
                if entry.rows != block.rows {
                    return Err(Error::Invalid(format!(
                        "a prefill places {} rows of an image whose features are {} rows",
                        block.rows, entry.rows
                    )));
                }
                Ok(Block {
                    token: block.token,
                    features: features.view(entry.slot, entry.rows, self.dtype, self.hidden),
                    grid: entry.grid,
                    start: block.start,
                })
            })
            .collect()
    }
}

/// The batch's rotary coordinates, `[axes, tokens]` axis-major: every token's text position on
/// every axis, and each image block's rows at the coordinates the model states for its grid.
///
/// # Errors
///
/// A block past the batch's tokens, or what the model refuses.
pub(crate) fn rope_positions(
    vision: &dyn Vision,
    positions: &[i32],
    blocks: &[Block],
) -> Result<Vec<i32>> {
    let tokens = positions.len();
    let axes = vision.position_axes() as usize;
    let mut out: Vec<i32> = (0..axes).flat_map(|_| positions.iter().copied()).collect();
    for block in blocks {
        let first = block.token as usize;
        let rows = vision.image_rows(block.grid).map_err(Error::from)? as usize;
        if first + rows > tokens {
            return Err(Error::Invalid(format!(
                "an image block of rows {first}..{} lies past the batch's {tokens} tokens",
                first + rows
            )));
        }
        let mut slices: Vec<&mut [i32]> = out
            .chunks_mut(tokens.max(1))
            .map(|axis| &mut axis[first..first + rows])
            .collect();
        vision
            .image_positions(block.grid, block.start, &mut slices)
            .map_err(Error::from)?;
    }
    Ok(out)
}
