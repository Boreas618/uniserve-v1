//! The decoder runner: one clip's latents, walked once per canvas and replayed (4.5), and the clip
//! spliced into the request's timeline by the component's second block when the deployment serves
//! one.
//!
//! A decoder is the encoder's mirror in every respect this runner cares about - one stream, a bucket
//! per declared extent, a captured graph per bucket, and the generation lifecycle the hot update
//! runs through - and it differs in two ways that belong here rather than in the model:
//!
//! **Its extent is a raster and its graphs are keyed by latents.** A decode bucket states the frames,
//! rows and columns of the media a clip decodes *to*; the tables, the arena and the captured graph
//! belong to the latents it decodes *from*. The ratios between the two are the checkpoint's, so the
//! model states the map ([`nn::capability::Decoder::layout_of`]) and this file never divides by a
//! temporal or spatial compression.
//!
//! **Its output is media, not hidden states.** The clip's raster is a column of this runner's own, so
//! a replay writes it where the transport reads it; an encode's output is read through resident
//! products instead, which is why that runner allocates its output from the arena and this one does
//! not.
//!
//! Its inputs are the model's own buffers. A media batch's columns are opaque bytes
//! (`batch::MediaColumns`): the model prepared them per request, in the order
//! [`nn::capability::Decoder::decode`] reads them, and this file copies them to device storage without
//! interpreting one of them. The order is stated once, by the model, in `prepare_layout`'s doc and in
//! the capability's: the first buffer is the clip's latents, and the rest are the tables
//! `layout_tables` names, in that order.
//!
//! **The tables are the model's and the clip is the denoiser's.** A layout's tables are prepared
//! here, from `prepare_layout`, because a batch that carried them would state the model's layout
//! arithmetic a second time. The trajectory is not this runner's to compute either: a denoise step
//! publishes it under the request's name, so `stage` copies it from the product store - ordered on
//! this runner's stream, which also works when the denoiser ran on another device - and reads a
//! batch buffer as the clip only when the store holds none.

use std::sync::{Arc, Mutex};

use nn::capability::{Clip, DecodedMedia, Decoder, Layout};
use ops::{DType, Shape, TensorView};
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{Batch, BatchOutput, CallId, Job, Jobs, answered};
use crate::plan::{Bucket, Buckets, Extent, Variant};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::Column;
use crate::runners::lifecycle::{Lifecycle, walk_capture, warmed_capture};
use crate::runners::products::{Product, Products};
use crate::runners::tables::Tables;
use crate::{Error, Result};

/// How long a clip waits for its predecessor's tail before it names the rank that never answered.
const TAIL_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// The channels of a decoded raster: red, green and blue, whatever the latents carry.
const RASTER_CHANNELS: u32 = 3;

/// The rows of the latent block `layout`'s decode reads: the decoder's own, or one per voxel.
fn block_rows<M: nn::capability::Decoder + ?Sized>(model: &M, layout: &Layout) -> Result<u64> {
    let video = layout.video;
    Ok(match model.latent_rows(layout).map_err(Error::from)? {
        Some(rows) => rows.len() as u64,
        None => u64::from(video.frames) * u64::from(video.height) * u64::from(video.width),
    })
}

/// The columns one clip's decode is staged into, sized for the largest canvas the runner serves.
///
/// A decoder's buckets differ by canvas, and its columns are sized by the largest of them: the clip
/// the bucket was captured for reads exactly its own prefix, the way an encoder's columns are read.
struct DecoderColumns {
    /// `[voxels, channels]` the clip's latents, in the type the decoder reads them in.
    latents: Column,
    /// The clip's raster, `[channels, frames, rows, columns]`, in the decoder's own element type.
    raster: Column,
}

impl DecoderColumns {
    /// The clip's own two columns. **The tables are not here:** `Tables` holds them by
    /// `(layout, name)` and `fill` puts the model's bytes in the cache the graph addresses
    /// (`walk` reads `self.tables.views`), so a host column per table would be a second copy that
    /// nothing reads — and the second copy is what let a staged buffer land in a table's place.
    fn new(
        device: DeviceId,
        layout: &Layout,
        latent: DType,
        dtype: DType,
        rows: u64,
    ) -> Result<Self> {
        let video = layout.video;
        let elements = rows
            .checked_mul(u64::from(video.channels))
            .ok_or_else(|| Error::Invalid("a clip's latent extent overflows".into()))?;
        let latents = Column::new(device, latent, elements)?;
        Ok(Self {
            latents,
            // Sized by `size_raster` when a bucket is first staged: the raster's frame count is the
            // bucket's rather than the layout's, since a layout states **latent** frames and the
            // clip writes the decoded ones. One element until then, because a column of a length
            // this constructor cannot know must not pretend to a length it can.
            raster: Column::new(device, dtype, 1)?,
        })
    }
}

/// The media computation a batch states, or the refusal a decoder runner gives a batch with none.
///
/// A decoder serves one computation and the mechanism states it in the batch rather than in the
/// runner: `Batch::media` is `Some` for a media batch and `None` for every token batch. The check is
/// its own function because it is the one part of `submit` that is a fact about the batch alone - a
/// device-free test can state a batch with no computation and read the message, where a test of
/// `submit` itself would need a constructed runner and therefore a device.
fn media_of(batch: &Batch) -> Result<&crate::batch::MediaColumns> {
    batch.media.as_ref().ok_or_else(|| {
        Error::Invalid(
            "a decoder runner serves a media batch; this one states no computation".into(),
        )
    })
}

/// The decoder's runner: one device stream, the canvas its model serves, and its graphs.
pub struct DecoderRunner<M: Decoder + Send + Sync> {
    id: RunnerId,
    phases: Vec<ipc::PhaseId>,
    model: Arc<M>,
    lifecycle: Lifecycle,
    buckets: Buckets,
    device: DeviceId,
    stream: platform::Stream,
    pools: Option<Arc<Mutex<Pools>>>,
    counters: Option<Arc<crate::device::SharedCounters>>,
    tables: Tables,
    /// The model's own map from each bucket's raster to the latents its graph is keyed by, asked once
    /// at bind: a bucket the model refuses is a plan that names a canvas its decoder does not serve,
    /// and the refusal belongs at load rather than at a capture.
    layouts: std::collections::BTreeMap<Bucket, Layout>,
    /// The element type the raster is stated in, as the model declares it.
    dtype: DType,
    /// The element type the model reads its latents in.
    latent: DType,
    columns: DecoderColumns,
    /// The bucket the columns are staged for, so a capture of a canvas knows how far they run.
    /// The calls of the batch staged last, which its replay answers.
    calls: Vec<CallId>,
    staged: Option<Bucket>,
    /// The raster's element count for the bucket staged last: a clip's raster is a function of its
    /// bucket, and the column is sized for the largest one.
    raster_elements: u64,
    /// The epoch's product store, or `None` when the executor attached none. A decode then writes
    /// its clip and publishes nothing, and the read that expects a frame refuses by name rather
    /// than reading whatever the column held before.
    products: Option<Arc<dyn Products>>,
    /// The request whose clip is staged. A decode bucket serves one clip (`clip_of` refuses two),
    /// so one name covers a batch, and a later decode of the same request replaces it — which is
    /// what a reader that copied in time expects.
    staged_request: Option<ipc::RequestKey>,
    /// The denoiser's trajectory, copied here from the store on this runner's stream, which is what
    /// orders a clip's read after the producing step and works across devices. Sized to the first one
    /// staged and grown after.
    trajectory: Option<Column>,
    /// The latent frame the clip staged last starts at, when its block is still to be gathered from
    /// `trajectory`: the replay gathers it, on the stream that then reads it.
    gather: Option<u32>,
    /// The elements of the trajectory staged last.
    trajectory_elements: u64,
    /// The request whose clip the raster column last held, so a decode of another can be refused
    /// while the first is unread ([`clip_is_releasable`]).
    published_for: Option<ipc::RequestKey>,
    /// The `video_decoder` component's second block, when the deployment serves one: the
    /// postprocessor that writes a clip's frames into the request's timeline with the seam fade and
    /// the checkpoint's de-normalization. Without one the runner publishes the clip's own raster.
    postprocessor: Option<Arc<dyn Decoder + Send + Sync>>,
    /// `[frames, rows, columns, 3]` bytes: the frames a splice wrote for the clip staged last.
    ///
    /// Sized at bind for the bucket's own clip raster, which no splice exceeds - a splice keeps at
    /// most the frames the decoder wrote, since the tail it adds is a subset of them.
    frames: Option<Column>,
    /// The tails this runner's clips leave, one `[channels, overlap, rows, columns]` slot per clip of
    /// the timeline in the model's type, which the next clip's five-frame fade reads. Sized on the
    /// first splice from the schedule's own overlap count, which is the model's number rather than
    /// this runner's, and reused after.
    tail: Option<Column>,
    /// The frames the staged request's timeline holds, from its own row
    /// (`StagedRequest::frames`): the schedule's length, which is what the last clip's keep depends
    /// on.
    staged_frames: Option<u32>,
    /// The frames the splice staged last wrote for the staged request's clip: the extent the product
    /// states, taken from the schedule rather than recomputed here.
    staged_keep: Option<u32>,
    /// The block-by-block diagnosis, off unless `UNISERVE_DIAG_DIR` names a directory.
    diag: crate::runners::diag::Diag,
    /// The timeline frame the clip staged last starts at, and the unit it is.
    staged_first: u32,
    staged_unit: u32,
    /// The units of `published_for`'s timeline this runner has published.
    published_units: u32,
}

/// Whether a decode of `request` may start while a clip published under `published` is still held.
///
/// **One raster column, and a clip nobody has read yet.** A decode writes this runner's raster
/// column, so decoding a second clip would overwrite the first before its artifact exists — silently,
/// after the step that produced it, with no event that could order the read (the transport reads
/// after the step, where a stream wait cannot reach). The first clip's life ends at
/// `Products::forget`, which the transport calls once it has the bytes, so this asks the store and
/// refuses by name: what a refusal means is *this request must wait for one read*, not *this
/// deployment cannot serve two*.
///
/// A re-decode of the **same** request is allowed: its clip is its own, and the transport has either
/// read it or not, in which case it is about to.
fn clip_is_releasable(
    request: ipc::RequestKey,
    published: Option<ipc::RequestKey>,
    still_held: bool,
) -> Result<()> {
    let Some(published) = published else {
        return Ok(());
    };
    if published == request || !still_held {
        return Ok(());
    }
    Err(Error::Invalid(format!(
        "request {} asks for a decode and this runner's raster column still holds request {}'s \
         clip: a decode writes the one column, so the second clip would replace the first before \
         its artifact was read. The clip's life ends at `forget`, which the reader calls once it \
         has the bytes; a second decode before that is a request that must wait, not a canvas this \
         device cannot serve",
        request.request_id.0, published.request_id.0
    )))
}

impl<M: Decoder + Send + Sync> DecoderRunner<M> {
    /// Binds a decoder to one device stream.
    ///
    /// Every decode bucket in `buckets` is mapped through the model's own `layout_of` here, so the
    /// bind fails by name when the plan holds a canvas the model's decoder does not serve. **That is
    /// the measured bound on a decoder runner's buckets:** a `VideoDecoder` is built for one canvas,
    /// so a plan offering it another is refused at load rather than at the canvas's first request.
    pub fn new(
        binding: Binding,
        model: Arc<M>,
        buckets: Buckets,
        generation: Arc<ops::Generation>,
        postprocessor: Option<Arc<dyn Decoder + Send + Sync>>,
    ) -> Result<Self> {
        let mut layouts = std::collections::BTreeMap::new();
        let mut largest: Option<(Bucket, Layout)> = None;
        for bucket in buckets.iter().copied() {
            if bucket.variant != Variant::Decode {
                continue;
            }
            let layout = Self::layout_of(&*model, bucket)?;
            if largest
                .as_ref()
                .is_none_or(|(held, _)| bucket.rows > held.rows)
            {
                largest = Some((bucket, layout));
            }
            layouts.insert(bucket, layout);
        }
        let (_, layout) = largest.as_ref().ok_or_else(|| {
            Error::Invalid(
                "a decoder runner was bound with no decode bucket to capture; its plan must state \
                 the canvases the checkpoint's decoder serves"
                    .into(),
            )
        })?;
        let dtype = model.element_type();
        let latent = model.latent_type();
        let rows = block_rows(model.as_ref(), layout)?;
        let columns = DecoderColumns::new(binding.device, layout, latent, dtype, rows)?;
        let lifecycle = Lifecycle::new(binding.id, generation, binding.stream.clone());
        Ok(Self {
            id: binding.id,
            phases: binding.phases,
            model,
            lifecycle,
            buckets,
            device: binding.device,
            stream: binding.stream,
            pools: None,
            counters: None,
            tables: Tables::new(),
            layouts,
            dtype,
            latent,
            columns,
            calls: Vec::new(),
            staged: None,
            raster_elements: 0,
            products: None,
            staged_request: None,
            trajectory: None,
            gather: None,
            trajectory_elements: 0,
            published_for: None,
            postprocessor,
            frames: None,
            tail: None,
            staged_frames: None,
            staged_keep: None,
            diag: crate::runners::diag::Diag::from_env(),
            staged_first: 0,
            staged_unit: 0,
            published_units: 0,
        })
    }

    /// One decode bucket's latent layout, from the model's map.
    fn layout_of(model: &M, bucket: Bucket) -> Result<Layout> {
        let Extent::Clip {
            frames,
            height,
            width,
        } = bucket.extent
        else {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is not a decode bucket and a decoder runner was asked for its \
                 layout"
            )));
        };
        model
            .layout_of(Clip {
                frames,
                height,
                width,
            })
            .map_err(|error| {
                Error::Invalid(format!(
                    "the decoder serves no clip of {frames} frame(s) at {height}x{width}: {error}"
                ))
            })
    }

    /// The buckets this runner captures: the decode ones, largest first.
    fn capture_order(&self) -> Vec<Bucket> {
        let mut buckets: Vec<Bucket> = self
            .buckets
            .iter()
            .filter(|bucket| bucket.variant == Variant::Decode)
            .copied()
            .collect();
        buckets.sort_by_key(|bucket| std::cmp::Reverse(bucket.rows));
        buckets
    }

    /// The layout of `bucket`, which `new` asked the model for.
    fn layout(&self, bucket: Bucket) -> Result<Layout> {
        self.layouts.get(&bucket).copied().ok_or_else(|| {
            Error::Invalid(format!(
                "no latent layout was mapped for the decode bucket of {bucket}"
            ))
        })
    }

    /// The key a layout's tables are held under.
    ///
    /// The bucket's extent states the raster and its layout states the latents, and the tables are
    /// keyed by the second: two canvases of one decoder are two table sets, and a canvas never reads
    /// another's.
    fn key(bucket: Bucket) -> String {
        format!("decode-{:?}-{}", bucket.extent, bucket.rows)
    }

    /// Allocates and fills the tables of every bucket's layout, once, when the runner loads (4.3).
    pub fn load(&mut self) -> Result<()> {
        if self.pools.is_none() {
            return Err(Error::Resource(
                "the decoder runner loaded before its device's pools were attached",
            ));
        }
        // **The plan is the floor of the device's share, checked here rather than discovered by a
        // capture that has to be thrown away.** The pool is the runtime's policy - a share of the
        // device per stream - and the model states what one decode of a layout reserves, so a share
        // below the plan is refused while the deployment is being assembled, naming the bucket and
        // both numbers, rather than at the first frame that reaches the arena.
        self.claim()?;
        for bucket in self.capture_order() {
            self.fill(bucket)?;
        }
        Ok(())
    }

    /// Claims the arena the largest bucket's decode plans.
    fn claim(&mut self) -> Result<()> {
        let mut needed = 0;
        for bucket in self.capture_order() {
            let layout = Self::layout_of(&self.model, bucket)?;
            // One rule for both media runners, so the two cannot read an absent plan and a refused
            // one differently: `floor::checked` passes over a decoder that states no plan and refuses
            // any other answer by name.
            needed = needed.max(
                super::floor::checked(bucket, self.model.arena_bytes(&layout), u64::MAX)?
                    .unwrap_or(0),
            );
        }
        if needed > 0 {
            let stream = self.stream_id();
            self.pool()?.claim(stream, needed)?;
        }
        Ok(())
    }

    /// The bytes `bucket`'s tables need, as the plan books them before admission (4.3).
    pub fn table_bytes(model: &M, bucket: Bucket) -> Result<u64> {
        let layout = Self::layout_of(model, bucket)?;
        let tables = model.layout_tables(&layout).map_err(Error::from)?;
        Tables::plan_bytes(&tables)
    }

    fn stream_id(&self) -> StreamId {
        StreamId(self.stream.raw())
    }

    fn pool(&self) -> Result<std::sync::MutexGuard<'_, Pools>> {
        let pools = self.pools.as_ref().ok_or_else(|| {
            Error::Resource(
                "the decoder runner has no graph pools; the device attaches them at start",
            )
        })?;
        Ok(pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// Allocates and fills `bucket`'s layout tables, on the device thread, outside any step.
    fn fill(&mut self, bucket: Bucket) -> Result<()> {
        let layout = self.layout(bucket)?;
        let tables = self.model.layout_tables(&layout).map_err(Error::from)?;
        if tables.is_empty() {
            return Ok(());
        }
        let key = Self::key(bucket);
        let model = Arc::clone(&self.model);
        self.tables.fill(
            self.device,
            &self.stream,
            self.lifecycle.generation.number(),
            &key,
            &tables,
            |host| model.prepare_layout(&layout, host).map_err(Error::from),
        )
    }

    /// Sizes this runner's raster column for `bucket`'s clip, the first time a bucket is staged.
    ///
    /// The raster is a column rather than an arena extent because a replay's output leaves the
    /// device: an arena is a graph's scratch and is reused by the next capture, while the clip a
    /// request asked for is read by the transport after the step.
    fn size_raster(&mut self, bucket: Bucket) -> Result<()> {
        let Extent::Clip {
            frames,
            height,
            width,
        } = bucket.extent
        else {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is not a decode bucket"
            )));
        };
        let layout = self.layout(bucket)?;
        let elements = u64::from(RASTER_CHANNELS)
            .checked_mul(u64::from(frames))
            .and_then(|elements| elements.checked_mul(u64::from(height)))
            .and_then(|elements| elements.checked_mul(u64::from(width)))
            .ok_or_else(|| Error::Invalid("a clip's raster extent overflows".into()))?;
        if elements > self.raster_elements {
            self.columns.raster = Column::new(self.device, self.dtype, elements)?;
            self.raster_elements = elements;
        }
        // The frames a splice writes are bytes - `[frames, rows, columns, 3]` - and there are as
        // many of them as the clip's own decoded frames at most: the body a splice keeps and the tail
        // it adds both come out of that raster, never beside it.
        if self.postprocessor.is_some() {
            let frames = u64::from(frames)
                .checked_mul(u64::from(height))
                .and_then(|elements| elements.checked_mul(u64::from(width)))
                .and_then(|elements| elements.checked_mul(3))
                .ok_or_else(|| Error::Invalid("a splice's raster extent overflows".into()))?;
            if self
                .frames
                .as_ref()
                .is_none_or(|column| column.device.numel() < frames)
            {
                self.frames = Some(Column::new(self.device, DType::U8, frames)?);
            }
        }
        Ok(())
    }

    /// Sizes the frames column for a timeline of `frames` frames.
    ///
    /// Each clip keeps its frames at their place in the timeline, so a decode that runs ahead of the
    /// encodes leaves every clip readable until its own encode copies it.
    fn size_timeline(&mut self, bucket: Bucket, frames: u32) -> Result<()> {
        let Extent::Clip { height, width, .. } = bucket.extent else {
            return Ok(());
        };
        if self.postprocessor.is_none() {
            return Ok(());
        }
        let elements = u64::from(frames)
            .checked_mul(u64::from(height))
            .and_then(|elements| elements.checked_mul(u64::from(width)))
            .and_then(|elements| elements.checked_mul(3))
            .ok_or_else(|| Error::Invalid("a timeline's raster extent overflows".into()))?;
        if self
            .frames
            .as_ref()
            .is_none_or(|column| column.device.numel() < elements)
        {
            self.frames = Some(Column::new(self.device, DType::U8, elements)?);
        }
        Ok(())
    }

    /// The one clip a decode bucket serves, or the refusal that names how many the batch carries.
    ///
    /// The bucket's extent names one raster, so a batch of two requests for it describes two clips in
    /// one graph - a plan the mechanism does not make, and one whose second clip would be decoded
    /// from the first clip's columns.
    fn clip_of(media: &crate::batch::MediaColumns) -> Result<&Vec<Vec<u8>>> {
        let [buffers] = media.inputs.as_slice() else {
            return Err(Error::Invalid(format!(
                "a decode bucket holds one clip and this batch carries {} request(s)",
                media.inputs.len()
            )));
        };
        Ok(buffers)
    }

    /// Stages `bucket`'s latents, and the model's own tables where the cache does not hold them.
    ///
    /// **Every table is the model's and every table lives in `Tables`.** `prepare_layout` writes the
    /// bytes from the layout and `fill` puts them in the cache the captured graph addresses, keyed
    /// by `(layout, name)` - so this function stages the clip's latents and nothing else. A batch
    /// that carried tables would state the model's layout arithmetic a second time, which
    /// `MediaColumns::staged` already rules out: "It does not own what the model prepares on the
    /// device - the encoded conditioning rows, the latent trajectory".
    ///
    /// A warm capture passes no latents, and **the model supplies no clip for one**: `prepare_layout`
    /// stages this decoder's *tables* (`video_vae`'s is one `video_positions` table), not a latent
    /// trajectory, so the column is zeroed - defined values in its own element type, which is what a
    /// capture needs and all it needs. Reading `prepare_layout`'s first buffer as a clip was wrong in
    /// the direction that does not fail: a table's bytes are shorter than the clip, so they were
    /// written over the front of the column and the capture read a clip made of rotary positions.
    ///
    /// A replay passes the batch's buffer - a staged row is the request's conditioning blob and its
    /// parameter record - but `stage` reaches this only when the product store holds no trajectory
    /// for the request, and a blob of another length is then refused rather than copied over a prefix.
    fn stage_columns(&mut self, bucket: Bucket, latents: Option<&[u8]>) -> Result<()> {
        self.size_raster(bucket)?;
        self.layout(bucket)?;
        // The tables the graph addresses, in the cache they are keyed in. `prepare_layout` is asked
        // exactly once per layout, here, and its bytes never touch the latents column.
        self.fill(bucket)?;
        match latents {
            Some(bytes) => {
                // The column is the model's extent for this layout, so a blob of another length is
                // a request for another clip: refused by name rather than copied over a prefix.
                let expected = self.columns.latents.host.len();
                if bytes.len() != expected {
                    return Err(Error::Invalid(format!(
                        "a decode of bucket {bucket} states {expected} byte(s) of latents and the \
                         batch carries {}: the clip is not this bucket's",
                        bytes.len()
                    )));
                }
                self.columns.latents.write(bytes)?;
            }
            None => {
                // The clip is this runner's own column, empty until a request publishes one: zeros
                // are defined, hold no clip's arithmetic and cost no copy.
                //
                // SAFETY: the column is this runner's own pinned storage, sized for the largest
                // bucket it was bound with, and nothing else writes it here: the capture this stages
                // is the next job on the stream, and a replay of this bucket is not concurrent with
                // a stage (a batch is staged once, before its jobs are issued).
                unsafe {
                    self.columns.latents.host.as_mut_slice().fill(0);
                }
            }
        }
        self.staged = Some(bucket);
        Ok(())
    }

    /// Stages a batch's clip into the columns `bucket`'s graph reads.
    fn stage(&mut self, batch: &Batch, bucket: Bucket) -> Result<()> {
        let media = batch.media.as_ref().ok_or_else(|| {
            Error::Invalid(
                "a decoder runner serves a media batch; this one states no computation".into(),
            )
        })?;
        // The batch names its requests, and a decode bucket serves one clip, so the name kept here
        // is the one the replay's publish uses.
        let request = media.request(0)?.request;
        // The column may still hold another request's clip: the store is where that clip's life
        // ends, so the store is asked.
        let still_held = match (self.products.as_deref(), self.published_for) {
            (Some(store), Some(last)) if last != request => {
                (0..self.published_units).any(|unit| store.holds(last, Product::Video(unit)))
            }
            _ => false,
        };
        clip_is_releasable(request, self.published_for, still_held)?;
        self.staged_request = Some(request);
        self.staged_frames = Some(media.request(0)?.frames);
        self.staged_unit = media.request(0)?.unit;
        self.size_timeline(bucket, media.request(0)?.frames)?;
        // **The trajectory is the denoiser's product, and the store is where it lives.** A staged row
        // is the request's conditioning blob and its parameter record; the latent trajectory is
        // explicitly not the batch's (`MediaColumns::staged`: "It does not own what the model
        // prepares on the device - the encoded conditioning rows, the latent trajectory"), and a
        // denoise step publishes it under this request's name at every step. So the clip is copied
        // from there, ordered on this runner's stream, which is the only path that also works when
        // the denoiser ran on another device.
        if let Some(products) = self.products.clone()
            && products.holds(request, Product::VideoLatents)
        {
            return self.stage_product(bucket, request, products.as_ref());
        }
        let row = Self::clip_of(media)?;
        let Some(latents) = row.first() else {
            return Err(Error::Invalid(
                "a staged decode row carries no buffer: a media row is the request's conditioning \
                 blob and its parameter record"
                    .into(),
            ));
        };
        self.stage_columns(bucket, Some(latents))
    }

    /// Stages the trajectory the denoiser published for `request`, and notes the clip to gather.
    ///
    /// **Nothing here waits or crosses to the host.** The copy is on this runner's stream and the store
    /// orders it after the producing step's write, which is what makes it correct when the denoiser ran
    /// on another stream or another device; the replay then gathers the clip's rows on the same stream.
    /// So staging the next clip costs the previous clip's decode nothing.
    fn stage_product(
        &mut self,
        bucket: Bucket,
        request: ipc::RequestKey,
        store: &dyn Products,
    ) -> Result<()> {
        self.size_raster(bucket)?;
        self.fill(bucket)?;
        let Some(published) = store.product(request, Product::VideoLatents)? else {
            return Err(crate::runners::products::absent(
                request,
                Product::VideoLatents,
            ));
        };
        // The boundary converts nothing: the decoder states the type it reads its latents in, and a
        // trajectory of another type would be a reinterpretation of its bytes.
        if published.dtype != self.latent {
            return Err(Error::Invalid(format!(
                "the decoder reads its latents as {} and the denoiser published {}",
                self.latent, published.dtype
            )));
        }
        let element = published
            .dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{} has no whole-byte size", published.dtype)))?;
        let elements = published.bytes / element;
        if self
            .trajectory
            .as_ref()
            .is_none_or(|column| column.device.numel() < elements)
        {
            self.trajectory = Some(Column::new(self.device, published.dtype, elements)?);
        }
        let column = self
            .trajectory
            .as_ref()
            .ok_or_else(|| Error::Invariant("the trajectory column was just sized".into()))?;
        let destination =
            TensorView::new(column.device.ptr, published.dtype, Shape::new(&[elements]));
        if !store.copy_into(request, Product::VideoLatents, &destination, &self.stream)? {
            return Err(crate::runners::products::absent(
                request,
                Product::VideoLatents,
            ));
        }
        self.gather = Some(self.model.clip_start(self.staged_unit));
        self.trajectory_elements = elements;
        self.staged = Some(bucket);
        Ok(())
    }

    /// Gathers the staged clip's block out of the trajectory, on the stream the replay runs on.
    fn gather_replayed(&mut self, bucket: Bucket, generation: u64, first: u32) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved();
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let forward = nn::Forward::new(&stream, &kernels, &arena);
        let layout = self.layout(bucket)?;
        let column = self
            .trajectory
            .as_ref()
            .ok_or_else(|| Error::Invariant("a clip was staged with no trajectory".into()))?;
        let trajectory = TensorView::new(
            column.device.ptr,
            self.latent,
            Shape::new(&[self.trajectory_elements]),
        );
        let block = self.latents(&layout)?;
        self.model
            .gather_latents(&forward, &layout, trajectory, first, block)
            .map_err(Error::from)
    }

    /// Stages `bucket` with no batch behind it, so a first capture has defined columns to read.
    fn stage_warm(&mut self, bucket: Bucket) -> Result<()> {
        if self.staged == Some(bucket) {
            return Ok(());
        }
        self.stage_columns(bucket, None)
    }

    /// Uploads the staged columns, in one place, so a capture and a replay agree on what the graph
    /// reads.
    fn upload(&self) -> Result<()> {
        self.columns.latents.upload(&self.stream)
    }

    /// The clip's latents, `[voxels, channels]`, as the model reads them.
    fn latents(&self, layout: &Layout) -> Result<TensorView> {
        let video = layout.video;
        let rows = block_rows(self.model.as_ref(), layout)?;
        Ok(TensorView::new(
            self.columns.latents.device.ptr,
            self.latent,
            Shape::new(&[rows, u64::from(video.channels)]),
        ))
    }

    /// The raster this runner's replay writes, `[channels, frames, rows, columns]`.
    fn raster(&self, bucket: Bucket, layout: &Layout) -> Result<TensorView> {
        let Extent::Clip {
            frames,
            height,
            width,
        } = bucket.extent
        else {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is not a decode bucket"
            )));
        };
        Ok(TensorView::new(
            self.columns.raster.device.ptr,
            self.dtype,
            Shape::new(&[
                u64::from(RASTER_CHANNELS),
                u64::from(frames),
                u64::from(height),
                u64::from(width),
            ]),
        ))
    }

    /// Publishes the clip a replay just wrote, so the transport can read it by the request's name.
    ///
    /// **A decoded clip has nowhere to go without this.** The raster is this runner's own column and
    /// no call outside it can name the region; `BatchOutput` carries calls and timings, and `Runner`
    /// has no accessor for a media output. The store is the tree's one way for a runner to hand
    /// another call a region, so the clip is published like the encoded states are.
    ///
    /// The contract is the store's: the view names memory the **reader** must copy before this
    /// runner's next decode of the bucket, and the copy is issued on the reader's stream. Publishing
    /// nothing when no store was attached is deliberate — the consumer's read refuses by name, and a
    /// decode that refused here would fail a step whose own work succeeded.
    fn publish(&mut self, bucket: Bucket) -> Result<()> {
        let Some(products) = self.products.as_ref() else {
            return Ok(());
        };
        let Some(request) = self.staged_request else {
            return Err(Error::Invariant(format!(
                "a replay of bucket {bucket} publishes a clip and no request is staged: the batch \
                 it served was never staged, so the clip has no name to publish under"
            )));
        };
        let layout = self.layout(bucket)?;
        // With a postprocessor the request's product is the timeline's frames - `[frames, rows,
        // columns, 3]` bytes, which is what an encoder takes - and without one it is the clip's own
        // raster, which is all such a deployment has to publish.
        let published = match self.postprocessor {
            Some(_) => self.frames_view(bucket)?,
            None => self.raster(bucket, &layout)?,
        };
        let written = crate::runners::products::written_on(&self.stream)?;
        products.publish(
            request,
            Product::Video(self.staged_unit),
            self.device,
            self.stream_id(),
            Some(written),
            &published,
        )?;
        // The column now holds this request's clip until a reader forgets it: that is what a second
        // decode of another request is refused against.
        if self.published_for != Some(request) {
            self.published_units = 0;
        }
        self.published_units = self.published_units.max(self.staged_unit + 1);
        self.published_for = Some(request);
        Ok(())
    }

    /// Walks the decode once for `bucket`, recording every operator launch's nodes into the
    /// generation's capture (4.5).
    fn walk(&mut self, bucket: Bucket, generation: &Arc<ops::Generation>) -> Result<()> {
        let stream = self.stream.clone();
        if self.staged != Some(bucket) {
            self.stage_warm(bucket)?;
        }
        let first = self.pool()?.begin(self.stream_id(), bucket)?;
        self.upload()?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let forward = nn::Forward::new(&stream, generation, &arena);
        let layout = self.layout(bucket)?;
        let tables = self.model.layout_tables(&layout).map_err(Error::from)?;
        let key = Self::key(bucket);
        if !tables.is_empty() {
            self.fill(bucket)?;
        }
        let state = self
            .tables
            .views(self.lifecycle.generation.number(), &key, &tables)?;
        let latents = self.latents(&layout)?;
        let raster = self.raster(bucket, &layout)?;
        self.model
            .decode(
                &forward,
                latents,
                &state,
                DecodedMedia {
                    video: Some(raster),
                    // A video decoder writes frames; a bucket that states a clip states no audio, so
                    // asking this block for audio would be a plan whose extent cannot say so.
                    audio: None,
                },
            )
            .map_err(Error::from)?;
        if first {
            self.pool()?
                .record(self.stream_id(), bucket, arena.used())?;
        }
        Ok(())
    }

    /// Splices the clip a replay just decoded, on the stream that replayed it.
    ///
    /// This is not part of the walk. A graph holds one bucket's decode, and a splice belongs to a
    /// request: which clip of the timeline this is, how many frames it keeps and how many it fades
    /// are that request's, and they change from clip to clip, so a capture has none to record. The
    /// stream's order is what lets it read the raster the graph wrote.
    fn splice_replayed(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        if self.postprocessor.is_none() {
            return Ok(());
        }
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let forward = nn::Forward::new(&stream, &kernels, &arena);
        let layout = self.layout(bucket)?;
        self.splice_clip(&forward, bucket, &layout)
    }

    /// Writes the clip a replay just decoded into its request's timeline.
    ///
    /// **The component has two entry points and this is the second:** a decode turns latents into a
    /// clip, and the postprocessor puts the clip's frames where the timeline holds them, fading its
    /// first kept frames against the predecessor's tail and de-normalising the bytes. Without a
    /// postprocessor there is no timeline - a deployment that serves none publishes the clip's own
    /// raster, which is what the encoder of such a deployment reads.
    ///
    /// The schedule is the block's (`Decoder::splices`), so the frame count a timeline holds, the
    /// body each clip keeps and the overlap the fade reads are all the checkpoint's arithmetic. This
    /// runner states only where the clip is: **each clip gets its own raster**, so the splice's
    /// destination starts at frame zero and the product names the frames this clip wrote.
    fn splice_clip(
        &mut self,
        forward: &nn::Forward,
        bucket: Bucket,
        layout: &Layout,
    ) -> Result<()> {
        let Some(postprocessor) = self.postprocessor.clone() else {
            return Ok(());
        };
        let request = self.staged_request.ok_or_else(|| {
            Error::Invariant(
                "a replay splices a clip and no request is staged: the splice would have no \
                 timeline to write into"
                    .into(),
            )
        })?;
        let frames = self.staged_frames.ok_or_else(|| {
            Error::Invariant(format!(
                "request {} was staged without a timeline length, so the splice schedule cannot be \
                 asked for",
                request.request_id.0
            ))
        })?;
        let index = self.staged_unit;
        let schedule = postprocessor.splices(frames).map_err(|error| {
            Error::Invalid(format!(
                "request {} asked for a timeline of {frames} frame(s) and the postprocessor serves \
                 no such timeline: {error}",
                request.request_id.0
            ))
        })?;
        let Some(splice) = schedule.get(index as usize).copied() else {
            return Err(Error::Invalid(format!(
                "request {} asks for clip {index} of a timeline of {frames} frame(s), and the \
                 timeline is {} clip(s)",
                request.request_id.0,
                schedule.len()
            )));
        };
        let Extent::Clip { height, width, .. } = bucket.extent else {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is not a decode bucket and a clip was spliced"
            )));
        };
        let channels = RASTER_CHANNELS;
        // What this clip fades against is the schedule's `overlap`, and what it must leave behind is
        // its **successor's**: the timeline's first clip fades nothing and still owes the second clip
        // its tail, and the last clip has no one to hand one to.
        let successor = tail_for_successor(&schedule, index);
        let overlap = successor.max(splice.overlap);
        self.size_tail(channels, height, width, overlap, schedule.len() as u32)?;
        let previous = if splice.overlap > 0 {
            Some(self.predecessor_tail(request, index)?)
        } else {
            None
        };
        let clip = self.raster(bucket, layout)?;
        if self.diag.enabled() {
            let _ = self.diag.write(
                self.device,
                &self.stream,
                "decoder",
                &format!("raster_{index}"),
                clip,
            );
            let _ = self.diag.write(
                self.device,
                &self.stream,
                "decoder",
                &format!("latents_{index}"),
                TensorView::new(
                    self.columns.latents.device.ptr,
                    self.latent,
                    Shape::new(&[self.columns.latents.device.numel()]),
                ),
            );
        }
        let column = self.frames.as_ref().ok_or_else(|| {
            Error::Invariant(
                "a postprocessor is bound and no frames column was sized for the bucket".into(),
            )
        })?;
        let video = TensorView::new(
            timeline_frame(column, splice.first, height, width),
            DType::U8,
            Shape::new(&[
                u64::from(splice.keep),
                u64::from(height),
                u64::from(width),
                3_u64,
            ]),
        );
        postprocessor
            .splice(
                forward,
                previous,
                clip,
                nn::capability::Splice { first: 0, ..splice },
                DecodedMedia {
                    video: Some(video),
                    audio: None,
                },
            )
            .map_err(Error::from)?;
        self.keep_tail(forward, request, index, &clip, channels, height, width, successor)?;
        self.staged_keep = Some(splice.keep);
        self.staged_first = splice.first;
        Ok(())
    }

    /// Sizes the tail columns for an overlap of `overlap` frames and a timeline of `clips` clips, the
    /// first time they are needed.
    ///
    /// The count is the schedule's, not this runner's: a timeline whose fade reads five frames and one
    /// that reads ten are one formula apart in the block and one column here. **Each clip keeps its
    /// own slot**, because a rank's next clip is decoded while its neighbour may still be reading the
    /// last one's tail.
    fn size_tail(
        &mut self,
        channels: u32,
        height: u32,
        width: u32,
        overlap: u32,
        clips: u32,
    ) -> Result<()> {
        if overlap == 0 {
            return Ok(());
        }
        let elements = u64::from(channels)
            .checked_mul(u64::from(overlap))
            .and_then(|elements| elements.checked_mul(u64::from(height)))
            .and_then(|elements| elements.checked_mul(u64::from(width)))
            .ok_or_else(|| Error::Invalid("a clip's tail extent overflows".into()))?;
        let slots = elements
            .checked_mul(u64::from(clips))
            .ok_or_else(|| Error::Invalid("a timeline's tails overflow".into()))?;
        if self
            .tail
            .as_ref()
            .is_none_or(|column| column.device.numel() < slots)
        {
            self.tail = Some(Column::new(self.device, self.dtype, slots)?);
        }
        Ok(())
    }

    /// The tail clip `index` fades against, which its predecessor's rank published.
    ///
    /// The predecessor may be on another rank, decoding at the same time, so this waits for the
    /// publication. A decode needs nothing but its own latents, so no rank waits on one that waits
    /// on it, and the wait is bounded by one decode; the bound here names a rank that never answers.
    fn predecessor_tail(&self, request: ipc::RequestKey, index: u32) -> Result<TensorView> {
        let store = self
            .products
            .as_ref()
            .ok_or_else(|| crate::runners::products::no_store(request))?;
        let kind = Product::Tail(index - 1);
        let deadline = std::time::Instant::now() + TAIL_WAIT;
        while !store.holds(request, kind) {
            if std::time::Instant::now() >= deadline {
                return Err(Error::Invalid(format!(
                    "request {} waited {} s for the tail of clip {} and its rank published none",
                    request.request_id.0,
                    TAIL_WAIT.as_secs(),
                    index - 1
                )));
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        // The tail lies in pinned host storage, which every device reads, so the splice reads it
        // where its rank left it: a copy between two pinned regions is the driver's host copy, which
        // waits for this stream and holds the device thread for the clip just launched.
        store
            .lent(request, kind, &self.stream)?
            .ok_or_else(|| crate::runners::products::absent(request, kind))
    }

    /// Keeps the clip's last `overlap` frames for the next clip's fade, and publishes them.
    ///
    /// The tail is one contiguous run per channel of the clip's own raster (`[channels, frames, rows,
    /// columns]`), so this is one block copy on the runner's stream and no arithmetic. It has to
    /// happen before the next decode, which writes over the clip's column. Both columns are pinned
    /// host memory, which the driver's `cuMemcpyAsync` copies on the host once the stream drains, so
    /// the copy is a kernel: a memcpy here held the device thread for the whole decode.
    #[allow(clippy::too_many_arguments)]
    fn keep_tail(
        &self,
        forward: &nn::Forward,
        request: ipc::RequestKey,
        index: u32,
        clip: &TensorView,
        channels: u32,
        height: u32,
        width: u32,
        overlap: u32,
    ) -> Result<()> {
        let Some(tail) = self.tail.as_ref() else {
            return Ok(());
        };
        if overlap == 0 {
            return Ok(());
        }
        let frames = clip.shape.dim(1);
        let start = frames
            .checked_sub(u64::from(overlap))
            .ok_or_else(|| Error::Invalid("a clip is shorter than its own tail".into()))?;
        let stride = u64::from(overlap)
            .saturating_mul(u64::from(height))
            .saturating_mul(u64::from(width));
        let slot = u64::from(index) * u64::from(channels) * stride;
        let plane = frames
            .saturating_mul(u64::from(height))
            .saturating_mul(u64::from(width));
        let block = Shape::new(&[u64::from(channels), 1, 1, stride]);
        let source = TensorView::with_strides(
            clip.element(&[0, start, 0, 0]),
            self.dtype,
            block,
            nn::Strides::new(&[plane, stride, stride, 1]),
        );
        let destination = TensorView::new(tail.device.element(&[slot]), self.dtype, block);
        let strides = [stride as i64, stride as i64, stride as i64, 1];
        nn::layers::CopyBlock::new(self.dtype)
            .map_err(Error::from)?
            .forward_wide(forward, source, destination, &strides)
            .map_err(Error::from)?;
        let store = self
            .products
            .as_ref()
            .ok_or_else(|| crate::runners::products::no_store(request))?;
        let published = TensorView::new(
            tail.device.element(&[slot]),
            self.dtype,
            Shape::new(&[
                u64::from(channels),
                u64::from(overlap),
                u64::from(height),
                u64::from(width),
            ]),
        );
        store.publish(
            request,
            Product::Tail(index),
            self.device,
            self.stream_id(),
            Some(crate::runners::products::written_on(&self.stream)?),
            &published,
        )
    }

    /// The frames the splice wrote, `[frames, rows, columns, 3]` bytes, as the product states them.
    fn frames_view(&self, bucket: Bucket) -> Result<TensorView> {
        let Extent::Clip { height, width, .. } = bucket.extent else {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is not a decode bucket and its frames were asked for"
            )));
        };
        let column = self.frames.as_ref().ok_or_else(|| {
            Error::Invariant("no frames column was sized for a spliced bucket".into())
        })?;
        let keep = self.staged_keep.ok_or_else(|| {
            Error::Invariant("a splice wrote frames and the schedule's keep was not staged".into())
        })?;
        Ok(TensorView::new(
            timeline_frame(column, self.staged_first, height, width),
            DType::U8,
            Shape::new(&[u64::from(keep), u64::from(height), u64::from(width), 3_u64]),
        ))
    }
}

/// The frames clip `index` must leave behind: what its successor fades against, none for the last.
///
/// A splice's `overlap` is the fade it **reads**, so reading it for the tail a clip keeps left the
/// timeline's first clip keeping nothing, and the second clip then faded against a column nobody wrote.
fn tail_for_successor(schedule: &[nn::capability::Splice], index: u32) -> u32 {
    schedule
        .get(index as usize + 1)
        .map_or(0, |next| next.overlap)
}

/// Where timeline frame `first` of a `[frames, rows, columns, 3]` byte column starts.
fn timeline_frame(column: &Column, first: u32, height: u32, width: u32) -> ops::DevicePtr {
    let elements = u64::from(first) * u64::from(height) * u64::from(width) * 3;
    TensorView::new(column.device.ptr, DType::U8, Shape::new(&[1])).offset_elements(elements)
}

impl<M: Decoder + Send + Sync> Runner for DecoderRunner<M> {
    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    /// A clip's replay reads and writes on the stream, in order, and each clip keeps its own frames
    /// in the timeline, so the next clip can be launched behind it and a finished clip's encode runs
    /// beside it. The diagnosis reads the columns between clips, so it waits for each.
    fn overlaps_replays(&self) -> bool {
        !self.diag.enabled()
    }

    fn stream(&self) -> Option<&platform::Stream> {
        Some(&self.stream)
    }

    /// Serves one clip: choose its bucket, stage its buffers, and return the replay (4.5).
    ///
    /// The bucket is chosen from the computation the batch states - the clip's raster is the extent -
    /// and never from how many requests it happens to carry: a decode graph is captured for one
    /// canvas, and a clip of another canvas must hear that it has no bucket rather than be served by
    /// a graph whose constants belong to a different raster.
    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        let media = media_of(&batch)?.clone();
        media.check()?;
        let computation = media.computation;
        if computation.variant() != Some(Variant::Decode) {
            return Err(Error::Invalid(format!(
                "a decoder runner serves a decode; this batch states a {:?}",
                computation.call
            )));
        }
        let bucket = self.buckets.select_media(&computation).ok_or_else(|| {
            Error::Invalid(format!(
                "no bucket was captured for {:?} of {} request(s) at {:?}",
                computation.call, computation.rows, computation.extent
            ))
        })?;
        self.stage(&batch, bucket)?;
        self.calls.clone_from(&batch.calls);
        Ok(Jobs::new([Job::Replay {
            runner: self.id,
            bucket,
            generation: batch.generation,
            batch: batch.id,
        }]))
    }

    fn media_plan(&self) -> Vec<Bucket> {
        self.capture_order()
    }

    fn capture_plan(&self) -> Vec<Bucket> {
        self.capture_order()
    }

    fn cache(&self) -> Option<Arc<Mutex<crate::memory::prefix_cache::PrefixCache>>> {
        // A decoder reads no paged cache: its latents arrive as the batch's own buffers.
        None
    }

    fn warmup(&mut self) -> Result<Jobs> {
        let generation = self.lifecycle.generation.number();
        let mut jobs = Vec::new();
        for bucket in self.capture_order() {
            if self.lifecycle.graphs.captured_for(bucket, generation).is_none() {
                jobs.push(Job::Capture {
                    runner: self.id,
                    bucket,
                    generation,
                });
            }
        }
        Ok(Jobs::new(jobs))
    }

    fn precopy(&mut self, next: &Arc<ops::Generation>) -> Result<Jobs> {
        self.lifecycle.precopy(next)
    }

    fn switchover(&mut self, generation: u64) -> Result<()> {
        self.lifecycle.switchover(generation)
    }

    fn revert(&mut self, generation: u64) {
        self.lifecycle.revert(generation);
    }

    fn purge(&mut self, generation: u64) {
        self.lifecycle.purge(generation);
        // A generation's tables are charged to resident and retire with it, the same rule as its
        // weights, so the release is here beside the graphs rather than on a second path (4.3).
        self.tables.release(generation);
    }

    fn discard(&mut self, generation: u64) -> Result<()> {
        self.lifecycle.discard(generation)
    }

    fn attach_products(&mut self, products: Arc<dyn Products>) {
        self.products = Some(products);
    }

    fn claim_arena(&mut self) -> Result<()> {
        self.claim()
    }

    fn attach_pools(&mut self, pools: Arc<Mutex<Pools>>) {
        self.tables.attach(Arc::clone(&pools));
        self.lifecycle.attach_pools(Arc::clone(&pools));
        self.pools = Some(pools);
    }

    fn attach_counters(&mut self, counters: Arc<crate::device::SharedCounters>) {
        self.counters = Some(counters);
    }

    fn serve(&mut self, job: &Job) -> Result<BatchOutput> {
        match job {
            Job::Replay {
                bucket, generation, ..
            } => {
                let stream = self.stream.clone();
                match self.gather.take() {
                    Some(first) => self.gather_replayed(*bucket, *generation, first)?,
                    None => self.upload()?,
                }
                let (served, captured) = self
                    .lifecycle
                    .graphs
                    .captured_for(*bucket, *generation)
                    .ok_or_else(|| {
                    Error::Invariant(format!(
                        "no graph of bucket {bucket} in generation {generation} or in a \
                             generation it replaces"
                    ))
                })?;
                if served != *generation {
                    tracing::debug!(
                        bucket = %bucket,
                        generation,
                        served,
                        "the bucket was prepared for an earlier generation; its programs did not change"
                    );
                }
                let launched = std::time::Instant::now();
                captured.launch(&stream)?;
                if let Some(counters) = &self.counters {
                    counters.launches(captured.kernel_nodes());
                }
                self.splice_replayed(*bucket, *generation)?;
                // The clip is written by the graph that just launched, and the stream's order is
                // what makes the view readable: a reader's copy is issued after this one.
                self.publish(*bucket)?;
                // The clip's raster is this runner's own column, written by the graph that just
                // launched: the stream's order is what makes it readable, and the completion that
                // turns it into an artifact is the transport's rather than this step's.
                Ok(BatchOutput {
                    calls: answered(&mut self.calls),
                    timing: ipc::TimingCounters {
                        device_us: launched.elapsed().as_micros() as u64,
                        ..Default::default()
                    },
                    forward_stats: None,
                })
            }
            Job::Capture {
                bucket, generation, ..
            } => self
                .capture(*bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Rebind {
                bucket, generation, ..
            } => self
                .rebind(*bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Join { .. } => Err(Error::Invariant(
                "a video decoder runner joins no expert step".into(),
            )),
            Job::Shadow { operator, key, .. } => Err(Error::Invariant(format!(
                "a decoder runner compares no operator, and the device asked it to shadow \
                 {operator} {key}"
            ))),
            // A host call is no device work: the media stage performs it, and a device runner
            // handed one was given a job another runner returned.
            Job::Host { call, .. } => Err(Error::Invariant(format!(
                "a runner that serves device work was asked for host call {}",
                call.0
            ))),
        }
    }
}

impl<M: Decoder + Send + Sync> DecoderRunner<M> {
    /// Captures one bucket of `generation`: the walk here, the graph kept by the lifecycle.
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = warmed_capture(&stream, || self.walk(bucket, &kernels))?;
        self.lifecycle.capture(bucket, generation, walked, &kernels)
    }

    /// Prepares one bucket of the next generation from a walk of the same bucket (4.5).
    fn rebind(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = walk_capture(&stream, || self.walk(bucket, &kernels))?;
        self.lifecycle.rebind(bucket, generation, walked, &kernels)
    }
}

#[cfg(test)]
mod tests {
    //! The decoder runner's behaviour at its own boundary: which canvas a bucket states, what the
    //! plan is told its tables need, and what a request for a canvas the decoder cannot serve gets.
    //!
    //! Nothing here needs a device: each behaviour is a function of the bucket and the model, which is
    //! why a model that refuses a canvas can be driven without one.

    /// A clip keeps the tail its successor fades against, which the timeline's first clip owes too
    /// though it fades nothing itself, and the last clip has no one to hand one to.
    #[test]
    fn a_clip_keeps_the_tail_its_successor_reads() {
        use nn::capability::Splice;
        let schedule = [
            Splice {
                first: 0,
                keep: 17,
                overlap: 0,
            },
            Splice {
                first: 17,
                keep: 17,
                overlap: 5,
            },
            Splice {
                first: 34,
                keep: 22,
                overlap: 5,
            },
        ];
        let tails: Vec<u32> = (0..3)
            .map(|index| super::tail_for_successor(&schedule, index))
            .collect();
        assert_eq!(tails, [5, 5, 0]);
    }

    /// A clip still held under another request refuses a second decode, by both requests' names.
    ///
    /// The rule is one raster column's: a decode of another request would replace the clip before its
    /// artifact was read, and nothing orders that read against the step that writes the column. What
    /// the refusal must therefore carry is the pair — who holds the column, and who is asking — because
    /// the reader that has to act on it is the holder's transport.
    #[test]
    fn a_second_clip_is_refused_while_the_first_is_unread() {
        let first = ipc::RequestKey::new(0, ipc::RequestId(1), 0);
        let second = ipc::RequestKey::new(0, ipc::RequestId(2), 0);

        assert!(
            clip_is_releasable(second, None, false).is_ok(),
            "a runner that has published no clip decodes any request"
        );
        assert!(
            clip_is_releasable(second, Some(first), false).is_ok(),
            "a clip the reader forgot no longer holds the column"
        );
        assert!(
            clip_is_releasable(first, Some(first), true).is_ok(),
            "the holder decoding again reads its own clip, which the transport is about to take"
        );

        let error = clip_is_releasable(second, Some(first), true)
            .expect_err("the column still holds the first request's clip");
        let message = format!("{error}");
        for expected in [
            "request 2 asks for a decode",
            "still holds request 1's clip",
            "`forget`",
        ] {
            assert!(
                message.contains(expected),
                "the refusal must name {expected:?}: {message}"
            );
        }
    }
    use super::*;
    use nn::capability::StateTable;

    /// The clip the stub decoder serves, and the layout it decodes it from.
    const SERVED: (u32, u32, u32) = (2, 3, 4);

    fn layout_of_served() -> Layout {
        Layout {
            video: nn::capability::LatentShape {
                channels: 8,
                frames: 1,
                height: 1,
                width: 1,
            },
            audio: None,
            text_rows: 2,
            condition_rows: 0,
            condition_elements: 0,
        }
    }

    /// A decoder that serves one canvas and refuses every other, with two tables of known sizes.
    struct Fixed;

    impl Decoder for Fixed {
        fn layout_tables(&self, _layout: &Layout) -> nn::Result<Vec<StateTable>> {
            Ok(vec![
                StateTable {
                    name: "positions",
                    dtype: DType::F32,
                    shape: Shape::new(&[6]),
                },
                StateTable {
                    name: "offsets",
                    dtype: DType::I32,
                    shape: Shape::new(&[2, 3]),
                },
            ])
        }

        fn prepare_layout(&self, _layout: &Layout, out: &mut [Vec<u8>]) -> nn::Result<()> {
            for buffer in out {
                buffer.fill(0);
            }
            Ok(())
        }

        fn element_type(&self) -> DType {
            DType::F16
        }

        fn layout_of(&self, clip: Clip) -> nn::Result<Layout> {
            if (clip.frames, clip.height, clip.width) == SERVED {
                return Ok(layout_of_served());
            }
            Err(nn::Error::Shape {
                owner: "the stub decoder",
                problem: format!("it serves {}x{}x{} only", SERVED.0, SERVED.1, SERVED.2),
            })
        }

        fn decode(
            &self,
            _f: &nn::Forward,
            _latents: TensorView,
            _state: &[TensorView],
            _out: DecodedMedia,
        ) -> nn::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn the_plan_is_told_what_every_bucket_of_a_canvas_needs() {
        // The two tables are six f32 and six i32: one number the mechanism books before admission,
        // derived from the model's declaration rather than from a runner's own constant.
        let bytes =
            DecoderRunner::<Fixed>::table_bytes(&Fixed, Bucket::clip(SERVED.0, SERVED.1, SERVED.2))
                .expect("the tables' bytes");
        assert_eq!(bytes, 6 * 4 + 6 * 4);
    }

    #[test]
    fn a_bucket_that_is_not_a_decode_states_what_it_is() {
        // A prefill bucket has no clip, and a decoder asked for its layout must say so by name rather
        // than answer with a layout that reads as one.
        let error =
            DecoderRunner::<Fixed>::table_bytes(&Fixed, Bucket::prefill(1, 16, Variant::Encode))
                .expect_err("a prefill bucket is not a decode");
        assert!(
            format!("{error}").contains("is not a decode bucket"),
            "the refusal names the bucket's kind: {error}"
        );
    }

    #[test]
    fn a_canvas_the_decoder_cannot_serve_is_refused_by_name() {
        // The bound is the decoder's, not the plan's: a canvas it was not built for is refused where
        // the plan asks for its tables - before a capture, and never mid-clip.
        let error = DecoderRunner::<Fixed>::table_bytes(&Fixed, Bucket::clip(2, 3, 5))
            .expect_err("a canvas the decoder does not serve");
        let message = format!("{error}");
        assert!(
            message.contains("serves no clip of 2 frame(s) at 3x5"),
            "the refusal names the clip: {message}"
        );
        assert!(
            message.contains("it serves 2x3x4 only"),
            "and it carries the decoder's own reason: {message}"
        );
    }

    #[test]
    fn a_batch_of_two_requests_for_one_canvas_is_refused_by_its_count() {
        // A bucket's extent names one raster, so two requests in one batch of it describe two clips in
        // one graph: the second would be decoded from the first's columns. The refusal names the count
        // rather than serving one and silently dropping the other.
        let media = crate::batch::MediaColumns {
            computation: crate::plan::Computation {
                call: ipc::MediaCall::VideoDecoding,
                rows: 2,
                tokens: 0,
                extent: crate::plan::Extent::Clip {
                    frames: SERVED.0,
                    height: SERVED.1,
                    width: SERVED.2,
                },
            },
            inputs: vec![vec![vec![0_u8; 8]], vec![vec![0_u8; 8]]],
        };
        let error = DecoderRunner::<Fixed>::clip_of(&media).expect_err("two clips, one bucket");
        assert!(
            format!("{error}").contains("holds one clip and this batch carries 2 request(s)"),
            "the refusal counts the batch: {error}"
        );

        let one = crate::batch::MediaColumns {
            inputs: vec![vec![vec![0_u8; 8]]],
            ..media
        };
        assert_eq!(
            DecoderRunner::<Fixed>::clip_of(&one)
                .expect("one clip")
                .len(),
            1,
            "one request carries the clip's one set of buffers"
        );
    }

    #[test]
    fn a_staged_row_is_the_conditioning_blob_the_batch_was_built_with() {
        // The row's shape is the engine's, not this file's: `MediaColumns::staged` writes the
        // request's conditioning blob and its parameter record, one row per request, and the record is
        // read by `media.request`. The blob is the clip only as a fallback: `stage` takes the
        // trajectory from the product store first, so this ordering is what a batch without a
        // published product has to satisfy.
        let media = crate::batch::MediaColumns {
            computation: crate::plan::Computation {
                call: ipc::MediaCall::VideoDecoding,
                rows: 1,
                tokens: 0,
                extent: crate::plan::Extent::Clip {
                    frames: SERVED.0,
                    height: SERVED.1,
                    width: SERVED.2,
                },
            },
            inputs: vec![vec![vec![7_u8; 32], vec![0_u8; 40]]],
        };
        let row = DecoderRunner::<Fixed>::clip_of(&media).expect("one clip");
        assert_eq!(row.len(), 2, "a staged row is the blob and the record");
        assert_eq!(
            row.first().map(Vec::len),
            Some(32),
            "the first buffer is the clip's latents"
        );
    }

    #[test]
    fn a_canvas_is_decoded_at_the_element_type_the_decoder_states() {
        // `dtype = "auto"` leaves the element type to the checkpoint, so the runner reads it from the
        // model rather than from its own deployment: the plan's columns are sized before a graph
        // exists, and a byte count cannot say whether a column holds f16 or bf16.
        assert_eq!(Fixed.element_type(), DType::F16);
        assert_eq!(DType::F16.bytes(), Some(2));
    }

    /// A batch with a phase, no calls and no media: what the mechanism sends when it has nothing to
    /// compute, and the shape whose refusal is a fact about the batch rather than about a runner.
    fn no_computation() -> Batch {
        Batch {
            page_loads: Vec::new(),
            page_offloads: Vec::new(),
            id: crate::batch::BatchId(1),
            phase: ipc::PhaseId(0),
            calls: Vec::new(),
            shape: crate::plan::Shape::decode(1, Variant::DecodeGreedy),
            columns: crate::batch::BatchColumns::default(),
            generation: 1,
            cache_units: Vec::new(),
            block_tables: Vec::new(),
            media: None,
        }
    }

    #[test]
    fn a_batch_that_states_no_computation_is_refused_by_name() {
        let refused =
            media_of(&no_computation()).expect_err("a batch with no media computation is refused");
        let message = refused.to_string();
        assert!(
            message.contains("a decoder runner serves a media batch")
                && message.contains("states no computation"),
            "the refusal names what the runner serves and what the batch states, got: {message}"
        );
    }
}
