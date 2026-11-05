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
use crate::batch::{Batch, BatchOutput, Job, Jobs};
use crate::plan::{Bucket, Buckets, Extent, Variant};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::Column;
use crate::runners::lifecycle::{Lifecycle, walk_capture};
use crate::runners::products::{Product, Products};
use crate::runners::tables::Tables;
use crate::{Error, Result};

/// The columns one clip's decode is staged into, sized for the largest canvas the runner serves.
///
/// A decoder's buckets differ by canvas, and its columns are sized by the largest of them: the clip
/// the bucket was captured for reads exactly its own prefix, the way an encoder's columns are read.
struct DecoderColumns {
    /// `[voxels, channels]` the clip's latents, in the decoder's own element type.
    latents: Column,
    /// The clip's raster, `[channels, frames, rows, columns]`, in the decoder's own element type.
    raster: Column,
}

impl DecoderColumns {
    /// The clip's own two columns. **The tables are not here:** `Tables` holds them by
    /// `(layout, name)` and `fill` puts the model's bytes in the cache the graph addresses
    /// (`walk` reads `self.tables.views`), so a host column per table would be a second copy that
    /// nothing reads — and the second copy is what let a staged buffer land in a table's place.
    fn new(device: DeviceId, layout: &Layout, dtype: DType) -> Result<Self> {
        let video = layout.video;
        let voxels = u64::from(video.frames) * u64::from(video.height) * u64::from(video.width);
        let elements = voxels
            .checked_mul(u64::from(video.channels))
            .ok_or_else(|| Error::Invalid("a clip's latent extent overflows".into()))?;
        let latents = Column::new(device, dtype, elements)?;
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
    /// The element type the latents and the raster are stated in, as the model declares it.
    dtype: DType,
    columns: DecoderColumns,
    /// The bucket the columns are staged for, so a capture of a canvas knows how far they run.
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
    /// The pinned host scratch a trajectory of another element type lands in, allocated the first
    /// time one is staged and reused after: the denoiser accumulates in FP32 and this decoder reads
    /// the checkpoint's type, so the crossing is one conversion per clip, on the host.
    crossing: Option<platform::PinnedBuffer>,
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
    /// The predecessor clip's tail, `[channels, overlap, rows, columns]` in the model's type, which
    /// the next clip's five-frame fade reads. Sized on the first splice from the schedule's own
    /// overlap count, which is the model's number rather than this runner's, and reused after.
    tail: Option<Column>,
    /// How many clips of each request's timeline this runner has spliced, which is where the next
    /// splice starts.
    spliced: std::collections::HashMap<ipc::RequestKey, u32>,
    /// The frames the staged request's timeline holds, from its own row
    /// (`StagedRequest::frames`): the schedule's length, which is what the last clip's keep depends
    /// on.
    staged_frames: Option<u32>,
    /// The frames the splice staged last wrote for the staged request's clip: the extent the product
    /// states, taken from the schedule rather than recomputed here.
    staged_keep: Option<u32>,
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
        let columns = DecoderColumns::new(binding.device, layout, dtype)?;
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
            columns,
            staged: None,
            raster_elements: 0,
            products: None,
            staged_request: None,
            crossing: None,
            published_for: None,
            postprocessor,
            frames: None,
            tail: None,
            spliced: std::collections::HashMap::new(),
            staged_frames: None,
            staged_keep: None,
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
        let offered = self.pool()?.reserved();
        for bucket in self.capture_order() {
            let layout = Self::layout_of(&self.model, bucket)?;
            // One rule for both media runners, so the two cannot read an absent plan and a refused
            // one differently: `floor::checked` passes over a decoder that states no plan, refuses
            // any other answer by name, and refuses a share below the plan with both numbers.
            super::floor::checked(bucket, self.model.arena_bytes(&layout), offered)?;
        }
        for bucket in self.capture_order() {
            self.fill(bucket)?;
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
        let elements = u64::from(layout.video.channels)
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
            (Some(store), Some(last)) if last != request => store.holds(last, Product::Video),
            _ => false,
        };
        clip_is_releasable(request, self.published_for, still_held)?;
        self.staged_request = Some(request);
        self.staged_frames = Some(media.request(0)?.frames);
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

    /// Stages the clip the denoiser published for `request` into the column `bucket`'s graph reads.
    ///
    /// The column is sized from the layout first - `prepare_layout`'s own clip - so the destination's
    /// extent is the model's rather than a length this runner recomputes, and `copy_into` refuses a
    /// product whose bytes disagree with it instead of copying a prefix. The copy is on this
    /// runner's stream and the store orders it after the producing step's write, which is what makes
    /// it correct when the denoiser ran on another stream or another device.
    fn stage_product(
        &mut self,
        bucket: Bucket,
        request: ipc::RequestKey,
        store: &dyn Products,
    ) -> Result<()> {
        self.stage_columns(bucket, None)?;
        let layout = self.layout(bucket)?;
        let Some(published) = store.product(request, Product::VideoLatents)? else {
            return Err(crate::runners::products::absent(
                request,
                Product::VideoLatents,
            ));
        };
        if published.dtype == self.dtype {
            let latents = self.latents(&layout);
            if !store.copy_into(request, Product::VideoLatents, &latents, &self.stream)? {
                return Err(crate::runners::products::absent(
                    request,
                    Product::VideoLatents,
                ));
            }
            return Ok(());
        }
        // **The boundary's one conversion.** The denoiser accumulates in FP32 (`SAMPLE_DTYPE`) and
        // this decoder reads the checkpoint's own type; the reference spans the two with
        // `torch.autocast`, which design 9 replaced with declared per-layer precision, so exactly
        // one conversion happens here - once per clip, on the host, not per step and not in a loop.
        let source = self.trajectory_scratch(published)?;
        if !store.copy_into(request, Product::VideoLatents, &source, &self.stream)? {
            return Err(crate::runners::products::absent(
                request,
                Product::VideoLatents,
            ));
        }
        // The copy is ordered on this stream, so the host may read the bytes once the stream reaches
        // it. This is the one place in the media seam that waits on the device rather than ordering
        // more work after it, and it waits once per clip.
        //
        // **It waits on the decode stream, so it also waits for whatever that stream still holds** -
        // and what it holds is one clip's launch, which is what the media path's one-in-flight
        // admission rule guarantees is nothing. A deployment that ever admits two media requests at
        // once makes this the place the second one queues, and the fix then is a stream of the
        // runner's own for the crossing rather than a narrower wait.
        self.stream.synchronize()?;
        let crossing = self
            .crossing
            .as_ref()
            .ok_or_else(|| Error::Invariant("the trajectory scratch was just staged".into()))?;
        // SAFETY: the buffer is this runner's own pinned storage, sized for exactly the trajectory
        // and written by the copy above, which the synchronize has completed; nothing else holds a
        // reference to it while the conversion reads it.
        let bytes = unsafe { crossing.as_slice() };
        let converted = nn::weights::convert(
            published.dtype,
            self.dtype,
            bytes,
            "the denoiser's trajectory",
        )
        .map_err(Error::from)?;
        self.columns.latents.write(&converted)?;
        Ok(())
    }

    /// The pinned host scratch a trajectory of another element type lands in, sized to it once.
    ///
    /// The conversion is a host conversion, so the product's bytes must come back to the host first -
    /// and a product of another type cannot be copied into this runner's column at all, which the
    /// store refuses by both types. Reusing one buffer keeps the crossing to one allocation per
    /// runner rather than one per clip.
    fn trajectory_scratch(
        &mut self,
        published: crate::runners::products::Published,
    ) -> Result<TensorView> {
        let element = published
            .dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{} has no whole-byte size", published.dtype)))?;
        let bytes = usize::try_from(published.bytes)
            .map_err(|_| Error::Invalid("a trajectory is larger than memory".into()))?;
        if self
            .crossing
            .as_ref()
            .is_none_or(|buffer| buffer.len() < bytes)
        {
            self.crossing = Some(platform::PinnedBuffer::new(self.device, bytes)?);
        }
        let buffer = self
            .crossing
            .as_ref()
            .ok_or_else(|| Error::Invariant("the trajectory scratch was just allocated".into()))?;
        Ok(TensorView::new(
            ops::DevicePtr::new(buffer.address()),
            published.dtype,
            Shape::new(&[published.bytes / element]),
        ))
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
    fn latents(&self, layout: &Layout) -> TensorView {
        let video = layout.video;
        TensorView::new(
            self.columns.latents.device.ptr,
            self.dtype,
            Shape::new(&[
                u64::from(video.frames),
                u64::from(video.height),
                u64::from(video.width),
                u64::from(video.channels),
            ]),
        )
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
                u64::from(layout.video.channels),
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
            Product::Video,
            self.device,
            self.stream_id(),
            Some(written),
            &published,
        )?;
        // The column now holds this request's clip until a reader forgets it: that is what a second
        // decode of another request is refused against.
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
        let reserved = self.pool()?.reserved();
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
        let latents = self.latents(&layout);
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
        self.splice_clip(&forward, bucket, &layout)?;
        if first {
            self.pool()?
                .record(self.stream_id(), bucket, arena.used())?;
        }
        Ok(())
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
        let index = self.spliced.get(&request).copied().unwrap_or(0);
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
        let channels = layout.video.channels;
        self.size_tail(channels, height, width, splice.overlap)?;
        let previous = match self.tail.as_ref() {
            Some(tail) if splice.overlap > 0 => Some(TensorView::new(
                tail.device.ptr,
                self.dtype,
                Shape::new(&[
                    u64::from(channels),
                    u64::from(splice.overlap),
                    u64::from(height),
                    u64::from(width),
                ]),
            )),
            _ => None,
        };
        let clip = self.raster(bucket, layout)?;
        let column = self.frames.as_ref().ok_or_else(|| {
            Error::Invariant(
                "a postprocessor is bound and no frames column was sized for the bucket".into(),
            )
        })?;
        let video = TensorView::new(
            column.device.ptr,
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
        self.keep_tail(&clip, channels, height, width, splice.overlap)?;
        // The timeline's last clip has no successor, so the request's place in it ends here: a
        // decode after it would be a clip the timeline does not hold, which the schedule refuses
        // above by naming the count.
        if index + 1 == schedule.len() as u32 {
            self.spliced.remove(&request);
        } else {
            self.spliced.insert(request, index + 1);
        }
        self.staged_keep = Some(splice.keep);
        Ok(())
    }

    /// Sizes the tail column for an overlap of `overlap` frames, the first time one is needed.
    ///
    /// The count is the schedule's, not this runner's: a timeline whose fade reads five frames and one
    /// that reads ten are one formula apart in the block and one column here.
    fn size_tail(&mut self, channels: u32, height: u32, width: u32, overlap: u32) -> Result<()> {
        if overlap == 0 {
            return Ok(());
        }
        let elements = u64::from(channels)
            .checked_mul(u64::from(overlap))
            .and_then(|elements| elements.checked_mul(u64::from(height)))
            .and_then(|elements| elements.checked_mul(u64::from(width)))
            .ok_or_else(|| Error::Invalid("a clip's tail extent overflows".into()))?;
        if self
            .tail
            .as_ref()
            .is_none_or(|column| column.device.numel() < elements)
        {
            self.tail = Some(Column::new(self.device, self.dtype, elements)?);
        }
        Ok(())
    }

    /// Keeps the clip's last `overlap` frames for the next clip's fade.
    ///
    /// The tail is one contiguous run per channel of the clip's own raster (`[channels, frames, rows,
    /// columns]`), so this is `channels` copies on the runner's stream and no arithmetic. It has to
    /// happen before the next decode, which writes over the clip's column.
    fn keep_tail(
        &self,
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
        let element = self
            .dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{} has no whole-byte size", self.dtype)))?;
        let frames = clip.shape.dim(1);
        let start = frames
            .checked_sub(u64::from(overlap))
            .ok_or_else(|| Error::Invalid("a clip is shorter than its own tail".into()))?;
        let bytes = usize::try_from(
            u64::from(overlap)
                .saturating_mul(u64::from(height))
                .saturating_mul(u64::from(width))
                .saturating_mul(element),
        )
        .map_err(|_| Error::Invalid("a clip's tail is larger than memory".into()))?;
        let stride = u64::from(overlap)
            .saturating_mul(u64::from(height))
            .saturating_mul(u64::from(width));
        for channel in 0..u64::from(channels) {
            let source = clip.element(&[channel, start, 0, 0]);
            let destination = tail.device.element(&[channel * stride]);
            // SAFETY: both regions are this runner's own buffers, the source's run of `bytes` lies
            // inside the clip's raster (its frames axis is at least `overlap` long) and the
            // destination's inside the tail column, sized for exactly this overlap; the copy is
            // ordered on the runner's stream before the next clip's splice reads it.
            unsafe {
                platform::memcpy_async(
                    destination.address(),
                    source.address(),
                    bytes,
                    &self.stream,
                )?;
            }
        }
        Ok(())
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
            column.device.ptr,
            DType::U8,
            Shape::new(&[u64::from(keep), u64::from(height), u64::from(width), 3_u64]),
        ))
    }
}

impl<M: Decoder + Send + Sync> Runner for DecoderRunner<M> {
    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
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
            if self.lifecycle.graphs.captured(bucket, generation).is_none() {
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

    fn attach_pools(&mut self, pools: Arc<Mutex<Pools>>) {
        self.tables.attach(Arc::clone(&pools));
        self.lifecycle.attach_pools(pools);
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
                self.upload()?;
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
                // The clip is written by the graph that just launched, and the stream's order is
                // what makes the view readable: a reader's copy is issued after this one.
                self.publish(*bucket)?;
                // The clip's raster is this runner's own column, written by the graph that just
                // launched: the stream's order is what makes it readable, and the completion that
                // turns it into an artifact is the transport's rather than this step's.
                Ok(BatchOutput {
                    calls: Vec::new(),
                    timing: ipc::TimingCounters {
                        device_us: launched.elapsed().as_micros() as u64,
                        ..Default::default()
                    },
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
        let walked = walk_capture(&stream, || self.walk(bucket, &kernels))?;
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
