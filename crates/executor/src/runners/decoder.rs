//! The decoder runner: one clip's latents, walked once per canvas and replayed (4.5).
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
    /// One column per table the largest layout declares, in `layout_tables` order.
    ///
    /// Each column takes its element type and length from the table it stages, because a decoder's
    /// tables are not one type: the rotary tables and the copy offsets are the shapes the model
    /// states and this file only copies them.
    state: Vec<Column>,
}

impl DecoderColumns {
    fn new(
        device: DeviceId,
        layout: &Layout,
        dtype: DType,
        tables: &[nn::capability::StateTable],
    ) -> Result<Self> {
        let video = layout.video;
        let voxels = u64::from(video.frames) * u64::from(video.height) * u64::from(video.width);
        let elements = voxels
            .checked_mul(u64::from(video.channels))
            .ok_or_else(|| Error::Invalid("a clip's latent extent overflows".into()))?;
        let latents = Column::new(device, dtype, elements)?;
        let mut state = Vec::with_capacity(tables.len());
        for table in tables {
            let numel = table.shape.numel();
            state.push(Column::new(device, table.dtype, numel)?);
        }
        Ok(Self {
            latents,
            // Sized by `size_raster` when a bucket is first staged: the raster's frame count is the
            // bucket's rather than the layout's, since a layout states **latent** frames and the
            // clip writes the decoded ones. One element until then, because a column of a length
            // this constructor cannot know must not pretend to a length it can.
            raster: Column::new(device, dtype, 1)?,
            state,
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
        let tables = model.layout_tables(layout).map_err(Error::from)?;
        let columns = DecoderColumns::new(binding.device, layout, dtype, &tables)?;
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

    /// One clip's buffers, refused unless their count is the model's: one latents column and one per
    /// table, in [`Decoder::layout_tables`]' order.
    ///
    /// The count is the model's statement, not the runner's - a decode whose batch carried two
    /// latents columns, or one table short, would write a column from another's bytes and read the
    /// missing table as whatever the previous clip left there.
    fn buffers_of<'a>(buffers: &'a Vec<Vec<u8>>, tables: usize) -> Result<&'a [Vec<u8>]> {
        let expected = tables + 1;
        if buffers.len() != expected {
            return Err(Error::Invalid(format!(
                "the decoder states {expected} buffer(s) for a clip - one latents and {tables} \
                 table(s) - and the batch carries {}",
                buffers.len()
            )));
        }
        Ok(buffers)
    }

    /// Stages a batch's buffers into the columns `bucket`'s graph reads.
    ///
    /// The buffers are the model's own: `prepare_layout` states the first as the clip's latents and
    /// the rest as the tables `layout_tables` names, in order. This function checks their count and
    /// their byte lengths against what the model declared and copies them; it reads no header and
    /// interprets no value, which is what keeps a decoder's input format out of the executor.
    fn stage(&mut self, batch: &Batch, bucket: Bucket) -> Result<()> {
        let media = batch.media.as_ref().ok_or_else(|| {
            Error::Invalid(
                "a decoder runner serves a media batch; this one states no computation".into(),
            )
        })?;
        self.size_raster(bucket)?;
        let layout = self.layout(bucket)?;
        let tables = self.model.layout_tables(&layout).map_err(Error::from)?;
        // The batch names its requests, and a decode bucket serves one clip, so the name kept here
        // is the one the replay's publish uses.
        self.staged_request = Some(media.request(0)?.request);
        let buffers = Self::buffers_of(Self::clip_of(media)?, tables.len())?;
        self.columns.latents.write(&buffers[0])?;
        for (column, bytes) in self.columns.state.iter_mut().zip(&buffers[1..]) {
            column.write(bytes)?;
        }
        self.staged = Some(bucket);
        Ok(())
    }

    /// Stages `bucket` with no batch behind it, so a first capture has defined columns to read.
    fn stage_warm(&mut self, bucket: Bucket) -> Result<()> {
        if self.staged == Some(bucket) {
            return Ok(());
        }
        self.size_raster(bucket)?;
        let layout = self.layout(bucket)?;
        let tables = self.model.layout_tables(&layout).map_err(Error::from)?;
        let mut host = vec![Vec::new(); tables.len() + 1];
        self.model
            .prepare_layout(&layout, &mut host)
            .map_err(Error::from)?;
        self.columns.latents.write(&host[0])?;
        for (column, bytes) in self.columns.state.iter_mut().zip(&host[1..]) {
            column.write(bytes)?;
        }
        self.staged = Some(bucket);
        Ok(())
    }

    /// Uploads the staged columns, in one place, so a capture and a replay agree on what the graph
    /// reads.
    fn upload(&self) -> Result<()> {
        self.columns.latents.upload(&self.stream)?;
        for column in &self.columns.state {
            column.upload(&self.stream)?;
        }
        Ok(())
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
    fn publish(&self, bucket: Bucket) -> Result<()> {
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
        let raster = self.raster(bucket, &layout)?;
        products.publish(
            request,
            Product::Video,
            self.device,
            self.stream_id(),
            &raster,
        )
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
        if first {
            self.pool()?
                .record(self.stream_id(), bucket, arena.used())?;
        }
        Ok(())
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
    fn a_clip_carries_one_latents_buffer_and_one_per_table() {
        // The count is the model's: the stub declares two tables, so a clip is three buffers, and a
        // batch carrying one fewer would read the missing table as whatever the last clip left.
        let buffers = vec![vec![0_u8; 8], vec![0_u8; 4], vec![0_u8; 4]];
        assert_eq!(
            DecoderRunner::<Fixed>::buffers_of(&buffers, 2)
                .expect("one latents and two tables")
                .len(),
            3
        );

        let short = vec![vec![0_u8; 8], vec![0_u8; 4]];
        let error =
            DecoderRunner::<Fixed>::buffers_of(&short, 2).expect_err("one table is missing");
        let message = format!("{error}");
        assert!(
            message.contains("states 3 buffer(s) for a clip")
                && message.contains("the batch carries 2"),
            "the refusal names both counts: {message}"
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
