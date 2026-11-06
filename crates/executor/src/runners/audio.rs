//! The audio decoder's runner: one track's latents into interleaved PCM, on one device stream.
//!
//! **Why this is a runner of its own and not a branch of the clip decoder's.** The two serve one
//! component of a deployment and are still two graphs: a clip's decode is a stack of
//! three-dimensional convolutions over a voxel grid whose subject is a raster, and a track's is a
//! vocoder's transposed and dilated one-dimensional convolutions whose subject is a waveform. Their
//! buckets differ by variant (`Variant::Decode` against `Variant::AudioDecode`), their extents by
//! kind (`Extent::Clip` against `Extent::Latents`), their columns by shape, and their products by
//! name (`Video` against `Audio`). Sharing one struct would have made every one of those five
//! differences an `if` in a function whose two branches share no line.
//!
//! **What it reads and writes.** The denoiser publishes the request's audio latent trajectory under
//! `Product::AudioLatents` at every step, so a decode copies that region into this runner's latents
//! column and publishes the PCM window it wrote under `Product::Track`: `Audio` is a
//! reference audio condition, and the decoded track is what an audio encode then reads. Both crossings go through
//! the store, which is the tree's one way for one runner to hand another a device region, and the
//! copy is issued on this runner's stream so it is ordered after the producing step's write.
//!
//! **Why the PCM window is a column and not an arena extent.** An arena is a graph's scratch and is
//! reused by the next capture; a track a request asked for is read by the transport after the step,
//! so it must outlive the replay, exactly as a clip does.

use std::sync::{Arc, Mutex};

use nn::capability::{DecodedMedia, Decoder, LatentShape, Layout};
use ops::{DType, Shape, TensorView};
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{Batch, BatchOutput, CallId, Job, Jobs, answered};
use crate::plan::{Bucket, Buckets, Extent, Variant};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::Column;
use crate::runners::lifecycle::{Lifecycle, walk_capture, warmed_capture};
use crate::runners::products::{Product, Products};
use crate::{Error, Result};

/// The two columns one track's decode is staged into, sized for the largest track this runner serves.
struct AudioColumns {
    /// The track's latents, `[frames, channels]`, in the decoder's own element type.
    latents: Column,
    /// The PCM window the decode writes, `[samples, channels]`, `i16` interleaved - the type
    /// `DecodedMedia::audio` states, which the model writes and no runner converts.
    pcm: Column,
}

/// The media computation a batch states, or the refusal this runner gives a batch with none.
fn media_of(batch: &Batch) -> Result<&crate::batch::MediaColumns> {
    batch.media.as_ref().ok_or_else(|| {
        Error::Invalid(
            "an audio decoder runner serves a media batch; this one states no computation".into(),
        )
    })
}

/// The audio decoder's runner: one device stream, the tracks its model serves, and their graphs.
pub struct AudioDecoderRunner<M: Decoder + Send + Sync> {
    id: RunnerId,
    phases: Vec<ipc::PhaseId>,
    model: Arc<M>,
    lifecycle: Lifecycle,
    buckets: Buckets,
    device: DeviceId,
    stream: platform::Stream,
    pools: Option<Arc<Mutex<Pools>>>,
    counters: Option<Arc<crate::device::SharedCounters>>,
    /// The layout of each audio bucket, which is the bucket's own extent: a track's decode is keyed
    /// by the layout the denoiser produced, not by a raster a caller computed from one.
    layouts: std::collections::BTreeMap<Bucket, Layout>,
    /// The element type the latents are stated in, as the model declares it.
    dtype: DType,
    columns: AudioColumns,
    /// The bucket the columns are staged for, so a capture knows what it walked.
    /// The calls of the batch staged last, which its replay answers.
    calls: Vec<CallId>,
    staged: Option<Bucket>,
    /// The epoch's product store, or `None` when the executor attached none. A decode then stages no
    /// latents and publishes nothing, and a reader refuses by name rather than reading a column's
    /// previous contents.
    products: Option<Arc<dyn Products>>,
    /// The request whose track is staged, which is the name a replay publishes under.
    staged_request: Option<ipc::RequestKey>,
    /// The request whose track the PCM column last held, so a decode of another can be refused while
    /// the first is unread.
    published_for: Option<ipc::RequestKey>,
}

impl<M: Decoder + Send + Sync> AudioDecoderRunner<M> {
    /// Binds an audio decoder to one device stream.
    ///
    /// Every audio bucket's layout is taken from the bucket's own extent, so a plan whose bucket
    /// states a layout with no audio track is refused here, at load, rather than at the first
    /// request that would have decoded silence.
    pub fn new(
        binding: Binding,
        model: Arc<M>,
        buckets: Buckets,
        generation: Arc<ops::Generation>,
    ) -> Result<Self> {
        let mut layouts = std::collections::BTreeMap::new();
        let mut largest: Option<(Bucket, LatentShape)> = None;
        for bucket in buckets.iter().copied() {
            if bucket.variant != Variant::AudioDecode {
                continue;
            }
            let layout = Self::layout_of(bucket)?;
            let audio = layout.audio.ok_or_else(|| {
                Error::Invalid(format!(
                    "bucket {bucket} is an audio decode and its layout states no audio track: a \
                     track's decode is keyed by the layout the denoiser produced, and a layout \
                     with no audio has none to decode"
                ))
            })?;
            if largest
                .as_ref()
                .is_none_or(|(_, held)| audio.frames > held.frames)
            {
                largest = Some((bucket, audio));
            }
            layouts.insert(bucket, layout);
        }
        let (_, audio) = largest.ok_or_else(|| {
            Error::Invalid(
                "an audio decoder runner was bound with no audio decode bucket to capture; its plan \
                 must state the tracks the checkpoint's decoder serves"
                    .into(),
            )
        })?;
        let dtype = model.element_type();
        let latents = Self::latent_elements(audio)?;
        let pcm = Self::pcm_elements(&*model, audio)?;
        let columns = AudioColumns {
            latents: Column::new(binding.device, dtype, latents)?,
            pcm: Column::new(binding.device, pcm.0, pcm.1)?,
        };
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
            layouts,
            dtype,
            columns,
            calls: Vec::new(),
            staged: None,
            products: None,
            staged_request: None,
            published_for: None,
        })
    }

    /// One audio bucket's latent layout, which is the extent the bucket states.
    fn layout_of(bucket: Bucket) -> Result<Layout> {
        let Extent::Latents(layout) = bucket.extent else {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is not an audio decode bucket and an audio runner was asked for \
                 its layout"
            )));
        };
        Ok(layout)
    }

    /// The elements of a track's latent column: frames times channels.
    fn latent_elements(audio: LatentShape) -> Result<u64> {
        u64::from(audio.frames)
            .checked_mul(u64::from(audio.channels))
            .ok_or_else(|| Error::Invalid("an audio track's latent extent overflows".into()))
    }

    /// The audio column's dtype and element count, from the model's own answer for the window.
    ///
    /// The window's length is the model's to state: it is the product of the vocoder's upsampling
    /// rates, which this crate cannot derive and would be a second statement of the checkpoint's if
    /// it tried. The dtype is `i16` because `DecodedMedia::audio` states interleaved `i16` PCM, and
    /// a column of another width is a waveform the muxer cannot read.
    fn pcm_elements(model: &M, audio: LatentShape) -> Result<(nn::DType, u64)> {
        let (samples, channels) = model.pcm_window(&audio).map_err(Error::from)?;
        let elements = samples
            .checked_mul(u64::from(channels))
            .ok_or_else(|| Error::Invalid("an audio track's PCM window overflows".into()))?;
        Ok((DType::I16, elements))
    }

    /// The audio buckets this runner captures, largest track first.
    fn capture_order(&self) -> Vec<Bucket> {
        let mut buckets: Vec<Bucket> = self
            .buckets
            .iter()
            .filter(|bucket| bucket.variant == Variant::AudioDecode)
            .copied()
            .collect();
        buckets.sort_by_key(|bucket| std::cmp::Reverse(bucket.rows));
        buckets
    }

    fn layout(&self, bucket: Bucket) -> Result<Layout> {
        self.layouts
            .get(&bucket)
            .copied()
            .ok_or_else(|| Error::Invalid(format!("no layout was mapped for {bucket}")))
    }

    /// The track `bucket` decodes.
    fn audio(&self, bucket: Bucket) -> Result<LatentShape> {
        self.layout(bucket)?.audio.ok_or_else(|| {
            Error::Invariant(format!(
                "bucket {bucket} states no audio track and this runner is asked for its shape"
            ))
        })
    }

    fn stream_id(&self) -> StreamId {
        StreamId(self.stream.raw())
    }

    fn pool(&self) -> Result<std::sync::MutexGuard<'_, Pools>> {
        let pools = self.pools.as_ref().ok_or_else(|| {
            Error::Resource(
                "the audio decoder runner has no graph pools; the device attaches them at start",
            )
        })?;
        Ok(pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// Checks every bucket's plan against the device's share, once, when the runner loads.
    ///
    /// The same rule as the clip decoder's, through the same `floor::checked`, so the two media
    /// runners cannot read an absent plan and a refused one differently.
    pub fn load(&mut self) -> Result<()> {
        if self.pools.is_none() {
            return Err(Error::Resource(
                "the audio decoder runner loaded before its device's pools were attached",
            ));
        }
        let offered = self.pool()?.reserved();
        for bucket in self.capture_order() {
            let layout = self.layout(bucket)?;
            super::floor::checked(bucket, self.model.arena_bytes(&layout), offered)?;
        }
        Ok(())
    }

    /// The track's latents, `[frames, channels]`, as the model reads them.
    fn latents(&self, audio: LatentShape) -> TensorView {
        TensorView::new(
            self.columns.latents.device.ptr,
            self.dtype,
            Shape::new(&[u64::from(audio.frames), u64::from(audio.channels)]),
        )
    }

    /// The PCM window the replay writes, `[samples, channels]`.
    fn pcm(&self, audio: LatentShape) -> Result<TensorView> {
        let (samples, channels) = self.model.pcm_window(&audio).map_err(Error::from)?;
        Ok(TensorView::new(
            self.columns.pcm.device.ptr,
            DType::I16,
            Shape::new(&[samples, u64::from(channels)]),
        ))
    }

    /// Uploads the staged columns, in one place, so a capture and a replay agree on what the graph
    /// reads.
    fn upload(&self) -> Result<()> {
        self.columns.latents.upload(&self.stream)
    }

    /// Stages a batch's track into the columns `bucket`'s graph reads.
    fn stage(&mut self, batch: &Batch, bucket: Bucket) -> Result<()> {
        let media = media_of(batch)?;
        let request = media.request(0)?.request;
        // One PCM column, and a track nobody has read yet: a decode of another request would
        // overwrite it before its artifact exists. The first track's life ends at
        // `Products::forget`, which the transport calls once it has the bytes.
        if let (Some(store), Some(published)) = (self.products.as_deref(), self.published_for)
            && published != request
            && store.holds(published, Product::Track)
        {
            return Err(Error::Invalid(format!(
                "request {} asks for an audio decode and this runner's PCM column still holds \
                 request {}'s track: a decode writes the one column, so the second track would \
                 replace the first before its artifact was read",
                request.request_id.0, published.request_id.0
            )));
        }
        let audio = self.audio(bucket)?;
        let latents = self.latents(audio);
        // The track is the denoiser's product: a denoise step publishes it under this request's name
        // at every step, and the copy below is ordered after that write by the store.
        let store = self.products.clone().ok_or_else(|| {
            Error::Invalid(
                "an audio decode reads the request's latent trajectory and no product store is \
                 attached: the track would be decoded from whatever the column last held"
                    .into(),
            )
        })?;
        if !store.copy_into(request, Product::AudioLatents, &latents, &self.stream)? {
            return Err(crate::runners::products::absent(
                request,
                Product::AudioLatents,
            ));
        }
        self.staged = Some(bucket);
        self.staged_request = Some(request);
        Ok(())
    }

    /// Stages `bucket` with no batch behind it, so a first capture has a defined column to read.
    fn stage_warm(&mut self, bucket: Bucket) -> Result<()> {
        if self.staged == Some(bucket) {
            return Ok(());
        }
        // Zeros are defined, hold no track's arithmetic and cost no copy.
        //
        // SAFETY: the column is this runner's own pinned storage, sized for the largest bucket it
        // was bound with, and nothing else writes it here: the capture this stages is the next job
        // on the stream, and a replay of this bucket is not concurrent with a stage.
        unsafe {
            self.columns.latents.host.as_mut_slice().fill(0);
        }
        self.staged = Some(bucket);
        Ok(())
    }

    /// Publishes the track a replay just wrote, so the transport can read it by the request's name.
    fn publish(&mut self, bucket: Bucket) -> Result<()> {
        let Some(products) = self.products.as_ref() else {
            return Ok(());
        };
        let Some(request) = self.staged_request else {
            return Err(Error::Invariant(format!(
                "a replay of bucket {bucket} publishes a track and no request is staged"
            )));
        };
        let audio = self.audio(bucket)?;
        let pcm = self.pcm(audio)?;
        let written = crate::runners::products::written_on(&self.stream)?;
        products.publish(
            request,
            Product::Track,
            self.device,
            self.stream_id(),
            Some(written),
            &pcm,
        )?;
        self.published_for = Some(request);
        Ok(())
    }

    /// Walks the decode once for `bucket`, recording every operator launch into the generation.
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
        let audio = self.audio(bucket)?;
        let latents = self.latents(audio);
        let pcm = self.pcm(audio)?;
        self.model
            .decode(
                &forward,
                latents,
                // Every table this decoder reads is a weight, bound at load: the capability states
                // that an audio decode is given none, so the runner passes none.
                &[],
                DecodedMedia {
                    video: None,
                    audio: Some(pcm),
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

impl<M: Decoder + Send + Sync> Runner for AudioDecoderRunner<M> {
    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    fn stream(&self) -> Option<&platform::Stream> {
        Some(&self.stream)
    }

    /// Serves one track: choose its bucket, stage its latents, and return the replay.
    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        let media = media_of(&batch)?.clone();
        media.check()?;
        let computation = media.computation;
        if computation.variant() != Some(Variant::AudioDecode) {
            return Err(Error::Invalid(format!(
                "an audio decoder runner serves an audio decode; this batch states a {:?}",
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
        // A decoder reads no paged cache: its latents arrive as the product the denoiser wrote.
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
    }

    fn discard(&mut self, generation: u64) -> Result<()> {
        self.lifecycle.discard(generation)
    }

    fn attach_products(&mut self, products: Arc<dyn Products>) {
        self.products = Some(products);
    }

    fn attach_pools(&mut self, pools: Arc<Mutex<Pools>>) {
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
                // The track is written by the graph that just launched, and the stream's order is
                // what makes the view readable: a reader's copy is issued after this one.
                self.publish(*bucket)?;
                Ok(BatchOutput {
                    calls: answered(&mut self.calls),
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
                "an audio decoder runner compares no operator, and the device asked it to shadow \
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

impl<M: Decoder + Send + Sync> AudioDecoderRunner<M> {
    /// Captures one bucket of `generation`: the walk here, the graph kept by the lifecycle.
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = warmed_capture(&stream, || self.walk(bucket, &kernels))?;
        self.lifecycle.capture(bucket, generation, walked, &kernels)
    }

    /// Prepares one bucket of the next generation from a walk of the same bucket.
    fn rebind(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = walk_capture(&stream, || self.walk(bucket, &kernels))?;
        self.lifecycle.capture(bucket, generation, walked, &kernels)
    }
}
