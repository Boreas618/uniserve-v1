//! Video model components beside the denoiser: media reconstruction (video
//! units with their RGB post-processing, and audio units), condition latent
//! encoding, and prompt and condition media encoding.
//!
//! A video decoding round hands each rank of the decoder's unit ring one
//! media unit: the rank unpacks the unit's window of the request's complete
//! packed latent into the bound window buffer, decodes it at its segment
//! (replaying the segment's captured graph when there is one) and
//! cross-fades and quantizes it into the unit's row of the round's product,
//! exchanging decoded overlaps with its ring neighbours. The request's
//! retained overlap lives in its slot's row of the overlap bank. Audio
//! rounds decode a contiguous run of audio units into their sample span.

use std::sync::Arc;

use tvm_ffi::{Any, AnyView};
use uniserve_core::Canvas;
use uniserve_cuda::Stream;

use crate::ffi::{Backing, DType, Fields, Function, Functions, Handle};
use crate::model::bind::{Backends, BindingRequest, ComponentBinding, StreamWorkspaces, TextSize};
use crate::runner::GraphPool;
use crate::runner::encoder::{EncoderConfig, EncoderInfo, EncoderRunner};
use crate::runner::graphs::{GraphMode, GraphTable, Lookup};
use crate::storage::memory::{Memory, StorageView, parse_dtype};
use crate::storage::{RequestSlots, SlotField};
use crate::{Error, Result};

/// What a decoder binding needs besides its buffers.
pub(crate) struct BindingSetup<'a> {
    pub(crate) bind: &'a Function,
    pub(crate) loaded: &'a Handle,
    pub(crate) component: &'a str,
    pub(crate) stream: Stream,
    pub(crate) device: i32,
    pub(crate) attention: String,
    pub(crate) allocate: tvm_ffi::Function,
}

impl BindingSetup<'_> {
    fn request(&self) -> Fields {
        Fields::new()
            .text("component", self.component)
            .with("stream", self.stream.raw() as i64)
            .text("attention", &self.attention)
            .with("allocate", self.allocate.clone())
    }
}

/// A segment: the native frames and raster one unit decodes to.
pub(crate) type Segment = (u32, Canvas);

fn segment_arguments(segment: Segment) -> [Any; 3] {
    [
        Any::from(i64::from(segment.0)),
        Any::from(i64::from(segment.1.width)),
        Any::from(i64::from(segment.1.height)),
    ]
}

/// One rank's video decoder and RGB post-processor.
pub(crate) struct VideoDecoding {
    prepare_segment: Function,
    prepare_canvas: Function,
    warm_segment: Function,
    warm_unit: Function,
    capture_segment: Function,
    decode_unit: Function,
    kernels: Function,
    close: Function,
    stream: Stream,
    /// Buffers the binding computes in, retained for its lifetime: the flat
    /// decoder input of the largest segment, the per-slot retained overlaps
    /// (`[slots + 1, elements]`) and the flat overlap the unit ring
    /// delivers. Only the binding reads and writes them.
    _window: StorageView,
    _overlap: RequestSlots,
    _exchange: StorageView,
    graphs: GraphTable<Segment>,
    pool: Option<Arc<GraphPool<Handle>>>,
}

impl VideoDecoding {
    /// Allocate the decoder's buffers and bind them. `window` and `overlap`
    /// are the largest segment input and canvas overlap as `(elements,
    /// dtype)`.
    pub(crate) fn new(
        setup: BindingSetup<'_>,
        window: (u64, DType),
        overlap: (u64, DType),
        slots: u32,
        pool: Option<Arc<GraphPool<Handle>>>,
    ) -> Result<Self> {
        let flat = |(elements, dtype): (u64, DType)| -> Result<StorageView> {
            let memory = Memory::device(
                setup.device,
                (elements * dtype.bytes() as u64).max(1) as usize,
            )?;
            StorageView::contiguous(memory as Arc<dyn Backing>, 0, dtype, &[elements as i64])
        };
        let window = flat(window)?;
        let exchange = flat(overlap)?;
        let overlap = RequestSlots::allocate(
            &[SlotField {
                name: "video_overlap".into(),
                dtype: overlap.1,
                shape: vec![overlap.0 as i64],
                host: false,
            }],
            slots,
            Some(setup.device),
        )?;
        let request = setup
            .request()
            .with("window", window.to_tensor()?)
            .with("overlap", overlap.bank("video_overlap")?.to_tensor()?)
            .with("exchange", exchange.to_tensor()?)
            .into_any();
        let functions = Functions::new(
            "video.bind_decoder",
            setup
                .bind
                .call(&[setup.loaded.view(), AnyView::from(&request)])?,
        )?;
        let release = functions.get("release")?;
        Ok(Self {
            prepare_segment: functions.get("prepare_segment")?,
            prepare_canvas: functions.get("prepare_canvas")?,
            warm_segment: functions.get("warm_segment")?,
            warm_unit: functions.get("warm_unit")?,
            capture_segment: functions.get("capture_segment")?,
            decode_unit: functions.get("decode_unit")?,
            kernels: functions.get("kernels")?,
            close: functions.get("close")?,
            stream: setup.stream,
            _window: window,
            _overlap: overlap,
            _exchange: exchange,
            graphs: GraphTable::new(format!("{}.decoder", setup.component), release),
            pool,
        })
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Prepare every segment (largest first) and canvas before any capture.
    pub(crate) fn prepare(&self, segments: &[Segment], canvases: &[Canvas]) -> Result<()> {
        for &segment in segments {
            self.prepare_segment
                .call_values(&segment_arguments(segment))?;
        }
        for canvas in canvases {
            self.prepare_canvas.call_values(&[
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
            ])?;
        }
        Ok(())
    }

    /// Decode a zero window of every segment eagerly, before any capture.
    pub(crate) fn warm(&self, segments: &[Segment]) -> Result<()> {
        for &segment in segments {
            self.warm_segment.call_values(&segment_arguments(segment))?;
        }
        Ok(())
    }

    /// Reconstruct a zero unit `unit` of a round of `count` ring members of
    /// a `frames`-frame video at `canvas`. Every member holding a unit of
    /// the round calls it, in the same order.
    pub(crate) fn warm_unit(
        &self,
        frames: u32,
        canvas: Canvas,
        unit: u32,
        count: u32,
    ) -> Result<()> {
        self.warm_unit.call_values(&[
            Any::from(i64::from(frames)),
            Any::from(i64::from(canvas.width)),
            Any::from(i64::from(canvas.height)),
            Any::from(i64::from(unit)),
            Any::from(i64::from(count)),
        ])?;
        Ok(())
    }

    /// Capture each segment's decode into the decoder's graph pool.
    pub(crate) fn capture(&mut self, segments: &[Segment]) -> Result<()> {
        let Some(pool) = &self.pool else {
            return Ok(());
        };
        for &segment in segments {
            if self.graphs.contains(&segment, None) {
                continue;
            }
            let mut arguments = segment_arguments(segment).to_vec();
            arguments.push(pool.value.to_any());
            let graph = Handle::new(self.capture_segment.call_values(&arguments)?)?;
            self.graphs.insert(segment, None, graph)?;
            self.graphs.record(GraphMode::Capture);
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Decode and reconstruct one media unit of a `frames`-frame video at
    /// `canvas` into `out` (`[longest, height, width, 3]` uint8): its
    /// leading frames receive the unit's RGB. `unit_count` ring members
    /// hold a unit of this round. Returns the unit's frames.
    #[expect(
        clippy::too_many_arguments,
        reason = "one unit's complete reconstruction inputs"
    )]
    pub(crate) fn decode(
        &mut self,
        slot: usize,
        latent: &StorageView,
        frames: u32,
        canvas: Canvas,
        segment: Segment,
        unit: u32,
        unit_count: u32,
        out: &StorageView,
    ) -> Result<(u32, GraphMode)> {
        let graph = match self.graphs.lookup(&segment, None) {
            Ok(Lookup::Replay(graph)) => Some(graph.clone()),
            _ => None,
        };
        let mode = if graph.is_some() {
            GraphMode::Replay
        } else {
            GraphMode::Eager
        };
        let graph = graph.map_or_else(|| Any::from(()), |graph| graph.to_any());
        let written: i64 = self.decode_unit.call_as(&[
            AnyView::from(&Any::from(slot as i64)),
            AnyView::from(&Any::from(latent.to_tensor()?)),
            AnyView::from(&Any::from(i64::from(frames))),
            AnyView::from(&Any::from(i64::from(canvas.width))),
            AnyView::from(&Any::from(i64::from(canvas.height))),
            AnyView::from(&Any::from(i64::from(unit))),
            AnyView::from(&Any::from(i64::from(unit_count))),
            AnyView::from(&Any::from(out.to_tensor()?)),
            AnyView::from(&graph),
        ])?;
        self.graphs.record(mode);
        let written = u32::try_from(written)
            .map_err(|_| Error::Interface("decode_unit returns its frames".into()))?;
        Ok((written, mode))
    }

    pub(crate) fn seal(&mut self) {
        self.graphs.seal();
    }

    pub(crate) fn kernels(&self) -> &Function {
        &self.kernels
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        let closed = self.close.invoke(&[AnyView::from(&Any::from(aborted))]);
        graphs.and(closed)
    }
}

/// One rank's audio decoder.
pub(crate) struct AudioDecoding {
    prepare: Function,
    warm: Function,
    decode: Function,
    kernels: Function,
    close: Function,
    stream: Stream,
}

impl AudioDecoding {
    pub(crate) fn new(setup: BindingSetup<'_>) -> Result<Self> {
        let request = setup.request().into_any();
        let functions = Functions::new(
            "video.bind_audio_decoder",
            setup
                .bind
                .call(&[setup.loaded.view(), AnyView::from(&request)])?,
        )?;
        Ok(Self {
            prepare: functions.get("prepare")?,
            warm: functions.get("warm")?,
            decode: functions.get("decode")?,
            kernels: functions.get("kernels")?,
            close: functions.get("close")?,
            stream: setup.stream,
        })
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Prepare decoding of tracks of `latent_frames` latent frames.
    pub(crate) fn prepare(&self, latent_frames: u32) -> Result<()> {
        self.prepare
            .invoke(&[AnyView::from(&Any::from(i64::from(latent_frames)))])
    }

    /// Decode units `[first, first + count)` of `total` of a zero track of
    /// `samples` samples whose complete latent has `shape`.
    pub(crate) fn warm(
        &self,
        samples: u64,
        first: u32,
        count: u32,
        total: u32,
        shape: &[u64],
    ) -> Result<()> {
        self.warm.call_values(&[
            Any::from(samples as i64),
            Any::from(i64::from(first)),
            Any::from(i64::from(count)),
            Any::from(i64::from(total)),
            Any::from(crate::ffi::integer_list(
                shape.iter().map(|&extent| extent as i64),
            )),
        ])?;
        Ok(())
    }

    /// Decode audio units `[first, first + count)` of `total` of a
    /// `samples`-sample track into `out` (`[samples, 2]` int16).
    pub(crate) fn decode(
        &self,
        latent: &StorageView,
        samples: u64,
        first: u32,
        count: u32,
        total: u32,
        out: &StorageView,
    ) -> Result<u64> {
        let written: i64 = self.decode.call_as(&[
            AnyView::from(&Any::from(latent.to_tensor()?)),
            AnyView::from(&Any::from(samples as i64)),
            AnyView::from(&Any::from(i64::from(first))),
            AnyView::from(&Any::from(i64::from(count))),
            AnyView::from(&Any::from(i64::from(total))),
            AnyView::from(&Any::from(out.to_tensor()?)),
        ])?;
        u64::try_from(written)
            .map_err(|_| Error::Interface("audio decode returns its samples".into()))
    }

    pub(crate) fn kernels(&self) -> &Function {
        &self.kernels
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        self.close.invoke(&[AnyView::from(&Any::from(aborted))])
    }
}

/// One rank's condition latent encoders.
pub(crate) struct LatentEncoding {
    encode_video: Function,
    encode_audio: Function,
    kernels: Function,
    close: Function,
    stream: Stream,
}

impl LatentEncoding {
    pub(crate) fn new(setup: BindingSetup<'_>) -> Result<Self> {
        let request = setup.request().into_any();
        let functions = Functions::new(
            "video.bind_latent_encoder",
            setup
                .bind
                .call(&[setup.loaded.view(), AnyView::from(&request)])?,
        )?;
        Ok(Self {
            encode_video: functions.get("encode_video")?,
            encode_audio: functions.get("encode_audio")?,
            kernels: functions.get("kernels")?,
            close: functions.get("close")?,
            stream: setup.stream,
        })
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Encode visual units, each `[pixel_offset, frames, width, height,
    /// unit]`, into consecutive rows of `out`. Returns each unit's rows.
    pub(crate) fn encode_video(
        &self,
        pixels: &StorageView,
        units: &[[i64; 5]],
        out: &StorageView,
    ) -> Result<Vec<i64>> {
        let units = crate::ffi::list(
            units
                .iter()
                .map(|unit| Any::from(crate::ffi::integer_list(*unit))),
        );
        let rows = self.encode_video.call_values(&[
            Any::from(pixels.to_tensor()?),
            Any::from(units),
            Any::from(out.to_tensor()?),
        ])?;
        crate::ffi::integers(rows).map_err(|error| Error::numerical("encode_video", &error))
    }

    /// Encode audio tracks of `lengths` samples into consecutive rows of
    /// `out`. Returns each track's rows.
    pub(crate) fn encode_audio(
        &self,
        samples: &StorageView,
        lengths: &[u64],
        out: &StorageView,
    ) -> Result<Vec<i64>> {
        let lengths = crate::ffi::integer_list(lengths.iter().map(|&length| length as i64));
        let rows = self.encode_audio.call_values(&[
            Any::from(samples.to_tensor()?),
            Any::from(lengths),
            Any::from(out.to_tensor()?),
        ])?;
        crate::ffi::integers(rows).map_err(|error| Error::numerical("encode_audio", &error))
    }

    pub(crate) fn kernels(&self) -> &Function {
        &self.kernels
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        self.close.invoke(&[AnyView::from(&Any::from(aborted))])
    }
}

/// What a video model's text encoder component is bound with.
pub(crate) struct TextSetup<'a> {
    pub(crate) interface: &'a crate::model::interface::Interface,
    pub(crate) loaded: &'a Handle,
    pub(crate) component: &'a str,
    /// The component's root module path.
    pub(crate) path: &'a str,
    pub(crate) device: i32,
    pub(crate) stream: Stream,
    pub(crate) attention: String,
    /// Prompt capacities, the longest admitted prompt and, as the row
    /// budget, the most vision patch rows one request carries.
    pub(crate) config: EncoderConfig,
    /// The device's product arena.
    pub(crate) products: StorageView,
    /// Whether the component also encodes condition media with its vision
    /// tower.
    pub(crate) vision: bool,
    /// The model dtype the encoders compute in.
    pub(crate) dtype: DType,
}

/// One rank's text encoder: the component binding, the prompt encoder and,
/// for conditioned tasks, the vision tower bound with it. The encoders
/// write prompt features and vision features into product regions of the
/// device's product arena.
pub(crate) struct TextEncoding {
    binding: ComponentBinding,
    /// The binding's workspaces; sealed once graphs captured them.
    workspaces: StreamWorkspaces,
    pub(crate) text: EncoderRunner,
    pub(crate) vision: Option<EncoderRunner>,
    stream: Stream,
}

impl TextEncoding {
    pub(crate) fn new(setup: TextSetup<'_>) -> Result<Self> {
        let TextSetup {
            interface,
            loaded,
            component,
            path,
            device,
            stream,
            attention,
            config,
            products,
            vision,
            dtype,
        } = setup;
        let text_info = EncoderInfo::describe(interface, loaded, path, dtype)?;
        let vision_info = if vision {
            Some(EncoderInfo::describe(
                interface,
                loaded,
                &format!("{path}.vision"),
                dtype,
            )?)
        } else {
            None
        };
        // One vision call encodes every vision block of a request. Their
        // patch rows together are within the row budget, and a block holds
        // at least one feature, so the budget also bounds the blocks.
        let config = match &vision_info {
            Some(EncoderInfo::Vision(tower)) => EncoderConfig {
                max_images: (config.row_budget / (tower.downsample as usize).pow(2)).max(1),
                ..config
            },
            _ => config,
        };
        let prepared = crate::model::bind::prepare(
            interface,
            loaded,
            BindingRequest {
                component: component.to_owned(),
                path: path.to_owned(),
                device: crate::config::Device::Cuda(device),
                stream: stream.raw(),
                // The binding serves both encoders, so its operators are
                // declared for the larger of their workloads: a prompt
                // padded to the largest capacity, and a request's vision
                // blocks.
                size: [
                    text_info.binding_size(&config),
                    vision_info.and_then(|info| info.binding_size(&config)),
                ]
                .into_iter()
                .flatten()
                .reduce(|left, right| TextSize {
                    tokens: left.tokens.max(right.tokens),
                    rows: left.rows.max(right.rows),
                }),
                backends: Backends {
                    attention,
                    ..Backends::default()
                },
                derive_host_lengths: false,
                experts: None,
                weights: None,
                cache: None,
            },
        )?;
        let mut workspaces = StreamWorkspaces::default();
        // The encoders declare no symmetric workspace, so none is
        // allocated over the worker's ranks.
        let binding = crate::model::bind::bind(interface, &mut workspaces, vec![prepared], None)?
            .pop()
            .ok_or_else(|| Error::Invariant("the text encoder was not bound".into()))?;
        let text = EncoderRunner::bind(
            interface,
            &binding,
            loaded,
            "",
            text_info,
            &config,
            products.clone(),
            Some(stream.clone()),
        )?;
        let vision = vision_info
            .map(|info| {
                EncoderRunner::bind(
                    interface,
                    &binding,
                    loaded,
                    "vision",
                    info,
                    &config,
                    products,
                    Some(stream.clone()),
                )
            })
            .transpose()?;
        Ok(Self {
            binding,
            workspaces,
            text,
            vision,
            stream,
        })
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Warm the prompt capacities and, with `pool`, capture one graph per
    /// capacity; `check` bounds graph storage after each capture.
    pub(crate) fn capture(
        &mut self,
        pool: Option<&Arc<GraphPool<Handle>>>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.text.capture(pool, check)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Refuse further captures and workspace growth.
    pub(crate) fn seal(&mut self) {
        self.text.seal();
        if let Some(vision) = &mut self.vision {
            vision.seal();
        }
        self.workspaces.seal();
    }

    /// The binding's selected kernels.
    pub(crate) fn kernels(&self) -> Result<Vec<crate::ffi::Record>> {
        self.binding.kernels()
    }

    /// Release the encoders' graphs, then the binding.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = self.text.close(aborted);
        if let Some(vision) = &mut self.vision {
            result = result.and(vision.close(aborted));
        }
        result.and(self.binding.close(aborted))
    }
}

/// Parse `[elements, dtype]` as the model reports a buffer.
pub(crate) fn buffer_size(value: Any) -> Result<(u64, DType)> {
    let values: tvm_ffi::Array<Any> = value
        .try_into()
        .map_err(|error| Error::numerical("video.model", &error))?;
    if values.len() != 2 {
        return Err(Error::Interface(
            "a buffer size is [elements, dtype]".into(),
        ));
    }
    let elements = i64::try_from(
        values
            .get(0)
            .map_err(|error| Error::numerical("video.model", &error))?,
    )
    .map_err(|error| Error::numerical("video.model", &error))?;
    let dtype = tvm_ffi::String::try_from(
        values
            .get(1)
            .map_err(|error| Error::numerical("video.model", &error))?,
    )
    .map_err(|error| Error::numerical("video.model", &error))?;
    Ok((elements.max(0) as u64, parse_dtype(dtype.as_str())?))
}
