//! What the H3 decoders hand the media stage: the RGB raster and the stereo PCM.
//!
//! `uniserve/models/minimax_h3/output.py` is the reference. The video decoder emits normalized RGB
//! per channel; the output stage restores it with `value * std + mean` — the ImageNet constants — then
//! clamps it to [0, 1] and quantizes it to one byte per channel. The audio decoder emits samples in
//! [-1, 1]; the output stage clamps and quantizes them to 16-bit integers. Both are the media stage's
//! input, so both layouts are stated once, here, and the decoders in [`crate::decoding`] fill them.
//!
//! ## The video raster
//!
//! `[frames, height, width, 3]` **uint8**, row-major and channel-last: the rgb24 rasters a media
//! encoder reads, with values 0 to 255. A clip is decoded to normalized frames in `[channels, frames,
//! height, width]`; the output stage denormalizes, clamps and quantizes it, and moves the channel axis
//! last as it stores.
//!
//! ## The audio raster
//!
//! `[samples, 2]` **int16**, interleaved stereo: left then right, one pair per sample. The decoder
//! writes the two channels channel-major and [`crate::audio_vae::Pcm`] states the quantization.
//!
//! ## The overlap
//!
//! Clips overlap by [`Rgb::OVERLAP_FRAMES`] frames. The decoder keeps the predecessor's tail as one
//! `[channels, 5, height, width]` block, cross-fades the successor's first five frames against it with
//! weights `i / 5`, and emits those five blended frames before the successor's body; the first blended
//! frame is entirely the predecessor's and the fifth one fifth the successor's.

use nn::Shape;
use nn::capability::StateTable;
use nn::layers::{RgbBytes, ScaledAdd};
use nn::{DType, Forward, TensorView, Weight};

use crate::config::CLIP_FRAMES;
use crate::decoding::{Buffer, CLIP_OUTPUT_FRAMES, OVERLAP_FRAMES};
use crate::{Error, Result};

/// The per-channel standard deviations the video decoder's RGB is normalized by, in red, green, blue
/// order: `value * std + mean` restores the pixel.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The per-channel means the video decoder's RGB is normalized by, in red, green, blue order.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// The one byte per channel the video raster stores.
pub const BYTE_MAX: f32 = 255.0;

/// The one integer step of the audio raster: a sample of `1.0` quantizes to it.
pub const PCM_MAX: f32 = 32767.0;

/// The extents of one request's video raster, and the bytes it occupies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgb {
    /// Frames of the timeline.
    pub frames: u32,
    /// Pixel rows.
    pub height: u32,
    /// Pixel columns.
    pub width: u32,
}

impl Rgb {
    /// Colour channels of the raster: red, green and blue.
    pub const CHANNELS: u32 = 3;

    /// Frames one clip shares with the next.
    pub const OVERLAP_FRAMES: u32 = 5;

    /// The raster of `frames` decoded frames at a pixel extent.
    pub fn of(frames: u32, height: u32, width: u32) -> Self {
        Self {
            frames,
            height,
            width,
        }
    }

    /// The bytes the raster occupies: one per channel per pixel.
    pub fn bytes(&self) -> u64 {
        u64::from(self.frames)
            * u64::from(self.height)
            * u64::from(self.width)
            * u64::from(Self::CHANNELS)
    }

    /// The frames the decoder's own workspace holds: the longest clip, the last, which keeps its tail.
    pub const fn workspace_frames() -> u32 {
        25
    }

    /// The extent of one overlap block, `[channels, OVERLAP_FRAMES, height, width]` as the decoder
    /// keeps it between clips.
    pub fn overlap_elements(&self) -> u64 {
        u64::from(Self::CHANNELS)
            * u64::from(Self::OVERLAP_FRAMES)
            * u64::from(self.height)
            * u64::from(self.width)
    }

    /// One normalized pixel's three bytes, the reference's arithmetic.
    ///
    /// `value * std + mean` per channel, clamped to [0, 1], scaled to 255, rounded half to even, and
    /// truncated to a byte. This is the reference of the operator the output stage runs; a kernel
    /// that computes it must agree with it on every input, including the ties a half-even round
    /// resolves away from zero.
    pub fn quantize(&self, normalized: [f32; 3]) -> [u8; 3] {
        let mut bytes = [0u8; 3];
        for (channel, value) in normalized.iter().enumerate() {
            let restored = value * PIXEL_STD[channel] + PIXEL_MEAN[channel];
            let clamped = restored.clamp(0.0, 1.0);
            bytes[channel] = (clamped * BYTE_MAX).round_ties_even() as u8;
        }
        bytes
    }
}

/// The extents of one request's audio raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pcm {
    /// Sample frames of the track.
    pub samples: u32,
}

impl Pcm {
    /// Interleaved channels: left and right.
    pub const CHANNELS: u32 = 2;

    /// The raster of `samples` sample frames.
    pub const fn of(samples: u32) -> Self {
        Self { samples }
    }

    /// The integers the raster holds: one per channel per sample.
    pub fn elements(&self) -> u64 {
        u64::from(self.samples) * u64::from(Self::CHANNELS)
    }

    /// The bytes the raster occupies, two per integer.
    pub fn bytes(&self) -> u64 {
        self.elements() * 2
    }

    /// One sample's integer: clamped to [-1, 1], scaled by 32767 and rounded half to even.
    ///
    /// The same arithmetic [`crate::audio_vae::Pcm::quantize`] states for the decoder's output
    /// stage; the two must agree on every input.
    pub fn quantize(&self, sample: f32) -> i16 {
        (sample.clamp(-1.0, 1.0) * PCM_MAX).round_ties_even() as i16
    }
}

/// The element type of the video raster the media stage reads.
pub const VIDEO_DTYPE: &str = "u8";

/// The element type of the audio raster the media stage reads.
pub const AUDIO_DTYPE: &str = "i16";

/// One clip's placement in the timeline, as the output stage reads it.
///
/// The body is the frames the timeline keeps from this clip; the overlap is the tail the next clip
/// cross-fades against. Both index the clip's own decoded frames, not the timeline's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Splice {
    /// The frames of the clip's own decode that the timeline keeps.
    pub body: (u32, u32),
    /// The frames of the clip's own decode that overlap the next clip.
    pub overlap: (u32, u32),
    /// Whether this clip ends the timeline and therefore writes its own tail.
    pub last: bool,
}

impl Splice {
    /// The splice the checkpoint's 25-frame clip decodes to: a 17-frame body to keep and a 5-frame
    /// tail after three VAE padding frames.
    pub const CLIP: Self = Self {
        body: (0, CLIP_FRAMES),
        overlap: (CLIP_OUTPUT_FRAMES - OVERLAP_FRAMES, CLIP_OUTPUT_FRAMES),
        last: false,
    };

    /// The splice of a clip that ends the timeline: it keeps its own body **and** its own tail, and it
    /// still drops the three padding frames between them - the tail is the last five frames of the
    /// video, not the padding that precedes it.
    pub const LAST: Self = Self {
        body: (0, CLIP_FRAMES),
        overlap: (CLIP_OUTPUT_FRAMES - OVERLAP_FRAMES, CLIP_OUTPUT_FRAMES),
        last: true,
    };

    /// How many frames of this clip the timeline writes.
    pub fn frames(&self) -> u32 {
        let body = self.body.1 - self.body.0;
        let tail = if self.last {
            self.overlap.1 - self.overlap.0
        } else {
            0
        };
        body + tail
    }
}

/// The video postprocessor: one clip's decoded frames into the request's bytes.
///
/// `output.py:43-70` and `models/decoder.py:356-620` are the reference, and this is the step the design
/// puts at the component's second entry point (`video_postprocessor.forward`): **the decoder decodes a
/// clip, this splices it**. Per clip it drops the three VAE padding frames, cross-fades the clip's first
/// five body frames against the predecessor's tail at weights `i/5` (the reference's
/// `arange(5)/5`, so the first blended frame is entirely the predecessor's), yields the tail to the next
/// clip as state, and writes the kept frames as `[frames, rows, columns, 3]` bytes.
///
/// The blend is `CrossFade` on the **frames** axis: the two sources are each one contiguous block (a
/// `[3, 25, H, W]` clip and a `[3, 5, H, W]` tail), which is the operator's condition, and the frames'
/// weights are the ones it applies along the leading positions of its axis, and they are the reference's
/// to the rounding: the weights are `i / 5` for frame `i` of the overlap, which is
/// `torch.arange(5, dtype=torch.float16)/5` in `output.py`, and each frame is two `scaled_add` launches
/// with those two host scalars. The ramp is built here in FP32 where the reference builds it in the
/// raster's type: one declared deviation, at the ramp's last bit and not in the blend's arithmetic. The arithmetic is the reference's decoded-precision
/// blend followed by the FP32 de-normalization, in that order: the de-normalization is [`RgbBytes`]' and
/// reads the same constants the checkpoint publishes.
pub struct VideoPostprocessor {
    bytes: RgbBytes,
    dtype: DType,
}

impl VideoPostprocessor {
    /// The blend over the five overlap frames and the byte packing of a raster of `dtype`.
    ///
    /// `mean` and `std` are the checkpoint's `[3]` de-normalization constants, published by the loader.
    pub fn new(dtype: DType, mean: Weight, std: Weight) -> Result<Self> {
        Ok(Self {
            bytes: RgbBytes::new(dtype, Rgb::CHANNELS, mean, std)?,
            dtype,
        })
    }

    /// The element type of the clip raster this reads.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The frames one clip of `splice` writes, which is what [`Self::forward`]'s `out` must hold.
    pub fn frames(splice: &Splice) -> u32 {
        splice.frames()
    }

    /// Write one clip's kept frames into `out`, blending with `previous` when it has a predecessor.
    ///
    /// `clip` is the clip's decoded raster `[3, frames, rows, columns]` in [`Self::dtype`]; `previous`
    /// is the predecessor's tail `[3, overlap, rows, columns]` and is `None` for the timeline's first
    /// clip, which has nothing to fade against; `out` is `[splice.frames(), rows, columns, 3]` bytes.
    /// The blend's scratch is reserved here, in the arena the executor sized from the plan: the model is
    /// the one reserver, and a buffer the runner placed would be a second mechanism.
    ///
    /// The clip's tail is not written here unless `splice.last`: it is the state the caller hands the
    /// next clip, and passing it on is the runner's.
    pub fn forward(
        &self,
        f: &Forward,
        previous: Option<TensorView>,
        clip: TensorView,
        splice: &Splice,
        out: TensorView,
    ) -> Result<()> {
        let raster = Raster::of(clip)?;
        raster.check_out(out, splice)?;
        let source = match previous {
            Some(previous) => {
                raster.check_previous(previous, self.dtype)?;
                let blended = f.arena().tensor(clip.dtype, clip.shape)?;
                self.blend(f, &raster, previous, clip, splice.overlap, blended)?;
                blended
            }
            None => clip,
        };
        // The byte step writes as many frames as it reads, so each range goes to its own slice of `out`:
        // the body to the clip's kept frames, and the last clip's tail after them. Handing `out` whole to
        // either write would be refused by the layer's own shape check, and the tail written at the front
        // of a wider output would overwrite the body's first frames.
        let (body_out, tail_out) = raster.destinations(out, splice)?;
        self.bytes
            .forward(f, raster.body(source, splice)?, body_out)?;
        if let Some(tail_out) = tail_out {
            self.bytes
                .forward(f, raster.written(source, splice.overlap)?, tail_out)?;
        }
        Ok(())
    }
}

impl VideoPostprocessor {
    /// Fade the clip in against its predecessor, five frames at a time.
    ///
    /// The reference's weights are `i/5` for frame `i` of the overlap, `0` to `0.8`: the blend is
    /// `previous · (1 − w) + clip · w`. The two scalars the sum takes are host values, so each frame is two
    /// launches of the operator every other sum in the model uses. The reference builds its ramp in the
    /// raster's type and this one in FP32; the products and the sum are FP32 either way and round to the
    /// raster once, so the difference is at most the ramp's own last bit. `CrossFade` is the row-and-column seam blend and refuses the frames axis; a
    /// blend along frames is not that operator with another argument, and this is the arithmetic with the
    /// operands a raster has - one contiguity per channel per frame, which is what `scaled_add` reads.
    fn blend(
        &self,
        f: &Forward,
        raster: &Raster,
        previous: TensorView,
        clip: TensorView,
        overlap: (u32, u32),
        blended: TensorView,
    ) -> Result<()> {
        let count = overlap.1 - overlap.0;
        let pixels = u64::from(raster.height) * u64::from(raster.width);
        for channel in 0..u64::from(Rgb::CHANNELS) {
            for frame in 0..count {
                // The reference weights the *current* clip `i/5` over the five frames: `0, 0.2, ..., 0.8`,
                // so the first blended frame is entirely the predecessor's and the last keeps a fifth of
                // it. A ramp that started at `1/5` would drop the predecessor from the last frame.
                let weight = frame as f32 / count as f32;
                let target = TensorView::new(
                    blended.element(&[channel, u64::from(overlap.0 + frame), 0, 0]),
                    blended.dtype,
                    Shape::new(&[1, pixels]),
                );
                let source = TensorView::new(
                    previous.element(&[channel, u64::from(frame), 0, 0]),
                    previous.dtype,
                    Shape::new(&[1, pixels]),
                );
                // `target ← (1 − w) · previous + w · clip`, in two steps because the operator scales its own
                // target: the adjacent frame of the clip is only read.
                ScaledAdd.forward(f, target, source, 1.0, 0.0)?;
                let current = TensorView::new(
                    clip.element(&[channel, u64::from(overlap.0 + frame), 0, 0]),
                    clip.dtype,
                    Shape::new(&[1, pixels]),
                );
                ScaledAdd.forward(f, target, current, weight, 1.0 - weight)?;
            }
        }
        Ok(())
    }
}

impl nn::capability::Decoder for VideoPostprocessor {
    /// No tables: the splice's parameters are its arguments, not state.
    ///
    /// The preprocessor holds two `[3]` constants the load reads once, and every other number a splice
    /// needs - which frames it keeps, how many fade, where on the timeline it writes - arrives as the
    /// [`nn::capability::Splice`] the runner passes. So an empty list is the *correct* answer for this
    /// block, not a refusal, and a runner that asks must get it.
    fn layout_tables(&self, _layout: &nn::capability::Layout) -> Vec<StateTable> {
        Vec::new()
    }

    /// Nothing to prepare, for the reason above: `layout_tables` answers no tables, so there are no
    /// bytes to write and `out` must already be empty.
    fn prepare_layout(
        &self,
        _layout: &nn::capability::Layout,
        out: &mut [Vec<u8>],
    ) -> nn::Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        Err(nn::Error::Shape {
            owner: "VideoPostprocessor",
            problem: format!(
                "a postprocessor has no layout tables, and {} buffers were offered",
                out.len()
            ),
        })
    }

    /// Refuse: the postprocessor has no latents to decode.
    ///
    /// It is a named block of the decoder kind because the splice is a decode of the *timeline*, not of one
    /// clip, and placing it under its own name is what lets a deployment serve without it. A runner that
    /// asks it to decode a clip hears which operation it should have called.
    fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        state: &[TensorView],
        out: nn::capability::DecodedMedia,
    ) -> nn::Result<()> {
        let _ = (f, latents, state, out);
        Err(nn::Error::Shape {
            owner: "VideoPostprocessor",
            problem: "a postprocessor splices; it decodes nothing".to_owned(),
        })
    }

    /// Write one clip's kept frames into the timeline, fading it against its predecessor.
    ///
    /// `previous` is the predecessor's own decode of the frames the two share,
    /// `[channels, splice.overlap, rows, columns]`, and is absent for the timeline's first clip;
    /// `out.video` is the whole timeline's byte raster `[frames, rows, columns, channels]`, whose frames at
    /// `splice.first` this writes. The clip keeps `splice.keep` frames from its first, so the clip that ends
    /// the timeline - the one that keeps its tail as well - states a larger `keep`, which is this block's
    /// own `last`.
    fn splice(
        &self,
        f: &Forward,
        previous: Option<TensorView>,
        clip: TensorView,
        splice: nn::capability::Splice,
        out: nn::capability::DecodedMedia,
    ) -> nn::Result<()> {
        let refuse = |problem: String| nn::Error::Shape {
            owner: "VideoPostprocessor",
            problem,
        };
        let video = out
            .video
            .ok_or_else(|| refuse("a splice writes a timeline's raster".to_owned()))?;
        if out.audio.is_some() {
            return Err(refuse("a splice writes no audio".to_owned()));
        }
        if splice.overlap > OVERLAP_FRAMES {
            return Err(refuse(format!(
                "{} frames of overlap, and the checkpoint fades {OVERLAP_FRAMES}",
                splice.overlap
            )));
        }
        let raster = Raster::of(clip).map_err(|error| refuse(format!("{error}")))?;
        // The checkpoint's clip is 25 decoded frames: 17 the timeline keeps and, after three padding
        // frames, a five-frame tail. A clip a successor follows hands that tail on; the clip that ends the
        // timeline keeps it, which the runner states by keeping frames beyond the body.
        let internal = Splice {
            body: (0, CLIP_FRAMES),
            overlap: (CLIP_OUTPUT_FRAMES - splice.overlap, CLIP_OUTPUT_FRAMES),
            last: splice.keep > CLIP_FRAMES,
        };
        // The clip's frames go where the runner says they do, so the view handed to the write starts at
        // `first`: the raster is `[frames, rows, columns, channels]` and a frame is one contiguous block.
        let view = TensorView::new(
            video.element(&[u64::from(splice.first), 0, 0, 0]),
            DType::U8,
            Shape::new(&[
                u64::from(Self::frames(&internal)),
                u64::from(raster.height),
                u64::from(raster.width),
                u64::from(Rgb::CHANNELS),
            ]),
        );
        self.forward(f, previous, clip, &internal, view)
            .map_err(|error| refuse(format!("{error}")))
    }
}

/// The extents of one clip's decoded raster, read off the view.
#[derive(Clone, Copy)]
struct Raster {
    frames: u32,
    height: u32,
    width: u32,
}

impl Raster {
    /// The raster `[channels, frames, rows, columns]` a `clip` view describes.
    fn of(clip: TensorView) -> Result<Self> {
        if clip.shape.rank() != 4 || clip.shape.dim(0) != u64::from(Rgb::CHANNELS) {
            return Err(Error::config(format!(
                "a decoded clip is [{}, frames, rows, columns], not {:?}",
                Rgb::CHANNELS,
                clip.shape.dims()
            )));
        }
        let frames = u32::try_from(clip.shape.dim(1))
            .map_err(|_| Error::config("a clip taller than a frame count".to_owned()))?;
        if frames != CLIP_OUTPUT_FRAMES {
            return Err(Error::config(format!(
                "a clip's decode is {CLIP_OUTPUT_FRAMES} frames and the view carries {frames}"
            )));
        }
        Ok(Self {
            frames,
            height: u32::try_from(clip.shape.dim(2))
                .map_err(|_| Error::config("a raster taller than a pixel count".to_owned()))?,
            width: u32::try_from(clip.shape.dim(3))
                .map_err(|_| Error::config("a raster wider than a pixel count".to_owned()))?,
        })
    }

    /// The view of `range` frames of `raster`, as `[channels, frames, rows, columns]`.
    ///
    /// A frame range is a column-block of every channel: the view keeps the clip's own strides and
    /// narrows the frames axis, so a clip that is a sub-block of a larger raster stays on the frames it
    /// names. The operators on this path read through the view's strides rather than by flattening.
    fn written(&self, raster: TensorView, range: (u32, u32)) -> Result<TensorView> {
        let (start, stop) = range;
        if start >= stop || stop > self.frames {
            return Err(Error::config(format!(
                "a clip's frames {start}..{stop} lie outside its {} decoded frames",
                self.frames
            )));
        }
        let element = raster.element(&[0, u64::from(start), 0, 0]);
        let shape = Shape::new(&[
            u64::from(Rgb::CHANNELS),
            u64::from(stop - start),
            u64::from(self.height),
            u64::from(self.width),
        ]);
        // The frame range's strides are the clip's own, so a clip that is a sub-block of a larger
        // raster keeps reading the right frames; only the frames extent narrows.
        let strides = nn::Strides::new(raster.strides.as_slice());
        Ok(TensorView::with_strides(
            element,
            raster.dtype,
            shape,
            strides,
        ))
    }

    /// The body view of one clip: what the timeline keeps before the tail.
    fn body(&self, raster: TensorView, splice: &Splice) -> Result<TensorView> {
        self.written(raster, splice.body)
    }

    /// The frames of `out` each write goes to: the body's, and the last clip's tail after them.
    ///
    /// `out` is `[frames, rows, columns, channels]` bytes for the whole clip, and a write covers exactly
    /// the frames its source reads, so a clip that keeps its own tail needs two destination views.
    fn destinations(
        &self,
        out: TensorView,
        splice: &Splice,
    ) -> Result<(TensorView, Option<TensorView>)> {
        let keep = splice.body.1 - splice.body.0;
        let body = frames_of(out, 0, keep)?;
        let tail = if splice.last {
            Some(frames_of(
                out,
                keep,
                keep + (splice.overlap.1 - splice.overlap.0),
            )?)
        } else {
            None
        };
        Ok((body, tail))
    }

    /// The output view of one clip: `[frames, rows, columns, channels]` bytes.
    fn check_out(&self, out: TensorView, splice: &Splice) -> Result<()> {
        let frames = u64::from(splice.frames());
        if out.dtype != DType::U8 {
            return Err(Error::config(format!(
                "the output raster is u8 bytes, not {out:?}"
            )));
        }
        if out.shape.dims() != [frames, u64::from(self.height), u64::from(self.width), 3] {
            return Err(Error::config(format!(
                "the output raster is [{} kept, {}, {}, 3] and the view is {:?}",
                splice.frames(),
                self.height,
                self.width,
                out.shape.dims()
            )));
        }
        Ok(())
    }

    /// The predecessor's tail, which must be one contiguous block of the clip's own type.
    fn check_previous(&self, previous: TensorView, dtype: DType) -> Result<()> {
        let shape = [
            u64::from(Rgb::CHANNELS),
            u64::from(OVERLAP_FRAMES),
            u64::from(self.height),
            u64::from(self.width),
        ];
        if previous.shape.dims() != shape {
            return Err(Error::config(format!(
                "a predecessor's tail is {shape:?} and the view is {:?}",
                previous.shape.dims()
            )));
        }
        if previous.dtype != dtype {
            return Err(Error::config(format!(
                "the predecessor's tail is {} and the clip is {dtype}",
                previous.dtype
            )));
        }
        Ok(())
    }
}

/// The blend's scratch for one clip: a clip raster in the decoded element type.
///
/// [`VideoPostprocessor::forward`] fades a clip against its predecessor's tail into a buffer of the
/// clip's own shape, because the fade's destination cannot be the source it reads; that buffer is the
/// only thing the splice reserves, and it is named here so the executor sizes the arena from
/// arithmetic rather than from the module's own guess. A clip with no predecessor reserves nothing.
pub fn splice_arena(
    config: &crate::VideoVaeConfig,
    canvas: &crate::video_vae::Canvas,
    dtype: DType,
) -> Result<Vec<Buffer>> {
    let element = match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        other => {
            return Err(Error::config(format!(
                "the splice blends in f16 or f32 decoded precision, not {other}"
            )));
        }
    };
    let (height, width) = canvas.pixels(config);
    Ok(vec![Buffer {
        name: "blended".to_owned(),
        element,
        elements: u64::from(Rgb::CHANNELS)
            * u64::from(crate::decoding::CLIP_OUTPUT_FRAMES)
            * u64::from(height)
            * u64::from(width),
    }])
}

/// The frames `start..stop` of an output raster: `[frames, rows, columns, channels]` bytes.
///
/// The last axis is dense, so a frame slice is that frame's own element address and a narrowed first
/// extent; the strides are the raster's own, which keeps the view right for a raster that is itself a
/// slice of a larger buffer.
fn frames_of(out: TensorView, start: u32, stop: u32) -> Result<TensorView> {
    let shape = out.shape;
    if shape.rank() != 4 || start >= stop || shape.dim(0) < u64::from(stop) {
        return Err(Error::config(format!(
            "frames {start}..{stop} of a {}-frame output raster do not exist",
            shape.dim(0)
        )));
    }
    let element = out.element(&[u64::from(start), 0, 0, 0]);
    Ok(TensorView::with_strides(
        element,
        out.dtype,
        Shape::new(&[
            u64::from(stop - start),
            shape.dim(1),
            shape.dim(2),
            shape.dim(3),
        ]),
        nn::Strides::new(out.strides.as_slice()),
    ))
}

/// The whole timeline of `clips` clips: how many frames it writes.
pub fn timeline_frames(clips: &[Splice]) -> Result<u32> {
    if clips.is_empty() {
        return Err(Error::config("a video timeline has a clip".to_owned()));
    }
    let mut frames = 0;
    for clip in clips {
        frames += clip.frames();
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constants(values: [f32; 3]) -> Weight {
        // The constants are host bytes the executor uploads; a test only needs a weight that names
        // three values, never a device that holds them.
        let bytes: &'static [u8] = Box::leak(
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        let ptr = nn::DevicePtr::new(bytes.as_ptr() as u64);
        Weight {
            backing: nn::Backing::new(ptr, bytes.len() as u64),
            view: TensorView::new(ptr, DType::F32, Shape::new(&[3])),
        }
    }

    #[test]
    fn the_byte_step_refuses_constants_that_are_not_three_channels() {
        let mean = constants(PIXEL_MEAN);
        let std = constants(PIXEL_STD);
        assert!(RgbBytes::new(DType::F16, 3, mean, std).is_ok());
        // Two channels cannot de-normalize an RGB raster; the layer refuses at construction rather
        // than reading past the constant.
        let two = {
            let bytes: &'static [u8] = Box::leak(vec![0u8; 8].into_boxed_slice());
            let ptr = nn::DevicePtr::new(bytes.as_ptr() as u64);
            Weight {
                backing: nn::Backing::new(ptr, bytes.len() as u64),
                view: TensorView::new(ptr, DType::F32, Shape::new(&[2])),
            }
        };
        assert!(RgbBytes::new(DType::F16, 2, two, constants(PIXEL_STD)).is_err());
    }

    #[test]
    fn the_last_clips_tail_is_written_after_its_body() {
        // A `[frames, rows, columns, channels]` output raster of 22 frames, the last clip's own count:
        // its body goes to the first 17 and its tail to the 5 after them. Handing the whole raster to
        // either write is what the byte layer refuses, and writing the tail at the front would overwrite
        // the body - so the destination views are the contract this test pins.
        let (height, width) = (4u32, 6u32);
        let channels = Rgb::CHANNELS;
        let out = TensorView::with_strides(
            nn::DevicePtr::new(0),
            DType::U8,
            Shape::new(&[22, u64::from(height), u64::from(width), u64::from(channels)]),
            nn::Strides::new(&[
                u64::from(height) * u64::from(width) * u64::from(channels),
                u64::from(width) * u64::from(channels),
                u64::from(channels),
                1,
            ]),
        );
        let raster = Raster {
            frames: 25,
            height,
            width,
        };
        let (body, tail) = raster
            .destinations(out, &Splice::LAST)
            .expect("the last clip's two destinations");
        assert_eq!(
            body.shape.dims()[0],
            u64::from(Splice::LAST.body.1 - Splice::LAST.body.0)
        );
        assert_eq!(body.element(&[0, 0, 0, 0]).address(), 0);
        let tail = tail.expect("the last clip keeps its tail");
        assert_eq!(
            tail.shape.dims()[0],
            u64::from(Splice::LAST.overlap.1 - Splice::LAST.overlap.0)
        );
        // The tail starts at the body's end, which is the 17th frame's own address.
        assert_eq!(
            tail.element(&[0, 0, 0, 0]).address(),
            out.element(&[17, 0, 0, 0]).address()
        );
        // A clip that is not last writes one destination, and it is the body.
        let (body, tail) = raster
            .destinations(
                TensorView::with_strides(
                    nn::DevicePtr::new(0),
                    DType::U8,
                    Shape::new(&[17, u64::from(height), u64::from(width), u64::from(channels)]),
                    nn::Strides::new(&[
                        u64::from(height) * u64::from(width) * u64::from(channels),
                        u64::from(width) * u64::from(channels),
                        u64::from(channels),
                        1,
                    ]),
                ),
                &Splice::CLIP,
            )
            .expect("the interior clip's destination");
        assert_eq!(body.shape.dims()[0], 17);
        assert!(tail.is_none());
    }

    #[test]
    fn the_splices_scratch_is_one_clip_raster() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 48, 84);
        let (height, width) = canvas.pixels(&config);
        let buffers = splice_arena(&config, &canvas, DType::F16).expect("the scratch");
        assert_eq!(
            buffers.len(),
            1,
            "the fade's destination is the only reservation"
        );
        // The blend writes one clip, so the scratch is the clip's raster: channels, the decoder's
        // frames, and the canvas's pixels - the same extent `forward` reserves.
        assert_eq!(
            buffers[0].elements,
            u64::from(Rgb::CHANNELS)
                * u64::from(CLIP_OUTPUT_FRAMES)
                * u64::from(height)
                * u64::from(width)
        );
        assert_eq!(buffers[0].element, "f16");
        assert!(splice_arena(&config, &canvas, DType::BF16).is_err());
    }

    #[test]
    fn a_normalized_pixel_restores_its_bytes() {
        let rgb = Rgb::of(1, 1, 1);
        // Mid grey of the normalization: value 0 restores the mean itself.
        assert_eq!(rgb.quantize([0.0, 0.0, 0.0]), [124, 116, 104]);
        // The ends clamp: a value past the range does not wrap.
        assert_eq!(rgb.quantize([4.0, 4.0, 4.0]), [255, 255, 255]);
        assert_eq!(rgb.quantize([-4.0, -4.0, -4.0]), [0, 0, 0]);
        // A normalized value is a multiple of the channel's standard deviation above its mean, so 1.0
        // restores the mean plus one standard deviation, not white.
        assert_eq!(rgb.quantize([1.0, 1.0, 1.0]), [182, 173, 161]);
        assert_eq!(rgb.quantize([-1.0, -1.0, -1.0]), [65, 59, 46]);
        // White and black are the values that de-normalize to the ends of the range.
        let white = [0usize, 1, 2].map(|channel| (1.0 - PIXEL_MEAN[channel]) / PIXEL_STD[channel]);
        let black = [0usize, 1, 2].map(|channel| -PIXEL_MEAN[channel] / PIXEL_STD[channel]);
        assert_eq!(rgb.quantize(white), [255, 255, 255]);
        assert_eq!(rgb.quantize(black), [0, 0, 0]);
    }

    #[test]
    fn a_tie_rounds_to_even() {
        // 0.5 * 255 = 127.5 rounds to 128; 2.5 rounds to 2, not 3.
        assert_eq!(0.5f32.round_ties_even(), 0.0);
        assert_eq!(2.5f32.round_ties_even(), 2.0);
        assert_eq!(127.5f32.round_ties_even(), 128.0);
        // A pixel that lands exactly on 127.5 quantizes to 128, the even byte.
        let rgb = Rgb::of(1, 1, 1);
        let value = (127.5 / BYTE_MAX - PIXEL_MEAN[0]) / PIXEL_STD[0];
        assert_eq!(rgb.quantize([value, 0.0, 0.0])[0], 128);
    }

    #[test]
    fn a_sample_quantizes_to_the_integer_pcm() {
        let pcm = Pcm::of(800);
        assert_eq!(pcm.quantize(0.0), 0);
        assert_eq!(pcm.quantize(1.0), 32767);
        assert_eq!(pcm.quantize(-1.0), -32767);
        assert_eq!(pcm.quantize(2.0), 32767);
        assert_eq!(pcm.quantize(-2.0), -32767);
        // Half a step is a tie: it rounds to even, which here is zero.
        assert_eq!(pcm.quantize(0.5 / PCM_MAX), 0);
        assert_eq!(pcm.quantize(1.5 / PCM_MAX), 2);
        assert_eq!(pcm.elements(), 1600);
        assert_eq!(pcm.bytes(), 3200);
    }

    /// The clip raster test views are built on: one contiguous `[3, 25, rows, columns]` F16 block, so
    /// the offsets a view answers are the ones the arithmetic would read on the device.
    fn raster_view(height: u32, width: u32) -> TensorView {
        let shape = Shape::new(&[
            u64::from(Rgb::CHANNELS),
            u64::from(CLIP_OUTPUT_FRAMES),
            u64::from(height),
            u64::from(width),
        ]);
        TensorView::new(nn::DevicePtr::new(0), nn::DType::F16, shape)
    }

    #[test]
    fn a_clip_view_reads_the_body_and_the_tail_and_never_the_padding() {
        let (height, width) = (4, 6);
        let clip = raster_view(height, width);
        let raster = Raster::of(clip).expect("a clip raster");
        let per_frame = u64::from(height) * u64::from(width);
        let per_channel = u64::from(CLIP_OUTPUT_FRAMES) * per_frame;
        // The body is the clip's first frames of every channel: the channel stride stays the whole
        // clip's, so the second channel is one channel into the block, not one body into it.
        let body = raster.body(clip, &Splice::CLIP).expect("a body");
        assert_eq!(
            body.shape.dims(),
            [
                u64::from(Rgb::CHANNELS),
                u64::from(CLIP_FRAMES),
                u64::from(height),
                u64::from(width),
            ]
        );
        assert_eq!(
            body.strides.as_slice(),
            [per_channel, per_frame, u64::from(width), 1]
        );
        assert_eq!(body.element(&[1, 0, 0, 0]), clip.element(&[1, 0, 0, 0]));
        // The tail's first frame is the frame after the three padding frames, with the same channel
        // stride - the offset that a wrong range would take from the padding instead.
        let tail = raster.written(clip, Splice::LAST.overlap).expect("a tail");
        assert_eq!(tail.shape.dim(1), u64::from(OVERLAP_FRAMES));
        assert_eq!(tail.strides.as_slice(), body.strides.as_slice());
        assert_eq!(
            tail.element(&[0, 0, 0, 0]),
            clip.element(&[0, u64::from(CLIP_OUTPUT_FRAMES - OVERLAP_FRAMES), 0, 0])
        );
        // The body and the tail do not meet: the frames between them belong to neither.
        let body_last = body.element(&[0, u64::from(CLIP_FRAMES) - 1, 0, 0]);
        assert_ne!(body_last, tail.element(&[0, 0, 0, 0]));
        assert_eq!(
            body_last,
            clip.element(&[0, u64::from(CLIP_FRAMES) - 1, 0, 0])
        );
    }

    #[test]
    fn a_strided_clip_keeps_its_own_frames_and_a_range_outside_it_is_refused() {
        let (height, width) = (4_u32, 6_u32);
        let per_frame = u64::from(height) * u64::from(width);
        let per_channel = u64::from(CLIP_OUTPUT_FRAMES) * per_frame;
        // A clip that is a sub-block of a wider raster: one frame padding between frames, so the
        // frame stride is the block's own and not the contiguous one.
        let padded = TensorView::with_strides(
            nn::DevicePtr::new(0),
            nn::DType::F16,
            Shape::new(&[
                3,
                u64::from(CLIP_OUTPUT_FRAMES),
                u64::from(height),
                u64::from(width),
            ]),
            nn::Strides::new(&[per_channel + per_frame, per_frame * 2, u64::from(width), 1]),
        );
        let raster = Raster::of(padded).expect("a clip raster");
        let body = raster.body(padded, &Splice::CLIP).expect("a body");
        // The view keeps the clip's strides; only the frames extent narrows.
        assert_eq!(body.strides.as_slice(), padded.strides.as_slice());
        assert_eq!(body.element(&[0, 1, 0, 0]), padded.element(&[0, 1, 0, 0]));
        assert_eq!(body.element(&[1, 0, 0, 0]), padded.element(&[1, 0, 0, 0]));
        // A range that leaves the clip is refused rather than read past it.
        assert!(
            raster
                .written(padded, (CLIP_FRAMES, CLIP_OUTPUT_FRAMES + 1))
                .is_err()
        );
        assert!(raster.written(padded, (CLIP_FRAMES, CLIP_FRAMES)).is_err());
    }

    #[test]
    fn a_clip_view_refuses_a_raster_that_is_not_one_clip() {
        // A view whose frames axis is not the checkpoint's 25 is refused before any launch.
        let short = TensorView::new(
            nn::DevicePtr::new(0),
            nn::DType::F16,
            Shape::new(&[3, 24, 4, 6]),
        );
        assert!(Raster::of(short).is_err());
        // And a raster that is not three channels is refused too.
        let grey = TensorView::new(
            nn::DevicePtr::new(0),
            nn::DType::F16,
            Shape::new(&[1, u64::from(CLIP_OUTPUT_FRAMES), 4, 6]),
        );
        assert!(Raster::of(grey).is_err());
        // The output view is `[kept frames, rows, columns, channels]` bytes.
        let raster = Raster::of(raster_view(4, 6)).expect("a clip raster");
        let kept = Splice::LAST.frames();
        let out = TensorView::new(
            nn::DevicePtr::new(0),
            nn::DType::U8,
            Shape::new(&[u64::from(kept), 4, 6, 3]),
        );
        assert!(raster.check_out(out, &Splice::LAST).is_ok());
        let short_out = TensorView::new(
            nn::DevicePtr::new(0),
            nn::DType::U8,
            Shape::new(&[u64::from(kept) - 1, 4, 6, 3]),
        );
        assert!(raster.check_out(short_out, &Splice::LAST).is_err());
    }

    #[test]
    fn the_raster_sizes_follow_the_layout() {
        let rgb = Rgb::of(107, 768, 1344);
        assert_eq!(rgb.bytes(), 107 * 768 * 1344 * 3);
        assert_eq!(rgb.overlap_elements(), 3 * 5 * 768 * 1344);
        assert_eq!(Rgb::workspace_frames(), CLIP_OUTPUT_FRAMES);
        // A clip writes its body and shares a tail; the last writes its body **and** its tail, and the
        // three padding frames between them are dropped, which is what makes the timeline `17n + 5`.
        assert_eq!(Splice::CLIP.frames(), CLIP_FRAMES);
        assert_eq!(Splice::LAST.frames(), CLIP_FRAMES + OVERLAP_FRAMES);
        let clips_n = 6;
        let mut clips = vec![Splice::CLIP; clips_n as usize];
        *clips.last_mut().expect("a clip") = Splice::LAST;
        assert_eq!(
            timeline_frames(&clips).expect("timeline"),
            CLIP_FRAMES * clips_n + OVERLAP_FRAMES
        );
        // The padding frames are neither kept nor shared: `LAST`'s two ranges leave them out.
        let (start, stop) = Splice::LAST.overlap;
        assert!(Splice::LAST.body.1 < start && stop - start == OVERLAP_FRAMES);
        assert!(timeline_frames(&[]).is_err());
    }
}
