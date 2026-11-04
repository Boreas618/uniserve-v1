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
use nn::layers::{CrossFade, RgbBytes};
use nn::{DType, Forward, TensorView, Weight};

use crate::config::CLIP_FRAMES;
use crate::decoding::{CLIP_OUTPUT_FRAMES, OVERLAP_FRAMES};
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
/// to the rounding: the operator builds its ramp as `position / extent` **in the raster's dtype**, which
/// is `torch.arange(5, dtype=…)/5` in `output.py`, so the blend needs no declared deviation. The
/// arithmetic is the reference's decoded-precision blend followed by the FP32 de-normalization, in that
/// order: the de-normalization is [`RgbBytes`]' and reads the same constants the checkpoint publishes.
pub struct VideoPostprocessor {
    fade: CrossFade,
    bytes: RgbBytes,
    dtype: DType,
}

impl VideoPostprocessor {
    /// The blend over the five overlap frames and the byte packing of a raster of `dtype`.
    ///
    /// `mean` and `std` are the checkpoint's `[3]` de-normalization constants, published by the loader.
    pub fn new(dtype: DType, mean: Weight, std: Weight) -> Result<Self> {
        let fade = CrossFade::new(dtype, OVERLAP_FRAMES, 1).map_err(|error| {
            Error::config(format!(
                "the clip splice fades five frames of the frames axis, and CrossFade refuses axis 1: \
                 {error}"
            ))
        })?;
        Ok(Self {
            fade,
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
    /// clip, which has nothing to fade against; `blended` is a scratch raster the same shape as `clip`,
    /// used only when there is a predecessor; `out` is `[splice.frames(), rows, columns, 3]` bytes.
    ///
    /// The clip's tail is not written here unless `splice.last`: it is the state the caller hands the
    /// next clip, and passing it on is the runner's.
    pub fn forward(
        &self,
        f: &Forward,
        previous: Option<TensorView>,
        clip: TensorView,
        splice: &Splice,
        blended: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let raster = Raster::of(clip)?;
        raster.check_out(out, splice)?;
        let source = match previous {
            Some(previous) => {
                raster.check_previous(previous, self.dtype)?;
                raster.check_scratch(blended, self.dtype)?;
                self.fade.forward(f, Some(previous), clip, blended)?;
                blended
            }
            None => clip,
        };
        self.bytes.forward(f, raster.body(source, splice)?, out)?;
        if splice.last {
            self.bytes
                .forward(f, raster.written(source, splice.overlap)?, out)?;
        }
        Ok(())
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

    /// The blend's scratch, which must be a raster the clip's own shape and type.
    fn check_scratch(&self, blended: TensorView, dtype: DType) -> Result<()> {
        let shape = [
            u64::from(Rgb::CHANNELS),
            u64::from(CLIP_OUTPUT_FRAMES),
            u64::from(self.height),
            u64::from(self.width),
        ];
        if blended.shape.dims() != shape || blended.dtype != dtype {
            return Err(Error::config(format!(
                "the blend's scratch is {shape:?} of {dtype} and the view is {:?} of {}",
                blended.shape.dims(),
                blended.dtype
            )));
        }
        Ok(())
    }
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
