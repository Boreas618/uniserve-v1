//! What the H3 decoders hand the media stage: the RGB raster and its splice into a timeline.
//!
//! The video decoder emits normalized RGB per channel; the output stage restores it with
//! `value * std + mean` — the ImageNet constants — then clamps it to [0, 1] and quantizes it to one
//! byte per channel. The audio decoder writes its own `[samples, 2]` PCM ([`crate::audio_vae`]).
//!
//! ## The video raster
//!
//! `[frames, height, width, 3]` **uint8**, row-major and channel-last: the rgb24 rasters a media
//! encoder reads, with values 0 to 255. A clip is decoded to normalized frames in `[channels,
//! frames, height, width]`; the output stage denormalizes, clamps and quantizes it, and moves the
//! channel axis last as it stores.
//!
//! ## The overlap
//!
//! Clips overlap by the decoder's tail ([`crate::video_vae::clips`]). A clip's first tail-length
//! body frames fade against its predecessor's tail with weights `i / overlap` (the frame rule of
//! [`nn::tiling::Blend`]), so the first blended frame is entirely the predecessor's.

use nn::Shape;
use nn::capability::StateTable;
use nn::layers::RgbBytes;
use nn::tiling::{Axis, Blend, TemporalClips};
use nn::{DType, Forward, TensorView, Weight};

use crate::{Error, Result};

/// The per-channel standard deviations the video decoder's RGB is normalized by, in red, green, blue
/// order: `value * std + mean` restores the pixel.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The per-channel means the video decoder's RGB is normalized by, in red, green, blue order.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// The one byte per channel the video raster stores.
pub const BYTE_MAX: f32 = 255.0;

/// Colour channels of the raster: red, green and blue.
pub const RGB_CHANNELS: u32 = 3;

/// The video postprocessor: one clip's decoded frames into the request's bytes.
///
/// **The decoder decodes a clip, this splices it.** Per clip it fades the clip's first body frames
/// against the predecessor's tail at weights `i / overlap` (the reference's `arange(5) / 5`, so the
/// first blended frame is entirely the predecessor's), writes the kept frames as
/// `[frames, rows, columns, 3]` bytes, and leaves the tail for the next clip; the clip that ends the
/// timeline writes its tail too. The fade is [`Blend`]'s frame rule, which rounds each step in the
/// raster's type; the de-normalization is [`RgbBytes`]'.
pub struct VideoPostprocessor {
    bytes: RgbBytes,
    dtype: DType,
    clips: TemporalClips,
    fade: Blend,
}

impl VideoPostprocessor {
    /// The fade over a clip's overlap and the byte packing of a raster of `dtype`, for the clips
    /// `clips` states.
    ///
    /// `mean` and `std` are the checkpoint's `[3]` de-normalization constants, published by the loader.
    pub fn new(dtype: DType, mean: Weight, std: Weight, clips: TemporalClips) -> Result<Self> {
        Ok(Self {
            bytes: RgbBytes::new(dtype, RGB_CHANNELS, mean, std)?,
            dtype,
            clips,
            fade: Blend::new(dtype, clips.overlap, Axis::Frames)?,
        })
    }

    /// Write one clip's kept frames into `out`, fading against `previous` when it has a predecessor.
    ///
    /// `clip` is the clip's cropped decode `[3, native, rows, columns]`; `previous` is the
    /// predecessor's tail `[3, overlap, rows, columns]`, absent for the timeline's first clip; `last`
    /// states whether the clip ends the timeline and so writes its tail after its body; `out` is
    /// `[body (+ overlap when last), rows, columns, 3]` bytes. The fade's output is reserved here.
    pub fn forward(
        &self,
        f: &Forward,
        previous: Option<TensorView>,
        clip: TensorView,
        last: bool,
        out: TensorView,
    ) -> Result<()> {
        let raster = Raster::of(clip, &self.clips)?;
        let body = self.clips.body;
        let tail = self.clips.tail();
        let kept = body + if last { self.clips.overlap } else { 0 };
        raster.check_out(out, kept)?;
        let body_out = frames_of(out, 0, body)?;
        match previous {
            Some(previous) => {
                raster.check_previous(previous, self.dtype, self.clips.overlap)?;
                let fade = self.clips.overlap;
                let faded = f.arena().tensor(clip.dtype, previous.shape)?;
                self.fade.forward(f, previous, clip, faded)?;
                self.bytes
                    .forward(f, faded, frames_of(body_out, 0, fade)?)?;
                self.bytes.forward(
                    f,
                    raster.written(clip, (fade, body))?,
                    frames_of(body_out, fade, body)?,
                )?;
            }
            None => self
                .bytes
                .forward(f, raster.written(clip, (0, body))?, body_out)?,
        }
        if last {
            self.bytes
                .forward(f, raster.written(clip, tail)?, frames_of(out, body, kept)?)?;
        }
        Ok(())
    }
}

impl nn::capability::Decoder for VideoPostprocessor {
    /// The type the raster it blends is stated in, which is the decoder's own.
    fn element_type(&self) -> nn::DType {
        self.dtype
    }

    /// No tables: the splice's parameters are its arguments, not state.
    fn layout_tables(&self, _layout: &nn::capability::Layout) -> nn::Result<Vec<StateTable>> {
        Ok(Vec::new())
    }

    /// Nothing to prepare: there are no tables, so `out` must be empty.
    fn prepare_layout(
        &self,
        _layout: &nn::capability::Layout,
        out: &mut [nn::TableValues],
    ) -> nn::Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        Err(Error::config(format!(
            "a postprocessor has no layout tables, and {} buffers were offered",
            out.len()
        ))
        .into())
    }

    /// Refuse: the postprocessor has no latents to decode.
    fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        state: &[TensorView],
        out: nn::capability::DecodedMedia,
    ) -> nn::Result<()> {
        let _ = (f, latents, state, out);
        Err(Error::config("a postprocessor splices; it decodes nothing".to_owned()).into())
    }

    /// Write one clip's kept frames into the timeline, fading it against its predecessor.
    ///
    /// `previous` is the predecessor's own decode of the frames the two share, absent for the
    /// timeline's first clip; `clip` is the decoder's raster, whose leading frames past the clip's
    /// native ones are cropped; `out.video` is the whole timeline's byte raster
    /// `[frames, rows, columns, channels]`, whose frames at `splice.first` this writes. The clip that
    /// ends the timeline keeps its tail, which the runner states with a `keep` past the body.
    fn splice(
        &self,
        f: &Forward,
        previous: Option<TensorView>,
        clip: TensorView,
        splice: nn::capability::Splice,
        out: nn::capability::DecodedMedia,
    ) -> nn::Result<()> {
        let video = out
            .video
            .ok_or_else(|| Error::config("a splice writes a timeline's raster".to_owned()))?;
        if out.audio.is_some() {
            return Err(Error::config("a splice writes no audio".to_owned()).into());
        }
        if splice.overlap > self.clips.overlap {
            return Err(Error::config(format!(
                "{} frames of overlap, and the checkpoint fades {}",
                splice.overlap, self.clips.overlap
            ))
            .into());
        }
        let clip = Raster::cropped(clip, &self.clips)?;
        let raster = Raster::of(clip, &self.clips)?;
        // `splice.overlap` counts the frames a clip fades against its **predecessor**; the tail is
        // the clip's own whatever it says, and the clip that ends the timeline keeps it.
        let last = splice.keep > self.clips.body;
        let kept = self.clips.body + if last { self.clips.overlap } else { 0 };
        let view = TensorView::new(
            video.element(&[u64::from(splice.first), 0, 0, 0]),
            DType::U8,
            Shape::new(&[
                u64::from(kept),
                u64::from(raster.height),
                u64::from(raster.width),
                u64::from(RGB_CHANNELS),
            ]),
        );
        Ok(self.forward(f, previous, clip, last, view)?)
    }

    /// The splices a timeline of `frames` frames is written in ([`TemporalClips::splices`]).
    fn splices(&self, frames: u32) -> nn::Result<Vec<nn::capability::Splice>> {
        self.clips.splices(frames)
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
    /// A decode's own frames, without the ones it leads with.
    ///
    /// The decoder emits whole latent frames, so its raster runs past a clip's native frames, and
    /// the surplus leads: the reference left-pads a clip to the temporal compression and crops those
    /// frames from the decoded timeline (`decoded[:, :, padding:]`).
    fn cropped(clip: TensorView, clips: &TemporalClips) -> Result<TensorView> {
        let native = u64::from(clips.native());
        if clip.shape.rank() != 4 || clip.shape.dim(1) < native {
            return Err(Error::config(format!(
                "a decoded clip is [channels, at least {native} frames, rows, columns], not {:?}",
                clip.shape.dims()
            )));
        }
        let leading = clip.shape.dim(1) - native;
        Ok(clip.narrow(1, leading, native)?)
    }

    /// The raster `[channels, native, rows, columns]` a `clip` view describes.
    fn of(clip: TensorView, clips: &TemporalClips) -> Result<Self> {
        if clip.shape.rank() != 4 || clip.shape.dim(0) != u64::from(RGB_CHANNELS) {
            return Err(Error::config(format!(
                "a decoded clip is [{RGB_CHANNELS}, frames, rows, columns], not {:?}",
                clip.shape.dims()
            )));
        }
        let frames = u32::try_from(clip.shape.dim(1))
            .map_err(|_| Error::config("a clip taller than a frame count".to_owned()))?;
        if frames != clips.native() {
            return Err(Error::config(format!(
                "a clip's decode is {} frames and the view carries {frames}",
                clips.native()
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

    /// The view of `range` frames of `raster`, as `[channels, frames, rows, columns]`, at the clip's
    /// own strides: only the frames extent narrows.
    fn written(&self, raster: TensorView, range: (u32, u32)) -> Result<TensorView> {
        let (start, stop) = range;
        if start >= stop || stop > self.frames {
            return Err(Error::config(format!(
                "a clip's frames {start}..{stop} lie outside its {} decoded frames",
                self.frames
            )));
        }
        Ok(raster.narrow(1, u64::from(start), u64::from(stop - start))?)
    }

    /// The output view of one clip: `[kept, rows, columns, channels]` bytes.
    fn check_out(&self, out: TensorView, kept: u32) -> Result<()> {
        let shape = [
            u64::from(kept),
            u64::from(self.height),
            u64::from(self.width),
            u64::from(RGB_CHANNELS),
        ];
        if out.dtype != DType::U8 || out.shape.dims() != shape {
            return Err(Error::config(format!(
                "the output raster is {shape:?} bytes and the view is {} {:?}",
                out.dtype,
                out.shape.dims()
            )));
        }
        Ok(())
    }

    /// The predecessor's tail, which must be one contiguous block of the clip's own type.
    fn check_previous(&self, previous: TensorView, dtype: DType, overlap: u32) -> Result<()> {
        let shape = [
            u64::from(RGB_CHANNELS),
            u64::from(overlap),
            u64::from(self.height),
            u64::from(self.width),
        ];
        if previous.shape.dims() != shape || previous.dtype != dtype {
            return Err(Error::config(format!(
                "a predecessor's tail is {shape:?} {dtype} and the view is {:?} {}",
                previous.shape.dims(),
                previous.dtype
            )));
        }
        Ok(())
    }
}

/// The frames `start..stop` of an output raster: `[frames, rows, columns, channels]` bytes, at the
/// raster's own strides.
fn frames_of(out: TensorView, start: u32, stop: u32) -> Result<TensorView> {
    let shape = out.shape;
    if shape.rank() != 4 || start >= stop || shape.dim(0) < u64::from(stop) {
        return Err(Error::config(format!(
            "frames {start}..{stop} of a {}-frame output raster do not exist",
            shape.dim(0)
        )));
    }
    Ok(out.narrow(0, u64::from(start), u64::from(stop - start))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clips() -> TemporalClips {
        crate::video_vae::clips(&crate::video_vae::tests::video_config())
    }

    fn constants(values: [f32; 3]) -> Weight {
        // The constants are host bytes the worker uploads; a test only needs a weight that names
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

    /// The operators a splice launches, in order.
    struct Launches(Vec<&'static str>);

    impl ops::capture::Sink for Launches {
        fn launched(&mut self, operator: &'static str, _key: &str) {
            self.0.push(operator);
        }
    }

    /// One recorded splice of a decoder's 28-frame raster: the launches and the arena's peak.
    fn recorded_splice(
        previous: bool,
        splice: nn::capability::Splice,
    ) -> (nn::Result<()>, Vec<&'static str>, u64) {
        use nn::capability::{DecodedMedia, Decoder};
        let (height, width) = (16_u64, 16_u64);
        let block = VideoPostprocessor::new(
            DType::F32,
            constants(PIXEL_MEAN),
            constants(PIXEL_STD),
            clips(),
        )
        .expect("a postprocessor");
        let arena = nn::Arena::new(nn::DevicePtr::new(0), 1 << 30);
        let held = nn::Arena::new(nn::DevicePtr::new(1 << 40), 1 << 30);
        let raster = held
            .tensor(DType::F32, Shape::new(&[3, 28, height, width]))
            .expect("the decoded raster");
        let tail = held
            .tensor(DType::F32, Shape::new(&[3, 5, height, width]))
            .expect("the predecessor's tail");
        let video = held
            .tensor(
                DType::U8,
                Shape::new(&[u64::from(splice.keep), height, width, 3]),
            )
            .expect("the timeline's frames");
        let forward = nn::Forward::recording(&arena);
        let mut launches = Launches(Vec::new());
        ops::capture::set_sink(Some(&mut launches));
        let spliced = block.splice(
            &forward,
            previous.then_some(tail),
            raster,
            splice,
            DecodedMedia {
                video: Some(video),
                audio: None,
            },
        );
        ops::capture::set_sink(None);
        (spliced, launches.0, arena.peak())
    }

    /// A clip after the first fades the head of its body and reads the rest where the decoder wrote
    /// it: the fade writes the overlap frames and one frame of scratch, and no frame of the body is
    /// read from a buffer the fade never wrote.
    #[test]
    fn a_fade_writes_the_overlap_frames_and_the_body_is_read_from_the_clip() {
        let clips = clips();
        let (spliced, launches, peak) = recorded_splice(
            true,
            nn::capability::Splice {
                first: 0,
                keep: clips.body,
                overlap: clips.overlap,
            },
        );
        spliced.expect("an interior clip");
        let faded = u64::from(RGB_CHANNELS) * u64::from(clips.overlap) * 16 * 16 * 4;
        let term = 16 * 16 * 4_u64;
        assert_eq!(
            peak,
            faded.next_multiple_of(nn::forward::ALIGNMENT)
                + term.next_multiple_of(nn::forward::ALIGNMENT)
        );
        let count = |operator| launches.iter().filter(|name| **name == operator).count();
        assert_eq!(
            count("scaled_add") as u32,
            3 * RGB_CHANNELS * clips.overlap,
            "two products and a sum per frame of every channel: {launches:?}"
        );
        assert_eq!(count("rgb_bytes"), 2, "the faded head and the body's rest");
    }

    /// The timeline's first clip fades nothing, and the only clip of a one-clip timeline still keeps
    /// its own tail: `overlap` counts the frames a clip fades against its **predecessor**.
    #[test]
    fn the_only_clip_of_a_timeline_keeps_its_own_tail() {
        let clips = clips();
        let schedule = clips
            .splices(clips.body + clips.overlap)
            .expect("a one-clip timeline");
        let (spliced, launches, peak) = recorded_splice(false, schedule[0]);
        spliced.expect("the only clip");
        assert_eq!(peak, 0, "a clip with no predecessor reserves nothing");
        assert_eq!(
            launches,
            ["rgb_bytes", "rgb_bytes"],
            "its body and its tail"
        );
    }

    /// The clip raster test views are built on: one contiguous `[3, 25, rows, columns]` F16 block.
    fn raster_view(height: u32, width: u32) -> TensorView {
        let shape = Shape::new(&[
            u64::from(RGB_CHANNELS),
            u64::from(clips().native()),
            u64::from(height),
            u64::from(width),
        ]);
        TensorView::new(nn::DevicePtr::new(0), nn::DType::F16, shape)
    }

    #[test]
    fn a_clip_view_reads_the_body_and_the_tail_and_never_the_padding() {
        let clips = clips();
        let (height, width) = (4, 6);
        let clip = raster_view(height, width);
        let raster = Raster::of(clip, &clips).expect("a clip raster");
        let per_frame = u64::from(height) * u64::from(width);
        let per_channel = u64::from(clips.native()) * per_frame;
        let body = raster.written(clip, (0, clips.body)).expect("a body");
        assert_eq!(
            body.strides.as_slice(),
            [per_channel, per_frame, u64::from(width), 1]
        );
        assert_eq!(body.element(&[1, 0, 0, 0]), clip.element(&[1, 0, 0, 0]));
        // The tail's first frame is the frame after the padding frames, with the same channel
        // stride.
        let tail = raster.written(clip, clips.tail()).expect("a tail");
        assert_eq!(tail.shape.dim(1), u64::from(clips.overlap));
        assert_eq!(
            tail.element(&[0, 0, 0, 0]),
            clip.element(&[0, u64::from(clips.native() - clips.overlap), 0, 0])
        );
        assert_ne!(
            body.element(&[0, u64::from(clips.body) - 1, 0, 0]),
            tail.element(&[0, 0, 0, 0])
        );
        // A range that leaves the clip is refused rather than read past it.
        assert!(
            raster
                .written(clip, (clips.body, clips.native() + 1))
                .is_err()
        );
        assert!(raster.written(clip, (clips.body, clips.body)).is_err());
    }

    #[test]
    fn a_clip_view_refuses_a_raster_that_is_not_one_clip() {
        let clips = clips();
        let short = TensorView::new(
            nn::DevicePtr::new(0),
            nn::DType::F16,
            Shape::new(&[3, 24, 4, 6]),
        );
        assert!(Raster::of(short, &clips).is_err());
        let grey = TensorView::new(
            nn::DevicePtr::new(0),
            nn::DType::F16,
            Shape::new(&[1, u64::from(clips.native()), 4, 6]),
        );
        assert!(Raster::of(grey, &clips).is_err());
        let raster = Raster::of(raster_view(4, 6), &clips).expect("a clip raster");
        let kept = clips.body + clips.overlap;
        let out = |frames: u32| {
            TensorView::new(
                nn::DevicePtr::new(0),
                nn::DType::U8,
                Shape::new(&[u64::from(frames), 4, 6, 3]),
            )
        };
        assert!(raster.check_out(out(kept), kept).is_ok());
        assert!(raster.check_out(out(kept - 1), kept).is_err());
    }
}
