//! What the H3 decoders hand the media stage: the RGB raster and its splice into a timeline.
//!
//! The video decoder emits normalized RGB per channel, `[channels, frames, height, width]`, normalized
//! by the ImageNet constants ([`PIXEL_NORM`]); the runtime restores each placed run with
//! `value * std + mean`, clamps it to [0, 1] and quantizes it to the rgb24 bytes a media encoder
//! reads. The audio decoder writes its planar waveform ([`crate::audio_vae`]).
//!
//! ## The overlap
//!
//! Clips overlap by the decoder's tail ([`crate::video_vae::clips`]). A clip's first tail-length
//! body frames fade against its predecessor's tail with weights `i / overlap` (the frame rule of
//! [`nn::tiling::Blend`]), so the first blended frame is entirely the predecessor's.

use nn::capability::{PixelNorm, Placed};
use nn::tiling::{Axis, Blend, TemporalClips};
use nn::{DType, Forward, Operand};

use crate::{Error, Result};

/// The ImageNet constants H3's video rasters are normalized by, in red, green, blue order: the video
/// decoder's output and the video encoder's condition pixels alike. `value * std + mean` restores
/// the pixel intensity.
pub const PIXEL_NORM: PixelNorm = PixelNorm {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
};

/// Colour channels of the raster: red, green and blue.
pub const RGB_CHANNELS: u32 = 3;

/// The video postprocessor: one clip's decoded frames placed on the request's timeline.
///
/// **The decoder decodes a clip, this splices it.** Per clip it fades the clip's first body frames
/// against the predecessor's tail at weights `i / overlap` (the reference's `arange(5) / 5`, so the
/// first blended frame is entirely the predecessor's), places the kept frames, and leaves the tail
/// for the next clip; the clip that ends the timeline places its tail too. The fade is [`Blend`]'s
/// frame rule, which rounds each step in the raster's type.
pub struct VideoPostprocessor {
    dtype: DType,
    clips: TemporalClips,
    fade: Blend,
}

impl VideoPostprocessor {
    /// The fade over a clip's overlap of a raster of `dtype`, for the clips `clips` states.
    pub fn new(dtype: DType, clips: TemporalClips) -> Result<Self> {
        Ok(Self {
            dtype,
            clips,
            fade: Blend::new(dtype, clips.overlap, Axis::Frames)?,
        })
    }

    /// Place one clip's kept frames, fading against `previous` when it has a predecessor.
    ///
    /// `clip` is the clip's cropped decode `[3, native, rows, columns]`; `previous` is the
    /// predecessor's tail `[3, overlap, rows, columns]`, absent for the timeline's first clip; `last`
    /// states whether the clip ends the timeline and so places its tail after its body. Answers the
    /// runs of `[body (+ overlap when last)]` kept frames in order; the fade's output is a fresh
    /// tensor.
    pub fn forward(
        &self,
        f: &Forward,
        previous: Option<Operand>,
        clip: Operand,
        last: bool,
    ) -> Result<Vec<Placed>> {
        check_clip(clip, &self.clips)?;
        let body = self.clips.body;
        let tail = self.clips.tail();
        let mut placed = Vec::with_capacity(3);
        match previous {
            Some(previous) => {
                check_previous(clip, previous, self.dtype, self.clips.overlap)?;
                let fade = self.clips.overlap;
                let faded = self.fade.forward(f, previous, clip)?;
                placed.push(Placed {
                    raster: faded,
                    frame: 0,
                });
                placed.push(Placed {
                    raster: frames(clip, (fade, body))?,
                    frame: fade,
                });
            }
            None => placed.push(Placed {
                raster: frames(clip, (0, body))?,
                frame: 0,
            }),
        }
        if last {
            placed.push(Placed {
                raster: frames(clip, tail)?,
                frame: body,
            });
        }
        Ok(placed)
    }
}

impl nn::capability::Splicer for VideoPostprocessor {
    /// The type the raster it blends is stated in, which is the decoder's own.
    fn element_type(&self) -> nn::DType {
        self.dtype
    }

    /// Place one clip's kept frames on the timeline, fading it against its predecessor.
    ///
    /// `previous` is the predecessor's own decode of the frames the two share, absent for the
    /// timeline's first clip; `clip` is the decoder's raster, whose leading frames past the clip's
    /// native ones are cropped. The clip that ends the timeline keeps its tail, which the runner
    /// states with a `keep` past the body.
    fn splice(
        &self,
        f: &Forward,
        previous: Option<Operand>,
        clip: Operand,
        splice: nn::capability::Splice,
    ) -> nn::Result<Vec<Placed>> {
        if splice.overlap > self.clips.overlap {
            return Err(Error::config(format!(
                "{} frames of overlap, and the checkpoint fades {}",
                splice.overlap, self.clips.overlap
            ))
            .into());
        }
        let clip = cropped(clip, &self.clips)?;
        // `splice.overlap` counts the frames a clip fades against its **predecessor**; the tail is
        // the clip's own whatever it says, and the clip that ends the timeline keeps it.
        let last = splice.keep > self.clips.body;
        Ok(self.forward(f, previous, clip, last)?)
    }

    /// The ImageNet constants the placed rasters are normalized by.
    fn pixel_norm(&self) -> PixelNorm {
        PIXEL_NORM
    }

    /// The splices a timeline of `frames` frames is written in ([`crate::video_vae::splices`]).
    fn splices(&self, frames: u32) -> nn::Result<Vec<nn::capability::Splice>> {
        Ok(crate::video_vae::splices(&self.clips, frames)?)
    }
}

/// A decode's own frames, without the ones it leads with.
///
/// The decoder emits whole latent frames, so its raster runs past a clip's native frames, and the
/// surplus leads: the reference left-pads a clip to the temporal compression and crops those
/// frames from the decoded timeline (`decoded[:, :, padding:]`).
fn cropped(clip: Operand, clips: &TemporalClips) -> Result<Operand> {
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

/// Refuses a `clip` view that is not one clip's cropped decode, `[3, native, rows, columns]`.
fn check_clip(clip: Operand, clips: &TemporalClips) -> Result<()> {
    if clip.shape.rank() != 4 || clip.shape.dim(0) != u64::from(RGB_CHANNELS) {
        return Err(Error::config(format!(
            "a decoded clip is [{RGB_CHANNELS}, frames, rows, columns], not {:?}",
            clip.shape.dims()
        )));
    }
    if clip.shape.dim(1) != u64::from(clips.native()) {
        return Err(Error::config(format!(
            "a clip's decode is {} frames and the view carries {}",
            clips.native(),
            clip.shape.dim(1)
        )));
    }
    Ok(())
}

/// The view of frames `start..stop` of `clip`, `[channels, frames, rows, columns]` at the clip's
/// own strides: only the frames extent narrows.
fn frames(clip: Operand, (start, stop): (u32, u32)) -> Result<Operand> {
    let held = clip.shape.dim(1);
    if start >= stop || u64::from(stop) > held {
        return Err(Error::config(format!(
            "a clip's frames {start}..{stop} lie outside its {held} decoded frames"
        )));
    }
    Ok(clip.narrow(1, u64::from(start), u64::from(stop - start))?)
}

/// Refuses a predecessor's tail that is not one contiguous block of `clip`'s raster and `dtype`.
fn check_previous(clip: Operand, previous: Operand, dtype: DType, overlap: u32) -> Result<()> {
    let shape = [
        u64::from(RGB_CHANNELS),
        u64::from(overlap),
        clip.shape.dim(2),
        clip.shape.dim(3),
    ];
    if previous.shape.dims() != shape || previous.dtype != dtype {
        return Err(Error::config(format!(
            "a predecessor's tail is {shape:?} {dtype} and the view is {:?} {}",
            previous.shape.dims(),
            ops::abi::name(previous.dtype),
            dtype = ops::abi::name(dtype)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn::Shape;

    fn clips() -> TemporalClips {
        crate::video_vae::clips(&crate::video_vae::tests::video_config())
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
    ) -> (nn::Result<Vec<Placed>>, Vec<&'static str>, u64) {
        use nn::capability::Splicer;
        let (height, width) = (16_u64, 16_u64);
        let block = VideoPostprocessor::new(DType::F32, clips()).expect("a postprocessor");
        let arena = nn::Arena::new(nn::DevicePtr::new(0), 1 << 30);
        let held = nn::Arena::new(nn::DevicePtr::new(1 << 40), 1 << 30);
        let raster = held
            .tensor(DType::F32, Shape::new(&[3, 28, height, width]))
            .expect("the decoded raster");
        let tail = held
            .tensor(DType::F32, Shape::new(&[3, 5, height, width]))
            .expect("the predecessor's tail");
        let forward = nn::Forward::recording(&arena);
        let mut launches = Launches(Vec::new());
        ops::capture::set_sink(Some(&mut launches));
        let spliced = block.splice(&forward, previous.then_some(tail), raster, splice);
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
        let placed = spliced.expect("an interior clip");
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
        // The faded head, then the body's rest where the decoder wrote it.
        let frames: Vec<u32> = placed.iter().map(|run| run.frame).collect();
        assert_eq!(
            frames,
            [0, clips.overlap],
            "the faded head and the body's rest"
        );
        assert_eq!(placed[0].raster.shape.dim(1), u64::from(clips.overlap));
        assert_eq!(
            placed[1].raster.shape.dim(1),
            u64::from(clips.body - clips.overlap)
        );
    }

    /// The timeline's first clip fades nothing, and the only clip of a one-clip timeline still keeps
    /// its own tail: `overlap` counts the frames a clip fades against its **predecessor**.
    #[test]
    fn the_only_clip_of_a_timeline_keeps_its_own_tail() {
        let clips = clips();
        let schedule = clips.splices(1).expect("a one-clip timeline");
        let (spliced, launches, peak) = recorded_splice(false, schedule[0]);
        let placed = spliced.expect("the only clip");
        assert_eq!(peak, 0, "a clip with no predecessor reserves nothing");
        assert!(launches.is_empty(), "nothing to fade: {launches:?}");
        let frames: Vec<u32> = placed.iter().map(|run| run.frame).collect();
        assert_eq!(frames, [0, clips.body], "its body and its tail");
    }

    /// The clip raster test views are built on: one contiguous `[3, 25, rows, columns]` F16 block.
    fn raster_view(height: u32, width: u32) -> Operand {
        let shape = Shape::new(&[
            u64::from(RGB_CHANNELS),
            u64::from(clips().native()),
            u64::from(height),
            u64::from(width),
        ]);
        Operand::new(nn::DevicePtr::new(0), nn::DType::F16, shape)
    }

    #[test]
    fn a_clip_view_reads_the_body_and_the_tail_and_never_the_padding() {
        let clips = clips();
        let (height, width) = (4, 6);
        let clip = raster_view(height, width);
        check_clip(clip, &clips).expect("a clip raster");
        let per_frame = u64::from(height) * u64::from(width);
        let per_channel = u64::from(clips.native()) * per_frame;
        let body = frames(clip, (0, clips.body)).expect("a body");
        assert_eq!(
            body.strides.as_slice(),
            [per_channel, per_frame, u64::from(width), 1]
        );
        assert_eq!(body.element(&[1, 0, 0, 0]), clip.element(&[1, 0, 0, 0]));
        // The tail's first frame is the frame after the padding frames, with the same channel
        // stride.
        let tail = frames(clip, clips.tail()).expect("a tail");
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
        assert!(frames(clip, (clips.body, clips.native() + 1)).is_err());
        assert!(frames(clip, (clips.body, clips.body)).is_err());
    }

    #[test]
    fn a_clip_view_refuses_a_raster_that_is_not_one_clip() {
        let clips = clips();
        let short = Operand::new(
            nn::DevicePtr::new(0),
            nn::DType::F16,
            Shape::new(&[3, 24, 4, 6]),
        );
        assert!(check_clip(short, &clips).is_err());
        let grey = Operand::new(
            nn::DevicePtr::new(0),
            nn::DType::F16,
            Shape::new(&[1, u64::from(clips.native()), 4, 6]),
        );
        assert!(check_clip(grey, &clips).is_err());
    }
}
