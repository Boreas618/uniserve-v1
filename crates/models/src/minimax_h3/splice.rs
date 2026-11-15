//! The clip splice: one decoded clip's frames placed on a request's timeline.
//!
//! The video decoder decodes a clip and stops. Clips overlap by the decoder's tail
//! ([`crate::minimax_h3::video_vae::temporal_clips`]): a clip's first tail-length frames fade against its
//! predecessor's tail with the weights `i / overlap`, so the first faded frame is entirely the
//! predecessor's (the reference's `blend(dec_overlap, clip_dec_chunk, frame_overlap, dim=-3)`,
//! `sglang:python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py:882-897`).
//! The fade is the one blend the tiles' seams use ([`nn::vae::blend`]), which rounds each step
//! in the raster's element type.

use nn::vae::{self, TemporalClips};
use nn::{DType, PixelNorm, Result, Splice, Splicer, Tensor};

use crate::minimax_h3::ID;
use crate::minimax_h3::video_vae::{self, PIXEL_NORM};

/// The channels of a decoded raster: red, green and blue.
const RGB_CHANNELS: usize = 3;

/// The video postprocessor: the splice of decoded clips into one timeline.
///
/// It holds no weights: the element type of the rasters it reads and the temporal clips of the
/// checkpoint. Per clip it crops the frames the decode leads with, fades the clip's first
/// frames against its predecessor's tail, and returns the runs of frames the timeline keeps;
/// the clip that ends the timeline keeps its tail too.
pub struct VideoPostprocessor {
    dtype: DType,
    clips: TemporalClips,
}

impl VideoPostprocessor {
    /// The splice of rasters of `dtype` over the clips `clips` states.
    ///
    /// # Errors
    ///
    /// Clips with no body, or with a tail longer than the body it fades into.
    pub fn new(dtype: DType, clips: TemporalClips) -> Result<Self> {
        if clips.body == 0 || clips.overlap > clips.body || clips.temporal == 0 {
            return Err(ID.config(format!(
                "{clips:?} are not clips whose tail fades into the next clip's body"
            )));
        }

        Ok(Self { dtype, clips })
    }

    /// The runs of `clip`'s kept frames ([`Splicer::splice`]).
    fn runs(
        &self,
        previous: Option<&Tensor>,
        clip: &Tensor,
        splice: Splice,
    ) -> Result<Vec<(Tensor, u32)>> {
        let clips = &self.clips;
        let (body, overlap, native) = (
            clips.body as usize,
            clips.overlap as usize,
            clips.native() as usize,
        );

        let &[RGB_CHANNELS, decoded, height, width] = clip.dims() else {
            return Err(ID.request(format!(
                "a decoded clip is [{RGB_CHANNELS}, frames, rows, columns], not {:?}",
                clip.dims()
            )));
        };
        if decoded < native || clip.dtype() != self.dtype {
            return Err(ID.request(format!(
                "a decoded clip holds at least {native} frames of {:?}, not {decoded} of {:?}",
                self.dtype,
                clip.dtype()
            )));
        }

        // A clip that ends the timeline keeps its tail after its body; any other keeps its
        // body alone.
        let last = splice.keep == clips.body + clips.overlap;
        if splice.overlap > clips.overlap || !(last || splice.keep == clips.body) {
            return Err(ID.request(format!(
                "{splice:?} keeps and fades other frames than a clip of {clips:?} has"
            )));
        }

        // The decoder writes whole latent frames, so its raster leads with the frames beyond
        // the clip's own, which the reference crops (`frame_pre_padding`).
        let clip = clip.narrow(1, decoded - native, native)?;

        let mut runs = Vec::with_capacity(3);
        match previous {
            None => runs.push((clip.narrow(1, 0, body)?, 0)),

            Some(previous) => {
                if previous.dims() != [RGB_CHANNELS, overlap, height, width]
                    || previous.dtype() != self.dtype
                {
                    return Err(ID.request(format!(
                        "a predecessor's tail is {:?} [{RGB_CHANNELS}, {overlap}, {height}, \
                         {width}], not {:?} {:?}",
                        self.dtype,
                        previous.dtype(),
                        previous.dims()
                    )));
                }

                // The faded head is computed; the rest of the body is read where the decoder
                // wrote it.
                let head = clip.narrow(1, 0, overlap)?;
                runs.push((vae::blend(previous, &head, overlap, 1)?, 0));
                if body > overlap {
                    runs.push((clip.narrow(1, overlap, body - overlap)?, clips.overlap));
                }
            }
        }

        if last {
            let tail = clips.tail();
            runs.push((clip.narrow(1, tail.start as usize, tail.len())?, clips.body));
        }
        Ok(runs)
    }
}

impl Splicer for VideoPostprocessor {
    fn element_type(&self) -> DType {
        self.dtype
    }

    /// The ImageNet constants the decoder's rasters are normalized by.
    fn pixel_norm(&self) -> PixelNorm {
        PIXEL_NORM
    }

    /// One splice per clip of the frame rule ([`video_vae::splices`]).
    fn splices(&self, frames: u32) -> nn::Result<Vec<Splice>> {
        video_vae::splices(&self.clips, frames)
    }

    /// Places one clip's kept frames, fading its first frames against `previous` when it has a
    /// predecessor.
    ///
    /// `clip` is the decoder's raster, `[3, frames, rows, columns]` with at least the clip's
    /// own frames, and `previous` the predecessor's tail, `[3, overlap, rows, columns]`. The
    /// runs are the faded head from frame zero, the rest of the body from frame `overlap`,
    /// and, for the clip that ends the timeline (a `keep` of the body and the tail), the tail
    /// from frame `body`.
    fn splice(
        &self,
        previous: Option<&Tensor>,
        clip: &Tensor,
        splice: Splice,
    ) -> nn::Result<Vec<(Tensor, u32)>> {
        self.runs(previous, clip, splice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_h3::VideoVaeConfig;
    use crate::minimax_h3::fixtures;
    use nn::Device;

    fn clips() -> TemporalClips {
        let config = VideoVaeConfig::parse(&fixtures::video_vae()).expect("the sidecar");
        video_vae::temporal_clips(&config)
    }

    /// A `[3, frames, 2, 2]` FP32 raster whose every element is its frame's index plus `base`.
    fn raster(frames: usize, base: f32) -> Tensor {
        Tensor::arange(0_f32, frames as f32, &Device::Cpu)
            .and_then(|index| index.affine(1.0, f64::from(base)))
            .and_then(|index| index.reshape((1, frames, 1, 1)))
            .and_then(|index| index.broadcast_as((RGB_CHANNELS, frames, 2, 2)))
            .and_then(|raster| raster.contiguous())
            .expect("a raster")
    }

    /// The first element of each frame of a `[3, frames, 2, 2]` raster.
    fn frames(raster: &Tensor) -> Vec<f32> {
        raster
            .narrow(0, 0, 1)
            .and_then(|raster| raster.narrow(2, 0, 1))
            .and_then(|raster| raster.narrow(3, 0, 1))
            .and_then(|raster| raster.flatten_all())
            .and_then(|raster| raster.to_vec1())
            .expect("a CPU raster")
    }

    #[test]
    fn the_only_clip_of_a_timeline_keeps_its_body_and_its_tail_and_never_the_padding() {
        let clips = clips();
        let splicer = VideoPostprocessor::new(DType::F32, clips).expect("a postprocessor");
        let schedule = splicer.splices(22).expect("a one-clip timeline");

        // The decoder's 28 frames lead with three the clip does not own; frame `f` of the
        // clip's own 25 holds `3 + f`.
        let runs = splicer
            .splice(None, &raster(28, 0.0), schedule[0])
            .expect("the only clip");
        assert_eq!(runs.len(), 2);

        // The body: the clip's frames 0..17.
        assert_eq!(runs[0].1, 0);
        assert_eq!(
            frames(&runs[0].0),
            (3..20).map(|f| f as f32).collect::<Vec<_>>()
        );

        // The tail: the clip's frames 20..25, past the three padding frames, placed at 17.
        assert_eq!(runs[1].1, clips.body);
        assert_eq!(
            frames(&runs[1].0),
            (23..28).map(|f| f as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_later_clip_fades_its_first_frames_against_its_predecessors_tail() {
        let clips = clips();
        let splicer = VideoPostprocessor::new(DType::F32, clips).expect("a postprocessor");
        let schedule = splicer.splices(39).expect("a two-clip timeline");

        // The predecessor's tail is all 100; the clip's own frame `f` holds `3 + f`.
        let tail = raster(5, 100.0)
            .affine(0.0, 100.0)
            .expect("a constant tail");
        let runs = splicer
            .splice(Some(&tail), &raster(28, 0.0), schedule[1])
            .expect("the last clip");
        assert_eq!(
            runs.iter().map(|run| run.1).collect::<Vec<_>>(),
            [0, clips.overlap, clips.body],
            "the faded head, the rest of the body, the tail"
        );

        // Faded frame `i` is `tail · (1 − i/5) + clip · (i/5)`: the first is the predecessor's.
        let faded = frames(&runs[0].0);
        for (index, value) in faded.iter().enumerate() {
            let ratio = index as f32 / 5.0;
            let expected = 100.0 * (1.0 - ratio) + (3.0 + index as f32) * ratio;
            assert!(
                (value - expected).abs() <= 1e-4,
                "faded frame {index}: {value} against {expected}"
            );
        }
        assert_eq!(faded[0], 100.0);

        // The rest of the body is the decoder's own frames 5..17.
        assert_eq!(
            frames(&runs[1].0),
            (8..20).map(|f| f as f32).collect::<Vec<_>>()
        );
        assert_eq!(runs[2].0.dims(), [3, 5, 2, 2]);
    }

    #[test]
    fn a_raster_that_is_not_one_clip_is_refused() {
        let clips = clips();
        let splicer = VideoPostprocessor::new(DType::F32, clips).expect("a postprocessor");
        let schedule = splicer.splices(39).expect("a two-clip timeline");

        // Fewer frames than the clip's own, another element type, a tail of another length,
        // and a splice no clip of the checkpoint has.
        assert!(splicer.splice(None, &raster(24, 0.0), schedule[0]).is_err());

        let narrow = raster(28, 0.0).to_dtype(DType::F16).expect("FP16");
        assert!(splicer.splice(None, &narrow, schedule[0]).is_err());

        let short = raster(4, 0.0);
        assert!(
            splicer
                .splice(Some(&short), &raster(28, 0.0), schedule[1])
                .is_err()
        );

        let longer = Splice {
            keep: clips.body + 1,
            ..schedule[0]
        };
        assert!(splicer.splice(None, &raster(28, 0.0), longer).is_err());
    }
}
