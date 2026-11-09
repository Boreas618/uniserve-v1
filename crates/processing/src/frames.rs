//! The frame rule of a video family whose VAE encodes a video in clips: which frame counts it
//! generates, how many latent frames a count encodes to, and how a condition is cut into clips.
//!
//! A family states the four values of [`FrameClips`] once, and every count the request processing,
//! the denoiser's layouts, the encoders and the decoders derive from a frame count follows from
//! them here.

use serde::{Deserialize, Serialize};

/// How a model's video VAE divides a video's frames into clips.
///
/// A generated video has `min_frames + n * clip_frames` frames: one decode unit per clip, so
/// `n + 1` units. The VAE encodes each clip to `clip_latent_frames` latent frames and the tail past
/// the whole clips to `tail_latent_frames`; the final clip drops the rest of its latent frames
/// ([`Self::token_drop`]), which the next clip's overlap would repeat. The tail is
/// `min_frames - clip_frames` frames ([`Self::tail_frames`]), at least one and at most one clip.
///
/// A condition is not generated but cut: any frame count is cut into clips of `clip_frames` frames,
/// the last one shorter, and encodes to `clip_latent_frames` latent frames per clip less the
/// dropped ones ([`Self::latent_frames`]); a single frame is a still and encodes to one. The two
/// counts agree on every frame count the rule generates.
///
/// For the released MiniMax-H3 checkpoint the rule is `17n + 5` frames, five latent frames a clip
/// and two for the tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameClips {
    /// Fewest frames of a video: one clip and the tail.
    pub min_frames: u32,
    /// Frames each further clip adds.
    pub clip_frames: u32,
    /// Latent frames one clip encodes to.
    pub clip_latent_frames: u32,
    /// Latent frames of the tail past the clips.
    pub tail_latent_frames: u32,
}

/// One clip a condition is cut into: the frames it reads and the latent frames it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipSpan {
    /// The clip's first frame.
    pub first: u32,
    /// Frames the clip reads.
    pub frames: u32,
    /// The clip's first latent frame.
    pub latent: u32,
    /// Latent frames the clip writes.
    pub latents: u32,
}

impl FrameClips {
    /// Frames of the tail past the whole clips, which a decoded clip hands its successor to fade
    /// against.
    pub const fn tail_frames(&self) -> u32 {
        self.min_frames.saturating_sub(self.clip_frames)
    }

    /// Latent frames the final clip of a video drops: what one clip yields past the tail's.
    pub const fn token_drop(&self) -> u32 {
        self.clip_latent_frames
            .saturating_sub(self.tail_latent_frames)
    }

    /// Whether the rule generates videos of `frames` frames.
    pub const fn is_legal(&self, frames: u32) -> bool {
        frames >= self.min_frames && (frames - self.min_frames).is_multiple_of(self.clip_frames)
    }

    /// The decode units of a video of `frames` legal frames: one per clip.
    pub const fn units(&self, frames: u32) -> u32 {
        frames.saturating_sub(self.min_frames) / self.clip_frames + 1
    }

    /// The fewest legal frame count that is at least `requested`: [`Self::min_frames`] for any
    /// request of fewer.
    pub const fn align_up(&self, requested: u32) -> u32 {
        let beyond = requested.saturating_sub(self.min_frames);
        self.min_frames + beyond.div_ceil(self.clip_frames) * self.clip_frames
    }

    /// The most legal frames of at most `available`, or `None` below the fewest.
    pub const fn align_down(&self, available: u32) -> Option<u32> {
        let Some(beyond) = available.checked_sub(self.min_frames) else {
            return None;
        };
        Some(self.min_frames + beyond / self.clip_frames * self.clip_frames)
    }

    /// The latent frames the VAE encodes `frames` frames to.
    ///
    /// The frames are cut into clips of [`Self::clip_frames`], each yielding
    /// [`Self::clip_latent_frames`] latent frames, and the last clip's trailing
    /// [`Self::token_drop`] latent frames are dropped: `ceil(frames / clip) * latents - drop`, so for
    /// the released rule 17 frames are 2 latent frames, 22 are 7 and 30 are 7. A single frame is a
    /// still and encodes to one: the network's causal front padding makes one frame one latent frame.
    /// No frame encodes to none.
    ///
    /// On a legal count this is `units * clip_latent_frames + tail_latent_frames`, the generated
    /// video's latent timeline.
    pub const fn latent_frames(&self, frames: u32) -> u32 {
        match frames {
            0 => 0,
            1 => 1,
            _ => frames.div_ceil(self.clip_frames) * self.clip_latent_frames - self.token_drop(),
        }
    }

    /// The legal frame count whose latent timeline is `latent_frames` long, the inverse of
    /// [`Self::latent_frames`] on legal counts; `None` for a latent count no legal count encodes to.
    pub const fn frames_of_latents(&self, latent_frames: u32) -> Option<u32> {
        let Some(clips) = latent_frames.checked_sub(self.tail_latent_frames) else {
            return None;
        };
        if clips == 0 || !clips.is_multiple_of(self.clip_latent_frames) {
            return None;
        }
        Some(self.min_frames + (clips / self.clip_latent_frames - 1) * self.clip_frames)
    }

    /// The clips a condition of `frames` frames is cut into, in order.
    ///
    /// Each clip is encoded alone: the frames it reads and the latent frames it writes, the last
    /// clip holding what is left of both ([`Self::latent_frames`]). A still is one clip of one frame;
    /// no frame is no clip.
    pub fn clips(&self, frames: u32) -> Vec<ClipSpan> {
        let total = self.latent_frames(frames);
        let mut spans = Vec::with_capacity(frames.div_ceil(self.clip_frames.max(1)) as usize);
        let (mut first, mut latent) = (0, 0);
        while first < frames {
            let length = self.clip_frames.min(frames - first);
            let latents = self.clip_latent_frames.min(total - latent);
            spans.push(ClipSpan {
                first,
                frames: length,
                latent,
                latents,
            });
            first += length;
            latent += latents;
        }
        spans
    }

    /// Whether the rule is positive, leaves a tail of at most one clip past the first clip, and
    /// keeps some of the final clip's latent frames.
    pub const fn is_valid(&self) -> bool {
        self.clip_frames > 0
            && self.min_frames > self.clip_frames
            && self.tail_frames() <= self.clip_frames
            && self.clip_latent_frames > 0
            && self.tail_latent_frames > 0
            && self.tail_latent_frames <= self.clip_latent_frames
    }
}

/// Frames of `seconds` of video at `fps`: the product rounded half to even, as the reference
/// implementations round with Python's `round`.
///
/// The caller bounds `seconds`; a product beyond `u32` saturates.
pub fn requested_frames(seconds: f64, fps: u32) -> u32 {
    (seconds * f64::from(fps)).round_ties_even() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released MiniMax-H3 rule: `17n + 5` frames, five latent frames a clip and two for the
    /// tail.
    const RULE: FrameClips = FrameClips {
        min_frames: 22,
        clip_frames: 17,
        clip_latent_frames: 5,
        tail_latent_frames: 2,
    };

    #[test]
    fn a_duration_aligns_up_to_the_next_generated_count() {
        assert_eq!(RULE.align_up(requested_frames(4.0, 24)), 107);
        assert_eq!(RULE.align_up(requested_frames(5.0, 24)), 124);
        assert_eq!(RULE.align_up(requested_frames(15.0, 24)), 362);
        assert_eq!(RULE.align_up(22), 22);
        assert_eq!(RULE.align_up(23), 39);
        // Fewer frames than one clip and its tail generate one clip and its tail.
        assert_eq!(RULE.align_up(1), 22);
        assert_eq!(RULE.align_down(21), None);
        assert_eq!(RULE.align_down(38), Some(22));
        assert!(RULE.is_legal(124) && !RULE.is_legal(123) && !RULE.is_legal(5));
    }

    #[test]
    fn requested_frames_round_half_to_even() {
        assert_eq!(requested_frames(2.5, 1), 2);
        assert_eq!(requested_frames(3.5, 1), 4);
        assert_eq!(requested_frames(4.0, 24), 96);
    }

    #[test]
    fn a_count_encodes_to_whole_clips_less_the_dropped_latents() {
        assert_eq!(RULE.token_drop(), 3);
        assert_eq!(RULE.tail_frames(), 5);
        assert_eq!(RULE.latent_frames(1), 1, "a still");
        assert_eq!(RULE.latent_frames(17), 2);
        assert_eq!(RULE.latent_frames(30), 7);
        // A generated count's latent timeline: five a clip and the tail's two, which the inverse
        // returns to the count.
        for frames in (22..=400).filter(|frames| RULE.is_legal(*frames)) {
            let latents = RULE.latent_frames(frames);
            assert_eq!(latents, RULE.units(frames) * 5 + 2);
            assert_eq!(RULE.frames_of_latents(latents), Some(frames));
        }
        assert_eq!(RULE.frames_of_latents(2), None);
        assert_eq!(RULE.frames_of_latents(8), None);
    }

    #[test]
    fn a_condition_is_cut_into_clips_that_partition_its_latents() {
        let spans = RULE.clips(30);
        assert_eq!(
            spans,
            [
                ClipSpan {
                    first: 0,
                    frames: 17,
                    latent: 0,
                    latents: 5
                },
                ClipSpan {
                    first: 17,
                    frames: 13,
                    latent: 5,
                    latents: 2
                },
            ]
        );
        assert_eq!(
            RULE.clips(1),
            [ClipSpan {
                first: 0,
                frames: 1,
                latent: 0,
                latents: 1
            }]
        );
        assert!(RULE.clips(0).is_empty());
    }
}
