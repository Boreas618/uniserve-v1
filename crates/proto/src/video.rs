//! The conditions of a video request, as they travel from request processing to the workers.
//!
//! A video model's request processing prepares each condition of a request (`processing`): the
//! length of its fetched media ([`Condition::bytes`]; the media themselves travel beside the
//! request, one memory file per condition, `crate::media`), what the media reader decodes them into
//! ([`ConditionMedia`]), what the text encoder's vision tower reads of it ([`ConditionVision`])
//! and the denoiser rows its latent encoders write. The engine carries a request's
//! [`Conditions`] encoded as their Protobuf message (`request.proto`'s `VideoConditions`,
//! converted by `wire`) beside the sizes it schedules by and never decodes them; the workers decode
//! them and hand each stage its part. The vocabulary is the video pipeline's, not
//! one model's: any video model whose request processing prepares a condition states it in these
//! terms.
//!
//! The latent [`Layout`] a denoiser prepares for one request is stated here too: the model states
//! it, the worker captures a graph for it and the scheduler names it in the batch it submits.
//!
//! The media a condition is decoded into ([`ConditionMedia`] and the rasters, clips and grids it
//! holds) is the processing program's vocabulary too: a family's request processing states it
//! through the processing interface, which owns it ([`scheduling::processing::video`]), and every
//! stage after it reads the same types here.

mod wire;

use serde::{Deserialize, Serialize};

pub use scheduling::processing::video::{
    AudioClip, Canvas, ConditionMedia, ConditionRole, ConditionVision, ImageFit, VideoClip,
    VisionGrid,
};
pub use wire::ConditionsError;

/// One condition of a video request, as its model planned it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Condition {
    /// Keyframe anchor or reference.
    pub role: ConditionRole,
    /// The length of the fetched media, which travel beside the request: the media reader reads
    /// this many bytes of the condition's memory file.
    pub bytes: u64,
    /// What the media reader decodes them into.
    pub media: ConditionMedia,
    /// What the conditioner reads, for a condition it reads.
    pub vision: Option<ConditionVision>,
    /// Denoiser rows each temporal unit of the condition's pixels encodes
    /// to, in unit order: one unit for a still image, one per video encoder
    /// window for a video. Empty for an audio reference.
    pub latent_units: Vec<u32>,
    /// Denoiser audio rows of the condition's audio track; zero without one.
    pub audio_rows: u32,
}

impl Condition {
    /// The sizes the engine schedules the condition's reading and encoding by.
    pub fn extent(&self) -> crate::request::ConditionExtent {
        crate::request::ConditionExtent {
            pixel_bytes: self.pixel_bytes(),
            audio_samples: self.media.audio().map_or(0, |track| track.samples),
            vision_patches: self
                .vision
                .as_ref()
                .map_or(0, |vision| vision.grid.patches()),
            vision_tokens: self.vision.as_ref().map_or(0, |vision| vision.tokens),
            latent_units: self.latent_units.clone(),
            audio_rows: self.audio_rows,
        }
    }

    /// Denoiser video rows of the condition.
    pub fn video_rows(&self) -> u64 {
        self.latent_units.iter().map(|rows| u64::from(*rows)).sum()
    }

    /// Bytes of the RGB24 pixels the video encoder encodes.
    pub fn pixel_bytes(&self) -> u64 {
        self.media.pixels().map_or(0, |(frames, canvas)| {
            u64::from(frames) * u64::from(canvas.width) * u64::from(canvas.height) * 3
        })
    }

    /// Checks the condition's internal consistency: a keyframe is one still
    /// image the conditioner may read; a condition with pixels encodes to
    /// positive rows in at least one unit and one without has none; a
    /// condition with an audio track encodes to positive audio rows and one
    /// without has none; a video's conditioner view samples kept frames in
    /// increasing order and an image's samples none; every extent is
    /// positive.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.bytes == 0 {
            return Err("a condition's media are never empty");
        }
        if self.role.is_keyframe() && !matches!(self.media, ConditionMedia::Image(_)) {
            return Err("a keyframe is one still image");
        }
        let canvases_positive = |canvas: Canvas| canvas.width > 0 && canvas.height > 0;
        match &self.media {
            ConditionMedia::Image(fit) => {
                if !canvases_positive(fit.size)
                    || !canvases_positive(fit.resized)
                    || u64::from(fit.left) + u64::from(fit.size.width)
                        > u64::from(fit.resized.width)
                    || u64::from(fit.top) + u64::from(fit.size.height)
                        > u64::from(fit.resized.height)
                {
                    return Err("an image fit keeps a window outside its resized image");
                }
            }
            ConditionMedia::Video { clip, .. } => {
                if !canvases_positive(clip.canvas)
                    || clip.vae_frames == 0
                    || clip.vae_frames > clip.frames
                {
                    return Err("a video clip encodes frames it does not keep");
                }
            }
            ConditionMedia::Audio(_) => {}
        }
        if let Some(track) = self.media.audio()
            && (track.sample_rate == 0 || track.source_samples == 0 || track.samples == 0)
        {
            return Err("an audio track keeps no samples");
        }
        let has_pixels = self.media.pixels().is_some();
        if has_pixels == self.latent_units.is_empty() || self.latent_units.contains(&0) {
            return Err("a condition's latent units disagree with its pixels");
        }
        if self.media.audio().is_some() != (self.audio_rows > 0) {
            return Err("a condition's audio rows disagree with its audio track");
        }
        if let Some(vision) = &self.vision {
            let video = match &self.media {
                ConditionMedia::Video { clip, .. } => Some(clip),
                ConditionMedia::Image(_) => None,
                ConditionMedia::Audio(_) => {
                    return Err("the conditioner reads no audio reference");
                }
            };
            if vision.grid.patches() == 0 || vision.tokens == 0 {
                return Err("a conditioner view covers no patches");
            }
            match video {
                None if !vision.frame_indices.is_empty() || vision.grid.t != 1 => {
                    return Err("an image's conditioner view samples one frame");
                }
                Some(clip)
                    if vision.frame_indices.is_empty()
                        || !vision
                            .frame_indices
                            .windows(2)
                            .all(|pair| pair[0] < pair[1])
                        || vision
                            .frame_indices
                            .last()
                            .is_some_and(|last| *last >= clip.frames) =>
                {
                    return Err("a video's conditioner view samples frames it does not keep");
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// What a video request brings to the workers beyond its sampling: its conditions in request
/// order, and where its prompt presents their vision content.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conditions {
    /// The conditions in request order.
    pub conditions: Vec<Condition>,
    /// Tokens of the prompt the text encoder encodes, vision content included.
    pub prompt_tokens: u32,
    /// The prompt rows each block of vision content occupies, as `(first row, rows)`, in prompt
    /// order.
    pub vision_spans: Vec<(u32, u32)>,
}

impl Conditions {
    /// The encoding the engine carries: the conditions' Protobuf message.
    pub fn encode(&self) -> Vec<u8> {
        prost::Message::encode_to_vec(&crate::request::wire::VideoConditions::from(self))
    }

    /// Decodes [`Self::encode`]'s encoding.
    ///
    /// # Errors
    ///
    /// Fails when `bytes` are no conditions message, or one that leaves out a part every
    /// condition states.
    pub fn decode(bytes: &[u8]) -> Result<Self, ConditionsError> {
        let message: crate::request::wire::VideoConditions = prost::Message::decode(bytes)
            .map_err(|error| ConditionsError::Undecodable(error.to_string()))?;
        message.try_into()
    }

    /// Checks the conditions: every condition is consistent ([`Condition::validate`]), and every
    /// vision span is a non-empty range of the prompt after the one before it.
    pub fn validate(&self) -> Result<(), &'static str> {
        self.conditions.iter().try_for_each(Condition::validate)?;
        let mut end = 0_u64;
        for &(first, rows) in &self.vision_spans {
            let stop = u64::from(first) + u64::from(rows);
            if u64::from(first) < end || rows == 0 || stop > u64::from(self.prompt_tokens) {
                return Err("a vision span is not an ordered range of the prompt");
            }
            end = stop;
        }
        Ok(())
    }
}

/// The latent layout a denoiser prepares for one request.
///
/// Ordered and hashed as a plain value of numbers: it is a bucket's extent
/// (`engine::batch::Extent`), and a captured graph is keyed by the bucket it holds. The model
/// states it (`nn::capability::Denoiser::layout`), the worker captures a graph for it and the
/// scheduler names it back in the batch it submits, so it is stated here, below all three.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct Layout {
    /// The video latent tensor.
    pub video: LatentShape,
    /// The audio latent tensor, when the task produces audio.
    pub audio: Option<LatentShape>,
    /// Text conditioning rows the layout reserves.
    pub text_rows: u32,
    /// Condition rows of the reference media.
    pub condition_rows: u32,
    /// Elements of every condition's initial-noise draw, in request order.
    ///
    /// The reference draws one chained stream — conditions, then video, then audio — and a model
    /// can derive the video and audio counts from [`Self::video`] and [`Self::audio`] but not the
    /// conditions': those are drawn at each condition's own unpacked latent shape before any
    /// packing, which is not what [`Self::condition_rows`] counts. Zero for a request with no
    /// conditions.
    pub condition_elements: u32,
}

/// The extents of one latent tensor.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct LatentShape {
    /// Channels of the latent space.
    pub channels: u32,
    /// Latent frames (or audio latent steps).
    pub frames: u32,
    /// Latent rows.
    pub height: u32,
    /// Latent columns.
    pub width: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One condition of each media kind: a cover-cropped keyframe, an image
    /// reference, a video reference with its soundtrack and an audio reference.
    fn reference_conditions() -> Vec<Condition> {
        let canvas = Canvas {
            width: 1024,
            height: 768,
        };
        let bytes = |index: u32| 1_000 + u64::from(index);
        let soundtrack = AudioClip {
            sample_rate: 44_100,
            start_sample: 22_050,
            source_samples: 40_425,
            samples: 29_334,
        };
        vec![
            Condition {
                role: ConditionRole::LastFrame,
                bytes: bytes(0),
                media: ConditionMedia::Image(ImageFit {
                    resized: Canvas {
                        width: 1365,
                        height: 768,
                    },
                    left: 170,
                    top: 0,
                    size: canvas,
                }),
                vision: None,
                latent_units: vec![768],
                audio_rows: 0,
            },
            Condition {
                role: ConditionRole::Reference,
                bytes: bytes(1),
                media: ConditionMedia::Image(ImageFit {
                    resized: Canvas {
                        width: 2048,
                        height: 2048,
                    },
                    left: 0,
                    top: 0,
                    size: Canvas {
                        width: 2048,
                        height: 2048,
                    },
                }),
                vision: Some(ConditionVision {
                    grid: VisionGrid {
                        t: 1,
                        h: 128,
                        w: 128,
                    },
                    tokens: 4096,
                    frame_indices: Vec::new(),
                }),
                latent_units: vec![4096],
                audio_rows: 0,
            },
            Condition {
                role: ConditionRole::Reference,
                bytes: bytes(2),
                media: ConditionMedia::Video {
                    clip: VideoClip {
                        canvas,
                        start_frame: 12,
                        frames: 22,
                        vae_frames: 22,
                    },
                    soundtrack: Some(soundtrack),
                },
                vision: Some(ConditionVision {
                    grid: VisionGrid { t: 1, h: 32, w: 48 },
                    tokens: 384,
                    frame_indices: vec![0, 12],
                }),
                latent_units: vec![3840, 1536],
                audio_rows: 74,
            },
            Condition {
                role: ConditionRole::Reference,
                bytes: bytes(3),
                media: ConditionMedia::Audio(soundtrack),
                vision: None,
                latent_units: Vec::new(),
                audio_rows: 74,
            },
        ]
    }

    /// A `ref2va` conditioning keeps its task, presentation tags and every
    /// condition's media, conditioner view and rows through its encoding.
    #[test]
    fn conditions_round_trip_through_their_encoding() {
        let conditions = Conditions {
            conditions: reference_conditions(),
            prompt_tokens: 4,
            vision_spans: vec![(1, 2)],
        };
        let decoded = Conditions::decode(&conditions.encode()).unwrap();
        assert_eq!(decoded, conditions);
    }

    /// Conditions whose vision spans leave the prompt or overlap are refused.
    #[test]
    fn vision_spans_are_ordered_ranges_of_the_prompt() {
        let spans = |vision_spans: Vec<(u32, u32)>| Conditions {
            conditions: reference_conditions(),
            prompt_tokens: 4,
            vision_spans,
        };
        assert!(spans(vec![(0, 2), (2, 2)]).validate().is_ok());
        assert!(spans(vec![(0, 5)]).validate().is_err(), "past the prompt");
        assert!(
            spans(vec![(0, 3), (2, 2)]).validate().is_err(),
            "overlapping"
        );
        assert!(spans(vec![(2, 0)]).validate().is_err(), "empty");
    }

    fn image(role: ConditionRole) -> Condition {
        let canvas = Canvas {
            width: 64,
            height: 32,
        };
        Condition {
            role,
            bytes: 10,
            media: ConditionMedia::Image(ImageFit {
                resized: canvas,
                left: 0,
                top: 0,
                size: canvas,
            }),
            vision: Some(ConditionVision {
                grid: VisionGrid { t: 1, h: 2, w: 4 },
                tokens: 2,
                frame_indices: Vec::new(),
            }),
            latent_units: vec![2],
            audio_rows: 0,
        }
    }

    /// A still image keyframe or reference is consistent, and its sizes
    /// follow from its raster and units.
    #[test]
    fn an_image_condition_is_consistent() {
        let condition = image(ConditionRole::FirstFrame);
        assert_eq!(condition.validate(), Ok(()));
        assert_eq!(condition.video_rows(), 2);
        assert_eq!(condition.pixel_bytes(), 64 * 32 * 3);
    }

    /// Inconsistent conditions are refused: a keyframe that is not an image,
    /// rows without pixels, and a video view sampling frames past the clip.
    #[test]
    fn inconsistent_conditions_are_refused() {
        let mut audio = image(ConditionRole::FirstFrame);
        audio.media = ConditionMedia::Audio(AudioClip {
            sample_rate: 48_000,
            start_sample: 0,
            source_samples: 48_000,
            samples: 32_000,
        });
        assert!(audio.validate().is_err());

        audio.role = ConditionRole::Reference;
        audio.vision = None;
        assert!(audio.validate().is_err(), "rows without pixels");
        audio.latent_units.clear();
        audio.audio_rows = 80;
        assert_eq!(audio.validate(), Ok(()));

        let mut video = image(ConditionRole::Reference);
        video.media = ConditionMedia::Video {
            clip: VideoClip {
                canvas: Canvas {
                    width: 64,
                    height: 32,
                },
                start_frame: 0,
                frames: 24,
                vae_frames: 22,
            },
            soundtrack: None,
        };
        video.vision = Some(ConditionVision {
            grid: VisionGrid { t: 1, h: 2, w: 4 },
            tokens: 2,
            frame_indices: vec![0, 12, 24],
        });
        assert!(video.validate().is_err(), "frame 24 is not kept");
    }
}
