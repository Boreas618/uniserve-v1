//! The conditions of a video request, as they travel from request processing to the workers.
//!
//! A video model's request processing plans each condition of a request (`processing`): what its
//! fetched bytes are ([`MediaLocator`]), what the media reader decodes them into
//! ([`ConditionMedia`]), what the text encoder's vision tower reads of it ([`ConditionVision`])
//! and the denoiser rows its latent encoders write. The engine carries a request's
//! [`Conditions`] JSON-encoded beside the sizes it schedules by and never decodes them; the
//! workers decode them and hand each stage its part. The vocabulary is the video pipeline's, not
//! one model's: any video model whose request processing plans a condition states it in these
//! terms.

use serde::{Deserialize, Serialize};

/// A video raster in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Canvas {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// How a condition relates to the generated video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionRole {
    /// A keyframe anchoring the first generated frame.
    FirstFrame,
    /// A keyframe anchoring the last generated frame.
    LastFrame,
    /// A reference conditioning the generation as a whole.
    Reference,
}

impl ConditionRole {
    /// Whether the condition anchors a generated frame.
    pub const fn is_keyframe(self) -> bool {
        matches!(self, Self::FirstFrame | Self::LastFrame)
    }
}

/// Where a condition's fetched media bytes are: a POSIX shared-memory object
/// on the host that runs the server and the media reader.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaLocator {
    /// Object name without its leading `/`.
    pub name: String,
    /// Published byte count.
    pub bytes: u64,
}

/// A decoded still image put on a raster: resized with LANCZOS to `resized`,
/// then the `size` window whose top-left corner is (`left`, `top`) kept.
///
/// A stretched image is resized straight to its raster (`resized == size`,
/// no offset); a cover-cropped keyframe is resized past the canvas and
/// centred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageFit {
    /// Size the decoded image is resized to.
    pub resized: Canvas,
    /// Left edge of the kept window in the resized image.
    pub left: u32,
    /// Top edge of the kept window in the resized image.
    pub top: u32,
    /// The kept raster, which the video encoder encodes.
    pub size: Canvas,
}

/// The frames of a reference video the media reader produces.
///
/// The video is resampled to the output frame rate and scaled onto `canvas`
/// in one decoding pass; its first `start_frame` frames are skipped and the
/// next `frames` kept. The conditioner samples the kept frames; the video encoder encodes
/// their leading `vae_frames`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoClip {
    /// The canvas of the video's own display aspect.
    pub canvas: Canvas,
    /// Frames of the output-rate timeline skipped at the start.
    pub start_frame: u32,
    /// Frames of the output-rate timeline kept after the start offset.
    pub frames: u32,
    /// The leading kept frames the video encoder encodes.
    pub vae_frames: u32,
}

/// The samples of a soundtrack or audio reference the media reader produces.
///
/// The track is decoded at its native rate, its first `start_sample` samples
/// skipped and the next `source_samples` kept, then resampled once to the
/// model's audio rate, giving `samples` stereo samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClip {
    /// The track's native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to the model's audio rate.
    pub samples: u64,
}

/// What the media reader decodes a condition's bytes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionMedia {
    /// A keyframe or an image reference: one frame.
    Image(ImageFit),
    /// A reference video and, when the request conditions on it, its
    /// soundtrack.
    Video {
        /// The frames.
        clip: VideoClip,
        /// The soundtrack.
        soundtrack: Option<AudioClip>,
    },
    /// An audio reference.
    Audio(AudioClip),
}

impl ConditionMedia {
    /// The audio track the condition carries, if any.
    pub const fn audio(&self) -> Option<&AudioClip> {
        match self {
            Self::Image(_) => None,
            Self::Video { soundtrack, .. } => soundtrack.as_ref(),
            Self::Audio(clip) => Some(clip),
        }
    }

    /// The frame count and raster of the pixels the video encoder encodes,
    /// if the condition has any.
    pub const fn pixels(&self) -> Option<(u32, Canvas)> {
        match self {
            Self::Image(fit) => Some((1, fit.size)),
            Self::Video { clip, .. } => Some((clip.vae_frames, clip.canvas)),
            Self::Audio(_) => None,
        }
    }
}

/// A vision patch grid: `t` temporal patches of `h` by `w` spatial patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisionGrid {
    /// Temporal patches; 1 for an image.
    pub t: u32,
    /// Patch rows.
    pub h: u32,
    /// Patch columns.
    pub w: u32,
}

impl VisionGrid {
    /// Patches of the grid.
    pub const fn patches(&self) -> u64 {
        self.t as u64 * self.h as u64 * self.w as u64
    }
}

/// What the conditioner reads of a visual condition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConditionVision {
    /// The patch grid the condition's frames are resized to.
    pub grid: VisionGrid,
    /// Vision tokens the condition occupies in the presentation, which is
    /// the count of its vision placeholders.
    pub tokens: u32,
    /// The kept frames of a reference video the conditioner samples, in
    /// order; empty for an image.
    pub frame_indices: Vec<u32>,
}

/// One condition of a video request, as its model planned it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Condition {
    /// Keyframe anchor or reference.
    pub role: ConditionRole,
    /// The fetched media bytes.
    pub source: MediaLocator,
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
        if self.source.name.is_empty() || self.source.name.contains('/') || self.source.bytes == 0 {
            return Err("a condition's media locator is invalid");
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
///
/// JSON keeps the encoding readable by request processing and workers of different builds as
/// long as fields are only added.
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
    /// The JSON encoding the engine carries.
    ///
    /// # Errors
    ///
    /// Fails only when serialization fails, which these plain values do not.
    pub fn encode(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Decodes [`Self::encode`]'s encoding.
    ///
    /// # Errors
    ///
    /// Fails when `bytes` are not conditions.
    pub fn decode(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
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
        let source = |index: u32| MediaLocator {
            name: format!("uniserve-media-7-{index}"),
            bytes: 1_000 + u64::from(index),
        };
        let soundtrack = AudioClip {
            sample_rate: 44_100,
            start_sample: 22_050,
            source_samples: 40_425,
            samples: 29_334,
        };
        vec![
            Condition {
                role: ConditionRole::LastFrame,
                source: source(0),
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
                source: source(1),
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
                source: source(2),
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
                source: source(3),
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
        let decoded = Conditions::decode(&conditions.encode().unwrap()).unwrap();
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
            source: MediaLocator {
                name: "uniserve-media-1".to_owned(),
                bytes: 10,
            },
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
