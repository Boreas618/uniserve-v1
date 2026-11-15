//! The Protobuf form of a video request's conditions (`request.proto`'s `VideoConditions` and the
//! messages it holds), and the one conversion between it and the types of this module.

use crate::request::wire;

use super::{
    AudioClip, Canvas, Condition, ConditionMedia, ConditionRole, ConditionVision, Conditions,
    ImageFit, VideoClip, VisionGrid,
};

/// Why an encoding is no conditions.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConditionsError {
    /// The bytes are no `VideoConditions` message.
    #[error("the conditions do not decode: {0}")]
    Undecodable(String),
    /// A message leaves out a part every condition states.
    #[error("a condition states no {0}")]
    Missing(&'static str),
    /// A role number of no [`ConditionRole`].
    #[error("condition role {0} is no role")]
    Role(i32),
}

impl From<&Conditions> for wire::VideoConditions {
    fn from(conditions: &Conditions) -> Self {
        Self {
            conditions: conditions.conditions.iter().map(Into::into).collect(),
            prompt_tokens: conditions.prompt_tokens,
            vision_spans: conditions
                .vision_spans
                .iter()
                .map(|&(first, rows)| wire::VisionSpan { first, rows })
                .collect(),
        }
    }
}

impl TryFrom<wire::VideoConditions> for Conditions {
    type Error = ConditionsError;

    fn try_from(conditions: wire::VideoConditions) -> Result<Self, ConditionsError> {
        Ok(Self {
            conditions: conditions
                .conditions
                .into_iter()
                .map(Condition::try_from)
                .collect::<Result<_, _>>()?,
            prompt_tokens: conditions.prompt_tokens,
            vision_spans: conditions
                .vision_spans
                .into_iter()
                .map(|span| (span.first, span.rows))
                .collect(),
        })
    }
}

impl From<&Condition> for wire::VideoCondition {
    fn from(condition: &Condition) -> Self {
        use wire::video_condition::Media;

        let role = match condition.role {
            ConditionRole::FirstFrame => wire::ConditionRole::FirstFrame,
            ConditionRole::LastFrame => wire::ConditionRole::LastFrame,
            ConditionRole::Reference => wire::ConditionRole::Reference,
        };
        let media = match &condition.media {
            ConditionMedia::Image(fit) => Media::Image(fit.into()),
            ConditionMedia::Video { clip, soundtrack } => Media::Video(wire::VideoReference {
                clip: Some(clip.into()),
                soundtrack: soundtrack.as_ref().map(Into::into),
            }),
            ConditionMedia::Audio(clip) => Media::Audio(clip.into()),
        };
        Self {
            role: role as i32,
            bytes: condition.bytes,
            media: Some(media),
            vision: condition
                .vision
                .as_ref()
                .map(|vision| wire::ConditionVision {
                    grid: Some(wire::VisionGrid {
                        t: vision.grid.t,
                        h: vision.grid.h,
                        w: vision.grid.w,
                    }),
                    tokens: vision.tokens,
                    frame_indices: vision.frame_indices.clone(),
                }),
            latent_units: condition.latent_units.clone(),
            audio_rows: condition.audio_rows,
        }
    }
}

impl TryFrom<wire::VideoCondition> for Condition {
    type Error = ConditionsError;

    fn try_from(condition: wire::VideoCondition) -> Result<Self, ConditionsError> {
        use wire::video_condition::Media;

        let role = match wire::ConditionRole::try_from(condition.role) {
            Ok(wire::ConditionRole::FirstFrame) => ConditionRole::FirstFrame,
            Ok(wire::ConditionRole::LastFrame) => ConditionRole::LastFrame,
            Ok(wire::ConditionRole::Reference) => ConditionRole::Reference,
            Ok(wire::ConditionRole::Unspecified) | Err(_) => {
                return Err(ConditionsError::Role(condition.role));
            }
        };
        let media = match condition.media.ok_or(ConditionsError::Missing("media"))? {
            Media::Image(fit) => ConditionMedia::Image(fit.try_into()?),
            Media::Video(video) => ConditionMedia::Video {
                clip: video
                    .clip
                    .ok_or(ConditionsError::Missing("video clip"))?
                    .try_into()?,
                soundtrack: video.soundtrack.map(Into::into),
            },
            Media::Audio(clip) => ConditionMedia::Audio(clip.into()),
        };
        let vision = condition
            .vision
            .map(|vision| -> Result<_, ConditionsError> {
                let grid = vision.grid.ok_or(ConditionsError::Missing("vision grid"))?;
                Ok(ConditionVision {
                    grid: VisionGrid {
                        t: grid.t,
                        h: grid.h,
                        w: grid.w,
                    },
                    tokens: vision.tokens,
                    frame_indices: vision.frame_indices,
                })
            })
            .transpose()?;
        Ok(Self {
            role,
            bytes: condition.bytes,
            media,
            vision,
            latent_units: condition.latent_units,
            audio_rows: condition.audio_rows,
        })
    }
}

impl From<Canvas> for wire::Canvas {
    fn from(canvas: Canvas) -> Self {
        Self {
            width: canvas.width,
            height: canvas.height,
        }
    }
}

/// The canvas a message states; a missing one is refused.
fn canvas(canvas: Option<wire::Canvas>, what: &'static str) -> Result<Canvas, ConditionsError> {
    let canvas = canvas.ok_or(ConditionsError::Missing(what))?;
    Ok(Canvas {
        width: canvas.width,
        height: canvas.height,
    })
}

impl From<&ImageFit> for wire::ImageFit {
    fn from(fit: &ImageFit) -> Self {
        Self {
            resized: Some(fit.resized.into()),
            left: fit.left,
            top: fit.top,
            size: Some(fit.size.into()),
        }
    }
}

impl TryFrom<wire::ImageFit> for ImageFit {
    type Error = ConditionsError;

    fn try_from(fit: wire::ImageFit) -> Result<Self, ConditionsError> {
        Ok(Self {
            resized: canvas(fit.resized, "resized raster")?,
            left: fit.left,
            top: fit.top,
            size: canvas(fit.size, "kept raster")?,
        })
    }
}

impl From<&VideoClip> for wire::VideoClip {
    fn from(clip: &VideoClip) -> Self {
        Self {
            canvas: Some(clip.canvas.into()),
            start_frame: clip.start_frame,
            frames: clip.frames,
            vae_frames: clip.vae_frames,
        }
    }
}

impl TryFrom<wire::VideoClip> for VideoClip {
    type Error = ConditionsError;

    fn try_from(clip: wire::VideoClip) -> Result<Self, ConditionsError> {
        Ok(Self {
            canvas: canvas(clip.canvas, "video canvas")?,
            start_frame: clip.start_frame,
            frames: clip.frames,
            vae_frames: clip.vae_frames,
        })
    }
}

impl From<&AudioClip> for wire::AudioClip {
    fn from(clip: &AudioClip) -> Self {
        Self {
            sample_rate: clip.sample_rate,
            start_sample: clip.start_sample,
            source_samples: clip.source_samples,
            samples: clip.samples,
        }
    }
}

impl From<wire::AudioClip> for AudioClip {
    fn from(clip: wire::AudioClip) -> Self {
        Self {
            sample_rate: clip.sample_rate,
            start_sample: clip.start_sample,
            source_samples: clip.source_samples,
            samples: clip.samples,
        }
    }
}
