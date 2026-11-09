//! The tasks MiniMax-H3 serves.
//!
//! A task is what a request conditions on: nothing but its prompt, keyframes anchoring its first
//! or last frame, or reference media. The public API states the references and the model derives
//! the task from them ([`VideoTask::of`]); the checkpoint states which tasks each of its networks
//! serves.

use serde::{Deserialize, Serialize};

/// A video-and-audio generation task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VideoTask {
    /// Text to video and audio (`t2va`).
    #[serde(rename = "t2va")]
    TextToVideoAudio,
    /// First and/or last keyframe to video and audio (`fl2va`).
    #[serde(rename = "fl2va")]
    FirstLastToVideoAudio,
    /// Image, video and audio references to video and audio (`ref2va`).
    #[serde(rename = "ref2va")]
    ReferenceToVideoAudio,
}

impl VideoTask {
    /// Every task, in canonical order.
    pub const ALL: [Self; 3] = [
        Self::TextToVideoAudio,
        Self::FirstLastToVideoAudio,
        Self::ReferenceToVideoAudio,
    ];

    /// The task's name, as the checkpoint and the denoiser's description state it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextToVideoAudio => "t2va",
            Self::FirstLastToVideoAudio => "fl2va",
            Self::ReferenceToVideoAudio => "ref2va",
        }
    }

    /// The task named `name`, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|task| task.as_str() == name)
    }

    /// The task a request of these conditions runs: `ref2va` with any reference, `fl2va` with
    /// keyframes alone, `t2va` with neither.
    pub const fn of(request: &nn::VideoRequest<'_>) -> Self {
        if request.references > 0 {
            Self::ReferenceToVideoAudio
        } else if request.first_frames + request.last_frames > 0 {
            Self::FirstLastToVideoAudio
        } else {
            Self::TextToVideoAudio
        }
    }

    /// Whether the task's requests carry conditions.
    pub const fn conditioned(self) -> bool {
        !matches!(self, Self::TextToVideoAudio)
    }
}
