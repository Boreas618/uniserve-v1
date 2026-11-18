//! Folds the runtime events of one video generation into its artifact.
//!
//! Every consumer of a video stream (the synchronous route and the job task)
//! waits for the same result: exactly one artifact
//! followed by a normal finish. [`ArtifactCollector`] classifies the events
//! once; each consumer keeps its own read loop, with its own cancellation and
//! progress handling. A [`VideoStreamError`] has one message and one code,
//! which the synchronous route renders as an OpenAI error and a job records
//! as its failure.

use std::ops::ControlFlow;

use proto::request::{ArtifactEvent, RejectionKind};

use crate::api::output::{FinishReason, RequestOutput, finish_reason_name};

/// Why a video generation stream did not end in an artifact.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum VideoStreamError {
    /// The engine refused the request before executing it.
    #[error("{message}")]
    Rejected {
        /// Whether the request is unservable or the engine is full.
        kind: RejectionKind,
        /// Caller-visible rejection description.
        message: String,
    },
    /// Generation failed.
    #[error("{message}")]
    Failed {
        /// Description of the failure.
        message: String,
    },
    /// The request finished without completing normally.
    #[error(
        "video generation ended without an artifact: {}",
        finish_reason_name(reason)
    )]
    Ended {
        /// How the request finished.
        reason: FinishReason,
    },
    /// The request was cancelled or aborted.
    #[error("video generation was cancelled")]
    Cancelled,
    /// The request completed normally but produced no artifact.
    #[error("video generation produced no artifact")]
    MissingArtifact,
    /// The runtime emitted a second artifact.
    #[error("video generation produced more than one artifact")]
    MultipleArtifacts,
    /// The runtime emitted an event a video generation never carries.
    #[error("the engine emitted an event a video generation does not carry")]
    Incompatible,
    /// The stream ended before the request finished.
    #[error("the engine closed the video stream before the request finished")]
    Closed,
}

impl VideoStreamError {
    /// Returns the stable machine-readable code of this error.
    ///
    /// A rejection has the code of the OpenAI error it answers as
    /// (`invalid_request_error` or `server_overloaded`), so a job reports the
    /// code the synchronous route answers the same rejection with.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Rejected {
                kind: RejectionKind::Invalid,
                ..
            } => "invalid_request_error",
            Self::Rejected {
                kind: RejectionKind::Overloaded,
                ..
            } => "server_overloaded",
            Self::Rejected {
                kind: RejectionKind::Role,
                ..
            } => "replica_unavailable",
            Self::Failed { .. } => "generation_failed",
            Self::Ended { .. } | Self::Cancelled => "generation_terminated",
            Self::MissingArtifact => "missing_artifact",
            Self::MultipleArtifacts | Self::Incompatible => "invalid_runtime_event",
            Self::Closed => "generation_stopped",
        }
    }
}

/// Holds the artifact of a video generation while its events arrive.
#[derive(Debug, Default)]
pub struct ArtifactCollector {
    artifact: Option<ArtifactEvent>,
}

impl ArtifactCollector {
    /// Folds one stream event, `None` for the end of the stream.
    ///
    /// Scheduling and progress events carry nothing the result needs and
    /// continue. The first terminal event breaks with the artifact of a
    /// normal finish, or the error that ended the request.
    pub fn push(
        &mut self,
        event: Option<RequestOutput>,
    ) -> ControlFlow<Result<ArtifactEvent, VideoStreamError>> {
        let result = match event {
            Some(RequestOutput::Artifact(value)) => {
                if self.artifact.replace(value).is_some() {
                    Err(VideoStreamError::MultipleArtifacts)
                } else {
                    return ControlFlow::Continue(());
                }
            }
            Some(RequestOutput::Finished {
                reason: FinishReason::Completed | FinishReason::Eos | FinishReason::Stop,
                ..
            }) => self
                .artifact
                .take()
                .ok_or(VideoStreamError::MissingArtifact),
            Some(RequestOutput::Finished {
                reason: FinishReason::Cancelled | FinishReason::Aborted,
                ..
            }) => Err(VideoStreamError::Cancelled),
            Some(RequestOutput::Finished { reason, .. }) => Err(VideoStreamError::Ended { reason }),
            Some(RequestOutput::Rejected { kind, message }) => {
                Err(VideoStreamError::Rejected { kind, message })
            }
            Some(RequestOutput::Failed { message }) => Err(VideoStreamError::Failed { message }),
            Some(RequestOutput::Scheduled { .. } | RequestOutput::MediaProgress { .. }) => {
                return ControlFlow::Continue(());
            }
            Some(_) => Err(VideoStreamError::Incompatible),
            None => Err(VideoStreamError::Closed),
        };
        ControlFlow::Break(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::output::RequestUsage;

    fn ended(event: Option<RequestOutput>) -> Result<ArtifactEvent, VideoStreamError> {
        match ArtifactCollector::default().push(event) {
            ControlFlow::Break(result) => result,
            ControlFlow::Continue(()) => panic!("the event does not end the stream"),
        }
    }

    fn finished(reason: FinishReason) -> Option<RequestOutput> {
        Some(RequestOutput::Finished {
            reason,
            stop_reason: None,
            usage: RequestUsage::default(),
        })
    }

    #[test]
    fn lifecycle_and_progress_events_continue() {
        let mut collector = ArtifactCollector::default();
        for event in [
            RequestOutput::Scheduled {
                queued_at: 1.0,
                scheduled_at: 2.0,
            },
            RequestOutput::MediaProgress {
                phase: "denoising".to_owned(),
                completed_steps: 1,
            },
        ] {
            assert!(collector.push(Some(event)).is_continue());
        }
    }

    #[test]
    fn a_normal_finish_without_an_artifact_is_missing_one() {
        assert_eq!(
            ended(finished(FinishReason::Completed)).unwrap_err(),
            VideoStreamError::MissingArtifact
        );
    }

    #[test]
    fn terminal_events_name_why_the_request_ended() {
        assert_eq!(
            ended(finished(FinishReason::MaxTokens)).unwrap_err(),
            VideoStreamError::Ended {
                reason: FinishReason::MaxTokens
            }
        );
        assert_eq!(
            ended(finished(FinishReason::Cancelled)).unwrap_err(),
            VideoStreamError::Cancelled
        );

        let rejected = RequestOutput::Rejected {
            kind: RejectionKind::Overloaded,
            message: "full".to_owned(),
        };
        assert_eq!(
            ended(Some(rejected)).unwrap_err(),
            VideoStreamError::Rejected {
                kind: RejectionKind::Overloaded,
                message: "full".to_owned()
            }
        );

        let failed = RequestOutput::Failed {
            message: "boom".to_owned(),
        };
        assert_eq!(
            ended(Some(failed)).unwrap_err(),
            VideoStreamError::Failed {
                message: "boom".to_owned()
            }
        );
    }

    #[test]
    fn a_stream_that_ends_early_or_carries_text_is_refused() {
        assert_eq!(ended(None).unwrap_err(), VideoStreamError::Closed);
        let text = RequestOutput::TextDelta {
            text: "hello".to_owned(),
        };
        assert_eq!(
            ended(Some(text)).unwrap_err(),
            VideoStreamError::Incompatible
        );
    }
}
