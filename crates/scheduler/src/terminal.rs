//! How a request ends: one type for token and media requests, mapped once
//! to the policy's end and once to the public terminal events.

use crate::{ArtifactEvent, Event, FinishReason, MigrationCause, RejectionKind, StopReason};

/// How a request ends.
///
/// A request latches its terminal as soon as it is decided (its
/// `terminal_intent`, `None` while it runs); the request is retired with it
/// once its calls in flight drain. A media request's completion with its
/// artifact is decided at that point, from its muxer's last round, and is
/// never latched.
#[derive(Debug, Clone)]
pub(crate) enum Terminal {
    /// It finished for `reason`; `stop_reason` names the matched stop
    /// condition of a token request.
    Finished {
        reason: FinishReason,
        stop_reason: Option<StopReason>,
    },
    /// A media request produced its artifact.
    Completed(ArtifactEvent),
    /// It leaves for another engine for `cause`, with the denoising steps
    /// of its trajectory the host latent pools stored (zero for none) and
    /// the record of where they lie (`Trajectory::stored_layout`); the
    /// daemon submits it again.
    Migrated {
        cause: MigrationCause,
        denoised_steps: u32,
        stored_layout: Vec<u8>,
    },
    /// The request failed with this message.
    Failed(String),
    /// A worker refused the request itself, for the reason its input-error
    /// completion states; the client receives an invalid-request rejection.
    Rejected(String),
}

/// What a request was charged, which its `Finished` event reports.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) images: usize,
}

impl Terminal {
    /// A finish for `reason` with no matched stop condition.
    pub(crate) const fn finish(reason: FinishReason) -> Self {
        Self::Finished {
            reason,
            stop_reason: None,
        }
    }

    /// A migration for `cause` naming no stored trajectory; a preempted
    /// media request's stored steps are named when it is retired.
    pub(crate) const fn migrate(cause: MigrationCause) -> Self {
        Self::Migrated {
            cause,
            denoised_steps: 0,
            stored_layout: Vec::new(),
        }
    }

    /// The migration cause, for a migration.
    pub(crate) const fn migration(&self) -> Option<MigrationCause> {
        match self {
            Self::Migrated { cause, .. } => Some(*cause),
            _ => None,
        }
    }

    /// Whether the request ends in failure or refusal.
    pub(crate) const fn is_failure(&self) -> bool {
        matches!(self, Self::Failed(_) | Self::Rejected(_))
    }

    /// How the policy is told the request ended.
    pub(crate) fn end(&self) -> scheduling::guest::End {
        use scheduling::guest::End;
        match self {
            Self::Migrated { .. } => End::Migratable,
            Self::Rejected(_) => End::Rejected,
            Self::Failed(_) => End::Failed,
            Self::Completed(_) => End::Completed,
            Self::Finished { reason, .. } => match reason {
                FinishReason::Cancelled => End::Cancelled,
                FinishReason::Error => End::Failed,
                _ => End::Completed,
            },
        }
    }

    /// The public events that end the request's output, in order: a
    /// completed artifact precedes its `Finished` event, which reports
    /// `usage`.
    pub(crate) fn events(self, usage: Usage) -> Vec<Event> {
        let finished = |reason, stop_reason| Event::Finished {
            reason,
            stop_reason,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            images: usage.images,
        };
        match self {
            Self::Finished {
                reason,
                stop_reason,
            } => vec![finished(reason, stop_reason)],
            Self::Completed(artifact) => vec![
                Event::Artifact(artifact),
                finished(FinishReason::Completed, None),
            ],
            Self::Migrated {
                cause,
                denoised_steps,
                stored_layout,
            } => vec![Event::Migratable {
                cause,
                denoised_steps,
                stored_layout,
            }],
            Self::Failed(message) => vec![Event::Error { message }],
            Self::Rejected(message) => vec![Event::Rejected {
                kind: RejectionKind::Invalid,
                message,
            }],
        }
    }
}

/// Latches a finish for `reason` unless a terminal is already latched: the
/// first decided terminal wins.
pub(crate) fn latch_finish(intent: &mut Option<Terminal>, reason: FinishReason) {
    if intent.is_none() {
        *intent = Some(Terminal::finish(reason));
    }
}
