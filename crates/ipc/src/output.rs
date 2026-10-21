//! Call completions and batch results: the [`CallOutput`] a rank emits
//! once for every call, the media artifacts it materializes, and the
//! [`BatchOutput`] that answers one batch.

use super::*;

/// Terminal status of one call's completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CallStatus {
    /// Execution completed successfully.
    Ok = 0,
    /// The call was skipped because its predicate was false.
    Predicated = 1,
    /// Execution failed with a deterministic error code.
    Error = 2,
}

/// The class of a rank's failure: the one vocabulary of a failed call's
/// completion and of an error response.
///
/// The rank classifies every failure it reports with these classes, and the
/// engine decides by them: an invariant violation is a control-plane defect
/// that stops the engine, an input error is a rejected request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The rank does not serve the call's kind.
    UnsupportedCall,
    /// A message, a call or a launch value breaks the protocol contract.
    InvalidDescriptor,
    /// The deployment asks for a placement, a device or a kernel the rank
    /// cannot provide.
    UnsupportedSetup,
    /// A request's inputs are invalid for its computation.
    InputError,
    /// Device or numerical computation failed.
    ComputeError,
    /// A reserved or physical resource could not satisfy the work.
    ResourceError,
    /// An invariant the rank relies on is broken.
    InvariantViolation,
    /// The rank failed in a way that leaves it unable to serve.
    FatalWorkerFailure,
}

impl ErrorCode {
    /// The class's name as the protocol spells it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedCall => "unsupported_call",
            Self::InvalidDescriptor => "invalid_descriptor",
            Self::UnsupportedSetup => "unsupported_setup",
            Self::InputError => "input_error",
            Self::ComputeError => "compute_error",
            Self::ResourceError => "resource_error",
            Self::InvariantViolation => "invariant_violation",
            Self::FatalWorkerFailure => "fatal_worker_failure",
        }
    }

    /// Whether a failure of this class leaves the rank unsafe to serve
    /// further work.
    pub const fn fatal(self) -> bool {
        matches!(self, Self::InvariantViolation | Self::FatalWorkerFailure)
    }
}

impl std::fmt::Display for ErrorCode {
    /// Formats the class as its protocol name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Device-observed finish candidates for a token call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct FinishFlags {
    /// Whether an end-of-sequence token was selected.
    pub eos: bool,
    /// Whether the configured length limit was reached.
    pub length: bool,
    /// Whether a configured stop condition matched.
    pub stop: bool,
}

/// Per-call timing counters. Accounting only; never state identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TimingCounters {
    /// Time spent waiting for worker execution, in microseconds.
    pub queued_us: u64,
    /// Time spent executing device work, in microseconds.
    pub device_us: u64,
    /// Time spent transferring completion data, in microseconds.
    pub copy_us: u64,
    /// Time spent processing the completion on the host, in microseconds.
    pub host_us: u64,
}

/// Transport handle for a completed media artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactHandle {
    /// Artifact stored in a POSIX shared-storage object.
    PosixShm {
        /// Shared-storage object name without a path separator.
        name: String,
    },
}

impl ArtifactHandle {
    /// Validates the transport-specific artifact identifier.
    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::PosixShm { name } => ensure_valid!(
                !name.is_empty() && !name.contains('/'),
                "POSIX shared-storage artifact name is invalid"
            ),
        }
        Ok(())
    }
}

/// Metadata describing one materialized media artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaOutput {
    /// Transport descriptor for the materialized artifact.
    pub handle: ArtifactHandle,
    /// Artifact length in bytes.
    pub bytes: u64,
}

/// The completion record a worker emits once for every call, after its copy
/// event is query-ready and its pinned fields are validated on the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallOutput {
    /// Request completed by the call.
    pub request_key: RequestKey,
    /// Request-local call identifier.
    pub call_id: CallId,
    /// Terminal execution status.
    pub status: CallStatus,
    /// Error classification when `status` is [`CallStatus::Error`].
    pub error_code: Option<ErrorCode>,
    /// Why the rank refused the call's request, on an
    /// [`ErrorCode::InputError`] completion whose rank states it; the engine
    /// rejects the request with it.
    pub error_message: Option<String>,
    /// Worker timing measurements for the call.
    pub timing_counters: TimingCounters,
    /// Accepted cache prefix available to a successor.
    pub cache_visible_len: u32,
    /// Cache extent initialized by execution, including rejected speculative positions.
    pub cache_computed_len: u32,
    /// Solver steps the call's trajectory has completed, which is also the
    /// index of its next step (diffusers' `step_index`): the step a
    /// denoising call's update publishes.
    pub step_index: u32,
    /// Token identifiers accepted by the sampler.
    pub committed_tokens: Vec<u32>,
    /// Natural-log probability of the final accepted token, when requested.
    pub sampled_logprob: Option<f32>,
    /// Ranked candidates for the final accepted token.
    pub top_logprobs: Vec<TokenLogprob>,
    /// Ranked candidates for each scored prompt position, in input order.
    pub prompt_logprobs: Vec<Vec<TokenLogprob>>,
    /// Natural-log probability of every candidate of a token-denoising
    /// readout, in the call's `Readout::candidate_ids` order.
    pub candidate_logprobs: Vec<f32>,
    /// Device-observed terminal conditions.
    pub finish_flags: FinishFlags,
    /// Completed artifact and its external storage lifetime, when produced.
    pub media_output: Option<MediaOutput>,
    /// Physical cache export, including this rank's tensor locations.
    pub cache_output: Option<CacheTransfer>,
}

impl CallOutput {
    /// Validates completion identity, status, accepted lengths, and the
    /// ownership of a cache export.
    ///
    /// Rules that relate a completion to the kind of the call it answers
    /// belong to the engine, which holds the call: a cache export answers a
    /// successful cache-export call, and candidate log-probabilities a
    /// successful token-denoising call.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "completion call id must be positive"
        );
        if let Some(export) = &self.cache_output {
            export.validate()?;
            ensure_valid!(
                self.status == CallStatus::Ok
                    && export.source.owner == self.request_key
                    && export.source.producer_call_id == self.call_id,
                "cache export does not belong to its successful completion"
            );
        }

        ensure_valid!(
            self.cache_visible_len <= self.cache_computed_len,
            "completion selected cache length exceeds computed length"
        );
        match self.status {
            CallStatus::Error => ensure_valid!(
                self.error_code.is_some(),
                "an error completion must carry an error code"
            ),
            CallStatus::Ok | CallStatus::Predicated => ensure_valid!(
                self.error_code.is_none(),
                "a non-error completion must not carry an error code"
            ),
        }
        ensure_valid!(
            self.error_message.is_none() || self.error_code == Some(ErrorCode::InputError),
            "only an input-error completion states why the rank refused its request"
        );
        ensure_valid!(
            self.candidate_logprobs.is_empty() || self.status == CallStatus::Ok,
            "candidate log-probabilities belong to a successful completion"
        );
        if self.status == CallStatus::Predicated {
            ensure_valid!(
                self.committed_tokens.as_slice().is_empty(),
                "a predicated completion must select its parent without semantic output"
            );
            ensure_valid!(
                !self.finish_flags.eos && !self.finish_flags.length && !self.finish_flags.stop,
                "a predicated completion must not select a terminal outcome"
            );
        }
        if let Some(output) = self.media_output.as_ref() {
            output.handle.validate()?;
            ensure_valid!(
                output.bytes > 0,
                "media output byte length must be positive"
            );
        }
        Ok(())
    }
}

/// One batch's complete result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchOutput {
    /// Batch identity copied from the submission.
    pub batch_id: u64,
    /// Call completions, one per call the batch carried.
    pub completions: Vec<CallOutput>,
    /// Product values published by completed calls.
    pub products: Vec<TensorExport>,
    /// Aggregate worker execution time in microseconds, when measured.
    pub worker_exec_us: Option<u64>,
    /// Model-forward statistics, when reported by the worker.
    pub forward_stats: Option<ForwardStats>,
}

impl BatchOutput {
    /// Iterates over call completions in report order.
    pub fn completions(&self) -> impl Iterator<Item = &CallOutput> {
        self.completions.iter()
    }

    /// Iterates over resolved product payloads in report order.
    pub fn products(&self) -> impl Iterator<Item = &TensorExport> {
        self.products.iter()
    }

    /// Validates completion and product identities for this batch.
    pub fn validate(&self) -> ValidationResult<()> {
        let mut identities = HashSet::with_capacity(self.completions.len());
        for completion in &self.completions {
            completion.validate()?;
            ensure_valid!(
                completion.call_id.batch_id == self.batch_id,
                "completion identity belongs to another logical batch"
            );
            ensure_valid!(
                identities.insert(completion.call_id),
                "completion report repeats a call"
            );
        }
        for payload in &self.products {
            payload.validate()?;
        }
        Ok(())
    }
}

/// One ranked vocabulary candidate at a generated or prompt token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// Vocabulary token identity.
    pub token_id: u32,
    /// Natural-log probability assigned to the token.
    pub logprob: f32,
    /// One-based competition rank at the position; tied log probabilities
    /// share a rank.
    pub rank: u32,
}
