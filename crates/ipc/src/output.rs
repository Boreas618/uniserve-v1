//! Call completions and batch results: the [`RequestOutput`] a rank emits
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

/// A deterministic error class for a failed completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ErrorCode {
    /// Call metadata or product declarations are invalid.
    InvalidCall = 0,
    /// A reserved or physical resource could not satisfy the call.
    ResourceExhausted = 1,
    /// Device computation failed.
    ComputeError = 2,
    /// The request was cancelled before completion.
    Cancelled = 3,
    /// An invariant failed within the worker.
    Internal = 4,
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
pub struct RequestOutput {
    /// Request completed by the call.
    pub request_key: RequestKey,
    /// Request-local call identifier.
    pub call_id: CallId,
    /// Terminal execution status.
    pub status: CallStatus,
    /// Allocation generations for products emitted by the call.
    pub product_generations: Vec<u32>,
    /// Error classification when `status` is [`CallStatus::Error`].
    pub error_code: Option<ErrorCode>,
    /// Worker timing measurements for the call.
    pub timing_counters: TimingCounters,
    /// Kind that produced this result; must match the submitted call.
    pub code: CallKind,
    /// Model position for the next input, including image-feedback position advances.
    pub position: u32,
    /// Accepted cache prefix available to a successor.
    pub cache_visible_len: u32,
    /// Cache extent initialized by execution, including rejected speculative positions.
    pub cache_computed_len: u32,
    /// Completed denoising steps, independent of latent tensor dimensions.
    pub num_completed_steps: u32,
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

impl RequestOutput {
    /// Validates completion identity, status, accepted lengths, and product generations.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "completion call id must be positive"
        );
        if let Some(export) = &self.cache_output {
            export.validate()?;
            ensure_valid!(
                self.status == CallStatus::Ok
                    && self.code == CallKind::Transfer(TransferMode::CacheExport)
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
            self.candidate_logprobs.is_empty()
                || (self.status == CallStatus::Ok
                    && self.code == CallKind::Forward(ForwardMode::TokenDenoising)),
            "candidate log-probabilities belong to a successful token-denoising completion"
        );
        if self.status == CallStatus::Predicated {
            ensure_valid!(
                self.committed_tokens.as_slice().is_empty() && self.product_generations.is_empty(),
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
    pub completions: Vec<RequestOutput>,
    /// Product values published by completed calls.
    pub products: Vec<TensorExport>,
    /// Aggregate worker execution time in microseconds, when measured.
    pub worker_exec_us: Option<u64>,
    /// Model-forward statistics, when reported by the worker.
    pub forward_stats: Option<ForwardStats>,
}

impl BatchOutput {
    /// Iterates over call completions in report order.
    pub fn completions(&self) -> impl Iterator<Item = &RequestOutput> {
        self.completions.iter()
    }

    /// Iterates over resolved product payloads in report order.
    pub fn products(&self) -> impl Iterator<Item = &TensorExport> {
        self.products.iter()
    }

    /// Validates completion and product identities for this run.
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
