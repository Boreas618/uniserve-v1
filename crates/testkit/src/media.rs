//! Responds to media batches, read from what the mechanism staged for each request.
//!
//! A media batch carries no token columns: each of its rows is one request's staged record
//! ([`ipc::execution::StagedRequest`]), which names the request, the diffusion step the call runs
//! and, for a muxing call, the encoded rounds it hands over. [`Media`] responds such a batch the way
//! a media runner's completion reads:
//!
//! - a latent preparation opens its trajectory at the step its restore stored, or at zero;
//! - a denoising call completes the step it runs (`step + 1`);
//! - the final muxing call, the one that hands no round, publishes the artifact into a POSIX
//!   shared-memory object the mechanism claims;
//! - every other call completes with nothing to report.
//!
//! A case can refuse a request's calls as the model would refuse its inputs, overstate the length
//! of a published artifact, and state the page layout a latent preparation reports.

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::{Arc, Mutex, PoisonError};

use ipc::execution::{
    Batch, BatchOutput, CallOutput, Computation, Extent, MediaColumns, Outcome, RowState,
};
use ipc::{CallStatus, ErrorCode, MediaCall, RankLatentLayout, RequestId, TimingCounters};

use crate::batches::OutputFn;

/// The media half of a fake worker: what its media calls response.
#[derive(Default)]
pub struct Media {
    /// Requests whose media calls the model refuses, with the reason it states.
    refused: Mutex<HashMap<RequestId, String>>,
    /// Bytes a published artifact claims beyond what it holds.
    overstatement: u64,
    /// The page layout every latent preparation reports, one entry per denoising rank.
    latent_layouts: Vec<RankLatentLayout>,
}

impl Media {
    /// Media calls that complete as a runner's do, refusing nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// States `extra` bytes more than every published artifact holds, which the mechanism's claim
    /// must refuse.
    #[must_use]
    pub fn overstating(mut self, extra: u64) -> Self {
        self.overstatement = extra;
        self
    }

    /// Reports `layouts` from every latent preparation, as the denoising ranks declare them.
    #[must_use]
    pub fn with_latent_layouts(mut self, layouts: Vec<RankLatentLayout>) -> Self {
        self.latent_layouts = layouts;
        self
    }

    /// Refuses every media call of request `id` with `reason`, as the model refuses its inputs.
    pub fn refuse(&self, id: RequestId, reason: &str) {
        self.refused
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, reason.to_owned());
    }

    /// The response of a fake whose batches are all media batches.
    pub fn output_fn(self: Arc<Self>) -> OutputFn {
        Arc::new(move |batch| self.output(batch))
    }

    /// The bucket a fake serves `computation` in: the extent the mechanism stated, so every media
    /// computation is served.
    pub fn serves(computation: &Computation) -> Option<Extent> {
        Some(computation.extent)
    }

    /// The completion of media batch `batch`; a batch without media columns fails, since no
    /// runner of this fake serves it.
    pub fn output(&self, batch: &Batch) -> Outcome<BatchOutput> {
        let Some(columns) = batch.media.as_ref() else {
            return failed(format!("batch {} carries no media columns", batch.id.0));
        };

        let mut calls = Vec::with_capacity(batch.calls.len());
        for (index, call) in batch.calls.iter().enumerate() {
            match self.complete(batch, columns, index) {
                Ok(output) => calls.push(output),
                Err(message) => return failed(message),
            }
            debug_assert_eq!(calls[index].call, *call);
        }

        Outcome::Success(BatchOutput {
            calls,
            timing: TimingCounters::default(),
            forward_stats: None,
        })
    }

    /// Call `index` of `batch`, which serves `columns.computation`.
    fn complete(
        &self,
        batch: &Batch,
        columns: &MediaColumns,
        index: usize,
    ) -> Result<CallOutput, String> {
        let call = batch.calls[index];
        let staged = columns.request(index).map_err(|error| error.to_string())?;
        let request = staged.request.request_id;

        if let Some(reason) = self
            .refused
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&request)
        {
            let mut output = CallOutput::completed(call, Vec::new(), RowState::default());
            output.status = CallStatus::Error;
            output.error_code = Some(ErrorCode::InputError);
            output.error_message = Some(reason.clone());
            return Ok(output);
        }

        let step_index = match columns.computation.call {
            // A restored trajectory opens at its stored step, any other at zero.
            MediaCall::LatentPreparation => batch
                .latent_restores
                .iter()
                .find(|(key, _)| *key == staged.request)
                .map_or(0, |(_, restore)| restore.steps),
            MediaCall::Denoising => staged.step + 1,
            _ => 0,
        };
        let mut output = CallOutput::completed(
            call,
            Vec::new(),
            RowState {
                step_index,
                cache_computed_len: 0,
                cache_visible_len: 0,
            },
        );

        match columns.computation.call {
            MediaCall::LatentPreparation => output.latent_layouts = self.latent_layouts.clone(),
            // The muxer takes each round's encoded units as they arrive and assembles the
            // artifact on the call that hands none.
            MediaCall::Muxing if staged.handed == 0 => {
                let payload = format!("uniserve-fake-video request={}", request.0);
                output.media_output = Some(publish(payload.as_bytes(), self.overstatement)?);
            }
            _ => {}
        }
        Ok(output)
    }
}

/// A batch outcome failed with `message`.
fn failed(message: String) -> Outcome<BatchOutput> {
    Outcome::Failed(Arc::from(Box::<ipc::execution::Fault>::from(message)))
}

/// Publishes `bytes` as a POSIX shared-memory object stating `extra` bytes more than it holds, as
/// a worker's media stage relinquishes an artifact for the mechanism to claim.
fn publish(bytes: &[u8], extra: u64) -> Result<ipc::MediaOutput, String> {
    let storage = || -> std::io::Result<String> {
        let mut storage = tempfile::Builder::new()
            .prefix("uniserve-fake-video-")
            .tempfile_in("/dev/shm")?;
        storage.write_all(bytes)?;
        let (_file, path) = storage.keep().map_err(|error| error.error)?;
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
            .ok_or_else(|| std::io::Error::other("the shared-memory object has no printable name"))
    };
    let name = storage().map_err(|error| format!("publishing the artifact: {error}"))?;

    Ok(ipc::MediaOutput {
        handle: ipc::ArtifactHandle::PosixShm { name },
        bytes: bytes.len() as u64 + extra,
    })
}
