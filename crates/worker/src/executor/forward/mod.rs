//! Numerical execution of a batch's calls, by capability runtime.
//!
//! Every batch carries calls of one kind on one component. [`execute`]
//! routes it to the runtime that serves the kind: text (`runner::causal_lm`), canvas,
//! image, video, host media (`media`), transfers (`transfer`); expert-only
//! ranks participate in expert steps through `experts`. These modules are
//! the integration seam of the capability runtimes: each owns its runners,
//! its own input preparation and result materialization. Inputs imported
//! from other ranks and workers are prepared for every runtime alike
//! (`inputs`), and products are published to them alike (`products`). A
//! runtime that has not been integrated reports its call kind as
//! unsupported, which fails the batch with `UnsupportedSetup` and leaves
//! the rank serving.

pub(crate) mod batch;
pub(crate) mod canvas;
pub(crate) mod experts;
pub(crate) mod image;
pub(crate) mod inputs;
pub(crate) mod launch;
pub(crate) mod media;
pub(crate) mod placement;
pub(crate) mod products;
pub(crate) mod transfer;
pub(crate) mod video;

use uniserve_ipc::{CallKind, ForwardMode, MediaCall};

use uniserve_ipc::Batch as BatchPlan;

use super::backend::{WorkerBackend, WorkerBatch};
use crate::Error;
use crate::error::WorkerError;
use crate::executor::BatchResult;
use crate::model::components::Capability;

/// Materialize launched work whose device results are ready.
pub(crate) fn poll(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    launched: &mut launch::Launched,
) -> std::result::Result<Option<BatchResult>, WorkerError> {
    Ok(launch::materialize(backend.requests_mut(), plan, launched)?)
}

/// Cancel launched work that will not be materialized.
pub(crate) fn abandon(
    backend: &mut WorkerBackend,
    launched: launch::Launched,
) -> std::result::Result<(), WorkerError> {
    Ok(launch::abandon(backend.requests_mut(), launched)?)
}

/// Fail every call of a batch whose execution failed before any runtime
/// committed work: a fatal failure stops the rank, any other completes the
/// batch with its calls failed under the failure's code.
pub(crate) fn fail(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    error: Error,
    context: &str,
) -> std::result::Result<(), WorkerError> {
    let error = WorkerError::classify(error, Some(context));
    if error.fatal {
        return Err(error);
    }
    let plan = batch.shared_plan();
    tracing::warn!(
        target: "uniserve_worker.execution",
        "batch {} failed: {error}",
        plan.batch_id
    );
    let output = launch::failed_output(
        backend.requests(),
        plan,
        batch.predecessors(),
        launch::error_code(error.code),
    )?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    Ok(())
}

/// The error of a call kind whose runtime is not available on this rank.
pub(crate) fn unsupported(kind: CallKind, runtime: &str) -> WorkerError {
    WorkerError::classify(
        Error::Unsupported(format!(
            "{} is served by the {runtime} runtime",
            kind_name(kind)
        )),
        None,
    )
}

fn kind_name(kind: CallKind) -> &'static str {
    match kind {
        CallKind::Forward(mode) => mode.as_str(),
        CallKind::Media(call) => call.as_str(),
        CallKind::Transfer(mode) => mode.as_str(),
    }
}

/// Execute a batch's calls on the runtime serving their kind.
pub(crate) fn execute(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    use CallKind::{Forward, Media, Transfer};
    use MediaCall::*;

    match kind {
        Forward(ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify) => {
            crate::runner::causal_lm::execute(backend, batch, kind)
        }
        Forward(ForwardMode::TokenDenoising) => canvas::execute(backend, batch, kind),
        Transfer(_) => transfer::execute(backend, batch, kind),
        Media(MediaReading | VideoEncoding | AudioEncoding | Muxing) => {
            let (plan, borrowed, state) = batch.media_parts();
            let mut parts = backend.media_parts()?;
            media::execute(&mut parts, plan, borrowed, state)
        }
        Media(ImageDecoding) => image::execute(backend, batch, kind),
        Media(VideoDecoding | AudioDecoding) => {
            let (plan, borrowed, state) = batch.media_parts();
            let mut parts = backend.media_parts()?;
            video::execute(&mut parts, plan, borrowed, state)
        }
        // Encoders and denoisers serve both image and video models; the
        // rank's model decides.
        Media(VisionEncoding | TextEncoding | LatentEncoding | LatentPreparation | Denoising) => {
            if backend.serves_video() {
                let (plan, borrowed, state) = batch.media_parts();
                let mut parts = backend.media_parts()?;
                video::execute(&mut parts, plan, borrowed, state)
            } else {
                image::execute(backend, batch, kind)
            }
        }
    }
}

/// Whether a component's model decodes video.
pub(crate) fn decodes_video(capabilities: impl IntoIterator<Item = Capability>) -> bool {
    capabilities.into_iter().any(|capability| {
        matches!(
            capability,
            Capability::VideoDenoiser | Capability::VideoPostprocessor
        )
    })
}
