//! Numerical execution of a batch's calls, by capability.
//!
//! Every batch carries calls of one kind on one component, and its forward
//! is a sequence of jobs of its lane: [`forward`] prepares the next job and
//! [`complete`] applies its completion. The first job completes the
//! batch's imported inputs; the others belong to what serves the kind: text
//! (`runner::causal_lm`), the diffusion LM runner (`runner::diffusion_lm`),
//! image batches (`batch::image`), a video model's media batches (`batch`),
//! host media (`media`), transfers (`transfer`); expert-only ranks
//! participate in expert steps through `experts`. These modules are the
//! integration seam of the capabilities: each owns its runners, its own
//! input preparation and result materialization. Inputs imported from other
//! ranks and workers are prepared for every runtime alike (`inputs`), and
//! products are published to them alike (`products`). A runtime that has
//! not been integrated reports its call kind as unsupported, which fails
//! the batch with `UnsupportedSetup` and leaves the rank serving.
//!
//! A text batch's jobs are operations its runner recorded, and so are a
//! video model's encoder batches (`batch::forward`). The forward of
//! every other capability, and the completion of imported inputs, still
//! issue their own stream work: each is one job holding that work as
//! [`Job::forward`], until its capability is converted.

pub(crate) mod batch;
pub(crate) mod inputs;
pub(crate) mod launch;
pub(crate) mod placement;
pub(crate) mod products;
pub(crate) mod transfer;

use uniserve_ipc::{CallKind, ForwardMode, MediaCall};

use uniserve_ipc::Batch as BatchPlan;

use super::backend::{WorkerBackend, WorkerBatch};
use crate::Error;
use crate::error::WorkerError;
use crate::executor::BatchResult;
use crate::lane::{Completion, Job};
use crate::model::components::Capability;

/// Where a batch's forward stands between its jobs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Its imported inputs are not resident yet.
    #[default]
    Inputs,
    /// Its capability's forward has jobs to run.
    Forward,
    /// Nothing is left for its lane.
    Done,
}

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

/// The next job of a batch's forward at its stage: the completion of its
/// imported inputs, then the jobs of the runtime serving `kind`. `None`
/// when the stage advanced without needing the lane.
pub(crate) fn forward<'a>(
    backend: &'a mut WorkerBackend,
    batch: &'a mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<Option<Job<'a>>, WorkerError> {
    use CallKind::Forward;

    match batch.stage() {
        Stage::Inputs if batch.plan().input_products.is_empty() => {
            // Nothing was imported: what remains is a check of resident
            // KV descriptors, which issues nothing.
            let completed = inputs::complete(backend, batch);
            imported(backend, batch, completed)?;
            Ok(None)
        }
        Stage::Inputs => {
            let job = Job::new();
            job.forward(move || {
                let completed = inputs::complete(backend, batch);
                batch.set_issued(completed);
            });
            Ok(Some(job))
        }
        Stage::Forward => match kind {
            Forward(ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify) => {
                crate::runner::causal_lm::forward(backend, batch, kind)
            }
            _ if records(backend, kind) => {
                let (plan, borrowed, state) = batch.media_parts();
                let mut parts = backend.media_parts()?;
                Ok(Some(batch::forward(&mut parts, plan, borrowed, state)?))
            }
            _ => {
                let job = Job::new();
                job.forward(move || {
                    let executed = execute(backend, batch, kind);
                    batch.set_executed(executed);
                });
                Ok(Some(job))
            }
        },
        Stage::Done => Ok(None),
    }
}

/// Apply the completion of the job [`forward`] prepared last.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    use CallKind::Forward;

    if batch.stage() == Stage::Inputs {
        let completed = batch
            .take_issued()
            .unwrap_or(Err(Error::State("the input completion did not run")));
        return imported(backend, batch, completed);
    }
    match kind {
        Forward(ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify) => {
            crate::runner::causal_lm::complete(backend, batch, completion)
        }
        _ if records(backend, kind) => {
            let done = {
                let (_, _, state) = batch.media_parts();
                let mut parts = backend.media_parts()?;
                batch::complete(&mut parts, state, completion)?
            };
            if done {
                batch.set_stage(Stage::Done);
            }
            Ok(())
        }
        _ => {
            batch.set_stage(Stage::Done);
            batch.take_executed().unwrap_or_else(|| {
                Err(WorkerError::classify(
                    Error::State("the forward did not run"),
                    None,
                ))
            })
        }
    }
}

/// Whether the forwards of `kind` on this rank are recorded into jobs
/// ([`batch::forward`]); the others issue their own stream work.
fn records(backend: &WorkerBackend, kind: CallKind) -> bool {
    use MediaCall::{LatentEncoding, TextEncoding, VisionEncoding};

    matches!(
        kind,
        CallKind::Media(TextEncoding | VisionEncoding | LatentEncoding)
    ) && backend.serves_video()
}

/// Continue with the forward once the imported inputs are resident; an
/// import that failed fails the batch's calls.
fn imported(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completed: crate::Result<()>,
) -> std::result::Result<(), WorkerError> {
    match completed {
        Ok(()) => batch.set_stage(Stage::Forward),
        Err(error) => {
            batch.set_stage(Stage::Done);
            fail(backend, batch, error, "input import")?;
        }
    }
    Ok(())
}

/// Execute a batch's calls on the runtime serving their kind, for the
/// capabilities whose forward issues its own stream work.
fn execute(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    use CallKind::{Forward, Media, Transfer};
    use MediaCall::*;

    match kind {
        Forward(ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify) => {
            Err(unsupported(kind, "text"))
        }
        Forward(ForwardMode::TokenDenoising) => {
            crate::runner::diffusion_lm::execute(backend, batch, kind)
        }
        Transfer(_) => transfer::execute(backend, batch, kind),
        Media(MediaReading | VideoEncoding | AudioEncoding | Muxing) => {
            let (plan, borrowed, state) = batch.media_parts();
            let mut parts = backend.media_parts()?;
            crate::runner::media::execute(&mut parts, plan, borrowed, state)
        }
        Media(ImageDecoding) => batch::image::execute(backend, batch, kind),
        Media(VideoDecoding | AudioDecoding) => {
            let (plan, borrowed, state) = batch.media_parts();
            let mut parts = backend.media_parts()?;
            batch::execute(&mut parts, plan, borrowed, state)
        }
        // Encoders and denoisers serve both image and video models; the
        // rank's model decides.
        Media(VisionEncoding | TextEncoding | LatentEncoding | LatentPreparation | Denoising) => {
            if backend.serves_video() {
                let (plan, borrowed, state) = batch.media_parts();
                let mut parts = backend.media_parts()?;
                batch::execute(&mut parts, plan, borrowed, state)
            } else {
                batch::image::execute(backend, batch, kind)
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
