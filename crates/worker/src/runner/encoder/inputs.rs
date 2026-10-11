//! Host preparation of an image encoding batch's inline images.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use uniserve_ipc::{CallKind, MediaCall};

use super::{PreparedImage, Tower};
use crate::error::WorkerError;
use crate::executor::Submission;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::host::HostTask;
use crate::host::image::{ImageResult, ImageTask, ImageWork};
use crate::{Error, Outcome, Result};

/// The host-prepared inline images of an encoding batch, by call index.
pub(crate) struct Inputs {
    tasks: Vec<(usize, Arc<HostTask<ImageTask>>)>,
}

impl Inputs {
    fn ready(&self) -> bool {
        self.tasks.iter().all(|(_, task)| task.completion.done())
    }

    /// The prepared images, by call index; a failed preparation fails the
    /// batch.
    pub(crate) fn take(self) -> Result<HashMap<usize, PreparedImage>> {
        let mut images = HashMap::new();
        for (index, task) in self.tasks {
            match task.completion.outcome() {
                Some(Outcome::Success(result)) => match &*result {
                    ImageResult::Prepared(image) => {
                        images.insert(index, image.clone());
                    }
                    ImageResult::Encoded(_) => {
                        return Err(Error::Invariant(
                            "an image preparation produced encoded bytes".into(),
                        ));
                    }
                },
                Some(Outcome::Failed(error)) => {
                    return Err(Error::Invalid(format!(
                        "inline image preparation failed: {error}"
                    )));
                }
                Some(Outcome::Cancelled) | None => {
                    return Err(Error::State("inline image preparation did not complete"));
                }
            }
        }
        Ok(images)
    }

    fn cancel(&self) {
        for (_, task) in &self.tasks {
            let _ = task.cancel(false);
        }
    }
}

/// Start the host preparation of a batch's inline images, and report
/// whether it completed. Notifies `submission` when it does.
pub(crate) fn prepare_inputs(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    submission: &Arc<Submission>,
) -> std::result::Result<bool, WorkerError> {
    if let Some(inputs) = batch.image_inputs() {
        return Ok(inputs.ready());
    }
    let plan = Arc::clone(batch.shared_plan());
    let Some(first) = plan.calls.first() else {
        return Ok(true);
    };
    let tower = match first.code {
        CallKind::Media(MediaCall::VisionEncoding) => Tower::Vit,
        CallKind::Media(MediaCall::LatentEncoding) => Tower::Vae,
        _ => return Ok(true),
    };
    let parts = backend.parts();
    let Some(image) = parts.image else {
        return Ok(true);
    };
    let mut tasks = Vec::new();
    let reserved = (|| -> Result<()> {
        for (index, call) in plan.calls.iter().enumerate() {
            let Some(payload) = &call.input_image else {
                continue;
            };
            let request = parts.requests.get(call.request_key.request_id.0)?;
            let input_images = request.admission().input_images;
            let route = image.encoders.route(&call.component, call.code)?;
            let pin = image.encoders.image[route].device().is_cuda();
            let task = image.host.reserve()?;
            tasks.push((index, Arc::clone(&task)));
            task.configure(ImageTask::new(ImageWork::Prepare {
                function: image.policy.prepare_input().clone(),
                loaded: image.policy.loaded().clone(),
                tower,
                payload: Arc::clone(payload),
                input_images,
                pin,
            }))?;
            let weak: Weak<Submission> = Arc::downgrade(submission);
            let notify: Box<dyn FnOnce() + Send> = Box::new(move || {
                if let Some(submission) = weak.upgrade() {
                    submission.notify_ready();
                }
            });
            if let Some(notify) = task.completion.subscribe(notify) {
                notify();
            }
            task.submit()?;
        }
        Ok(())
    })();
    let inputs = Inputs { tasks };
    if let Err(error) = reserved {
        inputs.cancel();
        return Err(WorkerError::classify(error, Some("image preparation")));
    }
    let ready = inputs.ready();
    batch.set_image_inputs(inputs);
    Ok(ready)
}
