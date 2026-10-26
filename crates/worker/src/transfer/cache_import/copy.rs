//! Cache copy actions on the shared native host executor.

use crate::host::HostAction;
use crate::{Error, Outcome, Result};

use super::ImportCopy;

impl HostAction for ImportCopy {
    type Output = ();

    fn run(&self) -> Result<()> {
        let _range = crate::profiling::range(c"uniserve.cache_import", None);
        self.write.start();
        let workspace = match self.importer.acquire(&self.write) {
            Ok(workspace) => workspace,
            Err(error) => {
                self.write.finish(true);
                return Err(error);
            }
        };

        let copied = (|| {
            self.importer.require_active(&self.write)?;
            // Transport reads run on other streams, but each records its
            // handoff on the workspace stream after this reset, and every
            // transport copy waits on its handoff before writing.
            self.reset(&workspace)?;
            self.copy(&workspace)
        })();

        let drained = self.drain(&workspace);
        self.write.finish(drained.is_ok());
        if let Err(error) = drained {
            // The drain's failure leaves the copy's completion unknown; the
            // copy's own failure, if any, is reported with it.
            return Err(match copied {
                Ok(()) => error,
                Err(cause) => Error::transport(format!(
                    "{error}; cache import cleanup also failed: {cause}"
                )),
            });
        }

        copied
    }

    fn input_outcome(&self) -> Outcome {
        // Queued cancellation has no numerical access to drain. Failed DMA
        // retains this action and its backing through the host pool's common
        // unknown-completion handling.
        self.write.task_done();
        if self.write.drained() {
            Outcome::Success(())
        } else {
            Outcome::Cancelled
        }
    }

    fn release(&self) -> Result<()> {
        self.retired.notify(self.importer.reap())
    }
}
