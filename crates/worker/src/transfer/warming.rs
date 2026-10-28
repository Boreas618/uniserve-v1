//! Transfer Server tasks borrowing scheduler-owned cache pages.

use storage::manifest::cache::Copy;
use storage::protocol::{Priority, TransferEnd};

use crate::memory::retained::Retained;
use crate::{Error, Result};

/// The rank's current warming round. Keeping the task at its connection
/// owner lets cancellation retire it even when the start reply was lost.
#[derive(Default)]
pub(crate) struct Copies {
    task: Option<u64>,
}

impl Copies {
    pub(crate) fn start(&mut self, retained: &Retained, copies: Vec<Copy>) -> Result<()> {
        if self.task.is_some() {
            return Err(Error::State("the previous cache transfer still runs"));
        }
        let spans: Vec<_> = copies
            .into_iter()
            .filter(|copy| {
                copy.worker_group == retained.scope().group && copy.rank == retained.scope().rank
            })
            .flat_map(|copy| copy.spans)
            .collect();
        if !spans.is_empty() {
            self.task = Some(retained.transfer(spans, Priority::Warming, retained.world())?);
        }
        Ok(())
    }

    pub(crate) fn poll(&mut self, retained: &Retained) -> Result<bool> {
        let Some(task) = self.task else {
            return Ok(true);
        };
        // A final event, including failure, is consumed exactly once. A
        // subsequent cancellation must not await that event a second time.
        let Some((_, end)) = retained.call("polling a cache transfer", |client| {
            client.poll_transfer(task)
        })?
        else {
            return Ok(false);
        };
        self.task = None;
        match end {
            TransferEnd::Complete => Ok(true),
            TransferEnd::Cancelled => Err(Error::Interface("cache transfer cancelled".into())),
            TransferEnd::Failed { message } => Err(Error::Interface(message)),
        }
    }

    pub(crate) fn cancel(&mut self, retained: &Retained) -> Result<()> {
        if let Some(task) = self.task {
            retained.cancel_transfer(task)?;
            self.task = None;
        }
        Ok(())
    }
}
