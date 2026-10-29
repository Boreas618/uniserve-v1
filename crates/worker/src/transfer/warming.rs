//! Transfer Server tasks borrowing scheduler-owned cache pages.

use ::storage::manifest::cache::Transfer;
use ::storage::protocol::TransferEnd;

use crate::storage::retained::Retained;
use crate::{Error, Result};

/// The rank's current warming round. Keeping the task at its connection
/// owner lets cancellation retire it even when the start reply was lost.
#[derive(Default)]
pub(crate) struct Copies {
    task: Option<u64>,
    session: Option<u64>,
}

impl Copies {
    pub(crate) fn open(&mut self, retained: &Retained, gpu: [u8; 16]) -> Result<()> {
        if self.session.is_some() {
            return Err(Error::State(
                "cache warming already owns a transfer session",
            ));
        }
        self.session = Some(retained.call("opening cache transfers", |client| {
            client.open_transfer(vec![gpu])
        })?);
        Ok(())
    }

    /// Close only after every task has reached physical completion.
    pub(crate) fn close(&mut self, retained: &Retained) -> Result<()> {
        if self.task.is_some() {
            return Err(Error::State("cache warming still has a transfer in flight"));
        }
        if let Some(session) = self.session {
            retained.call("closing cache transfers", |client| {
                client.close_transfer(session)
            })?;
            self.session = None;
        }
        Ok(())
    }

    /// Submit this rank's spans of `transfer` as one task at its priority.
    pub(crate) fn start(&mut self, retained: &Retained, transfer: Transfer) -> Result<()> {
        if self.task.is_some() {
            return Err(Error::State("the previous cache transfer still runs"));
        }
        let spans: Vec<_> = transfer
            .copies
            .into_iter()
            .filter(|copy| {
                copy.worker_group == retained.scope().group && copy.rank == retained.scope().rank
            })
            .flat_map(|copy| copy.spans)
            .collect();
        if !spans.is_empty() {
            self.task = Some(retained.transfer(spans, transfer.priority, retained.world())?);
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
