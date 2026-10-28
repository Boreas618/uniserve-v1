//! Host tasks' reports: the pipe the executor's host tasks report through,
//! the mailbox it fills, and the host work a launched batch finishes once
//! its tasks reported ([`HostTasks`]).
//!
//! Host calls (media reading, encoding, muxing) run on the executor's host
//! thread after `execute` returned. Each finished task writes one report to
//! a pipe the worker created ([`Reports`]): a header of six little-endian
//! `i64` (batch, call, row, status, value, payload bytes) and its payload
//! (`uniserve.executor.video.media`).
//! A thread of the worker reads the pipe into a mailbox and wakes the
//! service; the pipe's write and read order the task's results before the
//! worker reads them. What the tasks wrote is published once every task of
//! a call reported, after the batch's commit point.

use std::collections::HashMap;
use std::io::Read;
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use ipc::{Batch as BatchPlan, Call, CallStatus, ErrorCode, MediaSource, TensorRef};

use crate::executor::PendingOutput;
use crate::executor::batch::HostStage;
use crate::executor::products::{ProductOwners, Publications, export};
use crate::storage::ProductBuffer;
use crate::stream::CUDAStream;
use crate::transfer::SharedRead;
use crate::{Error, Result, lock};

/// A host task that finished.
pub(crate) const DONE: i64 = 1;

/// Fields of a report's header: batch, call, row, status, value and payload
/// bytes, each a little-endian `i64`.
const HEADER_FIELDS: usize = 6;

/// Bytes of a report's header.
const HEADER: usize = HEADER_FIELDS * size_of::<i64>();

/// One finished host task of a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Report {
    /// The row of the call's product the task wrote (an encoded unit's), or
    /// zero.
    pub(crate) row: i64,
    pub(crate) status: i64,
    /// The task's value: an encoded unit's framed length.
    pub(crate) value: i64,
    /// The task's bytes: a finished MP4.
    pub(crate) payload: Vec<u8>,
}

/// The reports received and not yet taken, by `(batch, call)`.
#[derive(Default)]
struct Mailbox {
    reports: HashMap<(u64, usize), Vec<Report>>,
    wake: Option<ipc::Wake>,
}

/// The pipe host tasks report through, its reader thread and the mailbox
/// it fills.
pub(crate) struct Reports {
    mailbox: Arc<Mutex<Mailbox>>,
    /// The end the executor writes; closing it ends the reader.
    writer: Option<OwnedFd>,
    reader: Option<JoinHandle<()>>,
}

impl Reports {
    pub(crate) fn new() -> Result<Self> {
        let (read, write) = rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC)
            .map_err(|error| Error::transport_source("creating the host report pipe", error))?;
        let mailbox = Arc::new(Mutex::new(Mailbox::default()));
        let shared = Arc::clone(&mailbox);
        let reader = std::thread::Builder::new()
            .name("uniserve-host-reports".into())
            .spawn(move || read_reports(std::fs::File::from(read), &shared))
            .map_err(|error| Error::transport_source("host report thread", error))?;
        Ok(Self {
            mailbox,
            writer: Some(write),
            reader: Some(reader),
        })
    }

    /// The descriptor the executor writes reports to.
    pub(crate) fn descriptor(&self) -> Result<i64> {
        self.writer
            .as_ref()
            .map(|writer| i64::from(writer.as_raw_fd()))
            .ok_or(Error::State("the host report pipe is closed"))
    }

    /// The mailbox launched batches take their reports from.
    pub(crate) fn inbox(&self) -> Inbox {
        Inbox(Arc::clone(&self.mailbox))
    }

    pub(crate) fn set_wake(&self, wake: Option<ipc::Wake>) {
        lock(&self.mailbox).wake = wake;
    }

    /// Close the pipe once the executor's host thread stopped, and wait for
    /// the reader to end.
    pub(crate) fn close(&mut self) {
        drop(self.writer.take());
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

/// Read reports until the pipe closes, each into the mailbox, waking the
/// service behind it.
fn read_reports(mut pipe: std::fs::File, mailbox: &Mutex<Mailbox>) {
    let mut header = [0_u8; HEADER];
    loop {
        if pipe.read_exact(&mut header).is_err() {
            return;
        }
        let (fields, _) = header.as_chunks::<{ size_of::<i64>() }>();
        let [batch, call, row, status, value, bytes] =
            std::array::from_fn::<i64, HEADER_FIELDS, _>(|index| i64::from_le_bytes(fields[index]));
        let mut payload = vec![0_u8; usize::try_from(bytes).unwrap_or(0)];
        if pipe.read_exact(&mut payload).is_err() {
            return;
        }
        let wake = {
            let mut mailbox = lock(mailbox);
            mailbox
                .reports
                .entry((batch as u64, call as usize))
                .or_default()
                .push(Report {
                    row,
                    status,
                    value,
                    payload,
                });
            mailbox.wake.clone()
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}

/// A launched batch's way to the reports of its host tasks.
pub(crate) struct Inbox(Arc<Mutex<Mailbox>>);

impl Inbox {
    /// Take the reports of call `call` of batch `batch` once `expected` of
    /// them arrived; `None` before.
    pub(crate) fn take(&self, batch: u64, call: usize, expected: usize) -> Option<Vec<Report>> {
        let mut mailbox = lock(&self.0);
        let arrived = mailbox.reports.get(&(batch, call)).map_or(0, Vec::len);
        if arrived < expected {
            return None;
        }
        Some(mailbox.reports.remove(&(batch, call)).unwrap_or_default())
    }

    /// Forget the reports of a batch that will not be materialized.
    pub(crate) fn discard(&self, batch: u64) {
        lock(&self.0)
            .reports
            .retain(|&(owner, _), _| owner != batch);
    }
}

/// How a call publishes what its host tasks wrote.
pub(crate) enum Publication {
    /// Products the task wrote whole: host products, or device products the
    /// task copied there (fenced on `stream`), exported over the device
    /// transport.
    Products {
        written: Vec<(TensorRef, ProductBuffer)>,
        stream: Option<Arc<CUDAStream>>,
    },
    /// One encoded unit per task in consecutive rows of one product, each
    /// exported up to its framed length.
    Encoded {
        product: TensorRef,
        write: ProductBuffer,
        /// The product row each task framed into.
        rows: Vec<u64>,
    },
    /// The finished MP4, delivered by the component's output rank.
    Media,
}

/// Release borrowed shared-storage reads: their producer may reuse the rows.
pub(crate) fn release_borrowed(borrowed: &[Arc<Mutex<SharedRead>>]) {
    for read in borrowed {
        lock(read).release();
    }
}

/// One call's host tasks in a launched batch, with how the call publishes
/// what they wrote.
pub(crate) struct HostCall {
    /// The call's index in the batch.
    pub(crate) index: usize,
    /// The reports the executor sends for it.
    pub(crate) reports: usize,
    pub(crate) publication: Option<Publication>,
    pub(crate) borrowed: Vec<Arc<Mutex<SharedRead>>>,
    /// The reports, once all arrived.
    received: Option<Vec<Report>>,
}

impl HostCall {
    pub(crate) fn new(
        index: usize,
        reports: usize,
        publication: Option<Publication>,
        borrowed: Vec<Arc<Mutex<SharedRead>>>,
    ) -> Self {
        Self {
            index,
            reports,
            publication,
            borrowed,
            received: None,
        }
    }
}

/// The host tasks a launched batch's results wait for. What they write is
/// published once every task reported, after the commit point: such a
/// product is committed and its exports registered at once.
pub(crate) struct HostTasks {
    pub(crate) calls: Vec<HostCall>,
    pub(crate) inbox: Inbox,
    pub(crate) batch: u64,
}

impl HostStage for HostTasks {
    fn poll(&mut self) -> Result<bool> {
        let mut done = true;
        for call in &mut self.calls {
            if call.received.is_none() {
                call.received = self.inbox.take(self.batch, call.index, call.reports);
                if call.received.is_some() {
                    release_borrowed(&call.borrowed);
                }
            }
            done &= call.received.is_some();
        }
        Ok(done)
    }

    fn finish(
        &mut self,
        plan: &BatchPlan,
        products: ProductOwners<'_>,
        outputs: &mut [PendingOutput],
    ) -> Result<()> {
        for host in std::mem::take(&mut self.calls) {
            let (Some(call), Some(output)) =
                (plan.calls.get(host.index), outputs.get_mut(host.index))
            else {
                return Err(Error::Invariant("a host task names no call".into()));
            };
            host.finish(call, products, output)?;
        }
        Ok(())
    }

    fn cancel(&mut self) {
        // The executor's host thread finishes what it started; its reports
        // are dropped, and borrowed rows are released when they arrive or
        // the inbox closes.
        self.inbox.discard(self.batch);
        for call in &self.calls {
            release_borrowed(&call.borrowed);
        }
    }
}

impl HostCall {
    /// Publish what the call's finished tasks wrote, or fail the call when
    /// a task did not succeed (its host products are abandoned then).
    fn finish(
        self,
        call: &Call,
        owners: ProductOwners<'_>,
        output: &mut PendingOutput,
    ) -> Result<()> {
        let reports = self.received.unwrap_or_default();
        if reports.iter().any(|report| report.status != DONE) {
            tracing::error!("a host call failed; the executor logged why");
            output.output.status = CallStatus::Error;
            output.output.error_code = Some(ErrorCode::ComputeError);
            let writes: Vec<ProductBuffer> = match self.publication {
                Some(Publication::Products { written, .. }) => {
                    written.into_iter().map(|(_, write)| write).collect()
                }
                Some(Publication::Encoded { write, .. }) => vec![write],
                _ => Vec::new(),
            };
            return owners.store.abandon_writes(&writes);
        }

        // Written by the host after the commit point: each product is
        // committed, exported and its exports registered at once.
        let mut publications = Publications::default();
        match self.publication {
            Some(Publication::Products { written, stream }) => {
                for (product, write) in written {
                    let view = write.view();
                    // A device product's write and export record their
                    // fences on the host stream; its copies drained already.
                    let device = matches!(view.device(), crate::ffi::MemoryKind::Cuda(_));
                    let fence = if device {
                        stream.as_deref().map(CUDAStream::stream).transpose()?
                    } else {
                        None
                    };
                    owners.store.write(&write, &view, fence, None)?;
                    owners.store.commit_writes(std::slice::from_ref(&write))?;
                    let (exported, locations) =
                        export(owners, call, &product, &write, fence.is_none(), None, fence)?;
                    publications.accept(write.id(), locations, exported, output)?;
                    publications.register(owners.store)?;
                }
            }
            Some(Publication::Encoded {
                product,
                write,
                rows,
            }) => {
                let view = write.view();
                owners.store.write(&write, &view, None, None)?;
                owners.store.commit_writes(std::slice::from_ref(&write))?;
                let width = view.shape().get(1).copied().unwrap_or(0) as u64;
                let mut regions = Vec::with_capacity(rows.len());
                for (local, &row) in rows.iter().enumerate() {
                    let report = reports
                        .iter()
                        .find(|report| report.row == local as i64)
                        .ok_or_else(|| {
                            Error::Invariant("an encoding task reported no unit".into())
                        })?;
                    let length = u64::try_from(report.value).unwrap_or(u64::MAX);
                    if length > width {
                        return Err(Error::Invariant("a framed unit exceeds its row".into()));
                    }
                    regions.push(vec![row..row + 1, 0..length]);
                }
                let (exported, locations) =
                    export(owners, call, &product, &write, true, Some(&regions), None)?;
                publications.accept(write.id(), locations, exported, output)?;
                publications.register(owners.store)?;
            }
            Some(Publication::Media) if output.reports_output => {
                for report in &reports {
                    if !report.payload.is_empty() {
                        let source = MediaSource::publish(&report.payload).map_err(|error| {
                            Error::transport_source("publishing media failed", error)
                        })?;
                        output.set_media(source)?;
                    }
                }
            }
            Some(Publication::Media) | None => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One report as the executor writes it.
    fn report(fields: [i64; 5], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for field in fields.into_iter().chain([payload.len() as i64]) {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        bytes.extend_from_slice(payload);
        bytes
    }

    /// Reports reach the call they name in the order they were sent; a
    /// call's reports are taken once every one it expects arrived, and a
    /// discarded batch's are dropped.
    #[test]
    fn reports_are_taken_by_call_once_every_expected_one_arrived() -> Result<()> {
        let mut reports = Reports::new()?;
        let inbox = reports.inbox();
        let mut message = report([7, 2, 0, DONE, 15], b"unit");
        message.extend(report([7, 3, 0, DONE, 0], b"mp4 bytes"));
        message.extend(report([7, 2, 1, 2, 0], b""));
        message.extend(report([8, 2, 0, DONE, 0], b""));
        let writer = reports
            .writer
            .as_ref()
            .ok_or(Error::State("the host report pipe is closed"))?;
        let mut remaining = &message[..];
        while !remaining.is_empty() {
            let written = rustix::io::write(writer, remaining)
                .map_err(|error| Error::transport_source("writing a report", error))?;
            remaining = &remaining[written..];
        }
        // Closing the pipe lets the reader drain it and end.
        reports.close();

        assert_eq!(inbox.take(7, 2, 3), None);
        assert_eq!(
            inbox.take(7, 2, 2),
            Some(vec![
                Report {
                    row: 0,
                    status: DONE,
                    value: 15,
                    payload: b"unit".to_vec(),
                },
                Report {
                    row: 1,
                    status: 2,
                    value: 0,
                    payload: Vec::new(),
                },
            ])
        );
        inbox.discard(7);
        assert_eq!(inbox.take(7, 3, 1), None);
        assert_eq!(inbox.take(8, 2, 1).map(|taken| taken.len()), Some(1));
        Ok(())
    }
}
