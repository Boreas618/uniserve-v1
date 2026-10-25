//! The rank's numerical half: the Python executor
//! (`uniserve.executor.Executor`) and the reports of its host tasks.
//!
//! The worker owns the control and data planes (requests, batches, lanes,
//! products, transfers and memory); the executor owns every numerical
//! decision and computation of the components it serves. The worker calls
//! it at a few points, each a [`MethodCall`] a job of the lane whose stream
//! the call works on records, or a call startup makes itself:
//!
//! | call | when | result |
//! | --- | --- | --- |
//! | `manifest` | after loading | JSON of what capacity planning and the handshake read |
//! | `bind` | after capacity planning | the runners are constructed on the granted storage |
//! | `warmup` | before the rank reports | every runner prepared, warmed and captured |
//! | `start` | before a batch that starts requests opens | JSON of the requests refused |
//! | `plan` | a batch's first job past its predicates | JSON of what each call reads and writes |
//! | `execute` | the batch's job that carries its device work | how each call's forward ran |
//! | `release` | a slot retires | — |
//! | `kernels`, `close` | startup's report, shutdown | — |
//!
//! Host calls (media reading, encoding, muxing) run on the executor's host
//! thread after `execute` returned. Each finished task writes one report to
//! a pipe the worker created ([`Reports`]): a header of six native-endian
//! `i64` (batch, call, row, status, value, payload bytes) and its payload.
//! A thread of the worker reads the pipe into a mailbox and wakes the
//! service; the pipe's write and read order the task's results before the
//! worker reads them.

use std::collections::HashMap;
use std::io::Read;
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use tvm_ffi::Any;

use crate::ffi::{Function, Handle, Interpreter};
use crate::memory::allocation::StorageView;
use crate::profiling::kernels::KernelRecord;
use crate::runner::methods::{Argument, MethodCall};
use crate::{Error, Result, lock};

const EXECUTOR: &str = "uniserve.executor.Executor";

/// A host task that finished.
pub(crate) const DONE: i64 = 1;

/// Bytes of a report's header: batch, call, row, status, value and payload
/// bytes, each a little-endian `i64`.
const HEADER: usize = 6 * size_of::<i64>();

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
    fn new() -> Result<Self> {
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
    fn descriptor(&self) -> Result<i64> {
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
    fn close(&mut self) {
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
        let field = |index: usize| {
            let mut bytes = [0_u8; 8];
            bytes.copy_from_slice(&header[index * 8..index * 8 + 8]);
            i64::from_le_bytes(bytes)
        };
        let (batch, call, row, status, value, bytes) =
            (field(0), field(1), field(2), field(3), field(4), field(5));
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

/// The methods the worker calls, each with the executor first.
struct Functions {
    manifest: Function,
    bind: Function,
    warmup: Function,
    start: Function,
    release: Function,
    plan: Function,
    execute: Function,
    kernels: Function,
    close: Function,
}

/// The rank's Python executor.
pub(crate) struct PythonExecutor {
    executor: Handle,
    functions: Functions,
    reports: Option<Reports>,
}

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}

impl PythonExecutor {
    /// Construct the executor of the loaded model on this rank's
    /// deployment `config` (JSON; see `uniserve.executor.Executor`).
    pub(crate) fn new(loaded: &Handle, config: &str) -> Result<Self> {
        let method = |name: &str| Interpreter::import(&format!("{EXECUTOR}.{name}"));
        let executor = Handle::new(
            Interpreter::import(EXECUTOR)?.call_values(&[loaded.to_any(), text(config)])?,
        )?;
        Ok(Self {
            executor,
            functions: Functions {
                manifest: method("manifest")?,
                bind: method("bind")?,
                warmup: method("warmup")?,
                start: method("start")?,
                release: method("release")?,
                plan: method("plan")?,
                execute: method("execute")?,
                kernels: method("kernels")?,
                close: method("close")?,
            },
            reports: None,
        })
    }

    fn call(&self, function: &Function, arguments: Vec<Argument>) -> MethodCall {
        let mut all = Vec::with_capacity(arguments.len() + 1);
        all.push(Argument::Handle(self.executor.clone()));
        all.extend(arguments);
        MethodCall::new(function, all)
    }

    fn texts(&self, function: &Function, arguments: Vec<Argument>) -> Result<String> {
        let value = self.call(function, arguments).invoke()?;
        tvm_ffi::String::try_from(value)
            .map(|text| text.as_str().to_owned())
            .map_err(|error| Error::numerical(function.name(), &error))
    }

    /// The JSON of what capacity planning and the handshake read.
    pub(crate) fn manifest(&self) -> Result<String> {
        self.texts(&self.functions.manifest, Vec::new())
    }

    /// Construct the executor's runners on the storage capacity planning
    /// granted: `memory` is the JSON `bind` reads, `pools` the graph pools
    /// in the order it names them, `scratches` the scratch of each of its
    /// lanes in the order they serialize and `banks` the latent banks of a
    /// denoising rank. The report pipe is created here, and its descriptor
    /// joins `memory` as `reports`.
    pub(crate) fn bind(
        &mut self,
        mut memory: serde_json::Value,
        pools: Vec<Handle>,
        scratches: Vec<Handle>,
        banks: Option<[StorageView; 2]>,
    ) -> Result<()> {
        let reports = Reports::new()?;
        memory["reports"] = serde_json::Value::from(reports.descriptor()?);
        memory["banks"] = serde_json::Value::from(banks.is_some());
        let mut arguments = vec![Argument::Text(memory.to_string())];
        arguments.extend(pools.into_iter().map(Argument::Handle));
        arguments.extend(scratches.into_iter().map(Argument::Handle));
        arguments.extend(banks.into_iter().flatten().map(Argument::Tensor));
        self.reports = Some(reports);
        self.call(&self.functions.bind, arguments)
            .invoke()
            .map(drop)
    }

    /// Prepare, warm and capture every runner; the caller waits.
    pub(crate) fn warmup(&self) -> Result<()> {
        self.call(&self.functions.warmup, Vec::new())
            .invoke()
            .map(drop)
    }

    /// The call that admits started requests (`requests`: JSON of their
    /// `NewRequest`s), whose completion's text is the JSON of the refused.
    pub(crate) fn start(&self, requests: String) -> MethodCall {
        self.call(&self.functions.start, vec![Argument::Text(requests)])
    }

    /// The call that plans a batch (`step`: JSON), whose completion's text
    /// is the JSON of each call's reads and writes.
    pub(crate) fn plan(&self, step: String) -> MethodCall {
        self.call(&self.functions.plan, vec![Argument::Text(step)])
    }

    /// The call that executes a batch on `tensors`, whose completion's
    /// integers are each call's graph mode.
    pub(crate) fn execute(&self, step: String, tensors: Vec<StorageView>) -> MethodCall {
        let mut arguments = vec![Argument::Text(step)];
        arguments.extend(tensors.into_iter().map(Argument::Tensor));
        self.call(&self.functions.execute, arguments)
    }

    /// Forget retired slots. The release is queued on the builder thread
    /// behind every call made before it; nothing waits for it.
    pub(crate) fn release(&self, slots: &[usize]) {
        let call = self.call(
            &self.functions.release,
            vec![Argument::Integers(
                slots.iter().map(|&slot| slot as i64).collect(),
            )],
        );
        Interpreter::submit(move || {
            if let Err(error) = call.invoke() {
                tracing::error!("releasing retired slots failed: {error}");
            }
        });
    }

    /// The kernels each served component's operators selected, by
    /// component.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<KernelRecord>)>> {
        let listing = self.texts(&self.functions.kernels, Vec::new())?;
        let parsed: Vec<(String, Vec<String>)> = serde_json::from_str(&listing)
            .map_err(|error| Error::Interface(format!("the executor's kernels: {error}")))?;
        Ok(parsed
            .into_iter()
            .map(|(label, records)| {
                (
                    label,
                    records.into_iter().map(KernelRecord::from_text).collect(),
                )
            })
            .collect())
    }

    /// The reports of the executor's host tasks.
    pub(crate) fn reports(&self) -> Result<&Reports> {
        self.reports
            .as_ref()
            .ok_or(Error::State("the executor was not bound"))
    }

    pub(crate) fn set_wake(&self, wake: Option<ipc::Wake>) {
        if let Some(reports) = &self.reports {
            reports.set_wake(wake);
        }
    }

    /// Release the runners after the lanes drained, then close the report
    /// pipe. An aborted close keeps what device work may still address.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let closed = self
            .call(&self.functions.close, vec![Argument::Bool(aborted)])
            .invoke()
            .map(drop);
        if let Some(mut reports) = self.reports.take() {
            reports.close();
        }
        closed
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
