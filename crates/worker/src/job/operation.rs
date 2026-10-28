//! A job's operations, and what performing them produced.
//!
//! A [`Job`] is an ordered list of operations on streams: copies, fills, event
//! records and waits, an executor call, a host function, a host check of what
//! the job's calls returned, a wait for the stream. The control thread builds
//! it: a batch's step and the storage owners it uses record what they would
//! issue ([`Job::on`], an [`Issue`]) and the executor calls the batch makes,
//! and do their host bookkeeping as they are called. Every operation owns its
//! data, so the job moves to the builder thread, which performs the operations
//! in order and posts a [`Completion`] with what the calls returned. The
//! control thread does not wait for it.
//!
//! An operation names its stream, so one job orders work across the rank's
//! device stream and the streams forked from it, and a job of a rank without a
//! CUDA device (no stream) is plain storage writes and calls.
//!
//! A job is performed completely or it fails at one operation. After a failure
//! the remaining copies, fills, calls, host functions and waits are not
//! performed, but the remaining event operations are: an event a job records
//! fences storage its owner already retains on it, and that storage is returned
//! only once the record is made.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

use cuda::{CopyBatch, DeviceId, Event, SourceOrder, Stream, StridedCopy};

use crate::ffi::MethodCall;
use crate::ffi::{DeferredCall, Handle};
use crate::stream::{Deferral, Deferred, EventPool, Fill, HostFunction, Issue, RowFill};
use crate::{Error, Result};

/// One item of work submitted to the builder thread. Every operation owns its
/// data, so a job moves to the thread that performs it and its submitter goes
/// on.
pub(crate) struct Job {
    operations: RefCell<Vec<Operation>>,
    /// The events that complete before the job's completion is posted, and the
    /// device whose streams recorded them.
    after: RefCell<Option<After>>,
    /// The deferred work whose events the operations record, or whose
    /// events were recorded before: whoever submits the job hands it on
    /// ([`Job::deferrals`]).
    deferrals: RefCell<Vec<Deferral>>,
}

/// The events a completion waits for and the device of their streams.
pub(super) type After = (DeviceId, Vec<Arc<Event>>);

/// The copies of one submission.
pub(super) enum Copies {
    Linear(CopyBatch),
    Strided(StridedCopy),
}

pub(super) enum Operation {
    /// Copies unordered among themselves. `_sources` holds the host bytes
    /// the copies read when their caller's sources did not outlive the
    /// recording call.
    Copy {
        stream: Option<Stream>,
        copies: Copies,
        _sources: Box<[u8]>,
    },
    Fill {
        stream: Option<Stream>,
        fill: Fill,
    },
    RowFill {
        stream: Option<Stream>,
        fill: RowFill,
    },
    Record {
        stream: Stream,
        event: Arc<Event>,
    },
    Wait {
        stream: Stream,
        event: Arc<Event>,
    },
    Call(Call),
    /// Run `function` on the host behind the stream's work issued before.
    HostFunction {
        stream: Option<Stream>,
        function: HostFunction,
    },
    /// Wait for the stream's work issued before.
    Synchronize {
        stream: Option<Stream>,
    },
    /// Check what the calls before it returned, on the thread performing
    /// the job; a refusal fails the job here.
    Check(Check),
}

/// A host check of a job's completion so far ([`Job::check`]): it reads what
/// the calls recorded before it returned and the host storage they wrote,
/// touches no stream, and refuses with an error.
pub(crate) type Check = Box<dyn FnOnce(&Completion) -> Result<()> + Send>;

/// An executor or library call of a job, made on the thread performing the job.
pub(super) enum Call {
    /// A library method.
    Method(MethodCall),
    /// A library method whose integers the completion carries.
    Returning(MethodCall),
    /// A library method whose returned object the completion carries.
    Handle(MethodCall),
    /// A function whose returned strings the completion carries.
    Texts(MethodCall),
    /// A library function over values that are no tensors.
    Function(DeferredCall),
}

/// The operation a job failed at.
#[derive(Debug)]
pub(crate) struct Failure {
    /// Its index in the job.
    pub(crate) operation: usize,
    pub(crate) error: Error,
}

/// The result of a job.
///
/// Whoever built the job applies its results in the order it recorded their
/// operations: each accessor takes the next one of its kind.
#[derive(Default)]
pub(crate) struct Completion {
    /// The integers each returning call returned.
    pub(super) returned: VecDeque<Vec<i64>>,
    /// The object each handle call returned.
    pub(super) handles: VecDeque<Handle>,
    /// The strings each text call returned.
    pub(super) texts: VecDeque<Vec<String>>,
    /// Where the job stopped, when it did not run to its end.
    pub(super) failure: Option<Failure>,
}

impl Completion {
    /// The completion of a job that ran to its end, or its failure.
    pub(crate) fn check(mut self) -> std::result::Result<Self, Failure> {
        match self.failure.take() {
            Some(failure) => Err(failure),
            None => Ok(self),
        }
    }

    /// The integers the next returning call returned
    /// ([`Job::call_returning`]): one for a call that returns an integer,
    /// the list's for a call that returns a list of integers.
    pub(crate) fn returned(&mut self) -> Result<Vec<i64>> {
        self.returned.pop_front().ok_or(MISSING)
    }

    /// The integers the returning call performed last returned, without
    /// taking them: what a [`Check`] recorded behind that call reads.
    pub(crate) fn last_returned(&self) -> Option<&[i64]> {
        self.returned.back().map(Vec::as_slice)
    }

    /// The object the next handle call returned ([`Job::call_handle`]).
    pub(crate) fn handle(&mut self) -> Result<Handle> {
        self.handles.pop_front().ok_or(MISSING)
    }

    /// The strings the next text call returned ([`Job::call_texts`]): one
    /// for a call that returns a string, a list's strings in its order.
    pub(crate) fn texts(&mut self) -> Result<Vec<String>> {
        self.texts.pop_front().ok_or(MISSING)
    }
}

pub(super) const MISSING: Error =
    Error::State("a job's completion lacks a result its builder recorded");

impl Default for Job {
    fn default() -> Self {
        Self {
            operations: RefCell::new(Vec::new()),
            after: RefCell::new(None),
            deferrals: RefCell::new(Vec::new()),
        }
    }
}

impl Job {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Operations recorded so far: the index of the next one.
    pub(crate) fn len(&self) -> usize {
        self.operations.borrow().len()
    }

    /// Whether the job holds nothing for its submitter: no operation and
    /// no deferred work.
    pub(crate) fn is_empty(&self) -> bool {
        self.operations.borrow().is_empty() && self.deferrals.borrow().is_empty()
    }

    /// Take the deferred work recorded so far. The job's submitter hands
    /// it to the event pool where it submits the job.
    pub(crate) fn deferrals(&self) -> Vec<Deferral> {
        std::mem::take(&mut self.deferrals.borrow_mut())
    }

    /// The import path of the first library method the job calls, which names
    /// the job when the builder thread is reported stalled inside it; `None`
    /// for a job that calls none (copies, waits).
    pub(crate) fn method(&self) -> Option<String> {
        self.operations
            .borrow()
            .iter()
            .find_map(|operation| match operation {
                Operation::Call(
                    Call::Method(method)
                    | Call::Returning(method)
                    | Call::Handle(method)
                    | Call::Texts(method),
                ) => Some(method.name().to_owned()),
                _ => None,
            })
    }

    fn push(&self, operation: Operation) {
        self.operations.borrow_mut().push(operation);
    }

    /// Keep of the operations recorded from `start` on only the event
    /// operations, in order: a failed attempt's events are still recorded
    /// and waited on, so the storage they fence is returned, and none of
    /// the work recorded between them is performed.
    fn keep_orders(&self, start: usize) {
        let mut operations = self.operations.borrow_mut();
        let attempted = operations.split_off(start);
        operations.extend(attempted.into_iter().filter(Operation::orders));
    }

    /// Record what its user issues as operations of this job on `stream`
    /// (host storage without one).
    pub(crate) fn on<'r>(&'r self, stream: Option<&'r Stream>) -> Recorder<'r> {
        Recorder { job: self, stream }
    }

    /// An eager call of a library method.
    pub(crate) fn call(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Method(method)));
    }

    /// An eager call of a library method that returns an integer or a list
    /// of integers (the record words an execution wrote, say), which the
    /// job's completion carries ([`Completion::returned`]).
    pub(crate) fn call_returning(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Returning(method)));
    }

    /// An eager call of a library method that returns an object the
    /// library holds (a profiler session), which the job's completion
    /// carries ([`Completion::handle`]).
    pub(crate) fn call_handle(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Handle(method)));
    }

    /// An eager call of a function that returns a string or a list of
    /// strings (a report the rank passes on), which the job's completion
    /// carries ([`Completion::texts`]).
    pub(crate) fn call_texts(&self, function: MethodCall) {
        self.push(Operation::Call(Call::Texts(function)));
    }

    /// A host check of what the calls recorded before it returned and
    /// wrote, performed in the job's order on the thread performing it. A
    /// check that refuses fails the job at this operation, so nothing
    /// recorded after it is performed but its event operations.
    pub(crate) fn check(&self, check: Check) {
        self.push(Operation::Check(check));
    }

    /// Post the job's completion only once `events`, recorded on streams of
    /// `device`, have completed: the completion then says that the device
    /// finished what the events fence. The builder thread does not wait for
    /// them; it goes on with its next job. A job a test performs on its own
    /// thread completes when its operations are performed.
    pub(crate) fn after(&self, device: DeviceId, events: Vec<Arc<Event>>) {
        *self.after.borrow_mut() = Some((device, events));
    }

    /// Perform every operation in order on the calling thread, outside the
    /// rank's job queue (`crate::job::JobQueue`), as tests do.
    #[cfg(test)]
    pub(crate) fn perform(self) -> Completion {
        self.perform_operations().0
    }

    /// Perform the operations; also return what the completion waits for.
    pub(super) fn perform_operations(self) -> (Completion, Option<After>) {
        let after = self.after.into_inner();
        let mut completion = Completion::default();
        let mut operations = self.operations.into_inner().into_iter().enumerate();
        for (index, operation) in operations.by_ref() {
            if let Err(error) = operation.perform(&mut completion) {
                completion.failure = Some(Failure {
                    operation: index,
                    error,
                });
                break;
            }
        }

        // After a failure the events are still recorded and ordered, so
        // the storage they fence is returned (see the module
        // documentation). The first failure is the job's.
        for (_, operation) in operations {
            if operation.orders() {
                let _ = operation.perform(&mut completion);
            }
        }
        (completion, after)
    }
}

impl Operation {
    /// Whether the operation only orders work: an event record or wait.
    fn orders(&self) -> bool {
        matches!(self, Self::Record { .. } | Self::Wait { .. })
    }

    fn perform(self, completion: &mut Completion) -> Result<()> {
        match self {
            Self::Copy { stream, copies, .. } => {
                let stream = stream.as_ref();
                match &copies {
                    // SAFETY: the recorder's caller keeps every range valid
                    // until the copies ran; sources read during the call are
                    // the operation's own bytes.
                    Copies::Linear(copies) => unsafe { stream.copy(copies) },
                    // SAFETY: the recorder's caller keeps every range valid
                    // until the copies ran.
                    Copies::Strided(copies) => unsafe { stream.copy_strided(copies) },
                }
            }
            // SAFETY: the recorder's caller keeps the destination valid
            // until the fill ran.
            Self::Fill { stream, fill } => unsafe { stream.as_ref().fill(fill) },
            // SAFETY: as for a fill.
            Self::RowFill { stream, fill } => unsafe { stream.as_ref().memset_2d(fill) },
            Self::Record { stream, event } => (&stream).record(&event),
            Self::Wait { stream, event } => (&stream).wait(&event),
            Self::Call(call) => call.perform(completion),
            Self::HostFunction { stream, function } => stream.as_ref().launch_host_fn(function),
            Self::Synchronize { stream } => Issue::synchronize(&stream.as_ref()),
            Self::Check(check) => check(completion),
        }
    }
}

impl Call {
    fn perform(self, completion: &mut Completion) -> Result<()> {
        match self {
            Self::Method(method) => drop(method.invoke()?),
            Self::Returning(method) => {
                let name = method.name().to_owned();
                let value = method.invoke()?;
                let integers = i64::try_from(value.clone())
                    .map(|integer| vec![integer])
                    .or_else(|_| crate::ffi::integers(value))
                    .map_err(|error| Error::numerical(&name, &error))?;
                completion.returned.push_back(integers);
            }
            Self::Handle(method) => {
                let object = Handle::new(method.invoke()?)?;
                completion.handles.push_back(object);
            }
            Self::Texts(function) => {
                let name = function.name().to_owned();
                let value = function.invoke()?;
                let texts = match tvm_ffi::String::try_from(value.clone()) {
                    Ok(text) => vec![text.as_str().to_owned()],
                    Err(_) => {
                        crate::ffi::texts(value).map_err(|error| Error::numerical(&name, &error))?
                    }
                };
                completion.texts.push_back(texts);
            }
            Self::Function(function) => function.invoke()?,
        }
        Ok(())
    }
}

/// Records into a job what its user issues on one stream.
#[derive(Clone, Copy)]
pub(crate) struct Recorder<'r> {
    job: &'r Job,
    stream: Option<&'r Stream>,
}

impl Recorder<'_> {
    fn owned(&self) -> Option<Stream> {
        self.stream.cloned()
    }
}

impl Issue for Recorder<'_> {
    type Host = Self;

    fn stream(&self) -> Option<&Stream> {
        self.stream
    }

    fn host(&self) -> Self {
        Self {
            job: self.job,
            stream: None,
        }
    }

    unsafe fn copy(&self, copies: &CopyBatch) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let (copies, sources) = match copies.order() {
            SourceOrder::Stream => (copies.clone(), Box::default()),
            // Sources read during the recording call are owned by the
            // operation, which reads them during the driver's call.
            SourceOrder::DuringCall => {
                let total = copies.iter().map(|(_, _, bytes)| bytes).sum();
                let mut sources = Vec::with_capacity(total);
                for (_, source, bytes) in copies.iter() {
                    // SAFETY: the caller's sources are host storage readable
                    // during this call.
                    sources.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(source as *const u8, bytes)
                    });
                }
                let sources = sources.into_boxed_slice();

                // The boxed bytes do not move with the operation.
                let mut owned = CopyBatch::new(SourceOrder::DuringCall);
                let mut offset = 0;
                for (destination, _, bytes) in copies.iter() {
                    owned.push(destination, sources.as_ptr() as u64 + offset as u64, bytes);
                    offset += bytes;
                }
                (owned, sources)
            }
        };
        self.job.push(Operation::Copy {
            stream: self.owned(),
            copies: Copies::Linear(copies),
            _sources: sources,
        });
        Ok(())
    }

    unsafe fn copy_strided(&self, copies: &StridedCopy) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        if copies.linear().order() != SourceOrder::Stream {
            return Err(Error::Invariant(
                "a job records strided copies whose sources are read in stream order".into(),
            ));
        }
        self.job.push(Operation::Copy {
            stream: self.owned(),
            copies: Copies::Strided(copies.clone()),
            _sources: Box::default(),
        });
        Ok(())
    }

    unsafe fn fill(&self, fill: Fill) -> Result<()> {
        self.job.push(Operation::Fill {
            stream: self.owned(),
            fill,
        });
        Ok(())
    }

    unsafe fn memset_2d(&self, fill: RowFill) -> Result<()> {
        self.job.push(Operation::RowFill {
            stream: self.owned(),
            fill,
        });
        Ok(())
    }

    fn record(&self, event: &Arc<Event>) -> Result<()> {
        let stream = self
            .owned()
            .ok_or(Error::State("an event is recorded on a stream"))?;
        // The event is incomplete from now until the record is made.
        event.queue_record();
        self.job.push(Operation::Record {
            stream,
            event: Arc::clone(event),
        });
        Ok(())
    }

    fn wait(&self, event: &Arc<Event>) -> Result<()> {
        let stream = self
            .owned()
            .ok_or(Error::State("an event is waited on by a stream"))?;
        self.job.push(Operation::Wait {
            stream,
            event: Arc::clone(event),
        });
        Ok(())
    }

    fn defer(&self, _events: &EventPool, fences: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
        // Nothing to wait for: the work runs now, as the pool runs it.
        if fences.is_empty() {
            return owner();
        }
        self.job.deferrals.borrow_mut().push(Deferral {
            events: fences,
            owner,
        });
        Ok(())
    }

    fn call(&self, call: DeferredCall) -> Result<()> {
        self.job.push(Operation::Call(Call::Function(call)));
        Ok(())
    }

    fn launch_host_fn(&self, function: HostFunction) -> Result<()> {
        self.job.push(Operation::HostFunction {
            stream: self.owned(),
            function,
        });
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.job.push(Operation::Synchronize {
            stream: self.owned(),
        });
        Ok(())
    }

    fn attempt<T>(&self, record: impl FnOnce() -> Result<T>) -> Result<T> {
        let start = self.job.len();
        let result = record();
        if result.is_err() {
            self.job.keep_orders(start);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::Function;

    /// A library function that fails.
    fn failing() -> DeferredCall {
        let function = tvm_ffi::Function::from_packed(|_| {
            Err(tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, "refused", ""))
        });
        DeferredCall::new(Function::new("test.failing", function), Vec::new())
    }

    fn byte(address: *mut u8, value: u8) -> Fill {
        Fill {
            address: address as u64,
            value: u32::from(value),
            width: 1,
            count: 1,
        }
    }

    #[test]
    fn host_functions_and_waits_run_in_the_jobs_order() -> Result<()> {
        use std::sync::atomic::{AtomicU8, Ordering};

        let mut target = 0_u8;
        let seen = Arc::new(AtomicU8::new(0));
        let job = Job::new();
        // SAFETY: one byte of the test.
        unsafe { job.on(None).fill(byte(&raw mut target, 5)) }?;
        let address = &raw const target as usize;
        let observed = Arc::clone(&seen);
        job.on(None).launch_host_fn(Box::new(move || {
            // SAFETY: the test's byte, read on the thread performing the job.
            observed.store(unsafe { *(address as *const u8) }, Ordering::Relaxed);
        }))?;
        Issue::synchronize(&job.on(None))?;

        // Recording runs nothing; performing runs the function behind the
        // fill recorded before it.
        assert_eq!(seen.load(Ordering::Relaxed), 0);
        job.perform().check().unwrap();
        assert_eq!(seen.load(Ordering::Relaxed), 5);
        Ok(())
    }

    #[test]
    fn a_row_fill_writes_its_rows_when_the_job_is_performed() -> Result<()> {
        // Three rows of four words; the fill covers two words of each of
        // the first two rows, from the row's second word.
        let mut words = [0_u32; 12];
        let job = Job::new();
        let fill = RowFill {
            address: (&raw mut words[1]) as u64,
            pitch: 16,
            value: 0x3F80_0000,
            width: 4,
            count: 2,
            height: 2,
        };
        // SAFETY: the rows lie inside the test's array.
        unsafe { job.on(None).memset_2d(fill) }?;

        assert_eq!(words, [0; 12]);
        job.perform().check().unwrap();
        let one = 0x3F80_0000;
        assert_eq!(words, [0, one, one, 0, 0, one, one, 0, 0, 0, 0, 0]);

        // The same fill issued at once, on rows of bytes.
        let mut bytes = [0_u8; 6];
        let fill = RowFill {
            address: bytes.as_mut_ptr() as u64,
            pitch: 3,
            value: 7,
            width: 1,
            count: 2,
            height: 2,
        };
        // SAFETY: the rows lie inside the test's array.
        unsafe { None::<&Stream>.memset_2d(fill) }?;
        assert_eq!(bytes, [7, 7, 0, 7, 7, 0]);
        Ok(())
    }

    #[test]
    fn a_returning_call_hands_its_integers_to_the_completion() -> Result<()> {
        let count = Function::new(
            "test.count",
            tvm_ffi::Function::from_packed(|_| Ok(tvm_ffi::Any::from(48_i64))),
        );
        let counts = Function::new(
            "test.counts",
            tvm_ffi::Function::from_packed(|_| {
                Ok(tvm_ffi::Any::from(crate::ffi::integer_list([12, 36])))
            }),
        );
        let job = Job::new();
        job.call_returning(MethodCall::new(&count, Vec::new()));
        job.call(MethodCall::new(&count, Vec::new()));
        job.call_returning(MethodCall::new(&counts, Vec::new()));
        // An object the library holds comes back as its handle.
        job.call_handle(MethodCall::new(&counts, Vec::new()));

        // Only returning calls are kept, in recording order.
        let mut completion = job.perform().check().unwrap();
        assert_eq!(completion.returned()?, vec![48]);
        assert_eq!(completion.returned()?, vec![12, 36]);
        assert!(completion.returned().is_err());
        assert!(completion.handle().is_ok());
        assert!(completion.handle().is_err());
        Ok(())
    }

    #[test]
    fn a_check_reads_what_the_calls_before_it_returned_and_its_refusal_stops_the_job() -> Result<()>
    {
        let count = Function::new(
            "test.count",
            tvm_ffi::Function::from_packed(|_| Ok(tvm_ffi::Any::from(5_i64))),
        );
        let mut written = [0_u8; 1];
        let seen = Arc::new(std::sync::Mutex::new(None));
        let job = Job::new();
        job.call_returning(MethodCall::new(&count, Vec::new()));
        let noted = Arc::clone(&seen);
        job.check(Box::new(move |completion| {
            *crate::lock(&noted) = completion.last_returned().map(<[i64]>::to_vec);
            Ok(())
        }));
        job.check(Box::new(|_| Err(Error::Invalid("refused".into()))));
        // SAFETY: a byte of the test's array, which outlives the job.
        unsafe { job.on(None).fill(byte(&raw mut written[0], 1)) }?;

        // The first check saw the call's integers; the second stopped the
        // job before the fill behind it.
        let failure = job.perform().check().err().unwrap();
        assert_eq!(failure.operation, 2);
        assert_eq!(*crate::lock(&seen), Some(vec![5]));
        assert_eq!(written, [0]);
        Ok(())
    }

    #[test]
    fn a_failed_attempt_performs_none_of_its_work() -> Result<()> {
        let mut target = [0_u8; 3];
        let job = Job::new();
        let issue = job.on(None);
        // SAFETY: bytes of the test's array, here and below.
        unsafe { issue.fill(byte(&raw mut target[0], 5)) }?;

        // An attempt that fails after recording work: its storage is given
        // back, so its work is dropped from the job.
        let failed: Result<()> = issue.attempt(|| {
            // SAFETY: a byte of the test's array, which outlives the job.
            unsafe { issue.fill(byte(&raw mut target[1], 6)) }?;
            Err(Error::State("the attempt failed"))
        });
        assert!(failed.is_err());

        // What was recorded before and after it stays, as does an attempt
        // that succeeds.
        // SAFETY: a byte of the test's array, which outlives the job.
        issue.attempt(|| unsafe { issue.fill(byte(&raw mut target[2], 7)) })?;
        assert_eq!(job.len(), 2);
        job.perform().check().unwrap();
        assert_eq!(target, [5, 0, 7]);

        // A stream issued a failed attempt's work at once.
        let mut issued = 0_u8;
        let failed: Result<()> = None::<&Stream>.attempt(|| {
            // SAFETY: a byte the test owns, filled at once while it is live.
            unsafe { None::<&Stream>.fill(byte(&raw mut issued, 9)) }?;
            Err(Error::State("the attempt failed"))
        });
        assert!(failed.is_err());
        assert_eq!(issued, 9);
        Ok(())
    }

    #[test]
    fn host_operations_run_when_the_job_is_performed() -> Result<()> {
        let mut target = [0_u8; 4];
        let job = Job::new();
        {
            // The copied source is gone before the job is performed.
            let source = [1_u8, 2, 3, 4];
            let mut copies = CopyBatch::new(SourceOrder::DuringCall);
            copies.push(target.as_mut_ptr() as u64, source.as_ptr() as u64, 4);
            // SAFETY: host arrays of the test.
            unsafe { job.on(None).copy(&copies) }?;
        }
        // SAFETY: one byte of the test's array.
        unsafe { job.on(None).fill(byte(&raw mut target[3], 9)) }?;

        // Recording issues nothing.
        assert_eq!(target, [0; 4]);
        assert_eq!(job.len(), 2);

        assert!(job.perform().check().is_ok());
        assert_eq!(target, [1, 2, 3, 9]);
        Ok(())
    }

    #[test]
    fn a_failed_call_stops_the_job_and_names_its_operation() -> Result<()> {
        let mut written = [0_u8; 2];
        let job = Job::new();
        // SAFETY: bytes of the test's array.
        unsafe { job.on(None).fill(byte(&raw mut written[0], 1)) }?;
        job.on(None).call(failing())?;
        // SAFETY: as above.
        unsafe { job.on(None).fill(byte(&raw mut written[1], 1)) }?;

        let failure = job.perform().check().err().unwrap();
        assert_eq!(failure.operation, 1);
        assert!(matches!(failure.error, Error::Numerical { .. }));
        assert_eq!(written, [1, 0]);

        // The same call issued at once fails its caller.
        assert!(None::<&Stream>.call(failing()).is_err());
        Ok(())
    }
}
