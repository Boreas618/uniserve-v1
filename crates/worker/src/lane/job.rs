//! Jobs: the work a lane's execution thread performs, and what it posts
//! back.
//!
//! A [`Job`] is an ordered list of operations on streams: copies, fills,
//! event records and waits, a graph launch, an eager call, output reads, an
//! expert step. The control thread builds it: a runner and the storage
//! owners it calls record what they would issue ([`Job::on`], an
//! [`Issue`]), and do their host bookkeeping as they are called. The lane's
//! execution thread performs the operations in order ([`Job::perform`]) and
//! posts a [`Completion`] with what the eager calls returned and what the
//! output reads produced.
//!
//! An operation names its stream, so one job orders work across the lane's
//! stream and the streams forked from it (expert microbatches), and a job
//! of a rank without a CUDA device (no stream) is plain memory writes and
//! calls.
//!
//! A job is performed completely or it fails at one operation. After a
//! failure the remaining copies, fills, launches, calls and reads are not
//! performed, but the remaining event operations are: an event a job
//! records fences storage its owner already retains on it, and that
//! storage is returned only once the record is made.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use uniserve_cuda::{CopyBatch, Event, SourceOrder, Stream, StridedCopy};
use uniserve_ipc::Wake;

use crate::cuda_graph::{CUDAGraph, GraphBreaks};
use crate::ffi::{DeferredCall, Handle};
use crate::parallel::joins::GraphFunctions;
use crate::parallel::{ExpertUnion, MicrobatchForward};
use crate::runner::methods::MethodCall;
use crate::stream::{EventPool, Fill, Issue};
use crate::{Error, Result};

/// One item of work submitted to a lane.
///
/// `'a` is what the forward of a capability not yet converted borrows from
/// its submitter ([`Job::forward`]); every other operation owns its data.
pub(crate) struct Job<'a> {
    operations: RefCell<Vec<Operation<'a>>>,
}

/// The copies of one submission.
enum Copies {
    Linear(CopyBatch),
    Strided(StridedCopy),
}

enum Operation<'a> {
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
    Record {
        stream: Stream,
        event: Arc<Event>,
    },
    Wait {
        stream: Stream,
        event: Arc<Event>,
    },
    /// Fire `wake` from `notification` once `event` completes.
    Wake {
        notification: Arc<Stream>,
        event: Arc<Event>,
        wake: Wake,
    },
    /// Every segment of a captured forward, and the breaks between them.
    Launch {
        stream: Stream,
        graph: CUDAGraph,
    },
    Call(Call<'a>),
    /// Copy `bytes` at `source` to the host and wait for them.
    Read {
        stream: Option<Stream>,
        source: u64,
        bytes: usize,
    },
    ExpertStep(ExpertStep),
}

/// Work the execution thread hands to the Python thread and waits for, or
/// runs itself where it enters Python on its own.
enum Call<'a> {
    /// A runner method.
    Method(MethodCall),
    /// A library function over values that are no tensors.
    Function(DeferredCall),
    Capture(Capture),
    /// The forward of a capability not yet converted, as it issues itself.
    Forward(Forward<'a>),
}

/// The capture of one runner method into a graph: `capture_begin`, the
/// method, `capture_end`, on the Python thread.
pub(crate) struct Capture {
    pub(crate) functions: GraphFunctions,
    /// The pool the captured allocations live in.
    pub(crate) pool: Handle,
    /// The raw handle and device of the stream the method issues on.
    pub(crate) stream: usize,
    pub(crate) device: i32,
    pub(crate) method: MethodCall,
    /// Where the method's graph breaks are noted while it is captured.
    pub(crate) breaks: Option<GraphBreaks>,
}

/// A forward that borrows its submitter, which waits for the job.
struct Forward<'a>(Box<dyn FnOnce() + 'a>);

// SAFETY: the forward runs on the execution thread while its submitter
// waits for the job and touches nothing it lent. What it borrows has no
// thread affinity: the rank's owners hold reference-counted TVM-FFI values
// with atomic counts, driver handles the driver accepts from any thread,
// and locks they do not hold across a job.
unsafe impl Send for Forward<'_> {}

/// One expert step: the union's agreement on the step's capacity, then the
/// forward of every populated microbatch and the join of every empty one at
/// that capacity.
pub(crate) struct ExpertStep {
    pub(crate) union: Arc<ExpertUnion>,
    /// The lane's stream, which every microbatch stream forks from and
    /// joins.
    pub(crate) lane: Stream,
    /// Tokens this rank's forwards send to each expert layer, and the kind
    /// of forward, for the agreement.
    pub(crate) tokens: usize,
    pub(crate) kind: i64,
    /// The capacity a warmup step runs at.
    pub(crate) forced: Option<usize>,
    /// The step at every capacity it can be agreed at.
    pub(crate) variants: Vec<ExpertVariant>,
    /// What each populated microbatch runs behind its forward in the same
    /// turn, whatever the capacity (the selection of rows that sample).
    pub(crate) after: Vec<Option<Job<'static>>>,
}

/// An expert step at one capacity.
pub(crate) struct ExpertVariant {
    pub(crate) capacity: usize,
    /// One forward per microbatch; an empty microbatch joins.
    pub(crate) forwards: Vec<Option<MicrobatchForward>>,
    /// What follows the step on the execution thread at this capacity (the
    /// captures of forwards that ran as their warmup).
    pub(crate) then: Job<'static>,
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
    /// The graph of each capture.
    captured: VecDeque<CUDAGraph>,
    /// The bytes of each output read.
    reads: VecDeque<Vec<u8>>,
    /// The capacity each expert step was agreed at.
    capacities: VecDeque<usize>,
    /// Where the job stopped, when it did not run to its end.
    failure: Option<Failure>,
}

impl Completion {
    /// The completion of a job that ran to its end, or its failure.
    pub(crate) fn check(mut self) -> std::result::Result<Self, Failure> {
        match self.failure.take() {
            Some(failure) => Err(failure),
            None => Ok(self),
        }
    }

    /// The graph of the next capture.
    pub(crate) fn captured(&mut self) -> Result<CUDAGraph> {
        self.captured.pop_front().ok_or(MISSING)
    }

    /// The bytes of the next output read.
    pub(crate) fn read(&mut self) -> Result<Vec<u8>> {
        self.reads.pop_front().ok_or(MISSING)
    }

    /// The capacity the next expert step was agreed at.
    pub(crate) fn capacity(&mut self) -> Result<usize> {
        self.capacities.pop_front().ok_or(MISSING)
    }
}

const MISSING: Error = Error::State("a job's completion lacks a result its builder recorded");

impl Default for Job<'_> {
    fn default() -> Self {
        Self {
            operations: RefCell::new(Vec::new()),
        }
    }
}

impl<'a> Job<'a> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Operations recorded so far: the index of the next one.
    pub(crate) fn len(&self) -> usize {
        self.operations.borrow().len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.operations.borrow().is_empty()
    }

    fn push(&self, operation: Operation<'a>) {
        self.operations.borrow_mut().push(operation);
    }

    /// Record what its user issues as operations of this job on `stream`
    /// (host memory without one).
    pub(crate) fn on<'r>(&'r self, stream: Option<&'r Stream>) -> Recorder<'r, 'a> {
        Recorder { job: self, stream }
    }

    /// An eager call of a runner method.
    pub(crate) fn call(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Method(method)));
    }

    /// The capture of a runner method. Its completion carries the graph.
    pub(crate) fn capture(&self, capture: Capture) {
        self.push(Operation::Call(Call::Capture(capture)));
    }

    /// The forward of a capability not yet converted: it issues its own
    /// stream work and Python calls when the execution thread runs it, and
    /// keeps its result where its submitter reads it.
    pub(crate) fn forward(&self, forward: impl FnOnce() + 'a) {
        self.push(Operation::Call(Call::Forward(Forward(Box::new(forward)))));
    }

    pub(crate) fn expert_step(&self, step: ExpertStep) {
        self.push(Operation::ExpertStep(step));
    }

    /// Continue this job with the operations of `other`.
    pub(crate) fn append(&self, other: Job<'a>) {
        self.operations
            .borrow_mut()
            .append(&mut other.operations.into_inner());
    }

    /// Perform every operation in order on the calling thread: the lane's
    /// execution thread, a microbatch thread in its turn of an expert step,
    /// or the submitter itself on a rank without a lane.
    pub(crate) fn perform(self) -> Completion {
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
        completion
    }
}

impl Operation<'_> {
    /// Whether the operation only orders work: an event record, wait or
    /// wake.
    fn orders(&self) -> bool {
        matches!(
            self,
            Self::Record { .. } | Self::Wait { .. } | Self::Wake { .. }
        )
    }

    fn perform(self, completion: &mut Completion) -> Result<()> {
        match self {
            Self::Copy { stream, copies, .. } => {
                let stream = stream.as_ref();
                // SAFETY: the recorder's caller keeps every range valid
                // until the copies ran; sources read during the call are
                // the operation's own bytes.
                match &copies {
                    Copies::Linear(copies) => unsafe { stream.copy(copies) },
                    Copies::Strided(copies) => unsafe { stream.copy_strided(copies) },
                }
            }
            // SAFETY: the recorder's caller keeps the destination valid
            // until the fill ran.
            Self::Fill { stream, fill } => unsafe { stream.as_ref().fill(fill) },
            Self::Record { stream, event } => (&stream).record(&event),
            Self::Wait { stream, event } => (&stream).wait(&event),
            Self::Wake {
                notification,
                event,
                wake,
            } => {
                notification.wait_event(&event)?;
                // The host function only signals the service's wake: it
                // never calls CUDA or blocks, as host functions must not.
                notification.launch_host_fn(move || wake.wake())?;
                Ok(())
            }
            Self::Launch { stream, graph } => graph.launch(&stream),
            Self::Call(call) => call.perform(completion),
            Self::Read {
                stream,
                source,
                bytes,
            } => {
                let mut value = vec![0_u8; bytes];
                match &stream {
                    Some(stream) => {
                        // SAFETY: the reader keeps the source valid until
                        // the job completes; the host bytes outlive the
                        // copy, which the stream is drained for.
                        unsafe {
                            uniserve_cuda::memcpy_async(
                                value.as_mut_ptr() as u64,
                                source,
                                bytes,
                                stream,
                            )
                        }?;
                        stream.synchronize()?;
                    }
                    // SAFETY: host bytes the reader keeps valid.
                    None => unsafe {
                        std::ptr::copy_nonoverlapping(
                            source as *const u8,
                            value.as_mut_ptr(),
                            bytes,
                        );
                    },
                }
                completion.reads.push_back(value);
                Ok(())
            }
            Self::ExpertStep(step) => step.perform(completion),
        }
    }
}

impl Call<'_> {
    fn perform(self, completion: &mut Completion) -> Result<()> {
        match self {
            // No converted forward reads what its method returns.
            Self::Method(method) => drop(method.invoke()?),
            Self::Function(function) => function.invoke()?,
            Self::Capture(capture) => {
                let Capture {
                    functions,
                    pool,
                    stream,
                    device,
                    method,
                    breaks,
                } = capture;
                let captured =
                    functions.capture(&pool, stream, device, || method.invoke().map(drop));

                // The breaks the method reached belong to this capture,
                // whether or not it succeeded.
                let breaks = breaks.as_ref().map_or_else(Vec::new, GraphBreaks::take);
                completion.captured.push_back(captured?.with_breaks(breaks));
            }
            Self::Forward(Forward(forward)) => forward(),
        }
        Ok(())
    }
}

impl ExpertStep {
    /// Agree on the step's capacity over the union's host group, run the
    /// step's forwards and joins at it, then what follows it.
    fn perform(self, completion: &mut Completion) -> Result<()> {
        let capacity = self.union.capacity(self.tokens, self.kind, self.forced)?;
        let variant = self
            .variants
            .into_iter()
            .find(|variant| variant.capacity == capacity)
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "an expert step was agreed at capacity {capacity}, which its job has no forward for"
                ))
            })?;
        for (forward, after) in variant.forwards.iter().zip(self.after) {
            if let (Some(forward), Some(after)) = (forward, after) {
                forward.job.append(after);
            }
        }
        {
            let _range = crate::profiling::range(c"uniserve.expert.step", None);
            self.union.run(&self.lane, variant.forwards, capacity)?;
        }
        completion.capacities.push_back(capacity);

        let then = variant.then.perform();
        completion.captured.extend(then.captured);
        completion.reads.extend(then.reads);
        then.failure.map_or(Ok(()), |failure| Err(failure.error))
    }
}

/// Records into a job what its user issues on one stream.
#[derive(Clone, Copy)]
pub(crate) struct Recorder<'r, 'a> {
    job: &'r Job<'a>,
    stream: Option<&'r Stream>,
}

impl Recorder<'_, '_> {
    fn owned(&self) -> Option<Stream> {
        self.stream.cloned()
    }

    /// The launch of `graph`.
    pub(crate) fn launch(&self, graph: &CUDAGraph) -> Result<()> {
        let stream = self
            .owned()
            .ok_or(Error::State("a graph is launched on a CUDA stream"))?;
        self.job.push(Operation::Launch {
            stream,
            graph: graph.clone(),
        });
        Ok(())
    }

    /// An output read of `bytes` bytes at `source`, which stay valid until
    /// the job completes. The job's completion carries them.
    pub(crate) fn read(&self, source: u64, bytes: usize) {
        self.job.push(Operation::Read {
            stream: self.owned(),
            source,
            bytes,
        });
    }
}

impl Issue for Recorder<'_, '_> {
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
                    // SAFETY: the caller's sources are host memory readable
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

    fn wake(&self, events: &EventPool, event: &Arc<Event>) -> Result<()> {
        if let Some((notification, wake)) = events.notification(event)? {
            self.job.push(Operation::Wake {
                notification,
                event: Arc::clone(event),
                wake,
            });
        }
        Ok(())
    }

    fn call(&self, call: DeferredCall) -> Result<()> {
        self.job.push(Operation::Call(Call::Function(call)));
        Ok(())
    }
}

/// Names a job from its submission to its completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub(super) u64);

/// What a lane's execution thread receives.
enum Message {
    Job(JobId, Job<'static>),
    /// The wake of the control thread, fired behind every completion from
    /// now on; `None` stops it.
    Wake(Option<Wake>),
}

/// The submitting end of a lane's job queue. Only the control thread
/// submits.
#[derive(Clone)]
pub struct JobQueue {
    messages: Sender<Message>,
}

/// The execution thread's end of a lane's job queue.
pub(super) struct Jobs {
    messages: Receiver<Message>,
    /// Where the lanes of a rank post their completions.
    completions: Sender<(JobId, Completion)>,
}

/// Posts the failure of the job a lane is performing if the lane's thread
/// unwinds, so whoever waits for that job learns that the lane is gone.
struct Performing<'a> {
    job: Option<JobId>,
    completions: &'a Sender<(JobId, Completion)>,
}

impl Drop for Performing<'_> {
    fn drop(&mut self) {
        if let Some(job) = self.job {
            let completion = Completion {
                failure: Some(Failure {
                    operation: 0,
                    error: Error::State("the lane's execution thread has exited"),
                }),
                ..Completion::default()
            };
            let _ = self.completions.send((job, completion));
        }
    }
}

impl JobQueue {
    pub(super) fn new(completions: Sender<(JobId, Completion)>) -> (Self, Jobs) {
        let (messages, received) = crossbeam_channel::unbounded();
        (
            Self { messages },
            Jobs {
                messages: received,
                completions,
            },
        )
    }

    /// Queue `job` as `id` on the lane's execution thread, which posts its
    /// completion under that name.
    pub(super) fn start(&self, id: JobId, job: Job<'static>) -> Result<()> {
        self.send(Message::Job(id, job))
    }

    pub(super) fn set_wake(&self, wake: Option<Wake>) -> Result<()> {
        self.send(Message::Wake(wake))
    }

    fn send(&self, message: Message) -> Result<()> {
        self.messages
            .send(message)
            .map_err(|_| Error::State("the lane's execution thread has exited"))
    }
}

impl Jobs {
    /// Perform jobs in arrival order on the calling thread, the lane's
    /// execution thread, until every submitting end is gone. Each
    /// completion is posted under its job's name, and the control thread's
    /// wake fired behind it.
    pub(super) fn run(self) {
        let mut wake = None;
        while let Ok(message) = self.messages.recv() {
            let (id, job) = match message {
                Message::Job(id, job) => (id, job),
                Message::Wake(installed) => {
                    wake = installed;
                    continue;
                }
            };
            let mut performing = Performing {
                job: Some(id),
                completions: &self.completions,
            };
            // The job and everything it holds is gone before its submitter
            // learns that it completed.
            let completion = job.perform();
            performing.job = None;

            if self.completions.send((id, completion)).is_err() {
                break;
            }
            if let Some(wake) = &wake {
                wake.wake();
            }
        }
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
    fn host_operations_run_when_the_job_is_performed_and_reads_come_back() -> Result<()> {
        let mut target = [0_u8; 4];
        let state = [7_u8, 8];
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
        job.on(None).read(state.as_ptr() as u64, 2);

        // Recording issues nothing.
        assert_eq!(target, [0; 4]);
        assert_eq!(job.len(), 3);

        let mut completion = job.perform().check().unwrap();
        assert_eq!(target, [1, 2, 3, 9]);
        assert_eq!(completion.read()?, vec![7, 8]);
        assert!(completion.read().is_err());
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
