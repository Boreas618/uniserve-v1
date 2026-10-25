//! Jobs: the work a lane's execution thread performs, and what it posts
//! back.
//!
//! A [`Job`] is an ordered list of operations on streams: copies, fills,
//! event records and waits, a graph launch, an eager call, output reads, a
//! host function, a wait for the stream, an expert step. The control thread
//! builds it: a runner and the storage owners it calls record what they
//! would issue ([`Job::on`], an [`Issue`]), and do their host bookkeeping
//! as they are called. Every operation owns its data, so the job moves to
//! the lane's execution thread, which performs the operations in order and
//! posts a [`Completion`] with what the eager calls returned and what the
//! output reads produced. The control thread does not wait for it.
//!
//! An operation names its stream, so one job orders work across the lane's
//! stream and the streams forked from it (expert microbatches), and a job
//! of a rank without a CUDA device (no stream) is plain memory writes and
//! calls.
//!
//! A job is performed completely or it fails at one operation. After a
//! failure the remaining copies, fills, launches, calls, reads, host
//! functions and waits are not performed, but the remaining event
//! operations are: an event a job
//! records fences storage its owner already retains on it, and that
//! storage is returned only once the record is made.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

use cuda::{CopyBatch, DeviceId, Event, SourceOrder, Stream, StridedCopy};

use crate::cuda_graph::{CUDAGraph, GraphBreaks};
use crate::ffi::{DeferredCall, Handle};
use crate::parallel::joins::GraphFunctions;
use crate::parallel::{ExpertUnion, MicrobatchForward};
use crate::runner::methods::MethodCall;
use crate::stream::{Deferral, Deferred, EventPool, Fill, HostFunction, Issue, RowFill};
use crate::{Error, Result};

/// One item of work submitted to a lane. Every operation owns its data, so
/// a job moves to the thread that performs it and its submitter goes on.
pub(crate) struct Job {
    operations: RefCell<Vec<Operation>>,
    /// The events a lane lets complete before it posts the job's
    /// completion, and the device whose streams recorded them.
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
    /// Every segment of a captured forward, and the breaks between them.
    Launch {
        stream: Stream,
        graph: CUDAGraph,
    },
    Call(Call),
    /// Copy `bytes` at `source` to the host and wait for them.
    Read {
        stream: Option<Stream>,
        source: u64,
        bytes: usize,
    },
    /// Run `function` on the host behind the stream's work issued before.
    HostFunction {
        stream: Option<Stream>,
        function: HostFunction,
    },
    /// Wait for the stream's work issued before.
    Synchronize {
        stream: Option<Stream>,
    },
    ExpertStep(ExpertStep),
    /// The union's next step for a rank with no forward of its own: one
    /// round of the agreement and, when a step was agreed, its join.
    ExpertJoin {
        union: Arc<ExpertUnion>,
        /// The rank is shutting down and leaves the union.
        leaving: bool,
    },
    /// The end of the union's startup warmup on this rank: the announced
    /// steps it follows, then the capture of its joins.
    ExpertWarmup(Arc<ExpertUnion>),
}

/// Work the execution thread hands to the builder thread and waits for, or
/// runs itself where it enters Python on its own.
pub(super) enum Call {
    /// A runner method.
    Method(MethodCall),
    /// A runner method whose integers the completion carries.
    Returning(MethodCall),
    /// A runner method whose returned object the completion carries.
    Handle(MethodCall),
    /// A runner method whose returned list the completion carries, value
    /// by value.
    Values(MethodCall),
    /// A function whose returned strings the completion carries.
    Texts(MethodCall),
    /// A library function over values that are no tensors.
    Function(DeferredCall),
    Capture(Capture),
}

/// The capture of one runner method into a graph: `capture_begin`, the
/// method, `capture_end`, on the builder thread.
pub(crate) struct Capture {
    pub(crate) functions: GraphFunctions,
    /// The pool the captured allocations on the stream's device live in.
    pub(crate) pool: Handle,
    /// The pool of each other device the method's modules run on (a
    /// generation tower), with its device; empty for a method on one
    /// device.
    pub(crate) peers: Vec<(DeviceId, Handle)>,
    /// The raw handle and device of the stream the method issues on.
    pub(crate) stream: usize,
    pub(crate) device: DeviceId,
    pub(crate) method: MethodCall,
    /// Where the method's graph breaks are noted while it is captured.
    pub(crate) breaks: Option<GraphBreaks>,
}

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
    pub(crate) after: Vec<Option<Job>>,
}

/// An expert step at one capacity.
pub(crate) struct ExpertVariant {
    pub(crate) capacity: usize,
    /// One forward per microbatch; an empty microbatch joins.
    pub(crate) forwards: Vec<Option<MicrobatchForward>>,
    /// What follows the step on the execution thread at this capacity (the
    /// captures of forwards that ran as their warmup).
    pub(crate) then: Job,
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
    pub(super) captured: VecDeque<CUDAGraph>,
    /// The bytes of each output read.
    pub(super) reads: VecDeque<Vec<u8>>,
    /// The capacity each expert step was agreed at.
    pub(super) capacities: VecDeque<usize>,
    /// For each expert join: whether a step ran, and whether every rank
    /// has left the union.
    pub(super) joins: VecDeque<(bool, bool)>,
    /// The integers each returning call returned.
    pub(super) returned: VecDeque<Vec<i64>>,
    /// The object each handle call returned.
    pub(super) handles: VecDeque<Handle>,
    /// The values of the list each value call returned.
    pub(super) values: VecDeque<Vec<Value>>,
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

    /// The next expert join: whether a step ran, and whether every rank
    /// has left the union.
    pub(crate) fn joined(&mut self) -> Result<(bool, bool)> {
        self.joins.pop_front().ok_or(MISSING)
    }

    /// The integers the next returning call returned
    /// ([`Job::call_returning`]): one for a call that returns an integer,
    /// and for a call that returns lists of integers (shapes) each list's
    /// length followed by its integers.
    pub(crate) fn returned(&mut self) -> Result<Vec<i64>> {
        self.returned.pop_front().ok_or(MISSING)
    }

    /// The object the next handle call returned ([`Job::call_handle`]).
    pub(crate) fn handle(&mut self) -> Result<Handle> {
        self.handles.pop_front().ok_or(MISSING)
    }

    /// The values of the list the next value call returned
    /// ([`Job::call_values`]), in the list's order.
    pub(crate) fn values(&mut self) -> Result<Vec<Value>> {
        self.values.pop_front().ok_or(MISSING)
    }

    /// The strings the next text call returned ([`Job::call_texts`]): one
    /// for a call that returns a string, a list's strings in its order.
    pub(crate) fn texts(&mut self) -> Result<Vec<String>> {
        self.texts.pop_front().ok_or(MISSING)
    }
}

pub(super) const MISSING: Error =
    Error::State("a job's completion lacks a result its builder recorded");

/// One value of the list a value call returned ([`Job::call_values`]): a
/// list of numbers.
///
/// The list is converted on the thread that made the call, so its reader
/// holds plain numbers.
pub(crate) enum Value {
    /// A shape or a list of integers.
    Integers(Vec<i64>),
    /// A list of numbers that are not all integers.
    Floats(Vec<f64>),
}

impl Value {
    /// Convert one value of the list `name` returned.
    fn new(value: tvm_ffi::Any, name: &str) -> Result<Self> {
        let numerical = |error: tvm_ffi::Error| Error::numerical(name, &error);
        if value.type_index() == tvm_ffi::TypeIndex::kTVMFFIShape as i32 {
            return crate::ffi::integers(value)
                .map(Self::Integers)
                .map_err(numerical);
        }
        let Ok(numbers) = tvm_ffi::Array::<tvm_ffi::Any>::try_from(value.clone()) else {
            return Err(Error::Interface(format!("{name} returns lists of numbers")));
        };

        let integer = tvm_ffi::TypeIndex::kTVMFFIInt as i32;
        if numbers.iter().all(|number| number.type_index() == integer) {
            return crate::ffi::integers(value)
                .map(Self::Integers)
                .map_err(numerical);
        }
        numbers
            .iter()
            .map(f64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Self::Floats)
            .map_err(numerical)
    }
}

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

    /// The import path of the first runner method the job calls or
    /// captures, which names the job when its lane is reported stalled;
    /// `None` for a job that calls none (a graph launch, copies, waits).
    pub(crate) fn method(&self) -> Option<String> {
        self.operations
            .borrow()
            .iter()
            .find_map(|operation| match operation {
                Operation::Call(
                    Call::Method(method)
                    | Call::Returning(method)
                    | Call::Handle(method)
                    | Call::Values(method)
                    | Call::Texts(method),
                ) => Some(method.name().to_owned()),
                Operation::Call(Call::Capture(capture)) => Some(capture.method.name().to_owned()),
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
    /// (host memory without one).
    pub(crate) fn on<'r>(&'r self, stream: Option<&'r Stream>) -> Recorder<'r> {
        Recorder { job: self, stream }
    }

    /// An eager call of a runner method.
    pub(crate) fn call(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Method(method)));
    }

    /// An eager call of a runner method that returns an integer, a list of
    /// integers (the counts of what it wrote, say) or a list of such lists
    /// (the shapes of what it wrote), which the job's completion carries
    /// ([`Completion::returned`]).
    pub(crate) fn call_returning(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Returning(method)));
    }

    /// An eager call of a runner method that returns an object the library
    /// holds (a prepared image), which the job's completion carries
    /// ([`Completion::handle`]).
    pub(crate) fn call_handle(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Handle(method)));
    }

    /// An eager call of a runner method that returns a list of objects
    /// the library holds and lists of numbers (a plan and its schedule),
    /// which the job's completion carries ([`Completion::values`]).
    pub(crate) fn call_values(&self, method: MethodCall) {
        self.push(Operation::Call(Call::Values(method)));
    }

    /// An eager call of a function that returns a string or a list of
    /// strings (a report the rank passes on), which the job's completion
    /// carries ([`Completion::texts`]).
    pub(crate) fn call_texts(&self, function: MethodCall) {
        self.push(Operation::Call(Call::Texts(function)));
    }

    /// The capture of a runner method. Its completion carries the graph.
    pub(crate) fn capture(&self, capture: Capture) {
        self.push(Operation::Call(Call::Capture(capture)));
    }

    pub(crate) fn expert_step(&self, step: ExpertStep) {
        self.push(Operation::ExpertStep(step));
    }

    /// End the union's startup warmup on this rank
    /// ([`ExpertUnion::finish_warmup`]).
    pub(crate) fn expert_warmup(&self, union: Arc<ExpertUnion>) {
        self.push(Operation::ExpertWarmup(union));
    }

    /// The union's next step for a rank with no forward of its own. The
    /// job's completion says whether a step ran and whether every rank has
    /// left the union.
    pub(crate) fn expert_join(&self, union: Arc<ExpertUnion>, leaving: bool) {
        self.push(Operation::ExpertJoin { union, leaving });
    }

    /// Post the job's completion only once `events`, recorded on streams of
    /// `device`, have completed: the completion then says that the device
    /// finished what the events fence. The lane does not wait for them; it
    /// goes on with its next job. A job performed outside a lane completes
    /// when its operations are performed.
    pub(crate) fn after(&self, device: DeviceId, events: Vec<Arc<Event>>) {
        *self.after.borrow_mut() = Some((device, events));
    }

    /// Continue this job with the operations and the deferred work of
    /// `other`.
    pub(crate) fn append(&self, other: Job) {
        self.operations
            .borrow_mut()
            .append(&mut other.operations.into_inner());
        self.deferrals
            .borrow_mut()
            .append(&mut other.deferrals.into_inner());
    }

    /// Perform every operation in order on the calling thread, outside a
    /// lane: a microbatch thread in its turn of an expert step, or the
    /// submitter itself on a rank without a lane.
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
                            cuda::memcpy_async(value.as_mut_ptr() as u64, source, bytes, stream)
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
            Self::HostFunction { stream, function } => stream.as_ref().launch_host_fn(function),
            Self::Synchronize { stream } => Issue::synchronize(&stream.as_ref()),
            Self::ExpertStep(step) => step.perform(completion),
            Self::ExpertJoin { union, leaving } => {
                completion.joins.push_back(union.idle(leaving)?);
                Ok(())
            }
            Self::ExpertWarmup(union) => union.finish_warmup(),
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
                completion
                    .returned
                    .push_back(integers(value).map_err(|error| Error::numerical(&name, &error))?);
            }
            Self::Handle(method) => {
                let object = Handle::new(method.invoke()?)?;
                completion.handles.push_back(object);
            }
            Self::Values(method) => {
                let name = method.name().to_owned();
                let list: tvm_ffi::Array<tvm_ffi::Any> = tvm_ffi::Array::try_from(method.invoke()?)
                    .map_err(|error| Error::numerical(&name, &error))?;
                let values = list
                    .iter()
                    .map(|value| Value::new(value, &name))
                    .collect::<Result<Vec<_>>>()?;
                completion.values.push_back(values);
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
            Self::Capture(capture) => {
                let Capture {
                    functions,
                    pool,
                    peers,
                    stream,
                    device,
                    method,
                    breaks,
                } = capture;
                let captured = functions.capture_with_peers(&pool, &peers, stream, device, || {
                    method.invoke().map(drop)
                });

                // The breaks the method reached belong to this capture,
                // whether or not it succeeded.
                let breaks = breaks.as_ref().map_or_else(Vec::new, GraphBreaks::take);
                completion.captured.push_back(captured?.with_breaks(breaks));
            }
        }
        Ok(())
    }
}

/// The integers a returning call returned: an integer alone, a list of
/// integers as it is, and a list of lists as each list's length followed by
/// its integers.
pub(super) fn integers(value: tvm_ffi::Any) -> std::result::Result<Vec<i64>, tvm_ffi::Error> {
    if let Ok(integer) = i64::try_from(value.clone()) {
        return Ok(vec![integer]);
    }
    if let Ok(flat) = crate::ffi::integers(value.clone()) {
        return Ok(flat);
    }
    let lists: tvm_ffi::Array<tvm_ffi::Any> = tvm_ffi::Array::try_from(value)?;
    let mut flat = Vec::new();
    for list in lists.iter() {
        let values = crate::ffi::integers(list)?;
        flat.push(values.len() as i64);
        flat.extend(values);
    }
    Ok(flat)
}

impl ExpertStep {
    /// Agree on the step's capacity over the union's host group, run the
    /// step's forwards and joins at it, then what follows it on the lane's
    /// execution thread.
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

        let (then, _) = variant.then.perform_operations();
        completion.captured.extend(then.captured);
        completion.reads.extend(then.reads);
        then.failure.map_or(Ok(()), |failure| Err(failure.error))
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

        // Lists of integers (shapes) come back as each list's length and
        // its integers.
        let shapes = Function::new(
            "test.shapes",
            tvm_ffi::Function::from_packed(|_| {
                Ok(tvm_ffi::Any::from(crate::ffi::list([
                    tvm_ffi::Any::from(crate::ffi::integer_list([4, 8])),
                    tvm_ffi::Any::from(crate::ffi::integer_list([2, 16, 3])),
                ])))
            }),
        );
        job.call_returning(MethodCall::new(&shapes, Vec::new()));
        // An object the library holds comes back as its handle.
        job.call_handle(MethodCall::new(&counts, Vec::new()));

        // Only returning calls are kept, in recording order.
        let mut completion = job.perform().check().unwrap();
        assert_eq!(completion.returned()?, vec![48]);
        assert_eq!(completion.returned()?, vec![12, 36]);
        assert_eq!(completion.returned()?, vec![2, 4, 8, 3, 2, 16, 3]);
        assert!(completion.returned().is_err());
        assert!(completion.handle().is_ok());
        assert!(completion.handle().is_err());
        Ok(())
    }

    #[test]
    fn a_value_call_hands_its_list_to_the_completion_value_by_value() -> Result<()> {
        let plan = Function::new(
            "test.plan",
            tvm_ffi::Function::from_packed(|_| {
                Ok(tvm_ffi::Any::from(crate::ffi::list([
                    tvm_ffi::Any::from(crate::ffi::list([
                        tvm_ffi::Any::from(1.0_f64),
                        tvm_ffi::Any::from(0.5_f64),
                    ])),
                    tvm_ffi::Any::from(crate::ffi::integer_list([3, 1])),
                ])))
            }),
        );
        let job = Job::new();
        job.call_values(MethodCall::new(&plan, Vec::new()));

        let mut completion = job.perform().check().unwrap();
        let values = completion.values()?;
        assert!(matches!(
            &values[..],
            [Value::Floats(times), Value::Integers(masks)]
                if times == &[1.0, 0.5] && masks == &[3, 1]
        ));
        assert!(completion.values().is_err());
        Ok(())
    }

    #[test]
    fn a_value_call_returning_an_object_fails_its_job() {
        let plan = Function::new(
            "test.plan",
            tvm_ffi::Function::from_packed(|_| {
                Ok(tvm_ffi::Any::from(crate::ffi::list([tvm_ffi::Any::from(
                    tvm_ffi::String::from("schedule"),
                )])))
            }),
        );
        let job = Job::new();
        job.call_values(MethodCall::new(&plan, Vec::new()));
        assert!(job.perform().check().is_err());
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
