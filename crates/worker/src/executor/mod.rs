//! Bounded batch execution shared by direct callers and the rank service.
//!
//! The [`Executor`] is the rank's control loop. It admits batches
//! ([`Executor::admit`]), orders their forwards and submits their jobs
//! ([`Executor::advance`]), applies the completions of jobs
//! ([`Executor::complete`]) and delivers one result per batch
//! ([`Executor::poll`]).
//!
//! A batch's forward is a sequence of jobs on its lane. The executor
//! submits one and returns; the job's completion, which arrives through the
//! caller's wake, is applied and the next job submitted from there. The
//! executor waits for no job: a lane that stopped making progress holds up
//! the forwards behind its job, while the executor goes on admitting
//! batches, revoking freed products, delivering the results of forwards
//! that completed and returning to its caller. Two rules order forwards so
//! that what a job needs has happened when the job is prepared:
//!
//! - a batch that follows another on a request, or reads what another
//!   produces, starts once that batch's forward has completed: its host
//!   state (visible products, pending calls) is committed by the completion
//!   of its last job;
//! - a batch starts only when no other batch's forward is in flight. A job
//!   that waits on an event recorded by another lane's job orders behind
//!   the record only if the record is already made, and what a batch
//!   records is prepared against host state that the jobs of a forward
//!   still in flight may read. Forwards therefore follow one another, as
//!   they did when the executor waited for each.
//!
//! A forward is complete when its lane has performed its jobs, which is
//! when the stream holds its work, not when the device finished it: the
//! next forward is queued on the stream behind work still running, and a
//! batch's outputs are read later, from the completion of a job of their
//! own. A batch a rule keeps from starting is started when the forward it
//! waits for completed.

pub(crate) mod backend;
pub(crate) mod batch;
pub(crate) mod image;
pub(crate) mod inputs;
pub(crate) mod lifecycle;
mod pending;
pub(crate) mod products;
pub(crate) mod transfer;
pub(crate) mod video;

pub(crate) use pending::{BatchResult, PendingOutput, request_output};

use std::collections::HashSet;
use std::sync::{Arc, Weak};

use crossbeam_channel::{Receiver, Sender, unbounded};
use indexmap::IndexMap;
use uniserve_ipc::Wake;

use self::backend::{WorkerBackend, WorkerBatch};
use crate::Error;
use crate::error::WorkerError;
use crate::lane::{Completion, JobId};
use crate::profiling::range;

/// A submitted batch and the resources used by its numerical backend.
pub(crate) struct Batch {
    id: u64,
    collective_seq: Option<u64>,
    requests: HashSet<u64>,
    producers: HashSet<u64>,
    data: WorkerBatch,
    submission: Option<Arc<Submission>>,
    /// The lane its forward runs on.
    lane: Option<usize>,
    /// It is to be started: it was admitted, or what its start waited for
    /// arrived, and the ordering rules have not let it start since.
    startable: bool,
    prepared: bool,
    /// Its forward has jobs to run or in flight.
    forwarding: bool,
    /// The job of its forward that is in flight.
    forward: Option<JobId>,
    /// Its forward completed: every job ran and its host state is
    /// committed.
    launched: bool,
    /// The job that admits its inputs on its lane, while it is in flight.
    inputs: Option<JobId>,
    /// The job whose completion says its outputs can be read.
    outputs: Option<JobId>,
    complete: bool,
    error: Option<WorkerError>,
}

impl Batch {
    /// Bind backend data to the scheduler's request and producer dependencies.
    pub(crate) fn from_plan(plan: &uniserve_ipc::Batch, data: WorkerBatch) -> Self {
        let requests = plan
            .calls
            .iter()
            .map(|call| call.request_key.request_id.0)
            .chain(
                plan.commands
                    .iter()
                    .map(|command| command.request_key().request_id.0),
            )
            .collect();
        let mut producers = HashSet::new();
        for call in &plan.calls {
            producers.extend(
                call.tensor_inputs()
                    .chain(call.predicate.iter())
                    .map(|input| input.producer_call_id.batch_id),
            );
            if let Some(input) = &call.cache_input {
                producers.insert(input.producer_call_id.batch_id);
            }
        }
        Self::new(
            plan.batch_id,
            (!plan.calls.is_empty()).then_some(plan.collective_seq),
            requests,
            producers,
            data,
        )
    }

    /// `requests` includes calls and lifecycle commands. `producers` names
    /// batches supplying tensor, predicate, or cache inputs on this rank.
    /// A lifecycle-only batch has no collective sequence.
    pub(crate) fn new(
        id: u64,
        collective_seq: Option<u64>,
        requests: HashSet<u64>,
        producers: HashSet<u64>,
        data: WorkerBatch,
    ) -> Self {
        Self {
            id,
            collective_seq,
            requests,
            producers,
            data,
            submission: None,
            lane: None,
            startable: false,
            prepared: false,
            forwarding: false,
            forward: None,
            launched: false,
            inputs: None,
            outputs: None,
            complete: false,
            error: None,
        }
    }

    #[expect(clippy::expect_used, reason = "only submitted batches are advanced")]
    fn advance_execution(
        &mut self,
        backend: &mut WorkerBackend,
        last_sequence: &mut Option<u64>,
        distributed: bool,
    ) {
        let result = (|| {
            if !self.prepared {
                let _range = range(c"uniserve.worker.prepare", Some(self.id));
                backend.prepare(&mut self.data)?;
                self.prepared = true;
            }

            let submission = self.submission.as_ref().expect("submitted batch");
            {
                let _range = range(c"uniserve.worker.inputs", Some(self.id));
                // What admitting the inputs issues on the lane is a job:
                // the batch goes on from its completion.
                if self.inputs.is_some() {
                    return Ok(());
                }
                self.inputs = backend.admit_inputs(&mut self.data)?;
                if self.inputs.is_some() {
                    return Ok(());
                }
                if !backend.prepare_inputs(&mut self.data, submission)? {
                    return backend.await_inputs(&mut self.data, submission);
                }
            }

            if distributed && let Some(sequence) = self.collective_seq {
                if let Some(previous) = *last_sequence
                    && sequence <= previous
                {
                    // Other ranks may already be inside this collective. A
                    // local refusal makes the rank unsafe for further work.
                    return Err(WorkerError::classify(
                        Error::Invariant(format!(
                            "collective sequence does not advance: batch {} carries {} after {}",
                            self.id, sequence, previous
                        )),
                        None,
                    ));
                }
                *last_sequence = Some(sequence);
            }

            self.forwarding = true;
            self.advance_forward(backend)
        })();

        if let Err(error) = result {
            self.fail(backend, error);
        }
    }

    /// Submit the next job of the batch's forward and return; once the
    /// forward has none left, begin the batch's retirement and ask for the
    /// notice that its outputs can be read.
    fn advance_forward(&mut self, backend: &mut WorkerBackend) -> Result<(), WorkerError> {
        {
            let _range = range(c"uniserve.worker.execute", Some(self.id));
            self.forward = backend.forward(&mut self.data)?;
            if self.forward.is_some() {
                return Ok(());
            }
            self.forwarding = false;
        }

        self.launched = true;
        {
            let _range = range(c"uniserve.worker.retire", Some(self.id));
            backend.begin_retirement(&mut self.data)?;
        }
        self.outputs = backend.await_outputs(&mut self.data)?;
        Ok(())
    }

    /// Apply the completion of the forward's job in flight and go on with
    /// the forward.
    fn complete_forward(
        &mut self,
        backend: &mut WorkerBackend,
        job: JobId,
        completion: Completion,
    ) {
        self.forward = None;
        if self.complete {
            return;
        }
        let advanced = backend
            .complete(&mut self.data, job, completion)
            .and_then(|()| self.advance_forward(backend));
        if let Err(error) = advanced {
            self.fail(backend, error);
        }
    }

    /// Apply the completion of the job that says the batch's outputs can
    /// be read.
    fn complete_outputs(&mut self, backend: &mut WorkerBackend, completion: Completion) {
        self.outputs = None;
        if self.complete {
            return;
        }
        if let Err(error) = backend.outputs_ready(&mut self.data, completion) {
            self.fail(backend, error);
        }
    }

    fn advance_completion(&mut self, backend: &mut WorkerBackend) -> bool {
        // While the device has not finished what the batch launched, there
        // is nothing to materialize or retire.
        if self.complete || !self.launched || self.outputs.is_some() {
            return false;
        }

        let result = {
            let _range = range(c"uniserve.worker.poll", Some(self.id));
            backend.poll(&mut self.data)
        };

        match result {
            Ok((advanced, complete)) => {
                self.complete = complete;
                advanced || complete
            }
            Err(error) => {
                self.fail(backend, error);
                true
            }
        }
    }

    fn fail(&mut self, backend: &mut WorkerBackend, error: WorkerError) {
        let _range = range(c"uniserve.worker.close", Some(self.id));
        let mut error = self.data.attribute(error);
        if let Err(cleanup) = backend.close_batch(&mut self.data) {
            error.note(&cleanup);
        }

        self.error = Some(error);
        self.forwarding = false;
        self.complete = true;
    }
}

/// A result handle with a thread-safe input-readiness notification.
///
/// Weak notifications cannot keep abandoned numerical resources alive. The
/// executor ignores notices from consumed batches, including warmup handles
/// whose batch numbers have since been reused by serving.
pub(crate) struct Submission {
    batch_id: u64,
    ready: Sender<Weak<Submission>>,
    completion_wake: Option<Wake>,
}

impl Submission {
    pub(crate) fn notify_ready(self: &Arc<Self>) {
        let _ = self.ready.send(Arc::downgrade(self));
        if let Some(wake) = &self.completion_wake {
            wake.wake();
        }
    }
}

/// Own admission, dependency order, launch order, and one result per batch.
pub(crate) struct Executor {
    backend: WorkerBackend,
    capacity: usize,
    distributed: bool,
    collective: bool,
    batches: IndexMap<u64, Batch>,
    last_batch: Option<u64>,
    last_sequence: Option<u64>,
    ready_sender: Sender<Weak<Submission>>,
    ready_receiver: Receiver<Weak<Submission>>,
    completion_wake: Option<Wake>,
    closed: bool,
}

impl Executor {
    /// Collective components serialize until prior results are consumed.
    /// Other distributed workers preserve launch order while overlapping
    /// completion; single-rank workers may prepare independent requests.
    pub(crate) fn new(
        backend: WorkerBackend,
        capacity: usize,
        distributed: bool,
        collective: bool,
    ) -> Result<Self, WorkerError> {
        if capacity == 0 {
            return Err(WorkerError::classify(
                Error::Invalid("worker admission capacity must be positive".into()),
                None,
            ));
        }

        let (ready_sender, ready_receiver) = unbounded();
        Ok(Self {
            backend,
            capacity,
            distributed,
            collective,
            batches: IndexMap::new(),
            last_batch: None,
            last_sequence: None,
            ready_sender,
            ready_receiver,
            completion_wake: None,
            closed: false,
        })
    }

    pub(crate) fn backend(&self) -> &WorkerBackend {
        &self.backend
    }

    pub(crate) fn backend_mut(&mut self) -> &mut WorkerBackend {
        &mut self.backend
    }

    /// The rank service supplies its channel wake for newly admitted batches.
    /// Direct callers drive progress through advance and poll themselves.
    pub(crate) fn set_completion_wake(&mut self, wake: Option<Wake>) {
        self.completion_wake = wake;
    }

    pub(crate) fn has_work(&self) -> bool {
        !self.batches.is_empty()
    }

    pub(crate) fn started(&self) -> bool {
        self.last_batch.is_some()
    }

    /// Accept increasing batch IDs, retaining capacity until result delivery.
    /// A full queue does not consume the ID. Other accepted IDs cannot be
    /// reused, including failed batches. Immediate failures are consumed and
    /// raised here when `propagate_errors` is set.
    pub(crate) fn admit(
        &mut self,
        mut batch: Batch,
        propagate_errors: bool,
    ) -> Result<Arc<Submission>, WorkerError> {
        self.require_open()?;
        if let Some(previous) = self.last_batch
            && batch.id <= previous
        {
            return Err(WorkerError::classify(
                Error::Invalid(format!(
                    "batch id {} must exceed previously submitted id {}",
                    batch.id, previous
                )),
                None,
            ));
        }
        if self.batches.len() == self.capacity {
            return Err(WorkerError::classify(
                Error::Resource("worker admission queue is full"),
                None,
            ));
        }

        self.last_batch = Some(batch.id);
        let submission = Arc::new(Submission {
            batch_id: batch.id,
            ready: self.ready_sender.clone(),
            completion_wake: self.completion_wake.clone(),
        });
        {
            let _range = range(c"uniserve.worker.admit", Some(batch.id));
            self.backend.admit(&mut batch.data)?;
        }

        batch.submission = Some(Arc::clone(&submission));
        batch.lane = self.backend.lane(&batch.data);
        batch.startable = true;
        self.batches.insert(batch.id, batch);
        self.start(submission.batch_id);
        let batch = &mut self.batches[&submission.batch_id];
        batch.advance_completion(&mut self.backend);

        if propagate_errors && batch.error.is_some() {
            return match self.poll(&submission) {
                Err(error) => Err(error),
                Ok(_) => unreachable!("failed submission has one terminal error"),
            };
        }

        Ok(submission)
    }

    /// Progress ready inputs and physical completions on the owning thread.
    pub(crate) fn advance(&mut self) -> Result<bool, WorkerError> {
        self.require_open()?;
        {
            let _range = range(c"uniserve.worker.reap", None);
            self.backend.reap()?;
        }

        let mut advanced = false;

        // The completions of jobs nobody waited for.
        while let Some((job, completion)) = self.backend.completed()? {
            self.complete(job, completion);
            advanced = true;
        }

        // Resume one notified preparation, then inspect every running batch.
        // An independent host task or GPU completion must not wait behind the
        // oldest request. Repeated or late notices never enqueue execution twice.
        while let Ok(notice) = self.ready_receiver.try_recv() {
            if let Some(submission) = notice.upgrade()
                && self.owns(&submission)
                && let Some(batch) = self.batches.get_mut(&submission.batch_id)
                && !batch.launched
                && !batch.complete
            {
                batch.startable = true;
                self.start(submission.batch_id);
                advanced = true;
                break;
            }
        }

        for batch in self.batches.values_mut() {
            advanced |= batch.advance_completion(&mut self.backend);
        }

        // What kept a batch from starting may be gone: the forward it
        // follows, or a forward on another lane, completed.
        let startable: Vec<_> = self
            .batches
            .values()
            .filter(|batch| batch.startable && !batch.complete)
            .map(|batch| batch.id)
            .collect();
        for id in startable {
            advanced |= self.start(id);
        }

        Ok(advanced)
    }

    /// Apply the completion of `job` to the batch it belongs to and go on
    /// with that batch: the next job of its forward, its inputs behind
    /// their admission, or its outputs. The completion of a job whose batch
    /// is gone (closed, or failed and consumed) has nothing left to apply
    /// to.
    pub(crate) fn complete(&mut self, job: JobId, completion: Completion) {
        let Some(batch) = self.batches.values_mut().find(|batch| {
            batch.forward == Some(job) || batch.outputs == Some(job) || batch.inputs == Some(job)
        }) else {
            return;
        };
        if batch.forward == Some(job) {
            batch.complete_forward(&mut self.backend, job, completion);
            return;
        }
        if batch.outputs == Some(job) {
            batch.complete_outputs(&mut self.backend, completion);
            return;
        }

        // Its inputs are admitted on the lane: the batch starts again from
        // where it submitted the job.
        let id = batch.id;
        batch.inputs = None;
        if batch.complete {
            return;
        }
        match self.backend.inputs_admitted(&mut batch.data, completion) {
            Ok(()) => {
                batch.startable = true;
                self.start(id);
            }
            Err(error) => batch.fail(&mut self.backend, error),
        }
    }

    /// Consume one completed result without submitting numerical work.
    #[expect(clippy::expect_used, reason = "ownership was checked before removal")]
    pub(crate) fn poll(
        &mut self,
        submission: &Arc<Submission>,
    ) -> Result<Option<BatchResult>, WorkerError> {
        self.require_open()?;
        if !self.owns(submission) {
            return Err(WorkerError::classify(
                Error::Invalid("poll names a batch no longer owned by this Worker".into()),
                None,
            ));
        }
        if !self.batches[&submission.batch_id].complete {
            return Ok(None);
        }

        let mut batch = self
            .batches
            .shift_remove(&submission.batch_id)
            .expect("owned submission");
        let result = match batch.error.take() {
            Some(error) => Err(error),
            None => {
                let _range = range(c"uniserve.worker.result", Some(batch.id));
                self.backend.result(&mut batch.data).map(Some)
            }
        };

        let _range = range(c"uniserve.worker.close", Some(batch.id));
        match (result, self.backend.close_batch(&mut batch.data)) {
            (Err(mut error), Err(cleanup)) => {
                error.note(&cleanup);
                Err(error)
            }
            (Ok(_), Err(error)) => Err(error),
            (result, Ok(())) => result,
        }
    }

    /// Reset startup numbering only after every warmup result is consumed.
    pub(crate) fn reset(&mut self) -> Result<(), WorkerError> {
        self.require_open()?;
        if self.has_work() {
            return Err(WorkerError::classify(
                Error::State("warmup completed with pending batches"),
                None,
            ));
        }

        self.last_batch = None;
        self.last_sequence = None;
        self.ready_receiver.try_iter().for_each(drop);
        Ok(())
    }

    /// Attempt every batch cleanup before returning the first failure.
    pub(crate) fn close(&mut self) -> Result<(), WorkerError> {
        self.closed = true;
        let mut failure: Option<WorkerError> = None;
        for (_, mut batch) in self.batches.drain(..) {
            let _range = range(c"uniserve.worker.close", Some(batch.id));
            if let Err(error) = self.backend.close_batch(&mut batch.data) {
                match &mut failure {
                    Some(first) => first.note(&error),
                    None => failure = Some(error),
                }
            }
        }

        self.ready_receiver.try_iter().for_each(drop);
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn start(&mut self, id: u64) -> bool {
        if !self.can_start(id) {
            return false;
        }

        let batch = &mut self.batches[&id];
        batch.startable = false;
        batch.advance_execution(&mut self.backend, &mut self.last_sequence, self.distributed);
        true
    }

    fn can_start(&self, id: u64) -> bool {
        let batch = &self.batches[&id];
        // A forward in flight goes on from its job's completion.
        if batch.launched || batch.complete || batch.forwarding {
            return false;
        }
        // Forwards follow one another: see the module documentation.
        if self
            .batches
            .values()
            .any(|other| other.forwarding && other.id != id)
        {
            return false;
        }
        if batch.prepared {
            return true;
        }

        for (&previous_id, previous) in &self.batches {
            if previous_id >= id {
                break;
            }
            if self.collective
                || (!previous.launched
                    && !previous.complete
                    && (self.distributed || !batch.requests.is_disjoint(&previous.requests)))
            {
                return false;
            }
        }

        !batch.producers.iter().any(|producer| {
            self.batches
                .get(producer)
                .is_some_and(|batch| !batch.launched)
        })
    }

    fn owns(&self, submission: &Arc<Submission>) -> bool {
        self.batches
            .get(&submission.batch_id)
            .and_then(|batch| batch.submission.as_ref())
            .is_some_and(|owned| Arc::ptr_eq(owned, submission))
    }

    fn require_open(&self) -> Result<(), WorkerError> {
        if self.closed {
            Err(WorkerError::classify(
                Error::State("worker executor is closed"),
                None,
            ))
        } else {
            Ok(())
        }
    }
}
