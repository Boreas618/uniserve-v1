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

pub mod backend;
pub(crate) mod forward;
mod inputs;
pub mod lifecycle;
mod pending;

pub use inputs::{BatchInputs, InputReady, InputWait};
pub use pending::{BatchResult, PendingOutput, request_output};

use std::collections::HashSet;
use std::sync::{Arc, Weak};

use crossbeam_channel::{Receiver, Sender, unbounded};
use indexmap::IndexMap;
use uniserve_ipc::Wake;

use crate::Error;
use crate::profiling::range;

/// A job's name and what was posted for it when it was performed.
pub type Completed<B> = (<B as Backend>::Job, <B as Backend>::Completion);

/// Execution resources and numerical operations borrowed by one executor.
///
/// Methods run on the executor thread. Readiness callbacks may only notify a
/// submission; they never invoke the backend or advance another batch.
pub trait Backend {
    type Batch;
    type Output;
    type Error;
    /// Names a job from its submission to its completion.
    type Job: Copy + Eq;
    /// What a lane posts when it has performed a job.
    type Completion;

    fn error(&self, error: Error) -> Self::Error;
    fn classify(&self, error: Self::Error, batch: &Self::Batch, context: &str) -> Self::Error;
    fn note_cleanup(&self, error: &mut Self::Error, cleanup: Self::Error);

    /// Revoke explicit Free commands before this batch can wait for storage.
    fn admit(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error>;

    /// Bind request storage and reserve numerical inputs once per batch.
    fn prepare(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error>;

    /// Submit the job that issues, on the batch's lane, what admitting its
    /// inputs needs there (the imports of what other calls published).
    /// `None` when nothing is to be issued now. Called each time the batch
    /// is started, before [`Self::prepare_inputs`]; the batch goes on once
    /// the job's completion was applied ([`Self::inputs_admitted`]).
    fn admit_inputs(&mut self, batch: &mut Self::Batch) -> Result<Option<Self::Job>, Self::Error>;

    /// Apply the completion of the job [`Self::admit_inputs`] submitted.
    fn inputs_admitted(
        &mut self,
        batch: &mut Self::Batch,
        completion: Self::Completion,
    ) -> Result<(), Self::Error>;

    /// Submit ready reads and report whether all numerical inputs are ready.
    fn prepare_inputs(
        &mut self,
        batch: &mut Self::Batch,
        submission: &Arc<Submission>,
    ) -> Result<bool, Self::Error>;

    /// Arrange a readiness notice when pending inputs can advance.
    fn await_inputs(
        &mut self,
        batch: &mut Self::Batch,
        submission: &Arc<Submission>,
    ) -> Result<(), Self::Error>;

    /// The lane the batch's forward runs on; `None` for a batch without a
    /// forward or without a lane.
    fn lane(&self, batch: &Self::Batch) -> Option<usize>;

    /// Prepare the next job of the batch's forward and submit it without
    /// waiting for it. `None` once the forward has nothing left to submit.
    /// The job's completion arrives through [`Self::completed`] and is
    /// applied by [`Self::complete`] before this is called again.
    fn forward(&mut self, batch: &mut Self::Batch) -> Result<Option<Self::Job>, Self::Error>;

    /// Apply the completion of the job [`Self::forward`] submitted last.
    fn complete(
        &mut self,
        batch: &mut Self::Batch,
        job: Self::Job,
        completion: Self::Completion,
    ) -> Result<(), Self::Error>;

    /// Submit the job whose completion says that the device finished the
    /// batch's launched work, so its outputs can be read. `None` when they
    /// can be read already.
    fn await_outputs(&mut self, batch: &mut Self::Batch) -> Result<Option<Self::Job>, Self::Error>;

    /// Apply the completion of the job [`Self::await_outputs`] submitted.
    /// The batch is not polled before.
    fn outputs_ready(
        &mut self,
        batch: &mut Self::Batch,
        completion: Self::Completion,
    ) -> Result<(), Self::Error>;

    /// The next completion that arrived, without waiting for one.
    fn completed(&mut self) -> Result<Option<Completed<Self>>, Self::Error>;

    fn begin_retirement(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error>;

    /// Materialize outputs and retire physical accesses. Returns (progress,
    /// complete), where complete allows the executor to deliver this result.
    fn poll(&mut self, batch: &mut Self::Batch) -> Result<(bool, bool), Self::Error>;
    fn result(&mut self, batch: &mut Self::Batch) -> Result<Self::Output, Self::Error>;

    /// Release or abandon resources; may follow cleanup after a failed call.
    fn close(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error>;
    fn reap(&mut self) -> Result<(), Self::Error>;
}

/// A submitted batch and the resources used by its numerical backend.
pub struct Batch<B: Backend> {
    id: u64,
    collective_seq: Option<u64>,
    requests: HashSet<u64>,
    producers: HashSet<u64>,
    data: B::Batch,
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
    forward: Option<B::Job>,
    /// Its forward completed: every job ran and its host state is
    /// committed.
    launched: bool,
    /// The job that admits its inputs on its lane, while it is in flight.
    inputs: Option<B::Job>,
    /// The job whose completion says its outputs can be read.
    outputs: Option<B::Job>,
    complete: bool,
    error: Option<B::Error>,
}

impl<B: Backend> Batch<B> {
    /// Bind backend data to the scheduler's request and producer dependencies.
    pub fn from_plan(plan: &uniserve_ipc::Batch, data: B::Batch) -> Self {
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
            if let Some(input) = &call.kv_input {
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
    /// batches supplying tensor, predicate, or KV inputs on this rank.
    /// A lifecycle-only batch has no collective sequence.
    pub fn new(
        id: u64,
        collective_seq: Option<u64>,
        requests: HashSet<u64>,
        producers: HashSet<u64>,
        data: B::Batch,
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
        backend: &mut B,
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
                    return Err(backend.error(Error::Invariant(format!(
                        "collective sequence does not advance: batch {} carries {} after {}",
                        self.id, sequence, previous
                    ))));
                }
                *last_sequence = Some(sequence);
            }

            self.forwarding = true;
            self.advance_forward(backend)
        })();

        if let Err(error) = result {
            self.fail(backend, error, "execute");
        }
    }

    /// Submit the next job of the batch's forward and return; once the
    /// forward has none left, begin the batch's retirement and ask for the
    /// notice that its outputs can be read.
    fn advance_forward(&mut self, backend: &mut B) -> Result<(), B::Error> {
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
    fn complete_forward(&mut self, backend: &mut B, job: B::Job, completion: B::Completion) {
        self.forward = None;
        if self.complete {
            return;
        }
        let advanced = backend
            .complete(&mut self.data, job, completion)
            .and_then(|()| self.advance_forward(backend));
        if let Err(error) = advanced {
            self.fail(backend, error, "execute");
        }
    }

    /// Apply the completion of the job that says the batch's outputs can
    /// be read.
    fn complete_outputs(&mut self, backend: &mut B, completion: B::Completion) {
        self.outputs = None;
        if self.complete {
            return;
        }
        if let Err(error) = backend.outputs_ready(&mut self.data, completion) {
            self.fail(backend, error, "completion");
        }
    }

    fn advance_completion(&mut self, backend: &mut B) -> bool {
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
                self.fail(backend, error, "completion materialization");
                true
            }
        }
    }

    fn fail(&mut self, backend: &mut B, error: B::Error, context: &str) {
        let _range = range(c"uniserve.worker.close", Some(self.id));
        let mut error = backend.classify(error, &self.data, context);
        if let Err(cleanup) = backend.close(&mut self.data) {
            backend.note_cleanup(&mut error, cleanup);
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
pub struct Submission {
    batch_id: u64,
    ready: Sender<Weak<Submission>>,
    completion_wake: Option<Wake>,
}

impl Submission {
    pub fn batch_id(&self) -> u64 {
        self.batch_id
    }

    pub fn notify_ready(self: &Arc<Self>) {
        let _ = self.ready.send(Arc::downgrade(self));
        if let Some(wake) = &self.completion_wake {
            wake.wake();
        }
    }
}

/// Own admission, dependency order, launch order, and one result per batch.
pub struct Executor<B: Backend> {
    backend: B,
    capacity: usize,
    distributed: bool,
    collective: bool,
    batches: IndexMap<u64, Batch<B>>,
    last_batch: Option<u64>,
    last_sequence: Option<u64>,
    ready_sender: Sender<Weak<Submission>>,
    ready_receiver: Receiver<Weak<Submission>>,
    completion_wake: Option<Wake>,
    closed: bool,
}

impl<B: Backend> Executor<B> {
    /// Collective components serialize until prior results are consumed.
    /// Other distributed workers preserve launch order while overlapping
    /// completion; single-rank workers may prepare independent requests.
    pub fn new(
        backend: B,
        capacity: usize,
        distributed: bool,
        collective: bool,
    ) -> Result<Self, B::Error> {
        if capacity == 0 {
            return Err(backend.error(Error::Invalid(
                "worker admission capacity must be positive".into(),
            )));
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

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// The rank service supplies its channel wake for newly admitted batches.
    /// Direct callers drive progress through advance and poll themselves.
    pub(crate) fn set_completion_wake(&mut self, wake: Option<Wake>) {
        self.completion_wake = wake;
    }

    pub fn has_work(&self) -> bool {
        !self.batches.is_empty()
    }

    pub fn started(&self) -> bool {
        self.last_batch.is_some()
    }

    /// Accept increasing batch IDs, retaining capacity until result delivery.
    /// A full queue does not consume the ID. Other accepted IDs cannot be
    /// reused, including failed batches. Immediate failures are consumed and
    /// raised here when `propagate_errors` is set.
    pub fn admit(
        &mut self,
        mut batch: Batch<B>,
        propagate_errors: bool,
    ) -> Result<Arc<Submission>, B::Error> {
        self.require_open()?;
        if let Some(previous) = self.last_batch
            && batch.id <= previous
        {
            return Err(self.backend.error(Error::Invalid(format!(
                "batch id {} must exceed previously submitted id {}",
                batch.id, previous
            ))));
        }
        if self.batches.len() == self.capacity {
            return Err(self
                .backend
                .error(Error::Resource("worker admission queue is full")));
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
    pub fn advance(&mut self) -> Result<bool, B::Error> {
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
    pub fn complete(&mut self, job: B::Job, completion: B::Completion) {
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
            Err(error) => batch.fail(&mut self.backend, error, "input admission"),
        }
    }

    /// Consume one completed result without submitting numerical work.
    #[expect(clippy::expect_used, reason = "ownership was checked before removal")]
    pub fn poll(&mut self, submission: &Arc<Submission>) -> Result<Option<B::Output>, B::Error> {
        self.require_open()?;
        if !self.owns(submission) {
            return Err(self.backend.error(Error::Invalid(
                "poll names a batch no longer owned by this Worker".into(),
            )));
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
        match (result, self.backend.close(&mut batch.data)) {
            (Err(mut error), Err(cleanup)) => {
                self.backend.note_cleanup(&mut error, cleanup);
                Err(error)
            }
            (Ok(_), Err(error)) => Err(error),
            (result, Ok(())) => result,
        }
    }

    /// Reset startup numbering only after every warmup result is consumed.
    pub fn reset(&mut self) -> Result<(), B::Error> {
        self.require_open()?;
        if self.has_work() {
            return Err(self
                .backend
                .error(Error::State("warmup completed with pending batches")));
        }

        self.last_batch = None;
        self.last_sequence = None;
        self.ready_receiver.try_iter().for_each(drop);
        Ok(())
    }

    /// Attempt every batch cleanup before returning the first failure.
    pub fn close(&mut self) -> Result<(), B::Error> {
        self.closed = true;
        let mut failure = None;
        for (_, mut batch) in self.batches.drain(..) {
            let _range = range(c"uniserve.worker.close", Some(batch.id));
            if let Err(error) = self.backend.close(&mut batch.data) {
                match &mut failure {
                    Some(first) => self.backend.note_cleanup(first, error),
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

    fn require_open(&self) -> Result<(), B::Error> {
        if self.closed {
            Err(self
                .backend
                .error(Error::State("worker executor is closed")))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch of the test backend: its name in the log, the lane its
    /// forward runs on, the admission job it submits first and the jobs of
    /// its forward still to submit.
    struct Planned {
        name: &'static str,
        lane: Option<usize>,
        admission: Option<u32>,
        jobs: Vec<u32>,
    }

    /// A backend that submits the jobs its batches plan and logs what the
    /// executor asks of it.
    #[derive(Default)]
    struct Recording {
        log: Vec<String>,
    }

    impl Backend for Recording {
        type Batch = Planned;
        type Output = ();
        type Error = Error;
        type Job = u32;
        type Completion = ();

        fn error(&self, error: Error) -> Error {
            error
        }

        fn classify(&self, error: Error, _batch: &Planned, _context: &str) -> Error {
            error
        }

        fn note_cleanup(&self, _error: &mut Error, _cleanup: Error) {}

        fn admit(&mut self, _batch: &mut Planned) -> Result<(), Error> {
            Ok(())
        }

        fn prepare(&mut self, batch: &mut Planned) -> Result<(), Error> {
            self.log.push(format!("{} prepared", batch.name));
            Ok(())
        }

        fn admit_inputs(&mut self, batch: &mut Planned) -> Result<Option<u32>, Error> {
            Ok(batch.admission.take())
        }

        fn inputs_admitted(&mut self, batch: &mut Planned, _completion: ()) -> Result<(), Error> {
            self.log.push(format!("{} inputs admitted", batch.name));
            Ok(())
        }

        fn prepare_inputs(
            &mut self,
            batch: &mut Planned,
            _submission: &Arc<Submission>,
        ) -> Result<bool, Error> {
            self.log.push(format!("{} inputs ready", batch.name));
            Ok(true)
        }

        fn await_inputs(
            &mut self,
            _batch: &mut Planned,
            _submission: &Arc<Submission>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn lane(&self, batch: &Planned) -> Option<usize> {
            batch.lane
        }

        fn forward(&mut self, batch: &mut Planned) -> Result<Option<u32>, Error> {
            if batch.jobs.is_empty() {
                self.log.push(format!("{} forwarded", batch.name));
                return Ok(None);
            }
            let job = batch.jobs.remove(0);
            self.log.push(format!("{} submits {job}", batch.name));
            Ok(Some(job))
        }

        fn complete(
            &mut self,
            batch: &mut Planned,
            job: u32,
            _completion: (),
        ) -> Result<(), Error> {
            self.log.push(format!("{} completes {job}", batch.name));
            Ok(())
        }

        fn await_outputs(&mut self, _batch: &mut Planned) -> Result<Option<u32>, Error> {
            Ok(None)
        }

        fn outputs_ready(&mut self, _batch: &mut Planned, _completion: ()) -> Result<(), Error> {
            Ok(())
        }

        fn completed(&mut self) -> Result<Option<Completed<Self>>, Error> {
            Ok(None)
        }

        fn begin_retirement(&mut self, _batch: &mut Planned) -> Result<(), Error> {
            Ok(())
        }

        fn poll(&mut self, _batch: &mut Planned) -> Result<(bool, bool), Error> {
            Ok((true, true))
        }

        fn result(&mut self, _batch: &mut Planned) -> Result<(), Error> {
            Ok(())
        }

        fn close(&mut self, _batch: &mut Planned) -> Result<(), Error> {
            Ok(())
        }

        fn reap(&mut self) -> Result<(), Error> {
            Ok(())
        }
    }

    /// A batch of one request of its own, so no batch follows another.
    fn batch(id: u64, data: Planned) -> Batch<Recording> {
        Batch::new(id, None, HashSet::from([id]), HashSet::new(), data)
    }

    /// The log entries since the last call.
    fn logged(executor: &mut Executor<Recording>) -> Vec<String> {
        std::mem::take(&mut executor.backend_mut().log)
    }

    #[test]
    fn a_batch_goes_on_from_the_completion_of_its_input_admission() -> Result<(), Error> {
        let mut executor = Executor::new(Recording::default(), 4, false, false)?;
        let planned = Planned {
            name: "a",
            lane: None,
            admission: Some(7),
            jobs: Vec::new(),
        };
        let submission = executor.admit(batch(1, planned), true)?;

        // The admission job is in flight: the batch has not gone further,
        // and advancing does not start it again.
        executor.advance()?;
        assert_eq!(logged(&mut executor), ["a prepared"]);
        assert!(executor.poll(&submission)?.is_none());

        // A completion of no job of the batch changes nothing; the
        // admission's completion lets the batch run to its result.
        executor.complete(8, ());
        assert!(logged(&mut executor).is_empty());
        executor.complete(7, ());
        assert_eq!(
            logged(&mut executor),
            ["a inputs admitted", "a inputs ready", "a forwarded"]
        );
        executor.advance()?;
        assert!(executor.poll(&submission)?.is_some());
        Ok(())
    }

    #[test]
    fn a_forward_is_submitted_job_by_job_from_each_completion() -> Result<(), Error> {
        let mut executor = Executor::new(Recording::default(), 4, false, false)?;
        let planned = Planned {
            name: "a",
            lane: Some(0),
            admission: None,
            jobs: vec![10, 11],
        };
        let submission = executor.admit(batch(1, planned), true)?;

        // The executor returned with the first job in flight; without its
        // completion nothing more is submitted and the batch has no result.
        assert_eq!(
            logged(&mut executor),
            ["a prepared", "a inputs ready", "a submits 10"]
        );
        executor.advance()?;
        assert!(logged(&mut executor).is_empty());
        assert!(executor.poll(&submission)?.is_none());

        // Each completion is applied and the next job submitted from it.
        executor.complete(10, ());
        assert_eq!(logged(&mut executor), ["a completes 10", "a submits 11"]);
        executor.complete(11, ());
        assert_eq!(logged(&mut executor), ["a completes 11", "a forwarded"]);
        executor.advance()?;
        assert!(executor.poll(&submission)?.is_some());
        Ok(())
    }

    #[test]
    fn a_batch_starts_when_no_other_forward_is_in_flight() -> Result<(), Error> {
        let mut executor = Executor::new(Recording::default(), 4, false, false)?;
        let planned = |name, lane, jobs| Planned {
            name,
            lane: Some(lane),
            admission: None,
            jobs,
        };

        // A forward is in flight on lane 0.
        executor.admit(batch(1, planned("a", 0, vec![10, 11])), true)?;
        assert_eq!(
            logged(&mut executor),
            ["a prepared", "a inputs ready", "a submits 10"]
        );

        // Batches admitted meanwhile are not started, on another lane or
        // on the same one, however often the executor passes.
        let other = executor.admit(batch(2, planned("b", 1, vec![20])), true)?;
        let same = executor.admit(batch(3, planned("c", 0, vec![30])), true)?;
        executor.complete(10, ());
        executor.advance()?;
        assert_eq!(logged(&mut executor), ["a completes 10", "a submits 11"]);

        // The forward completed: the oldest waiting batch starts at the
        // executor's next pass, and the next one behind its forward.
        executor.complete(11, ());
        assert_eq!(logged(&mut executor), ["a completes 11", "a forwarded"]);
        executor.advance()?;
        assert_eq!(
            logged(&mut executor),
            ["b prepared", "b inputs ready", "b submits 20"]
        );
        executor.complete(20, ());
        executor.advance()?;
        assert_eq!(
            logged(&mut executor),
            [
                "b completes 20",
                "b forwarded",
                "c prepared",
                "c inputs ready",
                "c submits 30"
            ]
        );
        executor.complete(30, ());
        executor.advance()?;
        assert!(executor.poll(&other)?.is_some());
        assert!(executor.poll(&same)?.is_some());
        Ok(())
    }
}
