//! Preparing a generation: its kernels are selected and installed on the module-load lane, every
//! device rebinds or recaptures its buckets on it while every scheduler prepares the policy, and
//! the schedulers shadow one step of their running requests when an operator's package changed.
//!
//! The same work makes an engine's first generation ready at its start and the next one ready
//! beside the serving one in an update; only its end differs. Each stage's completion wakes the
//! control thread, which then runs the next ([`MainProcess::continue_prepare`]); a failure drops
//! what was built, and serving - when anything serves - goes on.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use ipc::UpdateFailure;
use ipc::execution::Outcome;
use ipc::execution::Submission;
use ops::KernelEntry;
use proto::module::{PackageRevision, Packages};
use worker::device::Worker;
use worker::host::HostTask;

use super::MainProcess;
use super::device_command;
use super::lifecycle::Lifecycle;
use super::policy::{TOKEN_LIMIT, limit_refusal};
use crate::modules::load::{ModuleOutput, ModuleWork};
use crate::modules::select::{Routes, changed_operators, replaced_operators};

/// A scheduler's response to one step of a preparation.
type Response = crossbeam_channel::Receiver<std::result::Result<(), String>>;

/// The operator and descriptor key a failed profile run named, when that is what failed.
///
/// A generation holds every kernel its buckets launch: the preparation installs every key the
/// devices' profile runs reported (`worker::runners::recording`) in one admission. A forward
/// that still finds one missing names it here, and the preparation refuses with it: a key the
/// profile run missed is the profile run's defect, never a key to learn and install on another
/// attempt.
fn missing_kernel(submissions: &[Submission]) -> Option<(String, String)> {
    /// The kernel an error names as missing, whether it reached the worker as an operator's
    /// error or wrapped in the model layer's: a capture reports it through
    /// `nn::Forward::operate`, which resolves the kernel and converts the operator's error into
    /// the layer's own.
    fn named(error: &worker::Error) -> Option<(&'static str, &str)> {
        let ops = match error {
            worker::Error::Ops(ops) => ops,
            worker::Error::Nn(nn::Error::Ops(ops)) => ops,
            _ => return None,
        };
        match ops {
            ops::Error::MissingKernel {
                operator,
                descriptor,
                ..
            } => Some((operator, descriptor.as_str())),
            _ => None,
        }
    }
    for submission in submissions {
        let Some(Outcome::Failed(error)) = submission.outcome() else {
            continue;
        };
        // A submission carries its failure across the crossing as a fault; the worker raised it,
        // so it is the worker's own error.
        let Some(error) = error.downcast_ref::<worker::Error>() else {
            continue;
        };
        if let Some((operator, descriptor)) = named(error) {
            return Some((operator.to_owned(), descriptor.to_owned()));
        }
    }
    None
}

/// Reads the scheduler responses that arrived: `Ok(true)` once every one responded, and the
/// first refusal otherwise.
fn responded(responses: &mut Vec<Response>) -> std::result::Result<bool, String> {
    // A response is taken once and its sender is gone after it, so a receiver that responded
    // leaves the list: reading it again would read a scheduler that finished as one that was lost.
    while let Some(receiver) = responses.first() {
        match receiver.try_recv() {
            Ok(Ok(())) => {
                responses.remove(0);
            }
            Ok(Err(reason)) => return Err(reason),
            Err(crossbeam_channel::TryRecvError::Empty) => return Ok(false),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                return Err("a scheduler of the group is gone".to_owned());
            }
        }
    }
    Ok(true)
}

/// The failure a shadow step that never ran is.
///
/// The step runs inside a batch, and an engine with no running request forms none: the stage
/// waits for something that cannot happen, so the update is abandoned with the operators named
/// rather than passing without the comparison it was there to make.
fn shadow_stall_message(changed: &[String], waited: u64) -> String {
    let count = changed.len();
    let operators = changed.join(", ");
    let noun = if count == 1 {
        "changed operator"
    } else {
        "changed operators"
    };
    format!(
        "the shadow step of {count} {noun} ({operators}) waited {waited}s: no running request ran \
         a step, so nothing was compared - the update is abandoned rather than passed without the \
         comparison it was prepared for"
    )
}

/// The preparation of one generation, continued as its stages complete.
pub(super) struct Prepare {
    pub(super) generation: u64,
    /// The operators whose package this preparation changed, in sorted order: the set the shadow
    /// pass pairs, and the set a stall names. Empty when nothing changed - a first generation,
    /// or an update that touched no kernel.
    changed: Vec<String>,
    /// The policy package the generation installs.
    policy: PackageRevision,
    /// The packages the deployment names, by slot: the backends the selection prefers.
    named: Packages,
    /// The kernel packages the generation installs, by slot: every slot of each operator whose
    /// selected packages differ from what serves, which for a first generation is every operator.
    packages: Packages,
    /// The keys the Build Server selected each installed slot's package for.
    routes: Routes,
    /// When the stage the preparation is in began, which its bound is measured from.
    since: Instant,
    /// The generation built for each device, in `workers` order.
    next: Vec<Arc<ops::Generation>>,
    /// The responses of every scheduler that is preparing the generation's policy package. The
    /// build runs on the schedulers' own blocking threads.
    policies: Vec<Response>,
    stage: Stage,
}

/// The stage a preparation is in, with what completes it.
enum Stage {
    /// The Build Server's selection of the generation's packages, alone on the module-load
    /// lane: the installs are derived from it.
    Selection(Arc<HostTask<ModuleWork>>),
    /// One install per device for the packages the selection resolved.
    Installs(Vec<Arc<HostTask<ModuleWork>>>),
    /// The devices rebind or recapture their buckets while the schedulers prepare the policy.
    Devices(Vec<Submission>),
    /// The shadow pass compares the new implementations on the running requests, one step per
    /// scheduler; each responds when its step ran.
    Shadow(Vec<Response>),
}

impl Prepare {
    /// The instant the stage in flight is failed for standing still, when it has such a bound:
    /// a device stage and a shadow stage are bounded by twice `stall`, and the module-load lane
    /// compiles for as long as its builds take.
    pub(super) fn bound(&self, stall: Duration) -> Option<Instant> {
        match self.stage {
            Stage::Selection(_) | Stage::Installs(_) => None,
            Stage::Devices(_) | Stage::Shadow(_) => Some(self.since + stall * 2),
        }
    }
}

/// What a device stage is waiting on: each device that has not finished with its own account,
/// and each scheduler's response to the policy it was asked to build.
///
/// The two waits live in one stage, and a report that names only the devices says "the device
/// thread is the one to look at" while the device has finished and a policy build is what never
/// came back. The policy figure is stated first for that reason: it is the wait the devices' own
/// account cannot show.
fn device_work(
    submissions: &[Submission],
    workers: &[Arc<Worker>],
    policies: &[Response],
) -> String {
    let mut pending = Vec::new();
    for (rank, submission) in submissions.iter().enumerate() {
        if submission.outcome().is_some() {
            continue;
        }
        let Some(worker) = workers.get(rank) else {
            pending.push(format!("rank {rank}: no worker"));
            continue;
        };
        let performing = worker.performing().unwrap_or_else(|| "nothing".to_owned());
        pending.push(format!(
            "rank {rank}: {performing}, {} open capture(s)",
            worker.capturing()
        ));
    }
    let finished = submissions.len() - pending.len();
    let devices = format!(
        "{finished} of {} device preparation(s) finished ({})",
        submissions.len(),
        pending.join("; ")
    );
    let policies: String = if policies.is_empty() {
        "no policy build was asked for".to_owned()
    } else {
        let responded = policies
            .iter()
            .filter(|response| !response.is_empty())
            .count();
        format!(
            "{responded} of {} policy build(s) responded",
            policies.len()
        )
    };
    format!("{devices}; {policies}")
}

/// Whether every submission finished: `Ok(false)` while one is pending, and why the first failed
/// or cancelled one did not otherwise.
fn finished(submissions: &[Submission], what: &str) -> std::result::Result<bool, String> {
    for submission in submissions {
        match submission.outcome() {
            None => return Ok(false),
            Some(Outcome::Success(_)) => {}
            Some(Outcome::Failed(error)) => return Err(format!("{what}: {error}")),
            Some(Outcome::Cancelled) => return Err(format!("{what}: cancelled")),
        }
    }
    Ok(true)
}

impl MainProcess {
    /// Begins preparing `generation`: admits the Build Server's selection of its packages on the
    /// module-load lane; the installs follow from its result, and their builds run while serving
    /// goes on.
    ///
    /// An engine's first generation and an update go the same way. Every key this node's devices
    /// launch is asked of the Build Server once, with the backends `packages` names, and the
    /// server selects the package of each key (`BuildServer::cover`). The first generation
    /// installs every selected slot in one admission; an update installs only the operators whose
    /// selection differs from what serves - a newly installed revision of an operator the
    /// deployment names nothing for is such a difference - and a kernel that stays keeps its
    /// `Arc`, so its buckets and its graphs stay too.
    pub(super) fn start_prepare(
        &mut self,
        generation: u64,
        packages: &Packages,
        policy: &PackageRevision,
    ) -> Result<Option<UpdateFailure>> {
        let keys = self.generation_keys();
        tracing::info!(
            generation,
            named = packages.len(),
            operators = keys.len(),
            keys = keys
                .values()
                .map(std::collections::BTreeSet::len)
                .sum::<usize>(),
            devices = self.workers.len(),
            policy = %policy.backend,
            policy_revision = policy.revision,
            "asking the Build Server to select this generation's packages"
        );
        let task = match self.admit_selection(packages, keys) {
            Ok(task) => task,
            Err(failure) => return self.prepare_failed(generation, failure),
        };
        self.wake_on_task(&task);
        self.preparing = Some(Prepare {
            generation,
            changed: Vec::new(),
            policy: policy.clone(),
            named: packages.clone(),
            packages: Packages::new(),
            routes: Routes::new(),
            since: Instant::now(),
            next: Vec::new(),
            policies: Vec::new(),
            stage: Stage::Selection(task),
        });
        Ok(None)
    }

    /// The operators this node's devices could not compare, each with its reason, in device
    /// order and without repeats.
    ///
    /// It is carried by the preparation's acknowledgement so an update that has no evidence for
    /// an operator says so in the record the daemon reads, not only in this process's log: empty
    /// is the statement that every operator the update replaced was compared.
    fn uncompared(&self) -> Vec<String> {
        let mut entries: Vec<String> = Vec::new();
        for worker in &self.workers {
            for entry in worker.uncompared() {
                if !entries.contains(&entry) {
                    entries.push(entry);
                }
            }
        }
        entries
    }

    /// Continues the preparation in flight when the stage it is in completed, and fails it by
    /// name when the stage stood past its bound. A stage whose completion has not arrived is
    /// left as it is.
    pub(super) fn continue_prepare(&mut self) -> Result<Option<UpdateFailure>> {
        let Some(prepare) = self.preparing.take() else {
            return Ok(None);
        };
        match prepare.stage {
            Stage::Selection(_) => self.continue_selection(prepare),
            Stage::Installs(_) => self.continue_installs(prepare),
            Stage::Devices(_) => self.continue_devices(prepare),
            Stage::Shadow(_) => self.continue_shadow(prepare),
        }
    }

    /// Moves the preparation to `stage`, whose completions wake this thread.
    fn enter(&mut self, mut prepare: Prepare, stage: Stage) {
        tracing::info!(
            generation = prepare.generation,
            stage = match &stage {
                Stage::Selection(_) => "the Build Server's selection",
                Stage::Installs(_) => "the kernel loads",
                Stage::Devices(_) => "this node's devices",
                Stage::Shadow(_) => "the schedulers' shadow step",
            },
            "the preparation entered a stage"
        );
        prepare.since = Instant::now();
        prepare.stage = stage;
        self.preparing = Some(prepare);
    }

    /// The selection stage: once the Build Server has selected the package of every key, the
    /// operators whose selection differs from what serves are installed, one install per device
    /// admitted behind it on the module-load lane.
    fn continue_selection(&mut self, mut prepare: Prepare) -> Result<Option<UpdateFailure>> {
        let Stage::Selection(task) = &prepare.stage else {
            return Ok(None);
        };
        let generation = prepare.generation;
        let refuse = |reason: String| UpdateFailure::Prepare { reason };
        let selection = match task.outcome() {
            // The server has not responded yet: its response wakes this thread.
            None => {
                self.preparing = Some(prepare);
                return Ok(None);
            }
            Some(Outcome::Success(output)) => match &*output {
                ModuleOutput::Selected(selection) => selection.clone(),
                // This stage holds the selection alone; an install here is a fault in this
                // process, and it is refused by name rather than read as a selection.
                ModuleOutput::Installed(_) => {
                    let failure =
                        refuse("the module lane ended the selection with a kernel load".to_owned());
                    return self.prepare_failed(generation, failure);
                }
            },
            Some(Outcome::Failed(error)) => {
                let failure = refuse(format!("selecting this generation's packages: {error}"));
                return self.prepare_failed(generation, failure);
            }
            Some(Outcome::Cancelled) => {
                return self
                    .prepare_failed(generation, refuse("the selection was cancelled".to_owned()));
            }
        };

        let serving = &self.state.generations;
        let never_installed = serving
            .serving
            .iter()
            .all(|generation| generation.is_empty());
        if never_installed && selection.packages.is_empty() {
            let failure = refuse(
                "the devices' profile runs launch no kernel, so this generation has nothing to \
                 install"
                    .to_owned(),
            );
            return self.prepare_failed(generation, failure);
        }
        let replaced = replaced_operators(&selection.packages, &serving.serving_packages);
        prepare.packages = selection
            .packages
            .into_iter()
            .filter(|(slot, _)| replaced.contains(&slot.op))
            .collect();
        prepare.routes = selection
            .routes
            .into_iter()
            .filter(|(slot, _)| replaced.contains(&slot.op))
            .collect();
        tracing::info!(
            generation,
            named = prepare.named.len(),
            installed = prepare.packages.len(),
            operators = replaced.len(),
            keys = prepare.routes.values().map(Vec::len).sum::<usize>(),
            devices = self.workers.len(),
            "this process is installing the operators whose selected packages differ from what \
             serves, with every key its devices launch"
        );
        let tasks = match self.admit_installs(&prepare.packages, &prepare.routes) {
            Ok(tasks) => tasks,
            Err(failure) => return self.prepare_failed(generation, failure),
        };
        for task in &tasks {
            self.wake_on_task(task);
        }
        self.enter(prepare, Stage::Installs(tasks));
        Ok(None)
    }

    /// The install stage: every device's kernels loaded build the next generations.
    fn continue_installs(&mut self, prepare: Prepare) -> Result<Option<UpdateFailure>> {
        let Stage::Installs(tasks) = &prepare.stage else {
            return Ok(None);
        };
        let generation = prepare.generation;
        let refuse = |reason: String| UpdateFailure::Prepare { reason };
        let mut loaded = Vec::with_capacity(tasks.len());
        for task in tasks {
            match task.outcome() {
                // A build is still running: its end wakes this thread.
                None => {
                    self.preparing = Some(prepare);
                    return Ok(None);
                }
                Some(Outcome::Success(output)) => match &*output {
                    ModuleOutput::Installed(entries) => loaded.push(entries.clone()),
                    // This stage holds installs; the selection has its own stage.
                    ModuleOutput::Selected(_) => {
                        let failure = refuse(
                            "the module lane ended a kernel load with the selection".to_owned(),
                        );
                        return self.prepare_failed(generation, failure);
                    }
                },
                Some(Outcome::Failed(error)) => {
                    let failure = refuse(format!("loading this node's kernels: {error}"));
                    return self.prepare_failed(generation, failure);
                }
                Some(Outcome::Cancelled) => {
                    let failure = refuse("the module load was cancelled".to_owned());
                    return self.prepare_failed(generation, failure);
                }
            }
        }
        self.after_modules(prepare, loaded)
    }

    /// The generations the loaded kernels build, the candidate policy instantiated, and the
    /// devices' rebind or recapture admitted.
    fn after_modules(
        &mut self,
        mut prepare: Prepare,
        loaded: Vec<Vec<KernelEntry>>,
    ) -> Result<Option<UpdateFailure>> {
        let generation = prepare.generation;
        // One generation per device: a kernel belongs to the device context
        // that loaded it.
        let mut next = Vec::with_capacity(loaded.len());
        for (index, entries) in loaded.into_iter().enumerate() {
            let current = Arc::clone(&self.state.generations.serving[index]);
            next.push(Arc::new(current.replace(generation, entries)));
        }

        // The candidate policy is instantiated on the current state, and every decision it
        // would make is tried against the hard constraints before anything switches.
        let policies = match self.request_policy(&prepare.policy) {
            Ok(policies) => policies,
            Err(error) => {
                let failure = UpdateFailure::Policy {
                    error: format!("{error:#}"),
                };
                return self.prepare_failed(generation, failure);
            }
        };

        // Rebind or recapture, as jobs between serving batches.
        let mut submissions = Vec::with_capacity(next.len());
        for (index, built) in next.iter().enumerate() {
            let submission = self.workers[index].prepare(Arc::clone(built))?;
            self.wake_on_submission(&submission);
            submissions.push(submission);
        }
        prepare.next = next;
        prepare.policies = policies;
        self.enter(prepare, Stage::Devices(submissions));
        Ok(None)
    }

    /// The device stage: every device rebinds or recaptures its buckets on the prepared
    /// generation while every scheduler builds the generation's policy.
    fn continue_devices(&mut self, mut prepare: Prepare) -> Result<Option<UpdateFailure>> {
        let generation = prepare.generation;
        let policy = responded(&mut prepare.policies);
        let Stage::Devices(submissions) = &prepare.stage else {
            return Ok(None);
        };
        // The policy's module is built on each scheduler's blocking thread, so its response
        // arrives here as well: a policy that did not prepare fails the generation before any
        // device switches, which is what the policy's trial run is for.
        if let Err(reason) = &policy {
            let stated = prepare
                .policy
                .arguments
                .get(TOKEN_LIMIT)
                .map(|value| value.to_string());
            let error = limit_refusal(reason, stated.as_deref(), self.prefill_bound);
            return self.prepare_failed(generation, UpdateFailure::Policy { error });
        }
        let devices = finished(submissions, "preparing this node's devices");
        match (policy, devices) {
            (Ok(true), Ok(true)) => self.start_shadow(prepare),
            // A forward that found a kernel missing names the operator and key it needed: a key
            // the devices' profile runs missed, which no second attempt is made to learn.
            (_, Err(reason)) => {
                let reason = match missing_kernel(submissions) {
                    Some((operator, key)) => format!(
                        "a device's forward launches {operator} under {key}, which its profile \
                         run did not report, so this generation holds no kernel for it - the \
                         profile run is the defect ({reason})"
                    ),
                    None => reason,
                };
                self.prepare_failed(generation, UpdateFailure::Prepare { reason })
            }
            // Something has not finished. Preparing an installed generation is a rebind or a
            // recapture - seconds of capture and a pinned host-to-device copy, never a compile -
            // so a stage that has stood past twice the watchdog's stall bound is failed here by
            // name, saying what never finished. Serving keeps what it had; the generation does
            // not.
            (policy, _) => {
                let waited = prepare.since.elapsed();
                if waited <= self.watchdog.stall_bound() * 2 {
                    self.preparing = Some(prepare);
                    return Ok(None);
                }
                // The stage holds two waits, and naming the wrong one costs a run: a policy
                // build that never responds is not a device that never finished.
                let cause = if matches!(policy, Ok(false)) {
                    "a scheduler's policy build never responded"
                } else {
                    "this node's devices never finished their preparation"
                };
                let pending = device_work(submissions, &self.workers, &prepare.policies);
                let reason = format!(
                    "{cause} after {}s ({pending}): the wait named there is the one to look at",
                    waited.as_secs()
                );
                self.prepare_failed(generation, UpdateFailure::Prepare { reason })
            }
        }
    }

    /// The prepared generation is complete on every device: the shadow step is asked of every
    /// scheduler when an operator's package changed, and the preparation finishes when none did.
    fn start_shadow(&mut self, mut prepare: Prepare) -> Result<Option<UpdateFailure>> {
        let generation = prepare.generation;
        // The shadow pass pairs a changed operator's kernel with the one serving, in the
        // buckets a running request occupies, so it is due only when an operator's package
        // changed. The set is empty for a first generation - nothing serves, so no kernel has an
        // older one to be paired against - and asking every scheduler there waits for a step an
        // idle engine never runs.
        let changed =
            changed_operators(&prepare.packages, &self.state.generations.serving_packages);
        if changed.is_empty() {
            tracing::info!(
                generation,
                "no operator's package changed, so this preparation asks for no shadow step"
            );
            return self.finish_prepare(prepare);
        }
        tracing::info!(
            generation,
            changed = changed.len(),
            operators = %changed.join(", "),
            "the operators this update changed, whose kernels the shadow step pairs"
        );

        // Shadow execution: every scheduler shadows one step of its own running requests
        // against the prepared generation, in the buckets those requests occupy, and responds
        // once that step ran. The paired kernels compare themselves by `Op::tolerance` and
        // never write resident memory.
        let shadow = self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.shadow(generation))
            .collect();
        prepare.changed = changed;
        self.enter(prepare, Stage::Shadow(shadow));
        Ok(None)
    }

    /// The shadow stage: every scheduler responds once its shadow step ran, and a step no
    /// running request runs abandons the update past twice the stall bound.
    fn continue_shadow(&mut self, mut prepare: Prepare) -> Result<Option<UpdateFailure>> {
        let generation = prepare.generation;
        let Stage::Shadow(responses) = &mut prepare.stage else {
            return Ok(None);
        };
        match responded(responses) {
            Ok(true) => self.finish_prepare(prepare),
            Err(reason) => self.prepare_failed(generation, UpdateFailure::Prepare { reason }),
            // A shadow step runs inside a batch the scheduler forms from its running requests.
            // An engine with none forms no batch, so the step never runs and nothing is
            // compared - and a comparison nobody made is how a wrong kernel reaches a
            // deployment. The update is abandoned with the operators named, rather than
            // waiting out the daemon's whole bound and then passing without the comparison.
            Ok(false) => {
                let waited = prepare.since.elapsed();
                if waited <= self.watchdog.stall_bound() * 2 {
                    self.preparing = Some(prepare);
                    return Ok(None);
                }
                let reason = shadow_stall_message(&prepare.changed, waited.as_secs());
                self.prepare_failed(generation, UpdateFailure::Prepare { reason })
            }
        }
    }

    /// A preparation that needs no further stage: its generation is built, its policy
    /// instantiated and its devices rebound or recaptured.
    ///
    /// A starting engine installs it at once as its first generation: nothing serves, so there
    /// is nothing to switch from and nothing to keep. A serving engine holds it as *prepared*
    /// and acknowledges; the daemon's `Switch` is what makes it execute.
    fn finish_prepare(&mut self, mut prepare: Prepare) -> Result<Option<UpdateFailure>> {
        let generation = prepare.generation;
        let generations = &mut self.state.generations;
        // What was built is *prepared*, not serving: a descriptor lookup still reads the serving
        // generation until the switch.
        generations.prepared = std::mem::take(&mut prepare.next);
        // What serves after the switch: what serves now, with every operator this preparation
        // installed replaced whole - its slots may divide it differently than the serving ones.
        let replaced: std::collections::BTreeSet<&str> = prepare
            .packages
            .keys()
            .map(|slot| slot.op.as_str())
            .collect();
        generations.prepared_packages = generations
            .serving_packages
            .iter()
            .filter(|(slot, _)| !replaced.contains(slot.op.as_str()))
            .map(|(slot, package)| (slot.clone(), package.clone()))
            .collect();
        generations
            .prepared_packages
            .extend(prepare.packages.clone());

        if self.state.lifecycle() == Lifecycle::Starting {
            return self.install_first(generation);
        }
        let mut uncompared = self.uncompared();
        if prepare.changed.is_empty() {
            // "Nothing to compare" and "nobody compared it" are different facts, and only one of
            // them is safe: a preparation that shadowed nothing says which it was.
            uncompared.insert(
                0,
                "no operator's package changed, so no kernel had an older one to compare with"
                    .to_owned(),
            );
        }
        let report = self.state.prepared(generation, uncompared);
        self.send_report(report);
        Ok(None)
    }

    /// The preparation of `generation` failed. A starting engine has nothing else to serve, so
    /// the failure is the engine's; a serving one drops what was built and serves on.
    fn prepare_failed(
        &mut self,
        generation: u64,
        failure: UpdateFailure,
    ) -> Result<Option<UpdateFailure>> {
        if self.state.lifecycle() == Lifecycle::Starting {
            tracing::error!(
                generation,
                reason = %failure.reason(),
                "the engine's first generation could not be prepared"
            );
            self.preparing = None;
            return self.failure(failure);
        }
        self.abandon_prepare(failure.reason());
        Ok(None)
    }

    /// Abandons the preparation in flight: what it built is discarded on every device, the
    /// candidate policy is dropped, and the daemon is told nothing changed.
    pub(super) fn abandon_prepare(&mut self, reason: String) {
        let generation = match (self.preparing.take(), self.state.update()) {
            (Some(prepare), _) => prepare.generation,
            (None, super::generation::Generation::Preparing { generation }) => generation,
            (None, _) => {
                tracing::warn!(%reason, "a preparation failed with none in flight");
                return;
            }
        };
        self.discard_generation(generation);
        let report = self.state.prepare_failed(generation, reason);
        self.send_report(report);
    }

    /// Gives back what was built for `generation`, which never executed: the preparation still
    /// in flight, the candidate policy, and on every device its graphs and kernels.
    pub(super) fn discard_generation(&mut self, generation: u64) {
        self.preparing = None;
        self.discard_policy();
        self.discard_devices(generation);
    }

    /// Discards `generation`'s graphs and kernels on every device.
    ///
    /// The devices may already have rebound or recaptured its graphs, and a generation that
    /// never serves must give its captures back. It is discarded, not released: a release names
    /// a generation that executed. A device still busy with the capture discards after it, by its
    /// queue's own order.
    pub(super) fn discard_devices(&mut self, generation: u64) {
        for worker in &self.workers {
            match worker.discard(generation) {
                Ok(submission) => {
                    if let Err(error) =
                        device_command(&self.watchdog, worker.device(), "discard", submission)
                    {
                        // The generation never serves: a discard that did not finish holds
                        // storage until the process ends, and nothing that serves fails for it.
                        tracing::warn!(%error, generation, "discarding a generation on a device failed");
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, generation, "queueing a generation's discard failed");
                }
            }
        }
    }
}
