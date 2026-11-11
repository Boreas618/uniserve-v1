//! The precopy of the generation update: the new generation's kernels are installed on the
//! module-load lane, every device rebinds or recaptures its buckets on it while every scheduler
//! prepares the policy, and the schedulers shadow one step of their resident requests when an
//! operator's package changed. Each stage is polled by the serve loop and advanced at most once
//! per pass ([`PrecopyStage`]); a failure discards the generation and serving goes on.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use ipc::execution::Outcome;
use ipc::execution::Submission;
use ipc::{GenerationStatus, UpdateFailure};
use ops::{Generation, KernelEntry};
use proto::module::{PackageRevision, Packages};
use worker::device::Worker;
use worker::host::HostTask;

use super::MainProcess;
use super::device_command;
use super::policy::{TOKEN_LIMIT, limit_refusal};
use crate::modules::load::{ModuleAnswer, ModuleWork};
use crate::modules::select::{Routes, changed_operators, replaced_operators};

/// The operator and descriptor key a failed walk named, when that is what failed.
///
/// A generation holds every kernel its buckets launch: the precopy installs every key the devices'
/// recorded walks reported (`worker::runners::recording`) in one admission. A walk that still finds
/// one missing names it here, and the precopy refuses with it: a key the recording missed is the
/// recording's defect, never a key to learn and install on another attempt.
fn missing_kernel(submissions: &[Submission]) -> Option<(String, String)> {
    /// The kernel an error names as missing, whether it reached the worker as an operator's
    /// error or wrapped in the model layer's: a capture reports it through
    /// `nn::Forward::operate`, which resolves the kernel and converts the operator's error into the
    /// layer's own. Matching only the operator's arm would leave every capture's miss unrecognised,
    /// and the discovery loop would abandon a generation it could have completed.
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

/// Polls one set of scheduler answers.
///
/// The schedulers answer at their own boundary, so the serve pass polls them: the control thread
/// may not wait on a stage whose length the daemon's load decides.
fn poll_answers(
    answers: &mut Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>,
) -> Option<StageFailure> {
    // An answer is taken once and its sender is gone after it, so a receiver that answered leaves
    // the list: polling it again would read a scheduler that finished as one that was lost.
    while let Some(receiver) = answers.first() {
        match receiver.try_recv() {
            Ok(Ok(())) => {
                answers.remove(0);
            }
            Ok(Err(reason)) => return Some(StageFailure::Failed(reason)),
            Err(crossbeam_channel::TryRecvError::Empty) => return Some(StageFailure::Pending),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                return Some(StageFailure::Failed(
                    "a scheduler of the group is gone".to_owned(),
                ));
            }
        }
    }
    None
}

/// The failure a shadow step that never ran is.
///
/// The step runs inside a batch, and an engine with no resident request forms none: the stage waits
/// for something that cannot happen, so the update is abandoned with the operators named rather than
/// passing without the comparison it was there to make.
fn shadow_stall_message(changed: &[String], waited: u64) -> String {
    let count = changed.len();
    let operators = changed.join(", ");
    let noun = if count == 1 {
        "changed operator"
    } else {
        "changed operators"
    };
    format!(
        "the shadow step of {count} {noun} ({operators}) waited {waited}s: no resident request ran \
         a step, so nothing was compared - the update is abandoned rather than passed without the \
         comparison it was prepared for"
    )
}

/// The precopy of one generation, advanced between serve passes.
pub(super) struct Precopy {
    pub(super) generation: u64,
    /// The operators whose package this precopy changed, in sorted order: the set the shadow pass
    /// pairs, and the set a stall names. Empty when nothing changed - a first generation, or an
    /// update that touched no kernel.
    changed: Vec<String>,
    /// The policy package the update installs.
    policy: PackageRevision,
    /// The packages the deployment names, by slot: the backends the selection prefers.
    named: Packages,
    /// The kernel packages the update installs, by slot: every slot of each operator whose selected
    /// packages differ from what serves, which on the epoch's first generation is every operator.
    packages: Packages,
    /// The keys the Build Server selected each installed slot's package for.
    routes: Routes,
    /// The stage the precopy is in, since when, and whether the wait has already been reported:
    /// every stage of the precopy is a poll, so a stage that never advances is a run that hangs
    /// without a line unless the process says which step it is standing on.
    stage_name: &'static str,
    since: Instant,
    warned: bool,
    /// The generation built for each device, in `workers` order.
    next: Vec<Arc<Generation>>,
    /// The answers of every scheduler that is preparing the update's policy package. The build runs
    /// on the schedulers' own blocking threads, so the serve pass polls them instead of waiting.
    policies: Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>,
    stage: PrecopyStage,
}

/// The stage a precopy is in.
enum PrecopyStage {
    /// The module-load lane, in the two steps the update runs it in.
    Modules(ModuleLane),
    /// The workers rebind or recapture their buckets.
    Workers(Vec<Submission>),
    /// The shadow pass compares the new implementations on the resident requests, one step per
    /// scheduler; each answers when its step ran.
    Shadow(Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>),
    /// A stage being replaced; the driver never observes it.
    Done,
}

/// The module-load lane's two steps, in the order the update needs them.
enum ModuleLane {
    /// The Build Server's selection of the generation's packages, alone on the lane: the
    /// installs are derived from it.
    Selection(Arc<HostTask<ModuleWork>>),
    /// One install per device for the generation the listing resolved.
    Installs(Vec<Arc<HostTask<ModuleWork>>>),
}

/// The name of a precopy stage, for the line that says which step the process is standing on.
fn stage_name(stage: &PrecopyStage) -> &'static str {
    match stage {
        PrecopyStage::Modules(_) => "the module load lane",
        PrecopyStage::Workers(_) => "this node's devices",
        PrecopyStage::Shadow(_) => "the schedulers' shadow step",
        PrecopyStage::Done => "done",
    }
}

/// What polling a stage's submissions found.
enum StageFailure {
    /// At least one submission has not finished.
    Pending,
    /// A submission failed, or was cancelled.
    Failed(String),
}

/// What a device stage is waiting on: each unanswered device's own account, and each scheduler's
/// answer to the policy it was asked to build.
///
/// The two waits live in one stage, and a report that names only the devices says "the device thread
/// is the one to look at" while the device has answered and a policy build is what never came back -
/// which is a wrong name, worse than none. The policy figure is stated first for that reason: it is
/// the wait the devices' own account cannot show.
fn device_work(
    submissions: &[Submission],
    workers: &[Arc<Worker>],
    policies: &[crossbeam_channel::Receiver<std::result::Result<(), String>>],
) -> String {
    // The label alone says which stage stood; the device thread's own account says whether it is
    // working (a capture in progress, a bucket being recaptured) or waiting on something outside it.
    // A capture is open or it is not, and the two look identical from the control thread without it.
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
    let answered = submissions.len() - pending.len();
    let devices = format!(
        "{answered} of {} device preparation(s) answered ({})",
        submissions.len(),
        pending.join("; ")
    );
    let policies: String = if policies.is_empty() {
        "no policy build was asked for".to_owned()
    } else {
        let answered = policies.iter().filter(|answer| !answer.is_empty()).count();
        format!("{answered} of {} policy build(s) answered", policies.len())
    };
    format!("{devices}; {policies}")
}

/// What one stage is still waiting for, in one phrase.
///
/// A stage's name says where a hang is, not which of the stage's several waits it is: the module
/// lane holds one task per device, the worker stage one submission per device, and the shadow
/// stage one step per scheduler. The stall line prints the counts so a run names the wait that
/// never answered without a second run built for the purpose.
fn pending_work(
    stage: &PrecopyStage,
    workers: &[Arc<Worker>],
    policies: &[crossbeam_channel::Receiver<std::result::Result<(), String>>],
) -> String {
    match stage {
        PrecopyStage::Modules(ModuleLane::Selection(task)) => match task.outcome() {
            None => "the Build Server's selection of the generation's packages".to_owned(),
            Some(_) => "the Build Server's selection (answered)".to_owned(),
        },
        PrecopyStage::Modules(ModuleLane::Installs(tasks)) => {
            let answered = tasks.iter().filter(|task| task.outcome().is_some()).count();
            format!("{answered} of {} kernel load(s) answered", tasks.len())
        }
        PrecopyStage::Workers(submissions) => device_work(submissions, workers, policies),
        PrecopyStage::Shadow(answers) => {
            let answered = answers.iter().filter(|answer| !answer.is_empty()).count();
            format!("{answered} of {} shadow step(s) answered", answers.len())
        }
        // The driver replaces the stage with this before it looks at it, so it is never the wait.
        PrecopyStage::Done => "no stage".to_owned(),
    }
}

/// Whether every submission finished, and if not, why the first unfinished or
/// failed one did not.
fn unfinished(submissions: &[Submission], what: &str) -> Option<StageFailure> {
    for submission in submissions {
        match submission.outcome() {
            None => return Some(StageFailure::Pending),
            Some(Outcome::Success(_)) => {}
            Some(Outcome::Failed(error)) => {
                return Some(StageFailure::Failed(format!("{what}: {error}")));
            }
            Some(Outcome::Cancelled) => {
                return Some(StageFailure::Failed(format!("{what}: cancelled")));
            }
        }
    }
    None
}

impl MainProcess {
    /// The precopy, first half: admits the Build Server's selection of the new generation's
    /// packages on the module lane; the installs follow from its answer, and their builds run while
    /// serving goes on.
    ///
    /// The epoch's first generation and an update go the same way. Every key this node's devices
    /// launch is asked of the Build Server once, with the backends `packages` names, and the
    /// server selects the package of each key (`BuildServer::cover`). The first generation
    /// installs every selected slot in one admission; an update installs only the operators whose
    /// selection differs from what serves - a newly installed revision of an operator the
    /// deployment names nothing for is such a difference - and a kernel that stays keeps its `Arc`,
    /// so its buckets and its graphs stay too.
    pub(super) fn start_precopy(
        &mut self,
        generation: u64,
        packages: &Packages,
        policy: &PackageRevision,
    ) -> Result<()> {
        if let Err(error) = self.update.begin_precopy(generation) {
            let failure = UpdateFailure::Precopy {
                reason: error.to_string(),
            };
            self.answer(GenerationStatus::PrecopyFailed {
                generation,
                failure,
            });
            return Ok(());
        }

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
            "asking the Build Server to select this generation's packages"
        );
        let task = match self.admit_selection(packages, keys) {
            Ok(task) => task,
            Err(failure) => return self.abandon_precopy(generation, failure),
        };
        self.precopy = Some(Precopy {
            generation,
            changed: Vec::new(),
            policy: policy.clone(),
            named: packages.clone(),
            packages: Packages::new(),
            routes: Routes::new(),
            stage_name: "modules",
            since: Instant::now(),
            warned: false,
            next: Vec::new(),
            policies: Vec::new(),
            stage: PrecopyStage::Modules(ModuleLane::Selection(task)),
        });
        Ok(())
    }

    /// The operators this node's devices could not compare, each with its reason, in device order
    /// and without repeats.
    ///
    /// It is carried by the precopy's answer so an update that has no evidence for an operator says
    /// so in the record the daemon reads, not only in this process's log: empty is the statement
    /// that every operator the update replaced was compared.
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

    /// The precopy, second half: advances the precopy at most one stage. It
    /// never waits on a job: a stage that has not finished leaves the precopy
    /// as it is, and the next serve pass looks again.
    pub(super) fn advance_precopy(&mut self) -> Result<()> {
        let Some(mut precopy) = self.precopy.take() else {
            return Ok(());
        };
        let generation = precopy.generation;
        // Every stage of the precopy is polled rather than waited for, so a stage that never
        // advances is the one shape of hang this process can have. It says so twice: when it enters
        // a stage, and once, by name, when that stage has stood longer than the watchdog's stall
        // bound - the same bound the watchdog uses for a job, applied to the step itself.
        let entered = stage_name(&precopy.stage);
        if entered != precopy.stage_name {
            precopy.stage_name = entered;
            precopy.since = Instant::now();
            precopy.warned = false;
            tracing::info!(
                generation,
                stage = entered,
                "the precopy entered a stage of the update"
            );
        } else if !precopy.warned && precopy.since.elapsed() > self.watchdog.stall_bound() {
            precopy.warned = true;
            tracing::warn!(
                generation,
                stage = entered,
                waiting_on = %pending_work(&precopy.stage, &self.workers, &precopy.policies),
                waited_s = precopy.since.elapsed().as_secs(),
                "the precopy has stood in this stage past the watchdog's stall bound and nothing \
                 has answered: the Build Server, a device or a scheduler is the one to look at"
            );
        }
        // What the stage was waiting on, read before the stage is taken out: a bound that named
        // `no stage` because the driver had already replaced it would say nothing at all, which is
        // the opposite of what the line is for.
        let pending = pending_work(&precopy.stage, &self.workers, &precopy.policies);
        let stage = std::mem::replace(&mut precopy.stage, PrecopyStage::Done);

        match stage {
            PrecopyStage::Modules(ModuleLane::Selection(task)) => {
                self.advance_selection(precopy, task)
            }
            PrecopyStage::Modules(ModuleLane::Installs(tasks)) => {
                self.advance_installs(precopy, tasks)
            }
            PrecopyStage::Workers(submissions) => {
                self.advance_devices(precopy, submissions, &pending)
            }
            PrecopyStage::Shadow(answers) => self.advance_shadow(precopy, answers),
            PrecopyStage::Done => Ok(()),
        }
    }

    /// The selection stage: once the Build Server has selected the package of every key, the
    /// operators whose selection differs from what serves are installed, one install per device
    /// admitted behind it on the module lane.
    fn advance_selection(
        &mut self,
        mut precopy: Precopy,
        task: Arc<HostTask<ModuleWork>>,
    ) -> Result<()> {
        let generation = precopy.generation;
        let refuse = |reason: String| UpdateFailure::Precopy { reason };
        let selection = match task.outcome() {
            // The server has not answered yet: the lane keeps the task and this pass ends with the
            // precopy where it was, as every stage of the precopy does.
            None => {
                precopy.stage = PrecopyStage::Modules(ModuleLane::Selection(task));
                self.precopy = Some(precopy);
                return Ok(());
            }
            Some(Outcome::Success(answer)) => match &*answer {
                ModuleAnswer::Selected(selection) => selection.clone(),
                // This stage holds the selection alone; an install here is a fault in this
                // process, and it is refused by name rather than read as a selection.
                ModuleAnswer::Installed(_) => {
                    let failure = refuse(
                        "the module lane answered the selection with a kernel load".to_owned(),
                    );
                    return self.abandon_precopy(generation, failure);
                }
            },
            Some(Outcome::Failed(error)) => {
                let failure = refuse(format!("selecting this generation's packages: {error}"));
                return self.abandon_precopy(generation, failure);
            }
            Some(Outcome::Cancelled) => {
                let failure = refuse("the selection was cancelled".to_owned());
                return self.abandon_precopy(generation, failure);
            }
        };

        let never_installed = self
            .update
            .generations
            .iter()
            .all(|generation| generation.is_empty());
        if never_installed && selection.packages.is_empty() {
            let failure = refuse(
                "the devices' recorded walks launch no kernel, so this generation has nothing to \
                 install"
                    .to_owned(),
            );
            return self.abandon_precopy(generation, failure);
        }
        let replaced = replaced_operators(&selection.packages, &self.update.serving_packages);
        precopy.packages = selection
            .packages
            .into_iter()
            .filter(|(slot, _)| replaced.contains(&slot.op))
            .collect();
        precopy.routes = selection
            .routes
            .into_iter()
            .filter(|(slot, _)| replaced.contains(&slot.op))
            .collect();
        tracing::info!(
            generation,
            named = precopy.named.len(),
            installed = precopy.packages.len(),
            operators = replaced.len(),
            keys = precopy.routes.values().map(Vec::len).sum::<usize>(),
            devices = self.workers.len(),
            "this process is installing the operators whose selected packages differ from what \
             serves, with every key its devices launch"
        );
        let tasks = match self.admit_installs(&precopy.packages, &precopy.routes) {
            Ok(tasks) => tasks,
            Err(failure) => return self.abandon_precopy(generation, failure),
        };
        precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
        self.precopy = Some(precopy);
        Ok(())
    }

    /// The install stage: every device's kernels loaded build the next generations.
    fn advance_installs(
        &mut self,
        mut precopy: Precopy,
        tasks: Vec<Arc<HostTask<ModuleWork>>>,
    ) -> Result<()> {
        let generation = precopy.generation;
        let mut loaded = Vec::with_capacity(tasks.len());
        for task in &tasks {
            match task.outcome() {
                None => {
                    // A build is still running: the tasks keep their
                    // results and the lane keeps its capacity.
                    precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
                    self.precopy = Some(precopy);
                    return Ok(());
                }
                Some(Outcome::Success(answer)) => match &*answer {
                    ModuleAnswer::Installed(entries) => loaded.push(entries.clone()),
                    // This stage holds installs; the selection has its own stage.
                    ModuleAnswer::Selected(_) => {
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Precopy {
                                reason: "the module lane answered a kernel load with the \
                                         selection"
                                    .to_owned(),
                            },
                        );
                    }
                },
                Some(Outcome::Failed(error)) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!("loading this node's kernels: {error}"),
                        },
                    );
                }
                Some(Outcome::Cancelled) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: "the module load was cancelled".to_owned(),
                        },
                    );
                }
            }
        }
        self.after_modules(precopy, loaded)
    }

    /// The device stage: every device rebinds or recaptures its buckets on the prepared generation
    /// while every scheduler builds the update's policy. `pending` names what the stage was waiting
    /// on when this pass began, for the failure a stage past its bound states.
    fn advance_devices(
        &mut self,
        mut precopy: Precopy,
        submissions: Vec<Submission>,
        pending: &str,
    ) -> Result<()> {
        let generation = precopy.generation;
        // A device that never answers is a process that serves nothing until the epoch
        // ends: the stage is polled, so without this bound the only line it prints is the
        // stall warning, and the run waits out the daemon's whole prepare bound to learn
        // what the log already showed. Preparing an installed generation is a rebind or a
        // recapture - seconds of capture and a pinned upload, never a compile - so a stage
        // that has stood past twice the watchdog's stall bound is failed here by name,
        // saying what never answered. Serving keeps what it had; the update does not.
        if precopy.since.elapsed() > self.watchdog.stall_bound() * 2 {
            // The arm holds two waits, and naming the wrong one costs a run: a policy build
            // that never answers is not a device that never answered, and the previous
            // message said the device either way. The policy is the narrower of the two, so
            // it is asked first.
            let policy = matches!(
                poll_answers(&mut precopy.policies),
                Some(StageFailure::Pending)
            );
            let cause = if policy {
                "a scheduler's policy build never answered"
            } else {
                "this node's devices never answered their preparation"
            };
            return self.abandon_precopy(
                generation,
                UpdateFailure::Precopy {
                    reason: format!(
                        "{cause} after {}s ({pending}): the wait named there is the one \
                         to look at",
                        precopy.since.elapsed().as_secs()
                    ),
                },
            );
        }
        // The policy's module is built on each scheduler's blocking thread, so its answer
        // arrives here as well: a policy that did not prepare fails the update before any
        // device cuts over, which is what the policy's dry run is for.
        match poll_answers(&mut precopy.policies) {
            Some(StageFailure::Pending) => {
                precopy.stage = PrecopyStage::Workers(submissions);
                self.precopy = Some(precopy);
                return Ok(());
            }
            Some(StageFailure::Failed(reason)) => {
                let stated = precopy
                    .policy
                    .arguments
                    .get(TOKEN_LIMIT)
                    .map(|value| value.to_string());
                let error = limit_refusal(&reason, stated.as_deref(), self.catalogue);
                return self.abandon_precopy(generation, UpdateFailure::Policy { error });
            }
            None => {}
        }
        match unfinished(&submissions, "preparing this node's devices") {
            Some(StageFailure::Pending) => {
                precopy.stage = PrecopyStage::Workers(submissions);
                self.precopy = Some(precopy);
                Ok(())
            }
            // A walk that found a kernel missing names the operator and key it needed: a key the
            // devices' recorded walks missed, which no second attempt is made to learn.
            Some(StageFailure::Failed(reason)) => {
                let reason = match missing_kernel(&submissions) {
                    Some((operator, key)) => format!(
                        "a device's walk launches {operator} under {key}, which its recorded walk \
                         did not report, so this generation holds no kernel for it - the recording \
                         is the defect ({reason})"
                    ),
                    None => reason,
                };
                self.abandon_precopy(generation, UpdateFailure::Precopy { reason })
            }
            None => self.start_shadow(precopy),
        }
    }

    /// The prepared generation is complete on every device: the shadow step is asked of every
    /// scheduler when an operator's package changed, and the precopy finishes when none did.
    fn start_shadow(&mut self, mut precopy: Precopy) -> Result<()> {
        let generation = precopy.generation;
        // The precopy's shadow pass pairs a changed operator's kernel with the one serving,
        // in the buckets a resident request occupies, so it is owed only when an operator's
        // package changed. The set is empty on an epoch's first precopy - nothing
        // serves, so no kernel has an old one to be paired against - and asking every
        // scheduler there waits for a step an idle epoch never runs.
        let changed = changed_operators(&precopy.packages, &self.update.serving_packages);
        if changed.is_empty() {
            tracing::info!(
                generation,
                "no operator's package changed, so this precopy asks for no shadow step"
            );
            return self.finish_precopy(precopy);
        }
        tracing::info!(
            generation,
            changed = changed.len(),
            operators = %changed.join(", "),
            "the operators this update changed, whose kernels the shadow step pairs"
        );

        // Shadow execution: every scheduler shadows one step of its own resident requests
        // against the prepared generation, in the buckets those requests occupy, and
        // answers once that step ran. The paired kernels compare themselves
        // by `Op::tolerance` and never write resident memory.
        let mut shadow = Vec::with_capacity(self.schedulers.len());
        for scheduler in &self.schedulers {
            shadow.push(scheduler.shadow(generation));
        }
        precopy.changed = changed;
        precopy.stage = PrecopyStage::Shadow(shadow);
        self.precopy = Some(precopy);
        Ok(())
    }

    /// The shadow stage: every scheduler answers once its shadow step ran, and a step no resident
    /// request runs abandons the update past twice the stall bound.
    fn advance_shadow(
        &mut self,
        mut precopy: Precopy,
        mut answers: Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>,
    ) -> Result<()> {
        let generation = precopy.generation;
        // A shadow step runs inside a batch the scheduler forms from its resident requests.
        // An engine with none forms no batch, so the step never runs and nothing is
        // compared - and a comparison nobody made is how a wrong kernel reaches a
        // deployment. The update is abandoned with the operators named, rather than
        // waiting out the daemon's whole bound and then passing without the comparison.
        if precopy.since.elapsed() > self.watchdog.stall_bound() * 2 {
            let reason = shadow_stall_message(&precopy.changed, precopy.since.elapsed().as_secs());
            return self.abandon_precopy(generation, UpdateFailure::Precopy { reason });
        }
        // The shadow pass runs on the scheduler's own thread, one step of the batches it
        // formed, so the serve pass polls it rather than waiting.
        match poll_answers(&mut answers) {
            Some(StageFailure::Pending) => {
                precopy.stage = PrecopyStage::Shadow(answers);
                self.precopy = Some(precopy);
                return Ok(());
            }
            Some(StageFailure::Failed(reason)) => {
                return self.abandon_precopy(generation, UpdateFailure::Precopy { reason });
            }
            None => {}
        }

        self.finish_precopy(precopy)
    }

    /// A precopy that needs no further stage: its generation is built, its policy instantiated and
    /// its workers rebound or recaptured, so it waits as *prepared* for the switchover.
    ///
    /// Reached from the shadow stage and, directly, from a precopy whose changed set is empty - the
    /// two are the same finished state, and a first generation takes the second because there is no
    /// old kernel to shadow.
    fn finish_precopy(&mut self, mut precopy: Precopy) -> Result<()> {
        let generation = precopy.generation;
        // The generations built here are *prepared*, not serving: `generations` is what each device
        // serves now, and a precopy that finished is not serving until the switchover (steps 2
        // and 3). They wait here, so a descriptor lookup still reads the serving generation and a
        // revert has nothing to undo.
        self.update.prepared = std::mem::take(&mut precopy.next);
        // What serves after the switchover: what serves now, with every operator this precopy
        // installed replaced whole - its slots may divide it differently than the serving ones.
        let replaced: std::collections::BTreeSet<&str> = precopy
            .packages
            .keys()
            .map(|slot| slot.op.as_str())
            .collect();
        self.update.prepared_packages = self
            .update
            .serving_packages
            .iter()
            .filter(|(slot, _)| !replaced.contains(slot.op.as_str()))
            .map(|(slot, package)| (slot.clone(), package.clone()))
            .collect();
        self.update
            .prepared_packages
            .extend(precopy.packages.clone());
        if let Err(error) = self.update.precopied(generation) {
            // The generation this precopy built is not the one the machine waits for - a revert or a
            // later precopy took its place while the builds ran. The revert path discards what was
            // built and answers by name; the process goes on serving what it served.
            tracing::warn!(generation, %error, "a precopy finished for an update that moved on");
            return self
                .revert(
                    generation,
                    UpdateFailure::Precopy {
                        reason: format!(
                            "the precopy finished for a generation that is not in flight: {error}"
                        ),
                    },
                )
                .map(|_| ());
        }
        let mut uncompared = self.uncompared();
        if precopy.changed.is_empty() {
            // "Nothing to compare" and "nobody compared it" are different facts, and only one of
            // them is safe: a precopy that shadowed nothing says which it was.
            uncompared.insert(
                0,
                "no operator's package changed, so no kernel had an older one to compare with"
                    .to_owned(),
            );
        }
        self.answer(GenerationStatus::Precopied {
            generation,
            uncompared,
        });
        Ok(())
    }

    /// The generations the loaded kernels build, the candidate policy
    /// instantiated, and the workers' rebind or recapture admitted.
    fn after_modules(&mut self, precopy: Precopy, loaded: Vec<Vec<KernelEntry>>) -> Result<()> {
        let generation = precopy.generation;
        // One generation per device: a kernel belongs to the device context
        // that loaded it.
        let mut next = Vec::with_capacity(loaded.len());
        for (index, entries) in loaded.into_iter().enumerate() {
            let current = Arc::clone(&self.update.generations[index]);
            next.push(Arc::new(current.replace(generation, entries)));
        }

        // The candidate policy is instantiated on the current snapshot, and
        // every decision it would make is dry-run against the hard constraints
        // before anything switches.
        let policies = match self.request_policy(&precopy.policy) {
            Ok(policies) => policies,
            Err(error) => {
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Policy {
                        error: format!("{error:#}"),
                    },
                );
            }
        };

        // Rebind or recapture, as jobs between serving batches.
        let mut submissions = Vec::with_capacity(next.len());
        for (index, built) in next.iter().enumerate() {
            submissions.push(self.workers[index].precopy(Arc::clone(built))?);
        }
        let mut precopy = precopy;
        precopy.next = next;
        precopy.policies = policies;
        precopy.stage = PrecopyStage::Workers(submissions);
        self.precopy = Some(precopy);
        Ok(())
    }

    /// A failed precopy: the new generation is discarded and serving goes on.
    pub(super) fn abandon_precopy(
        &mut self,
        generation: u64,
        failure: UpdateFailure,
    ) -> Result<()> {
        self.discard_policy();
        // The workers already rebound or recaptured this generation's graphs, and a generation
        // that never serves must give its captures back. A purge names a generation a
        // switchover or a revert released, and this one was never cut over, so it is discarded: the
        // worker drops exactly this generation's graphs and kernels. A device still busy with the
        // capture is answered by the queue's own order.
        for worker in &self.workers {
            match worker.discard(generation) {
                Ok(submission) => {
                    if let Err(error) =
                        device_command(&self.watchdog, worker.device(), "discard", submission)
                    {
                        tracing::warn!(%error, "discarding a failed generation on a device failed");
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "queueing a failed generation's discard failed");
                }
            }
        }
        self.precopy = None;
        if let Err(error) = self.update.precopy_failed(generation, failure.clone()) {
            tracing::error!(%error, "discarding a failed precopy failed");
        }
        self.answer(GenerationStatus::PrecopyFailed {
            generation,
            failure,
        });
        Ok(())
    }
}
