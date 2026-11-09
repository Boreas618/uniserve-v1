//! The precopy of the program-layer update: the new generation's kernels are installed on the
//! module-load lane, every device rebinds or recaptures its buckets on it while every scheduler
//! prepares the policy, and the schedulers shadow one step of their resident requests when an
//! operator's program changed. Each stage is polled by the serve loop and advanced at most once
//! per pass ([`PrecopyStage`]); a failure discards the generation and serving goes on.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use ipc::{ProgramStatus, UpdateFailure};
use ops::{Generation, KernelEntry};
use program_server::Program;
use program_server::program::Programs;
use worker::batch::Submission;
use worker::completion::Outcome;
use worker::device::Worker;
use worker::host::HostTask;

use super::MainProcess;
use super::device_command;
use super::installs::Advance;
use super::policy::{TOKEN_LIMIT, limit_refusal};
use crate::programs::modules::{ListStore, ModuleAnswer, ModuleWork, UnavailableCell};
use crate::programs::select::{Candidates, changed_operators, updated_programs};

/// The operator and descriptor key a failed walk named, when that is what failed.
///
/// A generation holds every kernel its buckets launch, and a walk that finds one missing says
/// which: this is the only producer of an epoch's first generation's descriptor keys, because a
/// descriptor is what the model's forward computes for a bucket and nothing outside the forward can
/// list them.
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
        if let Some((operator, descriptor)) = named(&error) {
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
    /// The peer whose command this precopy answers, when a peer asked.
    pub(super) reply: Option<u32>,
    /// The operators whose program this precopy changed, in sorted order: the set the shadow pass
    /// pairs, and the set a stall names. Empty when nothing changed - a first generation, or an
    /// update that touched no kernel.
    changed: Vec<String>,
    /// The policy program the update installs.
    policy: Program,
    /// The kernel programs the update installs, by slot: kept so the discovery loop can admit the
    /// installs again with a key it just learned, or with the operator's next candidate program.
    pub(super) programs: Programs,
    /// Every program this host may fill an operator with, when this generation's programs came from
    /// the store's listing rather than from the daemon. Empty when the daemon named them: a named
    /// program is the deployment's choice and the walk does not second-guess it.
    pub(super) candidates: Candidates,
    /// The candidates this generation has already asked for. Each is tried once, and none again.
    pub(super) tried: std::collections::BTreeSet<(String, u32)>,
    /// The backends tried for each operator, so a refusal names every one of them.
    pub(super) backends: BTreeMap<String, Vec<String>>,
    /// The refusal the generation's installs met, shared with the device loads that found it.
    pub(super) unavailable: UnavailableCell,
    /// The stage the precopy is in, since when, and whether the wait has already been reported:
    /// every stage of the precopy is a poll, so a stage that never advances is a run that hangs
    /// without a line unless the process says which step it is standing on.
    stage_name: &'static str,
    since: Instant,
    warned: bool,
    /// The descriptor keys the walk has taught this generation, by operator. The epoch's first
    /// generation has no serving generation to take them from.
    pub(super) learned: BTreeMap<String, Vec<String>>,
    /// Whether every scheduler has already been asked for the candidate policy: the discovery loop
    /// re-enters the stages, and the policy is prepared once.
    policy_requested: bool,
    /// The generation built for each device, in `workers` order.
    next: Vec<Arc<Generation>>,
    /// The answers of every scheduler that is preparing the update's policy program. The build runs
    /// on the schedulers' own blocking threads, so the serve pass polls them instead of waiting.
    policies: Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>,
    stage: PrecopyStage,
}

impl Precopy {
    /// Whether this generation holds a program for `operator`.
    ///
    /// A program is named by a slot, and a slot names an operator and an optional phase, so the
    /// question a walk asks - "does this generation have an implementation of this operator?" - is
    /// answered by any slot of that operator, whatever phase it states.
    fn names(&self, operator: &str) -> bool {
        self.programs
            .keys()
            .any(|slot| ops::operator_name(&slot.op) == Some(operator))
    }
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
    /// The store's listing, alone on the lane: the installs are derived from it.
    Listing(Arc<HostTask<ModuleWork>>),
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
        PrecopyStage::Modules(ModuleLane::Listing(task)) => match task.outcome() {
            None => "the Program Server's store listing".to_owned(),
            Some(_) => "the store's listing (answered)".to_owned(),
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
    /// The precopy, first half: states the new generation to the Program Server
    /// and admits the lane's work - the store's listing, then one install per
    /// device - whose builds run while serving goes on.
    pub(super) fn start_precopy(
        &mut self,
        generation: u64,
        programs: &Programs,
        policy: &Program,
        reply: Option<u32>,
    ) -> Result<()> {
        if let Err(error) = self.update.begin_precopy(generation) {
            let failure = UpdateFailure::Precopy {
                reason: error.to_string(),
            };
            self.answer(
                reply,
                ProgramStatus::PrecopyFailed {
                    generation,
                    failure,
                },
            );
            return Ok(());
        }

        // A `Precopy` that names no kernel program is the startup switch: the daemon states the
        // scheduling policy alone, because on a host that runs the worker path there is no kernel
        // package to prepare (`crates/daemon/src/leader.rs`, `install_programs`). This process
        // installs what it can actually launch, so the store's listing is read first, on the module
        // lane, rather than capturing an empty generation.
        //
        // A policy-only *update* also names no kernel, and resolving there would be wrong twice
        // over: every operator would be re-installed as a new generation, so every kernel's `Arc`
        // would differ from the serving one and every captured bucket would be rebound or
        // recaptured for a change no program made. The two states are told apart by what this
        // process already serves - a startup has no kernel at all, an update has all of them.
        //
        // Each precopy reads the store at most once, and never one precopy's answer in another's
        // place: a package built between two updates is a program the second one may install.
        //
        // The first generation's read is the lane's. An update's - the keys a package declares, the
        // candidates an operator has - is still made here, on the control thread, by whichever step
        // asks for it first; it is the same call the startup switch no longer makes, and the step
        // that would carry it to the lane is the candidate walk, not this one.
        self.store = None;
        let never_installed = self
            .generations
            .iter()
            .all(|generation| generation.is_empty());
        // An update names the epoch's whole set, and only what differs from what serves is
        // installed: a kernel that stays keeps its `Arc`, so its buckets and its graphs stay too.
        let updated;
        let programs = if never_installed {
            programs
        } else {
            updated = updated_programs(programs, &self.serving_programs);
            &updated
        };
        let mut precopy = Precopy {
            generation,
            reply,
            changed: Vec::new(),
            policy: policy.clone(),
            programs: programs.clone(),
            candidates: Candidates::new(),
            tried: std::collections::BTreeSet::new(),
            backends: BTreeMap::new(),
            unavailable: UnavailableCell::default(),
            stage_name: "modules",
            since: Instant::now(),
            warned: false,
            learned: BTreeMap::new(),
            policy_requested: false,
            next: Vec::new(),
            policies: Vec::new(),
            stage: PrecopyStage::Done,
        };

        if never_installed {
            // The store's listing is the lane's first work and the one call that leaves this
            // process before a module does: it is admitted where a build is admitted, so a store
            // that never answers costs the lane's capacity and not the control thread.
            let list = ListStore {
                programs: Arc::clone(&self.programs),
            };
            match self.modules.admit(ModuleWork::List(list)) {
                Ok(task) => {
                    precopy.stage = PrecopyStage::Modules(ModuleLane::Listing(task));
                    self.precopy = Some(precopy);
                    return Ok(());
                }
                Err(error) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!("admitting the store's listing: {error}"),
                        },
                        reply,
                    );
                }
            }
        }

        tracing::info!(
            generation,
            operators = programs.len(),
            candidates = precopy
                .candidates
                .values()
                .filter(|ordered| ordered.len() > 1)
                .count(),
            devices = self.workers.len(),
            "this process is installing the generation the daemon named"
        );
        let (tasks, unavailable) = match self.admit_installs(programs, &BTreeMap::new()) {
            Ok(admitted) => admitted,
            Err(failure) => return self.abandon_precopy(generation, failure, reply),
        };
        precopy.unavailable = unavailable;
        precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
        self.precopy = Some(precopy);
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
                 has answered: the Program Server, a device or a scheduler is the one to look at"
            );
        }
        // What the stage was waiting on, read before the stage is taken out: a bound that named
        // `no stage` because the driver had already replaced it would say nothing at all, which is
        // the opposite of what the line is for.
        let pending = pending_work(&precopy.stage, &self.workers, &precopy.policies);
        let stage = std::mem::replace(&mut precopy.stage, PrecopyStage::Done);

        match stage {
            PrecopyStage::Modules(ModuleLane::Listing(task)) => self.advance_listing(precopy, task),
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

    /// The store's listing stage: once the Program Server has said what this host built, the
    /// generation's set is resolved from it and one install per device is admitted behind it on
    /// the module lane.
    fn advance_listing(
        &mut self,
        mut precopy: Precopy,
        task: Arc<HostTask<ModuleWork>>,
    ) -> Result<()> {
        let generation = precopy.generation;
        let answer = match task.outcome() {
            // The store has not answered yet: the lane keeps the task and this pass ends
            // with the precopy where it was, as every stage of the precopy does.
            None => {
                precopy.stage = PrecopyStage::Modules(ModuleLane::Listing(task));
                self.precopy = Some(precopy);
                return Ok(());
            }
            Some(Outcome::Success(answer)) => match &*answer {
                ModuleAnswer::Listed(listing) => listing.clone(),
                // This stage holds the listing alone; an install here is a fault in this
                // process, and it is refused by name rather than read as a listing.
                ModuleAnswer::Installed(_) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: "the module lane answered the store's listing with a \
                                     kernel load"
                                .to_owned(),
                        },
                        precopy.reply,
                    );
                }
            },
            // The listing's own refusal already names the read: it is the store's answer,
            // not the store, that failed.
            Some(Outcome::Failed(error)) => {
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Precopy {
                        reason: format!("reading the Program Server's store: {error}"),
                    },
                    precopy.reply,
                );
            }
            Some(Outcome::Cancelled) => {
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Precopy {
                        reason: "the store's listing was cancelled".to_owned(),
                    },
                    precopy.reply,
                );
            }
        };

        // The store has said what it holds: the first generation's set is resolved from it
        // and the installs are admitted on this same lane, behind the listing.
        self.store = Some(answer);
        let named = precopy.programs.clone();
        let (resolved, candidates) = match self.first_generation(&named) {
            Ok(resolved) => resolved,
            Err(failure) => {
                return self.abandon_precopy(generation, failure, precopy.reply);
            }
        };
        tracing::info!(
            generation,
            operators = resolved.len(),
            candidates = candidates
                .values()
                .filter(|ordered| ordered.len() > 1)
                .count(),
            devices = self.workers.len(),
            "this process is installing the generation the daemon named"
        );
        let (tasks, unavailable) = match self.admit_installs(&resolved, &BTreeMap::new()) {
            Ok(admitted) => admitted,
            Err(failure) => {
                return self.abandon_precopy(generation, failure, precopy.reply);
            }
        };
        precopy.programs = resolved;
        precopy.candidates = candidates;
        precopy.unavailable = unavailable;
        precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
        self.precopy = Some(precopy);
        Ok(())
    }

    /// The install stage: every device's kernels loaded build the next generations; a program the
    /// store refused for a shape of this generation moves the walk to the operator's next candidate.
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
                    // This stage holds installs; the listing has its own arm above.
                    ModuleAnswer::Listed(_) => {
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Precopy {
                                reason: "the module lane answered a kernel load with the \
                                         store's listing"
                                    .to_owned(),
                            },
                            precopy.reply,
                        );
                    }
                },
                Some(Outcome::Failed(error)) => {
                    // A program the store refused for a shape of this generation is not
                    // yet a failed generation: the operator may have another program this
                    // host built, and the walk tries each once before it gives up.
                    let refused = precopy
                        .unavailable
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .clone();
                    if let Some(refused) = refused {
                        match self.advance_candidate(&mut precopy, &refused) {
                            Advance::Retry(fresh) => {
                                precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(fresh));
                                self.precopy = Some(precopy);
                                return Ok(());
                            }
                            Advance::Exhausted(reason) | Advance::Failed(reason) => {
                                return self.abandon_precopy(
                                    generation,
                                    UpdateFailure::Precopy { reason },
                                    precopy.reply,
                                );
                            }
                            Advance::Named => {}
                        }
                    }
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!("loading this node's kernels: {error}"),
                        },
                        precopy.reply,
                    );
                }
                Some(Outcome::Cancelled) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: "the module load was cancelled".to_owned(),
                        },
                        precopy.reply,
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
                precopy.reply,
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
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Policy { error },
                    precopy.reply,
                );
            }
            None => {}
        }
        match unfinished(&submissions, "preparing this node's devices") {
            Some(StageFailure::Pending) => {
                precopy.stage = PrecopyStage::Workers(submissions);
                self.precopy = Some(precopy);
                Ok(())
            }
            // A walk that found a kernel missing names the operator and key it needed:
            // that is how the epoch's first generation learns what it must install.
            Some(StageFailure::Failed(reason)) => self.learn_kernel(precopy, &submissions, reason),
            None => self.start_shadow(precopy),
        }
    }

    /// A device preparation that failed: when its walk named a kernel the generation lacks, the
    /// generation learns the operator's descriptor key (borrowing a program from this host's store
    /// for an operator the deployment left unnamed) and its installs are admitted again; any other
    /// failure abandons the precopy with `reason`.
    fn learn_kernel(
        &mut self,
        mut precopy: Precopy,
        submissions: &[Submission],
        reason: String,
    ) -> Result<()> {
        let generation = precopy.generation;
        let Some((operator, key)) = missing_kernel(submissions) else {
            return self.abandon_precopy(
                generation,
                UpdateFailure::Precopy { reason },
                precopy.reply,
            );
        };
        // The walk can name an operator this generation holds no program for:
        // a deployment names the implementations it pins and says nothing about
        // the rest, so the rest come from this host's store - one borrow per
        // operator, and a store that describes none stops the update with a
        // sentence naming the package rather than a key.
        if !precopy.names(&operator) {
            match self.borrow_program(&operator) {
                Ok(Some((slot, program))) => {
                    tracing::info!(
                        %operator,
                        backend = %program.backend,
                        revision = program.revision,
                        generation,
                        "the walk named an operator the deployment left \
                         unnamed: this host's store supplies it"
                    );
                    precopy.programs.insert(slot, program);
                }
                Ok(None) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!(
                                "{operator} has no kernel in this host's \
                                 store and the deployment does not name one"
                            ),
                        },
                        precopy.reply,
                    );
                }
                Err(error) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy { reason: error },
                        precopy.reply,
                    );
                }
            }
        }
        if precopy
            .learned
            .get(&operator)
            .is_some_and(|keys| keys.contains(&key))
        {
            // The walk named a key this generation already installed, so no
            // further attempt can succeed: the loop stops with both names
            // rather than retrying a key it holds.
            let cause = if precopy.programs.is_empty() {
                " - and the daemon named no kernel program for this \
                 generation, so there is nothing to install it from (the \
                 startup switch states the scheduling policy alone)"
            } else {
                ""
            };
            return self.abandon_precopy(
                generation,
                UpdateFailure::Precopy {
                    reason: format!(
                        "{operator} still has no kernel under {key} after it \
                         was installed: the walk and the install disagree{cause}"
                    ),
                },
                precopy.reply,
            );
        }
        tracing::info!(
            %operator,
            %key,
            generation,
            "the walk names a descriptor key this generation learns"
        );
        precopy.learned.entry(operator).or_default().push(key);
        let tasks = match self.admit_installs(&precopy.programs, &precopy.learned) {
            Ok((tasks, unavailable)) => {
                precopy.unavailable = unavailable;
                tasks
            }
            Err(failure) => {
                return self.abandon_precopy(generation, failure, precopy.reply);
            }
        };
        precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
        self.precopy = Some(precopy);
        Ok(())
    }

    /// The prepared generation is complete on every device: the shadow step is asked of every
    /// scheduler when an operator's program changed, and the precopy finishes when none did.
    fn start_shadow(&mut self, mut precopy: Precopy) -> Result<()> {
        let generation = precopy.generation;
        // The precopy's shadow pass pairs a changed operator's kernel with the one serving,
        // in the buckets a resident request occupies, so it is owed only when an operator's
        // program changed. The set is empty on an epoch's first precopy - nothing
        // serves, so no kernel has an old one to be paired against - and asking every
        // scheduler there waits for a step an idle epoch never runs.
        let changed = changed_operators(&precopy.programs, &self.serving_programs);
        if changed.is_empty() {
            tracing::info!(
                generation,
                "no operator's program changed, so this precopy asks for no shadow step"
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
            return self.abandon_precopy(
                generation,
                UpdateFailure::Precopy { reason },
                precopy.reply,
            );
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
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Precopy { reason },
                    precopy.reply,
                );
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
        self.prepared = std::mem::take(&mut precopy.next);
        // What serves after the switchover: what serves now, with what this precopy installed.
        self.prepared_programs = self.serving_programs.clone();
        self.prepared_programs.extend(precopy.programs.clone());
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
                "no operator's program changed, so no kernel had an older one to compare with"
                    .to_owned(),
            );
        }
        self.answer(
            precopy.reply,
            ProgramStatus::Precopied {
                generation,
                uncompared,
            },
        );
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
            let current = Arc::clone(&self.generations[index]);
            next.push(Arc::new(current.replace(generation, entries)));
        }

        // The candidate policy is instantiated on the current snapshot, and
        // every decision it would make is dry-run against the hard constraints
        // before anything switches.
        //
        // Once per precopy, not once per install: the descriptor discovery loop comes back through
        // here with a larger key set, and a policy build on each pass would certify the same
        // program again on the scheduler's blocking thread, minutes at a time, while the update
        // waits for a generation that only needs another kernel.
        let policies = if precopy.policy_requested {
            Vec::new()
        } else {
            match self.request_policy(&precopy.policy) {
                Ok(policies) => policies,
                Err(error) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Policy {
                            error: format!("{error:#}"),
                        },
                        precopy.reply,
                    );
                }
            }
        };

        // Rebind or recapture, as jobs between serving batches.
        let mut submissions = Vec::with_capacity(next.len());
        for (index, built) in next.iter().enumerate() {
            submissions.push(self.workers[index].precopy(Arc::clone(built))?);
        }
        let mut precopy = precopy;
        precopy.next = next;
        if !policies.is_empty() {
            precopy.policies = policies;
            precopy.policy_requested = true;
        }
        precopy.stage = PrecopyStage::Workers(submissions);
        self.precopy = Some(precopy);
        Ok(())
    }

    /// A failed precopy: the new generation is discarded and serving goes on.
    pub(super) fn abandon_precopy(
        &mut self,
        generation: u64,
        failure: UpdateFailure,
        reply: Option<u32>,
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
        self.answer(
            reply,
            ProgramStatus::PrecopyFailed {
                generation,
                failure,
            },
        );
        Ok(())
    }
}
