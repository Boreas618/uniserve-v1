//! A serving engine's generation update (`specs/updates.md` §3.1, `docs/admin.md` "Updates"):
//! the jobs that make the next generation of kernel and policy modules ready beside the serving
//! one, switch to it at one batch boundary, and end its trial in a commit or a rollback.
//!
//! The engine numbers its generations and keeps the packages each was built from; the executors
//! hold each device's generation of kernels (`ops::Generation`): the one that serves and, during
//! an update, the one prepared beside it or kept through the trial, with the graphs of each. The
//! schedulers hold the policy's instance, and the one a switch replaced until the commit.
//!
//! - **Prepare**: [`named`] reads the packages the daemon's `Prepare` names; [`kernels`] selects and installs the kernels of every key the devices launch and
//!   has every executor rebind or recapture its graphs; [`policy`] resolves the policy package and
//!   instantiates it once against the ceilings, so that a policy that faults fails the
//!   preparation and not the switch; [`shadow`] compares each changed kernel with the serving one
//!   on the batches the schedulers serve. Serving is untouched, and [`discard`] gives back what a
//!   preparation built.
//! - **Switch**: [`switch`], at one boundary of every scheduler.
//! - **Trial**: [`commit`] releases the kept generation after its last batch; [`rollback`] serves
//!   the kept generation again from one boundary.
//!
//! The route instance of an engine of several schedulers switches, commits and rolls back on the
//! control thread, which holds it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use build_server::{Client, Target};
use crossbeam_channel::{RecvTimeoutError, Sender};
use proto::module::{ModuleSlot, PackageRevision, Packages, PhaseKind};
use scheduling_guest::limits::{Limits, Parameters};
use scheduling_guest::view::Snapshot;
use worker::completion::{Outcome, Submission};
use worker::{Executor, lock};

use super::operations::{ask, at_boundary, every, stopped};
use crate::data_plane::DataPlane;
use crate::executor::Shadow;
use crate::modules::load::{self, install, select};
use crate::modules::select::{Routes, replaced_operators};
use crate::scheduler::{Message, Policy};

/// The packages the daemon's `Prepare` names: each operator's package under its slot, and the
/// scheduling policy's package. The Protobuf carries a package's arguments as the TOML text of
/// each value, which are read back as the values they were; it carries no module's bytes, which
/// the Build Server resolves.
///
/// # Errors
///
/// A kernel or the policy that names no package, a phase that is no phase kind, or an argument
/// that is not a TOML value.
pub(crate) fn named(prepare: proto::process::Prepare) -> Result<(Packages, PackageRevision)> {
    let mut packages = Packages::new();
    for kernel in prepare.kernels {
        let package = kernel
            .package
            .with_context(|| format!("the preparation's kernel {} names no package", kernel.op))?;
        let slot = ModuleSlot {
            op: kernel.op,
            phase: kernel.phase.map(phase_of).transpose()?.flatten(),
            part: kernel.part,
        };
        packages.insert(slot, package_of(package)?);
    }

    let policy = prepare.policy.context("the preparation names no policy")?;
    Ok((packages, package_of(policy)?))
}

/// A package as the daemon's Protobuf states it, its arguments read back as TOML values.
fn package_of(package: proto::process::PackageRevision) -> Result<PackageRevision> {
    let mut arguments = toml::Table::new();
    for (name, value) in package.arguments {
        let value = value.parse::<toml::Value>().map_err(|_| {
            anyhow!(
                "the argument {name} of {} is not a TOML value",
                package.backend
            )
        })?;
        arguments.insert(name, value);
    }

    Ok(PackageRevision {
        backend: package.backend,
        revision: package.revision,
        arguments,
    })
}

/// A kernel's phase as the Protobuf states it; `None` for the unspecified value, which means the
/// package fills every phase of its operator.
fn phase_of(phase: i32) -> Result<Option<PhaseKind>> {
    use proto::process::PhaseKind as Wire;

    Ok(match Wire::try_from(phase) {
        Ok(Wire::Prefill) => Some(PhaseKind::Prefill),
        Ok(Wire::Decode) => Some(PhaseKind::Decode),
        Ok(Wire::Encode) => Some(PhaseKind::Encode),
        Ok(Wire::Denoise) => Some(PhaseKind::Denoise),
        Ok(Wire::Unspecified) => None,
        Err(_) => return Err(anyhow!("phase {phase} is not a phase kind")),
    })
}

/// Makes generation `generation` of kernels ready beside the serving one on every executor and
/// returns the packages it is built from (`Prepare`, `specs/updates.md` §3.1).
///
/// Every descriptor key the devices launch is asked of the Build Server once, with the packages
/// the daemon names (`named`), and the server selects the package of each key. The operators
/// whose selected packages differ from `serving`'s (the packages the serving generation was built
/// from) are installed on every device, in its own context; every other operator keeps its
/// kernels, and with them the graphs that launch them. Each executor then prepares the device's
/// generation beside the serving one: it rebinds the graphs whose launch table is unchanged and
/// recaptures the others, between the batches it serves. An executor of no device installs
/// nothing.
///
/// # Errors
///
/// The Build Server's refusal, an install that failed, or a device that did not prepare the
/// generation: what was prepared is given back on every device, and serving is untouched.
pub(crate) fn kernels(
    executors: &[Arc<Executor>],
    build: &(Arc<Mutex<Client>>, Vec<Target>),
    named: &Packages,
    serving: &Packages,
    generation: u64,
) -> Result<Packages> {
    let (client, targets) = build;
    let held = serving_kernels(executors)?;

    // An update changes implementations and never the keys: the serving generations' keys are
    // every key the devices' forwards launch.
    let mut keys: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for kernels in &held {
        for (operator, key, _) in kernels.iter() {
            keys.entry(operator.to_owned())
                .or_default()
                .insert(key.to_owned());
        }
    }

    let mut client = lock(client);
    let (selected, routes) = select(&mut client, targets, named, &keys)
        .context("selecting the generation's packages at the Build Server")?;
    let replaced = replaced_operators(&selected, serving);
    let packages: Packages = selected
        .into_iter()
        .filter(|(slot, _)| replaced.contains(&slot.op))
        .collect();
    let routes: Routes = routes
        .into_iter()
        .filter(|(slot, _)| replaced.contains(&slot.op))
        .collect();

    // Every device's kernels are installed, each in its own context, before any executor is
    // given its generation, so a failed install leaves nothing to give back.
    let mut installed = Vec::with_capacity(executors.len());
    for executor in executors {
        installed.push(match executor.device() {
            Some(device) => {
                install(&mut client, device, &packages, &routes).with_context(|| {
                    format!("installing the generation's kernels on device {}", device.0)
                })?
            }
            None => Vec::new(),
        });
    }
    drop(client);

    // One generation per device: a kernel belongs to the context that loaded it.
    let submitted: Vec<worker::Result<Submission>> = executors
        .iter()
        .zip(&held)
        .zip(installed)
        .map(|((executor, kernels), entries)| {
            executor.prepare(Arc::new(kernels.replace(generation, entries)))
        })
        .collect();
    if let Err(failure) = ended(submitted, &format!("preparing generation {generation}")) {
        if let Err(error) = discard(&[], executors, generation) {
            tracing::warn!(%error, generation, "a preparation that failed was not given back");
        }
        return Err(failure);
    }

    // What serves after the switch: what serves now, with every operator installed replaced
    // whole, since its slots may divide it otherwise than the serving ones.
    let mut prepared: Packages = serving
        .iter()
        .filter(|(slot, _)| !replaced.contains(&slot.op))
        .map(|(slot, package)| (slot.clone(), package.clone()))
        .collect();
    prepared.extend(packages);
    Ok(prepared)
}

/// The generation of kernels each executor's device serves, in the executors' order: what the
/// next one is built from.
///
/// # Errors
///
/// An executor that closed, or that holds no runner.
fn serving_kernels(executors: &[Arc<Executor>]) -> Result<Vec<Arc<ops::Generation>>> {
    executors
        .iter()
        .map(|executor| {
            let kernels = executor.call("reading the generation the rank serves", |runners| {
                runners
                    .first()
                    .map(|runner| runner.graphs().kernels())
                    .ok_or(worker::Error::State("the rank holds no runner"))
            })?;
            Ok(kernels)
        })
        .collect()
}

/// Resolves the policy package `package` at the Build Server and returns its module with the
/// arguments it was given (`Prepare`'s policy half). The module is instantiated once against
/// `ceilings`, the limits the executors hold, and dropped: a policy that faults at its start or
/// declares limits outside the ceilings fails the preparation, with serving untouched, rather
/// than the switch (`specs/scheduler-design.md` §3.9).
///
/// # Errors
///
/// The Build Server's refusal, or the policy's fault.
pub(crate) fn policy(
    build: &Arc<Mutex<Client>>,
    package: &PackageRevision,
    ceilings: &Limits,
) -> Result<(Arc<wasmtime::Module>, Parameters)> {
    let named = format!(
        "the policy package {}@{}",
        package.backend, package.revision
    );
    let (module, parameters) =
        load::policy(&mut lock(build), package).with_context(|| format!("resolving {named}"))?;

    Policy::new(
        Arc::clone(&module),
        parameters.clone(),
        Snapshot::default(),
        ceilings,
    )
    .map_err(|fault| anyhow!("{named}: {fault}"))?;
    Ok((module, parameters))
}

/// Compares each changed kernel of generation `generation`, which every executor has prepared,
/// with the serving one on the batches the schedulers serve: a pass of every scheduler's
/// [`Shadow`] runs each batch of a group and shape it has not shadowed as a shadow batch of the
/// generation, whose paired forwards compare the pairs within their tolerances. Returns the
/// operators the executors could not compare, each with the reason, without repeats; empty states
/// that every replaced operator a shadow batch reached was compared.
///
/// A pass answers once a batch of its scheduler ran, so an engine whose running requests run no
/// step within `bound` (twice the stall bound) compares nothing: the update is abandoned, naming
/// the operators that changed (`changed`), rather than passed without the comparison.
///
/// # Errors
///
/// A pair beyond its tolerance, a shadow batch that failed, or a pass not answered within
/// `bound`; every pass is ended then.
pub(crate) fn shadow(
    shadows: &[Arc<Mutex<Shadow>>],
    executors: &[Arc<Executor>],
    generation: u64,
    changed: &[String],
    bound: Duration,
) -> Result<Vec<String>> {
    let mut replies = Vec::with_capacity(shadows.len());
    for shadow in shadows {
        let (reply, replied) = crossbeam_channel::bounded(1);
        lock(shadow).open(generation, reply);
        replies.push(replied);
    }

    let deadline = Instant::now() + bound;
    let mut failure = None;
    for (index, replied) in replies.iter().enumerate() {
        let failed = match replied.recv_deadline(deadline) {
            Ok(Ok(())) => continue,
            Ok(Err(error)) => anyhow!("scheduler {index}: {error}"),
            Err(RecvTimeoutError::Timeout) => anyhow!(
                "the shadow pass of the changed operators ({}) waited {}s: no running request \
                 ran a step on scheduler {index}, so nothing was compared, and the update is \
                 abandoned rather than passed without the comparison",
                changed.join(", "),
                bound.as_secs()
            ),
            Err(RecvTimeoutError::Disconnected) => {
                anyhow!("the shadow pass of scheduler {index} was ended unanswered")
            }
        };
        failure = Some(failed);
        break;
    }
    if let Some(failure) = failure {
        for shadow in shadows {
            lock(shadow).end();
        }
        return Err(failure);
    }

    let mut uncompared: Vec<String> = Vec::new();
    for executor in executors {
        for entry in executor.uncompared() {
            if !uncompared.contains(&entry) {
                uncompared.push(entry);
            }
        }
    }
    Ok(uncompared)
}

/// Switches every scheduler and every executor to generation `generation` at one batch boundary
/// (`Switch`): every executor serves the prepared generation from its next batch, keeping the
/// one it replaces; every rank's cache and state launch their own work from it; every scheduler
/// instantiates `policy`, the generation's module and arguments, on its state, keeping the one it
/// replaces; and the batches formed from here on carry `generation` (`served`, which the
/// schedulers read). The control thread's route instance switches after.
///
/// # Errors
///
/// A device that did not switch, or a scheduler that refused the policy: the switch is rolled
/// back, so every scheduler serves its previous policy and every executor the generation that
/// served (the prepared one is given back), and `served` is unchanged; or a rank whose cache
/// and state did not take the kernels, with the switch made.
pub(crate) fn switch(
    schedulers: &[Sender<Message>],
    executors: &[Arc<Executor>],
    data: &DataPlane,
    served: &AtomicU64,
    generation: u64,
    policy: &(Arc<wasmtime::Module>, Parameters),
) -> Result<()> {
    at_boundary(schedulers, || {
        let what = format!("switching to generation {generation}");
        let switched = ended(
            executors
                .iter()
                .map(|executor| executor.switch(generation))
                .collect(),
            &what,
        );

        // The policy follows the devices; a scheduler that refuses it serves its own.
        let mut took = Vec::new();
        let set = switched.and_then(|()| {
            let (module, parameters) = policy;
            let replies = ask(schedulers, |reply| {
                Message::SetPolicy(Arc::clone(module), parameters.clone(), reply)
            })?;
            took = schedulers
                .iter()
                .zip(&replies)
                .filter(|(_, reply)| reply.is_ok())
                .map(|(inbox, _)| inbox.clone())
                .collect();
            every(replies, "the generation's policy")
        });

        if let Err(failure) = set {
            let back = ask(&took, Message::RollbackPolicy)
                .and_then(|replies| every(replies, "the previous policy"));
            if let Err(error) = back {
                tracing::error!(%error, generation, "a scheduler did not serve its policy again");
            }
            // A rollback returns a device that switched to the generation it kept, and gives
            // the prepared generation back on one that did not.
            let rolled = ended(
                executors
                    .iter()
                    .map(|executor| executor.rollback(generation))
                    .collect(),
                &format!("rolling generation {generation} back"),
            );
            if let Err(error) = rolled {
                tracing::error!(%error, generation, "a device did not return to its generation");
            }
            return Err(failure.context(what));
        }

        served.store(generation, Ordering::Release);
        data.set_kernels()
            .with_context(|| format!("the ranks' memories under generation {generation}"))?;
        Ok(())
    })
}

/// Ends the trial of generation `generation` in its favour (`Commit`): every executor releases the
/// generation it kept, behind the batches it holds and so after the kept generation's last one,
/// and every scheduler drops the policy it kept. The control thread's route instance commits too.
///
/// # Errors
///
/// A device that did not commit, or a scheduler that stopped.
pub(crate) fn commit(
    schedulers: &[Sender<Message>],
    executors: &[Arc<Executor>],
    generation: u64,
) -> Result<()> {
    ended(
        executors
            .iter()
            .map(|executor| executor.commit(generation))
            .collect(),
        &format!("committing generation {generation}"),
    )?;

    for (index, inbox) in schedulers.iter().enumerate() {
        inbox
            .send(Message::CommitPolicy)
            .map_err(|_| stopped(index))?;
    }
    Ok(())
}

/// Ends the trial of generation `generation` against it (`Rollback`): at one batch boundary every
/// executor serves `kept`, the generation the trial kept, again and releases `generation`'s
/// graphs, the batches formed from here on carry `kept`, every rank's cache and state launch
/// their own work from it, and every scheduler serves the policy it kept again. The control
/// thread's route instance rolls back too.
///
/// # Errors
///
/// A device that did not roll back, which has no consistent generation, so the epoch must end;
/// a rank whose memories did not take the kernels; a scheduler that stopped or whose previous
/// policy faults again. Each part is done whatever another did.
pub(crate) fn rollback(
    schedulers: &[Sender<Message>],
    executors: &[Arc<Executor>],
    data: &DataPlane,
    served: &AtomicU64,
    generation: u64,
    kept: u64,
) -> Result<()> {
    at_boundary(schedulers, || {
        let devices = ended(
            executors
                .iter()
                .map(|executor| executor.rollback(generation))
                .collect(),
            &format!("rolling generation {generation} back"),
        );
        served.store(kept, Ordering::Release);

        let memories = data
            .set_kernels()
            .with_context(|| format!("the ranks' memories under generation {kept}"));
        let policies = ask(schedulers, Message::RollbackPolicy)
            .and_then(|replies| every(replies, "the previous policy"));
        devices.and(memories).and(policies)
    })
}

/// Gives back what was prepared of generation `generation`, which never served (a `Rollback`
/// before the switch, or a preparation that failed or was abandoned): every scheduler's shadow
/// pass ends unanswered, and every executor releases the generation's graphs and the kernels only
/// they held. The policy was only tried, and nothing of it is held.
///
/// # Errors
///
/// A device that did not release the generation.
pub(crate) fn discard(
    shadows: &[Arc<Mutex<Shadow>>],
    executors: &[Arc<Executor>],
    generation: u64,
) -> Result<()> {
    for shadow in shadows {
        lock(shadow).end();
    }

    ended(
        executors
            .iter()
            .map(|executor| executor.rollback(generation))
            .collect(),
        &format!("discarding generation {generation}"),
    )
}

/// Waits for every submission, which the caller handed every executor before it waits for any,
/// in the executors' order, and returns the first failure once every one ended, naming `what`
/// and the executor.
fn ended(submitted: Vec<worker::Result<Submission>>, what: &str) -> Result<()> {
    let mut failure = None;
    for (rank, submission) in submitted.into_iter().enumerate() {
        let outcome = submission.map(|submission| submission.wait());
        let failed = match outcome {
            Ok(Outcome::Success(_)) => continue,
            Ok(Outcome::Failed(fault)) => anyhow!("{what} on executor {rank}: {fault}"),
            Ok(Outcome::Cancelled) => {
                anyhow!("{what} on executor {rank}: the executor closed before it ran")
            }
            Err(refused) => anyhow!("{what} on executor {rank}: {refused}"),
        };
        failure.get_or_insert(failed);
    }
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests;
