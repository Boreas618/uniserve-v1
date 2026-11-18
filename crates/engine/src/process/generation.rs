//! A serving engine's generation update (`specs/updates.md` §3.1, `docs/admin.md` "Updates"):
//! the jobs that make the next generation of kernel and policy modules ready beside the serving
//! one, switch to it at one batch boundary, and end its trial in a commit or a rollback.
//!
//! The engine numbers its generations and keeps the packages each was built from; the workers
//! hold each device's generation of kernels (`ops::Generation`): the one that serves and, during
//! an update, the one prepared beside it or kept through the trial, with the graphs of each. The
//! schedulers hold the policy's instance, and the one a switch replaced until the commit.
//!
//! - **Prepare**: [`named`] reads the packages the daemon's `Prepare` names; [`kernels`] selects
//!   and installs the kernels of every key the devices launch and has every worker hand the
//!   buckets whose kernels changed to its capture thread, which captures them while the worker
//!   serves; [`policy`] resolves the policy package and
//!   instantiates it once against the ceilings, so that a policy that faults fails the
//!   preparation and not the switch; [`compare`] compares each changed kernel with the serving one
//!   on the batches the schedulers serve ([`shadow`]) and collects what no rank could compare.
//!   Serving is untouched, and [`discard`] gives back what a preparation built.
//! - **Switch**: [`switch`]: the policy and the batches' stamps at one boundary of every
//!   scheduler, then each bucket of the kernels as soon as every rank installed it, then the
//!   kernels whole; the trial begins once the last bucket switched.
//! - **Trial**: [`commit`] releases the kept generation after its last batch; [`rollback`] serves
//!   the kept generation again from one boundary.
//!
//! The ranks of the epoch's peers switch, commit and roll back with this process's, in the order
//! of every group's jobs, through the node's route (`Route::switch`); each peer prepares from its
//! own daemon's command and its own host's Build Server, and a shadow pass waits for every
//! peer's preparation before its shadow batches reach them (`WeightBinding::prepared`); what a
//! peer's ranks could not compare is asked once the pass ended, so it covers the shadow batches
//! that reached them (`WeightBinding::uncompared_by_peers`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::worker::completion::{Outcome, Submission};
use anyhow::{Context as _, Result, anyhow};
use build_server::{Client, Target};
use crossbeam_channel::{RecvTimeoutError, Sender};
use proto::config::PolicyBudget;
use proto::module::{PackageRevision, Packages};
use scheduling::limits::{Limits, Parameters};
use scheduling::view::Snapshot;
use service::lock;

use crate::batch::Generations;
use crate::worker::Worker;

use super::operations::{ask, at_boundary, every, stopped};
use crate::executor::{Route, Shadow};
use crate::modules::load::{self, install, select};
use crate::modules::select::{Routes, replaced_operators};
use crate::process::binding::{UpdateRoom, WeightBinding};
use crate::scheduler::Message;
use crate::scheduler::policy::Policy;

/// The packages the daemon's `Prepare` names: each operator's package under its slot, and the
/// scheduling policy's package, each argument with the type the configuration gave it
/// (`proto::module`'s wire form). It carries no module's bytes, which the Build Server resolves.
///
/// # Errors
///
/// A kernel or the policy that names no package, a phase that is no phase kind, or an argument
/// that states no value.
pub(crate) fn named(prepare: proto::process::Prepare) -> Result<(Packages, PackageRevision)> {
    let packages = prepare
        .kernels
        .into_iter()
        .map(proto::module::kernel_of)
        .collect::<std::result::Result<Packages, _>>()
        .context("the preparation's kernels")?;
    let policy = prepare.policy.context("the preparation names no policy")?;
    let policy = PackageRevision::try_from(policy).context("the preparation's policy")?;
    Ok((packages, policy))
}

/// Makes generation `generation` of kernels ready beside the serving one on every worker and
/// returns the packages it is built from (`Prepare`, `specs/updates.md` §3.1).
///
/// Every descriptor key the devices launch is asked of the Build Server once, with the packages
/// the daemon names (`named`), and the server selects the package of each key. The operators
/// whose selected packages differ from `serving`'s (the packages the serving generation was built
/// from) are installed on every device, in its own context; every other operator keeps its
/// kernels, and with them the graphs that launch them. Each worker then begins the device's
/// generation beside the serving one and hands its buckets to its capture thread, which
/// captures again those whose kernels changed while the worker serves ([`Worker::prepare`]);
/// this returns once every worker began. A worker of no device installs nothing. When no
/// operator's package changed, the generation changes no kernel: no worker prepares anything,
/// and the generation's switch takes no graph.
///
/// Before any worker begins, `room` is given the operators whose kernels change, so the device
/// memory the update takes beside the serving graphs is made ([`update_room`]).
///
/// # Errors
///
/// The Build Server's refusal, an install that failed, the room that could not be made, or a
/// device that did not prepare the generation: what was prepared is given back on every device,
/// and serving is untouched.
pub(crate) fn kernels(
    workers: &[Arc<Worker>],
    build: &(Arc<Mutex<Client>>, Vec<Target>),
    (named, serving): (&Packages, &Packages),
    generation: u64,
    room: &dyn Fn(&BTreeSet<String>) -> Result<()>,
) -> Result<Packages> {
    let (client, targets) = build;
    let held = serving_kernels(workers)?;
    let keys = launched_keys(&held);

    let mut client = lock(client);
    let (selected, routes) = select(&mut client, targets, named, &keys)
        .context("selecting the generation's packages at the Build Server")?;
    let replaced = replaced_operators(&selected, serving);
    if replaced.is_empty() {
        return Ok(serving.clone());
    }
    let packages: Packages = selected
        .into_iter()
        .filter(|(slot, _)| replaced.contains(&slot.op))
        .collect();
    let routes: Routes = routes
        .into_iter()
        .filter(|(slot, _)| replaced.contains(&slot.op))
        .collect();

    // Every device's kernels are installed, each in its own context, before any worker is
    // given its generation, so a failed install leaves nothing to give back.
    let mut installed = Vec::with_capacity(workers.len());
    for worker in workers {
        installed.push(match worker.device() {
            Some(device) => {
                install(&mut client, device, &packages, &routes).with_context(|| {
                    format!("installing the generation's kernels on device {}", device.0)
                })?
            }
            None => Vec::new(),
        });
    }
    drop(client);
    room(&replaced).context("making the room the generation's graphs take")?;

    // One generation per device: a kernel belongs to the context that loaded it. Beginning a
    // preparation runs no forward, so every worker begins at once.
    let submitted = held
        .iter()
        .zip(installed)
        .zip(workers)
        .map(|((kernels, entries), worker)| {
            worker.prepare(Arc::new(kernels.replace(generation, entries)))
        })
        .collect();
    if let Err(failure) = ended(submitted, &format!("preparing generation {generation}")) {
        if let Err(error) = discard(&[], workers, generation) {
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

/// Every descriptor key the devices' forwards launch, by operator: the keys of the generations
/// `held` serve. An update changes implementations and never the keys.
fn launched_keys(held: &[Arc<ops::Generation>]) -> BTreeMap<String, BTreeSet<String>> {
    let mut keys: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for kernels in held {
        for (operator, key, _) in kernels.iter() {
            keys.entry(operator.to_owned())
                .or_default()
                .insert(key.to_owned());
        }
    }
    keys
}

/// What a generation update to the packages `named` would take on every rank's device, this
/// process's ranks then every peer's ([`update_room`]), measured without installing anything:
/// the Build Server selects the packages of every key the devices launch as the update's
/// preparation would ([`kernels`]), and the operators whose packages differ from `serving`'s
/// are those whose graphs it captures again. None when it replaces no operator. What a
/// validation of a change reports before the change begins (`MeasureRoom`).
///
/// # Errors
///
/// The Build Server's refusal, a worker that closed, or a rank's quota or device that cannot be
/// read, here or on a peer.
pub(crate) fn room(
    workers: &[Arc<Worker>],
    build: &(Arc<Mutex<Client>>, Vec<Target>),
    (named, serving): (&Packages, &Packages),
    binding: &WeightBinding,
) -> Result<Vec<UpdateRoom>> {
    let (client, targets) = build;
    let keys = launched_keys(&serving_kernels(workers)?);
    let (selected, _) = select(&mut lock(client), targets, named, &keys)
        .context("selecting the generation's packages at the Build Server")?;
    let replaced = replaced_operators(&selected, serving);
    if replaced.is_empty() {
        return Ok(Vec::new());
    }
    update_room(workers, binding, &replaced)
}

/// The generation of kernels each worker's device serves, in the workers' order: what the
/// next one is built from.
///
/// # Errors
///
/// A worker that closed, or that holds no runner.
fn serving_kernels(workers: &[Arc<Worker>]) -> Result<Vec<Arc<ops::Generation>>> {
    workers
        .iter()
        .map(|worker| {
            let kernels = worker.call("reading the generation the rank serves", |runners| {
                runners
                    .first()
                    .map(|runner| runner.graphs().kernels())
                    .ok_or(crate::worker::Error::State("the rank holds no runner"))
            })?;
            Ok(kernels)
        })
        .collect()
}

/// Resolves the policy package `package` at the Build Server and returns its module with the
/// arguments it was given (`Prepare`'s policy half). The module is instantiated once against
/// `ceilings`, the limits the workers hold, under the deployment's `budget`, and dropped: a policy
/// that faults at its start or
/// declares limits outside the ceilings fails the preparation, with serving untouched, rather
/// than the switch (`specs/scheduler-design.md` §3.9).
///
/// # Errors
///
/// The Build Server's refusal, or the policy's fault.
pub(crate) fn policy(
    build: &Arc<Mutex<Client>>,
    package: &PackageRevision,
    (ceilings, budget): (&Limits, &PolicyBudget),
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
        budget,
    )
    .map_err(|fault| anyhow!("{named}: {fault}"))?;
    Ok((module, parameters))
}

/// Compares the kernels of generation `generation` that changed (`changed`, the operators whose
/// package changed) with the serving ones on every rank, this process's and its peers': waits
/// until every peer's own preparation of the generation ended, runs the shadow pass when an
/// operator changed ([`shadow`]), and returns what the ranks could not compare once the pass
/// ended, this process's workers' then every peer's in rank order, without repeats. Nothing is
/// uncompared when no operator changed.
///
/// # Errors
///
/// A peer whose preparation failed or did not end within its bound, the shadow pass's failure,
/// or a peer's failure to answer.
pub(crate) fn compare(
    shadows: &[Arc<Mutex<Shadow>>],
    workers: &[Arc<Worker>],
    binding: &WeightBinding,
    generation: u64,
    changed: &[String],
    bound: Duration,
) -> Result<Vec<String>> {
    // No shadow batch of the generation reaches a peer's ranks before they hold it.
    binding.prepared(generation)?;
    if changed.is_empty() {
        return Ok(Vec::new());
    }

    shadow(shadows, generation, changed, bound)?;
    let mut uncompared = self::uncompared(workers);
    for entry in binding.uncompared_by_peers()? {
        if !uncompared.contains(&entry) {
            uncompared.push(entry);
        }
    }
    Ok(uncompared)
}

/// What `workers` have no comparison for in the generation update in flight, each with the
/// reason, in the workers' order and without repeats ([`Worker::uncompared`]).
pub(crate) fn uncompared(workers: &[Arc<Worker>]) -> Vec<String> {
    let mut uncompared: Vec<String> = Vec::new();
    for worker in workers {
        for entry in worker.uncompared() {
            if !uncompared.contains(&entry) {
                uncompared.push(entry);
            }
        }
    }
    uncompared
}

/// Compares each changed kernel of generation `generation`, which every rank has prepared, with
/// the serving one on the batches the schedulers serve: a pass of every scheduler's
/// [`Shadow`] runs each batch of a group and shape it has not shadowed as a shadow batch of the
/// generation, whose paired forwards compare the pairs within their tolerances. What a rank could
/// not compare its worker states once the pass ended ([`uncompared`]).
///
/// A pass answers once a batch of its scheduler ran, so an engine whose running requests run no
/// step within `bound` (`worker_config.compare_timeout`) compares nothing: the update is
/// abandoned, naming the operators that changed (`changed`), rather than passed without the
/// comparison.
///
/// # Errors
///
/// A pair beyond its tolerance, a shadow batch that failed, or a pass not answered within
/// `bound`; every pass is ended then.
pub(crate) fn shadow(
    shadows: &[Arc<Mutex<Shadow>>],
    generation: u64,
    changed: &[String],
    bound: Duration,
) -> Result<()> {
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
    Ok(())
}

/// What a generation update that replaces the kernels of `operators` asks of every rank's
/// device until its commit, this process's ranks then every peer's, in rank order: the device
/// bytes its graphs and rehearsals take beside the serving graphs, the room the device has, and
/// the KV units the rank's cache gives up while the update holds both generations' graphs when
/// the room does not hold them (`binding::UpdateRoom`). What a validation of the update states
/// before it begins; the KV policy demotes those units' pages to host memory and the ranks give
/// their segments back (`WeightBinding::release_segments`) before the update's captures, and
/// create them again once it committed or rolled back (`WeightBinding::create_segments`).
///
/// # Errors
///
/// A worker that closed, or a rank's quota or device that cannot be read, here or on a peer.
pub(crate) fn update_room(
    workers: &[Arc<Worker>],
    binding: &WeightBinding,
    operators: &BTreeSet<String>,
) -> Result<Vec<UpdateRoom>> {
    let extra = update_bytes(workers, operators)?;
    Ok(binding.update_rooms(operators, |device| {
        extra.get(&device.0).copied().unwrap_or(0)
    })?)
}

/// [`update_room`] of this process's ranks alone: what a peer answers.
///
/// # Errors
///
/// As [`update_room`].
pub(crate) fn update_room_here(
    workers: &[Arc<Worker>],
    binding: &WeightBinding,
    operators: &BTreeSet<String>,
) -> Result<Vec<UpdateRoom>> {
    let extra = update_bytes(workers, operators)?;
    Ok(binding.update_rooms_here(|device| extra.get(&device.0).copied().unwrap_or(0))?)
}

/// The bytes a generation update that replaces `operators` takes on each device beside what
/// serves, by CUDA ordinal, over the workers on it.
fn update_bytes(
    workers: &[Arc<Worker>],
    operators: &BTreeSet<String>,
) -> Result<BTreeMap<i32, u64>> {
    let mut extra: BTreeMap<i32, u64> = BTreeMap::new();
    for worker in workers {
        let Some(device) = worker.device() else {
            continue;
        };
        let bytes = worker.update_bytes(operators.clone())?;
        *extra.entry(device.0).or_default() += bytes;
    }
    Ok(extra)
}

/// The buckets one switch job moves to the generation, at most: a switch waits for the
/// readiness of this many buckets on every rank, then hands them to every rank as one job.
const SWITCHED_TOGETHER: usize = 16;

/// Switches every scheduler and every worker to generation `generation` (`Switch`).
///
/// At one batch boundary every scheduler instantiates `policy`, the generation's module and
/// arguments, on its state, keeping the one it replaces, and the batches formed from there on
/// carry `generation` as their policy and kernel stamps (`served`). When the generation changes
/// kernels on this process's workers, each bucket then switches to its new graph as soon as every
/// rank of every group installed it (`Worker::ready`, through `route`, the peers' ranks
/// included), in the order the first worker's capture thread takes them, a few buckets per job
/// that every rank takes between the same two batches (`Route::switch_buckets`); a bucket that
/// keeps its graph switches with them. Once every bucket switched, the workers switch to the
/// generation whole at one boundary: their eager forwards launch its kernels, every rank's cache
/// and state launch their own work from it, and its trial begins. The control thread's route
/// instance switches after.
///
/// # Errors
///
/// A device that did not switch or whose capture failed, or a scheduler that refused the
/// policy: the switch is rolled back, so every scheduler serves its previous policy and every
/// worker the generation that served (the prepared one is given back), and the stamps are the
/// previous generation's; or a rank whose cache and state did not take the kernels, with the
/// switch made.
pub(crate) fn switch(
    schedulers: &[Sender<Message>],
    route: Option<&Route>,
    workers: &[Arc<Worker>],
    binding: &WeightBinding,
    served: &Generations,
    generation: u64,
    policy: &(Arc<wasmtime::Module>, Parameters),
) -> Result<()> {
    let kept = served.policy();
    let set = at_boundary(schedulers, || {
        let (module, parameters) = policy;
        let replies = ask(schedulers, |reply| {
            Message::SetPolicy(Arc::clone(module), parameters.clone(), reply)
        })?;
        let took: Vec<Sender<Message>> = schedulers
            .iter()
            .zip(&replies)
            .filter(|(_, reply)| reply.is_ok())
            .map(|(inbox, _)| inbox.clone())
            .collect();

        // A scheduler that refuses the policy serves its own, and so do the others.
        if let Err(failure) = every(replies, "the generation's policy") {
            let back = ask(&took, Message::RollbackPolicy)
                .and_then(|replies| every(replies, "the previous policy"));
            if let Err(error) = back {
                tracing::error!(%error, generation, "a scheduler did not serve its policy again");
            }
            return Err(failure);
        }

        served.set_policy(generation);
        served.set_kernels(generation);
        Ok(())
    });
    if let Err(failure) = set {
        // The prepared generation is given back on every rank, which none switched to.
        let rolled = ended(
            on_ranks(route, |route| route.rollback(generation)),
            &format!("rolling generation {generation} back"),
        );
        if let Err(error) = rolled {
            tracing::error!(%error, generation, "a device did not give the generation back");
        }
        return Err(failure.context(format!("switching to generation {generation}")));
    }

    // The kernels follow bucket by bucket, then whole; a failure there returns every scheduler
    // and every rank to what served before the switch.
    let kernels = switch_kernels(route, workers, binding, generation);
    if let Err(failure) = kernels {
        let back = at_boundary(schedulers, || {
            let devices = ended(
                on_ranks(route, |route| route.rollback(generation)),
                &format!("rolling generation {generation} back"),
            );
            served.set_policy(kept);
            served.set_kernels(kept);
            let policies = ask(schedulers, Message::RollbackPolicy)
                .and_then(|replies| every(replies, "the previous policy"));
            devices.and(policies)
        });
        if let Err(error) = back {
            tracing::error!(%error, generation, "a switch that failed was not rolled back");
        }
        return Err(failure.context(format!("switching to generation {generation}")));
    }
    Ok(())
}

/// The kernels' half of [`switch`]: each bucket as every rank installed it, then the generation
/// whole. Nothing when no worker of this process prepared the generation, which then changes no
/// kernel.
fn switch_kernels(
    route: Option<&Route>,
    workers: &[Arc<Worker>],
    binding: &WeightBinding,
    generation: u64,
) -> Result<()> {
    let Some(route) = route else {
        return Ok(());
    };
    let Some(order) = workers.iter().find_map(|worker| worker.order(generation)) else {
        return Ok(());
    };

    let what = format!("switching the buckets of generation {generation}");
    for chunk in order.chunks(SWITCHED_TOGETHER) {
        let keys = chunk.to_vec();
        ended(vec![route.ready(generation, Some(keys.clone()))], &what)?;
        ended(vec![route.switch_buckets(generation, keys)], &what)?;
    }

    // The buckets of ranks this process's first worker does not hold (another group's, a
    // peer's) are installed too before the generation serves whole.
    ended(vec![route.ready(generation, None)], &what)?;
    ended(
        vec![route.switch(generation)],
        &format!("switching to generation {generation}"),
    )?;
    binding
        .set_kernels()
        .with_context(|| format!("the ranks' memories under generation {generation}"))
}

/// Ends the trial of generation `generation` in its favour (`Commit`): every worker releases the
/// generation it kept, behind the batches it holds and so after the kept generation's last one,
/// and every scheduler drops the policy it kept.
///
/// # Errors
///
/// A device that did not commit, or a scheduler that stopped.
pub(crate) fn commit(
    schedulers: &[Sender<Message>],
    route: Option<&Route>,
    workers: &[Arc<Worker>],
    generation: u64,
) -> Result<()> {
    // A generation that changed no kernel holds nothing on the ranks to commit.
    if workers
        .iter()
        .any(|worker| worker.order(generation).is_some())
    {
        ended(
            on_ranks(route, |route| route.commit(generation)),
            &format!("committing generation {generation}"),
        )?;
    }

    for (index, inbox) in schedulers.iter().enumerate() {
        inbox
            .send(Message::CommitPolicy)
            .map_err(|_| stopped(index))?;
    }
    Ok(())
}

/// Ends the trial of generation `generation` against it (`Rollback`): at one batch boundary every
/// worker serves `kept`, the generation the trial kept, again and releases `generation`'s
/// graphs, the batches formed from here on carry `kept`, every rank's cache and state launch
/// their own work from it, and every scheduler serves the policy it kept again.
///
/// # Errors
///
/// A device that did not roll back, which has no consistent generation, so the epoch must end;
/// a rank whose memories did not take the kernels; a scheduler that stopped or whose previous
/// policy faults again. Each part is done whatever another did.
pub(crate) fn rollback(
    schedulers: &[Sender<Message>],
    route: Option<&Route>,
    binding: &WeightBinding,
    served: &Generations,
    generation: u64,
    kept: u64,
) -> Result<()> {
    at_boundary(schedulers, || {
        let devices = ended(
            on_ranks(route, |route| route.rollback(generation)),
            &format!("rolling generation {generation} back"),
        );
        served.set_kernels(kept);
        served.set_policy(kept);

        let memories = binding
            .set_kernels()
            .with_context(|| format!("the ranks' memories under generation {kept}"));
        let policies = ask(schedulers, Message::RollbackPolicy)
            .and_then(|replies| every(replies, "the previous policy"));
        devices.and(memories).and(policies)
    })
}

/// Gives back what was prepared of generation `generation`, which never served (a `Rollback`
/// before the switch, or a preparation that failed or was abandoned): every scheduler's shadow
/// pass ends unanswered, and every worker releases the generation's graphs and the kernels only
/// they held. The policy was only tried, and nothing of it is held.
///
/// # Errors
///
/// A device that did not release the generation.
pub(crate) fn discard(
    shadows: &[Arc<Mutex<Shadow>>],
    workers: &[Arc<Worker>],
    generation: u64,
) -> Result<()> {
    for shadow in shadows {
        lock(shadow).end();
    }

    ended(
        workers
            .iter()
            .map(|worker| worker.rollback(generation))
            .collect(),
        &format!("discarding generation {generation}"),
    )
}

/// The submission of `job` on every rank of `route`, this node's and its peers' (`Route::switch`);
/// none for an engine of no rank.
fn on_ranks(
    route: Option<&Route>,
    job: impl FnOnce(&Route) -> crate::worker::Result<Submission>,
) -> Vec<crate::worker::Result<Submission>> {
    route.map(job).into_iter().collect()
}

/// Waits for every submission, which the caller handed every worker before it waits for any,
/// in the workers' order, and returns the first failure once every one ended, naming `what`
/// and the worker.
fn ended(submitted: Vec<crate::worker::Result<Submission>>, what: &str) -> Result<()> {
    let mut failure = None;
    for (rank, submission) in submitted.into_iter().enumerate() {
        let outcome = submission.map(|submission| submission.wait());
        let failed = match outcome {
            Ok(Outcome::Success(_)) => continue,
            Ok(Outcome::Failed(fault)) => anyhow!("{what} on worker {rank}: {fault}"),
            Ok(Outcome::Cancelled) => {
                anyhow!("{what} on worker {rank}: the worker closed before it ran")
            }
            Err(refused) => anyhow!("{what} on worker {rank}: {refused}"),
        };
        failure.get_or_insert(failed);
    }
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests;
