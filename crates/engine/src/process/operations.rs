//! The operator's operations inside an epoch (`docs/admin.md`): a weight update in place, sleep
//! and wake, the room a profiling session's storage takes from the KV cache, a cache flush, the
//! commit of an expert redistribution, the named operations beside the batches (LoRA adapters,
//! the experts' copies, profiling sessions, the fence probes' switch), and the scheduling
//! limits.
//!
//! Each operation is a job: a function of the parts it reads, which returns once the operation
//! ended and touches nothing of the control thread's. One that changes what a batch reads does so
//! at a batch boundary of every scheduler ([`at_boundary`]): every scheduler is paused, the work
//! runs once none has a batch in flight, and every scheduler resumes, whatever the work did
//! (`specs/scheduler-design.md` §2.4). The control thread runs these jobs one at a time on its
//! operations thread, so two boundaries never interleave; the named operations run beside the
//! batches, on threads of their own.
//!
//! The schedulers are named by their inboxes, in the order of the session's schedulers
//! (`Open::schedulers`); the head is the engine's connection to its host's Storage Server, which
//! states each change of the KV planes' segments (`crate::process::binding`, "The KV planes'
//! memory"). A `Resume` ends a drain as well as a pause, so no job that pauses the schedulers runs
//! while the engine drains.
//!
//! Every operation acts on the ranks of the epoch's peers too, through the weight binding's
//! links (`crate::process::binding`, "Ranks on other hosts"): a peer runs each that reaches it on
//! its own binding ([`operator`]), and the scheduler node's pause orders a boundary operation
//! against the batches of every rank, a peer's included.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::executor::peers::Operate;
use crate::executor::peers::messages::DataOperation;
use crate::process::binding::{Replacement, WeightsUpdated};
use anyhow::{Context as _, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender};
use proto::config::SchedulerConfig;
use proto::request::wire::WeightsInFlight;
use service::lock;
use storage::Key;

use super::epoch;
use crate::memory::client::StorageClient;
use crate::process::binding::{Target, WeightBinding};
use crate::scheduler::{self, Message};
use crate::worker::Worker;

/// How long a release that makes room waits before it asks a region again for units its requests
/// still hold: the requests that hold the tail run on, and the release takes it once they let it
/// go.
const RESIZE_RETRY: Duration = Duration::from_millis(10);

/// The scheduler at `index` of the session's has stopped: its thread ended, on a fault it
/// reported itself (`Output::Failed`) or at its shutdown.
pub(super) fn stopped(index: usize) -> anyhow::Error {
    anyhow!("scheduler {index} has stopped")
}

/// Sends every scheduler the message `message` makes with a reply channel, then waits for every
/// reply, and returns them in the schedulers' order. Every scheduler is asked before any reply is
/// waited for, so they answer at once.
///
/// # Errors
///
/// When a scheduler has stopped before it replied.
pub(super) fn ask<T>(
    schedulers: &[Sender<Message>],
    message: impl Fn(Sender<T>) -> Message,
) -> Result<Vec<T>> {
    let mut replies = Vec::with_capacity(schedulers.len());
    for (index, inbox) in schedulers.iter().enumerate() {
        let (reply, replied) = crossbeam_channel::bounded(1);
        inbox.send(message(reply)).map_err(|_| stopped(index))?;
        replies.push((index, replied));
    }

    replies
        .into_iter()
        .map(|(index, replied)| replied.recv().map_err(|_| stopped(index)))
        .collect()
}

/// The first refusal among the schedulers' replies, naming the scheduler and `what` it refused.
///
/// # Errors
///
/// The first scheduler's refusal.
pub(super) fn every(replies: Vec<Result<(), scheduler::Error>>, what: &str) -> Result<()> {
    for (index, reply) in replies.into_iter().enumerate() {
        reply.map_err(|error| anyhow!("scheduler {index} refused {what}: {error}"))?;
    }
    Ok(())
}

/// Sends `Resume` to every scheduler: each forms batches again from its next step, and one a
/// drain parked admits again. A scheduler that has stopped reported its own end.
pub(super) fn resume(schedulers: &[Sender<Message>]) {
    for inbox in schedulers {
        let _ = inbox.send(Message::Resume);
    }
}

/// Runs `work` at one batch boundary of every scheduler (`specs/scheduler-design.md` §2.4): each
/// is sent `Pause`, the work runs once every one replied that it has no batch in flight, and each
/// is sent `Resume`, whether the work, or a pause, failed or not. The schedulers keep taking
/// their messages while paused, so the work may send them `SetPolicy`, `Flush` or `Resize` and
/// wait for the replies.
///
/// # Errors
///
/// What the work fails with, or a scheduler that stopped before it paused; the work does not
/// run then.
pub(crate) fn at_boundary<T>(
    schedulers: &[Sender<Message>],
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let paused = ask(schedulers, Message::Pause);
    let done = paused.and_then(|_| work());

    resume(schedulers);
    done
}

/// Puts the weights of `target`, another checkpoint of the served model, in place on every rank
/// (`docs/admin.md`, `/admin/update_weights`), and returns each rank's report.
///
/// The model's load first walks the checkpoint's headers alone, and a checkpoint that loads other
/// tensors is refused with the weights as they were. Then, as `in_flight` says:
///
/// - `Keep`: the new weights load beside the batches, and at one boundary they are put in place
///   and the cache is flushed under the new weights' page seeds; the requests in flight go on
///   with the KV cache they reached and commit nothing from there. When the room beside the KV
///   cache does not hold the load, the graph storage is given back for it and the batches stop
///   for as long as it runs; when even that leaves too little room, the update is refused,
///   changing nothing.
/// - `Recompute`: the engine sleeps ([`sleep`]), so that the KV cache's storage holds the load,
///   puts the new weights in place, flushes under their seeds and wakes ([`wake`]); every
///   request in flight is computed again by the daemon's resubmission.
/// - `Abort`: every request the schedulers hold ends, aborted, then as `Recompute`.
///
/// # Errors
///
/// A checkpoint of another model, a load that failed (the ranks serve on as they did), a mapping
/// change that failed part of the way (the weight binding is poisoned, and the epoch must end), or
/// a scheduler that stopped.
pub(crate) fn update_weights(
    schedulers: &[Sender<Message>],
    head: &Mutex<StorageClient>,
    binding: &WeightBinding,
    target: Target,
    in_flight: WeightsInFlight,
) -> Result<Vec<WeightsUpdated>> {
    let checkpoint = target.checkpoint.display().to_string();
    let replacement = binding
        .pair(target, false)
        .with_context(|| format!("pairing the weights of {checkpoint}"))?;

    match in_flight {
        WeightsInFlight::Keep => {
            let bytes = replacement.bytes();
            let rooms = binding.room()?;
            let graphs = binding.graph_bytes()?;

            if rooms.iter().all(|room| room.holds(bytes, 0)) {
                let loaded = binding.load(replacement)?;
                at_boundary(schedulers, || {
                    binding.stage(loaded)?;
                    apply(schedulers, binding)
                })
            } else if rooms.iter().all(|room| room.holds(bytes, graphs)) {
                // The graph storage makes the room: nothing may replay a graph until it is back,
                // and it comes back only once the old weights' storage went.
                at_boundary(schedulers, || {
                    binding.suspend_graphs()?;
                    let applied = binding
                        .load(replacement)
                        .map_err(anyhow::Error::from)
                        .and_then(|loaded| {
                            binding.stage(loaded)?;
                            apply(schedulers, binding)
                        });
                    let restored = binding.restore_graphs();

                    let reports = applied?;
                    restored?;
                    Ok(reports)
                })
            } else {
                binding.discard(replacement);
                bail!(
                    "the weights of {checkpoint} take {bytes} bytes beside the KV cache, and the \
                     devices have {rooms:?} of room with {graphs} bytes of graph storage to give \
                     back: `keep` cannot load them, and `recompute` or `abort` gives the KV \
                     cache's storage to the load"
                );
            }
        }
        WeightsInFlight::Recompute | WeightsInFlight::Abort => {
            if in_flight == WeightsInFlight::Abort {
                let aborted = ask(schedulers, |reply| {
                    Message::Flush(None, WeightsInFlight::Abort, reply)
                })?;
                every(aborted, "aborting the requests it holds")?;
            }

            // The engine wakes whatever happened after the drain: a sleep that stopped part way
            // is undone by the wake, which grants back what the head withdrew.
            let applied = sleep(schedulers, head, binding).and_then(|()| {
                let loaded = binding.load(replacement)?;
                binding.stage(loaded)?;
                apply(schedulers, binding)
            });
            let woken = wake(schedulers, head, binding);

            let reports = applied?;
            woken?;
            Ok(reports)
        }
    }
}

/// Puts the staged weights in place on every rank, and has every scheduler flush its cache and
/// key the pages, encoder outputs and trajectories it stores from here on by the new weights. The
/// requests the schedulers hold continue: under a weight update they commit nothing more.
///
/// What keys them is what the rank start states in each group's `Capacity` for the weights it
/// serves (`crate::process::ranks::capacities`), now for the new ones: the page seed of each cache
/// group, from the ranks' page identities for the new weights (`storage::key::page_seed`); the
/// latent seed, from every rank's identity of its new weights (`storage::key::latent_seed`); and
/// per group whose ranks encode, the identity of their new weights
/// ([`WeightBinding::encoder_identities`]), which each scheduler takes for its groups. So no
/// block, encoder output or trajectory another set of weights computed is read. A weight binding
/// of no rank states no identity, and the seeds stay.
fn apply(schedulers: &[Sender<Message>], binding: &WeightBinding) -> Result<Vec<WeightsUpdated>> {
    let reports = binding.update_weights()?;

    let encoders = if reports.is_empty() {
        Vec::new()
    } else {
        binding.encoder_identities()?
    };
    let seeds = reports.first().map(|report| {
        let pages: Vec<Key> = report
            .identities
            .iter()
            .map(|identity| storage::key::page_seed(identity))
            .collect();
        let latent = storage::key::latent_seed(
            reports
                .iter()
                .filter_map(|report| report.identity.as_deref()),
        );
        (pages, latent, encoders)
    });

    let flushed = ask(schedulers, |reply| {
        Message::Flush(seeds.clone(), WeightsInFlight::Keep, reply)
    })?;
    every(flushed, "flushing the cache under the new weights")?;
    Ok(reports)
}

/// Puts the engine to sleep (`docs/admin.md`, `/admin/sleep`): every scheduler drains at once
/// with its device memory given back, as a drain to another layout does (token requests leave
/// as migratable and are computed again, video requests at their next denoising step with their
/// trajectories stored), the head withdraws every segment of the KV planes, and the ranks release
/// them and suspend their graph pools. The drained schedulers form no batch until [`wake`].
///
/// # Errors
///
/// A scheduler that stopped, the Storage Server's refusal of a withdrawal, or a rank's failure;
/// [`wake`] undoes what was done.
pub(crate) fn sleep(
    schedulers: &[Sender<Message>],
    head: &Mutex<StorageClient>,
    binding: &WeightBinding,
) -> Result<()> {
    let drained = epoch::drain(schedulers, Instant::now(), true)?;
    for (index, replied) in drained.into_iter().enumerate() {
        replied.recv().map_err(|_| stopped(index))?;
    }

    // Every segment leaves its space's ledger before any rank releases it: a segment the ledger
    // still held would be given to a region with no memory behind it.
    {
        let mut head = lock(head);
        let regions = epoch::kv_regions(epoch::session(&head)?.0);
        for index in regions {
            let (_, withdrawn) = epoch::withdraw_from(&mut head, index, 0);
            withdrawn.context("withdrawing the KV cache's segments for the sleep")?;
        }
    }
    binding.sleep()?;
    Ok(())
}

/// Wakes the engine from [`sleep`] (`docs/admin.md`, `/admin/wake_up`): the ranks restore their
/// graph pools, then create the segments of their KV planes their quotas hold beside what they
/// reserve, their warm processes included, as a grow does ([`epoch::create_fitting`]), the head
/// grants them, every scheduler grows its device regions back into them, and admits again.
/// Returns each device region's number with the units it reached.
///
/// # Errors
///
/// A rank's failure, the Storage Server's refusal of a grant, or a scheduler that stopped.
pub(crate) fn wake(
    schedulers: &[Sender<Message>],
    head: &Mutex<StorageClient>,
    binding: &WeightBinding,
) -> Result<Vec<(u8, u32)>> {
    binding
        .restore_graphs()
        .context("restoring the graph pools")?;
    epoch::create_fitting(&mut lock(head), binding, None)?;

    // The regions are grown first, so the first request admitted finds its units.
    let reached = epoch::grow_regions(schedulers, head)?;
    resume(schedulers);
    Ok(reached)
}

/// Gives the KV cache's tail segments back until every device of the engine has the room the
/// device's quota must hold beside everything it reserves: `bytes`, or for a GPU `devices` names
/// by its identity, the bytes stated for it (`ResizeCache { MakeRoom }`, a profiling session's
/// storage). The requests that hold the tail run on, and the release takes it once they let it
/// go; [`epoch::grow`] takes it back.
///
/// For each region of the KV planes, the segments released are the fewest from the tail whose
/// release leaves every device of the region's ranks the room; segment 0 is kept. Every region
/// is decided before any changes.
///
/// # Errors
///
/// When even a release of every segment past the first leaves a device too little room, with
/// nothing changed; a scheduler that stopped, the Storage Server's refusal, or a rank's failure.
pub(crate) fn make_room(
    schedulers: &[Sender<Message>],
    head: &Mutex<StorageClient>,
    binding: &WeightBinding,
    bytes: u64,
    devices: &BTreeMap<[u8; 16], u64>,
) -> Result<()> {
    let needed = |gpu: &[u8; 16]| devices.get(gpu).copied().unwrap_or(bytes);

    // Per region of the KV planes whose release makes room: its index, its number, the
    // scheduler that resizes it, the first segment released and the units the region keeps,
    // those below that segment's.
    let mut releases = Vec::new();
    {
        let head = lock(head);
        let (open, opened) = epoch::session(&head)?;
        for index in epoch::kv_regions(open) {
            let (Some(stated), Some(&(number, _)), Some(table)) = (
                open.regions.get(index),
                opened.regions.get(index),
                opened.segments.get(index),
            ) else {
                continue;
            };
            let rooms = binding.device_rooms(&epoch::groups_of(open, index))?;

            // The release from `first` on gives each device the bytes of the segments there.
            let leaves_room = |first: usize| {
                let units: u64 = table
                    .iter()
                    .skip(first)
                    .map(|&(_, _, units)| u64::from(units))
                    .sum();
                rooms.iter().all(|(gpu, room, unit_bytes)| {
                    room.holds(needed(gpu), units.saturating_mul(*unit_bytes))
                })
            };
            // Segment 0 holds the unit a pooled table names where a row holds no page: it stays.
            let held = table.len();
            let Some(first) = (1..=held.max(1)).rev().find(|first| leaves_room(*first)) else {
                bail!(
                    "a release of every KV cache segment of region {number} past the first leaves \
                     a device too little room for {bytes} bytes (by device {devices:?}): {rooms:?}"
                );
            };
            let Some(&(_, start, _)) = table.get(first) else {
                continue;
            };

            let scheduler = epoch::scheduler_of(opened, number)?;
            let kept = start.saturating_sub(stated.reserved);
            releases.push((index, number, scheduler, first, kept));
        }
    }

    for (index, number, scheduler, first, kept) in releases {
        // The region gives back its units from the segment's first on; the requests that hold
        // some of them run on, and the region is asked again until it reached the units it keeps.
        let inbox = schedulers
            .get(scheduler)
            .ok_or_else(|| stopped(scheduler))?;
        loop {
            let reached = ask(std::slice::from_ref(inbox), |reply| {
                Message::Resize(number, kept, reply)
            })?;
            if reached.first().is_some_and(|reached| *reached <= kept) {
                break;
            }
            std::thread::sleep(RESIZE_RETRY);
        }

        let (from, withdrawn, groups) = {
            let mut head = lock(head);
            let (from, withdrawn) = epoch::withdraw_from(&mut head, index, first);
            (
                from,
                withdrawn,
                epoch::groups_of(epoch::session(&head)?.0, index),
            )
        };
        // What the head withdrew is released whatever stopped the withdrawals.
        binding.release_segments(&groups, u32::try_from(from).unwrap_or(u32::MAX))?;
        withdrawn.context("withdrawing the KV cache's tail segments")?;
    }
    Ok(())
}

/// Flushes the cache (`docs/admin.md`, `/admin/flush_cache`): at one boundary every scheduler
/// evicts every cached block it knows of; the requests it holds keep theirs.
///
/// # Errors
///
/// A scheduler that stopped or refused.
pub(crate) fn flush(schedulers: &[Sender<Message>]) -> Result<()> {
    at_boundary(schedulers, || {
        let flushed = ask(schedulers, |reply| {
            Message::Flush(None, WeightsInFlight::Keep, reply)
        })?;
        every(flushed, "flushing the cache")
    })
}

/// Puts the redistribution of the routed experts the weight binding holds ready in place, at one
/// boundary of every scheduler, and returns each rank's report (`docs/admin.md`, "Expert
/// redistribution"). The numerical weights are the same, so nothing is flushed.
///
/// # Errors
///
/// As [`WeightBinding::commit_experts`], or a scheduler that stopped.
pub(crate) fn commit_experts(
    schedulers: &[Sender<Message>],
    binding: &WeightBinding,
    order: &BTreeMap<String, Vec<u32>>,
) -> Result<Vec<WeightsUpdated>> {
    at_boundary(schedulers, || {
        binding
            .commit_experts(order)
            .context("putting the experts' new order in place")
    })
}

/// Runs the operation the daemon names `name` with the JSON `payload`, beside the batches, and
/// returns each rank's result: an expert redistribution's stages (`uniserve.prepare_experts`,
/// `start_experts`, `experts_ready`, `abort_experts`), a LoRA adapter's load and unload
/// (`load_lora_adapter`, `unload_lora_adapter`), a profiling session's changes
/// (`uniserve.profile_start`, `profile_stop`, `profile_release`) and the fence probes' switch
/// (`uniserve.fence_switch`). Nothing here changes what a batch reads: a redistribution copies
/// into new storage, and its change of the served weights is [`commit_experts`].
///
/// # Errors
///
/// A name this process has no operation of, a payload the operation refuses, or the
/// operation's failure.
pub(crate) fn hook(binding: &Arc<WeightBinding>, name: &str, payload: &str) -> Result<Vec<String>> {
    let results = match name {
        "uniserve.prepare_experts" => {
            let orders: BTreeMap<String, Vec<u32>> =
                serde_json::from_str(payload).context("an expert order")?;
            binding.prepare_experts(orders)?
        }
        "uniserve.start_experts" => binding.start_experts()?,
        "uniserve.experts_ready" => binding.experts_ready()?,
        "uniserve.abort_experts" => binding.abort_experts()?,
        "load_lora_adapter" => binding.load_adapter(payload)?,
        "unload_lora_adapter" => binding.unload_adapter(payload)?,
        "uniserve.profile_start" => binding.start_profile(payload)?,
        "uniserve.profile_stop" => binding.stop_profile(payload)?,
        "uniserve.profile_release" => binding.release_profile(payload)?,
        "uniserve.fence_switch" => fence_switch(payload)?,
        other => bail!("this process has no operation named {other:?}"),
    };

    // The epoch's peers run it on their own ranks, each as this process runs it on its own.
    let mut results = results;
    results.extend(binding.hook_peers(name, payload)?);
    Ok(results)
}

/// The end of a process's own preparation of a generation, as its control thread states it: the
/// generation, and whether the preparation failed, with why.
pub(crate) type Preparation = (u64, std::result::Result<(), String>);

/// What a peer runs each operation of the scheduler node's data plane with
/// (`crate::executor::peers::Upstream::serve`): [`operate`] on `binding` and `workers`, this
/// process's, keeping a weight load's replacement between the operations that pair, load, stage
/// and discard it, and reading the ends of this process's own preparations from `preparations`,
/// which it waits for at most `bound`.
pub(crate) fn operator(
    binding: Arc<WeightBinding>,
    workers: Vec<Arc<Worker>>,
    preparations: Receiver<Preparation>,
    bound: Duration,
) -> Operate {
    let mut held: Option<Replacement> = None;
    Box::new(move |operation| {
        let preparations = (&preparations, bound);
        operate(&binding, &workers, &mut held, preparations, operation)
            .map_err(|error| format!("{error:#}"))
    })
}

/// Runs `operation`, which the scheduler node relayed, on this peer's `binding` and `workers`, and
/// returns the JSON of its value, as the scheduler node's binding reads it. `held` is the weight
/// load's replacement this process keeps from the pairing to the stage or discard that ends it.
///
/// # Errors
///
/// What the operation fails with; a load or a stage with nothing paired.
fn operate(
    binding: &Arc<WeightBinding>,
    workers: &[Arc<Worker>],
    held: &mut Option<Replacement>,
    preparations: (&Receiver<Preparation>, Duration),
    operation: DataOperation,
) -> Result<String> {
    use DataOperation as Operation;

    let reply = match operation {
        Operation::ReleaseSegments { groups, from } => {
            let released = binding.release_segments(&groups, from)?;
            let devices: Vec<u32> = released
                .into_iter()
                .flat_map(|(_, devices)| devices)
                .collect();
            encoded(&devices)
        }
        Operation::CreateSegments { groups, count } => {
            encoded(&binding.create_segments(&groups, count)?)
        }
        Operation::SegmentTables { groups } => encoded(&binding.segment_tables(&groups)?),
        Operation::FittingSegments { groups } => encoded(&binding.fitting_here(&groups)?),
        Operation::DeviceRooms { groups } => encoded(&binding.device_rooms(&groups)?),
        Operation::WarmSegments { groups } => encoded(&binding.warm_here(&groups)?),
        Operation::Room => encoded(&binding.room()?),
        Operation::GraphBytes => encoded(&binding.graph_bytes_here()),
        Operation::Sleep => encoded(&binding.sleep()?),
        Operation::SuspendGraphs => encoded(&binding.suspend_graphs()?),
        Operation::RestoreGraphs => encoded(&binding.restore_graphs()?),
        Operation::SetKernels => encoded(&binding.set_kernels()?),
        Operation::ReserveServing => encoded(&binding.reserve_serving()?),
        Operation::Pair {
            checkpoint,
            experts,
            rearranged,
        } => {
            let target = Target {
                checkpoint: PathBuf::from(checkpoint),
                experts,
            };
            let paired = binding.pair(target, rearranged)?;
            let bytes = paired.bytes();
            if let Some(previous) = held.replace(paired) {
                binding.discard(previous);
            }
            encoded(&bytes)
        }
        Operation::Load => {
            let paired = held.take().context("no weights are paired to load")?;
            *held = Some(binding.load(paired)?);
            encoded(&())
        }
        Operation::Stage => {
            let loaded = held.take().context("no weights are loaded to stage")?;
            binding.stage(loaded)?;
            encoded(&())
        }
        Operation::Discard => {
            if let Some(paired) = held.take() {
                binding.discard(paired);
            }
            encoded(&())
        }
        Operation::UpdateWeights => encoded(&binding.update_weights()?),
        Operation::CommitExperts { order } => encoded(&binding.commit_experts(&order)?),
        Operation::ServingOrder => encoded(&binding.served_here()),
        Operation::Prepared { generation } => {
            let (preparations, bound) = preparations;
            encoded(&prepared(preparations, generation, bound)?)
        }
        Operation::Uncompared => encoded(&super::generation::uncompared(workers)),
        Operation::Hook { name, payload } => encoded(&hook(binding, &name, &payload)?),
    };
    reply.context("encoding the operation's value")
}

/// The JSON of an operation's value.
fn encoded(value: &impl serde::Serialize) -> serde_json::Result<String> {
    serde_json::to_string(value)
}

/// The end of this process's own preparation of generation `generation`, which its control
/// thread states on `preparations` as each ends. An earlier generation's end, whose preparation
/// was abandoned, is passed over.
///
/// # Errors
///
/// A preparation that failed, or that did not end within `bound`.
fn prepared(preparations: &Receiver<Preparation>, generation: u64, bound: Duration) -> Result<()> {
    let deadline = Instant::now() + bound;
    loop {
        match preparations.recv_deadline(deadline) {
            Ok((ended, outcome)) if ended == generation => {
                return outcome.map_err(|reason| {
                    anyhow!(
                        "this process's preparation of generation {generation} failed: {reason}"
                    )
                });
            }
            Ok(_) => {}
            Err(_) => bail!(
                "this process's preparation of generation {generation} did not end within {} s",
                bound.as_secs()
            ),
        }
    }
}

/// Writes the fence probes' device switch (`docs/profiling.md`): `{"enabled": bool, "off":
/// [patterns]}`. The range markers read it in device storage, and writing it waits for nothing.
/// A process without range markers says so.
fn fence_switch(payload: &str) -> Result<Vec<String>> {
    let payload: serde_json::Value = serde_json::from_str(payload).context("fence_switch")?;
    let enabled = payload["enabled"]
        .as_bool()
        .context("fence_switch states no enabled")?;
    let off: Vec<String> = match payload.get("off") {
        Some(off) if !off.is_null() => {
            serde_json::from_value(off.clone()).context("fence_switch off")?
        }
        _ => Vec::new(),
    };

    let switched = crate::profiling::markers::Markers::installed()
        .map(|markers| markers.switch(enabled, &off))
        .transpose()?;
    Ok(vec![
        serde_json::json!({
            "probes": switched.is_some(),
            "enabled": enabled,
            "off": switched.unwrap_or_default(),
        })
        .to_string(),
    ])
}

/// Applies `config`'s scheduling limits in place on every scheduler (`Configure`,
/// `specs/updates.md` §1): each takes them from its next step on, or refuses a limit above what
/// its workers were started for and keeps its own.
///
/// # Errors
///
/// A scheduler that stopped or refused.
pub(crate) fn configure(schedulers: &[Sender<Message>], config: &SchedulerConfig) -> Result<()> {
    let configured = ask(schedulers, |reply| {
        Message::Configure(config.clone(), reply)
    })?;
    every(configured, "the scheduling limits")
}

#[cfg(test)]
pub(super) mod tests;
