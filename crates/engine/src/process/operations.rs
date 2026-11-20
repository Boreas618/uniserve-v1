//! The operator's operations inside an epoch (`docs/admin.md`): sleep and wake, the room a
//! profiling session's storage takes from the KV cache, a cache flush, the commit of an expert
//! redistribution, the operations every rank runs beside the batches (LoRA adapters, the experts'
//! copies, profiling sessions, the fence probes' switch: [`on_ranks`]), and the scheduling
//! limits.
//!
//! Each operation is a job: a function of the parts it reads, which returns once the operation
//! ended and touches nothing of the control thread's. One that changes what a batch reads does so
//! at a batch boundary of every scheduler ([`at_boundary`]): every scheduler is paused, the work
//! runs once none has a batch in flight, and every scheduler resumes, whatever the work did
//! (`specs/scheduler-design.md` §2.4). The control thread runs these jobs one at a time on its
//! operations thread, so two boundaries never interleave; the operations beside the batches run
//! on threads of their own.
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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::executor::peers::Operate;
use crate::executor::peers::messages::{DataOperation, FencesSwitched, RankOperation, RankReports};
use crate::process::binding::WeightsUpdated;
use anyhow::{Context as _, Result, anyhow, bail};
use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, Sender};
use proto::config::Tuning;

use service::lock;

use super::epoch;
use crate::memory::client::StorageClient;
use crate::process::binding::WeightBinding;
use crate::scheduler::{self, Message};
use crate::worker::Worker;

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

/// Puts the engine to sleep (`docs/admin.md`, `/admin/sleep`): every scheduler drains at once
/// with its device memory given back, as a drain to another layout does (token requests leave
/// as migratable and are computed again, video requests at the denoising step boundary their
/// calls in flight end at with their trajectories stored, and one whose steps are all scheduled
/// runs to its end), the head withdraws every segment of the KV planes, and the ranks release
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
    let drained = epoch::drain(schedulers, Some(Instant::now()), true)?;
    for (index, replied) in drained.into_iter().enumerate() {
        replied.recv().map_err(|_| stopped(index))?;
    }

    // Every segment of every device region, the KV planes' and the state units', leaves its
    // space's ledger before any rank releases it: a segment the ledger still held would be given
    // to a region with no memory behind it.
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
    epoch::create_fitting(&mut lock(head), binding, None, None)?;

    // The regions are grown first, so the first request admitted finds its units.
    let reached = epoch::grow_regions(schedulers, head)?;
    resume(schedulers);
    Ok(reached)
}

/// Gives the KV cache's tail segments back until every device of the engine has the room the
/// device's quota must hold beside everything it reserves: `bytes`, or for a GPU `devices` names
/// by its identity, the bytes stated for it (`ResizeCache { MakeRoom }`, a profiling session's
/// storage). The requests that hold the tail run on, and the release takes it once they let it
/// go; [`epoch::grow`] takes it back. With `demote`, as a generation update's room asks, each
/// region's scheduler first stores to the host tier the cached blocks its shrink evicts that have
/// no host copy (`Message::Demote`), so the prefixes the yield takes from the device stay
/// reusable from host memory.
///
/// For each region of the KV planes, the segments released are the fewest from the tail whose
/// release leaves every device of the region's ranks the room; segment 0 is kept. Every region
/// is decided before any changes. A state region gives nothing back here: its checkpoints are
/// the request slots' (`specs/engine-worker-port.md` §1.6).
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
    demote: bool,
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
            if epoch::is_state_region(open, index) {
                continue;
            }
            let (Some(stated), Some(&(number, _)), Some(table)) = (
                open.regions.get(index),
                opened.regions.get(index),
                opened.segments.get(index),
            ) else {
                continue;
            };
            let rooms = binding.device_rooms(&epoch::groups_of(open, index), number)?;

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
        // some of them run on, and the scheduler answers once the region reached the units it
        // keeps, which it does as they let them go.
        let inbox = schedulers
            .get(scheduler)
            .ok_or_else(|| stopped(scheduler))?;
        if demote {
            let demoted = ask(std::slice::from_ref(inbox), |reply| {
                Message::Demote(number, kept, reply)
            })?;
            tracing::info!(
                region = number,
                kept,
                demoted = demoted.first().copied().unwrap_or(0),
                "the KV cache demoted the blocks its yield evicts to the host tier"
            );
        }
        let reached = ask(std::slice::from_ref(inbox), |reply| {
            Message::Bound(number, kept, reply)
        })?;
        if let Some(&reached) = reached.first()
            && reached > kept
        {
            bail!(
                "region {number} holds {reached} units, past the {kept} a release of its tail \
                 keeps: a drain ended its bound before it reached them"
            );
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
        binding.release_segments(&groups, number, u32::try_from(from).unwrap_or(u32::MAX))?;
        withdrawn.context("withdrawing the KV cache's tail segments")?;
    }
    Ok(())
}

/// Lowers every rank's share of its GPU to `fractions`'s, by the rank's number
/// (`ResizeCache { Fraction }`, `mem_fraction_static` lowered): each GPU's cache gives back its
/// tail segments, the blocks the shrink evicts demoted to the host tier first, until the GPU's
/// usage fits the lower limit ([`make_room`]), and every rank then reserves under it. A share
/// that does not fall gives nothing back.
///
/// # Errors
///
/// As [`make_room`], with nothing changed, when the cache cannot give back enough; the quota's
/// refusal of the lower limit.
pub(crate) fn lower_fractions(
    schedulers: &[Sender<Message>],
    head: &Mutex<StorageClient>,
    binding: &WeightBinding,
    fractions: &BTreeMap<u32, f64>,
) -> Result<()> {
    let lowered = binding.lowered_limits(fractions)?;
    if !lowered.is_empty() {
        make_room(schedulers, head, binding, 0, &lowered, true)?;
    }
    binding
        .set_fractions(fractions)
        .context("reserving under the lower shares of the GPUs")
}

/// Flushes the cache (`docs/admin.md`, `/admin/flush_cache`): at one boundary every scheduler
/// evicts every cached block it knows of; the requests it holds keep theirs.
///
/// # Errors
///
/// A scheduler that stopped.
pub(crate) fn flush(schedulers: &[Sender<Message>]) -> Result<()> {
    at_boundary(schedulers, || ask(schedulers, Message::Flush).map(drop))
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

/// Runs `operation`, which the daemon commands on every rank, beside the batches, on this
/// process's ranks and then on every peer's, each of which runs it on its own as this process
/// does ([`operate`]), and returns every rank's reports, this process's first. Nothing here
/// changes what a batch reads: a redistribution copies into new storage, and its change of the
/// served weights is [`commit_experts`].
///
/// # Errors
///
/// The operation's failure on a rank of this process or of a peer.
pub(crate) fn on_ranks(
    binding: &Arc<WeightBinding>,
    operation: &RankOperation,
) -> Result<RankReports> {
    // Every process runs the operation at once, the peers' share on a thread of its own while
    // this process's ranks run theirs: an operation that lasts, as an expert redistribution's
    // copies do, takes as long as its slowest process.
    let (here, peers) = std::thread::scope(|scope| {
        let peers = scope.spawn(|| binding.on_peers(operation));
        (here(binding, operation), peers.join())
    });
    let mut reports = here?;
    let peers = peers.map_err(|_| anyhow!("relaying a rank operation to the peers panicked"))??;
    for peer in peers {
        reports.append(peer).map_err(|error| anyhow!(error))?;
    }
    Ok(reports)
}

/// Runs `operation` on this process's own ranks and returns their reports.
///
/// # Errors
///
/// The operation's failure.
fn here(binding: &Arc<WeightBinding>, operation: &RankOperation) -> Result<RankReports> {
    Ok(match operation {
        RankOperation::PrepareExperts { orders } => {
            RankReports::ExpertAllocations(binding.prepare_experts(orders)?)
        }
        RankOperation::StartExperts => {
            binding.start_experts()?;
            RankReports::None
        }
        RankOperation::AbortExperts => {
            binding.abort_experts()?;
            RankReports::None
        }
        RankOperation::LoadAdapter { slot, path } => {
            RankReports::Adapters(binding.load_adapter(*slot, path)?)
        }
        RankOperation::UnloadAdapter { slot } => {
            RankReports::Adapters(binding.unload_adapter(*slot))
        }
        // The step values are no rank's: each process's control thread takes them from its
        // snapshot, which the scheduler node stores at `Tune` and a peer as it is relayed
        // ([`operate`]).
        RankOperation::Tune(_) => RankReports::None,
        RankOperation::StartProfile(start) => {
            RankReports::ProfileStarts(binding.start_profile(start)?)
        }
        RankOperation::StopProfile { session, reason } => {
            RankReports::ProfileStops(binding.stop_profile(*session, reason)?)
        }
        RankOperation::ReleaseProfile {
            session,
            collector,
            reason,
        } => RankReports::ProfileReleases(binding.release_profile(*session, *collector, reason)?),
        RankOperation::SwitchFences {
            enabled,
            off,
            kernels,
        } => RankReports::Fences(vec![switch_fences(*enabled, off, *kernels)?]),
        RankOperation::ReadSpans { since } => RankReports::Spans(vec![read_spans(*since)]),
    })
}

/// The spans this process keeps in its rings (`docs/profiling.md`) that ended at or after
/// `since` (nanoseconds since the Unix epoch): its control thread's, its schedulers', its
/// workers' and the timed kernels of its devices, which the range markers name by their sites.
fn read_spans(since: u64) -> Vec<service::span::Span> {
    let markers = crate::profiling::markers::Markers::installed();
    let mut spans = service::span::spans(&service::span::records(), |site| {
        markers.and_then(|markers| markers.kernel(site))
    });
    spans.retain(|span| span.end >= since);
    spans
}

/// The end of a process's own preparation of a generation, as its control thread states it: the
/// generation, and whether the preparation failed, with why.
pub(crate) type Preparation = (u64, std::result::Result<(), String>);

/// What a peer runs each operation of the scheduler node's data plane with
/// (`crate::executor::peers::Upstream::serve`): [`operate`] on `binding` and `workers`, this
/// process's, reading the ends of this process's own preparations from `preparations`, which it
/// waits for at most the comparison bound of `tuning`, the peer's step values, which a relayed
/// `Tune` replaces. `config` is the epoch's engine configuration, whose bounds the graph
/// options of a measured generation are resolved against.
pub(crate) fn operator(
    (binding, workers): (Arc<WeightBinding>, Vec<Arc<Worker>>),
    preparations: Receiver<Preparation>,
    tuning: Arc<ArcSwap<Tuning>>,
    config: proto::config::EngineConfig,
) -> Operate {
    Box::new(move |operation| {
        let preparations = (&preparations, &*tuning);
        operate((&binding, &workers, &config), preparations, operation)
            .map_err(|error| format!("{error:#}"))
    })
}

/// Runs `operation`, which the scheduler node relayed, on this peer's `binding` and `workers`, and
/// returns its value in the peer link's encoding, as the scheduler node's binding reads it.
///
/// # Errors
///
/// What the operation fails with.
fn operate(
    (binding, workers, config): (
        &Arc<WeightBinding>,
        &[Arc<Worker>],
        &proto::config::EngineConfig,
    ),
    (preparations, tuning): (&Receiver<Preparation>, &ArcSwap<Tuning>),
    operation: DataOperation,
) -> Result<Vec<u8>> {
    use DataOperation as Operation;

    let reply = match operation {
        Operation::ReleaseSegments {
            groups,
            region,
            from,
        } => {
            let released = binding.release_segments(&groups, region, from)?;
            let devices: Vec<u32> = released
                .into_iter()
                .flat_map(|(_, devices)| devices)
                .collect();
            encoded(&devices)
        }
        Operation::CreateSegments {
            groups,
            region,
            count,
        } => encoded(&binding.create_segments(&groups, region, count)?),
        Operation::SegmentTables { groups, region } => {
            encoded(&binding.segment_tables(&groups, region)?)
        }
        Operation::FittingSegments { groups, region } => {
            encoded(&binding.fitting_here(&groups, region)?)
        }
        Operation::DeviceRooms { groups, region } => {
            encoded(&binding.device_rooms(&groups, region)?)
        }
        Operation::WarmSegments { groups, region } => encoded(&binding.warm_here(&groups, region)?),
        Operation::WarmRooms { regions } => encoded(&binding.warm_rooms_here(&regions)?),
        Operation::UpdateRoom { operators, options } => {
            let operators = operators.into_iter().collect();
            let shapes = match &options {
                Some(options) => super::generation::measure(
                    workers,
                    binding,
                    &super::shapes::runner_options(config, options)?,
                )?,
                None => super::generation::Shapes::new(),
            };
            encoded(&super::generation::update_room_here(
                workers, binding, &operators, &shapes,
            )?)
        }
        Operation::LoweredLimits { fractions } => encoded(&binding.lowered_here(&fractions)?),
        Operation::SetFractions { fractions } => encoded(&binding.set_fractions_here(&fractions)?),
        Operation::Sleep => encoded(&binding.sleep()?),
        Operation::SuspendGraphs => encoded(&binding.suspend_graphs()?),
        Operation::RestoreGraphs => encoded(&binding.restore_graphs()?),
        Operation::SetKernels => encoded(&binding.set_kernels()?),
        Operation::ReserveServing => encoded(&binding.reserve_serving()?),
        Operation::CommitExperts { order } => encoded(&binding.commit_experts(&order)?),
        Operation::ServingOrder => encoded(&binding.served_here()),
        Operation::Prepared { generation } => {
            // Within the comparison bound of the step values this peer holds now.
            let bound = tuning.load().compare_timeout;
            encoded(&prepared(preparations, generation, bound)?)
        }
        Operation::Uncompared => encoded(&super::generation::uncompared(workers)),
        // A peer runs the operation on its own ranks alone: the scheduler node asks every peer.
        // The scheduler node's step values become this peer's snapshot, which its control
        // thread takes at its next turn (`MainProcess::retune`).
        Operation::Rank(RankOperation::Tune(values)) => {
            tuning.store(Arc::new(*values));
            encoded(&RankReports::None)
        }
        Operation::Rank(operation) => encoded(&here(binding, &operation)?),
    };
    reply.context("encoding the operation's value")
}

/// An operation's value in the peer link's own encoding (MessagePack).
fn encoded(value: &impl serde::Serialize) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec(value)
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

/// Writes the fence probes' device switch (`docs/profiling.md`): on when `enabled`, with the
/// probed module classes `off` names switched off whatever `enabled` says, and the kernel probes
/// on while both `enabled` and `kernels` are. The range markers read it in device storage, and
/// writing it waits for nothing. A process without range markers says so.
fn switch_fences(enabled: bool, off: &[String], kernels: bool) -> Result<FencesSwitched> {
    let switched = crate::profiling::markers::Markers::installed()
        .map(|markers| markers.switch(enabled, off, kernels))
        .transpose()?;
    Ok(FencesSwitched {
        probes: switched.is_some(),
        enabled,
        off: switched.unwrap_or_default(),
    })
}

#[cfg(test)]
pub(super) mod tests;
