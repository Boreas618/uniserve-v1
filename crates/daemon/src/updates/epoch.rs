//! The epoch arm of a configuration change:
//! what `apply` calls when `epoch::layer` answers [`Layer::Epoch`].
//!
//! The arm decides nothing the other layers own. It answers `Ok(None)` for a
//! change of the host layer, of the programs alone, or of nothing, so `apply`
//! falls through to the path that handles those; only an epoch change builds
//! the target's [`EpochRecord`] and starts a [`Switch`].
//!
//! The deployment as data states everything the switch needs except four
//! things, which [`EpochChange`] carries: the two epoch numbers the records
//! are kept under, the update the switch belongs to, the worker group, the
//! command the main process is started with, and the two generations — the
//! serving epoch's and the one the target's children are spawned at, both
//! taken from the epoch layer's one allocator by `apply` (§2.5, M904), never
//! computed here.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::epoch::{Epoch, EpochRecord, Epochs, Layer, Status, Update, layer};
use crate::leader::UpdateError;
use crate::options::ServerArgs;
use crate::updates::change::Applied;
use crate::updates::switch::{Node, Roster, Start, Switch};

/// A configuration a revert marked unavailable, and why: a target that failed
/// on probation keeps its build from being tried again until an operator says
/// otherwise (§7.4).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Unavailable {
    pub digest: String,
    /// The epoch that executed it and was rolled back.
    pub epoch: u64,
    pub reason: String,
    /// Seconds since the Unix epoch.
    pub at: f64,
}

/// What `apply` hands the epoch arm that the two deployments cannot state.
pub(crate) struct EpochChange {
    /// The epoch that serves now, whose record the target's keeps as its
    /// previous generation.
    pub(crate) previous: u64,
    /// The epoch being built; its record is kept under this number.
    pub(crate) target: u64,
    /// The update the switch belongs to: its number, target and failure.
    pub(crate) update: Update,
    /// The worker group every main process of the deployment is in.
    pub(crate) group: String,
    /// How the target's main processes are started.
    pub(crate) start: Start,
    /// The epoch the serving epoch executes.
    pub(crate) previous_epoch: u64,
    /// The serving epoch's launch generation.
    pub(crate) previous_generation: u64,
    /// The launch generation the target's children are spawned at, from
    /// `Epochs::next_generation`: above every live generation of the group
    /// and rank, the serving epoch's standby included.
    pub(crate) spawn_generation: u64,
    /// What the previous epoch keeps through the trial
    /// (`update_config.previous_epoch`).
    pub(crate) keeps: crate::options::PreviousEpoch,
}

/// Runs the epoch layer of a change into `running`, or says the change is
/// another layer's.
///
/// Returns `Ok(None)` when `epoch::layer` answers `Host`, `Program` or
/// `None`, so the caller takes its other path; `Ok(Some(Applied::Planned))`
/// under `dry_run`, having started nothing and written nothing; and
/// `Ok(Some(Applied::Epoch))` with the switch's first report once the
/// target's main processes are asked to start, the switch itself kept in
/// `running` for the caller to advance, commit or revert.
///
/// `running` must be empty: one change runs at a time (`UpdateError::
/// InProgress` says another is under way).
pub(crate) fn epoch_arm(
    previous: &Epoch,
    target: &Epoch,
    change: EpochChange,
    epochs: &Arc<Epochs>,
    roster: &mut impl Roster,
    running: &mut Option<Switch>,
    dry_run: bool,
) -> Result<Option<Applied>, UpdateError> {
    match layer(previous, target) {
        Layer::Epoch => {}
        Layer::Host | Layer::Program | Layer::None => return Ok(None),
    }
    if running.is_some() {
        return Err(UpdateError::InProgress);
    }
    let nodes = nodes(&target.config)?;
    let requests = target
        .config
        .request_socket()
        .to_string_lossy()
        .into_owned();
    let record = record(previous, target, change.previous, epochs)?;

    if dry_run {
        // Nothing is written and nothing starts: the plan names what the
        // switch would do and the pause it would cause (§7.4).
        return Ok(Some(Applied::Planned(json!({
            "change": "epoch",
            "epoch": change.target,
            "executable": change.start.executable,
            "generation": change.spawn_generation,
            "nodes": nodes.iter().map(|node| node.host.clone()).collect::<Vec<_>>(),
            "pause": "from the first Promote to the last Drained of the previous epoch",
        }))));
    }

    let mut switch = Switch::new(
        Arc::clone(epochs),
        change.update,
        change.target,
        record,
        (change.previous, change.previous_epoch),
        change.previous_generation,
        change.spawn_generation,
        change.group,
        nodes,
        change.start,
        requests,
        Vec::new(),
        change.keeps,
    );
    switch.start(roster);
    let report = report(&switch);
    *running = Some(switch);
    Ok(Some(Applied::Epoch(report)))
}

/// The target epoch's record: what builds it and which programs fill it,
/// keeping the serving epoch's programs as the previous generation a revert
/// and a restart use (§7.3).
///
/// The serving record must exist: an epoch the daemon cannot describe cannot
/// be drained or resumed, and naming it afresh would invent a generation.
fn record(
    previous: &Epoch,
    target: &Epoch,
    previous_epoch: u64,
    epochs: &Epochs,
) -> Result<EpochRecord, UpdateError> {
    let serving = epochs.get(previous_epoch).ok_or_else(|| {
        UpdateError::Refused(format!(
            "epoch {previous_epoch} has no record: the serving epoch cannot be drained or resumed"
        ))
    })?;
    let mut record = serving.clone();
    record.config = target.engine.clone();
    record.kernels = target.programs.kernels.clone();
    record.policy = target.programs.policy.clone();
    record.previous_kernels = serving.kernels;
    record.previous_policy = Some(serving.policy);
    record.status = Status::Precopy;
    // The target's own generation is set only by a child's report (M904);
    // the previous epoch's names another process.
    record.generation = None;
    // The record is the serving one with the target's build and programs, so
    // that a field added to `EpochRecord` needs no change here. The target's
    // own launch generation is not part of the record yet; when it is added,
    // it is cleared here rather than inherited — a generation is set only from
    // a child's own report (M904), and the serving epoch's would name a
    // process of another epoch.
    //
    // The previous deployment is not used otherwise: it names the epoch, and
    // `layer` has already said the change is an epoch one.
    let _ = previous;
    Ok(record)
}

/// The nodes the epoch's main processes run on: this host at rank 0, and the
/// rank layout's hosts for the other ranks.
fn nodes(config: &ServerArgs) -> Result<Vec<Node>, UpdateError> {
    let count = config.nnodes.get();
    let mut nodes = Vec::with_capacity(count as usize);
    nodes.push(Node {
        host: config.host_identity.clone(),
        rank: 0,
    });
    for rank in 1..count {
        let index = (rank - 1) as usize;
        let host = config
            .device_config
            .worker_hosts
            .get(index)
            .ok_or_else(|| {
                UpdateError::Refused(format!(
                    "the deployment has {count} nodes but names no host for rank {rank}"
                ))
            })?;
        nodes.push(Node {
            host: host.clone(),
            rank,
        });
    }
    Ok(nodes)
}

/// The switch as the operator sees it right after it started.
pub(crate) fn report(switch: &Switch) -> Value {
    json!({
        "change": "epoch",
        "phase": switch.phase().name(),
        "generation": switch.generation(),
        "pause_ms": switch.pause().map(|pause| pause.as_millis()),
        "build": switch.record().config.executable,
        "failure": switch.failure().map(|failure| json!({
            "kind": failure.kind,
            "reason": failure.reason,
            "device": failure.device,
        })),
        "previous_epoch_kept": switch.previous_kept(),
        "host_refused": switch.host_refused(),
        "moved_to_host_bytes": switch.moved_bytes(),
    })
}
