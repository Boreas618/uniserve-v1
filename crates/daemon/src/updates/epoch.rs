//! The epoch arm of a configuration change (`specs/worker.md` §7.3, §7.4):
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

use crate::epoch::{Deployment, EpochRecord, Epochs, Layer, Status, Update, layer};
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
    /// The world that executed it and was rolled back.
    pub world: u64,
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
    /// The world the serving epoch executes.
    pub(crate) previous_world: u64,
    /// The serving epoch's launch generation.
    pub(crate) previous_generation: u64,
    /// The launch generation the target's children are spawned at, from
    /// `Epochs::next_generation`: above every live generation of the group
    /// and rank, the serving epoch's standby included.
    pub(crate) spawn_generation: u64,
    /// What the previous epoch keeps through the trial
    /// (`update_config.previous_world`).
    pub(crate) keeps: crate::options::PreviousWorld,
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
    previous: &Deployment,
    target: &Deployment,
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
    let mut record = record(previous, target, change.previous, epochs)?;
    // The world allocation this epoch executes: the epoch number today, so
    // the switch's claims and the record name the same world.
    record.world = change.target;

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
        (change.previous, change.previous_world),
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
    previous: &Deployment,
    target: &Deployment,
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
        "previous_world_kept": switch.previous_kept(),
        "host_refused": switch.host_refused(),
        "moved_to_host_bytes": switch.moved_bytes(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::path::PathBuf;

    use super::*;
    use crate::epoch::{FilledPrograms, Program};
    use crate::options::ServerArgs;
    use crate::updates::switch::{Event, Phase, Start};

    use proto::process::{self, Spawn};

    /// A roster that records what the switch asks of it and reports nothing
    /// on its own.
    #[derive(Default)]
    struct Fake {
        claims: Vec<(String, u64, Vec<process::Scope>, Vec<u64>)>,
        spawns: Vec<Spawn>,
        stops: Vec<(String, String, Option<u64>)>,
        commands: Vec<(String, u64, process::command::Kind)>,
        serves: Vec<(u64, String)>,
        resumes: Vec<(String, u64, String)>,
        scripted: Vec<Event>,
    }

    impl Roster for Fake {
        fn claim(
            &mut self,
            host: &str,
            world: u64,
            scopes: &[process::Scope],
            retire: &[u64],
        ) -> Result<(), String> {
            self.claims
                .push((host.to_owned(), world, scopes.to_vec(), retire.to_vec()));
            Ok(())
        }

        fn spawn(
            &mut self,
            spawn: Spawn,
            _launch: &proto::config::EngineArgs,
        ) -> Result<crate::updates::switch::Spawned, String> {
            let pid = self.spawns.len() as u32 + 1;
            let generation = spawn.generation;
            self.spawns.push(spawn);
            Ok(crate::updates::switch::Spawned { generation, pid })
        }

        fn stop(&mut self, host: &str, group: &str, generation: Option<u64>) {
            self.stops
                .push((host.to_owned(), group.to_owned(), generation));
        }

        fn command(&mut self, host: &str, generation: u64, command: process::Command) {
            self.commands.push((
                host.to_owned(),
                generation,
                command.kind.expect("a command"),
            ));
        }

        fn serve(&mut self, epoch: u64, requests: &str) {
            self.serves.push((epoch, requests.to_owned()));
        }

        fn resume(&mut self, host: &str, generation: u64, reason: &str) {
            self.resumes
                .push((host.to_owned(), generation, reason.to_owned()));
        }

        fn events(&mut self) -> Vec<Event> {
            std::mem::take(&mut self.scripted)
        }
    }

    fn previous() -> Deployment {
        Deployment::of(ServerArgs::default()).expect("the configuration resolves")
    }

    /// The same deployment at another executable: a change of the build
    /// config, so the epoch layer's.
    fn epoch_target(base: &Deployment) -> Deployment {
        let mut config = base.config.clone();
        config.worker_config.executable = PathBuf::from("/opt/other-uniserve");
        Deployment::with_programs(config, base.programs.clone())
            .expect("the configuration resolves")
    }

    /// The same deployment with one program revision changed: the program
    /// layer's, not the epoch's.
    fn program_target(base: &Deployment) -> Deployment {
        let mut target = base.clone();
        let mut policy: Program = target.programs.policy.clone();
        policy.revision += 1;
        target.programs = FilledPrograms {
            kernels: target.programs.kernels.clone(),
            policy,
        };
        target
    }

    fn serving_record(deployment: &Deployment) -> EpochRecord {
        EpochRecord {
            config: deployment.engine.clone(),
            kernels: deployment.programs.kernels.clone(),
            policy: deployment.programs.policy.clone(),
            previous_kernels: BTreeMap::new(),
            previous_policy: None,
            status: Status::Serving,
            generation: Some(4),
            world: 11,
        }
    }

    fn change(target: u64) -> EpochChange {
        EpochChange {
            previous: 11,
            target,
            update: Update::new(7, previous()),
            group: "engine".to_owned(),
            start: Start {
                executable: "/opt/uniserve".to_owned(),
                build: Some("2.0.0+new".to_owned()),
                launch: proto::config::EngineArgs {
                    engine_config: "/run/engine-config-13.json".into(),
                    model: "/models/qwen3".into(),
                    requests: "unix:/run/requests.sock".to_owned(),
                    processes: "unix:/run/processes.sock".to_owned(),
                    engine: 7,
                    program_server: "/run/programs.sock".to_owned(),
                    devices: vec![0],
                    group: "engine".to_owned(),
                    rank: 0,
                    deployment: "qwen3".to_owned(),
                    world: 13,
                    scheduler_node: true,
                    peers: Vec::new(),
                    watchdog_timeout_s: None,
                    rollback_s: None,
                    queue_depth: 4,
                    world_size: 1,
                    standby: false,
                },
                environment: HashMap::new(),
            },
            previous_world: 11,
            previous_generation: 5,
            spawn_generation: 7,
            keeps: crate::options::PreviousWorld::Parked,
        }
    }

    /// A change of nothing is no layer's: `apply` falls through.
    #[test]
    fn identical_deployments_are_no_layers_change() {
        let previous = previous();
        let mut running = None;
        let mut roster = Fake::default();
        let epochs = Arc::new(Epochs::new());
        let applied = epoch_arm(
            &previous,
            &previous,
            change(12),
            &epochs,
            &mut roster,
            &mut running,
            false,
        )
        .expect("no layer owns an unchanged deployment");
        assert!(applied.is_none());
        assert!(running.is_none());
        assert!(roster.spawns.is_empty());
        assert!(epochs.get(12).is_none());
    }

    /// The host layer's change is the handover's, and the program layer's is
    /// the program machine's: the arm answers neither.
    #[test]
    fn other_layers_fall_through() {
        let previous = previous();
        let epochs = Arc::new(Epochs::new());

        let mut logged = previous.config.clone();
        logged.logging_config.log_requests = true;
        let host = Deployment::with_programs(logged, previous.programs.clone())
            .expect("the configuration resolves");
        let mut running = None;
        let mut roster = Fake::default();
        assert!(
            epoch_arm(
                &previous,
                &host,
                change(12),
                &epochs,
                &mut roster,
                &mut running,
                false
            )
            .expect("the host layer is another path")
            .is_none()
        );
        assert!(running.is_none());

        let program = program_target(&previous);
        assert!(
            epoch_arm(
                &previous,
                &program,
                change(12),
                &epochs,
                &mut roster,
                &mut running,
                false
            )
            .expect("the program layer is another path")
            .is_none()
        );
        assert!(running.is_none());
        assert!(roster.spawns.is_empty());
    }

    /// An epoch change builds the target's record from the target and the
    /// serving epoch, starts the target's children at the allocated
    /// generation, and hands the running switch back in the slot.
    #[test]
    fn an_epoch_change_starts_the_switch() {
        let previous = previous();
        let target = epoch_target(&previous);
        let epochs = Arc::new(Epochs::new());
        epochs.record(11, serving_record(&previous));
        let mut running = None;
        let mut roster = Fake::default();

        let applied = epoch_arm(
            &previous,
            &target,
            change(12),
            &epochs,
            &mut roster,
            &mut running,
            false,
        )
        .expect("the epoch layer runs")
        .expect("the epoch layer owns the change");
        let Applied::Epoch(report) = applied else {
            panic!("an epoch change reports an epoch, got {applied:?}");
        };
        assert_eq!(report["change"], "epoch");
        assert_eq!(report["phase"], "preparing");
        assert_eq!(report["build"], target.engine.executable.clone());

        let running = running.expect("the switch is kept for the caller");
        assert_eq!(running.phase(), Phase::Preparing);
        assert_eq!(roster.spawns.len(), 1);
        assert_eq!(roster.spawns[0].generation, 7);
        assert_eq!(roster.spawns[0].group, "engine");
        assert_eq!(roster.claims[0].1, 12);

        let record = epochs.get(12).expect("the target's record");
        assert_eq!(record.status, Status::Precopy);
        assert_eq!(record.config, target.engine);
        assert_eq!(record.kernels, target.programs.kernels);
        assert_eq!(record.previous_kernels, target.programs.kernels);
        assert_eq!(record.previous_policy, Some(target.programs.policy.clone()));
    }

    /// A dry run plans the change and changes nothing: no switch, no record,
    /// no child.
    #[test]
    fn a_dry_run_starts_nothing() {
        let previous = previous();
        let target = epoch_target(&previous);
        let epochs = Arc::new(Epochs::new());
        epochs.record(11, serving_record(&previous));
        let mut running = None;
        let mut roster = Fake::default();

        let applied = epoch_arm(
            &previous,
            &target,
            change(12),
            &epochs,
            &mut roster,
            &mut running,
            true,
        )
        .expect("a dry run plans")
        .expect("the epoch layer owns the change");
        let Applied::Planned(plan) = applied else {
            panic!("a dry run plans, got {applied:?}");
        };
        assert_eq!(plan["change"], "epoch");
        assert_eq!(plan["epoch"], 12);
        assert_eq!(plan["generation"], 7);
        assert!(running.is_none());
        assert!(roster.spawns.is_empty());
        assert!(roster.claims.is_empty());
        assert!(epochs.get(12).is_none(), "a plan writes no record");
    }

    /// A serving epoch the daemon cannot describe is refused: there is no
    /// previous generation to drain or resume, and one is not invented.
    #[test]
    fn a_missing_serving_record_is_refused() {
        let previous = previous();
        let target = epoch_target(&previous);
        let epochs = Arc::new(Epochs::new());
        let mut running = None;
        let mut roster = Fake::default();
        let error = epoch_arm(
            &previous,
            &target,
            change(12),
            &epochs,
            &mut roster,
            &mut running,
            false,
        )
        .expect_err("no serving record");
        assert!(matches!(error, UpdateError::Refused(_)));
        assert!(running.is_none());
        assert!(roster.spawns.is_empty());
    }

    /// One change at a time: a switch already running refuses another.
    #[test]
    fn a_running_switch_refuses_another_change() {
        let previous = previous();
        let target = epoch_target(&previous);
        let epochs = Arc::new(Epochs::new());
        epochs.record(11, serving_record(&previous));
        let mut roster = Fake::default();
        let mut running = None;
        epoch_arm(
            &previous,
            &target,
            change(12),
            &epochs,
            &mut roster,
            &mut running,
            false,
        )
        .expect("the first change starts")
        .expect("the epoch layer owns the change");

        let error = epoch_arm(
            &previous,
            &target,
            change(13),
            &epochs,
            &mut roster,
            &mut running,
            false,
        )
        .expect_err("another change is in progress");
        assert!(matches!(error, UpdateError::InProgress));
    }
}
