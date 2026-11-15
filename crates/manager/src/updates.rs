//! Updates of several services that share GPUs: the manager decides the order in which their
//! updates run and calls each service's admin API in turn; no node waits for it, and every step
//! within one service is that service's own (`PUT /admin/config`, its trial, `POST
//! /admin/config/commit`).
//!
//! Each `--node` is the leader daemon of one service (one served model name). Before it starts
//! a service's update, the manager reads the GPUs the service's epochs hold (`GET
//! /admin/config`, `gpus`, each named by its host and UUID) and asks for the update's dry run
//! (`PUT /admin/config` with `dry_run`): whether it changes anything and whether it starts a new
//! epoch, which prepares beside the serving one on the service's GPUs. The decision:
//!
//! - Services are taken in the order given. One whose update starts a new epoch starts only
//!   when no update in progress (preparing, or on trial until the manager commits it) holds one
//!   of its GPUs, so two services that share a GPU are never prepared at once. Services that
//!   share no GPU update at once.
//! - A service that cannot start waits, and its dry run is made again whenever an update in
//!   progress ends, so it starts from what its node states then.
//! - Each update runs to its end: applied in place, or a new epoch on trial, which the manager
//!   commits once it served `--commit-after` seconds, or which ended by itself meanwhile
//!   (`update_config.rollback`), as the node's event log states. Without `--commit-after` the
//!   trial stays for the operator, holding its GPUs.
//! - A change of the daemon's own options restarts the daemon in place, and its successor
//!   applies the rest of the change in an operation of its own: the manager waits for that
//!   operation's end in the node's event log, which the restart carries over.
//! - A failed update or a trial that ended without its commit stops the services that have not
//!   started; those in progress go on to their end.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use proto::admin::{
    ApplyRequest, ConfigChange, ConfigStatus, EventBody, Following, Gpu, OperationKind,
    OperationResponse, OperationState, Phase, UpdateReport,
};

use crate::node::{Node, Polling};

/// What one service's update needs, from its status and its dry run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Needs {
    /// Whether the update changes anything.
    pub(crate) changes: bool,
    /// Whether it starts a new epoch, which prepares beside the serving
    /// one and serves on trial.
    pub(crate) epoch: bool,
    /// The GPUs the service's epochs hold, which a new epoch prepares on.
    pub(crate) gpus: BTreeSet<Gpu>,
}

impl Needs {
    /// What the update of a service needs, from its status (`GET /admin/config`) and its dry
    /// run's change (`PUT /admin/config` with `dry_run`): an update in place or of a
    /// generation starts no epoch; one of an epoch does, as does an idle node's bring-up of a
    /// deployment, which holds no GPU before it starts.
    ///
    /// # Errors
    ///
    /// A dry run that answered a change it made.
    pub(crate) fn of(status: &ConfigStatus, change: &ConfigChange) -> Result<Self> {
        use proto::admin::DryRun;

        let gpus = status
            .gpus
            .iter()
            .flatten()
            .map(|share| share.gpu.clone())
            .collect();
        let epoch = match change {
            ConfigChange::None => return Ok(Self::default()),
            ConfigChange::DryRun { dry_run } => match dry_run {
                DryRun::InPlace { .. } | DryRun::Generation { .. } => false,
                DryRun::Epoch { .. } | DryRun::Deployment { .. } => true,
            },
            made => bail!("a dry run answered a change it made: {made:?}"),
        };
        Ok(Self {
            changes: true,
            epoch,
            gpus,
        })
    }
}

/// Why an update that starts a new epoch waits: it shares GPU `gpu` with the update in progress
/// of service `with`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Wait {
    with: usize,
    gpu: Gpu,
}

impl Wait {
    /// The reason, naming services by their nodes in `nodes`.
    fn describe(&self, nodes: &[Node]) -> String {
        format!(
            "shares GPU {} with {}, whose update is in progress",
            self.gpu, nodes[self.with].base
        )
    }
}

/// Why the update `candidate` cannot start beside the updates in progress
/// `beside`, by their services' indices, if it cannot.
pub(crate) fn conflict(candidate: &Needs, beside: &BTreeMap<usize, Needs>) -> Option<Wait> {
    beside.iter().find_map(|(with, other)| {
        candidate
            .gpus
            .intersection(&other.gpus)
            .next()
            .map(|gpu| Wait {
                with: *with,
                gpu: gpu.clone(),
            })
    })
}

/// How an update is applied and its trial ended.
#[derive(Debug, Clone)]
pub(crate) struct Options {
    /// The request of `PUT /admin/config`, without `dry_run`.
    pub(crate) request: ApplyRequest,
    /// How long a new epoch serves on trial before the manager commits it;
    /// `None` leaves it on trial for the operator.
    pub(crate) commit_after: Option<Duration>,
    /// How the manager reads a node's event log while it waits.
    pub(crate) polling: Polling,
}

impl Options {
    /// The request of the update's dry run.
    fn dry_run(&self) -> ApplyRequest {
        ApplyRequest {
            dry_run: true,
            wait: true,
            ..self.request.clone()
        }
    }
}

/// How one service's update ended.
#[derive(Debug)]
enum Ended {
    /// Applied without a new epoch, or committed.
    Done,
    /// A new epoch serves on trial, for the operator to end.
    OnTrial,
}

/// The dry run of `request` at `node`, and what its update needs.
async fn needs(node: &Node, request: &ApplyRequest) -> Result<(ConfigChange, Needs)> {
    let status = node.config().await?;
    let response: OperationResponse<ConfigChange> = node
        .send(reqwest::Method::PUT, "/admin/config", request)
        .await?;
    let change = response.result.context("the dry run answered no change")?;
    let needs = Needs::of(&status, &change)?;
    Ok((change, needs))
}

/// Updates every service of `nodes` as the module describes, and fails
/// with every failed update and trial that ended without its commit, once every update in
/// progress ended.
pub(crate) async fn apply(nodes: &[Node], options: &Options) -> Result<()> {
    let request = options.dry_run();
    let mut pending: VecDeque<usize> = (0..nodes.len()).collect();
    // The updates in progress or on trial, and what each needs.
    let mut holding: BTreeMap<usize, Needs> = BTreeMap::new();
    let mut waiting: BTreeMap<usize, Wait> = BTreeMap::new();
    let mut on_trial = Vec::new();
    let mut failures = Vec::new();
    let mut running = tokio::task::JoinSet::new();
    loop {
        // Make every waiting service's dry run again and start those that may, in
        // order, while no update failed.
        let mut deferred = VecDeque::new();
        while failures.is_empty()
            && let Some(index) = pending.pop_front()
        {
            let node = &nodes[index];
            let needs = match needs(node, &request)
                .await
                .with_context(|| format!("{}: the dry run of its update", node.base))
            {
                Ok((_, needs)) => needs,
                Err(error) => {
                    failures.push(format!("{error:#}"));
                    break;
                }
            };
            if !needs.changes {
                println!("{}: unchanged", node.base);
                continue;
            }
            if needs.epoch
                && let Some(wait) = conflict(&needs, &holding)
            {
                println!("{}: waits: {}", node.base, wait.describe(nodes));
                waiting.insert(index, wait);
                deferred.push_back(index);
                continue;
            }
            waiting.remove(&index);
            println!("{}: updating", node.base);
            holding.insert(index, needs);
            let node = node.clone();
            let options = options.clone();
            running.spawn(async move { (index, update(&node, &options).await) });
        }
        // Services not reached after a failure stay pending, without a dry run.
        deferred.extend(pending.drain(..));
        pending = deferred;

        let Some(joined) = running.join_next().await else {
            break;
        };
        let (index, ended) = joined.context("an update's task ended without its result")?;
        match ended {
            Ok(Ended::Done) => {
                holding.remove(&index);
            }
            // A trial holds both epochs' storage until the operator ends it.
            Ok(Ended::OnTrial) => on_trial.push(index),
            Err(error) => {
                holding.remove(&index);
                failures.push(format!("{error:#}"));
            }
        }
    }

    for index in &on_trial {
        println!(
            "{}: on trial; commit it (`commit`) or roll it back",
            nodes[*index].base
        );
    }
    let unstarted: Vec<String> = pending
        .iter()
        .map(|index| match waiting.get(index) {
            Some(wait) => format!("{} ({})", nodes[*index].base, wait.describe(nodes)),
            None => nodes[*index].base.clone(),
        })
        .collect();
    if !failures.is_empty() {
        bail!(
            "{}{}",
            failures.join("; "),
            if unstarted.is_empty() {
                String::new()
            } else {
                format!("; not started: {}", unstarted.join(", "))
            }
        );
    }
    // Without a failure, a service that did not start shares a GPU with a trial the operator
    // ends: it updates once the manager applies again after it.
    if !unstarted.is_empty() {
        println!(
            "after the trials end, apply again for: {}",
            unstarted.join(", ")
        );
    }
    Ok(())
}

/// Reports each service's dry run and the order [`apply`] takes them in from
/// these: after the services before it that share one of its GPUs. Changes nothing.
pub(crate) async fn dry_run(nodes: &[Node], options: &Options) -> Result<()> {
    let request = options.dry_run();
    let mut services = Vec::new();
    for node in nodes {
        let (change, needs) = needs(node, &request)
            .await
            .with_context(|| format!("{}: the dry run of its update", node.base))?;
        println!("{}: {}", node.base, serde_json::to_string(&change)?);
        services.push(needs);
    }
    for (index, needs) in services.iter().enumerate() {
        let base = &nodes[index].base;
        if !needs.changes {
            println!("{base}: unchanged");
            continue;
        }
        if !needs.epoch {
            println!("{base}: changes without a new epoch, at once");
            continue;
        }
        let after: Vec<&str> = services[..index]
            .iter()
            .enumerate()
            .filter(|(_, other)| other.epoch && !other.gpus.is_disjoint(&needs.gpus))
            .map(|(other, _)| nodes[other].base.as_str())
            .collect();
        if after.is_empty() {
            println!("{base}: a new epoch, first");
        } else {
            println!("{base}: a new epoch, after {}", after.join(", "));
        }
    }
    Ok(())
}

/// Applies `options.request` to the service `node` serves and runs its trial to its end:
/// committed once it served `options.commit_after`, or left on trial without it.
async fn update(node: &Node, options: &Options) -> Result<Ended> {
    // The events from now on state how what this apply starts ends.
    let since = node.last_event().await?;
    let response: OperationResponse<ConfigChange> = node
        .send(reqwest::Method::PUT, "/admin/config", &options.request)
        .await
        .with_context(|| format!("{} did not change", node.base))?;
    let change = response
        .result
        .with_context(|| format!("{} answered no change", node.base))?;
    println!("{}: {}", node.base, serde_json::to_string(&change)?);
    let update = match change {
        ConfigChange::Generation { update } | ConfigChange::Epoch { update } => update,
        ConfigChange::Daemon {
            then: Some(Following::Update),
            ..
        } => match after_restart(node, since, response.operation.id, options.polling).await? {
            Some(update) => update,
            None => return Ok(Ended::Done),
        },
        ConfigChange::None
        | ConfigChange::InPlace { .. }
        | ConfigChange::DryRun { .. }
        | ConfigChange::Daemon { then: None, .. }
        | ConfigChange::Deployment { .. } => return Ok(Ended::Done),
    };
    // An update, in either form, serves on trial once the apply responded.
    if update.phase != Phase::Trial {
        return Ok(Ended::Done);
    }
    let Some(trial_window) = options.commit_after else {
        return Ok(Ended::OnTrial);
    };

    // The trial serves its window unless it ends by itself before, as an update event states.
    let ended_early = tokio::time::timeout(
        trial_window,
        node.wait_for(since, options.polling, |event| match &event.body {
            EventBody::Update(report) if report.update == update.update && report.phase.ended() => {
                Some(report.as_ref().clone())
            }
            _ => None,
        }),
    )
    .await;
    if let Ok(ended) = ended_early {
        let report = ended?;
        bail!(
            "{}: the trial ended {} before its commit ({})",
            node.base,
            report.phase,
            describe_failure(&report)
        );
    }
    // The update stands on trial as the node states it now, which the commit then decides.
    let standing = node.config().await?.update;
    match &standing {
        Some(report) if report.update == update.update && report.phase == Phase::Trial => {}
        Some(report) if report.update == update.update => bail!(
            "{}: the trial ended {} before its commit ({})",
            node.base,
            report.phase,
            describe_failure(report)
        ),
        _ => bail!(
            "{}: update {} is no longer the node's",
            node.base,
            update.update
        ),
    }
    let committed: OperationResponse<UpdateReport> = node
        .send(reqwest::Method::POST, "/admin/config/commit", &())
        .await?;
    println!(
        "{}: committed {}",
        node.base,
        serde_json::to_string(&committed.result)?
    );
    Ok(Ended::Done)
}

/// Why the update of `report` did not stand, as its failure states it.
fn describe_failure(report: &UpdateReport) -> String {
    report.failure.as_ref().map_or_else(
        || "no failure stated".to_owned(),
        |failure| format!("{}: {}", failure.kind, failure.reason),
    )
}

/// The update the change that a daemon's hot restart left to its successor started: the
/// successor applies it in an operation (`config`) after operation `restarted`, whose end the
/// node's event log after `since` states; `None` when that change started no update.
async fn after_restart(
    node: &Node,
    since: u64,
    restarted: u64,
    polling: Polling,
) -> Result<Option<UpdateReport>> {
    let operation = node
        .wait_for(since, polling, |event| match &event.body {
            EventBody::Operation(operation)
                if operation.kind == OperationKind::Config
                    && operation.id > restarted
                    && operation.state != OperationState::Running =>
            {
                Some(operation.clone())
            }
            _ => None,
        })
        .await?;
    if operation.state == OperationState::Failed {
        bail!(
            "{}: the change after the daemon's hot restart failed: {}",
            node.base,
            operation.error.unwrap_or_default()
        );
    }
    // An update is numbered by the operation that applied it.
    let update = node.config().await?.update;
    Ok(update.filter(|update| update.update == operation.id))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::extract::{Query, State};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use proto::admin::{
        ChangeForm, DryRun, EngineLaunch, Event, EventsPage, EventsQuery, Executing,
        ExecutingState, Failure, FailureKind, GpuShare, GpuUuid, NodeState, Operation,
        ReplicaStatus, Status,
    };
    use proto::module::PackageRevision;
    use tokio::sync::Notify;

    use super::*;

    /// What the stand-in daemons were asked, in order (`"{service}:{call}"`),
    /// and the services that committed.
    #[derive(Default)]
    struct Log {
        calls: Mutex<Vec<String>>,
        committed: Mutex<BTreeSet<String>>,
    }

    impl Log {
        fn record(&self, call: String) {
            self.calls.lock().unwrap().push(call);
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }

        fn committed(&self, service: &str) -> bool {
            self.committed.lock().unwrap().contains(service)
        }

        /// The position of `call` among the calls; panics when absent.
        fn at(&self, call: &str) -> usize {
            let calls = self.calls();
            calls
                .iter()
                .position(|made| made == call)
                .unwrap_or_else(|| panic!("no call {call} in {calls:?}"))
        }
    }

    /// How a stand-in daemon's apply goes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Apply {
        /// A new epoch serves on trial until committed.
        Trial,
        /// A new epoch serves on trial and rolls back by itself at once.
        RollsBack,
        /// The daemon's own options change: it restarts, and its successor starts the new
        /// epoch's update in an operation of its own.
        Restarts,
    }

    /// One service's leader daemon, as far as the manager reads and drives its admin API,
    /// answering the admin schema the daemons serve.
    struct Daemon {
        name: &'static str,
        /// The GPUs its epochs hold, as `GET /admin/config` names them.
        gpus: Vec<Gpu>,
        apply: Apply,
        /// The update in flight or the last one.
        update: Mutex<Option<UpdateReport>>,
        /// Its event log, and what wakes a reader waiting for the next event.
        events: Mutex<Vec<Event>>,
        appended: Notify,
        operations: Mutex<u64>,
        log: Arc<Log>,
    }

    impl Daemon {
        fn record(&self, body: EventBody) {
            let mut events = self.events.lock().unwrap();
            let seq = events.len() as u64 + 1;
            events.push(Event {
                seq,
                time: 0.0,
                message: String::new(),
                body,
            });
            drop(events);
            self.appended.notify_waiters();
        }

        /// A succeeded operation of `kind`, its start and end recorded in the event log.
        fn operation(&self, kind: OperationKind) -> Operation {
            let mut number = self.operations.lock().unwrap();
            *number += 1;
            let mut operation = Operation {
                id: *number,
                kind,
                state: OperationState::Running,
                started: 0.0,
                ended: None,
                error: None,
            };
            self.record(EventBody::Operation(operation.clone()));
            operation.state = OperationState::Succeeded;
            operation.ended = Some(0.0);
            self.record(EventBody::Operation(operation.clone()));
            operation
        }

        /// Update `number` in `phase`, its event recorded, as the daemon's state.
        fn enter(&self, number: u64, phase: Phase) -> UpdateReport {
            let failure = (phase == Phase::RolledBack).then(|| {
                Failure::new(
                    FailureKind::Threshold(proto::admin::Threshold::FailureRate),
                    "epoch 2 failed".to_owned(),
                )
            });
            let report = UpdateReport {
                change: ChangeForm::Epoch,
                update: number,
                phase,
                epoch: 2,
                rollback: phase == Phase::RolledBack,
                switch_ms: None,
                trial_s: None,
                deadline_s: None,
                failure,
                uncompared: Vec::new(),
                claim_failed: None,
                generation: None,
                epoch_form: None,
                trial: None,
                transfers: None,
            };
            *self.update.lock().unwrap() = Some(report.clone());
            self.record(EventBody::Update(Box::new(report.clone())));
            report
        }
    }

    async fn status(State(daemon): State<Arc<Daemon>>) -> Json<Status> {
        let next_event = daemon.events.lock().unwrap().len() as u64 + 1;
        Json(Status {
            version: "test".to_owned(),
            node: NodeState::Idle { last_error: None },
            operation: None,
            last_operation: None,
            next_event,
        })
    }

    async fn events(
        State(daemon): State<Arc<Daemon>>,
        Query(query): Query<EventsQuery>,
    ) -> Json<EventsPage> {
        let read = |daemon: &Daemon| {
            let events = daemon.events.lock().unwrap();
            let found: Vec<Event> = events[(query.after as usize).min(events.len())..].to_vec();
            let next = found.last().map_or(query.after, |event| event.seq);
            EventsPage {
                events: found,
                next,
                gap: false,
            }
        };
        let notified = daemon.appended.notified();
        let page = read(&daemon);
        if !page.events.is_empty() {
            return Json(page);
        }
        let _ = tokio::time::timeout(Duration::from_secs_f64(query.wait), notified).await;
        Json(read(&daemon))
    }

    async fn config(State(daemon): State<Arc<Daemon>>) -> Json<ConfigStatus> {
        let gpus = daemon
            .gpus
            .iter()
            .map(|gpu| GpuShare {
                gpu: gpu.clone(),
                total: 80,
                quota: None,
                used: 1,
            })
            .collect();
        let replica = ReplicaStatus {
            config: String::new(),
            digest: daemon.name.to_owned(),
            executing: Executing {
                epoch: 1,
                state: ExecutingState::Serving,
                reason: None,
            },
        };
        Json(ConfigStatus {
            first: Some(replica.clone()),
            replicas: vec![replica],
            gpus: Some(gpus),
            quotas: None,
            update: daemon.update.lock().unwrap().clone(),
            unavailable: Vec::new(),
            epochs: Vec::new(),
            processes: Vec::new(),
        })
    }

    async fn change(
        State(daemon): State<Arc<Daemon>>,
        Json(request): Json<ApplyRequest>,
    ) -> Json<OperationResponse<ConfigChange>> {
        let operation = daemon.operation(OperationKind::Config);
        if request.dry_run {
            daemon.log.record(format!("{}:dry_run", daemon.name));
            let dry_run = DryRun::Epoch {
                epoch: 2,
                engine: PackageRevision {
                    backend: "uniserve".to_owned(),
                    revision: 1,
                    arguments: toml::Table::new(),
                },
                replicas: 1,
                engines: vec![vec![EngineLaunch {
                    host: daemon.gpus[0].host.clone(),
                    launch_generation: 3,
                }]],
                shrink: false,
            };
            return Json(OperationResponse {
                operation,
                result: Some(ConfigChange::DryRun { dry_run }),
            });
        }
        daemon.log.record(format!("{}:apply", daemon.name));
        let change = match daemon.apply {
            Apply::Trial => ConfigChange::Epoch {
                update: daemon.enter(operation.id, Phase::Trial),
            },
            Apply::RollsBack => {
                let update = daemon.enter(operation.id, Phase::Trial);
                // The trial's threshold passes right after the apply answered.
                let rolling = Arc::clone(&daemon);
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    rolling.enter(update.update, Phase::RolledBack);
                });
                ConfigChange::Epoch { update }
            }
            Apply::Restarts => {
                // The successor applies the rest of the change once it serves.
                let successor = Arc::clone(&daemon);
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    let operation = successor.operation(OperationKind::Config);
                    successor.enter(operation.id, Phase::Trial);
                });
                ConfigChange::Daemon {
                    pid: 42,
                    then: Some(Following::Update),
                }
            }
        };
        Json(OperationResponse {
            operation,
            result: Some(change),
        })
    }

    async fn commit(State(daemon): State<Arc<Daemon>>) -> Json<OperationResponse<UpdateReport>> {
        daemon.log.record(format!("{}:commit", daemon.name));
        let operation = daemon.operation(OperationKind::Commit);
        let number = daemon
            .update
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, |update| update.update);
        let committed = daemon.enter(number, Phase::Committed);
        daemon
            .log
            .committed
            .lock()
            .unwrap()
            .insert(daemon.name.to_owned());
        Json(OperationResponse {
            operation,
            result: Some(committed),
        })
    }

    /// Serves `daemon`'s admin API on a port of the loopback interface and
    /// returns the manager's node of it.
    async fn serve(daemon: Daemon) -> Node {
        let router = Router::new()
            .route("/admin/status", get(status))
            .route("/admin/events", get(events))
            .route("/admin/config", get(config).put(change))
            .route("/admin/config/commit", post(commit))
            .with_state(Arc::new(daemon));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Node::new(&format!("http://{address}")).unwrap()
    }

    /// GPU `ordinal` of host `host`.
    fn gpu(host: &str, ordinal: u8) -> Gpu {
        Gpu {
            host: host.to_owned(),
            gpu: GpuUuid([ordinal; 16]),
        }
    }

    fn daemon(name: &'static str, gpus: &[Gpu], apply: Apply, log: &Arc<Log>) -> Daemon {
        Daemon {
            name,
            gpus: gpus.to_vec(),
            apply,
            update: Mutex::new(None),
            events: Mutex::new(Vec::new()),
            appended: Notify::new(),
            operations: Mutex::new(0),
            log: Arc::clone(log),
        }
    }

    fn options(commit_after: Option<Duration>) -> Options {
        Options {
            request: ApplyRequest {
                set: vec!["worker_config.engine_revision=2".to_owned()],
                ..ApplyRequest::default()
            },
            commit_after,
            polling: Polling {
                wait: Duration::from_secs(1),
                retry: Duration::from_millis(10),
            },
        }
    }

    /// How long the stand-in trials serve before the manager commits.
    const TRIAL: Duration = Duration::from_millis(200);

    /// Two services that share a GPU are never prepared at once: the second
    /// starts once the first committed, and a service on other GPUs and
    /// another host updates beside the first.
    #[tokio::test(flavor = "multi_thread")]
    async fn services_that_share_a_gpu_update_one_after_another() {
        let log = Arc::new(Log::default());
        let shared = [gpu("rank-0", 0)];
        let other = [gpu("rank-1", 1)];
        let nodes = vec![
            serve(daemon("a", &shared, Apply::Trial, &log)).await,
            serve(daemon("b", &shared, Apply::Trial, &log)).await,
            serve(daemon("c", &other, Apply::Trial, &log)).await,
        ];
        apply(&nodes, &options(Some(TRIAL))).await.unwrap();
        assert!(log.at("c:apply") < log.at("a:commit"), "{:?}", log.calls());
        assert!(log.at("b:apply") > log.at("a:commit"), "{:?}", log.calls());
        assert!(["a", "b", "c"].iter().all(|service| log.committed(service)));
    }

    /// A trial that rolled back stops the services that wait, which the manager names, as soon
    /// as the node's event log states it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_rolled_back_trial_stops_the_services_that_wait() {
        let log = Arc::new(Log::default());
        let shared = [gpu("rank-0", 0)];
        let nodes = vec![
            serve(daemon("a", &shared, Apply::RollsBack, &log)).await,
            serve(daemon("b", &shared, Apply::Trial, &log)).await,
        ];
        let started = std::time::Instant::now();
        let error = apply(&nodes, &options(Some(Duration::from_secs(60))))
            .await
            .expect_err("a rolled-back trial fails the apply");
        assert!(started.elapsed() < Duration::from_secs(30));
        let message = format!("{error:#}");
        assert!(message.contains("rolled_back"), "{message}");
        assert!(message.contains("failure_rate"), "{message}");
        assert!(message.contains(&nodes[1].base), "{message}");
        assert!(
            !log.calls().contains(&"b:apply".to_owned()),
            "{:?}",
            log.calls()
        );
        assert!(!log.committed("a"));
    }

    /// Without a commit, a new epoch stays on trial for the operator and
    /// holds its GPUs: a service that shares one waits for the next apply,
    /// and one on other GPUs updates.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_trial_left_for_the_operator_holds_its_gpus() {
        let log = Arc::new(Log::default());
        let shared = [gpu("rank-0", 0)];
        let other = [gpu("rank-1", 1)];
        let nodes = vec![
            serve(daemon("a", &shared, Apply::Trial, &log)).await,
            serve(daemon("b", &shared, Apply::Trial, &log)).await,
            serve(daemon("c", &other, Apply::Trial, &log)).await,
        ];
        apply(&nodes, &options(None)).await.unwrap();
        let calls = log.calls();
        assert!(calls.contains(&"a:apply".to_owned()), "{calls:?}");
        assert!(calls.contains(&"c:apply".to_owned()), "{calls:?}");
        assert!(!calls.contains(&"b:apply".to_owned()), "{calls:?}");
        assert!(
            !calls.iter().any(|call| call.ends_with(":commit")),
            "{calls:?}"
        );
    }

    /// A change of the daemon's own options restarts it, and the update its successor starts in
    /// an operation of its own, which the node's event log states, runs its trial and is
    /// committed.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_update_after_a_hot_restart_is_followed_to_its_commit() {
        let log = Arc::new(Log::default());
        let nodes = vec![serve(daemon("a", &[gpu("rank-0", 0)], Apply::Restarts, &log)).await];
        apply(&nodes, &options(Some(TRIAL))).await.unwrap();
        assert!(log.at("a:commit") > log.at("a:apply"), "{:?}", log.calls());
        assert!(log.committed("a"));
    }

    /// The GPUs an update uses are those its service's epochs hold; an update in place or of a
    /// generation starts no epoch, one that changes nothing needs nothing, and an idle node's
    /// bring-up starts an epoch on no GPU yet.
    #[test]
    fn an_updates_needs_come_from_its_status_and_its_dry_run() {
        let held = vec![gpu("rank-0", 0), gpu("rank-1", 1)];
        let status = ConfigStatus {
            first: None,
            replicas: Vec::new(),
            gpus: Some(
                held.iter()
                    .map(|gpu| GpuShare {
                        gpu: gpu.clone(),
                        total: 80,
                        quota: Some(40),
                        used: 10,
                    })
                    .collect(),
            ),
            quotas: None,
            update: None,
            unavailable: Vec::new(),
            epochs: Vec::new(),
            processes: Vec::new(),
        };
        let dry_run = |dry_run| ConfigChange::DryRun { dry_run };
        let package = PackageRevision {
            backend: "uniserve".to_owned(),
            revision: 2,
            arguments: toml::Table::new(),
        };
        let epoch = Needs::of(
            &status,
            &dry_run(DryRun::Epoch {
                epoch: 2,
                engine: package,
                replicas: 1,
                engines: Vec::new(),
                shrink: true,
            }),
        )
        .unwrap();
        assert!(epoch.changes && epoch.epoch);
        assert_eq!(epoch.gpus, held.into_iter().collect());

        let in_place = Needs::of(
            &status,
            &dry_run(DryRun::InPlace {
                replicas: Vec::new(),
            }),
        );
        assert!(in_place.unwrap().changes);
        let unchanged = Needs::of(&status, &ConfigChange::None).unwrap();
        assert_eq!(unchanged, Needs::default());
        let idle = ConfigStatus {
            gpus: None,
            ..status.clone()
        };
        let deployment = DryRun::Deployment {
            deployment: "qwen".to_owned(),
        };
        let brought = Needs::of(&idle, &dry_run(deployment)).unwrap();
        assert!(brought.epoch && brought.gpus.is_empty());
        // A dry run that answered a change it made is no dry run.
        assert!(
            Needs::of(
                &status,
                &ConfigChange::Deployment {
                    deployment: "qwen".to_owned()
                }
            )
            .is_err()
        );
    }
}
