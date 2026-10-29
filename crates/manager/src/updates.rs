//! Updates of several services that share GPUs and hosts
//! (`specs/online-operations.md` §5.1): the manager decides the order in
//! which their updates run and calls each service's admin API in turn; no
//! node waits for it, and every step within one service is that service's
//! own (`PUT /admin/config`, its trial, `POST /admin/config/commit`).
//!
//! Each `--node` is the leader daemon of one service (one served model
//! name). Before it starts a service's update, the manager reads the GPUs
//! the service's worlds hold (`GET /admin/config`, `gpus`) and asks for the
//! update's plan (`PUT /admin/config` with `dry_run`): whether it starts a
//! new world, the GPUs its successor prepares on, each host's host storage
//! it requires and admits, and whether every GPU and host it uses has the
//! storage it requires (the plan's `storage` and `redistribution`). The
//! decision:
//!
//! - Services are taken in the order given. One whose update starts a new
//!   world starts only when no update in progress (preparing, or on trial
//!   until the manager commits it) holds one of its GPUs, so two services
//!   that share a GPU are never prepared at once; when its plan, made just
//!   then, finds the storage it requires on every GPU and host; and when
//!   the host storage it requires on a host, beside what the updates in
//!   progress there required, is within what the host admits. Services
//!   that share no GPU, and whose host storage fits together, update at
//!   once.
//! - A service that cannot start waits, and is planned again whenever an
//!   update in progress ends: a commit releases the previous world's
//!   storage, which may make its plan fit.
//! - Each update runs to its end: applied in place, or a new world on
//!   trial, which the manager commits once it served `--commit-after`
//!   seconds (completing the service's remaining replica trials), or which
//!   rolled back by itself meanwhile (`update_config.rollback`). Without
//!   `--commit-after` the trial stays for the operator, holding its GPUs.
//! - A failed update or a rolled-back trial stops the services that have
//!   not started; those in progress go on to their end.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};

use crate::Node;

/// A GPU: its host and its name as `nvidia-smi` prints its UUID.
type Gpu = (String, String);

/// What one service's update needs, from its status and its plan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Needs {
    /// Whether the update changes anything.
    pub(crate) changes: bool,
    /// Whether it starts a new world, which prepares beside the serving
    /// one and serves on trial.
    pub(crate) world: bool,
    /// The GPUs the service's worlds hold and those its successor prepares
    /// on.
    pub(crate) gpus: BTreeSet<Gpu>,
    /// Each host's host storage the update requires and what the host
    /// admits of new host storage, in bytes, as the plan found them.
    pub(crate) hosts: BTreeMap<String, (u64, u64)>,
    /// Why the plan does not fit now: the refusals of its storage plans.
    pub(crate) refusal: Option<String>,
}

impl Needs {
    /// What the update of a service needs, from its status (`GET
    /// /admin/config`) and its dry run's result (`PUT /admin/config` with
    /// `dry_run`): one replica's plan, the plans of every independent
    /// replica (`replicas`), a group change's plan (`previous_worlds`), or
    /// an idle node's deployment (`deployment`).
    pub(crate) fn of(status: &Value, planned: &Value) -> Self {
        let mut needs = Self::default();
        let held = status["gpus"].as_array().into_iter().flatten();
        needs.gpus.extend(held.filter_map(gpu));
        if planned["change"] != "planned" {
            return needs;
        }
        let plan = &planned["plan"];
        let plans: Vec<&Value> = match plan["replicas"].as_array() {
            Some(replicas) => replicas.iter().collect(),
            None => vec![plan],
        };
        let mut refusals = Vec::new();
        for plan in plans {
            let world = !plan["steps"].is_null()
                || plan["deployment"].is_string()
                || plan["previous_worlds"].is_array();
            needs.world |= world;
            needs.changes |= world || plan["world"] == true || plan["daemon"] == true;

            // The storage plan names the GPUs the successor prepares on
            // beside the serving world, and each host's admission.
            let storage = &plan["storage"];
            let planned_gpus = storage["gpus"].as_array().into_iter().flatten();
            needs.gpus.extend(planned_gpus.filter_map(gpu));
            for host in storage["hosts"].as_array().into_iter().flatten() {
                let (Some(name), Some(required), Some(admits)) = (
                    host["host"].as_str(),
                    host["required"].as_u64(),
                    host["admits"].as_u64(),
                ) else {
                    continue;
                };
                // A service's replicas update one at a time: the largest
                // requirement on a host is the update's.
                let entry = needs.hosts.entry(name.to_owned()).or_default();
                *entry = (entry.0.max(required), admits);
            }
            if storage["feasible"] == false {
                refusals.push(
                    storage["refusal"]
                        .as_str()
                        .unwrap_or("the storage plan does not fit")
                        .to_owned(),
                );
            }

            // A successor on other GPUs (another parallel configuration,
            // another replica's free GPUs) names them in its weights' plan.
            let redistributions: Vec<&Value> = match plan["redistribution"].as_array() {
                Some(reports) => reports.iter().collect(),
                None => vec![&plan["redistribution"]],
            };
            for report in redistributions {
                let ranks = report["ranks"].as_array().into_iter().flatten();
                needs.gpus.extend(ranks.filter_map(gpu));
                if report["feasible"] == false {
                    refusals.push(
                        report["refusal"]
                            .as_str()
                            .unwrap_or("the target GPUs cannot hold the successor")
                            .to_owned(),
                    );
                }
            }
            if plan["feasible"] == false {
                let group = plan["refusals"].as_array().into_iter().flatten();
                refusals.extend(group.filter_map(Value::as_str).map(str::to_owned));
            }
        }
        refusals.dedup();
        needs.refusal = (!refusals.is_empty()).then(|| refusals.join("; "));
        needs
    }
}

/// A GPU an entry names with its `host` and `gpu`.
fn gpu(entry: &Value) -> Option<Gpu> {
    Some((
        entry["host"].as_str()?.to_owned(),
        entry["gpu"].as_str()?.to_owned(),
    ))
}

/// Why an update that starts a new world waits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Wait {
    /// It shares GPU `gpu` with the update in progress of service `with`.
    Shares { with: usize, gpu: Gpu },
    /// Its plan does not fit: the plan's refusal.
    Refused(String),
    /// Host `host` admits `admits` bytes of new host storage, and the
    /// update requires `required` beside the `others` the updates in
    /// progress there required.
    Host {
        host: String,
        required: u64,
        others: u64,
        admits: u64,
    },
}

impl Wait {
    /// The reason, naming services by their nodes in `nodes`.
    fn describe(&self, nodes: &[Node]) -> String {
        match self {
            Wait::Shares {
                with,
                gpu: (host, gpu),
            } => format!(
                "shares GPU {gpu} of {host} with {}, whose update is in progress",
                nodes[*with].base
            ),
            Wait::Refused(refusal) => format!("its plan does not fit: {refusal}"),
            Wait::Host {
                host,
                required,
                others,
                admits,
            } => format!(
                "host {host} admits {admits} bytes of new host storage, and the update \
                 requires {required} beside the {others} the updates in progress there required"
            ),
        }
    }
}

/// Why the update `candidate` cannot start beside the updates in progress
/// `beside`, by their services' indices, if it cannot.
pub(crate) fn conflict(candidate: &Needs, beside: &BTreeMap<usize, Needs>) -> Option<Wait> {
    for (with, other) in beside {
        if let Some(gpu) = candidate.gpus.intersection(&other.gpus).next() {
            return Some(Wait::Shares {
                with: *with,
                gpu: gpu.clone(),
            });
        }
    }
    if let Some(refusal) = &candidate.refusal {
        return Some(Wait::Refused(refusal.clone()));
    }
    for (host, &(required, admits)) in &candidate.hosts {
        let others: u64 = beside
            .values()
            .filter_map(|other| other.hosts.get(host))
            .map(|(required, _)| required)
            .sum();
        if others > 0 && required + others > admits {
            return Some(Wait::Host {
                host: host.clone(),
                required,
                others,
                admits,
            });
        }
    }
    None
}

/// How an update is applied and its trial ended.
#[derive(Debug, Clone)]
pub(crate) struct Options {
    /// The body of `PUT /admin/config`, without `dry_run`.
    pub(crate) body: Value,
    /// How long a new world serves on trial before the manager commits it;
    /// `None` leaves it on trial for the operator.
    pub(crate) commit_after: Option<Duration>,
    /// How often the manager reads a node's status while it waits for an
    /// operation the node runs on its own.
    pub(crate) poll: Duration,
}

/// How one service's update ended.
#[derive(Debug)]
enum Ended {
    /// Applied without a new world, or committed.
    Done,
    /// A new world serves on trial, for the operator to end.
    OnTrial,
}

/// Updates every service of `nodes` as the module describes, and fails
/// with every failed update, rolled-back trial and service that did not
/// fit, once every update in progress ended.
pub(crate) async fn apply(nodes: &[Node], options: &Options) -> Result<()> {
    let mut dry_run = options.body.clone();
    dry_run["dry_run"] = json!(true);
    dry_run["wait"] = json!(true);

    let mut pending: VecDeque<usize> = (0..nodes.len()).collect();
    // The updates in progress or on trial, and what each needs.
    let mut holding: BTreeMap<usize, Needs> = BTreeMap::new();
    let mut waiting: BTreeMap<usize, Wait> = BTreeMap::new();
    let mut on_trial = Vec::new();
    let mut failures = Vec::new();
    let mut running = tokio::task::JoinSet::new();
    loop {
        // Plan every waiting service again and start those that fit, in
        // order, while no update failed.
        let mut deferred = VecDeque::new();
        while failures.is_empty()
            && let Some(index) = pending.pop_front()
        {
            let node = &nodes[index];
            let planned = async {
                let status = node.get("/admin/config").await?.unwrap_or(Value::Null);
                let answer = node.put("/admin/config", &dry_run).await?;
                Ok::<_, anyhow::Error>(Needs::of(&status, &answer["result"]))
            }
            .await
            .with_context(|| format!("{}: planning its update", node.base));
            let needs = match planned {
                Ok(needs) => needs,
                Err(error) => {
                    failures.push(format!("{error:#}"));
                    break;
                }
            };
            if !needs.changes {
                println!("{}: unchanged", node.base);
                continue;
            }
            if needs.world
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
        // Services not reached after a failure stay pending, unplanned.
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
            // A trial holds both worlds' storage until the operator ends it.
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
    if !unstarted.is_empty() {
        // A service that waits for a trial the operator ends updates once
        // the manager applies again after it; one that fits nowhere while
        // no update is in progress is refused.
        let behind_trials = pending.iter().all(|index| {
            matches!(waiting.get(index), Some(Wait::Shares { with, .. }) if on_trial.contains(with))
        });
        if behind_trials {
            println!(
                "after the trials end, apply again for: {}",
                unstarted.join(", ")
            );
            return Ok(());
        }
        bail!(
            "no update in progress leaves room for: {}",
            unstarted.join(", ")
        );
    }
    Ok(())
}

/// Reports each service's plan and the order [`apply`] takes them in from
/// these plans: after the services before it that share one of its GPUs,
/// and once its plan fits. Changes nothing.
pub(crate) async fn plan(nodes: &[Node], options: &Options) -> Result<()> {
    let mut dry_run = options.body.clone();
    dry_run["dry_run"] = json!(true);
    dry_run["wait"] = json!(true);
    let mut planned = Vec::new();
    for node in nodes {
        let status = node.get("/admin/config").await?.unwrap_or(Value::Null);
        let answer = node
            .put("/admin/config", &dry_run)
            .await
            .with_context(|| format!("{}: planning its update", node.base))?;
        println!("{}: {}", node.base, answer["result"]);
        planned.push(Needs::of(&status, &answer["result"]));
    }
    for (index, needs) in planned.iter().enumerate() {
        let base = &nodes[index].base;
        if !needs.changes {
            println!("{base}: unchanged");
            continue;
        }
        if !needs.world {
            println!("{base}: changes without a new world, at once");
            continue;
        }
        let after: Vec<&str> = planned[..index]
            .iter()
            .enumerate()
            .filter(|(_, other)| other.world && !other.gpus.is_disjoint(&needs.gpus))
            .map(|(other, _)| nodes[other].base.as_str())
            .collect();
        if after.is_empty() {
            println!("{base}: a new world, first");
        } else {
            println!("{base}: a new world, after {}", after.join(", "));
        }
        if let Some(refusal) = &needs.refusal {
            println!("{base}: does not fit now: {refusal}");
        }
    }
    Ok(())
}

/// Applies `options.body` to the service `node` serves and runs its
/// trial to its end: committed once it served `options.commit_after`, with
/// the service's remaining replica trials, or left on trial without it.
async fn update(node: &Node, options: &Options) -> Result<Ended> {
    let answer = node
        .put("/admin/config", &options.body)
        .await
        .with_context(|| format!("{} did not change", node.base))?;
    let mut result = answer["result"].clone();
    println!("{}: {result}", node.base);
    if result["change"] == "daemon" && result["then"] == "world" {
        // The hot restart's successor changes the world in an operation of
        // its own, after the one that answered.
        let after = answer["id"].as_u64().unwrap_or(0);
        result = operation_after(node, after, options.poll).await?;
        println!("{}: {result}", node.base);
    }
    if result["change"] != "world" {
        return Ok(Ended::Done);
    }
    let Some(trial_window) = options.commit_after else {
        return Ok(Ended::OnTrial);
    };
    let mut trial = result["world"].clone();
    loop {
        tokio::time::sleep(trial_window).await;
        let status = node.get("/admin/config").await?.unwrap_or(Value::Null);
        let worlds = if trial["worlds"].is_array() {
            "worlds"
        } else {
            "world"
        };
        if status["trial"].is_null() || status["trial"][worlds] != trial[worlds] {
            bail!(
                "{}: the trial rolled back before its commit ({})",
                node.base,
                status["last"]
            );
        }
        let committed = node.post("/admin/config/commit", &json!({})).await?;
        let result = &committed["result"];
        println!("{}: committed {result}", node.base);
        if let Some(error) = result.get("rollout_error") {
            bail!("{}: {error}", node.base);
        }
        let Some(next) = result.get("next_trial") else {
            break;
        };
        trial = next.clone();
    }
    Ok(Ended::Done)
}

/// The result of the first `config` operation `node` ran after operation
/// `after`, once it ended, reading its status every `poll`.
async fn operation_after(node: &Node, after: u64, poll: Duration) -> Result<Value> {
    loop {
        if let Some(status) = node.get("/admin/status").await? {
            let last = &status["last_operation"];
            if status["operation"].is_null()
                && last["kind"] == "config"
                && last["id"].as_u64().is_some_and(|id| id > after)
            {
                if last["state"] == "succeeded" {
                    return Ok(last["result"].clone());
                }
                bail!(
                    "{}: the change after the daemon's hot restart failed: {}",
                    node.base,
                    last["error"]
                );
            }
        }
        tokio::time::sleep(poll).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::routing::{get, post};
    use axum::{Json, Router};

    use super::*;

    /// What the simulated daemons were asked, in order (`"{service}:{call}"`),
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

    /// A service's plan, as its daemon answers a dry run now.
    type Planner = Box<dyn Fn(&Log) -> Value + Send + Sync>;

    /// One service's leader daemon, as far as the manager reads and drives
    /// its admin API.
    struct Daemon {
        name: &'static str,
        /// The GPUs its worlds hold, as `GET /admin/config` names them.
        gpus: Vec<(&'static str, &'static str)>,
        plan: Planner,
        /// Whether its trials roll back by themselves.
        rolls_back: bool,
        trial: Mutex<Option<u64>>,
        log: Arc<Log>,
    }

    async fn config(State(daemon): State<Arc<Daemon>>) -> Json<Value> {
        let trial = (*daemon.trial.lock().unwrap()).filter(|_| !daemon.rolls_back);
        let gpus: Vec<Value> = daemon
            .gpus
            .iter()
            .map(|(host, gpu)| json!({ "host": host, "gpu": gpu, "used": 1 }))
            .collect();
        Json(json!({
            "gpus": gpus,
            "trial": trial.map(|world| json!({ "world": world })),
            "last": if daemon.rolls_back {
                json!({ "rolled_back": 2, "reason": "world 2 failed" })
            } else {
                Value::Null
            },
        }))
    }

    async fn change(State(daemon): State<Arc<Daemon>>, Json(body): Json<Value>) -> Json<Value> {
        if body["dry_run"] == true {
            daemon.log.record(format!("{}:plan", daemon.name));
            let plan = (daemon.plan)(&daemon.log);
            return Json(json!({
                "id": 1, "kind": "config", "state": "succeeded",
                "result": { "change": "planned", "plan": plan },
            }));
        }
        daemon.log.record(format!("{}:apply", daemon.name));
        *daemon.trial.lock().unwrap() = Some(2);
        Json(json!({
            "id": 2, "kind": "config", "state": "succeeded",
            "result": { "change": "world", "world": { "world": 2, "previous_world": 1 } },
        }))
    }

    async fn commit(State(daemon): State<Arc<Daemon>>) -> Json<Value> {
        daemon.log.record(format!("{}:commit", daemon.name));
        daemon.trial.lock().unwrap().take();
        daemon
            .log
            .committed
            .lock()
            .unwrap()
            .insert(daemon.name.to_owned());
        Json(json!({
            "id": 3, "kind": "commit", "state": "succeeded",
            "result": { "world": 2, "previous_world": 1 },
        }))
    }

    /// Serves `daemon`'s admin API on a port of the loopback interface and
    /// returns the manager's node of it.
    async fn serve(daemon: Daemon) -> Node {
        let router = Router::new()
            .route("/admin/config", get(config).put(change))
            .route("/admin/config/commit", post(commit))
            .with_state(Arc::new(daemon));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Node::new(&format!("http://{address}"), &reqwest::Client::new(), None)
    }

    /// The plan of a new world beside the serving one on `gpus`, requiring
    /// `required` bytes of host storage of host `rank-0`, which admits
    /// 1000, and refused for `refusal`, as a daemon's dry run states it.
    fn world(gpus: &[(&str, &str)], required: u64, refusal: Option<&str>) -> Value {
        let gpus: Vec<Value> = gpus
            .iter()
            .map(|(host, gpu)| json!({ "host": host, "gpu": gpu, "feasible": refusal.is_none() }))
            .collect();
        json!({
            "world": true,
            "daemon": false,
            "steps": [{ "step": "shrink" }, { "step": "prepare" }, { "step": "promote" }],
            "storage": {
                "feasible": refusal.is_none(),
                "refusal": refusal,
                "gpus": gpus,
                "hosts": [{ "host": "rank-0", "required": required, "admits": 1000 }],
            },
            "redistribution": null,
        })
    }

    fn daemon(
        name: &'static str,
        gpus: &[(&'static str, &'static str)],
        plan: Planner,
        log: &Arc<Log>,
    ) -> Daemon {
        Daemon {
            name,
            gpus: gpus.to_vec(),
            plan,
            rolls_back: false,
            trial: Mutex::new(None),
            log: Arc::clone(log),
        }
    }

    /// A plan that is always `plan`.
    fn always(plan: Value) -> Planner {
        Box::new(move |_| plan.clone())
    }

    fn options(commit_after: Option<Duration>) -> Options {
        Options {
            body: json!({ "set": ["worker_config.watchdog_timeout=301"], "wait": true }),
            commit_after,
            poll: Duration::from_millis(10),
        }
    }

    /// How long the simulated trials serve before the manager commits.
    const TRIAL: Duration = Duration::from_millis(200);

    /// Two services that share a GPU are never prepared at once: the second
    /// starts once the first committed, and a service on other GPUs and
    /// another host updates beside the first.
    #[tokio::test(flavor = "multi_thread")]
    async fn services_that_share_a_gpu_update_one_after_another() {
        let log = Arc::new(Log::default());
        let shared = [("rank-0", "GPU-0")];
        let other = [("rank-1", "GPU-1")];
        let nodes = vec![
            serve(daemon("a", &shared, always(world(&shared, 0, None)), &log)).await,
            serve(daemon("b", &shared, always(world(&shared, 0, None)), &log)).await,
            serve(daemon("c", &other, always(world(&other, 0, None)), &log)).await,
        ];
        apply(&nodes, &options(Some(TRIAL))).await.unwrap();
        assert!(log.at("c:apply") < log.at("a:commit"), "{:?}", log.calls());
        assert!(log.at("b:apply") > log.at("a:commit"), "{:?}", log.calls());
        assert!(["a", "b", "c"].iter().all(|service| log.committed(service)));
    }

    /// A service on GPUs of its own whose plan does not fit starts once an
    /// update ends and its plan, made again then, fits.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_update_starts_once_its_plan_fits() {
        let log = Arc::new(Log::default());
        let first = [("rank-0", "GPU-0")];
        let second = [("rank-0", "GPU-1")];
        let fits_after_first: Planner = Box::new(move |log: &Log| {
            let refusal = (!log.committed("a"))
                .then_some("GPU-1 of rank-0 leaves the serving world's KV cache 0 bytes");
            world(&second, 0, refusal)
        });
        let nodes = vec![
            serve(daemon("a", &first, always(world(&first, 0, None)), &log)).await,
            serve(daemon("b", &second, fits_after_first, &log)).await,
        ];
        apply(&nodes, &options(Some(TRIAL))).await.unwrap();
        assert!(log.at("b:apply") > log.at("a:commit"), "{:?}", log.calls());
        let plans = log.calls().iter().filter(|call| *call == "b:plan").count();
        assert_eq!(plans, 2, "{:?}", log.calls());
    }

    /// The host storage of updates on one host adds up: an update starts
    /// beside another on the host only when both fit what it admits.
    #[tokio::test(flavor = "multi_thread")]
    async fn host_storage_of_updates_on_one_host_adds_up() {
        let log = Arc::new(Log::default());
        let gpus = [
            [("rank-0", "GPU-0")],
            [("rank-0", "GPU-1")],
            [("rank-0", "GPU-2")],
        ];
        let nodes = vec![
            serve(daemon(
                "a",
                &gpus[0],
                always(world(&gpus[0], 600, None)),
                &log,
            ))
            .await,
            serve(daemon(
                "b",
                &gpus[1],
                always(world(&gpus[1], 600, None)),
                &log,
            ))
            .await,
            serve(daemon(
                "c",
                &gpus[2],
                always(world(&gpus[2], 300, None)),
                &log,
            ))
            .await,
        ];
        apply(&nodes, &options(Some(TRIAL))).await.unwrap();
        // 600 and 300 fit within 1000 together; 600 and 600 do not.
        assert!(log.at("c:apply") < log.at("a:commit"), "{:?}", log.calls());
        assert!(log.at("b:apply") > log.at("a:commit"), "{:?}", log.calls());
    }

    /// A trial that rolled back stops the services that wait; the manager
    /// names them.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_rolled_back_trial_stops_the_services_that_wait() {
        let log = Arc::new(Log::default());
        let shared = [("rank-0", "GPU-0")];
        let rolling_back = Daemon {
            rolls_back: true,
            ..daemon("a", &shared, always(world(&shared, 0, None)), &log)
        };
        let waiting = serve(daemon("b", &shared, always(world(&shared, 0, None)), &log)).await;
        let nodes = vec![serve(rolling_back).await, waiting];
        let error = apply(&nodes, &options(Some(TRIAL)))
            .await
            .expect_err("a rolled-back trial fails the apply");
        let message = format!("{error:#}");
        assert!(message.contains("rolled back"), "{message}");
        assert!(message.contains(&nodes[1].base), "{message}");
        assert!(
            !log.calls().contains(&"b:apply".to_owned()),
            "{:?}",
            log.calls()
        );
        assert!(!log.committed("a"));
    }

    /// Without a commit, a new world stays on trial for the operator and
    /// holds its GPUs: a service that shares one waits for the next apply,
    /// and one on other GPUs updates.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_trial_left_for_the_operator_holds_its_gpus() {
        let log = Arc::new(Log::default());
        let shared = [("rank-0", "GPU-0")];
        let other = [("rank-1", "GPU-1")];
        let nodes = vec![
            serve(daemon("a", &shared, always(world(&shared, 0, None)), &log)).await,
            serve(daemon("b", &shared, always(world(&shared, 0, None)), &log)).await,
            serve(daemon("c", &other, always(world(&other, 0, None)), &log)).await,
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

    /// A plan that fits nowhere while no update is in progress is refused,
    /// naming its reason.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_plan_that_never_fits_is_refused() {
        let log = Arc::new(Log::default());
        let gpus = [("rank-0", "GPU-0")];
        let refusal = "GPU-0 of rank-0 leaves the serving world's KV cache 0 bytes";
        let plan = always(world(&gpus, 0, Some(refusal)));
        let nodes = vec![serve(daemon("a", &gpus, plan, &log)).await];
        let error = apply(&nodes, &options(Some(TRIAL)))
            .await
            .expect_err("a plan that does not fit is refused");
        assert!(format!("{error:#}").contains(refusal), "{error:#}");
        assert_eq!(log.calls(), ["a:plan"]);
    }

    /// The GPUs an update uses are those its service's worlds hold, those
    /// its storage plan prepares on, and its successor's target ranks;
    /// every refusal of its plans is kept, and the largest host storage a
    /// replica requires on a host is the update's.
    #[test]
    fn an_updates_needs_come_from_its_status_and_every_replicas_plan() {
        let status = json!({ "gpus": [{ "host": "rank-0", "gpu": "GPU-0" }] });
        let replica = |gpu: &str, target: &str, required: u64, feasible: bool| {
            json!({
                "world": true,
                "steps": [{ "step": "prepare" }],
                "storage": {
                    "feasible": true,
                    "gpus": [{ "host": "rank-0", "gpu": gpu }],
                    "hosts": [{ "host": "rank-0", "required": required, "admits": 900 }],
                },
                "redistribution": {
                    "feasible": feasible,
                    "refusal": (!feasible).then_some("GPU-9 has 0 bytes available"),
                    "ranks": [{ "host": "rank-1", "gpu": target }],
                },
            })
        };
        let planned = json!({
            "change": "planned",
            "plan": { "replicas": [
                replica("GPU-0", "GPU-8", 100, true),
                replica("GPU-1", "GPU-9", 300, false),
            ] },
        });
        let needs = Needs::of(&status, &planned);
        assert!(needs.changes && needs.world);
        let gpus: Vec<(&str, &str)> = needs
            .gpus
            .iter()
            .map(|(host, gpu)| (host.as_str(), gpu.as_str()))
            .collect();
        assert_eq!(
            gpus,
            [
                ("rank-0", "GPU-0"),
                ("rank-0", "GPU-1"),
                ("rank-1", "GPU-8"),
                ("rank-1", "GPU-9")
            ]
        );
        assert_eq!(
            needs.hosts,
            BTreeMap::from([("rank-0".to_owned(), (300, 900))])
        );
        assert_eq!(
            needs.refusal.as_deref(),
            Some("GPU-9 has 0 bytes available")
        );

        // A change in place starts no world, and one that changes nothing
        // needs nothing.
        let in_place = json!({
            "change": "planned",
            "plan": { "world": true, "daemon": false, "steps": null },
        });
        let needs = Needs::of(&status, &in_place);
        assert!(needs.changes && !needs.world);
        let unchanged = Needs::of(&status, &json!({ "change": "none" }));
        assert!(!unchanged.changes);
    }
}
