//! The Dynamo sidecar (`dynamo_config`): the leader daemon's registration of the deployment in
//! Dynamo, as Dynamo's sidecars register an engine that serves on its own (`lib/sidecar` of
//! `ai-dynamo/dynamo`).
//!
//! The daemon serves as it always does; the sidecar makes the deployment's replicas Dynamo's
//! workers over Dynamo's own planes, with Dynamo's default transports (etcd discovery, the TCP
//! request plane and ZMQ events, no NATS):
//!
//! - **Discovery.** Every worker holds an etcd lease of its own, whose id is its instance id,
//!   and registers under it its instance (`v1/instances/<namespace>/<component>/generate/<id>`,
//!   the TCP address of the sidecar's request plane), its model card (`v1/mdc/...`: the served
//!   model, its files, its page size and capacity, its worker type) and its event publishers
//!   (`v1/event_channels/...`, `v1/event_sources/...`). The lease is kept alive at a third of
//!   its time to live and revoked when the sidecar ends, which removes every key of the worker.
//! - **Workers.** With `workers = "replicas"`, each replica (the deployment's scheduler
//!   `<id>`) is one worker of the role `replicas.<id>.role` states: a prefill replica a
//!   `prefill` worker, a decode replica a `backend` worker of type `decode`, and a mixed one a
//!   `backend` worker of type `aggregated`. A frontend then chooses both replicas of a request
//!   itself, which the leader serves without its routing program. With `workers =
//!   "deployment"`, the deployment is one aggregated worker whose data-parallel ranks are its
//!   replicas, and the leader's routing program chooses within it.
//! - **Requests** ([`plane`]). A frontend sends a token request to a worker's endpoint. A prefill
//!   worker's card states a bootstrap (the sidecar's own address), so the frontend sends the
//!   decode worker its half of the request beside the prefill's, both naming one bootstrap room:
//!   the decode half is served as one request routed to both replicas, and the prefill half
//!   answers with the room and ends once that request's prompt is computed.
//! - **Events** ([`events`]). Each worker relays its replicas' KV events, which their
//!   schedulers publish in SGLang's format (`kv_events_config`), to its own publisher as
//!   Dynamo's KV events, and publishes each rank's KV pages in use as its load (`kv_metrics`).
//! - **Planning.** Dynamo's planner states how many prefill and decode workers it wants through
//!   its virtual connector (`v1/<namespace>/planner/...`). Each new decision becomes the roles
//!   of the replicas ([`roles`]), written as a change of the deployment by the writer `dynamo`,
//!   and acknowledged (`scaled_decision_id`) once the change converged; the workers are then
//!   registered again under their new roles. A replica the decision gives no role serves on as
//!   a mixed replica that Dynamo does not reach.

mod etcd;
mod events;
mod plane;
mod roles;
#[cfg(test)]
mod tests;
mod wire;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context as _, Result};
use proto::request::{Event, FinishReason, GenerationRequest, SamplingParams};
use scheduling::routing::Role;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::api::events::Events as _;
use crate::config::{DynamoConfig, ReplicaConfig, Workers};
use crate::leader::Leader;
use crate::programs::{Output, Programs};

/// The writer the revisions of the planner's decisions record.
pub(crate) const WRITER: &str = "dynamo";

/// How long a prefill half waits for its decode half's request to compute the prompt.
const PREFILL_WAIT: Duration = Duration::from_secs(120);

/// The sidecar of a serving deployment: its task, ended by [`Sidecar::stop`].
pub(crate) struct Sidecar {
    stop: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

impl Sidecar {
    /// Starts the sidecar of the deployment `leader` serves, which `settings` configure; `None`
    /// when they name no etcd. `programs` reads the current revision and writes the planner's
    /// decisions.
    pub(crate) fn start(
        settings: &DynamoConfig,
        leader: Arc<Leader>,
        programs: Arc<Programs>,
    ) -> Option<Self> {
        let endpoint = settings.etcd.as_deref()?;
        let core = Arc::new(Core {
            settings: settings.clone(),
            etcd: etcd::Etcd::new(endpoint),
            leader,
            programs,
            address: Mutex::new(String::new()),
            workers: Mutex::new(HashMap::new()),
            rooms: Mutex::new(HashMap::new()),
        });
        let stop = CancellationToken::new();
        let task = tokio::spawn(Arc::clone(&core).run(stop.clone()));
        Some(Self { stop, task })
    }

    /// Ends the sidecar: its workers leave discovery and their leases are revoked.
    pub(crate) async fn stop(self) {
        self.stop.cancel();
        let _ = self.task.await;
    }
}

/// What a worker is: the replicas it serves and what it does.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Plan {
    kind: wire::Kind,
    /// The deployment's schedulers it serves, as its data-parallel ranks in order.
    ranks: Vec<u32>,
}

/// A registered worker.
struct Worker {
    plan: Plan,
    /// Its instance id: its lease's.
    instance: u64,
    lease: i64,
    path: String,
    /// Ends its tasks: its lease's keep-alive and its relay.
    stop: CancellationToken,
    loads: tokio::sync::Mutex<events::Publisher>,
}

/// A disaggregated request's two halves, met by their bootstrap room.
#[derive(Default)]
struct Room {
    /// The prefill half, waiting for the prompt to be computed.
    prefilled: Option<oneshot::Sender<()>>,
    /// The prompt was computed before the prefill half arrived.
    done: bool,
}

struct Core {
    settings: DynamoConfig,
    etcd: etcd::Etcd,
    leader: Arc<Leader>,
    programs: Arc<Programs>,
    /// The request plane's address, `host:port`.
    address: Mutex<String>,
    /// The registered workers, by their endpoint's path.
    workers: Mutex<HashMap<String, Arc<Worker>>>,
    rooms: Mutex<HashMap<u64, Room>>,
}

impl Core {
    async fn run(self: Arc<Self>, stop: CancellationToken) {
        if let Err(error) = Arc::clone(&self).serve(&stop).await {
            warn!("the Dynamo sidecar ended: {error:#}");
        }
        let workers: Vec<Arc<Worker>> = service::lock(&self.workers)
            .drain()
            .map(|(_, worker)| worker)
            .collect();
        for worker in workers {
            self.retire(&worker).await;
        }
    }

    async fn serve(self: Arc<Self>, stop: &CancellationToken) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", self.settings.port))
            .await
            .context("binding the request plane")?;
        let port = listener.local_addr()?.port();
        let host = self.host();
        *service::lock(&self.address) = format!("{host}:{port}");
        info!(
            address = %service::lock(&self.address),
            etcd = self.settings.etcd.as_deref().unwrap_or_default(),
            namespace = self.settings.namespace,
            "the Dynamo sidecar serves"
        );
        let core = Arc::clone(&self);
        let dispatch: plane::Dispatch = Arc::new(move |incoming| core.dispatch(incoming));
        tokio::spawn(plane::serve(listener, dispatch, stop.clone()));

        let mut current = self.programs.subscribe();
        let mut handled: Option<i64> = None;
        loop {
            let decided = match self.plan_decision(&mut handled).await {
                Ok(decided) => decided,
                Err(error) => {
                    warn!("the planner's decision was not read: {error:#}");
                    false
                }
            };
            if let Err(error) = self.reconcile(decided).await {
                warn!("the Dynamo workers were not registered: {error:#}");
            }
            self.publish_loads().await;
            tokio::select! {
                () = stop.cancelled() => return Ok(()),
                _ = current.changed() => {}
                () = tokio::time::sleep(self.settings.period) => {}
            }
        }
    }

    /// The address the frontends reach the sidecar at: `host`, or the leader's host name.
    fn host(&self) -> String {
        self.settings.host.clone().unwrap_or_else(|| {
            rustix::system::uname()
                .nodename()
                .to_string_lossy()
                .into_owned()
        })
    }

    /// The roles of the deployment's schedulers in the current revision, by number.
    fn current_roles(&self, numbers: &[u32]) -> BTreeMap<u32, Role> {
        let current = self.programs.current();
        let stated = &current.package.config.replicas;
        numbers
            .iter()
            .map(|number| {
                let role = stated
                    .get(&number.to_string())
                    .map_or(Role::Mixed, |replica| replica.role);
                (*number, role)
            })
            .collect()
    }

    /// The workers the deployment is now: with `decided`, a planner's decision stands, and a
    /// mixed replica it gave no role is no worker.
    fn plans(&self, decided: bool) -> Vec<Plan> {
        let numbers: Vec<u32> = self
            .leader
            .numbered()
            .iter()
            .map(|numbered| numbered.number)
            .collect();
        if numbers.is_empty() {
            return Vec::new();
        }
        match self.settings.workers {
            Workers::Deployment => vec![Plan {
                kind: wire::Kind::Aggregated,
                ranks: numbers,
            }],
            Workers::Replicas => self
                .current_roles(&numbers)
                .into_iter()
                .filter_map(|(number, role)| {
                    let kind = match role {
                        Role::Prefill => wire::Kind::Prefill,
                        Role::Decode => wire::Kind::Decode,
                        Role::Mixed if decided => return None,
                        Role::Mixed => wire::Kind::Aggregated,
                    };
                    Some(Plan {
                        kind,
                        ranks: vec![number],
                    })
                })
                .collect(),
        }
    }

    /// Registers the workers the deployment is now and retires the others.
    async fn reconcile(&self, decided: bool) -> Result<()> {
        let plans = self.plans(decided);
        let stale: Vec<Arc<Worker>> = {
            let mut workers = service::lock(&self.workers);
            let paths: Vec<String> = workers
                .iter()
                .filter(|(_, worker)| !plans.contains(&worker.plan))
                .map(|(path, _)| path.clone())
                .collect();
            paths
                .into_iter()
                .filter_map(|path| workers.remove(&path))
                .collect()
        };
        for worker in stale {
            self.retire(&worker).await;
        }
        for plan in plans {
            let registered = service::lock(&self.workers)
                .values()
                .any(|worker| worker.plan == plan);
            if !registered {
                let worker = self.register(plan).await?;
                service::lock(&self.workers).insert(worker.path.clone(), worker);
            }
        }
        Ok(())
    }

    /// Registers the worker of `plan` under a lease of its own.
    async fn register(&self, plan: Plan) -> Result<Arc<Worker>> {
        let namespace = &self.settings.namespace;
        let ttl = self.settings.lease_ttl.as_secs().max(1);
        let lease = self.etcd.grant(ttl).await?;
        let instance = u64::try_from(lease).context("etcd granted a negative lease")?;
        let component = plan.kind.component();
        let host = self.host();
        let address = service::lock(&self.address).clone();
        let numbered = self.leader.numbered();
        let served: Vec<_> = plan
            .ranks
            .iter()
            .filter_map(|number| numbered.iter().find(|numbered| numbered.number == *number))
            .collect();
        let first = served.first().context("a worker of no serving replica")?;
        let engine = first
            .engine
            .clone()
            .context("a replica serves no engine configuration")?;
        let engine = &engine;
        let block_size = engine.kv_cache.block_tokens();

        // The events: each rank's KV events, relayed from its scheduler's publisher by a
        // publisher of its own, and the worker's loads.
        let stop = CancellationToken::new();
        let loads = events::Publisher::bind(wire::KV_METRICS, &host).await?;
        let mut keys = vec![(
            wire::event_key(
                "event_channels",
                namespace,
                component,
                loads.topic,
                loads.id,
            ),
            wire::channel(namespace, component, loads.topic, loads.id, &loads.endpoint),
        )];
        let mut relays = Vec::new();
        for (numbered, rank) in served.iter().zip(0u32..) {
            let Some(source) = source_of(numbered, rank) else {
                continue;
            };
            let kv = events::Publisher::bind(wire::KV_EVENTS, &host).await?;
            keys.push((
                wire::event_key("event_channels", namespace, component, kv.topic, kv.id),
                wire::channel(namespace, component, kv.topic, kv.id, &kv.endpoint),
            ));
            keys.push((
                wire::event_key("event_sources", namespace, component, kv.topic, kv.id),
                wire::source(namespace, component, kv.id, instance, rank),
            ));
            relays.push((kv, source));
        }

        // The card, then the instance, which makes the worker reachable last.
        let info = first.info.clone();
        let total_blocks = info
            .as_ref()
            .and_then(|info| info.segments.iter().map(|group| group.units).max())
            .map(|units| u64::from(units.saturating_sub(1)) * served.len() as u64);
        let bootstrap_port = address
            .rsplit_once(':')
            .and_then(|(_, port)| port.parse::<u16>().ok())
            .unwrap_or_default();
        let model = self.leader.served_model_name();
        let card = wire::Card {
            namespace,
            instance,
            kind: plan.kind,
            model: &model,
            source: &engine.model.model,
            context_length: engine.model.max_model_len,
            block_size,
            total_blocks,
            max_num_seqs: engine.max_running_requests.get() as u64,
            max_num_batched_tokens: engine
                .max_num_batched_tokens
                .map(|tokens| tokens.get() as u64),
            ranks: u32::try_from(plan.ranks.len()).unwrap_or(u32::MAX),
            bootstrap: (plan.kind == wire::Kind::Prefill)
                .then_some((host.as_str(), bootstrap_port)),
            files: card_files(Path::new(&engine.model.model)),
        };
        keys.push((
            wire::card_key(namespace, component, instance),
            card.record(),
        ));
        keys.push((
            wire::instance_key(namespace, component, instance),
            wire::instance(namespace, component, instance, &address),
        ));
        for (key, value) in &keys {
            self.etcd.create(key, value, lease).await?;
        }

        // The lease lives while the sidecar does.
        let keeper = self.etcd.clone();
        let period = Duration::from_secs((ttl / 3).max(1));
        let keeping = stop.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = keeping.cancelled() => return,
                    () = tokio::time::sleep(period) => {}
                }
                match keeper.keep_alive(lease).await {
                    Ok(ttl) if ttl > 0 => {}
                    Ok(_) => warn!(lease, "the worker's lease expired"),
                    Err(error) => warn!(lease, "the worker's lease was not kept alive: {error:#}"),
                }
            }
        });
        for (kv, source) in relays {
            tokio::spawn(events::relay(
                kv,
                vec![source],
                instance,
                block_size,
                stop.clone(),
            ));
        }

        let path = wire::endpoint_path(instance, namespace, component);
        info!(
            instance = format!("{instance:x}"),
            component,
            ranks = ?plan.ranks,
            "a Dynamo worker is registered"
        );
        Ok(Arc::new(Worker {
            plan,
            instance,
            lease,
            path,
            stop,
            loads: tokio::sync::Mutex::new(loads),
        }))
    }

    /// Takes `worker` out of discovery: its instance first, then its lease and every key on it.
    async fn retire(&self, worker: &Worker) {
        worker.stop.cancel();
        let key = wire::instance_key(
            &self.settings.namespace,
            worker.plan.kind.component(),
            worker.instance,
        );
        if let Err(error) = self.etcd.delete(&key).await {
            warn!("a worker's instance was not deleted: {error:#}");
        }
        if let Err(error) = self.etcd.revoke(worker.lease).await {
            warn!("a worker's lease was not revoked: {error:#}");
        }
        info!(
            instance = format!("{:x}", worker.instance),
            "a Dynamo worker is retired"
        );
    }

    /// Publishes each worker's KV pages in use, rank by rank.
    async fn publish_loads(&self) {
        let numbered = self.leader.numbered();
        let totals: HashMap<u32, u64> = numbered
            .iter()
            .filter_map(|numbered| {
                let units = numbered
                    .info
                    .as_ref()?
                    .segments
                    .iter()
                    .map(|group| group.units)
                    .max()?;
                Some((numbered.number, u64::from(units.saturating_sub(1))))
            })
            .collect();
        let rows = self
            .leader
            .recorder()
            .store()
            .query(
                "SELECT labels['replica'] AS replica, labels['scheduler'] AS scheduler, value \
                 FROM stats WHERE name = 'uniserve_kv_cache_usage_ratio' AND time = \
                 (SELECT max(time) FROM stats WHERE name = 'uniserve_kv_cache_usage_ratio')",
            )
            .await;
        let rows = match rows.and_then(|rows| rows.json()) {
            Ok(json) => serde_json::from_slice(&json).unwrap_or(Value::Null),
            Err(_) => Value::Null,
        };
        let used = events::used_blocks(&rows, &totals);
        let workers: Vec<Arc<Worker>> = service::lock(&self.workers).values().cloned().collect();
        for worker in workers {
            let loads: Vec<wire::Load> = worker
                .plan
                .ranks
                .iter()
                .zip(0u32..)
                .map(|(number, rank)| wire::Load {
                    worker_id: worker.instance,
                    dp_rank: rank,
                    active_decode_blocks: None,
                    active_prefill_tokens: None,
                    kv_used_blocks: Some(used.get(number).copied().unwrap_or(0)),
                })
                .collect();
            events::publish_loads(&mut *worker.loads.lock().await, &loads).await;
        }
    }

    /// Reads the planner's decision and, when it is one not yet acknowledged, writes the roles
    /// it makes and acknowledges it once they converged. Returns whether a decision stands.
    async fn plan_decision(&self, handled: &mut Option<i64>) -> Result<bool> {
        if self.settings.workers != Workers::Replicas {
            return Ok(false);
        }
        let prefix = wire::planner_prefix(&self.settings.namespace);
        let mut values: HashMap<String, i64> = HashMap::new();
        for (key, value) in self.etcd.range(&prefix).await? {
            let Some(name) = key.strip_prefix(&prefix) else {
                continue;
            };
            if let Ok(number) = String::from_utf8_lossy(&value).trim().parse() {
                values.insert(name.to_owned(), number);
            }
        }
        let Some(&decision) = values.get(wire::DECISION) else {
            return Ok(false);
        };
        let scaled = values.get(wire::SCALED).copied().unwrap_or(-1);
        if decision < 0 {
            return Ok(false);
        }
        if scaled >= decision || *handled == Some(decision) {
            return Ok(true);
        }
        *handled = Some(decision);

        let numbers: Vec<u32> = self
            .leader
            .numbered()
            .iter()
            .map(|numbered| numbered.number)
            .collect();
        let current = self.current_roles(&numbers);
        let prefill = values.get(wire::PREFILL_WORKERS).copied().unwrap_or(-1);
        let decode = values.get(wire::DECODE_WORKERS).copied().unwrap_or(-1);
        let target = roles::assign(&current, prefill, decode);
        info!(decision, prefill, decode, roles = ?target, "the planner decided");
        if target != current {
            self.write_roles(&target)
                .await
                .map_err(anyhow::Error::msg)?;
        }
        self.etcd
            .put(
                &format!("{prefix}{}", wire::SCALED),
                decision.to_string().as_bytes(),
                None,
            )
            .await?;
        Ok(true)
    }

    /// Writes `roles` as the next revision of the deployment, by the writer `dynamo`: the
    /// revision it converged to, or why it did not.
    async fn write_roles(&self, roles: &BTreeMap<u32, Role>) -> Result<u64, String> {
        let current = self.programs.current();
        let input = current.package.config.clone();
        let mut output = input.clone();
        for (number, role) in roles {
            output
                .replicas
                .insert(number.to_string(), ReplicaConfig { role: *role });
        }
        let (done, result) = oneshot::channel();
        self.programs
            .writer()
            .send(Output {
                writer: WRITER.to_owned(),
                input,
                output,
                done,
            })
            .map_err(|_| "the deployment takes no more changes".to_owned())?;
        result
            .await
            .map_err(|_| "the change was dropped".to_owned())?
    }

    /// Takes a request a frontend sent: its worker serves it on a task of its own.
    fn dispatch(self: &Arc<Self>, incoming: plane::Incoming) -> Result<(), String> {
        let worker = service::lock(&self.workers)
            .get(&incoming.path)
            .cloned()
            .ok_or_else(|| format!("no worker serves `{}`", incoming.path))?;
        let core = Arc::clone(self);
        tokio::spawn(async move {
            let (outputs, received) = mpsc::channel(64);
            let cancel = CancellationToken::new();
            let control = incoming.control.clone();
            let responding = cancel.clone();
            let respond =
                tokio::spawn(async move { plane::respond(&control, received, responding).await });
            core.handle(&worker, incoming.request, outputs, cancel)
                .await;
            match respond.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => warn!("a response stream failed: {error:#}"),
                Err(error) => warn!("a response stream did not end: {error}"),
            }
        });
        Ok(())
    }

    /// Serves `request` on `worker`, its outputs into `outputs`, until it ends or `cancel`.
    async fn handle(
        &self,
        worker: &Worker,
        request: wire::Request,
        outputs: mpsc::Sender<Value>,
        cancel: CancellationToken,
    ) {
        if worker.plan.kind == wire::Kind::Prefill {
            self.prefill(&request, &outputs, &cancel).await;
            return;
        }

        // The replicas of the request: the deployment's routing program's choice for a
        // deployment worker; otherwise the frontend's prefill worker, when it chose one, and
        // this worker's replica.
        let decode = worker.plan.ranks[0];
        let route = match self.settings.workers {
            Workers::Deployment => None,
            Workers::Replicas => {
                let prefill = request
                    .routing
                    .as_ref()
                    .and_then(|routing| routing.prefill_worker_id)
                    .and_then(|id| self.worker_of(id))
                    .map_or(decode, |worker| worker.plan.ranks[0]);
                Some((prefill, decode))
            }
        };
        if let Some((prefill, decode)) = route
            && let Err(reason) = self.check_roles(prefill, decode)
        {
            let _ = outputs.send(wire::error(&reason)).await;
            return;
        }

        let room = request.room();
        let generation = generation_of(&request);
        let submitted = match route {
            Some((prefill, decode)) => {
                self.leader
                    .submit_routed(generation.into(), prefill, decode)
                    .await
            }
            None => self
                .leader
                .submit(generation.into(), None)
                .await
                .map(|(events, _)| events),
        };
        let mut events = match submitted {
            Ok(events) => events,
            Err(error) => {
                let _ = outputs.send(wire::error(&error.to_string())).await;
                return;
            }
        };

        let mut prefilled = room;
        loop {
            let event = tokio::select! {
                () = cancel.cancelled() => return,
                event = events.recv() => event,
            };
            let Some(event) = event else {
                let _ = outputs
                    .send(wire::error("the request ended with no outcome"))
                    .await;
                return;
            };
            let (output, last) = match event {
                Event::TextToken { id, .. } => (json!({"token_ids": [id]}), false),
                Event::Finished {
                    reason,
                    prompt_tokens,
                    completion_tokens,
                    ..
                } => (
                    json!({
                        "token_ids": [],
                        "finish_reason": finish(reason),
                        "completion_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }),
                    true,
                ),
                Event::Rejected { .. } | Event::Error { .. } => {
                    let _ = outputs.send(wire::error(&format!("{event:?}"))).await;
                    return;
                }
                _ => continue,
            };
            // The prompt is computed once the first token arrives: the prefill half ends.
            if let Some(room) = prefilled.take() {
                self.prefilled(room);
            }
            if outputs.send(wire::output(output)).await.is_err() || last {
                return;
            }
        }
    }

    /// The prefill half of a disaggregated request: answers with its bootstrap room, then ends
    /// once its decode half's request computed the prompt.
    async fn prefill(
        &self,
        request: &wire::Request,
        outputs: &mpsc::Sender<Value>,
        cancel: &CancellationToken,
    ) {
        let address = service::lock(&self.address).clone();
        let (host, port) = address.rsplit_once(':').unwrap_or((&address, "0"));
        let room = request
            .room()
            .unwrap_or_else(|| events::publisher_id() & (i64::MAX as u64));
        let bootstrap = json!({
            "bootstrap_host": host,
            "bootstrap_port": port.parse::<u16>().unwrap_or_default(),
            "bootstrap_room": room,
        });
        let first = json!({"token_ids": [], "disaggregated_params": bootstrap});
        if outputs.send(wire::output(first)).await.is_err() {
            return;
        }

        let (sender, prefilled) = oneshot::channel();
        let done = {
            let mut rooms = service::lock(&self.rooms);
            let entry = rooms.entry(room).or_default();
            if entry.done {
                rooms.remove(&room);
                true
            } else {
                entry.prefilled = Some(sender);
                false
            }
        };
        if !done {
            tokio::select! {
                () = cancel.cancelled() => {}
                _ = prefilled => {}
                () = tokio::time::sleep(PREFILL_WAIT) => {}
            }
            service::lock(&self.rooms).remove(&room);
        }
        let prompt = request.token_ids.len();
        let last = json!({
            "token_ids": [],
            "finish_reason": "length",
            "completion_usage": {"prompt_tokens": prompt, "completion_tokens": 0, "total_tokens": prompt},
        });
        let _ = outputs.send(wire::output(last)).await;
    }

    /// The prompt of the request of `room` is computed: its prefill half ends.
    fn prefilled(&self, room: u64) {
        let mut rooms = service::lock(&self.rooms);
        let entry = rooms.entry(room).or_default();
        match entry.prefilled.take() {
            Some(waiting) => {
                let _ = waiting.send(());
                rooms.remove(&room);
            }
            None => entry.done = true,
        }
    }

    /// The registered worker of instance `id`.
    fn worker_of(&self, id: u64) -> Option<Arc<Worker>> {
        service::lock(&self.workers)
            .values()
            .find(|worker| worker.instance == id)
            .cloned()
    }

    /// Whether replica `prefill` computes prompts and replica `decode` generates tokens, as the
    /// current revision states their roles.
    fn check_roles(&self, prefill: u32, decode: u32) -> Result<(), String> {
        let roles = self.current_roles(&[prefill, decode]);
        if !roles[&prefill].prefills() {
            return Err(format!("replica {prefill} computes no prompts"));
        }
        if !roles[&decode].decodes() {
            return Err(format!("replica {decode} generates no tokens"));
        }
        Ok(())
    }
}

/// Where the scheduler `numbered` publishes its KV events (`kv_events_config`), as rank `rank`
/// of its worker relays them: its engine's endpoint offset by the scheduler's number, reached at
/// the engine's host when it binds every interface; `None` when it publishes none.
fn source_of(numbered: &crate::leader::Numbered, rank: u32) -> Option<events::Source> {
    let engine = numbered.engine.as_ref()?;
    let events = &engine.kv_events;
    let endpoint =
        proto::config::KvEventsConfig::offset(events.endpoint.as_deref()?, numbered.number)?;
    let endpoint = endpoint.replacen("0.0.0.0", &engine.parallel.engine_host, 1);
    Some(events::Source {
        endpoint,
        topic: events.topic.clone(),
        rank,
    })
}

/// The finish reason Dynamo states for how a request ended.
fn finish(reason: FinishReason) -> Value {
    match reason {
        FinishReason::Eos => json!("eos"),
        FinishReason::MaxTokens | FinishReason::CacheCapacity => json!("length"),
        FinishReason::Completed | FinishReason::Stop | FinishReason::Repetition => json!("stop"),
        FinishReason::Cancelled | FinishReason::Aborted => json!("cancelled"),
        FinishReason::Error => json!({"error": "the request failed"}),
    }
}

/// The token request of a frontend's request: its prompt, its sampling and its stops; the
/// end-of-sequence tokens stop it unless it ignores them.
fn generation_of(request: &wire::Request) -> GenerationRequest {
    let sampling = &request.sampling_options;
    let stops = &request.stop_conditions;
    let ignore_eos = stops.ignore_eos.unwrap_or(false);
    let defaults = SamplingParams::default();
    let mut stop_token_ids = stops.stop_token_ids.clone().unwrap_or_default();
    if !ignore_eos {
        stop_token_ids.extend(&request.eos_token_ids);
    }
    GenerationRequest {
        request_id: proto::request::RequestId::default(),
        trace: 0,
        prompt_token_ids: request.token_ids.clone(),
        multimodal_inputs: Default::default(),
        sampling: SamplingParams {
            temperature: sampling.temperature.unwrap_or(defaults.temperature),
            top_k: sampling
                .top_k
                .and_then(|top_k| u32::try_from(top_k).ok())
                .unwrap_or(defaults.top_k),
            top_p: sampling.top_p.unwrap_or(defaults.top_p),
            ignore_eos,
            seed: sampling.seed.map(|seed| seed as u64),
            min_p: sampling.min_p.unwrap_or(defaults.min_p),
            repetition_penalty: sampling
                .repetition_penalty
                .unwrap_or(defaults.repetition_penalty),
            frequency_penalty: sampling
                .frequency_penalty
                .unwrap_or(defaults.frequency_penalty),
            presence_penalty: sampling
                .presence_penalty
                .unwrap_or(defaults.presence_penalty),
            min_tokens: stops.min_tokens.unwrap_or(0) as usize,
            ..defaults
        },
        max_und_tokens: stops
            .max_tokens
            .map_or(usize::MAX, |tokens| tokens as usize),
        stop_strings: Vec::new(),
        stop_token_ids,
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        generated_token_ids: Vec::new(),
        adapter: 0,
        hand_off: false,
        lend: None,
    }
}

/// The files of the checkpoint in `directory` a model card names for a frontend to process
/// requests with: its configuration, tokenizer, tokenizer configuration and generation
/// configuration, each with its BLAKE3 checksum and size, those it holds.
fn card_files(directory: &Path) -> serde_json::Map<String, Value> {
    let mut files = serde_json::Map::new();
    for (field, kind, name) in [
        ("model_info", "hf_config_json", "config.json"),
        ("tokenizer", "hf_tokenizer_json", "tokenizer.json"),
        (
            "prompt_formatter",
            "hf_tokenizer_config_json",
            "tokenizer_config.json",
        ),
        (
            "gen_config",
            "hf_generation_config_json",
            "generation_config.json",
        ),
    ] {
        let path = directory.join(name);
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let checked = json!({
            "path": path.to_string_lossy(),
            "checksum": format!("blake3:{}", blake3::hash(&bytes).to_hex()),
            "size": bytes.len(),
        });
        files.insert(field.to_owned(), json!({ kind: checked }));
    }
    files
}
