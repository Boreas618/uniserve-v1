//! The node: what one daemon process serves on its listeners, through one path whether it serves
//! no deployment, the deployment its command line configured, a deployment a `PUT /admin/config`
//! brought up, or a deployment a hot restart's predecessor handed it.
//!
//! The node binds its listeners when it starts ([`super::serve`]): the client API's address and
//! the admin API's two Unix sockets in its instance directory. It serves them for its whole life,
//! dispatching every request to the deployment that serves: the client API to its routes, `503`
//! while none serves; the admin sockets to its admin routes, beside the node's own (`GET
//! /admin/node`, and the `PUT /admin/config` that brings a deployment up while none serves).
//!
//! A deployment comes up in one step whatever starts it ([`Node::deploy`]): its daemon, its
//! engines and its first epoch, from a configuration. A hot restart's successor adopts the
//! deployment instead, and hands the node what it adopted ([`First::Serving`]). From there the
//! deployment is served the same way ([`Node::serve`]): its routes, its readiness, its restarts,
//! and its end, which drains its requests within the shutdown budget, shuts its engines down,
//! stops its daemon and, when its `POST /admin/shutdown` asked for it, releases its storage at
//! the host's Storage Server; or, once a successor serves it, detaches.
//!
//! How long the node lives is stated by how it was started ([`Lifetime`]): a node started for a
//! deployment ends with it; a node started without one serves the deployments `PUT /admin/config`
//! brings up, one at a time, until it is stopped.

use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;
use tracing::{info, warn};

use super::admin::{Admin, AdminState};
use super::handoff::Handoff;
use super::listener::{Listener, ListenerHandoff};
use super::{Ready, middleware, retire_deployment, router, video_body_limit};
use crate::api::openai::ApiError;
use crate::api::openai::video_jobs::VideoJobs;
use crate::leader::Leader;
use crate::options::{AdminConfig, ServerArgs, ServerConfig, Stated, resolve_text};
use crate::restart::Restarter;
use crate::supervision::Daemon;

/// How long a node lives, which how it was started states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Lifetime {
    /// Started for the deployment its configuration names (`uniserve serve MODEL`): the node
    /// ends with that deployment, at its `POST /admin/shutdown` or at a stop signal.
    Deployment,
    /// Started without a deployment (`uniserve serve`): the node serves the deployments `PUT
    /// /admin/config` brings up, one at a time, and serves none between them, until a stop
    /// signal.
    Deployments,
}

/// What the node serves.
enum Serving {
    /// No deployment.
    Idle,
    /// A deployment is coming up.
    Starting(String),
    /// A deployment serves through its routes.
    Deployment(Arc<Routes>),
    /// A deployment is coming down.
    Stopping(String),
}

/// A serving deployment's routes: the client API's, the operator's admin routes and the
/// harness's.
struct Routes {
    name: String,
    client: Router,
    admin: Router,
    harness: Router,
}

/// The state of a node, as `GET /admin/node` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NodeState {
    Idle,
    Starting,
    Serving,
    Stopping,
}

/// What `GET /admin/node` answers: the node's state, the deployment it is about, and why its
/// last deployment failed to start or to stop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct NodeStatus {
    pub(crate) state: NodeState,
    pub(crate) deployment: Option<String>,
    pub(crate) last_error: Option<String>,
}

/// A deployment brought up and not yet served: what [`Node::serve`] serves, whether the node's
/// own bring-up made it ([`Node::deploy`]) or a hot restart's successor adopted it.
pub(crate) struct Brought {
    /// The deployment's configuration and the paths it stated.
    pub(crate) config: ServerArgs,
    pub(crate) stated: Stated,
    pub(crate) leader: Arc<Leader>,
    /// The deployment's admin state, whose shutdown cancels `token`.
    pub(crate) admin: Arc<Admin>,
    pub(crate) jobs: Arc<VideoJobs>,
    /// The responses a hot restart's predecessor handed over, which the deployment goes on
    /// with; none for a deployment the node brought up.
    pub(crate) adopted: Vec<crate::restart::streams::Adopted>,
    /// The deployment's daemon service; `None` for a deployment served without one (the tests'
    /// in-process engines).
    pub(crate) daemon: Option<Daemon>,
    /// Set by the deployment's `POST /admin/shutdown`: its storage is released once its
    /// processes are down.
    pub(crate) retire: Arc<AtomicBool>,
    /// The deployment's end ([`Node::deployment_token`]).
    pub(crate) token: CancellationToken,
}

/// What a node serves first.
pub(crate) enum First {
    /// No deployment: the node waits for a `PUT /admin/config`.
    Idle,
    /// The deployment of `config`, brought up as a `PUT /admin/config` brings one up; `join` is
    /// the join listener the node bound at start for the deployment's other nodes, with the
    /// endpoint it reported.
    Configured {
        config: Box<ServerArgs>,
        stated: Stated,
        join: Option<(::service::Listener, ::service::Endpoint)>,
    },
    /// A deployment already brought up: a hot restart's successor's adoption.
    Serving(Box<Brought>),
}

/// What a node is started with: its model families, its own options, its instance directory and
/// readiness, its lifetime, and the stop signal that ends it.
pub(crate) struct Start {
    pub(crate) families: processing::Registry,
    /// The node's own options: its HTTP server's (`server_config`), its admin API's
    /// (`admin_config`) and the Build Server a deployment that states none prepares its modules at
    /// (`build_config`).
    pub(crate) options: ServerArgs,
    pub(crate) instance: service::Instance,
    pub(crate) ready: Ready,
    pub(crate) lifetime: Lifetime,
    pub(crate) shutdown: CancellationToken,
}

/// The token whose cancellation ends a deployment of a node whose lifetime is `lifetime` and
/// whose stop is `shutdown`: the node's stop itself for a [`Lifetime::Deployment`] node, whose
/// deployment's end is its end, else a child of it, so that the node's stop ends the deployment
/// as well.
pub(crate) fn deployment_token(
    lifetime: Lifetime,
    shutdown: &CancellationToken,
) -> CancellationToken {
    match lifetime {
        Lifetime::Deployment => shutdown.clone(),
        Lifetime::Deployments => shutdown.child_token(),
    }
}

/// How a node ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ended {
    /// Whether a hot restart's successor took the node over: its instance directory and the
    /// deployment's engines, service and storage are the successor's now.
    pub(crate) handed_over: bool,
}

/// The node's state.
pub(crate) struct Node {
    families: processing::Registry,
    server: ServerConfig,
    admin: AdminConfig,
    build: build_server::command::Settings,
    instance: PathBuf,
    lifetime: Lifetime,
    ready: Mutex<Ready>,
    /// The responses in flight a hot restart pauses and hands over, and the gate that admits
    /// requests until it does: the node's, since a restart hands the whole node over.
    handoff: Arc<Handoff>,
    /// The client listener's descriptors and what a successor needs of them, which each
    /// deployment's restarter hands over.
    listener: (ListenerHandoff, Vec<OwnedFd>),
    /// Cancelled once a hot restart's successor serves, or the node ends: the node stops
    /// accepting connections.
    accepting: CancellationToken,
    /// Cancelled when the node ends.
    shutdown: CancellationToken,
    serving: Mutex<Serving>,
    /// Why the last deployment failed to start or to stop.
    last_error: Mutex<Option<String>>,
    /// The deployment a `PUT /admin/config` brought up, while it runs.
    current: Mutex<Option<tokio::task::JoinHandle<Result<()>>>>,
}

/// Runs a node started with `start` on `listener` and `admins`, serving `first` first, until it
/// ends: at its stop signal; with its deployment, for a node whose [`Lifetime`] is the
/// deployment's; or once a hot restart's successor serves it.
///
/// At a stop, the node stops accepting connections, its deployment drains its requests within
/// the shutdown budget (`server_config.shutdown_timeout`) and comes down, and the node waits for
/// its connections for what is left of the budget. Once a successor serves, the node finishes
/// the requests and video jobs it took, however long they take, and its deployment detaches.
///
/// # Errors
///
/// Fails when the listener cannot be duplicated for a hot restart, the node's own deployment
/// (`First::Configured`, a [`Lifetime::Deployment`] node's) fails to come up or to come down, or
/// the HTTP server fails.
pub(crate) async fn run(
    start: Start,
    mut listener: Listener,
    admins: super::admin::Listeners,
    first: First,
) -> Result<Ended> {
    let Start {
        families,
        options,
        instance,
        ready,
        lifetime,
        shutdown,
    } = start;
    let node = Arc::new(Node {
        families,
        server: options.server_config.clone(),
        admin: options.admin_config.clone(),
        build: options.build_config.clone(),
        instance: instance.path().to_owned(),
        lifetime,
        ready: Mutex::new(ready),
        handoff: Arc::new(Handoff::new()),
        listener: listener
            .hand_over()
            .context("duplicating the client listener")?,
        accepting: CancellationToken::new(),
        shutdown,
        serving: Mutex::new(Serving::Idle),
        last_error: Mutex::new(None),
        current: Mutex::new(None),
    });

    super::admin::serve_routes(
        admins,
        node.admin_routes(),
        node.harness_routes(),
        &node.shutdown,
    );
    let mut server = tokio::spawn(super::conn::serve(
        listener,
        super::conn::Server {
            app: node.client_routes(),
            limits: super::conn::Limits::of(&node.server),
            admitting: node.handoff.admissions(),
        },
        node.accepting.clone(),
    ));
    // The node's own deployment, whose end is the node's.
    let mut own = match first {
        First::Idle => {
            info!("the node serves no deployment until PUT /admin/config");
            service::lock(&node.ready).report(service::State::Serving);
            None
        }
        First::Configured {
            config,
            stated,
            join,
        } => Some(tokio::spawn(
            Arc::clone(&node).deploy(*config, stated, join),
        )),
        First::Serving(brought) => Some(tokio::spawn(Arc::clone(&node).serve(*brought))),
    };

    // The node ends at its stop, when a successor takes it over, or with its own deployment.
    let own_ended = async {
        match own.as_mut() {
            Some(task) => task.await,
            None => std::future::pending().await,
        }
    };
    let failed = tokio::select! {
        () = node.shutdown.cancelled() => None,
        () = node.accepting.cancelled() => None,
        ended = own_ended => {
            own = None;
            match ended.context("the node's deployment task did not complete") {
                Ok(Ok(())) => None,
                Ok(Err(error)) | Err(error) => Some(error),
            }
        }
    };

    let handed_over = node.accepting.is_cancelled() && !node.shutdown.is_cancelled();
    let served = if handed_over {
        // The successor serves: this daemon finishes the requests it took, however long they
        // take, and then the deployment detaches.
        let served = server
            .await
            .context("the HTTP server's task did not complete")?;
        node.shutdown.cancel();
        served.context("HTTP server failed")
    } else {
        let deadline = tokio::time::Instant::now() + node.server.shutdown_timeout;
        node.accepting.cancel();
        node.shutdown.cancel();
        // The deployment drains within the same budget before its engines come down.
        node.end_current().await;
        if let Some(task) = own.take()
            && let Err(error) = task
                .await
                .context("the node's deployment task did not complete")
                .and_then(|ended| ended)
        {
            warn!("the node's deployment did not come down cleanly: {error:#}");
        }
        match tokio::time::timeout_at(deadline, &mut server).await {
            Ok(served) => served
                .context("the HTTP server's task did not complete")?
                .context("HTTP server failed"),
            Err(_) => {
                warn!(
                    timeout = ?node.server.shutdown_timeout,
                    "graceful shutdown deadline elapsed; the connections still open close"
                );
                server.abort();
                Ok(())
            }
        }
    };
    if handed_over {
        node.end_current().await;
        if let Some(task) = own.take() {
            task.await
                .context("the node's deployment task did not complete")??;
        }
        // A successor that took the node over owns the instance directory from then on.
        instance.hand_over();
    }
    if let Some(error) = failed {
        return Err(error);
    }
    served.map(|()| Ended { handed_over })
}

impl Node {
    /// The token whose cancellation ends a deployment of this node ([`deployment_token`]).
    fn deployment_token(&self) -> CancellationToken {
        deployment_token(self.lifetime, &self.shutdown)
    }

    fn set(&self, serving: Serving) {
        *service::lock(&self.serving) = serving;
    }

    /// The deployment's routes while one serves.
    fn routes(&self) -> Option<Arc<Routes>> {
        match &*service::lock(&self.serving) {
            Serving::Deployment(routes) => Some(Arc::clone(routes)),
            Serving::Idle | Serving::Starting(_) | Serving::Stopping(_) => None,
        }
    }

    /// Waits for the deployment a `PUT /admin/config` brought up, after the node's stop ended it.
    async fn end_current(&self) {
        let current = service::lock(&self.current).take();
        if let Some(task) = current
            && let Err(error) = task
                .await
                .context("the deployment's task did not complete")
                .and_then(|ended| ended)
        {
            warn!("the deployment did not come down cleanly: {error:#}");
        }
    }

    /// Brings the deployment of `config` up and serves it until it ends; `join` is the join
    /// listener the node bound for its other nodes at start, and the endpoint it reported, or
    /// `None` to bind one now when the configuration names a join endpoint
    /// (`dist_init_addr`).
    ///
    /// The deployment's daemon starts first and, on a deployment of several nodes, waits until
    /// every other node's daemon joined; its engines and first epoch follow. A stop while it
    /// comes up ends the bring-up, stopping what started.
    ///
    /// # Errors
    ///
    /// Fails when the daemon or the engines cannot start, or the deployment fails to come down.
    async fn deploy(
        self: Arc<Self>,
        config: ServerArgs,
        stated: Stated,
        join: Option<(::service::Listener, ::service::Endpoint)>,
    ) -> Result<()> {
        let name = config.served_model_name().to_owned();
        self.set(Serving::Starting(name.clone()));
        let brought = match self.bring_up(config, stated, join).await {
            Ok(Some(brought)) => brought,
            Ok(None) => {
                self.set(Serving::Idle);
                return Ok(());
            }
            Err(error) => {
                warn!(
                    deployment = name,
                    "the deployment did not come up: {error:#}"
                );
                *service::lock(&self.last_error) = Some(format!("{error:#}"));
                self.set(Serving::Idle);
                return Err(error);
            }
        };
        self.serve(brought).await
    }

    /// The bring-up of [`Node::deploy`]: the deployment's daemon, its engines, its admin state
    /// and video jobs; `None` when the deployment ended before it came up.
    async fn bring_up(
        &self,
        config: ServerArgs,
        stated: Stated,
        join: Option<(::service::Listener, ::service::Endpoint)>,
    ) -> Result<Option<Brought>> {
        let name = config.served_model_name().to_owned();
        let token = self.deployment_token();
        let mut options = config.daemon_options(&self.instance);
        let join = match (join, &config.dist_init_addr) {
            (Some(join), _) => Some(join),
            (None, Some(endpoint)) => Some(Daemon::bind_join(endpoint)?),
            (None, None) => None,
        };
        let join = join.map(|(listener, endpoint)| {
            service::lock(&self.ready).endpoints().join = Some(endpoint.clone());
            options.dist_init_addr = Some(endpoint);
            listener
        });
        // The leader's start waits for the other nodes to join, which blocks.
        let starting = tokio::task::spawn_blocking(move || Daemon::start(options, join));
        let daemon = tokio::select! {
            daemon = starting => daemon
                .context("the daemon's start did not complete")?
                .context("failed to start the deployment's daemon")?,
            () = token.cancelled() => return Ok(None),
        };
        let address = daemon
            .address()
            .context("the node's daemon leads no deployment")?
            .to_owned();
        let (updates, awake) = daemon
            .updater(name.clone())
            .context("the leader's daemon commands no engine")?;
        // The epoch records the admin reports and an update writes are one.
        let epochs = Arc::clone(updates.epochs());
        Daemon::advance_updates(Arc::clone(&updates), awake)
            .context("starting the updater's thread")?;
        let leader = tokio::select! {
            leader = Leader::start(&self.families, &config, &stated, address, &self.instance, updates) => leader,
            () = token.cancelled() => {
                stop_daemon(Some(daemon)).await?;
                return Ok(None);
            }
        };
        let leader = match leader {
            Ok(leader) => Arc::new(leader),
            Err(error) => {
                if let Err(stopped) = stop_daemon(Some(daemon)).await {
                    warn!("the daemon did not stop after the engine failed to start: {stopped:#}");
                }
                return Err(error);
            }
        };
        let retire = Arc::new(AtomicBool::new(false));
        let admin = Admin::with_epochs(
            token.clone(),
            Arc::clone(&retire),
            epochs,
            config.admin_config.event_log,
        );
        admin.record("deployment", format!("serving {name}"));
        let jobs = Arc::new(VideoJobs::new(
            config.video_job_config.clone(),
            token.child_token(),
        ));
        Ok(Some(Brought {
            config,
            stated,
            leader,
            admin,
            jobs,
            adopted: Vec::new(),
            daemon: Some(daemon),
            retire,
            token,
        }))
    }

    /// Serves the brought-up deployment until it ends: its routes take the node's requests, its
    /// readiness is the node's, and a hot restart may hand the node over. At its end it drains
    /// and comes down, or, once a successor serves, finishes its video jobs and detaches; the
    /// node then serves no deployment.
    ///
    /// # Errors
    ///
    /// Fails when the listener cannot be duplicated for a restart, or the deployment fails to
    /// come down.
    pub(crate) async fn serve(self: Arc<Self>, brought: Brought) -> Result<()> {
        let Brought {
            config,
            stated,
            leader,
            admin,
            jobs,
            adopted,
            daemon,
            retire,
            token,
        } = brought;
        let name = leader.served_model_name().to_owned();
        let daemon = Arc::new(Mutex::new(daemon));
        let (listener, descriptors) = &self.listener;
        let descriptors = descriptors
            .iter()
            .map(OwnedFd::try_clone)
            .collect::<std::io::Result<Vec<_>>>()
            .context("duplicating the client listener for a restart")?;
        let restarter = Arc::new(Restarter::new(
            Arc::clone(&daemon),
            (listener.clone(), descriptors),
            &self.instance,
            stated,
            self.accepting.clone(),
            self.lifetime,
        ));
        restarter.attach_handoff(Arc::clone(&self.handoff));
        let restarting = restarter.restarting();
        let mut decided = restarter.decide();
        admin.watch(&leader);
        let client = router(
            Arc::clone(&leader),
            Arc::clone(&jobs),
            &config.server_config,
            video_body_limit(&config.media_config, &config.server_config),
            Arc::clone(&self.handoff),
        );
        let state = AdminState {
            leader: Arc::clone(&leader),
            admin: Arc::clone(&admin),
            jobs: Arc::clone(&jobs),
            restarter: Some(Arc::clone(&restarter)),
        };
        self.set(Serving::Deployment(Arc::new(Routes {
            name: name.clone(),
            client: client.clone(),
            admin: super::admin::router(state.clone()),
            harness: super::admin::harness_router(state),
        })));
        {
            let mut ready = service::lock(&self.ready);
            ready.add_storage();
            admin.set_readiness(ready.record(service::State::Serving));
            ready.report(service::State::Serving);
        }
        info!(deployment = name, "the node serves the deployment");
        adopt_streams(
            adopted,
            &super::conn::Server {
                app: client.clone(),
                limits: super::conn::Limits::of(&config.server_config),
                admitting: self.handoff.admissions(),
            },
        );

        tokio::select! {
            () = token.cancelled() => {}
            () = self.accepting.cancelled() => {}
        }
        let handed_over = restarter.handoff_sent.load(Ordering::Acquire);
        if handed_over {
            // The jobs this daemon took run to their ends before it lets go of the engines.
            jobs.wait().await;
        } else {
            // New requests find no deployment; the requests in flight end within the budget.
            self.set(Serving::Stopping(name.clone()));
            let drained =
                tokio::time::timeout(config.server_config.shutdown_timeout, leader.drained()).await;
            if drained.is_err() {
                warn!(
                    deployment = name,
                    "the shutdown budget ran out; requests in flight abort"
                );
            }
            jobs.wait().await;
        }
        // A restart in flight speaks for the deployment: ending now would close the handoff
        // channel under the successor while it adopts the engines. A restart decides on every
        // path (the successor serves, gives up, or exits and closes the handoff socket), and
        // only one that still runs can be waited for: the decision starts at `None`, so its state
        // alone cannot tell "nothing asked" from "in flight".
        if restarting.load(Ordering::Acquire) && decided.borrow().is_none() {
            let _ = decided.wait_for(Option::is_some).await;
        }
        let handed_over = restarter.handoff_sent.load(Ordering::Acquire);
        let ended = finish(&config, &leader, daemon, retire, handed_over).await;
        if let Err(error) = &ended {
            *service::lock(&self.last_error) = Some(format!("{error:#}"));
        }
        self.set(Serving::Idle);
        info!(deployment = name, "the node serves no deployment");
        ended
    }

    /// The client API: the serving deployment's routes, or `503` while none serves.
    fn client_routes(self: &Arc<Self>) -> Router {
        let mut routes = Router::new()
            .fallback(dispatch_client)
            .with_state(Arc::clone(self));
        if let Some(key) = &self.server.api_key {
            let key: Arc<str> = Arc::from(key.as_str());
            // A deployment's own routes check the key themselves.
            routes = routes.route_layer(from_fn(move |request: Request, next: Next| {
                middleware::require_api_key(Arc::clone(&key), request, next)
            }));
        }
        routes
    }

    /// The operator's socket: the node's own verbs, and once a deployment serves, its admin
    /// routes.
    fn admin_routes(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/admin/config", get(idle_config).put(deploy))
            .route("/admin/node", get(node_status))
            .fallback(dispatch_admin)
            .with_state(Arc::clone(self))
    }

    /// The harness's socket: a serving deployment's harness routes.
    fn harness_routes(self: &Arc<Self>) -> Router {
        Router::new()
            .fallback(dispatch_harness)
            .with_state(Arc::clone(self))
    }
}

/// Serves the responses a hot restart's predecessor handed over: each connection writes the bytes
/// the predecessor did not write and goes on with the resumed body, through the deployment's
/// client routes.
fn adopt_streams(adopted: Vec<crate::restart::streams::Adopted>, server: &super::conn::Server) {
    for stream in adopted {
        let server = server.clone();
        tokio::spawn(async move {
            let request_id = stream.request_id.clone();
            let frozen_at_ns = stream.frozen_at_ns;
            let adopted = request_id.clone();
            let ended = request_id.clone();
            // The adoption is recorded here and nowhere else: this process serves the response,
            // and the moment it records is the one before its first frame of its own. The
            // record of the response's end names the response, not the connection this process
            // may go on serving.
            let error = super::conn::adopt(
                stream.descriptor,
                stream.http,
                stream.body,
                server,
                move || crate::restart::streams::adopted(&adopted, frozen_at_ns),
                move || crate::restart::streams::finished(&ended),
            )
            .await
            .err();
            if let Some(error) = error {
                warn!(%request_id, %error, "a connection of the handoff ended");
            }
        });
    }
}

/// Ends a deployment's serving: shuts its engines down, stops its daemon and, when `retire` was
/// set, releases its storage; or, when the deployment was handed to a hot restart's successor,
/// lets go of the engines and leaves the service and the storage to the successor.
///
/// # Errors
///
/// The first failure of the steps.
async fn finish(
    config: &ServerArgs,
    leader: &Arc<Leader>,
    daemon: Arc<Mutex<Option<Daemon>>>,
    retire: Arc<AtomicBool>,
    handed_over: bool,
) -> Result<()> {
    if handed_over {
        info!("the successor daemon serves the deployment; this daemon lets go of it");
        return leader.detach().await;
    }
    let stopped = leader.shutdown().await;
    let daemon = service::lock(&daemon).take();
    let stopped = stopped.and(stop_daemon(daemon).await);
    if retire.load(Ordering::Acquire) {
        let deployment = config.served_model_name().to_owned();
        let retired = tokio::task::spawn_blocking(move || retire_deployment(&deployment))
            .await
            .context("the deployment's release did not complete")?;
        return stopped.and(retired);
    }
    stopped
}

/// Stops a deployment's daemon, ending every process it supervises.
async fn stop_daemon(daemon: Option<Daemon>) -> Result<()> {
    tokio::task::spawn_blocking(move || match daemon {
        Some(mut daemon) => daemon.stop(),
        None => Ok(()),
    })
    .await
    .context("the daemon's stop did not complete")?
}

/// The body of `PUT /admin/config` on a node that serves no deployment, as a deployment's own
/// reads it (`docs/admin.md`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeployBody {
    /// The deployment's configuration, as a `--config` TOML file states it.
    #[serde(default)]
    config: String,
    /// `PATH=VALUE` assignments over it, as `--set` states them.
    #[serde(default)]
    set: Vec<String>,
    /// Validate the configuration and start nothing.
    #[serde(default)]
    dry_run: bool,
}

/// `PUT /admin/config`: on a node that serves no deployment, brings one up and responds once it
/// starts (`202`), with its name; on one that serves a deployment, the deployment's own.
///
/// The deployment's node-level options are its node's: its HTTP server (`server_config`) and its
/// admin API (`admin_config`), whose listeners the node bound, and, unless the configuration
/// states one, the Build Server it prepares its modules at (`build_config`).
async fn deploy(State(node): State<Arc<Node>>, request: Request) -> Response {
    if let Some(routes) = node.routes() {
        return serve_routes(Some(routes.admin.clone()), request).await;
    }
    let body = match axum::body::to_bytes(request.into_body(), node.server.max_request_body).await {
        Ok(body) => body,
        Err(error) => return ApiError::invalid_request(error.to_string(), None).into_response(),
    };
    let body: DeployBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
    };
    let (mut config, stated) = match resolve_text(&body.config, &body.set) {
        Ok(resolved) => resolved,
        Err(error) => {
            return ApiError::invalid_request(format!("{error:#}"), Some("config".to_owned()))
                .into_response();
        }
    };
    config.server_config = node.server.clone();
    config.admin_config = node.admin.clone();
    if !stated.contains_within("build_config") {
        config.build_config = node.build.clone();
    }
    if let Err(error) = config.validate(&stated) {
        return ApiError::invalid_request(format!("{error:#}"), Some("config".to_owned()))
            .into_response();
    }
    if !config.states_model() {
        return ApiError::invalid_request(
            "the configuration names no model",
            Some("config".to_owned()),
        )
        .into_response();
    }
    if !config.leads() {
        return ApiError::invalid_request(
            "a node brings up the deployments it leads (`node_rank` 0); another node of a \
             deployment starts with `uniserve serve` and joins its leader",
            Some("config".to_owned()),
        )
        .into_response();
    }
    let name = config.served_model_name().to_owned();
    if body.dry_run {
        return Json(serde_json::json!({
            "change": "dry_run",
            "dry_run": { "deployment": name, "epoch": true }
        }))
        .into_response();
    }
    {
        let mut serving = service::lock(&node.serving);
        if !matches!(*serving, Serving::Idle) {
            return ApiError::conflict("the node serves a deployment; shut it down first")
                .into_response();
        }
        *serving = Serving::Starting(name.clone());
    }
    *service::lock(&node.last_error) = None;
    let task = tokio::spawn(Arc::clone(&node).deploy(config, stated, None));
    *service::lock(&node.current) = Some(task);
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "deployment": name })),
    )
        .into_response()
}

/// `GET /admin/config`: the serving deployment's, or, while none serves, the same typed status
/// with nothing in it: no replica, no update, no GPU or quota and no target marked unavailable.
async fn idle_config(State(node): State<Arc<Node>>, request: Request) -> Response {
    if let Some(routes) = node.routes() {
        return serve_routes(Some(routes.admin.clone()), request).await;
    }
    Json(crate::updates::entry::ConfigStatus {
        first: None,
        replicas: Vec::new(),
        gpus: None,
        quotas: None,
        update: None,
        unavailable: Vec::new(),
    })
    .into_response()
}

/// `GET /admin/node`: what the node serves.
async fn node_status(State(node): State<Arc<Node>>) -> Json<NodeStatus> {
    let (state, deployment) = match &*service::lock(&node.serving) {
        Serving::Idle => (NodeState::Idle, None),
        Serving::Starting(name) => (NodeState::Starting, Some(name.clone())),
        Serving::Deployment(routes) => (NodeState::Serving, Some(routes.name.clone())),
        Serving::Stopping(name) => (NodeState::Stopping, Some(name.clone())),
    };
    Json(NodeStatus {
        state,
        deployment,
        last_error: service::lock(&node.last_error).clone(),
    })
}

/// The client API: the serving deployment's routes, or `503` while none serves.
async fn dispatch_client(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = node.routes().map(|routes| routes.client.clone());
    serve_routes(routes, request).await
}

/// Every other admin path on the operator's socket: the serving deployment's admin routes.
async fn dispatch_admin(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = node.routes().map(|routes| routes.admin.clone());
    serve_routes(routes, request).await
}

/// The harness's socket: the serving deployment's harness routes.
async fn dispatch_harness(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = node.routes().map(|routes| routes.harness.clone());
    serve_routes(routes, request).await
}

/// `request` through `routes`, or `503` while no deployment serves.
async fn serve_routes(routes: Option<Router>, request: Request) -> Response {
    match routes {
        Some(routes) => match routes.oneshot(request).await {
            Ok(response) => response,
            Err(never) => match never {},
        },
        None => ApiError::service_unavailable(
            "no_deployment",
            "the node serves no deployment; PUT /admin/config brings one up",
        )
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _};

    use super::*;

    /// Sends `request` on `stream` and reads the whole response, which the server ends by
    /// closing the connection the request asked it to close.
    async fn exchange(mut stream: impl AsyncRead + AsyncWrite + Unpin, request: &str) -> String {
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    /// A node started without a deployment answers on every listener at once: its client API
    /// with `503 no_deployment`, its operator's socket with its state, idle, and an empty
    /// configuration status; a stop ends it, and nothing was handed over.
    #[tokio::test]
    async fn a_node_without_a_deployment_serves_its_own_routes_until_stopped() {
        let runtime = tempfile::tempdir().unwrap();
        let runtime = service::RuntimeDirectory::within(runtime.path()).unwrap();
        let instance = service::Instance::create(&runtime, "daemon").unwrap();
        let listener = Listener::bind(&AdminConfig::default()).await.unwrap();
        let address = listener.endpoint().unwrap().to_string();
        let admins = super::super::admin::Listeners::bind(instance.path()).unwrap();
        let admin_socket = instance.path().join("admin.sock");
        let shutdown = CancellationToken::new();
        let start = Start {
            families: processing::Registry::new([]).unwrap(),
            options: ServerArgs::default(),
            ready: Ready::new(None, instance.path()),
            instance,
            lifetime: Lifetime::Deployments,
            shutdown: shutdown.clone(),
        };
        let node = tokio::spawn(run(start, listener, admins, First::Idle));

        let client = exchange(
            tokio::net::TcpStream::connect(&address).await.unwrap(),
            "GET /v1/models HTTP/1.1\r\nhost: node\r\nconnection: close\r\n\r\n",
        )
        .await;
        assert!(client.starts_with("HTTP/1.1 503"), "{client}");
        assert!(client.contains("no_deployment"), "{client}");
        let response = exchange(
            tokio::net::UnixStream::connect(&admin_socket)
                .await
                .unwrap(),
            "GET /admin/node HTTP/1.1\r\nhost: node\r\nconnection: close\r\n\r\n",
        )
        .await;
        let (_, body) = response.split_once("\r\n\r\n").unwrap();
        let status: NodeStatus = serde_json::from_str(body).unwrap();
        assert_eq!(
            status,
            NodeStatus {
                state: NodeState::Idle,
                deployment: None,
                last_error: None,
            }
        );

        // Its configuration is the typed status a deployment answers, with nothing in it.
        let response = exchange(
            tokio::net::UnixStream::connect(&admin_socket)
                .await
                .unwrap(),
            "GET /admin/config HTTP/1.1\r\nhost: node\r\nconnection: close\r\n\r\n",
        )
        .await;
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 200"), "{response}");
        let config: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(config["replicas"], serde_json::json!([]), "{config}");
        assert_eq!(config["update"], serde_json::Value::Null, "{config}");

        shutdown.cancel();
        let ended = node.await.unwrap().unwrap();
        assert!(!ended.handed_over);
    }
}
