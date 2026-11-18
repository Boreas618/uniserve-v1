//! The node: what one daemon process serves on its listeners, through one path whether it
//! serves the deployment of the package it started with or a deployment a hot restart's
//! predecessor handed it.
//!
//! The node binds its listeners when it starts ([`super::serve`]): the client API's address and
//! the admin API's two Unix sockets in its instance directory. It serves them for its whole life,
//! dispatching every request to the deployment that serves: the client API to its routes, `503`
//! while none serves; the admin sockets to its admin routes. The node itself answers its status
//! (`GET /admin/status`: starting, serving or stopping), its event log and the deployment's
//! policy (`/admin/policy`).
//!
//! Every change of the deployment goes through one entry, [`Node::apply`]: a package, compared
//! option by option with the current revision ([`proto::reload`]), written as the next revision
//! of the deployment's persistent state ([`crate::policy::revision`]), executed, and made current
//! once the deployment converged to it. A start is the first apply, from nothing
//! ([`Node::start`]); a node that starts with the package its current revision holds starts
//! that revision as its apply fixed it. `PUT /admin/policy` is every later one.
//!
//! The event log and the operations' record are the node's ([`Journal`]), which every
//! deployment it serves writes.
//!
//! A deployment, however it came up, is served the same way ([`Node::serve`]): its routes, its
//! readiness, its restarts, and its end, which drains its requests within the shutdown budget,
//! shuts its engines down, stops its daemon and, when its `POST /admin/shutdown` asked for it,
//! releases its storage at the host's Storage Server; or, once a successor serves it, detaches.
//! The node ends with its deployment.

use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context as _, Result};
use axum::extract::{Query, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use proto::admin::{EventBody, EventsQuery, NodeState, OperationKind, Phase};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;
use tracing::{info, warn};

use super::admin::{Admin, AdminState, Journal};
use super::handoff::Handoff;
use super::listener::{Listener, ListenerHandoff};
use super::{Ready, middleware, retire_deployment, router, video_body_limit};
use crate::api::openai::ApiError;
use crate::api::openai::video_jobs::VideoJobs;
use crate::config::{Config, ServerConfig, Stated};
use crate::leader::Leader;
use crate::policy::Package;
use crate::policy::revision::{KEPT, Outcome, Record, Revisions};
use crate::restart::Restarter;
use crate::supervision::Daemon;
use crate::updates::change::Applied;

/// The writer a revision records for the package a node started with.
const START: &str = "start";

/// The writer a revision records for a `PUT /admin/policy`.
const OPERATOR: &str = "operator";

/// What the node serves.
enum Serving {
    /// No deployment: before the first comes up, and after it came down.
    Idle,
    /// A deployment is coming up.
    Starting(String),
    /// A deployment serves through its routes.
    Deployment(Arc<Routes>),
    /// A deployment is coming down.
    Stopping(String),
}

/// A serving deployment: its client API's routes, the operator's and the harness's admin
/// routes, and what applies a change to it.
struct Routes {
    client: Router,
    admin: Router,
    harness: Router,
    leader: Arc<Leader>,
    deployment: Arc<Admin>,
    jobs: Arc<VideoJobs>,
    restarter: Arc<Restarter>,
}

/// Which admin socket a request came on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Socket {
    Operator,
    Harness,
}

impl Routes {
    /// The deployment's admin routes `socket` serves.
    fn admin(&self, socket: Socket) -> Router {
        match socket {
            Socket::Operator => self.admin.clone(),
            Socket::Harness => self.harness.clone(),
        }
    }
}

/// A deployment brought up and not yet served: what [`Node::serve`] serves, whether the node's
/// own start made it ([`Node::start`]) or a hot restart's successor adopted it.
pub(crate) struct Brought {
    /// The deployment's configuration and the paths it stated.
    pub(crate) config: Config,
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
    /// The deployment's end: the node's stop.
    pub(crate) token: CancellationToken,
    /// The revision a hot restart's predecessor converged to when it handed the deployment
    /// over for the daemon's own options: this daemon converges the rest of it.
    pub(crate) then: Option<u64>,
}

/// What a node serves first.
pub(crate) enum First {
    /// The package the node started with, applied as the deployment's start; `join` is the join
    /// listener the leader bound at start for the deployment's other nodes, with the endpoint it
    /// reported.
    Package {
        package: Box<Package>,
        join: Option<(::service::Listener, ::service::Endpoint)>,
    },
    /// A deployment already brought up: a hot restart's successor's adoption.
    Serving(Box<Brought>),
}

/// What a node is started with: its model families, its own options, its deployment's name and
/// persistent state, its instance directory and readiness, and the stop signal that ends it.
pub(crate) struct Start {
    pub(crate) families: processing::Registry,
    /// The node's own options: its HTTP server's (`server_config`) and its admin API's
    /// (`admin_config`), whose listeners it bound.
    pub(crate) options: Config,
    /// The deployment's name, which names its persistent state.
    pub(crate) deployment: String,
    /// The deployment's revisions on this node.
    pub(crate) revisions: Arc<Revisions>,
    pub(crate) instance: service::Instance,
    pub(crate) ready: Ready,
    pub(crate) shutdown: CancellationToken,
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
    deployment: String,
    revisions: Arc<Revisions>,
    instance: PathBuf,
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
    /// The node's event log and operations' record, which every deployment it serves writes.
    journal: Arc<Journal>,
    /// Why the last deployment failed to start or to stop.
    last_error: Mutex<Option<String>>,
    /// Held by the change that converges: changes converge one after another, a change that
    /// arrives meanwhile converging once the one before it ended.
    applying: tokio::sync::Mutex<()>,
}

/// Why a change of the deployment did not converge, with the status its request is answered
/// with.
struct Refused {
    status: StatusCode,
    message: String,
}

impl Refused {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    /// A failure of the node's own state directory.
    fn state(error: anyhow::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    }
}

impl IntoResponse for Refused {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": { "message": self.message } });
        (self.status, Json(body)).into_response()
    }
}

/// Runs a node started with `start` on `listener` and `admins`, serving `first`, until it ends:
/// at its stop signal, with its deployment, or once a hot restart's successor serves it.
///
/// At a stop, the node stops accepting connections, its deployment drains its requests within
/// the shutdown budget (`server_config.shutdown_timeout`) and comes down, and the node waits for
/// its connections for what is left of the budget. Once a successor serves, the node finishes
/// the requests and video jobs it took, however long they take, and its deployment detaches.
///
/// # Errors
///
/// Fails when the listener cannot be duplicated for a hot restart, the deployment fails to come
/// up or to come down, or the HTTP server fails.
pub(crate) async fn run(
    start: Start,
    mut listener: Listener,
    admins: super::admin::Listeners,
    first: First,
) -> Result<Ended> {
    let Start {
        families,
        options,
        deployment,
        revisions,
        instance,
        ready,
        shutdown,
    } = start;
    // A hot restart's successor goes on with its predecessor's journal, which the adopted
    // deployment writes; a node started otherwise starts one.
    let journal = match &first {
        First::Serving(brought) => Arc::clone(&brought.admin.journal),
        First::Package { .. } => Journal::new(options.admin_config.event_log),
    };
    let node = Arc::new(Node {
        families,
        server: options.server_config.clone(),
        deployment,
        revisions,
        instance: instance.path().to_owned(),
        ready: Mutex::new(ready),
        handoff: Arc::new(Handoff::new()),
        listener: listener
            .hand_over()
            .context("duplicating the client listener")?,
        accepting: CancellationToken::new(),
        shutdown,
        serving: Mutex::new(Serving::Idle),
        journal,
        last_error: Mutex::new(None),
        applying: tokio::sync::Mutex::new(()),
    });

    super::admin::serve_routes(
        admins,
        node.admin_routes(Socket::Operator),
        node.admin_routes(Socket::Harness),
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
    // The node's deployment, whose end is the node's.
    let mut own = Some(match first {
        First::Package { package, join } => tokio::spawn(Arc::clone(&node).start(*package, join)),
        First::Serving(brought) => tokio::spawn(Arc::clone(&node).serve(*brought)),
    });

    // The node ends at its stop, when a successor takes it over, or with its deployment.
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
    fn set(&self, serving: Serving) {
        *service::lock(&self.serving) = serving;
    }

    /// The deployment's routes while it serves.
    fn routes(&self) -> Option<Arc<Routes>> {
        match &*service::lock(&self.serving) {
            Serving::Deployment(routes) => Some(Arc::clone(routes)),
            Serving::Idle | Serving::Starting(_) | Serving::Stopping(_) => None,
        }
    }

    /// The deployment's start, its first apply: the start from nothing to `package`
    /// (`specs/configuration.md` §8).
    ///
    /// A package equal to the one the current revision holds starts that revision as its apply
    /// fixed it, module revisions included, and writes no revision. Any other package is the
    /// next revision: the deployment starts with it directly, and it becomes current once the
    /// deployment serves it; when it does not come up, the revision records why and the
    /// deployment starts with the current revision instead, so a start never makes current a
    /// package that did not serve.
    ///
    /// # Errors
    ///
    /// Fails when the state directory cannot be read or written, when neither the package nor
    /// the current revision comes up, or when the deployment fails to come down.
    async fn start(
        self: Arc<Self>,
        package: Package,
        join: Option<(::service::Listener, ::service::Endpoint)>,
    ) -> Result<()> {
        let began = Instant::now();
        let current = self.revisions.current()?;
        let stored = current
            .map(|number| self.revisions.load(number))
            .transpose()?;
        if let Some(stored) = &stored
            && stored.package == package
        {
            info!(
                revision = stored.record.revision,
                "the package is the current revision, which starts as its apply fixed it"
            );
            return match self
                .bring(&stored.package, join, stored.lock.clone())
                .await?
            {
                Some(brought) => self.serve(brought).await,
                None => Ok(()),
            };
        }

        let previous = stored.as_ref().map(|stored| &stored.package);
        let changes =
            proto::reload::changes(previous.map(|previous| &previous.config), &package.config);
        let number = self.revisions.begin(&package, START, previous, &changes)?;
        info!(
            revision = number,
            "the deployment starts with a new revision"
        );
        match self.bring(&package, join, None).await {
            Ok(Some(brought)) => {
                if let Some(modules) = brought.leader.modules() {
                    self.revisions.lock(number, &modules)?;
                }
                self.revisions
                    .converged(number, began.elapsed().as_secs_f64(), KEPT)?;
                self.serve(brought).await
            }
            Ok(None) => {
                self.revisions.ended(
                    number,
                    Outcome::Abandoned,
                    "the node stopped before the deployment came up".to_owned(),
                    began.elapsed().as_secs_f64(),
                )?;
                Ok(())
            }
            Err(error) => {
                self.revisions.ended(
                    number,
                    Outcome::RolledBack,
                    format!("{error:#}"),
                    began.elapsed().as_secs_f64(),
                )?;
                let Some(stored) = stored else {
                    return Err(error);
                };
                warn!(
                    revision = stored.record.revision,
                    "the package did not come up ({error:#}); the current revision starts"
                );
                match self
                    .bring(&stored.package, None, stored.lock.clone())
                    .await?
                {
                    Some(brought) => self.serve(brought).await,
                    None => Ok(()),
                }
            }
        }
    }

    /// Brings the deployment of `package` up, as its start: its daemon, its engines (with the
    /// module revisions `lock` fixed, or those its apply fixes now), its admin state and video
    /// jobs; `None` when the node stopped first. `join` is the join listener the node bound at
    /// start, or `None` to bind one now for a deployment of several nodes.
    ///
    /// The deployment's daemon starts first and, on a deployment of several nodes, waits until
    /// every other node's daemon joined; its engines and first epoch follow.
    ///
    /// # Errors
    ///
    /// Fails when the daemon or the engines cannot start.
    async fn bring(
        &self,
        package: &Package,
        join: Option<(::service::Listener, ::service::Endpoint)>,
        lock: Option<crate::epoch::ModuleSet>,
    ) -> Result<Option<Brought>> {
        let config = package.config.clone();
        let stated = package.stated.clone();
        let name = config.served_model_name().to_owned();
        self.set(Serving::Starting(name.clone()));
        self.journal
            .record(EventBody::Deployment, format!("{name} comes up"));
        let brought = self.bring_up(config, stated, join, lock).await;
        match &brought {
            Ok(Some(_)) => {}
            Ok(None) => {
                self.set(Serving::Idle);
                self.journal.record(
                    EventBody::Deployment,
                    format!("the node stopped before {name} came up"),
                );
            }
            Err(error) => {
                warn!(
                    deployment = name,
                    "the deployment did not come up: {error:#}"
                );
                *service::lock(&self.last_error) = Some(format!("{error:#}"));
                self.set(Serving::Idle);
                self.journal.record(
                    EventBody::Deployment,
                    format!("{name} did not come up: {error:#}"),
                );
            }
        }
        brought
    }

    /// The bring-up of [`Node::bring`].
    async fn bring_up(
        &self,
        config: Config,
        stated: Stated,
        join: Option<(::service::Listener, ::service::Endpoint)>,
        lock: Option<crate::epoch::ModuleSet>,
    ) -> Result<Option<Brought>> {
        let name = config.served_model_name().to_owned();
        let token = self.shutdown.clone();
        let identity = config.identity()?;
        let mut options = config.daemon_options(&identity, &self.instance);
        let join = match (join, &identity.join) {
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
        Daemon::advance_updates(Arc::clone(&updates), awake)
            .context("starting the updater's thread")?;
        let leader = tokio::select! {
            leader = Leader::start(&self.families, &config, &stated, address, &self.instance, updates, lock) => leader,
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
        let admin = Admin::new(
            token.clone(),
            Arc::clone(&retire),
            Arc::clone(&self.journal),
        );
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
            then: None,
        }))
    }

    /// Serves the brought-up deployment until it ends: its routes take the node's requests, its
    /// readiness is the node's, and a hot restart may hand the node over. At its end it drains
    /// and comes down, or, once a successor serves, finishes its video jobs and detaches; the
    /// node then serves no deployment. A deployment adopted in the middle of a change converges
    /// the rest of it first ([`Brought::then`]).
    ///
    /// # Errors
    ///
    /// Fails when the listener cannot be duplicated for a restart, or the deployment fails to
    /// come down.
    async fn serve(self: Arc<Self>, brought: Brought) -> Result<()> {
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
            then,
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
            self.deployment.clone(),
            stated,
            self.accepting.clone(),
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
            client: client.clone(),
            admin: super::admin::router(state.clone()),
            harness: super::admin::harness_router(state),
            leader: Arc::clone(&leader),
            deployment: Arc::clone(&admin),
            jobs: Arc::clone(&jobs),
            restarter: Arc::clone(&restarter),
        })));
        {
            let mut ready = service::lock(&self.ready);
            ready.add_storage();
            admin.set_readiness(ready.record(service::State::Serving));
            ready.report(service::State::Serving);
        }
        info!(deployment = name, "the node serves the deployment");
        admin.record(format!("serving {name}"));
        adopt_streams(
            adopted,
            &super::conn::Server {
                app: client.clone(),
                limits: super::conn::Limits::of(&config.server_config),
                admitting: self.handoff.admissions(),
            },
        );
        if let Some(number) = then {
            let node = Arc::clone(&self);
            tokio::spawn(async move { Box::pin(node.resume(number)).await });
        }

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
        let ended = finish(&name, &leader, daemon, retire, handed_over).await;
        match &ended {
            // A successor's journal records what the deployment does from now on.
            Ok(()) if handed_over => {}
            Ok(()) => self
                .journal
                .record(EventBody::Deployment, format!("{name} came down")),
            Err(error) => {
                *service::lock(&self.last_error) = Some(format!("{error:#}"));
                self.journal.record(
                    EventBody::Deployment,
                    format!("{name} did not come down cleanly: {error:#}"),
                );
            }
        }
        self.set(Serving::Idle);
        info!(deployment = name, "the node serves no deployment");
        ended
    }

    /// Changes the serving deployment to `package`, which `writer` wrote, as one change: the
    /// next revision, executed by its effects and made current once the deployment converged to
    /// it (module documentation). `expected` is the revision the writer read the deployment at
    /// (`If-Match`): a change since refuses this one. `force` applies a target a rollback marked
    /// unavailable.
    ///
    /// Changes converge one after another: one that arrives while another converges waits for
    /// it, and is then compared with what it converged to.
    ///
    /// Returns the revision's record as the change left it: converged, rolled back, or
    /// converging still when the daemon's own options changed and its successor converges the
    /// rest.
    async fn apply(
        &self,
        package: Package,
        writer: &str,
        expected: Option<u64>,
        force: bool,
    ) -> Result<Record, Refused> {
        let _turn = self.applying.lock().await;
        let routes = self.routes().ok_or_else(|| {
            Refused::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "the deployment does not serve: no change applies until it does",
            )
        })?;
        let current = self.revisions.current().map_err(Refused::state)?;
        if expected.is_some_and(|expected| Some(expected) != current) {
            return Err(Refused::new(
                StatusCode::PRECONDITION_FAILED,
                format!(
                    "the deployment changed since revision {}: read it again",
                    expected.unwrap_or_default()
                ),
            ));
        }
        let stored = current
            .map(|number| self.revisions.load(number))
            .transpose()
            .map_err(Refused::state)?;
        if let Some(stored) = &stored
            && stored.package == package
        {
            return Ok(stored.record.clone());
        }
        let previous = stored.as_ref().map(|stored| &stored.package);
        let changes =
            proto::reload::changes(previous.map(|previous| &previous.config), &package.config);
        let number = self
            .revisions
            .begin(&package, writer, previous, &changes)
            .map_err(Refused::state)?;
        Box::pin(self.converge(&routes, number, package, force)).await
    }

    /// Converges the deployment to revision `number`, which holds `package`, and records how it
    /// ended.
    async fn converge(
        &self,
        routes: &Routes,
        number: u64,
        package: Package,
        force: bool,
    ) -> Result<Record, Refused> {
        let began = Instant::now();
        let operation = self
            .journal
            .begin(OperationKind::Config)
            .map_err(|refusal| {
                Refused::new(
                    StatusCode::CONFLICT,
                    match refusal {
                        super::admin::Refusal::Running(other) => {
                            format!("operation {} ({}) runs", other.id, other.kind)
                        }
                        super::admin::Refusal::HandedOver => {
                            "the node was handed to a successor".to_owned()
                        }
                    },
                )
            })?;
        let id = operation.id;
        let leader = &routes.leader;
        let mut phases = leader.updates.watch();
        let applied =
            Box::pin(leader.apply(package.config.clone(), package.stated.clone(), force, id)).await;
        let ended = match applied {
            Ok(Applied::Unchanged | Applied::InPlace(_)) => Ok(true),
            Ok(Applied::Update(_)) => {
                // The update runs on by itself and commits at its trial's end, or rolls back.
                loop {
                    phases.borrow_and_update();
                    match leader.updates.phase_of(id) {
                        Some(Phase::Committed) => break Ok(true),
                        Some(Phase::RolledBack | Phase::Abandoned) | None => {
                            let reason = leader
                                .updates
                                .status()
                                .and_then(|update| update.failure)
                                .map_or_else(
                                    || "the update did not commit".to_owned(),
                                    |failure| format!("{}: {}", failure.kind, failure.reason),
                                );
                            break Err(Refused::new(StatusCode::CONFLICT, reason));
                        }
                        Some(Phase::Preparing | Phase::Switching | Phase::Trial) => {}
                    }
                    if phases.changed().await.is_err() {
                        break Err(Refused::new(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "the updater ended before the update did",
                        ));
                    }
                }
            }
            // The daemon's own options: its successor serves them and converges the rest.
            Ok(Applied::Daemon { config }) => routes
                .restarter
                .restart(
                    Some(*config),
                    Some(number),
                    leader,
                    &routes.deployment,
                    &routes.jobs,
                )
                .await
                .map(|pid| {
                    info!(
                        pid,
                        revision = number,
                        "the successor daemon converges the change"
                    );
                    false
                })
                .map_err(|error| {
                    Refused::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
                }),
            Err(error) => Err(Refused::new(
                super::admin::update_status(&error),
                format!("{:#}", anyhow::Error::from(error)),
            )),
        };
        let seconds = began.elapsed().as_secs_f64();
        let recorded = match &ended {
            Ok(true) => {
                if let Some(modules) = leader.modules() {
                    self.revisions
                        .lock(number, &modules)
                        .map_err(Refused::state)?;
                }
                self.revisions.converged(number, seconds, KEPT)
            }
            // The successor records how the change ended.
            Ok(false) => Ok(()),
            Err(refused) => self.revisions.ended(
                number,
                Outcome::RolledBack,
                refused.message.clone(),
                seconds,
            ),
        };
        self.journal.end(
            operation,
            ended
                .as_ref()
                .map(drop)
                .map_err(|refused| refused.message.clone()),
        );
        recorded.map_err(Refused::state)?;
        ended?;
        self.revisions.record(number).map_err(Refused::state)
    }

    /// Converges revision `number`, the change whose daemon options a hot restart applied
    /// before it handed the deployment to this daemon.
    async fn resume(&self, number: u64) {
        let _turn = self.applying.lock().await;
        let resumed = async {
            let routes = self.routes().ok_or_else(|| {
                Refused::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "the deployment does not serve",
                )
            })?;
            let stored = self.revisions.load(number).map_err(Refused::state)?;
            Box::pin(self.converge(&routes, number, stored.package, false)).await
        };
        match resumed.await {
            Ok(record) => {
                info!(revision = number, result = ?record.result, "the change after the hot restart ended")
            }
            Err(refused) => warn!(
                revision = number,
                "the change after the hot restart did not converge: {}", refused.message
            ),
        }
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

    /// The admin socket `socket`: the node's status, event log and policy, and once the
    /// deployment serves, its admin routes.
    fn admin_routes(self: &Arc<Self>, socket: Socket) -> Router {
        Router::new()
            .route(
                "/admin/status",
                get(move |State(node), request| status(node, socket, request)),
            )
            .route(
                "/admin/events",
                get(move |State(node), request| events(node, socket, request)),
            )
            .route("/admin/policy", get(get_policy).put(put_policy))
            .fallback(move |State(node): State<Arc<Node>>, request| async move {
                let routes = node.routes().map(|routes| routes.admin(socket));
                serve_routes(routes, request).await
            })
            .with_state(Arc::clone(self))
    }
}

/// `GET /admin/policy`'s query: the revision to read, the current one when unstated.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PolicyQuery {
    revision: Option<u64>,
}

/// `GET /admin/policy[?revision=N]`: the current revision's package, or revision `N`'s, as a tar
/// archive of its directory (`config.toml`, with what programs wrote, and `programs/`); its
/// `ETag` is the revision's number, which a `PUT` states in `If-Match` to change the deployment
/// only as it read it.
async fn get_policy(State(node): State<Arc<Node>>, Query(query): Query<PolicyQuery>) -> Response {
    let number = match query.revision {
        Some(number) => Some(number),
        None => match node.revisions.current() {
            Ok(current) => current,
            Err(error) => return Refused::state(error).into_response(),
        },
    };
    let Some(number) = number else {
        return Refused::new(StatusCode::NOT_FOUND, "the deployment has no revision yet")
            .into_response();
    };
    let archive = match node.revisions.load(number) {
        Ok(stored) => stored.package.archive(),
        Err(error) => {
            return Refused::new(StatusCode::NOT_FOUND, format!("{error:#}")).into_response();
        }
    };
    match archive {
        Ok(archive) => (
            [
                (
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/x-tar"),
                ),
                (header::ETAG, etag(number)),
            ],
            archive,
        )
            .into_response(),
        Err(error) => Refused::state(error).into_response(),
    }
}

/// `PUT /admin/policy[?force=true]`: changes the deployment to the package the body carries, a
/// tar archive of its directory, which replaces the target whole ([`Node::apply`]). Answers
/// with the revision's record once the change converged (`200`) or did not (an error status
/// with the reason); `202` when the daemon restarts for its own options and its successor
/// converges the rest. `If-Match: "N"` refuses the change with `412` when the current revision
/// is not `N`.
async fn put_policy(State(node): State<Arc<Node>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let force = parts
        .uri
        .query()
        .is_some_and(|query| query.split('&').any(|pair| pair == "force=true"));
    let expected = match parts.headers.get(header::IF_MATCH) {
        Some(value) => match value
            .to_str()
            .ok()
            .map(|text| text.trim().trim_matches('"'))
            .and_then(|text| text.parse::<u64>().ok())
        {
            Some(number) => Some(number),
            None => {
                return Refused::new(
                    StatusCode::BAD_REQUEST,
                    "`If-Match` states a revision number, as `GET /admin/policy`'s ETag does",
                )
                .into_response();
            }
        },
        None => None,
    };
    let body = match axum::body::to_bytes(body, node.server.max_request_body).await {
        Ok(body) => body,
        Err(error) => return ApiError::invalid_request(error.to_string(), None).into_response(),
    };
    let package = match Package::unpack(&body) {
        Ok(package) => package,
        Err(error) => {
            return Refused::new(StatusCode::BAD_REQUEST, format!("{error:#}")).into_response();
        }
    };
    // The same check a start makes: a target is refused while serving exactly when a start
    // with it would be.
    if let Err(error) = package.config.validate(&package.stated) {
        return Refused::new(StatusCode::BAD_REQUEST, format!("{error:#}")).into_response();
    }
    match Box::pin(node.apply(package, OPERATOR, expected, force)).await {
        Ok(record) => {
            let status = if record.result == Outcome::Converging {
                StatusCode::ACCEPTED
            } else {
                StatusCode::OK
            };
            (
                status,
                [(header::ETAG, etag(record.revision))],
                Json(record),
            )
                .into_response()
        }
        Err(refused) => refused.into_response(),
    }
}

/// A revision's number as an entity tag.
fn etag(number: u64) -> HeaderValue {
    HeaderValue::from_str(&format!("\"{number}\"")).unwrap_or(HeaderValue::from_static("\"0\""))
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
/// set, releases its storage, held at its Storage Servers under the name `storage`; or, when the deployment was handed to a hot restart's successor,
/// lets go of the engines and leaves the service and the storage to the successor.
///
/// # Errors
///
/// The first failure of the steps.
async fn finish(
    storage: &str,
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
        let deployment = storage.to_owned();
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

/// `GET /admin/status` on `socket`: the serving deployment's, or, while none serves, the node's
/// own: idle with its last deployment's error, starting or stopping, beside the node's
/// operations and the next event's sequence number.
async fn status(node: Arc<Node>, socket: Socket, request: Request) -> Response {
    let own = match &*service::lock(&node.serving) {
        Serving::Deployment(routes) => Err(routes.admin(socket)),
        Serving::Idle => Ok(NodeState::Idle {
            last_error: service::lock(&node.last_error).clone(),
        }),
        Serving::Starting(name) => Ok(NodeState::Starting { name: name.clone() }),
        Serving::Stopping(name) => Ok(NodeState::Stopping { name: name.clone() }),
    };
    match own {
        Ok(state) => Json(node.journal.status(state)).into_response(),
        Err(routes) => serve_routes(Some(routes), request).await,
    }
}

/// `GET /admin/events` on `socket`: the serving deployment's, or, while none serves, the node's
/// journal itself, which the deployments write.
async fn events(node: Arc<Node>, socket: Socket, request: Request) -> Response {
    if let Some(routes) = node.routes() {
        return serve_routes(Some(routes.admin(socket)), request).await;
    }
    match Query::<EventsQuery>::try_from_uri(request.uri()) {
        Ok(Query(query)) => super::admin::events(&node.journal, query).await,
        Err(rejection) => rejection.into_response(),
    }
}

/// The client API: the serving deployment's routes, or `503` while none serves.
async fn dispatch_client(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = node.routes().map(|routes| routes.client.clone());
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
            "the node serves no deployment: it starts or stops its deployment",
        )
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _};

    use super::*;

    /// Sends `head` and `body` on `stream` and reads the whole response, which the server ends
    /// by closing the connection the request asked it to close.
    async fn exchange(
        mut stream: impl AsyncRead + AsyncWrite + Unpin,
        head: &str,
        body: &[u8],
    ) -> (String, Vec<u8>) {
        stream.write_all(head.as_bytes()).await.unwrap();
        stream.write_all(body).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let split = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap();
        let head = String::from_utf8(response[..split].to_vec()).unwrap();
        (head, response[split + 4..].to_vec())
    }

    /// A `PUT /admin/policy` of `archive` with the extra headers `headers`.
    fn put(archive: &[u8], headers: &str) -> String {
        format!(
            "PUT /admin/policy HTTP/1.1\r\nhost: node\r\nconnection: close\r\n\
             content-length: {}\r\n{headers}\r\n",
            archive.len()
        )
    }

    /// The deployment's policy is its current revision: `GET /admin/policy` answers it as a
    /// package with the revision as its tag; a `PUT` of the same package changes nothing and
    /// answers the current record; a `PUT` that states another revision than the current one in
    /// `If-Match`, or a package a start would refuse, is refused and writes no revision.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_policy_is_the_current_revision() {
        let leader = Arc::new(crate::test_support::served_engine(
            crate::test_support::CHAT,
        ));
        let config = service::lock(&leader.replicas()[0].config).clone();
        let package = Package::of(config.clone(), BTreeMap::new()).unwrap();

        let state = tempfile::tempdir().unwrap();
        let revisions = Arc::new(Revisions::open(state.path().join("sim")).unwrap());
        let changes = proto::reload::changes(None, &package.config);
        let first = revisions.begin(&package, START, None, &changes).unwrap();
        revisions.converged(first, 0.0, KEPT).unwrap();

        let runtime = tempfile::tempdir().unwrap();
        let runtime = service::RuntimeDirectory::within(runtime.path()).unwrap();
        let instance = service::Instance::create(&runtime, "daemon").unwrap();
        let listener = Listener::bind(&crate::config::AdminConfig::default())
            .await
            .unwrap();
        let admins = super::super::admin::Listeners::bind(instance.path()).unwrap();
        let socket = instance.path().join("admin.sock");
        let shutdown = CancellationToken::new();
        let retire = Arc::new(AtomicBool::new(false));
        let admin = Admin::new(
            shutdown.clone(),
            Arc::clone(&retire),
            Journal::new(config.admin_config.event_log),
        );
        let jobs = Arc::new(VideoJobs::new(
            config.video_job_config.clone(),
            shutdown.child_token(),
        ));
        let brought = Brought {
            config: config.clone(),
            stated: package.stated.clone(),
            leader,
            admin,
            jobs,
            adopted: Vec::new(),
            daemon: None,
            retire,
            token: shutdown.clone(),
            then: None,
        };
        let start = Start {
            families: processing::Registry::new([]).unwrap(),
            options: config.clone(),
            deployment: "sim".to_owned(),
            revisions: Arc::clone(&revisions),
            ready: Ready::new(None, instance.path()),
            instance,
            shutdown: shutdown.clone(),
        };
        let node = tokio::spawn(run(
            start,
            listener,
            admins,
            First::Serving(Box::new(brought)),
        ));
        let connect = || async {
            loop {
                if let Ok(stream) = tokio::net::UnixStream::connect(&socket).await {
                    break stream;
                }
                tokio::task::yield_now().await;
            }
        };

        let (head, body) = exchange(
            connect().await,
            "GET /admin/policy HTTP/1.1\r\nhost: node\r\nconnection: close\r\n\r\n",
            b"",
        )
        .await;
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        assert!(head.to_lowercase().contains("etag: \"1\""), "{head}");
        assert_eq!(Package::unpack(&body).unwrap(), package);

        // The same package again changes nothing.
        let archive = package.archive().unwrap();
        let (head, body) = exchange(connect().await, &put(&archive, ""), &archive).await;
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        let record: Record = serde_json::from_slice(&body).unwrap();
        assert_eq!((record.revision, record.result), (1, Outcome::Converged));

        // A writer that read another revision is refused.
        let mut other = config.clone();
        other.max_logprobs = 5;
        let other = Package::of(other, BTreeMap::new())
            .unwrap()
            .archive()
            .unwrap();
        let (head, _) =
            exchange(connect().await, &put(&other, "if-match: \"7\"\r\n"), &other).await;
        assert!(head.starts_with("HTTP/1.1 412"), "{head}");

        // A package a start refuses is refused while serving.
        let mut invalid = config;
        invalid.mem_fraction_static = 2.0;
        let invalid = Package::of(invalid, BTreeMap::new())
            .unwrap()
            .archive()
            .unwrap();
        let (head, body) = exchange(connect().await, &put(&invalid, ""), &invalid).await;
        assert!(head.starts_with("HTTP/1.1 400"), "{head}");
        assert!(String::from_utf8_lossy(&body).contains("mem_fraction_static"));
        assert_eq!(revisions.current().unwrap(), Some(1));

        shutdown.cancel();
        let ended = node.await.unwrap().unwrap();
        assert!(!ended.handed_over);
    }
}
