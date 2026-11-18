//! The deployment's claim at its Storage Server on this node.
//!
//! Every deployment runs one Storage Server on each of its nodes, which holds nothing but the
//! deployment's storage (`storage::socket`). An engine asks every node's daemon to claim the
//! deployment before a worker starts there (`Claim`): the daemon starts the deployment's Storage
//! Server when the node has none, claims the scopes the engine's epoch places on the node, and
//! holds the claim's client (`storage::ClaimClient`) for as long as it runs. The claim therefore
//! outlives every engine and every epoch: an engine that succeeds another claims the scopes of
//! its own epoch under the same claim, and its workers hold the same token as the running epoch's,
//! so neither epoch's storage is touched by the other's claim. Each claim is made for its
//! claimant ([`storage::Claimant`]): the engine that asked, by its number, or the updater on its
//! epoch's behalf, so one engine's claim never ends the retention of another's storage. A claim
//! names the epochs whose storage on the node ends (`retire`): the epoch a commit superseded, or
//! the epoch a rollback discarded.
//!
//! The Storage Server serves with the settings the deployment states (`storage_config` and its
//! Mooncake connector), and the daemon keeps it serving with them ([`ClaimHolder::configure`]):
//! a server of the deployment that serves with other settings, one an earlier daemon started or
//! one the deployment's settings changed under, takes the settings it reads when it acts at once
//! (`storage::Client::configure`), and is replaced by a restart in place that serves with the
//! settings otherwise (`storage::Client::restart`), which goes on with every connection, tensor
//! and buffer pool, so the epochs it serves keep their storage.
//!
//! A previous epoch kept in host storage on trial (`KeepWeights` with
//! `to_host`) has its weights moved there by the server's Transfer Server
//! (`Request::Offload`). Every claim of an epoch returns that epoch's weights
//! to the GPUs they left before it gets its response (`Request::Restore`), after
//! the epochs it retires released their storage: an epoch started again at a
//! rollback claims, and its workers then map its weights where they were.

use std::os::fd::{AsFd as _, AsRawFd as _, OwnedFd};
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use proto::process::{Claim, Claimed, WeightsKept};

/// The deployment's claim at its Storage Server on this node, and the settings that server
/// serves with.
pub(crate) struct ClaimHolder {
    /// The node's runtime directory, where the deployment's Storage Server keeps its socket.
    runtime: service::RuntimeDirectory,
    /// The deployment this node's daemon serves, whose Storage Server it claims.
    deployment: String,
    /// Whether processes on other nodes reach this node's server (their
    /// engine's page requests, their Transfer Servers' reads of its
    /// storage): the server it starts then also listens on TCP.
    remote_pages: bool,
    /// This node's address other nodes reach, which names that listener.
    address: Option<std::net::IpAddr>,
    /// The settings of the deployment's Storage Server (`storage_config` and its Mooncake
    /// connector, `kv_cache_config.mooncake`), which a claim that states none of its own takes
    /// ([`Self::settings`]).
    settings: storage::Settings,
    /// The claim, with the `host:port` other nodes reach the server at (empty when they
    /// cannot), as its engines are told; `None` until an engine claimed the node.
    claim: Option<(storage::ClaimClient, String)>,
}

/// A node's claim as it crosses to the daemon's successor in a restart in place: its state and
/// the endpoint its engines were told. The claiming connection's descriptor travels beside it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Handoff {
    runtime: service::RuntimeDirectory,
    deployment: String,
    remote_pages: bool,
    settings: storage::Settings,
    claim: Option<(storage::ClaimHandoff, String)>,
}

/// How long a restart in place of the deployment's Storage Server may take: the server hands
/// itself over once nothing is in flight, which an engine's copies and the connector's tasks
/// end within their deadlines.
const RESTART_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

impl ClaimHolder {
    pub(crate) fn new(
        runtime: service::RuntimeDirectory,
        deployment: String,
        remote_pages: bool,
        address: Option<std::net::IpAddr>,
        settings: storage::Settings,
    ) -> Self {
        Self {
            runtime,
            deployment,
            remote_pages,
            address,
            settings,
            claim: None,
        }
    }

    /// The settings of the deployment's Storage Server, which the leader gives every claim that
    /// states none, so that the Storage Server each node starts for it serves with them.
    pub(crate) fn settings(&self) -> &storage::Settings {
        &self.settings
    }

    /// The socket of the deployment's Storage Server on this node.
    fn socket(&self) -> anyhow::Result<PathBuf> {
        storage::socket(&self.runtime, &self.deployment).with_context(|| {
            format!(
                "finding the Storage Server socket of deployment {}",
                self.deployment
            )
        })
    }

    /// Has the deployment's Storage Server on this node serve with `settings` from now on, and
    /// returns its readiness record: a server is started with them when none serves, a server
    /// that serves with others takes them at once when it can, and is restarted in place with
    /// them otherwise ([`storage::Settings::restarts`]).
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be started, reached, configured or restarted.
    fn serve(&self, settings: &storage::Settings) -> anyhow::Result<storage::Readiness> {
        let socket = self.socket()?;
        let document = settings
            .document()
            .context("writing the Storage Server's settings")?;
        let record: storage::Readiness = service::host::start(
            Path::new(crate::supervision::RUNNING_IMAGE),
            storage::SERVICE,
            &socket,
            &["--settings".to_owned(), "-".to_owned()],
            Some(document.as_bytes()),
        )
        .with_context(|| format!("starting the Storage Server of {}", self.deployment))?;
        if record.settings == *settings {
            return Ok(record);
        }

        let mut operator = storage::Client::connect(&socket, storage::Role::Operator)
            .with_context(|| format!("connecting to the Storage Server at {}", socket.display()))?;
        if record.settings.restarts(settings) {
            tracing::info!(
                deployment = %self.deployment,
                pid = record.pid,
                "the deployment's Storage Server is restarted in place with new settings"
            );
            restart(&mut operator, record.pid, settings)?;
        } else {
            operator
                .configure(settings.server.clone())
                .context("changing the Storage Server's settings")?;
            tracing::info!(
                deployment = %self.deployment,
                pid = record.pid,
                "the deployment's Storage Server serves with new settings"
            );
        }
        service::host::running(&socket)?.with_context(|| {
            format!(
                "the Storage Server of {} ended once its settings changed",
                self.deployment
            )
        })
    }

    /// Has the Storage Server of `deployment` serve with `settings` from now on, as the
    /// deployment states them anew: a server that serves takes them ([`Self::serve`]), and one this node
    /// starts later starts with them. The endpoint other nodes are told of moves with the
    /// server's listener.
    ///
    /// # Errors
    ///
    /// Fails when `deployment` is not this daemon's, or a serving server cannot take the
    /// settings.
    pub(crate) fn configure(
        &mut self,
        deployment: &str,
        settings: storage::Settings,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            deployment == self.deployment,
            "this node's daemon serves deployment {}, not {deployment}",
            self.deployment
        );
        self.settings = settings;
        let socket = self.socket()?;
        if self.claim.is_none() && service::host::running::<storage::Settings>(&socket)?.is_none() {
            return Ok(());
        }
        let record = self.serve(&self.settings)?;
        let tcp = self.tcp(&record);
        if let Some((_, told)) = &mut self.claim {
            *told = tcp;
        }
        Ok(())
    }

    /// Has the deployment's Storage Server on this node run `program` as the deployment's KV
    /// program from now on, or none (`storage::protocol::Request::Program`), and returns how its
    /// instance started. The program's state is saved in the deployment's `program-state/` on
    /// this node ([`crate::policy::revision::PROGRAM_STATE`]). A server is started with the
    /// deployment's settings when none serves: the program runs from now on, whether or not an
    /// epoch claimed the node yet. Running none where no server serves changes nothing.
    ///
    /// # Errors
    ///
    /// Fails when `deployment` is not this daemon's, the state directory has no place, the
    /// server cannot be started or reached, or it refuses the program, with why.
    pub(crate) fn program(
        &mut self,
        deployment: &str,
        program: Option<proto::process::KvProgram>,
    ) -> anyhow::Result<Option<storage::protocol::ProgramStart>> {
        anyhow::ensure!(
            deployment == self.deployment,
            "this node's daemon serves deployment {}, not {deployment}",
            self.deployment
        );
        let program = match program {
            Some(program) => Some(storage::protocol::KvProgram {
                args: serde_json::from_str(&program.args)
                    .context("reading the KV program's args")?,
                name: program.name,
                module: program.module,
                period_ms: program.period_ms,
                tiering: program.tiering,
                state: crate::policy::revision::directory(deployment)?
                    .join(crate::policy::revision::PROGRAM_STATE),
            }),
            None => None,
        };
        let socket = self.socket()?;
        let serving =
            self.claim.is_some() || service::host::running::<storage::Settings>(&socket)?.is_some();
        if program.is_none() && !serving {
            return Ok(None);
        }
        if !serving {
            self.serve(&self.settings)?;
        }

        let started = match &mut self.claim {
            Some((client, _)) => client.program(program),
            None => self.head()?.program(program),
        };
        started.with_context(|| format!("setting the KV program of deployment {deployment}"))
    }

    /// What the deployment's KV program on this node logged and how its outputs fared since the
    /// last call (`storage::protocol::Request::ProgramRecords`); none where no server serves.
    ///
    /// # Errors
    ///
    /// Fails when `deployment` is not this daemon's, or the server cannot be reached.
    pub(crate) fn program_records(
        &mut self,
        deployment: &str,
    ) -> anyhow::Result<Vec<storage::protocol::ProgramRecord>> {
        anyhow::ensure!(
            deployment == self.deployment,
            "this node's daemon serves deployment {}, not {deployment}",
            self.deployment
        );
        let records = match &mut self.claim {
            Some((client, _)) => client.program_records(),
            None => {
                let socket = self.socket()?;
                if service::host::running::<storage::Settings>(&socket)?.is_none() {
                    return Ok(Vec::new());
                }
                self.head()?.program_records()
            }
        };
        records
            .with_context(|| format!("reading the KV program records of deployment {deployment}"))
    }

    /// The spans the deployment's Storage Server on this node keeps in its rings
    /// (`storage::protocol::Request::Spans`); none where no server serves.
    ///
    /// # Errors
    ///
    /// Fails when `deployment` is not this daemon's, or the server cannot be reached.
    pub(crate) fn spans(&mut self, deployment: &str) -> anyhow::Result<Vec<service::span::Span>> {
        anyhow::ensure!(
            deployment == self.deployment,
            "this node's daemon serves deployment {}, not {deployment}",
            self.deployment
        );
        let spans = match &mut self.claim {
            Some((client, _)) => client.spans(),
            None => {
                let socket = self.socket()?;
                if service::host::running::<storage::Settings>(&socket)?.is_none() {
                    return Ok(Vec::new());
                }
                self.head()?.spans()
            }
        };
        spans.with_context(|| format!("reading the Storage Server's spans of {deployment}"))
    }

    /// A connection to the deployment's Storage Server on this node as a head of the deployment
    /// that claims nothing: what sets and reads the KV program before an epoch claimed the node.
    fn head(&self) -> anyhow::Result<storage::Client> {
        let socket = self.socket()?;
        let role = storage::Role::Head {
            deployment: self.deployment.clone(),
            epoch: 0,
            generation: 0,
        };
        storage::Client::connect(&socket, role)
            .with_context(|| format!("connecting to the Storage Server at {}", socket.display()))
    }

    /// The TCP endpoint (`host:port`) other nodes reach this node's Storage Server at, starting
    /// the server with the deployment's settings when none serves; `None` when it accepts no
    /// other hosts.
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be started or its readiness record read.
    pub(crate) fn endpoint(&self) -> anyhow::Result<Option<String>> {
        let socket = self.socket()?;
        let record = match service::host::running::<storage::Settings>(&socket)? {
            Some(record) => record,
            None => self.serve(&self.settings)?,
        };
        let tcp = self.tcp(&record);
        Ok((!tcp.is_empty()).then_some(tcp))
    }

    /// The `host:port` other nodes reach the server of `record` at: this node's address, on the
    /// port it bound; empty when they reach it not.
    fn tcp(&self, record: &storage::Readiness) -> String {
        match (&record.endpoints.peer, self.address) {
            (Some(peer), Some(address)) if self.remote_pages => peer.with_host(address).to_string(),
            _ => String::new(),
        }
    }

    /// Claims the deployment for `claim`'s epoch at its Storage Server on this node as
    /// `claimant`, starting the server when the node has none; a claimant that claimed the same
    /// scopes and retires nothing keeps the claim held. Either way the epoch's weights an
    /// offload left in host storage return to their GPUs first.
    ///
    /// # Errors
    ///
    /// Fails when the claim names another deployment than this daemon's, the server cannot be
    /// started or reached, or it refuses the claim because another head of the deployment
    /// holds it.
    pub(crate) fn claim(
        &mut self,
        claimant: storage::Claimant,
        claim: &Claim,
    ) -> anyhow::Result<Claimed> {
        anyhow::ensure!(
            claim.deployment == self.deployment,
            "this node's daemon serves deployment {}, not {}",
            self.deployment,
            claim.deployment
        );
        let socket = self.socket()?;
        let scopes: Vec<storage::Scope> = claim
            .scopes
            .iter()
            .map(|scope| storage::Scope {
                deployment: claim.deployment.clone(),
                group: scope.group.clone(),
                rank: scope.rank,
            })
            .collect();

        if self.claim.is_none() {
            if let Some(stated) = &claim.storage {
                self.settings = storage::Settings::try_from(stated)
                    .map_err(anyhow::Error::msg)
                    .context("reading the claim's Storage Server settings")?;
            }
            let record = self.serve(&self.settings)?;
            let client =
                storage::ClaimClient::connect(&socket, &claim.deployment).with_context(|| {
                    format!("connecting to the storage server at {}", socket.display())
                })?;
            self.claim = Some((client, self.tcp(&record)));
        }
        let Some((client, tcp)) = &mut self.claim else {
            anyhow::bail!("the deployment's claim is held once it is made");
        };

        let token = client
            .claim(claimant, claim.epoch, scopes, claim.retire.clone())
            .with_context(|| format!("claiming deployment {}", claim.deployment))?;
        // The scopes of the replicas another epoch serves now end in this one.
        if !claim.release.is_empty() {
            let released = claim
                .release
                .iter()
                .map(|scope| storage::Scope {
                    deployment: claim.deployment.clone(),
                    group: scope.group.clone(),
                    rank: scope.rank,
                })
                .collect();
            client
                .unclaim(claim.epoch, released)
                .with_context(|| format!("ending scopes of epoch {}", claim.epoch))?;
        }
        restore(client, claim.epoch)?;
        tracing::info!(
            deployment = %claim.deployment,
            epoch = claim.epoch,
            ?claimant,
            scopes = claim.scopes.len(),
            retired = ?claim.retire,
            "claimed the deployment's storage for an epoch"
        );

        Ok(Claimed {
            socket: socket.to_string_lossy().into_owned(),
            claim: token.0.to_vec(),
            tcp: tcp.clone(),
        })
    }

    /// The claim this node holds of `deployment`.
    fn held(&mut self, deployment: &str) -> anyhow::Result<&mut storage::ClaimClient> {
        self.claim
            .as_mut()
            .filter(|_| deployment == self.deployment)
            .map(|(client, _)| client)
            .with_context(|| format!("this node holds no claim of deployment {deployment}"))
    }

    /// Publishes committed epochs through this node's deployment claim.
    pub(crate) fn publish_epochs(
        &mut self,
        deployment: &str,
        epochs: &[u64],
    ) -> anyhow::Result<()> {
        self.held(deployment)?.publish(epochs.to_vec())?;
        Ok(())
    }

    /// Has this node's Storage Server keep the weights alone of `epochs` of
    /// `deployment` (`Request::KeepWeights`), over the claim this daemon
    /// holds, and with `to_host` move them to host storage
    /// (`Request::Offload`). Returns how many storages it released, the
    /// bytes it moved, and why it kept the weights on the GPUs when the
    /// server refused or failed the move: the weights then stay where they
    /// are, as a previous epoch kept as its weights keeps them.
    ///
    /// # Errors
    ///
    /// Fails when this daemon holds no claim of the deployment, or the
    /// server cannot be reached or refuses to keep the weights alone.
    pub(crate) fn keep_weights(
        &mut self,
        deployment: &str,
        epochs: &[u64],
        to_host: bool,
        cache_keys: &[Vec<u8>],
    ) -> anyhow::Result<WeightsKept> {
        let client = self.held(deployment)?;
        let cache_keys = cache_keys
            .iter()
            .map(|key| {
                key.as_slice()
                    .try_into()
                    .map(storage::Key)
                    .map_err(|_| anyhow::anyhow!("a retained cache key has 32 bytes"))
            })
            .collect::<anyhow::Result<_>>()?;
        let released = client
            .keep_weights(epochs.to_vec(), cache_keys)
            .with_context(|| {
                format!("retaining weights and rollback cache of epochs {epochs:?}")
            })?;
        tracing::info!(
            deployment,
            ?epochs,
            released,
            "epochs keep their weights alone"
        );

        let mut kept = WeightsKept {
            released,
            ..WeightsKept::default()
        };
        if !to_host {
            return Ok(kept);
        }
        for &epoch in epochs {
            match client.offload(epoch) {
                Ok((storages, bytes)) => {
                    kept.moved_bytes += bytes;
                    tracing::info!(
                        deployment,
                        epoch,
                        storages,
                        bytes,
                        "an epoch's weights moved to host storage"
                    );
                }
                // The refusal is the answer's: the update records it in its status
                // (`host_refused`), and the epoch's weights stay where they are.
                Err(error) => {
                    tracing::warn!(deployment, epoch, %error, "an epoch's weights stay on the GPUs");
                    kept.refused = format!("epoch {epoch}: {error}");
                    break;
                }
            }
        }
        Ok(kept)
    }

    /// Halves, or restores when not `halve`, the byte budgets of the links
    /// `deployment`'s tasks use at this node's Transfer Server, over the
    /// claim this daemon holds: a halving takes every link a task of the
    /// deployment copies over now, a restoration every link a halving
    /// reduced (`Request::Throttle` with no links).
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be reached or runs no Transfer Server.
    pub(crate) fn throttle(&mut self, deployment: &str, halve: bool) -> anyhow::Result<()> {
        // A node that holds no claim of the deployment runs none of its
        // tasks: no epoch of it has claimed the node yet.
        let Some((client, _)) = self
            .claim
            .as_mut()
            .filter(|_| deployment == self.deployment)
        else {
            tracing::debug!(
                deployment,
                "no claim of the deployment: nothing to throttle"
            );
            return Ok(());
        };
        let throttle = if halve {
            storage::protocol::Throttle::Halve
        } else {
            storage::protocol::Throttle::Restore
        };
        client
            .throttle(throttle)
            .with_context(|| format!("throttling the tasks of deployment {deployment}"))?;
        tracing::info!(
            deployment,
            ?throttle,
            "changed the budgets of the tasks' links"
        );
        Ok(())
    }

    /// The quotas this node's Storage Server keeps for `deployment`, one
    /// per GPU of the node it uses, what the host admits of new host
    /// storage, and what the deployment's tasks copied on the host by
    /// priority, read over the claim this daemon holds.
    ///
    /// # Errors
    ///
    /// Fails when this daemon holds no claim of the deployment, or the
    /// server cannot be reached.
    pub(crate) fn usage(&mut self, deployment: &str) -> anyhow::Result<storage::protocol::Usage> {
        self.held(deployment)?
            .usage()
            .with_context(|| format!("reading the quotas of deployment {deployment}"))
    }

    /// Hands the claim to this daemon's successor ([`ClaimHolder::adopt`]): its state, and the
    /// claiming connection's descriptor in the second value. This daemon holds no claim
    /// afterwards.
    ///
    /// # Errors
    ///
    /// Fails when the claiming connection has unread messages.
    pub(crate) fn handoff(&mut self) -> anyhow::Result<(Handoff, Vec<OwnedFd>)> {
        let mut descriptors = Vec::new();
        let claim = match self.claim.take() {
            Some((client, tcp)) => {
                let (claim, descriptor) = client.handoff()?;
                descriptors.push(descriptor);
                Some((claim, tcp))
            }
            None => None,
        };
        let handoff = Handoff {
            runtime: self.runtime.clone(),
            deployment: self.deployment.clone(),
            remote_pages: self.remote_pages,
            settings: self.settings.clone(),
            claim,
        };
        Ok((handoff, descriptors))
    }

    /// Goes on with the claim a predecessor handed over, with its connection's descriptor;
    /// `address` is this node's.
    pub(crate) fn adopt(
        address: Option<std::net::IpAddr>,
        handoff: Handoff,
        descriptors: Vec<OwnedFd>,
    ) -> storage::Result<Self> {
        let mut holder = Self::new(
            handoff.runtime,
            handoff.deployment,
            handoff.remote_pages,
            address,
            handoff.settings,
        );
        if let (Some((claim, tcp)), Some(descriptor)) =
            (handoff.claim, descriptors.into_iter().next())
        {
            holder.claim = Some((storage::ClaimClient::resume(claim, descriptor)?, tcp));
        }
        Ok(holder)
    }
}

/// Restarts the Storage Server of process `predecessor`, reached by `operator`, in place with
/// `settings`, from the running build's image, and returns once the successor serves: the
/// predecessor exits once it handed itself over. A successor that gives the restart up ends
/// first, and the predecessor goes on serving with its settings.
///
/// # Errors
///
/// Fails when the image cannot be opened, the server refuses the restart, the successor gives
/// it up, or the handover does not end within [`RESTART_TIMEOUT`].
fn restart(
    operator: &mut storage::Client,
    predecessor: u32,
    settings: &storage::Settings,
) -> anyhow::Result<()> {
    let image = std::fs::File::open(crate::supervision::RUNNING_IMAGE)
        .context("opening the running build's image")?;
    let mut arguments = service::logging::arguments();
    arguments.push(storage::SERVICE.to_owned());
    let successor = operator
        .restart(image.as_fd(), arguments, Some(settings.clone()))
        .context("restarting the Storage Server in place")?;
    let pidfd = |pid: u32| -> anyhow::Result<OwnedFd> {
        let pid = rustix::process::Pid::from_raw(pid as i32)
            .with_context(|| format!("{pid} names no process"))?;
        Ok(rustix::process::pidfd_open(
            pid,
            rustix::process::PidfdFlags::empty(),
        )?)
    };
    let (predecessor, successor) = (pidfd(predecessor)?, pidfd(successor)?);
    let mut descriptors = [predecessor.as_raw_fd(), successor.as_raw_fd()].map(|fd| libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    });
    // Whichever process ends first decides the restart: the predecessor once it handed over,
    // the successor once it gave up.
    service::wait_ready(&mut descriptors, Some(RESTART_TIMEOUT))?;
    anyhow::ensure!(
        descriptors.iter().any(|descriptor| descriptor.revents != 0),
        "the Storage Server did not hand itself over within {RESTART_TIMEOUT:?}"
    );
    anyhow::ensure!(
        descriptors[0].revents != 0,
        "the Storage Server's successor gave the restart up; the server serves with its \
         earlier settings (its log tells why)"
    );
    Ok(())
}

/// Returns `epoch`'s weights that an offload left in host storage to the
/// GPUs they left (`Request::Restore`), which responds at once when there
/// are none.
fn restore(client: &mut storage::ClaimClient, epoch: u64) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let (storages, bytes) = client
        .restore(epoch)
        .with_context(|| format!("returning the weights of epoch {epoch} to its GPUs"))?;
    if storages > 0 {
        tracing::info!(
            deployment = client.deployment(),
            epoch,
            storages,
            bytes,
            "an epoch's weights returned from host storage"
        );
    }
    Ok(())
}
