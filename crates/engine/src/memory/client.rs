//! The engine's client of its host's Storage Server (`specs/storage.md` §3.2): one
//! [`StorageClient`] per connection, the head's of the epoch or a rank's.
//!
//! **The head.** The engine's main process is the head of its epoch on the host's Storage Server
//! ([`storage::Role::Head`], [`StorageClient::connect_head`]). Over that connection it opens the
//! epoch's session with the deployment's buffer pool (`Open`, `specs/buffer-pool.md` §11) and
//! closes it once its workers have stopped (`Close`), and it states what only the head knows: a
//! segment of a device region its ranks created ([`StorageClient::grant`]), one they are about
//! to release ([`StorageClient::withdraw`]), and where a rank of another host keeps its part of a
//! region's units ([`StorageClient::place_remote`], BP §15).
//!
//! The answer to `Open` carries one queue pair per scheduler: the engine's end of each is mapped
//! here and handed to its scheduler, whose submissions and completions travel in it with no system
//! call. The same queue pair holds the registers of its scheduler's groups: each group's
//! `completed` word, which the group's workers write through a mapping of their own
//! ([`StorageClient::completed_registers`]), and the capacity words the buffer pool writes.
//!
//! **A rank.** Everything a rank would lose by exiting, and that costs time or memory to rebuild,
//! is storage the host's Storage Server retains: the weights, which every process that maps them
//! shares read-only, and the exclusive storages of the rank's scope (the cache planes, the memory
//! of a buffer pool region). Each rank of the engine speaks to the same server on a connection of
//! its own ([`storage::Role::Worker`], [`StorageClient::connect_rank`]), which the
//! [`StorageLaunch`] names. The owners of exclusive storage take it through the client
//! ([`ExclusiveStorage`]), the weight store publishes and maps weights through it
//! ([`WeightStore`](crate::memory::weights::WeightStore)), and the rank's share of its GPU's quota
//! is stated and read through it. Over it the rank also states where its part of a device region's
//! units lies ([`StorageClient::place`]) and takes its part of a host region
//! ([`StorageClient::host_part`]), BP §14.2.
//!
//! ## A region's storage across processes and epochs
//!
//! A cache plane's storage is keyed by its region's memory and what fixes its shape: a segment's
//! part by the segment the region's segment table names (the epoch that created it and its
//! ordinal), and a storage that is not split into segments by the epoch of the table's first
//! segment ([`ExclusiveConfig::epoch`]). Every epoch that names the region, a standby and a
//! restarted engine of the same scope map the same storages, writable, with an access that
//! records no writer (`Mode::Region`): the buffer pool, not a writer, keeps the blocks they
//! write apart, so nothing of the cache is copied or handed over between them.
//!
//! A rank makes no Storage Server call while it serves: only at startup, when the segments of
//! its planes are given back or created again, and at its orderly close.

use std::collections::{BTreeMap, BTreeSet};
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, bail};
use platform::{DeviceId, Protection};
use proto::request::StandbyStorage;
use serde::{Deserialize, Serialize};
use service::lock;
use storage::protocol::{Access, ErrorKind, TensorListing, UnitRun};
use storage::runtime::{GroupId, Open, Opened, Segments, Tier};
use storage::{
    Claim, Client, HostStorage, Item, Key, Location, Mode, QueuePair, Quota, Role, Scope, Segment,
    StorageRecord, TensorRecord,
};

use crate::memory::allocation::Storage;
use crate::memory::cache::info::CachePlaneRun;
use crate::memory::exclusive::{ExclusiveConfig, ExclusiveStorage, SegmentPart, SegmentedConfig};
use crate::worker::{Error, Result};

/// The Storage Server a rank retains its storage at, and its deployment's claim there: what a
/// rank's connection is made from ([`StorageClient::connect_rank`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageLaunch {
    /// The Unix socket of the worker's host's Storage Server.
    pub socket: String,
    /// The deployment the worker's scope belongs to: the served model name.
    pub deployment: String,
    /// The token of the claim the head holds at that server, which the
    /// worker states so that the server knows it belongs to the current
    /// head.
    pub claim: [u8; 16],
    /// Where an engine on another host reaches the server (`host:port`), for
    /// page requests; `None` on the engine's host, or when the server accepts
    /// no other host.
    pub tcp: Option<String>,
    /// Where processes of this host reach the deployment's Storage Servers
    /// on its other nodes (`host:port`), each of which may hold the storage
    /// of the deployment's other epochs: the weights a rank receives from
    /// another host. Empty for a deployment of one node.
    pub peers: Vec<String>,
    /// The epoch the worker belongs to: the exclusive storages it creates
    /// are its epoch's, and it retains what it uses in its epoch.
    pub epoch: u64,
    /// Whether the worker stands by for the worker of its scope: it maps
    /// the exclusive storages that worker holds without taking write
    /// access, warms up on scratch units, parks with the storages read-only
    /// and serves once promoted.
    pub standby: bool,
    /// How the worker, parked as a standby, keeps the device storage of its
    /// graph pools and scratch (`worker_config.standby_storage`).
    pub standby_storage: StandbyStorage,
}

/// A deployment's limit of a GPU of `total` bytes: `fraction` of the whole of it
/// (`mem_fraction_static`), rounded down, as the Storage Server's quota states it.
pub fn quota_limit(total: u64, fraction: f64) -> u64 {
    (total as f64 * fraction) as u64
}

/// A connection to the host's Storage Server (module documentation): the head's of the epoch,
/// with the session it opened with the deployment's buffer pool, or a rank's, with what the rank
/// retains there and its share of each GPU's quota.
///
/// One client serves every owner of a rank, behind one lock, so two owners' requests never
/// interleave on the connection. The head's session changes only through `&mut self`.
pub struct StorageClient {
    /// The deployment the connection belongs to.
    deployment: String,
    /// The rank's scope, which keys and retains its storage; `None` on the head's connection,
    /// which retains nothing of a scope.
    scope: Option<Scope>,
    /// The epoch the connection belongs to, which retains what a rank uses and keys the
    /// storages of no segmented region.
    epoch: u64,
    state: Mutex<State>,
    /// The layout the head opened the session with and the buffer pool's answer to it; `None`
    /// on a rank's connection and until the head opens the session.
    session: Option<(Open, Opened)>,
}

/// What the connection holds between requests.
struct State {
    client: Client,
    /// The GPU UUID of each device the rank addressed, by ordinal.
    gpus: BTreeMap<i32, [u8; 16]>,
    /// The exclusive storages the rank took and maps.
    taken: BTreeSet<Key>,
}

impl State {
    /// The UUID of `device`, which retained keys and records name.
    fn gpu(&mut self, device: DeviceId) -> Result<[u8; 16]> {
        if let Some(uuid) = self.gpus.get(&device.0) {
            return Ok(*uuid);
        }

        let uuid = device.uuid()?.0;
        self.gpus.insert(device.0, uuid);
        Ok(uuid)
    }

    /// Where the rank's retained storage lies: the GPUs it addressed, and host storage. A key
    /// has a copy at each location that holds it, so the rank retains and releases the copies
    /// at its own locations and leaves other ranks' copies to them.
    fn locations(&self) -> Vec<Location> {
        self.gpus
            .values()
            .map(|gpu| Location::Device { gpu: *gpu })
            .chain(std::iter::once(Location::Host { numa_node: None }))
            .collect()
    }
}

impl StorageClient {
    /// Connects to the Storage Server at `socket` as the head of `epoch` of `deployment`: the
    /// main process of the engine its daemon started at launch `generation`
    /// (`EngineArgs::engine`), whose ranks state the same generation. The generation names the
    /// engine's session among those of its epoch (a standby's beside the serving engine's).
    ///
    /// # Errors
    ///
    /// When the server cannot be reached or refuses the head.
    pub fn connect_head(
        socket: &Path,
        deployment: &str,
        epoch: u64,
        generation: u64,
    ) -> anyhow::Result<Self> {
        let role = Role::Head {
            deployment: deployment.to_owned(),
            epoch,
            generation,
        };
        let client = Client::connect(socket, role).with_context(|| {
            format!(
                "connecting to the Storage Server at {} as the head of epoch {epoch}",
                socket.display()
            )
        })?;

        Ok(Self::over(client, deployment.to_owned(), None, epoch))
    }

    /// Connects as rank `rank` of worker group `group`, started by the group's launch
    /// `generation`, to the server `launch` names.
    ///
    /// # Errors
    ///
    /// When the server cannot be reached or refuses the rank.
    pub fn connect_rank(
        launch: &StorageLaunch,
        group: &str,
        rank: u32,
        generation: u64,
    ) -> Result<Arc<Self>> {
        let scope = Scope {
            deployment: launch.deployment.clone(),
            group: group.to_owned(),
            rank,
        };
        let client = Client::connect(
            Path::new(&launch.socket),
            Role::Worker {
                scope: scope.clone(),
                generation,
                claim: Claim(launch.claim),
                epoch: launch.epoch,
            },
        )
        .map_err(|error| storage_error("connecting to the storage server", error))?;

        Ok(Arc::new(Self::over(
            client,
            launch.deployment.clone(),
            Some(scope),
            launch.epoch,
        )))
    }

    /// The client over `client`, a connection of `deployment` in `epoch`, of the rank of `scope`
    /// or, without one, of the head.
    fn over(client: Client, deployment: String, scope: Option<Scope>, epoch: u64) -> Self {
        Self {
            deployment,
            scope,
            epoch,
            state: Mutex::new(State {
                client,
                gpus: BTreeMap::new(),
                taken: BTreeSet::new(),
            }),
            session: None,
        }
    }

    /// The rank's scope, which its epoch's quota reservations name; `None` on the head's
    /// connection.
    pub fn scope(&self) -> Option<&Scope> {
        self.scope.as_ref()
    }

    /// The scope of the rank whose connection this is, which keys what it retains.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] on the head's connection, which retains nothing of a scope.
    fn rank_scope(&self) -> Result<&Scope> {
        self.scope.as_ref().ok_or_else(|| {
            Error::Invalid("the head's connection retains no storage of a rank".to_owned())
        })
    }

    /// The connection's epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Makes one request on the connection, as the rank doing `what`.
    ///
    /// # Errors
    ///
    /// The server's refusal or the connection's failure, stated with `what`.
    pub fn call<T>(
        &self,
        what: &str,
        request: impl FnOnce(&mut Client) -> storage::Result<T>,
    ) -> Result<T> {
        request(&mut lock(&self.state).client).map_err(|error| storage_error(what, error))
    }

    /// The UUID of `device`, which the keys and records of its storage name.
    pub(super) fn gpu(&self, device: DeviceId) -> Result<[u8; 16]> {
        lock(&self.state).gpu(device)
    }

    /// Ends the epoch's retention of `keys` at the rank's locations, as the rank doing `doing`.
    pub(super) fn end_retention(&self, keys: &BTreeSet<Key>, doing: &str) -> Result<()> {
        self.release_keys(&mut lock(&self.state), keys, doing)
    }

    /// The key of an exclusive storage of the rank: the digest of its scope, the epoch of its
    /// region's memory (the rank's own for a storage of no segmented region), its name, its
    /// length and its GPU. It covers the storage's shape only, so every epoch that names the
    /// region, and a restart of the epoch, maps the storage again whatever else changed.
    fn exclusive_key(&self, config: &ExclusiveConfig<'_>, gpu: &[u8; 16]) -> Result<Key> {
        Ok(storage::key::exclusive(
            self.rank_scope()?,
            config.name,
            config.bytes as u64,
            gpu,
            config.epoch.unwrap_or(self.epoch),
        ))
    }

    /// The key of one part of a segmented storage, as the exclusive storage
    /// `{name}#{column}/{ordinal}` of the part's length that the segment's epoch created.
    fn part_key(&self, name: &str, part: &SegmentPart, gpu: &[u8; 16]) -> Result<Key> {
        Ok(storage::key::exclusive(
            self.rank_scope()?,
            &segment_name(name, part),
            part.length as u64,
            gpu,
            part.segment.epoch,
        ))
    }

    /// The scope's retained device bytes on `device`, which NVML attributes to no process: the
    /// bytes of the storages the rank uses (its weights and the exclusive storages it took),
    /// and the bytes of the exclusive storages a predecessor left that the rank has not taken.
    ///
    /// The owners take the second again, or release them when their shape changed, before
    /// allocating storage of their own, so a rank sizing its memory counts the first as its
    /// own and the second as free.
    pub(crate) fn scope_bytes(&self, device: DeviceId) -> Result<(u64, u64)> {
        let scope = self.rank_scope()?;
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;

        let total: u64 = listing
            .device_bytes
            .iter()
            .filter(|entry| {
                entry.gpu == Some(gpu) && entry.scope == *scope && entry.epoch == self.epoch
            })
            .map(|entry| entry.bytes)
            .sum();

        // Every exclusive tensor is a storage of its own. Those another epoch uses, as the
        // storages of a region two epochs name, are not free.
        let untaken: u64 = listing
            .tensors
            .iter()
            .filter(|tensor| {
                tensor.access == Access::Exclusive
                    && tensor.location == Location::Device { gpu }
                    && self.retains(tensor)
                    && tensor
                        .retainers
                        .iter()
                        .all(|retainer| retainer.epoch == self.epoch)
                    && !state.taken.contains(&tensor.key)
            })
            .map(|tensor| tensor.storage_bytes)
            .sum();

        Ok((total.saturating_sub(untaken), untaken))
    }

    /// Releases the exclusive storages a predecessor left that no owner of this rank took and
    /// that `taking` does not take now. An owner that allocates new storage calls this first:
    /// [`Self::scope_bytes`] counted those storages as free, and an owner whose shape changed,
    /// or that this rank does not have, leaves them to no one.
    fn release_untaken(&self, state: &mut State, taking: &BTreeSet<Key>) -> Result<()> {
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;

        let mut keep = Vec::new();
        let mut untaken = 0;
        for tensor in listing.tensors.iter().filter(|tensor| self.retains(tensor)) {
            if tensor.access == Access::Exclusive
                && !state.taken.contains(&tensor.key)
                && !taking.contains(&tensor.key)
            {
                untaken += 1;
            } else {
                keep.push(tensor.key);
            }
        }

        if untaken > 0 {
            let at = state.locations();
            state
                .client
                .retain(keep, at)
                .map_err(|error| storage_error("releasing untaken storages", error))?;
            tracing::info!(
                released = untaken,
                "released exclusive storages no owner takes again"
            );
        }
        Ok(())
    }

    /// Ends the epoch's retention of `keys` alone. The server's retention is the scope's in the
    /// epoch, which every worker of the scope shares, and the rank's owners take their storages
    /// one after another in its startup, so every other storage the scope retains stays.
    fn release_keys(&self, state: &mut State, keys: &BTreeSet<Key>, doing: &str) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }

        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;
        let keep: Vec<Key> = listing
            .tensors
            .iter()
            .filter(|tensor| self.retains(tensor) && !keys.contains(&tensor.key))
            .map(|tensor| tensor.key)
            .collect();

        let at = state.locations();
        state
            .client
            .retain(keep, at)
            .map_err(|error| storage_error(doing, error))?;
        Ok(())
    }

    /// Whether the rank's scope retains `tensor` in the rank's epoch; the head's connection
    /// retains nothing.
    fn retains(&self, tensor: &TensorListing) -> bool {
        tensor.retainers.iter().any(|retainer| {
            self.scope.as_ref() == Some(&retainer.scope) && retainer.epoch == self.epoch
        })
    }

    /// States this epoch's quota of the GPU of `device`, `fraction` of the whole of it: every
    /// storage retained there from now on fits it or is refused, and the cache's segments are
    /// given back against it. What this rank holds already stays.
    pub(crate) fn state_quota(&self, device: DeviceId, fraction: f64) -> Result<()> {
        let total = device.memory_info()?.total;

        self.reserve(device, total, quota_limit(total, fraction), Vec::new())
            .map(drop)
    }

    /// The deployment's quota of the GPU of `device`, and what the host admits of new host
    /// storage.
    pub fn usage(&self, device: DeviceId) -> Result<(Quota, HostStorage)> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let usage = state
            .client
            .usage(vec![gpu])
            .map_err(|error| storage_error("reading the quota", error))?;

        Ok((self.of_deployment(gpu, usage.quotas), usage.host))
    }

    /// Reserves `items` of the rank on the GPU of `device` of `total` bytes, stating the
    /// epoch's limit `quota` there, and returns the deployment's quota of the GPU after it.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] with the server's refusal, which names every reservation of the
    /// quota, when the reservation would take the GPU's usage past its limit.
    pub(crate) fn reserve(
        &self,
        device: DeviceId,
        total: u64,
        quota: u64,
        items: Vec<(Item, u64)>,
    ) -> Result<Quota> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let quotas = state
            .client
            .reserve(gpu, total, Some(quota), items)
            .map_err(|error| match error {
                storage::Error::Server {
                    kind: ErrorKind::Quota,
                    message,
                } => Error::Unsupported(format!(
                    "the device storage the rank holds on {device} does not fit its quota: \
                     {message}"
                )),
                error => storage_error("reserving device storage", error),
            })?;

        Ok(self.of_deployment(gpu, quotas))
    }

    /// The quota of the rank's deployment among `quotas` of `gpu`; an empty one when the
    /// server keeps none yet.
    fn of_deployment(&self, gpu: [u8; 16], quotas: Vec<Quota>) -> Quota {
        quotas
            .into_iter()
            .find(|quota| quota.deployment == self.deployment)
            .unwrap_or(Quota {
                gpu,
                deployment: self.deployment.clone(),
                total: 0,
                limit: None,
                reservations: Vec::new(),
                host_link_bytes_per_second: None,
                transfer_context: 0,
            })
    }

    /// Publishes the storage of part `index` of `config`, which `descriptor` maps, as an
    /// exclusive tensor of the rank's scope.
    fn publish_segment(
        state: &mut State,
        config: &SegmentedConfig<'_>,
        index: usize,
        key: Key,
        gpu: [u8; 16],
        descriptor: &OwnedFd,
    ) -> Result<()> {
        let part = &config.parts[index];
        let bytes = part.length as u64;

        let record = StorageRecord {
            location: Location::Device { gpu },
            bytes,
            // The quota reserves the storages of one segment together, as the cache's.
            segment: Some(Segment {
                epoch: part.segment.epoch,
                ordinal: part.segment.ordinal,
            }),
            fabric: None,
        };
        let tensor = byte_tensor(
            key,
            segment_name(config.name, part),
            0,
            bytes,
            Access::Exclusive,
        );

        state
            .client
            .put(vec![tensor], &[(record, descriptor.as_fd())])
            .map_err(|error| storage_error("publishing a segment", error))
    }
}

/// The head's session with the deployment's buffer pool, and what the head states of the
/// memory spaces of its device regions.
impl StorageClient {
    /// Opens the epoch's session with the buffer pool of its deployment, stating `layout`.
    /// Returns the engine's end of each scheduler's queue pair, in the order of
    /// `layout.schedulers`; the layout and the answer are kept ([`Self::session`]).
    ///
    /// The segments each device region of `layout` states were created by the engine's ranks
    /// before this call, on every rank that maps the region (BP §14.2). A region an earlier
    /// session named keeps the numbering the buffer pool holds for it (`Opened::limits`), which
    /// may differ from the one `layout` states when the epochs measured their devices apart: the
    /// kept layout states each region with the numbering the answer states, which the ranks lay
    /// their memory out by and the schedulers grow their regions to.
    ///
    /// # Errors
    ///
    /// A session open already; the server's refusal of the layout; an answer that does not state
    /// every region and scheduler of the layout, or a queue pair that cannot be mapped.
    pub fn open(&mut self, mut layout: Open) -> anyhow::Result<Vec<QueuePair>> {
        if self.session.is_some() {
            bail!(
                "the session of epoch {} with the buffer pool is open already",
                layout.epoch
            );
        }

        let (opened, descriptors) = lock(&self.state)
            .client
            .open_pool(layout.clone())
            .with_context(|| format!("opening epoch {}'s session", layout.epoch))?;
        if opened.regions.len() != layout.regions.len()
            || opened.segments.len() != layout.regions.len()
            || opened.limits.len() != layout.regions.len()
        {
            bail!(
                "a layout of {} regions was answered for {} regions, {} segment tables and {} \
                 numberings",
                layout.regions.len(),
                opened.regions.len(),
                opened.segments.len(),
                opened.limits.len()
            );
        }
        for (region, &limit) in layout.regions.iter_mut().zip(&opened.limits) {
            region.limit = limit;
        }

        // The engine's end of each queue pair: the scheduler writes its submission queue and
        // reads its completion queue.
        let pairs = descriptors
            .into_iter()
            .map(|descriptor| {
                QueuePair::open(descriptor, true).context("mapping a scheduler's queue pair")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.session = Some((layout, opened));
        Ok(pairs)
    }

    /// States that the session's region at `index` serves `units` units from here on, fewer than
    /// it asked for at `Open`: the slot rows and the product buffers of a media group whose warm
    /// ranks kept fewer request slots than they warmed up with and gave the storages of the rest
    /// back (`crate::process::ranks::keep_warm_slots`). The region keeps its numbering; its
    /// schedulers resize it to these units (`crate::process::epoch::grow_regions`), and no grow
    /// takes it past them.
    ///
    /// # Errors
    ///
    /// No session, or no region at `index`.
    pub(crate) fn keep_units(&mut self, index: usize, units: u32) -> anyhow::Result<()> {
        let (layout, _) = self
            .session
            .as_mut()
            .context("the head opened no session whose region keeps units")?;
        let region = layout
            .regions
            .get_mut(index)
            .with_context(|| format!("the session states no region {index}"))?;
        region.units = region.units.min(units);
        Ok(())
    }

    /// The layout the session was opened with, each region numbered as the buffer pool holds it,
    /// and the buffer pool's answer to it; `None` before the session is open and on a rank's
    /// connection. The answer's segment table of each region
    /// is the one the Storage Server holds for the region's memory space: as `Open` answered it,
    /// with the segments this head granted since ([`Self::grant`]) and without those it withdrew
    /// ([`Self::withdraw`]).
    pub(crate) fn session(&self) -> Option<(&Open, &Opened)> {
        self.session
            .as_ref()
            .map(|(layout, opened)| (layout, opened))
    }

    /// Where the `completed` word of `group` lies: the engine's mapping of the queue pair of the
    /// scheduler that lists the group, mapped again from `pairs` (the queue pairs [`Self::open`]
    /// returned, in their order), and the word's index among its registers. What the group's
    /// workers write their ended sequence into (`crate::worker::sequences::Sequences::new`); the
    /// mapping is theirs for as long as they may write it.
    ///
    /// # Errors
    ///
    /// A session not open, a group no scheduler of the layout lists, or a queue pair that cannot
    /// be mapped again.
    pub(crate) fn completed_registers(
        &self,
        group: GroupId,
        pairs: &[QueuePair],
    ) -> anyhow::Result<Vec<(QueuePair, usize)>> {
        let (layout, opened) = self
            .session()
            .context("the session with the buffer pool is not open")?;

        let mut registers = Vec::new();
        for (scheduler, groups) in layout.schedulers.iter().enumerate() {
            let Some(place) = groups.iter().position(|listed| *listed == group) else {
                continue;
            };
            let index = opened
                .registers
                .get(scheduler)
                .and_then(|registers| registers.completed.get(place))
                .with_context(|| {
                    format!(
                        "the answer states no completed word of group {} of scheduler \
                         {scheduler}",
                        group.0
                    )
                })?;
            let pair = pairs
                .get(scheduler)
                .with_context(|| format!("scheduler {scheduler} has no queue pair"))?;

            let descriptor = pair
                .descriptor()
                .try_clone_to_owned()
                .context("duplicating a queue pair's descriptor")?;
            let mapping =
                QueuePair::open(descriptor, true).context("mapping a queue pair's registers")?;
            registers.push((mapping, *index as usize));
        }

        if registers.is_empty() {
            bail!("no scheduler of the session lists group {}", group.0);
        }
        Ok(registers)
    }

    /// Ends the session: the epoch's workers have stopped, and the buffer pool does what it
    /// does for an engine that is gone (BP §12.4).
    ///
    /// # Errors
    ///
    /// A session not open, or the server's refusal.
    pub(crate) fn close(&mut self) -> anyhow::Result<()> {
        if self.session.is_none() {
            bail!("the session with the buffer pool is not open");
        }

        lock(&self.state)
            .client
            .close_pool()
            .context("closing the session with the buffer pool")?;
        self.session = None;
        Ok(())
    }

    /// States that every rank that maps the device region `region` created the storages of
    /// `segment`, `units` units from `first` of the region's memory space: they join the space's
    /// free ranges, from which a resize grows its regions.
    ///
    /// # Errors
    ///
    /// The server's refusal: the region is no device region of the deployment's buffer pool, or
    /// a unit of the segment lies past its numbering or in a region already.
    pub(crate) fn grant(
        &mut self,
        region: &str,
        segment: Segment,
        first: u32,
        units: u32,
    ) -> anyhow::Result<()> {
        lock(&self.state)
            .client
            .grant(region.to_owned(), segment, first, units)
            .with_context(|| {
                format!(
                    "granting segment {}/{} of region {region}",
                    segment.epoch, segment.ordinal
                )
            })?;

        // The space's table holds the segment from here on, in the order of its first unit.
        self.change_table(region, |table| {
            let at = table.partition_point(|(_, from, _)| *from < first);
            table.insert(at, (segment, first, units));
        });
        Ok(())
    }

    /// States that the ranks are about to release the storages of `segment` of the memory space
    /// of the device region `region`: once this returns, no region grows into its units until a
    /// grant.
    ///
    /// # Errors
    ///
    /// The server's refusal: a unit of the segment is in a region (a resize shrinks the regions
    /// first), or the space's table holds no such segment.
    pub(crate) fn withdraw(&mut self, region: &str, segment: Segment) -> anyhow::Result<()> {
        lock(&self.state)
            .client
            .withdraw(region.to_owned(), segment)
            .with_context(|| {
                format!(
                    "withdrawing segment {}/{} of region {region}",
                    segment.epoch, segment.ordinal
                )
            })?;

        self.change_table(region, |table| {
            table.retain(|(held, _, _)| *held != segment);
        });
        Ok(())
    }

    /// Applies `change` to the segment table of the memory space of the region named `region`,
    /// as the session's answer states it for every region of that space: the Storage Server keeps
    /// one table per space. Nothing changes before the session is open, or for a region it does
    /// not state.
    fn change_table(&mut self, region: &str, change: impl Fn(&mut Segments)) {
        let Some((layout, opened)) = self.session.as_mut() else {
            return;
        };
        let Some(space) = layout
            .regions
            .iter()
            .find(|stated| stated.name == region)
            .map(|stated| stated.space.clone())
        else {
            return;
        };

        let tables = layout.regions.iter().zip(opened.segments.iter_mut());
        for (_, table) in tables.filter(|(stated, _)| stated.space == space) {
            change(table);
        }
    }

    /// States where part `part` of the units of the device region `region` lies in the device
    /// memory `memory`, for its rank of the host whose Storage Server listens at `endpoint` (the
    /// rank whose index in its worker group is `part`): `runs`, as that rank stated them for its
    /// own server ([`unit_runs`]), each with the location of its storage there.
    ///
    /// # Errors
    ///
    /// The server's refusal.
    pub(crate) fn place_remote(
        &mut self,
        region: &str,
        memory: &str,
        part: u32,
        endpoint: &str,
        runs: &[UnitRun],
    ) -> anyhow::Result<()> {
        lock(&self.state)
            .client
            .place_remote(
                region.to_owned(),
                memory.to_owned(),
                part,
                endpoint.to_owned(),
                runs.to_vec(),
            )
            .with_context(|| {
                format!("placing part {part} of {endpoint} in region {region} of memory {memory}")
            })
    }

    /// Opens, on a peer's host, the session of the host regions of `layout`, the layout the
    /// scheduler node opened its session with at its own host (`specs/buffer-pool.md` §15): the
    /// host regions alone, with no class, group, scheduler or device region, so that this host's
    /// Storage Server creates the parts of them its ranks take and follows the epoch's processes
    /// for the session's close. Its buffer pool holds no table anyone reads: the scheduler node's
    /// holds the regions' tables.
    ///
    /// # Errors
    ///
    /// As [`Self::open`].
    pub(crate) fn open_hosts(&mut self, layout: &Open) -> anyhow::Result<()> {
        let hosts = Open {
            regions: layout
                .regions
                .iter()
                .filter(|region| region.tier == Tier::Host)
                .cloned()
                .collect(),
            classes: Vec::new(),
            trajectory: 0,
            groups: Vec::new(),
            schedulers: Vec::new(),
            peers: Vec::new(),
            ..layout.clone()
        };
        self.open(hosts).map(drop)
    }
}

/// A rank's parts of the buffer pool's regions.
impl StorageClient {
    /// States where the rank's part of the units of the device region `region` lies in the
    /// device memory `memory` its group addresses: `runs`, each field of a unit as the rank's
    /// cache planes hold it (`KvCacheMemory::info`). A transfer or a store that touches the
    /// rank's units is turned into copies with it.
    ///
    /// # Errors
    ///
    /// The server's refusal or the connection's failure.
    pub(crate) fn place(
        &self,
        region: &str,
        memory: &str,
        part: u32,
        runs: &[UnitRun],
    ) -> Result<()> {
        self.call("placing the rank's units", |client| {
            client.place(region.to_owned(), memory.to_owned(), part, runs.to_vec())
        })
    }

    /// Takes part `part` of the host region `region`: `bytes` of every unit, on `numa_node` when
    /// the server creates it. Returns the part's units and its memfd, which the rank maps and
    /// registers with its device. A part is named by the rank's index in its worker group: the
    /// ranks of one index, in every group and replica, hold the same share of every unit and
    /// share the part.
    ///
    /// # Errors
    ///
    /// The server's refusal (no open session states the region, or the parts' bytes of a unit
    /// exceed the region's unit size) or the connection's failure.
    pub(crate) fn host_part(
        &self,
        region: &str,
        part: u32,
        bytes: u64,
        numa_node: Option<u32>,
    ) -> Result<(u32, OwnedFd)> {
        self.call("taking the rank's part of a host region", |client| {
            client.host_part(region.to_owned(), part, bytes, numa_node)
        })
    }
}

/// The buffer pool's statement of `runs`: each run of a field of the rank's units, with the GPU
/// that holds its storage as the location. What a rank places at its own server
/// ([`StorageClient::place`]) and a peer's rank has the scheduler node place for it
/// ([`StorageClient::place_remote`]).
pub(crate) fn unit_runs(runs: &[CachePlaneRun]) -> Vec<UnitRun> {
    runs.iter()
        .map(|run| UnitRun {
            field: run.field.clone(),
            key: run.key,
            location: Location::Device { gpu: run.gpu },
            first: run.first,
            count: run.count,
            offset: run.offset,
            bytes: run.bytes,
        })
        .collect()
}

/// The storages of a cache's planes are the memory of a buffer pool region: they are taken
/// with `Mode::Region`, which records no writer and shoots no other process down, so every
/// epoch that names the region maps them writable at once (`specs/buffer-pool.md` §14.1).
impl ExclusiveStorage for StorageClient {
    /// Maps the storages the server holds under the configs' keys, and creates and publishes
    /// the others.
    fn exclusive(&self, configs: &[ExclusiveConfig<'_>]) -> Result<Vec<(Arc<Storage>, bool)>> {
        let mut state = lock(&self.state);
        let state = &mut *state;

        let mut keyed = Vec::with_capacity(configs.len());
        for config in configs {
            let gpu = state.gpu(config.device)?;
            keyed.push((self.exclusive_key(config, &gpu)?, gpu));
        }
        let keys: Vec<Key> = keyed.iter().map(|(key, _)| *key).collect();

        let (got, descriptors) = state
            .client
            .get(keys, Mode::Region, Vec::new())
            .map_err(|error| storage_error("taking exclusive storages", error))?;
        if !got.missing.is_empty() {
            let taking = got.tensors.iter().map(|tensor| tensor.key).collect();
            self.release_untaken(state, &taking)?;
        }

        let mut storages: Vec<Option<(StorageRecord, OwnedFd)>> = got
            .storages
            .into_iter()
            .zip(descriptors)
            .map(Some)
            .collect();
        let mut found = BTreeMap::new();
        for tensor in &got.tensors {
            let storage = storages
                .get_mut(tensor.storage as usize)
                .and_then(Option::take)
                .ok_or_else(|| Error::Interface("a retained storage was named twice".into()))?;
            found.insert(tensor.key, storage);
        }

        let mut taken = Vec::with_capacity(configs.len());
        for (config, (key, gpu)) in configs.iter().zip(&keyed) {
            let (storage, created) = match found.remove(key) {
                Some((record, descriptor)) => {
                    let storage = match record.location {
                        Location::Host { .. } => Storage::map_host(
                            descriptor,
                            record.bytes as usize,
                            config.device,
                            true,
                        )?,
                        Location::Device { .. } => Storage::map_device(
                            &descriptor,
                            record.bytes as usize,
                            config.device,
                            config.access,
                            Protection::ReadWrite,
                        )?,
                    };
                    (storage, false)
                }

                None => {
                    let (storage, descriptor) =
                        Storage::retained(config.device, config.bytes, config.access)?;
                    let bytes = storage.bytes() as u64;
                    let record = StorageRecord {
                        location: Location::Device { gpu: *gpu },
                        segment: None,
                        bytes,
                        fabric: None,
                    };
                    let tensor =
                        byte_tensor(*key, config.name.to_owned(), 0, bytes, Access::Exclusive);

                    state
                        .client
                        .put(vec![tensor], &[(record, descriptor.as_fd())])
                        .map_err(|error| storage_error("publishing an exclusive storage", error))?;
                    (storage, true)
                }
            };

            state.taken.insert(*key);
            taken.push((storage, created));
        }

        tracing::debug!(
            mapped = configs.len() - got.missing.len(),
            created = got.missing.len(),
            "took exclusive storages"
        );
        Ok(taken)
    }

    fn key(&self, config: &ExclusiveConfig<'_>) -> Result<Option<(Key, [u8; 16])>> {
        let gpu = self.gpu(config.device)?;

        Ok(Some((self.exclusive_key(config, &gpu)?, gpu)))
    }

    /// Takes the parts of a segmented storage as [`Self::exclusive`] takes whole storages,
    /// except that a missing part whose `create` is unset (a segment given back) is left
    /// unmapped.
    fn segmented(&self, config: &SegmentedConfig<'_>) -> Result<(Arc<Storage>, Vec<bool>)> {
        let mut state = lock(&self.state);
        let state = &mut *state;

        let gpu = state.gpu(config.device)?;
        let keys: Vec<Key> = config
            .parts
            .iter()
            .map(|part| self.part_key(config.name, part, &gpu))
            .collect::<Result<_>>()?;

        let (got, descriptors) = state
            .client
            .get(keys.clone(), Mode::Region, Vec::new())
            .map_err(|error| storage_error("taking a segmented storage", error))?;

        // The storages a predecessor left that no owner takes again are released before a
        // missing part is created in their place. Parts left unmapped (segments given back)
        // create nothing, and the storages not taken yet include other planes' segments, which
        // their owner takes next: the value planes' after the key planes'.
        let creates = got.missing.iter().any(|missing| {
            keys.iter()
                .position(|key| key == missing)
                .is_some_and(|index| config.parts[index].create)
        });
        if creates {
            let taking = got.tensors.iter().map(|tensor| tensor.key).collect();
            self.release_untaken(state, &taking)?;
        }

        let mut storages: Vec<Option<(StorageRecord, OwnedFd)>> = got
            .storages
            .into_iter()
            .zip(descriptors)
            .map(Some)
            .collect();
        let mut found = BTreeMap::new();
        for tensor in &got.tensors {
            let (record, descriptor) = storages
                .get_mut(tensor.storage as usize)
                .and_then(Option::take)
                .ok_or_else(|| Error::Interface("a retained storage was named twice".into()))?;
            let index = keys
                .iter()
                .position(|key| *key == tensor.key)
                .ok_or_else(|| Error::Interface("an unasked segment was found".into()))?;

            if record.bytes as usize != config.parts[index].length {
                return Err(Error::Interface(
                    "a retained segment has another length".into(),
                ));
            }
            found.insert(tensor.key, descriptor);
        }

        let storage = Storage::segmented(config.device, &config.lengths(), config.access)?;
        let mut created = vec![false; keys.len()];
        for (index, key) in keys.iter().enumerate() {
            match found.remove(key) {
                Some(descriptor) => storage.import_segment(index, &descriptor)?,
                // A part nobody holds stays unmapped when it was given back.
                None if !config.parts[index].create => continue,
                None => {
                    let descriptor = storage.create_segment(index)?;
                    Self::publish_segment(state, config, index, *key, gpu, &descriptor)?;
                    created[index] = true;
                }
            }
            state.taken.insert(*key);
        }

        Ok((storage, created))
    }

    fn segment_key(
        &self,
        config: &SegmentedConfig<'_>,
        index: usize,
    ) -> Result<Option<(Key, [u8; 16])>> {
        let part = config
            .parts
            .get(index)
            .ok_or_else(|| Error::Invalid(format!("{} has no part {index}", config.name)))?;
        let gpu = self.gpu(config.device)?;

        Ok(Some((self.part_key(config.name, part, &gpu)?, gpu)))
    }

    /// Unmaps the parts `segments` and has the Storage Server drop them.
    fn release_segments(
        &self,
        config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(config.device)?;

        let mut given = BTreeSet::new();
        for &index in segments {
            storage.release_segment(index)?;

            let key = self.part_key(config.name, &config.parts[index], &gpu)?;
            state.taken.remove(&key);
            given.insert(key);
        }

        self.release_keys(&mut state, &given, "giving back segments")
    }

    /// Creates the parts `segments` again, as segments of the rank's own, and publishes them.
    fn restore_segments(
        &self,
        config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(config.device)?;

        for &index in segments {
            let key = self.part_key(config.name, &config.parts[index], &gpu)?;
            let descriptor = storage.create_segment(index)?;
            Self::publish_segment(&mut state, config, index, key, gpu, &descriptor)?;

            state.taken.insert(key);
        }
        Ok(())
    }
}

/// The record of the one-dimensional byte tensor of `bytes` bytes at byte `offset` of the
/// storage a `put` carries with it. The server never interprets a tensor's bytes, so every
/// storage this crate publishes is stated as bytes; its owner knows what they hold.
pub(super) fn byte_tensor(
    key: Key,
    name: String,
    offset: u64,
    bytes: u64,
    access: Access,
) -> TensorRecord {
    TensorRecord {
        key,
        name,
        storage: 0,
        offset,
        dtype: "uint8".into(),
        shape: vec![bytes],
        strides: vec![1],
        access,
    }
}

/// The name a part of a segmented storage is retained under.
fn segment_name(name: &str, part: &SegmentPart) -> String {
    format!("{name}#{}/{}", part.column, part.segment.ordinal)
}

/// A Storage Server failure while the rank was doing `what`.
fn storage_error(what: &str, error: storage::Error) -> Error {
    Error::transport_source(what, error)
}
