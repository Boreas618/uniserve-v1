//! The engine's one client of its host's Storage Server (`specs/storage.md` §3.2).
//!
//! The engine's main process is the head of its epoch on the host's Storage Server
//! ([`storage::Role::Head`]). Over that connection, [`StorageClient`], it opens the epoch's
//! session with the deployment's buffer pool (`Open`, `specs/buffer-pool.md` §11) and closes it
//! once its executors have stopped (`Close`), and it states what only the head knows: a segment
//! of a device region its ranks created ([`StorageClient::grant`]), one they are about to release
//! ([`StorageClient::withdraw`]), and where a rank of another host keeps its part of a region's
//! units ([`StorageClient::place_remote`], BP §15).
//!
//! The answer to `Open` carries one queue pair per scheduler: the engine's end of each is mapped
//! here and handed to its scheduler, whose submissions and completions travel in it with no system
//! call. The same queue pair holds the registers of its scheduler's groups: each group's
//! `completed` word, which the group's executors write through a mapping of their own
//! ([`StorageClient::completed_registers`]), and the capacity words the buffer pool writes.
//!
//! Each rank of the engine speaks to the same server on a connection of its own, as a worker
//! (`worker::memory::client::StorageClient`): over it the rank states where its part of a device
//! region's units lies ([`place`]) and takes its part of a host region ([`host_part`]), BP §14.2.

use std::os::fd::OwnedFd;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use storage::protocol::UnitRun;
use storage::runtime::{GroupId, Open, Opened, Segments};
use storage::{Client, Location, QueuePair, Role, Segment};

/// The engine's head connection to its host's Storage Server, and the session it opened with the
/// buffer pool (module documentation).
pub struct StorageClient {
    client: Client,
    /// The layout the session was opened with and the buffer pool's answer to it; `None` until
    /// the session is open.
    session: Option<(Open, Opened)>,
}

impl StorageClient {
    /// Connects to the Storage Server at `socket` as the head of `epoch` of `deployment`.
    ///
    /// # Errors
    ///
    /// When the server cannot be reached or refuses the head.
    pub fn connect(socket: &Path, deployment: &str, epoch: u64) -> Result<Self> {
        let role = Role::Head {
            deployment: deployment.to_owned(),
            epoch,
        };
        let client = Client::connect(socket, role).with_context(|| {
            format!(
                "connecting to the Storage Server at {} as the head of epoch {epoch}",
                socket.display()
            )
        })?;

        Ok(Self {
            client,
            session: None,
        })
    }

    /// Opens the epoch's session with the buffer pool of its deployment, stating `layout`.
    /// Returns the engine's end of each scheduler's queue pair, in the order of
    /// `layout.schedulers`; the layout and the answer are kept ([`Self::session`]).
    ///
    /// The segments each device region of `layout` states were created by the engine's ranks
    /// before this call, on every rank that maps the region (BP §14.2).
    ///
    /// # Errors
    ///
    /// A session open already; the server's refusal of the layout; an answer that does not state
    /// every region and scheduler of the layout, or a queue pair that cannot be mapped.
    pub fn open(&mut self, layout: Open) -> Result<Vec<QueuePair>> {
        if self.session.is_some() {
            bail!(
                "the session of epoch {} with the buffer pool is open already",
                layout.epoch
            );
        }

        let (opened, descriptors) = self
            .client
            .open_pool(layout.clone())
            .with_context(|| format!("opening epoch {}'s session", layout.epoch))?;
        if opened.regions.len() != layout.regions.len()
            || opened.segments.len() != layout.regions.len()
        {
            bail!(
                "a layout of {} regions was answered for {} regions and {} segment tables",
                layout.regions.len(),
                opened.regions.len(),
                opened.segments.len()
            );
        }

        // The engine's end of each queue pair: the scheduler writes its submission queue and
        // reads its completion queue.
        let pairs = descriptors
            .into_iter()
            .map(|descriptor| {
                QueuePair::open(descriptor, true).context("mapping a scheduler's queue pair")
            })
            .collect::<Result<Vec<_>>>()?;

        self.session = Some((layout, opened));
        Ok(pairs)
    }

    /// The layout the session was opened with and the buffer pool's answer to it; `None` before
    /// the session is open. The answer's segment table of each region is the one the Storage
    /// Server holds for the region's memory space: as `Open` answered it, with the segments this
    /// head granted since ([`Self::grant`]) and without those it withdrew ([`Self::withdraw`]).
    pub fn session(&self) -> Option<(&Open, &Opened)> {
        self.session
            .as_ref()
            .map(|(layout, opened)| (layout, opened))
    }

    /// Where the `completed` word of `group` lies: the engine's mapping of the queue pair of the
    /// scheduler that lists the group, mapped again from `pairs` (the queue pairs [`Self::open`]
    /// returned, in their order), and the word's index among its registers. What the group's
    /// executors write their ended sequence into (`worker::executor::Sequences::new`); the
    /// mapping is theirs for as long as they may write it.
    ///
    /// # Errors
    ///
    /// A session not open, a group no scheduler of the layout lists, or a queue pair that cannot
    /// be mapped again.
    pub fn completed_registers(
        &self,
        group: GroupId,
        pairs: &[QueuePair],
    ) -> Result<Vec<(QueuePair, usize)>> {
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

    /// Ends the session: the epoch's executors have stopped, and the buffer pool does what it
    /// does for an engine that is gone (BP §12.4).
    ///
    /// # Errors
    ///
    /// A session not open, or the server's refusal.
    pub fn close(&mut self) -> Result<()> {
        if self.session.is_none() {
            bail!("the session with the buffer pool is not open");
        }

        self.client
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
    pub fn grant(&mut self, region: &str, segment: Segment, first: u32, units: u32) -> Result<()> {
        self.client
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
    pub fn withdraw(&mut self, region: &str, segment: Segment) -> Result<()> {
        self.client
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

    /// States where rank `rank` of the device memory `memory`, a rank of the host whose Storage
    /// Server listens at `endpoint`, keeps its part of the units of the device region `region`:
    /// `runs`, as that rank states them ([`place`]).
    ///
    /// # Errors
    ///
    /// The server's refusal.
    pub fn place_remote(
        &mut self,
        region: &str,
        memory: &str,
        rank: u32,
        endpoint: &str,
        runs: &[ipc::CachePlaneRun],
    ) -> Result<()> {
        self.client
            .place_remote(
                region.to_owned(),
                memory.to_owned(),
                rank,
                endpoint.to_owned(),
                unit_runs(runs),
            )
            .with_context(|| {
                format!("placing rank {rank} of {endpoint} in region {region} of memory {memory}")
            })
    }
}

/// States, on the connection of the rank whose client is `client`, where its part of the units
/// of the device region `region` lies in the device memory `memory` its group addresses: `runs`,
/// each field of a unit as the rank's cache planes hold it (`KvCacheMemory::info`). A transfer
/// or a store that touches the rank's units is turned into copies with it.
///
/// # Errors
///
/// The server's refusal or the connection's failure.
pub fn place(
    client: &worker::memory::client::StorageClient,
    region: &str,
    memory: &str,
    runs: &[ipc::CachePlaneRun],
) -> Result<()> {
    let runs = unit_runs(runs);

    client
        .call("placing the rank's units", |client| {
            client.place(region.to_owned(), memory.to_owned(), runs)
        })
        .with_context(|| format!("placing the rank's units of region {region} in {memory}"))
}

/// Takes, on the connection of the rank whose client is `client`, the rank's part of the host
/// region `region`: `bytes` of every unit, on `numa_node` when the server creates it. Returns the
/// part's units and its memfd, which the rank maps and registers with its device; ranks of one
/// rank index share a part.
///
/// # Errors
///
/// The server's refusal (no open session states the region, or the parts' bytes of a unit
/// exceed the region's unit size) or the connection's failure.
pub fn host_part(
    client: &worker::memory::client::StorageClient,
    region: &str,
    bytes: u64,
    numa_node: Option<u32>,
) -> Result<(u32, OwnedFd)> {
    client
        .call("taking the rank's part of a host region", |client| {
            client.host_part(region.to_owned(), bytes, numa_node)
        })
        .with_context(|| format!("taking the rank's part of host region {region}"))
}

/// The buffer pool's statement of `runs`: each run of a field of the rank's units, with the GPU
/// that holds its storage as the location.
fn unit_runs(runs: &[ipc::CachePlaneRun]) -> Vec<UnitRun> {
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
