//! The memory a rank's media runners read and write: a request's media products, its encoder
//! outputs and its trajectory, as blocks of the buffer pool that the rank's device block tables
//! name.
//!
//! The buffer pool in the Storage Server decides which units of a region a block takes; the
//! rank's [`BlockTables`] keep, by slot, class and position, the runs of each block that lie in
//! the memory the rank's process maps. [`BufferMemory`] is the memory those runs address on one
//! rank, and lends it.
//!
//! There is one [`BufferMemory`] per rank, not per device: a device that holds ranks of several
//! worker groups (a text encoder's rank and a denoising rank, as a deployment that places two
//! groups on the same devices does) holds one memory per rank, each with its own product buffer,
//! its group's trajectory rows and its own rank index, taken from that rank's graph pool. The
//! ranks of a process are told apart by their rank index alone ([`BufferMemory::connect`]); two
//! of them on one device reach each other's memory as ranks on two devices do.
//!
//! What a rank's memory holds:
//!
//! - **The product buffer.** One region of [`UNIT_BYTES`]-byte units, held for as long as the
//!   rank serves. A media rank's is storage the Storage Server retains, its parts segments back
//!   to back (`StorageClient::region_storage`), which the rank places, so the Transfer Server
//!   copies a block into another process's product buffer for a crossing; a text rank's encoder
//!   outputs, which no other process reads, are taken from the device's resident pool
//!   (`GraphPool::allocator`). A media product has
//!   one contiguous run in the product buffer of every rank that backs it, and an encoder output
//!   one run; [`BufferMemory::lend`] lends the bytes of the block an entry of the tables names as
//!   a tensor, and [`BufferMemory::address`] states where a run's bytes lie for a copy. A media
//!   rank's buffer is mapped in one part per request slot, back to back, each the units one
//!   request's products take in whole allocation granules ([`BufferMemory::slot_units`]).
//! - **The trajectory rows.** The live state of a media request in its slot: one row per slot of
//!   the slot region, each the bytes of the largest trajectory the rank holds, starting on a
//!   [`UNIT_BYTES`] boundary, and on a device a part of whole granules of its own. Slot `s`
//!   (slots count from one) owns row `s - 1`, the run `(slot region, s, 1)` names it, and
//!   [`BufferMemory::trajectory`] lends from it. A trajectory is laid out from its row's first
//!   byte, so a smaller one is a prefix of the row.
//! - **The slots kept.** A media rank warms up holding the storages of the slots its division
//!   bounds, then keeps the slots whose storages fit beside its warm process and gives the parts
//!   of the rest back to the device ([`BufferMemory::give_back`]), the last first: the slot
//!   region and the product buffer then serve the slots kept, from slot 1 and unit 0.
//! - **The parts of host regions.** The host encoder region and the host latent region are the
//!   Storage Server's: one memfd per rank index that maps one
//!   (`storage::protocol::Request::HostPart`), unit `u` of a part at byte `u × bytes`, mapped by
//!   that rank writable and registered with its device ([`BufferMemory::map_part`]). A host
//!   block's run is the same units in every part, so its bytes in one part are one contiguous
//!   range of it.
//! - **The process's other ranks.** The ranks of one process reach each other's product buffers
//!   by peer access (the storage grants it to every device of the process that can take it; two
//!   ranks of one device address each other's memory directly) and each other's parts through their host
//!   mappings, which every device of the process may address ([`BufferMemory::connect`]). A rank
//!   lends a block from its own run, and from another rank's when it backs none of the block.
//!
//! ## A block on the parts of a host block
//!
//! A rank copies between its own memory and one part of a host block:
//!
//! - A block whose units exist on every rank of its region, a KV block, moves each rank's share
//!   of every unit to that rank's own part: the KV cache's host tier, one page per unit
//!   (`crate::memory::cache`).
//! - A block that one rank holds whole has one host copy on each host that wrote it, in part 0
//!   there. An encoder output in the rank's product buffer is one: the buffer pool sizes its host
//!   copy by the bytes one part holds of a unit (`storage::runtime::Region::part_bytes`), and
//!   the rank that holds it writes it. A trajectory is one too: every rank of a denoising group
//!   holds the whole samples in its own row, so the group's first rank in each process writes the
//!   host copy into its process's part and every rank of the process reads it back from there.
//!   The part is the one the rank copies with ([`BufferMemory::part_holder`]): its own when it
//!   maps one, otherwise the one part of the region another rank of its process maps. A media
//!   deployment maps part 0 of each of its host regions on the first rank of each process, so a
//!   host copy one group wrote on a host is where every group of every epoch that names the region
//!   reads it on that host; the buffer pool offers a group a host copy only when it lies on every
//!   host part the group copies with (`storage::runtime::Group::parts`), so a copy written on
//!   another host is a miss, never other bytes.
//!
//! ## Copies
//!
//! A `Copy` completion queues the copies a rank runs in its tables, and the worker hands each, with
//! the next batch, to a runner that maps its memory
//! ([`Runner::maps`](crate::runner::Runner::maps)). Every media runner of a rank shares one
//! [`BufferMemory`] and maps its product buffer and the parts of host regions it copies with
//! ([`BufferMemory::holds`]); the trajectory rows are the denoiser's alone
//! ([`BufferMemory::rows`]), so it issues every `Save` and `Restore`, a `Save` on its group's
//! first rank in each process alone. A runner issues its copies through [`BufferMemory::copy`],
//! and the denoiser a restore of a trajectory written in another encoding through
//! [`BufferMemory::restore`]:
//!
//! | From | To | Request | Bytes copied |
//! | --- | --- | --- | --- |
//! | product buffer | product buffer | `Clone` | the block's, both runs of one size |
//! | product buffer | the part the rank copies with | `Store` of an encoder output | the device block's, which the part's share of the host block holds |
//! | the part the rank copies with | product buffer | `Load` of an encoder output | the part's share of the host block, which the device block holds |
//! | a slot's row | the part the rank copies with | `Save` | the row's first bytes, as many as the part's share of the host block holds |
//! | the part the rank copies with | a slot's row | `Restore` of a trajectory saved in the class's encoding | as above |
//! | the saving ranks' parts | a slot's row | `Restore` of a trajectory saved in another encoding | what the copy manifest of the two layouts has the rank read ([`BufferMemory::restored`]) |
//!
//! The size of a sized block is the scheduler's (`specs/buffer-pool.md` §11): a trajectory block
//! holds the trajectory of the request's layout, which is the first bytes of its row. An encoder
//! output keeps one byte layout under every parallel configuration, its rows in order held whole
//! by one rank, so its class states one encoding everywhere and its copies never need a manifest.
//!
//! One more copy moves a product between the ranks of a process, outside the buffer pool's:
//! [`BufferMemory::share`], by which the rank that runs a call copies each output it wrote into
//! its own run to the runs of the backing ranks of its process that do not run the call.
//!
//! ## Ordering
//!
//! A copy is issued on the stream of the runner that serves the batch it starts with, before
//! that batch's forward, and the batch's end covers it. A block a batch writes is read by a later
//! batch, which the scheduler forms only once the writer's output arrived, so every read is
//! ordered after its write by completion, on whichever runner's stream either was issued, and a
//! shared output reaches the other backing ranks before the runner returns its output. Nothing
//! passes between two ranks' memories inside a call: the splice tail a decode round passes from
//! one rank to the next travels over the decoder ranks' communicator (`runner::decoder`).
//!
//! ## Without a device
//!
//! [`BufferMemory::meta`] holds no memory: its tensors are meta tensors, its addresses are the
//! byte offsets of each memory from 0, and its copies are checked and issue nothing. Its
//! [`BufferMemory::byte_copies`] are the copies a memory on a device issues.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use platform::{Allocator, Buffer, DeviceId, PoolKind, Stream};
use storage::protocol::UnitRun;
use storage::runtime::{RequestId, Run};
use tensor::{DType, Tensor};

use crate::memory::allocation::Storage;
use crate::memory::client::StorageClient;
use crate::memory::division;
use crate::memory::latent::restore::{self, Stored};
use crate::memory::lend;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::worker::{Error, Result};
use service::lock;

/// The bytes of one unit of a rank's product buffer: every product and encoder output starts on
/// a 256-byte boundary, and the region's unit size in `Open` is this.
pub(crate) const UNIT_BYTES: u64 = 256;

/// A byte of the media memory a rank reaches: the region whose memory holds it, by the number
/// the answer to `Open` gave it; the rank index of the rank whose memory it is (its own, or
/// another rank's of its process); and its offset from that memory's first byte (a product
/// buffer's, the trajectory rows' for the slot region, or a part's of a host region).
pub(crate) type Place = (u8, u32, u64);

/// One copy of bytes: where it reads, where it writes, and how many bytes.
pub(crate) type ByteCopy = (Place, Place, u64);

/// A product buffer the Storage Server retains (`StorageClient::region_storage`): its storage,
/// its parts segments back to back, and where the units of each part lie, a run per part.
pub(crate) type Retained = (Arc<Storage>, Vec<UnitRun>);

/// A part of a host region a rank reaches: the region, the rank index whose part it is, the
/// bytes of every unit of the part, its units, and its mapping (`None` for memory that holds no
/// device).
type Part = (u8, u32, u64, u32, Option<Arc<Storage>>);

/// A rank's media memory: see the module documentation.
pub(crate) struct BufferMemory {
    /// The rank's index: the index of its parts of host regions.
    rank: u32,
    /// The device the memory lies on, with the first byte of the product buffer and of the
    /// trajectory rows taken from its resident pool, each `None` when it holds no byte; `None`
    /// for memory that holds no device.
    device: Option<(DeviceId, Option<u64>, Option<u64>)>,
    /// The parts the product buffer and the trajectory rows are mapped in, back to back from
    /// their first bytes: the storage a device memory holds, which [`BufferMemory::give_back`]
    /// releases the tail of. Empty for memory that holds no device; no part of a product buffer
    /// the Storage Server retains.
    held: Mutex<(Vec<Buffer>, Vec<Buffer>)>,
    /// The product buffer's storage when the Storage Server retains it, a media rank's, whose
    /// units a crossing between processes reads and writes: its parts as segments back to back,
    /// with where the units of each part it holds lie, which the rank places, a run per part in
    /// order. `None` for a product buffer the resident pool holds, a text rank's encoder outputs,
    /// which no other process reads.
    retained: Option<(Arc<Storage>, Mutex<Vec<UnitRun>>)>,
    /// The product buffer's region and its units.
    products: (u8, u32),
    /// The parts the product buffer is made of: one, or one per slot, whose units are the
    /// buffer's over its slots and which a give-back releases with their slots.
    product_parts: u32,
    /// The slot region and its slots, numbered from 1.
    slots: (u8, u32),
    /// The slots, from slot 1, whose storages the memory holds: every slot until a give-back
    /// keeps fewer. A run of a slot or of product units past them is refused.
    kept: AtomicU32,
    /// The bytes of one slot's trajectory; 0 for a rank that holds none.
    trajectory: u64,
    /// The bytes from one row's first byte to the next's: the trajectory's, rounded up to a
    /// whole unit, and on a device to whole allocation granules, each row being a part of its
    /// own.
    row: u64,
    /// The parts of host regions the rank reaches: its own, and those of the other ranks of its
    /// process.
    parts: Vec<Part>,
    /// The product buffers of the process's other ranks, which the device reaches by peer
    /// access: the region, the rank index, its units, and its first byte's device address
    /// (`None` for memory that holds no device). Their memory lives as long as this rank's: the
    /// ranks of a process serve and retire together.
    peers: Vec<(u8, u32, u32, Option<u64>)>,
}

/// The bytes of `shape` elements of `dtype`.
fn tensor_bytes(shape: &[usize], dtype: DType) -> Result<u64> {
    let element = ops::abi::bytes(dtype).ok_or_else(|| {
        Error::Unsupported(format!(
            "{} has no whole-byte element to lend",
            ops::abi::name(dtype)
        ))
    })?;

    shape
        .iter()
        .try_fold(element, |bytes, &extent| {
            u64::try_from(extent)
                .ok()
                .and_then(|extent| bytes.checked_mul(extent))
        })
        .ok_or_else(|| Error::Invalid(format!("a tensor of {shape:?} elements overflows")))
}

/// The bytes from one trajectory row's first byte to the next's.
fn row_bytes(trajectory: u64) -> Option<u64> {
    trajectory.checked_next_multiple_of(UNIT_BYTES)
}

/// The bytes a `Store` or a `Load` for `request` moves between a device block of `block` bytes
/// and the rank's part of a host block, `host` bytes: a store moves the device block, which the
/// part holds, and a load the part, which the device block holds; otherwise the copy is refused.
fn stored(request: RequestId, (block, host): (u64, u64), load: bool) -> Result<u64> {
    let (moved, holder) = if load { (host, block) } else { (block, host) };
    if holder < moved {
        return Err(Error::Invalid(format!(
            "request {} copies an encoder output between a device block of {block} bytes and \
             the rank's part of its host block, which holds {host}",
            request.0
        )));
    }

    Ok(moved)
}

impl BufferMemory {
    /// The device bytes the media memory of a rank takes, in allocations of whole granules of
    /// `granule` bytes: a product buffer of `units` units and the rows of `slots` slots for
    /// trajectories of `trajectory` bytes, each row a part of its own. The rows, and a text rank's
    /// encoder outputs, are the device's resident pool's, which is reserved with them when the
    /// device's `GraphPool` opens; a media rank's product buffer is storage the Storage Server
    /// retains.
    pub(crate) fn device_bytes(units: u32, slots: u32, trajectory: u64, granule: u64) -> u64 {
        let rows = row_bytes(trajectory)
            .map(|row| match row {
                0 => 0,
                row => division::granules(row, granule.max(UNIT_BYTES)),
            })
            .and_then(|row| row.checked_mul(u64::from(slots)))
            .unwrap_or(u64::MAX);

        (u64::from(units) * UNIT_BYTES).saturating_add(rows)
    }

    /// The units of a product buffer a slot's part takes for `units` units of the products one
    /// request holds, on a device whose allocations are whole granules of `granule` bytes: whole
    /// granules of them, so the parts of a buffer split per slot lie back to back at the units
    /// the region numbers ([`BufferMemory::open`]). `granule` is [`UNIT_BYTES`] for memory that
    /// holds no device, whose parts are its units.
    pub(crate) fn slot_units(units: u64, granule: u64) -> u64 {
        let granule = granule.max(UNIT_BYTES);
        (units.max(1) * UNIT_BYTES).next_multiple_of(granule) / UNIT_BYTES
    }

    /// The media memory of rank index `rank`, holding no device: `products` is the product
    /// buffer's region and units, `slots` the slot region and its slots, `trajectory` the bytes
    /// of one slot's trajectory (0 for a rank that holds none), and `parts` the rank's part of
    /// each host region it addresses (the region, the bytes of every unit of the part, its
    /// units). Its tensors are meta tensors, and its copies are checked and issue nothing.
    ///
    /// # Errors
    ///
    /// Two memories stated under one region, a part without bytes, or a trajectory whose rows
    /// overflow.
    pub(crate) fn meta(
        rank: u32,
        products: (u8, u32),
        slots: (u8, u32),
        trajectory: u64,
        parts: &[(u8, u64, u32)],
    ) -> Result<Self> {
        if products.0 == slots.0 {
            return Err(Error::Invalid(format!(
                "region {} is stated as the product buffer and as the slot rows",
                slots.0
            )));
        }
        let row = row_bytes(trajectory)
            .filter(|row| row.checked_mul(u64::from(slots.1)).is_some())
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "{} trajectory rows of {trajectory} bytes overflow",
                    slots.1
                ))
            })?;

        let mut memory = Self {
            rank,
            device: None,
            held: Mutex::new((Vec::new(), Vec::new())),
            retained: None,
            products,
            product_parts: 1,
            slots,
            kept: AtomicU32::new(slots.1),
            trajectory,
            row,
            parts: Vec::new(),
            peers: Vec::new(),
        };
        for &(region, bytes, units) in parts {
            memory.check_part(region, rank, bytes)?;
            memory.parts.push((region, rank, bytes, units, None));
        }
        Ok(memory)
    }

    /// The media memory of rank index `rank` on the device of `allocator`, as
    /// [`BufferMemory::meta`] states it with no host part: the trajectory rows are taken from the
    /// resident pool, and the product buffer is `retained`, the storage the Storage Server
    /// retains with where the units of its parts lie (`StorageClient::region_storage`), or taken
    /// from the resident pool when it is `None`; both at addresses that stay fixed for as long as
    /// the memory lives. The host parts are mapped afterwards ([`BufferMemory::map_part`]).
    /// Their contents are unspecified: a block is written before a batch reads it, and a
    /// trajectory row by its request's latent preparation or by a `Restore`.
    ///
    /// The product buffer is mapped in `parts` parts back to back: one, or one per slot, each the
    /// buffer's units over its slots, which must be whole allocation granules
    /// ([`BufferMemory::slot_units`]) for the parts to lie at the units the region numbers. Each
    /// trajectory row is a part of its own, of whole granules. A memory of a part per slot gives
    /// the storages of the slots it does not keep back ([`BufferMemory::give_back`]).
    ///
    /// # Errors
    ///
    /// As [`BufferMemory::meta`]; a product buffer that does not split into `parts` parts of
    /// whole granules, or a retained one of another number of parts or too short; and when the
    /// resident pool cannot hold the memory.
    pub(crate) fn open(
        allocator: &mut Allocator,
        rank: u32,
        (products, retained): ((u8, u32), Option<Retained>),
        slots: (u8, u32),
        trajectory: u64,
        parts: u32,
    ) -> Result<Self> {
        let mut memory = Self::meta(rank, products, slots, trajectory, &[])?;
        let parts = parts.max(1);
        if !products.1.is_multiple_of(parts) {
            return Err(Error::Invalid(format!(
                "a product buffer of {} units does not split into {parts} parts",
                products.1
            )));
        }

        let mut take = |bytes: u64, count: u32| -> Result<Vec<Buffer>> {
            if bytes == 0 || count == 0 {
                return Ok(Vec::new());
            }
            let bytes = usize::try_from(bytes)
                .map_err(|_| Error::Resource("a rank's media memory exceeds the address space"))?;

            Ok(allocator.allocate_parts(PoolKind::Resident, bytes, count as usize)?)
        };
        let part = u64::from(products.1 / parts) * UNIT_BYTES;
        let buffer = match &retained {
            Some((storage, runs)) => {
                // Parts rounded past their units would leave a gap between the units of two
                // slots, and each part's run states its units.
                let whole = storage.bytes() as u64 == part * u64::from(parts)
                    || parts == 1 && storage.bytes() as u64 >= part;
                if runs.len() != parts as usize || !whole {
                    return Err(Error::Invalid(format!(
                        "a retained product buffer of {} bytes in {} parts does not lie as {} \
                         units in {parts} parts of {part} bytes",
                        storage.bytes(),
                        runs.len(),
                        products.1
                    )));
                }
                Vec::new()
            }
            None => take(part, parts)?,
        };
        // Parts rounded past their units would leave a gap between the units of two slots.
        if parts > 1 && buffer.iter().any(|held| held.len() as u64 != part) {
            return Err(Error::Invalid(format!(
                "a product buffer of {} units in {parts} parts of {part} bytes is not whole \
                 allocation granules of the device",
                products.1
            )));
        }
        let rows = take(memory.row, slots.1)?;
        if let Some(first) = rows.first() {
            memory.row = first.len() as u64;
        }

        let first = match &retained {
            Some((storage, _)) => Some(storage.address()),
            None => buffer.first().map(Buffer::address),
        };
        memory.device = Some((allocator.device(), first, rows.first().map(Buffer::address)));
        memory.held = Mutex::new((buffer, rows));
        memory.retained = retained.map(|(storage, runs)| (storage, Mutex::new(runs)));
        memory.product_parts = parts;
        Ok(memory)
    }

    /// Where the units of the parts of a product buffer the Storage Server retains lie, a run per
    /// part the memory holds: what the rank places for the buffer's region in its process's
    /// memory (`Place`), which a crossing reads and writes. Empty for a product buffer the
    /// resident pool holds.
    pub(crate) fn placed(&self) -> Vec<UnitRun> {
        self.retained
            .as_ref()
            .map(|(_, runs)| lock(runs).clone())
            .unwrap_or_default()
    }

    /// The device bytes one slot's storages take in this memory: its part of a product buffer
    /// split per slot and its trajectory row. Zero for a memory whose product buffer is one part
    /// and which holds no trajectory: its slots take nothing of it.
    pub(crate) fn slot_bytes(&self) -> u64 {
        let products = if self.product_parts > 1 {
            u64::from(self.products.1 / self.product_parts) * UNIT_BYTES
        } else {
            0
        };
        let rows = if self.trajectory > 0 { self.row } else { 0 };
        products + rows
    }

    /// The slots, from slot 1, whose storages this memory holds.
    pub(crate) fn held_slots(&self) -> u32 {
        self.kept.load(Ordering::Acquire)
    }

    /// Gives the storages of the slots past the first `kept` back: their trajectory rows to
    /// `allocator`, the device's allocator they were taken from, and of a product buffer split
    /// per slot their parts of it, to the resident pool or, for a buffer the Storage Server
    /// retains, to the server through `client`; the last first. The GPU's quota drops by them,
    /// and a run of those slots or of their product units is refused from here on. The rank's
    /// tables name none of them: the scheduler's regions are resized to the slots kept before
    /// they serve. Returns the bytes given back.
    ///
    /// # Errors
    ///
    /// Keeping no slot, a retained product buffer and no client, or the driver's or the server's
    /// refusal of a release, after which the memory holds what it did not release yet.
    pub(crate) fn give_back(
        &self,
        allocator: &mut Allocator,
        client: Option<&StorageClient>,
        kept: u32,
    ) -> Result<u64> {
        if kept >= self.held_slots() {
            return Ok(0);
        }
        if kept == 0 {
            return Err(Error::Invalid(
                "a rank's media memory keeps at least one slot".into(),
            ));
        }

        // Nothing addresses the slots past `kept` from here on.
        self.kept.store(kept, Ordering::Release);
        let mut held = lock(&self.held);
        let (buffer, rows) = &mut *held;
        // A product buffer of one part is the slots' together, and stays whole.
        let mut parts = vec![rows];
        if self.product_parts > 1 {
            parts.push(buffer);
        }

        let mut released = 0;
        for held in parts {
            let past = held.split_off((kept as usize).min(held.len()));
            for part in past.into_iter().rev() {
                released += part.len() as u64;
                allocator.release(part)?;
            }
        }

        // A retained buffer's parts past the slots kept, by their index in its storage.
        if let Some((storage, runs)) = self.retained.as_ref().filter(|_| self.product_parts > 1) {
            let client = client.ok_or(Error::Invalid(
                "a product buffer the Storage Server retains is given back through its client"
                    .into(),
            ))?;
            let mut runs = lock(runs);
            let held = runs.len();
            let past = runs.split_off((kept as usize).min(held));
            let indexed: Vec<(usize, &UnitRun)> = past
                .iter()
                .enumerate()
                .map(|(index, run)| (kept as usize + index, run))
                .rev()
                .collect();
            client.release_region_storage(storage, &indexed)?;
            released += storage.bytes() as u64 / u64::from(self.product_parts) * past.len() as u64;
        }
        Ok(released)
    }

    /// Maps the rank's part of the host region `region`, its number and its name, which holds
    /// `bytes` of every unit, on `numa_node` when the Storage Server creates it: writable and
    /// registered with the device, so device copies read and write it directly. Returns the
    /// part's units.
    ///
    /// # Errors
    ///
    /// Memory that holds no device; a region the memory holds already, or a part without bytes;
    /// the server's refusal; a mapping that fails.
    pub(crate) fn map_part(
        &mut self,
        client: &StorageClient,
        (region, name): (u8, &str),
        bytes: u64,
        numa_node: Option<u32>,
    ) -> Result<u32> {
        let Some((device, ..)) = &self.device else {
            return Err(Error::Invalid(
                "media memory that holds no device maps no host part: it states its parts".into(),
            ));
        };
        let device = *device;
        self.check_part(region, self.rank, bytes)?;

        // A region of blocks held whole by one rank has one part, the region's part 0, which
        // every epoch and replica on this host that names the region shares
        // (`specs/buffer-pool.md` §15).
        let (units, descriptor) = client.host_part(name, 0, bytes, numa_node)?;
        let length = u64::from(units)
            .checked_mul(bytes)
            .and_then(|length| usize::try_from(length).ok())
            .ok_or(Error::Resource(
                "a rank's part of a host region exceeds the address space",
            ))?;
        let storage = Storage::map_host(descriptor, length, device, true)?;
        tracing::info!(
            region = name,
            rank = self.rank,
            units,
            bytes,
            numa_node,
            "mapped the rank's part of a host region"
        );

        self.parts
            .push((region, self.rank, bytes, units, Some(storage)));
        Ok(units)
    }

    /// Joins the media memories of the ranks of one process, after each mapped its parts: each
    /// reaches every other's product buffer, by peer access or on its own device, and every
    /// other's parts of host regions. A memory joined before is joined again with the memories
    /// given.
    ///
    /// The memories are keyed by their rank index: several of them may lie on one device, the
    /// ranks of several worker groups that the deployment places on it, each with its own product
    /// buffer and rows.
    ///
    /// The caller keeps every memory joined, and the device memory behind it, for as long as any
    /// of them may address another's: the ranks of a process serve and retire together.
    ///
    /// # Errors
    ///
    /// Memories of one rank index or of one product buffer; memories on a device beside memories
    /// that hold none; two parts of one region of other bytes of a unit. Nothing changes when
    /// they are refused.
    pub(crate) fn connect(memories: &mut [BufferMemory]) -> Result<()> {
        for (index, memory) in memories.iter().enumerate() {
            for other in &memories[index + 1..] {
                if memory.rank == other.rank
                    || memory.products.0 == other.products.0
                    || memory.device.is_some() != other.device.is_some()
                {
                    return Err(Error::Invalid(format!(
                        "the media memories of ranks {} and {} share a rank index or a product \
                         buffer, or one holds a device and the other none",
                        memory.rank, other.rank
                    )));
                }
            }
        }

        // Each rank's own parts and product buffer, as the others reach them.
        let mut owned = Vec::new();
        let mut buffers = Vec::new();
        for memory in memories.iter() {
            owned.extend(
                memory
                    .parts
                    .iter()
                    .filter(|part| part.1 == memory.rank)
                    .cloned(),
            );
            let address = memory
                .device
                .as_ref()
                .map(|(_, buffer, _)| buffer.unwrap_or(0));
            buffers.push((memory.products.0, memory.rank, memory.products.1, address));
        }
        for memory in memories.iter() {
            for (region, rank, bytes, ..) in &owned {
                let unlike = memory
                    .parts
                    .iter()
                    .any(|part| part.0 == *region && part.2 != *bytes);
                if unlike {
                    return Err(Error::Invalid(format!(
                        "rank {rank}'s part of host region {region} holds {bytes} bytes of a \
                         unit, and rank {}'s another count",
                        memory.rank
                    )));
                }
            }
        }

        for memory in memories.iter_mut() {
            for part in &owned {
                let held = memory
                    .parts
                    .iter()
                    .any(|own| own.0 == part.0 && own.1 == part.1);
                if !held {
                    memory.parts.push(part.clone());
                }
            }
            for buffer in &buffers {
                if buffer.1 != memory.rank && !memory.peers.contains(buffer) {
                    memory.peers.push(*buffer);
                }
            }
        }
        Ok(())
    }

    /// The device the memory lies on; `None` for memory that holds none.
    pub(crate) fn device(&self) -> Option<DeviceId> {
        self.device.as_ref().map(|(device, ..)| *device)
    }

    /// The rank's index, which its parts of host regions are created under.
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// Whether `region` is memory of this rank's media that holds blocks: its product buffer, or
    /// a host region with a part it copies with ([`BufferMemory::part_holder`]). Every media
    /// runner of the rank maps it; the trajectory rows are the denoiser's alone
    /// ([`BufferMemory::rows`]). A runner that is handed the copies of other memory too, such as
    /// the KV cache's, issues through this memory those that name it.
    pub(crate) fn holds(&self, region: u8) -> bool {
        region == self.products.0 || self.copied(region).is_some()
    }

    /// The rank index whose part of host region `region` this rank copies a block it holds whole
    /// to and from (module documentation): its own when it maps one, otherwise the one part of
    /// the region another rank of its process maps. `None` when it reaches no part of the
    /// region, or several and none of its own.
    pub(crate) fn part_holder(&self, region: u8) -> Option<u32> {
        self.copied(region).map(|(_, rank, ..)| *rank)
    }

    /// Whether `region` is the slot region and this memory holds trajectories in its rows: the
    /// memory a denoiser reads and writes, and whose `Save` and `Restore` it alone lays out
    /// across encodings.
    pub(crate) fn rows(&self, region: u8) -> bool {
        region == self.slots.0 && self.trajectory > 0
    }

    /// Lends the bytes of the block that `slot` holds at `position` of `class` in the rank's
    /// tables, from byte `offset` of its run in a product buffer, as a tensor of `shape`
    /// elements of `dtype`: on the device, or a meta tensor for memory that holds none. The run
    /// is the rank's own, or for a block this rank does not back the run of another rank of its
    /// process, which the device reaches by peer access. `writes` states that a batch writes the
    /// tensor.
    ///
    /// The tensor addresses a product buffer, which lives as long as this memory: its holder
    /// keeps the memory while it uses the tensor.
    ///
    /// # Errors
    ///
    /// A position the tables hold no entry for; a write to an entry they do not state writable;
    /// a block with no run in a product buffer the rank reaches; elements that reach past the
    /// block.
    pub(crate) fn lend(
        &self,
        tables: &BlockTables,
        (slot, class, position): (u32, u8, u32),
        writes: bool,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        let runs = tables.runs(slot, class, position);
        if runs.is_empty() {
            return Err(Error::Invalid(format!(
                "slot {slot} holds no block at position {position} of class {class} on this rank"
            )));
        }
        if writes && !tables.writable(slot, class, position) {
            return Err(Error::Invalid(format!(
                "a batch writes the block of slot {slot} at position {position} of class \
                 {class}, which is read-only"
            )));
        }

        let own = runs.iter().find(|run| run.0 == self.products.0);
        let reached = || runs.iter().find(|run| self.peer(run.0).is_some());
        let Some(&run) = own.or_else(reached) else {
            return Err(Error::Invalid(format!(
                "the block of slot {slot} at position {position} of class {class} lies in no \
                 product buffer the rank reaches"
            )));
        };

        let place = self.window(run, offset, tensor_bytes(shape, dtype)?)?;
        self.tensor(place, shape, dtype)
    }

    /// Lends `slot`'s trajectory row from byte `offset` as a tensor of `shape` elements of
    /// `dtype`: on the device, or a meta tensor for memory that holds none. A row holds the
    /// trajectory of its slot's request, laid out from its first byte.
    ///
    /// # Errors
    ///
    /// A slot outside the slot region, a rank that holds no trajectory, or elements that reach
    /// past the trajectory's bytes.
    pub(crate) fn trajectory(
        &self,
        slot: u32,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        let place = self.window((self.slots.0, slot, 1), offset, tensor_bytes(shape, dtype)?)?;
        self.tensor(place, shape, dtype)
    }

    /// The address of `bytes` bytes from byte `offset` of `run` in the memory this rank reaches:
    /// a device address in a product buffer or a slot's row, a host address in the rank's part of
    /// a host region, or for memory that holds no device the offset of the bytes from its
    /// memory's first byte.
    ///
    /// # Errors
    ///
    /// A run of a region that is no memory of this rank's media, a run past its memory's end, or
    /// bytes that reach past the run.
    pub(crate) fn address(&self, run: Run, offset: u64, bytes: u64) -> Result<u64> {
        self.at(self.window(run, offset, bytes)?)
    }

    /// The bytes `copies` move in this rank's memory, one copy per pair of runs, as
    /// [`BufferMemory::copy`] issues them: the copies of a `Clone`, a `Store` or a `Load` of an
    /// encoder output, a `Save` or a `Restore` of a trajectory (module documentation).
    ///
    /// # Errors
    ///
    /// A run that is no memory of this rank's media or lies past its memory's end; a clone
    /// between runs of different sizes; a host block whose part on this rank cannot hold the
    /// device block it stores, or a device block that cannot hold the part it loads; any other
    /// pair of memories; a copy of a block written in another encoding, which only
    /// [`BufferMemory::restore`] lays out.
    pub(crate) fn byte_copies(&self, copies: &[BlockCopy]) -> Result<Vec<ByteCopy>> {
        let mut moved = Vec::new();
        for copy in copies {
            if copy.encodings.is_some() {
                return Err(Error::Invalid(format!(
                    "request {} copies a block written in another encoding, which a copy of \
                     bytes as they are does not lay out",
                    copy.request.0
                )));
            }
            for &(from, to) in &copy.runs {
                moved.push(self.byte_copy(copy.request, from, to)?);
            }
        }

        Ok(moved)
    }

    /// Issues `copies` on `stream`, ahead of the batch whose end covers them, for memory on a
    /// device, and with `None` for memory that holds none, whose copies are checked and issue
    /// nothing. Every copy is checked before one is issued.
    ///
    /// # Errors
    ///
    /// As [`BufferMemory::byte_copies`]; a stream stated for memory that holds no device, or the
    /// reverse; the driver's refusal, after which the copies issued before it may still run.
    pub(crate) fn copy(&self, copies: &[BlockCopy], stream: Option<&Stream>) -> Result<()> {
        let moved = self.byte_copies(copies)?;
        self.issue(&moved, stream)
    }

    /// The bytes the restore `copy` moves into a slot's row from a trajectory saved in another
    /// encoding: the copies of the copy manifest from `sources`, each saving rank's part by its
    /// rank index with how its pages hold its share of the samples, to `target`, how this rank's
    /// row holds its own (`crate::memory::latent::restore`). A saved page this rank lays out as
    /// a saving rank did is copied whole; every other region of its samples from the saved page
    /// that holds it. The layouts are the request's, which the runner that restores it computes
    /// from the two encodings the copy names.
    ///
    /// # Errors
    ///
    /// A copy that is not one restore of a block written in another encoding; layouts the
    /// manifest refuses, or saving ranks that do not hold every value the rank holds; a saving
    /// rank whose part this rank does not reach, or whose pages are not its part's units; a copy
    /// that reaches past the host block, a part or the trajectory.
    pub(crate) fn restored(
        &self,
        copy: &BlockCopy,
        sources: &[(u32, Stored)],
        target: &Stored,
    ) -> Result<Vec<ByteCopy>> {
        let ([(host, row)], Some(_)) = (copy.runs.as_slice(), copy.encodings) else {
            return Err(Error::Invalid(format!(
                "request {} is no restore of a trajectory written in another encoding",
                copy.request.0
            )));
        };
        if row.0 != self.slots.0 {
            return Err(Error::Invalid(format!(
                "request {} restores into region {}, which holds no trajectory row",
                copy.request.0, row.0
            )));
        }
        let (written, length) = self.extent(*row)?;

        // Each saving rank's pages are its part's units of the host block, back to back.
        for (rank, stored) in sources {
            let part = self.part(host.0, *rank).ok_or_else(|| {
                Error::Invalid(format!(
                    "rank {rank}'s part of host region {}, which the restore reads, is not \
                     reached by rank {}",
                    host.0, self.rank
                ))
            })?;
            if part.2 != stored.page_bytes {
                return Err(Error::Invalid(format!(
                    "rank {rank} states pages of {} bytes and its part units of {}",
                    stored.page_bytes, part.2
                )));
            }
        }
        let layouts: Vec<Stored> = sources.iter().map(|(_, stored)| stored.clone()).collect();

        let mut moved = Vec::new();
        for page in restore::copies(&layouts, target)? {
            let (rank, stored) = &sources[page.source];
            let (first, bytes) = self.part_extent(*host, *rank)?;
            let from = page
                .page
                .checked_mul(stored.page_bytes)
                .and_then(|start| start.checked_add(page.offset));
            let to = page.target.checked_add(page.bytes);

            match (from, to) {
                (Some(from), Some(to)) if from + page.bytes <= bytes && to <= length => {
                    let read = (host.0, *rank, first + from);
                    let wrote = (row.0, self.rank, written + page.target);
                    moved.push((read, wrote, page.bytes));
                }
                _ => {
                    return Err(Error::Invalid(format!(
                        "request {} copies {} bytes from byte {} of page {} of rank {rank}'s \
                         pages to byte {} of a row of {length}, past the host block or the row",
                        copy.request.0, page.bytes, page.offset, page.page, page.target
                    )));
                }
            }
        }
        Ok(moved)
    }

    /// Issues the restore `copy` of a trajectory saved in another encoding on `stream`, ahead of
    /// the batch whose end covers it, as [`BufferMemory::restored`] states its copies, and with
    /// `None` for memory that holds none, whose copies are checked and issue nothing.
    ///
    /// # Errors
    ///
    /// As [`BufferMemory::restored`] and [`BufferMemory::copy`].
    pub(crate) fn restore(
        &self,
        copy: &BlockCopy,
        (sources, target): (&[(u32, Stored)], &Stored),
        stream: Option<&Stream>,
    ) -> Result<()> {
        let moved = self.restored(copy, sources, target)?;
        self.issue(&moved, stream)
    }

    /// Copies the block that `slot` holds at `position` of `class`, which this rank wrote into
    /// its own run, into the block's runs on the other ranks of its process, whole, on `stream`:
    /// the backing ranks of a product that do not run the call that writes it. Returns the
    /// copies; with `None` for memory that holds no device they are checked and issue nothing.
    ///
    /// # Errors
    ///
    /// A block without a run in this rank's product buffer, a run of another size, or as
    /// [`BufferMemory::copy`].
    pub(crate) fn share(
        &self,
        tables: &BlockTables,
        (slot, class, position): (u32, u8, u32),
        stream: Option<&Stream>,
    ) -> Result<Vec<ByteCopy>> {
        let runs = tables.runs(slot, class, position);
        let Some(&own) = runs.iter().find(|run| run.0 == self.products.0) else {
            return Err(Error::Invalid(format!(
                "the block of slot {slot} at position {position} of class {class} has no run in \
                 this rank's product buffer to share"
            )));
        };
        let (from, bytes) = self.extent(own)?;

        let mut moved = Vec::new();
        for &run in runs.iter().filter(|run| self.peer(run.0).is_some()) {
            let (to, length) = self.extent(run)?;
            if length != bytes {
                return Err(Error::Invalid(format!(
                    "a product of {bytes} bytes on this rank has a run of {length} on rank {}",
                    self.rank_of(run.0)
                )));
            }
            let read = (own.0, self.rank, from);
            moved.push((read, (run.0, self.rank_of(run.0), to), bytes));
        }

        self.issue(&moved, stream)?;
        Ok(moved)
    }

    /// The run among `runs`, the runs of one block, that lies in the product buffer of the rank
    /// of this process whose index is `rank`: this rank's own, or another's it reaches. `None`
    /// when none of them does.
    pub(crate) fn run_on(&self, runs: &[Run], rank: u32) -> Option<Run> {
        runs.iter().copied().find(|run| {
            let own = run.0 == self.products.0 && rank == self.rank;
            own || self.peer(run.0).is_some_and(|peer| peer.1 == rank)
        })
    }

    /// Whether rank index `rank` is a media rank of this memory's process: this rank, or one
    /// whose product buffer it reaches ([`BufferMemory::connect`]). A rank of another process
    /// holds runs this memory cannot address.
    pub(crate) fn reaches(&self, rank: u32) -> bool {
        rank == self.rank || self.peers.iter().any(|peer| peer.1 == rank)
    }

    /// Issues `moved` on `stream`: see [`BufferMemory::copy`].
    fn issue(&self, moved: &[ByteCopy], stream: Option<&Stream>) -> Result<()> {
        let stream = match (&self.device, stream) {
            (None, None) => return Ok(()),
            (Some(_), Some(stream)) => stream,
            _ => {
                return Err(Error::Invalid(
                    "a rank's media copies are issued on a stream exactly when its memory is on \
                     a device"
                        .into(),
                ));
            }
        };
        let addresses = moved
            .iter()
            .map(|&(from, to, bytes)| Ok((self.at(from)?, self.at(to)?, bytes)))
            .collect::<Result<Vec<_>>>()?;

        for (source, destination, bytes) in addresses {
            if bytes == 0 {
                continue;
            }
            let bytes = usize::try_from(bytes)
                .map_err(|_| Error::Resource("a copy exceeds the address space"))?;

            // SAFETY: both ranges lie inside memory this rank reaches for as long as it lives
            // (`Self::window`, `Self::connect`). The buffer pool holds the block a copy reads
            // and the one it writes until the copy's `Finish` is taken, which follows the end of
            // the batch this stream runs next; no batch addresses the destination until then,
            // since no table entry names it or, for a slot's row, the scheduler runs no step of
            // the slot meanwhile. A shared product's other runs are read only by later batches,
            // which the scheduler forms once this rank's output arrived.
            unsafe { platform::memcpy_async(destination, source, bytes, stream) }?;
        }
        Ok(())
    }

    /// The bytes a copy from `from` to `to`, for `request`, moves: see the module
    /// documentation.
    fn byte_copy(&self, request: RequestId, from: Run, to: Run) -> Result<ByteCopy> {
        // `extent` stated every run's region: the product buffer, the slot rows, or the part of
        // a host region the rank copies with.
        let (read, written) = (self.extent(from)?, self.extent(to)?);
        let (products, slots) = (self.products.0, self.slots.0);
        let part = |region: u8| self.copied(region).is_some();
        let own = |region: u8| region == products || region == slots || part(region);
        if !own(from.0) || !own(to.0) {
            return Err(Error::Invalid(format!(
                "request {} copies region {} into region {}, and the memory this rank copies \
                 with holds one of them only",
                request.0, from.0, to.0
            )));
        }

        let bytes = match (from.0, to.0) {
            (source, destination) if source == products && destination == products => {
                if read.1 != written.1 {
                    return Err(Error::Invalid(format!(
                        "request {} clones a block of {} bytes into one of {}",
                        request.0, read.1, written.1
                    )));
                }
                read.1
            }

            // An encoder output between the product buffer and the rank's part: the host block
            // was sized for the device block's bytes on one part, and a loaded device block for
            // a part's bytes.
            (source, destination) if source == products && part(destination) => {
                stored(request, (read.1, written.1), false)?
            }
            (source, destination) if part(source) && destination == products => {
                stored(request, (written.1, read.1), true)?
            }

            // A trajectory between a slot's row and the rank's part: the block holds the
            // request's trajectory, the first bytes of the row.
            (source, destination)
                if (source == slots && part(destination))
                    || (part(source) && destination == slots) =>
            {
                read.1.min(written.1)
            }

            (source, destination) => {
                return Err(Error::Invalid(format!(
                    "request {} copies region {source} into region {destination}, which is no \
                     copy of a rank's media memory",
                    request.0
                )));
            }
        };

        let place = |region: u8, offset: u64| (region, self.rank_of(region), offset);
        Ok((place(from.0, read.0), place(to.0, written.0), bytes))
    }

    /// The first byte and the bytes of `run` in the memory it lies in, from that memory's first
    /// byte: this rank's product buffer or slot rows, the part it copies with, or another rank's
    /// product buffer.
    fn extent(&self, (region, first, count): Run) -> Result<(u64, u64)> {
        if region == self.slots.0 {
            // A slot's run is its row: slot `s` owns row `s - 1`.
            let held = self.held_slots();
            if self.trajectory == 0 || count != 1 || first == 0 || first > held {
                return Err(Error::Invalid(format!(
                    "the run of {count} slots from slot {first} names no trajectory row of a \
                     rank that holds {held} rows of {} bytes",
                    self.trajectory
                )));
            }
            return Ok((u64::from(first - 1) * self.row, self.trajectory));
        }

        let (unit, units) = if region == self.products.0 {
            (UNIT_BYTES, self.held_units())
        } else if let Some(&(_, _, units, _)) = self.peer(region) {
            (UNIT_BYTES, units)
        } else if let Some((_, _, bytes, units, _)) = self.copied(region) {
            (*bytes, *units)
        } else {
            return Err(Error::Invalid(format!(
                "region {region} is no memory of this rank's media"
            )));
        };
        Self::units((region, first, count), unit, units)
    }

    /// The units of the product buffer the memory holds: every unit, or for a buffer of a part
    /// per slot the units of the slots it keeps.
    fn held_units(&self) -> u32 {
        if self.product_parts > 1 {
            self.products.1 / self.product_parts * self.held_slots()
        } else {
            self.products.1
        }
    }

    /// The first byte and the bytes of `run` in rank `rank`'s part of its host region.
    fn part_extent(&self, run: Run, rank: u32) -> Result<(u64, u64)> {
        let (_, _, bytes, units, _) = self.part(run.0, rank).ok_or_else(|| {
            Error::Invalid(format!(
                "rank {rank}'s part of region {} is not reached by rank {}",
                run.0, self.rank
            ))
        })?;
        Self::units(run, *bytes, *units)
    }

    /// The first byte and the bytes of `run` in a memory of `units` units of `unit` bytes.
    fn units((region, first, count): Run, unit: u64, units: u32) -> Result<(u64, u64)> {
        let end = first.checked_add(count);
        if count == 0 || end.is_none_or(|end| end > units) {
            return Err(Error::Invalid(format!(
                "the run of {count} units from unit {first} of region {region} lies outside its \
                 {units} units"
            )));
        }
        Ok((u64::from(first) * unit, u64::from(count) * unit))
    }

    /// The place of `bytes` bytes from byte `offset` of `run`.
    fn window(&self, run: Run, offset: u64, bytes: u64) -> Result<Place> {
        let (first, length) = self.extent(run)?;

        if offset.checked_add(bytes).is_none_or(|end| end > length) {
            return Err(Error::Invalid(format!(
                "{bytes} bytes from byte {offset} reach past the {length} bytes of the run of {} \
                 units from unit {} of region {}",
                run.2, run.1, run.0
            )));
        }
        Ok((run.0, self.rank_of(run.0), first + offset))
    }

    /// The rank index whose memory region `region` is, among those `extent` addresses: another
    /// rank's product buffer, the part this rank copies with, or this rank's own memory.
    fn rank_of(&self, region: u8) -> u32 {
        match self.peer(region) {
            Some((_, rank, ..)) => *rank,
            None => self.part_holder(region).unwrap_or(self.rank),
        }
    }

    /// The address of `place`: see [`BufferMemory::address`].
    fn at(&self, (region, rank, offset): Place) -> Result<u64> {
        let Some((_, buffer, rows)) = &self.device else {
            return Ok(offset);
        };

        let base = if region == self.products.0 && rank == self.rank {
            *buffer
        } else if region == self.slots.0 && rank == self.rank {
            *rows
        } else if let Some((_, _, _, address)) = self.peer(region).filter(|peer| peer.1 == rank) {
            *address
        } else {
            self.part(region, rank)
                .and_then(|(.., storage)| storage.as_ref())
                .map(|storage| storage.address())
        };

        base.map(|base| base + offset).ok_or_else(|| {
            Error::Invalid(format!(
                "region {region} of rank {rank} has no memory this rank's device reaches"
            ))
        })
    }

    /// The tensor over `place`, a place in a product buffer or the trajectory rows.
    fn tensor(&self, place: Place, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let address = self.at(place)?;

        // SAFETY: the window lies inside a product buffer or the trajectory rows
        // (`Self::window`), which this memory reaches for as long as it lives and its holder
        // keeps it while the tensor's work may read it. Between batches only the copies this
        // memory issues write them, each into a block no batch addresses until the copy ended; a
        // meta tensor's address is never read.
        unsafe { lend(address, shape, dtype, self.device()) }
    }

    /// Rank `rank`'s part of host region `region`, among those the rank reaches.
    fn part(&self, region: u8, rank: u32) -> Option<&Part> {
        self.parts
            .iter()
            .find(|(held, of, ..)| *held == region && *of == rank)
    }

    /// The part of host region `region` this rank copies with: see
    /// [`BufferMemory::part_holder`].
    fn copied(&self, region: u8) -> Option<&Part> {
        if let Some(own) = self.part(region, self.rank) {
            return Some(own);
        }

        let mut reached = self.parts.iter().filter(|(held, ..)| *held == region);
        match (reached.next(), reached.next()) {
            (Some(only), None) => Some(only),
            _ => None,
        }
    }

    /// The product buffer of another rank of the process in `region`, if the rank reaches one.
    fn peer(&self, region: u8) -> Option<&(u8, u32, u32, Option<u64>)> {
        self.peers.iter().find(|(held, ..)| *held == region)
    }

    /// Whether rank `rank`'s part of `region`, of `bytes` of every unit, can be added to this
    /// memory.
    fn check_part(&self, region: u8, rank: u32, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Err(Error::Invalid(format!(
                "the rank's part of host region {region} holds no byte of a unit"
            )));
        }
        if region == self.products.0
            || region == self.slots.0
            || self.part(region, rank).is_some()
            || self.peer(region).is_some()
        {
            return Err(Error::Invalid(format!(
                "region {region} is stated twice in a rank's media memory"
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use storage::Key;
    use storage::runtime::{Completion, GroupId, OwnerId, Request};

    use super::*;
    use crate::memory::latent::restore::tests::{holds_its_samples, pages, share};

    const GROUP: GroupId = GroupId(0);

    /// The slot rows, the product buffers of two ranks, and the host encoder and latent regions.
    const SLOTS: u8 = 0;
    const PRODUCTS: u8 = 2;
    const OTHER_PRODUCTS: u8 = 3;
    const ENCODERS: u8 = 5;
    const LATENTS: u8 = 6;

    /// The rank indices of the two ranks of one process.
    const RANK: u32 = 0;
    const OTHER: u32 = 1;

    /// The class of media products.
    const PRODUCT: u8 = 4;

    /// The bytes of a unit of each rank's part of the host latent region.
    const PAGE: u64 = 256;

    /// A slot's trajectory: rows of 1,024 bytes, the trajectory rounded up to whole units.
    const TRAJECTORY: u64 = 1000;
    const ROW: u64 = 1024;

    /// The unit sizes of the regions a copy may name: a slot row, a product
    /// buffer's 256-byte unit, a host page.
    const UNIT_SIZES: [(u8, u64); 5] = [
        (SLOTS, ROW),
        (PRODUCTS, UNIT_BYTES),
        (OTHER_PRODUCTS, UNIT_BYTES),
        (ENCODERS, PAGE),
        (LATENTS, PAGE),
    ];

    fn attach(tables: &mut BlockTables, owner: u64, slot: u32) {
        let request = Request::Attach {
            owner: OwnerId(owner),
            group: GROUP,
        };

        tables
            .map(slot, RequestId(0), &request, &Completion::Attached { slot })
            .unwrap();
    }

    /// Hands `tables` the `Copy` that answers `request`, one key from `from` to `to` in
    /// `encodings`, and starts it as the next batch would.
    fn started(
        tables: &mut BlockTables,
        (id, request): (u64, &Request),
        (from, to): (Run, Run),
        encodings: Option<([u8; 32], [u8; 32])>,
    ) -> BlockCopy {
        let copy = Completion::Copy {
            group: GROUP,
            keys: vec![Key([7; 32])],
            from: vec![from],
            to: vec![to],
            evicted: vec![],
            encodings,
            parts: Vec::new(),
        };
        tables.map(0, RequestId(id), request, &copy).unwrap();

        tables.start_copy().unwrap()
    }

    /// Hands `tables` the allocation of one product, `units` units at `runs` on the ranks that
    /// back it, at position `at` of the owner in `slot`.
    fn product(tables: &mut BlockTables, slot: u32, at: u32, runs: &[Run]) {
        let allocate = Request::Allocate {
            owner: OwnerId(1),
            class: PRODUCT,
            at,
            count: 1,
            units: runs[0].2,
        };
        let allocated = Completion::Allocated {
            runs: runs.to_vec(),
            evicted: vec![],
        };

        tables
            .map(slot, RequestId(1 + u64::from(at)), &allocate, &allocated)
            .unwrap();
    }

    /// The memories of two ranks of one process on the meta device, each with its product buffer
    /// of 16 units, three slot rows and its part of the host latent region, joined.
    fn process(trajectory: u64) -> Vec<BufferMemory> {
        let memory = |rank, products| {
            let parts = [(LATENTS, PAGE, 32)];
            BufferMemory::meta(rank, (products, 16), (SLOTS, 3), trajectory, &parts).unwrap()
        };
        let mut memories = vec![memory(RANK, PRODUCTS), memory(OTHER, OTHER_PRODUCTS)];
        BufferMemory::connect(&mut memories).unwrap();
        memories
    }

    /// Performs `copies` on host mirrors of the ranks' memories, by region and rank index.
    fn apply(mirrors: &mut HashMap<(u8, u32), Vec<u8>>, copies: &[ByteCopy]) {
        for &((from, reader, source), (to, writer, destination), bytes) in copies {
            let (source, destination, bytes) =
                (source as usize, destination as usize, bytes as usize);

            let read = mirrors[&(from, reader)][source..source + bytes].to_vec();
            let written = mirrors.get_mut(&(to, writer)).unwrap();
            written[destination..destination + bytes].copy_from_slice(&read);
        }
    }

    /// The address a lent tensor's first element lies at.
    fn address_of(tensor: &Tensor) -> u64 {
        ops::operand(tensor).unwrap().ptr.address()
    }

    #[test]
    fn a_rank_lends_a_product_from_its_own_run_among_the_runs_of_its_process() {
        // Two products of three units each, backed by two ranks of one process: the runs are
        // block-major, one per backing rank, and each rank's tables keep both.
        let runs = [
            [(PRODUCTS, 0, 3), (OTHER_PRODUCTS, 8, 3)],
            [(PRODUCTS, 3, 3), (OTHER_PRODUCTS, 11, 3)],
        ];
        let memories = process(0);

        for (memory, (own, other), second) in [
            (&memories[0], (PRODUCTS, OTHER_PRODUCTS), 3u32),
            (&memories[1], (OTHER_PRODUCTS, PRODUCTS), 11),
        ] {
            let mut tables =
                BlockTables::new(GROUP, &[SLOTS, own], &UNIT_SIZES).with_peers(&[other]);
            attach(&mut tables, 1, 2);
            for (at, block) in runs.iter().enumerate() {
                product(&mut tables, 2, at as u32, block);
            }
            assert_eq!(tables.runs(2, PRODUCT, 1), runs[1]);

            // The second product's bytes from byte 16 lie 16 bytes into the rank's own run.
            let tensor = memory
                .lend(&tables, (2, PRODUCT, 1), true, 16, &[4, 8], DType::F32)
                .unwrap();
            assert_eq!(tensor.dims(), [4, 8]);
            let expected = u64::from(second) * UNIT_BYTES + 16;
            assert_eq!(address_of(&tensor), expected);
            assert_eq!(memory.address((own, second, 3), 16, 128).unwrap(), expected);

            // The whole block, and not a byte past it.
            assert!(
                memory
                    .lend(&tables, (2, PRODUCT, 1), false, 0, &[768], DType::U8)
                    .is_ok()
            );
            assert!(
                memory
                    .lend(&tables, (2, PRODUCT, 1), false, 1, &[768], DType::U8)
                    .is_err()
            );

            // A position without an entry, and a slot that holds none, lend nothing.
            assert!(
                memory
                    .lend(&tables, (2, PRODUCT, 2), false, 0, &[1], DType::U8)
                    .is_err()
            );
            assert!(
                memory
                    .lend(&tables, (3, PRODUCT, 0), false, 0, &[1], DType::U8)
                    .is_err()
            );
        }
    }

    #[test]
    fn a_block_another_rank_of_the_process_holds_is_lent_from_its_run() {
        // An encoder output in the other rank's product buffer, pinned: read by this rank too.
        let pin = Request::Pin {
            owner: OwnerId(1),
            class: PRODUCT,
            at: 0,
            keys: vec![Key([1; 32])],
        };
        let pinned = Completion::Pinned {
            blocks: vec![vec![(OTHER_PRODUCTS, 8, 2)]],
            host: 0,
        };
        let mut tables =
            BlockTables::new(GROUP, &[SLOTS, PRODUCTS], &UNIT_SIZES).with_peers(&[OTHER_PRODUCTS]);
        attach(&mut tables, 1, 1);
        tables.map(1, RequestId(1), &pin, &pinned).unwrap();
        assert_eq!(tables.run(1, PRODUCT, 0), None);

        let memories = process(0);
        let read = memories[0]
            .lend(&tables, (1, PRODUCT, 0), false, 0, &[64], DType::U8)
            .unwrap();
        assert_eq!(address_of(&read), 8 * UNIT_BYTES);
        assert!(
            memories[0]
                .lend(&tables, (1, PRODUCT, 0), true, 0, &[64], DType::U8)
                .is_err()
        );

        // A rank whose memory was joined to no other reaches no other rank's buffer.
        let alone = BufferMemory::meta(RANK, (PRODUCTS, 16), (SLOTS, 3), 0, &[]).unwrap();
        assert!(
            alone
                .lend(&tables, (1, PRODUCT, 0), false, 0, &[64], DType::U8)
                .is_err()
        );
    }

    #[test]
    fn a_read_only_block_is_lent_only_for_reading() {
        let memory = BufferMemory::meta(RANK, (PRODUCTS, 16), (SLOTS, 4), 0, &[]).unwrap();
        let mut tables = BlockTables::new(GROUP, &[SLOTS, PRODUCTS], &UNIT_SIZES);
        attach(&mut tables, 1, 1);

        // An encoder output that a request found cached: pinned, so read-only.
        let pin = Request::Pin {
            owner: OwnerId(1),
            class: PRODUCT,
            at: 0,
            keys: vec![Key([1; 32])],
        };
        let pinned = Completion::Pinned {
            blocks: vec![vec![(PRODUCTS, 4, 2)]],
            host: 0,
        };
        tables.map(1, RequestId(1), &pin, &pinned).unwrap();

        let read = memory
            .lend(&tables, (1, PRODUCT, 0), false, 0, &[64], DType::U8)
            .unwrap();
        assert_eq!(address_of(&read), 4 * UNIT_BYTES);
        assert!(
            memory
                .lend(&tables, (1, PRODUCT, 0), true, 0, &[64], DType::U8)
                .is_err()
        );
    }

    #[test]
    fn an_output_reaches_the_backing_ranks_that_do_not_run_its_call() {
        let memories = process(0);
        let mut tables =
            BlockTables::new(GROUP, &[SLOTS, PRODUCTS], &UNIT_SIZES).with_peers(&[OTHER_PRODUCTS]);
        attach(&mut tables, 1, 1);
        product(
            &mut tables,
            1,
            0,
            &[(PRODUCTS, 0, 3), (OTHER_PRODUCTS, 8, 3)],
        );

        // The rank that ran the call copies its run whole into the other rank's.
        let shared = memories[0].share(&tables, (1, PRODUCT, 0), None).unwrap();
        assert_eq!(
            shared,
            [(
                (PRODUCTS, RANK, 0),
                (OTHER_PRODUCTS, OTHER, 8 * UNIT_BYTES),
                3 * UNIT_BYTES
            )]
        );
        let output: Vec<u8> = (0..3 * UNIT_BYTES).map(|byte| (byte % 253) as u8).collect();
        let mut mirrors = HashMap::from([
            (
                (PRODUCTS, RANK),
                [output.clone(), vec![0; 13 * 256]].concat(),
            ),
            ((OTHER_PRODUCTS, OTHER), vec![0; 16 * 256]),
        ]);
        apply(&mut mirrors, &shared);
        let other = &mirrors[&(OTHER_PRODUCTS, OTHER)];
        assert_eq!(other[8 * 256..11 * 256], output[..]);

        // A rank that backs none of a block has nothing to share.
        let mut reader = BlockTables::new(GROUP, &[SLOTS, OTHER_PRODUCTS], &UNIT_SIZES);
        attach(&mut reader, 1, 1);
        product(&mut reader, 1, 0, &[(PRODUCTS, 4, 2)]);
        assert!(memories[1].share(&reader, (1, PRODUCT, 0), None).is_err());
    }

    #[test]
    fn a_trajectory_saved_and_restored_round_trips_through_its_host_block() {
        let memory = BufferMemory::meta(
            RANK,
            (PRODUCTS, 16),
            (SLOTS, 3),
            TRAJECTORY,
            &[(LATENTS, PAGE, 32)],
        )
        .unwrap();
        let mut tables = BlockTables::new(GROUP, &[SLOTS, PRODUCTS, LATENTS], &UNIT_SIZES);
        attach(&mut tables, 1, 2);
        attach(&mut tables, 2, 1);
        let save = |owner, units| Request::Save {
            owner: OwnerId(owner),
            key: Key([owner as u8; 32]),
            units,
        };

        // Slot 2 owns the second row, which a step lends from its first byte.
        let row = memory.trajectory(2, 0, &[250], DType::F32).unwrap();
        assert_eq!(address_of(&row), ROW);
        assert!(memory.trajectory(2, 4, &[250], DType::F32).is_err());
        assert!(memory.trajectory(4, 0, &[1], DType::F32).is_err());

        let trajectory: Vec<u8> = (0..TRAJECTORY).map(|byte| (byte % 251) as u8).collect();
        let mut mirrors = HashMap::from([
            ((SLOTS, RANK), vec![0xa5; (3 * ROW) as usize]),
            ((LATENTS, RANK), vec![0; (32 * PAGE) as usize]),
        ]);
        let slot = ROW as usize..(ROW + TRAJECTORY) as usize;
        mirrors.get_mut(&(SLOTS, RANK)).unwrap()[slot.clone()].copy_from_slice(&trajectory);
        let others = mirrors[&(SLOTS, RANK)].clone();

        // A save of four units from unit 5: the row's trajectory lands at the block's first byte
        // of the rank's part.
        let copy = started(
            &mut tables,
            (2, &save(1, 4)),
            ((SLOTS, 2, 1), (LATENTS, 5, 4)),
            None,
        );
        let copies = std::slice::from_ref(&copy);
        apply(&mut mirrors, &memory.byte_copies(copies).unwrap());
        let block = (5 * PAGE) as usize..(5 * PAGE + TRAJECTORY) as usize;
        assert_eq!(mirrors[&(LATENTS, RANK)][block], trajectory[..]);
        assert!(memory.copy(copies, None).is_ok());

        // The slot's row is lost, and a restore brings it back from the block.
        mirrors.get_mut(&(SLOTS, RANK)).unwrap()[slot.clone()].fill(0);
        let restore = Request::Restore {
            owner: OwnerId(1),
            key: Key([1; 32]),
        };
        let copy = started(
            &mut tables,
            (3, &restore),
            ((LATENTS, 5, 4), (SLOTS, 2, 1)),
            None,
        );
        apply(&mut mirrors, &memory.byte_copies(&[copy]).unwrap());
        assert_eq!(mirrors[&(SLOTS, RANK)], others);

        // A block of three units holds the first 768 bytes of slot 1's row, which a smaller
        // trajectory fills; a slot outside the rows, or a run past the part, is refused.
        let copy = started(
            &mut tables,
            (4, &save(2, 3)),
            ((SLOTS, 1, 1), (LATENTS, 20, 3)),
            None,
        );
        assert_eq!(
            memory.byte_copies(&[copy]).unwrap(),
            [((SLOTS, RANK, 0), (LATENTS, RANK, 20 * PAGE), 3 * PAGE)]
        );
        for (from, to) in [
            ((SLOTS, 4, 1), (LATENTS, 0, 4)),
            ((SLOTS, 1, 1), (LATENTS, 30, 4)),
        ] {
            let copy = BlockCopy {
                request: RequestId(5),
                runs: vec![(from, to)],
                encodings: None,
            };
            assert!(memory.byte_copies(&[copy]).is_err());
        }
    }

    #[test]
    fn a_trajectory_saved_by_two_ranks_is_restored_by_one_through_the_manifest() {
        // Saved under Ulysses 2: each rank's share of the samples in its own part of units 5
        // and 6. Restored by one rank of the same process, which holds every sample.
        let sources = [
            (RANK, share((0, 22), (0, 5))),
            (OTHER, share((22, 18), (5, 7))),
        ];
        let target = share((0, 40), (0, 12));
        let row = 3 * PAGE;
        let memories = process(row);
        let mut mirrors = HashMap::from([((SLOTS, RANK), vec![0; (3 * row) as usize])]);
        for (rank, stored) in &sources {
            let mut part = vec![0; (32 * PAGE) as usize];
            let saved = pages(stored);
            part[(5 * PAGE) as usize..(5 * PAGE) as usize + saved.len()].copy_from_slice(&saved);
            mirrors.insert((LATENTS, *rank), part);
        }

        let mut tables = BlockTables::new(GROUP, &[SLOTS, PRODUCTS, LATENTS], &UNIT_SIZES);
        attach(&mut tables, 1, 2);
        let restore = Request::Restore {
            owner: OwnerId(1),
            key: Key([1; 32]),
        };
        let copy = started(
            &mut tables,
            (2, &restore),
            ((LATENTS, 5, 2), (SLOTS, 2, 1)),
            Some(([1; 32], [2; 32])),
        );

        // A copy of bytes as they are does not lay another encoding out.
        assert!(
            memories[0]
                .byte_copies(std::slice::from_ref(&copy))
                .is_err()
        );

        let moved = memories[0].restored(&copy, &sources, &target).unwrap();
        assert!(moved.iter().any(|(from, ..)| from.1 == OTHER));
        apply(&mut mirrors, &moved);
        let restored = &mirrors[&(SLOTS, RANK)][row as usize..(2 * row) as usize];
        assert!(holds_its_samples(&target, restored));
        assert!(
            memories[0]
                .restore(&copy, (&sources, &target), None)
                .is_ok()
        );

        // A rank that reaches no other rank's part cannot read its share.
        let alone = BufferMemory::meta(
            RANK,
            (PRODUCTS, 16),
            (SLOTS, 3),
            row,
            &[(LATENTS, PAGE, 32)],
        )
        .unwrap();
        assert!(alone.restored(&copy, &sources, &target).is_err());
    }

    #[test]
    fn a_rank_without_a_part_copies_with_the_part_its_process_maps() {
        // Rank 0 maps the process's part of the host latent and encoder regions; the other rank
        // maps none, and holds trajectories in its rows.
        let parts = [(LATENTS, PAGE, 32), (ENCODERS, 512, 8)];
        let mapping = BufferMemory::meta(RANK, (PRODUCTS, 16), (SLOTS, 3), 0, &parts).unwrap();
        let rows = BufferMemory::meta(OTHER, (OTHER_PRODUCTS, 16), (SLOTS, 3), TRAJECTORY, &[]);
        let mut memories = vec![mapping, rows.unwrap()];
        assert!(!memories[1].holds(LATENTS));
        BufferMemory::connect(&mut memories).unwrap();

        let other = &memories[1];
        assert!(other.holds(LATENTS) && other.holds(ENCODERS));
        assert_eq!(other.part_holder(LATENTS), Some(RANK));
        assert_eq!(memories[0].part_holder(LATENTS), Some(RANK));
        let copy = |from, to| BlockCopy {
            request: RequestId(1),
            runs: vec![(from, to)],
            encodings: None,
        };

        // A save from the other rank's slot 2, a restore into its slot 1, and a store of an
        // encoder output in its buffer each move the whole block to or from rank 0's part.
        let copies = [
            copy((SLOTS, 2, 1), (LATENTS, 5, 4)),
            copy((LATENTS, 5, 4), (SLOTS, 1, 1)),
            copy((OTHER_PRODUCTS, 10, 5), (ENCODERS, 2, 3)),
        ];
        assert_eq!(
            other.byte_copies(&copies).unwrap(),
            [
                ((SLOTS, OTHER, ROW), (LATENTS, RANK, 5 * PAGE), TRAJECTORY),
                ((LATENTS, RANK, 5 * PAGE), (SLOTS, OTHER, 0), TRAJECTORY),
                (
                    (OTHER_PRODUCTS, OTHER, 10 * UNIT_BYTES),
                    (ENCODERS, RANK, 1024),
                    5 * UNIT_BYTES
                ),
            ]
        );

        // A rank that reaches two ranks' parts of a region and none of its own copies with
        // neither, and two memories of one rank index are not joined.
        let third = BufferMemory::meta(2, (7, 16), (SLOTS, 3), TRAJECTORY, &[]).unwrap();
        let mut joined = process(TRAJECTORY);
        joined.push(third);
        BufferMemory::connect(&mut joined).unwrap();
        assert_eq!(joined[2].part_holder(LATENTS), None);
        assert!(!joined[2].holds(LATENTS));
        assert!(
            joined[2]
                .byte_copies(&[copy((SLOTS, 2, 1), (LATENTS, 5, 4))])
                .is_err()
        );

        let twice = BufferMemory::meta(RANK, (8, 16), (SLOTS, 3), 0, &[]).unwrap();
        let mut refused = vec![
            BufferMemory::meta(RANK, (7, 16), (SLOTS, 3), 0, &[]).unwrap(),
            twice,
        ];
        assert!(BufferMemory::connect(&mut refused).is_err());
    }

    #[test]
    fn an_encoder_output_is_stored_and_loaded_whole_or_not_at_all() {
        // Host encoder units of 512 bytes on this rank's part.
        let memory =
            BufferMemory::meta(RANK, (PRODUCTS, 64), (SLOTS, 2), 0, &[(ENCODERS, 512, 8)]).unwrap();
        let copy = |from, to| BlockCopy {
            request: RequestId(1),
            runs: vec![(from, to)],
            encodings: None,
        };

        // An output of five product units fills three host units of the part, and the six
        // product units a load gives it hold the part's three whole.
        let store = copy((PRODUCTS, 10, 5), (ENCODERS, 2, 3));
        let load = copy((ENCODERS, 2, 3), (PRODUCTS, 30, 6));
        assert_eq!(
            memory.byte_copies(&[store, load]).unwrap(),
            [
                (
                    (PRODUCTS, RANK, 10 * UNIT_BYTES),
                    (ENCODERS, RANK, 1024),
                    5 * UNIT_BYTES
                ),
                (
                    (ENCODERS, RANK, 1024),
                    (PRODUCTS, RANK, 30 * UNIT_BYTES),
                    3 * 512
                ),
            ]
        );

        // A host block too small for the output, a device block too small for the part, a
        // clone between blocks of two sizes, and a copy between two host blocks are refused.
        for refused in [
            copy((PRODUCTS, 10, 5), (ENCODERS, 2, 2)),
            copy((ENCODERS, 2, 3), (PRODUCTS, 30, 5)),
            copy((PRODUCTS, 10, 5), (PRODUCTS, 20, 4)),
            copy((ENCODERS, 0, 1), (ENCODERS, 1, 1)),
        ] {
            assert!(memory.byte_copies(&[refused]).is_err());
        }
        assert_eq!(
            memory
                .byte_copies(&[copy((PRODUCTS, 10, 5), (PRODUCTS, 20, 5))])
                .unwrap(),
            [(
                (PRODUCTS, RANK, 10 * UNIT_BYTES),
                (PRODUCTS, RANK, 20 * UNIT_BYTES),
                5 * UNIT_BYTES
            )]
        );
    }
}
