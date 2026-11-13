//! The memory a rank's media runners read and write: a request's media products, its encoder
//! outputs and its trajectory, as blocks of the buffer pool that the rank's device block tables
//! name.
//!
//! The buffer pool in the Storage Server decides which units of a region a block takes; the
//! rank's [`BlockTables`] keep, by slot, class and position, the run of each block that lies in
//! this rank's memory. [`BufferMemory`] is the memory those runs address on one rank, and lends
//! it:
//!
//! - **The product buffer.** One region of [`UNIT_BYTES`]-byte units, taken from the device's
//!   resident pool (`GraphPool::allocator`) for as long as the rank serves. A media product has
//!   one contiguous run in the product buffer of every rank that backs it, and an encoder output
//!   one run; [`BufferMemory::lend`] lends the bytes of the block an entry of the tables names as
//!   a tensor, and [`BufferMemory::address`] states where a run's bytes lie for a copy.
//! - **The trajectory rows.** The live state of a media request in its slot: one row per slot of
//!   the slot region, each the bytes of the largest trajectory the rank holds, starting on a
//!   [`UNIT_BYTES`] boundary. Slot `s` (slots count from one) owns row `s - 1`, the run
//!   `(slot region, s, 1)` names it, and [`BufferMemory::trajectory`] lends from it. A trajectory
//!   is laid out from its row's first byte, so a smaller one is a prefix of the row.
//! - **The rank's parts of host regions.** The host encoder region and the host latent region
//!   are the Storage Server's: one memfd per rank index (`storage::protocol::Request::HostPart`),
//!   unit `u` of a rank's part at byte `u × bytes`, mapped here writable and registered with the
//!   device ([`BufferMemory::map_part`]). A host block's run is the same units in every rank's
//!   part, so its bytes on this rank are one contiguous range of the part.
//!
//! ## Copies
//!
//! A `Copy` completion queues the copies a rank runs in its tables, and the runner that serves
//! the next batch issues them ([`Runner::copy`](crate::Runner::copy)). Every media runner of a
//! rank shares one [`BufferMemory`], so whichever runner serves that batch issues every copy
//! through [`BufferMemory::copy`]:
//!
//! | From | To | Request | Bytes copied |
//! | --- | --- | --- | --- |
//! | product buffer | product buffer | `Clone` | the block's, both runs of one size |
//! | product buffer | a host part | `Store` of an encoder output | the device block's, which the rank's part of the host block holds |
//! | a host part | product buffer | `Load` of an encoder output | as above |
//! | a slot's row | a host part | `Save` | the row's first bytes, as many as the rank's part of the host block holds |
//! | a host part | a slot's row | `Restore` | as above |
//!
//! The size of a sized block is the scheduler's (`specs/buffer-pool.md` §11): a trajectory block
//! holds the trajectory of the request's layout, which is the first bytes of its row.
//!
//! ## Ordering
//!
//! A copy is issued on the stream of the runner that serves the batch it starts with, before
//! that batch's forward, and the batch's end covers it. A block a batch writes is read by a later
//! batch, which the scheduler forms only once the writer's output arrived, so every read is
//! ordered after its write by completion, on whichever runner's stream either was issued.
//!
//! ## Without a device
//!
//! [`BufferMemory::meta`] holds no memory: its tensors are meta tensors, its addresses are the
//! byte offsets of each memory from 0, and its copies are checked and issue nothing. Its
//! [`BufferMemory::byte_copies`] are the copies a memory on a device issues.

use std::sync::Arc;

use platform::{Allocator, Buffer, DeviceId, PoolKind, Stream};
use storage::runtime::{RequestId, Run};
use uniserve_tensor::{DType, Tensor};

use crate::memory::allocation::Storage;
use crate::memory::client::StorageClient;
use crate::memory::lend;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::{Error, Result};

/// The bytes of one unit of a rank's product buffer: every product and encoder output starts on
/// a 256-byte boundary, and the region's unit size in `Open` is this.
pub const UNIT_BYTES: u64 = 256;

/// A byte of the rank's media memory: the region whose memory holds it, by the number the answer
/// to `Open` gave it, and its offset from that memory's first byte (the product buffer's, the
/// trajectory rows' for the slot region, or a host part's).
pub type Place = (u8, u64);

/// One copy of bytes: where it reads, where it writes, and how many bytes.
pub type ByteCopy = (Place, Place, u64);

/// A rank's media memory: see the module documentation.
pub struct BufferMemory {
    /// The device the memory lies on, with the product buffer and the trajectory rows taken from
    /// its resident pool, each `None` when it holds no byte; `None` for memory that holds no
    /// device.
    device: Option<(DeviceId, Option<Buffer>, Option<Buffer>)>,
    /// The product buffer's region and its units.
    products: (u8, u32),
    /// The slot region and its slots, numbered from 1.
    slots: (u8, u32),
    /// The bytes of one slot's trajectory; 0 for a rank that holds none.
    trajectory: u64,
    /// The bytes from one row's first byte to the next's: the trajectory's, rounded up to a
    /// whole unit.
    row: u64,
    /// The rank's part of each host region it maps: the region, the bytes of every unit of the
    /// part, its units, and its mapping (`None` for memory that holds no device).
    parts: Vec<(u8, u64, u32, Option<Arc<Storage>>)>,
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
/// and the rank's part of a host block, `host` bytes: the device block's, which the part holds
/// whole or the copy is refused.
fn stored(request: RequestId, block: u64, host: u64) -> Result<u64> {
    if host < block {
        return Err(Error::Invalid(format!(
            "request {} copies an encoder output of {block} bytes, and the rank's part of its \
             host block holds {host}",
            request.0
        )));
    }

    Ok(block)
}

impl BufferMemory {
    /// The device bytes the media memory of a rank takes from the device's resident pool: a
    /// product buffer of `units` units and the rows of `slots` slots for trajectories of
    /// `trajectory` bytes. The pool is reserved with them when the device's `GraphPool` opens.
    pub fn device_bytes(units: u32, slots: u32, trajectory: u64) -> u64 {
        let rows = row_bytes(trajectory)
            .and_then(|row| row.checked_mul(u64::from(slots)))
            .unwrap_or(u64::MAX);

        (u64::from(units) * UNIT_BYTES).saturating_add(rows)
    }

    /// The media memory of a rank, holding no device: `products` is the product buffer's region
    /// and units, `slots` the slot region and its slots, `trajectory` the bytes of one slot's
    /// trajectory (0 for a rank that holds none), and `parts` the rank's part of each host region
    /// it addresses (the region, the bytes of every unit of the part, its units). Its tensors
    /// are meta tensors, and its copies are checked and issue nothing.
    ///
    /// # Errors
    ///
    /// Two memories stated under one region, a part without bytes, or a trajectory whose rows
    /// overflow.
    pub fn meta(
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
            device: None,
            products,
            slots,
            trajectory,
            row,
            parts: Vec::new(),
        };
        for &(region, bytes, units) in parts {
            memory.check_part(region, bytes)?;
            memory.parts.push((region, bytes, units, None));
        }
        Ok(memory)
    }

    /// The media memory of a rank on the device of `allocator`, as [`BufferMemory::meta`] states
    /// it with no host part: the product buffer and the trajectory rows are taken from the
    /// resident pool, at addresses that stay fixed for as long as the memory lives, and the host
    /// parts are mapped afterwards ([`BufferMemory::map_part`]). Their contents are unspecified:
    /// a block is written before a batch reads it, and a trajectory row by its request's latent
    /// preparation or by a `Restore`.
    ///
    /// # Errors
    ///
    /// As [`BufferMemory::meta`], and when the resident pool cannot hold the memory.
    pub fn open(
        allocator: &mut Allocator,
        products: (u8, u32),
        slots: (u8, u32),
        trajectory: u64,
    ) -> Result<Self> {
        let mut memory = Self::meta(products, slots, trajectory, &[])?;

        let mut take = |bytes: u64| -> Result<Option<Buffer>> {
            if bytes == 0 {
                return Ok(None);
            }
            let bytes = usize::try_from(bytes)
                .map_err(|_| Error::Resource("a rank's media memory exceeds the address space"))?;

            Ok(Some(allocator.allocate(PoolKind::Resident, bytes)?))
        };
        let buffer = take(u64::from(products.1) * UNIT_BYTES)?;
        let rows = take(memory.row * u64::from(slots.1))?;

        memory.device = Some((allocator.device(), buffer, rows));
        Ok(memory)
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
    pub fn map_part(
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
        self.check_part(region, bytes)?;

        let (units, descriptor) = client
            .call("taking the rank's part of a host region", |client| {
                client.host_part(name.to_owned(), bytes, numa_node)
            })?;
        let length = u64::from(units)
            .checked_mul(bytes)
            .and_then(|length| usize::try_from(length).ok())
            .ok_or(Error::Resource(
                "a rank's part of a host region exceeds the address space",
            ))?;
        let storage = Storage::map_host(descriptor, length, device, true)?;
        tracing::info!(
            region = name,
            units,
            bytes,
            numa_node,
            "mapped the rank's part of a host region"
        );

        self.parts.push((region, bytes, units, Some(storage)));
        Ok(units)
    }

    /// The device the memory lies on; `None` for memory that holds none.
    pub fn device(&self) -> Option<DeviceId> {
        self.device.as_ref().map(|(device, ..)| *device)
    }

    /// Whether `region` is memory of this rank's media: its product buffer, its slot rows when it
    /// holds trajectories, or a host region whose part it maps. A runner that is handed the copies
    /// of other memory too, such as the KV cache's, issues through this memory those that name
    /// it.
    pub fn holds(&self, region: u8) -> bool {
        region == self.products.0
            || (region == self.slots.0 && self.trajectory > 0)
            || self.part(region).is_some()
    }

    /// Lends the bytes of the block that `slot` holds at `position` of `class` in the rank's
    /// tables, from byte `offset` of its run in the product buffer, as a tensor of `shape`
    /// elements of `dtype`: on the device, or a meta tensor for memory that holds none. `writes`
    /// states that a batch writes the tensor.
    ///
    /// The tensor addresses the product buffer, which lives as long as this memory: its holder
    /// keeps the memory while it uses the tensor.
    ///
    /// # Errors
    ///
    /// A position the tables hold no entry for; a write to an entry they do not state writable;
    /// a block that does not lie in the product buffer; elements that reach past the block.
    pub fn lend(
        &self,
        tables: &BlockTables,
        (slot, class, position): (u32, u8, u32),
        writes: bool,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        let run = tables.run(slot, class, position).ok_or_else(|| {
            Error::Invalid(format!(
                "slot {slot} holds no block at position {position} of class {class} on this rank"
            ))
        })?;
        if writes && !tables.writable(slot, class, position) {
            return Err(Error::Invalid(format!(
                "a batch writes the block of slot {slot} at position {position} of class \
                 {class}, which is read-only"
            )));
        }
        if run.0 != self.products.0 {
            return Err(Error::Invalid(format!(
                "the block of slot {slot} at position {position} of class {class} lies in \
                 region {}, which is not the rank's product buffer",
                run.0
            )));
        }

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
    pub fn trajectory(
        &self,
        slot: u32,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        let place = self.window((self.slots.0, slot, 1), offset, tensor_bytes(shape, dtype)?)?;
        self.tensor(place, shape, dtype)
    }

    /// The address of `bytes` bytes from byte `offset` of `run` in this rank's memory: a device
    /// address in the product buffer or a slot's row, a host address in a host part, or for
    /// memory that holds no device the offset of the bytes from its memory's first byte.
    ///
    /// # Errors
    ///
    /// A run of a region that is no memory of this rank's media, a run past its memory's end, or
    /// bytes that reach past the run.
    pub fn address(&self, run: Run, offset: u64, bytes: u64) -> Result<u64> {
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
    /// device block it stores or loads; any other pair of memories.
    pub fn byte_copies(&self, copies: &[BlockCopy]) -> Result<Vec<ByteCopy>> {
        let mut moved = Vec::new();
        for copy in copies {
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
    pub fn copy(&self, copies: &[BlockCopy], stream: Option<&Stream>) -> Result<()> {
        let moved = self.byte_copies(copies)?;

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

        for (from, to, bytes) in moved {
            if bytes == 0 {
                continue;
            }
            let (source, destination) = (self.at(from)?, self.at(to)?);
            let bytes = usize::try_from(bytes)
                .map_err(|_| Error::Resource("a copy exceeds the address space"))?;

            // SAFETY: both ranges lie inside memory this rank maps for as long as it lives
            // (`Self::window`). The buffer pool holds the block a copy reads and the one it writes
            // until the copy's `Finish` is taken, which follows the end of the batch this stream
            // runs next; no batch addresses the destination until then, since no table entry
            // names it or, for a slot's row, the scheduler runs no step of the slot meanwhile.
            unsafe { platform::memcpy_async(destination, source, bytes, stream) }?;
        }
        Ok(())
    }

    /// The bytes a copy from `from` to `to`, for `request`, moves: see the module
    /// documentation.
    fn byte_copy(&self, request: RequestId, from: Run, to: Run) -> Result<ByteCopy> {
        // `extent` stated every run's region: the product buffer, the slot rows, or a host part.
        let (read, written) = (self.extent(from)?, self.extent(to)?);
        let (products, slots) = (self.products.0, self.slots.0);
        let part = |region: u8| region != products && region != slots;

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

            // An encoder output between the product buffer and a host part: the host block was
            // sized for the device block's bytes, which the rank's part of it must hold.
            (source, destination) if source == products && part(destination) => {
                stored(request, read.1, written.1)?
            }
            (source, destination) if part(source) && destination == products => {
                stored(request, written.1, read.1)?
            }

            // A trajectory between a slot's row and a host part: the block holds the request's
            // trajectory, the first bytes of the row.
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

        Ok(((from.0, read.0), (to.0, written.0), bytes))
    }

    /// The first byte and the bytes of `run` in this rank's memory, from its memory's first
    /// byte.
    fn extent(&self, (region, first, count): Run) -> Result<(u64, u64)> {
        if region == self.slots.0 {
            // A slot's run is its row: slot `s` owns row `s - 1`.
            if self.trajectory == 0 || count != 1 || first == 0 || first > self.slots.1 {
                return Err(Error::Invalid(format!(
                    "the run of {count} slots from slot {first} names no trajectory row of a \
                     rank that holds {} rows of {} bytes",
                    self.slots.1, self.trajectory
                )));
            }
            return Ok((u64::from(first - 1) * self.row, self.trajectory));
        }

        let (unit, units) = if region == self.products.0 {
            (UNIT_BYTES, self.products.1)
        } else if let Some((_, bytes, units, _)) = self.part(region) {
            (*bytes, *units)
        } else {
            return Err(Error::Invalid(format!(
                "region {region} is no memory of this rank's media"
            )));
        };

        let end = first.checked_add(count);
        if count == 0 || end.is_none_or(|end| end > units) {
            return Err(Error::Invalid(format!(
                "the run of {count} units from unit {first} of region {region} lies outside its \
                 {units} units on this rank"
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
        Ok((run.0, first + offset))
    }

    /// The address of `place`: see [`BufferMemory::address`].
    fn at(&self, (region, offset): Place) -> Result<u64> {
        let Some((_, buffer, rows)) = &self.device else {
            return Ok(offset);
        };

        let base = if region == self.products.0 {
            buffer.as_ref().map(Buffer::address)
        } else if region == self.slots.0 {
            rows.as_ref().map(Buffer::address)
        } else {
            self.part(region)
                .and_then(|(.., storage)| storage.as_ref())
                .map(|storage| storage.address())
        };

        base.map(|base| base + offset).ok_or_else(|| {
            Error::Invalid(format!(
                "region {region} has no memory on this rank's device"
            ))
        })
    }

    /// The tensor over `place`, a place in the product buffer or the trajectory rows.
    fn tensor(&self, place: Place, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let address = self.at(place)?;

        // SAFETY: the window lies inside the product buffer or the trajectory rows
        // (`Self::window`), which this memory holds for as long as it lives and its holder keeps
        // it while the tensor's work may read it. Between batches only the copies this memory
        // issues write them, each into a block no batch addresses until the copy ended; a meta
        // tensor's address is never read.
        unsafe { lend(address, shape, dtype, self.device()) }
    }

    /// The rank's part of host region `region`.
    fn part(&self, region: u8) -> Option<&(u8, u64, u32, Option<Arc<Storage>>)> {
        self.parts.iter().find(|(held, ..)| *held == region)
    }

    /// Whether the part of `region`, of `bytes` of every unit, can be added to this memory.
    fn check_part(&self, region: u8, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Err(Error::Invalid(format!(
                "the rank's part of host region {region} holds no byte of a unit"
            )));
        }
        if region == self.products.0 || region == self.slots.0 || self.part(region).is_some() {
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

    const GROUP: GroupId = GroupId(0);

    /// The slot rows, the product buffers of two ranks, and the host encoder and latent regions.
    const SLOTS: u8 = 0;
    const PRODUCTS: u8 = 2;
    const OTHER_PRODUCTS: u8 = 3;
    const ENCODERS: u8 = 5;
    const LATENTS: u8 = 6;

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

    /// Hands `tables` the `Copy` that answers `request`, one key from `from` to `to`, and starts
    /// it as the next batch would.
    fn started(
        tables: &mut BlockTables,
        id: u64,
        request: &Request,
        from: Run,
        to: Run,
    ) -> BlockCopy {
        let copy = Completion::Copy {
            group: GROUP,
            keys: vec![Key([7; 32])],
            from: vec![from],
            to: vec![to],
            evicted: vec![],
        };
        tables.map(0, RequestId(id), request, &copy).unwrap();

        tables.start_copy().unwrap()
    }

    /// Performs `copies` on host mirrors of the rank's memories, by region.
    fn apply(mirrors: &mut HashMap<u8, Vec<u8>>, copies: &[ByteCopy]) {
        for &((from, source), (to, destination), bytes) in copies {
            let (source, destination, bytes) =
                (source as usize, destination as usize, bytes as usize);

            let read = mirrors[&from][source..source + bytes].to_vec();
            mirrors.get_mut(&to).unwrap()[destination..destination + bytes].copy_from_slice(&read);
        }
    }

    /// The address a lent tensor's first element lies at.
    fn address_of(tensor: &Tensor) -> u64 {
        backend::operand(tensor).unwrap().ptr.address()
    }

    #[test]
    fn a_products_runs_on_each_rank_lend_its_bytes_at_their_offsets() {
        // Two products of three units each, backed by two ranks: the runs are block-major,
        // one per backing rank.
        let allocate = Request::Allocate {
            owner: OwnerId(1),
            class: PRODUCT,
            at: 0,
            count: 2,
            units: 3,
        };
        let allocated = Completion::Allocated {
            runs: vec![
                (PRODUCTS, 0, 3),
                (OTHER_PRODUCTS, 8, 3),
                (PRODUCTS, 3, 3),
                (OTHER_PRODUCTS, 11, 3),
            ],
            evicted: vec![],
        };

        for (region, second) in [(PRODUCTS, 3u32), (OTHER_PRODUCTS, 11)] {
            let memory = BufferMemory::meta((region, 16), (SLOTS, 4), 0, &[]).unwrap();
            let mut tables = BlockTables::new(GROUP, &[SLOTS, region], &UNIT_SIZES);
            attach(&mut tables, 1, 2);
            tables.map(2, RequestId(1), &allocate, &allocated).unwrap();

            // The second product's bytes from byte 16 lie 16 bytes into the rank's own run.
            let tensor = memory
                .lend(&tables, (2, PRODUCT, 1), true, 16, &[4, 8], DType::F32)
                .unwrap();
            assert_eq!(tensor.dims(), [4, 8]);
            let expected = u64::from(second) * UNIT_BYTES + 16;
            assert_eq!(address_of(&tensor), expected);
            assert_eq!(
                memory.address((region, second, 3), 16, 128).unwrap(),
                expected
            );

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
    fn a_read_only_block_is_lent_only_for_reading() {
        let memory = BufferMemory::meta((PRODUCTS, 16), (SLOTS, 4), 0, &[]).unwrap();
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
            runs: vec![(PRODUCTS, 4, 2)],
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
    fn a_trajectory_saved_and_restored_round_trips_through_its_host_block() {
        let memory = BufferMemory::meta(
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
            (SLOTS, vec![0xa5; (3 * ROW) as usize]),
            (LATENTS, vec![0; (32 * PAGE) as usize]),
        ]);
        let slot = ROW as usize..(ROW + TRAJECTORY) as usize;
        mirrors.get_mut(&SLOTS).unwrap()[slot.clone()].copy_from_slice(&trajectory);
        let others = mirrors[&SLOTS].clone();

        // A save of four units from unit 5: the row's trajectory lands at the block's first byte
        // of the rank's part.
        let copy = started(&mut tables, 2, &save(1, 4), (SLOTS, 2, 1), (LATENTS, 5, 4));
        let copies = std::slice::from_ref(&copy);
        apply(&mut mirrors, &memory.byte_copies(copies).unwrap());
        let block = (5 * PAGE) as usize..(5 * PAGE + TRAJECTORY) as usize;
        assert_eq!(mirrors[&LATENTS][block], trajectory[..]);
        assert!(memory.copy(copies, None).is_ok());

        // The slot's row is lost, and a restore brings it back from the block.
        mirrors.get_mut(&SLOTS).unwrap()[slot.clone()].fill(0);
        let restore = Request::Restore {
            owner: OwnerId(1),
            key: Key([1; 32]),
        };
        let copy = started(&mut tables, 3, &restore, (LATENTS, 5, 4), (SLOTS, 2, 1));
        apply(&mut mirrors, &memory.byte_copies(&[copy]).unwrap());
        assert_eq!(mirrors[&SLOTS], others);

        // A block of three units holds the first 768 bytes of slot 1's row, which a smaller
        // trajectory fills; a slot outside the rows, or a run past the part, is refused.
        let copy = started(&mut tables, 4, &save(2, 3), (SLOTS, 1, 1), (LATENTS, 20, 3));
        assert_eq!(
            memory.byte_copies(&[copy]).unwrap(),
            [((SLOTS, 0), (LATENTS, 20 * PAGE), 3 * PAGE)]
        );
        for (from, to) in [
            ((SLOTS, 4, 1), (LATENTS, 0, 4)),
            ((SLOTS, 1, 1), (LATENTS, 30, 4)),
        ] {
            let copy = BlockCopy {
                request: RequestId(5),
                runs: vec![(from, to)],
            };
            assert!(memory.byte_copies(&[copy]).is_err());
        }
    }

    #[test]
    fn an_encoder_output_is_stored_and_loaded_whole_or_not_at_all() {
        // Host encoder units of 512 bytes on this rank.
        let memory =
            BufferMemory::meta((PRODUCTS, 64), (SLOTS, 2), 0, &[(ENCODERS, 512, 8)]).unwrap();
        let copy = |from, to| BlockCopy {
            request: RequestId(1),
            runs: vec![(from, to)],
        };

        // An output of five product units fills three host units on this rank.
        let store = copy((PRODUCTS, 10, 5), (ENCODERS, 2, 3));
        let load = copy((ENCODERS, 2, 3), (PRODUCTS, 30, 5));
        assert_eq!(
            memory.byte_copies(&[store, load]).unwrap(),
            [
                (
                    (PRODUCTS, 10 * UNIT_BYTES),
                    (ENCODERS, 1024),
                    5 * UNIT_BYTES
                ),
                (
                    (ENCODERS, 1024),
                    (PRODUCTS, 30 * UNIT_BYTES),
                    5 * UNIT_BYTES
                ),
            ]
        );

        // A host block too small for the output, a clone between blocks of two sizes, and a copy
        // between two host blocks are refused.
        for refused in [
            copy((PRODUCTS, 10, 5), (ENCODERS, 2, 2)),
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
                (PRODUCTS, 10 * UNIT_BYTES),
                (PRODUCTS, 20 * UNIT_BYTES),
                5 * UNIT_BYTES
            )]
        );
    }
}
