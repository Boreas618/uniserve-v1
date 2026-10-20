//! Physical byte ranges backing scheduler-placed cross-call buffers.

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_ipc::{BufferAllocation, BufferId};

use crate::ffi::MemoryKind;
use crate::storage::memory::Storage;
use crate::{Error, Result};

/// One pool-issued physical range. Views retain their allocation; this
/// handle identifies the occupied range until all accesses retire.
pub struct BufferBinding {
    buffer: BufferId,
    device: MemoryKind,
    physical_offset: u64,
    physical_bytes: u64,
}

impl BufferBinding {
    pub fn physical_offset(&self) -> u64 {
        self.physical_offset
    }
}

struct Arena {
    backing: Arc<Storage>,
    active: HashMap<BufferId, Arc<BufferBinding>>,
}

/// Bind nonoverlapping ranges of fixed per-device arenas at the offsets the
/// engine placed them, in the pool this rank backs.
/// The owner serializes binding, view creation and release.
pub struct BufferPool {
    byte_capacity: u64,
    arenas: HashMap<MemoryKind, Arena>,
}

impl BufferPool {
    fn new(byte_capacity: u64, arenas: HashMap<MemoryKind, Arc<Storage>>) -> Self {
        Self {
            byte_capacity,
            arenas: arenas
                .into_iter()
                .map(|(device, backing)| {
                    (
                        device,
                        Arena {
                            backing,
                            active: HashMap::new(),
                        },
                    )
                })
                .collect(),
        }
    }

    /// Reserve a scheduler allocation at its offset and borrow its arena for
    /// view creation. Release this handle if view creation fails.
    pub fn bind(
        &mut self,
        allocation: BufferAllocation,
        device: MemoryKind,
        required: u64,
        element_bytes: u64,
    ) -> Result<(Arc<BufferBinding>, &Arc<Storage>)> {
        if required == 0 || element_bytes == 0 {
            return Err(Error::Invalid(
                "buffer tensor has an invalid byte extent".into(),
            ));
        }
        if required > allocation.bytes {
            return Err(Error::Invalid(
                "buffer allocation is smaller than its output tensor".into(),
            ));
        }
        if !allocation.offset.is_multiple_of(element_bytes) {
            return Err(Error::Invalid(
                "buffer allocation is not aligned for its output dtype".into(),
            ));
        }

        let extent = allocation.bytes;
        let arena = self.arenas.get_mut(&device).ok_or_else(|| {
            Error::Invalid("buffer allocation names an undeclared worker device".into())
        })?;
        if arena.active.contains_key(&allocation.buffer) {
            return Err(Error::Invalid("buffer allocation is already bound".into()));
        }

        let start = allocation.offset;
        let end = start
            .checked_add(extent)
            .filter(|&end| end <= self.byte_capacity)
            .ok_or_else(|| {
                Error::Invalid("buffer allocation exceeds the worker buffer pool".into())
            })?;
        if arena.active.values().any(|active| {
            start < active.physical_offset + active.physical_bytes && active.physical_offset < end
        }) {
            return Err(Error::Invalid(
                "buffer allocation overlaps a live worker buffer".into(),
            ));
        }

        let binding = Arc::new(BufferBinding {
            buffer: allocation.buffer,
            device,
            physical_offset: start,
            physical_bytes: extent,
        });
        arena.active.insert(allocation.buffer, Arc::clone(&binding));
        Ok((binding, &arena.backing))
    }

    /// Release exactly the handle issued for this live range. An old handle
    /// cannot release a replacement even if its logical buffer and span match.
    /// The owner must drain device uses and transport readers first.
    pub fn release(&mut self, binding: &Arc<BufferBinding>) -> Result<()> {
        let arena = self
            .arenas
            .get_mut(&binding.device)
            .ok_or_else(|| Error::Invariant("stale persistent buffer binding".into()))?;
        if !arena
            .active
            .get(&binding.buffer)
            .is_some_and(|active| Arc::ptr_eq(active, binding))
        {
            return Err(Error::Invariant("stale persistent buffer binding".into()));
        }

        arena.active.remove(&binding.buffer);
        Ok(())
    }
}

impl BufferPool {
    /// Allocate one arena of `byte_capacity` bytes per memory in `devices`.
    /// Device arenas are exclusive storages `memory` provides, exportable in
    /// place, so CUDA VMM exports of persistent products are made in place;
    /// host arenas serve CPU lanes. Contents are uninitialized.
    pub fn allocate(
        byte_capacity: u64,
        devices: &[MemoryKind],
        memory: &mut dyn crate::storage::retained::ExclusiveMemory,
    ) -> Result<Self> {
        let bytes = usize::try_from(byte_capacity)
            .map_err(|_| Error::Invalid("buffer pool capacity overflows".into()))?;
        let arenas = devices
            .iter()
            .map(|&device| {
                let arena = match device {
                    MemoryKind::Cuda(index) => memory
                        .exclusive(&[crate::storage::retained::ExclusiveSpec {
                            name: "buffer_pool",
                            device: index,
                            bytes,
                            access: &[],
                        }])?
                        .pop()
                        .map(|taken| taken.storage)
                        .ok_or_else(|| {
                            Error::Invariant("the buffer pool's storage is missing".into())
                        })?,
                    MemoryKind::Host | MemoryKind::PinnedHost => Storage::host(bytes),
                };
                Ok((device, arena))
            })
            .collect::<Result<_>>()?;
        Ok(Self::new(byte_capacity, arenas))
    }
}
