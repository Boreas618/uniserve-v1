//! Physical byte ranges backing scheduler-placed cross-call buffers.

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_ipc::{BufferAllocation, BufferId};

use crate::ffi::MemoryKind;
use crate::storage::memory::Memory;
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
    backing: Arc<Memory>,
    active: HashMap<BufferId, Arc<BufferBinding>>,
}

/// Bind nonoverlapping ranges of fixed per-device arenas.
/// The owner serializes binding, view creation and release.
pub struct BufferPool {
    byte_capacity: u64,
    compact: bool,
    arenas: HashMap<MemoryKind, Arena>,
}

impl BufferPool {
    fn new(byte_capacity: u64, compact: bool, arenas: HashMap<MemoryKind, Arc<Memory>>) -> Self {
        Self {
            byte_capacity,
            compact,
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

    /// Reserve a scheduler allocation and borrow its arena for view creation.
    /// Compact pools use the first aligned physical gap; other pools retain
    /// the scheduler offset. Release this handle if view creation fails.
    pub fn bind(
        &mut self,
        allocation: BufferAllocation,
        device: MemoryKind,
        required: u64,
        element_bytes: u64,
    ) -> Result<(Arc<BufferBinding>, &Arc<Memory>)> {
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

        let extent = if self.compact {
            align(required)
                .ok_or_else(|| Error::Invalid("buffer allocation byte extent overflows".into()))?
        } else {
            allocation.bytes
        };
        let arena = self.arenas.get_mut(&device).ok_or_else(|| {
            Error::Invalid("buffer allocation names an undeclared worker device".into())
        })?;
        if arena.active.contains_key(&allocation.buffer) {
            return Err(Error::Invalid("buffer allocation is already bound".into()));
        }

        let start = if self.compact {
            compact_offset(arena, extent, self.byte_capacity)?
        } else {
            allocation.offset
        };
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

fn compact_offset(arena: &Arena, extent: u64, capacity: u64) -> Result<u64> {
    let mut spans: Vec<_> = arena
        .active
        .values()
        .map(|binding| (binding.physical_offset, binding.physical_bytes))
        .collect();
    spans.sort_unstable();

    let mut cursor = 0;
    for &(offset, bytes) in &spans {
        if let Some(start) = align(cursor)
            && start.checked_add(extent).is_some_and(|end| end <= offset)
        {
            return Ok(start);
        }
        cursor = cursor.max(offset + bytes);
    }

    align(cursor)
        .filter(|&start| start.checked_add(extent).is_some_and(|end| end <= capacity))
        .ok_or_else(|| {
            Error::Invalid(format!(
                "physical buffer allocation exceeds the worker buffer pool: \
                 {extent} bytes requested from {capacity} bytes with live spans {spans:?}",
            ))
        })
}

/// Match the 256-byte product alignment used by startup capacity accounting.
fn align(bytes: u64) -> Option<u64> {
    bytes.checked_add(255).map(|bytes| bytes / 256 * 256)
}

impl BufferPool {
    /// Allocate one arena of `byte_capacity` bytes per memory in `devices`.
    /// Device arenas are exportable
    /// allocations, so CUDA VMM exports of persistent products are made in
    /// place; host arenas serve CPU lanes. Contents are uninitialized.
    pub fn allocate(byte_capacity: u64, compact: bool, devices: &[MemoryKind]) -> Result<Self> {
        let bytes = usize::try_from(byte_capacity)
            .map_err(|_| Error::Invalid("buffer pool capacity overflows".into()))?;
        let arenas = devices
            .iter()
            .map(|&device| {
                let arena = match device {
                    MemoryKind::Cuda(index) => Memory::exportable(index, bytes, &[])?,
                    MemoryKind::Host | MemoryKind::PinnedHost => Memory::host(bytes),
                };
                Ok((device, arena))
            })
            .collect::<Result<_>>()?;
        Ok(Self::new(byte_capacity, compact, arenas))
    }
}
