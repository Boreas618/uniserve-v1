//! Physical range binding of scheduler-placed buffers in host arenas.

use uniserve_core::{CallId, RequestId};
use uniserve_ipc::{BufferAllocation, BufferId, RequestKey};
use uniserve_worker::ffi::{Backing, MemoryKind};
use uniserve_worker::storage::BufferPool;
use uniserve_worker::storage::LocalMemory;
use uniserve_worker::{Error, Result};

const HOST: MemoryKind = MemoryKind::Host;

fn allocation(output_index: u16, offset: u64, bytes: u64) -> BufferAllocation {
    BufferAllocation {
        buffer: BufferId {
            owner: RequestKey::new(1, RequestId(1), 1),
            producer_call_id: CallId::new(1, 0),
            output_index,
            generation: 1,
        },
        offset,
        bytes,
    }
}

#[test]
fn fixed_ranges_reserve_their_padding_until_released() -> Result<()> {
    let mut pool = BufferPool::allocate(32, &[HOST], &mut LocalMemory)?;

    let first = allocation(0, 0, 16);
    let (binding, arena) = pool.bind(first, HOST, 8, 4)?;
    let base = arena.address();
    assert_eq!(binding.physical_offset(), 0);

    // The unused tail remains occupied until its physical binding releases.
    assert!(pool.bind(allocation(1, 8, 8), HOST, 8, 4).is_err());
    let (neighbor, arena) = pool.bind(allocation(1, 16, 16), HOST, 16, 4)?;
    assert_eq!(neighbor.physical_offset(), 16);
    assert_eq!(arena.address(), base, "one arena per memory");
    // A memory the pool holds no arena of binds nothing.
    assert!(
        pool.bind(allocation(2, 0, 8), MemoryKind::PinnedHost, 8, 4)
            .is_err()
    );

    pool.release(&binding)?;
    let (replacement, _) = pool.bind(allocation(2, 8, 8), HOST, 8, 4)?;
    assert_eq!(replacement.physical_offset(), 8);
    pool.release(&replacement)?;
    pool.release(&neighbor)
}

#[test]
fn released_ranges_rebind_without_accepting_stale_or_foreign_handles() -> Result<()> {
    let mut pool = BufferPool::allocate(1024, &[HOST], &mut LocalMemory)?;
    let mut other = BufferPool::allocate(1024, &[HOST], &mut LocalMemory)?;
    let first = allocation(0, 0, 512);
    let second = allocation(1, 512, 512);
    let (binding, _) = pool.bind(first, HOST, 128, 4)?;
    let (foreign, _) = other.bind(first, HOST, 128, 4)?;
    assert!(matches!(pool.release(&foreign), Err(Error::Invariant(_))));

    // Every range binds at the offset the engine placed it.
    let (neighbor, _) = pool.bind(second, HOST, 128, 4)?;
    assert_eq!(binding.physical_offset(), 0);
    assert_eq!(neighbor.physical_offset(), 512);
    assert!(pool.bind(allocation(2, 256, 512), HOST, 128, 4).is_err());
    // A range beyond the pool this rank backs binds nothing.
    assert!(pool.bind(allocation(2, 1024, 256), HOST, 128, 4).is_err());

    pool.release(&binding)?;
    let (replacement, _) = pool.bind(first, HOST, 128, 4)?;
    assert_eq!(replacement.physical_offset(), 0);
    assert!(matches!(pool.release(&binding), Err(Error::Invariant(_))));
    assert!(pool.bind(first, HOST, 128, 4).is_err());

    pool.release(&replacement)?;
    pool.release(&neighbor)?;
    other.release(&foreign)?;

    assert!(
        pool.bind(allocation(2, 0, u64::MAX), HOST, u64::MAX, 1)
            .is_err()
    );
    let (replacement, _) = pool.bind(first, HOST, 128, 4)?;
    pool.release(&replacement)
}
