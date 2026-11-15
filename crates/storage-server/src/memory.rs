//! The host memory the server creates: memfds on the host's huge pages when it has them to
//! give, placed on a NUMA node.

use std::io;
use std::os::fd::OwnedFd;

/// A memfd named `name` of `bytes`, backed by huge pages when the host has
/// them to give, whose pages are allocated on `numa_node` when one is
/// given: a rank's part of a host region, or the host copy of an epoch's
/// weights.
pub(crate) fn shared_memory(name: &str, bytes: u64, numa_node: Option<u32>) -> io::Result<OwnedFd> {
    use rustix::fs::{MemfdFlags, ftruncate, memfd_create};

    // Huge pages hold the memory in fewer TLB entries for the copies that
    // read and write it. A hugetlb memfd is sized in whole pages of the host's default huge page
    // size, which the file states as its block size; a shared mapping reserves its pages: a
    // mapping the host cannot reserve fails, so one is made and dropped here before the memory
    // is passed on.
    let huge = memfd_create(name, MemfdFlags::CLOEXEC | MemfdFlags::HUGETLB)
        .ok()
        .and_then(|descriptor| {
            let page = u64::try_from(rustix::fs::fstat(&descriptor).ok()?.st_blksize).ok()?;
            let size = bytes.checked_next_multiple_of(page)?;
            (ftruncate(&descriptor, size).is_ok() && reserves(&descriptor, size))
                .then_some((descriptor, size))
        });
    let (descriptor, size) = match huge {
        Some(huge) => huge,
        None => {
            let descriptor = memfd_create(name, MemfdFlags::CLOEXEC)?;
            ftruncate(&descriptor, bytes)?;
            (descriptor, bytes)
        }
    };
    if let Some(node) = numa_node {
        prefer_node(&descriptor, size, node)?;
    }
    Ok(descriptor)
}

/// Make `node` the preferred NUMA node of the pages of the shared memory
/// object `descriptor` holds, `bytes` long: a shared object keeps the
/// policy of a mapping of it for every process that maps it later, so the
/// pages land on `node` whoever first touches them. Pages fall back to
/// other nodes when `node` has none free.
pub(crate) fn prefer_node(descriptor: &OwnedFd, bytes: u64, node: u32) -> io::Result<()> {
    use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

    let length = usize::try_from(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "host memory too large"))?;
    // The node mask holds the bits of nodes 0 to `node`, in whole words: a node the kernel does
    // not know is refused by `mbind`.
    let mut mask = vec![0_u64; node as usize / 64 + 1];
    mask[node as usize / 64] |= 1 << (node % 64);
    let mask_bits = mask.len() as u64 * 64;
    // SAFETY: a new shared mapping of the memfd, which this function unmaps;
    // nothing reads or writes it.
    let address = unsafe {
        mmap(
            std::ptr::null_mut(),
            length,
            ProtFlags::READ,
            MapFlags::SHARED,
            descriptor,
            0,
        )
    }?;
    // SAFETY: mbind of the mapping made above, with a node mask of `mask_bits` bits.
    let bound = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            address,
            length,
            libc::MPOL_PREFERRED,
            mask.as_ptr(),
            // The kernel reads one bit fewer than `maxnode` states (mbind(2)).
            mask_bits + 1,
            0_u32,
        )
    };
    let error = (bound != 0).then(io::Error::last_os_error);
    // SAFETY: the mapping made above.
    unsafe { munmap(address, length) }?;
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Whether a shared mapping of `bytes` of `descriptor` can be made.
fn reserves(descriptor: &OwnedFd, bytes: u64) -> bool {
    use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

    let Ok(length) = usize::try_from(bytes) else {
        return false;
    };
    // SAFETY: a new mapping this function makes and unmaps; nothing reads
    // or writes it.
    match unsafe {
        mmap(
            std::ptr::null_mut(),
            length,
            ProtFlags::READ,
            MapFlags::SHARED,
            descriptor,
            0,
        )
    } {
        Ok(address) => {
            // SAFETY: the mapping made above.
            let _ = unsafe { munmap(address, length) };
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_for_a_numa_node_allocates_its_pages_there() {
        use std::os::fd::AsFd;

        use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

        // The last node with CPUs, as a host's sockets are: on a host of one
        // node, node 0.
        let online = std::fs::read_to_string("/sys/devices/system/node/has_cpu")
            .unwrap_or_else(|_| "0".into());
        let node: u32 = online
            .trim()
            .rsplit([',', '-'])
            .next()
            .and_then(|node| node.parse().ok())
            .unwrap_or(0);
        let bytes = 4 << 20;
        let descriptor = shared_memory("uniserve-host-memory", bytes, Some(node)).unwrap();

        // A process's first touch allocates the pages, wherever it runs.
        // SAFETY: a new shared mapping of the memfd, unmapped below.
        let address = unsafe {
            mmap(
                std::ptr::null_mut(),
                bytes as usize,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                descriptor.as_fd(),
                0,
            )
        }
        .unwrap();
        // SAFETY: writes inside the mapping made above.
        unsafe { std::ptr::write_bytes(address.cast::<u8>(), 1, bytes as usize) };
        let maps = std::fs::read_to_string("/proc/self/numa_maps").unwrap();
        let line = maps
            .lines()
            .find(|line| line.starts_with(&format!("{:x} ", address as usize)))
            .unwrap()
            .to_owned();
        // SAFETY: the mapping made above.
        unsafe { munmap(address, bytes as usize) }.unwrap();
        assert!(line.contains(&format!("prefer:{node}")), "{line}");
        assert!(line.contains(&format!("N{node}=")), "{line}");
    }
}
