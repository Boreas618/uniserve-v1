//! Facts of the host the process runs on, as opposed to its devices.

use std::io;
use std::path::Path;

/// The host memory available to new work of this process without swapping, in
/// bytes: the kernel's `MemAvailable` estimate (`/proc/meminfo`), bounded by
/// what the process's cgroup (v2) still lets it allocate when the cgroup has a
/// memory limit.
///
/// # Errors
///
/// Fails when `/proc/meminfo` cannot be read or states no `MemAvailable`: a
/// caller that gates work on host memory refuses the work rather than run it
/// ungated.
pub fn available_memory() -> io::Result<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let available = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.trim().strip_suffix("kB"))
        .and_then(|kib| kib.trim().parse::<u64>().ok())
        .map(|kib| kib * 1024)
        .ok_or_else(|| io::Error::other("/proc/meminfo states no MemAvailable"))?;
    Ok(cgroup_room().map_or(available, |room| available.min(room)))
}

/// What the process's cgroup (v2) lets it allocate beyond what the cgroup
/// holds: its limit less its usage; `None` without a limit, or without a
/// unified hierarchy to read one from.
fn cgroup_room() -> Option<u64> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim();
    let directory = Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let read = |name: &str| std::fs::read_to_string(directory.join(name)).ok();
    // `memory.max` is `max` without a limit, which does not parse as a number.
    let limit = read("memory.max")?.trim().parse::<u64>().ok()?;
    let usage = read("memory.current")?.trim().parse::<u64>().ok()?;
    Some(limit.saturating_sub(usage))
}

/// The NUMA nodes of the host, by number, as a set of nodes 0 to 1023.
pub type NumaNodes = [u64; 16];

/// `get_mempolicy(2)`'s flags (`linux/mempolicy.h`): ask for the node of a page, of the policy
/// of an address, or of the nodes the process may allocate on.
const MPOL_F_NODE: libc::c_ulong = 1;
const MPOL_F_ADDR: libc::c_ulong = 1 << 1;
const MPOL_F_MEMS_ALLOWED: libc::c_ulong = 1 << 2;

/// `get_mempolicy(2)` with `flags` at `address`: the mode word and the set of nodes.
fn mempolicy(address: usize, flags: libc::c_ulong) -> io::Result<(libc::c_int, NumaNodes)> {
    let mut mode: libc::c_int = 0;
    let mut nodes: NumaNodes = [0; 16];
    let bits = libc::c_ulong::try_from(nodes.len() * 64).map_err(io::Error::other)?;
    // SAFETY: the kernel writes the mode word and at most `bits` bits of the node set, the sizes
    // passed; it reads nothing at `address`.
    let result = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &raw mut mode,
            nodes.as_mut_ptr(),
            bits,
            address,
            flags,
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((mode, nodes))
}

/// The NUMA nodes the process may allocate memory on.
///
/// # Errors
///
/// Fails when the kernel refuses the question (a kernel without NUMA support).
pub fn allowed_numa_nodes() -> io::Result<NumaNodes> {
    Ok(mempolicy(0, MPOL_F_MEMS_ALLOWED)?.1)
}

/// The memory policy of the mapped page at `address`: its mode (`libc::MPOL_*`) and its nodes.
///
/// # Errors
///
/// Fails when `address` is not mapped.
pub fn page_policy(address: usize) -> io::Result<(libc::c_int, NumaNodes)> {
    mempolicy(address, MPOL_F_ADDR)
}

/// The NUMA node the page at `address` lies on; the page is allocated when it is not yet.
///
/// # Errors
///
/// Fails when `address` is not mapped.
pub fn page_node(address: usize) -> io::Result<u32> {
    let (node, _) = mempolicy(address, MPOL_F_ADDR | MPOL_F_NODE)?;
    u32::try_from(node).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Linux host states its available memory, and it is never zero for a running process.
    #[test]
    fn the_host_states_its_available_memory() {
        assert!(available_memory().unwrap() > 0);
    }
}
