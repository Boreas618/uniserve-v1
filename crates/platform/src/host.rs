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

#[cfg(test)]
mod tests {
    use super::*;

    /// A Linux host states its available memory, and it is never zero for a running process.
    #[test]
    fn the_host_states_its_available_memory() {
        assert!(available_memory().unwrap() > 0);
    }
}
