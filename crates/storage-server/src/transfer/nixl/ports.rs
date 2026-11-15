//! The host's RDMA ports UCX can open, as the kernel states them.
//!
//! UCX fails to start when it cannot open a device it found, such as an Ethernet port that holds
//! no address, so the agent's backend names the ports it may use (UCX's `NET_DEVICES`). A port is
//! usable when its link is active and its GID table holds an address: an InfiniBand port's first
//! GID carries its port GUID, a RoCE port's GIDs carry its interface's IP addresses, and a port
//! whose every GID has a zero interface identifier holds none. The kernel states both in the
//! InfiniBand class's sysfs attributes (`Documentation/ABI/stable/sysfs-class-infiniband`), the
//! interface `libibverbs` reads them through.

use std::io;
use std::path::Path;

/// Where the kernel states the InfiniBand class's devices.
pub(crate) const CLASS: &str = "/sys/class/infiniband";

/// The state attribute's value of a port whose link is active (`IB_PORT_ACTIVE`).
const ACTIVE: &str = "4: ACTIVE";

/// The usable ports of the devices under `class`, as UCX names them (`mlx5_0:1`), sorted.
///
/// # Errors
///
/// Fails when the class's directory or a port's attribute cannot be read; a host without the
/// class's directory has no RDMA device and fails with `NotFound`.
pub(crate) fn usable(class: &Path) -> io::Result<Vec<String>> {
    let mut usable = Vec::new();
    for device in std::fs::read_dir(class)? {
        let device = device?;
        for port in std::fs::read_dir(device.path().join("ports"))? {
            let port = port?;
            let path = port.path();
            let state = std::fs::read_to_string(path.join("state"))?;
            if state.trim() != ACTIVE || !holds_address(&path.join("gids"))? {
                continue;
            }
            usable.push(format!(
                "{}:{}",
                device.file_name().to_string_lossy(),
                port.file_name().to_string_lossy()
            ));
        }
    }
    usable.sort();
    Ok(usable)
}

/// Whether a port's GID table (`gids`, one file per index, each a GID as eight groups of four
/// hexadecimal digits) holds a GID whose interface identifier, its last 64 bits, is not zero.
fn holds_address(gids: &Path) -> io::Result<bool> {
    for gid in std::fs::read_dir(gids)? {
        let text = std::fs::read_to_string(gid?.path())?;
        // The interface identifier is the last four groups.
        let identifier = text.trim().split(':').skip(4);
        if identifier
            .map(|group| u16::from_str_radix(group, 16))
            .any(|group| group.is_ok_and(|group| group != 0))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One port of `device` under `class`, in `state`, with `gids`.
    fn port(class: &Path, device: &str, number: u32, state: &str, gids: &[&str]) {
        let path = class.join(device).join("ports").join(number.to_string());
        std::fs::create_dir_all(path.join("gids")).unwrap();
        std::fs::write(path.join("state"), format!("{state}\n")).unwrap();
        for (index, gid) in gids.iter().enumerate() {
            std::fs::write(
                path.join("gids").join(index.to_string()),
                format!("{gid}\n"),
            )
            .unwrap();
        }
    }

    const ZERO: &str = "0000:0000:0000:0000:0000:0000:0000:0000";

    /// Active ports that hold an address are usable: an InfiniBand port by its GUID, a RoCE port
    /// by an IP-derived GID past the first index; an active Ethernet port without an address and
    /// a port whose link is down are not, even with a GID prefix.
    #[test]
    fn active_ports_that_hold_an_address_are_usable() {
        let class = tempfile::tempdir().unwrap();
        let root = class.path();
        port(
            root,
            "mlx5_0",
            1,
            ACTIVE,
            &["fe80:0000:0000:0000:5000:e603:0049:dc5c"],
        );
        port(root, "mlx5_1", 1, ACTIVE, &[ZERO, ZERO]);
        port(
            root,
            "mlx5_2",
            1,
            ACTIVE,
            &[ZERO, "0000:0000:0000:0000:0000:ffff:0a00:0002"],
        );
        port(
            root,
            "mlx5_3",
            1,
            "1: DOWN",
            &["fe80:0000:0000:0000:0000:0000:0000:0000"],
        );
        port(
            root,
            "mlx5_4",
            1,
            "1: DOWN",
            &["fe80:0000:0000:0000:5000:e603:0049:dc60"],
        );

        assert_eq!(usable(root).unwrap(), ["mlx5_0:1", "mlx5_2:1"]);
    }

    /// A host without the class's directory has no RDMA device.
    #[test]
    fn a_host_without_the_class_has_no_rdma_device() {
        let directory = tempfile::tempdir().unwrap();
        let error = usable(&directory.path().join("infiniband")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }
}
