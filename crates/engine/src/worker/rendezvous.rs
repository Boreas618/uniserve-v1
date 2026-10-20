//! The address a worker group's ranks rendezvous at.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};

use anyhow::Context;

/// A worker group's collective rendezvous.
pub(crate) struct Rendezvous {
    /// The store address every rank connects to, as `host:port`.
    pub address: String,
    /// The bound, listening socket the group's first rank serves the store
    /// on, held by the process that spawns that rank until it hands the
    /// socket over. `None` when the first rank runs on another host, whose
    /// launcher holds that host's reservation, and once the socket has been
    /// taken for the spawn.
    pub listener: Option<TcpListener>,
    /// The descriptor number the group's first rank inherits the socket at
    /// when a launcher on another host holds it, as that launcher reported.
    pub listen_fd: Option<std::os::fd::RawFd>,
}

/// Reserves the address a worker group's ranks rendezvous at.
///
/// The collective store is a TCP store so ranks on different hosts can reach
/// it; a group on one host reserves a loopback address, and one spanning
/// hosts reserves an address those hosts can route to (see `interface`).
/// The head calls this when the group's first rank runs on its own host: it
/// binds the socket here, and that rank, which it spawns, inherits the socket
/// and serves the store on it. The port therefore stays bound from this
/// reservation until that rank exits, so no other process on the host can
/// take it between the reservation and the rank's start.
pub(crate) fn reserve_rendezvous(head: Option<IpAddr>) -> anyhow::Result<Rendezvous> {
    let (bind, advertise) = interface(head);
    let listener = TcpListener::bind(SocketAddr::from((bind, 0)))
        .context("reserving the collective rendezvous address")?;
    let port = listener
        .local_addr()
        .context("reading the collective rendezvous address")?
        .port();
    Ok(Rendezvous {
        address: format!("{advertise}:{port}"),
        listener: Some(listener),
        listen_fd: None,
    })
}

/// Returns the interface to bind and the host a descriptor names.
///
/// `head` is an address of this host that the other hosts route to, and none
/// when every rank runs here. A loopback address reaches only the host that
/// binds it, so a group spanning hosts binds every interface and names
/// itself by that routable address rather than by the wildcard, which names
/// no interface, or by the placement's host identity, which need not resolve
/// on other hosts.
fn interface(head: Option<IpAddr>) -> (Ipv4Addr, String) {
    match head {
        Some(address) => (Ipv4Addr::UNSPECIFIED, address.to_string()),
        None => (Ipv4Addr::LOCALHOST, Ipv4Addr::LOCALHOST.to_string()),
    }
}
