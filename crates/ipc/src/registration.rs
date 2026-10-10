//! A rank's report of the channel endpoint the head connects to.

use serde::{Deserialize, Serialize};

/// The mechanism name of a channel over a Unix socket, served by a rank on
/// the head's host.
///
/// The name travels as a string in the launch descriptor's
/// `channel_transport` and in the registration report's `transport`.
pub const UNIX_CHANNEL: &str = "unix";
/// The mechanism name of a channel over a TCP stream, served by a rank off
/// the head's host.
pub const TCP_CHANNEL: &str = "tcp";

/// One rank's report of the channel endpoint the engine connects to.
///
/// The rank sends one JSON line after binding its endpoint and before loading
/// model weights. The head uses it to connect the rank's channel.
#[derive(Debug, Serialize, Deserialize)]
pub struct RankReport {
    /// Worker identity the rank was launched under.
    pub worker_id: String,
    /// Global rank within that worker.
    pub rank: u32,
    /// Channel mechanism the endpoint names.
    pub transport: String,
    /// Address the engine connects this rank's channel to.
    pub endpoint: String,
}
