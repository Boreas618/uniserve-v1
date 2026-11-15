//! A main process's introduction to a peer main process.
//!
//! The main processes of one epoch are one build: the daemon starts them from
//! the same build config, so their peer channel carries Rust messages and no
//! cross-build compatibility is needed for it. The scheduler node's main process
//! reaches every rank's main process, and an expert-parallel group's schedulers
//! agree on the step they switch generations at (§7.3 step 3).
//!
//! A peer connects to the address its launch names and sends this frame first,
//! so the accepting side knows which rank the connection serves before it reads
//! commands. A registration that names another build is refused: the message
//! types are the only definition of the protocol and their encoding is
//! positional, so the two ends of a connection must be the same build, and a
//! message type reordered or retyped between two builds would fail to decode
//! or, worse, decode into other fields. The build is `proto::BUILD`: the
//! package version and a digest of the workspace sources, which tells apart two
//! builds of one package version.

use serde::{Deserialize, Serialize};

/// The first frame a main process sends a peer of its epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct PeerRegistration {
    /// The worker rank this main process serves within the epoch. The
    /// scheduler node's fan-out and an expert group's agreement address ranks
    /// by it.
    pub rank: u32,
    /// The deployment's engine number, which the daemon numbers each epoch
    /// with: a channel whose number differs belongs to another epoch and is
    /// refused.
    pub engine: u64,
    /// The build the peer runs (`proto::BUILD`).
    pub version: String,
}

impl PeerRegistration {
    /// Whether this registration belongs to `engine` and was built as this
    /// process was.
    pub(crate) fn accepted_by(&self, engine: u64) -> bool {
        self.engine == engine && self.version == proto::BUILD
    }
}

/// The response to a [`PeerRegistration`]: the accepting main process keeps the
/// connection under this rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct PeerAccepted {
    /// The rank the accepting side keeps the connection under.
    pub rank: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(engine: u64, version: &str) -> PeerRegistration {
        PeerRegistration {
            rank: 3,
            engine,
            version: version.to_owned(),
        }
    }

    #[test]
    fn a_registration_of_this_build_and_epoch_is_accepted() {
        assert!(registration(7, proto::BUILD).accepted_by(7));
    }

    #[test]
    fn another_epoch_or_another_build_is_refused() {
        assert!(!registration(8, proto::BUILD).accepted_by(7));
        assert!(!registration(7, "2.0.0+0").accepted_by(7));
    }

    #[test]
    fn a_registration_survives_the_peer_encoding() {
        let registration = registration(7, proto::BUILD);
        let encoded = rmp_serde::to_vec(&registration).expect("a registration encodes");
        let decoded: PeerRegistration = rmp_serde::from_slice(&encoded).expect("and decodes");
        assert_eq!(decoded, registration);
    }
}
