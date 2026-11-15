//! The peer link between the engine processes of an epoch: the codec its frames take on the
//! workspace's one connection (`service::Connection`).
//!
//! Each [`ToPeer`], [`FromPeer`], [`PeerRegistration`] and [`PeerAccepted`] is one frame
//! (`service::frame`) whose payload is the message in MessagePack (`rmp-serde`). The message types
//! are the only definition of the protocol: the encoding is the one their `serde` derives
//! describe, and both ends are built from this workspace (a registration states the build,
//! `proto::BUILD`), so no schema, generated code or version field accompanies it, and no bound
//! beyond the frame header's limits a message.
//!
//! The encoding is positional: a struct is the array of its fields in declaration order, without
//! their names. Every message type must therefore decode from a sequence, which rules out three
//! `serde` forms in anything a message reaches: a flattened member, a struct variant of an
//! adjacently tagged or untagged enum, and a field omitted when absent (`skip_serializing_if`)
//! anywhere but last in its struct. A message type also gives none of its fields a default: a
//! shorter array, as another build would send, must fail to decode rather than be completed. The
//! round-trip tests of this crate carry every message family through its encoding.

use std::io;

use super::messages::{FromPeer, ToPeer};
use super::registration::{PeerAccepted, PeerRegistration};

/// Implements the workspace's connection codec for each message of the peer link.
macro_rules! peer_messages {
    ($($message:ty),* $(,)?) => {
        $(
            impl service::Message for $message {

                fn encode(&self) -> io::Result<Vec<u8>> {
                    rmp_serde::to_vec(self).map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("encoding a peer message: {error}"),
                        )
                    })
                }

                fn decode(payload: &[u8]) -> io::Result<Self> {
                    rmp_serde::from_slice(payload).map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("decoding a peer message: {error}"),
                        )
                    })
                }
            }
        )*
    };
}

peer_messages!(ToPeer, FromPeer, PeerRegistration, PeerAccepted);

#[cfg(test)]
mod tests;
