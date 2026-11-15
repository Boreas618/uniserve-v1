//! The MessagePack codec of the protocols whose messages are serde types: the Storage Server's
//! protocol, a hot restart's handoff between two daemons and a Storage Server's takeover. A
//! message is encoded with its fields named, so the payload names what it carries, and travels in
//! the workspace's one frame layout ([`crate::frame`]) like every other protocol's.

use std::io;

use serde::Serialize;
use serde::de::DeserializeOwned;

/// `message`'s payload: its MessagePack encoding with named fields.
///
/// # Errors
///
/// Fails with `InvalidInput` when the message has no MessagePack form.
pub fn encode<T: Serialize>(message: &T) -> io::Result<Vec<u8>> {
    rmp_serde::to_vec_named(message)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))
}

/// The message `payload` encodes ([`encode`]).
///
/// # Errors
///
/// Fails with `InvalidData` when the payload is no encoding of a `T`.
pub fn decode<T: DeserializeOwned>(payload: &[u8]) -> io::Result<T> {
    rmp_serde::from_slice(payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Serialize, serde::Deserialize)]
    struct Message {
        id: u64,
        name: String,
    }

    /// A message reads back as itself, and a payload of another shape is refused as invalid
    /// data.
    #[test]
    fn a_message_round_trips_and_another_shape_is_refused() {
        let message = Message {
            id: 7,
            name: "seven".to_owned(),
        };
        let payload = encode(&message).unwrap();
        assert_eq!(decode::<Message>(&payload).unwrap(), message);
        let refused = decode::<Message>(&encode(&7_u64).unwrap()).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::InvalidData);
    }
}
