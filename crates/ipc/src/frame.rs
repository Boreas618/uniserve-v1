//! Frames: the unit a channel carries.
//!
//! A frame is the length of one encoded message, as a little-endian `u32`,
//! followed by that message in MessagePack (`rmp-serde`). The message types of
//! this crate are the only definition of the protocol: the encoding is the one
//! their `serde` derives describe, and both ends are built from this
//! workspace, so no schema, generated code or version field accompanies it.
//!
//! The encoding is positional: a struct is the array of its fields in
//! declaration order, without their names. Every message type must therefore
//! decode from a sequence, which rules out three `serde` forms in anything a
//! message reaches: a flattened member, a struct variant of an adjacently
//! tagged or untagged enum, and a field omitted when absent
//! (`skip_serializing_if`) anywhere but last in its struct. A message type
//! also gives none of its fields a default: a shorter array, as another
//! build would send, must fail to decode rather than be completed. The
//! round-trip tests of this crate carry every message family through a
//! frame.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{IpcError, IpcResult};

/// One message as a channel carries it: a length, then the encoding.
pub struct Frame {
    /// The length field followed by the encoded message it counts.
    bytes: Vec<u8>,
}

impl Frame {
    /// Bytes of the length field that starts every frame.
    pub const LENGTH_BYTES: usize = 4;

    /// Encodes `message` as one frame.
    ///
    /// Fails when the message cannot be encoded or is longer than the length
    /// field can state.
    pub fn encode<M: Serialize>(message: &M) -> IpcResult<Self> {
        // Encode behind a placeholder length, which is known only afterwards.
        let mut bytes = vec![0_u8; Self::LENGTH_BYTES];
        rmp_serde::encode::write(&mut bytes, message)
            .map_err(|error| IpcError::Encoding(format!("encoding a message: {error}")))?;

        let length = bytes.len() - Self::LENGTH_BYTES;
        let field = u32::try_from(length).map_err(|_| {
            IpcError::Encoding(format!(
                "a message of {length} bytes exceeds the largest frame of {} bytes",
                u32::MAX
            ))
        })?;
        bytes[..Self::LENGTH_BYTES].copy_from_slice(&field.to_le_bytes());
        Ok(Self { bytes })
    }

    /// Decodes this frame's message.
    ///
    /// Fails when the bytes are not an encoding of `M`, for example a message
    /// of another type or of another build.
    pub fn decode<M: DeserializeOwned>(&self) -> IpcResult<M> {
        rmp_serde::from_slice(&self.bytes[Self::LENGTH_BYTES..])
            .map_err(|error| IpcError::Encoding(format!("decoding a message: {error}")))
    }

    /// Reads a length field: the bytes of encoded message that follow it.
    pub fn message_len(field: [u8; Self::LENGTH_BYTES]) -> usize {
        u32::from_le_bytes(field) as usize
    }

    /// Takes one complete frame as it was received: a length field and the
    /// bytes it counts.
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Returns the frame as a transport sends it: length field, then message.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the frame's bytes, as [`Self::as_bytes`], without copying them.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
