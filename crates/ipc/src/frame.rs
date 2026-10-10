//! Frames: the unit a rank channel carries.
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
//! (`skip_serializing_if`) anywhere but last in its struct. The round-trip
//! tests of this crate carry every message family through a frame.
//!
//! A [`Message`] is validated on both sides of a frame: [`Frame::encode`]
//! refuses a message that violates the protocol contract before any byte is
//! sent, and [`Frame::decode`] refuses one after decoding it, so a receiver
//! never acts on an unchecked message.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::ValidationResult;
use crate::iceoryx::{IpcError, IpcResult};

/// A type that crosses a rank channel as the one message of a [`Frame`].
pub trait Message: Serialize + DeserializeOwned {
    /// Checks the protocol invariants this message carries.
    fn validate(&self) -> ValidationResult<()>;
}

/// One message as a rank channel carries it: a length, then the encoding.
pub struct Frame {
    /// The length field followed by the encoded message it counts.
    bytes: Vec<u8>,
}

impl Frame {
    /// Bytes of the length field that starts every frame.
    pub const LENGTH_BYTES: usize = 4;

    /// Validates `message` and encodes it as one frame.
    ///
    /// `bound` is the largest encoded message, in bytes, the channel carries;
    /// the sender refuses a longer one so that it fails here rather than at
    /// the receiver. Fails with [`IpcError::Validation`] when the message
    /// violates the protocol contract, and with [`IpcError::Encoding`] when it
    /// cannot be encoded or exceeds `bound`.
    pub fn encode<M: Message>(message: &M, bound: usize) -> IpcResult<Self> {
        message.validate()?;

        // Encode behind a placeholder length, which is known only afterwards.
        let mut bytes = vec![0_u8; Self::LENGTH_BYTES];
        rmp_serde::encode::write(&mut bytes, message)
            .map_err(|error| IpcError::Encoding(format!("encoding a message: {error}")))?;

        let length = bytes.len() - Self::LENGTH_BYTES;
        let field = checked_length(length, bound)?;
        bytes[..Self::LENGTH_BYTES].copy_from_slice(&field.to_le_bytes());
        Ok(Self { bytes })
    }

    /// Decodes this frame's message and validates it.
    ///
    /// Fails with [`IpcError::Encoding`] when the bytes are not an encoding of
    /// `M`, for example a truncated message or one of another type, and with
    /// [`IpcError::Validation`] when the decoded message violates the protocol
    /// contract.
    pub fn decode<M: Message>(&self) -> IpcResult<M> {
        let message: M = rmp_serde::from_slice(&self.bytes[Self::LENGTH_BYTES..])
            .map_err(|error| IpcError::Encoding(format!("decoding a message: {error}")))?;
        message.validate()?;
        Ok(message)
    }

    /// Reads a length field: the bytes of encoded message that follow it.
    ///
    /// A stream receiver calls this as soon as the field has arrived, so a
    /// length above `bound` fails before the bytes it announces are buffered.
    pub fn message_len(field: [u8; Self::LENGTH_BYTES], bound: usize) -> IpcResult<usize> {
        let length = u32::from_le_bytes(field) as usize;
        checked_length(length, bound)?;
        Ok(length)
    }

    /// Takes one complete frame as it was received.
    ///
    /// Fails when `bytes` is shorter than a length field, when the length it
    /// states exceeds `bound`, and when it differs from the bytes that follow,
    /// which is how a truncated or overlong frame is refused.
    pub fn from_bytes(bytes: Vec<u8>, bound: usize) -> IpcResult<Self> {
        let Some(field) = bytes.first_chunk::<{ Self::LENGTH_BYTES }>() else {
            return Err(IpcError::Encoding(format!(
                "a frame of {} bytes has no length field",
                bytes.len()
            )));
        };
        let length = Self::message_len(*field, bound)?;

        let carried = bytes.len() - Self::LENGTH_BYTES;
        if carried != length {
            return Err(IpcError::Encoding(format!(
                "a frame states a message of {length} bytes and carries {carried}"
            )));
        }
        Ok(Self { bytes })
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

/// Returns `length` as a length field after checking it against `bound`.
///
/// The field is a `u32`, so a longer message has no frame whatever the bound.
fn checked_length(length: usize, bound: usize) -> IpcResult<u32> {
    if length > bound {
        return Err(IpcError::Encoding(format!(
            "a message of {length} bytes exceeds the channel's {bound} byte bound"
        )));
    }
    u32::try_from(length).map_err(|_| {
        IpcError::Encoding(format!(
            "a message of {length} bytes exceeds the largest frame of {} bytes",
            u32::MAX
        ))
    })
}
