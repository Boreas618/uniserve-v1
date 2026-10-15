//! The published form of an exportable allocation's shareable handle.
//!
//! A CUDA VMM location carries it as `allocation_handle`, and a peer
//! allocation publishes it to its group. A fabric handle is its 64 bytes
//! and imports on any host of the fabric. A POSIX descriptor is its 4-byte
//! number in the exporting process: it marks the handle as host-local, and a
//! process of the same host obtains its own descriptor from the exporter's
//! grant socket ([`fetch_descriptor`]).

use std::os::fd::AsRawFd;

use uniserve_cuda::ShareableHandle;

use super::grants::fetch_descriptor;
use crate::{Error, Result};

/// Bytes of a published fabric handle.
const FABRIC_BYTES: usize = 64;

/// Bytes of a published descriptor number.
const DESCRIPTOR_BYTES: usize = 4;

/// An exportable allocation's shareable handle as readers receive it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllocationHandle {
    /// A fabric handle, importable on any host of the fabric.
    Fabric([u8; FABRIC_BYTES]),
    /// The exporter's process descriptor, which its grant socket serves
    /// to processes of its host.
    Descriptor(i32),
}

impl AllocationHandle {
    /// The published form of the exporter's `handle`.
    pub(crate) fn of(handle: &ShareableHandle) -> Self {
        match handle {
            ShareableHandle::Fabric(bytes) => Self::Fabric(*bytes),
            ShareableHandle::Descriptor(fd) => Self::Descriptor(fd.as_raw_fd()),
        }
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        match self {
            Self::Fabric(bytes) => bytes.to_vec(),
            Self::Descriptor(fd) => fd.to_ne_bytes().to_vec(),
        }
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        match bytes.len() {
            FABRIC_BYTES => {
                Ok(Self::Fabric(bytes.try_into().map_err(|_| {
                    Error::Invariant("a fabric handle has 64 bytes".into())
                })?))
            }
            DESCRIPTOR_BYTES => Ok(Self::Descriptor(i32::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| Error::Invariant("a descriptor number has 4 bytes".into()))?,
            ))),
            _ => Err(Error::Invalid(
                "an allocation handle is a fabric handle or a descriptor".into(),
            )),
        }
    }

    /// Whether a process on another host can import the allocation.
    pub(crate) fn crosses_hosts(&self) -> bool {
        matches!(self, Self::Fabric(_))
    }

    /// This process's handle of the allocation: the fabric handle, or a
    /// descriptor fetched from the exporter's grant socket `endpoint`,
    /// which serves it under `export`.
    pub(crate) fn import(&self, endpoint: &str, export: &str) -> Result<ShareableHandle> {
        match self {
            Self::Fabric(bytes) => Ok(ShareableHandle::Fabric(*bytes)),
            Self::Descriptor(_) => fetch_descriptor(endpoint, export)
                .map(ShareableHandle::Descriptor)
                .map_err(|error| Error::Transport(error.to_string())),
        }
    }
}
