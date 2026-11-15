//! Media bytes between the processes of a node and its peers: anonymous memory files
//! ([`crate::shared_memory::anonymous`]) mapped read-only ([`MappedMedia`]).
//!
//! A medium has no name. Over a Unix socket it crosses as a descriptor beside the frame that
//! states it; over a connection to another host it crosses as bounded parts ahead of that frame
//! (`MediaPart`, at most [`PART_BYTES`] each), which the receiver writes into a memory file of its
//! own ([`MediaReceiver`]). The kernel frees a medium once its last descriptor and mapping are
//! gone, so nothing outlives a process that ends, however it ends.
//!
//! A video request's condition media travel from the daemon to the engine beside the request's
//! `Submit`, one per condition in request order, and the engine's media reader decodes them; a
//! request's generated artifact travels from the engine to the daemon beside its event.

use std::io::{self, Write as _};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};

use crate::request::ArtifactMedia;

/// The most bytes of a medium one part carries to a process on another host: a part is written
/// whole before the next is encoded, so it bounds what either side holds of a medium beside its
/// memory file, and the time one frame takes on the connection while other frames wait.
pub const PART_BYTES: usize = 1 << 20;

// A part is a frame well inside the longest message a connection takes.
const _: () = assert!(PART_BYTES < crate::MESSAGE_BYTES);

/// A medium arriving in parts ([`PART_BYTES`]). Bytes go directly into an anonymous memory file,
/// without keeping a second complete payload in a `Vec`, and the finished medium is mapped from
/// it. Dropping an incomplete transfer releases its storage.
pub struct MediaReceiver {
    file: std::fs::File,
    size: u64,
    written: u64,
}

impl MediaReceiver {
    /// Reserve the medium's declared length before accepting any parts.
    pub fn new(size: u64) -> io::Result<Self> {
        let length = usize::try_from(size).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "a medium's size exceeds the address space",
            )
        })?;
        let file = crate::shared_memory::anonymous(length)?;
        Ok(Self {
            file,
            size,
            written: 0,
        })
    }

    /// Append one part, preserving the declared length across all parts.
    pub fn append(&mut self, size: u64, data: &[u8]) -> io::Result<()> {
        if size != self.size || data.len() as u64 > self.size - self.written {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a medium's parts exceed or change their declared length",
            ));
        }
        self.file.write_all(data)?;
        self.written += data.len() as u64;
        Ok(())
    }

    /// Whether every byte of the declared length arrived.
    pub fn complete(&self) -> bool {
        self.written == self.size
    }

    /// Seal the complete medium and publish it as an immutable mapping. A truncated transfer or
    /// a frame declaring another size is an error.
    pub fn finish(self, size: u64) -> io::Result<MappedMedia> {
        if size != self.size || self.written != self.size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a medium's transfer is incomplete or its frame changes the length",
            ));
        }
        crate::shared_memory::seal(&self.file)?;
        MappedMedia::map(self.file.into(), size)
    }
}

/// The first `bytes` bytes of an anonymous memory file, mapped read-only for as long as this
/// value lives: a medium this process copied ([`MappedMedia::copy`]), received in parts, or was
/// passed by descriptor.
///
/// Every medium is sealed against writes and size changes before it is mapped, and a mapping
/// refuses a file that is not: no process can change the bytes a mapping reads, whoever passed
/// the descriptor.
#[derive(Debug)]
pub struct MappedMedia {
    address: *mut libc::c_void,
    bytes: usize,
    descriptor: OwnedFd,
}

// SAFETY: the pointer refers to a `PROT_READ` mapping of a file sealed against writes and size
// changes (`MappedMedia::map` refuses another), which stays mapped until this value drops, so
// moving it across threads cannot race.
unsafe impl Send for MappedMedia {}
// SAFETY: as for Send above; shared references only read the immutable mapping.
unsafe impl Sync for MappedMedia {}

impl MappedMedia {
    /// Copies `bytes` into a new anonymous memory file, seals it and maps it: a medium this
    /// process produced, to read or to hand to another process.
    ///
    /// # Errors
    ///
    /// Fails when `bytes` is empty, or when the kernel refuses the file, its storage, its seals
    /// or the mapping; no file is left behind.
    pub fn copy(bytes: &[u8]) -> io::Result<Self> {
        let mut file = crate::shared_memory::anonymous(bytes.len())?;
        file.write_all(bytes)?;
        crate::shared_memory::seal(&file)?;
        Self::map(file.into(), bytes.len() as u64)
    }

    /// Maps the whole of the sealed memory file `descriptor` names, which another process passed
    /// beside a frame: the file's length is the medium's.
    ///
    /// # Errors
    ///
    /// As [`MappedMedia::map`].
    pub fn whole(descriptor: OwnedFd) -> io::Result<Self> {
        let length = file_length(&descriptor)?;
        Self::map(descriptor, length)
    }

    /// Maps the first `bytes` bytes of the sealed memory file `descriptor` names.
    ///
    /// # Errors
    ///
    /// Fails when `bytes` is zero, the file is shorter or not sealed against writes and size
    /// changes, or the mapping fails.
    pub fn map(descriptor: OwnedFd, bytes: u64) -> io::Result<Self> {
        let invalid =
            |message: &str| io::Error::new(io::ErrorKind::InvalidData, message.to_owned());
        let bytes = usize::try_from(bytes).map_err(|_| invalid("the medium is too large"))?;
        if bytes == 0 {
            return Err(invalid("a medium is never empty"));
        }
        if (file_length(&descriptor)?) < bytes as u64 {
            return Err(invalid("the medium's file is shorter than its bytes"));
        }
        if !crate::shared_memory::sealed(&descriptor)? {
            return Err(invalid(
                "the medium's file is not sealed against writes and size changes",
            ));
        }

        // SAFETY: the descriptor names a readable file of at least `bytes` bytes; the mapping is
        // read-only and private to this value.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ,
                libc::MAP_SHARED,
                descriptor.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            address,
            bytes,
            descriptor,
        })
    }
}

/// The length of the file `descriptor` names.
fn file_length(descriptor: &OwnedFd) -> io::Result<u64> {
    // SAFETY: an all-zero `stat` is a valid value for the call to fill.
    let mut status: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: the descriptor is open and `status` a valid out-pointer.
    if unsafe { libc::fstat(descriptor.as_raw_fd(), &mut status) } != 0 {
        return Err(io::Error::last_os_error());
    }
    u64::try_from(status.st_size)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "a file of negative length"))
}

impl ArtifactMedia for MappedMedia {
    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping covers `bytes` readable bytes until drop, and its file is sealed
        // against writes.
        unsafe { std::slice::from_raw_parts(self.address.cast::<u8>(), self.bytes) }
    }

    fn descriptor(&self) -> Option<BorrowedFd<'_>> {
        Some(self.descriptor.as_fd())
    }
}

/// Two media are equal when they hold the same bytes.
impl PartialEq for MappedMedia {
    fn eq(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
}

impl Eq for MappedMedia {}

impl Drop for MappedMedia {
    fn drop(&mut self) {
        // SAFETY: the address and length are this value's own mapping, which nothing borrows
        // past its lifetime.
        unsafe {
            libc::munmap(self.address, self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A copied medium reads back its bytes, and another process's mapping of its descriptor,
    /// which a frame carried, reads the same.
    #[test]
    fn a_copied_medium_maps_whole_from_its_descriptor() {
        let medium = MappedMedia::copy(b"condition bytes").unwrap();
        assert_eq!(medium.bytes(), b"condition bytes");

        let passed = medium.descriptor().unwrap().try_clone_to_owned().unwrap();
        let received = MappedMedia::whole(passed).unwrap();
        assert_eq!(received, medium);
    }

    /// A file anyone could still write is refused: a mapping reads bytes no process changes.
    #[test]
    fn an_unsealed_file_is_refused() {
        let mut file = crate::shared_memory::anonymous(5).unwrap();
        file.write_all(b"bytes").unwrap();
        assert!(MappedMedia::whole(file.into()).is_err());
    }

    /// A medium's parts are written into a file of the receiver's own and mapped once complete.
    #[test]
    fn a_medium_arrives_in_parts() {
        let mut receiver = MediaReceiver::new(10).unwrap();
        receiver.append(10, b"first").unwrap();
        assert!(!receiver.complete());
        receiver.append(10, b"after").unwrap();
        assert!(receiver.complete());
        assert!(receiver.append(10, b"x").is_err());

        let medium = receiver.finish(10).unwrap();
        assert_eq!(medium.bytes(), b"firstafter");
    }

    #[test]
    fn empty_media_are_refused() {
        assert!(MappedMedia::copy(b"").is_err());
        assert!(MediaReceiver::new(0).is_err());
    }
}
