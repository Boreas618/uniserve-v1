//! A frame carries a descriptor to the other end of a Unix connection, which
//! maps the bytes it names.

use std::io::Write as _;
use std::os::fd::{AsFd as _, OwnedFd};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use proto::request::{ArtifactMedia as _, wire};
use proto::{Connection, MappedMedia};

/// An anonymous shared-memory file holding `bytes`.
fn storage(bytes: &[u8]) -> OwnedFd {
    let name = c"artifact";
    // SAFETY: memfd_create takes a NUL-terminated name and flags.
    let descriptor = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    assert!(descriptor >= 0);
    // SAFETY: memfd_create returned this descriptor for this test to own.
    let mut file = std::fs::File::from(unsafe {
        <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(descriptor)
    });
    file.write_all(bytes).unwrap();
    file.into()
}

#[test]
fn a_frame_carries_its_descriptor_and_the_receiver_maps_the_bytes() {
    let (left, right) = UnixStream::pair().unwrap();
    let mut sender = Connection::from_unix(left).unwrap();
    let mut receiver = Connection::from_unix(right).unwrap();

    // Two frames, the first with a descriptor and the second without: each
    // descriptor comes with its own frame, in order.
    let bytes = b"generated video bytes";
    let shared = storage(bytes);
    let announce = |content_type: &str| wire::Artifact {
        media_kind: wire::MediaKind::Video as i32,
        content_type: content_type.to_owned(),
        bytes: bytes.len() as u64,
    };
    sender
        .send_with_descriptor(&announce("video/mp4"), shared.as_fd())
        .unwrap();
    drop(shared);
    sender.send(&announce("none")).unwrap();

    let first: wire::Artifact = receiver.recv(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(first.content_type, "video/mp4");
    let descriptor = receiver.take_descriptor().expect("the frame's descriptor");
    let mapped = MappedMedia::map(descriptor, first.bytes).unwrap();
    assert_eq!(mapped.bytes(), bytes);

    let second: wire::Artifact = receiver.recv(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(second.content_type, "none");
    assert!(receiver.take_descriptor().is_none());

    // A mapping is never longer than the object it maps.
    assert!(MappedMedia::map(storage(b"short"), 64).is_err());
}
