use std::io::Seek as _;

use super::*;

const TIMEOUT: Duration = Duration::from_secs(5);

/// A message of length-prefixed frames: its text, which must be UTF-8.
#[derive(Clone, Debug, PartialEq)]
struct Line(String);

impl Message for Line {
    const FRAMING: Framing = Framing::Length;

    fn encode(&self) -> io::Result<Vec<u8>> {
        Ok(self.0.clone().into_bytes())
    }

    fn decode(payload: &[u8]) -> io::Result<Self> {
        String::from_utf8(payload.to_vec())
            .map(Self)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))
    }
}

/// A message of descriptor frames: its text.
#[derive(Clone, Debug, PartialEq)]
struct Note(String);

impl Message for Note {
    const FRAMING: Framing = Framing::Descriptors;

    fn encode(&self) -> io::Result<Vec<u8>> {
        Ok(self.0.clone().into_bytes())
    }

    fn decode(payload: &[u8]) -> io::Result<Self> {
        Line::decode(payload).map(|line| Self(line.0))
    }
}

/// `length` bytes of text.
fn line(length: usize) -> Line {
    Line("m".repeat(length))
}

/// The bytes of the frame that carries `message`.
fn frame_of(message: &Line) -> Vec<u8> {
    frame::length_prefixed(&message.encode().unwrap()).unwrap()
}

/// A temporary file holding `text`, as a descriptor to pass.
fn file(text: &str) -> std::fs::File {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(text.as_bytes()).unwrap();
    file.rewind().unwrap();
    file
}

/// What the file a received descriptor opens holds.
fn read(descriptor: OwnedFd) -> String {
    std::io::read_to_string(std::fs::File::from(descriptor)).unwrap()
}

/// Takes the connection that is on its way to `listener`.
fn accept(listener: &Listener) -> Connection {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if let Some(connection) = listener.try_accept().unwrap() {
            return connection;
        }
        assert!(Instant::now() < deadline, "no connection arrived");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A connection and the raw stream of its peer, over loopback TCP.
fn with_raw_peer() -> (Connection, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let connection = Connection::connect(&address).unwrap();
    let (peer, _) = listener.accept().unwrap();
    peer.set_read_timeout(Some(TIMEOUT)).unwrap();
    (connection, peer)
}

/// Waits for the next message, or returns the failure that ends the wait.
fn receive<M: Message>(connection: &mut Connection) -> io::Result<M> {
    connection
        .recv(TIMEOUT)?
        .ok_or_else(|| io::Error::from(ErrorKind::TimedOut))
}

/// Restricts `connection`'s socket send buffer so that a large frame spans several writes.
fn restrict_send_buffer(connection: &Connection) {
    let capacity: libc::c_int = 4096;
    assert_eq!(
        // SAFETY: the descriptor is the connection's open socket, and the option value points to
        // the local `capacity`, whose size is the option length.
        unsafe {
            libc::setsockopt(
                connection.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                (&capacity as *const libc::c_int).cast(),
                std::mem::size_of_val(&capacity) as libc::socklen_t,
            )
        },
        0
    );
}

/// A Unix and a TCP listener in `directory`.
fn listeners(directory: &tempfile::TempDir) -> [Listener; 2] {
    let unix = format!(
        "unix:{}",
        directory.path().join("connection.sock").display()
    );
    [
        Listener::bind(&unix).unwrap(),
        Listener::bind("127.0.0.1:0").unwrap(),
    ]
}

#[test]
fn messages_cross_unix_and_tcp_connections_in_both_directions() {
    let directory = tempfile::tempdir().unwrap();
    for listener in listeners(&directory) {
        let mut near = Connection::connect(listener.address()).unwrap();
        let mut far = accept(&listener);

        // Only a TCP connection has addresses, and each end's local address is the other's peer
        // address.
        let tcp = !listener.address().starts_with(UNIX_PREFIX);
        assert_eq!(near.local_addr().is_some(), tcp);
        assert_eq!(near.local_addr(), far.peer_addr());
        assert_eq!(near.peer_addr(), far.local_addr());

        near.send(&Line("request".into())).unwrap();
        assert_eq!(receive::<Line>(&mut far).unwrap(), Line("request".into()));
        far.send(&line(16)).unwrap();
        assert_eq!(receive::<Line>(&mut near).unwrap(), line(16));
    }
}

/// A Unix listener's path goes with it, but for one handed to another process, which keeps
/// serving at the path.
#[test]
fn a_unix_listener_removes_its_path_unless_handed_over() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("connection.sock");
    let address = format!("unix:{}", path.display());

    let listener = Listener::bind(&address).unwrap();
    assert!(path.exists());
    assert!(Listener::bind(&address).is_err());
    drop(listener);
    assert!(!path.exists());

    let (descriptor, unix, bound) = Listener::bind(&address).unwrap().hand_over();
    assert!(path.exists());
    let resumed = Listener::resume(descriptor, unix, bound).unwrap();
    assert_eq!(resumed.address(), address);
    let mut near = Connection::connect(&address).unwrap();
    let mut far = accept(&resumed);
    near.send(&Line("resumed".into())).unwrap();
    assert_eq!(receive::<Line>(&mut far).unwrap(), Line("resumed".into()));
}

#[test]
fn completed_messages_survive_a_truncated_last_frame() {
    // The peer writes one complete frame a byte at a time, then a frame cut short, then closes.
    // The complete message must still be delivered; only the truncated one fails, and the
    // connection is then closed.
    let (mut connection, mut peer) = with_raw_peer();
    let first = line(5);
    for byte in frame_of(&first) {
        peer.write_all(&[byte]).unwrap();
    }
    let second = frame_of(&line(9));
    peer.write_all(&second[..second.len() - 2]).unwrap();
    peer.shutdown(std::net::Shutdown::Write).unwrap();

    assert_eq!(receive::<Line>(&mut connection).unwrap(), first);
    let error = receive::<Line>(&mut connection).expect_err("truncation must fail");
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
    assert!(error.to_string().contains("incomplete frame"), "{error}");
    assert!(connection.is_closed());
    assert!(connection.send(&line(1)).is_err());
}

#[test]
fn a_frame_decodes_only_as_the_message_it_carries() {
    let (mut connection, mut peer) = with_raw_peer();
    peer.write_all(&frame::length_prefixed(&[0xff, 0xfe]).unwrap())
        .unwrap();
    let error = receive::<Line>(&mut connection).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidData);
}

/// A wake ends a wait that no message would have ended, fires coalesce into the one wait that
/// observes them, and the wake rearms afterwards.
#[test]
fn a_local_wake_interrupts_a_wait_once_and_rearms() {
    let (mut connection, _peer) = with_raw_peer();
    let wake = connection.wake();
    let long = Duration::from_secs(30);
    let short = Duration::from_millis(50);
    let waited = |connection: &mut Connection, timeout| {
        let started = Instant::now();
        connection.wait(timeout).unwrap();
        started.elapsed()
    };

    assert!(waited(&mut connection, short) >= short, "nothing had fired");

    wake.wake();
    wake.wake();
    assert!(
        waited(&mut connection, long) < long,
        "the wake did not end the wait"
    );
    assert!(
        waited(&mut connection, short) >= short,
        "one wait consumed both fires"
    );

    // A wake may fire from another thread, however many waits preceded it.
    for _ in 0..3 {
        let wake = wake.clone();
        std::thread::spawn(move || wake.wake()).join().unwrap();
        assert!(waited(&mut connection, long) < long);
    }
}

/// An owner that cannot take a message right now still blocks in `wait`: a message it has left
/// buffered does not end the wait, and a new one does.
#[test]
fn a_buffered_message_does_not_end_a_wait_and_a_new_one_does() {
    let (mut connection, mut peer) = with_raw_peer();
    let short = Duration::from_millis(100);
    for length in [1, 2] {
        peer.write_all(&frame_of(&line(length))).unwrap();
    }
    assert_eq!(receive::<Line>(&mut connection).unwrap(), line(1));

    // The second message stays buffered, and nothing else happens.
    let started = Instant::now();
    connection.wait(short).unwrap();
    assert!(
        started.elapsed() >= short,
        "a buffered message ended the wait"
    );

    let writer = std::thread::spawn(move || {
        std::thread::sleep(short);
        peer.write_all(&frame_of(&line(3))).unwrap();
        peer
    });
    let started = Instant::now();
    connection.wait(TIMEOUT).unwrap();
    assert!(
        started.elapsed() < TIMEOUT,
        "a new message did not end the wait"
    );
    let _peer = writer.join().unwrap();
    for length in [2, 3] {
        assert_eq!(connection.try_recv::<Line>().unwrap(), Some(line(length)));
    }
}

/// A wait reports the peer's close as an error, after the message that arrived with it, and at
/// once from then on, so a caller that waits on a closed connection neither blocks for its
/// timeout nor spins.
#[test]
fn a_wait_fails_once_the_peer_has_closed() {
    let (mut connection, mut peer) = with_raw_peer();
    peer.write_all(&frame_of(&line(5))).unwrap();
    drop(peer);

    // The last message and the close may be observed by one wait or two.
    let deadline = Instant::now() + TIMEOUT;
    while connection.wait(Duration::from_millis(50)).is_ok() {
        assert!(Instant::now() < deadline, "the close was never reported");
    }
    let started = Instant::now();
    let error = connection.wait(TIMEOUT).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
    assert!(
        started.elapsed() < TIMEOUT,
        "a closed connection blocked a wait"
    );
    assert_eq!(connection.try_recv::<Line>().unwrap(), Some(line(5)));
}

/// A message queued behind a large, partly written one reaches the peer when the owner flushes
/// before it closes the connection.
#[test]
fn a_flush_delivers_queued_messages_before_a_close() {
    let (mut connection, mut peer) = with_raw_peer();
    restrict_send_buffer(&connection);
    let large = line(1 << 20);
    let last = line(4);
    connection.send(&large).unwrap();
    connection.send(&last).unwrap();
    assert!(connection.wants_write());

    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        peer.read_to_end(&mut bytes).unwrap();
        bytes
    });
    connection.flush().unwrap();
    assert!(!connection.wants_write());
    drop(connection);

    let mut expected = frame_of(&large);
    expected.extend(frame_of(&last));
    assert!(
        reader.join().unwrap() == expected,
        "the peer missed a message"
    );
}

/// A frame larger than the socket's send buffer is written across the owner's waits, each of
/// which ends when part of the frame has been written out.
#[test]
fn a_large_message_is_written_across_waits() {
    let (mut connection, mut peer) = with_raw_peer();
    restrict_send_buffer(&connection);
    let large = line(1 << 20);
    connection.send(&large).unwrap();

    let expected = frame_of(&large);
    let length = expected.len();
    let reader = std::thread::spawn(move || {
        let mut bytes = vec![0; length];
        peer.read_exact(&mut bytes).unwrap();
        (bytes, peer)
    });
    let deadline = Instant::now() + TIMEOUT;
    while connection.wants_write() {
        assert!(Instant::now() < deadline, "the frame was not written out");
        connection.wait(TIMEOUT).unwrap();
    }
    let (wire, _peer) = reader.join().unwrap();
    assert!(wire == expected, "the frame arrived changed");
}

/// A message above 64 MiB, the size of one decoded video unit, crosses a TCP connection intact:
/// a connection carries any message its frame's length field can state unless the message bounds
/// it.
#[test]
fn a_message_above_64_mib_crosses_a_tcp_connection_intact() {
    let listener = Listener::bind("127.0.0.1:0").unwrap();
    let mut near = Connection::connect(listener.address()).unwrap();
    let mut far = accept(&listener);
    let large = line(72 << 20);

    // The sender writes the frame out while the receiver reads it, as two engines on two hosts
    // do.
    let sent = large.clone();
    let sender = std::thread::spawn(move || {
        near.send(&sent).unwrap();
        near.flush().unwrap();
        near
    });
    let received: Line = far.recv(Duration::from_secs(60)).unwrap().unwrap();
    let _near = sender.join().unwrap();
    assert!(received == large, "the message arrived changed");
}

/// A frame longer than its message's bound is refused before it is buffered whole.
#[test]
fn a_frame_beyond_its_messages_bound_is_refused() {
    /// A message of at most 8 bytes.
    #[derive(Debug)]
    struct Short;

    impl Message for Short {
        const FRAMING: Framing = Framing::Length;
        const MAX_BYTES: usize = 8;

        fn encode(&self) -> io::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn decode(_: &[u8]) -> io::Result<Self> {
            Ok(Self)
        }
    }

    let (mut connection, mut peer) = with_raw_peer();
    peer.write_all(&frame_of(&line(9))[..6]).unwrap();
    let error = receive::<Short>(&mut connection).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidData);
}

/// A hangup observer sees the peer's close without the connection being called; unread data from
/// the peer does not hide or fake it, and its owner can cancel the wait.
#[test]
fn a_hangup_reports_the_peers_close_and_nothing_else() {
    let directory = tempfile::tempdir().unwrap();
    for listener in listeners(&directory) {
        let connection = Connection::connect(listener.address()).unwrap();
        let mut peer = accept(&listener);
        let hangup = connection.hangup().unwrap();
        let (cancel, canceller) = UnixStream::pair().unwrap();

        peer.send(&line(1)).unwrap();
        let cancelling = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(canceller);
        });
        assert!(
            !hangup.wait(&cancel).unwrap(),
            "unread data read as a close"
        );
        cancelling.join().unwrap();

        let (cancel, _canceller) = UnixStream::pair().unwrap();
        drop(peer);
        assert!(hangup.wait(&cancel).unwrap());
    }
}

/// A descriptor frame hands its receiver exactly its own descriptors, more than one send carries,
/// in order; the frame after it carries none.
#[test]
fn descriptor_frames_carry_their_descriptors_in_order_beyond_one_send() {
    let (left, right) = UnixStream::pair().unwrap();
    let mut sender = Connection::new(left).unwrap();
    let mut receiver = Connection::new(right).unwrap();

    let files: Vec<std::fs::File> = (0..MAX_DESCRIPTORS_PLUS)
        .map(|index| file(&index.to_string()))
        .collect();
    let borrowed: Vec<BorrowedFd<'_>> = files.iter().map(AsFd::as_fd).collect();
    sender.send_with(&Note("with".into()), &borrowed).unwrap();
    sender.send(&Note("after".into())).unwrap();
    sender.flush().unwrap();

    let (first, descriptors) = receiver.recv_with::<Note>(TIMEOUT).unwrap().unwrap();
    assert_eq!(first, Note("with".into()));
    assert_eq!(descriptors.len(), MAX_DESCRIPTORS_PLUS);
    for (index, descriptor) in descriptors.into_iter().enumerate() {
        assert_eq!(read(descriptor), index.to_string());
    }
    let (second, descriptors) = receiver.recv_with::<Note>(TIMEOUT).unwrap().unwrap();
    assert_eq!((second, descriptors.len()), (Note("after".into()), 0));
}

/// More descriptors than one send carries, so a frame's travel on carrier bytes too.
const MAX_DESCRIPTORS_PLUS: usize = frame::MAX_DESCRIPTORS + 7;

/// A length-prefixed frame's descriptor waits for its receiver after the message that announces
/// it, and the frame after it brings none; TCP carries no descriptor.
#[test]
fn a_length_prefixed_frames_descriptor_is_taken_after_its_message() {
    let (left, right) = UnixStream::pair().unwrap();
    let mut sender = Connection::new(left).unwrap();
    let mut receiver = Connection::new(right).unwrap();

    let shared = file("shared bytes");
    sender
        .send_with(&Line("announce".into()), &[shared.as_fd()])
        .unwrap();
    drop(shared);
    sender.send(&Line("plain".into())).unwrap();

    assert_eq!(
        receive::<Line>(&mut receiver).unwrap(),
        Line("announce".into())
    );
    let descriptor = receiver.take_descriptor().expect("the frame's descriptor");
    assert_eq!(read(descriptor), "shared bytes");
    assert_eq!(
        receive::<Line>(&mut receiver).unwrap(),
        Line("plain".into())
    );
    assert!(receiver.take_descriptor().is_none());

    let (mut tcp, _peer) = with_raw_peer();
    let refused = tcp
        .send_with(&Line("x".into()), &[file("x").as_fd()])
        .unwrap_err();
    assert_eq!(refused.kind(), ErrorKind::InvalidInput);
}

/// A connection handed to another process goes on with the bytes it had received; one with a
/// descriptor not yet taken is not handed over, since the descriptor would not cross.
#[test]
fn a_handed_over_connection_goes_on_with_what_it_received() {
    let (left, right) = UnixStream::pair().unwrap();
    let mut sender = Connection::new(left).unwrap();
    let mut receiver = Connection::new(right).unwrap();

    sender.send(&Line("first".into())).unwrap();
    sender.send(&Line("second".into())).unwrap();
    assert_eq!(
        receive::<Line>(&mut receiver).unwrap(),
        Line("first".into())
    );

    let mut resumed = Connection::resume(receiver.hand_over().unwrap()).unwrap();
    assert_eq!(
        receive::<Line>(&mut resumed).unwrap(),
        Line("second".into())
    );

    let shared = file("held");
    sender
        .send_with(&Line("announce".into()), &[shared.as_fd()])
        .unwrap();
    assert_eq!(
        receive::<Line>(&mut resumed).unwrap(),
        Line("announce".into())
    );
    let refused = resumed.hand_over().err().expect("a held descriptor stays");
    assert_eq!(refused.kind(), ErrorKind::WouldBlock);
}

/// A closed peer ends a receive that would otherwise wait for its whole timeout.
#[test]
fn a_closed_peer_ends_a_receive() {
    let (left, right) = UnixStream::pair().unwrap();
    drop(left);
    let mut receiver = Connection::new(right).unwrap();
    let started = Instant::now();
    let error = receiver.recv::<Note>(Duration::from_secs(30)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
    assert!(started.elapsed() < Duration::from_secs(30));
}
