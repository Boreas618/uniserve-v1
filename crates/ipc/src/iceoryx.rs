//! Request-response endpoints built on iceoryx2 shared storage.
//!
//! This is the rank channel for a rank on the head's host: the engine holds a
//! [`ClientEndpoint`] and the rank a [`ServerEndpoint`], both opened on one
//! iceoryx2 request-response service named by [`service_name`]. A request or
//! response sample holds one [`Frame`], the same bytes the socket transport in
//! `crate::socket` writes to its stream. Companion event services provide
//! blocking wakeups without polling the request rings.
//!
//! A response is matched to its request by the envelope's `message_id`: the
//! server keeps each received request until a response states its identity.
//!
//! The module also owns the error vocabulary shared by both transports:
//! [`IpcError`] and [`IpcResult`].
//!
//! # Node ownership
//!
//! Each endpoint owns the iceoryx2 node its ports were created from and drops
//! it after them. iceoryx2 removes a node's resources when the last handle to
//! the node drops, and a port releases its handle before it removes the tag
//! it keeps in the node's directory. A port that outlives the node therefore
//! runs that removal while its own tag still occupies the directory: the
//! removal fails with a warning and leaves the directory behind. A
//! `WakeSender` holds a port too, so whoever holds a clone of one drops it
//! before the endpoint it came from.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::frame::Frame;
use crate::{ValidationError, WorkerRequest, WorkerResponse};
use iceoryx2::active_request::ActiveRequest;
use iceoryx2::pending_response::PendingResponse;
use iceoryx2::port::client::Client;
use iceoryx2::port::server::Server;
use iceoryx2::prelude::*;

mod events;
use events::{ClientEvents, ServerEvents};
pub use events::{EVT_COMPLETION, EVT_DEATH, EVT_REQUEST, EVT_RESULT, WakeEvents, WakeSender};

/// Default namespace prefix for per-worker iceoryx2 services.
pub const DEFAULT_SERVICE_PREFIX: &str = "uniserve/worker";

/// Result type returned by worker transport calls.
pub type IpcResult<T> = std::result::Result<T, IpcError>;

/// Encoding, validation and transport failures at the IPC boundary.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// Submission was not accepted because the bounded send queue is full.
    ///
    /// Only the socket transport's send queue produces this. The shared-storage
    /// transport reports an exhausted iceoryx2 request limit as `Transport`.
    #[error("rank channel send queue is full")]
    WouldBlock,
    /// A message could not be encoded, or received bytes are not a frame of
    /// the message expected.
    #[error("worker message encoding error: {0}")]
    Encoding(String),
    /// A message violates the worker protocol contract.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// A rank channel's setup or one of its calls failed.
    #[error("worker IPC error: {0}")]
    Transport(String),
}

impl IpcError {
    /// Constructs a transport error with contextual text.
    pub(crate) fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }
}

/// Extension methods for attaching transport context to fallible calls.
trait IpcContext<T> {
    /// Replaces a missing value or source error with fixed transport context.
    fn context(self, message: &str) -> IpcResult<T>;

    /// Adds lazily constructed transport context to a missing value or error.
    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display;
}

impl<T, E> IpcContext<T> for std::result::Result<T, E>
where
    E: std::fmt::Debug,
{
    /// Wraps a source error with fixed transport context.
    fn context(self, message: &str) -> IpcResult<T> {
        self.map_err(|error| IpcError::transport(format!("{message}: {error:?}")))
    }

    /// Wraps a source error with lazily constructed transport context.
    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.map_err(|error| IpcError::transport(format!("{}: {error:?}", message())))
    }
}

impl<T> IpcContext<T> for Option<T> {
    /// Converts absence into a transport error with fixed context.
    fn context(self, message: &str) -> IpcResult<T> {
        self.ok_or_else(|| IpcError::transport(message))
    }

    /// Converts absence into a transport error with lazily constructed context.
    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.ok_or_else(|| IpcError::transport(message().to_string()))
    }
}

macro_rules! ipc_bail {
    ($($arg:tt)*) => {
        return Err(IpcError::transport(format!($($arg)*)))
    };
}

/// Thread-safe iceoryx2 IPC service type used by endpoint aliases.
pub(crate) type IxService = iceoryx2::service::ipc_threadsafe::Service;
/// Concrete iceoryx2 client port used by the host endpoint.
type IxClient = Client<IxService, [u8], (), [u8], ()>;
/// Concrete iceoryx2 server port used by the worker endpoint.
type IxServer = Server<IxService, [u8], (), [u8], ()>;
/// Active request handle retained until the worker publishes its response.
type IxActive = ActiveRequest<IxService, [u8], (), [u8], ()>;
/// Pending response handle returned after a request is sent.
pub type Pending = PendingResponse<IxService, [u8], (), [u8], ()>;

/// Builds the iceoryx2 service name for one worker identifier.
///
/// An identifier containing `/` is taken as a complete service name; any other
/// identifier is placed under [`DEFAULT_SERVICE_PREFIX`]. The event companions
/// append their own suffixes to the service name an endpoint opens.
pub fn service_name(id: &str) -> String {
    if id.contains('/') {
        id.to_string()
    } else {
        format!("{DEFAULT_SERVICE_PREFIX}/{id}")
    }
}

/// Host-side endpoint for sending requests and receiving worker responses.
pub struct ClientEndpoint {
    /// Request-response port used to loan and send request frames.
    client: IxClient,
    /// Largest encoded message in bytes accepted in either direction.
    max_payload: usize,
    /// How long `send_request` keeps retrying while no server is connected.
    connect_timeout: Duration,
    /// Directional wake ports paired with the request-response service.
    events: ClientEvents,
    /// Node owner that keeps all client ports alive. Declared last so it is
    /// dropped after them (see the module's node ownership section).
    _node: Node<IxService>,
}

impl ClientEndpoint {
    /// Connects a host endpoint to a worker service, creating it if absent.
    ///
    /// `max_payload` bounds every encoded message in bytes. `max_inflight` is
    /// the number of [`Pending`] handles this client may hold at once; the engine passes
    /// its per-rank queue depth, which `RankProcess::finish_startup` requires
    /// to equal the worker's advertised `queue_depth`. A send while that many
    /// handles are alive fails with a transport error.
    pub fn connect(service: &str, max_payload: usize, max_inflight: usize) -> IpcResult<Self> {
        // Size the service for one client and one server, `max_inflight`
        // outstanding requests and one response per request, and disable
        // safe overflow so a full buffer never discards an unconsumed frame.
        // `ServerEndpoint::bind` requests identical settings: either side may
        // create the service through `open_or_create`, and iceoryx2 refuses to
        // open an existing service whose settings are incompatible.
        let service_name = ServiceName::new(service).context("invalid iceoryx2 service name")?;
        let node = NodeBuilder::new()
            .create::<IxService>()
            .context("creating iceoryx2 client node")?;
        let factory = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .max_active_requests_per_client(max_inflight.max(1))
            .max_response_buffer_size(1)
            .max_loaned_requests(max_inflight.max(1))
            .max_servers(1)
            .max_clients(1)
            .max_borrowed_responses_per_pending_response(1)
            .enable_safe_overflow_for_requests(false)
            .enable_safe_overflow_for_responses(false)
            .open_or_create()
            .context("opening iceoryx2 request-response service")?;

        // Start with a slice of at most 64 KiB; power-of-two growth
        // accommodates larger frames.
        let client = factory
            .client_builder()
            .initial_max_slice_len(max_payload.clamp(1, 64 * 1024))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .create()
            .context("creating iceoryx2 client port")?;

        // Companion event ports provide file-descriptor-based wakeups.
        let events =
            ClientEvents::open(&node, service).context("opening client event companions")?;

        Ok(Self {
            _node: node,
            client,
            max_payload,
            connect_timeout: Duration::from_secs(300),
            events,
        })
    }

    /// Parks for {result, death} until a wake fires or `timeout`
    /// elapses. Returns which sources fired; none set means no wake was
    /// pending when the wait ended.
    pub fn wait_wake(&self, timeout: Duration) -> IpcResult<WakeEvents> {
        self.events.wait(timeout)
    }

    /// Drains queued wake ids after a composite executor parked on this
    /// listener's descriptor.
    pub fn drain_wakes(&self) -> IpcResult<WakeEvents> {
        self.events.drain()
    }

    /// Returns the borrowed descriptor used by an external poll loop.
    ///
    /// The descriptor stays owned by this endpoint and is valid only while it
    /// lives. After it polls readable, call [`Self::drain_wakes`].
    pub fn wake_file_descriptor(&self) -> i32 {
        self.events.file_descriptor()
    }

    /// Returns a cloneable wake source for worker-process exit.
    pub fn death_wake(&self) -> WakeSender {
        self.events.death_wake()
    }

    /// Encodes and sends a request, retrying while no server is connected.
    ///
    /// A request sent before the worker's server port exists reaches no one, so
    /// such a send is discarded and repeated every 20 ms until a server is
    /// connected or `connect_timeout` elapses. Any other send failure returns
    /// immediately. The engine uses [`Self::send_request_attempt`] and its own
    /// connectivity loop instead; this blocking form serves tests.
    pub fn send_request(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let frame = Frame::encode(req, self.max_payload)?;
        let deadline = Instant::now() + self.connect_timeout;
        loop {
            let pending = self.send_frame(&frame)?;
            if pending.number_of_server_connections() > 0 {
                return Ok(pending);
            }

            // Release the unanswerable request's active-request slot before
            // retrying.
            drop(pending);
            if Instant::now() >= deadline {
                ipc_bail!("iceoryx2 worker service has no connected server");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Encodes and sends a request once and wakes the worker's request
    /// listener, whether or not a server is connected.
    ///
    /// Fails when the request is invalid or its encoding exceeds
    /// `max_payload`, and with a transport error when iceoryx2 refuses the
    /// loan or send, for example while `max_inflight` pending handles are
    /// alive. Success does not imply that a server received the request; the
    /// caller checks `Pending::number_of_server_connections` on the returned
    /// handle.
    pub fn send_request_attempt(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let frame = Frame::encode(req, self.max_payload)?;
        self.send_frame(&frame)
    }

    /// Sends one frame as a request sample.
    fn send_frame(&self, frame: &Frame) -> IpcResult<Pending> {
        // Loan exact shared-storage capacity, copy the frame into it, then
        // transfer ownership to the request-response service.
        let bytes = frame.as_bytes();
        let request = self
            .client
            .loan_slice_uninit(bytes.len())
            .context("loaning iceoryx2 request slice")?;
        let request = request.write_from_slice(bytes);
        let pending = request.send().context("sending iceoryx2 request")?;

        // Notify only after the request is in the ring.
        self.events.notify_request();
        Ok(pending)
    }

    /// Attempts to receive the response associated with `pending`.
    ///
    /// Returns `Ok(None)` when no response has arrived. Fails when the sample
    /// is not a whole frame within `max_payload` or its message does not
    /// decode to a valid response.
    pub fn try_recv_response(&self, pending: &Pending) -> IpcResult<Option<WorkerResponse>> {
        let Some(response) = pending.receive().context("receiving iceoryx2 response")? else {
            return Ok(None);
        };
        let frame = Frame::from_bytes(response.payload().to_vec(), self.max_payload)?;
        frame.decode().map(Some)
    }

    /// Waits up to `timeout` for the response associated with `pending`.
    ///
    /// Returns `Ok(None)` at the deadline. Host wakes observed while waiting,
    /// including a worker-death wake, are drained without being reported, so a
    /// caller that tracks death must not rely on this method to surface it.
    pub fn recv_response_timeout(
        &self,
        pending: &Pending,
        timeout: Duration,
    ) -> IpcResult<Option<WorkerResponse>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(response) = self.try_recv_response(pending)? {
                return Ok(Some(response));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            self.events.wait(deadline - now)?;
        }
    }
}

/// Worker-side endpoint for receiving requests and publishing responses.
pub struct ServerEndpoint {
    /// Request-response port used to receive and answer request frames.
    server: IxServer,
    /// Largest encoded message in bytes accepted in either direction.
    max_payload: usize,
    /// Active requests retained until their matching `message_id` is answered.
    active: VecDeque<(u64, IxActive)>,
    /// Directional wake ports paired with the request-response service.
    events: ServerEvents,
    /// Node owner that keeps all server ports alive. Declared last so it is
    /// dropped after them (see the module's node ownership section).
    _node: Node<IxService>,
}

impl ServerEndpoint {
    /// Creates the worker endpoint and its directional wake services.
    ///
    /// `max_payload` and `max_inflight` size the service as in
    /// [`ClientEndpoint::connect`]; each side enforces its own `max_payload` on
    /// the frames it sends and receives.
    pub fn bind(service: &str, max_payload: usize, max_inflight: usize) -> IpcResult<Self> {
        // Mirror `ClientEndpoint::connect` exactly; iceoryx2 refuses to open a
        // service whose existing settings are incompatible with the request.
        let service_name = ServiceName::new(service).context("invalid iceoryx2 service name")?;
        let node = NodeBuilder::new()
            .create::<IxService>()
            .context("creating iceoryx2 server node")?;
        let factory = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .max_active_requests_per_client(max_inflight.max(1))
            .max_response_buffer_size(1)
            .max_loaned_requests(max_inflight.max(1))
            .max_servers(1)
            .max_clients(1)
            .max_borrowed_responses_per_pending_response(1)
            .enable_safe_overflow_for_requests(false)
            .enable_safe_overflow_for_responses(false)
            .open_or_create()
            .context("opening iceoryx2 request-response service")?;

        // Allocate response buffers with the same variable-size strategy.
        let server = factory
            .server_builder()
            .initial_max_slice_len(max_payload.clamp(1, 64 * 1024))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .max_loaned_responses_per_request(1)
            .create()
            .context("creating iceoryx2 server port")?;

        // Companion event ports wake request and asynchronous-completion paths.
        let events =
            ServerEvents::open(&node, service).context("opening server event companions")?;

        Ok(Self {
            _node: node,
            server,
            max_payload,
            active: VecDeque::new(),
            events,
        })
    }

    /// Attempts to receive one request without blocking.
    ///
    /// A received request's active request is retained until
    /// [`Self::respond`] answers its `message_id`. A sample that is not a whole
    /// frame within `max_payload`, or whose message does not decode to a valid
    /// request, is dropped together with its active request, so no response
    /// can be sent for it.
    pub fn try_recv(&mut self) -> IpcResult<Option<WorkerRequest>> {
        let Some(active) = self
            .server
            .receive()
            .context("receiving iceoryx2 request")?
        else {
            return Ok(None);
        };
        // Receiving from the ring consumes the work represented by wake hints.
        self.events.drain_worker_wakes()?;
        let frame = Frame::from_bytes(active.payload().to_vec(), self.max_payload)?;
        let request: WorkerRequest = frame.decode()?;

        // Retain transport ownership until a response with this message id arrives.
        self.active
            .push_back((request.message_id().unwrap_or_default(), active));
        Ok(Some(request))
    }

    /// Blocks until one request arrives; there is no overall deadline.
    ///
    /// Each park is bounded by `connect_timeout`, after which the ring is
    /// checked again. Returns only with a request or an error.
    pub fn recv(&mut self) -> IpcResult<WorkerRequest> {
        loop {
            if let Some(request) = self.try_recv()? {
                return Ok(request);
            }
            self.wait_incoming(self.connect_timeout())?;
        }
    }

    /// Parks until an inbound request or asynchronous completion wake fires, or
    /// `timeout` elapses. The caller re-checks all progress sources after return.
    ///
    /// Returns `Ok(())` on timeout as well as on a wake.
    pub fn wait_incoming(&self, timeout: Duration) -> IpcResult<()> {
        self.events.wait_request(timeout)
    }

    /// Returns a thread-safe signal for asynchronous worker completion.
    pub fn completion_wake(&self) -> WakeSender {
        self.events.completion_wake()
    }

    /// Encodes and publishes a response for the active request.
    ///
    /// The response's `message_id` selects the oldest retained request with
    /// that identity, `None` standing for zero. Answering removes the request,
    /// so a further response under that identity fails as unknown unless
    /// another retained request carries it. An invalid response, or one whose
    /// encoding exceeds `max_payload`, fails before any request is consumed,
    /// but once the request is resolved a failed loan or send still consumes
    /// it and no response can follow.
    pub fn respond(&mut self, resp: &WorkerResponse) -> IpcResult<()> {
        let frame = Frame::encode(resp, self.max_payload)?;
        let message_id = resp.message_id().unwrap_or_default();

        // Resolve and remove the exact active transport request before loaning
        // its response buffer.
        let pos = self
            .active
            .iter()
            .position(|(active_id, _)| *active_id == message_id)
            .with_context(|| {
                format!("respond called for unknown active request message_id {message_id}")
            })?;
        let active = self
            .active
            .remove(pos)
            .map(|(_, active)| active)
            .context("active request position disappeared")?;

        // Initialize the loaned response completely before publishing it.
        let bytes = frame.as_bytes();
        let response = active
            .loan_slice_uninit(bytes.len())
            .context("loaning iceoryx2 response slice")?;
        let response = response.write_from_slice(bytes);
        response.send().context("sending iceoryx2 response")?;

        // Notify only after the ring owns the fully initialized response.
        self.events.notify_response();
        Ok(())
    }

    /// Returns the bound on one park inside [`Self::recv`].
    fn connect_timeout(&self) -> Duration {
        Duration::from_secs(300)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkerResponseError;

    /// An error response whose encoded message is `filler` bytes of text plus
    /// a fixed envelope.
    fn response_of(message_id: u64, filler: usize) -> WorkerResponse {
        let mut response = WorkerResponse::error(WorkerResponseError {
            message: "m".repeat(filler),
            code: None,
            fatal: false,
            phase: None,
            route: None,
            calls: Vec::new(),
        });
        response.set_call_id(Some(message_id));
        response
    }

    #[test]
    fn shared_storage_enforces_the_payload_bound_in_both_directions() {
        // The bound is the encoded length of the response the server sends
        // back, so that response fits exactly and one a byte longer does not.
        let fitting = response_of(7, 200);
        let bound = Frame::encode(&fitting, usize::MAX)
            .unwrap()
            .as_bytes()
            .len()
            - Frame::LENGTH_BYTES;
        let service = format!("payload-bound-{}", std::process::id());
        let mut server = ServerEndpoint::bind(&service, bound, 2).unwrap();
        let client = ClientEndpoint::connect(&service, bound, 2).unwrap();

        let mut request = WorkerRequest::info();
        request.set_call_id(Some(7));
        let pending = client.send_request(&request).unwrap();
        assert_eq!(server.recv().unwrap(), request);

        // The oversized response is refused before it consumes the request,
        // so the in-bound response can still answer it.
        assert!(matches!(
            server.respond(&response_of(7, 201)),
            Err(IpcError::Encoding(_))
        ));
        server.respond(&fitting).unwrap();
        let response = client
            .recv_response_timeout(&pending, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(response, fitting);

        // A client under a smaller bound refuses to send what this server
        // would have accepted.
        drop(pending);
        drop(client);
        let small = ClientEndpoint::connect(&format!("{service}-small"), 4, 2).unwrap();
        assert!(matches!(
            small.send_request_attempt(&request),
            Err(IpcError::Encoding(_))
        ));
    }
}
