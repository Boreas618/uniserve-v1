//! The responses in flight of one daemon, and how a hot restart freezes them
//! (design `specs/worker.md` §7.5.1).
//!
//! Every response the daemon produces itself — a chat completion streamed as
//! server-sent events, one answered as a JSON body, a video artifact streamed
//! from retained media — registers here while it is written. A restart asks
//! the registry to pause them: each response stops at its next event boundary,
//! reports the state a successor continues from together with a descriptor of
//! its socket, and then writes nothing until the restart tells it whether the
//! successor adopted it.
//!
//! The registry is the daemon's, not the connection's: a response registers
//! when its body starts and leaves the registry when it ends, so the restart
//! sees exactly the responses that are in flight. Every response kind item 1
//! covers has a state a successor continues from; a covered response whose
//! state cannot be frozen is a defect, and its connection ends it rather than
//! keeping it here (see [`crate::http::stream`]).
//!
//! A response that registers after the restart began pausing is asked to
//! pause too, so the restart never hands over a connection whose next response
//! is already being written.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{oneshot, watch};

use proto::handoff;

/// The HTTP framing of a frozen response, as the handoff state carries it.
pub(crate) use proto::handoff::Http;

/// The current time on `CLOCK_MONOTONIC`, in nanoseconds.
///
/// The predecessor records when it stopped a response's socket and the
/// successor records when it first writes there; both processes of a node
/// read the same clock, so the difference is the response's pause.
pub(crate) fn now_ns() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `CLOCK_MONOTONIC` is a valid clock and `time` a valid pointer.
    let started = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    assert_eq!(started, 0, "CLOCK_MONOTONIC is always readable");
    u64::try_from(time.tv_sec).expect("a positive monotonic clock") * 1_000_000_000
        + u64::try_from(time.tv_nsec).expect("nanoseconds below a second")
}

/// What a restart decides about one frozen response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// The successor did not take the response over; write it here.
    Resume,
    /// The successor owns the response. This process ends its consumption
    /// of the request's engine stream without cancelling it (the engine's
    /// next consumer takes the stream over, design §7.5.1 step 4), drops the
    /// socket, and never writes on it again.
    Adopt,
}

/// One response in flight, frozen at an event boundary: everything a
/// successor needs of it besides the state and the socket.
#[derive(Debug)]
pub(crate) struct FrozenStream {
    /// The response's identity within this handoff.
    pub(crate) id: u64,
    /// The request's identity as the deployment named it, which the pause
    /// record reports.
    pub(crate) request_id: String,
    /// When this daemon stopped the response's socket, on `CLOCK_MONOTONIC`.
    pub(crate) frozen_at_ns: u64,
    /// The state a successor continues from, with the HTTP framing the
    /// connection filled in.
    pub(crate) state: handoff::Stream,
    /// A descriptor of the response's socket: the same socket object the
    /// kernel holds the connection's bytes in.
    pub(crate) descriptor: std::os::fd::OwnedFd,
    /// Where the restart's decision is delivered.
    pub(crate) decision: oneshot::Sender<Decision>,
}

/// The responses in flight of one daemon.
#[derive(Debug, Clone, Default)]
pub(crate) struct Handoff {
    inner: Arc<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    next: AtomicU64,
    /// Whether a hot restart is handing the deployment over: a response that
    /// registers now pauses at once.
    pausing: AtomicBool,
    streaming: Mutex<BTreeMap<u64, Arc<Registration>>>,
}

impl Handoff {
    /// A registry of a daemon that is serving.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Registers one response that is about to write a body this daemon can
    /// hand over.
    ///
    /// The registration leaves the registry when it is dropped, which is when
    /// the response ended on this daemon.
    pub(crate) fn register(&self) -> Arc<Registration> {
        let id = self.inner.next.fetch_add(1, Ordering::Relaxed) + 1;
        let (request, requested) = watch::channel(false);
        let (report, frozen) = oneshot::channel();
        let registration = Arc::new(Registration {
            id,
            request,
            requested,
            registrations: Arc::downgrade(&self.inner),
            report: Mutex::new(Some(report)),
            frozen: tokio::sync::Mutex::new(Some(frozen)),
        });
        crate::leader::locked(&self.inner.streaming).insert(id, Arc::clone(&registration));
        // A restart that began pausing before this response started must not
        // hand over a connection whose next response is already being
        // written: ask this one to pause too.
        if self.inner.pausing.load(Ordering::Acquire) {
            registration.request();
        }
        registration
    }

    /// Whether a hot restart is pausing the responses.
    pub(crate) fn pausing(&self) -> bool {
        self.inner.pausing.load(Ordering::Acquire)
    }

    /// Asks every response in flight to stop at its next event boundary and
    /// returns the frozen ones, with their sockets.
    ///
    /// Responses that end before they freeze are not returned, and neither
    /// are responses that cannot be handed over: the connection ends such a
    /// one rather than keeping it. Responses that register while the restart
    /// pauses are asked too, so the set is the responses in flight when the
    /// last of them stops.
    pub(crate) async fn pause(&self) -> Vec<FrozenStream> {
        self.inner.pausing.store(true, Ordering::Release);
        let mut asked = BTreeSet::new();
        let mut frozen = Vec::new();
        loop {
            let fresh: Vec<Arc<Registration>> = {
                let streaming = crate::leader::locked(&self.inner.streaming);
                streaming
                    .values()
                    .filter(|registration| !asked.contains(&registration.id))
                    .map(Arc::clone)
                    .collect()
            };
            if fresh.is_empty() {
                // Every response registered when this round began has
                // answered, and a response that ended meanwhile is gone from
                // the registry: the handoff holds everything in flight.
                //
                // A response may still register between that round and this
                // check, so yield and look again before concluding.
                tokio::task::yield_now().await;
                let all_asked = crate::leader::locked(&self.inner.streaming)
                    .keys()
                    .all(|id| asked.contains(id));
                if all_asked {
                    return frozen;
                }
                continue;
            }
            for registration in &fresh {
                asked.insert(registration.id);
                registration.request();
            }
            for registration in fresh {
                let answered = {
                    let mut slot = registration.frozen.lock().await;
                    slot.take()
                };
                let Some(answered) = answered else {
                    continue;
                };
                if let Ok(Some(stream)) = answered.await {
                    frozen.push(stream);
                }
            }
        }
    }

    /// Ends a pause that did not hand the responses over: they go on being
    /// written here.
    pub(crate) fn release(&self) {
        self.inner.pausing.store(false, Ordering::Release);
        for registration in crate::leader::locked(&self.inner.streaming).values() {
            registration.release();
        }
    }
}

/// One response in flight, from its body's start to its end.
#[derive(Debug)]
pub(crate) struct Registration {
    id: u64,
    /// Tells the response to pause at its next event boundary.
    request: watch::Sender<bool>,
    requested: watch::Receiver<bool>,
    registrations: std::sync::Weak<Inner>,
    /// Armed at registration. The connection takes the sender when it freezes;
    /// the restart takes the receiver when it pauses.
    report: Mutex<Option<oneshot::Sender<Option<FrozenStream>>>>,
    frozen: tokio::sync::Mutex<Option<oneshot::Receiver<Option<FrozenStream>>>>,
}

impl Registration {
    /// The response's identity within this daemon.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Whether a pause was asked for.
    pub(crate) fn paused(&self) -> bool {
        *self.requested.borrow()
    }

    /// Watches for this response being asked to pause, for a connection that
    /// waits for the next body frame: `changed()` resolves on every request
    /// and on the restart releasing the responses.
    pub(crate) fn watch(&self) -> watch::Receiver<bool> {
        self.requested.clone()
    }

    /// Waits until a hot restart asks this response to pause.
    ///
    /// Returns `false` when the restart ended without pausing this response.
    pub(crate) async fn pausing(&mut self) -> bool {
        self.requested.wait_for(|requested| *requested).await.is_ok()
    }

    /// Reports the response frozen and waits for the restart's decision.
    ///
    /// `frozen` is `None` when this response cannot be handed over; the
    /// response then keeps running here. The socket is not written and the
    /// request's engine stream is not read until the decision arrives, so the
    /// response still belongs to this daemon: [`Decision::Resume`] continues
    /// it here, [`Decision::Adopt`] ends it.
    pub(crate) async fn report(&self, frozen: Option<FrozenStream>) -> Decision {
        let reported = {
            let mut slot = crate::leader::locked(&self.report);
            slot.take()
        };
        let Some(reported) = reported else {
            // A second pause request: the first one owns the decision.
            return Decision::Resume;
        };
        let Some(mut frozen) = frozen else {
            let _ = reported.send(None);
            return Decision::Resume;
        };
        let (decide, decided) = oneshot::channel();
        frozen.decision = decide;
        if reported.send(Some(frozen)).is_err() {
            return Decision::Resume;
        }
        decided.await.unwrap_or(Decision::Resume)
    }

    fn request(&self) {
        self.request.send_replace(true);
    }

    fn release(&self) {
        self.request.send_replace(false);
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(inner) = self.registrations.upgrade() {
            crate::leader::locked(&inner.streaming).remove(&self.id);
            // Wake a restart waiting for this response's frozen state: it
            // ended before it could be frozen.
            let slot = {
                let mut slot = crate::leader::locked(&self.report);
                slot.take()
            };
            if let Some(reported) = slot {
                let _ = reported.send(None);
            }
        }
    }
}
