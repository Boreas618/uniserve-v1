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
//! A pause closes the daemon's admissions when it starts: from then on the
//! deployment's new requests belong to the successor, and a connection ends at
//! its next request boundary instead of starting a response this process
//! cannot hand over (design §7.5.1 step 1). A response that registers while the
//! pause's round runs is asked to pause too, so the restart never hands over a
//! connection whose next response is already being written; the round clears
//! `pausing` when it ends, so `pausing` is true exactly while a round runs and
//! will read what a response reports.

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
    // `CLOCK_MONOTONIC` starts at the boot and both fields are non-negative;
    // the default keeps the conversion total without a panic in the handoff
    // path, which must not abort a stream.
    u64::try_from(time.tv_sec).unwrap_or_default() * 1_000_000_000
        + u64::try_from(time.tv_nsec).unwrap_or_default()
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

/// Whether this daemon admits new requests, as its connections read it.
///
/// A hot restart closes admissions when its pause starts: from then on the
/// deployment's new requests belong to the successor (design §7.5.1 step 1),
/// and a connection that would send one ends at that boundary instead of being
/// served by a process that is leaving. A response already in flight is
/// unaffected — it is what the pause is for.
///
/// It reopens when a handoff gives the deployment back ([`Handoff::release`]):
/// a successor that gives up leaves this daemon serving, and then it must
/// admit again. An atomic rather than a `CancellationToken`, because a
/// cancellation is final and this state is not.
#[derive(Debug, Clone, Default)]
pub(crate) struct Admissions(Arc<AtomicBool>);

impl Admissions {
    /// Whether a request may be admitted now.
    pub(crate) fn open(&self) -> bool {
        !self.0.load(Ordering::Acquire)
    }

    /// Stops admitting: the deployment's next request is the successor's.
    pub(crate) fn close(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Admits again, after a pause that handed nothing over.
    fn reopen(&self) {
        self.0.store(false, Ordering::Release);
    }
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
    /// registers now pauses at once. True exactly while a pause's round runs,
    /// which is what makes the round the reader of every report it asks for.
    pausing: AtomicBool,
    /// Whether new requests are admitted; closed for the whole handover.
    admitting: Admissions,
    /// The responses in flight, weakly: the connection owns its registration,
    /// and the registry must not keep one alive — a response that ended has to
    /// reach `Drop` for the pause to see it go.
    streaming: Mutex<BTreeMap<u64, std::sync::Weak<Registration>>>,
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
        crate::leader::locked(&self.inner.streaming).insert(id, Arc::downgrade(&registration));
        // A restart that began pausing before this response started must not
        // hand over a connection whose next response is already being
        // written: ask this one to pause too. `pausing` is true only while the
        // round runs, and the round reads every registration it asked, so a
        // response that starts after it is written here rather than parked for
        // a decision the finished round will not make.
        if self.pausing() {
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
    /// one rather than keeping it. Responses that register while the round
    /// runs are asked too, so the set is the responses in flight when the last
    /// of them stops; the round ends when the registry is stable, and closing
    /// admissions first is what bounds it.
    pub(crate) async fn pause(&self) -> Vec<FrozenStream> {
        // Design §7.5.1 step 1: the successor takes the deployment's listener,
        // so this daemon takes no new request from here on. Closing before the
        // round is also what keeps the round conclusive — a connection cannot
        // start a request that would register after it.
        self.inner.admitting.close();
        self.inner.pausing.store(true, Ordering::Release);
        let mut asked = BTreeSet::new();
        let mut frozen = Vec::new();
        loop {
            let fresh: Vec<Arc<Registration>> = {
                let streaming = crate::leader::locked(&self.inner.streaming);
                streaming
                    .values()
                    .filter_map(std::sync::Weak::upgrade)
                    .filter(|registration| !asked.contains(&registration.id))
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
                    .values()
                    .filter_map(std::sync::Weak::upgrade)
                    .all(|registration| asked.contains(&registration.id));
                if all_asked {
                    // The round is over, and `pausing` stops being true: a
                    // response that starts now is written here, and the
                    // registrations it asked stop watching for a boundary.
                    self.end_pause();
                    return frozen;
                }
                continue;
            }
            for registration in fresh {
                // Claim the report before asking for it: a response waits for
                // a decision only when a round holds its report, and a
                // response whose report nobody claimed resumes at once
                // (`Registration::report`). A registration an earlier claim
                // already took is never asked twice.
                let answered = {
                    let mut slot = registration.frozen.lock().await;
                    slot.take()
                };
                asked.insert(registration.id);
                let Some(answered) = answered else {
                    continue;
                };
                registration.request();
                // This loop's own reference goes before the wait: a response
                // that ends now must reach `Drop`, which is what tells the
                // pause it left (the registry holds the registration weakly).
                drop(registration);
                if let Ok(Some(stream)) = answered.await {
                    frozen.push(stream);
                }
            }
        }
    }

    /// Ends the pause's round: no response asks for a boundary any more, and
    /// a response that starts from here on is written here.
    fn end_pause(&self) {
        self.inner.pausing.store(false, Ordering::Release);
        for registration in crate::leader::locked(&self.inner.streaming)
            .values()
            .filter_map(std::sync::Weak::upgrade)
        {
            registration.release();
        }
    }

    /// Gives the deployment back to this daemon: the pause ended without a
    /// handover, so the responses in flight go on being written here and new
    /// requests are admitted again.
    pub(crate) fn release(&self) {
        self.end_pause();
        self.inner.admitting.reopen();
    }

    /// Whether new requests are admitted, for the connections that serve them.
    pub(crate) fn admissions(&self) -> Admissions {
        self.inner.admitting.clone()
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
    pub(crate) async fn pausing(&self) -> bool {
        let mut requested = self.requested.clone();
        requested.wait_for(|requested| *requested).await.is_ok()
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
        // A pause claims a registration's report before it asks it to freeze,
        // so an unclaimed report means no round is waiting for this one: the
        // response is written here. That is the belt for the one registration
        // admissions cannot exclude — a request admitted just before the pause
        // whose handler registered just after it — and it is what keeps a
        // response from waiting for a decision nobody will make.
        if self.frozen.lock().await.is_some() {
            return Decision::Resume;
        }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::{Decision, FrozenStream, Handoff};

    /// A frozen response's minimal state: a socket descriptor is all the
    /// pause needs of it beyond the state the connection filled in.
    fn frozen(id: u64, request_id: &str) -> FrozenStream {
        let (decision, _decided) = tokio::sync::oneshot::channel();
        let socket: std::os::fd::OwnedFd = std::fs::File::open("/dev/null")
            .expect("a socket stand-in")
            .into();
        FrozenStream {
            id,
            request_id: request_id.to_owned(),
            frozen_at_ns: super::now_ns(),
            state: proto::handoff::Stream::default(),
            descriptor: socket,
            decision,
        }
    }

    /// A pause ends when every response in flight has answered: one that ends
    /// before it freezes neither holds the pause nor is handed over.
    #[tokio::test]
    async fn a_response_that_ends_before_it_freezes_does_not_hold_the_pause() {
        let handoff = Handoff::new();
        let kept = handoff.register();
        let ended = handoff.register();
        let freezing = tokio::spawn({
            let kept = Arc::clone(&kept);
            async move {
                assert!(kept.pausing().await, "the kept response is asked to pause");
                kept.report(Some(frozen(kept.id(), "kept"))).await
            }
        });

        // The other response's connection ends before it can freeze.
        drop(ended);

        let paused = tokio::time::timeout(Duration::from_secs(10), handoff.pause())
            .await
            .expect("the pause ends once the responses in flight answered");
        assert_eq!(
            paused.len(),
            1,
            "only the response that froze is handed over"
        );
        assert_eq!(paused[0].request_id, "kept");

        // The restart's decision reaches the response that froze.
        paused
            .into_iter()
            .next()
            .expect("the frozen response")
            .decision
            .send(Decision::Resume)
            .expect("the response awaits its decision");
        assert_eq!(freezing.await.expect("the freeze task"), Decision::Resume);
    }

    /// A response that registers after the pause's round ended is written
    /// here: no round will read what it reports, so it is neither asked to
    /// pause nor left waiting for a decision.
    #[tokio::test]
    async fn a_response_that_registers_after_the_pause_is_written_here() {
        let handoff = Handoff::new();
        let paused = handoff.pause().await;
        assert!(paused.is_empty(), "no response was in flight");
        assert!(!handoff.pausing(), "the round ended with the pause");
        assert!(
            !handoff.admissions().open(),
            "the daemon takes no new request while it hands the deployment over"
        );

        let late = handoff.register();
        assert!(!late.paused(), "the finished round asks nothing to pause");
        // A connection that froze it anyway — a request admitted just before
        // the pause, whose handler registered just after — is answered at
        // once, because no round holds its report.
        let decision = tokio::time::timeout(
            Duration::from_secs(10),
            late.report(Some(frozen(late.id(), "late"))),
        )
        .await
        .expect("the response is not left waiting for a decision");
        assert_eq!(decision, Decision::Resume, "the response goes on here");
    }

    /// A pause that hands nothing over gives the deployment back: this daemon
    /// admits again, and the next handover pauses and hands over the responses
    /// in flight then — a daemon hands a deployment over more than once.
    #[tokio::test]
    async fn a_released_pause_admits_again_and_the_next_one_hands_over() {
        let handoff = Handoff::new();
        assert!(handoff.pause().await.is_empty());
        handoff.release();
        assert!(handoff.admissions().open(), "the daemon admits again");

        let kept = handoff.register();
        let freezing = tokio::spawn({
            let kept = Arc::clone(&kept);
            async move {
                assert!(kept.pausing().await, "the second round asks it to pause");
                kept.report(Some(frozen(kept.id(), "kept"))).await
            }
        });
        let paused = tokio::time::timeout(Duration::from_secs(10), handoff.pause())
            .await
            .expect("the second pause ends");
        assert!(
            !handoff.admissions().open(),
            "the second handover took the new requests"
        );
        assert_eq!(paused.len(), 1, "the response in flight is handed over");
        assert_eq!(paused[0].request_id, "kept");
        paused
            .into_iter()
            .next()
            .expect("the frozen response")
            .decision
            .send(Decision::Resume)
            .expect("the response awaits its decision");
        assert_eq!(freezing.await.expect("the freeze task"), Decision::Resume);
    }

    /// A pause hands over a response that registers while it is pausing, so
    /// the handed set is what is in flight when the pause ends.
    ///
    /// One response holds the pause open until the test says so, so a
    /// registration that happens while the pause runs is a registration into a
    /// pause that has not concluded — the condition §7.5.1 step 4 names for an
    /// idle keep-alive connection that starts a request during the handover.
    #[tokio::test]
    async fn a_response_that_registers_during_the_pause_is_handed_over_too() {
        let handoff = Handoff::new();
        let held = handoff.register();
        let (go_on, released) = tokio::sync::oneshot::channel();
        let freezing = tokio::spawn({
            let held = Arc::clone(&held);
            async move {
                assert!(held.pausing().await, "the held response is asked to pause");
                released.await.expect("the pause goes on");
                held.report(Some(frozen(held.id(), "held"))).await
            }
        });
        let pausing = tokio::spawn({
            let handoff = handoff.clone();
            async move { handoff.pause().await }
        });
        while !handoff.pausing() {
            tokio::task::yield_now().await;
        }

        let late = handoff.register();
        let waiting = tokio::spawn(async move {
            assert!(late.pausing().await, "the late response is asked to pause");
            late.report(Some(frozen(late.id(), "late"))).await
        });
        go_on.send(()).expect("the pause goes on");

        let paused = tokio::time::timeout(Duration::from_secs(10), pausing)
            .await
            .expect("the pause ends")
            .expect("the pause task");
        assert!(
            paused.iter().any(|stream| stream.request_id == "late"),
            "the response registered during the pause is handed over"
        );
        assert!(
            paused.iter().any(|stream| stream.request_id == "held"),
            "the response that held the pause is handed over"
        );
        for stream in paused {
            stream
                .decision
                .send(Decision::Resume)
                .expect("the response awaits its decision");
        }
        assert_eq!(freezing.await.expect("the held response"), Decision::Resume);
        assert_eq!(
            waiting.await.expect("the late registration"),
            Decision::Resume
        );
    }
}
