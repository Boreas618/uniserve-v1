//! The responses in flight of one daemon, and how a hot restart freezes them.
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
//! sees exactly the responses that are in flight. A response whose state
//! cannot be frozen at its boundary reports so, and its connection ends it
//! rather than keeping it here (see [`crate::http::stream`]).
//!
//! A response registers when its handler attaches its body, not when its
//! request is admitted: a handler that generates before it responds (the
//! synchronous video route) has no body to hand over while it runs, and the
//! pause must not wait for its generation. Such a request, admitted before the
//! pause closed admissions, is written by this daemon, as is any response
//! whose body registers after the pause's round ended.
//!
//! A pause closes the daemon's admissions when it starts: from then on the
//! deployment's new requests belong to the successor, and a connection ends at
//! its next request boundary instead of starting a response this process
//! cannot hand over (design §7.5.1 step 1). A response that registers while the
//! pause's round runs is asked to pause too, so the restart never hands over a
//! connection whose next response is already being written.
//!
//! The round asks a response by arming its report and then raising its pause
//! request, so a response that sees the request always has a round to report
//! to; a response is asked at most once, since after its first pause the
//! restart owns its decision. Every report is answered: a frozen response is
//! decided by the restart, and one the restart drops undecided resumes here
//! ([`FrozenStream`]).

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
    let time = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    // `CLOCK_MONOTONIC` counts from the boot, so both fields are non-negative.
    time.tv_sec.unsigned_abs() * 1_000_000_000 + time.tv_nsec.unsigned_abs()
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
/// successor needs of it besides the state and the socket, and the channel
/// the restart's decision reaches it by.
///
/// A frozen response is decided once ([`FrozenStream::decide`]); one dropped
/// undecided — the successor gave up, or the takeover failed before it named
/// the responses it took — resumes here, so the response that waits for its
/// decision always receives one.
#[derive(Debug)]
pub(crate) struct FrozenStream {
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
    /// Where the restart's decision is delivered, once the response reported
    /// itself ([`Registration::report`]).
    decision: Option<oneshot::Sender<Decision>>,
}

impl FrozenStream {
    /// The response `request_id` frozen now with `state`, on the socket
    /// `descriptor`.
    pub(crate) fn new(
        request_id: String,
        state: handoff::Stream,
        descriptor: std::os::fd::OwnedFd,
    ) -> Self {
        Self {
            request_id,
            frozen_at_ns: now_ns(),
            state,
            descriptor,
            decision: None,
        }
    }

    /// Delivers the restart's decision to the response. A response whose
    /// connection already ended takes none, and needs none.
    pub(crate) fn decide(mut self, decision: Decision) {
        if let Some(decided) = self.decision.take() {
            let _ = decided.send(decision);
        }
    }
}

impl Drop for FrozenStream {
    /// A frozen response nobody decided resumes here: the handoff gave it back.
    fn drop(&mut self) {
        if let Some(decided) = self.decision.take() {
            let _ = decided.send(Decision::Resume);
        }
    }
}

/// What a response reports to the round that asked it.
#[derive(Debug)]
enum Report {
    /// It froze at its boundary: the successor may take it over.
    Frozen(Box<FrozenStream>),
    /// Its state cannot be frozen at its boundary; its connection ends it.
    Unhandable,
}

/// The responses in flight of one daemon.
#[derive(Debug, Clone, Default)]
pub(crate) struct Handoff {
    inner: Arc<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    next: AtomicU64,
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
    /// the response ended on this daemon. A response registered while a round
    /// runs is asked by that round; one registered after it is not asked.
    pub(crate) fn register(&self) -> Arc<Registration> {
        let id = self.inner.next.fetch_add(1, Ordering::Relaxed) + 1;
        let (request, requested) = watch::channel(false);
        let registration = Arc::new(Registration {
            id,
            request,
            requested,
            registrations: Arc::downgrade(&self.inner),
            asked: AtomicBool::new(false),
            report: Mutex::new(None),
        });
        service::lock(&self.inner.streaming).insert(id, Arc::downgrade(&registration));
        registration
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
        let mut seen = BTreeSet::new();
        let mut frozen = Vec::new();
        loop {
            let fresh: Vec<Arc<Registration>> = {
                let streaming = service::lock(&self.inner.streaming);
                let fresh: Vec<_> = streaming
                    .values()
                    .filter_map(std::sync::Weak::upgrade)
                    .filter(|registration| !seen.contains(&registration.id))
                    .collect();
                if fresh.is_empty() {
                    // Every response in flight has answered, and the lock is
                    // held while the round ends, so a response registers either
                    // before this check, and is asked, or after the round, and
                    // is written here.
                    release_all(&streaming);
                    return frozen;
                }
                fresh
            };
            for registration in fresh {
                seen.insert(registration.id);
                let Some(answered) = registration.ask() else {
                    // An earlier round asked it, and the restart owns its
                    // decision since.
                    continue;
                };
                // This loop's own reference goes before the wait: a response
                // that ends now must reach `Drop`, which is what tells the
                // pause it left (the registry holds the registration weakly).
                drop(registration);
                // A response that ended before it froze dropped its report.
                if let Ok(Report::Frozen(stream)) = answered.await {
                    frozen.push(*stream);
                }
            }
        }
    }

    /// Gives the deployment back to this daemon: the pause ended without a
    /// handover, so the responses in flight go on being written here and new
    /// requests are admitted again.
    pub(crate) fn release(&self) {
        release_all(&service::lock(&self.inner.streaming));
        self.inner.admitting.reopen();
    }

    /// Whether new requests are admitted, for the connections that serve them.
    pub(crate) fn admissions(&self) -> Admissions {
        self.inner.admitting.clone()
    }
}

/// Lowers the pause request of every registration in `streaming`: a response
/// that starts from here on is written here, and the ones a round asked stop
/// watching for a boundary.
fn release_all(streaming: &BTreeMap<u64, std::sync::Weak<Registration>>) {
    for registration in streaming.values().filter_map(std::sync::Weak::upgrade) {
        registration.request.send_replace(false);
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
    /// Whether a round asked this response; it is asked at most once.
    asked: AtomicBool,
    /// Where the response reports to the round that asked it: armed by the
    /// round before it raises the pause request, taken by the response's
    /// report or by its drop.
    report: Mutex<Option<oneshot::Sender<Report>>>,
}

impl Registration {
    /// Watches for this response being asked to pause, for a connection that
    /// waits for the next body frame: `changed()` resolves on every request
    /// and on the restart releasing the responses.
    pub(crate) fn watch(&self) -> watch::Receiver<bool> {
        self.requested.clone()
    }

    /// Arms this response's report and asks it to pause; the receiver of its
    /// report, or `None` when a round asked it before.
    fn ask(&self) -> Option<oneshot::Receiver<Report>> {
        if self.asked.swap(true, Ordering::AcqRel) {
            return None;
        }
        let (report, answered) = oneshot::channel();
        *service::lock(&self.report) = Some(report);
        // The report is armed before the request is visible, so a response
        // that sees the request finds its report.
        self.request.send_replace(true);
        Some(answered)
    }

    /// Takes the report the round armed before asking this response to pause.
    fn take_report(&self) -> oneshot::Sender<Report> {
        #[expect(
            clippy::expect_used,
            reason = "a response reports once, after it saw its pause request, which the round \
                      raises only once it armed the report"
        )]
        let report = service::lock(&self.report).take().expect("an armed report");
        report
    }

    /// Reports the response frozen and waits for the restart's decision.
    ///
    /// The connection calls this once, after it saw the pause request. The
    /// socket is not written and the request's engine stream is not read until
    /// the decision arrives, so the response still belongs to this daemon:
    /// [`Decision::Resume`] continues it here, [`Decision::Adopt`] ends it.
    pub(crate) async fn report(&self, mut frozen: FrozenStream) -> Decision {
        let report = self.take_report();
        let (decide, decided) = oneshot::channel();
        frozen.decision = Some(decide);
        // A round that is gone drops the response it was handed, which resumes
        // it (`FrozenStream`'s drop), so the decision always arrives.
        let _ = report.send(Report::Frozen(Box::new(frozen)));
        #[expect(
            clippy::expect_used,
            reason = "a frozen response is decided, or resumes when it is dropped undecided"
        )]
        let decision = decided.await.expect("a decision");
        decision
    }

    /// Reports that the response cannot be handed over at its boundary; its
    /// connection ends it. The connection calls this once, after it saw the
    /// pause request.
    pub(crate) fn unhandable(&self) {
        let _ = self.take_report().send(Report::Unhandable);
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(inner) = self.registrations.upgrade() {
            service::lock(&inner.streaming).remove(&self.id);
        }
        // A round waiting for this response's report sees it end before it
        // froze, when the armed report is dropped with it.
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{Decision, FrozenStream, Handoff, Registration};

    /// A frozen response's minimal state: a socket descriptor is all the
    /// pause needs of it beyond the state the connection filled in.
    fn frozen(request_id: &str) -> FrozenStream {
        let (socket, _peer) = std::os::unix::net::UnixStream::pair().expect("a socket pair");
        FrozenStream::new(
            request_id.to_owned(),
            proto::handoff::Stream::default(),
            socket.into(),
        )
    }

    /// Waits until a round asks `registration` to pause.
    async fn asked(registration: &Registration) {
        registration
            .watch()
            .wait_for(|asked| *asked)
            .await
            .expect("the registration's request");
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
                asked(&kept).await;
                kept.report(frozen("kept")).await
            }
        });

        // The other response's connection ends before it can freeze.
        drop(ended);

        let paused = handoff.pause().await;
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
            .decide(Decision::Adopt);
        assert_eq!(freezing.await.expect("the freeze task"), Decision::Adopt);
    }

    /// A frozen response the restart drops without a decision resumes here.
    #[tokio::test]
    async fn a_frozen_response_dropped_undecided_resumes() {
        let handoff = Handoff::new();
        let kept = handoff.register();
        let freezing = tokio::spawn({
            let kept = Arc::clone(&kept);
            async move {
                asked(&kept).await;
                kept.report(frozen("kept")).await
            }
        });
        drop(handoff.pause().await);
        assert_eq!(freezing.await.expect("the freeze task"), Decision::Resume);
    }

    /// A response that registers after the pause's round ended is written
    /// here: no round will read what it reports, so it is not asked to pause.
    #[tokio::test]
    async fn a_response_that_registers_after_the_pause_is_written_here() {
        let handoff = Handoff::new();
        let paused = handoff.pause().await;
        assert!(paused.is_empty(), "no response was in flight");
        assert!(
            !handoff.admissions().open(),
            "the daemon takes no new request while it hands the deployment over"
        );

        let late = handoff.register();
        assert!(
            !*late.watch().borrow(),
            "the finished round asks nothing to pause"
        );
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
                asked(&kept).await;
                kept.report(frozen("kept")).await
            }
        });
        let paused = handoff.pause().await;
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
            .decide(Decision::Resume);
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
        let (round_runs, running) = tokio::sync::oneshot::channel();
        let (go_on, released) = tokio::sync::oneshot::channel();
        let freezing = tokio::spawn({
            let held = Arc::clone(&held);
            async move {
                asked(&held).await;
                round_runs.send(()).expect("the test waits for the round");
                released.await.expect("the pause goes on");
                held.report(frozen("held")).await
            }
        });
        let pausing = tokio::spawn({
            let handoff = handoff.clone();
            async move { handoff.pause().await }
        });
        running.await.expect("the round asked the held response");

        let late = handoff.register();
        let waiting = tokio::spawn(async move {
            asked(&late).await;
            late.report(frozen("late")).await
        });
        go_on.send(()).expect("the pause goes on");

        let paused = pausing.await.expect("the pause task");
        assert!(
            paused.iter().any(|stream| stream.request_id == "late"),
            "the response registered during the pause is handed over"
        );
        assert!(
            paused.iter().any(|stream| stream.request_id == "held"),
            "the response that held the pause is handed over"
        );
        for stream in paused {
            stream.decide(Decision::Resume);
        }
        assert_eq!(freezing.await.expect("the held response"), Decision::Resume);
        assert_eq!(
            waiting.await.expect("the late registration"),
            Decision::Resume
        );
    }
}
