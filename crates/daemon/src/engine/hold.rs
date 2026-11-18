//! Holding requests while no engine serves.
//!
//! The leader submits every request through its [`RequestHold`]. While an
//! engine is ready, a submission goes straight to it. While none is, a
//! submission waits until one is, instead of being refused; once the hold
//! closes, because no engine will serve again, held and new submissions fail.
//! While the hold is paused (`paused`), submissions wait
//! as well, and the requests the engine already serves go on; while a
//! successor daemon takes the deployment over, they go to it. An epoch switch
//! holds nothing: new requests go to its target from the target's first
//! `Serving`, and the serving epoch goes on with what it holds.
//! A held submission is bounded only by its HTTP request's own timeout and by
//! its client, whose disconnect drops it.

use std::sync::Arc;

use proto::request::Request;
use tokio::sync::watch;

use crate::engine::client::{EngineClient, EngineState, EventStream, SubmitError};

/// Whom a submission goes to.
#[derive(Clone)]
enum Serving {
    /// No engine is ready: submissions wait.
    Holding,
    /// This engine serves submissions while it is ready.
    Engine(Arc<EngineClient>),
    /// The successor daemon owns readiness and engine recovery.
    Remote(Arc<crate::restart::requests::RemoteHold>),
    /// No engine will serve; submissions fail with the reason.
    Closed(String),
}

/// The changes of a hold from the moment they were taken ([`RequestHold::changes`]): of the
/// engine that serves, the hold's holding or closing, and its pause.
pub(crate) struct HoldChanges {
    serving: watch::Receiver<Serving>,
    paused: watch::Receiver<bool>,
}

impl HoldChanges {
    /// Resolves at the hold's next change; never once the hold is gone, which changes nothing
    /// more.
    pub(crate) async fn changed(&mut self) {
        tokio::select! {
            Ok(()) = self.serving.changed() => {}
            Ok(()) = self.paused.changed() => {}
            else => std::future::pending().await,
        }
    }
}

/// The leader's submissions and the engine that serves them.
pub struct RequestHold {
    serving: watch::Sender<Serving>,
    /// Whether submissions wait although an engine serves.
    paused: watch::Sender<bool>,
    /// The state the requests migrating out of an epoch continue with
    /// (`Self::keep`).
    kept: std::sync::Mutex<Option<Kept>>,
}

/// The state the requests migrating out of epoch `epoch` continue with,
/// although the epoch that serves next computes other numerics: an epoch
/// change's `in_flight = keep`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Kept {
    pub epoch: u64,
    /// The prefix seeds of the pages a token request continues with
    /// (`CachePolicy::state_seeds`), when the epochs' pages differ.
    pub prefix_seeds: Option<Vec<[u8; 32]>>,
    /// The latent seed of the trajectories a request continues with
    /// (`Trajectory::state_seed`: a video request's, or the image a token
    /// request had begun), when the epochs' trajectories differ.
    pub latent_seed: Option<u64>,
}

impl Kept {
    /// What the requests leaving `previous`'s engine keep when `next`'s
    /// engine serves them on: the state they reached, keyed by what
    /// `previous` computes wherever `next` computes otherwise. `None` when the
    /// two compute the same pages and trajectories, whose state `next` reads
    /// as its own.
    pub fn between(previous: &EngineClient, next: &EngineClient) -> Option<Self> {
        let (before, after) = (previous.info(), next.info());
        let prefix_seeds =
            (before.prefix_seeds != after.prefix_seeds).then(|| before.prefix_seeds.clone());
        let latent_seed = (before.latent_seed != after.latent_seed).then_some(before.latent_seed);
        (prefix_seeds.is_some() || latent_seed.is_some()).then(|| Self {
            epoch: previous.epoch(),
            prefix_seeds,
            latent_seed,
        })
    }
}

/// What a hold does with a submission now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoldState {
    /// It goes to an engine in this state.
    Engine(EngineState),
    /// It waits for an engine.
    Holding,
    /// It goes to the engine the successor daemon of a hot restart serves
    /// it with.
    HandedOver,
    /// It fails: no engine will serve.
    Closed(String),
}

enum Destination {
    Engine(Arc<EngineClient>),
    Remote(Arc<crate::restart::requests::RemoteHold>),
}

impl Default for RequestHold {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestHold {
    /// A hold no engine serves yet: submissions wait.
    pub fn new() -> Self {
        Self {
            serving: watch::Sender::new(Serving::Holding),
            paused: watch::Sender::new(false),
            kept: std::sync::Mutex::new(None),
        }
    }

    /// The state a request migrating out of epoch `epoch` continues with,
    /// if it keeps it.
    pub fn kept(&self, epoch: u64) -> Option<Kept> {
        ::service::lock(&self.kept)
            .clone()
            .filter(|kept| kept.epoch == epoch)
    }

    /// Has the requests that migrate out of `kept.epoch` from now on continue
    /// with the state `kept` states: an epoch change whose requests in
    /// flight keep the state they reached (`in_flight = keep`) states it as
    /// the hold moves to the next epoch's engine.
    pub fn keep(&self, kept: Kept) {
        *::service::lock(&self.kept) = Some(kept);
    }

    /// Holds later submissions until [`Self::resume`], whatever engine
    /// serves; the requests the engine serves go on. Returns whether the hold
    /// was paused already.
    pub fn pause(&self) -> bool {
        self.paused.send_replace(true)
    }

    /// Lets submissions through again.
    pub fn resume(&self) {
        self.paused.send_replace(false);
    }

    /// The hold's changes from now on ([`HoldChanges`]).
    pub(crate) fn changes(&self) -> HoldChanges {
        HoldChanges {
            serving: self.serving.subscribe(),
            paused: self.paused.subscribe(),
        }
    }

    /// Whether the hold is paused.
    pub fn is_paused(&self) -> bool {
        *self.paused.borrow()
    }

    /// What the hold does with a submission now, paused or not.
    pub fn state(&self) -> HoldState {
        match &*self.serving.borrow() {
            Serving::Engine(client) => HoldState::Engine(client.state()),
            Serving::Holding => HoldState::Holding,
            Serving::Remote(_) => HoldState::HandedOver,
            Serving::Closed(reason) => HoldState::Closed(reason.clone()),
        }
    }

    /// Has `client`'s engine serve every held and later submission.
    pub fn serve(&self, client: Arc<EngineClient>) {
        self.serving.send_replace(Serving::Engine(client));
    }

    /// Transfer continuation admission to the successor daemon. The old
    /// engine's watcher no longer owns this hold, but its streams stay open.
    pub(crate) fn hand_over(&self, remote: Arc<crate::restart::requests::RemoteHold>) {
        self.serving.send_replace(Serving::Remote(remote));
        self.paused.send_replace(false);
    }

    /// Holds later submissions until an engine serves again; the engine that
    /// served them so far stops receiving them.
    pub fn hold(&self) {
        self.serving.send_replace(Serving::Holding);
    }

    /// Fails every held and later submission with `reason`.
    pub fn close(&self, reason: impl Into<String>) {
        self.serving.send_replace(Serving::Closed(reason.into()));
    }

    /// The engine that serves submissions, ready or not; `None` while the
    /// hold holds them or after it closed.
    pub fn client(&self) -> Option<Arc<EngineClient>> {
        match &*self.serving.borrow() {
            Serving::Engine(client) => Some(Arc::clone(client)),
            Serving::Holding | Serving::Closed(_) | Serving::Remote(_) => None,
        }
    }

    /// Whether `client`'s engine is the one that serves submissions.
    pub fn serves(&self, client: &Arc<EngineClient>) -> bool {
        matches!(&*self.serving.borrow(), Serving::Engine(serving) if Arc::ptr_eq(serving, client))
    }

    /// Whether a submission now goes straight to a ready engine.
    pub fn is_serving(&self) -> bool {
        !self.is_paused()
            && matches!(
                &*self.serving.borrow(),
                Serving::Engine(client) if client.state() == EngineState::Serving
            )
    }

    /// Submits `request` to the engine that serves, once one is ready, naming
    /// its scheduler `scheduler`, and returns what `EngineClient::submit`
    /// returns.
    ///
    /// Waits while no engine is ready: while the hold holds submissions, and
    /// while its engine is not ready, until the hold changes.
    ///
    /// # Errors
    ///
    /// Fails when the hold closes, or is dropped, before an engine serves the
    /// request, and as `EngineClient::submit` fails.
    pub async fn submit(
        &self,
        request: Request,
        scheduler: Option<u32>,
    ) -> Result<(EventStream, Option<u32>), SubmitError> {
        self.ready(None).await?.0.submit(request, scheduler).await
    }

    /// Waits for this replica to accept new requests. The deployment router
    /// waits on all replicas when none is ready, without polling.
    pub(crate) async fn ready(
        &self,
        previous_epoch: Option<u64>,
    ) -> Result<(Arc<EngineClient>, Option<Kept>), SubmitError> {
        let (destination, kept) = self.destination(previous_epoch).await?;
        match destination {
            Destination::Engine(client) => Ok((client, kept)),
            Destination::Remote(remote) => {
                let (client, remote_kept) = remote.ready(previous_epoch).await?;
                Ok((client, kept.or(remote_kept)))
            }
        }
    }

    /// Attach a predecessor daemon to the current engine: the engine dials the
    /// predecessor's request socket `requests` unless the predecessor holds a
    /// connection to it already. A further handoff forwards the request to the
    /// final owner.
    pub(crate) async fn attach(
        &self,
        requests: String,
        known_engine: Option<u64>,
        previous_epoch: Option<u64>,
    ) -> Result<crate::restart::requests::Attachment, SubmitError> {
        let (destination, kept) = self.destination(previous_epoch).await?;
        match destination {
            Destination::Engine(client) => {
                let engine = client.engine().ok_or_else(|| {
                    SubmitError::Unavailable("the serving engine has no process identity".into())
                })?;
                // The engine numbers each connection's requests apart, so the
                // predecessor's connection numbers its own.
                if known_engine != Some(engine) {
                    client.attach(&requests);
                }
                Ok(crate::restart::requests::Attachment {
                    engine,
                    epoch: client.epoch(),
                    kept,
                })
            }
            Destination::Remote(remote) => {
                let mut attachment = remote
                    .attach(requests, known_engine, previous_epoch)
                    .await?;
                attachment.kept = kept.or(attachment.kept);
                Ok(attachment)
            }
        }
    }

    async fn destination(
        &self,
        previous_epoch: Option<u64>,
    ) -> Result<(Destination, Option<Kept>), SubmitError> {
        let mut serving = self.serving.subscribe();
        let mut paused = self.paused.subscribe();
        loop {
            let held = *paused.borrow_and_update();
            let current = serving.borrow_and_update().clone();
            match current {
                Serving::Engine(client) if !held && client.state() == EngineState::Serving => {
                    return Ok((
                        Destination::Engine(client),
                        previous_epoch.and_then(|epoch| self.kept(epoch)),
                    ));
                }
                Serving::Remote(remote) if !held => {
                    return Ok((
                        Destination::Remote(remote),
                        previous_epoch.and_then(|epoch| self.kept(epoch)),
                    ));
                }
                Serving::Closed(reason) => return Err(SubmitError::Unavailable(reason)),
                // An engine that is not ready serves nothing until whoever
                // watches it replaces it or closes the hold.
                Serving::Engine(_) | Serving::Holding | Serving::Remote(_) => {}
            }
            let changed = tokio::select! {
                changed = serving.changed() => changed,
                changed = paused.changed() => changed,
            };
            if changed.is_err() {
                return Err(SubmitError::Unavailable(
                    "the daemon stopped serving requests".to_owned(),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proto::request::{Event, GenerationRequest, SamplingParams};

    use super::*;
    use crate::test_support::{
        SERVED_MODEL, engine_info, fake_client, scripted_client_with, text_fake,
    };

    /// A text request that generates until the fake's end of text.
    fn request() -> Request {
        GenerationRequest {
            trace: 0,
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            sampling: SamplingParams::default(),
            max_und_tokens: 64,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            generated_token_ids: Vec::new(),
            adapter: 0,
            hand_off: false,
            lend: None,
        }
        .into()
    }

    /// A client of one scheduler over the default fake.
    fn client() -> Arc<EngineClient> {
        Arc::new(fake_client(
            testkit::args(SERVED_MODEL),
            testkit::Fixture::new(text_fake),
        ))
    }

    /// A submission waits while no engine serves, and the engine that comes
    /// to serve accepts it.
    #[tokio::test]
    async fn a_held_request_is_served_once_an_engine_is_ready() {
        let hold = RequestHold::new();
        let submission = hold.submit(request(), None);
        tokio::pin!(submission);
        assert!(
            futures::poll!(&mut submission).is_pending(),
            "a held request waits"
        );
        assert!(!hold.is_serving());

        hold.serve(client());
        let (mut events, scheduler) = submission.await.expect("the request is served");
        assert_eq!(scheduler, Some(0));
        let mut finished = false;
        while let Some(event) = events.recv().await {
            finished |= matches!(event, Event::Finished { .. });
        }
        assert!(finished, "the served request finishes");
    }

    /// Closing the hold fails the requests it holds and every later one.
    #[tokio::test]
    async fn closing_the_hold_fails_held_and_later_requests() {
        let hold = RequestHold::new();
        let submission = hold.submit(request(), None);
        tokio::pin!(submission);
        assert!(futures::poll!(&mut submission).is_pending());

        hold.close("no engine serves");
        let expected = SubmitError::Unavailable("no engine serves".to_owned());
        assert_eq!(submission.await.err(), Some(expected.clone()));
        assert_eq!(hold.submit(request(), None).await.err(), Some(expected));
    }

    /// Requests submitted while the engine is withdrawn wait for the engine
    /// that serves next.
    #[tokio::test]
    async fn a_withdrawn_engine_receives_no_request() {
        let hold = RequestHold::new();
        hold.serve(client());
        assert!(hold.is_serving());

        hold.hold();
        let submission = hold.submit(request(), None);
        tokio::pin!(submission);
        assert!(
            futures::poll!(&mut submission).is_pending(),
            "a withdrawn engine takes nothing"
        );

        hold.serve(client());
        let (_events, scheduler) = submission.await.expect("the next engine serves");
        assert_eq!(scheduler, Some(0));
    }

    /// A paused hold keeps its engine but holds submissions until resumed.
    #[tokio::test]
    async fn a_paused_hold_holds_submissions_until_resumed() {
        let hold = RequestHold::new();
        hold.serve(client());
        assert!(!hold.pause(), "the hold was not paused before");
        assert!(hold.pause(), "a second pause finds it paused");
        assert!(!hold.is_serving());
        assert_eq!(hold.state(), HoldState::Engine(EngineState::Serving));

        let submission = hold.submit(request(), None);
        tokio::pin!(submission);
        assert!(
            futures::poll!(&mut submission).is_pending(),
            "a paused hold holds requests"
        );

        hold.resume();
        let (_events, scheduler) = submission.await.expect("the engine serves");
        assert_eq!(scheduler, Some(0));
        assert!(hold.is_serving());
    }

    /// Requests leaving an engine keep the state it computed where the next
    /// engine computes other pages or trajectories, and nothing where both
    /// compute the same; what an epoch keeps reaches only the requests that
    /// leave that epoch.
    #[test]
    fn requests_keep_the_state_of_the_epoch_they_leave_where_the_next_differs() {
        let engine = |epoch, seed: u8, latent_seed| {
            let mut info = engine_info();
            info.prefix_seeds = vec![[seed; 32]];
            info.latent_seed = latent_seed;
            scripted_client_with(epoch, info, |_| Vec::new()).0
        };
        let first = engine(1, 1, 5);
        assert_eq!(Kept::between(&first, &engine(2, 1, 5)), None);
        let kept = Kept::between(&first, &engine(2, 2, 5)).expect("the pages differ");
        assert_eq!(
            kept,
            Kept {
                epoch: 1,
                prefix_seeds: Some(vec![[1; 32]]),
                latent_seed: None,
            }
        );
        assert_eq!(
            Kept::between(&first, &engine(2, 1, 6)),
            Some(Kept {
                epoch: 1,
                prefix_seeds: None,
                latent_seed: Some(5),
            })
        );

        let hold = RequestHold::new();
        hold.keep(kept.clone());
        assert_eq!(hold.kept(1), Some(kept));
        assert_eq!(hold.kept(2), None);
    }
}
