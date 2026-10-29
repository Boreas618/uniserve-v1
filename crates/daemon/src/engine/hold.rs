//! Holding requests while no engine serves.
//!
//! The leader submits every request through its [`RequestHold`]. While an
//! engine is ready, a submission goes straight to it. While none is, a
//! submission waits until one is, instead of being refused; once the hold
//! closes, because no engine will serve again, held and new submissions fail.
//! While the hold is paused (`/admin/pause_generation`), submissions wait
//! as well, and the requests the engine already serves go on.
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
    /// A group reshape transferred this replica's requests to these owners.
    Successor(Arc<RequestHold>),
    /// The successor daemon owns readiness and engine recovery.
    Remote(Arc<crate::restart::requests::RemoteHold>),
    /// No engine will serve; submissions fail with the reason.
    Closed(String),
}

/// The leader's submissions and the engine that serves them.
pub struct RequestHold {
    serving: watch::Sender<Serving>,
    /// Whether submissions wait although an engine serves.
    paused: watch::Sender<bool>,
    /// The state the requests migrating out of a world continue with
    /// (`Self::kept`).
    kept: std::sync::Mutex<Option<Kept>>,
}

/// The state the requests migrating out of world `world` continue with,
/// although the world that serves next computes other numerics: a world
/// change's `in_flight = keep`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Kept {
    pub world: u64,
    /// The prefix seed of the pages a token request continues with
    /// (`CachePolicy::state_seed`), when the worlds' pages differ.
    pub prefix_seed: Option<u64>,
    /// The latent seed of the trajectories a request continues with
    /// (`Trajectory::state_seed`: a video request's, or the image a token
    /// request had begun), when the worlds' trajectories differ.
    pub latent_seed: Option<u64>,
}

/// What a hold does with a submission now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoldState {
    /// It goes to an engine in this state.
    Engine(EngineState),
    /// It waits for an engine.
    Holding,
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

    /// Has the requests that migrate out of `kept.world` continue with the
    /// state they reached there: a world change's `in_flight = keep` when
    /// the world that serves next computes other numerics. `None` ends it.
    pub fn keep(&self, kept: Option<Kept>) {
        *self
            .kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = kept;
    }

    /// The state a request migrating out of world `world` continues with,
    /// if it keeps it.
    pub fn kept(&self, world: u64) -> Option<Kept> {
        self.kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .filter(|kept| kept.world == world)
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

    /// Whether the hold is paused.
    pub fn is_paused(&self) -> bool {
        *self.paused.borrow()
    }

    /// What the hold does with a submission now, paused or not.
    pub fn state(&self) -> HoldState {
        match &*self.serving.borrow() {
            Serving::Engine(client) => HoldState::Engine(client.state()),
            Serving::Holding => HoldState::Holding,
            Serving::Successor(_) | Serving::Remote(_) => HoldState::Holding,
            Serving::Closed(reason) => HoldState::Closed(reason.clone()),
        }
    }

    /// Has `client`'s engine serve every held and later submission.
    pub fn serve(&self, client: Arc<EngineClient>) {
        self.serving.send_replace(Serving::Engine(client));
    }

    /// Transfers continuations to a successor replica. Existing HTTP
    /// streams retain this hold, while each successor owns its later world
    /// changes. The caller must avoid cycles and restore predecessor holds
    /// before redirecting successors during rollback.
    pub(crate) fn redirect(&self, successor: Arc<RequestHold>) {
        self.serving.send_replace(Serving::Successor(successor));
        self.paused.send_replace(false);
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
            Serving::Holding | Serving::Closed(_) | Serving::Successor(_) | Serving::Remote(_) => {
                None
            }
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
                Serving::Engine(client) if client.state() == EngineState::Ready
            )
    }

    /// Submits `request` to the engine that serves, once one is ready, and
    /// returns what `EngineClient::submit` returns.
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
    ) -> Result<(EventStream, Option<u32>), SubmitError> {
        self.ready(None).await?.0.submit(request).await
    }

    /// Waits for this replica to accept new requests. The deployment router
    /// waits on all replicas when none is ready, without polling.
    pub(crate) async fn ready(
        &self,
        previous_world: Option<u64>,
    ) -> Result<(Arc<EngineClient>, Option<Kept>), SubmitError> {
        let (destination, kept) = self.destination(previous_world).await?;
        match destination {
            Destination::Engine(client) => Ok((client, kept)),
            Destination::Remote(remote) => {
                let (client, remote_kept) = remote.ready(previous_world).await?;
                Ok((client, kept.or(remote_kept)))
            }
        }
    }

    /// Attach a predecessor daemon to the current engine. A further handoff
    /// forwards the request to the final owner, which alone allocates IDs.
    pub(crate) async fn attach(
        &self,
        requests: String,
        known_engine: Option<u64>,
        previous_world: Option<u64>,
    ) -> Result<crate::restart::requests::Attachment, SubmitError> {
        let (destination, kept) = self.destination(previous_world).await?;
        match destination {
            Destination::Engine(client) => {
                let engine = client.engine().ok_or_else(|| {
                    SubmitError::Unavailable("the serving engine has no process identity".into())
                })?;
                let first_request = if known_engine == Some(engine) {
                    None
                } else {
                    let first = client.reserve_request_ids();
                    client.attach(&requests);
                    Some(first)
                };
                Ok(crate::restart::requests::Attachment {
                    engine,
                    world: client.world(),
                    first_request,
                    kept,
                })
            }
            Destination::Remote(remote) => {
                let mut attachment = remote
                    .attach(requests, known_engine, previous_world)
                    .await?;
                attachment.kept = kept.or(attachment.kept);
                Ok(attachment)
            }
        }
    }

    async fn destination(
        &self,
        previous_world: Option<u64>,
    ) -> Result<(Destination, Option<Kept>), SubmitError> {
        let mut serving = self.serving.subscribe();
        let mut paused = self.paused.subscribe();
        loop {
            let held = *paused.borrow_and_update();
            let current = serving.borrow_and_update().clone();
            match current {
                Serving::Engine(client) if !held && client.state() == EngineState::Ready => {
                    return Ok((
                        Destination::Engine(client),
                        previous_world.and_then(|world| self.kept(world)),
                    ));
                }
                Serving::Remote(remote) if !held => {
                    return Ok((
                        Destination::Remote(remote),
                        previous_world.and_then(|world| self.kept(world)),
                    ));
                }
                Serving::Closed(reason) => return Err(SubmitError::Unavailable(reason)),
                Serving::Successor(successor) if !held => {
                    tokio::select! {
                        changed = serving.changed() => {
                            changed.map_err(|_| SubmitError::Unavailable("the daemon stopped serving requests".into()))?;
                        }
                        ready = Box::pin(successor.destination(previous_world)) => {
                            let (destination, kept) = ready?;
                            return Ok((destination, previous_world.and_then(|world| self.kept(world)).or(kept)));
                        }
                    }
                    continue;
                }
                // An engine that is not ready serves nothing until whoever
                // watches it replaces it or closes the hold.
                Serving::Engine(_)
                | Serving::Holding
                | Serving::Successor(_)
                | Serving::Remote(_) => {}
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
    use std::time::Duration;

    use engine::{EngineConfig, SimEngine};
    use proto::request::{
        Event, GenerationConstraint, GenerationRequest, ImageGenerationConfig, ImageParams,
        SamplingParams,
    };

    use super::*;
    use crate::test_support::{SERVED_MODEL, sim_client, sim_core};

    /// A text request that generates until the simulator's EOS.
    fn request() -> Request {
        GenerationRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint: GenerationConstraint::UndOnly,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_images: 1,
            max_und_tokens: 64,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: ImageGenerationConfig::default(),
            readout: Vec::new(),
            generated_token_ids: Vec::new(),
            adapter: 0,
            generated_images: Vec::new(),
            trajectory: Default::default(),
        }
        .into()
    }

    /// A client of one simulated scheduler.
    fn client() -> Arc<EngineClient> {
        Arc::new(sim_client(vec![sim_core(
            EngineConfig::sim(SERVED_MODEL),
            SimEngine::new(),
        )]))
    }

    /// A submission waits while no engine serves, and the engine that comes
    /// to serve accepts it.
    #[tokio::test]
    async fn a_held_request_is_served_once_an_engine_is_ready() {
        let hold = Arc::new(RequestHold::new());
        let submission = tokio::spawn({
            let hold = Arc::clone(&hold);
            async move { hold.submit(request()).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!submission.is_finished(), "a held request waits");
        assert!(!hold.is_serving());

        hold.serve(client());
        let (mut events, scheduler) = submission.await.unwrap().expect("the request is served");
        assert_eq!(scheduler, Some(0));
        let mut finished = false;
        while let Some(event) = events.next().await {
            finished |= matches!(event, Event::Finished { .. });
        }
        assert!(finished, "the served request finishes");
    }

    /// Closing the hold fails the requests it holds and every later one.
    #[tokio::test]
    async fn closing_the_hold_fails_held_and_later_requests() {
        let hold = Arc::new(RequestHold::new());
        let submission = tokio::spawn({
            let hold = Arc::clone(&hold);
            async move { hold.submit(request()).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        hold.close("no engine serves");
        let expected = SubmitError::Unavailable("no engine serves".to_owned());
        assert_eq!(submission.await.unwrap().err(), Some(expected.clone()));
        assert_eq!(hold.submit(request()).await.err(), Some(expected));
    }

    /// Requests submitted while the engine is withdrawn wait for the engine
    /// that serves next.
    #[tokio::test]
    async fn a_withdrawn_engine_receives_no_request() {
        let hold = Arc::new(RequestHold::new());
        let first = client();
        hold.serve(Arc::clone(&first));
        assert!(hold.is_serving());

        hold.hold();
        let submission = tokio::spawn({
            let hold = Arc::clone(&hold);
            async move { hold.submit(request()).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !submission.is_finished(),
            "a withdrawn engine takes nothing"
        );

        hold.serve(client());
        let (_events, scheduler) = submission.await.unwrap().expect("the next engine serves");
        assert_eq!(scheduler, Some(0));
    }

    /// A paused hold keeps its engine but holds submissions until resumed.
    #[tokio::test]
    async fn a_paused_hold_holds_submissions_until_resumed() {
        let hold = Arc::new(RequestHold::new());
        hold.serve(client());
        assert!(!hold.pause(), "the hold was not paused before");
        assert!(hold.pause(), "a second pause finds it paused");
        assert!(!hold.is_serving());
        assert_eq!(hold.state(), HoldState::Engine(EngineState::Ready));

        let submission = tokio::spawn({
            let hold = Arc::clone(&hold);
            async move { hold.submit(request()).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!submission.is_finished(), "a paused hold holds requests");

        hold.resume();
        let (_events, scheduler) = submission.await.unwrap().expect("the engine serves");
        assert_eq!(scheduler, Some(0));
        assert!(hold.is_serving());
    }
}
