//! Holding requests while no engine serves.
//!
//! The leader submits every request through its [`RequestHold`]. While an
//! engine is ready, a submission goes straight to it. While none is, a
//! submission waits until one is, instead of being refused; once the hold
//! closes, because no engine will serve again, held and new submissions fail.
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
    /// No engine will serve; submissions fail with the reason.
    Closed(String),
}

/// The leader's submissions and the engine that serves them.
pub struct RequestHold {
    serving: watch::Sender<Serving>,
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
        }
    }

    /// Has `client`'s engine serve every held and later submission.
    pub fn serve(&self, client: Arc<EngineClient>) {
        self.serving.send_replace(Serving::Engine(client));
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
            Serving::Holding | Serving::Closed(_) => None,
        }
    }

    /// Whether `client`'s engine is the one that serves submissions.
    pub fn serves(&self, client: &Arc<EngineClient>) -> bool {
        matches!(&*self.serving.borrow(), Serving::Engine(serving) if Arc::ptr_eq(serving, client))
    }

    /// Whether a submission now goes straight to a ready engine.
    pub fn is_serving(&self) -> bool {
        matches!(
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
        let mut serving = self.serving.subscribe();
        loop {
            let current = serving.borrow_and_update().clone();
            match current {
                Serving::Engine(client) if client.state() == EngineState::Ready => {
                    return client.submit(request).await;
                }
                Serving::Closed(reason) => return Err(SubmitError::Unavailable(reason)),
                // An engine that is not ready serves nothing until whoever
                // watches it replaces it or closes the hold.
                Serving::Engine(_) | Serving::Holding => {}
            }
            if serving.changed().await.is_err() {
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
}
