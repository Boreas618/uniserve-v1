//! The state of the API's routes: the served model, and this daemon's
//! connections, which write every response body and hand it to a successor
//! on a hot restart.

use std::sync::Arc;

use axum::response::Response;

use crate::api::body::{Opening, Session};
use crate::api::openai::ApiError;
use crate::http::handoff::Handoff;
use crate::http::stream::{HandedBody, ResponseStream, attach};
use crate::leader::Leader;

/// The leader, with the registry a hot restart pauses its responses through.
pub(crate) struct Served {
    /// The served model, which the routes submit requests to.
    pub(crate) leader: Arc<Leader>,
    handoff: Arc<Handoff>,
}

impl Served {
    /// The routes' state over `leader`, whose response bodies register with
    /// `handoff`.
    pub(crate) fn shared(leader: Arc<Leader>, handoff: Arc<Handoff>) -> Arc<Self> {
        Arc::new(Self { leader, handoff })
    }

    /// Makes `session` the body of `response`, opened as `opening`: this
    /// daemon's connection writes its frames, and a hot restart can hand it
    /// to a successor, which names the response `request_id`.
    ///
    /// # Errors
    ///
    /// Fails when the body cannot be registered for a handoff.
    pub(crate) fn attach(
        &self,
        response: &mut Response,
        session: Box<dyn Session>,
        opening: Opening,
        request_id: String,
    ) -> Result<(), ApiError> {
        // The response registers before it writes a frame, so a hot restart
        // pauses it at its next engine event boundary.
        let registration = self.handoff.register();
        let stream = match opening {
            Opening::Held => ResponseStream::start_held(session, false),
            Opening::Streamed { keep_alive } => ResponseStream::start(session, keep_alive),
        };
        let body = HandedBody::new(stream, Some(registration), request_id)
            .ok_or_else(|| ApiError::server_error("the response cannot be handed over"))?;
        attach(response, body);
        Ok(())
    }
}
