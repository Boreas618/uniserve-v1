//! The state of the API's routes: the served model, and this daemon's
//! connections, which write every response body and hand it to a successor
//! on a hot restart.

use std::sync::Arc;
use std::time::Duration;

use axum::http::header;
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
    /// How long a server-sent-events body may wait for its engine before a
    /// comment keeps its connection alive (`server_config.stream_keep_alive`).
    keep_alive: Duration,
}

impl Served {
    /// The routes' state over `leader`, whose response bodies register with
    /// `handoff`; a server-sent-events body writes a keep-alive comment after
    /// `keep_alive` without output.
    pub(crate) fn shared(
        leader: Arc<Leader>,
        handoff: Arc<Handoff>,
        keep_alive: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            leader,
            handoff,
            keep_alive,
        })
    }

    /// Makes `session` the body of `response`, opened as `opening`: this
    /// daemon's connection writes its frames, and a hot restart can hand it
    /// to a successor, which names the response `request_id`.
    ///
    /// The body's length is the `Content-Length` the route declared on
    /// `response`, which moves to the body: the connection states it in the
    /// head and holds the body to it. A response without one is chunked.
    ///
    /// # Errors
    ///
    /// Fails when the declared length is not a length.
    pub(crate) fn attach(
        &self,
        response: &mut Response,
        session: Box<dyn Session>,
        opening: Opening,
        request_id: String,
    ) -> Result<(), ApiError> {
        let length = response
            .headers_mut()
            .remove(header::CONTENT_LENGTH)
            .map(|value| {
                value
                    .to_str()
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .ok_or_else(|| {
                        ApiError::server_error(format!("a response declared the length {value:?}"))
                    })
            })
            .transpose()?;
        // The response registers before it writes a frame, so a hot restart
        // pauses it at its next engine event boundary.
        let registration = self.handoff.register();
        let stream = match opening {
            Opening::Held => ResponseStream::start_held(session),
            Opening::Streamed { keep_alive } => {
                ResponseStream::start(session, keep_alive.then_some(self.keep_alive))
            }
        };
        attach(
            response,
            HandedBody::new(stream, registration, request_id, length),
        );
        Ok(())
    }
}
