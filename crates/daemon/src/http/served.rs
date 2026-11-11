//! The served model as the API's routes reach it (`frontend::openai::Server`):
//! the leader's request facade, and this daemon's connections, which write
//! every response body and hand it to a successor on a hot restart.

use std::sync::Arc;

use axum::response::Response;
use futures::FutureExt as _;
use futures::future::BoxFuture;

use frontend::body::{Opening, Session};
use frontend::events::Events;
use frontend::input::video::{ProcessedVideo, VideoGenerationRequest};
use frontend::input::{AdapterSelection, Processor, ServeError};
use frontend::openai::{ApiError, Server};
use frontend::output::{OutputProcessor, RequestOutputStream};
use frontend::protocol::ChatCompletionRequest;

use crate::http::handoff::Handoff;
use crate::http::stream::{HandedBody, ResponseStream, attach};
use crate::leader::Leader;

/// The leader, with the registry a hot restart pauses its responses through.
pub(crate) struct Served {
    leader: Arc<Leader>,
    handoff: Arc<Handoff>,
}

impl Served {
    /// The routes' server over `leader`, whose response bodies register with
    /// `handoff`.
    pub(crate) fn shared(leader: Arc<Leader>, handoff: Arc<Handoff>) -> Arc<dyn Server> {
        Arc::new(Self { leader, handoff })
    }
}

impl Server for Served {
    fn served_model_name(&self) -> &str {
        self.leader.served_model_name()
    }

    fn adapter_names(&self) -> Vec<String> {
        self.leader.adapter_names()
    }

    fn adapter(&self, name: &str) -> Option<AdapterSelection> {
        self.leader.adapter(name)
    }

    fn processor(&self) -> &Processor {
        self.leader.processor()
    }

    fn generate(
        &self,
        request_id: String,
        request: ChatCompletionRequest,
        adapter: Option<AdapterSelection>,
    ) -> BoxFuture<'_, Result<(Box<dyn Events>, OutputProcessor), ApiError>> {
        self.leader.generate(request_id, request, adapter).boxed()
    }

    fn generate_video(
        &self,
        request_id: String,
        request: VideoGenerationRequest,
    ) -> BoxFuture<'_, Result<(ProcessedVideo, RequestOutputStream), ServeError>> {
        self.leader.generate_video(request_id, request).boxed()
    }

    fn attach(
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
