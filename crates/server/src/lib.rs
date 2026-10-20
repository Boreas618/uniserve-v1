//! UniServe HTTP frontend, model profiles, request lowering, and output assembly.
//!
//! The crate serves one option tree (`options::ServeConfig`). [`Engine`] is
//! the serving engine: it loads the model, starts one engine core per
//! data-parallel rank, and turns each request into a stream of public events
//! through the model's input processing and output assembly (`serving`). The
//! HTTP layer (`http`) exposes OpenAI-compatible routes over it.

/// The serving engine over the in-process engine cores.
pub mod engine;
/// HTTP listener, middleware, and route construction.
pub mod http;
/// OpenAI-compatible schemas and conversion helpers.
pub mod openai;
/// Serving options: the command line and the configuration file.
pub mod options;
pub mod profile;
pub mod serving;
mod state;
mod video_jobs;

use std::sync::Arc;

use anyhow::Result;

pub use crate::engine::Engine;
pub use crate::http::{ApiError, build_router, serve};
use crate::options::{ServeConfig, Stated};
pub use crate::profile::ModelDescription;
pub use crate::serving::chat::ChatTemplateContentFormatOption;
pub use crate::state::AppState;
pub use uniserve_engine::SchedulingPolicy;

/// Starts the engine and wraps it in the HTTP layer's shared state.
///
/// `config` must have passed [`ServeConfig::validate`]; `stated` holds the
/// option paths the operator set.
///
/// # Errors
///
/// Fails as [`Engine::start`] does.
pub async fn build_state(config: &ServeConfig, stated: &Stated) -> Result<Arc<AppState>> {
    let engine = Engine::start(config, stated).await?;
    let server = &config.server_config;
    Ok(Arc::new(
        AppState::new(engine)
            .with_request_id_headers(server.request_id_headers)
            .with_api_key(server.api_key.clone())
            .with_request_timeout(server.request_timeout)
            .with_max_concurrent_requests(server.max_concurrent_requests.map(|limit| limit.get()))
            .with_video_body_limit(config.media_config.body_limit())
            .with_video_jobs(config.video_job_config.clone()),
    ))
}
