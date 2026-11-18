//! The admin listener: the Unix socket a node serves its admin API on, apart from the client API.
//!
//! The admin API has two endpoints, both the node's (`super::node`): `/admin/policy`, which reads
//! the deployment's policy package and writes the next one, the only way the deployment changes
//! while it serves; and `/admin/query`, which reads the observation store's `stats` and `events`
//! tables with SQL, the only way its state is read. What the deployment executes, owns and did
//! is rows of those tables (`crate::status`), and what an operator does to it is an option of its
//! package.
//!
//! Access is the socket's file permissions: its owner alone.

use axum::Router;
use axum::http::StatusCode;
use tokio_util::sync::CancellationToken;

use crate::leader::UpdateError;

/// The file mode of the admin socket: its owner alone.
const ADMIN_MODE: u32 = 0o600;

/// The admin socket, bound when the daemon starts and served for the node's whole life
/// ([`serve`]).
pub(crate) struct Listener {
    listener: super::listener::Listener,
    path: std::path::PathBuf,
}

impl Listener {
    /// Binds the admin socket `admin.sock` in the daemon's instance directory `instance`, with
    /// mode `0600`, taking its path over from whoever holds it (a hot restart's predecessor).
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be bound at its path.
    pub(crate) fn bind(instance: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context as _;

        let path = instance.join("admin.sock");
        let listener = super::listener::Listener::bind_over(&path, ADMIN_MODE)
            .with_context(|| format!("binding the admin socket {}", path.display()))?;
        Ok(Self { listener, path })
    }

    /// The socket's endpoint, as the readiness record states it.
    pub(crate) fn endpoint(&self) -> service::Endpoint {
        service::Endpoint::unix(&self.path)
    }
}

/// Serves `routes` on the admin socket until `shutdown` is cancelled, on a task of its own.
pub(crate) fn serve(listener: Listener, routes: Router, shutdown: &CancellationToken) {
    let Listener { listener, path } = listener;
    tracing::info!(socket = %path.display(), "serving the admin API");
    let stopped = shutdown.clone();
    tokio::spawn(async move {
        let served = axum::serve(listener, routes)
            .with_graceful_shutdown(async move { stopped.cancelled().await })
            .await;
        if let Err(error) = served {
            tracing::warn!(%error, "the admin listener ended");
        }
    });
}

/// The status a change of the deployment that failed is answered with.
pub(crate) fn update_status(error: &UpdateError) -> StatusCode {
    match error {
        UpdateError::InProgress => StatusCode::CONFLICT,
        UpdateError::Refused(_) => StatusCode::BAD_REQUEST,
        UpdateError::NoEngine => StatusCode::SERVICE_UNAVAILABLE,
        UpdateError::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}
