//! Continuation admission through the daemon that owns the deployment.
//!
//! Existing HTTP streams stay in their original daemon. Only their next
//! engine connection is obtained over the handoff channel; tokens and media
//! continue over the ordinary engine request protocol.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use proto::request::wire;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use super::{Channel, Message, send};
use crate::engine::client::{EngineClient, SubmitError};
use crate::engine::control::EngineControl;
use crate::engine::hold::Kept;

pub(super) type Writer = Arc<Mutex<Channel>>;

/// The engine a successor resolved for a predecessor's continuation.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Attachment {
    pub engine: u64,
    pub world: u64,
    /// A new front uses this interval; an existing front keeps its counter.
    pub first_request: Option<u64>,
    pub kept: Option<Kept>,
}

type Answer = Result<Attachment, String>;

pub(super) struct Requests {
    writer: Writer,
    next: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Answer>>>,
}

impl Requests {
    pub(super) fn new(writer: Writer) -> Arc<Self> {
        Arc::new(Self {
            writer,
            next: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
        })
    }

    pub(super) fn answer(&self, id: u64, answer: Answer) {
        if let Some(pending) = crate::leader::locked(&self.pending).remove(&id) {
            let _ = pending.send(answer);
        }
    }

    pub(super) fn close(&self, reason: String) {
        for (_, pending) in crate::leader::locked(&self.pending).drain() {
            let _ = pending.send(Err(reason.clone()));
        }
    }

    async fn attach(
        &self,
        replica: usize,
        requests: String,
        known_engine: Option<u64>,
        previous_world: Option<u64>,
    ) -> Result<Attachment, SubmitError> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (answer, answered) = oneshot::channel();
        crate::leader::locked(&self.pending).insert(id, answer);
        if let Err(error) = send(
            &mut crate::leader::locked(&self.writer),
            &Message::Resume {
                id,
                replica,
                requests,
                known_engine,
                previous_world,
            },
            &[],
        ) {
            crate::leader::locked(&self.pending).remove(&id);
            return Err(SubmitError::Unavailable(format!(
                "the successor daemon is unavailable: {error}"
            )));
        }
        answered
            .await
            .unwrap_or_else(|_| Err("the successor daemon closed its continuation relay".into()))
            .map_err(SubmitError::Unavailable)
    }
}

/// One replica's borrowed front and its admission owner across a handoff.
pub(crate) struct RemoteHold {
    requests: Arc<Requests>,
    replica: usize,
    control: Arc<EngineControl>,
    client: tokio::sync::Mutex<Arc<EngineClient>>,
}

impl RemoteHold {
    pub(super) fn new(
        requests: Arc<Requests>,
        replica: usize,
        control: Arc<EngineControl>,
        client: Arc<EngineClient>,
    ) -> Self {
        Self {
            requests,
            replica,
            control,
            client: tokio::sync::Mutex::new(client),
        }
    }

    pub(crate) async fn attach(
        &self,
        requests: String,
        known_engine: Option<u64>,
        previous_world: Option<u64>,
    ) -> Result<Attachment, SubmitError> {
        self.requests
            .attach(self.replica, requests, known_engine, previous_world)
            .await
    }

    pub(crate) async fn ready(
        &self,
        previous_world: Option<u64>,
    ) -> Result<(Arc<EngineClient>, Option<Kept>), SubmitError> {
        // Concurrent continuations share one front and its request counter.
        let mut client = self.client.lock().await;
        let attachment = self
            .attach(
                self.control.request_address().to_owned(),
                client.engine(),
                previous_world,
            )
            .await?;
        if let Some(first) = attachment.first_request {
            let control = Arc::clone(&self.control);
            let world = attachment.world;
            let engine = attachment.engine;
            let attached = tokio::task::spawn_blocking(move || {
                let client = control.adopt(
                    engine,
                    engine + 1,
                    wire::EngineSpec {
                        world,
                        ..Default::default()
                    },
                    None,
                    Box::new(|_, _, _| {}),
                )?;
                client.number_requests_from(first);
                Ok::<_, anyhow::Error>(client)
            })
            .await
            .map_err(|error| SubmitError::Unavailable(error.to_string()))?
            .map_err(|error| {
                SubmitError::Unavailable(format!("attaching the recovered engine: {error:#}"))
            })?;
            *client = Arc::new(attached);
        }
        Ok((Arc::clone(&client), attachment.kept))
    }
}

pub(super) fn writer(channel: &Channel) -> anyhow::Result<Writer> {
    let stream = match channel.stream() {
        storage::transport::Socket::Unix(stream) => stream.try_clone()?,
        storage::transport::Socket::Tcp(_) => anyhow::bail!("daemon handoff uses a Unix socket"),
    };
    Ok(Arc::new(Mutex::new(Channel::new(stream))))
}
