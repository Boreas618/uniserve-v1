//! The rank's connection to the head and its registration there.
//!
//! A rank connects to the head before it loads anything, so the head observes
//! a startup failure as the connection closing. Its first frame is its
//! [`Registration`]; the head answers with the rank's [`LaunchDescriptor`].
//! The same connection then carries requests and responses, and its close is
//! the rank's signal to exit.

use std::time::{Duration, Instant};

use uniserve_ipc::{Channel, LaunchDescriptor, Registration, VERSION, Wake};

use super::Arguments;
use crate::config::IpcConfig;
use crate::{Error, Result};

/// Bounds connecting to the head and receiving the launch descriptor; it does
/// not bound model startup.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(60);

/// The rank's connection to the head.
pub struct Connection {
    channel: Channel,
}

impl Connection {
    /// Connect to the head, register, and receive the launch descriptor.
    pub fn register(arguments: &Arguments) -> Result<(Self, LaunchDescriptor)> {
        let failed =
            |error| Error::Transport(format!("registering with {}: {error}", arguments.head));
        let mut channel =
            Channel::connect(&arguments.head, REGISTRATION_TIMEOUT).map_err(failed)?;
        channel
            .send(&Registration::Rank {
                worker_id: arguments.worker_id.clone(),
                rank: arguments.rank,
                generation: arguments.generation,
                version: VERSION.to_owned(),
            })
            .map_err(failed)?;

        // The head answers once every rank of the launch has registered. It
        // closes the connection instead when it refuses the registration.
        let deadline = Instant::now() + REGISTRATION_TIMEOUT;
        loop {
            if let Some(descriptor) = channel.try_recv().map_err(failed)? {
                return Ok((Self { channel }, descriptor));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::Transport(format!(
                    "the head at {} sent no launch descriptor",
                    arguments.head
                )));
            }
            channel.wait(remaining).map_err(failed)?;
        }
    }

    /// Size the channel for the batches and results the descriptor allows.
    pub fn configure(&mut self, ipc: &IpcConfig) {
        self.channel
            .set_capacity(ipc.max_payload_bytes, ipc.queue_depth);
    }

    pub fn channel(&mut self) -> &mut Channel {
        &mut self.channel
    }

    /// The wake completion callbacks fire to interrupt the service's waits.
    pub fn completion_wake(&self) -> Wake {
        self.channel.wake()
    }
}
