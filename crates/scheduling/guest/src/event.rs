//! What the mechanism tells a policy.
//!
//! The mechanism hands a policy the events since the previous step, in the
//! order they happened, once per step (`Policy::observe`). Events are
//! proportional to state changes, never to resident requests or cache pages:
//! an ordered structure such as an LRU list is the policy's own, maintained
//! from [`Event::Cached`], [`Event::Hit`] and [`Event::Freed`] (design §4.4).
//!
//! The list is append-only. A policy module older than the mechanism skips an
//! event kind it does not know: the wire decodes it into [`Event::Unknown`].

use serde::{Deserialize, Serialize};

use crate::ids::{PageId, PhaseId, RequestId};
use crate::time::{Duration, Nanos};
use crate::view::{EncodingView, PageView, RequestView};

/// How a finished request ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "end", rename_all = "snake_case")]
pub enum End {
    /// The request produced its whole response.
    Completed,
    /// The request left this main process and the frontend resubmits it, with
    /// the tokens it generated, to the epoch that serves next (design §2.2).
    Migratable,
    /// The mechanism rejected the request as over capacity.
    Rejected,
    /// The client cancelled it.
    Cancelled,
    /// It failed.
    Failed,
    /// An end kind this policy does not know.
    #[serde(other)]
    Unknown,
}

/// A state change the mechanism reports to the policy.
///
/// The variants are the interface of design §4.4 in its order. A new kind of
/// state change is appended here and the mechanism that records it is an epoch
/// layer change; a policy that does not know it ignores it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    /// A request reached the main process, before `route` decides its rank.
    Arrived { request: RequestView },
    /// The mechanism admitted a queued request.
    Admitted { request: RequestId },
    /// A resident request's view changed in a way the policy must see: its
    /// next phase changed, its prompt progress or cached prefix moved, its
    /// page count changed. Emitted before the `dispatch` of the step the
    /// change belongs to, and always after the event that caused it.
    Updated { request: RequestView },
    /// A call of the request advanced: its phase ran and produced `tokens`
    /// tokens (or latent units, for a media phase). One event per completed
    /// piece of work the policy put in a `Plan`.
    Advanced {
        request: RequestId,
        phase: PhaseId,
        tokens: u32,
    },
    /// The request emitted `tokens` to the client.
    Emitted {
        request: RequestId,
        tokens: u32,
        at: Nanos,
    },
    /// The request ended.
    Finished { request: RequestId, end: End },
    /// A scheduling step ended; the next `observe` begins at `step`.
    Stepped { step: u64, duration: Duration },
    /// A page entered the cache.
    Cached { page: PageView },
    /// A resident (device) page was used again.
    Hit { page: PageId },
    /// A page was freed.
    Freed { page: PageId },
    /// An encoder product entered the encoder cache.
    Encoded { entry: EncodingView },
    /// An event kind this policy does not know; skipped.
    #[serde(other)]
    Unknown,
}
