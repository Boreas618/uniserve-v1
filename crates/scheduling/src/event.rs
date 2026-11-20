//! What the mechanism tells a policy.
//!
//! The mechanism hands a policy the events since the previous step, in the
//! order they happened, once per step (`Policy::observe`). Events are
//! proportional to state changes, never to running requests or cache pages:
//! an ordered structure such as an LRU list is the policy's own, maintained
//! from [`Event::Cached`], [`Event::Hit`], [`Event::Released`],
//! [`Event::Held`] and [`Event::Freed`] (design §4.4), and a request's view is
//! the last one [`Event::Arrived`] or [`Event::Updated`] stated, advanced by
//! [`Event::Admitted`] and [`Event::Emitted`].
//!
//! An event is encoded in serde's externally tagged form, an object of one
//! member that names its kind and holds its fields (`{"hit": {"page": 7}}`),
//! which a reader decodes straight from the message.

use serde::{Deserialize, Serialize};

use crate::ids::{PageId, PhaseId, RequestId};
use crate::time::{Duration, Nanos};
use crate::view::{EncodingView, PageView, RequestView};

/// How a finished request ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
}

/// A state change the mechanism reports to the policy, in the order of design §4.4's list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// A request reached its scheduler, which the leader's routing program chose.
    Arrived { request: RequestView },
    /// The mechanism admitted a waiting request.
    Admitted { request: RequestId },
    /// A request's view changed in a way the policy must see: its next phase
    /// changed, its prompt progress or cached prefix moved, its page count
    /// changed. Emitted before the `dispatch` of the step the change belongs
    /// to, and always after the event that caused it.
    ///
    /// `next_tokens`, `cached_pages` and `waiting_transfer` move
    /// without any other event carrying them, and a policy cannot derive
    /// them. The event is emitted only when the view differs from the one the
    /// policy holds by the events before it (the last view stated, advanced
    /// by `Admitted` and `Emitted`), so a request whose step moved only what
    /// those events carry is not restated, and the event count stays
    /// proportional to the step's state changes. The view states no
    /// `last_emitted_at`: the policy keeps the one `Emitted` gave.
    Updated { request: RequestView },
    /// A call of the request advanced: its phase ran and produced `tokens`
    /// tokens (or latent units, for a media phase). One event per completed
    /// piece of work the policy put in a `DispatchOutput`.
    Advanced {
        request: RequestId,
        phase: PhaseId,
        tokens: u32,
    },
    /// The request emitted `tokens` to the client: it generated as many more
    /// tokens (`RequestView::generated_tokens`), may generate as many fewer
    /// (`RequestView::remaining_tokens`, at least 0) and last emitted `at`.
    /// A running request's every generated token is emitted, so a policy
    /// keeps both counts exact from this event, and no `Updated` restates
    /// them.
    Emitted {
        request: RequestId,
        tokens: u32,
        at: Nanos,
    },
    /// The request ended.
    Finished { request: RequestId, end: End },
    /// A scheduling step ended; the next `observe` begins at `step`.
    Stepped { step: u64, duration: Duration },
    /// A page entered the cache, or its tier or whether a request holds it
    /// changed other than by a release or a hold: the page as it stands now.
    ///
    /// It also states a page the cache held when the instance started that
    /// its snapshot did not carry (`Snapshot::pages`): such a page was last
    /// used no later than every page the snapshot carried, and the instance's
    /// first steps state those pages least recently used first, each before
    /// any other event names it.
    Cached { page: PageView },
    /// A device page was used again by one more request; whether a request
    /// holds it did not change. The page is one the snapshot or a `Cached`
    /// stated.
    Hit { page: PageId },
    /// A page was freed.
    Freed { page: PageId },
    /// An encoder product entered the encoder cache.
    Encoded { entry: EncodingView },
    /// Pages no request holds any more, in the order they joined the eviction
    /// order, least recently used first: each is evictable from now on and was
    /// used just now, and nothing else of it changed. Each is a page the
    /// snapshot or a `Cached` stated. One event lists every page that
    /// consecutive releases gave up, so a request that ends holding many blocks
    /// costs one event rather than one restated page per block.
    Released { pages: Vec<PageId> },
    /// Pages no request held and a request holds again, in the order they
    /// were taken: each leaves the eviction order, is not evictable from now
    /// on and was used just now, and nothing else of it changed. Each is a page
    /// the snapshot or a `Cached` stated. The opposite of [`Event::Released`]:
    /// an admission that reuses many cached blocks costs one event rather than
    /// one restated page per block.
    Held { pages: Vec<PageId> },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::EncodingId;
    use crate::view::{RequestClass, RequestStatus};
    use crate::wire;

    fn request() -> RequestView {
        RequestView {
            id: RequestId(3),
            state: RequestStatus::Running,
            phase: Some(PhaseId(4)),
            priority: -2,
            arrived_at: Nanos(17),
            admitted_at: Some(Nanos(19)),
            prompt_tokens: 512,
            prefill_tokens: 512,
            generated_tokens: 9,
            cached_prefix_tokens: 64,
            prefilled_tokens: 512,
            cached_pages: 9,
            last_emitted_at: Some(Nanos(23)),
            class: RequestClass::Token,
            waiting_transfer: true,
            next_tokens: 1,
            reservation_units: 24,
            prefetch_tokens: 256,
            refill_tokens: Some(7),
            remaining_tokens: 1015,
            migratable: true,
        }
    }

    fn page(id: u64, evictable: bool) -> PageView {
        PageView {
            id: PageId(id),
            tokens: 64,
            last_used: Nanos(41),
            on_host: false,
            evictable,
        }
    }

    /// One event of every kind.
    fn every_kind() -> Vec<Event> {
        vec![
            Event::Arrived { request: request() },
            Event::Admitted {
                request: RequestId(3),
            },
            Event::Updated { request: request() },
            Event::Advanced {
                request: RequestId(3),
                phase: PhaseId(4),
                tokens: 2,
            },
            Event::Emitted {
                request: RequestId(3),
                tokens: 2,
                at: Nanos(29),
            },
            Event::Finished {
                request: RequestId(3),
                end: End::Cancelled,
            },
            Event::Stepped {
                step: 31,
                duration: Duration(37),
            },
            Event::Cached {
                page: page(u64::MAX, true),
            },
            Event::Hit { page: PageId(43) },
            Event::Freed { page: PageId(47) },
            Event::Encoded {
                entry: EncodingView {
                    id: EncodingId(53),
                    last_used: Nanos(59),
                },
            },
            Event::Released {
                pages: vec![PageId(61), PageId(u64::MAX)],
            },
            Event::Held {
                pages: vec![PageId(71)],
            },
        ]
    }

    /// Every event the mechanism encodes decodes to the same event in a step's list.
    #[test]
    fn every_kind_of_event_decodes_to_what_was_encoded() {
        let events = every_kind();
        let decoded: Vec<Event> = wire::decode(&wire::encode(&events).unwrap()).unwrap();
        assert_eq!(decoded, events);
    }

    /// A kind this release does not have, or a kind that lacks a field, is refused.
    #[test]
    fn an_unknown_kind_and_a_missing_field_are_refused() {
        assert!(wire::decode::<Event>(br#"{"evicted":{"page":7}}"#).is_err());
        assert!(wire::decode::<Event>(br#"{"hit":{}}"#).is_err());
    }
}
