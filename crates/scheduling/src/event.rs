//! What the mechanism tells a policy.
//!
//! The mechanism hands a policy the events since the previous step, in the
//! order they happened, once per step (`Policy::observe`). Events are
//! proportional to state changes, never to running requests or cache pages:
//! an ordered structure such as an LRU list is the policy's own, maintained
//! from [`Event::Cached`], [`Event::Hit`], [`Event::Released`] and
//! [`Event::Freed`] (design §4.4).
//!
//! The list is append-only. A policy module older than the mechanism skips an
//! event kind it does not know: the wire decodes it into [`Event::Unknown`].
//!
//! # Decoding
//!
//! An event is an object whose `kind` member names its variant (serde's
//! internally tagged form), and [`Serialize`] writes `kind` first. Serde's
//! derived decoder for that form holds every member of the object before it
//! reads the kind, which made the holding most of what decoding a step's events
//! cost a policy. [`Event`]'s decoder reads the kind first and then the
//! variant's members straight from the message; it holds an object's members
//! as JSON only when the kind is not the first of them.

use std::borrow::Cow;
use std::fmt;

use serde::de::value::{MapDeserializer, StrDeserializer};
use serde::de::{
    self, DeserializeSeed, Deserializer, EnumAccess, IgnoredAny, MapAccess, Unexpected,
    VariantAccess, Visitor,
};
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
///
/// It is encoded by serde's derive and decoded by the module's own reader (see
/// the module documentation), which reads the same objects.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    /// A request reached the main process, before `route` decides its rank.
    Arrived { request: RequestView },
    /// The mechanism admitted a waiting request.
    Admitted { request: RequestId },
    /// A running request's view changed in a way the policy must see: its
    /// next phase changed, its prompt progress or cached prefix moved, its
    /// page count changed. Emitted before the `dispatch` of the step the
    /// change belongs to, and always after the event that caused it.
    ///
    /// This kind is an addition to §4.4's list, which allows a mechanism to
    /// append events it needs: `next_tokens`, `remaining_tokens`,
    /// `cached_pages`, `cache_rank` and `waiting_transfer` move without any
    /// other event carrying them, and a policy cannot derive them. It is
    /// emitted only when the view actually moved, so the event count stays
    /// proportional to the step's state changes.
    Updated { request: RequestView },
    /// A call of the request advanced: its phase ran and produced `tokens`
    /// tokens (or latent units, for a media phase). One event per completed
    /// piece of work the policy put in a `DispatchOutput`.
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
    /// A page entered the cache, or its tier or whether a request holds it
    /// changed other than by a release: the page as it stands now.
    Cached { page: PageView },
    /// A device page was used again.
    Hit { page: PageId },
    /// A page was freed.
    Freed { page: PageId },
    /// An encoder product entered the encoder cache.
    Encoded { entry: EncodingView },
    /// Pages no request holds any more, in the order they joined the eviction
    /// order, least recently used first: each is evictable from now on and was
    /// used just now, and nothing else of it changed. One event lists every
    /// page that consecutive releases gave up, so a request that ends holding
    /// many blocks costs one event rather than one restated page per block.
    Released { pages: Vec<PageId> },
    /// An event kind this policy does not know; skipped.
    #[serde(other)]
    Unknown,
}

/// The member of an event object that names its kind.
const TAG: &str = "kind";

impl<'de> Deserialize<'de> for Event {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(EventVisitor)
    }
}

/// Reads one event object: its kind, then the members of the variant it names.
struct EventVisitor;

impl<'de> Visitor<'de> for EventVisitor {
    type Value = Event;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an event: an object whose `kind` names it")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Event, A::Error> {
        let Some(Name(first)) = map.next_key()? else {
            return Err(de::Error::missing_field(TAG));
        };

        // The kind first, as `Serialize` writes it: the members after it are
        // the variant's, read where they lie in the message.
        if first == TAG {
            let Name(kind) = map.next_value()?;
            return Variants::deserialize(Members { kind: &kind, map });
        }

        // Another member first: the members are held as JSON until the kind
        // is found among them.
        let mut members = serde_json::Map::new();
        members.insert(first.into_owned(), map.next_value()?);
        while let Some((name, value)) = map.next_entry::<String, serde_json::Value>()? {
            members.insert(name, value);
        }
        let kind = match members.remove(TAG) {
            Some(serde_json::Value::String(kind)) => kind,
            Some(other) => {
                return Err(de::Error::invalid_type(
                    Unexpected::Other(&other.to_string()),
                    &"the name of a kind of event",
                ));
            }
            None => return Err(de::Error::missing_field(TAG)),
        };
        let map = MapDeserializer::<_, serde_json::Error>::new(members.into_iter());
        Variants::deserialize(Members { kind: &kind, map }).map_err(de::Error::custom)
    }
}

/// A member name or a kind, borrowed from the message when it holds no escape.
#[derive(Deserialize)]
#[serde(transparent)]
struct Name<'a>(#[serde(borrow)] Cow<'a, str>);

/// [`Event`]'s variants as serde's derive reads an externally tagged enum: the
/// kind names the variant and the members are its fields, which is how
/// [`EventVisitor`] reads the members that follow an object's kind. The derive
/// builds [`Event`] itself (`remote`), so a field that differs from
/// [`Event`]'s does not compile.
#[derive(Deserialize)]
#[serde(remote = "Event", rename_all = "snake_case")]
enum Variants {
    Arrived {
        request: RequestView,
    },
    Admitted {
        request: RequestId,
    },
    Updated {
        request: RequestView,
    },
    Advanced {
        request: RequestId,
        phase: PhaseId,
        tokens: u32,
    },
    Emitted {
        request: RequestId,
        tokens: u32,
        at: Nanos,
    },
    Finished {
        request: RequestId,
        end: End,
    },
    Stepped {
        step: u64,
        duration: Duration,
    },
    Cached {
        page: PageView,
    },
    Hit {
        page: PageId,
    },
    Freed {
        page: PageId,
    },
    Encoded {
        entry: EncodingView,
    },
    Released {
        pages: Vec<PageId>,
    },
    Unknown,
}

/// The kind of an event object and its members after the kind, presented to
/// [`Variants`] as the variant the kind names and that variant's fields.
struct Members<'k, A> {
    kind: &'k str,
    map: A,
}

impl<'de, A: MapAccess<'de>> Deserializer<'de> for Members<'_, A> {
    type Error = A::Error;

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, A::Error> {
        // A kind this revision does not know is the unknown event, whose
        // members are skipped.
        let kind = if variants.contains(&self.kind) {
            self.kind
        } else {
            "unknown"
        };
        visitor.visit_enum(Members {
            kind,
            map: self.map,
        })
    }

    fn deserialize_any<V: Visitor<'de>>(self, _visitor: V) -> Result<V::Value, A::Error> {
        Err(de::Error::custom(
            "an event's members are read as the variant its kind names",
        ))
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct identifier ignored_any
    }
}

impl<'de, A: MapAccess<'de>> EnumAccess<'de> for Members<'_, A> {
    type Error = A::Error;
    type Variant = Self;

    fn variant_seed<S: DeserializeSeed<'de>>(self, seed: S) -> Result<(S::Value, Self), A::Error> {
        let variant = seed.deserialize(StrDeserializer::<A::Error>::new(self.kind))?;
        Ok((variant, self))
    }
}

impl<'de, A: MapAccess<'de>> VariantAccess<'de> for Members<'_, A> {
    type Error = A::Error;

    /// The unknown event: whatever members it has are skipped.
    fn unit_variant(mut self) -> Result<(), A::Error> {
        while self.map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(())
    }

    fn newtype_variant_seed<S: DeserializeSeed<'de>>(self, _seed: S) -> Result<S::Value, A::Error> {
        Err(de::Error::invalid_type(
            Unexpected::NewtypeVariant,
            &"the members of an event",
        ))
    }

    fn tuple_variant<V: Visitor<'de>>(
        self,
        _len: usize,
        _visitor: V,
    ) -> Result<V::Value, A::Error> {
        Err(de::Error::invalid_type(
            Unexpected::TupleVariant,
            &"the members of an event",
        ))
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, A::Error> {
        visitor.visit_map(self.map)
    }
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
            cache_rank: Some(1),
        }
    }

    /// One event of every kind, with members that differ from their defaults.
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
                page: PageView {
                    id: PageId(u64::MAX),
                    tokens: 64,
                    last_used: Nanos(41),
                    on_host: true,
                    evictable: true,
                },
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
                pages: vec![PageId(61), PageId(u64::MAX), PageId(67)],
            },
            Event::Unknown,
        ]
    }

    /// The JSON an event encodes to, the form two events are compared in.
    fn json(event: &Event) -> serde_json::Value {
        serde_json::to_value(event).unwrap()
    }

    /// Every event the mechanism encodes decodes to the same event, alone and in a step's list.
    #[test]
    fn every_kind_of_event_decodes_to_what_was_encoded() {
        let events = every_kind();
        let decoded: Vec<Event> = wire::decode(&wire::encode(&events).unwrap()).unwrap();
        assert_eq!(decoded.len(), events.len());
        for (event, decoded) in events.iter().zip(&decoded) {
            assert_eq!(json(decoded), json(event));
        }
    }

    /// JSON states no order of an object's members: an event whose kind comes after its other
    /// members, or is written with an escape, decodes as the same event.
    #[test]
    fn an_event_reads_the_same_whatever_the_order_of_its_members() {
        for event in every_kind() {
            let serde_json::Value::Object(members) = json(&event) else {
                panic!("an event is an object");
            };
            let mut reordered = serde_json::Map::new();
            for (name, value) in members.iter().filter(|(name, _)| *name != TAG) {
                reordered.insert(name.clone(), value.clone());
            }
            reordered.insert(TAG.to_owned(), members[TAG].clone());
            let text = serde_json::to_string(&reordered).unwrap();
            let decoded: Event = wire::decode(text.as_bytes()).unwrap();
            assert_eq!(json(&decoded), json(&event), "{text}");
        }

        let escaped = r#"{"kind":"rel\u0065ased","pages":[1,2]}"#;
        let decoded: Event = wire::decode(escaped.as_bytes()).unwrap();
        assert!(matches!(decoded, Event::Released { pages } if pages == [PageId(1), PageId(2)]));
    }

    /// A kind this revision does not know reads as the unknown event, wherever its kind stands
    /// and whatever members it carries; a kind it knows that lacks a member it requires is
    /// refused.
    #[test]
    fn an_unknown_kind_is_skipped_and_a_missing_member_refused() {
        for text in [
            r#"{"kind":"evicted","pages":[1],"at":{"tier":2}}"#,
            r#"{"pages":[1],"at":{"tier":2},"kind":"evicted"}"#,
        ] {
            let decoded: Event = wire::decode(text.as_bytes()).unwrap();
            assert!(matches!(decoded, Event::Unknown), "{text}");
        }

        let list: Vec<Event> =
            wire::decode(br#"[{"kind":"evicted","page":7},{"kind":"hit","page":7}]"#).unwrap();
        assert!(matches!(
            list[..],
            [Event::Unknown, Event::Hit { page: PageId(7) }]
        ));

        assert!(wire::decode::<Event>(br#"{"kind":"hit"}"#).is_err());
        assert!(wire::decode::<Event>(br#"{"page":7}"#).is_err());
    }
}
