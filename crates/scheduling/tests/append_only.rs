//! The wire between the mechanism and a policy is append-only (design §4.4):
//! the kinds of state change and the fields of views and decisions may be
//! added, never renamed or removed, and a policy built against one revision
//! runs against a mechanism built from a later one.
//!
//! The mechanism runs in the main process and each policy is a module the
//! Program Server builds, so a policy older than the mechanism it runs under is
//! the normal case after an upgrade, not an accident. These cases state the two
//! directions of that promise on the encoding itself, which is the only thing
//! both sides share.
#![cfg(test)]

use scheduling::guest::wire;
use scheduling::guest::{
    End, Event, Limits, Plan, RequestId, RequestState, RequestView, Room, Shortage, Snapshot,
    Storage, Victim, Work,
};

/// Decodes the way the mechanism decodes a policy's answer, and the way a
/// policy decodes a message from the mechanism.
fn decode<T: serde::de::DeserializeOwned>(json: &str) -> T {
    wire::decode(json.as_bytes()).expect("an append-only message decodes")
}

/// A state change kind this host does not know is ignored, not refused: a newer
/// mechanism's events reach a policy that predates them.
#[test]
fn an_event_kind_the_host_does_not_know_is_ignored() {
    let event: Event = decode(r#"{"kind":"evicted","page":7}"#);
    assert!(matches!(event, Event::Unknown));
}

/// A field appended to a view the host already knows does not fail the decode,
/// and the fields it does know keep their meaning.
#[test]
fn a_field_the_host_does_not_know_does_not_hide_the_ones_it_does() {
    let request: RequestView = decode(
        r#"{"id":3,"state":{"state":"resident"},"class":{"class":"token"},
            "prompt_tokens":64,"cached_prefix_tokens":32,"slo_class":2}"#,
    );
    assert_eq!(request.id, RequestId(3));
    assert_eq!(request.state, RequestState::Resident);
    assert_eq!(request.cached_prefix_tokens, 32);

    // The same on a decision a policy answers with.
    let plan: Plan = decode(
        r#"{"admit":[1],"work":[{"request":1,"phase":2,"tokens":16}],
            "admission_window":4}"#,
    );
    assert_eq!(plan.admit, vec![RequestId(1)]);
    assert_eq!(plan.work.len(), 1);
    assert_eq!(plan.work[0].tokens, 16);
}

/// A field a later mechanism appends and an older policy omits answers its
/// default: an answer built against the older revision still means what it
/// said.
#[test]
fn an_answer_that_omits_a_later_field_means_its_default() {
    // A policy that predates `prefetch` and `refill` says nothing about them,
    // which is "no deepening, no refill".
    let plan: Plan = decode(r#"{"admit":[],"reject":[2],"work":[]}"#);
    assert!(plan.prefetch.is_empty());
    assert!(plan.refill.is_empty());
    assert!(plan.offload.is_empty());

    let victims: Vec<Victim> = decode(r#"[{"page":1},{"request":2},{"encoding":3}]"#);
    assert_eq!(victims.len(), 3);
    assert_eq!(victims[0].page(), Some(scheduling::guest::PageId(1)));

    let limits: Limits = decode(r#"{"chunked_prefill_size":2048,"max_batch":8}"#);
    assert_eq!(limits.chunked_prefill_size, 2048);
    assert_eq!(
        limits.max_num_batched_tokens,
        Limits::default().max_num_batched_tokens
    );
}

/// The snapshot is a set of lists, and a list may grow at its end: a policy
/// instantiates from a snapshot whose views carry a field it does not know.
#[test]
fn a_snapshot_survives_a_field_appended_to_a_view() {
    let snapshot: Snapshot = decode(
        r#"{"step":4,
            "requests":[{"id":1,"state":{"state":"queued"},"class":{"class":"token"},
                         "waiting_transfer":false,"affinity":7}],
            "pages":[{"id":9,"tokens":16,"last_used":0,"on_host":false,"evictable":true,
                      "tier":1}],
            "ranks":[{"index":0,"in_flight":0,"alive":true}]}"#,
    );
    assert_eq!(snapshot.step, 4);
    assert_eq!(snapshot.requests.len(), 1);
    assert_eq!(snapshot.pages[0].tokens, 16);
    assert_eq!(snapshot.ranks[0].index, 0);
}

/// A shortage, an end and the room of a step carry their known parts through,
/// with or without fields this revision does not have: the mechanism reads what
/// it understands and refuses the rest by its own rules, never by a decode
/// failure.
#[test]
fn a_shortage_an_end_and_a_room_carry_their_known_parts_through() {
    let shortage: Shortage = decode(r#"{"kind":{"storage":"kv_units"},"need":4,"hint":1}"#);
    assert_eq!(shortage.kind, Storage::KvUnits);
    assert_eq!(shortage.need, 4);

    let end: End = decode(r#"{"end":"migratable","reason":"drained"}"#);
    assert_eq!(end, End::Migratable);

    let room: Room =
        decode(r#"{"step":2,"kv_units":64,"slots":4,"token_budget":128,"gpu_slots":8}"#);
    assert_eq!(room.step, 2);
    assert_eq!(room.token_budget, 128);

    let work: Work = decode(r#"{"request":1,"phase":1,"tokens":8,"deadline":0}"#);
    assert_eq!(work.tokens, 8);
}
