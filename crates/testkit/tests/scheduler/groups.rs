//! Groups (SD §8.1): a request's phases on the worker groups of a deployment (SD §3.5 `place`;
//! §4.2 "Owners of the classes"; §7.3 "Routes").
//!
//! Each phase a request runs is placed on a group that serves it. A request placed nowhere yet
//! goes where the policy's `place` names, and its later phases follow it to a group that holds
//! it. A request is attached to every group it is
//! placed on by an owner of its own, and a product one group writes and another reads is mapped
//! and stamped by both.

use std::collections::HashMap;

use engine::batch::Phase;
use proto::request::{Event, RejectionKind, RequestId};
use storage::runtime::{GroupId, OwnerId, Request};

use crate::harness::{A, B, Deployment, Harness, chat_image, token, video};

/// The owners the buffer pool attached, by group, in the order of their `Attach`.
fn attached(harness: &Harness) -> Vec<(GroupId, OwnerId)> {
    let attaches = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Attach { owner, group } => Some((group, owner)),
            _ => None,
        });
    attaches.collect()
}

/// The groups that stamped the `Detach` of `owner` the buffer pool took, in group order.
fn detached_by(harness: &Harness, owner: OwnerId) -> Option<Vec<GroupId>> {
    harness
        .taken()
        .into_iter()
        .find_map(|request| match request {
            Request::Detach { owner: of, after } if of == owner => {
                let mut groups: Vec<GroupId> = after.iter().map(|(group, _)| *group).collect();
                groups.sort_unstable_by_key(|group| group.0);
                Some(groups)
            }
            _ => None,
        })
}

/// On two groups that serve every phase alike, a request placed nowhere goes where the policy's
/// `place` names, the group with the fewest requests placed (the first on a tie), and its later
/// phases stay on the group that holds it: the first request, a chat request, prefills, encodes
/// its image and decodes on the first group; the second request runs on the other (SD §3.5
/// `place`; the `arrival` policy's `place`).
#[test]
fn a_request_goes_where_the_policy_places_it_and_its_later_phases_follow_it() {
    let mut harness = Deployment::tokens().images().copies(2).build();
    let chat = harness.submit(chat_image(1, 12, 6, 4));
    let other = harness.submit(token(2, 10, 4));
    harness.settle(200);

    let groups = |id: RequestId| {
        let mut placed: Vec<(Phase, GroupId)> = harness
            .batches_of(id)
            .iter()
            .map(|batch| (batch.phase, batch.group))
            .collect();
        placed.dedup();
        placed
    };
    let chat_groups = groups(chat);
    assert!(
        chat_groups
            .iter()
            .any(|(phase, _)| *phase == Phase::VisionEncode),
        "{chat_groups:?}"
    );
    assert!(
        chat_groups.iter().all(|(_, group)| *group == A),
        "{chat_groups:?}"
    );
    let other_groups = groups(other);
    assert!(
        other_groups.iter().all(|(_, group)| *group == B),
        "{other_groups:?}"
    );
    assert_eq!(harness.tokens(chat).len(), 4);
    assert_eq!(harness.tokens(other).len(), 4);
}

/// A video request placed on two groups (the first prepares its latents and denoises them, the
/// second decodes, encodes and muxes) holds one owner on each; each group runs only the phases it
/// serves; a product the first group's owner holds and the second group reads (the denoised latents
/// its decode reads) is handed to the executor for both groups, and that owner's `Detach` is
/// stamped by both (SD §4.2 "Owners of the classes", §3.1).
#[test]
fn a_video_request_holds_one_owner_per_group_and_a_product_is_read_by_both() {
    let denoiser: &[Phase] = &[
        Phase::MediaRead,
        Phase::TextEncode,
        Phase::LatentPrepare,
        Phase::Denoise,
    ];
    let decoder: &[Phase] = &[
        Phase::VideoDecode,
        Phase::AudioDecode,
        Phase::VideoEncode,
        Phase::AudioEncode,
        Phase::Mux,
    ];
    let mut harness = Deployment::video(&[denoiser, decoder]).build();
    let id = harness.submit_video(video(1, 1));
    harness.settle(200);
    assert!(matches!(harness.terminal(id), Some(Event::Finished { .. })));

    let owners: HashMap<GroupId, OwnerId> = attached(&harness).into_iter().collect();
    assert_eq!(attached(&harness).len(), 2, "one owner per group");
    assert_ne!(owners[&A], owners[&B]);
    for batch in harness.batches_of(id) {
        let serves = if batch.group == A { denoiser } else { decoder };
        assert!(
            serves.contains(&batch.phase),
            "{:?} on group {}",
            batch.phase,
            batch.group.0
        );
    }

    // A product block of the denoiser group's owner, mapped on the decoder group too.
    let mapped = harness.fake.mapped();
    let read_by_both = mapped.iter().any(|(group, _, number, request, _)| {
        let Request::Allocate { owner, .. } = request else {
            return false;
        };
        *group == B
            && *owner == owners[&A]
            && mapped
                .iter()
                .any(|(other, _, of, ..)| *other == A && of == number)
    });
    assert!(read_by_both, "{mapped:#?}");
    assert_eq!(detached_by(&harness, owners[&A]), Some(vec![A, B]));
    assert_eq!(harness.held(), 0);
}

/// No phase is placed on a group that does not serve it: on a deployment whose images are
/// encoded by a group of their own, a chat request's encode runs there and its token calls on
/// the token group; on a deployment where no group encodes images, a chat request is rejected
/// at its arrival (SD §3.5 `place`, §7.2 "Validation of a token request").
#[test]
fn no_phase_is_placed_on_a_group_that_does_not_serve_it() {
    let mut harness = Deployment::tokens().encoder().build();
    let id = harness.submit(chat_image(1, 12, 6, 3));
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 3);
    for batch in harness.batches_of(id) {
        let expected = if batch.phase == Phase::VisionEncode {
            B
        } else {
            A
        };
        assert_eq!(batch.group, expected, "{:?}", batch.phase);
    }

    let mut harness = Deployment::tokens().build();
    let id = harness.submit(chat_image(1, 12, 6, 3));
    let events = harness.events(id);
    let [Event::Rejected { kind, .. }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
}
