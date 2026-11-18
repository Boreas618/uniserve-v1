//! Groups (SD §8.1): a request's phases on the worker groups of a deployment (SD §3.5 `place`,
//! "Prefill to decode transfer"; §4.2 "Owners of the classes"; §7.3 "Routes").
//!
//! Each phase a request runs is placed on a group that serves it and whose role allows it: a
//! decode or a verification never on a `Prefill` group, every other phase never on a `Decode`
//! group. A request placed nowhere yet goes where the policy's `place` names, and its later
//! phases follow it to a group that holds it. A request prefilled on a `Prefill` group moves to
//! a decode group by `Transfer` once its prompt is computed, and decodes there under the slot its
//! `Transferred` gave; a failed transfer ends it as migratable, its owner detached with no stamp,
//! since no device block table holds its entries. A request is attached to every group it is
//! placed on by an owner of its own, and a product one group writes and another reads is mapped
//! and stamped by both.

use std::collections::{HashMap, HashSet};

use engine::batch::{Batch, Phase};
use proto::report::SchedulerStats;
use proto::request::{Event, MigrationCause, RejectionKind, RequestId};
use storage::runtime::{Completion, GroupId, OwnerId, Request};

use crate::harness::{A, B, BLOCK, Deployment, Harness, chat_image, token, video};

/// The slot of request `id`'s row in `batch`: a token row's, or a media row's.
fn slot_in(batch: &Batch, id: RequestId) -> Option<u32> {
    let row = batch.requests.iter().position(|of| *of == id)?;
    let token = batch.columns.rows.calls.get(row).map(|call| call.slot);
    token.or_else(|| batch.media.get(row).map(|media| media.slot))
}

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

/// The transfers the scheduler's statistics counted over every report it sent, once it sent the
/// next one: those answered `Transferred`, those answered `Failed`, and the blocks the target
/// groups received.
fn transfers_counted(harness: &mut Harness) -> (u64, u64, u64) {
    harness.next_report();
    let sum = |field: fn(&SchedulerStats) -> u64| {
        let reports = harness.reports().into_iter();
        reports.map(field).sum::<u64>()
    };
    (
        sum(|report| report.transfers),
        sum(|report| report.failed_transfers),
        sum(|report| report.transferred_blocks),
    )
}

/// On a deployment that prefills on one group and decodes on another, every prefill of every
/// request runs on the `Prefill` group and every decode on the `Decode` group (SD §3.5 `place`).
#[test]
fn each_phase_is_placed_on_a_group_whose_role_allows_it() {
    let mut harness = Deployment::tokens().disaggregated().build();
    let ids: Vec<RequestId> = (1..=3)
        .map(|id| harness.submit(token(id, 6 + 3 * id as u32, 4)))
        .collect();
    harness.settle(200);

    for id in &ids {
        assert_eq!(harness.tokens(*id).len(), 4);
        let batches = harness.batches_of(*id);
        assert!(batches.iter().any(|batch| batch.phase == Phase::Decode));
        for batch in batches {
            let expected = match batch.phase {
                Phase::Prefill => A,
                Phase::Decode => B,
                other => panic!("request {} ran {other:?}", id.0),
            };
            assert_eq!(
                batch.group, expected,
                "{:?} of request {}",
                batch.phase, id.0
            );
        }
    }
}

/// On two groups that serve every phase alike, a request placed nowhere goes where the policy's
/// `place` names, the group with the fewest requests placed (the first on a tie), and its later
/// phases stay on the group that holds it: the first request, a chat request, prefills, encodes
/// its image and decodes on the first group; the second request runs on the other (SD §3.5
/// `place`; the `arrival` policy's `place`).
#[test]
fn a_request_goes_where_the_policy_places_it_and_its_later_phases_follow_it() {
    let mut harness = Deployment::tokens().images().replicas(2).build();
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

/// A request prefilled on a `Prefill` group moves to a decode group once its prompt is computed:
/// its one owner, attached to the prefill group, is moved by `Transfer`, which reaches the buffer
/// pool with the prefill group's stamp (its workers removed the owner's entries); the decode
/// group is handed the `Transferred` runs under the slot it gave, its first decode carries the
/// request's `NewSlot`, and the request decodes there to its end (SD §3.5 "Prefill to decode
/// transfer").
#[test]
fn a_prefill_then_its_transfer_then_its_decodes_on_another_group() {
    let mut harness = Deployment::tokens().disaggregated().build();
    let id = harness.submit(token(1, 9, 5));
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 5);

    let owners = attached(&harness);
    let [(group, owner)] = owners[..] else {
        panic!("one owner, on the prefill group: {owners:?}");
    };
    assert_eq!(group, A);
    let transfer = harness
        .taken()
        .into_iter()
        .find_map(|request| match request {
            Request::Transfer {
                owner: of,
                group,
                after,
                ..
            } if of == owner => Some((group, after)),
            _ => None,
        });
    let (target, after) = transfer.unwrap();
    assert_eq!(target, B);
    assert!(after.iter().any(|(group, _)| *group == A), "{after:?}");

    let transferred =
        harness
            .fake
            .mapped()
            .into_iter()
            .find_map(
                |(group, slot, _, request, completion)| match (request, completion) {
                    (
                        Request::Transfer { owner: of, .. },
                        Completion::Transferred { slot: given, runs },
                    ) if of == owner && group == B => {
                        assert_eq!(slot, given, "mapped under the slot it gave");
                        Some((given, runs))
                    }
                    _ => None,
                },
            );
    let (slot, runs) = transferred.unwrap();

    // The statistics count one transfer and the blocks the decode group received, one per entry
    // `Transferred` stated (SD §7.5): the prompt's blocks and the low watermark's, which the
    // prefill group held when its prompt was computed (SD §4.4).
    let entries: HashSet<(u8, u32)> = runs
        .iter()
        .map(|&(class, position, _, _)| (class, position))
        .collect();
    assert_eq!(entries.len() as u32, 9_u32.div_ceil(BLOCK) + 1, "{runs:?}");
    assert_eq!(
        transfers_counted(&mut harness),
        (1, 0, entries.len() as u64)
    );

    let batches = harness.batches_of(id);
    let decodes: Vec<&Batch> = batches
        .iter()
        .filter(|batch| batch.phase == Phase::Decode)
        .collect();
    assert!(!decodes.is_empty());
    for decode in &decodes {
        assert_eq!((decode.group, slot_in(decode, id)), (B, Some(slot)));
    }
    let introduced = decodes[0].new_slots.iter().any(|new| new.slot == slot);
    assert!(
        introduced,
        "its first decode on the decode group carries its slot"
    );
    let last_prefill = batches
        .iter()
        .rposition(|batch| batch.phase == Phase::Prefill);
    let first_decode = batches
        .iter()
        .position(|batch| batch.phase == Phase::Decode);
    assert!(
        last_prefill < first_decode,
        "it decodes once the transfer ended"
    );
}

/// A transfer that fails leaves the owner attached to the prefill group, holding blocks no
/// device block table holds: the request ends as `Migratable { Transfer }` for the daemon to
/// resubmit, its owner is detached with no stamp, and the buffer pool holds nothing of it (SD
/// §3.5 "Prefill to decode transfer", §7.6; `specs/buffer-pool.md` §12.4).
#[test]
fn a_failed_transfer_ends_as_migratable_and_detaches_its_owner_with_no_stamp() {
    let mut harness = Deployment::tokens().disaggregated().build();
    harness.pool.fail_tasks(true);
    let id = harness.submit(token(1, 9, 5));
    harness.settle(100);

    let failed = harness
        .answers()
        .into_iter()
        .any(|answer| matches!(answer, (Request::Transfer { .. }, Completion::Failed)));
    assert!(failed);
    assert!(matches!(
        harness.terminal(id),
        Some(Event::Migratable {
            cause: MigrationCause::Transfer,
            ..
        })
    ));
    let owners = attached(&harness);
    let [(_, owner)] = owners[..] else {
        panic!("{owners:?}");
    };
    assert_eq!(detached_by(&harness, owner), Some(Vec::new()));
    assert!(
        harness
            .batches_of(id)
            .iter()
            .all(|batch| batch.phase == Phase::Prefill),
        "it never decoded"
    );
    assert_eq!(harness.held(), 0);

    // The statistics count the failed transfer, and no block received (SD §7.5).
    assert_eq!(transfers_counted(&mut harness), (0, 1, 0));
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
