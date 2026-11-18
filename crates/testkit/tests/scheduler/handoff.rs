//! Replicas and handoffs (`specs/kv-cache.md` §2.1, §4.2): a replica's role decides which
//! requests it takes from the next submission on, and a request a prefill replica computed the
//! prompt of continues on a decode replica from the KV it lent.
//!
//! The cases run one scheduler, a mixed replica, for both sides of a handoff, its buffer pool
//! lending and importing within one session: a request that hands off ends as migratable with
//! its lend once its first token is sampled, and its continuation, submitted under its id with
//! the tokens it generated and the lend as the daemon submits it, asks the KV program how the
//! lent pages arrive ([`Harness::route`] answers for it) and runs from what arrived.

use proto::request::{Event, GenerationRequest, Lend, MigrationCause, RejectionKind, RequestId};
use scheduling::kv::{Path, Tier, Transport};
use scheduling::routing::Role;
use storage::runtime::{Completion, Request};

use crate::harness::{BLOCK, Deployment, Harness, token};

/// The prompt of the cases' requests: two full pages and two tokens of a third.
const PROMPT: u32 = 2 * BLOCK + 2;

/// Whether request `id`'s life ended rejected for its kind of request under the replica's role.
fn rejected_for_role(harness: &Harness, id: RequestId) -> bool {
    matches!(
        harness.terminal(id),
        Some(Event::Rejected {
            kind: RejectionKind::Role,
            ..
        })
    )
}

/// Request `id` of [`PROMPT`] tokens that may generate `max_tokens` tokens and hands off.
fn handing_off(id: u64, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        hand_off: true,
        ..token(id, PROMPT, max_tokens)
    }
}

/// Runs `request`, which hands off, to its end on the prefill side; returns the lend it ended
/// with and the tokens it generated.
fn prefill(harness: &mut Harness, request: GenerationRequest) -> (Lend, Vec<u32>) {
    let id = harness.submit(request);
    harness.run_until(100, |harness| harness.terminal(id).is_some());
    let Some(Event::Migratable {
        cause: MigrationCause::Handoff,
        lend: Some(lend),
        ..
    }) = harness.terminal(id)
    else {
        panic!("the request ended with {:?}", harness.terminal(id));
    };
    (lend, harness.tokens(id))
}

/// The continuation of `request` after it generated `generated` and lent `lend`, as the daemon
/// submits it to the decode replica.
fn continuation(request: GenerationRequest, generated: Vec<u32>, lend: Lend) -> GenerationRequest {
    GenerationRequest {
        hand_off: false,
        lend: Some(lend),
        generated_token_ids: generated,
        ..request
    }
}

/// Submits the continuation of `request` and answers its handoff with `path`; returns the
/// batches of the request before it.
fn decode(
    harness: &mut Harness,
    request: GenerationRequest,
    (lend, generated): (Lend, Vec<u32>),
    path: Path,
) -> usize {
    let id = request.request_id;
    let before = harness.batches_of(id).len();
    harness.submit(continuation(request, generated, lend));
    harness.run_until(20, |harness| {
        harness.handoffs().iter().any(|(of, _)| *of == id)
    });
    harness.route(id, path);
    harness.settle(200);
    before
}

/// The imports the buffer pool answered, with their answers.
fn imports(harness: &Harness) -> Vec<(Request, Completion)> {
    let answers = harness.answers().into_iter();
    answers
        .filter(|(request, _)| matches!(request, Request::Import { .. }))
        .collect()
}

/// A replica's role decides what it takes from the next submission on: a prefill replica takes
/// only requests that hand off, a decode replica only continuations, of a lend or of another
/// migration, and a mixed one every request; a request taken before the role changed serves to
/// its end.
#[test]
fn a_role_decides_which_requests_a_replica_takes() {
    let mut harness = Deployment::tokens().build();
    let running = harness.submit(token(1, PROMPT, 6));
    harness.run(2);

    harness.tune(|tuning| tuning.roles = vec![Role::Prefill]);
    let plain = harness.submit(token(2, PROMPT, 2));
    let handed = harness.submit(handing_off(3, 4));
    // The lend holds the handed request's blocks until an import ends it: the case waits for
    // the requests' ends rather than for the pool's blocks.
    harness.run_until(200, |harness| {
        [running, plain, handed]
            .iter()
            .all(|id| harness.terminal(*id).is_some())
    });
    assert!(rejected_for_role(&harness, plain));
    assert!(matches!(
        harness.terminal(handed),
        Some(Event::Migratable {
            cause: MigrationCause::Handoff,
            ..
        })
    ));
    assert_eq!(harness.tokens(running).len(), 6, "it serves to its end");

    harness.tune(|tuning| tuning.roles = vec![Role::Decode]);
    let plain = harness.submit(token(4, PROMPT, 2));
    let handing = harness.submit(handing_off(5, 2));
    // A request another migration moved, with the token it generated before, continues here.
    let moved = harness.submit(GenerationRequest {
        generated_token_ids: vec![42],
        ..token(7, PROMPT, 3)
    });
    harness.run_until(200, |harness| {
        [plain, handing, moved]
            .iter()
            .all(|id| harness.terminal(*id).is_some())
    });
    assert!(rejected_for_role(&harness, plain));
    assert!(rejected_for_role(&harness, handing));
    assert!(!rejected_for_role(&harness, moved));
    assert_eq!(
        harness.tokens(moved).len(),
        2,
        "it generates the rest of its tokens"
    );

    harness.tune(|tuning| tuning.roles = vec![Role::Mixed]);
    let plain = harness.submit(token(6, PROMPT, 2));
    harness.run_until(200, |harness| harness.terminal(plain).is_some());
    assert_eq!(harness.tokens(plain).len(), 2);
}

/// A request that hands off ends as migratable once its prompt is computed and one token
/// sampled, with a lend of the prompt's tokens; its blocks stay held for the lend.
#[test]
fn a_request_that_hands_off_ends_with_a_lend_of_its_prompt() {
    let mut harness = Deployment::tokens().build();
    let (lend, generated) = prefill(&mut harness, handing_off(1, 4));
    assert_eq!(lend.tokens, PROMPT);
    assert_eq!(generated.len(), 1);
    assert!(
        harness
            .taken()
            .iter()
            .any(|request| matches!(request, Request::Lend { .. }))
    );
}

/// A continuation whose lent pages are read (`Read`) or written to it (`Push`) imports every
/// page of the lend into blocks of its own and runs from the lend's tokens: its first prefill
/// computes no prompt token, and it generates the rest of its tokens.
#[test]
fn a_read_or_a_push_imports_every_lent_page_and_runs_from_the_lend() {
    for (path, push) in [
        (Path::Read(Transport::Nixl), false),
        (Path::Push(Transport::Nixl), true),
    ] {
        let mut harness = Deployment::tokens().build();
        let request = handing_off(1, 4);
        let lent = prefill(&mut harness, request.clone());
        let skip = decode(&mut harness, request, lent, path);

        let id = RequestId(1);
        let [
            (
                Request::Import {
                    tables,
                    push: pushed,
                    ..
                },
                Completion::Imported { runs, .. },
            ),
        ] = &imports(&harness)[..]
        else {
            panic!("{path:?} answered {:?}", imports(&harness));
        };
        assert_eq!(tables, &[(0, 0, PROMPT.div_ceil(BLOCK))]);
        assert_eq!(*pushed, push);
        assert_eq!(runs.len() as u32, PROMPT.div_ceil(BLOCK));
        assert_eq!(harness.first_written(id, skip), Some(PROMPT as i32));
        assert_eq!(harness.tokens(id).len(), 4, "{path:?}");
        assert!(harness.finished_with(id, proto::request::FinishReason::MaxTokens));
    }
}

/// A continuation that pulls its pages from a tier reads the full pages the tiers hold as any
/// admission reads a prefix, ends the lend with an import of no page, and computes its prompt
/// from past its hit, the prompt's partial last page among it.
#[test]
fn a_pull_reads_the_full_pages_and_computes_the_tail() {
    let mut harness = Deployment::tokens().build();
    let request = handing_off(1, 4);
    let lent = prefill(&mut harness, request.clone());
    let skip = decode(&mut harness, request, lent, Path::Pull(Tier::Host));

    let id = RequestId(1);
    let [(Request::Import { tables, .. }, Completion::Imported { runs, .. })] =
        &imports(&harness)[..]
    else {
        panic!("the pull answered {:?}", imports(&harness));
    };
    assert!(tables.is_empty() && runs.is_empty());
    let hit = (PROMPT / BLOCK * BLOCK) as i32;
    assert_eq!(harness.first_written(id, skip), Some(hit));
    assert_eq!(harness.tokens(id).len(), 4);
}

/// The handoff states what the replica's tiers hold of the lent pages, and the KV program's
/// input counts the full pages and the tail.
#[test]
fn a_handoff_states_the_lent_pages_and_what_the_tiers_hold() {
    let mut harness = Deployment::tokens().build();
    let request = handing_off(1, 4);
    let (lend, generated) = prefill(&mut harness, request.clone());
    harness.submit(continuation(request, generated, lend));
    harness.run_until(20, |harness| !harness.handoffs().is_empty());

    let (_, handoff) = harness.handoffs().remove(0);
    assert_eq!(
        (handoff.pages, handoff.tail),
        (PROMPT / BLOCK, PROMPT % BLOCK)
    );
    assert!(handoff.local);
    let device = handoff
        .found
        .iter()
        .find(|found| found.tier == Tier::Device);
    assert_eq!(device.map(|found| found.pages), Some(PROMPT / BLOCK));
}

/// An import that fails leaves the continuation to an admission of its own, which computes
/// what no tier holds: the request still generates all its tokens.
#[test]
fn a_failed_import_falls_back_to_computing_the_prompt() {
    let mut harness = Deployment::tokens().build();
    let request = handing_off(1, 4);
    let lent = prefill(&mut harness, request.clone());
    harness.pool.fail_tasks(true);
    decode(&mut harness, request, lent, Path::Read(Transport::Nixl));

    let id = RequestId(1);
    assert!(matches!(
        &imports(&harness)[..],
        [(Request::Import { .. }, Completion::Failed)]
    ));
    assert_eq!(harness.tokens(id).len(), 4);
    assert_eq!(harness.failure(), None);
}
