//! The daemon's client of a Storage Server: one deployment's claim, which
//! lasts while the client lives and crosses to another process with its
//! connection.

use ::storage::protocol::ErrorKind;
use ::storage::{ClaimClient, ClaimHandoff, Error, Mode};

use crate::server::{Running, key, scope, worker};

/// An epoch that claims the scopes it holds already keeps the claim held;
/// a further epoch claims under the same token; and a second head is
/// refused while the claim's client lives.
#[test]
fn a_claim_lasts_as_long_as_its_client_and_holds_every_epoch() {
    let server = Running::with_transfer(|_| None);
    let mut claim = ClaimClient::connect(&server.socket, "model").unwrap();

    let token = claim.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    let again = claim.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    assert_eq!(again, token);

    // The scopes of an epoch are compared whatever their order.
    let scopes = vec![scope("text", 1), scope("text", 0)];
    let second = claim.claim(2, scopes.clone(), Vec::new()).unwrap();
    assert_eq!(second, token);
    let reordered = scopes.into_iter().rev().collect();
    assert_eq!(claim.claim(2, reordered, Vec::new()).unwrap(), token);

    // Another daemon cannot take the deployment while this claim lives.
    let mut rival = ClaimClient::connect(&server.socket, "model").unwrap();
    match rival.claim(3, vec![scope("text", 0)], Vec::new()) {
        Err(Error::Server {
            kind: ErrorKind::Claimed,
            ..
        }) => {}
        other => panic!("a second head claimed the deployment: {other:?}"),
    }

    // A worker that states the claim's token is the head's.
    let mut rank = worker(&server.socket, scope("text", 0), 1, token);
    let (found, _) = rank.get(vec![key(1)], Mode::Read, Vec::new()).unwrap();
    assert_eq!(found.missing, vec![key(1)]);
}

/// A claim passed to another process goes on over the same connection: the
/// server sees the same head, so the successor claims further epochs under
/// the same token, and what the claim knew of its epochs crosses with it.
#[test]
fn a_claim_crosses_to_a_successor_with_its_connection() {
    let server = Running::with_transfer(|_| None);
    let mut claim = ClaimClient::connect(&server.socket, "model").unwrap();
    let token = claim.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();

    let (handoff, descriptor) = claim.handoff().unwrap();
    assert_eq!(handoff.claim, Some(token));

    // The state crosses a process boundary as bytes.
    let bytes = serde_json::to_vec(&handoff).unwrap();
    let handoff: ClaimHandoff = serde_json::from_slice(&bytes).unwrap();
    let mut successor = ClaimClient::resume(handoff, descriptor).unwrap();
    assert_eq!(successor.deployment(), "model");

    // The epoch the predecessor claimed is held, and a new one claims under
    // the same token, which a new head would not be given.
    let held = successor
        .claim(1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    assert_eq!(held, token);
    let next = successor.claim(2, vec![scope("text", 0)], vec![1]).unwrap();
    assert_eq!(next, token);

    let usage = successor.usage().unwrap();
    assert!(usage.quotas.is_empty(), "{:?}", usage.quotas);
    assert_eq!(
        successor.endpoint().unwrap().as_deref(),
        Some(server.tcp.as_str())
    );
}
