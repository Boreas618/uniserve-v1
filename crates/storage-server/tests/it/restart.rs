//! Restarts in place: a server whose successor does not take it over goes on.

use ::storage::runtime::{Completion, OwnerId, Request};
use ::storage::{Client, Role};

use crate::server::{FORKING, Running};
use crate::sessions::{DECODE, PREFILL, Scheduler, layout};

/// A restart in place whose successor ends without taking the server over
/// is given up: the buffer pools take requests again and the server goes on
/// serving.
#[test]
fn a_server_whose_successor_ends_without_taking_it_over_goes_on() {
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let server = Running::with_transfer(|_| None);
    let mut engine = crate::server::engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let mut schedulers = Scheduler::all(&opened, descriptors);

    let mut operator = Client::connect(&server.socket, Role::Operator).unwrap();
    let successor = operator.restart(Some("/bin/false".into())).unwrap();
    assert!(successor > 0);

    // The pools serve again once the restart is given up.
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    };
    assert_eq!(schedulers[0].ask(attach), Completion::Attached { slot: 1 });
    assert_eq!(operator.list().unwrap().buffer_pools.len(), 1);

    // The restart is no longer under way: another may begin, and is given
    // up alike.
    operator.restart(Some("/bin/false".into())).unwrap();
    let attach = Request::Attach {
        owner: OwnerId(2),
        group: DECODE,
    };
    assert_eq!(schedulers[1].ask(attach), Completion::Attached { slot: 1 });
}
