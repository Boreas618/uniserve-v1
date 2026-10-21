//! Microbatch threads and their rotation against a real driver. The tests
//! need a CUDA device for the microbatch streams; run them with
//! `cargo test -p worker --test it microbatch_lanes -- --ignored`.

use std::sync::{Arc, Mutex, PoisonError};

use cuda::{DeviceId, Stream};
use worker::Error;
use worker::parallel::{MicrobatchCall, MicrobatchLanes, yield_microbatch};

fn streams(count: usize) -> (Stream, Vec<Stream>) {
    let parent = Stream::new(DeviceId(0)).unwrap();
    (
        parent,
        (0..count)
            .map(|_| Stream::new(DeviceId(0)).unwrap())
            .collect(),
    )
}

#[test]
#[ignore = "needs a CUDA device"]
fn microbatches_take_turns_at_each_yield_on_their_own_threads() {
    let lanes = MicrobatchLanes::new(3).unwrap();
    let (parent, streams) = streams(3);
    let order = Arc::new(Mutex::new(Vec::new()));
    let names = Arc::new(Mutex::new(Vec::new()));
    for round in 0..2 {
        let calls: Vec<MicrobatchCall<usize>> = [3, 1, 2]
            .into_iter()
            .enumerate()
            .map(|(index, layers)| {
                let order = Arc::clone(&order);
                let names = Arc::clone(&names);
                Box::new(move || {
                    let name = std::thread::current().name().map(str::to_owned);
                    names
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push(name);
                    for layer in 0..layers {
                        order
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push((round, index, layer));
                        yield_microbatch()?;
                    }
                    Ok(index * 10)
                }) as MicrobatchCall<usize>
            })
            .collect();
        assert_eq!(lanes.run(&parent, &streams, calls).unwrap(), [0, 10, 20]);
    }
    // Every invocation submits its dispatches in the same cyclic order; a
    // finished microbatch takes no further turn.
    let expected: Vec<_> = (0..2)
        .flat_map(|round| {
            [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2)]
                .map(|(index, layer)| (round, index, layer))
        })
        .collect();
    assert_eq!(*order.lock().unwrap(), expected);
    // Each microbatch runs on its own persistent thread.
    let mut names: Vec<_> = names.lock().unwrap().iter().flatten().cloned().collect();
    names.sort();
    names.dedup();
    assert_eq!(names, ["microbatch-0", "microbatch-1", "microbatch-2"]);
}

#[test]
#[ignore = "needs a CUDA device"]
fn the_first_failure_wakes_suspended_peers_and_is_reported() {
    let mut lanes = MicrobatchLanes::new(2).unwrap();
    let (parent, streams) = streams(2);
    let calls: Vec<MicrobatchCall<()>> = vec![
        // Suspended at its yield when its peer fails.
        Box::new(|| {
            yield_microbatch()?;
            yield_microbatch()
        }),
        Box::new(|| Err(Error::Invalid("expert layer failed".into()))),
    ];
    match lanes.run(&parent, &streams, calls) {
        Err(Error::Invalid(message)) => assert_eq!(message, "expert layer failed"),
        other => panic!("expected the failing microbatch's error, got {other:?}"),
    }
    // The rotation is reusable after a failure, then closes its threads.
    let calls: Vec<MicrobatchCall<u8>> = vec![Box::new(|| Ok(1)), Box::new(|| Ok(2))];
    assert_eq!(lanes.run(&parent, &streams, calls).unwrap(), [1, 2]);
    lanes.close().unwrap();
    let calls: Vec<MicrobatchCall<u8>> = vec![Box::new(|| Ok(1)), Box::new(|| Ok(2))];
    assert!(lanes.run(&parent, &streams, calls).is_err());
}
