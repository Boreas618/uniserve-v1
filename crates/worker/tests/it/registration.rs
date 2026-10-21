//! A rank's registration with the engine, against an engine stand-in.

use std::time::{Duration, Instant};

use ipc::{Listener, Registration, VERSION};
use worker::launch::{Arguments, register};

/// The command line of rank 3 of worker `text`'s second launch, for an engine
/// at `engine`.
fn arguments(engine: &str) -> Arguments {
    Arguments::parse(&[
        "--engine",
        engine,
        "--worker-id",
        "text",
        "--rank",
        "3",
        "--generation",
        "2",
    ])
    .unwrap()
}

/// A rank waits for its launch descriptor for as long as the engine keeps the
/// connection, since the engine answers only once every rank of the launch has
/// registered. The engine closing the connection, as it does when it refuses a
/// registration or gives up a launch, is what ends the wait.
#[test]
fn the_wait_for_the_descriptor_ends_when_the_engine_closes() {
    let listener = Listener::bind(&["127.0.0.1:0"]).unwrap();
    let address = listener.addresses()[0].clone();
    let held = Duration::from_millis(500);

    let engine = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut channel = loop {
            if let Some(channel) = listener.try_accept().unwrap() {
                break channel;
            }
            assert!(Instant::now() < deadline, "the rank did not connect");
            std::thread::sleep(Duration::from_millis(5));
        };
        let registration = loop {
            if let Some(registration) = channel.try_recv::<Registration>().unwrap() {
                break registration;
            }
            assert!(Instant::now() < deadline, "the rank did not register");
            channel.wait(Duration::from_millis(50)).unwrap();
        };

        // The engine keeps the rank waiting, then closes without answering.
        std::thread::sleep(held);
        drop(channel);
        registration
    });

    let started = Instant::now();
    let error = register(&arguments(&address))
        .err()
        .expect("a closed connection ends the registration")
        .to_string();
    let waited = started.elapsed();

    assert_eq!(
        engine.join().unwrap(),
        Registration::Worker {
            worker_id: "text".into(),
            rank: 3,
            generation: 2,
            version: VERSION.into(),
        }
    );
    assert!(waited >= held, "the rank gave up before the engine closed");
    assert!(
        waited < Duration::from_secs(30),
        "the rank outwaited the engine's close"
    );
    assert!(error.contains(&address), "{error}");
}

#[test]
fn an_engine_that_cannot_be_reached_fails_the_registration() {
    // The listener is bound and dropped, so nothing listens at its address.
    let address = Listener::bind(&["127.0.0.1:0"]).unwrap().addresses()[0].clone();
    assert!(register(&arguments(&address)).is_err());
}
