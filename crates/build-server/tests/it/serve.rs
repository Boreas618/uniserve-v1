//! Serving on a Unix socket (design section 6.3): a server serves its
//! clients' requests on the socket its caller bound, answers a request it
//! refuses on the request's connection, and stops when its stop descriptor
//! becomes readable or is closed.

use build_server::Error;
use build_server::protocol::Client;
use build_server::server::BuildServer;

use crate::support::{self, arch};

/// The manifest of a vendor cubin `rms_norm` package.
fn manifest() -> String {
    support::vendor_manifest("rms_norm", "vendor", 0, r#"{ dtype = ["bf16"] }"#, &arch())
}

#[test]
fn a_server_serves_an_installation_and_a_listing_over_its_socket() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let serving = support::serve(server, &temporary.path().join("build-server.sock"));
    let mut client = serving.client();

    let directory = support::vendor_package(&temporary.path().join("package"), &manifest(), &[]);
    let installed = client.install(&directory).unwrap();
    assert_eq!(installed.op, "rms_norm");
    assert_eq!(installed.backend, "vendor");
    assert_eq!(installed.revision, 1);

    let listing = client.list().unwrap();
    assert_eq!(listing.len(), 1);
    assert_eq!(listing[0].0, installed);
    assert!(listing[0].2.is_empty(), "installing builds nothing");

    drop(client);
    serving.stop().unwrap();
}

/// A request the server refuses is answered on its connection with the
/// reason, and the connection goes on serving the client's next request.
#[test]
fn a_refused_request_is_answered_and_the_connection_serves_the_next_one() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let serving = support::serve(server, &temporary.path().join("build-server.sock"));
    let mut client = serving.client();
    let directory = support::vendor_package(&temporary.path().join("package"), &manifest(), &[]);
    client.install(&directory).unwrap();

    let refused = client
        .prepare(
            &support::latest("rms_norm", "cuda", &["dtype_bf16"], vec![arch()]),
            &mut |_| {},
        )
        .expect_err("no cuda backend was installed");
    match refused {
        Error::Refused { reason, upstream } => {
            assert!(reason.contains("backend cuda"), "{reason}");
            assert_eq!(upstream, None);
        }
        other => panic!("the refusal was {other:?}"),
    }

    assert_eq!(client.list().unwrap().len(), 1);
    drop(client);
    serving.stop().unwrap();
}

/// A server serves until its stop descriptor becomes readable, or its other
/// end is closed; it then returns and removes its socket, so nothing
/// connects to it any more.
#[test]
fn serving_stops_when_the_stop_descriptor_becomes_readable_or_is_closed() {
    let _exclusive = support::exclusive();
    for closed in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let socket = temporary.path().join("build-server.sock");
        let server = BuildServer::open(&support::config(temporary.path())).unwrap();
        let serving = support::serve(server, &socket);

        // The server serves before it is stopped.
        let mut client = serving.client();
        assert!(client.list().unwrap().is_empty());
        drop(client);

        if closed {
            serving.close().unwrap();
        } else {
            serving.stop().unwrap();
        }
        assert!(!socket.exists(), "the stopped server left its socket");
        assert!(Client::connect(&socket).is_err());
    }
}
