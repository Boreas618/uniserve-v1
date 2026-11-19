//! A policy package's own program crates (design section 9, item 12): installed from the files a
//! package carries into the workspace of the server's source tree, which holds the program SDK,
//! and built to the wasm32 module of the program kind the crate's code makes it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use build_server::Error;
use build_server::manifest::{ModuleForm, SINGLE_KEY};
use build_server::server::BuildServer;
use build_server::source::MODULE_WASM;
use build_server::target::Target;
use program::Kind;
use proto::module::PROGRAM;

use crate::support;

/// The crate of a routing program that sends every request to the first replica, stating its
/// dependencies on the SDK workspace's own; `routed` is its state's initial count, so two
/// values are two crates.
fn routing_crate(routed: u64) -> BTreeMap<PathBuf, Vec<u8>> {
    let manifest = "[package]\nname = \"program-first\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\
                    publish = false\n\n[lib]\ncrate-type = [\"cdylib\"]\n\n[dependencies]\n\
                    scheduling.workspace = true\nserde.workspace = true\n";
    let source = format!(
        "use scheduling::prelude::*;\n\n\
         #[derive(Deserialize, Serialize)]\n\
         pub struct First {{\n    routed: u64,\n}}\n\n\
         impl Program for First {{\n    type Args = ();\n    type Input = Routing;\n    \
         type Output = Route;\n\n    fn new(_: ()) -> Self {{\n        First {{ routed: {routed} }}\n    }}\n\n    \
         fn step(&mut self, _: &Routing) -> Route {{\n        self.routed += 1;\n        \
         Route {{ prefill: 0, decode: 0 }}\n    }}\n}}\n\nexport!(First);\n"
    );
    BTreeMap::from([
        (PathBuf::from("Cargo.toml"), manifest.as_bytes().to_vec()),
        (PathBuf::from("src/lib.rs"), source.into_bytes()),
    ])
}

/// A program crate is installed as a revision of its own program package and built to a wasm32
/// module that states its kind; the same files again are the same revision, and changed files
/// the next one, which has no commit, since it came from no tree.
#[test]
#[ignore = "builds a program crate in the workspace's SDK for wasm32 with cargo: run with --ignored"]
fn a_program_crate_is_installed_and_built_in_the_sdk() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();

    let installed = server.install_program("first", &routing_crate(0)).unwrap();
    assert_eq!(
        (
            installed.op.as_str(),
            installed.backend.as_str(),
            installed.revision
        ),
        (PROGRAM, "programs.first", 1)
    );
    let again = server.install_program("first", &routing_crate(0)).unwrap();
    assert_eq!(again, installed, "the same files are the same revision");

    server
        .prepare(
            &support::latest(PROGRAM, "programs.first", &[SINGLE_KEY], Vec::new()),
            &mut |_| {},
        )
        .expect("the server builds the program crate");
    let module = server
        .resolve(
            PROGRAM,
            &support::revision("programs.first", 1),
            &Target::wasm(),
            SINGLE_KEY,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(module.form, ModuleForm::Wasm);
    let bytes = support::file_of(&module, MODULE_WASM);
    assert_eq!(
        program::kind(&bytes).unwrap(),
        (Kind::Routing, scheduling::program::INTERFACE_VERSION)
    );
    assert_eq!(
        module.interface,
        service::Versions::one(scheduling::program::INTERFACE_VERSION)
    );
    assert_eq!(module.origin.commit, None);
    assert!(!module.origin.clean);

    let changed = server.install_program("first", &routing_crate(1)).unwrap();
    assert_eq!(changed.revision, 2, "changed files are the next revision");
}

/// A program crate is refused when its directory is no plain name, a file leaves it, it has no
/// manifest, or the server has no source tree to build it in.
#[test]
fn a_program_crate_outside_the_contract_is_refused() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();

    let refused = |result: Result<_, Error>, named: &str| match result {
        Err(Error::Refused { reason, .. }) => assert!(reason.contains(named), "{reason}"),
        other => panic!("{named}: {other:?}"),
    };
    refused(
        server.install_program("../escape", &routing_crate(0)),
        "letters, digits",
    );
    let mut leaving = routing_crate(0);
    leaving.insert(PathBuf::from("../outside.rs"), Vec::new());
    refused(server.install_program("first", &leaving), "outside.rs");
    let mut unmanifested = routing_crate(0);
    unmanifested.remove(&PathBuf::from("Cargo.toml"));
    refused(server.install_program("first", &unmanifested), "Cargo.toml");
    drop(server);

    let other = tempfile::tempdir().unwrap();
    let config = build_server::server::ServerConfig {
        source: None,
        ..support::config(other.path())
    };
    let server = BuildServer::open(&config).unwrap();
    refused(
        server.install_program("first", &routing_crate(0)),
        "no source tree",
    );
}
