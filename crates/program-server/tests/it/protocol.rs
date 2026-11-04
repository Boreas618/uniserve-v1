//! The protocol and its client (design section 6.3): framed Protobuf
//! messages, the descriptors that ride on a frame, and the client's four
//! calls.

use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;

use program_server::manifest::ModuleForm;
use program_server::protocol::{
    Client, Connection, Descriptors, PrepareRequest, ProgramRequest, Selector, wire,
};
use program_server::store::ModulePath;
use program_server::target::Target;
use program_server::{Error, PhaseKind, Program, ProgramSlot};

/// A program of the Triton `rms_norm` package.
fn program() -> Program {
    Program {
        backend: "triton".into(),
        revision: 2,
        arguments: toml::Table::new(),
    }
}

/// Runs `serve` on one end of a socket pair and answers the client of the
/// other.
fn server_on(serve: impl FnOnce(&mut Connection) + Send + 'static) -> Client {
    let (client, server) = UnixStream::pair().unwrap();
    std::thread::spawn(move || {
        let mut connection = Connection::new(server);
        serve(&mut connection);
    });
    Client::new(Connection::new(client))
}

#[test]
fn a_request_crosses_the_connection_with_the_descriptor_of_its_frame() {
    let (left, right) = UnixStream::pair().unwrap();
    let mut sender = Connection::new(left);
    let mut receiver = Connection::new(right);

    let mut file = tempfile::tempfile().unwrap();
    file.write_all(b"a cubin").unwrap();
    let descriptor = OwnedFd::from(file);
    sender
        .send(
            &wire::Request {
                request: Some(wire::request::Request::Install(wire::Install {
                    package: "/packages/rms_norm/triton".into(),
                })),
            },
            std::slice::from_ref(&descriptor),
        )
        .unwrap();

    let (request, descriptors) = receiver.receive::<wire::Request>().unwrap();
    match request.request {
        Some(wire::request::Request::Install(install)) => {
            assert_eq!(install.package, "/packages/rms_norm/triton");
        }
        other => panic!("unexpected request {other:?}"),
    }
    assert_eq!(descriptors.len(), 1);
    let mut file = std::fs::File::from(descriptors.into_iter().next().unwrap());
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(contents, b"a cubin");
}

#[test]
fn install_answers_the_revision_the_upstream_gave_the_package() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Installed(wire::Installed {
                        package: Some(wire::Package {
                            op: "rms_norm".into(),
                            backend: "triton".into(),
                            revision: 4,
                        }),
                    })),
                },
                &[],
            )
            .unwrap();
    });

    let installed = client
        .install(std::path::Path::new("/packages/rms_norm/triton"))
        .unwrap();
    assert_eq!(installed.package.revision, 4);
}

#[test]
fn an_install_another_server_refuses_names_the_upstream() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Refused(wire::Refused {
                        reason: "only the upstream Program Server installs packages".into(),
                        upstream: "unix:/run/uniserve/leader-program-server.sock".into(),
                    })),
                },
                &[],
            )
            .unwrap();
    });

    let refused = client.install(std::path::Path::new("/packages/rms_norm/triton"));
    match refused {
        Err(Error::Refused { upstream, .. }) => assert_eq!(
            upstream.as_deref(),
            Some("unix:/run/uniserve/leader-program-server.sock")
        ),
        other => panic!("unexpected answer {other:?}"),
    }
}

#[test]
fn prepare_reports_each_build_and_answers_the_selected_programs() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        let progress = |state: wire::BuildState, detail: &str| wire::ServerMessage {
            message: Some(wire::server_message::Message::Progress(wire::Progress {
                op: "rms_norm".into(),
                backend: "triton".into(),
                revision: 2,
                target: "sm_100a".into(),
                descriptor: "dtype_bf16".into(),
                state: state as i32,
                detail: detail.into(),
            })),
        };
        connection
            .send(&progress(wire::BuildState::Queued, ""), &[])
            .unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"cubin").unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Prepared(wire::Prepared {
                        programs: vec![wire::Chosen {
                            op: "rms_norm".into(),
                            phase: wire::ProgramPhase::All as i32,
                            program: Some(wire::ProgramRef {
                                backend: "triton".into(),
                                revision: 2,
                                arguments: "".into(),
                            }),
                        }],
                    })),
                },
                &[],
            )
            .unwrap();
    });

    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("rms_norm"),
            selector: Selector::Latest {
                backend: "triton".into(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            op: "rms_norm".into(),
            keys: vec!["dtype_bf16".into()],
        }],
    };
    let mut seen = Vec::new();
    let prepared = client
        .prepare(&request, &mut |progress| seen.push(progress))
        .unwrap();

    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].path,
        ModulePath::new("rms_norm", "triton", 2, "sm_100a", "dtype_bf16").unwrap()
    );
    assert_eq!(prepared.programs.len(), 1);
    assert_eq!(prepared.programs[0].slot, ProgramSlot::all("rms_norm"));
    assert_eq!(prepared.programs[0].program.revision, 2);
}

#[test]
fn a_module_comes_back_with_its_files_over_their_descriptors() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"the cubin bytes").unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Resolved(wire::Resolved {
                        form: wire::ModuleForm::Cubin as i32,
                        files: vec![wire::ModuleFile {
                            name: "module.cubin".into(),
                            size: "the cubin bytes".len() as u64,
                        }],
                        entry: String::new(),
                        entries: std::collections::HashMap::new(),
                        metadata: "metadata.json".into(),
                    })),
                },
                &[OwnedFd::from(file)],
            )
            .unwrap();
    });

    let module = client
        .resolve(
            "rms_norm",
            &program(),
            &Target::new("sm_100a"),
            "dtype_bf16",
            &mut |_| {},
        )
        .unwrap();

    assert_eq!(module.form, ModuleForm::Cubin);
    assert_eq!(module.files.len(), 1);
    assert_eq!(module.files[0].name, "module.cubin");
    assert_eq!(module.files[0].size, "the cubin bytes".len() as u64);
    assert_eq!(module.metadata.as_deref(), Some("metadata.json"));
    assert!(module.entries.is_empty());
    let mut bytes = Vec::new();
    let mut file = &module.files[0].file;
    file.seek(SeekFrom::Start(0)).unwrap();
    file.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"the cubin bytes");
}

#[test]
fn a_listing_names_the_packages_revisions_and_modules() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Listing(wire::Listing {
                        packages: vec![wire::ListedPackage {
                            op: "rms_norm".into(),
                            backend: "triton".into(),
                            revision: 2,
                            modules: vec![wire::ListedModule {
                                target: "sm_100a".into(),
                                descriptor: "dtype_bf16".into(),
                                form: wire::ModuleForm::Cubin as i32,
                            }],
                        }],
                    })),
                },
                &[],
            )
            .unwrap();
    });

    let listing = client.list().unwrap();
    assert_eq!(listing.packages.len(), 1);
    let package = &listing.packages[0];
    assert_eq!(package.revision, 2);
    assert_eq!(package.modules.len(), 1);
    assert_eq!(package.modules[0].descriptor, "dtype_bf16");
    assert_eq!(package.modules[0].form, ModuleForm::Cubin);
}

#[test]
fn a_native_module_comes_back_with_its_entries_by_role() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"the shared object").unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Resolved(wire::Resolved {
                        form: wire::ModuleForm::Native as i32,
                        files: vec![wire::ModuleFile {
                            name: "module.so".into(),
                            size: "the shared object".len() as u64,
                        }],
                        entry: String::new(),
                        entries: [
                            ("launch".to_owned(), "uniserve_launch".to_owned()),
                            ("restore".to_owned(), "uniserve_restore".to_owned()),
                        ]
                        .into_iter()
                        .collect(),
                        metadata: String::new(),
                    })),
                },
                &[OwnedFd::from(file)],
            )
            .unwrap();
    });

    let module = client
        .resolve(
            "expert_exchange",
            &program(),
            &Target::new("sm_100a"),
            "dtype_bf16",
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(module.form, ModuleForm::Native);
    assert_eq!(
        module.entries.get("launch").map(String::as_str),
        Some("uniserve_launch")
    );
    assert_eq!(module.entries.len(), 2);
    assert!(module.metadata.is_none());
}

#[test]
fn a_listing_names_a_policy_module_as_a_wasm_module() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Listing(wire::Listing {
                        packages: vec![wire::ListedPackage {
                            op: "scheduling".into(),
                            backend: "fcfs".into(),
                            revision: 1,
                            modules: vec![wire::ListedModule {
                                target: "wasm32".into(),
                                descriptor: "default".into(),
                                form: wire::ModuleForm::Wasm as i32,
                            }],
                        }],
                    })),
                },
                &[],
            )
            .unwrap();
    });

    let listing = client.list().unwrap();
    assert_eq!(listing.packages[0].modules[0].form, ModuleForm::Wasm);
}

#[test]
fn a_listed_module_of_an_unknown_form_is_refused_rather_than_guessed() {
    let mut client = server_on(|connection| {
        let _ = connection.receive::<wire::Request>().unwrap();
        connection
            .send(
                &wire::ServerMessage {
                    message: Some(wire::server_message::Message::Listing(wire::Listing {
                        packages: vec![wire::ListedPackage {
                            op: "rms_norm".into(),
                            backend: "triton".into(),
                            revision: 1,
                            modules: vec![wire::ListedModule {
                                target: "sm_100a".into(),
                                descriptor: "dtype_bf16".into(),
                                form: 99,
                            }],
                        }],
                    })),
                },
                &[],
            )
            .unwrap();
    });

    assert!(matches!(client.list(), Err(Error::Protocol(_))));
}

#[test]
fn the_protocol_speaks_phases_and_arguments_of_its_own_vocabulary() {
    // The wire's phases map onto the record's, and arguments travel as a
    // TOML table, so a program's types survive the connection.
    let mut arguments = toml::Table::new();
    arguments.insert("chunked_prefill_size".into(), toml::Value::Integer(4096));
    let message = wire::ProgramRef::from(&Program {
        backend: "fcfs".into(),
        revision: 7,
        arguments,
    });
    let read = Program::try_from(message).unwrap();
    assert_eq!(read.revision, 7);
    assert_eq!(
        read.arguments.get("chunked_prefill_size"),
        Some(&toml::Value::Integer(4096))
    );
    assert_eq!(
        Option::<PhaseKind>::try_from(wire::ProgramPhase::Decode).unwrap(),
        Some(PhaseKind::Decode)
    );
    assert_eq!(
        Option::<PhaseKind>::try_from(wire::ProgramPhase::All).unwrap(),
        None
    );
}
