//! The server's operations (design section 6.3): what an installation numbers
//! and holds, the package root an upstream server owns, what a server that is
//! not the upstream returns, the errors a request gets before any build
//! starts, a scheduling policy's one module, and a store a release ships.

use std::fs;
use std::path::{Path, PathBuf};

use build_server::Error;
use build_server::builder::BuildState;
use build_server::inputs::{Inputs, RECORD};
use build_server::manifest::{ModuleForm, SINGLE_KEY, ToolchainKind};
use build_server::protocol::{CoverRequest, PackageRequest, PrepareRequest, Selector};
use build_server::server::BuildServer;
use build_server::store::{ModulePath, Root, Store};
use build_server::target::Target;
use build_server::toolchain::Toolchains;
use proto::module::{ModuleSlot, PackageRevision};

use crate::support::{self, arch, other_arch, write};

/// The vendor cubin `rms_norm` package these tests install, covering one
/// descriptor and stating one parameter.
fn manifest() -> String {
    let manifest =
        support::vendor_manifest("rms_norm", "vendor", 0, r#"{ dtype = ["bf16"] }"#, &arch());
    format!("{manifest}\n[parameters]\ntile = 1024\nepsilon = 1e-6\n")
}

/// Writes the package of [`manifest`] into `directory` and returns it.
fn package(directory: &Path) -> PathBuf {
    support::vendor_package(directory, &manifest(), &["dtype_bf16"])
}

/// The directory of the installed revision `revision` of the `rms_norm`
/// package under `package_root`.
fn revision_directory(package_root: &Path, revision: u32) -> PathBuf {
    package_root.join(format!("rms_norm/vendor/{revision}"))
}

/// The installed revisions a server lists, as `op backend revision`.
fn listed(server: &BuildServer) -> Vec<String> {
    server
        .list()
        .unwrap()
        .into_iter()
        .map(|(package, _)| format!("{} {} {}", package.op, package.backend, package.revision))
        .collect()
}

/// The error a server that cannot open returns.
fn refused_open(config: &build_server::ServerConfig) -> Error {
    match BuildServer::open(config) {
        Ok(_) => panic!("the server opened"),
        Err(error) => error,
    }
}

#[test]
fn the_server_opens_over_the_packages_installed_before_it_started() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let source = package(&temporary.path().join("source"));
    let first = BuildServer::open(&config).unwrap();
    first.install(&source).unwrap();
    first.install(&source).unwrap();
    drop(first);

    let server = BuildServer::open(&config).unwrap();
    let listing = server.list().unwrap();
    assert_eq!(listed(&server), ["rms_norm vendor 1", "rms_norm vendor 2"]);
    assert!(listing.iter().all(|(_, modules)| modules.is_empty()));
}

#[test]
fn an_installation_numbers_the_next_revision_of_its_operator_and_backend() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();
    let source = package(&temporary.path().join("source"));

    assert_eq!(server.install(&source).unwrap().revision, 1);
    assert_eq!(server.install(&source).unwrap().revision, 2);

    // Each revision keeps the package it was installed from, with the record
    // of its build inputs, under the package root.
    for revision in [1, 2] {
        let directory = revision_directory(&config.package_root, revision);
        assert!(directory.join("package.toml").is_file());
        assert!(directory.join("artifact/module_dtype_bf16.cubin").is_file());
        assert_eq!(Inputs::read(&directory).unwrap().toolchain, "none");
    }
    assert_eq!(server.list().unwrap().len(), 2);
}

/// Every installed revision records its build inputs, which name its modules
/// in the store: a package root holding a revision without its record cannot
/// say what that revision's modules were built from, and the server refuses
/// to open over it rather than serve modules of unknown inputs.
#[test]
fn a_revision_without_the_record_of_its_inputs_refuses_the_server() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let first = BuildServer::open(&config).unwrap();
    first
        .install(&package(&temporary.path().join("source")))
        .unwrap();
    drop(first);

    fs::remove_file(revision_directory(&config.package_root, 1).join(RECORD)).unwrap();
    let refused = refused_open(&config).to_string();
    assert!(refused.contains(RECORD), "{refused}");
}

/// A source the tree shares between packages through a symbolic link is
/// installed as the bytes it names: the revision holds its sources whatever
/// becomes of the tree it was installed from.
#[test]
fn an_installation_holds_what_a_linked_source_names() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();
    let shared = temporary.path().join("shared.cubin");
    fs::write(&shared, b"the shared cubin").unwrap();
    let source = package(&temporary.path().join("linked"));
    let linked = source.join("artifact/module_dtype_bf16.cubin");
    fs::remove_file(&linked).unwrap();
    std::os::unix::fs::symlink(&shared, &linked).unwrap();

    server.install(&source).unwrap();
    let installed =
        revision_directory(&config.package_root, 1).join("artifact/module_dtype_bf16.cubin");
    assert!(
        !installed.is_symlink(),
        "the revision holds a file, not the link"
    );
    fs::remove_file(&shared).unwrap();
    assert_eq!(fs::read(&installed).unwrap(), b"the shared cubin");
}

#[test]
fn a_server_that_is_not_the_upstream_refuses_an_installation_and_names_it() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let upstream = temporary.path().join("leader.sock");
    let mut config = support::config(temporary.path());
    config.upstream = Some(upstream.clone());
    let server = BuildServer::open(&config).unwrap();

    match server.install(&package(&temporary.path().join("source"))) {
        Err(Error::Refused {
            upstream: named, ..
        }) => assert_eq!(
            named.as_deref(),
            Some(upstream.to_str().unwrap()),
            "the refusal names where the upstream is"
        ),
        other => panic!("unexpected response {other:?}"),
    }
    assert!(server.list().unwrap().is_empty());
}

/// An open upstream server is the one writer of its package root: a second
/// upstream server over the root, and a store build into it, are refused
/// before they install anything, while a downstream server, which installs
/// nothing, opens beside it and serves its revisions. Once the upstream is
/// closed, another may open.
#[test]
fn an_open_upstream_server_is_the_one_writer_of_its_package_root() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let upstream = BuildServer::open(&config).unwrap();
    upstream
        .install(&package(&temporary.path().join("source")))
        .unwrap();

    let refused = refused_open(&config);
    assert!(
        matches!(&refused, Error::Io(error) if error.kind() == std::io::ErrorKind::WouldBlock),
        "{refused}"
    );

    let tree = temporary.path().join("kernels");
    package(&tree.join("rms_norm/vendor"));
    let keys = [("rms_norm".to_owned(), "dtype_bf16".to_owned())];
    build_server::build::build(&config, &tree, &keys, Some(&arch()), &[], &mut |_, _, _| {})
        .expect_err("a store build into an open upstream's package root is refused");
    assert_eq!(listed(&upstream), ["rms_norm vendor 1"]);

    let mut downstream = config.clone();
    downstream.upstream = Some(temporary.path().join("leader.sock"));
    downstream.store_roots = vec![Root {
        path: temporary.path().join("downstream-modules"),
        writable: true,
    }];
    assert_eq!(
        listed(&BuildServer::open(&downstream).unwrap()),
        ["rms_norm vendor 1"]
    );

    drop(upstream);
    BuildServer::open(&config).unwrap();
}

#[test]
fn a_package_the_manifest_refuses_is_not_installed() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();
    // The manifest names tuning data the package does not have.
    let broken = manifest().replace(
        "interface_version = 1",
        "interface_version = 1\ntuning = [\"tuning/tactics.json\"]",
    );
    let directory = support::vendor_package(&temporary.path().join("broken"), &broken, &[]);

    assert!(matches!(
        server.install(&directory),
        Err(Error::Manifest { .. })
    ));
    assert!(server.list().unwrap().is_empty());
    assert!(!revision_directory(&config.package_root, 1).exists());
}

#[test]
fn a_package_root_with_an_invalid_package_is_refused_when_the_server_opens() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    write(
        &revision_directory(&config.package_root, 1),
        &[("package.toml", b"op = \"rms_norm\"\n")],
    );

    assert!(matches!(refused_open(&config), Error::Manifest { .. }));
}

/// The manifest of a Triton package, which the Build Server's Python
/// environment compiles.
const TRITON: &str = r#"
op = "rms_norm"
backend = "triton"
interface_version = 1

[toolchain]
kind = "triton"
version = ""

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/rms.py"]
script = "src/build.py"

[kernel]
architectures = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

/// An installation records what builds the revision's modules on this host,
/// so a host without the package's toolchain refuses the installation with
/// the reason the toolchain is missing, and records nothing; a host that has
/// it records the toolchain's identity.
#[test]
fn an_installation_records_the_hosts_toolchain_or_is_refused_without_it() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();
    let source = temporary.path().join("source");
    write(
        &source,
        &[
            ("package.toml", support::for_arch(TRITON).as_bytes()),
            ("src/rms.py", b"# the kernel\n"),
            ("src/build.py", b"# the build module\n"),
        ],
    );

    // What this host has, as the server discovers it when it opens.
    let installed = config.package_root.join("rms_norm/triton/1");
    match Toolchains::discover().identity(ToolchainKind::Triton, &source) {
        Err(missing) => {
            let refused = server
                .install(&source)
                .expect_err("a host without the toolchain installs nothing");
            assert!(
                matches!(
                    &refused,
                    Error::Toolchain {
                        kind: ToolchainKind::Triton,
                        ..
                    }
                ),
                "{refused}"
            );
            assert_eq!(refused.to_string(), missing.to_string());
            assert!(server.list().unwrap().is_empty());
            assert!(!installed.exists());
        }
        Ok(identity) => {
            server.install(&source).unwrap();
            assert_eq!(Inputs::read(&installed).unwrap().toolchain, identity);
        }
    }
}

/// Writes a Cargo workspace whose one member is a scheduling policy package,
/// and returns the member's directory: what a policy is installed from. Its
/// lockfile is the one cargo writes for a workspace without dependencies.
fn policy_workspace(root: &Path) -> PathBuf {
    write(
        root,
        &[
            (
                "Cargo.toml",
                b"[workspace]\nmembers = [\"policy\"]\nresolver = \"3\"\n",
            ),
            (
                "Cargo.lock",
                b"# This file is automatically @generated by Cargo.\n\
                  # It is not intended for manual editing.\n\
                  version = 4\n\n\
                  [[package]]\n\
                  name = \"fixture-policy\"\n\
                  version = \"0.1.0\"\n",
            ),
            (
                "policy/Cargo.toml",
                b"[package]\nname = \"fixture-policy\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\
                  publish = false\n\n[lib]\ncrate-type = [\"cdylib\"]\n",
            ),
            (
                "policy/src/lib.rs",
                b"#[unsafe(no_mangle)]\npub extern \"C\" fn decide() -> i32 {\n    1\n}\n",
            ),
            (
                "policy/package.toml",
                // The crate's files are the ones cargo packages, so the
                // manifest lists none.
                b"op = \"scheduling\"\nbackend = \"fixture\"\ninterface_version = 1\n\n\
                  [toolchain]\nkind = \"cargo\"\nversion = \"\"\n\n\
                  [module]\nform = \"wasm\"\n\n\
                  [source]\nkind = \"wasm\"\nfiles = []\n",
            ),
        ],
    );
    root.join("policy")
}

/// A server whose package root holds the installed policy of
/// [`policy_workspace`], and whose store holds the module a build of that
/// revision committed; returns the server, its configuration and the
/// module's path.
fn installed_policy(temporary: &Path) -> (BuildServer, build_server::ServerConfig, ModulePath) {
    let config = support::config(temporary);
    let server = BuildServer::open(&config).unwrap();
    let source = policy_workspace(&temporary.join("workspace"));
    assert_eq!(server.install(&source).unwrap().revision, 1);

    let inputs = Inputs::read(&config.package_root.join("scheduling/fixture/1"))
        .unwrap()
        .digest();
    let path = ModulePath::new(
        "scheduling",
        "fixture",
        1,
        inputs,
        Target::wasm().as_str(),
        SINGLE_KEY,
    )
    .unwrap();
    let built = temporary.join("built");
    write(&built, &[("module.wasm", b"\0asm a policy")]);
    Store::open(config.store_roots.clone())
        .unwrap()
        .publish(&path, &built, &["module.wasm".to_owned()], 1)
        .unwrap();
    (server, config, path)
}

/// Installing a policy takes the snapshot of its whole workspace and builds
/// nothing; its module is listed as the one wasm module of the package.
#[test]
fn an_installed_policy_holds_its_workspace_and_lists_its_one_wasm_module() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let (server, config, path) = installed_policy(temporary.path());

    let revision = config.package_root.join("scheduling/fixture/1");
    for file in [
        "package.toml",
        "workspace/Cargo.toml",
        "workspace/Cargo.lock",
        "workspace/policy/Cargo.toml",
        "workspace/policy/src/lib.rs",
        "workspace/policy/package.toml",
    ] {
        assert!(
            revision.join(file).is_file(),
            "the revision holds no {file}"
        );
    }
    assert!(!revision.join("module.wasm").exists());

    let listing = server.list().unwrap();
    assert_eq!(listing.len(), 1);
    let (package, modules) = &listing[0];
    assert_eq!(package.op, "scheduling");
    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0].target, Target::wasm());
    assert_eq!(modules[0].descriptor, path.descriptor);
    assert_eq!(modules[0].form, ModuleForm::Wasm);
}

/// A policy is built once, for wasm32, whatever targets the epoch's kernels
/// are built for: `prepare` finds its committed module and builds nothing.
#[test]
fn prepare_takes_a_policy_for_wasm32_whatever_the_kernel_targets() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let (server, _config, path) = installed_policy(temporary.path());

    let mut progress = Vec::new();
    let prepared = server
        .prepare(
            &support::latest("scheduling", "fixture", &[SINGLE_KEY], vec![arch()]),
            &mut |step| progress.push(step),
        )
        .unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].1.revision, 1);
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0].path, path);
    assert_eq!(progress[0].state, BuildState::Committed);
}

/// A policy's module resolves for wasm32 and its one key, and for no GPU
/// architecture and no descriptor key.
#[test]
fn a_policy_module_resolves_for_wasm32_only() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let (server, _config, _path) = installed_policy(temporary.path());
    let revision = support::revision("fixture", 1);

    let resolved = server
        .resolve(
            "scheduling",
            &revision,
            &Target::wasm(),
            SINGLE_KEY,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(resolved.form, ModuleForm::Wasm);
    assert_eq!(
        support::file_of(&resolved, "module.wasm"),
        b"\0asm a policy"
    );
    assert_eq!(resolved.files.len(), 1);
    assert!(resolved.metadata.is_none());
    assert!(resolved.entries.is_empty());

    assert!(matches!(
        server.resolve("scheduling", &revision, &arch(), SINGLE_KEY, &mut |_| {}),
        Err(Error::Build { .. })
    ));
    assert!(matches!(
        server.resolve(
            "scheduling",
            &revision,
            &Target::wasm(),
            "dtype_bf16",
            &mut |_| {}
        ),
        Err(Error::UnsupportedDescriptor { .. })
    ));
}

/// A server with the `rms_norm` package installed twice.
fn two_revisions(temporary: &Path) -> BuildServer {
    let server = BuildServer::open(&support::config(temporary)).unwrap();
    let source = package(&temporary.join("source"));
    server.install(&source).unwrap();
    server.install(&source).unwrap();
    server
}

#[test]
fn prepare_refuses_an_operator_with_no_installed_backend() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = two_revisions(temporary.path());
    let request = support::latest("rms_norm", "triton", &[], vec![arch()]);

    assert!(matches!(
        server.prepare(&request, &mut |_| {}),
        Err(Error::NotInstalled { .. })
    ));
}

#[test]
fn prepare_refuses_a_revision_that_is_not_installed() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = two_revisions(temporary.path());
    let request = PrepareRequest {
        targets: vec![arch()],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("rms_norm"),
            selector: Selector::Fixed(support::revision("vendor", 9)),
        }],
        descriptors: Vec::new(),
    };

    match server.prepare(&request, &mut |_| {}) {
        Err(Error::UnknownRevision { revision, .. }) => assert_eq!(revision, 9),
        other => panic!("unexpected response {other:?}"),
    }
}

/// `prepare` selects a backend's latest revision, and what it returns is the
/// package revision that will run: the package's parameter defaults fill
/// every argument the request leaves out, and an argument it states replaces
/// its default.
#[test]
fn prepare_selects_the_latest_revision_with_its_parameter_defaults() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = two_revisions(temporary.path());
    let mut arguments = toml::Table::new();
    arguments.insert("tile".to_owned(), toml::Value::Integer(512));
    let request = PrepareRequest {
        targets: vec![arch()],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("rms_norm"),
            selector: Selector::Latest {
                backend: "vendor".to_owned(),
                arguments,
            },
        }],
        descriptors: Vec::new(),
    };

    let prepared = server.prepare(&request, &mut |_| {}).unwrap();
    let revision: &PackageRevision = &prepared[0].1;
    assert_eq!(revision.revision, 2);
    assert_eq!(
        revision.arguments.get("tile"),
        Some(&toml::Value::Integer(512))
    );
    assert_eq!(
        revision.arguments.get("epsilon"),
        Some(&toml::Value::Float(1e-6))
    );
}

/// A key the selected revision does not cover names the descriptor and the
/// revision that fails to cover it, before anything is built.
#[test]
fn prepare_refuses_a_descriptor_the_selected_revision_does_not_cover() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = two_revisions(temporary.path());
    let request = support::latest("rms_norm", "vendor", &["dtype_fp32"], vec![arch()]);

    match server.prepare(&request, &mut |_| {}) {
        Err(Error::UnsupportedDescriptor {
            revision,
            descriptor,
            ..
        }) => {
            assert_eq!(revision, 2, "the latest revision was chosen");
            assert_eq!(descriptor, "dtype_fp32");
        }
        other => panic!("unexpected response {other:?}"),
    }
}

#[test]
fn prepare_refuses_descriptors_of_an_operator_no_revision_fills() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = two_revisions(temporary.path());
    let mut request = support::latest("rms_norm", "vendor", &[], vec![arch()]);
    request.descriptors[0].slot = ModuleSlot::all("paged_attention");
    request.descriptors[0].keys = vec!["dtype_bf16".to_owned()];

    assert!(matches!(
        server.prepare(&request, &mut |_| {}),
        Err(Error::Protocol(_))
    ));
}

#[test]
fn resolve_refuses_a_target_the_package_does_not_build_for() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = two_revisions(temporary.path());

    match server.resolve(
        "rms_norm",
        &support::revision("vendor", 1),
        &other_arch(),
        "dtype_bf16",
        &mut |_| {},
    ) {
        Err(Error::Build { target, .. }) => assert_eq!(target, other_arch().to_string()),
        other => panic!("unexpected response {other:?}"),
    }
}

#[test]
fn resolve_refuses_a_descriptor_the_revision_does_not_cover() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = two_revisions(temporary.path());

    assert!(matches!(
        server.resolve(
            "rms_norm",
            &support::revision("vendor", 1),
            &arch(),
            "dtype_fp32",
            &mut |_| {},
        ),
        Err(Error::UnsupportedDescriptor { .. })
    ));
}

/// An executable that fills a whole interface with one module, built for
/// the host's platform under the one key.
const SAMPLER: &str = r#"
op = "sampler"
backend = "process"
interface_version = 1

[toolchain]
kind = "none"
version = ""

[module]
form = "executable"

[source]
kind = "executable"
files = ["bin/sampler"]
"#;

/// A module set resolves from a store a release shipped: the server serves
/// the store's package revisions and their modules read-only, selects among
/// them, builds nothing, writes nothing into the shipped store, and numbers
/// its own installations after the shipped revisions.
#[test]
fn a_module_set_resolves_from_a_shipped_store() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();

    // What a release ships: a store build of a kernel key and of a
    // single-module package.
    let shipped = temporary.path().join("shipped");
    let tree = temporary.path().join("kernels");
    package(&tree.join("rms_norm/vendor"));
    let sampler = temporary.path().join("sampler");
    write(
        &sampler,
        &[
            ("package.toml", SAMPLER.as_bytes()),
            ("bin/sampler", b"#!/bin/sh\nexit 0\n"),
        ],
    );
    let mut store = support::config(&shipped);
    store.cache = temporary.path().join("cache");
    let failures = build_server::build::build(
        &store,
        &tree,
        &[("rms_norm".to_owned(), "dtype_bf16".to_owned())],
        Some(&arch()),
        std::slice::from_ref(&sampler),
        &mut |_, _, _| {},
    )
    .unwrap();
    assert!(failures.is_empty(), "{failures:?}");

    // A host whose own roots hold nothing.
    let mut config = support::config(&temporary.path().join("host"));
    config.read_only_stores = vec![shipped.clone()];
    let server = BuildServer::open(&config).unwrap();

    // The apply's selection: every candidate of each interface is a shipped
    // revision.
    let candidates = |op: &str| {
        server
            .cover(&CoverRequest {
                op: op.to_owned(),
                targets: Vec::new(),
                keys: Vec::new(),
                preferred: Vec::new(),
                fixed: Vec::new(),
            })
            .unwrap()
            .candidates
            .into_iter()
            .map(|package| format!("{} {}", package.backend, package.revision))
            .collect::<Vec<_>>()
    };
    assert_eq!(candidates("rms_norm"), ["vendor 1"]);
    assert_eq!(candidates("sampler"), ["process 1"]);

    // Each module of the set resolves from the shipped store; nothing is
    // built.
    let mut progress = Vec::new();
    let kernel = server
        .resolve(
            "rms_norm",
            &support::revision("vendor", 1),
            &arch(),
            "dtype_bf16",
            &mut |step| progress.push(step),
        )
        .expect("the shipped kernel module resolves");
    assert_eq!(
        support::file_of(&kernel, "module.cubin"),
        support::cubin("dtype_bf16")
    );
    let executable = server
        .resolve(
            "sampler",
            &support::revision("process", 1),
            &Target::host(),
            SINGLE_KEY,
            &mut |step| progress.push(step),
        )
        .expect("the shipped executable resolves");
    assert!(executable.files[0].path.starts_with(&shipped));
    assert!(progress.is_empty(), "a module the store holds is not built");

    // The host's own store root stays empty.
    assert!(
        fs::read_dir(&config.store_roots[0].path)
            .unwrap()
            .next()
            .is_none()
    );

    // An installation on this host numbers after the shipped revision and
    // lands in its own package root.
    let installed = server
        .install(&package(&temporary.path().join("source")))
        .unwrap();
    assert_eq!(installed.revision, 2);
    assert!(
        revision_directory(&config.package_root, 2)
            .join("package.toml")
            .is_file()
    );
    assert!(!revision_directory(&shipped.join("packages"), 2).exists());
}
