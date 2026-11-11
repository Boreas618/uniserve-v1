//! The server's operations (design section 6.3): what an installation
//! numbers and where it lands, what a server that is not the upstream
//! answers, and the errors a request gets before any build starts.

use std::path::{Path, PathBuf};

use build_server::Error;
use build_server::builder::{BuildState, Limits};
use build_server::inputs::Inputs;
use build_server::protocol::{Descriptors, PackageRequest, PrepareRequest, Selector};
use build_server::server::{BuildServer, ServerConfig};
use build_server::store::{ModulePath, Root};
use build_server::target::{Device, Target};
use build_server::toolchain::ToolchainConfig;
use proto::module::{ModuleSlot, PackageRevision};

/// The manifest of a Triton `rms_norm` package.
const MANIFEST: &str = r#"
op = "rms_norm"
backend = "triton"
interface_version = 1

[toolchain]
kind = "triton"
version = "3.7.1"

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/rms.py"]
script = "src/compile.py"
kernel = "rms_norm"

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] } }

[parameters]
tile = 1024
"#;

/// Writes a package into `directory` and answers it.
fn package(directory: &Path) -> PathBuf {
    std::fs::create_dir_all(directory.join("src")).unwrap();
    std::fs::write(directory.join("package.toml"), MANIFEST).unwrap();
    std::fs::write(directory.join("src/rms.py"), b"# the kernel\n").unwrap();
    std::fs::write(directory.join("src/compile.py"), b"# the driver\n").unwrap();
    directory.to_path_buf()
}

/// A server configuration over a fresh package root and module store.
fn config(temporary: &Path, upstream: Option<&Path>) -> ServerConfig {
    ServerConfig {
        socket: temporary.join("build-server.sock"),
        package_root: temporary.join("packages"),
        store_roots: vec![Root {
            path: temporary.join("store"),
            writable: true,
        }],
        upstream: upstream.map(Path::to_path_buf),
        read_only_stores: Vec::new(),
        limits: Limits::default(),
        toolchains: ToolchainConfig {
            nvcc: None,
            python: Some(PathBuf::from("/python/bin/python")),
            wasm: None,
            cargo: None,
            max_jobs: 1,
        },
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    }
}

/// Installs `MANIFEST`'s package as revision `revision` under `package_root`,
/// as an earlier installation would have left it.
fn installed(package_root: &Path, revision: u32) {
    let directory = package_root
        .join("rms_norm")
        .join("triton")
        .join(revision.to_string());
    package(&directory);
    record(&directory);
}

/// Records the build inputs of the revision laid out in `directory`, as its
/// installation would have, and answers their digest.
fn record(directory: &Path) -> String {
    // The digest of the revision's own tree: what the installation of a package that is its
    // directory records. A policy's is its crate's snapshot, which these revisions do not hold,
    // and no test here reads the digest for more than naming the modules it stores.
    let inputs = Inputs {
        package: build_server::snapshot::tree_digest(directory).unwrap(),
        toolchain: "triton 3.7.1".to_owned(),
    };
    inputs.write(directory).unwrap();
    inputs.digest()
}

#[test]
fn the_server_opens_over_the_packages_installed_before_it_started() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 3);

    let server = BuildServer::open(&configuration).unwrap();
    let listing = server.list().unwrap();
    assert_eq!(listing.packages.len(), 1);
    assert_eq!(listing.packages[0].op, "rms_norm");
    assert_eq!(listing.packages[0].revision, 3);
    assert!(listing.packages[0].modules.is_empty());
}

#[test]
fn an_installation_numbers_the_next_revision_of_its_operator_and_backend() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let server = BuildServer::open(&configuration).unwrap();
    let first = package(&temporary.path().join("first"));

    let installed = server.install(&first).unwrap();
    assert_eq!(installed.package.revision, 1);
    let again = server.install(&first).unwrap();
    assert_eq!(again.package.revision, 2);

    // Each revision keeps the package it was installed from, under the
    // package root.
    for revision in [1, 2] {
        let directory = configuration
            .package_root
            .join("rms_norm")
            .join("triton")
            .join(revision.to_string());
        assert!(directory.join("package.toml").is_file());
        assert!(directory.join("src/rms.py").is_file());
    }
    assert_eq!(server.list().unwrap().packages.len(), 2);
}

/// A revision whose installation recorded no build inputs is not served, because modules found
/// under its number may have been built from other inputs; its number stays taken, so the next
/// installation is numbered after it.
#[test]
fn a_revision_without_a_record_of_its_inputs_is_not_served_and_keeps_its_number() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    package(&configuration.package_root.join("rms_norm/triton/1"));

    let server = BuildServer::open(&configuration).unwrap();
    assert!(server.list().unwrap().packages.is_empty());
    let installed = server
        .install(&package(&temporary.path().join("again")))
        .unwrap();
    assert_eq!(installed.package.revision, 2);
    assert_eq!(server.list().unwrap().packages.len(), 1);
}

/// A source the tree shares between packages through a symbolic link is installed as the bytes it
/// names: the revision holds its sources whatever becomes of the tree it was installed from.
#[test]
fn an_installation_holds_what_a_linked_source_names() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let server = BuildServer::open(&configuration).unwrap();
    let shared = temporary.path().join("shared.py");
    std::fs::write(&shared, b"# the shared kernel\n").unwrap();
    let source = package(&temporary.path().join("linked"));
    std::fs::remove_file(source.join("src/rms.py")).unwrap();
    std::os::unix::fs::symlink(&shared, source.join("src/rms.py")).unwrap();

    server.install(&source).unwrap();
    let installed = configuration
        .package_root
        .join("rms_norm/triton/1/src/rms.py");
    assert!(
        !installed.is_symlink(),
        "the revision holds a file, not the link"
    );
    assert_eq!(std::fs::read(&installed).unwrap(), b"# the shared kernel\n");
    std::fs::remove_file(&shared).unwrap();
    assert_eq!(std::fs::read(&installed).unwrap(), b"# the shared kernel\n");
}

#[test]
fn a_server_that_is_not_the_upstream_refuses_an_installation_and_names_it() {
    let temporary = tempfile::tempdir().unwrap();
    let upstream = temporary.path().join("leader.sock");
    let configuration = config(temporary.path(), Some(&upstream));
    let server = BuildServer::open(&configuration).unwrap();
    let directory = package(&temporary.path().join("package"));

    match server.install(&directory) {
        Err(Error::Refused {
            upstream: named, ..
        }) => assert_eq!(
            named.as_deref(),
            Some(upstream.to_str().unwrap()),
            "the refusal names where the upstream is"
        ),
        other => panic!("unexpected answer {other:?}"),
    }
}

#[test]
fn a_package_the_manifest_refuses_is_not_installed() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let server = BuildServer::open(&configuration).unwrap();
    let directory = temporary.path().join("broken");
    std::fs::create_dir_all(directory.join("src")).unwrap();
    std::fs::write(directory.join("package.toml"), MANIFEST).unwrap();
    // `src/compile.py` is missing: the manifest names a file the package
    // does not have.

    assert!(matches!(
        server.install(&directory),
        Err(Error::Manifest { .. })
    ));
    assert!(server.list().unwrap().packages.is_empty());
}

#[test]
fn a_package_root_with_an_invalid_package_is_refused_when_the_server_opens() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let directory = configuration
        .package_root
        .join("rms_norm")
        .join("triton")
        .join("1");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("package.toml"), "op = \"rms_norm\"\n").unwrap();

    assert!(matches!(
        BuildServer::open(&configuration),
        Err(Error::Manifest { .. })
    ));
}

/// The manifest of a wasm scheduling policy.
const POLICY: &str = r#"
op = "scheduling"
backend = "fcfs"
interface_version = 1

[toolchain]
kind = "wasm"
version = "wasmtime-31"

[module]
form = "wasm"

[source]
kind = "wasm"
files = ["src/lib.rs"]
"#;

#[test]
fn a_policy_module_is_listed_as_the_wasm_module_it_is() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let package_root = configuration
        .package_root
        .join("scheduling")
        .join("fcfs")
        .join("1");
    std::fs::create_dir_all(package_root.join("src")).unwrap();
    std::fs::write(package_root.join("package.toml"), POLICY).unwrap();
    std::fs::write(package_root.join("src/lib.rs"), b"// the policy\n").unwrap();
    let inputs = record(&package_root);
    let module = configuration.store_roots[0]
        .path
        .join(format!("scheduling/fcfs/1/{inputs}/wasm32/default"));
    std::fs::create_dir_all(&module).unwrap();
    std::fs::write(module.join("module.wasm"), b"a precompiled module").unwrap();
    std::fs::write(module.join(".committed"), b"interface_version = 1\n").unwrap();

    let server = BuildServer::open(&configuration).unwrap();
    let listing = server.list().unwrap();
    assert_eq!(listing.packages.len(), 1);
    let package = &listing.packages[0];
    assert_eq!(package.op, "scheduling");
    assert_eq!(package.modules.len(), 1);
    assert_eq!(package.modules[0].target.as_str(), "wasm32");
    assert_eq!(package.modules[0].descriptor, "default");
    assert_eq!(
        package.modules[0].form,
        build_server::manifest::ModuleForm::Wasm
    );
}

#[test]
fn prepare_builds_a_scheduling_policy_once_for_wasm32() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let package_root = configuration
        .package_root
        .join("scheduling")
        .join("fcfs")
        .join("1");
    std::fs::create_dir_all(package_root.join("src")).unwrap();
    std::fs::write(package_root.join("package.toml"), POLICY).unwrap();
    std::fs::write(package_root.join("src/lib.rs"), b"// the policy\n").unwrap();
    let inputs = record(&package_root);
    let module = configuration.store_roots[0]
        .path
        .join(format!("scheduling/fcfs/1/{inputs}/wasm32/default"));
    std::fs::create_dir_all(&module).unwrap();
    std::fs::write(module.join("module.wasm"), b"a precompiled module").unwrap();
    std::fs::write(module.join(".committed"), b"interface_version = 1\n").unwrap();

    let server = BuildServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        // The daemon names no target: each host builds for its own devices,
        // and a policy is built for wasm32 whatever those are.
        targets: Vec::new(),
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("scheduling"),
            selector: Selector::Latest {
                backend: "fcfs".into(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            slot: proto::module::ModuleSlot::all("scheduling"),
            keys: vec!["default".into()],
        }],
    };
    let mut progress = Vec::new();
    let prepared = server
        .prepare(&request, &mut |step| progress.push(step))
        .unwrap();

    assert_eq!(prepared.packages.len(), 1);
    assert_eq!(progress.len(), 1);
    assert_eq!(
        progress[0].path,
        ModulePath::new("scheduling", "fcfs", 1, inputs, "wasm32", "default").unwrap()
    );
    assert_eq!(progress[0].state, BuildState::Committed);
}

#[test]
fn a_policy_module_resolves_for_wasm32_only() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let package_root = configuration
        .package_root
        .join("scheduling")
        .join("fcfs")
        .join("1");
    std::fs::create_dir_all(package_root.join("src")).unwrap();
    std::fs::write(package_root.join("package.toml"), POLICY).unwrap();
    std::fs::write(package_root.join("src/lib.rs"), b"// the policy\n").unwrap();
    let inputs = record(&package_root);
    let module = configuration.store_roots[0]
        .path
        .join(format!("scheduling/fcfs/1/{inputs}/wasm32/default"));
    std::fs::create_dir_all(&module).unwrap();
    std::fs::write(module.join("module.wasm"), b"a precompiled module").unwrap();
    std::fs::write(module.join(".committed"), b"interface_version = 1\n").unwrap();

    let server = BuildServer::open(&configuration).unwrap();
    let revision = PackageRevision {
        backend: "fcfs".into(),
        revision: 1,
        arguments: toml::Table::new(),
    };
    let resolved = server
        .resolve(
            "scheduling",
            &revision,
            &Target::new("wasm32"),
            "default",
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(resolved.form, build_server::manifest::ModuleForm::Wasm);
    assert_eq!(resolved.files.len(), 1);
    assert_eq!(resolved.files[0].name, "module.wasm");
    assert!(resolved.metadata.is_none());
    assert!(resolved.entries.is_empty());

    // A policy is not built for a GPU architecture, and a policy names no
    // descriptor other than `default`.
    assert!(matches!(
        server.resolve(
            "scheduling",
            &revision,
            &Target::new("sm_100a"),
            "default",
            &mut |_| {}
        ),
        Err(Error::Build { .. })
    ));
    assert!(matches!(
        server.resolve(
            "scheduling",
            &revision,
            &Target::new("wasm32"),
            "dtype_bf16",
            &mut |_| {}
        ),
        Err(Error::UnsupportedDescriptor { .. })
    ));
}

#[test]
fn prepare_refuses_an_operator_with_no_installed_backend() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let server = BuildServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("rms_norm"),
            selector: Selector::Latest {
                backend: "triton".into(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: Vec::new(),
    };

    assert!(matches!(
        server.prepare(&request, &mut |_| {}),
        Err(Error::NotInstalled { .. })
    ));
}

#[test]
fn prepare_refuses_a_revision_that_is_not_installed() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 1);
    let server = BuildServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("rms_norm"),
            selector: Selector::Fixed(PackageRevision {
                backend: "triton".into(),
                revision: 9,
                arguments: toml::Table::new(),
            }),
        }],
        descriptors: Vec::new(),
    };

    match server.prepare(&request, &mut |_| {}) {
        Err(Error::UnknownRevision { revision, .. }) => assert_eq!(revision, 9),
        other => panic!("unexpected answer {other:?}"),
    }
}

#[test]
fn prepare_selects_the_latest_revision_of_a_backend() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 1);
    installed(&configuration.package_root, 2);
    let server = BuildServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("rms_norm"),
            selector: Selector::Latest {
                backend: "triton".into(),
                arguments: toml::Table::new(),
            },
        }],
        // The package covers `dtype_bf16` only: asking for another value
        // names the descriptor and the package revision that fails to cover it.
        descriptors: vec![Descriptors {
            slot: proto::module::ModuleSlot::all("rms_norm"),
            keys: vec!["dtype_fp32".into()],
        }],
    };

    match server.prepare(&request, &mut |_| {}) {
        Err(Error::UnsupportedDescriptor {
            revision,
            descriptor,
            ..
        }) => {
            assert_eq!(revision, 2, "the latest revision was chosen");
            assert_eq!(descriptor, "dtype_fp32");
        }
        other => panic!("unexpected answer {other:?}"),
    }
}

#[test]
fn prepare_refuses_descriptors_of_an_operator_no_revision_fills() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 1);
    let server = BuildServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("rms_norm"),
            selector: Selector::Latest {
                backend: "triton".into(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            slot: proto::module::ModuleSlot::all("paged_attention"),
            keys: vec!["dtype_bf16".into()],
        }],
    };

    assert!(matches!(
        server.prepare(&request, &mut |_| {}),
        Err(Error::Protocol(_))
    ));
}

#[test]
fn resolve_refuses_a_target_the_package_does_not_build_for() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 1);
    let server = BuildServer::open(&configuration).unwrap();
    let revision = PackageRevision {
        backend: "triton".into(),
        revision: 1,
        arguments: toml::Table::new(),
    };

    match server.resolve(
        "rms_norm",
        &revision,
        &Target::new("sm_90a"),
        "dtype_bf16",
        &mut |_| {},
    ) {
        Err(Error::Build { target, .. }) => assert_eq!(target, "sm_90a"),
        other => panic!("unexpected answer {other:?}"),
    }
}

#[test]
fn resolve_refuses_a_descriptor_the_revision_does_not_cover() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 1);
    let server = BuildServer::open(&configuration).unwrap();
    let revision = PackageRevision {
        backend: "triton".into(),
        revision: 1,
        arguments: toml::Table::new(),
    };

    assert!(matches!(
        server.resolve(
            "rms_norm",
            &revision,
            &Target::new("sm_100a"),
            "dtype_fp32",
            &mut |_| {},
        ),
        Err(Error::UnsupportedDescriptor { .. })
    ));
}

/// The manifest of an engine package: a crate cargo builds to one executable.
const ENGINE: &str = r#"
op = "engine"
backend = "uniserve"
interface_version = 1

[toolchain]
kind = "cargo"
version = "1.99.1"

[module]
form = "executable"

[source]
kind = "cargo"
files = []
binary = "uniserve-engine"
"#;

/// A module set resolves from a store a release shipped, with no toolchain on the host: the
/// server serves the store's package revisions and their modules read-only, selects among them,
/// builds nothing, writes nothing into the shipped store, and numbers its own installations
/// after the shipped revisions.
#[test]
fn a_module_set_resolves_from_a_shipped_store_with_no_toolchain() {
    let temporary = tempfile::tempdir().unwrap();
    let shipped = temporary.path().join("shipped");

    // What `build --store` leaves: the installed revisions with their records, and the modules
    // built from them under the digest of those records.
    let kernel = shipped.join("packages/rms_norm/triton/1");
    package(&kernel);
    let kernel_inputs = record(&kernel);
    let kernel_module = shipped.join(format!(
        "modules/rms_norm/triton/1/{kernel_inputs}/sm_100a/dtype_bf16"
    ));
    std::fs::create_dir_all(&kernel_module).unwrap();
    std::fs::write(kernel_module.join("module.cubin"), b"a cubin").unwrap();
    std::fs::write(
        kernel_module.join("metadata.json"),
        br#"{"name": "rms_norm"}"#,
    )
    .unwrap();
    std::fs::write(kernel_module.join(".committed"), b"interface_version = 1\n").unwrap();

    let engine = shipped.join("packages/engine/uniserve/1");
    std::fs::create_dir_all(&engine).unwrap();
    std::fs::write(engine.join("package.toml"), ENGINE).unwrap();
    let engine_inputs = record(&engine);
    let host = Target::host();
    let engine_module = shipped.join(format!(
        "modules/engine/uniserve/1/{engine_inputs}/{host}/default"
    ));
    std::fs::create_dir_all(&engine_module).unwrap();
    std::fs::write(engine_module.join("uniserve-engine"), b"an executable").unwrap();
    std::fs::write(engine_module.join(".committed"), b"interface_version = 1\n").unwrap();

    // A host with no compiler at all, and roots of its own that hold nothing.
    let mut configuration = config(temporary.path(), None);
    configuration.read_only_stores = vec![shipped.clone()];
    configuration.toolchains.cargo = Some(PathBuf::from("/no/toolchain/cargo"));
    let server = BuildServer::open(&configuration).unwrap();

    // The apply's selection: every candidate of each interface is a shipped revision.
    let candidates = |op: &str| {
        server
            .cover(&build_server::CoverRequest {
                op: op.to_owned(),
                targets: Vec::new(),
                keys: Vec::new(),
                named: Vec::new(),
                fixed: Vec::new(),
            })
            .unwrap()
            .candidates
            .into_iter()
            .map(|package| (package.backend, package.revision))
            .collect::<Vec<_>>()
    };
    assert_eq!(candidates("rms_norm"), vec![("triton".to_owned(), 1)]);
    assert_eq!(candidates("engine"), vec![("uniserve".to_owned(), 1)]);

    // Each module of the set resolves from the shipped store; nothing is queued to build.
    let revision = |backend: &str| PackageRevision {
        backend: backend.to_owned(),
        revision: 1,
        arguments: toml::Table::new(),
    };
    let mut progress = Vec::new();
    let kernel = server
        .resolve(
            "rms_norm",
            &revision("triton"),
            &Target::new("sm_100a"),
            "dtype_bf16",
            &mut |step| progress.push(step),
        )
        .expect("the shipped kernel module resolves");
    assert_eq!(kernel.interface_version, 1);
    assert_eq!(kernel.files.len(), 2);
    let engine = server
        .resolve(
            "engine",
            &revision("uniserve"),
            &host,
            build_server::manifest::SINGLE_KEY,
            &mut |step| progress.push(step),
        )
        .expect("the shipped engine module resolves");
    assert_eq!(engine.files[0].path, engine_module.join("uniserve-engine"));
    assert!(progress.is_empty(), "a module the store holds is not built");

    // The shipped store is read, never written, and the host's own roots stay empty.
    assert!(
        std::fs::read_dir(&configuration.store_roots[0].path)
            .unwrap()
            .next()
            .is_none()
    );

    // An installation on this host numbers after the shipped revision and lands in its own
    // package root.
    let source = package(&temporary.path().join("source"));
    let installed = server.install(&source).unwrap();
    assert_eq!(installed.package.revision, 2);
    assert!(
        configuration
            .package_root
            .join("rms_norm/triton/2/package.toml")
            .is_file()
    );
    assert!(!shipped.join("packages/rms_norm/triton/2").exists());
}
