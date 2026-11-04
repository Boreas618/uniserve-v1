//! The server's four operations (design section 6.3): what an installation
//! numbers and where it lands, what a server that is not the upstream
//! answers, and the errors a request gets before any build starts.

use std::path::{Path, PathBuf};

use program_server::builder::Limits;
use program_server::protocol::{Descriptors, PrepareRequest, ProgramRequest, Selector};
use program_server::server::{ProgramServer, ServerConfig};
use program_server::store::Root;
use program_server::target::{Device, Target};
use program_server::toolchain::ToolchainConfig;
use program_server::{Error, Program, ProgramSlot};

/// The manifest of a Triton `rms_norm` package.
const MANIFEST: &str = r#"
op = "rms_norm"
backend = "triton"

[toolchain]
kind = "python"
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
        socket: temporary.join("program-server.sock"),
        package_root: temporary.join("packages"),
        store_roots: vec![Root {
            path: temporary.join("store"),
            writable: true,
        }],
        upstream: upstream.map(Path::to_path_buf),
        limits: Limits::default(),
        toolchains: ToolchainConfig {
            nvcc: None,
            python: PathBuf::from("/python/bin/python"),
            wasmtime: None,
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
}

#[test]
fn the_server_opens_over_the_packages_installed_before_it_started() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 3);

    let server = ProgramServer::open(&configuration).unwrap();
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
    let server = ProgramServer::open(&configuration).unwrap();
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

#[test]
fn a_server_that_is_not_the_upstream_refuses_an_installation_and_names_it() {
    let temporary = tempfile::tempdir().unwrap();
    let upstream = temporary.path().join("leader.sock");
    let configuration = config(temporary.path(), Some(&upstream));
    let server = ProgramServer::open(&configuration).unwrap();
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
    let server = ProgramServer::open(&configuration).unwrap();
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
        ProgramServer::open(&configuration),
        Err(Error::Manifest { .. })
    ));
}

/// The manifest of a wasm scheduling policy.
const POLICY: &str = r#"
op = "scheduling"
backend = "fcfs"

[toolchain]
kind = "wasm"
version = "wasmtime-31"

[module]
form = "wasm"

[source]
kind = "wasm"
files = ["src/lib.rs"]

[kernel]
architectures = ["wasm32"]
descriptors = { keys = ["default"] }
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
    let module = configuration.store_roots[0]
        .path
        .join("scheduling/fcfs/1/wasm32/default");
    std::fs::create_dir_all(&module).unwrap();
    std::fs::write(module.join("module.cwasm"), b"a precompiled module").unwrap();
    std::fs::write(module.join(".committed"), b"").unwrap();

    let server = ProgramServer::open(&configuration).unwrap();
    let listing = server.list().unwrap();
    assert_eq!(listing.packages.len(), 1);
    let package = &listing.packages[0];
    assert_eq!(package.op, "scheduling");
    assert_eq!(package.modules.len(), 1);
    assert_eq!(package.modules[0].target.as_str(), "wasm32");
    assert_eq!(package.modules[0].descriptor, "default");
    assert_eq!(
        package.modules[0].form,
        program_server::manifest::ModuleForm::Wasm
    );
}

#[test]
fn prepare_refuses_an_operator_with_no_installed_backend() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    let server = ProgramServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("rms_norm"),
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
    let server = ProgramServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("rms_norm"),
            selector: Selector::Fixed(Program {
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
    let server = ProgramServer::open(&configuration).unwrap();
    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("rms_norm"),
            selector: Selector::Latest {
                backend: "triton".into(),
                arguments: toml::Table::new(),
            },
        }],
        // The package covers `dtype_bf16` only: asking for another value
        // names the descriptor and the program that fails to cover it.
        descriptors: vec![Descriptors {
            op: "rms_norm".into(),
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
fn prepare_refuses_descriptors_of_an_operator_no_program_fills() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 1);
    let server = ProgramServer::open(&configuration).unwrap();
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
            op: "paged_attention".into(),
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
    let server = ProgramServer::open(&configuration).unwrap();
    let program = Program {
        backend: "triton".into(),
        revision: 1,
        arguments: toml::Table::new(),
    };

    match server.resolve(
        "rms_norm",
        &program,
        &Target::new("sm_90a"),
        "dtype_bf16",
        &mut |_| {},
    ) {
        Err(Error::Build { target, .. }) => assert_eq!(target, "sm_90a"),
        other => panic!("unexpected answer {other:?}"),
    }
}

#[test]
fn resolve_refuses_a_descriptor_the_program_does_not_cover() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path(), None);
    installed(&configuration.package_root, 1);
    let server = ProgramServer::open(&configuration).unwrap();
    let program = Program {
        backend: "triton".into(),
        revision: 1,
        arguments: toml::Table::new(),
    };

    assert!(matches!(
        server.resolve(
            "rms_norm",
            &program,
            &Target::new("sm_100a"),
            "dtype_fp32",
            &mut |_| {},
        ),
        Err(Error::UnsupportedDescriptor { .. })
    ));
}
