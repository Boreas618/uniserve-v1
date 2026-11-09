//! `cover` (design section 6.3): which installed package builds each of an
//! operator's descriptor keys. Every fixture package declares its own
//! descriptors in `[kernel.descriptors]`, so the answer is read from the
//! manifests alone and nothing is compiled.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use program_server::builder::Limits;
use program_server::protocol::{Client, CoverRequest, Coverage, Covered, Package};
use program_server::server::{ProgramServer, ServerConfig};
use program_server::store::{ModulePath, Root, Store};
use program_server::target::{Device, Target};
use program_server::toolchain::ToolchainConfig;

/// The operator every fixture package implements.
const OP: &str = "dense_attention";

/// A BF16 key, an FP32 key and an FP16 key of the operator.
const BF16: &str = "dtype_bf16_head_dim_128";
const F32: &str = "dtype_f32_head_dim_128";
const F16: &str = "dtype_f16_head_dim_128";

/// The manifest of a Triton package of [`OP`] under `backend`, covering the
/// dtypes `dtypes` at head width 128 and building for `architecture`.
fn manifest(backend: &str, dtypes: &[&str], architecture: &str) -> String {
    let dtypes: Vec<String> = dtypes.iter().map(|dtype| format!("\"{dtype}\"")).collect();
    format!(
        r#"
op = "{OP}"
backend = "{backend}"

[toolchain]
kind = "triton"
version = "3.7.1"

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/kernel.py"]
script = "src/compile.py"
kernel = "attention"

[kernel]
architectures = ["{architecture}"]
descriptors = {{ dimensions = {{ dtype = [{}], head_dim = ["128"] }} }}
"#,
        dtypes.join(", ")
    )
}

/// Writes the package `manifest` into `directory`.
fn write_package(directory: &Path, manifest: &str) {
    std::fs::create_dir_all(directory.join("src")).unwrap();
    std::fs::write(directory.join("package.toml"), manifest).unwrap();
    std::fs::write(directory.join("src/kernel.py"), b"# the kernel\n").unwrap();
    std::fs::write(directory.join("src/compile.py"), b"# the driver\n").unwrap();
}

/// Installs a package of `backend` as revision `revision`, as an earlier
/// installation would have left it under the package root.
fn installed(configuration: &ServerConfig, backend: &str, revision: u32, manifest: &str) {
    let directory = configuration
        .package_root
        .join(OP)
        .join(backend)
        .join(revision.to_string());
    write_package(&directory, manifest);
}

/// Commits a module of `backend`'s `revision` for `key` on `sm_100a`, as a
/// build ahead of this server would have left it.
fn committed(configuration: &ServerConfig, backend: &str, revision: u32, key: &str) {
    let store = Store::open(configuration.store_roots.clone()).unwrap();
    let workspace = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("module.cubin"), b"cubin").unwrap();
    let path = ModulePath::new(OP, backend, revision, "sm_100a", key).unwrap();
    store
        .publish(&path, workspace.path(), &["module.cubin".to_owned()])
        .unwrap();
}

/// A server configuration over a fresh package root and module store, for a
/// host of one `sm_100a` device.
fn config(temporary: &Path) -> ServerConfig {
    ServerConfig {
        socket: temporary.join("program-server.sock"),
        package_root: temporary.join("packages"),
        store_roots: vec![Root {
            path: temporary.join("store"),
            writable: true,
        }],
        upstream: None,
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

/// A `cover` of `keys` for `sm_100a`.
fn request(keys: &[&str]) -> CoverRequest {
    CoverRequest {
        op: OP.to_owned(),
        targets: vec![Target::new("sm_100a")],
        keys: keys.iter().map(|key| (*key).to_owned()).collect(),
    }
}

/// The entry of an answer naming `backend`'s `revision` with `keys`.
fn covered(backend: &str, revision: u32, keys: &[&str]) -> Covered {
    Covered {
        package: Package {
            op: OP.to_owned(),
            backend: backend.to_owned(),
            revision,
        },
        keys: keys.iter().map(|key| (*key).to_owned()).collect(),
    }
}

/// Connects to a socket being served, waiting for it to listen.
fn connect(socket: &Path) -> Client {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match Client::connect(socket) {
            Ok(client) => return client,
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!(
                "the program server never listened on {}: {error}",
                socket.display()
            ),
        }
    }
}

/// Two packages declaring disjoint descriptors each get their own keys, and a
/// key neither declares is named as uncovered: the answer a client reads over
/// the socket after installing both.
#[test]
fn each_key_goes_to_the_package_that_declares_it_over_the_socket() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    let server = Arc::new(ProgramServer::open(&configuration).unwrap());
    std::thread::spawn(move || {
        if let Err(error) = server.serve() {
            eprintln!("the server stopped: {error}");
        }
    });
    let mut client = connect(&configuration.socket);

    let f32 = temporary.path().join("f32");
    write_package(&f32, &manifest("f32", &["f32"], "sm_100a"));
    let fa4 = temporary.path().join("fa4");
    write_package(&fa4, &manifest("fa4", &["bf16"], "sm_100a"));
    client.install(&f32).unwrap();
    client.install(&fa4).unwrap();

    let coverage = client.cover(&request(&[BF16, F32, F16, BF16])).unwrap();
    // `f32` sorts before `fa4` at the same revision, so it is listed first.
    assert_eq!(
        coverage,
        Coverage {
            covered: vec![covered("f32", 1, &[F32]), covered("fa4", 1, &[BF16])],
            uncovered: vec![F16.to_owned()],
        }
    );
}

/// A key two packages declare goes to the highest revision, and among equal
/// revisions to the backend whose name sorts first.
#[test]
fn a_key_several_packages_cover_goes_to_the_preferred_one() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    installed(
        &configuration,
        "beta",
        1,
        &manifest("beta", &["bf16", "f16"], "sm_100a"),
    );
    installed(
        &configuration,
        "alpha",
        1,
        &manifest("alpha", &["bf16"], "sm_100a"),
    );
    installed(
        &configuration,
        "gamma",
        2,
        &manifest("gamma", &["f16"], "sm_100a"),
    );
    let server = ProgramServer::open(&configuration).unwrap();

    let coverage = server.cover(&request(&[BF16, F16])).unwrap();
    assert_eq!(
        coverage,
        Coverage {
            covered: vec![covered("gamma", 2, &[F16]), covered("alpha", 1, &[BF16])],
            uncovered: Vec::new(),
        }
    );
}

/// A package whose module of a key is already committed keeps that key
/// before a preferred package that would have to build it.
#[test]
fn a_key_stays_with_the_package_that_built_it() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    installed(
        &configuration,
        "alpha",
        1,
        &manifest("alpha", &["bf16"], "sm_100a"),
    );
    installed(
        &configuration,
        "beta",
        1,
        &manifest("beta", &["bf16"], "sm_100a"),
    );
    committed(&configuration, "beta", 1, BF16);
    let server = ProgramServer::open(&configuration).unwrap();

    let coverage = server.cover(&request(&[BF16])).unwrap();
    assert_eq!(coverage.covered, vec![covered("beta", 1, &[BF16])]);
}

/// Only a backend's latest revision answers, as `prepare` selects it: a key
/// an earlier revision declared and the latest does not is uncovered.
#[test]
fn only_the_latest_revision_of_a_backend_covers() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    installed(
        &configuration,
        "fa4",
        1,
        &manifest("fa4", &["bf16", "f16"], "sm_100a"),
    );
    installed(
        &configuration,
        "fa4",
        2,
        &manifest("fa4", &["bf16"], "sm_100a"),
    );
    let server = ProgramServer::open(&configuration).unwrap();

    let coverage = server.cover(&request(&[BF16, F16])).unwrap();
    assert_eq!(
        coverage,
        Coverage {
            covered: vec![covered("fa4", 2, &[BF16])],
            uncovered: vec![F16.to_owned()],
        }
    );
}

/// A package that does not build for the requested target covers nothing
/// there, whatever its descriptors declare.
#[test]
fn a_package_for_another_target_covers_nothing() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    installed(
        &configuration,
        "fa4",
        1,
        &manifest("fa4", &["bf16"], "sm_90a"),
    );
    let server = ProgramServer::open(&configuration).unwrap();

    let coverage = server.cover(&request(&[BF16])).unwrap();
    assert!(coverage.covered.is_empty());
    assert_eq!(coverage.uncovered, vec![BF16.to_owned()]);
}
