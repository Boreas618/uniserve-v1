//! `cover` (design section 6.3): which installed package is selected for each
//! of an operator's descriptor keys. Every fixture package declares its own
//! descriptors in `[kernel.descriptors]`, so the answer is read from the
//! manifests alone and nothing is compiled.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use build_server::builder::Limits;
use build_server::inputs::Inputs;
use build_server::protocol::{Client, CoverRequest, Covered, Named, Package};
use build_server::server::{BuildServer, ServerConfig};
use build_server::store::Root;
use build_server::target::{Device, Target};
use build_server::toolchain::ToolchainConfig;
use proto::module::PhaseKind;

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
interface_version = 1

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

/// `manifest` with the package's `priority` stated.
fn prioritized(manifest: &str, priority: u32) -> String {
    manifest.replacen(
        "\n[toolchain]",
        &format!("priority = {priority}\n\n[toolchain]"),
        1,
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
    Inputs::of(&directory, "triton 3.7.1".to_owned())
        .unwrap()
        .write(&directory)
        .unwrap();
}

/// A server configuration over a fresh package root and module store, for a
/// host of one `sm_100a` device.
fn config(temporary: &Path) -> ServerConfig {
    ServerConfig {
        socket: temporary.join("build-server.sock"),
        package_root: temporary.join("packages"),
        store_roots: vec![Root {
            path: temporary.join("store"),
            writable: true,
        }],
        upstream: None,
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

/// A `cover` of `keys` for `sm_100a`.
fn request(keys: &[&str]) -> CoverRequest {
    CoverRequest {
        op: OP.to_owned(),
        targets: vec![Target::new("sm_100a")],
        keys: keys.iter().map(|key| (*key).to_owned()).collect(),
        named: Vec::new(),
        fixed: Vec::new(),
    }
}

/// `backend` named for `phase`, at its latest revision.
fn named(phase: Option<PhaseKind>, backend: &str) -> Named {
    Named {
        phase,
        backend: backend.to_owned(),
        revision: None,
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
                "the build server never listened on {}: {error}",
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
    let server = Arc::new(BuildServer::open(&configuration).unwrap());
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
    // `f32` sorts before `fa4` at the same priority, so it is listed first.
    assert_eq!(
        coverage.covered,
        vec![covered("f32", 1, &[F32]), covered("fa4", 1, &[BF16])]
    );
    assert_eq!(coverage.uncovered, vec![F16.to_owned()]);
}

/// A key several packages declare goes to the one of highest priority, and
/// among equal priorities to the backend whose name sorts first: a later
/// revision of another backend does not outrank them.
#[test]
fn a_key_several_packages_cover_goes_to_the_highest_priority() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    installed(
        &configuration,
        "beta",
        1,
        &prioritized(&manifest("beta", &["bf16", "f16"], "sm_100a"), 2),
    );
    installed(
        &configuration,
        "alpha",
        1,
        &manifest("alpha", &["bf16", "f32"], "sm_100a"),
    );
    installed(
        &configuration,
        "gamma",
        3,
        &manifest("gamma", &["f32"], "sm_100a"),
    );
    let server = BuildServer::open(&configuration).unwrap();

    let coverage = server.cover(&request(&[BF16, F16, F32])).unwrap();
    assert_eq!(
        coverage.covered,
        vec![
            covered("beta", 1, &[BF16, F16]),
            covered("alpha", 1, &[F32]),
        ]
    );
    assert!(coverage.uncovered.is_empty());
}

/// A named backend takes every key it covers before any priority, and a key it
/// does not cover goes on to the packages that do.
#[test]
fn a_named_backend_takes_the_keys_it_covers() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    installed(
        &configuration,
        "alpha",
        1,
        &prioritized(&manifest("alpha", &["bf16", "f32"], "sm_100a"), 5),
    );
    installed(
        &configuration,
        "beta",
        1,
        &manifest("beta", &["bf16"], "sm_100a"),
    );
    let server = BuildServer::open(&configuration).unwrap();

    let mut asked = request(&[BF16, F32]);
    asked.named = vec![named(None, "beta")];
    let coverage = server.cover(&asked).unwrap();
    assert_eq!(
        coverage.covered,
        vec![covered("alpha", 1, &[F32]), covered("beta", 1, &[BF16])]
    );
}

/// A backend named for one phase is preferred for that phase's keys alone.
#[test]
fn a_backend_named_for_a_phase_takes_that_phase() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    let phased = |backend: &str| {
        manifest(backend, &["bf16"], "sm_100a").replace(
            "descriptors = { dimensions = { dtype",
            "descriptors = { dimensions = { phase = [\"prefill\", \"decode\"], dtype",
        )
    };
    installed(&configuration, "alpha", 1, &phased("alpha"));
    installed(&configuration, "beta", 1, &phased("beta"));
    let server = BuildServer::open(&configuration).unwrap();

    let prefill = "phase_prefill_dtype_bf16_head_dim_128";
    let decode = "phase_decode_dtype_bf16_head_dim_128";
    let mut asked = request(&[prefill, decode]);
    asked.named = vec![named(Some(PhaseKind::Decode), "beta")];
    let coverage = server.cover(&asked).unwrap();
    assert_eq!(
        coverage.covered,
        vec![
            covered("alpha", 1, &[prefill]),
            covered("beta", 1, &[decode])
        ]
    );
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
    let server = BuildServer::open(&configuration).unwrap();

    let coverage = server.cover(&request(&[BF16, F16])).unwrap();
    assert_eq!(coverage.covered, vec![covered("fa4", 2, &[BF16])]);
    assert_eq!(coverage.uncovered, vec![F16.to_owned()]);
}

/// The answer lists every candidate in selection order, which is what a module set fixes; and a
/// request that states a module set's revisions selects among exactly those: a revision or a
/// package installed since is no candidate, and a fixed revision the server lacks is refused.
#[test]
fn a_module_sets_revisions_are_the_only_candidates() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    installed(
        &configuration,
        "fa4",
        1,
        &manifest("fa4", &["bf16"], "sm_100a"),
    );
    let server = BuildServer::open(&configuration).unwrap();
    let package = |backend: &str, revision: u32| build_server::protocol::Package {
        op: OP.to_owned(),
        backend: backend.to_owned(),
        revision,
    };

    // What the daemon asks when a configuration is applied: no key, every candidate.
    let applied = server.cover(&request(&[])).unwrap();
    assert_eq!(applied.candidates, vec![package("fa4", 1)]);

    // Installed after the set was fixed: a later revision of `fa4`, and another package.
    installed(
        &configuration,
        "fa4",
        2,
        &manifest("fa4", &["bf16", "f16"], "sm_100a"),
    );
    installed(
        &configuration,
        "f32",
        1,
        &manifest("f32", &["f32"], "sm_100a"),
    );
    let server = BuildServer::open(&configuration).unwrap();
    let latest = server.cover(&request(&[BF16, F16, F32])).unwrap();
    assert_eq!(
        latest.candidates,
        vec![package("f32", 1), package("fa4", 2)]
    );
    assert!(latest.uncovered.is_empty());

    let mut fixed = request(&[BF16, F16, F32]);
    fixed.fixed = vec![("fa4".to_owned(), 1)];
    let coverage = server.cover(&fixed).unwrap();
    assert_eq!(coverage.candidates, vec![package("fa4", 1)]);
    assert_eq!(coverage.covered, vec![covered("fa4", 1, &[BF16])]);
    assert_eq!(coverage.uncovered, vec![F16.to_owned(), F32.to_owned()]);

    fixed.fixed = vec![("fa4".to_owned(), 3)];
    let refused = server.cover(&fixed).unwrap_err();
    assert!(refused.to_string().contains("revision 3"), "{refused}");
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
    let server = BuildServer::open(&configuration).unwrap();

    let coverage = server.cover(&request(&[BF16])).unwrap();
    assert!(coverage.covered.is_empty());
    assert_eq!(coverage.uncovered, vec![BF16.to_owned()]);
}
