//! The end-to-end witness of the Program Server (design sections 5.2, 5.3 and
//! 6.3): a real socket, a real package installed into the package root, its
//! module compiled by the real toolchain, committed into the store, and
//! resolved with the module's files arriving as descriptors.
//!
//! The build compiles `crates/ops/kernels/rms_norm/triton` with
//! `python/.venv`'s Triton, which is compiling and needs no GPU. The
//! environment is provisioned, not part of the repository:
//!
//! ```text
//! uv sync --project crates/program-server/python --python /usr/bin/python3.12
//! ```
//!
//! Without it the test reports the provisioning it needs and does not run;
//! with it, it runs the whole path a deployment takes.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use program_server::builder::{BuildState, Limits};
use program_server::manifest::ModuleForm;
use program_server::protocol::{Descriptors, PrepareRequest, ProgramRequest, Selector};
use program_server::server::{ProgramServer, ServerConfig};
use program_server::store::{ModulePath, Root};
use program_server::target::{Device, Target};
use program_server::toolchain::ToolchainConfig;
use program_server::{Client, ProgramSlot};

/// The repository root, three directories above this crate.
fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the repository root above the crate")
}

/// The interpreter of the Program Server's own Python environment.
fn interpreter() -> PathBuf {
    repository().join("crates/program-server/python/.venv/bin/python")
}

/// The `rms_norm` package of the tree: a Triton kernel program whose one
/// descriptor this test builds.
fn package() -> PathBuf {
    repository().join("crates/ops/kernels/rms_norm/triton")
}

/// The descriptor built here: `bf16`, 2048 columns, contiguous, which the
/// package's `[kernel.descriptors]` covers.
fn descriptor() -> String {
    "dtype_bf16_width_2048_layout_contiguous".to_owned()
}

/// Starts a server on a socket and answers it with the stop flag.
fn start(
    configuration: &ServerConfig,
) -> (
    Arc<ProgramServer>,
    Arc<AtomicBool>,
    std::thread::JoinHandle<()>,
) {
    let server = Arc::new(ProgramServer::open(configuration).expect("the server opens"));
    let stop = Arc::new(AtomicBool::new(false));
    let running = Arc::clone(&server);
    let stopping = Arc::clone(&stop);
    let thread = std::thread::spawn(move || {
        if let Err(error) = running.serve_until(stopping) {
            panic!("the server stopped with an error: {error}");
        }
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    while !configuration.socket.exists() {
        assert!(
            Instant::now() < deadline,
            "the server did not bind {} in 30 seconds",
            configuration.socket.display()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    (server, stop, thread)
}

#[test]
fn install_prepare_committed_resolve() {
    let interpreter = interpreter();
    if !interpreter.is_file() {
        eprintln!(
            "SKIPPED (provisioning): no Python environment at {}.\n\
             Create it with `uv sync --project crates/program-server/python \
             --python /usr/bin/python3.12`, then run this test again.",
            interpreter.display()
        );
        return;
    }

    let temporary = tempfile::tempdir().unwrap();
    let package_root = temporary.path().join("packages");
    std::fs::create_dir_all(&package_root).unwrap();
    let store = temporary.path().join("store");
    std::fs::create_dir_all(&store).unwrap();
    let configuration = ServerConfig {
        socket: temporary.path().join("program-server.sock"),
        package_root: package_root.clone(),
        store_roots: vec![Root {
            path: store.clone(),
            writable: true,
        }],
        upstream: None,
        limits: Limits::default(),
        toolchains: ToolchainConfig {
            nvcc: None,
            python: interpreter,
            wasm: None,
            max_jobs: 2,
        },
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    };
    let (_server, stop, thread) = start(&configuration);
    let mut client = Client::connect(&configuration.socket).unwrap();

    // 1. Install: the package is copied into the package root as revision 1.
    let installed = client.install(&package()).unwrap();
    assert_eq!(installed.package.op, "rms_norm");
    assert_eq!(installed.package.backend, "triton");
    assert_eq!(installed.package.revision, 1);
    assert!(
        package_root
            .join("rms_norm/triton/1/package.toml")
            .is_file(),
        "the installed package is under the package root"
    );

    // 2. Prepare: the descriptor's module is built and committed.
    let key = descriptor();
    let target = Target::new("sm_100a");
    let request = PrepareRequest {
        targets: vec![target.clone()],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("rms_norm"),
            selector: Selector::Latest {
                backend: "triton".into(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            op: "rms_norm".into(),
            keys: vec![key.clone()],
        }],
    };
    let mut progress = Vec::new();
    let prepared = client
        .prepare(&request, &mut |step| progress.push(step))
        .unwrap();
    assert_eq!(prepared.programs.len(), 1);
    assert_eq!(prepared.programs[0].program.revision, 1);
    // The package's `[parameters]` fill the program's arguments.
    assert_eq!(
        prepared.programs[0]
            .program
            .arguments
            .get("eps")
            .and_then(toml::Value::as_float),
        Some(1e-06)
    );
    assert_eq!(progress.len(), 1, "one descriptor is one build");
    let path = ModulePath::new("rms_norm", "triton", 1, target.as_str(), &key).unwrap();
    assert_eq!(progress[0].path, path);
    assert_eq!(
        progress[0].state,
        BuildState::Committed,
        "the build failed: {}",
        progress[0].detail
    );

    // 3. Committed: the module is in the store, and the store says so.
    let module = store
        .join("rms_norm")
        .join("triton")
        .join("1")
        .join(target.as_str())
        .join(&key);
    for name in ["module.cubin", "metadata.json", ".committed"] {
        assert!(
            module.join(name).is_file(),
            "the committed module has no {name}"
        );
    }

    // 4. Resolve: the files come back as descriptors, and they are the
    // module the store holds.
    let program = prepared.programs[0].program.clone();
    let resolved = client
        .resolve("rms_norm", &program, &target, &key, &mut |_| {})
        .unwrap();
    assert_eq!(resolved.form, ModuleForm::Cubin);
    assert_eq!(resolved.metadata.as_deref(), Some("metadata.json"));
    assert!(
        resolved.entries.is_empty(),
        "a cubin is launched, not called"
    );

    let cubin = resolved
        .files
        .iter()
        .find(|file| file.name == "module.cubin")
        .expect("the module's cubin arrives on a descriptor");
    let mut bytes = Vec::new();
    let mut file = cubin.file.try_clone().unwrap();
    file.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes.len() as u64, cubin.size);
    assert_eq!(
        &bytes[..4],
        b"\x7fELF",
        "a cubin is an ELF file the CUDA driver loads"
    );
    assert_eq!(
        bytes,
        std::fs::read(module.join("module.cubin")).unwrap(),
        "the resolved cubin is the committed one"
    );

    let metadata = resolved
        .files
        .iter()
        .find(|file| file.name == "metadata.json")
        .expect("the module's metadata arrives on a descriptor");
    let mut text = String::new();
    metadata
        .file
        .try_clone()
        .unwrap()
        .read_to_string(&mut text)
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("the metadata is JSON");
    assert!(
        !parsed
            .get("name")
            .and_then(|name| name.as_str())
            .unwrap_or_default()
            .is_empty(),
        "the metadata names the kernel it launches"
    );

    // 5. List: the package and its committed module are visible.
    let listing = client.list().unwrap();
    let listed = listing
        .packages
        .iter()
        .find(|package| package.op == "rms_norm" && package.backend == "triton")
        .expect("the installed package is listed");
    assert_eq!(listed.revision, 1);
    assert!(listed.modules.iter().any(|module| {
        module.target == target && module.descriptor == key && module.form == ModuleForm::Cubin
    }));

    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
}

/// A vendor cubin package: the package carries the cubin and the launch
/// metadata its vendor produced, and nothing compiles it (design section 5.3).
const CUBIN_MANIFEST: &str = r#"
op = "paged_attention"
backend = "trtllm"

[toolchain]
kind = "none"
version = ""

[module]
form = "cubin"
metadata = "metadata_{descriptor}.json"

[source]
kind = "cubin"
files = ["artifact/module_{descriptor}.cubin", "artifact/metadata_{descriptor}.json"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"], page = ["64"] } }
"#;

/// The bytes of a cubin, as a package carries them: an ELF object with the
/// little-endian CUDA magic. The Program Server stores it as it is; no driver
/// reads it here.
fn cubin_bytes() -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
    bytes.extend_from_slice(b"\x33\x22\x11\x00");
    bytes
}

#[test]
fn a_vendor_cubin_is_installed_stored_and_resolved_as_it_is() {
    let temporary = tempfile::tempdir().unwrap();
    let package = temporary.path().join("paged_attention-trtllm");
    std::fs::create_dir_all(package.join("artifact")).unwrap();
    std::fs::write(package.join("package.toml"), CUBIN_MANIFEST).unwrap();
    let key = "dtype_bf16_page_64";
    let cubin = cubin_bytes();
    std::fs::write(package.join(format!("artifact/module_{key}.cubin")), &cubin).unwrap();
    std::fs::write(
        package.join(format!("artifact/metadata_{key}.json")),
        br#"{"name": "uniserve_paged_attention_trtllm"}"#,
    )
    .unwrap();

    let package_root = temporary.path().join("packages");
    std::fs::create_dir_all(&package_root).unwrap();
    let store = temporary.path().join("store");
    std::fs::create_dir_all(&store).unwrap();
    let configuration = ServerConfig {
        socket: temporary.path().join("program-server.sock"),
        package_root: package_root.clone(),
        store_roots: vec![Root {
            path: store.clone(),
            writable: true,
        }],
        upstream: None,
        limits: Limits::default(),
        // A vendor cubin compiles nothing, so this server needs no compiler
        // at all: neither the Python environment nor nvcc exists here.
        toolchains: ToolchainConfig {
            nvcc: None,
            python: temporary.path().join("no-python"),
            wasm: None,
            max_jobs: 1,
        },
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    };
    let (_server, stop, thread) = start(&configuration);
    let mut client = Client::connect(&configuration.socket).unwrap();

    let installed = client.install(&package).unwrap();
    assert_eq!(installed.package.revision, 1);

    let target = Target::new("sm_100a");
    let request = PrepareRequest {
        targets: vec![target.clone()],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("paged_attention"),
            selector: Selector::Latest {
                backend: "trtllm".into(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            op: "paged_attention".into(),
            keys: vec![key.to_owned()],
        }],
    };
    let mut progress = Vec::new();
    let prepared = client
        .prepare(&request, &mut |step| progress.push(step))
        .unwrap();
    assert_eq!(prepared.programs.len(), 1);
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0].state, BuildState::Committed);

    let module = store
        .join("paged_attention/trtllm/1")
        .join(target.as_str())
        .join(key);
    assert_eq!(
        std::fs::read(module.join("module.cubin")).unwrap(),
        cubin,
        "the module is the package's cubin, byte for byte"
    );

    let program = prepared.programs[0].program.clone();
    let resolved = client
        .resolve("paged_attention", &program, &target, key, &mut |_| {})
        .unwrap();
    assert_eq!(resolved.form, ModuleForm::Cubin);
    assert_eq!(
        resolved.metadata.as_deref(),
        Some(format!("metadata_{key}.json").as_str())
    );
    let mut bytes = Vec::new();
    resolved
        .files
        .iter()
        .find(|file| file.name == "module.cubin")
        .expect("the cubin arrives on a descriptor")
        .file
        .try_clone()
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes, cubin);

    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
}
