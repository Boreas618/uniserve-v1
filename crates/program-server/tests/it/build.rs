//! The end-to-end witness of the Program Server (design sections 5.2, 5.3 and
//! 6.3): a real socket, a real package installed into the package root, its
//! module compiled by the real toolchain, committed into the store, and
//! resolved with the module's files arriving as descriptors.
//!
//! Each witness takes a package of the tree by the kind of source it declares,
//! so it follows the kernel programs as they are renamed and rewritten
//! against the model's operator names. Three need compilers on the host and
//! report what to provision when one is absent; the vendor cubin needs none.
//!
//! The Python environment is provisioned, not part of the repository, and
//! `uv sync --project crates/program-server/python --python /usr/bin/python3.12`
//! creates it (5.1 GB under `crates/program-server/python/.venv`).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use program_server::builder::{BuildState, Limits};
use program_server::manifest::ModuleForm;
use program_server::protocol::{Descriptors, Module, PrepareRequest, ProgramRequest, Selector};
use program_server::server::{ProgramServer, ServerConfig};
use program_server::store::{ModulePath, Root};
use program_server::target::{Device, Target};
use program_server::toolchain::ToolchainConfig;
use program_server::{Client, ProgramSlot};

/// The repository root, two directories above this crate.
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

/// A package of the tree and the pieces a request names it by: its directory,
/// its operator, its backend, and one descriptor its `[kernel.descriptors]`
/// covers — the first value of every dimension, in the order the package
/// writes them.
struct TreePackage {
    directory: PathBuf,
    op: String,
    backend: String,
    descriptor: String,
}

/// Finds the first package of `crates/ops/kernels` whose source is of `kind`,
/// in the order the tree names its operators.
fn tree_package(kind: &str) -> Option<TreePackage> {
    let root = repository().join("crates/ops/kernels");
    let mut operators = std::fs::read_dir(&root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    operators.sort();
    for operator in operators {
        let mut backends = std::fs::read_dir(&operator)
            .ok()?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        backends.sort();
        for backend in backends {
            let manifest = backend.join("package.toml");
            let Ok(text) = std::fs::read_to_string(&manifest) else {
                continue;
            };
            let Ok(parsed) = text.parse::<toml::Table>() else {
                continue;
            };
            let source_kind = parsed
                .get("source")
                .and_then(|source| source.get("kind"))
                .and_then(toml::Value::as_str);
            if source_kind != Some(kind) {
                continue;
            }
            let (Some(op), Some(backend_name)) = (
                parsed.get("op").and_then(toml::Value::as_str),
                parsed.get("backend").and_then(toml::Value::as_str),
            ) else {
                continue;
            };
            let Some(descriptor) = descriptor_key(&text) else {
                continue;
            };
            return Some(TreePackage {
                directory: backend,
                op: op.to_owned(),
                backend: backend_name.to_owned(),
                descriptor,
            });
        }
    }
    None
}

/// One descriptor a package covers, as the packages declare them: a list of
/// names under `keys`, or every dimension's first value joined by `_`.
fn descriptor_key(text: &str) -> Option<String> {
    // The section header on its own line: the packages' prose also names it.
    let kernel = text.split("\n[kernel.descriptors]\n").nth(1)?;
    if let Some(keys) = kernel
        .split("keys = [")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
    {
        let first = keys.split(',').next()?.trim().trim_matches('"');
        if !first.is_empty() {
            return Some(first.to_owned());
        }
    }
    let dimensions = kernel.split("dimensions = {").nth(1)?.split('}').next()?;
    // Each dimension is written `field = ["first", ...]`; the first value and
    // the field name, in the order the package writes them, are the key.
    let mut name = Vec::new();
    let mut rest = dimensions;
    while let Some(start) = rest.find("= [\"") {
        let field = rest[..start].trim_start_matches(',').trim();
        let values = &rest[start + 4..];
        let end = values.find('"')?;
        name.push(field.to_owned());
        name.push(values[..end].to_owned());
        // Skip the rest of this dimension's values, up to its `]`.
        let after = &values[end..];
        rest = &after[after.find(']')? + 1..];
    }
    (!name.is_empty()).then(|| name.join("_"))
}

/// Whether the host has a program, which a witness needs before it runs.
fn have(program: &str, arguments: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(arguments)
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Starts a server on its socket and answers it with its stop flag.
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

/// A server over a fresh package root and module store, with the compilers
/// `toolchains` names.
fn configuration(temporary: &Path, toolchains: ToolchainConfig) -> ServerConfig {
    let package_root = temporary.join("packages");
    std::fs::create_dir_all(&package_root).expect("the package root");
    let store = temporary.join("store");
    std::fs::create_dir_all(&store).expect("the store");
    ServerConfig {
        socket: temporary.join("program-server.sock"),
        package_root,
        store_roots: vec![Root {
            path: store,
            writable: true,
        }],
        upstream: None,
        limits: Limits::default(),
        toolchains,
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    }
}

/// What one witness observed: the module `resolve` answered, the listing, and
/// the things the test shuts down at its end.
struct Witness {
    _temporary: tempfile::TempDir,
    stop: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<()>,
    /// The program `prepare` selected: its revision, and the package's
    /// parameter defaults merged into its arguments.
    program: program_server::Program,
    progress: Vec<program_server::protocol::Progress>,
    resolved: Module,
}

/// Installs `package` over a fresh server, prepares one descriptor of it for
/// `sm_100a`, and resolves the committed module.
fn witness(
    package: &Path,
    op: &str,
    backend: &str,
    key: &str,
    toolchains: ToolchainConfig,
) -> Witness {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = configuration(temporary.path(), toolchains);
    let (_server, stop, thread) = start(&configuration);
    let mut client = Client::connect(&configuration.socket).unwrap();

    let installed = client.install(package).unwrap();
    assert_eq!(installed.package.revision, 1);
    assert_eq!(installed.package.op, op);
    assert_eq!(installed.package.backend, backend);

    let target = Target::new("sm_100a");
    let request = PrepareRequest {
        targets: vec![target.clone()],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all(op),
            selector: Selector::Latest {
                backend: backend.to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            op: op.to_owned(),
            keys: vec![key.to_owned()],
        }],
    };
    let mut progress = Vec::new();
    let prepared = client
        .prepare(&request, &mut |step| progress.push(step))
        .unwrap();
    assert_eq!(prepared.programs.len(), 1);
    assert_eq!(prepared.programs[0].program.revision, 1);

    let program = prepared.programs[0].program.clone();
    let resolved = client
        .resolve(op, &program, &target, key, &mut |_| {})
        .unwrap();
    Witness {
        _temporary: temporary,
        stop,
        thread,
        program,
        progress,
        resolved,
    }
}

/// Reads a file of a resolved module, by its name.
fn file_of(module: &Module, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    module
        .files
        .iter()
        .find(|file| file.name == name)
        .unwrap_or_else(|| {
            panic!(
                "the module has no {name}, only {:?}",
                module
                    .files
                    .iter()
                    .map(|file| file.name.as_str())
                    .collect::<Vec<_>>()
            )
        })
        .file
        .try_clone()
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

#[test]
fn a_triton_package_builds_a_cubin_module_and_resolves_it() {
    if !interpreter().is_file() {
        eprintln!(
            "SKIPPED (provisioning): no Python environment at {}.\n\
             Create it with `uv sync --project crates/program-server/python \
             --python /usr/bin/python3.12`, then run this test again.",
            interpreter().display()
        );
        return;
    }
    let package = tree_package("triton").expect("the tree has a Triton package");
    let witness = witness(
        &package.directory,
        &package.op,
        &package.backend,
        &package.descriptor,
        ToolchainConfig {
            nvcc: None,
            python: interpreter(),
            wasm: None,
            max_jobs: 2,
        },
    );

    // The package's `[parameters]` fill the selected program's arguments.
    let manifest = std::fs::read_to_string(package.directory.join("package.toml")).unwrap();
    let parameters = manifest
        .split("[parameters]")
        .nth(1)
        .map(|rest| {
            rest.split('\n')
                .take_while(|line| !line.starts_with('['))
                .collect::<String>()
        })
        .unwrap_or_default();
    let declared: toml::Table = parameters.parse().expect("the package's parameters");
    for (name, default) in &declared {
        assert_eq!(
            witness.program.arguments.get(name),
            Some(default),
            "the program carries the package's {name} default"
        );
    }

    // One descriptor is one build, and the build committed.
    assert_eq!(witness.progress.len(), 1, "one descriptor is one build");
    assert_eq!(
        witness.progress[0].state,
        BuildState::Committed,
        "the build failed: {}",
        witness.progress[0].detail
    );
    let path = ModulePath::new(
        &package.op,
        &package.backend,
        1,
        "sm_100a",
        &package.descriptor,
    )
    .unwrap();
    assert_eq!(witness.progress[0].path, path);

    // The module is a cubin and the metadata that says how to launch it.
    assert_eq!(witness.resolved.form, ModuleForm::Cubin);
    let metadata = witness
        .resolved
        .metadata
        .clone()
        .expect("a cubin module names its launch metadata");
    assert!(
        witness.resolved.entries.is_empty(),
        "a cubin has no C entry"
    );
    let cubin = file_of(&witness.resolved, "module.cubin");
    assert_eq!(
        &cubin[..4],
        b"\x7fELF",
        "a cubin is an ELF file the CUDA driver loads"
    );
    let text = String::from_utf8(file_of(&witness.resolved, &metadata)).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("the metadata is JSON");
    assert!(
        !parsed
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .is_empty(),
        "the metadata names the kernel it launches"
    );

    witness.stop.store(true, Ordering::SeqCst);
    witness.thread.join().unwrap();
}

#[test]
fn a_cuda_package_builds_a_native_module_with_its_entries() {
    if !have("nvcc", &["--version"]) {
        eprintln!("SKIPPED (provisioning): no nvcc on this host");
        return;
    }
    let Some(package) = tree_package("cuda") else {
        eprintln!("SKIPPED: the tree holds no CUDA package");
        return;
    };
    let witness = witness(
        &package.directory,
        &package.op,
        &package.backend,
        &package.descriptor,
        ToolchainConfig {
            nvcc: None,
            python: interpreter(),
            wasm: None,
            max_jobs: 2,
        },
    );

    assert_eq!(witness.progress[0].state, BuildState::Committed);
    assert_eq!(witness.resolved.form, ModuleForm::Native);
    assert!(
        witness.resolved.metadata.is_none(),
        "a native module has no launch metadata"
    );
    let package_manifest = std::fs::read_to_string(package.directory.join("package.toml")).unwrap();
    let launch = package_manifest
        .split("[module.entries]")
        .nth(1)
        .and_then(|entries| entries.split("launch = \"").nth(1))
        .and_then(|rest| rest.split('"').next())
        .expect("a native package names its launch entry");
    assert_eq!(
        witness.resolved.entries.get("launch").map(String::as_str),
        Some(launch),
        "the module's launch symbol is the package's"
    );
    let library = file_of(&witness.resolved, "module.so");
    assert_eq!(
        &library[..4],
        b"\x7fELF",
        "a native module is a shared object"
    );
    assert!(
        String::from_utf8_lossy(&library).contains(launch),
        "the module exports the entry its package names"
    );

    witness.stop.store(true, Ordering::SeqCst);
    witness.thread.join().unwrap();
}

#[test]
fn a_cute_dsl_package_builds_a_native_module_with_its_entries() {
    if !interpreter().is_file() {
        eprintln!(
            "SKIPPED (provisioning): no Python environment at {}",
            interpreter().display()
        );
        return;
    }
    let Some(package) = tree_package("cute_dsl") else {
        eprintln!("SKIPPED: the tree holds no CuTe DSL package");
        return;
    };
    let witness = witness(
        &package.directory,
        &package.op,
        &package.backend,
        &package.descriptor,
        ToolchainConfig {
            nvcc: None,
            python: interpreter(),
            wasm: None,
            max_jobs: 2,
        },
    );

    assert_eq!(witness.progress[0].state, BuildState::Committed);
    assert_eq!(witness.resolved.form, ModuleForm::Native);
    assert!(witness.resolved.metadata.is_none());
    let library = file_of(&witness.resolved, "module.so");
    assert_eq!(&library[..4], b"\x7fELF");

    witness.stop.store(true, Ordering::SeqCst);
    witness.thread.join().unwrap();
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
/// CUDA magic. The Program Server stores it as it is; no driver reads it here.
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

    let configuration = configuration(
        temporary.path(),
        ToolchainConfig {
            nvcc: None,
            python: temporary.path().join("no-python"),
            wasm: None,
            max_jobs: 1,
        },
    );
    let store = configuration.store_roots[0].path.clone();
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
    for name in [
        "module.cubin",
        "metadata_dtype_bf16_page_64.json",
        ".committed",
    ] {
        assert!(
            module.join(name).is_file(),
            "the committed module has no {name}"
        );
    }

    let program = prepared.programs[0].program.clone();
    let resolved = client
        .resolve("paged_attention", &program, &target, key, &mut |_| {})
        .unwrap();
    assert_eq!(resolved.form, ModuleForm::Cubin);
    assert_eq!(
        resolved.metadata.as_deref(),
        Some(format!("metadata_{key}.json").as_str()),
        "the resolve answer names the metadata the module holds"
    );
    assert_eq!(file_of(&resolved, "module.cubin"), cubin);

    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
}
