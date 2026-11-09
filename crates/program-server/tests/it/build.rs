//! The Program Server's witnesses that need no repository directory: a vendor cubin and an executable
//! program are installed, stored and resolved as they are, a scheduling policy is built when it is
//! installed and served afterwards without a compiler, and the command makes every path absolute.
//!
//! The policy witness runs a Python build script and the wasm precompiler, which a host provisions:
//! `UNISERVE_PROGRAM_SERVER_PYTHON` names the interpreter and `UNISERVE_SCHEDULING_PRECOMPILE` the
//! precompiler, and the witness says what is missing when either is not named. The builds of the
//! repository's own packages are `cargo run -p program-server --example check_builds`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use program_server::builder::{BuildState, Limits};
use program_server::manifest::ModuleForm;
use program_server::protocol::{Descriptors, Module, PrepareRequest, ProgramRequest, Selector};
use program_server::server::{ProgramServer, ServerConfig};
use program_server::store::Root;
use program_server::target::{Device, Target};
use program_server::toolchain::ToolchainConfig;
use program_server::{Client, ProgramSlot};

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
            python: Some(temporary.path().join("no-python")),
            wasm: None,
            cargo: None,
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

/// An executable program package: a program the main process runs as its own
/// process, stored as it is (design section 5.1).
const EXECUTABLE_MANIFEST: &str = r#"
op = "sampler"
backend = "process"

[toolchain]
kind = "none"
version = ""

[module]
form = "executable"

[source]
kind = "executable"
files = ["bin/sampler"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] } }
"#;

#[test]
fn an_executable_program_is_installed_stored_and_resolved_as_it_is() {
    let temporary = tempfile::tempdir().unwrap();
    let package = temporary.path().join("sampler-process");
    std::fs::create_dir_all(package.join("bin")).unwrap();
    std::fs::write(package.join("package.toml"), EXECUTABLE_MANIFEST).unwrap();
    // A program is what its bytes say it is; this one only has to survive the
    // round trip, so the test does not run it.
    let program = b"#!/bin/sh\nexit 0\n".to_vec();
    std::fs::write(package.join("bin/sampler"), &program).unwrap();

    let witness = witness(
        &package,
        "sampler",
        "process",
        "dtype_bf16",
        ToolchainConfig {
            nvcc: None,
            python: Some(temporary.path().join("no-python")),
            wasm: None,
            cargo: None,
            max_jobs: 1,
        },
    );
    assert_eq!(witness.progress[0].state, BuildState::Committed);
    assert_eq!(witness.resolved.form, ModuleForm::Executable);
    assert!(witness.resolved.metadata.is_none());
    assert!(witness.resolved.entries.is_empty());
    assert_eq!(
        file_of(&witness.resolved, "bin/sampler"),
        program,
        "the module is the package's program, byte for byte"
    );

    witness.stop.store(true, Ordering::SeqCst);
    witness.thread.join().unwrap();
}

/// A scheduling policy is built when it is installed, and what the store keeps is
/// its module: a second server over the same package root — the restart a
/// deployment does — prepares and resolves it with no wasm toolchain at all.
///
/// A policy is the one kind whose build needs its repository's ancestry: its
/// crate inherits a workspace from the directories above it, so it can only be
/// built while the publisher's directory exists. Building it at install is what
/// removes that dependency from everything after (design section 6.2).
#[test]
fn a_policy_is_built_when_it_is_installed_and_needs_no_compiler_after() {
    let (Some(python), Some(precompiler)) = (
        std::env::var_os("UNISERVE_PROGRAM_SERVER_PYTHON").map(PathBuf::from),
        std::env::var_os("UNISERVE_SCHEDULING_PRECOMPILE").map(PathBuf::from),
    ) else {
        eprintln!(
            "SKIPPED (provisioning): name the interpreter with UNISERVE_PROGRAM_SERVER_PYTHON \
             and the precompiler with UNISERVE_SCHEDULING_PRECOMPILE"
        );
        return;
    };
    let temporary = tempfile::tempdir().unwrap();
    let package = temporary.path().join("policy");
    std::fs::create_dir_all(package.join("src")).unwrap();
    // The smallest module wasmtime accepts as a policy's input: the wasm header
    // and nothing else. What the policy computes is a scheduling question, not
    // this path's.
    std::fs::write(
        package.join("src/build.py"),
        "import os, pathlib\n\
         out = pathlib.Path(os.environ['UNISERVE_OUT'])\n\
         out.mkdir(parents=True, exist_ok=True)\n\
         (out / 'policy.wasm').write_bytes(b'\\x00asm\\x01\\x00\\x00\\x00')\n",
    )
    .unwrap();
    std::fs::write(
        package.join("package.toml"),
        r#"# A synthetic policy: the program layer's wasm path, nothing else.
op = "scheduling"
backend = "fixture"

[toolchain]
kind = "wasm"
version = ""

[module]
form = "wasm"

[source]
kind = "wasm"
files = ["src/build.py"]
script = "src/build.py"
"#,
    )
    .unwrap();

    let root = temporary.path().join("packages");
    std::fs::create_dir_all(&root).unwrap();
    let store = temporary.path().join("store");
    std::fs::create_dir_all(&store).unwrap();
    let with_precompiler = ServerConfig {
        socket: temporary.path().join("first.sock"),
        package_root: root.clone(),
        store_roots: vec![Root {
            path: store.clone(),
            writable: true,
        }],
        upstream: None,
        limits: Limits::default(),
        toolchains: ToolchainConfig {
            nvcc: None,
            python: Some(python.clone()),
            wasm: Some(precompiler),
            cargo: None,
            max_jobs: 1,
        },
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    };
    let (_server, stop, thread) = start(&with_precompiler);
    let mut client = Client::connect(&with_precompiler.socket).unwrap();
    client.install(&package).unwrap();
    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();

    // The artifact is beside the installed manifest, so a host that has no
    // precompiler — the deployment's serving host — can still prepare it.
    let revision = root.join("scheduling").join("fixture").join("1");
    assert!(
        revision.join("module.wasm").is_file(),
        "install kept no module beside the manifest at {}",
        revision.display()
    );

    let mut without_precompiler = with_precompiler.clone();
    without_precompiler.socket = temporary.path().join("second.sock");
    without_precompiler.toolchains.wasm = None;
    let (_server, stop, thread) = start(&without_precompiler);
    let mut client = Client::connect(&without_precompiler.socket).unwrap();
    let request = PrepareRequest {
        targets: Vec::new(),
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("scheduling"),
            selector: Selector::Latest {
                backend: "fixture".to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: Vec::new(),
    };
    let prepared = client
        .prepare(&request, &mut |_| {})
        .expect("a policy whose module the store holds needs no compiler");
    let target = Target::new(program_server::manifest::WASM_TARGET);
    let module = client
        .resolve(
            "scheduling",
            &prepared.programs[0].program,
            &target,
            program_server::manifest::POLICY_DESCRIPTOR,
            &mut |_| {},
        )
        .expect("the policy's module resolves");
    assert_eq!(module.form, ModuleForm::Wasm);
    assert!(
        !file_of(&module, program_server::source::MODULE_WASM).is_empty(),
        "the resolved module holds no wasm"
    );
    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
}

/// Every path the command takes is absolute before the server sees it.
///
/// A build runs with its working directory set to the store's copy of a package, so a
/// toolchain path typed relative to the shell's directory would name that copy instead
/// of this host's compiler — and fail as `No such file or directory` from a directory
/// the caller never named. The command is the place a person types those paths, so the
/// command is where they are made absolute.
#[test]
fn the_command_makes_every_path_absolute_before_the_server_sees_it() {
    use clap::Parser as _;
    use program_server::command::{Args, Command};

    let args = Args::try_parse_from([
        "program-server",
        "serve",
        "--socket",
        "relative/program-server.sock",
        "--package-root",
        "relative/packages",
        "--store-root",
        "relative/store",
        // Relative to the directory this test runs in, which is the crate's:
        // the command resolves a path where it was typed.
        "--python",
        "relative/python",
        "--wasm",
        "target/debug/scheduling-precompile",
    ])
    .expect("the command's own flags parse");
    let Command::Serve(serve) = args.command else {
        panic!("serve did not parse as serve");
    };
    let config = serve.settings.config(Vec::new());
    let python = config
        .toolchains
        .python
        .clone()
        .expect("an interpreter was named");
    for (name, path) in [
        ("socket", &config.socket),
        ("package root", &config.package_root),
        ("store root", &config.store_roots[0].path),
        ("python", &python),
        (
            "wasm",
            config
                .toolchains
                .wasm
                .as_ref()
                .expect("a wasm precompiler was named"),
        ),
    ] {
        assert!(path.is_absolute(), "the {name} stayed {}", path.display());
    }
    // The interpreter must resolve against the directory the command was typed in, not
    // against whatever directory a build later runs in. This is path arithmetic and
    // nothing else: it holds on a host whose Python environment was never created,
    // which is every host but the one that ran `uv sync`.
    let typed_in = std::env::current_dir().expect("the directory the test runs in");
    assert_eq!(
        python,
        typed_in.join("relative/python"),
        "the relative interpreter did not resolve against the directory it was typed in",
    );
}
