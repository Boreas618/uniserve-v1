//! The Build Server's witnesses that need no repository directory: a vendor cubin and an executable
//! package revision are installed, stored and resolved as they are, a scheduling policy is built when it is
//! installed and served afterwards without a compiler, the command makes every path absolute, and
//! `build` extends a store from key lists without reinstalling an unchanged package.
//!
//! The policy witness runs a Python build script and the wasm precompiler, which a host provisions:
//! `UNISERVE_BUILD_SERVER_PYTHON` names the interpreter and `UNISERVE_SCHEDULING_PRECOMPILE` the
//! precompiler, and the witness says what is missing when either is not named. The builds of the
//! repository's own packages are `cargo run -p build-server --example check_builds`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use build_server::Client;
use build_server::builder::{BuildState, Limits};
use build_server::inputs::Inputs;
use build_server::manifest::ModuleForm;
use build_server::protocol::{Descriptors, Module, PackageRequest, PrepareRequest, Selector};
use build_server::server::{BuildServer, ServerConfig};
use build_server::store::Root;
use build_server::target::{Device, Target};
use build_server::toolchain::ToolchainConfig;
use proto::module::ModuleSlot;

/// Starts a server on its socket and returns it with its stop flag.
fn start(
    configuration: &ServerConfig,
) -> (
    Arc<BuildServer>,
    Arc<AtomicBool>,
    std::thread::JoinHandle<()>,
) {
    let server = Arc::new(BuildServer::open(configuration).expect("the server opens"));
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
        socket: temporary.join("build-server.sock"),
        package_root,
        store_roots: vec![Root {
            path: store,
            writable: true,
        }],
        upstream: None,
        read_only_stores: Vec::new(),
        limits: Limits::default(),
        toolchains,
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    }
}

/// What one witness observed: the module `resolve` returned, the listing, and
/// the things the test shuts down at its end.
struct Witness {
    _temporary: tempfile::TempDir,
    stop: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<()>,
    progress: Vec<build_server::protocol::Progress>,
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
    assert_eq!(installed.revision, 1);
    assert_eq!(installed.op, op);
    assert_eq!(installed.backend, backend);

    let target = Target::new("sm_100a");
    let request = PrepareRequest {
        targets: vec![target.clone()],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all(op),
            selector: Selector::Latest {
                backend: backend.to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            slot: proto::module::ModuleSlot::all(op),
            keys: vec![key.to_owned()],
        }],
    };
    let mut progress = Vec::new();
    let prepared = client
        .prepare(&request, &mut |step| progress.push(step))
        .unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].1.revision, 1);

    let revision = prepared[0].1.clone();
    let resolved = client
        .resolve(op, &revision, &target, key, &mut |_| {})
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
interface_version = 1

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
/// CUDA magic. The Build Server stores it as it is; no driver reads it here.
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
    assert_eq!(installed.revision, 1);

    let target = Target::new("sm_100a");
    let request = PrepareRequest {
        targets: vec![target.clone()],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("paged_attention"),
            selector: Selector::Latest {
                backend: "trtllm".into(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            slot: proto::module::ModuleSlot::all("paged_attention"),
            keys: vec![key.to_owned()],
        }],
    };
    let mut progress = Vec::new();
    let prepared = client
        .prepare(&request, &mut |step| progress.push(step))
        .unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0].state, BuildState::Committed);

    let inputs = Inputs::read(&configuration.package_root.join("paged_attention/trtllm/1"))
        .unwrap()
        .expect("the installation recorded its build inputs")
        .digest();
    let module = store
        .join(format!("paged_attention/trtllm/1/{inputs}"))
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

    let revision = prepared[0].1.clone();
    let resolved = client
        .resolve("paged_attention", &revision, &target, key, &mut |_| {})
        .unwrap();
    assert_eq!(resolved.form, ModuleForm::Cubin);
    assert_eq!(
        resolved.metadata.as_deref(),
        Some(format!("metadata_{key}.json").as_str()),
        "the resolve response names the metadata the module holds"
    );
    assert_eq!(file_of(&resolved, "module.cubin"), cubin);

    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
}

/// An executable package: an executable run as its own process, stored as it
/// is. It implements version 3 of its interface.
const EXECUTABLE_MANIFEST: &str = r#"
op = "sampler"
backend = "process"
interface_version = 3

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
fn an_executable_revision_is_installed_stored_and_resolved_as_it_is() {
    let temporary = tempfile::tempdir().unwrap();
    let package = temporary.path().join("sampler-process");
    std::fs::create_dir_all(package.join("bin")).unwrap();
    std::fs::write(package.join("package.toml"), EXECUTABLE_MANIFEST).unwrap();
    // An executable is what its bytes say it is; this one only has to survive the
    // round trip, so the test does not run it.
    let executable = b"#!/bin/sh\nexit 0\n".to_vec();
    std::fs::write(package.join("bin/sampler"), &executable).unwrap();

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
        executable,
        "the module is the package's executable, byte for byte"
    );
    assert_eq!(
        witness.resolved.interface_version, 3,
        "the module carries the interface version its package's manifest states"
    );

    witness.stop.store(true, Ordering::SeqCst);
    witness.thread.join().unwrap();
}

/// A scheduling policy is built by the server from the snapshot its installation took: the
/// tree it was installed from is not read again, and a second server over the same roots, with
/// no toolchain at all, resolves the module the store keeps.
#[test]
#[ignore = "builds a policy crate for wasm32 with cargo and needs the wasm precompiler (UNISERVE_SCHEDULING_PRECOMPILE)"]
fn a_policy_is_built_from_its_installed_snapshot() {
    let precompiler = std::env::var_os("UNISERVE_SCHEDULING_PRECOMPILE")
        .map(PathBuf::from)
        .expect(
            "UNISERVE_SCHEDULING_PRECOMPILE must name the precompiler: \
             `cargo build -p scheduling --bin scheduling-precompile` builds it",
        );
    // The tree's default policy: a crate that depends on the guest crate through its workspace.
    let package =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../scheduling/policies/fcfs");
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("packages");
    std::fs::create_dir_all(&root).unwrap();
    let store = temporary.path().join("store");
    std::fs::create_dir_all(&store).unwrap();
    let with_toolchain = ServerConfig {
        socket: temporary.path().join("first.sock"),
        package_root: root.clone(),
        store_roots: vec![Root {
            path: store.clone(),
            writable: true,
        }],
        upstream: None,
        read_only_stores: Vec::new(),
        limits: Limits::default(),
        toolchains: ToolchainConfig {
            nvcc: None,
            python: None,
            wasm: Some(precompiler),
            cargo: None,
            max_jobs: 4,
        },
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    };
    let request = PrepareRequest {
        targets: Vec::new(),
        packages: vec![PackageRequest {
            slot: ModuleSlot::all("scheduling"),
            selector: Selector::Latest {
                backend: "fcfs".to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            slot: ModuleSlot::all("scheduling"),
            keys: vec![build_server::manifest::SINGLE_KEY.to_owned()],
        }],
    };

    let (_server, stop, thread) = start(&with_toolchain);
    let mut client = Client::connect(&with_toolchain.socket).unwrap();
    client.install(&package).unwrap();
    // The revision holds the policy's crate and the guest crate it depends on, and no module:
    // installing built nothing.
    let revision = root.join("scheduling").join("fcfs").join("1");
    assert!(
        revision
            .join("workspace/crates/scheduling/guest/Cargo.toml")
            .is_file()
    );
    assert!(!revision.join("module.wasm").exists());
    client
        .prepare(&request, &mut |_| {})
        .expect("the server builds the policy from its snapshot");
    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();

    let mut without_toolchain = with_toolchain.clone();
    without_toolchain.socket = temporary.path().join("second.sock");
    without_toolchain.toolchains.wasm = None;
    let (_server, stop, thread) = start(&without_toolchain);
    let mut client = Client::connect(&without_toolchain.socket).unwrap();
    let prepared = client
        .prepare(&request, &mut |_| {})
        .expect("a policy whose module the store holds needs no compiler");
    let target = Target::new(build_server::manifest::WASM_TARGET);
    let module = client
        .resolve(
            "scheduling",
            &prepared[0].1,
            &target,
            build_server::manifest::SINGLE_KEY,
            &mut |_| {},
        )
        .expect("the policy's module resolves");
    assert_eq!(module.form, ModuleForm::Wasm);
    assert!(
        !file_of(&module, build_server::source::MODULE_WASM).is_empty(),
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
    use build_server::command::{Args, Command};
    use clap::Parser as _;

    let args = Args::try_parse_from([
        "build-server",
        "serve",
        "--socket",
        "relative/build-server.sock",
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

/// The vendor cubin package of [`CUBIN_MANIFEST`] for two page sizes, in a
/// package tree as `build` reads one: `<tree>/<operator>/<backend>`.
fn cubin_tree(tree: &Path) -> PathBuf {
    let package = tree.join("paged_attention/trtllm");
    std::fs::create_dir_all(package.join("artifact")).unwrap();
    std::fs::write(
        package.join("package.toml"),
        CUBIN_MANIFEST.replace(r#"page = ["64"]"#, r#"page = ["64", "128"]"#),
    )
    .unwrap();
    for key in ["dtype_bf16_page_64", "dtype_bf16_page_128"] {
        std::fs::write(
            package.join(format!("artifact/module_{key}.cubin")),
            cubin_bytes(),
        )
        .unwrap();
        std::fs::write(
            package.join(format!("artifact/metadata_{key}.json")),
            br#"{"name": "uniserve_paged_attention_trtllm"}"#,
        )
        .unwrap();
    }
    package
}

/// Runs `uniserve build-server build` over `tree` into `store` for the
/// `paged_attention` keys named, from a key list written beside the store.
fn build_store(
    temporary: &Path,
    tree: &Path,
    store: &Path,
    keys: &[&str],
) -> Result<(), Box<build_server::Error>> {
    use clap::Parser as _;

    // A cubin package compiles nothing, so the interpreter is only a file the
    // command finds.
    let python = temporary.join("python");
    std::fs::write(&python, b"").unwrap();
    let list = temporary.join("keys.txt");
    let lines: Vec<String> = keys
        .iter()
        .map(|key| format!("paged_attention\t{key}\n"))
        .collect();
    std::fs::write(&list, lines.concat()).unwrap();

    let args = build_server::command::Args::try_parse_from([
        "build-server",
        "build",
        "--keys",
        list.to_str().unwrap(),
        "--packages",
        tree.to_str().unwrap(),
        "--target",
        "sm_100a",
        "--store",
        store.to_str().unwrap(),
        "--python",
        python.to_str().unwrap(),
    ])
    .expect("the build command parses");
    // `--target` names the architecture, so the host's devices are never read.
    build_server::command::run(args, &|| Ok(Vec::new())).map_err(Box::new)
}

/// The installed revisions of the cubin package in a built store.
fn revisions(store: &Path) -> Vec<String> {
    let installed = store.join("packages/paged_attention/trtllm");
    let mut revisions: Vec<String> = std::fs::read_dir(installed)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    revisions.sort();
    revisions
}

/// Whether a built store holds the committed module of `key` at `revision`, under the build
/// inputs that revision records.
fn committed(store: &Path, revision: u32, key: &str) -> bool {
    let installed = store.join(format!("packages/paged_attention/trtllm/{revision}"));
    let inputs = Inputs::read(&installed)
        .unwrap()
        .expect("an installed revision records its build inputs")
        .digest();
    store
        .join(format!(
            "modules/paged_attention/trtllm/{revision}/{inputs}/sm_100a"
        ))
        .join(key)
        .join(".committed")
        .is_file()
}

/// A store is extended by building more keys into it: a package whose files
/// are unchanged keeps its installed revision, so the modules built earlier
/// stay under the revision the store serves, and Python bytecode beside the
/// sources is not a change. A changed package is installed as a new revision.
#[test]
fn a_build_into_an_existing_store_keeps_the_revision_of_an_unchanged_package() {
    let temporary = tempfile::tempdir().unwrap();
    let tree = temporary.path().join("kernels");
    let package = cubin_tree(&tree);
    let store = temporary.path().join("store");

    build_store(temporary.path(), &tree, &store, &["dtype_bf16_page_64"]).unwrap();
    assert_eq!(revisions(&store), ["1"]);
    assert!(committed(&store, 1, "dtype_bf16_page_64"));

    // Bytecode a Python build or check left beside the sources.
    std::fs::create_dir_all(package.join("__pycache__")).unwrap();
    std::fs::write(package.join("__pycache__/build.cpython-312.pyc"), b"\0").unwrap();
    build_store(temporary.path(), &tree, &store, &["dtype_bf16_page_128"]).unwrap();
    assert_eq!(
        revisions(&store),
        ["1"],
        "an unchanged package was installed again"
    );
    assert!(committed(&store, 1, "dtype_bf16_page_64"));
    assert!(committed(&store, 1, "dtype_bf16_page_128"));

    std::fs::write(
        package.join("artifact/metadata_dtype_bf16_page_64.json"),
        br#"{"name": "uniserve_paged_attention_trtllm", "shared": 0}"#,
    )
    .unwrap();
    build_store(temporary.path(), &tree, &store, &["dtype_bf16_page_64"]).unwrap();
    assert_eq!(
        revisions(&store),
        ["1", "2"],
        "a changed package kept its revision"
    );
    assert!(committed(&store, 2, "dtype_bf16_page_64"));
}

/// A store's modules are found by what built them, not by a revision number alone: a package root
/// installed again from other files numbers its revision 1 again, and the store returns the
/// module of those files rather than the one the earlier revision 1 committed.
#[test]
fn a_revision_numbered_again_never_serves_the_module_of_other_files() {
    let temporary = tempfile::tempdir().unwrap();
    let tree = temporary.path().join("kernels");
    let package = cubin_tree(&tree);
    let store = temporary.path().join("store");
    let key = "dtype_bf16_page_64";
    build_store(temporary.path(), &tree, &store, &[key]).unwrap();

    std::fs::remove_dir_all(store.join("packages")).unwrap();
    let mut edited = cubin_bytes();
    edited.extend_from_slice(b"edited");
    std::fs::write(
        package.join(format!("artifact/module_{key}.cubin")),
        &edited,
    )
    .unwrap();
    build_store(temporary.path(), &tree, &store, &[key]).unwrap();
    assert_eq!(revisions(&store), ["1"]);

    let mut serving = configuration(
        temporary.path(),
        ToolchainConfig {
            nvcc: None,
            python: None,
            wasm: None,
            cargo: None,
            max_jobs: 1,
        },
    );
    serving.package_root = store.join("packages");
    serving.store_roots[0].path = store.join("modules");
    let server = BuildServer::open(&serving).unwrap();
    let revision = proto::module::PackageRevision {
        backend: "trtllm".into(),
        revision: 1,
        arguments: toml::Table::new(),
    };
    let module = server
        .resolve(
            "paged_attention",
            &revision,
            &Target::new("sm_100a"),
            key,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(
        file_of(&module, "module.cubin"),
        edited,
        "the store served the module of the earlier installation's files"
    );
}

/// A build fails with the keys it could not build, and still builds the rest.
#[test]
fn a_build_fails_with_the_keys_no_package_built() {
    let temporary = tempfile::tempdir().unwrap();
    let tree = temporary.path().join("kernels");
    cubin_tree(&tree);
    let store = temporary.path().join("store");

    let error = build_store(
        temporary.path(),
        &tree,
        &store,
        &["dtype_bf16_page_64", "dtype_bf16_page_32"],
    )
    .expect_err("a key no package covers fails the build");
    match *error {
        build_server::Error::Unbuilt { keys, total, .. } => {
            assert_eq!(keys, ["paged_attention dtype_bf16_page_32"]);
            assert_eq!(total, 2);
        }
        other => panic!("the build failed with {other}, not the keys it did not build"),
    }
    assert!(committed(&store, 1, "dtype_bf16_page_64"));
}
