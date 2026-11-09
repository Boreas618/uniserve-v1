//! The Program Server's end-to-end builds of real packages, over the directories and toolchains the
//! caller names: a Triton, a CUDA and a CuTe DSL package of the kernel root are installed into a
//! fresh server, compiled by the real toolchain, committed and resolved; a package that declares no
//! toolchain version builds on any host and one that names another version is refused; a package
//! compiles from the copy its install kept; the policy fixture installs, is precompiled, and is
//! served by a second server with no precompiler.
//!
//!     cargo run -p program-server --example check_builds -- \
//!         --kernels <kernel package root> --fixtures <program fixture root> \
//!         --python <the Program Server's interpreter> --nvcc <nvcc> \
//!         --precompile <scheduling-precompile> --cargo <cargo> [--keys <key file>]
//!
//! `--keys` names a key file: a build of a source kind then takes the package of the first operator
//! the file lists. The exit status is 1 when any build fails.
#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "each check is an assertion run under catch_unwind, and a failed expectation is its report"
)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
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

struct Args {
    kernels: PathBuf,
    fixtures: PathBuf,
    python: PathBuf,
    nvcc: PathBuf,
    precompile: PathBuf,
    cargo: PathBuf,
    keys: Option<PathBuf>,
}

fn arguments() -> Result<Args, String> {
    let (mut kernels, mut fixtures, mut python, mut nvcc, mut precompile, mut cargo, mut keys) =
        (None, None, None, None, None, None, None);
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{flag} takes a value"))
        };
        match flag.as_str() {
            "--kernels" => kernels = Some(value()?),
            "--fixtures" => fixtures = Some(value()?),
            "--python" => python = Some(value()?),
            "--nvcc" => nvcc = Some(value()?),
            "--precompile" => precompile = Some(value()?),
            "--cargo" => cargo = Some(value()?),
            "--keys" => keys = Some(value()?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        kernels: kernels.ok_or("--kernels names the kernel package root")?,
        fixtures: fixtures.ok_or("--fixtures names the program fixture root")?,
        python: python.ok_or("--python names the Program Server's interpreter")?,
        nvcc: nvcc.ok_or("--nvcc names the CUDA compiler")?,
        precompile: precompile.ok_or("--precompile names scheduling-precompile")?,
        cargo: cargo.ok_or("--cargo names the cargo that builds the policy fixture")?,
        keys,
    })
}

fn main() -> ExitCode {
    let args = match arguments() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("check_builds: {message}");
            return ExitCode::from(2);
        }
    };
    #[expect(
        clippy::type_complexity,
        reason = "a check is its name and its function, listed once here"
    )]
    let checks: [(&str, fn(&Args)); 6] = [
        (
            "a Triton package builds a cubin module",
            a_triton_package_builds_a_cubin_module_and_resolves_it,
        ),
        (
            "a CUDA package builds its declared module",
            a_cuda_package_builds_the_module_its_manifest_declares,
        ),
        (
            "a CuTe DSL package builds a native module",
            a_cute_dsl_package_builds_a_native_module_with_its_entries,
        ),
        (
            "a package without a toolchain version builds",
            a_package_that_declares_no_toolchain_version_builds_on_any_host,
        ),
        (
            "a package compiles from its installed copy",
            a_package_compiles_from_the_copy_its_install_kept,
        ),
        (
            "the policy fixture serves without a compiler",
            the_policy_fixture_installs_and_serves_without_a_compiler,
        ),
    ];
    let mut failed = false;
    for (name, check) in checks {
        match std::panic::catch_unwind(|| check(&args)) {
            Ok(()) => println!("passed  {name}"),
            Err(_) => {
                failed = true;
                println!("FAILED  {name}");
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// The operators a key file lists, in the order it lists them.
fn key_operators(path: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut operators = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some((operator, _)) = line.split_once('\t')
            && !operators
                .iter()
                .any(|known: &String| known == operator.trim())
        {
            operators.push(operator.trim().to_owned());
        }
    }
    operators
}

/// The toolchains a CUDA package needs: the compiler and the interpreter the caller named.
fn toolchains(args: &Args) -> ToolchainConfig {
    ToolchainConfig {
        nvcc: Some(args.nvcc.clone()),
        python: Some(args.python.clone()),
        wasm: None,
        cargo: None,
        max_jobs: 1,
    }
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
    /// The compile script the package names, when it names one. A `cuda`
    /// package whose script is Python needs the Program Server's environment to
    /// run it, which is a different thing to provision than nvcc.
    script: Option<String>,
}

/// The package of the kernel root whose source is of `kind` and whose
/// operator the slice names first, so a witness builds the kernel a deployment
/// needs rather than whichever backend the tree lists first; when the slice's
/// fixture is absent, the first package of that kind.
fn tree_package(args: &Args, kind: &str) -> Option<TreePackage> {
    let slice = args.keys.as_deref().map(key_operators).unwrap_or_default();
    let packages = tree_packages(&args.kernels, kind);
    // A script that is not Python builds the module without the Program
    // Server's environment, so a witness of the source kind alone prefers it:
    // otherwise the witness silently becomes a test of that environment.
    let plain = |package: &TreePackage| {
        package
            .script
            .as_deref()
            .is_none_or(|script| !script.ends_with(".py"))
    };
    let from_slice = |wanted: bool| {
        slice.iter().find_map(|operator| {
            packages
                .iter()
                .find(|package| &package.op == operator && (!wanted || plain(package)))
                .map(repackage)
        })
    };
    from_slice(true)
        .or_else(|| {
            packages
                .iter()
                .find(|package| plain(package))
                .map(repackage)
        })
        .or_else(|| from_slice(false))
        .or_else(|| packages.first().map(repackage))
}

/// A package of `tree_packages`, by value, so the caller can own it.
fn repackage(package: &TreePackage) -> TreePackage {
    TreePackage {
        directory: package.directory.clone(),
        op: package.op.clone(),
        backend: package.backend.clone(),
        descriptor: package.descriptor.clone(),
        script: package.script.clone(),
    }
}

/// Every package of `root` whose source is of `kind`, in the
/// order the tree names its operators.
fn tree_packages(root: &Path, kind: &str) -> Vec<TreePackage> {
    let mut found = Vec::new();
    let mut operators = match std::fs::read_dir(root) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>(),
        Err(_) => return found,
    };
    operators.sort();
    for operator in operators {
        let mut backends = match std::fs::read_dir(&operator) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.is_dir())
                .collect::<Vec<_>>(),
            Err(_) => continue,
        };
        backends.sort();
        for backend in backends {
            let manifest = backend.join("package.toml");
            let Ok(text) = std::fs::read_to_string(&manifest) else {
                continue;
            };
            let Ok(parsed) = text.parse::<toml::Table>() else {
                continue;
            };
            let source = parsed.get("source");
            let source_kind = source
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
            let script = source
                .and_then(|source| source.get("script"))
                .and_then(toml::Value::as_str)
                .map(str::to_owned);
            found.push(TreePackage {
                directory: backend,
                op: op.to_owned(),
                backend: backend_name.to_owned(),
                descriptor,
                script,
            });
        }
    }
    found
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

/// A copy of a package in `temporary` that declares this host's version of
/// the toolchain, or `None` when the host has no such toolchain.
///
/// A checked-in package is written for one toolkit version, and a version
/// change is a new revision (design sections 5.1 and 5.2), so a package of the
/// tree can only build on a host whose compiler is that exact version. A
/// witness must not fail because a host's compiler patch differs: it copies the
/// package and declares the version the host's toolchain reports, which leaves
/// the form, the entries and the sources under test untouched.
fn at_host_version(
    source: &Path,
    temporary: &Path,
    toolchains: &ToolchainConfig,
) -> Option<PathBuf> {
    let manifest = program_server::manifest::Manifest::load(source).ok()?;
    let kind = manifest.toolchain.kind;
    let version = program_server::toolchain::Toolchains::discover(toolchains)
        .ok()?
        .version(kind)
        .ok()?
        .to_owned();
    let name = source
        .file_name()
        .map(|name| name.to_owned())
        .unwrap_or_else(|| "package".into());
    let destination = temporary.join(name);
    copy_tree(source, &destination);
    let text = std::fs::read_to_string(destination.join("package.toml")).ok()?;
    std::fs::write(
        destination.join("package.toml"),
        declare_toolchain_version(&text, &version),
    )
    .ok()?;
    Some(destination)
}

/// Every file and directory of `source`, under `destination`.
fn copy_tree(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination).expect("the copied package's directory");
    for entry in std::fs::read_dir(source).expect("the package's files") {
        let entry = entry.expect("a package file");
        let target = destination.join(entry.file_name());
        if entry.file_type().expect("a file type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("a copied package file");
        }
    }
}

/// `manifest` with the `[toolchain]` section's `version` set to `version`.
fn declare_toolchain_version(manifest: &str, version: &str) -> String {
    let mut lines = Vec::new();
    let mut in_toolchain = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_toolchain = trimmed == "[toolchain]";
        }
        if in_toolchain && trimmed.starts_with("version") && trimmed.contains('=') {
            lines.push(format!("version = \"{version}\""));
        } else {
            lines.push(line.to_owned());
        }
    }
    lines.join("\n") + "\n"
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

/// Installs and prepares one package, answering the failure's message when it
/// fails.
fn prepare_outcome(
    package: &Path,
    op: &str,
    backend: &str,
    key: &str,
    toolchains: ToolchainConfig,
) -> std::result::Result<(), String> {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = configuration(temporary.path(), toolchains);
    let (_server, stop, thread) = start(&configuration);
    let mut client = Client::connect(&configuration.socket).unwrap();
    let outcome = client
        .install(package)
        .map_err(|error| error.to_string())
        .and_then(|_| {
            let request = PrepareRequest {
                targets: vec![Target::new("sm_100a")],
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
            client
                .prepare(&request, &mut |_| {})
                .map(|_| ())
                .map_err(|error| error.to_string())
        });
    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
    outcome
}

fn a_triton_package_builds_a_cubin_module_and_resolves_it(args: &Args) {
    let python = args.python.clone();
    let Some(package) = tree_package(args, "triton") else {
        eprintln!("SKIPPED: the tree holds no Triton package");
        return;
    };
    let toolchains = ToolchainConfig {
        nvcc: Some(args.nvcc.clone()),
        python: Some(python.clone()),
        wasm: None,
        cargo: None,
        max_jobs: 2,
    };
    // The tree's package is written for one Triton version; this host's may be
    // another patch, which is a new revision and not a defect of the witness.
    let temporary = tempfile::tempdir().unwrap();
    let Some(directory) = at_host_version(&package.directory, temporary.path(), &toolchains) else {
        eprintln!("SKIPPED (provisioning): this host reports no Triton version to build with");
        return;
    };
    let witness = witness(
        &directory,
        &package.op,
        &package.backend,
        &package.descriptor,
        toolchains,
    );

    // The package's `[parameters]` fill the selected program's arguments.
    let manifest = std::fs::read_to_string(directory.join("package.toml")).unwrap();
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

fn a_cuda_package_builds_the_module_its_manifest_declares(args: &Args) {
    let Some(package) = tree_package(args, "cuda") else {
        eprintln!("SKIPPED: the tree holds no CUDA package");
        return;
    };
    let temporary = tempfile::tempdir().unwrap();
    let toolchains = ToolchainConfig {
        nvcc: Some(args.nvcc.clone()),
        python: Some(args.python.clone()),
        wasm: None,
        cargo: None,
        max_jobs: 2,
    };
    // A package of the tree declares one toolkit version, and a version change
    // is a new revision; the witness builds this host's toolkit instead.
    let Some(directory) = at_host_version(&package.directory, temporary.path(), &toolchains) else {
        eprintln!("SKIPPED (provisioning): this host reports no nvcc version to build with");
        return;
    };
    // What the package declares is what the module must be: nvcc writes a
    // cubin and its launch metadata, or a native library with the entries the
    // package names (design section 5.3).
    let manifest = program_server::manifest::Manifest::load(&directory).expect("the package loads");
    let witness = witness(
        &directory,
        &package.op,
        &package.backend,
        &package.descriptor,
        toolchains,
    );

    assert_eq!(witness.progress[0].state, BuildState::Committed);
    assert_eq!(witness.resolved.form, manifest.module.form);
    match manifest.module.form {
        ModuleForm::Cubin => {
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
            assert_eq!(&cubin[..4], b"\x7fELF");
            let text = String::from_utf8(file_of(&witness.resolved, &metadata)).unwrap();
            let parsed: serde_json::Value =
                serde_json::from_str(&text).expect("the metadata is JSON");
            assert!(
                !parsed
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .is_empty(),
                "the metadata names the kernel it launches"
            );
        }
        ModuleForm::Native => {
            assert!(witness.resolved.metadata.is_none());
            assert_eq!(
                witness.resolved.entries.get("launch"),
                manifest.module.entries.get("launch"),
                "the module's launch symbol is the package's"
            );
            let library = file_of(&witness.resolved, "module.so");
            assert_eq!(&library[..4], b"\x7fELF");
            let launch = manifest
                .module
                .entries
                .get("launch")
                .expect("a native package names its launch entry");
            assert!(
                String::from_utf8_lossy(&library).contains(launch),
                "the module exports the entry its package names"
            );
        }
        other => panic!("a CUDA package builds a cubin or a native module, not {other:?}"),
    }

    witness.stop.store(true, Ordering::SeqCst);
    witness.thread.join().unwrap();
}

fn a_cute_dsl_package_builds_a_native_module_with_its_entries(args: &Args) {
    let python = args.python.clone();
    let Some(package) = tree_package(args, "cute_dsl") else {
        eprintln!("SKIPPED: the tree holds no CuTe DSL package");
        return;
    };
    let toolchains = ToolchainConfig {
        nvcc: Some(args.nvcc.clone()),
        python: Some(python.clone()),
        wasm: None,
        cargo: None,
        max_jobs: 2,
    };
    let temporary = tempfile::tempdir().unwrap();
    let Some(directory) = at_host_version(&package.directory, temporary.path(), &toolchains) else {
        eprintln!("SKIPPED (provisioning): this host reports no CuTe DSL version to build with");
        return;
    };
    let witness = witness(
        &directory,
        &package.op,
        &package.backend,
        &package.descriptor,
        toolchains,
    );

    assert_eq!(witness.progress[0].state, BuildState::Committed);
    assert_eq!(witness.resolved.form, ModuleForm::Native);
    assert!(witness.resolved.metadata.is_none());
    let library = file_of(&witness.resolved, "module.so");
    assert_eq!(&library[..4], b"\x7fELF");

    witness.stop.store(true, Ordering::SeqCst);
    witness.thread.join().unwrap();
}

/// A package that declares no toolchain version is built by whatever this host
/// has; one that names a version this host does not have is refused before its
/// build starts.
///
/// The version is the one part of a package's declaration that is about the
/// host rather than about the program, and a fixture whose build does not
/// depend on the patch release declares it empty so that it builds on whichever
/// host the caller's test uses (design section 5.2). The same package with a
/// version no host has proves the comparison is still made.
fn a_package_that_declares_no_toolchain_version_builds_on_any_host(args: &Args) {
    let toolchains = toolchains(args);
    let source = args.kernels.join("moe_sort/trtllm");
    let manifest = std::fs::read_to_string(source.join("package.toml")).unwrap();
    let Some(key) = descriptor_key(&manifest) else {
        panic!("the moe_sort package declares no descriptor");
    };
    let temporary = tempfile::tempdir().unwrap();

    // The same sources twice: once declaring no version, once declaring one no
    // host has.
    let mut packages = Vec::new();
    for (name, version) in [("any-version", ""), ("another-version", "0.0.1")] {
        let directory = temporary.path().join(name);
        copy_tree(&source, &directory);
        std::fs::write(
            directory.join("package.toml"),
            declare_toolchain_version(&manifest, version),
        )
        .unwrap();
        packages.push(directory);
    }

    let built = prepare_outcome(&packages[0], "moe_sort", "trtllm", &key, toolchains.clone());
    assert!(
        built.is_ok(),
        "a package declaring no version was not built: {built:?}"
    );

    let refused = prepare_outcome(&packages[1], "moe_sort", "trtllm", &key, toolchains);
    let message = refused.expect_err("a package declaring another version was built");
    assert!(
        message.contains("is written for nvcc"),
        "the refusal does not name the version: {message}"
    );
}

/// A package compiles from the copy its install kept, not from the tree it was
/// installed from: the publisher's directory is deleted before the build, and the
/// module still commits.
///
/// This is the property a deployment depends on. The upstream Program Server
/// installs a package from a checkout it may later lose, and a downstream server's
/// package root is a read-only mount of the upstream's; neither has the tree, so
/// what compiles must be what was installed.
fn a_package_compiles_from_the_copy_its_install_kept(args: &Args) {
    let toolchains = toolchains(args);
    let source = args.kernels.join("moe_sort/trtllm");
    let Ok(manifest) = std::fs::read_to_string(source.join("package.toml")) else {
        eprintln!("SKIPPED: the tree holds no moe_sort package");
        return;
    };
    let Some(key) = descriptor_key(&manifest) else {
        panic!("the moe_sort package declares no descriptor");
    };

    let temporary = tempfile::tempdir().unwrap();
    // The publisher's directory: a copy of the tree package, declaring this
    // host's toolkit so the witness is not about the version.
    let publisher = temporary.path().join("publisher");
    std::fs::create_dir_all(&publisher).unwrap();
    let Some(to_build_from) = at_host_version(&source, &publisher, &toolchains) else {
        eprintln!("SKIPPED (provisioning): this host reports no nvcc version to build with");
        return;
    };

    let configuration = configuration(temporary.path(), toolchains);
    let (_server, stop, thread) = start(&configuration);
    let mut client = Client::connect(&configuration.socket).unwrap();
    client.install(&to_build_from).unwrap();

    // The tree the install named is gone; the installed copy is all there is.
    std::fs::remove_dir_all(&to_build_from).unwrap();
    assert!(!to_build_from.exists());

    let request = PrepareRequest {
        targets: vec![Target::new("sm_100a")],
        programs: vec![ProgramRequest {
            slot: ProgramSlot::all("moe_sort"),
            selector: Selector::Latest {
                backend: "trtllm".to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            op: "moe_sort".to_owned(),
            keys: vec![key],
        }],
    };
    let mut states = Vec::new();
    client
        .prepare(&request, &mut |progress| states.push(progress.state))
        .expect("the package compiles from the installed copy");
    assert!(
        states.contains(&BuildState::Committed),
        "the build of the installed copy committed nothing: {states:?}"
    );
    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
}

/// The policy fixture the delivery test installs: a real policy crate, built for
/// wasm32 when the package is installed, precompiled for this host's CPU, and then
/// served by a second server that has no precompiler and no tree.
///
/// This is the whole path the engine's program-layer test walks: `install` a
/// package whose `op`/`backend` are the deployment's policy, so the program's
/// revision moves; the module the daemon then resolves is the one the install
/// built. The package itself carries neither a wasm binary nor the policy's
/// source — the build asks cargo for sched's own crate — which is why the install
/// is the step that must have the tree.
fn the_policy_fixture_installs_and_serves_without_a_compiler(args: &Args) {
    let precompiler = args.precompile.clone();
    let package = args.fixtures.join("policy");
    if !package.join("package.toml").is_file() {
        eprintln!("SKIPPED: the tree holds no policy fixture");
        return;
    }

    let temporary = tempfile::tempdir().unwrap();
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
            python: Some(args.python.clone()),
            wasm: Some(precompiler),
            cargo: Some(args.cargo.clone()),
            max_jobs: 2,
        },
        devices: vec![Device {
            arch: Target::new("sm_100a"),
            model: Target::new("b200"),
        }],
    };
    let (_server, stop, thread) = start(&with_precompiler);
    let mut client = Client::connect(&with_precompiler.socket).unwrap();
    let installed = client
        .install(&package)
        .expect("the policy fixture installs");
    assert_eq!(installed.package.op, "scheduling");
    assert_eq!(installed.package.backend, "fcfs");
    assert_eq!(installed.package.revision, 1);
    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();

    // The precompiled module is beside the installed manifest, and it is wasmtime's
    // artifact rather than the raw module the policy crate produced.
    let revision = root.join("scheduling").join("fcfs").join("1");
    let artifact = revision.join("module.wasm");
    assert!(
        artifact.is_file(),
        "the install kept no module beside the manifest at {}",
        revision.display()
    );
    let bytes = std::fs::metadata(&artifact).unwrap().len();
    assert!(
        bytes > 512 * 1024,
        "the artifact is {bytes} bytes, too small to be a precompiled policy"
    );

    // A server with no precompiler — the deployment's serving host — serves it.
    let mut without_precompiler = with_precompiler.clone();
    without_precompiler.socket = temporary.path().join("second.sock");
    without_precompiler.toolchains.wasm = None;
    let (_server, stop, thread) = start(&without_precompiler);
    let mut client = Client::connect(&without_precompiler.socket).unwrap();
    let program = program_server::Program {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: toml::Table::new(),
    };
    let module = client
        .resolve(
            "scheduling",
            &program,
            &Target::new(program_server::manifest::WASM_TARGET),
            program_server::manifest::POLICY_DESCRIPTOR,
            &mut |_| {},
        )
        .expect("the installed policy resolves without a compiler");
    assert_eq!(module.form, ModuleForm::Wasm);
    assert_eq!(
        file_of(&module, program_server::source::MODULE_WASM).len() as u64,
        bytes,
        "the resolved module is the artifact the install kept"
    );
    stop.store(true, Ordering::SeqCst);
    thread.join().unwrap();
}
