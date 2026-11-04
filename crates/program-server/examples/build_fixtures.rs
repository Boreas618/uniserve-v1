//! Builds a deployment's descriptor keys through the Program Server.
//!
//! This is the end-to-end proof of design section 5.3 for a fixture: for every
//! `operator<TAB>key` line of the key files, the operator's packages are
//! installed into a store, the keys are prepared for one target, and each key
//! is resolved to the module it names. It builds real modules — Triton, nvcc,
//! CuTe DSL and wasm all run — so it needs the host's toolchains and takes
//! minutes rather than seconds.
//!
//! The keys come from `crates/ops`'s own generators, which is what makes the
//! run meaningful: they are the descriptors a deployment asks for, not a
//! sample this tool chooses.
//!
//! ```text
//! cargo run -p program-server --example build_fixtures -- \
//!     --keys notes/programs/qwen3-moe-keys.txt \
//!     --keys notes/programs/minimax-h3-keys.txt \
//!     --target sm_100a
//! ```
//!
//! It prints one line per key, `ok` or `FAILED` with the reason, and exits
//! non-zero when any key failed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use program_server::protocol::{Descriptors, PrepareRequest, ProgramRequest, Selector};
use program_server::server::ServerConfig;
use program_server::toolchain::ToolchainConfig;
use program_server::{Device, Limits, Manifest, ProgramServer, ProgramSlot, Root, Target};

/// One line of a key file: the operator and one descriptor key.
struct Fixture {
    op: String,
    key: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("build_fixtures: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let options = Options::parse()?;
    run_with(options)
}

fn run_with(options: Options) -> Result<(), String> {
    let fixtures = options.fixtures()?;
    if fixtures.is_empty() {
        return Err("no descriptor keys to build".to_owned());
    }

    // Every path the server is given is absolute: a build runs in its own
    // directory with a copy of the package, so a relative root would be read
    // from that copy.
    std::fs::create_dir_all(&options.store)
        .map_err(|error| format!("creating {}: {error}", options.store.display()))?;
    let store = absolute(&options.store)?;
    let packages = absolute(&options.packages)?;
    let python = absolute(&options.python)?;
    let config = ServerConfig {
        socket: store.join("program-server.sock"),
        // `install` copies a package into the root, so the root is the store's
        // and the tree is only read: a root inside the tree would install a
        // package into itself.
        package_root: store.join("packages"),
        store_roots: vec![Root {
            path: store.join("modules"),
            writable: true,
        }],
        upstream: None,
        limits: Limits::default(),
        toolchains: ToolchainConfig {
            nvcc: None,
            python,
            wasm: None,
            max_jobs: options.jobs,
        },
        devices: vec![Device {
            arch: options.target.clone(),
            model: Target::new("b200"),
        }],
    };
    let server = std::sync::Arc::new(
        ProgramServer::open(&config).map_err(|error| error.to_string())?,
    );
    // The build queue is driven while the server answers, so the server runs as
    // its own process would: one thread serving, the rest of this program
    // asking it for modules.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let serving = std::thread::spawn({
        let server = std::sync::Arc::clone(&server);
        let stop = std::sync::Arc::clone(&stop);
        move || server.serve_until(stop)
    });

    // Install every package of every operator the fixtures name: the store is
    // filled from packages the tree holds, exactly as `install` fills it.
    let mut backends: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ops: Vec<String> = fixtures.iter().map(|one| one.op.clone()).collect();
    ops.sort();
    ops.dedup();
    for op in &ops {
        let directory = packages.join(op);
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) => return Err(format!("reading {}: {error}", directory.display())),
        };
        for entry in entries {
            let entry = entry.map_err(|error| error.to_string())?;
            if !entry.path().is_dir() || !entry.path().join("package.toml").is_file() {
                continue;
            }
            let installed = server
                .install(&entry.path())
                .map_err(|error| format!("{}: {error}", entry.path().display()))?;
            backends
                .entry(installed.package.op.clone())
                .or_default()
                .push(installed.package.backend.clone());
        }
    }

    // A key is covered when one installed backend prepares and resolves it. A
    // second backend that cannot is not a failure: the operator is served.
    let mut failures = Vec::new();
    let mut built = 0_usize;
    for op in &ops {
        let keys: Vec<String> = fixtures
            .iter()
            .filter(|one| &one.op == op)
            .map(|one| one.key.clone())
            .collect();
        let mut resolved: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        for backend in backends.get(op).map(Vec::as_slice).unwrap_or_default() {
            let request = PrepareRequest {
                targets: vec![options.target.clone()],
                programs: vec![ProgramRequest {
                    slot: ProgramSlot::all(op),
                    selector: Selector::Latest {
                        backend: backend.clone(),
                        arguments: toml::Table::new(),
                    },
                }],
                descriptors: vec![Descriptors {
                    op: op.clone(),
                    keys: keys.clone(),
                }],
            };
            let prepared = match server.prepare(&request, &mut |_| {}) {
                Ok(prepared) => prepared,
                Err(error) => {
                    failures.push(format!("{op} {backend}: prepare: {error}"));
                    continue;
                }
            };
            let program = prepared.programs[0].program.clone();
            for key in &keys {
                match server.resolve(op, &program, &options.target, key, &mut |_| {}) {
                    Ok(module) => {
                        resolved
                            .entry(key.clone())
                            .or_default()
                            .insert(backend.clone(), module.files.len().to_string());
                    }
                    Err(error) => {
                        failures.push(format!("{op} {backend} {key}: resolve: {error}"));
                    }
                }
            }
        }
        for key in &keys {
            match resolved.get(key) {
                Some(backends) => {
                    built += 1;
                    let named: Vec<String> = backends
                        .iter()
                        .map(|(backend, files)| format!("{backend} ({files} files)"))
                        .collect();
                    println!("ok\t{op}\t{key}\t{}", named.join(", "));
                }
                None => {
                    failures.push(format!("{op} {key}: no backend resolved it"));
                    println!("FAILED\t{op}\t{key}");
                }
            }
        }
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = serving.join();

    println!(
        "\n{built} of {} descriptors built into the store at {}",
        fixtures.len(),
        config.store_roots[0].path.display()
    );
    if !failures.is_empty() {
        eprintln!("\n{} failure(s):", failures.len());
        for failure in &failures {
            eprintln!("  {failure}");
        }
        return Err(format!("{} of {} keys did not build", failures.len(), fixtures.len()));
    }
    Ok(())
}

/// What the run was asked to do.
struct Options {
    keys: Vec<PathBuf>,
    packages: PathBuf,
    target: Target,
    store: PathBuf,
    python: PathBuf,
    jobs: usize,
}

impl Options {
    /// The defaults are this repository's: both fixtures of `crates/ops`'s
    /// generators, the tree's kernel packages, and the Program Server's own
    /// Python environment.
    fn parse() -> Result<Self, String> {
        let mut options = Options {
            keys: Vec::new(),
            packages: PathBuf::from("crates/ops/kernels"),
            target: Target::new("sm_100a"),
            store: std::env::temp_dir().join("uniserve-build-fixtures"),
            python: PathBuf::from("crates/program-server/python/.venv/bin/python"),
            jobs: 4,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let mut value = || {
                arguments
                    .next()
                    .ok_or_else(|| format!("{flag} takes a value"))
            };
            match flag.as_str() {
                "--keys" => options.keys.push(PathBuf::from(value()?)),
                "--packages" => options.packages = PathBuf::from(value()?),
                "--target" => options.target = Target::new(&value()?),
                "--store" => options.store = PathBuf::from(value()?),
                "--python" => options.python = PathBuf::from(value()?),
                "--jobs" => {
                    options.jobs = value()?
                        .parse()
                        .map_err(|error| format!("--jobs: {error}"))?
                }
                "-h" | "--help" => {
                    println!("{}", include_str!("build_fixtures.rs").lines().take(20).collect::<Vec<_>>().join("\n"));
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }
        if options.keys.is_empty() {
            options.keys = vec![
                PathBuf::from("notes/programs/qwen3-moe-keys.txt"),
                PathBuf::from("notes/programs/minimax-h3-keys.txt"),
            ];
        }
        if !options.python.is_file() {
            return Err(format!(
                "no interpreter at {}: `uv sync --project crates/program-server/python` \
                 builds the Program Server's environment",
                options.python.display()
            ));
        }
        Ok(options)
    }

    /// The fixtures, read from the key files: `operator<TAB>key`, `#` comments.
    fn fixtures(&self) -> Result<Vec<Fixture>, String> {
        let mut fixtures = Vec::new();
        for path in &self.keys {
            let text = std::fs::read_to_string(path)
                .map_err(|error| format!("reading {}: {error}", path.display()))?;
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let Some((op, key)) = line.split_once('\t') else {
                    return Err(format!(
                        "{}: a fixture line is `operator<TAB>key`, not {line:?}",
                        path.display()
                    ));
                };
                fixtures.push(Fixture {
                    op: op.trim().to_owned(),
                    key: key.trim().to_owned(),
                });
            }
        }
        Ok(fixtures)
    }
}

/// The absolute form of a path, without resolving its last link.
///
/// The environment's interpreter is a symlink to the host's Python, and the
/// toolchains are discovered through the environment that link names: the path
/// must stay the environment's, so this makes it absolute and nothing else.
fn absolute(path: &Path) -> Result<PathBuf, String> {
    std::path::absolute(path).map_err(|error| format!("{}: {error}", path.display()))
}

/// Reads a package's manifest without installing it: the tool checks a package
/// exists where a fixture names one before the store says so less directly.
#[allow(dead_code)]
fn manifest_of(package: &Path) -> Result<Manifest, String> {
    Manifest::load(package).map_err(|error| format!("{}: {error}", package.display()))
}
