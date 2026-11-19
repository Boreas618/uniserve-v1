//! The module set an apply fixes at a real Build Server, over the operators the deployment's
//! engine revision states: an operator its engine package defines is fixed, pinnable and moved by
//! a newer installed package however old the daemon's own build is, and an operator the engine
//! revision does not state is neither, whatever the daemon's build knows.

use std::io::Write as _;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use build_server::server::{BuildServer, ServerConfig};
use build_server::store::Root;
use daemon::config::{Config, Stated};
use daemon::engine::control::{self, with_store_modules};
use daemon::epoch::{Change, Epoch, change};
use proto::module::{ModuleSlot, PackageRevision, Pin};

/// An operator no build of this tree defines: the engine revision of [`Tree`] defines it, as a
/// later engine revision defines an operator the daemon's build predates.
const LATER: &str = "later_attention";

/// An operator of the daemon's own build that the engine revision of [`Tree`] does not state.
const UNSTATED: &str = proto::module::operator::PAGED_ATTENTION;

/// The descriptor key the deployment pins [`LATER`]'s package for.
const KEY: &str = "dtype_bf16";

/// Writes each file under `root`, creating its directories.
fn write(root: &Path, files: &[(&str, &str)]) {
    for (name, text) in files {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
}

/// A source tree of the three packages every apply fixes, as a workspace of three crates: the
/// engine package, an executable whose vocabulary (`engine/operators.toml`) is [`LATER`] alone,
/// the default scheduling policy's program package and the default processing program's package.
/// None is built: fixing a module set selects among installed revisions and builds nothing.
struct Tree {
    engine: PathBuf,
    policy: PathBuf,
    processing: PathBuf,
}

impl Tree {
    fn write(root: &Path) -> Self {
        let policy = format!(
            "op = \"program\"\nbackend = \"{}\"\n\n[source]\nkind = \"wasm\"\n",
            proto::config::DEFAULT_SCHEDULING
        );
        let processing = format!(
            "op = \"program\"\nbackend = \"{}\"\n\n[source]\nkind = \"wasm\"\n",
            proto::config::DEFAULT_PROCESSING
        );
        write(
            root,
            &[
                (
                    "Cargo.toml",
                    "[workspace]\nmembers = [\"engine\", \"policy\", \"processing\"]\n\
                     resolver = \"2\"\n",
                ),
                (
                    "engine/Cargo.toml",
                    "[package]\nname = \"engine-fixture\"\nversion = \"0.1.0\"\n\
                     edition = \"2024\"\npublish = false\n",
                ),
                ("engine/src/main.rs", "fn main() {}\n"),
                (
                    "engine/operators.toml",
                    &format!("operators = [\"{LATER}\"]\n"),
                ),
                (
                    "engine/package.toml",
                    &format!(
                        "op = \"engine\"\nbackend = \"{}\"\noperators = \"engine/operators.toml\"\n\n\
                         [module]\nform = \"executable\"\n\n\
                         [source]\nkind = \"cargo\"\nbinary = \"engine-fixture\"\n",
                        proto::config::DEFAULT_ENGINE
                    ),
                ),
                (
                    "policy/Cargo.toml",
                    "[package]\nname = \"policy-fixture\"\nversion = \"0.1.0\"\n\
                     edition = \"2024\"\npublish = false\n\n[lib]\ncrate-type = [\"cdylib\"]\n",
                ),
                ("policy/src/lib.rs", ""),
                ("policy/package.toml", &policy),
                (
                    "processing/Cargo.toml",
                    "[package]\nname = \"processing-fixture\"\nversion = \"0.1.0\"\n\
                     edition = \"2024\"\npublish = false\n\n[lib]\ncrate-type = [\"cdylib\"]\n",
                ),
                ("processing/src/lib.rs", ""),
                ("processing/package.toml", &processing),
            ],
        );
        Self {
            engine: root.join("engine"),
            policy: root.join("policy"),
            processing: root.join("processing"),
        }
    }
}

/// Writes a vendor cubin package of `op`'s backend `vendor` covering [`KEY`] into `directory`,
/// whose cubin holds `bytes`: two calls with other bytes are two revisions of one package.
fn kernel_package(directory: &Path, op: &str, bytes: &str) -> PathBuf {
    let manifest = format!(
        "op = \"{op}\"\nbackend = \"vendor\"\ninterface_version = 1\n\n\
         [module]\nmetadata = \"metadata_{{descriptor}}.json\"\n\n\
         [source]\nkind = \"cubin\"\ncubin = \"artifact/module_{{descriptor}}.cubin\"\n\
         metadata = \"artifact/metadata_{{descriptor}}.json\"\n\n\
         [kernel]\ndescriptors = [{{ dimensions = {{ dtype = [\"bf16\"] }} }}]\n"
    );
    write(
        directory,
        &[
            ("package.toml", &manifest),
            (&format!("artifact/module_{KEY}.cubin"), bytes),
            (
                &format!("artifact/metadata_{KEY}.json"),
                r#"{"name": "vendor_kernel"}"#,
            ),
        ],
    );
    directory.to_path_buf()
}

/// A Build Server of fresh roots serving on its socket on a thread of its own, the upstream that
/// takes installations, until it is dropped.
struct Serving {
    socket: PathBuf,
    /// The other end of the descriptor the server stops on.
    stop: UnixStream,
    thread: Option<JoinHandle<()>>,
}

impl Serving {
    /// Opens a server over roots in `directory`, installs `packages` before it serves, and binds
    /// its socket before this returns, so a client connects at once.
    fn start(directory: &Path, packages: &[&Path]) -> Self {
        let server = BuildServer::open(&ServerConfig {
            package_root: directory.join("packages"),
            store_roots: vec![Root {
                path: directory.join("modules"),
                writable: true,
            }],
            upstream: None,
            read_only_stores: Vec::new(),
            limits: build_server::Limits::default(),
            targets: Vec::new(),
            cache: directory.join("cache"),
            // No Python environment: nothing here is compiled.
            source: None,
        })
        .unwrap();
        for package in packages {
            server.install(package).unwrap();
        }

        let socket = directory.join("build-server.sock");
        let (listener, starting) = service::host::bind::<build_server::command::Settings>(&socket)
            .unwrap()
            .owned(&socket)
            .expect("no other server owns the socket");
        let lock = starting
            .finish(&build_server::readiness(
                &socket,
                &build_server::command::Settings::default(),
            ))
            .unwrap();
        let (stop, stopped) = UnixStream::pair().unwrap();
        let server = Arc::new(server);
        let thread = std::thread::spawn(move || server.serve_on(listener, lock, &stopped).unwrap());
        Self {
            socket,
            stop,
            thread: Some(thread),
        }
    }

    /// The server as the leader's daemon reaches it when it fixes a module set.
    fn local(&self) -> control::BuildServer {
        control::BuildServer {
            host: "leader".to_owned(),
            socket: self.socket.clone(),
            targets: Vec::new(),
        }
    }
}

impl Drop for Serving {
    fn drop(&mut self) {
        (&self.stop).write_all(&[1]).unwrap();
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// A deployment whose `[kernels]` table pins `op`'s key [`KEY`] to `vendor@latest`.
fn pinning(op: &str) -> Config {
    let mut config = Config {
        model: Some("org/model".to_owned()),
        ..Config::default()
    };
    config.kernels.insert(
        op.to_owned(),
        [(
            KEY.to_owned(),
            Pin {
                backend: "vendor".to_owned(),
                revision: None,
            },
        )]
        .into(),
    );
    config
}

/// The package `fixed` holds in `slot`, by its backend and revision.
fn held(fixed: &proto::module::ModuleSet, slot: &ModuleSlot) -> Option<(String, u32)> {
    fixed
        .kernels
        .get(slot)
        .map(|package: &PackageRevision| (package.backend.clone(), package.revision))
}

/// An operator the engine revision states and the daemon's build lacks is pinned and fixed in
/// the epoch's module set, a newer installed package of it moves the set as a kernel generation,
/// and an operator of the daemon's build the engine revision does not state is neither fixed nor
/// pinnable.
#[test]
fn the_module_set_is_fixed_over_the_engine_revision_s_operators() {
    let directory = tempfile::tempdir().unwrap();
    let tree = Tree::write(&directory.path().join("tree"));
    let later = kernel_package(&directory.path().join("later"), LATER, "first");
    let unstated = kernel_package(&directory.path().join("unstated"), UNSTATED, "first");
    let serving = Serving::start(
        &directory.path().join("server"),
        &[
            &tree.engine,
            &tree.policy,
            &tree.processing,
            &later,
            &unstated,
        ],
    );

    // The pin is the deployment's to state: the configuration names no engine revision yet.
    let config = pinning(LATER);
    config
        .validate(&Stated::from_paths(Vec::new()))
        .expect("a pin of an operator a later engine defines is a valid configuration");

    let named = daemon::epoch::modules_of(&config);
    let fixed = with_store_modules(&serving.local(), &named).unwrap();
    let pinned = ModuleSlot::key(LATER, KEY);
    let candidate = ModuleSlot {
        op: LATER.to_owned(),
        key: None,
        part: Some("vendor".to_owned()),
    };
    assert_eq!(held(&fixed, &pinned), Some(("vendor".to_owned(), 1)));
    assert_eq!(held(&fixed, &candidate), Some(("vendor".to_owned(), 1)));
    assert!(
        fixed.kernels.keys().all(|slot| slot.op != UNSTATED),
        "the set fixes no package of an operator the engine revision does not state: {:?}",
        fixed.kernels
    );

    // A newer package of the operator is installed: the next apply of the same configuration
    // fixes it, a change of the kernels alone, which the serving engines take as a generation.
    let newer = kernel_package(&directory.path().join("later"), LATER, "second");
    let installed = build_server::Client::connect(&serving.socket)
        .unwrap()
        .install(&newer)
        .unwrap();
    assert_eq!(installed.revision, 2);
    let refixed = with_store_modules(&serving.local(), &named).unwrap();
    assert_eq!(held(&refixed, &pinned), Some(("vendor".to_owned(), 2)));
    assert_eq!(held(&refixed, &candidate), Some(("vendor".to_owned(), 2)));
    assert_eq!(refixed.engine, fixed.engine);
    assert_eq!(
        change(
            &Epoch::with_modules(config.clone(), fixed).unwrap(),
            &Epoch::with_modules(config, refixed).unwrap()
        ),
        Change::Generation
    );

    // An operator the daemon's build has but the engine revision does not state is no
    // operator of the epoch: its pin is refused, naming the engine revision.
    let refused = with_store_modules(
        &serving.local(),
        &daemon::epoch::modules_of(&pinning(UNSTATED)),
    )
    .expect_err("the engine revision states no such operator");
    let refused = format!("{refused:#}");
    assert!(
        refused.contains(UNSTATED) && refused.contains("engine"),
        "{refused}"
    );
}
