//! The `uniserve build-server` command (design section 6.1): the arguments
//! an operator types and a daemon passes, and a deployment's `build_config`,
//! read into the server's configuration. No server runs here.

use std::path::PathBuf;

use build_server::command::{Args, Command, ServeArgs, Settings};
use build_server::manifest::ToolchainKind;
use build_server::store::Root;
use build_server::toolchain::PeakMemory;
use clap::Parser as _;

use crate::support::{arch, other_arch};

const GIB: u64 = 1 << 30;

fn serve(arguments: &[&str]) -> ServeArgs {
    let args = Args::try_parse_from(["build-server", "serve"].iter().chain(arguments))
        .expect("the serve command parses");
    match args.command {
        Command::Serve(serve) => *serve,
        other => panic!("a serve command, not {other:?}"),
    }
}

#[test]
fn serve_takes_the_roots_the_targets_and_the_limits() {
    let arch = arch();
    let serve = serve(&[
        "--socket",
        "/run/uniserve/build-server.sock",
        "--package-root",
        "/var/lib/uniserve/packages",
        "--store-root",
        "/var/lib/uniserve/modules",
        "--store-root",
        "/shared/uniserve/modules",
        "--target",
        arch.as_str(),
        "--max-jobs",
        "4",
        "--host-reserve",
        "8589934592",
        "--peak-memory",
        "triton=12884901888",
        "--peak-memory",
        "cargo=34359738368",
    ]);
    assert!(!serve.daemon);
    assert_eq!(
        serve.settings.socket().expect("a named socket"),
        PathBuf::from("/run/uniserve/build-server.sock")
    );

    let targets = serve
        .settings
        .targets(&|| panic!("named targets read no host device"))
        .expect("named targets need no driver");
    assert_eq!(targets, std::slice::from_ref(&arch));

    let config = serve.settings.config(targets).unwrap();
    assert_eq!(
        config.package_root,
        PathBuf::from("/var/lib/uniserve/packages")
    );
    // The first store root is the one this host writes; a shared root after
    // it is read-only (design section 5.3).
    assert_eq!(
        config.store_roots,
        vec![
            Root {
                path: PathBuf::from("/var/lib/uniserve/modules"),
                writable: true,
            },
            Root {
                path: PathBuf::from("/shared/uniserve/modules"),
                writable: false,
            },
        ]
    );
    assert_eq!(config.targets, [arch]);
    assert_eq!(config.limits.max_jobs, 4);
    assert_eq!(config.limits.host_reserve, 8 * GIB);
    // A stated peak replaces its toolchain's default; the others keep theirs.
    let defaults = PeakMemory::default_for(4);
    assert_eq!(config.limits.peaks.triton, 12 * GIB);
    assert_eq!(config.limits.peaks.cargo, 32 * GIB);
    assert_eq!(config.limits.peaks.nvcc, defaults.nvcc);
    assert_eq!(config.limits.peaks.cute_dsl, defaults.cute_dsl);
    assert!(config.upstream.is_none(), "the upstream is optional");
}

/// Without a `--target` the server builds for the architecture of each of
/// the host's devices, each once, and a host whose devices cannot be read is
/// an error that says why.
#[test]
fn serve_without_a_target_builds_for_the_hosts_devices() {
    let settings = serve(&[]).settings;
    let devices = [arch(), arch(), other_arch()];
    assert_eq!(
        settings.targets(&|| Ok(devices.to_vec())).unwrap(),
        [arch(), other_arch()]
    );

    let refused = settings
        .targets(&|| Err("no CUDA driver on this host".to_owned()))
        .unwrap_err();
    assert!(
        refused.to_string().contains("no CUDA driver on this host"),
        "{refused}"
    );
}

/// The settings a daemon starts a host's Build Server with.
fn deployment() -> Settings {
    Settings {
        socket: None,
        package_root: Some(PathBuf::from("/var/lib/uniserve/packages")),
        store_roots: vec![
            PathBuf::from("/var/lib/uniserve/modules"),
            PathBuf::from("/shared/uniserve/modules"),
        ],
        upstream: Some(PathBuf::from("/run/uniserve/leader.sock")),
        read_only_stores: vec![PathBuf::from("/opt/uniserve/store")],
        targets: vec![arch().to_string(), other_arch().to_string()],
        max_jobs: Some(2),
        host_reserve: Some(GIB),
        peak_memory: vec![
            (ToolchainKind::Triton, 12 * GIB),
            (ToolchainKind::Cargo, 32 * GIB),
        ],
    }
}

/// A daemon starts a host's Build Server from its deployment's settings: the
/// flags it passes after the socket read back as the same settings.
#[test]
fn settings_survive_the_command_line() {
    let settings = deployment();
    let arguments = settings.arguments();
    let arguments: Vec<&str> = arguments.iter().map(String::as_str).collect();
    assert_eq!(serve(&arguments).settings, settings);
    assert!(Settings::default().arguments().is_empty());
}

/// A deployment states the peak memory of its builds in its `build_config` as
/// a table of toolchain kind to bytes, which reads back as written; a kind no
/// toolchain has is refused, in the table as on the command line.
#[test]
fn a_deployment_states_peak_memory_as_a_table_of_toolchain_kinds() {
    let settings: Settings = toml::from_str(
        r#"
max_jobs = 2

[peak_memory]
cargo = 34359738368
triton = 12884901888
"#,
    )
    .unwrap();
    assert_eq!(settings.max_jobs, Some(2));
    assert_eq!(
        settings.peak_memory,
        [
            (ToolchainKind::Triton, 12 * GIB),
            (ToolchainKind::Cargo, 32 * GIB)
        ]
    );

    let written = toml::to_string(&deployment()).unwrap();
    let table: toml::Table = toml::from_str(&written).unwrap();
    assert_eq!(
        table["peak_memory"]["cargo"].as_integer(),
        Some(32 << 30),
        "{written}"
    );
    assert_eq!(toml::from_str::<Settings>(&written).unwrap(), deployment());

    assert!(toml::from_str::<Settings>("[peak_memory]\nwasm = 1\n").is_err());
    for refused in ["wasm=1", "triton", "triton=a lot"] {
        assert!(
            Args::try_parse_from(["build-server", "serve", "--peak-memory", refused]).is_err(),
            "--peak-memory {refused} is refused"
        );
    }
}

/// An operator states nothing: the roots and the cache are this user's own.
#[test]
fn serve_needs_no_flag() {
    let serve = serve(&[]);
    // The socket is this user's default one, in the user's runtime directory; a process
    // without a runtime directory has no default socket and is refused.
    match (serve.settings.socket(), build_server::default_socket()) {
        (Ok(socket), Ok(default)) => assert_eq!(socket, default),
        (socket, default) => assert!(
            socket.is_err() && default.is_err(),
            "a default socket without a runtime directory"
        ),
    }

    let config = serve.settings.config(Vec::new()).unwrap();
    assert!(
        config
            .package_root
            .ends_with("uniserve/build-server/packages")
    );
    assert_eq!(config.store_roots.len(), 1);
    assert!(
        config.store_roots[0]
            .path
            .ends_with("uniserve/build-server/modules")
    );
    assert!(config.store_roots[0].writable);
    assert!(config.cache.ends_with("uniserve/build-server"));
}

/// Every path the command takes is absolute before the server sees it.
///
/// A build runs with its working directory set to the store's copy of a
/// package, so a path typed relative to the shell's directory would name
/// something under that copy. The command is where a person types those
/// paths, so the command makes them absolute against the directory it was
/// typed in.
#[test]
fn the_command_makes_every_path_absolute_before_the_server_sees_it() {
    let serve = serve(&[
        "--socket",
        "relative/build-server.sock",
        "--package-root",
        "relative/packages",
        "--store-root",
        "relative/modules",
        "--upstream",
        "relative/leader.sock",
        "--read-only-store",
        "relative/shipped",
    ]);
    // The directory this test runs in is the one the paths were typed in.
    let typed_in = std::env::current_dir().unwrap();

    assert_eq!(
        serve.settings.socket().expect("a named socket"),
        typed_in.join("relative/build-server.sock")
    );
    let config = serve.settings.config(Vec::new()).unwrap();
    assert_eq!(config.package_root, typed_in.join("relative/packages"));
    assert_eq!(
        config.store_roots[0].path,
        typed_in.join("relative/modules")
    );
    assert_eq!(config.upstream, Some(typed_in.join("relative/leader.sock")));
    assert_eq!(config.read_only_stores, [typed_in.join("relative/shipped")]);
}

/// A store build names what it builds: kernel keys are built for a named
/// target from a named package tree, and a build of nothing is refused, each
/// before the store is touched.
#[test]
fn a_store_build_names_its_keys_target_and_package_tree() {
    let temporary = tempfile::tempdir().unwrap();
    let keys = temporary.path().join("keys.txt");
    std::fs::write(&keys, "rms_norm\tdtype_bf16\n").unwrap();
    let store = temporary.path().join("store");
    let keys = keys.to_str().unwrap();
    let arch = arch();
    let build = |arguments: &[&str]| {
        let args = Args::try_parse_from(
            ["build-server", "build", "--store", store.to_str().unwrap()]
                .iter()
                .chain(arguments),
        )
        .expect("the build command parses");
        // Every refusal comes before the host's devices would be read.
        build_server::command::run(
            args,
            &|| panic!("a refused build reads no device"),
            &service::logging::Levels::new(None),
        )
        .expect_err("the build is refused")
            .to_string()
    };

    let nothing = build(&[]);
    assert!(nothing.contains("nothing to build"), "{nothing}");
    let no_target = build(&["--keys", keys, "--packages", "kernels"]);
    assert!(no_target.contains("--target"), "{no_target}");
    let no_tree = build(&["--keys", keys, "--target", arch.as_str()]);
    assert!(no_tree.contains("--packages"), "{no_tree}");
    assert!(!store.exists(), "a refused build touched its store");
}

#[test]
fn install_names_a_package_on_the_server_host_and_list_needs_nothing() {
    let args = Args::try_parse_from([
        "build-server",
        "install",
        "/var/lib/uniserve/upload/rms_norm-triton",
        "--socket",
        "/run/uniserve/build-server.sock",
    ])
    .unwrap();
    match args.command {
        Command::Install { socket, package } => {
            assert_eq!(
                socket,
                Some(PathBuf::from("/run/uniserve/build-server.sock"))
            );
            assert_eq!(
                package,
                PathBuf::from("/var/lib/uniserve/upload/rms_norm-triton")
            );
        }
        other => panic!("an install command, not {other:?}"),
    }

    let args = Args::try_parse_from(["build-server", "list"]).unwrap();
    assert!(matches!(args.command, Command::List { socket: None }));
}

#[test]
fn a_command_without_a_subcommand_or_a_package_is_refused() {
    assert!(
        Args::try_parse_from(["build-server"]).is_err(),
        "build-server needs a subcommand"
    );
    assert!(
        Args::try_parse_from(["build-server", "install"]).is_err(),
        "an install command names the package it installs"
    );
}
