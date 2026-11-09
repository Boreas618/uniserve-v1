//! The `uniserve program-server` command (design section 6.1): the arguments
//! an operator types and a daemon passes, read into the server's
//! configuration. No server runs here.

use std::path::PathBuf;

use clap::Parser as _;
use program_server::command::{Args, Command, ServeArgs, Settings};
use program_server::store::Root;

fn serve(arguments: &[&str]) -> ServeArgs {
    let args = Args::try_parse_from(["program-server", "serve"].iter().chain(arguments))
        .expect("the serve command parses");
    match args.command {
        Command::Serve(serve) => *serve,
        other => panic!("a serve command, not {other:?}"),
    }
}

#[test]
fn serve_takes_the_roots_the_targets_and_the_limits() {
    let serve = serve(&[
        "--socket",
        "/run/uniserve/program-server.sock",
        "--package-root",
        "/var/lib/uniserve/packages",
        "--store-root",
        "/var/lib/uniserve/modules",
        "--store-root",
        "/shared/uniserve/modules",
        "--target",
        "sm_100a",
        "--max-jobs",
        "4",
        "--host-reserve",
        "8589934592",
        "--python",
        "/srv/program-server/bin/python",
    ]);
    assert!(!serve.daemon);

    let devices = serve
        .settings
        .devices()
        .expect("named targets need no driver");
    assert_eq!(devices.len(), 1, "one --target is one device");
    assert_eq!(devices[0].arch.as_str(), "sm_100a");

    let config = serve.settings.config(devices);
    assert_eq!(
        config.socket,
        PathBuf::from("/run/uniserve/program-server.sock")
    );
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
    assert_eq!(config.limits.max_jobs, 4);
    assert_eq!(config.limits.host_reserve, 8589934592);
    assert_eq!(
        config.toolchains.python,
        Some(PathBuf::from("/srv/program-server/bin/python"))
    );
    assert!(config.upstream.is_none(), "the upstream is optional");
}

/// A daemon starts a host's Program Server from its deployment's settings:
/// the flags it passes after the socket read back as the same settings.
#[test]
fn settings_survive_the_command_line() {
    let settings = Settings {
        socket: None,
        package_root: Some(PathBuf::from("/var/lib/uniserve/packages")),
        store_roots: vec![
            PathBuf::from("/var/lib/uniserve/modules"),
            PathBuf::from("/shared/uniserve/modules"),
        ],
        upstream: Some(PathBuf::from("/run/uniserve/leader.sock")),
        targets: vec!["sm_100a".into(), "sm_90a".into()],
        max_jobs: Some(2),
        host_reserve: Some(1 << 30),
        nvcc: Some(PathBuf::from("/usr/local/cuda/bin/nvcc")),
        python: Some(PathBuf::from("/srv/program-server/bin/python")),
        wasm: Some(PathBuf::from("/srv/bin/scheduling-precompile")),
        cargo: Some(PathBuf::from("/srv/bin/cargo")),
    };
    let arguments = settings.arguments();
    let arguments: Vec<&str> = arguments.iter().map(String::as_str).collect();
    assert_eq!(serve(&arguments).settings, settings);
    assert!(Settings::default().arguments().is_empty());
}

/// An operator states nothing but the interpreter: the roots are this user's
/// own, and a host without an interpreter is missing only its Python
/// toolchains.
#[test]
fn serve_needs_no_flag() {
    let serve = serve(&[]);
    let config = serve.settings.config(Vec::new());
    assert_eq!(config.socket, program_server::default_socket());
    assert!(config.package_root.ends_with("uniserve/programs/packages"));
    assert_eq!(config.store_roots.len(), 1);
    assert!(
        config.store_roots[0]
            .path
            .ends_with("uniserve/programs/modules")
    );
    assert!(config.store_roots[0].writable);
    assert!(config.toolchains.python.is_none());
    assert!(config.toolchains.nvcc.is_none());
}

#[test]
fn install_names_a_package_on_the_server_host_and_list_needs_nothing() {
    let args = Args::try_parse_from([
        "program-server",
        "install",
        "/var/lib/uniserve/upload/rms_norm-triton",
        "--socket",
        "/run/uniserve/program-server.sock",
    ])
    .unwrap();
    match args.command {
        Command::Install { socket, package } => {
            assert_eq!(
                socket,
                Some(PathBuf::from("/run/uniserve/program-server.sock"))
            );
            assert_eq!(
                package,
                PathBuf::from("/var/lib/uniserve/upload/rms_norm-triton")
            );
        }
        other => panic!("an install command, not {other:?}"),
    }

    let args = Args::try_parse_from(["program-server", "list"]).unwrap();
    assert!(matches!(args.command, Command::List { socket: None }));
}

#[test]
fn a_command_without_a_subcommand_or_a_package_is_refused() {
    assert!(
        Args::try_parse_from(["program-server"]).is_err(),
        "program-server needs a subcommand"
    );
    assert!(
        Args::try_parse_from(["program-server", "install"]).is_err(),
        "an install command names the package it installs"
    );
}
