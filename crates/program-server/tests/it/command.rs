//! The `uniserve program-server` command (design section 6.1): the arguments
//! the executable mounts and the daemon names, parsed into what the server
//! needs. No server runs here: the test checks the contract the executable and
//! the daemon depend on.

use std::path::PathBuf;

use program_server::command::{Action, command, default_socket, options};
use program_server::manifest::ToolchainKind;
use program_server::store::Root;

#[test]
fn serve_takes_the_socket_the_roots_the_targets_and_the_limits() {
    let matches = command()
        .try_get_matches_from([
            "program-server",
            "serve",
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
        ])
        .expect("the daemon's command line parses");
    let parsed = options(&matches).expect("the options are read");

    assert_eq!(
        parsed.socket,
        PathBuf::from("/run/uniserve/program-server.sock")
    );
    let Action::Serve(serve) = parsed.action else {
        panic!("a serve command");
    };
    assert_eq!(
        serve.package_root,
        PathBuf::from("/var/lib/uniserve/packages")
    );
    // The first store root is the one this host writes; a shared root after
    // it is read-only (design section 5.3).
    let configuration = serve.config(parsed.socket.clone());
    assert_eq!(
        configuration.store_roots,
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
    assert_eq!(serve.limits.max_jobs, 4);
    assert_eq!(serve.limits.host_reserve, 8589934592);
    assert_eq!(serve.devices.len(), 1, "one --target is one device");
    assert_eq!(serve.devices[0].arch.as_str(), "sm_100a");
    assert_eq!(configuration.package_root, serve.package_root);
    assert!(configuration.upstream.is_none(), "the upstream is optional");
}

#[test]
fn serve_defaults_to_the_standard_socket_and_the_bundled_python() {
    let matches = command()
        .try_get_matches_from([
            "program-server",
            "serve",
            "--package-root",
            "/var/lib/uniserve/packages",
            "--store-root",
            "/var/lib/uniserve/modules",
        ])
        .expect("a serve command needs only its roots");
    let parsed = options(&matches).unwrap();

    assert_eq!(parsed.socket, default_socket());
    let Action::Serve(serve) = parsed.action else {
        panic!("a serve command");
    };
    // A CUDA toolkit is found on the host; the Python environment is the
    // Program Server's own, beside the crate (design section 6.2).
    assert!(serve.toolchains.nvcc.is_none());
    assert_eq!(
        serve.toolchains.python,
        PathBuf::from("python/.venv/bin/python")
    );
    assert!(serve.toolchains.wasm.is_none());
    assert!(
        serve.devices.is_empty(),
        "no --target names this host's devices"
    );
}

#[test]
fn a_second_server_of_one_host_names_its_upstream() {
    let matches = command()
        .try_get_matches_from([
            "program-server",
            "serve",
            "--package-root",
            "/mnt/upstream/packages",
            "--store-root",
            "/var/lib/uniserve/modules",
            "--upstream",
            "/run/uniserve/leader.sock",
            "--nvcc",
            "/usr/local/cuda/bin/nvcc",
        ])
        .unwrap();
    let parsed = options(&matches).unwrap();
    let Action::Serve(serve) = parsed.action else {
        panic!("a serve command");
    };
    assert_eq!(
        serve.upstream,
        Some(PathBuf::from("/run/uniserve/leader.sock")),
        "install is refused and names the upstream"
    );
    assert_eq!(
        serve.toolchains.nvcc,
        Some(PathBuf::from("/usr/local/cuda/bin/nvcc"))
    );
    // A toolchain the host has not got is discovered as missing; the command
    // does not refuse it, because only a build that needs it fails.
    assert_eq!(ToolchainKind::Nvcc.as_str(), "nvcc");
}

#[test]
fn install_names_a_package_on_the_server_host_and_list_needs_nothing() {
    let matches = command()
        .try_get_matches_from([
            "program-server",
            "install",
            "/var/lib/uniserve/upload/rms_norm-triton",
            "--socket",
            "/run/uniserve/program-server.sock",
        ])
        .unwrap();
    let parsed = options(&matches).unwrap();
    assert_eq!(
        parsed.socket,
        PathBuf::from("/run/uniserve/program-server.sock")
    );
    match parsed.action {
        Action::Install { package } => assert_eq!(
            package,
            PathBuf::from("/var/lib/uniserve/upload/rms_norm-triton")
        ),
        other => panic!("an install command, not {other:?}"),
    }

    let matches = command()
        .try_get_matches_from(["program-server", "list"])
        .unwrap();
    assert!(matches!(options(&matches).unwrap().action, Action::List));
}

#[test]
fn a_command_without_a_subcommand_or_a_root_is_refused() {
    // The command refuses at parse time, before anything runs: a server that
    // does not know which of its three jobs it was given, or where its
    // packages are, cannot start.
    assert!(
        command().try_get_matches_from(["program-server"]).is_err(),
        "program-server needs a subcommand"
    );
    assert!(
        command()
            .try_get_matches_from(["program-server", "serve"])
            .is_err(),
        "a serve command names the package root it serves from"
    );
    assert!(
        command()
            .try_get_matches_from(["program-server", "install"])
            .is_err(),
        "an install command names the package it installs"
    );
}
