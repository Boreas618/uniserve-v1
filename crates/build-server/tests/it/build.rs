//! Builds through the server (design sections 5.3 and 6.2): a vendor cubin and
//! an executable are installed, stored and resolved as they are, a failed
//! build stays failed for the server's life, and a store build
//! (`build_server::build`) fills and extends a store from key lists.

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

use build_server::Error;
use build_server::builder::BuildState;
use build_server::manifest::{Manifest, ModuleForm, SINGLE_KEY};
use build_server::server::{BuildServer, ServerConfig};
use build_server::source::MODULE_WASM;
use build_server::target::Target;

use crate::support::{self, arch};

/// The operator and backend of the vendor cubin package these tests build.
const OP: &str = "paged_attention";
const BACKEND: &str = "trtllm";

/// Two keys of the package, and one it does not cover.
const PAGE_64: &str = "dtype_bf16_page_64";
const PAGE_128: &str = "dtype_bf16_page_128";
const PAGE_32: &str = "dtype_bf16_page_32";

/// The vendor cubin package of [`OP`] for two page sizes, carrying the cubin
/// of each, in `directory`.
fn cubin_package(directory: &Path) -> PathBuf {
    let manifest = support::vendor_manifest(
        OP,
        BACKEND,
        0,
        r#"{ dtype = ["bf16"], page = ["64", "128"] }"#,
        &arch(),
    );
    support::vendor_package(directory, &manifest, &[PAGE_64, PAGE_128])
}

#[test]
fn a_vendor_cubin_is_installed_stored_and_resolved_as_it_is() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();
    let serving = support::serve(server, &temporary.path().join("build-server.sock"));
    let mut client = serving.client();

    let package = cubin_package(&temporary.path().join("package"));
    let installed = client.install(&package).unwrap();
    assert_eq!(installed.revision, 1);

    let mut progress = Vec::new();
    let prepared = client
        .prepare(
            &support::latest(OP, BACKEND, &[PAGE_64], vec![arch()]),
            &mut |step| progress.push(step),
        )
        .unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0].state, BuildState::Committed);
    assert_eq!(progress[0].path.target, arch().to_string());
    assert_eq!(progress[0].path.descriptor, PAGE_64);

    let resolved = client
        .resolve(OP, &prepared[0].1, &arch(), PAGE_64, &mut |_| {})
        .unwrap();
    assert_eq!(resolved.form, ModuleForm::Cubin);
    assert_eq!(resolved.interface_version, 1);
    let names: Vec<&str> = resolved
        .files
        .iter()
        .map(|file| file.name.as_str())
        .collect();
    assert_eq!(
        names,
        [format!("metadata_{PAGE_64}.json").as_str(), "module.cubin"]
    );
    assert_eq!(
        resolved.metadata.as_deref(),
        Some(format!("metadata_{PAGE_64}.json").as_str()),
        "the resolve response names the metadata the module holds"
    );
    assert_eq!(
        support::file_of(&resolved, "module.cubin"),
        support::cubin(PAGE_64),
        "the module is the package's cubin, byte for byte"
    );
    assert!(
        resolved
            .files
            .iter()
            .all(|file| file.path.starts_with(&config.store_roots[0].path)),
        "the module is in this host's writable store root"
    );

    let listing = client.list().unwrap();
    assert_eq!(listing[0].1.len(), 1);
    assert_eq!(listing[0].1[0].descriptor, PAGE_64);
    assert_eq!(listing[0].1[0].form, ModuleForm::Cubin);

    drop(client);
    serving.stop().unwrap();
}

/// An executable package: an executable the main process runs as its own
/// process, stored as it is, implementing version 3 of its interface.
const EXECUTABLE: &str = r#"
op = "sampler"
backend = "process"
interface_version = 3

[source]
kind = "executable"
files = ["bin/sampler"]

[kernel]
requires = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

/// An executable is stored as it is, executable: a process of the server's
/// host starts it from the path `resolve` answers.
#[test]
fn an_executable_revision_is_installed_stored_and_resolved_as_it_is() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let package = temporary.path().join("package");
    let executable = b"#!/bin/sh\nexit 0\n";
    support::write(
        &package,
        &[
            (Manifest::FILE, support::for_arch(EXECUTABLE).as_bytes()),
            ("bin/sampler", executable),
        ],
    );
    let program = package.join("bin/sampler");
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();

    let installed = server.install(&package).unwrap();
    let revision = support::revision(&installed.backend, installed.revision);
    let resolved = server
        .resolve("sampler", &revision, &arch(), "dtype_bf16", &mut |_| {})
        .unwrap();

    assert_eq!(resolved.form, ModuleForm::Executable);
    assert!(resolved.metadata.is_none());
    assert!(resolved.entries.is_empty());
    assert_eq!(
        resolved.interface_version, 3,
        "the module carries the interface version its package's manifest states"
    );
    assert_eq!(support::file_of(&resolved, "bin/sampler"), executable);
    let mode = std::fs::metadata(&resolved.files[0].path)
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(
        mode & 0o111,
        0o111,
        "the stored executable stays executable"
    );
}

/// A failed build stays failed for the server's life, for `resolve` as for
/// `prepare`, so a broken package cannot spin; a server opened again builds
/// it anew.
#[test]
fn a_failed_build_stays_failed_for_the_servers_life() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();
    let package = cubin_package(&temporary.path().join("package"));
    // Launch metadata that names no kernel: a module nothing can launch,
    // which the build refuses.
    let metadata = format!("artifact/metadata_{PAGE_64}.json");
    std::fs::write(package.join(&metadata), br#"{"shared": 0}"#).unwrap();
    let installed = server.install(&package).unwrap();
    let revision = support::revision(BACKEND, installed.revision);

    let mut progress = Vec::new();
    let failed = server
        .resolve(OP, &revision, &arch(), PAGE_64, &mut |step| {
            progress.push(step)
        })
        .expect_err("a module nothing can launch is not built");
    assert!(matches!(&failed, Error::Build { .. }), "{failed}");
    assert!(failed.to_string().contains("no name field"), "{failed}");
    assert!(matches!(progress[..], [ref step] if matches!(step.state, BuildState::Failed(_))));

    // An installed revision is never edited in service; the test repairs this
    // one so that a build retried now would succeed, which leaves the
    // remembered failure as the only reason for another.
    let installed_directory = config
        .package_root
        .join(format!("{OP}/{BACKEND}/{}", installed.revision));
    std::fs::write(installed_directory.join(&metadata), support::METADATA).unwrap();

    let mut retried = Vec::new();
    let again = server
        .resolve(OP, &revision, &arch(), PAGE_64, &mut |step| {
            retried.push(step)
        })
        .expect_err("the failure stays");
    assert_eq!(again.to_string(), failed.to_string());
    assert!(retried.is_empty(), "no build ran again");
    let prepared = server.prepare(
        &support::latest(OP, BACKEND, &[PAGE_64], vec![arch()]),
        &mut |_| {},
    );
    assert!(matches!(prepared, Err(Error::Build { .. })));

    drop(server);
    let reopened = BuildServer::open(&config).unwrap();
    let module = reopened
        .resolve(OP, &revision, &arch(), PAGE_64, &mut |_| {})
        .unwrap();
    assert_eq!(
        support::file_of(&module, "module.cubin"),
        support::cubin(PAGE_64)
    );
}

/// What one store build reported: each key with the backends that built it.
type Reports = Vec<(String, Vec<String>)>;

/// Builds the [`OP`] `keys` from the package tree `tree` into the store
/// directory `store`, for [`arch`], and returns the failures and what was
/// reported of each key.
fn store_build(store: &Path, cache: &Path, tree: &Path, keys: &[&str]) -> (Vec<String>, Reports) {
    let keys: Vec<(String, String)> = keys
        .iter()
        .map(|key| (OP.to_owned(), (*key).to_owned()))
        .collect();
    let mut reports = Reports::new();
    let failures = build_server::build::build(
        &store_config(store, cache),
        tree,
        &keys,
        Some(&arch()),
        &[],
        &mut |op, key, backends| {
            assert_eq!(op, OP);
            let backends = backends
                .iter()
                .map(|(backend, _)| backend.clone())
                .collect();
            reports.push((key.to_owned(), backends));
        },
    )
    .unwrap();
    (failures, reports)
}

/// The configuration of a server over the store directory `store`.
fn store_config(store: &Path, cache: &Path) -> ServerConfig {
    let mut config = support::config(store);
    config.cache = cache.to_path_buf();
    config
}

/// What a server over the store directory `store` serves: each revision of
/// the [`OP`] package with the keys of its modules.
fn served(store: &Path, cache: &Path) -> Vec<(u32, Vec<String>)> {
    let server = BuildServer::open(&store_config(store, cache)).unwrap();
    server
        .list()
        .unwrap()
        .into_iter()
        .map(|(package, modules)| {
            let keys = modules
                .into_iter()
                .map(|module| module.descriptor)
                .collect();
            (package.revision, keys)
        })
        .collect()
}

/// A store is extended by building more keys into it: a package whose files
/// are unchanged keeps its installed revision, so the modules built earlier
/// stay under the revision the store serves, and Python bytecode beside the
/// sources is not a change. A changed package is installed as a new revision.
#[test]
fn a_build_into_an_existing_store_keeps_the_revision_of_an_unchanged_package() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let (store, cache) = (
        temporary.path().join("store"),
        temporary.path().join("cache"),
    );
    let tree = temporary.path().join("kernels");
    let package = cubin_package(&tree.join(OP).join(BACKEND));

    let (failures, _) = store_build(&store, &cache, &tree, &[PAGE_64]);
    assert!(failures.is_empty(), "{failures:?}");
    assert_eq!(served(&store, &cache), [(1, vec![PAGE_64.to_owned()])]);

    // Bytecode a Python build or check left beside the sources.
    support::write(&package, &[("__pycache__/build.cpython-312.pyc", b"\0")]);
    store_build(&store, &cache, &tree, &[PAGE_128]);
    assert_eq!(
        served(&store, &cache),
        [(1, vec![PAGE_128.to_owned(), PAGE_64.to_owned()])],
        "an unchanged package was installed again"
    );

    support::write(
        &package,
        &[(
            &format!("artifact/metadata_{PAGE_64}.json"),
            br#"{"name": "edited"}"#,
        )],
    );
    store_build(&store, &cache, &tree, &[PAGE_64]);
    assert_eq!(
        served(&store, &cache),
        [
            (1, vec![PAGE_128.to_owned(), PAGE_64.to_owned()]),
            (2, vec![PAGE_64.to_owned()])
        ],
        "a changed package kept its revision"
    );
}

/// A store's modules are found by what built them, not by a revision number
/// alone: a package root installed again from other files numbers its
/// revision 1 again, and the store returns the module of those files rather
/// than the one the earlier revision 1 committed.
#[test]
fn a_revision_numbered_again_never_serves_the_module_of_other_files() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let (store, cache) = (
        temporary.path().join("store"),
        temporary.path().join("cache"),
    );
    let tree = temporary.path().join("kernels");
    let package = cubin_package(&tree.join(OP).join(BACKEND));
    store_build(&store, &cache, &tree, &[PAGE_64]);

    std::fs::remove_dir_all(store.join(build_server::build::PACKAGES)).unwrap();
    let edited = [support::cubin(PAGE_64).as_slice(), b"edited"].concat();
    support::write(
        &package,
        &[(&format!("artifact/module_{PAGE_64}.cubin"), &edited)],
    );
    store_build(&store, &cache, &tree, &[PAGE_64]);
    assert_eq!(served(&store, &cache), [(1, vec![PAGE_64.to_owned()])]);

    let server = BuildServer::open(&store_config(&store, &cache)).unwrap();
    let module = server
        .resolve(
            OP,
            &support::revision(BACKEND, 1),
            &arch(),
            PAGE_64,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(
        support::file_of(&module, "module.cubin"),
        edited,
        "the store served the module of the earlier installation's files"
    );
}

/// A store build reports each key with the backends that built it, names a
/// key no package built among its failures, and still builds the rest.
#[test]
fn a_build_reports_the_keys_no_package_built_and_builds_the_rest() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let (store, cache) = (
        temporary.path().join("store"),
        temporary.path().join("cache"),
    );
    let tree = temporary.path().join("kernels");
    cubin_package(&tree.join(OP).join(BACKEND));

    let (failures, reports) = store_build(&store, &cache, &tree, &[PAGE_64, PAGE_32]);
    assert_eq!(
        reports,
        [
            (PAGE_64.to_owned(), vec![BACKEND.to_owned()]),
            (PAGE_32.to_owned(), Vec::new()),
        ]
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains(PAGE_32), "{failures:?}");
    assert_eq!(served(&store, &cache), [(1, vec![PAGE_64.to_owned()])]);
}

/// The directory of the workspace member whose manifest is the package `op`
/// `backend`, as `cargo metadata` lists the members of this crate's
/// workspace.
fn workspace_package(op: &str, backend: &str) -> PathBuf {
    let output = std::process::Command::new("cargo")
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|member| Path::new(member["manifest_path"].as_str()?).parent())
        .filter(|directory| directory.join(Manifest::FILE).is_file())
        .find(|directory| {
            let manifest = Manifest::load(directory).unwrap();
            manifest.op == op && manifest.backend == backend
        })
        .unwrap_or_else(|| panic!("no workspace member is the package {op} {backend}"))
        .to_path_buf()
}

/// A scheduling policy is built by the server from the snapshot its
/// installation took of its whole workspace, with the lockfile's sources
/// vendored and nothing fetched; a second server over the same roots resolves
/// the committed module without building it.
#[test]
#[ignore = "builds the workspace's arrival policy for wasm32 with cargo: run with --ignored"]
fn the_workspace_policy_is_built_from_its_installed_snapshot() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let config = support::config(temporary.path());
    let server = BuildServer::open(&config).unwrap();
    let installed = server
        .install(&workspace_package("scheduling", "arrival"))
        .unwrap();
    let revision = support::revision(&installed.backend, installed.revision);
    server
        .prepare(
            &support::latest("scheduling", "arrival", &[SINGLE_KEY], Vec::new()),
            &mut |_| {},
        )
        .expect("the server builds the policy from its snapshot");
    drop(server);

    let server = BuildServer::open(&config).unwrap();
    let mut progress = Vec::new();
    let module = server
        .resolve(
            "scheduling",
            &revision,
            &Target::wasm(),
            SINGLE_KEY,
            &mut |step| progress.push(step),
        )
        .unwrap();
    assert!(progress.is_empty(), "the committed module was built again");
    assert_eq!(module.form, ModuleForm::Wasm);
    assert!(
        support::file_of(&module, MODULE_WASM).starts_with(b"\0asm"),
        "the module is a wasm32 binary"
    );
}
