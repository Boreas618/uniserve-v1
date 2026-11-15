//! The module store (design section 5.3): paths made of names, modules of a
//! writable root and shared read-only roots, the `.committed` mark that makes
//! a directory a module and records its interface version, and publication
//! through a staging directory its writer owns.

use std::fs;
use std::io;
use std::os::fd::AsRawFd as _;
use std::path::{Path, PathBuf};

use build_server::Error;
use build_server::store::{COMMITTED, ModulePath, Root, Store};

use crate::support::{arch, write};

/// The inputs digest the paths of these tests name.
const INPUTS: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// A store whose one root is the writable directory `root`.
fn store(root: &Path) -> Store {
    Store::open(vec![Root {
        path: root.to_path_buf(),
        writable: true,
    }])
    .unwrap()
}

/// The path of a module of `rms_norm`, backend `triton`, at `revision`.
fn path_at(revision: u32) -> ModulePath {
    ModulePath::new(
        "rms_norm",
        "triton",
        revision,
        INPUTS,
        arch().as_str(),
        "dtype_bf16_hidden_2048",
    )
    .unwrap()
}

fn path() -> ModulePath {
    path_at(3)
}

/// Writes a build's files into the workspace directory `directory` and
/// returns it.
fn workspace(directory: &Path, files: &[(&str, &[u8])]) -> PathBuf {
    write(directory, files);
    directory.to_path_buf()
}

/// The names `directory` holds, sorted.
fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn a_published_module_is_found_with_its_files_and_the_mark_is_not_one_of_them() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(&temporary.path().join("store"));
    let workspace = workspace(
        &temporary.path().join("workspace"),
        &[("module.cubin", b"cubin bytes"), ("metadata.json", b"{}")],
    );

    store
        .publish(
            &path(),
            &workspace,
            &["module.cubin".to_owned(), "metadata.json".to_owned()],
            1,
        )
        .unwrap();

    assert!(store.contains(&path()));
    let directory = store.find(&path()).expect("the module's directory");
    assert!(directory.join(COMMITTED).is_file());
    let files = store.files(&path()).unwrap().expect("the module's files");
    let names: Vec<&str> = files.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["metadata.json", "module.cubin"]);
    assert_eq!(fs::read(&files[1].1).unwrap(), b"cubin bytes");
}

/// Only a mark that records an interface version makes a directory a module:
/// a directory without one, or whose mark states no version, is found by no
/// lookup and listed by nothing.
#[test]
fn a_directory_without_a_recorded_mark_is_not_a_module() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(&temporary.path().join("store"));

    let unmarked = store.directory(&path_at(3));
    write(&unmarked, &[("module.cubin", b"half written")]);
    let unrecorded = store.directory(&path_at(4));
    write(
        &unrecorded,
        &[("module.cubin", b"unrecorded"), (COMMITTED, b"")],
    );

    for path in [path_at(3), path_at(4)] {
        assert!(!store.contains(&path), "{path} is no module");
        assert!(store.files(&path).unwrap().is_none());
        assert_eq!(store.interface_version(&path), None);
    }
    assert!(store.list().unwrap().is_empty());
}

/// A module's mark records the version of the interface its package
/// implements, which `resolve` hands to the loader with its files.
#[test]
fn a_module_records_the_interface_version_it_was_built_for() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(&temporary.path().join("store"));
    let workspace = workspace(
        &temporary.path().join("workspace"),
        &[("module.cubin", b"recorded")],
    );

    store
        .publish(&path(), &workspace, &["module.cubin".to_owned()], 2)
        .unwrap();
    assert_eq!(store.interface_version(&path()), Some(2));
    assert_eq!(store.list().unwrap(), [path()]);
}

#[test]
fn a_module_of_a_shared_read_only_root_is_found_and_never_written() {
    let temporary = tempfile::tempdir().unwrap();
    let shared = temporary.path().join("shared");
    let built = temporary.path().join("built");

    // Another host built this module into the root it shares.
    let elsewhere = workspace(
        &temporary.path().join("elsewhere"),
        &[("module.cubin", b"shared cubin")],
    );
    self::store(&shared)
        .publish(&path(), &elsewhere, &["module.cubin".to_owned()], 1)
        .unwrap();

    let store = Store::open(vec![
        Root {
            path: built.clone(),
            writable: true,
        },
        Root {
            path: shared.clone(),
            writable: false,
        },
    ])
    .unwrap();
    assert!(store.contains(&path()));
    let files = store.files(&path()).unwrap().expect("the shared module");
    assert_eq!(fs::read(&files[0].1).unwrap(), b"shared cubin");

    // Publishing goes to the writable root, and leaves the shared one alone.
    let here = workspace(
        &temporary.path().join("here"),
        &[("module.cubin", b"built cubin")],
    );
    store
        .publish(&path(), &here, &["module.cubin".to_owned()], 1)
        .unwrap();
    assert_eq!(
        fs::read(shared.join(path().relative()).join("module.cubin")).unwrap(),
        b"shared cubin"
    );
    assert_eq!(
        fs::read(built.join(path().relative()).join("module.cubin")).unwrap(),
        b"built cubin"
    );
}

/// A path fixes a module's content, so a second publish of a committed path
/// is refused, and the module keeps its files and its record.
#[test]
fn a_committed_module_is_never_replaced() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(&temporary.path().join("store"));
    let first = workspace(
        &temporary.path().join("first"),
        &[("module.cubin", b"first")],
    );
    store
        .publish(&path(), &first, &["module.cubin".to_owned()], 1)
        .unwrap();

    let second = workspace(
        &temporary.path().join("second"),
        &[("module.cubin", b"second")],
    );
    match store.publish(&path(), &second, &["module.cubin".to_owned()], 2) {
        Err(Error::Io(error)) => assert_eq!(error.kind(), io::ErrorKind::AlreadyExists),
        other => panic!("a second publish returned {other:?}"),
    }
    let files = store.files(&path()).unwrap().unwrap();
    assert_eq!(fs::read(&files[0].1).unwrap(), b"first");
    assert_eq!(store.interface_version(&path()), Some(1));
}

/// A writer stages a module beside its place, in a directory it owns through
/// a lock it holds for the staging's life. A crashed writer's staging
/// directory, whose lock no process holds, is removed by the next publish
/// beside it; a live writer's is left alone; and a publish leaves nothing
/// beside the module of its own.
#[test]
fn a_publish_sweeps_a_crashed_writers_staging_and_leaves_a_live_ones() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(&temporary.path().join("store"));
    let parent = store.directory(&path()).parent().unwrap().to_path_buf();

    // What a writer that crashed left: its staging directory and its lock.
    write(
        &parent,
        &[
            (".staged-crashed.lock", b""),
            (".staged-crashed/module.cubin", b"partial"),
        ],
    );
    // A writer still staging holds its lock.
    let live = fs::File::create(parent.join(".staged-live.lock")).unwrap();
    // SAFETY: flock on a descriptor this test owns for the call.
    assert_eq!(unsafe { libc::flock(live.as_raw_fd(), libc::LOCK_EX) }, 0);
    write(&parent, &[(".staged-live/module.cubin", b"in progress")]);

    let workspace = workspace(
        &temporary.path().join("workspace"),
        &[("module.cubin", b"whole")],
    );
    store
        .publish(&path(), &workspace, &["module.cubin".to_owned()], 1)
        .unwrap();

    assert_eq!(
        names(&parent),
        [
            ".staged-live",
            ".staged-live.lock",
            path().descriptor.as_str()
        ]
    );
    let files = store.files(&path()).unwrap().unwrap();
    assert_eq!(fs::read(&files[0].1).unwrap(), b"whole");
}

/// A listing names each committed module once, from whichever roots hold it.
#[test]
fn list_names_every_committed_module_of_every_root_once() {
    let temporary = tempfile::tempdir().unwrap();
    let built = temporary.path().join("built");
    let shared = temporary.path().join("shared");
    let workspace = workspace(
        &temporary.path().join("workspace"),
        &[("module.cubin", b"bytes")],
    );
    let publish = |root: &Path, path: &ModulePath| {
        self::store(root)
            .publish(path, &workspace, &["module.cubin".to_owned()], 1)
            .unwrap();
    };
    // This host built revisions 3 and 4; another host built 3 and 5 into the
    // shared root.
    publish(&built, &path_at(3));
    publish(&built, &path_at(4));
    publish(&shared, &path_at(3));
    publish(&shared, &path_at(5));

    let store = Store::open(vec![
        Root {
            path: built,
            writable: true,
        },
        Root {
            path: shared,
            writable: false,
        },
    ])
    .unwrap();
    assert_eq!(store.list().unwrap(), [path_at(3), path_at(4), path_at(5)]);
}

#[test]
fn a_module_path_is_made_of_names_that_cannot_leave_their_root() {
    for name in ["", ".", "..", "a/b", "../sibling"] {
        let path = ModulePath::new("rms_norm", name, 1, INPUTS, arch().as_str(), "dtype_bf16");
        assert!(path.is_err(), "{name:?} is not one name");
    }
    let path = ModulePath::new(
        "rms_norm",
        "triton",
        1,
        INPUTS,
        arch().as_str(),
        "dtype_bf16",
    )
    .unwrap();
    assert_eq!(
        path.to_string(),
        format!("rms_norm/triton/1/{INPUTS}/{}/dtype_bf16", arch())
    );
}

#[test]
fn a_build_may_not_publish_a_file_outside_the_module_directory() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(&temporary.path().join("store"));
    let cubin = workspace(
        &temporary.path().join("cubin"),
        &[("module.cubin", b"bytes")],
    );

    for name in ["/etc/passwd", "../escape", "tuning/../../escape"] {
        let refused = store.publish(&path(), &cubin, &[name.to_owned()], 1);
        assert!(refused.is_err(), "{name:?} escapes the module directory");
        assert!(!store.contains(&path()));
    }
    // A file in a subdirectory of the module is a name, not an escape.
    let tuning = workspace(
        &temporary.path().join("tuning"),
        &[("tuning/tactics.json", b"{}")],
    );
    store
        .publish(&path(), &tuning, &["tuning/tactics.json".to_owned()], 1)
        .unwrap();
    let files = store.files(&path()).unwrap().unwrap();
    assert_eq!(files[0].0, "tuning/tactics.json");
}

/// A store has exactly one writable root, the first: an empty set, a second
/// writable root and a read-only first root are refused.
#[test]
fn a_store_has_one_writable_root_and_it_comes_first() {
    let temporary = tempfile::tempdir().unwrap();
    let root = |name: &str, writable: bool| Root {
        path: temporary.path().join(name),
        writable,
    };

    assert!(matches!(Store::open(Vec::new()), Err(Error::Io(_))));
    assert!(matches!(
        Store::open(vec![root("one", true), root("two", true)]),
        Err(Error::Io(_))
    ));
    assert!(matches!(
        Store::open(vec![root("shared", false), root("built", true)]),
        Err(Error::Io(_))
    ));
}

/// A descriptor key names a module's last path segment, so it must be one
/// file name: a key longer than a filesystem allows one name is refused when
/// the path is made, with both lengths in the refusal, rather than failing
/// later in a build whose log cannot explain it.
#[test]
fn a_descriptor_key_too_long_to_name_a_directory_is_refused() {
    let longest = build_server::store::DESCRIPTOR_NAME_LIMIT;
    let at_limit = "d".repeat(longest);
    let module = |key: &str| {
        ModulePath::new(
            "attention_gather_rows",
            "triton",
            0,
            INPUTS,
            arch().as_str(),
            key,
        )
        .map_err(|error| error.to_string())
    };
    assert!(
        module(&at_limit).is_ok(),
        "a key of exactly the limit names a directory"
    );

    let over = format!("{at_limit}d");
    let message = module(&over).expect_err("a key over the limit names no directory");
    assert!(
        message.contains(&format!("{} bytes", over.len()))
            && message.contains(&longest.to_string()),
        "the refusal states the key's length and the limit: {message}"
    );
}
