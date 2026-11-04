//! The module store (design section 5.3): paths made of names, modules of a
//! writable root and shared read-only roots, and the `.committed` mark that
//! makes a directory a module.

use std::fs;

use program_server::Error;
use program_server::store::{COMMITTED, ModulePath, Root, Store};

/// A store with one writable root under a fresh temporary directory.
fn store(directory: &std::path::Path) -> Store {
    Store::open(vec![Root {
        path: directory.join("store"),
        writable: true,
    }])
    .unwrap()
}

/// A path of one module of `rms_norm`, backend `triton`, revision 3.
fn path() -> ModulePath {
    ModulePath::new("rms_norm", "triton", 3, "sm_100a", "dtype_bf16_hidden_2048").unwrap()
}

/// Writes a build's files into a fresh workspace directory and answers it.
fn workspace(directory: &std::path::Path, files: &[(&str, &[u8])]) -> std::path::PathBuf {
    let workspace = directory.join("workspace");
    fs::create_dir_all(&workspace).unwrap();
    for (name, bytes) in files {
        let path = workspace.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, bytes).unwrap();
    }
    workspace
}

#[test]
fn a_published_module_is_found_with_its_files_and_the_mark_is_not_one_of_them() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(temporary.path());
    let workspace = workspace(
        temporary.path(),
        &[("module.cubin", b"cubin bytes"), ("metadata.json", b"{}")],
    );

    store
        .publish(
            &path(),
            &workspace,
            &["module.cubin".to_owned(), "metadata.json".to_owned()],
        )
        .unwrap();

    assert!(store.contains(&path()));
    let directory = store.find(&path()).expect("the module's directory");
    assert!(directory.join(COMMITTED).is_file());
    let files = store.files(&path()).unwrap().expect("the module's files");
    let names: Vec<&str> = files.iter().map(|file| file.name.as_str()).collect();
    assert_eq!(names, ["metadata.json", "module.cubin"]);
    assert_eq!(fs::read(&files[1].path).unwrap(), b"cubin bytes");
}

#[test]
fn a_directory_without_the_mark_is_not_a_module() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(temporary.path());
    let directory = temporary.path().join("store").join(path().relative());
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("module.cubin"), b"half written").unwrap();

    assert!(!store.contains(&path()));
    assert!(store.files(&path()).unwrap().is_none());
    assert!(store.list().unwrap().is_empty());
}

#[test]
fn a_module_of_a_shared_read_only_root_is_found_and_never_written() {
    let temporary = tempfile::tempdir().unwrap();
    let shared = temporary.path().join("shared");
    let store = Store::open(vec![
        Root {
            path: temporary.path().join("built"),
            writable: true,
        },
        Root {
            path: shared.clone(),
            writable: false,
        },
    ])
    .unwrap();

    // Another host built this module into the shared root.
    let elsewhere = shared.join(path().relative());
    fs::create_dir_all(&elsewhere).unwrap();
    fs::write(elsewhere.join("module.cubin"), b"shared cubin").unwrap();
    fs::write(elsewhere.join(COMMITTED), b"").unwrap();

    assert!(store.contains(&path()));
    let files = store.files(&path()).unwrap().expect("the shared module");
    assert_eq!(fs::read(&files[0].path).unwrap(), b"shared cubin");

    // Publishing goes to the writable root, and leaves the shared one alone.
    let workspace = workspace(temporary.path(), &[("module.cubin", b"built cubin")]);
    store
        .publish(&path(), &workspace, &["module.cubin".to_owned()])
        .unwrap();
    assert_eq!(
        fs::read(elsewhere.join("module.cubin")).unwrap(),
        b"shared cubin"
    );
    let built = temporary.path().join("built").join(path().relative());
    assert_eq!(
        fs::read(built.join("module.cubin")).unwrap(),
        b"built cubin"
    );
}

#[test]
fn a_committed_module_is_never_replaced() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(temporary.path());
    let first = workspace(temporary.path(), &[("module.cubin", b"first")]);
    store
        .publish(&path(), &first, &["module.cubin".to_owned()])
        .unwrap();

    let workspace = workspace(temporary.path(), &[("module.cubin", b"second")]);
    let replaced = store.publish(&path(), &workspace, &["module.cubin".to_owned()]);
    assert!(replaced.is_err());
    let files = store.files(&path()).unwrap().unwrap();
    assert_eq!(fs::read(&files[0].path).unwrap(), b"first");
}

#[test]
fn a_publish_that_died_before_the_mark_is_retried_over() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(temporary.path());
    let abandoned = temporary.path().join("store").join(path().relative());
    fs::create_dir_all(&abandoned).unwrap();
    fs::write(abandoned.join("module.cubin"), b"torn").unwrap();

    let workspace = workspace(temporary.path(), &[("module.cubin", b"whole")]);
    store
        .publish(&path(), &workspace, &["module.cubin".to_owned()])
        .unwrap();
    let files = store.files(&path()).unwrap().unwrap();
    assert_eq!(fs::read(&files[0].path).unwrap(), b"whole");
}

#[test]
fn list_names_every_committed_module_of_every_root_once() {
    let temporary = tempfile::tempdir().unwrap();
    let store = store(temporary.path());
    let workspace = workspace(temporary.path(), &[("module.cubin", b"bytes")]);
    let first = path();
    let second =
        ModulePath::new("rms_norm", "triton", 4, "sm_100a", "dtype_bf16_hidden_2048").unwrap();
    for path in [&first, &second] {
        store
            .publish(path, &workspace, &["module.cubin".to_owned()])
            .unwrap();
    }

    let listed = store.list().unwrap();
    assert_eq!(listed, vec![first, second]);
}

#[test]
fn a_module_path_is_made_of_names_that_cannot_leave_their_root() {
    for name in ["", ".", "..", "a/b", "../sibling"] {
        let path = ModulePath::new("rms_norm", name, 1, "sm_100a", "dtype_bf16");
        assert!(path.is_err(), "{name:?} is not one name");
    }
    let path = ModulePath::new("rms_norm", "triton", 1, "sm_100a", "dtype_bf16").unwrap();
    assert_eq!(path.to_string(), "rms_norm/triton/1/sm_100a/dtype_bf16");
}

#[test]
fn an_empty_store_root_set_is_refused() {
    assert!(matches!(Store::open(Vec::new()), Err(Error::Io(_))));
}

#[test]
fn a_store_with_two_writable_roots_is_refused() {
    let temporary = tempfile::tempdir().unwrap();
    let refused = Store::open(vec![
        Root {
            path: temporary.path().join("one"),
            writable: true,
        },
        Root {
            path: temporary.path().join("two"),
            writable: true,
        },
    ]);
    assert!(matches!(refused, Err(Error::Io(_))));
}
