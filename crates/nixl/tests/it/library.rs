//! Loading NIXL's C API library.

use std::path::Path;

/// A library that cannot be opened is an error of the load, not an abort
/// of the process.
#[test]
fn a_missing_library_is_an_error() {
    let missing = Path::new("/nonexistent/libnixl_capi.so");
    assert!(matches!(
        nixl::load_library(missing),
        Err(nixl::Error::Load { .. })
    ));
}

/// The library `NIXL_CAPI_LIBRARY` names provides every entry point the
/// crate calls: a load names the first one it lacks.
#[test]
#[ignore = "requires NIXL's C API library (NIXL_CAPI_LIBRARY)"]
fn the_library_provides_every_entry_point_the_crate_calls() {
    let library =
        std::env::var_os("NIXL_CAPI_LIBRARY").expect("NIXL_CAPI_LIBRARY names libnixl_capi.so");
    if let Err(error) = nixl::load_library(Path::new(&library)) {
        panic!("{error}");
    }
}
