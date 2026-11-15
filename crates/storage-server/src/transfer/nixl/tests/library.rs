//! Loading NIXL's C API library.

/// The library the NIXL package installs on the loader's default search path provides every
/// entry point the crate calls: a load names the first one it lacks.
#[test]
#[ignore = "requires the NIXL package's C API on the loader's default search path"]
fn the_installed_library_provides_every_entry_point_the_crate_calls() {
    if let Err(error) = super::super::library::api() {
        panic!("{error}");
    }
}
