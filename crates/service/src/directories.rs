//! The calling user's data and cache directories, as the XDG Base Directory
//! Specification states them: the one place a host service of this executable
//! finds where it keeps what it writes.

use std::io;
use std::path::PathBuf;

/// The directory of this user's UniServe data: `$XDG_DATA_HOME/uniserve`, or
/// `$HOME/.local/share/uniserve` when `XDG_DATA_HOME` is unset or empty.
///
/// # Errors
///
/// Fails when neither variable states a directory.
pub fn data_directory() -> io::Result<PathBuf> {
    base("XDG_DATA_HOME", ".local/share")
}

/// The directory of this user's UniServe caches: `$XDG_CACHE_HOME/uniserve`,
/// or `$HOME/.cache/uniserve` when `XDG_CACHE_HOME` is unset or empty.
///
/// # Errors
///
/// Fails when neither variable states a directory.
pub fn cache_directory() -> io::Result<PathBuf> {
    base("XDG_CACHE_HOME", ".cache")
}

/// `$variable/uniserve`, or `$HOME/<fallback>/uniserve` as the specification
/// defaults it. A relative value is ignored, as the specification requires.
fn base(variable: &str, fallback: &str) -> io::Result<PathBuf> {
    let stated = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    let root = match stated(variable) {
        Some(root) => root,
        None => stated("HOME")
            .map(|home| home.join(fallback))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("neither {variable} nor HOME states an absolute directory"),
                )
            })?,
    };
    Ok(root.join("uniserve"))
}
