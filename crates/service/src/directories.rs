//! The calling user's data and cache directories, as the XDG Base Directory
//! Specification states them, and the directory of one deployment inside each of
//! UniServe's directories: the one place a deployment's servers find where they
//! keep what they write.

use std::io;
use std::path::{Path, PathBuf};

/// The directory inside each of UniServe's directories that holds one directory per
/// deployment, named after it.
const DEPLOYMENTS: &str = "deployments";

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

/// The directory of the deployment `name` inside `directory`, one of UniServe's own
/// (`deployments/<name>`): every file a deployment's servers keep there lies inside it, so no
/// other deployment's server reads or writes it. A deployment is named by its policy package's
/// directory, so the name is one path component.
///
/// # Errors
///
/// `InvalidInput` when `name` is empty, `.` or `..`, or holds a `/` or a NUL: it would name
/// another directory than the deployment's own.
pub fn deployment(directory: &Path, name: &str) -> io::Result<PathBuf> {
    let component = !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\0']);
    if !component {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name:?} names no deployment: a deployment's name is one path component"),
        ));
    }
    Ok(directory.join(DEPLOYMENTS).join(name))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A deployment's directory is its own inside the directory it is asked of, and a name that
    /// would lead elsewhere is refused.
    #[test]
    fn a_deployment_names_its_own_directory() {
        let root = Path::new("/data/uniserve");
        assert_eq!(
            deployment(root, "qwen3").unwrap(),
            Path::new("/data/uniserve/deployments/qwen3")
        );
        for name in ["", ".", "..", "a/b", "a\0b"] {
            let refused = deployment(root, name).unwrap_err();
            assert_eq!(refused.kind(), io::ErrorKind::InvalidInput, "{name:?}");
        }
    }
}
