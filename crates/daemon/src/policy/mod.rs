//! Policy packages: the one source of a deployment's configuration (`specs/configuration.md`
//! §3).
//!
//! A package is a directory: `config.toml`, the deployment's [`Config`], and `programs/`, the
//! sources of the WebAssembly programs its configuration lists. `uniserve serve <package>` reads
//! one from a directory ([`Package::read`]); `PUT /admin/policy` carries one as a tar archive of
//! that directory ([`Package::unpack`]) and `GET /admin/policy` answers with one
//! ([`Package::archive`]). Every converged package is a revision of the deployment's persistent
//! state ([`revision`]).

pub mod revision;

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context as _, ensure};

use crate::config::{Config, Stated};

/// The file of a package that holds its configuration.
pub const CONFIG: &str = "config.toml";

/// The directory of a package that holds its programs' sources.
pub const PROGRAMS: &str = "programs";

/// A policy package: a configuration and the sources of its programs.
#[derive(Debug, Clone, PartialEq)]
pub struct Package {
    /// The configuration its `config.toml` states, every unstated option at its default.
    pub config: Config,
    /// The option paths its `config.toml` states.
    pub stated: Stated,
    /// The files under `programs/`, by their path relative to that directory, with their
    /// contents.
    pub programs: BTreeMap<PathBuf, Vec<u8>>,
}

impl Package {
    /// The package of `config`'s options and no program: what a writer that changes options
    /// alone, such as a control program, produces.
    ///
    /// # Errors
    ///
    /// Fails when the configuration has no TOML form.
    pub fn of(config: Config, programs: BTreeMap<PathBuf, Vec<u8>>) -> anyhow::Result<Self> {
        // The stated paths are the printed file's: what a reader of the package's `config.toml`
        // finds stated, which is what a start from that file would read.
        let (config, stated) = crate::config::parse(&crate::config::printed(&config)?)?;
        Ok(Self {
            config,
            stated,
            programs,
        })
    }

    /// The package `text`, a `config.toml`, and `programs` make.
    ///
    /// # Errors
    ///
    /// Fails when `text` is no configuration ([`crate::config::parse`]).
    fn from_parts(text: &str, programs: BTreeMap<PathBuf, Vec<u8>>) -> anyhow::Result<Self> {
        let (config, stated) = crate::config::parse(text)?;
        Ok(Self {
            config,
            stated,
            programs,
        })
    }

    /// Reads the package in `directory`: its `config.toml` and every file under its `programs/`.
    ///
    /// # Errors
    ///
    /// Fails when `config.toml` is missing or no configuration, or a program file cannot be
    /// read.
    pub fn read(directory: &Path) -> anyhow::Result<Self> {
        let file = directory.join(CONFIG);
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("reading the package's {}", file.display()))?;
        let mut programs = BTreeMap::new();
        let root = directory.join(PROGRAMS);
        if root.is_dir() {
            collect(&root, &root, &mut programs)?;
        }
        Self::from_parts(&text, programs).with_context(|| format!("in {}", file.display()))
    }

    /// The package a tar archive of its directory holds, as `PUT /admin/policy` carries it.
    ///
    /// # Errors
    ///
    /// Fails when the archive does not read, holds no `config.toml`, or holds a file outside
    /// `config.toml` and `programs/`, or a path that leaves the package.
    pub fn unpack(archive: &[u8]) -> anyhow::Result<Self> {
        let mut text = None;
        let mut programs = BTreeMap::new();
        let mut entries = tar::Archive::new(archive);
        for entry in entries.entries().context("reading the package archive")? {
            let mut entry = entry.context("reading the package archive")?;
            let kind = entry.header().entry_type();
            let path = entry
                .path()
                .context("reading a path of the package archive")?
                .into_owned();
            let path = relative(&path)?;
            if kind.is_dir() {
                continue;
            }
            ensure!(
                kind.is_file(),
                "the package archive's {} is neither a file nor a directory",
                path.display()
            );
            let mut contents = Vec::new();
            entry
                .read_to_end(&mut contents)
                .with_context(|| format!("reading the package archive's {}", path.display()))?;
            if path == Path::new(CONFIG) {
                text = Some(String::from_utf8(contents).context("the package's config.toml")?);
            } else if let Ok(program) = path.strip_prefix(PROGRAMS) {
                programs.insert(program.to_owned(), contents);
            } else {
                anyhow::bail!(
                    "the package archive holds {}: a package holds `{CONFIG}` and `{PROGRAMS}/` \
                     alone",
                    path.display()
                );
            }
        }
        let text = text.with_context(|| format!("the package archive holds no `{CONFIG}`"))?;
        Self::from_parts(&text, programs)
    }

    /// The package's `config.toml`: its configuration's non-default options.
    ///
    /// # Errors
    ///
    /// Fails when the configuration has no TOML form.
    pub fn text(&self) -> anyhow::Result<String> {
        crate::config::printed(&self.config)
    }

    /// The package as a tar archive of its directory: `config.toml`, then `programs/`.
    ///
    /// # Errors
    ///
    /// Fails when the configuration has no TOML form.
    pub fn archive(&self) -> anyhow::Result<Vec<u8>> {
        let mut builder = tar::Builder::new(Vec::new());
        let mut append = |path: &Path, contents: &[u8]| -> anyhow::Result<()> {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_entry_type(tar::EntryType::Regular);
            builder
                .append_data(&mut header, path, contents)
                .with_context(|| format!("archiving {}", path.display()))
        };
        append(Path::new(CONFIG), self.text()?.as_bytes())?;
        for (path, contents) in &self.programs {
            append(&Path::new(PROGRAMS).join(path), contents)?;
        }
        builder
            .into_inner()
            .context("finishing the package archive")
    }

    /// Writes the package into the empty or missing directory `directory`, every file synced
    /// to its storage before this returns.
    ///
    /// # Errors
    ///
    /// Fails when a file or directory cannot be written or synced.
    pub fn write(&self, directory: &Path) -> anyhow::Result<()> {
        let mut files = vec![(PathBuf::from(CONFIG), self.text()?.into_bytes())];
        files.extend(
            self.programs
                .iter()
                .map(|(path, contents)| (Path::new(PROGRAMS).join(path), contents.clone())),
        );
        for (path, contents) in files {
            revision::write_synced(&directory.join(path), &contents)?;
        }
        Ok(())
    }
}

/// Collects every file under `directory` into `files`, by its path relative to `root`.
fn collect(
    root: &Path,
    directory: &Path,
    files: &mut BTreeMap<PathBuf, Vec<u8>>,
) -> anyhow::Result<()> {
    let entries = std::fs::read_dir(directory)
        .with_context(|| format!("reading the package's {}", directory.display()))?;
    for entry in entries {
        let path = entry
            .with_context(|| format!("reading the package's {}", directory.display()))?
            .path();
        if path.is_dir() {
            collect(root, &path, files)?;
        } else {
            let contents = std::fs::read(&path)
                .with_context(|| format!("reading the package's {}", path.display()))?;
            let relative = path
                .strip_prefix(root)
                .with_context(|| format!("{} lies outside the package", path.display()))?;
            files.insert(relative.to_owned(), contents);
        }
    }
    Ok(())
}

/// `path` as a path inside the package: relative, without a parent component, and with a
/// leading `./` removed, as `tar` of a directory writes it.
fn relative(path: &Path) -> anyhow::Result<PathBuf> {
    let mut inside = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => inside.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!(
                    "the package archive's {} leaves the package's directory",
                    path.display()
                )
            }
        }
    }
    Ok(inside)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "model = \"org/model\"\nmax_running_requests = 8\n";

    /// A package read from its directory and the same package carried as an archive are one
    /// package, and its archive reads back as itself.
    #[test]
    fn a_directory_and_its_archive_are_one_package() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join(CONFIG), TEXT).unwrap();
        let program = directory.path().join(PROGRAMS).join("chunk/src");
        std::fs::create_dir_all(&program).unwrap();
        std::fs::write(program.join("lib.rs"), "fn main() {}").unwrap();

        let read = Package::read(directory.path()).unwrap();
        assert_eq!(read.config.max_running_requests.get(), 8);
        assert_eq!(
            read.programs.keys().collect::<Vec<_>>(),
            [Path::new("chunk/src/lib.rs")]
        );
        assert!(read.stated.contains("max_running_requests"));

        let unpacked = Package::unpack(&read.archive().unwrap()).unwrap();
        assert_eq!(unpacked, read);

        let copy = tempfile::tempdir().unwrap();
        read.write(copy.path()).unwrap();
        assert_eq!(Package::read(copy.path()).unwrap(), read);
    }

    /// An archive `tar` made of the package's directory, with its leading `./`, reads as the
    /// package; one without `config.toml`, with a path that leaves the package or with a file
    /// outside its two parts is refused.
    #[test]
    fn an_archive_holds_a_package_alone() {
        let archive = |files: &[(&str, &str)]| {
            let mut builder = tar::Builder::new(Vec::new());
            for (path, contents) in files {
                let mut header = tar::Header::new_gnu();
                header.set_size(contents.len() as u64);
                header.set_mode(0o644);
                header.set_entry_type(tar::EntryType::Regular);
                // Written raw, so a path `append_data` would refuse reaches the reader.
                header.as_gnu_mut().unwrap().name[..path.len()].copy_from_slice(path.as_bytes());
                header.set_cksum();
                builder.append(&header, contents.as_bytes()).unwrap();
            }
            builder.into_inner().unwrap()
        };
        let package = Package::unpack(&archive(&[("./config.toml", TEXT)])).unwrap();
        assert_eq!(package.config.model.as_deref(), Some("org/model"));

        for (files, named) in [
            (vec![("programs/a.rs", "")], "no `config.toml`"),
            (
                vec![("config.toml", TEXT), ("../escape", "")],
                "leaves the package",
            ),
            (vec![("config.toml", TEXT), ("notes.txt", "")], "notes.txt"),
            (vec![("config.toml", "max_batch = 1\n")], "max_batch"),
        ] {
            let error = Package::unpack(&archive(&files)).unwrap_err();
            assert!(format!("{error:#}").contains(named), "{files:?}: {error:#}");
        }
    }
}
