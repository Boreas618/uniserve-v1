//! A point's result bundle, and the archive of the runs a new run supersedes.
//!
//! Every file is written whole under a temporary name in its directory, synced,
//! then renamed into place, so a reader sees an old file or the new one. JSON is
//! written with sorted keys, so equal payloads give identical files. Media
//! samples are content-addressed (the digest of their bytes and an extension),
//! so identical outputs of different requests share one file.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use crate::record::{DecodedImage, canonical};

/// One bundle's directory and its `samples/`.
pub struct Bundle {
    pub directory: PathBuf,
    samples: PathBuf,
}

/// A value with every object's keys sorted.
pub fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(entries) => {
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|key| (key.clone(), sorted(&entries[key])))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// Writes `data` to `path` atomically: a temporary file in the same directory,
/// synced, renamed into place.
fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let directory = path.parent().context("a file in a directory")?;
    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(
            ".{}.",
            path.file_name()
                .map(|name| name.to_string_lossy())
                .unwrap_or_default()
        ))
        .suffix(".tmp")
        .tempfile_in(directory)?;
    temporary.write_all(data)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| anyhow::anyhow!("writing {}: {}", path.display(), error.error))?;
    Ok(())
}

impl Bundle {
    /// Creates the bundle's directories.
    pub fn create(directory: &Path) -> Result<Self> {
        let samples = directory.join("samples");
        std::fs::create_dir_all(&samples)
            .with_context(|| format!("creating {}", samples.display()))?;
        Ok(Self {
            directory: directory.to_path_buf(),
            samples,
        })
    }

    /// A JSON file, indented, keys sorted.
    pub fn write_json(&self, name: &str, payload: &Value) -> Result<PathBuf> {
        let path = self.directory.join(name);
        let mut text = serde_json::to_string_pretty(&sorted(payload))?;
        text.push('\n');
        write_atomic(&path, text.as_bytes())?;
        Ok(path)
    }

    /// A JSON Lines file, one value a line, keys sorted.
    pub fn write_jsonl(&self, name: &str, payloads: &[Value]) -> Result<PathBuf> {
        let path = self.directory.join(name);
        let mut text = String::new();
        for payload in payloads {
            text.push_str(&serde_json::to_string(&sorted(payload))?);
            text.push('\n');
        }
        write_atomic(&path, text.as_bytes())?;
        Ok(path)
    }

    /// A text file.
    pub fn write_text(&self, name: &str, text: &str) -> Result<PathBuf> {
        let path = self.directory.join(name);
        write_atomic(&path, text.as_bytes())?;
        Ok(path)
    }

    /// An image sample, reused when the same bytes are already stored.
    ///
    /// # Errors
    ///
    /// When a sample of the same name holds other bytes.
    pub fn write_image(&self, image: &DecodedImage) -> Result<PathBuf> {
        let path = self.samples.join(&image.sample_filename);
        if path.exists() {
            ensure!(
                std::fs::read(&path)? == image.data,
                "content-addressed image sample collision"
            );
            return Ok(path);
        }
        write_atomic(&path, &image.data)?;
        Ok(path)
    }

    /// A response body kept as it arrived, even one inspection rejected, at a
    /// request-qualified path; different bytes are never written over it.
    pub fn write_original(path: &Path, data: &[u8]) -> Result<PathBuf> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if path.exists() {
            if std::fs::read(path)? != data {
                bail!("original media already exists: {}", path.display());
            }
            return Ok(path.to_path_buf());
        }
        write_atomic(path, data)?;
        Ok(path.to_path_buf())
    }
}

/// The directory under an output root that holds superseded runs, as
/// `superseded/<point>/<UTC time>/<role>`.
pub const SUPERSEDED: &str = "superseded";

/// Moves an earlier run's evidence of `point` out of its canonical paths:
/// every path of `evidence` (by the name each keeps in the archive) that holds
/// a file or a non-empty directory moves under
/// `<output_root>/superseded/<point>/<UTC time>`; `None` when none held any.
///
/// # Errors
///
/// When an archive of the same point and time exists.
pub fn supersede(
    output_root: &Path,
    point: &str,
    evidence: &[(String, PathBuf)],
) -> Result<Option<PathBuf>> {
    let held: Vec<&(String, PathBuf)> = evidence
        .iter()
        .filter(|(_, path)| {
            path.is_file()
                || std::fs::read_dir(path).is_ok_and(|mut entries| entries.next().is_some())
        })
        .collect();
    if held.is_empty() {
        return Ok(None);
    }
    // Microseconds keep two supersessions of one point apart.
    let now = time::OffsetDateTime::now_utc();
    let stamp = format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}.{:06}Z",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
        now.microsecond()
    );
    let archive = output_root.join(SUPERSEDED).join(point).join(stamp);
    ensure!(
        !archive.exists(),
        "the archive {} exists",
        archive.display()
    );
    std::fs::create_dir_all(&archive)?;
    for (name, path) in held {
        let destination = archive.join(name);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // A rename within a filesystem; across one, a copy and a removal.
        if std::fs::rename(path, &destination).is_err() {
            copy_tree(path, &destination)?;
            if path.is_dir() {
                std::fs::remove_dir_all(path)?;
            } else {
                std::fs::remove_file(path)?;
            }
        }
    }
    Ok(Some(archive))
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    if source.is_file() {
        std::fs::copy(source, destination)?;
        return Ok(());
    }
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        copy_tree(&entry.path(), &destination.join(entry.file_name()))?;
    }
    Ok(())
}

/// The canonical JSON digest form, for the summary's identities.
pub fn digest_text(value: &Value) -> String {
    canonical(&sorted(value))
}
