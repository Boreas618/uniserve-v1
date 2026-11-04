//! A checkpoint: its safetensors shards, the tensors they hold, and the identity of every file.
//!
//! The loader never converts a checkpoint's weights in place and never holds a whole shard in host
//! memory: each shard is mapped read-only with `memmap2`, the safetensors crate reads its header,
//! and one tensor's bytes are copied when a weight is converted. A shard is mapped for the life of
//! the [`Checkpoint`], which is the model's life; the mapping costs virtual address space, not
//! resident memory.
//!
//! ## File identity
//!
//! Retained weights are keyed by the identity of the files they were read from (specs/worker.md 8),
//! so a checkpoint edited after it was loaded cannot silently serve the old retained bytes. The rule
//! is the one `uniserve/loading/retained.py` implements:
//!
//! - a file of a Hugging Face cache is a link to a blob named by a digest of its contents (SHA-256
//!   for a large file, the git blob SHA-1 otherwise), and the cache writes a blob once, so the blob's
//!   name is the file's identity;
//! - any other file's identity is its device, inode, modification time and status-change time;
//! - a file whose status-change time is younger than one second may still be written again under the
//!   same clock tick, so its identity is unique to this read and matches no later load.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use ops::{DType, Shape};

use crate::error::{Error, Result};

/// How long after its last write a file's identity is trusted to stay the same.
///
/// The kernel's clock tick is coarser than this on some filesystems, so a younger file may be
/// rewritten under the same `ctime`; such a file gets an identity unique to this read.
const SETTLED_NS: u128 = 1_000_000_000;

/// The identity of one checkpoint file, as retained weights are keyed by it.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct FileIdentity(String);

impl FileIdentity {
    /// The identity of `path`, a link followed.
    pub fn of(path: &Path) -> Result<Self> {
        let resolved = std::fs::canonicalize(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if is_hf_blob(&resolved) {
            let name = resolved.file_name().unwrap_or_default().to_string_lossy();
            return Ok(Self(format!("blob:{name}")));
        }
        let status = std::fs::metadata(&resolved).map_err(|source| Error::Io {
            path: resolved.clone(),
            source,
        })?;
        let mtime = status.mtime() as i128 * 1_000_000_000 + i128::from(status.mtime_nsec());
        let ctime = status.ctime() as i128 * 1_000_000_000 + i128::from(status.ctime_nsec());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|age| age.as_nanos() as i128)
            .unwrap_or(0);
        if now - ctime < SETTLED_NS as i128 {
            return Ok(Self::unsettled(&resolved));
        }
        Ok(Self(format!(
            "{}:{}:{}:{}",
            status.dev(),
            status.ino(),
            mtime,
            ctime
        )))
    }

    /// An identity that matches no later load of `path`, for a file whose write may not be settled.
    fn unsettled(path: &Path) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        // The reading process and the moment of the read together name one read of one path; the
        // counter separates two reads that land in the same nanosecond.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|age| age.as_nanos())
            .unwrap_or(0);
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in path.as_os_str().as_encoded_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self(format!(
            "unsettled:{:016x}{:08x}{:016x}",
            nanos as u64,
            std::process::id(),
            hash ^ count
        ))
    }

    /// The identity as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FileIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Whether `path` is a blob of a Hugging Face cache: a `blobs` directory entry named by a digest.
fn is_hf_blob(path: &Path) -> bool {
    let in_blobs = path
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|parent| parent == "blobs");
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let hex = !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    in_blobs && hex && (name.len() == 64 || name.len() == 40)
}

/// One mapped safetensors shard.
pub struct CheckpointFile {
    path: PathBuf,
    identity: FileIdentity,
    /// Bytes of the file before the tensor payload.
    header: u64,
    map: Mmap,
}

impl CheckpointFile {
    /// The shard's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The shard's identity.
    pub fn identity(&self) -> &FileIdentity {
        &self.identity
    }
}

/// Where one tensor lives: which shard, which element type and extents, and which bytes.
#[derive(Clone, Debug)]
pub struct TensorLocation {
    /// Index into [`Checkpoint::files`].
    pub file: usize,
    /// The tensor's name in the checkpoint.
    pub tensor: String,
    /// Its element type.
    pub dtype: DType,
    /// Its extents.
    pub shape: Shape,
    /// Its bytes within the shard's payload, as a half-open range.
    pub bytes: (u64, u64),
}

/// One tensor's bytes, read on the host.
#[derive(Clone, Debug)]
pub struct HostTensor {
    /// The element type in the checkpoint.
    pub dtype: DType,
    /// The extents in the checkpoint.
    pub shape: Shape,
    /// The tensor's bytes, row-major.
    pub bytes: Vec<u8>,
}

/// A checkpoint: its mapped shards and the tensors they hold, or a pipeline's components.
///
/// A flat checkpoint is one shard set; a *pipeline* root holds one subdirectory per component
/// (`transformer/`, `vae/`, `text_encoder/`, …), each with its own index and therefore its own file
/// identities, which §8's retained keys name. Opening a root discovers which of the two it is: a
/// root with a shard set of its own is flat and behaves as it always has, and a root without one is
/// read as the components its subdirectories hold.
pub struct Checkpoint {
    root: PathBuf,
    files: Vec<CheckpointFile>,
    tensors: BTreeMap<String, TensorLocation>,
    components: BTreeMap<String, Checkpoint>,
}

impl Checkpoint {
    /// Open the checkpoint at `root`.
    ///
    /// A `model.safetensors.index.json` names each tensor's shard; without one every
    /// `*.safetensors` file of the directory is read and its tensors are taken from its header.
    pub fn open(root: &Path) -> Result<Self> {
        let index = root.join("model.safetensors.index.json");
        let paths = if index.is_file() {
            shards_from_index(&index)?
        } else if let Some(paths) = shards_in(root)? {
            paths
        } else {
            return Self::pipeline(root);
        };

        let mut files = Vec::with_capacity(paths.len());
        let mut locations = BTreeMap::new();
        for path in paths {
            let file_index = files.len();
            let file = File::open(&path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            // SAFETY: the mapping is read-only and the file is not written through it; another
            // process truncating the checkpoint while it is mapped is the operator's error, and
            // the loading boundary reports it as such rather than reading past the mapping.
            let map = unsafe { Mmap::map(&file) }.map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            let identity = FileIdentity::of(&path)?;
            let (header, metadata) =
                safetensors::SafeTensors::read_metadata(&map).map_err(|source| {
                    Error::Safetensors {
                        path: path.clone(),
                        source,
                    }
                })?;
            // `read_metadata` answers the length of the JSON header alone; the payload starts
            // after the 8-byte little-endian length that precedes it.
            let header = u64::try_from(header + 8).map_err(|_| Error::Checkpoint {
                path: path.clone(),
                problem: "the header is larger than u64::MAX bytes".to_owned(),
            })?;
            for (name, info) in metadata.tensors() {
                let dtype = dtype_of(path.as_path(), info.dtype)?;
                let shape = shape_of(path.as_path(), &name, &info.shape)?;
                let (start, end) = info.data_offsets;
                if locations
                    .insert(
                        name.clone(),
                        TensorLocation {
                            file: file_index,
                            tensor: name,
                            dtype,
                            shape,
                            bytes: (start as u64, end as u64),
                        },
                    )
                    .is_some()
                {
                    return Err(Error::Checkpoint {
                        path: path.clone(),
                        problem: "two shards hold a tensor of the same name".to_owned(),
                    });
                }
            }
            files.push(CheckpointFile {
                path,
                identity,
                header,
                map,
            });
        }

        Ok(Self {
            root: root.to_path_buf(),
            files,
            tensors: locations,
            components: BTreeMap::new(),
        })
    }

    /// Open a pipeline root: one component per subdirectory that holds a shard set.
    fn pipeline(root: &Path) -> Result<Self> {
        let entries = std::fs::read_dir(root).map_err(|source| Error::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let mut components = BTreeMap::new();
        for entry in entries {
            let entry = entry.map_err(|source| Error::Io {
                path: root.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if shards_in(&path)?.is_none() {
                continue;
            }
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            components.insert(name, Self::open(&path)?);
        }
        if components.is_empty() {
            return Err(Error::Checkpoint {
                path: root.to_path_buf(),
                problem: "no shard set here and none in a component below it".to_owned(),
            });
        }
        Ok(Self {
            root: root.to_path_buf(),
            files: Vec::new(),
            tensors: BTreeMap::new(),
            components,
        })
    }

    /// The directory this checkpoint was opened with.
    ///
    /// A pipeline's components are opened under it, so a model's `load()` can reach each component
    /// without a second copy of the path being threaded through the engine.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether this checkpoint is a pipeline of components rather than one model's shards.
    pub fn is_pipeline(&self) -> bool {
        !self.components.is_empty()
    }

    /// The component numbered `name`, with its own shards and file identities.
    pub fn component(&self, name: &str) -> Result<&Checkpoint> {
        self.components.get(name).ok_or_else(|| Error::Checkpoint {
            path: PathBuf::from(name),
            problem: format!(
                "this checkpoint has no component `{name}`; it holds {}",
                self.components
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
    }

    /// The names of the components, in order.
    pub fn components(&self) -> impl Iterator<Item = &str> {
        self.components.keys().map(String::as_str)
    }

    /// Every element type the checkpoint or one of its components holds, each once.
    pub fn dtypes(&self) -> Vec<DType> {
        let mut dtypes: Vec<DType> = Vec::new();
        for dtype in self
            .tensors
            .values()
            .map(|location| location.dtype)
            .chain(self.components.values().flat_map(Checkpoint::dtypes))
        {
            if !dtypes.contains(&dtype) {
                dtypes.push(dtype);
            }
        }
        dtypes
    }

    /// The checkpoint's shards.
    pub fn files(&self) -> &[CheckpointFile] {
        &self.files
    }

    /// The location of the tensor named `name`.
    pub fn tensor(&self, name: &str) -> Option<&TensorLocation> {
        self.tensors.get(name)
    }

    /// Every tensor's name, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// How many tensors the checkpoint holds.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint holds no tensor.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The identity of the file that holds `tensor`.
    pub fn identity(&self, tensor: &str) -> Option<&FileIdentity> {
        let location = self.tensors.get(tensor)?;
        Some(self.files[location.file].identity())
    }

    /// Read the tensor named `name` into host memory.
    pub fn read(&self, name: &str) -> Result<HostTensor> {
        let location = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        Ok(HostTensor {
            dtype: location.dtype,
            shape: location.shape,
            bytes: self.bytes(location)?.to_vec(),
        })
    }

    /// Borrow the bytes of `location` inside its mapped shard.
    pub fn bytes(&self, location: &TensorLocation) -> Result<&[u8]> {
        let file = &self.files[location.file];
        let start = file
            .header
            .checked_add(location.bytes.0)
            .ok_or_else(|| Error::Checkpoint {
                path: file.path.clone(),
                problem: format!("tensor `{}` starts past u64::MAX", location.tensor),
            })?;
        let end = file
            .header
            .checked_add(location.bytes.1)
            .ok_or_else(|| Error::Checkpoint {
                path: file.path.clone(),
                problem: format!("tensor `{}` ends past u64::MAX", location.tensor),
            })?;
        let range = usize::try_from(start)
            .ok()
            .zip(usize::try_from(end).ok())
            .and_then(|(start, end)| file.map.get(start..end));
        range.ok_or_else(|| Error::Checkpoint {
            path: file.path.clone(),
            problem: format!(
                "tensor `{}` claims bytes {start}..{end} of a {}-byte shard",
                location.tensor,
                file.map.len()
            ),
        })
    }
}

/// The shard paths an index file names, in order of first mention.
fn shards_from_index(index: &Path) -> Result<Vec<PathBuf>> {
    let text = std::fs::read_to_string(index).map_err(|source| Error::Io {
        path: index.to_path_buf(),
        source,
    })?;
    let parsed: IndexFile = serde_json::from_str(&text).map_err(|error| Error::Checkpoint {
        path: index.to_path_buf(),
        problem: format!("the shard index is not valid JSON: {error}"),
    })?;
    let root = index.parent().unwrap_or(Path::new("."));
    let mut seen = std::collections::BTreeSet::new();
    let mut paths = Vec::new();
    for shard in parsed.weight_map.into_values() {
        if seen.insert(shard.clone()) {
            paths.push(root.join(shard));
        }
    }
    if paths.is_empty() {
        return Err(Error::Checkpoint {
            path: index.to_path_buf(),
            problem: "the shard index names no file".to_owned(),
        });
    }
    Ok(paths)
}

/// The `*.safetensors` shards of a directory in name order, or `None` when it holds none.
///
/// The index, when the directory has one, is what names the shards; a directory with neither is not
/// a shard set at all, which a pipeline root's discovery uses to tell components from other files.
fn shards_in(root: &Path) -> Result<Option<Vec<PathBuf>>> {
    let index = root.join("model.safetensors.index.json");
    if index.is_file() {
        return shards_from_index(&index).map(Some);
    }
    let mut paths = Vec::new();
    let entries = std::fs::read_dir(root).map_err(|source| Error::Io {
        path: root.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok((!paths.is_empty()).then_some(paths))
}

/// The index file's shape: the metadata block is not read.
#[derive(serde::Deserialize)]
struct IndexFile {
    weight_map: BTreeMap<String, String>,
}

/// The operator ABI element type of a safetensors dtype.
fn dtype_of(path: &Path, dtype: safetensors::Dtype) -> Result<DType> {
    use safetensors::Dtype as Safe;
    Ok(match dtype {
        Safe::U8 => DType::U8,
        Safe::I32 => DType::I32,
        Safe::I64 => DType::I64,
        Safe::F16 => DType::F16,
        Safe::BF16 => DType::BF16,
        Safe::F32 => DType::F32,
        other => {
            return Err(Error::Checkpoint {
                path: path.to_path_buf(),
                problem: format!("element type {other:?} has no operator ABI type yet"),
            });
        }
    })
}

/// A shape the operator ABI can hold.
fn shape_of(path: &Path, tensor: &str, dims: &[usize]) -> Result<Shape> {
    let mut extents = [0_u64; ops::MAX_RANK];
    if dims.len() > ops::MAX_RANK {
        return Err(Error::Checkpoint {
            path: path.to_path_buf(),
            problem: format!(
                "tensor `{tensor}` has {} axes, more than the ABI's {MAX_RANK}",
                dims.len(),
                MAX_RANK = ops::MAX_RANK
            ),
        });
    }
    for (axis, &extent) in dims.iter().enumerate() {
        extents[axis] = extent as u64;
    }
    Ok(Shape::new(&extents[..dims.len()]))
}
