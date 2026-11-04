//! A checkpoint: its safetensors shards, the tensors they hold, and the identity of every file.
//!
//! The loader never converts a checkpoint's weights in place and never holds a whole shard in host
//! memory: each shard is mapped read-only with `memmap2`, [`read_header`] reads its header — eight
//! bytes of length, the JSON body, and each tensor's `{dtype, shape, data_offsets}` — and one
//! tensor's bytes are copied when a weight is converted.
//!
//! The header reader is ours and not a crate's because the dtype set is the **format's**, while a
//! crate's `Dtype` enum is a subset of it: a calibrated NVFP4 export stores its block scales as
//! `F8_E4M3`, and a reader that cannot name that cannot open the model. Every dtype name our
//! operators speak is mapped onto an operator ABI [`DType`], and any other name is refused at open,
//! with the name and the tensor in the message, rather than reach a launch that would misread it. A shard is mapped for the life of
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
    /// Every root this checkpoint's tensors came from, in precedence order; one entry unless it was
    /// opened over several roots ([`Checkpoint::open_roots`]). A missing tensor names them, because
    /// a union that spans an export and a base is where "not found" costs the most to chase.
    roots: Vec<PathBuf>,
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
            let (header, entries) = read_header(&path, &map)?;
            for (name, entry) in entries {
                if locations
                    .insert(
                        name.clone(),
                        TensorLocation {
                            file: file_index,
                            tensor: name.clone(),
                            dtype: entry.dtype,
                            shape: entry.shape,
                            bytes: entry.offsets,
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
            roots: vec![root.to_path_buf()],
            files,
            tensors: locations,
            components: BTreeMap::new(),
        })
    }

    /// Open `roots` as **one** checkpoint: the first root's tensors shadow the later ones'.
    ///
    /// A deployment that ships a delta export over a base is one checkpoint whose file set spans
    /// both, so the union is per tensor and not per component: components are enumerated in each
    /// root as [`Checkpoint::open`] already does; a component present in several roots is one
    /// unioned component whose tensors come from the earliest root that holds each name; a
    /// component that only a later root holds is that root's; and the same shadowing applies at the
    /// top level, where an export's tensor replaces the base's.
    ///
    /// A shadowed tensor is expected and is **not** a byte comparison: a delta export legitimately
    /// differs from the base it was trained from, and refusing the difference would refuse the
    /// deployment. Two roots of *one* root's shard set disagreeing is still an error, because that
    /// is a malformed checkpoint and no precedence can say which is meant.
    ///
    /// [`Checkpoint::root`] answers the first root: the export's, which is the one a log names.
    pub fn open_roots(roots: &[&Path]) -> Result<Self> {
        let mut roots = roots.iter();
        let first = roots.next().ok_or_else(|| Error::Checkpoint {
            path: PathBuf::new(),
            problem: "a checkpoint needs at least one root".to_owned(),
        })?;
        let mut held = Self::open(first)?;
        for root in roots {
            held = held.merge(Self::open(root)?);
        }
        Ok(held)
    }

    /// The union of `self` (earlier, shadowing) and `later`, as [`Checkpoint::open_roots`] defines
    /// it. File indices of the later root are rebased onto the union's file list, so a tensor
    /// location keeps naming the shard it was read from.
    fn merge(mut self, later: Self) -> Self {
        let base = self.files.len();
        self.roots.extend(later.roots);
        for (name, mut location) in later.tensors {
            // The earlier root wins: a name the union already holds keeps its own bytes.
            if self.tensors.contains_key(&name) {
                continue;
            }
            location.file += base;
            self.tensors.insert(name, location);
        }
        self.files.extend(later.files);
        for (name, component) in later.components {
            let union = match self.components.remove(&name) {
                Some(held) => held.merge(component),
                None => component,
            };
            self.components.insert(name, union);
        }
        self
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
            roots: vec![root.to_path_buf()],
            files: Vec::new(),
            tensors: BTreeMap::new(),
            components,
        })
    }

    /// The directory this checkpoint was opened with — and, for a union of roots, the **first** of
    /// them: the export's, which is the one a log names.
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
        let location = self.tensors.get(name).ok_or_else(|| {
            if self.roots.len() == 1 {
                return Error::MissingTensor(name.to_owned());
            }
            // A union that spans several roots is where a bare "not found" is least useful: the
            // caller needs to know what was searched, and in which order precedence ran.
            let tried = self
                .roots
                .iter()
                .map(|root| root.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Error::Checkpoint {
                path: self.root.clone(),
                problem: format!(
                    "no root holds a tensor `{name}`; searched, in precedence order: {tried}"
                ),
            }
        })?;
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

/// One tensor as a shard's header states it.
struct HeaderTensor {
    dtype: DType,
    shape: Shape,
    /// The byte range of the tensor inside the payload, which starts after the header.
    offsets: (u64, u64),
}

/// One header entry as the format states it.
#[derive(serde::Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

/// Read one shard's header: the byte length of the header (its JSON body plus the eight-byte
/// length that precedes it) and every tensor it names.
///
/// The format is eight little-endian bytes of JSON length, that many bytes of JSON, then the
/// payload. The JSON is an object of tensor name to `{dtype, shape, data_offsets}`, with an optional
/// `__metadata__` entry that carries no tensor — accepted and ignored, as the format says.
///
/// **This is the crate's job done here**, deliberately: `safetensors` 0.3.3's `Dtype` has no FP8 or
/// FP4 variant, so a shard carrying `F8_E4M3` — which a calibrated NVFP4 checkpoint's block scales
/// are — fails in the crate's own deserialization, before any lookup. The dtype set is a property of
/// the format, and the format's names are mapped here onto what our operators speak (specs/worker.md
/// 9, `DType`); a name outside that set is refused with the name in the message rather than read as
/// something it is not.
fn read_header(path: &Path, bytes: &[u8]) -> Result<(u64, Vec<(String, HeaderTensor)>)> {
    let length = bytes.get(..8).ok_or_else(|| Error::Checkpoint {
        path: path.to_path_buf(),
        problem: "a shard begins with eight bytes of header length".to_owned(),
    })?;
    let mut word = [0_u8; 8];
    word.copy_from_slice(length);
    let json_length = u64::from_le_bytes(word);
    let header = json_length
        .checked_add(8)
        .filter(|end| *end <= bytes.len() as u64)
        .ok_or_else(|| Error::Checkpoint {
            path: path.to_path_buf(),
            problem: format!(
                "a {json_length}-byte header does not fit a {}-byte shard",
                bytes.len()
            ),
        })?;
    let payload = bytes.len() as u64 - header;
    let body =
        std::str::from_utf8(&bytes[8..header as usize]).map_err(|error| Error::Checkpoint {
            path: path.to_path_buf(),
            problem: format!("the header is not UTF-8: {error}"),
        })?;
    let parsed: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(body).map_err(|error| Error::Checkpoint {
            path: path.to_path_buf(),
            problem: format!("the header is not valid JSON: {error}"),
        })?;

    let mut tensors = Vec::with_capacity(parsed.len());
    for (name, value) in parsed {
        // The format's optional metadata block is an object of strings, not a tensor.
        if name == "__metadata__" {
            continue;
        }
        let raw: RawTensor = serde_json::from_value(value).map_err(|error| Error::Checkpoint {
            path: path.to_path_buf(),
            problem: format!("tensor `{name}` is not {{dtype, shape, data_offsets}}: {error}"),
        })?;
        let dtype = dtype_of(path, &name, &raw.dtype)?;
        let shape = shape_of(path, &name, &raw.shape)?;
        let (start, end) = raw.data_offsets;
        if start > end || end > payload {
            return Err(Error::Checkpoint {
                path: path.to_path_buf(),
                problem: format!(
                    "tensor `{name}` claims bytes {start}..{end} of a {payload}-byte payload"
                ),
            });
        }
        tensors.push((
            name,
            HeaderTensor {
                dtype,
                shape,
                offsets: (start, end),
            },
        ));
    }
    Ok((header, tensors))
}

/// The operator ABI element type a safetensors dtype **name** states.
///
/// The whole set our operators speak is accepted, FP8 and FP4 included; a name that is not in the
/// set — the format's `BOOL`, `I8`, `U16`, F6 and F8_E5M2 among them, and any future one — is
/// refused by name here, at open, rather than reaching a launch that would misread its bytes.
fn dtype_of(path: &Path, tensor: &str, name: &str) -> Result<DType> {
    Ok(match name {
        "U8" => DType::U8,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "F8_E4M3" => DType::F8E4M3,
        "F4_E2M1" | "F4" => DType::F4E2M1,
        other => {
            return Err(Error::Checkpoint {
                path: path.to_path_buf(),
                problem: format!(
                    "tensor `{tensor}` is `{other}`, which the operator ABI has no type for"
                ),
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
