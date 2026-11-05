//! A checkpoint: its safetensors shards, the tensors they hold, and the identity of every file.
//!
//! The loader never converts a checkpoint's weights in place and never holds a whole shard in host
//! memory: each shard is mapped read-only with `memmap2`, the `safetensors` crate parses its header
//! through [`read_header`], and one tensor's bytes are copied when a weight is converted. A shard is
//! mapped for the life of the [`Checkpoint`], which is the model's life; the mapping costs virtual
//! address space, not resident memory.
//!
//! **The format is read by the format's own crate, and the element type is ours.** `safetensors`
//! parses the header, checks each tensor's byte range against its extents and element width, and
//! drops the optional metadata block; its `Dtype` now names every type this checkpoint set carries,
//! FP8 and FP4 included. What stays here is the map from that `Dtype` onto the operator ABI's
//! [`DType`], because the ABI names a subset: a format type it cannot express is refused at open
//! with the tensor and the type in the message, rather than reach a launch that would misread its
//! bytes. The match names every variant of the crate's enum, and the catch-all that its
//! `#[non_exhaustive]` makes the language require **refuses and prints the variant it saw**, so a
//! type a future release adds is a shard this loader will not open, never a number read as another.
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

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use ops::{DType, Shape};
use safetensors::Dtype as FormatDtype;
use safetensors::SafeTensors;

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

/// What a load asked a checkpoint for, and what it left alone.
///
/// Both lists are in name order, so a report can be diffed between two loads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reading {
    /// The tensors a `read` asked for.
    pub read: Vec<String>,
    /// The tensors of the same set that no `read` asked for.
    pub unread: Vec<String>,
}

impl Reading {
    /// Whether every tensor the checkpoint holds was asked for.
    pub fn is_complete(&self) -> bool {
        self.unread.is_empty()
    }
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
    /// The tensors a `read` has asked this checkpoint for, so [`Checkpoint::reading`] can answer the
    /// other half of the question: what the load never touched.
    read: Mutex<BTreeSet<String>>,
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
            read: Mutex::new(BTreeSet::new()),
        })
    }

    /// A pipeline whose components are the subdirectories of `root` and whose tensors are never read.
    ///
    /// It exists for a walk that states a model's extents without its bytes: every component handle
    /// resolves, so a `load` that chooses a component per weight runs unchanged, and a `read` refuses
    /// because no tensor is known. Unlike [`Checkpoint::open`], a component directory holding an index
    /// and no shards is a component here rather than a directory that is skipped — which is what a
    /// metadata-only deployment on a machine without the weights looks like.
    pub fn unopened(root: &Path) -> Result<Self> {
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
            let Some(name) = path.file_name().map(|name| name.to_string_lossy().into_owned()) else {
                continue;
            };
            components.insert(
                name,
                Self {
                    root: path.clone(),
                    roots: vec![path],
                    files: Vec::new(),
                    tensors: BTreeMap::new(),
                    components: BTreeMap::new(),
                    read: Mutex::new(BTreeSet::new()),
                },
            );
        }
        if components.is_empty() {
            return Err(Error::Checkpoint {
                path: root.to_path_buf(),
                problem: "no component directory below it".to_owned(),
            });
        }
        Ok(Self {
            root: root.to_path_buf(),
            roots: vec![root.to_path_buf()],
            files: Vec::new(),
            tensors: BTreeMap::new(),
            components,
            read: Mutex::new(BTreeSet::new()),
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
            read: Mutex::new(BTreeSet::new()),
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
        if self.files.is_empty() && self.tensors.is_empty() && !self.roots.is_empty() {
            // No shard set was ever opened for this handle: either an unopened pipeline or one of its
            // components. Say so, rather than reporting a name as absent from a table never filled.
            if self.components.is_empty() {
                return Err(Error::Checkpoint {
                    path: self.root.clone(),
                    problem: format!(
                        "this handle states a component's extents and reads no tensor (`{name}` was asked for)"
                    ),
                });
            }
        }
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
        if let Ok(mut read) = self.read.lock() {
            read.insert(name.to_owned());
        }
        Ok(HostTensor {
            dtype: location.dtype,
            shape: location.shape,
            bytes: self.bytes(location)?.to_vec(),
        })
    }

    /// What a load has asked this checkpoint for, and what it has left alone.
    ///
    /// A component of a multi-component export legitimately reads a subset of what its shards hold —
    /// a video VAE's per-layer `scale1`/`scale2`, a tied head, the calibration fields of a layer the
    /// model runs dense — so a non-empty [`Reading::unread`] is a **report and never a refusal**. It
    /// answers the one question a half-read export leaves open: when the numbers are wrong, which
    /// tensors did nobody ask for. It is device-free: it needs the claims the index states, not the
    /// tensors' values.
    ///
    /// A pipeline root holds no tensors of its own, so its answer is its components', each name
    /// qualified `component/name` — two components may hold one name each.
    pub fn reading(&self) -> Reading {
        if !self.components.is_empty() {
            let mut reading = Reading::default();
            for (name, component) in &self.components {
                let one = component.reading();
                reading.read.extend(
                    one.read
                        .into_iter()
                        .map(|tensor| format!("{name}/{tensor}")),
                );
                reading.unread.extend(
                    one.unread
                        .into_iter()
                        .map(|tensor| format!("{name}/{tensor}")),
                );
            }
            reading.read.sort();
            reading.unread.sort();
            return reading;
        }
        let read = self
            .read
            .lock()
            .map(|held| held.clone())
            .unwrap_or_default();
        let mut answer = Reading::default();
        for name in self.tensors.keys() {
            if read.contains(name) {
                answer.read.push(name.clone());
            } else {
                answer.unread.push(name.clone());
            }
        }
        answer
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

/// Read one shard's header through the `safetensors` crate: the byte length of the header (its JSON
/// body plus the eight-byte length that precedes it) and every tensor it names.
///
/// The format's own reader is used rather than one written here. `read_metadata` parses the length
/// and the header body, checks every tensor's byte range against its extents and element width, and
/// drops the optional `__metadata__` block, which carries no tensor. **What stays ours is the element
/// type**, because the crate's `Dtype` is the format's and ours is the operator ABI's: a type we
/// cannot express is refused here, at open, with the tensor in the message, rather than reach a
/// launch that would misread its bytes.
fn read_header(path: &Path, bytes: &[u8]) -> Result<(u64, Vec<(String, HeaderTensor)>)> {
    let (header, metadata) = SafeTensors::read_metadata(bytes).map_err(|problem| Error::Checkpoint {
        path: path.to_path_buf(),
        problem: format!("the safetensors header does not parse: {problem}"),
    })?;
    // **The crate answers the header's JSON length, not the payload's offset.** `read_metadata`
    // reads the file's leading eight bytes as `n` and parses the body from `N_LEN..N_LEN + n`,
    // then answers `Ok((n, metadata))` (`safetensors-0.8.0/src/tensor.rs`), so the payload starts at
    // `N_LEN + n`. Adding `n` alone shifts every tensor eight bytes early — the right shape over
    // the wrong bytes, which is why the read-back test asserts content rather than length.
    const HEADER_PREFIX: u64 = 8;
    let header = HEADER_PREFIX + header as u64;
    let payload = bytes.len() as u64 - header;

    // The crate answers its tensors in a hash map's order; the checkpoint keys them by name and a
    // report is diffed between two loads, so they are taken in name order here.
    let mut stated = metadata.tensors().into_iter().collect::<Vec<_>>();
    stated.sort_by(|left, right| left.0.cmp(&right.0));

    let mut tensors = Vec::with_capacity(stated.len());
    for (name, info) in stated {
        let dtype = element_type(path, &name, info.dtype)?;
        let shape = extents(path, &name, &info.shape)?;
        let start = info.data_offsets.0 as u64;
        let end = info.data_offsets.1 as u64;
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

/// The operator ABI element type the format's `Dtype` names.
///
/// **Every variant of the crate's enum is named here and there is no catch-all.** A variant the
/// crate adds must be a compile error in this file, because the alternative is a `_` arm that reads
/// a future element type as whatever the arm says, and a silent default in a checkpoint parser is a
/// wrong number with a right shape. The variants the ABI has no type for are refused by name: they
/// are real format types this operator set does not speak, and reading one as another type would be
/// worse than refusing the shard.
fn element_type(path: &Path, tensor: &str, dtype: FormatDtype) -> Result<DType> {
    let refuses = |kind: &str| {
        Err(Error::Checkpoint {
            path: path.to_path_buf(),
            problem: format!("tensor `{tensor}` is `{kind}`, which the operator ABI has no type for"),
        })
    };
    match dtype {
        FormatDtype::U8 => Ok(DType::U8),
        FormatDtype::I16 => Ok(DType::I16),
        FormatDtype::I32 => Ok(DType::I32),
        FormatDtype::I64 => Ok(DType::I64),
        FormatDtype::F16 => Ok(DType::F16),
        FormatDtype::BF16 => Ok(DType::BF16),
        FormatDtype::F32 => Ok(DType::F32),
        FormatDtype::F8_E4M3 => Ok(DType::F8E4M3),
        FormatDtype::F4 => Ok(DType::F4E2M1),
        FormatDtype::BOOL => refuses("BOOL"),
        FormatDtype::I8 => refuses("I8"),
        FormatDtype::U16 => refuses("U16"),
        FormatDtype::U32 => refuses("U32"),
        FormatDtype::U64 => refuses("U64"),
        FormatDtype::F64 => refuses("F64"),
        FormatDtype::C64 => refuses("C64"),
        FormatDtype::F6_E2M3 => refuses("F6_E2M3"),
        FormatDtype::F6_E3M2 => refuses("F6_E3M2"),
        FormatDtype::F8_E5M2 => refuses("F8_E5M2"),
        FormatDtype::F8_E8M0 => refuses("F8_E8M0"),
        FormatDtype::F8_E4M3FNUZ => refuses("F8_E4M3FNUZ"),
        FormatDtype::F8_E5M2FNUZ => refuses("F8_E5M2FNUZ"),
        // The crate marks `Dtype` `#[non_exhaustive]`, so the language requires a catch-all: a
        // release may name an element type this build has never heard of. It **refuses** and prints
        // the variant it saw rather than reading it as one of ours, so the arm the exhaustiveness
        // rule forces is the one that cannot become a silent default.
        other => refuses(&format!("{other:?}")),
    }
}

/// A shape the operator ABI can hold.
///
/// The format allows any rank; the ABI's is a fixed maximum, so a shard of deeper tensors is refused
/// by name rather than truncated to the axes that fit.
fn extents(path: &Path, tensor: &str, dims: &[usize]) -> Result<Shape> {
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
    let mut extents = [0_u64; ops::MAX_RANK];
    for (axis, &extent) in dims.iter().enumerate() {
        extents[axis] = extent as u64;
    }
    Ok(Shape::new(&extents[..dims.len()]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pipeline handle names every component directory — including one that holds an index and no
    /// shards, which `open` skips — and reading a tensor from it refuses by name rather than reporting
    /// a name absent from a table that was never filled.
    ///
    /// This is what a walk over a metadata-only deployment uses: the weights' shapes come from the
    /// model's own `load`, and the checkpoint is a handle to the component directories only.
    #[test]
    fn an_unopened_pipeline_names_components_and_reads_no_tensor() {
        let root = tempfile::tempdir().expect("a temp directory");
        for component in ["vae", "text_encoder"] {
            std::fs::create_dir(root.path().join(component)).expect("a component directory");
        }
        // An index with no shard beside it is what a metadata-only deployment looks like.
        std::fs::write(
            root.path()
                .join("vae")
                .join("diffusion_pytorch_model.safetensors.index.json"),
            br#"{"weight_map":{}}"#,
        )
        .expect("an index");
        std::fs::write(root.path().join("model_index.json"), b"{}").expect("a pipeline index");

        let checkpoint = Checkpoint::unopened(root.path()).expect("the pipeline handle");
        let mut names = checkpoint.components().collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(names, vec!["text_encoder", "vae"]);
        assert!(checkpoint.component("vae").is_ok());
        assert!(checkpoint.component("audio_vae").is_err());

        let problem = checkpoint
            .component("vae")
            .expect("the vae handle")
            .read("weight")
            .expect_err("an unopened component reads nothing");
        assert!(
            problem.to_string().contains("reads no tensor"),
            "the refusal says what this handle is: {problem}"
        );
    }

    /// Write one shard of `tensors` — each `(name, dtype, shape, bytes)` — into `directory` and
    /// answer the root to open.
    ///
    /// The header is composed here from the format's own grammar (an eight-byte length, the JSON
    /// body, then the payload laid out in the order the offsets state), so the test states the
    /// format rather than going through the parser it is testing.
    fn shard(directory: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) -> PathBuf {
        let mut body = String::from("{");
        let mut payload = Vec::new();
        for (index, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if index > 0 {
                body.push(',');
            }
            let start = payload.len();
            payload.extend_from_slice(bytes);
            let extents = shape
                .iter()
                .map(|extent| extent.to_string())
                .collect::<Vec<_>>()
                .join(",");
            body.push_str(&format!(
                r#""{name}":{{"dtype":"{dtype}","shape":[{extents}],"data_offsets":[{start},{}]}}"#,
                payload.len()
            ));
        }
        body.push('}');

        // The format states the header length as a multiple of 8 so that the payload is
        // 8-byte aligned, and every real exporter obeys it; JSON permits the trailing spaces a
        // padded body needs. A fixture that omits a rule the real files obey tests the fixture.
        let padded = body.len().div_ceil(8) * 8;
        body.push_str(&" ".repeat(padded - body.len()));

        let mut file = Vec::new();
        file.extend_from_slice(&(body.len() as u64).to_le_bytes());
        file.extend_from_slice(body.as_bytes());
        file.extend_from_slice(&payload);
        let path = directory.join("model.safetensors");
        std::fs::write(&path, &file).expect("a shard");
        path
    }

    /// A shard's tensors read back **byte for byte at the payload-relative offsets the header
    /// states**, each with its own element type and extents, and a non-float tensor is named.
    ///
    /// The trap this pins is the one the crate's `read_metadata` makes easy to fall into: the
    /// payload starts *after* the header's length, so a reader that adds the wrong offset answers a
    /// slice of the right length for every tensor and every shape agrees. The bytes are therefore
    /// unlike the header's own text and unlike each other, and the integer tensor is checked because
    /// `names()` must keep its meaning for it: a parser that filtered to float types would make a
    /// mixed component look uniform.
    #[test]
    fn a_shards_tensors_read_back_at_their_own_offsets_and_types() {
        let root = tempfile::tempdir().expect("a temp directory");
        // Two elements of BF16: 2 × 3 extents, twelve bytes.
        let halfs: [u8; 12] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc];
        // Four I32 elements, sixteen bytes.
        let words: [u8; 16] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ];
        shard(
            root.path(),
            &[
                ("halfs", "BF16", &[2, 3], &halfs),
                ("words", "I32", &[4], &words),
            ],
        );

        let checkpoint = Checkpoint::open(root.path()).expect("the shard opens");
        let mut names = checkpoint.names().collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["halfs", "words"],
            "a non-float tensor is named like any other"
        );

        let halfs_read = checkpoint.read("halfs").expect("halfs");
        assert_eq!(halfs_read.dtype, DType::BF16, "the tensor's own type");
        assert_eq!(halfs_read.shape, Shape::new(&[2, 3]), "its own extents");
        assert_eq!(halfs_read.bytes, halfs, "the bytes the header points at");

        let words_read = checkpoint.read("words").expect("words");
        assert_eq!(words_read.dtype, DType::I32, "an integer keeps its type");
        assert_eq!(words_read.shape, Shape::new(&[4]));
        assert_eq!(words_read.bytes, words, "and its own bytes");
    }

    /// A shard whose tensor is a real format type the operator ABI does not speak is refused **at
    /// open**, naming the tensor and the type, rather than read as one of ours.
    #[test]
    fn a_tensor_of_an_unknown_element_type_is_refused_by_name() {
        let root = tempfile::tempdir().expect("a temp directory");
        // `F8_E5M2` is eight bits an element, so two elements are two bytes and the crate's own
        // header validation is satisfied: what refuses is this loader's type match.
        shard(root.path(), &[("wide", "F8_E5M2", &[2], &[0x00, 0x01])]);

        let problem = Checkpoint::open(root.path())
            .err()
            .expect("a type the ABI has no name for is refused");
        let message = problem.to_string();
        assert!(
            message.contains("wide") && message.contains("F8_E5M2"),
            "the refusal names the tensor and the type: {message}"
        );
    }

}
