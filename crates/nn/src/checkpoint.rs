//! A checkpoint: its safetensors shards, the tensors they hold, and the identity of every file.
//!
//! The loader never converts a checkpoint's weights in place and never holds a whole shard in host
//! memory: each shard is mapped read-only with `memmap2`, the `safetensors` crate parses its header
//! through `read_metadata`, and one tensor's bytes are copied when a weight is converted. A shard is
//! mapped for the life of the [`Checkpoint`], which is the model's life; the mapping costs virtual
//! address space, not resident memory.
//!
//! **The format is read by the format's own crate, and the element type is ours.** `safetensors`
//! parses the header and checks each tensor's byte range against its extents and element width,
//! and the optional metadata block is kept per file as text; its `Dtype` now names every type this
//! checkpoint set carries, FP8 and FP4 included. What stays here is the map from that `Dtype` onto the operator ABI's
//! [`DType`], because the ABI names a subset: a format type it cannot express is refused at open
//! with the tensor and the type in the message, rather than reach a launch that would misread its
//! bytes. The match names every variant of the crate's enum, and the catch-all that its
//! `#[non_exhaustive]` makes the language require **refuses and prints the variant it saw**, so a
//! type a future release adds is a shard this loader will not open, never a number read as another.
//!
//! ## File identity
//!
//! Retained weights are keyed by the identity of the files they were read from, so a
//! checkpoint edited after it was loaded cannot serve the old retained bytes. The caller states
//! where a checkpoint's files come from ([`Source`]), and that statement decides the identity:
//!
//! - a file of a Hugging Face snapshot is named by the repository, the commit the snapshot holds
//!   and the file's path within it, because a commit's files never change;
//! - a local file is named by its status: device, inode, modification and status-change times,
//!   a link followed. Rewriting the file changes its times, at the filesystem's timestamp
//!   granularity, so a load after the rewrite is keyed apart from the bytes retained before it.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use memmap2::Mmap;
use safetensors::Dtype as FormatDtype;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;
use tensor::nditer::MAX_DIMS;
use tensor::{DType, Shape};

use crate::adapter::{Adapter, Fusion};
use crate::error::{Error, Result};

/// The bytes of a shard before its header's JSON body: the body's length, as a `u64`.
const HEADER_PREFIX: u64 = 8;

/// Where a directory of checkpoint files comes from, as the caller resolved it: the statement
/// that decides each file's [`FileIdentity`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A directory of the local filesystem, as a deployment names it.
    Local(PathBuf),
    /// A directory of a Hugging Face snapshot: a repository's files at one commit.
    Snapshot(Snapshot),
}

/// A directory of a Hugging Face repository's snapshot at one commit, as the asset resolver
/// fetched it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// The repository id, `<owner>/<name>`.
    repository: String,
    /// The commit the snapshot holds.
    revision: String,
    /// Where the directory lies on this host.
    directory: PathBuf,
    /// The directory's path within the snapshot, empty for the snapshot's root.
    within: PathBuf,
}

impl Snapshot {
    /// The repository id, `<owner>/<name>`.
    pub fn repository(&self) -> &str {
        &self.repository
    }

    /// The commit the snapshot holds.
    pub fn revision(&self) -> &str {
        &self.revision
    }
}

impl Source {
    /// The root of the snapshot of `repository` at the commit `revision`, which lies at
    /// `directory`.
    pub fn snapshot(
        repository: impl Into<String>,
        revision: impl Into<String>,
        directory: impl Into<PathBuf>,
    ) -> Self {
        Self::Snapshot(Snapshot {
            repository: repository.into(),
            revision: revision.into(),
            directory: directory.into(),
            within: PathBuf::new(),
        })
    }

    /// Where the directory lies on this host.
    pub fn directory(&self) -> &Path {
        match self {
            Self::Local(directory) => directory,
            Self::Snapshot(snapshot) => &snapshot.directory,
        }
    }

    /// The directory `name` below this one, from the same place.
    pub fn join(&self, name: impl AsRef<Path>) -> Self {
        let name = name.as_ref();
        match self {
            Self::Local(directory) => Self::Local(directory.join(name)),
            Self::Snapshot(snapshot) => Self::Snapshot(Snapshot {
                directory: snapshot.directory.join(name),
                within: snapshot.within.join(name),
                ..snapshot.clone()
            }),
        }
    }

    /// The identity of `file`, a file of this directory or of a directory below it.
    ///
    /// # Errors
    ///
    /// When a local file's status cannot be read, or `file` lies outside a snapshot's directory.
    pub fn identity(&self, file: &Path) -> Result<FileIdentity> {
        match self {
            Self::Local(_) => FileIdentity::of(file),
            Self::Snapshot(snapshot) => {
                let relative =
                    file.strip_prefix(&snapshot.directory)
                        .map_err(|_| Error::Checkpoint {
                            path: file.to_path_buf(),
                            problem: format!(
                                "lies outside the snapshot directory {} it is read from",
                                snapshot.directory.display()
                            ),
                        })?;
                Ok(FileIdentity(format!(
                    "{}@{}/{}",
                    snapshot.repository,
                    snapshot.revision,
                    snapshot.within.join(relative).display()
                )))
            }
        }
    }
}

/// The identity of one checkpoint file, as retained weights are keyed by it.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct FileIdentity(String);

impl FileIdentity {
    /// The identity of the local file `path`: its device, inode, modification and status-change
    /// times, a link followed.
    ///
    /// # Errors
    ///
    /// When the file's status cannot be read.
    pub fn of(path: &Path) -> Result<Self> {
        let resolved = std::fs::canonicalize(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let status = std::fs::metadata(&resolved).map_err(|source| Error::Io {
            path: resolved.clone(),
            source,
        })?;

        // Each time in nanoseconds since the epoch: its seconds and its nanosecond remainder.
        let mtime = status.mtime() as i128 * 1_000_000_000 + i128::from(status.mtime_nsec());
        let ctime = status.ctime() as i128 * 1_000_000_000 + i128::from(status.ctime_nsec());
        Ok(Self(format!(
            "{}:{}:{}:{}",
            status.dev(),
            status.ino(),
            mtime,
            ctime
        )))
    }

    /// The identity of a value produced from the files `base` and `adapter` names together: an
    /// adapter-fused parameter, which neither file holds alone.
    pub(crate) fn joined(base: &Self, adapter: &Self) -> Self {
        Self(format!("{}+adapter:{}", base.0, adapter.0))
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

/// One mapped safetensors shard.
pub struct CheckpointFile {
    path: PathBuf,
    identity: FileIdentity,
    /// Bytes of the file before the tensor payload.
    header: u64,
    /// The header's `__metadata__` block: the exporter's free-form statements about the file.
    metadata: BTreeMap<String, String>,
    map: Mmap,
}

impl CheckpointFile {
    /// The shard's identity.
    pub fn identity(&self) -> &FileIdentity {
        &self.identity
    }

    /// The shard's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The header's `__metadata__` block, empty when the exporter wrote none.
    ///
    /// No weight reads it; an adapter states its format, rank and provenance there
    /// ([`crate::adapter::Adapter::open`]).
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
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

/// A checkpoint: its mapped shards and the tensors they hold, or a pipeline's stages.
///
/// A flat checkpoint is one shard set; a *pipeline* root holds one subdirectory per stage
/// (`transformer/`, `vae/`, `text_encoder/`, …), each with its own index and therefore its own file
/// identities, which §8's retained keys name. Opening a root discovers which of the two it is: a
/// root with a shard set of its own is flat and behaves as it always has, and a root without one is
/// read as the stages its subdirectories hold.
pub struct Checkpoint {
    /// The directory this checkpoint was opened at, and where its files come from.
    source: Source,
    /// Every root this checkpoint's tensors came from, in precedence order; one entry unless it was
    /// opened over several roots ([`Checkpoint::open_roots`]). A missing tensor names them, because
    /// a union that spans an export and a base is where "not found" costs the most to chase.
    roots: Vec<PathBuf>,
    files: Vec<CheckpointFile>,
    tensors: BTreeMap<String, TensorLocation>,
    stages: BTreeMap<String, Checkpoint>,
    /// The tensors a `read` has asked this checkpoint for, so [`Checkpoint::reading`] can tell the
    /// other half of the question: what the load never touched.
    read: Mutex<BTreeSet<String>>,
    /// The adapter fused into this shard set's tensors ([`Checkpoint::attach`]).
    adapter: Option<Fusion>,
}

impl Checkpoint {
    /// Open the checkpoint in the local directory `root`.
    ///
    /// A `model.safetensors.index.json` names each tensor's shard; without one every
    /// `*.safetensors` file of the directory is read and its tensors are taken from its header.
    pub fn open(root: &Path) -> Result<Self> {
        Self::open_at(Source::Local(root.to_path_buf()))
    }

    /// Open the checkpoint at `source`, a directory whose files' identities it states.
    fn open_at(source: Source) -> Result<Self> {
        let root = source.directory();
        let index = root.join("model.safetensors.index.json");
        let paths = if index.is_file() {
            shards_from_index(&index)?
        } else if let Some(paths) = shards_in(root)? {
            paths
        } else {
            return Self::pipeline(source);
        };
        Self::from_shards(&source, paths)
    }

    /// Open the shard set `paths`, files of the directory `source`, as one flat checkpoint.
    pub(crate) fn from_shards(source: &Source, paths: Vec<PathBuf>) -> Result<Self> {
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
            let identity = source.identity(&path)?;
            let (length, stated) =
                SafeTensors::read_metadata(&map).map_err(|problem| Error::Checkpoint {
                    path: path.clone(),
                    problem: format!("the safetensors header does not parse: {problem}"),
                })?;

            // **The format crate returns the header's JSON length, not the payload's offset.**
            // `read_metadata` reads the file's leading eight bytes as `n` and parses the body
            // from `N_LEN..N_LEN + n` (`safetensors-0.8.0/src/tensor.rs`), so the payload starts
            // at `N_LEN + n`. Adding `n` alone shifts every tensor eight bytes early, the right
            // shape over the wrong bytes, which is why the read-back test asserts content and
            // not length.
            //
            // **The subtraction cannot underflow, and this is the dependency, not an
            // assumption.** The same crate refuses a header whose `8 + n` runs past the buffer
            // (`InvalidHeaderLength`) and requires `payload_end + 8 + n == buffer_len` exactly
            // (`MetadataIncompleteBuffer`). A version that relaxed either would let this line
            // read past the file: a bump that changes those two checks must visit it.
            let header = HEADER_PREFIX + length as u64;
            let payload = map.len() as u64 - header;

            let metadata = stated
                .metadata()
                .as_ref()
                .map(|block| block.clone().into_iter().collect())
                .unwrap_or_default();
            for location in header_tensors(&path, &stated, file_index, payload)? {
                if locations
                    .insert(location.tensor.clone(), location)
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
                metadata,
                map,
            });
        }

        Ok(Self {
            source: source.clone(),
            roots: vec![source.directory().to_path_buf()],
            files,
            tensors: locations,
            stages: BTreeMap::new(),
            read: Mutex::new(BTreeSet::new()),
            adapter: None,
        })
    }

    /// Open `roots` as **one** checkpoint: the first root's tensors shadow the later ones'.
    ///
    /// A deployment that ships a delta export over a base is one checkpoint whose file set spans
    /// both, so the union is per tensor and not per stage: stages are enumerated in each
    /// root as [`Checkpoint::open`] already does; a stage present in several roots is one
    /// unioned stage whose tensors come from the earliest root that holds each name; a
    /// stage that only a later root holds is that root's; and the same shadowing applies at the
    /// top level, where an export's tensor replaces the base's.
    ///
    /// A shadowed tensor is expected and is **not** a byte comparison: a delta export legitimately
    /// differs from the base it was trained from, and refusing the difference would refuse the
    /// deployment. Two roots of *one* root's shard set disagreeing is still an error, because that
    /// is a malformed checkpoint and no precedence can say which is meant.
    ///
    /// [`Checkpoint::root`] returns the first root: the export's, which is the one a log names.
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
        for (name, stage) in later.stages {
            let union = match self.stages.remove(&name) {
                Some(held) => held.merge(stage),
                None => stage,
            };
            self.stages.insert(name, union);
        }
        self
    }

    /// Open a pipeline root: one stage per subdirectory that holds a shard set.
    fn pipeline(source: Source) -> Result<Self> {
        let root = source.directory();
        let entries = std::fs::read_dir(root).map_err(|source| Error::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let mut stages = BTreeMap::new();
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
            stages.insert(name.clone(), Self::open_at(source.join(&name))?);
        }
        if stages.is_empty() {
            return Err(Error::Checkpoint {
                path: root.to_path_buf(),
                problem: "no shard set here and none in a stage below it".to_owned(),
            });
        }
        Ok(Self {
            roots: vec![root.to_path_buf()],
            source,
            files: Vec::new(),
            tensors: BTreeMap::new(),
            stages,
            read: Mutex::new(BTreeSet::new()),
            adapter: None,
        })
    }

    /// The directory this checkpoint was opened with — and, for a union of roots, the **first** of
    /// them: the export's, which is the one a log names.
    ///
    /// A pipeline's stages are opened under it, so a model's `load()` can reach each stage
    /// without a second copy of the path being threaded through the engine.
    pub fn root(&self) -> &Path {
        self.source.directory()
    }

    /// Every directory this checkpoint spans, the first root's first.
    ///
    /// A pipeline's sidecars live in these directories rather than in a shard set: a processor has
    /// no weights, so no stage holds it.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Whether this checkpoint is a pipeline of stages rather than one model's shards.
    pub fn is_pipeline(&self) -> bool {
        !self.stages.is_empty()
    }

    /// The stage numbered `name`, with its own shards and file identities.
    pub fn stage(&self, name: &str) -> Result<&Checkpoint> {
        self.stages.get(name).ok_or_else(|| Error::Checkpoint {
            path: PathBuf::from(name),
            problem: format!(
                "this checkpoint has no stage `{name}`; it holds {}",
                self.stages
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
    }

    /// The names of the stages, in order.
    pub fn stages(&self) -> impl Iterator<Item = &str> {
        self.stages.keys().map(String::as_str)
    }

    /// Every element type the checkpoint or one of its stages holds, each once.
    pub fn dtypes(&self) -> Vec<DType> {
        let mut dtypes: Vec<DType> = Vec::new();
        for dtype in self
            .tensors
            .values()
            .map(|location| location.dtype)
            .chain(self.stages.values().flat_map(Checkpoint::dtypes))
        {
            if !dtypes.contains(&dtype) {
                dtypes.push(dtype);
            }
        }
        dtypes
    }

    /// The floating-point element types this checkpoint's own tensors are stored in, each once and
    /// with the name of one tensor that carries it.
    ///
    /// Integer tensors (a table, an index) are not weights a precision describes and do not count.
    /// A stage lists its own shards'; a pipeline root with no shards of its own lists
    /// none.
    pub fn stored_types(&self) -> Vec<(DType, String)> {
        let mut found: Vec<(DType, String)> = Vec::new();
        for (name, location) in &self.tensors {
            if location.dtype.is_float() && !found.iter().any(|(dtype, _)| *dtype == location.dtype)
            {
                found.push((location.dtype, name.clone()));
            }
        }
        found
    }

    /// The one floating-point element type this checkpoint's weights are stored in.
    ///
    /// A loader that converts nothing keeps each weight in its stored type, so a consumer that
    /// takes one element type for a whole stage (a block whose activations follow its
    /// weights) asks this. A checkpoint of several stored types is legal — each weight loads in its
    /// own — and is refused here only because the caller asked for one, with every type named.
    ///
    /// # Errors
    ///
    /// When the checkpoint holds no floating-point tensor, or more than one floating-point type.
    pub fn stored_dtype(&self) -> Result<DType> {
        let found = self.stored_types();
        match found.as_slice() {
            [(dtype, _)] => Ok(*dtype),
            [] => Err(Error::Checkpoint {
                path: self.root().to_path_buf(),
                problem: "holds no floating-point tensor, so no element type describes its weights"
                    .to_owned(),
            }),
            several => Err(Error::Checkpoint {
                path: self.root().to_path_buf(),
                problem: format!(
                    "stores {} floating-point types ({}), and the consumer that asked takes one \
                     element type for the whole checkpoint; each weight still loads in its own \
                     stored type, so this is a limit of that consumer, not of the export",
                    several.len(),
                    several
                        .iter()
                        .map(|(dtype, example)| format!(
                            "{dtype} (such as `{example}`)",
                            dtype = ops::abi::name(*dtype)
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }),
        }
    }

    /// The JSON document `relative` names under the first of this checkpoint's roots that holds
    /// it, or `None` when no root does.
    ///
    /// A checkpoint's configuration lives beside its shards rather than in them (`config.json`, a
    /// pipeline's `model_index.json`, a stage's `vae/config.json`, a quantization
    /// declaration), and a checkpoint opened over an export and the base it continues reads each
    /// document from the export first. Whether an absent document is a refusal is the reader's
    /// question, so absence is `None`.
    ///
    /// # Errors
    ///
    /// When a root holds the file and it cannot be read or is not JSON.
    pub fn sidecar(&self, relative: &str) -> Result<Option<serde_json::Value>> {
        for root in &self.roots {
            let path = root.join(relative);
            if !path.is_file() {
                continue;
            }
            let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            return serde_json::from_str(&text)
                .map(Some)
                .map_err(|error| Error::Checkpoint {
                    path,
                    problem: format!("is not JSON: {error}"),
                });
        }
        Ok(None)
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
    ///
    /// An adapter-fused tensor is produced from the base shard and the adapter together, and its
    /// identity names both ([`Checkpoint::attach`]).
    pub fn identity(&self, tensor: &str) -> Option<&FileIdentity> {
        if let Some(delta) = self
            .adapter
            .as_ref()
            .and_then(|fusion| fusion.delta(tensor))
        {
            return Some(&delta.identity);
        }
        let location = self.tensors.get(tensor)?;
        Some(self.files[location.file].identity())
    }

    /// The location of the tensor named `name`, as its shard's header states it.
    ///
    /// Unlike [`Checkpoint::tensor`], an absent name is an error, and for a union of several roots
    /// the error lists the roots searched in precedence order. No payload byte is touched.
    pub fn locate(&self, name: &str) -> Result<&TensorLocation> {
        self.tensors.get(name).ok_or_else(|| {
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
                path: self.root().to_path_buf(),
                problem: format!(
                    "no root holds a tensor `{name}`; searched, in precedence order: {tried}"
                ),
            }
        })
    }

    /// Reads the bytes of the tensor named `name` into host memory, row-major, in the element
    /// type and extents [`Checkpoint::locate`] states for it; the fused value of a tensor an
    /// attached adapter edits.
    ///
    /// # Errors
    ///
    /// When the checkpoint holds no tensor of the name.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let location = self.locate(name)?;
        if let Ok(mut read) = self.read.lock() {
            read.insert(name.to_owned());
        }

        let length = usize::try_from(location.bytes.1 - location.bytes.0)
            .map_err(|_| Error::weight(name, "the tensor's bytes are past usize::MAX"))?;
        self.payload(location, &[(0, length)])
    }

    /// The bytes `ranges` name of the tensor at `location`, concatenated in order: the shard's own
    /// bytes, or the fused value of a tensor an attached adapter edits.
    ///
    /// Each range is a half-open byte range within the tensor's own bytes. Every weight a load
    /// publishes and every tensor [`Checkpoint::read`] returns is taken through here, so an
    /// adapter's edit reaches every reader of the tensor, and a tensor no adapter edits is copied
    /// from its mapping unchanged.
    pub(crate) fn payload(
        &self,
        location: &TensorLocation,
        ranges: &[(usize, usize)],
    ) -> Result<Vec<u8>> {
        if let Some(delta) = self
            .adapter
            .as_ref()
            .and_then(|fusion| fusion.delta(&location.tensor))
        {
            return crate::adapter::fused(self, location, delta, ranges);
        }
        let bytes = self.bytes(location)?;
        let total = ranges
            .iter()
            .map(|(from, to)| to.saturating_sub(*from))
            .sum();
        let mut gathered = Vec::with_capacity(total);
        for &(from, to) in ranges {
            let range = bytes.get(from..to).ok_or_else(|| {
                Error::weight(
                    &location.tensor,
                    format!(
                        "bytes {from}..{to} reach past the tensor's {} bytes",
                        bytes.len()
                    ),
                )
            })?;
            gathered.extend_from_slice(range);
        }
        Ok(gathered)
    }

    /// Fuse `adapter` into this checkpoint's tensors, or into the stage `stage` of a
    /// pipeline root, and return the adapted checkpoint.
    ///
    /// From then on every byte read of a tensor the adapter edits is its fused value, a tensor the
    /// adapter assigns is one of the checkpoint's tensors, and every other tensor reads as it did
    /// ([`crate::adapter`]). Which stage an adapter edits is the model family's statement, as
    /// its checkpoint mapping is ([`crate::family::VideoFamily::adapter_stage`]).
    ///
    /// # Errors
    ///
    /// Fails when the stage does not exist or already holds an adapter, when the adapter edits
    /// a tensor the checkpoint does not hold, a non-floating-point tensor, or one whose extents its
    /// delta does not match, or when it assigns a tensor the checkpoint already holds.
    pub fn attach(mut self, stage: Option<&str>, adapter: Adapter) -> Result<Self> {
        if let Some(name) = stage {
            let held = self.stages.remove(name).ok_or_else(|| Error::Checkpoint {
                path: self.root().to_path_buf(),
                problem: format!("an adapter edits the stage `{name}`, which is not here"),
            })?;
            let adapted = held.attach(None, adapter)?;
            self.stages.insert(name.to_owned(), adapted);
            return Ok(self);
        }
        if self.adapter.is_some() || !self.stages.is_empty() {
            return Err(Error::Checkpoint {
                path: self.root().to_path_buf(),
                problem: "an adapter is fused into one shard set that holds none yet".to_owned(),
            });
        }
        let fusion = adapter.fuse_into(&mut self)?;
        self.adapter = Some(fusion);
        Ok(self)
    }

    /// The adapter fused into this shard set, when one is.
    pub fn adapter(&self) -> Option<&Fusion> {
        self.adapter.as_ref()
    }

    /// Append `files` to this checkpoint's files and return the index the first one takes.
    pub(crate) fn adopt(&mut self, files: Vec<CheckpointFile>) -> usize {
        let first = self.files.len();
        self.files.extend(files);
        first
    }

    /// Hold `location` as one of this checkpoint's tensors, under its own name.
    pub(crate) fn place(&mut self, location: TensorLocation) {
        self.tensors.insert(location.tensor.clone(), location);
    }

    /// This flat checkpoint's files and tensor locations, taken apart.
    pub(crate) fn into_shards(self) -> (Vec<CheckpointFile>, BTreeMap<String, TensorLocation>) {
        (self.files, self.tensors)
    }

    /// The tensors a load has read from this checkpoint, and the tensors it has left alone, each
    /// list in name order so two loads can be compared.
    ///
    /// A stage of a multi-stage export legitimately reads a subset of what its shards hold
    /// (a second calibration scheme's per-layer `scale1`/`scale2`, a tied head, the calibration
    /// fields of a layer the model runs dense), so a non-empty second list is never a refusal. It
    /// says, when the numbers are wrong, which tensors nobody asked for. It reads no tensor's
    /// values.
    ///
    /// A pipeline root holds no tensors of its own, so its lists are its stages', each name
    /// qualified `stage/name`: two stages may hold one name each.
    pub fn reading(&self) -> (Vec<String>, Vec<String>) {
        let (mut read, mut unread) = (Vec::new(), Vec::new());

        if !self.stages.is_empty() {
            for (name, stage) in &self.stages {
                let (stage_read, stage_unread) = stage.reading();
                read.extend(stage_read.iter().map(|tensor| format!("{name}/{tensor}")));
                unread.extend(stage_unread.iter().map(|tensor| format!("{name}/{tensor}")));
            }
            read.sort();
            unread.sort();
            return (read, unread);
        }

        let asked = self
            .read
            .lock()
            .map(|held| held.clone())
            .unwrap_or_default();
        for name in self.tensors.keys() {
            if asked.contains(name) {
                read.push(name.clone());
            } else {
                unread.push(name.clone());
            }
        }
        (read, unread)
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
/// a shard set at all, which a pipeline root's discovery uses to tell stages from other files.
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

/// Every tensor the header `stated` of shard `file` names, in name order, each with its element
/// type, its extents and its byte range in a payload of `payload` bytes.
///
/// The format's own crate parsed the header and checked every tensor's byte range against its
/// extents and element width. **What stays this crate's is the element type**, because the
/// format crate's `Dtype` is the format's and the tensor library's is what a kernel computes on:
/// a type that cannot be expressed is refused here, at open, with the tensor in the message,
/// before it could reach a launch that would misread its bytes.
fn header_tensors(
    path: &Path,
    stated: &Metadata,
    file: usize,
    payload: u64,
) -> Result<Vec<TensorLocation>> {
    // The crate lists its tensors in a hash map's order; the checkpoint keys them by name and two
    // loads' readings are compared, so they are taken in name order here.
    let mut entries = stated.tensors().into_iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));

    let mut tensors = Vec::with_capacity(entries.len());
    for (name, info) in entries {
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
        tensors.push(TensorLocation {
            file,
            tensor: name,
            dtype,
            shape,
            bytes: (start, end),
        });
    }
    Ok(tensors)
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
            problem: format!(
                "tensor `{tensor}` is `{kind}`, which the operator ABI has no type for"
            ),
        })
    };
    match dtype {
        FormatDtype::U8 => Ok(DType::U8),
        FormatDtype::U32 => Ok(DType::U32),
        FormatDtype::I16 => Ok(DType::I16),
        FormatDtype::I32 => Ok(DType::I32),
        FormatDtype::I64 => Ok(DType::I64),
        FormatDtype::F16 => Ok(DType::F16),
        FormatDtype::BF16 => Ok(DType::BF16),
        FormatDtype::F32 => Ok(DType::F32),
        FormatDtype::F64 => Ok(DType::F64),
        FormatDtype::F8_E4M3 => Ok(DType::F8E4M3),
        FormatDtype::F4 => Ok(DType::F4),
        FormatDtype::BOOL => refuses("BOOL"),
        FormatDtype::I8 => refuses("I8"),
        FormatDtype::U16 => refuses("U16"),
        FormatDtype::U64 => refuses("U64"),
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

/// The extents of a tensor a kernel argument can carry.
///
/// The format allows any rank; a tensor's strided operations and a kernel argument take at most
/// [`MAX_DIMS`](tensor::nditer::MAX_DIMS) axes, so a shard of deeper tensors is refused by name
/// instead of truncated to the axes that fit.
fn extents(path: &Path, tensor: &str, dims: &[usize]) -> Result<Shape> {
    if dims.len() > MAX_DIMS {
        return Err(Error::Checkpoint {
            path: path.to_path_buf(),
            problem: format!(
                "tensor `{tensor}` has {} axes, more than a kernel argument's {MAX_DIMS}",
                dims.len()
            ),
        });
    }
    Ok(Shape::from(dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write one shard of `tensors` — each `(name, dtype, shape, bytes)` — into `directory` and
    /// return the root to open.
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

    /// A tensor's name is matched exactly as the checkpoint stores it.
    ///
    /// A layout states the names a checkpoint holds, so a name that differs by a `model.` prefix is a
    /// different name: the map is keyed by the stored spelling and a load asks for that spelling. The
    /// audit over a deployment may normalize the prefix, because it compares the checkpoint's names
    /// against the model's key rules; this is the load's side of that distinction, and it is what makes
    /// the audit's normalized count a statement a reader can check.
    #[test]
    fn a_tensor_name_is_matched_exactly_as_the_checkpoint_stores_it() {
        let root = tempfile::tempdir().expect("a temp directory");
        // Four BF16 elements, eight bytes.
        let halfs = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        shard(
            root.path(),
            &[("layers.0.attention.q_proj.weight", "BF16", &[2, 2], &halfs)],
        );

        let checkpoint = Checkpoint::open(root.path()).expect("the shard opens");
        assert!(
            checkpoint
                .tensor("layers.0.attention.q_proj.weight")
                .is_some(),
            "the stored spelling resolves"
        );
        assert!(
            checkpoint
                .tensor("model.layers.0.attention.q_proj.weight")
                .is_none(),
            "a name that differs by the prefix is a different name, not a fallback"
        );
    }

    /// A shard's tensors read back **byte for byte at the payload-relative offsets the header
    /// states**, each with its own element type and extents, and a non-float tensor is named.
    ///
    /// The trap this pins is the one the crate's `read_metadata` makes easy to fall into: the
    /// payload starts *after* the header's length, so a reader that adds the wrong offset returns a
    /// slice of the right length for every tensor and every shape agrees. The bytes are therefore
    /// unlike the header's own text and unlike each other, and the integer tensor is checked because
    /// `names()` must keep its meaning for it: a parser that filtered to float types would make a
    /// mixed stage look uniform.
    #[test]
    fn a_shards_tensors_read_back_at_their_own_offsets_and_types() {
        let root = tempfile::tempdir().expect("a temp directory");
        // Two elements of BF16: 2 × 3 extents, twelve bytes.
        let halfs: [u8; 12] = [
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        ];
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

        let located = checkpoint.locate("halfs").expect("halfs");
        assert_eq!(located.dtype, DType::BF16, "the tensor's own type");
        assert_eq!(located.shape.dims(), [2, 3], "its own extents");
        assert_eq!(
            checkpoint.read("halfs").expect("halfs"),
            halfs,
            "the bytes the header points at"
        );

        let located = checkpoint.locate("words").expect("words");
        assert_eq!(located.dtype, DType::I32, "an integer keeps its type");
        assert_eq!(located.shape.dims(), [4]);
        assert_eq!(
            checkpoint.read("words").expect("words"),
            words,
            "and its own bytes"
        );
    }

    /// Every format type the operator ABI numbers is read as that element type, the unsigned
    /// 32-bit integers of an index table and the FP64 values of a schedule among them.
    #[test]
    fn a_tensor_of_every_element_type_the_abi_numbers_keeps_its_type() {
        let root = tempfile::tempdir().expect("a temp directory");
        // Sixteen bytes: two FP64 or I64 elements, four of 32 bits, eight of 16 bits, sixteen
        // of one byte and thirty-two of four bits.
        let bytes: [u8; 16] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ];
        let stored: [(&str, DType, usize); 11] = [
            ("U8", DType::U8, 16),
            ("U32", DType::U32, 4),
            ("I16", DType::I16, 8),
            ("I32", DType::I32, 4),
            ("I64", DType::I64, 2),
            ("F16", DType::F16, 8),
            ("BF16", DType::BF16, 8),
            ("F32", DType::F32, 4),
            ("F64", DType::F64, 2),
            ("F8_E4M3", DType::F8E4M3, 16),
            ("F4", DType::F4, 32),
        ];
        let shapes: Vec<[usize; 1]> = stored.iter().map(|(_, _, count)| [*count]).collect();
        let tensors: Vec<(&str, &str, &[usize], &[u8])> = stored
            .iter()
            .zip(&shapes)
            .map(|((name, _, _), shape)| (*name, *name, &shape[..], &bytes[..]))
            .collect();
        shard(root.path(), &tensors);

        let checkpoint = Checkpoint::open(root.path()).expect("the shard opens");
        for (name, dtype, count) in stored {
            let located = checkpoint.locate(name).expect(name);
            assert_eq!(located.dtype, dtype, "{name}");
            assert_eq!(located.shape.dims(), [count], "{name}");
            assert_eq!(checkpoint.read(name).expect(name), bytes, "{name}");

            // The ABI numbers every type a checkpoint may hold.
            assert!(ops::abi::code(dtype).is_ok(), "{name}");
        }
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
