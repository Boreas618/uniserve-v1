//! Checkpoints on disk: the ones a test writes, and the committed fixtures a test embeds.
//!
//! A written checkpoint follows the transformers declaration of a shard set
//! (`nn::Library::Transformers`), which is what `nn::Checkpoint::open` reads a directory by: one
//! shard is `model.safetensors`, and several are `model-0000i-of-0000n.safetensors`, each
//! tensor's shard named by `model.safetensors.index.json`. A test of another file name writes it
//! with [`write_file`]. The files are composed by the safetensors format's own writer
//! (`safetensors::serialize`), so a test states tensors and never the format's grammar.
//!
//! A committed fixture (a tiny checkpoint, a reference's tensors) is embedded in the test binary
//! with [`fixture!`](crate::fixture), from paths relative to the test's own source file, and
//! written out to a temporary directory where the readers that open paths find it: the test
//! locates nothing at run time.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use nn::{Checkpoint, DType, Device, Tensor};
use safetensors::Dtype as FormatDtype;
use safetensors::tensor::TensorView;
use tempfile::TempDir;

/// One tensor a test writes: its name, element type, extents and little-endian bytes.
pub type Stored<N> = (N, DType, Vec<usize>, Vec<u8>);

/// The file a checkpoint of one shard holds its tensors in.
const SINGLE_SHARD: &str = "model.safetensors";

/// The index that names each tensor's shard in a checkpoint of several.
const SHARD_INDEX: &str = "model.safetensors.index.json";

/// The format's name for `dtype`: every element type `nn::Checkpoint` reads, each the format
/// type it reads as that element type.
fn format_dtype(dtype: DType) -> FormatDtype {
    match dtype {
        DType::U8 => FormatDtype::U8,
        DType::U32 => FormatDtype::U32,
        DType::I16 => FormatDtype::I16,
        DType::I32 => FormatDtype::I32,
        DType::I64 => FormatDtype::I64,
        DType::F16 => FormatDtype::F16,
        DType::BF16 => FormatDtype::BF16,
        DType::F32 => FormatDtype::F32,
        DType::F64 => FormatDtype::F64,
        DType::F8E4M3 => FormatDtype::F8_E4M3,
        DType::F4 => FormatDtype::F4,
        other => panic!("a checkpoint stores no {other:?} tensor"),
    }
}

/// Writes the safetensors file `path` holding `tensors`, with `metadata` as its header's
/// `__metadata__` block: what the file's writer states about it. A file of that path is
/// replaced.
///
/// A shape and a byte count that disagree are a defect of the test and fail here.
pub fn write_file<N: AsRef<str>>(path: &Path, metadata: &[(&str, &str)], tensors: &[Stored<N>]) {
    let views = tensors.iter().map(|(name, dtype, shape, bytes)| {
        let name = name.as_ref();
        let view =
            TensorView::new(format_dtype(*dtype), shape.clone(), bytes).unwrap_or_else(|error| {
                panic!(
                    "`{name}` states {shape:?} of {dtype:?} over {} bytes: {error}",
                    bytes.len()
                )
            });
        (name, view)
    });
    let metadata = (!metadata.is_empty()).then(|| {
        metadata
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    });

    let file = safetensors::serialize(views, metadata)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    std::fs::write(path, file).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
}

/// Writes the checkpoint of `shards` in `directory` as the format declares one: a single shard
/// is `model.safetensors`; several are `model-0000i-of-0000n.safetensors`, in order, listed by
/// `model.safetensors.index.json`.
pub fn write_checkpoint<N: AsRef<str>>(directory: &Path, shards: &[&[Stored<N>]]) {
    match shards {
        [] => panic!("a checkpoint of {} holds no shard", directory.display()),
        [shard] => write_file(&directory.join(SINGLE_SHARD), &[], shard),
        shards => {
            let mut weight_map = BTreeMap::new();

            for (index, shard) in shards.iter().enumerate() {
                let file = format!("model-{:05}-of-{:05}.safetensors", index + 1, shards.len());
                write_file(&directory.join(&file), &[], shard);

                for (name, ..) in shard.iter() {
                    weight_map.insert(name.as_ref().to_owned(), file.clone());
                }
            }

            let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
            let path = directory.join(SHARD_INDEX);
            std::fs::write(&path, index.to_string())
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        }
    }
}

/// `tensors`, each `(name, extents, values)`, stored as `dtype`: the values encoded as its
/// elements, rounded to nearest even.
pub fn stored<N: AsRef<str>>(
    dtype: DType,
    tensors: &[(N, Vec<usize>, Vec<f32>)],
) -> Vec<Stored<&str>> {
    tensors
        .iter()
        .map(|(name, shape, values)| {
            (
                name.as_ref(),
                dtype,
                shape.clone(),
                crate::encode(dtype, values),
            )
        })
        .collect()
}

/// The checkpoint of one shard in `directory` holding `tensors`, each `(name, extents, values)`
/// stored as `dtype` ([`stored`]), written and opened.
pub fn checkpoint<N: AsRef<str>>(
    directory: &Path,
    dtype: DType,
    tensors: &[(N, Vec<usize>, Vec<f32>)],
) -> Checkpoint {
    write_checkpoint(directory, &[&stored(dtype, tensors)]);

    Checkpoint::open(directory).unwrap_or_else(|error| panic!("{}: {error}", directory.display()))
}

/// Embeds the committed fixture files `files` of the directory `directory`, both relative to
/// the invoking source file, and writes them out as a [`Fixture`] under their relative paths.
///
/// ```text
/// let fixture = nn_testing::fixture!("data/tiny": ["model/config.json", "model/model.safetensors"]);
/// let checkpoint = fixture.checkpoint("model");
/// ```
///
/// The files are compiled into the test binary (`include_bytes!`), so a change to one rebuilds
/// the test and no run looks for the source tree.
#[macro_export]
macro_rules! fixture {
    ($directory:literal: [$($file:literal),+ $(,)?]) => {
        $crate::Fixture::write(&[
            $(($file, include_bytes!(concat!($directory, "/", $file)).as_slice())),+
        ])
    };
}

/// Committed fixture files written out to a temporary directory, which lives as long as the
/// value.
pub struct Fixture {
    directory: TempDir,
}

impl Fixture {
    /// Writes `files`, each `(relative path, contents)`, under a new temporary directory.
    ///
    /// [`fixture!`](crate::fixture) states the files a test embeds.
    pub fn write(files: &[(&str, &[u8])]) -> Self {
        let directory = tempfile::tempdir()
            .unwrap_or_else(|error| panic!("a directory for a fixture: {error}"));

        for (relative, contents) in files {
            let path = directory.path().join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .unwrap_or_else(|error| panic!("{}: {error}", parent.display()));
            }
            std::fs::write(&path, contents)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        }
        Self { directory }
    }

    /// The path of the fixture's file or directory `relative`.
    pub fn path(&self, relative: &str) -> PathBuf {
        self.directory.path().join(relative)
    }

    /// The checkpoint of the fixture's directory `relative`, opened by its declaration.
    pub fn checkpoint(&self, relative: &str) -> Checkpoint {
        let path = self.path(relative);
        Checkpoint::open(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    /// Every tensor of the fixture's safetensors file `relative`, by name, as a CPU tensor of
    /// the element type and extents the file states, read with the reader the families load
    /// with (`nn::Checkpoint::open_file`).
    pub fn tensors(&self, relative: &str) -> HashMap<String, Tensor> {
        let path = self.path(relative);
        let file = Checkpoint::open_file(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));

        file.names()
            .map(|name| {
                let read = file.locate(name).and_then(|location| {
                    let bytes = file.read(name)?;
                    Ok(Tensor::from_raw_buffer(
                        &bytes,
                        location.dtype,
                        location.shape.dims(),
                        &Device::Cpu,
                    )?)
                });
                let tensor =
                    read.unwrap_or_else(|error| panic!("{}: `{name}`: {error}", path.display()));

                (name.to_owned(), tensor)
            })
            .collect()
    }
}

/// The adapter container the tests declare, as a model family declares its own
/// (`nn::adapter::Format`): the adapter is a directory's `adapter.safetensors`, each tensor is
/// named `<parameter>.<role>` (`down`, `up`, `diff` or `assign`), and the header metadata states
/// what the file carries as `carries = "<low-rank tensors>,<diff tensors>,<assigned tensors>"`.
pub struct Stated;

impl nn::adapter::Format for Stated {
    fn file(&self) -> &'static str {
        "adapter.safetensors"
    }

    fn role(&self, name: &str) -> Option<(String, nn::adapter::Role)> {
        let (parameter, role) = name.rsplit_once('.')?;
        let role = match role {
            "down" => nn::adapter::Role::Down,
            "up" => nn::adapter::Role::Up,
            "diff" => nn::adapter::Role::Diff,
            "assign" => nn::adapter::Role::Assign,
            _ => return None,
        };
        Some((parameter.to_owned(), role))
    }

    fn check(
        &self,
        metadata: &BTreeMap<String, String>,
        contents: &nn::adapter::Contents,
    ) -> Result<(), String> {
        let carried = format!(
            "{},{},{}",
            contents.low_rank, contents.diffs, contents.assigned
        );
        match metadata.get("carries") {
            Some(stated) if *stated == carried => Ok(()),
            stated => Err(format!(
                "states it carries {stated:?} and carries {carried}"
            )),
        }
    }
}
