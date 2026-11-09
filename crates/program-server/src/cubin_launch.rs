//! The plan of one launch of an installed kernel module, read from a recipe.
//!
//! Three packages in the tree are a module with no launcher of their own — `topk_softmax`'s
//! `nvcc` cubin, and `paged_attention`'s and `gather_gemm_nvfp4`'s `trtllm` cubins — so their
//! arithmetic cannot be observed by any script: observing it needs a launch, and a launch needs
//! operator code those packages do not have. The `cubin-launch` binary is that launch, stated as
//! data rather than as code. This module is the part of it that reads that data and checks it
//! against the module it names; it touches no device, so it is exercised on a host without one.
//!
//! ```json
//! {
//!   "module": "path/to/module-directory",
//!   "cubin": "module.cubin",
//!   "entry": "uniserve_topk_softmax",
//!   "grid": [4, 1, 1],
//!   "buffers": {
//!     "scores": {"bytes": 65536, "fill": "scores.bin", "dump": "scores.bin"},
//!     "ids": {"bytes": 16384, "dump": "ids.bin"},
//!     "weights": {"bytes": 65536, "dump": "weights.bin"}
//!   },
//!   "parameters": [
//!     {"pointer": "scores"}, {"i32": 1}, {"i64": 5120}, {"i32": 4},
//!     {"baked": "experts"}, {"baked": "top"}, {"i32": 1},
//!     {"pointer": "scores"}, {"i32": 1}, {"pointer": "ids"}, {"pointer": "weights"}
//!   ]
//! }
//! ```
//!
//! `parameters` are the kernel's words in declaration order: `pointer` is a buffer's device
//! address, `i32`/`u32`/`i64`/`u64`/`f32` a scalar, `baked` a constant from the module's own
//! `baked.json` (so a case hands the kernel the values it was built for rather than values a
//! caller guessed), and `bytes` an inline blob — how a structure passed by value is one packed
//! pointer in the parameter array.
//!
//! The block shape, dynamic shared memory, cluster, programmatic dependent launch and
//! cooperative launch come from the module's `metadata.json`, which is where the loader reads
//! them: the recipe states only what the module cannot, which is the grid and the words.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One case's launch, as its recipe states it.
#[derive(Debug)]
pub struct Recipe {
    /// The recipe file this was read from, absolute: the directory a relative `module` was resolved
    /// against, and the directory a run's files are written under when it states none.
    origin: PathBuf,
    /// The module directory: `module.cubin` (or `cubin`), `metadata.json`, and `baked.json`.
    ///
    /// Absolute, resolved against the recipe's own directory when the recipe states it relatively.
    /// It is resolved at read time because the recipe path is the only thing that can say what a
    /// relative `module` means: a reader that took the path again could be handed a different one -
    /// a bare file name, whose parent is the empty path - and would then look for the module
    /// relative to the process's working directory.
    pub module: PathBuf,
    /// The cubin's file name inside the module directory.
    pub cubin: String,
    /// The kernel inside the cubin.
    pub entry: String,
    /// Blocks per grid dimension; the module's metadata states the block.
    pub grid: [u32; 3],
    /// The device the module is loaded on.
    pub device: i32,
    /// Device storage the launch reads and writes, by name.
    pub buffers: BTreeMap<String, BufferPlan>,
    /// The kernel's words in declaration order.
    pub parameters: Vec<Parameter>,
}

impl Recipe {
    /// Reads one recipe file.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
        let module = text_of(&value, "module")
            .ok_or_else(|| format!("{}: a recipe names its module directory", path.display()))?;
        let entry = text_of(&value, "entry")
            .ok_or_else(|| format!("{}: a recipe names the entry to launch", path.display()))?;
        let origin =
            std::path::absolute(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let stated = PathBuf::from(module);
        let module = match stated.is_absolute() {
            true => stated,
            false => origin
                .parent()
                .map(|parent| parent.join(&stated))
                .unwrap_or(stated),
        };
        Ok(Self {
            origin,
            module,
            cubin: text_of(&value, "cubin")
                .unwrap_or("module.cubin")
                .to_owned(),
            entry: entry.to_owned(),
            grid: read_grid(&value)?,
            device: value
                .get("device")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0) as i32,
            buffers: BufferPlan::read_all(value.get("buffers"))?,
            parameters: Parameter::read_all(value.get("parameters"))?,
        })
    }

    /// The module directory, absolute and resolved when this recipe was read.
    pub fn module_dir(&self) -> PathBuf {
        self.module.clone()
    }

    /// The directory a buffer's `dump` is written under when the run states none.
    pub fn dump_dir(&self) -> PathBuf {
        self.origin
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

fn text_of<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(serde_json::Value::as_str)
}

fn read_grid(value: &serde_json::Value) -> Result<[u32; 3], String> {
    let Some(grid) = value.get("grid") else {
        return Ok([1, 1, 1]);
    };
    let dimensions = grid
        .as_array()
        .ok_or("a grid is an array of one to three block counts")?;
    if dimensions.is_empty() || dimensions.len() > 3 {
        return Err(format!(
            "a grid has one to three dimensions, got {}",
            dimensions.len()
        ));
    }
    let mut resolved = [1u32; 3];
    for (axis, dimension) in dimensions.iter().enumerate() {
        let blocks = dimension
            .as_u64()
            .ok_or("a grid dimension is a block count")?;
        if blocks == 0 {
            return Err("a grid dimension is at least one block".to_owned());
        }
        resolved[axis] = u32::try_from(blocks)
            .map_err(|_| format!("{blocks} blocks do not fit a grid dimension"))?;
    }
    Ok(resolved)
}

/// One device buffer the launch reads or writes.
#[derive(Debug)]
pub struct BufferPlan {
    /// The bytes the buffer holds.
    pub bytes: usize,
    /// A host file copied in before the launch.
    pub fill: Option<PathBuf>,
    /// A host file the buffer is copied to after the launch.
    pub dump: Option<PathBuf>,
}

impl BufferPlan {
    fn read_all(value: Option<&serde_json::Value>) -> Result<BTreeMap<String, Self>, String> {
        let Some(value) = value else {
            return Ok(BTreeMap::new());
        };
        let entries = value
            .as_object()
            .ok_or("buffers is an object of names to plans")?;
        let mut plans = BTreeMap::new();
        for (name, plan) in entries {
            let bytes = plan
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("buffer {name} states its byte count"))?;
            let bytes = usize::try_from(bytes)
                .map_err(|_| format!("buffer {name} is larger than this host addresses"))?;
            if bytes == 0 {
                return Err(format!("buffer {name} is empty"));
            }
            plans.insert(
                name.clone(),
                Self {
                    bytes,
                    fill: plan
                        .get("fill")
                        .and_then(serde_json::Value::as_str)
                        .map(PathBuf::from),
                    dump: plan
                        .get("dump")
                        .and_then(serde_json::Value::as_str)
                        .map(PathBuf::from),
                },
            );
        }
        Ok(plans)
    }
}

/// One word of the kernel's parameter array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parameter {
    /// A buffer's device address: one pointer-sized word.
    Pointer(String),
    /// A scalar word of the stated width, in native byte order.
    Word { bytes: Vec<u8>, kind: String },
    /// A value the module was built with, from its `baked.json`.
    Baked(String),
    /// An inline blob, held where the parameter array points.
    Inline(Vec<u8>),
}

impl Parameter {
    fn read_all(value: Option<&serde_json::Value>) -> Result<Vec<Self>, String> {
        let Some(value) = value else {
            return Ok(Vec::new());
        };
        let words = value
            .as_array()
            .ok_or("parameters is an array of words in declaration order")?;
        words.iter().map(Self::read).collect()
    }

    fn read(word: &serde_json::Value) -> Result<Self, String> {
        let object = word
            .as_object()
            .ok_or("a parameter is an object naming its kind")?;
        if object.len() != 1 {
            return Err(format!(
                "a parameter names exactly one kind, got {}",
                object.len()
            ));
        }
        // The length was checked to be one, so the iteration answers exactly one entry.
        let (kind, value) = match object.iter().next() {
            Some(entry) => entry,
            None => return Err("a parameter names exactly one kind, got none".to_owned()),
        };
        match kind.as_str() {
            "pointer" => Ok(Self::Pointer(
                value.as_str().ok_or("a pointer names a buffer")?.to_owned(),
            )),
            "i32" => Ok(Self::Word {
                bytes: signed(value, "i32")?.to_ne_bytes()[..4].to_vec(),
                kind: "i32".to_owned(),
            }),
            "i64" => Ok(Self::Word {
                bytes: signed(value, "i64")?.to_ne_bytes().to_vec(),
                kind: "i64".to_owned(),
            }),
            "u32" => Ok(Self::Word {
                bytes: unsigned(value, "u32")?.to_ne_bytes()[..4].to_vec(),
                kind: "u32".to_owned(),
            }),
            "u64" => Ok(Self::Word {
                bytes: unsigned(value, "u64")?.to_ne_bytes().to_vec(),
                kind: "u64".to_owned(),
            }),
            "f32" => {
                let number = value.as_f64().ok_or("f32 takes a number")?;
                Ok(Self::Word {
                    bytes: (number as f32).to_ne_bytes().to_vec(),
                    kind: "f32".to_owned(),
                })
            }
            "baked" => Ok(Self::Baked(
                value.as_str().ok_or("baked names a key")?.to_owned(),
            )),
            "bytes" => Ok(Self::Inline(decode_hex(
                value.as_str().ok_or("bytes takes a hex string")?,
            )?)),
            other => Err(format!(
                "{other} is not a parameter kind: pointer, i32, u32, i64, u64, f32, baked, bytes"
            )),
        }
    }
}

fn signed(value: &serde_json::Value, kind: &str) -> Result<i64, String> {
    value
        .as_i64()
        .ok_or_else(|| format!("{kind} takes an integer"))
}

fn unsigned(value: &serde_json::Value, kind: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{kind} takes a non-negative integer"))
}

fn decode_hex(text: &str) -> Result<Vec<u8>, String> {
    let trimmed = text.trim();
    if !trimmed.len().is_multiple_of(2) {
        return Err("bytes holds an even number of hex digits".to_owned());
    }
    (0..trimmed.len())
        .step_by(2)
        .map(|at| {
            u8::from_str_radix(&trimmed[at..at + 2], 16).map_err(|error| format!("bytes: {error}"))
        })
        .collect()
}

/// A module's `metadata.json`: the launch configuration it declares and, when the package
/// publishes one, its parameter list.
#[derive(Debug, Default)]
pub struct Metadata {
    /// The kernel's symbol, as the module states it.
    pub name: Option<String>,
    /// The warps one block holds.
    pub num_warps: u32,
    /// The thread-block cluster the grid groups into, in blocks.
    pub num_ctas: u32,
    /// Dynamic shared memory per block, in bytes.
    pub shared: u32,
    /// Programmatic dependent launch.
    pub launch_pdl: bool,
    /// Cooperative launch.
    pub launch_cooperative_grid: bool,
    /// The parameters the source was compiled from, in declaration order.
    pub signature: Option<Vec<(String, String)>>,
}

impl Metadata {
    /// Reads a module's `metadata.json`.
    pub fn read(module: &Path) -> Result<Self, String> {
        let path = module.join("metadata.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
        let number = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32
        };
        let flag = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        };
        Ok(Self {
            name: text_of(&value, "name").map(str::to_owned),
            num_warps: number("num_warps"),
            num_ctas: number("num_ctas"),
            shared: number("shared"),
            launch_pdl: flag("launch_pdl"),
            launch_cooperative_grid: flag("launch_cooperative_grid"),
            signature: value.get("signature").and_then(|signature| {
                signature.as_object().map(|entries| {
                    entries
                        .iter()
                        .map(|(name, kind)| (name.clone(), kind.as_str().unwrap_or("").to_owned()))
                        .collect()
                })
            }),
        })
    }

    /// The parameters the module launches with: the signature without its constexprs.
    pub fn words(&self) -> Option<Vec<(String, String)>> {
        self.signature.as_ref().map(|signature| {
            signature
                .iter()
                .filter(|(_, kind)| kind != "constexpr")
                .cloned()
                .collect()
        })
    }
}

/// The constants a module was built with, where its package writes them.
#[derive(Debug, Default, Clone)]
pub struct Baked {
    values: BTreeMap<String, i64>,
}

impl Baked {
    /// Reads a module's `baked.json`, or nothing when it holds none.
    pub fn read(module: &Path) -> Result<Self, String> {
        let path = module.join("baked.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("{}: baked constants are an object", path.display()))?;
        Ok(Self {
            values: object
                .iter()
                .filter_map(|(key, entry)| entry.as_i64().map(|number| (key.clone(), number)))
                .collect(),
        })
    }

    /// One constant, as the `i32` word a parameter carries.
    pub fn i32(&self, key: &str) -> Result<i32, String> {
        let value = self
            .values
            .get(key)
            .ok_or_else(|| format!("baked.json holds no {key}"))?;
        i32::try_from(*value).map_err(|_| format!("baked {key} = {value} is not an i32"))
    }

    /// Every constant, for the run's own record.
    pub fn all(&self) -> &BTreeMap<String, i64> {
        &self.values
    }
}

/// One parameter word, ready to point at.
#[derive(Clone, Debug)]
pub struct Word {
    bytes: Vec<u8>,
    kind: String,
    buffer: Option<String>,
}

impl Word {
    /// The bytes the word occupies in the parameter array.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the word occupies no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// What the word is, for the run's record: a scalar's kind, a baked constant's name, or the
    /// buffer a pointer addresses.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The buffer this word addresses, when it is a pointer.
    pub fn buffer(&self) -> Option<&str> {
        self.buffer.as_deref()
    }

    /// The word's bytes.
    #[cfg(test)]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The pointer the parameter array carries for this word: its bytes' own address.
    pub fn parameter(&mut self) -> *mut std::ffi::c_void {
        self.bytes.as_mut_ptr().cast()
    }

    /// Fills a pointer word with the address it addresses.
    pub fn set_address(&mut self, address: u64) {
        if self.buffer.is_some() {
            self.bytes.copy_from_slice(&address.to_ne_bytes());
        }
    }
}

/// One launch, planned against its module but not yet performed.
#[derive(Debug)]
pub struct Launch {
    /// The parameter words, in declaration order.
    pub words: Vec<Word>,
    /// Blocks per grid dimension.
    pub grid: [u32; 3],
    /// Threads per block dimension, from the module's warp count.
    pub block: [u32; 3],
    /// Dynamic shared memory per block, in bytes.
    pub shared: u32,
    /// The thread-block cluster, when the module declares more than one CTA.
    pub cluster: Option<[u32; 3]>,
    /// Programmatic dependent launch.
    pub pdl: bool,
    /// Cooperative launch.
    pub cooperative: bool,
    /// The entry the recipe names.
    pub entry: String,
}

impl Launch {
    /// Turns a recipe into a launch: the block shape and launch attributes from the module's
    /// metadata, the baked constants from the module's build, the words from the recipe.
    pub fn plan(recipe: &Recipe, metadata: &Metadata, baked: &Baked) -> Result<Self, String> {
        let warps = if metadata.num_warps == 0 {
            4
        } else {
            metadata.num_warps
        };
        let ctas = metadata.num_ctas.max(1);
        let mut words = Vec::with_capacity(recipe.parameters.len());
        for parameter in &recipe.parameters {
            words.push(match parameter {
                Parameter::Pointer(name) => {
                    if !recipe.buffers.contains_key(name) {
                        return Err(format!("parameter points at no buffer {name}"));
                    }
                    Word {
                        bytes: vec![0; 8],
                        kind: format!("pointer {name}"),
                        buffer: Some(name.clone()),
                    }
                }
                Parameter::Word { bytes, kind } => Word {
                    bytes: bytes.clone(),
                    kind: kind.clone(),
                    buffer: None,
                },
                Parameter::Baked(key) => Word {
                    bytes: baked.i32(key)?.to_ne_bytes().to_vec(),
                    kind: format!("baked {key}"),
                    buffer: None,
                },
                Parameter::Inline(bytes) => Word {
                    bytes: bytes.clone(),
                    kind: format!("{} inline bytes", bytes.len()),
                    buffer: None,
                },
            });
        }
        Ok(Self {
            words,
            grid: recipe.grid,
            block: [warps * 32, 1, 1],
            shared: metadata.shared,
            cluster: (ctas > 1).then_some([ctas, 1, 1]),
            pdl: metadata.launch_pdl,
            cooperative: metadata.launch_cooperative_grid,
            entry: recipe.entry.clone(),
        })
    }

    /// Refuses a word array the module's published signature does not describe.
    ///
    /// A cubin has no other way to say so: the driver reads the array per the kernel's own
    /// signature, so a wrong array is mis-packed silently. A module whose package publishes no
    /// signature is launched unchecked, and the run says that — the absence is the export
    /// contract's own gap.
    pub fn check_signature(&self, metadata: &Metadata) -> Result<bool, String> {
        let Some(declared) = metadata.words() else {
            return Ok(false);
        };
        if self.words.len() != declared.len() {
            return Err(format!(
                "the module declares {} parameters ({}), the recipe gives {} ({})",
                declared.len(),
                declared
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                self.words.len(),
                self.words
                    .iter()
                    .map(|word| word.kind())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        for ((name, kind), word) in declared.iter().zip(&self.words) {
            let expected = width_of(kind);
            if expected != word.len() {
                return Err(format!(
                    "parameter {name} is {expected} bytes in the module's signature ({kind}), \
                     the recipe's word is {} bytes ({})",
                    word.len(),
                    word.kind()
                ));
            }
        }
        Ok(true)
    }

    /// The one line a run prints before it launches, so the launch is on the record whether or
    /// not the kernel turns out to be correct.
    pub fn describe(&self, recipe: &Recipe, metadata: &Metadata, baked: &Baked) -> String {
        let words = self
            .words
            .iter()
            .enumerate()
            .map(|(index, word)| format!("{index}:{}={}B", word.kind(), word.len()))
            .collect::<Vec<_>>()
            .join(" ");
        let baked = baked
            .all()
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        let buffers = recipe
            .buffers
            .iter()
            .map(|(name, plan)| format!("{name}={}", plan.bytes))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "module {} entry {} symbol {} grid {:?} block {:?} shared {} cluster {:?} pdl {} \
             cooperative {} words [{}] baked [{}] buffers [{}]",
            recipe.module.display(),
            self.entry,
            metadata.name.as_deref().unwrap_or("-"),
            self.grid,
            self.block,
            self.shared,
            self.cluster,
            self.pdl,
            self.cooperative,
            words,
            baked,
            buffers
        )
    }
}

/// The bytes one signature kind occupies in the parameter array.
fn width_of(kind: &str) -> usize {
    match kind {
        "i1" | "i8" | "u8" => 1,
        "i16" | "u16" | "f16" | "bf16" => 2,
        "i32" | "u32" | "f32" => 4,
        "i64" | "u64" | "f64" => 8,
        // Every pointer Triton spells starts with `*`; an unknown kind is taken as one word and
        // the run reports the kind it assumed.
        _ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recipe and its module on disk, and the directory that holds them.
    ///
    /// The directory is returned because a `TempDir` deletes its files when it is dropped: a
    /// caller that lets it fall here reads a recipe whose module no longer exists, and the baked
    /// constants then read as an empty set rather than as a missing file.
    fn plan_of(
        recipe: serde_json::Value,
        metadata: serde_json::Value,
    ) -> (tempfile::TempDir, Recipe, Metadata) {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let recipe_path = directory.path().join("case.json");
        std::fs::write(&recipe_path, recipe.to_string()).expect("the recipe is written");
        let module = directory.path().join("module");
        std::fs::create_dir(&module).expect("the module directory is made");
        std::fs::write(module.join("metadata.json"), metadata.to_string())
            .expect("the metadata is written");
        std::fs::write(module.join("baked.json"), "{\"experts\": 128, \"top\": 8}")
            .expect("the baked constants are written");
        (
            directory,
            Recipe::read(&recipe_path).expect("the recipe reads"),
            Metadata::read(&module).expect("the metadata reads"),
        )
    }

    #[test]
    fn a_relative_module_resolves_against_the_recipes_own_directory() {
        // The module a recipe names relatively is a directory beside the recipe. It is resolved
        // when the recipe is read, and nowhere else: a reader that passed the path again could
        // pass a bare file name, whose parent is the empty path, and the module would then be
        // sought under the process's working directory. That is a directory, so the constants
        // read as an empty set rather than as a missing file.
        let (directory, recipe, _metadata) = plan_of(
            serde_json::json!({"module": "module", "entry": "entry"}),
            serde_json::json!({}),
        );

        assert_eq!(recipe.module_dir(), directory.path().join("module"));
        assert!(recipe.module_dir().is_absolute());
        assert_eq!(recipe.dump_dir(), directory.path());
        let baked =
            Baked::read(&recipe.module_dir()).expect("the constants beside the recipe read");
        assert_eq!(baked.i32("experts"), Ok(128));
    }

    #[test]
    fn an_absolute_module_is_taken_as_it_stands() {
        let module = tempfile::tempdir().expect("a temporary module");
        std::fs::write(module.path().join("metadata.json"), "{}").expect("the metadata is written");
        let recipe_directory = tempfile::tempdir().expect("a temporary recipe directory");
        let recipe_path = recipe_directory.path().join("case.json");
        std::fs::write(
            &recipe_path,
            serde_json::json!({
                "module": module.path().to_str().expect("a temporary path is text"),
                "entry": "entry",
            })
            .to_string(),
        )
        .expect("the recipe is written");

        let recipe = Recipe::read(&recipe_path).expect("the recipe reads");

        assert_eq!(recipe.module_dir(), module.path());
        assert_eq!(recipe.dump_dir(), recipe_directory.path());
    }

    #[test]
    fn a_baked_constant_reaches_the_word_the_recipe_names() {
        let (_directory, recipe, metadata) = plan_of(
            serde_json::json!({
                "module": "module",
                "entry": "uniserve_topk_softmax",
                "grid": [4, 1, 1],
                "buffers": {"scores": {"bytes": 4096}},
                "parameters": [{"pointer": "scores"}, {"baked": "experts"}],
            }),
            serde_json::json!({"num_warps": 4}),
        );
        let baked = Baked::read(&recipe.module_dir()).expect("the baked constants read");
        let launch = Launch::plan(&recipe, &metadata, &baked).expect("the launch plans");
        assert_eq!(launch.words[0].len(), 8);
        assert_eq!(launch.words[0].buffer(), Some("scores"));
        assert_eq!(launch.words[1].bytes(), 128i32.to_ne_bytes());
        assert_eq!(launch.block, [128, 1, 1]);
        assert_eq!(launch.cluster, None);
    }

    #[test]
    fn a_word_array_the_signature_does_not_describe_is_refused() {
        let (_directory, recipe, metadata) = plan_of(
            serde_json::json!({
                "module": "module",
                "entry": "uniserve_topk_softmax",
                "buffers": {"scores": {"bytes": 4096}},
                "parameters": [{"pointer": "scores"}, {"i64": 5120}],
            }),
            serde_json::json!({
                "num_warps": 4,
                "signature": {"scores": "*bf16", "rows": "i32", "wide": "constexpr"},
            }),
        );
        let baked = Baked::read(&recipe.module_dir()).expect("the baked constants read");
        let launch = Launch::plan(&recipe, &metadata, &baked).expect("the launch plans");
        let refused = launch.check_signature(&metadata);
        assert!(
            refused.is_err(),
            "an i64 where the module declares an i32 must be refused: {refused:?}"
        );
    }

    #[test]
    fn a_module_without_a_signature_is_launched_unchecked_and_says_so() {
        let (_directory, recipe, metadata) = plan_of(
            serde_json::json!({
                "module": "module",
                "entry": "uniserve_topk_softmax",
                "parameters": [{"i32": 1}],
            }),
            serde_json::json!({"num_warps": 4}),
        );
        let baked = Baked::read(&recipe.module_dir()).expect("the baked constants read");
        let launch = Launch::plan(&recipe, &metadata, &baked).expect("the launch plans");
        assert_eq!(
            launch.check_signature(&metadata),
            Ok(false),
            "the absence of a signature is reported, not assumed"
        );
    }

    #[test]
    fn a_pointer_to_no_buffer_is_refused_before_anything_is_allocated() {
        let (_directory, recipe, metadata) = plan_of(
            serde_json::json!({
                "module": "module",
                "entry": "uniserve_topk_softmax",
                "buffers": {"scores": {"bytes": 4096}},
                "parameters": [{"pointer": "ids"}],
            }),
            serde_json::json!({"num_warps": 4}),
        );
        let baked = Baked::default();
        assert!(Launch::plan(&recipe, &metadata, &baked).is_err());
    }

    #[test]
    fn a_structure_passed_by_value_is_one_word_of_its_own_bytes() {
        let (_directory, recipe, metadata) = plan_of(
            serde_json::json!({
                "module": "module",
                "entry": "uniserve_moe_sort",
                "buffers": {"tokens": {"bytes": 1024}},
                "parameters": [{"bytes": "0100000002000000"}],
            }),
            serde_json::json!({"num_warps": 4, "num_ctas": 2, "shared": 1536, "launch_pdl": true}),
        );
        let launch = Launch::plan(&recipe, &metadata, &Baked::default()).expect("the launch plans");
        assert_eq!(launch.words[0].len(), 8);
        assert_eq!(launch.words[0].bytes(), &[1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(launch.shared, 1536);
        assert_eq!(launch.cluster, Some([2, 1, 1]));
        assert!(launch.pdl);
    }
}
