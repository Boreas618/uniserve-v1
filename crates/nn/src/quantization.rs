//! What a calibrated checkpoint declares about its own numeric representation, and the method
//! each layer reads a weight with.
//!
//! A quantized checkpoint states it in its metadata, not in its tensors: ModelOpt writes
//! `quantization_config` into `config.json` (as an algorithm name, or as a compressed-tensors
//! `config_groups` block) or a sibling `hf_quant_config.json`, and each names the modules the
//! calibration left in their source dtype and the scheme the key/value cache scales follow.
//! [`Declaration::read`] is the one place those fields are read; a representation this library
//! cannot serve is refused by name before any tensor is read.
//!
//! ## The shard decides
//!
//! A declaration states what a calibration intended; the shards state what it did. The two can
//! differ — an exclusion list that names a module the export then packed, or a declaration that
//! quantizes a vocabulary matrix the shard holds in BF16 — and the tensor is the fact.
//! [`Declaration::resolve`] records which weights the shards store packed (a `weight_scale` beside
//! the weight), and [`Declaration::scheme`] answers from that record, falling back to the
//! declaration only when no shard was consulted.
//!
//! ## The method follows the layer
//!
//! There is no table of roles: a layer asks for its weight through [`crate::weights::Scope`], and
//! the layer's type decides what a packed weight means. A [`crate::layers::Linear`] reads a packed
//! weight as NVFP4 (an activation encoder and a packed product); every other layer reads dense
//! weights only, so a packed tensor reaching it is refused by name rather than read as a matrix.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::checkpoint::Checkpoint;
use crate::config::Fields;
use crate::error::{Error, Result};

/// The declaration's own file, when the export writes one beside `config.json`.
const DECLARATION_FILE: &str = "hf_quant_config.json";

/// How one weight is stored, and so how the layer that owns it reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scheme {
    /// The checkpoint's own floating-point representation.
    Dense,
    /// ModelOpt NVFP4: `weight` (packed E2M1), `weight_scale` (K16 block scales in `F8_E4M3`),
    /// `weight_scale_2` (one tensor scale) and `input_scale` (the calibration's activation scale).
    Nvfp4,
}

/// The scheme a checkpoint's key/value cache scales follow.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvCache {
    /// The cache is stored at the activations' own type; no scale is calibrated.
    Unquantized,
    /// FP8 E4M3 with calibrated per-layer key and value scales.
    Fp8,
}

/// What a checkpoint declares about the representation of its weights.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Declaration {
    /// The scheme the calibration applies to the linear projections it covers.
    scheme: Scheme,
    /// The modules the calibration left in their source dtype, as `config.json` writes them.
    excluded: Vec<String>,
    /// The scheme the key/value cache scales follow.
    kv_cache: KvCache,
    /// The weights whose shards store a block scale beside them, once [`Self::resolve`] has read
    /// the shards; `None` before, when the declaration alone answers.
    packed: Option<BTreeSet<String>>,
}

impl Declaration {
    /// A checkpoint that declares nothing: every weight is in its own dtype.
    pub fn dense() -> Self {
        Self {
            scheme: Scheme::Dense,
            excluded: Vec::new(),
            kv_cache: KvCache::Unquantized,
            packed: None,
        }
    }

    /// Read the declaration of `checkpoint`, from its `config.json` and its export's own file.
    ///
    /// A checkpoint that declares nothing is [`Self::dense`]: the absence of a declaration is a
    /// statement about the checkpoint (nothing was quantized), not a gap to fill with a guess. Two
    /// files that declare different algorithms are refused; two exclusion lists need not agree,
    /// because the shards decide every weight once [`Self::resolve`] has read them.
    ///
    /// # Errors
    ///
    /// When a file states an algorithm, a method, a group or a cache scheme this library does not
    /// serve, or the two files state different algorithms.
    pub fn read(checkpoint: &Checkpoint) -> Result<Self> {
        let from_config = match checkpoint.sidecar("config.json")? {
            Some(config) => match config.get("quantization_config") {
                Some(value) if !value.is_null() => Some(Declared::of(&Fields::of(
                    "config.json.quantization_config",
                    value,
                )?)?),
                _ => None,
            },
            None => None,
        };
        let from_side = match checkpoint.sidecar(DECLARATION_FILE)? {
            Some(side) => {
                // The export nests its fields under `quantization`; a file that states them at
                // the top level is read the same way, because both spellings are in circulation.
                let nested = side.get("quantization").unwrap_or(&side);
                let fields = Fields::of(format!("{DECLARATION_FILE}.quantization"), nested)?;
                if fields.contains("quant_algo") || fields.contains("quant_method") {
                    let mut declared = Declared::of(&fields)?;
                    // ModelOpt names itself as the producer rather than as a method.
                    if declared.method.is_none() {
                        declared.method = side
                            .get("producer")
                            .and_then(|producer| producer.get("name"))
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                    }
                    Some(declared)
                } else {
                    None
                }
            }
            None => None,
        };
        let declared = match (from_config, from_side) {
            (None, None) => return Ok(Self::dense()),
            (Some(declared), None) | (None, Some(declared)) => declared,
            (Some(config), Some(side)) => {
                if config.algorithm != side.algorithm {
                    return Err(refuse(
                        "the checkpoint",
                        format!(
                            "declares two different quantizations: `config.json` says {} and \
                             `{DECLARATION_FILE}` says {}; a checkpoint written by two tools is \
                             one neither of them owns",
                            config.algorithm, side.algorithm
                        ),
                    ));
                }
                config
            }
        };
        declared.declaration()
    }

    /// This declaration with every weight's packing taken from what `checkpoint`'s shards hold: a
    /// weight is packed when the shard stores a `weight_scale` beside it.
    pub fn resolve(&self, checkpoint: &Checkpoint) -> Self {
        if self.scheme == Scheme::Dense {
            return self.clone();
        }
        let packed = checkpoint
            .names()
            .filter_map(|name| name.strip_suffix("_scale"))
            .filter(|weight| weight.ends_with("weight"))
            .map(str::to_owned)
            .collect();
        Self {
            packed: Some(packed),
            ..self.clone()
        }
    }

    /// How the checkpoint stores `weight`, a weight's full checkpoint name.
    ///
    /// The shards' answer once [`Self::resolve`] has read them; the declaration's intent (packed
    /// unless an excluded module owns it) before.
    pub fn scheme(&self, weight: &str) -> Scheme {
        if self.scheme == Scheme::Dense {
            return Scheme::Dense;
        }
        let packed = match &self.packed {
            Some(packed) => packed.contains(weight),
            None => !self
                .excluded
                .iter()
                .any(|module| matches_module(module, weight)),
        };
        if packed { self.scheme } else { Scheme::Dense }
    }

    /// Whether any weight is declared packed: whether the encoded programs are the ones that run.
    pub fn is_encoded(&self) -> bool {
        self.scheme != Scheme::Dense
    }

    /// The scheme the key/value cache scales follow.
    pub fn kv_cache(&self) -> KvCache {
        self.kv_cache
    }
}

/// A refusal of `what` a declaration states.
fn refuse(what: &str, problem: impl Into<String>) -> Error {
    Error::Config {
        owner: what.to_owned(),
        field: String::new(),
        problem: problem.into(),
    }
}

/// One file's declaration fields, before they are checked as a whole.
struct Declared {
    /// `quant_algo`, or the algorithm a compressed-tensors group states.
    algorithm: String,
    /// `quant_method` (`modelopt`, `compressed-tensors`, …), or the producing tool's name.
    method: Option<String>,
    /// The modules the calibration left in their source dtype.
    excluded: Vec<String>,
    /// The key/value cache scheme the file states.
    kv_cache: KvCache,
    /// Where the fields were read, for refusals.
    owner: String,
}

impl Declared {
    /// The fields `fields` states.
    fn of(fields: &Fields<'_>) -> Result<Self> {
        let excluded: Vec<String> = fields
            .optional_texts("exclude_modules")?
            .unwrap_or_default()
            .into_iter()
            .map(str::to_owned)
            .collect();
        // The compressed-tensors spelling states the same quantization as a group instead of an
        // algorithm name, its excluded modules under `ignore`, and the cache under
        // `kv_cache_scheme`.
        let (algorithm, ignored) = match fields.optional_text("quant_algo")? {
            Some(algorithm) => (algorithm.to_owned(), Vec::new()),
            None => match fields.optional_object("config_groups")? {
                Some(groups) => (
                    group_algorithm(&groups)?,
                    fields
                        .optional_texts("ignore")?
                        .unwrap_or_default()
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                ),
                None => (String::new(), Vec::new()),
            },
        };
        if !ignored.is_empty() && !excluded.is_empty() {
            return Err(fields.refuse(
                "ignore",
                "is stated beside `exclude_modules`, and a file that says the same thing twice in \
                 two spellings is one this loader cannot trust",
            ));
        }
        let kv_cache = match fields.optional_text("kv_cache_quant_algo")? {
            None => match fields.optional_object("kv_cache_scheme")? {
                None => KvCache::Unquantized,
                Some(scheme) => match (
                    scheme.optional_index("num_bits")?,
                    scheme.optional_text("type")?,
                ) {
                    (Some(8), Some("float")) => KvCache::Fp8,
                    (bits, kind) => {
                        return Err(fields.refuse(
                            "kv_cache_scheme",
                            format!(
                                "states {bits:?} bits of {kind:?}, and the cache scales served \
                                 are FP8"
                            ),
                        ));
                    }
                },
            },
            Some(algorithm) if algorithm.eq_ignore_ascii_case("FP8") => KvCache::Fp8,
            Some(other) => {
                return Err(fields.refuse(
                    "kv_cache_quant_algo",
                    format!("names `{other}`, and the cache scales served are FP8"),
                ));
            }
        };
        Ok(Self {
            algorithm,
            method: fields.optional_text("quant_method")?.map(str::to_owned),
            excluded: if excluded.is_empty() {
                ignored
            } else {
                excluded
            },
            kv_cache,
            owner: fields.owner().to_owned(),
        })
    }

    /// The declaration these fields state, refused when this library cannot serve it.
    fn declaration(self) -> Result<Declaration> {
        let scheme = match self.algorithm.to_ascii_uppercase().as_str() {
            // Absent, empty or an explicit "none" is a dense checkpoint that wrote the field anyway.
            "" | "NONE" | "BF16" => Scheme::Dense,
            "NVFP4" => {
                if let Some(method) = &self.method
                    && !method.eq_ignore_ascii_case("modelopt")
                {
                    return Err(refuse(
                        &self.owner,
                        format!(
                            "states the quantization method `{method}`, and NVFP4 is served in \
                             ModelOpt's `weight`, `weight_scale`, `weight_scale_2` and \
                             `input_scale` spelling; a `{method}` export states the same numbers \
                             under names this loader would not find"
                        ),
                    ));
                }
                Scheme::Nvfp4
            }
            other => {
                return Err(refuse(
                    &self.owner,
                    format!(
                        "declares quantization `{other}`, which is not served: NVFP4 (ModelOpt's \
                         spelling) and an unquantized checkpoint are. Loading `{other}` as if it \
                         were dense would read quantized storage as a matrix"
                    ),
                ));
            }
        };
        Ok(Declaration {
            scheme,
            excluded: self.excluded,
            kv_cache: self.kv_cache,
            packed: None,
        })
    }
}

/// The algorithm a compressed-tensors `config_groups` block states, refused when it is not served.
///
/// The block names a bit width, a kind and a block size, and those three *are* the algorithm: four
/// bits of a float over a block of sixteen is NVFP4. The groups must target linear projections,
/// the only module class a packed weight is read for.
fn group_algorithm(groups: &Fields<'_>) -> Result<String> {
    let mut algorithm: Option<&str> = None;
    for name in groups.keys() {
        let group = groups.object(name)?;
        let weights = group.object("weights")?;
        let this = match (
            weights.optional_index("num_bits")?,
            weights.optional_text("type")?,
            weights.optional_index("group_size")?,
        ) {
            (Some(4), Some("float"), Some(16)) => "NVFP4",
            (bits, kind, block) => {
                return Err(refuse(
                    group.owner(),
                    format!(
                        "states {bits:?} bits of {kind:?} over a block of {block:?}, and this \
                         tree serves NVFP4: four bits of E2M1 over a block of sixteen with an \
                         F8_E4M3 scale. Reading another quantization under those names would read \
                         its storage as a matrix"
                    ),
                ));
            }
        };
        if let Some(targets) = group.optional_texts("targets")?
            && targets.iter().any(|target| *target != "Linear")
        {
            return Err(refuse(
                group.owner(),
                format!(
                    "targets {targets:?}, and packed weights are read for linear projections: a \
                     calibration that quantizes another module class states a checkpoint whose \
                     packing the model cannot follow"
                ),
            ));
        }
        if let Some(previous) = algorithm
            && previous != this
        {
            return Err(refuse(
                groups.owner(),
                format!("states two quantizations at once, `{previous}` and `{this}`"),
            ));
        }
        algorithm = Some(this);
    }
    algorithm
        .map(str::to_owned)
        .ok_or_else(|| refuse(groups.owner(), "states no group"))
}

/// Whether a declaration's excluded-module entry names `weight`.
///
/// The exports write three shapes and all three appear in one checkpoint's list: a bare module name
/// (`lm_head`), a dotted prefix that excludes a subtree (`model.layers.0.mlp.gate`), and a pattern
/// with `*` (`model.layers.*.mlp.gate`). An entry matches when the weight is the entry, is under
/// it, or matches it with `*` standing for any run of characters.
fn matches_module(entry: &str, weight: &str) -> bool {
    // An entry names a *module*, and a module owns every weight under it, so the comparison is
    // against the weight's module path — the weight without its final component.
    if entry == weight {
        return true;
    }
    let module = weight.rsplit_once('.').map_or(weight, |(module, _)| module);
    if module == entry || module.starts_with(&format!("{entry}.")) {
        return true;
    }
    if !entry.contains('*') {
        return false;
    }
    // A glob is matched by splitting on `*` and walking the pieces in order: the first piece anchors
    // the front, the last anchors the back, and the middle pieces are found in sequence.
    let pieces: Vec<&str> = entry.split('*').collect();
    let (first, rest) = (pieces[0], &pieces[1..]);
    let Some(mut cursor) = module.strip_prefix(first) else {
        return false;
    };
    let last_index = rest.len().saturating_sub(1);
    for (index, piece) in rest.iter().enumerate() {
        if piece.is_empty() {
            continue;
        }
        if index == last_index {
            return cursor.ends_with(piece);
        }
        match cursor.find(piece) {
            Some(at) => cursor = &cursor[at + piece.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_excluded_entry_names_a_module_a_subtree_or_a_pattern() {
        assert!(matches_module("lm_head", "lm_head.weight"));
        assert!(matches_module(
            "model.layers.0",
            "model.layers.0.mlp.gate.weight"
        ));
        assert!(matches_module(
            "model.layers.*.mlp.gate",
            "model.layers.17.mlp.gate.weight"
        ));
        assert!(!matches_module(
            "model.layers.*.mlp.gate",
            "model.layers.17.mlp.experts.0.gate_proj.weight"
        ));
        assert!(!matches_module(
            "model.layers.1",
            "model.layers.10.mlp.gate.weight"
        ));
    }
}
