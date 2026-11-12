//! What a calibrated checkpoint declares about its own numeric representation, and the method
//! each module reads a weight with.
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
//! the weight), and [`Declaration::scheme`] reads that record, falling back to the
//! declaration only when no shard was consulted.
//!
//! ## Mixed precision
//!
//! A ModelOpt `MIXED_PRECISION` export states one algorithm per module (`quantized_layers`), and
//! each weight takes the scheme its module states: per-tensor FP8 ([`Scheme::Fp8`]), NVFP4 with
//! encoded activations ([`Scheme::Nvfp4`]) or NVFP4 weights under 16-bit activations
//! ([`Scheme::Nvfp4Weights`], ModelOpt's `W4A16_NVFP4`). The per-module statement wins over the
//! file's `config_groups`, which summarize it.
//!
//! ## Block-scaled FP8
//!
//! An export may store a weight as E4M3 values with one scale per block of rows and columns in
//! a `weight_scale_inv` tensor beside it ([`Scheme::Fp8Blocks`]; ModelOpt's `FP8_BLOCK_SCALES`
//! and the `quant_method: "fp8"` exports with a `weight_block_size`). The shard states it
//! whatever the declaration says, and every module reads such a weight as the dense matrix its
//! values and scales state, decoded once at load: no operator of this build multiplies
//! block-scaled FP8 weights, so a module whose weights are too many to hold dense refuses the
//! checkpoint itself.
//!
//! ## The method follows the module
//!
//! There is no table of roles: a module asks for its weight through its
//! [`VarBuilder`](crate::VarBuilder), and the module's type decides what a packed weight means. A
//! [`Linear`](crate::modules::Linear) reads an NVFP4
//! weight as an activation encoder and a packed product, and an FP8 weight as a product that
//! encodes its activations by their static scale. A weight-only NVFP4 weight is read by every
//! module as the dense matrix its codes and scales state, decoded once at load
//! ([`VarBuilder::load`](crate::VarBuilder::load)), so its products are the 16-bit products the
//! declaration states. Every other packed tensor reaching a module is refused by name instead of
//! being read as a matrix.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::checkpoint::Checkpoint;
use crate::config::Fields;
use crate::error::{Error, Result};

mod nvfp4;

pub use nvfp4::{Nvfp4Names, Nvfp4Share, Nvfp4Stack, Nvfp4Weight, ValueLayout};

/// The declaration's own file, when the export writes one beside `config.json`.
const DECLARATION_FILE: &str = "hf_quant_config.json";

/// How one weight is stored, and so how the module that owns it reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scheme {
    /// The checkpoint's own floating-point representation.
    Dense,
    /// ModelOpt NVFP4: `weight` (packed E2M1), `weight_scale` (K16 block scales in `F8_E4M3`),
    /// `weight_scale_2` (one tensor scale) and `input_scale` (the calibration's activation scale).
    Nvfp4,
    /// ModelOpt `W4A16_NVFP4`: the same fields, and activations that stay in their 16-bit type, so
    /// the product is the 16-bit activation times the matrix the codes and scales state.
    Nvfp4Weights,
    /// ModelOpt per-tensor FP8: `weight` in `F8_E4M3`, one `weight_scale` and the static
    /// `input_scale` the activations are encoded by.
    Fp8,
    /// Block-scaled FP8: `weight` `[rows, columns]` in `F8_E4M3` and `weight_scale_inv`
    /// `[rows / r, columns / c]`, the scale each `r × c` block of the weight is multiplied by.
    /// Read by every module as the dense matrix `e4m3(weight) · weight_scale_inv`, decoded once
    /// at load.
    Fp8Blocks,
}

/// The scheme a checkpoint's key/value cache scales follow.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvCacheScheme {
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
    kv_cache: KvCacheScheme,
    /// The weights whose shards store a block scale beside them, once [`Self::resolve`] has read
    /// the shards; `None` before, when the declaration alone decides.
    packed: Option<BTreeSet<String>>,
    /// A mixed-precision export's scheme per module, longest name first; `None` when one scheme
    /// covers every packed weight.
    layers: Option<Vec<(String, Scheme)>>,
    /// The weights whose shards store a `weight_scale_inv` beside them, once [`Self::resolve`]
    /// has read the shards: block-scaled FP8, whatever the declaration states.
    blocks: Option<BTreeSet<String>>,
}

impl Declaration {
    /// A checkpoint that declares nothing: every weight is in its own dtype.
    pub fn dense() -> Self {
        Self {
            scheme: Scheme::Dense,
            excluded: Vec::new(),
            kv_cache: KvCacheScheme::Unquantized,
            packed: None,
            layers: None,
            blocks: None,
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
                Some(value) if !value.is_null() => Some(stated(
                    &Fields::of("config.json.quantization_config", value)?,
                    None,
                )?),
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
                    // ModelOpt names itself as the producer, not as a method.
                    let producer = side
                        .get("producer")
                        .and_then(|producer| producer.get("name"))
                        .and_then(Value::as_str);
                    Some(stated(&fields, producer)?)
                } else {
                    None
                }
            }
            None => None,
        };
        match (from_config, from_side) {
            (None, None) => Ok(Self::dense()),
            (Some((_, declaration)), None) | (None, Some((_, declaration))) => declaration,
            (Some((config, declaration)), Some((side, _))) => {
                if config != side {
                    return Err(refuse(
                        "the checkpoint",
                        format!(
                            "declares two different quantizations: `config.json` says {config} \
                             and `{DECLARATION_FILE}` says {side}; a checkpoint written by two \
                             tools is one neither of them owns"
                        ),
                    ));
                }
                declaration
            }
        }
    }

    /// This declaration with every weight's packing taken from what `checkpoint`'s shards hold: a
    /// weight is packed when the shard stores a `weight_scale` beside it.
    pub fn resolve(&self, checkpoint: &Checkpoint) -> Self {
        let beside = |suffix: &'static str| -> BTreeSet<String> {
            checkpoint
                .names()
                .filter_map(|name| name.strip_suffix(suffix))
                .filter(|weight| weight.ends_with("weight"))
                .map(str::to_owned)
                .collect()
        };

        // A block-scaled weight is stated by its shard alone.
        let blocks = beside("_scale_inv");
        let blocks = (!blocks.is_empty()).then_some(blocks);
        if self.scheme == Scheme::Dense && self.layers.is_none() {
            return Self {
                blocks,
                ..self.clone()
            };
        }
        Self {
            packed: Some(beside("_scale")),
            blocks,
            ..self.clone()
        }
    }

    /// How the checkpoint stores `weight`, a weight's full checkpoint name.
    ///
    /// What the shards hold once [`Self::resolve`] has read them; the declaration's intent (packed
    /// unless an excluded module owns it) before.
    pub fn scheme(&self, weight: &str) -> Scheme {
        if self
            .blocks
            .as_ref()
            .is_some_and(|blocks| blocks.contains(weight))
        {
            return Scheme::Fp8Blocks;
        }
        if let Some(layers) = &self.layers {
            // A mixed export: a weight its shard stores packed takes its module's scheme.
            let packed = self
                .packed
                .as_ref()
                .is_none_or(|packed| packed.contains(weight));
            if !packed {
                return Scheme::Dense;
            }
            return layers
                .iter()
                .find(|(module, _)| owns(module, weight))
                .map_or(Scheme::Dense, |(_, scheme)| *scheme);
        }
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

    /// Whether any weight is declared packed: whether the encoded kernels are the ones that run.
    pub fn is_encoded(&self) -> bool {
        self.scheme == Scheme::Nvfp4
            || self
                .layers
                .iter()
                .flatten()
                .any(|(_, scheme)| *scheme == Scheme::Nvfp4)
    }

    /// The scheme the key/value cache scales follow.
    pub fn kv_cache(&self) -> KvCacheScheme {
        self.kv_cache
    }
}

/// The E2M1 values of the eight unsigned codes, the sign in the code's fourth bit.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// An E4M3 byte (`F8_E4M3`, bias 7, no infinities, `S.1111.111` NaN) as an FP32 value.
pub(crate) fn e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((byte >> 3) & 0x0f);
    let mantissa = f32::from(byte & 0x07);
    if exponent == 0x0f && byte & 0x07 == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        // Subnormal: no implicit one, the smallest exponent.
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    sign * magnitude
}

/// `value` as BF16 bits, rounded to nearest even.
pub(crate) fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lower = bits & 0xffff;
    let mut upper = bits >> 16;
    if lower > 0x8000 || (lower == 0x8000 && upper & 1 == 1) {
        upper += 1;
    }
    upper as u16
}

/// The BF16 bytes of the `[rows, columns]` matrix a ModelOpt NVFP4 weight states: `packed` holds
/// two E2M1 codes a byte (the even element in the low nibble), `scales` one E4M3 block scale per
/// sixteen elements of a row, and `tensor_scale` the weight's one tensor scale.
///
/// Each element is `code · block_scale · tensor_scale` evaluated in FP32 and rounded once to BF16.
/// `code · block_scale` is exact in BF16 (two and four significant bits), so the one rounding is
/// the tensor scale's, where a 16-bit-activation kernel applies the tensor scale to its FP32
/// accumulator instead: the two differ by at most half a BF16 unit of each weight.
pub(crate) fn decode_nvfp4(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    columns: usize,
) -> Vec<u8> {
    let elements = packed.len() * 2;
    let mut out = Vec::with_capacity(elements * 2);
    for element in 0..elements {
        let byte = packed[element / 2];
        let code = if element % 2 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        };
        let value = E2M1[usize::from(code & 0x07)] * if code & 0x08 != 0 { -1.0 } else { 1.0 };
        let (row, column) = (element / columns, element % columns);
        let scale = e4m3(scales[row * (columns / 16) + column / 16]);
        out.extend_from_slice(&bf16_bits(value * scale * tensor_scale).to_le_bytes());
    }
    out
}

/// Whether the module `module` owns `weight`, a weight's full checkpoint name: the weight is a
/// leaf of the module or of a module under it (`mlp.experts` owns `mlp.experts.3.up_proj.weight`).
fn owns(module: &str, weight: &str) -> bool {
    weight
        .strip_prefix(module)
        .is_some_and(|rest| rest.starts_with('.'))
}

/// The scheme of one `quantized_layers` entry of a mixed-precision export.
fn layer_scheme(owner: &str, module: &str, layer: &Fields<'_>) -> Result<Scheme> {
    let algorithm = layer.text("quant_algo")?;
    let scheme = match algorithm.to_ascii_uppercase().as_str() {
        "FP8" => Scheme::Fp8,
        "NVFP4" => Scheme::Nvfp4,
        "W4A16_NVFP4" => Scheme::Nvfp4Weights,
        // ModelOpt's two spellings of weight-only FP8 with per-block scales.
        "FP8_BLOCK_SCALES" | "FP8_PB_WO" => Scheme::Fp8Blocks,
        other => {
            return Err(refuse(
                owner,
                format!(
                    "states `{other}` for {module}, and the per-module algorithms served are FP8, \
                     NVFP4, W4A16_NVFP4 and FP8_BLOCK_SCALES (FP8_PB_WO)"
                ),
            ));
        }
    };
    if matches!(scheme, Scheme::Nvfp4 | Scheme::Nvfp4Weights)
        && let Some(group) = layer.optional_index("group_size")?
        && group != 16
    {
        return Err(refuse(
            owner,
            format!("states NVFP4 blocks of {group} for {module}; NVFP4 blocks are sixteen"),
        ));
    }
    Ok(scheme)
}

/// A refusal of `what` a declaration states.
fn refuse(what: &str, problem: impl Into<String>) -> Error {
    Error::Config {
        owner: what.to_owned(),
        field: String::new(),
        problem: problem.into(),
    }
}

/// What one file states: the algorithm it names, and the declaration that algorithm and the
/// file's other fields make, or why this library cannot serve it.
///
/// The two are returned apart because a checkpoint may state its quantization in two files:
/// [`Declaration::read`] compares the two algorithms before it takes either file's declaration,
/// so two files that disagree are refused as a disagreement and not as whichever one is unserved.
/// `producer` is the tool the file says wrote it, which names the method when the file states no
/// `quant_method`.
///
/// # Errors
///
/// When a field is of another JSON type than its reader takes, or states a cache scheme, a group
/// or a per-module algorithm that is not served.
fn stated(fields: &Fields<'_>, producer: Option<&str>) -> Result<(String, Result<Declaration>)> {
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
            None => KvCacheScheme::Unquantized,
            Some(scheme) => match (
                scheme.optional_index("num_bits")?,
                scheme.optional_text("type")?,
            ) {
                (Some(8), Some("float")) => KvCacheScheme::Fp8,
                (bits, kind) => {
                    return Err(fields.refuse(
                        "kv_cache_scheme",
                        format!(
                            "states {bits:?} bits of {kind:?}, and the cache scales served are FP8"
                        ),
                    ));
                }
            },
        },
        Some(algorithm) if algorithm.eq_ignore_ascii_case("FP8") => KvCacheScheme::Fp8,
        Some(other) => {
            return Err(fields.refuse(
                "kv_cache_quant_algo",
                format!("names `{other}`, and the cache scales served are FP8"),
            ));
        }
    };

    let layers = match fields.optional_object("quantized_layers")? {
        None => None,
        Some(map) => {
            let mut layers = Vec::new();
            for module in map.keys() {
                let layer = map.object(module)?;
                layers.push((
                    module.to_owned(),
                    layer_scheme(map.owner(), module, &layer)?,
                ));
            }
            // The longest module first, so a module nested in another takes its own entry.
            layers.sort_by_key(|(module, _)| std::cmp::Reverse(module.len()));
            Some(layers)
        }
    };

    let method = fields.optional_text("quant_method")?.or(producer);
    let excluded = if excluded.is_empty() {
        ignored
    } else {
        excluded
    };
    let declaration = declaration(
        fields.owner(),
        &algorithm,
        method,
        excluded,
        kv_cache,
        layers,
    );
    Ok((algorithm, declaration))
}

/// The declaration the file `owner` makes by naming `algorithm`, refused when this library cannot
/// serve the algorithm or the `method` it was written by.
fn declaration(
    owner: &str,
    algorithm: &str,
    method: Option<&str>,
    excluded: Vec<String>,
    kv_cache: KvCacheScheme,
    layers: Option<Vec<(String, Scheme)>>,
) -> Result<Declaration> {
    let scheme = match algorithm.to_ascii_uppercase().as_str() {
        // Absent, empty or an explicit "none" is a dense checkpoint that wrote the field anyway.
        "" | "NONE" | "BF16" => Scheme::Dense,
        // Each module states its own algorithm; a weight takes its module's.
        "MIXED_PRECISION" => {
            if layers.as_ref().is_none_or(Vec::is_empty) {
                return Err(refuse(
                    owner,
                    "declares MIXED_PRECISION and no `quantized_layers`: a mixed export states \
                     each module's algorithm",
                ));
            }
            Scheme::Dense
        }
        "NVFP4" => {
            if let Some(method) = method
                && !method.eq_ignore_ascii_case("modelopt")
            {
                return Err(refuse(
                    owner,
                    format!(
                        "states the quantization method `{method}`, and NVFP4 is served in \
                         ModelOpt's `weight`, `weight_scale`, `weight_scale_2` and `input_scale` \
                         spelling; a `{method}` export states the same numbers under names this \
                         loader would not find"
                    ),
                ));
            }
            Scheme::Nvfp4
        }
        other => {
            return Err(refuse(
                owner,
                format!(
                    "declares quantization `{other}`, which is not served: NVFP4 (ModelOpt's \
                     spelling) and an unquantized checkpoint are. Loading `{other}` as if it \
                     were dense would read quantized storage as a matrix"
                ),
            ));
        }
    };

    let layers = if algorithm.eq_ignore_ascii_case("MIXED_PRECISION") {
        layers
    } else {
        None
    };
    Ok(Declaration {
        scheme,
        excluded,
        kv_cache,
        packed: None,
        layers,
        blocks: None,
    })
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
    // against the weight's module path — the weight without its final segment.
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
    // A glob is matched by splitting on `*` and taking the pieces in order: the first piece anchors
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

    /// An E4M3 byte decodes to its value: normal, subnormal, the largest finite and NaN.
    #[test]
    fn an_e4m3_byte_decodes_to_its_value() {
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0xb8), -1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0x3c), 1.5);
        assert!(e4m3(0x7f).is_nan());
    }

    /// A weight-only NVFP4 row decodes to `code · block scale · tensor scale`, the even element in
    /// the low nibble, each sixteen elements reading their own scale.
    #[test]
    fn an_nvfp4_matrix_decodes_to_its_codes_times_its_scales() {
        // Thirty-two elements of one row: two blocks. Byte 0x2f holds code 0xf (−6) then 0x2 (1).
        let mut packed = vec![0_u8; 16];
        packed[0] = 0x2f;
        packed[8] = 0x07; // element 16: code 7 (6), element 17: 0
        // The first block's scale is 1, the second's 2.
        let scales = [0x38, 0x40];
        let decoded = decode_nvfp4(&packed, &scales, 0.5, 32);
        let value = |index: usize| {
            f32::from_bits(
                u32::from(u16::from_le_bytes([
                    decoded[2 * index],
                    decoded[2 * index + 1],
                ])) << 16,
            )
        };
        assert_eq!(value(0), -3.0);
        assert_eq!(value(1), 0.5);
        assert_eq!(value(2), 0.0);
        assert_eq!(value(16), 6.0);
        assert_eq!(value(17), 0.0);
        // A product that is not a BF16 value rounds to the nearest: 1.5 · 1 · (1 + 2⁻⁹) lies a
        // quarter of a BF16 unit above 1.5 and rounds to it.
        let rounded = decode_nvfp4(&[0x03], &[0x38], 1.0 + 2f32.powi(-9), 2);
        assert_eq!(u16::from_le_bytes([rounded[0], rounded[1]]), 0x3fc0);
    }

    /// A mixed export's weights take their modules' algorithms, the nested module its own entry,
    /// and a weight its shard holds unpacked stays dense.
    #[test]
    fn a_mixed_export_states_each_modules_scheme() {
        let value = serde_json::json!({
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "m.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
                "m.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                "lm_head": {"quant_algo": "NVFP4", "group_size": 16},
            }
        });
        let (algorithm, declaration) = stated(&Fields::of("q", &value).unwrap(), None).unwrap();
        assert_eq!(algorithm, "MIXED_PRECISION");
        let declaration = declaration.unwrap();
        assert_eq!(
            declaration.scheme("m.layers.0.linear_attn.in_proj_qkv.weight"),
            Scheme::Fp8
        );
        assert_eq!(
            declaration.scheme("m.layers.0.mlp.experts.7.up_proj.weight"),
            Scheme::Nvfp4Weights
        );
        assert_eq!(declaration.scheme("lm_head.weight"), Scheme::Nvfp4);
        assert_eq!(
            declaration.scheme("m.layers.0.linear_attn.in_proj_a.weight"),
            Scheme::Dense
        );
        assert!(declaration.is_encoded());
    }

    #[test]
    fn a_mixed_export_refuses_an_algorithm_it_does_not_serve() {
        let value = serde_json::json!({
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"m.q_proj": {"quant_algo": "INT4_AWQ"}}
        });
        let error = stated(&Fields::of("q", &value).unwrap(), None)
            .and_then(|(_, declaration)| declaration)
            .expect_err("an unserved per-module algorithm");
        assert!(error.to_string().contains("INT4_AWQ"), "{error}");
        let empty = serde_json::json!({"quant_algo": "MIXED_PRECISION"});
        assert!(
            stated(&Fields::of("q", &empty).unwrap(), None)
                .and_then(|(_, declaration)| declaration)
                .is_err()
        );
    }

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
