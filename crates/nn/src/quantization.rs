//! What a calibrated checkpoint declares about its own numeric representation, and the method
//! each module reads a weight with.
//!
//! A quantized checkpoint states it in its metadata, not in its tensors: `config.json`'s
//! `quantization_config`, whose `quant_method` names the producer whose schema the rest of the
//! object follows. [`Declaration::read`] is the one place it is read, and it reads one
//! producer's schema in that producer's spelling: ModelOpt's (`modelopt`). Another producer, or
//! an algorithm no operator of this build serves, is refused by name before any tensor is read.
//!
//! ## The shard decides
//!
//! A declaration states what a calibration intended; the shards state what it did. The two can
//! differ — an exclusion list that names a module the export then packed, or a declaration that
//! quantizes a vocabulary matrix the shard holds in BF16 — and the tensor is the fact.
//! [`Declaration::resolve`] reads the shard headers once and records which weights they store
//! packed (a `weight_scale` beside the weight) into a [`Representation`], the one record
//! [`Representation::scheme`] reads.
//!
//! ## Mixed precision
//!
//! A ModelOpt `MIXED_PRECISION` export states one algorithm per module (`quantized_layers`), and
//! each packed weight takes the scheme its module states: per-tensor FP8 ([`Scheme::Fp8`]) or
//! NVFP4 with encoded activations ([`Scheme::Nvfp4`]).
//!
//! ## Representations without an operator
//!
//! No operator of this build multiplies weight-only representations: NVFP4 weights under 16-bit
//! activations (ModelOpt's `W4A16_NVFP4`) and FP8 weights scaled per block (ModelOpt's
//! `FP8_PB_WO`, the `fp8` producer's `weight_block_size`, a `weight_scale_inv` beside a weight in
//! a shard). A checkpoint that declares or stores one is refused, naming the module, until an
//! operator serves it; it is never decoded into a dense matrix in its place.
//!
//! ## The method follows the module
//!
//! There is no table of roles: a module asks for its weight through its
//! [`VarBuilder`](crate::VarBuilder), and the module's type decides what a packed weight means. A
//! [`Linear`](crate::modules::Linear) reads an NVFP4 weight as an activation encoder and a packed
//! product, and an FP8 weight as a product that encodes its activations by their static scale.
//! Every other packed tensor reaching a module is refused by name instead of being read as a
//! matrix.
//!
//! ## The spelling and the activation scales
//!
//! A calibrated NVFP4 projection is four fields (packed values, block scales, a tensor scale and
//! the activation's tensor scale), and two things about them are the export's and not the
//! format's: the names of the fields ([`Nvfp4Names`]) and where the activation's scale is
//! stated. A ModelOpt unified checkpoint states both in its tensors ([`UNIFIED_EXPORT`], an
//! `input_scale` beside each weight). A packed export of a diffusers pipeline states its
//! calibration in a manifest ([`ModeloptManifest`]): its fields are in the manifest export's
//! spelling ([`ModeloptManifest::NAMES`]) and each module's activation scale is the manifest's.
//! A declaration holds both facts, and every module that reads packed fields reads them by the
//! representation its loader resolved ([`Representation::names`],
//! [`Representation::activation_scale`]).
//!
//! ## The deployment's policy
//!
//! A deployment states its precision as a policy (`quantization`: a mode and selections by
//! module), which each family interprets against what its checkpoint declares when it opens
//! it ([`crate::family`]). A family that states no mode and no selection serves the checkpoint's
//! own representation and refuses any other policy ([`own_representation`]).
//!
//! A family that converts a dense checkpoint's modules states the projections it converts in
//! the declaration of their stage ([`Declaration::converted`]): each named projection's weight is
//! read as [`Scheme::DynamicFp8`] or [`Scheme::DynamicNvfp4`], encoded once at load from the
//! checkpoint's own values under the whole tensor's largest magnitude ([`dynamic`]), and its
//! product encodes each activation at a scale measured from that activation on every call
//! ([`ops::dynamic_scale`]). Every other tensor of the stage, the projection's bias included,
//! stays dense, and a module that reads a converted weight with no encoded form is refused.

use std::collections::{BTreeMap, BTreeSet};

use proto::config::Quantization;

use crate::checkpoint::Checkpoint;
use crate::config::Fields;
use crate::error::{Error, Result};

pub(crate) mod dynamic;
mod modelopt;
mod nvfp4;

pub use modelopt::ModeloptManifest;
pub use nvfp4::{Nvfp4Names, Nvfp4Share, Nvfp4Stack, Nvfp4Weight};

/// The spelling of a ModelOpt unified checkpoint's calibrated NVFP4 projection: the four fields
/// are tensors beside each other under the projection's module, the activation scale among
/// them.
pub const UNIFIED_EXPORT: Nvfp4Names = Nvfp4Names {
    export: "a ModelOpt unified checkpoint",
    values: "weight",
    block_scale: "weight_scale",
    tensor_scale: "weight_scale_2",
    input_scale: "input_scale",
};

/// How one weight is stored, and so how the module that owns it reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scheme {
    /// The checkpoint's own floating-point representation.
    Dense,
    /// ModelOpt NVFP4: packed E2M1 values, K16 block scales in `F8_E4M3`, one tensor scale and
    /// the calibration's activation scale, under the names of the export's spelling
    /// ([`Nvfp4Names`]).
    Nvfp4,
    /// ModelOpt per-tensor FP8: `weight` in `F8_E4M3`, one `weight_scale` and the static
    /// `input_scale` the activations are encoded by.
    Fp8,
    /// A dense weight the deployment's policy converts at load into per-tensor FP8: the E4M3
    /// values `e4m3(x · 448 / amax_w)` under the whole checkpoint tensor's largest magnitude
    /// `amax_w`. Its product encodes each activation at a scale measured from that activation
    /// on every call.
    DynamicFp8,
    /// A dense weight converted at load into NVFP4: packed E2M1 codes and an E4M3 scale per
    /// sixteen elements under the tensor scale `amax_w / (6 · 448)`, as a calibrated export
    /// stores them. Its product encodes each activation into NVFP4 at a scale measured from
    /// that activation on every call.
    DynamicNvfp4,
}

/// What a checkpoint declares about the representation of its weights, before its shards are
/// consulted.
#[derive(Clone, PartialEq, Debug)]
pub struct Declaration {
    /// The scheme the calibration applies to the linear projections it packs; [`Scheme::Dense`]
    /// for a checkpoint that packs nothing and for a mixed export, whose modules state their
    /// own.
    scheme: Scheme,
    /// A mixed-precision export's scheme per module, longest name first; `None` when one scheme
    /// covers every packed weight.
    layers: Option<Vec<(String, Scheme)>>,
    /// The spelling the export writes a calibrated NVFP4 projection's fields in.
    names: Nvfp4Names,
    /// Each calibrated module's activation tensor scale, by the module's name in the checkpoint,
    /// when the export states its calibration in a manifest; `None` when each packed weight
    /// stores its activation scale beside it.
    activations: Option<BTreeMap<String, f32>>,
    /// The projections of a dense checkpoint converted at load: a module pattern and the scheme
    /// the module's `weight` takes, [`Scheme::DynamicFp8`] or [`Scheme::DynamicNvfp4`].
    converted: Vec<(String, Scheme)>,
}

impl Declaration {
    /// A checkpoint that declares nothing: every weight is in its own dtype.
    pub fn dense() -> Self {
        Self {
            scheme: Scheme::Dense,
            layers: None,
            names: UNIFIED_EXPORT,
            activations: None,
            converted: Vec::new(),
        }
    }

    /// A dense checkpoint whose projections `modules` names are converted at load.
    ///
    /// Each entry is a module pattern, in which `*` stands for any run of characters and which
    /// matches a module's whole name, with the scheme the module's `weight` is read in:
    /// [`Scheme::DynamicFp8`] or [`Scheme::DynamicNvfp4`]. A module no pattern names, and every
    /// tensor of a named module but its `weight`, stays dense.
    ///
    /// # Errors
    ///
    /// When an entry states another scheme: the others are what a checkpoint's shards state, not
    /// a conversion.
    pub fn converted(modules: Vec<(String, Scheme)>) -> Result<Self> {
        if let Some((module, scheme)) = modules
            .iter()
            .find(|(_, scheme)| !matches!(scheme, Scheme::DynamicFp8 | Scheme::DynamicNvfp4))
        {
            return Err(refuse(
                module,
                format!("{scheme:?} is a checkpoint's representation, not a conversion at load"),
            ));
        }

        Ok(Self {
            converted: modules,
            ..Self::dense()
        })
    }

    /// Read the declaration of `checkpoint`: `config.json`'s `quantization_config`, whose
    /// `quant_method` names the producer whose schema the rest of the object follows.
    ///
    /// A checkpoint that states no `quantization_config` is [`Self::dense`]: the absence of a
    /// declaration is a statement about the checkpoint (nothing was quantized), not a gap to
    /// fill with a guess. The producer served is ModelOpt (`quant_method: modelopt`), read in
    /// its own spelling ([`modelopt_config`]); every other producer is refused by name, because
    /// its fields would be read under names that mean something else.
    ///
    /// # Errors
    ///
    /// When the declaration names another producer, or states an algorithm, a group or a
    /// per-module algorithm this library does not serve.
    pub fn read(checkpoint: &Checkpoint) -> Result<Self> {
        let Some(config) = checkpoint.sidecar("config.json")? else {
            return Ok(Self::dense());
        };
        let Some(stated) = config
            .get("quantization_config")
            .filter(|value| !value.is_null())
        else {
            return Ok(Self::dense());
        };

        let fields = Fields::of("config.json.quantization_config", stated)?;
        match fields.text("quant_method")? {
            "modelopt" => modelopt_config(&fields),
            other => Err(fields.refuse(
                "quant_method",
                format!(
                    "names the producer `{other}`; this build reads ModelOpt's declaration \
                     (`modelopt`), and another producer's fields would be read under names that \
                     mean something else"
                ),
            )),
        }
    }

    /// This declaration with every weight's packing taken from what `checkpoint`'s shards hold:
    /// a weight is packed when its shard stores the block scale (`weight_scale`) beside it.
    ///
    /// Both spellings name the block scales `weight_scale`, so a packed module's weight is
    /// `<module>.weight` under either, whichever tensor holds its values.
    ///
    /// # Errors
    ///
    /// When a shard stores block-scaled FP8 (a `weight_scale_inv` beside a weight), which no
    /// operator of this build multiplies, or a manifest calibrates a module whose shard stores
    /// no packed weight: the export states an activation scale for a product the checkpoint
    /// cannot compute encoded.
    pub fn resolve(&self, checkpoint: &Checkpoint) -> Result<Representation> {
        let beside = |suffix: &'static str| -> BTreeSet<String> {
            checkpoint
                .names()
                .filter_map(|name| name.strip_suffix(suffix))
                .filter(|weight| weight.ends_with("weight"))
                .map(str::to_owned)
                .collect()
        };

        if let Some(weight) = beside("_scale_inv").into_iter().next() {
            return Err(Error::weight(
                &weight,
                "is stored block-scaled (a `weight_scale_inv` beside it), and no operator of \
                 this build multiplies block-scaled FP8 weights: the checkpoint is refused \
                 until one does",
            ));
        }

        let packed = if self.scheme == Scheme::Dense && self.layers.is_none() {
            BTreeSet::new()
        } else {
            beside("_scale")
        };
        for module in self.activations.iter().flat_map(BTreeMap::keys) {
            if !packed.contains(&format!("{module}.weight")) {
                return Err(Error::weight(
                    &format!("{module}.{}", self.names.values),
                    format!(
                        "the export's manifest calibrates the activations of `{module}`, and the \
                         checkpoint stores no packed weight for it (no `{module}.{}`)",
                        self.names.block_scale
                    ),
                ));
            }
        }

        Ok(Representation {
            declaration: self.clone(),
            packed,
        })
    }

    /// Whether any weight is declared packed with encoded activations: whether the encoded
    /// kernels are the ones that run.
    pub fn is_encoded(&self) -> bool {
        self.scheme == Scheme::Nvfp4
            || self
                .layers
                .iter()
                .flatten()
                .any(|(_, scheme)| *scheme == Scheme::Nvfp4)
    }

    /// The calibrated activation tensor scale of `module`, a module's name in the checkpoint,
    /// when the export states its calibration in a manifest. `None` says the scale is the
    /// tensor the export's spelling names beside the module's weight
    /// ([`Nvfp4Names::input_scale`]).
    ///
    /// # Errors
    ///
    /// When a manifest calibrates the checkpoint and lists nothing for `module`: a packed
    /// projection it does not calibrate has no scale to encode its activations by.
    pub fn activation_scale(&self, module: &str) -> Result<Option<f32>> {
        let Some(activations) = &self.activations else {
            return Ok(None);
        };

        activations.get(module).copied().map(Some).ok_or_else(|| {
            Error::config(
                ModeloptManifest::FILE,
                module,
                "is stored packed, and the manifest lists no calibration that names it: its \
                 activations have no tensor scale to be encoded by",
            )
        })
    }
}

/// A declaration resolved against the shards of the checkpoint it describes: how each weight
/// is stored, the one record a module's weight is read by.
///
/// The declaration states what a calibration intended and the shards what it did: an
/// exclusion list may name a module the export then packed, or a declaration may quantize a
/// matrix the shard holds in BF16, and the tensor is the fact. [`Declaration::resolve`] reads
/// the shard headers once, and [`Self::scheme`] reads only what it recorded.
#[derive(Clone, PartialEq, Debug)]
pub struct Representation {
    /// What the checkpoint declares.
    declaration: Declaration,
    /// The weights whose shards store a block scale beside them.
    packed: BTreeSet<String>,
}

impl Representation {
    /// The representation of a checkpoint that declares nothing and packs nothing.
    pub fn dense() -> Self {
        Self {
            declaration: Declaration::dense(),
            packed: BTreeSet::new(),
        }
    }

    /// How the checkpoint stores `weight`, a weight's full checkpoint name.
    pub fn scheme(&self, weight: &str) -> Scheme {
        let declaration = &self.declaration;
        let packed = self.packed.contains(weight);

        if let Some(layers) = &declaration.layers {
            // A mixed export: a weight its shard stores packed takes its module's scheme.
            if !packed {
                return Scheme::Dense;
            }
            return layers
                .iter()
                .find(|(module, _)| owns(module, weight))
                .map_or(Scheme::Dense, |(_, scheme)| *scheme);
        }
        if declaration.scheme == Scheme::Dense {
            return declaration.conversion(weight).unwrap_or(Scheme::Dense);
        }
        if packed {
            declaration.scheme
        } else {
            Scheme::Dense
        }
    }

    /// The spelling the checkpoint's calibrated NVFP4 fields are read in.
    pub fn names(&self) -> Nvfp4Names {
        self.declaration.names
    }

    /// The calibrated activation tensor scale of `module` ([`Declaration::activation_scale`]).
    ///
    /// # Errors
    ///
    /// As [`Declaration::activation_scale`].
    pub fn activation_scale(&self, module: &str) -> Result<Option<f32>> {
        self.declaration.activation_scale(module)
    }
}

impl Declaration {
    /// The scheme a dense checkpoint's `weight` is converted to at load, when a pattern of
    /// [`Self::converted`] names the module whose `weight` it is; `None` for a weight the
    /// declaration converts nothing of.
    pub fn conversion(&self, weight: &str) -> Option<Scheme> {
        let module = weight.strip_suffix(".weight")?;

        self.converted
            .iter()
            .find(|(pattern, _)| glob(pattern, module))
            .map(|(_, scheme)| *scheme)
    }
}

/// Refuses a precision policy for `family`, a family that states no precision mode and no
/// module selection, naming the family and the policy.
///
/// Such a family serves every checkpoint in the representation the checkpoint states: its own
/// element type, or the quantization its export declares ([`Declaration::read`]). An empty
/// policy selects exactly that and is accepted; any other would be ignored by the load, so it is
/// refused instead.
///
/// # Errors
///
/// [`Error::Unsupported`] when `policy` states a mode or a module selection.
pub fn own_representation(family: &str, policy: &Quantization) -> Result<()> {
    if policy.is_empty() {
        return Ok(());
    }

    Err(Error::Unsupported(format!(
        "`quantization = {policy}`: the {family} family states no precision mode and no module \
         selection; it serves each checkpoint in the checkpoint's own representation (an NVFP4 \
         or FP8 export as its quantization declaration states), so leave `quantization` empty \
         and serve the export of the precision wanted"
    )))
}

/// Whether the module `module` owns `weight`, a weight's full checkpoint name: the weight is a
/// leaf of the module or of a module under it (`mlp.experts` owns `mlp.experts.3.up_proj.weight`).
fn owns(module: &str, weight: &str) -> bool {
    weight
        .strip_prefix(module)
        .is_some_and(|rest| rest.starts_with('.'))
}

/// The scheme of one algorithm ModelOpt's declaration states for the projections it covers,
/// the whole checkpoint's (`quant_algo`) or one module's (a `quantized_layers` entry), refused
/// when no operator of this build serves it.
///
/// Served are NVFP4 with encoded activations (`NVFP4`, over blocks of sixteen) and per-tensor
/// FP8 (`FP8`). ModelOpt's weight-only algorithms are refused by name: `W4A16_NVFP4` keeps
/// 16-bit activations against NVFP4 weights and `FP8_PB_WO` scales FP8 weights per block, and
/// no operator of this build multiplies either.
fn algorithm_scheme(owner: &str, module: &str, fields: &Fields<'_>) -> Result<Scheme> {
    let algorithm = fields.text("quant_algo")?;
    let scheme = match algorithm {
        "FP8" => Scheme::Fp8,
        "NVFP4" => Scheme::Nvfp4,
        "W4A16_NVFP4" | "FP8_PB_WO" => {
            return Err(refuse(
                owner,
                format!(
                    "states `{algorithm}` for {module}: no operator of this build multiplies \
                     its weight-only representation, so the checkpoint is refused until one does"
                ),
            ));
        }
        other => {
            return Err(refuse(
                owner,
                format!(
                    "states `{other}` for {module}; the algorithms served are NVFP4 and FP8, \
                     and loading `{other}` as either would read its storage under the wrong \
                     names"
                ),
            ));
        }
    };

    if scheme == Scheme::Nvfp4
        && let Some(group) = fields.optional_index("group_size")?
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

/// The declaration a ModelOpt unified checkpoint's `quantization_config` states.
///
/// ModelOpt's schema there: `quant_algo` names the algorithm of every packed projection, or
/// `MIXED_PRECISION` with `quantized_layers` stating one per module (each entry its own
/// `quant_algo` and, for NVFP4, `group_size`). The other fields ModelOpt writes describe what
/// the shards then state (`ignore`, the modules left dense; `config_groups`, a summary of the
/// algorithms) or the key/value cache's calibration (`kv_cache_scheme`), which a deployment's
/// cache type decides; the packing is the shards' ([`Declaration::resolve`]), so no load reads
/// them.
///
/// # Errors
///
/// When the algorithm, or a module's, is not served, or a mixed export states no module.
fn modelopt_config(fields: &Fields<'_>) -> Result<Declaration> {
    let owner = fields.owner();

    let (scheme, layers) = match fields.text("quant_algo")? {
        "MIXED_PRECISION" => {
            let map = fields.object("quantized_layers")?;
            let mut layers = Vec::new();
            for module in map.keys() {
                layers.push((
                    module.to_owned(),
                    algorithm_scheme(map.owner(), module, &map.object(module)?)?,
                ));
            }
            if layers.is_empty() {
                return Err(refuse(
                    owner,
                    "declares MIXED_PRECISION and no `quantized_layers`: a mixed export states \
                     each module's algorithm",
                ));
            }

            // The longest module first, so a module nested in another takes its own entry.
            layers.sort_by_key(|(module, _)| std::cmp::Reverse(module.len()));
            (Scheme::Dense, Some(layers))
        }
        _ => (
            algorithm_scheme(owner, "every packed projection", fields)?,
            None,
        ),
    };

    Ok(Declaration {
        scheme,
        layers,
        ..Declaration::dense()
    })
}

/// Whether `text` matches `pattern` whole, `*` in the pattern standing for any run of characters.
///
/// The pattern is split on `*` and its pieces taken in order: the first piece anchors the front,
/// the last anchors the back, and the middle pieces are found in sequence. A pattern without `*`
/// matches only itself.
fn glob(pattern: &str, text: &str) -> bool {
    let pieces: Vec<&str> = pattern.split('*').collect();
    let (first, rest) = (pieces[0], &pieces[1..]);
    let Some(mut cursor) = text.strip_prefix(first) else {
        return false;
    };
    let Some((last, middle)) = rest.split_last() else {
        return cursor.is_empty();
    };

    for piece in middle {
        match cursor.find(piece) {
            Some(at) => cursor = &cursor[at + piece.len()..],
            None => return false,
        }
    }
    cursor.len() >= last.len() && cursor.ends_with(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ModelOpt declaration `value` states, as `config.json`'s `quantization_config`.
    fn declared(value: serde_json::Value) -> Result<Declaration> {
        modelopt_config(&Fields::of("config.json.quantization_config", &value).unwrap())
    }

    /// A representation over `packed` weights, as resolving the declaration against shards
    /// that store a block scale beside each of them records it.
    fn packed(declaration: Declaration, weights: &[&str]) -> Representation {
        Representation {
            declaration,
            packed: weights.iter().map(|weight| (*weight).to_owned()).collect(),
        }
    }

    /// A mixed export's packed weights take their modules' algorithms, the nested module its
    /// own entry, and a weight its shard holds unpacked stays dense.
    #[test]
    fn a_mixed_export_states_each_modules_scheme() {
        let declaration = declared(serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "m.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
                "m.layers.0.linear_attn": {"quant_algo": "NVFP4", "group_size": 16},
                "lm_head": {"quant_algo": "NVFP4", "group_size": 16},
            }
        }))
        .unwrap();
        assert!(declaration.is_encoded());

        let representation = packed(
            declaration,
            &[
                "m.layers.0.linear_attn.in_proj_qkv.weight",
                "m.layers.0.linear_attn.out_proj.weight",
                "lm_head.weight",
            ],
        );
        for (weight, scheme) in [
            ("m.layers.0.linear_attn.in_proj_qkv.weight", Scheme::Fp8),
            ("m.layers.0.linear_attn.out_proj.weight", Scheme::Nvfp4),
            ("lm_head.weight", Scheme::Nvfp4),
            ("m.layers.0.linear_attn.in_proj_a.weight", Scheme::Dense),
        ] {
            assert_eq!(representation.scheme(weight), scheme, "{weight}");
        }
    }

    /// A declaration states one algorithm or one per module; an algorithm no operator serves
    /// is refused naming it, the weight-only ones among them, and a mixed export without
    /// modules is refused.
    #[test]
    fn an_algorithm_no_operator_serves_is_refused_by_name() {
        for algorithm in ["INT4_AWQ", "W4A16_NVFP4", "FP8_PB_WO"] {
            let error = declared(serde_json::json!({
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {"m.q_proj": {"quant_algo": algorithm}}
            }))
            .expect_err("an unserved per-module algorithm");
            assert!(error.to_string().contains(algorithm), "{error}");

            let error = declared(serde_json::json!({
                "quant_method": "modelopt",
                "quant_algo": algorithm,
            }))
            .expect_err("an unserved algorithm");
            assert!(error.to_string().contains(algorithm), "{error}");
        }

        let empty = serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {},
        });
        assert!(declared(empty).is_err());
    }

    /// A conversion names a projection's `weight` by its module's whole name; its bias, its
    /// sub-modules and a module the pattern only begins or ends are dense, and a scheme that is
    /// no conversion is refused.
    #[test]
    fn a_conversion_names_the_weight_of_each_module_its_pattern_matches_whole() {
        let declaration = Declaration::converted(vec![
            ("blocks.*.attn.to_q".to_owned(), Scheme::DynamicFp8),
            ("blocks.*.ff.net.2".to_owned(), Scheme::DynamicNvfp4),
        ])
        .unwrap();
        assert!(!declaration.is_encoded());
        assert_ne!(declaration, Declaration::dense());

        let representation = packed(declaration, &[]);
        for (weight, scheme) in [
            ("blocks.0.attn.to_q.weight", Scheme::DynamicFp8),
            ("blocks.49.ff.net.2.weight", Scheme::DynamicNvfp4),
            ("blocks.0.attn.to_q.bias", Scheme::Dense),
            ("blocks.0.attn.to_q.lora.weight", Scheme::Dense),
            ("refiner.blocks.0.attn.to_q.weight", Scheme::Dense),
            ("blocks.0.ff.net.20.weight", Scheme::Dense),
            ("blocks.0.attn.norm_q.weight", Scheme::Dense),
        ] {
            assert_eq!(representation.scheme(weight), scheme, "{weight}");
        }

        assert!(Declaration::converted(vec![("proj".to_owned(), Scheme::Nvfp4)]).is_err());
    }
}
