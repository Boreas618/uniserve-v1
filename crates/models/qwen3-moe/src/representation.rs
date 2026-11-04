//! What a calibrated checkpoint declares about its own numeric representation.
//!
//! A checkpoint that was quantized states it in its metadata, not in its tensors: ModelOpt writes
//! `quantization_config` into `config.json` (a vLLM export) or a sibling `hf_quant_config.json`
//! (the older export), and both name a `quant_algo` and the modules the calibration left in their
//! source dtype. This module is the one place that reads those fields, so the loader above it
//! decides per parameter from a typed value rather than from a dictionary, and a representation this
//! tree cannot serve is refused by name before a single tensor is read.
//!
//! ## Why the declaration decides, and not the tensors
//!
//! A checkpoint's shard is a bag of tensors: a quantized projection is a `weight` of packed E2M1, a
//! `weight_scale` of K16 block scales and a `weight_scale_2` of one tensor scale, and a checkpoint
//! that quantized nothing has a plain BF16 `weight` with the same name. Reading the bag and taking
//! what is there would therefore accept a *half-converted* checkpoint — one whose `weight` is packed
//! and whose scales nothing normalizes, or the reverse — and produce a model that runs and measures
//! nothing. The declaration is what says a tensor is packed, and it is read once, before any tensor.
//!
//! ## The refusals this module owns
//!
//! - an algorithm this tree does not serve (`FP8`, `MXFP4`, `W4A16`, …), named with what is served;
//! - a quantization method that is not ModelOpt's, named the same way;
//! - `config.json` and `hf_quant_config.json` disagreeing about either field — a checkpoint written
//!   by two tools is a checkpoint neither of them owns;
//! - a declaration that quantizes a parameter whose role this tree serves dense (the embedding, the
//!   vocabulary head, a norm, the router). Those are refused rather than encoded, because encoding
//!   one would need a program this tree does not have, and reading it as dense would read packed
//!   bytes as a matrix.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::{Config, Error, Result};

/// The declaration's own file, when the export writes one beside `config.json`.
const DECLARATION_FILE: &str = "hf_quant_config.json";

/// What a checkpoint declares about the representation of its weights.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Representation {
    /// Nothing is quantized: every tensor is in the checkpoint's own dtype.
    Dense,
    /// ModelOpt NVFP4: a projection is `.weight` (packed E2M1), `.weight_scale` (K16 block scales in
    /// `F8_E4M3`), `.weight_scale_2` (one tensor scale) and `.input_scale` (the calibration's
    /// activation scale), except for the modules `excluded` names.
    Nvfp4 {
        /// The modules the calibration left in their source dtype, as `config.json` writes them.
        excluded: Vec<String>,
        /// The same list as the export's own file writes it, when it wrote one. The two need not be
        /// equal — they list different module classes — but a parameter they disagree about is a
        /// checkpoint no single reading is right for, and [`Representation::adopts`] refuses it.
        also_excluded: Vec<String>,
    },
}

impl Representation {
    /// Read the declaration of the checkpoint at `dir`.
    ///
    /// A checkpoint that declares nothing is [`Representation::Dense`]: the absence of a declaration
    /// is a statement about the checkpoint (nothing was quantized), not a gap to fill with a guess.
    pub fn read(dir: &Path) -> Result<Self> {
        let from_config = config_declaration(dir)?;
        let from_side = side_declaration(dir)?;
        match (from_config, from_side) {
            (None, None) => Ok(Self::Dense),
            (Some(declared), None) | (None, Some(declared)) => Self::from_fields(&declared),
            (Some(config), Some(side)) => {
                if config.algorithm != side.algorithm {
                    return Err(Error::Config(format!(
                        "the checkpoint declares two different quantizations: `config.json` says \
                         {} and `{DECLARATION_FILE}` says {}; a checkpoint written by two tools is \
                         one neither of them owns",
                        config.algorithm, side.algorithm,
                    )));
                }
                // The two exclusion lists need not be identical: a compressed-tensors block lists \
                // what *it* would quantize (its `targets`) while ModelOpt lists every module it \
                // left alone, so the lists differ in names that are not projections at all. What \
                // must not differ is a parameter's fate, and that is decided per parameter in \
                // `adopts`, where the caller states the role.
                Self::from_fields(&Declared {
                    excluded: config.excluded,
                    also_excluded: side.excluded,
                    ..config
                })
            }
        }
    }

    /// Whether the representation is NVFP4, i.e. whether the encoded programs are the ones that run.
    pub fn is_encoded(&self) -> bool {
        matches!(self, Self::Nvfp4 { .. })
    }

    /// The declaration as one line, for a run's first output: what it is, and how many modules it
    /// keeps out of it.
    ///
    /// `Debug` prints every excluded module, and a calibrated checkpoint excludes one per layer, so a
    /// run that printed the declaration whole would bury its own first line in module paths.
    pub fn summary(&self) -> String {
        match self {
            Self::Dense => "dense".to_owned(),
            Self::Nvfp4 {
                excluded,
                also_excluded,
            } => {
                let mut summary = format!("nvfp4, {} module(s) excluded", excluded.len());
                if !also_excluded.is_empty() {
                    summary.push_str(&format!(" and {} on the side file", also_excluded.len()));
                }
                summary
            }
        }
    }

    /// How the checkpoint stores `parameter`, whose role the caller states.
    ///
    /// A role this tree serves dense is refused when the declaration quantizes it: the refusal names
    /// the parameter, the role and the declaration, because the alternative — reading a packed
    /// `weight` as a matrix, or silently keeping it dense — produces a model that runs and measures
    /// nothing.
    pub fn adopts(&self, parameter: &str, role: Role) -> Result<Adopted> {
        let Self::Nvfp4 {
            excluded,
            also_excluded,
        } = self
        else {
            return Ok(Adopted::Source);
        };
        let excluded_here = excluded
            .iter()
            .any(|module| matches_module(module, parameter));
        // The two files' lists may differ in names neither of them quantizes, but a parameter they
        // disagree about is a checkpoint no single reading is right for.
        if !also_excluded.is_empty()
            && also_excluded
                .iter()
                .any(|module| matches_module(module, parameter))
                != excluded_here
        {
            return Err(Error::Config(format!(
                "the checkpoint's two declarations disagree about `{parameter}`: `config.json` \
                 {} it and `{DECLARATION_FILE}` {} it, and a packed tensor read as dense is a \
                 matrix of E2M1",
                if excluded_here { "excludes" } else { "quantizes" },
                if excluded_here { "quantizes" } else { "excludes" },
            )));
        }
        // A calibration quantizes a *module class* — ModelOpt's NVFP4 packs `Linear` and nothing
        // else — so a role outside that class is source by its class, not by being listed: the
        // checkpoint whose declaration this loader reads lists no norm and no vocabulary matrix, and
        // its shards carry them in BF16. The router is a `Linear` in the checkpoint's own module
        // tree, which is exactly why it has to be named in the exclusion list and why a calibration
        // that forgot to name it is a fact worth refusing rather than a nuisance.
        if !LinearClass::covers(role) {
            return Ok(Adopted::Source);
        }
        match (role, excluded_here) {
            (_, true) => Ok(Adopted::Source),
            (Role::Projection, false) => Ok(Adopted::Quantized),
            (role, false) => Err(Error::Config(format!(
                "the checkpoint's NVFP4 calibration quantizes `{parameter}`, which is {}, and this \
                 tree serves NVFP4 only for a linear projection: a quantized one would need a \
                 program this tree does not have, and reading it as dense would read packed E2M1 as \
                 a matrix. Exclude it in the calibration, or serve the role",
                role.description(),
            ))),
        }
    }

    /// The fields of one declaration, refused when this tree cannot serve them.
    fn from_fields(declared: &Declared) -> Result<Self> {
        let algorithm = declared.algorithm.to_ascii_uppercase();
        match algorithm.as_str() {
            // Absent, empty or an explicit "none" is a dense checkpoint that wrote the field anyway.
            "" | "NONE" | "BF16" => Ok(Self::Dense),
            "NVFP4" => {
                if let Some(method) = &declared.method
                    && !method.eq_ignore_ascii_case("modelopt")
                {
                    return Err(Error::Config(format!(
                        "the checkpoint's quantization method is `{method}`, and this tree serves \
                         ModelOpt's NVFP4 (the `weight`, `weight_scale`, `weight_scale_2` and \
                         `input_scale` spelling); a `{method}` export states the same numbers under \
                         names this loader would not find"
                    )));
                }
                Ok(Self::Nvfp4 {
                    excluded: declared.excluded.clone(),
                    also_excluded: declared.also_excluded.clone(),
                })
            }
            other => Err(Error::Config(format!(
                "the checkpoint declares quantization `{other}`, which this tree does not serve: it \
                 serves NVFP4 (ModelOpt's `weight`/`weight_scale`/`weight_scale_2` spelling) and an \
                 unquantized checkpoint. Loading `{other}` as if it were dense would read quantized \
                 storage as a matrix"
            ))),
        }
    }
}

/// Every parameter this model reads, with the role that decides how it may be stored.
///
/// The list is the boundary's own statement of what the model consumes: one entry per checkpoint
/// tensor, in the names the checkpoint uses. It is built from the configuration rather than from a
/// shard's contents, because a checkpoint's *shard* is a bag of tensors while the model's set is a
/// fact of its architecture — and the difference is what lets a declaration be checked before a
/// tensor is opened.
///
/// The router appears only on a layer that routes, and the dense MLP's three projections only on a
/// layer that does not, which is the checkpoint's own `mlp_only_layers` rule.
pub fn parameters(config: &Config) -> Vec<(String, Role)> {
    let mut parameters: Vec<(String, Role)> = vec![
        ("model.embed_tokens.weight".to_owned(), Role::Embedding),
        ("lm_head.weight".to_owned(), Role::Head),
        ("model.norm.weight".to_owned(), Role::Norm),
    ];
    for layer in 0..config.num_hidden_layers {
        let base = format!("model.layers.{layer}");
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            parameters.push((format!("{base}.self_attn.{projection}.weight"), Role::Projection));
        }
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            parameters.push((format!("{base}.{norm}.weight"), Role::Norm));
        }
        for norm in ["q_norm", "k_norm"] {
            parameters.push((format!("{base}.self_attn.{norm}.weight"), Role::Norm));
        }
        if config.sparse(layer) {
            parameters.push((format!("{base}.mlp.gate.weight"), Role::Router));
            for expert in 0..config.num_experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    parameters.push((
                        format!("{base}.mlp.experts.{expert}.{projection}.weight"),
                        Role::Projection,
                    ));
                }
            }
        } else {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                parameters.push((format!("{base}.mlp.{projection}.weight"), Role::Projection));
            }
        }
    }
    parameters
}

/// The module class an NVFP4 calibration packs, and the roles that fall in it.
///
/// The checkpoint's own module tree is the authority: `mlp.gate` and every projection are `Linear`,
/// the norms and the vocabulary matrices are not. The `Role` split follows that tree, so this is the
/// one place that decides what a calibration's statement can even be about.
struct LinearClass;

impl LinearClass {
    fn covers(role: Role) -> bool {
        match role {
            Role::Projection | Role::Router => true,
            Role::Embedding | Role::Head | Role::Norm => false,
        }
    }
}

/// A parameter's role, which decides whether a quantized form of it is served.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// A linear projection: attention `q/k/v/o`, a dense MLP's `gate/up/down`, an expert's three.
    Projection,
    /// The token embedding table.
    Embedding,
    /// The vocabulary head.
    Head,
    /// A normalization weight.
    Norm,
    /// The router's scoring projection.
    Router,
}

impl Role {
    /// The role as a refusal names it.
    const fn description(self) -> &'static str {
        match self {
            Self::Projection => "a linear projection",
            Self::Embedding => "the token embedding",
            Self::Head => "the vocabulary head",
            Self::Norm => "a normalization weight",
            Self::Router => "the router's scoring projection",
        }
    }
}

/// How the checkpoint stores a parameter, as the loader must read it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Adopted {
    /// The parameter is in the checkpoint's own dtype.
    Source,
    /// The parameter is ModelOpt NVFP4: packed values, block scales and a tensor scale.
    Quantized,
}

/// One declaration's fields, read from whichever file states them.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Declared {
    /// `quant_algo`, absent when the file states only a method.
    algorithm: String,
    /// `quant_method` (`modelopt`, `compressed-tensors`, …), when the export writes it.
    method: Option<String>,
    /// The modules the calibration left in their source dtype.
    excluded: Vec<String>,
    /// The other file's list, when both files state one; empty when only one file does.
    also_excluded: Vec<String>,
}

/// The declaration inside `config.json`, when it states one.
fn config_declaration(dir: &Path) -> Result<Option<Declared>> {
    let path = dir.join("config.json");
    let text = fs::read_to_string(&path).map_err(|source| Error::Io {
        path: path.display().to_string(),
        source,
    })?;
    let raw: Value = serde_json::from_str(&text).map_err(|source| Error::Json {
        path: path.display().to_string(),
        source,
    })?;
    raw.get("quantization_config")
        .map(|value| Declared::of(value, "config.json's `quantization_config`"))
        .transpose()
}

/// The declaration in the export's own file, when it wrote one.
fn side_declaration(dir: &Path) -> Result<Option<Declared>> {
    let path = dir.join(DECLARATION_FILE);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        // Absent is a fact about the checkpoint, not an error: the older exports write only this one
        // and a dense checkpoint writes neither.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(Error::Io {
                path: path.display().to_string(),
                source,
            });
        }
    };
    let raw: Value = serde_json::from_str(&text).map_err(|source| Error::Json {
        path: path.display().to_string(),
        source,
    })?;
    // The export nests its fields under `quantization`; a file that states them at the top level is
    // read the same way, because both spellings are in circulation and they name the same fields.
    let fields = raw.get("quantization").unwrap_or(&raw);
    if fields.get("quant_algo").is_none() && fields.get("quant_method").is_none() {
        return Ok(None);
    }
    Ok(Some(Declared::of(
        fields,
        "`hf_quant_config.json`'s `quantization`",
    )?))
}

impl Declared {
    /// The fields one object states, with `what` naming the place in a refusal.
    fn of(value: &Value, what: &str) -> Result<Self> {
        let string = |field: &str| -> Result<Option<String>> {
            match value.get(field) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(text)) => Ok(Some(text.clone())),
                Some(other) => Err(Error::Config(format!(
                    "{what} states `{field}` as {other}, and a string is expected"
                ))),
            }
        };
        let excluded = match value.get("exclude_modules") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| match item {
                    Value::String(name) => Ok(name.clone()),
                    other => Err(Error::Config(format!(
                        "{what} lists {other} among its excluded modules, and a name is expected"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
            Some(other) => {
                return Err(Error::Config(format!(
                    "{what} states `exclude_modules` as {other}, and a list of names is expected"
                )));
            }
        };
        // The compressed-tensors spelling states the same quantization as a group instead of an
        // algorithm name: `config_groups.group_0.weights` carries the bits, the kind and the block.
        let (algorithm, group_excluded) = match string("quant_algo")? {
            Some(algorithm) => (algorithm, Vec::new()),
            None => match value.get("config_groups") {
                Some(groups) => {
                    let algorithm = group_algorithm(groups, what)?;
                    let ignored = match value.get("ignore") {
                        None | Some(Value::Null) => Vec::new(),
                        Some(Value::Array(items)) => items
                            .iter()
                            .map(|item| {
                                item.as_str().map(str::to_owned).ok_or_else(|| {
                                    Error::Config(format!(
                                        "{what} lists {item} among its ignored modules, and a name \
                                         is expected"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>>>()?,
                        Some(other) => {
                            return Err(Error::Config(format!(
                                "{what} states `ignore` as {other}, and a list of names is expected"
                            )));
                        }
                    };
                    (algorithm, ignored)
                }
                None => (String::new(), Vec::new()),
            },
        };
        if !group_excluded.is_empty() && !excluded.is_empty() {
            return Err(Error::Config(format!(
                "{what} states both `ignore` and `exclude_modules`, and a file that says the same \
                 thing twice in two spellings is one this loader cannot trust"
            )));
        }
        Ok(Self {
            algorithm,
            // The compressed-tensors spelling is written by ModelOpt's own exporter, so it carries
            // no method of its own; the group it states is what pins the spelling (see below).
            method: string("quant_method")?.or(producer_method(value)),
            also_excluded: Vec::new(),
            excluded: if excluded.is_empty() { group_excluded } else { excluded },
        })
    }
}

/// The algorithm a compressed-tensors `config_groups` block states, refused when this tree does not
/// serve it.
///
/// The block names a bit width, a kind and a block size, and those three *are* the algorithm: four
/// bits of E2M1 over a block of sixteen with an F8_E4M3 block scale is NVFP4, and any other triple is
/// a quantization whose tensors this loader would read under names that mean something else. The
/// group's `targets` matter too: this tree quantizes linear projections, so a calibration that
/// targets another class states a checkpoint whose packing this model cannot follow.
fn group_algorithm(groups: &Value, what: &str) -> Result<String> {
    let groups = groups.as_object().ok_or_else(|| {
        Error::Config(format!(
            "{what} states `config_groups` as {groups}, and an object of groups is expected"
        ))
    })?;
    let mut algorithm: Option<String> = None;
    for (name, group) in groups {
        let weights = group.get("weights").ok_or_else(|| {
            Error::Config(format!("{what}'s group `{name}` states no `weights`"))
        })?;
        let bits = weights.get("num_bits").and_then(Value::as_u64);
        let kind = weights.get("type").and_then(Value::as_str);
        let block = weights.get("group_size").and_then(Value::as_u64);
        let this = match (bits, kind, block) {
            (Some(4), Some("float"), Some(16)) => "NVFP4",
            (bits, kind, block) => {
                return Err(Error::Config(format!(
                    "{what}'s group `{name}` states {bits:?} bits of {kind:?} over a block of \
                     {block:?}, and this tree serves NVFP4: four bits of E2M1 over a block of \
                     sixteen with an F8_E4M3 scale. Reading another quantization under those names \
                     would read its storage as a matrix"
                )));
            }
        };
        match group.get("targets") {
            None => {}
            Some(Value::Array(targets))
                if targets.iter().all(|target| target.as_str() == Some("Linear")) => {}
            Some(other) => {
                return Err(Error::Config(format!(
                    "{what}'s group `{name}` targets {other}, and this tree quantizes linear \
                     projections: a calibration that quantizes another module class states a \
                     checkpoint whose packing the model cannot follow"
                )));
            }
        }
        if let Some(previous) = &algorithm
            && previous != this
        {
            return Err(Error::Config(format!(
                "{what} states two quantizations at once, `{previous}` and `{this}`"
            )));
        }
        algorithm = Some(this.to_owned());
    }
    algorithm.ok_or_else(|| Error::Config(format!("{what}'s `config_groups` is empty")))
}

/// The producing tool, when the export names one instead of a method: ModelOpt writes
/// `producer: {"name": "modelopt", …}` in `hf_quant_config.json`.
fn producer_method(value: &Value) -> Option<String> {
    value
        .get("producer")
        .and_then(|producer| producer.get("name"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// Whether a declaration's excluded-module entry names `parameter`.
///
/// The exports write three shapes and all three appear in one checkpoint's list: a bare module name
/// (`lm_head`), a dotted prefix that excludes a subtree (`model.layers.0.mlp.gate`), and a pattern
/// with `*` (`model.layers.*.mlp.gate`). An entry matches when the parameter is the entry, is under
/// it, or matches it with `*` standing for any run of characters.
fn matches_module(entry: &str, parameter: &str) -> bool {
    // An entry names a *module*, and a module owns every parameter under it, so the comparison is
    // against the parameter's module path — the parameter without its final component — and a
    // subtree is excluded by naming its root.
    if entry == parameter {
        return true;
    }
    let module = parameter
        .rsplit_once('.')
        .map_or(parameter, |(module, _)| module);
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
    fn a_checkpoint_that_declares_nothing_is_dense() {
        let dir = tempdir();
        std::fs::write(dir.join("config.json"), r#"{"architectures": ["Qwen3MoeForCausalLM"]}"#)
            .unwrap();
        assert_eq!(Representation::read(&dir).unwrap(), Representation::Dense);
        // And every role is read in the checkpoint's own dtype.
        for role in [
            Role::Projection,
            Role::Embedding,
            Role::Head,
            Role::Norm,
            Role::Router,
        ] {
            assert_eq!(
                Representation::Dense
                    .adopts("model.layers.0.mlp.gate", role)
                    .unwrap(),
                Adopted::Source
            );
        }
    }

    #[test]
    fn an_nvfp4_declaration_quantizes_projections_and_leaves_the_excluded_modules_alone() {
        let dir = tempdir();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures": ["Qwen3MoeForCausalLM"],
                "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                    "exclude_modules": ["lm_head", "model.layers.*.mlp.gate",
                        "model.layers.0.input_layernorm"]}}"#,
        )
        .unwrap();
        let declared = Representation::read(&dir).unwrap();
        assert!(declared.is_encoded());
        // A projection the calibration covered is packed.
        assert_eq!(
            declared
                .adopts("model.layers.3.self_attn.q_proj.weight", Role::Projection)
                .unwrap(),
            Adopted::Quantized
        );
        // The three excluded shapes: a bare name, a pattern, and a dotted name.
        assert_eq!(
            declared.adopts("lm_head.weight", Role::Head).unwrap(),
            Adopted::Source
        );
        assert_eq!(
            declared
                .adopts("model.layers.7.mlp.gate.weight", Role::Router)
                .unwrap(),
            Adopted::Source
        );
        assert_eq!(
            declared
                .adopts("model.layers.0.input_layernorm.weight", Role::Norm)
                .unwrap(),
            Adopted::Source
        );
    }

    #[test]
    fn a_role_this_tree_serves_dense_is_refused_when_the_declaration_quantizes_it() {
        let dir = tempdir();
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                "exclude_modules": ["lm_head"]}}"#,
        )
        .unwrap();
        let declared = Representation::read(&dir).unwrap();
        let refused = declared
            .adopts("model.layers.4.mlp.gate.weight", Role::Router)
            .expect_err("a quantized router");
        let message = format!("{refused}");
        assert!(message.contains("model.layers.4.mlp.gate.weight"), "{message}");
        assert!(message.contains("the router's scoring projection"), "{message}");
        // The head is excluded by the checkpoint, so it is read as it is stored.
        assert_eq!(
            declared.adopts("lm_head.weight", Role::Head).unwrap(),
            Adopted::Source
        );
    }

    #[test]
    fn an_algorithm_this_tree_does_not_serve_is_refused_by_name() {
        for (algo, method) in [("FP8", "modelopt"), ("MXFP4", "modelopt"), ("W4A16", "compressed-tensors")] {
            let dir = tempdir();
            std::fs::write(
                dir.join("config.json"),
                format!(
                    r#"{{"quantization_config": {{"quant_method": "{method}", "quant_algo": "{algo}"}}}}"#
                ),
            )
            .unwrap();
            let refused = Representation::read(&dir).expect_err("an unserved algorithm");
            let message = format!("{refused}");
            assert!(
                message.contains(algo) || message.contains(&algo.to_ascii_uppercase()),
                "{message}"
            );
        }
        // And a method that is not ModelOpt's is refused even under the served algorithm, because
        // the same numbers under other names would not be found.
        let dir = tempdir();
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization_config": {"quant_method": "compressed-tensors", "quant_algo": "NVFP4"}}"#,
        )
        .unwrap();
        let refused = Representation::read(&dir).expect_err("a non-ModelOpt method");
        assert!(format!("{refused}").contains("compressed-tensors"), "{refused}");
    }

    #[test]
    fn the_side_file_declares_the_same_thing_and_a_disagreement_is_refused() {
        let dir = tempdir();
        std::fs::write(dir.join("config.json"), r#"{"architectures": []}"#).unwrap();
        std::fs::write(
            dir.join(DECLARATION_FILE),
            r#"{"producer": {"name": "modelopt"},
                "quantization": {"quant_algo": "NVFP4", "exclude_modules": ["lm_head"]}}"#,
        )
        .unwrap();
        assert_eq!(
            Representation::read(&dir).unwrap(),
            Representation::Nvfp4 {
                excluded: vec!["lm_head".to_owned()],
                also_excluded: Vec::new(),
            }
        );

        // A different algorithm is a different checkpoint, and no reading resolves it.
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}"#,
        )
        .unwrap();
        let refused = Representation::read(&dir).expect_err("two different declarations");
        assert!(format!("{refused}").contains("two different quantizations"), "{refused}");
    }

    #[test]
    fn two_files_that_disagree_about_a_projection_refuse_that_projection_and_no_other() {
        let dir = tempdir();
        // The two lists differ where it does not matter — a name neither file quantizes — and agree
        // where it does: `mlp.gate` is excluded by both, the projections by neither.
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                "exclude_modules": ["model.layers.0.mlp.gate"]}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join(DECLARATION_FILE),
            r#"{"quantization": {"quant_algo": "NVFP4",
                "exclude_modules": ["model.layers.0.mlp.gate", "lm_head"]}}"#,
        )
        .unwrap();
        let representation = Representation::read(&dir).expect("a consistent declaration");
        assert_eq!(
            representation
                .adopts("model.layers.0.self_attn.q_proj.weight", Role::Projection)
                .expect("a packed projection"),
            Adopted::Quantized
        );
        assert_eq!(
            representation
                .adopts("model.layers.0.mlp.gate.weight", Role::Router)
                .expect("an excluded router"),
            Adopted::Source
        );

        // Now the one disagreement that matters: a projection one file packs and the other does not.
        std::fs::write(
            dir.join(DECLARATION_FILE),
            r#"{"quantization": {"quant_algo": "NVFP4",
                "exclude_modules": ["model.layers.0.mlp.gate", "model.layers.0.self_attn.q_proj"]}}"#,
        )
        .unwrap();
        let representation = Representation::read(&dir).expect("the algorithm still agrees");
        let refused = representation
            .adopts("model.layers.0.self_attn.q_proj.weight", Role::Projection)
            .expect_err("a projection the two files disagree about");
        let message = format!("{refused}");
        assert!(message.contains("disagree about"), "{message}");
        assert!(message.contains("q_proj"), "{message}");
        // A parameter they agree about is still answered normally.
        assert_eq!(
            representation
                .adopts("model.layers.3.self_attn.o_proj.weight", Role::Projection)
                .expect("a projection both files pack"),
            Adopted::Quantized
        );
    }

    #[test]
    fn the_calibrated_checkpoints_compressed_tensors_block_is_read_as_nvfp4() {
        // The shape `Qwen3-30B-A3B-NVFP4`'s `config.json` states: the calibration as a group (bits,
        // kind, block, target class) rather than as an algorithm name, and its excluded modules under
        // `ignore`. Written by ModelOpt's compressed-tensors exporter, so the block is the whole
        // statement and the side file's `producer` is the method.
        let dir = tempdir();
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization_config": {
                "config_groups": {"group_0": {
                    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float",
                                          "group_size": 16},
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float",
                                "group_size": 16},
                    "targets": ["Linear"]}},
                "ignore": ["model.layers.0.mlp.gate", "model.layers.1.mlp.gate"]}}"#,
        )
        .unwrap();
        let representation = Representation::read(&dir).expect("the calibrated declaration");
        assert!(representation.is_encoded());
        assert_eq!(
            representation
                .adopts("model.layers.0.self_attn.q_proj.weight", Role::Projection)
                .expect("a packed projection"),
            Adopted::Quantized
        );
        assert_eq!(
            representation
                .adopts("model.layers.0.mlp.experts.7.down_proj.weight", Role::Projection)
                .expect("a packed expert projection"),
            Adopted::Quantized
        );
        assert_eq!(
            representation
                .adopts("model.layers.0.mlp.gate.weight", Role::Router)
                .expect("the router is ignored by name"),
            Adopted::Source
        );
        assert_eq!(
            representation
                .adopts("model.embed_tokens.weight", Role::Embedding)
                .expect("the vocabulary matrix is not a linear projection"),
            Adopted::Source
        );

        // The block *is* the algorithm: four bits of a float over sixteen is NVFP4, and another
        // triple is a packing whose tensors these names would misread.
        for (bits, kind, block) in [(8, "float", 16), (4, "int", 16), (4, "float", 32)] {
            std::fs::write(
                dir.join("config.json"),
                format!(
                    r#"{{"quantization_config": {{
                        "config_groups": {{"group_0": {{
                            "weights": {{"num_bits": {bits}, "type": "{kind}",
                                         "group_size": {block}}},
                            "targets": ["Linear"]}}}}}}}}"#
                ),
            )
            .unwrap();
            let refused = Representation::read(&dir).expect_err("an unserved packing");
            let message = format!("{refused}");
            assert!(message.contains("this tree serves NVFP4"), "{message}");
        }
    }

    /// A directory under the system's temporary root, removed when the test ends.
    fn tempdir() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "qwen3-moe-representation-{}-{serial}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
