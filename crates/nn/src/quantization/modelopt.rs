//! The manifest of a packed ModelOpt export: the calibration the export states beside its
//! tensors.
//!
//! A packed NVFP4 export of a diffusers pipeline writes `modelopt_manifest.json` at the
//! pipeline's root. Its calibrated projections store their fields in the diffusers spelling
//! ([`Nvfp4Names::diffusers`]: `weight_packed`, `weight_scale`, `weight_tensor_scale`) and no
//! activation scale beside them: the manifest lists every calibrated module with the largest
//! activation magnitude its calibration measured, and the activation's tensor scale is that
//! magnitude in ModelOpt's scale domain,
//!
//! ```text
//! activation scale = activation_amax / (6 · 448)
//! ```
//!
//! six being the largest E2M1 magnitude and 448 the largest E4M3 one, so the largest calibrated
//! activation encodes at the largest code under the largest block scale. It is the number a
//! unified export stores as its `input_scale` tensor.
//!
//! ## Module paths
//!
//! The manifest lists a module by its path in the exporter's own module tree, which is neither
//! the checkpoint's tensor names nor shared between model families. [`ModeloptManifest`] keeps
//! the paths as listed; the model crate that owns the checkpoint's names maps each one to the
//! module of the stage that stores it ([`ModeloptManifest::declaration`]).

use std::collections::BTreeMap;

use ops::fp4::BLOCK_ELEMENTS;
use ops::fp8::E4M3_MAX;
use serde_json::Value;

use super::{Declaration, KvCacheScheme, Nvfp4Names, Scheme};
use crate::config::Fields;
use crate::error::Result;

/// The manifest version this reader reads.
const SCHEMA_VERSION: u32 = 1;

/// The key the manifest lists its pipeline stages under.
const STAGES: &str = "components";

/// The representation a packed export states under `numerical_format`, beside its block size:
/// static NVFP4 weights and activations (`w4`, `a4`) of E2M1 values under E4M3 block scales and
/// an FP32 tensor scale. Another value under any of these names is another quantization stored
/// under the same tensor names.
const NUMERICAL_FORMAT: [(&str, &str); 6] = [
    ("activation", "a4"),
    ("block_scale", "fp8_e4m3"),
    ("output", "bf16"),
    ("tensor_scale", "fp32"),
    ("values", "e2m1"),
    ("weight", "w4"),
];

/// The largest magnitude an E2M1 code states.
const E2M1_MAX: f64 = 6.0;

/// The calibration a packed ModelOpt export states in its `modelopt_manifest.json`.
#[derive(Clone, PartialEq, Debug)]
pub struct ModeloptManifest {
    /// Each calibrated module's activation tensor scale, by the path the manifest lists the
    /// module under.
    activations: BTreeMap<String, f32>,
}

impl ModeloptManifest {
    /// The manifest's file, at the root of the export.
    pub const FILE: &'static str = "modelopt_manifest.json";

    /// The manifest `value` states.
    ///
    /// # Errors
    ///
    /// [`Error::Config`](crate::Error::Config) naming the field when the manifest is another
    /// version, states another numerical format than static NVFP4 over sixteen-element blocks,
    /// lists a stage whose `enabled` and `modules` disagree (an enabled stage with no module,
    /// or modules under a stage that is not enabled), lists a module without a positive finite
    /// `activation_amax` or under two stages, or calibrates no module at all.
    pub fn parse(value: &Value) -> Result<Self> {
        let manifest = Fields::of(Self::FILE, value)?;

        let version = manifest.index("schema_version")?;
        if version != SCHEMA_VERSION {
            return Err(manifest.refuse(
                "schema_version",
                format!("is {version}, and the manifest read here is version {SCHEMA_VERSION}"),
            ));
        }
        numerical_format(&manifest.object("numerical_format")?)?;

        let stages = manifest.object(STAGES)?;
        let mut activations = BTreeMap::new();

        for name in stages.keys() {
            let stage = stages.object(name)?;
            let enabled = stage.flag("enabled")?;
            let modules = stage.object("modules")?;

            // A stage is calibrated exactly when it lists modules: either statement alone is a
            // manifest whose export cannot be told from a half-written one.
            let listed = modules.keys().count();
            if enabled && listed == 0 {
                return Err(stage.refuse(
                    "modules",
                    "lists no module, and the stage is enabled: a calibrated stage states each \
                     module's activation scale",
                ));
            }
            if !enabled && listed > 0 {
                return Err(stage.refuse(
                    "modules",
                    format!("lists {listed} calibrated modules, and the stage is not enabled"),
                ));
            }

            for path in modules.keys() {
                let module = modules.object(path)?;
                let amax = module.positive("activation_amax")?;

                // The scale is a launch value of an FP32 kernel operand: a magnitude whose
                // scale is not a positive FP32 number encodes nothing.
                let scale = (amax / (E2M1_MAX * f64::from(E4M3_MAX))) as f32;
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(module.refuse(
                        "activation_amax",
                        format!("is {amax}, whose tensor scale is not a positive FP32 number"),
                    ));
                }

                if activations.insert(path.to_owned(), scale).is_some() {
                    return Err(modules.refuse(
                        path,
                        "is listed under two stages, and a module is calibrated once",
                    ));
                }
            }
        }

        if activations.is_empty() {
            return Err(manifest.refuse(
                STAGES,
                "calibrates no module: a packed export states the activation scale of every \
                 projection it packs",
            ));
        }
        Ok(Self { activations })
    }

    /// The paths the manifest lists its calibrated modules under, in lexical order.
    pub fn modules(&self) -> impl Iterator<Item = &str> {
        self.activations.keys().map(String::as_str)
    }

    /// What the manifest declares of one stage of its export: the stage's packed projections in
    /// the diffusers spelling, each with the activation scale the manifest lists for it.
    ///
    /// `module` maps a listed path to the module that stores it in the stage's checkpoint, as
    /// the checkpoint names it (`transformer_blocks.0.ff.net.2`), and returns `None` for a path
    /// of another stage. Several listed paths may name one stored module (the two halves of a
    /// projection the checkpoint stores fused), and they must then agree. A stage none of the
    /// listed paths belong to is dense.
    ///
    /// # Errors
    ///
    /// [`Error::Config`](crate::Error::Config) naming the listed path when two paths of one
    /// stored module state different activation scales.
    pub fn declaration(&self, module: impl Fn(&str) -> Option<String>) -> Result<Declaration> {
        let mut activations = BTreeMap::new();

        for (listed, scale) in &self.activations {
            let Some(stored) = module(listed) else {
                continue;
            };

            if let Some(previous) = activations.insert(stored.clone(), *scale)
                && previous != *scale
            {
                return Err(crate::Error::config(
                    Self::FILE,
                    listed.as_str(),
                    format!(
                        "calibrates the activations of `{stored}` at a tensor scale of {scale}, \
                         and another path of the same stored projection at {previous}: one \
                         product encodes its activations once"
                    ),
                ));
            }
        }

        if activations.is_empty() {
            return Ok(Declaration::dense());
        }
        Ok(Declaration {
            blocks: None,
            scheme: Scheme::Nvfp4,
            excluded: Vec::new(),
            kv_cache: KvCacheScheme::Unquantized,
            packed: None,
            layers: None,
            names: Nvfp4Names::diffusers(),
            activations: Some(activations),
            converted: Vec::new(),
        })
    }
}

/// Checks that `format`, the manifest's `numerical_format`, states static NVFP4 over
/// sixteen-element blocks and nothing else.
fn numerical_format(format: &Fields<'_>) -> Result<()> {
    for (key, expected) in NUMERICAL_FORMAT {
        let stated = format.text(key)?;
        if stated != expected {
            return Err(format.refuse(
                key,
                format!("is `{stated}`, and a packed NVFP4 export states `{expected}`"),
            ));
        }
    }

    let block = format.count("block_size")?;
    if block != BLOCK_ELEMENTS {
        return Err(format.refuse(
            "block_size",
            format!("is {block}, and an NVFP4 block scale covers {BLOCK_ELEMENTS} elements"),
        ));
    }

    // A field this reader does not know states something about the representation that the
    // loader would then ignore.
    if let Some(unknown) = format
        .keys()
        .find(|key| *key != "block_size" && !NUMERICAL_FORMAT.iter().any(|(known, _)| known == key))
    {
        return Err(format.refuse(
            unknown,
            "is not a field of the static NVFP4 format this reader reads",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Error;

    /// The released four-step FastH3 export's manifest, trimmed to the first layer of each
    /// calibrated stage: every field the export writes, the three stages, and one module of
    /// each kind the export lists.
    fn released() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/modelopt_manifest.json")).unwrap()
    }

    /// The field and the problem of a configuration refusal.
    fn refusal(error: Error) -> (String, String) {
        match error {
            Error::Config { field, problem, .. } => (field, problem),
            other => panic!("not a configuration refusal: {other}"),
        }
    }

    #[test]
    fn the_released_manifest_lists_each_calibrated_module_with_its_activation_scale() {
        let manifest = ModeloptManifest::parse(&released()).unwrap();

        // Three paths of the denoiser's first block and seven of the video decoder's; the text
        // encoder is listed and not calibrated.
        let modules: Vec<&str> = manifest.modules().collect();
        assert_eq!(modules.len(), 10, "{modules:?}");
        assert!(modules.contains(&"denoiser.transformer.layers.0.mlp.down"));
        assert!(modules.contains(&"video_decoder.decoder.decoder.decoder.layers.0.output"));
        assert!(!modules.iter().any(|path| path.starts_with("text_encoder")));

        // An activation's tensor scale is its calibrated magnitude over 6 · 448.
        let down = manifest.activations["denoiser.transformer.layers.0.mlp.down"];
        assert_eq!(down, (602.315_246_582_031_2_f64 / 2688.0) as f32);
    }

    #[test]
    fn a_stage_s_declaration_holds_its_stored_modules_in_the_diffusers_spelling() {
        let manifest = ModeloptManifest::parse(&released()).unwrap();

        // The denoiser's stored modules: the gate and the up halves are one stored projection.
        let stored = |listed: &str| {
            let rest = listed.strip_prefix("denoiser.transformer.layers.0.mlp.")?;
            Some(
                if rest == "down" {
                    "blocks.0.ff.net.2"
                } else {
                    "blocks.0.ff.net.0.proj"
                }
                .to_owned(),
            )
        };
        let declaration = manifest.declaration(stored).unwrap();

        assert_eq!(declaration.names(), Nvfp4Names::diffusers());
        assert!(declaration.is_encoded());
        assert_eq!(
            declaration.activation_scale("blocks.0.ff.net.2").unwrap(),
            Some((602.315_246_582_031_2_f64 / 2688.0) as f32)
        );
        assert_eq!(
            declaration
                .activation_scale("blocks.0.ff.net.0.proj")
                .unwrap(),
            Some((485.011_444_091_796_9_f64 / 2688.0) as f32)
        );

        // A packed module the manifest does not calibrate has no activation scale to encode by.
        let (field, problem) = refusal(
            declaration
                .activation_scale("blocks.1.ff.net.2")
                .unwrap_err(),
        );
        assert_eq!(field, "blocks.1.ff.net.2");
        assert!(problem.contains("lists no calibration"), "{problem}");

        // A stage none of the listed paths belong to is dense.
        assert_eq!(
            manifest.declaration(|_| None).unwrap(),
            Declaration::dense()
        );
    }

    #[test]
    fn two_paths_of_one_stored_projection_must_agree() {
        let gate = "denoiser.transformer.layers.0.mlp.gate_up.projections.gate";
        let mut value = released();
        value["components"]["denoiser"]["modules"][gate]["activation_amax"] = json!(486.0);
        let manifest = ModeloptManifest::parse(&value).unwrap();

        let error = manifest
            .declaration(|listed| {
                listed
                    .contains("gate_up")
                    .then(|| "blocks.0.ff.net.0.proj".to_owned())
            })
            .unwrap_err();
        let (field, problem) = refusal(error);
        assert!(field.contains("gate_up.projections"), "{field}");
        assert!(problem.contains("blocks.0.ff.net.0.proj"), "{problem}");
    }

    #[test]
    fn a_missing_or_inconsistent_entry_is_refused_by_its_field() {
        let refused = |edit: &dyn Fn(&mut Value)| {
            let mut value = released();
            edit(&mut value);
            refusal(ModeloptManifest::parse(&value).unwrap_err())
        };
        let down = "denoiser.transformer.layers.0.mlp.down";

        // A listed module without its calibrated magnitude, or with one that scales nothing.
        let (field, problem) = refused(&|value| {
            value["components"]["denoiser"]["modules"][down] = json!({});
        });
        assert_eq!(
            (field.as_str(), problem.as_str()),
            ("activation_amax", "is missing")
        );

        for amax in [json!(0.0), json!(-3.5), json!("602.3"), json!(true)] {
            let (field, _) = refused(&|value| {
                value["components"]["denoiser"]["modules"][down]["activation_amax"] = amax.clone();
            });
            assert_eq!(field, "activation_amax", "{amax}");
        }

        // A stage whose two statements disagree.
        let (field, problem) = refused(&|value| {
            value["components"]["denoiser"]["enabled"] = json!(false);
        });
        assert_eq!(field, "modules");
        assert!(problem.contains("not enabled"), "{problem}");

        let (field, problem) = refused(&|value| {
            value["components"]["text_encoder"]["enabled"] = json!(true);
        });
        assert_eq!(field, "modules");
        assert!(problem.contains("lists no module"), "{problem}");

        // A module listed under two stages.
        let (field, _) = refused(&|value| {
            value["components"]["text_encoder"] = json!({
                "enabled": true,
                "modules": {down: {"activation_amax": 1.0}},
            });
        });
        assert_eq!(field, down);

        // Another representation under the same tensor names, another version, no stage.
        let (field, problem) = refused(&|value| {
            value["numerical_format"]["activation"] = json!("a16");
        });
        assert_eq!(field, "activation");
        assert!(problem.contains("`a4`"), "{problem}");

        let (field, _) = refused(&|value| {
            value["numerical_format"]["block_size"] = json!(32);
        });
        assert_eq!(field, "block_size");

        let (field, _) = refused(&|value| {
            value["numerical_format"]["rounding"] = json!("stochastic");
        });
        assert_eq!(field, "rounding");

        let (field, _) = refused(&|value| {
            value["schema_version"] = json!(2);
        });
        assert_eq!(field, "schema_version");

        let (field, _) = refused(&|value| {
            value.as_object_mut().unwrap().remove("components");
        });
        assert_eq!(field, "components");

        let (field, problem) = refused(&|value| {
            for stage in ["denoiser", "video_vae"] {
                value["components"][stage] = json!({"enabled": false, "modules": {}});
            }
        });
        assert_eq!(field, "components");
        assert!(problem.contains("calibrates no module"), "{problem}");
    }
}
