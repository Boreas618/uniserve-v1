//! The deployment's precision policy (`quantization`), interpreted against a checkpoint.
//!
//! A policy names a mode and selects representations by module, under four names: `attention`
//! (the denoisers' attention projections), `mlp` (the denoisers' feed-forward projections),
//! `text_encoder` (the text encoder's language layers) and `video_vae` (the video VAE decoder's
//! projections). Each mode gives the four these representations (the reference's tiers):
//!
//! ```text
//! mode          attention  mlp    text_encoder  video_vae
//! quality       bf16       bf16   bf16          fp16
//! balanced      bf16       bf16   bf16          nvfp4
//! performance   bf16       fp8    bf16          nvfp4
//! maximum       bf16       nvfp4  fp8           nvfp4
//! ```
//!
//! A selection under `modules` replaces its module's representation; a policy that states no
//! mode takes `quality`'s for the modules it does not select.
//!
//! A dense checkpoint keeps its own representation for a module in `bf16` (the video VAE
//! decoder: `fp16`, the type its reference computes in). A module in `fp8` or `nvfp4` has its
//! projections converted when the stage is loaded, and their products encode each activation at
//! a scale measured from it on every call ([`nn::quantization::Scheme::DynamicFp8`],
//! [`nn::quantization::Scheme::DynamicNvfp4`]); [`apply`] states those projections in the
//! declaration of the stage that stores them, before any weight is read:
//!
//! ```text
//! module        stage                  projections (module patterns of the stage)
//! attention     every denoiser network transformer_blocks.*.attn.{to_q, to_k, to_v,
//!                                      to_gate_compress, to_out.0}
//! mlp           every denoiser network transformer_blocks.*.ff.{net.0.proj, net.2}
//! text_encoder  text_encoder           model.language_model.layers.*.self_attn.{q,k,v,o}_proj,
//!                                      model.language_model.layers.*.mlp.{gate,up,down}_proj
//! video_vae     vae                    decoder.transformer_blocks.*.attn.{to_q, to_k, to_v,
//!                                      to_out.0}, decoder.transformer_blocks.*.ff.{net.0.proj,
//!                                      net.2}
//! ```
//!
//! The video VAE's set is the one a packed NVFP4 export calibrates, and its decoder computes in
//! FP16 under it as under the export. `video_vae = "bf16"` is refused: the decoder's raster, its
//! splice and its codec are FP16. A packed ModelOpt export states each module's representation in
//! its manifest and takes no policy.

use nn::Result;
use nn::quantization::{Declaration, Scheme};
use proto::config::{PrecisionMode, Quantization, WeightFormat};

use crate::minimax_h3::{Config, ID, transformer, video_vae};

/// The modules a policy selects, in the order of every table below.
const MODULES: [&str; 4] = ["attention", "mlp", "text_encoder", "video_vae"];

/// The positions of the modules in [`MODULES`].
const ATTENTION: usize = 0;
const MLP: usize = 1;
const TEXT_ENCODER: usize = 2;
const VIDEO_VAE: usize = 3;

/// The language layers of the text encoder, as its checkpoint stores them, and below a layer
/// the projections of its attention and of its feed-forward.
const LANGUAGE_LAYERS: &str = "model.language_model.layers";
const TEXT_ATTENTION: [&str; 4] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
];
const TEXT_FEED_FORWARD: [&str; 3] = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];

/// Each mode's representation of [`MODULES`].
const fn tier(mode: PrecisionMode) -> [WeightFormat; 4] {
    use WeightFormat::{Bf16, Fp8, Fp16, Nvfp4};
    match mode {
        PrecisionMode::Quality => [Bf16, Bf16, Bf16, Fp16],
        PrecisionMode::Balanced => [Bf16, Bf16, Bf16, Nvfp4],
        PrecisionMode::Performance => [Bf16, Fp8, Bf16, Nvfp4],
        PrecisionMode::Maximum => [Bf16, Nvfp4, Fp8, Nvfp4],
    }
}

/// The representations a selection may give each of [`MODULES`].
const SELECTABLE: [&[WeightFormat]; 4] = {
    use WeightFormat::{Bf16, Fp8, Fp16, Nvfp4};
    [
        &[Bf16, Fp8, Nvfp4],
        &[Bf16, Fp8, Nvfp4],
        &[Bf16, Fp8, Nvfp4],
        &[Fp16, Bf16, Nvfp4],
    ]
};

/// The scheme a projection of a module in `format` is converted to at load, or `None` for the
/// checkpoint's own representation.
const fn conversion(format: WeightFormat) -> Option<Scheme> {
    match format {
        WeightFormat::Fp8 => Some(Scheme::DynamicFp8),
        WeightFormat::Nvfp4 => Some(Scheme::DynamicNvfp4),
        WeightFormat::Bf16 | WeightFormat::Fp16 => None,
    }
}

/// The module patterns `prefix.*.<projection>` of `projections`, each converted to `scheme`.
fn patterns(prefix: &str, projections: &[&str], scheme: Scheme) -> Vec<(String, Scheme)> {
    projections
        .iter()
        .map(|projection| (format!("{prefix}.*.{projection}"), scheme))
        .collect()
}

/// Applies `policy` to the checkpoint `config` describes: each stage's declaration names the
/// projections the policy converts at load. An empty policy leaves the checkpoint's own
/// representation.
///
/// # Errors
///
/// [`Error::Unsupported`] naming the policy when the checkpoint declares its own quantization;
/// when a selection names a module the family does not have or a representation its module does
/// not take; or when the video VAE resolves to `bf16`, which its decoder does not compute in.
pub(crate) fn apply(policy: &Quantization, config: &mut Config) -> Result<()> {
    if policy.is_empty() {
        return Ok(());
    }

    // A packed export's manifest states each stage's representation.
    let dense = Declaration::dense();
    let declared = config
        .denoisers
        .values()
        .map(|denoiser| &denoiser.quantization)
        .chain([
            &config.text_encoder.quantization,
            &config.video_vae.quantization,
        ])
        .any(|declaration| *declaration != dense);
    if declared {
        return Err(ID.unsupported(format!(
            "`quantization = {policy}`: checkpoint format 'modelopt_nvfp4' owns its numerical \
             configuration (its `{}` states each module's representation); drop `quantization`",
            nn::quantization::ModeloptManifest::FILE
        )));
    }

    for (module, format) in &policy.modules {
        let Some(index) = MODULES.iter().position(|name| name == module) else {
            return Err(ID.unsupported(format!(
                "`quantization.modules.{module}`: MiniMax-H3 selects the representation of {}, \
                 not of `{module}`",
                listed(MODULES.iter().map(|name| format!("`{name}`")))
            )));
        };
        if !SELECTABLE[index].contains(format) {
            return Err(ID.unsupported(format!(
                "`quantization.modules.{module} = \"{format}\"`: `{module}` takes {}",
                listed(SELECTABLE[index].iter().map(|format| format!("`{format}`")))
            )));
        }
    }

    let given = tier(policy.mode.unwrap_or(PrecisionMode::Quality));
    let mut resolved = given;
    for (index, module) in MODULES.iter().enumerate() {
        if let Some(format) = policy.modules.get(*module) {
            resolved[index] = *format;
        }
    }

    if resolved[VIDEO_VAE] == WeightFormat::Bf16 {
        return Err(ID.unsupported(format!(
            "`quantization = {policy}` gives `video_vae` in bf16, and the video VAE decoder \
             computes in fp16, the type its reference computes in: its raster, splice and codec \
             read fp16; select `video_vae` in fp16 or nvfp4"
        )));
    }

    // The denoiser networks: every block's attention and feed-forward projections.
    let blocks = transformer::BLOCKS;
    let mut network = Vec::new();
    if let Some(scheme) = conversion(resolved[ATTENTION]) {
        let attention: Vec<String> = transformer::INPUTS
            .iter()
            .chain([&transformer::ATTENTION_OUTPUT])
            .map(|projection| format!("{}.{projection}", transformer::ATTENTION))
            .collect();
        let attention: Vec<&str> = attention.iter().map(String::as_str).collect();
        network.extend(patterns(blocks, &attention, scheme));
    }
    if let Some(scheme) = conversion(resolved[MLP]) {
        let feed_forward: Vec<&str> = transformer::FEED_FORWARD.to_vec();
        network.extend(patterns(blocks, &feed_forward, scheme));
    }
    for denoiser in config.denoisers.values_mut() {
        denoiser.quantization = Declaration::converted(network.clone())?;
    }

    // The text encoder's language layers; its vision tower keeps its own representation.
    if let Some(scheme) = conversion(resolved[TEXT_ENCODER]) {
        let projections: Vec<&str> = TEXT_ATTENTION
            .iter()
            .chain(&TEXT_FEED_FORWARD)
            .copied()
            .collect();
        config.text_encoder.quantization =
            Declaration::converted(patterns(LANGUAGE_LAYERS, &projections, scheme))?;
    }

    // The video VAE decoder's blocks: the attention's four projections and the feed-forward's
    // two.
    if let Some(scheme) = conversion(resolved[VIDEO_VAE]) {
        let projections: Vec<String> = video_vae::QKV
            .iter()
            .chain([&video_vae::ATTENTION_OUTPUT])
            .map(|projection| format!("{}.{projection}", video_vae::ATTENTION))
            .chain(
                video_vae::FEED_FORWARD_PROJECTIONS
                    .iter()
                    .map(|projection| format!("{}.{projection}", video_vae::FEED_FORWARD)),
            )
            .collect();
        let projections: Vec<&str> = projections.iter().map(String::as_str).collect();
        config.video_vae.quantization =
            Declaration::converted(patterns(video_vae::BLOCKS, &projections, scheme))?;
    }
    Ok(())
}

/// `items` as prose: `a`, `a and b`, `a, b and c`.
fn listed(items: impl Iterator<Item = String>) -> String {
    let items: Vec<String> = items.collect();
    match items.split_last() {
        None => String::new(),
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use crate::minimax_h3::stage::Stage;
    use nn::quantization::ModeloptManifest;

    use super::*;
    use crate::minimax_h3::config::tests_support::{dmd_export, read, release};

    /// A dense FastH3 export's configuration.
    fn dense() -> Config {
        read(&dmd_export(release(), "VIDEO_SPARSE_ATTN_H3"), None).unwrap()
    }

    fn policy(mode: Option<PrecisionMode>, modules: &[(&str, WeightFormat)]) -> Quantization {
        Quantization {
            mode,
            modules: modules
                .iter()
                .map(|(module, format)| ((*module).to_owned(), *format))
                .collect(),
        }
    }

    /// The configuration `policy` gives the dense export.
    fn applied(policy: &Quantization) -> Config {
        let mut config = dense();
        apply(policy, &mut config).unwrap_or_else(|error| panic!("{policy}: {error}"));
        config
    }

    /// The scheme each module's representative weights are read in under `config` over a dense
    /// checkpoint, which reads a weight no conversion names dense: a denoiser
    /// block's query and gate-compress projections, its feed-forward's two, a language layer's
    /// query and down projections, and a decoder block's value and feed-forward projections.
    fn schemes(config: &Config) -> [[Scheme; 2]; 4] {
        let network = &config.denoisers[&Stage::Transformer].quantization;
        let text = &config.text_encoder.quantization;
        let decoder = &config.video_vae.quantization;
        [
            [
                network
                    .conversion("transformer_blocks.7.attn.to_q.weight")
                    .unwrap_or(Scheme::Dense),
                network
                    .conversion("transformer_blocks.49.attn.to_gate_compress.weight")
                    .unwrap_or(Scheme::Dense),
            ],
            [
                network
                    .conversion("transformer_blocks.0.ff.net.0.proj.weight")
                    .unwrap_or(Scheme::Dense),
                network
                    .conversion("transformer_blocks.3.ff.net.2.weight")
                    .unwrap_or(Scheme::Dense),
            ],
            [
                text.conversion("model.language_model.layers.0.self_attn.q_proj.weight")
                    .unwrap_or(Scheme::Dense),
                text.conversion("model.language_model.layers.49.mlp.down_proj.weight")
                    .unwrap_or(Scheme::Dense),
            ],
            [
                decoder
                    .conversion("decoder.transformer_blocks.35.attn.to_v.weight")
                    .unwrap_or(Scheme::Dense),
                decoder
                    .conversion("decoder.transformer_blocks.0.ff.net.0.proj.weight")
                    .unwrap_or(Scheme::Dense),
            ],
        ]
    }

    /// Each mode converts exactly the modules its tier narrows, to the scheme of the
    /// representation it gives them, and nothing else: the norms, the biases, the refiner, the
    /// vision tower and the decoder's own end projections stay dense.
    #[test]
    fn each_mode_converts_the_projections_of_the_modules_its_tier_narrows() {
        use Scheme::{Dense, DynamicFp8, DynamicNvfp4};

        for (policy, expected) in [
            (Quantization::default(), [Dense, Dense, Dense, Dense]),
            (
                policy(Some(PrecisionMode::Quality), &[]),
                [Dense, Dense, Dense, Dense],
            ),
            (
                policy(Some(PrecisionMode::Balanced), &[]),
                [Dense, Dense, Dense, DynamicNvfp4],
            ),
            (
                policy(Some(PrecisionMode::Performance), &[]),
                [Dense, DynamicFp8, Dense, DynamicNvfp4],
            ),
            (
                policy(Some(PrecisionMode::Maximum), &[]),
                [Dense, DynamicNvfp4, DynamicFp8, DynamicNvfp4],
            ),
        ] {
            let config = applied(&policy);
            assert_eq!(
                schemes(&config),
                expected.map(|scheme| [scheme; 2]),
                "{policy}"
            );
        }

        let config = applied(&policy(Some(PrecisionMode::Maximum), &[]));
        let network = &config.denoisers[&Stage::Transformer].quantization;
        let text = &config.text_encoder.quantization;
        let decoder = &config.video_vae.quantization;
        for (declaration, weight) in [
            (network, "transformer_blocks.0.ff.net.2.bias"),
            (network, "transformer_blocks.0.norm1.weight"),
            (network, "transformer_blocks.0.attn.norm_q.weight"),
            (network, "token_refiner.refiner_blocks.0.mlp.fc1.weight"),
            (text, "model.visual.blocks.0.mlp.linear_fc1.weight"),
            (text, "model.language_model.layers.0.input_layernorm.weight"),
            (decoder, "decoder.proj_out.weight"),
            (decoder, "decoder.transformer_blocks.0.attn.to_q.bias"),
            (decoder, "encoder.down_blocks.0.resnets.0.conv1.conv.weight"),
        ] {
            assert_eq!(declaration.conversion(weight), None, "{weight}");
        }
    }

    /// A selection replaces what its mode gives its module, in either direction, and selects
    /// a conversion of a module no mode narrows.
    #[test]
    fn a_selection_replaces_its_modules_tier() {
        use Scheme::{Dense, DynamicFp8, DynamicNvfp4};

        for (policy, expected) in [
            (
                policy(
                    Some(PrecisionMode::Performance),
                    &[
                        ("mlp", WeightFormat::Bf16),
                        ("video_vae", WeightFormat::Fp16),
                    ],
                ),
                [Dense, Dense, Dense, Dense],
            ),
            (
                policy(None, &[("attention", WeightFormat::Fp8)]),
                [DynamicFp8, Dense, Dense, Dense],
            ),
            (
                policy(
                    Some(PrecisionMode::Balanced),
                    &[
                        ("attention", WeightFormat::Nvfp4),
                        ("text_encoder", WeightFormat::Nvfp4),
                    ],
                ),
                [DynamicNvfp4, Dense, DynamicNvfp4, DynamicNvfp4],
            ),
        ] {
            let config = applied(&policy);
            assert_eq!(
                schemes(&config),
                expected.map(|scheme| [scheme; 2]),
                "{policy}"
            );
        }
    }

    /// The video VAE decoder computes in FP16: a policy that resolves it to BF16 is refused,
    /// naming the module.
    #[test]
    fn a_video_vae_in_bf16_is_refused() {
        for refused in [
            policy(None, &[("video_vae", WeightFormat::Bf16)]),
            policy(
                Some(PrecisionMode::Maximum),
                &[("video_vae", WeightFormat::Bf16)],
            ),
        ] {
            let message = apply(&refused, &mut dense()).unwrap_err().to_string();
            assert!(message.contains("`video_vae` in bf16"), "{message}");
        }
    }

    /// A selection of a module the family does not have, or of a representation its module does
    /// not take, is refused by its path.
    #[test]
    fn a_selection_outside_the_family_is_refused_by_path() {
        for (selection, path) in [
            (("vae", WeightFormat::Fp16), "`quantization.modules.vae`"),
            (
                ("video_vae", WeightFormat::Fp8),
                "`quantization.modules.video_vae = \"fp8\"`",
            ),
            (
                ("mlp", WeightFormat::Fp16),
                "`quantization.modules.mlp = \"fp16\"`",
            ),
        ] {
            let message = apply(&policy(None, &[selection]), &mut dense())
                .unwrap_err()
                .to_string();
            assert!(message.contains(path), "{message}");
        }
    }

    /// A packed export states its own representation: it takes no policy, not even `quality`,
    /// and serves its own without one.
    #[test]
    fn a_packed_export_takes_no_policy() {
        let mut files = dmd_export(release(), "VIDEO_SPARSE_ATTN_H3");
        files.insert(
            ModeloptManifest::FILE.to_owned(),
            serde_json::from_str(include_str!("modelopt_manifest.json")).unwrap(),
        );
        let mut config = read(&files, None).unwrap();

        assert!(apply(&Quantization::default(), &mut config).is_ok());
        let message = apply(&policy(Some(PrecisionMode::Quality), &[]), &mut config)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("checkpoint format 'modelopt_nvfp4' owns its numerical configuration"),
            "{message}"
        );
    }
}
