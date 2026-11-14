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
//! What a dense checkpoint is served in is `quality`: the checkpoint's BF16 projections and the
//! video VAE decoder in FP16, the type its reference computes in under autocast. No load of this
//! build converts a dense weight into FP8 or NVFP4, because both products read a calibrated
//! activation scale (`fp8_gemm`, `nvfp4_quantize`) that only a quantized export states, and the
//! video VAE decoder has no BF16 form. A policy that resolves a module to anything else is
//! refused with every such module named, before any weight is read. A packed ModelOpt export
//! states each module's representation in its manifest and takes no policy.

use nn::quantization::Declaration;
use proto::config::{PrecisionMode, Quantization, WeightFormat};

use crate::minimax_h3::{Config, Error, Result};

/// The modules a policy selects, in the order of every table below.
const MODULES: [&str; 4] = ["attention", "mlp", "text_encoder", "video_vae"];

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

/// The representation this build serves each of [`MODULES`] of a dense checkpoint in: the
/// checkpoint's own projections, and the video VAE decoder in its reference's FP16.
const SERVED: [WeightFormat; 4] = {
    use WeightFormat::{Bf16, Fp16};
    [Bf16, Bf16, Bf16, Fp16]
};

/// Checks `policy` against the checkpoint `config` describes: accepted when it states nothing,
/// or when every module resolves to the representation this build serves it in.
///
/// # Errors
///
/// [`Error::Unsupported`] naming the policy when the checkpoint declares its own quantization;
/// when a selection names a module the family does not have or a representation its module does
/// not take; or when the policy resolves a module to a representation this build does not serve
/// for a dense checkpoint, naming every such module.
pub(crate) fn check(policy: &Quantization, config: &Config) -> Result<()> {
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
        return Err(Error::Unsupported(format!(
            "`quantization = {policy}`: checkpoint format 'modelopt_nvfp4' owns its numerical \
             configuration (its `{}` states each module's representation); drop `quantization`",
            nn::quantization::ModeloptManifest::FILE
        )));
    }

    for (module, format) in &policy.modules {
        let Some(index) = MODULES.iter().position(|name| name == module) else {
            return Err(Error::Unsupported(format!(
                "`quantization.modules.{module}`: MiniMax-H3 selects the representation of {}, \
                 not of `{module}`",
                listed(MODULES.iter().map(|name| format!("`{name}`")))
            )));
        };
        if !SELECTABLE[index].contains(format) {
            return Err(Error::Unsupported(format!(
                "`quantization.modules.{module} = \"{format}\"`: `{module}` takes {}",
                listed(SELECTABLE[index].iter().map(|format| format!("`{format}`")))
            )));
        }
    }

    let given = tier(policy.mode.unwrap_or(PrecisionMode::Quality));
    let unserved: Vec<String> = MODULES
        .iter()
        .zip(given)
        .zip(SERVED)
        .filter_map(|((module, given), served)| {
            let resolved = policy.modules.get(*module).copied().unwrap_or(given);
            (resolved != served).then(|| format!("`{module}` in {resolved}"))
        })
        .collect();
    if unserved.is_empty() {
        return Ok(());
    }

    Err(Error::Unsupported(format!(
        "`quantization = {policy}` gives {}, and a dense checkpoint is served in quality \
         alone (`attention`, `mlp` and `text_encoder` in bf16, `video_vae` in fp16): no load \
         converts a dense weight into fp8 or nvfp4, whose products read a calibrated activation \
         scale only a quantized export states, and the video VAE decoder computes in fp16, the \
         type its reference computes in; serve `quality`, or a quantized export without \
         `quantization`",
        listed(unserved.into_iter())
    )))
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

    /// An empty policy, `quality`, and selections of what `quality` gives are served; each
    /// other mode is refused naming every module it narrows and the representation it gives.
    #[test]
    fn each_mode_resolves_to_its_tier_and_only_quality_is_served() {
        let config = dense();
        for accepted in [
            Quantization::default(),
            policy(Some(PrecisionMode::Quality), &[]),
            policy(None, &[("video_vae", WeightFormat::Fp16)]),
            policy(
                Some(PrecisionMode::Quality),
                &[
                    ("attention", WeightFormat::Bf16),
                    ("mlp", WeightFormat::Bf16),
                ],
            ),
        ] {
            assert!(check(&accepted, &config).is_ok(), "{accepted}");
        }

        // Each mode names exactly the modules it narrows, with the representation it gives
        // them, in the order of the table.
        for (mode, given) in [
            (PrecisionMode::Balanced, "`video_vae` in nvfp4"),
            (
                PrecisionMode::Performance,
                "`mlp` in fp8 and `video_vae` in nvfp4",
            ),
            (
                PrecisionMode::Maximum,
                "`mlp` in nvfp4, `text_encoder` in fp8 and `video_vae` in nvfp4",
            ),
        ] {
            let message = check(&policy(Some(mode), &[]), &config)
                .unwrap_err()
                .to_string();
            assert!(
                message.contains(&format!(" gives {given}, and ")),
                "{message}"
            );
        }
    }

    /// A selection replaces what its mode gives its module: `performance` with its two narrowed
    /// modules selected back is `quality`, and a selection narrows a module of `quality`.
    #[test]
    fn a_selection_replaces_its_modules_tier() {
        let config = dense();
        let restored = policy(
            Some(PrecisionMode::Performance),
            &[
                ("mlp", WeightFormat::Bf16),
                ("video_vae", WeightFormat::Fp16),
            ],
        );
        assert!(check(&restored, &config).is_ok());

        let narrowed = policy(None, &[("mlp", WeightFormat::Fp8)]);
        let message = check(&narrowed, &config).unwrap_err().to_string();
        assert!(message.contains(" gives `mlp` in fp8, and "), "{message}");

        let widened = policy(
            Some(PrecisionMode::Performance),
            &[("video_vae", WeightFormat::Bf16)],
        );
        let message = check(&widened, &config).unwrap_err().to_string();
        assert!(
            message.contains(" gives `mlp` in fp8 and `video_vae` in bf16, and "),
            "{message}"
        );
    }

    /// A selection of a module the family does not have, or of a representation its module does
    /// not take, is refused by its path.
    #[test]
    fn a_selection_outside_the_family_is_refused_by_path() {
        let config = dense();
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
            let message = check(&policy(None, &[selection]), &config)
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
            serde_json::from_str(include_str!(
                "../../../../nn/tests/fixtures/modelopt_manifest.json"
            ))
            .unwrap(),
        );
        let config = read(&files, None).unwrap();

        assert!(check(&Quantization::default(), &config).is_ok());
        let message = check(&policy(Some(PrecisionMode::Quality), &[]), &config)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("checkpoint format 'modelopt_nvfp4' owns its numerical configuration"),
            "{message}"
        );
    }
}
