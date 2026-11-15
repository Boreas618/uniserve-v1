//! A packed ModelOpt export's calibration, as the quantization each stage declares.
//!
//! A packed NVFP4 export of the family states its calibration in `modelopt_manifest.json`
//! ([`ModeloptManifest`]): every calibrated projection, listed by its path in the exporter's
//! module tree, with the activation magnitude its calibration measured. The paths are that
//! tree's, not the checkpoint's; each one names a projection a stage directory stores under its
//! own name:
//!
//! ```text
//! listed path                                              stage            stored module
//! denoiser.transformer.layers.<i>.mlp.gate_up.projections.{gate, up}
//!                                                          transformer      transformer_blocks.<i>.ff.net.0.proj
//! denoiser.transformer.layers.<i>.mlp.down                 transformer      transformer_blocks.<i>.ff.net.2
//! reference_denoiser.transformer.layers.<i>.mlp.…          transformer_ref  as the denoiser's
//! video_decoder.decoder.decoder.decoder.layers.<i>.qkv.projections.{q, k, v}
//!                                                          vae              decoder.transformer_blocks.<i>.attn.to_{q, k, v}
//! video_decoder.decoder.decoder.decoder.layers.<i>.output  vae              decoder.transformer_blocks.<i>.attn.to_out.0
//! video_decoder.decoder.decoder.decoder.layers.<i>.mlp.gate_up.projections.{gate, up}
//!                                                          vae              decoder.transformer_blocks.<i>.ff.net.0.proj
//! video_decoder.decoder.decoder.decoder.layers.<i>.mlp.down
//!                                                          vae              decoder.transformer_blocks.<i>.ff.net.2
//! ```
//!
//! The exporter lists the gate and the up half of a feed-forward apart, and the checkpoint
//! stores them as one projection, so the two paths name one stored module and state one
//! activation scale. A stage the manifest lists nothing for (the text encoder and the audio
//! VAE of the released exports) is dense.

use nn::quantization::{Declaration, ModeloptManifest};

use super::Component;
use crate::minimax_h3::{Error, Result, transformer, video_vae};

/// The stored projection of a gated feed-forward the exporter lists as `listed`, its path
/// below the feed-forward: `gate_up` for either fused half, `down` for the projection back.
fn feed_forward<'n>(listed: &[&str], [gate_up, down]: [&'n str; 2]) -> Option<&'n str> {
    match listed {
        ["gate_up", "projections", "gate" | "up"] => Some(gate_up),
        ["down"] => Some(down),
        _ => None,
    }
}

/// The stage directory and the stored module of the path `listed` in a packed export's
/// manifest, or `None` for a path that names no projection this family stores packed.
fn stored_module(listed: &str) -> Option<(&'static str, String)> {
    let path: Vec<&str> = listed.split('.').collect();

    match path.as_slice() {
        [
            root @ ("denoiser" | "reference_denoiser"),
            "transformer",
            "layers",
            block,
            "mlp",
            projection @ ..,
        ] => {
            let network = if *root == "denoiser" {
                Component::Transformer
            } else {
                Component::TransformerRef
            };
            let block: u32 = block.parse().ok()?;
            let projection = feed_forward(projection, transformer::FEED_FORWARD)?;

            Some((
                network.name(),
                format!("{}.{block}.{projection}", transformer::BLOCKS),
            ))
        }

        [
            "video_decoder",
            "decoder",
            "decoder",
            "decoder",
            "layers",
            block,
            projection @ ..,
        ] => {
            let block: u32 = block.parse().ok()?;
            let [query, key, value] = video_vae::QKV;

            let (module, projection) = match projection {
                ["qkv", "projections", "q"] => (video_vae::ATTENTION, query),
                ["qkv", "projections", "k"] => (video_vae::ATTENTION, key),
                ["qkv", "projections", "v"] => (video_vae::ATTENTION, value),
                ["output"] => (video_vae::ATTENTION, video_vae::ATTENTION_OUTPUT),
                ["mlp", projection @ ..] => (
                    video_vae::FEED_FORWARD,
                    feed_forward(projection, video_vae::FEED_FORWARD_PROJECTIONS)?,
                ),
                _ => return None,
            };

            Some((
                Component::Vae.name(),
                format!("{}.{block}.{module}.{projection}", video_vae::BLOCKS),
            ))
        }

        _ => None,
    }
}

/// Checks that every path `manifest` lists names a projection this family stores packed.
///
/// # Errors
///
/// A listed path of another module tree, named as the manifest lists it: a calibration this
/// family cannot place would otherwise be dropped, and its projection read without one.
pub(super) fn check(manifest: &ModeloptManifest) -> Result<()> {
    match manifest
        .modules()
        .find(|listed| stored_module(listed).is_none())
    {
        None => Ok(()),
        Some(listed) => Err(Error::config(format!(
            "{} calibrates `{listed}`, which names no projection a MiniMax-H3 stage stores \
             packed: the feed-forward projections of a network's blocks and the attention and \
             feed-forward projections of the video decoder's blocks",
            ModeloptManifest::FILE
        ))),
    }
}

/// The quantization the stage stored in `directory` declares: what `manifest` lists for it,
/// and dense without a manifest.
///
/// # Errors
///
/// Two listed paths of one stored projection that state different activation scales.
pub(super) fn declaration(
    manifest: Option<&ModeloptManifest>,
    directory: &str,
) -> Result<Declaration> {
    let Some(manifest) = manifest else {
        return Ok(Declaration::dense());
    };

    Ok(manifest.declaration(|listed| {
        stored_module(listed)
            .filter(|(stage, _)| *stage == directory)
            .map(|(_, module)| module)
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listed_path_names_the_projection_its_stage_stores() {
        let named = |stage: &'static str, module: &str| Some((stage, module.to_owned()));

        // Both halves of a fused feed-forward projection are one stored module.
        for half in ["gate", "up"] {
            assert_eq!(
                stored_module(&format!(
                    "denoiser.transformer.layers.7.mlp.gate_up.projections.{half}"
                )),
                named("transformer", "transformer_blocks.7.ff.net.0.proj")
            );
        }
        assert_eq!(
            stored_module("denoiser.transformer.layers.49.mlp.down"),
            named("transformer", "transformer_blocks.49.ff.net.2")
        );
        assert_eq!(
            stored_module("reference_denoiser.transformer.layers.0.mlp.down"),
            named("transformer_ref", "transformer_blocks.0.ff.net.2")
        );

        let decoder = "video_decoder.decoder.decoder.decoder.layers.35";
        for (listed, module) in [
            ("qkv.projections.q", "attn.to_q"),
            ("qkv.projections.k", "attn.to_k"),
            ("qkv.projections.v", "attn.to_v"),
            ("output", "attn.to_out.0"),
            ("mlp.gate_up.projections.gate", "ff.net.0.proj"),
            ("mlp.gate_up.projections.up", "ff.net.0.proj"),
            ("mlp.down", "ff.net.2"),
        ] {
            assert_eq!(
                stored_module(&format!("{decoder}.{listed}")),
                named("vae", &format!("decoder.transformer_blocks.35.{module}"))
            );
        }

        // A path of another tree, a block that is no number, a projection the family does not
        // pack.
        for listed in [
            "text_encoder.network.layers.0.mlp.down",
            "denoiser.transformer.layers.first.mlp.down",
            "denoiser.transformer.layers.0.attention.output",
            "denoiser.transformer.layers.0.mlp.gate_up",
            "video_decoder.decoder.decoder.decoder.layers.0.qkv",
        ] {
            assert_eq!(stored_module(listed), None, "{listed}");
        }
    }
}
