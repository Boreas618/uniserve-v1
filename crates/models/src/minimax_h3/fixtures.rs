//! The released checkpoint's sidecars, as the family's unit tests read them.
//!
//! They are the values `docs/minimax_h3/minimax_h3.md` and the released checkpoints state: a
//! 64-layer Qwen3-VL text encoder of which H3 keeps 50, a 50-block network conditioned on
//! 5120-wide text, a 17-frame video VAE of 24 latent channels, and a 32-channel audio VAE at
//! 32 kHz.

use std::collections::BTreeMap;

use serde_json::Value;

use nn::{PipelineIndex, Result};

use crate::minimax_h3::ID;
use crate::minimax_h3::config::*;

/// A fixture's JSON.
pub(crate) fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap()
}

/// The text encoder checkpoint's configuration.
pub(crate) fn text_encoder() -> Value {
    json(
        r#"{
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "image_token_id": 151655,
            "video_token_id": 151656,
            "tie_word_embeddings": false,
            "text_config": {
                "attention_bias": false,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "intermediate_size": 25600,
                "max_position_embeddings": 262144,
                "num_attention_heads": 64,
                "num_hidden_layers": 64,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default"
                },
                "rope_theta": 5000000,
                "vocab_size": 151936
            },
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 5120,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        }"#,
    )
}

/// The processor's pixel normalization.
pub(crate) fn processor() -> Value {
    json(r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}"#)
}

/// The video VAE's configuration.
pub(crate) fn video_vae() -> Value {
    let statistics = |value: f64| Value::Array((0..24).map(|_| Value::from(value)).collect());

    serde_json::json!({
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 24,
        "block_out_channels": [128, 256, 256, 512, 512, 1024],
        "layers_per_block": 2,
        "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
        "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
        "norm_num_groups": 32,
        "norm_eps": 1e-06,
        "spatial_padding_mode": "reflect",
        "decoder_num_layers": 36,
        "decoder_num_attention_heads": 32,
        "decoder_attention_head_dim": 64,
        "decoder_num_register_tokens": 4,
        "decoder_ffn_mult": 4,
        "decoder_rope_theta": 100.0,
        "decoder_rope_dim_ratio": 0.75,
        "decoder_norm_eps": 1e-05,
        "clip_length": 17,
        "token_drop": 3,
        "latents_mean": statistics(0.0),
        "latents_std": statistics(1.0),
    })
}

/// The audio VAE's configuration.
pub(crate) fn audio_vae() -> Value {
    serde_json::json!({
        "encoder_dim": 64,
        "encoder_rates": [2, 4, 4, 5, 5],
        "latent_dim": 2048,
        "latent_channels": 32,
        "decoder_dim": 1024,
        "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
        "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
        "num_attention_heads": 8,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "sampling_rate": 32000,
        "latents_mean": vec![0.0_f64; 32],
        "latents_std": vec![1.0_f64; 32],
    })
}

/// One network's configuration.
pub(crate) fn transformer() -> Value {
    json(
        r#"{
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "hidden_size": 5376,
            "num_layers": 50,
            "num_refiner_layers": 2,
            "ffn_dim": 14336,
            "in_channels": 24,
            "audio_in_channels": 32,
            "patch_size": [1, 2, 2],
            "text_dim": 5120,
            "freq_dim": 256,
            "time_embed_hidden_dim": 5376,
            "time_embed_dim": 2688,
            "rope_freq_dim": 16,
            "rope_theta": 10000.0,
            "norm_eps": 1e-05,
            "qk_norm_eps": 1e-05,
            "final_norm_eps": 1e-05
        }"#,
    )
}

/// The base release's sidecars, by checkpoint-relative path: a pipeline index over both
/// networks.
pub(crate) fn release() -> BTreeMap<String, Value> {
    let index = r#"{"_class_name": "MiniMaxH3ModularPipeline",
                    "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
                    "transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel"]}"#;

    [
        ("modular_model_index.json", json(index)),
        ("text_encoder/config.json", text_encoder()),
        ("processor/preprocessor_config.json", processor()),
        ("vae/config.json", video_vae()),
        ("audio_vae/config.json", audio_vae()),
        (
            "scheduler/scheduler_config.json",
            json(r#"{"shift": 12.0}"#),
        ),
        (
            "audio_scheduler/scheduler_config.json",
            json(r#"{"shift": 3.0}"#),
        ),
        ("transformer/config.json", transformer()),
        ("transformer_ref/config.json", transformer()),
    ]
    .into_iter()
    .map(|(path, value)| (path.to_owned(), value))
    .collect()
}

/// A FastH3 DMD export of the base network over `files`: its contract, its scheduler's
/// shift and no reference network.
pub(crate) fn dmd_export(
    mut files: BTreeMap<String, Value>,
    backend: &str,
) -> BTreeMap<String, Value> {
    files.remove("transformer_ref/config.json");
    files.insert(
        "scheduler/scheduler_config.json".to_owned(),
        json(r#"{"shift": 10.0}"#),
    );
    files.insert(Contract::FILE.to_owned(), dmd_contract(backend));
    files
}

/// The contract of an eight-rung DMD student of the base network, attending with `backend`.
pub(crate) fn dmd_contract(backend: &str) -> Value {
    serde_json::json!({
        "schema_version": Contract::SCHEMA,
        "task": "t2av",
        "video_scheduler_shift": 10.0,
        "audio_scheduler_shift": 3.0,
        "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
        "attention_backend": backend,
        "vsa_tile_size": 64,
        "vsa_sparsity": 0.8,
    })
}

/// The configuration the family composes of `files`, a checkpoint's sidecars: the student
/// `adapter` contracts when an adapter is fused, else the student the checkpoint's own
/// contract states, else the networks its pipeline index declares.
pub(crate) fn read(files: &BTreeMap<String, Value>, adapter: Option<&Contract>) -> Result<Config> {
    let networks = match (adapter, files.get(Contract::FILE)) {
        (Some(contract), _) => Networks::Student(contract.clone()),
        (None, Some(contract)) => Networks::Student(Contract::parse(contract)?),
        (None, None) => {
            let index = files.get("modular_model_index.json").ok_or_else(|| {
                ID.checkpoint("the fixture states no modular_model_index.json".to_owned())
            })?;
            let index =
                PipelineIndex::parse(std::path::Path::new("modular_model_index.json"), index)?;
            Networks::Released(released_networks(&index)?)
        }
    };

    Config::read(
        |relative| Ok(files.get(relative).cloned()),
        &networks,
        "fixture",
    )
}

/// The released base configuration.
pub(crate) fn released_config() -> Config {
    read(&release(), None).unwrap()
}
