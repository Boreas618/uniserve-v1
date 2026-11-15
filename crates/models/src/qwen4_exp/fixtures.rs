//! A released configuration of the family, as its unit tests read it.

/// The Qwen3.8-Flash-Next configuration, eight layers deep (six linear, two attention).
pub(crate) fn flash_next() -> serde_json::Value {
    // The stack's own fields, then the fields of its other submodules.
    let mut text = serde_json::json!({
            "attention_bias": false,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "hc_count": 4,
            "hc_lowrank": 320,
            "head_dim": 256,
            "heads_per_ngram": 8,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_hidden_layers": 8,
            "num_key_value_heads": 2,
            "output_gate_type": "sigmoid",
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "vocab_size": 248320
    });
    let others = serde_json::json!({
            "make_ngram_vocab_size_divisible_by": 128,
            "moe_intermediate_size": 640,
            "mtp": {"layer_types": ["full_attention"], "num_hidden_layers": 1},
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
            "shared_expert_intermediate_size": 640,
            "split_ngram_parts": 128
    });
    if let (Some(text), Some(others)) = (text.as_object_mut(), others.as_object()) {
        text.extend(others.clone());
    }

    serde_json::json!({
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248056,
        "video_token_id": 248057,
        "tie_word_embeddings": false,
        "text_config": text,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    })
}
