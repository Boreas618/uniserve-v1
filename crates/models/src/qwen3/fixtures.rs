//! Released configurations of the family, as its unit tests read them: each trimmed to the
//! fields the family reads.

/// A Qwen3-VL text encoder's configuration (the MiniMax-H3 text encoder's), trimmed to the
/// fields the family reads.
pub(crate) fn vl() -> serde_json::Value {
    serde_json::json!({
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
    })
}

/// A Qwen3-30B-A3B configuration, four layers deep.
pub(crate) fn moe() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3MoeForCausalLM"],
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "use_sliding_window": false
    })
}

/// A Qwen3.6-35B-A3B configuration, four layers deep (three linear, one attention).
pub(crate) fn hybrid() -> serde_json::Value {
    let mut root = vl();
    root["architectures"] = serde_json::json!(["Qwen3_5MoeForConditionalGeneration"]);
    root["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([]);
    root["vision_config"]["out_hidden_size"] = serde_json::json!(2048);
    root["text_config"] = serde_json::json!({
        "attn_output_gate": true,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                        "full_attention"],
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "moe_intermediate_size": 512,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default"
        },
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320
    });
    root
}
