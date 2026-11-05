//! The table the loading boundary owes for a calibrated checkpoint: which tensors are read in the
//! checkpoint's NVFP4 representation and which stay in its source dtype.
//!
//! The table is the only piece of NVFP4 evidence a machine without an NVFP4 checkpoint can produce,
//! and it is the checkable form of the trap: a loader that quantized the router or the vocabulary
//! head would still run, still print a coherent sentence, and measure nothing.

use std::path::{Path, PathBuf};

use qwen3_moe::{Adopted, Config, Representation, Role};

/// The architecture of the vertical slice, two layers deep: layer 0 routes, layer 1 does not.
fn config_json(declaration: &str) -> String {
    format!(
        r#"{{
            "architectures": ["Qwen3MoeForCausalLM"],
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 16,
            "norm_topk_prob": true,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [1],
            {declaration}
            "attention_bias": false
        }}"#
    )
}

fn directory(declaration: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "qwen3-moe-representation-table-{}-{serial}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("the fixture directory: {error}"));
    write_config(&dir, declaration);
    dir
}

fn write_config(dir: &Path, declaration: &str) {
    std::fs::write(dir.join("config.json"), config_json(declaration))
        .unwrap_or_else(|error| panic!("the fixture config: {error}"));
}

/// What a ModelOpt NVFP4 calibration of this architecture states: the linear projections are packed,
/// and every module whose role this model serves dense is excluded by name.
const CALIBRATED: &str = r#""quantization_config": {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "exclude_modules": [
            "lm_head",
            "model.embed_tokens",
            "model.norm",
            "model.layers.*.input_layernorm",
            "model.layers.*.post_attention_layernorm",
            "model.layers.*.self_attn.q_norm",
            "model.layers.*.self_attn.k_norm",
            "model.layers.*.mlp.gate"
        ]
    },"#;

#[test]
fn a_calibrated_checkpoint_is_packed_exactly_at_its_linear_projections() {
    let dir = directory(CALIBRATED);
    let config = Config::read(&dir).expect("the architecture");
    let representation = Representation::read(&dir).expect("the declaration");
    assert!(representation.is_encoded());

    let parameters = qwen3_moe::representation::parameters(&config);
    let mut packed = 0usize;
    let mut source = 0usize;
    for (parameter, role) in &parameters {
        let adoption = representation
            .adopts(parameter, *role)
            .unwrap_or_else(|error| panic!("{parameter} ({role:?}): {error}"));
        let expected = match role {
            Role::Projection => Adopted::Quantized,
            Role::Embedding | Role::Head | Role::Norm | Role::Router => Adopted::Source,
        };
        assert_eq!(adoption, expected, "{parameter} ({role:?})");
        match adoption {
            Adopted::Quantized => packed += 1,
            Adopted::Source => source += 1,
        }
    }

    // The table, as counts: layer 0 packs four attention projections and 2 experts × 3 = 10, layer 1
    // packs four attention projections and its dense three = 7, and nothing else is packed — 17. The
    // twelve left alone are the embedding, the head, the final norm, two norms and two Q/K norms per
    // layer (eight), and the router of the one routing layer.
    assert_eq!(packed, 17, "the projections the calibration packs");
    assert_eq!(source, 12, "the tensors it leaves in their source dtype");
    assert_eq!(source + packed, parameters.len());
}

#[test]
fn a_calibration_that_quantizes_the_router_is_refused_by_name() {
    let dir = directory(
        r#""quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "exclude_modules": ["lm_head"]
        },"#,
    );
    let config = Config::read(&dir).expect("the architecture");
    let representation = Representation::read(&dir).expect("the declaration");
    let refused = representation
        .adopts("model.layers.0.mlp.gate.weight", Role::Router)
        .expect_err("a packed router");
    assert!(
        format!("{refused}").contains("the router's scoring projection"),
        "{refused}"
    );

    // And the model's whole set is refused, not just the name a caller happened to ask about: the
    // walk stops at the router — a `Linear` in the checkpoint's module tree, which is the reason the
    // calibrated checkpoint names it in its exclusion list.
    let stopped = qwen3_moe::representation::parameters(&config)
        .into_iter()
        .find_map(|(parameter, role)| {
            representation
                .adopts(&parameter, role)
                .err()
                .map(|error| (parameter, error))
        })
        .expect("a parameter the calibration packed and the model serves dense");
    assert_eq!(stopped.0, "model.layers.0.mlp.gate.weight");
}

#[test]
fn an_unserved_algorithm_is_refused_before_the_architecture_is_even_read() {
    let dir =
        directory(r#""quantization_config": {"quant_method": "modelopt", "quant_algo": "MXFP4"},"#);
    let refused = Representation::read(&dir).expect_err("an unserved algorithm");
    assert!(format!("{refused}").contains("MXFP4"), "{refused}");
}

/// The declaration's one-line form names what it is and how many modules it keeps out, and does not
/// carry the modules themselves: a run's first line is not the place for forty-eight paths.
#[test]
fn the_declaration_states_itself_in_one_line() {
    let excluded: Vec<String> = (0..48)
        .map(|layer| format!("model.layers.{layer}.mlp.gate"))
        .collect();
    let declaration = Representation::Nvfp4 {
        excluded: excluded.clone(),
        also_excluded: Vec::new(),
        packed: None,
    };
    let line = declaration.summary();
    assert_eq!(line, "nvfp4, 48 module(s) excluded");
    assert!(!line.contains("model.layers"), "{line}");
    assert!(line.len() < 64, "{line}");

    let side = Representation::Nvfp4 {
        excluded,
        also_excluded: vec!["lm_head".to_owned()],
        packed: None,
    };
    assert_eq!(
        side.summary(),
        "nvfp4, 48 module(s) excluded and 1 on the side file"
    );
    assert_eq!(Representation::Dense.summary(), "dense");
}
