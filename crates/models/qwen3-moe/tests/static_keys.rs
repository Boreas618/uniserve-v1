//! The walk with no weights and no device: a configuration and a partition state every program.
//!
//! The checkpoint this model serves is `Qwen3-30B-A3B`; this file reads its `config.json` when the
//! metadata is present and otherwise states the same architecture by hand, so the test runs on a
//! machine that holds no shards — which is exactly what the static walk exists for.

use nn::DType;
use qwen3_moe::keys::keys;
use qwen3_moe::{Config, Partition};

/// Qwen3-30B-A3B's architecture, as its `config.json` states it.
fn config() -> Config {
    let path = std::path::Path::new("/opt/uv/models/Qwen3-30B-A3B");
    if path.join("config.json").is_file() {
        return Config::read(path).unwrap_or_else(|error| panic!("the checkpoint's metadata: {error}"));
    }
    panic!("Qwen3-30B-A3B's config.json")
}

fn partition() -> Partition {
    Partition::new(&config(), 1, 0, 1, 0)
        .unwrap_or_else(|error| panic!("a single rank: {error}"))
}

/// The deployment's own values: the cache's page, the exchange's capacity, the buckets.
const PAGE: u32 = 64;

fn walk(dtype: DType, cache: DType, buckets: &[u32], experts: bool, projections: bool) -> Vec<(String, String)> {
    keys(&config(), &partition(), dtype, PAGE, cache, 0, buckets, experts, projections)
        .unwrap_or_else(|error| panic!("enumerating: {error}"))
}

#[test]
fn the_walk_states_the_model_with_no_weights_and_no_device() {
    let walked = walk(DType::BF16, DType::BF16, &[], false, false);
    let named: Vec<&str> = walked.iter().map(|(operator, _)| operator.as_str()).collect();
    for operator in [
        "embedding",
        "add_rms_norm",
        "rotary_factors",
        "qk_norm_rope",
        "gemm",
        "kv_write",
        "paged_attention",
        "topk_softmax",
        "expert_gather",
        "moe_sort",
        "gather_gemm",
        "route_gemm",
        "moe_unpermute",
        "rms_norm",
        "vocab_unpad",
    ] {
        assert!(named.contains(&operator), "{operator} is not in {named:?}");
    }
    // A dense deployment compiles no encoded program and no exchange.
    assert!(!named.iter().any(|operator| operator.starts_with("nvfp4_")), "{named:?}");
    assert!(!named.contains(&"expert_exchange"), "{named:?}");

    // The print, which is the store's input: one line per row, the operator then its key.
    for (operator, key) in &walked {
        println!("{operator} {key}");
    }
}

#[test]
fn an_e4m3_cache_is_written_by_the_scaled_write_and_no_other_plane_is() {
    let fp8 = walk(DType::BF16, DType::F8E4M3, &[], false, false);
    let named: Vec<&str> = fp8.iter().map(|(operator, _)| operator.as_str()).collect();
    assert!(named.contains(&"kv_write_scaled"), "{named:?}");
    assert!(!named.contains(&"kv_write"), "{named:?}");

    let bf16 = walk(DType::BF16, DType::BF16, &[], false, false);
    let named: Vec<&str> = bf16.iter().map(|(operator, _)| operator.as_str()).collect();
    assert!(named.contains(&"kv_write"), "{named:?}");
    assert!(!named.contains(&"kv_write_scaled"), "{named:?}");
}

#[test]
fn the_encoded_experts_add_the_encoder_and_their_two_products_and_no_bucket_rows() {
    let dense = walk(DType::BF16, DType::BF16, &[], false, false);
    let encoded = walk(DType::BF16, DType::BF16, &[8, 32], true, false);
    let named = |rows: &[(String, String)]| -> Vec<String> {
        rows.iter().map(|(operator, _)| operator.clone()).collect()
    };
    let before = named(&dense);
    let after = named(&encoded);
    assert!(!before.iter().any(|operator| operator == "nvfp4_quantize"));
    assert!(after.iter().any(|operator| operator == "nvfp4_quantize"));
    assert!(after.iter().any(|operator| operator == "gather_gemm_nvfp4"));
    assert!(after.iter().any(|operator| operator == "route_gemm_nvfp4"));
    assert!(!after.iter().any(|operator| operator == "gather_gemm"));
    // The expert products' grid is the route count, so the buckets do not multiply them.
    let counted = |rows: &[(String, String)], operator: &str| {
        rows.iter().filter(|(name, _)| name == operator).count()
    };
    let without = walk(DType::BF16, DType::BF16, &[], true, false);
    assert_eq!(counted(&encoded, "gather_gemm_nvfp4"), counted(&without, "gather_gemm_nvfp4"));
    assert_eq!(counted(&encoded, "nvfp4_quantize"), counted(&without, "nvfp4_quantize"));
    for (operator, key) in &encoded {
        if operator.starts_with("nvfp4_") || operator.ends_with("_nvfp4") {
            println!("{operator} {key}");
        }
    }
}

#[test]
fn a_packed_projection_costs_one_product_per_bucket_and_refuses_an_empty_list() {
    let one = walk(DType::BF16, DType::BF16, &[8], false, true);
    let three = walk(DType::BF16, DType::BF16, &[1, 8, 64], false, true);
    let products = |rows: &[(String, String)]| {
        rows.iter().filter(|(name, _)| name == "nvfp4_gemm").count()
    };
    assert_eq!(products(&one) * 3, products(&three), "one product per bucket per geometry");

    let refused = keys(&config(), &partition(), DType::BF16, PAGE, DType::BF16, 0, &[], false, true)
        .expect_err("a packed projection with no buckets");
    assert!(format!("{refused}").contains("bucket"), "{refused}");
}
