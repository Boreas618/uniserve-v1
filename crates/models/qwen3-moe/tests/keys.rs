//! The rows a deployment compiles, pinned where a store can be checked against them.

use nn::DType;
use nn::layers::Attention;

/// The slice's own attention geometry: 128 elements a head, 32 query heads over 4 key heads, 64-token
/// pages, a 40 960-key bound and no window.
fn attention() -> Attention {
    Attention::paged(128, 32, 4, 64, 40_960, None, 1.0)
        .unwrap_or_else(|error| panic!("the slice's attention geometry: {error}"))
}

#[test]
fn an_e4m3_plane_is_written_by_the_scaled_write_and_no_other_plane_is() {
    let (operator, key) = qwen3_moe::keys::write_row(&attention(), DType::BF16, DType::F8E4M3);
    assert_eq!(
        operator, "kv_write_scaled",
        "an E4M3 plane is written through the scaled write, which divides by the page's own scale"
    );
    // The plane's own type is the operator's name — this program writes E4M3 and no other writer
    // does — so the key carries the page's scale type and block instead.
    assert!(key.contains("block_128"), "the page's block is a key field: {key}");
    assert!(key.contains("scale_f32"), "the page scale's type is a key field: {key}");

    let (operator, key) = qwen3_moe::keys::write_row(&attention(), DType::BF16, DType::BF16);
    assert_eq!(operator, "kv_write", "every other plane is a plain scatter");
    assert!(key.contains("bf16"), "the plane's type is a key field here: {key}");
}
