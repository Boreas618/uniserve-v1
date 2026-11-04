//! The descriptor keys H3's attention compiles, one `operator<TAB>key<TAB>use` line each.
//!
//! `crates/models/minimax-h3` does not exist yet, so this is where the geometries its three
//! attention paths will ask for are written down and checked: a kernel program must package exactly
//! these variants, and the caller compares this output with the catalogue rows in
//! `notes/programs/attention_moe.md`. Run it with
//!
//! ```text
//! cargo run -p ops --example h3_attention_keys
//! ```
//!
//! The numbers are the deployment's, from specs/worker.md 9's H3 rows:
//!
//! - the video transformer's self- and cross-attention is dense BF16 attention over packed
//!   sequences, 24 query heads of 128 features against 8 KV heads, sequences up to 8192 tokens;
//! - its block-sparse path (VSA) reads 64-row query tiles against 64-row key blocks over a 65536-row
//!   key extent, and does not need the softmax normalizer: nothing downstream rescales it;
//! - the audio encoder's attention is FP32, 8 heads of 64 features over sequences up to 4096.
//!
//! The launch extents (`batch`, token counts, `selected`, `heads` as a runtime extent) are not part
//! of a descriptor: they are arguments, so one compiled variant serves every batch.

use ops::DType;
use ops::attention::Mask;
use ops::attention::dense::DenseDescriptor;
use ops::attention::sparse::SparseDescriptor;

fn main() {
    let dense = [
        // The video transformer's self- and cross-attention, BF16 with a dense mask. A
        // cross-attention's key offsets are not its query offsets, which is why the call carries
        // both tables rather than one.
        (
            "video self- and cross-attention",
            DenseDescriptor {
                query: DType::BF16,
                key_value: DType::BF16,
                output: DType::BF16,
                head_dim: 128,
                q_heads: 24,
                kv_heads: 8,
                mask: Mask::Dense,
                windowed: false,
                max_kv_len: 8192,
            },
        ),
        // The same path when a sequence attends a prefix of itself: the causal mask instead.
        (
            "packed causal sequences",
            DenseDescriptor {
                query: DType::BF16,
                key_value: DType::BF16,
                output: DType::BF16,
                head_dim: 128,
                q_heads: 24,
                kv_heads: 8,
                mask: Mask::Causal,
                windowed: false,
                max_kv_len: 8192,
            },
        ),
        // The audio encoder: FP32 in and out, every head its own key head.
        (
            "audio encoder",
            DenseDescriptor {
                query: DType::F32,
                key_value: DType::F32,
                output: DType::F32,
                head_dim: 64,
                q_heads: 8,
                kv_heads: 8,
                mask: Mask::Dense,
                windowed: false,
                max_kv_len: 4096,
            },
        ),
    ];
    for (name, descriptor) in dense {
        println!("dense_attention\t{descriptor}\t{name}");
    }

    let sparse = [
        // VSA with 64-row tiles and no normalizer output.
        SparseDescriptor {
            block: 64,
            dtype: DType::BF16,
            output: DType::BF16,
            head_dim: 128,
            lse: false,
            keys: 65536,
        },
        // VSA with 128-row tiles, the other specialisation the package builds.
        SparseDescriptor {
            block: 128,
            dtype: DType::BF16,
            output: DType::BF16,
            head_dim: 128,
            lse: false,
            keys: 65536,
        },
    ];
    for descriptor in sparse {
        println!("sparse_attention\t{descriptor}\tvideo sparse attention");
    }
}
