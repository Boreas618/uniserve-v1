//! Counter-based random draws of the rank's token sampler and of the engine
//! simulator.
//!
//! Philox4x32-10 maps request lineage and semantic draw coordinates to stable
//! uniforms. This addressing makes a draw independent of batching, execution
//! depth, completion order, and replay.
//!
//! The rank's control plane generates its sampler's draws through this module
//! (`crates/worker/src/sampling`) and passes them to the sampling kernels as
//! tensor contents; the engine simulator draws the same values for the same
//! coordinates.
//!
//! Layout:
//!
//! - key = a `splitmix` fold of `(session_seed, engine_id, request_id,
//!   epoch, draw_layout)`, split into two 32-bit words. A request's stream
//!   ([`request_sampling_key`]) folds its seed alone when it has one, so a
//!   seeded request draws the same values whenever it is sent, and its
//!   lineage when it has none;
//! - counter = `[semantic_token_index low, semantic_token_index high,
//!   processor_stage, draw_index]`;
//! - uniform = `(word0 >> 8) * 2^-24`, a 24-bit dyadic value in `[0, 1)`.

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_KEY_BUMP_0: u32 = 0x9E37_79B9;
const PHILOX_KEY_BUMP_1: u32 = 0xBB67_AE85;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Target-model sampling draw space.
///
/// Draw-space values match the IPC `DrawLayout` discriminants and isolate
/// semantic streams so activity in one stream cannot shift another.
pub const DRAW_LAYOUT_TARGET: u64 = 0;
/// Draw-space identifier for speculative proposal sampling.
pub const DRAW_LAYOUT_PROPOSAL: u64 = 1;
/// Draw-space identifier for diffusion flow noise.
pub const DRAW_LAYOUT_FLOW_NOISE: u64 = 2;

/// Folds one semantic coordinate into an evolving key with SplitMix64 avalanche.
fn splitmix_coordinate(seed: u64, coordinate: u64) -> u64 {
    let mut value = seed.wrapping_add(coordinate.wrapping_add(1).wrapping_mul(SPLITMIX_GAMMA));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Returns the high and low halves of one wrapping 32-bit multiplication.
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64).wrapping_mul(b as u64);
    ((product >> 32) as u32, product as u32)
}

/// Applies one Philox round to both counter lanes under the supplied key.
fn single_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(PHILOX_M0, counter[0]);
    let (hi1, lo1) = mulhilo(PHILOX_M1, counter[2]);
    [
        hi1 ^ counter[1] ^ key[0],
        lo1,
        hi0 ^ counter[3] ^ key[1],
        lo0,
    ]
}

/// Applies the ten-round Philox4x32 bijection to a counter and key.
pub fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        // Round zero consumes the caller's key; every later round advances both
        // lanes by the Philox Weyl constants before permuting the counter.
        if round > 0 {
            key = [
                key[0].wrapping_add(PHILOX_KEY_BUMP_0),
                key[1].wrapping_add(PHILOX_KEY_BUMP_1),
            ];
        }
        counter = single_round(counter, key);
    }
    counter
}

/// Derives the 64-bit Philox key for one request lineage and draw space.
pub fn sampling_key(
    session_seed: u64,
    engine_id: u64,
    request_id: u64,
    request_epoch: u64,
    draw_layout: u64,
) -> u64 {
    // Fold coordinates in protocol order so the same tuple addresses the same
    // random stream in every runtime.
    [engine_id, request_id, request_epoch, draw_layout]
        .into_iter()
        .fold(session_seed, splitmix_coordinate)
}

/// Derives the key of one request's stream in one draw space.
///
/// A seeded request's stream is a function of its seed alone: sending the
/// same request again, on any engine and under any request identity, draws
/// the same value at every semantic coordinate. An unseeded request's stream
/// is its lineage's, `(engine_id, request_id, request_epoch)`, which keeps
/// the streams of distinct requests apart.
pub fn request_sampling_key(
    seed: Option<u64>,
    engine_id: u64,
    request_id: u64,
    request_epoch: u64,
    draw_layout: u64,
) -> u64 {
    match seed {
        Some(seed) => sampling_key(seed, 0, 0, 0, draw_layout),
        None => sampling_key(0, engine_id, request_id, request_epoch, draw_layout),
    }
}

/// Maps one Philox word to an exactly representable 24-bit uniform value.
fn uniform_from_word(word: u32) -> f32 {
    ((word >> 8) as f32) * (1.0f32 / 16_777_216.0f32)
}

/// Returns one uniform draw in `[0, 1)` for a semantic coordinate and request key.
pub fn sampling_uniform(
    key: u64,
    semantic_token_index: u64,
    processor_stage: u64,
    draw_index: u64,
) -> f32 {
    // The counter layout gives each token, processor stage, and draw index an
    // independent address without relying on mutable generator state. The
    // token index keeps all 64 bits across two words; the stage and draw
    // index are truncated to their low 32 bits.
    let counter = [
        semantic_token_index as u32,
        (semantic_token_index >> 32) as u32,
        processor_stage as u32,
        draw_index as u32,
    ];
    let words = philox4x32_10(counter, [key as u32, (key >> 32) as u32]);
    uniform_from_word(words[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_matches_the_reference_known_answer_vectors() {
        // Random123 Philox4x32-10 known-answer vectors establish the exact
        // round count, lane ordering, and key schedule.
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([0xffff_ffff; 4], [0xffff_ffff; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
    }

    #[test]
    fn a_uniform_draw_stays_in_the_unit_interval() {
        for coordinate in 0..4096u64 {
            let key = sampling_key(0x1234_5678_9abc_def0, 7, 11, 3, DRAW_LAYOUT_TARGET);
            let draw = sampling_uniform(key, coordinate, 0, 0);
            assert!((0.0..1.0).contains(&draw), "draw {draw} out of range");
        }
    }

    #[test]
    fn a_seeded_request_stream_depends_on_its_seed_alone() {
        let seeded = request_sampling_key(Some(7), 1, 2, 3, DRAW_LAYOUT_TARGET);
        assert_eq!(
            seeded,
            request_sampling_key(Some(7), 4, 5, 6, DRAW_LAYOUT_TARGET)
        );
        assert_ne!(
            seeded,
            request_sampling_key(Some(8), 1, 2, 3, DRAW_LAYOUT_TARGET)
        );
        assert_ne!(
            seeded,
            request_sampling_key(Some(7), 1, 2, 3, DRAW_LAYOUT_PROPOSAL)
        );
        // Unseeded requests keep one stream per lineage.
        let unseeded = request_sampling_key(None, 1, 2, 3, DRAW_LAYOUT_TARGET);
        assert_eq!(
            unseeded,
            request_sampling_key(None, 1, 2, 3, DRAW_LAYOUT_TARGET)
        );
        assert_ne!(
            unseeded,
            request_sampling_key(None, 1, 5, 3, DRAW_LAYOUT_TARGET)
        );
        assert_ne!(
            unseeded,
            request_sampling_key(None, 1, 2, 4, DRAW_LAYOUT_TARGET)
        );
    }

    #[test]
    fn draw_layout_separates_the_proposal_and_target_spaces() {
        let session_seed = 0x0102_0304_0506_0708;
        let target = sampling_key(session_seed, 1, 2, 3, DRAW_LAYOUT_TARGET);
        let proposal = sampling_key(session_seed, 1, 2, 3, DRAW_LAYOUT_PROPOSAL);
        assert_ne!(target, proposal);
        assert_ne!(
            sampling_uniform(target, 5, 0, 0),
            sampling_uniform(proposal, 5, 0, 0)
        );
    }
}
