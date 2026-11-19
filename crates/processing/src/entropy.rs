//! The randomness of a processing module: none it can draw on.
//!
//! A program reaches nothing but its host functions, and the interface gives it no entropy. The
//! JSON Schema validator a tag tool call's values are typed with seeds its hash maps through
//! `getrandom`, which a `wasm32-unknown-unknown` build reaches through its `custom` backend (the
//! workspace's `.cargo/config.toml` selects it for this target). The maps hold one request's own
//! schemas, so their seed guards against nothing a request does not already control: the
//! backend fills every request with the same deterministic bytes, a SplitMix64 sequence, and no
//! import of the JavaScript runtime the default wasm backend would need is linked.

use std::sync::atomic::{AtomicU64, Ordering};

/// The state of the sequence, advanced by every byte drawn.
static STATE: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

/// The `getrandom` 0.3 custom backend: fills `len` bytes at `dest` from the sequence.
///
/// # Safety
///
/// `dest` points to `len` writable bytes, as `getrandom` guarantees its backend.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom::Error> {
    // SAFETY: the caller's obligation.
    let bytes = unsafe { std::slice::from_raw_parts_mut(dest, len) };
    for chunk in bytes.chunks_mut(8) {
        let mut word = STATE
            .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        word ^= word >> 31;
        chunk.copy_from_slice(&word.to_le_bytes()[..chunk.len()]);
    }
    Ok(())
}
