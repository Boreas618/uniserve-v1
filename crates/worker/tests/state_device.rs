//! Recurrent state on a device: a keep and a restore move a slot's state through a state page
//! of the KV cache, and a settle takes a trail's entry as a slot's live state.
//!
//! Every case needs GPU 0; the case that launches the block-copy kernels also needs a running
//! Build Server (`UNISERVE_BUILD_SERVER_SOCKET`). Run them on a GPU host:
//!
//! ```text
//! cargo test -p worker --test state_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod support;

use std::sync::Arc;

use ipc::PrefixCacheDtype;
use ipc::batch::StateDirection;
use nn::family::{StateField, StateShape};
use ops::Generation;
use platform::{Budget, DeviceId, PoolKind, Stream};
use tensor::{DType, Tensor};
use worker::memory::cache::{CacheLayout, KvCacheConfig, KvCacheMemory};
use worker::memory::exclusive::LocalStorage;
use worker::memory::state::{StateConfig, StateMemory};

use support::{DEVICE, Kernels, Outcome};

/// The bytes of the state's second field, ten FP32 elements.
const FIELD_BYTES: usize = 40;

/// A budget that books whatever it is asked.
struct Unbounded;

impl Budget for Unbounded {
    fn book(&self, _: DeviceId, _: PoolKind, _: u64) -> platform::Result<()> {
        Ok(())
    }
}

/// A recurrent state of two layers: a 12-byte window and 40 bytes of state.
fn shape() -> StateShape {
    StateShape {
        layers: 2,
        fields: vec![
            StateField {
                name: "conv",
                dtype: DType::BF16,
                dims: vec![2, 3],
                stepwise: false,
            },
            StateField {
                name: "recurrent",
                dtype: DType::F32,
                dims: vec![10],
                stepwise: false,
            },
        ],
    }
}

/// The cache of two attention layers of 4-token pages of one 8-wide BF16 head beside that
/// state: a state page is two units.
fn cache() -> Outcome<KvCacheConfig> {
    let layout = CacheLayout::hybrid(4, 2, 1, 8, PrefixCacheDtype::BFloat16, &[shape()])?;

    Ok(KvCacheConfig {
        layout: Arc::new(layout),
        units: 16,
        slots: 3,
        width: 4,
        segments: None,
        workspace_bytes: 1 << 20,
    })
}

fn config(trail: u32) -> StateConfig {
    StateConfig {
        shapes: vec![shape()],
        slots: 3,
        rows: 2,
        trail,
        owned: 1,
        drafter: Vec::new(),
    }
}

/// The device's allocator with room for the state's buffers and its snapshot.
fn allocator() -> Outcome<platform::Allocator> {
    let mut allocator =
        platform::Allocator::new(&platform::Device::new(DEVICE)?, Arc::new(Unbounded))?;
    allocator.reserve(PoolKind::Resident, 64 << 20)?;
    allocator.reserve(PoolKind::Scratch, 64 << 20)?;
    Ok(allocator)
}

/// The device address of row `row` of the state's second field in `buffer`.
fn row_address(buffer: &Tensor, row: usize) -> Outcome<u64> {
    Ok(backend::operand(buffer)?.ptr.address() + (row * FIELD_BYTES) as u64)
}

/// Reads row `row` of the state's second field in `buffer` back to the host.
fn read_row(stream: &Stream, buffer: &Tensor, row: usize) -> Outcome<Vec<f32>> {
    let mut values = vec![0_f32; FIELD_BYTES / 4];

    // SAFETY: the source is one row of the field's buffer, which the state owns, and the
    // destination holds the row's ten elements; the stream is waited for before it is read.
    unsafe {
        platform::memcpy_async(
            values.as_mut_ptr() as u64,
            row_address(buffer, row)?,
            FIELD_BYTES,
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(values)
}

/// Writes `values` into row `row` of the state's second field in `buffer`.
fn write_row(stream: &Stream, buffer: &Tensor, row: usize, values: &[f32]) -> Outcome {
    // The pattern is placed on the device first, so the row's copy is between device ranges.
    let (_allocation, source) = support::weight(stream, values, &[values.len()])?;

    // SAFETY: the destination is one row of the field's buffer, which the state owns, and the
    // source the ten elements just placed; the stream is waited for before either goes.
    unsafe {
        platform::memcpy_async(
            row_address(buffer, row)?,
            backend::operand(&source)?.ptr.address(),
            FIELD_BYTES,
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_keep_and_a_restore_move_a_checkpoint_through_a_state_page() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let cache_config = cache()?;
    let cache = KvCacheMemory::open(cache_config.clone(), DEVICE, &LocalStorage, &stream)?;

    // The generation holds the kernels the state's own profile states.
    let keys = StateMemory::meta(config(0), &cache_config)?.keys()?;
    let kernels = Kernels::open()?.generation(1, &keys)?;

    let mut allocator = allocator()?;
    let mut state = StateMemory::open(
        config(0),
        &cache_config,
        &mut allocator,
        stream.clone(),
        kernels,
    )?;

    // Slot 1's checkpoint row holds a pattern, which a keep moves into the page of units 5
    // and 6 and a restore moves from there into slot 3's live row.
    let pattern: Vec<f32> = (0..10).map(|value| value as f32).collect();
    write_row(&stream, &state.checkpoint()?[0][1], 0, &pattern)?;

    state.copy(&cache, &[(1, 5..7)], StateDirection::Keep)?;
    state.copy(&cache, &[(3, 5..7)], StateDirection::Restore)?;

    assert_eq!(read_row(&stream, &state.live()?[0][1], 2)?, pattern);
    Ok(())
}

#[test]
#[ignore = "requires a GPU"]
fn a_settle_takes_the_trails_entry_as_the_slots_live_state() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let cache_config = cache()?;

    // A settle launches no kernel, so the generation holds none.
    let mut allocator = allocator()?;
    let state = StateMemory::open(
        config(4),
        &cache_config,
        &mut allocator,
        stream.clone(),
        Arc::new(Generation::new(1, [])),
    )?;

    let pattern: Vec<f32> = (0..10).map(|value| 0.5 * value as f32).collect();
    let trail = state.trail()?.expect("a trail of four tokens");
    write_row(&stream, &trail[1][1], 3, &pattern)?;

    state.settle(2, 3)?;
    assert_eq!(read_row(&stream, &state.live()?[1][1], 1)?, pattern);
    Ok(())
}
