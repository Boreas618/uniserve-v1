//! Recurrent state on a device: a keep and a restore move a slot's state through a unit of the
//! state region, a window's history alone, and a settle takes a trail's entry as a slot's live
//! state; a switch to the state of other drafts moves each slot's standing state into it, and a
//! switch back returns it.
//!
//! Every case needs GPU 0; the cases that launch the block-copy kernels also need this host's
//! Build Server (`uniserve build-server serve`). Run them on a GPU host:
//!
//! ```text
//! cargo test -p engine --test state_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod device;

use std::sync::Arc;

use engine::batch::StateDirection;
use engine::memory::cache::{CacheLayout, KvCacheConfig, KvCacheMemory};
use engine::memory::exclusive::LocalStorage;
use engine::memory::state::{Standing, StateConfig, StateMemory};
use nn::family::{StateField, StateRows, StateShape};
use ops::Generation;
use platform::{Budget, DeviceId, PoolKind, Stream};
use proto::request::PrefixCacheDtype;
use tensor::{DType, Tensor};

use device::{DEVICE, Kernels, Outcome};

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
                rows: nn::family::StateRows::Whole,
            },
            StateField {
                name: "recurrent",
                dtype: DType::F32,
                dims: vec![10],
                stepwise: false,
                rows: nn::family::StateRows::Whole,
            },
        ],
    }
}

/// The keys and values of two attention layers of 4-token pages of one 8-wide BF16 head.
fn keys_and_values() -> Outcome<CacheLayout> {
    Ok(CacheLayout::of(
        4,
        &nn::family::CacheField::keys_and_values(2, 1, 8),
        PrefixCacheDtype::BFloat16,
    )?)
}

/// The cache of two attention layers of 4-token pages of one 8-wide BF16 head beside that
/// state, with a state region of eight checkpoints of four 64-byte planes each.
fn cache() -> Outcome<KvCacheConfig> {
    let layout = keys_and_values()?.with_state(&[shape()])?;

    Ok(KvCacheConfig {
        layout: Arc::new(layout),
        units: 16,
        slots: 3,
        width: 4,
        segments: None,
        workspace_bytes: 1 << 20,
        states: (8, None),
        windows: Vec::new(),
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
        platform::Allocator::new(&platform::Device::new(DEVICE)?, &[], Arc::new(Unbounded))?;
    allocator.reserve(PoolKind::Resident, 64 << 20)?;
    allocator.reserve(PoolKind::Scratch, 64 << 20)?;
    Ok(allocator)
}

/// The device address of row `row` of the state's second field in `buffer`.
fn row_address(buffer: &Tensor, row: usize) -> Outcome<u64> {
    Ok(ops::operand(buffer)?.ptr.address() + (row * FIELD_BYTES) as u64)
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
    let (_allocation, source) = device::weight(stream, values, &[values.len()])?;

    // SAFETY: the destination is one row of the field's buffer, which the state owns, and the
    // source the ten elements just placed; the stream is waited for before either goes.
    unsafe {
        platform::memcpy_async(
            row_address(buffer, row)?,
            ops::operand(&source)?.ptr.address(),
            FIELD_BYTES,
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_keep_and_a_restore_move_a_checkpoint_through_a_state_unit() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let cache_config = cache()?;
    let cache = KvCacheMemory::open(cache_config.clone(), (DEVICE, &[]), &LocalStorage, &stream)?;

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

    // Slot 1's checkpoint row holds a pattern, which a keep moves into state unit 5 and a
    // restore moves from there into slot 3's live row.
    let pattern: Vec<f32> = (0..10).map(|value| value as f32).collect();
    write_row(&stream, &state.checkpoint()?[0][1], 0, &pattern)?;

    state.copy(&cache, &[(1, 5)], StateDirection::Keep)?;
    state.copy(&cache, &[(3, 5)], StateDirection::Restore)?;

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

/// A recurrent state of one layer for a deployment that drafts `drafts` tokens a step: a window
/// of two inputs of three FP32 channels with a row per draft, a state of four FP32 elements
/// that keeps a slot per verified token when the deployment drafts, and a ring of groups of two
/// positions of two FP32 channels.
fn drafting(drafts: usize) -> StateShape {
    StateShape {
        layers: 1,
        fields: vec![
            StateField {
                name: "conv",
                dtype: DType::F32,
                dims: vec![2 + drafts, 3],
                stepwise: false,
                rows: StateRows::Window { history: 2 },
            },
            StateField {
                name: "recurrent",
                dtype: DType::F32,
                dims: vec![4],
                stepwise: drafts > 0,
                rows: StateRows::Whole,
            },
            StateField {
                name: "ring",
                dtype: DType::F32,
                dims: vec![(2 + drafts).next_multiple_of(2), 2],
                stepwise: false,
                rows: StateRows::Ring { group: 2 },
            },
        ],
    }
}

/// The cache of two attention layers of 4-token pages of one 8-wide BF16 head beside the
/// drafting state: a checkpoint keeps the window's two inputs and the state, one 64-byte plane
/// each, whatever the drafts, and nothing of the ring.
fn drafting_cache() -> Outcome<KvCacheConfig> {
    let layout = keys_and_values()?.with_state(&[drafting(0)])?;

    Ok(KvCacheConfig {
        layout: Arc::new(layout),
        units: 16,
        slots: 3,
        width: 4,
        segments: None,
        workspace_bytes: 1 << 20,
        states: (8, None),
        windows: Vec::new(),
    })
}

/// The state of three slots of a deployment that drafts `drafts` tokens a step.
fn drafting_config(drafts: usize) -> StateConfig {
    StateConfig {
        shapes: vec![drafting(drafts)],
        slots: 3,
        rows: 2,
        trail: 0,
        owned: drafts as u32 + 1,
        drafter: Vec::new(),
    }
}

/// Reads `count` FP32 values `offset` values into `buffer` back to the host.
fn read(stream: &Stream, buffer: &Tensor, offset: usize, count: usize) -> Outcome<Vec<f32>> {
    let mut values = vec![0_f32; count];

    // SAFETY: the source lies inside the buffer, which the state owns, and the destination
    // holds `count` values; the stream is waited for before it is read.
    unsafe {
        platform::memcpy_async(
            values.as_mut_ptr() as u64,
            ops::operand(buffer)?.ptr.address() + (offset * 4) as u64,
            count * 4,
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(values)
}

/// Writes `values` as FP32 `offset` values into `buffer`.
fn write(stream: &Stream, buffer: &Tensor, offset: usize, values: &[f32]) -> Outcome {
    let (_allocation, source) = device::weight(stream, values, &[values.len()])?;

    // SAFETY: the destination lies inside the buffer, which the state owns, and the source the
    // values just placed; the stream is waited for before either goes.
    unsafe {
        platform::memcpy_async(
            ops::operand(buffer)?.ptr.address() + (offset * 4) as u64,
            ops::operand(&source)?.ptr.address(),
            values.len() * 4,
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

/// The values `from..from + count`.
fn counting(from: usize, count: usize) -> Vec<f32> {
    (from..from + count).map(|value| value as f32).collect()
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_keep_and_a_restore_move_a_windows_history_and_leave_its_draft_rows() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let cache_config = drafting_cache()?;
    let cache = KvCacheMemory::open(cache_config.clone(), (DEVICE, &[]), &LocalStorage, &stream)?;

    let keys = StateMemory::meta(drafting_config(2), &cache_config)?.keys()?;
    let kernels = Kernels::open()?.generation(1, &keys)?;
    let mut allocator = allocator()?;
    let mut state = StateMemory::open(
        drafting_config(2),
        &cache_config,
        &mut allocator,
        stream.clone(),
        kernels,
    )?;

    // Slot 1's checkpoint window holds its two inputs and two draft rows; a keep moves the
    // inputs into state unit 5, and a restore moves them into slot 3's window, whose draft
    // rows stay as they were.
    write(&stream, &state.checkpoint()?[0][0], 0, &counting(1, 12))?;
    write(&stream, &state.live()?[0][0], 2 * 12, &[-1.0; 12])?;
    state.copy(&cache, &[(1, 5)], StateDirection::Keep)?;
    state.copy(&cache, &[(3, 5)], StateDirection::Restore)?;

    let window = read(&stream, &state.live()?[0][0], 2 * 12, 12)?;
    assert_eq!(window[..6], counting(1, 6)[..]);
    assert_eq!(window[6..], [-1.0; 6]);
    Ok(())
}

#[test]
#[ignore = "requires a GPU"]
fn a_switch_moves_each_slots_standing_state_into_the_other_drafts_and_back() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let cache_config = drafting_cache()?;
    let mut allocator = allocator()?;
    // The moves are copies, which launch no kernel.
    let mut state = StateMemory::open(
        drafting_config(2),
        &cache_config,
        &mut allocator,
        stream.clone(),
        Arc::new(Generation::new(1, [])),
    )?;

    // Under two drafts a request owns three slots of the state, rows 3 + 2(s − 1) .. beside its
    // home row, and a window of four rows; the ring holds four positions.
    let live = state.live()?;
    let (conv, recurrent, ring) = (&live[0][0], &live[0][1], &live[0][2]);
    // Slot 1 accepted all three tokens of its verification at a verified length of five, slot 2
    // decoded one at two, and slot 3 accepted two and holds no token.
    let standing = [
        Standing {
            accepted: 3,
            verified: 5,
        },
        Standing {
            accepted: 1,
            verified: 2,
        },
        Standing {
            accepted: 2,
            verified: 0,
        },
    ];
    write(&stream, recurrent, 4 * 4, &counting(10, 4))?;
    write(&stream, recurrent, 4, &counting(20, 4))?;
    write(&stream, recurrent, 7 * 4, &counting(30, 4))?;
    for slot in 0..3 {
        write(&stream, conv, slot * 12, &counting(100 * (slot + 1), 12))?;
        write(&stream, ring, slot * 8, &counting(200 * (slot + 1), 8))?;
    }

    // The state of no drafts: one slot a request, a window of its two inputs, a ring of two
    // positions.
    state.prepare(7, drafting_config(0), &mut allocator)?;
    assert!(state.switch(7, &standing)?);
    assert_eq!(state.config(), &drafting_config(0));
    let live = state.live()?;
    let (conv, recurrent, ring) = (&live[0][0], &live[0][1], &live[0][2]);

    // Each slot's state is the one its accepted count names, in its home row.
    assert_eq!(read(&stream, recurrent, 0, 4)?, counting(10, 4));
    assert_eq!(read(&stream, recurrent, 4, 4)?, counting(20, 4));
    assert_eq!(read(&stream, recurrent, 8, 4)?, counting(30, 4));
    // A window's history starts at row `accepted − 1` of the slot's window.
    assert_eq!(read(&stream, conv, 0, 6)?, counting(106, 6));
    assert_eq!(read(&stream, conv, 6, 6)?, counting(200, 6));
    assert_eq!(read(&stream, conv, 12, 6)?, counting(303, 6));
    // Positions 3 and 4 of slot 1 lay at rows 3 and 0 of four, and lie at rows 1 and 0 of two;
    // slot 2's positions 0 and 1 at rows 0 and 1.
    assert_eq!(read(&stream, ring, 0, 4)?, [200.0, 201.0, 206.0, 207.0]);
    assert_eq!(read(&stream, ring, 4, 4)?, counting(400, 4));

    // The generation's state serves, and is given back only once the switch is rolled back.
    assert!(state.release(7, &mut allocator).is_err());
    let one = standing.map(|slot| Standing {
        accepted: 1,
        ..slot
    });
    assert!(state.switch_back(7, &one)?);
    assert_eq!(state.config(), &drafting_config(2));
    let live = state.live()?;
    let (conv, recurrent, ring) = (&live[0][0], &live[0][1], &live[0][2]);
    assert_eq!(read(&stream, recurrent, 0, 4)?, counting(10, 4));
    assert_eq!(read(&stream, conv, 0, 6)?, counting(106, 6));
    // Slot 1's positions 3 and 4 return to rows 3 and 0 of four.
    assert_eq!(read(&stream, ring, 0, 2)?, [200.0, 201.0]);
    assert_eq!(read(&stream, ring, 6, 2)?, [206.0, 207.0]);
    state.release(7, &mut allocator)?;

    // A committed generation's state serves on alone, and the one it replaced is given back.
    state.prepare(8, drafting_config(1), &mut allocator)?;
    assert!(state.switch(8, &one)?);
    state.commit(8, &mut allocator)?;
    assert_eq!(state.config(), &drafting_config(1));
    assert_eq!(read(&stream, &state.live()?[0][1], 0, 4)?, counting(10, 4));
    Ok(())
}
