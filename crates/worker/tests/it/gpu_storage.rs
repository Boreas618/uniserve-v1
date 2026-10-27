//! Storage owners against a real driver: block tables, output readback and
//! cache planes. These tests need a CUDA device and use a few megabytes of
//! memory; run them with
//! `cargo test -p worker --test it gpu_storage -- --ignored`.

use std::collections::HashMap;
use std::sync::Arc;

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream, memset_d8};
use ipc::BlockTable;
use ipc::{PrefixCacheDtype, UnitId};
use worker::ffi::{Backing, DType, Handle};
use worker::memory::LocalMemory;
use worker::memory::{
    BlockTableStorage, BlockTables, CacheGroupPlan, CachePlan, CachePlanes, GroupAxis, GroupShape,
    OutputBuffer, OutputStorage, Storage, StorageView,
};
use worker::{EventPool, Result};

use crate::device::exclusive_driver;

const DEVICE: DeviceId = DeviceId(0);

/// Copy a device view's bytes to the host (drains `stream`).
fn download(view: &StorageView, stream: &Stream) -> Result<Vec<u8>> {
    let bytes = view.byte_len()?;
    let host = Storage::pinned(DEVICE, bytes.max(1))?;
    let mut batch = CopyBatch::new(SourceOrder::Stream);
    batch.push(host.address(), view.address(), bytes);
    // SAFETY: both ranges hold `bytes`; the stream drains below.
    unsafe { batch.submit(stream) }?;
    stream.synchronize()?;
    // SAFETY: the copy into the pinned buffer completed.
    Ok(unsafe { host.host_bytes() }.unwrap_or_default()[..bytes].to_vec())
}

fn words(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_ne_bytes(*word))
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn block_table_installs_and_clears_write_device_tables() -> Result<()> {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE)?;
    let groups = vec![
        GroupShape::new(16, 1, None)?,
        GroupShape::new(32, 2, Some(64))?,
    ];
    let storage = BlockTableStorage::allocate(&groups, 3, 4, Some(DEVICE), Some(&stream))?;
    let mut tables = BlockTables::new(groups, 3, 4, 64)?;
    let views: HashMap<_, _> = storage.views().into_iter().collect();
    let table = |name: &str| -> Result<Vec<i32>> { Ok(words(&download(&views[name], &stream)?)) };
    assert_eq!(table("table_shapes")?, [0, 16, -1, 1, 32, 64, 1, 32, 64]);

    let mut install = |assignments: &[BlockTable]| -> Result<()> {
        let update = tables.prepare(assignments)?;
        storage.install(&update, Some(&stream))?;
        tables.commit(update);
        Ok(())
    };
    install(&[
        BlockTable {
            request_pool_idx: 3,
            group_id: 0,
            start_page: 0,
            unit_ids: [5, 6, 7, 8].map(UnitId).to_vec(),
            allocated_tokens: 64,
        },
        BlockTable {
            request_pool_idx: 3,
            group_id: 1,
            start_page: 1,
            unit_ids: [9, 10, 11, 12].map(UnitId).to_vec(),
            allocated_tokens: 96,
        },
    ])?;
    // A shorter row replaces the longer one completely.
    install(&[BlockTable {
        request_pool_idx: 3,
        group_id: 0,
        start_page: 0,
        unit_ids: [13].map(UnitId).to_vec(),
        allocated_tokens: 16,
    }])?;
    // A rejected update writes nothing.
    assert!(
        tables
            .prepare(&[BlockTable {
                request_pool_idx: 4,
                group_id: 0,
                start_page: 0,
                unit_ids: vec![UnitId(1)],
                allocated_tokens: 1,
            }])
            .is_err()
    );

    let units = table("unit_tables")?;
    let row =
        |table: usize, slot: usize| &units[(table * 4 + slot) * 4..(table * 4 + slot + 1) * 4];
    assert_eq!(row(0, 3), [13, 0, 0, 0]);
    assert_eq!(row(1, 3), [9, 11, 0, 0]);
    assert_eq!(row(2, 3), [10, 12, 0, 0]);
    assert_eq!(table("start_pages")?, [0, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(table("alloced_lens")?, [0, 0, 0, 16]);

    storage.set_verified(3, 12, Some(&stream))?;
    assert_eq!(table("verified_lengths")?, [0, 0, 0, 12]);
    storage.clear(&[3], Some(&stream))?;
    tables.release(&[3]);
    assert!(table("unit_tables")?.iter().all(|&unit| unit == 0));
    assert_eq!(table("verified_lengths")?, [0; 4]);
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn output_readback_waits_for_device_copies_into_lent_words() -> Result<()> {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE)?;
    let events = EventPool::default();
    let tokens = Storage::device(DEVICE, 3 * size_of::<i64>())?;
    // SAFETY: the allocation holds three words; the stream drains before
    // the readback reads them.
    unsafe { memset_d8(tokens.address(), 0xFF, 3 * size_of::<i64>(), &stream) }?;

    let mut output = OutputBuffer::new(
        OutputStorage::allocate(Some(DEVICE), 4)?,
        1,
        &[DEVICE],
        true,
    );
    // The producer copies its device words into the lent ones on its stream.
    let (offset, lent) = output.lend_words()?;
    let mut copy = CopyBatch::new(SourceOrder::Stream);
    copy.push(lent.address(), tokens.address(), 3 * size_of::<i64>());
    // SAFETY: both ranges hold three words; the lease keeps its pinned
    // words and the test its device words until the stream drained.
    unsafe { copy.submit(&stream) }?;
    assert!(output.readback().is_err(), "readback before sealing");
    output.seal(&[&stream], &events)?;
    output.register_lent(3, Vec::new())?;
    // The lease does not ask the driver: its fences have completed once
    // it is told so, here by the test that waited for the stream.
    assert!(output.readback().is_err(), "readback before its fences");
    stream.synchronize()?;
    output.fences_completed();
    output.readback()?;
    assert_eq!(output.read_tokens(offset, 3)?, [-1, -1, -1]);
    let timing = output.observe(0)?;
    assert!(timing.iter().all(|&micros| micros < 60_000_000));
    Ok(())
}

/// A one-group FP8 plan: two layers in two columns, 8-token pages of two
/// 4-wide heads (8-byte rows, 64-byte planes, a flag and an f32 scale per
/// plane).
fn fp8_plan() -> CachePlan {
    let plane_bytes = 8 * 2 * 4;
    CachePlan {
        columns: 2,
        plane_bytes,
        quantized: true,
        unit_bytes: 2 * 2 * (plane_bytes + 5),
        dtype: PrefixCacheDtype::Float8E4m3Fn,
        groups: vec![CacheGroupPlan {
            window: None,
            state: false,
            block_size: 8,
            units_per_page: 1,
            layers: vec!["a".into(), "b".into()],
            layer_ids: vec![0, 1],
            axis: GroupAxis {
                offset: 0,
                total: 2,
            },
            num_kv_heads: 2,
            total_kv_heads: 2,
            kv_head_offset: 0,
            head_dim: 4,
            dtype: DType::F8E4M3,
            compute_dtype: DType::BF16,
        }],
        states: Vec::new(),
        handle: Handle::none(),
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn cache_planes_start_reset_and_reset_their_leading_units() -> Result<()> {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE)?;
    let plan = Arc::new(fp8_plan());
    assert_eq!(plan.unit_bytes, 276);
    let planes =
        CachePlanes::allocate(plan, Some(DEVICE), 6, None, Some(&stream), &mut LocalMemory)?;
    let views: HashMap<_, _> = planes.views().into_iter().collect();
    assert_eq!(views["key.values"].shape(), &[2, 6, 64]);
    assert_eq!(views["key.scale"].shape(), &[2, 6]);
    assert!(
        download(&views["value.values"], &stream)?
            .iter()
            .all(|&byte| byte == 0)
    );
    assert!(
        floats(&download(&views["value.scale"], &stream)?)
            .iter()
            .all(|&scale| scale == 1.0)
    );

    // Dirty every field, then reset the leading units 1..=3 natively.
    for view in views.values() {
        let bytes = view.byte_len()?;
        // SAFETY: each view spans its field's allocation; the stream drains
        // before the next read.
        unsafe { memset_d8(view.address(), 7, bytes, &stream) }?;
    }
    planes.zero_leading(3, Some(&stream))?;
    let values = download(&views["key.values"], &stream)?;
    let unit = |column: usize, unit: usize| {
        &values[(column * 6 + unit) * 64..(column * 6 + unit + 1) * 64]
    };
    assert!(unit(1, 2).iter().all(|&byte| byte == 0));
    assert!(
        unit(0, 4).iter().all(|&byte| byte == 7),
        "units past the scratch keep their bytes"
    );
    let scales = floats(&download(&views["key.scale"], &stream)?);
    assert_eq!(scales[6 + 3], 1.0);
    let flags = download(&views["value.initialized"], &stream)?;
    assert_eq!(&flags[1..5], &[0, 0, 0, 7]);
    Ok(())
}
