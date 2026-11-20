//! `CausalLmRunner` over a model of DeepSeek-V4.1-Flash's cache fields (`testkit::latent`), on
//! the meta path: what the rank's cache lays out for them, what a forward is lent, and the
//! slots a prefill, a replay of a prefix hit and a decode write in each group.
//!
//! The cache keeps the full history's group (the compressed latent and the index keys of four
//! owners) over the KV pool, the sliding window's group (the window latent of 43 layers) over a
//! pool of its own, and the compressor's remainder in the state region.

use std::sync::Arc;

use crate::batch::philox::DrawAddress;
use crate::batch::{
    Batch, BatchId, Columns, DraftWork, Phase, Row, Rows, Shape as BatchShape, Variant,
};
use nn::NO_SLOT;
use nn::family::{INDEX_KEY, TextModel};
use ops::Generation;
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use proto::module::sampler::{PARAMETER_COLUMNS, TOP_P};
use proto::request::{PrefixCacheDtype, RequestId};
use storage::Key;
use storage::runtime::{Completion, GroupId, OwnerId, Request, Run};
use testkit::latent::{COMPRESSED, Formats, LATENT, LatentModel, PAGE, WINDOW};

use super::columns::Column;
use super::{CausalLmConfig, CausalLmRunner};
use crate::memory::cache::{CacheLayout, KvCacheConfig, KvCacheMemory, window_units};
use crate::memory::state::{StateConfig, StateMemory};
use crate::memory::tables::BlockTables;
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::graphs::Graphs;
use crate::runner::{Forward, Runner};

/// The regions the tests number the cache's memory by: the KV pool's units, the window's and
/// the state units.
const UNITS: u8 = 0;
const WINDOWS: u8 = 1;
const STATES: u8 = 2;

/// The runner's request slots, the pages a slot's table row names and the KV pool's units.
const SLOTS: u32 = 4;
const WIDTH: u32 = 8;
const KV_UNITS: u32 = 32;
/// The tokens of the runner's largest prefill bucket.
const TOKENS: u32 = 256;

/// The drafts of a verification of the block-drafting runner ([`drafting`]): a block of five,
/// DeepSeek-V4.1's DSpark drafter's.
const BLOCK: u32 = 5;

/// The stub, a runner over a meta cache of its fields with every bucket captured, and the
/// cache.
fn runner() -> (Arc<LatentModel>, CausalLmRunner, KvCacheMemory) {
    over(
        Arc::new(LatentModel::new(0, Formats::REFERENCE).unwrap()),
        0,
    )
}

/// The stub with its drafter of the block form, a runner that verifies its blocks of
/// [`BLOCK`] over a meta cache of its fields with every bucket captured, and the cache.
fn drafting() -> (Arc<LatentModel>, CausalLmRunner, KvCacheMemory) {
    over(
        Arc::new(LatentModel::drafting(BLOCK, Formats::REFERENCE).unwrap()),
        BLOCK,
    )
}

/// `model`, a runner of `drafts` drafts a verification over a meta cache of its fields with
/// every bucket captured, and the cache.
fn over(model: Arc<LatentModel>, drafts: u32) -> (Arc<LatentModel>, CausalLmRunner, KvCacheMemory) {
    let window = WINDOW + model.block_drafter().map_or(0, |drafter| drafter.block());
    let layout = CacheLayout::of(PAGE, &model.cache(), PrefixCacheDtype::BFloat16)
        .and_then(|layout| layout.with_state(&model.state_layers()))
        .unwrap();
    let cache = KvCacheMemory::meta(KvCacheConfig {
        layout: Arc::new(layout),
        units: KV_UNITS,
        slots: SLOTS,
        width: WIDTH,
        segments: None,
        workspace_bytes: 1 << 16,
        states: (8, None),
        windows: vec![window_units(window, PAGE, SLOTS, TOKENS)],
    })
    .unwrap();

    let buckets = Buckets::causal_lm(&[2, 4], &[TOKENS], true);
    let config = CausalLmConfig {
        phases: CausalLmRunner::served_phases(&buckets.with_drafts(drafts), model.as_ref()),
        buckets,
        slots: SLOTS,
        page_size: PAGE,
        max_kv_len: PAGE * WIDTH,
        candidates: 1,
        scratch_pages: 0,
        drafts,
        draft_pool: None,
    };
    let state = StateConfig::of(model.as_ref(), SLOTS, &config.buckets, drafts).unwrap();
    let state = StateMemory::meta(state, cache.config()).unwrap();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config, model.as_ref()).unwrap(),
    );

    let model_dyn: Arc<dyn TextModel> = Arc::clone(&model) as Arc<dyn TextModel>;
    let runner =
        CausalLmRunner::over(model_dyn, config, graphs, None, &cache, Some(&state)).unwrap();
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }
    (model, runner, cache)
}

/// The device block table of a rank of the regions the tests number.
fn tables() -> BlockTables {
    BlockTables::new(GroupId(0), &[UNITS, WINDOWS, STATES], &[])
}

/// Attaches the owner of `slot` to `tables`.
fn attach(tables: &mut BlockTables, slot: u32) {
    let request = Request::Attach {
        owner: OwnerId(u64::from(slot)),
        group: GroupId(0),
    };
    tables
        .map(
            slot,
            storage::runtime::RequestId(0),
            &request,
            &Completion::Attached { slot },
        )
        .unwrap();
}

/// Hands `tables` an `Allocate` of one block per unit of `units` of `region`, at `at` of
/// `class`, for the owner of `slot`.
fn allocate(
    tables: &mut BlockTables,
    slot: u32,
    (class, region): (u8, u8),
    at: u32,
    units: &[u32],
) {
    let request = Request::Allocate {
        owner: OwnerId(u64::from(slot)),
        class,
        at,
        count: units.len() as u32,
        units: 0,
    };
    let runs: Vec<Run> = units.iter().map(|unit| (region, *unit, 1)).collect();
    let completion = Completion::Allocated {
        runs,
        evicted: vec![],
    };
    tables
        .map(slot, storage::runtime::RequestId(1), &request, &completion)
        .unwrap();
}

/// Commits the full-history block at `at` of the owner of `slot`: the prefix hit a replay reads.
fn commit_entry(tables: &mut BlockTables, slot: u32, at: u32) {
    tables
        .apply(&Request::Commit {
            owner: OwnerId(u64::from(slot)),
            class: 0,
            at,
            key: Key([at as u8 + 1; 32]),
            tokens: PAGE as u16,
            after: vec![],
        })
        .unwrap();
}

/// A prefill of positions `start..end` on `slot` that replays `replay`, one call of one row.
fn prefill(slot: u32, (start, end): (u32, u32), replay: std::ops::Range<u32>) -> Batch {
    let tokens = (end - start) as usize;
    let mut parameters = [0.0; PARAMETER_COLUMNS as usize];
    parameters[TOP_P as usize] = 1.0;
    let mut commit = vec![0; COMMIT_FIELDS];
    commit[column::SLOT] = i64::from(slot);
    commit[column::MODE] = mode::CACHE;
    commit[column::TASK] = -1;

    let columns = Columns {
        rows: Rows {
            calls: vec![Row {
                slot,
                kind: Phase::Prefill,
                replay,
                rows: 0..1,
            }],
            positions: (start as i32..end as i32).collect(),
            cache_positions: (start as i32..end as i32).collect(),
            seq_lens: vec![end as i32],
            query_offsets: vec![0, tokens as i32],
            logits_rows: vec![-1],
        },
        input_ids: vec![5; tokens],
        slots: vec![slot as i32],
        parameters: parameters.to_vec(),
        draws: vec![DrawAddress::default()],
        commit,
        checkpoints: vec![0],
        page_size: PAGE,
        max_kv_len: PAGE * WIDTH,
        ..Columns::default()
    };
    Batch {
        id: BatchId(3),
        group: GroupId(0),
        phase: Phase::Prefill,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: BatchShape::prefill(1, tokens as u32, Variant::PrefillWrite),
        requests: vec![RequestId(9)],
        columns,
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// A decode of one row per `(slot, position, replay)`, each the token at its position.
fn decode(rows: &[(u32, u32, std::ops::Range<u32>)]) -> Batch {
    let count = rows.len();
    let mut parameters = [0.0; PARAMETER_COLUMNS as usize];
    parameters[TOP_P as usize] = 1.0;
    let mut commit = vec![0; COMMIT_FIELDS * count];
    for (row, (slot, ..)) in rows.iter().enumerate() {
        commit[column::SLOT * count + row] = i64::from(*slot);
        commit[column::MODE * count + row] = mode::DECODE;
        commit[column::TASK * count + row] = row as i64;
    }

    let columns = Columns {
        rows: Rows {
            calls: rows
                .iter()
                .enumerate()
                .map(|(row, (slot, _, replay))| Row {
                    slot: *slot,
                    kind: Phase::Decode,
                    replay: replay.clone(),
                    rows: row as u32..row as u32 + 1,
                })
                .collect(),
            positions: rows.iter().map(|(_, at, _)| *at as i32).collect(),
            cache_positions: rows.iter().map(|(_, at, _)| *at as i32).collect(),
            seq_lens: rows.iter().map(|(_, at, _)| *at as i32 + 1).collect(),
            query_offsets: (0..=count as i32).collect(),
            logits_rows: (0..count as i32).collect(),
        },
        input_ids: vec![0; count],
        slots: rows.iter().map(|(slot, ..)| *slot as i32).collect(),
        parameters: parameters.repeat(count),
        draws: vec![DrawAddress::default(); count],
        commit,
        checkpoints: vec![0; count],
        page_size: PAGE,
        max_kv_len: PAGE * WIDTH,
        ..Columns::default()
    };
    Batch {
        id: BatchId(4),
        group: GroupId(0),
        phase: Phase::Decode,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: BatchShape::decode(count as u32, Variant::DecodeGreedy),
        requests: (0..count as u64).map(RequestId).collect(),
        columns,
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// The slot each of the `tokens` tokens of the step the runner prepared last writes in group
/// `group`, and each of its `rows` rows' window start.
fn prepared(runner: &CausalLmRunner, bucket: &Bucket, group: usize) -> (Vec<u32>, Vec<u32>) {
    let (tokens, rows) = (bucket.tokens as usize, bucket.rows as usize);
    let layout = &runner.layout;
    let read = |column: Column, elements: usize| -> Vec<u32> {
        let start = layout.offset(column) as usize;
        bytemuck::pod_collect_to_vec(&runner.staged[start..start + elements * 4])
    };
    let slots = read(Column::SlotMapping, layout.groups() * tokens);
    (
        slots[group * tokens..][..tokens].to_vec(),
        read(Column::Starts, rows),
    )
}

/// A forward is lent the full history's group and the window's, each a plane per owner of its
/// fields at the owner's entries per page, readers resolved to their owners' planes, and the
/// window's rows' starts.
#[test]
fn a_forward_is_lent_a_group_per_history_with_a_plane_per_owner() {
    let (model, _runner, cache) = runner();
    let forwards = model.forwards();
    let [full, window] = forwards
        .last()
        .expect("the capture ran forwards")
        .as_slice()
    else {
        panic!("two groups a forward");
    };

    // The compressed latents and the index keys of layers 2, 8, 14 and 20, NVFP4 and MXFP4,
    // one entry per two tokens but layer 20's per token, over the KV pool's units.
    let planes: Vec<(&str, u32, Vec<usize>)> = full
        .planes
        .iter()
        .map(|(field, layer, _, dims)| (*field, *layer, dims.clone()))
        .collect();
    let units = KV_UNITS as usize;
    assert_eq!(
        planes,
        [
            (COMPRESSED, 2, vec![units, 32, 1, 288]),
            (COMPRESSED, 8, vec![units, 32, 1, 288]),
            (COMPRESSED, 14, vec![units, 32, 1, 288]),
            (COMPRESSED, 20, vec![units, 64, 1, 288]),
            (INDEX_KEY, 2, vec![units, 32, 1, 68]),
            (INDEX_KEY, 8, vec![units, 32, 1, 68]),
            (INDEX_KEY, 14, vec![units, 32, 1, 68]),
            (INDEX_KEY, 20, vec![units, 64, 1, 68]),
        ]
    );
    // Layers 3 to 7 read layer 2's compressed latent, and 24, 28, 32 and 36 layer 20's keys.
    assert_eq!(full.planes[0].2, [3, 4, 5, 6, 7]);
    assert_eq!(full.planes[7].2, [24, 28, 32, 36]);
    assert!(full.starts.is_none());

    // The window latent of 43 layers over the window's own pool, read from each row's start.
    let windows = cache.config().windows[0] as usize;
    assert_eq!(window.planes.len(), 43);
    assert!(
        window
            .planes
            .iter()
            .enumerate()
            .all(|(layer, (field, owner, _, dims))| *field == LATENT
                && *owner == layer as u32
                && *dims == [windows, 64, 1, 528])
    );
    let rows = window.block_tables[0];
    assert_eq!(window.starts.as_deref(), Some(&[rows][..]));
    assert_eq!(full.slot_mapping, window.slot_mapping);
}

/// A prefill writes every group at its token's page of the group's table: the full history's
/// pages are units of the KV pool, the window's of its own pool.
#[test]
fn a_prefill_writes_each_group_through_its_own_table() {
    let (_, mut runner, _) = runner();
    let mut tables = tables();
    attach(&mut tables, 1);
    allocate(&mut tables, 1, (0, UNITS), 0, &[7, 3]);
    allocate(&mut tables, 1, (1, WINDOWS), 0, &[5, 2]);

    let batch = prefill(1, (0, 100), 0..0);
    let bucket = runner.prepare_inputs(&batch, &tables).unwrap();
    let (full, starts) = prepared(&runner, &bucket, 0);
    let (window, _) = prepared(&runner, &bucket, 1);

    let slot = |unit: u32, position: u32| unit * PAGE + position % PAGE;
    for position in 0..100 {
        let (unit, held) = if position < PAGE { (7, 5) } else { (3, 2) };
        assert_eq!(full[position as usize], slot(unit, position), "{position}");
        assert_eq!(
            window[position as usize],
            slot(held, position),
            "{position}"
        );
    }
    assert!(full[100..].iter().all(|slot| *slot == NO_SLOT));
    assert_eq!(starts[0], 0);
}

/// A replay of a prefix hit recomputes the window before the hit: its tokens before the hit's
/// end write the window alone and none of the hit's read-only full-history pages, the tokens
/// past it write both, and the window's readers read from where the replay starts.
#[test]
fn a_replay_writes_the_window_alone_before_the_end_of_the_hit() {
    let (_, mut runner, _) = runner();
    let mut tables = tables();
    attach(&mut tables, 2);

    // A hit of three pages, 192 tokens, read-only; the replay rewinds the 128-token window to
    // position 64, whose window pages are new, and computes on to 256.
    allocate(&mut tables, 2, (0, UNITS), 0, &[11, 12, 13, 14]);
    for at in 0..3 {
        commit_entry(&mut tables, 2, at);
    }
    allocate(&mut tables, 2, (1, WINDOWS), 1, &[4, 6, 8]);

    let batch = prefill(2, (64, 256), 64..192);
    let bucket = runner.prepare_inputs(&batch, &tables).unwrap();
    let (full, starts) = prepared(&runner, &bucket, 0);
    let (window, _) = prepared(&runner, &bucket, 1);

    for (token, position) in (64..256).enumerate() {
        let full_slot = (position >= 192).then(|| 14 * PAGE + position % PAGE);
        assert_eq!(full[token], full_slot.unwrap_or(NO_SLOT), "{position}");
        let unit = [4, 6, 8][(position / PAGE - 1) as usize];
        assert_eq!(window[token], unit * PAGE + position % PAGE, "{position}");
    }
    assert_eq!(starts[0], 64);

    // Without its replay, the same call writes the hit's read-only pages and is refused.
    let refused = runner
        .prepare_inputs(&prefill(2, (64, 256), 0..0), &tables)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("read-only"), "{refused}");
}

/// A window's page the scheduler retired behind the window holds no entry: a call that would
/// write there is refused rather than writing into a page another request holds.
#[test]
fn a_call_that_writes_a_retired_window_page_is_refused() {
    let (_, mut runner, _) = runner();
    let mut tables = tables();
    attach(&mut tables, 3);
    allocate(&mut tables, 3, (0, UNITS), 0, &[9, 10]);
    allocate(&mut tables, 3, (1, WINDOWS), 1, &[3]);

    let refused = runner
        .prepare_inputs(&prefill(3, (0, 100), 0..0), &tables)
        .unwrap_err()
        .to_string();
    assert!(
        refused.contains("class 1") && refused.contains("holds no entry"),
        "{refused}"
    );
}

/// A decode writes each row's token in both groups, and each row's window readers read from
/// where the row's replay started: a row that replayed a hit from 64, and one that computed its
/// whole prompt.
#[test]
fn a_decode_writes_both_groups_and_reads_each_window_from_its_start() {
    let (_, mut runner, _) = runner();
    let mut tables = tables();
    for slot in [1, 2] {
        attach(&mut tables, slot);
    }
    allocate(&mut tables, 1, (0, UNITS), 0, &[7, 3]);
    allocate(&mut tables, 1, (1, WINDOWS), 0, &[5, 2]);
    allocate(&mut tables, 2, (0, UNITS), 0, &[11, 12, 13, 14, 15]);
    for at in 0..3 {
        commit_entry(&mut tables, 2, at);
    }
    allocate(&mut tables, 2, (1, WINDOWS), 2, &[6, 8, 9]);

    let batch = decode(&[(1, 100, 0..0), (2, 260, 64..192)]);
    let bucket = runner.prepare_inputs(&batch, &tables).unwrap();
    let (full, starts) = prepared(&runner, &bucket, 0);
    let (window, _) = prepared(&runner, &bucket, 1);

    assert_eq!(full[..2], [3 * PAGE + 100 % PAGE, 15 * PAGE + 260 % PAGE]);
    assert_eq!(window[..2], [2 * PAGE + 100 % PAGE, 9 * PAGE + 260 % PAGE]);
    assert_eq!(starts[..2], [0, 64]);
}

/// The bytes the runner staged last of `column`, its first `elements` elements.
fn staged<T: bytemuck::Pod>(runner: &CausalLmRunner, column: Column, elements: usize) -> Vec<T> {
    let start = runner.layout.offset(column) as usize;
    let bytes = elements * std::mem::size_of::<T>();
    bytemuck::pod_collect_to_vec(&runner.staged[start..start + bytes])
}

/// A verification on `slot` of the committed token at `position` and [`BLOCK`] drafts after it,
/// one call of one row that proposes a block after the drafts it accepts.
fn verification(slot: u32, position: u32) -> Batch {
    let tokens = BLOCK as usize + 1;
    let mut parameters = [0.0; PARAMETER_COLUMNS as usize];
    parameters[TOP_P as usize] = 1.0;
    let mut commit = vec![0; COMMIT_FIELDS];
    commit[column::SLOT] = i64::from(slot);
    commit[column::MODE] = mode::VERIFY;
    commit[column::TASK] = 0;
    commit[column::LOGICAL] = i64::from(position);
    commit[column::SAMPLING] = 1;
    commit[column::VISIBLE] = i64::from(position);

    let end = position as i32 + tokens as i32;
    let columns = Columns {
        rows: Rows {
            calls: vec![Row {
                slot,
                kind: Phase::Verify,
                replay: 0..0,
                rows: 0..1,
            }],
            positions: (position as i32..end).collect(),
            cache_positions: (position as i32..end).collect(),
            seq_lens: vec![end],
            query_offsets: vec![0, tokens as i32],
            logits_rows: vec![0],
        },
        input_ids: vec![3; tokens],
        slots: vec![slot as i32],
        parameters: parameters.to_vec(),
        draws: vec![DrawAddress {
            key: 1,
            position: u64::from(position) + tokens as u64,
        }],
        commit,
        checkpoints: vec![0],
        page_size: PAGE,
        max_kv_len: PAGE * WIDTH,
        drafts: vec![DraftWork {
            proposals: BLOCK,
            ..DraftWork::default()
        }],
        ..Columns::default()
    };
    Batch {
        id: BatchId(5),
        group: GroupId(0),
        phase: Phase::Verify,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: BatchShape::prefill(1, tokens as u32, Variant::DecodeGreedy),
        requests: vec![RequestId(9)],
        columns,
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// A runner whose model drafts in the block form verifies over the model's own cache: it serves
/// verifications and no decode, captures its drafter's bucket behind each verification and
/// prefill size, and each drafter forward is lent the context of the bucket's tokens and a
/// block of [`BLOCK`] tokens per row over the model's two groups, the window's with its rows'
/// starts, every row's tables and slots the forward's own.
#[test]
fn a_block_drafter_drafts_over_the_model_s_own_cache() {
    let (model, runner, _) = drafting();
    let buckets = runner.buckets();
    assert!(!buckets.iter().any(|bucket| bucket.variant.is_decode()));
    for rows in [2, 4] {
        let verify = Bucket::verify(rows, BLOCK, Variant::VerifyGreedy);
        assert!(buckets.contains(&verify), "{verify}");
        assert!(buckets.contains(&Bucket::draft(rows, verify.tokens)));
    }

    // The capture of the drafter's bucket behind a verification of four rows of six tokens.
    let drafts = model.drafts();
    let drafted = drafts
        .iter()
        .find(|draft| draft.context.1 == 24)
        .expect("the drafter's bucket of 24 tokens was captured");
    assert_eq!(drafted.block.1, 4 * BLOCK as usize);
    for (groups, tokens) in [&drafted.context, &drafted.block] {
        let [full, window] = groups.as_slice() else {
            panic!("two groups a forward, not {groups:?}");
        };
        assert_eq!(full.slot_mapping, [*tokens]);
        assert_eq!(window.slot_mapping, [*tokens]);
        assert_eq!(full.block_tables[0], 4);
        assert_eq!(window.block_tables[0], 4);
        assert_eq!(window.starts.as_deref(), Some(&[4][..]));
        assert!(full.starts.is_none());
        // The window latent of the 40 target layers and of the three draft layers.
        assert_eq!(window.planes.len(), 43);
    }
}

/// A verification's block lies after the drafts it accepts, and its window slots are staged for
/// every length it may agree on: the cells of the positions after its first token, `2 D` of
/// them, each the slot of its page of the window's table, and none in the full history, which
/// the drafter does not keep. The block's positions and its anchor are stated before the
/// accepted drafts, which the device adds.
#[test]
fn a_verification_stages_its_block_s_window_slots_for_every_accepted_length() {
    let (_, mut runner, _) = drafting();
    let mut tables = tables();
    attach(&mut tables, 1);
    allocate(&mut tables, 1, (0, UNITS), 0, &[7, 3]);
    allocate(&mut tables, 1, (1, WINDOWS), 0, &[5, 2]);

    // Six tokens from 120: the block after them may lie from 121 up to 130, across the window's
    // pages 1 (unit 2) and 2, which the slot does not hold.
    let bucket = runner
        .prepare_inputs(&verification(1, 120), &tables)
        .unwrap();
    let rows = bucket.rows as usize;
    let cells = 2 * BLOCK as usize;
    let slots: Vec<u32> = staged(&runner, Column::BlockCells, 2 * rows * cells);
    let (full, window) = slots.split_at(rows * cells);

    assert!(full.iter().all(|slot| *slot == NO_SLOT), "{full:?}");
    let expected: Vec<u32> = (121..121 + cells as u32)
        .map(|position| match position / PAGE {
            1 => 2 * PAGE + position % PAGE,
            _ => NO_SLOT,
        })
        .collect();
    assert_eq!(window[..cells], expected[..]);
    assert!(window[cells..].iter().all(|slot| *slot == NO_SLOT));

    let positions: Vec<i32> = staged(&runner, Column::BlockPositions, BLOCK as usize);
    assert_eq!(positions, [121, 122, 123, 124, 125]);
    let anchors: Vec<u32> = staged(&runner, Column::BlockAnchors, 1);
    assert_eq!(anchors, [0]);
}
