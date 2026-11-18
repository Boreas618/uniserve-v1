//! `DecoderRunner` without a device: where a decode round's copies read and write in the rank's
//! product buffer, the tail one round leaves for the next, the deal of rounds to the ranks in
//! order and in reverse alternately, a round dealt to the ranks of one process with the tails
//! they exchange over the decoder ranks' communicator, the refusals of a row, and the profile run
//! and capture of a clip's, a splice's and a track's buckets, over decoders whose forwards launch
//! only the runner's own conversions.

use std::sync::Arc;

use crate::batch::{Batch, BatchId, Columns, Extent, MediaRow, Phase, Shape, Variant};
use nn::vae::TemporalClips;
use nn::{Clip, Decoder, LatentShape, PixelNorm, Splice, Splicer};
use ops::Generation;
use platform::PitchedBatch;
use proto::module::operator::{PCM_INTERLEAVE, RGB_BYTES};
use proto::request::RequestId;
use proto::video::Layout;
use storage::Key;
use storage::runtime::{self, Completion, GroupId, OwnerId, Request, Run};
use tensor::{DType, Tensor};

use super::{DecoderConfig, DecoderRunner, RoundConfig, Unit};
use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::tables::BlockTables;
use crate::runner::Runner;
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::graphs::Graphs;

const GROUP: GroupId = GroupId(0);

/// The slot rows, the rank's product buffer and the other ranks' of its process, and the slot
/// the request runs on.
const SLOTS: u8 = 0;
const PRODUCTS: u8 = 2;
const OTHER: u8 = 3;
const THIRD: u8 = 7;
const SLOT: u32 = 1;

/// The classes of the latents, of the decoded media and of the splice tails.
const CLASSES: [u8; 3] = [4, 5, 6];

/// A timeline of three units of four frames and a tail of two: 14 frames, from four latent
/// frames, each clip's window of two starting at its own index.
const FRAMES: u32 = 14;
const TRAJECTORY: u32 = 4;

/// The bytes of one packed frame of an 8 by 8 raster.
const FRAME: u64 = 8 * 8 * 3;

/// A unit of the product buffer, as an address of a meta memory.
const UNIT: u64 = UNIT_BYTES;

/// A decoder of four-channel latents, four decoded frames and pixels per latent frame and cell;
/// its clip `k` reads the latent frames from `k`.
struct Clips;

impl Decoder for Clips {
    fn element_type(&self) -> DType {
        DType::F16
    }

    fn latent_type(&self) -> DType {
        DType::F32
    }

    fn gather_latents(
        &self,
        _layout: &Layout,
        _trajectory: &Tensor,
        _first: u32,
        _block: &Tensor,
    ) -> nn::Result<()> {
        Ok(())
    }

    fn clip_start(&self, clip: u32) -> u32 {
        clip
    }

    fn layout_of(&self, clip: Clip) -> nn::Result<Layout> {
        Ok(layout(
            LatentShape {
                channels: 4,
                frames: clip.frames / 4,
                height: clip.height / 4,
                width: clip.width / 4,
            },
            None,
        ))
    }

    fn pcm_window(&self, audio: &LatentShape) -> nn::Result<(u64, u32)> {
        Ok((u64::from(audio.frames) * 4, 2))
    }

    fn decode(
        &self,
        _layout: &Layout,
        _latents: &Tensor,
        _video: Option<&Tensor>,
        _audio: Option<&Tensor>,
    ) -> nn::Result<()> {
        Ok(())
    }
}

/// A splicer whose every splice places one run: the clip's first frames it keeps.
struct Timeline(TemporalClips);

impl Splicer for Timeline {
    fn element_type(&self) -> DType {
        DType::F16
    }

    fn pixel_norm(&self) -> PixelNorm {
        PixelNorm {
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }

    fn splices(&self, frames: u32) -> nn::Result<Vec<Splice>> {
        self.0.splices((frames - self.0.overlap) / self.0.body)
    }

    fn splice(
        &self,
        _previous: Option<&Tensor>,
        clip: &Tensor,
        splice: Splice,
    ) -> nn::Result<Vec<(Tensor, u32)>> {
        Ok(vec![(clip.narrow(1, 0, splice.keep as usize)?, 0)])
    }
}

fn layout(video: LatentShape, audio: Option<LatentShape>) -> Layout {
    Layout {
        video,
        audio,
        text_rows: 4,
        condition_rows: 0,
        condition_elements: 0,
    }
}

/// The layout of a track of eight latent steps of four channels.
fn track() -> Layout {
    let shape = |channels, frames| LatentShape {
        channels,
        frames,
        height: 1,
        width: 1,
    };
    layout(shape(4, 2), Some(shape(4, 8)))
}

/// A runner over the meta device of `buckets` alone in its process, placing clips with the
/// timeline's splicer when `video`.
fn runner(buckets: Vec<Bucket>, video: bool) -> DecoderRunner {
    let memory = BufferMemory::meta(0, (PRODUCTS, 32), (SLOTS, 4), 0, &[]).unwrap();
    let round = RoundConfig {
        ranks: vec![0],
        units: 2,
    };

    decoder(
        buckets,
        video,
        (Arc::new(memory), round),
        (FRAMES, TRAJECTORY),
    )
}

/// A runner over the meta device of `buckets` over `memory`, dealing rounds by `round`, for the
/// timeline `timeline`, placing clips with the timeline's splicer when `video`.
fn decoder(
    buckets: Vec<Bucket>,
    video: bool,
    (memory, round): (Arc<BufferMemory>, RoundConfig),
    timeline: (u32, u32),
) -> DecoderRunner {
    let splicer = TemporalClips {
        body: 4,
        padding: 0,
        overlap: 2,
        temporal: 4,
        advance: 1,
    };
    let splicer: Option<Arc<dyn Splicer + Send + Sync>> = match video {
        true => Some(Arc::new(Timeline(splicer))),
        false => None,
    };
    let phase = match video {
        true => Phase::VideoDecode,
        false => Phase::AudioDecode,
    };

    DecoderRunner::new(
        Arc::new(Clips),
        splicer,
        DecoderConfig {
            phases: vec![phase],
            buckets: Buckets::new(buckets),
            classes: CLASSES,
            timelines: vec![timeline],
            round,
        },
        Graphs::meta(Arc::new(Generation::new(0, [])), 0),
        (None, None),
        memory,
    )
    .unwrap()
}

/// The rank's tables with the request's products at `blocks`, `(class, first position, runs,
/// pinned)` each: allocated, or pinned read-only.
fn tables(blocks: &[(u8, u32, &[Run], bool)]) -> BlockTables {
    let tables = BlockTables::new(GROUP, &[SLOTS, PRODUCTS], &[(PRODUCTS, UNIT_BYTES)]);
    filled(tables, blocks, 1)
}

/// `tables` given the request's products at `blocks`, as [`tables`] states them, each block
/// with a run on each of `ranks` ranks, block after block.
fn filled(mut tables: BlockTables, blocks: &[(u8, u32, &[Run], bool)], ranks: u32) -> BlockTables {
    let owner = OwnerId(1);
    let attach = Request::Attach {
        owner,
        group: GROUP,
    };
    let attached = Completion::Attached { slot: SLOT };
    tables
        .map(SLOT, runtime::RequestId(0), &attach, &attached)
        .unwrap();

    for (id, &(class, at, runs, pinned)) in (1..).zip(blocks) {
        let count = runs.len() as u32 / ranks;
        let (request, completion) = match pinned {
            false => (
                Request::Allocate {
                    owner,
                    class,
                    at,
                    count,
                    units: runs[0].2,
                },
                Completion::Allocated {
                    runs: runs.to_vec(),
                    evicted: vec![],
                },
            ),
            true => (
                Request::Pin {
                    owner,
                    class,
                    at,
                    keys: vec![Key([id as u8; 32]); runs.len()],
                },
                Completion::Pinned {
                    blocks: runs.iter().map(|run| vec![*run]).collect(),
                    host: 0,
                },
            ),
        };
        let id = runtime::RequestId(id);
        tables.map(SLOT, id, &request, &completion).unwrap();
    }
    tables
}

/// The products of a video request at positions 0, 1 and 2 to 4: its latents at unit 0, its
/// units' frames from unit 1, read-only when `pinned`, and its three tails of three units each
/// from unit 12.
fn video_tables(pinned: bool) -> BlockTables {
    let tails = [(PRODUCTS, 12, 3), (PRODUCTS, 15, 3), (PRODUCTS, 18, 3)];

    tables(&[
        (CLASSES[0], 0, &[(PRODUCTS, 0, 1)], false),
        (CLASSES[1], 1, &[(PRODUCTS, 1, 11)], pinned),
        (CLASSES[2], 2, &tails, false),
    ])
}

/// The batch of one call of `phase` for the request: its row from `unit`, naming `results` of
/// which it reads the first `reads`, run by rank 0 alone.
fn batch(phase: Phase, variant: Variant, unit: u32, results: Vec<u32>, reads: u32) -> Batch {
    dealt(phase, variant, unit, (results, reads), Some(0))
}

/// As [`batch`], for a row that names `rank`, or no rank for one every rank runs.
fn dealt(
    phase: Phase,
    variant: Variant,
    unit: u32,
    (results, reads): (Vec<u32>, u32),
    rank: Option<u32>,
) -> Batch {
    Batch {
        id: BatchId(1),
        group: GROUP,
        phase,
        kernels: 0,
        policy: 0,
        revision: 0,
        shape: Shape {
            rows: 1,
            tokens: 0,
            variant,
            extent: Extent::Request {
                frames: FRAMES,
                width: 8,
                height: 8,
                text_rows: 4,
                condition_rows: 0,
            },
        },
        requests: vec![RequestId(1)],
        columns: Columns::default(),
        media: vec![MediaRow {
            request: RequestId(1),
            slot: SLOT,
            conditioning: Vec::new(),
            media: Vec::new(),
            step: 0,
            steps: 0,
            seed: 0,
            width: 8,
            height: 8,
            frames: FRAMES,
            rounds: 0,
            unit,
            rank,
            image: None,
            results,
            reads,
        }],
        new_slots: Vec::new(),
    }
}

/// A video decode round from `unit`.
fn round(unit: u32, results: Vec<u32>, reads: u32) -> Batch {
    batch(Phase::VideoDecode, Variant::Decode, unit, results, reads)
}

#[test]
fn a_decode_round_writes_its_units_frames_and_tails_into_the_blocks_its_row_names() {
    let runner = runner(vec![Bucket::clip(8, 8, 8)], true);
    let (bucket, units) = runner
        .units(&round(0, vec![0, 1, 2, 3], 1), &video_tables(false))
        .unwrap();
    assert_eq!(bucket, Bucket::clip(8, 8, 8));
    assert_eq!(units.len(), 2, "one unit per tail the round writes");

    // A unit's window is its two latent frames of each of four channels: 16 bytes a frame,
    // a trajectory of four frames apart.
    let windows: Vec<_> = units
        .iter()
        .map(|(before, ..)| {
            let copy = before.copies()[0];
            (copy.source, copy.source_pitch, copy.width, copy.height)
        })
        .collect();
    assert_eq!(windows, [(0, 64, 32, 4), (16, 64, 32, 4)]);

    // Each places its own splice and keeps its bytes from its first frame of the timeline in the
    // units block: the first keeps four frames, the second four, the first two of them faded.
    let kept: Vec<_> = units.iter().map(|unit| (unit.2.2, unit.3)).collect();
    let splice = |keep, overlap| {
        Some(Splice {
            keep,
            overlap,
            first: 0,
        })
    };
    assert_eq!(
        kept,
        [(splice(4, 0), UNIT), (splice(4, 2), UNIT + 4 * FRAME)]
    );

    // Each leaves its successor the raster's last two frames of each of three channels, 128
    // bytes a frame, in its own tail block, and the second fades against the first's.
    let tails: Vec<_> = units
        .iter()
        .map(|unit| {
            let copy = unit.1.0.copies()[0];
            (copy.destination, copy.width, copy.height)
        })
        .collect();
    assert_eq!(tails, [(12 * UNIT, 256, 3), (15 * UNIT, 256, 3)]);
    assert!(
        units
            .iter()
            .all(|unit| (unit.1.1, unit.2.0) == (None, None)),
        "a rank that decodes the whole round hands no tail to another"
    );
    assert!(
        units[0].2.1.is_empty(),
        "the timeline's first unit fades against nothing"
    );
    assert_eq!(units[1].2.1.copies()[0].source, 12 * UNIT);
}

#[test]
fn the_tail_a_round_leaves_is_the_one_the_next_round_fades_against() {
    let runner = runner(vec![Bucket::clip(8, 8, 8)], true);
    let held = video_tables(false);
    let (_, first) = runner.units(&round(0, vec![0, 1, 2, 3], 1), &held).unwrap();
    let (_, second) = runner.units(&round(2, vec![0, 3, 1, 4], 2), &held).unwrap();

    // The last unit of the first round left its tail where the next round reads it: the whole
    // tail of two frames of three channels, 768 bytes.
    let left = first[1].1.0.copies()[0].destination;
    let read = second[0].2.1.copies()[0];
    assert_eq!((read.source, read.width), (left, 768));

    // The timeline's last unit reads the trajectory's last latent frames, keeps its body and its
    // tail from frame 8, and leaves no tail.
    assert_eq!(second[0].0.copies()[0].source, 32);
    assert_eq!(second[0].2.2.map(|splice| splice.keep), Some(6));
    assert_eq!(second[0].3, UNIT + 8 * FRAME);
    assert!(second[0].1.0.is_empty());
}

/// A timeline of six units of four frames and a tail of two, from seven latent frames.
const LONG: u32 = 26;
const LONG_TRAJECTORY: u32 = 7;

/// The product buffer of each rank of the process, and the first unit of each rank's products in
/// it.
const BUFFERS: [u8; 3] = [PRODUCTS, OTHER, THIRD];
const BASES: [u32; 3] = [0, 64, 128];

/// The bytes of a tail of two frames of three channels of an 8 by 8 raster of FP16.
const TAIL: u64 = 768;

/// The decoders of ranks 0 to `count - 1` of one process, over memories joined to each other,
/// dealing rounds of `units` units to each, and each rank's tables with the request's products of
/// the long timeline on every rank: its latents at unit 0 of each rank's products, its units'
/// frames from unit 2, and its six tails of three units each from unit 22, at positions 0, 1 and
/// 2 to 7.
fn dealt_ranks(count: usize, units: u32) -> Vec<(DecoderRunner, BlockTables)> {
    let ranks: Vec<u32> = (0..count as u32).collect();
    let mut memories: Vec<BufferMemory> = ranks
        .iter()
        .map(|&rank| {
            let products = (BUFFERS[rank as usize], 256);
            BufferMemory::meta(rank, products, (SLOTS, 4), 0, &[]).unwrap()
        })
        .collect();
    BufferMemory::connect(&mut memories).unwrap();

    let runs = |first: u32, units: u32| -> Vec<Run> {
        let at = |rank: u32| (BUFFERS[rank as usize], BASES[rank as usize] + first, units);
        ranks.iter().map(|&rank| at(rank)).collect()
    };
    let latents = runs(0, 2);
    let frames = runs(2, 20);
    let tails: Vec<Vec<Run>> = (0..6).map(|tail| runs(22 + 3 * tail, 3)).collect();
    let sizes: Vec<(u8, u64)> = BUFFERS[..count]
        .iter()
        .map(|&region| (region, UNIT_BYTES))
        .collect();

    memories
        .into_iter()
        .map(|memory| {
            let own = BUFFERS[memory.rank() as usize];
            let others: Vec<u8> = BUFFERS[..count]
                .iter()
                .copied()
                .filter(|&region| region != own)
                .collect();
            let round = RoundConfig {
                ranks: ranks.clone(),
                units,
            };
            let runner = decoder(
                vec![Bucket::clip(8, 8, 8)],
                true,
                (Arc::new(memory), round),
                (LONG, LONG_TRAJECTORY),
            );

            let tables = BlockTables::new(GROUP, &[SLOTS, own], &sizes).with_peers(&others);
            let mut blocks: Vec<(u8, u32, &[Run], bool)> = vec![
                (CLASSES[0], 0, &latents, false),
                (CLASSES[1], 1, &frames, false),
            ];
            blocks.extend(
                (2..)
                    .zip(&tails)
                    .map(|(at, runs)| (CLASSES[2], at, runs.as_slice(), false)),
            );
            (runner, filled(tables, &blocks, count as u32))
        })
        .collect()
}

/// A decode round of the long timeline from `unit` whose row names no rank.
fn long_round(unit: u32, results: Vec<u32>, reads: u32) -> Batch {
    let mut batch = dealt(
        Phase::VideoDecode,
        Variant::Decode,
        unit,
        (results, reads),
        None,
    );
    batch.media[0].frames = LONG;
    if let Extent::Request { frames, .. } = &mut batch.shape.extent {
        *frames = LONG;
    }
    batch
}

/// The address of unit `unit` of rank `rank`'s products, and of the tail block `index` there.
fn at(rank: usize, unit: u32) -> u64 {
    u64::from(BASES[rank] + unit) * UNIT
}

fn tail(rank: usize, index: u32) -> u64 {
    at(rank, 22 + 3 * index)
}

/// The window each of `units` reads its latents from.
fn windows(units: &[Unit]) -> Vec<u64> {
    units.iter().map(|unit| unit.0.copies()[0].source).collect()
}

/// What each of `units` exchanges with other ranks: the member it sends its tail to with the
/// tail's bytes, then the member it receives its predecessor's tail from with that tail's bytes.
fn exchanged(units: &[Unit]) -> Vec<[Option<(u32, u64)>; 2]> {
    units
        .iter()
        .map(|unit| [unit.1.1.map(|(_, to, bytes)| (to, bytes)), unit.2.0])
        .collect()
}

#[test]
fn rounds_are_dealt_to_the_ranks_in_order_and_in_reverse_alternately() {
    let round = RoundConfig {
        ranks: vec![5, 7, 9],
        units: 2,
    };

    // Round 0 deals its shares of two units in the order of the ranks, round 1 in reverse, round
    // 2 in order again.
    let dealt: Vec<u32> = (0..14).map(|unit| round.rank_of(unit).unwrap()).collect();
    assert_eq!(dealt, [5, 5, 7, 7, 9, 9, 9, 9, 7, 7, 5, 5, 5, 5]);

    // So every round's first unit falls to the rank that decoded the last unit before it.
    for first in [6, 12, 18, 24] {
        assert_eq!(
            round.rank_of(first),
            round.rank_of(first - 1),
            "unit {first}"
        );
    }

    // A rank's member index is its place among the ranks; a deal of no unit deals nothing.
    assert_eq!(
        (round.member(5), round.member(9), round.member(6)),
        (Some(0), Some(2), None)
    );
    let empty = RoundConfig {
        ranks: Vec::new(),
        units: 2,
    };
    assert_eq!(empty.rank_of(0), None);
}

#[test]
fn a_round_that_names_no_rank_is_dealt_to_the_ranks_and_exchanges_a_tail_between_them() {
    let ranks = dealt_ranks(2, 2);
    let [(first, first_tables), (second, second_tables)] = ranks.as_slice() else {
        panic!("two ranks");
    };

    // Four units on two ranks: rank 0 decodes units 0 and 1 from its own latents, rank 1 units 2
    // and 3 from its own.
    let call = long_round(0, vec![0, 1, 2, 3, 4, 5], 1);
    let (_, ours) = first.units(&call, first_tables).unwrap();
    let (_, theirs) = second.units(&call, second_tables).unwrap();
    assert_eq!(windows(&ours), [at(0, 0), at(0, 0) + 16]);
    assert_eq!(windows(&theirs), [at(1, 0) + 32, at(1, 0) + 48]);

    // Unit 1 leaves its tail in rank 0's run and sends it from there to member 1, which decodes
    // unit 2; unit 2 receives those bytes from member 0 into its tail buffer, which no copy
    // fills.
    assert_eq!(ours[1].1.0.copies()[0].destination, tail(0, 1));
    assert_eq!(ours[1].1.1, Some((tail(0, 1), 1, TAIL)));
    assert_eq!((theirs[0].1.1, theirs[0].2.0), (None, Some((0, TAIL))));
    assert!(theirs[0].2.1.is_empty());

    // Unit 3 fades against unit 2's tail on its own rank and sends its own to nobody: the next
    // round reads it once this one completed.
    assert_eq!(theirs[1].2.1.copies()[0].source, tail(1, 2));
    assert_eq!(theirs[1].1.0.copies()[0].destination, tail(1, 3));
    assert_eq!(exchanged(&theirs[1..]), [[None, None]]);
    assert_eq!(exchanged(&ours[..1]), [[None, None]]);

    // Each unit reads and writes in its rank's run what rank 0 alone does in its own when it
    // decodes the whole round, and fades against the same tail: unit 2's is the one unit 1 sent
    // from where it left it.
    let mut alone = long_round(0, vec![0, 1, 2, 3, 4, 5], 1);
    alone.media[0].rank = Some(0);
    let (_, single) = first.units(&alone, first_tables).unwrap();
    let mut dealt: Vec<_> = (ours.iter().map(|unit| relative(unit, at(0, 0))))
        .chain(theirs.iter().map(|unit| relative(unit, at(1, 0))))
        .collect();
    dealt[2].2 = ours[1].1.1.map(|(sent, ..)| sent - at(0, 0));
    let single: Vec<_> = single.iter().map(|unit| relative(unit, at(0, 0))).collect();
    assert_eq!(dealt, single);

    // The next round is dealt in reverse: its units are rank 1's, and unit 4 fades against unit
    // 3's tail where rank 1 left it, exchanging nothing; rank 0 decodes nothing of it.
    let next = long_round(4, vec![0, 5, 1, 6, 7], 2);
    let (_, ours) = first.units(&next, first_tables).unwrap();
    let (_, theirs) = second.units(&next, second_tables).unwrap();
    assert!(ours.is_empty());
    assert_eq!(windows(&theirs), [at(1, 0) + 64, at(1, 0) + 80]);
    assert_eq!(theirs[0].2.1.copies()[0].source, tail(1, 3));
    assert_eq!(exchanged(&theirs), [[None, None]; 2]);

    // A row that names no rank covers one round from its first unit.
    assert!(
        first
            .units(&long_round(2, vec![0, 3, 1, 4, 5], 2), first_tables)
            .is_err()
    );
    let wide = long_round(0, vec![0, 1, 2, 3, 4, 5, 6], 1);
    assert!(first.units(&wide, first_tables).is_err());
}

#[test]
fn a_share_of_one_unit_receives_its_predecessor_s_tail_and_sends_its_own() {
    let ranks = dealt_ranks(3, 1);
    let units = |call: &Batch| -> Vec<Vec<Unit>> {
        ranks
            .iter()
            .map(|(runner, tables)| runner.units(call, tables).unwrap().1)
            .collect()
    };

    // Round 0 deals units 0, 1 and 2 to members 0, 1 and 2: each but the first receives its
    // predecessor's tail from the member before it, and each but the last sends its own to the
    // member after it, a share of one unit doing both at once.
    let round = units(&long_round(0, vec![0, 1, 2, 3, 4], 1));
    assert_eq!(exchanged(&round[0]), [[Some((1, TAIL)), None]]);
    assert_eq!(exchanged(&round[1]), [[Some((2, TAIL)), Some((0, TAIL))]]);
    assert_eq!(exchanged(&round[2]), [[None, Some((1, TAIL))]]);

    // Round 1 deals units 3, 4 and 5 to members 2, 1 and 0: unit 3 fades against unit 2's tail
    // in member 2's own run, and the tails pass the other way; unit 5 ends the timeline and
    // leaves none.
    let round = units(&long_round(3, vec![0, 4, 1, 5, 6, 7], 2));
    assert_eq!(windows(&round[2]), [at(2, 0) + 48]);
    assert_eq!(round[2][0].2.1.copies()[0].source, tail(2, 2));
    assert_eq!(exchanged(&round[2]), [[Some((1, TAIL)), None]]);
    assert_eq!(exchanged(&round[1]), [[Some((0, TAIL)), Some((2, TAIL))]]);
    assert_eq!(exchanged(&round[0]), [[None, Some((1, TAIL))]]);
    assert!(round[0][0].1.0.is_empty());
}

/// What `unit` reads and writes, from `base`, the first byte of its rank's products: its
/// window, the tail it leaves, the tail it fades against, its splice and its bytes.
fn relative(unit: &Unit, base: u64) -> (u64, Option<u64>, Option<u64>, Option<Splice>, u64) {
    let (before, (left, _), (_, fade, splice), kept) = unit;
    let first = |copies: &PitchedBatch| copies.copies().first().copied();

    (
        before.copies()[0].source - base,
        first(left).map(|copy| copy.destination - base),
        first(fade).map(|copy| copy.source - base),
        *splice,
        kept - base,
    )
}

#[test]
fn a_row_that_names_a_block_it_cannot_use_is_refused() {
    let runner = runner(vec![Bucket::clip(8, 8, 8)], true);
    let held = video_tables(false);

    // A position the tables hold no entry for; a round after the first that names no tail
    // before it; a timeline the deployment does not serve.
    assert!(runner.units(&round(0, vec![0, 1, 2, 9], 1), &held).is_err());
    assert!(runner.units(&round(2, vec![0, 1, 4], 1), &held).is_err());
    let mut other = round(0, vec![0, 1, 2, 3], 1);
    other.media[0].frames = 18;
    assert!(runner.units(&other, &held).is_err());

    // Frames a cached block holds are never written.
    let pinned = video_tables(true);
    assert!(
        runner
            .units(&round(0, vec![0, 1, 2, 3], 1), &pinned)
            .is_err()
    );
}

#[test]
fn an_audio_decode_reads_its_track_and_writes_its_samples() {
    let bucket = Bucket::audio(1, track()).serving(FRAMES, 8, 8);
    let runner = runner(vec![bucket], false);
    let held = tables(&[
        (CLASSES[0], 0, &[(PRODUCTS, 3, 1)], false),
        (CLASSES[1], 1, &[(PRODUCTS, 7, 1)], false),
    ]);

    // Eight steps of four FP32 channels into the latent buffer; 32 stereo samples of 16 bits.
    let audio = batch(Phase::AudioDecode, Variant::AudioDecode, 0, vec![0, 1], 1);
    let (chosen, units) = runner.units(&audio, &held).unwrap();
    assert_eq!(chosen, bucket);
    let [(before, (left, _), (_, _, splice), kept)] = units.as_slice() else {
        panic!("a track is one unit");
    };
    let copy = before.copies()[0];
    assert_eq!(
        (copy.destination, copy.source, copy.width),
        (0, 3 * UNIT, 128)
    );
    assert_eq!(*kept, 7 * UNIT);
    assert!(
        left.is_empty() && splice.is_none(),
        "a track leaves no tail and places no splice"
    );
}

#[test]
fn every_bucket_is_profiled_and_captured_with_its_conversions() {
    let launches = |runner: &DecoderRunner, bucket: &Bucket, operator: &str| {
        let table = runner.graphs().launch_table(bucket).unwrap();
        table.iter().filter(|(name, ..)| *name == operator).count()
    };

    // A clip's graph decodes alone; its splice into the timeline runs eagerly, and the splice's
    // profile packs the bytes of each of the timeline's three distinct splices.
    let clip = Bucket::clip(8, 8, 8);
    let video = runner(vec![clip], true);
    let splice = clip.serving(FRAMES, 8, 8);
    assert_eq!(video.eager_buckets(), [splice]);
    video.graphs().profile_run(&video).unwrap();
    assert_eq!(launches(&video, &clip, RGB_BYTES), 0);
    assert_eq!(launches(&video, &splice, RGB_BYTES), 3);
    video.graphs().capture(&video, &clip).unwrap();

    // A track's graph interleaves its waveform once.
    let bucket = Bucket::audio(1, track()).serving(FRAMES, 8, 8);
    let audio = runner(vec![bucket], false);
    audio.graphs().profile_run(&audio).unwrap();
    assert_eq!(launches(&audio, &bucket, PCM_INTERLEAVE), 1);
    audio.graphs().capture(&audio, &bucket).unwrap();
}
