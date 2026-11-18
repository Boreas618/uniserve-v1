//! `DenoiserRunner` without a device: where a latent preparation and the steps of a request read
//! and write its slot's row and its product blocks, the save and restore of its trajectory in one
//! encoding and across two, and the profile run and capture of its buckets, over a denoiser whose
//! methods launch nothing.

use std::collections::HashMap;
use std::sync::Arc;

use crate::batch::{Batch, BatchId, Columns, Extent, MediaRow, Phase, Shape, Variant};
use nn::{Conditioning, Denoiser, LatentShape, Step, VideoRequest};
use ops::Generation;
use proto::request::RequestId;
use proto::video::Layout;
use storage::Key;
use storage::runtime::{self, Completion, GroupId, OwnerId, Request, Run};
use tensor::{DType, Tensor};

use super::{DenoiserConfig, DenoiserRunner};
use crate::memory::buffers::{BufferMemory, ByteCopy, UNIT_BYTES};
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::Runner;
use crate::runner::address;
use crate::runner::graphs::Graphs;

const GROUP: GroupId = GroupId(0);

/// The slot rows, the product buffers of two ranks of one process, and the host latent region.
const SLOTS: u8 = 0;
const PRODUCTS: u8 = 2;
const OTHER_PRODUCTS: u8 = 3;
const LATENTS: u8 = 6;

/// The classes of the conditioning, the condition pixels and samples, and the two latents.
const CLASSES: [u8; 5] = [4, 5, 6, 7, 8];

/// A unit of a rank's part of the host latent region.
const PAGE: u64 = 256;

/// The request's slot.
const SLOT: u32 = 2;

/// A request of 8 frames at 32 by 32: video latents `[4, 2, 4, 4]` and audio latents `[2, 6]` in
/// FP32, 512 and 48 bytes; 8 prompt rows and the zero row of conditioning, 32 wide for the
/// denoiser and 16 for the encoder in FP16, 576 and 288 bytes. Its row holds the latents from
/// byte 0, the denoiser's rows from byte 768 and the encoder's from byte 1536: 1824 bytes, rows
/// 2048 apart.
const VIDEO: u64 = 512;
const AUDIO: u64 = 48;
const TEXT: u64 = 768;
const SOURCE: u64 = 1536;
const ROW: u64 = 2048;

/// A denoiser whose layout is its request's raster in cells of eight, two schedule steps, and
/// a conditioner that widens the encoder's rows to twice their width.
struct Model;

impl Denoiser for Model {
    fn layout(&self, request: &VideoRequest) -> nn::Result<Layout> {
        let shape = |channels, frames, height, width| LatentShape {
            channels,
            frames,
            height,
            width,
        };
        Ok(Layout {
            video: shape(4, request.frames / 4, request.height / 8, request.width / 8),
            audio: Some(shape(2, 6, 1, 1)),
            text_rows: 8,
            condition_rows: 0,
            condition_elements: 0,
        })
    }

    fn schedule(&self, _request: &VideoRequest) -> nn::Result<Vec<Step>> {
        let step = |index, sigma, sigma_next| Step {
            index,
            count: 2,
            sigma,
            sigma_next,
        };
        Ok(vec![step(0, 1.0, 0.5), step(1, 0.5, 0.0)])
    }

    fn text_width(&self) -> Option<u32> {
        Some(32)
    }

    fn draw_noise(&self, _: u64, _: &Layout, _: &Tensor, _: Option<&Tensor>) -> nn::Result<()> {
        Ok(())
    }

    fn prepare_latents(&self, _: &Conditioning, _: &Tensor, _: Option<&Tensor>) -> nn::Result<()> {
        Ok(())
    }

    fn prepare_conditioning(&self, _: &VideoRequest, _: &Conditioning) -> nn::Result<()> {
        Ok(())
    }

    fn forward(
        &self,
        _: Step,
        _: &VideoRequest,
        _: &Tensor,
        _: Option<&Tensor>,
        _: &Conditioning,
    ) -> nn::Result<()> {
        Ok(())
    }

    fn initialize(&self) -> nn::Result<()> {
        Ok(())
    }
}

/// A candidate of `frames` frames at 32 by 32 with a prompt of `tokens` tokens.
fn candidate(frames: u32, tokens: u32) -> VideoRequest {
    VideoRequest {
        width: 32,
        height: 32,
        frames,
        fps: 24,
        roles: Vec::new(),
        text_tokens: tokens,
        conditions: Vec::new(),
        vision_spans: Vec::new(),
        condition_rows: None,
    }
}

fn config(candidates: Vec<VideoRequest>, ranks: u32) -> DenoiserConfig {
    DenoiserConfig {
        phases: vec![Phase::LatentPrepare, Phase::Denoise],
        candidates: candidates
            .into_iter()
            .map(|candidate| (candidate, Vec::new()))
            .collect(),
        stack: DType::F16,
        sample: DType::F32,
        width: 16,
        classes: DenoiserRunner::PRODUCTS.into_iter().zip(CLASSES).collect(),
        ranks,
        page_bytes: PAGE,
        first: true,
    }
}

/// A runner over the meta device on `memory`.
fn runner(config: DenoiserConfig, memory: BufferMemory) -> DenoiserRunner {
    let graphs = Graphs::meta(Arc::new(Generation::new(0, [])), 0);

    DenoiserRunner::new(Arc::new(Model), config, graphs, None, Arc::new(memory)).unwrap()
}

/// The meta memory of rank `rank` with its product buffer in `products`, three slot rows and,
/// when it `maps` one, its part of the host latent region.
fn memory(rank: u32, products: u8, maps: bool) -> BufferMemory {
    let row = DenoiserRunner::row_bytes(&Model, &config(vec![candidate(8, 8)], 1)).unwrap();
    assert_eq!(row, SOURCE + 288);

    let parts: &[(u8, u64, u32)] = if maps { &[(LATENTS, PAGE, 32)] } else { &[] };
    BufferMemory::meta(rank, (products, 64), (SLOTS, 3), row, parts).unwrap()
}

/// The tables of a rank with its product buffer in `products`, the request attached on `slot`
/// and the blocks of `products` at `(class, position, run)` each.
fn tables(products: u8, slot: u32, blocks: &[(u8, u32, Run)]) -> BlockTables {
    let units = [(SLOTS, ROW), (products, UNIT_BYTES), (LATENTS, PAGE)];
    let mut tables = BlockTables::new(GROUP, &[SLOTS, products, LATENTS], &units);
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: GROUP,
    };
    tables
        .map(
            slot,
            runtime::RequestId(0),
            &attach,
            &Completion::Attached { slot },
        )
        .unwrap();

    for (id, &(class, at, run)) in (1..).zip(blocks) {
        let allocate = Request::Allocate {
            owner: OwnerId(1),
            class,
            at,
            count: 1,
            units: run.2,
        };
        let allocated = Completion::Allocated {
            runs: vec![run],
            evicted: vec![],
        };
        tables
            .map(slot, runtime::RequestId(id), &allocate, &allocated)
            .unwrap();
    }
    tables
}

/// Hands `tables` the `Copy` that answers `request` from `from` to `to` in `encodings`, and
/// starts it as the next batch would.
fn started(
    tables: &mut BlockTables,
    (id, request): (u64, &Request),
    (from, to): (Run, Run),
    encodings: Option<([u8; 32], [u8; 32])>,
) -> BlockCopy {
    let copy = Completion::Copy {
        group: GROUP,
        keys: vec![Key([7; 32])],
        from: vec![from],
        to: vec![to],
        evicted: vec![],
        encodings,
        parts: Vec::new(),
    };
    tables
        .map(0, runtime::RequestId(id), request, &copy)
        .unwrap();
    tables.start_copy().unwrap()
}

/// The batch of the request's call of `variant` at step `step` of two, naming `results` of
/// which it reads the first `reads`.
fn batch(variant: Variant, step: u32, results: Vec<u32>, reads: u32) -> Batch {
    let conditioning = proto::video::Conditions {
        conditions: Vec::new(),
        prompt_tokens: 5,
        vision_spans: Vec::new(),
    };
    let phase = match variant {
        Variant::LatentPrepare => Phase::LatentPrepare,
        _ => Phase::Denoise,
    };
    let extent = Extent::Request {
        frames: 8,
        width: 32,
        height: 32,
        text_rows: 5,
        condition_rows: 0,
    };

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
            extent,
        },
        requests: vec![RequestId(1)],
        columns: Columns::default(),
        media: vec![MediaRow {
            request: RequestId(1),
            slot: SLOT,
            conditioning: conditioning.encode(),
            media: Vec::new(),
            step,
            steps: 2,
            seed: 7,
            width: 32,
            height: 32,
            frames: 8,
            rounds: 0,
            unit: 0,
            rank: None,
            image: None,
            results,
            reads,
        }],
        new_slots: Vec::new(),
    }
}

/// Performs `copies` on host mirrors of the ranks' memories, by region and rank index.
fn apply(mirrors: &mut HashMap<(u8, u32), Vec<u8>>, copies: &[ByteCopy]) {
    for &((from, reader, source), (to, writer, destination), bytes) in copies {
        let (source, destination, bytes) = (source as usize, destination as usize, bytes as usize);
        let read = mirrors[&(from, reader)][source..source + bytes].to_vec();
        let written = mirrors.get_mut(&(to, writer)).unwrap();
        written[destination..destination + bytes].copy_from_slice(&read);
    }
}

#[test]
fn a_preparation_and_two_steps_keep_the_state_in_the_row_and_the_last_writes_the_latents() {
    // The conditioning at unit 0, the video latents at units 4 and 5, the audio latents at 8.
    let held = tables(
        PRODUCTS,
        SLOT,
        &[
            (CLASSES[0], 0, (PRODUCTS, 0, 1)),
            (CLASSES[3], 1, (PRODUCTS, 4, 2)),
            (CLASSES[4], 2, (PRODUCTS, 8, 1)),
        ],
    );
    let runner = runner(config(vec![candidate(8, 8)], 1), memory(0, PRODUCTS, true));

    // The preparation reads the prompt's five rows from the conditioning block, and opens the
    // request over its slot's row: the latents from the row's first byte, then the two widths
    // of conditioning rows.
    let prepare = batch(Variant::LatentPrepare, 0, vec![0], 1);
    let (bucket, row) = runner.call(&prepare).unwrap();
    assert_eq!(bucket.variant, Variant::LatentPrepare);
    let (_, request, _, [conditioning, pixels, samples]) =
        runner.opening(&bucket, row, &held).unwrap();
    let conditioning = conditioning.unwrap();
    assert_eq!(request.text_tokens, 5);
    assert_eq!(conditioning.dims(), [5, 16]);
    assert_eq!(address(&conditioning).unwrap(), 0);
    assert!(pixels.is_none() && samples.is_none());

    let (_, layout, ..) = runner.served(&bucket).unwrap();
    let (video, audio, text, source) = runner.state(layout, Some(SLOT)).unwrap();
    let at = |tensor: &Tensor| address(tensor).unwrap() - ROW;
    assert_eq!((video.dims(), at(&video)), ([4, 2, 4, 4].as_slice(), 0));
    let audio = audio.unwrap();
    assert_eq!((audio.dims(), at(&audio)), ([2, 6].as_slice(), VIDEO));
    assert_eq!((text.dims(), at(&text)), ([9, 32].as_slice(), TEXT));
    assert_eq!((source.dims(), at(&source)), ([9, 16].as_slice(), SOURCE));

    // Each step reads the row's latents and conditioning into the state buffer and writes the
    // latents back; the first writes nothing else.
    let first = batch(Variant::Denoise, 0, Vec::new(), 0);
    let (bucket, row) = runner.call(&first).unwrap();
    assert_eq!(runner.staged(&bucket, row).unwrap().0.index, 0);
    let (before, after, written) = runner.moves(&bucket, row, &held, false).unwrap();
    let copies = |batch: &platform::PitchedBatch| -> Vec<(u64, u64, usize)> {
        let copies = batch.copies().iter();
        copies
            .map(|copy| (copy.source, copy.destination, copy.width))
            .collect()
    };
    let latents = (VIDEO + AUDIO) as usize;
    assert_eq!(copies(&before), [(ROW, 0, (TEXT + 576) as usize)]);
    assert_eq!(copies(&after), [(0, ROW, latents)]);
    assert!(written.is_empty());

    // The last also copies the latents into the blocks its row names.
    let last = batch(Variant::Denoise, 1, vec![1, 2], 0);
    let (bucket, row) = runner.call(&last).unwrap();
    let (_, after, written) = runner.moves(&bucket, row, &held, true).unwrap();
    let unit = UNIT_BYTES;
    assert_eq!(
        copies(&after),
        [
            (0, ROW, latents),
            (0, 4 * unit, VIDEO as usize),
            (VIDEO, 8 * unit, AUDIO as usize)
        ]
    );
    assert_eq!(written, [(CLASSES[3], 1), (CLASSES[4], 2)]);

    // A step before the last names no result, the last names its latents, and a preparation
    // reads no latents.
    assert!(runner.moves(&bucket, row, &held, false).is_err());
    let (bucket, row) = runner.call(&first).unwrap();
    assert!(runner.moves(&bucket, row, &held, true).is_err());
    let reading = batch(Variant::LatentPrepare, 0, vec![0, 1], 2);
    let (bucket, row) = runner.call(&reading).unwrap();
    assert!(runner.opening(&bucket, row, &held).is_err());
}

#[test]
fn a_trajectory_saved_and_restored_in_its_encoding_returns_to_the_rows_latents() {
    let mut held = tables(PRODUCTS, SLOT, &[]);
    let attach = Request::Attach {
        owner: OwnerId(2),
        group: GROUP,
    };
    held.map(
        1,
        runtime::RequestId(50),
        &attach,
        &Completion::Attached { slot: 1 },
    )
    .unwrap();
    let mut runner = runner(config(vec![candidate(8, 8)], 1), memory(0, PRODUCTS, true));

    // The trajectory of the request's shape is its latents, three pages of the host region.
    let shapes = runner.media_shapes().unwrap();
    let [(_, trajectory, latents)] = shapes[..] else {
        panic!("one served shape");
    };
    assert_eq!((trajectory, latents), (VIDEO + AUDIO, [VIDEO, AUDIO]));
    let pages = trajectory.div_ceil(PAGE) as u32;

    let saved: Vec<u8> = (0..trajectory).map(|byte| (byte % 251) as u8).collect();
    let mut mirrors = HashMap::from([
        ((SLOTS, 0), vec![0xa5; (3 * ROW) as usize]),
        ((LATENTS, 0), vec![0; (32 * PAGE) as usize]),
    ]);
    let row = ROW as usize..(ROW + trajectory) as usize;
    mirrors.get_mut(&(SLOTS, 0)).unwrap()[row].copy_from_slice(&saved);

    // Saved from slot 2 into units 5 to 7, and restored into slot 1's row, whose latents a step
    // reads from its first byte.
    let save = Request::Save {
        owner: OwnerId(1),
        key: Key([1; 32]),
        units: pages,
    };
    let save = started(
        &mut held,
        (51, &save),
        ((SLOTS, SLOT, 1), (LATENTS, 5, pages)),
        None,
    );
    runner.copy(std::slice::from_ref(&save)).unwrap();
    apply(&mut mirrors, &runner.memory.byte_copies(&[save]).unwrap());

    let restore = Request::Restore {
        owner: OwnerId(2),
        key: Key([1; 32]),
    };
    let restore = started(
        &mut held,
        (52, &restore),
        ((LATENTS, 5, pages), (SLOTS, 1, 1)),
        None,
    );
    runner.copy(std::slice::from_ref(&restore)).unwrap();
    apply(
        &mut mirrors,
        &runner.memory.byte_copies(&[restore]).unwrap(),
    );

    assert_eq!(mirrors[&(SLOTS, 0)][..trajectory as usize], saved[..]);
    let (_, layout, ..) = runner.served(&runner.buckets()[0]).unwrap();
    let (video, ..) = runner.state(layout, Some(1)).unwrap();
    assert_eq!(address(&video).unwrap(), 0);
}

#[test]
fn a_trajectory_saved_in_another_encoding_is_restored_from_the_part_its_process_maps() {
    // Two ranks of one process: rank 0 maps the process's part of the host latent region, and
    // rank 1, a rank of a group of two that restores, maps none and copies with rank 0's.
    let mut memories = vec![memory(0, PRODUCTS, true), memory(1, OTHER_PRODUCTS, false)];
    BufferMemory::connect(&mut memories).unwrap();
    let other = memories.pop().unwrap();
    let mut runner = runner(config(vec![candidate(8, 8)], 2), other);

    // The class states one host copy, whatever the width of the group that wrote it.
    let (_, encoding) = runner.trajectory_digests();
    let alone = self::runner(config(vec![candidate(8, 8)], 1), memory(0, PRODUCTS, true));
    assert_eq!(alone.trajectory_digests().1, encoding);

    let trajectory = VIDEO + AUDIO;
    let pages = trajectory.div_ceil(PAGE) as u32;
    let saved: Vec<u8> = (0..trajectory).map(|byte| (byte % 239) as u8).collect();
    let mut part = vec![0; (32 * PAGE) as usize];
    part[(5 * PAGE) as usize..(5 * PAGE + trajectory) as usize].copy_from_slice(&saved);
    let mut mirrors = HashMap::from([
        ((SLOTS, 1), vec![0; (3 * ROW) as usize]),
        ((LATENTS, 0), part),
    ]);

    // Saved by a deployment whose largest layout is another: another encoding of the class's
    // tensors, which the request's own layout lays out alike.
    let mut held = tables(OTHER_PRODUCTS, SLOT, &[]);
    let restore = Request::Restore {
        owner: OwnerId(1),
        key: Key([1; 32]),
    };
    let runs = ((LATENTS, 5, pages), (SLOTS, SLOT, 1));
    let elsewhere = [9; 32];
    let copy = started(&mut held, (50, &restore), runs, Some((elsewhere, encoding)));

    let (sources, target) = runner.restoring(&copy).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].0, 0, "the part rank 0 maps holds the host copy");
    let moved = runner.memory.restored(&copy, &sources, &target).unwrap();
    apply(&mut mirrors, &moved);
    let row = ROW as usize..(ROW + trajectory) as usize;
    assert_eq!(mirrors[&(SLOTS, 1)][row], saved[..]);
    runner.copy(std::slice::from_ref(&copy)).unwrap();

    // A copy into an encoding other than the class's, or one naming the class's on both sides,
    // is refused.
    for encodings in [(elsewhere, [8; 32]), (encoding, encoding)] {
        let copy = BlockCopy {
            encodings: Some(encodings),
            ..copy.clone()
        };
        assert!(runner.copy(&[copy]).is_err());
    }
}

#[test]
fn each_layout_is_profiled_its_steps_captured_and_a_batch_with_no_row_returns_no_call() {
    // A shorter prompt of the first shape takes its layout: two buckets for three candidates.
    let candidates = vec![candidate(8, 8), candidate(8, 4), candidate(16, 8)];
    let mut runner = runner(config(candidates, 1), memory(0, PRODUCTS, true));
    let (steps, preparations) = (runner.buckets(), runner.eager_buckets());
    assert_eq!(steps.len(), 2);
    assert!(
        steps
            .iter()
            .all(|bucket| bucket.variant == Variant::Denoise)
    );
    assert!(
        preparations
            .iter()
            .all(|bucket| bucket.variant == Variant::LatentPrepare)
    );
    let (_, _, warm, _) = runner.served(&steps[0]).unwrap();
    assert_eq!(
        warm.text_tokens, 8,
        "a collapsed layout keeps its first candidate"
    );

    runner.graphs().profile_run(&runner).unwrap();
    for bucket in &preparations {
        assert!(runner.graphs().launch_table(bucket).is_some());
    }
    for bucket in &steps {
        runner.graphs().capture(&runner, bucket).unwrap();
    }
    assert_eq!(runner.graphs().buckets().len(), 2);

    let mut empty = batch(Variant::Denoise, 0, Vec::new(), 0);
    empty.requests.clear();
    empty.media.clear();
    let output = runner
        .execute(&empty, &tables(PRODUCTS, SLOT, &[]))
        .unwrap();
    assert!(output.calls.is_empty());
}
