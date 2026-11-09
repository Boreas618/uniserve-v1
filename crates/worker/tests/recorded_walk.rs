//! A runner's recorded walk (`worker::runners::recording`): the programs and the arena extent it
//! reports are the ones the runner's own forwards launch and reserve, captured and eager alike, and
//! the walk issues nothing on the device.
//!
//! The models here are fakes whose every forward reserves a stated number of bytes and launches one
//! program of a stated key, so the expected footprint is a sum the test states. A runner binds
//! pinned columns and a stream, so these cases need a device; they issue no device work of their
//! own. Run them on a GPU host:
//!
//! ```text
//! cargo test -p worker --test recorded_walk -- --ignored
//! ```

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use nn::capability::{
    Clip, Conditioning, DecodedMedia, Decoder, Denoiser, LatentShape, Latents, Layout, Seed,
    StateTable, Step, VideoRequest, WarmRequest,
};
use nn::{Forward, Result};
use ops::logits::{GatherArgs, GatherDescriptor, LogitsGather};
use ops::{DType, Op, Shape, TensorView};
use platform::{Allocator, Budget, Device, DeviceId, PoolKind, Stream};
use worker::arena::Pools;
use worker::plan::{Bucket, Buckets};
use worker::runner::{Binding, Runner, RunnerId};
use worker::runners::denoiser::{DenoiserRunner, DenoiserShape};
use worker::runners::{DecoderRunner, recording::Footprint};

const DEVICE: DeviceId = DeviceId(0);

/// A budget that never refuses: these cases are about what a walk reports, not about memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> platform::Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// The device's pools, as a worker attaches them before it walks its runners.
fn pools() -> Arc<Mutex<Pools>> {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let totals = Arc::new(Pools::totals_of(u64::MAX / 2));
    let allocator = Allocator::new(&device, Arc::new(Allowance::default()) as Arc<dyn Budget>)
        .expect("an allocator over the device");
    Arc::new(Mutex::new(
        Pools::new(allocator, totals, 1 << 20, 0, 0, 2).expect("the device's pools"),
    ))
}

fn binding() -> Binding {
    Binding {
        id: RunnerId(0),
        phases: Vec::new(),
        device: DEVICE,
        stream: Stream::new(DEVICE).expect("a stream"),
    }
}

/// The program a fake forward launches: a gather whose width names the forward.
fn gather(columns: u32) -> GatherDescriptor {
    GatherDescriptor {
        dtype: DType::I32,
        columns,
        block: 128,
        pdl: false,
    }
}

/// One fake forward: reserve `bytes` of the arena and launch `gather(columns)` over them.
fn launch(f: &Forward<'_>, bytes: u64, columns: u32) -> Result<()> {
    let rows = bytes / 4 / u64::from(columns);
    let reserved = f
        .arena()
        .tensor(DType::I32, Shape::new(&[rows, u64::from(columns)]))?;
    f.operate::<LogitsGather>(
        &gather(columns),
        GatherArgs {
            hidden: reserved,
            rows: reserved.reshape(Shape::new(&[rows * u64::from(columns)])),
            out: reserved,
        },
    )
}

/// The keys of `gather(columns)` for each of `columns`, as a footprint states them.
fn keys(columns: &[u32]) -> BTreeMap<String, BTreeSet<String>> {
    BTreeMap::from([(
        LogitsGather::NAME.to_owned(),
        columns
            .iter()
            .map(|columns| ops::descriptor::key(&gather(*columns)))
            .collect(),
    )])
}

fn latent(channels: u32, frames: u32, height: u32, width: u32) -> LatentShape {
    LatentShape {
        channels,
        frames,
        height,
        width,
    }
}

/// A denoiser whose constants, opening forwards and step each reserve a stated extent.
///
/// The opening's three forwards share one arena from its first address (2 + 3 + 4 KiB) and the
/// captured step starts again at it (8 KiB), so the stream's peak is the opening's 9 KiB: an
/// eager forward beside a captured one is what sets it.
struct Stepping;

impl Denoiser for Stepping {
    fn layout(&self, _request: &VideoRequest<'_>) -> Layout {
        Layout {
            video: latent(4, 1, 2, 2),
            audio: None,
            text_rows: 8,
            condition_rows: 0,
            condition_elements: 0,
        }
    }

    fn schedule(&self, _request: &VideoRequest<'_>) -> Result<Vec<Step>> {
        Ok(vec![Step {
            index: 0,
            count: 1,
            sigma: 1.0,
            sigma_next: 0.0,
        }])
    }

    fn draw_noise(&self, f: &Forward, _: Seed, _: Layout, _: Latents) -> Result<()> {
        launch(f, 2048, 2)
    }

    fn prepare_conditioning(&self, f: &Forward, _: Conditioning<'_>) -> Result<()> {
        launch(f, 3072, 3)
    }

    fn prepare_latents(&self, f: &Forward, _: Conditioning<'_>, _: Latents) -> Result<()> {
        launch(f, 4096, 4)
    }

    fn denoise(&self, f: &Forward, _: Step, _: Latents, _: Conditioning<'_>) -> Result<()> {
        launch(f, 8192, 5)
    }

    fn evaluations(&self) -> u32 {
        1
    }

    fn constant_tables(&self) -> Vec<StateTable> {
        Vec::new()
    }

    fn prepare_constants(&self, f: &Forward, _: &[TensorView]) -> Result<()> {
        launch(f, 1024, 1)
    }

    fn layout_tables(&self, _: &Layout) -> Result<Vec<StateTable>> {
        Ok(Vec::new())
    }

    fn prepare_layout(&self, _: &Layout, _: &mut [Vec<u8>]) -> Result<()> {
        Ok(())
    }

    fn state_tables(&self, _: &Layout) -> Vec<StateTable> {
        vec![StateTable {
            name: "step",
            dtype: DType::I32,
            shape: Shape::new(&[1]),
        }]
    }

    fn prepare_state(&self, _: &VideoRequest<'_>, _: &Layout, _: &mut [Vec<u8>]) -> Result<()> {
        Ok(())
    }

    fn prepare_evaluation(&self, _: u32, _: &mut [Vec<u8>]) -> Result<()> {
        Ok(())
    }
}

fn warm() -> WarmRequest {
    WarmRequest {
        width: 16,
        height: 16,
        frames: 1,
        fps: 24,
        first_frames: 0,
        last_frames: 0,
        references: 0,
        text_tokens: 8,
        conditions: Vec::new(),
        vision_spans: Vec::new(),
        condition_rows: None,
    }
}

/// A denoiser runner's footprint is its constants, the opening of each bucket's warm request and
/// the step its graph captures: exactly their programs, and the furthest extent any of them
/// reached - the eager opening's, which a captured step alone would understate.
#[test]
#[ignore = "requires a CUDA device"]
fn a_denoiser_records_its_opening_and_its_step() {
    let model = Arc::new(Stepping);
    let mut runner = DenoiserRunner::new(
        binding(),
        Arc::clone(&model),
        vec![warm()],
        DenoiserShape {
            stack: DType::BF16,
            sample: DType::F32,
            width: 16,
            text_width: 16,
        },
        Arc::new(ops::Generation::new(0, [])),
    )
    .expect("a runner of one bucket");
    runner.attach_pools(pools());

    let footprint: Footprint = runner.record_walk().expect("a recorded walk");

    assert_eq!(footprint.keys, keys(&[1, 2, 3, 4, 5]));
    assert_eq!(footprint.peak, 2048 + 3072 + 4096);
    assert_eq!(
        footprint.widest,
        Some(Bucket::denoise(1, model.layout(&warm().view())).serving(1, 16, 16))
    );
}

/// A decoder whose gather, decode and splice each reserve a stated extent.
struct Decoding;

impl Decoder for Decoding {
    fn layout_tables(&self, _: &Layout) -> Result<Vec<StateTable>> {
        Ok(Vec::new())
    }

    fn prepare_layout(&self, _: &Layout, _: &mut [Vec<u8>]) -> Result<()> {
        Ok(())
    }

    fn element_type(&self) -> DType {
        DType::F32
    }

    fn layout_of(&self, clip: Clip) -> Result<Layout> {
        Ok(Layout {
            video: latent(4, clip.frames, clip.height / 8, clip.width / 8),
            audio: None,
            text_rows: 0,
            condition_rows: 0,
            condition_elements: 0,
        })
    }

    fn gather_latents(
        &self,
        f: &Forward,
        _: &Layout,
        _: TensorView,
        _: u32,
        _: TensorView,
    ) -> Result<()> {
        launch(f, 1024, 6)
    }

    fn decode(&self, f: &Forward, _: TensorView, _: &[TensorView], _: DecodedMedia) -> Result<()> {
        launch(f, 16384, 8)
    }
}

/// A decoder runner's footprint is the gather a replay issues before its graph and the decode the
/// graph captures, and the peak is the decode's.
#[test]
#[ignore = "requires a CUDA device"]
fn a_decoder_records_its_gather_and_its_decode() {
    let bucket = Bucket::clip(1, 16, 16);
    let mut runner = DecoderRunner::new(
        binding(),
        Arc::new(Decoding),
        Buckets::new([bucket]),
        Arc::new(ops::Generation::new(0, [])),
        None,
    )
    .expect("a runner of one canvas");
    runner.attach_pools(pools());

    let footprint = runner.record_walk().expect("a recorded walk");

    assert_eq!(footprint.keys, keys(&[6, 8]));
    assert_eq!(footprint.peak, 16384);
    assert_eq!(footprint.widest, Some(bucket));
}
