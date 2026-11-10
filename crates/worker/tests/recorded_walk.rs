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

use ipc::execution::Variant;
use nn::capability::{
    Clip, ConditionPiece, ConditionRole, ConditionShape, Conditioning, DecodedMedia, Decoder,
    Denoiser, Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, ImageShape,
    Latents, MediaRole, PatchFormat, PixelNorm, Seed, StateTable, Step, TableValues, TokenBatch,
    VideoRequest, WarmPrompt, WarmRequest,
};
use nn::{Forward, Result};
use ops::logits::{GatherArgs, GatherDescriptor, LogitsGather};
use ops::{DType, Op, Shape, TensorView};
use platform::{Allocator, Budget, Device, DeviceId, PoolKind, Stream};
use proto::video::{LatentShape, Layout};
use worker::arena::Pools;
use worker::plan::{Bucket, Buckets};
use worker::runner::{Binding, Runner, RunnerId};
use worker::runners::denoiser::{DenoiserRunner, DenoiserShape};
use worker::runners::encoder::{EncoderRunner, EncoderShape, WarmVision};
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
    // An allowance the device can reserve address space for; the walks book a few KiB.
    let totals = Arc::new(Pools::totals_of(1 << 30));
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
            // One row index per gathered row, read from the start of the reserved rows.
            rows: ops::TensorView::new(reserved.ptr, DType::I32, Shape::new(&[rows])),
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
    fn layout(&self, _request: &VideoRequest<'_>) -> Result<Layout> {
        Ok(Layout {
            video: latent(4, 1, 2, 2),
            audio: None,
            text_rows: 8,
            condition_rows: 0,
            condition_elements: 0,
        })
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

    fn prepare_layout(&self, _: &Layout, _: &mut [TableValues]) -> Result<()> {
        Ok(())
    }

    fn state_tables(&self, _: &Layout) -> Result<Vec<StateTable>> {
        Ok(vec![StateTable {
            name: "step",
            dtype: DType::I32,
            shape: Shape::new(&[1]),
        }])
    }

    fn prepare_state(&self, _: &VideoRequest<'_>, _: &Layout, _: &mut [TableValues]) -> Result<()> {
        Ok(())
    }

    fn prepare_evaluation(&self, _: u32, _: &mut [TableValues]) -> Result<()> {
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
        openings: Vec::new(),
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
        Some(Bucket::denoise(1, model.layout(&warm().view()).unwrap()).serving(1, 16, 16))
    );
}

/// A decoder whose gather, decode and splice each reserve a stated extent.
struct Decoding;

impl Decoder for Decoding {
    fn layout_tables(&self, _: &Layout) -> Result<Vec<StateTable>> {
        Ok(Vec::new())
    }

    fn prepare_layout(&self, _: &Layout, _: &mut [TableValues]) -> Result<()> {
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

/// The row width of the fake media format's patch rows.
const PATCH_WIDTH: u32 = 24;

/// Patch rows of `PATCH_WIDTH` BF16 elements (`3 · 2 · 2²`): what an encoder's recorded walk lays a
/// vision prompt's placeholder rows out by.
fn patches() -> PatchFormat {
    let format = PatchFormat {
        side: 2,
        temporal: 2,
        merge: 2,
        channels: 3,
        norm: PixelNorm {
            mean: [0.5; 3],
            std: [0.5; 3],
        },
        dtype: DType::BF16,
    };
    assert_eq!(format.width(), PATCH_WIDTH);
    format
}

/// An encoder whose text encode reserves 1 KiB and whose vision encode reserves 4 KiB per image,
/// launching a gather whose width names the image count (`2` for text, `4 · images` with them).
///
/// Its vision encode refuses patch rows that are not the media format's width or that do not hold
/// each image's grid, so a walk that laid them out otherwise fails instead of reporting.
struct Prompting;

impl Encoder for Prompting {
    fn layout(&self, batch: &TokenBatch, images: &[ImageShape]) -> EncoderLayout {
        EncoderLayout {
            rows: batch.num_tokens,
            tokens: batch.num_tokens,
            images: images.to_vec(),
        }
    }

    fn state_tables(&self, layout: &EncoderLayout) -> Result<Vec<StateTable>> {
        Ok(vec![StateTable {
            name: "positions",
            dtype: DType::I32,
            shape: Shape::new(&[u64::from(layout.tokens)]),
        }])
    }

    fn prepare_state(
        &self,
        _: &EncoderPrompt<'_>,
        _: &EncoderLayout,
        _: &mut [TableValues],
    ) -> Result<()> {
        Ok(())
    }

    fn encode(&self, f: &Forward, inputs: EncoderInputs<'_>, _: EncoderOutputs) -> Result<()> {
        let images = inputs.images.len() as u64;
        if images == 0 {
            return launch(f, 1024, 2);
        }
        for (image, shape) in inputs.images.iter().zip(&inputs.layout.images) {
            let rows: u64 = shape.grid.iter().map(|extent| u64::from(*extent)).product();
            if image.pixels.shape.dims() != [rows, u64::from(PATCH_WIDTH)] {
                return Err(nn::Error::Unsupported(format!(
                    "patch rows {:?} for a grid {:?}",
                    image.pixels.shape.dims(),
                    shape.grid
                )));
            }
        }
        launch(f, 4096 * images, 4 * images as u32)
    }
}

/// An encoder runner's footprint holds the eager encode of every admitted vision prompt beside the
/// captured text encode of each bucket that prompt rides: a one-row bucket of at least its vision
/// tokens. The vision encode reserves more than the text encode, so a walk of the captured encode
/// alone would claim a region the eager one overruns.
#[test]
#[ignore = "requires a CUDA device"]
fn an_encoder_records_its_vision_prompts() {
    let one = Bucket::prefill(1, 8, Variant::Encode);
    let two = Bucket::prefill(2, 8, Variant::Encode);
    let image = ImageShape { grid: [1, 2, 4] };
    let mut runner = EncoderRunner::new(
        binding(),
        Arc::new(Prompting),
        Buckets::new([one, two]),
        WarmVision {
            prompts: vec![
                // Two image tokens: the one-row bucket of eight serves it.
                WarmPrompt {
                    images: vec![image],
                    tokens: 2,
                },
                // Sixteen tokens: no bucket serves it, so it is never walked.
                WarmPrompt {
                    images: vec![image, image],
                    tokens: 16,
                },
            ],
            patches: patches(),
        },
        EncoderShape {
            hidden: 16,
            dtype: DType::BF16,
        },
        Arc::new(ops::Generation::new(0, [])),
    )
    .expect("a runner of two buckets");
    runner.attach_pools(pools());

    let footprint = runner.record_walk().expect("a recorded walk");

    assert_eq!(footprint.keys, keys(&[2, 4]));
    assert_eq!(footprint.peak, 4096);
    assert_eq!(footprint.widest, Some(one));
}

/// A denoiser like [`Stepping`] that takes keyframes: each condition is one piece of its pixels,
/// and a conditioning encode of `n` pieces reserves `n × 16` KiB and launches `gather(8 · n)`,
/// against 3 KiB and `gather(3)` without any.
///
/// Its encode refuses media without the pixels of each piece, so an opening that laid out no
/// region for them fails instead of reporting.
struct Anchoring;

impl Denoiser for Anchoring {
    fn layout(&self, request: &VideoRequest<'_>) -> Result<Layout> {
        Stepping.layout(request)
    }

    fn schedule(&self, request: &VideoRequest<'_>) -> Result<Vec<Step>> {
        Stepping.schedule(request)
    }

    fn condition_pieces(&self, request: &VideoRequest<'_>) -> Result<Vec<ConditionPiece>> {
        Ok((0_u32..)
            .zip(request.conditions)
            .map(|(condition, shape)| ConditionPiece {
                condition,
                role: match shape.role {
                    ConditionRole::FirstFrame => MediaRole::FirstFrame,
                    ConditionRole::LastFrame => MediaRole::LastFrame,
                    ConditionRole::Reference => MediaRole::Reference,
                },
                rows: 4,
                audio_rows: 0,
                pixels: Some([
                    1,
                    3,
                    u64::from(shape.frames),
                    u64::from(shape.height),
                    u64::from(shape.width),
                ]),
                track: None,
                draw: None,
                draw_offset: 0,
            })
            .collect())
    }

    fn draw_noise(&self, f: &Forward, seed: Seed, layout: Layout, latents: Latents) -> Result<()> {
        Stepping.draw_noise(f, seed, layout, latents)
    }

    fn prepare_conditioning(&self, f: &Forward, cond: Conditioning<'_>) -> Result<()> {
        let pieces = cond.media.len() as u64;
        if pieces == 0 {
            return Stepping.prepare_conditioning(f, cond);
        }
        if cond.media.iter().any(|piece| piece.pixels.is_none()) {
            return Err(nn::Error::Unsupported(
                "a keyframe without its pixels".into(),
            ));
        }
        launch(f, 16384 * pieces, 8 * pieces as u32)
    }

    fn prepare_latents(&self, f: &Forward, cond: Conditioning<'_>, latents: Latents) -> Result<()> {
        Stepping.prepare_latents(f, cond, latents)
    }

    fn denoise(
        &self,
        f: &Forward,
        step: Step,
        latents: Latents,
        cond: Conditioning<'_>,
    ) -> Result<()> {
        Stepping.denoise(f, step, latents, cond)
    }

    fn evaluations(&self) -> u32 {
        1
    }

    fn constant_tables(&self) -> Vec<StateTable> {
        Stepping.constant_tables()
    }

    fn prepare_constants(&self, f: &Forward, tables: &[TensorView]) -> Result<()> {
        Stepping.prepare_constants(f, tables)
    }

    fn layout_tables(&self, layout: &Layout) -> Result<Vec<StateTable>> {
        Stepping.layout_tables(layout)
    }

    fn prepare_layout(&self, layout: &Layout, out: &mut [TableValues]) -> Result<()> {
        Stepping.prepare_layout(layout, out)
    }

    fn state_tables(&self, layout: &Layout) -> Result<Vec<StateTable>> {
        Stepping.state_tables(layout)
    }

    fn prepare_state(
        &self,
        request: &VideoRequest<'_>,
        layout: &Layout,
        out: &mut [TableValues],
    ) -> Result<()> {
        Stepping.prepare_state(request, layout, out)
    }

    fn prepare_evaluation(&self, index: u32, out: &mut [TableValues]) -> Result<()> {
        Stepping.prepare_evaluation(index, out)
    }
}

/// A keyframe of the warm request's 16x16 canvas.
fn keyframe(role: ConditionRole) -> ConditionShape {
    ConditionShape {
        role,
        frames: 1,
        height: 16,
        width: 16,
        audio_samples: 0,
    }
}

/// A denoiser runner's footprint holds the opening of every condition set its bucket admits beside
/// the warm request's own: the conditioning encode of the widest set is what sets the stream's
/// peak, and each set's programs are recorded.
#[test]
#[ignore = "requires a CUDA device"]
fn a_denoiser_records_the_openings_its_conditions_admit() {
    let model = Arc::new(Anchoring);
    let warm = WarmRequest {
        first_frames: 1,
        last_frames: 1,
        openings: vec![
            vec![keyframe(ConditionRole::FirstFrame)],
            vec![
                keyframe(ConditionRole::FirstFrame),
                keyframe(ConditionRole::LastFrame),
            ],
        ],
        ..warm()
    };
    let mut runner = DenoiserRunner::new(
        binding(),
        Arc::clone(&model),
        vec![warm.clone()],
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

    let footprint = runner.record_walk().expect("a recorded walk");

    // The constants, the warm opening (draw, conditioning, latents), the step, and the encode of
    // one keyframe and of two.
    assert_eq!(footprint.keys, keys(&[1, 2, 3, 4, 5, 8, 16]));
    // Two keyframes' opening: the draw, their 32 KiB encode, then the latents' start.
    assert_eq!(footprint.peak, 2048 + 32768 + 4096);
    assert_eq!(
        footprint.widest,
        Some(Bucket::denoise(1, model.layout(&warm.view()).unwrap()).serving(1, 16, 16))
    );
}
