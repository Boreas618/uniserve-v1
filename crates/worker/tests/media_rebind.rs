//! A media runner prepares a generation update's buckets under the lifecycle's rebind rule: a rebind
//! that names the generation the runner serves, or a bucket the runner holds no serving graph of,
//! is refused before anything is walked, so a live graph is never replaced under the batches
//! running on it.
//!
//! The audio decoder runner is the case: it once prepared its buckets by capturing them afresh,
//! which skipped the rule and replaced the serving generation's graph when a rebind named it. A
//! runner binds pinned columns and a stream, so this needs a device; it issues no device work. Run
//! it on a GPU host:
//!
//! ```text
//! cargo test -p worker --test media_rebind -- --ignored
//! ```

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]

use std::sync::{Arc, Mutex};

use nn::capability::{DecodedMedia, Decoder};
use nn::{Forward, Result};
use ops::Operand;
use platform::{Allocator, Budget, Device, DeviceId, PoolKind, Stream};
use proto::video::{LatentShape, Layout};
use uniserve_tensor::DType;
use worker::arena::Pools;
use worker::batch::Job;
use worker::plan::{Bucket, Buckets};
use worker::runner::{Binding, Runner, RunnerId};
use worker::runners::AudioDecoderRunner;

const DEVICE: DeviceId = DeviceId(0);

/// The generation the runner is built serving.
const SERVING: u64 = 0;

/// A budget that never refuses: the case is about which rebind is refused, not about memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> platform::Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// The device's pools, as a worker attaches them before it captures its runners.
fn pools() -> Arc<Mutex<Pools>> {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let totals = Arc::new(Pools::totals_of(1 << 30));
    let allocator = Allocator::new(&device, Arc::new(Allowance::default()) as Arc<dyn Budget>)
        .expect("an allocator over the device");
    Arc::new(Mutex::new(
        Pools::new(allocator, totals, 1 << 20, 0, 0, 2).expect("the device's pools"),
    ))
}

/// An audio decoder of a 64-sample hop whose decode launches nothing.
struct Vocoder;

impl Decoder for Vocoder {
    fn element_type(&self) -> DType {
        DType::F32
    }

    fn pcm_window(&self, audio: &LatentShape) -> Result<(u64, u32)> {
        Ok((u64::from(audio.frames) * 64, 2))
    }

    fn decode(&self, _: &Forward, _: &Layout, _: Operand, _: DecodedMedia) -> Result<()> {
        Ok(())
    }
}

/// The one track bucket the runner serves.
fn track() -> Bucket {
    let latent = |frames| LatentShape {
        channels: 4,
        frames,
        height: 1,
        width: 1,
    };
    Bucket::audio(
        1,
        Layout {
            video: latent(1),
            audio: Some(latent(8)),
            text_rows: 0,
            condition_rows: 0,
            condition_elements: 0,
        },
    )
}

fn rebind(bucket: Bucket, generation: u64) -> Job {
    Job::Rebind {
        runner: RunnerId(0),
        bucket,
        generation,
    }
}

/// A rebind naming the serving generation, or a bucket with no serving graph, is the lifecycle's
/// refusal by name, before the runner walks the bucket.
#[test]
#[ignore = "requires a CUDA device"]
fn an_audio_rebind_is_refused_where_the_lifecycle_refuses_it() {
    let bucket = track();
    let mut runner = AudioDecoderRunner::new(
        Binding {
            id: RunnerId(0),
            phases: Vec::new(),
            device: DEVICE,
            stream: Stream::new(DEVICE).expect("a stream"),
        },
        Arc::new(Vocoder),
        Buckets::new([bucket]),
        Arc::new(ops::Generation::new(SERVING, [])),
    )
    .expect("a runner of one track");
    runner.attach_pools(pools());

    let serving = runner
        .serve(&rebind(bucket, SERVING))
        .expect_err("a rebind of the serving generation replaces a live graph")
        .to_string();
    assert!(
        serving.contains("serving generation"),
        "the refusal names the serving generation: {serving}"
    );

    let unserved = runner
        .serve(&rebind(bucket, SERVING + 1))
        .expect_err("a bucket with no serving graph has no structure to rebind")
        .to_string();
    assert!(
        unserved.contains("no graph of bucket"),
        "the refusal names the missing serving graph: {unserved}"
    );
}
