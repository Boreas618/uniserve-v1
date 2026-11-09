//! 7.3 step 2 on a device: a paired forward accepts a re-implementation that agrees numerically
//! and refuses one that does not.
//!
//! The mechanism this file drives is the worker's, end to end and without a mock: a `Generation`
//! of kernels (`ops::Generation`, one entry per operator and descriptor), `nn::forward::Forward`'s
//! shadow branch (`Forward::operate` runs the new program into `ops::Twins` and the old one into the
//! real output), and the worker's harness (`worker::shadow::Shadow`) which compares the pairs
//! with the tolerance the operator recorded.
//!
//! Three properties, each of which the test reads off the device:
//!
//! - **an agreeing re-implementation passes**, and the real output is the *old* kernel's - so the
//!   paired forward leaves the forward it belongs to on the previous program's results;
//! - **a re-implementation that differs passes only if the comparison allows it**: the wrong kernel
//!   differs by exactly 1 in every element and is refused, naming element 0 with the two values;
//! - **the comparison is what refuses it**: delete `Shadow::compare` and the run leaves the wrong
//!   twins in place with no error, which is the failure this test exists to catch.
//!
//! Run it on a GPU host, alone on the device:
//!
//! ```text
//! cargo test -p worker --test shadow_update -- --ignored --test-threads=1
//! ```
//!
//! It needs `CUDA_HOME` to name the CUDA toolkit, and `UNISERVE_PROBE_ARCH` names the architecture it compiles for
//! (default `sm_100a`).
//!
//! The update's other two paths are measured at the process level instead, by the delivery test of
//! the slice (`tests/python/e2e/test_program_update_http.py`, the caller's run): rebind and
//! recapture there are driven by the Program Server's packages and a real serving deployment, and a
//! Rust test that fabricated a rebind's node parameters would measure its own fabrication - the walk
//! that computes them (`ops::capture::launched`) has no caller yet.

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]

use std::fmt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use nn::forward::Forward;
use ops::{
    Arguments, CubinMetadata, DType, DevicePtr, Generation, Kernel, KernelEntry, LaunchGrid, Op,
    Shape, TensorView, Tolerance, Twins, descriptor,
};
use platform::{
    Allocator, Budget, Buffer, Device, DeviceId, PinnedBuffer, PoolKind, Stream, memcpy_async,
    memset_d32,
};
use worker::shadow::Shadow;

const DEVICE: DeviceId = DeviceId(0);
/// The words the probe operates on, and the value the test adds to each of them.
const WORDS: usize = 32;
const VALUE: u32 = 7;
const BYTES: usize = WORDS * 4;

/// A budget that records the pool's cumulative total and never refuses: the probe is about the
/// shadow region's addresses, not about running out of device memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> platform::Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// The probe operator: add `value` to each of `count` words in place.
///
/// It stands for any operator whose output is one of its operands, which is the case `Twins::in_place`
/// exists for: the paired forward's twin must start from the real bytes, because the new program
/// modifies what it reads.
struct Probe;

/// What one compilation of the probe covers: how many words it is launched for.
///
/// The word count is a descriptor value rather than a launch value only because the probe has no
/// per-call parameters worth keying on; a real operator keys its descriptor on the dtypes and layout
/// it was compiled for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProbeDescriptor {
    words: u32,
}

impl fmt::Display for ProbeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "words={}", self.words)
    }
}

/// One call's arguments: the activation and the two scalars the kernel reads.
#[derive(Clone, Copy)]
struct ProbeArgs {
    data: TensorView,
    count: u32,
    value: u32,
}

impl Op for Probe {
    const NAME: &'static str = "probe_accumulate";

    type Descriptor = ProbeDescriptor;
    type Args<'a> = ProbeArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> ops::Result<()> {
        let mut arguments = Arguments::new();
        arguments.view(args.data).u32(args.count).u32(args.value);
        let threads = 128_u64;
        let blocks = u64::from(descriptor.words).div_ceil(threads).max(1);
        kernel.launch(LaunchGrid::new([blocks, 1, 1])?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // The probe writes its input in place, so the twin starts as a copy of the real bytes and
        // the new program modifies the copy. `in_place` does both.
        Self::Args {
            data: twins.in_place(args.data, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // An integer addition: two implementations that add the same values agree bit for bit, so a
        // difference of one element is a difference in the program and not rounding.
        Tolerance::exact()
    }
}

/// Compiles the probe's three generations into one cubin.
fn cubin() -> PathBuf {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/probe_generation.cu");
    let output =
        std::env::temp_dir().join(format!("probe-generation-{}.cubin", std::process::id()));
    let arch = std::env::var("UNISERVE_PROBE_ARCH").unwrap_or_else(|_| "sm_100a".to_owned());
    let status = Command::new(
        std::path::Path::new(
            &std::env::var_os("CUDA_HOME")
                .expect("CUDA_HOME names the toolkit whose nvcc compiles this probe's kernel"),
        )
        .join("bin/nvcc"),
    )
    .arg("-cubin")
    .arg(format!("-arch={arch}"))
    .arg("-o")
    .arg(&output)
    .arg(&source)
    .status()
    .expect("running nvcc");
    assert!(status.success(), "nvcc failed on {}", source.display());
    output
}

/// The words `address` holds, read through pinned staging after the stream completed.
fn words(stream: &Stream, address: u64) -> Vec<u32> {
    let host = PinnedBuffer::new(DEVICE, BYTES).expect("pinned staging");
    // SAFETY: the copy reads `BYTES` from the allocation into a pinned buffer of the same size, and
    // the stream that wrote the allocation is synchronized before the buffer is read.
    unsafe { memcpy_async(host.address(), address, BYTES, stream) }.expect("copy the words back");
    stream.synchronize().expect("synchronize the stream");
    // SAFETY: the copy completed and the buffer holds `BYTES` readable bytes.
    unsafe { host.as_slice() }
        .chunks_exact(4)
        .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_paired_forward_accepts_agreement_and_refuses_a_wrong_kernel() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let mut allocator =
        Allocator::new(&device, Arc::new(Allowance::default())).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, 1 << 20)
        .expect("reserve the graph pool");
    // The twin region is carved from the scratch pool, and an allocator refuses a pool it was never
    // given: the paired forward's own region is the probe's second reservation (4.1, 7.3).
    allocator
        .reserve(PoolKind::Scratch, 1 << 20)
        .expect("reserve the scratch pool the twins are carved from");
    let buffer: Buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("the activation buffer");
    let stream = Stream::new(DEVICE).expect("a stream");
    // SAFETY: the fill lies inside the allocation.
    unsafe { memset_d32(buffer.address(), 0, WORDS, &stream) }.expect("zero the activation");

    let image = std::fs::read(cubin()).expect("the compiled probe kernels");
    let kernel = |symbol: &str| -> Arc<Kernel> {
        Arc::new(
            Kernel::load_cubin(&device, &image, CubinMetadata::named(symbol))
                .unwrap_or_else(|error| panic!("loading {symbol}: {error}")),
        )
    };
    let descriptor = ProbeDescriptor {
        words: WORDS as u32,
    };
    let key = descriptor::key(&descriptor);
    // One generation per program: the serving one and two candidates for it.
    let serving = Generation::new(
        1,
        [KernelEntry::keyed(
            Probe::NAME,
            key.clone(),
            kernel("probe_exact"),
        )],
    );
    let equivalent = Generation::new(
        2,
        [KernelEntry::keyed(
            Probe::NAME,
            key.clone(),
            kernel("probe_equivalent"),
        )],
    );
    let wrong = Generation::new(
        3,
        [KernelEntry::keyed(
            Probe::NAME,
            key.clone(),
            kernel("probe_wrong"),
        )],
    );

    let twins = Twins::new(&mut allocator, BYTES as u64).expect("the twin region");
    let mut harness = Shadow::new(DEVICE, twins).expect("the shadow harness");
    // The probe reserves no activation, so its forward needs no arena: the region is empty and any
    // reservation inside it would fail.
    let arena = nn::Arena::new(DevicePtr::new(0), 0);
    let data = TensorView::new(
        DevicePtr::new(buffer.address()),
        DType::I32,
        Shape::new(&[WORDS as u64]),
    );
    let args = ProbeArgs {
        data,
        count: WORDS as u32,
        value: VALUE,
    };

    // A re-implementation that agrees: the paired forward runs, the comparison passes, and the real
    // output is the old kernel's, because the new one wrote only the twins.
    harness.reset();
    {
        let shadow = nn::forward::Shadow::new(Probe::NAME, key.clone(), &serving, harness.twins());
        let forward = Forward::new(&stream, &equivalent, &arena).with_shadow(shadow);
        forward
            .operate::<Probe>(&descriptor, args)
            .expect("the paired forward runs");
    }
    let tolerance = harness
        .tolerance()
        .expect("the paired forward recorded its tolerance");
    let compared = harness
        .compare(&stream, tolerance)
        .expect("two implementations that add the same values must agree");
    assert_eq!(
        compared.pairs, 1,
        "the probe's in-place output is one pair to compare"
    );
    assert_eq!(compared.elements, WORDS as u64);
    assert_eq!(
        words(&stream, buffer.address()),
        vec![VALUE; WORDS],
        "the real output must be the previous program's: the new one wrote the twins only"
    );

    // A re-implementation that differs by one in every element: refused, and the refusal names the
    // element with both values, which are the fixture's arithmetic (7 + 7 and 7 + 7 + 1).
    harness.reset();
    {
        let shadow = nn::forward::Shadow::new(Probe::NAME, key.clone(), &serving, harness.twins());
        let forward = Forward::new(&stream, &wrong, &arena).with_shadow(shadow);
        forward
            .operate::<Probe>(&descriptor, args)
            .expect("the paired forward runs");
    }
    let tolerance = harness
        .tolerance()
        .expect("the paired forward recorded its tolerance");
    let error = harness
        .compare(&stream, tolerance)
        .expect_err("a program that adds one more than the old one must be refused");
    let message = error.to_string();
    assert!(message.contains("differs beyond tolerance"), "{message}");
    assert!(message.contains("element 0"), "{message}");
    assert!(
        message.contains(&format!("reference {}", 2 * VALUE)),
        "the old kernel's result is 7 + 7: {message}"
    );
    assert!(
        message.contains(&format!("candidate {}", 2 * VALUE + 1)),
        "the wrong kernel's result is 7 + 7 + 1: {message}"
    );
    assert_eq!(
        words(&stream, buffer.address()),
        vec![2 * VALUE; WORDS],
        "the refused candidate must not have written the real output"
    );
}
