//! 4.5's rebind on a device: a captured bucket, given a new program's launch parameters, launches
//! that program when it is replayed.
//!
//! The mechanism this file drives is the executor's, without a mock: a `platform::Capture` records a
//! graph of one operator launch while `nn::forward::Forward` runs it, the runner's own sink
//! (`executor::runners::causal_lm::Recording`) collects the per-launch node table that `ops::capture`
//! reports, and `Captured::rebind_launch` sets a walk's parameters on the captured executable. Three
//! properties, each read off the device:
//!
//! - **a capture records a node table.** The proof that `ops::capture::launched` has a caller: a
//!   capture whose node table is empty fails here by name, because a graph with no node table serves
//!   correctly and can never be rebound - the state the slice was in while the operator launches went
//!   unreported;
//! - **the node table names the operator and its descriptor key**, so `Graphs::using` can say which
//!   buckets launch an operator a program update changes;
//! - **a rebound graph launches the walk's parameters, not the ones it was captured with.** The probe
//!   adds `value` to every word, so the captured graph leaves 7s and the rebound one leaves 21s: if
//!   `rebind_launch` set nothing, the replay yields the captured value and this test fails.
//!
//! Run it on a GPU host, alone on the device:
//!
//! ```text
//! cargo test -p executor --test rebind -- --ignored --test-threads=1
//! ```
//!
//! It needs `nvcc` on `PATH`, and `UNISERVE_PROBE_ARCH` names the architecture it compiles for
//! (default `sm_100a`).

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]

use std::fmt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use executor::captured::Captured;
use executor::runners::causal_lm::Recording;
use nn::forward::Forward;
use ops::{
    Arguments, CubinMetadata, DType, DevicePtr, Generation, Kernel, KernelEntry, LaunchGrid, Op,
    Shape, TensorView, Tolerance, Twins, descriptor,
};
use platform::{
    Allocator, Budget, Buffer, CaptureMode, Device, DeviceId, GraphNodeKind, PinnedBuffer, PoolKind,
    Stream, memcpy_async, memset_d32,
};

const DEVICE: DeviceId = DeviceId(0);
/// The words the probe operates on, the value the captured graph adds, and the value the walk adds.
const WORDS: usize = 32;
const BYTES: usize = WORDS * 4;
const CAPTURED: u32 = 7;
const WALKED: u32 = 21;

/// A budget that records the pool's cumulative total and never refuses: the probe is about a graph's
/// node parameters, not about running out of device memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> platform::Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// The probe operator: add `value` to each of `count` words in place.
struct Probe;

/// What one compilation of the probe covers: how many words it is launched for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProbeDescriptor {
    words: u32,
}

impl fmt::Display for ProbeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "words={}", self.words)
    }
}

/// One call's arguments: the activation and the two scalars the kernel reads. `value` is what a
/// rebind changes: it is a launch argument, not a descriptor value, so the two generations' keys are
/// the same and the parameters are the only thing that moves.
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
        Self::Args {
            data: twins.in_place(args.data, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

/// Compiles the probe's kernels into one cubin.
fn cubin() -> PathBuf {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/probe_generation.cu");
    let output = std::env::temp_dir().join(format!("probe-generation-{}.cubin", std::process::id()));
    let arch = std::env::var("UNISERVE_PROBE_ARCH").unwrap_or_else(|_| "sm_100a".to_owned());
    let status = Command::new("nvcc")
        .arg("-cubin")
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&output)
        .arg(&source)
        .status()
        .expect("this probe compiles its kernels with nvcc, which must be on PATH");
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

/// One capture of `generation`'s forward, as the runner makes it: the template graph a rebind reads
/// parameters from, and the node table the operator launches reported.
fn capture(
    stream: &Stream,
    generation: &Generation,
    arena: &nn::Arena,
    descriptor: &ProbeDescriptor,
    args: ProbeArgs,
) -> (platform::Graph, executor::captured::Nodes) {
    let capture = Arc::new(Mutex::new(
        platform::Capture::begin(stream, CaptureMode::Relaxed).expect("a capture begins"),
    ));
    let mut recording = Recording::new(Arc::clone(&capture));
    ops::capture::set_sink(Some(&mut recording));
    let forward = Forward::new(stream, generation, arena);
    forward
        .operate::<Probe>(descriptor, args)
        .expect("the probe launches");
    ops::capture::set_sink(None);
    if let Some(failure) = recording.failure() {
        panic!("the capture failed while recording a launch: {failure}");
    }
    let nodes = recording.into_nodes();
    let template = Arc::try_unwrap(capture)
        .unwrap_or_else(|_| panic!("the capture is still shared by its recorder"))
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .end()
        .expect("the capture ends");
    (template, nodes)
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_rebound_bucket_launches_the_walk_s_parameters() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let mut allocator =
        Allocator::new(&device, Arc::new(Allowance::default())).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, 1 << 20)
        .expect("reserve the graph pool");
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
    // The serving generation and the one a program update prepares. The keys are the same - the two
    // programs are keyed by the same descriptor - so what the rebind moves is the parameters.
    let serving = Generation::new(
        1,
        [KernelEntry::keyed(
            Probe::NAME,
            key.clone(),
            kernel("probe_exact"),
        )],
    );
    let prepared = Generation::new(
        2,
        [KernelEntry::keyed(
            Probe::NAME,
            key.clone(),
            kernel("probe_equivalent"),
        )],
    );
    // The probe reserves no activation, so its forward needs no arena.
    let arena = nn::Arena::new(DevicePtr::new(0), 0);
    let data = TensorView::new(
        DevicePtr::new(buffer.address()),
        DType::I32,
        Shape::new(&[WORDS as u64]),
    );

    // The serving capture, as the runner's warmup makes it.
    let (template, nodes) = capture(
        &stream,
        &serving,
        &arena,
        &descriptor,
        ProbeArgs {
            data,
            count: WORDS as u32,
            value: CAPTURED,
        },
    );

    // The node table is the proof that `ops::capture::launched` has a caller. Without it the graph
    // serves and can never be rebound, which is the failure this assertion exists to catch.
    assert!(
        !nodes.is_empty(),
        "the capture recorded no node table: `Forward::operate` did not report its operator launch \
         to `ops::capture`, so this bucket can never be rebound"
    );
    for (operator, captured_key, launch) in &nodes {
        assert_eq!(*operator, Probe::NAME, "the launch was recorded under its operator");
        assert_eq!(*captured_key, key, "the launch was recorded under its descriptor key");
        assert!(
            !launch.is_empty(),
            "the launch added no graph node: the captured graph would have nothing to rebind"
        );
    }

    let captured = Captured::new(template, nodes, vec![kernel("probe_exact")])
        .expect("a captured bucket");
    assert!(
        captured.kernel_nodes() > 0,
        "the captured graph has no kernel node: `kernel_nodes` is what a release's counters read"
    );
    assert!(captured.uses(Probe::NAME), "the graph launches the operator");
    assert_eq!(
        captured.descriptors(Probe::NAME),
        vec![key.as_str()],
        "the graph names the descriptor key it was captured for"
    );

    // The precopy walk: the prepared generation's forward, captured on the same stream, which is
    // what computes the parameters a rebind sets.
    let (walked_template, walked_nodes) = capture(
        &stream,
        &prepared,
        &arena,
        &descriptor,
        ProbeArgs {
            data,
            count: WORDS as u32,
            value: WALKED,
        },
    );
    assert!(
        captured
            .same_structure(&walked_nodes)
            .expect("the launch structures compare"),
        "one launch of one kernel in both captures is the same structure: this bucket is rebound, \
         not recaptured"
    );

    let mut params = Vec::new();
    for (_, _, launch) in &walked_nodes {
        for node in launch {
            if node.kind().expect("a node kind") == GraphNodeKind::Kernel {
                params.push(
                    walked_template
                        .kernel_node_params(node)
                        .expect("a kernel node's parameters"),
                );
            }
        }
    }
    assert!(!params.is_empty(), "the walk launched no kernel to take parameters from");
    let applied = captured
        .rebind_launch(Probe::NAME, 0, &params)
        .expect("the bucket is rebound");
    assert_eq!(
        applied,
        params.len(),
        "every kernel node of the launch is set, not a subset"
    );

    // The replay is the measurement: the walk added 21 to every word, so a graph that launches the
    // walk's parameters leaves 21s and one that still launches the capture's leaves 7s.
    captured.launch(&stream).expect("the rebound graph replays");
    assert_eq!(
        words(&stream, buffer.address()),
        vec![WALKED; WORDS],
        "the rebound graph launched the parameters it was captured with, not the walk's"
    );
}
