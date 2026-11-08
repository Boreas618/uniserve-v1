//! 4.5's rebind on a device: a captured bucket, given a new program's launch parameters, launches
//! that program when it is replayed.
//!
//! The mechanism this file drives is the worker's, without a mock: a `platform::Capture` records a
//! graph of one operator launch while `nn::forward::Forward` runs it, the runner's own sink
//! (`worker::runners::causal_lm::Recording`) collects the per-launch node table that `ops::capture`
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
//! cargo test -p worker --test rebind -- --ignored --test-threads=1
//! ```
//!
//! It needs `nvcc` on `PATH`, and `UNISERVE_PROBE_ARCH` names the architecture it compiles for
//! (default `sm_100a`).

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]

use std::cell::RefCell;
use std::fmt;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use nn::forward::Forward;
use ops::{
    Arguments, CubinMetadata, DType, DevicePtr, Generation, Kernel, KernelEntry, LaunchGrid, Op,
    Shape, TensorView, Tolerance, Twins, descriptor,
};
use platform::{
    Allocator, Budget, Buffer, CaptureMode, Device, DeviceId, GraphNodeKind, PinnedBuffer,
    PoolKind, Stream, memcpy_async, memset_d32,
};
use worker::captured::Captured;
use worker::graph::Graphs;
use worker::plan::{Bucket, Shape as PlanShape, Variant};
use worker::runners::causal_lm::Recording;

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
    let output =
        std::env::temp_dir().join(format!("probe-generation-{}.cubin", std::process::id()));
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
) -> (platform::Graph, worker::captured::Nodes) {
    let capture = Rc::new(RefCell::new(
        platform::Capture::begin(stream, CaptureMode::Relaxed).expect("a capture begins"),
    ));
    let mut recording = Recording::new(Rc::clone(&capture));
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
    let template = Rc::try_unwrap(capture)
        .unwrap_or_else(|_| panic!("the capture is still shared by its recorder"))
        .into_inner()
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
        assert_eq!(
            *operator,
            Probe::NAME,
            "the launch was recorded under its operator"
        );
        assert_eq!(
            *captured_key, key,
            "the launch was recorded under its descriptor key"
        );
        assert!(
            !launch.is_empty(),
            "the launch added no graph node: the captured graph would have nothing to rebind"
        );
    }

    let captured =
        Captured::new(template, nodes, vec![kernel("probe_exact")]).expect("a captured bucket");
    assert!(
        captured.kernel_nodes() > 0,
        "the captured graph has no kernel node: `kernel_nodes` is what a release's counters read"
    );
    assert!(
        captured.uses(Probe::NAME),
        "the graph launches the operator"
    );
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
        for walked in launch {
            if walked.node.kind().expect("a node kind") == GraphNodeKind::Kernel {
                params.push(
                    walked_template
                        .kernel_node_params(&walked.node)
                        .expect("a kernel node's parameters"),
                );
            }
        }
    }
    assert!(
        !params.is_empty(),
        "the walk launched no kernel to take parameters from"
    );
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

/// 7.3 steps 3-4 on a device: a **revert after a recapture** replays the old generation's graph
/// bitwise, and a **rebind** leaves the serving generation's graph alone.
///
/// The two arms exist because they fail for different reasons and their repairs live in different
/// files. A recaptured graph that clobbers an address the old graph reads is an arena defect
/// (`Pools`/`Allocator` ranges); a rebind that changes what the serving generation computes is a
/// defect in `Captured::from_template`'s rule that the prepared generation gets its **own**
/// executable from the same `Graph` - the rule that makes a revert possible at all.
///
/// What this test cannot measure and does not claim: the mapped-bytes figure coming back on purge.
/// The activation buffer is the test's own here, not the runner's, so `mapped_bytes` would be
/// constant by construction and the assertion would be vacuous; the reuse of a bucket's bytes across
/// two generations is pinned on the host by `two_generations_capture_inside_one_mapped_range`.
#[test]
#[ignore = "requires a CUDA device"]
fn a_revert_after_a_recapture_replays_the_serving_generation_bitwise() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let mut allocator =
        Allocator::new(&device, Arc::new(Allowance::default())).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, 1 << 20)
        .expect("reserve the graph pool");
    let stream = Stream::new(DEVICE).expect("a stream");
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
    // The probe reserves no activation, so its forward needs no arena.
    let arena = nn::Arena::new(DevicePtr::new(0), 0);

    // Each generation addresses its own activation, as two generations in one graph range do.
    let serving_buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("the serving generation's activation");
    let successor_buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("the successor's activation");
    let view = |buffer: &Buffer| {
        TensorView::new(
            DevicePtr::new(buffer.address()),
            DType::I32,
            Shape::new(&[WORDS as u64]),
        )
    };
    // The operator adds in place, so every replay starts from the same storage: the graph addresses
    // the buffer by number, and its content is the only thing a replay may depend on.
    let replay = |captured: &Captured, buffer: &Buffer| -> Vec<u32> {
        // SAFETY: the fill lies inside the allocation, which the stream still owns.
        unsafe { memset_d32(buffer.address(), 0, WORDS, &stream) }.expect("zero the activation");
        captured.launch(&stream).expect("the graph replays");
        words(&stream, buffer.address())
    };

    // The serving generation, as `precopy` finds it, and the benchmark it leaves.
    let (serving_template, serving_nodes) = capture(
        &stream,
        &Generation::new(
            1,
            [KernelEntry::keyed(
                Probe::NAME,
                key.clone(),
                kernel("probe_exact"),
            )],
        ),
        &arena,
        &descriptor,
        ProbeArgs {
            data: view(&serving_buffer),
            count: WORDS as u32,
            value: CAPTURED,
        },
    );
    let serving = Captured::new(serving_template, serving_nodes, vec![kernel("probe_exact")])
        .expect("the serving generation's bucket");
    let reference = replay(&serving, &serving_buffer);
    assert_eq!(
        reference,
        vec![CAPTURED; WORDS],
        "the serving generation adds its launch argument to every word"
    );

    // The successor: a **recapture** under the same descriptor key, whose program is a different
    // kernel (`probe_wrong` adds one more), so the two graphs cannot be confused.
    let (successor_template, successor_nodes) = capture(
        &stream,
        &Generation::new(
            2,
            [KernelEntry::keyed(
                Probe::NAME,
                key.clone(),
                kernel("probe_wrong"),
            )],
        ),
        &arena,
        &descriptor,
        ProbeArgs {
            data: view(&successor_buffer),
            count: WORDS as u32,
            value: CAPTURED,
        },
    );
    let successor = Captured::new(
        successor_template,
        successor_nodes,
        vec![kernel("probe_wrong")],
    )
    .expect("the successor's bucket");
    assert_eq!(
        replay(&successor, &successor_buffer),
        vec![CAPTURED + 1; WORDS],
        "the recaptured bucket runs the successor's program: a graph that still ran the serving \
         program would leave the same words and the revert below would prove nothing"
    );

    // The revert: the serving generation serves again, and it must compute what it computed before -
    // the successor's capture mapped its own storage beside it.
    assert_eq!(
        replay(&serving, &serving_buffer),
        reference,
        "the serving generation's graph no longer replays bitwise after the successor's recapture: \
         the successor's storage overlaps an address the old graph reads"
    );

    // And the same for a **rebind**, which must not touch the serving generation at all: the
    // prepared generation gets its own executable from the shared `Graph`, so setting its parameters
    // leaves the captured one alone.
    let (walked_template, walked_nodes) = capture(
        &stream,
        &Generation::new(
            3,
            [KernelEntry::keyed(
                Probe::NAME,
                key.clone(),
                kernel("probe_equivalent"),
            )],
        ),
        &arena,
        &descriptor,
        ProbeArgs {
            data: view(&serving_buffer),
            count: WORDS as u32,
            value: WALKED,
        },
    );
    let mut params = Vec::new();
    for (_, _, launch) in &walked_nodes {
        for walked in launch {
            if walked.node.kind().expect("a node kind") == GraphNodeKind::Kernel {
                params.push(
                    walked_template
                        .kernel_node_params(&walked.node)
                        .expect("a kernel node's parameters"),
                );
            }
        }
    }
    // The runner's own path (`Captured::from_template`): the prepared generation's executable is
    // instantiated from the serving generation's `Graph`, which is what a revert relies on.
    let rebound = Captured::from_template(
        serving.shared_template(),
        walked_nodes.clone(),
        vec![kernel("probe_equivalent")],
    )
    .expect("the prepared generation's bucket");
    let applied = rebound
        .rebind_launch(Probe::NAME, 0, &params)
        .expect("the prepared generation is rebound");
    assert_eq!(
        applied,
        params.len(),
        "every kernel node of the launch is set"
    );
    assert_eq!(
        replay(&rebound, &serving_buffer),
        vec![WALKED; WORDS],
        "the rebound graph launches the prepared generation's parameters"
    );
    assert_eq!(
        replay(&serving, &serving_buffer),
        reference,
        "the rebind reached the serving generation's graph: its parameters are one executable's, \
         not the template's, which is what makes a revert possible after a program update"
    );
}

/// A commit must not end an executable a bucket it did not prepare still serves from.
///
/// The sequence is item 5's: generation 1 serves two buckets, generation 2 rebinds one of them and
/// leaves the other alone, the switchover serves generation 2, and the commit purges generation 1.
/// The rebound bucket's generation-1 capture is what generation 2 replaced, so it goes; the
/// untouched bucket's capture is the only program it has, so it stays and its batch still replays.
///
/// This is the arm that fails as *"every request in it fails after Commit"*: the table would answer
/// `select` for that bucket with nothing, and the runner would have no graph to launch.
#[test]
#[ignore = "requires a CUDA device"]
fn a_commit_keeps_the_capture_an_unprepared_bucket_still_serves_from() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let mut allocator =
        Allocator::new(&device, Arc::new(Allowance::default())).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, 1 << 20)
        .expect("reserve the graph pool");
    let stream = Stream::new(DEVICE).expect("a stream");
    let image = std::fs::read(cubin()).expect("the compiled probe kernels");
    let kernel = |symbol: &str| -> Arc<Kernel> {
        Arc::new(
            Kernel::load_cubin(&device, &image, CubinMetadata::named(symbol))
                .unwrap_or_else(|error| panic!("loading {symbol}: {error}")),
        )
    };
    let arena = nn::Arena::new(DevicePtr::new(0), 0);
    let descriptor = ProbeDescriptor {
        words: WORDS as u32,
    };
    let key = descriptor::key(&descriptor);
    // Two buckets, each with its own activation, as two shapes' plans have.
    let prepared_bucket = Bucket::decode(WORDS as u32, Variant::DecodeGreedy);
    let untouched_bucket = Bucket::decode(2 * WORDS as u32, Variant::DecodeGreedy);
    let prepared_buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("the prepared bucket's activation");
    let untouched_buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("the untouched bucket's activation");
    let view = |buffer: &Buffer| {
        TensorView::new(
            DevicePtr::new(buffer.address()),
            DType::I32,
            Shape::new(&[WORDS as u64]),
        )
    };
    let capture_bucket = |generation: u64, buffer: &Buffer, symbol: &str, value: u32| {
        let (template, nodes) = capture(
            &stream,
            &Generation::new(
                generation,
                [KernelEntry::keyed(Probe::NAME, key.clone(), kernel(symbol))],
            ),
            &arena,
            &descriptor,
            ProbeArgs {
                data: view(buffer),
                count: WORDS as u32,
                value,
            },
        );
        Captured::new(template, nodes, vec![kernel(symbol)]).expect("a captured bucket")
    };

    let mut graphs = Graphs::new();
    // Generation 1 captures both buckets.
    graphs.insert(
        prepared_bucket,
        1,
        capture_bucket(1, &prepared_buffer, "probe_exact", CAPTURED),
    );
    graphs.insert(
        untouched_bucket,
        1,
        capture_bucket(1, &untouched_buffer, "probe_exact", CAPTURED),
    );
    // Generation 2 rebinds the prepared one and prepares nothing for the other.
    let (walked_template, walked_nodes) = capture(
        &stream,
        &Generation::new(
            2,
            [KernelEntry::keyed(
                Probe::NAME,
                key.clone(),
                kernel("probe_equivalent"),
            )],
        ),
        &arena,
        &descriptor,
        ProbeArgs {
            data: view(&prepared_buffer),
            count: WORDS as u32,
            value: WALKED,
        },
    );
    // The prepared generation's executable is instantiated from a `Graph` of its own, as the runner's
    // rebind path does when no serving template is shared.
    let walked_template = Arc::new(walked_template);
    let rebound = Captured::from_template(
        Arc::clone(&walked_template),
        walked_nodes.clone(),
        vec![kernel("probe_equivalent")],
    )
    .expect("the prepared generation's bucket");
    let mut params = Vec::new();
    for (_, _, launch) in &walked_nodes {
        for walked in launch {
            if walked.node.kind().expect("a node kind") == GraphNodeKind::Kernel {
                params.push(
                    walked_template
                        .kernel_node_params(&walked.node)
                        .expect("a kernel node's parameters"),
                );
            }
        }
    }
    rebound
        .rebind_launch(Probe::NAME, 0, &params)
        .expect("the prepared generation is rebound");
    graphs.insert(prepared_bucket, 2, rebound);

    // The commit: generation 1 is purged while generation 2 serves.
    let released = graphs.purge(1, 2);
    assert!(
        released.contains(&prepared_bucket),
        "the bucket generation 2 replaced kept its generation-1 capture, which nothing serves from"
    );
    assert!(
        !released.contains(&untouched_bucket),
        "the purge released the capture an unprepared bucket still serves from: every request in \
         that bucket fails after the commit"
    );
    assert!(
        graphs.holds(untouched_bucket),
        "the table no longer holds the untouched bucket, so its arena region would be released \
         under a graph that still addresses it"
    );

    // Serving generation 2 afterwards: both buckets answer, and the untouched one answers with the
    // program it captured.
    let untouched_shape = PlanShape::decode(WORDS as u32, Variant::DecodeGreedy);
    let (bucket, served) = graphs
        .select(&untouched_shape, 2)
        .expect("the unprepared bucket still serves under the committed generation");
    assert_eq!(
        bucket, untouched_bucket,
        "the smallest bucket that accepts the shape serves"
    );
    // SAFETY: the fill lies inside the allocation the served graph addresses.
    unsafe { memset_d32(untouched_buffer.address(), 0, WORDS, &stream) }
        .expect("zero the activation");
    served.launch(&stream).expect("the served graph replays");
    assert_eq!(
        words(&stream, untouched_buffer.address()),
        vec![CAPTURED; WORDS],
        "the bucket the update did not prepare still runs the program it captured"
    );

    let prepared_shape = PlanShape::decode(WORDS as u32, Variant::DecodeGreedy);
    let (bucket, served) = graphs
        .select(&prepared_shape, 2)
        .expect("the prepared bucket serves under the committed generation");
    assert_eq!(bucket, prepared_bucket);
    // SAFETY: as above, for the prepared bucket's allocation.
    unsafe { memset_d32(prepared_buffer.address(), 0, WORDS, &stream) }
        .expect("zero the activation");
    served.launch(&stream).expect("the rebound graph replays");
    assert_eq!(
        words(&stream, prepared_buffer.address()),
        vec![WALKED; WORDS],
        "after the commit the prepared bucket runs the generation it switched over to"
    );
}

/// A reverted update releases its whole capture set - the case a commit's rule cannot see.
///
/// After a revert the failed generation sits **above** the serving one, so nothing supersedes it and
/// a rule written for a commit releases nothing: the executables, their nodes and the arena regions
/// they address stay for the process's life, one set per reverted update, until the graph pool
/// refuses a capture. The purge takes the serving generation into account, and this measures the
/// object lifetime that follows: generation 2's `Graph` has no holder once the purge ran, and
/// generation 1's still has.
#[test]
#[ignore = "requires a CUDA device"]
fn a_revert_releases_the_whole_capture_set_of_the_failed_generation() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let mut allocator =
        Allocator::new(&device, Arc::new(Allowance::default())).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, 1 << 20)
        .expect("reserve the graph pool");
    let stream = Stream::new(DEVICE).expect("a stream");
    let image = std::fs::read(cubin()).expect("the compiled probe kernels");
    let kernel = |symbol: &str| -> Arc<Kernel> {
        Arc::new(
            Kernel::load_cubin(&device, &image, CubinMetadata::named(symbol))
                .unwrap_or_else(|error| panic!("loading {symbol}: {error}")),
        )
    };
    let arena = nn::Arena::new(DevicePtr::new(0), 0);
    let descriptor = ProbeDescriptor {
        words: WORDS as u32,
    };
    let key = descriptor::key(&descriptor);
    let bucket = Bucket::decode(WORDS as u32, Variant::DecodeGreedy);
    let buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("the bucket's activation");
    let data = TensorView::new(
        DevicePtr::new(buffer.address()),
        DType::I32,
        Shape::new(&[WORDS as u64]),
    );
    // One capture per generation, each kept as an `Arc` so its lifetime is observable.
    let capture_generation = |generation: u64, symbol: &str, value: u32| {
        let (template, nodes) = capture(
            &stream,
            &Generation::new(
                generation,
                [KernelEntry::keyed(Probe::NAME, key.clone(), kernel(symbol))],
            ),
            &arena,
            &descriptor,
            ProbeArgs {
                data,
                count: WORDS as u32,
                value,
            },
        );
        let template = Arc::new(template);
        let captured = Captured::from_template(Arc::clone(&template), nodes, vec![kernel(symbol)])
            .expect("a captured bucket");
        (template, captured)
    };
    let (serving_graph, serving) = capture_generation(1, "probe_exact", CAPTURED);
    let (failed_graph, failed) = capture_generation(2, "probe_wrong", CAPTURED);
    let mut graphs = Graphs::new();
    graphs.insert(bucket, 1, serving);
    graphs.insert(bucket, 2, failed);

    // Served generation 1, reverted from 2: the purge names the failed generation and the one that
    // serves.
    let released = graphs.purge(2, 1);
    assert_eq!(
        released,
        vec![bucket],
        "the purge did not name the bucket whose capture it released"
    );
    assert!(
        graphs.captured(bucket, 2).is_none(),
        "the failed generation's capture is still in the table: every reverted update leaks a \
         whole capture set until the graph pool refuses a capture"
    );
    assert!(
        graphs.captured(bucket, 1).is_some(),
        "the purge took the serving generation's capture"
    );
    assert!(
        Arc::downgrade(&failed_graph).upgrade().is_none(),
        "generation 2's graph still has a holder after the purge: its executable, nodes and arena \
         regions outlive the update that was abandoned"
    );
    assert!(
        Arc::downgrade(&serving_graph).upgrade().is_some(),
        "generation 1's graph lost its holder, which is what a purge must never do"
    );
}

/// A program that orders its work differently is a different structure, and the rebind condition
/// must decline it.
///
/// The case is programmatic dependent launch, which the slice's attention kernels use: a kernel
/// launched with PDL may start at its predecessor's trigger point rather than after it finishes, and
/// the graph records that as a `Programmatic` edge instead of a `Completion` one. A rebind
/// instantiates a fresh executable from the **serving** graph's template and sets node parameters on
/// it, so the prepared generation would keep the serving edges - a kernel that no longer synchronizes
/// itself with its predecessor would then read what the predecessor has not written. So the edge
/// kinds are part of 4.5's structure, and this test is the one that says so on a device.
///
/// Two launches in one capture, because the interesting edge is the one between them: the first
/// launch's node is a root of the graph and has no incoming edge to read.
#[test]
#[ignore = "requires a CUDA device"]
fn a_programmatic_edge_is_a_different_structure() {
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
    let kernel = |pdl: bool| -> Arc<Kernel> {
        let mut metadata = CubinMetadata::named("probe_exact");
        // What a Triton module states for a kernel that launches with PDL: the attribute the
        // driver turns into a programmatic dependency on the preceding kernel of the stream.
        metadata.launch_pdl = pdl;
        Arc::new(
            Kernel::load_cubin(&device, &image, metadata)
                .unwrap_or_else(|error| panic!("loading probe_exact: {error}")),
        )
    };
    let descriptor = ProbeDescriptor {
        words: WORDS as u32,
    };
    let key = descriptor::key(&descriptor);
    let arena = nn::Arena::new(DevicePtr::new(0), 0);
    let data = TensorView::new(
        DevicePtr::new(buffer.address()),
        DType::I32,
        Shape::new(&[WORDS as u64]),
    );

    // One capture of two launches of the same operator, the second stating `pdl`.
    let capture_two = |pdl: bool, number: u64| -> (platform::Graph, worker::captured::Nodes) {
        let generation = Generation::new(
            number,
            [KernelEntry::keyed(Probe::NAME, key.clone(), kernel(pdl))],
        );
        let capture = Rc::new(RefCell::new(
            platform::Capture::begin(&stream, CaptureMode::Relaxed).expect("a capture begins"),
        ));
        let mut recording = Recording::new(Rc::clone(&capture));
        ops::capture::set_sink(Some(&mut recording));
        let forward = Forward::new(&stream, &generation, &arena);
        for _ in 0..2 {
            forward
                .operate::<Probe>(
                    &descriptor,
                    ProbeArgs {
                        data,
                        count: WORDS as u32,
                        value: CAPTURED,
                    },
                )
                .expect("the probe launches");
        }
        ops::capture::set_sink(None);
        if let Some(failure) = recording.failure() {
            panic!("the capture failed while recording a launch: {failure}");
        }
        let nodes = recording.into_nodes();
        assert_eq!(nodes.len(), 2, "two launches were recorded");
        let template = Rc::try_unwrap(capture)
            .unwrap_or_else(|_| panic!("the capture is still shared by its recorder"))
            .into_inner()
            .end()
            .expect("the capture ends");
        (template, nodes)
    };

    let (template, completion) = capture_two(false, 1);
    let (_, programmatic) = capture_two(true, 2);
    // The edge the second launch states: one dependency on the first launch's node, and its kind is
    // the whole point of the test.
    let kind_of = |nodes: &worker::captured::Nodes| nodes[1].2[0].incoming.clone();
    assert_eq!(
        kind_of(&completion),
        vec![platform::EdgeKind::Completion],
        "a plain launch orders the second kernel after the first one has finished"
    );
    assert_eq!(
        kind_of(&programmatic),
        vec![platform::EdgeKind::Programmatic],
        "a launch with PDL states a programmatic dependency, not a completion one"
    );

    let captured =
        Captured::new(template, completion, vec![kernel(false)]).expect("a captured bucket");
    assert!(
        captured
            .same_structure(captured.nodes())
            .expect("the structure compares"),
        "a capture is its own structure"
    );
    let reason = captured
        .structure_mismatch(&programmatic)
        .expect("the structures compare")
        .expect("a programmatic edge is a different structure, so the bucket must be recaptured");
    assert!(
        reason.contains("Programmatic"),
        "the refusal did not name the edge kind that decided it: {reason}"
    );
    assert!(
        !captured
            .same_structure(&programmatic)
            .expect("the structures compare"),
        "a bucket was rebound onto a structure whose edges order the work differently"
    );
}
