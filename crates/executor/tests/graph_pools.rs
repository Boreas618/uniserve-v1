//! Design section 12's open question: can a new and an old generation of
//! programs share one graph pool?
//!
//! The design leaves open whether the graphs of two generations - the serving
//! one and the one a precopy filled - can live in one device's graph pool at
//! the same time, or whether each generation needs a pool of its own. This
//! probe answers it by measurement on one GPU: every activation the graphs
//! address is mapped before the first capture and never reallocated, two
//! generations' graphs are captured over those same buffers, a third is
//! captured while both are alive, and the three are replayed alternately with
//! each generation's own arithmetic visible in the result.
//!
//! What it measures, and why each part is the question:
//!
//! - **capture allocates nothing**: `Allocator::mapped_bytes` is unchanged
//!   across all three captures, which is design 4.5's "no allocation during a
//!   step" applied to capture;
//! - **both generations coexist**: a second capture succeeds while the first
//!   generation's executable is instantiated and its buffers are mapped;
//! - **a replay is the generation it was captured from**: generation 0 adds
//!   `value` once, generation 1 adds it twice, and replays interleave, so a
//!   re-pointed or shared node would show the wrong arithmetic;
//! - **a third capture while two are live**: the pool holds three executables'
//!   worth of graphs without a second reservation.
//!
//! Run it on a GPU host, alone on the device:
//!
//! ```text
//! cargo test -p executor --test graph_pools -- --ignored --test-threads=1
//! ```
//!
//! It needs `nvcc` on `PATH`, and `UNISERVE_PROBE_ARCH` names the architecture
//! it compiles for (default `sm_100a`).

use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use platform::{
    Allocator, Budget, Buffer, Capture, CaptureMode, Device, DeviceId, Function, GraphExec,
    LaunchConfig, Library, PinnedBuffer, PoolKind, Result, Stream, memcpy_async, memset_d32,
};

const DEVICE: DeviceId = DeviceId(0);
const WORDS: usize = 32;
const BYTES: usize = WORDS * 4;
const RESERVATION: usize = 1 << 20;

/// A budget that records the pool's cumulative total and never refuses: the
/// probe is about the pool's addresses, not about running out of device memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// Compiles the probe's two generations of one kernel into a cubin.
fn cubin() -> PathBuf {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generations.cu");
    let output = std::env::temp_dir().join(format!("generations-{}.cubin", std::process::id()));
    let arch = std::env::var("UNISERVE_PROBE_ARCH").unwrap_or_else(|_| "sm_100a".to_owned());
    let status = Command::new("nvcc")
        .arg("-cubin")
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&output)
        .arg(&source)
        .status()
        .expect("this probe compiles its kernel with nvcc, which must be on PATH");
    assert!(status.success(), "nvcc failed on {}", source.display());
    output
}

/// Launches `kernel` inside one capture and instantiates the graph it records.
///
/// `data` and `value` must outlive the capture: the driver records their
/// addresses as the node's arguments, and the graph is replayed from them.
fn capture(
    stream: &Stream,
    kernel: &Function,
    data: &mut u64,
    value: &mut u32,
) -> GraphExec {
    let mut count = WORDS as i32;
    let mut parameters: [*mut c_void; 3] = [
        (data as *mut u64).cast(),
        (&raw mut count).cast(),
        (value as *mut u32).cast(),
    ];
    let mut recording = Capture::begin(stream, CaptureMode::Relaxed).expect("begin a capture");
    kernel
        .launch_ex(
            stream,
            &LaunchConfig::new([1, 1, 1], [WORDS as u32, 1, 1]),
            &mut parameters,
        )
        .expect("launch inside the capture");
    recording.record().expect("record the launch's node");
    recording.end().expect("end the capture").instantiate().expect(
        "instantiate the graph - failing here means two generations cannot share the pool",
    )
}

/// Replays `graph` and answers the buffer's words.
fn replay(stream: &Stream, graph: &GraphExec, data: u64) -> Vec<u32> {
    graph.launch(stream).expect("replay");
    let host = PinnedBuffer::new(DEVICE, BYTES).expect("pinned staging");
    // SAFETY: the copy reads `BYTES` from the allocation into a pinned buffer
    // of the same size, and the stream that wrote the allocation is
    // synchronized before the buffer is read.
    unsafe { memcpy_async(host.address(), data, BYTES, stream) }.expect("copy back");
    stream.synchronize().expect("synchronize");
    // SAFETY: the copy completed and the buffer holds `BYTES` readable bytes.
    unsafe { host.as_slice() }
        .chunks_exact(4)
        .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_generations_share_one_graph_pool() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let allowance = Arc::new(Allowance::default());
    let mut allocator = Allocator::new(&device, allowance).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, RESERVATION)
        .expect("reserve the graph pool");
    let buffer: Buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("one activation buffer");
    let data = buffer.address();
    let stream = Stream::new(DEVICE).expect("a stream");
    let image = std::fs::read(cubin()).expect("the compiled probe kernel");
    let library = Library::load(&device, &image).expect("load the probe kernel");
    let first = library.kernel("generation_0").expect("generation_0");
    let second = library.kernel("generation_1").expect("generation_1");

    // SAFETY: the fill lies inside the allocation.
    unsafe { memset_d32(data, 0, WORDS, &stream) }.expect("zero the buffer");

    let mapped_before = allocator.mapped_bytes(PoolKind::Graph);
    let mut address = data;
    let mut value = 1u32;
    let serving = capture(&stream, &first, &mut address, &mut value);
    let precopied = capture(&stream, &second, &mut address, &mut value);
    let mapped_after_two = allocator.mapped_bytes(PoolKind::Graph);

    assert_eq!(
        mapped_before, mapped_after_two,
        "a capture mapped storage: the pool grew from {mapped_before} to {mapped_after_two} bytes"
    );
    assert_eq!(buffer.address(), address, "the activation buffer moved");

    assert_eq!(
        replay(&stream, &serving, data),
        vec![1; WORDS],
        "the first generation's graph does not add value once"
    );
    assert_eq!(
        replay(&stream, &precopied, data),
        vec![3; WORDS],
        "the second generation's graph does not add value twice"
    );

    let mut value = 10u32;
    let late = capture(&stream, &first, &mut address, &mut value);
    let mapped_after_three = allocator.mapped_bytes(PoolKind::Graph);
    assert_eq!(
        mapped_before, mapped_after_three,
        "the third capture mapped storage: {mapped_before} -> {mapped_after_three} bytes"
    );

    assert_eq!(
        replay(&stream, &serving, data),
        vec![4; WORDS],
        "the first generation's graph no longer replays after another capture"
    );
    assert_eq!(
        replay(&stream, &late, data),
        vec![14; WORDS],
        "the third graph does not add ten"
    );
    assert_eq!(
        replay(&stream, &precopied, data),
        vec![16; WORDS],
        "the second generation's graph no longer replays after two more captures"
    );

    // Dropping every executable hands the block back without unmapping it; the
    // pool's figure is what the budget was charged, so it stays until `free`.
    drop(serving);
    drop(precopied);
    drop(late);
    allocator.free(buffer);
    assert_eq!(
        allocator.mapped_bytes(PoolKind::Graph),
        0,
        "a freed buffer still counts against the graph pool"
    );
}
