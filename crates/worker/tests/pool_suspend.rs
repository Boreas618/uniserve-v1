//! Design 8's exchange transport and the world switch: a pool's storage is unmapped while its
//! virtual range stays reserved, so a captured graph over it stays valid, and a restore maps
//! storage at the same addresses again - refusing a layout the captured graphs do not address.
//!
//! Both cases need a real device: the question is what the driver does with a graph whose
//! pointers were unmapped and remapped, which no host-side double can answer.

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use platform::{
    Allocator, Budget, Capture, CaptureMode, Device, DeviceId, Function, GraphExec, LaunchConfig,
    Library, PinnedBuffer, PoolKind, Result, Stream, memcpy_async, memset_d32,
};
use worker::arena::{Pools, StreamId};
use worker::plan::Bucket;

const DEVICE: DeviceId = DeviceId(0);
const WORDS: usize = 32;
const BYTES: usize = WORDS * 4;
const RESERVATION: u64 = 1 << 20;

/// A budget that records the device's cumulative total and never refuses: these cases are about
/// a pool's addresses, not about running out of device memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// Compiles the fixture's one kernel into a cubin.
fn cubin() -> PathBuf {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generations.cu");
    let output = std::env::temp_dir().join(format!("suspend-{}.cubin", std::process::id()));
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

/// One device's pools, with its graph storage reserved and no owner carved from it.
fn pools() -> (Pools, Arc<Allowance>) {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let allowance = Arc::new(Allowance::default());
    let totals = Arc::new(Pools::totals_of(u64::MAX / 2));
    let allocator = Allocator::new(&device, Arc::clone(&allowance) as Arc<dyn Budget>)
        .expect("an allocator over the device");
    let pools = Pools::new(allocator, totals, RESERVATION, 0, 0, 1).expect("the device's pools");
    (pools, allowance)
}

/// Adds `value` to every word of `data` inside one capture, and answers the graph it recorded.
fn capture(stream: &Stream, kernel: &Function, data: &mut u64, value: &mut u32) -> GraphExec {
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
    recording
        .end()
        .expect("end the capture")
        .instantiate()
        .expect("instantiate the graph")
}

/// Replays `graph` and answers the words it left in `data`.
fn replay(stream: &Stream, graph: &GraphExec, data: u64) -> Vec<u32> {
    graph.launch(stream).expect("replay");
    let host = PinnedBuffer::new(DEVICE, BYTES).expect("pinned staging");
    // SAFETY: the copy reads `BYTES` from the allocation into a pinned buffer of the same size,
    // and the stream that wrote the allocation is synchronized before the buffer is read.
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
fn a_suspended_pool_restores_at_the_same_addresses_and_its_graph_replays() {
    let (mut pools, _allowance) = pools();
    let stream = Stream::new(DEVICE).expect("a stream");
    let image = std::fs::read(cubin()).expect("the compiled kernel");
    let device = Device::new(DEVICE).expect("a CUDA device");
    let library = Library::load(&device, &image).expect("load the kernel");
    let kernel = library.kernel("generation_0").expect("generation_0");

    let range = pools.base(StreamId(0)).expect("the stream's range");
    let data = range.address();
    // SAFETY: the fill lies inside the pool's range, which is `RESERVATION` bytes and is mapped.
    unsafe { memset_d32(data, 7, WORDS, &stream) }.expect("fill the range");
    stream.synchronize().expect("synchronize");

    let before = pools.layout();
    let mut address = data;
    let mut value = 1u32;
    let graph = capture(&stream, &kernel, &mut address, &mut value);

    pools.suspend().expect("unmap the pool's storage");
    assert_eq!(
        pools.mapped_bytes(PoolKind::Graph),
        0,
        "a suspended pool still reports mapped storage"
    );
    pools.restore().expect("map the storage again");
    assert_eq!(
        before,
        pools.layout(),
        "the restored pool holds a different layout from the one the graph addresses"
    );
    assert_eq!(
        pools.mapped_bytes(PoolKind::Graph) >= BYTES as u64,
        true,
        "a restored pool maps less than one activation"
    );

    // The graph was captured over this range; a restore that copied the saved contents back and
    // mapped the same addresses leaves it addressing live storage, and the replay shows it.
    let words = replay(&stream, &graph, data);
    assert!(
        words.iter().all(|word| *word == 8),
        "the replay of a graph over a restored range left {words:?}, not 8 per word"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_restore_whose_layout_the_graphs_do_not_address_is_refused() {
    let (mut pools, _allowance) = pools();
    let stream = Stream::new(DEVICE).expect("a stream");
    let bucket = Bucket::decode(1, worker::plan::Variant::DecodeGreedy);

    pools.base(StreamId(0)).expect("the stream's range");
    pools.record(StreamId(0), bucket, 64).expect("a plan");
    let before = pools.layout();

    pools.suspend().expect("unmap the pool's storage");
    // A second stream's range appears between the two calls - what a world switch that opened a
    // new generation's stream inside the window would do.
    pools.base(StreamId(1)).expect("a second stream's range");
    let error = pools
        .restore()
        .expect_err("a restore at a layout the graphs do not address");
    let text = error.to_string();
    assert!(
        text.contains("stream") && text.contains('1'),
        "the refusal does not name the range that appeared: {text}"
    );
    assert_eq!(
        before.len() + 1,
        pools.layout().len(),
        "the refusal did not leave the restored pools reported as they are"
    );

    // A pool that is not suspended cannot be restored: the layout it would have to honour is gone.
    assert!(
        pools.restore().is_err(),
        "a second restore invented a layout to honour"
    );
}
