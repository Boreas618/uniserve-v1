//! Probe: does a capture attribute the graph nodes each launch added?
//!
//! 4.5's rebind maps an operator to a **range of the template's nodes**, which it reads from the
//! per-launch table a capture records: the sink is told about every launch, and after each one the
//! capture is asked which nodes the work just issued added (`platform::Capture::record_walk`). A
//! served capture in this tree named 23 nodes for 733 launches, so rebind cannot attribute the work
//! it must rewrite, and the question is which capture mode makes attribution work.
//!
//! `record_walk` walks from the frontier `cuStreamGetCaptureInfo` reports. In
//! `CaptureMode::Relaxed` - the mode the executor captures in, because another device's executor in
//! one process may issue unsafe calls meanwhile - the driver need not report a usable frontier, so
//! every call after the first finds nothing new and the table is nearly all empty. In
//! `CaptureMode::ThreadLocal` the frontier is the capturing thread's and can be walked.
//!
//! This probe captures the same three launches in both modes and prints, per mode, how many nodes
//! each launch's record returned:
//!
//! ```text
//! probe(capture-frontier): mode=Relaxed per_launch=[1, 0, 0] graph_nodes=3
//! probe(capture-frontier): mode=ThreadLocal per_launch=[1, 1, 1] graph_nodes=3
//! ```
//!
//! **A pass is: at least one mode attributes a node to every launch**, because a capture that
//! attributes nothing cannot be rebound and 4.5 needs a mode that can. Which mode is the answer is
//! what the two lines print, and a run that shows neither mode attributing says the mechanism needs
//! another way to map an operator to its nodes - a finding, not a test failure to be worked around.
//!
//! Run on one GPU host, alone on the device:
//!
//! ```text
//! cargo test -p executor --test capture_frontier -- --ignored --nocapture
//! ```
//!
//! It needs `nvcc` on `PATH`; `UNISERVE_PROBE_ARCH` names the architecture it compiles for (default
//! `sm_100a`).

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use platform::{
    Allocator, Budget, Capture, CaptureMode, Device, DeviceId, Function, LaunchConfig, Library,
    PoolKind, Result, Stream,
};

const DEVICE: DeviceId = DeviceId(0);
const WORDS: usize = 32;
const BYTES: usize = WORDS * 8;
const RESERVATION: usize = 1 << 20;

/// A budget that records the pool's total and never refuses: the probe is about a capture's
/// attribution, not about running out of device memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// Compiles the probe's kernel into a cubin: the same fixture the graph-pool probe uses, because
/// the question is about the capture and not about what the kernel computes.
fn cubin() -> PathBuf {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generations.cu");
    let output = std::env::temp_dir().join(format!("frontier-{}.cubin", std::process::id()));
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

/// Captures three launches of `kernel` in `mode` and answers, per launch, how many nodes the
/// capture attributed to it, with the graph's own node count beside it.
///
/// `data` must outlive the capture: the driver records its address as the node's argument.
fn attributed(stream: &Stream, kernel: &Function, data: &mut u64, mode: CaptureMode) -> (Vec<usize>, usize) {
    let mut count = WORDS as i32;
    let mut value = 1u32;
    let mut parameters: [*mut c_void; 3] = [
        (data as *mut u64).cast(),
        (&raw mut count).cast(),
        (&raw mut value).cast(),
    ];
    let mut recording = Capture::begin(stream, mode).expect("begin a capture");
    let mut per_launch = Vec::with_capacity(3);
    for _ in 0..3 {
        kernel
            .launch_ex(
                stream,
                &LaunchConfig::new([1, 1, 1], [WORDS as u32, 1, 1]),
                &mut parameters,
            )
            .expect("launch inside the capture");
        per_launch.push(
            recording
                .record()
                .expect("record the launch's nodes")
                .len(),
        );
    }
    let graph = recording.end().expect("end the capture");
    (per_launch, graph.nodes().expect("the graph's nodes").len())
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_capture_attributes_the_nodes_each_launch_added() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let stream = Stream::new(DEVICE).expect("a stream");
    let image = std::fs::read(cubin()).expect("the compiled probe kernel");
    let library = Library::load(&device, &image).expect("load the probe kernel");
    let kernel: Function = library.kernel("generation_0").expect("the fixture's kernel");
    let allowance = Arc::new(Allowance::default());
    let mut allocator = Allocator::new(&device, allowance).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, RESERVATION)
        .expect("reserve the graph pool");
    let buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("the probe's activation buffer");
    let mut data = buffer.address();

    // Both orders are run, so the second mode is not paying for the first's state.
    let mut answers = Vec::new();
    for mode in [CaptureMode::Relaxed, CaptureMode::ThreadLocal] {
        let (per_launch, graph_nodes) = attributed(&stream, &kernel, &mut data, mode);
        println!(
            "probe(capture-frontier): mode={mode:?} per_launch={per_launch:?} \
             graph_nodes={graph_nodes}"
        );
        answers.push((mode, per_launch, graph_nodes));
    }

    // 4.5's rebind sets one node's parameters per launch and cannot guess which of the graph's
    // nodes are whose, so attribution must not depend on the mode the executor happens to capture
    // in: every mode the executor may use has to name each launch's nodes.
    for (mode, per_launch, graph_nodes) in &answers {
        assert_eq!(
            *graph_nodes, 3,
            "the capture holds {graph_nodes} nodes for three launches in {mode:?}"
        );
        assert!(
            per_launch.iter().all(|&nodes| nodes > 0),
            "{mode:?} attributed {per_launch:?} nodes to the three launches: a launch of one kernel \
             adds one node, and a walk that names none of them cannot be rebound"
        );
    }
}
