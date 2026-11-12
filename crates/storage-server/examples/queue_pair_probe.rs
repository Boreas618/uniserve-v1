//! Measures whether a queue pair on the step path changes the step time of a
//! replayed captured graph (`specs/buffer-pool.md` §19, D0b).
//!
//! An engine loop replays one captured graph per step and awaits its stream
//! event. With a queue pair, each step also does the engine's whole contact
//! with the buffer pool (§10, F1): it reads the completions that arrived and
//! the capacity registers, writes the step's requests into the submission
//! queue and publishes `completed`. A second process runs the buffer pool's
//! loop with no buffer pool logic: it polls the submission queue, answers
//! every request with one completion, and backs off to a bounded sleep when
//! the queue is empty. Three modes:
//!
//! - `baseline`: the graph alone, no queue pair and no second process.
//! - `live`: the queue pair with the buffer pool's process running.
//! - `stopped`: the same, with the buffer pool's process stopped (`SIGSTOP`)
//!   after the warm-up steps and continued after the measured ones.
//!
//! ```text
//! cargo run --release -p storage-server --example queue_pair_probe -- \
//!     --mode live --device 0 --nodes 512 --node-bytes 16777216 \
//!     --steps 20000 --warmup 2000 --requests 16 --work DIR --samples FILE
//! ```
//!
//! The graph is `--nodes` launches of a kernel that fills `--node-bytes`. The
//! kernel is compiled to PTX by `nvcc` (`--nvcc`) into `--work`. The process
//! prints one JSON line and, with `--samples`, writes every measured step's
//! nanoseconds, one per line.

use std::ffi::c_void;
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use platform::{
    Capture, CaptureMode, Device, DeviceAllocation, DeviceId, Event, EventFlags, LaunchConfig,
    Library,
};
use storage::QueuePair;

const KERNEL: &str = r#"
extern "C" __global__ void fill(unsigned int* words, unsigned long long count, unsigned int seed) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        words[i] = seed ^ (unsigned int)i;
    }
}
"#;

/// A ring's bytes, and the bytes of one request and of its completion.
const RING_BYTES: usize = 1 << 20;
const REQUEST_BYTES: usize = 64;
const COMPLETION_BYTES: usize = 32;

/// Register 0 is `completed`, written by the engine; registers 1 to 3 are one
/// region's capacity words, written by the buffer pool.
const REGISTERS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Baseline,
    Live,
    Stopped,
}

struct Options {
    mode: Mode,
    device: DeviceId,
    nodes: u32,
    node_bytes: usize,
    steps: usize,
    warmup: usize,
    requests: usize,
    work: PathBuf,
    nvcc: String,
    samples: Option<PathBuf>,
    /// Set for the buffer pool's process: the inherited queue pair.
    pool: Option<i32>,
}

fn options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut options = Options {
        mode: Mode::Live,
        device: DeviceId(0),
        nodes: 512,
        node_bytes: 16 << 20,
        steps: 20_000,
        warmup: 2_000,
        requests: 16,
        work: std::env::temp_dir(),
        nvcc: "nvcc".into(),
        samples: None,
        pool: None,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--mode" => {
                options.mode = match value.as_str() {
                    "baseline" => Mode::Baseline,
                    "live" => Mode::Live,
                    "stopped" => Mode::Stopped,
                    other => return Err(format!("unknown mode {other}").into()),
                }
            }
            "--device" => options.device = DeviceId(value.parse()?),
            "--nodes" => options.nodes = value.parse()?,
            "--node-bytes" => options.node_bytes = value.parse()?,
            "--steps" => options.steps = value.parse()?,
            "--warmup" => options.warmup = value.parse()?,
            "--requests" => options.requests = value.parse()?,
            "--work" => options.work = value.into(),
            "--nvcc" => options.nvcc = value,
            "--samples" => options.samples = Some(value.into()),
            "--pool-descriptor" => options.pool = Some(value.parse()?),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    Ok(options)
}

/// The buffer pool's loop without its logic: one completion per request, the
/// capacity registers rewritten, `completed` read, and a bounded sleep once
/// the submission queue has stayed empty.
fn pool(descriptor: i32) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: the engine's process passed this descriptor to this process alone.
    let mut queue = QueuePair::open(unsafe { OwnedFd::from_raw_fd(descriptor) }, false)?;
    let mut request = Vec::new();
    let mut pending: Option<[u8; COMPLETION_BYTES]> = None;
    let mut served = 0u64;
    let mut idle = 0u32;

    loop {
        let mut progressed = false;
        loop {
            // A completion the queue had no room for is written before the next
            // request is taken, so every request gets exactly one.
            if let Some(completion) = pending {
                if !queue.complete(&completion)? {
                    break;
                }
                pending = None;
                served += 1;
                progressed = true;
            }
            if !queue.next_request(&mut request)? {
                break;
            }
            // The last request tells the loop to end.
            if request.first() == Some(&0xff) {
                return Ok(());
            }
            let mut completion = [0u8; COMPLETION_BYTES];
            completion[..8].copy_from_slice(&request[..8]);
            pending = Some(completion);
        }

        let completed = queue.register(0).load(Ordering::Acquire);
        for word in 1..REGISTERS {
            queue
                .register(word)
                .store(served + completed, Ordering::Release);
        }

        if progressed {
            idle = 0;
        } else {
            idle += 1;
            if idle < 1_000 {
                std::hint::spin_loop();
            } else {
                // At most a millisecond without looking.
                std::thread::sleep(Duration::from_micros(
                    u64::from(idle - 1_000).clamp(50, 1_000),
                ));
            }
        }
    }
}

/// Compiles the kernel to PTX in `work` and returns its text.
fn compile(nvcc: &str, work: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(work)?;
    let source = work.join(format!("queue_pair_probe.{}.cu", std::process::id()));
    let ptx = source.with_extension("ptx");
    std::fs::write(&source, KERNEL)?;
    let status = Command::new(nvcc)
        .args(["-ptx", "-O3", "-o"])
        .arg(&ptx)
        .arg(&source)
        .status()?;
    assert!(status.success(), "nvcc failed");
    Ok(std::fs::read(ptx)?)
}

/// Sends `signal` to the buffer pool's process.
fn signal(child: &std::process::Child, signal: i32) {
    // SAFETY: a signal to a child this process started and has not reaped.
    let sent = unsafe { libc::kill(child.id() as i32, signal) };
    assert_eq!(sent, 0, "signalling the buffer pool's process");
}

fn engine(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(options.device)?;
    let stream = device.stream()?;
    let library = Library::load(&device, &compile(&options.nvcc, &options.work)?)?;
    let fill = library.kernel("fill")?;

    // The graph: `nodes` kernel launches over one buffer.
    let buffer = DeviceAllocation::linear(options.device, options.node_bytes)?;
    let mut words = buffer.address();
    let mut count = (options.node_bytes / 4) as u64;
    let capture = Capture::begin(&stream, CaptureMode::Relaxed)?;
    for node in 0..options.nodes {
        let mut seed = node;
        fill.launch_ex(
            &stream,
            &LaunchConfig::new([1024, 1, 1], [256, 1, 1]),
            &mut [
                (&raw mut words).cast::<c_void>(),
                (&raw mut count).cast(),
                (&raw mut seed).cast(),
            ],
        )?;
    }
    let graph = capture.end()?;
    let executable = graph.instantiate()?;
    let event = Event::new(EventFlags::default());

    // The queue pair and the buffer pool's process.
    let mut queue = None;
    let mut child = None;
    if options.mode != Mode::Baseline {
        let created = QueuePair::create(RING_BYTES, REGISTERS)?;
        let descriptor = created.descriptor().try_clone_to_owned()?;
        // SAFETY: clearing close-on-exec on a descriptor this process owns.
        let cleared = unsafe { libc::fcntl(descriptor.as_raw_fd(), libc::F_SETFD, 0) };
        assert_eq!(cleared, 0, "clearing close-on-exec");
        child = Some(
            Command::new(std::env::current_exe()?)
                .args(["--pool-descriptor", &descriptor.as_raw_fd().to_string()])
                .spawn()?,
        );
        queue = Some(QueuePair::open(descriptor, true)?);
    }

    let mut samples = Vec::with_capacity(options.steps);
    let mut completion = Vec::new();
    let mut request = [0u8; REQUEST_BYTES];
    let (mut submitted, mut refused, mut completions, mut capacity) = (0u64, 0u64, 0u64, 0u64);
    let mut refused_from = None;

    for step in 0..options.warmup + options.steps {
        if step == options.warmup && options.mode == Mode::Stopped {
            signal(
                child.as_ref().ok_or("no buffer pool process to stop")?,
                libc::SIGSTOP,
            );
        }
        let began = Instant::now();

        // What arrived since the last step.
        if let Some(queue) = &mut queue {
            while queue.next_completion(&mut completion)? {
                completions += 1;
            }
            for word in 1..REGISTERS {
                capacity = capacity.wrapping_add(queue.register(word).load(Ordering::Acquire));
            }
        }

        executable.launch(&stream)?;
        event.record(&stream)?;

        // The step's requests; one the queue has no room for is kept by the
        // caller, which here only counts it.
        if let Some(queue) = &mut queue {
            for _ in 0..options.requests {
                request[..8].copy_from_slice(&submitted.to_le_bytes());
                request[0] &= 0x7f;
                if queue.submit(&request)? {
                    submitted += 1;
                } else {
                    refused += 1;
                    refused_from.get_or_insert(step - options.warmup.min(step));
                }
            }
        }

        event.synchronize()?;
        if let Some(queue) = &queue {
            queue.register(0).store(step as u64, Ordering::Release);
        }

        if step >= options.warmup {
            samples.push(began.elapsed().as_nanos() as u64);
        }
    }

    // Let the buffer pool's process answer what is outstanding, then end it.
    let mut drained_after = None;
    if let (Some(queue), Some(mut child)) = (queue.as_mut(), child) {
        if options.mode == Mode::Stopped {
            signal(&child, libc::SIGCONT);
        }
        let resumed = Instant::now();
        while completions < submitted && resumed.elapsed() < Duration::from_secs(10) {
            while queue.next_completion(&mut completion)? {
                completions += 1;
            }
        }
        drained_after = Some(resumed.elapsed().as_secs_f64());
        request[0] = 0xff;
        while !queue.submit(&request)? {}
        child.wait()?;
    }

    if let Some(path) = &options.samples {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for sample in &samples {
            writeln!(file, "{sample}")?;
        }
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let at = |quantile: f64| {
        sorted[((sorted.len() - 1) as f64 * quantile).round() as usize] as f64 / 1e3
    };
    let mean = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1e3;
    let mode = match options.mode {
        Mode::Baseline => "baseline",
        Mode::Live => "live",
        Mode::Stopped => "stopped",
    };
    println!(
        "{{\"mode\":\"{mode}\",\"nodes\":{},\"node_bytes\":{},\"steps\":{},\"requests_per_step\":{},\
         \"step_us\":{{\"mean\":{mean:.2},\"min\":{:.2},\"p50\":{:.2},\"p90\":{:.2},\"p99\":{:.2},\"p999\":{:.2},\"max\":{:.2}}},\
         \"submitted\":{submitted},\"refused\":{refused},\"refused_from_step\":{},\"completions\":{completions},\
         \"drained_s\":{},\"capacity_checksum\":{capacity}}}",
        options.nodes,
        options.node_bytes,
        samples.len(),
        if options.mode == Mode::Baseline {
            0
        } else {
            options.requests
        },
        at(0.0),
        at(0.5),
        at(0.9),
        at(0.99),
        at(0.999),
        at(1.0),
        refused_from.map_or("null".into(), |step: usize| step.to_string()),
        drained_after.map_or("null".into(), |seconds| format!("{seconds:.6}")),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = options()?;
    match options.pool {
        Some(descriptor) => pool(descriptor),
        None => engine(&options),
    }
}
