//! Measures whether two processes can map one device allocation writable and
//! write disjoint ranges of it at once (`specs/buffer-pool.md` §19, D0a).
//!
//! The first process (the creator) creates a retainable allocation, exports its
//! POSIX descriptor and starts a second process (the peer) that inherits the
//! descriptor and maps the same physical storage read-write in a CUDA context
//! of its own. Each then writes its own half with a kernel for a fixed time,
//! reads every write back on the device, and at the end reads its half back to
//! the host. Three modes separate the cost of sharing from the cost of two
//! contexts on one device:
//!
//! - `solo`: one process, one allocation: the rate of one writer alone.
//! - `separate`: two processes, each its own allocation, at once: the control.
//! - `shared`: two processes, the two halves of one allocation, at once.
//!
//! Sharing costs nothing when the rates of `shared` equal those of `separate`.
//!
//! ```text
//! cargo run --release -p platform --example shared_mapping_probe -- \
//!     --mode shared --device 0 --half-bytes 2147483648 --seconds 10 --work DIR
//! ```
//!
//! The kernels are compiled to PTX by `nvcc` (`--nvcc`, `nvcc` on the search
//! path by default) into `--work`. Each process prints one JSON line; the
//! creator prints the peer's line before its own.

use std::ffi::c_void;
use std::io::{BufRead, BufReader, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use platform::{
    Device, DeviceAllocation, DeviceId, LaunchConfig, Library, PeerMapping, PinnedBuffer,
    PrimaryContext, Protection, Stream, memcpy_async,
};

/// `fill` writes `seed ^ index` into every 32-bit word; `check` counts the
/// words that do not hold it.
const KERNELS: &str = r#"
extern "C" __global__ void fill(unsigned int* words, unsigned long long count, unsigned int seed) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        words[i] = seed ^ (unsigned int)i;
    }
}

extern "C" __global__ void check(const unsigned int* words, unsigned long long count, unsigned int seed, unsigned int* mismatches) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned int wrong = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        wrong += words[i] != (seed ^ (unsigned int)i);
    }
    if (wrong) atomicAdd(mismatches, wrong);
}
"#;

/// Blocks and threads per block of both kernels: 2^20 threads, each striding
/// over the words.
const GRID: u32 = 4096;
const BLOCK: u32 = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Solo,
    Separate,
    Shared,
}

struct Options {
    mode: Mode,
    device: DeviceId,
    half_bytes: usize,
    seconds: f64,
    work: PathBuf,
    nvcc: String,
    /// Set for the peer: the kernels' PTX, and in `shared` mode the inherited
    /// descriptor of the creator's allocation.
    peer: Option<(PathBuf, Option<i32>)>,
}

fn options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut options = Options {
        mode: Mode::Shared,
        device: DeviceId(0),
        half_bytes: 2 << 30,
        seconds: 10.0,
        work: std::env::temp_dir(),
        nvcc: "nvcc".into(),
        peer: None,
    };
    let mut ptx = None;
    let mut descriptor = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--mode" => {
                options.mode = match value.as_str() {
                    "solo" => Mode::Solo,
                    "separate" => Mode::Separate,
                    "shared" => Mode::Shared,
                    other => return Err(format!("unknown mode {other}").into()),
                }
            }
            "--device" => options.device = DeviceId(value.parse()?),
            "--half-bytes" => options.half_bytes = value.parse()?,
            "--seconds" => options.seconds = value.parse()?,
            "--work" => options.work = value.into(),
            "--nvcc" => options.nvcc = value,
            "--peer-ptx" => ptx = Some(PathBuf::from(value)),
            "--peer-descriptor" => descriptor = Some(value.parse()?),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if let Some(ptx) = ptx {
        options.peer = Some((ptx, descriptor));
    }
    Ok(options)
}

/// Compiles the kernels to PTX in `work` and returns the file.
fn compile(nvcc: &str, work: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(work)?;
    let source = work.join("shared_mapping_probe.cu");
    let ptx = work.join("shared_mapping_probe.ptx");
    std::fs::write(&source, KERNELS)?;
    let status = Command::new(nvcc)
        .args(["-ptx", "-O3", "-o"])
        .arg(&ptx)
        .arg(&source)
        .status()?;
    assert!(status.success(), "nvcc failed");
    Ok(ptx)
}

/// What one process measured.
struct Measurement {
    iterations: u64,
    elapsed: Duration,
    /// Words the device check found wrong, over every iteration.
    device_mismatches: u64,
    /// Words the final host read-back found wrong.
    host_mismatches: u64,
    last_seed: u32,
    /// Seconds per iteration: minimum, median, maximum.
    spread: [f64; 3],
}

/// Writes and checks `[address, address + bytes)` with seeds `tag | iteration`
/// until `seconds` have passed, then reads the range back to the host.
fn write_range(
    device: &Device,
    stream: &Stream,
    library: &Library,
    address: u64,
    bytes: usize,
    tag: u32,
    seconds: f64,
) -> Result<Measurement, Box<dyn std::error::Error>> {
    let fill = library.kernel("fill")?;
    let check = library.kernel("check")?;
    let config = LaunchConfig::new([GRID, 1, 1], [BLOCK, 1, 1]);
    let mut words = address;
    let mut count = (bytes / 4) as u64;

    // The device counter of wrong words and the host word it is copied to.
    let counter = DeviceAllocation::linear(device.id(), 4)?;
    let host = PinnedBuffer::new(device.id(), 4)?;
    // SAFETY: the counter holds one aligned word, valid until the synchronize.
    unsafe { platform::memset_d32(counter.address(), 0, 1, stream)? };
    stream.synchronize()?;
    let mut counter_address = counter.address();

    let mut durations = Vec::new();
    let mut seed = tag;
    let start = Instant::now();
    let deadline = Duration::from_secs_f64(seconds);
    while start.elapsed() < deadline {
        seed = tag | (durations.len() as u32 & 0x00ff_ffff);
        let began = Instant::now();
        let mut value = seed;
        fill.launch_ex(
            stream,
            &config,
            &mut [
                (&raw mut words).cast::<c_void>(),
                (&raw mut count).cast(),
                (&raw mut value).cast(),
            ],
        )?;
        check.launch_ex(
            stream,
            &config,
            &mut [
                (&raw mut words).cast::<c_void>(),
                (&raw mut count).cast(),
                (&raw mut value).cast(),
                (&raw mut counter_address).cast(),
            ],
        )?;
        stream.synchronize()?;
        durations.push(began.elapsed().as_secs_f64());
    }
    let elapsed = start.elapsed();

    // SAFETY: both words are valid and the copy completes at the synchronize.
    unsafe { memcpy_async(host.address(), counter.address(), 4, stream)? };
    stream.synchronize()?;
    // SAFETY: no copy into the buffer is pending.
    let device_mismatches = u32::from_ne_bytes(unsafe { host.as_slice() }.try_into()?);

    let host_mismatches = read_back(device, stream, address, bytes, seed)?;
    durations.sort_by(f64::total_cmp);
    Ok(Measurement {
        iterations: durations.len() as u64,
        elapsed,
        device_mismatches: device_mismatches.into(),
        host_mismatches,
        last_seed: seed,
        spread: [
            durations[0],
            durations[durations.len() / 2],
            durations[durations.len() - 1],
        ],
    })
}

/// Copies `[address, address + bytes)` to the host and counts the words that
/// are not `seed ^ index`.
fn read_back(
    device: &Device,
    stream: &Stream,
    address: u64,
    bytes: usize,
    seed: u32,
) -> Result<u64, Box<dyn std::error::Error>> {
    let host = PinnedBuffer::new(device.id(), bytes)?;
    // SAFETY: both ranges are valid and the copy completes at the synchronize.
    unsafe { memcpy_async(host.address(), address, bytes, stream)? };
    stream.synchronize()?;
    // SAFETY: no copy into the buffer is pending.
    let read = unsafe { host.as_slice() };
    let wrong = read
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .filter(|(index, word)| u32::from_ne_bytes(**word) != seed ^ *index as u32)
        .count();
    Ok(wrong as u64)
}

fn report(role: &str, mode: Mode, bytes: usize, measurement: &Measurement, extra: &str) -> String {
    let mode = match mode {
        Mode::Solo => "solo",
        Mode::Separate => "separate",
        Mode::Shared => "shared",
    };
    let written = measurement.iterations as f64 * bytes as f64;
    format!(
        "{{\"role\":\"{role}\",\"mode\":\"{mode}\",\"half_bytes\":{bytes},\"iterations\":{},\
         \"elapsed_s\":{:.6},\"write_gib_per_s\":{:.3},\"iteration_s\":{{\"min\":{:.6},\"median\":{:.6},\"max\":{:.6}}},\
         \"device_mismatches\":{},\"host_mismatches\":{},\"last_seed\":{}{extra}}}",
        measurement.iterations,
        measurement.elapsed.as_secs_f64(),
        written / measurement.elapsed.as_secs_f64() / (1u64 << 30) as f64,
        measurement.spread[0],
        measurement.spread[1],
        measurement.spread[2],
        measurement.device_mismatches,
        measurement.host_mismatches,
        measurement.last_seed,
    )
}

/// The driver's sticky error of the device's context, as a JSON field.
fn sticky(device: DeviceId) -> String {
    match PrimaryContext::retain(device).and_then(|context| context.sticky_error()) {
        Ok(None) => ",\"device_error\":null".into(),
        Ok(Some(error)) | Err(error) => format!(",\"device_error\":\"{error}\""),
    }
}

/// The peer: maps the inherited allocation (or creates its own), says `ready`,
/// waits for `go` and writes the second half.
fn peer(
    options: &Options,
    ptx: &Path,
    descriptor: Option<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(options.device)?;
    let stream = device.stream()?;
    let library = Library::load(&device, &std::fs::read(ptx)?)?;
    let half = options.half_bytes;

    // Keeps the storage mapped for the measurement.
    let (_own, _shared, address) = match descriptor {
        Some(descriptor) => {
            // SAFETY: the creator passed this descriptor to this process alone.
            let descriptor = unsafe { OwnedFd::from_raw_fd(descriptor) };
            let mapping = PeerMapping::import_retained(
                options.device,
                &descriptor,
                2 * half,
                &[],
                Protection::ReadWrite,
            )?;
            let address = mapping.address() + half as u64;
            (None, Some(mapping), address)
        }
        None => {
            let allocation = DeviceAllocation::retainable(options.device, half, &[])?;
            let address = allocation.address();
            (Some(allocation), None, address)
        }
    };

    println!("ready");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;

    let measurement = write_range(
        &device,
        &stream,
        &library,
        address,
        half,
        0xb000_0000,
        options.seconds,
    )?;
    println!(
        "{}",
        report(
            "peer",
            options.mode,
            half,
            &measurement,
            &sticky(options.device)
        )
    );
    std::io::stdout().flush()?;

    // Stay mapped until the creator has read this half back.
    line.clear();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(())
}

fn creator(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let ptx = compile(&options.nvcc, &options.work)?;
    let device = Device::new(options.device)?;
    let stream = device.stream()?;
    let library = Library::load(&device, &std::fs::read(&ptx)?)?;
    let half = options.half_bytes;

    let bytes = if options.mode == Mode::Shared {
        2 * half
    } else {
        half
    };
    let allocation = DeviceAllocation::retainable(options.device, bytes, &[])?;

    let mut child = None;
    if options.mode != Mode::Solo {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .args([
                "--mode",
                if options.mode == Mode::Shared {
                    "shared"
                } else {
                    "separate"
                },
            ])
            .args(["--device", &options.device.0.to_string()])
            .args(["--half-bytes", &half.to_string()])
            .args(["--seconds", &options.seconds.to_string()])
            .arg("--peer-ptx")
            .arg(&ptx)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());

        // The peer inherits the allocation's descriptor, as an engine receives
        // one from the Storage Server.
        let descriptor = if options.mode == Mode::Shared {
            let descriptor = allocation
                .export_descriptor()?
                .ok_or("a retainable allocation exports no descriptor")?;
            // SAFETY: clearing close-on-exec on a descriptor this process owns.
            let cleared = unsafe { libc::fcntl(descriptor.as_raw_fd(), libc::F_SETFD, 0) };
            assert_eq!(cleared, 0, "clearing close-on-exec");
            command.args(["--peer-descriptor", &descriptor.as_raw_fd().to_string()]);
            Some(descriptor)
        } else {
            None
        };

        let mut spawned = command.spawn()?;
        drop(descriptor);
        let mut output = BufReader::new(spawned.stdout.take().ok_or("the peer has no output")?);
        let mut line = String::new();
        output.read_line(&mut line)?;
        assert_eq!(line.trim(), "ready", "the peer did not become ready");
        let mut input = spawned.stdin.take().ok_or("the peer has no input")?;
        writeln!(input, "go")?;
        child = Some((spawned, input, output));
    }

    let measurement = write_range(
        &device,
        &stream,
        &library,
        allocation.address(),
        half,
        0xa000_0000,
        options.seconds,
    )?;

    let mut extra = sticky(options.device);
    if let Some((mut spawned, mut input, mut output)) = child {
        let mut line = String::new();
        output.read_line(&mut line)?;
        print!("{line}");

        if options.mode == Mode::Shared {
            // The peer's half, read through this process's mapping, holds the
            // peer's last pattern: the two processes wrote one allocation.
            let seed: u32 = line
                .split("\"last_seed\":")
                .nth(1)
                .and_then(|rest| rest.split([',', '}']).next())
                .and_then(|seed| seed.parse().ok())
                .ok_or("the peer's result states no last seed")?;
            let wrong = read_back(
                &device,
                &stream,
                allocation.address() + half as u64,
                half,
                seed,
            )?;
            extra.push_str(&format!(",\"peer_half_mismatches\":{wrong}"));
        }
        writeln!(input, "done")?;
        let status = spawned.wait()?;
        extra.push_str(&format!(",\"peer_exit\":{}", status.code().unwrap_or(-1)));
    }
    println!(
        "{}",
        report("creator", options.mode, half, &measurement, &extra)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = options()?;
    match &options.peer {
        Some((ptx, descriptor)) => peer(&options, ptx, *descriptor),
        None => creator(&options),
    }
}
