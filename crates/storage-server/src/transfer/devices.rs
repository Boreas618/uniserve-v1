//! The GPUs the Transfer Server copies on, and what it measures of them
//! when it starts: every link's bandwidth and the device memory its CUDA
//! context takes on each GPU.
//!
//! A link's bandwidth is measured on the path its copies take, with memory
//! of the kinds they copy: CUDA VMM allocations that every GPU able to
//! address them is granted access to, and memfds registered with CUDA,
//! bound to the GPU's closest host NUMA node as page pools are. Each link
//! runs one untimed copy of [`SAMPLE_BYTES`], then [`SAMPLES`] timed ones
//! back to back on the stream of the GPU that runs its copies (the
//! destination's when it is a GPU), timed by CUDA events; the host-to-host
//! link is timed on the host around the CPU's copies.
//!
//! The context's device memory is what NVML attributes to the process once
//! the measurement's memory is released: the contexts, their streams and
//! the copy paths the measurement brought up. The memory the Transfer
//! Server maps later belongs to its owners (NVML attributes an imported
//! allocation to no importer).

use std::os::fd::OwnedFd;
use std::time::Instant;

use cuda::{
    DeviceAllocation, DeviceAttribute, DeviceId, Event, EventFlags, HostRegistration, Nvml,
    PrimaryContext, Stream, memcpy_async,
};
use storage::protocol::{ContextBytes, Link, LinkBandwidth, Memory};

use super::Error;
use super::memory::HostMap;

/// The bytes of one measured copy.
const SAMPLE_BYTES: usize = 64 << 20;

/// The timed copies of each link, after one untimed copy.
const SAMPLES: usize = 8;

/// `CUDA_ERROR_NO_DEVICE`: the driver sees no GPU, as when
/// `CUDA_VISIBLE_DEVICES` names none.
const CUDA_ERROR_NO_DEVICE: i32 = 100;

/// One GPU, as this process numbers it.
pub(super) struct Gpu {
    pub(super) device: DeviceId,
    pub(super) uuid: [u8; 16],
    /// The host NUMA node closest to it.
    pub(super) numa_node: Option<u32>,
    /// The GPUs whose copy engines can address its memory, itself first.
    pub(super) readers: Vec<DeviceId>,
    /// The device's primary context, held for the process's life: every
    /// stream and mapping of the GPU lives in it.
    _context: PrimaryContext,
}

/// The GPUs this process sees.
pub(super) struct Gpus {
    gpus: Vec<Gpu>,
}

impl Gpus {
    /// The GPUs the driver shows this process. A host without a driver, or
    /// whose driver shows the process no GPU, has none: its Transfer
    /// Server copies between host memory only.
    pub(super) fn discover() -> Result<Self, cuda::Error> {
        let count = match DeviceId::count() {
            Ok(count) => count,
            Err(cuda::Error::Load { .. }) => {
                tracing::info!("no CUDA driver: the Transfer Server copies host memory only");
                0
            }
            Err(cuda::Error::Driver {
                code: CUDA_ERROR_NO_DEVICE,
                ..
            }) => {
                tracing::info!("no visible GPU: the Transfer Server copies host memory only");
                0
            }
            Err(error) => return Err(error),
        };
        let devices: Vec<DeviceId> = (0..count).map(DeviceId).collect();
        let mut gpus = Vec::with_capacity(devices.len());
        for &device in &devices {
            let mut readers = vec![device];
            for &peer in &devices {
                if peer != device && peer.can_access_peer(device)? {
                    readers.push(peer);
                }
            }
            gpus.push(Gpu {
                device,
                uuid: device.uuid()?.0,
                numa_node: device
                    .attribute(DeviceAttribute::HostNumaId)
                    .ok()
                    .and_then(|node| u32::try_from(node).ok()),
                readers,
                _context: PrimaryContext::retain(device)?,
            });
        }
        Ok(Self { gpus })
    }

    pub(super) fn by_uuid(&self, uuid: [u8; 16]) -> Option<&Gpu> {
        self.gpus.iter().find(|gpu| gpu.uuid == uuid)
    }

    pub(super) fn by_device(&self, device: DeviceId) -> Option<&Gpu> {
        self.gpus.iter().find(|gpu| gpu.device == device)
    }

    /// The GPU that runs the copies no GPU of their own runs: those between
    /// host memory and another host's GPU.
    pub(super) fn first(&self) -> Option<&Gpu> {
        self.gpus.first()
    }
}

/// The measurement's memory on one GPU: two device allocations, a host
/// buffer on the GPU's NUMA node, and the stream the GPU's copies run on.
struct Scratch {
    stream: Stream,
    source: DeviceAllocation,
    destination: DeviceAllocation,
    // The registration is undone before the buffer is unmapped.
    registration: HostRegistration,
    _host: HostMap,
}

impl Scratch {
    fn new(gpu: &Gpu) -> Result<Self, Error> {
        let stream = Stream::new(gpu.device)?;
        let source = DeviceAllocation::exportable(gpu.device, SAMPLE_BYTES, &gpu.readers)?;
        let destination = DeviceAllocation::exportable(gpu.device, SAMPLE_BYTES, &gpu.readers)?;
        let host = HostMap::new(&host_buffer(gpu.numa_node)?, SAMPLE_BYTES, true)?;
        // SAFETY: `_host` maps the range until after the registration drops.
        let registration =
            unsafe { HostRegistration::new(gpu.device, host.address() as *mut u8, SAMPLE_BYTES) }?;
        Ok(Self {
            stream,
            source,
            destination,
            registration,
            _host: host,
        })
    }
}

/// Measure every link of `gpus` and the device memory of each GPU's
/// context.
pub(super) fn measure(gpus: &Gpus) -> Result<(Vec<LinkBandwidth>, Vec<ContextBytes>), Error> {
    let mut links = Vec::new();
    links.push(LinkBandwidth {
        link: Link {
            source: Memory::Host,
            destination: Memory::Host,
        },
        bytes_per_second: host_to_host()?,
    });
    {
        let scratch: Vec<Scratch> = gpus
            .gpus
            .iter()
            .map(Scratch::new)
            .collect::<Result<_, _>>()?;
        let device = |gpu: &Gpu| Memory::Device { gpu: gpu.uuid };
        for (gpu, own) in gpus.gpus.iter().zip(&scratch) {
            let mut measured = |source: Memory, destination: Memory, to: u64, from: u64| {
                // SAFETY: both ranges are measurement memory of
                // `SAMPLE_BYTES` that lives until the stream drained.
                let bytes_per_second = unsafe { time_copies(&own.stream, to, from) }?;
                links.push(LinkBandwidth {
                    link: Link {
                        source,
                        destination,
                    },
                    bytes_per_second,
                });
                Ok::<(), Error>(())
            };
            measured(
                device(gpu),
                device(gpu),
                own.destination.address(),
                own.source.address(),
            )?;
            measured(
                Memory::Host,
                device(gpu),
                own.destination.address(),
                own.registration.address(),
            )?;
            measured(
                device(gpu),
                Memory::Host,
                own.registration.address(),
                own.source.address(),
            )?;
            // Peer links into this GPU, on its stream: the destination's
            // copy engine reads the source's memory over the GPUs' link.
            for (peer, theirs) in gpus.gpus.iter().zip(&scratch) {
                if peer.device != gpu.device && peer.readers.contains(&gpu.device) {
                    measured(
                        device(peer),
                        device(gpu),
                        own.destination.address(),
                        theirs.source.address(),
                    )?;
                }
            }
        }
    }
    // The measurement's memory is released before the context is measured.
    cuda::reclaim::flush()?;
    let contexts = if gpus.gpus.is_empty() {
        Vec::new()
    } else {
        let nvml = Nvml::init()?;
        gpus.gpus
            .iter()
            .map(|gpu| {
                Ok(ContextBytes {
                    gpu: gpu.uuid,
                    bytes: nvml.process_memory(gpu.device)?,
                })
            })
            .collect::<Result<_, cuda::Error>>()?
    };
    for measured in &links {
        tracing::info!(
            source = %memory_name(measured.link.source),
            destination = %memory_name(measured.link.destination),
            gigabytes_per_second = measured.bytes_per_second as f64 / 1e9,
            "measured a link"
        );
    }
    Ok((links, contexts))
}

/// Copies `SAMPLE_BYTES` from `source` to `destination` on `stream` once,
/// then `SAMPLES` times between two timing events, and returns the rate.
///
/// # Safety
///
/// Both ranges hold `SAMPLE_BYTES` and stay valid until the call returns,
/// which drains the stream.
unsafe fn time_copies(stream: &Stream, destination: u64, source: u64) -> Result<u64, Error> {
    let timing = EventFlags {
        timing: true,
        ..EventFlags::default()
    };
    let (start, end) = (Event::new(timing), Event::new(timing));
    // SAFETY: forwarded from the caller.
    unsafe { memcpy_async(destination, source, SAMPLE_BYTES, stream) }?;
    start.record(stream)?;
    for _ in 0..SAMPLES {
        // SAFETY: forwarded from the caller.
        unsafe { memcpy_async(destination, source, SAMPLE_BYTES, stream) }?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let seconds = f64::from(start.elapsed_ms(&end)?) / 1e3;
    Ok(rate(SAMPLES * SAMPLE_BYTES, seconds))
}

/// The rate of the CPU's copies between two host buffers.
fn host_to_host() -> Result<u64, Error> {
    let source = HostMap::new(&host_buffer(None)?, SAMPLE_BYTES, true)?;
    let destination = HostMap::new(&host_buffer(None)?, SAMPLE_BYTES, true)?;
    let copy = || {
        // SAFETY: two distinct mappings of `SAMPLE_BYTES` this function
        // holds.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.address() as *const u8,
                destination.address() as *mut u8,
                SAMPLE_BYTES,
            );
        }
    };
    // The untimed copy faults both buffers' pages in.
    copy();
    let started = Instant::now();
    for _ in 0..SAMPLES {
        copy();
    }
    Ok(rate(
        SAMPLES * SAMPLE_BYTES,
        started.elapsed().as_secs_f64(),
    ))
}

/// A memfd of `SAMPLE_BYTES`, its pages preferring `numa_node`.
fn host_buffer(numa_node: Option<u32>) -> Result<OwnedFd, Error> {
    let descriptor = rustix::fs::memfd_create(
        c"uniserve-transfer-measure",
        rustix::fs::MemfdFlags::CLOEXEC,
    )
    .map_err(std::io::Error::from)?;
    rustix::fs::ftruncate(&descriptor, SAMPLE_BYTES as u64).map_err(std::io::Error::from)?;
    if let Some(node) = numa_node {
        crate::pages::prefer_node(&descriptor, SAMPLE_BYTES as u64, node)?;
    }
    Ok(descriptor)
}

/// Bytes per second of `bytes` in `seconds`.
fn rate(bytes: usize, seconds: f64) -> u64 {
    if seconds <= 0.0 {
        return u64::MAX;
    }
    (bytes as f64 / seconds) as u64
}

/// A link end's name for the log.
pub(super) fn memory_name(memory: Memory) -> String {
    match memory {
        Memory::Host => "host".into(),
        Memory::Device { gpu } => cuda::DeviceUuid(gpu).to_string(),
    }
}
