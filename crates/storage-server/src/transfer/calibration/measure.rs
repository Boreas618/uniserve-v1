//! The calibration step, `storage-server calibrate`: it times the host's copy paths on the
//! storage the Transfer Server copies and writes the record ([`Calibration`]) the Transfer Server
//! copies by.
//!
//! Each path is timed on the path its copies take, with storage of the kinds they copy: CUDA VMM
//! allocations that every GPU able to address them is granted access to, and memfds registered
//! with CUDA, bound to the GPU's closest host NUMA node as host regions' parts are. A path's
//! bandwidth is one untimed copy of the method's sample, then its timed copies back to back:
//! on the stream of the GPU whose copy engines run the path's copies (the destination's when it
//! is a GPU), timed by CUDA events; on the host around the CPU's copies between host buffers;
//! around each READ from its post to its completion. A path's fixed cost is the least time, as
//! the host observes it, of one copy of one page, over as many copies as the method times.
//!
//! The device storage the Transfer Server's context takes on each GPU is what NVML attributes to
//! this process once the measurement's storage is released: the contexts, their streams and the
//! copy paths the measurement brought up. The storage the Transfer Server maps later belongs to
//! its owners (NVML attributes an imported allocation to no importer).
//!
//! On request it also times copies from another host into each end here:
//!
//! - **Fabric** (`--fabric`): copies into each GPU from another GPU's allocation imported by its
//!   fabric handle, as the copy engines of a GPU copy another host's device storage of the
//!   NVLink fabric; from every other GPU of the host in turn (the GPU itself on a host of one),
//!   and the slowest of them is the path's cost.
//! - **READs** (`--reads`): NIXL READs into host storage and into each GPU from host storage
//!   another agent of this process registered, both agents limited to RDMA
//!   ([`Transports::Rdma`]), so the bytes cross the host's RDMA ports as a READ from another
//!   host's does.
//! - **Transfer Engine READs** (`--mooncake-reads`): Mooncake Transfer Engine READs into host
//!   storage and into each GPU from host storage another engine of this process registered,
//!   over the RDMA transport each engine installs, as the Transfer Server's engine READs another
//!   host's storage.

use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ::storage::protocol::{ContextBytes, Link, LinkEnd};
use ::storage::settings::Transports;
use anyhow::Context as _;
use platform::{
    DeviceAllocation, Event, EventFlags, HostRegistration, Nvml, PeerMapping, PrimaryContext,
    ShareableHandle, Stream, memcpy_async,
};

use super::{Calibration, Cost, LinkCost, Method, RemoteCost, RemotePath};
use crate::Error;
use crate::transfer::devices::{Gpu, Gpus};
use crate::transfer::storage::{HostMap, nixl_key};
use crate::transfer::{mooncake, nixl};

/// The `storage-server calibrate` command's arguments.
#[derive(Debug, Clone, clap::Args)]
pub struct Arguments {
    /// Where to write the calibration (JSON), which `storage_config.transfer_calibration` names.
    #[arg(long, value_name = "PATH")]
    pub output: PathBuf,
    /// The MiB of each timed copy.
    #[arg(long, default_value_t = 64)]
    pub sample_mib: u64,
    /// The timed copies of each path, after one untimed copy.
    #[arg(long, default_value_t = 8)]
    pub samples: u32,
    /// Calibrate host storage alone, without loading the CUDA driver: the Transfer Server then
    /// copies on no GPU.
    #[arg(long)]
    pub host_only: bool,
    /// Also time copies into each GPU from another GPU's allocation imported by its fabric
    /// handle, as copies from another host of the NVLink fabric run; needs GPUs that export
    /// fabric handles (an IMEX channel).
    #[arg(long, conflicts_with = "host_only")]
    pub fabric: bool,
    /// Also time NIXL READs into host storage and each GPU through the host's RDMA ports, as
    /// READs from another host run; needs NIXL and a usable RDMA port.
    #[arg(long)]
    pub reads: bool,
    /// Also time Mooncake Transfer Engine READs into host storage and each GPU through the
    /// host's RDMA ports, as the Transfer Engine's READs from another host run; needs the
    /// Transfer Engine (`docs/mooncake.md`) and a usable RDMA port.
    #[arg(long)]
    pub mooncake_reads: bool,
}

/// Runs the step: calibrates the host as `arguments` state and writes the record to its output.
///
/// # Errors
///
/// Fails when a path cannot be timed or the record cannot be written.
pub fn run(arguments: &Arguments) -> anyhow::Result<()> {
    let calibration = calibrate(arguments)?;
    let document = serde_json::to_vec_pretty(&calibration)?;
    std::fs::write(&arguments.output, document)
        .with_context(|| format!("writing {}", arguments.output.display()))?;
    for measured in &calibration.links {
        tracing::info!(
            source = %crate::transfer::storage_name(&measured.link.source),
            destination = %crate::transfer::storage_name(&measured.link.destination),
            gigabytes_per_second = measured.cost.bytes_per_second as f64 / 1e9,
            overhead_us = measured.cost.overhead_ns as f64 / 1e3,
            "calibrated a link"
        );
    }
    for remote in &calibration.remote {
        tracing::info!(
            path = ?remote.path,
            destination = %crate::transfer::storage_name(&remote.destination),
            gigabytes_per_second = remote.cost.bytes_per_second as f64 / 1e9,
            overhead_us = remote.cost.overhead_ns as f64 / 1e3,
            "calibrated copies from another host"
        );
    }
    Ok(())
}

/// The calibration of this host as `arguments` state, checked; the process holds no CUDA
/// context when it returns.
///
/// # Errors
///
/// Fails when the driver shows no GPU (without `--host-only`), a path cannot be timed, or the
/// record does not hold together.
pub fn calibrate(arguments: &Arguments) -> Result<Calibration, Error> {
    let method = Method {
        sample_bytes: arguments.sample_mib << 20,
        samples: arguments.samples,
    };
    let timing = Timing::of(method)?;
    let gpus = if arguments.host_only {
        Gpus::of(&[])?
    } else {
        Gpus::shown()?
    };
    if !arguments.host_only && gpus.is_empty() {
        return Err(Error::Calibration(
            "step finds no GPU the driver shows: calibrate host storage alone with --host-only"
                .into(),
        ));
    }

    // The contexts are held through the measurement and until their storage is measured,
    // whatever objects of the measurement live in them.
    let held = gpus
        .iter()
        .map(|gpu| PrimaryContext::retain(gpu.device))
        .collect::<Result<Vec<_>, _>>()?;
    let mut links = vec![LinkCost {
        link: Link {
            source: LinkEnd::Host,
            destination: LinkEnd::Host,
        },
        cost: timing.host_to_host()?,
    }];
    let mut remote = Vec::new();
    {
        let scratch: Vec<Scratch> = gpus
            .iter()
            .map(|gpu| Scratch::new(gpu, timing.sample))
            .collect::<Result<_, _>>()?;
        let device = |gpu: &Gpu| LinkEnd::Device { gpu: gpu.uuid };
        for (gpu, own) in gpus.iter().zip(&scratch) {
            let mut timed = |source: LinkEnd, destination: LinkEnd, from: u64| {
                // SAFETY: both ranges are measurement storage of the sample's bytes that lives
                // until the stream drained.
                let cost = unsafe { timing.copies(&own.stream, own.destination.address(), from) }?;
                links.push(LinkCost {
                    link: Link {
                        source,
                        destination,
                    },
                    cost,
                });
                Ok::<(), Error>(())
            };
            timed(device(gpu), device(gpu), own.source.address())?;
            timed(LinkEnd::Host, device(gpu), own.registration.address())?;
            // Peer links into this GPU, on its stream: the destination's copy engine reads the
            // source's storage over the GPUs' link.
            for (peer, theirs) in gpus.iter().zip(&scratch) {
                if peer.device != gpu.device && peer.readers.contains(&gpu.device) {
                    timed(device(peer), device(gpu), theirs.source.address())?;
                }
            }
            // SAFETY: as above.
            let outward = unsafe {
                timing.copies(
                    &own.stream,
                    own.registration.address(),
                    own.source.address(),
                )
            }?;
            links.push(LinkCost {
                link: Link {
                    source: device(gpu),
                    destination: LinkEnd::Host,
                },
                cost: outward,
            });
        }
        if arguments.fabric {
            for (gpu, own) in gpus.iter().zip(&scratch) {
                remote.push(RemoteCost {
                    path: RemotePath::Fabric,
                    destination: device(gpu),
                    cost: timing.fabric(&gpus, gpu, own)?,
                });
            }
        }
        if arguments.reads {
            remote.extend(timing.reads(&gpus, &scratch)?);
        }
        if arguments.mooncake_reads {
            remote.extend(timing.engine_reads(&gpus, &scratch)?);
        }
    }
    // The measurement's storage is released before the contexts are measured.
    platform::reclaim::flush()?;
    let contexts = if gpus.is_empty() {
        Vec::new()
    } else {
        let nvml = Nvml::init()?;
        gpus.iter()
            .map(|gpu| {
                Ok(ContextBytes {
                    gpu: gpu.uuid,
                    bytes: nvml.process_storage(gpu.device)?,
                })
            })
            .collect::<Result<_, platform::Error>>()?
    };
    drop(held);
    platform::reclaim::flush()?;

    let calibration = Calibration {
        method,
        gpus: contexts,
        links,
        remote,
    };
    calibration.check().map_err(Error::Calibration)?;
    Ok(calibration)
}

/// The measurement's storage on one GPU: two device allocations, a host buffer on the GPU's
/// NUMA node, and the stream the GPU's copies run on.
struct Scratch {
    stream: Stream,
    source: DeviceAllocation,
    destination: DeviceAllocation,
    // The registration is undone before the buffer is unmapped.
    registration: HostRegistration,
    _host: HostMap,
}

impl Scratch {
    fn new(gpu: &Gpu, bytes: usize) -> Result<Self, Error> {
        let stream = Stream::new(gpu.device)?;
        let source = DeviceAllocation::exportable(gpu.device, bytes, &gpu.readers)?;
        let destination = DeviceAllocation::exportable(gpu.device, bytes, &gpu.readers)?;
        let host = HostMap::new(&host_buffer(gpu.numa_node, bytes)?, bytes, true)?;
        // SAFETY: `host` maps the range until after the registration drops.
        let registration =
            unsafe { HostRegistration::new(gpu.device, host.address() as *mut u8, bytes) }?;
        Ok(Self {
            stream,
            source,
            destination,
            registration,
            _host: host,
        })
    }
}

/// How each path is timed: the method's sample and copies, and the host's page.
struct Timing {
    method: Method,
    /// The bytes of one timed copy.
    sample: usize,
    /// The bytes of the copy a fixed cost is timed by.
    page: usize,
}

impl Timing {
    fn of(method: Method) -> Result<Self, Error> {
        let sample = usize::try_from(method.sample_bytes).map_err(|_| {
            Error::Calibration(format!("sample of {} bytes overflows", method.sample_bytes))
        })?;
        let page = rustix::param::page_size();
        if sample < page || method.samples == 0 {
            return Err(Error::Calibration(format!(
                "method times {} copies of {sample} bytes: at least one copy of at least a page \
                 ({page} bytes)",
                method.samples
            )));
        }
        Ok(Self {
            method,
            sample,
            page,
        })
    }

    /// The least of `samples` timings of `copy`, which returns how long it took.
    fn least(&self, mut copy: impl FnMut() -> Result<Duration, Error>) -> Result<Duration, Error> {
        let mut least = Duration::MAX;
        for _ in 0..self.method.samples {
            least = least.min(copy()?);
        }
        Ok(least)
    }

    /// The cost of copies from `source` to `destination` on `stream`: the sample copied once,
    /// then its timed copies between two timing events; and the least time of one page's copy,
    /// from its submission until a blocking event recorded after it completed.
    ///
    /// # Safety
    ///
    /// Both ranges hold the sample's bytes and stay valid until the call returns, which drains
    /// the stream.
    unsafe fn copies(&self, stream: &Stream, destination: u64, source: u64) -> Result<Cost, Error> {
        let timing = EventFlags {
            timing: true,
            ..EventFlags::default()
        };
        let (start, end) = (Event::new(timing), Event::new(timing));
        // SAFETY: forwarded from the caller.
        unsafe { memcpy_async(destination, source, self.sample, stream) }?;
        start.record(stream)?;
        for _ in 0..self.method.samples {
            // SAFETY: forwarded from the caller.
            unsafe { memcpy_async(destination, source, self.sample, stream) }?;
        }
        end.record(stream)?;
        end.synchronize()?;
        let seconds = f64::from(start.elapsed_ms(&end)?) / 1e3;
        let bytes_per_second = self.rate(seconds)?;

        let completion = Event::new(EventFlags {
            blocking: true,
            ..EventFlags::default()
        });
        let overhead = self.least(|| {
            let started = Instant::now();
            // SAFETY: forwarded from the caller; a page lies within the sample.
            unsafe { memcpy_async(destination, source, self.page, stream) }?;
            completion.record(stream)?;
            completion.synchronize()?;
            Ok(started.elapsed())
        })?;
        Ok(cost(bytes_per_second, overhead))
    }

    /// The cost of the CPU's copies between two host buffers.
    fn host_to_host(&self) -> Result<Cost, Error> {
        let source = HostMap::new(&host_buffer(None, self.sample)?, self.sample, true)?;
        let destination = HostMap::new(&host_buffer(None, self.sample)?, self.sample, true)?;
        let copy = |bytes: usize| {
            let started = Instant::now();
            // SAFETY: two distinct mappings of the sample's bytes this function holds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.address() as *const u8,
                    destination.address() as *mut u8,
                    bytes,
                );
            }
            started.elapsed()
        };
        // The untimed copy faults both buffers' pages in.
        copy(self.sample);
        let started = Instant::now();
        for _ in 0..self.method.samples {
            copy(self.sample);
        }
        let bytes_per_second = self.rate(started.elapsed().as_secs_f64())?;
        let overhead = self.least(|| Ok(copy(self.page)))?;
        Ok(cost(bytes_per_second, overhead))
    }

    /// The cost of copies into `gpu`'s scratch (`own`) from an allocation of another GPU of
    /// `gpus` imported on `gpu` by its fabric handle: the slowest over every other GPU, or over
    /// `gpu` itself on a host of one GPU.
    fn fabric(&self, gpus: &Gpus, gpu: &Gpu, own: &Scratch) -> Result<Cost, Error> {
        let mut exporters: Vec<&Gpu> = gpus
            .iter()
            .filter(|other| other.device != gpu.device)
            .collect();
        if exporters.is_empty() {
            exporters.push(gpu);
        }
        let mut slowest: Option<Cost> = None;
        for exporter in exporters {
            let exported = DeviceAllocation::retainable(exporter.device, self.sample, &[])?;
            let Some(ShareableHandle::Fabric(handle)) = exported.export()? else {
                return Err(Error::Calibration(format!(
                    "step finds that {} exports no fabric handle: calibrate without --fabric",
                    platform::DeviceUuid(exporter.uuid)
                )));
            };
            let import =
                PeerMapping::import(gpu.device, &ShareableHandle::Fabric(handle), exported.len())?;
            // SAFETY: the import maps at least the sample's bytes and lives until the stream
            // drained, as does the destination.
            let cost =
                unsafe { self.copies(&own.stream, own.destination.address(), import.address()) }?;
            slowest = Some(match slowest {
                Some(slowest) => Cost {
                    bytes_per_second: slowest.bytes_per_second.min(cost.bytes_per_second),
                    overhead_ns: slowest.overhead_ns.max(cost.overhead_ns),
                },
                None => cost,
            });
        }
        slowest.ok_or_else(|| Error::Calibration("step timed no fabric copy".into()))
    }

    /// The cost of NIXL READs into host storage and into each GPU's scratch, from host storage
    /// another agent of this process registered; both agents take RDMA alone.
    fn reads(&self, gpus: &Gpus, scratch: &[Scratch]) -> Result<Vec<RemoteCost>, Error> {
        let backend = nixl::Backend::of_host(Transports::Rdma)?;
        let source = nixl::Agent::new(&crate::transfer::network::agent_name()?, &backend)?;
        let reader = nixl::Agent::new(&crate::transfer::network::agent_name()?, &backend)?;

        let source_storage = HostMap::new(&host_buffer(None, self.sample)?, self.sample, true)?;
        let source_region = nixl::Region {
            segment: nixl::Segment::Host,
            address: source_storage.address(),
            len: self.sample,
            key: nixl_key(),
        };
        // SAFETY: `source_storage` maps the region writable until after the registration drops,
        // and nothing in this process references its bytes.
        let _source = unsafe { source.register(source_region, nixl::Access::ReadWrite) }?;
        let peer = reader.load_peer(&source.metadata()?)?;

        let host = HostMap::new(&host_buffer(None, self.sample)?, self.sample, true)?;
        let mut destinations = vec![(LinkEnd::Host, nixl::Segment::Host, host.address())];
        for (gpu, own) in gpus.iter().zip(scratch) {
            destinations.push((
                LinkEnd::Device { gpu: gpu.uuid },
                nixl::Segment::Device,
                own.destination.address(),
            ));
        }
        let mut costs = Vec::with_capacity(destinations.len());
        for (destination, segment, address) in destinations {
            let region = nixl::Region {
                segment,
                address,
                len: self.sample,
                key: nixl_key(),
            };
            // SAFETY: the host buffer or the GPU's scratch maps the region writable until after
            // the registration drops, and nothing in this process references its bytes.
            let _registration = unsafe { reader.register(region, nixl::Access::ReadWrite) }?;
            let read = |len: usize| -> Result<Duration, Error> {
                let span = nixl::Span {
                    source: nixl::Region {
                        len,
                        ..source_region
                    },
                    destination: nixl::Region { len, ..region },
                };
                let started = Instant::now();
                reader.read(&peer, &[span])?.wait(started)?;
                Ok(started.elapsed())
            };
            // The first READ over the connection also sets it up.
            read(self.sample)?;
            let mut elapsed = Duration::ZERO;
            for _ in 0..self.method.samples {
                elapsed += read(self.sample)?;
            }
            let bytes_per_second = self.rate(elapsed.as_secs_f64())?;
            let overhead = self.least(|| read(self.page))?;
            costs.push(RemoteCost {
                path: RemotePath::Read,
                destination,
                cost: cost(bytes_per_second, overhead),
            });
        }
        Ok(costs)
    }

    /// The rate of the method's timed copies in `seconds`.
    /// The cost of Transfer Engine READs into host storage and into each GPU's scratch, from
    /// host storage another engine of this process registered: the reading engine opens the
    /// source engine's segment by the peer-to-peer handshake, as it opens another host's.
    fn engine_reads(&self, gpus: &Gpus, scratch: &[Scratch]) -> Result<Vec<RemoteCost>, Error> {
        let host = crate::host_name()?;
        let topology = mooncake::Topology::of_host()?;
        let source = mooncake::Engine::new(&host, &topology)?;
        let reader = mooncake::Engine::new(&host, &topology)?;

        let source_storage = HostMap::new(&host_buffer(None, self.sample)?, self.sample, true)?;
        // SAFETY: `source_storage` maps the range writable until after the registration drops,
        // and nothing in this process references its bytes.
        let _source = unsafe { source.register(source_storage.address(), self.sample) }?;
        let segment = reader.open(source.name())?;

        let host_storage = HostMap::new(&host_buffer(None, self.sample)?, self.sample, true)?;
        let mut destinations = vec![(LinkEnd::Host, host_storage.address())];
        for (gpu, own) in gpus.iter().zip(scratch) {
            destinations.push((LinkEnd::Device { gpu: gpu.uuid }, own.destination.address()));
        }
        let mut costs = Vec::with_capacity(destinations.len());
        for (destination, address) in destinations {
            // SAFETY: the host buffer or the GPU's scratch maps the range writable until after
            // the registration drops, and nothing in this process references its bytes.
            let _registration = unsafe { reader.register(address, self.sample) }?;
            let read = |len: usize| -> Result<Duration, Error> {
                let span = mooncake::Span {
                    local: address,
                    remote: source_storage.address(),
                    len: len as u64,
                };
                let started = Instant::now();
                reader.read(segment, &[span])?.wait(started)?;
                Ok(started.elapsed())
            };
            // The first READ over the connection also sets it up.
            read(self.sample)?;
            let mut elapsed = Duration::ZERO;
            for _ in 0..self.method.samples {
                elapsed += read(self.sample)?;
            }
            let bytes_per_second = self.rate(elapsed.as_secs_f64())?;
            let overhead = self.least(|| read(self.page))?;
            costs.push(RemoteCost {
                path: RemotePath::Mooncake,
                destination,
                cost: cost(bytes_per_second, overhead),
            });
        }
        Ok(costs)
    }

    fn rate(&self, seconds: f64) -> Result<u64, Error> {
        if seconds <= 0.0 {
            return Err(Error::Calibration(
                "step timed copies that took no measurable time: time larger samples".into(),
            ));
        }
        let bytes = self.method.sample_bytes as f64 * f64::from(self.method.samples);
        Ok((bytes / seconds) as u64)
    }
}

/// A cost of `bytes_per_second` and a fixed cost of `overhead`.
fn cost(bytes_per_second: u64, overhead: Duration) -> Cost {
    Cost {
        bytes_per_second,
        overhead_ns: u64::try_from(overhead.as_nanos()).unwrap_or(u64::MAX),
    }
}

/// A memfd of `bytes`, its pages preferring `numa_node`.
fn host_buffer(numa_node: Option<u32>, bytes: usize) -> Result<OwnedFd, Error> {
    let descriptor = rustix::fs::memfd_create(
        c"uniserve-transfer-calibration",
        rustix::fs::MemfdFlags::CLOEXEC,
    )
    .map_err(std::io::Error::from)?;
    rustix::fs::ftruncate(&descriptor, bytes as u64).map_err(std::io::Error::from)?;
    if let Some(node) = numa_node {
        crate::memory::prefer_node(&descriptor, bytes as u64, node)?;
    }
    Ok(descriptor)
}
