//! CUPTI's PC sampling as a rank runs it.
//!
//! CUPTI attributes the samples of a CUDA graph's kernels to their program counters only when PC
//! sampling was enabled when the graph was captured, and stayed enabled since; otherwise it counts
//! them as samples of non-user kernels, without records. A rank captures its graphs at warmup, so
//! a deployment that profiles them (`profiling_config.graph_pc_sampling`) has every rank enable PC
//! sampling, stopped, before it captures, and keep it enabled for its life ([`enable`]); sessions
//! only configure, start and stop the rank's sampler ([`of`]). CUPTI also refuses to enable an
//! activity record once PC sampling is enabled, unless the record's kinds were enabled before it,
//! so the rank first holds the kinds of its always-on stream record enabled for its life; the
//! stream record and the timelines that trace graphs by node start later beside it. While it is
//! enabled CUPTI refuses PM sampling, the range profiler and whole-graph activity records, which
//! the leader refuses at the session request
//! (`proto::profile::SessionRequest::refused_beside_graph_pc_sampling`). Without the option, a
//! session enables a sampler of its own, which attributes only kernels launched outside CUDA
//! graphs.
//!
//! CUPTI keeps the samples in a hardware buffer of process host storage, which its own thread
//! empties; a buffer too small for the sampling rate fills, and the samples are lost.
//! [`hardware_buffer`] sizes it for a period, and a read whose buffer filled tells the session
//! the period is too short for the device, which releases the collector as failed
//! (`super::collectors`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use platform::cupti::activity::KindsHold;
use platform::cupti::pc_sampling::PcSampler;
use platform::{DeviceId, PrimaryContext};

use crate::worker::{Error, Result};

use service::lock;

/// The program counters one take of the sampler reads at most.
pub(crate) const MAX_PCS: usize = 10_000;

/// Each rank's sampler, by its device, enabled for its life when the deployment profiles its
/// graphs' kernels.
static RANKS: Mutex<Option<HashMap<DeviceId, Arc<Mutex<PcSampler>>>>> = Mutex::new(None);

/// The stream record's activity kinds, held enabled from before the rank's sampler for its life.
static KINDS: OnceLock<KindsHold> = OnceLock::new();

/// Enables PC sampling on `context`, stopped, for the rank's life, at the default period, before
/// the rank captures any CUDA graph.
///
/// # Errors
///
/// Fails when CUPTI refuses PC sampling, as when another profiler holds the process's CUPTI
/// subscription.
pub(crate) fn enable(context: &PrimaryContext) -> Result<()> {
    let period = proto::profile::PcSamplingOptions::default().period;
    let bytes = hardware_buffer(period);
    if KINDS.get().is_none() {
        let hold = KindsHold::take().map_err(|error| {
            Error::Unsupported(format!("the activity records beside PC sampling: {error}"))
        })?;
        let _ = KINDS.set(hold);
    }
    let sampler =
        PcSampler::enable(context.handle(), period, MAX_PCS, Some(bytes)).map_err(|error| {
            Error::Unsupported(format!("PC sampling for the rank's graphs: {error}"))
        })?;
    let mut ranks = lock(&RANKS);
    let ranks = ranks.get_or_insert_with(HashMap::new);
    if ranks
        .insert(context.device(), Arc::new(Mutex::new(sampler)))
        .is_some()
    {
        return Err(Error::State("the rank enabled PC sampling twice"));
    }
    tracing::info!(
        period,
        hardware_buffer = bytes,
        "enabled PC sampling for the rank's CUDA graphs"
    );
    Ok(())
}

/// The sampler of the rank of `device`, when the rank keeps one.
pub(crate) fn of(device: DeviceId) -> Option<Arc<Mutex<PcSampler>>> {
    lock(&RANKS).as_ref()?.get(&device).cloned()
}

/// CUPTI's largest hardware buffer, which is also its default: 512 MiB.
const LARGEST_HARDWARE_BUFFER: usize = 512 << 20;

/// The hardware buffer, in bytes, a session samples at `period` into.
///
/// Sized from a measurement on GB200 with `nvidia-cuda-cupti` 13.0 with every SM busy: a size
/// holds when no window of 0.6 s filled it, each period takes twice the smallest size that held
/// three windows, and the size taken held ten more (`docs/profiling.md` states the table). A
/// period shorter than that table's takes CUPTI's largest buffer. A device that samples faster
/// fills a buffer sooner: the first read whose buffer filled releases the collector as failed,
/// naming the period, rather than recording a sample of what ran.
pub(crate) fn hardware_buffer(period: u32) -> usize {
    const MIB: usize = 1 << 20;
    match period {
        19.. => 2 * MIB,
        18 => 4 * MIB,
        17 => 8 * MIB,
        15 | 16 => 32 * MIB,
        14 => 128 * MIB,
        13 => 256 * MIB,
        _ => LARGEST_HARDWARE_BUFFER,
    }
}
