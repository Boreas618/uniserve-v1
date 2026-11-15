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
//! empties; a buffer too small for the sampling rate fills, and the samples are lost. A session
//! states the buffer with its period (`pc_sampling.hardware_buffer_mib`, CUPTI's own largest by
//! default), and a read whose buffer filled tells the session the period is too short for the
//! device and the buffer, which releases the collector as failed (`super::collectors`).

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

/// Enables PC sampling on `context`, stopped, for the rank's life, before the rank captures any
/// CUDA graph: at a session's default period, and into the smallest hardware buffer a session may
/// state, since a stopped sampler gathers nothing and each session sets its own period and buffer
/// at its first arm.
///
/// # Errors
///
/// Fails when CUPTI refuses PC sampling, as when another profiler holds the process's CUPTI
/// subscription.
pub(crate) fn enable(context: &PrimaryContext) -> Result<()> {
    let options = proto::profile::PcSamplingOptions {
        hardware_buffer_mib: *proto::profile::PcSamplingOptions::HARDWARE_BUFFERS_MIB.start(),
        ..proto::profile::PcSamplingOptions::default()
    };
    let (period, bytes) = (options.period, options.hardware_buffer());
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
