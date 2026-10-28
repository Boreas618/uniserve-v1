//! CUPTI's PC sampling as a rank runs it.
//!
//! CUPTI attributes the samples of a CUDA graph's kernels to their program
//! counters only when PC sampling was enabled when the graph was captured,
//! and stayed enabled since; otherwise it counts them as samples of non-user
//! kernels, without records. A rank captures its graphs at warmup, so a
//! deployment that profiles them (`profiling_config.graph_pc_sampling`) has
//! every rank enable PC sampling, stopped, before it captures, and keep it
//! enabled for its life ([`enable`]); sessions only configure, start and
//! stop the rank's sampler ([`rank`]). CUPTI also refuses to enable an
//! activity record once PC sampling is enabled, unless the record's kinds
//! were enabled before it, so the rank first holds the kinds of its
//! always-on stream record enabled for its life; the stream record and
//! the timelines that trace graphs by node start later beside it. While it is enabled CUPTI refuses PM
//! sampling, the range profiler and whole-graph activity records, which
//! the leader refuses at the session request
//! (`proto::profile::SessionRequest::refused_beside_graph_pc_sampling`).
//! Without the option, a session enables a sampler of its own, which
//! attributes only kernels launched outside CUDA graphs.
//!
//! CUPTI keeps the samples in a hardware buffer of process host storage,
//! which its own thread empties; a buffer too small for the sampling rate
//! fills, and the samples are lost. [`hardware_buffer`] sizes it for a
//! period from measurement.

use std::sync::{Mutex, OnceLock};

use cuda::PrimaryContext;
use cuda::cupti::activity::KindsHold;
use cuda::cupti::pc_sampling::PcSampler;

use crate::{Error, Result};

/// The program counters one take of the sampler reads at most.
pub(crate) const MAX_PCS: usize = 10_000;

/// The rank's sampler, enabled for its life when the deployment profiles its
/// graphs' kernels.
static RANK: OnceLock<Mutex<PcSampler>> = OnceLock::new();

/// The stream record's activity kinds, held enabled from before the rank's
/// sampler for its life.
static KINDS: OnceLock<KindsHold> = OnceLock::new();

/// Enable PC sampling on `context`, stopped, for the rank's life, at the
/// default period, before the rank captures any CUDA graph.
///
/// # Errors
///
/// Fails when CUPTI refuses PC sampling, as when another profiler holds the
/// process's CUPTI subscription.
pub(crate) fn enable(context: &PrimaryContext) -> Result<()> {
    let period = proto::profile::PcSamplingOptions::default().period;
    let bytes = hardware_buffer(period).ok_or_else(|| {
        Error::Invalid(format!(
            "no hardware buffer holds PC sampling at period {period}"
        ))
    })?;
    let hold = KindsHold::take().map_err(|error| {
        Error::Unsupported(format!("the activity records beside PC sampling: {error}"))
    })?;
    let _ = KINDS.set(hold);
    let sampler =
        PcSampler::enable(context.handle(), period, MAX_PCS, Some(bytes)).map_err(|error| {
            Error::Unsupported(format!("PC sampling for the rank's graphs: {error}"))
        })?;
    RANK.set(Mutex::new(sampler))
        .map_err(|_| Error::State("the rank enabled PC sampling twice"))?;
    tracing::info!(
        period,
        hardware_buffer = bytes,
        "enabled PC sampling for the rank's CUDA graphs"
    );
    Ok(())
}

/// The rank's sampler, when the rank keeps one.
pub(crate) fn rank() -> Option<&'static Mutex<PcSampler>> {
    RANK.get()
}

/// The hardware buffer, in bytes, that holds PC samples at `period` when
/// every SM is busy, or `None` when no buffer of up to CUPTI's 512 MiB does.
///
/// Measured on GB200 with `nvidia-cuda-cupti` 13.0 with every SM busy: a
/// size holds when no window of 0.6 s filled it. Each period takes twice
/// the smallest size that held three windows, capped at CUPTI's 512 MiB,
/// and the size taken held ten more (`docs/profiling.md` states the table).
/// Below period 12 even 512 MiB fills.
pub(crate) fn hardware_buffer(period: u32) -> Option<usize> {
    const MIB: usize = 1 << 20;
    match period {
        19.. => Some(2 * MIB),
        18 => Some(4 * MIB),
        17 => Some(8 * MIB),
        15 | 16 => Some(32 * MIB),
        14 => Some(128 * MIB),
        13 => Some(256 * MIB),
        12 => Some(512 * MIB),
        _ => None,
    }
}
