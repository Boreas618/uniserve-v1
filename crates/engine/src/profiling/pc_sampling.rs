//! CUPTI's PC sampling as a rank runs it.
//!
//! CUPTI attributes the samples of a CUDA graph's kernels to their program counters only when PC
//! sampling was enabled when the graph was captured, and stayed enabled since; otherwise it counts
//! them as samples of non-user kernels, without records. A rank captures its graphs at warmup, so
//! a deployment that profiles them (`profiling_config.graph_pc_sampling`) has every rank enable PC
//! sampling, stopped, before it captures, and keep it enabled for as long as it serves
//! ([`enable`]); sessions only configure, start and stop the rank's sampler ([`of`]). A kernel
//! generation that switches the option captures every graph again: the process switches CUPTI's
//! mode first ([`switch`]), and back when the generation is given back. CUPTI also refuses to enable an
//! activity record once PC sampling is enabled, unless the record's kinds were enabled before it,
//! so the rank first holds the kinds of its always-on stream record enabled for its life; the
//! stream record and the timelines that trace graphs by node start later beside it. While it is
//! enabled CUPTI refuses PM sampling, the range profiler and whole-graph activity records, which
//! the rank refuses at the session's start
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
use std::sync::{Arc, Mutex};

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

/// The stream record's activity kinds, held enabled from before the first rank's sampler for as
/// long as a rank keeps one.
static KINDS: Mutex<Option<KindsHold>> = Mutex::new(None);

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
    {
        let mut kinds = lock(&KINDS);
        if kinds.is_none() {
            let hold = KindsHold::take().map_err(|error| {
                Error::Unsupported(format!("the activity records beside PC sampling: {error}"))
            })?;
            *kinds = Some(hold);
        }
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

/// Disables the PC sampling of the rank of `device`, once no session samples with it: its
/// sampler drops, which disables it on the device's context, and the activity kinds held for
/// the samplers are released with the last one. A rank that keeps no sampler is left as it is.
pub(crate) fn disable(device: DeviceId) {
    let sampler = lock(&RANKS)
        .as_mut()
        .and_then(|ranks| ranks.remove(&device));
    drop(sampler);

    let none = lock(&RANKS).as_ref().is_none_or(HashMap::is_empty);
    if none {
        lock(&KINDS).take();
    }
    tracing::info!(
        device = device.0,
        "disabled PC sampling for the rank's CUDA graphs"
    );
}

/// Switches CUPTI's PC sampling of the CUDA graphs of `devices` to `on`, for a kernel generation
/// that captures every graph of the devices again under the new mode, or for the generation it
/// gives back: on, every device's sampler is enabled, stopped ([`enable`]), and `counters`, the
/// process's always-on counters of `streams`, record a graph's launch by its nodes, since CUPTI
/// refuses whole-graph records beside PC sampling; off, every device's sampler is disabled
/// ([`disable`]) and the counters record a graph's launch whole again. Each executable then
/// serves in the tracing mode it is instantiated in again (`super::graph_renewals`).
///
/// The caller has no profiling session running on any of the devices: a session's collectors
/// were chosen against the mode it started under.
///
/// # Errors
///
/// CUPTI's refusal of a device's sampler: the devices enabled before it are disabled again, and
/// the counters record as they did.
pub(crate) fn switch(
    devices: &[DeviceId],
    on: bool,
    (counters, streams): (
        &Mutex<Option<super::counters::Counters>>,
        &[(&'static str, platform::Stream)],
    ),
) -> Result<()> {
    let mut held = lock(counters);
    let counted = held.is_some();
    // The counters' record of each graph's launch is refused beside PC sampling, and restarted
    // in the mode the samplers leave.
    drop(held.take());

    let mut switched = Ok(());
    if on {
        super::set_whole_by_default(false);
        for (index, device) in devices.iter().enumerate() {
            let enabled = PrimaryContext::retain(*device)
                .map_err(Error::from)
                .and_then(|context| enable(&context));
            if let Err(error) = enabled {
                for earlier in &devices[..index] {
                    disable(*earlier);
                }
                switched = Err(error);
                break;
            }
        }
    } else {
        for device in devices {
            disable(*device);
        }
    }

    let sampling = devices.iter().any(|device| of(*device).is_some());
    if counted {
        *held = super::counters::Counters::start(streams, !sampling);
    }
    switched
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::{Arc, Mutex};

    use platform::{DeviceAllocation, DeviceId, DevicePtr, Stream};
    use tensor::DType;

    use super::{of, switch};

    const ROWS: usize = 4096;
    const HIDDEN: usize = 4096;

    /// A graph of `ROWS` normalizations over `x`, captured on `stream` under `kernels` in
    /// `region`, as a runner's capture records one.
    fn captured(
        stream: &Stream,
        kernels: &Arc<ops::Generation>,
        (region, x, gain): (&DeviceAllocation, &DeviceAllocation, &DeviceAllocation),
    ) -> platform::GraphExec {
        let on_device = |allocation: &DeviceAllocation, dims: &[usize]| {
            ops::retained(allocation.address(), dims, DType::F32, &ops::cuda(0)).unwrap()
        };
        let context = ops::Context::new(
            0,
            stream.clone(),
            Arc::clone(kernels),
            platform::DeviceRange::new(DevicePtr::from_address(region.address()), region.len()),
        );
        let capture = platform::Capture::begin(stream, platform::CaptureMode::Relaxed).unwrap();
        {
            let _guard = context.enter();
            let x = on_device(x, &[ROWS, HIDDEN]);
            let gain = on_device(gain, &[HIDDEN]);
            for _ in 0..8 {
                nn::functional::rms_norm(&x, &gain, 1e-6, false, DType::F32).unwrap();
            }
        }
        capture.end().unwrap().instantiate().unwrap()
    }

    /// The program counters a session of the device's sampler attributes over 64 replays of
    /// `graph`.
    fn sampled(device: DeviceId, stream: &Stream, graph: &platform::GraphExec) -> usize {
        let sampler = of(device).expect("the rank keeps a sampler");
        let mut sampler = service::lock(&sampler);
        sampler.take().unwrap();
        sampler.start().unwrap();
        for _ in 0..64 {
            graph.launch(stream).unwrap();
        }
        stream.synchronize().unwrap();
        sampler.stop().unwrap();
        sampler.take().unwrap().pcs.len()
    }

    /// A generation that switches PC sampling on captures its graphs under it, whose kernels'
    /// samples are attributed to program counters, while a graph captured before keeps
    /// replaying beside them. Switched off again, both graphs replay, the device keeps no
    /// sampler, and switching on once more enables one again, which attributes the graphs
    /// captured under it.
    #[test]
    #[ignore = "requires a GPU and the host's Build Server holding the rms_norm package"]
    fn a_switch_of_pc_sampling_attributes_the_graphs_captured_under_it() {
        let device = DeviceId(0);
        crate::profiling::counters::subscribe();
        let context = platform::PrimaryContext::retain(device).unwrap();
        let _current = context.enter().unwrap();

        // The kernels of the normalization, from the Build Server of `device-cases`.
        let meta = ops::Context::meta(None);
        meta.record_launches();
        {
            let _guard = meta.enter();
            let weight = |address, dims: &[usize]| {
                ops::retained(address, dims, DType::F32, &ops::meta()).unwrap()
            };
            nn::functional::rms_norm(
                &weight(1 << 40, &[ROWS, HIDDEN]),
                &weight(2 << 40, &[HIDDEN]),
                1e-6,
                false,
                DType::F32,
            )
            .unwrap();
        }
        let mut wanted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (operator, key, _, _) in meta.take_launches() {
            wanted.entry(operator.to_owned()).or_default().insert(key);
        }
        let socket = build_server::socket(
            &service::RuntimeDirectory::of_user().unwrap(),
            "device-cases",
        )
        .unwrap();
        let mut client = build_server::Client::connect(&socket).unwrap();
        let targets = [build_server::Target::arch(device.compute_capability().unwrap()).unwrap()];
        let (packages, routes) =
            crate::modules::load::select(&mut client, &targets, &Default::default(), &wanted)
                .unwrap();
        let entries =
            crate::modules::load::install(&mut client, device, &packages, &routes).unwrap();
        let kernels = Arc::new(ops::Generation::new(1, entries));

        let stream = Stream::new(device).unwrap();
        let region = DeviceAllocation::linear(device, 4 * ROWS * HIDDEN * 9).unwrap();
        let x = DeviceAllocation::linear(device, 4 * ROWS * HIDDEN).unwrap();
        let gain = DeviceAllocation::linear(device, 4 * HIDDEN).unwrap();
        let memory = (&region, &x, &gain);
        let counters = Mutex::new(None);

        let before = captured(&stream, &kernels, memory);
        switch(&[device], true, (&counters, &[])).unwrap();
        let under = captured(&stream, &kernels, memory);

        assert!(
            sampled(device, &stream, &under) > 0,
            "the graph captured under sampling"
        );
        // The graph captured before replays beside it under the sampler.
        sampled(device, &stream, &before);

        switch(&[device], false, (&counters, &[])).unwrap();
        assert!(of(device).is_none());
        for graph in [&before, &under] {
            graph.launch(&stream).unwrap();
        }
        stream.synchronize().unwrap();

        switch(&[device], true, (&counters, &[])).unwrap();
        let again = captured(&stream, &kernels, memory);
        assert!(
            sampled(device, &stream, &again) > 0,
            "sampling enabled again attributes the graphs captured under it"
        );
        switch(&[device], false, (&counters, &[])).unwrap();
    }
}
