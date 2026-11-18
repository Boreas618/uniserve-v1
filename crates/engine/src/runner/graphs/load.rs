//! The expert load of a runner: device counters of the tokens each routed layer's experts take,
//! and their periodic read to the host.
//!
//! The runner owns one device allocation of `Σ experts` I64 counters, one `[experts]` run per
//! routed layer in module-path order, zeroed on the runner's stream when it is made. The launching
//! context lends each run to its layer under the layer's module path (`ops::moe::ExpertLoad`), so
//! the top-k of every serving forward, eager or replayed from a graph captured beside it, adds
//! each live token's routes into its layer's run on the device. Nothing of a step reads the
//! counters on the host.
//!
//! [`Counters::take`] reads them without waiting for the device: it copies the counters behind
//! everything the stream holds into page-locked memory and records an event behind the copy, and
//! the next take, once that event has completed, states what each counter gained since the read
//! before. A take whose copy has not landed yet states nothing and leaves the copy in flight, so
//! the counts reach the host one period after the steps that made them, and none is lost.

use std::collections::BTreeMap;

use ops::moe::ExpertLoad;
use platform::{DeviceAllocation, Event, EventFlags, PinnedBuffer, Stream};
use tensor::DType;

use crate::worker::{Error, Result};

/// The bytes of one counter.
const COUNTER: usize = 8;

/// The expert load counters of one runner (module documentation).
pub(super) struct Counters {
    /// Each routed layer's module path and its experts, in the order of the runs.
    layers: Vec<(String, usize)>,
    /// The counters on the device, `Σ experts` I64.
    device: DeviceAllocation,
    /// Where a read copies them.
    host: PinnedBuffer,
    /// Recorded behind each read's copy.
    landed: Event,
    /// Whether a read's copy was issued that no take has consumed yet.
    reading: bool,
    /// Every counter's value at the last read a take consumed.
    taken: Vec<u64>,
    /// The stream the counters are written and read on: the runner's.
    stream: Stream,
}

impl Counters {
    /// Zeroed counters for `layers`, each routed layer's module path with its expert count, on
    /// the device of `stream`, which the runner's forwards launch on, and the load that lends
    /// them.
    ///
    /// # Errors
    ///
    /// The driver's refusal of an allocation, of the fill or of the event; [`Error::Invalid`]
    /// for no layer, or a layer of no expert.
    pub(super) fn new(
        layers: &BTreeMap<String, u32>,
        stream: &Stream,
    ) -> Result<(Self, ExpertLoad)> {
        if layers.is_empty() || layers.values().any(|&experts| experts == 0) {
            return Err(Error::Invalid(format!(
                "an expert load counts the experts of routed layers, and {} layers of {:?} \
                 experts were given",
                layers.len(),
                layers.values().collect::<Vec<_>>()
            )));
        }

        let layers: Vec<(String, usize)> = layers
            .iter()
            .map(|(module, &experts)| (module.clone(), experts as usize))
            .collect();
        let total: usize = layers.iter().map(|(_, experts)| experts).sum();
        let bytes = total * COUNTER;

        let device = DeviceAllocation::linear(stream.device(), bytes)?;
        // SAFETY: the range is the allocation just made, which this value keeps; the stream
        // orders the fill before every forward that adds into it.
        unsafe { platform::memset_d8(device.address(), 0, bytes, stream) }?;
        let host = PinnedBuffer::new(stream.device(), bytes)?;

        // Each layer's run, as the external tensor its top-k adds into.
        let mut lent = BTreeMap::new();
        let mut offset = 0;
        for (module, experts) in &layers {
            let address = device.address() + (offset * COUNTER) as u64;
            // SAFETY: the run lies inside the allocation this value keeps, which outlives the
            // graphs that address it: the runner's graphs own both, and drop the counters only
            // after their stream ran what it holds (`Drop`).
            let counters = unsafe {
                ops::external(
                    address,
                    *experts,
                    DType::I64,
                    &ops::cuda(stream.device().0 as usize),
                )
            }?;
            lent.insert(module.clone(), counters);
            offset += experts;
        }

        Ok((
            Self {
                layers,
                device,
                host,
                landed: Event::new(EventFlags::default()),
                reading: false,
                taken: vec![0; total],
                stream: stream.clone(),
            },
            ExpertLoad::new(lent),
        ))
    }

    /// The tokens each layer's experts took between the two latest reads a take consumed, by
    /// layer, of the layers whose experts took any; then the next read is issued. Empty while the
    /// read before has not landed, which stays in flight for the next take. Called on the thread
    /// that issues the runner's work, between batches.
    ///
    /// # Errors
    ///
    /// The driver's refusal of the event's query, the copy or the event.
    pub(super) fn take(&mut self) -> Result<BTreeMap<String, Vec<u64>>> {
        let mut gained = BTreeMap::new();

        if self.reading {
            if !self.landed.query()? {
                return Ok(gained);
            }
            self.reading = false;
            gained = self.gained();
        }

        // SAFETY: the source is the counters on the stream's device and the destination this
        // value's page-locked memory of the same length; no take reads it before the event
        // recorded behind the copy has completed, and no other copy writes it meanwhile.
        unsafe {
            platform::memcpy_async(
                self.host.address(),
                self.device.address(),
                self.host.len(),
                &self.stream,
            )
        }?;
        self.landed.record(&self.stream)?;
        self.reading = true;

        Ok(gained)
    }

    /// What each counter gained since the read the last take consumed, from the read that
    /// landed, which becomes that read.
    fn gained(&mut self) -> BTreeMap<String, Vec<u64>> {
        // SAFETY: the copy into the memory completed (`take` queried its event) and no copy is
        // in flight into it.
        let bytes = unsafe { std::slice::from_raw_parts(self.host.as_ptr(), self.host.len()) };
        let read = bytes
            .as_chunks::<COUNTER>()
            .0
            .iter()
            .map(|counter| u64::from_ne_bytes(*counter));

        let deltas: Vec<u64> = read
            .zip(self.taken.iter_mut())
            .map(|(now, before)| {
                // A counter only grows on the device; the read is its value at the copy.
                let delta = now.saturating_sub(*before);
                *before = now;
                delta
            })
            .collect();

        let mut gained = BTreeMap::new();
        let mut offset = 0;
        for (module, experts) in &self.layers {
            let tokens = &deltas[offset..offset + experts];
            if tokens.iter().any(|&count| count > 0) {
                gained.insert(module.clone(), tokens.to_vec());
            }
            offset += experts;
        }
        gained
    }
}

impl Drop for Counters {
    fn drop(&mut self) {
        // The counters' memory is freed with this value: every forward and copy that addresses
        // it must have run first.
        if let Err(error) = self.stream.synchronize() {
            tracing::warn!(%error, "the stream of a runner's expert load could not be waited for");
        }
    }
}
