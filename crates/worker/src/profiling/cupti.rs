//! CUPTI's attachment to a serving rank: the watch of the driver calls that
//! make its threads wait for the device ([`super::blocking`]) and the record of
//! its streams' device time ([`cuda::cupti::activity`], reported as
//! `ForwardStats::stream_us`).
//!
//! A rank attaches CUPTI when it starts serving. The engine's built-in worker
//! hooks `uniserve.detach_cupti` and `uniserve.attach_cupti` (`POST
//! /admin/cupti`) detach and attach it while the rank runs no batch
//! (`crate::service`). Detached, CUPTI has no callback subscription and no
//! activity record enabled: the threads' waits for the device are neither
//! counted nor fail a serving call, and the streams' device time is not
//! reported. The library stays loaded (`cuda::cupti`: removing it leaves the
//! device work submitted afterwards waiting forever). A profiler that holds
//! CUPTI's one subscription (Nsight Systems) keeps the rank detached.

use cuda::cupti::BlockingWatch;
use cuda::cupti::activity::StreamActivity;

use crate::Result;

/// The rank's CUPTI attachment.
pub(crate) struct Cupti {
    /// The rank's streams by name (`device`, `copy`), whose device time an
    /// attachment records.
    streams: Vec<(&'static str, cuda::Stream)>,
    attached: Option<Attached>,
}

/// What an attachment holds. The activity record drops before the watch.
struct Attached {
    /// `None` when CUPTI refused the activity records.
    activity: Option<StreamActivity>,
    watch: BlockingWatch,
}

impl Cupti {
    /// A detached attachment that records `streams` once attached.
    pub(crate) fn new(streams: Vec<(&'static str, cuda::Stream)>) -> Self {
        Self {
            streams,
            attached: None,
        }
    }

    /// Attach CUPTI, unless attached. Fails when `libcupti.so` is missing or a
    /// profiler holds CUPTI's subscription. The streams' device time is
    /// recorded unless CUPTI refuses the activity records, which is logged.
    pub(crate) fn attach(&mut self) -> Result<()> {
        if self.attached.is_some() {
            return Ok(());
        }
        let watch = BlockingWatch::start()?;
        let activity =
            StreamActivity::start(self.streams.iter().map(|(name, stream)| (*name, stream)))
                .inspect_err(|error| {
                    tracing::warn!(
                        "the device time of the rank's streams is not recorded: {error}"
                    );
                })
                .ok();
        self.attached = Some(Attached { activity, watch });
        tracing::info!("CUPTI is attached ({})", cuda::cupti::library());
        Ok(())
    }

    /// Detach CUPTI, unless detached: the activity records stop and the
    /// buffered ones are delivered, their time travelling with the next
    /// result; then the subscription ends.
    pub(crate) fn detach(&mut self) -> Result<()> {
        let Some(Attached { activity, watch }) = self.attached.take() else {
            return Ok(());
        };
        drop(activity);
        drop(watch);
        tracing::info!("CUPTI is detached");
        Ok(())
    }

    /// The attachment as the hooks answer it: whether CUPTI is attached and
    /// whether it records the streams' device time.
    pub(crate) fn state(&self) -> serde_json::Value {
        serde_json::json!({
            "attached": self.attached.is_some(),
            "stream_activity": self
                .attached
                .as_ref()
                .is_some_and(|attached| attached.activity.is_some()),
        })
    }
}
