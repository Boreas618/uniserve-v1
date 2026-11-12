//! The lifecycle's work between epochs: a drain that parks every scheduler at its boundary and
//! hands the prefix index over, the promotion that makes a parked engine serve, and admission.

use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};

use super::MainProcess;

/// A drain whose schedulers have not all parked yet.
pub(super) struct Draining {
    /// The epoch `Drained` names.
    epoch: u64,
    /// The schedulers still to respond, in order.
    responses:
        Vec<crossbeam_channel::Receiver<std::result::Result<scheduler::ParkedCache, String>>>,
    /// What the ones that responded parked with, in the schedulers' order.
    parked: Vec<scheduler::ParkedCache>,
    /// When the drain began, which its bound is measured from.
    since: Instant,
}

impl Draining {
    /// The instant a scheduler that has not parked fails the drain.
    pub(super) fn bound(&self, stall: Duration) -> Instant {
        self.since + stall
    }
}

/// The longest a promoted engine's schedulers hold the requests that continue from the drained
/// engine's pages while its index is adopted: the index is adopted in the same command, so the
/// bound only matters when a scheduler never responds; the requests then compute what the pages
/// held.
const INDEX_WAIT: Duration = Duration::from_secs(30);

impl MainProcess {
    /// Begins the drain of `epoch`: every scheduler parks at the boundary it reaches, and the
    /// process goes on living.
    ///
    /// A scheduler parks once its migratable requests ended with their state kept and the others
    /// finished, and executes nothing more. What it parks with is kept, so a promotion that makes
    /// this engine serve again resumes it in place. Weights and storage stay mapped and the
    /// threads stay up: a drain is not an exit.
    pub(super) fn drain(&mut self, epoch: u64, discarded: Option<u64>) -> Result<()> {
        // An update cannot cross the drain: what a preparation built is given back now rather
        // than left for a boundary that cannot come, and the daemon is told.
        if let Some(generation) = discarded {
            self.discard_generation(generation);
            let report = self
                .state
                .rolled_back("the engine drained before the generation was switched to".to_owned());
            self.send_report(report);
        }
        // The daemon's client reads this before the first request ends migratable, on the same
        // connection, so a request that moves never finds this engine serving: it waits for the
        // engine that serves next rather than being submitted to one that admits nothing.
        self.announce(false)?;
        // A scheduler parks once its requests have ended, and a request ends only when the
        // daemon acknowledges it through this thread, so the responses are read as they wake it:
        // waiting here would hold the acknowledgement the park is waiting for.
        self.draining = Some(Draining {
            epoch,
            responses: self
                .schedulers
                .iter()
                .map(|scheduler| scheduler.park(false))
                .collect(),
            parked: Vec::with_capacity(self.schedulers.len()),
            since: Instant::now(),
        });
        self.continue_drain()
    }

    /// Takes the responses of the drain in flight, and acknowledges it once every scheduler
    /// parked.
    pub(super) fn continue_drain(&mut self) -> Result<()> {
        let bound = self.watchdog.stall_bound();
        let Some(mut draining) = self.draining.take() else {
            return Ok(());
        };
        while let Some(response) = draining.responses.first() {
            match response.try_recv() {
                Ok(Ok(cache)) => {
                    draining.parked.push(cache);
                    draining.responses.remove(0);
                }
                Ok(Err(reason)) => bail!("draining the engine: a scheduler did not park: {reason}"),
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if draining.since.elapsed() > bound {
                        bail!("draining the engine: a scheduler did not park within {bound:?}");
                    }
                    self.draining = Some(draining);
                    return Ok(());
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    bail!("draining the engine: a scheduler is gone");
                }
            }
        }
        // The index each scheduler parked with goes with the acknowledgement: the engine that
        // maps this one's cache segments reads the pages it names. The parked caches are kept
        // as well, for a promotion that makes this engine serve again.
        let index = self
            .schedulers
            .iter()
            .zip(&draining.parked)
            .flat_map(|(scheduler, parked)| parked.clone().into_indexes(&scheduler.cache_workers()))
            .collect();
        self.parked = Some(draining.parked);
        let (report, serves_again) = self.state.drained(draining.epoch, index);
        self.send_report(report);
        self.announce(false)?;
        // A promotion that overtook the drain serves this engine again, now that it can.
        if serves_again {
            self.promote(None)?;
        }
        Ok(())
    }

    /// The promotion: this parked engine serves.
    ///
    /// An engine that parked at its start holds no request and no index: with a `handoff` its
    /// schedulers adopt the index of the engine that drained - whose pages lie in the cache
    /// segments this one maps - before they admit, and the requests that migrate continue from
    /// those pages; without one it is a standby promoted after its failure domain was lost, and
    /// it computes what it is asked. An engine that parked by draining resumes the schedulers it
    /// parked with what they parked with, and forgets the pages its index names: another epoch
    /// ran on its cache's storage while it stood drained.
    pub(super) fn promote(&mut self, handoff: Option<proto::process::Handoff>) -> Result<()> {
        let drained = self.parked.is_some();
        let adopts = handoff.as_ref().filter(|_| !drained);
        if let Some(handoff) = adopts {
            // The scheduler's own promotion takes `recompute` and `grows` only with segments its
            // predecessor lent, and a drain lends none: the index below is what is adopted.
            tracing::info!(
                groups = handoff.index.len(),
                recompute = handoff.recompute,
                grows = handoff.grows,
                "the promoted engine adopts the drained engine's index"
            );
            for scheduler in &self.schedulers {
                scheduler.await_index(INDEX_WAIT);
            }
        }
        // An engine that parked with its graph pools' storage given back takes it again, at the
        // addresses the graphs hold, before anything is admitted.
        if std::mem::take(&mut self.graphs_suspended) {
            self.data
                .restore_graphs()
                .context("restoring the promoted engine's graph pools")?;
            tracing::info!("the promoted engine restored its graph storage");
        }
        // The generation the schedulers stamp this engine's batches with: the one it holds.
        self.record_generation(self.state.generations.current());
        self.open_admission()
            .context("promoting this engine in place")?;
        if drained {
            // The daemon moves requests only after the acknowledgement below, so every request
            // comes after this command.
            for scheduler in &self.schedulers {
                scheduler.flush_cache();
            }
        }
        if let Some(handoff) = adopts {
            self.adopt_index(&handoff.index)?;
        }
        let report = self.state.promoted();
        self.send_report(report);
        // The request link says so too: the daemon's client waits on it to move its requests.
        self.announce(true)
    }

    /// Adopts the index the drained engine parked with, each worker group's in the schedulers of
    /// that group. A scheduler whose group the index names nothing of adopts none, and admits
    /// the requests it held for it.
    ///
    /// Each adoption is bounded by the watchdog's stall bound, as a park is.
    fn adopt_index(&mut self, index: &[proto::request::GroupIndex]) -> Result<()> {
        let bound = self.watchdog.stall_bound();
        let responses: Vec<_> = self
            .schedulers
            .iter()
            .map(|scheduler| {
                // Each of the scheduler's groups adopts the pages the drained engine's group
                // held: a prefill/decode scheduler's groups are two.
                let parked = scheduler
                    .cache_workers()
                    .into_iter()
                    .filter_map(|(group, _)| {
                        let indexed = index.iter().find(|indexed| indexed.worker_group == group)?;
                        Some((group, scheduler::ParkedCache::from_index(indexed)))
                    })
                    .collect();
                scheduler.inherit(parked)
            })
            .collect();
        for response in responses {
            match response.recv_timeout(bound) {
                Ok(Ok(())) => {}
                // A page the scheduler does not adopt is computed again: the requests go on.
                Ok(Err(reason)) => {
                    tracing::warn!(%reason, "a scheduler did not adopt the drained engine's index");
                }
                Err(error) => {
                    bail!(
                        "adopting the drained engine's index: no response within {bound:?}: {error}"
                    )
                }
            }
        }
        Ok(())
    }

    /// Opens admission: what the ranks hold to serve is recorded in their devices' usage, the
    /// schedulers resume, and the serving path warms up.
    ///
    /// Every bucket a batch can be formed into is captured by now: a start ends, and a
    /// promotion happens, only after the captures of the generation that executes.
    pub(super) fn open_admission(&mut self) -> Result<()> {
        tracing::info!("opening admission");
        // A rank past its quota does not serve.
        self.data
            .reserve_serving()
            .context("reserving what the ranks hold in their devices' quotas")?;
        self.resume_schedulers()?;
        // The serving path's warm-up submits ordinary requests straight to each scheduler, so
        // it starts here, where every bucket its batches replay is captured. Only an engine's
        // first opening starts it; a reopening after a drain finds it started.
        for scheduler in &self.schedulers {
            scheduler.warm_serving_path();
        }
        Ok(())
    }

    /// Resumes every scheduler a drain parked, with the cache it parked with: the pages it still
    /// holds are reused and the requests the drain ended are resubmitted by the daemon. A
    /// scheduler that never parked has nothing to resume.
    fn resume_schedulers(&mut self) -> Result<()> {
        let Some(parked) = self.parked.take() else {
            return Ok(());
        };
        let bound = self.watchdog.stall_bound();
        for (scheduler, cache) in self.schedulers.iter().zip(parked) {
            // A resumed engine recomputes from the tokens the daemon resubmits rather than
            // lending from a predecessor: it is the same engine, and the requests it ended at
            // the drain are new requests to it.
            scheduler
                .promote(cache, true, false)
                .context("resuming a scheduler")?;
            // The promotion states the cache; it is the resume that ends the park. A scheduler
            // left parked takes requests into its queue and admits none of them.
            match scheduler.resume(true).recv_timeout(bound) {
                Ok(Ok(())) => {}
                Ok(Err(reason)) => bail!("resuming a scheduler: {reason}"),
                Err(error) => bail!("resuming a scheduler: no response within {bound:?}: {error}"),
            }
        }
        Ok(())
    }
}
