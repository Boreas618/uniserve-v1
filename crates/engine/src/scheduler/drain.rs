//! Leaving the engine (`specs/scheduler-design.md` §3.8): a drain admits nothing, ends every
//! queued request as migratable at once, lets the running ones end by themselves until its
//! deadline and ends the rest as migratable then, a video request at a step boundary. A drain to
//! an epoch of another layout also gives the device memory back as requests end, by `Resize`.
//!
//! The drain is answered by `reply_at_boundary` (`mod.rs`) once no request remains and, with
//! `shrink`, no `Detach` and no `Resize` of the scheduler is still to be taken or answered and
//! every device region answered a `Resize` submitted after the last `Detach` was answered.
//! `shrink_regions` therefore asks for a region's units while something there is free beyond
//! what the running requests may still take before they end, or still retiring once no request
//! remains, and once after each `Detach` answered, whose units the capacity words read with its
//! answer may not show yet; the shrink ends once nothing more frees.

use std::time::Instant;

use crate::batch::Phase;
use crossbeam_channel::Sender;
use proto::request::MigrationCause;
use storage::runtime::{self, Tier};

use super::Scheduler;
use super::update::migratable;

impl Scheduler {
    /// Starts a drain for `cause` (SD §3.8): from here on nothing is admitted; every queued
    /// request leaves at once as migratable, holding nothing worth keeping, and the daemon
    /// resubmits it; the running requests end by themselves or at `deadline`. With `shrink`
    /// (a drain to an epoch of another layout) the device memory is given back as requests end.
    /// `reply` is answered once no request remains.
    pub(super) fn drain(
        &mut self,
        cause: MigrationCause,
        deadline: Instant,
        shrink: bool,
        reply: Sender<()>,
    ) {
        self.draining = Some((reply, cause, deadline, shrink));
        self.detached.fill(false);
        self.drain_due(Instant::now());
    }

    /// A drain's work at every pass of the loop: every queued request leaves as migratable, a
    /// request that arrived since the drain began included; once the deadline has passed every
    /// running request does too, but a video request whose denoising is fully scheduled, which
    /// runs to its end; and with `shrink` the device regions give back what came free.
    pub(super) fn drain_due(&mut self, now: Instant) {
        let Some(&(_, cause, deadline, shrink)) = self.draining.as_ref() else {
            return;
        };

        // A queued video request leaves with no stored step, even one its admission restored
        // (SD §7.3 "Preemption").
        let queued: Vec<_> = self.waiting.iter().copied().collect();
        for id in queued {
            if let Some(request) = self.requests.get_mut(&id) {
                request.scheduled[Phase::Denoise.index()] = 0;
                request.completed[Phase::Denoise.index()] = 0;
            }
            self.finish(id, migratable(cause));
        }

        if now >= deadline {
            for id in self.running.clone() {
                let Some(request) = self.requests.get(&id) else {
                    continue;
                };
                if !request.runs_to_end(&self.groups) {
                    self.finish(id, migratable(cause));
                }
            }
        }

        if shrink {
            self.shrink_regions();
        }
    }

    /// Gives a drain's device memory back (SD §3.8): for each device region of this queue pair
    /// with no `Resize` held or outstanding, `Resize { region, units }` giving back the units free
    /// or evictable now that no running request may still take before it ends
    /// (`Scheduler::units_to_end`), when there are such units; once no request remains, also
    /// while units still retire, which come free once the batches that may read them have ended;
    /// and when a `Detach` was answered since the region's last `Resize` (`Scheduler::detached`).
    /// A running request steps on to its end, and gives its units back as it leaves. The buffer
    /// pool evicts in the order this queue pair installed, the whole of each region's order
    /// while it shrinks (`install_victims`), and never takes a private or pinned block, so a
    /// shrink stops short while requests run and each `Resized` lets the next one take what has
    /// come free since. Each `Resized` is reported as `Output::Shrunk` (`answered`).
    fn shrink_regions(&mut self) {
        let ended = self.requests.is_empty();
        let kept = self.units_to_end();
        for (index, number) in self.device_regions() {
            if self.resizing_region(number) {
                continue;
            }
            let available = self.available.get(index).copied().unwrap_or(0);
            let free = available.saturating_sub(kept.get(index).copied().unwrap_or(0));
            let retiring = self.retiring.get(index).copied().unwrap_or(0);
            let detached = self.detached.get(index).copied().unwrap_or(false);
            if free == 0 && (retiring == 0 || !ended) && !detached {
                continue;
            }

            let capacity = self.capacity.get(index).copied().unwrap_or(0);
            self.submit(runtime::Request::Resize {
                region: number,
                units: capacity.saturating_sub(free),
            });
            if let Some(detached) = self.detached.get_mut(index) {
                *detached = false;
            }
        }
    }

    /// The device regions this queue pair uses: each one's index in the session's regions and
    /// the number the buffer pool names it by.
    pub(super) fn device_regions(&self) -> Vec<(usize, u8)> {
        self.registers
            .capacity
            .iter()
            .filter_map(|&(number, _)| {
                let index = self.region_index(number)?;
                let (_, region) = self.regions.get(index)?;
                (region.tier == Tier::Device).then_some((index, number))
            })
            .collect()
    }

    /// Whether a `Resize` of region `number` is held or outstanding: one at a time per region.
    fn resizing_region(&self, number: u8) -> bool {
        let of = |request: &runtime::Request| match request {
            runtime::Request::Resize { region, .. } => *region == number,
            _ => false,
        };
        self.held.iter().any(of) || self.outstanding.values().any(|(_, request, _)| of(request))
    }
}

#[cfg(test)]
mod tests {
    //! Drains over the scheduler of `scheduler::tests::bench`, whose four regions, numbered 4 to
    //! 7, are device regions of 64, 64, 64 and 16 units.

    use std::time::{Duration, Instant};

    use crossbeam_channel::Receiver;
    use proto::request::{Event, MigrationCause, RequestId};
    use storage::runtime::{self, Completion};

    use super::super::tests::{A, Bench, OWNER_A, bench};
    use super::super::{Output, RequestStatus};

    fn scheduler() -> (Bench, Receiver<Output>) {
        let mut bench = bench(4096);
        let (output, outputs) = crossbeam_channel::unbounded();
        bench.scheduler.output = output;
        (bench, outputs)
    }

    /// The requests that left as migratable for `Drain`, in order.
    fn migrated(outputs: &Receiver<Output>) -> Vec<RequestId> {
        outputs
            .try_iter()
            .filter_map(|output| match output {
                Output::Event(
                    id,
                    Event::Migratable {
                        cause: MigrationCause::Drain,
                        denoised_steps: 0,
                        ..
                    },
                ) => Some(id),
                _ => None,
            })
            .collect()
    }

    /// The requests the scheduler wrote into the submission queue.
    fn written(bench: &mut Bench) -> Vec<runtime::Request> {
        bench.scheduler.post().unwrap();
        bench
            .take()
            .into_iter()
            .map(|(_, request)| request)
            .collect()
    }

    /// A queued request leaves at once, one that arrives during the drain at the next pass, and a
    /// running one at the deadline.
    #[test]
    fn a_drain_ends_queued_requests_at_once_and_running_ones_at_its_deadline() {
        let (mut bench, outputs) = scheduler();
        bench.attach(&[(A, OWNER_A, 3)]);
        let scheduler = &mut bench.scheduler;
        let queued = RequestId(1);
        let running = RequestId(2);
        let mut request = scheduler.requests[&queued].clone();
        request.id = running;
        request.status = RequestStatus::Running;
        request.queued = false;
        request.attachments.clear();
        scheduler.requests.insert(running, request);
        scheduler.waiting.push_back(queued);
        scheduler.running.push(running);

        let now = Instant::now();
        let (reply, _replied) = crossbeam_channel::bounded(1);
        scheduler.drain(
            MigrationCause::Drain,
            now + Duration::from_secs(60),
            false,
            reply,
        );
        assert_eq!(migrated(&outputs), [queued]);
        assert!(scheduler.requests.contains_key(&running));

        // A request that arrives while the drain runs leaves at the next pass.
        let mut arrived = scheduler.requests[&running].clone();
        arrived.id = RequestId(3);
        arrived.status = RequestStatus::Waiting;
        arrived.queued = true;
        scheduler.requests.insert(arrived.id, arrived);
        scheduler.waiting.push_back(RequestId(3));
        scheduler.drain_due(now);
        assert_eq!(migrated(&outputs), [RequestId(3)]);

        scheduler.drain_due(now + Duration::from_secs(61));
        assert_eq!(migrated(&outputs), [running]);
        assert!(scheduler.requests.is_empty());
    }

    /// A drain to another layout gives back what is free in each device region, keeping what the
    /// live requests hold, one `Resize` per region at a time; each `Resized` is reported, and
    /// once nothing more frees no further `Resize` is asked for.
    #[test]
    fn a_shrinking_drain_gives_back_what_came_free_until_nothing_more_does() {
        let (mut bench, outputs) = scheduler();
        let scheduler = &mut bench.scheduler;
        scheduler.available = vec![10, 0, 0, 0];
        let (reply, _replied) = crossbeam_channel::bounded(1);
        scheduler.drain(MigrationCause::Drain, Instant::now(), true, reply);
        scheduler.drain_due(Instant::now());

        let resize = runtime::Request::Resize {
            region: 4,
            units: 54,
        };
        let taken = {
            bench.scheduler.post().unwrap();
            bench.take()
        };
        assert_eq!(
            taken
                .iter()
                .map(|(_, request)| request.clone())
                .collect::<Vec<_>>(),
            [resize]
        );

        // Units still retiring once no request remains come free soon: the region is asked again.
        bench.scheduler.retiring[1] = 3;
        bench.scheduler.drain_due(Instant::now());
        assert_eq!(
            written(&mut bench),
            [runtime::Request::Resize {
                region: 5,
                units: 64,
            }]
        );

        bench.complete(
            taken[0].0,
            &Completion::Resized {
                units: 54,
                evicted: Vec::new(),
            },
        );
        while let Some((id, owner, request, completion)) =
            bench.scheduler.next_completion().unwrap()
        {
            bench.scheduler.answered(id, owner, request, completion);
        }
        assert_eq!(bench.scheduler.capacity[0], 54);
        assert!(
            outputs
                .try_iter()
                .any(|output| matches!(output, Output::Shrunk(4, 54)))
        );

        // Nothing is free or retiring in region 4 any more.
        bench.scheduler.available[0] = 0;
        bench.scheduler.retiring[1] = 0;
        bench.scheduler.drain_due(Instant::now());
        assert!(written(&mut bench).is_empty());
    }

    /// The buffer pool writes a `Detach`'s answer before the capacity words that show what it
    /// gave back: a shrinking drain whose last request was detached is answered only once every
    /// device region answered a `Resize` submitted after that answer (SD §3.8).
    #[test]
    fn a_shrinking_drain_answers_after_a_resize_that_follows_the_last_detach() {
        let (mut bench, _outputs) = scheduler();
        bench.attach(&[(A, OWNER_A, 3)]);
        let scheduler = &mut bench.scheduler;
        let request = scheduler.requests.get_mut(&RequestId(1)).unwrap();
        request.status = RequestStatus::Running;
        request.queued = false;
        scheduler.running.push(RequestId(1));
        scheduler.available = vec![0; 4];

        // At its deadline the running request leaves, and its owner's `Detach` is handed to
        // group A's workers; nothing is free yet.
        let (reply, replied) = crossbeam_channel::bounded(1);
        scheduler.drain(MigrationCause::Drain, Instant::now(), true, reply);
        assert!(bench.scheduler.requests.is_empty());
        let detach = |after| runtime::Request::Detach {
            owner: OWNER_A,
            after,
        };
        assert_eq!(bench.stamps(), [(A, detach(Vec::new()))]);
        bench.scheduler.stamped(vec![detach(vec![(A, 1)])]);
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        let [(detached, ref request)] = taken[..] else {
            panic!("the `Detach` is written");
        };
        assert_eq!(*request, detach(vec![(A, 1)]));

        // Its answer arrives with words that show nothing free: every device region is asked
        // once more, and the drain waits for the answers.
        bench.complete(detached, &Completion::Done);
        assert!(bench.answers().is_empty(), "its owner is forgotten");
        bench.scheduler.drain_due(Instant::now());
        bench.scheduler.reply_at_boundary();
        assert!(replied.try_recv().is_err());
        bench.scheduler.post().unwrap();
        let resizes = bench.take();
        let regions: Vec<u8> = resizes
            .iter()
            .filter_map(|(_, request)| match request {
                runtime::Request::Resize { region, .. } => Some(*region),
                _ => None,
            })
            .collect();
        assert_eq!(regions, [4, 5, 6, 7]);

        for (id, request) in resizes {
            let runtime::Request::Resize { units, .. } = request else {
                continue;
            };
            let resized = Completion::Resized {
                units,
                evicted: Vec::new(),
            };
            bench.complete(id, &resized);
        }
        while let Some((id, owner, request, completion)) =
            bench.scheduler.next_completion().unwrap()
        {
            bench.scheduler.answered(id, owner, request, completion);
        }
        bench.scheduler.drain_due(Instant::now());
        bench.scheduler.reply_at_boundary();
        assert_eq!(replied.try_recv(), Ok(()));
    }
}
