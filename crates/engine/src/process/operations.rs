//! The operator's operations inside an epoch: pause and continue, a cache flush, sleep and wake
//! (a cache resize), LoRA load and unload, expert rebalancing and profiling sessions. The weight
//! update in place is `weights`.
//!
//! Each operation is started here and given to whoever does the work: a scheduler, which
//! applies it at its next batch boundary, or a work thread of the data plane. Its completion
//! wakes the control thread, which then acknowledges it ([`MainProcess::continue_operations`]);
//! the control thread waits for none of them.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::MainProcess;
use super::requests::{QUOTAS, fence_switch};

/// The completion of one part of an operation: its results, or why it failed.
pub(super) type Completion = crossbeam_channel::Receiver<std::result::Result<Vec<String>, String>>;

/// An operation still working: what completes it, and the daemon front that is told.
pub(super) struct Operation {
    id: u64,
    front: usize,
    completions: Vec<Completion>,
    results: Vec<String>,
    failure: Option<String>,
    effect: Effect,
}

/// What an operation changes of what this process states about itself, once it succeeded.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Effect {
    /// Nothing.
    None,
    /// What the storage holds: what this process announced of it is read again.
    Storage,
    /// As `Storage`, and the schedulers now sleep or do not.
    Sleep(bool),
}

/// Reads the completions that arrived; `true` once none is left to wait for.
pub(super) fn collect(
    completions: &mut Vec<Completion>,
    results: &mut Vec<String>,
    failure: &mut Option<String>,
) -> bool {
    completions.retain(|completion| match completion.try_recv() {
        Ok(Ok(completed)) => {
            results.extend(completed);
            false
        }
        Ok(Err(error)) => {
            failure.get_or_insert(error);
            false
        }
        Err(crossbeam_channel::TryRecvError::Empty) => true,
        Err(crossbeam_channel::TryRecvError::Disconnected) => {
            failure.get_or_insert_with(|| "the work ended before it completed".to_owned());
            false
        }
    });
    completions.is_empty()
}

impl MainProcess {
    /// Holds or releases admissions (`/admin/pause_generation`, `/admin/continue_generation`):
    /// submissions are refused while the gate is held, and no request in flight is ended.
    pub(super) fn pause(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Revokes every cached prefix page; requests in flight keep theirs. Each scheduler applies
    /// it at its next batch boundary.
    pub(super) fn flush_cache(&mut self) {
        for scheduler in &self.schedulers {
            scheduler.flush_cache();
        }
    }

    /// Installs the expert copies a rebalancing prepared: every scheduler applies the new order
    /// at one common drained step boundary.
    pub(super) fn commit_experts(
        &mut self,
        front: usize,
        commit: proto::request::wire::CommitExperts,
    ) {
        let order: BTreeMap<String, Vec<u32>> = commit
            .order
            .into_iter()
            .map(|(module, order)| (module, order.experts))
            .collect();
        let together = scheduler::HookBarrier::new(self.schedulers.len());
        let completions = self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.commit_experts(order.clone(), Arc::clone(&together)))
            .collect();
        self.operate(front, commit.id, completions, Effect::Storage);
    }

    /// Changes the KV cache's device storage - a sleep, a wake, a resize: every scheduler
    /// withdraws, gives back or restores its units at its next batch boundary.
    pub(super) fn resize_cache(
        &mut self,
        front: usize,
        resize: &proto::request::wire::ResizeCache,
    ) {
        // The storage of a previous epoch is each node's daemon's to keep or release over its
        // claim; this process holds no claim of the deployment's.
        if !resize.keep_weights.is_empty() {
            self.refuse(front, resize.id, QUOTAS.to_owned());
            return;
        }
        let change = match scheduler::CacheChange::from_wire(resize) {
            Ok(change) => change,
            Err(error) => {
                self.refuse(front, resize.id, error);
                return;
            }
        };
        tracing::info!(?change, "changing the KV cache storage");
        let effect = match change {
            scheduler::CacheChange::Sleep => Effect::Sleep(true),
            scheduler::CacheChange::Wake => Effect::Sleep(false),
            _ => Effect::Storage,
        };
        let completions = self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.resize_cache(change.clone()))
            .collect();
        self.operate(front, resize.id, completions, effect);
    }

    /// Serves one named operation.
    ///
    /// The names are the daemon's own: an expert rebalancing's stages, a LoRA adapter's load and
    /// unload, and a profiling session's. They run beside the batches: nothing here touches what
    /// a batch reads, because a rebalancing's copies go to new storage and its one change of the
    /// served weights is the `CommitExperts` that follows, at a boundary.
    pub(super) fn named_operation(&mut self, front: usize, hook: proto::request::wire::Hook) {
        let id = hook.id;
        match hook.name.as_str() {
            "uniserve.prepare_experts" => {
                let orders: BTreeMap<String, Vec<u32>> = match serde_json::from_str(&hook.payload) {
                    Ok(orders) => orders,
                    Err(error) => {
                        self.refuse(front, id, format!("an expert order: {error}"));
                        return;
                    }
                };
                self.data_work(front, id, move |data| data.prepare_experts(orders));
            }
            "uniserve.start_experts" => {
                let data = Arc::clone(&self.data);
                self.data_work(front, id, move |_| data.start_experts());
            }
            "uniserve.experts_ready" => match self.data.experts_ready() {
                Ok(results) => self.respond(front, id, results),
                Err(error) => self.refuse(front, id, error.to_string()),
            },
            "uniserve.abort_experts" => {
                self.data_work(front, id, |data| data.abort_experts());
            }
            "load_lora_adapter" => {
                let payload = hook.payload;
                self.data_work(front, id, move |data| data.load_adapter(&payload));
            }
            "unload_lora_adapter" => {
                let payload = hook.payload;
                self.data_work(front, id, move |data| data.unload_adapter(&payload));
            }
            // A session's collectors begin and end between two batches, which each rank's device
            // thread alone knows: the work thread waits for it, and the control thread does not.
            "uniserve.profile_start" => {
                let payload = hook.payload;
                self.data_work(front, id, move |data| data.start_profile(&payload));
            }
            "uniserve.profile_stop" => {
                let payload = hook.payload;
                self.data_work(front, id, move |data| data.stop_profile(&payload));
            }
            "uniserve.profile_release" => {
                let payload = hook.payload;
                self.data_work(front, id, move |data| data.release_profile(&payload));
            }
            // The range markers read the switch in device storage, and writing it waits for
            // nothing.
            "uniserve.fence_switch" => match fence_switch(&hook.payload) {
                Ok(results) => self.respond(front, id, results),
                Err(error) => self.refuse(front, id, error),
            },
            other => {
                self.refuse(
                    front,
                    id,
                    format!("this process has no operation named {other:?}"),
                );
            }
        }
    }

    /// Runs one operation on a work thread of the data plane, whose end wakes the control
    /// thread: an expert rebalancing walks the model's load, and an adapter reads its file.
    fn data_work(
        &mut self,
        front: usize,
        id: u64,
        work: impl FnOnce(&crate::data_plane::DataPlane) -> worker::Result<Vec<String>> + Send + 'static,
    ) {
        let (complete, completion) = crossbeam_channel::bounded(1);
        let data = Arc::clone(&self.data);
        let wake = self.wake.clone();
        let spawned = std::thread::Builder::new()
            .name("data-plane".into())
            .spawn(move || {
                let _ = complete.send(work(&data).map_err(|error| error.to_string()));
                wake.wake();
            });
        if spawned.is_err() {
            self.refuse(
                front,
                id,
                "the data plane's work thread did not start".to_owned(),
            );
            return;
        }
        self.operate(front, id, vec![completion], Effect::None);
    }

    /// Keeps operation `id` of `front` until `completions` arrive.
    fn operate(&mut self, front: usize, id: u64, completions: Vec<Completion>, effect: Effect) {
        self.operations.push(Operation {
            id,
            front,
            completions,
            results: Vec::new(),
            failure: None,
            effect,
        });
        // A scheduler that completed at once has already woken this thread, or never will.
        self.continue_operations();
    }

    /// Acknowledges every operation whose completions arrived.
    pub(super) fn continue_operations(&mut self) {
        let mut finished = Vec::new();
        for (index, operation) in self.operations.iter_mut().enumerate() {
            if collect(
                &mut operation.completions,
                &mut operation.results,
                &mut operation.failure,
            ) {
                finished.push(index);
            }
        }
        for index in finished.into_iter().rev() {
            let operation = self.operations.remove(index);
            if let Effect::Sleep(asleep) = operation.effect
                && operation.failure.is_none()
            {
                self.asleep = asleep;
            }
            if operation.effect != Effect::None {
                self.refresh_announced();
            }
            self.send_event(
                operation.front,
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: operation.id,
                    results: operation.results,
                    error: operation.failure.unwrap_or_default(),
                }),
            );
        }
    }

    /// Reads again what a successor and an attaching daemon are told of this process's storage. A
    /// change of the cache's segments or of the weights' identity leaves the old description
    /// stale.
    pub(super) fn refresh_announced(&mut self) {
        match self.engine_info() {
            Ok(info) => self.announced = Some(info),
            Err(error) => tracing::warn!(%error, "the engine's description was not read again"),
        }
    }

    /// Acknowledges operation `id` of `front` with `results`.
    fn respond(&mut self, front: usize, id: u64, results: Vec<String>) {
        self.send_event(
            front,
            proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                id,
                results,
                error: String::new(),
            }),
        );
    }

    /// Refuses operation `id` of `front` with `error`.
    pub(super) fn refuse(&mut self, front: usize, id: u64, error: String) {
        self.send_event(
            front,
            proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                id,
                results: Vec::new(),
                error,
            }),
        );
    }
}
