//! Storage commands on the request interface (a cache resize, a sleep or wake, a hook): each is
//! handed to whoever does the work and answered by the serve pass that sees it end, so the control
//! thread waits for none of them.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::MainProcess;
use super::requests::{LEDGERS, fence_switch};

/// A storage command (a cache resize, a sleep or wake, a hook) still working: what answers it, and
/// who waits.
pub(super) struct Owed {
    id: u64,
    front: usize,
    answers: Vec<crossbeam_channel::Receiver<std::result::Result<Vec<String>, String>>>,
    results: Vec<String>,
    failure: Option<String>,
    then: Owing,
}

/// What follows the end of a storage command.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Owing {
    /// The answer alone.
    Answer,
    /// What the storage holds changed: what this process announced of it is read again.
    Storage,
    /// As `Storage`, and the schedulers now sleep or do not.
    Sleep(bool),
}

pub(super) type DataAnswer = crossbeam_channel::Receiver<std::result::Result<Vec<String>, String>>;

/// Reads whatever `answers` holds; `true` once none is left to wait for.
pub(super) fn collect(
    answers: &mut Vec<DataAnswer>,
    results: &mut Vec<String>,
    failure: &mut Option<String>,
) -> bool {
    answers.retain(|answer| match answer.try_recv() {
        Ok(Ok(answered)) => {
            results.extend(answered);
            false
        }
        Ok(Err(error)) => {
            failure.get_or_insert(error);
            false
        }
        Err(crossbeam_channel::TryRecvError::Empty) => true,
        Err(crossbeam_channel::TryRecvError::Disconnected) => {
            failure.get_or_insert_with(|| "the scheduler ended before it answered".to_owned());
            false
        }
    });
    answers.is_empty()
}

impl MainProcess {
    /// Waits for `answers` on later passes, then answers `id` on `front`.
    pub(super) fn owe(&mut self, front: usize, id: u64, answers: Vec<DataAnswer>, then: Owing) {
        self.owed.push(Owed {
            id,
            front,
            answers,
            results: Vec::new(),
            failure: None,
            then,
        });
    }

    /// Answers every storage command that ended.
    pub(super) fn advance_owed(&mut self) {
        let mut finished = Vec::new();
        for (index, owed) in self.owed.iter_mut().enumerate() {
            owed.answers.retain(|answer| match answer.try_recv() {
                Ok(Ok(results)) => {
                    owed.results.extend(results);
                    false
                }
                Ok(Err(error)) => {
                    owed.failure.get_or_insert(error);
                    false
                }
                Err(crossbeam_channel::TryRecvError::Empty) => true,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    owed.failure
                        .get_or_insert_with(|| "the work ended before it answered".to_owned());
                    false
                }
            });
            if owed.answers.is_empty() {
                finished.push(index);
            }
        }
        for index in finished.into_iter().rev() {
            let owed = self.owed.remove(index);
            if let Owing::Sleep(asleep) = owed.then
                && owed.failure.is_none()
            {
                self.asleep = asleep;
            }
            if owed.then != Owing::Answer {
                self.refresh_announced();
            }
            self.send_event(
                owed.front,
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: owed.id,
                    results: owed.results,
                    error: owed.failure.unwrap_or_default(),
                }),
            );
        }
    }

    /// Reads again what a successor and an attaching daemon are told of this process's storage. A
    /// change of the cache's segments or of the weights' identity leaves the old answer stale.
    pub(super) fn refresh_announced(&mut self) {
        match self.engine_info() {
            Ok(info) => self.announced = Some(info),
            Err(error) => tracing::warn!(%error, "the engine's description was not read again"),
        }
    }

    /// Runs one operation the daemon names on a work thread of its own, and answers it from a later
    /// pass: an expert redistribution's plan walks the model's load, and an adapter reads its file.
    fn data_work(
        &mut self,
        front: usize,
        id: u64,
        work: impl FnOnce(&worker::data_plane::DataPlane) -> worker::Result<Vec<String>>
        + Send
        + 'static,
    ) {
        let (answer, answered) = crossbeam_channel::bounded(1);
        let data = Arc::clone(&self.data);
        let spawned = std::thread::Builder::new()
            .name("data-plane".into())
            .spawn(move || {
                let _ = answer.send(work(&data).map_err(|error| error.to_string()));
            });
        if let Err(error) = spawned {
            let _ = error;
            self.send_event(
                front,
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id,
                    results: Vec::new(),
                    error: "the data plane's work thread did not start".to_owned(),
                }),
            );
            return;
        }
        self.owe(front, id, vec![answered], Owing::Answer);
    }

    /// Serves one named operation.
    ///
    /// The names are the daemon's own: an expert redistribution's stages and a LoRA adapter's load
    /// and unload. They run beside the batches: nothing here touches what a batch reads, because a
    /// redistribution's copies go to new storage and its one change of the served weights is the
    /// `CommitExperts` that follows, at a boundary.
    pub(super) fn engine_operation(&mut self, front: usize, hook: proto::request::wire::Hook) {
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
                Ok(results) => self.answer_now(front, id, results),
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
            // The probes read the switch in device storage, and writing it waits for nothing.
            "uniserve.fence_switch" => match fence_switch(&hook.payload) {
                Ok(results) => self.answer_now(front, id, results),
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

    fn answer_now(&mut self, front: usize, id: u64, results: Vec<String>) {
        self.send_event(
            front,
            proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                id,
                results,
                error: String::new(),
            }),
        );
    }

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

    /// Changes the KV cache's device storage: every scheduler withdraws, gives back or
    /// restores its units, and the pass that sees the last one end answers.
    pub(super) fn resize_cache(
        &mut self,
        front: usize,
        resize: &proto::request::wire::ResizeCache,
    ) {
        // The storage of a previous epoch is each node's daemon's to keep or release over its
        // claim; this process holds no claim of the deployment's.
        if !resize.keep_weights.is_empty() {
            self.refuse(front, resize.id, LEDGERS.to_owned());
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
        let then = match change {
            scheduler::CacheChange::Sleep => Owing::Sleep(true),
            scheduler::CacheChange::Wake => Owing::Sleep(false),
            _ => Owing::Storage,
        };
        let answers = self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.resize_cache(change.clone()))
            .collect();
        self.owe(front, resize.id, answers, then);
    }
}
