//! An in-place weight update: plan the checkpoint's load, quiesce the schedulers, publish the new
//! weights beside the serving ones, apply them at a batch boundary, and resume. Each stage is
//! ended by a serve pass ([`WeightStage`]).

use std::path::PathBuf;
use std::sync::Arc;

use super::MainProcess;
use super::operations::{DataAnswer, collect};

/// A weight update in progress: one stage at a time, each ended by a pass.
pub(super) struct WeightUpdate {
    id: u64,
    front: usize,
    path: String,
    in_flight: scheduler::WeightsInFlight,
    stage: WeightStage,
    /// Whether this update put the schedulers to sleep, so it wakes them.
    slept: bool,
    /// Whether this update holds the schedulers' batches and gives the graph storage to a `keep`
    /// load.
    suspended: bool,
    /// What each rank reported of the new weights.
    results: Vec<String>,
    /// Why the update failed, once it did: the stages that remain only put things back.
    failure: Option<String>,
}

/// Where a weight update stands.
enum WeightStage {
    /// The pass moving it from one stage to the next.
    Between,
    /// The checkpoint's load is walked, moving no byte.
    Planning(crossbeam_channel::Receiver<worker::Result<worker::data_plane::Planned>>),
    /// The schedulers quiesce: for `recompute` and `abort` they sleep, so the cache's storage holds
    /// the new checkpoint, and for a `keep` the free storage does not hold, they stop forming
    /// batches. The load waits.
    Quiescing(Vec<DataAnswer>, Option<worker::data_plane::Planned>),
    /// The new weights are published, beside the weights that serve.
    Loading(crossbeam_channel::Receiver<worker::Result<worker::data_plane::Planned>>),
    /// The schedulers put the new weights in place at a boundary.
    Applying(Vec<DataAnswer>),
    /// The graph storage a `keep` load took comes back.
    Restoring(crossbeam_channel::Receiver<worker::Result<()>>),
    /// The schedulers wake, or form batches again.
    Resuming(Vec<DataAnswer>),
}

impl MainProcess {
    /// Starts an in-place weight update (`docs/admin.md`).
    pub(super) fn start_weight_update(
        &mut self,
        front: usize,
        update: proto::request::wire::UpdateWeights,
    ) {
        let in_flight = match proto::request::wire::WeightsInFlight::try_from(update.in_flight) {
            Ok(proto::request::wire::WeightsInFlight::Keep) => scheduler::WeightsInFlight::Keep,
            Ok(proto::request::wire::WeightsInFlight::Recompute) => {
                scheduler::WeightsInFlight::Recompute
            }
            Ok(proto::request::wire::WeightsInFlight::Abort) => scheduler::WeightsInFlight::Abort,
            Err(_) => {
                self.refuse(
                    front,
                    update.id,
                    format!("unknown in-flight choice {}", update.in_flight),
                );
                return;
            }
        };
        if self.weights.is_some() {
            self.refuse(
                front,
                update.id,
                "another weight update is in progress".to_owned(),
            );
            return;
        }
        tracing::info!(
            model = update.model_path,
            ?in_flight,
            "updating the weights in place"
        );
        let mut target = self.data.target();
        target.checkpoint = PathBuf::from(&update.model_path);
        let (answer, answered) = crossbeam_channel::bounded(1);
        let data = Arc::clone(&self.data);
        let spawned = std::thread::Builder::new()
            .name("weight-plan".into())
            .spawn(move || {
                let _ =
                    answer.send(data.plan(target, worker::memory::reload::Matching::SameTensors));
            });
        if spawned.is_err() {
            self.refuse(
                front,
                update.id,
                "the weight update's work thread did not start".to_owned(),
            );
            return;
        }
        self.weights = Some(WeightUpdate {
            id: update.id,
            front,
            path: update.model_path,
            in_flight,
            stage: WeightStage::Planning(answered),
            slept: false,
            suspended: false,
            results: Vec::new(),
            failure: None,
        });
    }

    /// Moves a weight update to its next stage, when the one it is in ended.
    pub(super) fn advance_weights(&mut self) {
        let Some(mut update) = self.weights.take() else {
            return;
        };
        update.stage = match std::mem::replace(&mut update.stage, WeightStage::Between) {
            WeightStage::Between => WeightStage::Between,
            WeightStage::Planning(answer) => match answer.try_recv() {
                Err(crossbeam_channel::TryRecvError::Empty) => WeightStage::Planning(answer),
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    update.failure = Some("the weight update's plan ended unanswered".to_owned());
                    return self.finish_weights(update);
                }
                Ok(Err(error)) => {
                    update.failure = Some(error.to_string());
                    return self.finish_weights(update);
                }
                Ok(Ok(planned)) => match self.quiesce(&mut update, &planned) {
                    Ok(answers) if answers.is_empty() => self.load_weights(planned, false),
                    Ok(answers) => WeightStage::Quiescing(answers, Some(planned)),
                    Err(refusal) => {
                        update.failure = Some(refusal);
                        self.data.discard(planned);
                        return self.finish_weights(update);
                    }
                },
            },
            WeightStage::Quiescing(mut answers, planned) => {
                if !collect(&mut answers, &mut Vec::new(), &mut update.failure) {
                    WeightStage::Quiescing(answers, planned)
                } else {
                    let Some(planned) = planned else {
                        return self.finish_weights(update);
                    };
                    if update.failure.is_some() {
                        self.data.discard(planned);
                        self.resume_after(update);
                        return;
                    }
                    self.load_weights(planned, update.suspended)
                }
            }
            WeightStage::Loading(answer) => match answer.try_recv() {
                Err(crossbeam_channel::TryRecvError::Empty) => WeightStage::Loading(answer),
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    update.failure = Some("the weight load ended unanswered".to_owned());
                    self.resume_after(update);
                    return;
                }
                Ok(Err(error)) => {
                    update.failure = Some(error.to_string());
                    self.resume_after(update);
                    return;
                }
                Ok(Ok(planned)) => {
                    self.data.stage(planned);
                    WeightStage::Applying(
                        self.schedulers
                            .iter()
                            .map(|scheduler| {
                                scheduler.update_weights(update.path.clone(), update.in_flight)
                            })
                            .collect(),
                    )
                }
            },
            WeightStage::Applying(mut answers) => {
                let mut results = std::mem::take(&mut update.results);
                let finished = collect(&mut answers, &mut results, &mut update.failure);
                update.results = results;
                if finished {
                    self.data.discard_staged();
                    self.resume_after(update);
                    return;
                }
                WeightStage::Applying(answers)
            }
            WeightStage::Restoring(answer) => match answer.try_recv() {
                Err(crossbeam_channel::TryRecvError::Empty) => WeightStage::Restoring(answer),
                result => {
                    if let Ok(Err(error)) = result {
                        update.failure.get_or_insert(error.to_string());
                    }
                    self.release_schedulers(update);
                    return;
                }
            },
            WeightStage::Resuming(mut answers) => {
                if collect(&mut answers, &mut Vec::new(), &mut update.failure) {
                    return self.finish_weights(update);
                }
                WeightStage::Resuming(answers)
            }
        };
        self.weights = Some(update);
    }

    /// Whether the update's load needs the schedulers still first, and what stills them.
    ///
    /// `recompute` and `abort` give the cache's storage to the load, as a sleep does. `keep` holds
    /// the requests' KV cache, so its load goes beside it and the batches go on, unless the free
    /// storage does not hold the new weights: the graph storage then goes too, which stops batches
    /// for as long as the load runs. Storage that holds neither refuses the update, with nothing
    /// changed.
    fn quiesce(
        &mut self,
        update: &mut WeightUpdate,
        planned: &worker::data_plane::Planned,
    ) -> std::result::Result<Vec<DataAnswer>, String> {
        if self.asleep {
            return Ok(Vec::new());
        }
        let change = match update.in_flight {
            scheduler::WeightsInFlight::Recompute => {
                update.slept = true;
                scheduler::CacheChange::Sleep
            }
            scheduler::WeightsInFlight::Abort => {
                update.slept = true;
                scheduler::CacheChange::SleepAborting
            }
            scheduler::WeightsInFlight::Keep => {
                let bytes = planned.bytes();
                let rooms = self.data.room().map_err(|error| error.to_string())?;
                let graphs = self.data.graph_bytes();
                tracing::info!(
                    bytes,
                    graphs,
                    ?rooms,
                    "the room a `keep` load has beside the cache"
                );
                if rooms.iter().all(|room| room.holds(bytes, 0)) {
                    return Ok(Vec::new());
                }
                if !rooms.iter().all(|room| room.holds(bytes, graphs)) {
                    return Err(format!(
                        "the new weights take {bytes} bytes beside the KV cache, and each device \
                         has {rooms:?} of room with {graphs} bytes of graph storage to give back: \
                         `keep` cannot load them, and `recompute` or `abort` gives the cache's \
                         storage to the load"
                    ));
                }
                update.suspended = true;
                scheduler::CacheChange::Hold { held: true }
            }
        };
        Ok(self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.resize_cache(change.clone()))
            .collect())
    }

    /// Starts publishing the new weights beside the ones that serve, after giving the graph storage
    /// up to them when `suspend` says so. A thread of its own: the load reads a checkpoint for
    /// minutes, and neither a scheduler nor this thread may wait for it.
    fn load_weights(&mut self, planned: worker::data_plane::Planned, suspend: bool) -> WeightStage {
        let (answer, answered) = crossbeam_channel::bounded(1);
        let data = Arc::clone(&self.data);
        let spawned = std::thread::Builder::new()
            .name("weight-load".into())
            .spawn(move || {
                let loaded = match suspend.then(|| data.suspend_graphs()) {
                    Some(Err(error)) => {
                        data.discard(planned);
                        Err(error)
                    }
                    _ => data.load(planned),
                };
                let _ = answer.send(loaded);
            });
        if let Err(error) = spawned {
            tracing::error!(%error, "the weight load's thread did not start");
        }
        WeightStage::Loading(answered)
    }

    /// Puts back what the update stilled, and answers once it is.
    fn resume_after(&mut self, mut update: WeightUpdate) {
        if !update.suspended {
            return self.release_schedulers(update);
        }
        let (answer, answered) = crossbeam_channel::bounded(1);
        let data = Arc::clone(&self.data);
        let spawned = std::thread::Builder::new()
            .name("graph-restore".into())
            .spawn(move || {
                let _ = answer.send(data.restore_graphs());
            });
        if let Err(error) = spawned {
            update.failure.get_or_insert(error.to_string());
        }
        update.stage = WeightStage::Restoring(answered);
        self.weights = Some(update);
    }

    /// Lets the schedulers run again, when this update stilled them.
    fn release_schedulers(&mut self, mut update: WeightUpdate) {
        let change = if update.slept {
            scheduler::CacheChange::Wake
        } else if update.suspended {
            scheduler::CacheChange::Hold { held: false }
        } else {
            return self.finish_weights(update);
        };
        update.stage = WeightStage::Resuming(
            self.schedulers
                .iter()
                .map(|scheduler| scheduler.resize_cache(change.clone()))
                .collect(),
        );
        self.weights = Some(update);
    }

    /// Answers the update.
    fn finish_weights(&mut self, update: WeightUpdate) {
        self.refresh_announced();
        let WeightUpdate {
            id,
            front,
            results,
            failure,
            path,
            ..
        } = update;
        match &failure {
            Some(reason) => {
                tracing::warn!(model = path, %reason, "the weights were not updated");
            }
            None => tracing::info!(model = path, "updated the weights in place"),
        }
        self.send_event(
            front,
            proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                id,
                results,
                error: failure.unwrap_or_default(),
            }),
        );
    }
}
