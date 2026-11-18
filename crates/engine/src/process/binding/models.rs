//! The model a kernel generation of other drafts serves its shapes with (`docs/admin.md`,
//! "Updates"): the serving checkpoint loaded again beside the weights that serve, at the
//! generation's drafts, with its drafter loaded for drafts above zero or released for none.
//!
//! A rank's model loads its drafter only when the deployment speculates
//! (`nn::family::TextPlacement::drafter`), and a model whose layers keep a recurrent state lays
//! its state out for the drafts (`nn::family::TextPlacement::draft_tokens`: a window's draft rows,
//! a stepwise field's slots, a ring's groups). A generation of other drafts therefore loads the
//! model again for them, through the rank's own store: every weight the rank maps already is
//! returned as it is, and only the drafter's weights are published, when it starts speculating.
//! The
//! load runs twice, as an expert redistribution's does ([`crate::memory::reload`]): first over
//! meta weights, which states what the shapes are measured with and the device bytes the
//! drafter's weights take beside what serves ([`WeightBinding::meta_models`]), then on the
//! device once the room is made ([`WeightBinding::load_models`]).
//!
//! The generation's commit makes the loaded model's weights the ones the rank serves and ends
//! the retention of the weights only the replaced model read: the drafter's, when its drafts
//! stop speculating ([`WeightBinding::commit_models`]). A rollback or a discard ends the
//! retention of the weights the load added ([`WeightBinding::release_models`]). Either runs once
//! the workers released the shape whose model read them, so no kernel reads a released weight.

use std::collections::HashSet;
use std::sync::Arc;

use nn::WeightKey;
use nn::family::TextModel;
use service::lock;

use super::{Target, WeightBinding};
use crate::memory::reload::MetaWeights;
use crate::worker::{Error, Result, Worker};

/// The model a kernel generation serves a rank's shapes with, loaded over meta weights, and the
/// device bytes the weights its load adds to the rank's take ([`WeightBinding::meta_models`]).
pub(crate) type MetaModel = (Arc<dyn TextModel>, u64);

/// A model one rank loaded for a kernel generation: its weights in its load's order, and those
/// of them the rank's serving model does not read, which the load published.
pub(super) struct Loaded {
    /// The rank's index among the binding's ranks.
    rank: usize,
    order: Vec<(WeightKey, bool)>,
    added: Vec<WeightKey>,
}

/// The models of one kernel generation on this process's ranks ([`WeightBinding::load_models`]).
pub(super) struct Drafting {
    generation: u64,
    target: Target,
    ranks: Vec<Loaded>,
}

impl WeightBinding {
    /// Whether a kernel generation that drafts `drafts` tokens a step serves its shapes with a
    /// model other than the serving one: its drafts differ from the serving ones, which load or
    /// release the drafter and lay the model's recurrent state out (module documentation).
    pub(crate) fn loads_model(&self, drafts: u32) -> bool {
        lock(&self.target).drafts != drafts
    }

    /// The target the ranks serve, at `drafts`.
    fn drafting(&self, drafts: u32) -> Target {
        Target {
            drafts,
            ..self.target()
        }
    }

    /// The index of the rank whose worker is `worker`, among the binding's ranks.
    fn rank_of(&self, worker: &Arc<Worker>) -> Option<usize> {
        self.ranks.iter().position(|rank| {
            lock(rank)
                .worker
                .upgrade()
                .is_some_and(|held| Arc::ptr_eq(&held, worker))
        })
    }

    /// For each of `workers`, in their order, the model a kernel generation that drafts `drafts`
    /// tokens a step serves its shapes with, loaded over meta weights, and the device bytes the
    /// weights its load adds to the rank's take ([`crate::memory::weights::WeightStore::added_bytes`]);
    /// `None` for a worker of no text rank of this process, and for every worker when the
    /// generation serves the serving model ([`Self::loads_model`]).
    ///
    /// # Errors
    ///
    /// The model's refusal of the drafts: a checkpoint that carries no drafter, or a placement
    /// that loads none.
    pub(crate) fn meta_models(
        &self,
        workers: &[Arc<Worker>],
        drafts: u32,
    ) -> Result<Vec<Option<MetaModel>>> {
        if !self.loads_model(drafts) {
            return Ok(vec![None; workers.len()]);
        }
        let target = self.drafting(drafts);

        workers
            .iter()
            .map(|worker| {
                let Some(index) = self.rank_of(worker) else {
                    return Ok(None);
                };
                let (model, weights, serving) = {
                    let rank = lock(&self.ranks[index]);
                    let serving: HashSet<WeightKey> =
                        rank.serving.iter().map(|(key, _)| key.clone()).collect();
                    (Arc::clone(&rank.model), Arc::clone(&rank.weights), serving)
                };

                let meta = MetaWeights::new();
                let loaded = model.text(&target, &meta).map_err(Error::Invalid)?;
                let added: Vec<(WeightKey, bool, u64)> = meta
                    .into_loads()
                    .into_iter()
                    .filter(|(key, ..)| !serving.contains(key))
                    .collect();
                Ok(Some((loaded, weights.added_bytes(&added))))
            })
            .collect()
    }

    /// Loads, on each rank of this process whose worker is one of `workers`, the model kernel
    /// generation `generation`, which drafts `drafts` tokens a step, serves its shapes with,
    /// through the rank's store beside the weights that serve, and returns each worker's, in
    /// their order (`None` as [`Self::meta_models`] states it). The generation's commit or its
    /// release ends what the loads published.
    ///
    /// # Errors
    ///
    /// Another generation's models held, or a rank's load that failed: what the loads before it
    /// published is released, and the ranks serve as they did.
    pub(crate) fn load_models(
        &self,
        workers: &[Arc<Worker>],
        generation: u64,
        drafts: u32,
    ) -> Result<Vec<Option<Arc<dyn TextModel>>>> {
        if !self.loads_model(drafts) {
            return Ok(vec![None; workers.len()]);
        }
        let mut held = lock(&self.models);
        if let Some(drafting) = held.as_ref() {
            return Err(Error::State(if drafting.generation == generation {
                "the generation's models are loaded already"
            } else {
                "another generation's models are held"
            }));
        }
        let target = self.drafting(drafts);

        let mut drafting = Drafting {
            generation,
            target: target.clone(),
            ranks: Vec::new(),
        };
        let mut models = Vec::with_capacity(workers.len());
        for worker in workers {
            let Some(index) = self.rank_of(worker) else {
                models.push(None);
                continue;
            };
            match self.load_rank(index, &target) {
                Ok((model, loaded)) => {
                    models.push(Some(model));
                    drafting.ranks.push(loaded);
                }
                Err(error) => {
                    if let Err(released) = self.forget_added(&drafting) {
                        tracing::warn!(%released, generation, "a failed load's weights were not released");
                    }
                    return Err(error);
                }
            }
        }

        *held = Some(drafting);
        Ok(models)
    }

    /// One rank's load of `target` through its store ([`Self::load_models`]).
    fn load_rank(&self, index: usize, target: &Target) -> Result<(Arc<dyn TextModel>, Loaded)> {
        let (model, weights, serving) = {
            let rank = lock(&self.ranks[index]);
            let serving: HashSet<WeightKey> =
                rank.serving.iter().map(|(key, _)| key.clone()).collect();
            (Arc::clone(&rank.model), Arc::clone(&rank.weights), serving)
        };

        let loaded = model
            .text(target, weights.as_ref())
            .map_err(Error::Invalid)
            .and_then(|model| Ok((model, weights.finish()?)));
        let (model, order) = match loaded {
            Ok(loaded) => loaded,
            Err(error) => {
                // What the load reached and the rank did not serve is what it published.
                let added: Vec<WeightKey> = weights
                    .abandon()
                    .into_iter()
                    .map(|(key, _)| key)
                    .filter(|key| !serving.contains(key))
                    .collect();
                if let Err(released) = weights.forget(&added) {
                    tracing::warn!(%released, "a failed load's weights were not released");
                }
                return Err(error);
            }
        };

        let added = order
            .iter()
            .map(|(key, _)| key)
            .filter(|key| !serving.contains(*key))
            .cloned()
            .collect();
        Ok((
            model,
            Loaded {
                rank: index,
                order,
                added,
            },
        ))
    }

    /// Kernel generation `generation` committed: each rank serves the weights of the model it
    /// loaded for it from here on, and the retention of the weights only the model it replaced
    /// read ends. Nothing is held for a generation that loaded no model. The workers released
    /// the replaced shapes before this runs.
    ///
    /// # Errors
    ///
    /// The first rank's store that refused to end a retention; every rank serves its new weights
    /// all the same.
    pub(crate) fn commit_models(&self, generation: u64) -> Result<()> {
        let drafting = {
            let mut held = lock(&self.models);
            match held.take() {
                Some(drafting) if drafting.generation == generation => drafting,
                other => {
                    *held = other;
                    return Ok(());
                }
            }
        };

        let mut first = Ok(());
        for loaded in drafting.ranks {
            let (weights, released) = {
                let mut rank = lock(&self.ranks[loaded.rank]);
                let kept: HashSet<&WeightKey> = loaded.order.iter().map(|(key, _)| key).collect();
                let released: Vec<WeightKey> = rank
                    .serving
                    .iter()
                    .map(|(key, _)| key)
                    .filter(|key| !kept.contains(key))
                    .cloned()
                    .collect();
                rank.serving = loaded.order;
                (Arc::clone(&rank.weights), released)
            };
            let result = weights.forget(&released);
            if first.is_ok() {
                first = result;
            }
        }
        *lock(&self.target) = drafting.target;
        first
    }

    /// Gives back the models kernel generation `generation` loaded, which never served or whose
    /// rollback the workers completed: the retention of every weight their loads added ends.
    ///
    /// # Errors
    ///
    /// The first rank's store that refused to end a retention.
    pub(crate) fn release_models(&self, generation: u64) -> Result<()> {
        let drafting = {
            let mut held = lock(&self.models);
            match held.take() {
                Some(drafting) if drafting.generation == generation => drafting,
                other => {
                    *held = other;
                    return Ok(());
                }
            }
        };
        self.forget_added(&drafting)
    }

    /// Gives back the models a kernel generation before `generation` loaded, which a peer's
    /// process could not give back as its rollback was told since its ranks still served them.
    ///
    /// # Errors
    ///
    /// As [`Self::release_models`].
    pub(crate) fn release_models_before(&self, generation: u64) -> Result<()> {
        let earlier = lock(&self.models)
            .as_ref()
            .map(|drafting| drafting.generation)
            .filter(|held| *held < generation);
        match earlier {
            Some(held) => self.release_models(held),
            None => Ok(()),
        }
    }

    /// Ends the retention of every weight `drafting`'s loads added, rank by rank.
    fn forget_added(&self, drafting: &Drafting) -> Result<()> {
        let mut first = Ok(());
        for loaded in &drafting.ranks {
            let weights = Arc::clone(&lock(&self.ranks[loaded.rank]).weights);
            let result = weights.forget(&loaded.added);
            if first.is_ok() {
                first = result;
            }
        }
        first
    }
}
