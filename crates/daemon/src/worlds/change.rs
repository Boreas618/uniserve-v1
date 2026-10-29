//! What a world change does, decided once before anything starts: the
//! steps it executes, in order, and what becomes of each derived state of
//! the serving world in its successor (`specs/online-operations.md` §4.5,
//! §7).
//!
//! [`WorldChange`] is the one place these decisions are made. The dry run
//! (`PUT /admin/config` with `dry_run`) reports them, and the change
//! executes them: its preparation, warming, promotion and retention of the
//! previous world each ask it what to do. The one input the decisions lack
//! before the successor's workers start is whether those workers compute
//! other numerics than the serving world's ([`Numerics`]): the root of
//! their pages' prefix hashes and the seed of their stored trajectories,
//! which only the workers derive from the components they load. Every
//! decision takes it as a parameter, so a dry run states the outcome under
//! the same numerics with the outcome under other numerics beside it where
//! they differ, and the change's report states the outcome it observed.

use serde::Serialize;
use storage::protocol::{LinkEnd, gpu_name};

use super::InFlight;
use crate::engine::client::Successor;
use crate::options::PreviousWorld;

/// The decisions of one world change, made from the two worlds' specs, the
/// change's options and the storage plan that admitted it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorldChange {
    /// What the successor adds beside what it shares with the serving world,
    /// as the storage plan settled it: whether it loads its weights, maps
    /// none of the serving world's KV cache, stages its warm copies in host
    /// storage, and whether the serving world runs eagerly meanwhile.
    pub(crate) successor: Successor,
    /// The successor holds other regions of the logical tensors on each GPU
    /// (another parallel configuration): it receives the weights it does
    /// not map through copy manifests.
    pub(crate) redistributes: bool,
    /// The serving world gives back the tail of its KV cache for the
    /// successor: they share GPUs.
    pub(crate) shrinks: bool,
    /// The successor's KV cache is encoded otherwise than the serving
    /// world's, so warming copies it only under an exact conversion.
    pub(crate) reencodes_cache: bool,
    pub(crate) in_flight: InFlight,
    pub(crate) previous_world: PreviousWorld,
    /// Whether the serving world and the successor speculate, and so keep a
    /// draft cache in their KV cache pools.
    pub(crate) serving_drafts: bool,
    pub(crate) successor_drafts: bool,
}

/// Whether the successor's workers compute other numerics than the serving
/// world's, which a change learns once they started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Numerics {
    /// The root of their pages' prefix hashes differs: their KV pages,
    /// recurrent checkpoints and draft cache are keyed otherwise.
    pub(crate) pages: bool,
    /// The seed of their stored trajectories differs: their latents are
    /// keyed otherwise.
    pub(crate) trajectories: bool,
}

impl Numerics {
    /// The successor computes what the serving world computes.
    pub(crate) const SAME: Self = Self {
        pages: false,
        trajectories: false,
    };
    /// The successor computes other numerics for every derived state.
    pub(crate) const OTHER: Self = Self {
        pages: true,
        trajectories: true,
    };
}

/// What a world change does with one derived state of the serving world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Treatment {
    /// The successor reads the serving world's copy read-only: the same
    /// storage on the same GPU, or the shared host tier it is keyed in.
    Mapped,
    /// The Transfer Server copies it into storage of the successor's own,
    /// laid out for the successor.
    Moved,
    /// The successor computes it again: weights from the model's files,
    /// request state from the requests' tokens, optional state in the
    /// background.
    Recomputed,
    /// The requests in flight continue with the state they reached, which
    /// the serving world computed under its numerics.
    Continued,
    /// The successor holds no such state: it does not speculate.
    Dropped,
}

/// One derived state's treatment, and where it is not decided yet, the
/// treatment of each outcome that changes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct Treated {
    /// The treatment under the outcomes the change observed, or, before it
    /// observed them, under the same numerics and an exact conversion.
    pub(crate) treatment: Treatment,
    /// The treatment when the successor computes other numerics, where the
    /// change has not observed them and the treatment differs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) when_numerics_change: Option<Treatment>,
    /// The treatment when the warming finds no exact conversion of the
    /// cache's encoding, where the change re-encodes the cache, has not
    /// warmed yet and the treatment differs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) without_exact_conversion: Option<Treatment>,
}

/// What becomes of each derived state of the serving world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct DerivedState {
    pub(crate) weights: Treated,
    /// The requests' K/V pages (target attention state).
    pub(crate) kv_pages: Treated,
    /// The checkpoints of recurrent layers' state at page boundaries, in
    /// the pools' state group (models with recurrent layers).
    pub(crate) recurrent_checkpoints: Treated,
    /// The drafter's K/V of a deployment that speculates: optional state.
    pub(crate) draft_cache: Treated,
    /// Encoder outputs in the host encoder pools every world maps, keyed by
    /// what computes them and what they read: optional state.
    pub(crate) encoder_outputs: Treated,
    /// Image and video denoising trajectories, stored in the host latent
    /// pools at the park and restored at the step they reached.
    pub(crate) latents: Treated,
}

/// What a change observed so far: `None` for an outcome it has not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Observed {
    pub(crate) numerics: Option<Numerics>,
    /// Whether the warming copied the requests' pages under an exact
    /// conversion of the cache's encoding (always for one encoding).
    pub(crate) converted: Option<bool>,
}

/// One step of a world change, in the order the change executes them; the
/// names are the phases `GET /admin/config` reports the change in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Step {
    /// The serving world switches to eager forwards, releasing its graph
    /// pool to the successor (`update_config.eager_prepare`).
    Eager,
    /// The serving world gives back the tail of its KV cache past the limit
    /// each GPU's ledger leaves it beside the successor.
    Shrink,
    /// The serving world's eager steps make the transfers' baseline.
    Baseline,
    /// The successor starts: it maps, receives or loads its weights, maps
    /// or lays out its KV cache, warms up and captures its graphs.
    Prepare,
    /// The Transfer Server copies the requests' complete pages into the
    /// successor's cache, or into host storage when staged.
    WarmCache,
    /// New requests wait, and the serving world parks at a step boundary.
    Park,
    /// The parked world evicts its cache segments, whose pages host storage
    /// holds, for the staged copies.
    Evict,
    /// The successor loads the copies staged in host storage.
    Materialize,
    /// The successor imports the parked world's index, takes the requests
    /// in flight and admits the waiting ones.
    Promote,
    /// The previous world is kept as `update_config.previous_world` states.
    KeepPrevious,
    /// The successor serves on trial until a commit or a rollback.
    Trial,
}

impl Step {
    /// The step's name, as the plan and the status report it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Step::Eager => "eager",
            Step::Shrink => "shrink",
            Step::Baseline => "baseline",
            Step::Prepare => "prepare",
            Step::WarmCache => "warm_cache",
            Step::Park => "park",
            Step::Evict => "evict",
            Step::Materialize => "materialize",
            Step::Promote => "promote",
            Step::KeepPrevious => "keep_previous",
            Step::Trial => "trial",
        }
    }
}

impl WorldChange {
    /// Whether the requests in flight compute their state again on the
    /// successor: it computes other numerics and the change recomputes.
    pub(crate) fn recomputes(&self, numerics: Numerics) -> bool {
        numerics.pages && self.in_flight == InFlight::Recompute
    }

    /// Whether the successor warms copies of the requests' pages before
    /// its promotion: it maps none of the serving world's cache, and the
    /// requests continue from their pages, or the copies back up the
    /// serving world's pages in host storage for a rollback.
    pub(crate) fn warms(&self, numerics: Numerics) -> bool {
        self.successor.unmapped_cache && (!self.recomputes(numerics) || self.successor.host_cache)
    }

    /// How a warming under `numerics` reuses the serving world's pages: as
    /// they are, as the state the requests continue with, or as rollback
    /// copies alone while the requests compute their state again.
    pub(super) fn reuse(&self, numerics: Numerics) -> super::warming::Reuse {
        if self.recomputes(numerics) {
            super::warming::Reuse::Recompute
        } else if numerics.pages {
            super::warming::Reuse::Keep
        } else {
            super::warming::Reuse::Exact
        }
    }

    /// The steps the change executes, in order, under `numerics`.
    pub(crate) fn steps(&self, numerics: Numerics) -> Vec<Step> {
        let eager = self.successor.eager_source;
        let staged = self.successor.host_cache;
        [
            (Step::Eager, eager),
            (Step::Shrink, self.shrinks),
            (Step::Baseline, eager),
            (Step::Prepare, true),
            (Step::WarmCache, self.warms(numerics)),
            (Step::Park, true),
            (Step::Evict, staged),
            (Step::Materialize, staged),
            (Step::Promote, true),
            (Step::KeepPrevious, true),
            (Step::Trial, true),
        ]
        .into_iter()
        .filter_map(|(step, taken)| taken.then_some(step))
        .collect()
    }

    /// The treatment of the requests' K/V pages under `numerics`, the warm
    /// copies made under an exact conversion or not (`converted`).
    fn kv_pages(&self, numerics: Numerics, converted: bool) -> Treatment {
        if !self.successor.unmapped_cache {
            // The successor maps the serving world's segments: the pages of
            // other numerics are read only by the requests that continue
            // with them, and recomputed when the change recomputes.
            return match (numerics.pages, self.in_flight) {
                (false, _) => Treatment::Mapped,
                (true, InFlight::Keep) => Treatment::Continued,
                (true, InFlight::Recompute) => Treatment::Recomputed,
            };
        }
        if !self.warms(numerics) || self.recomputes(numerics) || !converted {
            return Treatment::Recomputed;
        }
        if numerics.pages {
            Treatment::Continued
        } else {
            Treatment::Moved
        }
    }

    /// The treatment of the draft cache: absent in a successor that does
    /// not speculate; recomputed by refills where the serving world kept
    /// none or the successor maps none of its cache (warming copies the
    /// required groups alone); otherwise as the pages it lies beside.
    fn draft_cache(&self, numerics: Numerics, converted: bool) -> Treatment {
        if !self.successor_drafts {
            Treatment::Dropped
        } else if !self.serving_drafts || self.successor.unmapped_cache {
            Treatment::Recomputed
        } else {
            self.kv_pages(numerics, converted)
        }
    }

    /// The treatment of the stored trajectories: restored into the
    /// successor's banks at the step they reached, laid out for its
    /// distribution, when it computes the same; otherwise continued from
    /// them under its numerics, or generated again from noise.
    fn latents(&self, numerics: Numerics) -> Treatment {
        match (numerics.trajectories, self.in_flight) {
            (false, _) => Treatment::Moved,
            (true, InFlight::Keep) => Treatment::Continued,
            (true, InFlight::Recompute) => Treatment::Recomputed,
        }
    }

    /// What becomes of each derived state under the outcomes `observed`,
    /// with the treatment of each outcome not observed yet beside it where
    /// it differs.
    pub(crate) fn derived_state(&self, observed: Observed) -> DerivedState {
        let numerics = observed.numerics.unwrap_or(Numerics::SAME);
        let converted = observed.converted.unwrap_or(true);
        // Each state's treatment under the observed or assumed outcomes,
        // and under each outcome not observed yet.
        let treated = |treat: &dyn Fn(Numerics, bool) -> Treatment| {
            let treatment = treat(numerics, converted);
            let when_numerics_change = observed
                .numerics
                .is_none()
                .then(|| treat(Numerics::OTHER, converted))
                .filter(|other| *other != treatment);
            let without_exact_conversion = (self.reencodes_cache && observed.converted.is_none())
                .then(|| treat(numerics, false))
                .filter(|other| *other != treatment);
            Treated {
                treatment,
                when_numerics_change,
                without_exact_conversion,
            }
        };
        let weights = if self.successor.loads_weights {
            Treatment::Recomputed
        } else if self.redistributes {
            Treatment::Moved
        } else {
            Treatment::Mapped
        };
        DerivedState {
            weights: treated(&|_, _| weights),
            kv_pages: treated(&|numerics, converted| self.kv_pages(numerics, converted)),
            // Checkpoints lie in the same pool as the pages, are mapped and
            // warmed with them, and are keyed by the same numerics.
            recurrent_checkpoints: treated(&|numerics, converted| {
                self.kv_pages(numerics, converted)
            }),
            draft_cache: treated(&|numerics, converted| self.draft_cache(numerics, converted)),
            // Every world maps the host encoder pools; an output's key holds
            // the identity of the encoder that computes it, which the
            // successor's workers report to their scheduler alone, so the
            // treatment stays per output: read where the key hits.
            encoder_outputs: Treated {
                treatment: Treatment::Mapped,
                when_numerics_change: Some(Treatment::Recomputed),
                without_exact_conversion: None,
            },
            latents: treated(&|numerics, _| self.latents(numerics)),
        }
    }

    /// The steps as a plan reports them before the successor started: the
    /// steps under the same numerics and those under other numerics, in
    /// order, each taken under one outcome alone naming it (`when`), with
    /// the details `detail` adds to a step.
    pub(crate) fn planned_steps(
        &self,
        mut detail: impl FnMut(Step) -> serde_json::Value,
    ) -> Vec<serde_json::Value> {
        let same = self.steps(Numerics::SAME);
        let other = self.steps(Numerics::OTHER);
        let mut steps: Vec<Step> = same.iter().chain(&other).copied().collect();
        steps.sort();
        steps.dedup();
        steps
            .into_iter()
            .map(|step| {
                let mut entry = detail(step);
                if !entry.is_object() {
                    entry = serde_json::json!({});
                }
                entry["step"] = step.name().into();
                match (same.contains(&step), other.contains(&step)) {
                    (true, false) => entry["when"] = "numerics_unchanged".into(),
                    (false, true) => entry["when"] = "numerics_change".into(),
                    _ => {}
                }
                entry
            })
            .collect()
    }
}

/// A link's end as the plan and the report name it: a GPU by its name
/// (`GPU-…`) and host, host storage by its host (`null` where the copy's
/// host is not named), or storage of another host by its Storage Server's
/// endpoint. `hosts` names the hosts of the GPUs it knows.
pub(crate) fn link_end(
    end: &LinkEnd,
    hosts: &std::collections::BTreeMap<[u8; 16], String>,
) -> serde_json::Value {
    match end {
        LinkEnd::Host => serde_json::json!({ "host_storage": null }),
        LinkEnd::Device { gpu } => gpu_end(&gpu_name(gpu), hosts.get(gpu).map(String::as_str)),
        LinkEnd::Remote { endpoint, gpu } => serde_json::json!({
            "remote": endpoint,
            "gpu": gpu.as_ref().map(gpu_name),
        }),
    }
}

/// A GPU as a link's end: its name and its host, when known.
pub(crate) fn gpu_end(gpu: &str, host: Option<&str>) -> serde_json::Value {
    serde_json::json!({ "gpu": gpu, "host": host })
}

/// A host's storage as a link's end.
pub(crate) fn host_end(host: &str) -> serde_json::Value {
    serde_json::json!({ "host_storage": host })
}

/// The bytes one link carries: `{"source", "target", "bytes"}`.
pub(crate) fn link(
    source: serde_json::Value,
    target: serde_json::Value,
    bytes: u64,
) -> serde_json::Value {
    serde_json::json!({ "source": source, "target": target, "bytes": bytes })
}

/// `loads` summed per link, each named by [`link_end`], with the estimated
/// seconds of each where the loads carry them.
pub(crate) fn link_loads<'a>(
    loads: impl IntoIterator<Item = &'a storage::manifest::LinkLoad>,
    hosts: &std::collections::BTreeMap<[u8; 16], String>,
) -> Vec<serde_json::Value> {
    let mut summed: std::collections::BTreeMap<(LinkEnd, LinkEnd), (u64, f64)> =
        std::collections::BTreeMap::new();
    for load in loads {
        let entry = summed
            .entry((load.source.clone(), load.target.clone()))
            .or_default();
        entry.0 += load.bytes;
        entry.1 += load.seconds;
    }
    summed
        .into_iter()
        .map(|((source, target), (bytes, seconds))| {
            let mut entry = link(link_end(&source, hosts), link_end(&target, hosts), bytes);
            entry["seconds"] = seconds.into();
            entry
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A change that keeps every key and layout: an executable or graph
    /// change on the same GPUs.
    fn same_layout() -> WorldChange {
        WorldChange {
            successor: Successor {
                captures_graphs: true,
                ..Successor::default()
            },
            redistributes: false,
            shrinks: true,
            reencodes_cache: false,
            in_flight: InFlight::Keep,
            previous_world: PreviousWorld::Parked,
            serving_drafts: false,
            successor_drafts: false,
        }
    }

    fn names(steps: &[Step]) -> Vec<&'static str> {
        steps.iter().map(|step| step.name()).collect()
    }

    /// A change that keeps the keys maps the weights and the KV cache and
    /// warms nothing; whether the requests then continue with the state of
    /// other numerics is told beside the treatment until the successor
    /// started, and the report states what it observed.
    #[test]
    fn a_change_of_the_same_layout_maps_its_state() {
        let change = same_layout();
        assert_eq!(
            names(&change.steps(Numerics::SAME)),
            [
                "shrink",
                "prepare",
                "park",
                "promote",
                "keep_previous",
                "trial"
            ]
        );
        assert_eq!(change.steps(Numerics::SAME), change.steps(Numerics::OTHER));

        let planned = change.derived_state(Observed::default());
        assert_eq!(planned.weights.treatment, Treatment::Mapped);
        assert_eq!(planned.weights.when_numerics_change, None);
        assert_eq!(planned.kv_pages.treatment, Treatment::Mapped);
        assert_eq!(
            planned.kv_pages.when_numerics_change,
            Some(Treatment::Continued)
        );
        assert_eq!(planned.recurrent_checkpoints, planned.kv_pages);
        assert_eq!(planned.draft_cache.treatment, Treatment::Dropped);
        assert_eq!(planned.latents.treatment, Treatment::Moved);
        assert_eq!(
            planned.latents.when_numerics_change,
            Some(Treatment::Continued)
        );
        assert_eq!(planned.encoder_outputs.treatment, Treatment::Mapped);

        let reported = change.derived_state(Observed {
            numerics: Some(Numerics {
                pages: true,
                trajectories: false,
            }),
            converted: None,
        });
        assert_eq!(reported.kv_pages.treatment, Treatment::Continued);
        assert_eq!(reported.kv_pages.when_numerics_change, None);
        assert_eq!(reported.latents.treatment, Treatment::Moved);
    }

    /// A change of the parallel configuration receives the weights through
    /// manifests and warms the requests' pages into the successor's own
    /// cache: under `recompute` and other numerics it warms nothing, which
    /// the plan says of the warming step.
    #[test]
    fn a_parallel_change_moves_weights_and_pages() {
        let change = WorldChange {
            successor: Successor {
                unmapped_cache: true,
                captures_graphs: true,
                ..Successor::default()
            },
            redistributes: true,
            in_flight: InFlight::Recompute,
            ..same_layout()
        };
        assert_eq!(
            names(&change.steps(Numerics::SAME)),
            [
                "shrink",
                "prepare",
                "warm_cache",
                "park",
                "promote",
                "keep_previous",
                "trial"
            ]
        );
        assert!(!change.steps(Numerics::OTHER).contains(&Step::WarmCache));
        let planned = change.planned_steps(|_| serde_json::Value::Null);
        let warm = planned
            .iter()
            .find(|step| step["step"] == "warm_cache")
            .unwrap();
        assert_eq!(warm["when"], "numerics_unchanged");
        assert!(
            planned
                .iter()
                .filter(|step| step["step"] != "warm_cache")
                .all(|step| step.get("when").is_none())
        );

        let state = change.derived_state(Observed::default());
        assert_eq!(state.weights.treatment, Treatment::Moved);
        assert_eq!(state.kv_pages.treatment, Treatment::Moved);
        assert_eq!(
            state.kv_pages.when_numerics_change,
            Some(Treatment::Recomputed)
        );
        assert_eq!(
            state.latents.when_numerics_change,
            Some(Treatment::Recomputed)
        );
    }

    /// Host staging adds the eviction and the loading of the staged copies
    /// to the promotion, and warms (the rollback copies) even when the
    /// requests compute their state again; an eager preparation switches
    /// the serving world first and waits for its eager baseline.
    #[test]
    fn a_staged_eager_change_evicts_and_materializes() {
        let change = WorldChange {
            successor: Successor {
                unmapped_cache: true,
                host_cache: true,
                eager_source: true,
                captures_graphs: true,
                ..Successor::default()
            },
            in_flight: InFlight::Recompute,
            ..same_layout()
        };
        let steps = names(&change.steps(Numerics::OTHER));
        assert_eq!(
            steps,
            [
                "eager",
                "shrink",
                "baseline",
                "prepare",
                "warm_cache",
                "park",
                "evict",
                "materialize",
                "promote",
                "keep_previous",
                "trial"
            ]
        );
        let state = change.derived_state(Observed {
            numerics: Some(Numerics::OTHER),
            converted: Some(true),
        });
        assert_eq!(state.kv_pages.treatment, Treatment::Recomputed);
    }

    /// A change of the KV cache's encoding warms under an exact conversion
    /// alone, which the plan states beside the moved pages; the report
    /// states what the warming found.
    #[test]
    fn a_change_of_encoding_recomputes_without_a_conversion() {
        let change = WorldChange {
            successor: Successor {
                unmapped_cache: true,
                ..Successor::default()
            },
            reencodes_cache: true,
            ..same_layout()
        };
        let planned = change.derived_state(Observed::default());
        assert_eq!(planned.kv_pages.treatment, Treatment::Moved);
        assert_eq!(
            planned.kv_pages.without_exact_conversion,
            Some(Treatment::Recomputed)
        );
        assert_eq!(planned.weights.without_exact_conversion, None);
        let reported = change.derived_state(Observed {
            numerics: Some(Numerics::SAME),
            converted: Some(false),
        });
        assert_eq!(reported.kv_pages.treatment, Treatment::Recomputed);
        assert_eq!(reported.kv_pages.without_exact_conversion, None);
    }

    /// Speculation turned on refills the draft cache; kept on, its pages
    /// are mapped with the K/V pages; a successor that loads other weights
    /// recomputes them from the model's files.
    #[test]
    fn the_draft_cache_follows_speculation() {
        let on = WorldChange {
            successor_drafts: true,
            ..same_layout()
        };
        let state = on.derived_state(Observed::default());
        assert_eq!(state.draft_cache.treatment, Treatment::Recomputed);
        assert_eq!(state.draft_cache.when_numerics_change, None);

        let kept = WorldChange {
            serving_drafts: true,
            ..on
        };
        let state = kept.derived_state(Observed::default());
        assert_eq!(state.draft_cache.treatment, Treatment::Mapped);
        assert_eq!(
            state.draft_cache.when_numerics_change,
            Some(Treatment::Continued)
        );

        let reloaded = WorldChange {
            successor: Successor {
                loads_weights: true,
                ..Successor::default()
            },
            ..kept
        };
        let state = reloaded.derived_state(Observed::default());
        assert_eq!(state.weights.treatment, Treatment::Recomputed);
    }
}
