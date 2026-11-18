//! The `expert-order` control program: each routed layer's experts ordered over their slots so
//! that the layer's expert ranks take near-equal tokens.
//!
//! Expert parallelism divides a layer's physical expert slots among its expert ranks in equal
//! contiguous blocks (`ranks` of them); `moe_config.expert_order` names, for each routed layer by
//! its module path, the logical expert each slot holds, and a layer it does not name holds the
//! checkpoint's order (expert `i` in slot `i`). A rank whose slots hold the experts the router
//! picks most is the layer's bottleneck.
//!
//! Every `period` the program reads the tokens each expert of each layer took over the last
//! `window` (`uniserve_expert_tokens_total` by `layer` and `expert`, the layer label being the
//! layer's key in `expert_order`). A layer whose busiest rank took more than `imbalance` times
//! the ranks' mean is given a balanced order: the experts, heaviest first, each to the rank with
//! the fewest tokens that has a slot left, an expert that stays on its rank keeping its slot so
//! that the change moves only the experts that change rank. The order is written only when it
//! lowers the layer's busiest rank.
//!
//! The order is prepared on the copy stream beside the batches and committed at one computation
//! boundary (`order`); the program outputs a change only when validation accepts it and serving
//! goes on meanwhile, and holds back for `pause` after a change of its own was rolled back.

use std::collections::BTreeMap;

use scheduling::prelude::*;

/// The option the program writes: each layer's logical experts in slot order.
pub const ORDER: &str = "moe_config.expert_order";

/// The series of the tokens each layer's experts took.
pub const TOKENS: &str = "uniserve_expert_tokens_total";

/// The program's arguments, its package entry's `args`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Args {
    /// The expert ranks each layer's slots are divided among, in equal contiguous blocks.
    pub ranks: u32,
    /// The span the tokens are counted over (10 min unless stated).
    #[serde(default = "window")]
    pub window: Duration,
    /// How many times the ranks' mean a layer's busiest rank may take before the layer is
    /// reordered (1.1 unless stated).
    #[serde(default = "imbalance")]
    pub imbalance: f64,
    /// How long the program holds back after a change of its own was rolled back (1 h unless
    /// stated).
    #[serde(default = "pause")]
    pub pause: Duration,
}

fn window() -> Duration {
    mins(10)
}

fn imbalance() -> f64 {
    1.1
}

fn pause() -> Duration {
    hours(1)
}

/// The program; its arguments are all its state.
#[derive(Deserialize, Serialize)]
pub struct ExpertOrder {
    args: Args,
}

/// The tokens each rank's block of `order` takes, `tokens` by logical expert.
fn rank_loads(order: &[u32], tokens: &[f64], ranks: usize) -> Vec<f64> {
    let block = order.len() / ranks;
    let mut loads = vec![0.0; ranks];
    for (slot, &expert) in order.iter().enumerate() {
        loads[slot / block] += tokens.get(expert as usize).copied().unwrap_or(0.0);
    }
    loads
}

fn busiest(loads: &[f64]) -> f64 {
    loads.iter().copied().fold(0.0, f64::max)
}

/// The order of `order`'s experts that balances `tokens` over `ranks` equal blocks of slots.
///
/// The experts go heaviest first (ties by id) to the rank with the fewest tokens so far that has
/// a slot left (ties by rank), the greedy longest-processing-time partition. Within each rank's
/// block an expert the rank already held keeps its slot, and the slots freed are filled with the
/// rank's new experts by id, so only the experts that change rank move.
fn balance(order: &[u32], tokens: &[f64], ranks: usize) -> Vec<u32> {
    let block = order.len() / ranks;
    let load = |expert: u32| tokens.get(expert as usize).copied().unwrap_or(0.0);

    let mut experts = order.to_vec();
    experts.sort_by(|a, b| load(*b).total_cmp(&load(*a)).then(a.cmp(b)));

    let mut held: Vec<Vec<u32>> = vec![Vec::with_capacity(block); ranks];
    let mut loads = vec![0.0_f64; ranks];
    for expert in experts {
        // The blocks' slots are as many as the experts, so one has room.
        let rank = (0..ranks)
            .filter(|&rank| held[rank].len() < block)
            .min_by(|&a, &b| loads[a].total_cmp(&loads[b]).then(a.cmp(&b)));
        if let Some(rank) = rank {
            held[rank].push(expert);
            loads[rank] += load(expert);
        }
    }

    let mut next = vec![u32::MAX; order.len()];
    for (rank, experts) in held.iter_mut().enumerate() {
        let slots = rank * block..(rank + 1) * block;

        // The experts that stay keep their slots.
        experts.retain(|expert| {
            let slot = order[slots.clone()].iter().position(|held| held == expert);
            if let Some(slot) = slot {
                next[rank * block + slot] = *expert;
            }
            slot.is_none()
        });

        // The incoming experts fill the freed slots in order.
        experts.sort_unstable();
        let freed: Vec<usize> = slots.filter(|&slot| next[slot] == u32::MAX).collect();
        for (slot, expert) in freed.into_iter().zip(experts.iter()) {
            next[slot] = *expert;
        }
    }
    next
}

impl ExpertOrder {
    /// The balanced order of `layer`, or `None` when it is balanced enough, cannot be divided
    /// among the ranks, took no tokens, or would not lower its busiest rank.
    fn reorder(&self, state: &State, layer: &str, current: Option<&Vec<u32>>) -> Option<Vec<u32>> {
        let ranks = self.args.ranks as usize;
        let counted = state.stats[TOKENS]
            .window(self.args.window)
            .label("layer", layer)
            .by("expert")
            .sum();

        // The tokens by logical expert id; the experts are the stated order's, or as many as
        // the router names in the checkpoint's order.
        let mut tokens = Vec::new();
        for (expert, count) in counted {
            let Ok(expert) = expert.parse::<usize>() else {
                continue;
            };
            if tokens.len() <= expert {
                tokens.resize(expert + 1, 0.0);
            }
            tokens[expert] = count;
        }
        let order: Vec<u32> = match current {
            Some(order) => order.clone(),
            None => (0..tokens.len() as u32).collect(),
        };
        if ranks == 0 || order.is_empty() || !order.len().is_multiple_of(ranks) {
            warn!(
                "layer {layer}: {} experts do not divide among {ranks} ranks",
                order.len()
            );
            return None;
        }

        let loads = rank_loads(&order, &tokens, ranks);
        let total: f64 = loads.iter().sum();
        if total <= 0.0 || busiest(&loads) <= self.args.imbalance * total / ranks as f64 {
            return None;
        }

        let next = balance(&order, &tokens, ranks);
        let balanced = busiest(&rank_loads(&next, &tokens, ranks));
        if balanced >= busiest(&loads) {
            return None;
        }
        info!(
            "layer {layer}: busiest rank {:.0} -> {balanced:.0} of {total:.0} tokens",
            busiest(&loads)
        );
        Some(next)
    }
}

impl Program for ExpertOrder {
    type Args = Args;
    type Input = State;
    type Output = Option<Config>;

    fn new(args: Args) -> Self {
        ExpertOrder { args }
    }

    fn step(&mut self, state: &State) -> Option<Config> {
        // The whole table: a layer's key is a module path, whose dots are no option path's.
        let mut orders: BTreeMap<String, Vec<u32>> = state.config.get(ORDER).unwrap_or_default();
        let layers = state.stats[TOKENS].window(self.args.window).values("layer");

        let mut changed = false;
        for layer in layers {
            if let Some(next) = self.reorder(state, &layer, orders.get(&layer)) {
                orders.insert(layer, next);
                changed = true;
            }
        }
        if !changed || state.rolled_back(self.args.pause) {
            return None;
        }

        let mut config = state.config.clone();
        let table = orders
            .into_iter()
            .map(|(layer, order)| (layer, Value::from(order)))
            .collect();
        config.set(ORDER, Value::Object(table));
        state.admits(&config).then_some(config)
    }
}

export!(ExpertOrder);

#[cfg(test)]
mod tests {
    use super::*;

    /// The balanced order moves only the experts that change rank, keeps every expert once, and
    /// lowers the busiest rank.
    #[test]
    fn a_balanced_order_moves_only_the_experts_that_change_rank() {
        // Two ranks of four slots; rank 0 holds the four heaviest experts.
        let order = [0, 1, 2, 3, 4, 5, 6, 7];
        let tokens = [40.0, 30.0, 20.0, 10.0, 4.0, 3.0, 2.0, 1.0];
        let next = balance(&order, &tokens, 2);

        let mut sorted = next.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, order);
        assert_eq!(rank_loads(&next, &tokens, 2), vec![55.0, 55.0]);

        // Experts 0 and 3 stay on rank 0 in their slots, 5 and 6 stay on rank 1.
        assert_eq!(next, vec![0, 4, 7, 3, 1, 5, 6, 2]);
    }
}
