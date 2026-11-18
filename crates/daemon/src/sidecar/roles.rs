//! The replicas' roles a planner's decision makes: Dynamo's planner states how many prefill and
//! decode workers it wants, and the deployment's replicas, a fixed set its layout states, take
//! those roles in place.
//!
//! A decision of `p` prefill and `d` decode workers that the replicas hold (`p + d` at most
//! their count) gives exactly `p` replicas the prefill role and `d` the decode role, and leaves
//! the others mixed, which Dynamo does not reach. One that asks for more workers than there are
//! replicas divides the replicas in its proportion, by the largest remainder, keeping at least
//! one replica of each role it asks for. A count the decision leaves out (`-1`, never written)
//! keeps the replicas of that role there are.
//!
//! The replicas that change are as few as the counts allow: a role keeps the replicas that have
//! it, then takes mixed replicas, then the other role's, each in the order of their numbers.

use std::collections::BTreeMap;

use scheduling::routing::Role;

/// The roles of the replicas `current` states, after a decision of `prefill` and `decode`
/// workers (`-1` for a count left out).
pub(super) fn assign(
    current: &BTreeMap<u32, Role>,
    prefill: i64,
    decode: i64,
) -> BTreeMap<u32, Role> {
    let count = |role| current.values().filter(|&&have| have == role).count();
    let wanted = |stated: i64, role| usize::try_from(stated).unwrap_or_else(|_| count(role));
    let (prefill, decode) = counts(
        current.len(),
        wanted(prefill, Role::Prefill),
        wanted(decode, Role::Decode),
    );

    let mut assigned: BTreeMap<u32, Role> = current
        .keys()
        .map(|&number| (number, Role::Mixed))
        .collect();
    let take = |role: Role, other: Role, wanted: usize, assigned: &mut BTreeMap<u32, Role>| {
        let mut taken = 0;
        for preferred in [role, Role::Mixed, other] {
            for (&number, &have) in current {
                if taken == wanted {
                    return;
                }
                if have == preferred && assigned[&number] == Role::Mixed {
                    assigned.insert(number, role);
                    taken += 1;
                }
            }
        }
    };
    // Each role first keeps its own replicas, so neither takes one the other keeps.
    let keeps = |role| current.values().filter(|&&have| have == role).count();
    let first = if keeps(Role::Prefill) >= keeps(Role::Decode) {
        [
            (Role::Prefill, Role::Decode, prefill),
            (Role::Decode, Role::Prefill, decode),
        ]
    } else {
        [
            (Role::Decode, Role::Prefill, decode),
            (Role::Prefill, Role::Decode, prefill),
        ]
    };
    for (role, other, wanted) in first {
        take(role, other, wanted, &mut assigned);
    }
    assigned
}

/// The prefill and decode replicas of `replicas` a decision of `prefill` and `decode` workers
/// makes.
fn counts(replicas: usize, prefill: usize, decode: usize) -> (usize, usize) {
    if prefill + decode <= replicas {
        return (prefill, decode);
    }
    // The proportion over every replica, by the largest remainder of the prefill share.
    let share = replicas as f64 * prefill as f64 / (prefill + decode) as f64;
    let mut prefills = share.round() as usize;
    let floor = usize::from(prefill > 0).min(replicas);
    let ceiling = replicas - usize::from(decode > 0).min(replicas - floor);
    prefills = prefills.clamp(floor, ceiling);
    (prefills, replicas - prefills)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles(text: &str) -> BTreeMap<u32, Role> {
        text.chars()
            .zip(0u32..)
            .map(|(role, number)| {
                let role = match role {
                    'p' => Role::Prefill,
                    'd' => Role::Decode,
                    _ => Role::Mixed,
                };
                (number, role)
            })
            .collect()
    }

    /// A decision the replicas hold gives exactly its counts, changing as few replicas as it
    /// can; one past them divides them in its proportion; a count left out keeps its role's.
    #[test]
    fn a_decision_becomes_the_fewest_role_changes() {
        // From mixed replicas, the first ones in order.
        assert_eq!(assign(&roles("mmmm"), 1, 2), roles("pddm"));
        // A decode replica becomes a prefill one, the other replicas keep theirs: the decode role
        // keeps its first replicas in order.
        assert_eq!(assign(&roles("pddd"), 2, 2), roles("pddp"));
        assert_eq!(assign(&roles("ppdd"), 1, 3), roles("pddd"));
        // More workers than replicas: their proportion, a replica of each role at least.
        assert_eq!(assign(&roles("mmmm"), 3, 9), roles("pddd"));
        assert_eq!(assign(&roles("mm"), 10, 1), roles("pd"));
        // A count left out keeps the replicas of its role.
        assert_eq!(assign(&roles("pdmm"), -1, 3), roles("pddd"));
        // None of either: every replica mixed.
        assert_eq!(assign(&roles("pd"), 0, 0), roles("mm"));
    }
}
