//! Which programs a generation installs: the candidates this host's store built for each
//! operator, the deployment's named programs resolved against them into slots, the descriptor keys
//! each package is asked for, and the operators an update changed.

use std::collections::BTreeMap;

use ops::Generation;
use program_server::program::Programs;
use program_server::{
    Descriptors, PhaseKind, Program, ProgramRequest, ProgramSlot, Selector, Target,
};

/// Every program this host may fill an operator with, by operator, in the order the walk tries them:
/// the highest revision first, ties by backend name.
pub(crate) type Candidates = BTreeMap<String, Vec<(String, u32)>>;

/// The programs of an update that differ from what serves.
///
/// The update names every operator the store holds, and an operator this process does not serve is
/// no part of it. A program is compared by its backend and revision: its arguments are the package's
/// defaults, which the two sides state in different words.
pub(crate) fn updated_programs(named: &Programs, serving: &Programs) -> Programs {
    named
        .iter()
        .filter(|(slot, program)| {
            serving
                .iter()
                .filter(|(held, _)| {
                    held.op == slot.op
                        && (held.phase == slot.phase
                            || held.phase.is_none()
                            || slot.phase.is_none())
                })
                .any(|(_, held)| {
                    held.backend != program.backend || held.revision != program.revision
                })
        })
        .map(|(slot, program)| (slot.clone(), program.clone()))
        .collect()
}

/// The operators whose programs differ between what serves and what a precopy prepared.
///
/// `changed` is not `new`. An operator the prepared generation holds and the serving one does not
/// has no old kernel to pair a shadowed step against, so it is not changed; the set is exactly the
/// intersection where the programs differ. It is empty when nothing serves yet - there is no old
/// kernel then either - which is what makes an epoch's first precopy pass without a shadow stage,
/// and why no rule here needs a generation number.
///
/// An operator is one operator however many packages fill it: a store that splits attention into a
/// prefill module and a decode one holds two slots of it, and changing either of them changes what
/// the operator launches. So the comparison is between the two sides' *sets* of programs per
/// operator, not between slots that a split may have renamed.
pub(crate) fn changed_operators(prepared: &Programs, serving: &Programs) -> Vec<String> {
    fn of<'a>(programs: &'a Programs, op: &str) -> Vec<(Option<PhaseKind>, &'a Program)> {
        let mut held: Vec<(Option<PhaseKind>, &Program)> = programs
            .iter()
            .filter(|(slot, _)| slot.op == op)
            .map(|(slot, program)| (slot.phase, program))
            .collect();
        held.sort_by(|left, right| (left.0, &left.1.backend).cmp(&(right.0, &right.1.backend)));
        held
    }
    let mut operators: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for slot in prepared.keys().chain(serving.keys()) {
        operators.insert(slot.op.as_str());
    }
    let mut changed: Vec<String> = Vec::new();
    for op in operators {
        let next = of(prepared, op);
        let old = of(serving, op);
        // An operator either side does not hold is new or retired, not changed: neither has a pair
        // of programs to compare.
        if next.is_empty() || old.is_empty() {
            continue;
        }
        if next != old {
            changed.push(op.to_owned());
        }
    }
    changed
}

/// The package the store holds for `operator` when none of them is built: the highest revision,
/// ties broken by the backend's name so the choice is the same on every run.
///
/// This is the answer for an operator nothing has captured yet - the package that states its keys
/// in its manifest but has no module for this host - which is what `select_programs` cannot see,
/// because its own question is what this host can launch *now*.
pub(crate) fn declared_program(
    packages: &[program_server::protocol::ListedPackage],
    operator: &str,
) -> Option<(String, String, u32)> {
    let mut declared: Vec<&program_server::protocol::ListedPackage> = packages
        .iter()
        .filter(|package| ops::operator_name(&package.op) == Some(operator))
        .collect();
    declared.sort_by(|left, right| {
        right
            .revision
            .cmp(&left.revision)
            .then_with(|| left.backend.cmp(&right.backend))
    });
    let package = declared.first()?;
    Some((
        package.op.clone(),
        package.backend.clone(),
        package.revision,
    ))
}

/// The programs each operator of `packages` may be installed from, in the order the precopy's walk
/// tries them: the highest installed revision first, and among equal revisions the backend whose
/// name sorts first, so the order is the same on every run.
///
/// Nothing here excludes a candidate. An operator built under two backends is one slot with two
/// programs rather than one choice made before the walk, because whether a program can serve a
/// descriptor is only known when the walk names the descriptor (`advance_candidate`).
///
/// Two kinds of package are left out, and both would cost more than they buy:
///
/// - one with no module built for one of `targets`. Installing it asks `prepare` to *build* it for
///   this host, so a store built for other targets would spend a startup's whole precopy clock
///   compiling modules the first token does not need. The operator then stays missing, and the walk
///   names it - the honest answer for a host the store was not built for.
/// - one whose operator this binary does not implement: `admit_installs` refuses a precopy for such
///   an operator, and a program this process could never launch is not worth failing a switchover
///   over.
pub(crate) fn select_programs(
    packages: &[program_server::protocol::ListedPackage],
    targets: &[Target],
) -> BTreeMap<String, Vec<(String, u32)>> {
    let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
    for package in packages {
        if package.modules.is_empty() {
            tracing::debug!(
                op = %package.op,
                backend = %package.backend,
                "a listed package has no built module"
            );
            continue;
        }
        if !package
            .modules
            .iter()
            .any(|module| targets.contains(&module.target))
        {
            // A store built for another device is the one store-side defect a run cannot name from
            // the outside: the package is there, the operator exists, and nothing installs. Both
            // sets are printed, because the difference between them is the fix.
            tracing::info!(
                op = %package.op,
                backend = %package.backend,
                held = ?package
                    .modules
                    .iter()
                    .map(|module| module.target.to_string())
                    .collect::<Vec<_>>(),
                wanted = ?targets.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "the store holds this operator for another target: not installed"
            );
            continue;
        }
        if ops::operator_name(&package.op).is_none() {
            tracing::debug!(
                op = %package.op,
                backend = %package.backend,
                "the store holds an operator this build does not implement"
            );
            continue;
        }
        let candidate = (package.backend.clone(), package.revision);
        let held = chosen.entry(package.op.clone()).or_default();
        if let Some((backend, revision)) = held.last() {
            if *backend == candidate.0 && *revision == candidate.1 {
                continue;
            }
            if *revision != candidate.1 || backend.as_str() <= candidate.0.as_str() {
                tracing::info!(
                    op = %package.op,
                    taken = %candidate.0,
                    revision = candidate.1,
                    held = %backend,
                    held_revision = revision,
                    "an operator is built under more than one backend: both are candidates, the \
                     walk tries the one that can serve its descriptor"
                );
            }
        }
        held.push(candidate);
    }
    for candidates in chosen.values_mut() {
        candidates.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| left.0.as_str().cmp(right.0.as_str()))
        });
    }
    chosen
}

/// The descriptor keys this host has already built for a named package, for this node's targets.
///
/// The epoch's first generation has no serving generation to copy keys from, and without this the
/// walk learns one descriptor per capture. What the Program Server lists is what it holds *built* -
/// not what the package declares - so this accelerates a store that was built ahead of the epoch
/// (every deployment whose programs were built offline) and answers nothing on a host that has built
/// none yet, where the walk remains the source of truth. A module built for another target does not
/// cover this node and is not claimed.
pub(crate) fn built_keys(
    listing: &program_server::Listing,
    op: &str,
    program: &Program,
    targets: &[program_server::Target],
) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for package in &listing.packages {
        if package.op != op
            || package.backend != program.backend
            || package.revision != program.revision
        {
            continue;
        }
        for module in &package.modules {
            if !targets.contains(&module.target) || keys.contains(&module.descriptor) {
                continue;
            }
            keys.push(module.descriptor.clone());
        }
    }
    keys
}

/// The phase a descriptor key states, when its operator's key names one.
///
/// A kernel program may be split by phase, and the phase is the first field of the key's own
/// `Display` form (`phase_decode_dtype_bf16_kv_...`). An operator whose key states no phase - a
/// `gemm`, an `expert_exchange` - answers `None`, and that `None` is what keeps such an operator
/// from ever being split across packages: nothing in its keys says which of them a package covers.
fn key_phase(key: &str) -> Option<PhaseKind> {
    let rest = key.strip_prefix("phase_")?;
    let (name, _) = rest.split_once('_')?;
    PhaseKind::ALL
        .into_iter()
        .find(|phase| phase.as_str() == name)
}

/// Whether a descriptor key belongs to a slot's phase.
///
/// A generation may hold two packages of one operator, one per phase, and the Program Server hands
/// a request's program every key the request names for its operator: a slot that took the other
/// phase's key would have its package asked for a key it cannot build. A key whose text states no
/// phase fits every slot - it is the only kind a phase-less operator has.
pub(crate) fn key_fits_phase(key: &str, phase: Option<PhaseKind>) -> bool {
    match phase {
        None => true,
        Some(phase) => key_phase(key).is_none_or(|held| held == phase),
    }
}

/// The slots one operator's generation holds, from what each candidate package built for this host.
///
/// An operator may be built by more than one package - `paged_attention` is fa4's prefill module and
/// trtllm's decode one - and then its generation must hold *both*: the forward launches a prefill
/// key and a decode key, and a generation that holds one package serves one of them and refuses the
/// other ("no kernel for operator paged_attention descriptor phase_prefill_..."). So the packages
/// are split by the phase their own built keys state, one slot per phase, `holders` being in the
/// deployment's preference order.
///
/// The split is only stated when it is exact: every key of a holder names a phase, and no phase is
/// claimed twice. A package covering everything, an operator whose keys state no phase, or packages
/// overlapping in one phase fall back to the single slot this has always been - the deployment's
/// preference, or the store's order, decides - because a wrong split is worse than no split: two
/// slots of one operator would have each package asked for a key it cannot serve.
fn operator_slots(
    op: &str,
    holders: &[(Program, Vec<String>)],
    fallback: Option<&Program>,
) -> Vec<(ProgramSlot, Program)> {
    let slot = |phase: Option<PhaseKind>| ProgramSlot {
        op: op.to_owned(),
        phase,
        part: None,
    };
    let (claims, refusal) = split_claims(holders);
    if refusal.is_none() && claims.len() > 1 {
        return claims
            .into_iter()
            .map(|(phase, index)| (slot(Some(phase)), holders[index].0.clone()))
            .collect();
    }

    // Packages whose built keys share none divide the operator by descriptor - a bf16 module and an
    // f32 one, whose keys state no phase - so each holds a slot of its own: one program would serve
    // its own keys and refuse the other's at the first capture that launches them.
    if divides_descriptors(holders) {
        return holders
            .iter()
            .map(|(program, _)| {
                (
                    ProgramSlot {
                        part: Some(program.backend.clone()),
                        ..slot(None)
                    },
                    program.clone(),
                )
            })
            .collect();
    }

    // The fallback is right - a wrong split would have a package asked for a key it cannot build -
    // and it used to be silent, which is what made a run's `no kernel for operator X descriptor
    // phase_...` read as a fresh mystery instead of the consequence of a decision. So the decision
    // says itself here, in the words `split_claims` gives it.
    if let Some(reason) = refusal {
        let packages: Vec<&str> = holders
            .iter()
            .map(|(program, _)| program.backend.as_str())
            .collect();
        tracing::warn!(
            op = %op,
            %reason,
            packages = ?packages,
            "this operator's phases are not split across packages, so one program fills it: a key \
             of another phase will fail at its capture unless every package covers it"
        );
    }
    match holders.first().map(|(program, _)| program).or(fallback) {
        Some(program) => vec![(slot(None), program.clone())],
        None => Vec::new(),
    }
}

/// Whether every package built keys of its own, none of them shared: then each one's descriptors are
/// its alone, and the generation needs them all.
fn divides_descriptors(holders: &[(Program, Vec<String>)]) -> bool {
    if holders.len() < 2 || holders.iter().any(|(_, keys)| keys.is_empty()) {
        return false;
    }
    let mut seen = std::collections::BTreeSet::new();
    holders
        .iter()
        .flat_map(|(_, keys)| keys)
        .all(|key| seen.insert(key))
}

/// The phases an operator's packages claim, and - when they cannot be split across them - why.
///
/// `claims` is one entry per (phase, package) a package's built keys name; an operator whose claims
/// are one per phase is split across them, and anything else is filled by one program. `Some(reason)`
/// is the two cases that refuse a split, stated with the packages that brought them about: two
/// packages building one phase, or a package whose key states no phase at all. `None` with fewer than
/// two packages is not a refusal - there is nothing to split.
fn split_claims(holders: &[(Program, Vec<String>)]) -> (Vec<(PhaseKind, usize)>, Option<String>) {
    let phaseful = holders
        .iter()
        .all(|(_, keys)| keys.iter().all(|key| key_phase(key).is_some()));
    let mut claims: Vec<(PhaseKind, usize)> = Vec::new();
    let mut clash: Option<(PhaseKind, usize, usize)> = None;
    if phaseful && holders.len() > 1 {
        for (index, (_, keys)) in holders.iter().enumerate() {
            let mut phases: Vec<PhaseKind> = keys.iter().filter_map(|key| key_phase(key)).collect();
            phases.sort_unstable();
            phases.dedup();
            for phase in phases {
                if let Some((_, held)) = claims.iter().find(|(claimed, _)| *claimed == phase) {
                    clash = clash.or(Some((phase, *held, index)));
                }
                claims.push((phase, index));
            }
        }
    }
    let refusal = match (holders.len(), clash) {
        (0 | 1, _) => None,
        (_, Some((phase, first, second))) => Some(format!(
            "two packages build the {} phase: {} and {}",
            phase, holders[first].0.backend, holders[second].0.backend
        )),
        (_, None) if !phaseful => {
            let unphased = holders
                .iter()
                .find(|(_, keys)| keys.iter().any(|key| key_phase(key).is_none()))
                .map(|(program, _)| program.backend.as_str())
                .unwrap_or("one of them");
            Some(format!("a key {unphased} built states no phase"))
        }
        _ => None,
    };
    (claims, refusal)
}

/// The generation's programs, from the deployment's named programs and this host's store.
///
/// `chosen` is the store's candidates per operator, in its own preference order; `built` is what
/// each package built for this host's targets, keyed by the package. Pure, so the resolution a run
/// depends on can be tested with the shape a `Precopy` actually hands it.
pub(crate) fn resolve_slots(
    named: &Programs,
    chosen: &BTreeMap<String, Vec<(String, u32)>>,
    built: &BTreeMap<(String, String, u32), Vec<String>>,
    packages: &[program_server::protocol::ListedPackage],
) -> std::result::Result<(Programs, Candidates), String> {
    let keys_of = |op: &str, program: &Program| -> Vec<String> {
        built
            .get(&(op.to_owned(), program.backend.clone(), program.revision))
            .cloned()
            .unwrap_or_default()
    };
    let mut programs = Programs::new();
    let mut candidates = Candidates::new();

    // Every operator the store supplies, and every one the deployment names - a named program whose
    // operator the store does not supply is still installed, and the walk reports what it cannot
    // bind.
    let mut operators: Vec<String> = chosen.keys().cloned().collect();
    for slot in named.keys() {
        if !operators.contains(&slot.op) {
            operators.push(slot.op.clone());
        }
    }
    let named_here = |op: &str| -> Vec<(Option<PhaseKind>, Program)> {
        named
            .iter()
            .filter(|(slot, _)| slot.op == op)
            .map(|(slot, program)| (slot.phase, program.clone()))
            .collect()
    };
    for op in operators {
        let stated = named_here(&op);
        // A deployment that names a program has chosen an operator's implementation, and the name
        // is a fact rather than a preference: this host either built that program or it did not.
        // When it did not, substituting another program is not the smaller failure - the named
        // program's keys are unknown here, so a substitute may cover no more than the phase it was
        // not named for, and the walk finds the rest one capture at a time (a generation that holds
        // one program of a two-phase operator stands at its first warmup). The refusal names what
        // the store does hold, which is the fact the operator needs to build or to name.
        if let Some((_, program)) = stated
            .iter()
            .find(|(_, program)| keys_of(&op, program).is_empty())
        {
            // What the store has built for the operator, and what it holds without a module: the
            // two are different facts and the message states both, because a package that is
            // declared and unbuilt is a store build away from serving, while a package that is not
            // there at all is a deployment question.
            let held: Vec<String> = chosen
                .get(&op)
                .into_iter()
                .flatten()
                .map(|(backend, revision)| format!("{backend} revision {revision}"))
                .collect();
            let built_here = |backend: &str, revision: u32| {
                chosen
                    .get(&op)
                    .is_some_and(|held| held.iter().any(|(b, r)| b == backend && *r == revision))
            };
            let declared: Vec<String> = packages
                .iter()
                .filter(|package| {
                    package.op == op && !built_here(&package.backend, package.revision)
                })
                .map(|package| format!("{} revision {}", package.backend, package.revision))
                .collect();
            let mut answer = format!(
                "the deployment names {op}/{} revision {} and this host has built no module of it \
                 for its targets: the store holds {}",
                program.backend,
                program.revision,
                if held.is_empty() {
                    format!("no built package of {op}")
                } else {
                    held.join(", ")
                }
            );
            if !declared.is_empty() {
                answer.push_str(&format!(
                    " and declares {} without a module for this host's targets",
                    declared.join(", ")
                ));
            }
            answer.push_str(
                " - build the named package for this host's targets, or name a backend the store \
                 holds",
            );
            return Err(answer);
        }
        // A deployment that states a backend per phase - a prefill attention backend and a decode
        // one - has already said which keys each of them covers, so nothing here is decided from
        // the store: both programs are the operator's, each in its own phase.
        if stated.iter().any(|(phase, _)| phase.is_some()) {
            for (phase, program) in &stated {
                programs.insert(
                    ProgramSlot {
                        op: op.clone(),
                        phase: *phase,
                        part: None,
                    },
                    program.clone(),
                );
            }
            if let Some(ordered) = chosen.get(&op) {
                candidates.insert(op, ordered.clone());
            }
            continue;
        }
        // The deployment's program comes first: it decides the phases its own package covers, and
        // the store's order decides the rest. A program is one package however it was written down,
        // so the named form and the store's form of the same package are one entry - with the
        // deployment's arguments, which are the ones the epoch pinned.
        let mut ordered: Vec<Program> = stated.into_iter().map(|(_, program)| program).collect();
        for (backend, revision) in chosen.get(&op).cloned().unwrap_or_default() {
            ordered.push(Program {
                backend,
                revision,
                arguments: Default::default(),
            });
        }
        let mut seen: Vec<(String, u32)> = Vec::with_capacity(ordered.len());
        ordered.retain(|program| {
            let package = (program.backend.clone(), program.revision);
            if seen.contains(&package) {
                return false;
            }
            seen.push(package);
            true
        });
        // What each program built for this host's targets: the keys a phase split is read from.
        let holders: Vec<(Program, Vec<String>)> = ordered
            .iter()
            .map(|program| (program.clone(), keys_of(&op, program)))
            .filter(|(_, keys)| !keys.is_empty())
            .collect();
        let was_named = named.keys().any(|slot| slot.op == op);
        if was_named && holders.first().map(|(program, _)| program) != ordered.first() {
            // The deployment named a program this host has built nothing of, and another package
            // has: the store decides rather than the name, so say so - silently serving another
            // backend than the one the deployment named would be the harder fact to find.
            tracing::warn!(
                op = %op,
                named = %ordered.first().map(|program| program.backend.as_str()).unwrap_or(""),
                built = %holders.first().map(|(program, _)| program.backend.as_str()).unwrap_or(""),
                "the named program is not built for this host's targets: the store's own package \
                 fills its operator"
            );
        }
        for (slot, program) in operator_slots(&op, &holders, ordered.first()) {
            programs.insert(slot, program);
        }
        if let Some(ordered) = chosen.get(&op) {
            candidates.insert(op, ordered.clone());
        }
    }
    Ok((programs, candidates))
}

/// Adds one slot's keys to a package's request, merging with a slot of an operator the package
/// already fills.
///
/// A request names one program per operator, and the Program Server hands that program every key
/// the request names for the operator, so two slots of one operator filled by the same package
/// belong in one entry rather than two that would each be asked for both key sets.
pub(crate) fn merge_request(
    group: &mut (Vec<ProgramRequest>, Vec<Descriptors>),
    slot: &ProgramSlot,
    program: &Program,
    keys: Vec<String>,
) {
    if let Some(index) = group
        .1
        .iter()
        .position(|descriptor| descriptor.op == slot.op)
    {
        let descriptor = &mut group.1[index];
        descriptor.keys.extend(keys);
        descriptor.keys.sort_unstable();
        descriptor.keys.dedup();
        return;
    }
    group.0.push(ProgramRequest {
        slot: slot.clone(),
        selector: Selector::Fixed(program.clone()),
    });
    group.1.push(Descriptors {
        op: slot.op.clone(),
        keys,
    });
}

pub(crate) fn descriptor_keys(generation: &Generation, operator: &'static str) -> Vec<String> {
    generation
        .iter()
        .filter(|(name, _, _)| *name == operator)
        .map(|(_, key, _)| key.to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use program_server::ModuleForm;
    use program_server::protocol::{ListedModule, ListedPackage};

    /// An update names the whole set, and what it installs is what differs from what serves.
    ///
    /// The operator that stayed keeps its kernel, and with it every bucket it launches in: an update
    /// of `rms_norm` installs and compares `rms_norm` alone.
    #[test]
    fn an_update_installs_only_the_programs_that_changed() {
        let program = |backend: &str, revision| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let slot = |op: &str| program_server::program::ProgramSlot::all(op);
        let serving: Programs = [
            (slot("rms_norm"), program("triton", 1)),
            (slot("paged_attention"), program("trtllm", 1)),
        ]
        .into_iter()
        .collect();
        let named: Programs = [
            (slot("rms_norm"), program("triton", 2)),
            (slot("paged_attention"), program("trtllm", 1)),
            (slot("gelu"), program("triton", 1)),
        ]
        .into_iter()
        .collect();
        let updated = updated_programs(&named, &serving);
        assert_eq!(updated.len(), 1, "{updated:?}");
        assert_eq!(updated[&slot("rms_norm")].revision, 2);
    }

    /// An epoch's first precopy has no serving generation, so no operator has an old kernel to pair
    /// a shadowed step against: the changed set is empty and the precopy finishes without asking
    /// for a step an idle epoch never runs.
    #[test]
    fn a_first_generation_changes_no_operator_and_asks_for_no_shadow() {
        let mut prepared = Programs::new();
        prepared.insert(
            program_server::program::ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "fa4".to_owned(),
                revision: 3,
                arguments: Default::default(),
            },
        );

        assert!(
            changed_operators(&prepared, &Programs::new()).is_empty(),
            "a first generation reported a changed operator, which would wait for a shadow step \
             that cannot run"
        );
    }

    /// `changed` is the intersection where the two programs differ, and not `new`: an operator the
    /// prepared generation adds has no old kernel either, so it is not changed.
    #[test]
    fn only_an_operator_whose_program_differs_is_changed() {
        let slot = |op: &str| program_server::program::ProgramSlot {
            op: op.to_owned(),
            phase: None,
            part: None,
        };
        let program = |backend: &str, revision: u32| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let mut serving = Programs::new();
        serving.insert(slot("gemm"), program("cublas", 1));
        serving.insert(slot("softmax"), program("reference", 2));

        let mut prepared = serving.clone();
        prepared.insert(slot("gemm"), program("cublas", 2));
        prepared.insert(slot("rms_norm"), program("reference", 1));

        assert_eq!(changed_operators(&prepared, &serving), vec!["gemm"]);
    }

    /// A deployment names **one** backend for attention (`attention_backend=trtllm`, a phase-less
    /// slot), and this host's store holds that package's decode module *and* fa4's prefill one. The
    /// generation must hold both - the forward launches a prefill key and a decode key - and the
    /// deployment's arguments stay on the package it named.
    ///
    /// The named package and the store's listing of it differ only in their arguments, and they are
    /// one candidate: counted twice, the second copy's decode key would collide with the first's and
    /// the operator would be filled by one program, which serves no prefill.
    #[test]
    fn a_deployment_that_names_one_backend_still_holds_the_phases_the_store_builds() {
        let prefill = "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let decode = "phase_decode_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let mut arguments = toml::Table::new();
        arguments.insert("window".to_owned(), toml::Value::Boolean(false));

        // What the `Precopy` hands the install: one phase-less slot for attention, naming trtllm.
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "trtllm".to_owned(),
                revision: 1,
                arguments: arguments.clone(),
            },
        );

        // What this host's store holds for the operator: both packages, in its own order.
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert(
            "paged_attention".to_owned(),
            vec![("fa4".to_owned(), 1), ("trtllm".to_owned(), 1)],
        );
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned()],
        );
        built.insert(
            ("paged_attention".to_owned(), "trtllm".to_owned(), 1),
            vec![decode.to_owned()],
        );

        let (programs, candidates) =
            resolve_slots(&named, &chosen, &built, &[]).expect("the store built what was named");

        assert_eq!(
            programs.len(),
            2,
            "the generation held one program of an operator whose keys span two phases, which is \
             the stop this test is written from"
        );
        assert_eq!(
            programs
                .get(&ProgramSlot {
                    op: "paged_attention".to_owned(),
                    phase: Some(PhaseKind::Prefill),
                    part: None,
                })
                .map(|program| program.backend.as_str()),
            Some("fa4"),
            "the prefill key's package was not bound for prefill"
        );
        let decode_slot = ProgramSlot {
            op: "paged_attention".to_owned(),
            phase: Some(PhaseKind::Decode),
            part: None,
        };
        let Some(decode_program) = programs.get(&decode_slot).cloned() else {
            panic!("the decode phase bound no program");
        };
        assert_eq!(decode_program.backend, "trtllm");
        assert_eq!(
            decode_program.arguments, arguments,
            "the deployment's arguments were replaced by the store's defaults"
        );
        assert_eq!(
            candidates.get("paged_attention").map(Vec::len),
            Some(2),
            "the walk lost the operator's alternatives"
        );
    }

    /// The other half of the same rule: when the named package covers every phase its operator's
    /// keys span, one slot is right and a split would be two packages asked for the same keys.
    #[test]
    fn a_named_package_that_covers_every_phase_stays_one_slot() {
        let prefill = "phase_prefill_dtype_bf16";
        let decode = "phase_decode_dtype_bf16";
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "fa4".to_owned(),
                revision: 1,
                arguments: Default::default(),
            },
        );
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert(
            "paged_attention".to_owned(),
            vec![("fa4".to_owned(), 1), ("trtllm".to_owned(), 1)],
        );
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned(), decode.to_owned()],
        );
        built.insert(
            ("paged_attention".to_owned(), "trtllm".to_owned(), 1),
            vec![decode.to_owned()],
        );

        let (programs, _) =
            resolve_slots(&named, &chosen, &built, &[]).expect("the store built what was named");

        assert_eq!(
            programs.len(),
            1,
            "one package covering both phases was split"
        );
        let slot = ProgramSlot {
            op: "paged_attention".to_owned(),
            phase: None,
            part: None,
        };
        assert_eq!(
            programs.get(&slot).map(|program| program.backend.as_str()),
            Some("fa4")
        );
    }

    /// An operator is one operator however many packages fill it: a store that splits attention
    /// into a prefill module and a decode one holds two slots of it, and changing either changes
    /// what the operator launches, so the shadow stage is owed for it.
    #[test]
    fn an_operator_filled_by_two_packages_is_one_changed_operator() {
        let program = |backend: &str, revision: u32| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let slot = |phase: Option<PhaseKind>| ProgramSlot {
            op: "paged_attention".to_owned(),
            phase,
            part: None,
        };
        // What serves: one package filling both phases.
        let mut serving = Programs::new();
        serving.insert(slot(None), program("fa4", 3));
        // What was prepared: the store's split - the prefill cell unchanged, the decode cell taken
        // over by another package.
        let mut prepared = Programs::new();
        prepared.insert(slot(Some(PhaseKind::Prefill)), program("fa4", 3));
        prepared.insert(slot(Some(PhaseKind::Decode)), program("trtllm", 7));

        assert_eq!(
            changed_operators(&prepared, &serving),
            vec!["paged_attention"],
            "a split of one operator was not seen as a change of that operator"
        );
        // The same two slots on both sides is no change: a split that serves on both sides is the
        // same operator running the same programs.
        assert!(changed_operators(&prepared, &prepared).is_empty());
    }

    /// The Program Server hands a request's program every key the request names for its operator,
    /// so a generation's slots are grouped by the package that fills them: two slots of one
    /// operator from one package are one request entry with both key sets, and two packages are
    /// two requests that never name each other's keys.
    #[test]
    fn a_package_asks_once_per_operator_for_the_keys_it_built() {
        let program = |backend: &str, revision: u32| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let slot = |op: &str, phase: Option<PhaseKind>| ProgramSlot {
            op: op.to_owned(),
            phase,
            part: None,
        };

        let mut group: (Vec<ProgramRequest>, Vec<Descriptors>) = (Vec::new(), Vec::new());
        let prefill = program("fa4", 3);
        merge_request(
            &mut group,
            &slot("paged_attention", Some(PhaseKind::Prefill)),
            &prefill,
            vec!["phase_prefill_dtype_bf16".to_owned()],
        );
        merge_request(
            &mut group,
            &slot("paged_attention", Some(PhaseKind::Decode)),
            &prefill,
            vec!["phase_decode_dtype_bf16".to_owned()],
        );
        merge_request(
            &mut group,
            &slot("gemm", None),
            &prefill,
            vec!["dtype_f16_m_128_n_128_k_7168".to_owned()],
        );

        assert_eq!(group.0.len(), 2, "one package asked twice for one operator");
        assert!(
            group
                .0
                .iter()
                .all(|request| request.selector == Selector::Fixed(prefill.clone()))
        );
        let attention = group
            .1
            .iter()
            .find(|descriptor| descriptor.op == "paged_attention")
            .map(|descriptor| descriptor.keys.clone())
            .unwrap_or_default();
        assert_eq!(
            attention,
            vec![
                "phase_decode_dtype_bf16".to_owned(),
                "phase_prefill_dtype_bf16".to_owned()
            ],
            "a phase's key was dropped from the package's one request"
        );
    }

    /// A listed package with one built module for `target`.
    fn package(op: &str, backend: &str, revision: u32, target: &str) -> ListedPackage {
        ListedPackage {
            op: op.to_owned(),
            backend: backend.to_owned(),
            revision,
            modules: vec![ListedModule {
                target: Target::new(target),
                descriptor: "a-descriptor".to_owned(),
                form: ModuleForm::Cubin,
            }],
        }
    }

    fn targets() -> Vec<Target> {
        vec![Target::new("sm_90a")]
    }

    /// An operator name this build implements, taken from the build rather than written down, so a
    /// renamed operator does not fail a test about selection.
    fn implemented() -> &'static str {
        ops::operators()
            .first()
            .copied()
            .expect("the build implements operators")
    }

    /// The later installation of an operator is the one the walk tries first, whatever order the
    /// store lists them in.
    #[test]
    fn the_highest_revision_of_an_operator_is_chosen() {
        let op = implemented();
        for packages in [
            vec![
                package(op, "triton", 1, "sm_90a"),
                package(op, "triton", 3, "sm_90a"),
            ],
            vec![
                package(op, "triton", 3, "sm_90a"),
                package(op, "triton", 1, "sm_90a"),
            ],
        ] {
            let chosen = select_programs(&packages, &targets());
            assert_eq!(
                chosen.get(op).and_then(|ordered| ordered.first()),
                Some(&("triton".to_owned(), 3)),
                "the highest revision is tried first"
            );
        }
    }

    /// When two backends hold the same revision, the one that sorts first is tried first - and the
    /// other stays a candidate, so a program that cannot serve the walk's descriptor is answered
    /// rather than fatal.
    #[test]
    fn a_tie_on_revision_is_broken_by_the_backend_name() {
        let op = implemented();
        for order in [["triton", "flashinfer"], ["flashinfer", "triton"]] {
            let packages = order
                .into_iter()
                .map(|backend| package(op, backend, 2, "sm_90a"))
                .collect::<Vec<_>>();
            let chosen = select_programs(&packages, &targets());
            assert_eq!(
                chosen.get(op).and_then(|ordered| ordered.first()),
                Some(&("flashinfer".to_owned(), 2)),
                "the same backend is tried first whatever order the store lists them in"
            );
            assert_eq!(
                chosen[op].len(),
                2,
                "and the other backend is still a candidate"
            );
        }
    }

    /// Three packages that cannot launch here, for three different reasons, and one that can: only
    /// the last is installed.
    #[test]
    fn only_packages_this_host_could_launch_are_installed() {
        let op = implemented();
        let mut unbuilt = package(op, "triton", 9, "sm_90a");
        unbuilt.modules.clear();
        let packages = vec![
            unbuilt,
            package(op, "triton", 8, "sm_70"),
            package("an-operator-this-build-lacks", "triton", 7, "sm_90a"),
            package(op, "triton", 1, "sm_90a"),
        ];

        let chosen = select_programs(&packages, &targets());
        assert_eq!(chosen.len(), 1, "one operator can launch here: {chosen:?}");
        assert_eq!(
            chosen[op],
            vec![("triton".to_owned(), 1)],
            "the package with a module for this target, whatever the others claim"
        );
    }

    /// One store, read twice: with the deployment naming one backend and with it naming nothing.
    /// The two must produce the same slot set, because the name chooses the implementation of the
    /// keys its own package covers and the store keeps the other phases. A relaunch hands the engine
    /// the epoch's kernels - the named form of a start that named nothing - so the two forms meeting
    /// one store is the equality a relaunch depends on.
    #[test]
    fn a_named_start_and_an_unnamed_one_agree_on_the_slots_of_one_store() {
        let prefill = "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let decode = "phase_decode_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert(
            "paged_attention".to_owned(),
            vec![("fa4".to_owned(), 1), ("trtllm".to_owned(), 1)],
        );
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned()],
        );
        built.insert(
            ("paged_attention".to_owned(), "trtllm".to_owned(), 1),
            vec![decode.to_owned()],
        );
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "trtllm".to_owned(),
                revision: 1,
                arguments: Default::default(),
            },
        );

        let (unnamed, _) = resolve_slots(&Programs::new(), &chosen, &built, &[])
            .expect("the store decides when nothing is named");
        let (with_name, _) =
            resolve_slots(&named, &chosen, &built, &[]).expect("the named package is built here");

        let slots = |programs: &Programs| {
            let mut held: Vec<(String, Option<PhaseKind>, String)> = programs
                .iter()
                .map(|(slot, program)| (slot.op.clone(), slot.phase, program.backend.clone()))
                .collect();
            held.sort();
            held
        };
        assert_eq!(
            slots(&with_name),
            slots(&unnamed),
            "naming one backend changed the slot set of one store"
        );
        assert_eq!(
            slots(&with_name),
            vec![
                (
                    "paged_attention".to_owned(),
                    Some(PhaseKind::Prefill),
                    "fa4".to_owned()
                ),
                (
                    "paged_attention".to_owned(),
                    Some(PhaseKind::Decode),
                    "trtllm".to_owned()
                ),
            ],
            "the deployment's own backend did not take the phase its keys cover"
        );
    }

    /// A program this host has built nothing of is refused by name rather than substituted. The
    /// keys the name owned are not known here, so a substitute may cover a different phase and the
    /// walk would find out one capture at a time, each a device preparation that stands until the
    /// watchdog's bound.
    #[test]
    fn a_named_program_this_host_did_not_build_is_refused_by_name() {
        let prefill = "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert("paged_attention".to_owned(), vec![("fa4".to_owned(), 1)]);
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned()],
        );
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "trtllm".to_owned(),
                revision: 1,
                arguments: Default::default(),
            },
        );

        // The store holds the named package and has never built it: that is a store build away
        // from serving, and the refusal says so rather than naming a substitution.
        let packages = vec![program_server::protocol::ListedPackage {
            op: "paged_attention".to_owned(),
            backend: "trtllm".to_owned(),
            revision: 1,
            modules: Vec::new(),
        }];
        let error = resolve_slots(&named, &chosen, &built, &packages)
            .expect_err("nothing of the named package was built here");
        assert!(
            error.contains("paged_attention/trtllm revision 1"),
            "the refusal does not name the program the deployment named: {error}"
        );
        assert!(
            error.contains("fa4 revision 1"),
            "the refusal does not name what the store holds instead: {error}"
        );
        assert!(
            error.contains("declares trtllm revision 1 without a module"),
            "the refusal does not tell a declared-but-unbuilt package from a missing one: {error}"
        );
    }

    /// An operator no model forward reaches - the runner launches `slot_commit` itself - is a
    /// package the store holds and has never built, so `select_programs` cannot see it. The walk
    /// names the operator, and the declaration is what lets this host build the key it needs: the
    /// highest revision of the operator's packages, ties broken by the backend so two runs choose
    /// alike.
    #[test]
    fn a_declared_package_is_chosen_for_an_operator_nothing_built() {
        let package = |backend: &str, revision: u32| program_server::protocol::ListedPackage {
            op: "slot_commit".to_owned(),
            backend: backend.to_owned(),
            revision,
            modules: Vec::new(),
        };
        let packages = vec![
            package("triton", 1),
            package("triton", 3),
            package("cuda", 3),
        ];
        assert_eq!(
            declared_program(&packages, "slot_commit"),
            Some(("slot_commit".to_owned(), "cuda".to_owned(), 3)),
            "the highest revision did not win, or the tie was not broken by the backend's name"
        );
        assert_eq!(
            declared_program(&packages, "paged_attention"),
            None,
            "an operator the store holds no package of was answered"
        );
    }
}
