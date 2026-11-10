//! Which programs a generation installs: the slots the Program Server's selection fills for each
//! operator's descriptor keys, the keys each slot's package is asked for, and the operators a
//! selection replaces or changes against what serves.
//!
//! The engine chooses nothing here. The Program Server owns coverage and selection
//! (`ProgramServer::cover`): for every key it answers the one package that builds it, the backend
//! the deployment names first and the manifests' priorities after. This module only turns that
//! answer into the slots a generation records.

use std::collections::{BTreeMap, BTreeSet};

use program_server::Named;
use proto::program::{PhaseKind, Program, ProgramSlot, Programs};

/// The descriptor keys of each slot a selection filled, by slot: the keys its package was selected
/// for, and the only keys that package is asked to build.
pub(crate) type Routes = BTreeMap<ProgramSlot, Vec<String>>;

/// The backends a deployment names for `op`, as the Program Server's selection takes them: each
/// slot's backend, its phase, and its revision when the daemon pinned one.
pub(crate) fn named_backends(op: &str, named: &Programs) -> Vec<Named> {
    named
        .iter()
        .filter(|(slot, _)| slot.op == op)
        .map(|(slot, program)| Named {
            phase: slot.phase,
            backend: program.backend.clone(),
            revision: (program.revision != 0).then_some(program.revision),
        })
        .collect()
}

/// The slots one operator's keys fill, from the Program Server's selection of them.
///
/// The selection names, for every key, the one package that builds it, so the packages' key sets
/// are disjoint. One package fills the operator's single slot; several each hold a part slot of
/// their own keys, named by the backend. A program the deployment named keeps the arguments the
/// daemon gave it; any other package runs on its own parameter defaults, which `prepare` fills.
///
/// A key no installed package covers refuses the operator, naming every such key: the forward
/// launches it, and no package of this store could build it.
pub(crate) fn selected_slots(
    op: &str,
    coverage: &program_server::Coverage,
    named: &Programs,
) -> std::result::Result<(Vec<(ProgramSlot, Program)>, Routes), String> {
    if !coverage.uncovered.is_empty() {
        return Err(format!(
            "no installed {op} package covers descriptor {} for this host's targets: the model's \
             forward launches it, so install a package whose [kernel.descriptors] declares it",
            coverage.uncovered.join(", ")
        ));
    }
    let divided = coverage.covered.len() > 1;
    let mut slots = Vec::with_capacity(coverage.covered.len());
    let mut routes = Routes::new();
    for covered in &coverage.covered {
        let backend = &covered.package.backend;
        let slot = ProgramSlot {
            op: op.to_owned(),
            phase: None,
            part: divided.then(|| backend.clone()),
        };
        let arguments = named
            .iter()
            .find(|(held, program)| held.op == op && program.backend == *backend)
            .map(|(_, program)| program.arguments.clone())
            .unwrap_or_default();
        slots.push((
            slot.clone(),
            Program {
                backend: backend.clone(),
                revision: covered.package.revision,
                arguments,
            },
        ));
        routes.insert(slot, covered.keys.clone());
    }
    Ok((slots, routes))
}

/// One operator's programs as a comparison reads them: each slot's phase and part with its
/// package, in a stated order. Arguments are left out - the two sides state a package's defaults in
/// different words - so a program is its backend and revision.
fn held<'a>(
    programs: &'a Programs,
    op: &str,
) -> Vec<(Option<PhaseKind>, Option<&'a str>, &'a str, u32)> {
    let mut held: Vec<_> = programs
        .iter()
        .filter(|(slot, _)| slot.op == op)
        .map(|(slot, program)| {
            (
                slot.phase,
                slot.part.as_deref(),
                program.backend.as_str(),
                program.revision,
            )
        })
        .collect();
    held.sort_unstable();
    held
}

/// The operators whose selected programs differ from what serves, including every operator nothing
/// serves yet: the operators a precopy installs. An operator the selection keeps exactly keeps its
/// kernels, and with them every bucket it launches in.
pub(crate) fn replaced_operators(selected: &Programs, serving: &Programs) -> BTreeSet<String> {
    selected
        .keys()
        .map(|slot| slot.op.as_str())
        .filter(|op| held(selected, op) != held(serving, op))
        .map(str::to_owned)
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
/// An operator is one operator however many packages fill it, so the comparison is between the two
/// sides' *sets* of programs per operator, not between slots that a division may have renamed.
pub(crate) fn changed_operators(prepared: &Programs, serving: &Programs) -> Vec<String> {
    let operators: BTreeSet<&str> = prepared
        .keys()
        .chain(serving.keys())
        .map(|slot| slot.op.as_str())
        .collect();
    operators
        .into_iter()
        .filter(|op| {
            let next = held(prepared, op);
            let old = held(serving, op);
            // An operator either side does not hold is new or retired, not changed: neither has a
            // pair of programs to compare.
            !next.is_empty() && !old.is_empty() && next != old
        })
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(backend: &str, revision: u32) -> Program {
        Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        }
    }

    /// An epoch's first precopy has no serving generation, so no operator has an old kernel to pair
    /// a shadowed step against: the changed set is empty and the precopy finishes without asking
    /// for a step an idle epoch never runs.
    #[test]
    fn a_first_generation_changes_no_operator_and_asks_for_no_shadow() {
        let mut prepared = Programs::new();
        prepared.insert(ProgramSlot::all("paged_attention"), program("fa4", 3));

        assert!(changed_operators(&prepared, &Programs::new()).is_empty());
        assert_eq!(
            replaced_operators(&prepared, &Programs::new()),
            BTreeSet::from(["paged_attention".to_owned()]),
            "a first generation installs every operator it selected"
        );
    }

    /// `changed` is the intersection where the two programs differ, and not `new`: an operator the
    /// prepared generation adds has no old kernel either, so it is not changed - but it is
    /// installed.
    #[test]
    fn only_an_operator_whose_program_differs_is_changed() {
        let mut serving = Programs::new();
        serving.insert(ProgramSlot::all("gemm"), program("cublas", 1));
        serving.insert(ProgramSlot::all("softmax"), program("reference", 2));

        let mut prepared = serving.clone();
        prepared.insert(ProgramSlot::all("gemm"), program("cublas", 2));
        prepared.insert(ProgramSlot::all("rms_norm"), program("reference", 1));

        assert_eq!(changed_operators(&prepared, &serving), vec!["gemm"]);
        assert_eq!(
            replaced_operators(&prepared, &serving),
            BTreeSet::from(["gemm".to_owned(), "rms_norm".to_owned()])
        );
    }

    /// An operator one package filled and two packages fill now is one changed operator, and the
    /// same set on both sides is no change.
    #[test]
    fn an_operator_filled_by_two_packages_is_one_changed_operator() {
        let part = |backend: &str| ProgramSlot {
            op: "paged_attention".to_owned(),
            phase: None,
            part: Some(backend.to_owned()),
        };
        let mut serving = Programs::new();
        serving.insert(ProgramSlot::all("paged_attention"), program("trtllm", 3));
        let mut prepared = Programs::new();
        prepared.insert(part("trtllm"), program("trtllm", 3));
        prepared.insert(part("triton"), program("triton", 7));

        assert_eq!(
            changed_operators(&prepared, &serving),
            vec!["paged_attention"]
        );
        assert!(changed_operators(&prepared, &prepared).is_empty());
        assert!(replaced_operators(&prepared, &prepared).is_empty());
    }

    /// A `cover` answer naming `entries` - each a package's backend and the keys it was selected
    /// for, all at revision 1 - and the keys no package covers.
    fn coverage(entries: &[(&str, &[&str])], uncovered: &[&str]) -> program_server::Coverage {
        program_server::Coverage {
            covered: entries
                .iter()
                .map(|(backend, keys)| program_server::Covered {
                    package: program_server::protocol::Package {
                        op: "dense_attention".to_owned(),
                        backend: (*backend).to_owned(),
                        revision: 1,
                    },
                    keys: keys.iter().map(|key| (*key).to_owned()).collect(),
                })
                .collect(),
            uncovered: uncovered.iter().map(|key| (*key).to_owned()).collect(),
        }
    }

    /// The keys two packages are selected for fill two part slots, each package with its own keys
    /// alone, so neither is asked for a key only the other declares.
    #[test]
    fn keys_two_packages_are_selected_for_fill_a_part_slot_each() {
        let bf16 = "dtype_bf16_head_dim_128";
        let f32 = "dtype_f32_head_dim_64";
        let answer = coverage(&[("f32", &[f32]), ("fa4", &[bf16])], &[]);

        let (slots, routes) = selected_slots("dense_attention", &answer, &Programs::new())
            .expect("every key is covered");
        let part = |backend: &str| ProgramSlot {
            op: "dense_attention".to_owned(),
            phase: None,
            part: Some(backend.to_owned()),
        };
        let held: Vec<(ProgramSlot, String)> = slots
            .iter()
            .map(|(slot, program)| (slot.clone(), program.backend.clone()))
            .collect();
        assert_eq!(
            held,
            vec![
                (part("f32"), "f32".to_owned()),
                (part("fa4"), "fa4".to_owned())
            ]
        );
        assert_eq!(routes[&part("f32")], vec![f32.to_owned()]);
        assert_eq!(routes[&part("fa4")], vec![bf16.to_owned()]);
    }

    /// Keys one package is selected for fill the operator's single slot, and a package the
    /// deployment named keeps the arguments the daemon gave it.
    #[test]
    fn keys_one_package_is_selected_for_fill_one_slot() {
        let keys = ["dtype_bf16_head_dim_128", "dtype_bf16_head_dim_64"];
        let answer = coverage(&[("fa4", &keys)], &[]);
        let mut named = Programs::new();
        let mut stated = program("fa4", 1);
        stated.arguments.insert("tile".to_owned(), 64.into());
        named.insert(ProgramSlot::all("dense_attention"), stated.clone());

        let (slots, routes) =
            selected_slots("dense_attention", &answer, &named).expect("every key is covered");
        let slot = ProgramSlot::all("dense_attention");
        assert_eq!(slots, vec![(slot.clone(), stated)]);
        assert_eq!(routes[&slot], keys.map(str::to_owned).to_vec());
    }

    /// A key the forward launches and no installed package covers refuses the operator, naming the
    /// operator and the key, rather than handing it to a package that would refuse it at build.
    #[test]
    fn a_key_no_package_covers_refuses_the_operator() {
        let uncovered = "dtype_f16_head_dim_96";
        let answer = coverage(&[("fa4", &["dtype_bf16_head_dim_128"])], &[uncovered]);

        let error = selected_slots("dense_attention", &answer, &Programs::new())
            .expect_err("a key is uncovered");
        assert!(error.contains("dense_attention"), "{error}");
        assert!(error.contains(uncovered), "{error}");
    }

    /// The named backends of an operator carry their phase, and a revision only once the daemon
    /// pinned one.
    #[test]
    fn named_backends_carry_their_phase_and_pinned_revision() {
        let mut named = Programs::new();
        named.insert(
            ProgramSlot::phase("paged_attention", PhaseKind::Prefill),
            program("trtllm", 4),
        );
        named.insert(
            ProgramSlot::phase("paged_attention", PhaseKind::Decode),
            program("triton", 0),
        );
        named.insert(ProgramSlot::all("gemm"), program("cublas", 1));

        assert_eq!(
            named_backends("paged_attention", &named),
            vec![
                Named {
                    phase: Some(PhaseKind::Prefill),
                    backend: "trtllm".to_owned(),
                    revision: Some(4),
                },
                Named {
                    phase: Some(PhaseKind::Decode),
                    backend: "triton".to_owned(),
                    revision: None,
                },
            ]
        );
    }
}
