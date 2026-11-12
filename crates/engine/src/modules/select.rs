//! Which packages a generation installs: the slots the Build Server's selection fills for each
//! operator's descriptor keys, the keys each slot's package is asked for, and the operators a
//! selection replaces or changes against what serves.
//!
//! The engine chooses nothing here. The Build Server owns coverage and selection
//! (`BuildServer::cover`): for every key it returns the one package that builds it, the backend
//! the deployment names first and the manifests' priorities after. This module only turns that
//! response into the slots a generation records.

use std::collections::{BTreeMap, BTreeSet};

use build_server::Preference;
use proto::module::{ModuleSlot, PackageRevision, Packages, PhaseKind};

/// The descriptor keys of each slot a selection filled, by slot: the keys its package was selected
/// for, and the only keys that package is asked to build.
pub(crate) type Routes = BTreeMap<ModuleSlot, Vec<String>>;

/// The backends a deployment names for `op`, as the Build Server's selection takes them: each
/// slot's backend, its phase, and its revision when the daemon pinned one. A slot with a part is
/// a candidate of the module set, not a preference, so it is none of these.
pub(crate) fn named_backends(op: &str, named: &Packages) -> Vec<Preference> {
    named
        .iter()
        .filter(|(slot, _)| slot.op == op && slot.part.is_none())
        .map(|(slot, package)| Preference {
            phase: slot.phase,
            backend: package.backend.clone(),
            revision: (package.revision != 0).then_some(package.revision),
        })
        .collect()
}

/// The revisions the epoch's module set fixed for `op`'s packages, each a backend and its
/// revision: every slot of the operator whose revision the daemon fixed. The Build Server
/// selects among exactly these, so a package or a revision installed after the set was fixed is
/// not run. Empty when the set states none, which leaves the selection to what is installed.
pub(crate) fn fixed_revisions(op: &str, named: &Packages) -> Vec<(String, u32)> {
    let mut fixed: Vec<(String, u32)> = named
        .iter()
        .filter(|(slot, package)| slot.op == op && package.revision != 0)
        .map(|(_, package)| (package.backend.clone(), package.revision))
        .collect();
    fixed.sort();
    fixed.dedup();
    fixed
}

/// The slots one operator's keys fill, from the Build Server's selection of them.
///
/// The selection names, for every key, the one package that builds it, so the packages' key sets
/// are disjoint. One package fills the operator's single slot; several each hold a part slot of
/// their own keys, named by the backend. A package the deployment named keeps the arguments the
/// daemon gave it; any other package runs on its own parameter defaults, which `prepare` fills.
///
/// A key no installed package covers refuses the operator, naming every such key: the forward
/// launches it, and no package of this store could build it.
pub(crate) fn selected_slots(
    op: &str,
    coverage: &build_server::Coverage,
    named: &Packages,
) -> std::result::Result<(Vec<(ModuleSlot, PackageRevision)>, Routes), String> {
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
    for (covering, keys) in &coverage.covered {
        let backend = &covering.backend;
        let slot = ModuleSlot {
            op: op.to_owned(),
            phase: None,
            part: divided.then(|| backend.clone()),
        };
        let arguments = named
            .iter()
            .find(|(held, package)| held.op == op && package.backend == *backend)
            .map(|(_, package)| package.arguments.clone())
            .unwrap_or_default();
        slots.push((
            slot.clone(),
            PackageRevision {
                backend: backend.clone(),
                revision: covering.revision,
                arguments,
            },
        ));
        routes.insert(slot, keys.clone());
    }
    Ok((slots, routes))
}

/// Refuses a module built for another version of `operator`'s interface than this build
/// launches (`proto::module::operator::version`).
///
/// `built_for` is the version the Build Server recorded with the module, which its package's
/// manifest states. A module of another version would be launched through a descriptor or an
/// argument layout it was not written for, so the generation that names it is not installed and
/// what serves is untouched.
pub(crate) fn interface_agrees(
    operator: &str,
    package: &PackageRevision,
    built_for: u32,
) -> std::result::Result<(), String> {
    let launched = proto::module::operator::version(operator);
    if built_for == launched {
        return Ok(());
    }
    Err(format!(
        "the {operator} package {} revision {} implements version {built_for} of the operator's \
         interface and this engine launches version {launched}",
        package.backend, package.revision
    ))
}

/// One operator's packages as a comparison reads them: each slot's phase and part with its
/// package, in a stated order. Arguments are left out - the two sides state a package's defaults in
/// different words - so a package is its backend and revision.
fn held<'a>(
    packages: &'a Packages,
    op: &str,
) -> Vec<(Option<PhaseKind>, Option<&'a str>, &'a str, u32)> {
    let mut held: Vec<_> = packages
        .iter()
        .filter(|(slot, _)| slot.op == op)
        .map(|(slot, package)| {
            (
                slot.phase,
                slot.part.as_deref(),
                package.backend.as_str(),
                package.revision,
            )
        })
        .collect();
    held.sort_unstable();
    held
}

/// The operators whose selected packages differ from what serves, including every operator nothing
/// serves yet: the operators a precopy installs. An operator the selection keeps exactly keeps its
/// kernels, and with them every bucket it launches in.
pub(crate) fn replaced_operators(selected: &Packages, serving: &Packages) -> BTreeSet<String> {
    selected
        .keys()
        .map(|slot| slot.op.as_str())
        .filter(|op| held(selected, op) != held(serving, op))
        .map(str::to_owned)
        .collect()
}

/// The operators whose packages differ between what serves and what a precopy prepared.
///
/// `changed` is not `new`. An operator the prepared generation holds and the serving one does not
/// has no old kernel to pair a shadowed step against, so it is not changed; the set is exactly the
/// intersection where the packages differ. It is empty when nothing serves yet - there is no old
/// kernel then either - which is what makes an epoch's first precopy pass without a shadow stage,
/// and why no rule here needs a generation number.
///
/// An operator is one operator however many packages fill it, so the comparison is between the two
/// sides' *sets* of packages per operator, not between slots that a division may have renamed.
pub(crate) fn changed_operators(prepared: &Packages, serving: &Packages) -> Vec<String> {
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
            // pair of packages to compare.
            !next.is_empty() && !old.is_empty() && next != old
        })
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(backend: &str, revision: u32) -> PackageRevision {
        PackageRevision {
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
        let mut prepared = Packages::new();
        prepared.insert(ModuleSlot::all("paged_attention"), package("fa4", 3));

        assert!(changed_operators(&prepared, &Packages::new()).is_empty());
        assert_eq!(
            replaced_operators(&prepared, &Packages::new()),
            BTreeSet::from(["paged_attention".to_owned()]),
            "a first generation installs every operator it selected"
        );
    }

    /// `changed` is the intersection where the two packages differ, and not `new`: an operator the
    /// prepared generation adds has no old kernel either, so it is not changed - but it is
    /// installed.
    #[test]
    fn only_an_operator_whose_package_differs_is_changed() {
        let mut serving = Packages::new();
        serving.insert(ModuleSlot::all("gemm"), package("cublas", 1));
        serving.insert(ModuleSlot::all("softmax"), package("reference", 2));

        let mut prepared = serving.clone();
        prepared.insert(ModuleSlot::all("gemm"), package("cublas", 2));
        prepared.insert(ModuleSlot::all("rms_norm"), package("reference", 1));

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
        let part = |backend: &str| ModuleSlot {
            op: "paged_attention".to_owned(),
            phase: None,
            part: Some(backend.to_owned()),
        };
        let mut serving = Packages::new();
        serving.insert(ModuleSlot::all("paged_attention"), package("trtllm", 3));
        let mut prepared = Packages::new();
        prepared.insert(part("trtllm"), package("trtllm", 3));
        prepared.insert(part("triton"), package("triton", 7));

        assert_eq!(
            changed_operators(&prepared, &serving),
            vec!["paged_attention"]
        );
        assert!(changed_operators(&prepared, &prepared).is_empty());
        assert!(replaced_operators(&prepared, &prepared).is_empty());
    }

    /// A `cover` response naming `entries` - each a package's backend and the keys it was selected
    /// for, all at revision 1 - and the keys no package covers.
    fn coverage(entries: &[(&str, &[&str])], uncovered: &[&str]) -> build_server::Coverage {
        build_server::Coverage {
            covered: entries
                .iter()
                .map(|(backend, keys)| {
                    (
                        build_server::protocol::Package {
                            op: "dense_attention".to_owned(),
                            backend: (*backend).to_owned(),
                            revision: 1,
                        },
                        keys.iter().map(|key| (*key).to_owned()).collect(),
                    )
                })
                .collect(),
            uncovered: uncovered.iter().map(|key| (*key).to_owned()).collect(),
            candidates: Vec::new(),
        }
    }

    /// The keys two packages are selected for fill two part slots, each package with its own keys
    /// alone, so neither is asked for a key only the other declares.
    #[test]
    fn keys_two_packages_are_selected_for_fill_a_part_slot_each() {
        let bf16 = "dtype_bf16_head_dim_128";
        let f32 = "dtype_f32_head_dim_64";
        let covering = coverage(&[("f32", &[f32]), ("fa4", &[bf16])], &[]);

        let (slots, routes) = selected_slots("dense_attention", &covering, &Packages::new())
            .expect("every key is covered");
        let part = |backend: &str| ModuleSlot {
            op: "dense_attention".to_owned(),
            phase: None,
            part: Some(backend.to_owned()),
        };
        let held: Vec<(ModuleSlot, String)> = slots
            .iter()
            .map(|(slot, package)| (slot.clone(), package.backend.clone()))
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
        let covering = coverage(&[("fa4", &keys)], &[]);
        let mut named = Packages::new();
        let mut stated = package("fa4", 1);
        stated.arguments.insert("tile".to_owned(), 64.into());
        named.insert(ModuleSlot::all("dense_attention"), stated.clone());

        let (slots, routes) =
            selected_slots("dense_attention", &covering, &named).expect("every key is covered");
        let slot = ModuleSlot::all("dense_attention");
        assert_eq!(slots, vec![(slot.clone(), stated)]);
        assert_eq!(routes[&slot], keys.map(str::to_owned).to_vec());
    }

    /// A key the forward launches and no installed package covers refuses the operator, naming the
    /// operator and the key, rather than handing it to a package that would refuse it at build.
    #[test]
    fn a_key_no_package_covers_refuses_the_operator() {
        let uncovered = "dtype_f16_head_dim_96";
        let covering = coverage(&[("fa4", &["dtype_bf16_head_dim_128"])], &[uncovered]);

        let error = selected_slots("dense_attention", &covering, &Packages::new())
            .expect_err("a key is uncovered");
        assert!(error.contains("dense_attention"), "{error}");
        assert!(error.contains(uncovered), "{error}");
    }

    /// A module built for the version of its operator's interface this build launches is
    /// accepted, and one built for another version is refused by name.
    #[test]
    fn a_module_of_another_interface_version_is_refused() {
        let package = package("triton", 4);
        let launched = proto::module::operator::version("rms_norm");

        interface_agrees("rms_norm", &package, launched).expect("the versions agree");
        let refusal = interface_agrees("rms_norm", &package, launched + 1)
            .expect_err("another version is refused");
        assert!(refusal.contains("rms_norm"), "{refusal}");
        assert!(refusal.contains("triton"), "{refusal}");
        assert!(refusal.contains(&(launched + 1).to_string()), "{refusal}");
    }

    /// The engine package's manifest states the engine interface's version, which carries the
    /// storage protocol's: an engine built from this tree is the one a daemon of this tree
    /// accepts at its hello.
    #[test]
    fn the_engine_package_states_the_engine_interface_version() {
        let manifest: toml::Table = include_str!("../../package.toml")
            .parse()
            .expect("the engine package's manifest is TOML");

        assert_eq!(
            manifest["interface_version"].as_integer(),
            Some(i64::from(proto::ENGINE_INTERFACE_VERSION))
        );
    }

    /// The revisions a module set fixed for an operator are its named packages' and its
    /// candidates', each once; a candidate is no preference.
    #[test]
    fn a_module_sets_candidates_are_fixed_and_not_preferred() {
        let candidate = |backend: &str| ModuleSlot {
            op: "paged_attention".to_owned(),
            phase: None,
            part: Some(backend.to_owned()),
        };
        let mut set = Packages::new();
        set.insert(ModuleSlot::all("paged_attention"), package("trtllm", 4));
        set.insert(candidate("trtllm"), package("trtllm", 4));
        set.insert(candidate("triton"), package("triton", 2));
        set.insert(ModuleSlot::all("gemm"), package("cublas", 0));

        assert_eq!(
            fixed_revisions("paged_attention", &set),
            vec![("triton".to_owned(), 2), ("trtllm".to_owned(), 4)]
        );
        assert_eq!(
            named_backends("paged_attention", &set),
            vec![Preference {
                phase: None,
                backend: "trtllm".to_owned(),
                revision: Some(4),
            }]
        );
        assert!(
            fixed_revisions("gemm", &set).is_empty(),
            "a revision the daemon has not fixed is not stated"
        );
    }

    /// The named backends of an operator carry their phase, and a revision only once the daemon
    /// pinned one.
    #[test]
    fn named_backends_carry_their_phase_and_pinned_revision() {
        let mut named = Packages::new();
        named.insert(
            ModuleSlot::phase("paged_attention", PhaseKind::Prefill),
            package("trtllm", 4),
        );
        named.insert(
            ModuleSlot::phase("paged_attention", PhaseKind::Decode),
            package("triton", 0),
        );
        named.insert(ModuleSlot::all("gemm"), package("cublas", 1));

        assert_eq!(
            named_backends("paged_attention", &named),
            vec![
                Preference {
                    phase: Some(PhaseKind::Prefill),
                    backend: "trtllm".to_owned(),
                    revision: Some(4),
                },
                Preference {
                    phase: Some(PhaseKind::Decode),
                    backend: "triton".to_owned(),
                    revision: None,
                },
            ]
        );
    }
}
