//! Which packages a generation installs: the slots the Build Server's selection fills for each
//! operator's descriptor keys, the keys each slot's package is asked for, what the devices of a
//! start serve together, and the operators a selection replaces or changes against what serves.
//!
//! The engine chooses nothing here. The Build Server owns coverage and selection
//! (`BuildServer::cover`): for every key it returns the one package that builds it, the package
//! the deployment's `[kernels]` table pins for the key first and the manifests' priorities after.
//! This module only turns that response into the slots a generation records.

use std::collections::{BTreeMap, BTreeSet};

use proto::module::{ModuleSlot, PackageRevision, Packages, Pin};

/// The descriptor keys of each slot a selection filled, by slot: the keys its package was selected
/// for, and the only keys that package is asked to build.
pub type Routes = BTreeMap<ModuleSlot, Vec<String>>;

/// The deployment's pins of `op`'s descriptor keys, as the Build Server's selection takes them:
/// each pinned slot's key with its package's backend and revision.
///
/// Every revision of a launch's module set is fixed: the daemon fixes each package the
/// deployment pins, at the revision it states or its latest installed one, before it launches
/// an epoch or prepares an update (`daemon::engine::control::with_store_modules`), so the engine
/// passes each one on as it is and the Build Server refuses a revision it does not hold.
pub(crate) fn pins(op: &str, named: &Packages) -> BTreeMap<String, Pin> {
    named
        .iter()
        .filter(|(slot, _)| slot.op == op)
        .filter_map(|(slot, package)| {
            let pin = Pin {
                backend: package.backend.clone(),
                revision: Some(package.revision),
            };
            Some((slot.key.clone()?, pin))
        })
        .collect()
}

/// The revisions the epoch's module set fixed for `op`'s packages, each a backend and its
/// revision: every slot of the operator, named or a candidate. The Build Server selects among
/// exactly these, so a package or a revision installed after the set was fixed is not run. Empty
/// when the set names no package of the operator, which leaves the selection to what is
/// installed.
pub(crate) fn fixed_revisions(op: &str, named: &Packages) -> Vec<(String, u32)> {
    let mut fixed: Vec<(String, u32)> = named
        .iter()
        .filter(|(slot, _)| slot.op == op)
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
            key: None,
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

/// What the devices of a start serve together, from `selections`, each device's selection of the
/// keys its own profile run names ([`crate::modules::load::select`]): every operator's packages
/// in the slots one selection over the keys of every device fills, one package in the
/// operator's single slot and several in a part slot each, named by the backend
/// ([`selected_slots`]).
///
/// The Build Server selects each key's package alone, so the packages the devices' selections
/// name are those an update's selection over all their keys names, and an operator an update
/// keeps compares as unchanged ([`replaced_operators`], [`changed_operators`]) however the
/// devices' keys divided it. A package is its backend: the module set fixes one revision of each.
pub(crate) fn served<'a>(selections: impl IntoIterator<Item = &'a Packages>) -> Packages {
    let mut by_operator: BTreeMap<&str, BTreeMap<&str, &PackageRevision>> = BTreeMap::new();
    for selection in selections {
        for (slot, package) in selection {
            by_operator
                .entry(slot.op.as_str())
                .or_default()
                .entry(package.backend.as_str())
                .or_insert(package);
        }
    }

    let mut served = Packages::new();
    for (op, packages) in by_operator {
        let divided = packages.len() > 1;
        for (backend, package) in packages {
            let slot = ModuleSlot {
                op: op.to_owned(),
                key: None,
                part: divided.then(|| backend.to_owned()),
            };
            served.insert(slot, package.clone());
        }
    }
    served
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

/// One operator's packages as a comparison reads them: each slot's pinned key and part with its
/// package, in a stated order. Arguments are left out - the two sides state a package's defaults in
/// different words - so a package is its backend and revision.
fn held<'a>(
    packages: &'a Packages,
    op: &str,
) -> Vec<(Option<&'a str>, Option<&'a str>, &'a str, u32)> {
    let mut held: Vec<_> = packages
        .iter()
        .filter(|(slot, _)| slot.op == op)
        .map(|(slot, package)| {
            (
                slot.key.as_deref(),
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
            key: None,
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
            key: None,
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

    /// A start serves what its devices selected, in the slots one selection over every device's
    /// keys fills: one device whose keys one package covers and one whose keys two packages
    /// divide serve the two packages in a part slot each. An update that selects the same
    /// packages for the same keys under the module set's candidates, which name every backend
    /// installed, replaces and changes no operator, so it installs no kernel again and shadows
    /// none.
    #[test]
    fn a_start_serves_what_its_devices_selected_and_an_update_of_the_same_changes_nothing() {
        let bf16 = "dtype_bf16_head_dim_128";
        let f32 = "dtype_f32_head_dim_64";
        let candidate = |backend: &str| ModuleSlot {
            op: "dense_attention".to_owned(),
            key: None,
            part: Some(backend.to_owned()),
        };
        let mut candidates = Packages::new();
        for backend in ["f32", "fa4", "flashinfer"] {
            candidates.insert(candidate(backend), package(backend, 1));
        }
        let selection = |covering: &build_server::Coverage| {
            let (slots, _) = selected_slots("dense_attention", covering, &candidates)
                .expect("every key is covered");
            slots.into_iter().collect::<Packages>()
        };

        let one = selection(&coverage(&[("fa4", &[bf16])], &[]));
        let two = selection(&coverage(&[("f32", &[f32]), ("fa4", &[bf16])], &[]));
        let serving = served([&one, &two]);

        let update = selection(&coverage(&[("f32", &[f32]), ("fa4", &[bf16])], &[]));
        assert_eq!(serving, update);
        assert!(replaced_operators(&update, &serving).is_empty());
        assert!(changed_operators(&update, &serving).is_empty());

        // Alone, the first device serves its one package in the operator's single slot.
        assert_eq!(
            served([&one]),
            Packages::from([(ModuleSlot::all("dense_attention"), package("fa4", 1))])
        );
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

    /// The revisions a module set fixed for an operator are its pinned packages' and its
    /// candidates', each once; a candidate is no pin.
    #[test]
    fn a_module_sets_candidates_are_fixed_and_not_pinned() {
        let candidate = |backend: &str| ModuleSlot {
            op: "paged_attention".to_owned(),
            key: None,
            part: Some(backend.to_owned()),
        };
        let decode = "phase_decode_head_128";
        let mut set = Packages::new();
        set.insert(
            ModuleSlot::key("paged_attention", decode),
            package("trtllm", 4),
        );
        set.insert(candidate("trtllm"), package("trtllm", 4));
        set.insert(candidate("triton"), package("triton", 2));
        set.insert(ModuleSlot::key("gemm", "m_64"), package("cublas", 1));

        assert_eq!(
            fixed_revisions("paged_attention", &set),
            vec![("triton".to_owned(), 2), ("trtllm".to_owned(), 4)]
        );
        assert_eq!(
            pins("paged_attention", &set),
            BTreeMap::from([(
                decode.to_owned(),
                Pin {
                    backend: "trtllm".to_owned(),
                    revision: Some(4),
                }
            )])
        );
        assert!(
            fixed_revisions("rms_norm", &set).is_empty(),
            "an operator the set names no package of leaves the selection to the store"
        );
    }
}
