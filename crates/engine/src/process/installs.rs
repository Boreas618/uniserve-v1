//! The kernel installs of a precopy: the generation's programs resolved against this host's
//! Program Server store, one install per device admitted on the module-load lane, and the walk to an
//! operator's next candidate program when the store refuses one for a shape of the generation.

use std::collections::BTreeMap;
use std::sync::Arc;

use ipc::UpdateFailure;
use program_server::program::Programs;
use program_server::{Descriptors, PrepareRequest, Program, ProgramRequest, Target};
use worker::host::HostTask;

use super::MainProcess;
use super::precopy::Precopy;
use crate::programs::modules::{Install, ModuleWork, Unavailable, UnavailableCell};
use crate::programs::select::{
    Candidates, built_keys, declared_program, descriptor_keys, key_fits_phase, merge_request,
    resolve_slots, select_programs,
};

/// What the walk does when a program cannot serve a shape of this generation.
pub(super) enum Advance {
    /// The operator's next candidate was chosen and its install admitted.
    Retry(Vec<Arc<HostTask<ModuleWork>>>),
    /// Every program this host built for the operator has been tried.
    Exhausted(String),
    /// The programs are the daemon's, so there is no candidate to move to.
    Named,
    /// The store could not be read, so no candidate can be chosen.
    Failed(String),
}

impl MainProcess {
    /// The epoch's first generation: the store's set, with the deployment's named programs deciding
    /// the phases their own packages cover.
    ///
    /// A named program chooses its operator's implementation - its backend and revision - and never
    /// states the operator set: a deployment that pins one operator's kernel still runs every other
    /// operator its buckets launch. So the first generation takes this host's store as its set and
    /// the named programs resolved against it, which is the only reading under which `kernels=1` on
    /// a host that built twenty-seven launches a model rather than a fragment.
    ///
    /// A named program also states nothing about the phases its operator's key spans, and an
    /// operator split across two packages - a prefill module and a decode one - needs both, so the
    /// name chooses the implementation *for its own keys* and the store keeps the other phases
    /// (`operator_slots`).
    pub(super) fn first_generation(
        &self,
        named: &Programs,
    ) -> std::result::Result<(Programs, Candidates), UpdateFailure> {
        let (listed, ordered) = self.listed_programs(named)?;
        tracing::info!(
            slots = listed.len(),
            named = named.len(),
            operators = ordered.len(),
            alternative = ordered.values().filter(|held| held.len() > 1).count(),
            "this host's built programs, with the deployment's named ones resolved first"
        );
        Ok((listed, ordered))
    }

    /// The Program Server's store as it stands: every installed package, its revision and the
    /// descriptors its modules cover for each target.
    fn listing(&self) -> std::result::Result<program_server::Listing, String> {
        // One read per precopy, on the module-load lane; every later reader of that precopy answers
        // from it. The update asks twice - once for the set, once for the keys a package declares -
        // and a second read here would put the unbounded call back on the control thread, which is
        // where the lane exists to keep it from. A reader with no listing in hand is one outside a
        // precopy, and it reads the store with the lines below.
        if let Some(store) = &self.store {
            return Ok(store.clone());
        }
        tracing::info!("asking the Program Server for its installed programs");
        let mut client = self
            .programs
            .lock()
            .map_err(|_| "the Program Server client was poisoned".to_owned())?;
        let listing = client
            .list()
            .map_err(|error| format!("reading the Program Server's store: {error}"))?;
        tracing::info!(
            packages = listing.packages.len(),
            "the Program Server's listing answered"
        );
        Ok(listing)
    }

    /// The programs this host has built, one per operator, for a precopy that names none.
    ///
    /// The epoch's first generation has no serving generation to take keys from and a startup
    /// switch states the scheduling policy alone, so the engine resolves what it can launch: the
    /// store's listing is the authority, because its packages are the modules actually built for
    /// this host's targets.
    ///
    /// A program is chosen by **identity**, not by a specification: the operator's highest installed
    /// revision, ties broken by the backend's name so the choice is the same on every run. The
    /// listing carries no build arguments and none are needed - the store matches a program on
    /// `(operator, backend)` and fills the package's own parameter defaults, so `arguments` only ever
    /// overrides a default that the package already states.
    ///
    /// An operator built under more than one backend is one slot with several candidates, which is
    /// why the answer carries the whole ordered list: the first is installed, and the walk moves to
    /// the next one if the first cannot serve a descriptor it meets (`advance_candidate`). A
    /// deployment that wants one particular backend says so through the daemon, which names it in
    /// the `Precopy`, and then the walk does not choose at all.
    fn listed_programs(
        &self,
        named: &Programs,
    ) -> std::result::Result<(Programs, Candidates), UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
        let listing = self.listing().map_err(invalid)?;
        let chosen = select_programs(&listing.packages, &self.targets);
        if chosen.is_empty() && named.is_empty() {
            return Err(invalid(format!(
                "the precopy named no kernel program and this host has built none: the store's \
                 listing holds {} package(s) and none has a module for this host's targets, so \
                 there is nothing to install",
                listing.packages.len()
            )));
        }
        tracing::info!(
            packages = listing.packages.len(),
            operators = chosen.len(),
            named = named.len(),
            "resolved this host's built programs, the deployment's named ones first"
        );
        // What this host's store supplies, against what this build implements: the difference is
        // the fact an operator needs, because it names the package nobody built (an operator whose
        // module has no package cannot be supplied by the deployment either), and it is bounded -
        // an operator the deployment never launches may be in it, which is why the line states the
        // count rather than claiming these are needed.
        let supplied: std::collections::BTreeSet<&str> = chosen
            .keys()
            .filter_map(|op| ops::operator_name(op))
            .collect();
        let missing: Vec<&str> = ops::operators()
            .iter()
            .copied()
            .filter(|operator| !supplied.contains(operator))
            .collect();
        if !missing.is_empty() {
            tracing::warn!(
                supplied = supplied.len(),
                implemented = ops::operators().len(),
                missing = ?missing,
                "this host's store supplies fewer operators than the build implements"
            );
        }
        // What this host has built for every package the store holds, keyed by the package -
        // (operator, backend, revision) - because that is what a phase split is read from: the keys
        // a package built are the phases it can serve.
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        for package in &listing.packages {
            let mut keys: Vec<String> = package
                .modules
                .iter()
                .filter(|module| self.targets.contains(&module.target))
                .map(|module| module.descriptor.clone())
                .collect();
            keys.sort_unstable();
            keys.dedup();
            if !keys.is_empty() {
                built.insert(
                    (
                        package.op.clone(),
                        package.backend.clone(),
                        package.revision,
                    ),
                    keys,
                );
            }
        }
        resolve_slots(named, &chosen, &built, &listing.packages).map_err(invalid)
    }

    /// One operator's program from this host's store, for a walk that named an operator the
    /// deployment left unnamed.
    ///
    /// The store's listing is the authority for what this host can launch: its packages are the
    /// modules built for this host's targets. The first candidate of the operator is taken - the
    /// deployment expressed no preference, so the store's order decides - and `None` says the store
    /// describes nothing for the operator, which the caller states as a missing package rather than
    /// a missing key.
    pub(super) fn borrow_program(
        &self,
        operator: &str,
    ) -> std::result::Result<Option<(program_server::program::ProgramSlot, Program)>, String> {
        let listing = self.listing()?;
        for (op, ordered) in select_programs(&listing.packages, &self.targets) {
            if ops::operator_name(&op) != Some(operator) {
                continue;
            }
            let Some((backend, revision)) = ordered.first() else {
                continue;
            };
            return Ok(Some((
                program_server::program::ProgramSlot {
                    op,
                    phase: None,
                    part: None,
                },
                Program {
                    backend: backend.clone(),
                    revision: *revision,
                    arguments: Default::default(),
                },
            )));
        }
        // A package the store holds and has never built is invisible to `select_programs`, and an
        // operator no model forward reaches is exactly that: `slot_commit` is launched by the
        // runner's own step, so no capture it is missing from comes before the walk names it, and
        // nothing in the deployment can name its package either. The store holds the package and
        // declares the keys it can build (`[kernel.descriptors]`), so installing it and asking for
        // the walked key is what builds it - otherwise a generation refuses an operator the store
        // holds, which reads as a store defect and is not one.
        match declared_program(&self.listing()?.packages, operator) {
            Some((op, backend, revision)) => Ok(Some((
                program_server::program::ProgramSlot {
                    op,
                    phase: None,
                    part: None,
                },
                Program {
                    backend,
                    revision,
                    arguments: Default::default(),
                },
            ))),
            None => Ok(None),
        }
    }

    /// The module-load tasks one generation needs, one per device.
    ///
    /// An update changes implementations and never the set of descriptors, so the keys each
    /// operator must cover are the ones the serving generation holds. The epoch's *first*
    /// generation has no serving generation to take them from, and nothing else can state them: a
    /// descriptor is what the model's forward computes for a bucket, and only a walk of that
    /// forward says which ones it launches. So the first generation starts with the keys already
    /// learned and finds the rest by walking - each attempt that finds a kernel missing names the
    /// operator and key it needed, and the install is admitted again with that key added.
    #[expect(
        clippy::type_complexity,
        reason = "the modules to fetch and the cell they leave unavailable are one answer"
    )]
    pub(super) fn admit_installs(
        &mut self,
        programs: &Programs,
        learned: &BTreeMap<String, Vec<String>>,
    ) -> std::result::Result<(Vec<Arc<HostTask<ModuleWork>>>, UnavailableCell), UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
        // The Program Server's listing, read at most once and only when a program's keys cannot be
        // taken from a generation that already launched it: the epoch's first generation has
        // nothing to copy from, and the walk would otherwise discover one descriptor per capture.
        let mut listing: Option<program_server::Listing> = None;
        // One request per package, not one per generation: the Program Server hands every chosen
        // slot of an operator all the keys a request names for that operator, so a generation whose
        // operator the store split across two backends - fa4's prefill module and trtllm's decode
        // one - would have each of them asked for the other's keys, and the package that cannot
        // build a phase it does not declare would refuse the generation. The keys of a slot are
        // its own package's, which is what makes the package the request's unit.
        let mut groups: BTreeMap<(String, u32), (Vec<ProgramRequest>, Vec<Descriptors>)> =
            BTreeMap::new();
        for (slot, program) in programs {
            let Some(operator) = ops::operator_name(&slot.op) else {
                return Err(invalid(format!(
                    "operator {} has no implementation in this build",
                    slot.op
                )));
            };
            // A generation may hold two packages of one operator, one per phase: a key belongs to
            // the phase its own text states, so a phase's slot takes only the keys of its phase and
            // no package is ever asked for a key it cannot build. A key that states no phase fits
            // every slot - it is the only key a phase-less operator has.
            let mut keys: Vec<String> = descriptor_keys(&self.generations[0], operator)
                .into_iter()
                .filter(|key| key_fits_phase(key, slot.phase))
                .collect();
            // A generation that launched this operator states the keys it runs; a key the walk has
            // taught since is added to them, never substituted for them. The first generation has
            // nothing serving, so its keys come from the package - once, not one capture at a time -
            // and a taught key joins them for the packages the store does not describe.
            let mut known: Vec<String> = learned
                .get(operator)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|key| key_fits_phase(key, slot.phase))
                .collect();
            if keys.is_empty() {
                if listing.is_none() {
                    let read = self.listing().map_err(invalid)?;
                    listing = Some(read);
                }
                // The slot was filled immediately above. Reading it through a match rather than a
                // panic keeps a fault in this function a named refusal, not a dead process: the
                // workspace denies `expect` for exactly that reason.
                let declared = match listing.as_ref() {
                    Some(store) => built_keys(store, &slot.op, program, &self.targets),
                    None => {
                        return Err(invalid(
                            "the Program Server's listing was not read".to_owned(),
                        ));
                    }
                };
                if declared.is_empty() {
                    // The listing names the modules this host has *built*, not what the package
                    // declares, so an empty answer is a host that built none yet - not a program
                    // that covers nothing. The walk is the authority on which descriptors the
                    // model needs, and it teaches them one capture at a time; the listing only
                    // makes the first capture enough when the store already holds them.
                    tracing::debug!(
                        op = %slot.op,
                        backend = %program.backend,
                        revision = program.revision,
                        "no module of this package is built yet: the walk will name its descriptors"
                    );
                }
                keys = declared;
            }
            keys.append(&mut known);
            // A slot of a divided operator holds its own package's descriptors alone: a generation
            // that launched the operator states all of them.
            if slot.part.is_some() {
                if listing.is_none() {
                    listing = Some(self.listing().map_err(invalid)?);
                }
                if let Some(store) = listing.as_ref() {
                    let built = built_keys(store, &slot.op, program, &self.targets);
                    keys.retain(|key| built.contains(key));
                }
            }
            keys.sort_unstable();
            keys.dedup();
            let group = groups
                .entry((program.backend.clone(), program.revision))
                .or_default();
            merge_request(group, slot, program, keys);
        }

        let mut tasks = Vec::with_capacity(self.workers.len());
        // One cell shared by every device's load of this generation: the first structured refusal
        // any of them meets is the one the walk answers with another candidate program.
        let unavailable = UnavailableCell::default();
        for worker in &self.workers {
            let ordinal = worker.device();
            let device = platform::Device::new(ordinal)
                .map_err(|error| invalid(format!("opening device {}: {error}", ordinal.0)))?;
            // `targets` holds each distinct architecture once, so a device's own is read from it.
            let target = Target::of(ordinal)
                .map_err(|error| invalid(format!("reading device {}: {error}", ordinal.0)))?;
            if !self.targets.contains(&target) {
                return Err(invalid(format!("device {} has no build target", ordinal.0)));
            }
            // One request per package, in the store's own order so two runs ask alike.
            let requests: Vec<PrepareRequest> = groups
                .values()
                .map(|(programs, descriptors)| PrepareRequest {
                    targets: vec![target.clone()],
                    programs: programs.clone(),
                    descriptors: descriptors.clone(),
                })
                .collect();
            let install = Install {
                programs: Arc::clone(&self.programs),
                device,
                target,
                requests,
                unavailable: Arc::clone(&unavailable),
            };
            match self.modules.admit(ModuleWork::Install(install)) {
                Ok(task) => tasks.push(task),
                Err(error) => {
                    return Err(invalid(format!("admitting a module load: {error}")));
                }
            }
        }
        Ok((tasks, unavailable))
    }

    /// Moves the generation on to the next program its operator may be filled by.
    ///
    /// Each candidate this host built for the operator is tried once, in a stated order: the
    /// candidate whose built modules cover the most of the keys learned so far first, then the
    /// highest revision, then the backend's name - so two runs choose alike and a store with two
    /// backends costs one extra install rather than a failed generation. When every candidate has
    /// been tried the refusal names the operator, the descriptor it could not serve, and every
    /// backend tried; it never falls back silently and never speaks generically.
    ///
    /// A program the daemon named is the deployment's choice and is not second-guessed here: with no
    /// candidate list the walk answers [`Advance::Named`] and the refusal stands as it is.
    pub(super) fn advance_candidate(
        &mut self,
        precopy: &mut Precopy,
        refused: &Unavailable,
    ) -> Advance {
        let op = refused.op().to_owned();
        let Some(ordered) = precopy.candidates.get(&op).cloned() else {
            return Advance::Named;
        };
        // The slot the walk moves on is the one whose *program* the store refused: an operator the
        // store splits across two packages has a slot per phase, and advancing the other one would
        // replace a program that served its own keys perfectly.
        let refused_backend = refused.backend().to_owned();
        let Some((slot, current)) = precopy
            .programs
            .iter()
            .find(|(slot, program)| slot.op == op && program.backend == refused_backend)
            .or_else(|| precopy.programs.iter().find(|(slot, _)| slot.op == op))
            .map(|(slot, program)| (slot.clone(), program.clone()))
        else {
            return Advance::Named;
        };
        precopy
            .tried
            .insert((current.backend.clone(), current.revision));
        let tried = precopy.backends.entry(op.clone()).or_default();
        if !tried.contains(&current.backend) {
            tried.push(current.backend.clone());
        }

        let listing = match self.listing() {
            Ok(listing) => listing,
            Err(reason) => return Advance::Failed(reason),
        };
        let learned = precopy.learned.get(&op).cloned().unwrap_or_default();
        let mut remaining: Vec<(usize, (String, u32))> = ordered
            .into_iter()
            .filter(|candidate| !precopy.tried.contains(candidate))
            .map(|candidate| {
                let program = Program {
                    backend: candidate.0.clone(),
                    revision: candidate.1,
                    arguments: Default::default(),
                };
                let covers = built_keys(&listing, &op, &program, &self.targets)
                    .iter()
                    .filter(|key| learned.contains(key))
                    .count();
                (covers, candidate)
            })
            .collect();
        // The stated order: coverage of the keys learned so far, then the highest revision, then the
        // backend name. Coverage only orders; it never excludes a candidate.
        remaining.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.1.cmp(&left.1.1))
                .then_with(|| left.1.0.as_str().cmp(right.1.0.as_str()))
        });
        let Some((covers, next)) = remaining.into_iter().next() else {
            // Every candidate is spent: the refusal names the operator, what it could not serve and
            // every backend that was tried, so the next reader knows what to install.
            let backends = precopy.backends.get(&op).cloned().unwrap_or_default();
            let descriptor = refused.descriptor().unwrap_or("the key it was asked for");
            return Advance::Exhausted(format!(
                "{op} has no program this host built that serves {descriptor}: every candidate was \
                 tried (backends: {}), so this generation cannot be installed from this host's \
                 store",
                backends.join(", ")
            ));
        };
        tracing::info!(
            op = %op,
            from = %current.backend,
            to = %next.0,
            revision = next.1,
            covers,
            learned = learned.len(),
            "a program cannot serve this generation's shape: trying the operator's next candidate"
        );
        precopy.programs.insert(
            slot,
            Program {
                backend: next.0.clone(),
                revision: next.1,
                arguments: Default::default(),
            },
        );
        match self.admit_installs(&precopy.programs, &precopy.learned) {
            Ok((tasks, unavailable)) => {
                precopy.unavailable = unavailable;
                Advance::Retry(tasks)
            }
            Err(failure) => Advance::Failed(failure.reason()),
        }
    }
}
