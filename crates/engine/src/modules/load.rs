//! The engine's module load at the Build Server, as plain functions its callers run where they
//! may wait: the selection of a generation's kernel packages for the keys its devices launch
//! ([`select`]), one device's installs of the selected packages' modules ([`install`]), and the
//! scheduling policy's module ([`policy`]).
//!
//! A rank's start calls them on its load thread (`specs/engine-worker-port.md` §1.1, step 9) and
//! a generation update on a job thread of its own: a call to the Build Server has no deadline of
//! its own, since a module it does not hold is built first, and a build takes minutes, so no
//! scheduler, worker or control thread makes one. Each module is loaded into the device's
//! context as a kernel: a cubin module's image with its launch metadata, or a native module's
//! library from the store's own file, each the file the Build Server's module contract names for
//! its form (`build_server::source`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use build_server::{
    Client, CoverRequest, Descriptors, PackageRequest, PrepareRequest, Selector, Target,
};
use ops::KernelEntry;
use platform::DeviceId;
use proto::module::{PackageRevision, Packages};
use scheduling::limits::Parameters;

use crate::modules::select::{
    Routes, fixed_revisions, interface_agrees, named_backends, selected_slots,
};

/// The Build Server's selection of a generation's packages for `keys`, every descriptor key the
/// generation launches by operator, on `targets`, the devices' architectures: the packages by
/// slot, and the keys each slot's package builds.
///
/// Every operator is asked once, with every key it launches and the backends the deployment
/// names for it in `named`; the server returns the package of each key (`BuildServer::cover`).
///
/// # Errors
///
/// An operator this build has no implementation of, the Build Server's refusal, or a key no
/// installed package covers.
pub fn select(
    client: &mut Client,
    targets: &[Target],
    named: &Packages,
    keys: &BTreeMap<String, BTreeSet<String>>,
) -> Result<(Packages, Routes)> {
    let mut packages = Packages::new();
    let mut routes = Routes::new();

    for (op, keys) in keys {
        if proto::module::operator::named(op).is_none() {
            bail!("operator {op} has no implementation in this build");
        }
        if keys.is_empty() {
            continue;
        }

        let request = CoverRequest {
            op: op.clone(),
            targets: targets.to_vec(),
            keys: keys.iter().cloned().collect(),
            preferred: named_backends(op, named),
            fixed: fixed_revisions(op, named),
        };
        let coverage = client
            .cover(&request)
            .with_context(|| format!("asking the Build Server to select {op}'s packages"))?;
        let (slots, selected) = selected_slots(op, &coverage, named).map_err(anyhow::Error::msg)?;
        for (slot, package) in &slots {
            tracing::info!(
                %op,
                backend = %package.backend,
                revision = package.revision,
                part = slot.part.as_deref().unwrap_or(""),
                keys = selected.get(slot).map_or(0, Vec::len),
                named = !request.preferred.is_empty(),
                "the Build Server selected a package for this operator's keys"
            );
        }
        packages.extend(slots);
        routes.extend(selected);
    }
    Ok((packages, routes))
}

/// `device`'s kernels of the generation whose packages are `packages`, each slot's package asked
/// for exactly the keys `routes` selected for it: the Build Server prepares every slot's modules
/// in one request for the device's architecture, each module is resolved, and each is loaded
/// into the device's context. A generation that changes no kernel package asks nothing.
///
/// # Errors
///
/// A device that states no architecture or has no context, the Build Server's refusal to prepare
/// or resolve a module, a module whose interface or form this build does not install, or the
/// driver's refusal to load it.
pub fn install(
    client: &mut Client,
    device: DeviceId,
    packages: &Packages,
    routes: &Routes,
) -> Result<Vec<KernelEntry>> {
    if packages.is_empty() {
        return Ok(Vec::new());
    }

    let capability = device
        .compute_capability()
        .with_context(|| format!("reading device {}'s architecture", device.0))?;
    let target = Target::arch(capability)?;
    let ordinal = device;
    let device =
        platform::Device::new(device).with_context(|| format!("opening device {}", ordinal.0))?;

    // The generation's one admission: every slot's package with exactly the keys selected for
    // it, so the Build Server prepares the whole generation at once and no package is asked for
    // another slot's keys.
    let request = PrepareRequest {
        targets: vec![target.clone()],
        packages: packages
            .iter()
            .map(|(slot, package)| PackageRequest {
                slot: slot.clone(),
                selector: Selector::Fixed(package.clone()),
            })
            .collect(),
        descriptors: packages
            .keys()
            .map(|slot| Descriptors {
                slot: slot.clone(),
                keys: routes.get(slot).cloned().unwrap_or_default(),
            })
            .collect(),
    };

    // The one call that leaves this process while a generation is prepared: a run that hangs
    // before it reaches the Build Server and one that hangs inside it look the same from the
    // daemon, so both sides state it, naming every slot's package and key count.
    let asking: Vec<String> = packages
        .iter()
        .map(|(slot, package)| {
            let keys = routes.get(slot).map_or(0, Vec::len);
            format!(
                "{}/{}@{} keys={keys}",
                slot.op, package.backend, package.revision
            )
        })
        .collect();
    tracing::info!(
        packages = ?asking,
        %target,
        "asking the Build Server to prepare this generation's modules"
    );
    let prepared = client
        .prepare(&request, &mut |progress| {
            tracing::debug!(
                path = %progress.path,
                state = ?progress.state,
                "building a module for the new generation"
            );
        })
        .context("building the new generation")?;

    let mut entries = Vec::new();
    for (slot, package) in &prepared {
        let operator = proto::module::operator::named(&slot.op)
            .with_context(|| format!("operator {} has no implementation in this build", slot.op))?;

        // A slot's package is resolved for the keys selected for that slot alone.
        for key in routes.get(slot).into_iter().flatten() {
            let module = client
                .resolve(&slot.op, package, &target, key, &mut |progress| {
                    tracing::debug!(
                        path = %progress.path,
                        state = ?progress.state,
                        "resolving a module of the new generation"
                    );
                })
                .with_context(|| format!("resolving {} for {key}", slot.op))?;
            interface_agrees(operator, package, module.interface_version)
                .map_err(anyhow::Error::msg)?;
            let module =
                Resolved::of(module).with_context(|| format!("reading {} for {key}", slot.op))?;

            let entry = ops::install(&device, operator, key, module.as_ops())
                .with_context(|| format!("installing {operator} for {key}"))?;
            if let Some(entry) = entry {
                entries.push(entry);
            }
        }
    }

    tracing::info!(
        modules = entries.len(),
        %target,
        "this device's modules are installed for the new generation"
    );
    Ok(entries)
}

/// The scheduling policy of `package` (`specs/scheduler-design.md` §5.3): its module, which the
/// Build Server resolves for the store's wasm target under the operator's single descriptor and
/// builds first when its store does not hold it, loaded by `scheduler::policy::load` (fuel
/// metered), and the package's arguments as the policy's
/// [`Parameters`]. The module is loaded once here, and every scheduler and the routing policy
/// instantiate it.
///
/// # Errors
///
/// The Build Server's refusal, a module of another version of the policy interface or with no
/// wasm, bytes the engine does not load, or an argument that is not a policy parameter or does
/// not have its type.
pub(crate) fn policy(
    client: &mut Client,
    package: &PackageRevision,
) -> Result<(Arc<wasmtime::Module>, Parameters)> {
    let parameters = parameters(package)?;
    let target = Target::wasm();
    let named = format!("the {}/{} policy", package.backend, package.revision);

    let started = std::time::Instant::now();
    tracing::info!(
        backend = %package.backend,
        revision = package.revision,
        %target,
        "resolving the scheduling policy"
    );
    let module = client
        .resolve(
            proto::module::POLICY,
            package,
            &target,
            build_server::manifest::SINGLE_KEY,
            &mut |progress| {
                tracing::info!(
                    path = %progress.path,
                    state = ?progress.state,
                    "the Build Server is building the policy's module"
                );
            },
        )
        .with_context(|| format!("resolving {named} for {target}"))?;
    tracing::info!(
        backend = %package.backend,
        revision = package.revision,
        ms = started.elapsed().as_millis() as u64,
        "the scheduling policy resolved"
    );

    // A module built for another version of the policy interface would be asked decisions
    // through calls whose messages mean something else to it.
    if module.interface_version != scheduling::INTERFACE_VERSION {
        bail!(
            "{named} implements version {} of the scheduling policy interface and this engine \
             instantiates version {}",
            module.interface_version,
            scheduling::INTERFACE_VERSION
        );
    }
    let file = module
        .files
        .into_iter()
        .find(|file| file.name == build_server::source::MODULE_WASM)
        .with_context(|| {
            format!(
                "{named}'s module carries no {}",
                build_server::source::MODULE_WASM
            )
        })?;
    let bytes = read(&file).with_context(|| format!("reading {named}'s module"))?;

    let module = crate::scheduler::policy::load(&bytes)
        .map_err(|error| anyhow::anyhow!("loading {named}'s module: {error}"))?;
    Ok((Arc::new(module), parameters))
}

/// The policy's [`Parameters`] a package's `arguments` state, each a TOML value of the
/// parameter's type under the parameter's name; an argument the parameters do not name is
/// refused, as a misspelt limit would otherwise be dropped unsaid.
///
/// # Errors
///
/// An argument that is not a policy parameter or does not have its type.
pub(crate) fn parameters(package: &PackageRevision) -> Result<Parameters> {
    let named = serde_json::to_value(Parameters::default())?;
    let named = named
        .as_object()
        .context("the policy's parameters are a table")?;
    if let Some(unknown) = package
        .arguments
        .keys()
        .find(|name| !named.contains_key(name.as_str()))
    {
        bail!(
            "the {} policy package states `{unknown}`, which no policy parameter is named",
            package.backend
        );
    }

    let stated = serde_json::to_value(&package.arguments)?;
    serde_json::from_value(stated).with_context(|| {
        format!(
            "the {}/{} policy package's arguments are not its parameters",
            package.backend, package.revision
        )
    })
}

/// One module the Build Server resolved, in the form `ops::install` takes, each part the file the
/// Build Server's module contract names for the module's form: a cubin module's image
/// ([`build_server::source::MODULE_CUBIN`]) with the launch metadata the module names, whose
/// kernel is the symbol it launches; a native module's library
/// ([`build_server::source::MODULE_SO`]) with the launch entry the package declared. The Build
/// Server checks both at build: a cubin module carries its image and metadata naming a kernel,
/// and a native library defines every entry its package declares.
///
/// A native library is loaded from the store's own file ([`build_server::ModuleFile::path`]),
/// which the server and this process share on their host and which names the module's bytes for
/// as long as the store holds them. The object finds the files beside it (a `paged_attention`
/// module's `cubins/`) from the path it was loaded from (`dladdr` of its own entry), so the path
/// is the module directory's own file and never a copy or a descriptor's.
enum Resolved {
    Cubin {
        image: Vec<u8>,
        metadata: Vec<u8>,
        entry: String,
    },
    Native {
        library: PathBuf,
        entry: String,
    },
}

impl Resolved {
    /// Reads `module` into the form `ops::install` takes.
    ///
    /// # Errors
    ///
    /// A form this process does not install as a kernel (a scheduling policy's wasm, an
    /// executable), a module that lacks the file or the entry its form names, a file that cannot
    /// be read, or launch metadata that names no kernel.
    fn of(module: build_server::Module) -> Result<Self> {
        let named = |name: &str| {
            module
                .files
                .iter()
                .find(|file| file.name == name)
                .with_context(|| format!("the {:?} module carries no {name}", module.form))
        };

        match module.form {
            build_server::manifest::ModuleForm::Cubin => {
                let metadata_name = module
                    .metadata
                    .as_deref()
                    .context("the cubin module names no launch metadata")?;
                let image = read(named(build_server::source::MODULE_CUBIN)?)?;
                let metadata = read(named(metadata_name)?)?;
                let entry = ops::CubinMetadata::parse(&metadata)
                    .with_context(|| format!("reading the cubin's metadata {metadata_name}"))?
                    .name;
                Ok(Self::Cubin {
                    image,
                    metadata,
                    entry,
                })
            }
            build_server::manifest::ModuleForm::Native => {
                let entry = module
                    .entries
                    .get(build_server::manifest::LAUNCH_ENTRY)
                    .context("the native module states no launch entry")?
                    .clone();
                let library = named(build_server::source::MODULE_SO)?.path.clone();
                Ok(Self::Native { library, entry })
            }
            other => bail!("a {other:?} module is not a kernel this process installs"),
        }
    }

    /// The module as `ops::install` takes it.
    fn as_ops(&self) -> ops::Module<'_> {
        match self {
            Self::Cubin {
                image,
                metadata,
                entry,
            } => ops::Module::Cubin {
                image,
                metadata: Some(metadata),
                entry,
            },
            Self::Native { library, entry } => ops::Module::Native {
                path: library,
                entry,
            },
        }
    }
}

/// A module file's bytes, read through the handle the Build Server sent.
fn read(file: &build_server::ModuleFile) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(file.size as usize);
    (&file.file)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", file.name))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A policy package's arguments are its parameters by name, each a TOML value of the
    /// parameter's type, and a name no parameter has is refused rather than dropped.
    #[test]
    fn a_policy_package_s_arguments_are_its_parameters() {
        let package = |arguments: &str| PackageRevision {
            backend: "fcfs".to_owned(),
            revision: 3,
            arguments: toml::from_str(arguments).unwrap(),
        };

        let stated = parameters(&package(
            "max_batch = 8\ntimeout = 1000\nexpected_speedup = 1.5",
        ))
        .unwrap();
        assert_eq!(stated.max_batch, Some(8));
        assert_eq!(stated.timeout, Some(scheduling::time::Nanos(1000)));
        assert_eq!(stated.expected_speedup, Some(1.5));
        assert_eq!(stated.chunked_prefill_size, None);

        let unknown = parameters(&package("max_batches = 8")).unwrap_err();
        assert!(
            format!("{unknown:#}").contains("max_batches"),
            "{unknown:#}"
        );
        assert!(parameters(&package("max_batch = \"eight\"")).is_err());
    }

    /// A native module's library is the file the module contract names, `module.so`, loaded
    /// from the store's own path whatever the order of its files, so the files beside it are in
    /// the directory the object looks in; a cubin module's image is `module.cubin` and its kernel
    /// the one its metadata names, whatever other file the module holds.
    #[test]
    fn a_module_s_files_are_read_by_the_roles_its_form_names() {
        let store = tempfile::tempdir().unwrap();
        let file = |name: &str, bytes: &[u8]| stored(store.path(), name, bytes);

        // `cubins/` sorts before `module.so`, as in a `paged_attention` module.
        let native = build_server::Module {
            form: build_server::manifest::ModuleForm::Native,
            files: vec![
                file("cubins/kernel.cubin", b"cubin"),
                file(build_server::source::MODULE_SO, b"library"),
            ],
            metadata: None,
            entries: BTreeMap::from([(
                build_server::manifest::LAUNCH_ENTRY.to_owned(),
                "launch_attention".to_owned(),
            )]),
            interface_version: 1,
            executable: None,
        };
        let Resolved::Native { library, entry } = Resolved::of(native).unwrap() else {
            panic!("a native module reads as a native library");
        };
        assert_eq!(library, store.path().join(build_server::source::MODULE_SO));
        assert_eq!(entry, "launch_attention");
        assert!(
            library
                .with_file_name("cubins")
                .join("kernel.cubin")
                .is_file(),
            "the module's sibling is not beside the path the library is loaded from"
        );

        let cubin = build_server::Module {
            form: build_server::manifest::ModuleForm::Cubin,
            files: vec![
                file("launch.json", br#"{"name":"rms_norm_kernel"}"#),
                file(build_server::source::MODULE_CUBIN, b"image"),
                file("tuning.json", b"{}"),
            ],
            metadata: Some("launch.json".to_owned()),
            entries: BTreeMap::new(),
            interface_version: 1,
            executable: None,
        };
        let Resolved::Cubin { image, entry, .. } = Resolved::of(cubin).unwrap() else {
            panic!("a cubin module reads as a cubin");
        };
        assert_eq!(image, b"image");
        assert_eq!(entry, "rms_norm_kernel");
    }

    /// A module that lacks the file its form names is refused by that name.
    #[test]
    fn a_module_without_the_file_its_form_names_is_refused() {
        let store = tempfile::tempdir().unwrap();
        let module = build_server::Module {
            form: build_server::manifest::ModuleForm::Native,
            files: vec![stored(store.path(), "other.so", b"library")],
            metadata: None,
            entries: BTreeMap::from([(
                build_server::manifest::LAUNCH_ENTRY.to_owned(),
                "launch".to_owned(),
            )]),
            interface_version: 1,
            executable: None,
        };

        let Err(refused) = Resolved::of(module) else {
            panic!("a native module without module.so was read");
        };
        assert!(
            format!("{refused:#}").contains(build_server::source::MODULE_SO),
            "{refused:#}"
        );
    }

    /// One file of a module under test, written into `store` and opened as the store's handles
    /// are.
    fn stored(store: &std::path::Path, name: &str, bytes: &[u8]) -> build_server::ModuleFile {
        let path = store.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        build_server::ModuleFile {
            name: name.to_owned(),
            size: bytes.len() as u64,
            file: std::fs::File::open(&path).unwrap(),
            path,
        }
    }
}
