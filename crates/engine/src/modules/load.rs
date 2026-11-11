//! The module-load lane's work: the Build Server's selection of a generation's packages and one
//! device's installs, and the loading of each resolved module into the device's context as a
//! kernel, a cubin or a native library chosen by the entry it defines.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use build_server::{Client, CoverRequest, PrepareRequest, Target};
use ops::KernelEntry;
use proto::module::Packages;
use worker::completion::Outcome;
use worker::host::HostAction;

use crate::modules::select::{
    Routes, fixed_revisions, interface_agrees, named_backends, selected_slots,
};

/// The Build Server's selection of a generation's packages, on the module-load lane.
///
/// Every operator the generation launches is asked once, with every descriptor key it launches and
/// the backends the deployment names for it; the server answers the package of each key
/// (`BuildServer::cover`), and the answer is the generation's slots and the keys each slot's
/// package builds. The call has no deadline of its own, so it runs where a build runs: on the lane,
/// a store that never answers costs a build slot and not the control thread that answers the
/// daemon, the requests and the watchdog.
pub(crate) struct Select {
    pub(crate) packages: Arc<Mutex<Client>>,
    pub(crate) targets: Vec<Target>,
    /// The packages the deployment names, by slot.
    pub(crate) named: Packages,
    /// Every descriptor key the generation launches, by operator.
    pub(crate) keys: BTreeMap<String, BTreeSet<String>>,
}

/// What a selection answered: the generation's packages by slot, and the keys of each slot.
#[derive(Clone, Debug, Default)]
pub(crate) struct Selection {
    pub(crate) packages: Packages,
    pub(crate) routes: Routes,
}

impl HostAction for Select {
    type Output = Selection;

    fn run(&self) -> worker::Result<Self::Output> {
        let mut client = self
            .packages
            .lock()
            .map_err(|_| worker::Error::State("the Build Server client was poisoned"))?;
        let mut selection = Selection::default();
        for (op, keys) in &self.keys {
            if keys.is_empty() {
                continue;
            }
            let request = CoverRequest {
                op: op.clone(),
                targets: self.targets.clone(),
                keys: keys.iter().cloned().collect(),
                named: named_backends(op, &self.named),
                fixed: fixed_revisions(op, &self.named),
            };
            let coverage = client.cover(&request).map_err(|error| {
                worker::Error::Invalid(format!(
                    "asking the Build Server to select {op}'s packages: {error}"
                ))
            })?;
            let (slots, routes) =
                selected_slots(op, &coverage, &self.named).map_err(worker::Error::Invalid)?;
            for (slot, package) in &slots {
                tracing::info!(
                    %op,
                    backend = %package.backend,
                    revision = package.revision,
                    part = slot.part.as_deref().unwrap_or(""),
                    keys = routes.get(slot).map_or(0, Vec::len),
                    named = !request.named.is_empty(),
                    "the Build Server selected a package for this operator's keys"
                );
            }
            selection.packages.extend(slots);
            selection.routes.extend(routes);
        }
        Ok(selection)
    }

    /// Nothing to give back: the client this action borrows belongs to the process, and its reply
    /// holds no device or pinned memory.
    fn release(&self) -> worker::Result<()> {
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        Outcome::Success(())
    }
}

/// One piece of the module-load lane's work.
///
/// The lane has two tenants and one order: the selection, which says which package builds each
/// key, and the installs that follow from it. Keeping them in one lane keeps that order where it
/// belongs and keeps both off the control thread.
pub(crate) enum ModuleWork {
    Select(Select),
    Install(Install),
}

impl HostAction for ModuleWork {
    type Output = ModuleAnswer;

    fn run(&self) -> worker::Result<Self::Output> {
        match self {
            Self::Select(select) => select.run().map(ModuleAnswer::Selected),
            Self::Install(install) => install.run().map(ModuleAnswer::Installed),
        }
    }

    fn release(&self) -> worker::Result<()> {
        match self {
            Self::Select(select) => select.release(),
            Self::Install(install) => install.release(),
        }
    }

    fn input_outcome(&self) -> Outcome {
        match self {
            Self::Select(select) => select.input_outcome(),
            Self::Install(install) => install.input_outcome(),
        }
    }
}

/// What one piece of the module lane's work answered.
pub(crate) enum ModuleAnswer {
    /// The generation's packages, as the Build Server selected them.
    Selected(Selection),
    /// One device's kernels for the generation being prepared.
    Installed(Vec<KernelEntry>),
}

/// One device's half of the precopy, on the module-load lane: the Build
/// Server compiles and caches what the generation's descriptors need, each
/// module is resolved for this node's target, and each is loaded into this
/// device's context as a kernel.
///
/// It runs on the blocking pool because compiling can take minutes and no
/// device stream and no scheduler thread may wait on it.
pub(crate) struct Install {
    pub(crate) packages: Arc<Mutex<Client>>,
    pub(crate) device: platform::Device,
    pub(crate) target: Target,
    /// The generation's one admission: every slot's package with exactly the keys selected for
    /// it, so the Build Server prepares the whole generation at once and no package is asked
    /// for another slot's keys.
    pub(crate) request: PrepareRequest,
}

/// One module the Build Server resolved, held open while its kernel loads.
///
/// The server sends the module's files as open handles rather than as paths, so the image is read
/// through the handle: a cubin's bytes are small and read once, and a native library is loaded
/// through the handle's `/proc/self/fd` path, which names the same inode the server sent. Every
/// handle stays open until the kernel has loaded, so neither read reaches a file that has gone.
struct Resolved {
    image: Option<Vec<u8>>,
    metadata: Option<Vec<u8>>,
    native: Option<NativeModule>,
    entry: String,
}

/// The path `dlopen` loads a native module from.
///
/// `dlopen` takes a path, and the handle the Build Server sends has none of its own. The file is
/// used where it still has a name - glibc keys a loaded object by the name it was opened with, so a
/// name this process derives from a descriptor would answer for the next module to land on that
/// number, and one library's handle would serve another. When the store's file has no name left, the
/// bytes are copied to one this process owns and that copy goes when the module does: `dlopen` maps
/// the file, so the mapping outlives the name.
struct NativeModule {
    path: std::path::PathBuf,
    /// The copy, when the handle the server sent named no file of its own.
    copy: Option<std::path::PathBuf>,
}

impl NativeModule {
    /// Whether this module was loaded from a copy rather than from the store's own file.
    fn copied(&self) -> bool {
        self.copy.is_some()
    }
}

impl Drop for NativeModule {
    fn drop(&mut self) {
        let Some(copy) = &self.copy else {
            return;
        };
        if let Err(error) = std::fs::remove_file(copy) {
            tracing::warn!(
                path = %copy.display(),
                %error,
                "removing this process's copy of a native module failed"
            );
        }
    }
}

/// The file a native module's library is, and the path it is loaded from.
///
/// **Never by position.** A `paged_attention` module holds `module.so` beside `cubins/`, and `cubins`
/// sorts before `module.so`, so taking the first file handed `dlopen` a cubin and the answer was
/// `cannot open shared object file`. The library is the file that defines the entry the package
/// declared, which is the property `dlopen` is for; a module whose files define it in more than one
/// place, or in none, is refused by name with its files listed.
///
/// A module with siblings is never loaded from a copy, and never from a descriptor number. Its entry
/// receives no path - `ops::Entry` takes the call structure and the stream - so the object finds its
/// own `cubins/` from the file it was loaded from: `paged_attention`'s module does it with `dladdr`
/// of its own entry and takes the directory of `dli_fname`
/// (`crates/ops/kernels/paged_attention/trtllm/src/entry.cpp`, `module_directory`). `dli_fname` is the
/// path handed to `dlopen`, so that path must be the module's own file: `/proc/self/fd/<n>` gives the
/// directory `/proc/self/fd`, and a copy gives the copy's directory. Both are refusals here instead
/// of a load that dies at the first cubin.
fn native_library(
    files: Vec<build_server::ModuleFile>,
    entry: &str,
) -> worker::Result<NativeModule> {
    let invalid = |reason: String| worker::Error::Invalid(reason);
    let siblings = files.len() > 1;
    let chosen = match files.len() {
        0 => return Err(invalid("a native module carries no library".to_owned())),
        // One file: the module's library is that file, and there is no order to get wrong.
        1 => 0,
        _ => {
            let defining: Vec<usize> = files
                .iter()
                .enumerate()
                .filter(|(_, file)| defines_entry(file, entry))
                .map(|(index, _)| index)
                .collect();
            let names = |chosen: &[usize]| {
                chosen
                    .iter()
                    .filter_map(|index| files.get(*index))
                    .map(|file| file.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            match defining.as_slice() {
                [only] => *only,
                [] => {
                    return Err(invalid(format!(
                        "no file of this native module defines the entry the package declared \
                         ({entry}), so nothing in it is that module's library: {}",
                        names(&(0..files.len()).collect::<Vec<_>>())
                    )));
                }
                several => {
                    return Err(invalid(format!(
                        "several files of this native module define the entry the package declared \
                         ({entry}), so its library is ambiguous: {}",
                        names(several)
                    )));
                }
            }
        }
    };
    let file = files
        .into_iter()
        .nth(chosen)
        .ok_or_else(|| invalid("the chosen library file is missing".to_owned()))?;
    if siblings {
        // The question a reader of a failed load has: where does this object look for the files
        // beside it? A module that resolves them itself uses the path it was loaded from - `dladdr`
        // of its own entry - so that directory is the answer, and `origin` says whether it also
        // states `$ORIGIN` in a run path, which is the other way a sibling can be found.
        let module = match std::fs::read_link(format!("/proc/self/fd/{}", as_raw_fd(&file))) {
            Ok(target) => target,
            Err(_) => std::path::PathBuf::new(),
        };
        tracing::info!(
            library = %file.name,
            loaded_from = %module.display(),
            directory = %module.parent().map(|path| path.display().to_string()).unwrap_or_default(),
            origin = ?origin_in_runpath(&file),
            "this native module's library sits beside other files of the module"
        );
    }
    native_module(&file, !siblings)
}

/// Whether a file defines `entry` among the functions it exports.
fn defines_entry(file: &build_server::ModuleFile, entry: &str) -> bool {
    bytes_of(file)
        .and_then(|bytes| platform::NativeLibrary::defined_functions(&bytes))
        .is_some_and(|names| names.iter().any(|name| name == entry))
}

/// The file's bytes, read from the start: the handle the server sent may have been read already.
fn bytes_of(file: &build_server::ModuleFile) -> Option<Vec<u8>> {
    use std::io::{Read as _, Seek as _};
    let mut handle = &file.file;
    handle.seek(std::io::SeekFrom::Start(0)).ok()?;
    let mut bytes = Vec::with_capacity(file.size as usize);
    handle.read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

/// Whether the object states `$ORIGIN` in its run path: the one way it can find a sibling file by
/// its own location. `None` when the object states no run path, or cannot be read.
fn origin_in_runpath(file: &build_server::ModuleFile) -> Option<bool> {
    let bytes = bytes_of(file)?;
    let path = platform::NativeLibrary::run_path(&bytes)?;
    Some(path.contains("$ORIGIN"))
}

/// The descriptor behind a module's file, for the line that says where the object will look for its
/// siblings.
fn as_raw_fd(file: &build_server::ModuleFile) -> std::os::fd::RawFd {
    use std::os::fd::AsRawFd as _;
    file.file.as_raw_fd()
}

/// The path a native module is loaded from: the store's own file when it still has one, and a copy
/// this process owns when the handle the server sent has outlived its name.
fn native_module(
    file: &build_server::ModuleFile,
    copy_allowed: bool,
) -> worker::Result<NativeModule> {
    use std::io::Read as _;
    use std::os::fd::AsRawFd as _;

    let invalid = |reason: String| worker::Error::Invalid(reason);
    let link = format!("/proc/self/fd/{}", file.file.as_raw_fd());
    if let Ok(target) = std::fs::read_link(&link) {
        let named = !target.to_string_lossy().ends_with(" (deleted)");
        if named && target.is_file() {
            return Ok(NativeModule {
                path: target,
                copy: None,
            });
        }
    }

    // The handle names nothing this process can open: the server unlinked the file after sending
    // it, or the descriptor addresses something that is not a file. One copy, under a name no
    // other module shares - and only for a module that stands alone, because a copy is a different
    // location and every path the module resolves from its own location would break.
    if !copy_allowed {
        return Err(invalid(format!(
            "the library of this native module ({}) has no name this process can open, and the \
             module holds files beside it: loading it from a copy would lose them, because the \
             entry receives no path and the object finds them through its own location",
            file.name
        )));
    }

    static COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let ordinal = COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!("uniserve-native-{}", std::process::id()));
    std::fs::create_dir_all(&directory)
        .map_err(|error| invalid(format!("preparing {}: {error}", directory.display())))?;
    let name: String = file
        .name
        .chars()
        .map(|held| {
            if held.is_ascii_alphanumeric() || held == '.' || held == '_' || held == '-' {
                held
            } else {
                '_'
            }
        })
        .collect();
    let path = directory.join(format!("{ordinal}-{name}"));
    let mut bytes = Vec::with_capacity(file.size as usize);
    (&file.file)
        .read_to_end(&mut bytes)
        .map_err(|error| invalid(format!("reading {}: {error}", file.name)))?;
    std::fs::write(&path, &bytes)
        .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    tracing::warn!(
        module = %file.name,
        path = %path.display(),
        "the store's file has no name this process can open, so the module is loaded from a copy \
         it owns"
    );
    Ok(NativeModule {
        copy: Some(path.clone()),
        path,
    })
}

impl Resolved {
    /// Whether this module was loaded from a copy rather than from the store's own file: the one
    /// fact that tells a store whose handles name their files from a store whose do not.
    fn copied(&self) -> bool {
        self.native.as_ref().is_some_and(NativeModule::copied)
    }

    /// The module as `ops::install` takes it.
    fn as_ops(&self) -> worker::Result<ops::Module<'_>> {
        match (&self.image, &self.native) {
            (Some(image), _) => Ok(ops::Module::Cubin {
                image,
                metadata: self.metadata.as_deref(),
                entry: &self.entry,
            }),
            (None, Some(native)) => Ok(ops::Module::Native {
                path: native.path.as_path(),
                entry: &self.entry,
            }),
            (None, None) => Err(worker::Error::Invalid(
                "a resolved module carries neither a cubin image nor a native library".to_owned(),
            )),
        }
    }
}

/// An error with every cause behind it, in one line.
///
/// A refusal that names only its own variant is the one thing a log cannot act on: `platform: ...`
/// with the driver's own message behind it says what failed, and the chain is where that message
/// lives. Each cause is taken from `source`, so nothing here has to know which errors there are.
fn causes<E: std::error::Error + std::fmt::Debug>(error: &E) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }

    // A failure whose own message is empty is the one a reader cannot act on, and an error type is
    // free to have one: the whole value says what it was, so it is what gets said.
    if text.trim().is_empty() {
        return format!("{error:?}");
    }
    text
}

/// States what an object defines when the entry a package declared is not among its symbols.
///
/// `dlsym` answers yes or no and nothing else, so a `NativeSymbol` failure names the entry that was
/// looked for and leaves "what does this object have" unanswered - the fact that turns the failure
/// into a fix. The line never changes the failure: the install still returns exactly what it did.
fn report_missing_entry(error: &ops::Error) {
    let ops::Error::Platform(platform::Error::NativeSymbol { path, symbol, .. }) = error else {
        return;
    };
    match defined_entries(path) {
        Some((total, names)) => tracing::warn!(
            symbol = %symbol,
            path = %path.display(),
            defined = total,
            entries = %names.join(", "),
            "the object does not define the entry the package declared: these are the entries it \
             does define"
        ),
        None => tracing::warn!(
            symbol = %symbol,
            path = %path.display(),
            "the object does not define the entry the package declared, and this process could \
             not read the object's own entry table"
        ),
    }
}

/// The functions the object at `path` defines for the dynamic linker: how many, and the first few.
/// `None` when the file cannot be read or is not an object `platform`'s reader understands.
fn defined_entries(path: &std::path::Path) -> Option<(usize, Vec<String>)> {
    let bytes = std::fs::read(path).ok()?;
    let names = platform::NativeLibrary::defined_functions(&bytes)?;
    let first = names.iter().take(8).cloned().collect();
    Some((names.len(), first))
}

/// Reads a resolved module into the form `ops::install` takes, or refuses a form this build cannot
/// install: a scheduling policy's wasm package and a package the main process runs as its
/// own process are not kernels.
fn resolved(module: build_server::Module) -> worker::Result<Resolved> {
    use std::io::Read as _;

    let invalid = |reason: String| worker::Error::Invalid(reason);
    let mut files = module.files.into_iter();
    match module.form {
        build_server::manifest::ModuleForm::Cubin => {
            let mut image = None;
            let mut metadata = None;
            for file in &mut files {
                let mut bytes = Vec::with_capacity(file.size as usize);
                let mut handle = &file.file;
                handle
                    .read_to_end(&mut bytes)
                    .map_err(|error| invalid(format!("reading {}: {error}", file.name)))?;
                if module.metadata.as_deref() == Some(file.name.as_str()) {
                    metadata = Some(bytes);
                } else {
                    image = Some(bytes);
                }
            }
            let Some(image) = image else {
                return Err(invalid("a cubin module carries no image".to_owned()));
            };
            // The image's kernel symbol is its metadata's name; without metadata the package names
            // the entry itself, and `ops` falls back to the module's single entry.
            let entry = match &metadata {
                Some(metadata) => {
                    ops::CubinMetadata::parse(metadata)
                        .map_err(|error| invalid(format!("the cubin's metadata: {error}")))?
                        .name
                }
                None => String::new(),
            };
            Ok(Resolved {
                image: Some(image),
                metadata,
                native: None,
                entry,
            })
        }
        build_server::manifest::ModuleForm::Native => {
            let Some(entry) = module.entries.get(build_server::manifest::LAUNCH_ENTRY) else {
                return Err(invalid("a native module states no launch entry".to_owned()));
            };
            Ok(Resolved {
                image: None,
                metadata: None,
                native: Some(native_library(files.collect(), entry)?),
                entry: entry.clone(),
            })
        }
        other => Err(invalid(format!(
            "a {other:?} module is not a kernel this process installs"
        ))),
    }
}

impl HostAction for Install {
    type Output = Vec<KernelEntry>;

    /// This device's half of the precopy: the generation's packages prepared in one request, each
    /// slot's keys resolved for this node's target, and each module loaded into this device's
    /// context as a kernel. A generation that changes no kernel package asks nothing.
    fn run(&self) -> worker::Result<Self::Output> {
        let request = &self.request;
        if request.packages.is_empty() {
            return Ok(Vec::new());
        }
        let mut client = self
            .packages
            .lock()
            .map_err(|_| worker::Error::State("the Build Server client was poisoned"))?;
        // The one call that leaves this process while a generation is prepared: a run that hangs
        // before it reaches the Build Server and one that hangs inside it look the same from the
        // daemon, so both sides state it, naming every slot's package and key count.
        let asking: Vec<String> = request
            .descriptors
            .iter()
            .map(|descriptors| {
                let package = request
                    .packages
                    .iter()
                    .find(|entry| entry.slot == descriptors.slot)
                    .and_then(|entry| match &entry.selector {
                        build_server::Selector::Fixed(package) => {
                            Some(format!("{}@{}", package.backend, package.revision))
                        }
                        build_server::Selector::Latest { .. } => None,
                    })
                    .unwrap_or_default();
                format!(
                    "{}/{package} keys={}",
                    descriptors.slot.op,
                    descriptors.keys.len()
                )
            })
            .collect();
        tracing::info!(
            packages = request.packages.len(),
            packages = ?asking,
            target = %self.target,
            "asking the Build Server to prepare this generation's modules"
        );
        let prepared = client
            .prepare(request, &mut |progress| {
                tracing::debug!(
                    path = %progress.path,
                    state = ?progress.state,
                    "building a module for the new generation"
                );
            })
            .map_err(|error| {
                worker::Error::Invalid(format!("building the new generation: {}", causes(&error)))
            })?;

        let mut entries = Vec::new();
        let mut copies = 0_usize;
        for chosen in &prepared.packages {
            let operator = proto::module::operator::named(&chosen.slot.op).ok_or_else(|| {
                worker::Error::Invalid(format!(
                    "operator {} has no implementation in this build",
                    chosen.slot.op
                ))
            })?;
            // A slot's package is resolved for the keys selected for that slot alone.
            let keys = request
                .descriptors
                .iter()
                .filter(|descriptor| descriptor.slot == chosen.slot)
                .flat_map(|descriptor| descriptor.keys.iter());
            for key in keys {
                let module = client
                    .resolve(
                        &chosen.slot.op,
                        &chosen.package,
                        &self.target,
                        key,
                        &mut |progress| {
                            tracing::debug!(
                                path = %progress.path,
                                state = ?progress.state,
                                "resolving a module of the new generation"
                            );
                        },
                    )
                    .map_err(|error| {
                        tracing::warn!(
                            op = %chosen.slot.op,
                            key = %key,
                            error = ?error,
                            "resolving this generation's module failed"
                        );
                        worker::Error::Invalid(format!(
                            "resolving {} for {key}: {}",
                            chosen.slot.op,
                            causes(&error)
                        ))
                    })?;
                interface_agrees(operator, &chosen.package, module.interface_version)
                    .map_err(worker::Error::Invalid)?;
                let module = resolved(module)?;
                if module.copied() {
                    copies += 1;
                }
                let entry = match ops::install(&self.device, operator, key, module.as_ops()?) {
                    Ok(entry) => entry,
                    Err(error) => {
                        // The refusal that travels is one line; this is the whole value, so a
                        // failure whose own message is empty still shows which call, status and
                        // code it carried.
                        tracing::warn!(
                            operator,
                            key = %key,
                            error = ?error,
                            "installing this generation's kernel failed"
                        );
                        report_missing_entry(&error);
                        return Err(worker::Error::Invalid(format!(
                            "installing {operator} for {key}: {}",
                            causes(&error)
                        )));
                    }
                };
                if let Some(entry) = entry {
                    entries.push(entry);
                }
            }
        }
        tracing::info!(
            modules = entries.len(),
            copies,
            target = %self.target,
            "this device's modules are installed for the new generation"
        );
        Ok(entries)
    }

    fn release(&self) -> worker::Result<()> {
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        Outcome::Success(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The library of a multi-file native module is chosen by what it defines, never by position.
    ///
    /// A `paged_attention` module holds `module.so` beside `cubins/`, and `cubins/` sorts first, so
    /// the first file is not the library. Here two real objects
    /// stand in for those files with the library named so it sorts **second**, and the entry declared
    /// is one only it defines.
    ///
    /// The directory is the module, and the mechanism is the package's own: `paged_attention`'s entry
    /// takes no path and finds `cubins/` from `dladdr` of its own symbol, whose `dli_fname` is the
    /// path this process handed `dlopen`. So the path must be the module's own file - which is what
    /// the assertions below hold: the library is that file, it is not a copy, and the sibling sits in
    /// the directory the object will look in. A `/proc/self/fd/<n>` path would give it
    /// `/proc/self/fd`, and a copy would give it the copy's directory.
    #[test]
    fn a_native_library_is_the_file_that_defines_the_entry_not_the_first_one() {
        let Some(libc) = loaded_library("libc.so.6") else {
            return;
        };
        let Some(libm) = loaded_library("libm.so.6") else {
            return;
        };
        let directory =
            std::env::temp_dir().join(format!("uniserve-native-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("a temporary directory for this test");
        let sibling = directory.join("aa-sibling.so");
        let library = directory.join("zz-module.so");
        std::fs::copy(&libm, &sibling).expect("the sibling object is copied");
        std::fs::copy(&libc, &library).expect("the library object is copied");

        let module = native_library(
            vec![
                opened_file("aa-sibling.so", &sibling),
                opened_file("zz-module.so", &library),
            ],
            "malloc",
        )
        .expect("the file that defines the entry is the module's library");
        assert_eq!(
            module.path, library,
            "the library was resolved by position in the module, not by the entry it defines"
        );
        assert!(
            !module.copied(),
            "the library has a name of its own, so no copy is made"
        );
        assert!(
            module.path.with_file_name("aa-sibling.so").is_file(),
            "the module's sibling is not reachable from the location the library is loaded from"
        );

        // The same module with its library unlinked after the handle was taken: a copy would carry
        // no sibling, so the module is refused by name rather than loaded from one.
        let nameless = directory.join("nameless.so");
        std::fs::copy(&libc, &nameless)
            .expect("the library object is copied under a temporary name");
        let handle = std::fs::File::open(&nameless).expect("the copy opens");
        std::fs::remove_file(&nameless).expect("the copy is unlinked, as a store's handle may be");

        let refused = match native_library(
            vec![
                opened_file("aa-sibling.so", &sibling),
                build_server::ModuleFile {
                    name: "nameless.so".to_owned(),
                    size: std::fs::metadata(&sibling)
                        .map(|meta| meta.len())
                        .unwrap_or_default(),
                    file: handle,
                    path: nameless.clone(),
                },
            ],
            "malloc",
        ) {
            Ok(_) => panic!("a module with siblings was loaded from a copy"),
            Err(error) => format!("{error}"),
        };
        assert!(
            refused.contains("nameless.so") && refused.contains("copy"),
            "the refusal names neither the file nor the copy: {refused}"
        );

        let _ = std::fs::remove_dir_all(&directory);
    }

    /// The object a pattern names, as the process's own maps record it: a real ELF on this host.
    fn loaded_library(pattern: &str) -> Option<std::path::PathBuf> {
        let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
        maps.lines()
            .filter_map(|line| line.split_whitespace().last())
            .find(|path| path.contains(pattern) && path.starts_with('/'))
            .map(std::path::PathBuf::from)
    }

    /// One file of a module under test, opened the way the store's handles are.
    fn opened_file(name: &str, path: &std::path::Path) -> build_server::ModuleFile {
        build_server::ModuleFile {
            name: name.to_owned(),
            size: std::fs::metadata(path)
                .map(|meta| meta.len())
                .unwrap_or_default(),
            file: std::fs::File::open(path).expect("the test's object opens"),
            path: path.to_path_buf(),
        }
    }
}
