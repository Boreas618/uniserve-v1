//! Store coverage for a key fixture: can this host install a module for every key the fixture lists,
//! and what must each module carry?
//!
//! A key fixture (such as the one `tools/build_qwen3_fixture.sh` writes) names one descriptor key per line as
//! `<operator>\t<descriptor key>`. For the rewrite to launch that module, three facts must hold, and
//! this program reports all three per key:
//!
//! 1. **A package covers the descriptor.** The package's manifest declares dimension axes with the
//!    values it supports (`[kernel.descriptors]`), or names keys outright; `Manifest::covers` decides.
//! 2. **The package builds for the target.** Its declared architectures must include the target arch,
//!    so a key can be covered by a package that cannot be built for this host (design section 5.3).
//! 3. **The module's metadata is named.** A cubin module carries entry symbol, warp count and dynamic
//!    shared bytes in a metadata JSON written by the build; the manifest names that file, so a caller
//!    can read it from the installed module directory. A native module needs no metadata.
//!
//! The check is over the **package set in the tree**, not over the store: whether the modules are
//! already built and installed is the epoch's business, and a coverage failure here is the one that
//! reads `generation 1 has no kernel for operator '…' descriptor '…'` at run time.
//!
//! ```text
//! cargo run -p program-server --example key_coverage -- \
//!     --fixture <keys file> --target sm_100a
//! ```
//!
//! The exit code is 1 when any key has no package that covers it for the target, so the example can
//! stand in a check without parsing its text.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use program_server::manifest::{Manifest, ModuleForm};
use program_server::source::PackageSource;

/// What a manifest knows about one fixture key.
struct Covering {
    backend: String,
    /// Every backend that covers this key for the target, in package order.
    ///
    /// More than one is a defect rather than a choice: two packages claiming one key means two hosts
    /// can build different modules under one name, and a store built here would publish whichever the
    /// directory walk offered first. `build_store.py` refuses such a fixture before it compiles; this
    /// example states it before a fixture is written, which is the cheaper end to catch it at.
    claimants: Vec<String>,
    form: ModuleForm,
    /// The metadata JSON's name in the module directory, for this descriptor.
    metadata: Option<String>,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut fixture = None;
    let mut declared_blocked = false;
    let mut target = "sm_100a".to_owned();
    let mut kernels = None;

    let mut argv = args.iter();
    while let Some(flag) = argv.next() {
        let value = || argv.clone().next().cloned();
        match flag.as_str() {
            "--fixture" => fixture = value(),
            // The fixture's declared-blocked block: the rows no package fills yet, in the one form
            // both the fixture and the store test read. One derivation, two consumers.
            "--declared-blocked" => declared_blocked = true,
            "--target" => target = value().unwrap_or(target),
            "--kernels" => kernels = value().map(PathBuf::from),
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
        argv.next();
    }
    let (Some(fixture), Some(kernels)) = (fixture, kernels) else {
        eprintln!(
            "usage: key_coverage --fixture <path> --kernels <package root> [--target sm_100a]"
        );
        return ExitCode::from(2);
    };

    let packages = match load_packages(&kernels) {
        Ok(packages) => packages,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let rows = match read_fixture(Path::new(&fixture)) {
        Ok(rows) => rows,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    // A fixture with no rows is a walk that failed to print, not a store that needs nothing: every
    // key of an empty set is trivially covered, so the check would pass over no evidence.
    if rows.is_empty() {
        eprintln!(
            "{fixture} names no rows: an empty fixture is a walk that printed nothing, and every \
             coverage question about it is vacuously true"
        );
        return ExitCode::FAILURE;
    }

    if !declared_blocked {
        println!(
            "# {} keys of {fixture} against {} packages, target {target}",
            rows.len(),
            packages.values().flatten().count()
        );
    }

    let mut uncovered = 0;
    let mut double_claimed: Vec<(&str, &str, Vec<String>)> = Vec::new();
    for (op, key) in &rows {
        match cover(&packages, op, key, &target) {
            Ok(covering) if declared_blocked => {
                let _ = covering;
            }
            Ok(covering) => {
                if covering.claimants.len() > 1 {
                    double_claimed.push((op, key, covering.claimants.clone()));
                }
                println!(
                    "covered\t{op}\t{}\t{}\t{}\t{key}",
                    covering.backend,
                    match covering.form {
                        ModuleForm::Cubin => "cubin",
                        ModuleForm::Native => "native",
                        ModuleForm::Wasm => "wasm",
                        ModuleForm::Executable => "executable",
                    },
                    covering.metadata.as_deref().unwrap_or("-"),
                )
            }
            Err(reason) => {
                uncovered += 1;
                if declared_blocked {
                    println!("# blocked:\t{op}\t{key}\t{reason}");
                } else {
                    println!("MISSING\t{op}\t{reason}\t{key}");
                }
            }
        }
    }

    // The reverse direction: a package that fills an operator nothing declares, and a declared
    // operator no package fills. The first is a program for a name no epoch record can carry; the
    // second can never be installed, so every key of it is uncovered for a deeper reason.
    let declared: BTreeSet<&str> = ops::install::operators().iter().copied().collect();
    let filled: BTreeSet<&str> = packages.keys().map(String::as_str).collect();

    // With `--declared-blocked` the output is the block itself: nothing else goes to stdout, so a
    // regeneration of the fixture is a paste and not a transcription.
    if declared_blocked {
        // The block is pasted into a fixture, so the count has to name the set it was counted
        // over: a number beside no fixture is the failure this lane keeps quoting.
        eprintln!("{uncovered} blocked row(s) of {} in {fixture}", rows.len());
        return ExitCode::SUCCESS;
    }

    let orphaned: Vec<&str> = filled.difference(&declared).copied().collect();
    if !orphaned.is_empty() {
        println!(
            "\n# packages whose operator is not declared: {}",
            orphaned.join(" ")
        );
    }
    let unfilled: Vec<&str> = declared.difference(&filled).copied().collect();
    if !unfilled.is_empty() {
        println!(
            "\n# declared operators with no package ({}): {}",
            unfilled.len(),
            unfilled.join(" ")
        );
    }

    if !double_claimed.is_empty() {
        println!(
            "\n# {} key(s) are covered by more than one package for {target}:",
            double_claimed.len()
        );
        for (op, key, claimants) in &double_claimed {
            println!("#   {op}\t{key}\tclaimed by {}", claimants.join(" "));
        }
    }

    if uncovered == 0 && double_claimed.is_empty() {
        println!("\n# every key is covered for {target}");
        // A key two packages claim is covered, and cannot be installed deterministically, so the
        // second line says which of the two facts this run established.
        println!("# every key is covered by exactly one package");
        ExitCode::SUCCESS
    } else {
        if uncovered > 0 {
            println!(
                "\n# {uncovered} of {} keys are not covered for {target}",
                rows.len()
            );
        } else {
            println!("\n# every key is covered for {target}, but not once");
        }
        ExitCode::FAILURE
    }
}

/// Loads every package under `<kernels>/<op>/<backend>/`, keyed by the operator its manifest names.
fn load_packages(kernels: &Path) -> Result<BTreeMap<String, Vec<PackageSource>>, String> {
    let mut packages: BTreeMap<String, Vec<PackageSource>> = BTreeMap::new();
    let ops = std::fs::read_dir(kernels)
        .map_err(|error| format!("reading {}: {error}", kernels.display()))?;
    for op in ops {
        let op = op.map_err(|error| format!("reading {}: {error}", kernels.display()))?;
        if !op.path().is_dir() {
            continue;
        }
        let backends = std::fs::read_dir(op.path())
            .map_err(|error| format!("reading {}: {error}", op.path().display()))?;
        for backend in backends {
            let backend =
                backend.map_err(|error| format!("reading {}: {error}", op.path().display()))?;
            let root = backend.path();
            if !root.join("package.toml").is_file() {
                continue;
            }
            let manifest =
                Manifest::load(&root).map_err(|error| format!("{}: {error}", root.display()))?;
            packages
                .entry(manifest.op.clone())
                .or_default()
                .push(PackageSource::new(root, manifest));
        }
    }
    for candidates in packages.values_mut() {
        candidates.sort_by(|a, b| a.manifest.backend.cmp(&b.manifest.backend));
    }
    Ok(packages)
}

/// The fixture's rows, in file order: `<operator>\t<descriptor key>`.
fn read_fixture(path: &Path) -> Result<Vec<(String, String)>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("reading {}: {error}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (op, key) = line
            .split_once('\t')
            .ok_or_else(|| format!("{}: not `<operator>\\t<key>`: {line}", path.display()))?;
        rows.push((op.to_owned(), key.to_owned()));
    }
    Ok(rows)
}

/// Which package covers this key for this target, or why none does.
fn cover(
    packages: &BTreeMap<String, Vec<PackageSource>>,
    op: &str,
    key: &str,
    target: &str,
) -> Result<Covering, String> {
    let Some(candidates) = packages.get(op) else {
        return Err("no package names this operator".to_owned());
    };
    let covering: Vec<&PackageSource> = candidates
        .iter()
        .filter(|candidate| candidate.manifest.covers(key))
        .collect();
    if covering.is_empty() {
        // Name the axis and value that leave the key uncovered, per package: the difference between
        // "the package needs another value added" and "this operator has no package for this key".
        let mut reasons = Vec::new();
        for candidate in candidates {
            reasons.push(format!(
                "{}/{}: {}",
                op,
                candidate.manifest.backend,
                explain(&candidate.manifest, key)
            ));
        }
        return Err(reasons.join("; "));
    }
    let built: Vec<&&PackageSource> = covering
        .iter()
        .filter(|candidate| candidate.manifest.builds_for(target))
        .collect();
    let Some(chosen) = built.first() else {
        let architectures: BTreeSet<&str> = covering
            .iter()
            .flat_map(|candidate| match &candidate.manifest.kernel {
                Some(kernel) => kernel
                    .architectures
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                None => vec!["wasm32"],
            })
            .collect();
        return Err(format!(
            "covered but not buildable for {target} (declares {})",
            architectures.into_iter().collect::<Vec<_>>().join(" ")
        ));
    };

    let source = PackageSource::new(chosen.root.clone(), chosen.manifest.clone());
    Ok(Covering {
        backend: chosen.manifest.backend.clone(),
        claimants: built
            .iter()
            .map(|candidate| candidate.manifest.backend.clone())
            .collect(),
        form: chosen.manifest.module.form,
        metadata: match chosen.manifest.module.form {
            // A native module's C entry builds its own launches, so there is no metadata to read.
            ModuleForm::Native | ModuleForm::Wasm | ModuleForm::Executable => None,
            ModuleForm::Cubin => Some(source.metadata_name(key)),
        },
    })
}

/// Why a package does not cover a key, in one phrase.
///
/// A package may declare one table of shapes or several; every shape it declares gets its own
/// reason, because "this key is the second shape's" and "this key is no shape's" are different
/// faults and a reader has to be able to tell them apart.
fn explain(manifest: &Manifest, key: &str) -> String {
    let Some(kernel) = &manifest.kernel else {
        return "a scheduling policy covers only its own descriptor".to_owned();
    };
    let shapes = kernel.descriptors.shapes();
    if shapes.len() == 1 {
        return explain_shape(&shapes[0], key);
    }
    shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| format!("shape {}: {}", index + 1, explain_shape(shape, key)))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Why one declared shape of a package does not cover a key.
fn explain_shape(descriptors: &program_server::manifest::Descriptors, key: &str) -> String {
    if descriptors.keys.iter().any(|named| named == key) {
        return "named outright".to_owned();
    }
    if descriptors.dimensions.is_empty() {
        return "declares no keys and no dimensions".to_owned();
    }
    let Some(pairs) = split_key(&descriptors.dimensions, key) else {
        return format!(
            "key is not a sequence of its dimensions ({})",
            descriptors
                .dimensions
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        );
    };
    let mut faults = Vec::new();
    for (dimension, value) in pairs {
        match descriptors.dimensions.get(&dimension) {
            None => faults.push(format!("{dimension} is not a dimension")),
            Some(values) if !values.iter().any(|allowed| allowed == value) => {
                faults.push(format!("{dimension}={value} not declared"));
            }
            Some(_) => {}
        }
    }
    if faults.is_empty() {
        "covered".to_owned()
    } else {
        faults.join(", ")
    }
}

/// Splits a descriptor key into `dimension_value` pairs, the way [`Manifest::covers`] does, so a
/// failure can name the pair instead of the whole key. The declared dimension names are matched
/// longest first, and a value runs to the next place where a declared name is followed by a value.
fn split_key<'a>(
    dimensions: &BTreeMap<String, Vec<String>>,
    key: &'a str,
) -> Option<Vec<(String, &'a str)>> {
    let mut names: Vec<&str> = dimensions.keys().map(String::as_str).collect();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let mut pairs = Vec::new();
    let mut rest = key;
    while !rest.is_empty() {
        let name = names.iter().find(|name| rest.starts_with(**name))?;
        let after = rest[name.len()..].strip_prefix('_')?;
        let end = names
            .iter()
            .filter_map(|next| after.find(&format!("_{next}_")))
            .min()
            .unwrap_or(after.len());
        let value = &after[..end];
        if value.is_empty() {
            return None;
        }
        pairs.push(((*name).to_owned(), value));
        rest = after[end..].strip_prefix('_').unwrap_or_default();
    }
    Some(pairs)
}
