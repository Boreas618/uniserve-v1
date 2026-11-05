//! Counts which kernel packages can be observed by a check, and which fixture
//! descriptors are therefore unobserved.
//!
//! A package is *observable* when it carries a check script: a Python file
//! under its `test/` directory other than `__init__.py`. The store's proof
//! says every fixture descriptor compiles; this says how many of them any
//! script has ever looked at. The two numbers answer different questions, and
//! the second is the one that decides whether a family's arithmetic is
//! evidence or an assumption.
//!
//! ```text
//! cargo run -p program-server --example package_census -- \
//!     --keys notes/programs/qwen3-moe-keys.txt \
//!     --keys notes/programs/minimax-h3-keys.txt
//! ```
//!
//! It reads manifests only and builds nothing, so it runs in a second.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use program_server::Manifest;

/// One line of a key file: the operator and one descriptor key.
struct Fixture {
    op: String,
    key: String,
}

/// A package on disk, with what the census needs of it.
struct Package {
    op: String,
    backend: String,
    /// Whether the package has a check script.
    observed: bool,
    /// The fixture keys this package covers.
    keys: BTreeSet<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("package_census: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut packages_root = PathBuf::from("crates/ops/kernels");
    let mut keys: Vec<PathBuf> = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} takes a value"))
        };
        match flag.as_str() {
            "--packages" => packages_root = PathBuf::from(value()?),
            "--keys" => keys.push(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if keys.is_empty() {
        keys.push(PathBuf::from("notes/programs/qwen3-moe-keys.txt"));
        keys.push(PathBuf::from("notes/programs/minimax-h3-keys.txt"));
    }

    let fixtures = read_fixtures(&keys)?;
    let packages = read_packages(&packages_root, &fixtures)?;
    report(&packages, &fixtures);
    Ok(())
}

/// Reads the `operator<TAB>key` lines of the fixture files.
fn read_fixtures(paths: &[PathBuf]) -> Result<Vec<Fixture>, String> {
    let mut fixtures = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((op, key)) = line.split_once('\t') else {
                return Err(format!(
                    "{}: a fixture line is `operator<TAB>key`, not {line:?}",
                    path.display()
                ));
            };
            fixtures.push(Fixture {
                op: op.trim().to_owned(),
                key: key.trim().to_owned(),
            });
        }
    }
    Ok(fixtures)
}

/// Reads every `package.toml` under the root, two names deep as the store's
/// paths are (`<op>/<backend>`).
fn read_packages(root: &Path, fixtures: &[Fixture]) -> Result<Vec<Package>, String> {
    let mut packages = Vec::new();
    let ops =
        std::fs::read_dir(root).map_err(|error| format!("reading {}: {error}", root.display()))?;
    for op_entry in ops {
        let op_entry = op_entry.map_err(|error| format!("{}: {error}", root.display()))?;
        if !op_entry.path().is_dir() {
            continue;
        }
        let backends = match std::fs::read_dir(op_entry.path()) {
            Ok(backends) => backends,
            Err(_) => continue,
        };
        for backend_entry in backends {
            let backend_entry =
                backend_entry.map_err(|error| format!("{}: {error}", op_entry.path().display()))?;
            let package_root = backend_entry.path();
            if !package_root.join("package.toml").is_file() {
                continue;
            }
            let manifest = Manifest::load(&package_root).map_err(|error| {
                format!("{}: {error}", package_root.join("package.toml").display())
            })?;
            let covered = fixtures
                .iter()
                .filter(|fixture| {
                    fixture.op == manifest.op
                        && manifest
                            .kernel
                            .as_ref()
                            .is_some_and(|kernel| kernel.descriptors.covers(&fixture.key))
                })
                .map(|fixture| fixture.key.clone())
                .collect();
            packages.push(Package {
                op: manifest.op.clone(),
                backend: manifest.backend.clone(),
                observed: has_check_script(&package_root),
                keys: covered,
            });
        }
    }
    packages.sort_by(|left, right| (&left.op, &left.backend).cmp(&(&right.op, &right.backend)));
    Ok(packages)
}

/// Whether the package carries a check script: a Python file under `test/`
/// other than the package marker.
fn has_check_script(package_root: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(package_root.join("test")) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let path = entry.path();
        path.extension().is_some_and(|extension| extension == "py")
            && path.file_name().is_some_and(|name| name != "__init__.py")
    })
}

/// Prints the two counts and the packages the fixture asks for that no script
/// observes.
fn report(packages: &[Package], fixtures: &[Fixture]) {
    let observed_packages = packages.iter().filter(|package| package.observed).count();
    println!(
        "{} packages: {observed_packages} with a check script, {} without",
        packages.len(),
        packages.len() - observed_packages
    );

    // A descriptor is observed when some package that covers it has a script.
    let mut observers: BTreeMap<&str, bool> = BTreeMap::new();
    for package in packages {
        for key in &package.keys {
            let entry = observers.entry(key.as_str()).or_insert(false);
            *entry = *entry || package.observed;
        }
    }
    let asked: BTreeSet<&str> = fixtures
        .iter()
        .map(|fixture| fixture.key.as_str())
        .collect();
    let unclaimed: Vec<&&str> = asked
        .iter()
        .filter(|key| !observers.contains_key(**key))
        .collect();
    let observed = observers.values().filter(|observed| **observed).count();
    println!(
        "{} fixture descriptors: {observed} covered by a package with a check script, {} without",
        observers.len(),
        observers.len() - observed
    );
    if !unclaimed.is_empty() {
        println!(
            "{} fixture descriptors no package covers at all",
            unclaimed.len()
        );
        for key in unclaimed {
            println!("  no package   {key}");
        }
    }

    // Which packages the fixture asks for that nothing observes, the ones with
    // the most descriptors first: those are where a missing script costs most.
    let mut silent: Vec<&Package> = packages
        .iter()
        .filter(|package| !package.observed && !package.keys.is_empty())
        .collect();
    silent.sort_by(|left, right| {
        right
            .keys
            .len()
            .cmp(&left.keys.len())
            .then_with(|| (&left.op, &left.backend).cmp(&(&right.op, &right.backend)))
    });
    println!("\npackages the fixture asks for with no check script:");
    for package in &silent {
        println!(
            "  {:<40} {:<10} {} descriptors",
            package.op,
            package.backend,
            package.keys.len()
        );
    }
}
