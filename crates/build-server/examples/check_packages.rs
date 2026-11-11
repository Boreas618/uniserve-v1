//! The kernel packages' rules, checked over the directories the caller names: every package
//! parses as design section 5.2 says, names the operator and backend of its directories, declares
//! its toolchain version, names an operator the `ops` crate declares, and carries element type
//! names `ops` prints.
//!
//!     cargo run -p build-server --example check_packages -- \
//!         --kernels <kernel package root> --fixtures <package fixture root> [--keys <key file>]
//!
//! `--keys` names a key file (`operator<TAB>key` per row): every row must be covered by a package
//! of its operator, or declared blocked in the file. Without it that check is not run, and the
//! report says so. The exit status is 1 when any check fails.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use build_server::manifest::{Dimension, Manifest};

struct Args {
    kernels: PathBuf,
    fixtures: PathBuf,
    keys: Option<PathBuf>,
}

fn arguments() -> Result<Args, String> {
    let mut kernels = None;
    let mut fixtures = None;
    let mut keys = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} takes a value"))
        };
        match flag.as_str() {
            "--kernels" => kernels = Some(PathBuf::from(value()?)),
            "--fixtures" => fixtures = Some(PathBuf::from(value()?)),
            "--keys" => keys = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        kernels: kernels.ok_or("--kernels names the kernel package root")?,
        fixtures: fixtures.ok_or("--fixtures names the package fixture root")?,
        keys,
    })
}

fn main() -> ExitCode {
    let args = match arguments() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("check_packages: {message}");
            return ExitCode::from(2);
        }
    };
    let packages = match packages(&args.kernels) {
        Ok(packages) if !packages.is_empty() => packages,
        Ok(_) => {
            eprintln!(
                "check_packages: {} holds no package",
                args.kernels.display()
            );
            return ExitCode::from(2);
        }
        Err(message) => {
            eprintln!("check_packages: {message}");
            return ExitCode::from(2);
        }
    };
    let checks: Vec<(&str, Result<(), String>)> = vec![
        ("every package installs", every_package_installs(&packages)),
        (
            "every element type is one ops prints",
            every_element_type_is_one_ops_prints(&packages),
        ),
        (
            "every shipping package declares its toolchain version",
            every_shipping_package_declares_its_toolchain_version(&packages, &args.fixtures),
        ),
        (
            "the exchange keeps its group sizes apart",
            the_exchange_keeps_its_group_sizes_apart(&args.kernels),
        ),
        (
            "every package names an operator of ops",
            every_package_names_an_operator(&packages),
        ),
    ];
    let mut failed = false;
    for (name, result) in checks {
        match result {
            Ok(()) => println!("passed  {name}"),
            Err(message) => {
                failed = true;
                println!("FAILED  {name}\n{message}");
            }
        }
    }
    match &args.keys {
        Some(keys) => match every_key_is_covered(&packages, keys) {
            Ok(()) => println!("passed  every key of {} is covered", keys.display()),
            Err(message) => {
                failed = true;
                println!(
                    "FAILED  every key of {} is covered\n{message}",
                    keys.display()
                );
            }
        },
        None => println!("not run every key is covered: no --keys"),
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// The packages under `kernels`: `<kernels>/<op>/<backend>/package.toml`.
fn packages(kernels: &Path) -> Result<Vec<PathBuf>, String> {
    let mut packages = Vec::new();
    let operators =
        std::fs::read_dir(kernels).map_err(|error| format!("{}: {error}", kernels.display()))?;
    for operator in operators {
        let operator = operator.map_err(|error| error.to_string())?.path();
        if !operator.is_dir() {
            continue;
        }
        let backends = std::fs::read_dir(&operator)
            .map_err(|error| format!("{}: {error}", operator.display()))?;
        for backend in backends {
            let backend = backend.map_err(|error| error.to_string())?.path();
            if backend.join("package.toml").is_file() {
                packages.push(backend);
            }
        }
    }
    packages.sort();
    Ok(packages)
}

fn every_package_installs(packages: &[PathBuf]) -> Result<(), String> {
    let mut failures = Vec::new();
    for package in packages {
        match Manifest::load(package) {
            Ok(manifest) => {
                let operator = package
                    .parent()
                    .and_then(Path::file_name)
                    .and_then(|name| name.to_str());
                let backend = package.file_name().and_then(|name| name.to_str());
                if Some(manifest.op.as_str()) != operator
                    || Some(manifest.backend.as_str()) != backend
                {
                    failures.push(format!(
                        "{}: manifest says {} {}, directories say {operator:?} {backend:?}",
                        package.display(),
                        manifest.op,
                        manifest.backend,
                    ));
                }
            }
            Err(error) => failures.push(format!("{}: {error}", package.display())),
        }
    }
    joined(failures)
}

/// The rows of a key file: everything that is not a `#` comment, split at the tab.
fn rows_of(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            let (operator, key) = line.split_once('\t')?;
            Some((operator.trim().to_owned(), key.trim().to_owned()))
        })
        .collect()
}

/// The blockers a key file declares, keyed by `(operator, key)` with the coverage tool's sentence.
fn declared_blocked_with_reasons(text: &str) -> BTreeMap<(String, String), String> {
    text.lines()
        .filter_map(|line| line.strip_prefix("# blocked:\t"))
        .filter_map(|row| {
            let mut fields = row.split('\t');
            let operator = fields.next()?.trim().to_owned();
            let key = fields.next()?.trim().to_owned();
            let why = fields.next().unwrap_or("").trim().to_owned();
            Some(((operator, key), why))
        })
        .collect()
}

fn every_key_is_covered(packages: &[PathBuf], keys: &Path) -> Result<(), String> {
    let text =
        std::fs::read_to_string(keys).map_err(|error| format!("{}: {error}", keys.display()))?;
    let rows = rows_of(&text);
    if rows.is_empty() {
        return Err(format!("{} names no keys", keys.display()));
    }
    let blocked = declared_blocked_with_reasons(&text);
    let manifests: Vec<Manifest> = packages
        .iter()
        .filter_map(|package| Manifest::load(package).ok())
        .collect();
    let mut failures: Vec<((String, String), String)> = Vec::new();
    for (operator, key) in &rows {
        let mut covering = manifests
            .iter()
            .filter(|manifest| &manifest.op == operator)
            .peekable();
        if covering.peek().is_none() {
            failures.push((
                (operator.clone(), key.clone()),
                format!("{operator}: no package fills it"),
            ));
            continue;
        }
        if !covering.any(|manifest| manifest.covers(key)) {
            failures.push((
                (operator.clone(), key.clone()),
                format!(
                    "{operator}: no package covers {key} (its descriptors: {})",
                    manifests
                        .iter()
                        .filter(|manifest| &manifest.op == operator)
                        .map(|manifest| format!(
                            "{} {:?}",
                            manifest.backend,
                            manifest.kernel.as_ref().map(|kernel| &kernel.descriptors)
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
            ));
        }
    }
    let mut problems = Vec::new();
    for (row, message) in &failures {
        if !blocked.contains_key(row) {
            problems.push(format!(
                "row {}\t{} is not declared blocked and does not build: {message}",
                row.0, row.1
            ));
        }
    }
    for (row, why) in &blocked {
        if !failures.iter().any(|(failing, _)| failing == row) {
            problems.push(format!(
                "row {}\t{} is declared blocked and now builds (it was: {why})",
                row.0, row.1
            ));
        }
    }
    joined(problems)
}

/// The names `ops` spells the package ABI's element types with in a key: the vocabulary a
/// descriptor value that carries an element type must come from.
fn element_type_names() -> Vec<String> {
    [
        uniserve_tensor::DType::U8,
        uniserve_tensor::DType::U32,
        uniserve_tensor::DType::I16,
        uniserve_tensor::DType::I32,
        uniserve_tensor::DType::I64,
        uniserve_tensor::DType::F16,
        uniserve_tensor::DType::BF16,
        uniserve_tensor::DType::F32,
        uniserve_tensor::DType::F64,
        uniserve_tensor::DType::F8E4M3,
        uniserve_tensor::DType::F4,
    ]
    .iter()
    .map(|dtype| ops::abi::name(*dtype).to_owned())
    .collect()
}

/// The descriptor fields whose values are element types (as `ops::abi::name` spells them). A value in one
/// of them is examined only when it is not a bare integer, which tells an element type from a width.
const ELEMENT_TYPE_FIELDS: [&str; 7] = [
    "dtype",
    "input",
    "weight",
    "output",
    "accumulate",
    "scores",
    "scales",
];

fn every_element_type_is_one_ops_prints(packages: &[PathBuf]) -> Result<(), String> {
    let names = element_type_names();
    let mut failures = Vec::new();
    for package in packages {
        let manifest =
            Manifest::load(package).map_err(|error| format!("{}: {error}", package.display()))?;
        let Some(kernel) = &manifest.kernel else {
            continue;
        };
        for shape in kernel.descriptors.shapes() {
            for (field, dimension) in &shape.dimensions {
                // An element type is a specialization, so it is always an enumeration.
                let Dimension::Values(values) = dimension else {
                    continue;
                };
                if !ELEMENT_TYPE_FIELDS.contains(&field.as_str()) {
                    continue;
                }
                for value in values {
                    if value.parse::<u64>().is_err() && !names.contains(value) {
                        failures.push(format!(
                            "{} {}: {field} = {value:?} is not one of {names:?}",
                            manifest.op, manifest.backend
                        ));
                    }
                }
            }
        }
    }
    joined(failures)
}

/// Every package that ships declares the toolchain version it was built against; only a package revision
/// fixture may leave it empty, and one of them does.
fn every_shipping_package_declares_its_toolchain_version(
    packages: &[PathBuf],
    fixtures: &Path,
) -> Result<(), String> {
    let mut failures = Vec::new();
    for package in packages {
        let Ok(manifest) = Manifest::load(package) else {
            continue;
        };
        if manifest.toolchain.kind == build_server::manifest::ToolchainKind::None {
            continue;
        }
        if manifest.toolchain.version.is_empty() {
            failures.push(format!(
                "{}: {} {} declares no toolchain version",
                package.display(),
                manifest.op,
                manifest.backend
            ));
        }
    }
    let mut empty = 0;
    let mut seen = 0;
    for entry in
        std::fs::read_dir(fixtures).map_err(|error| format!("{}: {error}", fixtures.display()))?
    {
        let manifest = entry
            .map_err(|error| error.to_string())?
            .path()
            .join("package.toml");
        if !manifest.is_file() {
            continue;
        }
        seen += 1;
        let text = std::fs::read_to_string(&manifest)
            .map_err(|error| format!("{}: {error}", manifest.display()))?;
        let parsed = text
            .parse::<toml::Table>()
            .map_err(|error| format!("{}: {error}", manifest.display()))?;
        let version = parsed
            .get("toolchain")
            .and_then(|toolchain| toolchain.get("version"))
            .and_then(toml::Value::as_str)
            .ok_or_else(|| format!("{}: no [toolchain] version", manifest.display()))?;
        if version.is_empty() {
            empty += 1;
        }
    }
    if seen == 0 {
        failures.push(format!("{} holds no fixture package", fixtures.display()));
    } else if empty == 0 {
        failures.push("no fixture takes the empty version the rule exists for".to_owned());
    }
    joined(failures)
}

/// The expert exchange's topology is a descriptor dimension, so a module built for one group size
/// is never resolved for another. Whether a group divides the experts relates two dimensions, which
/// the exchange's create checks for the call rather than the manifest for the key.
fn the_exchange_keeps_its_group_sizes_apart(kernels: &Path) -> Result<(), String> {
    let package = kernels.join("expert_exchange/alltoall");
    let manifest =
        Manifest::load(&package).map_err(|error| format!("{}: {error}", package.display()))?;
    let key = |ranks: u32| {
        format!(
            "mode_dispatch_hidden_2048_intermediate_768_top_8_experts_128_max_tokens_8192_ranks_{ranks}_attention_0\
             _dtype_bf16"
        )
    };
    let mut failures = Vec::new();
    for ranks in [1, 2, 4, 8] {
        if !manifest.covers(&key(ranks)) {
            failures.push(format!(
                "the package does not build the module of a group of {ranks} ranks"
            ));
        }
    }
    joined(failures)
}

/// Every package names an operator the `ops` crate declares: a package under another name is a
/// package revision nothing can request.
fn every_package_names_an_operator(packages: &[PathBuf]) -> Result<(), String> {
    let operators = proto::module::operator::ALL;
    let mut failures = Vec::new();
    for package in packages {
        let Ok(manifest) = Manifest::load(package) else {
            continue;
        };
        if !operators.contains(&manifest.op.as_str()) {
            failures.push(format!(
                "{}: op `{}` is no operator of `ops`",
                package.display(),
                manifest.op
            ));
        }
    }
    joined(failures)
}

fn joined(failures: Vec<String>) -> Result<(), String> {
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}
