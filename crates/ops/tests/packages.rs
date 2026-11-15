//! The kernel packages' rules: every package under `kernels/` installs as
//! design section 5.2 says and names the operator and backend of its
//! directories and an operator this build has, carries element type names
//! `ops` prints, and declares the toolchain version it was validated against;
//! only a package fixture may take the empty version, and one does.

#![cfg(test)]

use std::path::{Path, PathBuf};

use build_server::manifest::{Dimension, Manifest, ToolchainKind};
use tensor::DType;

/// The kernel packages: `kernels/<op>/<backend>/package.toml`.
fn kernel_packages() -> Vec<PathBuf> {
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut packages = Vec::new();
    for operator in subdirectories(&kernels) {
        for backend in subdirectories(&operator) {
            if backend.join(Manifest::FILE).is_file() {
                packages.push(backend);
            }
        }
    }
    assert!(
        !packages.is_empty(),
        "{} holds no package",
        kernels.display()
    );
    packages
}

/// The package fixtures the delivery tests install: `package-fixtures/<name>/package.toml`.
fn fixture_packages() -> Vec<PathBuf> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("package-fixtures");
    let packages: Vec<PathBuf> = subdirectories(&fixtures)
        .into_iter()
        .filter(|fixture| fixture.join(Manifest::FILE).is_file())
        .collect();
    assert!(
        !packages.is_empty(),
        "{} holds no fixture",
        fixtures.display()
    );
    packages
}

/// The subdirectories of `directory`, in name order.
fn subdirectories(directory: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    found.sort();
    found
}

/// The manifest of each kernel package, which every test but the first takes
/// as loaded.
fn kernel_manifests() -> Vec<(PathBuf, Manifest)> {
    kernel_packages()
        .into_iter()
        .map(|package| {
            let manifest = Manifest::load(&package)
                .unwrap_or_else(|error| panic!("{}: {error}", package.display()));
            (package, manifest)
        })
        .collect()
}

/// Fails with every line of `failures`, when there is one.
fn assert_none(failures: &[String]) {
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Every package installs: its manifest loads as the Build Server reads it,
/// and names the operator and backend its directories do.
#[test]
fn every_package_installs() {
    let mut failures = Vec::new();
    for package in kernel_packages() {
        let manifest = match Manifest::load(&package) {
            Ok(manifest) => manifest,
            Err(error) => {
                failures.push(format!("{}: {error}", package.display()));
                continue;
            }
        };
        let backend = package.file_name().and_then(|name| name.to_str());
        let operator = package
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str());
        if (Some(manifest.op.as_str()), Some(manifest.backend.as_str())) != (operator, backend) {
            failures.push(format!(
                "{}: the manifest says {} {}, its directories {operator:?} {backend:?}",
                package.display(),
                manifest.op,
                manifest.backend,
            ));
        }
    }
    assert_none(&failures);
}

/// The element types of the package ABI (`ops::abi`), as a kernel key spells
/// them.
fn element_type_names() -> Vec<&'static str> {
    [
        DType::U8,
        DType::U32,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::F8E4M3,
        DType::F4,
    ]
    .into_iter()
    .map(ops::abi::name)
    .collect()
}

/// The descriptor dimensions whose values are element types. Manifests do not
/// yet declare a dimension's value type, so the dimensions that carry one are
/// named here; a value of one of them is examined only when it is not a bare
/// integer, which tells an element type from a width.
const ELEMENT_TYPE_FIELDS: [&str; 7] = [
    "dtype",
    "input",
    "weight",
    "output",
    "accumulate",
    "scores",
    "scales",
];

/// Every element type a descriptor enumerates is one `ops::abi::name` prints,
/// so the keys the operators print are the keys the packages cover.
#[test]
fn every_element_type_is_one_ops_prints() {
    let names = element_type_names();
    let mut failures = Vec::new();
    for (_, manifest) in kernel_manifests() {
        for shape in manifest.descriptors() {
            for (field, dimension) in &shape.dimensions {
                // An element type is a specialization, so it is always an enumeration.
                let Dimension::Values(values) = dimension else {
                    continue;
                };
                if !ELEMENT_TYPE_FIELDS.contains(&field.as_str()) {
                    continue;
                }
                for value in values {
                    if value.parse::<u64>().is_err() && !names.contains(&value.as_str()) {
                        failures.push(format!(
                            "{} {}: {field} = {value:?} is not one of {names:?}",
                            manifest.op, manifest.backend
                        ));
                    }
                }
            }
        }
    }
    assert_none(&failures);
}

/// Every package that ships declares the toolchain version its modules were
/// validated against. The empty version, any version the host has, is for a
/// fixture that must build on whichever host its test runs on, and at least
/// one fixture takes it.
#[test]
fn every_shipping_package_declares_its_toolchain_version() {
    let failures: Vec<String> = kernel_manifests()
        .into_iter()
        .filter(|(_, manifest)| manifest.toolchain.kind != ToolchainKind::None)
        .filter(|(_, manifest)| manifest.toolchain.version.is_empty())
        .map(|(package, manifest)| {
            format!(
                "{}: {} {} declares no toolchain version",
                package.display(),
                manifest.op,
                manifest.backend
            )
        })
        .collect();
    assert_none(&failures);

    let unversioned = fixture_packages()
        .into_iter()
        .map(|fixture| {
            Manifest::load(&fixture)
                .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()))
        })
        .filter(|manifest| manifest.toolchain.version.is_empty())
        .count();
    assert!(
        unversioned > 0,
        "no fixture takes the empty version the rule exists for"
    );
}

/// Every package names an operator this build has: a package under another
/// name is a package revision nothing can request.
#[test]
fn every_package_names_an_operator_of_ops() {
    let operators = proto::module::operator::ALL;
    let failures: Vec<String> = kernel_manifests()
        .into_iter()
        .filter(|(_, manifest)| !operators.contains(&manifest.op.as_str()))
        .map(|(package, manifest)| {
            format!(
                "{}: `{}` is no operator of `ops`",
                package.display(),
                manifest.op
            )
        })
        .collect();
    assert_none(&failures);
}
