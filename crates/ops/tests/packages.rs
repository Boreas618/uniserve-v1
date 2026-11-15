//! The kernel packages' rules: every package under `kernels/` installs as
//! design section 5.2 says and names the operator and backend of its
//! directories and an operator this build has, carries element type names
//! `ops` prints, and states only what its source kind does not fix.

#![cfg(test)]

use std::path::{Path, PathBuf};

use build_server::manifest::{
    CUBIN_METADATA, Dimension, ENTRY_UNIT, Manifest, SourceKind, launch_entry,
};
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

/// Every package states only what is its own. The toolchain, the module's form, a cubin's
/// metadata file, a native module's launch entry and a CuTe DSL module's entry unit are what the
/// source kind fixes (`build_server::manifest`), and the toolchain's version is the Build
/// Server's environment's: a manifest that restates one of them is a copy of the Build Server's
/// rule, which drifts from it when the rule changes.
#[test]
fn every_package_states_only_what_its_source_kind_does_not_fix() {
    let mut failures = Vec::new();
    for package in kernel_packages().into_iter().chain(fixture_packages()) {
        let path = package.join(Manifest::FILE);
        let declared: toml::Table = toml::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let manifest = Manifest::load(&package)
            .unwrap_or_else(|error| panic!("{}: {error}", package.display()));
        let kind = manifest.source.kind;
        let table = |name: &str| declared.get(name).and_then(toml::Value::as_table);
        let stated = |table: Option<&toml::Table>, key: &str| {
            table
                .and_then(|table| table.get(key))
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        };

        let mut restated = Vec::new();
        if table("toolchain").is_some_and(|toolchain| toolchain.contains_key("version")) {
            restated.push("[toolchain] version".to_owned());
        }
        if stated(table("toolchain"), "kind").as_deref() == Some(kind.toolchain().as_str()) {
            restated.push("[toolchain] kind".to_owned());
        }
        let form = table("module")
            .and_then(|module| module.get("form"))
            .map(|form| {
                form.clone()
                    .try_into::<build_server::manifest::ModuleForm>()
            });
        if let Some(Ok(form)) = form
            && form == kind.module_form()
        {
            restated.push("[module] form".to_owned());
        }
        if stated(table("module"), "metadata").as_deref() == Some(CUBIN_METADATA) {
            restated.push("[module] metadata".to_owned());
        }
        let entries = table("module")
            .and_then(|module| module.get("entries"))
            .and_then(toml::Value::as_table);
        if stated(entries, "launch") == Some(launch_entry(&manifest.op)) {
            restated.push("[module.entries] launch".to_owned());
        }
        if kind == SourceKind::CuteDsl
            && stated(table("source"), "unit").as_deref() == Some(ENTRY_UNIT)
        {
            restated.push("[source] unit".to_owned());
        }
        if !restated.is_empty() {
            failures.push(format!(
                "{}: restates what its source kind fixes: {restated:?}",
                package.display()
            ));
        }
    }
    assert_none(&failures);
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

/// The routed NVFP4 products' packages compile their kernels for the weights' block-scale layout
/// the operator states, and select by the descriptor's own layout for the tokens.
///
/// A product's kernels are compiled with one layout for each operand, and nothing compares the
/// two at a launch: a package built for one form and handed a table in the other reads every
/// block scale at the wrong offset and computes plausible numbers.
#[test]
fn a_routed_nvfp4_package_reads_the_weight_scale_layout_its_operator_states() {
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    for operator in ["gather_gemm_nvfp4", "route_gemm_nvfp4"] {
        let package = kernels.join(operator).join("trtllm");
        let manifest: toml::Table = std::fs::read_to_string(package.join(Manifest::FILE))
            .unwrap()
            .parse()
            .unwrap();
        let selected = &manifest["parameters"]["selected"];
        assert_eq!(
            selected["mSfLayoutA"].as_str(),
            Some(ops::moe::grouped::WEIGHT_SCALE_LAYOUT.to_string().as_str()),
            "{operator}'s package compiles for another weight scale layout than its operator's"
        );
        assert_eq!(
            selected["mSfLayoutB"]["field"].as_str(),
            Some("layout"),
            "{operator}'s package selects its kernels by another token layout than the key's"
        );
    }
}
