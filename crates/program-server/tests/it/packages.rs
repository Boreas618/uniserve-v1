//! The packages of `crates/ops/kernels` are the manifest's contract: every
//! package in the tree parses as the design section 5.2 says, names the
//! operator and backend of its directories, and its declared files exist.

use std::collections::BTreeMap;
use std::path::Path;

use program_server::manifest::Manifest;

/// The kernel packages of this repository: `crates/ops/kernels/<op>/<backend>/`.
fn packages() -> Vec<std::path::PathBuf> {
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ops/kernels");
    let mut packages = Vec::new();
    let Ok(operators) = std::fs::read_dir(&kernels) else {
        return packages;
    };
    for operator in operators {
        let operator = operator.expect("a kernel operator directory").path();
        if !operator.is_dir() {
            continue;
        }
        for backend in std::fs::read_dir(&operator).expect("a backend directory") {
            let backend = backend.expect("a backend directory").path();
            if backend.join("package.toml").is_file() {
                packages.push(backend);
            }
        }
    }
    packages.sort();
    packages
}

#[test]
fn every_kernel_package_in_the_tree_is_a_package_the_server_can_install() {
    let packages = packages();
    assert!(
        !packages.is_empty(),
        "the tree ships the kernel packages of design section 9"
    );
    let mut failures = Vec::new();
    for package in &packages {
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
                        "{}: manifest says {} {}, directories say {:?} {:?}",
                        package.display(),
                        manifest.op,
                        manifest.backend,
                        operator,
                        backend
                    ));
                }
            }
            Err(error) => failures.push(format!("{}: {error}", package.display())),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The Qwen3-MoE vertical slice's descriptor keys, which the model's own
/// example writes: `operator<TAB>key` per variant.
///
/// The file is generated, not committed (`cargo run -p ops --example
/// qwen3_moe_keys` writes `.local/qwen3-moe-keys.txt`), so the test reports
/// how to produce it when it is absent.
fn slice_keys() -> Option<Vec<(String, String)>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/qwen3-moe-keys.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIPPED (fixture): no {}; write it with `cargo run -p ops --example qwen3_moe_keys`",
            path.display()
        );
        return None;
    };
    Some(
        text.lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .filter_map(|line| {
                let (operator, key) = line.split_once('\t')?;
                Some((operator.trim().to_owned(), key.trim().to_owned()))
            })
            .collect(),
    )
}

#[test]
fn every_qwen3_moe_key_of_the_slice_is_covered_by_its_operator_s_package() {
    let Some(keys) = slice_keys() else {
        return;
    };
    assert!(!keys.is_empty(), "the slice's fixture names no keys");

    let mut manifests = Vec::new();
    for package in packages() {
        if let Ok(manifest) = Manifest::load(&package) {
            manifests.push(manifest);
        }
    }

    let mut failures = Vec::new();
    for (operator, key) in &keys {
        let mut covering = manifests
            .iter()
            .filter(|manifest| &manifest.op == operator)
            .peekable();
        if covering.peek().is_none() {
            failures.push(format!("{operator}: no package of the tree fills it"));
            continue;
        }
        if !covering.any(|manifest| manifest.covers(key)) {
            failures.push(format!(
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
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} slice keys are not covered:\n{}",
        failures.len(),
        keys.len(),
        failures.join("\n")
    );
}

/// The names `ops` prints for its element types: the vocabulary a descriptor
/// value that carries an element type must come from. Taken from `ops`' own
/// `DType`, so a package cannot invent a spelling the operator will never
/// produce (`fp16` for `f16`, say) and then silently cover no key.
fn element_type_names() -> Vec<String> {
    [
        ops::DType::U8,
        ops::DType::I32,
        ops::DType::I64,
        ops::DType::F16,
        ops::DType::BF16,
        ops::DType::F32,
        ops::DType::F8E4M3,
        ops::DType::F4E2M1,
    ]
    .iter()
    .map(ToString::to_string)
    .collect()
}

/// The descriptor fields whose values are element types (`ops`' `DType`
/// `Display`), across the operators of design section 9.
const ELEMENT_TYPE_FIELDS: [&str; 7] =
    ["dtype", "input", "weight", "output", "accumulate", "scores", "scales"];

#[test]
fn every_element_type_a_descriptor_can_carry_is_one_ops_prints() {
    let names = element_type_names();
    let mut failures = Vec::new();
    for package in packages() {
        let manifest = Manifest::load(&package).expect("a package the tree ships");
        let Some(kernel) = &manifest.kernel else {
            continue;
        };
        for (field, values) in &kernel.descriptors.dimensions {
            if !ELEMENT_TYPE_FIELDS.contains(&field.as_str()) {
                continue;
            }
            for value in values {
                if !names.contains(value) {
                    failures.push(format!(
                        "{} {}: {field} = {value:?} is not one of {names:?}",
                        manifest.op, manifest.backend
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} descriptor values name no element type `ops` prints:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The keys a fixture asks for, as `operator<TAB>key` lines: the H3 file is the one
/// model commits, the slice's is generated.
fn fixture_keys(path: &Path) -> Option<Vec<(String, String)>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("SKIPPED (fixture): no {}", path.display());
        return None;
    };
    Some(
        text.lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .filter_map(|line| {
                let (operator, key) = line.split_once('\t')?;
                Some((operator.trim().to_owned(), key.trim().to_owned()))
            })
            .collect(),
    )
}

/// The field names and values a key parses to under a package's declared dimensions.
///
/// The rule is the one `Descriptors::covers` applies — a declared dimension name is
/// matched longest first, and its value runs to the next matched name — asked in the
/// direction that method does not ask: every declared dimension must be *found* in the
/// key, so a package cannot declare a field its operator never prints, and a key cannot
/// carry a field its package never declared. A key that leaves a declared dimension
/// unread is not a key of that package.
fn parse_key(dimensions: &BTreeMap<String, Vec<String>>, key: &str) -> Option<Vec<(String, String)>> {
    let mut names: Vec<&str> = dimensions.keys().map(String::as_str).collect();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let mut fields = Vec::new();
    let mut rest = key;
    while !rest.is_empty() {
        let name = names
            .iter()
            .find(|name| rest.starts_with(**name) && rest[name.len()..].starts_with('_'))?;
        let after = &rest[name.len() + 1..];
        // The value ends where the next declared name begins at a separator, or at the
        // key's end; a value may itself contain underscores and dashes.
        let mut end = after.len();
        for (index, _) in after.match_indices('_') {
            let tail = &after[index + 1..];
            if names
                .iter()
                .any(|candidate| tail.starts_with(*candidate) && tail[candidate.len()..].starts_with('_'))
            {
                end = index;
                break;
            }
        }
        fields.push(((*name).to_owned(), after[..end].to_owned()));
        rest = if end == after.len() { "" } else { &after[end + 1..] };
    }
    Some(fields)
}

/// Every fixture key parses to exactly the fields one package of its operator declares.
///
/// The gates beside this one compare a package's declared values against `ops`'s domains,
/// which leaves the other direction open: a package that declares a field the operator
/// never prints, or one that declares dimensions its key cannot fill, stays green until a
/// build of that exact key fails. This one reads the keys a fixture asks for and asks each
/// operator's packages whether the key is one of theirs — field for field.
#[test]
fn every_fixture_key_parses_to_the_fields_its_operator_s_packages_declare() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixtures = [
        root.join("../../.local/qwen3-moe-keys.txt"),
        root.join("../../notes/programs/minimax-h3-keys.txt"),
    ];
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for fixture in &fixtures {
        let Some(keys) = fixture_keys(fixture) else {
            continue;
        };
        for (operator, key) in keys {
            let declared: Vec<(String, BTreeMap<String, Vec<String>>)> = packages()
                .into_iter()
                .filter(|package| {
                    package
                        .parent()
                        .and_then(Path::file_name)
                        .and_then(|name| name.to_str())
                        == Some(operator.as_str())
                })
                .filter_map(|package| {
                    let manifest = Manifest::load(&package).ok()?;
                    let descriptors = manifest.kernel?.descriptors;
                    Some((package.display().to_string(), descriptors.dimensions))
                })
                .filter(|(_, dimensions)| !dimensions.is_empty())
                .collect();
            // An operator with no package at all is the coverage gate's business.
            if declared.is_empty() {
                continue;
            }
            checked += 1;
            let covering = declared.iter().any(|(_, dimensions)| {
                parse_key(dimensions, &key).is_some_and(|fields| {
                    fields.len() == dimensions.len()
                        && fields.iter().all(|(name, value)| {
                            dimensions
                                .get(name)
                                .is_some_and(|values| values.iter().any(|allowed| allowed == value))
                        })
                })
            });
            if !covering {
                let fields: Vec<String> = declared
                    .iter()
                    .map(|(package, dimensions)| {
                        let names: Vec<&str> = dimensions.keys().map(String::as_str).collect();
                        format!("{package} declares [{}]", names.join(", "))
                    })
                    .collect();
                failures.push(format!(
                    "{operator}\t{key}: no package declares this key's fields exactly; {}",
                    fields.join("; ")
                ));
            }
        }
    }
    assert!(
        checked > 0,
        "no fixture key met a package; both fixture files are missing"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every package that ships declares the toolchain version it was built and
/// validated against; only the fixtures under `crates/ops/program-fixtures` may
/// leave it empty.
///
/// An empty version means "any version this host has" (design section 5.2), which
/// is a real weakening of the rule that a toolchain change is a new revision: it
/// is what a fixture whose build does not depend on a patch release declares, so
/// that a delivery test does not fail for the wrong reason. A shipping package
/// keeps the guard, and this test is what makes the waiver impossible to take by
/// accident.
#[test]
fn every_shipping_package_declares_its_toolchain_version() {
    let shipping = packages();
    assert!(!shipping.is_empty(), "the tree ships the kernel packages");
    let mut failures = Vec::new();
    for package in &shipping {
        let Ok(manifest) = Manifest::load(package) else {
            continue; // The install gate reports a manifest that does not load.
        };
        // A package that compiles nothing has no compiler version to match.
        if manifest.toolchain.kind == program_server::manifest::ToolchainKind::None {
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
    assert!(
        failures.is_empty(),
        "a shipping package must declare its toolchain version; only a fixture \
         may leave it empty:\n{}",
        failures.join("\n")
    );

    // The other half of the rule: the fixtures are where an empty version is
    // allowed, and one of them takes it.
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/ops/program-fixtures")
        .canonicalize()
        .expect("the fixtures' directory");
    let mut empty = 0;
    let mut seen = 0;
    for entry in std::fs::read_dir(&fixtures).expect("the fixtures") {
        let manifest = entry.expect("a fixture").path().join("package.toml");
        if !manifest.is_file() {
            continue;
        }
        seen += 1;
        let text = std::fs::read_to_string(&manifest).expect("a fixture manifest");
        let parsed = text.parse::<toml::Table>().expect("a fixture manifest");
        let version = parsed
            .get("toolchain")
            .and_then(|toolchain| toolchain.get("version"))
            .and_then(toml::Value::as_str)
            .expect("a fixture's [toolchain] version");
        if version.is_empty() {
            empty += 1;
        }
    }
    assert!(seen > 0, "the tree ships the fixture packages");
    assert!(
        empty > 0,
        "no fixture takes the empty version the rule exists for"
    );
}
