//! The packages of `crates/ops/kernels` are the manifest's contract: every
//! package in the tree parses as the design section 5.2 says, names the
//! operator and backend of its directories, and its declared files exist.

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
