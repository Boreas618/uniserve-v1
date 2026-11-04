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
