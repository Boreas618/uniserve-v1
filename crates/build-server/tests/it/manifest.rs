//! The package manifest, `package.toml` (design section 5.2): what it says,
//! the descriptors it covers, and the packages it refuses.

use std::fs;
use std::path::{Path, PathBuf};

use build_server::manifest::{Manifest, ModuleForm, SourceKind, ToolchainKind};

/// Writes a package into a fresh directory and answers its root.
fn package(text: &str, files: &[&str]) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("package.toml"), text).unwrap();
    for file in files {
        let path = root.path().join(file);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, b"# a source\n").unwrap();
    }
    root
}

/// The manifest of a Triton `rms_norm` package: the shape every package of
/// section 5.2 has.
const REASONABLE: &str = r#"
op = "rms_norm"
backend = "triton"
tuning = ["tuning/tactics.json"]

[toolchain]
kind = "triton"
version = "3.7.1"

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/rms.py"]
script = "src/compile.py"
kernel = "rms_norm"

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16", "fp32"], hidden = ["2048"] } }
selection = { tile = 1024 }

[parameters]
tile = 1024
"#;

#[test]
fn a_manifest_says_what_the_package_is_and_how_it_is_built() {
    let root = package(
        REASONABLE,
        &["src/rms.py", "src/compile.py", "tuning/tactics.json"],
    );
    let manifest = Manifest::load(root.path()).unwrap();

    assert_eq!(manifest.op, "rms_norm");
    assert_eq!(manifest.backend, "triton");
    assert_eq!(manifest.toolchain.kind, ToolchainKind::Triton);
    assert_eq!(manifest.toolchain.version, "3.7.1");
    assert_eq!(manifest.module.form, ModuleForm::Cubin);
    assert_eq!(manifest.source.kind, SourceKind::Triton);
    assert_eq!(manifest.source.kernel.as_deref(), Some("rms_norm"));
    assert_eq!(manifest.kernel.as_ref().unwrap().architectures, ["sm_100a"]);
    assert_eq!(manifest.tuning, ["tuning/tactics.json"]);
    assert_eq!(
        manifest.parameters.get("tile"),
        Some(&toml::Value::Integer(1024))
    );
}

#[test]
fn a_package_covers_a_descriptor_by_its_enumerated_values() {
    let root = package(
        REASONABLE,
        &["src/rms.py", "src/compile.py", "tuning/tactics.json"],
    );
    let manifest = Manifest::load(root.path()).unwrap();
    let descriptors = &manifest.kernel.as_ref().unwrap().descriptors;

    assert!(descriptors.covers("dtype_bf16_hidden_2048"));
    assert!(descriptors.covers("dtype_fp32_hidden_2048"));
    // A value the package does not declare is not covered.
    assert!(!descriptors.covers("dtype_bf16_hidden_2816"));
    // Neither is a dimension the package does not declare.
    assert!(!descriptors.covers("dtype_bf16_page_64"));
    // Nor a key that does not split into declared dimensions at all.
    assert!(!descriptors.covers("nonsense"));
}

/// The manifest of a Triton `cache_block_scatter` package whose `descriptors`
/// line is `descriptors`: width-generic dimensions declared by constraint.
fn generic(descriptors: &str) -> String {
    format!(
        r#"
op = "cache_block_scatter"
backend = "triton"

[toolchain]
kind = "triton"
version = "3.7.1"

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/scatter.py"]
script = "src/build.py"

[kernel]
architectures = ["sm_100a"]
descriptors = {descriptors}
"#
    )
}

/// A width the kernel handles generically is declared by the constraint the
/// kernel imposes, and the package covers every key whose value meets it.
#[test]
fn a_package_covers_a_generic_dimension_by_its_constraint() {
    let text = generic(
        r#"{ dimensions = { dtype = ["bf16"], columns = { multiple = 16 }, block = { power_of_two = true, max = 8192 }, first = { min = 0 }, shape = { separator = "x" }, epsilon = { bits = 32 } } }"#,
    );
    let root = package(&text, &["src/scatter.py", "src/build.py"]);
    let manifest = Manifest::load(root.path()).unwrap();
    let key = |columns: &str, block: &str, first: &str, shape: &str, epsilon: &str| {
        format!(
            "dtype_bf16_columns_{columns}_block_{block}_first_{first}_shape_{shape}_epsilon_{epsilon}"
        )
    };

    // Any width that meets every constraint, listed nowhere.
    assert!(manifest.covers(&key("5376", "1024", "0", "2048x64x1x128", "0x358637bd")));
    assert!(manifest.covers(&key("16", "8192", "151936", "7", "0x3727c5ac")));
    // Each constraint refuses what it excludes.
    assert!(!manifest.covers(&key("5377", "1024", "0", "7", "0x358637bd")));
    assert!(!manifest.covers(&key("16", "1536", "0", "7", "0x358637bd")));
    assert!(!manifest.covers(&key("16", "16384", "0", "7", "0x358637bd")));
    assert!(!manifest.covers(&key("0", "1024", "0", "7", "0x358637bd")));
    assert!(!manifest.covers(&key("16", "1024", "0", "2048x0x1", "0x358637bd")));
    assert!(!manifest.covers(&key("16", "1024", "0", "7", "0x123456789")));
    // A key prints canonical text, so another spelling of a value names nothing.
    assert!(!manifest.covers(&key("016", "1024", "0", "7", "0x358637bd")));
    assert!(!manifest.covers(&key("16", "1024", "0", "7", "0x358637BD")));
    assert!(!manifest.covers(&key("16", "1024", "-1", "7", "0x358637bd")));
    // An enumeration beside them still admits only its values.
    assert!(!manifest.covers(&key("16", "1024", "0", "7", "0x358637bd").replace("bf16", "f16")));
}

/// A constraint that admits no value, or mixes a float with integer fields, is
/// not a statement about a kernel, and the manifest is refused.
#[test]
fn a_contradictory_constraint_is_refused() {
    for descriptors in [
        r#"{ dimensions = { columns = { multiple = 0 } } }"#,
        r#"{ dimensions = { columns = { min = 8, max = 4 } } }"#,
        r#"{ dimensions = { epsilon = { bits = 16 } } }"#,
        r#"{ dimensions = { epsilon = { bits = 32, min = 0 } } }"#,
        r#"{ dimensions = { shape = { separator = "_" } } }"#,
        r#"{ dimensions = { columns = { step = 8 } } }"#,
    ] {
        let root = package(&generic(descriptors), &["src/scatter.py", "src/build.py"]);
        assert!(
            Manifest::load(root.path()).is_err(),
            "{descriptors} is refused"
        );
    }
}

#[test]
fn a_manifest_that_names_a_file_the_package_lacks_is_refused() {
    let root = package(REASONABLE, &["src/rms.py"]);
    let refused = Manifest::load(root.path());
    assert!(
        refused.is_err(),
        "compile.py and the tuning data are missing"
    );
}

#[test]
fn a_native_module_names_a_launch_entry_and_any_lifecycle_entries() {
    let text = r#"
op = "expert_exchange"
backend = "deepep"

[toolchain]
kind = "nvcc"
version = "13.0"

[module]
form = "native"

[module.entries]
launch = "uniserve_exchange_launch"
create = "uniserve_exchange_create"
suspend = "uniserve_exchange_suspend"
restore = "uniserve_exchange_restore"

[source]
kind = "cuda"
files = ["src/exchange.cu"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] } }
"#;
    let root = package(text, &["src/exchange.cu"]);
    let manifest = Manifest::load(root.path()).unwrap();
    assert_eq!(
        manifest.module.entries.get("launch").map(String::as_str),
        Some("uniserve_exchange_launch")
    );
    assert_eq!(manifest.module.entries.len(), 4);
}

/// A native package whose launch symbol differs per descriptor: the compile
/// script writes the module's entries, and `[module.entries]` stays the
/// package's default.
const ENTRIES_FILE: &str = r#"
op = "expert_exchange"
backend = "deepep"

[toolchain]
kind = "nvcc"
version = "13.0.48"

[module]
form = "native"
entries_file = "entries.json"

[module.entries]
launch = "uniserve_exchange_launch"

[source]
kind = "cuda"
files = ["src/exchange.cu"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] } }
"#;

#[test]
fn a_native_module_may_name_a_file_its_build_writes_its_entries_into() {
    let root = package(ENTRIES_FILE, &["src/exchange.cu"]);
    let manifest = Manifest::load(root.path()).unwrap();
    assert_eq!(
        manifest.module.entries_file.as_deref(),
        Some("entries.json")
    );
    assert_eq!(
        manifest.module.entries.get("launch").map(String::as_str),
        Some("uniserve_exchange_launch"),
        "the package keeps its default launch entry"
    );
}

/// What a Triton compile writes beside its cubin: the kernel's symbol, in Triton's own field.
const METADATA: &str = "{\"name\": \"kernel\"}\n";

/// The manifest names its metadata file and the package holds a file by that name.
fn assert_names_its_metadata(manifest: &Manifest, package: &std::path::Path) {
    let name = manifest
        .module
        .metadata_name("any_descriptor")
        .expect("a Triton cubin module names its metadata file");
    assert!(
        package.join(&name).is_file(),
        "the package names {name} and does not hold it"
    );
}

/// A Triton kernel's kernel symbol and parameters are in the metadata its compile writes, so a
/// package that does not name that file serves an image the main process cannot find the kernel of.
#[test]
fn a_triton_cubin_module_must_name_its_metadata_file() {
    let text = REASONABLE.replace("metadata = \"metadata.json\"\n", "");
    let root = package(
        &text,
        &["src/rms.py", "src/compile.py", "tuning/tactics.json"],
    );
    let refused = Manifest::load(root.path()).expect_err("the rule refuses the manifest");
    assert!(
        refused
            .to_string()
            .contains("a Triton cubin module must name its metadata file"),
        "the refusal was {refused}"
    );
}

#[test]
fn a_cubin_module_may_not_name_an_entries_file() {
    let text = REASONABLE.replace(
        "metadata = \"metadata.json\"",
        "metadata = \"metadata.json\"\nentries_file = \"entries.json\"",
    );
    let root = package(
        &text,
        &["src/rms.py", "src/compile.py", "tuning/tactics.json"],
    );
    assert!(Manifest::load(root.path()).is_err());
}

#[test]
fn a_cubin_module_with_a_c_entry_is_refused() {
    let text = REASONABLE.replace(
        "metadata = \"metadata.json\"",
        "metadata = \"metadata.json\"\n[module.entries]\nlaunch = \"nope\"",
    );
    let root = package(
        &text,
        &["src/rms.py", "src/compile.py", "tuning/tactics.json"],
    );
    assert!(Manifest::load(root.path()).is_err());
}

#[test]
fn a_native_module_without_a_c_entry_is_refused() {
    let text = r#"
op = "gather_gemm"
backend = "cute"

[toolchain]
kind = "triton"
version = "4.6.2"

[module]
form = "native"

[source]
kind = "cute_dsl"
files = ["src/gemm.py"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] } }
"#;
    let root = package(text, &["src/gemm.py"]);
    assert!(Manifest::load(root.path()).is_err());
}

#[test]
fn a_manifest_without_architectures_is_refused() {
    let text = r#"
op = "rms_norm"
backend = "triton"

[toolchain]
kind = "triton"
version = "3.7.1"

[module]
form = "cubin"

[source]
kind = "triton"
files = ["src/rms.py"]

[kernel]
architectures = []
descriptors = { dimensions = { dtype = ["bf16"] } }
"#;
    let root = package(text, &["src/rms.py"]);
    assert!(Manifest::load(root.path()).is_err());
}

#[test]
fn a_manifest_of_an_unknown_source_kind_is_refused() {
    let text = REASONABLE.replace("kind = \"triton\"", "kind = \"magic\"");
    let root = package(
        &text,
        &["src/rms.py", "src/compile.py", "tuning/tactics.json"],
    );
    assert!(Manifest::load(root.path()).is_err());
}

#[test]
fn a_package_may_carry_one_file_per_descriptor() {
    let text = r#"
op = "paged_attention"
backend = "trtllm-gen"

[toolchain]
kind = "nvcc"
version = "13.0"

[module]
form = "native"

[module.entries]
launch = "uniserve_paged_attention_launch"

[source]
kind = "cubin"
files = ["cubin/{descriptor}.cubin", "cubin/{descriptor}.json"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"], head_dim = ["128"] } }
"#;
    let root = package(text, &[]);
    let manifest = Manifest::load(root.path()).expect("a per-descriptor package");

    assert!(manifest.source.per_descriptor());
    assert_eq!(
        manifest.source.files_for("dtype_bf16_head_dim_128"),
        [
            "cubin/dtype_bf16_head_dim_128.cubin",
            "cubin/dtype_bf16_head_dim_128.json"
        ]
    );
}

#[test]
fn a_scheduling_policy_is_a_wasm_package_without_metadata_or_entry() {
    // The shape a policy package has: no kernel package, a build script that
    // leaves the raw wasm, the precompiler's version, and the policy's own
    // parameters (design sections 5.2 and 5.3).
    let text = r#"
op = "scheduling"
backend = "fcfs"
tuning = ["tuning/weights.json"]

[toolchain]
kind = "wasm"
version = "50.0.1"

[module]
form = "wasm"

[source]
kind = "wasm"
files = ["src/lib.rs"]
script = "src/build.py"

[parameters]
bucket = 256
"#;
    let root = package(text, &["src/lib.rs", "tuning/weights.json"]);
    let manifest = Manifest::load(root.path()).unwrap();
    assert_eq!(manifest.module.form, ModuleForm::Wasm);
    assert!(manifest.module.metadata.is_none());
    assert!(manifest.module.entries.is_empty());
    // One module fills the whole interface, for the one target wasm has.
    assert!(manifest.covers("default"));
    assert!(manifest.builds_for("wasm32"));
    assert!(!manifest.builds_for("sm_100a"));
    assert_eq!(manifest.source.script.as_deref(), Some("src/build.py"));
    assert_eq!(
        manifest.parameters.get("bucket"),
        Some(&toml::Value::Integer(256)),
        "a policy's parameters are its own"
    );
    assert_eq!(manifest.tuning, ["tuning/weights.json"]);
}

#[test]
fn a_wasm_package_that_names_a_kernel_revision_is_refused() {
    let text = r#"
op = "scheduling"
backend = "fcfs"

[toolchain]
kind = "wasm"
version = "50.0.1"

[module]
form = "wasm"

[source]
kind = "wasm"
files = ["src/lib.rs"]

[kernel]
architectures = ["wasm32"]
descriptors = { dimensions = { mode = ["default"] } }
"#;
    let root = package(text, &["src/lib.rs"]);
    assert!(Manifest::load(root.path()).is_err());
}

#[test]
fn a_wasm_module_with_a_c_entry_is_refused() {
    let text = r#"
op = "scheduling"
backend = "fcfs"

[toolchain]
kind = "wasm"
version = "wasmtime-31"

[module]
form = "wasm"
entry = "uniserve_fcfs"

[source]
kind = "wasm"
files = ["src/lib.rs"]
"#;
    let root = package(text, &["src/lib.rs"]);
    assert!(Manifest::load(root.path()).is_err());
}

/// An executable: a package revision the main process runs as its own process,
/// stored as it is (design section 5.1).
const EXECUTABLE: &str = r#"
op = "sampler"
backend = "process"

[toolchain]
kind = "none"
version = ""

[module]
form = "executable"

[source]
kind = "executable"
files = ["bin/sampler"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] } }
"#;

#[test]
fn an_executable_package_stores_its_binary_and_compiles_nothing() {
    let root = package(EXECUTABLE, &["bin/sampler"]);
    let manifest = Manifest::load(root.path()).unwrap();
    assert_eq!(manifest.module.form, ModuleForm::Executable);
    assert_eq!(manifest.toolchain.kind, ToolchainKind::None);
    assert_eq!(
        manifest.source(root.path()).unwrap().toolchain(),
        ToolchainKind::None,
        "an executable is stored as it is: nothing compiles it"
    );
    assert!(manifest.module.metadata.is_none());
    assert!(manifest.module.entries.is_empty());
}

#[test]
fn a_package_missing_its_manifest_is_refused() {
    let root: PathBuf = tempfile::tempdir().unwrap().keep();
    assert!(Manifest::load(Path::new(&root)).is_err());
}

/// A package whose operator's key shape varies by variant declares one table per
/// shape, and covers each: an operator is served by one backend name, not by one
/// per shape.
///
/// The two shapes here are `qk_norm_rope`'s: the wide rotation, whose key has no
/// `rotated`, and the narrowed one, which does. A key of either is covered; a key
/// that parses against neither is refused exactly as it is refused today.
#[test]
fn a_package_may_declare_one_descriptor_table_per_key_shape() {
    let directory = tempfile::tempdir().unwrap();
    let package = directory.path();
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(package.join("src/build.py"), "print('built')\n").unwrap();
    std::fs::write(package.join("metadata.json"), METADATA).unwrap();
    std::fs::write(
        package.join("package.toml"),
        r#"op = "qk_norm_rope"
backend = "triton"

[toolchain]
kind = "triton"
version = "3.7.1"
packages = []

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/build.py"]
script = "src/build.py"

[kernel]
target = "arch"
architectures = ["sm_100a"]

[[kernel.descriptors]]
dimensions = { dtype = ["bf16"], head_dim = ["128"], q_heads = ["56"], rotation = ["half"] }

[[kernel.descriptors]]
dimensions = { dtype = ["bf16"], head_dim = ["128"], q_heads = ["56"], rotation = ["interleaved"], rotated = ["96"] }
"#,
    )
    .unwrap();

    let manifest = build_server::manifest::Manifest::load(package).expect("the package loads");
    assert_names_its_metadata(&manifest, package);
    assert_eq!(
        manifest.descriptors().len(),
        2,
        "both shapes are kept, in the manifest's order"
    );
    let wide = "dtype_bf16_head_dim_128_q_heads_56_rotation_half";
    let narrow = "dtype_bf16_head_dim_128_q_heads_56_rotation_interleaved_rotated_96";
    assert!(manifest.covers(wide), "the wide key is covered");
    assert!(manifest.covers(narrow), "the narrowed key is covered");
    assert!(
        !manifest.covers("dtype_bf16_head_dim_128_q_heads_56_rotation_half_rotated_96"),
        "a key that parses against neither shape is not covered"
    );
}

/// A package whose keys all have one shape writes one `[kernel.descriptors]`
/// table, and covers the keys its dimensions admit.
#[test]
fn a_package_may_declare_one_descriptor_table() {
    let directory = tempfile::tempdir().unwrap();
    let package = directory.path();
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(package.join("src/build.py"), "print('built')\n").unwrap();
    std::fs::write(package.join("metadata.json"), METADATA).unwrap();
    std::fs::write(
        package.join("package.toml"),
        r#"op = "channel_affine"
backend = "triton"

[toolchain]
kind = "triton"
version = "3.7.1"
packages = []

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/build.py"]
script = "src/build.py"

[kernel]
target = "arch"
architectures = ["sm_100a"]

[kernel.descriptors]
dimensions = { dtype = ["bf16"], channels = ["512"] }
"#,
    )
    .unwrap();

    let manifest = build_server::manifest::Manifest::load(package).expect("the package loads");
    assert_names_its_metadata(&manifest, package);
    assert_eq!(manifest.descriptors().len(), 1);
    assert!(manifest.covers("dtype_bf16_channels_512"));
    assert!(!manifest.covers("dtype_bf16_channels_256"));
}

/// The Python distributions every kernel package and package fixture of this tree declares are the
/// ones the Build Server's environment locks, by name and at the locked version: a build checks
/// them against the installed environment (`Toolchains::check_packages`), which `uv sync --locked`
/// makes the lock, so a declaration the lock disagrees with is a package no host builds.
#[test]
fn every_declared_distribution_is_the_locked_one() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lock: toml::Table =
        fs::read_to_string(repository.join("crates/build-server/python/uv.lock"))
            .expect("the Build Server's lock is in the tree")
            .parse()
            .expect("the lock is TOML");
    let normalized = |name: &str| name.to_lowercase().replace(['_', '.'], "-");
    let locked: std::collections::BTreeMap<String, String> = lock["package"]
        .as_array()
        .expect("the lock lists its packages")
        .iter()
        .map(|entry| {
            (
                normalized(entry["name"].as_str().unwrap()),
                entry["version"].as_str().unwrap().to_owned(),
            )
        })
        .collect();

    let mut packages: Vec<PathBuf> = Vec::new();
    for root in ["crates/ops/kernels", "crates/ops/package-fixtures"] {
        let mut pending = vec![repository.join(root)];
        while let Some(directory) = pending.pop() {
            if directory.join("package.toml").is_file() {
                packages.push(directory.clone());
            }
            for entry in fs::read_dir(&directory).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                }
            }
        }
    }
    assert!(!packages.is_empty(), "the tree holds kernel packages");

    let mut disagreements = Vec::new();
    for package in &packages {
        let manifest = Manifest::load(package).expect("every package in the tree loads");
        for declared in &manifest.toolchain.packages {
            match locked.get(&normalized(&declared.name)) {
                Some(version) if *version == declared.version => {}
                found => disagreements.push(format!(
                    "{} {}: {} {} (the lock has {found:?})",
                    manifest.op, manifest.backend, declared.name, declared.version
                )),
            }
        }
    }
    assert!(disagreements.is_empty(), "{disagreements:#?}");
}
