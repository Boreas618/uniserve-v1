//! The package manifest, `package.toml` (design section 5.2): what it says,
//! the descriptors it covers, and the packages it refuses.

use std::fs;
use std::path::{Path, PathBuf};

use program_server::manifest::{Manifest, ModuleForm, SourceKind, ToolchainKind};

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

[toolchain]
kind = "python"
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

tuning = ["tuning/tactics.json"]
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
    assert_eq!(manifest.toolchain.kind, ToolchainKind::Python);
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
fn a_package_covers_a_descriptor_by_dimension_values_or_by_name() {
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

#[test]
fn a_package_may_name_descriptors_the_dimensions_do_not_express() {
    let text = r#"
op = "paged_attention"
backend = "trtllm"

[toolchain]
kind = "nvcc"
version = "13.0"

[module]
form = "native"

[module.entries]
launch = "uniserve_paged_attention_launch"

[source]
kind = "cubin"
files = ["cubin/attention.cubin"]

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] }, keys = ["head_dim_128_page_64_mask"] }
"#;
    let root = package(text, &["cubin/attention.cubin"]);
    let manifest = Manifest::load(root.path()).unwrap();
    assert!(manifest.covers("head_dim_128_page_64_mask"));
    assert!(manifest.covers("dtype_bf16"));
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
kind = "python"
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
kind = "python"
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
    let text = r#"
op = "scheduling"
backend = "fcfs"

[toolchain]
kind = "wasm"
version = "wasmtime-31"

[module]
form = "wasm"

[source]
kind = "wasm"
files = ["src/lib.rs"]
"#;
    let root = package(text, &["src/lib.rs"]);
    let manifest = Manifest::load(root.path()).unwrap();
    assert_eq!(manifest.module.form, ModuleForm::Wasm);
    assert!(manifest.module.metadata.is_none());
    assert!(manifest.module.entries.is_empty());
    assert!(manifest.covers("default"));
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

#[test]
fn a_package_missing_its_manifest_is_refused() {
    let root: PathBuf = tempfile::tempdir().unwrap().keep();
    assert!(Manifest::load(Path::new(&root)).is_err());
}
