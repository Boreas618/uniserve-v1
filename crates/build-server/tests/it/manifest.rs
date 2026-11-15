//! The package manifest, `package.toml` (design section 5.2): what it says,
//! the descriptors it covers, the targets it builds for, and the packages it
//! refuses.

use std::fs;

use build_server::manifest::{Manifest, ModuleForm, SINGLE_KEY, SourceKind, ToolchainKind};
use build_server::target::Target;

use crate::support::{arch, for_arch, other_arch, write};

/// Writes a package of manifest `text`, with `{arch}` named, and of `files`
/// into a fresh directory, and returns its root.
fn package(text: &str, files: &[&str]) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    write(root.path(), &[(Manifest::FILE, for_arch(text).as_bytes())]);
    for file in files {
        write(root.path(), &[(file, b"# a source\n")]);
    }
    root
}

/// Loads the package of manifest `text` and `files`.
fn load(text: &str, files: &[&str]) -> Result<Manifest, String> {
    Manifest::load(package(text, files).path()).map_err(|error| error.to_string())
}

/// The manifest of a Triton `rms_norm` package: the shape every kernel
/// package of section 5.2 has.
const REASONABLE: &str = r#"
op = "rms_norm"
backend = "triton"
interface_version = 1
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

[kernel]
architectures = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16", "fp32"], hidden = ["2048"] } }]

[parameters]
tile = 1024
"#;

/// The files [`REASONABLE`] names.
const REASONABLE_FILES: [&str; 3] = ["src/rms.py", "src/compile.py", "tuning/tactics.json"];

#[test]
fn a_manifest_says_what_the_package_is_and_how_it_is_built() {
    let root = package(REASONABLE, &REASONABLE_FILES);
    let manifest = Manifest::load(root.path()).unwrap();

    assert_eq!(manifest.op, "rms_norm");
    assert_eq!(manifest.backend, "triton");
    assert_eq!(manifest.toolchain.kind, ToolchainKind::Triton);
    assert_eq!(manifest.toolchain.version, "3.7.1");
    assert_eq!(manifest.module.form, ModuleForm::Cubin);
    assert_eq!(manifest.source.kind, SourceKind::Triton);
    assert_eq!(manifest.source.script.as_deref(), Some("src/compile.py"));
    assert_eq!(manifest.tuning, ["tuning/tactics.json"]);
    assert_eq!(
        manifest.parameters.get("tile"),
        Some(&toml::Value::Integer(1024))
    );
    // A kernel package builds for the architectures it names, and for no other.
    assert_eq!(
        manifest.kernel.as_ref().unwrap().architectures,
        [arch().to_string()]
    );
    assert!(manifest.builds_for(arch().as_str()));
    assert!(!manifest.builds_for(other_arch().as_str()));
    assert!(!manifest.covers(SINGLE_KEY));
}

#[test]
fn a_package_covers_a_descriptor_by_its_enumerated_values() {
    let manifest = load(REASONABLE, &REASONABLE_FILES).unwrap();

    assert!(manifest.covers("dtype_bf16_hidden_2048"));
    assert!(manifest.covers("dtype_fp32_hidden_2048"));
    // A value the package does not declare is not covered.
    assert!(!manifest.covers("dtype_bf16_hidden_2816"));
    // Neither is a dimension the package does not declare.
    assert!(!manifest.covers("dtype_bf16_page_64"));
    // Nor a key that does not split into declared dimensions at all.
    assert!(!manifest.covers("nonsense"));
}

/// The manifest of a Triton `cache_block_scatter` package whose one
/// descriptor shape is `dimensions`.
fn generic(dimensions: &str) -> String {
    r#"
op = "cache_block_scatter"
backend = "triton"
interface_version = 1

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
architectures = ["{arch}"]
descriptors = [{ dimensions = DIMENSIONS }]
"#
    .replace("DIMENSIONS", dimensions)
}

/// The files [`generic`] names.
const GENERIC_FILES: [&str; 2] = ["src/scatter.py", "src/build.py"];

/// A key is read into its fields once, by the manifest: in the key's order, a
/// value that holds an underscore or another dimension's name is one value, a
/// constraint's value ends at the next underscore, and a key that does not
/// read whole has no fields.
#[test]
fn a_key_is_read_into_the_fields_a_build_script_is_handed() {
    let manifest = load(
        &generic(
            r#"{ dtype = ["bf16", "f32"], layout = ["contiguous", "block_major"], block = { power_of_two = true }, shape = { separator = "x" }, epsilon = { bits = 32 } }"#,
        ),
        &GENERIC_FILES,
    )
    .unwrap();
    let fields = |key: &str| {
        manifest.fields(key).map(|fields| {
            fields
                .into_iter()
                .map(|(dimension, value)| format!("{dimension}={value}"))
                .collect::<Vec<_>>()
        })
    };

    assert_eq!(
        fields("layout_block_major_dtype_bf16_block_64_shape_2048x64_epsilon_0x358637bd"),
        Some(vec![
            "layout=block_major".to_owned(),
            "dtype=bf16".to_owned(),
            "block=64".to_owned(),
            "shape=2048x64".to_owned(),
            "epsilon=0x358637bd".to_owned(),
        ])
    );
    // A dimension the key leaves out is absent, not refused.
    assert_eq!(fields("dtype_f32"), Some(vec!["dtype=f32".to_owned()]));
    // A value its dimension does not admit, an undeclared dimension and a dangling name.
    assert_eq!(fields("dtype_f16"), None);
    assert_eq!(fields("dtype_bf16_block_48"), None);
    assert_eq!(fields("dtype_bf16_page_64"), None);
    assert_eq!(fields("dtype_bf16_block"), None);
    assert!(!manifest.covers("dtype_bf16_block_48"));
}

/// A width the kernel handles generically is declared by the constraint the
/// kernel imposes, and the package covers every key whose value meets it.
#[test]
fn a_package_covers_a_generic_dimension_by_its_constraint() {
    let manifest = load(
        &generic(
            r#"{ dtype = ["bf16"], columns = { multiple = 16 }, block = { power_of_two = true, max = 8192 }, first = { min = 0 }, shape = { separator = "x" }, epsilon = { bits = 32 } }"#,
        ),
        &GENERIC_FILES,
    )
    .unwrap();
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
    for dimensions in [
        r#"{ columns = { multiple = 0 } }"#,
        r#"{ columns = { min = 8, max = 4 } }"#,
        r#"{ epsilon = { bits = 16 } }"#,
        r#"{ epsilon = { bits = 32, min = 0 } }"#,
        r#"{ shape = { separator = "_" } }"#,
        r#"{ columns = { step = 8 } }"#,
    ] {
        assert!(
            load(&generic(dimensions), &GENERIC_FILES).is_err(),
            "{dimensions} is refused"
        );
    }
}

/// A package whose operator's key shape varies by variant declares one table
/// per shape, and covers each: an operator is served by one backend name, not
/// by one per shape.
///
/// The two shapes here are `qk_norm_rope`'s: the wide rotation, whose key has
/// no `rotated`, and the narrowed one, which does.
#[test]
fn a_package_may_declare_one_descriptor_table_per_key_shape() {
    let manifest = load(
        r#"
op = "qk_norm_rope"
backend = "triton"
interface_version = 1

[toolchain]
kind = "triton"
version = "3.7.1"

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["src/build.py"]
script = "src/build.py"

[kernel]
architectures = ["{arch}"]

[[kernel.descriptors]]
dimensions = { dtype = ["bf16"], head_dim = ["128"], q_heads = ["56"], rotation = ["half"] }

[[kernel.descriptors]]
dimensions = { dtype = ["bf16"], head_dim = ["128"], q_heads = ["56"], rotation = ["interleaved"], rotated = ["96"] }
"#,
        &["src/build.py"],
    )
    .unwrap();

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
        "a key that reads against neither shape is not covered"
    );
}

/// A manifest states the version of the interface its package implements:
/// one that states none, or states zero, is refused, because a loader could
/// not tell what it would be loading.
#[test]
fn a_manifest_states_the_interface_version_it_implements() {
    assert_eq!(
        load(REASONABLE, &REASONABLE_FILES)
            .unwrap()
            .interface_version,
        1
    );
    let later = REASONABLE.replace("interface_version = 1", "interface_version = 4");
    assert_eq!(
        load(&later, &REASONABLE_FILES).unwrap().interface_version,
        4
    );

    let unstated = REASONABLE.replace("interface_version = 1\n", "");
    let error = load(&unstated, &REASONABLE_FILES).unwrap_err();
    assert!(error.contains("interface_version"), "{error}");

    let zero = REASONABLE.replace("interface_version = 1", "interface_version = 0");
    let error = load(&zero, &REASONABLE_FILES).unwrap_err();
    assert!(error.contains("start at 1"), "{error}");
}

/// A kernel package names the architectures it supports, each a target the
/// Build Server describes: none, or one the description lacks, is refused
/// and the refusal names it.
#[test]
fn a_kernel_package_names_architectures_the_build_server_describes() {
    let architectures = |names: &str| {
        REASONABLE.replace(
            r#"architectures = ["{arch}"]"#,
            &format!("architectures = {names}"),
        )
    };
    let both = format!(r#"["{}", "{}"]"#, arch(), other_arch());
    let manifest = load(&architectures(&both), &REASONABLE_FILES).unwrap();
    assert!(manifest.builds_for(arch().as_str()));
    assert!(manifest.builds_for(other_arch().as_str()));

    let error = load(&architectures("[]"), &REASONABLE_FILES).unwrap_err();
    assert!(error.contains("must name the architectures"), "{error}");

    let undescribed = format!(r#"["{}", "gpu_undescribed"]"#, arch());
    let error = load(&architectures(&undescribed), &REASONABLE_FILES).unwrap_err();
    assert!(
        error.contains("does not describe") && error.contains("gpu_undescribed"),
        "{error}"
    );
    assert!(Target::new("gpu_undescribed").facts().is_none());
}

/// The manifest of a CuTe DSL package, a native module its build module's
/// kernel and its C++ entry unit link into, whose `SOURCE` stands for the
/// `[source]` lines that name them.
const CUTE_DSL: &str = r#"
op = "gather_gemm"
backend = "cute"
interface_version = 1

[toolchain]
kind = "cute_dsl"
version = "4.6.2"

[module]
form = "native"

[module.entries]
launch = "uniserve_gather_gemm"

[source]
kind = "cute_dsl"
files = ["src/gemm.py", "src/entry.cpp"]
SOURCE

[kernel]
architectures = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

/// A source a build module compiles (Triton, CUDA, CuTe DSL) names the module
/// in `[source] script`, and a CuTe DSL source names its C++ entry unit too;
/// a source that compiles nothing (a vendor cubin, an executable) names no
/// build module.
#[test]
fn a_build_module_source_names_its_script_and_no_other_source_does() {
    let files = ["src/gemm.py", "src/entry.cpp"];
    let cute = |source: &str| CUTE_DSL.replace("SOURCE", source);
    let manifest = load(
        &cute("script = \"src/gemm.py\"\nunit = \"src/entry.cpp\""),
        &files,
    )
    .unwrap();
    assert_eq!(manifest.source.unit.as_deref(), Some("src/entry.cpp"));

    let error = load(&cute("unit = \"src/entry.cpp\""), &files).unwrap_err();
    assert!(error.contains("[source] script"), "{error}");
    let error = load(&cute("script = \"src/gemm.py\""), &files).unwrap_err();
    assert!(error.contains("[source] unit"), "{error}");
    let unscripted = REASONABLE.replace("script = \"src/compile.py\"\n", "");
    let error = load(&unscripted, &REASONABLE_FILES).unwrap_err();
    assert!(error.contains("[source] script"), "{error}");

    let scripted_executable = EXECUTABLE.replace(
        "files = [\"bin/sampler\"]",
        "files = [\"bin/sampler\"]\nscript = \"src/build.py\"",
    );
    let error = load(&scripted_executable, &["bin/sampler", "src/build.py"]).unwrap_err();
    assert!(error.contains("no build module"), "{error}");
    let scripted_cubin = VENDOR.replace(
        "kind = \"cubin\"",
        "kind = \"cubin\"\nscript = \"src/build.py\"",
    );
    let error = load(&scripted_cubin, &["src/build.py"]).unwrap_err();
    assert!(error.contains("no build module"), "{error}");
}

/// A vendor cubin package: one cubin and its launch metadata per descriptor,
/// which nothing compiles.
const VENDOR: &str = r#"
op = "paged_attention"
backend = "trtllm"
interface_version = 1

[toolchain]
kind = "none"
version = ""

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "cubin"
files = ["cubin/{descriptor}.cubin", "cubin/{descriptor}.json"]
cubin = "cubin/{descriptor}.cubin"
metadata = "cubin/{descriptor}.json"

[kernel]
architectures = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16"], head_dim = ["128"] } }]
"#;

/// A vendor cubin source names its cubin and its launch metadata, which may
/// be one file per descriptor: a source that leaves out either is refused,
/// because no file's extension says which role it plays.
#[test]
fn a_vendor_cubin_source_names_its_cubin_and_its_launch_metadata() {
    let manifest = load(VENDOR, &[]).expect("a per-descriptor vendor package");
    assert!(manifest.source.per_descriptor());
    assert_eq!(
        manifest.source.files_for("dtype_bf16_head_dim_128"),
        [
            "cubin/dtype_bf16_head_dim_128.cubin",
            "cubin/dtype_bf16_head_dim_128.json"
        ]
    );

    for role in [
        "cubin = \"cubin/{descriptor}.cubin\"\n",
        "metadata = \"cubin/{descriptor}.json\"\n",
    ] {
        let error = load(&VENDOR.replacen(role, "", 1), &[]).unwrap_err();
        assert!(error.contains("[source] cubin and metadata"), "{error}");
    }
}

/// A cubin module's kernel symbol and parameters are in its launch metadata,
/// so it names that file; and it has no C entry, by role or in an entries
/// file.
#[test]
fn a_cubin_module_names_its_metadata_and_no_c_entry() {
    let unnamed = REASONABLE.replace("metadata = \"metadata.json\"\n", "");
    let error = load(&unnamed, &REASONABLE_FILES).unwrap_err();
    assert!(
        error.contains("a cubin module must name its metadata file"),
        "{error}"
    );

    for entries in [
        "metadata = \"metadata.json\"\nentries_file = \"entries.json\"",
        "metadata = \"metadata.json\"\n[module.entries]\nlaunch = \"nope\"",
    ] {
        let text = REASONABLE.replace("metadata = \"metadata.json\"", entries);
        let error = load(&text, &REASONABLE_FILES).unwrap_err();
        assert!(
            error.contains("only a native module has C entries"),
            "{error}"
        );
    }
}

/// A native package of an expert exchange: a launch entry and the lifecycle
/// entries around it, and a file its build writes the module's own entries
/// into.
const NATIVE: &str = r#"
op = "expert_exchange"
backend = "deepep"
interface_version = 1

[toolchain]
kind = "nvcc"
version = "13.0.48"

[module]
form = "native"
entries_file = "entries.json"

[module.entries]
launch = "uniserve_exchange_launch"
create = "uniserve_exchange_create"
suspend = "uniserve_exchange_suspend"
restore = "uniserve_exchange_restore"

[source]
kind = "cuda"
files = ["src/exchange.cu"]
script = "src/build.py"

[kernel]
architectures = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

/// A native module names the entry that launches its kernel and any
/// lifecycle entries with the same calling convention, and may name a file its
/// build writes the module's own entries into; one that names no launch entry
/// is refused.
#[test]
fn a_native_module_names_its_launch_entry_and_any_lifecycle_entries() {
    let files = ["src/exchange.cu", "src/build.py"];
    let manifest = load(NATIVE, &files).unwrap();
    assert_eq!(
        manifest.module.entries.get("launch").map(String::as_str),
        Some("uniserve_exchange_launch")
    );
    assert_eq!(manifest.module.entries.len(), 4);
    assert_eq!(
        manifest.module.entries_file.as_deref(),
        Some("entries.json")
    );

    let unlaunched = NATIVE.replace("launch = \"uniserve_exchange_launch\"\n", "");
    let error = load(&unlaunched, &files).unwrap_err();
    assert!(error.contains("launch entry"), "{error}");
}

/// The manifest of an engine package with the `[source]` lines `source` and
/// the `kernel` section.
fn cargo(source: &str, kernel: &str) -> String {
    format!(
        r#"
op = "engine"
backend = "uniserve"
interface_version = 1

[toolchain]
kind = "cargo"
version = "1.99.1"

[module]
form = "executable"

[source]
kind = "cargo"
files = []
{source}
{kernel}
"#
    )
}

/// A cargo package names the executable it builds and no kernel: it builds
/// one module, under the one key of a package that fills a whole interface,
/// for the host's platform.
#[test]
fn a_cargo_package_names_its_binary_and_no_kernel() {
    let binary = r#"binary = "uniserve-engine""#;
    let manifest = load(&cargo(binary, ""), &[]).unwrap();
    assert_eq!(manifest.toolchain.kind, ToolchainKind::Cargo);
    assert_eq!(manifest.source.binary.as_deref(), Some("uniserve-engine"));
    assert!(manifest.covers(SINGLE_KEY));
    assert!(!manifest.covers("dtype_bf16"));
    assert_eq!(manifest.single_target(), Target::host());
    assert!(manifest.builds_for(Target::host().as_str()));
    assert!(!manifest.builds_for(arch().as_str()));

    let error = load(&cargo("", ""), &[]).unwrap_err();
    assert!(error.contains("[source] binary"), "{error}");

    let kernel = "[kernel]\narchitectures = [\"{arch}\"]\ndescriptors = [{ dimensions = { dtype = [\"bf16\"] } }]";
    let error = load(&cargo(binary, kernel), &[]).unwrap_err();
    assert!(error.contains("no [kernel]"), "{error}");
}

/// A scheduling policy: a crate cargo builds for wasm32, which fills the
/// whole policy interface with one module (design sections 5.2 and 5.3).
const POLICY: &str = r#"
op = "scheduling"
backend = "fcfs"
interface_version = 1

[toolchain]
kind = "cargo"
version = "1.99.1"

[module]
form = "wasm"

[source]
kind = "wasm"
files = []

[parameters]
bucket = 256
"#;

/// A policy has no kernel package: one module, under the one key, for
/// wasm32, with neither launch metadata nor a C entry, and its own
/// parameters.
#[test]
fn a_scheduling_policy_is_a_wasm_package_of_one_module() {
    let manifest = load(POLICY, &[]).unwrap();
    assert_eq!(manifest.module.form, ModuleForm::Wasm);
    assert_eq!(manifest.toolchain.kind, ToolchainKind::Cargo);
    assert!(manifest.module.metadata.is_none());
    assert!(manifest.module.entries.is_empty());
    assert!(manifest.covers(SINGLE_KEY));
    assert_eq!(manifest.single_target(), Target::wasm());
    assert!(manifest.builds_for(Target::wasm().as_str()));
    assert!(!manifest.builds_for(arch().as_str()));
    assert_eq!(
        manifest.parameters.get("bucket"),
        Some(&toml::Value::Integer(256)),
        "a policy's parameters are its own"
    );

    let kernel = format!(
        "{POLICY}\n[kernel]\narchitectures = [\"{{arch}}\"]\ndescriptors = [{{ dimensions = {{ mode = [\"default\"] }} }}]\n"
    );
    let error = load(&kernel, &[]).unwrap_err();
    assert!(error.contains("names no kernel package"), "{error}");

    let entry = POLICY.replace(
        "form = \"wasm\"",
        "form = \"wasm\"\n\n[module.entries]\nlaunch = \"uniserve_fcfs\"",
    );
    let error = load(&entry, &[]).unwrap_err();
    assert!(
        error.contains("only a native module has C entries"),
        "{error}"
    );
}

/// An executable: a package revision the main process runs as its own
/// process, stored as it is (design section 5.1).
const EXECUTABLE: &str = r#"
op = "sampler"
backend = "process"
interface_version = 1

[toolchain]
kind = "none"
version = ""

[module]
form = "executable"

[source]
kind = "executable"
files = ["bin/sampler"]

[kernel]
architectures = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

#[test]
fn an_executable_package_stores_its_binary_and_compiles_nothing() {
    let manifest = load(EXECUTABLE, &["bin/sampler"]).unwrap();
    assert_eq!(manifest.module.form, ModuleForm::Executable);
    assert_eq!(manifest.toolchain.kind, ToolchainKind::None);
    assert_eq!(manifest.source.kind, SourceKind::Executable);
    assert!(manifest.module.metadata.is_none());
    assert!(manifest.module.entries.is_empty());
}

#[test]
fn a_manifest_that_names_a_file_the_package_lacks_is_refused() {
    let error = load(REASONABLE, &["src/rms.py"]).unwrap_err();
    assert!(
        error.contains("src/compile.py") && error.contains("tuning/tactics.json"),
        "the refusal names the build module and the tuning data: {error}"
    );
}

#[test]
fn a_manifest_of_an_unknown_source_kind_is_refused() {
    let text = REASONABLE.replace("[source]\nkind = \"triton\"", "[source]\nkind = \"magic\"");
    assert!(load(&text, &REASONABLE_FILES).is_err());
}

#[test]
fn a_package_missing_its_manifest_is_refused() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("kernel.py"), b"# a source\n").unwrap();
    assert!(Manifest::load(root.path()).is_err());
}
