//! The package manifest, `package.toml` (design section 5.2): what it says,
//! what its source kind fixes when it says nothing, the descriptors it covers,
//! the targets it builds for, and the packages it refuses.

use std::fs;

use build_server::manifest::{Manifest, ModuleForm, SINGLE_KEY, SourceKind, ToolchainKind};
use build_server::target::{Feature, Requirement, Target};

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
/// package of section 5.2 has, stating only what is the package's own.
const REASONABLE: &str = r#"
op = "rms_norm"
backend = "triton"
interface_version = 1
tuning = ["tuning/tactics.json"]

[source]
kind = "triton"

[launch]
kernel = "_rms_norm_kernel"
warps = { field = "hidden", steps = [[2048, 8]], otherwise = 4 }

[launch.types]
x = { pointer = "dtype" }

[launch.constants]
HIDDEN = { field = "hidden" }

[kernel]
requires = ["{arch}"]
descriptors = [{ dimensions = { dtype = ["bf16", "fp32"], hidden = ["2048"] } }]

[parameters]
tile = 1024
"#;

/// The files [`REASONABLE`] needs: its kernel module and its tuning data.
const REASONABLE_FILES: [&str; 2] = ["src/rms_norm.py", "tuning/tactics.json"];

#[test]
fn a_manifest_says_what_the_package_is_and_how_it_is_built() {
    let root = package(REASONABLE, &REASONABLE_FILES);
    let manifest = Manifest::load(root.path()).unwrap();

    assert_eq!(manifest.op, "rms_norm");
    assert_eq!(manifest.backend, "triton");
    // The source kind fixes the toolchain, the module's form and its metadata file.
    assert_eq!(manifest.toolchain.kind, ToolchainKind::Triton);
    assert_eq!(manifest.module.form, ModuleForm::Cubin);
    assert_eq!(manifest.module.metadata.as_deref(), Some("metadata.json"));
    assert_eq!(manifest.source.kind, SourceKind::Triton);
    assert!(manifest.launch.is_some());
    assert_eq!(manifest.tuning, ["tuning/tactics.json"]);
    assert_eq!(
        manifest.parameters.get("tile"),
        Some(&toml::Value::Integer(1024))
    );
    // A kernel package builds for the targets that meet its requirements, and for no other.
    assert_eq!(
        manifest.kernel.as_ref().unwrap().requires,
        [Requirement::Family(arch().to_string())]
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

[source]
kind = "triton"

[launch]
kernel = "_scatter_kernel"
warps = 4

[kernel]
descriptors = [{ dimensions = DIMENSIONS }]
"#
    .replace("DIMENSIONS", dimensions)
}

/// The kernel module [`generic`] needs.
const GENERIC_FILES: [&str; 1] = ["src/cache_block_scatter.py"];

/// A key is read into its fields once, by the manifest: in the key's order, a
/// value that holds an underscore or another dimension's name is one value, a
/// constraint's value ends at the next underscore, and a key that does not
/// read whole has no fields.
#[test]
fn a_key_is_read_into_the_fields_a_compilation_is_handed() {
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

    // A compilation is handed each value typed as its dimension declares it: an integer, an
    // extent tuple, a float of the bit pattern, a name.
    assert_eq!(
        manifest.typed_fields(
            "layout_block_major_dtype_bf16_block_64_shape_2048x64_epsilon_0x358637bd"
        ),
        Some(
            serde_json::json!({
                "layout": "block_major",
                "dtype": "bf16",
                "block": 64,
                "shape": [2048, 64],
                "epsilon": f64::from(f32::from_bits(0x3586_37bd)),
            })
            .as_object()
            .unwrap()
            .clone()
        )
    );
}

/// An enumeration of `true` and `false` is a flag, an enumeration of integers holds the
/// integers a kernel is specialized on and an enumeration of extent tuples the shapes it is
/// written for, so a compilation is handed a boolean, a number and a list of numbers; any other
/// enumeration's value is its name.
#[test]
fn a_flag_an_enumerated_integer_and_an_enumerated_shape_are_typed() {
    let manifest = load(
        &generic(
            r#"{ pdl = ["true", "false"], head_dim = ["64", "128"], window = ["false"], mode = ["0", "fused"], tile = ["128x128", "256x128"], layout = ["128x4", "swizzled_128x4"] }"#,
        ),
        &GENERIC_FILES,
    )
    .unwrap();
    assert_eq!(
        manifest
            .typed_fields("pdl_true_head_dim_128_window_false_mode_0_tile_256x128_layout_128x4"),
        Some(
            serde_json::json!({
                "pdl": true,
                "head_dim": 128,
                "window": false,
                "mode": "0",
                "tile": [256, 128],
                "layout": "128x4",
            })
            .as_object()
            .unwrap()
            .clone()
        )
    );
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

[source]
kind = "triton"

[launch]
kernel = "_qk_norm_rope_kernel"
warps = 4

[[kernel.descriptors]]
dimensions = { dtype = ["bf16"], head_dim = ["128"], q_heads = ["56"], rotation = ["half"] }

[[kernel.descriptors]]
dimensions = { dtype = ["bf16"], head_dim = ["128"], q_heads = ["56"], rotation = ["interleaved"], rotated = ["96"] }
"#,
        &["src/qk_norm_rope.py"],
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
    let stated = |text: &str| {
        load(text, &REASONABLE_FILES)
            .unwrap()
            .interface_version
            .map(std::num::NonZeroU32::get)
    };
    assert_eq!(stated(REASONABLE), Some(1));
    let later = REASONABLE.replace("interface_version = 1", "interface_version = 4");
    assert_eq!(stated(&later), Some(4));

    let unstated = REASONABLE.replace("interface_version = 1\n", "");
    let error = load(&unstated, &REASONABLE_FILES).unwrap_err();
    assert!(error.contains("interface_version"), "{error}");

    let zero = REASONABLE.replace("interface_version = 1", "interface_version = 0");
    assert!(load(&zero, &REASONABLE_FILES).is_err());
}

/// A kernel package builds for every target of the Build Server's description that meets its
/// requirements: a package that requires nothing builds for every one, a hardware feature
/// selects the targets that have it, and a requirement the description cannot meet (no feature
/// and no family of it) is refused by name.
#[test]
fn a_kernel_package_builds_for_the_targets_that_meet_its_requirements() {
    let requiring = |names: &str| {
        REASONABLE.replace(r#"requires = ["{arch}"]"#, &format!("requires = {names}"))
    };
    let unrestricted = load(&requiring("[]"), &REASONABLE_FILES).unwrap();
    assert!(unrestricted.builds_for(arch().as_str()));
    assert!(unrestricted.builds_for(other_arch().as_str()));
    assert!(!unrestricted.builds_for("gpu_undescribed"));

    // Tensor memory is a fact of the Blackwell target and not of Hopper's.
    let tensor = load(&requiring(r#"["tensor_memory"]"#), &REASONABLE_FILES).unwrap();
    assert_eq!(
        tensor.kernel.as_ref().unwrap().requires,
        [Requirement::Feature(Feature::TensorMemory)]
    );
    assert!(tensor.builds_for(arch().as_str()));
    assert!(!tensor.builds_for(other_arch().as_str()));

    let error = load(&requiring(r#"["gpu_undescribed"]"#), &REASONABLE_FILES).unwrap_err();
    assert!(error.contains("gpu_undescribed"), "{error}");
    assert!(Target::new("gpu_undescribed").facts().is_none());
}

/// A field the launch table reads is a dimension some descriptor declares, unless a derivation
/// gives it; a table that reads another, or names no kernel, is refused.
#[test]
fn a_launch_table_reads_the_fields_its_descriptors_declare() {
    let error = load(
        &REASONABLE.replace(
            r#"HIDDEN = { field = "hidden" }"#,
            r#"HIDDEN = { field = "width" }"#,
        ),
        &REASONABLE_FILES,
    )
    .unwrap_err();
    assert!(error.contains("width"), "{error}");

    let derived = REASONABLE
        .replace(
            r#"HIDDEN = { field = "hidden" }"#,
            r#"HIDDEN = { field = "width" }"#,
        )
        .replace("warps = {", "derive = \"derive\"\nwarps = {");
    assert!(load(&derived, &REASONABLE_FILES).is_ok());

    let unlaunched = REASONABLE.replace("[launch]\n", "[unlaunched]\n");
    assert!(load(&unlaunched, &REASONABLE_FILES).is_err());
    let untabled = REASONABLE.split("[launch]").next().unwrap().to_owned()
        + "[kernel]\ndescriptors = [{ dimensions = { dtype = [\"bf16\"] } }]\n";
    let error = load(&untabled, &REASONABLE_FILES).unwrap_err();
    assert!(error.contains("[launch]"), "{error}");
}

/// The manifest of a CuTe DSL package, a native module its build module's
/// kernel and its C++ entry unit link into, whose `SOURCE` stands for the
/// further `[source]` lines.
const CUTE_DSL: &str = r#"
op = "gather_gemm"
backend = "cute"
interface_version = 1

[source]
kind = "cute_dsl"
SOURCE

[kernel]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

/// A CuTe DSL or CUDA source is compiled by its build module `src/build.py`, and a CuTe DSL
/// module's C entries are defined by its entry unit, `src/entry.cpp` unless it names another;
/// its launch entry is the operator's `uniserve_<op>`. A package that lacks the files its source
/// kind names is refused, and no kind but an executable names files of its own.
#[test]
fn a_build_module_source_compiles_its_build_module_and_entry_unit() {
    let cute = |source: &str| CUTE_DSL.replace("SOURCE", source);
    let manifest = load(&cute(""), &["src/build.py", "src/entry.cpp"]).unwrap();
    assert_eq!(manifest.source.unit.as_deref(), Some("src/entry.cpp"));
    assert_eq!(manifest.toolchain.kind, ToolchainKind::CuteDsl);
    assert_eq!(manifest.module.form, ModuleForm::Native);
    assert_eq!(
        manifest.module.entries.get("launch").map(String::as_str),
        Some("uniserve_gather_gemm")
    );

    let named = load(
        &cute("unit = \"src/gemm_entry.cpp\""),
        &["src/build.py", "src/gemm_entry.cpp"],
    )
    .unwrap();
    assert_eq!(named.source.unit.as_deref(), Some("src/gemm_entry.cpp"));

    let error = load(&cute(""), &["src/entry.cpp"]).unwrap_err();
    assert!(error.contains("src/build.py"), "{error}");
    let error = load(&cute(""), &["src/build.py"]).unwrap_err();
    assert!(error.contains("src/entry.cpp"), "{error}");
    let error = load(
        &cute("files = [\"src/gemm.py\"]"),
        &["src/build.py", "src/entry.cpp", "src/gemm.py"],
    )
    .unwrap_err();
    assert!(error.contains("names no [source] files"), "{error}");

    let united_cubin = VENDOR.replace(
        "kind = \"cubin\"",
        "kind = \"cubin\"\nunit = \"src/entry.cpp\"",
    );
    let error = load(&united_cubin, &["src/entry.cpp"]).unwrap_err();
    assert!(error.contains("no C++ entry unit"), "{error}");
}

/// A vendor cubin package: one cubin and its launch metadata per descriptor,
/// which nothing compiles.
const VENDOR: &str = r#"
op = "paged_attention"
backend = "trtllm"
interface_version = 1

[source]
kind = "cubin"
cubin = "cubin/{descriptor}.cubin"
metadata = "cubin/{descriptor}.json"

[kernel]
descriptors = [{ dimensions = { dtype = ["bf16"], head_dim = ["128"] } }]
"#;

/// A vendor cubin source names its cubin and its launch metadata, which may
/// be one file per descriptor: a source that leaves out either is refused,
/// because no file's extension says which role it plays.
#[test]
fn a_vendor_cubin_source_names_its_cubin_and_its_launch_metadata() {
    let manifest = load(VENDOR, &[]).expect("a per-descriptor vendor package");
    assert_eq!(manifest.toolchain.kind, ToolchainKind::None);
    assert_eq!(manifest.module.metadata.as_deref(), Some("metadata.json"));

    for role in [
        "cubin = \"cubin/{descriptor}.cubin\"\n",
        "metadata = \"cubin/{descriptor}.json\"\n",
    ] {
        let error = load(&VENDOR.replacen(role, "", 1), &[]).unwrap_err();
        assert!(error.contains("[source] cubin and metadata"), "{error}");
    }
}

/// A cubin module's kernel symbol and parameters are in its launch metadata,
/// `metadata.json` unless the manifest names another file; and it has no C
/// entry, by role or in an entries file.
#[test]
fn a_cubin_module_names_its_metadata_and_no_c_entry() {
    let named = REASONABLE.replace(
        "[source]\n",
        "[module]\nmetadata = \"launch.json\"\n\n[source]\n",
    );
    let manifest = load(&named, &REASONABLE_FILES).unwrap();
    assert_eq!(manifest.module.metadata.as_deref(), Some("launch.json"));

    for entries in [
        "entries_file = \"entries.json\"",
        "[module.entries]\nlaunch = \"nope\"",
    ] {
        let text = REASONABLE.replace("[source]\n", &format!("[module]\n{entries}\n\n[source]\n"));
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

[module]
entries_file = "entries.json"

[module.entries]
launch = "uniserve_exchange_launch"
create = "uniserve_exchange_create"
suspend = "uniserve_exchange_suspend"
restore = "uniserve_exchange_restore"

[source]
kind = "cuda"

[kernel]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

/// A native module launches through its operator's `uniserve_<op>` unless it names another
/// entry, and names any lifecycle entries with the same calling convention, and may name a file
/// its build writes the module's own entries into.
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
    let manifest = load(&unlaunched, &files).unwrap();
    assert_eq!(
        manifest.module.entries.get("launch").map(String::as_str),
        Some("uniserve_expert_exchange"),
        "the operator's own entry"
    );
}

/// A CUDA package that declares its compilation: a native module of two
/// units whose definitions read the descriptor's fields (one as the generated
/// ABI header's enumerator it names) and a parameter, and which links
/// cuBLASLt.
const DECLARED: &str = r#"
op = "dense_attention"
backend = "f32"
interface_version = 1

[source]
kind = "cuda"

[compile]
units = ["src/entry.cpp", "src/dense.cu"]
standard = "c++17"
libraries = ["cublasLt"]
host_contraction = false

[compile.definitions]
UNISERVE_MASK = { field = "mask", enumeration = "uniserve_mask" }
UNISERVE_SEQUENCES = { field = "cu_seq_lens", absent = true }
UNISERVE_TILE = { parameter = "tile" }

[[kernel.descriptors]]
dimensions = { mask = ["causal", "dense"] }

[[kernel.descriptors]]
dimensions = { mask = ["causal", "dense"], cu_seq_lens = ["false"] }

[parameters]
tile = 64
"#;

/// The units [`DECLARED`] compiles.
const DECLARED_FILES: [&str; 2] = ["src/entry.cpp", "src/dense.cu"];

/// A CUDA package may declare its compilation instead of writing a build module: its units, the
/// standard, the definitions bound to literals, fields, the enumerators fields name and
/// parameters, the libraries and its own options, which the manifest types; it then has no build
/// module, and its native module still launches through its operator's entry.
#[test]
fn a_cuda_package_declares_its_compilation() {
    use build_server::compile::{Definition, Library, Standard};

    let manifest = load(DECLARED, &DECLARED_FILES).unwrap();
    let compile = manifest.compile.as_ref().unwrap();
    assert_eq!(compile.units, DECLARED_FILES);
    assert_eq!(compile.standard, Standard::Cxx17);
    assert_eq!(compile.libraries, [Library::CublasLt]);
    assert!(!compile.host_contraction);
    assert_eq!(compile.definitions.len(), 3);
    assert_eq!(
        compile.definitions["UNISERVE_MASK"],
        Definition::Enumerator {
            field: "mask".into(),
            enumeration: "uniserve_mask".into()
        }
    );
    assert!(compile.kernel.is_none());
    assert!(!manifest.has_build_module());
    assert_eq!(manifest.toolchain.kind, ToolchainKind::Nvcc);
    assert_eq!(manifest.module.form, ModuleForm::Native);
    assert_eq!(
        manifest.module.entries.get("launch").map(String::as_str),
        Some("uniserve_dense_attention")
    );

    // The host compiler's contraction is the default.
    let contracted = load(
        &DECLARED.replace("host_contraction = false\n", ""),
        &DECLARED_FILES,
    )
    .unwrap();
    assert!(contracted.compile.unwrap().host_contraction);
}

/// A declared compilation is refused when it reads a field no descriptor declares or a
/// parameter the manifest lacks, names a unit the package lacks, a library the toolchain
/// description does not state or no C identifier, sits beside a build module, or belongs to
/// another source kind.
#[test]
fn a_declared_compilation_is_checked_against_its_package() {
    let refused = |text: &str, files: &[&str], reason: &str| {
        let error = load(text, files).unwrap_err();
        assert!(error.contains(reason), "{reason}: {error}");
    };
    refused(
        &DECLARED.replace(r#"field = "mask""#, r#"field = "window""#),
        &DECLARED_FILES,
        "window",
    );
    refused(
        &DECLARED.replace(r#"parameter = "tile""#, r#"parameter = "rows""#),
        &DECLARED_FILES,
        "rows",
    );
    refused(DECLARED, &DECLARED_FILES[..1], "src/dense.cu");
    refused(
        DECLARED,
        &["src/entry.cpp", "src/dense.cu", "src/build.py"],
        "src/build.py",
    );
    refused(
        &DECLARED.replace("UNISERVE_TILE =", "\"2TILE\" ="),
        &DECLARED_FILES,
        "no C identifier",
    );
    refused(
        &DECLARED.replace("\"uniserve_mask\"", "\"uniserve mask\""),
        &DECLARED_FILES,
        "no C identifier",
    );
    assert!(load(&DECLARED.replace("cublasLt", "cufft"), &DECLARED_FILES).is_err());
    assert!(load(&DECLARED.replace("c++17", "c++98"), &DECLARED_FILES).is_err());
    refused(
        &DECLARED.replace("kind = \"cuda\"", "kind = \"cute_dsl\""),
        &["src/entry.cpp", "src/dense.cu", "src/build.py"],
        "only a CUDA source declares its compilation",
    );
}

/// A declared compilation includes header sets of the locked distributions its package names in
/// `[toolchain] packages`, each a relative directory of the distribution's installation, and
/// states relaxed constexpr for units that include CuTe or CUTLASS; a header set of a
/// distribution the package does not name, or a path leaving the distribution's root, is
/// refused.
#[test]
fn a_declared_compilation_includes_its_distributions_headers() {
    let headers = r#"headers = [{ distribution = "torch", path = "torch/include" }]
relaxed_constexpr = true
host_contraction = false"#;
    let declaring = DECLARED.replace("host_contraction = false", headers);
    let named = declaring.replacen(
        "[source]",
        "[toolchain]\npackages = [\"torch\"]\n\n[source]",
        1,
    );

    let manifest = load(&named, &DECLARED_FILES).unwrap();
    let compile = manifest.compile.as_ref().unwrap();
    assert_eq!(compile.headers.len(), 1);
    assert_eq!(compile.headers[0].distribution, "torch");
    assert_eq!(compile.headers[0].path, "torch/include");
    assert!(compile.relaxed_constexpr);
    assert!(
        !load(DECLARED, &DECLARED_FILES)
            .unwrap()
            .compile
            .unwrap()
            .relaxed_constexpr
    );

    let error = load(&declaring, &DECLARED_FILES).unwrap_err();
    assert!(error.contains("torch"), "{error}");
    let error = load(
        &named.replace("torch/include", "../include"),
        &DECLARED_FILES,
    )
    .unwrap_err();
    assert!(error.contains("relative directory"), "{error}");
}

/// A cubin module whose unit defines several kernels states the one each descriptor's field
/// selects; the selecting field is a declared dimension, and every selected kernel is checked.
#[test]
fn a_declared_cubin_selects_its_kernel_by_a_field() {
    let selecting = r#"
op = "range_marker"
backend = "nvcc"
interface_version = 1

[module]
form = "cubin"

[source]
kind = "cuda"

[compile]
units = ["src/range_marker.cu"]
standard = "c++20"

[compile.kernel]
field = "function"

[compile.kernel.values.stamp]
name = "uniserve_range_marker"
warps = 1
signature = [["cell", "*u32"], ["mark", "u32"]]

[compile.kernel.values.clock]
name = "uniserve_clock"
warps = 1
signature = [["out", "*u64"]]

[[kernel.descriptors]]
dimensions = { function = ["stamp", "clock"] }
"#;
    let files = ["src/range_marker.cu"];
    let manifest = load(selecting, &files).unwrap();
    let kernels = manifest.compile.as_ref().unwrap().kernel.as_ref().unwrap();
    assert_eq!(kernels.field(), Some("function"));
    let mut names: Vec<&str> = kernels
        .each()
        .into_iter()
        .map(|kernel| kernel.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["uniserve_clock", "uniserve_range_marker"]);

    let error = load(
        &selecting.replace(r#"field = "function""#, r#"field = "role""#),
        &files,
    )
    .unwrap_err();
    assert!(error.contains("role"), "{error}");
    let error = load(
        &selecting.replace(r#"name = "uniserve_clock""#, r#"name = "2clock""#),
        &files,
    )
    .unwrap_err();
    assert!(error.contains("no C identifier"), "{error}");
}

/// A cubin module's declared compilation states its kernel: its symbol, its warps and its
/// signature in parameter order, from one unit and linking nothing; a native module's states no
/// kernel.
#[test]
fn a_declared_cubin_states_its_kernel() {
    let cubin = r#"
op = "topk_softmax"
backend = "nvcc"
interface_version = 1

[module]
form = "cubin"

[source]
kind = "cuda"

[compile]
units = ["src/topk_softmax.cu"]
standard = "c++20"
definitions = { UNISERVE_EXPERTS = { field = "experts" }, UNISERVE_WARPS = { parameter = "warps" } }

[compile.kernel]
name = "uniserve_topk_softmax"
warps = { parameter = "warps" }
signature = [["scores", { pointer = "scores" }], ["rows", "i32"]]

[[kernel.descriptors]]
dimensions = { scores = ["bf16"], experts = { min = 1 } }

[parameters]
warps = 4
"#;
    let files = ["src/topk_softmax.cu"];
    let manifest = load(cubin, &files).unwrap();
    let kernels = manifest.compile.as_ref().unwrap().kernel.as_ref().unwrap();
    let build_server::compile::Kernels::One(kernel) = kernels else {
        panic!("one kernel is stated: {kernels:?}");
    };
    assert_eq!(kernel.name, "uniserve_topk_softmax");
    let names: Vec<&str> = kernel
        .signature
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, ["scores", "rows"], "the kernel's parameter order");
    assert_eq!(manifest.module.metadata.as_deref(), Some("metadata.json"));

    let refused = |text: &str, reason: &str| {
        let error = load(text, &files).unwrap_err();
        assert!(error.contains(reason), "{reason}: {error}");
    };
    refused(
        &cubin.replace(r#"pointer = "scores""#, r#"pointer = "dtype""#),
        "dtype",
    );
    refused(
        &cubin.replace("[module]\nform = \"cubin\"\n", ""),
        "[compile.kernel]",
    );
    refused(
        &cubin.replace(r#"["rows", "i32"]"#, r#"["scores", "i32"]"#),
        "twice",
    );
    refused(
        &cubin.replace(
            "standard = \"c++20\"",
            "standard = \"c++20\"\nlibraries = [\"cuda\"]",
        ),
        "links no library",
    );
    let uncompiled = cubin.replace(
        r#"units = ["src/topk_softmax.cu"]"#,
        r#"units = ["src/topk_softmax.cu", "src/other.cu"]"#,
    );
    let error = load(&uncompiled, &["src/topk_softmax.cu", "src/other.cu"]).unwrap_err();
    assert!(error.contains("one translation unit"), "{error}");
}

/// The manifest of an engine package with the `[source]` lines `source` and
/// the `kernel` section: it names the vocabulary of the operators it launches,
/// and states neither its interface version, which its executable states, nor
/// its compiler release, which its workspace's `rust-toolchain.toml` selects.
fn cargo(source: &str, kernel: &str) -> String {
    format!(
        r#"
op = "engine"
backend = "uniserve"
operators = "crates/proto/src/module/operators.toml"

[source]
kind = "cargo"
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

    let kernel = "[kernel]\ndescriptors = [{ dimensions = { dtype = [\"bf16\"] } }]";
    let error = load(&cargo(binary, kernel), &[]).unwrap_err();
    assert!(error.contains("no [kernel]"), "{error}");
}

/// A cargo package states its interface version and its compiler release
/// once each, where they are owned: its executable states the interface it
/// implements and its workspace's `rust-toolchain.toml` the release, so a
/// manifest that states either is refused rather than kept as a second copy.
#[test]
fn a_cargo_package_states_no_interface_version_and_no_compiler_release() {
    let binary = r#"binary = "uniserve-engine""#;
    let manifest = load(&cargo(binary, ""), &[]).unwrap();
    assert_eq!(manifest.interface_version, None);

    let versioned = cargo(binary, "").replace(
        "op = \"engine\"\n",
        "op = \"engine\"\ninterface_version = 9\n",
    );
    let error = load(&versioned, &[]).unwrap_err();
    assert!(error.contains("interface_version"), "{error}");

    let released = format!("{}\n[toolchain]\nversion = \"1.99.1\"\n", cargo(binary, ""));
    let error = load(&released, &[]).unwrap_err();
    assert!(error.contains("version"), "{error}");
}

/// The engine names the vocabulary of the operators it launches, a file inside its snapshot,
/// and no other package names one: the operators an epoch fixes kernel packages for are the
/// engine revision's.
#[test]
fn only_an_engine_package_names_its_operator_vocabulary_inside_its_workspace() {
    let binary = r#"binary = "uniserve-engine""#;
    let manifest = load(&cargo(binary, ""), &[]).unwrap();
    assert_eq!(
        manifest.operators.as_deref(),
        Some("crates/proto/src/module/operators.toml")
    );

    let stated = r#"operators = "crates/proto/src/module/operators.toml""#;
    let unstated = cargo(binary, "").replace(&format!("{stated}\n"), "");
    let error = load(&unstated, &[]).unwrap_err();
    assert!(error.contains("`operators`"), "{error}");
    for outside in ["../operators.toml", "/operators.toml", ""] {
        let named = cargo(binary, "").replace(stated, &format!("operators = {outside:?}"));
        let error = load(&named, &[]).unwrap_err();
        assert!(error.contains("inside the package"), "{error}");
    }

    let kernel = REASONABLE.replace(
        "backend = \"triton\"\n",
        &format!("backend = \"triton\"\n{stated}\n"),
    );
    let error = load(&kernel, &["src/rms_norm.py", "tuning/tactics.json"]).unwrap_err();
    assert!(error.contains("only an engine package"), "{error}");
}

/// Every engine package of the tree, the engine and the sanitizer suite's `memcheck` package
/// that runs it, names the vocabulary the tree's engine is compiled with, as a file its snapshot
/// holds: an installation of either lists exactly the operators the executable launches, each at
/// the version of its interface the executable launches.
#[test]
fn the_tree_s_engine_packages_name_the_vocabulary_the_engine_is_compiled_with() {
    let compiled: Vec<build_server::manifest::Operator> = proto::module::operator::ALL
        .iter()
        .map(|name| build_server::manifest::Operator {
            name: (*name).to_owned(),
            interface_version: Some(proto::module::operator::version(name)),
        })
        .collect();

    let repository = crate::support::repository();
    let engine = Manifest::load(&repository.join("crates/engine")).unwrap();
    let path = repository.join(engine.operators.as_deref().unwrap());
    assert_eq!(
        build_server::manifest::read_operators(&path).unwrap(),
        compiled
    );

    let sanitizer = repository.join("tests/python/sanitizer");
    let memcheck = Manifest::load(&sanitizer).unwrap();
    let path = sanitizer.join(memcheck.operators_file().unwrap());
    assert_eq!(
        build_server::manifest::read_operators(&path).unwrap(),
        compiled
    );
}

/// A vocabulary names each operator once, by a name that is one store path segment, with the
/// version of its interface: the operators come back in the order it lists them, each at the
/// version `interface_versions` states, else 1. A vocabulary without the table is of the
/// previous format and states no version of any operator, rather than version 1; a version of 0
/// or of an operator the vocabulary does not list is refused.
#[test]
fn a_vocabulary_names_each_operator_once_by_a_store_name_and_its_interface_version() {
    let directory = tempfile::tempdir().unwrap();
    let read = |text: &str| {
        let path = directory.path().join("operators.toml");
        fs::write(&path, text).unwrap();
        build_server::manifest::read_operators(&path).map_err(|error| error.to_string())
    };
    let operator = |name: &str, interface_version: Option<u32>| build_server::manifest::Operator {
        name: name.to_owned(),
        interface_version,
    };

    assert_eq!(
        read(
            "operators = [\"rms_norm\", \"gemm\", \"activation1d\"]\n\n\
             [interface_versions]\ngemm = 3\n"
        )
        .unwrap(),
        [
            operator("rms_norm", Some(1)),
            operator("gemm", Some(3)),
            operator("activation1d", Some(1))
        ]
    );
    assert_eq!(
        read("operators = [\"rms_norm\", \"gemm\"]\n").unwrap(),
        [operator("rms_norm", None), operator("gemm", None)]
    );
    for (text, refused) in [
        (
            "operators = [\"rms_norm\", \"rms_norm\"]\n[interface_versions]\n",
            "twice",
        ),
        (
            "operators = [\"RmsNorm\"]\n[interface_versions]\n",
            "no operator name",
        ),
        (
            "operators = [\"rms/norm\"]\n[interface_versions]\n",
            "no operator name",
        ),
        (
            "operators = []\nversion = 2\n[interface_versions]\n",
            "version",
        ),
        (
            "operators = [\"gemm\"]\n[interface_versions]\nrms_norm = 2\n",
            "does not list",
        ),
        (
            "operators = [\"gemm\"]\n[interface_versions]\ngemm = 0\n",
            "version 0",
        ),
    ] {
        let error = read(text).unwrap_err();
        assert!(error.contains(refused), "{text}: {error}");
    }
}

/// A scheduling policy: a crate cargo builds for wasm32, which fills the
/// whole policy interface with one module (design sections 5.2 and 5.3).
const POLICY: &str = r#"
op = "program"
backend = "fcfs"

[source]
kind = "wasm"

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
        "{POLICY}\n[kernel]\ndescriptors = [{{ dimensions = {{ mode = [\"default\"] }} }}]\n"
    );
    let error = load(&kernel, &[]).unwrap_err();
    assert!(error.contains("names no kernel package"), "{error}");

    let entry = POLICY.replace(
        "[source]",
        "[module.entries]\nlaunch = \"uniserve_fcfs\"\n\n[source]",
    );
    let error = load(&entry, &[]).unwrap_err();
    assert!(
        error.contains("only a native module has C entries"),
        "{error}"
    );

    // The module states its interface version in its own section, so the manifest states none.
    let versioned = POLICY.replace(
        "backend = \"fcfs\"\n",
        "backend = \"fcfs\"\ninterface_version = 6\n",
    );
    let error = load(&versioned, &[]).unwrap_err();
    assert!(error.contains("interface_version"), "{error}");
}

/// An executable: a package revision the main process runs as its own
/// process, stored as it is (design section 5.1).
const EXECUTABLE: &str = r#"
op = "sampler"
backend = "process"
interface_version = 1

[source]
kind = "executable"
files = ["bin/sampler"]
binary = "bin/sampler"

[kernel]
descriptors = [{ dimensions = { dtype = ["bf16"] } }]
"#;

#[test]
fn an_executable_package_stores_its_binary_and_compiles_nothing() {
    let manifest = load(EXECUTABLE, &["bin/sampler"]).unwrap();
    assert_eq!(
        manifest.source.files_for("dtype_bf16"),
        ["bin/sampler"],
        "an executable stores the files it names"
    );
    assert_eq!(manifest.module.form, ModuleForm::Executable);
    assert_eq!(manifest.toolchain.kind, ToolchainKind::None);
    assert_eq!(manifest.source.kind, SourceKind::Executable);
    assert!(manifest.module.metadata.is_none());
    assert!(manifest.module.entries.is_empty());
    assert_eq!(manifest.executable(), Some("bin/sampler"));
}

/// An executable module names the file its process runs as, and that file is one the package
/// carries: a process is never started from a file guessed among the module's.
#[test]
fn an_executable_package_names_one_of_its_files_as_its_executable() {
    let unnamed = EXECUTABLE.replace("binary = \"bin/sampler\"\n", "");
    let error = load(&unnamed, &["bin/sampler"]).unwrap_err();
    assert!(error.contains("[source] binary"), "{error}");

    let elsewhere = EXECUTABLE.replace("binary = \"bin/sampler\"", "binary = \"bin/other\"");
    let error = load(&elsewhere, &["bin/sampler", "bin/other"]).unwrap_err();
    assert!(error.contains("bin/other"), "{error}");
}

#[test]
fn a_manifest_that_names_a_file_the_package_lacks_is_refused() {
    let error = load(REASONABLE, &[]).unwrap_err();
    assert!(
        error.contains("src/rms_norm.py") && error.contains("tuning/tactics.json"),
        "the refusal names the kernel module and the tuning data: {error}"
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
