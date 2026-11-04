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
    // The committed copy first, so a clean checkout checks both fixtures; the
    // generated one beside it is what the model's example writes, and a host
    // that regenerated the keys today should be checking today's.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let committed = root.join("notes/programs/qwen3-moe-keys.txt");
    let generated = root.join(".local/qwen3-moe-keys.txt");
    let path = if committed.is_file() {
        committed
    } else {
        generated
    };
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
///
/// A name here does not settle what a package's field means: `output` is a
/// convolution's output element type and also `adaptive_pool`'s pooled width, and
/// `input` is a `gemm`'s activations' type and would be a count anywhere else. So a
/// value in one of these fields is examined only when it is not a bare integer —
/// which is what tells an element type (`bf16`, and a package's mistake `float16`)
/// from a width or an extent (`32`).
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
        for shape in kernel.descriptors.shapes() {
            for (field, values) in &shape.dimensions {
                if !ELEMENT_TYPE_FIELDS.contains(&field.as_str()) {
                    continue;
                }
                for value in values {
                    if value.parse::<u64>().is_ok() {
                        // A width or an extent, not an element type.
                        continue;
                    }
                    if !names.contains(value) {
                        failures.push(format!(
                            "{} {}: {field} = {value:?} is not one of {names:?}",
                            manifest.op, manifest.backend
                        ));
                    }
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
                    // One entry per shape the package declares: a key has to
                    // parse against *one* of an operator's shapes, not against a
                    // merged table.
                    let shapes = manifest.kernel?.descriptors.shapes().to_vec();
                    Some(
                        shapes
                            .into_iter()
                            .filter(|shape| !shape.dimensions.is_empty())
                            .map(move |shape| (package.display().to_string(), shape.dimensions)),
                    )
                })
                .flatten()
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
                // Name what is wrong rather than that something is: a key whose
                // values a package does not declare and a key whose field names
                // are different are both "not covered", and the fix for each is
                // different work by a different actor.
                let why: Vec<String> = declared
                    .iter()
                    .map(|(package, dimensions)| match parse_key(dimensions, key.as_str()) {
                        None => {
                            let names: Vec<&str> = dimensions.keys().map(String::as_str).collect();
                            format!("{package} names the fields [{}]", names.join(", "))
                        }
                        Some(fields) => {
                            let missing: Vec<String> = fields
                                .iter()
                                .filter(|(name, value)| {
                                    !dimensions
                                        .get(name)
                                        .is_some_and(|allowed| allowed.iter().any(|one| one == value))
                                })
                                .map(|(name, value)| format!("{name}_{value}"))
                                .collect();
                            if missing.is_empty() {
                                format!("{package} declares every value")
                            } else {
                                format!("{package} does not declare {}", missing.join(", "))
                            }
                        }
                    })
                    .collect();
                failures.push(format!("{operator}\t{key}: no package covers it; {}", why.join("; ")));
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

/// The expert exchange's topology is a descriptor dimension, so the store keeps the
/// group sizes apart — and the package covers exactly the one a host can serve today.
///
/// `ranks` being in the module key is what stops a module built for one group from
/// being resolved for another. The larger groups have no module on purpose: the
/// executor's `group` interface, over which a package would exchange peer handles,
/// is not named yet, and `create` refuses those groups by name rather than falling
/// back to one rank (`test/create_check.py` in the package checks that refusal).
#[test]
fn the_exchange_covers_one_group_size_and_no_other() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/ops/kernels/expert_exchange/alltoall");
    let manifest = Manifest::load(&package).expect("the exchange's package parses");
    let key = |ranks: u32| {
        format!(
            "mode_dispatch_hidden_2048_top_8_experts_128_max_tokens_8192_ranks_{ranks}_dtype_bf16"
        )
    };
    assert!(
        manifest.covers(&key(1)),
        "the package must build the module of the one rank it serves"
    );
    for ranks in [2, 4, 8] {
        assert!(
            !manifest.covers(&key(ranks)),
            "the package declares a module for a group of {ranks} ranks, which no host \
             can serve: the transport for a group is not named yet"
        );
    }
}

/// Every package names an operator the `ops` crate declares.
///
/// A package is requested by its operator's name — `install` and `prepare` take an operator name,
/// and a program slot is built from one — so a package published under a name no operator carries
/// is a program nothing can ever request. The names move: an operator renamed in `crates/ops`
/// leaves its package behind, and nothing else in the tree would notice, because the store builds
/// a package by its manifest and never asks whether an operator exists.
#[test]
fn every_package_names_an_operator_the_ops_crate_declares() {
    let operators = ops::operators();
    let mut failures = Vec::new();
    for package in packages() {
        let Ok(manifest) = Manifest::load(&package) else {
            continue;
        };
        if !operators.contains(&manifest.op.as_str()) {
            failures.push(format!(
                "{}: op `{}` is no operator of `crates/ops`",
                package.display(),
                manifest.op
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "a package under a name no operator carries is a program nothing can request:\n{}",
        failures.join("\n")
    );
}

/// The three convolution packages are one source in three places, and the store builds
/// each package from its own copy of its root, so a source shared by path is not
/// available to them. Identity in bytes is the invariant that keeps them one program:
/// the entry differs only by the package's `[kernel.entry]`, and the element types only
/// by the build script's defines. A divergence would make one backend compute what the
/// other two do not, which no per-package check could see.
#[test]
fn the_convolutions_are_one_source_in_three_packages() {
    const CONVOLUTIONS: [&str; 3] = ["conv1d", "conv3d", "conv_transpose1d"];
    const SHARED: [&str; 4] = [
        "src/conv.cu",
        "include/uniserve_conv.h",
        "test/conv_check.py",
        "test/refusal_check.py",
    ];
    for shared in SHARED {
        let sources = CONVOLUTIONS.map(|op| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../ops/kernels")
                .join(op)
                .join("cudnn")
                .join(shared);
            std::fs::read(&path).unwrap_or_else(|error| {
                panic!("{shared} of {op} is not readable at {}: {error}", path.display())
            })
        });
        assert!(
            sources[0] == sources[1] && sources[1] == sources[2],
            "{shared} differs between the three convolution packages"
        );
    }
}

/// Operators that are one computation with one difference share one check script, and each
/// set of packages must carry the same bytes of it: whoever edits the reference, the
/// interpreter's truncation or a case edits every copy, or they diverge and one operator's
/// evidence quietly stops being the others'.
#[test]
fn a_family_of_operators_shares_one_check() {
    const FAMILIES: [(&str, &[&str]); 3] = [
        (
            "triton/test/row_norm_check.py",
            &["rms_norm", "add_rms_norm"],
        ),
        ("triton/test/softmax_check.py", &["softmax", "log_softmax"]),
        (
            "triton/test/sampling_check.py",
            &[
                "logits_gather",
                "vocab_unpad",
                "logits_shape",
                "filter_candidates",
                "inverse_cdf",
                "argmax",
            ],
        ),
    ];
    for (shared, operators) in FAMILIES {
        let scripts: Vec<String> = operators
            .iter()
            .map(|op| {
                let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../ops/kernels")
                    .join(op)
                    .join(shared);
                std::fs::read_to_string(&path).unwrap_or_else(|error| {
                    panic!("{op}'s check is not readable at {}: {error}", path.display())
                })
            })
            .collect();
        for (index, script) in scripts.iter().enumerate().skip(1) {
            assert!(
                script == &scripts[0],
                "{shared} differs between {} and {}",
                operators[0],
                operators[index]
            );
        }
        for op in operators {
            assert!(
                scripts[0].contains("--op") && scripts[0].contains(op),
                "{shared} does not name {op} on its command line"
            );
        }
    }
}
