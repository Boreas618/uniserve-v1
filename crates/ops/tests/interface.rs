//! The behaviour of the operator interface that a caller — the engine's loader, a package's
//! author, the worker's shadow harness — observes and relies on:
//!
//! - descriptor keys are canonical text, because they are the last segment of a module's path;
//! - a cubin's launch metadata is read from Triton's JSON, field for field;
//! - a launch grid is checked against the driver's limits before it reaches the driver;
//! - a tolerance follows from the reduction length and the dtypes, and compares as
//!   `|candidate − reference| ≤ atol + rtol·|reference|`;
//! - the native call structures have the layouts the packages' headers mirror;
//! - the element type numbering a package header carries does not move.
//!
//! What needs a driver — loading a module, carving a twin region, a generation holding loaded
//! kernels — is exercised by the caller's device test, because a kernel cannot be stood in for
//! without standing in for the CUDA boundary.

use std::fmt;

use ops::descriptor::Key;
use ops::kernel::{CubinMetadata, LaunchGrid};
use ops::tolerance::{accumulate, unit_roundoff};
use ops::{Error, Float, Tolerance, abi};
use tensor::DType;

/// A descriptor of the test's own, shaped like the real ones: fields in a fixed order.
struct TestDescriptor {
    dtype: DType,
    columns: u32,
    scale: Float,
}

impl fmt::Display for TestDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("columns", self.columns)
            .field("scale", self.scale)
            .finish(formatter)
    }
}

#[test]
fn descriptor_key_is_canonical() {
    let descriptor = TestDescriptor {
        dtype: DType::BF16,
        columns: 2048,
        scale: Float::from(0.5_f32),
    };
    assert_eq!(
        descriptor.to_string(),
        "dtype_bf16_columns_2048_scale_0x3f000000"
    );

    // The same values give the same key, whatever built them.
    let same = TestDescriptor {
        dtype: DType::BF16,
        columns: 2048,
        scale: Float::from(0.5_f32),
    };
    assert_eq!(descriptor.to_string(), same.to_string());

    // A float enters a key by its bits: one unit in the last place is a different key, and the
    // decimal spelling never enters.
    let next = Float::from(f32::from_bits(0.5_f32.to_bits() + 1));
    let changed = TestDescriptor {
        dtype: DType::BF16,
        columns: 2048,
        scale: next,
    };
    assert_ne!(descriptor.to_string(), changed.to_string());
    assert_eq!(Float::from(0.1_f32).to_string(), "0x3dcccccd");
    assert_eq!(Float::from(1.0_f64).to_string(), "0x3ff0000000000000");
}

#[test]
fn descriptor_key_reports_an_empty_descriptor_as_empty_text() {
    assert_eq!(Key::new().text(), "");
    assert_eq!(
        Key::new()
            .field("page", 64)
            .variant("mask", "causal")
            .text(),
        "page_64_mask_causal"
    );
}

/// A metadata record as Triton writes it, with the fields a launch reads and some it does not.
const TRITON_RECORD: &str = r#"{
    "name": "rms_norm_kernel",
    "num_warps": 8,
    "shared": 4096,
    "num_ctas": 2,
    "launch_pdl": true,
    "launch_cooperative_grid": false,
    "signature": {"x_ptr": "*bf16", "rows": "i32", "eps": "fp32"},
    "global_scratch_size": 1048576,
    "global_scratch_align": 128,
    "profile_scratch_size": 0,
    "profile_scratch_align": 1,
    "definitions": {"BLOCK": 2048, "PDL": true, "COMPUTE": "fp32"},
    "num_stages": 3,
    "cluster_dims": [2, 1, 1],
    "target": {"arch": 100, "backend": "cuda"}
}"#;

#[test]
fn cubin_metadata_reads_triton_fields_and_skips_what_it_does_not_need() {
    let metadata = CubinMetadata::parse(TRITON_RECORD.as_bytes()).unwrap();
    assert_eq!(metadata.name, "rms_norm_kernel");
    assert_eq!(metadata.threads(), 8 * 32);
    assert_eq!(metadata.shared, 4096);
    assert_eq!(metadata.num_ctas, 2);
    assert!(metadata.launch_pdl);
    assert!(!metadata.launch_cooperative_grid);
    assert_eq!(metadata.global_scratch_size, 1_048_576);
    assert_eq!(metadata.global_scratch_align, 128);
    assert!(metadata.needs_scratch());
    // The signature keeps the record's order, which is the kernel's parameter order.
    let names: Vec<&str> = metadata
        .signature
        .iter()
        .map(|parameter| parameter.name.as_str())
        .collect();
    assert_eq!(names, ["x_ptr", "rows", "eps"]);
    assert!(metadata.signature[0].is_pointer());
    assert!(!metadata.signature[2].is_pointer());
}

#[test]
fn cubin_metadata_answers_an_integer_definition_and_refuses_any_other_by_name() {
    let metadata = CubinMetadata::parse(TRITON_RECORD.as_bytes()).unwrap();
    assert_eq!(metadata.integer("BLOCK").unwrap(), 2048);

    // A definition the module was not compiled with, and one that is no integer.
    for (name, refusal) in [
        ("BLOCK_Q", "no definition `BLOCK_Q`"),
        ("COMPUTE", "\"fp32\""),
    ] {
        let error = metadata.integer(name).unwrap_err();
        assert!(matches!(error, Error::Metadata(_)), "{name}: {error}");
        assert!(error.to_string().contains("rms_norm_kernel"), "{error}");
        assert!(error.to_string().contains(refusal), "{error}");
    }
}

#[test]
fn cubin_metadata_refuses_a_record_without_a_field_a_launch_reads() {
    let record: serde_json::Value = serde_json::from_str(TRITON_RECORD).unwrap();
    for field in [
        "name",
        "num_warps",
        "shared",
        "num_ctas",
        "launch_pdl",
        "launch_cooperative_grid",
        "signature",
        "global_scratch_size",
        "global_scratch_align",
        "profile_scratch_size",
        "profile_scratch_align",
        "definitions",
    ] {
        let mut partial = record.clone();
        partial.as_object_mut().unwrap().remove(field);
        let error = CubinMetadata::parse(partial.to_string().as_bytes()).unwrap_err();
        assert!(matches!(error, Error::Metadata(_)), "{field}: {error}");
        assert!(error.to_string().contains(field), "{field}: {error}");
    }

    let mut idle = record;
    idle["num_warps"] = 0.into();
    let error = CubinMetadata::parse(idle.to_string().as_bytes()).unwrap_err();
    assert!(error.to_string().contains("0 warps"), "{error}");

    let error = CubinMetadata::parse(b"not json").unwrap_err();
    assert!(matches!(error, Error::Metadata(_)));
}

#[test]
fn launch_grid_is_checked_before_the_driver_sees_it() {
    assert_eq!(LaunchGrid::linear(7).unwrap().blocks, [7, 1, 1]);
    assert_eq!(LaunchGrid::new([2, 3, 4]).unwrap().blocks, [2, 3, 4]);
    assert_eq!(LaunchGrid::covering(2049, 2048).unwrap().blocks, [2, 1, 1]);
    assert_eq!(LaunchGrid::covering(2048, 2048).unwrap().blocks, [1, 1, 1]);
    assert!(matches!(LaunchGrid::linear(0).unwrap_err(), Error::Grid(_)));
    assert!(matches!(
        LaunchGrid::new([u64::from(u32::MAX) + 1, 1, 1]).unwrap_err(),
        Error::Grid(_)
    ));
    assert!(matches!(
        LaunchGrid::covering(1, 0).unwrap_err(),
        Error::Grid(_)
    ));
}

#[test]
fn tolerance_follows_the_reduction_length_and_dtypes() {
    // γ₀ is zero, γₙ grows with the reduction length, and a wider element type rounds harder.
    assert_eq!(ops::tolerance::gamma(0, DType::F32), 0.0);
    assert!(ops::tolerance::gamma(4096, DType::F32) > ops::tolerance::gamma(1024, DType::F32));
    assert!(ops::tolerance::gamma(1024, DType::BF16) > ops::tolerance::gamma(1024, DType::F32));
    assert_eq!(ops::tolerance::gamma(4, DType::I32), 0.0);

    // A scale reduction: 2γₙ plus one output rounding.
    let norm = Tolerance::rounding(2048, DType::F32, DType::BF16);
    let expected = 2.0 * ops::tolerance::gamma(2048, DType::F32) + unit_roundoff(DType::BF16);
    assert!((f64::from(norm.rtol) - expected).abs() < 1e-9);
    // Its absolute part is the output's subnormal unit: a BF16 result stored below 2^-126 is
    // compared on that grid.
    let subnormal = ops::tolerance::underflow(DType::BF16);
    assert_eq!(f64::from(norm.atol), subnormal);
    assert!(norm.allows(0.0, subnormal as f32));
    assert!(!norm.allows(0.0, 8.0 * subnormal as f32));

    // A sum of mixed-sign products is bounded by the same γₙ scaled by √n.
    let dot = Tolerance::dot(2048, DType::F32, DType::BF16);
    let expected_dot = 2.0 * ops::tolerance::gamma(2048, DType::F32) * (2048.0_f64).sqrt()
        + unit_roundoff(DType::BF16);
    assert!((f64::from(dot.rtol) - expected_dot).abs() < 1e-9);
    assert!(dot.rtol > norm.rtol);

    // An exact-length reduction is bounded by the output dtype alone.
    let single = Tolerance::rounding(0, DType::F32, DType::BF16);
    assert!((f64::from(single.rtol) - unit_roundoff(DType::BF16)).abs() < 1e-9);
}

#[test]
fn tolerance_compares_against_the_reference_and_an_envelope_extends_it() {
    let tolerance = Tolerance::new(1e-3, 1e-2);
    assert!(tolerance.allows(1.0, 1.0));
    assert!(tolerance.allows(1.0, 1.010));
    assert!(!tolerance.allows(1.0, 1.02));
    // The absolute term covers a reference at zero, which the relative term cannot.
    assert!(tolerance.allows(0.0, 1e-3));
    assert!(!tolerance.allows(0.0, 1e-2));

    let exact = Tolerance::exact();
    assert!(exact.allows(0.5, 0.5));
    assert!(!exact.allows(0.5, 0.5 + f32::EPSILON));

    let envelope = Tolerance::dot(1024, DType::F32, DType::BF16).with_atol(2.0);
    assert_eq!(envelope.atol, 2.0);
}

#[test]
fn element_types_keep_their_abi_numbering_and_round_off() {
    // A package's header carries these numbers, so they cannot move.
    assert_eq!(abi::code(DType::U8).unwrap(), 0);
    assert_eq!(abi::code(DType::I32).unwrap(), 1);
    assert_eq!(abi::code(DType::I64).unwrap(), 2);
    assert_eq!(abi::code(DType::F16).unwrap(), 3);
    assert_eq!(abi::code(DType::BF16).unwrap(), 4);
    assert_eq!(abi::code(DType::F32).unwrap(), 5);
    assert_eq!(abi::code(DType::F8E4M3).unwrap(), 6);
    assert_eq!(abi::code(DType::F4).unwrap(), 7);
    // Appended for the audio decoders' PCM output; the seven numbers before it never move.
    assert_eq!(abi::code(DType::I16).unwrap(), 8);
    // Appended for the tensor primitives' indices and for double precision.
    assert_eq!(abi::code(DType::U32).unwrap(), 9);
    assert_eq!(abi::code(DType::F64).unwrap(), 10);
    assert_eq!(abi::bytes(DType::I16), Some(2));
    assert_eq!(unit_roundoff(DType::I16), 0.0, "an integer has no rounding");
    assert!(!DType::I16.is_float());

    assert_eq!(abi::bytes(DType::BF16), Some(2));
    assert_eq!(abi::bytes(DType::F4), None, "two values share one byte");
    assert_eq!(unit_roundoff(DType::F32), 2.0_f64.powi(-24));
    assert_eq!(unit_roundoff(DType::BF16), 2.0_f64.powi(-8));
    assert_eq!(abi::bytes(DType::F16), Some(2));
    assert_eq!(abi::bytes(DType::F8E4M3), Some(1));
    assert_eq!(accumulate(DType::F32), DType::F32);
    assert_eq!(abi::name(DType::BF16), "bf16");
    assert_eq!(abi::name(DType::F4), "f4e2m1");
    assert!(!DType::I32.is_float());
    assert!(DType::F8E4M3.is_float());
}

#[test]
fn a_generation_reports_a_missing_kernel_with_what_was_asked_for() {
    let empty = ops::Generation::new(7, vec![]);
    assert_eq!(empty.number(), 7);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    let error = empty.kernel("gemm", "input_bf16_k_2048").unwrap_err();
    match error {
        Error::MissingKernel {
            operator,
            descriptor,
            generation,
        } => {
            assert_eq!(operator, "gemm");
            assert_eq!(descriptor, "input_bf16_k_2048");
            assert_eq!(generation, 7);
        }
        other => panic!("expected a missing-kernel error, found {other}"),
    }
}

#[test]
fn modulation_keys_separate_every_compiled_variant() {
    use ops::Float;
    use ops::modulation::{GatedResidualDescriptor, ModulatedRmsNormDescriptor};
    use tensor::DType;

    let norm = ModulatedRmsNormDescriptor {
        dtype: DType::BF16,
        output: DType::BF16,
        indices: DType::I32,
        width: 3072,
        block: 4096,
        epsilon: Float::from(1e-6_f32),
        gate_stride: 3072,
        shift_stride: 3072,
        scale_stride: 3072,
        update: true,
        retain: false,
    };
    let key = ops::descriptor::key(&norm);
    // The same descriptor prints the same key, so a program is compiled once.
    assert_eq!(key, ops::descriptor::key(&norm));
    // The epsilon enters by bit pattern, so two epsilons that differ in one bit are two variants.
    let other_epsilon = ModulatedRmsNormDescriptor {
        epsilon: Float::from(1e-6_f32 + f32::EPSILON * 1e-6_f32),
        ..norm
    };
    assert_ne!(key, ops::descriptor::key(&other_epsilon));
    assert!(
        !key.contains("0.000001"),
        "a decimal spelling would alias: {key}"
    );
    for (name, variant) in [
        (
            "update",
            ModulatedRmsNormDescriptor {
                update: false,
                ..norm
            },
        ),
        (
            "retain",
            ModulatedRmsNormDescriptor {
                retain: true,
                ..norm
            },
        ),
        (
            "output",
            ModulatedRmsNormDescriptor {
                output: DType::F8E4M3,
                ..norm
            },
        ),
        (
            "gate_stride",
            ModulatedRmsNormDescriptor {
                gate_stride: 6144,
                ..norm
            },
        ),
        (
            "width",
            ModulatedRmsNormDescriptor {
                width: 1536,
                block: 2048,
                ..norm
            },
        ),
    ] {
        let changed = ops::descriptor::key(&variant);
        assert_ne!(key, changed, "changing {name} must select another variant");
    }

    // The gated residual's row count is a launch value, so it is in no key: what selects a variant
    // is the geometry the kernel is compiled against, and the gate's row stride is the last of them.
    let residual = GatedResidualDescriptor {
        dtype: DType::BF16,
        indices: DType::I32,
        width: 3072,
        gate_stride: 3072,
    };
    assert_ne!(
        ops::descriptor::key(&residual),
        ops::descriptor::key(&GatedResidualDescriptor {
            gate_stride: 6144,
            ..residual
        }),
        "the gate's row stride is part of the compiled variant"
    );
}

#[test]
fn an_operator_is_found_by_the_name_its_programs_are_filed_under() {
    // A generation layer's packages name their operators as text; the worker resolves one to the
    // static name a generation is keyed by, and gets nothing when this build has no such operator.
    assert_eq!(
        proto::module::operator::named("rms_norm"),
        Some(<ops::norm::RmsNorm as ops::Op>::NAME)
    );
    assert_eq!(
        proto::module::operator::named(<ops::gemm::Gemm as ops::Op>::NAME),
        Some(<ops::gemm::Gemm as ops::Op>::NAME),
        "an operator's own name resolves to itself, not to a prefix or a neighbour"
    );
    assert_eq!(proto::module::operator::named("no_such_operator"), None);
    assert_eq!(proto::module::operator::named(""), None);
    assert_eq!(
        proto::module::operator::named("gemm_prefill"),
        None,
        "a descriptor is not a name"
    );

    // Every name the build lists is a real operator name, and no name is listed twice: the list is
    // what a package's coverage is checked against.
    let mut names: Vec<&'static str> = proto::module::operator::ALL.to_vec();
    let listed = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), listed, "an operator name is listed once");
    for name in proto::module::operator::ALL {
        assert!(!name.is_empty());
        assert!(
            name.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{name} is not a program directory name"
        );
        assert_eq!(
            proto::module::operator::named(name),
            Some(*name),
            "{name} resolves to itself"
        );
    }
}

#[test]
fn a_sectioned_rotation_names_its_axes_in_its_key() {
    let plain = ops::rope::FactorsDescriptor {
        dtype: tensor::DType::F32,
        positions: tensor::DType::I32,
        columns: 64,
        block: 1024,
        scale: ops::Float::from(1_000_000.0_f32),
        pdl: false,
        axes: 1,
        sections: [0; ops::rope::MAX_AXES],
        interleaved: false,
    };
    let key = ops::descriptor::key(&plain);
    assert!(
        !key.contains("axes"),
        "{key} names no axis for one-dimensional positions"
    );
    assert!(!key.contains("sections"), "{key} carries no section");

    // M-RoPE's cut of a 128-element head into (time, height, width) exported as 128 elements, and
    // the same head with a different cut: both are compiled variants of their own.
    let three = ops::rope::FactorsDescriptor {
        axes: 3,
        sections: [16, 24, 24, 0],
        interleaved: false,
        columns: 64,
        ..plain
    };
    let sectioned = ops::descriptor::key(&three);
    assert_ne!(sectioned, key);
    assert!(sectioned.contains("axes_3"), "{sectioned}");
    assert!(sectioned.contains("sections_16-24-24"), "{sectioned}");

    let other = ops::descriptor::key(&ops::rope::FactorsDescriptor {
        sections: [32, 32, 0, 0],
        axes: 2,
        ..three
    });
    assert_ne!(other, sectioned, "the cut is part of the compilation");
}
