//! The behaviour of the operator interface that a caller — the engine's loader, a program package's
//! author, the executor's shadow harness — observes and relies on:
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
use ops::tensor::DType;
use ops::{Error, Float, Tolerance};

/// A descriptor of the test's own, shaped like the real ones: fields in a fixed order.
struct TestDescriptor {
    dtype: DType,
    columns: u32,
    scale: Float,
}

impl fmt::Display for TestDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
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

#[test]
fn cubin_metadata_reads_triton_fields_and_skips_what_it_does_not_need() {
    let json = br#"{
        "name": "rms_norm_kernel",
        "num_warps": 8,
        "shared": 4096,
        "num_ctas": 2,
        "launch_pdl": true,
        "launch_cooperative_grid": false,
        "global_scratch_size": 1048576,
        "global_scratch_align": 128,
        "num_stages": 3,
        "cluster_dims": [2, 1, 1],
        "target": {"arch": 100, "backend": "cuda"}
    }"#;
    let metadata = CubinMetadata::parse(json).unwrap();
    assert_eq!(metadata.name, "rms_norm_kernel");
    assert_eq!(metadata.threads(), 8 * 32);
    assert_eq!(metadata.shared, 4096);
    assert_eq!(metadata.num_ctas, 2);
    assert!(metadata.launch_pdl);
    assert!(!metadata.launch_cooperative_grid);
    assert_eq!(metadata.global_scratch_size, 1_048_576);
    assert_eq!(metadata.global_scratch_align, 128);
    assert!(metadata.needs_scratch());
}

#[test]
fn cubin_metadata_defaults_what_triton_omits_and_refuses_what_it_must_have() {
    let metadata = CubinMetadata::parse(br#"{"name": "copy"}"#).unwrap();
    assert_eq!(metadata.num_warps, 1);
    assert_eq!(metadata.threads(), 32);
    assert_eq!(metadata.num_ctas, 1);
    assert_eq!(metadata.shared, 0);
    assert!(!metadata.needs_scratch());

    let error = CubinMetadata::parse(br#"{"num_warps": 4}"#).unwrap_err();
    assert!(matches!(error, Error::Metadata(_)));
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
    let expected = 2.0 * ops::tolerance::gamma(2048, DType::F32) + DType::BF16.unit_roundoff();
    assert!((f64::from(norm.rtol) - expected).abs() < 1e-9);
    assert_eq!(norm.atol, 0.0);

    // A sum of mixed-sign products is bounded by the same γₙ scaled by √n.
    let dot = Tolerance::dot(2048, DType::F32, DType::BF16);
    let expected_dot = 2.0 * ops::tolerance::gamma(2048, DType::F32) * (2048.0_f64).sqrt()
        + DType::BF16.unit_roundoff();
    assert!((f64::from(dot.rtol) - expected_dot).abs() < 1e-9);
    assert!(dot.rtol > norm.rtol);

    // An exact-length reduction is bounded by the output dtype alone.
    let single = Tolerance::rounding(0, DType::F32, DType::BF16);
    assert!((f64::from(single.rtol) - DType::BF16.unit_roundoff()).abs() < 1e-9);
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
fn native_call_structures_have_the_layouts_their_headers_mirror() {
    let gemm = &ops::gemm::GEMM_CALL_LAYOUT;
    assert_eq!(std::mem::size_of::<ops::gemm::Call>(), gemm.size);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, out), gemm.out);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, input), gemm.input);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, weight), gemm.weight);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, bias), gemm.bias);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, m), gemm.m);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, n), gemm.n);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, k), gemm.k);
    assert_eq!(
        std::mem::offset_of!(ops::gemm::Call, input_row_stride),
        gemm.input_row_stride
    );
    assert_eq!(
        std::mem::offset_of!(ops::gemm::Call, weight_row_stride),
        gemm.weight_row_stride
    );
    assert_eq!(
        std::mem::offset_of!(ops::gemm::Call, out_row_stride),
        gemm.out_row_stride
    );
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, alpha), gemm.alpha);
    assert_eq!(std::mem::offset_of!(ops::gemm::Call, error), gemm.error);

    // The convolution structure places its geometry where the header does.
    assert_eq!(std::mem::offset_of!(ops::conv::Call, out), 0);
    assert_eq!(std::mem::offset_of!(ops::conv::Call, input), 8);
    assert_eq!(std::mem::offset_of!(ops::conv::Call, weight), 16);
    assert_eq!(std::mem::offset_of!(ops::conv::Call, bias), 24);
    assert_eq!(std::mem::offset_of!(ops::conv::Call, input_shape), 32);
    assert_eq!(
        std::mem::offset_of!(ops::conv::Call, weight_shape),
        32 + 5 * std::mem::size_of::<i64>()
    );
    assert_eq!(
        std::mem::offset_of!(ops::conv::Call, out_shape),
        32 + 10 * std::mem::size_of::<i64>()
    );
    assert_eq!(
        std::mem::offset_of!(ops::conv::Call, kernel),
        32 + 15 * std::mem::size_of::<i64>()
    );
}

#[test]
fn element_types_keep_their_abi_numbering_and_round_off() {
    // A package's header carries these numbers, so they cannot move.
    assert_eq!(DType::U8.as_i64(), 0);
    assert_eq!(DType::I32.as_i64(), 1);
    assert_eq!(DType::I64.as_i64(), 2);
    assert_eq!(DType::F16.as_i64(), 3);
    assert_eq!(DType::BF16.as_i64(), 4);
    assert_eq!(DType::F32.as_i64(), 5);
    assert_eq!(DType::F8E4M3.as_i64(), 6);
    assert_eq!(DType::F4E2M1.as_i64(), 7);
    for number in 0..8 {
        assert_eq!(DType::from_i64(number).map(DType::as_i64), Some(number));
    }
    assert_eq!(DType::from_i64(8), None);

    assert_eq!(DType::BF16.bytes(), Some(2));
    assert_eq!(DType::F4E2M1.bytes(), None, "two values share one byte");
    assert_eq!(DType::F32.unit_roundoff(), 2.0_f64.powi(-24));
    assert_eq!(DType::BF16.unit_roundoff(), 2.0_f64.powi(-8));
    assert_eq!(DType::F16.bytes(), Some(2));
    assert_eq!(DType::F8E4M3.bytes(), Some(1));
    assert_eq!(DType::F32.accumulate(), DType::F32);
    assert_eq!(DType::BF16.to_string(), "bf16");
    assert!(!DType::I32.is_float());
    assert!(DType::F8E4M3.is_float());
}

#[test]
fn shapes_are_row_major_views_with_a_row_count() {
    let shape = ops::Shape::new(&[3, 5, 128]);
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.numel(), 3 * 5 * 128);
    assert_eq!(shape.rows(), 15);
    assert_eq!(shape.last(), 128);
    assert_eq!(shape.dim(1), 5);
    assert_eq!(shape.dims(), &[3, 5, 128]);
    assert_eq!(
        ops::Strides::row_major(shape).as_slice(),
        &[5 * 128, 128, 1]
    );
    assert_eq!(ops::Strides::row_major(shape).last(), 1);
    assert_eq!(shape.with_last(256).dims(), &[3, 5, 256]);
    assert!(shape.matches(&ops::Shape::new(&[3, 5, 128])));
    assert!(!shape.matches(&ops::Shape::new(&[3, 5, 256])));

    let unshaped = ops::Shape::new(&[]);
    assert_eq!(unshaped.rank(), 0);
    assert_eq!(unshaped.numel(), 1);
    assert_eq!(unshaped.rows(), 1);
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
    use ops::modulation::{GatedResidualDescriptor, ModulatedRmsNormDescriptor};
    use ops::{DType, Float};

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

    let residual = GatedResidualDescriptor {
        dtype: DType::BF16,
        indices: DType::I32,
        width: 3072,
        elements: 3072 * 64,
        gate_stride: 3072,
    };
    assert_ne!(
        ops::descriptor::key(&residual),
        ops::descriptor::key(&GatedResidualDescriptor {
            elements: 3072 * 65,
            ..residual
        }),
        "the walked element count is part of the compiled variant"
    );
}

#[test]
fn an_operator_is_found_by_the_name_its_programs_are_filed_under() {
    // A program layer's packages name their operators as text; the executor resolves one to the
    // static name a generation is keyed by, and gets nothing when this build has no such operator.
    assert_eq!(
        ops::operator_name("rms_norm"),
        Some(<ops::norm::RmsNorm as ops::Op>::NAME)
    );
    assert_eq!(
        ops::operator_name(<ops::gemm::Gemm as ops::Op>::NAME),
        Some(<ops::gemm::Gemm as ops::Op>::NAME),
        "an operator's own name resolves to itself, not to a prefix or a neighbour"
    );
    assert_eq!(ops::operator_name("no_such_operator"), None);
    assert_eq!(ops::operator_name(""), None);
    assert_eq!(
        ops::operator_name("gemm_prefill"),
        None,
        "a descriptor is not a name"
    );

    // Every name the build lists is a real operator name, and no name is listed twice: the list is
    // what a package's coverage is checked against.
    let mut names: Vec<&'static str> = ops::operators().to_vec();
    let listed = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), listed, "an operator name is listed once");
    for name in ops::operators() {
        assert!(!name.is_empty());
        assert!(
            name.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{name} is not a program directory name"
        );
        assert_eq!(
            ops::operator_name(name),
            Some(*name),
            "{name} resolves to itself"
        );
    }
}

#[test]
fn a_sectioned_rotation_names_its_axes_in_its_key() {
    let plain = ops::rope::FactorsDescriptor {
        dtype: ops::DType::F32,
        positions: ops::DType::I32,
        columns: 64,
        block: 1024,
        scale: ops::Float::from(1_000_000.0_f32),
        pdl: false,
        axes: 1,
        sections: [0; ops::rope::MAX_AXES],
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
