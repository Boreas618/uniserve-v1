//! The resamplers against the reference implementations' outputs.
//!
//! The fixtures under `tests/fixtures/resample/` are produced by
//! `tests/fixtures/resample/generate.py` from Pillow 12.3, torchaudio 2.11,
//! torchvision 0.28 and torch 2.13 (the read-only `.venv` this port replaces).
//! Each kind's `cases.txt` names its cases and their extents; the inputs and
//! expected outputs are raw little-endian bytes in the layout the function
//! takes and returns.

use std::path::{Path, PathBuf};

use super::{bicubic_antialias, lanczos_resize, sinc_resample};

/// The committed reference fixtures.
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/resample")
}

/// One case's name, its numeric fields and its input and expected bytes.
struct Case {
    name: String,
    fields: Vec<usize>,
    input: Vec<u8>,
    expected: Vec<u8>,
}

/// Reads one kind's `cases.txt` and each case's files.
fn cases(kind: &str) -> Vec<Case> {
    let directory = fixtures().join(kind);
    let manifest = std::fs::read_to_string(directory.join("cases.txt")).unwrap();
    let mut cases = Vec::new();
    for line in manifest.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let name = fields[0].to_owned();
        cases.push(Case {
            input: std::fs::read(directory.join(format!("{name}.in"))).unwrap(),
            expected: std::fs::read(directory.join(format!("{name}.out"))).unwrap(),
            fields: fields[1..].iter().map(|value| value.parse().unwrap()).collect(),
            name,
        });
    }
    assert!(!cases.is_empty(), "the {kind} fixtures are missing");
    cases
}

#[test]
fn lanczos_matches_pillow() {
    for case in cases("lanczos") {
        let (channels, height, width, out_height, out_width) = (
            case.fields[0],
            case.fields[1],
            case.fields[2],
            case.fields[3],
            case.fields[4],
        );
        assert_eq!(channels, 3, "Pillow resizes RGB images here");
        let actual = lanczos_resize(
            &case.input,
            width as u32,
            height as u32,
            out_width as u32,
            out_height as u32,
        )
        .unwrap();
        assert_eq!(
            actual, case.expected,
            "lanczos case {} disagrees with Pillow",
            case.name
        );
    }
}

#[test]
fn lanczos_resizes_to_the_same_size_as_a_copy() {
    let source: Vec<u8> = (0..(4 * 3 * 3)).map(|value| value as u8 * 5).collect();
    let actual = lanczos_resize(&source, 4, 3, 4, 3).unwrap();
    assert_eq!(actual, source);
}

#[test]
fn lanczos_refuses_a_mismatched_extent() {
    let source = vec![0u8; 4 * 3 * 3 - 1];
    assert!(lanczos_resize(&source, 4, 3, 8, 6).is_err());
    assert!(lanczos_resize(&[], 0, 3, 8, 6).is_err());
}

#[test]
fn bicubic_matches_torchvision() {
    for case in cases("bicubic") {
        let (channels, height, width, out_height, out_width) = (
            case.fields[0],
            case.fields[1],
            case.fields[2],
            case.fields[3],
            case.fields[4],
        );
        let actual = bicubic_antialias(
            &case.input,
            channels,
            height as u32,
            width as u32,
            out_height as u32,
            out_width as u32,
        )
        .unwrap();
        assert_eq!(
            actual, case.expected,
            "bicubic case {} disagrees with torchvision",
            case.name
        );
    }
}

#[test]
fn bicubic_resizes_to_the_same_size_as_a_copy() {
    let source: Vec<u8> = (0..(2 * 3 * 4)).map(|value| value as u8 * 3).collect();
    let actual = bicubic_antialias(&source, 2, 3, 4, 3, 4).unwrap();
    assert_eq!(actual, source);
}

#[test]
fn bicubic_refuses_a_mismatched_extent() {
    let source = vec![0u8; 3 * 4 * 5 - 1];
    assert!(bicubic_antialias(&source, 3, 4, 5, 8, 10).is_err());
    assert!(bicubic_antialias(&[], 0, 4, 5, 8, 10).is_err());
}

#[test]
fn sinc_matches_torchaudio() {
    // The reference's convolution sums in torch's vectorized order; this
    // module sums in tap order. Both are float32, so the values differ by a
    // few rounding steps of a unit-amplitude signal.
    const TOLERANCE: f32 = 1e-5;
    for case in cases("sinc") {
        let (channels, orig, new, length) = (
            case.fields[0],
            case.fields[1] as u32,
            case.fields[2] as u32,
            case.fields[3],
        );
        let input: Vec<f32> = case
            .input
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let expected: Vec<f32> = case
            .expected
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let actual = sinc_resample(&input, channels, length, orig, new).unwrap();
        assert_eq!(
            actual.len(),
            expected.len(),
            "sinc case {} resampled to another count",
            case.name
        );
        for (index, (value, reference)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (value - reference).abs() <= TOLERANCE,
                "sinc case {} sample {index}: {value} against {reference}",
                case.name
            );
        }
    }
}

#[test]
fn sinc_at_an_unchanged_rate_is_a_copy() {
    let input: Vec<f32> = (0..20).map(|value| value as f32 / 8.0).collect();
    let actual = sinc_resample(&input, 2, 10, 16_000, 16_000).unwrap();
    assert_eq!(actual, input);
}

#[test]
fn sinc_refuses_a_mismatched_extent() {
    let input = vec![0.0f32; 19];
    assert!(sinc_resample(&input, 2, 10, 16_000, 8_000).is_err());
    assert!(sinc_resample(&input, 1, 19, 0, 8_000).is_err());
}
