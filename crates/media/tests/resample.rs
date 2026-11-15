//! The host resamplers against the outputs of the libraries whose mathematics
//! they implement.
//!
//! The fixtures under `fixtures/resample/` were written by its `generate.py`
//! with the library versions its metadata pins: Pillow's LANCZOS resize,
//! torchvision's antialiased bicubic resize of uint8 images, torch's
//! antialiased bicubic interpolation of float32 images and torchaudio's sinc
//! resample. Each kind's `cases.txt` names its cases and their extents; each
//! case's input and expected output are raw little-endian bytes in the layout
//! the function takes and returns, embedded here from the files beside it.

// Always true in an integration test crate; it makes clippy treat the helpers
// below as test code, where a failed unwrap is the test's failure report.
#![cfg(test)]

use media::resample::{bicubic_antialias_f32, bicubic_antialias_u8, lanczos_resize, sinc_resample};

/// One embedded case: its name, its input and its expected output.
type Embedded = (&'static str, &'static [u8], &'static [u8]);

/// The case `$name` of the fixture kind `$kind`: `$name.in` and `$name.out`.
macro_rules! case {
    ($kind:literal, $name:literal) => {
        (
            $name,
            include_bytes!(concat!("fixtures/resample/", $kind, "/", $name, ".in")).as_slice(),
            include_bytes!(concat!("fixtures/resample/", $kind, "/", $name, ".out")).as_slice(),
        )
    };
}

const LANCZOS_MANIFEST: &str = include_str!("fixtures/resample/lanczos/cases.txt");
const LANCZOS: &[Embedded] = &[
    case!("lanczos", "upscale"),
    case!("lanczos", "downscale"),
    case!("lanczos", "aspect"),
    case!("lanczos", "wide"),
];

const BICUBIC_MANIFEST: &str = include_str!("fixtures/resample/bicubic/cases.txt");
const BICUBIC: &[Embedded] = &[
    case!("bicubic", "upscale"),
    case!("bicubic", "downscale"),
    case!("bicubic", "aspect"),
    case!("bicubic", "wide"),
    case!("bicubic", "single"),
];

const BICUBIC_F32_MANIFEST: &str = include_str!("fixtures/resample/bicubic_f32/cases.txt");
const BICUBIC_F32: &[Embedded] = &[
    case!("bicubic_f32", "upscale"),
    case!("bicubic_f32", "downscale"),
    case!("bicubic_f32", "aspect"),
    case!("bicubic_f32", "wide"),
    case!("bicubic_f32", "single"),
];

const SINC_MANIFEST: &str = include_str!("fixtures/resample/sinc/cases.txt");
const SINC: &[Embedded] = &[
    case!("sinc", "mono_down"),
    case!("sinc", "stereo_up"),
    case!("sinc", "same"),
    case!("sinc", "ratio"),
];

/// One case: its name, its numeric fields, its input and expected bytes.
struct Case {
    name: &'static str,
    fields: Vec<usize>,
    input: &'static [u8],
    expected: &'static [u8],
}

/// Each manifest line with its embedded files; every case of the manifest is
/// embedded and no other.
fn cases(manifest: &str, embedded: &[Embedded]) -> Vec<Case> {
    let cases: Vec<Case> = manifest
        .lines()
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let (name, input, expected) = *embedded
                .iter()
                .find(|(name, _, _)| *name == fields[0])
                .unwrap_or_else(|| panic!("the fixture case {} is not embedded", fields[0]));
            Case {
                name,
                fields: fields[1..]
                    .iter()
                    .map(|value| value.parse().expect("a numeric field"))
                    .collect(),
                input,
                expected,
            }
        })
        .collect();
    assert_eq!(
        cases.len(),
        embedded.len(),
        "the embedded cases are not the manifest's"
    );
    cases
}

/// The float32 samples of little-endian bytes.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

#[test]
fn lanczos_matches_pillow() {
    for case in cases(LANCZOS_MANIFEST, LANCZOS) {
        let (channels, height, width, out_height, out_width) = (
            case.fields[0],
            case.fields[1],
            case.fields[2],
            case.fields[3],
            case.fields[4],
        );
        assert_eq!(channels, 3, "Pillow resizes RGB images here");
        let actual = lanczos_resize(
            case.input,
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
fn bicubic_matches_torchvision() {
    for case in cases(BICUBIC_MANIFEST, BICUBIC) {
        let (channels, height, width, out_height, out_width) = (
            case.fields[0],
            case.fields[1],
            case.fields[2],
            case.fields[3],
            case.fields[4],
        );
        let actual = bicubic_antialias_u8(
            case.input,
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
fn bicubic_f32_matches_torch() {
    // torch accumulates the float32 convolution in its vectorized order; this
    // crate accumulates in tap order. The weights are float32 in both, so the
    // values differ only by a few rounding steps.
    const TOLERANCE: f32 = 1e-6;
    for case in cases(BICUBIC_F32_MANIFEST, BICUBIC_F32) {
        let (channels, height, width, out_height, out_width) = (
            case.fields[0],
            case.fields[1],
            case.fields[2],
            case.fields[3],
            case.fields[4],
        );
        let input = floats(case.input);
        let expected = floats(case.expected);
        let actual = bicubic_antialias_f32(
            &input,
            channels,
            height as u32,
            width as u32,
            out_height as u32,
            out_width as u32,
        )
        .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (value, reference)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (value - reference).abs() <= TOLERANCE,
                "bicubic_f32 case {} sample {index}: {value} against {reference}",
                case.name
            );
        }
    }
}

#[test]
fn sinc_matches_torchaudio() {
    // torchaudio's convolution sums in torch's vectorized order; this crate
    // sums in tap order. Both are float32, so the values differ by a few
    // rounding steps of a unit-amplitude signal.
    const TOLERANCE: f32 = 1e-5;
    for case in cases(SINC_MANIFEST, SINC) {
        let (channels, orig, new, length) = (
            case.fields[0],
            case.fields[1] as u32,
            case.fields[2] as u32,
            case.fields[3],
        );
        let input = floats(case.input);
        let expected = floats(case.expected);
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
