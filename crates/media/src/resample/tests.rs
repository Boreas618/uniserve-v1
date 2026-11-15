//! The resamplers' identities and refusals. Their agreement with the
//! libraries whose mathematics they implement is tested against checked-in
//! fixtures in `tests/resample.rs`.

use super::{bicubic_antialias_f32, bicubic_antialias_u8, lanczos_resize, sinc_resample};

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
fn bicubic_resizes_to_the_same_size_as_a_copy() {
    let source: Vec<u8> = (0..(2 * 3 * 4)).map(|value| value as u8 * 3).collect();
    let actual = bicubic_antialias_u8(&source, 2, 3, 4, 3, 4).unwrap();
    assert_eq!(actual, source);
}

#[test]
fn bicubic_refuses_a_mismatched_extent() {
    let source = vec![0u8; 3 * 4 * 5 - 1];
    assert!(bicubic_antialias_u8(&source, 3, 4, 5, 8, 10).is_err());
    assert!(bicubic_antialias_u8(&[], 0, 4, 5, 8, 10).is_err());
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

#[test]
fn bicubic_f32_resizes_to_the_same_size_as_a_copy() {
    let source: Vec<f32> = (0..(2 * 3 * 4)).map(|value| value as f32 / 16.0).collect();
    let actual = bicubic_antialias_f32(&source, 2, 3, 4, 3, 4).unwrap();
    assert_eq!(actual, source);
}
