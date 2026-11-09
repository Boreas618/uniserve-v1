//! What the media layers launch and refuse: the diffusion step, the tiling drivers, the
//! convolutions and the audio networks, walked on a recording forward.
//!
//! A recording forward has no device: every view addresses a counting arena that no memory backs,
//! and a meta loader reads extents from a hand-written safetensors header, so the tests observe the
//! programs a walk states and the arguments a layer refuses, never a number.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::path::Path;

use nn::diffusion::{Euler, Grid};
use nn::{Arena, Checkpoint, DType, Forward, Shape, TensorView, WeightLoader};

/// The programs a walk reported, as `(operator, key)` pairs.
#[derive(Default)]
struct Launches(Vec<(String, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

/// Runs `walk` with a capture sink installed and answers what it reported.
fn recorded<T>(walk: impl FnOnce() -> T) -> (T, Vec<(String, String)>) {
    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let answer = walk();
    ops::capture::set_sink(None);
    (answer, launches.0)
}

/// The operator names of a walk's launches, in order.
fn operators(launches: &[(String, String)]) -> Vec<&str> {
    launches
        .iter()
        .map(|(operator, _)| operator.as_str())
        .collect()
}

/// A checkpoint in `directory` holding `tensors`, each `(name, dtype name, extents)` of zeros.
fn checkpoint(directory: &Path, tensors: &[(&str, &str, &[u64])]) -> Checkpoint {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (index, (name, dtype, shape)) in tensors.iter().enumerate() {
        let element = match *dtype {
            "F32" | "I32" => 4,
            "F16" | "BF16" => 2,
            other => panic!("the fixture writes no {other} tensor"),
        };
        let bytes = shape.iter().product::<u64>() * element;
        if index > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[{offset},{}]}}"#,
            offset + bytes
        ));
        offset += bytes;
    }
    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + offset as usize, 0);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();
    Checkpoint::open(directory).unwrap()
}

/// A checkpoint whose only tensor is a placeholder: for layers that publish constants alone.
fn empty_checkpoint(directory: &Path) -> Checkpoint {
    checkpoint(directory, &[("placeholder", "F32", &[1])])
}

/// The Euler step gathers its coefficient by the step index and adds the scaled velocity in place;
/// a schedule that scales its draw launches one more add, and one that starts from unit noise none.
#[test]
fn an_euler_step_gathers_its_coefficient_and_adds_the_scaled_velocity() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = empty_checkpoint(directory.path());
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let sigmas = Grid::Uniform { points: 5 }.sigmas(3.0);
    let euler = Euler::load(&mut loader.scope("denoiser.video"), &sigmas, None).unwrap();
    let scaled = Euler::load(&mut loader.scope("denoiser.audio"), &sigmas, Some(0.999)).unwrap();
    assert_eq!(euler.steps(), 4);
    assert!(Euler::load(&mut loader.scope("one"), &[1.0], None).is_err());

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let latents = |dtype| arena.tensor(dtype, Shape::new(&[16, 3, 4, 4])).unwrap();
    let (sample, velocity) = (latents(DType::F32), latents(DType::F32));
    let step = arena.tensor(DType::I32, Shape::new(&[1])).unwrap();
    let gathered = arena.tensor(DType::F32, Shape::new(&[1, 1])).unwrap();

    let (result, launches) = recorded(|| euler.forward(&f, sample, velocity, step, gathered));
    result.unwrap();
    assert_eq!(operators(&launches), ["logits_gather", "scaled_residual"]);

    let (result, launches) = recorded(|| euler.prepare_latents(&f, sample));
    result.unwrap();
    assert!(launches.is_empty(), "a unit-noise schedule scales nothing");
    let (result, launches) = recorded(|| scaled.prepare_latents(&f, sample));
    result.unwrap();
    assert_eq!(operators(&launches), ["scaled_residual"]);

    // A velocity of another type or extent, and a strided sample, are refused before any launch.
    let narrow = latents(DType::BF16);
    assert!(euler.forward(&f, sample, narrow, step, gathered).is_err());
    let short = arena.tensor(DType::F32, Shape::new(&[8, 3, 4, 4])).unwrap();
    assert!(euler.forward(&f, sample, short, step, gathered).is_err());
    let strided = TensorView::with_strides(
        sample.ptr,
        DType::F32,
        Shape::new(&[16, 3, 4, 2]),
        nn::Strides::new(&[48, 16, 4, 2]),
    );
    assert!(scaled.prepare_latents(&f, strided).is_err());
}
