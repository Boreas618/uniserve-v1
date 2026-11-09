//! What the media layers launch and refuse: the diffusion step, the tiling drivers, the
//! convolutions and the audio networks, walked on a recording forward.
//!
//! A recording forward has no device: every view addresses a counting arena that no memory backs,
//! and a meta loader reads extents from a hand-written safetensors header, so the tests observe the
//! programs a walk states and the arguments a layer refuses, never a number.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::path::Path;

use nn::codec::{DacEncoder, DacSpec};
use nn::diffusion::{Euler, Grid};
use nn::layers::{CausalPad, Conv1d, ConvSpec, ConvTranspose1d, Padding, Posterior};
use nn::vocoder::{BigVgan, BigVganSpec};
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

/// A convolution loads torch's weight shape for its arguments, folds a weight-norm pair, and refuses
/// a filter its arguments do not state; the transposed form reads torch's `[in, out, k]` filter.
#[test]
fn a_convolution_loads_the_filter_its_torch_arguments_state() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("conv.weight_v", "F32", &[8, 4, 3]),
            ("conv.weight_g", "F32", &[8, 1, 1]),
            ("conv.bias", "F32", &[8]),
            ("up.weight", "F32", &[8, 4, 4]),
            ("up.bias", "F32", &[4]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let spec = ConvSpec {
        padding: [1],
        weight_norm: true,
        ..ConvSpec::new(4, 8, [3])
    };
    let conv = Conv1d::load(&mut loader.scope("conv"), spec).unwrap();
    assert!(
        Conv1d::load(
            &mut loader.scope("conv"),
            ConvSpec {
                out_channels: 16,
                ..spec
            }
        )
        .is_err(),
        "a filter of another width than the spec states"
    );
    assert!(
        Conv1d::load(&mut loader.scope("conv"), ConvSpec { groups: 0, ..spec }).is_err(),
        "no groups is no convolution"
    );
    assert!(
        Conv1d::load(
            &mut loader.scope("conv"),
            ConvSpec {
                channels_last: true,
                ..spec
            }
        )
        .is_err(),
        "a 1-d convolution has no channels-last storage"
    );
    let up = ConvTranspose1d::load(
        &mut loader.scope("up"),
        ConvSpec {
            stride: [2],
            padding: [1],
            ..ConvSpec::new(8, 4, [4])
        },
    )
    .unwrap();

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view = |dims: &[u64]| arena.tensor(DType::F32, Shape::new(dims)).unwrap();
    let (input, out) = (view(&[2, 4, 16]), view(&[2, 8, 16]));
    let ((), launches) = recorded(|| conv.forward(&f, input, out).unwrap());
    assert_eq!(operators(&launches), ["conv1d"]);
    let ((), launches) = recorded(|| up.forward(&f, out, view(&[2, 4, 32])).unwrap());
    assert_eq!(operators(&launches), ["conv_transpose1d"]);
}

/// A causal pad is one `frame_pad` launch, mirroring or repeating the border as its mode states.
#[test]
fn a_causal_pad_is_one_frame_pad_launch() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view = |dims: &[u64]| arena.tensor(DType::F32, Shape::new(dims)).unwrap();
    let (values, padded) = (view(&[1, 3, 5, 16, 16]), view(&[1, 3, 7, 18, 18]));
    let launches = [Padding::Reflect, Padding::Replicate].map(|padding| {
        let pad = CausalPad {
            top: 1,
            left: 1,
            front: 2,
            padding,
        };
        recorded(|| pad.forward(&f, values, padded).unwrap()).1
    });
    assert_eq!(operators(&launches[0]), ["frame_pad"]);
    assert_ne!(launches[0], launches[1], "the border mode is compiled in");
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

/// The weight-normalized module `name` of a 1-d convolution: its `weight_v`, `weight_g` and,
/// with `bias`, its bias of the filter's output channels.
fn normalized_conv(name: &str, filter: [u64; 3], bias: Option<u64>) -> Vec<(String, Vec<u64>)> {
    let mut tensors = vec![
        (format!("{name}.weight_v"), filter.to_vec()),
        (format!("{name}.weight_g"), vec![filter[0], 1, 1]),
    ];
    if let Some(channels) = bias {
        tensors.push((format!("{name}.bias"), vec![channels]));
    }
    tensors
}

/// An alias-free activation module of `channels` and its 12-tap filters.
fn activation(name: &str, channels: u64) -> Vec<(String, Vec<u64>)> {
    vec![
        (format!("{name}.act.alpha"), vec![channels]),
        (format!("{name}.act.beta"), vec![channels]),
        (format!("{name}.upsample.filter"), vec![1, 1, 12]),
        (format!("{name}.downsample.lowpass.filter"), vec![1, 1, 12]),
    ]
}

/// A one-stage BigVGAN loads its modules by the generator's names and runs pre-convolution,
/// upsample, the block's dilation steps, the post activation and the post-convolution; the output
/// is the input length times the stage rates.
#[test]
fn a_bigvgan_generator_upsamples_through_its_stages() {
    let mut tensors = normalized_conv("decoder.conv_pre", [8, 4, 7], Some(8));
    tensors.extend(normalized_conv("decoder.ups.0.0", [8, 4, 4], Some(4)));
    for step in 0..2 {
        let block = "decoder.resblocks.0";
        tensors.extend(normalized_conv(
            &format!("{block}.convs1.{step}"),
            [4, 4, 3],
            Some(4),
        ));
        tensors.extend(normalized_conv(
            &format!("{block}.convs2.{step}"),
            [4, 4, 3],
            Some(4),
        ));
        tensors.extend(activation(&format!("{block}.activations.{}", 2 * step), 4));
        tensors.extend(activation(
            &format!("{block}.activations.{}", 2 * step + 1),
            4,
        ));
    }
    tensors.extend(activation("decoder.activation_post", 4));
    tensors.extend(normalized_conv("decoder.conv_post", [1, 4, 7], None));
    let header: Vec<(&str, &str, &[u64])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), "F32", shape.as_slice()))
        .collect();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), &header);
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let spec = BigVganSpec {
        in_channels: 4,
        channels: 8,
        upsample_rates: vec![2],
        upsample_kernels: vec![4],
        resblock_kernels: vec![3],
        resblock_dilations: vec![vec![1, 3]],
        activation_ratio: 2,
        final_bias: nn::layers::Bias::Absent,
    };
    let generator = BigVgan::load(&mut loader.scope("decoder"), &spec).unwrap();
    assert_eq!(generator.samples(16), 32);
    let odd = BigVganSpec {
        upsample_kernels: vec![4, 4],
        ..spec.clone()
    };
    assert!(BigVgan::load(&mut loader.scope("decoder"), &odd).is_err());

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view = |dims: &[u64]| arena.tensor(DType::F32, Shape::new(dims)).unwrap();
    let input = view(&[2, 4, 16]);
    let ((), launches) = recorded(|| generator.forward(&f, input, view(&[2, 1, 32])).unwrap());
    let count = |name: &str| launches.iter().filter(|(op, _)| op == name).count();
    // Two dilation steps of two activations each, then the post activation, each once per batch
    // row; the block's four convolutions between the pre- and post-convolutions.
    assert_eq!(count("audio_activation"), 2 * (2 * 2 + 1));
    assert_eq!(count("conv1d"), 1 + 4 + 1);
    assert_eq!(count("conv_transpose1d"), 1);
    assert_eq!(launches.first().map(|(op, _)| op.as_str()), Some("conv1d"));
    assert!(generator.forward(&f, input, view(&[2, 1, 16])).is_err());
}

/// A DAC encoder loads `encoder.block`'s numbered modules, runs three Snake residual units and a
/// widening convolution per stage, and refuses a track that is not a whole number of hops.
#[test]
fn a_dac_encoder_widens_and_strides_through_its_stages() {
    let mut tensors = normalized_conv("enc.0", [2, 1, 7], Some(2));
    for unit in 0..3 {
        let base = format!("enc.1.block.{unit}.block");
        tensors.extend(normalized_conv(&format!("{base}.1"), [2, 2, 7], Some(2)));
        tensors.extend(normalized_conv(&format!("{base}.3"), [2, 2, 1], Some(2)));
        tensors.push((format!("{base}.0.alpha"), vec![2]));
        tensors.push((format!("{base}.2.alpha"), vec![2]));
    }
    tensors.push(("enc.1.block.3.alpha".to_owned(), vec![2]));
    tensors.extend(normalized_conv("enc.1.block.4", [4, 2, 4], Some(4)));
    tensors.push(("enc.2.alpha".to_owned(), vec![4]));
    tensors.extend(normalized_conv("enc.3", [8, 4, 3], Some(8)));
    let header: Vec<(&str, &str, &[u64])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), "F32", shape.as_slice()))
        .collect();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), &header);
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let spec = DacSpec {
        channels: 2,
        strides: vec![2],
        latent_channels: 8,
    };
    let encoder = DacEncoder::load(&mut loader.scope("enc"), &spec).unwrap();
    assert_eq!(encoder.hop(), 2);
    let no_stage = DacSpec {
        strides: Vec::new(),
        ..spec
    };
    assert!(DacEncoder::load(&mut loader.scope("enc"), &no_stage).is_err());

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view = |dims: &[u64]| arena.tensor(DType::F32, Shape::new(dims)).unwrap();
    let ((), launches) = recorded(|| {
        encoder
            .forward(&f, view(&[2, 1, 16]), view(&[2, 8, 8]))
            .unwrap()
    });
    let count = |name: &str| launches.iter().filter(|(op, _)| op == name).count();
    assert_eq!(count("snake"), 3 * 2 + 1 + 1);
    assert_eq!(count("conv1d"), 1 + 3 * 2 + 1 + 1);
    assert!(
        encoder
            .forward(&f, view(&[2, 1, 15]), view(&[2, 8, 7]))
            .is_err()
    );
}

/// A posterior sample reads the mean and log-variance as the halves of one channel-major moments
/// raster and draws a sample of half its channels, in one launch; rasters of another width, type,
/// extent or storage are refused.
#[test]
fn a_posterior_sample_reads_the_two_halves_of_its_moments() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let raster = |channels: u64, frames: u64, dtype| {
        arena
            .tensor(dtype, Shape::new(&[channels, frames, 6, 10]))
            .unwrap()
    };
    let moments = raster(8, 3, DType::F32);
    let (noise, out) = (raster(4, 3, DType::F32), raster(4, 3, DType::F32));
    let ((), launches) = recorded(|| Posterior.sample(&f, moments, noise, out).unwrap());
    assert_eq!(operators(&launches), ["posterior"]);

    // Moments of the sample's own width have no log-variance; a draw of their width is two draws.
    assert!(Posterior.sample(&f, noise, noise, out).is_err());
    assert!(Posterior.sample(&f, moments, moments, out).is_err());
    // A half-precision draw, another clip's draw, and planes that overlap.
    assert!(
        Posterior
            .sample(&f, moments, raster(4, 3, DType::F16), out)
            .is_err()
    );
    assert!(
        Posterior
            .sample(&f, moments, raster(4, 2, DType::F32), out)
            .is_err()
    );
    let overlapping = TensorView::with_strides(
        noise.ptr,
        DType::F32,
        noise.shape,
        nn::Strides::new(&[60, 60, 10, 1]),
    );
    assert!(Posterior.sample(&f, moments, overlapping, out).is_err());
}
