//! What a stack's rotary embedding launches: one factor program per forward, and per layer either
//! the fused q/k-norm rotation or the rotation alone.
//!
//! A meta loader publishes the spectrum at a synthetic address and a recording forward reports the
//! programs instead of launching them, so nothing here needs a device.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::layers::{Factors, Gains, Precision, QkNorm, Rotary, RotarySpec, Spectrum};
use nn::{Arena, Checkpoint, DType, Forward, Shape, TensorView, WeightLoader};

/// Runs `walk` with a capture sink installed and answers the operators it reported.
fn launched<T>(walk: impl FnOnce() -> T) -> (T, Vec<(String, String)>) {
    #[derive(Default)]
    struct Launches(Vec<(String, String)>);
    impl ops::capture::Sink for Launches {
        fn launched(&mut self, operator: &'static str, key: &str) {
            self.0.push((operator.to_owned(), key.to_owned()));
        }
    }
    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let answer = walk();
    ops::capture::set_sink(None);
    (answer, launches.0)
}

/// A checkpoint of one placeholder tensor: a rotary reads no tensor, it only publishes its table.
fn checkpoint(directory: &std::path::Path) -> Checkpoint {
    let header = r#"{"placeholder":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&[0; 4]);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();
    Checkpoint::open(directory).unwrap()
}

/// The rotary of `spectrum` over `heads` query and `kv_heads` key heads of `head_dim` channels.
fn rotary(
    spectrum: Spectrum,
    head_dim: u32,
    heads: u32,
    kv_heads: u32,
    qk_norm: QkNorm,
) -> nn::Result<Rotary> {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path());
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    Rotary::load(
        &mut loader.scope("rotary_emb"),
        spectrum,
        RotarySpec {
            head_dim,
            heads,
            kv_heads,
            qk_norm,
        },
    )
}

/// The DiT's arrangement: 16 pairs repeated over three axes of a 128-channel head.
fn repeated() -> Spectrum {
    Spectrum::Repeated {
        theta: 10_000.0,
        pairs: 16,
        axes: 3,
        precision: Precision::F64,
    }
}

fn plain(dim: u32) -> Spectrum {
    Spectrum::Default {
        theta: 1_000_000.0,
        dim,
        precision: Precision::F64,
    }
}

const FUSED: QkNorm = QkNorm::Fused {
    epsilon: 1e-6,
    gain: nn::layers::Gain::Plain,
};

/// A rotary's factor tables are one column per rotated pair, and a repeated spectrum reads one
/// coordinate per axis, axis-major: a flat position list is refused rather than read transposed.
#[test]
fn the_factors_read_one_coordinate_per_axis_into_one_column_per_pair() {
    let rotary = rotary(repeated(), 128, 4, 4, FUSED).unwrap();
    assert_eq!(rotary.columns(), 48);
    assert_eq!(rotary.axes(), 3);

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let tensor = |dtype, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
    let (result, launches) = launched(|| rotary.factors(&f, tensor(DType::I32, &[3, 10])));
    let factors = result.unwrap();
    assert_eq!(
        factors.cosine.shape.dims(),
        [10, 48],
        "one column per rotated pair"
    );
    assert_eq!(
        (factors.sine.shape, factors.sine.dtype),
        (factors.cosine.shape, DType::F32)
    );
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].0, "rotary_factors");

    assert!(
        rotary.factors(&f, tensor(DType::I32, &[30])).is_err(),
        "a sectioned rotation reads axis-major positions"
    );
}

/// A fused rotary normalizes and turns both projections in one launch, and a norm with learned
/// gains is another program than one without.
#[test]
fn a_fused_rotary_turns_queries_and_keys_in_one_launch() {
    let rotary = rotary(plain(128), 128, 8, 2, FUSED).unwrap();
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let tensor = |dims: &[u64]| arena.tensor(DType::BF16, Shape::new(dims)).unwrap();
    let factors = || arena.tensor(DType::F32, Shape::new(&[5, 64])).unwrap();
    let (query, key) = (tensor(&[5, 8, 128]), tensor(&[5, 2, 128]));
    let gains = Gains {
        query: tensor(&[128]),
        key: tensor(&[128]),
    };
    let factors = Factors {
        cosine: factors(),
        sine: factors(),
    };
    let turn = |gains: Option<Gains>| launched(|| rotary.forward(&f, query, key, gains, factors));

    let (learned, with_gains) = turn(Some(gains));
    let (turned_query, turned_key) = learned.unwrap();
    assert_eq!(
        (turned_query.shape, turned_key.shape),
        (query.shape, key.shape)
    );
    let (bare, without_gains) = turn(None);
    bare.unwrap();
    assert_eq!(with_gains.len(), 1);
    assert_eq!(with_gains[0].0, "qk_norm_rope");
    assert_eq!(without_gains.len(), 1);
    assert_ne!(with_gains[0].1, without_gains[0].1);

    // The strided form turns two views of one buffer in place, without gains.
    let (strided, launches) = launched(|| rotary.rotate(&f, query, query, factors));
    strided.unwrap();
    assert_eq!(launches[0].0, "qk_norm_rope");

    // Gains of another width are refused before a launch.
    let narrow = Gains {
        query: tensor(&[64]),
        key: tensor(&[64]),
    };
    assert!(
        rotary
            .forward(&f, query, key, Some(narrow), factors)
            .is_err()
    );
}

/// A rotary without the norm turns the queries and then the keys, and refuses gains it would not
/// apply and the strided norm it does not have.
#[test]
fn a_rotary_without_the_norm_turns_each_projection_alone() {
    let rotary = rotary(
        Spectrum::Repeated {
            theta: 10_000.0,
            pairs: 18,
            axes: 2,
            precision: Precision::F32,
        },
        72,
        16,
        16,
        QkNorm::None,
    )
    .unwrap();
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let tensor = |dims: &[u64]| arena.tensor(DType::BF16, Shape::new(dims)).unwrap();
    let factors = || arena.tensor(DType::F32, Shape::new(&[6, 36])).unwrap();
    let (query, key) = (tensor(&[6, 16, 72]), tensor(&[6, 16, 72]));
    let factors = Factors {
        cosine: factors(),
        sine: factors(),
    };
    let (result, launches) = launched(|| rotary.forward(&f, query, key, None, factors));
    result.unwrap();
    let operators: Vec<&str> = launches.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(operators, ["rope_apply", "rope_apply"]);

    let gains = Gains {
        query: tensor(&[72]),
        key: tensor(&[72]),
    };
    assert!(
        rotary
            .forward(&f, query, key, Some(gains), factors)
            .is_err()
    );
    assert!(rotary.rotate(&f, query, key, factors).is_err());
}

/// A rotation wider than the head, or query heads that do not group the key heads, are refused at
/// load.
#[test]
fn a_rotary_refuses_a_head_it_cannot_turn() {
    assert!(rotary(plain(128), 64, 8, 8, FUSED).is_err());
    assert!(rotary(plain(128), 128, 8, 3, FUSED).is_err());
    assert!(rotary(plain(127), 128, 8, 8, FUSED).is_err());
    // A partial rotation passes the trailing channels through.
    assert!(rotary(repeated(), 128, 8, 8, FUSED).is_ok());
}

/// The views a rotation checks: a projection of another head count is refused by name.
#[test]
fn a_projection_of_another_head_count_is_refused() {
    let rotary = rotary(plain(64), 64, 4, 4, FUSED).unwrap();
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view =
        |dims: &[u64]| -> TensorView { arena.tensor(DType::BF16, Shape::new(dims)).unwrap() };
    let table = arena.tensor(DType::F32, Shape::new(&[3, 32])).unwrap();
    let factors = Factors {
        cosine: table,
        sine: table,
    };
    let error = rotary
        .forward(&f, view(&[3, 2, 64]), view(&[3, 4, 64]), None, factors)
        .expect_err("two query heads for a four-head rotary");
    assert!(format!("{error}").contains("query"), "{error}");
}
