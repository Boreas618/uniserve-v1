//! What the shared blocks launch and reserve: each block is loaded by a meta loader from a
//! checkpoint of zero tensors and walked on a recording forward, so the programs it states and the
//! arena it leaves behind are observable without a device.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::blocks::{GateUp, GatedMlp, GatedMlpSpec};
use nn::family::Split;
use nn::layers::Rounding;
use nn::{Arena, Checkpoint, DType, Forward, Shape, TensorView, WeightLoader};
use ops::activation::Activation;

/// Runs `walk` with a capture sink installed and answers the operators it reported, in order.
fn launched<T>(walk: impl FnOnce() -> T) -> (T, Vec<String>) {
    #[derive(Default)]
    struct Launches(Vec<String>);
    impl ops::capture::Sink for Launches {
        fn launched(&mut self, operator: &'static str, _key: &str) {
            self.0.push(operator.to_owned());
        }
    }
    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let answer = walk();
    ops::capture::set_sink(None);
    (answer, launches.0)
}

/// A checkpoint in `directory` holding BF16 zero tensors of the given names and extents.
fn checkpoint(directory: &std::path::Path, tensors: &[(&str, &[u64])]) -> Checkpoint {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<u64>() * 2;
        let extents: Vec<String> = shape.iter().map(u64::to_string).collect();
        header.push_str(&format!(
            r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{offset},{}]}},"#,
            extents.join(","),
            offset + bytes
        ));
        offset += bytes;
    }
    header.pop();
    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + offset as usize, 0);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();
    Checkpoint::open(directory).unwrap()
}

/// A view of `dims` BF16 elements reserved from `arena`.
fn view(arena: &Arena, dims: &[u64]) -> TensorView {
    arena.tensor(DType::BF16, Shape::new(dims)).unwrap()
}

const HIDDEN: u64 = 64;
const INTERMEDIATE: u64 = 128;

fn mlp_spec(rounding: Rounding, gate_up: GateUp<'_>) -> GatedMlpSpec<'_> {
    GatedMlpSpec {
        hidden: HIDDEN as u32,
        intermediate: INTERMEDIATE as u32,
        activation: Activation::Silu,
        rounding,
        gate_up,
        down: "down_proj",
    }
}

const SPLIT: GateUp<'static> = GateUp::Split {
    gate: "gate_proj",
    up: "up_proj",
};

/// A split gated MLP projects twice, gates at its rounding, projects back, and leaves the arena
/// as it found it: its intermediates are its own.
#[test]
fn a_split_gated_mlp_gates_at_its_rounding_and_releases_its_intermediates() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("mlp.gate_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let (x, out) = (view(&arena, &[5, HIDDEN]), view(&arena, &[5, HIDDEN]));

    for (rounding, gate) in [
        (Rounding::ValueProduct, "swiglu"),
        (Rounding::Fused, "act_and_mul"),
    ] {
        let mlp = GatedMlp::load(
            &mut loader.scope("mlp"),
            mlp_spec(rounding, SPLIT),
            Split::WHOLE,
        )
        .unwrap();
        let used = arena.used();
        let (result, launches) = launched(|| mlp.forward(&f, x, out));
        result.unwrap();
        assert_eq!(launches, ["gemm", "gemm", gate, "gemm"], "{rounding:?}");
        assert_eq!(arena.used(), used, "the intermediates are released");
    }
}

/// A value-first projection is read once and gated from its two halves.
#[test]
fn a_value_first_gated_mlp_reads_one_packed_projection() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("ff.proj.weight", &[2 * INTERMEDIATE, HIDDEN]),
            ("ff.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let mlp = GatedMlp::load(
        &mut loader.scope("ff"),
        mlp_spec(Rounding::Stepwise, GateUp::ValueFirst("proj")),
        Split::WHOLE,
    )
    .unwrap();
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let x = view(&arena, &[3, HIDDEN]);
    let (out, launches) = launched(|| mlp.apply(&f, x));
    assert_eq!(out.unwrap().shape.dims(), &[3, HIDDEN]);
    assert_eq!(launches, ["gemm", "act_and_mul", "gemm"]);
}

/// Over a tensor-parallel group the block holds its rank's share of the gated width and reduces in
/// its down projection, which a forward without a group refuses.
#[test]
fn a_tensor_parallel_gated_mlp_reduces_in_its_down_projection() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("mlp.gate_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let mlp = GatedMlp::load(
        &mut loader.scope("mlp"),
        mlp_spec(Rounding::ValueProduct, SPLIT),
        Split { size: 2, rank: 1 },
    )
    .unwrap();
    let arena = Arena::counting();
    let (x, out) = (view(&arena, &[4, HIDDEN]), view(&arena, &[4, HIDDEN]));

    let split = Forward::recording(&arena).with_split(1, 2).unwrap();
    assert!(mlp.forward(&split, x, out).is_ok());
    assert!(
        mlp.forward(&Forward::recording(&arena), x, out).is_err(),
        "a row-parallel projection with no group to reduce over"
    );

    // A split that does not divide the gated width is refused at load.
    assert!(
        GatedMlp::load(
            &mut loader.scope("mlp"),
            mlp_spec(Rounding::ValueProduct, SPLIT),
            Split { size: 3, rank: 0 },
        )
        .is_err()
    );
}
