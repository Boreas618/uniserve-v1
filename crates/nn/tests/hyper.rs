//! The hyper-connected residual (mHC) and the hashed n-gram memory on the CPU device: each
//! function computes its reference formula ([`nn_testing::reference`], FP64 with the reference's
//! roundings) within its operator's bound, the residual reads every sub-layer with the pre-mix
//! the connection before it computed, the memory composes its hash, its table and its gate, and a
//! profile run states the keys a device launches.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::functional::{self, MhcMixes, MhcMixing, WeightOrder};
use nn::modules::{
    Linear, MhcConfig, MhcConnection, MhcResidual, NgramConfig, NgramHashing, NgramMemory,
    NgramTable, TableShard, Tokens,
};
use nn::{CpuWeights, DType, Device, Tensor, WeightLoader};
use nn_testing::reference::{self, Mixes, Mixing, assert_within, narrow};
use nn_testing::{Stored, encode, sequence, tensor, values, write_checkpoint};
use ops::Op;
use ops::mhc::{BoundaryDescriptor, MhcBoundary, MhcPost, MhcPre, PostDescriptor, PreDescriptor};
use ops::ngram::{GateDescriptor, NgramGate};
use ops::tolerance::{self, Tolerance};

/// Four streams of 64 elements over five tokens, as the model's connections are four streams.
const TOKENS: usize = 5;
const STREAMS: usize = 4;
const HIDDEN: usize = 64;
const MIXES: usize = STREAMS * (STREAMS + 2);

const CONFIG: MhcConfig = MhcConfig {
    streams: STREAMS as u32,
    hidden: HIDDEN as u32,
    iterations: 20,
    eps: 1e-6,
    rms_eps: 1e-20,
    norm_eps: 1e-20,
};

/// A connection's FP32 parameters drawn from `seed`, scaled as a checkpoint's are: a projection
/// of a few units, scales of a tenth, biases of a few units.
fn parameters(seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let projection = sequence(MIXES * STREAMS * HIDDEN, seed)
        .iter()
        .map(|x| 4.0 * x)
        .collect();
    let scale = sequence(3, seed + 1)
        .iter()
        .map(|x| 0.1 + 0.05 * x)
        .collect();
    let base = sequence(MIXES, seed + 2).iter().map(|x| 3.0 * x).collect();
    let norm = sequence(HIDDEN, seed + 3)
        .iter()
        .map(|x| 1.0 + 0.5 * x)
        .collect();
    (projection, scale, base, norm)
}

fn connection(seed: u32) -> MhcConnection {
    let (projection, scale, base, norm) = parameters(seed);
    MhcConnection::new(
        &CONFIG,
        tensor(&projection, (MIXES, STREAMS * HIDDEN), DType::F32),
        tensor(&scale, 3, DType::F32),
        tensor(&base, MIXES, DType::F32),
        tensor(&norm, HIDDEN, DType::BF16),
    )
    .unwrap()
}

/// The streams of `seed` in BF16, as values and as the tensor.
fn streams(seed: u32) -> (Vec<f32>, Tensor) {
    let h = tensor(
        &sequence(TOKENS * STREAMS * HIDDEN, seed),
        (TOKENS, STREAMS, HIDDEN),
        DType::BF16,
    );
    (values(&h), h)
}

fn shape() -> (usize, usize, usize) {
    (TOKENS, STREAMS, HIDDEN)
}

/// Mixes of the library as the ports hold them.
fn ported(mixes: &MhcMixes) -> Mixes {
    let wide = |tensor: &Tensor| values(tensor).into_iter().map(f64::from).collect();
    Mixes {
        pre: wide(&mixes.pre),
        post: wide(&mixes.post),
        comb: wide(&mixes.comb),
    }
}

fn boundary_tolerance(post: bool) -> Tolerance {
    MhcBoundary::tolerance(&BoundaryDescriptor::new(
        DType::BF16,
        STREAMS as u32,
        HIDDEN as u32,
        20,
        post,
        TOKENS as u64,
    ))
}

#[test]
fn a_boundary_writes_mixes_and_reads_as_the_reference_formulas_do() {
    let (h, streams) = streams(1);
    let (projection, scale, base, norm) = parameters(10);
    let mixing = Mixing {
        projection: &projection,
        scale: &scale,
        base: &base,
        iterations: 20,
        rms_eps: 1e-20,
        eps: 1e-6,
    };
    let output = tensor(&sequence(TOKENS * HIDDEN, 2), (TOKENS, HIDDEN), DType::BF16);
    let carried = tensor(
        &sequence(TOKENS * STREAMS, 3),
        (TOKENS, STREAMS),
        DType::F32,
    )
    .affine(0.5, 0.5)
    .unwrap();
    let pending = functional::mhc_boundary(
        &streams,
        None,
        &carried,
        connection(20).mixing(),
        connection(20).norm(),
        1e-20,
    )
    .unwrap()
    .2;

    let module = connection(10);
    let carried_values: Vec<f64> = values(&carried).into_iter().map(f64::from).collect();
    for written in [false, true] {
        let (reached, input, mixes) = functional::mhc_boundary(
            &streams,
            written.then_some((&output, &pending)),
            &carried,
            module.mixing(),
            module.norm(),
            1e-20,
        )
        .unwrap();

        // The streams the boundary reaches: the output written back, or the streams.
        let expected = if written {
            reference::post(
                &h,
                &values(&output),
                &ported(&pending),
                shape(),
                DType::BF16,
            )
        } else {
            h.clone()
        };
        let post = MhcPost::tolerance(&PostDescriptor::new(DType::BF16, 4, HIDDEN as u32));
        assert_within("the reached streams", &values(&reached), &expected, post);

        // The mixes of the reached streams and the input read with the carried pre-mix.
        let reached = values(&reached);
        let ports = reference::mixes(&reached, TOKENS, STREAMS, HIDDEN, mixing);
        let tolerance = boundary_tolerance(written);
        assert_within(
            "the pre-mix",
            &values(&mixes.pre),
            &narrow(&ports.pre),
            tolerance,
        );
        assert_within(
            "the post-mix",
            &values(&mixes.post),
            &narrow(&ports.post),
            tolerance,
        );
        assert_within(
            "the combination",
            &values(&mixes.comb),
            &narrow(&ports.comb),
            tolerance,
        );
        let expected = reference::pre(
            &reached,
            &carried_values,
            &values(&tensor(&norm, HIDDEN, DType::BF16)),
            shape(),
            1e-20,
            DType::BF16,
        );
        assert_within("the input", &values(&input), &expected, tolerance);

        // A combination matrix is doubly stochastic to within the Sinkhorn steps' epsilon.
        let comb = values(&mixes.comb);
        for token in 0..TOKENS {
            for k in 0..STREAMS {
                let column: f32 = (0..STREAMS)
                    .map(|j| comb[(token * STREAMS + j) * STREAMS + k])
                    .sum();
                assert!(
                    (column - 1.0).abs() < 1e-4,
                    "column {k} of token {token} sums to {column}"
                );
            }
        }
    }
}

#[test]
fn a_read_alone_and_a_write_alone_are_the_reference_formulas() {
    let (h, streams) = streams(4);
    let output = tensor(&sequence(TOKENS * HIDDEN, 5), (TOKENS, HIDDEN), DType::BF16);
    let mixes = functional::mhc_boundary(
        &streams,
        None,
        &Tensor::ones((TOKENS, STREAMS), DType::F32, &Device::Cpu).unwrap(),
        connection(30).mixing(),
        connection(30).norm(),
        1e-20,
    )
    .unwrap()
    .2;

    let written = functional::mhc_post(&streams, &output, &mixes).unwrap();
    let expected = reference::post(&h, &values(&output), &ported(&mixes), shape(), DType::BF16);
    let post = MhcPost::tolerance(&PostDescriptor::new(DType::BF16, 4, HIDDEN as u32));
    assert_within("the written streams", &values(&written), &expected, post);

    let norm = tensor(&sequence(HIDDEN, 6), HIDDEN, DType::BF16);
    let read = functional::mhc_pre(&streams, &mixes.pre, &norm, 1e-6).unwrap();
    let expected = reference::pre(
        &h,
        &ported(&mixes).pre,
        &values(&norm),
        shape(),
        1e-6,
        DType::BF16,
    );
    let pre = MhcPre::tolerance(&PreDescriptor::new(DType::BF16, 4, HIDDEN as u32));
    assert_within("the read row", &values(&read), &expected, pre);
}

#[test]
fn the_residual_reads_each_sublayer_with_the_pre_mix_of_the_connection_before() {
    let embedded = tensor(
        &sequence(TOKENS * HIDDEN, 40),
        (TOKENS, HIDDEN),
        DType::BF16,
    );
    let first = connection(41);
    let second = connection(42);
    // A sub-layer that halves its input: exact in BF16, so the ports and the module see the
    // same outputs of the same inputs.
    let body = |input: &Tensor| -> nn::Result<Tensor> { Ok(input.affine(0.5, 0.0)?) };

    let mut residual = MhcResidual::expand(&CONFIG, &embedded).unwrap();
    residual.sublayer(&first, body).unwrap();
    residual.sublayer(&second, body).unwrap();
    let norm = tensor(&sequence(HIDDEN, 50), HIDDEN, DType::BF16);
    let collapsed = residual.collapse(&norm, 1e-20).unwrap();

    let mut ported = reference::Residual::expand(&values(&embedded), shape(), DType::BF16);
    for seed in [41, 42] {
        let (projection, scale, base, norm) = parameters(seed);
        let mixing = Mixing {
            projection: &projection,
            scale: &scale,
            base: &base,
            iterations: 20,
            rms_eps: 1e-20,
            eps: 1e-6,
        };
        let norm = values(&tensor(&norm, HIDDEN, DType::BF16));
        ported.sublayer(mixing, &norm, 1e-20, |input| {
            input.iter().map(|x| 0.5 * x).collect()
        });
    }
    let expected = ported.collapse(&values(&norm), 1e-20);
    // Eight roundings into BF16 lie on the collapsed row's path (two for each read, one for each
    // write, two for the collapse), and an FP32 computation may round each to the neighbour of
    // the FP64 one's: each such difference is at most two units of the element, which the
    // doubly stochastic writes carry on with a gain of at most one.
    let bound = Tolerance {
        atol: tolerance::underflow(DType::BF16) as f32,
        rtol: 16.0 * tolerance::unit_roundoff(DType::BF16) as f32,
        cancels: true,
    };
    assert_within(
        "the collapsed residual",
        &values(&collapsed),
        &expected,
        bound,
    );
}

#[test]
fn a_connection_loads_its_mixing_parameters_in_fp32() {
    let directory = tempfile::tempdir().unwrap();
    let (projection, scale, base, norm) = parameters(60);
    let shard: &[Stored<&str>] = &[
        (
            "layers.0.hc_attn_fn",
            DType::F32,
            vec![MIXES, STREAMS * HIDDEN],
            encode(DType::F32, &projection),
        ),
        (
            "layers.0.hc_attn_scale",
            DType::F32,
            vec![3],
            encode(DType::F32, &scale),
        ),
        (
            "layers.0.hc_attn_base",
            DType::F32,
            vec![MIXES],
            encode(DType::F32, &base),
        ),
        (
            "layers.0.attn_norm.weight",
            DType::BF16,
            vec![HIDDEN],
            encode(DType::BF16, &norm),
        ),
    ];
    write_checkpoint(directory.path(), &[shard]);
    let checkpoint = nn::Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, Some(DType::BF16));
    let layer = loader.root().pp("layers.0");

    let loaded = MhcConnection::load(&CONFIG, &layer, "hc_attn", layer.pp("attn_norm")).unwrap();
    let mixing = loaded.mixing();
    assert_eq!(mixing.projection.dtype(), DType::F32);
    assert_eq!(values(mixing.projection), projection);
    assert_eq!(values(mixing.base), base);
    assert_eq!(loaded.norm().dtype(), DType::BF16);
}

/// A table of `rows` MXFP8 rows of 32 values: E4M3 bytes that are numbers, and scales around
/// one.
fn table(rows: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
    let values = sequence(rows * 32, seed)
        .iter()
        .map(|x| {
            let byte = ((x + 1.0) * 127.5) as u8;
            // 0x7F and 0xFF are not numbers in E4M3.
            if byte & 0x7F == 0x7F { byte - 1 } else { byte }
        })
        .collect();
    let scales = sequence(rows, seed + 1)
        .iter()
        .map(|x| (127.0 + 4.0 * x) as u8)
        .collect();
    (values, scales)
}

fn bytes(values: &[u8], dims: (usize, usize), dtype: DType) -> Tensor {
    Tensor::from_vec(values.to_vec(), dims, &Device::Cpu)
        .unwrap()
        .view(dtype)
        .unwrap()
}

#[test]
fn a_table_shard_dequantizes_the_rows_it_holds_and_zeros_the_others() {
    let (values_bytes, scales) = table(10, 70);
    let ids = Tensor::new(&[3_i64, 2, 11, 12, 0, 5, 7], &Device::Cpu).unwrap();
    let first = 2;
    let rows = functional::embedding_mxfp8(
        &ids,
        &bytes(&values_bytes, (10, 32), DType::F8E4M3),
        &bytes(&scales, (10, 1), DType::U8),
        first,
        DType::BF16,
    )
    .unwrap();

    let expected = reference::mxfp8(
        &values_bytes,
        &scales,
        32,
        &[3, 2, 11, 12, 0, 5, 7],
        first as i64,
    );
    assert_eq!(values(&rows), expected);
    assert!(values(&rows)[3 * 32..5 * 32].iter().all(|x| *x == 0.0));
}

/// A hashing of contexts of four tokens into two heads of each length over a vocabulary of
/// 50, and its tables.
fn hashing() -> (NgramHashing, Vec<i32>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let normalize: Vec<i32> = (0..50).map(|id| id * 7 % 23).collect();
    let multipliers = vec![3_i64, 1_000_003, 77_777_777_777, 5];
    let moduli = vec![11_i64, 13, 17, 19, 23, 29];
    let offsets = vec![0_i64, 11, 24, 41, 60, 83];
    let hashing = NgramHashing::new(
        Tensor::from_vec(normalize.clone(), 50, &Device::Cpu).unwrap(),
        Tensor::from_vec(multipliers.clone(), 4, &Device::Cpu).unwrap(),
        Tensor::from_vec(moduli.clone(), (3, 2), &Device::Cpu).unwrap(),
        Tensor::from_vec(offsets.clone(), (3, 2), &Device::Cpu).unwrap(),
    )
    .unwrap();
    (hashing, normalize, multipliers, moduli, offsets)
}

#[test]
fn the_hash_names_the_reference_rows() {
    let (hashing, normalize, multipliers, moduli, offsets) = hashing();
    // The first token's context is filled before the request; the last id is past the table.
    let context: Vec<u32> = vec![
        2, 2, 2, 9, 2, 2, 9, 31, 2, 9, 31, 4, 9, 31, 4, 49, 31, 4, 49, 60,
    ];
    let rows = hashing
        .rows(&Tensor::from_vec(context.clone(), (5, 4), &Device::Cpu).unwrap())
        .unwrap();

    let expected = reference::hash(&context, 4, &normalize, &multipliers, &moduli, &offsets);
    assert_eq!(rows.dims(), [5, 6]);
    assert_eq!(
        rows.flatten_all().unwrap().to_vec1::<i64>().unwrap(),
        expected
    );
}

#[test]
fn the_memory_gates_its_recalled_rows_into_every_stream_as_the_reference_does() {
    let (hashing, normalize, multipliers, moduli, offsets) = hashing();
    let config = NgramConfig {
        order: 4,
        heads: 2,
        row: 32,
        streams: STREAMS as u32,
        hidden: HIDDEN as u32,
        eps: 1e-20,
    };
    let (table_bytes, scales) = table(112, 80);
    let table = NgramTable::new(
        bytes(&table_bytes, (112, 32), DType::F8E4M3),
        bytes(&scales, (112, 1), DType::U8),
        0,
        TableShard::Whole,
    )
    .unwrap();
    let width = (STREAMS + 1) * HIDDEN;
    let projection = sequence(width * 6 * 32, 81)
        .iter()
        .map(|x| 0.05 * x)
        .collect::<Vec<_>>();
    let query = sequence(STREAMS * HIDDEN, 82)
        .iter()
        .map(|x| 1.0 + 0.2 * x)
        .collect::<Vec<_>>();
    let key = sequence(STREAMS * HIDDEN, 83)
        .iter()
        .map(|x| 1.0 + 0.2 * x)
        .collect::<Vec<_>>();
    let memory = NgramMemory::new(
        &config,
        hashing,
        table,
        Linear::new(
            tensor(&projection, (width, 6 * 32), DType::BF16),
            None,
            WeightOrder::NxK,
        )
        .unwrap(),
        tensor(&query, (STREAMS, HIDDEN), DType::BF16),
        tensor(&key, (STREAMS, HIDDEN), DType::BF16),
    )
    .unwrap();

    let context: Vec<u32> = vec![
        2, 2, 2, 9, 2, 2, 9, 31, 2, 9, 31, 4, 9, 31, 4, 49, 31, 4, 49, 7,
    ];
    let (h, streams) = streams(84);
    let gated = memory
        .forward(
            &streams,
            &Tensor::from_vec(context.clone(), (5, 4), &Device::Cpu).unwrap(),
            Tokens::Shared,
        )
        .unwrap();

    // The reference: the rows, their values, the projection stored in BF16, then the gate.
    let rows = reference::hash(&context, 4, &normalize, &multipliers, &moduli, &offsets);
    let recalled = reference::round(
        DType::BF16,
        &reference::mxfp8(&table_bytes, &scales, 32, &rows, 0)
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
    );
    let weight = values(&tensor(&projection, (width, 6 * 32), DType::BF16));
    let projected: Vec<f64> = (0..TOKENS)
        .flat_map(|token| {
            let row = &recalled[token * 192..(token + 1) * 192];
            let weight = &weight;
            (0..width).map(move |output| {
                row.iter()
                    .zip(&weight[output * 192..(output + 1) * 192])
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>()
            })
        })
        .collect();
    let projected = reference::round(DType::BF16, &projected);
    let expected = reference::gate(
        &h,
        &projected,
        &values(&tensor(&query, (STREAMS, HIDDEN), DType::BF16)),
        &values(&tensor(&key, (STREAMS, HIDDEN), DType::BF16)),
        shape(),
        1e-20,
        DType::BF16,
    );
    // The gate's bound, and one rounding of the projected value it adds, which an FP32 product
    // may round to the neighbour of the FP64 one's.
    let gate = NgramGate::tolerance(&GateDescriptor::new(DType::BF16, 4, HIDDEN as u32));
    let bound = Tolerance {
        rtol: gate.rtol + 2.0 * tolerance::unit_roundoff(DType::BF16) as f32,
        ..gate
    };
    assert_within("the gated streams", &values(&gated), &expected, bound);
}

#[test]
fn a_profile_run_states_the_block_a_call_of_its_tokens_launches() {
    let meta = |dims: &[usize], dtype| nn_testing::meta(dims, dtype);
    for (tokens, block) in [(1, 16), (512, 16), (4096, 64)] {
        let (result, recording) = nn_testing::record(&[], || {
            let streams = meta(&[tokens, 4, 5120], DType::BF16);
            let output = meta(&[tokens, 5120], DType::BF16);
            let mixes = MhcMixes {
                pre: meta(&[tokens, 4], DType::F32),
                post: meta(&[tokens, 4], DType::F32),
                comb: meta(&[tokens, 4, 4], DType::F32),
            };
            let projection = meta(&[24, 20480], DType::F32);
            let scale = meta(&[3], DType::F32);
            let base = meta(&[24], DType::F32);
            let norm = meta(&[5120], DType::BF16);
            let mixing = MhcMixing {
                projection: &projection,
                scale: &scale,
                base: &base,
                iterations: 20,
                rms_eps: 1e-20,
                eps: 1e-6,
            };
            functional::mhc_boundary(
                &streams,
                Some((&output, &mixes)),
                &mixes.pre,
                mixing,
                &norm,
                1e-20,
            )
        });
        result.unwrap();

        let boundary: Vec<_> = recording
            .launches
            .iter()
            .filter(|(operator, _)| *operator == "mhc_boundary")
            .collect();
        assert_eq!(
            boundary.len(),
            1,
            "{tokens} tokens: {:?}",
            recording.launches
        );
        assert_eq!(
            boundary[0].1,
            format!(
                "dtype_bf16_streams_4_hidden_5120_iterations_20_post_true_tokens_{block}_block_64"
            )
        );
        // A split call clears its arrival counters first.
        assert_eq!(recording.launches[0].0, "fill", "{tokens} tokens");
    }
}

/// A table split by hash heads over ranks that serve tokens of their own: a rank gathers every
/// rank's row ids, reads its heads' rows of all of them and hands each rank its own tokens'
/// columns back, so its lookup answers its own tokens' every column; it reads only its heads'
/// rows, and a rank with no token of its own takes part with none. A block narrower than the
/// rank's tokens, and a table split by rows, are refused for such ranks.
#[test]
fn a_table_split_by_heads_serves_ranks_that_look_up_their_own_tokens() {
    let rank = nn::Split::new(4, 1).unwrap();
    let (columns, width) = (24, 256);
    let shard = |shard| {
        NgramTable::new(
            nn_testing::meta(&[1000, width], DType::F8E4M3),
            nn_testing::meta(&[1000, width / 32], DType::U8),
            6000,
            shard,
        )
        .unwrap()
    };
    let table = shard(TableShard::Heads(rank));

    let (looked_up, recording) = nn_testing::record(&[rank], || {
        let rows = nn_testing::meta(&[3, columns], DType::I64);
        table.lookup(&rows, DType::BF16, Tokens::Own { block: 8 })
    });
    assert_eq!(looked_up.unwrap().dims(), [3, columns * width]);
    // One read of this rank's 6 heads' rows of 4 ranks' blocks of 8 tokens.
    let reads: Vec<_> = recording
        .launches
        .iter()
        .filter(|(operator, _)| *operator == "embedding_mxfp8")
        .collect();
    assert_eq!(reads.len(), 1, "{:?}", recording.launches);

    let (joined, _) = nn_testing::record(&[rank], || {
        let none = nn_testing::meta(&[0, columns], DType::I64);
        table.lookup(&none, DType::BF16, Tokens::Own { block: 8 })
    });
    assert_eq!(joined.unwrap().dims(), [0, columns * width]);

    let (crowded, _) = nn_testing::record(&[rank], || {
        let rows = nn_testing::meta(&[9, columns], DType::I64);
        table.lookup(&rows, DType::BF16, Tokens::Own { block: 8 })
    });
    assert!(crowded.is_err());
    let (summed, _) = nn_testing::record(&[rank], || {
        let rows = nn_testing::meta(&[3, columns], DType::I64);
        shard(TableShard::Rows(rank)).lookup(&rows, DType::BF16, Tokens::Own { block: 8 })
    });
    let message = summed.err().unwrap().to_string();
    assert!(message.contains("hash heads"), "{message}");
}
