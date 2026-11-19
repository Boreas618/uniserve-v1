//! DeepSeek-V4.1-Flash against its reference on the CPU device, over the release's own files.
//!
//! | Test | Reference | What is compared |
//! | --- | --- | --- |
//! | n-gram hashing | `inference/engram.py`, run as it is (`generate_engram.py`, `data/engram.json`) | the normalized vocabulary, the multipliers, the primes, every table row of a fixed token sequence, exactly |
//! | layer 1's residual | `Block.hc_mixes`, `hc_split_sinkhorn`, `hc_pre`, `hc_post`, `RMSNorm` (`inference/model.py:909-996`, `kernel.py:407-474`), ported ([`nn_testing::reference`]) | the residual of the fixed tokens' embeddings through layer 1's two connections and the output norm |
//! | layer 1's n-gram memory | `ParallelEngramEmbedding`, `Engram.forward` (`inference/model.py:296-365`), ported | the streams the memory gates, over the release's table rows, projection and gate |
//!
//! The tests read the release's tensors from its shards ([`super::shards`]); they are ignored by
//! default and run with `--ignored` where the release resolves (`nn_testing::Release`).
//!
//! ## The bounds
//!
//! The ports compute in FP64 and store into BF16 where the reference stores, so they differ from
//! the library's FP32 arithmetic only where an FP32 result rounds to the other neighbour of the
//! FP64 one's: two units of the element at most, for each rounding on an output's path. The
//! residual's collapsed row lies behind eight such roundings (two for each read, one for each
//! write, two for the collapse), which the doubly stochastic writes carry on with a gain of at
//! most one. The memory's gated streams follow the gate's operator bound
//! (`ops::ngram::NgramGate::tolerance`) and the one rounding of the projected value it adds.
//! No bound is chosen here.

use std::path::{Path, PathBuf};

use models::deepseek_v41::{Config, Hashing};
use nn::config::Fields;
use nn::functional::WeightOrder;
use nn::modules::{
    Linear, MhcConnection, MhcResidual, NgramMemory, NgramTable, TableShard, Tokens,
};
use nn::{DType, Device, Tensor};
use nn_testing::reference::{self, Mixing, assert_within};
use nn_testing::{tensor, values};
use ops::Op;
use ops::ngram::{GateDescriptor, NgramGate};
use ops::tolerance::{self, Tolerance};
use sha2::{Digest, Sha256};

use super::shards::Shards;
use crate::releases::DEEPSEEK_V41_FLASH;

/// What the reference code computed (`generate_engram.py`).
fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("data/engram.json")).expect("the fixture is JSON")
}

/// The release's snapshot.
fn snapshot() -> PathBuf {
    DEEPSEEK_V41_FLASH.snapshot()
}

/// The configuration the release's `config.json` states.
fn config(root: &Path) -> Config {
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).expect("config.json"))
            .expect("config.json is JSON");
    Config::parse(&Fields::of("config.json", &value).expect("fields")).expect("the configuration")
}

/// The fixture's integers at `key`, flattened.
fn integers(value: &serde_json::Value) -> Vec<i64> {
    match value {
        serde_json::Value::Array(items) => items.iter().flat_map(integers).collect(),
        number => vec![number.as_i64().expect("an integer")],
    }
}

/// The fixture's token sequence.
fn tokens(fixture: &serde_json::Value) -> Vec<u32> {
    integers(&fixture["tokens"])
        .into_iter()
        .map(|token| token as u32)
        .collect()
}

/// Each token's context of `order` ids, `[tokens, order]`: the ids before it, oldest first, then
/// its own, a position before the sequence holding the padding token `pad`.
fn context(tokens: &[u32], order: usize, pad: u32) -> Tensor {
    let ids: Vec<u32> = (0..tokens.len())
        .flat_map(|position| {
            (0..order).map(move |slot| {
                let back = order - 1 - slot;
                position
                    .checked_sub(back)
                    .map_or(pad, |earlier| tokens[earlier])
            })
        })
        .collect();
    Tensor::from_vec(ids, (tokens.len(), order), &Device::Cpu).expect("a context")
}

#[test]
#[ignore = "reads the Hugging Face release deepseek-ai/DeepSeek-V4.1-Flash-1017"]
fn the_ngram_hashing_is_the_reference_codes() {
    let root = snapshot();
    let config = config(&root);
    let engram = config
        .engram
        .clone()
        .expect("the release has n-gram memories");
    let fixture = fixture();
    let hashing = Hashing::open(&config, &root.join("tokenizer.json")).expect("the hashing");

    // The normalized vocabulary.
    let normalize = hashing.normalize();
    assert_eq!(
        normalize.len() as i64,
        fixture["vocabulary"].as_i64().unwrap()
    );
    let bytes: Vec<u8> = normalize.iter().flat_map(|id| id.to_le_bytes()).collect();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, fixture["token_map_sha256"].as_str().unwrap());
    assert_eq!(
        i64::from(normalize[engram.pad_token_id as usize]),
        fixture["pad"].as_i64().unwrap()
    );

    // Every layer's multipliers, primes and table rows of the fixed tokens.
    let tokens = tokens(&fixture);
    let order = engram.max_ngram_size as usize;
    let context = context(&tokens, order, engram.pad_token_id);
    for (index, layer) in engram.layer_ids.iter().enumerate() {
        assert_eq!(
            hashing.multipliers(*layer).unwrap(),
            integers(&fixture["multipliers"][index]).as_slice(),
            "layer {layer}'s multipliers"
        );
        let primes: Vec<i64> = hashing
            .primes(*layer)
            .unwrap()
            .iter()
            .flatten()
            .map(|&prime| prime as i64)
            .collect();
        assert_eq!(
            primes,
            integers(&fixture["primes"][index]),
            "layer {layer}'s primes"
        );

        let rows = hashing.host(*layer).unwrap().rows(&context).unwrap();
        assert_eq!(
            rows.flatten_all().unwrap().to_vec1::<i64>().unwrap(),
            integers(&fixture["rows"][index]),
            "layer {layer}'s rows"
        );
    }
}

/// One connection's parameters as the ports read them: the projection, the scales, the biases
/// and the input norm's gain.
struct Parameters {
    projection: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
    norm: Vec<f32>,
}

/// Layer 1's connection of `sublayer` (`attn` or `ffn`), from the release's tensors: the
/// library's module and the ports' parameters.
fn connection(shards: &Shards, config: &Config, sublayer: &str) -> (MhcConnection, Parameters) {
    let mhc = config.mhc();
    let (n, d) = (mhc.streams as usize, mhc.hidden as usize);
    let m = n * (n + 2);
    let projection = shards.f32(&format!("layers.1.hc_{sublayer}_fn"));
    let scale = shards.f32(&format!("layers.1.hc_{sublayer}_scale"));
    let base = shards.f32(&format!("layers.1.hc_{sublayer}_base"));
    let norm = shards.bf16(&format!("layers.1.{sublayer}_norm.weight"));

    let module = MhcConnection::new(
        &mhc,
        tensor(&projection, (m, n * d), DType::F32),
        tensor(&scale, 3, DType::F32),
        tensor(&base, m, DType::F32),
        tensor(&norm, d, DType::BF16),
    )
    .expect("the connection");
    let parameters = Parameters {
        projection,
        scale,
        base,
        norm,
    };
    (module, parameters)
}

#[test]
#[ignore = "reads the Hugging Face release deepseek-ai/DeepSeek-V4.1-Flash-1017"]
fn the_layer_one_residual_is_the_reference_formulas() {
    let root = snapshot();
    let config = config(&root);
    let shards = Shards::open(&root);
    let mhc = config.mhc();
    let tokens = tokens(&fixture());
    let shape = (tokens.len(), mhc.streams as usize, mhc.hidden as usize);
    let eps = f64::from(config.rms_norm_eps);

    let rows: Vec<usize> = tokens.iter().map(|&token| token as usize).collect();
    let embedded = shards.bf16_rows("embed.weight", &rows);
    let norm = shards.bf16("norm.weight");
    let connections = [
        connection(&shards, &config, "attn"),
        connection(&shards, &config, "ffn"),
    ];

    // Layer 1's attention and feed-forward sub-layers, each halving its input (exact in BF16),
    // then the output norm.
    let mut residual =
        MhcResidual::expand(&mhc, &tensor(&embedded, (shape.0, shape.2), DType::BF16)).unwrap();
    let mut ported = reference::Residual::expand(&embedded, shape, DType::BF16);
    for (module, parameters) in &connections {
        residual
            .sublayer(module, |input| Ok(input.affine(0.5, 0.0)?))
            .unwrap();
        let mixing = Mixing {
            projection: &parameters.projection,
            scale: &parameters.scale,
            base: &parameters.base,
            iterations: mhc.iterations as usize,
            rms_eps: eps,
            eps: f64::from(mhc.eps),
        };
        ported.sublayer(mixing, &parameters.norm, eps, |input| {
            input.iter().map(|x| 0.5 * x).collect()
        });
    }
    let collapsed = residual
        .collapse(&tensor(&norm, shape.2, DType::BF16), config.rms_norm_eps)
        .unwrap();
    let expected = ported.collapse(&norm, eps);

    let bound = Tolerance {
        atol: tolerance::underflow(DType::BF16) as f32,
        rtol: 16.0 * tolerance::unit_roundoff(DType::BF16) as f32,
        cancels: true,
    };
    assert_within(
        "layer 1's collapsed residual",
        &values(&collapsed),
        &expected,
        bound,
    );
}

/// The release's projection `name` (`weight`, E4M3, and `scale`, UE8M0 per 32 × 32 block),
/// dequantized: exact in BF16, a power of two times an E4M3 value.
fn dequantized(shards: &Shards, name: &str) -> (Vec<f32>, (usize, usize)) {
    let shape = shards.shape(&format!("{name}.weight"));
    let (outputs, inputs) = (shape[0], shape[1]);
    let codes = shards.bytes(&format!("{name}.weight"), "F8_E4M3", 1);
    let scales = shards.bytes(&format!("{name}.scale"), "F8_E8M0", 1);
    let decoded = values(
        &Tensor::from_vec(codes, outputs * inputs, &Device::Cpu)
            .unwrap()
            .view(DType::F8E4M3)
            .unwrap(),
    );
    let blocks = inputs / 32;

    let weight = decoded
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let (row, column) = (index / inputs, index % inputs);
            let exponent = i32::from(scales[(row / 32) * blocks + column / 32]) - 127;
            value * 2_f32.powi(exponent)
        })
        .collect();
    (weight, (outputs, inputs))
}

#[test]
#[ignore = "reads the Hugging Face release deepseek-ai/DeepSeek-V4.1-Flash-1017"]
fn the_layer_one_ngram_memory_is_the_reference_formulas() {
    const TOKENS: usize = 8;

    let root = snapshot();
    let config = config(&root);
    let engram = config.engram.clone().unwrap();
    let ngram = config.ngram().unwrap();
    let shards = Shards::open(&root);
    let tokens: Vec<u32> = tokens(&fixture())[..TOKENS].to_vec();
    let (n, d, r) = (
        ngram.streams as usize,
        ngram.hidden as usize,
        ngram.row as usize,
    );
    let columns = ngram.columns() as usize;

    // The rows the tokens' n-grams name, and the release's rows of them, as a table of just
    // those rows in that order.
    let hashing = Hashing::open(&config, &root.join("tokenizer.json")).unwrap();
    let rows = hashing
        .host(1)
        .unwrap()
        .rows(&context(
            &tokens,
            engram.max_ngram_size as usize,
            engram.pad_token_id,
        ))
        .unwrap();
    let ids: Vec<usize> = rows
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let table_values = shards.rows("layers.1.engram.embed.weight", "F8_E4M3", r, &ids);
    let table_scales = shards.rows("layers.1.engram.embed.scale", "F8_E8M0", r / 32, &ids);
    let held = |bytes: &[u8], width: usize, dtype: DType| {
        Tensor::from_vec(bytes.to_vec(), (ids.len(), width), &Device::Cpu)
            .unwrap()
            .view(dtype)
            .unwrap()
    };
    let table = NgramTable::new(
        held(&table_values, r, DType::F8E4M3),
        held(&table_scales, r / 32, DType::U8),
        0,
        TableShard::Whole,
    )
    .unwrap();
    let local: Vec<i64> = (0..ids.len() as i64).collect();
    let local_rows = Tensor::from_vec(local.clone(), (TOKENS, columns), &Device::Cpu).unwrap();

    // The projection, exact in BF16, and the gate's weights.
    let (projection, (outputs, inputs)) = dequantized(&shards, "layers.1.engram.wkv");
    let query = shards.bf16("layers.1.engram.q_weight");
    let key = shards.bf16("layers.1.engram.k_weight");
    let memory = NgramMemory::new(
        &ngram,
        hashing.host(1).unwrap(),
        table,
        Linear::new(
            tensor(&projection, (outputs, inputs), DType::BF16),
            None,
            WeightOrder::NxK,
        )
        .unwrap(),
        tensor(&query, (n, d), DType::BF16),
        tensor(&key, (n, d), DType::BF16),
    )
    .unwrap();

    // Streams of the tokens' embeddings, stream k of token t holding token t + k's.
    let embedded: Vec<usize> = (0..TOKENS)
        .flat_map(|token| {
            let tokens = &tokens;
            (0..n).map(move |k| tokens[(token + k) % TOKENS] as usize)
        })
        .collect();
    let h = shards.bf16_rows("embed.weight", &embedded);
    let streams = tensor(&h, (TOKENS, n, d), DType::BF16);

    let recalled = memory
        .lookup(&local_rows, DType::BF16, Tokens::Shared)
        .unwrap();
    let gated = memory.gate(&streams, &recalled).unwrap();

    // The reference: the rows' values, the projection stored in BF16, the gate.
    let rows = reference::round(
        DType::BF16,
        &reference::mxfp8(&table_values, &table_scales, r, &local, 0)
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
    );
    assert_eq!(values(&recalled), rows, "the recalled rows");
    let projected: Vec<f64> = (0..TOKENS)
        .flat_map(|token| {
            let row = &rows[token * inputs..(token + 1) * inputs];
            let projection = &projection;
            (0..outputs).map(move |output| {
                row.iter()
                    .zip(&projection[output * inputs..(output + 1) * inputs])
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>()
            })
        })
        .collect();
    let projected = reference::round(DType::BF16, &projected);
    let expected = reference::gate(
        &h,
        &projected,
        &query,
        &key,
        (TOKENS, n, d),
        f64::from(ngram.eps),
        DType::BF16,
    );

    let gate = NgramGate::tolerance(&GateDescriptor::new(DType::BF16, n as u32, d as u32));
    let bound = Tolerance {
        rtol: gate.rtol + 2.0 * tolerance::unit_roundoff(DType::BF16) as f32,
        ..gate
    };
    assert_within("layer 1's gated streams", &values(&gated), &expected, bound);
}
