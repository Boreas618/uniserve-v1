//! A calibrated (NVFP4) Qwen3-MoE checkpoint through the family's real loading boundary: its
//! projections load packed and its router dense, and a row-parallel packed projection reads its own
//! rank's columns.
//!
//! The shard is written here, by hand: the format is a header JSON plus a payload, which is exactly
//! what the loader reads, and a fixture states exactly the bytes a calibrated export holds (FP8
//! scales among them, which the `safetensors` crate's writer cannot state).

use std::path::Path;

use nn::weights::{RetainedWeights, WeightKey};
use nn::{Backing, Checkpoint, DType, DevicePtr, Shape, WeightLoader};
use tempfile::TempDir;

/// One tensor of the hand-written shard: name, extents, element type, payload.
struct Tensor<'a> {
    name: &'a str,
    shape: Vec<u64>,
    dtype: &'a str,
    bytes: Vec<u8>,
}

impl Tensor<'_> {
    fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Bytes one element occupies, from the safetensors format's own table.
    fn width(dtype: &str) -> u64 {
        match dtype {
            "U8" => 1,
            "F8_E4M3" => 1,
            "BF16" => 2,
            "F32" => 4,
            _ => panic!("the fixture uses U8, F8_E4M3, BF16 and F32 only"),
        }
    }
}

/// Write `tensors` as one safetensors file: an 8-byte header length, the JSON header, the payload.
fn write_shard(dir: &Path, tensors: &[Tensor<'_>]) -> std::path::PathBuf {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    let mut payload = Vec::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let bytes = tensor.elements() * Tensor::width(tensor.dtype);
        // The payload is exactly what the header declares: a fixture that deliberately mis-declares
        // a field's element type is padded to that type's width, so the dtype is what the loader
        // objects to and not the length.
        let mut payload_tensor = tensor.bytes.clone();
        payload_tensor.resize(bytes as usize, 0);
        if index > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""{}":{{"dtype":"{}","shape":{:?},"data_offsets":[{},{}]}}"#,
            tensor.name,
            tensor.dtype,
            tensor.shape,
            offset,
            offset + bytes
        ));
        offset += bytes;
        payload.extend_from_slice(&payload_tensor);
    }
    header.push('}');

    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    let path = dir.join("model.safetensors");
    std::fs::write(&path, file)
        .map_err(|error| format!("the fixture shard: {error}"))
        .unwrap_or_else(|error| panic!("{error}"));
    path
}

/// A store that keeps what it was given, in host memory, the way a layer's store does.
#[derive(Default)]
struct Retaining {
    published: Vec<(WeightKey, Vec<u8>, DType, Shape, u64)>,
    next: u64,
}

impl RetainedWeights for Retaining {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<Backing>> {
        Ok(self
            .published
            .iter()
            .find(|(published, ..)| published == key)
            .map(|(_, bytes, _, _, address)| {
                Backing::new(DevicePtr::new(*address), bytes.len() as u64)
            }))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        self.next += 0x1_0000;
        let address = self.next;
        self.published
            .push((key.clone(), bytes.to_vec(), dtype, shape, address));
        Ok(Backing::new(DevicePtr::new(address), bytes.len() as u64))
    }
}

/// A whole tiny model whose **attention projections are packed** and whose router, norms and
/// vocabulary are not — the checkpoint a calibrated export of this architecture states, one layer
/// deep and small enough to write by hand.
///
/// The expert stacks need 128 rows to be published in the product's own arrangement, and a packed
/// projection reduces over whole sixteen-element blocks, so the smallest geometry that can
/// carry both is 128 wide with 128 experts' worth of intermediate.
const PACKED_HIDDEN: u64 = 128;
const PACKED_VOCABULARY: u64 = 256;
const PACKED_HEAD_DIM: u64 = 8;
const PACKED_EXPERTS: u64 = 2;

/// The block scale a linear table holds at `index`, nonzero so that padding stays distinguishable.
fn scale_byte(index: u64) -> u8 {
    (index % 255 + 1) as u8
}

fn packed_model_shard() -> TempDir {
    let dir = TempDir::new().unwrap_or_else(|error| panic!("a temporary directory: {error}"));
    let config = format!(
        r#"{{
            "architectures": ["Qwen3MoeForCausalLM"],
            "vocab_size": {PACKED_VOCABULARY},
            "hidden_size": {PACKED_HIDDEN},
            "intermediate_size": {PACKED_HIDDEN},
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": {PACKED_HEAD_DIM},
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false,
            "num_experts": {PACKED_EXPERTS},
            "num_experts_per_tok": 1,
            "moe_intermediate_size": {PACKED_HIDDEN},
            "norm_topk_prob": true,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "quantization_config": {{
                "config_groups": {{"group_0": {{
                    "weights": {{"num_bits": 4, "type": "float", "group_size": 16}},
                    "targets": ["Linear"]}}}},
                "ignore": ["model.layers.0.mlp.gate"]}},
            "attention_bias": false
        }}"#
    );
    std::fs::write(dir.path().join("config.json"), config)
        .unwrap_or_else(|error| panic!("the fixture config: {error}"));

    // Every tensor the model reads, in the checkpoint's own spelling. A packed projection carries
    // four: the pair, the block scales, the weight's tensor scale and the activation's.
    let bf16 = |elements: u64| vec![0u8; (elements * 2) as usize];
    let mut owned: Vec<(String, Vec<u64>, String, Vec<u8>)> = vec![
        (
            "model.embed_tokens.weight".to_owned(),
            vec![PACKED_VOCABULARY, PACKED_HIDDEN],
            "BF16".to_owned(),
            bf16(PACKED_VOCABULARY * PACKED_HIDDEN),
        ),
        (
            "lm_head.weight".to_owned(),
            vec![PACKED_VOCABULARY, PACKED_HIDDEN],
            "BF16".to_owned(),
            bf16(PACKED_VOCABULARY * PACKED_HIDDEN),
        ),
        (
            "model.norm.weight".to_owned(),
            vec![PACKED_HIDDEN],
            "BF16".to_owned(),
            bf16(PACKED_HIDDEN),
        ),
    ];
    for norm in ["input_layernorm", "post_attention_layernorm"] {
        owned.push((
            format!("model.layers.0.{norm}.weight"),
            vec![PACKED_HIDDEN],
            "BF16".to_owned(),
            bf16(PACKED_HIDDEN),
        ));
    }
    for norm in ["q_norm", "k_norm"] {
        owned.push((
            format!("model.layers.0.self_attn.{norm}.weight"),
            vec![PACKED_HEAD_DIM],
            "BF16".to_owned(),
            bf16(PACKED_HEAD_DIM),
        ));
    }
    owned.push((
        "model.layers.0.mlp.gate.weight".to_owned(),
        vec![PACKED_EXPERTS, PACKED_HIDDEN],
        "BF16".to_owned(),
        bf16(PACKED_EXPERTS * PACKED_HIDDEN),
    ));

    // The four attention projections, packed: `q`, `k` and `v` reduce over the hidden width, `o` over
    // the query width, and every one of them is a whole number of sixteen-element blocks.
    let mut packed_tensor = |name: String, rows: u64, columns: u64, seed: u8| {
        owned.push((
            format!("{name}.weight"),
            vec![rows, columns / 2],
            "U8".to_owned(),
            vec![seed; (rows * columns / 2) as usize],
        ));
        // The block scales identify their own position, so a region read from the wrong rank or with
        // the wrong stride changes the *set* of published bytes and not only their order.
        owned.push((
            format!("{name}.weight_scale"),
            vec![rows, columns / 16],
            "F8_E4M3".to_owned(),
            (0..rows * columns / 16).map(scale_byte).collect(),
        ));
        owned.push((
            format!("{name}.weight_scale_2"),
            Vec::new(),
            "F32".to_owned(),
            f32::from(seed).to_le_bytes().to_vec(),
        ));
        owned.push((
            format!("{name}.input_scale"),
            Vec::new(),
            "F32".to_owned(),
            f32::from(seed).to_le_bytes().to_vec(),
        ));
    };
    let query = 4 * PACKED_HEAD_DIM;
    let key_value = 2 * PACKED_HEAD_DIM;
    packed_tensor(
        "model.layers.0.self_attn.q_proj".to_owned(),
        query,
        PACKED_HIDDEN,
        3,
    );
    packed_tensor(
        "model.layers.0.self_attn.k_proj".to_owned(),
        key_value,
        PACKED_HIDDEN,
        5,
    );
    packed_tensor(
        "model.layers.0.self_attn.v_proj".to_owned(),
        key_value,
        PACKED_HIDDEN,
        7,
    );
    packed_tensor(
        "model.layers.0.self_attn.o_proj".to_owned(),
        PACKED_HIDDEN,
        query,
        11,
    );
    for expert in 0..PACKED_EXPERTS {
        for (projection, rows, columns, seed) in [
            ("gate", PACKED_HIDDEN, PACKED_HIDDEN, 13u8),
            ("up", PACKED_HIDDEN, PACKED_HIDDEN, 17u8),
            ("down", PACKED_HIDDEN, PACKED_HIDDEN, 19u8),
        ] {
            packed_tensor(
                format!("model.layers.0.mlp.experts.{expert}.{projection}_proj"),
                rows,
                columns,
                seed,
            );
        }
    }

    let borrowed: Vec<Tensor<'_>> = owned
        .iter()
        .map(|(name, shape, dtype, bytes)| Tensor {
            name,
            shape: shape.clone(),
            dtype,
            bytes: bytes.clone(),
        })
        .collect();
    write_shard(dir.path(), &borrowed);
    dir
}

/// The checkpoint at `dir`, opened and loaded through the family as execution loads it, at rank
/// `rank` of a tensor split of `ranks`, its weights published to `store`.
fn load_model(
    dir: &Path,
    store: &mut Retaining,
    rank: u32,
    ranks: u32,
) -> nn::Result<std::sync::Arc<dyn nn::family::TextModel>> {
    let nn::family::Model::Text(family) = qwen3::family::FAMILY.model else {
        panic!("Qwen3-MoE is a text family");
    };
    let checkpoint = Checkpoint::open(dir)?;
    let opened = family.open(&checkpoint)?;
    let mut loader = WeightLoader::new(&checkpoint, store, None, None);
    let placement = nn::family::TextPlacement {
        tensor: nn::family::Split { size: ranks, rank },
        experts: nn::family::Split::WHOLE,
        drafter: false,
    };
    opened.load(
        &checkpoint,
        &mut loader,
        &placement,
        &nn::family::ExpertOrder::new(),
    )
}

#[test]
fn a_whole_calibrated_checkpoint_loads_its_projections_packed_and_its_router_dense() {
    let dir = packed_model_shard();
    let mut store = Retaining::default();
    let model = load_model(dir.path(), &mut store, 0, 1).expect("the calibrated checkpoint loads");

    // The forward launches both: the projections' product once per row class, and the experts'
    // products, which carry no row count. One row and two hundred lie in two classes
    // (`PACKED_M_CLASSES`); a pair inside one class compiles one program, so it would launch half as
    // many keys.
    let mut keys = std::collections::BTreeSet::new();
    for tokens in [1, 200] {
        keys.extend(recorded_forward(&*model, tokens));
    }
    let named: Vec<&str> = keys.iter().map(|(operator, _)| operator.as_str()).collect();
    // `q` is `[2 · head_dim, hidden]`, `o` is `[hidden, 2 · head_dim]`, and `k` and `v` are one
    // geometry between them — so three packed products per row class, and no dense MLP's at all.
    assert_eq!(
        named
            .iter()
            .filter(|operator| **operator == "nvfp4_gemm")
            .count(),
        3 * 2,
        "three attention geometries at two row classes, and no dense MLP's: {named:?}"
    );
    assert!(named.contains(&"nvfp4_quantize"), "{named:?}");
    assert!(named.contains(&"gather_gemm_nvfp4"), "{named:?}");
    assert!(named.contains(&"route_gemm_nvfp4"), "{named:?}");
    assert!(
        named.contains(&"topk_softmax"),
        "the router is dense: {named:?}"
    );
}

/// The programs a launch reported, as `(operator, key)` pairs.
#[derive(Default)]
struct Launches(std::collections::BTreeSet<(String, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.insert((operator.to_owned(), key.to_owned()));
    }
}

/// The programs one prefill forward of `tokens` tokens launches, walked on a recording forward.
///
/// Every view addresses a counting arena that no memory backs: a recording forward reads nothing
/// and launches nothing, and reports each launch's program to the capture sink.
fn recorded_forward(
    model: &dyn nn::family::TextModel,
    tokens: u64,
) -> std::collections::BTreeSet<(String, String)> {
    use nn::capability::{KvGroup, KvPlanes, KvView, TokenBatch};
    use proto::module::PhaseKind;

    let cache = model.cache();
    let arena = nn::Arena::counting();
    let f = nn::Forward::recording(&arena);
    let tensor = |dtype: DType, dims: &[u64]| {
        arena
            .tensor(dtype, Shape::new(dims))
            .unwrap_or_else(|error| panic!("a counting arena holds any reservation: {error}"))
    };

    // The fixture's checkpoint states 128 positions, so a batch of more tokens than that is
    // several sequences of at most 128 each; the attention reads a row's keys in whole pages.
    let page_size = 64_u32;
    let row_keys = tokens.min(128).div_ceil(u64::from(page_size)) * u64::from(page_size);
    let rows = tokens.div_ceil(128);
    let row_pages = row_keys / u64::from(page_size);
    let pages = rows * row_pages;
    let batch = TokenBatch {
        positions: tensor(DType::I32, &[tokens]),
        slot_mapping: tensor(DType::I32, &[tokens]),
        seq_lens: tensor(DType::I32, &[rows]),
        query_offsets: tensor(DType::I32, &[rows + 1]),
        logits_rows: tensor(DType::I32, &[rows]),
        phase: PhaseKind::Prefill,
        page_size,
        num_tokens: u32::try_from(tokens).unwrap_or_else(|_| panic!("{tokens} tokens")),
        max_kv_len: u32::try_from(row_keys).unwrap_or_else(|_| panic!("{row_keys} keys")),
        draft_lens: None,
        rope_positions: None,
    };
    let planes_shape = [
        pages,
        u64::from(page_size),
        u64::from(cache.kv_heads),
        u64::from(cache.head_dim),
    ];
    let layers: Vec<KvPlanes> = (0..cache.layers)
        .map(|_| KvPlanes {
            key: tensor(DType::BF16, &planes_shape),
            value: tensor(DType::BF16, &planes_shape),
            key_initialized: tensor(DType::U8, &[pages, 1, 1, 1]),
            value_initialized: tensor(DType::U8, &[pages, 1, 1, 1]),
            key_scale: None,
            value_scale: None,
        })
        .collect();
    let groups = [KvGroup {
        block_tables: tensor(DType::I32, &[rows, row_pages]),
        page_size,
        dtype: DType::BF16,
        layers: &layers,
    }];
    let hidden = tensor(DType::BF16, &[tokens, u64::from(model.hidden())]);
    // The attention program's scratch is the runtime's; a recording forward reads none of it.
    let workspace = tensor(DType::U8, &[1 << 20]);

    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let cache = KvView {
        groups: &groups,
        workspace,
        state: None,
    };
    let walked = model.forward(&f, &batch, &cache, hidden);
    ops::capture::set_sink(None);
    walked.unwrap_or_else(|error| panic!("the recorded forward: {error}"));
    launches.0
}

/// The published block scales of `projection` at `rank`, in the product's arrangement, as the store
/// was handed them.
fn published_scales(store: &Retaining, projection: &str) -> Vec<u8> {
    let mut found: Vec<&Vec<u8>> = store
        .published
        .iter()
        .filter(|(key, ..)| {
            key.layout == nn::weights::WeightLayout::NVFP4_SCALES_SWIZZLED
                && key
                    .region
                    .tensors
                    .iter()
                    .any(|tensor| tensor.contains(projection))
        })
        .map(|(_, bytes, ..)| bytes)
        .collect();
    assert_eq!(found.len(), 1, "one published table for {projection}");
    found.remove(0).clone()
}

/// A row-parallel packed projection is read from **this rank's columns of the file**, and the table
/// the store receives is that rank's scales permuted — not rank 0's, and not the row-major padding of
/// them, which is what a skipped permutation or an off-by-one stride would publish.
#[test]
fn a_rank_one_packed_projection_reads_its_own_columns() {
    let dir = packed_model_shard();
    let mut tables = Vec::new();
    for rank in [0_u32, 1] {
        let mut store = Retaining::default();
        load_model(dir.path(), &mut store, rank, 2)
            .expect("the calibrated checkpoint loads at this rank");
        // `o_proj` is row-parallel: the file is `[hidden, query]` and this rank takes `query / 2`
        // columns, so its scales are one block of every two in each row.
        tables.push(published_scales(&store, "o_proj"));
    }

    let rows = PACKED_HIDDEN;
    let file_blocks = (4 * PACKED_HEAD_DIM) / 16;
    for (rank, table) in tables.iter().enumerate() {
        // The padded 128x4 table: `rows` padded to 128, this rank's one block padded to four.
        assert_eq!(
            table.len() as u64,
            rows.div_ceil(128) * 128 * 4,
            "rank {rank}'s table size"
        );

        let mut published: Vec<u8> = table.iter().copied().filter(|byte| *byte != 0).collect();
        let mut expected: Vec<u8> = (0..rows)
            .map(|row| scale_byte(row * file_blocks + rank as u64))
            .collect();
        published.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            published, expected,
            "rank {rank} publishes exactly its own block scales"
        );

        let mut row_major = vec![0_u8; table.len()];
        for row in 0..rows {
            row_major[(row * 4) as usize] = scale_byte(row * file_blocks + rank as u64);
        }
        assert_ne!(
            *table, row_major,
            "rank {rank} publishes the permuted table, not the row-major one"
        );
    }
    assert_ne!(tables[0], tables[1], "the two ranks read different columns");
}
