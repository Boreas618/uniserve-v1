//! DeepSeek-V4.1-Flash's DSpark drafter ([`Dspark`]) on the CPU device, loaded from the
//! release's own `mtp.*` parameters with the model's token embedding and FP32 head, against the
//! checkpoint's reference code run on its own kernels (`generate_dspark.py`).
//!
//! The reference drives `Transformer.forward_spec` over the target hidden states of a fixed
//! prompt (the mean over the four streams of the residual entering layers 37, 38 and 39): at
//! `start_pos = P` the window of each draft layer holds the latents of positions `P − 127 ..= P`
//! (all of them from 0 while `P < 127`), and the block of the anchor `ids[P + 1]` and four noise
//! tokens at positions `P + 1 ..= P + 5` attends the window and itself. Here every step of a
//! group is one row of one forward: its context is those positions' rows, written into the
//! row's own pages of the window, and its block those five tokens.
//!
//! | Compared | Bound |
//! | --- | --- |
//! | the logits of each block position, the Markov term of the reference's own chain added | over every step and position: mean KL divergence of the drafter's distribution from the reference's at most 0.1235, top-1 agreement at least 0.824 |
//! | each step's drafts, the drafter's own chain | equal up to a first difference, at which each side's draft is among the other's five most likely (vLLM's `check_logprobs_close`) |
//!
//! The KL and top-1 bounds are the ones the family's teacher-forced test holds the served model
//! to (`tests/python/e2e/test_deepseek_v41_http.py`: four times the reference's largest mean
//! divergence across two of its own compositions, and its lowest top-1 agreement less 0.02),
//! fixed there before the served model was measured; the draft layers are three layers of the
//! same operators. The vectors are generated from the checkpoint (they hold its outputs) into
//! `artifacts/deepseek_v41/dspark-reference`, and the test is ignored by default. The CPU
//! formulas decode every expert of a layer's stack, so run it optimized:
//!
//! ```text
//! python crates/models/tests/it/deepseek_v41/generate_dspark.py \
//!     --checkpoint <DeepSeek-V4.1-Flash-1017> \
//!     --prompt artifacts/deepseek_v41/reference/prompt_1024.safetensors \
//!     --out artifacts/deepseek_v41/dspark-reference
//! cargo test --release -p models --test it deepseek_v41::dspark -- --ignored --nocapture
//! ```
//!
//! `DSPARK_REFERENCE_STEPS=N` runs only the first `N` steps of each group, which proves the
//! setup and its footprint cheaply before the whole run; the bounds are those of the whole run.

use std::collections::HashMap;
use std::path::PathBuf;

use models::deepseek_v41::{Config, Dspark, cache};
use nn::collective::Split;
use nn::family::{History, TextPlacement};
use nn::modules::{Embedding, EmbeddingConfig, LmHead, LmHeadConfig};
use nn::{
    BlockDrafter, Checkpoint, CpuWeights, DType, Device, DraftBlock, DraftContext, KvCache,
    NO_SLOT, Phase, Tensor, WeightLoader,
};
use nn_testing::text;

use crate::releases::DEEPSEEK_V41_FLASH;

/// The model's width and the drafter's block.
const HIDDEN: usize = 5120;
const BLOCK: usize = 5;
/// The context entries a block reads: the draft layers' window.
const WINDOW: u32 = 128;
/// Tokens of a cache page, and the pages of each row's own window table.
const PAGE: u32 = 64;
const ROW_PAGES: usize = 16;

/// The bound on the mean KL divergence and on the top-1 agreement (module documentation).
const MEAN_KL: f64 = 0.1235;
const TOP1: f64 = 0.824;
/// The likeliest tokens of a side among which the other side's draft lies at a first difference.
const NEAR: usize = 5;

/// The directory the reference vectors are generated into.
fn reference() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/deepseek_v41/dspark-reference")
}

/// Every tensor of the reference vectors, by name.
fn vectors() -> HashMap<String, Tensor> {
    let path = reference().join("vectors.safetensors");
    let file = Checkpoint::open_file(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}; generate it with generate_dspark.py",
            path.display()
        )
    });
    file.names()
        .map(|name| {
            let location = file.locate(name).unwrap();
            let bytes = file.read(name).unwrap();
            let tensor = Tensor::from_raw_buffer(
                &bytes,
                location.dtype,
                location.shape.dims(),
                &Device::Cpu,
            )
            .unwrap();
            (name.to_owned(), tensor)
        })
        .collect()
}

/// A CPU tensor of `values`.
fn column<T: tensor::WithDType>(values: Vec<T>, shape: impl Into<nn::Shape>) -> Tensor {
    Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
}

/// The integers of `tensor`, row-major.
fn integers(tensor: &Tensor) -> Vec<i64> {
    tensor
        .to_dtype(DType::I64)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// The values of `tensor` as FP32, row-major.
fn floats(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// The log-probabilities of `logits`, in FP64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let most = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits
        .iter()
        .map(|&value| (value as f64 - most).exp())
        .sum();
    let log_sum = sum.ln() + most;
    logits.iter().map(|&value| value as f64 - log_sum).collect()
}

/// The indices of the `count` largest of `values`, largest first.
fn top(values: &[f32], count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
    order.truncate(count);
    order
}

/// The drafter, the model's token embedding and its FP32 head, read from the release on the CPU.
struct Loaded {
    config: Config,
    drafter: Dspark,
    embed: Embedding,
    head: LmHead,
}

impl Loaded {
    fn open() -> Self {
        let checkpoint = DEEPSEEK_V41_FLASH.checkpoint();
        let config = Config::open(&checkpoint).expect("the release's configuration");
        let store = CpuWeights::new();
        let mut loader = WeightLoader::new(&checkpoint, &store, None);
        loader
            .declare(&config.quantization)
            .expect("the declaration");
        let vb = loader.root();

        let placement = TextPlacement {
            drafter: true,
            ..TextPlacement::WHOLE
        };
        let drafter = Dspark::load(&config, vb.clone(), &placement).expect("the drafter");
        let embed = Embedding::load(
            &EmbeddingConfig {
                vocabulary: config.vocab_size,
                hidden: config.hidden_size,
                split: Split::WHOLE,
            },
            vb.pp("embed"),
        )
        .expect("the embedding");
        let head = LmHead::load(
            &LmHeadConfig {
                vocabulary: config.vocab_size,
                hidden: config.hidden_size,
                split: Split::WHOLE,
            },
            vb.pp("head").to_dtype(DType::F32),
        )
        .expect("the head");
        Self {
            config,
            drafter,
            embed,
            head,
        }
    }

    /// The model's cache group of `history` over `pages` pages, its rows' tables `tables`, its
    /// slots `slots` and, for a window, its rows' starts.
    fn group(&self, history: History, pages: usize, (tables, slots): (&Tensor, Tensor)) -> KvCache {
        let fields: Vec<_> = cache::fields(&self.config)
            .unwrap()
            .into_iter()
            .filter(|field| field.history == history)
            .collect();
        let rows = tables.dim(0).unwrap();
        let mut group = text::cache(
            &fields,
            DType::BF16,
            (pages, PAGE),
            (
                tables.clone(),
                slots,
                Tensor::zeros(1 << 16, DType::U8, &Device::Cpu).unwrap(),
            ),
            |shape, dtype| Ok(Tensor::zeros(shape, dtype, &Device::Cpu)?),
        )
        .unwrap();
        if history != History::Full {
            group.starts = Some(column(vec![0_u32; rows], rows));
        }
        group
    }
}

/// What one group of reference steps compares to (module documentation).
struct Compared {
    /// Each step's and position's KL divergence of the drafter from the reference.
    divergences: Vec<f64>,
    /// Whether each step's and position's argmax is the reference's.
    agreeing: Vec<bool>,
    /// The drafts of the drafter's own chain that equal the reference's, from each step's
    /// start up to its first difference, and those the prompt continues with.
    equal_prefix: Vec<usize>,
    accepted: Vec<usize>,
}

/// The steps a forward runs, a row each: few, since the CPU formula of the routed experts holds
/// each routed row's expert weights decoded.
const CHUNK: usize = 4;

/// One forward of the drafter over the steps at `positions`, whose anchors are `anchors`, a row
/// each: the head's logits of every block position, `[rows · 5, vocabulary]` FP32, and the
/// drafts of the drafter's own chain, `[rows · 5]`.
fn draft(
    loaded: &Loaded,
    main_hidden: &Tensor,
    (positions, anchors): (&[u32], &[u32]),
) -> (Tensor, Vec<i64>) {
    let rows = positions.len();

    // Row `r`'s logical page `k` is physical page `r · ROW_PAGES + k`, so that no row's block
    // writes another row's context.
    let tables: Vec<u32> = (0..rows * ROW_PAGES).map(|page| page as u32).collect();
    let tables = column(tables, (rows, ROW_PAGES));
    let pages = rows * ROW_PAGES;
    let slot = |row: usize, position: u32| (row * ROW_PAGES) as u32 * PAGE + position;
    assert!(
        positions
            .iter()
            .all(|&p| p + (BLOCK as u32) < ROW_PAGES as u32 * PAGE),
        "every block lies in its row's pages"
    );

    // The context: each row's window of the target's rows, read from its own pages.
    let (mut context, mut context_slots, mut offsets, mut lens) =
        (Vec::new(), Vec::new(), vec![0_u32], Vec::new());
    for (row, &position) in positions.iter().enumerate() {
        let first = position.saturating_sub(WINDOW - 1);
        for at in first..=position {
            context.push(at);
            context_slots.push(slot(row, at));
        }
        offsets.push(context.len() as u32);
        lens.push(position + 1);
    }
    let tokens = context.len();
    let rows_of = column(context.clone(), tokens);
    let hidden = Tensor::cat(
        &[
            Tensor::zeros((tokens, HIDDEN), DType::BF16, &Device::Cpu).unwrap(),
            main_hidden.index_select(&rows_of, 0).unwrap(),
        ],
        1,
    )
    .unwrap();
    let context_batch = text::batch(
        Phase::Prefill,
        [
            column(
                context.iter().map(|&p| p as i32).collect::<Vec<_>>(),
                tokens,
            ),
            column(lens.clone(), rows),
            column(offsets, rows + 1),
            column(lens, rows),
        ],
        None,
        PAGE,
        PAGE * ROW_PAGES as u32,
    );
    let full = loaded.group(
        History::Full,
        1,
        (
            &column(vec![0_u32; rows], (rows, 1)),
            column(vec![NO_SLOT; tokens], tokens),
        ),
    );
    let window = loaded.group(
        History::Window { tokens: WINDOW },
        pages,
        (&tables, column(context_slots, tokens)),
    );

    // The blocks: each row's anchor and noise tokens after its context, in the same pages.
    let blocks = rows * BLOCK;
    let block_positions: Vec<u32> = positions
        .iter()
        .flat_map(|&position| (1..=BLOCK as u32).map(move |ahead| position + ahead))
        .collect();
    let block_slots: Vec<u32> = block_positions
        .iter()
        .enumerate()
        .map(|(token, &position)| slot(token / BLOCK, position))
        .collect();
    let block_lens: Vec<u32> = positions.iter().map(|&p| p + BLOCK as u32 + 1).collect();
    let block_ids: Vec<u32> = anchors
        .iter()
        .flat_map(|&anchor| {
            std::iter::once(anchor).chain((1..BLOCK).map(|_| loaded.drafter.fill()))
        })
        .collect();
    let block_batch = text::batch(
        Phase::Prefill,
        [
            column(
                block_positions
                    .iter()
                    .map(|&p| p as i32)
                    .collect::<Vec<_>>(),
                blocks,
            ),
            column(block_lens.clone(), rows),
            column(
                (0..=rows as u32)
                    .map(|row| row * BLOCK as u32)
                    .collect::<Vec<_>>(),
                rows + 1,
            ),
            column(block_lens, rows),
        ],
        None,
        PAGE,
        PAGE * ROW_PAGES as u32,
    );
    let block_cache = [
        KvCache {
            slot_mapping: column(vec![NO_SLOT; blocks], blocks),
            ..full.clone()
        },
        KvCache {
            slot_mapping: column(block_slots, blocks),
            ..window.clone()
        },
    ];
    let embedded = loaded.embed.forward(&column(block_ids, blocks)).unwrap();

    let drafted = loaded
        .drafter
        .forward(
            &DraftContext {
                batch: &context_batch,
                cache: &[full, window],
                hidden: &hidden,
            },
            &DraftBlock {
                batch: &block_batch,
                cache: &block_cache,
                embedded: &embedded,
            },
            None,
        )
        .expect("the drafter's forward");
    assert_eq!(drafted.dims(), [blocks, HIDDEN]);
    let logits = loaded
        .head
        .forward(&drafted.to_dtype(DType::F32).unwrap())
        .unwrap();
    let proposed = integers(
        &loaded
            .drafter
            .propose(&logits, &column(anchors.to_vec(), rows))
            .unwrap(),
    );
    (logits, proposed)
}

/// Runs the steps of reference group `name`, [`CHUNK`] rows a forward, and compares them.
fn compare(loaded: &Loaded, vectors: &HashMap<String, Tensor>, name: &str) -> Compared {
    let get = |key: &str| {
        vectors
            .get(&format!("{name}.{key}"))
            .unwrap_or_else(|| panic!("the vectors hold no `{name}.{key}`"))
    };
    let ids = integers(&vectors["ids"]);
    let steps = std::env::var("DSPARK_REFERENCE_STEPS")
        .ok()
        .map_or(usize::MAX, |steps| steps.parse().expect("a count of steps"));
    let positions: Vec<u32> = integers(get("positions"))
        .iter()
        .take(steps)
        .map(|&p| p as u32)
        .collect();
    let anchors: Vec<u32> = integers(get("anchors"))
        .iter()
        .take(steps)
        .map(|&t| t as u32)
        .collect();
    let output_ids = integers(get("output_ids"));
    let vocabulary = loaded.config.vocab_size as usize;
    let rows = positions.len();
    let reference_logits = get("logits").narrow(0, 0, rows).unwrap();
    let blocks = rows * BLOCK;

    let mut logits = Vec::new();
    let mut proposed = Vec::with_capacity(blocks);
    for (positions, anchors) in positions.chunks(CHUNK).zip(anchors.chunks(CHUNK)) {
        let (chunk, drafts) = draft(loaded, &vectors["main_hidden"], (positions, anchors));
        logits.push(chunk);
        proposed.extend(drafts);
    }
    let logits = Tensor::cat(&logits, 0).unwrap();

    // The logits of each position with the Markov term of the reference's chain: the anchor,
    // then the reference's drafts.
    let chain: Vec<u32> = (0..rows)
        .flat_map(|row| (0..BLOCK).map(move |offset| (row, offset)))
        .map(|(row, offset)| output_ids[row * (BLOCK + 1) + offset] as u32)
        .collect();
    let forced = logits
        .add(&loaded.drafter.markov(&column(chain, blocks)).unwrap())
        .unwrap();
    let forced = floats(&forced);
    let expected = floats(&reference_logits);
    assert_eq!(expected.len(), blocks * vocabulary);

    let mut compared = Compared {
        divergences: Vec::with_capacity(blocks),
        agreeing: Vec::with_capacity(blocks),
        equal_prefix: Vec::with_capacity(rows),
        accepted: Vec::with_capacity(rows),
    };
    for row in 0..rows {
        let mut equal = 0;
        let mut differed = false;
        for offset in 0..BLOCK {
            let at = (row * BLOCK + offset) * vocabulary;
            let ours = &forced[at..at + vocabulary];
            let theirs = &expected[at..at + vocabulary];
            let (log_ours, log_theirs) = (log_softmax(ours), log_softmax(theirs));
            let divergence: f64 = log_theirs
                .iter()
                .zip(&log_ours)
                .map(|(t, o)| t.exp() * (t - o))
                .sum();
            compared.divergences.push(divergence);
            compared.agreeing.push(top(ours, 1) == top(theirs, 1));

            // The drafter's own chain against the reference's, up to their first difference,
            // where the logits of both still follow one chain.
            if differed {
                continue;
            }
            let (mine, reference) = (
                proposed[row * BLOCK + offset] as usize,
                output_ids[row * (BLOCK + 1) + offset + 1] as usize,
            );
            if mine == reference {
                equal += 1;
            } else {
                differed = true;
                assert!(
                    top(ours, NEAR).contains(&reference) && top(theirs, NEAR).contains(&mine),
                    "{name} step {row} position {offset}: the drafter's {mine} and the \
                     reference's {reference} are not among each other's {NEAR} likeliest"
                );
            }
        }
        compared.equal_prefix.push(equal);

        // What the prompt continues with after the anchor, which a verification would accept.
        let position = positions[row] as usize;
        let accepted = (0..BLOCK)
            .take_while(|&offset| {
                ids.get(position + 2 + offset)
                    .is_some_and(|&token| token == proposed[row * BLOCK + offset])
            })
            .count();
        compared.accepted.push(accepted);
    }
    compared
}

#[test]
#[ignore = "reads the Hugging Face release and the reference vectors of generate_dspark.py"]
fn the_drafter_drafts_as_the_reference_code_does() {
    let vectors = vectors();
    let loaded = Loaded::open();
    assert_eq!(loaded.drafter.block() as usize, BLOCK);

    let mut divergences = Vec::new();
    let mut agreeing = Vec::new();
    for name in ["g0", "g1"] {
        let compared = compare(&loaded, &vectors, name);
        let steps = compared.equal_prefix.len();
        let mean = |values: &[usize]| values.iter().sum::<usize>() as f64 / values.len() as f64;
        println!(
            "{name}: {steps} steps, mean KL {:.4}, largest {:.4}, top-1 {:.3}, drafts equal to \
             the reference's {:.2} of {BLOCK} a step, accepted against the prompt {:.2}",
            compared.divergences.iter().sum::<f64>() / compared.divergences.len() as f64,
            compared.divergences.iter().copied().fold(0.0, f64::max),
            compared.agreeing.iter().filter(|&&same| same).count() as f64
                / compared.agreeing.len() as f64,
            mean(&compared.equal_prefix),
            mean(&compared.accepted),
        );
        divergences.extend(compared.divergences);
        agreeing.extend(compared.agreeing);
    }

    let mean_kl = divergences.iter().sum::<f64>() / divergences.len() as f64;
    let top1 = agreeing.iter().filter(|&&same| same).count() as f64 / agreeing.len() as f64;
    assert!(mean_kl <= MEAN_KL, "mean KL {mean_kl} above {MEAN_KL}");
    assert!(top1 >= TOP1, "top-1 agreement {top1} below {TOP1}");
}
