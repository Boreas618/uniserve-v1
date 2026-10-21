//! Encoding and decoding throughput benchmarks for Hugging Face tokenizers.
//!
//! The fixture resolves `tokenizer.json` for `MODEL_ID` from the Hugging Face
//! cache configured by the environment and downloads it through `hf-hub` when
//! it is absent, so a cold cache needs network access. Encoding throughput is
//! reported in input bytes and decoding throughput in token IDs.

use std::error::Error;

use criterion::{Criterion, Throughput, black_box};
use hf_hub::api::sync::ApiBuilder;
use server::profile::tokenizer::HuggingFaceTokenizer;

const MODEL_ID: &str = "Qwen/Qwen3.5-0.8B";
const SAMPLE_TEXT: &str =
    "<|im_start|>user\nSummarize this request.\n<|im_end|>\n<|im_start|>assistant\n";

/// Loaded tokenizer plus a sample text and its encoding.
struct BenchFixture {
    tokenizer: HuggingFaceTokenizer,
    text: String,
    token_ids: Vec<u32>,
}

impl BenchFixture {
    /// Load the tokenizer and encode and decode the sample once, so the
    /// measured calls repeat calls known to succeed.
    fn load() -> Result<Self, Box<dyn Error>> {
        let path = ApiBuilder::from_env()
            .with_progress(false)
            .build()?
            .model(MODEL_ID.to_string())
            .get("tokenizer.json")?;
        let tokenizer = HuggingFaceTokenizer::new(&path)?;
        let text = SAMPLE_TEXT.repeat(32);
        let token_ids = tokenizer.encode(&text, false)?;
        tokenizer.decode(&token_ids, false)?;
        Ok(Self {
            tokenizer,
            text,
            token_ids,
        })
    }
}

fn bench_encode(c: &mut Criterion, fixture: &BenchFixture) {
    let mut group = c.benchmark_group("tokenizer_encode");
    group.throughput(Throughput::Bytes(fixture.text.len() as u64));
    group.bench_function("configured", |b| {
        b.iter(|| {
            fixture
                .tokenizer
                .encode(black_box(&fixture.text), black_box(false))
        })
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion, fixture: &BenchFixture) {
    let mut group = c.benchmark_group("tokenizer_decode");
    group.throughput(Throughput::Elements(fixture.token_ids.len() as u64));
    group.bench_function("configured", |b| {
        b.iter(|| {
            fixture
                .tokenizer
                .decode(black_box(&fixture.token_ids), black_box(false))
        })
    });
    group.finish();
}

/// Load the fixture once, then run both groups as `criterion_main!` would.
fn main() -> Result<(), Box<dyn Error>> {
    let fixture = BenchFixture::load()?;
    let mut criterion = Criterion::default().configure_from_args();
    bench_encode(&mut criterion, &fixture);
    bench_decode(&mut criterion, &fixture);
    criterion.final_summary();
    Ok(())
}
