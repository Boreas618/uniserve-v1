//! Encoding and decoding throughput of a chat checkpoint's tokenizer.
//!
//! `cargo bench --bench tokenizer_hf -- --checkpoint <owner>/<name>[@<revision>] [FILTER]`
//!
//! The checkpoint is a Hugging Face repository, resolved as the daemon resolves one: through the
//! Hub the standard Hugging Face environment configures, into its cache (a cold cache needs the
//! network, an offline host a populated cache). The sample is the conversation of `--turns` user
//! and assistant turns rendered through the checkpoint's own chat template by the tree's processing
//! program (`crates/processing/program`, built for wasm32 by the cargo that runs the bench), as a
//! chat request's prompt is. Encoding throughput is reported in input bytes and decoding throughput
//! in token IDs; `FILTER` selects benchmarks by name, as Criterion's does.

use std::error::Error;

use clap::Parser;
use criterion::{Criterion, Throughput, black_box};
use daemon::api::input::LoadedBinding;
use daemon::api::model::ModelConfig;
use daemon::api::model::assets::{Hub, HubAsset, Source};
use daemon::api::program::Program;
use daemon::api::protocol::ChatCompletionRequest;
use daemon::api::tokenizer::DynTokenizer;
use scheduling::processing::Segment;

/// The bench's protocol.
#[derive(Debug, Parser)]
struct Arguments {
    /// The chat checkpoint whose tokenizer is measured: `<owner>/<name>[@<revision>]`.
    #[arg(long)]
    checkpoint: String,
    /// The user and assistant turns of the rendered conversation: 32 renders a prompt of a few
    /// thousand tokens, the size of a long chat request's.
    #[arg(long, default_value_t = 32)]
    turns: usize,
    /// Runs only the benchmarks whose names contain it.
    filter: Option<String>,
    /// Passed by `cargo bench` to every bench executable.
    #[arg(long, hide = true)]
    bench: bool,
}

/// Loaded tokenizer plus a sample text and its encoding.
struct BenchFixture {
    tokenizer: DynTokenizer,
    text: String,
    token_ids: Vec<u32>,
}

impl BenchFixture {
    /// Loads the checkpoint's tokenizer and renders the sample, and encodes and decodes it once,
    /// so the measured calls repeat calls known to succeed.
    async fn load(arguments: &Arguments) -> Result<Self, Box<dyn Error>> {
        let hub = Hub::from_env()?;
        let program = Program::load(
            &program_testing::module!("processing-program"),
            proto::module::PackageRevision {
                backend: "processing".to_owned(),
                revision: 1,
                arguments: toml::Table::new(),
            },
        )?;
        let loaded = ModelConfig::load(
            &hub,
            &Source::Hub(HubAsset::parse(&arguments.checkpoint)?),
            None,
            None,
            &Default::default(),
            &serde_json::Map::new(),
            &program,
        )
        .await?;
        let LoadedBinding::Chat(binding) = loaded.binding else {
            return Err(format!("{} is not a chat checkpoint", arguments.checkpoint).into());
        };
        let tokenizer = loaded.tokenizer;

        let messages = (0..arguments.turns)
            .flat_map(|turn| {
                [
                    serde_json::json!({"role": "user", "content": format!("Summarize request {turn}.")}),
                    serde_json::json!({"role": "assistant", "content": format!("Request {turn} asks for a summary.")}),
                ]
            })
            .collect::<Vec<_>>();
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": arguments.checkpoint,
            "messages": messages,
        }))?;
        let mut session = binding.checkout()?;
        let prompt = session
            .render(&daemon::api::input::chat::render(&request, &[]))?
            .map_err(|refusal| refusal.message)?;
        let text: String = prompt
            .segments
            .into_iter()
            .map(|segment| match segment {
                Segment::Text(text) => text,
                Segment::Image(_) => String::new(),
            })
            .collect();
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

/// Loads the fixture once, then runs both groups as `criterion_main!` would.
fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse();
    let fixture = tokio::runtime::Runtime::new()?.block_on(BenchFixture::load(&arguments))?;
    let mut criterion = Criterion::default();
    if let Some(filter) = &arguments.filter {
        criterion = criterion.with_filter(filter);
    }
    bench_encode(&mut criterion, &fixture);
    bench_decode(&mut criterion, &fixture);
    criterion.final_summary();
    Ok(())
}
