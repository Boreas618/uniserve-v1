//! Print the keys of the runner-owned operators a runner of a deployment's buckets launches.
//!
//! `slot_commit` and the step's `logits_gather` of block tables and input tokens are the runner's:
//! no model calls them, so the model's own key enumeration cannot name them, and a store built from
//! that enumeration alone stops a run at the first warmup. This example is the other half of the
//! list. It calls [`commit_descriptors`] and [`gather_descriptors`], the functions the runner
//! launches from, so the keys it prints are the keys the runner asks for and not a second statement
//! of them.
//!
//! The buckets, the model length and the cache's page size are the deployment's, which are
//! configuration: pass them. The block tables are as wide as a request of the model length fills
//! ([`block_table_width`]).
//!
//! ```text
//! cargo run -p worker --example runner_keys -- \
//!     --rows 1,2,4,8 --tokens 128,512 --sampled true --vocabulary 151936 \
//!     --max-model-len 40960 --page-size 64
//! ```
//!
//! The output is `operator<TAB>key` lines, which is what
//! `tools/build_qwen3_fixture.sh` unions with the model's rows.

use worker::capacity::block_table_width;
use worker::plan::Buckets;
use worker::runners::causal_lm::{commit_descriptors, gather_descriptors};

fn main() -> Result<(), String> {
    let mut rows: Vec<u32> = Vec::new();
    let mut tokens: Vec<u32> = Vec::new();
    let mut sampled = false;
    let mut vocabulary: Option<u32> = None;
    let mut max_model_len: Option<u32> = None;
    let mut page_size: Option<u32> = None;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        let number = |text: String, name: &str| -> Result<u32, String> {
            text.parse()
                .map_err(|_| format!("{name} takes a count, not {text:?}"))
        };
        match argument.as_str() {
            "--rows" => rows = list(&value()?)?,
            "--tokens" => tokens = list(&value()?)?,
            "--sampled" => sampled = matches!(value()?.as_str(), "true" | "1"),
            "--vocabulary" => vocabulary = Some(number(value()?, "--vocabulary")?),
            "--max-model-len" => max_model_len = Some(number(value()?, "--max-model-len")?),
            "--page-size" => page_size = Some(number(value()?, "--page-size")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }

    let vocabulary = vocabulary.ok_or("--vocabulary is the deployment's and has no default")?;
    let max_model_len =
        max_model_len.ok_or("--max-model-len is the deployment's and has no default")?;
    let page_size = page_size
        .filter(|size| *size > 0)
        .ok_or("--page-size is the deployment's cache page and has no default")?;
    if rows.is_empty() || tokens.is_empty() {
        return Err("--rows and --tokens are the deployment's and have no default".to_string());
    }

    let buckets = Buckets::causal_lm(&rows, &tokens, sampled);
    let width = block_table_width(max_model_len, page_size);
    let commits = commit_descriptors(&buckets, vocabulary);
    let gathers = gather_descriptors(&buckets, width);

    println!(
        "# runner keys for rows={rows:?} tokens={tokens:?} sampled={sampled} \
         vocabulary={vocabulary} max_model_len={max_model_len} page_size={page_size}"
    );
    println!(
        "# {} slot_commit keys, one per distinct launch block; {} logits_gather keys over block \
         tables {width} pages wide",
        commits.len(),
        gathers.len()
    );
    for descriptor in commits {
        println!("slot_commit\t{descriptor}");
    }
    for descriptor in gathers {
        println!("logits_gather\t{descriptor}");
    }

    Ok(())
}

/// Parse `1,2,4,8` into the list it names.
fn list(value: &str) -> Result<Vec<u32>, String> {
    value
        .split(',')
        .map(|item| {
            item.trim()
                .parse()
                .map_err(|_| format!("{item:?} is not a row or token count"))
        })
        .collect()
}
