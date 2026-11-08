//! Print the `slot_commit` keys a runner of a deployment's buckets launches.
//!
//! The commit is a runner-owned operator: no model calls it, so the model's own key
//! enumeration cannot name it and a store built from that enumeration alone stops a
//! run at the first generation. This example is the other half of the list - it calls
//! [`commit_descriptors`], the same function the epoch calls, so the keys it prints
//! are the keys the runner asks for and not a second statement of them.
//!
//! The buckets are the deployment's, which are configuration: pass them.
//!
//! ```text
//! cargo run -p worker --example runner_keys -- \
//!     --rows 1,2,4,8 --tokens 128,512 --sampled true --vocabulary 151936
//! ```
//!
//! The output is `operator<TAB>key` lines, which is what
//! `tools/build_qwen3_fixture.sh` unions with the model's rows.

use worker::plan::Buckets;
use worker::runners::causal_lm::commit_descriptors;

fn main() -> Result<(), String> {
    let mut rows: Vec<u32> = Vec::new();
    let mut tokens: Vec<u32> = Vec::new();
    let mut sampled = false;
    let mut vocabulary: Option<u32> = None;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        match argument.as_str() {
            "--rows" => rows = list(&value()?)?,
            "--tokens" => tokens = list(&value()?)?,
            "--sampled" => sampled = matches!(value()?.as_str(), "true" | "1"),
            "--vocabulary" => {
                vocabulary = Some(value()?.parse().map_err(|_| "--vocabulary".to_string())?)
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }

    let vocabulary = vocabulary.ok_or("--vocabulary is the deployment's and has no default")?;
    if rows.is_empty() || tokens.is_empty() {
        return Err("--rows and --tokens are the deployment's and have no default".to_string());
    }

    let buckets = Buckets::causal_lm(&rows, &tokens, sampled);
    let descriptors = commit_descriptors(&buckets, vocabulary);

    println!(
        "# slot_commit keys for rows={rows:?} tokens={tokens:?} sampled={sampled} vocabulary={vocabulary}"
    );
    println!(
        "# {} keys, one per distinct launch block",
        descriptors.len()
    );
    for descriptor in descriptors {
        println!("slot_commit\t{descriptor}");
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
