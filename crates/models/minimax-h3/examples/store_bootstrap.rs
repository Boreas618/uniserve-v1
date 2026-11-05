//! Every row the model's layers state, unioned over a ladder of deployments.
//!
//! This is **not a fixture**. A fixture is *one* deployment's rows: `examples/programs` prints them
//! from the layers' own descriptor constructors, and `examples/h3_fixture --walk <print> --commit
//! <rev>` writes the file with the tree it came from. A walk-derived fixture is therefore complete
//! for the components the walk reached and silent about the rest - which is the property that makes
//! it a record rather than a guess.
//!
//! A component that has **never run** cannot have been walked, so it cannot appear in any fixture yet.
//! To let it run once and write its own fixture, the store it launches against must be a superset:
//! every row every layer states, over every deployment shape the build serves. That union is this
//! file's whole job.
//!
//! It states no rows of its own. It runs `examples/programs` once per deployment in the ladder and
//! unions the results, so a component added to that generator is in the bootstrap by construction:
//! the failure mode of this file is being mistaken for a fixture, never being a second list.
//!
//! ```text
//! cargo build -p minimax-h3 --example programs
//! cargo run -p minimax-h3 --example store_bootstrap -- \
//!     --programs target/release/examples/programs --root /workspace/models/MiniMax-H3 \
//!     --dtypes bf16,f16 --ulysses 1,2 --conditioned true --samples 256 \
//!     --rasters 480x832 --tokens 4096 --rows 32000 --elements 123456 --track 44100 --tp 1 \
//!     > notes/programs/minimax-h3-bootstrap.txt
//! ```
//!
//! The rows are printed in the fixture's own format (`operator<TAB>key`), so the store's tools read
//! this file as they read a fixture. The header says which it is, because the two files differ in
//! exactly the way that matters: one is evidence about a tree, the other is permission to run.

use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut programs: Option<PathBuf> = None;
    let mut root: Option<String> = None;
    let mut dtypes: Vec<String> = Vec::new();
    let mut degrees: Vec<String> = Vec::new();
    // Everything the deployment generator needs, passed through unchanged: this file is a driver, and
    // a parameter it invented would be a second opinion about how a deployment is described.
    let mut passed: Vec<(String, String)> = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--programs" => programs = Some(PathBuf::from(value)),
            "--root" => root = Some(value),
            "--dtypes" => dtypes.extend(value.split(',').map(|item| item.trim().to_owned())),
            "--ulysses" => degrees.extend(value.split(',').map(|item| item.trim().to_owned())),
            other => passed.push((other.to_owned(), value)),
        }
    }
    let programs = programs.ok_or("--programs names the built `programs` example")?;
    let root = root.ok_or("--root names the pipeline export's root")?;
    if dtypes.is_empty() {
        return Err("--dtypes names the element types the build serves, such as bf16,f16".into());
    }
    if degrees.is_empty() {
        return Err("--ulysses names the Ulysses degrees the build serves, such as 1,2".into());
    }
    if !programs.is_file() {
        return Err(format!(
            "{}: build it with `cargo build -p minimax-h3 --example programs`",
            programs.display()
        )
        .into());
    }

    let mut rows: Vec<String> = Vec::new();
    let mut ran: Vec<String> = Vec::new();
    for dtype in &dtypes {
        for degree in &degrees {
            let mut command = Command::new(&programs);
            command
                .arg(&root)
                .arg("--dtype")
                .arg(dtype)
                .arg("--ulysses")
                .arg(degree);
            for (flag, value) in &passed {
                command.arg(flag).arg(value);
            }
            let output = command
                .output()
                .map_err(|error| format!("{}: {error}", programs.display()))?;
            let text = String::from_utf8_lossy(&output.stdout);
            if !output.status.success() {
                // A deployment the generator refuses is a deployment the build does not serve: say
                // which, with the generator's own sentence, and go on. A ladder that stops at its
                // first refusal is a ladder nobody can read.
                let complaint = String::from_utf8_lossy(&output.stderr);
                eprintln!(
                    "not walked: dtype {dtype}, ulysses {degree}: {}",
                    complaint.trim().split('\n').next().unwrap_or("")
                );
                continue;
            }
            let mut count = 0;
            for line in text.lines() {
                if line.starts_with('#') || line.trim().is_empty() {
                    continue;
                }
                rows.push(line.to_owned());
                count += 1;
            }
            ran.push(format!("{dtype} x{degree}: {count} rows"));
        }
    }
    if ran.is_empty() {
        return Err("the ladder ran nothing: every deployment was refused".into());
    }
    rows.sort();
    rows.dedup();
    let operators: std::collections::BTreeSet<&str> = rows
        .iter()
        .filter_map(|row| row.split('\t').next())
        .collect();

    println!(
        "# bootstrap, not a fixture: {} rows, {} operators",
        rows.len(),
        operators.len()
    );
    println!(
        "# every row is a statement by the model's own layers: this file runs the deployment \
         generator once per deployment and unions the results"
    );
    println!(
        "# ladder: dtypes [{}], ulysses [{}] - each run's row count follows",
        dtypes.join(", "),
        degrees.join(", ")
    );
    for line in &ran {
        println!("#   {line}");
    }
    println!(
        "# a fixture is ONE deployment's rows (examples/programs, then examples/h3_fixture \
         --walk <print> --commit <rev>): this file says a component exists, a fixture says it ran"
    );
    for row in &rows {
        println!("{row}");
    }
    Ok(())
}
