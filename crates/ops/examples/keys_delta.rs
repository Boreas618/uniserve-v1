//! The two-way delta between a fixture's rows and a model's own walk.
//!
//! ```text
//! cargo run -p ops --example keys_delta -- --fixture A.txt --walk B.txt
//! ```
//!
//! Both inputs are prints in the fixture's format (`operator<TAB>key`, `#` lines ignored), so this
//! tool needs no dependency on the model crate and compares any pair: the `ops` mirror against a
//! model's walk, or one walk against another after a loader or layer changed.
//!
//! It answers the question a one-way check cannot: **which rows exist on only one side, and which
//! operator is responsible.** A check that a fixture equals its generator proves the two agree; it
//! does not prove either is right, and a mirror whose rows were derived from a design rather than from
//! a launch is exactly the case where the two agree and both are wrong. So the delta is reported in
//! both directions, per operator, and the walk's own `not walked:` lines are echoed — **a partial walk
//! must not read as a complete one.**
//!
//! Exit status is 0 when the rows are equal, 1 when they differ, and 2 when an input cannot be read;
//! the delta is the output either way, because the difference is the finding rather than the error.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut fixture: Option<std::path::PathBuf> = None;
    let mut walk: Option<std::path::PathBuf> = None;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next();
        match argument.as_str() {
            "--fixture" => fixture = value().map(std::path::PathBuf::from),
            "--walk" => walk = value().map(std::path::PathBuf::from),
            other => {
                eprintln!("unknown argument {other}");
                return usage();
            }
        }
    }
    let (Some(fixture), Some(walk)) = (fixture, walk) else {
        return usage();
    };

    let (fixture_rows, _) = match read(&fixture) {
        Ok(rows) => rows,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let (walk_rows, not_walked) = match read(&walk) {
        Ok(rows) => rows,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    if !not_walked.is_empty() {
        println!("# the walk states {} component(s) it does not walk:", not_walked.len());
        for component in &not_walked {
            println!("#   not walked: {component}");
        }
        println!("# a delta below covers only the components the walk reached");
    }

    println!("# {} rows in the fixture, {} in the walk", fixture_rows.len(), walk_rows.len());

    let only_fixture: Vec<&String> = fixture_rows.difference(&walk_rows).collect();
    let only_walk: Vec<&String> = walk_rows.difference(&fixture_rows).collect();
    println!(
        "# {} rows only in the fixture, {} only in the walk, {} in both",
        only_fixture.len(),
        only_walk.len(),
        fixture_rows.intersection(&walk_rows).count()
    );

    // The dangerous class, named separately: a fixture row whose operator the walk *does* launch but
    // whose key it does not state. A row like that is not merely absent — a store built from it holds
    // a program that launches and returns a number for a geometry the model does not serve, and no
    // package check can see it, because the package exists and the key builds.
    let walk_operators: BTreeSet<&str> = walk_rows
        .iter()
        .filter_map(|row| row.split_once('\t').map(|(operator, _)| operator))
        .collect();
    let (dangerous, unbound): (Vec<&String>, Vec<&String>) = only_fixture
        .iter()
        .partition(|row| {
            row.split_once('\t')
                .is_some_and(|(operator, _)| walk_operators.contains(operator))
        });

    report(
        "only in the fixture, operator the walk also launches",
        &dangerous,
        "a program that would launch for a geometry the walk does not state",
    );
    report(
        "only in the fixture, operator the walk never launches",
        &unbound,
        "a replaced design, or a component the walk has not bound",
    );
    report("only in the walk", &only_walk, "the walk states a program the fixture lacks");

    if only_fixture.is_empty() && only_walk.is_empty() {
        println!("# verdict: the fixture is the walk's rows");
        ExitCode::SUCCESS
    } else {
        println!(
            "# verdict: {} difference(s) — a store must not be built from this fixture",
            only_fixture.len() + only_walk.len()
        );
        ExitCode::FAILURE
    }
}

/// The rows of one print, and the components it says it did not walk.
fn read(path: &Path) -> Result<(BTreeSet<String>, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut rows = BTreeSet::new();
    let mut not_walked = Vec::new();
    for line in text.lines() {
        if let Some(component) = line.strip_prefix("# not walked: ") {
            not_walked.push(component.trim().to_owned());
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        rows.insert(line.trim_end().to_owned());
    }
    Ok((rows, not_walked))
}

/// One direction of the delta, grouped by operator so the responsible component is visible.
fn report(heading: &str, rows: &[&String], why: &str) {
    if rows.is_empty() {
        return;
    }
    let mut by_operator: BTreeMap<String, u32> = BTreeMap::new();
    for row in rows {
        let operator = row
            .split_once('\t')
            .map_or_else(|| (*row).clone(), |(operator, _)| operator.to_owned());
        *by_operator.entry(operator).or_default() += 1;
    }
    println!("# {heading}: {why}");
    for (operator, count) in by_operator {
        println!("#   {operator}: {count} row(s)");
    }
    for row in rows.iter().take(40) {
        println!("{row}");
    }
    if rows.len() > 40 {
        println!("# … {} more", rows.len() - 40);
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: keys_delta --fixture <print> --walk <print>");
    ExitCode::from(2)
}
