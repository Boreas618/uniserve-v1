//! The schedule diagnosis: the sigma points and network times one checkpoint's schedule evaluates,
//! written where a comparator can read them beside the reference's own evaluation.
//!
//! Host-only: a schedule is arithmetic on the checkpoint's statement (`scheduler_config.json` and
//! `fastvideo_inference.json`), so this runs without a device or a weight. It is a **diagnosis**
//! (design 10), never an acceptance gate: a row that differs names the step whose sigma the two
//! implementations disagree about, and nothing about a build.
//!
//! ```text
//! cargo run -q -p minimax-h3 --example schedule_dump -- --root /opt/uv/models/MiniMax-H3 \
//!     --out artifacts/diagnosis/minimax-h3/schedule --commit <sha>
//! ```
//!
//! Writes `schedule-<dit>.json` per DiT: the grid's kind, both shifts, the number of steps, and every
//! sigma point with the network time it implies. The checkpoint path and the commit travel with the
//! numbers, because a sigma ladder with no commit beside it is a number nobody can date.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use minimax_h3::checkpoint::{Dit, Layout, Sidecars};
use minimax_h3::config::Config;
use minimax_h3::schedule::{Modality, Schedule};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut root = PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut out: Option<PathBuf> = None;
    let mut commit = String::from("unstated");
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => {
                root = arguments
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--root wants a path")?
            }
            "--out" => out = arguments.next().map(PathBuf::from),
            "--commit" => commit = arguments.next().ok_or("--commit wants a revision")?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let layout = Layout::detect(&root)?;
    let sidecars = Sidecars::read(&root, None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;

    for dit in [Dit::Base, Dit::Reference] {
        let denoiser = &config.denoisers[&dit];
        let schedule = Schedule::new(&denoiser.schedule)?;
        let video = schedule.sigmas(Modality::Video);
        let audio = schedule.sigmas(Modality::Audio);
        println!(
            "schedule {}: {} grid, {} video points, {} audio points, steps {}",
            dit.directory(),
            denoiser.schedule.grid.kind(),
            video.len(),
            audio.len(),
            schedule.num_steps()?,
        );

        let mut json = String::new();
        writeln!(json, "{{")?;
        writeln!(json, "  \"checkpoint\": {:?},", root.display().to_string())?;
        writeln!(
            json,
            "  \"generator\": \"cargo run -q -p minimax-h3 --example schedule_dump\","
        )?;
        writeln!(json, "  \"commit\": {commit:?},")?;
        writeln!(json, "  \"dit\": {:?},", dit.directory())?;
        writeln!(json, "  \"grid\": {:?},", denoiser.schedule.grid.kind())?;
        writeln!(
            json,
            "  \"statement\": {},",
            statement(&denoiser.schedule.grid)
        )?;
        writeln!(json, "  \"steps\": {},", schedule.num_steps()?)?;
        // One block per modality, joined with a comma: a trailing one is not JSON, and the file is
        // read by a comparator rather than by a person.
        let mut blocks = Vec::new();
        for modality in [Modality::Video, Modality::Audio] {
            let sigmas = schedule.sigmas(modality);
            let times = schedule.timesteps(modality);
            blocks.push(format!(
                "  {:?}: {{\n    \"shift\": {},\n    \"sigmas\": [{}],\n    \"timesteps\": [{}]\n  }}",
                modality.as_str(),
                schedule.shift(modality),
                sigmas
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                times
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
        writeln!(json, "{}", blocks.join(",\n"))?;
        writeln!(json, "}}")?;
        if let Some(directory) = &out {
            write_group(directory, dit.directory(), &json)?;
        }
    }
    Ok(())
}

/// One DiT's file under `directory`, and the directory made when it is absent.
fn write_group(directory: &Path, dit: &str, json: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!("schedule-{dit}.json"));
    std::fs::write(&path, json)?;
    println!("wrote {}", path.display());
    Ok(())
}

/// The statement the grid was evaluated from, so a comparator rebuilds the reference's own grid from
/// the same fields rather than from this dump's numbers.
fn statement(grid: &minimax_h3::config::ScheduleGrid) -> String {
    use minimax_h3::config::ScheduleGrid;
    match grid {
        ScheduleGrid::Uniform { points } => {
            format!("{{\"kind\": \"uniform\", \"points\": {points}}}")
        }
        ScheduleGrid::Dmd {
            rungs,
            num_train_timesteps,
        } => format!(
            "{{\"kind\": \"dmd\", \"rungs\": [{}], \"num_train_timesteps\": {num_train_timesteps}}}",
            rungs
                .iter()
                .map(|rung| rung.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ScheduleGrid::Blocks {
            intervals,
            nodes,
            max_t,
        } => format!(
            "{{\"kind\": \"blocks\", \"intervals\": {intervals}, \"nodes\": [{}], \"max_t\": {max_t}}}",
            nodes
                .iter()
                .map(|node| node.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
