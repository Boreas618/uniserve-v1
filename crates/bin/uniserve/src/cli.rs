//! Command-line parsing for the `uniserve` binary.
//!
//! The binary's own options, its log levels, are global: every subcommand
//! takes them. Each subcommand declares its own options; `serve` takes the
//! serving command line (`uniserve_server::options::ServeArgs`), which
//! resolves one serving configuration from a `--config` file, a few flags
//! and `--set` assignments.

use clap::{Args, Parser, Subcommand};
use uniserve_server::options::ServeArgs;

/// Top-level parser for the `uniserve` binary.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve",
    about = "UniServe OpenAI-compatible server: Rust engine + scheduler, Python forwards only."
)]
pub(crate) struct Cli {
    #[command(flatten)]
    pub log: LogLevels,
    #[command(subcommand)]
    pub command: Command,
}

/// The process's log levels, which every subcommand takes.
#[derive(Debug, Clone, Args)]
#[command(next_help_heading = "Logging")]
pub(crate) struct LogLevels {
    /// Log level of UniServe components, such as `info` or `debug`.
    #[arg(long, global = true)]
    pub log_level: Option<String>,
    /// Log level of the HTTP server components.
    #[arg(long, global = true)]
    pub log_level_http: Option<String>,
}

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run the UniServe OpenAI server: the Rust engine and scheduler run
    /// in-process, driving a forward-only worker.
    #[command(
        override_usage = "uniserve serve [MODEL] [OPTIONS] [--config FILE] [--set PATH=VALUE]..."
    )]
    Serve(Box<ServeArgs>),
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use uniserve_server::Config;

    use super::*;

    fn serve(args: &[&str]) -> Result<ServeArgs, clap::Error> {
        let cli = Cli::try_parse_from(["uniserve", "serve"].iter().chain(args))?;
        let Command::Serve(serve) = cli.command;
        Ok(*serve)
    }

    fn lowered(args: &[&str]) -> anyhow::Result<Config> {
        let (config, stated) = serve(args)?.resolve()?;
        config.into_config(&stated)
    }

    /// Writes `body` to a file of this test process and thread and returns
    /// its path.
    fn written(name: &str, body: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "uniserve-{name}-{}-{:?}.toml",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, body).expect("write test file");
        path
    }

    #[test]
    fn the_model_is_required_from_the_command_line_or_the_file() {
        let error = lowered(&[]).unwrap_err();
        assert!(error.to_string().contains("`model` is required"), "{error}");
        assert_eq!(lowered(&["/models/m"]).unwrap().model, "/models/m");

        let file = written("model", "model = \"/models/file\"\n");
        let config = lowered(&["--config", file.to_str().unwrap()]).unwrap();
        assert_eq!(config.model, "/models/file");
    }

    /// The log levels are the binary's global options, stated before the
    /// subcommand or after it.
    #[test]
    fn log_levels_are_global() {
        let levels = |args: &[&str]| {
            let cli = Cli::try_parse_from(args).expect("parses");
            (cli.log.log_level, cli.log.log_level_http)
        };
        assert_eq!(
            levels(&["uniserve", "--log-level", "debug", "serve", "m"]),
            (Some("debug".into()), None)
        );
        assert_eq!(
            levels(&["uniserve", "serve", "m", "--log-level-http", "warn"]),
            (None, Some("warn".into()))
        );
    }

    /// Without a deployment file each replica runs one component
    /// tensor-parallel over its ranks; the placement shorthand is refused
    /// beside a file.
    #[test]
    fn the_shorthand_places_replicas_and_is_refused_beside_a_file() {
        let config = lowered(&[
            "m",
            "--tensor-parallel-size",
            "2",
            "--data-parallel-size",
            "2",
            "--set",
            "placement_config.device=cpu",
        ])
        .unwrap();
        let workers = &config.engine.workers;
        assert_eq!(workers.len(), 2);
        assert!(workers.iter().all(|worker| worker.ranks.len() == 2));

        let deployment = written(
            "deployment",
            "[[workers]]\nid = \"model\"\nranks = [\"cpu*2\"]\nqueue_depth = 2\n\
             [workers.components.model]\ntensor_parallel_size = 2\n",
        );
        let path = deployment.to_str().unwrap();
        assert_eq!(
            lowered(&["m", "--workers", path])
                .unwrap()
                .engine
                .workers
                .len(),
            1
        );
        for shorthand in [
            ["--tensor-parallel-size", "2"],
            ["--set", "placement_config.device=cpu"],
            ["--set", "placement_config.queue_depth=3"],
        ] {
            let mut args = vec!["m", "--workers", path];
            args.extend(shorthand);
            let error = lowered(&args).unwrap_err();
            assert!(
                error.to_string().contains("deployment file"),
                "{shorthand:?}: {error}"
            );
        }
    }

    /// A deployment rank entry without a host runs on this process's host.
    #[test]
    fn a_deployment_places_unnamed_ranks_on_this_host() {
        let deployment = written(
            "head-deployment",
            "[[workers]]\nid = \"model\"\nranks = [\"cpu\", \"rank-1/cpu\"]\nqueue_depth = 2\n\
             [workers.components.model]\ntensor_parallel_size = 2\n",
        );
        let config = lowered(&[
            "m",
            "--workers",
            deployment.to_str().unwrap(),
            "--host-identity",
            "rank-0",
        ])
        .unwrap();
        let nodes = config.engine.workers[0]
            .ranks
            .iter()
            .map(|rank| rank.node.as_str())
            .collect::<Vec<_>>();
        assert_eq!(nodes, ["rank-0", "rank-1"]);
        assert!(lowered(&["m", "--workers", "/nonexistent/deployment.toml"]).is_err());
    }

    /// Capture lists and capacities are checked before any rank starts.
    #[test]
    fn capture_lists_are_strictly_increasing_and_shapes_distinct() {
        let config = lowered(&[
            "m",
            "--set",
            "video_config.text_capacities=1024,4096",
            "--set",
            "cuda_graph_config.flow_shapes=1152x2048,2048x1152",
        ])
        .unwrap();
        let worker = &config.engine.worker_process;
        assert_eq!(worker.video_text_capacities, Some(vec![1024, 4096]));
        assert_eq!(
            worker.flow_graph_shapes,
            Some(vec![[1152, 2048], [2048, 1152]])
        );

        for refused in [
            "video_config.text_capacities=4096,1024",
            "cuda_graph_config.decode_batch_sizes=1,0",
            "cuda_graph_config.flow_shapes=1152x2048,1152x2048",
            "cuda_graph_config.flow_shapes=1152",
        ] {
            assert!(
                lowered(&["m", "--set", refused]).is_err(),
                "{refused} must be refused"
            );
        }
    }

    /// Stated model-family tables are recorded for startup to check
    /// against the served model; defaults are not.
    #[test]
    fn stated_family_tables_are_recorded() {
        let defaults = lowered(&["m"]).unwrap();
        assert_eq!(
            defaults.family_options,
            uniserve_server::FamilyOptions::default()
        );

        let stated = lowered(&[
            "m",
            "--set",
            "readout_config.canvas=compact",
            "--set",
            "video_config.max_seconds=5",
        ])
        .unwrap();
        assert_eq!(stated.family_options.diffusion_gemma, ["readout_config"]);
        assert_eq!(stated.family_options.video, ["video_config"]);
        assert!(stated.family_options.chat.is_empty());
    }

    /// Lanes, the precision policy, the reasoning parser, expert
    /// parallelism and prefix caching reach the server configuration as
    /// stated.
    #[test]
    fn grouped_options_reach_their_owners() {
        let config = lowered(&[
            "m",
            "--set",
            "lanes=[{ lane_id = \"decode\", sm_budget = 64, domains = [\"decode\"] }]",
            "--set",
            "quantization_config.mode=performance",
            "--set",
            "chat_config.reasoning_parser=none",
            "--set",
            "moe_config={ expert_parallel = true, exchange = \"dwdp\" }",
            "--set",
            "kv_cache_config.prefix_caching=false",
        ])
        .unwrap();
        let worker = &config.engine.worker_process;
        assert_eq!(worker.lanes.len(), 1);
        assert_eq!(worker.quantization_config["mode"], "performance");
        assert!(!config.reasoning_parsing);
        assert_eq!(
            config.engine.expert_parallel,
            Some(uniserve_engine::ExpertExchange::Dwdp)
        );
        assert!(!config.engine.prefix_cache);

        let lane = "lanes=[{ lane_id = \"decode\", sm_budget = 0, domains = [\"decode\"] }]";
        assert!(lowered(&["m", "--set", lane]).is_err());
    }
}
