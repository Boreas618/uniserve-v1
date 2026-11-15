//! The worker-group layouts the engine's media tests place a pipeline in, each stated once as the
//! `[[worker_groups]]` of a deployment file and expanded by the configuration's own parser.
//!
//! The tests state the layouts they exercise here instead of reading the example deployments a
//! release ships, so an example may change without changing what a test covers. The two-host
//! layouts name their hosts `rank-0` and `rank-1`, four devices each.

use proto::config::WorkerGroupConfig;

/// FastH3 on one host of eight devices: the `text` group's eight ranks encode prompts eight-way
/// tensor parallel, and eight one-rank `flow` replicas serve every other phase, replica `i` on
/// the device of text rank `i`, each reading and muxing its own requests.
const FASTH3_DP8_TEXT_TP8: &str = r#"
[[worker_groups]]
id = "text"
ranks = ["cuda:0-7"]
queue_depth = 24
mem_fraction_static = 0.18

[worker_groups.blocks.text_encoder]
tensor_parallel_size = 8

[[worker_groups]]
id = "flow"
replicas = 8
ranks = ["cuda:0-7"]
queue_depth = 6
mem_fraction_static = 0.81

[worker_groups.blocks.transformer]

[worker_groups.blocks.video_decoder]

[worker_groups.blocks.audio_decoder]
distribution = "temporal_units"

[worker_groups.blocks.latent_encoder]
distribution = "temporal_units"

[worker_groups.blocks.video_codec]

[worker_groups.blocks.muxer]
ranks = [0]

[worker_groups.blocks.media_reader]
ranks = [0]
"#;

/// FastH3 in two replicas, one per host: each host's four ranks encode prompts four-way tensor
/// parallel in its `text` replica and denoise four-way Ulysses parallel in its `flow` replica, so
/// no product crosses the hosts.
const FASTH3_ULYSSES4X2_TWO_HOST: &str = r#"
[[worker_groups]]
id = "text"
replicas = 2
ranks = ["rank-0/cuda:0-3", "rank-1/cuda:0-3"]
queue_depth = 6
mem_fraction_static = 0.26

[worker_groups.blocks.text_encoder]
tensor_parallel_size = 4

[[worker_groups]]
id = "flow"
replicas = 2
ranks = ["rank-0/cuda:0-3", "rank-1/cuda:0-3"]
queue_depth = 6
mem_fraction_static = 0.73

[worker_groups.blocks.transformer]
sequence_parallel = { kind = "ulysses", ulysses_degree = 4, projection = "exchange" }

[worker_groups.blocks.video_decoder]

[worker_groups.blocks.audio_decoder]
distribution = "temporal_units"

[worker_groups.blocks.latent_encoder]
distribution = "temporal_units"

[worker_groups.blocks.video_codec]

[worker_groups.blocks.muxer]
ranks = [0]

[worker_groups.blocks.media_reader]
ranks = [0]
"#;

/// One Ulysses8 replica of `denoiser` over both hosts' ranks (the `model` group's ranks 0-3 on
/// `rank-0`, 4-7 on `rank-1`), beside a `muxer` group and a `reader` group of one rank each on
/// `rank-0`. `text_encoder`, `latent_encoder` and `reader` are further lines of the text
/// encoder's block, the latent encoder's block and the reader's group.
fn ulysses8(denoiser: &str, text_encoder: &str, latent_encoder: &str, reader: &str) -> String {
    format!(
        r#"
[[worker_groups]]
id = "model"
ranks = ["rank-0/cuda:0-3", "rank-1/cuda:0-3"]
queue_depth = 6

[worker_groups.blocks.{denoiser}]
sequence_parallel = {{ kind = "ulysses", ulysses_degree = 8, projection = "exchange" }}

[worker_groups.blocks.text_encoder]
{text_encoder}

[worker_groups.blocks.video_decoder]

[worker_groups.blocks.audio_decoder]
distribution = "temporal_units"

[worker_groups.blocks.latent_encoder]
{latent_encoder}distribution = "temporal_units"

[worker_groups.blocks.video_codec]

[[worker_groups]]
id = "muxer"
ranks = ["rank-0/cuda:0"]
queue_depth = 6

[worker_groups.blocks.muxer]

[[worker_groups]]
id = "reader"
ranks = ["rank-0/cuda:0"]
queue_depth = 6
{reader}
[worker_groups.blocks.media_reader]
"#
    )
}

/// The groups of `text`, a layout's `[[worker_groups]]`, on `engine_host`.
#[expect(
    clippy::unwrap_used,
    reason = "every layout is a constant the parser accepts"
)]
fn parsed(text: &str, engine_host: &str) -> Vec<WorkerGroupConfig> {
    WorkerGroupConfig::parse_file(text, engine_host).unwrap()
}

/// [`FASTH3_DP8_TEXT_TP8`], on `engine_host`.
pub fn fasth3_dp8_text_tp8(engine_host: &str) -> Vec<WorkerGroupConfig> {
    parsed(FASTH3_DP8_TEXT_TP8, engine_host)
}

/// [`FASTH3_ULYSSES4X2_TWO_HOST`], whose leader is `engine_host`.
pub fn fasth3_ulysses4x2_two_host(engine_host: &str) -> Vec<WorkerGroupConfig> {
    parsed(FASTH3_ULYSSES4X2_TWO_HOST, engine_host)
}

/// MiniMax-H3's dense denoiser in one Ulysses8 replica over both hosts, whose text encoder holds
/// all eight ranks eight-way tensor parallel; the leader is `engine_host`.
pub fn ulysses8_two_host(engine_host: &str) -> Vec<WorkerGroupConfig> {
    parsed(
        &ulysses8("transformer", "tensor_parallel_size = 8", "", ""),
        engine_host,
    )
}

/// [`ulysses8_two_host`] with the reference denoiser, whose text encoder holds `rank-0`'s four
/// ranks four-way tensor parallel (none of `rank-1`'s loads it), whose latent encoder deals
/// condition units to `rank-1`'s ranks first and whose reader takes 4% of its device's memory;
/// the leader is `engine_host`.
pub fn ulysses8_reference_two_host(engine_host: &str) -> Vec<WorkerGroupConfig> {
    parsed(
        &ulysses8(
            "transformer_ref",
            "ranks = [0, 1, 2, 3]\ntensor_parallel_size = 4",
            "ranks = [4, 5, 6, 7, 0, 1, 2, 3]\n",
            "mem_fraction_static = 0.04\n",
        ),
        engine_host,
    )
}
