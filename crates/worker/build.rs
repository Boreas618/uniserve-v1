//! The FFmpeg run path the worker's unit tests carry (`crates/media/build_support/run_path.rs`).

include!("../media/build_support/ffmpeg_prefix.rs");
include!("../media/build_support/run_path.rs");

fn main() {
    state_run_path("../media/build_support");
}
