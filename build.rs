//! The FFmpeg run path `uniserve` carries (`crates/media/build_support/run_path.rs`).

include!("crates/media/build_support/ffmpeg_prefix.rs");
include!("crates/media/build_support/run_path.rs");

fn main() {
    state_run_path("crates/media/build_support");
}
