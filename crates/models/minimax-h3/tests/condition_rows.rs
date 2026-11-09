//! The request rules count a request's condition rows before the request reaches a rank, and the
//! count selects the graph the rank replays. A count below the packing's would select a layout that
//! cannot hold the request, and one above it a layout larger than the request needs, so the two
//! statements are held against each other here over every kind of condition.
#![cfg(test)]

use minimax_h3::packing::{self, Condition, Frame, Role};
use minimax_h3::processing::plan::{
    AudioClip, ConditionInput, FramePosition, KeyframeFit, Prepared, VideoClip,
};
use minimax_h3::processing::{ConditionPacking, FrameClips};
use proto::video::Canvas;

const CLIPS: FrameClips = FrameClips {
    min_frames: 22,
    clip_frames: 17,
    clip_latent_frames: 5,
    tail_latent_frames: 2,
};

const CLIP: minimax_h3::encoding::Clip = minimax_h3::encoding::Clip {
    frames: 17,
    latents: 5,
    token_drop: 3,
};

const TARGET: Canvas = Canvas {
    width: 1344,
    height: 768,
};

fn canvas(width: u32, height: u32) -> minimax_h3::config::Canvas {
    minimax_h3::config::Canvas {
        short_edge: width.min(height),
        aspect: (width, height),
        width,
        height,
    }
}

fn rule_rows(plan: &ConditionInput, tile: u32) -> u32 {
    let (time, height, width) = packing::tile_shape(tile).unwrap();
    plan.packed_rows(
        TARGET,
        &ConditionPacking {
            capacity: u32::MAX,
            tile_rows: tile,
            video_tile: [time, height, width],
        },
    )
}

fn packed_rows(condition: Condition, tile: u32) -> u32 {
    let target = canvas(TARGET.width, TARGET.height);
    packing::condition_segments(&[condition], &target, CLIP)
        .unwrap()
        .iter()
        .map(|segment| packing::segment_tiles(segment, tile).unwrap() * tile)
        .sum()
}

fn audio(latents: u32) -> AudioClip {
    AudioClip {
        sample_rate: 32_000,
        start_sample: 0,
        source_samples: u64::from(latents) * 800,
        samples: u64::from(latents) * 800,
        latents,
    }
}

fn plan(prepared: Prepared, audio_rows: u32) -> ConditionInput {
    ConditionInput {
        index: 0,
        prepared,
        vision: None,
        audio_rows,
    }
}

#[test]
fn a_rule_counts_a_condition_as_the_packing_does() {
    for tile in [64, 128] {
        let keyframe = plan(
            Prepared::Keyframe(KeyframeFit {
                position: FramePosition::First,
                cover_crop: None,
            }),
            0,
        );
        let packed = Condition {
            role: Role::FirstFrame,
            frame: Some(Frame {
                canvas: canvas(TARGET.width, TARGET.height),
                num_frames: 1,
            }),
            audio_samples: 0,
        };
        assert_eq!(rule_rows(&keyframe, tile), packed_rows(packed, tile));

        // The released image sizes, the enlarged one of each aspect among them.
        for (width, height) in [
            (3648, 2048),
            (2560, 1440),
            (2048, 2048),
            (768, 768),
            (1088, 800),
        ] {
            let image = plan(Prepared::Image(Canvas { width, height }), 0);
            let packed = Condition {
                role: Role::Reference,
                frame: Some(Frame {
                    canvas: canvas(width, height),
                    num_frames: 1,
                }),
                audio_samples: 0,
            };
            assert_eq!(rule_rows(&image, tile), packed_rows(packed, tile));
        }

        for latents in [1, 63, 64, 200, 240] {
            let voice = audio(latents);
            let track = plan(Prepared::Audio(voice), voice.rows());
            let packed = Condition {
                role: Role::Reference,
                frame: None,
                audio_samples: latents * minimax_h3::AUDIO_HOP,
            };
            assert_eq!(rule_rows(&track, tile), packed_rows(packed, tile));
        }

        // A reference video of every legal length a five-to-fifteen-second target reads, on a
        // canvas whose grid the tiles divide and on one they do not.
        for (width, height) in [(1344, 768), (800, 1088), (768, 768)] {
            let mut frames = CLIPS.min_frames;
            while frames <= 379 {
                for latents in [0, 240] {
                    let track = (latents > 0).then(|| audio(latents));
                    let video = plan(
                        Prepared::Video(VideoClip {
                            canvas: Canvas { width, height },
                            start_frame: 0,
                            frames,
                            vae_frames: frames,
                            vae_clips: CLIPS,
                            soundtrack: track,
                        }),
                        track.map_or(0, |track| track.rows()),
                    );
                    let packed = Condition {
                        role: Role::Reference,
                        frame: Some(Frame {
                            canvas: canvas(width, height),
                            num_frames: frames,
                        }),
                        audio_samples: latents * minimax_h3::AUDIO_HOP,
                    };
                    assert_eq!(rule_rows(&video, tile), packed_rows(packed, tile));
                }
                frames += CLIPS.clip_frames;
            }
        }
    }
}
