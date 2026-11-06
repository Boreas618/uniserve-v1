//! The warm set a denoiser is bound with: one request per layout the checkpoint serves.
//!
//! [`DenoiserRunner::new`](super::denoiser::DenoiserRunner::new) takes a
//! [`WarmRequest`] for every layout the checkpoint serves, and refuses a set that states none. The
//! set is **not** the product a deployment usually names, because a layout counts a request's
//! conditioning rows and a model's padding collapses different requests onto one layout: two
//! candidates that differ in their anchors can share a bucket, and a runner that booked both would
//! capture a graph for a bucket no staged request ever selects - a graph that costs a capture and
//! serves nothing.
//!
//! So the collapse is **asked of the model** rather than assumed here. `Denoiser::layout` is a pure
//! function of the request, so the runner asks it once per candidate and keeps one candidate per
//! distinct answer, in the caller's order. What the caller's order decides is which candidate
//! represents a layout: the first one that produces it, which is why the first-seen rule is stated
//! rather than left to a map's iteration.
//!
//! # What is not here
//!
//! **The candidates themselves.** Only the model knows the `ConditionShape` of a still against a
//! clip at a canvas, and only the model knows the ladder of condition counts its checkpoint serves -
//! whether a request with one anchor and one with a reference share a layout is arithmetic the
//! model's `layout` performs over shapes this module cannot construct. A caller that enumerated
//! them here would be restating the model's padding rule in the mechanism, which is the defect that
//! put `columns_` in a key and a width in a table.

use nn::Denoiser;

use super::denoiser::WarmRequest;

/// One warm request per distinct layout, in the caller's order.
///
/// A candidate that shares a layout with an earlier one is dropped; the earlier one stays, so the
/// caller decides which request walks a bucket by ordering its candidates. The result may be empty,
/// and an empty warm set is refused where the runner is bound rather than here, so the refusal names
/// the deployment's missing statement rather than this function's input.
pub fn distinct<M: Denoiser + ?Sized>(model: &M, candidates: &[WarmRequest]) -> Vec<WarmRequest> {
    let mut kept: Vec<WarmRequest> = Vec::new();
    let mut layouts: Vec<nn::Layout> = Vec::new();
    for candidate in candidates {
        let layout = model.layout(&candidate.view());
        if !layouts.contains(&layout) {
            layouts.push(layout);
            kept.push(candidate.clone());
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    use nn::capability::{
        ConditionShape, Conditioning, LatentShape, Layout, Seed, StateTable, Step, VideoRequest,
        VideoTask,
    };
    use nn::{Error, Forward, Latents, Result};

    /// A denoiser whose layout is the only behaviour the collapse reads.
    ///
    /// Its `layout` is deliberately a *collapsing* function - a request's condition rows rounded up
    /// to a multiple of four, which is the shape of a model's per-tile padding - so the test can
    /// state a collapse without a real checkpoint's arithmetic.
    struct Padding {
        tile: u32,
    }

    impl Denoiser for Padding {
        fn layout(&self, request: &VideoRequest<'_>) -> Layout {
            let rows = request.first_frames + request.last_frames + request.references;
            let condition_rows = rows.div_ceil(self.tile) * self.tile;
            Layout {
                video: LatentShape {
                    channels: 16,
                    frames: 1,
                    height: 2,
                    width: 2,
                },
                audio: None,
                text_rows: request.text_tokens,
                condition_rows,
                condition_elements: 0,
            }
        }

        fn draw_noise(&self, _: &Forward, _: Seed, _: Layout, _: Latents) -> Result<()> {
            unreachable!("the collapse draws nothing")
        }

        fn prepare_conditioning(&self, _: &Forward, _: Conditioning<'_>) -> Result<()> {
            // The collapse reads no conditioning: it is a shape probe, not a generation.
            Ok(())
        }

        fn prepare_latents(&self, _: &Forward, _: Conditioning<'_>, _: Latents) -> Result<()> {
            unreachable!("the collapse prepares nothing")
        }

        fn denoise(&self, _: &Forward, _: Step, _: Latents, _: Conditioning<'_>) -> Result<()> {
            unreachable!("the collapse denoises nothing")
        }

        fn evaluations(&self) -> u32 {
            1
        }

        fn constant_tables(&self) -> Vec<StateTable> {
            Vec::new()
        }

        fn prepare_constants(&self, _: &Forward, _: &[ops::TensorView]) -> Result<()> {
            Err(Error::Unsupported("no constants".into()))
        }

        fn layout_tables(&self, _: &Layout) -> Result<Vec<StateTable>> {
            Ok(Vec::new())
        }

        fn prepare_layout(&self, _: &Layout, _: &mut [Vec<u8>]) -> Result<()> {
            Ok(())
        }

        fn state_tables(&self, _: &Layout) -> Vec<StateTable> {
            Vec::new()
        }

        fn prepare_state(&self, _: &VideoRequest<'_>, _: &Layout, _: &mut [Vec<u8>]) -> Result<()> {
            Ok(())
        }

        fn prepare_evaluation(&self, _: u32, _: &mut [Vec<u8>]) -> Result<()> {
            Ok(())
        }
    }

    fn warm(first: u32, last: u32, references: u32) -> WarmRequest {
        WarmRequest {
            task: VideoTask::TextToVideoAudio,
            width: 768,
            height: 1344,
            frames: 121,
            fps: 24,
            first_frames: first,
            last_frames: last,
            references,
            text_tokens: 32,
            conditions: Vec::<ConditionShape>::new(),
            vision_spans: Vec::new(),
            condition_rows: None,
        }
    }

    /// The rule the runner depends on: a layout is one bucket, so candidates the model's padding
    /// collapses into one layout become one warm request - and the survivor is the first stated.
    #[test]
    fn one_warm_request_per_layout_the_model_states() {
        let model = Padding { tile: 4 };
        // Rows 1 and 2 both round up to 4, so the first two share a layout and the third does not.
        let kept = distinct(&model, &[warm(1, 0, 0), warm(0, 2, 0), warm(0, 0, 5)]);

        assert_eq!(
            kept.len(),
            2,
            "two distinct layouts are two buckets, not three candidates"
        );
        assert_eq!(
            (kept[0].first_frames, kept[0].last_frames),
            (1, 0),
            "the first candidate of a layout is the one that walks its bucket"
        );
        assert_eq!(
            kept[1].references, 5,
            "a candidate of its own layout is kept as it was stated"
        );
    }

    /// A caller's order decides which request represents a layout, so the collapse preserves it
    /// rather than sorting by the layout - a map's order would choose a different representative
    /// and the chosen one writes the bucket's tables.
    #[test]
    fn the_callers_order_decides_the_representative() {
        let model = Padding { tile: 4 };
        let kept = distinct(&model, &[warm(0, 0, 5), warm(2, 0, 0), warm(1, 0, 0)]);

        assert_eq!(kept.len(), 2);
        assert_eq!(
            kept[0].references, 5,
            "the first stated candidate stays first"
        );
        assert_eq!(
            kept[1].last_frames, 0,
            "the collapsed pair keeps the earlier of the two"
        );
        assert_eq!(
            kept[1].first_frames, 2,
            "the second candidate of a layout survives when it is the first of that layout"
        );
    }

    /// Candidates that are already distinct come back unchanged and in order: the collapse may not
    /// reorder or drop a set the model has nothing to collapse.
    #[test]
    fn distinct_candidates_are_returned_unchanged() {
        let model = Padding { tile: 1 };
        // Rows 0, 1 and 2: with no padding to collapse them, three candidates are three layouts.
        let candidates = [warm(0, 0, 0), warm(1, 0, 0), warm(1, 1, 0)];
        let kept = distinct(&model, &candidates);

        assert_eq!(kept.len(), candidates.len());
        for (kept, stated) in kept.iter().zip(candidates.iter()) {
            assert_eq!(
                (kept.first_frames, kept.last_frames, kept.references),
                (stated.first_frames, stated.last_frames, stated.references)
            );
        }
    }
}
