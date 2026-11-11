//! The canvas rule of a video family: how a display aspect resolves to the raster it generates, and
//! how an image is fitted onto a raster.
//!
//! A family states the rule's values once ([`CanvasRule`]); the request processing resolves a
//! requested or a keyframe's aspect with it, and the family names the canvases it serves by it.

use proto::video::Canvas;
use serde::{Deserialize, Serialize};

/// How a display aspect resolves to a canvas.
///
/// The short edge starts at [`Self::short_edge`] pixels, an area above [`Self::max_pixels`] scales
/// both sides down, and each side rounds half to even to a multiple of [`Self::multiple`], at least
/// one multiple, so the final area may end up slightly above the cap. Only aspects from
/// [`Self::narrowest`] to [`Self::widest`] resolve. This is the diffusers `adapt_shape_v1` rule;
/// the released MiniMax-H3 checkpoint states a 768-pixel short edge, `768 * 1344` pixels, a
/// 32-pixel multiple and 1:4 to 4:1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanvasRule {
    /// The short edge a canvas starts at, in pixels: the resolution class the rule generates.
    pub short_edge: u32,
    /// The area above which both sides scale down, in pixels.
    pub max_pixels: u32,
    /// What every side is a multiple of, in pixels.
    pub multiple: u32,
    /// The narrowest aspect that resolves, as `(width, height)` of the ratio.
    pub narrowest: (u32, u32),
    /// The widest aspect that resolves, as `(width, height)` of the ratio.
    pub widest: (u32, u32),
}

/// An aspect the canvas rule cannot resolve ([`CanvasRule::canvas`]).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum CanvasError {
    /// A side is zero, negative or not a number.
    #[error("aspect must be positive, got {width}:{height}")]
    NonPositive {
        /// The aspect's width.
        width: f64,
        /// The aspect's height.
        height: f64,
    },
    /// The width-to-height ratio lies outside the rule's range.
    #[error(
        "aspect {width}:{height} lies outside {}:{} to {}:{}",
        narrowest.0, narrowest.1, widest.0, widest.1
    )]
    OutOfRange {
        /// The aspect's width.
        width: f64,
        /// The aspect's height.
        height: f64,
        /// The narrowest aspect the rule resolves.
        narrowest: (u32, u32),
        /// The widest aspect the rule resolves.
        widest: (u32, u32),
    },
}

impl CanvasRule {
    /// The narrowest width-to-height ratio that resolves.
    pub fn min_ratio(&self) -> f64 {
        f64::from(self.narrowest.0) / f64::from(self.narrowest.1)
    }

    /// The widest width-to-height ratio that resolves.
    pub fn max_ratio(&self) -> f64 {
        f64::from(self.widest.0) / f64::from(self.widest.1)
    }

    /// Whether a `width` x `height` raster lies within the rule's aspects, compared exactly in
    /// integers.
    pub fn admits(&self, width: u32, height: u32) -> bool {
        let (width, height) = (u64::from(width), u64::from(height));
        width * u64::from(self.widest.1) <= u64::from(self.widest.0) * height
            && width * u64::from(self.narrowest.1) >= u64::from(self.narrowest.0) * height
    }

    /// Whether the rule's values are positive, its range holds a square and its cap holds one.
    pub fn is_valid(&self) -> bool {
        self.short_edge > 0
            && self.multiple > 0
            && u64::from(self.max_pixels) >= u64::from(self.short_edge).pow(2)
            && self.narrowest.0 > 0
            && self.narrowest.1 > 0
            && self.widest.0 > 0
            && self.widest.1 > 0
            && self.admits(1, 1)
    }

    /// Resolves a display aspect into a canvas.
    ///
    /// Only the ratio of the arguments matters.
    ///
    /// # Errors
    ///
    /// Returns [`CanvasError`] when the ratio is not finite and positive or lies outside the rule's
    /// range.
    pub fn canvas(&self, aspect_width: f64, aspect_height: f64) -> Result<Canvas, CanvasError> {
        if !(aspect_width > 0.0 && aspect_height > 0.0) {
            return Err(CanvasError::NonPositive {
                width: aspect_width,
                height: aspect_height,
            });
        }
        let ratio = aspect_width / aspect_height;
        if !(self.min_ratio()..=self.max_ratio()).contains(&ratio) {
            return Err(CanvasError::OutOfRange {
                width: aspect_width,
                height: aspect_height,
                narrowest: self.narrowest,
                widest: self.widest,
            });
        }
        let short_edge = f64::from(self.short_edge);
        let (mut width, mut height) = if ratio >= 1.0 {
            (short_edge * ratio, short_edge)
        } else {
            (short_edge, short_edge / ratio)
        };
        let area = width * height;
        let max_pixels = f64::from(self.max_pixels);
        if area > max_pixels {
            // `powf(0.5)` rather than `sqrt` reproduces the reference's `** 0.5`.
            let scale = (max_pixels / area).powf(0.5);
            width *= scale;
            height *= scale;
        }
        Ok(Canvas {
            width: self.nearest_multiple(width),
            height: self.nearest_multiple(height),
        })
    }

    /// Rounds a side half to even to the nearest multiple of [`Self::multiple`], at least one
    /// multiple.
    pub fn nearest_multiple(&self, value: f64) -> u32 {
        let multiple = f64::from(self.multiple);
        // Sides stay far below `u32::MAX`, so the conversion is exact.
        ((value / multiple).round_ties_even() * multiple).max(multiple) as u32
    }

    /// Every canvas the rule resolves a display aspect to, each once, from the narrowest aspect to
    /// the widest.
    ///
    /// [`Self::canvas`] is piecewise constant in the ratio: a side changes only where its unrounded
    /// length crosses an odd multiple of half [`Self::multiple`]. Below the area cap the long side is
    /// the short edge times the ratio (its reciprocal for a portrait aspect); above it the long side
    /// is `sqrt(cap * ratio)` and the short side `sqrt(cap / ratio)`. Those crossings, the ends of the
    /// range and the ratio the cap starts at cut the range into intervals of one canvas each, so the
    /// canvases at the cuts and at the midpoints between them are all of them.
    pub fn canvases(&self) -> Vec<Canvas> {
        let short_edge = f64::from(self.short_edge);
        let max_pixels = f64::from(self.max_pixels);
        let multiple = f64::from(self.multiple);
        let (min_ratio, max_ratio) = (self.min_ratio(), self.max_ratio());
        let cap = max_pixels / (short_edge * short_edge);
        let mut cuts = vec![min_ratio, max_ratio, 1.0, cap, 1.0 / cap];
        // No side of the range is longer than the long side at the widest or the narrowest aspect
        // under the cap.
        let longest = (max_pixels * max_ratio.max(1.0 / min_ratio)).sqrt();
        let mut crossing = multiple / 2.0;
        while crossing <= longest + multiple {
            for ratio in [
                crossing / short_edge,
                crossing * crossing / max_pixels,
                max_pixels / (crossing * crossing),
            ] {
                cuts.extend([ratio, 1.0 / ratio]);
            }
            crossing += multiple;
        }
        cuts.retain(|ratio| (min_ratio..=max_ratio).contains(ratio));
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();

        let midpoints: Vec<f64> = cuts
            .windows(2)
            .map(|pair| f64::midpoint(pair[0], pair[1]))
            .collect();
        let mut ratios = cuts;
        ratios.extend(midpoints);
        ratios.sort_by(f64::total_cmp);
        let mut canvases = Vec::new();
        // Every ratio lies within the range, which the rule accepts.
        for size in ratios
            .into_iter()
            .filter_map(|ratio| self.canvas(ratio, 1.0).ok())
        {
            if !canvases.contains(&size) {
                canvases.push(size);
            }
        }
        canvases
    }
}

/// An aspect-preserving resize followed by a centred crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CoverCrop {
    /// Width the image is resized to before cropping.
    pub width: u32,
    /// Height the image is resized to before cropping.
    pub height: u32,
    /// Left edge of the crop in the resized image.
    pub left: u32,
    /// Top edge of the crop in the resized image.
    pub top: u32,
}

/// The resize and centred crop of a `width` x `height` image that cover `canvas`.
///
/// The scale is the larger of the two side ratios; each resized side rounds half to even and never
/// falls below the canvas, and the crop is centred with floor division.
pub fn cover_crop(width: u32, height: u32, canvas: Canvas) -> CoverCrop {
    let scale = (f64::from(canvas.width) / f64::from(width))
        .max(f64::from(canvas.height) / f64::from(height));
    let resized_width = canvas
        .width
        .max((f64::from(width) * scale).round_ties_even() as u32);
    let resized_height = canvas
        .height
        .max((f64::from(height) * scale).round_ties_even() as u32);
    CoverCrop {
        width: resized_width,
        height: resized_height,
        left: (resized_width - canvas.width) / 2,
        top: (resized_height - canvas.height) / 2,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// The released MiniMax-H3 rule.
    const RULE: CanvasRule = CanvasRule {
        short_edge: 768,
        max_pixels: 768 * 1344,
        multiple: 32,
        narrowest: (1, 4),
        widest: (4, 1),
    };

    #[test]
    fn an_aspect_resolves_to_its_canvas_on_the_grid() {
        let canvas = |width, height| RULE.canvas(width, height).unwrap();
        assert_eq!(
            canvas(16.0, 9.0),
            Canvas {
                width: 1344,
                height: 768
            }
        );
        assert_eq!(
            canvas(21.0, 9.0),
            Canvas {
                width: 1536,
                height: 672
            }
        );
        assert_eq!(
            canvas(4.0, 3.0),
            Canvas {
                width: 1024,
                height: 768
            }
        );
        assert_eq!(
            canvas(1.0, 1.0),
            Canvas {
                width: 768,
                height: 768
            }
        );
        let error = RULE.canvas(5.0, 1.0).unwrap_err();
        assert_eq!(error.to_string(), "aspect 5:1 lies outside 1:4 to 4:1");
        assert!(RULE.canvas(0.0, 1.0).is_err());
    }

    /// The listed canvases are exactly those a dense sweep of the range resolves to, each listed
    /// once.
    #[test]
    fn the_listed_canvases_are_every_canvas_of_the_range() {
        let listed = RULE.canvases();
        let unique: HashSet<Canvas> = listed.iter().copied().collect();
        assert_eq!(unique.len(), listed.len());

        let steps = 1_000_000;
        let swept: HashSet<Canvas> = (0..=steps)
            .map(|step| {
                let ratio = RULE.min_ratio()
                    + (RULE.max_ratio() - RULE.min_ratio()) * f64::from(step) / f64::from(steps);
                RULE.canvas(ratio, 1.0).unwrap()
            })
            .collect();
        assert_eq!(swept, unique);
    }

    #[test]
    fn the_range_is_compared_exactly() {
        assert!(RULE.admits(400, 100) && RULE.admits(100, 400));
        assert!(!RULE.admits(401, 100) && !RULE.admits(100, 401));
    }

    #[test]
    fn a_cover_crop_covers_the_canvas_and_centres_it() {
        let crop = cover_crop(
            1000,
            1000,
            Canvas {
                width: 1344,
                height: 768,
            },
        );
        assert_eq!(
            crop,
            CoverCrop {
                width: 1344,
                height: 1344,
                left: 0,
                top: 288
            }
        );
    }
}
