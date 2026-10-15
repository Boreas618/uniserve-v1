//! MiniMax H3 output rasters: the trained resolution and aspect-ratio buckets.
//!
//! A deployment provisions the cartesian product of its configured
//! resolutions and aspect ratios; the worker prepares every product raster at
//! startup and a request selects one of them. The bucket table is the request
//! rules' ([`uniserve_core::video::plan::bucket`]).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uniserve_core::Canvas;
use uniserve_core::video::plan::{NAMED_ASPECT_RATIOS, bucket};

use crate::profile::omni::resolution::ResolutionName;

/// Why a raster selection is refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VideoRasterError {
    /// A resolution class name the checkpoint was not trained on.
    #[error("unknown video resolution {0:?}; expected 768p or 480p")]
    UnknownResolution(String),
    /// A deployment that names no resolution or no aspect ratio.
    #[error("video resolutions and aspect ratios must not be empty")]
    Empty,
    /// A deployment that names a resolution twice.
    #[error("video resolution {0} is repeated")]
    RepeatedResolution(VideoResolution),
    /// A deployment that names an aspect ratio without a training bucket.
    #[error("video aspect ratio {aspect_ratio} has no trained bucket; expected one of {trained}")]
    UntrainedAspectRatio {
        /// The refused aspect ratio.
        aspect_ratio: &'static str,
        /// The aspect ratios of the training buckets.
        trained: String,
    },
    /// A deployment that names an aspect ratio twice.
    #[error("video aspect ratio {0} is repeated")]
    RepeatedAspectRatio(&'static str),
    /// A request for a resolution the deployment does not provision.
    #[error("resolution must be one of {0}")]
    UnservedResolution(String),
    /// A request for an aspect ratio the deployment does not provision.
    #[error("aspect_ratio must be one of {0}")]
    UnservedAspectRatio(String),
    /// A served resolution and aspect ratio without a training bucket.
    #[error("no trained bucket for {resolution} {aspect_ratio}")]
    NoBucket {
        /// The resolution class.
        resolution: VideoResolution,
        /// The aspect ratio.
        aspect_ratio: &'static str,
    },
}

/// Short-side resolution class of an H3 output raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoResolution {
    /// 768-pixel class, the checkpoint's native training resolution.
    #[serde(rename = "768p")]
    P768,
    /// 480-pixel class.
    #[serde(rename = "480p")]
    P480,
}

impl VideoResolution {
    /// Every resolution class the checkpoint was trained on.
    pub const ALL: [Self; 2] = [Self::P768, Self::P480];

    /// Returns the public name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::P768 => "768p",
            Self::P480 => "480p",
        }
    }

    /// The `target.short_edge`, in pixels, that names this class: the short
    /// edge its buckets start from before an area cap, so a capped bucket
    /// such as 21:9 has a shorter side.
    pub const fn short_edge(self) -> u32 {
        match self {
            Self::P768 => 768,
            Self::P480 => 480,
        }
    }
}

impl fmt::Display for VideoResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for VideoResolution {
    type Err = VideoRasterError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|resolution| resolution.as_str() == value)
            .ok_or_else(|| VideoRasterError::UnknownResolution(value.to_owned()))
    }
}

/// Aspect ratios of the checkpoint's training buckets, landscape first: the
/// names of [`NAMED_ASPECT_RATIOS`], in its order.
pub const VIDEO_ASPECT_RATIOS: [ResolutionName; 6] = [
    ResolutionName::Landscape21x9,
    ResolutionName::Landscape16x9,
    ResolutionName::Landscape4x3,
    ResolutionName::Square,
    ResolutionName::Portrait3x4,
    ResolutionName::Portrait9x16,
];

/// One trained output raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoRaster {
    /// Resolution class.
    pub resolution: VideoResolution,
    /// Aspect ratio.
    pub aspect_ratio: ResolutionName,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl VideoRaster {
    /// Returns the training bucket for a resolution and aspect ratio, or
    /// `None` when the checkpoint has no such bucket.
    pub fn new(resolution: VideoResolution, aspect_ratio: ResolutionName) -> Option<Self> {
        let named = VIDEO_ASPECT_RATIOS
            .iter()
            .position(|&name| name == aspect_ratio)?;
        let canvas = bucket(resolution.short_edge(), NAMED_ASPECT_RATIOS[named])?;
        Some(Self {
            resolution,
            aspect_ratio,
            width: canvas.width,
            height: canvas.height,
        })
    }
}

/// The rasters a deployment serves: every configured resolution crossed with
/// every configured aspect ratio, resolution-major.
///
/// The first resolution and the first aspect ratio are the defaults of a
/// request that names neither, as a Dynamo request without a `size` does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoRasters {
    resolutions: Vec<VideoResolution>,
    aspect_ratios: Vec<ResolutionName>,
}

impl Default for VideoRasters {
    /// 768p landscape and portrait, the rasters every graph-captured
    /// deployment can hold.
    fn default() -> Self {
        Self {
            resolutions: vec![VideoResolution::P768],
            aspect_ratios: vec![ResolutionName::Landscape16x9, ResolutionName::Portrait9x16],
        }
    }
}

impl VideoRasters {
    /// Validates a deployment's raster selection.
    ///
    /// # Errors
    ///
    /// Refuses a selection whose either list is empty, repeats an entry, or
    /// names an aspect ratio with no training bucket.
    pub fn new(
        resolutions: &[VideoResolution],
        aspect_ratios: &[ResolutionName],
    ) -> Result<Self, VideoRasterError> {
        if resolutions.is_empty() || aspect_ratios.is_empty() {
            return Err(VideoRasterError::Empty);
        }
        for (index, resolution) in resolutions.iter().enumerate() {
            if resolutions[..index].contains(resolution) {
                return Err(VideoRasterError::RepeatedResolution(*resolution));
            }
        }
        for (index, aspect_ratio) in aspect_ratios.iter().enumerate() {
            if !VIDEO_ASPECT_RATIOS.contains(aspect_ratio) {
                return Err(VideoRasterError::UntrainedAspectRatio {
                    aspect_ratio: aspect_ratio.as_str(),
                    trained: join(VIDEO_ASPECT_RATIOS.iter().map(|name| name.as_str())),
                });
            }
            if aspect_ratios[..index].contains(aspect_ratio) {
                return Err(VideoRasterError::RepeatedAspectRatio(aspect_ratio.as_str()));
            }
        }
        Ok(Self {
            resolutions: resolutions.to_vec(),
            aspect_ratios: aspect_ratios.to_vec(),
        })
    }

    /// Configured resolutions, the default first.
    pub fn resolutions(&self) -> &[VideoResolution] {
        &self.resolutions
    }

    /// Configured aspect ratios, the default first.
    pub fn aspect_ratios(&self) -> &[ResolutionName] {
        &self.aspect_ratios
    }

    /// Every served raster, resolution-major.
    pub fn rasters(&self) -> impl Iterator<Item = VideoRaster> + '_ {
        self.resolutions.iter().flat_map(|&resolution| {
            self.aspect_ratios
                .iter()
                .filter_map(move |&aspect_ratio| VideoRaster::new(resolution, aspect_ratio))
        })
    }

    /// Selects a served raster; an omitted field takes its configured default.
    ///
    /// # Errors
    ///
    /// Refuses, naming the served values, a resolution or aspect ratio the
    /// deployment does not provision.
    pub fn select(
        &self,
        resolution: Option<VideoResolution>,
        aspect_ratio: Option<ResolutionName>,
    ) -> Result<VideoRaster, VideoRasterError> {
        // `new` admits no empty list, so each has a first entry.
        let resolution = resolution.unwrap_or(self.resolutions[0]);
        let aspect_ratio = aspect_ratio.unwrap_or(self.aspect_ratios[0]);
        if !self.resolutions.contains(&resolution) {
            return Err(VideoRasterError::UnservedResolution(join(
                self.resolutions.iter().map(|value| value.as_str()),
            )));
        }
        if !self.aspect_ratios.contains(&aspect_ratio) {
            return Err(VideoRasterError::UnservedAspectRatio(join(
                self.aspect_ratios.iter().map(|value| value.as_str()),
            )));
        }
        VideoRaster::new(resolution, aspect_ratio).ok_or(VideoRasterError::NoBucket {
            resolution,
            aspect_ratio: aspect_ratio.as_str(),
        })
    }

    /// Returns the served raster with exactly these pixel dimensions.
    pub fn find_size(&self, width: u32, height: u32) -> Option<VideoRaster> {
        self.rasters()
            .find(|raster| raster.width == width && raster.height == height)
    }

    /// The canvases the deployment's workers prepare, resolution-major.
    pub fn canvases(&self) -> Vec<Canvas> {
        self.rasters()
            .map(|raster| Canvas {
                width: raster.width,
                height: raster.height,
            })
            .collect()
    }
}

/// Joins names into a human-readable list.
fn join<S: AsRef<str>>(values: impl Iterator<Item = S>) -> String {
    values
        .map(|value| value.as_ref().to_owned())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each trained aspect name denotes the named ratio at its position, so
    /// a raster is the request rules' bucket of that ratio at its
    /// resolution's short edge; an untrained name has none.
    #[test]
    fn rasters_are_the_named_buckets() {
        for (name, (width, height)) in VIDEO_ASPECT_RATIOS.into_iter().zip(NAMED_ASPECT_RATIOS) {
            assert_eq!(name.as_str(), format!("{width}:{height}"));
            for resolution in VideoResolution::ALL {
                let raster = VideoRaster::new(resolution, name).unwrap();
                let expected = bucket(resolution.short_edge(), (width, height)).unwrap();
                assert_eq!(
                    (raster.width, raster.height),
                    (expected.width, expected.height)
                );
            }
        }
        // The 480p 21:9 bucket is the training table's, not a rescaled 768p
        // canvas.
        let narrow =
            VideoRaster::new(VideoResolution::P480, ResolutionName::Landscape21x9).unwrap();
        assert_eq!((narrow.width, narrow.height), (992, 416));
        assert!(VideoRaster::new(VideoResolution::P768, ResolutionName::Landscape3x2).is_none());
    }

    #[test]
    fn selection_defaults_to_the_first_configured_values() {
        let rasters = VideoRasters::new(
            &[VideoResolution::P480, VideoResolution::P768],
            &[ResolutionName::Portrait9x16, ResolutionName::Square],
        )
        .unwrap();
        let default = rasters.select(None, None).unwrap();
        assert_eq!((default.width, default.height), (480, 832));
        let square = rasters
            .select(Some(VideoResolution::P768), Some(ResolutionName::Square))
            .unwrap();
        assert_eq!((square.width, square.height), (768, 768));
        assert!(
            rasters
                .select(None, Some(ResolutionName::Landscape16x9))
                .is_err()
        );
        let canvas = |width, height| Canvas { width, height };
        assert_eq!(
            rasters.canvases(),
            vec![
                canvas(480, 832),
                canvas(480, 480),
                canvas(768, 1344),
                canvas(768, 768)
            ]
        );
        assert_eq!(rasters.find_size(768, 768), Some(square));
        assert_eq!(rasters.find_size(1344, 768), None);
    }

    #[test]
    fn invalid_selections_are_rejected() {
        let p768 = [VideoResolution::P768];
        assert!(VideoRasters::new(&[], &[ResolutionName::Square]).is_err());
        assert!(VideoRasters::new(&p768, &[]).is_err());
        assert!(VideoRasters::new(&p768, &[ResolutionName::Landscape3x2]).is_err());
        assert!(
            VideoRasters::new(&p768, &[ResolutionName::Square, ResolutionName::Square]).is_err()
        );
        assert!(
            VideoRasters::new(
                &[VideoResolution::P768, VideoResolution::P768],
                &[ResolutionName::Square]
            )
            .is_err()
        );
        assert_eq!(
            "480p".parse::<VideoResolution>().unwrap(),
            VideoResolution::P480
        );
        assert!("720p".parse::<VideoResolution>().is_err());
    }
}
