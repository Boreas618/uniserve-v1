//! A loaded model's image facts and the host-side image interface.
//!
//! Read once at startup from the runners that own each fact: the model's
//! image processing (which towers it declares, its pixel dtype and feature
//! injection) from `EncoderRunner`, its image denoiser's latent layout from
//! `DenoiserRunner` and its image decoder's pixel range from
//! `DecoderRunner`. Rust derives every per-call geometry from these facts:
//! latent rows, sequence lengths with framing, and denoising positions.
//!
//! Two host functions complete the policy: `EncoderRunner.patch_grid` (the
//! vision tower's patch grid of an image, which places its features),
//! asked in jobs of the Python thread at input admission and kept, and
//! `DenoiserRunner.guidance_prefix` (a guidance prefix's tokens, by the
//! model's prompt framing and tokenizer).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ffi::{DType, Function, Handle, Interpreter, integers};
use crate::lane::{Completion, Job};
use crate::request::diffusion::{BranchSource, Prefix};
use crate::runner::encoder::ImagePreparation;
use crate::runner::methods::{Argument, MethodCall};
use crate::runner::vision::{FeatureInjection, FeatureLayout, PositionLayout};
use crate::{Error, Result, lock};

const ENCODER: &str = "uniserve.runner.encoder.EncoderRunner";
const DENOISER: &str = "uniserve.runner.denoiser.DenoiserRunner";
const DECODER: &str = "uniserve.runner.decoder.DecoderRunner";

/// Language positions a generated image spans in its conversation: its
/// closing framing token sits this far past the opening one, and the
/// request's logical position advances by it after the image.
pub(crate) const ROPE_ADVANCE: u64 = 2;

/// A model's image processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageProcessing {
    /// The understanding tower (vision encoding) is declared.
    pub(crate) vit: bool,
    /// The latent tower (latent encoding) is declared.
    pub(crate) vae: bool,
    /// The understanding tower packs patch rows (its features' positions
    /// follow its patch grid).
    pub(crate) patch_tower: bool,
    pub(crate) pixel_dtype: DType,
    pub(crate) injection: Option<FeatureInjection>,
}

/// An image denoiser's latent layout (`ImageDenoiser`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageDenoiserLayout {
    pub patch_size: u32,
    pub latent_channels: u32,
    /// Output pixels per latent patch on each spatial axis.
    pub downsample: u32,
    /// Framing tokens around a generated image: zero, or one leading and
    /// one trailing token.
    pub framing_tokens: u32,
    /// What the image-unconditional guidance branch conditions on.
    pub image_unconditional: BranchSource,
    /// The longest image sequence, in latent patch rows (framing excluded).
    pub max_sequence_tokens: u32,
    pub prediction_dtype: DType,
}

impl ImageDenoiserLayout {
    /// Latent patch rows of a `height` x `width` image
    /// (`ImageDenoiser.latent_shape`).
    pub(crate) fn latent_rows(&self, height: u32, width: u32) -> Result<u32> {
        let rows = (height / self.downsample) * (width / self.downsample);
        if rows == 0 {
            return Err(Error::Invalid(
                "image dimensions must contain at least one latent patch".into(),
            ));
        }
        Ok(rows)
    }

    /// Values per latent patch row.
    pub(crate) fn width(&self) -> u32 {
        self.patch_size * self.patch_size * self.latent_channels
    }

    /// Tokens of an image's sequence: its latent rows and framing.
    pub(crate) fn sequence_length(&self, height: u32, width: u32) -> Result<u32> {
        Ok(self.latent_rows(height, width)? + self.framing_tokens)
    }

    /// The image sequence's `[3, tokens]` positions in model sequence order
    /// (rows: temporal, row, column). The temporal axis is `temporal`
    /// everywhere, framing tokens included; the spatial axes hold each
    /// latent patch's raster coordinates, and framing tokens zero.
    pub(crate) fn positions(
        &self,
        height: u32,
        width: u32,
        temporal: u64,
    ) -> Result<[Vec<i64>; 3]> {
        let count = self.sequence_length(height, width)? as usize;
        let columns = (width / self.downsample) as usize;
        let first = if self.framing_tokens == 0 { 0 } else { 1 };
        let mut rows = vec![0; count];
        let mut cols = vec![0; count];
        for index in 0..self.latent_rows(height, width)? as usize {
            rows[first + index] = (index / columns) as i64;
            cols[first + index] = (index % columns) as i64;
        }
        Ok([vec![temporal as i64; count], rows, cols])
    }

    /// The positions of a latent image inserted as conditioning (a latent
    /// ingest or feedback prefill): the image's positions one past the
    /// conditioning position, with the opening framing token at it and the
    /// closing one `ROPE_ADVANCE` past it.
    pub(crate) fn conditioning_positions(
        &self,
        height: u32,
        width: u32,
        conditioning_position: u64,
    ) -> Result<[Vec<i64>; 3]> {
        if self.framing_tokens != 2 {
            return Err(Error::Invalid(
                "latent feature export requires framed image conditioning".into(),
            ));
        }
        let mut positions = self.positions(height, width, conditioning_position + 1)?;
        let last = positions[0].len() - 1;
        positions[0][0] = conditioning_position as i64;
        positions[0][last] = (conditioning_position + ROPE_ADVANCE) as i64;
        Ok(positions)
    }
}

/// The image facts of a loaded model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ImageModel {
    pub(crate) processing: Option<ImageProcessing>,
    pub(crate) denoiser: Option<ImageDenoiserLayout>,
    /// The image decoder's `(low, high)` pixel range.
    pub(crate) decoder_range: Option<(f32, f32)>,
}

/// The dtype a fact method names.
fn dtype(function: &Function, name: tvm_ffi::String) -> Result<DType> {
    DType::from_name(name.as_str()).ok_or_else(|| {
        Error::Interface(format!(
            "{} reports the unknown dtype {:?}",
            function.name(),
            name.as_str()
        ))
    })
}

impl ImageModel {
    /// Read a loaded model's image facts from the runners that own them:
    /// the image policy from `EncoderRunner`, the latent layout from
    /// `DenoiserRunner` and the pixel range from `DecoderRunner`.
    pub(crate) fn describe(loaded: &Handle) -> Result<Self> {
        let encoder = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
        let denoiser = |name: &str| Interpreter::import(&format!("{DENOISER}.{name}"));
        let model = [loaded.view()];

        let processing = if encoder("processes_images")?.call_as::<bool>(&model)? {
            let towers: tvm_ffi::Array<tvm_ffi::String> =
                encoder("image_towers")?.call_as(&model)?;
            let declares = |tower: &str| towers.iter().any(|name| name.as_str() == tower);
            let layouts: tvm_ffi::Array<tvm_ffi::String> =
                encoder("feature_injection")?.call_as(&model)?;
            let layouts: Vec<String> = layouts
                .iter()
                .map(|name| name.as_str().to_owned())
                .collect();
            let injection = match &layouts[..] {
                [] => None,
                [layout, positions] => {
                    let tokens =
                        integers(encoder("injection_tokens")?.call_values(&[loaded.to_any()])?)
                            .map_err(|error| Error::numerical("injection_tokens", &error))?;
                    let [start, end] = tokens[..] else {
                        return Err(Error::Interface(
                            "feature injection names a start and an end token".into(),
                        ));
                    };
                    // A negative id stands for a token that is not placed.
                    let token = |id: i64| u32::try_from(id).ok();
                    Some(FeatureInjection {
                        layout: match layout.as_str() {
                            "direct" => FeatureLayout::Direct,
                            "framed" => FeatureLayout::Framed,
                            other => {
                                return Err(Error::Interface(format!(
                                    "unknown feature layout {other:?}"
                                )));
                            }
                        },
                        positions: match positions.as_str() {
                            "temporal" => PositionLayout::Temporal,
                            "temporal_spatial" => PositionLayout::TemporalSpatial,
                            "sequential" => PositionLayout::Sequential,
                            other => {
                                return Err(Error::Interface(format!(
                                    "unknown feature position layout {other:?}"
                                )));
                            }
                        },
                        start_token: token(start),
                        end_token: token(end),
                    })
                }
                _ => {
                    return Err(Error::Interface(
                        "feature injection names a layout and a position layout".into(),
                    ));
                }
            };
            let pixels = encoder("pixel_dtype")?;
            Some(ImageProcessing {
                vit: declares("vit"),
                vae: declares("vae"),
                patch_tower: encoder("packs_patches")?.call_as::<bool>(&model)?,
                pixel_dtype: dtype(&pixels, pixels.call_as(&model)?)?,
                injection,
            })
        } else {
            None
        };

        let layout = integers(denoiser("image_layout")?.call_values(&[loaded.to_any()])?)
            .map_err(|error| Error::numerical("image_layout", &error))?;
        let denoiser = match layout[..] {
            [] => None,
            [
                patch_size,
                latent_channels,
                downsample,
                framing_tokens,
                max_sequence_tokens,
            ] => {
                let extent = |value: i64| -> Result<u32> {
                    u32::try_from(value).map_err(|_| {
                        Error::Interface("an image denoiser's layout is out of range".into())
                    })
                };
                let unconditional: tvm_ffi::String =
                    denoiser("image_unconditional")?.call_as(&model)?;
                let prediction = denoiser("prediction_dtype")?;
                let layout = ImageDenoiserLayout {
                    patch_size: extent(patch_size)?,
                    latent_channels: extent(latent_channels)?,
                    downsample: extent(downsample)?,
                    framing_tokens: extent(framing_tokens)?,
                    image_unconditional: BranchSource::parse(unconditional.as_str())?,
                    max_sequence_tokens: extent(max_sequence_tokens)?,
                    prediction_dtype: dtype(&prediction, prediction.call_as(&model)?)?,
                };
                if layout.downsample == 0 || !matches!(layout.framing_tokens, 0 | 2) {
                    return Err(Error::Interface(
                        "an image denoiser frames an image with zero or two tokens".into(),
                    ));
                }
                Some(layout)
            }
            _ => {
                return Err(Error::Interface(
                    "an image denoiser's layout names five sizes".into(),
                ));
            }
        };

        let range: tvm_ffi::Array<f64> =
            Interpreter::import(&format!("{DECODER}.value_range"))?.call_as(&model)?;
        let range: Vec<f64> = range.iter().collect();
        let decoder_range = match range[..] {
            [] => None,
            [low, high] => Some((low as f32, high as f32)),
            _ => {
                return Err(Error::Interface(
                    "an image decoder range names two bounds".into(),
                ));
            }
        };
        Ok(Self {
            processing,
            denoiser,
            decoder_range,
        })
    }

    /// Whether text rows carry three rotary axes: a model with an image
    /// denoiser (whose sequences place latent images spatially) or one
    /// placing vision features spatially. Others read one temporal axis.
    pub(crate) fn spatial_positions(&self) -> bool {
        self.denoiser.is_some()
            || self
                .processing
                .and_then(|processing| processing.injection)
                .is_some_and(|injection| injection.positions == PositionLayout::TemporalSpatial)
    }
}

/// The host functions of the image policy.
#[derive(Clone)]
pub(crate) struct ImagePolicy {
    loaded: Handle,
    preparation: ImagePreparation,
    patch_grid: Function,
    guidance_prefix: Function,
    /// The patch grid of every image size asked so far
    /// ([`Self::ask_grid`]): facts of the model, asked once and shared by
    /// the policy's clones.
    grids: Arc<Mutex<HashMap<GridKey, [u32; 2]>>>,
}

/// An image's height and width in pixels and its request's input image
/// count, zero for a generated image: what its patch grid depends on.
pub(crate) type GridKey = (u32, u32, u32);

impl ImagePolicy {
    pub(crate) fn new(loaded: Handle) -> Result<Self> {
        Ok(Self {
            preparation: ImagePreparation::new(loaded.clone())?,
            loaded,
            patch_grid: Interpreter::import(&format!("{ENCODER}.patch_grid"))?,
            guidance_prefix: Interpreter::import(&format!("{DENOISER}.guidance_prefix"))?,
            grids: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// The policy's image preparation (called from host lanes, and for
    /// the facts of resident images).
    pub(crate) fn preparation(&self) -> &ImagePreparation {
        &self.preparation
    }

    /// The vision tower's patch grid of an image, when it was asked
    /// ([`Self::ask_grid`]).
    pub(crate) fn grid(&self, key: GridKey) -> Option<[u32; 2]> {
        lock(&self.grids).get(&key).copied()
    }

    /// Record in `job`, a job of the Python thread, the call asking the
    /// patch grid of `key`: the job's completion carries `[rows, columns]`,
    /// which [`Self::answer_grid`] keeps.
    pub(crate) fn ask_grid(&self, key: GridKey, job: &Job) {
        let (height, width, input_images) = key;
        job.call_returning(MethodCall::new(
            &self.patch_grid,
            vec![
                Argument::Handle(self.loaded.clone()),
                Argument::Integer(i64::from(height)),
                Argument::Integer(i64::from(width)),
                Argument::Integer(i64::from(input_images)),
            ],
        ));
    }

    /// Keep the grid of `key` the next value of `completion` carries
    /// ([`Self::ask_grid`]).
    pub(crate) fn answer_grid(&self, key: GridKey, completion: &mut Completion) -> Result<()> {
        let grid = match completion.returned()?[..] {
            [rows, columns] if rows > 0 && columns > 0 => [rows as u32, columns as u32],
            _ => {
                return Err(Error::Interface(format!(
                    "{} reports a grid that is not two positive extents",
                    self.patch_grid.name()
                )));
            }
        };
        lock(&self.grids).insert(key, grid);
        Ok(())
    }

    /// Record the call asking the prefix a guidance branch of `source`
    /// conditions on for the image prompt `image_prompt`: the job's
    /// completion carries `[reuses, *tokens]`, which [`Self::prefix`] reads.
    ///
    /// The negative prompt the model is given is empty: the negative
    /// source's prefix is the admitted negative tokens
    /// ([`BranchSource::admitted`]), and no other source reads it.
    pub(crate) fn ask_prefix(&self, source: BranchSource, image_prompt: &str, job: &Job) {
        job.call_returning(MethodCall::new(
            &self.guidance_prefix,
            vec![
                Argument::Handle(self.loaded.clone()),
                Argument::Text(source.name().to_owned()),
                Argument::Text(image_prompt.to_owned()),
                Argument::Text(String::new()),
            ],
        ));
    }

    /// The prefix the model frames for a guidance branch of `source` of a
    /// request without an image prompt or a negative prompt, asked now on
    /// the Python thread (startup work).
    pub(crate) fn prefix_of(&self, source: BranchSource) -> Result<Prefix> {
        let text = |value: &str| tvm_ffi::Any::from(tvm_ffi::String::from(value));
        let values = self.guidance_prefix.call_values(&[
            self.loaded.to_any(),
            text(source.name()),
            text(""),
            text(""),
        ])?;
        let values = integers(values)
            .map_err(|error| Error::numerical(self.guidance_prefix.name(), &error))?;
        self.prefix(&values)
    }

    /// The prefix a prefix call returned as `[reuses, *tokens]`.
    pub(crate) fn prefix(&self, values: &[i64]) -> Result<Prefix> {
        let (reuses, tokens) = values.split_first().ok_or_else(|| {
            Error::Interface(format!(
                "{} returns [reuses, *tokens]",
                self.guidance_prefix.name()
            ))
        })?;
        let tokens = tokens
            .iter()
            .map(|&token| {
                u32::try_from(token).map_err(|_| {
                    Error::Interface("guidance prefix token is outside the token range".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Prefix {
            tokens: Arc::from(tokens),
            reuses_conditioning: *reuses != 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(framing: u32) -> ImageDenoiserLayout {
        ImageDenoiserLayout {
            patch_size: 2,
            latent_channels: 16,
            downsample: 16,
            framing_tokens: framing,
            image_unconditional: BranchSource::Conditioning,
            max_sequence_tokens: 4096,
            prediction_dtype: DType::BF16,
        }
    }

    #[test]
    fn image_positions_raster_latent_patches_inside_their_framing() -> Result<()> {
        // A 32x48 image of 16-pixel latent patches: a 2x3 grid.
        let framed = layout(2);
        assert_eq!(framed.latent_rows(32, 48)?, 6);
        assert_eq!(framed.sequence_length(32, 48)?, 8);
        assert_eq!(framed.width(), 64);
        assert_eq!(
            framed.positions(32, 48, 7)?,
            [
                vec![7; 8],
                vec![0, 0, 0, 0, 1, 1, 1, 0],
                vec![0, 0, 1, 2, 0, 1, 2, 0],
            ]
        );
        let unframed = layout(0);
        assert_eq!(
            unframed.positions(32, 48, 0)?,
            [vec![0; 6], vec![0, 0, 0, 1, 1, 1], vec![0, 1, 2, 0, 1, 2]]
        );
        assert!(unframed.latent_rows(8, 48).is_err());
        Ok(())
    }

    #[test]
    fn latent_conditioning_frames_the_image_around_its_rope_span() -> Result<()> {
        let positions = layout(2).conditioning_positions(32, 48, 10)?;
        assert_eq!(positions[0], vec![10, 11, 11, 11, 11, 11, 11, 12]);
        assert!(layout(0).conditioning_positions(32, 48, 10).is_err());
        Ok(())
    }
}
