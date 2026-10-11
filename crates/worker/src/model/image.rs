//! A loaded model's image facts and the host-side image interface.
//!
//! `DenoiserRunner.describe` reports, once at startup, what the image
//! runtime plans with: the model's image processing (which towers it declares, its pixel
//! dtype and feature injection), its image denoiser's latent layout and its
//! image decoder's pixel range. Rust derives every per-call geometry from
//! these facts: latent rows, sequence lengths with framing, and denoising
//! positions.
//!
//! Two host functions complete the policy: `EncoderRunner.patch_grid` (the
//! vision tower's patch grid of an image, which places its features) and
//! `DenoiserRunner.guidance_prefix` (a guidance prefix's tokens, by the
//! model's prompt framing and tokenizer).
//!
//! FIXME: these facts and host functions belong with the runners they
//! describe. The file stays while `executor/forward/token/` names
//! `model::image::ROPE_ADVANCE`.

use std::sync::Arc;

use tvm_ffi::Any;

use crate::ffi::{DType, Function, Handle, Record, integers};
use crate::model::interface::Interface;
use crate::request::diffusion::{BranchSource, Prefix};
use crate::runner::vision::{FeatureInjection, FeatureLayout, PositionLayout};
use crate::{Error, Result};

const ENCODER: &str = "uniserve_models.runner.encoder.EncoderRunner";
const DENOISER: &str = "uniserve_models.runner.denoiser.DenoiserRunner";

/// Language positions a generated image spans in its conversation: its
/// closing framing token sits this far past the opening one, and the
/// request's logical position advances by it after the image.
pub const ROPE_ADVANCE: u64 = 2;

/// A model's image processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageProcessing {
    /// The understanding tower (vision encoding) is declared.
    pub vit: bool,
    /// The latent tower (latent encoding) is declared.
    pub vae: bool,
    /// The understanding tower packs patch rows (its features' positions
    /// follow its patch grid).
    pub patch_tower: bool,
    pub pixel_dtype: DType,
    pub injection: Option<FeatureInjection>,
}

/// An image denoiser's latent layout (`ImageDenoiser`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDenoiserLayout {
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
    pub fn latent_rows(&self, height: u32, width: u32) -> Result<u32> {
        let rows = (height / self.downsample) * (width / self.downsample);
        if rows == 0 {
            return Err(Error::Invalid(
                "image dimensions must contain at least one latent patch".into(),
            ));
        }
        Ok(rows)
    }

    /// Values per latent patch row.
    pub fn width(&self) -> u32 {
        self.patch_size * self.patch_size * self.latent_channels
    }

    /// Tokens of an image's sequence: its latent rows and framing.
    pub fn sequence_length(&self, height: u32, width: u32) -> Result<u32> {
        Ok(self.latent_rows(height, width)? + self.framing_tokens)
    }

    /// The image sequence's `[3, tokens]` positions in model sequence order
    /// (rows: temporal, row, column). The temporal axis is `temporal`
    /// everywhere, framing tokens included; the spatial axes hold each
    /// latent patch's raster coordinates, and framing tokens zero.
    pub fn positions(&self, height: u32, width: u32, temporal: u64) -> Result<[Vec<i64>; 3]> {
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
    /// closing one [`ROPE_ADVANCE`] past it.
    pub fn conditioning_positions(
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
pub struct ImageModel {
    pub processing: Option<ImageProcessing>,
    pub denoiser: Option<ImageDenoiserLayout>,
    /// The image decoder's `(low, high)` pixel range.
    pub decoder_range: Option<(f32, f32)>,
}

fn dtype(record: &Record, key: &str) -> Result<DType> {
    let name = record.string(key)?;
    DType::from_name(&name).ok_or_else(|| {
        Error::Interface(format!(
            "{} reports the unknown dtype {name:?}",
            record.owner()
        ))
    })
}

fn extent(record: &Record, key: &str) -> Result<u32> {
    u32::try_from(record.count(key)?)
        .map_err(|_| Error::Interface(format!("{} {key} overflows", record.owner())))
}

fn token(record: &Record, key: &str) -> Result<Option<u32>> {
    record
        .optional_count(key)?
        .map(|value| {
            u32::try_from(value)
                .map_err(|_| Error::Interface(format!("{} {key} overflows", record.owner())))
        })
        .transpose()
}

impl ImageModel {
    /// Describe a loaded model's image facts.
    pub fn describe(interface: &Interface, loaded: &Handle) -> Result<Self> {
        // FIXME: the image facts still cross as one record.
        let function = interface.import(&format!("{DENOISER}.describe"))?;
        Self::from_record(&Record::new(
            function.name(),
            function.call_values(&[loaded.to_any()])?,
        )?)
    }

    fn from_record(record: &Record) -> Result<Self> {
        let processing = record
            .optional_record("processor")?
            .map(|processor| -> Result<ImageProcessing> {
                let injection = processor
                    .optional_record("injection")?
                    .map(|injection| -> Result<FeatureInjection> {
                        Ok(FeatureInjection {
                            layout: match injection.string("layout")?.as_str() {
                                "direct" => FeatureLayout::Direct,
                                "framed" => FeatureLayout::Framed,
                                other => {
                                    return Err(Error::Interface(format!(
                                        "unknown feature layout {other:?}"
                                    )));
                                }
                            },
                            positions: match injection.string("positions")?.as_str() {
                                "temporal" => PositionLayout::Temporal,
                                "temporal_spatial" => PositionLayout::TemporalSpatial,
                                "sequential" => PositionLayout::Sequential,
                                other => {
                                    return Err(Error::Interface(format!(
                                        "unknown feature position layout {other:?}"
                                    )));
                                }
                            },
                            start_token: token(&injection, "start_token")?,
                            end_token: token(&injection, "end_token")?,
                        })
                    })
                    .transpose()?;
                Ok(ImageProcessing {
                    vit: processor.flag("vit")?,
                    vae: processor.flag("vae")?,
                    patch_tower: processor.flag("patch_tower")?,
                    pixel_dtype: dtype(&processor, "pixel_dtype")?,
                    injection,
                })
            })
            .transpose()?;
        let denoiser = record
            .optional_record("denoiser")?
            .map(|denoiser| -> Result<ImageDenoiserLayout> {
                let layout = ImageDenoiserLayout {
                    patch_size: extent(&denoiser, "patch_size")?,
                    latent_channels: extent(&denoiser, "latent_channels")?,
                    downsample: extent(&denoiser, "downsample")?,
                    framing_tokens: extent(&denoiser, "framing_tokens")?,
                    image_unconditional: BranchSource::parse(
                        &denoiser.string("image_unconditional")?,
                    )?,
                    max_sequence_tokens: extent(&denoiser, "max_sequence_tokens")?,
                    prediction_dtype: dtype(&denoiser, "prediction_dtype")?,
                };
                if layout.downsample == 0 || !matches!(layout.framing_tokens, 0 | 2) {
                    return Err(Error::Interface(
                        "an image denoiser frames an image with zero or two tokens".into(),
                    ));
                }
                Ok(layout)
            })
            .transpose()?;
        let decoder_range = record
            .optional_value("decoder_range")?
            .map(|range| -> Result<(f32, f32)> {
                let values: tvm_ffi::Array<Any> = tvm_ffi::Array::try_from(range)
                    .map_err(|error| Error::numerical(record.owner(), &error))?;
                let bound = |index: usize| -> Result<f32> {
                    f64::try_from(
                        values
                            .get(index)
                            .map_err(|error| Error::numerical(record.owner(), &error))?,
                    )
                    .map(|value| value as f32)
                    .map_err(|error| Error::numerical(record.owner(), &error))
                };
                if values.len() != 2 {
                    return Err(Error::Interface(
                        "an image decoder range names two bounds".into(),
                    ));
                }
                Ok((bound(0)?, bound(1)?))
            })
            .transpose()?;
        Ok(Self {
            processing,
            denoiser,
            decoder_range,
        })
    }
}

/// The host functions of the image policy.
#[derive(Clone)]
pub struct ImagePolicy {
    loaded: Handle,
    prepare_input: Function,
    patch_grid: Function,
    guidance_prefix: Function,
}

impl ImagePolicy {
    pub fn new(interface: &Interface, loaded: Handle) -> Result<Self> {
        Ok(Self {
            loaded,
            prepare_input: interface.import(&format!("{ENCODER}.prepare_image"))?,
            patch_grid: interface.import(&format!("{ENCODER}.patch_grid"))?,
            guidance_prefix: interface.import(&format!("{DENOISER}.guidance_prefix"))?,
        })
    }

    pub fn loaded(&self) -> &Handle {
        &self.loaded
    }

    /// `EncoderRunner.prepare_image` (called from host lanes).
    pub fn prepare_input(&self) -> &Function {
        &self.prepare_input
    }

    /// The vision tower's patch grid of an image; `input_images` is the
    /// request's input image count, `None` for a generated image.
    pub fn patch_grid(
        &self,
        height: u32,
        width: u32,
        input_images: Option<u32>,
    ) -> Result<[u32; 2]> {
        let value = self.patch_grid.call_values(&[
            self.loaded.to_any(),
            Any::from(i64::from(height)),
            Any::from(i64::from(width)),
            Any::from(i64::from(input_images.unwrap_or(0))),
        ])?;
        let grid =
            integers(value).map_err(|error| Error::numerical(self.patch_grid.name(), &error))?;
        match grid[..] {
            [rows, columns] if rows > 0 && columns > 0 => Ok([rows as u32, columns as u32]),
            _ => Err(Error::Interface(format!(
                "{} reports a grid that is not two positive extents",
                self.patch_grid.name()
            ))),
        }
    }

    /// The prefix a guidance branch of `source` conditions on. Admitted
    /// negative tokens take precedence over the negative prompt.
    pub fn guidance_prefix(
        &self,
        source: BranchSource,
        image_prompt: &str,
        negative_prompt: &str,
        negative_tokens: &[u32],
    ) -> Result<Prefix> {
        if source == BranchSource::NegativeOrStart && !negative_tokens.is_empty() {
            return Ok(Prefix {
                tokens: Arc::from(negative_tokens),
                reuses_conditioning: false,
            });
        }
        // `[reuses, *tokens]`.
        let value = self.guidance_prefix.call_values(&[
            self.loaded.to_any(),
            Any::from(tvm_ffi::String::from(source.name())),
            Any::from(tvm_ffi::String::from(image_prompt)),
            Any::from(tvm_ffi::String::from(negative_prompt)),
        ])?;
        let values = integers(value)
            .map_err(|error| Error::numerical(self.guidance_prefix.name(), &error))?;
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
    use crate::ffi::Fields;

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

    #[test]
    fn descriptions_parse_processing_denoiser_and_decoder() -> Result<()> {
        let record = Record::new(
            "image.describe",
            Fields::new()
                .with(
                    "processor",
                    Fields::new()
                        .with("vit", true)
                        .with("vae", false)
                        .with("patch_tower", true)
                        .text("pixel_dtype", "bfloat16")
                        .with(
                            "injection",
                            Fields::new()
                                .text("layout", "framed")
                                .text("positions", "temporal")
                                .with("start_token", 5_i64)
                                .with("end_token", 6_i64)
                                .map(),
                        )
                        .map(),
                )
                .with(
                    "denoiser",
                    Fields::new()
                        .with("patch_size", 2_i64)
                        .with("latent_channels", 16_i64)
                        .with("downsample", 16_i64)
                        .with("framing_tokens", 2_i64)
                        .text("image_unconditional", "conditioning")
                        .with("max_sequence_tokens", 4096_i64)
                        .text("prediction_dtype", "bfloat16")
                        .map(),
                )
                .with(
                    "decoder_range",
                    crate::ffi::list([Any::from(0.0_f64), Any::from(1.0_f64)]),
                )
                .into_any(),
        )?;
        let model = ImageModel::from_record(&record)?;
        let processing = model.processing.expect("processing");
        assert_eq!(
            processing.injection.map(|value| value.start_token),
            Some(Some(5))
        );
        assert_eq!(model.denoiser, Some(layout(2)));
        assert_eq!(model.decoder_range, Some((0.0, 1.0)));

        let empty = Record::new(
            "image.describe",
            Fields::new()
                .none("processor")
                .none("denoiser")
                .none("decoder_range")
                .into_any(),
        )?;
        assert_eq!(
            ImageModel::from_record(&empty)?,
            ImageModel {
                processing: None,
                denoiser: None,
                decoder_range: None,
            }
        );
        Ok(())
    }
}
