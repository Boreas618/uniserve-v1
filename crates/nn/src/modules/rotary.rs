//! The rotary position embedding of a stack: its inverse frequencies, and the factors of a
//! batch's positions.
//!
//! A rotary embedding has two halves of different lifetimes. The inverse frequencies are a
//! constant of the checkpoint's rope parameters: one per rotated pair, computed once at load
//! and published as a constant of the module. The factors are per forward: the batch's
//! positions become FP32 cosine and sine tables once ([`Rotary::forward`]), and every attention
//! layer turns its queries and keys by those tables ([`crate::functional::qk_norm_rope`],
//! [`crate::functional::rotary`]). One [`Rotary`] is therefore a module of the stack, not of
//! its layers.
//!
//! ## Spectra
//!
//! ```text
//! Default    θ^(−2i/d), i < d/2
//! Yarn       Default interpolated by the extension, ramped over the correction range
//! Sectioned  Default over d, its frequencies cycled over the M-RoPE axes  (interleaved M-RoPE)
//! Repeated   one spectrum of d / axes channels, repeated once per axis    (N-d RoPE)
//! ```
//!
//! **Precision is a named value** ([`Precision`]), because the two formulas a checkpoint's
//! reference uses do not agree bitwise: the f64 rule (`1 / θ^(2i/d)` evaluated in f64, rounded
//! once) and the f32 rule (`1 / θ^((2i) as f32 / d as f32)` evaluated in f32) differ by one to
//! three ulp in the table and widely in the factors. A model keeps the rule its reference uses,
//! so the table is computed in scalar arithmetic at that precision and published as it is; a
//! tensor expression has no power of that rounding. The f64 power and the f64 reciprocal of
//! the power give the same table at every shipped base, so f64 is one rule.
//!
//! **The YaRN extension is `max_position_embeddings / original_max_position_embeddings`**,
//! which is transformers' rule; the declared `factor` is read and not applied (vLLM applies
//! it). The two agree for a checkpoint that states consistent numbers.

use std::f64::consts::PI;

use ops::rope::MAX_AXES;
use tensor::{DType, Tensor};

use crate::config::Fields;
use crate::error::{Error, Result};
use crate::functional;
use crate::var_builder::VarBuilder;

/// The arithmetic an inverse-frequency table and its softmax scale are computed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// `1 / θ^(2i/d)` and `1 / sqrt(head_dim)` in f64, each rounded once to f32.
    F64,
    /// `1 / θ^((2i) as f32 / d as f32)` and `1 / sqrt(head_dim as f32)` in f32.
    F32,
}

/// The YaRN interpolation of a rotary spectrum, as a checkpoint states it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    /// The extension factor the checkpoint declares; read, not applied (module documentation).
    pub factor: f32,
    /// The context length the recipe was fitted to.
    pub original_max_position_embeddings: u32,
    /// The extended context length: the numerator of the applied extension.
    pub max_position_embeddings: u32,
    /// The attention scaling, when the checkpoint declares one.
    pub attention_factor: Option<f32>,
    /// The high-frequency correction boundary.
    pub beta_fast: f32,
    /// The low-frequency correction boundary.
    pub beta_slow: f32,
    /// The YaRN magnitude scale, when declared.
    pub mscale: Option<f32>,
    /// The all-dimension magnitude scale, when declared.
    pub mscale_all_dim: Option<f32>,
    /// Whether the recipe truncates the correction at the boundaries; read, not applied.
    pub truncate: bool,
}

/// How a rotary embedding's frequencies follow from its base and are laid over the position
/// axes; see the module table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spectrum {
    /// The power series over the rotated channels, read by one position axis.
    Default,
    /// The f64 power series under a YaRN interpolation, read by one position axis.
    Yarn(Yarn),
    /// The power series over the rotated channels, cut over the M-RoPE axes: `widths` holds the
    /// frequencies each of the first `axes` axes owns, in axis order.
    ///
    /// The axes cycle over the whole spectrum (interleaved M-RoPE): frequency `j` reads axis
    /// `j mod axes` while `j < widths[j mod axes] · axes`, and axis 0 otherwise, so every axis
    /// keeps frequencies from the whole range.
    Sectioned {
        /// Frequencies per axis; the entries past `axes` are zero.
        widths: [u32; MAX_AXES],
        /// The position axes the cut reads.
        axes: u32,
    },
    /// One power series of `dim / axes` channels repeated once per axis: each axis rotates its
    /// own contiguous block of pairs by its own coordinate.
    Repeated {
        /// The position axes, each with the same frequencies.
        axes: u32,
    },
}

impl Spectrum {
    /// The M-RoPE cut of `widths` frequencies per axis, in axis order.
    ///
    /// # Errors
    ///
    /// No axis, more than [`MAX_AXES`], or an axis of no frequency.
    pub fn sectioned(widths: &[u32]) -> Result<Self> {
        if widths.is_empty() || widths.len() > MAX_AXES || widths.contains(&0) {
            return Err(Error::shape(
                "Rotary",
                format!(
                    "sections {widths:?}: one to {MAX_AXES} axes, each of at least one frequency"
                ),
            ));
        }

        let mut stored = [0; MAX_AXES];
        stored[..widths.len()].copy_from_slice(widths);
        Ok(Self::Sectioned {
            widths: stored,
            axes: widths.len() as u32,
        })
    }
}

/// What a checkpoint states of a [`Rotary`]: its base, the channels it rotates and its
/// spectrum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryConfig {
    /// The base θ of the power series.
    pub theta: f32,
    /// The channels of a head the embedding rotates: twice its frequencies. A head wider than
    /// this passes its trailing channels through.
    pub dim: u32,
    /// How the frequencies follow from the base and which position axis each reads.
    pub spectrum: Spectrum,
    /// The arithmetic the table and the softmax scale are computed in; a YaRN spectrum is
    /// computed in f64.
    pub precision: Precision,
}

impl RotaryConfig {
    /// The configuration of a Hugging Face text configuration over `dim` rotated channels.
    ///
    /// Reads `rope_theta` (default 10000), `partial_rotary_factor` (only 1 is served), and the
    /// recipe of `rope_parameters` or `rope_scaling` (`rope_parameters` takes precedence, and a
    /// value either states is read through both in that order): `default` or `yarn`.
    /// `max_position_embeddings` is the configuration's own, the numerator of a YaRN extension.
    ///
    /// # Errors
    ///
    /// A field of the wrong type, conflicting recipe aliases, a recipe other than `default` or
    /// `yarn`, a partial rotation, or a value outside its domain.
    pub fn from_hf(fields: &Fields<'_>, dim: u32, max_position_embeddings: u32) -> Result<Self> {
        let scaling = fields.optional_object("rope_scaling")?;
        let parameters = fields.optional_object("rope_parameters")?;
        let maps = [parameters.as_ref(), scaling.as_ref()];

        let stated_type = match &scaling {
            Some(scaling) => scaling.optional_text("type")?,
            None => None,
        };
        let kind = match &parameters {
            Some(parameters) => parameters.optional_text("rope_type")?,
            None => None,
        }
        .or(stated_type)
        .unwrap_or("default");
        if let Some(other) = stated_type
            && other != kind
        {
            return Err(fields.refuse(
                "rope_scaling",
                "states a rotary type that conflicts with rope_parameters",
            ));
        }

        let theta = match fields.optional_number("rope_theta")? {
            Some(theta) => Some(theta),
            None => first(&maps, |map| map.optional_number("rope_theta"))?,
        }
        .unwrap_or(10_000.0);
        if !(theta.is_finite() && theta > 0.0) {
            return Err(fields.refuse("rope_theta", "must be finite and positive"));
        }

        let partial = match fields.optional_number("partial_rotary_factor")? {
            Some(partial) => Some(partial),
            None => first(&maps, |map| map.optional_number("partial_rotary_factor"))?,
        }
        .unwrap_or(1.0);
        if partial != 1.0 {
            return Err(fields.refuse(
                "partial_rotary_factor",
                "is not served: every channel of a head rotates",
            ));
        }

        let spectrum = match kind {
            "default" => Spectrum::Default,
            "yarn" => Spectrum::Yarn(yarn(fields, &maps, max_position_embeddings)?),
            other => {
                return Err(fields.refuse(
                    "rope_scaling",
                    format!("names the recipe {other:?}; default and yarn are served"),
                ));
            }
        };
        let config = Self {
            theta: theta as f32,
            dim,
            spectrum,
            precision: Precision::F64,
        };

        config.check()?;
        Ok(config)
    }

    /// The channels one power series spans: the rotated channels, or one axis's share of them
    /// when the series repeats per axis.
    fn series_width(&self) -> u32 {
        match self.spectrum {
            Spectrum::Repeated { axes } => self.dim / axes.max(1),
            _ => self.dim,
        }
    }

    /// Refuses a configuration with no pair, an odd width, a base outside its domain, or a
    /// spectrum that does not lay over the rotated channels.
    fn check(&self) -> Result<()> {
        const OWNER: &str = "Rotary";

        match self.spectrum {
            Spectrum::Default => {}
            Spectrum::Yarn(_) => {
                if self.theta <= 1.0 {
                    return Err(Error::shape(
                        OWNER,
                        format!(
                            "a YaRN correction range needs a base above one, not {}",
                            self.theta
                        ),
                    ));
                }
                if self.precision != Precision::F64 {
                    return Err(Error::shape(
                        OWNER,
                        "a YaRN spectrum is computed in f64".to_owned(),
                    ));
                }
            }
            Spectrum::Sectioned { widths, axes } => {
                let stated = widths.iter().take(axes as usize);
                let covered: u32 = stated.clone().sum();

                if axes == 0
                    || axes as usize > MAX_AXES
                    || stated.clone().any(|width| *width == 0)
                    || covered != self.dim / 2
                {
                    return Err(Error::shape(
                        OWNER,
                        format!(
                            "sections {:?} cover {covered} of the {} frequencies of {} channels",
                            &widths[..(axes as usize).min(MAX_AXES)],
                            self.dim / 2,
                            self.dim
                        ),
                    ));
                }
            }
            Spectrum::Repeated { axes } => {
                if axes == 0 || axes as usize > MAX_AXES || !self.dim.is_multiple_of(axes) {
                    return Err(Error::shape(
                        OWNER,
                        format!(
                            "{axes} repeated axes over {} channels: one to {MAX_AXES} axes of \
                             equal shares",
                            self.dim
                        ),
                    ));
                }
            }
        }

        let width = self.series_width();
        if width == 0 || !width.is_multiple_of(2) {
            return Err(Error::shape(
                OWNER,
                format!("a rotary width of {width} channels has no whole pair"),
            ));
        }
        if !(self.theta.is_finite() && self.theta > 0.0) {
            return Err(Error::shape(
                OWNER,
                format!("the rotary base {} is not positive", self.theta),
            ));
        }
        Ok(())
    }

    /// The `[dim / 2]` FP32 inverse frequencies, in the order the factor tables are laid out.
    ///
    /// # Errors
    ///
    /// A configuration with no whole pair, a base outside its domain or a spectrum that does
    /// not lay over the rotated channels, or a YaRN window or extension that is not positive.
    pub fn inverse_frequencies(&self) -> Result<Vec<f32>> {
        self.check()?;

        match self.spectrum {
            Spectrum::Default | Spectrum::Sectioned { .. } => {
                Ok(power_series(self.theta, self.dim, self.precision))
            }
            Spectrum::Repeated { axes } => {
                let series = power_series(self.theta, self.series_width(), self.precision);
                Ok((0..axes).flat_map(|_| series.iter().copied()).collect())
            }
            Spectrum::Yarn(yarn) => Ok(yarn_frequencies(self.theta, self.dim, &yarn)?.0),
        }
    }

    /// The softmax scale a head of `head_dim` channels attends with under this embedding.
    ///
    /// `1 / sqrt(head_dim)` in the configuration's precision; a YaRN spectrum multiplies it by
    /// its attention factor squared, because the reference scales both the cosine and the sine
    /// by it and a product of two rotated vectors has the square.
    ///
    /// # Errors
    ///
    /// A head of no channel, or a YaRN extension that is not positive.
    pub fn score_scale(&self, head_dim: u32) -> Result<f32> {
        if head_dim == 0 {
            return Err(Error::shape(
                "Rotary",
                "a head of no channel has no softmax scale".to_owned(),
            ));
        }
        let denominator = f64::from(head_dim).sqrt();

        match (self.spectrum, self.precision) {
            (Spectrum::Yarn(yarn), _) => {
                let attention = yarn_frequencies(self.theta, self.dim, &yarn)?.1;
                Ok((f64::from(attention * attention) / denominator) as f32)
            }
            (_, Precision::F64) => Ok((1.0 / denominator) as f32),
            (_, Precision::F32) => Ok(1.0 / (head_dim as f32).sqrt()),
        }
    }

    /// The rotated channels each position axis owns, in axis order, and whether the axes cycle
    /// over the frequencies; no sections for a spectrum of one position axis.
    fn sections(&self) -> (Vec<u32>, bool) {
        match self.spectrum {
            Spectrum::Default | Spectrum::Yarn(_) => (Vec::new(), false),

            // A section counts frequencies and the factors count rotated channels, so each
            // width is doubled.
            Spectrum::Sectioned { widths, axes } => {
                let stated = widths.iter().take(axes as usize);
                (stated.map(|width| 2 * width).collect(), true)
            }
            Spectrum::Repeated { axes } => (vec![self.series_width(); axes as usize], false),
        }
    }
}

/// `θ^(−2i/dim)` for `i < dim / 2` under `precision`'s rule (module documentation).
fn power_series(theta: f32, dim: u32, precision: Precision) -> Vec<f32> {
    (0..dim / 2)
        .map(|index| match precision {
            Precision::F64 => {
                (1.0 / f64::from(theta).powf(2.0 * f64::from(index) / f64::from(dim))) as f32
            }
            Precision::F32 => 1.0 / theta.powf((2 * index) as f32 / dim as f32),
        })
        .collect()
}

/// transformers' `_compute_yarn_parameters`: the interpolated table and the attention factor.
///
/// The correction range `beta_fast`/`beta_slow` splits the pairs: the pairs below it keep their
/// extrapolated frequency and the ones above it are interpolated by the extension, with a
/// linear ramp between. The high frequencies (a pair at index 0 has a period of 2π tokens) do
/// not move.
fn yarn_frequencies(theta: f32, dim: u32, yarn: &Yarn) -> Result<(Vec<f32>, f32)> {
    let base = f64::from(theta);
    let half = (dim / 2) as usize;
    let extrapolation: Vec<f64> = (0..half)
        .map(|index| 1.0 / base.powf(2.0 * index as f64 / f64::from(dim)))
        .collect();

    // The length the recipe was fitted to is both the window the correction range is measured
    // over and the denominator of the extension.
    let fitted = f64::from(yarn.original_max_position_embeddings);
    let factor = f64::from(yarn.max_position_embeddings) / fitted;
    if !(fitted > 0.0 && factor.is_finite() && factor > 0.0) {
        return Err(Error::shape(
            "Rotary",
            format!(
                "a YaRN extension of {} positions over a {fitted}-position window",
                yarn.max_position_embeddings
            ),
        ));
    }

    let (low, high) = correction_range(
        f64::from(yarn.beta_fast),
        f64::from(yarn.beta_slow),
        f64::from(dim),
        base,
        fitted,
    );
    let frequencies = (0..half)
        .map(|index| {
            let interpolation = extrapolation[index] / factor;

            // The ramp is 1 above the correction range, where the pairs are interpolated, and
            // 0 below it, where they keep their extrapolated frequency.
            let ramp = if low == high {
                f64::from(index as f64 >= high)
            } else {
                ((index as f64 - low) / (high - low)).clamp(0.0, 1.0)
            };
            (interpolation * ramp + extrapolation[index] * (1.0 - ramp)) as f32
        })
        .collect();

    let mscale = f64::from(yarn.mscale.unwrap_or(1.0));
    let attention = match (yarn.attention_factor, yarn.mscale, yarn.mscale_all_dim) {
        (Some(declared), ..) => f64::from(declared),
        (None, Some(_), Some(all)) => {
            yarn_mscale(factor, mscale) / yarn_mscale(factor, f64::from(all))
        }
        (None, ..) => yarn_mscale(factor, mscale),
    };
    Ok((frequencies, attention as f32))
}

/// transformers' `_yarn_get_mscale`.
fn yarn_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// transformers' `_find_correction_range` over the rotary width.
fn correction_range(
    beta_fast: f64,
    beta_slow: f64,
    dim: f64,
    base: f64,
    max_position: f64,
) -> (f64, f64) {
    let dimension = |rotations: f64| -> f64 {
        (dim * (max_position / (rotations * 2.0 * PI)).ln()) / (2.0 * base.ln())
    };
    let low = dimension(beta_fast).floor().max(0.0);
    let high = dimension(beta_slow).ceil().min(dim - 1.0);

    (low, high)
}

/// The first of `maps` that states a value through `read`.
fn first<T>(
    maps: &[Option<&Fields<'_>>],
    read: impl Fn(&Fields<'_>) -> Result<Option<T>>,
) -> Result<Option<T>> {
    for map in maps.iter().flatten() {
        if let Some(value) = read(map)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// The YaRN recipe's fields from `maps`, the recipe's objects in precedence order.
fn yarn(fields: &Fields<'_>, maps: &[Option<&Fields<'_>>], extended: u32) -> Result<Yarn> {
    let number = |key: &str| first(maps, |map| map.optional_number(key));

    let factor = number("factor")?
        .ok_or_else(|| fields.refuse("rope_scaling", "a yarn recipe requires factor"))?
        as f32;
    let original = first(maps, |map| {
        map.optional_index("original_max_position_embeddings")
    })?
    .unwrap_or(extended);
    let beta_fast = number("beta_fast")?.map_or(32.0, |value| value as f32);
    let beta_slow = number("beta_slow")?.map_or(1.0, |value| value as f32);

    for (name, value) in [
        ("factor", factor),
        ("beta_fast", beta_fast),
        ("beta_slow", beta_slow),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(fields.refuse(
                "rope_scaling",
                format!("yarn {name} must be finite and positive"),
            ));
        }
    }
    if original == 0 {
        return Err(fields.refuse(
            "rope_scaling",
            "yarn original_max_position_embeddings must be positive",
        ));
    }

    Ok(Yarn {
        factor,
        original_max_position_embeddings: original,
        max_position_embeddings: extended,
        attention_factor: number("attention_factor")?.map(|value| value as f32),
        beta_fast,
        beta_slow,
        mscale: number("mscale")?.map(|value| value as f32),
        mscale_all_dim: number("mscale_all_dim")?.map(|value| value as f32),
        truncate: first(maps, |map| map.optional_flag("truncate"))?.unwrap_or(true),
    })
}

/// One stack's rotary embedding: its published inverse frequencies and the factors of a
/// batch's positions.
pub struct Rotary {
    inv_freq: Tensor,
    config: RotaryConfig,
}

impl Rotary {
    /// The rotary embedding of `config`, with its inverse frequencies published as the
    /// constant `inv_freq` under `vb`'s path.
    ///
    /// # Errors
    ///
    /// A configuration [`RotaryConfig::inverse_frequencies`] refuses, or a constant the loader
    /// cannot publish.
    pub fn load(config: &RotaryConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let frequencies = config.inverse_frequencies()?;

        Ok(Self {
            inv_freq: vb.constant("inv_freq", DType::F32, frequencies.len(), &frequencies)?,
            config: *config,
        })
    }

    /// The configuration the embedding was loaded with.
    pub const fn config(&self) -> &RotaryConfig {
        &self.config
    }

    /// The columns of one factor table row: one per rotated pair.
    pub const fn columns(&self) -> u32 {
        self.config.dim / 2
    }

    /// The position axes the factors read: one, or one per section or repetition.
    pub const fn axes(&self) -> u32 {
        match self.config.spectrum {
            Spectrum::Default | Spectrum::Yarn(_) => 1,
            Spectrum::Sectioned { axes, .. } | Spectrum::Repeated { axes } => axes,
        }
    }

    /// The factor tables of a batch: `(cos, sin)`, FP32 `[tokens, columns]` each, of
    /// `positions`.
    ///
    /// `positions` is `[tokens]`, or `[axes, tokens]` axis-major for a sectioned or repeated
    /// spectrum ([`functional::rotary_factors`]).
    ///
    /// # Errors
    ///
    /// Positions that are not one coordinate per axis and token, or the operator's refusal.
    pub fn forward(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let (sections, interleaved) = self.config.sections();

        functional::rotary_factors(positions, &self.inv_freq, &sections, interleaved)
    }
}
