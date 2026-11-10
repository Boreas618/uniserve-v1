//! Rotary position embedding: the frequency spectrum, its factors and their application.
//!
//! A rotary embedding has two halves of different lifetimes. The **spectrum** ([`Spectrum`]) is a
//! constant of the checkpoint's rope parameters: one inverse frequency per rotated pair, computed
//! once at load and published as a device constant. The **rotation** is per forward: the batch's
//! positions become FP32 cosine and sine tables once ([`Rotary::factors`], the `rotary_factors`
//! operator), and every attention layer turns its queries and keys by those tables
//! ([`Rotary::apply`]). One [`Rotary`] is therefore owned by a stack, not by its layers: it holds the
//! one published table, and each layer hands it the per-head gains it owns.
//!
//! ## Spectra
//!
//! ```text
//! Default    θ^(−2i/d), i < d/2
//! Yarn       Default interpolated by the extension, ramped over the correction range
//! Sectioned  Default over d, its frequencies cycled over the M-RoPE axes  (interleaved M-RoPE)
//! Repeated   one spectrum of p pairs, repeated once per coordinate axis   (N-d RoPE)
//! ```
//!
//! **Precision is a named parameter**, because the two formulas a checkpoint's reference uses do
//! not agree bitwise: the f64 rule (`1 / θ^(2i/d)` evaluated in f64, rounded once) and the f32 rule
//! (`1 / θ^((2i) as f32 / d as f32)` evaluated in f32) differ by one to three ulp in the table and
//! widely in the device factors. A consumer keeps the rule its baseline uses. The f64 power and the
//! f64 reciprocal of the power give the same table at every shipped base, so f64 is one rule.
//!
//! **The YaRN extension is `max_position_embeddings / original_max_position_embeddings`**, which is
//! transformers' rule; the declared `factor` is recorded and not applied (vLLM applies it). The two
//! agree for a checkpoint that states consistent numbers.
//!
//! ## The q/k normalization
//!
//! A stack whose attention normalizes each query and key head before rotating it states the norm's
//! epsilon in [`QkNorm::Fused`]: the rotation then launches the fused `qk_norm_rope` program, which
//! normalizes in FP32, rotates and rounds once. That is not the same arithmetic as a separate
//! per-head RMS norm followed by the rotation (the separate form rounds in between), so a consumer
//! whose baseline fuses keeps the fused form. A stack without the norm ([`QkNorm::None`]) rotates
//! through `rope_apply` alone. The rotation is always the half-split pairing (`x[i]` against
//! `x[i + d/2]`, GPT-NeoX), and a head may rotate only its leading channels: the factor
//! tables are as wide as the spectrum, and a head wider than the rotated width passes its trailing
//! channels through.

use std::f64::consts::PI;

use ops::TensorView;
use ops::rope::{
    FactorsArgs, FactorsDescriptor, MAX_AXES, QkNormRope, QkNormRopeArgs, QkNormRopeDescriptor,
    RopeApply, RopeApplyArgs, RopeApplyDescriptor, RotaryFactors, Rotation,
};

use crate::config::Fields;
use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::Gain;
use crate::layers::launch::{PDL, QK_NORM_ROWS, ROTARY_BLOCK};
use crate::weights::{Scope, Weight};

/// The tokens one program of the strided q/k rotation normalizes, per head.
const STRIDED_TOKENS: u32 = 8;

/// The tokens one `rope_apply` program rotates.
const APPLY_TOKENS: u32 = 8;

/// The head elements one `rope_apply` program loads at a time.
const APPLY_BLOCK: u32 = 128;

/// The arithmetic an inverse-frequency table and its score scale are computed in.
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
    /// The extension factor the checkpoint declares; recorded, not applied (module doc).
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
    /// Whether the recipe truncates the correction at the boundaries; recorded, not applied.
    pub truncate: bool,
}

/// The M-RoPE cut of a spectrum: how many frequencies each position axis owns.
///
/// The axes cycle over the whole spectrum (interleaved M-RoPE): frequency `j` reads axis
/// `j % axes` while `j < widths[j % axes] · axes`, and axis 0 otherwise
/// (`ops::rope::interleaved_axes`), so every axis keeps frequencies from the whole range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sections {
    widths: [u32; MAX_AXES],
    axes: u32,
}

impl Sections {
    /// The cut of `widths` frequencies per axis, in axis order.
    ///
    /// # Errors
    ///
    /// No axis, more than [`MAX_AXES`], or an axis of no frequency.
    pub fn new(widths: &[u32]) -> Result<Self> {
        if widths.is_empty() || widths.len() > MAX_AXES {
            return Err(Error::shape(
                "Sections",
                format!("{} sections, one to {MAX_AXES} axes", widths.len()),
            ));
        }
        let mut stored = [0_u32; MAX_AXES];
        for (slot, width) in stored.iter_mut().zip(widths) {
            if *width == 0 {
                return Err(Error::shape(
                    "Sections",
                    "a section of no frequency rotates nothing".to_owned(),
                ));
            }
            *slot = *width;
        }
        Ok(Self {
            widths: stored,
            axes: widths.len() as u32,
        })
    }

    /// The frequencies each axis owns, in axis order.
    pub fn widths(&self) -> &[u32] {
        &self.widths[..self.axes as usize]
    }

    /// How many position axes the cut reads.
    pub const fn axes(&self) -> u32 {
        self.axes
    }
}

/// The inverse-frequency spectrum of a rotary embedding; see the module table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spectrum {
    /// `θ^(−2i/dim)` over `dim` rotated channels.
    Default {
        theta: f32,
        dim: u32,
        precision: Precision,
    },
    /// The default f64 spectrum over `dim` channels under a YaRN interpolation.
    Yarn { theta: f32, dim: u32, yarn: Yarn },
    /// The default spectrum over `dim` channels, its frequencies cut over the M-RoPE axes.
    Sectioned {
        theta: f32,
        dim: u32,
        sections: Sections,
        precision: Precision,
    },
    /// One spectrum of `pairs` frequencies (`θ^(−2i/(2·pairs))`) repeated once per axis: axis `a`
    /// rotates pairs `a·pairs .. (a+1)·pairs` by its own coordinate, the axes in contiguous blocks.
    Repeated {
        theta: f32,
        pairs: u32,
        axes: u32,
        precision: Precision,
    },
}

impl Spectrum {
    /// The spectrum of a Hugging Face text configuration over `dim` rotated channels.
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

        let theta = theta as f32;
        let spectrum = match kind {
            "default" => Self::Default {
                theta,
                dim,
                precision: Precision::F64,
            },
            "yarn" => Self::Yarn {
                theta,
                dim,
                yarn: yarn(fields, &maps, max_position_embeddings)?,
            },
            other => {
                return Err(fields.refuse(
                    "rope_scaling",
                    format!("names the recipe {other:?}; default and yarn are served"),
                ));
            }
        };
        spectrum.check()?;
        Ok(spectrum)
    }

    /// Refuses a spectrum with no pair, an odd width or a base outside its domain.
    fn check(&self) -> Result<()> {
        let (theta, width) = match *self {
            Self::Default { theta, dim, .. } | Self::Yarn { theta, dim, .. } => (theta, dim),
            Self::Sectioned {
                theta,
                dim,
                sections,
                ..
            } => {
                let covered: u32 = sections.widths().iter().sum();
                if covered != dim / 2 {
                    return Err(Error::shape(
                        "Spectrum",
                        format!(
                            "sections {:?} cover {covered} of the {} frequencies of {dim} channels",
                            sections.widths(),
                            dim / 2
                        ),
                    ));
                }
                (theta, dim)
            }
            Self::Repeated {
                theta, pairs, axes, ..
            } => {
                if axes == 0 || axes as usize > MAX_AXES {
                    return Err(Error::shape(
                        "Spectrum",
                        format!("{axes} repeated axes, one to {MAX_AXES}"),
                    ));
                }
                (theta, 2 * pairs)
            }
        };
        if width == 0 || !width.is_multiple_of(2) {
            return Err(Error::shape(
                "Spectrum",
                format!("a rotary width of {width} channels has no whole pair"),
            ));
        }
        if !(theta.is_finite() && theta > 0.0) {
            return Err(Error::shape(
                "Spectrum",
                format!("the rotary base {theta} is not positive"),
            ));
        }
        if matches!(self, Self::Yarn { .. }) && theta <= 1.0 {
            return Err(Error::shape(
                "Spectrum",
                format!("a YaRN correction range needs a base above one, not {theta}"),
            ));
        }
        Ok(())
    }

    /// The channels of a head the spectrum rotates: twice its frequency count.
    pub fn rotated(&self) -> u32 {
        match *self {
            Self::Default { dim, .. } | Self::Yarn { dim, .. } | Self::Sectioned { dim, .. } => dim,
            Self::Repeated { pairs, axes, .. } => 2 * pairs * axes,
        }
    }

    /// The precision the table and the score scale are computed in.
    pub const fn precision(&self) -> Precision {
        match *self {
            Self::Yarn { .. } => Precision::F64,
            Self::Default { precision, .. }
            | Self::Sectioned { precision, .. }
            | Self::Repeated { precision, .. } => precision,
        }
    }

    /// The `[rotated / 2]` FP32 inverse frequencies, in the order the factor tables are laid out.
    ///
    /// # Errors
    ///
    /// A spectrum [`Self::check`] refuses, or a YaRN window or extension that is not positive.
    pub fn inverse_frequencies(&self) -> Result<Vec<f32>> {
        self.check()?;
        match *self {
            Self::Default {
                theta,
                dim,
                precision,
            }
            | Self::Sectioned {
                theta,
                dim,
                precision,
                ..
            } => Ok(power_series(theta, dim, precision)),
            Self::Repeated {
                theta,
                pairs,
                axes,
                precision,
            } => {
                let spectrum = power_series(theta, 2 * pairs, precision);
                Ok((0..axes).flat_map(|_| spectrum.iter().copied()).collect())
            }
            Self::Yarn { theta, dim, yarn } => Ok(yarn_frequencies(theta, dim, &yarn)?.0),
        }
    }

    /// The softmax scale a head of `head_dim` channels attends with under this spectrum.
    ///
    /// `1 / sqrt(head_dim)` in the spectrum's precision; a YaRN spectrum multiplies it by its
    /// attention factor squared, because the reference scales both the cosine and the sine by it
    /// and a product of two rotated vectors carries the square.
    ///
    /// # Errors
    ///
    /// A head of no channel, or a YaRN extension that is not positive.
    pub fn score_scale(&self, head_dim: u32) -> Result<f32> {
        if head_dim == 0 {
            return Err(Error::shape(
                "Spectrum::score_scale",
                "a head of no channel has no softmax scale".to_owned(),
            ));
        }
        let denominator = f64::from(head_dim).sqrt();
        match (*self, self.precision()) {
            (Self::Yarn { theta, dim, yarn }, _) => {
                let attention = yarn_frequencies(theta, dim, &yarn)?.1;
                Ok((f64::from(attention * attention) / denominator) as f32)
            }
            (_, Precision::F64) => Ok((1.0 / denominator) as f32),
            (_, Precision::F32) => Ok(1.0 / (head_dim as f32).sqrt()),
        }
    }

    /// The M-RoPE axes the factors read, in rotated channels, with whether they interleave.
    fn axes(&self) -> Option<([u32; MAX_AXES], u32, bool)> {
        match *self {
            Self::Default { .. } | Self::Yarn { .. } => None,
            // Each section counts frequencies and the operator counts rotated channels, so each
            // width is doubled.
            Self::Sectioned { sections, .. } => {
                let mut widths = [0; MAX_AXES];
                for (slot, width) in widths.iter_mut().zip(sections.widths()) {
                    *slot = 2 * width;
                }
                Some((widths, sections.axes(), true))
            }
            Self::Repeated { pairs, axes, .. } => {
                let mut widths = [0; MAX_AXES];
                for slot in widths.iter_mut().take(axes as usize) {
                    *slot = 2 * pairs;
                }
                Some((widths, axes, false))
            }
        }
    }
}

/// `θ^(−2i/dim)` for `i < dim / 2` under `precision`'s rule (module doc).
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
/// extrapolated frequency and the ones above it are interpolated by the extension, with a linear
/// ramp between - the high frequencies (a pair at index 0 has period 2π tokens) must not move.
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
            "Spectrum::Yarn",
            format!(
                "an extension of {} positions over a {fitted}-position window",
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
            // The ramp is 1 above the correction range, where the pairs are interpolated, and 0
            // below it, where they keep their extrapolated frequency.
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

/// Whether the rotation normalizes each query and key head first, fused into the same program.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QkNorm {
    /// The rotation alone (`rope_apply`).
    None,
    /// A per-head RMS norm with this variance floor and gain, fused into the rotation
    /// (`qk_norm_rope`).
    Fused { epsilon: f32, gain: Gain },
}

/// The heads one [`Rotary`] turns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotarySpec {
    /// Channels of one head; at least the spectrum's rotated width, the rest pass through.
    pub head_dim: u32,
    /// Query heads one token carries on this rank.
    pub heads: u32,
    /// Key heads one token carries on this rank.
    pub kv_heads: u32,
    /// Whether each head is normalized before it turns.
    pub qk_norm: QkNorm,
}

/// The per-head gains of a fused q/k norm: one `[head_dim]` weight for the queries, one for the keys.
#[derive(Clone, Copy, Debug)]
pub struct Gains {
    pub query: TensorView,
    pub key: TensorView,
}

/// One stack's rotary embedding: the published spectrum, the factors of a batch and the rotation.
pub struct Rotary {
    /// The published spectrum [`Self::factors`] reads; absent for a rotation whose factor tables
    /// the runner computes on the host ([`Self::new`]), which publishes nothing it never reads.
    table: Option<Weight>,
    spectrum: Spectrum,
    spec: RotarySpec,
}

impl Rotary {
    /// The rotary of `spectrum` for heads of `spec`, publishing the spectrum under `scope` as the
    /// constant `inv_freq` its device factors read.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], or a spectrum [`Spectrum::inverse_frequencies`] refuses.
    pub fn load(scope: &mut Scope<'_, '_>, spectrum: Spectrum, spec: RotarySpec) -> Result<Self> {
        let frequencies = spectrum.inverse_frequencies()?;
        let mut rotary = Self::new(spectrum, spec)?;
        rotary.table =
            Some(scope.constant_f32("inv_freq", &[frequencies.len() as u64], &frequencies)?);
        Ok(rotary)
    }

    /// The rotation of `spectrum` for heads of `spec` alone: its factor tables come from the
    /// caller, so it publishes no spectrum and [`Self::factors`] refuses.
    ///
    /// # Errors
    ///
    /// No head, a head narrower than the rotated width, or query heads that do not group the key
    /// heads.
    pub fn new(spectrum: Spectrum, spec: RotarySpec) -> Result<Self> {
        let rotated = spectrum.rotated();
        if spec.heads == 0 || spec.kv_heads == 0 || !spec.heads.is_multiple_of(spec.kv_heads) {
            return Err(Error::shape(
                "Rotary",
                format!(
                    "{} query heads over {} key heads",
                    spec.heads, spec.kv_heads
                ),
            ));
        }
        if rotated > spec.head_dim {
            return Err(Error::shape(
                "Rotary",
                format!(
                    "a rotation of {rotated} channels does not fit a {} channel head",
                    spec.head_dim
                ),
            ));
        }
        Ok(Self {
            table: None,
            spectrum,
            spec,
        })
    }

    /// The spectrum this rotary turns by.
    pub const fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }

    /// The columns of one factor table row: one per rotated pair.
    pub fn columns(&self) -> u32 {
        self.spectrum.rotated() / 2
    }

    /// The position axes the factors read: one for a plain rotation, one per section otherwise.
    pub fn axes(&self) -> u32 {
        self.spectrum.axes().map_or(1, |(_, axes, _)| axes)
    }

    /// The factor table's descriptor, for positions of `positions` and a table of `factors`.
    fn factors_descriptor(&self, positions: ops::DType, factors: ops::DType) -> FactorsDescriptor {
        let axes = self.spectrum.axes();
        FactorsDescriptor {
            dtype: factors,
            positions,
            columns: self.columns(),
            block: ROTARY_BLOCK,
            scale: 1.0_f32.into(),
            pdl: PDL,
            axes: axes.map_or(1, |(_, axes, _)| axes),
            interleaved: axes.is_some_and(|(_, _, interleaved)| interleaved),
            sections: axes.map_or([0; MAX_AXES], |(widths, _, _)| widths),
        }
    }

    /// The fused rotation's descriptor, for a projection of `input` and an output of `output`.
    ///
    /// `affine` states whether the per-head normalization has a learned weight, which is a
    /// different program: with one it loads the weight, without one it has no weight ports.
    fn fused_descriptor(
        &self,
        (epsilon, gain): (f32, Gain),
        input: ops::DType,
        output: ops::DType,
        affine: bool,
    ) -> QkNormRopeDescriptor {
        let head = u64::from(self.spec.head_dim);
        QkNormRopeDescriptor {
            dtype: input,
            output,
            head_dim: self.spec.head_dim,
            q_heads: self.spec.heads,
            k_heads: self.spec.kv_heads,
            rotation: Rotation::Half,
            epsilon: epsilon.into(),
            rotated: self.spectrum.rotated(),
            rows: QK_NORM_ROWS,
            block: head.next_power_of_two().min(1024) as u32,
            affine,
            pdl: PDL,
            factors: ops::DType::F32,
            strided: false,
            centred: affine && gain == Gain::Centred,
        }
    }

    /// The epsilon of the fused norm, refusing a rotary built without one.
    fn epsilon(&self, owner: &'static str) -> Result<(f32, Gain)> {
        match self.spec.qk_norm {
            QkNorm::Fused { epsilon, gain } => Ok((epsilon, gain)),
            QkNorm::None => Err(Error::shape(
                owner,
                "this rotary normalizes no head; its rotation has no fused norm".to_owned(),
            )),
        }
    }

    /// The batch's factor tables: `positions` in, FP32 `cosine` and `sine` out.
    ///
    /// `positions` is `[tokens]`, or `[axes, tokens]` axis-major for a sectioned or repeated
    /// spectrum; `cosine` and `sine` are `[tokens, columns]`.
    ///
    /// # Errors
    ///
    /// A rotation built without its spectrum ([`Self::new`]), positions that are not one coordinate
    /// per axis and token, tables of another extent, or the
    /// operator's refusal.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        let columns = u64::from(self.columns());
        cosine
            .require_row("Rotary::factors", cosine.dtype, columns)
            .map_err(Error::from)?;
        sine.require_row("Rotary::factors", cosine.dtype, columns)
            .map_err(Error::from)?;
        let axes = u64::from(self.axes());
        if self.spectrum.axes().is_some()
            && (positions.shape.rank() < 2 || positions.shape.dim(0) != axes)
        {
            return Err(Error::shape(
                "Rotary::factors",
                format!("sectioned positions need one leading coordinate per axis, {axes} in all"),
            ));
        }
        if !positions.numel().is_multiple_of(axes) {
            return Err(Error::shape(
                "Rotary::factors",
                format!(
                    "{} position coordinates do not divide into {axes} axes",
                    positions.numel()
                ),
            ));
        }
        let tokens = positions.numel() / axes;
        if cosine.rows() != tokens || sine.rows() != tokens {
            return Err(Error::shape(
                "Rotary::factors",
                format!(
                    "{tokens} positions but {} and {} factor rows",
                    cosine.rows(),
                    sine.rows()
                ),
            ));
        }
        let table = self.table.as_ref().ok_or_else(|| {
            Error::shape(
                "Rotary::factors",
                "this rotation published no spectrum: its factors are the caller's".to_owned(),
            )
        })?;
        let descriptor = self.factors_descriptor(positions.dtype, cosine.dtype);
        let args = FactorsArgs {
            positions,
            frequencies: table.view(),
            cosine,
            sine,
        };

        f.operate::<RotaryFactors>(&descriptor, args)
    }

    /// Turn `query` and `key` into `q_out` and `k_out` by the `cosine` and `sine` tables.
    ///
    /// `query` and `q_out` are `[tokens, heads, head_dim]`, `key` and `k_out`
    /// `[tokens, kv_heads, head_dim]`, and the tables `[tokens, columns]` FP32. A fused rotary
    /// normalizes each head first, by `gains` when the stack's heads carry learned gains and
    /// without a weight when they do not; a rotary without the norm takes no gains and turns the
    /// queries and then the keys.
    ///
    /// # Errors
    ///
    /// Views of another geometry, gains of another width, gains given to a rotary without the
    /// norm, or the operator's refusal.
    #[allow(
        clippy::too_many_arguments,
        reason = "a rotation call carries the two projections, the layer's gains, the two factor \
                  tables and the two outputs; the design states them separately"
    )]
    pub fn apply(
        &self,
        f: &Forward,
        query: TensorView,
        key: TensorView,
        gains: Option<Gains>,
        cosine: TensorView,
        sine: TensorView,
        q_out: TensorView,
        k_out: TensorView,
    ) -> Result<()> {
        let head_dim = u64::from(self.spec.head_dim);
        let heads = u64::from(self.spec.heads);
        let kv_heads = u64::from(self.spec.kv_heads);
        for (name, view, count) in [
            ("query", query, heads),
            ("q_out", q_out, heads),
            ("key", key, kv_heads),
            ("k_out", k_out, kv_heads),
        ] {
            if view.shape.rank() != 3 || view.shape.dim(1) != count || view.shape.dim(2) != head_dim
            {
                return Err(Error::shape(
                    "Rotary::apply",
                    format!(
                        "{name} is {:?}, [tokens, {count}, {head_dim}] expected",
                        view.shape.dims(),
                    ),
                ));
            }
            view.require_row("Rotary::apply", view.dtype, head_dim)
                .map_err(Error::from)?;
        }
        let tokens = query.shape.dim(0);
        if key.shape.dim(0) != tokens
            || q_out.shape.dim(0) != tokens
            || k_out.shape.dim(0) != tokens
        {
            return Err(Error::shape(
                "Rotary::apply",
                "the query and key tensors have different token counts".to_owned(),
            ));
        }
        let columns = u64::from(self.columns());
        if cosine.numel() != tokens * columns || sine.numel() != tokens * columns {
            return Err(Error::shape(
                "Rotary::apply",
                format!(
                    "{} and {} factor elements for {tokens} tokens of {columns} factors",
                    cosine.numel(),
                    sine.numel()
                ),
            ));
        }
        if let Some(gains) = gains
            && (gains.query.numel() != head_dim || gains.key.numel() != head_dim)
        {
            return Err(Error::shape(
                "Rotary::apply",
                format!("per-head gains of {head_dim} elements"),
            ));
        }

        match self.spec.qk_norm {
            QkNorm::Fused { epsilon, gain } => {
                let descriptor = self.fused_descriptor(
                    (epsilon, gain),
                    query.dtype,
                    q_out.dtype,
                    gains.is_some(),
                );
                let args = QkNormRopeArgs {
                    query,
                    key,
                    q_weight: gains.map(|gains| gains.query),
                    k_weight: gains.map(|gains| gains.key),
                    cosine,
                    sine,
                    q_out,
                    k_out,
                };
                f.operate::<QkNormRope>(&descriptor, args)
            }
            QkNorm::None => {
                if gains.is_some() {
                    return Err(Error::shape(
                        "Rotary::apply",
                        "this rotary normalizes no head, so it takes no gains".to_owned(),
                    ));
                }
                for (x, out, count) in [
                    (query, q_out, self.spec.heads),
                    (key, k_out, self.spec.kv_heads),
                ] {
                    let descriptor = RopeApplyDescriptor {
                        dtype: x.dtype,
                        output: out.dtype,
                        head_dim: self.spec.head_dim,
                        heads: count,
                        rotation: Rotation::Half,
                        rotated: self.spectrum.rotated(),
                        rows: APPLY_TOKENS,
                        block: APPLY_BLOCK,
                        pdl: PDL,
                    };
                    let args = RopeApplyArgs {
                        x,
                        cosine,
                        sine,
                        out,
                    };
                    f.operate::<RopeApply>(&descriptor, args)?;
                }
                Ok(())
            }
        }
    }

    /// Normalize and turn `query` and `key` in place: two views of `[tokens, heads, head_dim]` into
    /// wider rows, the norm without a learned weight.
    ///
    /// `cosine` and `sine` are `[table rows, columns]` of a type the caller chose; token `t` reads
    /// row `t % table rows`, so tiles decoded in one launch share one table
    /// (`ops::rope::QkNormRopeDescriptor::strided`).
    ///
    /// # Errors
    ///
    /// A rotary without the fused norm, or the operator's refusal.
    pub fn apply_strided(
        &self,
        f: &Forward,
        query: TensorView,
        key: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        let epsilon = self.epsilon("Rotary::apply_strided")?;
        let descriptor = QkNormRopeDescriptor {
            k_heads: self.spec.heads,
            rows: STRIDED_TOKENS,
            factors: cosine.dtype,
            strided: true,
            ..self.fused_descriptor(epsilon, query.dtype, query.dtype, false)
        };
        let args = QkNormRopeArgs {
            query,
            key,
            q_weight: None,
            k_weight: None,
            cosine,
            sine,
            q_out: query,
            k_out: key,
        };

        f.operate::<QkNormRope>(&descriptor, args)
    }
}

/// How [`grid_positions`] places the cells of one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// Cell `i` of an axis of `n` cells sits at its centre in `[-1, 1)`: `2 · (i + 0.5) / n − 1`,
    /// evaluated in f32 as `(i + 0.5) / n` first.
    CellCentre,
}

/// The rotary coordinates of every cell of a grid of `extents`, axis-major `[axes, cells]`.
///
/// Cells run in raster order (the first axis slowest), and each coordinate is the cell's
/// normalized position along its axis multiplied by `scale` in f32: a spectrum of raw inverse
/// frequencies then turns a cell by `scale · position · frequency` (a `2π` scale gives the angle
/// of a reference that multiplies its grid by `2π` at the rotary call). Axis-major because a
/// repeated or sectioned rotary reads one leading coordinate per axis.
///
/// # Errors
///
/// An axis of no cells.
pub fn grid_positions(
    extents: &[u32],
    normalization: Normalization,
    scale: f32,
) -> Result<Vec<f32>> {
    if extents.contains(&0) {
        return Err(Error::shape(
            "grid_positions",
            format!("a grid of {extents:?} cells has an empty axis"),
        ));
    }
    let coordinate = |index: u32, size: u32| match normalization {
        Normalization::CellCentre => {
            let centre = (index as f32 + 0.5) / size as f32;
            scale * (2.0 * centre - 1.0)
        }
    };
    let cells: usize = extents.iter().map(|extent| *extent as usize).product();
    let mut table = vec![0.0_f32; extents.len() * cells];
    for cell in 0..cells {
        // The cell's index along each axis, the last axis fastest.
        let mut rest = cell;
        for (axis, extent) in extents.iter().enumerate().rev() {
            let size = *extent as usize;
            table[axis * cells + cell] = coordinate((rest % size) as u32, *extent);
            rest /= size;
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid's coordinates are each cell's centre along its axis, scaled, axis-major and in raster
    /// order.
    #[test]
    fn grid_positions_are_scaled_cell_centres_in_raster_order() {
        let table = grid_positions(&[2, 3], Normalization::CellCentre, 1.0).unwrap();
        // Two rows of three columns: the rows' coordinates, then the columns'.
        assert_eq!(&table[..6], &[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
        let third = (0.5_f32 / 3.0) * 2.0 - 1.0;
        assert_eq!(table[6], third);
        assert_eq!(table[7], 0.0);
        assert_eq!(table[8], -third);
        assert_eq!(&table[9..], &table[6..9]);

        // The scale multiplies the f32 coordinate after the centre is formed.
        let turned =
            grid_positions(&[3], Normalization::CellCentre, std::f32::consts::TAU).unwrap();
        assert_eq!(turned[0], std::f32::consts::TAU * third);
        assert!(grid_positions(&[2, 0], Normalization::CellCentre, 1.0).is_err());
    }

    /// A rotation built without its spectrum turns heads by the caller's factors and refuses to
    /// compute factors of its own.
    #[test]
    fn a_rotation_without_its_spectrum_computes_no_factors() {
        use crate::forward::{Arena, Forward};
        use ops::{DType, Shape};

        let rotary = Rotary::new(
            Spectrum::Default {
                theta: 10_000.0,
                dim: 16,
                precision: Precision::F32,
            },
            RotarySpec {
                head_dim: 16,
                heads: 2,
                kv_heads: 2,
                qk_norm: QkNorm::None,
            },
        )
        .unwrap();
        let arena = Arena::counting();
        let f = Forward::recording(&arena);
        let tensor = |dtype, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
        let (cosine, sine) = (tensor(DType::F32, &[3, 8]), tensor(DType::F32, &[3, 8]));
        let positions = tensor(DType::I32, &[3]);
        assert!(rotary.factors(&f, positions, cosine, sine).is_err());
        let heads = || tensor(DType::BF16, &[3, 2, 16]);
        let (query, key) = (heads(), heads());
        rotary
            .apply(&f, query, key, None, cosine, sine, heads(), heads())
            .unwrap();
    }

    /// A plain spectrum is the f64 power series, its first pair unity and the rest falling.
    #[test]
    fn a_default_spectrum_is_the_power_series_of_its_base() {
        let spectrum = Spectrum::Default {
            theta: 1_000_000.0,
            dim: 128,
            precision: Precision::F64,
        };
        let table = spectrum.inverse_frequencies().unwrap();
        assert_eq!(table.len(), 64);
        assert_eq!(table[0], 1.0);
        assert_eq!(table[1], (1.0 / 1_000_000_f64.powf(2.0 / 128.0)) as f32);
        assert!(table.windows(2).all(|pair| pair[0] > pair[1]));
        assert_eq!(spectrum.rotated(), 128);
    }

    /// The f32 rule is evaluated in f32: its table is the f32 power, which differs from the f64
    /// rule's in the last bits, and the consumer chooses which one by name.
    #[test]
    fn the_f32_rule_is_evaluated_in_f32() {
        let spectrum = |precision| Spectrum::Default {
            theta: 5_000_000.0,
            dim: 128,
            precision,
        };
        let narrow = spectrum(Precision::F32).inverse_frequencies().unwrap();
        for (index, value) in narrow.iter().enumerate() {
            assert_eq!(
                *value,
                1.0 / 5_000_000_f32.powf((2 * index) as f32 / 128.0),
                "pair {index}"
            );
        }
        let wide = spectrum(Precision::F64).inverse_frequencies().unwrap();
        assert_ne!(narrow, wide);
    }

    /// A repeated spectrum gives every axis the same frequencies, each axis in its own block.
    #[test]
    fn a_repeated_spectrum_gives_every_axis_one_block_of_the_same_frequencies() {
        let spectrum = Spectrum::Repeated {
            theta: 10_000.0,
            pairs: 16,
            axes: 3,
            precision: Precision::F64,
        };
        let table = spectrum.inverse_frequencies().unwrap();
        assert_eq!(table.len(), 48);
        assert_eq!(spectrum.rotated(), 96);
        for axis in 1..3 {
            assert_eq!(table[axis * 16..(axis + 1) * 16], table[..16]);
        }
        // The per-axis spectrum is over 32 channels, not the 96 the axes rotate together.
        assert_eq!(table[1], (1.0 / 10_000_f64.powf(2.0 / 32.0)) as f32);
    }

    /// M-RoPE sections count frequencies and must cover the spectrum exactly.
    #[test]
    fn sections_cover_the_spectrum_exactly() {
        let sectioned = |widths: &[u32]| Spectrum::Sectioned {
            theta: 5_000_000.0,
            dim: 128,
            sections: Sections::new(widths).unwrap(),
            precision: Precision::F32,
        };
        assert_eq!(
            sectioned(&[24, 20, 20])
                .inverse_frequencies()
                .unwrap()
                .len(),
            64
        );
        assert!(sectioned(&[24, 20, 19]).inverse_frequencies().is_err());
        assert!(Sections::new(&[]).is_err());
        assert!(Sections::new(&[0, 32, 32]).is_err());
        assert!(Sections::new(&[8, 8, 8, 8, 32]).is_err());
    }

    /// YaRN keeps the fastest pair, divides the slowest by the extension `max / original`, and
    /// squares its attention factor into the score scale.
    #[test]
    fn yarn_interpolates_the_slow_pairs_and_scales_the_scores() {
        let yarn = Yarn {
            factor: 4.0,
            original_max_position_embeddings: 4096,
            max_position_embeddings: 40_960,
            attention_factor: None,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: None,
            mscale_all_dim: None,
            truncate: false,
        };
        let plain = Spectrum::Default {
            theta: 1_000_000.0,
            dim: 128,
            precision: Precision::F64,
        };
        let scaled = Spectrum::Yarn {
            theta: 1_000_000.0,
            dim: 128,
            yarn,
        };
        let (plain_table, table) = (
            plain.inverse_frequencies().unwrap(),
            scaled.inverse_frequencies().unwrap(),
        );
        assert_eq!(table[0], plain_table[0]);
        assert!((table[63] - plain_table[63] / 10.0).abs() < 1e-9);
        assert!(table.windows(2).all(|pair| pair[0] > pair[1]));
        let expected = (0.1 * 10.0_f32.ln() + 1.0).powi(2) / 128.0_f32.sqrt();
        assert!((scaled.score_scale(128).unwrap() - expected).abs() < 1e-6);
    }

    /// The score scale follows the spectrum's precision: the two rules differ at 72 channels.
    #[test]
    fn the_score_scale_is_computed_in_the_spectrums_precision() {
        let spectrum = |precision| Spectrum::Default {
            theta: 10_000.0,
            dim: 36,
            precision,
        };
        assert_eq!(
            spectrum(Precision::F64).score_scale(72).unwrap(),
            (1.0 / 72_f64.sqrt()) as f32
        );
        assert_eq!(
            spectrum(Precision::F32).score_scale(72).unwrap(),
            1.0 / 72_f32.sqrt()
        );
        assert_ne!(
            spectrum(Precision::F64).score_scale(72).unwrap(),
            spectrum(Precision::F32).score_scale(72).unwrap()
        );
        assert!(spectrum(Precision::F64).score_scale(0).is_err());
    }

    /// A Hugging Face configuration reads its recipe from either spelling, and refuses a recipe
    /// it does not serve.
    #[test]
    fn a_hugging_face_recipe_is_read_from_either_spelling() {
        let read = |value: serde_json::Value| {
            Spectrum::from_hf(&Fields::of("config.json", &value).unwrap(), 128, 40_960)
        };
        assert_eq!(
            read(serde_json::json!({"rope_theta": 1_000_000.0})).unwrap(),
            Spectrum::Default {
                theta: 1_000_000.0,
                dim: 128,
                precision: Precision::F64
            }
        );
        let yarn = read(serde_json::json!({
            "rope_theta": 1_000_000.0,
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                                "original_max_position_embeddings": 4096},
        }))
        .unwrap();
        assert!(matches!(
            yarn,
            Spectrum::Yarn { yarn, .. } if yarn.max_position_embeddings == 40_960
                && yarn.original_max_position_embeddings == 4096
        ));
        assert!(
            read(serde_json::json!({"rope_scaling": {"type": "linear", "factor": 2.0}})).is_err()
        );
        assert!(read(serde_json::json!({"partial_rotary_factor": 0.5})).is_err());
        assert!(
            read(serde_json::json!({
                "rope_scaling": {"type": "yarn", "factor": 2.0},
                "rope_parameters": {"rope_type": "default"},
            }))
            .is_err()
        );
    }
}
