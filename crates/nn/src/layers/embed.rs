//! Patch embedding: a latent raster cut into patch tokens and projected into a model's width.
//!
//! ```text
//! tokens = patchify(latent)               [tokens, channels · t · s²]   exact, a layout move
//! hidden = tokens · weightᵀ + bias        [tokens, hidden]              the projection
//! ```
//!
//! The pair is the reference's kernel-equals-stride convolution over the latent, evaluated as the
//! layout move and a matrix product. The two stages are called apart, because a diffusion model
//! routes the tokens between them (it gathers the rows a rank owns, or mixes a condition's tokens
//! with noise), and the embedding unpacks a prediction back into a raster with the same
//! tokenization.

use ops::Operand;

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::{Bias, Linear, Parallel, Patchify};
use crate::weights::Scope;

/// The geometry of a [`PatchEmbed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchEmbedSpec {
    /// Latent channels.
    pub channels: u32,
    /// Frames one patch spans.
    pub temporal: u32,
    /// Rows and columns one patch spans.
    pub spatial: u32,
    /// The width the tokens are projected into.
    pub hidden: u32,
}

/// A patch embedding: the tokenization of a latent and the projection of its tokens.
pub struct PatchEmbed {
    patchify: Patchify,
    proj: Linear,
}

impl PatchEmbed {
    /// The embedding whose projection is the module at `scope` (its `weight` `[hidden,
    /// token_width]` and `bias`), at the element type the scope reads.
    ///
    /// # Errors
    ///
    /// A zero extent, a token wider than a projection holds, or a tensor the loader refuses.
    pub fn load(scope: &mut Scope<'_, '_>, spec: PatchEmbedSpec) -> Result<Self> {
        let patchify = Patchify::new(spec.channels, spec.temporal, spec.spatial)?;
        let width = u32::try_from(patchify.token_width()).map_err(|_| {
            Error::shape(
                "PatchEmbed",
                format!(
                    "a token of {} elements is wider than a projection",
                    patchify.token_width()
                ),
            )
        })?;
        let proj = Linear::load(scope, width, spec.hidden, Bias::Present, Parallel::None)?;
        Ok(Self { patchify, proj })
    }

    /// The elements of one token.
    pub fn token_width(&self) -> u64 {
        self.patchify.token_width()
    }

    /// The projection's weight, whose element type is the embedding's.
    pub fn weight(&self) -> Operand {
        self.proj.weight()
    }

    /// `latent` `[channels, frames, rows, columns]` packed into `[tokens, token_width]` tokens.
    ///
    /// # Errors
    ///
    /// As [`Patchify::forward`].
    pub fn patchify(&self, f: &Forward, latent: Operand) -> Result<Operand> {
        self.patchify.forward(f, latent)
    }

    /// `tokens` `[rows, token_width]` projected to `[rows, hidden]` in their element type.
    ///
    /// # Errors
    ///
    /// As [`Linear::forward`].
    pub fn project(&self, f: &Forward, tokens: Operand) -> Result<Operand> {
        self.proj.forward(f, tokens)
    }

    /// `tokens` `[tokens, token_width]` unpacked back into a raster of `raster` extents, the inverse
    /// of the tokenization (a prediction in token order returns to the latent's layout).
    ///
    /// # Errors
    ///
    /// As [`Patchify::inverse`].
    pub fn unpatchify(
        &self,
        f: &Forward,
        tokens: Operand,
        raster: ops::Shape,
    ) -> Result<Operand> {
        self.patchify.inverse(f, tokens, raster)
    }
}
