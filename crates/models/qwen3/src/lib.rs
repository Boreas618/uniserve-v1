//! The Qwen3 family: its checkpoint configuration, the decoder layer and stack every Qwen3 model
//! composes, and the models built from them.
//!
//! - [`config`]: [`Config`], read once from a checkpoint's `config.json`;
//! - [`decoder`]: [`DecoderLayer`] and [`DecoderStack`], Hugging Face's `Qwen3DecoderLayer` and its
//!   MoE and hybrid variants;
//! - [`causal_lm`]: [`Qwen3ForCausalLm`], `Qwen3MoeForCausalLM` and the Qwen3.5-MoE language
//!   model, implementing [`nn::CausalLm`];
//! - [`mtp`]: the Qwen3.5 multi-token-prediction drafter;
//! - [`vision`]: the Qwen3-VL vision tower, implementing [`nn::capability::Vision`];
//! - [`vl`]: [`Qwen3VlModel`], `Qwen3VLModel` as an encoder over a prefix of its layers;
//! - [`positions`]: the host arithmetic of M-RoPE coordinates and the tower's position samples;
//! - [`family`]: the family values execution loads and request processing serves.
//!
//! The Qwen-VL processor rules that size media to the tower's patch grid (`smart_resize`, frame
//! sampling) are request processing's ([`processing::qwen_vl`]).
//!
//! A model receives borrowed [`nn::Forward`] resources and tensor views; every device operation is
//! a call to an `nn` layer, and every tensor a forward computes is reserved from the forward.

pub mod causal_lm;
pub mod config;
pub mod decoder;
pub mod family;
pub mod mtp;
pub mod positions;
pub mod vision;
pub mod vl;

pub use causal_lm::Qwen3ForCausalLm;
pub use config::{Config, Mixer, MoeConfig, TextConfig, VisionConfig};
pub use decoder::{DecoderLayer, DecoderStack};
pub use vision::VisionTower;
pub use vl::Qwen3VlModel;

use nn::layers::{Access, Copy};
use nn::{Forward, Shape, TensorView};

/// What this crate can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The checkpoint metadata is not a supported Qwen3 configuration, a deployment's split does
    /// not divide what it shards, or a call's inputs do not describe the model.
    #[error("qwen3: {0}")]
    Config(String),
    /// A layer, an operator or the weight loader refused the call.
    #[error(transparent)]
    Nn(#[from] nn::Error),
}

impl From<ops::Error> for Error {
    fn from(error: ops::Error) -> Self {
        Self::Nn(error.into())
    }
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for nn::Error {
    /// A shared-library failure passes through unchanged; a refusal of this crate's own is
    /// [`nn::Error::Model`] naming the family.
    fn from(error: Error) -> Self {
        match error {
            Error::Nn(error) => error,
            other => nn::Error::model("qwen3", other),
        }
    }
}

/// Copy the row-major `[rows, columns]` tensor `source` into `destination`, a row-major view of
/// the same extents and element type: the caller's buffer a capability writes its answer into.
///
/// # Errors
///
/// Views of other extents, element types or layouts, or the copy's refusal.
pub(crate) fn write(f: &Forward, source: TensorView, destination: TensorView) -> Result<()> {
    if source.shape != destination.shape
        || source.dtype != destination.dtype
        || source.shape.rank() != 2
        || !source.is_row_major()
        || !destination.is_row_major()
    {
        return Err(Error::Config(format!(
            "a {} {:?} answer written into a {} {:?} buffer",
            source.dtype,
            source.shape.dims(),
            destination.dtype,
            destination.shape.dims()
        )));
    }
    copy_block(
        f,
        source.reshape(block(source.shape)),
        destination.reshape(block(source.shape)),
    )
}

/// `[rows, columns]` as the four-axis block a copy moves: `[1, 1, rows, columns]`.
fn block(shape: Shape) -> Shape {
    Shape::new(&[1, 1, shape.dim(0), shape.dim(1)])
}

/// Copy the four-axis block `source` into `destination`, each read through its own strides; whole
/// vectors move when every stride but the last, the row and both addresses allow it.
pub(crate) fn copy_block(f: &Forward, source: TensorView, destination: TensorView) -> Result<()> {
    // `copy_block`'s wide form: strides and the row in whole runs of 16 elements, 16-byte addresses.
    const WIDE: u64 = 16;
    let strides = destination.strides.as_slice();
    let mut out = [0_i64; 4];
    for (slot, stride) in out.iter_mut().zip(strides) {
        *slot = i64::try_from(*stride)
            .map_err(|_| Error::Config(format!("a copy stride of {stride}")))?;
    }
    let aligned = |view: &TensorView| {
        view.strides.as_slice()[..3]
            .iter()
            .all(|stride| stride.is_multiple_of(WIDE))
            && view.strides.last() == 1
            && view.ptr.address().is_multiple_of(16)
    };
    let access = if destination.shape.last().is_multiple_of(WIDE)
        && aligned(&source)
        && aligned(&destination)
    {
        Access::Vector
    } else {
        Access::Element
    };
    Copy::new(destination.dtype, access)?.forward(f, source, destination, &out)?;
    Ok(())
}
