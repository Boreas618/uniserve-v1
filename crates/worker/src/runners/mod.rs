//! The runners: one implementation of [`Runner`](crate::Runner) per capability a model exposes.
//!
//! A runner holds its model through a capability trait of `nn` and dispatches on nothing else.
//! What the implementations share (the profile run, capture, replay, the input buffer and the
//! generations) is [`Graphs`](crate::Graphs)'s.
//!
//! | Runner | Capability | Phases |
//! | --- | --- | --- |
//! | [`CausalLmRunner`] | `nn::family::TextModel` | prefill, decode, verification |
//! | [`EncoderRunner`] | `nn::capability::Encoder` | a video request's text encoding |
//! | [`VisionRunner`] | `nn::capability::Vision` of a `TextModel` | a token request's image encoding |
//! | [`MediaHost`] | none | a video request's host work: reading its condition media, encoding its video and audio, muxing its artifact |
//! | [`DenoiserRunner`] | `nn::Denoiser` | a video request's latent preparation and its denoising steps, over its state in its slot's row |
//! | [`DecoderRunner`] | `nn::Decoder`, with `nn::Splicer` for a video | a video request's decode of its clips and its audio track, from the latents in the rank's media products |
//!
//! A media runner reads and writes a request's products and encoder outputs in the blocks the
//! rank's device block tables hold (`memory::buffers::BufferMemory`), and issues the copies that
//! start with its batches through it, on its own stream.

pub mod causal_lm;
// The host conversions of condition media: an image's patch rows for the vision runner, a
// video request's condition pixels and tracks for the media host.
pub(crate) mod conditions;
pub mod decoder;
pub mod denoiser;
pub mod encoder;
pub mod media;
pub mod vision;

use ipc::batch::{CallOutput, CallStatus, ErrorCode};
use platform::Stream;
use tensor::Tensor;

pub use causal_lm::CausalLmRunner;
pub use decoder::{DecoderConfig, DecoderRunner};
pub use denoiser::{DenoiserConfig, DenoiserRunner};
pub use encoder::EncoderRunner;
pub use media::MediaHost;
pub use vision::VisionRunner;

use crate::Result;

/// A copy of device bytes: from an address, to an address, and how many bytes.
pub(crate) type DeviceCopy = (u64, u64, u64);

/// A call's output before anything is known of it: performed, with no token and the default
/// state.
pub(crate) fn performed() -> CallOutput {
    CallOutput {
        status: CallStatus::Ok,
        error_code: None,
        error_message: None,
        tokens: Vec::new(),
        step_index: 0,
        cache_computed_len: 0,
        cache_visible_len: 0,
        token_logprobs: Vec::new(),
        prompt_logprobs: Vec::new(),
        media_output: None,
        draft_token_ids: Vec::new(),
    }
}

/// The output of a call whose request's inputs this rank refused, saying why.
pub(crate) fn refused(message: String) -> CallOutput {
    CallOutput {
        status: CallStatus::Error,
        error_code: Some(ErrorCode::InputError),
        error_message: Some(message),
        ..performed()
    }
}

/// The bytes of one element of `dtype`.
///
/// # Errors
///
/// [`crate::Error::Invalid`] for an element type with no whole-byte layout.
pub(crate) fn element_bytes(dtype: tensor::DType) -> Result<u64> {
    ops::abi::bytes(dtype).ok_or_else(|| {
        crate::Error::Invalid(format!(
            "{} has no whole-byte element",
            ops::abi::name(dtype)
        ))
    })
}

/// The address of `tensor`'s first element: a device address, or for a meta tensor the address
/// it was made at, which nothing reads.
pub(crate) fn address(tensor: &Tensor) -> Result<u64> {
    Ok(backend::operand(tensor)?.ptr.address())
}

/// Issues `copies` on `stream`, behind the work issued on it before; with no stream, for a
/// runner whose memory holds no device, nothing is issued.
///
/// # Safety
///
/// Every range lies in device memory its owner keeps mapped until the stream has passed the
/// copies, and nothing but these copies writes the destinations meanwhile.
pub(crate) unsafe fn issue(copies: &[DeviceCopy], stream: Option<&Stream>) -> Result<()> {
    let Some(stream) = stream else {
        return Ok(());
    };

    for &(from, to, bytes) in copies {
        let bytes = usize::try_from(bytes)
            .map_err(|_| crate::Error::Resource("a copy exceeds the address space"))?;

        // SAFETY: forwarded from the caller.
        unsafe { platform::memcpy_async(to, from, bytes, stream) }?;
    }
    Ok(())
}

/// Waits for everything issued on `stream`, a media runner's: its forwards, the copies of their
/// outputs and the copies that started with its batch. With no stream nothing was issued.
pub(crate) fn settle(stream: Option<&Stream>) -> Result<()> {
    match stream {
        Some(stream) => Ok(stream.synchronize()?),
        None => Ok(()),
    }
}
