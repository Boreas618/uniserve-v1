//! An image request's encodings in its engine's encoder cache (the device
//! tier) and in the host encoder pools (the host tier,
//! `crate::encoder_outputs`).
//!
//! An encoding is the output of one encoder over one image the request
//! reads: an input image's block (`next_context_encode`), the feedback
//! encoding of an image the request generated, and the feedback encoding of
//! an image a migrated request carries when its context is rebuilt
//! (`generation::continuation`). Its bytes follow from what the encoder
//! reads and from what computes it, so both tiers key it by:
//!
//! - **the image**: an input image's digest (`ImageInput::digest`, which the
//!   server computed over the image's bytes, a request-dependent
//!   preprocessing parameter and the request's cache partition), or a
//!   generated image's digest over the values or the PNG its feedback reads
//!   and the request's cache partition ([`generated_image`]). The engine
//!   knows a generated image's values once its decoding resolved: a
//!   feedback encoding planned before, queued behind the decoding
//!   (`next_generation_phase`), is computed without the host tier, and the
//!   continuation that carries the image, which knows its values, looks its
//!   encoding up and stores it;
//! - **the draw**, for an encoder that samples its posterior
//!   (`ipc::ImageEncoderOutput::draws`): the request's image seed and the
//!   encoding's draw (`Call::latent_draw`), which its noise follows. An
//!   encoder that does not sample is keyed without them, so the requests
//!   reading one image share its output whatever their seeds;
//! - **what computes it**: its output's name, and in the host tier the
//!   identity its ranks report (`ipc::EncoderPoolInfo::identities`). The
//!   device tier is its world's alone, whose encoders compute alike.
//!
//! The device tier is the nearer: an input image's encoding it holds is
//! pinned for the request without a call. Otherwise the encoding is looked
//! up in the host tier before its call is planned (`encoding_ready`), and
//! the call reads its output from the pool's pages instead of encoding
//! (an import), or encodes and writes its output to them (an export), or
//! encodes alone when the pools cannot serve it
//! ([`Scheduler::attach_host_encoding`]).

use super::*;
use crate::encoder_outputs::{Access, OutputId};
use storage::key::Digest;

/// The output of an image model's vision encoder: one feature per cache
/// token of a vision block, at most.
pub(crate) const VISION_FEATURES: &str = "vision_features";
/// The output of an image model's latent encoder: one latent row per cache
/// token of a latent block, at most.
pub(crate) const LATENT_FEATURES: &str = "latent_features";

/// The name of `encoder`'s output.
pub(super) fn output_name(encoder: crate::ImageEncoder) -> &'static str {
    match encoder {
        crate::ImageEncoder::Vision => VISION_FEATURES,
        crate::ImageEncoder::Latent => LATENT_FEATURES,
    }
}

/// The digest of a generated image as a feedback encoding reads it: the
/// decoded image product's values, or its PNG for feedback that reads the
/// artifact, in the cache partition `isolation`, so that requests of other
/// partitions share no encoding of it.
pub(super) fn generated_image(image: &crate::FeedbackImage, isolation: Option<u64>) -> [u8; 32] {
    let mut digest = Digest::new(b"generated-image");
    match image {
        crate::FeedbackImage::Pixels(pixels) => {
            digest.bytes(b"uniserve-generated-image-values/1\0");
            digest.number(u64::from(pixels.height));
            digest.number(u64::from(pixels.width));
            digest.number(match pixels.range {
                crate::PixelRange::Unit => 0,
                crate::PixelRange::SignedUnit => 1,
            });
            digest.bytes(&pixels.values);
        }
        crate::FeedbackImage::Png(png) => {
            digest.bytes(b"uniserve-generated-image-png/1\0");
            digest.bytes(png.as_bytes());
        }
    }
    match isolation {
        Some(key) => {
            digest.number(1);
            digest.number(key);
        }
        None => digest.number(0),
    }
    digest.finish()
}

/// The digest of what an encoding read: the image's digest and, for an
/// encoder that samples, the request's image `seed` and the encoding's
/// `draw`.
fn inputs(image: &[u8; 32], draw: Option<(u64, u32)>) -> [u8; 32] {
    let mut digest = Digest::new(b"image-encoding");
    digest.bytes(image);
    match draw {
        Some((seed, draw)) => {
            digest.number(1);
            digest.number(seed);
            digest.number(u64::from(draw));
        }
        None => digest.number(0),
    }
    digest.finish()
}

/// The device tier's key of an encoding of output `name` that read what
/// `inputs` digests.
fn device_key(name: &str, inputs: &[u8; 32]) -> crate::kv_cache::EncoderKey {
    let mut digest = Digest::new(b"encoder-cache");
    digest.text(name);
    digest.bytes(inputs);
    digest.finish()
}

/// One encoding of an image request, as both tiers key it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Encoding {
    /// Its output among the request's.
    pub(super) output: OutputId,
    /// The digest of what it read.
    pub(super) inputs: [u8; 32],
    /// The cache tokens of the block its features write, which bound its
    /// rows (`ipc::HostOutput::rows`).
    pub(super) rows: u32,
}

impl Encoding {
    /// The device tier's key of the encoding.
    pub(super) fn device_key(&self) -> crate::kv_cache::EncoderKey {
        device_key(self.output.name, &self.inputs)
    }
}

/// What an encoding encodes, for [`Scheduler::encoding`].
#[derive(Debug, Clone, Copy)]
pub(super) enum Encoded<'a> {
    /// An input image's block.
    Input(generation::BlockCursor),
    /// Feedback encoder `encoder` of the request's generated image
    /// `image_index` (from one), whose digest is `image`.
    Feedback {
        image_index: u32,
        encoder: usize,
        image: &'a [u8; 32],
    },
}

impl Scheduler {
    /// The encoding of `request`'s image that `encoded` names, keyed as
    /// the module documentation says. `None` for a block or encoder the
    /// request does not have, or one whose cache tokens are unbounded.
    pub(super) fn encoding(
        &self,
        request: &GenerationRequest,
        encoded: Encoded<'_>,
    ) -> Option<Encoding> {
        let images = &request.multimodal_inputs.images;
        let inputs_before: usize = images.iter().map(|image| image.encoders.len()).sum();
        let feedback = &request.image_generation.feedback_encoders;
        // The encoding's ordinal among the request's encodings: its input
        // images' blocks in context order, then the feedback encodings of
        // each image it generates.
        let (encoder, image, ordinal, latent) = match encoded {
            Encoded::Input(block) => {
                let image = images.get(block.image)?;
                let encoder = *image.encoders.get(block.encoder)?;
                let ordinal = images[..block.image]
                    .iter()
                    .map(|image| image.encoders.len())
                    .sum::<usize>()
                    + block.encoder;
                (
                    encoder,
                    image.digest,
                    ordinal,
                    generation::LatentEncoding::Input(block),
                )
            }
            Encoded::Feedback {
                image_index,
                encoder: index,
                image,
            } => {
                let encoder = *feedback.get(index)?;
                let ordinal = inputs_before
                    + (image_index.saturating_sub(1) as usize) * feedback.len()
                    + index;
                (
                    encoder,
                    *image,
                    ordinal,
                    generation::LatentEncoding::Feedback {
                        image_index,
                        encoder: index,
                    },
                )
            }
        };
        let name = output_name(encoder.encoder);
        // An encoder that samples reads the request's draw besides the
        // image: the seed of the image parameters the request's admission
        // carries (`NewRequest::image`, none for a request that generates no
        // image) and the encoding's ordinal among its latent encodings.
        let draws = self
            .info
            .image_encoders
            .get(name)
            .is_some_and(|output| output.draws);
        let draw = draws.then(|| {
            let seed = if request.generates_images() {
                request.image.seed.unwrap_or(0)
            } else {
                0
            };
            (seed, generation::latent_draw(request, latent))
        });
        Some(Encoding {
            output: OutputId {
                name,
                ordinal: u32::try_from(ordinal).ok()?,
            },
            inputs: inputs(&image, draw),
            rows: encoder.cache_token_capacity(&self.generation_limits).ok()?,
        })
    }

    /// Whether request `id`'s `encoding` may be planned now: its lookup in
    /// the host encoder pools resolved, or it has none. Starts the lookup
    /// when the request has not looked the encoding up yet, ending the
    /// lookup of an encoding it no longer plans. The encoding is looked up
    /// on the replica its call will run on, to which the request is bound
    /// now, when the pools serve its output there and the request's cache
    /// policy reads or writes cached outputs (`CachePolicy`); while that
    /// replica has no room for the call, the encoding waits.
    pub(super) fn encoding_ready(&mut self, id: RequestId, encoding: Encoding) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        if self.encoder_outputs.is_none() {
            return true;
        }
        let request_key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let previous = state.host_encoding;
        let access = Access {
            read: state.req.cache.read,
            write: state.req.cache.write,
        };
        if previous == Some(encoding) {
            return self
                .encoder_outputs
                .as_ref()
                .is_none_or(|store| store.resolved(request_key, encoding.output));
        }
        if let (Some(previous), Some(store)) = (previous, self.encoder_outputs.as_mut()) {
            store.drop_output(request_key, previous.output);
        }

        let kind = CallKind::Media(match encoding.output.name {
            LATENT_FEATURES => MediaCall::LatentEncoding,
            _ => MediaCall::VisionEncoding,
        });
        let Some(row_bytes) = self
            .info
            .image_encoders
            .get(encoding.output.name)
            .map(|output| output.row_bytes)
        else {
            // A rank that states no rows of the output stores none of it.
            tracing::debug!(
                request = id.0,
                output = encoding.output.name,
                "the image encoding is computed: no rank states its rows"
            );
            if let Some(state) = self.running.get_mut(&id) {
                state.host_encoding = Some(encoding);
            }
            return true;
        };
        // The replica the call runs on, which has no room for it now: the
        // encoding waits, as its call would.
        let Some(worker) =
            self.placement
                .place(self.executor.as_ref(), &self.info, request_key, kind)
        else {
            return false;
        };
        let started = self.encoder_outputs.as_mut().is_some_and(|store| {
            store.serves(&worker, encoding.output.name)
                && store.lookup(
                    request_key,
                    &worker,
                    encoding.output,
                    &encoding.inputs,
                    encoding.rows,
                    row_bytes,
                    access,
                )
        });
        tracing::debug!(
            request = id.0,
            output = encoding.output.name,
            ordinal = encoding.output.ordinal,
            worker = %worker.0,
            rows = encoding.rows,
            started,
            "an image encoding is looked up in the host encoder pool"
        );
        // An encoding that could not be looked up is computed: its call
        // carries no host output.
        if let Some(state) = self.running.get_mut(&id) {
            state.host_encoding = Some(encoding);
        }
        self.encoder_outputs
            .as_ref()
            .is_none_or(|store| store.resolved(request_key, encoding.output))
    }

    /// Ends request `id`'s lookup of `encoding` in the host encoder pools,
    /// which the request reads from its engine's encoder cache instead.
    pub(super) fn drop_host_encoding(&mut self, id: RequestId, encoding: Encoding) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        if state.host_encoding != Some(encoding) {
            return;
        }
        state.host_encoding = None;
        let request_key = RequestKey::new(self.engine_id, id, state.request_epoch);
        if let Some(store) = self.encoder_outputs.as_mut() {
            store.drop_output(request_key, encoding.output);
        }
    }

    /// Gives `call`, request `id`'s image encoding that is registered with
    /// its call identity, the host output of the encoding the request looked
    /// up (`Call::host_output`): it reads its output from the pages its
    /// lookup pinned, or writes the output it computes to the pages its
    /// lookup reserved; no host output when the pools serve neither. The
    /// store counts the call in flight until `call_completed`.
    pub(super) fn attach_host_encoding(&mut self, id: RequestId, call: &mut Call) {
        if !matches!(
            call.kind,
            CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
        ) {
            return;
        }
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        let Some(encoding) = state.host_encoding.take() else {
            return;
        };
        let request_key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let Some(store) = self.encoder_outputs.as_mut() else {
            return;
        };
        // The encoding's output is the call's whole output: its first row is
        // the stored output's first, and its rows are bounded by the block's
        // cache tokens.
        call.host_output = store.plan_call(
            request_key,
            encoding.output,
            call.call_id,
            [0, encoding.rows],
            true,
        );
    }
}
