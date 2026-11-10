//! Tokenized generation requests and their capacity calculations.
//!
//! Requests own their input data and effective parameters. The engine owns
//! mutable computation progress, output delivery, and physical allocations.
//!
//! The server's model preprocessing builds a [`GenerationRequest`] and sizes
//! it with the capacity helpers here against the engine's resolved limits
//! before submission. The engine's admission (`Scheduler::enqueue`) runs
//! [`GenerationRequest::validate`], checks
//! [`GenerationRequest::required_features`] with
//! [`GenerationLimits::covers`], and then runs
//! [`GenerationRequest::validate_resources`] and
//! [`GenerationRequest::max_cache_tokens`] against the loaded limits.

use serde::{Deserialize, Serialize};

use super::{ImageEncoder, RequestId, SamplingParams, SamplingParamsError};

/// Positioned media consumed alongside the already-tokenized positive prompt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultimodalInputs {
    /// Input images in nondecreasing prompt-token position order, as
    /// `GenerationRequest::validate` requires.
    pub images: Vec<ImageInput>,
}

/// An encoded image and the model's requirements for adding it to context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInput {
    /// The SHA-256 digest of what the image's encoders read of the request:
    /// the image's bytes, any request-dependent preprocessing parameter that
    /// changes what they compute (a pixel budget the request's input images
    /// share), and the request's cache partition
    /// (`CachePolicy::isolation_key`), which the server folds in so that
    /// requests of other partitions share no encoder output. The engine keys
    /// the image's encoder outputs, in its encoder cache and in the host
    /// encoder pools, by this digest, the encoder and, for an encoder that
    /// samples, the request's draw.
    pub digest: [u8; 32],
    /// Base64-encoded input image.
    pub b64: String,
    /// Exclusive prompt-token position at the end of this image's marker gap.
    /// Equal positions preserve input order, including multiple images at one marker.
    pub position: u32,
    /// Logical positions contributed after all encoders finish; independent of cache length.
    pub num_positions: u32,
    /// Required encoder stages in context write order, with their cache contributions.
    pub encoders: Vec<ImageEncoderInput>,
}

impl ImageInput {
    /// Whether every encoder of the image injects token-model features (a
    /// vision encoder), so the image enters the context as attention blocks
    /// of a context prefill. An image with a latent encoder is ingested by
    /// calls of its own instead.
    pub fn injects_features(&self) -> bool {
        !self.encoders.is_empty()
            && self
                .encoders
                .iter()
                .all(|input| input.encoder == ImageEncoder::Vision)
    }
}

/// One encoder input and its physical cache contribution.
///
/// `num_cache_tokens` is exact when known. Otherwise `max_cache_tokens` bounds
/// the worker-selected length; absent limits use the loaded encoder capacity
/// (see [`ImageEncoderInput::cache_token_capacity`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageEncoderInput {
    /// The encoder consuming this image. List order determines context write
    /// order.
    pub encoder: ImageEncoder,
    /// Exact cache tokens this encoder writes, when known before execution.
    pub num_cache_tokens: Option<u32>,
    /// Upper bound on a worker-selected cache length. When both it and
    /// `num_cache_tokens` are present, `GenerationRequest::validate` requires
    /// `num_cache_tokens <= max_cache_tokens`.
    pub max_cache_tokens: Option<u32>,
    /// The height and width in pixels the encoder's transform resizes the
    /// image to, when known before execution: a latent encoder encodes the
    /// image at that size, which a deployment under CUDA graphs must have
    /// captured.
    pub height: Option<u32>,
    pub width: Option<u32>,
}

/// Returns the runtime feature that executes one image encoder.
const fn encoder_feature(encoder: ImageEncoder) -> GenerationFeatures {
    match encoder {
        ImageEncoder::Latent => GenerationFeatures::LATENT_ENCODE,
        ImageEncoder::Vision => GenerationFeatures::VISION_ENCODE,
    }
}

bitflags::bitflags! {
    /// Generation branches present in a runtime or required by one request.
    ///
    /// Bit 3 is unassigned: it named a branch no runtime serves, and a set
    /// carried between processes keeps every other bit's meaning.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
    pub struct GenerationFeatures: u8 {
        /// Autoregressive understanding execution.
        const UNDERSTANDING = 1 << 0;
        /// Vision feature encoding.
        const VISION_ENCODE = 1 << 1;
        /// Variational-autoencoder latent encoding.
        const LATENT_ENCODE = 1 << 2;
        /// Denoising passes of token canvases that read a prompt's prefix cache
        /// without writing it.
        const TOKEN_DENOISING = 1 << 4;
    }
}

impl GenerationFeatures {
    /// Returns the admission diagnostic name of one represented feature.
    ///
    /// Features are checked in the order understanding, latent encode, vision
    /// encode, token denoising; an empty set yields
    /// `runtime_generation_features`.
    pub fn name(self) -> &'static str {
        if self.contains(Self::UNDERSTANDING) {
            "runtime_und_execution"
        } else if self.contains(Self::LATENT_ENCODE) {
            "runtime_vae_encode"
        } else if self.contains(Self::VISION_ENCODE) {
            "runtime_vit_encode"
        } else if self.contains(Self::TOKEN_DENOISING) {
            "runtime_token_denoise"
        } else {
            "runtime_generation_features"
        }
    }
}

impl std::fmt::Display for GenerationFeatures {
    /// Writes the diagnostic name chosen by [`GenerationFeatures::name`].
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Worker and scheduler limits needed to size a request's encoder work.
///
/// The server supplies the loaded model's sizing facts, and the engine's
/// `resolve_generation_limits` takes the served features from the workers'
/// call kinds and clamps the capacities to what they report. Zero in any
/// numeric field declares no capacity for that resource;
/// [`GenerationRequest::validate_resources`] rejects requests that need it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GenerationLimits {
    /// Generation features implemented by the runtime.
    pub features: GenerationFeatures,
    /// Maximum VAE grid tokens for a worker-selected length.
    pub max_vae_grid_tokens: u32,
    /// Maximum vision grid tokens for a worker-selected length.
    pub max_vit_grid_tokens: u32,
    /// Maximum VAE feature product size in bytes.
    pub max_latent_feature_bytes: u64,
    /// Maximum vision feature product size in bytes.
    pub max_vision_feature_bytes: u64,
    /// Maximum concurrently pinned encoder-cache entries.
    pub encoder_cache_entries: u32,
    /// The most likely tokens a request may ask for at a generated or a prompt position
    /// (`n_logprobs`, `n_prompt_logprobs`): what the workers' samplers compute for every row.
    pub max_logprobs: u32,
}

impl GenerationLimits {
    /// Checks that the workers compute as many candidates at a position as `sampling` asks for
    /// at a generated or a prompt one.
    ///
    /// A position's candidates are computed by the workers' captured graphs, `max_logprobs` of
    /// them for every row, so a request may ask for that many and no more (vLLM's rule).
    ///
    /// # Errors
    ///
    /// `LogprobCount`, naming the request field that asks for more.
    pub fn covers_logprobs(
        &self,
        sampling: &SamplingParams,
    ) -> Result<(), GenerationResourceError> {
        for (requested, asked, parameter) in [
            (
                sampling.n_logprobs,
                sampling.generated_logprobs_requested(),
                "logprobs",
            ),
            (
                sampling.n_prompt_logprobs,
                sampling.prompt_logprobs_requested(),
                "prompt_logprobs",
            ),
        ] {
            if asked && requested > self.max_logprobs {
                return Err(GenerationResourceError::LogprobCount {
                    parameter,
                    requested,
                    available: self.max_logprobs,
                });
            }
        }
        Ok(())
    }

    /// Checks whether the runtime covers every feature required by a request.
    ///
    /// A failure carries the missing feature set for admission diagnostics.
    pub fn covers(&self, needs: GenerationFeatures) -> Result<(), GenerationFeatures> {
        let missing = needs.difference(self.features);
        if !missing.is_empty() {
            return Err(missing);
        }
        Ok(())
    }
}

impl GenerationRequest {
    /// Maximum physical cache tokens required by the input and output budget.
    ///
    /// Sums the prompt, the `max_und_tokens` budget and each context image's
    /// encoder cache bounds. Saturation preserves conservative admission for
    /// host-sized token budgets.
    ///
    /// # Errors
    ///
    /// Returns `GenerationResourceError::UnboundedImageCache` when a counted
    /// encoder input resolves to a zero cache length.
    pub fn max_cache_tokens(
        &self,
        limits: &GenerationLimits,
    ) -> Result<usize, GenerationResourceError> {
        let input_images = self
            .multimodal_inputs
            .images
            .iter()
            .try_fold(0usize, |total, image| {
                Ok(total.saturating_add(encoder_cache_bound(&image.encoders, limits)?))
            })?;
        Ok(self
            .prompt_token_ids
            .len()
            .saturating_add(self.max_und_tokens)
            .saturating_add(input_images))
    }

    /// Encoder-cache entries that admission reserves for the ordered input encoders.
    /// Each encoder retains its reservation even if another image has the same hash.
    ///
    /// The count is zero when the cache policy neither reads nor writes.
    pub fn num_encoder_cache_entries(&self) -> usize {
        if self.cache.read || self.cache.write {
            self.multimodal_inputs
                .images
                .iter()
                .map(|image| image.encoders.len())
                .fold(0usize, usize::saturating_add)
        } else {
            0
        }
    }

    /// Checks reachable encoder work against loaded capacity. Admission
    /// separately acquires prefix cache and cache entries from their pools.
    ///
    /// # Errors
    ///
    /// Returns the first failure among, in check order: the
    /// [`GenerationRequest::max_cache_tokens`] error; `MissingRuntimeBound` for a
    /// reachable encoder kind whose feature-byte limit is zero; and
    /// `MissingRuntimeBound` or `EncoderCacheCapacity` when reserved
    /// encoder-cache entries exceed `encoder_cache_entries`; and `LogprobCount` when the
    /// request asks for more candidates at a generated or a prompt position than `max_logprobs`.
    pub fn validate_resources(
        &self,
        limits: &GenerationLimits,
    ) -> Result<(), GenerationResourceError> {
        self.max_cache_tokens(limits)?;
        limits.covers_logprobs(&self.sampling)?;

        // Each encoder kind reachable from context images needs a nonzero
        // feature-product size.
        for (step, bytes, resource) in [
            (
                ImageEncoder::Latent,
                limits.max_latent_feature_bytes,
                "max_latent_feature_bytes",
            ),
            (
                ImageEncoder::Vision,
                limits.max_vision_feature_bytes,
                "max_vision_feature_bytes",
            ),
        ] {
            let used = self.context_encoders().any(|encoder| encoder == step);
            if used && bytes == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound { resource });
            }
        }

        let entries = self.num_encoder_cache_entries();
        if entries > 0 {
            if limits.encoder_cache_entries == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound {
                    resource: "encoder_cache_entries",
                });
            }
            if entries > limits.encoder_cache_entries as usize {
                return Err(GenerationResourceError::EncoderCacheCapacity {
                    requested: entries,
                    available: limits.encoder_cache_entries,
                });
            }
        }
        Ok(())
    }
}

/// Sums [`ImageEncoderInput::cache_token_capacity`] over one image's encoder
/// inputs, saturating instead of overflowing.
fn encoder_cache_bound(
    inputs: &[ImageEncoderInput],
    limits: &GenerationLimits,
) -> Result<usize, GenerationResourceError> {
    inputs.iter().try_fold(0usize, |total, input| {
        Ok(total.saturating_add(input.cache_token_capacity(limits)? as usize))
    })
}

impl ImageEncoderInput {
    /// Maximum cache contribution of this encoder, resolving dynamic lengths from the model.
    ///
    /// Uses `num_cache_tokens` when present, then `max_cache_tokens`, then the loaded
    /// grid-token capacity for the encoder kind. Fails with
    /// `GenerationResourceError::UnboundedImageCache` when the resolved length is
    /// zero.
    pub fn cache_token_capacity(
        &self,
        limits: &GenerationLimits,
    ) -> Result<u32, GenerationResourceError> {
        let capacity = match self.encoder {
            ImageEncoder::Latent => limits.max_vae_grid_tokens,
            ImageEncoder::Vision => limits.max_vit_grid_tokens,
        };
        let tokens = self
            .num_cache_tokens
            .or(self.max_cache_tokens)
            .unwrap_or(capacity);
        if tokens == 0 {
            return Err(GenerationResourceError::UnboundedImageCache {
                call: self.encoder.as_str(),
            });
        }
        Ok(tokens)
    }
}

/// Failures while deriving bounded generation resources.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenerationResourceError {
    /// A worker-selected image cache length has no declared or loaded maximum.
    #[error("{call} has a worker-selected KV length but the runtime declares no bound")]
    UnboundedImageCache {
        /// Encoder call missing a bound.
        call: &'static str,
    },
    /// A resource the request needs has a zero runtime maximum.
    #[error("the request requires the runtime to declare {resource}")]
    MissingRuntimeBound {
        /// Name of the missing resource maximum.
        resource: &'static str,
    },
    /// Context encoder-cache demand exceeds runtime capacity.
    #[error("requested encoder-cache entries ({requested}) exceed runtime capacity ({available})")]
    EncoderCacheCapacity {
        /// Requested cache entry count.
        requested: usize,
        /// Available cache entry count.
        available: u32,
    },
    /// More candidates asked for at a position than the deployment computes (vLLM's message).
    #[error("requested {parameter} of {requested}, which is greater than max allowed: {available}")]
    LogprobCount {
        /// The request field: `logprobs` (generated positions) or `prompt_logprobs`.
        parameter: &'static str,
        /// The candidates asked for; a request for every candidate asks for `u32::MAX`.
        requested: u32,
        /// The deployment's `max_logprobs`.
        available: u32,
    },
}

/// Scheduler-relevant prefix-cache behavior resolved during compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheResume {
    pub epoch: u64,
    pub request_id: u64,
}

/// Scheduler-relevant prefix-cache behavior resolved during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Whether admission may reuse cached prefixes or encoder outputs.
    pub read: bool,
    /// Whether completed context may populate caches.
    pub write: bool,
    /// Optional tenant or request-group cache namespace.
    pub isolation_key: Option<u64>,
    /// The prefix seed of the epoch whose state the request continues with:
    /// a request an epoch change migrates with `in_flight = keep`, when the
    /// new epoch computes other numerics. Its prefix lookups match the
    /// pages that epoch computed, it reads no host KV cache, whose pages are
    /// keyed by the epoch that serves, and a request continuing so writes
    /// no public page (`write`), since its pages are neither epoch's. Its
    /// private continuation retains them for subsequent migrations.
    #[serde(default)]
    pub state_seeds: Vec<[u8; 32]>,
    /// Private recovered pages of a prior request. This is runtime
    /// provenance, independent of the content keys of shared cache pages.
    pub resume: Option<CacheResume>,
}

impl Default for CachePolicy {
    /// Returns a shared-cache policy with reads and writes enabled.
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            isolation_key: None,
            state_seeds: Vec::new(),
            resume: None,
        }
    }
}

/// Validated scheduler-facing generation request.
///
/// The value contains immutable request inputs. Capacity requirements are derived
/// from these inputs and loaded model limits; the engine owns actual reservations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Engine request identity.
    pub request_id: RequestId,
    /// Final positive model prompt, including template and image-marker tokens.
    pub prompt_token_ids: Vec<u32>,
    /// Media positions refer directly to `prompt_token_ids`.
    pub multimodal_inputs: MultimodalInputs,
    /// Text sampling parameters.
    pub sampling: SamplingParams,
    /// Maximum understanding tokens generated by the request.
    pub max_und_tokens: usize,
    /// Text sequences that terminate generation.
    pub stop_strings: Vec<String>,
    /// Token identities that terminate generation.
    pub stop_token_ids: Vec<u32>,
    /// Scheduler priority.
    pub priority: i32,
    /// Prefix and encoder-cache policy.
    pub cache: CachePolicy,
    /// Whether a matched stopping token is included in public text output.
    pub include_stop_token: bool,
    /// The tokens a migrated request generated on the engines that served it
    /// before, in order, every one it counted as generated; empty for a new
    /// request. They continue its prompt: the engine
    /// computes their cache and generates after them, and counts them as
    /// generated against `max_und_tokens` and `min_tokens`, in the penalties
    /// and bad words of its sampling, and in its usage.
    #[serde(default)]
    pub generated_token_ids: Vec<u32>,
    /// The LoRA adapter the request computes with: the slot of the
    /// worker's adapter pool the daemon loaded it into, or 0 for the base
    /// model alone (`uniserve.adapters`).
    #[serde(default)]
    pub adapter: u32,
}

impl GenerationRequest {
    /// Returns whether the request resumes generation it began on another
    /// engine.
    pub fn is_migrated(&self) -> bool {
        !self.generated_token_ids.is_empty()
    }

    /// Returns the runtime features the request's computation reaches:
    /// understanding and the encoders of its context images. A request a
    /// block-diffusion runtime generates in canvases needs
    /// [`GenerationRequest::canvas_features`] instead.
    pub fn required_features(&self) -> GenerationFeatures {
        self.context_encoders()
            .fold(GenerationFeatures::UNDERSTANDING, |needs, step| {
                needs | encoder_feature(step)
            })
    }

    /// Returns the features of denoising canvases over the request's context:
    /// token denoising and the encoders of the context images.
    pub fn canvas_features(&self) -> GenerationFeatures {
        self.context_encoders()
            .fold(GenerationFeatures::TOKEN_DENOISING, |needs, step| {
                needs | encoder_feature(step)
            })
    }

    /// Returns the encoding calls of the request's context images.
    fn context_encoders(&self) -> impl Iterator<Item = ImageEncoder> + '_ {
        self.multimodal_inputs
            .images
            .iter()
            .flat_map(|image| image.encoders.iter().map(|input| input.encoder))
    }

    /// Validates input positions, image encoders, and sampling.
    ///
    /// These checks depend only on the request; checks against loaded
    /// capacity are in [`GenerationRequest::validate_resources`].
    pub fn validate(&self) -> Result<(), GenerationRequestError> {
        // Validate request-wide policy and parameter invariants before walking
        // the positioned multimodal inputs.
        if self.prompt_token_ids.is_empty() && self.multimodal_inputs.images.is_empty() {
            return Err(GenerationRequestError::EmptyContext);
        }
        if self.max_und_tokens == 0 {
            return Err(GenerationRequestError::ZeroMaxUndTokens);
        }
        self.sampling
            .validate()
            .map_err(GenerationRequestError::InvalidSampling)?;
        if self.sampling.min_tokens > self.max_und_tokens {
            return Err(GenerationRequestError::MinTokensExceedsMaximum {
                min_tokens: self.sampling.min_tokens,
                max_und_tokens: self.max_und_tokens,
            });
        }
        if self.is_migrated() {
            // Its prompt's log-probabilities were reported where it began.
            if self.sampling.return_prompt_logprobs {
                return Err(GenerationRequestError::MigratedPromptLogprobs);
            }
            // A request still running when it left had generated fewer
            // tokens than its limit allows.
            let generated = self.generated_token_ids.len();
            if generated >= self.max_und_tokens {
                return Err(GenerationRequestError::MigratedPastLimit {
                    generated,
                    max_und_tokens: self.max_und_tokens,
                });
            }
        }

        // Image positions belong to the final token vector. The input order is
        // also the order of encoder contributions when positions are equal.
        let context_tokens = self.prompt_token_ids.len();
        let mut previous_position = 0;
        for image in &self.multimodal_inputs.images {
            if image.b64.is_empty() {
                return Err(GenerationRequestError::EmptyImagePayload);
            }
            validate_image_encoders(&image.encoders, image.num_positions)?;
            // A feature-injected image is a block of a context prefill whose
            // later tokens follow it in the same call, so its cache length must
            // be known when the call is planned.
            if image.injects_features()
                && image
                    .encoders
                    .iter()
                    .any(|input| input.num_cache_tokens.is_none())
            {
                return Err(GenerationRequestError::FeatureImageCacheUnknown);
            }
            if image.position as usize > context_tokens {
                return Err(GenerationRequestError::ImagePositionBeyondPrompt {
                    position: image.position,
                    prompt_tokens: context_tokens,
                });
            }
            if image.position < previous_position {
                return Err(GenerationRequestError::UnorderedImageInputs);
            }
            previous_position = image.position;
        }

        // Empty stop strings would match every output position and do not form
        // a meaningful termination boundary.
        if self.stop_strings.iter().any(String::is_empty) {
            return Err(GenerationRequestError::EmptyStopString);
        }
        Ok(())
    }
}

/// Validates an image's encoder inputs and logical position contribution.
fn validate_image_encoders(
    inputs: &[ImageEncoderInput],
    num_positions: u32,
) -> Result<(), GenerationRequestError> {
    if inputs.is_empty() {
        return Err(GenerationRequestError::MissingImageEncoders);
    }
    if num_positions == 0 {
        return Err(GenerationRequestError::ZeroImageLogicalPositions);
    }
    for input in inputs {
        if input.num_cache_tokens == Some(0) || input.max_cache_tokens == Some(0) {
            return Err(GenerationRequestError::ZeroImageCacheBound);
        }
        if let (Some(tokens), Some(max_tokens)) = (input.num_cache_tokens, input.max_cache_tokens)
            && tokens > max_tokens
        {
            return Err(GenerationRequestError::ImageCacheExceedsCapacity { tokens, max_tokens });
        }
    }
    Ok(())
}

/// Validation failures for a scheduler-facing generation request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GenerationRequestError {
    /// The positive input contains neither tokens nor images.
    #[error("generation input must contain tokens or images")]
    EmptyContext,
    /// Understanding decode has a zero token budget.
    #[error("max_und_tokens must be positive")]
    ZeroMaxUndTokens,
    /// An image input declares no encoder stages.
    #[error("image input has no encoders")]
    MissingImageEncoders,
    /// An input image carries no encoded payload.
    #[error("image payload must not be empty")]
    EmptyImagePayload,
    /// An image refers beyond the final positive prompt.
    #[error("image position {position} exceeds the {prompt_tokens}-token prompt")]
    ImagePositionBeyondPrompt {
        /// Exclusive token position of the image marker.
        position: u32,
        /// Length of the final token vector.
        prompt_tokens: usize,
    },
    /// Images would be consumed in a different order than their prompt positions.
    #[error("input images must follow prompt-token position order")]
    UnorderedImageInputs,
    /// An image input contributes no logical model positions.
    #[error("image inputs must consume at least one logical position")]
    ZeroImageLogicalPositions,
    /// An explicit image cache length or capacity is zero.
    #[error("image KV lengths and capacities must be positive")]
    ZeroImageCacheBound,
    /// An exact image contribution must fit its declared capacity.
    #[error("image KV length {tokens} exceeds capacity {max_tokens}")]
    ImageCacheExceedsCapacity { tokens: u32, max_tokens: u32 },
    /// A feature-injected input image does not declare the exact cache length
    /// of every encoder.
    #[error(
        "an input image whose encoders inject features must declare each encoder's exact KV length"
    )]
    FeatureImageCacheUnknown,
    /// A configured stop string is empty.
    #[error("stop strings must not be empty")]
    EmptyStopString,
    /// A migrated request asks for its prompt's log-probabilities again.
    #[error("a migrated request does not report prompt log-probabilities again")]
    MigratedPromptLogprobs,
    /// A migrated request generated as many tokens as its limit allows, so it
    /// had finished.
    #[error("a migrated request generated {generated} tokens of its {max_und_tokens}-token limit")]
    MigratedPastLimit {
        /// Tokens the request generated before it migrated.
        generated: usize,
        /// The request's token limit.
        max_und_tokens: usize,
    },
    /// A request a block-diffusion runtime generates in canvases names no
    /// seed for its draws.
    #[error("a canvas-generating request requires a sampling seed")]
    UnseededCanvas,
    /// Text sampling parameters are invalid.
    #[error("invalid sampling parameters: {0}")]
    InvalidSampling(#[source] SamplingParamsError),
    /// Minimum-token floor exceeds the generation budget.
    #[error("min_tokens ({min_tokens}) exceeds max_und_tokens ({max_und_tokens})")]
    MinTokensExceedsMaximum {
        /// Required minimum generated tokens.
        min_tokens: usize,
        /// Maximum generated understanding tokens.
        max_und_tokens: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_limits() -> GenerationLimits {
        GenerationLimits {
            features: GenerationFeatures::all(),
            max_vae_grid_tokens: 64,
            max_vit_grid_tokens: 64,
            max_latent_feature_bytes: 1 << 20,
            max_vision_feature_bytes: 1 << 20,
            encoder_cache_entries: 4,
            max_logprobs: 20,
        }
    }

    /// A request with one context image read by a bounded VAE input and an
    /// exact ViT input. Its cache policy neither reads nor writes, so
    /// encoder-cache reservations are opt-in per test.
    fn complete_request() -> GenerationRequest {
        GenerationRequest {
            request_id: RequestId(7),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: MultimodalInputs {
                images: vec![ImageInput {
                    digest: [17; 32],
                    b64: "aW1hZ2U=".into(),
                    position: 2,
                    num_positions: 1,
                    encoders: vec![
                        ImageEncoderInput {
                            encoder: ImageEncoder::Latent,
                            num_cache_tokens: None,
                            max_cache_tokens: Some(64),
                            height: None,
                            width: None,
                        },
                        ImageEncoderInput {
                            encoder: ImageEncoder::Vision,
                            num_cache_tokens: Some(32),
                            max_cache_tokens: None,
                            height: None,
                            width: None,
                        },
                    ],
                }],
            },
            sampling: SamplingParams::default(),
            max_und_tokens: 16,
            include_stop_token: false,
            stop_strings: vec!["stop".into()],
            stop_token_ids: vec![2],
            priority: 3,
            cache: CachePolicy {
                read: false,
                write: false,
                isolation_key: Some(91),
                state_seeds: vec![[5; 32]],
                resume: None,
            },
            generated_token_ids: Vec::new(),
            adapter: 4,
        }
    }

    #[test]
    fn the_canonical_request_validates() {
        complete_request()
            .validate()
            .expect("valid canonical request");
    }

    #[test]
    fn a_request_reaches_understanding_and_its_context_encoders() {
        let request = complete_request();
        assert_eq!(
            request.required_features(),
            GenerationFeatures::UNDERSTANDING
                | GenerationFeatures::LATENT_ENCODE
                | GenerationFeatures::VISION_ENCODE
        );
        assert_eq!(
            request.canvas_features(),
            GenerationFeatures::TOKEN_DENOISING
                | GenerationFeatures::LATENT_ENCODE
                | GenerationFeatures::VISION_ENCODE
        );
    }

    #[test]
    fn conservative_resources_cover_every_ingest_step() {
        let mut request = complete_request();
        request.cache.read = true;
        request.cache.write = true;
        request
            .validate_resources(&runtime_limits())
            .expect("supported image input");
        // Prompt, Und budget, and the context image's VAE bound and exact
        // ViT length.
        assert_eq!(
            request.max_cache_tokens(&runtime_limits()).unwrap(),
            4 + 16 + 64 + 32
        );
        assert_eq!(request.num_encoder_cache_entries(), 2);
    }

    #[test]
    fn conservative_resources_use_each_exact_image_ingest_step() {
        // Exact encoder lengths are charged as declared even above the loaded
        // grid capacities.
        let mut request = complete_request();
        request.prompt_token_ids = vec![1; 35];
        request.multimodal_inputs.images = vec![ImageInput {
            digest: [17; 32],
            b64: "aW1hZ2U=".into(),
            position: 35,
            num_positions: 1,
            encoders: vec![
                ImageEncoderInput {
                    encoder: ImageEncoder::Latent,
                    num_cache_tokens: Some(1_026),
                    max_cache_tokens: None,
                    height: None,
                    width: None,
                },
                ImageEncoderInput {
                    encoder: ImageEncoder::Vision,
                    num_cache_tokens: Some(1_371),
                    max_cache_tokens: None,
                    height: None,
                    width: None,
                },
            ],
        }];
        request.max_und_tokens = 256;

        assert_eq!(
            request.max_cache_tokens(&runtime_limits()).unwrap(),
            35 + 256 + 1_026 + 1_371
        );
    }

    #[test]
    fn request_capacity_rejects_unbounded_encoders_and_insufficient_model_limits() {
        let mut request = complete_request();
        let input = &mut request.multimodal_inputs.images[0].encoders[1];
        input.num_cache_tokens = None;
        input.max_cache_tokens = None;
        let mut limits = runtime_limits();
        limits.max_vit_grid_tokens = 0;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::UnboundedImageCache {
                call: "vision_encoding"
            })
        );

        // The remaining cases keep the unbounded ViT input, which resolves to
        // the loaded `max_vit_grid_tokens` under the default limits.
        let mut limits = runtime_limits();
        limits.max_vision_feature_bytes = 0;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_vision_feature_bytes",
            })
        );

        let mut limits = runtime_limits();
        limits.max_latent_feature_bytes = 0;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_feature_bytes",
            })
        );

        request.cache.read = true;
        let mut limits = runtime_limits();
        limits.encoder_cache_entries = 1;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::EncoderCacheCapacity {
                requested: 2,
                available: 1,
            })
        );
        // Disabling cache use removes its reservation; encoder computation remains.
        request.cache.read = false;
        request.validate_resources(&limits).unwrap();
    }

    #[test]
    fn request_validation_rejects_invalid_image_boundaries() {
        let mut empty_ingest = complete_request();
        empty_ingest.multimodal_inputs.images[0].encoders.clear();
        assert_eq!(
            empty_ingest.validate(),
            Err(GenerationRequestError::MissingImageEncoders)
        );

        // Exact contributions and capacities are validated before admission.
        for (tokens, capacity, error) in [
            (Some(0), None, GenerationRequestError::ZeroImageCacheBound),
            (None, Some(0), GenerationRequestError::ZeroImageCacheBound),
            (
                Some(65),
                Some(64),
                GenerationRequestError::ImageCacheExceedsCapacity {
                    tokens: 65,
                    max_tokens: 64,
                },
            ),
        ] {
            let mut request = complete_request();
            let input = &mut request.multimodal_inputs.images[0].encoders[0];
            input.num_cache_tokens = tokens;
            input.max_cache_tokens = capacity;
            assert_eq!(request.validate(), Err(error));
        }

        let mut exact_capacity = complete_request();
        exact_capacity.multimodal_inputs.images[0].encoders[0].num_cache_tokens = Some(64);
        assert_eq!(exact_capacity.validate(), Ok(()));

        // An image whose encoders all inject features joins a context
        // prefill, so each encoder must declare its exact cache length; a
        // latent encoder's image may leave it to the worker.
        let mut injected = complete_request();
        injected.multimodal_inputs.images[0].encoders.truncate(1);
        injected.multimodal_inputs.images[0].encoders[0] = ImageEncoderInput {
            encoder: ImageEncoder::Vision,
            num_cache_tokens: None,
            max_cache_tokens: Some(64),
            height: None,
            width: None,
        };
        assert_eq!(
            injected.validate(),
            Err(GenerationRequestError::FeatureImageCacheUnknown)
        );
        injected.multimodal_inputs.images[0].encoders[0].num_cache_tokens = Some(64);
        assert_eq!(injected.validate(), Ok(()));

        let mut invalid_sampling = complete_request();
        invalid_sampling.sampling.top_p = 0.0;
        assert!(matches!(
            invalid_sampling.validate(),
            Err(GenerationRequestError::InvalidSampling(
                SamplingParamsError::TopP { .. }
            ))
        ));

        let mut invalid_floor = complete_request();
        invalid_floor.sampling.min_tokens = invalid_floor.max_und_tokens + 1;
        assert!(matches!(
            invalid_floor.validate(),
            Err(GenerationRequestError::MinTokensExceedsMaximum { .. })
        ));

        let mut misplaced_image = complete_request();
        misplaced_image.multimodal_inputs.images[0].position = 5;
        assert_eq!(
            misplaced_image.validate(),
            Err(GenerationRequestError::ImagePositionBeyondPrompt {
                position: 5,
                prompt_tokens: 4
            })
        );

        let mut unordered_images = complete_request();
        let mut preceding = unordered_images.multimodal_inputs.images[0].clone();
        preceding.position = 1;
        unordered_images.multimodal_inputs.images.push(preceding);
        assert_eq!(
            unordered_images.validate(),
            Err(GenerationRequestError::UnorderedImageInputs)
        );
    }

    #[test]
    fn a_migrated_request_continues_below_its_token_limit() {
        let mut request = complete_request();
        request.generated_token_ids = vec![5; 15];
        assert_eq!(request.validate(), Ok(()));
        request.generated_token_ids.push(5);
        assert_eq!(
            request.validate(),
            Err(GenerationRequestError::MigratedPastLimit {
                generated: 16,
                max_und_tokens: 16,
            })
        );
    }
}
