//! Multimodal prompt preprocessing for SenseNova and Bagel profiles.
//!
//! `InputProcessor::preprocess` in `serving::model` calls
//! [`preprocess_omni`] with the loaded profile's [`OmniProfile`], which
//! fills the model computation inputs of the `GenerationRequest`
//! (`prompt_token_ids`, `negative_prompt_token_ids`, `multimodal_inputs`,
//! `image`, and `image_generation`) and returns the request's chat output
//! processor. [`prepare_generation_resources`] then resolves sampling, cache
//! policy, and the cache budget.
//!
//! Input images arrive already resolved (`serving::media`), in
//! `ChatRequest::image_urls` order, and are carried through chat rendering as
//! request-scoped text placeholders. After the chat template renders, each
//! placeholder becomes the profile's slot text (SenseNova's image markers, or
//! nothing for Bagel), and its byte
//! offset in the cleaned text is converted into a prompt-token position by
//! tokenizing the text before it. The position becomes
//! `ImageInput::position`, the prompt-token boundary at which the image's
//! encoder output enters the context. A text prompt, which only the image
//! generation route sends, carries no input images.

mod output;

use base64::Engine as _;

use crate::profile::omni::OmniProfile;
use crate::profile::omni::resolution::{ResolutionPolicy, resolve_resolution};
use uniserve_core::{
    GenerationConstraint, GenerationRequest, ImageEncoderInput, ImageInput, ImageParams,
    MultimodalInputs,
};

use crate::profile::tokenizer::DynTokenizer;
use crate::serving::TokenizeError;
use crate::serving::chat::ChatOutputProcessor;
use crate::serving::chat::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, GenerationPromptMode, HfChatRenderer,
};
use crate::serving::input::Prompt;

pub(crate) use output::SenseNovaReasoningParser;

type OmniResult<T> = std::result::Result<T, TokenizeError>;

/// Input image whose prompt position is resolved but whose encoder inputs are not.
#[derive(Debug, Clone)]
struct RenderedImage {
    /// The resolved image. Its content hash starts the encoder-cache
    /// identity: the profile's input-image salt and, in
    /// `prepare_generation_resources`, the request's cache isolation key join
    /// it.
    image: crate::serving::ImageInput,
    /// Prompt-token position with the meaning of `ImageInput::position`.
    position: u32,
}

/// Fills a multimodal family's prompt, image inputs, and image-generation
/// parameters, and returns the request's chat output processor.
///
/// On success the request's prompt, negative prompt, multimodal inputs, image
/// parameters, and image-generation policy are replaced; on error
/// `generation` is left unmodified. A text prompt's negative prompt renders
/// the classifier-free guidance context; a chat has none.
///
/// # Errors
///
/// Fails, among other conditions, when the processor has no chat template
/// (checked for text prompts too), rendering or tokenization fails, rendering
/// loses or reorders an image slot, a chat prompt's image parts do not match
/// `images` one to one, the prompt is empty, image controls are invalid, the
/// output filter cannot be built, or the profile cannot size the image
/// encoders or the requested output resolution.
pub(super) fn preprocess_omni(
    profile: &dyn OmniProfile,
    processor: &crate::serving::InputProcessor,
    request_id: &str,
    prompt: Prompt,
    images: Vec<crate::serving::ImageInput>,
    image_params: Option<crate::serving::ImageSamplingParams>,
    generation: &mut GenerationRequest,
) -> OmniResult<ChatOutputProcessor> {
    let negative_prompt = match &prompt {
        Prompt::Text {
            negative_prompt, ..
        } => negative_prompt.clone(),
        Prompt::Chat(_) => None,
    };
    let renderer = processor
        .chat_renderer()
        .ok_or(crate::serving::chat::Error::MissingChatTemplate)?;
    let (prompt_token_ids, images) = match prompt {
        Prompt::Text { prompt, .. } => (
            profile.render_prompt_ids(&processor.tokenizer, generation.constraint, &prompt)?,
            Vec::new(),
        ),
        Prompt::Chat(chat) => render_chat(
            profile,
            &processor.tokenizer,
            renderer,
            request_id,
            chat,
            images,
            generation.constraint,
        )?,
    };
    validate_prompt(&prompt_token_ids)?;
    let output = match profile.output_filter() {
        Some(filter) => ChatOutputProcessor::new(
            Some(Box::new(
                SenseNovaReasoningParser::new(
                    filter.clone(),
                    std::sync::Arc::clone(&processor.tokenizer),
                )
                .map_err(|error| {
                    crate::serving::chat::Error::ParserInitialization {
                        kind: "reasoning",
                        name: "delimited".to_owned(),
                        error: Box::new(error),
                    }
                })?,
            )),
            None,
        ),
        None => ChatOutputProcessor::default(),
    };

    let negative_prompt_token_ids = profile.render_negative_prompt_ids(
        &processor.tokenizer,
        negative_prompt.as_deref().unwrap_or(""),
    )?;
    let (image, max_images) = resolve_image_params(
        profile.image_defaults(),
        profile.resolution_policy(),
        image_params.unwrap_or_default(),
        generation.sampling.seed,
    )?;
    let mut multimodal_inputs = prepare_image_inputs(
        prompt_token_ids.len(),
        images,
        profile.image_num_positions(),
        |width, height, count| {
            profile
                .image_encoders_for_dimensions(width, height, count)
                .map_err(TokenizeError::from)
        },
    )?;
    // An input image's encoder product may depend on the request's input
    // image count, which then joins the image's encoder-cache identity.
    if let Some(salt) = profile.input_image_cache_salt(multimodal_inputs.images.len()) {
        for image in &mut multimodal_inputs.images {
            image.hash = combine_cache_keys(image.hash, salt);
        }
    }
    let policy = profile.image_generation_for_dimensions(image.width, image.height)?;

    // Every fallible step is complete; publish the prepared inputs together.
    generation.prompt_token_ids = prompt_token_ids;
    generation.negative_prompt_token_ids = negative_prompt_token_ids;
    generation.multimodal_inputs = multimodal_inputs;
    generation.image = image;
    generation.max_images = max_images;
    generation.image_generation = policy;
    Ok(output)
}

/// Renders a multimodal chat while preserving one placeholder per input
/// image, and resolves each image's prompt position.
///
/// Inserts the profile's default system prompt when the request has no
/// system message and the constraint defines one. A non-empty assistant
/// prefix for the constraint is appended as a final assistant message that
/// the template continues instead of opening a new assistant turn. After
/// rendering, each image placeholder becomes the profile's slot text; an
/// image enters at the slot text's anchor token when the profile has one
/// (checked by tokenizing the text through the slot), and after the text
/// before the slot otherwise, which assumes that prefix encodes to a prefix
/// of the prompt's tokens.
fn render_chat(
    profile: &dyn OmniProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request_id: &str,
    mut chat: ChatRequest,
    images: Vec<crate::serving::ImageInput>,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    if !chat
        .messages
        .iter()
        .any(|message| matches!(message, ChatMessage::System { .. }))
        && let Some(system) = profile.default_system_prompt(constraint)
    {
        chat.messages.insert(0, ChatMessage::system(system));
    }
    let assistant_prefix = profile.assistant_prefix(constraint);
    chat.chat_options.generation_prompt_mode = if assistant_prefix.is_empty() {
        GenerationPromptMode::StartNewAssistant
    } else {
        chat.messages
            .push(ChatMessage::assistant_text(assistant_prefix));
        GenerationPromptMode::ContinueFinalAssistant
    };
    let placeholders = replace_chat_images(request_id, &mut chat.messages, images.len())?;
    let rendered = renderer.render(&chat)?;

    let (slot_text, anchor) = profile.image_slot();
    let (clean, byte_offsets) = replace_rendered_slots(&rendered, &placeholders, &slot_text)?;
    let prompt_ids = tokenizer.encode(&clean, false)?;
    let images = images
        .into_iter()
        .zip(byte_offsets)
        .map(|(image, byte_offset)| -> OmniResult<RenderedImage> {
            let prefix = tokenizer.encode(&clean[..byte_offset], false)?;
            let position = match anchor {
                None => prefix.len(),
                Some(anchor) => {
                    let position = prefix.len().checked_sub(1).ok_or_else(|| {
                        TokenizeError::Invalid("image slot encodes to no tokens".to_owned())
                    })?;
                    if prefix[position] != anchor {
                        return Err(TokenizeError::Invalid(
                            "image slot does not end at its configured token".to_owned(),
                        ));
                    }
                    position
                }
            };
            let position = u32::try_from(position).map_err(|_| {
                TokenizeError::Invalid("image position exceeds the token range".to_owned())
            })?;
            Ok(rendered_image(image, position))
        })
        .collect::<OmniResult<Vec<_>>>()?;
    Ok((prompt_ids, images))
}

/// Resolves sampling and checks multimodal requirements against loaded model limits.
///
/// Runs after [`preprocess_omni`] has filled the prompt and image inputs. It
/// replaces `generation.sampling` with the model's sampling defaults under
/// the request's controls, keeping its resolved seed, sets
/// `max_und_tokens`, may clear `cache.read`, and rewrites each input image's
/// `hash` into its cache-isolated form, so it must run exactly once per
/// request. On error `generation` may be partially updated; the caller
/// discards it.
///
/// # Errors
///
/// Fails when the prompt length does not fit in `u32`, the prompt fills the
/// model context while the request decodes text, sampling or stop controls
/// are invalid, `GenerationRequest::validate_resources` rejects the request
/// against the loaded limits, the cache requirement exceeds
/// `ModelConfig::max_model_len` even after shrinking the text budget, or
/// the final request fails `GenerationRequest::validate`.
pub(super) fn prepare_generation_resources(
    processor: &crate::serving::InputProcessor,
    controls: &crate::serving::SamplingParams,
    generation: &mut GenerationRequest,
) -> OmniResult<()> {
    generation.sampling = super::sampling::resolve_sampling(
        &processor.tokenizer,
        &processor.config.sampling_defaults,
        controls,
        generation.sampling.seed,
    )?;
    // Resolve the autoregressive budget only for requests whose behavior can
    // enter text decoding.
    let prompt_tokens = u32::try_from(generation.prompt_token_ids.len()).map_err(|_| {
        TokenizeError::Invalid("generation prompt exceeds the supported token count".to_owned())
    })?;
    generation.max_und_tokens = if generation.decodes_text() {
        crate::serving::sampling::resolve_max_tokens(
            controls.max_tokens,
            processor.config.sampling_defaults.max_output_tokens,
            Some(processor.config.max_model_len),
            prompt_tokens,
        )? as usize
    } else {
        0
    };

    // Sampling choices determine logprob delivery and whether prefix-cache
    // reads remain semantically valid for this request.
    let prompt_logprobs_requested = generation.sampling.prompt_logprobs_requested();
    generation.cache.read &= !prompt_logprobs_requested;
    // Encoder-cache identity folds in the request's cache isolation key, so
    // the same image in different cache partitions maps to different entries.
    for image in &mut generation.multimodal_inputs.images {
        image.hash = isolated_cache_key(image.hash, generation.cache.isolation_key);
    }

    // Resolve conservative capacity before admission. Text output may shrink to
    // fit the model context, but non-text resource requirements remain fixed.
    generation.validate_resources(&processor.limits)?;
    let mut max_cache_tokens = generation.max_cache_tokens(&processor.limits)?;
    // Shrink the text budget by the excess once; at least one text token must
    // remain. The check after this block covers requests that do not decode
    // text and so have no budget to shrink.
    if max_cache_tokens > processor.config.max_model_len as usize && generation.decodes_text() {
        let excess = max_cache_tokens - processor.config.max_model_len as usize;
        generation.max_und_tokens = generation.max_und_tokens
            .checked_sub(excess)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                TokenizeError::Invalid(format!(
                    "generation context requires at least {} KV tokens, exceeding the {}-token runtime limit",
                    max_cache_tokens.saturating_sub(generation.max_und_tokens),
                    processor.config.max_model_len
                ))
            })?;
        max_cache_tokens = generation.max_cache_tokens(&processor.limits)?;
    }

    if max_cache_tokens > processor.config.max_model_len as usize {
        return Err(TokenizeError::Invalid(format!(
            "generation requires {} KV tokens, exceeding the {}-token runtime limit",
            max_cache_tokens, processor.config.max_model_len
        )));
    }

    generation.validate()?;
    Ok(())
}

/// Merges profile defaults with request image controls and resolves generation geometry.
///
/// Request fields override profile defaults field by field. The sampling seed
/// (already resolved from the image and text seeds) overrides the profile
/// seed. Returns the diffusion parameters and the request's image budget,
/// which only the scheduler reads.
fn resolve_image_params(
    defaults: &crate::profile::omni::ImageGenerationDefaults,
    resolution_policy: &ResolutionPolicy,
    image: crate::serving::ImageSamplingParams,
    sampling_seed: Option<u64>,
) -> OmniResult<(ImageParams, u16)> {
    let resolution = resolve_resolution(
        resolution_policy,
        image.resolution,
        image.width,
        image.height,
    )?;
    let cfg_interval = image
        .cfg_interval
        .map(|interval| (interval[0], interval[1]))
        .unwrap_or(defaults.cfg_interval);
    let max_images = image.max_images.unwrap_or(defaults.max_images);
    // `GenerationRequest::validate` checks the resolved steps, guidance,
    // interval, and shift (`ImageParams::validate`) once the request is built.
    let params = ImageParams {
        steps: image.steps.unwrap_or(defaults.steps),
        cfg_text_scale: image.cfg_text_scale.unwrap_or(defaults.cfg_text_scale),
        cfg_img_scale: image.cfg_img_scale.unwrap_or(defaults.cfg_img_scale),
        cfg_renorm_type: image.cfg_renorm_type.unwrap_or(defaults.cfg_renorm_type),
        cfg_renorm_min: image.cfg_renorm_min.unwrap_or(defaults.cfg_renorm_min),
        cfg_interval,
        timestep_shift: image.timestep_shift.unwrap_or(defaults.timestep_shift),
        height: resolution.height,
        width: resolution.width,
        seed: sampling_seed.or(defaults.seed),
    };
    Ok((params, max_images))
}

/// Resolves each image's encoder inputs at its final prompt position.
///
/// `encoders_for_dimensions` receives the image's header width and height in
/// pixels and the request's total image count. The output is ordered by
/// prompt position, and each image travels to the worker as standard base64.
fn prepare_image_inputs(
    num_prompt_tokens: usize,
    images: Vec<RenderedImage>,
    num_positions: u32,
    mut encoders_for_dimensions: impl FnMut(u32, u32, usize) -> OmniResult<Vec<ImageEncoderInput>>,
) -> OmniResult<MultimodalInputs> {
    let image_count = images.len();
    let mut images = images
        .into_iter()
        .map(|RenderedImage { image, position }| {
            if position as usize > num_prompt_tokens {
                return Err(TokenizeError::Invalid(
                    "image position exceeds the tokenized prompt".into(),
                ));
            }
            let encoders = encoders_for_dimensions(image.width(), image.height(), image_count)?;
            Ok(ImageInput {
                hash: image.hash(),
                b64: base64::engine::general_purpose::STANDARD.encode(image.bytes()),
                position,
                num_positions,
                encoders,
            })
        })
        .collect::<OmniResult<Vec<_>>>()?;
    // The sort is stable, so images sharing a position keep their input order
    // as `ImageInput::position` requires.
    images.sort_by_key(|image| image.position);
    Ok(MultimodalInputs { images })
}

/// Replaces chat image parts with unique template placeholders.
///
/// Returns the placeholders in message order, which is the
/// `ChatRequest::image_urls` order of the resolved images. Assistant messages are
/// skipped, so images in assistant content are not collected.
///
/// # Errors
///
/// Fails when the messages hold a number of image parts other than
/// `image_count`, the number of resolved images.
fn replace_chat_images(
    request_id: &str,
    messages: &mut [ChatMessage],
    image_count: usize,
) -> OmniResult<Vec<String>> {
    let mut placeholders = Vec::new();
    for message in messages {
        let content = match message {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        let ChatContent::Parts(parts) = content else {
            continue;
        };
        for part in parts {
            if !matches!(part, ChatContentPart::ImageUrl { .. }) {
                continue;
            }
            let placeholder = image_placeholder(request_id, placeholders.len());
            placeholders.push(placeholder.clone());
            *part = ChatContentPart::Text { text: placeholder };
        }
    }
    if placeholders.len() != image_count {
        return Err(TokenizeError::Invalid(format!(
            "chat request has {} image parts but {image_count} resolved input images",
            placeholders.len()
        )));
    }
    Ok(placeholders)
}

/// Returns the placeholder for one image slot.
///
/// The placeholder embeds a hash of the request identifier and the slot index.
/// `replace_rendered_slots` rejects a rendered prompt in which any placeholder
/// does not occur exactly once, including when request text repeats it.
fn image_placeholder(request_id: &str, index: usize) -> String {
    format!(
        "[IMAGE_SLOT_{:016x}_{index}]",
        uniserve_core::Fnv1a::hash(request_id.as_bytes())
    )
}

/// Replaces each rendered image placeholder exactly once and returns its byte offset.
///
/// Placeholders must appear in `placeholders` order. Each returned offset
/// indexes the cleaned text just past that placeholder's `replacement`.
fn replace_rendered_slots(
    rendered: &str,
    placeholders: &[String],
    replacement: &str,
) -> OmniResult<(String, Vec<usize>)> {
    let mut clean = String::with_capacity(rendered.len());
    let mut cursor = 0;
    let mut byte_offsets = Vec::with_capacity(placeholders.len());
    for placeholder in placeholders {
        if rendered.matches(placeholder).count() != 1 {
            return Err(TokenizeError::Invalid(
                "chat rendering must preserve one slot per input image".to_string(),
            ));
        }
        let relative = rendered[cursor..].find(placeholder).ok_or_else(|| {
            TokenizeError::Invalid("chat rendering changed input-image order".to_owned())
        })?;
        let start = cursor + relative;
        clean.push_str(&rendered[cursor..start]);
        clean.push_str(replacement);
        byte_offsets.push(clean.len());
        cursor = start + placeholder.len();
    }
    clean.push_str(&rendered[cursor..]);
    Ok((clean, byte_offsets))
}

/// Pairs a resolved image with its prompt position.
fn rendered_image(image: crate::serving::ImageInput, position: u32) -> RenderedImage {
    RenderedImage { image, position }
}

/// Rejects an empty prompt token sequence.
fn validate_prompt(prompt_ids: &[u32]) -> OmniResult<()> {
    if prompt_ids.is_empty() {
        Err(TokenizeError::Invalid(
            "generation requires a non-empty prompt".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Builds a cache key isolated by request namespace and salt.
///
/// Returns `content_key` unchanged when the request has no isolation key, so
/// unisolated requests share encoder-cache entries for identical image
/// payloads.
fn isolated_cache_key(content_key: u64, isolation_key: Option<u64>) -> u64 {
    isolation_key.map_or(content_key, |isolation_key| {
        combine_cache_keys(content_key, isolation_key)
    })
}

/// Mixes a value that changes an encoder product into its cache key.
fn combine_cache_keys(key: u64, value: u64) -> u64 {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&key.to_le_bytes());
    bytes[8..].copy_from_slice(&value.to_le_bytes());
    uniserve_core::Fnv1a::hash(&bytes)
}
