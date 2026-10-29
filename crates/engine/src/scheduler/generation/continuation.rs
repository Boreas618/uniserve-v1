//! How a migrated image-generating request rebuilds its context.
//!
//! A token request that generates images leaves an engine as migratable like
//! any token request (`TerminalIntent::Migrate`), and the daemon submits it
//! again with its facts: the tokens it generated, every one it counted, and
//! the images it generated, each with what its feedback encoders read
//! (`GenerationRequest::generated_images`). Its context is derived state:
//! the engine that continues it rebuilds it from those facts by the
//! computation that wrote it.
//!
//! The context of such a request is not its prompt followed by its tokens.
//! Its calls wrote, in order:
//!
//! - the prompt, with its input images, after which the prefill samples the
//!   first token, or, when the prompt ends with an image, a decode of BOS;
//! - one token per decode: the decode's input, the request's pending token
//!   (`RequestState::next_token`), after which it samples the next;
//! - per image, the pending token again as the close of the cache the image
//!   is conditioned on (`GenerationPhase::CloseCache`, which does not
//!   advance the logical position), then each feedback encoder's feature
//!   (`GenerationPhase::FeedbackState`), which advance it by the image's
//!   feedback positions; the last samples the next token when the model
//!   samples its feedback continuation, and otherwise the configured
//!   continuation token becomes the pending token.
//!
//! A sampled token decides what follows as output resolution decides it
//! (`Scheduler::resolve`): a direct image trigger opens an image without
//! becoming the pending token, a round-close token is written by one more
//! decode whose sample decides the round, and any other token becomes the
//! pending token and may complete a trigger sequence. [`plan`] runs that
//! decision over the request's facts and returns what the continuation
//! writes: its prompt extended by every token a decode wrote, the images at
//! their offsets in that extension with the token each image's close
//! wrote, and the state the request resumes in. The continuation writes
//! them by the same calls, sampling nothing until it resumes: a decode of
//! its pending token, the feedback write of its last image that samples,
//! or the opening of the image it had begun. That image continues from the
//! step of its trajectory the preempting engine stored in the host latent
//! pools (`GenerationRequest::trajectory`, `crate::trajectories`) when they
//! hold it, conditioned on the context the continuation rebuilt, which is
//! the one it was conditioned on; it starts from noise otherwise
//! (`Scheduler::image_restore`).

use std::collections::VecDeque;

use crate::{FeedbackImage, FeedbackNextToken, GenerationRequest};

use super::super::SpecialTokenIds;

/// One generated image a continuation writes into its context.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReplayImage {
    /// The prompt tokens, the request's own and those the continuation
    /// extends it with, written before the image.
    pub(crate) offset: u32,
    /// The token the image's cache close wrote before it: the request's
    /// pending token when the image opened.
    pub(crate) close_token: u32,
    /// The image's ordinal among the request's images, from one.
    pub(crate) image_index: u32,
    /// What the image's feedback encoders read.
    pub(crate) image: FeedbackImage,
    /// The rebuild's next call for the image.
    pub(crate) stage: ReplayStage,
}

/// The next call that writes a generated image into a continuation's
/// context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayStage {
    /// The cache close before the image.
    Close,
    /// The encoding of the image by its feedback encoder of this index.
    Encode(usize),
    /// The write of that encoder's feature.
    Write(usize),
}

/// What a continuation does once its context is rebuilt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Resume {
    /// It decodes its pending `token`; with `round_closing`, the token
    /// closed a round, which the decode's completion decides.
    Decode { token: u32, round_closing: bool },
    /// The feature write of its last image samples its next token.
    FeedbackSample,
    /// It opens the image it had begun, closing its cache with its pending
    /// `token`.
    Open { token: u32 },
}

/// What a migrated image-generating request writes into its context before
/// it generates again.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Continuation {
    /// The tokens its decodes wrote after its prompt, in order, which extend
    /// its prompt.
    pub(crate) written: Vec<u32>,
    /// Its images, in order.
    pub(crate) images: VecDeque<ReplayImage>,
    /// The tokens its penalties, bad words and trigger sequences read
    /// (`RequestState::generated_token_ids`).
    pub(crate) history: Vec<u32>,
    /// The tokens since its last round boundary
    /// (`RequestState::round_token_ids`).
    pub(crate) round: Vec<u32>,
    /// What it does once its context is rebuilt.
    pub(crate) resume: Resume,
}

/// Why a migrated request's facts describe no request its engine served.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ContinuationError {
    /// An image does not open where the request's tokens open one.
    #[error("generated image {image} does not open where the request's tokens open an image")]
    MisplacedImage { image: usize },
    /// Tokens follow an image the request did not finish.
    #[error("generated tokens follow an image the request did not finish")]
    TokensAfterOpenImage,
    /// A round-close token is followed by tokens though its round opens no
    /// image, which would have finished the request.
    #[error("generated tokens follow a round that finished the request")]
    TokensAfterClosedRound,
    /// Images feed back without a continuation token.
    #[error("the request's image feedback declares no continuation token")]
    MissingFeedbackToken,
    /// The request generated nothing: it is no continuation.
    #[error("a continuation carries no generated token")]
    NothingGenerated,
}

/// The call whose sample the next fact answers.
#[derive(Clone, Copy)]
enum Sample {
    /// The prefill that completes the prompt.
    Prefill,
    /// A decode that writes `input`.
    Decode { input: u32, round_closing: bool },
    /// The feature write of an image's last feedback encoder.
    Feedback,
}

/// What the request does next while its facts are read.
enum Next {
    Sample(Sample),
    /// It opens an image, its pending token closing the cache.
    Open,
}

/// The continuation of `request`, a migrated request that generates images,
/// from its facts: its prompt, its generated tokens and its images
/// (module documentation). `special` names the control tokens its feedback
/// continuation may be.
///
/// # Errors
///
/// Fails when its facts describe no request the engine served: an image
/// that does not open where its tokens open one, or tokens after an image or
/// a round that ended it.
pub(crate) fn plan(
    request: &GenerationRequest,
    special: &SpecialTokenIds,
) -> Result<Continuation, ContinuationError> {
    let config = &request.image_generation;
    let trigger = &config.trigger;
    let max_images = usize::from(request.max_images);
    let mut images = request.generated_images.iter().enumerate().peekable();
    let mut tokens = request.generated_token_ids.iter().copied();

    let mut continuation = Continuation {
        written: Vec::new(),
        images: VecDeque::new(),
        history: Vec::new(),
        round: Vec::new(),
        resume: Resume::Open { token: 0 },
    };
    // `RequestState::next_token` starts at zero, and a prompt that ends with
    // an image block samples nothing: decode starts from BOS.
    let mut pending = 0u32;
    let mut counted = 0usize;
    let mut completed = 0usize;
    let ends_with_image = request
        .multimodal_inputs
        .images
        .last()
        .is_some_and(|image| image.position as usize == request.prompt_token_ids.len());
    let mut next = if ends_with_image {
        pending = special.bos;
        Next::Sample(Sample::Decode {
            input: pending,
            round_closing: false,
        })
    } else {
        Next::Sample(Sample::Prefill)
    };

    loop {
        let can_open = request.generates_images() && completed < max_images;
        match next {
            Next::Open => {
                let Some((index, image)) = images.peek() else {
                    // The image it had begun opens again once it resumes,
                    // which records its trigger then.
                    if tokens.next().is_some() {
                        return Err(ContinuationError::TokensAfterOpenImage);
                    }
                    continuation.resume = Resume::Open { token: pending };
                    return Ok(continuation);
                };
                if image.tokens as usize != counted {
                    return Err(ContinuationError::MisplacedImage { image: *index });
                }
                // `Scheduler::record_gen_trigger_for_replay`, as the image
                // opened.
                if let Some(token) = trigger.direct_token()
                    && request.continues_after_image()
                    && counted > continuation.history.len()
                    && continuation.history.last() != Some(&token)
                {
                    continuation.history.push(token);
                }
                continuation.images.push_back(ReplayImage {
                    offset: u32::try_from(
                        request.prompt_token_ids.len() + continuation.written.len(),
                    )
                    .unwrap_or(u32::MAX),
                    close_token: pending,
                    image_index: u32::try_from(index + 1).unwrap_or(u32::MAX),
                    image: image.image.clone(),
                    stage: ReplayStage::Close,
                });
                images.next();
                completed += 1;
                continuation.round.clear();
                next = if config.sample_feedback_continuation {
                    Next::Sample(Sample::Feedback)
                } else {
                    pending = match config.feedback_next_token {
                        FeedbackNextToken::None => {
                            return Err(ContinuationError::MissingFeedbackToken);
                        }
                        FeedbackNextToken::Bos => special.bos,
                        FeedbackNextToken::EndOfImage => special.end_of_image,
                        FeedbackNextToken::Token { token_id } => token_id,
                    };
                    Next::Sample(Sample::Decode {
                        input: pending,
                        round_closing: false,
                    })
                };
            }
            Next::Sample(sample) => {
                // A round-close token's decode writes it and decides the
                // round; its sample counts for nothing.
                if let Sample::Decode {
                    input,
                    round_closing: true,
                } = sample
                {
                    let opens = trigger.matches_round_close(&continuation.round, input)
                        && can_open
                        && counted < request.max_und_tokens;
                    if images.peek().is_none() && tokens.clone().next().is_none() {
                        continuation.resume = Resume::Decode {
                            token: input,
                            round_closing: true,
                        };
                        return Ok(continuation);
                    }
                    if !opens {
                        return Err(ContinuationError::TokensAfterClosedRound);
                    }
                    continuation.written.push(input);
                    next = Next::Open;
                    continue;
                }
                let Some(token) = tokens.next() else {
                    if let Some((index, _)) = images.peek() {
                        return Err(ContinuationError::MisplacedImage { image: *index });
                    }
                    continuation.resume = match sample {
                        Sample::Decode {
                            input,
                            round_closing,
                        } => Resume::Decode {
                            token: input,
                            round_closing,
                        },
                        Sample::Feedback => Resume::FeedbackSample,
                        Sample::Prefill => return Err(ContinuationError::NothingGenerated),
                    };
                    return Ok(continuation);
                };
                if let Sample::Decode { input, .. } = sample {
                    continuation.written.push(input);
                }
                counted += 1;
                next = match sample {
                    // The prompt itself may open an image, and an image-only
                    // request opens one after its prompt: either discards
                    // the prefill's sample.
                    Sample::Prefill
                        if can_open
                            && (trigger.matches_generated(&request.prompt_token_ids)
                                || request.starts_with_image()) =>
                    {
                        Next::Open
                    }
                    Sample::Decode { .. } if trigger.round_close_token_ids().contains(&token) => {
                        pending = token;
                        Next::Sample(Sample::Decode {
                            input: token,
                            round_closing: true,
                        })
                    }
                    _ if can_open && trigger.direct_token() == Some(token) => Next::Open,
                    _ => {
                        continuation.history.push(token);
                        pending = token;
                        // The prompt's sample does not join the round.
                        if !matches!(sample, Sample::Prefill) {
                            continuation.round.push(token);
                        }
                        if can_open && trigger.matches_generated(&continuation.history) {
                            Next::Open
                        } else {
                            Next::Sample(Sample::Decode {
                                input: token,
                                round_closing: false,
                            })
                        }
                    }
                };
            }
        }
    }
}
