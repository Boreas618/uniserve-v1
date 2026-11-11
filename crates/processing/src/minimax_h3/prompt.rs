//! The conditioner prompt: the token ids the Qwen3-VL conditioner reads and the rows its vision
//! blocks occupy.
//!
//! No chat template is applied. `fl2va` labels each keyframe `"<Picture i>: "` before its vision
//! block. `ref2va` labels its references in request order, numbered per modality: a soundtrack
//! `"<Audio j>: "` (a video's soundtrack before its `"<Video k>: "` label), an image
//! `"<Picture i>: "` before its vision block, and a video one `"<{t:.1f} seconds>"` label before
//! each of its vision blocks. Audio contributes labels only and `ref2va` keyframes contribute
//! nothing. The prompt follows verbatim. Every text segment is tokenized on its own without special
//! tokens and the pieces are concatenated.
//!
//! A vision block is `<|vision_start|>`, one `<|image_pad|>` or `<|video_pad|>` per vision token,
//! and `<|vision_end|>`. A block's rows, markers included, are a vision span of the prompt, which
//! the denoiser modulates as video rows; every other row is text.

use std::borrow::Cow;

use super::rules::{Generation, PROMPT, Preparation, Vision};
use crate::video::InputError;
use crate::vocabulary::Vocabulary;

const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
const IMAGE_PAD: &str = "<|image_pad|>";
const VIDEO_PAD: &str = "<|video_pad|>";

/// One piece of the prompt, before tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment<'a> {
    /// Text, tokenized on its own.
    Text(Cow<'a, str>),
    /// An image block of this many `<|image_pad|>` tokens between the markers.
    Image(u32),
    /// A video block of this many `<|video_pad|>` tokens between the markers.
    Video(u32),
}

/// The tokenized prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// Token ids, labels and vision blocks first, the request's prompt last.
    pub token_ids: Vec<u32>,
    /// The rows each vision block occupies, markers included, as `(first row, rows)`.
    pub vision_spans: Vec<(u32, u32)>,
}

/// Lists the prompt of a generation, the request's own prompt last.
pub fn segments<'a>(generation: &Generation, prompt: &'a str) -> Vec<Segment<'a>> {
    let mut segments = Vec::new();
    let (mut pictures, mut videos, mut audios) = (0, 0, 0);
    let text = |value: String| Segment::Text(Cow::Owned(value));
    for condition in &generation.conditions {
        match (&condition.preparation, &condition.vision) {
            (Preparation::Audio(_), _) => {
                audios += 1;
                segments.push(text(format!("<Audio {audios}>: ")));
            }
            (
                Preparation::Video(clip),
                Some(Vision::Video {
                    block_tokens,
                    block_timestamps,
                    ..
                }),
            ) => {
                if clip.soundtrack.is_some() {
                    audios += 1;
                    segments.push(text(format!("<Audio {audios}>: ")));
                }
                videos += 1;
                segments.push(text(format!("<Video {videos}>: ")));
                for timestamp in block_timestamps {
                    // Like Python's `{:.1f}`, this rounds the exact value half to even: a
                    // 0.25-second mean renders as "0.2".
                    segments.push(text(format!("<{timestamp:.1} seconds>")));
                    segments.push(Segment::Video(*block_tokens));
                }
            }
            (_, Some(Vision::Image { tokens, .. })) => {
                pictures += 1;
                segments.push(text(format!("<Picture {pictures}>: ")));
                segments.push(Segment::Image(*tokens));
            }
            // Keyframes of a ref2va request are not presented.
            _ => {}
        }
    }
    segments.push(Segment::Text(Cow::Borrowed(prompt)));
    segments
}

/// Tokenizes the prompt of a generation with the checkpoint's `tokenizer`, whose vocabulary holds
/// the special tokens its configuration declares.
///
/// # Errors
///
/// Refuses `prompt` when it is blank or contains a vision placeholder or marker, which would
/// misalign the conditioner's vision inputs, and fails internally when the tokenizer fails or lacks
/// the vision tokens a vision block needs.
pub fn present(
    tokenizer: &dyn Vocabulary,
    generation: &Generation,
    prompt: &str,
) -> Result<Prompt, InputError> {
    if prompt.trim().is_empty() {
        return Err(InputError::invalid(PROMPT, "must not be blank"));
    }
    // A vision block needs the vision vocabulary; a prompt can only produce the vision tokens the
    // tokenizer has, which it must not contain.
    let token = |name: &'static str| (name, tokenizer.token_id(name));
    let required = |(name, id): (&str, Option<u32>)| {
        id.ok_or_else(|| InputError::Internal(format!("the tokenizer has no {name} token")))
    };
    let (start, end) = (token(VISION_START), token(VISION_END));
    let (image_pad, video_pad) = (token(IMAGE_PAD), token(VIDEO_PAD));
    let markers: Vec<u32> = [start, end, image_pad, video_pad]
        .iter()
        .filter_map(|(_, id)| *id)
        .collect();

    let segments = segments(generation, prompt);
    let last = segments.len() - 1;
    let mut token_ids: Vec<u32> = Vec::new();
    let mut vision_spans = Vec::new();
    for (position, segment) in segments.iter().enumerate() {
        let (pad, tokens) = match segment {
            Segment::Image(tokens) => (image_pad, *tokens),
            Segment::Video(tokens) => (video_pad, *tokens),
            Segment::Text(text) => {
                let ids = tokenizer.encode(text).map_err(|error| {
                    InputError::Internal(format!("tokenizing the prompt failed: {error}"))
                })?;
                if position == last && ids.iter().any(|id| markers.contains(id)) {
                    return Err(InputError::invalid(
                        PROMPT,
                        "must not contain vision placeholder tokens",
                    ));
                }
                token_ids.extend(ids);
                continue;
            }
        };
        let first = u32::try_from(token_ids.len())
            .map_err(|_| InputError::Internal("a prompt longer than u32 rows".to_owned()))?;
        token_ids.push(required(start)?);
        token_ids.extend(std::iter::repeat_n(required(pad)?, tokens as usize));
        token_ids.push(required(end)?);
        vision_spans.push((first, tokens + 2));
    }
    Ok(Prompt {
        token_ids,
        vision_spans,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;

    use serde_json::Value;

    use super::super::rules::tests::Request;
    use super::super::vectors;
    use super::{Segment, present, segments};
    use crate::vocabulary::{Vocabulary, VocabularyError};

    /// The ids of the vision markers and placeholders in [`CharacterTokenizer`]'s vocabulary.
    const VISION_TOKENS: [(&str, u32); 4] = [
        ("<|vision_start|>", 128),
        ("<|vision_end|>", 129),
        ("<|image_pad|>", 130),
        ("<|video_pad|>", 131),
    ];

    /// A deterministic tokenizer: one token per character, its code point, plus the vision markers
    /// and placeholders as special tokens.
    pub(in crate::minimax_h3) struct CharacterTokenizer;

    impl Vocabulary for CharacterTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, VocabularyError> {
            let mut ids = Vec::new();
            let mut rest = text;
            'text: while let Some(first) = rest.chars().next() {
                for (token, id) in VISION_TOKENS {
                    if let Some(after) = rest.strip_prefix(token) {
                        ids.push(id);
                        rest = after;
                        continue 'text;
                    }
                }
                ids.push(u32::from(first));
                rest = &rest[first.len_utf8()..];
            }
            Ok(ids)
        }

        fn token_id(&self, token: &str) -> Option<u32> {
            VISION_TOKENS
                .iter()
                .find(|(name, _)| *name == token)
                .map(|&(_, id)| id)
        }
    }

    /// The checkpoint tokenizer's recorded output on the texts of the vectors' prompts, and its
    /// recorded vision token ids.
    struct RecordedTokenizer {
        tokens: HashMap<String, u32>,
        texts: HashMap<String, Vec<u32>>,
    }

    impl RecordedTokenizer {
        fn new(vectors: &Value) -> Self {
            let tokens = vectors["tokens"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, id)| (name.clone(), id.as_u64().unwrap() as u32))
                .collect();
            let mut texts = HashMap::new();
            for case in vectors["requests"].as_array().unwrap() {
                let Some(expected) = case.get("expected") else {
                    continue;
                };
                for segment in expected["segments"].as_array().unwrap() {
                    if let Some(text) = segment.get("text") {
                        let ids = segment["token_ids"].as_array().unwrap();
                        texts.insert(
                            text.as_str().unwrap().to_owned(),
                            ids.iter().map(|id| id.as_u64().unwrap() as u32).collect(),
                        );
                    }
                }
            }
            Self { tokens, texts }
        }
    }

    impl Vocabulary for RecordedTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, VocabularyError> {
            self.texts
                .get(text)
                .cloned()
                .ok_or_else(|| VocabularyError(format!("no recorded tokens for {text:?}").into()))
        }

        fn token_id(&self, token: &str) -> Option<u32> {
            self.tokens.get(token).copied()
        }
    }

    fn expected_segments(case: &Value) -> Vec<Segment<'static>> {
        case["expected"]["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|segment| match segment.get("vision") {
                Some(pad) => {
                    let tokens = segment["tokens"].as_u64().unwrap() as u32;
                    if pad == "image" {
                        Segment::Image(tokens)
                    } else {
                        Segment::Video(tokens)
                    }
                }
                None => Segment::Text(Cow::Owned(segment["text"].as_str().unwrap().to_owned())),
            })
            .collect()
    }

    /// The expected token ids and vision spans of a vectors request, rebuilt from its segments and
    /// the recorded token ids of the checkpoint tokenizer.
    fn expected_prompt(vectors: &Value, case: &Value) -> (Vec<u32>, Vec<(u32, u32)>) {
        let token = |name: &str| vectors["tokens"][name].as_u64().unwrap() as u32;
        let mut token_ids = Vec::new();
        let mut spans = Vec::new();
        for segment in case["expected"]["segments"].as_array().unwrap() {
            if let Some(pad) = segment.get("vision") {
                let pad = token(&format!("<|{}_pad|>", pad.as_str().unwrap()));
                let count = segment["tokens"].as_u64().unwrap() as usize;
                spans.push((token_ids.len() as u32, count as u32 + 2));
                token_ids.push(token("<|vision_start|>"));
                token_ids.extend(std::iter::repeat_n(pad, count));
                token_ids.push(token("<|vision_end|>"));
            } else {
                let ids = segment["token_ids"].as_array().unwrap();
                token_ids.extend(ids.iter().map(|id| id.as_u64().unwrap() as u32));
            }
        }
        (token_ids, spans)
    }

    /// Requests present the reference's labels, timestamps and vision blocks in the reference's
    /// order.
    #[test]
    fn segments_match_the_reference() {
        for case in vectors()["requests"].as_array().unwrap() {
            if case.get("expected").is_none() {
                continue;
            }
            let generation = Request::parse(case).prepare().unwrap();
            let prompt = case["prompt"].as_str().unwrap();
            assert_eq!(
                segments(&generation, prompt),
                expected_segments(case),
                "{}",
                case["name"]
            );
        }
    }

    /// With the checkpoint tokenizer's recorded output, the prompt's token ids and vision spans
    /// equal the reference's: labels and the prompt are tokenized segment by segment, and every
    /// vision block, markers included, is one span.
    #[test]
    fn prompts_match_the_reference() {
        let vectors = vectors();
        let tokenizer = RecordedTokenizer::new(&vectors);
        for case in vectors["requests"].as_array().unwrap() {
            if case.get("expected").is_none() {
                continue;
            }
            let generation = Request::parse(case).prepare().unwrap();
            let prompt = present(&tokenizer, &generation, case["prompt"].as_str().unwrap())
                .unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
            let (token_ids, spans) = expected_prompt(&vectors, case);
            assert_eq!(prompt.token_ids, token_ids, "{}", case["name"]);
            assert_eq!(prompt.vision_spans, spans, "{}", case["name"]);
        }
    }

    /// A prompt that is empty or carries vision tokens is refused.
    #[test]
    fn prompts_must_be_plain_text() {
        let generation = Request::parse(&vectors()["requests"][0]).prepare().unwrap();
        for prompt in ["", " \n", "a <|image_pad|> b", "<|vision_start|>"] {
            let error = present(&CharacterTokenizer, &generation, prompt).unwrap_err();
            assert!(
                matches!(&error, crate::InputError::Invalid { param, .. } if param == "prompt"),
                "{prompt:?}: {error}"
            );
        }
    }
}
