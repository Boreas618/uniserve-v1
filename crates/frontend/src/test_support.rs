//! Fixtures shared by the crate's tests.
//!
//! `configured_tokenizer` builds the deterministic byte tokenizer the serving
//! tests decode with: every printable ASCII character is one token whose ID is
//! its code point, and the chat, vision and thinking special tokens encode
//! atomically. It is compiled for the crate's own tests and, through the
//! `testing` feature, for the tests of the crates that serve over it.

#![expect(
    clippy::expect_used,
    reason = "a fixture that cannot be built is the test's failure report"
)]

use std::sync::Arc;

use tokenizers::AddedToken;
use tokenizers::Tokenizer as TokenizerBuilder;
use tokenizers::models::bpe::{BPE, Vocab};

use crate::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

/// Loads a `tokenizers` definition in storage, as the frontend loads a
/// checkpoint's `tokenizer.json`.
pub fn load_tokenizer(definition: &TokenizerBuilder) -> HuggingFaceTokenizer {
    let json = definition.to_string(false).expect("serialize tokenizer");
    HuggingFaceTokenizer::from_json(serde_json::from_str(&json).expect("parse tokenizer"))
        .expect("load tokenizer")
}

/// Builds the deterministic byte-oriented tokenizer the serving tests use.
///
/// Every printable ASCII character encodes to the token whose ID equals its
/// code point, so tests write IDs as `b'a' as u32`. ID 0 is the `<unk>` token,
/// and IDs 1 through 6 become the chat, vision, and thinking special tokens.
/// The BPE model has no merges, so encoding is one token per character.
pub fn configured_tokenizer() -> DynTokenizer {
    let mut vocab = (0_u32..=127)
        .map(|id| {
            let token = if id == 0 {
                "<unk>".to_string()
            } else {
                char::from_u32(id).expect("ASCII code point").to_string()
            };
            (token, id)
        })
        .collect::<Vocab>();
    for (id, token) in [
        (1, "<|im_start|>"),
        (2, "<|im_end|>"),
        (3, "<|vision_start|>"),
        (4, "<|vision_end|>"),
        (5, "<think>"),
        (6, "</think>"),
        (97, "a"),
        (98, "b"),
    ] {
        vocab.retain(|_, existing_id| *existing_id != id);
        vocab.insert(token.to_string(), id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .expect("build configured tokenizer model");
    let mut tokenizer = TokenizerBuilder::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|im_start|>", true),
        AddedToken::from("<|im_end|>", true),
        AddedToken::from("<|vision_start|>", true),
        AddedToken::from("<|vision_end|>", true),
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
    ]);
    Arc::new(load_tokenizer(&tokenizer))
}
