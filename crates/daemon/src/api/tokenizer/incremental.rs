//! Stateful incremental decoding with stable-prefix emission.
//!
//! Each generated token advances `fastokens`' decode stream ([`fastokens::decode_stream_step`]),
//! which decodes a window of tokens and emits only the text past the window's previous decode, so
//! a UTF-8 character split across tokens appears once its last byte arrives; until then the
//! decode ends in U+FFFD and the stream emits nothing. The stream's state is held here rather
//! than in `fastokens::DecodeStream`, which keeps it private, so a stream can be handed to a
//! successor process ([`IncrementalDecoder::state`]). Emitted text accumulates in `output`, where
//! the serving output stages match stop strings, and `next_chunk` streams it while withholding a
//! trailing `min_bytes_to_buffer` bytes so a matched stop string can still be truncated by
//! `flush`.

use std::mem::take;
use std::sync::Arc;

use proto::handoff::IncrementalDecode;

use crate::api::tokenizer::{HuggingFaceTokenizer, Result};

/// The replacement character a decode ends in while its last character is incomplete.
const REPLACEMENT: char = '\u{FFFD}';

/// Stateful incremental decoder built on the configured Hugging Face tokenizer with prefix-diffing.
///
/// One decoder serves one generation stream: `push_token` for every generated
/// token, `next_chunk` for streamed text, and `flush` once at the end of the
/// stream or at a stop-string match.
pub struct IncrementalDecoder {
    /// The tokenizer the stream decodes with; owned, so a decoder (and the
    /// stream state it carries) outlives the borrow of whoever started it and
    /// can cross to a successor process.
    tokenizer: Arc<HuggingFaceTokenizer>,
    skip_special_tokens: bool,
    min_bytes_to_buffer: usize,
    /// The stream's token window. It starts as the prompt, decode context
    /// that is never output; each emit drops the tokens before
    /// `prefix_index`, keeping the tokens that produced the emitted text as
    /// left context for the next decode.
    ids: Vec<u32>,
    /// Decode of the window's first `prefix_index` tokens at the last emit or
    /// seed (without a trailing U+FFFD when the prompt ends in an incomplete
    /// character). Text past `prefix.len()` in a later window decode is new.
    prefix: String,
    /// Number of leading window tokens the next emit drops.
    prefix_index: usize,
    /// All text emitted by `push_token`, returned by `output`.
    cumulative_output: String,
    /// Byte offset in `cumulative_output` up to which `next_chunk` has
    /// returned text.
    output_index: usize,
    /// Whether the prompt seeded the window, which the first `push_token`
    /// does: so also whether any generated token entered the window, which
    /// `flush` needs to avoid emitting the prompt as output.
    prompt_seeded: bool,
}

impl IncrementalDecoder {
    /// Creates an incremental decoder for one generation stream.
    ///
    /// Construction does not decode. The prompt is seeded as decode context on
    /// the first `push_token`, which reports any decode error from seeding.
    pub fn new(
        tokenizer: Arc<HuggingFaceTokenizer>,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            min_bytes_to_buffer,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
            cumulative_output: String::new(),
            output_index: 0,
            prompt_seeded: false,
        }
    }

    /// Seeds the stream with the prompt: its whole decode is the prefix generated text extends.
    ///
    /// A prompt whose decode ends inside a multi-byte character keeps every token in the window
    /// (`prefix_index` 0) with the incomplete character stripped from the prefix, so the
    /// character completes with the generated tokens. `fastokens`' own seeding leaves such a
    /// prefix empty, which would emit the whole prompt as the first generated text.
    fn seed(&mut self) -> Result<()> {
        let decoded = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        if decoded.ends_with(REPLACEMENT) {
            self.prefix = decoded.trim_end_matches(REPLACEMENT).to_owned();
            self.prefix_index = 0;
        } else {
            self.prefix = decoded;
            self.prefix_index = self.ids.len();
        }
        Ok(())
    }

    /// Pushes one generated token and returns how many bytes it appended to
    /// `output`.
    ///
    /// Returns 0 when nothing is appended: while the window's decode adds no
    /// text, or ends in an incomplete character, the stream holds the token,
    /// and `flush` emits whatever it still holds. The first call also seeds the
    /// prompt context. Errors are the tokenizer's decode failures, and a decode
    /// that no longer begins with the text already emitted, which `fastokens`
    /// reports as a broken stream; an ID outside the vocabulary decodes to no
    /// text.
    pub fn push_token(&mut self, token_id: u32) -> Result<usize> {
        if !self.prompt_seeded {
            self.prompt_seeded = true;
            if !self.ids.is_empty() {
                self.seed()?;
            }
        }

        let emitted = fastokens::decode_stream_step(
            self.tokenizer.inner(),
            vec![token_id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| tokenizer_error!("decoding the stream failed: {error}"))?;
        let Some(text) = emitted else {
            return Ok(0);
        };
        self.cumulative_output.push_str(&text);
        Ok(text.len())
    }

    /// Returns emitted text that no earlier call has returned, excluding the
    /// trailing `min_bytes_to_buffer` bytes of `output`.
    ///
    /// The cutoff is rounded down to a character boundary, so at least
    /// `min_bytes_to_buffer` bytes stay withheld (all of `output` when it is
    /// shorter). Returns `None` when no text
    /// before the cutoff is pending. This does not decode.
    pub fn next_chunk(&mut self) -> Option<String> {
        let cutoff = self
            .cumulative_output
            .len()
            .saturating_sub(self.min_bytes_to_buffer);
        // Ensure we split at a utf-8 char boundary.
        let cutoff = self.cumulative_output.floor_char_boundary(cutoff);
        (cutoff > self.output_index).then(|| {
            let chunk = self.cumulative_output[self.output_index..cutoff].to_string();
            self.output_index = cutoff;
            chunk
        })
    }

    /// Flushes buffered tokens and optionally truncates final output bytes.
    ///
    /// Decodes the generated tokens the window still holds (including a
    /// trailing U+FFFD) into the output, then truncates the output to
    /// `truncate_output_to` bytes when given. The offset must lie on a
    /// character boundary, as a stop-string match offset does, or
    /// `String::truncate` panics. Before the first `push_token` the window
    /// holds only the unseeded prompt, which is decode context rather than
    /// output, so a stream that generated no token flushes no text.
    ///
    /// Returns the text not yet returned by `next_chunk` (`None` if there is
    /// none) and the complete output. The window and output are emptied, so
    /// callers flush once per stream.
    pub fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)> {
        if !self.prompt_seeded {
            self.ids.clear();
        } else if !self.ids.is_empty() {
            let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            let prefix_len = self.prefix.len();
            self.ids.clear();
            self.prefix.clear();
            self.prefix_index = 0;
            // Ensure we split at a utf-8 char boundary.
            self.cumulative_output
                .push_str(&string[string.floor_char_boundary(prefix_len)..]);
        }

        if let Some(truncate_output_to) = truncate_output_to {
            self.cumulative_output.truncate(truncate_output_to);
        }
        let last_chunk = (self.output_index < self.cumulative_output.len())
            .then(|| self.cumulative_output[self.output_index..].to_string());
        self.output_index = 0;
        Ok((last_chunk, take(&mut self.cumulative_output)))
    }

    /// Returns all text emitted by `push_token` so far, including the bytes
    /// that `next_chunk` still withholds.
    pub fn output(&self) -> &str {
        &self.cumulative_output
    }

    /// Returns the stream's place, so that another process continues this
    /// request's decoding where this one stopped.
    ///
    /// Every value is the decoder's own state: the decode window, the text it
    /// has emitted, the offsets already returned, and the two buffering
    /// settings the stream was started with. The tokenizer is not part of it;
    /// [`IncrementalDecoder::resume`] takes the one the state was taken on.
    pub fn state(&self) -> IncrementalDecode {
        IncrementalDecode {
            ids: self.ids.clone(),
            prefix: self.prefix.clone(),
            prefix_index: self.prefix_index as u64,
            cumulative_output: self.cumulative_output.clone(),
            output_index: self.output_index as u64,
            prompt_seeded: self.prompt_seeded,
            skip_special_tokens: self.skip_special_tokens,
            min_bytes_to_buffer: self.min_bytes_to_buffer as u64,
        }
    }

    /// Continues the stream `state` stopped, on `tokenizer`, which is the one
    /// the state was taken on.
    pub fn resume(state: &IncrementalDecode, tokenizer: Arc<HuggingFaceTokenizer>) -> Self {
        Self {
            tokenizer,
            skip_special_tokens: state.skip_special_tokens,
            min_bytes_to_buffer: state.min_bytes_to_buffer as usize,
            ids: state.ids.clone(),
            prefix: state.prefix.clone(),
            prefix_index: state.prefix_index as usize,
            cumulative_output: state.cumulative_output.clone(),
            output_index: state.output_index as usize,
            prompt_seeded: state.prompt_seeded,
        }
    }
}

#[cfg(test)]
mod tests {
    /// A stream that ends before its first generated token, such as an
    /// image-only generation or a request whose first sampled token is EOS,
    /// produced no text: the prompt is decode context, never output.
    #[test]
    fn flush_without_generated_tokens_emits_nothing() {
        let tokenizer = crate::api::test_support::configured_tokenizer();
        for prompt in ["p", "a longer prompt"] {
            let prompt_token_ids: Vec<u32> = prompt.bytes().map(u32::from).collect();
            let mut decoder = tokenizer.create_decode_stream(&prompt_token_ids, true, 0);

            let (last_chunk, output) = decoder.flush(None).unwrap();

            assert_eq!(last_chunk, None, "prompt {prompt:?}");
            assert_eq!(output, "", "prompt {prompt:?}");
        }
    }

    /// Generated text is the text past the prompt's decode, emitted token by
    /// token; a stream handed to another decoder continues where it stopped.
    #[test]
    fn generated_text_follows_the_prompt_across_a_handoff() {
        let tokenizer = crate::api::test_support::configured_tokenizer();
        let ids = |text: &str| text.bytes().map(u32::from).collect::<Vec<u32>>();
        let mut decoder = tokenizer.create_decode_stream(&ids("prompt "), false, 0);

        for token in ids("he") {
            decoder.push_token(token).unwrap();
        }
        let mut resumed =
            super::IncrementalDecoder::resume(&decoder.state(), std::sync::Arc::clone(&tokenizer));
        for token in ids("llo") {
            resumed.push_token(token).unwrap();
        }

        assert_eq!(resumed.next_chunk().as_deref(), Some("hello"));
        assert_eq!(resumed.flush(None).unwrap(), (None, "hello".to_owned()));
    }
}
