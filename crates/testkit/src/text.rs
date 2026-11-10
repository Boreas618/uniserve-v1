//! What a fake's text model samples beyond its draws: an end of text, and the control tokens a
//! scheduler over it names.
//!
//! A device sampler's choice depends on the request's Philox stream and the sampled position
//! alone ([`crate::draw_token`]). A text that ends is the same rule with one more fact: every
//! position at or past [`EndOfText::position`] samples [`EndOfText::token`], so a case states
//! where a request's text ends in the logical positions the mechanism writes, a prompt of `p`
//! tokens sampling its `n`-th token at position `p + n`.

use scheduler::SpecialTokenIds;

/// The end-of-text token of [`special_tokens`]: the first EOS id, so a request whose text ends
/// finishes with `Eos`.
pub const EOS: u32 = 151_645;

/// The raw log-probability the fake reports for every token it samples.
pub const LOGPROB: f32 = -0.5;

/// Where a fake's text ends and what it samples there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndOfText {
    /// The first logical position that samples [`Self::token`].
    pub position: u64,
    /// The token sampled from [`Self::position`] on.
    pub token: u32,
}

impl EndOfText {
    /// Text that ends with [`EOS`] at `position`.
    pub const fn at(position: u64) -> Self {
        Self {
            position,
            token: EOS,
        }
    }

    /// Text that ends with [`EOS`] after `tokens` sampled tokens of a prompt of `prompt` tokens.
    pub const fn after(prompt: u64, tokens: u64) -> Self {
        Self::at(prompt + tokens)
    }
}

/// The control tokens of a deployment over the fake: [`EOS`] ends a request.
pub fn special_tokens() -> SpecialTokenIds {
    SpecialTokenIds {
        bos: 151_644,
        eos: vec![EOS, 151_643],
    }
}
