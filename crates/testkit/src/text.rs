//! What a fake's text model samples beyond its draws: an end of text.
//!
//! A device sampler's choice depends on the request's Philox stream and the sampled position
//! alone ([`crate::draw_token`]). A text that ends is the same rule with one more fact: every
//! position at or past [`EndOfText::position`] samples [`EndOfText::token`], so a case states
//! where a request's text ends in the logical positions the mechanism writes, a prompt of `p`
//! tokens sampling its `n`-th token at position `p + n`.

/// The end-of-text token a fake samples, which a fake engine's launch states as its model's end
/// of sequence (`crate::args`), so a request whose text ends finishes with `Eos`.
pub const EOS: u32 = 151_645;

/// The raw log-probability the fake reports for every token it samples.
pub const LOGPROB: f32 = -0.5;

/// The scores the fake reports for a sampled `token` (`ipc::execution::CallOutput::token_logprobs`):
/// the token itself at [`LOGPROB`], its position's most likely, and no other candidate.
pub fn scored(token: u32) -> Vec<ipc::TokenLogprob> {
    vec![ipc::TokenLogprob {
        token_id: token,
        logprob: LOGPROB,
        rank: 1,
    }]
}

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
