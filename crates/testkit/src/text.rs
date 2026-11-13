//! What a fake's text model samples: its draws, and an end of text.
//!
//! A device sampler's choice depends on the request's Philox stream and the sampled position
//! alone ([`draw_token`]). A text that ends is the same rule with one more fact: every
//! position at or past [`EndOfText::position`] samples [`EndOfText::token`], so a case states
//! where a request's text ends in the logical positions the mechanism writes, a prompt of `p`
//! tokens sampling its `n`-th token at position `p + n`.

use engine::batch::philox::DrawAddress;

/// The first token id a fake samples: every token it samples lies in `SAMPLED..SAMPLED + 5000`, a
/// window of legal ids that no prompt of the tests uses.
pub const SAMPLED: u32 = 1000;

/// Width of the window of sampled token ids above [`SAMPLED`].
const VOCABULARY: u32 = 5_000;

/// The token a fake samples at `draw`: a function of the request's Philox stream and the logical
/// position of the token alone, as a device sampler's draw is.
///
/// A case computes what a request samples from the request's sampling key
/// (`engine::batch::philox::request_sampling_key`) and the position, so a request that continues on
/// another engine, in another slot or another batch, samples what it would have sampled
/// uninterrupted.
pub fn draw_token(draw: DrawAddress) -> u32 {
    // SplitMix64's finalizer over the two halves of the address: a fixed bijective mix, so nearby
    // keys and positions give unrelated tokens.
    let mut mixed = draw.key ^ draw.position.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;

    SAMPLED + (mixed % u64::from(VOCABULARY)) as u32
}

/// The end-of-text token a fake samples, which a fake engine's launch states as its model's end
/// of sequence (`crate::args`), so a request whose text ends finishes with `Eos`.
pub const EOS: u32 = 151_645;

/// The raw log-probability the fake reports for every token it samples.
pub const LOGPROB: f32 = -0.5;

/// The scores the fake reports for a sampled `token` (`engine::batch::CallOutput::token_logprobs`):
/// the token itself at [`LOGPROB`], its position's most likely, and no other candidate.
pub fn scored(token: u32) -> Vec<proto::request::TokenLogprob> {
    vec![proto::request::TokenLogprob {
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
