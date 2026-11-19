//! How DeepSeek-V4.1 hashes the n-grams of its n-gram memories (Engram) into table rows:
//! the normalized vocabulary, each layer's multipliers and the prime ranges of its heads, as the
//! checkpoint's reference code builds them (`inference/engram.py`).
//!
//! The reference derives all three at load, from the tokenizer and the configuration:
//!
//! - the **normalization** maps every vocabulary id onto the ids of the distinct texts its tokens
//!   normalize to (NFKC, NFD, accents stripped, lower case, runs of white space as one space,
//!   stripped, a token of one space kept), in vocabulary order; a token that decodes to a part of
//!   a UTF-8 character is keyed by its raw form (`build_compressed_token_map`);
//! - each layer's **multipliers** are odd numbers drawn by numpy's default generator (PCG64
//!   seeded through a `SeedSequence`) from the seed `10007 · layer`, bounded so that a product
//!   with a normalized id cannot overflow 64 bits (`compute_hash_multipliers`);
//! - each head of each n-gram length of each layer, in that order, takes the next prime above the
//!   configuration's `engram_vocab_size − 1` that no head before took, and the heads' ranges of a
//!   layer's table follow one another (`EngramLayout.from_args`).
//!
//! [`Hashing`] holds the three for every n-gram layer and states one layer's
//! [`nn::modules::NgramHashing`].

use std::path::Path;

use nn::modules::NgramHashing;
use nn::{Device, Result, Tensor, VarBuilder};
use tokenizers::normalizers::{Lowercase, NFD, NFKC, Replace, Sequence, Strip, StripAccents};
use tokenizers::{NormalizedString, Normalizer, NormalizerWrapper, Tokenizer};

use crate::deepseek_v41::ID;
use crate::deepseek_v41::config::{Config, EngramConfig};

/// The seed of layer `layer`'s multipliers is this times the layer.
const SEED_FACTOR: u64 = 10_007;

/// A private-use character a token of one space is kept through `Strip` as.
const SENTINEL: &str = "\u{e000}";

/// The n-gram hashing of every n-gram layer of a checkpoint.
#[derive(Clone, PartialEq, Debug)]
pub struct Hashing {
    /// The layers, in the configuration's order.
    layers: Vec<u32>,
    /// `[vocabulary]`: each vocabulary id's normalized id.
    normalize: Vec<i32>,
    /// `[layer][N]`: each layer's multipliers, the token's own first.
    multipliers: Vec<Vec<i64>>,
    /// `[layer][N − 1][H]`: each head's prime modulus.
    primes: Vec<Vec<Vec<u64>>>,
}

impl Hashing {
    /// The hashing `config` states, with the normalization of the tokenizer at `tokenizer`
    /// (`tokenizer.json` of the checkpoint), covering the embedding's whole vocabulary.
    ///
    /// # Errors
    ///
    /// A model without n-gram memories, a tokenizer that cannot be read, or a normalized
    /// vocabulary of another size than the configuration states: every multiplier is drawn for
    /// that size, so another would hash every row elsewhere.
    pub fn open(config: &Config, tokenizer: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|error| {
            ID.checkpoint(format!("the tokenizer {}: {error}", tokenizer.display()))
        })?;
        let (normalize, size) = normalization(&tokenizer, config.vocab_size as usize)?;
        Self::new(config, normalize, size)
    }

    /// The hashing `config` states over a normalization `normalize` of `size` normalized ids.
    ///
    /// # Errors
    ///
    /// A model without n-gram memories, or a normalized vocabulary of another size than the
    /// configuration states.
    pub fn new(config: &Config, normalize: Vec<i32>, size: u32) -> Result<Self> {
        let engram = engram(config)?;
        if size != engram.compressed_vocab_size {
            return Err(ID.checkpoint(format!(
                "the tokenizer normalizes into {size} ids, and the configuration's multipliers \
                 are drawn for {}",
                engram.compressed_vocab_size
            )));
        }

        Ok(Self {
            layers: engram.layer_ids.clone(),
            multipliers: engram
                .layer_ids
                .iter()
                .map(|&layer| multipliers(layer, engram.max_ngram_size, size))
                .collect(),
            primes: primes(engram),
            normalize,
        })
    }

    /// The normalized id of every vocabulary id.
    pub fn normalize(&self) -> &[i32] {
        &self.normalize
    }

    /// Layer `layer`'s multipliers, the token's own first.
    pub fn multipliers(&self, layer: u32) -> Option<&[i64]> {
        let index = self.index(layer)?;
        Some(&self.multipliers[index])
    }

    /// Layer `layer`'s prime moduli, `[N − 1][H]`.
    pub fn primes(&self, layer: u32) -> Option<&[Vec<u64>]> {
        let index = self.index(layer)?;
        Some(&self.primes[index])
    }

    /// Layer `layer`'s first table row of each head, `[N − 1][H]`: the primes' sums before it.
    pub fn offsets(&self, layer: u32) -> Option<Vec<Vec<u64>>> {
        let primes = self.primes(layer)?;
        let mut next = 0;
        Some(
            primes
                .iter()
                .map(|heads| {
                    heads
                        .iter()
                        .map(|prime| {
                            let first = next;
                            next += prime;
                            first
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// Layer `layer`'s hashing as the n-gram memory computes it, its tables published under `vb`
    /// as the layer's constants.
    ///
    /// # Errors
    ///
    /// A layer without an n-gram memory, or what the loader refuses.
    pub fn layer(&self, layer: u32, vb: &VarBuilder<'_, '_>) -> Result<NgramHashing> {
        let [normalize, multipliers, moduli, offsets] = self.tables(layer)?;
        NgramHashing::new(
            vb.constant_from("normalize", &normalize)?,
            vb.constant_from("multipliers", &multipliers)?,
            vb.constant_from("moduli", &moduli)?,
            vb.constant_from("offsets", &offsets)?,
        )
    }

    /// Layer `layer`'s hashing over host tensors, for a forward on the CPU device.
    ///
    /// # Errors
    ///
    /// A layer without an n-gram memory.
    pub fn host(&self, layer: u32) -> Result<NgramHashing> {
        let [normalize, multipliers, moduli, offsets] = self.tables(layer)?;
        NgramHashing::new(normalize, multipliers, moduli, offsets)
    }

    /// Layer `layer`'s tables as host tensors: the normalization `[vocabulary]` I32, the
    /// multipliers `[N]` I64, and the moduli and offsets `[N − 1, H]` I64.
    fn tables(&self, layer: u32) -> Result<[Tensor; 4]> {
        let (Some(multipliers), Some(primes), Some(offsets)) = (
            self.multipliers(layer),
            self.primes(layer),
            self.offsets(layer),
        ) else {
            return Err(ID.config(format!("layer {layer} holds no n-gram memory")));
        };
        let (order, heads) = (multipliers.len(), primes[0].len());
        let wide = |values: &[Vec<u64>]| -> Result<Tensor> {
            let flat = values
                .iter()
                .flatten()
                .map(|&value| value as i64)
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(flat, (order - 1, heads), &Device::Cpu)?)
        };

        Ok([
            Tensor::from_vec(self.normalize.clone(), self.normalize.len(), &Device::Cpu)?,
            Tensor::from_vec(multipliers.to_vec(), order, &Device::Cpu)?,
            wide(primes)?,
            wide(&offsets)?,
        ])
    }

    fn index(&self, layer: u32) -> Option<usize> {
        self.layers.iter().position(|&candidate| candidate == layer)
    }
}

/// The n-gram memories `config` states.
fn engram(config: &Config) -> Result<&EngramConfig> {
    config
        .engram
        .as_ref()
        .ok_or_else(|| ID.config("the model has no n-gram memory".to_owned()))
}

/// The normalizer the reference keys a token's text by.
fn normalizer() -> Result<Sequence> {
    let replace = |pattern: &str, content: &str, regex: bool| -> Result<NormalizerWrapper> {
        let replace = if regex {
            Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::Regex(pattern.to_owned()),
                content,
            )
        } else {
            Replace::new(pattern, content)
        };
        Ok(replace
            .map_err(|error| ID.config(format!("the normalizer's pattern {pattern:?}: {error}")))?
            .into())
    };

    Ok(Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        replace("[ \t\r\n]+", " ", true)?,
        replace("^ $", SENTINEL, true)?,
        Strip::new(true, true).into(),
        replace(SENTINEL, " ", false)?,
    ]))
}

/// The normalized id of every id of `tokenizer`'s vocabulary, and the count of normalized ids:
/// ids in the order of their first token, a token keyed by its normalized text (its decoded
/// text when that normalizes to nothing, its raw form when it decodes to a part of a UTF-8
/// character). The map covers `vocabulary` ids; an id past the tokenizer's vocabulary, which the
/// embedding pads and no request holds, maps to the padding of the first id.
///
/// # Errors
///
/// A token the tokenizer cannot decode, or a normalizer the tokenizer library refuses.
pub fn normalization(tokenizer: &Tokenizer, vocabulary: usize) -> Result<(Vec<i32>, u32)> {
    let normalizer = normalizer()?;
    let size = tokenizer.get_vocab_size(true);
    let mut keys = std::collections::HashMap::<String, i32>::new();
    let mut normalize = Vec::with_capacity(vocabulary.max(size));

    for id in 0..size as u32 {
        let text = tokenizer
            .decode(&[id], false)
            .map_err(|error| ID.checkpoint(format!("decoding token {id}: {error}")))?;
        let key = if text.contains('\u{fffd}') {
            tokenizer.id_to_token(id).unwrap_or_default()
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(|error| ID.checkpoint(format!("normalizing token {id}: {error}")))?;
            match normalized.get() {
                "" => text,
                normalized => normalized.to_owned(),
            }
        };
        let next = keys.len() as i32;
        normalize.push(*keys.entry(key).or_insert(next));
    }
    let count = keys.len() as u32;
    normalize.resize(
        vocabulary.max(size),
        normalize.first().copied().unwrap_or(0),
    );
    Ok((normalize, count))
}

/// Whether `candidate` is prime: trial division, exact for the moduli's magnitude.
fn is_prime(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }
    let mut divisor = 2;
    while divisor * divisor <= candidate {
        if candidate.is_multiple_of(divisor) {
            return false;
        }
        divisor += if divisor == 2 { 1 } else { 2 };
    }
    true
}

/// Every layer's prime moduli, `[layer][N − 1][H]`: each head of each n-gram length of each
/// layer in order takes the smallest prime above `vocab_size − 1` (searched again from there
/// for every n-gram length) that no head before it took.
fn primes(engram: &EngramConfig) -> Vec<Vec<Vec<u64>>> {
    let mut seen = std::collections::BTreeSet::new();
    engram
        .layer_ids
        .iter()
        .map(|_| {
            (1..engram.max_ngram_size)
                .map(|_| {
                    let mut current = engram.vocab_size - 1;
                    (0..engram.heads)
                        .map(|_| {
                            current += 1;
                            while !is_prime(current) || seen.contains(&current) {
                                current += 1;
                            }
                            seen.insert(current);
                            current
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Layer `layer`'s `order` multipliers for a normalized vocabulary of `vocabulary` ids:
/// `2·x + 1` for `x` drawn uniformly below `⌊⌊(2⁶³ − 1) / vocabulary⌋ / 2⌋` by numpy's
/// `default_rng(10007 · layer).integers`.
fn multipliers(layer: u32, order: u32, vocabulary: u32) -> Vec<i64> {
    let bound = ((i64::MAX as u64 / u64::from(vocabulary)) / 2).max(1);
    let mut generator = Pcg64::seeded(SEED_FACTOR * u64::from(layer));

    (0..order)
        .map(|_| generator.below(bound) as i64 * 2 + 1)
        .collect()
}

/// numpy's `SeedSequence` of one integer seed, drawing the 32-bit words a bit generator is
/// seeded with (`numpy/random/bit_generator.pyx`).
struct SeedSequence {
    pool: [u32; 4],
}

impl SeedSequence {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;

    /// The sequence of `seed`, whose entropy is its little-endian 32-bit words.
    fn new(seed: u64) -> Self {
        let mut entropy = Vec::new();
        let mut rest = seed;
        loop {
            entropy.push(rest as u32);
            rest >>= 32;
            if rest == 0 {
                break;
            }
        }

        let mut hash = Self::INIT_A;
        let mut hashmix = |value: u32| -> u32 {
            let mut value = value ^ hash;
            hash = hash.wrapping_mul(Self::MULT_A);
            value = value.wrapping_mul(hash);
            value ^ (value >> Self::XSHIFT)
        };
        let mix = |x: u32, y: u32| -> u32 {
            let result = Self::MIX_MULT_L
                .wrapping_mul(x)
                .wrapping_sub(Self::MIX_MULT_R.wrapping_mul(y));
            result ^ (result >> Self::XSHIFT)
        };

        let mut pool = [0_u32; 4];
        for (index, word) in pool.iter_mut().enumerate() {
            *word = hashmix(entropy.get(index).copied().unwrap_or(0));
        }
        for source in 0..pool.len() {
            for destination in 0..pool.len() {
                if source != destination {
                    pool[destination] = mix(pool[destination], hashmix(pool[source]));
                }
            }
        }
        for &word in entropy.iter().skip(pool.len()) {
            for cell in pool.iter_mut() {
                *cell = mix(*cell, hashmix(word));
            }
        }
        Self { pool }
    }

    /// `words` 64-bit words of state, each two 32-bit words little-endian.
    fn state(&self, words: usize) -> Vec<u64> {
        let mut hash = Self::INIT_B;
        let halves: Vec<u32> = (0..2 * words)
            .map(|index| {
                let mut value = self.pool[index % self.pool.len()] ^ hash;
                hash = hash.wrapping_mul(Self::MULT_B);
                value = value.wrapping_mul(hash);
                value ^ (value >> Self::XSHIFT)
            })
            .collect();
        halves
            .chunks(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect()
    }
}

/// numpy's PCG64 bit generator (128-bit state, XSL-RR output) and its bounded 64-bit draws
/// (`numpy/random/src/pcg64`, `distributions.c`).
struct Pcg64 {
    state: u128,
    increment: u128,
}

impl Pcg64 {
    const MULTIPLIER: u128 = (2_549_297_995_355_413_924 << 64) | 4_865_540_595_714_422_341;

    /// The generator numpy's `default_rng(seed)` holds.
    fn seeded(seed: u64) -> Self {
        let words = SeedSequence::new(seed).state(4);
        let initial = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let sequence = (u128::from(words[2]) << 64) | u128::from(words[3]);

        let mut generator = Self {
            state: 0,
            increment: (sequence << 1) | 1,
        };
        generator.step();
        generator.state = generator.state.wrapping_add(initial);
        generator.step();
        generator
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.increment);
    }

    /// The next 64-bit output.
    fn next(&mut self) -> u64 {
        self.step();
        let folded = ((self.state >> 64) as u64) ^ (self.state as u64);
        folded.rotate_right((self.state >> 122) as u32)
    }

    /// A draw uniform below `bound` (more than 2³²), by Lemire's multiply-and-reject, as
    /// `Generator.integers(0, bound)` draws an `int64`.
    fn below(&mut self, bound: u64) -> u64 {
        let range = bound - 1;
        let mut product = u128::from(self.next()) * u128::from(bound);
        if (product as u64) < bound {
            let threshold = (u64::MAX - range) % bound;
            while (product as u64) < threshold {
                product = u128::from(self.next()) * u128::from(bound);
            }
        }
        (product >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy 2.3.5's `default_rng(10007 · layer).integers(0, bound, 4)` for the checkpoint's
    /// normalized vocabulary of 99,092 ids, times two plus one.
    #[test]
    fn the_multipliers_are_numpys_draws() {
        assert_eq!(
            multipliers(1, 4, 99_092),
            [
                76_632_096_046_245,
                4_839_876_093_313,
                35_959_672_319_349,
                73_987_337_458_391
            ]
        );
        assert_eq!(
            multipliers(14, 4, 99_092),
            [
                67_716_810_739_261,
                51_510_806_800_915,
                30_921_347_202_721,
                82_619_226_485_591
            ]
        );
    }

    #[test]
    fn every_head_takes_the_next_prime_no_head_before_took() {
        let engram = EngramConfig {
            layer_ids: vec![1, 14],
            num_embeddings: vec![384_006_168, 384_016_682],
            max_ngram_size: 4,
            vocab_size: 16_000_000,
            heads: 8,
            head_dim: 256,
            pad_token_id: 2,
            compressed_vocab_size: 99_092,
        };
        let primes = primes(&engram);

        // The first and last moduli of each layer, as the reference's layout states them.
        assert_eq!(primes[0][0][..3], [16_000_057, 16_000_079, 16_000_081]);
        assert_eq!(primes[0][2][7], 16_000_463);
        assert_eq!(primes[1][0][0], 16_000_477);
        assert_eq!(primes[1][2][7], 16_000_889);
        // A layer's table holds exactly its heads' ranges.
        for (layer, rows) in primes.iter().zip(&engram.num_embeddings) {
            assert_eq!(layer.iter().flatten().sum::<u64>(), *rows);
        }
    }
}
