//! The keys of the state the Storage Server keeps.
//!
//! The server holds storage and never interprets it: every key is computed
//! by the process that derives the state, from the inputs that decide what
//! the stored bytes mean, through the functions of this module. The inputs
//! are, per kind of state:
//!
//! | State | Key inputs | Function |
//! | --- | --- | --- |
//! | retained weights | the weight's canonical text (`nn::WeightKey`): the identity of each checkpoint file it is read from, the element type, the layout name the reading operator declares and its region of the logical tensor | [`weight`], one copy per GPU [`device_weight`] |
//! | exclusive storages (KV cache segments, buffers, latent banks) | the owner's scope, the storage's name, the epoch that created it, its bytes and its GPU | [`exclusive`] |
//! | KV pages, recurrent-state checkpoints | the chain of the group's pages before it, the group, the modality and the page's tokens, rooted at the group's page seed (what computes the pages and how they are laid out, [`page_seed`]) and the request's isolation key | [`page_root`], [`page_key`] |
//! | encoder outputs | the output's name, what computes it and the digest of what it read; one key per page | [`encoder_output`], [`encoder_page`] |
//! | latents | what computes the trajectory (the latent seed), the trajectory and its step count | [`latent_seed`], [`latent_content`] |
//!
//! A KV page has one key at every level of the cache: the device prefix
//! cache (L1) indexes it by the key the chain gives it, the host page pools
//! (L2) hold it under the same key, and the KV storage (L3) keeps it under
//! that key with the rank's KV-head range. A key carries no program's
//! backend name or revision: replacing a kernel program keeps the
//! mathematical content, so derived state keeps hitting. A change of
//! mathematics, precision or format is a build-config change, which changes
//! the identity a seed is computed from and so every key.

use sha2::{Digest as _, Sha256};

use crate::Key;

/// A SHA-256 digest of length-prefixed parts under a domain tag. Every part
/// is length prefixed, so no two different sequences of parts produce the
/// same bytes, and the tag keeps a key of one kind from equalling a key of
/// another kind with the same inputs.
pub struct Digest(Sha256);

impl Digest {
    pub fn new(tag: &[u8]) -> Self {
        let mut digest = Self(Sha256::new());
        digest.bytes(tag);
        digest
    }

    pub fn bytes(&mut self, part: &[u8]) {
        self.0.update((part.len() as u64).to_le_bytes());
        self.0.update(part);
    }

    pub fn text(&mut self, part: &str) {
        self.bytes(part.as_bytes());
    }

    pub fn number(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    pub fn numbers(&mut self, values: &[u64]) {
        self.number(values.len() as u64);
        for value in values {
            self.number(*value);
        }
    }

    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }

    /// The digest as a storage key.
    pub fn key(self) -> Key {
        Key(self.finish())
    }
}

/// The page seed of a cache group whose pages are computed and laid out as
/// `identity` states: the group's upstream component identities and its
/// page layout, as the ranks report them (`PrefixCacheInfo::identities`).
/// The zero key for an empty identity, which a protocol consumer without a
/// model reports: its pages are keyed by their tokens alone.
pub fn page_seed(identity: &str) -> Key {
    if identity.is_empty() {
        return Key([0; 32]);
    }
    let mut digest = Digest::new(b"uniserve-page-seed");
    digest.text(identity);
    digest.key()
}

/// The root of a request's page chain in a group of seed `seed`: the key
/// "before" its first page. An isolation key keeps tenants' prefixes apart:
/// requests of different keys chain different keys and share no page,
/// while requests without one share the pages of each other.
pub fn page_root(seed: &Key, isolation: Option<u64>) -> Key {
    let mut digest = Digest::new(b"uniserve-page-root");
    digest.bytes(&seed.0);
    match isolation {
        Some(key) => {
            digest.number(1);
            digest.number(key);
        }
        None => digest.number(0),
    }
    digest.key()
}

/// The key of the page of `tokens` after the page keyed `parent` (or the
/// root, for the first page) in cache group `group`; `modality` tells apart
/// pages of equal tokens of different modalities. Equal keys identify equal
/// complete prefixes of one seed and isolation key.
pub fn page_key(parent: &Key, group: u32, modality: u8, tokens: &[u32]) -> Key {
    let mut digest = Sha256::new();
    digest.update(parent.0);
    digest.update(group.to_le_bytes());
    digest.update([modality]);
    digest.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    Key(digest.finalize().into())
}

/// The key a weight is retained under, from its canonical text
/// (`nn::WeightKey`'s): every file's identity, the element type, the layout
/// name and the region, and nothing about which program reads it.
pub fn weight(canonical: &str) -> Key {
    let mut digest = Digest::new(b"uniserve-retained-weight");
    digest.text(canonical);
    digest.key()
}

/// The key one GPU's copy of the weight retained under `weight` is held
/// under: a record names the allocation on the device that published it,
/// so each GPU keeps its own copy rather than reading a peer's over the
/// link on every forward.
pub fn device_weight(weight: &Key, gpu: &[u8; 16]) -> Key {
    let mut digest = Digest::new(b"uniserve-device-weight");
    digest.bytes(&weight.0);
    digest.bytes(gpu);
    digest.key()
}

/// The key of the exclusive storage `name` of `bytes` bytes that the worker
/// of `scope` created in `epoch` on the GPU of identity `gpu`: a restarted
/// main process of the same scope and epoch maps it again.
pub fn exclusive(scope: &crate::Scope, name: &str, bytes: u64, gpu: &[u8; 16], epoch: u64) -> Key {
    let mut digest = Digest::new(b"uniserve-exclusive-storage");
    digest.text(&scope.deployment);
    digest.text(&scope.group);
    digest.text(name);
    digest.number(u64::from(scope.rank));
    digest.number(epoch);
    digest.number(bytes);
    digest.bytes(gpu);
    digest.key()
}

/// The content key of encoder output `name`, computed by `identity` (the
/// weights' values and the function and format identities of the modules
/// computing it) from what `inputs` digests.
pub fn encoder_output(name: &str, identity: &str, inputs: &[u8; 32]) -> [u8; 32] {
    let mut digest = Digest::new(b"encoder-output");
    digest.text(name);
    digest.text(identity);
    digest.bytes(inputs);
    digest.finish()
}

/// The key of page `page` of an encoder output whose content key is
/// `content`.
pub fn encoder_page(content: &[u8; 32], page: u32) -> Key {
    let mut digest = Digest::new(b"encoder-page");
    digest.bytes(content);
    digest.number(u64::from(page));
    digest.key()
}

/// The latent seed of what a deployment's denoising computes, from the
/// identities its denoising ranks report (`LatentPoolInfo::identity`): the
/// first eight bytes of the digest of the distinct identities in order, so
/// that a parallel configuration of other ranks computing the same derives
/// the same seed. Zero when no rank reports one.
pub fn latent_seed<'a>(identities: impl IntoIterator<Item = &'a str>) -> u64 {
    let distinct: std::collections::BTreeSet<&str> = identities.into_iter().collect();
    if distinct.is_empty() {
        return 0;
    }
    let mut digest = Digest::new(b"uniserve-latent-seed");
    for identity in distinct {
        digest.text(identity);
    }
    let digest = digest.finish();
    u64::from_le_bytes(digest[..8].try_into().unwrap_or_default())
}

/// The digest of what a trajectory's values are after `steps` steps: what
/// computed them (`seed`, [`latent_seed`]) and the trajectory.
pub fn latent_content(seed: u64, trajectory: u64, steps: u32) -> [u8; 32] {
    let mut digest = Digest::new(b"uniserve-latent-content");
    digest.number(seed);
    digest.number(trajectory);
    digest.number(u64::from(steps));
    digest.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(seed: &Key, isolation: Option<u64>, pages: &[&[u32]]) -> Vec<Key> {
        let mut parent = page_root(seed, isolation);
        pages
            .iter()
            .map(|tokens| {
                parent = page_key(&parent, 0, 0, tokens);
                parent
            })
            .collect()
    }

    #[test]
    fn a_page_key_names_its_whole_prefix() {
        let seed = page_seed("model|page-layout:1");
        let first = chain(&seed, None, &[&[1, 2], &[3, 4]]);
        // The same prefix is the same key, page by page.
        assert_eq!(first, chain(&seed, None, &[&[1, 2], &[3, 4]]));
        // A page after another prefix is another page.
        let other = chain(&seed, None, &[&[9, 2], &[3, 4]]);
        assert_ne!(first[1], other[1]);
        // Another group or modality of the same tokens is another page.
        let root = page_root(&seed, None);
        assert_ne!(
            page_key(&root, 0, 0, &[1, 2]),
            page_key(&root, 1, 0, &[1, 2])
        );
        assert_ne!(
            page_key(&root, 0, 0, &[1, 2]),
            page_key(&root, 0, 1, &[1, 2])
        );
    }

    #[test]
    fn a_seed_and_an_isolation_key_part_the_pages() {
        let pages: &[&[u32]] = &[&[1, 2]];
        let seed = page_seed("model|page-layout:1");
        let keys = chain(&seed, None, pages);
        assert_ne!(keys, chain(&page_seed("model|page-layout:2"), None, pages));
        assert_ne!(keys, chain(&seed, Some(7), pages));
        assert_ne!(chain(&seed, Some(7), pages), chain(&seed, Some(8), pages));
        // A consumer without a model keys its pages by their tokens alone.
        assert_eq!(page_seed(""), Key([0; 32]));
    }

    #[test]
    fn state_of_different_kinds_never_shares_a_key() {
        let content = [7_u8; 32];
        assert_ne!(encoder_page(&content, 0), weight("blob:abc"));
        assert_ne!(
            encoder_output("vision", "id", &content),
            latent_content(1, 2, 3)
        );
        assert_ne!(device_weight(&weight("a"), &[0; 16]), weight("a"));
        assert_ne!(
            device_weight(&weight("a"), &[0; 16]),
            device_weight(&weight("a"), &[1; 16])
        );
    }

    #[test]
    fn a_latent_seed_names_the_distinct_identities() {
        assert_eq!(latent_seed([]), 0);
        assert_eq!(latent_seed(["a", "b"]), latent_seed(["b", "a", "a"]));
        assert_ne!(latent_seed(["a"]), latent_seed(["b"]));
    }
}
