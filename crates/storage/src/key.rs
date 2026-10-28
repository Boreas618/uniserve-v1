//! Content keys shared by retained tensors and cached outputs.

use sha2::{Digest as _, Sha256};

/// A SHA-256 digest of length-prefixed parts under a domain tag.
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
}
