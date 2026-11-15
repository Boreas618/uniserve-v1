//! Descriptor keys: the canonical text of the parameter values one compilation covers.
//!
//! A descriptor is the set of values a kernel package bakes into a module, and its `Display` is the
//! key: `dtype_bf16_head_dim_128_page_64`. That text is the last segment of
//! the module's path, so it has to be canonical: the same values must give
//! the same bytes on every host and in every build. Two rules keep it so:
//!
//! - Fields are appended in the order the descriptor declares them, as `_<name>_<value>`, with no
//!   separators inside a value beyond what the value itself prints.
//! - A floating-point value enters as the hexadecimal bit pattern of its `f32`/`f64` form
//!   (`float`), never as decimal text, so `0.1` and its nearest printed form cannot disagree.
//!
//! Boolean values print as `true`/`false`; enums print their chosen short name.

use std::fmt::{self, Display, Write};

/// The elements one program of a block-walking operator covers: the block each such operator's
/// descriptor constructor names.
///
/// One power of two serves them all: wide enough to spread a program's index arithmetic and its
/// launch over a thousand elements, and narrow enough that a tensor of a few thousand elements
/// still spans several programs. The block is a field of each descriptor and so part of each
/// module's key; a caller that needs another tile states it in the descriptor it builds.
pub const BLOCK: u32 = 1024;

/// The canonical key of one descriptor.
///
/// ```ignore
/// impl Display for Descriptor {
///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
///         Key::new()
///             .field("dtype", self.dtype)
///             .field("head_dim", self.head_dim)
///             .field("page", self.page)
///             .finish(formatter)
///     }
/// }
/// ```
#[derive(Clone, Default, Debug)]
pub struct Key {
    text: String,
}

impl Key {
    /// An empty key.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `<name>_<value>`.
    pub fn field(mut self, name: &str, value: impl Display) -> Self {
        if !self.text.is_empty() {
            self.text.push('_');
        }
        self.text.push_str(name);
        self.text.push('_');
        // Writing into a `String` cannot fail.
        let _ = write!(self.text, "{value}");
        self
    }

    /// Append `<name>_<value>` for a value that has a name of its own.
    pub fn variant(mut self, name: &str, value: &str) -> Self {
        if !self.text.is_empty() {
            self.text.push('_');
        }
        self.text.push_str(name);
        self.text.push('_');
        self.text.push_str(value);
        self
    }

    /// The key's text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Write the key, for a descriptor's `Display`.
    pub fn finish(self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl Display for Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

/// A floating-point value as it enters a key: the exact bit pattern, in hexadecimal.
///
/// A descriptor that holds a scale or an epsilon writes it through this, because no decimal text is
/// canonical: two spellings of one `f32` and one spelling of two adjacent `f32` values would
/// otherwise be indistinguishable (or vice versa), and the module path is keyed by the text.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Float(pub u64);

impl From<f32> for Float {
    fn from(value: f32) -> Self {
        Self(u64::from(value.to_bits()))
    }
}

impl From<f64> for Float {
    fn from(value: f64) -> Self {
        Self(value.to_bits())
    }
}

impl Display for Float {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "0x{:x}", self.0)
    }
}

/// The key of `descriptor`, as a string.
pub fn key(descriptor: &impl Display) -> String {
    descriptor.to_string()
}
