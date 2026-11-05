//! The multimodal request rules that outlive any single family.
//!
//! Named output resolutions and aspect-ratio bucket selection are properties
//! of a request rather than of one model family, so they stay here while the
//! families that used them do not.

pub mod resolution;
