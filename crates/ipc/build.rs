//! Names the build this crate is compiled in: a digest of the workspace
//! sources every artifact of the build is compiled from.
//!
//! The head, a node's daemon and the rank runtime library are installed as
//! separate artifacts and exchange this crate's messages in a positional
//! encoding, so each registration states the build it runs and the engine
//! refuses any other (`registration::VERSION`). The package version cannot
//! tell two builds of one version apart; the digest emitted here as
//! `UNISERVE_SOURCE_DIGEST` does.

include!("../proto/source_digest.rs");

fn main() -> io::Result<()> {
    emit_source_digest()
}
