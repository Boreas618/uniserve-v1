//! Profiler ranges: the module a forward is inside.
//!
//! A module states its forward with [`range`], giving its class, which names the kind of module
//! across the deployment (`"Mlp"`, `"Attention"`), and its path in the model, which names this
//! instance and is the prefix its `VarBuilder` loaded it under
//! ([`VarBuilder::path`](crate::VarBuilder::path)). The range changes nothing the forward
//! computes: a profile run files every launch inside it under the path, and a deployment that
//! times modules places range markers around it (`backend::profiler`).

use crate::Error;

/// The failure of a range: its body's own, or the profiler's.
///
/// The backend's range takes a body whose error it can make from its own; a model's body fails
/// with the model's error, which is made from this crate's, so the two are told apart here and
/// converted once the range has ended.
enum Failure<E> {
    /// The body failed.
    Body(E),
    /// A range marker could not be launched.
    Range(backend::Error),
}

impl<E> From<backend::Error> for Failure<E> {
    fn from(error: backend::Error) -> Self {
        Self::Range(error)
    }
}

/// Runs `body`, the forward of the module instance at `path` of `class`, as a profiler range.
///
/// On CPU tensors with no context installed this is `body()`.
///
/// # Errors
///
/// What `body` returns, or the profiler's failure to launch a range marker.
pub fn range<T, E: From<Error>>(
    class: &str,
    path: &str,
    body: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    let ranged = backend::profiler::range(class, path, || body().map_err(Failure::Body));

    ranged.map_err(|failure| match failure {
        Failure::Body(error) => error,
        Failure::Range(error) => E::from(Error::Backend(error)),
    })
}

#[cfg(test)]
mod tests {
    use super::range;
    use crate::{Error, Result};

    #[test]
    fn a_range_returns_what_its_body_returns() {
        let output: Result<u32> = range("Mlp", "layers.0.mlp", || Ok(7));
        assert_eq!(output.unwrap(), 7);

        let refused: Result<u32> = range("Mlp", "layers.0.mlp", || {
            Err(Error::shape("Mlp", "rows of another width"))
        });
        assert!(matches!(refused, Err(Error::Shape { owner: "Mlp", .. })));
    }
}
