//! Profiler ranges: the module a forward is inside.
//!
//! A module states its forward with [`range`], giving its class, which names the kind of module
//! across the deployment, and its path in the model, which names this instance. A range does two
//! things with that, and neither changes what the forward computes:
//!
//! - while a launch table is being recorded, the innermost range's path is the module path of
//!   every launch inside it;
//! - when execution set a [`Profiler`] on a context that launches, the range's body runs between
//!   two range markers on the stream, which a captured graph keeps and every replay runs.

use std::error::Error as StdError;

use ops::platform::Stream;

use crate::{Error, context};

/// What execution implements to time a model's modules on the device.
///
/// Which classes carry markers is the deployment's choice, so [`Profiler::enter`] returns `None`
/// for a class it does not time and the module then runs as if no profiler were set.
pub trait Profiler: Send + Sync {
    /// The module instance at `path` of `class` is entered on `stream`: launches its entry
    /// marker and returns the instance [`Profiler::exit`] is given, or `None` when the class
    /// carries no marker.
    ///
    /// # Errors
    ///
    /// When the marker could not be launched.
    fn enter(
        &self,
        class: &str,
        path: &str,
        stream: &Stream,
    ) -> std::result::Result<Option<u32>, Box<dyn StdError + Send + Sync>>;

    /// The instance [`Profiler::enter`] returned ends on `stream`: launches its exit marker.
    ///
    /// # Errors
    ///
    /// When the marker could not be launched.
    fn exit(
        &self,
        instance: u32,
        stream: &Stream,
    ) -> std::result::Result<(), Box<dyn StdError + Send + Sync>>;
}

/// Runs `body`, the forward of the module instance at `path` of `class`, as a profiler range.
///
/// With no context installed, as when a module computes on CPU tensors alone, this is `body()`.
/// A body that fails leaves its range without an exit marker; its error is the forward's.
///
/// # Errors
///
/// What `body` returns, or [`Error::Range`] when the profiler could not launch a marker.
pub fn range<T, E: From<Error>>(
    class: &str,
    path: &str,
    body: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    let Some(context) = context::installed() else {
        return body();
    };

    let failed = |source| Error::Range {
        class: class.to_owned(),
        path: path.to_owned(),
        source,
    };
    let (profiler, recording) = {
        let state = context.state();
        (state.profiler.clone(), state.launches.is_some())
    };

    // Only a context that launches has a stream to put markers on.
    let marker = match (&profiler, context.stream()) {
        (Some(profiler), Some(stream)) => profiler
            .enter(class, path, stream)
            .map_err(failed)?
            .map(|instance| (profiler, stream, instance)),
        _ => None,
    };

    // The path is kept only for the launch table, and comes off whether or not the body failed.
    if recording {
        context.state().paths.push(path.to_owned());
    }
    let output = body();
    if recording {
        context.state().paths.pop();
    }
    let output = output?;

    if let Some((profiler, stream, instance)) = marker {
        profiler.exit(instance, stream).map_err(failed)?;
    }
    Ok(output)
}

/// The path of the innermost range the thread's forward is inside, or the empty path outside
/// every range: the module path of a launch.
pub(crate) fn path(paths: &[String]) -> String {
    paths.last().cloned().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::range;
    use crate::launch::tests::launch_rms_norm;
    use crate::{Context, Error, Result};

    /// The module paths of the launch table `context` recorded.
    fn paths(context: &Context) -> Vec<String> {
        context
            .take_launches()
            .into_iter()
            .map(|(_, _, path)| path)
            .collect()
    }

    #[test]
    fn a_launch_is_recorded_under_the_innermost_range() {
        let context = Context::meta(None);
        context.record_launches();
        let _guard = context.enter();

        launch_rms_norm().unwrap();
        range("Block", "layers.0", || {
            launch_rms_norm()?;
            range("Mlp", "layers.0.mlp", launch_rms_norm)?;
            // The inner range ended: the outer one is the innermost again.
            launch_rms_norm()
        })
        .unwrap();
        launch_rms_norm().unwrap();

        assert_eq!(
            paths(&context),
            ["", "layers.0", "layers.0.mlp", "layers.0", ""]
        );
    }

    #[test]
    fn a_range_whose_body_fails_still_ends() {
        let context = Context::meta(None);
        context.record_launches();
        let _guard = context.enter();

        let failed: Result<()> = range("Block", "layers.0", || {
            Err(Error::NoContext {
                operation: "a failing body",
            })
        });
        assert!(failed.is_err());
        launch_rms_norm().unwrap();

        assert_eq!(paths(&context), [""]);
    }

    #[test]
    fn a_range_with_no_context_installed_runs_its_body() {
        let output: Result<u32> = range("Block", "layers.0", || Ok(7));
        assert_eq!(output.unwrap(), 7);
    }
}
