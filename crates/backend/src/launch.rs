//! The one dispatch point: every kernel runs through [`launch`].
//!
//! The generic primitives and the fused operators of `nn` both state an `ops` operator, its
//! descriptor and its arguments, and call [`launch`]. Because nothing reaches a kernel any other
//! way, what holds for one launch holds for all of them: the kernel comes from the context's
//! generation, so a kernel update reaches it; its key is in the launch table, so a profile run
//! states it and a captured graph's node for it can be found; and it is offered to the context's
//! route, so execution can run two implementations of it side by side.

use ops::{Checked, Op};

use crate::context::{self, Context};
use crate::{Error, Result, profiler};

/// Checks the call against `O`'s rules, then launches `O`'s kernel for `descriptor` by the
/// context installed on this thread.
///
/// Under a meta context the call is checked exactly as a launching context checks it and
/// nothing is launched, so a key a profile run states always comes from a call the device would
/// accept. Under a context that launches, the kernel is resolved from the context's generation
/// and the call is offered to the context's route when one is set; a call the route declines,
/// and every call without a route, is one launch of that kernel on the context's stream.
///
/// While the context records a launch table, each call adds `(O::NAME, key, module path)` to
/// it, whichever way the call went. A call the route took is one entry, however many kernels
/// the route launched for it.
///
/// # Errors
///
/// [`Error::NoContext`] when no context is installed; [`Error::Launch`], naming the operator and
/// the key, when the arguments break the operator's rules, the generation holds no kernel for
/// the key, or the route or the launch fails.
pub fn launch<'t, O: Op>(descriptor: &O::Descriptor, args: O::Args<'t>) -> Result<()>
where
    O::Args<'t>: Copy,
{
    let refused = |source| Error::Launch {
        operator: O::NAME,
        key: ops::descriptor::key(descriptor),
        source,
    };

    let context = context::current("a launch")?;
    O::check(descriptor, &args).map_err(refused)?;

    let (kernels, route, recording) = {
        let state = context.state();
        (
            state.kernels.clone(),
            state.route.clone(),
            state.launches.is_some(),
        )
    };

    // Every launch of a serving step passes through here and a recording is the exception, so
    // the key is formatted only when the table will hold it.
    let key = recording.then(|| ops::descriptor::key(descriptor));

    if let (Some(stream), Some(kernels)) = (context.stream(), kernels) {
        let kernel = match &key {
            Some(key) => kernels.kernel(O::NAME, key),
            None => kernels.op::<O>(descriptor),
        }
        .map_err(refused)?;

        let routed = match route {
            Some(route) => route
                .launch(&Checked::<O>::new(descriptor, args), &kernels, stream)
                .map_err(refused)?,
            None => false,
        };

        if !routed {
            O::launch(kernel, descriptor, args, stream).map_err(refused)?;
        }
    }

    if let Some(key) = key {
        record(&context, O::NAME, key);
    }
    Ok(())
}

/// Adds a launch of `operator` with `key` to the launch table `context` is recording, under the
/// innermost profiler range.
fn record(context: &Context, operator: &'static str, key: String) {
    let mut state = context.state();
    let path = profiler::path(&state.paths);

    if let Some(launches) = state.launches.as_mut() {
        launches.push((operator, key, path));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use ops::norm::{RmsNorm, RmsNormArgs, RmsNormDescriptor};
    use ops::{DevicePtr, Operand, Shape};
    use tensor::DType;

    use super::launch;
    use crate::{Context, Error, Result};

    /// One compilation of `rms_norm` over rows of `columns` BF16 elements.
    fn descriptor(columns: u32) -> RmsNormDescriptor {
        RmsNormDescriptor {
            dtype: DType::BF16,
            columns,
            tile: columns.next_power_of_two(),
            pdl: false,
            centred: false,
            groups: 1,
        }
    }

    /// The arguments of `rms_norm` over two rows of `columns` elements with a weight of `weight`
    /// elements. The addresses are never read: a meta context launches nothing.
    fn args(columns: u64, weight: u64) -> RmsNormArgs {
        let rows = Operand::new(
            DevicePtr::new(0x1000),
            DType::BF16,
            Shape::new(&[2, columns]),
        );

        RmsNormArgs {
            x: rows,
            weight: Operand::new(DevicePtr::new(0x2000), DType::BF16, Shape::new(&[weight])),
            out: rows,
            eps: 1e-6,
        }
    }

    /// One accepted launch of `rms_norm`, for the tests that observe a context through its
    /// launch table.
    pub(crate) fn launch_rms_norm() -> Result<()> {
        launch::<RmsNorm>(&descriptor(8), args(8, 8))
    }

    #[test]
    fn a_launch_with_no_context_installed_names_what_is_missing() {
        let refused = launch_rms_norm().unwrap_err();
        assert!(matches!(
            refused,
            Error::NoContext {
                operation: "a launch"
            }
        ));
    }

    #[test]
    fn a_meta_context_records_operator_key_and_path_in_launch_order() {
        let context = Context::meta(None);
        context.record_launches();
        let _guard = context.enter();

        launch::<RmsNorm>(&descriptor(8), args(8, 8)).unwrap();
        launch::<RmsNorm>(&descriptor(16), args(16, 16)).unwrap();

        assert_eq!(
            context.take_launches(),
            [
                ("rms_norm", descriptor(8).to_string(), String::new()),
                ("rms_norm", descriptor(16).to_string(), String::new()),
            ]
        );
    }

    #[test]
    fn launches_are_recorded_only_between_the_start_and_the_take() {
        let context = Context::meta(None);
        let _guard = context.enter();

        launch_rms_norm().unwrap();
        assert!(context.take_launches().is_empty());

        context.record_launches();
        launch_rms_norm().unwrap();
        assert_eq!(context.take_launches().len(), 1);

        // Taking the table ended the recording.
        launch_rms_norm().unwrap();
        assert!(context.take_launches().is_empty());
    }

    #[test]
    fn a_meta_context_refuses_arguments_the_operator_refuses() {
        let context = Context::meta(None);
        context.record_launches();
        let _guard = context.enter();

        // A weight of four elements for rows of eight.
        let refused = launch::<RmsNorm>(&descriptor(8), args(8, 4)).unwrap_err();

        let Error::Launch {
            operator,
            key,
            source,
        } = refused
        else {
            panic!("a refused launch names its operator and key, not {refused}");
        };
        assert_eq!(operator, "rms_norm");
        assert_eq!(key, descriptor(8).to_string());
        assert!(matches!(source, ops::Error::Argument { .. }));

        // A refused call is not a launch.
        assert!(context.take_launches().is_empty());
    }
}
