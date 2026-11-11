//! Forwards under a meta context: tensor operations on meta tensors through the tensor
//! library's public interface, observed by what a caller of this crate can see — output shapes
//! and element types, the launch table, addresses, the allocator's peak and the input buffer.
//!
//! Nothing here touches a GPU: a meta context launches nothing.

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

use backend::{Context, Error, cuda, empty, external, meta, operand, retained};
use ops::platform::{DeviceRange, PlatformDevicePtr};
use uniserve_tensor::{D, DType, Device, DeviceLocation, Result, Shape, Tensor};

/// The key of a primitive compiled for the block the backend launches with.
const TAIL: &str = "block_1024_pdl_false";

/// A meta input of `shape` elements of `dtype`: a tensor over the meta device, which takes no
/// block of the context's allocator.
fn input(shape: impl Into<Shape>, dtype: DType) -> Tensor {
    retained(0, shape, dtype, &meta()).unwrap()
}

/// Runs `forward` under `context` while it records, and returns the launch table as
/// `(operator, key)` in launch order.
fn recorded(
    context: &Context,
    forward: impl FnOnce() -> Result<()>,
) -> Vec<(&'static str, String)> {
    context.record_launches();
    {
        let _guard = context.enter();
        forward().unwrap();
    }

    context
        .take_launches()
        .into_iter()
        .map(|(operator, key, _)| (operator, key))
        .collect()
}

/// The launch table `forward` records under a new meta context.
fn launches(forward: impl FnOnce() -> Result<()>) -> Vec<(&'static str, String)> {
    recorded(&Context::meta(None), forward)
}

/// `(operator, key)` pairs as the recorded table holds them.
fn table(entries: &[(&'static str, String)]) -> Vec<(&'static str, String)> {
    entries.to_vec()
}

/// The address a tensor's first element is at.
fn address(tensor: &Tensor) -> u64 {
    operand(tensor).unwrap().ptr.address()
}

/// The tensor library's error without the backtrace it carries when backtraces are captured.
fn plain(error: &uniserve_tensor::Error) -> &uniserve_tensor::Error {
    match error {
        uniserve_tensor::Error::WithBacktrace { inner, .. } => plain(inner),
        error => error,
    }
}

/// The backend's own error under a tensor library error.
fn cause(error: &uniserve_tensor::Error) -> &Error {
    match plain(error) {
        uniserve_tensor::Error::Backend(error) => error
            .downcast_ref::<Error>()
            .expect("a backend error of this crate"),
        other => panic!("expected a backend error, not {other}"),
    }
}

const META: DeviceLocation = DeviceLocation::Backend {
    kind: "meta",
    ordinal: 0,
};

#[test]
fn a_broadcast_bias_a_permutation_and_a_cast_state_their_kernels() {
    let table_of = launches(|| {
        let x = input((4, 8), DType::F32);
        let bias = input(8, DType::F32);

        // The bias repeats along the rows: its stride there is zero, so two axes are traversed.
        let biased = x.broadcast_add(&bias)?;
        assert_eq!(biased.dims(), [4, 8]);
        assert_eq!(biased.dtype(), DType::F32);
        assert_eq!(biased.device().location(), META);

        // A transposed source against a contiguous destination: two axes again.
        let transposed = biased.t()?.contiguous()?;
        assert_eq!(transposed.dims(), [8, 4]);

        // Contiguous on both sides: any shape is one axis.
        let half = transposed.to_dtype(DType::BF16)?;
        assert_eq!(half.dims(), [8, 4]);
        assert_eq!(half.dtype(), DType::BF16);

        // A conversion to the tensor's own element type is no operation.
        let same = half.to_dtype(DType::BF16)?;
        assert_eq!(same.dtype(), DType::BF16);
        Ok(())
    });

    assert_eq!(
        table_of,
        table(&[
            (
                "elementwise_binary",
                format!("function_add_dtype_f32_rank_2_{TAIL}")
            ),
            ("copy_strided", format!("dtype_f32_rank_2_{TAIL}")),
            ("cast", format!("input_f32_output_bf16_rank_1_{TAIL}")),
        ])
    );
}

#[test]
fn concatenation_and_padding_copy_each_part_into_its_block_of_the_result() {
    let table_of = launches(|| {
        let x = input((4, 8), DType::F32);

        // Along the first axis each part lands contiguously.
        let stacked = Tensor::cat(&[&x, &x], 0)?;
        assert_eq!(stacked.dims(), [8, 8]);

        // Along the last axis each part's rows are a result row apart.
        let widened = Tensor::cat(&[&x, &x], 1)?;
        assert_eq!(widened.dims(), [4, 16]);

        // One column of zeros before and two after, then the three parts.
        let padded = x.pad_with_zeros(1, 1, 2)?;
        assert_eq!(padded.dims(), [4, 11]);
        assert_eq!(padded.dtype(), DType::F32);
        Ok(())
    });

    let contiguous = ("copy_strided", format!("dtype_f32_rank_1_{TAIL}"));
    let rows = ("copy_strided", format!("dtype_f32_rank_2_{TAIL}"));
    let zeros = ("fill", format!("dtype_f32_rank_1_{TAIL}"));

    assert_eq!(
        table_of,
        table(&[
            contiguous.clone(),
            contiguous.clone(),
            rows.clone(),
            rows.clone(),
            zeros.clone(),
            zeros,
            // A column is one element per row: its rows merge into one axis.
            contiguous,
            rows.clone(),
            rows,
        ])
    );
}

#[test]
fn indexing_states_gathers_and_scatters() {
    let table_of = launches(|| {
        let x = input((4, 8), DType::F32);
        let rows = input(3, DType::U32);

        let selected = x.index_select(&rows, 0)?;
        assert_eq!(selected.dims(), [3, 8]);
        assert_eq!(selected.dtype(), DType::F32);

        let columns = input((4, 2), DType::U32);
        let gathered = x.gather(&columns, 1)?;
        assert_eq!(gathered.dims(), [4, 2]);

        // A copy of `x`, then the selected rows added into it.
        let added = x.index_add(&rows, &selected, 0)?;
        assert_eq!(added.dims(), [4, 8]);

        // Rows 1..4 of a new tensor, contiguous on both sides.
        let target = empty((4, 8), DType::F32).unwrap();
        target.slice_set(&selected, 0, 1)?;
        assert_eq!(target.dims(), [4, 8]);
        Ok(())
    });

    assert_eq!(
        table_of,
        table(&[
            (
                "index",
                format!("function_gather_dtype_f32_indices_u32_rank_2_{TAIL}")
            ),
            (
                "index",
                format!("function_gather_dtype_f32_indices_u32_rank_2_{TAIL}")
            ),
            ("copy_strided", format!("dtype_f32_rank_1_{TAIL}")),
            (
                "index",
                format!("function_scatter_add_dtype_f32_indices_u32_rank_1_{TAIL}")
            ),
            ("copy_strided", format!("dtype_f32_rank_1_{TAIL}")),
        ])
    );
}

#[test]
fn reductions_state_their_axes_and_the_class_of_their_length() {
    let table_of = launches(|| {
        let x = input((4, 8), DType::F32);

        let sums = x.sum_keepdim(1)?;
        assert_eq!(sums.dims(), [4, 1]);
        assert_eq!(sums.dtype(), DType::F32);

        let largest = x.max(0)?;
        assert_eq!(largest.dims(), [8]);

        let positions = x.argmax(D::Minus1)?;
        assert_eq!(positions.dims(), [4]);
        assert_eq!(positions.dtype(), DType::U32);
        Ok(())
    });

    assert_eq!(
        table_of,
        table(&[
            (
                "reduce",
                format!("function_sum_dtype_f32_kept_1_reduced_1_tile_8_{TAIL}")
            ),
            (
                "reduce",
                format!("function_max_dtype_f32_kept_1_reduced_1_tile_4_{TAIL}")
            ),
            (
                "reduce",
                format!("function_argmax_dtype_f32_kept_1_reduced_1_tile_8_{TAIL}")
            ),
        ])
    );
}

#[test]
fn a_selection_a_product_and_a_composed_expression_state_their_kernels() {
    let table_of = launches(|| {
        let x = input((4, 8), DType::F32);
        let y = input((4, 8), DType::F32);

        let mask = x.ge(&y)?;
        assert_eq!(mask.dtype(), DType::U8);
        let chosen = mask.where_cond(&x, &y)?;
        assert_eq!(chosen.dims(), [4, 8]);
        assert_eq!(chosen.dtype(), DType::F32);

        // A weight stored `[k, n]`, and one stored `[n, k]` read through its transpose.
        let weight = input((8, 5), DType::F32);
        let product = x.matmul(&weight)?;
        assert_eq!(product.dims(), [4, 5]);
        assert_eq!(product.dtype(), DType::F32);

        let stored = input((5, 8), DType::F32);
        let projected = x.matmul(&stored.t()?)?;
        assert_eq!(projected.dims(), [4, 5]);

        let composed = ((&x * 2.0)? + 1.0)?.relu()?.sqrt()?;
        assert_eq!(composed.dims(), [4, 8]);
        Ok(())
    });

    let gemm = |order: &str| {
        (
            "gemm",
            format!("input_f32_weight_f32_accumulate_f32_output_f32_order_{order}_epilogue_none"),
        )
    };
    let unary = |function: &str| {
        (
            "elementwise_unary",
            format!("function_{function}_dtype_f32_rank_1_{TAIL}"),
        )
    };

    assert_eq!(
        table_of,
        table(&[
            (
                "elementwise_binary",
                format!("function_ge_dtype_f32_rank_1_{TAIL}")
            ),
            (
                "elementwise_where",
                format!("dtype_f32_condition_u8_rank_1_{TAIL}")
            ),
            gemm("kxn"),
            gemm("nxk"),
            unary("affine"),
            unary("affine"),
            unary("relu"),
            unary("sqrt"),
        ])
    );
}

#[test]
fn a_batch_of_distinct_matrices_is_one_product_per_member() {
    let table_of = launches(|| {
        let lhs = input((3, 4, 8), DType::BF16);
        let rhs = input((3, 8, 5), DType::BF16);

        let product = lhs.matmul(&rhs)?;
        assert_eq!(product.dims(), [3, 4, 5]);
        assert_eq!(product.dtype(), DType::BF16);

        // One right-hand matrix for the whole batch folds the batch into the rows.
        let weight = input((8, 5), DType::BF16);
        let folded = lhs.broadcast_matmul(&weight)?;
        assert_eq!(folded.dims(), [3, 4, 5]);
        Ok(())
    });

    let gemm = (
        "gemm",
        "input_bf16_weight_bf16_accumulate_f32_output_bf16_order_kxn_epilogue_none".to_owned(),
    );
    assert_eq!(table_of, vec![gemm; 4]);
}

#[test]
fn convolutions_launch_the_convolution_operators() {
    let table_of = launches(|| {
        let signal = input((1, 2, 16), DType::F32);
        let filter = input((3, 2, 4), DType::F32);
        let convolved = signal.conv1d(&filter, 1, 2, 1, 1)?;
        assert_eq!(convolved.dims(), [1, 3, 8]);

        let image = input((1, 2, 8, 8), DType::F32);
        let window = input((3, 2, 3, 3), DType::F32);
        let filtered = image.conv2d(&window, 1, 1, 1, 1)?;
        assert_eq!(filtered.dims(), [1, 3, 8, 8]);

        // `[in, out, taps]`: sixteen samples upsampled by two.
        let taps = input((2, 3, 4), DType::F32);
        let upsampled = signal.conv_transpose1d(&taps, 1, 0, 2, 1, 1)?;
        assert_eq!(upsampled.dims(), [1, 3, 32]);
        Ok(())
    });

    let key = "input_f32_weight_f32_output_f32".to_owned();
    assert_eq!(
        table_of,
        table(&[
            ("conv1d", key.clone()),
            ("conv3d", key.clone()),
            ("conv_transpose1d", key),
        ])
    );
}

#[test]
fn operations_over_no_element_launch_nothing_and_return_their_empty_output() {
    let table_of = launches(|| {
        let x = Tensor::zeros((0, 8), DType::F32, &meta())?;
        let bias = input(8, DType::F32);
        assert_eq!(x.dims(), [0, 8]);

        assert_eq!(x.relu()?.dims(), [0, 8]);
        assert_eq!(x.broadcast_add(&bias)?.dims(), [0, 8]);
        assert_eq!(x.to_dtype(DType::BF16)?.dtype(), DType::BF16);
        assert_eq!(x.t()?.contiguous()?.dims(), [8, 0]);
        assert_eq!(x.copy()?.dims(), [0, 8]);
        assert_eq!(Tensor::cat(&[&x, &x], 0)?.dims(), [0, 8]);
        assert_eq!(x.max(1)?.dims(), [0]);

        let weight = input((8, 5), DType::F32);
        assert_eq!(x.matmul(&weight)?.dims(), [0, 5]);

        let none = input(0, DType::U32);
        assert_eq!(bias.index_select(&none, 0)?.dims(), [0]);
        Ok(())
    });

    assert_eq!(table_of, []);
}

#[test]
fn a_sum_over_an_axis_of_no_element_still_writes_its_zeros() {
    let table_of = launches(|| {
        let x = input((4, 0), DType::F32);
        assert_eq!(x.sum_keepdim(1)?.dims(), [4, 1]);
        Ok(())
    });

    assert_eq!(
        table_of,
        table(&[(
            "reduce",
            format!("function_sum_dtype_f32_kept_1_reduced_1_tile_1_{TAIL}")
        )])
    );
}

/// A forward that keeps some tensors and drops others, and the addresses of the ones it keeps.
fn forward() -> Result<Vec<u64>> {
    let x = input((4, 8), DType::F32);
    let bias = input(8, DType::F32);

    let biased = x.broadcast_add(&bias)?;
    let transposed = biased.t()?.contiguous()?;
    drop(biased);

    let half = transposed.to_dtype(DType::BF16)?;
    let sums = half.sum_keepdim(0)?;
    let scaled = (&sums * 0.5)?;

    Ok(vec![
        address(&transposed),
        address(&half),
        address(&sums),
        address(&scaled),
    ])
}

/// Resets `context` and runs [`forward`] under it.
fn run(context: &Context) -> Vec<u64> {
    context.reset();
    let _guard = context.enter();

    forward().unwrap()
}

#[test]
fn two_runs_of_one_forward_from_a_reset_receive_the_same_addresses() {
    let region = DeviceRange::new(PlatformDevicePtr::from_address(0x1000_0000), 1 << 20);
    let context = Context::meta(Some(region));

    let first = run(&context);
    let second = run(&context);

    assert_eq!(first, second);
    assert!(first.iter().all(|address| *address >= 0x1000_0000));
}

#[test]
fn a_counting_context_places_and_peaks_as_a_context_over_a_region_does() {
    const BASE: u64 = 0x1000_0000;
    let region = DeviceRange::new(PlatformDevicePtr::from_address(BASE), 1 << 20);
    let placed = Context::meta(Some(region));
    let counting = Context::meta(None);

    let addresses = run(&placed);
    let offsets = run(&counting);

    let relative: Vec<u64> = addresses.iter().map(|address| address - BASE).collect();
    assert_eq!(relative, offsets);
    assert_eq!(placed.peak(), counting.peak());

    // The four tensors the forward keeps are alive at once, each in one 256-byte block; the
    // one it drops gave its block to the next.
    assert_eq!(counting.peak(), 4 * 256);
}

#[test]
fn a_dropped_tensor_returns_its_block_to_the_allocator() {
    let context = Context::meta(None);
    let _guard = context.enter();
    let x = input((4, 8), DType::F32);

    let step = || -> Result<u64> {
        let positive = x.relu()?;
        let root = positive.sqrt()?;
        Ok(address(&root))
    };

    let first = step().unwrap();
    let peak = context.peak();
    for _ in 0..100 {
        // Both intermediates of the step before are free again.
        assert_eq!(step().unwrap(), first);
    }

    assert_eq!(context.peak(), peak);
    assert_eq!(peak, 2 * 256);
}

#[test]
fn reading_a_meta_tensor_is_an_error() {
    let context = Context::meta(None);
    let _guard = context.enter();

    let x = input(8, DType::F32);
    let refused = x.to_vec1::<f32>().unwrap_err();
    assert!(matches!(cause(&refused), Error::MetaRead));

    // So is reading a result, and moving one to the host.
    let doubled = (&x * 2.0).unwrap();
    assert!(doubled.to_device(&Device::Cpu).is_err());
    assert!(doubled.sum_all().unwrap().to_scalar::<f32>().is_err());
}

#[test]
fn a_host_tensor_and_a_backend_tensor_do_not_meet_in_one_operation() {
    let context = Context::meta(None);
    let _guard = context.enter();

    let host = Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap();
    let device = input(8, DType::F32);

    for refused in [host.add(&device), device.add(&host)] {
        assert!(matches!(
            plain(&refused.unwrap_err()),
            uniserve_tensor::Error::DeviceMismatchBinaryOp { .. }
        ));
    }
    assert!(host.matmul(&device.reshape((8, 1)).unwrap()).is_err());
    assert!(matches!(operand(&host), Err(Error::ForeignTensor { .. })));
}

/// Moves two host tensors to the device: twelve bytes of FP32, then 300 bytes.
fn copy_inputs(context: &Context, value: u8) {
    context.reset();
    let _guard = context.enter();

    let floats = [f32::from(value), 2.0, 3.0];
    let small = Tensor::new(&floats, &Device::Cpu)
        .unwrap()
        .to_device(&meta())
        .unwrap();
    assert_eq!(small.dims(), [3]);
    assert_eq!(small.dtype(), DType::F32);

    // Whichever device is named, a meta context gives a meta tensor: one with no data to read.
    let large = Tensor::from_vec(vec![value; 300], 300, &Device::Cpu)
        .unwrap()
        .to_device(&cuda(0))
        .unwrap();
    assert_eq!(large.dims(), [300]);
    assert_eq!(large.dtype(), DType::U8);

    let refused = large.to_vec1::<u8>().unwrap_err();
    assert!(matches!(cause(&refused), Error::MetaRead));
}

#[test]
fn a_host_tensor_moved_to_the_device_writes_its_bytes_into_the_input_buffer() {
    // An ordinary heap buffer stands in for page-locked memory: a meta context only writes it.
    let mut memory = vec![0_u8; 1024];
    let context = Context::meta(None);
    // SAFETY: `memory` outlives the context's forwards and nothing else writes it meanwhile.
    unsafe { context.set_input_buffer(memory.as_mut_ptr(), memory.len() as u64) };

    copy_inputs(&context, 5);

    // Twelve bytes at offset 0, and 300 at the next multiple of 256.
    let floats: Vec<u8> = [5.0_f32, 2.0, 3.0]
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(memory[..12], floats);
    assert_eq!(memory[12], 0);
    assert_eq!(memory[256..556], [5; 300]);
    assert_eq!(memory[556], 0);
    assert_eq!(context.input_peak(), 556);

    // The same copies after a reset write the next values at the same offsets.
    copy_inputs(&context, 9);

    assert_eq!(memory[..4], 9.0_f32.to_ne_bytes());
    assert_eq!(memory[256..556], [9; 300]);
    assert_eq!(context.input_peak(), 556);

    // A meta context issues nothing, so it has no input copy to hand to a graph.
    assert!(context.take_input_copies().is_empty());
}

#[test]
fn a_kernel_argument_states_the_address_dimensions_and_strides_of_a_view() {
    const BASE: u64 = 0x1_0000;
    let matrix = retained(BASE, (4, 6), DType::F32, &cuda(0)).unwrap();

    let whole = operand(&matrix).unwrap();
    assert_eq!(whole.ptr.address(), BASE);
    assert_eq!(whole.dtype, DType::F32);
    assert_eq!(whole.shape.dims(), [4, 6]);
    assert_eq!(whole.strides.as_slice(), [6, 1]);

    // Columns 2..5: two FP32 elements further, at the matrix's own strides.
    let columns = operand(&matrix.narrow(1, 2, 3).unwrap()).unwrap();
    assert_eq!(columns.ptr.address(), BASE + 2 * 4);
    assert_eq!(columns.shape.dims(), [4, 3]);
    assert_eq!(columns.strides.as_slice(), [6, 1]);

    // Rows 1..3: one row of six elements further.
    let rows = operand(&matrix.narrow(0, 1, 2).unwrap()).unwrap();
    assert_eq!(rows.ptr.address(), BASE + 6 * 4);
    assert_eq!(rows.shape.dims(), [2, 6]);

    let transposed = operand(&matrix.t().unwrap()).unwrap();
    assert_eq!(transposed.ptr.address(), BASE);
    assert_eq!(transposed.shape.dims(), [6, 4]);
    assert_eq!(transposed.strides.as_slice(), [1, 6]);

    // A repeated axis is a zero stride.
    let row = retained(BASE, (1, 6), DType::BF16, &cuda(0)).unwrap();
    let repeated = operand(&row.broadcast_as((4, 6)).unwrap()).unwrap();
    assert_eq!(repeated.dtype, DType::BF16);
    assert_eq!(repeated.shape.dims(), [4, 6]);
    assert_eq!(repeated.strides.as_slice(), [0, 1]);

    assert_eq!(matrix.device().location(), cuda(0).location());
}

#[test]
fn a_tensor_of_more_axes_than_an_operand_carries_is_no_kernel_argument() {
    let deep = retained(0, [1_usize; 7].as_slice(), DType::F32, &meta()).unwrap();

    assert!(matches!(
        operand(&deep),
        Err(Error::Rank { rank: 7, limit: 6 })
    ));
}

#[test]
fn memory_is_lent_on_a_device_of_this_backend() {
    assert!(matches!(
        retained(0x1000, 8, DType::F32, &Device::Cpu),
        Err(Error::ForeignTensor { .. })
    ));

    // SAFETY: the tensor is meta, so the address is never read.
    let lent = unsafe { external(0, (2, 3), DType::I64, &meta()) }.unwrap();
    assert_eq!(lent.dims(), [2, 3]);
    assert_eq!(lent.dtype(), DType::I64);
    assert_eq!(lent.device().location(), META);

    // A 6-bit element has no byte layout to lend.
    assert!(matches!(
        retained(0x1000, 8, DType::F6E2M3, &cuda(0)),
        Err(Error::ElementType { .. })
    ));
}

#[test]
fn an_operation_without_a_kernel_is_refused_by_name() {
    let context = Context::meta(None);
    context.record_launches();
    let _guard = context.enter();

    let image = input((1, 2, 8, 8), DType::F32);
    let refused = image.avg_pool2d(2).unwrap_err();
    assert!(matches!(
        cause(&refused),
        Error::Unsupported {
            operation: "avg_pool2d",
            ..
        }
    ));
    assert!(image.max_pool2d(2).is_err());
    assert!(image.upsample_nearest2d(16, 16).is_err());

    // The product of integer matrices is no kernel of `gemm`.
    let counts = input((4, 8), DType::I64);
    let refused = counts.matmul(&input((8, 5), DType::I64)).unwrap_err();
    assert!(matches!(
        cause(&refused),
        Error::Unsupported {
            operation: "matmul",
            ..
        }
    ));

    // A transposed convolution with output padding is none of `conv_transpose1d`'s.
    let signal = input((1, 2, 16), DType::F32);
    let taps = input((2, 3, 4), DType::F32);
    assert!(signal.conv_transpose1d(&taps, 1, 1, 2, 1, 1).is_err());

    assert!(context.take_launches().is_empty());
}

#[test]
fn a_tensor_operation_with_no_context_installed_names_what_is_missing() {
    let x = input(8, DType::F32);

    let refused = x.relu().unwrap_err();
    assert!(matches!(cause(&refused), Error::NoContext { .. }));
}

#[test]
fn zeros_asked_through_either_device_are_meta_under_a_meta_context() {
    let table_of = launches(|| {
        for device in [cuda(0), meta()] {
            let zeros = Tensor::zeros((2, 3), DType::F32, &device)?;
            assert_eq!(zeros.dims(), [2, 3]);
            assert_eq!(zeros.device().location(), META);
        }

        let ones = Tensor::ones((2, 3), DType::I64, &meta())?;
        assert_eq!(ones.dtype(), DType::I64);
        Ok(())
    });

    assert_eq!(
        table_of,
        table(&[
            ("fill", format!("dtype_f32_rank_1_{TAIL}")),
            ("fill", format!("dtype_f32_rank_1_{TAIL}")),
            ("fill", format!("dtype_i64_rank_1_{TAIL}")),
        ])
    );
}
