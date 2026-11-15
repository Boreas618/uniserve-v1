//! Forwards under a meta context: tensor operations on meta tensors through the tensor
//! library's public interface, observed by what a caller of this crate can see — output shapes
//! and element types, the launch table, addresses, the allocator's peak and the input buffer.
//!
//! Nothing here touches a GPU: a meta context launches nothing.

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

use ops::{Context, Error, cuda, empty, external, meta, operand, retained};
use platform::{DevicePtr, DeviceRange};
use tensor::{D, DType, Device, DeviceLocation, Result, Shape, Tensor};

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
        .map(|(operator, key, ..)| (operator, key))
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
fn plain(error: &tensor::Error) -> &tensor::Error {
    match error {
        tensor::Error::WithBacktrace { inner, .. } => plain(inner),
        error => error,
    }
}

/// The backend's own error under a tensor library error.
fn cause(error: &tensor::Error) -> &Error {
    match plain(error) {
        tensor::Error::Backend(error) => error
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
fn integer_operations_and_a_sigmoid_state_their_kernels() {
    let table_of = launches(|| {
        let tokens = input((4, 8), DType::I64);
        let multipliers = input((4, 8), DType::I64);
        let sizes = input(8, DType::I64);

        // The n-gram hash of the reference: a wrapping product, an exclusive or, and the
        // remainder by one table size per column, which repeats along the rows.
        let mixed = tokens.mul(&multipliers)?.bitwise_xor(&tokens)?;
        let rows = mixed.broadcast_remainder(&sizes)?;
        assert_eq!(rows.dims(), [4, 8]);
        assert_eq!(rows.dtype(), DType::I64);
        assert_eq!(rows.device().location(), META);

        assert_eq!(tokens.bitwise_and(&mixed)?.dtype(), DType::I64);
        assert_eq!(tokens.bitwise_or(&mixed)?.dims(), [4, 8]);
        assert_eq!(tokens.bitwise_left_shift(&mixed)?.dims(), [4, 8]);

        // A transposed operand against a contiguous one traverses two axes.
        let counts = input((8, 4), DType::I64);
        let shifted = tokens.bitwise_right_shift(&counts.t()?)?;
        assert_eq!(shifted.dims(), [4, 8]);

        let words = input((4, 8), DType::U32);
        assert_eq!(words.remainder(&words)?.dtype(), DType::U32);

        // A scalar operand of an integer tensor is the operation of two tensors: the scalar is
        // one value repeated everywhere, which is one axis.
        let next = (&tokens + 1.0)?;
        assert_eq!(next.dtype(), DType::I64);
        let halved = (&tokens / 2.0)?;
        assert_eq!(halved.dims(), [4, 8]);
        let low = tokens.bitwise_and(0xffff_i64)?;
        assert_eq!(low.dims(), [4, 8]);

        let gate = input((4, 8), DType::BF16);
        let opened = gate.sigmoid()?;
        assert_eq!(opened.dims(), [4, 8]);
        assert_eq!(opened.dtype(), DType::BF16);
        Ok(())
    });

    let binary = |function: &str, dtype: &str, rank: u8| {
        (
            "elementwise_binary",
            format!("function_{function}_dtype_{dtype}_rank_{rank}_{TAIL}"),
        )
    };
    assert_eq!(
        table_of,
        table(&[
            binary("mul", "i64", 1),
            binary("bitwise_xor", "i64", 1),
            binary("remainder", "i64", 2),
            binary("bitwise_and", "i64", 1),
            binary("bitwise_or", "i64", 1),
            binary("bitwise_left_shift", "i64", 1),
            binary("bitwise_right_shift", "i64", 2),
            binary("remainder", "u32", 1),
            binary("add", "i64", 1),
            binary("div", "i64", 1),
            binary("bitwise_and", "i64", 1),
            (
                "elementwise_unary",
                format!("function_sigmoid_dtype_bf16_rank_1_{TAIL}")
            ),
        ])
    );
}

#[test]
fn an_integer_operation_on_floats_and_a_sigmoid_of_integers_are_refused() {
    let context = Context::meta(None);
    context.record_launches();
    let _guard = context.enter();

    let floats = input((4, 8), DType::F32);
    for refused in [
        floats.remainder(&floats),
        floats.bitwise_and(&floats),
        floats.bitwise_or(&floats),
        floats.bitwise_xor(&floats),
        floats.bitwise_left_shift(&floats),
        floats.bitwise_right_shift(&floats),
    ] {
        assert!(matches!(
            cause(&refused.unwrap_err()),
            Error::Launch {
                operator: "elementwise_binary",
                ..
            }
        ));
    }

    let counts = input((4, 8), DType::I64);
    assert!(matches!(
        cause(&counts.sigmoid().unwrap_err()),
        Error::Launch {
            operator: "elementwise_unary",
            ..
        }
    ));

    // A scalar that is no value of the element type never reaches the backend.
    assert!(matches!(
        plain(&(&counts * 0.5).unwrap_err()),
        tensor::Error::Msg(_)
    ));

    assert!(context.take_launches().is_empty());
}

#[test]
fn a_view_as_another_element_type_shares_the_storage_and_launches_nothing() {
    let context = Context::meta(None);
    context.record_launches();
    let _guard = context.enter();

    let floats = empty((4, 8), DType::F32).unwrap();
    assert_eq!(context.peak(), 256);

    let bits = floats.view(DType::U32).unwrap();
    assert_eq!(bits.dims(), [4, 8]);
    assert_eq!(bits.dtype(), DType::U32);
    assert_eq!(bits.device().location(), META);
    assert_eq!(address(&bits), address(&floats));

    // A strided tensor read as another element type keeps its shape, its strides and its start.
    {
        let window = floats.t().unwrap().narrow(0, 2, 3).unwrap();
        let signed = operand(&window.view(DType::I32).unwrap()).unwrap();
        assert_eq!(signed.dtype, DType::I32);
        assert_eq!(signed.ptr.address(), address(&window));
        assert_eq!(signed.dims(), [3, 4]);
        assert_eq!(signed.strides(), [1, 8]);
    }

    // An element type of another size would hold another number of elements.
    for dtype in [DType::I64, DType::BF16, DType::U8, DType::F4] {
        let refused = floats.view(dtype).unwrap_err();
        assert!(
            matches!(cause(&refused), Error::ElementType { .. }),
            "{dtype:?}"
        );
    }

    // Nothing was allocated and nothing launched.
    assert_eq!(context.peak(), 256);
    assert!(context.take_launches().is_empty());

    // The block is the two tensors' together: it is free once both are dropped.
    let first = address(&floats);
    drop(floats);
    let other = empty((4, 8), DType::F32).unwrap();
    assert_ne!(address(&other), first);
    assert_eq!(context.peak(), 512);

    drop(bits);
    let reused = empty((4, 8), DType::F32).unwrap();
    assert_eq!(address(&reused), first);
    assert_eq!(context.peak(), 512);

    // Lent memory is read as another element type where it is.
    let bytes = retained(0x4000, (2, 8), DType::U8, &cuda(0)).unwrap();
    let narrow = bytes.view(DType::F8E4M3).unwrap();
    assert_eq!(narrow.dtype(), DType::F8E4M3);
    assert_eq!(address(&narrow), 0x4000);
    assert_eq!(narrow.device().location(), cuda(0).location());
    assert_eq!(context.peak(), 512);
}

#[test]
fn an_fp8_tensor_is_zeroed_copied_gathered_and_selected_as_its_bytes() {
    let table_of = launches(|| {
        let table = input((16, 8), DType::F8E4M3);
        let ids = input(3, DType::U32);

        let zeros = Tensor::zeros((2, 8), DType::F8E4M3, &meta())?;
        assert_eq!(zeros.dtype(), DType::F8E4M3);

        let rows = table.index_select(&ids, 0)?;
        assert_eq!(rows.dims(), [3, 8]);
        assert_eq!(rows.dtype(), DType::F8E4M3);

        let columns = input((16, 2), DType::I64);
        assert_eq!(table.gather(&columns, 1)?.dims(), [16, 2]);

        let joined = Tensor::cat(&[&rows, &zeros], 0)?;
        assert_eq!(joined.dims(), [5, 8]);

        let transposed = table.t()?.contiguous()?;
        assert_eq!(transposed.dims(), [8, 16]);

        let cloned = table.copy()?;
        assert_eq!(cloned.dtype(), DType::F8E4M3);

        let mask = input((3, 8), DType::U8);
        let chosen = mask.where_cond(&rows, &rows.zeros_like()?)?;
        assert_eq!(chosen.dtype(), DType::F8E4M3);

        // Rows written back into a table, and the table's bytes read as they are.
        let target = empty((16, 8), DType::F8E4M3).unwrap();
        target.slice_set(&rows, 0, 4)?;
        assert_eq!(target.view(DType::U8)?.dtype(), DType::U8);
        Ok(())
    });

    let copy = |rank: u8| ("copy_strided", format!("dtype_f8e4m3_rank_{rank}_{TAIL}"));
    let fill = ("fill", format!("dtype_f8e4m3_rank_1_{TAIL}"));
    assert_eq!(
        table_of,
        table(&[
            fill.clone(),
            (
                "index",
                format!("function_gather_dtype_f8e4m3_indices_u32_rank_2_{TAIL}")
            ),
            (
                "index",
                format!("function_gather_dtype_f8e4m3_indices_i64_rank_2_{TAIL}")
            ),
            copy(1),
            copy(1),
            copy(2),
            copy(1),
            fill,
            (
                "elementwise_where",
                format!("dtype_f8e4m3_condition_u8_rank_1_{TAIL}")
            ),
            copy(1),
        ])
    );
}

#[test]
fn computing_with_an_fp8_tensor_and_moving_a_packed_one_are_refused() {
    let context = Context::meta(None);
    context.record_launches();
    let _guard = context.enter();

    let launch_refused = |result: Result<Tensor>, operator: &str| match cause(&result.unwrap_err())
    {
        Error::Launch {
            operator: named, ..
        } => assert_eq!(*named, operator),
        other => panic!("expected a refused launch of {operator}, not {other}"),
    };

    // An FP8 element moves and is never computed with: its arithmetic is a fused operator's.
    let narrow = input((4, 8), DType::F8E4M3);
    launch_refused(narrow.add(&narrow), "elementwise_binary");
    launch_refused(narrow.relu(), "elementwise_unary");
    launch_refused(narrow.sum_keepdim(1), "reduce");
    launch_refused(narrow.to_dtype(DType::F32), "cast");
    launch_refused(input((4, 8), DType::F32).to_dtype(DType::F8E4M3), "cast");

    let ids = input(4, DType::U32);
    launch_refused(narrow.index_add(&ids, &narrow, 0), "index");

    // Two NVFP4 elements share a byte: no primitive addresses one.
    let packed = input(16, DType::F4);
    launch_refused(packed.copy(), "copy_strided");
    launch_refused(Tensor::zeros(16, DType::F4, &meta()), "fill");
    assert!(matches!(
        cause(&packed.view(DType::U8).unwrap_err()),
        Error::ElementType { .. }
    ));

    // Its packed bytes, stated as the bytes they are, move like any other tensor.
    let bytes = input(8, DType::U8);
    assert_eq!(bytes.copy().unwrap().dims(), [8]);

    let launched = context.take_launches();
    // The copy an `index_add` starts with, and the copy of the bytes.
    assert_eq!(launched.len(), 2);
    assert!(
        launched
            .iter()
            .all(|(operator, ..)| *operator == "copy_strided")
    );
}

#[test]
fn operands_a_product_or_a_convolution_cannot_read_in_place_are_made_contiguous_first() {
    let table_of = launches(|| {
        // A row repeated over the batch has no stride between its rows; every other row has one.
        let row = input((1, 8), DType::F32);
        let weight = input((8, 5), DType::F32);
        let repeated = row.broadcast_as((4, 8))?.matmul(&weight)?;
        assert_eq!(repeated.dims(), [4, 5]);

        // Every other column of a wider weight is no run of elements in either order.
        let wide = input((8, 10), DType::F32);
        let stepped = Tensor::cat(&[&wide.narrow(1, 0, 1)?, &wide.narrow(1, 4, 1)?], 1)?;
        assert_eq!(stepped.dims(), [8, 2]);
        let x = input((4, 8), DType::F32);
        assert_eq!(x.matmul(&wide.narrow(1, 2, 5)?)?.dims(), [4, 5]);

        // `[n, l, c]` read as `[n, c, l]`, and a signal one sample repeated over the batch.
        let signal = input((1, 16, 2), DType::F32).transpose(1, 2)?;
        let filter = input((3, 2, 4), DType::F32);
        assert_eq!(signal.conv1d(&filter, 1, 2, 1, 1)?.dims(), [1, 3, 8]);

        let batch = input((1, 2, 16), DType::F32).broadcast_as((4, 2, 16))?;
        assert_eq!(batch.conv1d(&filter, 1, 2, 1, 1)?.dims(), [4, 3, 8]);
        Ok(())
    });

    let copy = |rank: u8| ("copy_strided", format!("dtype_f32_rank_{rank}_{TAIL}"));
    let gemm = (
        "gemm",
        "input_f32_weight_f32_accumulate_f32_output_f32_order_kxn_epilogue_none".to_owned(),
    );
    let conv = ("conv", "input_f32_weight_f32_output_f32".to_owned());
    assert_eq!(
        table_of,
        table(&[
            // The repeated row copied into four rows, then the product.
            copy(2),
            gemm.clone(),
            // The two columns of the concatenation.
            copy(1),
            copy(1),
            // A window of a wider weight keeps the wider weight's row stride: read in place.
            gemm,
            copy(2),
            conv.clone(),
            copy(2),
            conv,
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
fn convolutions_launch_the_convolution_operator() {
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

    // Every rank and direction is one operator of one key: the call carries the rest.
    let key = "input_f32_weight_f32_output_f32".to_owned();
    assert_eq!(
        table_of,
        table(&[("conv", key.clone()), ("conv", key.clone()), ("conv", key),])
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
    let region = DeviceRange::new(DevicePtr::from_address(0x1000_0000), 1 << 20);
    let context = Context::meta(Some(region));

    let first = run(&context);
    let second = run(&context);

    assert_eq!(first, second);
    assert!(first.iter().all(|address| *address >= 0x1000_0000));
}

#[test]
fn a_counting_context_places_and_peaks_as_a_context_over_a_region_does() {
    const BASE: u64 = 0x1000_0000;
    let region = DeviceRange::new(DevicePtr::from_address(BASE), 1 << 20);
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
            tensor::Error::DeviceMismatchBinaryOp { .. }
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

    // A meta context issues nothing and states the input copies a capture would record: the
    // offsets and the sizes of the last forward's copies.
    let copies: Vec<(u64, u64)> = context
        .take_input_copies()
        .into_iter()
        .map(|(offset, _, bytes)| (offset, bytes))
        .collect();
    assert_eq!(copies, [(0, 12), (256, 300)]);
}

#[test]
fn a_kernel_argument_states_the_address_dimensions_and_strides_of_a_view() {
    const BASE: u64 = 0x1_0000;
    let matrix = retained(BASE, (4, 6), DType::F32, &cuda(0)).unwrap();

    let whole = operand(&matrix).unwrap();
    assert_eq!(whole.ptr.address(), BASE);
    assert_eq!(whole.dtype, DType::F32);
    assert_eq!(whole.dims(), [4, 6]);
    assert_eq!(whole.strides(), [6, 1]);

    // Columns 2..5: two FP32 elements further, at the matrix's own strides.
    let columns = operand(&matrix.narrow(1, 2, 3).unwrap()).unwrap();
    assert_eq!(columns.ptr.address(), BASE + 2 * 4);
    assert_eq!(columns.dims(), [4, 3]);
    assert_eq!(columns.strides(), [6, 1]);

    // Rows 1..3: one row of six elements further.
    let rows = operand(&matrix.narrow(0, 1, 2).unwrap()).unwrap();
    assert_eq!(rows.ptr.address(), BASE + 6 * 4);
    assert_eq!(rows.dims(), [2, 6]);

    let transposed = operand(&matrix.t().unwrap()).unwrap();
    assert_eq!(transposed.ptr.address(), BASE);
    assert_eq!(transposed.dims(), [6, 4]);
    assert_eq!(transposed.strides(), [1, 6]);

    // A repeated axis is a zero stride.
    let row = retained(BASE, (1, 6), DType::BF16, &cuda(0)).unwrap();
    let repeated = operand(&row.broadcast_as((4, 6)).unwrap()).unwrap();
    assert_eq!(repeated.dtype, DType::BF16);
    assert_eq!(repeated.dims(), [4, 6]);
    assert_eq!(repeated.strides(), [0, 1]);

    assert_eq!(matrix.device().location(), cuda(0).location());
}

#[test]
fn a_tensor_of_more_axes_than_an_operand_carries_is_no_kernel_argument() {
    // An operand carries as many axes as the tensor library's strided operations take.
    let limit = tensor::nditer::MAX_DIMS;
    let deep = retained(0, vec![1_usize; limit + 1].as_slice(), DType::F32, &meta()).unwrap();

    assert!(matches!(
        operand(&deep),
        Err(Error::Rank { rank, limit: refused }) if rank == limit + 1 && refused == limit
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

#[test]
fn a_host_to_device_copy_takes_memory_no_earlier_tensor_of_the_forward_had() {
    let context = Context::meta(None);
    context.reset();
    let _guard = context.enter();

    // A tensor of four blocks is computed and dropped, so its addresses are free again.
    let dropped = input((4, 64), DType::F32).affine(2.0, 0.0).unwrap();
    assert_eq!(context.peak(), 1024);
    drop(dropped);

    // The copy's destination must hold its value from before the replay's first kernel, which
    // an address the dropped tensor's kernel writes cannot: it lies past everything so far.
    let copied = Tensor::from_vec(vec![1_u8; 256], 256, &Device::Cpu)
        .unwrap()
        .to_device(&meta())
        .unwrap();
    assert_eq!(context.peak(), 1280);

    // An ordinary tensor still reuses what the dropped one gave back.
    let reused = input((4, 64), DType::F32).affine(2.0, 0.0).unwrap();
    assert_eq!(context.peak(), 1280);
    drop((copied, reused));
}

#[test]
fn a_tensor_moved_to_the_device_under_a_meta_context_is_on_the_meta_device() {
    let context = Context::meta(None);
    context.reset();
    let _guard = context.enter();

    let moved = Tensor::from_vec(vec![1_u8; 8], 8, &Device::Cpu)
        .unwrap()
        .to_device(&cuda(0))
        .unwrap();

    assert!(moved.device().same_device(&meta()));
    assert_eq!(
        format!("{:?}", moved.device().location()),
        format!("{:?}", meta().location())
    );
}

#[test]
fn a_weight_and_a_meta_tensor_concatenate() {
    let context = Context::meta(None);
    context.reset();
    let _guard = context.enter();

    let weight = retained(0x1000, (2, 4), DType::F32, &cuda(0)).unwrap();
    let joined = Tensor::cat(&[&weight, &input((3, 4), DType::F32)], 0).unwrap();

    assert_eq!(joined.dims(), [5, 4]);
    assert!(joined.device().same_device(&meta()));
}
