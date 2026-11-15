//! The CPU backend's benchmarks as one table: each operation, for each element type and each
//! shape or layout it is measured at.

use criterion::{BatchSize, Criterion, Throughput, criterion_group};
use half::{bf16, f16};
use std::hint::black_box;
use tensor::cpu::kernels::VecOps;
use tensor::{DType, Device, Tensor};

use super::{Case, Run, bytes_of, measure, uniform};

/// The floating element types most operations are measured for.
const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

/// The elements of the one-matrix inputs: `(1, 1024, 1024)`.
const MATRIX: (usize, usize, usize) = (1, 1024, 1024);
const MATRIX_ELEMENTS: usize = MATRIX.0 * MATRIX.1 * MATRIX.2;

/// The matrix products: the lhs and rhs shapes of common GEMM scenarios.
const MATMUL_SHAPES: [(&str, &[usize], &[usize]); 7] = [
    // A matrix-vector product.
    ("gemv", &[1, 1, 2048], &[1, 2048, 2048]),
    // Multi-head attention scores.
    ("attn_4d_small", &[484, 6, 144, 32], &[484, 6, 32, 144]),
    ("attn_4d_large", &[121, 24, 144, 32], &[121, 24, 32, 144]),
    // Square matrices.
    ("square_512", &[512, 512], &[512, 512]),
    ("square_1024", &[1024, 1024], &[1024, 1024]),
    // A batch of attention scores.
    ("batch_1000", &[1000, 144, 32], &[1000, 32, 144]),
    // A transformer feed-forward projection.
    ("linear_large", &[17424, 768], &[768, 3072]),
];

/// The shape of the tensors `cat` joins, `[batch, seq, hidden]`, and how many it joins.
const CAT_SHAPE: (usize, usize, usize) = (4, 128, 64);
const CAT_COUNT: usize = 8;

/// The shape `contiguous` copies, `[batch, seq, head_dim]`, and the padding of its narrowed view.
const KV_SHAPE: (usize, usize, usize) = (8, 512, 64);
const KV_PAD: usize = 16;

/// The causal mask of `where_cond`'s masked fill: `[batch, heads, seq, seq]`.
const CAUSAL_SHAPE: (usize, usize, usize, usize) = (1, 4, 512, 512);

/// The narrowed innermost axis of `where_cond`: `[batch, seq, padded]` narrowed to `inner`.
const NARROW_SHAPE: (usize, usize, usize) = (256, 512, 8);
const NARROW_INNER: usize = 3;

/// The lengths of the vector dot products.
const DOT_LENGTHS: [usize; 4] = [64, 128, 256, 512];

/// `0, 1, ..` as `dtype` in `shape`.
fn arange(count: usize, shape: &[usize], dtype: DType, device: &Device) -> Tensor {
    Tensor::arange(0f32, count as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .reshape(shape)
        .unwrap()
}

/// A U8 condition of `shape`, the `index`-th element `value(index)`.
fn condition(shape: &[usize], value: impl Fn(usize) -> u8, device: &Device) -> Tensor {
    let count = shape.iter().product::<usize>();
    let values: Vec<u8> = (0..count).map(value).collect();
    Tensor::from_vec(values, shape, device).unwrap()
}

/// The element-wise, broadcast and reduction cases of `dtype`.
fn floating_cases(dtype: DType, device: &Device, cases: &mut Vec<Case>) {
    let name = |operation: &str| format!("{operation}_{}", dtype.as_str());
    let matrix = [MATRIX.0, MATRIX.1, MATRIX.2];

    cases.push(Case::tensors(
        name("binary_mul"),
        vec![
            arange(MATRIX_ELEMENTS, &matrix, dtype, device),
            arange(MATRIX_ELEMENTS, &matrix, dtype, device),
        ],
        |t| t[0].mul(&t[1]).unwrap(),
    ));

    // Two tensors of one shape; a conv2d output and its bias; a tensor and a scalar.
    let ones = |shape: &[usize]| Tensor::ones(shape, dtype, device).unwrap();
    let broadcast_add: fn(&[Tensor]) -> Tensor = |t| t[0].broadcast_add(&t[1]).unwrap();
    cases.push(Case::tensors(
        name("broadcast_add_contiguous"),
        vec![ones(&[128, 126, 126]), ones(&[128, 126, 126])],
        broadcast_add,
    ));
    cases.push(Case::tensors(
        name("broadcast_add"),
        vec![ones(&[1, 1, 126, 126]), ones(&[1, 128, 1, 1])],
        broadcast_add,
    ));
    cases.push(Case::tensors(
        name("broadcast_scalar_add"),
        vec![ones(&[1, 256, 64]), ones(&[1])],
        broadcast_add,
    ));

    cases.push(Case::tensors(
        name("conv_transpose2d"),
        vec![
            arange(10000, &[1, 4, 50, 50], dtype, device),
            arange(100, &[4, 1, 5, 5], dtype, device),
        ],
        |t| t[0].conv_transpose2d(&t[1], 1, 0, 1, 2).unwrap(),
    ));

    // Along the last axis of a contiguous tensor, and of its transpose.
    let values = uniform(-1000., 1000., MATRIX, dtype, device);
    for (layout, a) in [
        ("", values.clone()),
        ("_strided", values.transpose(0, 2).unwrap()),
    ] {
        cases.push(Case::tensors(
            format!("reduce_{}{layout}", dtype.as_str()),
            vec![a.clone()],
            |t| t[0].sum_keepdim(2).unwrap(),
        ));
        cases.push(Case::tensors(
            format!("arg_reduce_{}{layout}", dtype.as_str()),
            vec![a],
            |t| t[0].argmin_keepdim(2).unwrap(),
        ));
    }

    cases.push(Case::tensors(
        name("sqrt"),
        vec![arange(MATRIX_ELEMENTS, &matrix, dtype, device)],
        |t| t[0].sqrt().unwrap(),
    ));
    let to = if dtype == DType::F32 {
        DType::F16
    } else {
        DType::F32
    };
    let cast: fn(&[Tensor]) -> Tensor = match to {
        DType::F16 => |t| t[0].to_dtype(DType::F16).unwrap(),
        _ => |t| t[0].to_dtype(DType::F32).unwrap(),
    };
    cases.push(Case::tensors(
        format!("cast_{}_{}", dtype.as_str(), to.as_str()),
        vec![arange(MATRIX_ELEMENTS, &matrix, dtype, device)],
        cast,
    ));

    where_cond_cases(dtype, device, cases);
}

/// `where_cond` over the layouts of a mask: alternating, causal, a broadcast causal mask, a
/// transposed operand and a narrowed innermost axis.
fn where_cond_cases(dtype: DType, device: &Device, cases: &mut Vec<Case>) {
    let name = |layout: &str| format!("where_cond{layout}_{}", dtype.as_str());
    let select: fn(&[Tensor]) -> Tensor = |t| t[0].where_cond(&t[1], &t[2]).unwrap();
    let scalar = |value: f32, shape: &[usize]| {
        Tensor::new(value, device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .broadcast_as(shape)
            .unwrap()
    };
    let zeros = |shape: &[usize]| Tensor::zeros(shape, dtype, device).unwrap();
    let alternating = |index: usize| (index % 2) as u8;
    let (b, h, s, _) = CAUSAL_SHAPE;
    let causal = |index: usize| u8::from(index % s > (index / s) % s);

    let matrix = [MATRIX.0, MATRIX.1, MATRIX.2];
    cases.push(Case::tensors(
        name(""),
        vec![
            condition(&matrix, alternating, device),
            Tensor::ones(&matrix[..], dtype, device).unwrap(),
            zeros(&matrix),
        ],
        select,
    ));

    let heads = [b, h, s, s];
    cases.push(Case::tensors(
        name("_masked_fill"),
        vec![
            condition(&heads, causal, device),
            scalar(1., &heads),
            zeros(&heads),
        ],
        select,
    ));
    cases.push(Case::tensors(
        name("_masked_fill_bcast"),
        vec![
            condition(&[1, 1, s, s], causal, device)
                .broadcast_as(&heads[..])
                .unwrap(),
            scalar(1., &heads),
            zeros(&heads),
        ],
        select,
    ));

    let (m, k) = (MATRIX.1, MATRIX.2);
    cases.push(Case::tensors(
        name("_transposed"),
        vec![
            condition(&[m, k], alternating, device),
            Tensor::ones((m, k), dtype, device).unwrap(),
            zeros(&[k, m]).transpose(0, 1).unwrap(),
        ],
        select,
    ));

    let (nb, ns, padded) = NARROW_SHAPE;
    let padded = [nb, ns, padded];
    cases.push(Case::tensors(
        name("_narrow_inner"),
        vec![
            condition(&padded, alternating, device)
                .narrow(2, 0, NARROW_INNER)
                .unwrap(),
            scalar(0., &[nb, ns, NARROW_INNER]),
            zeros(&padded).narrow(2, 0, NARROW_INNER).unwrap(),
        ],
        select,
    ));
}

/// Every case of the table on `device`.
fn cases(device: &Device) -> Vec<Case> {
    let mut cases = Vec::new();

    for dtype in [DType::F32, DType::F16, DType::BF16, DType::F8E4M3] {
        cases.push(Case::tensors(
            format!("affine_{}", dtype.as_str()),
            vec![Tensor::zeros(MATRIX, dtype, device).unwrap()],
            |t| t[0].affine(12.34, 56.78).unwrap(),
        ));
    }
    for dtype in FLOATS {
        floating_cases(dtype, device, &mut cases);
    }

    // `cat` of contiguous tensors, and of transposed ones, along the first two axes.
    let (batch, seq, hidden) = CAT_SHAPE;
    let contiguous: Vec<Tensor> = (0..CAT_COUNT)
        .map(|_| Tensor::zeros(CAT_SHAPE, DType::F32, device).unwrap())
        .collect();
    let strided: Vec<Tensor> = (0..CAT_COUNT)
        .map(|_| {
            Tensor::zeros((hidden, seq, batch), DType::F32, device)
                .unwrap()
                .transpose(0, 2)
                .unwrap()
        })
        .collect();
    for (layout, tensors) in [("", contiguous), ("_strided", strided)] {
        cases.push(Case::tensors(
            format!("cat{layout}_dim0"),
            tensors.clone(),
            |t| Tensor::cat(t, 0).unwrap(),
        ));
        cases.push(Case::tensors(format!("cat{layout}_dim1"), tensors, |t| {
            Tensor::cat(t, 1).unwrap()
        }));
    }

    // `contiguous` of a contiguous tensor (a forced copy), of a narrowed one whose rows are
    // contiguous blocks, and of a transposed one.
    let (batch, seq, head) = KV_SHAPE;
    let kv = Tensor::zeros(KV_SHAPE, DType::F32, device).unwrap();
    cases.push(Case::tensors("contiguous", vec![kv.clone()], |t| {
        t[0].force_contiguous().unwrap()
    }));
    let padded = Tensor::zeros((batch, seq + KV_PAD, head), DType::F32, device).unwrap();
    cases.push(Case::tensors(
        "contiguous_uniform",
        vec![padded.narrow(1, 0, seq).unwrap()],
        |t| t[0].contiguous().unwrap(),
    ));
    cases.push(Case::tensors(
        "contiguous_strided",
        vec![kv.transpose(0, 1).unwrap()],
        |t| t[0].contiguous().unwrap(),
    ));

    // A host attention mask, `[batch, query, key]`, copied into a tensor.
    let mask = vec![vec![vec![0f32; 1024]; 1]; 128];
    let mask_tensor = Tensor::new(mask.clone(), device).unwrap();
    let device_for_copy = device.clone();
    cases.push(Case {
        name: "copy_mask".to_owned(),
        work: Throughput::Bytes(2 * bytes_of(&mask_tensor)),
        prepare: Box::new(move || {
            let (mask, device) = (mask.clone(), device_for_copy.clone());
            Box::new(move || Tensor::new(mask, &device).unwrap()) as Run
        }),
    });

    // A matrix product counts two floating-point operations per multiply-add.
    for (shape, lhs, rhs) in MATMUL_SHAPES {
        let batch = lhs[..lhs.len() - 2].iter().product::<usize>();
        let (m, k, n) = (lhs[lhs.len() - 2], lhs[lhs.len() - 1], rhs[rhs.len() - 1]);
        cases.push(Case {
            work: Throughput::Elements((2 * batch * m * k * n) as u64),
            ..Case::tensors(
                format!("matmul_{shape}"),
                vec![
                    Tensor::zeros(lhs, DType::F32, device).unwrap(),
                    Tensor::zeros(rhs, DType::F32, device).unwrap(),
                ],
                |t| t[0].broadcast_matmul(&t[1]).unwrap(),
            )
        });
    }

    cases
}

/// The dot product kernel of `T` over vectors of `k` elements, counted in elements.
fn vec_dot<T: VecOps + From<u8>>(c: &mut Criterion, dtype: DType, k: usize) {
    let a: Vec<T> = (0..k).map(|i| T::from((i % 7) as u8)).collect();
    let b: Vec<T> = (0..k).map(|i| T::from((i % 5) as u8)).collect();

    let mut group = c.benchmark_group(format!("cpu_vec_dot_{}_k{k}", dtype.as_str()));
    group.throughput(Throughput::Elements(k as u64));
    group.bench_function("iter", |bench| {
        bench.iter_batched(
            T::zero,
            // SAFETY: `a` and `b` hold `k` elements each, and `out` is one element.
            |mut out| unsafe {
                T::vec_dot(
                    black_box(a.as_ptr()),
                    black_box(b.as_ptr()),
                    &mut out,
                    black_box(k),
                );
                out
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    for k in DOT_LENGTHS {
        vec_dot::<f32>(c, DType::F32, k);
        vec_dot::<f16>(c, DType::F16, k);
        vec_dot::<bf16>(c, DType::BF16, k);
    }

    let device = Device::Cpu;
    for case in cases(&device) {
        measure(c, &device, case);
    }
}

criterion_group!(benches, criterion_benchmark);
