//! Direct ports of the reference formulas of a hyper-connected residual with manifold-constrained
//! connections (mHC) and of a hashed n-gram memory, in FP64 over plain slices, rounding into the
//! streams' element type exactly where the reference implementation stores a tensor of it.
//!
//! The ports are what the library's functions and modules are compared with on the CPU, and the
//! sums are FP64, so a comparison measures the library's own FP32 arithmetic and its roundings
//! against the formula. Every token is independent and every stream is `[hidden]`; a tensor is
//! row-major `[tokens, streams, hidden]`.
//!
//! ```text
//! mixes    r = 1/√(mean(flat(h)²) + ε_rms);  z = (flat(h)·fnᵀ)·r
//!          pre = σ(z[:n]·s₀ + b) + ε;  post = 2σ(z[n:2n]·s₁ + b);
//!          comb = sinkhorn(z[2n:]·s₂ + b)                     row softmax + ε, then columns,
//!                                                            then rows and columns again
//! post     h'_k = round(post_k·y + Σ_j comb[j, k]·h_j)
//! pre      x = round(Σ_j pre_j·h_j);  round(x/√(mean(x²) + ε)·w)
//! gate     dot = Σ(h·q·κ·key)/√(mean(h²) + ε)/√(mean(key²) + ε)/√d
//!          round(h + σ(copysign(√max(|dot|, 10⁻⁶), dot))·value)
//! hash     the rolling exclusive-or of each normalized id times its multiplier, modulo each
//!          head's prime, plus its offset
//! ```

use nn::{DType, Device, Tensor};
use ops::tolerance::Tolerance;

use crate::values::{decode, encode};

/// `values` rounded to FP32 and then into `dtype`, as FP32: the reference's store of an FP32
/// result into a tensor of `dtype`.
pub fn round(dtype: DType, values: &[f64]) -> Vec<f32> {
    let wide: Vec<f32> = values.iter().map(|&value| value as f32).collect();
    if dtype == DType::F32 {
        return wide;
    }
    decode(dtype, &encode(dtype, &wide))
}

/// One connection's mixes, per token: `pre` and `post` `[tokens·n]`, `comb` `[tokens·n·n]`.
#[derive(Clone, Debug)]
pub struct Mixes {
    pub pre: Vec<f64>,
    pub post: Vec<f64>,
    pub comb: Vec<f64>,
}

/// One connection's mixing parameters.
#[derive(Clone, Copy, Debug)]
pub struct Mixing<'a> {
    /// `[n·(n + 2), n·d]`.
    pub projection: &'a [f32],
    /// `[3]`.
    pub scale: &'a [f32],
    /// `[n·(n + 2)]`.
    pub base: &'a [f32],
    pub iterations: usize,
    pub rms_eps: f64,
    pub eps: f64,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The mixes of `h`, `[tokens, n, d]` (`Block.hc_mixes` and `hc_split_sinkhorn`).
pub fn mixes(h: &[f32], tokens: usize, n: usize, d: usize, mixing: Mixing<'_>) -> Mixes {
    let width = n * d;
    let count = n * (n + 2);
    let (mut pre, mut post, mut comb) = (
        Vec::with_capacity(tokens * n),
        Vec::with_capacity(tokens * n),
        Vec::with_capacity(tokens * n * n),
    );

    for token in 0..tokens {
        let row = &h[token * width..(token + 1) * width];
        let mean = row
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / width as f64;
        let inverse = 1.0 / (mean + mixing.rms_eps).sqrt();
        let z: Vec<f64> = (0..count)
            .map(|mix| {
                let weights = &mixing.projection[mix * width..(mix + 1) * width];
                let dot: f64 = row
                    .iter()
                    .zip(weights)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum();
                dot * inverse
            })
            .collect();
        let affine = |mix: usize, group: usize| {
            z[mix] * f64::from(mixing.scale[group]) + f64::from(mixing.base[mix])
        };

        for j in 0..n {
            pre.push(sigmoid(affine(j, 0)) + mixing.eps);
            post.push(2.0 * sigmoid(affine(n + j, 1)));
        }

        let mut matrix: Vec<f64> = (0..n * n).map(|index| affine(2 * n + index, 2)).collect();
        for j in 0..n {
            let cells = &mut matrix[j * n..(j + 1) * n];
            let largest = cells.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = cells.iter().map(|&x| (x - largest).exp()).sum();
            for cell in cells.iter_mut() {
                *cell = (*cell - largest).exp() / total + mixing.eps;
            }
        }
        let columns = |matrix: &mut Vec<f64>| {
            for k in 0..n {
                let total: f64 = (0..n).map(|j| matrix[j * n + k]).sum();
                for j in 0..n {
                    matrix[j * n + k] /= total + mixing.eps;
                }
            }
        };
        columns(&mut matrix);
        for _ in 1..mixing.iterations {
            for j in 0..n {
                let total: f64 = matrix[j * n..(j + 1) * n].iter().sum();
                for cell in &mut matrix[j * n..(j + 1) * n] {
                    *cell /= total + mixing.eps;
                }
            }
            columns(&mut matrix);
        }
        comb.extend(matrix);
    }

    Mixes { pre, post, comb }
}

/// `y`, `[tokens, d]`, written back into `h` (`Block.hc_post`), stored in `dtype`.
pub fn post(
    h: &[f32],
    y: &[f32],
    mixes: &Mixes,
    (tokens, n, d): (usize, usize, usize),
    dtype: DType,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens * n * d);
    for token in 0..tokens {
        for k in 0..n {
            for c in 0..d {
                let mixed: f64 = (0..n)
                    .map(|j| {
                        mixes.comb[(token * n + j) * n + k] * f64::from(h[(token * n + j) * d + c])
                    })
                    .sum();
                out.push(mixes.post[token * n + k] * f64::from(y[token * d + c]) + mixed);
            }
        }
    }
    round(dtype, &out)
}

/// `h` read with `pre`, `[tokens·n]`, stored in `dtype` and RMS-normalized with `norm`
/// (`Block.hc_pre`, then `RMSNorm`), stored in `dtype` again.
pub fn pre(
    h: &[f32],
    pre: &[f64],
    norm: &[f32],
    (tokens, n, d): (usize, usize, usize),
    eps: f64,
    dtype: DType,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens * d);
    for token in 0..tokens {
        let collapsed: Vec<f64> = (0..d)
            .map(|c| {
                (0..n)
                    .map(|j| pre[token * n + j] * f64::from(h[(token * n + j) * d + c]))
                    .sum()
            })
            .collect();
        let collapsed: Vec<f64> = round(dtype, &collapsed)
            .into_iter()
            .map(f64::from)
            .collect();
        let mean = collapsed.iter().map(|x| x * x).sum::<f64>() / d as f64;
        let inverse = 1.0 / (mean + eps).sqrt();
        out.extend(
            collapsed
                .iter()
                .zip(norm)
                .map(|(&x, &w)| f64::from(w) * x * inverse),
        );
    }
    round(dtype, &out)
}

/// The residual of the reference's decoder, carried from one sub-layer to the next in the
/// shifted form (`Transformer.forward`, `Block.forward`): every sub-layer's mixes from the
/// streams it reaches, its input read with the pre-mix of the connection before.
pub struct Residual {
    /// `[tokens, n, d]`, before the pending output is written back.
    h: Vec<f32>,
    /// `[tokens·n]`: the pre-mix the next input is read with.
    pre: Vec<f64>,
    /// The last sub-layer's output and the mixes it was read with.
    pending: Option<(Vec<f32>, Mixes)>,
    shape: (usize, usize, usize),
    dtype: DType,
}

impl Residual {
    /// `embedded`, `[tokens, d]`, in every one of `n` streams, read first with the pre-mix that
    /// selects stream 0 (`make_identity_pre_mix`).
    pub fn expand(embedded: &[f32], (tokens, n, d): (usize, usize, usize), dtype: DType) -> Self {
        Self {
            h: embedded.chunks(d).flat_map(|row| row.repeat(n)).collect(),
            pre: (0..tokens * n)
                .map(|index| if index % n == 0 { 1.0 } else { 0.0 })
                .collect(),
            pending: None,
            shape: (tokens, n, d),
            dtype,
        }
    }

    /// One sub-layer: the pending output written back, the mixes of `mixing`, the input read
    /// and normalized with `norm` and `eps`, and `body`'s output of it left pending.
    pub fn sublayer(
        &mut self,
        mixing: Mixing<'_>,
        norm: &[f32],
        eps: f64,
        body: impl FnOnce(&[f32]) -> Vec<f32>,
    ) {
        let (tokens, n, d) = self.shape;
        self.streams();
        let mixes = mixes(&self.h, tokens, n, d, mixing);
        let input = pre(&self.h, &self.pre, norm, self.shape, eps, self.dtype);
        self.pre = mixes.pre.clone();
        self.pending = Some((body(&input), mixes));
    }

    /// The streams with the pending output written back.
    pub fn streams(&mut self) -> &[f32] {
        if let Some((output, mixes)) = self.pending.take() {
            self.h = post(&self.h, &output, &mixes, self.shape, self.dtype);
        }
        &self.h
    }

    /// Replaces the streams, which a layer between two sub-layers computed from [`Self::streams`].
    pub fn set_streams(&mut self, streams: Vec<f32>) {
        assert!(self.pending.is_none() && streams.len() == self.h.len());
        self.h = streams;
    }

    /// The streams collapsed with the last pre-mix into the output norm `norm`
    /// (`hc_pre` and `Transformer.norm`).
    pub fn collapse(mut self, norm: &[f32], eps: f64) -> Vec<f32> {
        self.streams();
        pre(&self.h, &self.pre, norm, self.shape, eps, self.dtype)
    }
}

/// `h` gated by `projected`'s keys and value (`Engram.forward` after its projection), stored in
/// `dtype`: `projected` is `[tokens, (n + 1)·d]`, `query` and `key` `[n, d]`.
#[allow(clippy::too_many_arguments)]
pub fn gate(
    h: &[f32],
    projected: &[f32],
    query: &[f32],
    key: &[f32],
    (tokens, n, d): (usize, usize, usize),
    eps: f64,
    dtype: DType,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens * n * d);
    let scale = (d as f64).powf(-0.5);
    for token in 0..tokens {
        let row = &projected[token * (n + 1) * d..(token + 1) * (n + 1) * d];
        let value = &row[n * d..];
        for k in 0..n {
            let stream = &h[(token * n + k) * d..(token * n + k + 1) * d];
            let keys = &row[k * d..(k + 1) * d];
            let mean = |x: &[f32]| x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / d as f64;
            let inverse = 1.0 / (mean(stream) + eps).sqrt() / (mean(keys) + eps).sqrt();
            let dot: f64 = (0..d)
                .map(|c| {
                    f64::from(stream[c])
                        * f64::from(query[k * d + c])
                        * f64::from(key[k * d + c])
                        * f64::from(keys[c])
                })
                .sum::<f64>()
                * inverse
                * scale;
            let gate = sigmoid(dot.abs().max(1e-6).sqrt().copysign(dot));
            out.extend((0..d).map(|c| f64::from(stream[c]) + gate * f64::from(value[c])));
        }
    }
    round(dtype, &out)
}

/// The table rows of each token's n-grams (`NgramHashState.forward`): `context` is
/// `[tokens, order]` vocabulary ids, oldest first, then the token's own; `moduli` and
/// `offsets` `[(order − 1)·heads]`.
pub fn hash(
    context: &[u32],
    order: usize,
    normalize: &[i32],
    multipliers: &[i64],
    moduli: &[i64],
    offsets: &[i64],
) -> Vec<i64> {
    let heads = moduli.len() / (order - 1);
    let mut rows = Vec::new();
    for ids in context.chunks(order) {
        let mut rolling = 0_i64;
        for back in 0..order {
            let id = ids[order - 1 - back] as usize;
            let term = i64::from(normalize[id.min(normalize.len() - 1)]) * multipliers[back];
            if back == 0 {
                rolling = term;
                continue;
            }
            rolling ^= term;
            for head in 0..heads {
                let column = (back - 1) * heads + head;
                rows.push(rolling.rem_euclid(moduli[column]) + offsets[column]);
            }
        }
    }
    rows
}

/// The MXFP8 rows of `ids` from a table of `row`-element rows that opens at row `first`, exactly
/// (`ParallelEngramEmbedding.forward` before its narrowing): `values` the E4M3 bytes, `scales`
/// the UE8M0 bytes, one per 32 values; an id outside the table reads zeros.
pub fn mxfp8(values: &[u8], scales: &[u8], row: usize, ids: &[i64], first: i64) -> Vec<f32> {
    let decoded = Tensor::from_vec(values.to_vec(), values.len(), &Device::Cpu)
        .and_then(|bytes| bytes.view(DType::F8E4M3))
        .and_then(|e4m3| e4m3.to_dtype(DType::F32))
        .and_then(|values| values.to_vec1::<f32>())
        .unwrap_or_else(|error| panic!("{} E4M3 bytes: {error}", values.len()));
    let rows = (values.len() / row) as i64;
    let mut out = Vec::with_capacity(ids.len() * row);
    for &id in ids {
        let local = id - first;
        if local < 0 || local >= rows {
            out.extend(std::iter::repeat_n(0.0, row));
            continue;
        }
        let local = local as usize;
        for c in 0..row {
            let exponent = i32::from(scales[local * row / 32 + c / 32]) - 127;
            out.push(decoded[local * row + c] * 2_f32.powi(exponent));
        }
    }
    out
}

/// Asserts that every element of `actual` is within `tolerance` of `expected`'s, an output whose
/// largest magnitude is the peak of a bound that cancels, naming `what` and the worst element.
pub fn assert_within(what: &str, actual: &[f32], expected: &[f32], tolerance: Tolerance) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: the element counts differ"
    );
    let peak = expected
        .iter()
        .fold(0_f32, |peak, value| peak.max(value.abs()));
    let mut worst = (0.0_f64, 0);

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            tolerance.allows_within(peak, expected, actual),
            "{what}: element {index} is {actual} against {expected} (peak {peak}), beyond \
             {tolerance:?}"
        );
        let difference = (f64::from(actual) - f64::from(expected)).abs();
        if difference > worst.0 {
            worst = (difference, index);
        }
    }
    eprintln!(
        "{what}: largest difference {:.3e} at element {} (peak {peak:.3e}), within {tolerance:?}",
        worst.0, worst.1
    );
}

/// `values` as FP32, for a comparison with [`assert_within`].
pub fn narrow(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&value| value as f32).collect()
}
