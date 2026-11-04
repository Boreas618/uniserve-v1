//! A dense projection: activations times a weight, with an optional bias.
//!
//! The layer holds the weight in the order its checkpoint gives (Qwen3 stores `[n, k]` rows of
//! output channels) and names that order in the `gemm` descriptor, so the product is one operator
//! call whatever the storage order is. The contraction length is geometry the descriptor bakes in
//! (specs/worker.md 4.2), so it is checked against the weight at construction and against the input
//! at every call.

use ops::TensorView;
use ops::gemm::{self, Descriptor, Epilogue, Gemm, WeightOrder};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::ACCUMULATE;
use crate::weights::Weight;

/// A dense projection `out = input · weight + bias`.
pub struct Linear {
    weight: Weight,
    bias: Option<Weight>,
    order: WeightOrder,
    k: u32,
    n: u32,
    input: Option<ops::DType>,
}

impl Linear {
    /// A projection with `weight` stored in `order`.
    ///
    /// `weight` is `[k, n]` in [`WeightOrder::KxN`] or `[n, k]` in [`WeightOrder::NxK`]; the bias,
    /// when present, has `n` elements.
    pub fn new(weight: Weight, bias: Option<Weight>, order: WeightOrder) -> Result<Self> {
        let (k, n) = match order {
            WeightOrder::KxN => {
                if weight.shape().rank() != 2 {
                    return Err(Error::shape(
                        "Linear",
                        format!("a kxn weight is a matrix, not {:?}", weight.shape().dims()),
                    ));
                }
                (weight.shape().dim(0), weight.shape().dim(1))
            }
            WeightOrder::NxK => {
                if weight.shape().rank() != 2 {
                    return Err(Error::shape(
                        "Linear",
                        format!("an nxk weight is a matrix, not {:?}", weight.shape().dims()),
                    ));
                }
                (weight.shape().dim(1), weight.shape().dim(0))
            }
        };
        if let Some(bias) = bias
            && bias.shape().numel() != n
        {
            return Err(Error::shape(
                "Linear",
                format!(
                    "a bias of {} elements cannot be added to {n} output channels",
                    bias.shape().numel()
                ),
            ));
        }
        Ok(Self {
            weight,
            bias,
            order,
            k: u32::try_from(k)
                .map_err(|_| Error::shape("Linear", format!("a contraction of {k} elements")))?,
            n: u32::try_from(n)
                .map_err(|_| Error::shape("Linear", format!("{n} output channels")))?,
            input: None,
        })
    }

    /// The same projection, with the activation element type its descriptor is compiled for.
    ///
    /// A model whose activations are one dtype everywhere needs nothing more; a projection on an
    /// FP32 activation path (H3's latent input and output heads, specs/worker.md 4.3) states it
    /// here rather than discovering it at launch.
    pub fn with_input(mut self, dtype: ops::DType) -> Self {
        self.input = Some(dtype);
        self
    }

    /// The contraction length.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// The number of output channels.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// The weight view.
    pub fn weight(&self) -> TensorView {
        self.weight.view()
    }

    /// `out = input · weight + bias`.
    ///
    /// `input` is `[m, k]` and `out` is `[m, n]`, both row-major on their last axis.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let k = u64::from(self.k);
        let n = u64::from(self.n);
        let input_dtype = self.input.unwrap_or(input.dtype);
        input
            .require_row("Linear", input_dtype, k)
            .map_err(Error::from)?;
        out.require_row("Linear", out.dtype, n)
            .map_err(Error::from)?;
        if out.rows() != input.rows() {
            return Err(Error::shape(
                "Linear",
                format!(
                    "{} input rows against {} output rows",
                    input.rows(),
                    out.rows()
                ),
            ));
        }
        let descriptor = Descriptor {
            input: input_dtype,
            weight: self.weight.dtype(),
            accumulate: ACCUMULATE,
            output: out.dtype,
            order: self.order,
            epilogue: if self.bias.is_some() {
                Epilogue::Bias
            } else {
                Epilogue::None
            },
            k: self.k,
        };
        let args = gemm::Args {
            input,
            weight: self.weight.view(),
            bias: self.bias.as_ref().map(Weight::view),
            out,
            alpha: 1.0,
        };

        f.operate::<Gemm>(&descriptor, args)
    }
}
