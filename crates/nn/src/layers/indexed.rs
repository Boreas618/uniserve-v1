//! A projection whose weights are chosen per step from a device index: one row of a table each.
//!
//! A schedule that predicts with a different head per evaluation (a parallel-decoding student's
//! fused heads) stores every head as one row of a `[steps, out · in]` weight table and a
//! `[steps, out]` bias table. A forward gathers the row its `[1]` i32 step view names and projects
//! with it, so one captured graph serves every step: the gathered buffers sit at fixed addresses
//! with contents that change per evaluation.

use ops::gemm::WeightOrder;
use ops::{Shape, Operand};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::{IndexRows, Linear};
use crate::weights::Weight;

/// A [`Linear`] per step, gathered from a table by the step's device index.
pub struct StepIndexed {
    /// `[steps, out · in]`: step `s`'s weight, `[out, in]` row-major.
    weight: Weight,
    /// `[steps, out]`: step `s`'s bias.
    bias: Weight,
    inputs: u32,
    outputs: u32,
}

impl StepIndexed {
    /// The projections of `inputs` to `outputs` channels whose weights are the rows of `weight`
    /// `[steps, outputs · inputs]` and `bias` `[steps, outputs]`, both of one element type.
    ///
    /// # Errors
    ///
    /// Tables of another extent or of two element types.
    pub fn new(weight: Weight, bias: Weight, inputs: u32, outputs: u32) -> Result<Self> {
        let per_step = u64::from(outputs) * u64::from(inputs);
        let (weights, biases) = (weight.shape(), bias.shape());
        if weights.rank() != 2
            || biases.rank() != 2
            || weights.last() != per_step
            || biases.last() != u64::from(outputs)
            || weights.dim(0) != biases.dim(0)
            || weight.dtype() != bias.dtype()
        {
            return Err(Error::shape(
                "StepIndexed",
                format!(
                    "a {} weight table of {:?} and a {} bias table of {:?} for [steps, {per_step}] \
                     and [steps, {outputs}]",
                    ops::abi::name(weight.dtype()),
                    weights.dims(),
                    ops::abi::name(bias.dtype()),
                    biases.dims()
                ),
            ));
        }
        Ok(Self {
            weight,
            bias,
            inputs,
            outputs,
        })
    }

    /// The steps the tables hold.
    pub fn steps(&self) -> u64 {
        self.weight.shape().dim(0)
    }

    /// `input` `[rows, in]` projected to `[rows, out]` in `input`'s element type, with the weights
    /// of the step `step` (`[1]` i32) names.
    ///
    /// The step's weight and bias rows are gathered in the tables' element type, which is the
    /// projection's, after the output, and handed back once the product is enqueued.
    ///
    /// # Errors
    ///
    /// A step view the gather refuses, views the projection refuses, or an arena that cannot hold
    /// the rows.
    pub fn forward(&self, f: &Forward, step: Operand, input: Operand) -> Result<Operand> {
        let (inputs, outputs) = (u64::from(self.inputs), u64::from(self.outputs));
        let out = f.empty(input.dtype, Shape::new(&[input.rows(), outputs]))?;
        // The gathered rows are dead once the product is enqueued, so they live in a scope after
        // the output.
        f.scoped(|f| {
            let wide = IndexRows.gather(f, self.weight.view(), step)?;
            let narrow = IndexRows.gather(f, self.bias.view(), step)?;
            let projection = Linear::new(
                Weight::borrowed(wide.reshape(Shape::new(&[outputs, inputs])))?,
                Some(Weight::borrowed(narrow.reshape(Shape::new(&[outputs])))?),
                WeightOrder::NxK,
            )?;
            projection.launch(f, input, out)
        })?;
        Ok(out)
    }
}
