//! The numeric data type a deployment loads its model in.

crate::tensor::dtype_subset! {
    /// The element type of a model's parameters and activations (`dtype`): a subset of the
    /// interfaces' element types ([`crate::tensor::DType`]), named by its names.
    ModelDtype, "model dtype" {
        /// IEEE 754 half precision.
        Float16 = F16,
        /// Brain floating-point half precision.
        BFloat16 = BF16,
        /// IEEE 754 single precision.
        Float32 = F32,
    }
}
