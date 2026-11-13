//! `VarBuilder`: how a module names and loads its weights.
//!
//! A module loads itself from the builder its parent hands it: [`VarBuilder::get`] reads
//! `<path>.<name>`, and [`VarBuilder::pp`] gives a child the path `<path>.<child>`, so a
//! checkpoint name is composed exactly as `named_parameters()` composes it and no model writes a
//! name out. A builder may pin its tensors to one element type ([`VarBuilder::to_dtype`]), which
//! its children inherit.
//!
//! The builder is a path over a [`WeightLoader`]: what a weight is converted to, where it is
//! published and whether the load is a meta load are the loader's, so a module's `load` is the
//! same code for a device, for the CPU and for a profile run.
//!
//! A builder reads dense weights. A weight the checkpoint stores packed has a module that can
//! read it ([`Linear`](crate::modules::Linear), which loads its packed fields itself through
//! [`VarBuilder::loader`]); any other module that asks for one is refused by name, because
//! reading packed E2M1 as a matrix makes a model that runs and computes nothing.

use std::fmt;

use tensor::{DType, Shape, Tensor};

use crate::error::{Error, Result};
use crate::quantization::Scheme;
use crate::weights::{WeightLayout, WeightLoader, WeightSource, host_bytes, numel, source_names};

/// The class the constants a module computes are filed under: the producer of their retained
/// keys. Its text is part of every stored key of such a constant, so it does not follow the
/// type's name.
const PRODUCER: &str = "nn::weights::Scope";

/// One module's view of a loader: the module's path, and the tensors named under it.
#[derive(Clone)]
pub struct VarBuilder<'l, 'a> {
    loader: &'l WeightLoader<'a>,
    /// The module's path in the checkpoint; empty at the root.
    path: String,
    /// The element type every tensor under this builder is read as, when the module pins one.
    dtype: Option<DType>,
}

impl<'l, 'a> VarBuilder<'l, 'a> {
    /// The builder of `loader`'s checkpoint root.
    pub(crate) fn new(loader: &'l WeightLoader<'a>) -> Self {
        Self {
            loader,
            path: String::new(),
            dtype: None,
        }
    }

    /// The module path this builder names its tensors under: the path a module states its
    /// profiler range with ([`crate::profiler::range`]).
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The full checkpoint name of this module's `name`.
    pub fn name(&self, name: &str) -> String {
        if self.path.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{name}", self.path)
        }
    }

    /// The builder of the child module `prefix`, which inherits this builder's pin.
    pub fn pp(&self, prefix: impl fmt::Display) -> Self {
        Self {
            loader: self.loader,
            path: self.name(&prefix.to_string()),
            dtype: self.dtype,
        }
    }

    /// This builder with every tensor read as `dtype`, whatever the loader's policy states.
    ///
    /// A representation a model pins per module (the reference keeps a timestep embedding in
    /// FP32 and a VAE decoder's projections in FP16 at every preset) is the module's statement,
    /// and the pin is part of each weight's retained key.
    pub fn to_dtype(&self, dtype: DType) -> Self {
        Self {
            loader: self.loader,
            path: self.path.clone(),
            dtype: Some(dtype),
        }
    }

    /// The element type this builder pins its tensors to, when it pins one.
    pub fn dtype(&self) -> Option<DType> {
        self.dtype
    }

    /// Whether the checkpoint holds this module's `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.loader.checkpoint().tensor(&self.name(name)).is_some()
    }

    /// The whole of this module's `name`, a tensor of `shape`.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the tensor, holds another element count, stores it packed, or
    /// cannot convert it to the pinned type.
    pub fn get(&self, shape: impl Into<Shape>, name: &str) -> Result<Tensor> {
        self.get_rows(shape, name, 0)
    }

    /// The rows of this module's `name` that start at row `first`: a contiguous window of
    /// `shape`, whose first extent counts the rows and whose other extents are a row's.
    ///
    /// # Errors
    ///
    /// As [`Self::get`], and when the window reaches past the tensor.
    pub fn get_rows(&self, shape: impl Into<Shape>, name: &str, first: usize) -> Result<Tensor> {
        let shape = shape.into();
        let tensor = self.name(name);
        let row: u64 = shape.dims().iter().skip(1).map(|dim| *dim as u64).product();

        let source = WeightSource::Region {
            tensor: tensor.clone(),
            start: first as u64 * row,
            shape: shape.clone(),
        };
        self.load(&tensor, &source, WeightLayout::ROW_MAJOR, &shape, false)
    }

    /// The weight `name` gathered from `source` in `layout`, as a tensor of `shape`, at this
    /// builder's pin; `nonzero` asks that it hold a value ([`WeightLoader::load`]).
    ///
    /// `name` and the tensors `source` names are full checkpoint names ([`Self::name`]): a
    /// source may fuse tensors of several modules.
    ///
    /// # Errors
    ///
    /// As [`WeightLoader::load`], and when a tensor the source names is stored packed in a
    /// scheme a dense read cannot state.
    pub fn load(
        &self,
        name: &str,
        source: &WeightSource,
        layout: WeightLayout,
        shape: &Shape,
        nonzero: bool,
    ) -> Result<Tensor> {
        let names = source_names(source);

        // A weight-only NVFP4 matrix and a block-scaled FP8 matrix are read as the dense matrix
        // they state, by every module.
        for scheme in [Scheme::Nvfp4Weights, Scheme::Fp8Blocks] {
            if !names.is_empty()
                && names
                    .iter()
                    .all(|tensor| self.loader.scheme(tensor) == scheme)
            {
                return self
                    .loader
                    .load_decoded(name, source, layout, shape, scheme);
            }
        }

        self.refuse_packed(&names)?;
        self.loader
            .load(name, source, layout, shape, self.dtype, nonzero)
    }

    /// This module's weight-normalized `name`, stored as `<name>_v` and `<name>_g`, as one
    /// weight ([`WeightLoader::weight_norm`]) at this builder's pin.
    ///
    /// # Errors
    ///
    /// As [`WeightLoader::weight_norm`], and when the weight is stored packed.
    pub fn weight_norm(&self, name: &str) -> Result<Tensor> {
        let tensor = self.name(name);
        self.refuse_packed(std::slice::from_ref(&tensor))?;

        let (value, gain) = (format!("{tensor}_v"), format!("{tensor}_g"));
        self.loader.weight_norm(&tensor, &value, &gain, self.dtype)
    }

    /// A table this module computes instead of reading: `values` encoded as `dtype`, rounded to
    /// nearest with ties to even, published as `<path>.<name>` in `shape`.
    ///
    /// The module path names the constant, and the retained key's provenance digests the bytes
    /// ([`WeightLoader::constant`]), so two modules' constants never share a name and one
    /// module's two computations of different numbers never share storage.
    ///
    /// # Errors
    ///
    /// When `values` is not `shape`'s element count or `dtype` is not F32, BF16 or F16.
    pub fn constant(
        &self,
        name: &str,
        dtype: DType,
        shape: impl Into<Shape>,
        values: &[f32],
    ) -> Result<Tensor> {
        let bytes = ops::values::encode(dtype, values)?;
        self.loader
            .constant(&self.name(name), dtype, &shape.into(), &bytes, PRODUCER)
    }

    /// A table this module computed as the CPU tensor `table`, published as `<path>.<name>`
    /// with the table's shape and element type and its elements unchanged: an index row
    /// (`Tensor::arange`), a table of positions, a table of zeros.
    ///
    /// # Errors
    ///
    /// When `table` is not a CPU tensor of integers or floats.
    pub fn constant_from(&self, name: &str, table: &Tensor) -> Result<Tensor> {
        let tensor = self.name(name);
        let bytes = host_bytes(&tensor, table)?;

        self.loader
            .constant(&tensor, table.dtype(), table.shape(), &bytes, PRODUCER)
    }

    /// A table of zeros of `dtype` this module's launches write once after the load
    /// ([`WeightLoader::writable_constant`]): it stays writable on the device and is this
    /// process's own.
    ///
    /// # Errors
    ///
    /// When `dtype` has no whole-byte element size.
    pub fn writable_zeros(
        &self,
        name: &str,
        dtype: DType,
        shape: impl Into<Shape>,
    ) -> Result<Tensor> {
        let (tensor, shape) = (self.name(name), shape.into());
        let element = ops::abi::bytes(dtype).ok_or_else(|| {
            Error::weight(
                &tensor,
                format!("{} has no whole-byte element", ops::abi::name(dtype)),
            )
        })?;

        // Every supported type's zero is the all-zero pattern.
        let bytes = vec![0; (numel(&shape) * element) as usize];
        self.loader
            .writable_constant(&tensor, dtype, &shape, &bytes, PRODUCER)
    }

    /// How the checkpoint stores this module's `name`, by the declaration the model's loader
    /// was given: what a module that has an encoded form chooses its representation by.
    pub fn scheme(&self, name: &str) -> Scheme {
        self.loader.scheme(&self.name(name))
    }

    /// The single positive scale this module's `name` holds ([`WeightLoader::tensor_scale`]).
    ///
    /// # Errors
    ///
    /// As [`WeightLoader::tensor_scale`].
    pub fn tensor_scale(&self, name: &str) -> Result<f32> {
        self.loader.tensor_scale(&self.name(name))
    }

    /// The loader, for a module that reads its own packed fields.
    pub fn loader(&self) -> &'l WeightLoader<'a> {
        self.loader
    }

    /// Refuses a dense read of a weight the checkpoint stores packed.
    fn refuse_packed(&self, names: &[String]) -> Result<()> {
        match names
            .iter()
            .find(|name| self.loader.scheme(name) != Scheme::Dense)
        {
            None => Ok(()),
            Some(name) => Err(Error::weight(
                name,
                "the checkpoint stores this weight packed, and the module reading it has no \
                 encoded form: reading it as dense would read packed E2M1 as a matrix. Load a \
                 checkpoint whose calibration excludes it",
            )),
        }
    }
}

impl fmt::Debug for VarBuilder<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VarBuilder")
            .field("path", &self.path)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}
