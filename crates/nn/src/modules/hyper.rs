//! The multi-stream residual of a hyper-connected decoder: manifold-constrained hyper-connections
//! (mHC), whose combination of the streams is a doubly stochastic matrix.
//!
//! A hyper-connected decoder carries its residual as `n` streams of the hidden width,
//! `[tokens, n, d]`, and every sub-layer (an attention, a feed-forward layer) sits between two
//! connections: it reads its input from the streams and writes its output back into them. Each
//! sub-layer has its own [`MhcConnection`], whose mixes the streams themselves determine
//! ([`functional::mhc_boundary`]). [`MhcResidual`] is the state the decoder carries from one
//! sub-layer to the next, in the shifted form the reference evaluates:
//!
//! ```text
//! expand      h = [e, e, …, e];   pre = (1, 0, …, 0)            the embedding in every stream
//! sub-layer   h ← post(h, y_prev, mixes_prev)                  the previous output written back
//!             mixes ← mixes_c(h)                               this connection's mixes
//!             y ← body(pre(h, pre))                            the input read with the carried
//!             pre ← mixes.pre                                  pre-mix, then carried on
//! collapse    rms_norm(pre(post(h, y_prev, mixes_prev), pre))  into the output norm
//! ```
//!
//! A sub-layer's output stays pending until the next boundary writes it back, so the write and
//! the next read are one pass over the streams. A layer that adds into the streams between two
//! sub-layers (a hashed n-gram memory) takes them with every output written back
//! ([`MhcResidual::streams`]) and hands back what it computed ([`MhcResidual::set_streams`]); the
//! carried pre-mix is unchanged by it, as the reference keeps it.

use tensor::{DType, Tensor};

use crate::error::{Error, Result};
use crate::functional::{self, MhcMixes, MhcMixing};
use crate::var_builder::VarBuilder;

/// What a checkpoint states of an mHC residual.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MhcConfig {
    /// Streams of the residual, `n`.
    pub streams: u32,
    /// Elements of one stream: the decoder's hidden width, `d`.
    pub hidden: u32,
    /// Sinkhorn iterations of a combination matrix.
    pub iterations: u32,
    /// The epsilon of a pre-mix and of every Sinkhorn step.
    pub eps: f32,
    /// The epsilon of the flattened streams' mean square, which scales the mixes.
    pub rms_eps: f32,
    /// The epsilon of a sub-layer input's RMS normalization.
    pub norm_eps: f32,
}

impl MhcConfig {
    /// The mixes of one connection, `n·(n + 2)`.
    pub const fn mixes(&self) -> u32 {
        self.streams * (self.streams + 2)
    }
}

/// One sub-layer's connection: the projection, scales and biases of its mixes (FP32), and the
/// gain of the RMS normalization its input is read with.
pub struct MhcConnection {
    /// `[n·(n + 2), n·d]` FP32.
    projection: Tensor,
    /// `[3]` FP32.
    scale: Tensor,
    /// `[n·(n + 2)]` FP32.
    base: Tensor,
    /// `[d]`: the input norm's gain, in the streams' type.
    norm: Tensor,
    config: MhcConfig,
}

impl MhcConnection {
    /// The connection whose mixing parameters `vb` names as the reference implementation names
    /// them, `<prefix>_fn`, `<prefix>_scale` and `<prefix>_base` (read in FP32 whatever the
    /// loader's policy is), and whose input norm is the `weight` of `norm`.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks a tensor or holds other extents.
    pub fn load(
        config: &MhcConfig,
        vb: &VarBuilder<'_, '_>,
        prefix: &str,
        norm: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let (n, d, m) = (
            config.streams as usize,
            config.hidden as usize,
            config.mixes() as usize,
        );
        let exact = vb.to_dtype(DType::F32);

        Self::new(
            config,
            exact.get((m, n * d), &format!("{prefix}_fn"))?,
            exact.get(3, &format!("{prefix}_scale"))?,
            exact.get(m, &format!("{prefix}_base"))?,
            norm.get(d, "weight")?,
        )
    }

    /// The connection over the tensors given: `projection` `[n·(n + 2), n·d]`, `scale` `[3]`
    /// and `base` `[n·(n + 2)]`, all FP32, and the input norm's gain `norm` `[d]`.
    ///
    /// # Errors
    ///
    /// When a tensor has other extents or the mixing parameters are not FP32.
    pub fn new(
        config: &MhcConfig,
        projection: Tensor,
        scale: Tensor,
        base: Tensor,
        norm: Tensor,
    ) -> Result<Self> {
        let (n, d, m) = (
            config.streams as usize,
            config.hidden as usize,
            config.mixes() as usize,
        );
        for (name, tensor, dims) in [
            ("projection", &projection, [m, n * d].as_slice()),
            ("scale", &scale, &[3]),
            ("base", &base, &[m]),
        ] {
            if tensor.dims() != dims || tensor.dtype() != DType::F32 {
                return Err(Error::shape(
                    "MhcConnection",
                    format!(
                        "the {name} is {} {:?}, FP32 {dims:?} expected",
                        ops::abi::name(tensor.dtype()),
                        tensor.dims()
                    ),
                ));
            }
        }
        if norm.dims() != [d] {
            return Err(Error::shape(
                "MhcConnection",
                format!("the norm's gain is {:?}, [{d}] expected", norm.dims()),
            ));
        }

        Ok(Self {
            projection,
            scale,
            base,
            norm,
            config: *config,
        })
    }

    /// The mixing parameters, as a boundary reads them.
    pub fn mixing(&self) -> MhcMixing<'_> {
        MhcMixing {
            projection: &self.projection,
            scale: &self.scale,
            base: &self.base,
            iterations: self.config.iterations,
            rms_eps: self.config.rms_eps,
            eps: self.config.eps,
        }
    }

    /// The gain of the input's RMS normalization.
    pub fn norm(&self) -> &Tensor {
        &self.norm
    }
}

impl std::fmt::Debug for MhcConnection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MhcConnection")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// The residual a hyper-connected decoder carries from one sub-layer to the next (module
/// documentation).
pub struct MhcResidual {
    /// `[tokens, n, d]`: the streams, before the pending output is written back.
    streams: Tensor,
    /// The last sub-layer's output and the mixes its connection computed, which the next
    /// boundary writes back.
    pending: Option<(Tensor, MhcMixes)>,
    /// `[tokens, n]` FP32: the pre-mix the next sub-layer's input is read with.
    pre: Tensor,
    config: MhcConfig,
}

impl MhcResidual {
    /// The residual of `embedded`, `[tokens, d]`: the embedding in every stream, read first by
    /// the pre-mix that selects stream 0.
    ///
    /// # Errors
    ///
    /// Rows of another width, or what a tensor operation refuses.
    pub fn expand(config: &MhcConfig, embedded: &Tensor) -> Result<Self> {
        let (tokens, width) = embedded.dims2()?;
        let n = config.streams as usize;
        if width != config.hidden as usize {
            return Err(Error::shape(
                "MhcResidual",
                format!(
                    "rows of {width} elements against streams of {}",
                    config.hidden
                ),
            ));
        }

        let streams = embedded
            .unsqueeze(1)?
            .broadcast_as((tokens, n, width))?
            .contiguous()?;
        let device = embedded.device();
        let first = Tensor::ones((tokens, 1), DType::F32, device)?;
        let pre = if n == 1 {
            first
        } else {
            Tensor::cat(
                &[first, Tensor::zeros((tokens, n - 1), DType::F32, device)?],
                1,
            )?
        };

        Ok(Self {
            streams,
            pending: None,
            pre,
            config: *config,
        })
    }

    /// One sub-layer between `connection` and the next: the boundary writes the pending output
    /// back, computes the connection's mixes and reads the input `body` computes the sub-layer's
    /// output of, `[tokens, d]`, which stays pending.
    ///
    /// # Errors
    ///
    /// What the boundary or `body` refuses, or an output of other extents.
    pub fn sublayer(
        &mut self,
        connection: &MhcConnection,
        body: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<()> {
        let pending = self.pending.as_ref().map(|(output, mixes)| (output, mixes));
        let (streams, input, mixes) = functional::mhc_boundary(
            &self.streams,
            pending,
            &self.pre,
            connection.mixing(),
            connection.norm(),
            self.config.norm_eps,
        )?;

        let output = body(&input)?;
        if output.dims() != input.dims() || output.dtype() != input.dtype() {
            return Err(Error::shape(
                "MhcResidual",
                format!(
                    "a sub-layer returned {} {:?} for an input of {} {:?}",
                    ops::abi::name(output.dtype()),
                    output.dims(),
                    ops::abi::name(input.dtype()),
                    input.dims()
                ),
            ));
        }

        self.streams = streams;
        self.pre = mixes.pre.clone();
        self.pending = Some((output, mixes));
        Ok(())
    }

    /// The streams with the pending output written back, `[tokens, n, d]`.
    ///
    /// # Errors
    ///
    /// What the write refuses.
    pub fn streams(&mut self) -> Result<&Tensor> {
        if let Some((output, mixes)) = self.pending.take() {
            self.streams = functional::mhc_post(&self.streams, &output, &mixes)?;
        }
        Ok(&self.streams)
    }

    /// Replaces the streams with `streams` of the same extents and type, which a layer between
    /// two sub-layers computed from [`Self::streams`]; the carried pre-mix stays.
    ///
    /// # Errors
    ///
    /// Streams of other extents or type, or an output still pending.
    pub fn set_streams(&mut self, streams: Tensor) -> Result<()> {
        if self.pending.is_some()
            || streams.dims() != self.streams.dims()
            || streams.dtype() != self.streams.dtype()
        {
            return Err(Error::shape(
                "MhcResidual",
                format!(
                    "streams of {:?} replace streams of {:?} with every output written back",
                    streams.dims(),
                    self.streams.dims()
                ),
            ));
        }
        self.streams = streams;
        Ok(())
    }

    /// The pre-mix the next read takes, `[tokens, n]` FP32.
    pub fn pre(&self) -> &Tensor {
        &self.pre
    }

    /// The residual collapsed into one row per token after the last sub-layer: the pending output
    /// written back, the streams read with the last connection's pre-mix and RMS-normalized with
    /// the decoder's output norm `norm`, `[d]`, and `eps`: `[tokens, d]`.
    ///
    /// # Errors
    ///
    /// What the write or the read refuses.
    pub fn collapse(mut self, norm: &Tensor, eps: f32) -> Result<Tensor> {
        self.streams()?;
        functional::mhc_pre(&self.streams, &self.pre, norm, eps)
    }
}

impl std::fmt::Debug for MhcResidual {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MhcResidual")
            .field("streams", &self.streams.dims())
            .field("pending", &self.pending.is_some())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
