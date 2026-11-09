//! The text encoder of MiniMax-H3: the Qwen3-VL model ([`qwen3::vl::Qwen3VlModel`]) as H3
//! conditions on it.
//!
//! H3 places the Qwen3-VL language model and the vision tower, runs only the first
//! [`RETAINED_LAYERS`] of the checkpoint's decoder layers, and returns the last retained layer's
//! raw hidden state - the checkpoint's final normalization is not part of H3's conditioning path
//! (`network.norm = Identity`). The tower's merged tokens replace the prompt's image and video
//! placeholder tokens, and DeepStack feature `j` is added to the residual stream after retained
//! layer `j`, whichever tower block it was merged from.
//!
//! ## The DeepStack cut
//!
//! A DeepStack addition is the one place H3's reference cuts the stack's recurrence: it adds the
//! layer's own output to the residual (`residual + hidden`, one rounding), then the features (a
//! second rounding), and starts the next layer from a zero update. Adding zero is exact, so the
//! next layer instead normalizes the residual alone, which is the same bits.

use nn::capability::{
    Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, ImageShape, StateTable,
    TableValues,
};
use nn::layers::{IndexRows, Residual};
use nn::weights::WeightLoader;
use nn::{DType, Forward};
use qwen3::VlConfig;
use qwen3::decoder::{Components, Hidden};
use qwen3::vl::{DeepStack, Qwen3VlModel};

use crate::Result;
use crate::config::RETAINED_LAYERS;
use crate::partition::Partition;

/// The components a deployment's fence probes name (`profiling_config.fence_components`): one layer
/// of the text encoder's stack, and no sublayer.
const COMPONENTS: Components = Components {
    layer: "models.minimax_h3.text_encoder.TransformerLayer",
    attention: None,
    mlp: None,
};

/// The Qwen3-VL text encoder of H3, its vision tower included.
pub struct TextEncoder {
    model: Qwen3VlModel,
}

impl TextEncoder {
    /// The encoder of `config`, reading the checkpoint through `loader`.
    ///
    /// `dtype` is the activation element type the loading boundary chose. The language model is
    /// built for `partition`'s tensor-parallel position and truncated to [`RETAINED_LAYERS`], with
    /// no final norm; the vision tower is loaded whole on every rank.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &VlConfig,
        partition: &Partition,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            model: Qwen3VlModel::load(
                loader,
                config,
                partition.text_split()?,
                dtype,
                RETAINED_LAYERS,
                false,
                COMPONENTS,
            )?,
        })
    }
}

/// H3's DeepStack placement: after retained layer `index`, table `index` joins the stream when the
/// tower returned one.
///
/// The layer's own update is added into the stream first, then the features are added at their
/// placeholder rows, and the next layer reads the stream alone (module doc, the DeepStack cut).
fn deepstack_cut(
    f: &Forward,
    index: usize,
    state: Hidden,
    deepstack: &DeepStack,
) -> qwen3::Result<Hidden> {
    let Some(stream) = state.residual else {
        return Ok(state);
    };
    if index >= deepstack.len() {
        return Ok(state);
    }
    Residual::default().forward(f, stream, state.hidden, None, stream)?;
    if let Some(features) = deepstack.features(index)? {
        IndexRows.scatter_add(f, features, deepstack.destinations(), stream)?;
    }
    Ok(Hidden {
        hidden: stream,
        residual: None,
    })
}

impl Encoder for TextEncoder {
    /// The layout of a batch and its images.
    ///
    /// The encoder writes one row per presented token, so the row capacity is the batch's; a grid
    /// that cannot be served is refused where a failure can be reported - [`Encoder::prepare_state`]
    /// and [`Encoder::encode`] - because this method answers with the layout alone.
    fn layout(&self, batch: &nn::capability::TokenBatch, images: &[ImageShape]) -> EncoderLayout {
        EncoderLayout {
            rows: batch.num_tokens,
            tokens: batch.num_tokens,
            images: images.to_vec(),
        }
    }

    fn state_tables(&self, layout: &EncoderLayout) -> nn::Result<Vec<StateTable>> {
        Ok(self.model.state_tables(layout)?)
    }

    fn prepare_state(
        &self,
        prompt: &EncoderPrompt<'_>,
        layout: &EncoderLayout,
        out: &mut [TableValues],
    ) -> nn::Result<()> {
        Ok(self.model.prepare_state(prompt.ids, layout, out)?)
    }

    fn encode(
        &self,
        f: &Forward,
        inputs: EncoderInputs<'_>,
        out: EncoderOutputs,
    ) -> nn::Result<()> {
        Ok(self
            .model
            .encode(f, inputs, out, |index, state, deepstack| {
                deepstack_cut(f, index, state, deepstack)
            })?)
    }
}
