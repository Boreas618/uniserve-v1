//! The Qwen3-VL model as an encoder: a prompt's hidden states, its media merged in.
//!
//! [`Qwen3VlModel`] is Hugging Face's `Qwen3VLModel` over a prefix of its decoder layers
//! (vLLM's `Qwen3LLMModel.forward`, `refs/vllm/vllm/model_executor/models/
//! qwen3_vl.py:1852-1905`):
//!
//! ```text
//! stream            = embed(ids)
//! stream[vision]    = tower(images)                  merged rows replace the placeholder rows
//! for layer j:      hidden, stream = layer_j(…)      interleaved M-RoPE over (time, row, column)
//!                   hidden[vision] += deepstack_j    for j below the DeepStack count
//! out               = norm(hidden + stream), or hidden + stream when the consumer reads the
//!                     state before the final norm
//! ```
//!
//! The DeepStack addition is `Qwen3VLTextModel._deepstack_process`: after decoder layer `j`
//! the tower's DeepStack feature `j` is added at the vision rows.
//!
//! The values a prompt decides are computed in the forward as CPU tensors and moved to the
//! device: its M-RoPE coordinates ([`positions::rope_index`]) and the prompt row each merged
//! vision row takes ([`positions::vision_rows`]). Their shapes follow from the layout's
//! counts, so two prompts of one layout replay one captured graph.
//!
//! The language model is tensor parallel; the tower is replicated on every rank, since every
//! rank needs its features.

use nn::modules::{Embedding, EmbeddingConfig, RmsNorm, Rotary};
use nn::{Encoder, EncoderLayout, PixelNorm, Split, Tensor, TokenBatch, VarBuilder};
use processing::qwen_vl::ProcessorConfig;

use crate::qwen3::config::Config;
use crate::qwen3::decoder::{self, DecoderLayer};
use crate::qwen3::vision::VisionTower;
use crate::qwen3::{Error, Result, positions};

/// The Qwen3-VL model: the language model's embedding, a prefix of its decoder layers, the
/// vision tower, and the final norm when the consumer reads normalized states.
pub struct Qwen3VlModel {
    config: Config,
    embed_tokens: Embedding,
    rotary_emb: Rotary,
    layers: Vec<DecoderLayer>,
    visual: VisionTower,
    norm: Option<RmsNorm>,
}

impl Qwen3VlModel {
    /// The model of `config` at `vb` (the checkpoint's root): its first `layers` decoder
    /// layers at `split`'s rank of the tensor-parallel group, the tower whole with the
    /// processor rules and pixel normalization `images` states, and the final norm when
    /// `final_norm` asks for it.
    ///
    /// # Errors
    ///
    /// A configuration without a tower, more layers than the checkpoint holds or none, a
    /// split that does not divide the heads, or a module the loader refuses.
    pub fn load(
        config: &Config,
        vb: VarBuilder<'_, '_>,
        images: (ProcessorConfig, PixelNorm),
        split: Split,
        layers: u32,
        final_norm: bool,
    ) -> Result<Self> {
        let text = &config.text;
        let vision = config
            .vision
            .as_ref()
            .ok_or_else(|| Error::Config("a Qwen3-VL model states no vision tower".to_owned()))?;
        if layers == 0 || layers > text.num_hidden_layers {
            return Err(Error::Config(format!(
                "decoder layers 0..{layers} of a {}-layer stack",
                text.num_hidden_layers
            )));
        }
        text.local_heads(split)?;

        let model = vb.pp("model.language_model");
        let stack = (0..layers)
            .map(|layer| {
                DecoderLayer::load(
                    config,
                    model.pp("layers").pp(layer),
                    text.linear_layers[layer as usize],
                    config.routes(layer),
                    (split, Split::WHOLE),
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = if final_norm {
            Some(RmsNorm::load(
                &decoder::stream_norm(text),
                model.pp("norm"),
            )?)
        } else {
            None
        };

        let (processor, pixels) = images;
        Ok(Self {
            embed_tokens: Embedding::load(
                &EmbeddingConfig {
                    vocabulary: text.vocab_size,
                    hidden: text.hidden_size,
                    split,
                },
                model.pp("embed_tokens"),
            )?,
            rotary_emb: Rotary::load(&text.rope, model.pp("rotary_emb"))?,
            layers: stack,
            visual: VisionTower::load(vision, vb.pp("model.visual"), processor, pixels)?,
            norm,
            config: config.clone(),
        })
    }

    /// The configuration this model was built for.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The vision tower.
    pub const fn tower(&self) -> &VisionTower {
        &self.visual
    }

    /// The tower's rows of the images of `layout`, in prompt order: `[merged,
    /// hidden · (1 + deepstack)]`, or `None` for a prompt that presents no image.
    fn vision_features(&self, images: &[Tensor], layout: &EncoderLayout) -> Result<Option<Tensor>> {
        if layout.images.is_empty() {
            return Ok(None);
        }

        let features = layout
            .images
            .iter()
            .zip(images)
            .map(|(grid, pixels)| self.visual.forward(*grid, pixels))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Tensor::cat(&features, 0)?))
    }

    /// [`Encoder::encode`] over the prompt `prompt`.
    fn forward(
        &self,
        prompt: &[u32],
        ids: &Tensor,
        batch: &TokenBatch,
        images: &[Tensor],
        layout: &EncoderLayout,
        out: &Tensor,
    ) -> Result<()> {
        let tokens = layout.tokens as usize;
        let hidden = self.config.text.hidden_size as usize;
        if layout.rows != layout.tokens
            || batch.tokens() != tokens
            || prompt.len() != tokens
            || ids.elem_count() != tokens
            || out.dims() != [tokens, hidden]
            || images.len() != layout.images.len()
        {
            return Err(Error::Config(format!(
                "an encode of {} rows for {} tokens: {} batch tokens, {} prompt ids, {} device \
                 ids, an output of {:?}, {} images for {} grids",
                layout.rows,
                layout.tokens,
                batch.num_tokens,
                prompt.len(),
                ids.elem_count(),
                out.dims(),
                images.len(),
                layout.images.len()
            )));
        }
        let vision = self.visual.config();
        let (image, video) = (vision.image_token_id, vision.video_token_id);
        let device = ids.device();

        // The prompt's M-RoPE coordinates, and the row each merged vision row takes.
        let coordinates = positions::rope_index(
            prompt,
            &layout.images,
            image,
            video,
            vision.spatial_merge_size,
        )?
        .to_device(device)?;
        let rows = positions::vision_rows(prompt, image, video)?;

        // The embeddings, the tower's rows replacing the placeholders' (`inputs_embeds[mask]
        // = image_embeds`): each column of a row is written at the row's place.
        let mut stream = self.embed_tokens.forward(ids)?;
        let features = self.vision_features(images, layout)?;
        let merged = features.as_ref().map_or(0, |features| features.dims()[0]);
        if merged != rows.elem_count() {
            return Err(Error::Config(format!(
                "the images merge to {merged} rows and the prompt holds {} placeholders",
                rows.elem_count()
            )));
        }

        // Only a prompt that presents vision rows moves their places to the device.
        let vision_rows = match &features {
            Some(features) => {
                let rows = rows.to_device(device)?;
                let index = rows
                    .unsqueeze(1)?
                    .broadcast_as((merged, hidden))?
                    .contiguous()?;

                stream =
                    stream.scatter(&index, &features.narrow(1, 0, hidden)?.contiguous()?, 0)?;
                Some((rows, features))
            }
            None => None,
        };

        // The decoder layers over the packed prompt, each sequence attending itself; DeepStack
        // feature `j` joins layer `j`'s output at the vision rows (`hidden_states +
        // deepstack_input_embeds`, `qwen3_vl.py:1884-1888`).
        let (cos, sin) = self.rotary_emb.forward(&coordinates)?;
        let deepstack = vision.deepstack_visual_indexes.len();
        let (mut output, mut residual) = (stream, None);
        for (index, layer) in self.layers.iter().enumerate() {
            let (mut hidden_states, stream) = layer.forward_varlen(
                &output,
                residual.as_ref(),
                (&cos, &sin),
                &batch.query_offsets,
                layout.tokens,
            )?;

            if let Some((rows, features)) = vision_rows.as_ref().filter(|_| index < deepstack) {
                let feature = features.narrow(1, (index + 1) * hidden, hidden)?;
                hidden_states = hidden_states.index_add(rows, &feature.contiguous()?, 0)?;
            }
            (output, residual) = (hidden_states, Some(stream));
        }
        let residual =
            residual.ok_or_else(|| Error::Config("a decoder stack of no layers".to_owned()))?;

        // The last output joins the stream: added and normalized, or, for a consumer that
        // reads the state before the final norm, added alone.
        let encoded = match &self.norm {
            Some(norm) => norm.forward_residual(&output, &residual)?.0,
            None => (output + residual)?,
        };
        Ok(out.slice_set(&encoded, 0, 0)?)
    }
}

impl Encoder for Qwen3VlModel {
    /// One output row per presented token, the images' grids as the batch states them.
    fn layout(&self, batch: &TokenBatch, images: &[[u32; 3]]) -> EncoderLayout {
        EncoderLayout {
            rows: batch.num_tokens,
            tokens: batch.num_tokens,
            images: images.to_vec(),
        }
    }

    fn encode(
        &self,
        prompt: &[u32],
        ids: &Tensor,
        batch: &TokenBatch,
        images: &[Tensor],
        layout: &EncoderLayout,
        out: &Tensor,
    ) -> nn::Result<()> {
        Ok(self.forward(prompt, ids, batch, images, layout, out)?)
    }
}
