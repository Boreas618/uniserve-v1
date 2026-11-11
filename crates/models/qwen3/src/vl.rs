//! The Qwen3-VL model as an encoder: a prompt's hidden states, its media spliced in.
//!
//! [`Qwen3VlModel`] is Hugging Face's `Qwen3VLModel` over a prefix of its decoder layers:
//!
//! ```text
//! stream            = embed(ids)
//! stream[vision]    = tower(images)                  merged rows replace the placeholder rows
//! for layer j:      stream = layer_j(stream)         interleaved M-RoPE over (time, row, column)
//!                   stream[vision] += deepstack_j    for j below the DeepStack count
//! hidden            = norm(stream), or stream        when the consumer reads the raw state
//! ```
//!
//! The DeepStack addition is `Qwen3VLTextModel._deepstack_process`: after decoder layer `j` the
//! tower's DeepStack table `j` is added to the stream at the vision rows.
//!
//! The values a prompt decides are computed in the forward and uploaded: its M-RoPE coordinates
//! ([`positions::rope_index`]) and the prompt row each merged vision row lands on
//! ([`positions::vision_destinations`]). Their shapes follow from the layout's counts, so two
//! prompts of one layout replay one capture.
//!
//! The language model is tensor parallel; the tower is replicated on every rank, since every rank
//! needs its features.

use nn::blocks::{AttentionKind, Keys, RopeFactors};
use nn::capability::{
    Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, ImageShape, PixelNorm,
    TokenBatch,
};
use nn::family::Split;
use nn::layers::{
    IndexRows, Residual, ResidualStream, RmsNorm, Rotary, Varlen, VocabParallelEmbedding,
};
use nn::weights::WeightLoader;
use nn::{DType, Forward, Shape};

use crate::config::{Config, POSITION_AXES};
use crate::decoder::{self, DecoderStack, Mlp};
use crate::vision::VisionTower;
use crate::{Error, Result, positions, write};
use processing::qwen_vl::ProcessorConfig;

/// The checkpoint's module path of the language model.
const LANGUAGE_MODEL: &str = "model.language_model";

/// The Qwen3-VL model: the language model's embedding, a prefix of its decoder layers, the vision
/// tower, and the final norm when the consumer reads normalized states.
pub struct Qwen3VlModel {
    config: Config,
    embedding: VocabParallelEmbedding,
    rotary: Rotary,
    stack: DecoderStack,
    tower: VisionTower,
    norm: Option<RmsNorm>,
}

impl Qwen3VlModel {
    /// The model of `config` read through `loader` in `dtype`: its first `layers` decoder layers at
    /// `split`'s rank of the tensor-parallel group, the tower whole with the processor rules
    /// `images` states, and the final norm when `final_norm` asks for it.
    ///
    /// # Errors
    ///
    /// A configuration without a tower, more layers than the checkpoint holds, a split that does
    /// not divide the heads, or a module the loader refuses.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &Config,
        images: (ProcessorConfig, PixelNorm),
        split: Split,
        dtype: DType,
        layers: u32,
        final_norm: bool,
    ) -> Result<Self> {
        let text = &config.text;
        let vision = config
            .vision
            .as_ref()
            .ok_or_else(|| Error::Config("a Qwen3-VL model states no vision tower".to_owned()))?;
        let mut language = loader.scope(LANGUAGE_MODEL);
        let embedding = VocabParallelEmbedding::load(
            &mut language.scope("embed_tokens"),
            text.vocab_size,
            text.hidden_size,
            split,
        )?;
        let rotary = decoder::rotary(&mut language.scope("rotary_emb"), text, text.rope, split)?;
        let stack = DecoderStack::load(
            &mut language,
            config,
            0..layers,
            // A prompt's key bound is its own rows, a launch value.
            AttentionKind::Dense {
                max_keys: u32::MAX,
                causal: true,
                window: None,
            },
            split,
            dtype,
            &mut |_, scope| Mlp::dense(scope, text, split),
        )?;
        let norm = final_norm
            .then(|| {
                RmsNorm::load(
                    &mut language.scope("norm"),
                    text.hidden_size,
                    text.rms_norm_eps,
                    text.rms_norm_gain,
                    dtype,
                )
            })
            .transpose()?;
        let (processor, pixels) = images;
        Ok(Self {
            tower: VisionTower::load(loader, vision, processor, pixels, dtype)?,
            config: config.clone(),
            embedding,
            rotary,
            stack,
            norm,
        })
    }

    /// The configuration this model was built for.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The vision tower.
    pub const fn tower(&self) -> &VisionTower {
        &self.tower
    }
}

impl Encoder for Qwen3VlModel {
    /// One output row per presented token, the images' grids as the batch states them.
    fn layout(&self, batch: &TokenBatch, images: &[ImageShape]) -> EncoderLayout {
        EncoderLayout {
            rows: batch.num_tokens,
            tokens: batch.num_tokens,
            images: images.to_vec(),
        }
    }

    fn encode(
        &self,
        f: &Forward,
        prompt: &EncoderPrompt<'_>,
        inputs: EncoderInputs<'_>,
        out: EncoderOutputs,
    ) -> nn::Result<()> {
        Ok(self.forward(f, prompt.ids, inputs, out)?)
    }
}

impl Qwen3VlModel {
    /// [`Encoder::encode`] over the prompt `ids`.
    fn forward(
        &self,
        f: &Forward,
        ids: &[u32],
        inputs: EncoderInputs<'_>,
        out: EncoderOutputs,
    ) -> Result<()> {
        let layout = inputs.layout;
        let tokens = u64::from(layout.tokens);
        let hidden = u64::from(self.config.text.hidden_size);
        if layout.rows != layout.tokens
            || inputs.batch.num_tokens != layout.tokens
            || ids.len() as u64 != tokens
            || inputs.ids.numel() != tokens
            || out.hidden.shape.dims() != [tokens, hidden]
            || inputs.images.len() != layout.images.len()
        {
            return Err(Error::Config(format!(
                "an encode of {} rows for {} tokens: {} batch tokens, {} host ids, {} device ids, \
                 an output of {:?}, {} images for {} grids",
                layout.rows,
                layout.tokens,
                inputs.batch.num_tokens,
                ids.len(),
                inputs.ids.numel(),
                out.hidden.shape.dims(),
                inputs.images.len(),
                layout.images.len()
            )));
        }
        let vision = self.tower.config();
        let grids: Vec<[u32; 3]> = layout.images.iter().map(|image| image.grid).collect();
        let merge = vision.spatial_merge_size;

        // The prompt's host values: its M-RoPE coordinates, axis-major, and the row each merged
        // vision row takes.
        let (image_grids, video_grids) = positions::run_grids(
            ids,
            &grids,
            vision.image_token_id,
            vision.video_token_id,
            merge,
        )?;
        let coordinates = positions::rope_index(
            ids,
            &image_grids,
            &video_grids,
            vision.image_token_id,
            vision.video_token_id,
            merge,
        )?;
        let mut axes = vec![0_i32; POSITION_AXES as usize * ids.len()];
        for (token, coordinate) in coordinates.iter().enumerate() {
            for (axis, value) in coordinate.iter().enumerate() {
                axes[axis * ids.len() + token] = i32::try_from(*value).map_err(|_| {
                    Error::Config("an M-RoPE coordinate wider than an i32".to_owned())
                })?;
            }
        }
        let coordinates = f.upload(&axes, Shape::new(&[u64::from(POSITION_AXES), tokens]))?;
        let rows =
            positions::vision_destinations(ids, vision.image_token_id, vision.video_token_id);
        let merged = rows.len() as u64;
        let destinations = f.upload(&rows, Shape::new(&[merged]))?;

        // The embeddings, the vision rows replacing the placeholders: the residual stream the
        // stack accumulates in place. Each image's tower runs in its own scope; its DeepStack
        // tables persist until their layers.
        let stream = self.embedding.forward(f, inputs.ids)?;
        let deepstack = (0..vision.deepstack_visual_indexes.len())
            .map(|_| f.empty(stream.dtype, Shape::new(&[merged, hidden])))
            .collect::<nn::Result<Vec<_>>>()?;
        let mut offset = 0_u64;
        for (grid, image) in grids.iter().zip(inputs.images) {
            offset += f.scoped(|f| -> Result<u64> {
                let (features, tables) = self.tower.forward(f, *grid, image.pixels)?;
                let count = features.rows();
                let landing = destinations.narrow(0, offset, count)?;
                IndexRows.scatter(f, features, landing, stream)?;
                for (table, kept) in tables.into_iter().zip(&deepstack) {
                    write(f, table, kept.narrow(0, offset, count)?)?;
                }
                Ok(count)
            })?;
        }
        if offset != merged {
            return Err(Error::Config(format!(
                "the images merge to {offset} rows and the prompt holds {merged} placeholders"
            )));
        }

        // The decoder layers over the packed prompt; DeepStack table `j` joins layer `j`'s output
        // at the vision rows (vLLM's `hidden_states + deepstack_input_embeds`).
        let rope = RopeFactors {
            rotary: &self.rotary,
            factors: self.rotary.factors(f, coordinates)?,
        };
        let keys = Keys::Dense(Some(Varlen {
            cu_q: inputs.batch.query_offsets,
            cu_k: inputs.batch.query_offsets,
            max_q: None,
            max_k: layout.tokens,
        }));
        let (last, residual) = self.stack.forward(
            f,
            stream,
            rope,
            |_| Ok(keys),
            |index, output| {
                if let Some(table) = deepstack.get(index)
                    && merged > 0
                {
                    IndexRows.scatter_add(f, *table, destinations, output)?;
                }
                Ok(())
            },
        )?;
        // The last output joins the stream: added and normalized, or, for a consumer that reads
        // the raw state, added alone; either lands in the runner's output rows.
        match &self.norm {
            Some(norm) => {
                let normalized = norm.forward(f, last, Some(ResidualStream::in_place(residual)))?;
                write(f, normalized, out.hidden)?;
            }
            None => {
                let joined = Residual::default().forward(f, residual, last, None)?;
                write(f, joined, out.hidden)?;
            }
        }
        Ok(())
    }
}
