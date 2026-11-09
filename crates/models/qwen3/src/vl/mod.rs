//! The Qwen3-VL language model and vision tower: a prompt's hidden states, its media spliced in.
//!
//! [`Qwen3VlModel`] is Hugging Face's `Qwen3VLModel` over a prefix of its decoder layers: the
//! tower's merged tokens replace the prompt's image and video placeholder tokens, the decoder
//! stack rotates by interleaved M-RoPE over each token's `(time, row, column)` coordinates, and a
//! consumer's hook runs between layers, which is where DeepStack features join the stream. A
//! consumer that truncates the stack loads a prefix of its layers, and one that reads the raw
//! hidden state loads no final norm: the last update is then added into the stream once.
//!
//! ## What the request decides, and where it enters
//!
//! Every value a request decides enters as a host table the encoder runner uploads
//! ([`Qwen3VlModel::state_tables`], filled by [`Qwen3VlModel::prepare_state`]): two follow the
//! *prompt* - its M-RoPE positions ([`positions::rope_index`]) and the prompt row each merged
//! vision row lands on ([`positions::vision_destinations`]) - and five follow the *layout*, the
//! vision tower's per-patch tables ([`vision`]). A captured call reads only the device copies.
//!
//! ## Precision and partitioning
//!
//! `dtype` is the checkpoint's own representation unless the deployment asked for another, and it
//! is decided once at the loading boundary: no layer converts. The language model is
//! tensor-parallel (query, key and value heads, the MLP's intermediate width and the embedding's
//! vocabulary are split, and the output and down projections reduce over the group). The vision
//! tower is *replicated* on every rank: it is small, every rank needs its features, and its
//! row-parallel projections would otherwise need a bias added once after the reduction.

pub mod interpolate;
pub mod positions;
pub mod vision;

use nn::blocks::{AttentionKind, Keys};
use nn::capability::{EncoderInputs, EncoderLayout, EncoderOutputs, StateTable, TableValues};
use nn::family::Split;
use nn::layers::{
    IndexRows, Residual, ResidualStream, RmsNorm, Rounding, Varlen, VocabParallelEmbedding,
};
use nn::weights::WeightLoader;
use nn::{DType, Forward, Shape, TensorView};

use crate::config::VlConfig;
use crate::decoder::{Components, DecoderStack, Hidden, LayerForm};
use crate::{Error, Result};
use vision::{BlockRows, ImageTables, Outputs, VisionTower};

/// The checkpoint's module path of the language model.
const LANGUAGE_MODEL: &str = "model.language_model";

/// What one encode call of a layout covers.
///
/// An encode call reads these numbers: the prompt's rows, the images of the request (one patch grid
/// each) and the language model's widths. They shape the call's tables and its arena reservations.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Plan {
    /// Rows the call writes: the packed prompt tokens, placeholders included.
    ///
    /// An encoder's output is one row per presented token, so this is the batch's `num_tokens`; the
    /// layout's `rows` is the capacity a worker reserves and must equal it (the model refuses a
    /// padded output, whose rows the batch's offsets do not describe).
    pub rows: u32,
    /// Merged vision rows of all the request's images together.
    pub merged: u32,
    /// Patch rows per image, in prompt order.
    pub images: Vec<u32>,
    /// Time steps per image, in prompt order.
    pub time: Vec<u32>,
    /// The images' patch grids as the layout stated them.
    pub grids: Vec<[u32; 3]>,
    /// The language model's width.
    pub hidden: u32,
    /// DeepStack feature tables the checkpoint's tower returns.
    pub deepstack: u32,
}

impl Plan {
    /// The plan of one prompt of `tokens` rows whose images have the patch grids `grids`.
    ///
    /// The grids are the request's host values; `grids.len()` is the number of tower calls. A
    /// text-only prompt has no grids.
    pub fn new(config: &VlConfig, tokens: u32, grids: &[[u32; 3]]) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::Config(
                "a prompt of no tokens has nothing to encode".to_owned(),
            ));
        }
        let mut images = Vec::with_capacity(grids.len());
        let mut time = Vec::with_capacity(grids.len());
        let mut merged = 0_u32;
        for &grid in grids {
            let rows = BlockRows::of(&config.vision, grid)?;
            merged = merged.checked_add(rows.merged).ok_or_else(|| {
                Error::Config("the images merge to more than a u32 of rows".to_owned())
            })?;
            images.push(rows.patches);
            time.push(rows.time);
        }
        Ok(Self {
            rows: tokens,
            merged,
            images,
            time,
            grids: grids.to_vec(),
            hidden: config.text.hidden_size,
            deepstack: config.vision.deepstack_visual_indexes.len() as u32,
        })
    }

    /// The plan of the layout a worker prepared.
    ///
    /// A layout that reserves more output rows than the prompt presents is refused: the batch's
    /// query offsets describe exactly the presented rows, and nothing masks a pad row.
    pub fn of_layout(config: &VlConfig, layout: &EncoderLayout) -> Result<Self> {
        let grids = layout
            .images
            .iter()
            .map(|image| image.grid)
            .collect::<Vec<_>>();
        let plan = Self::new(config, layout.tokens, &grids)?;
        if layout.rows != plan.rows {
            return Err(Error::Config(format!(
                "the layout reserves {} output rows for a {} token prompt",
                layout.rows, plan.rows
            )));
        }
        Ok(plan)
    }
}

/// The DeepStack features of one encode call, which a consumer's hook adds between layers.
pub struct DeepStack {
    /// `[count · merged, hidden]`: the tables, one after another.
    tables: TensorView,
    /// Merged vision rows of the call.
    merged: u64,
    /// Tables the tower returned.
    count: usize,
    /// `[merged]` I32: the prompt row of every merged vision row.
    destinations: TensorView,
}

impl DeepStack {
    /// How many feature tables the tower returned.
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether the tower returned no table.
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Table `index`, `[merged, hidden]`, or `None` for a call of no vision row.
    ///
    /// # Errors
    ///
    /// An index past the tables.
    pub fn features(&self, index: usize) -> Result<Option<TensorView>> {
        if index >= self.count {
            return Err(shape(format!("DeepStack table {index} of {}", self.count)));
        }
        if self.merged == 0 {
            return Ok(None);
        }
        Ok(Some(self.tables.narrow(
            0,
            index as u64 * self.merged,
            self.merged,
        )?))
    }

    /// `[merged]` I32: the prompt row each merged vision row of a table lands on.
    pub const fn destinations(&self) -> TensorView {
        self.destinations
    }
}

/// The Qwen3-VL model: the language model's embedding and decoder stack, the vision tower, and
/// the final norm when the consumer reads normalized states.
pub struct Qwen3VlModel {
    config: VlConfig,
    dtype: DType,
    embedding: VocabParallelEmbedding,
    stack: DecoderStack,
    tower: VisionTower,
    final_norm: Option<RmsNorm>,
}

impl Qwen3VlModel {
    /// The model of `config`, reading the checkpoint through `loader`: the first `layers` decoder
    /// layers at `split`'s rank of the tensor-parallel group, the vision tower whole, and the final
    /// norm when `final_norm` asks for it.
    ///
    /// `dtype` is the activation element type the loading boundary chose; `components` name the
    /// stack's modules to the deployment's fence probes.
    ///
    /// # Errors
    ///
    /// A module the loader refuses, a split that does not divide the heads, or more layers than the
    /// checkpoint holds.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &VlConfig,
        split: Split,
        dtype: DType,
        layers: u32,
        final_norm: bool,
        components: Components,
    ) -> Result<Self> {
        let text = &config.text;
        let mut language = loader.scope(LANGUAGE_MODEL);
        let embedding = VocabParallelEmbedding::load(
            &mut language.scope("embed_tokens"),
            text.vocab_size,
            text.hidden_size,
            split,
        )?;
        let form = LayerForm {
            // A call's key bound is its own prompt's rows, which the encoder admits whatever their
            // count: the layer states no ceiling below the descriptor's range.
            attention: AttentionKind::Dense {
                max_keys: u32::MAX,
                causal: true,
                window: None,
            },
            // The gate's product is rounded once, as the reference's `act_fn(gate) * up` in one
            // expression.
            rounding: Rounding::Fused,
            components,
        };
        let stack =
            DecoderStack::load(&mut language, text, form, split, 0..layers, |_, _| Ok(None))?;
        let final_norm = if final_norm {
            Some(RmsNorm::load(
                &mut language.scope("norm"),
                text.hidden_size,
                text.rms_norm_eps,
            )?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            dtype,
            embedding,
            stack,
            tower: VisionTower::load(loader, &config.vision, dtype)?,
            final_norm,
        })
    }

    /// The configuration this model was built for.
    pub const fn config(&self) -> &VlConfig {
        &self.config
    }

    /// The table names and shapes `layout` needs, in the order [`Self::prepare_state`] fills them.
    ///
    /// # Errors
    ///
    /// A layout the model cannot serve: the plan that cannot hold it refuses it here, rather than
    /// answering with tables no encode of it would read.
    pub fn state_tables(&self, layout: &EncoderLayout) -> Result<Vec<StateTable>> {
        let plan = Plan::of_layout(&self.config, layout)?;
        Ok(table_specs(
            &self.config,
            layout.rows,
            &plan.images,
            &plan.time,
            plan.merged,
        ))
    }

    /// Write the host values of every table of `layout` for the prompt `ids`, in
    /// [`Self::state_tables`]' order.
    ///
    /// # Errors
    ///
    /// A prompt whose placeholder runs do not hold the layout's grids, or buffers of other shapes
    /// than the tables'.
    pub fn prepare_state(
        &self,
        ids: &[u32],
        layout: &EncoderLayout,
        out: &mut [TableValues],
    ) -> Result<()> {
        write_state(&self.config, ids, layout, out)
    }

    /// Encode one prompt into `out.hidden`: its media's features, then the decoder layers over the
    /// spliced rows, with `after` run after each layer.
    ///
    /// `after` answers the state the next layer reads from layer `index`'s output and the call's
    /// DeepStack features; a consumer that adds no feature answers the state it is handed.
    ///
    /// # Errors
    ///
    /// Inputs that do not describe the layout's plan, or what a layer or `after` refuses.
    pub fn encode(
        &self,
        f: &Forward,
        inputs: EncoderInputs<'_>,
        out: EncoderOutputs,
        mut after: impl FnMut(usize, Hidden, &DeepStack) -> Result<Hidden>,
    ) -> Result<()> {
        let plan = Plan::of_layout(&self.config, inputs.layout)?;
        self.check(&inputs, &out, &plan)?;
        let prepared = self.prepared(&inputs, &plan)?;

        let tokens = u64::from(plan.rows);
        let hidden = out.hidden;
        let width = u64::from(plan.hidden);
        let merged = u64::from(plan.merged);
        // Without a final norm the residual stream lives in the output, where the last update is
        // added once; with one, the norm reads the stream from its own region and writes the
        // output.
        let stream = match self.final_norm {
            None => Some(hidden),
            Some(_) => None,
        };
        let buffers = self
            .stack
            .buffers(f.arena(), tokens, plan.hidden, self.dtype, stream)?;
        let features = f.arena().tensor(self.dtype, Shape::new(&[merged, width]))?;
        let deepstack = f.arena().tensor(
            self.dtype,
            Shape::new(&[u64::from(plan.deepstack) * merged, width]),
        )?;

        // 1. The vision tower, one call per image, in prompt order: the k-th merged row of the
        //    concatenation takes the k-th vision placeholder of the prompt.
        let mut offset = 0_u64;
        let mut patch_offset = 0_u64;
        let mut length_offset = 0_u64;
        let mut tables = Vec::with_capacity(self.tower.deepstack_features());
        for (index, image) in inputs.images.iter().enumerate() {
            let patches = u64::from(plan.images[index]);
            let time = plan.time[index];
            let vision = &prepared.vision;
            let image_tables = ImageTables::new(
                &self.config.vision,
                image,
                plan.images[index],
                time,
                &[
                    vision[0].narrow(0, patch_offset, patches)?,
                    vision[1].narrow(0, patch_offset, patches)?,
                    vision[2].narrow(0, patch_offset, patches)?,
                    vision[3].narrow(0, patch_offset, patches)?,
                    vision[4].narrow(0, length_offset, u64::from(time) + 1)?,
                ],
            )?;
            // A block's tower buffers are dead once its features are written, and the next block's
            // launches follow on the stream, so each block reuses the bytes of the one before it: a
            // request of several enlarged images took more than the arena holds.
            let block = f.arena().mark();
            let tower = self.tower.buffers(f.arena(), plan.images[index])?;
            let rows = u64::from(image_tables.rows.merged);
            tables.clear();
            for table in 0..self.tower.deepstack_features() {
                tables.push(deepstack.narrow(0, table as u64 * merged + offset, rows)?);
            }
            self.tower.encode(
                f,
                &image_tables,
                &tower,
                Outputs {
                    features: features.narrow(0, offset, rows)?,
                    deepstack: &tables,
                },
            )?;
            f.arena().release(block);
            offset += rows;
            patch_offset += patches;
            length_offset += u64::from(time) + 1;
        }
        debug_assert_eq!(offset, merged);

        // 2. The prompt's embeddings, with the placeholders replaced by the vision features.
        self.embedding.forward(f, inputs.ids, hidden)?;
        if !inputs.images.is_empty() {
            IndexRows.scatter(f, features, prepared.destinations, hidden)?;
        }

        // 3. The prompt's M-RoPE factors, once for the whole stack, and the attention descriptor
        //    whose key bound is the batch's own (the reference declares none).
        if inputs.batch.max_kv_len == 0 {
            return Err(shape(
                "the batch states no key bound its rows read".to_owned(),
            ));
        }
        let (cosine, sine) = self.stack.factors(f, prepared.positions, tokens)?;

        // 4. The decoder layers, the consumer's hook after each.
        let offsets = inputs.batch.query_offsets;
        let rows = u32::try_from(tokens)
            .map_err(|_| shape("a prompt longer than a key bound counts".to_owned()))?;
        let features = DeepStack {
            tables: deepstack,
            merged,
            count: self.tower.deepstack_features(),
            destinations: prepared.destinations,
        };
        let last = self.stack.forward(
            f,
            Hidden {
                hidden,
                residual: None,
            },
            cosine,
            sine,
            |_| {
                Keys::Dense(Some(Varlen {
                    cu_q: offsets,
                    cu_k: offsets,
                    max_q: None,
                    max_k: rows,
                }))
            },
            &buffers,
            |index, state| after(index, state, &features),
        )?;

        // 5. The last update joins the stream: added once into the output, or added and
        //    normalized into it.
        match (&self.final_norm, last.residual) {
            (None, Some(residual)) => {
                Residual::default().forward(f, residual, last.hidden, None, residual)?
            }
            (None, None) => {}
            (Some(norm), residual) => norm.forward(
                f,
                last.hidden,
                residual.map(ResidualStream::in_place),
                hidden,
            )?,
        }
        Ok(())
    }

    /// The encoder's tables read the way a call reads them.
    fn prepared<'a>(&self, inputs: &EncoderInputs<'a>, plan: &Plan) -> Result<Prepared<'a>> {
        let tables = table_specs(
            &self.config,
            plan.rows,
            &plan.images,
            &plan.time,
            plan.merged,
        );
        let state = inputs.state;
        if state.len() != tables.len() {
            return Err(shape(format!(
                "{} state tables for a layout of {}",
                state.len(),
                tables.len()
            )));
        }
        for (view, table) in state.iter().zip(&tables) {
            if view.dtype != table.dtype || view.shape != table.shape {
                return Err(shape(format!(
                    "the {} table is {:?} {:?}, {:?} {:?} expected",
                    table.name,
                    view.dtype,
                    view.shape.dims(),
                    table.dtype,
                    table.shape.dims()
                )));
            }
        }
        Ok(Prepared {
            positions: state[0],
            destinations: state[1],
            vision: &state[2..],
        })
    }

    /// Refuse a call whose batch, layout or outputs do not agree.
    ///
    /// The tables are checked against their declared shapes by [`Self::prepared`]; this checks the
    /// three host values a captured call is addressed by.
    fn check(&self, inputs: &EncoderInputs<'_>, out: &EncoderOutputs, plan: &Plan) -> Result<()> {
        let tokens = u64::from(plan.rows);
        if inputs.batch.num_tokens != plan.rows {
            return Err(shape(format!(
                "the batch packs {} tokens and the layout {}",
                inputs.batch.num_tokens, plan.rows
            )));
        }
        let hidden = u64::from(plan.hidden);
        if out.hidden.rows() != tokens || out.hidden.last() != hidden {
            return Err(shape(format!(
                "the output is {} rows of {}, a {tokens} row prompt of {hidden} expected",
                out.hidden.rows(),
                out.hidden.last()
            )));
        }
        if inputs.ids.numel() != tokens {
            return Err(shape(format!(
                "{} token ids for a {tokens} row prompt",
                inputs.ids.numel()
            )));
        }
        if inputs.images.len() != plan.images.len() {
            return Err(shape(format!(
                "{} images for a layout of {}",
                inputs.images.len(),
                plan.images.len()
            )));
        }
        Ok(())
    }
}

/// The tables a call reads, in the order [`table_specs`] declares them.
struct Prepared<'a> {
    /// `[axes, tokens]` I32: the prompt's M-RoPE coordinates.
    positions: TensorView,
    /// `[merged]` I32: the prompt row of every merged vision row, in prompt order.
    destinations: TensorView,
    /// The five vision tables, concatenated over the prompt's images in order.
    vision: &'a [TensorView],
}

/// A refusal of a call whose inputs do not describe its plan.
fn shape(problem: String) -> Error {
    Error::Nn(nn::Error::shape("Qwen3VlModel", problem))
}

/// Write the host values of every table of `layout`, in the order `table_specs` declares them.
///
/// The arithmetic is the reference's request arithmetic, evaluated in FP32 on the host
/// ([`positions`]): a captured call reads only the device copies. Nothing here allocates device
/// memory, synchronizes or reads anything back - `out` holds the worker's host buffers.
fn write_state(
    config: &VlConfig,
    ids: &[u32],
    layout: &EncoderLayout,
    out: &mut [TableValues],
) -> Result<()> {
    let plan = Plan::of_layout(config, layout)?;
    let tables = table_specs(config, layout.rows, &plan.images, &plan.time, plan.merged);
    if out.len() != tables.len() {
        return Err(Error::Config(format!(
            "{} buffers for {} tables",
            out.len(),
            tables.len()
        )));
    }
    let vision = &config.vision;
    let merge = vision.spatial_merge_size;
    let tokens = plan.rows as usize;
    if ids.len() != tokens {
        return Err(Error::Config(format!(
            "{} token ids for a {tokens} token prompt",
            ids.len()
        )));
    }

    // A vision grid reaches `rope_index` in the shape the prompt presents it: an image block is
    // one run of `time · rows · columns` placeholders with its whole grid, a video block is one
    // run per temporal patch, each of `rows · columns` placeholders on a single time step.
    let (image_grids, video_grids) = positions::run_grids(
        ids,
        &plan.grids,
        config.image_token_id,
        config.video_token_id,
        merge,
    )?;
    let coordinates = positions::rope_index(
        ids,
        &image_grids,
        &video_grids,
        config.image_token_id,
        config.video_token_id,
        merge,
    )?;
    // `[tokens][3]` to the operator's `[axes, tokens]`.
    let axes = VlConfig::POSITION_AXES as usize;
    let mut coordinates_by_axis = vec![0_i32; axes * tokens];
    for (token, coordinate) in coordinates.iter().enumerate() {
        for (axis, value) in coordinate.iter().enumerate() {
            coordinates_by_axis[axis * tokens + token] = i32::try_from(*value)
                .map_err(|_| Error::Config("an M-RoPE coordinate wider than an i32".to_owned()))?;
        }
    }
    out[0].write("prompt_positions", &TableValues::I32(coordinates_by_axis))?;

    let destinations =
        positions::vision_destinations(ids, config.image_token_id, config.video_token_id);
    if destinations.len() != plan.merged as usize {
        return Err(Error::Config(format!(
            "{} placeholders for {} merged vision rows",
            destinations.len(),
            plan.merged
        )));
    }
    out[1].write("vision_destinations", &TableValues::I32(destinations))?;
    if plan.images.is_empty() {
        return Ok(());
    }

    let side = BlockRows::position_grid(vision)?;
    let head_dim = BlockRows::head_dim(vision)?;
    let rotary = positions::VisionRotary::new(head_dim)?;
    let frequencies = vision::spectrum(vision)?.inverse_frequencies()?;
    let mut index = Vec::new();
    let mut weight = Vec::new();
    let mut cosine = Vec::new();
    let mut sine = Vec::new();
    let mut lengths = Vec::new();
    // The tables are one run over the prompt's blocks, in prompt order, so a block's rows sit after
    // its predecessors' and the check below counts against the running total, not the block's own.
    let mut covered = 0_usize;
    let columns = (head_dim / 2) as usize;
    for (grid, patches) in plan.grids.iter().zip(&plan.images) {
        let [time, height, width] = *grid;
        let (step_index, step_weight) = positions::position_samples(side, height, width, merge)?;
        let step_coordinates = positions::rotary_coordinates(height, width, merge)?;
        for _ in 0..time {
            index.extend(step_index.iter().flatten().copied());
            weight.extend(step_weight.iter().flatten().copied());
            let mut step_cosine = vec![0.0_f32; step_coordinates.len() * columns];
            let mut step_sine = vec![0.0_f32; step_coordinates.len() * columns];
            rotary
                .factors(
                    &step_coordinates,
                    &frequencies,
                    &mut step_cosine,
                    &mut step_sine,
                )
                .map_err(|error| {
                    Error::Config(format!("the rotary factors of {grid:?}: {error}"))
                })?;
            cosine.extend_from_slice(&step_cosine);
            sine.extend_from_slice(&step_sine);
        }
        lengths.extend_from_slice(&positions::vision_lengths(time, height, width)?);
        covered += *patches as usize;
        if index.len() != covered * positions::POSITION_TAPS as usize
            || weight.len() != covered * positions::POSITION_TAPS as usize
            || cosine.len() != covered * columns
            || sine.len() != covered * columns
        {
            return Err(Error::Config(format!(
                "the tables through the {grid:?} grid cover {} patches, {covered} expected",
                cosine.len() / columns
            )));
        }
    }
    out[2].write("vision_position_index", &TableValues::I32(index))?;
    out[3].write("vision_position_weight", &TableValues::F32(weight))?;
    out[4].write("vision_cosine", &TableValues::F32(cosine))?;
    out[5].write("vision_sine", &TableValues::F32(sine))?;
    Ok(out[6].write("vision_lengths", &TableValues::I32(lengths))?)
}

/// The tables' names and shapes for a prompt of `rows` rows whose images are a concatenation of
/// `patches` patch rows over `time` time steps, merging to `merged` rows.
///
/// Two tables follow the *prompt* - its M-RoPE positions and the prompt rows its vision rows land
/// on - so their values change with every request; the five vision tables follow the *layout*
/// alone, so a worker uploads them once per bucket. A layout with no images needs the two prompt
/// tables only.
///
/// Total in its arguments: the shapes are arithmetic on the row counts, so a worker can ask for
/// them before it knows whether the layout can be served.
fn table_specs(
    config: &VlConfig,
    rows: u32,
    patches: &[u32],
    time: &[u32],
    merged: u32,
) -> Vec<StateTable> {
    let tokens = u64::from(rows);
    let patch_rows: u64 = patches.iter().map(|patches| u64::from(*patches)).sum();
    let steps: u64 = time.iter().map(|time| u64::from(*time) + 1).sum();
    // The tower rotates on its own head width, which is not the language model's: the released
    // checkpoint states 1152 elements over 16 heads, so the factors carry 36 frequencies rather
    // than the language model's 64. The division is total here (this function reports nothing)
    // and any configuration that does not divide is refused by `write_state`, which derives the
    // same width through `BlockRows::head_dim`.
    let tower_columns = u64::from(config.vision.hidden_size / config.vision.num_heads.max(1) / 2);
    let table = |name: &'static str, dtype: DType, shape: Shape| StateTable { name, dtype, shape };
    let mut tables = vec![
        table(
            "prompt_positions",
            DType::I32,
            Shape::new(&[u64::from(VlConfig::POSITION_AXES), tokens]),
        ),
        table(
            "vision_destinations",
            DType::I32,
            Shape::new(&[u64::from(merged)]),
        ),
    ];
    if !patches.is_empty() {
        tables.push(table(
            "vision_position_index",
            DType::I32,
            Shape::new(&[patch_rows, positions::POSITION_TAPS]),
        ));
        tables.push(table(
            "vision_position_weight",
            DType::F32,
            Shape::new(&[patch_rows, positions::POSITION_TAPS]),
        ));
        tables.push(table(
            "vision_cosine",
            DType::F32,
            Shape::new(&[patch_rows, tower_columns]),
        ));
        tables.push(table(
            "vision_sine",
            DType::F32,
            Shape::new(&[patch_rows, tower_columns]),
        ));
        tables.push(table("vision_lengths", DType::I32, Shape::new(&[steps])));
    }
    tables
}

#[cfg(test)]
mod tests {
    use nn::capability::ImageShape;

    use super::*;

    /// The checkpoint's placeholder token ids.
    const IMAGE: u32 = 151_655;
    const VIDEO: u32 = 151_656;

    /// The released configuration.
    fn config() -> VlConfig {
        crate::config::fixtures::released()
    }

    /// A host table of the type and extents `table` states, as the worker hands it over.
    fn buffer_for(table: &StateTable) -> TableValues {
        TableValues::zeroed(table).expect("a host table")
    }

    /// The `index`-th element of an i32 host table.
    fn i32_at(table: &TableValues, index: usize) -> i32 {
        match table {
            TableValues::I32(values) => values[index],
            TableValues::F32(_) => panic!("an f32 table read as i32"),
        }
    }

    /// The `index`-th element of an f32 host table.
    fn f32_at(table: &TableValues, index: usize) -> f32 {
        match table {
            TableValues::F32(values) => values[index],
            TableValues::I32(_) => panic!("an i32 table read as f32"),
        }
    }

    #[test]
    fn a_text_prompt_needs_the_two_prompt_tables_only() {
        let config = config();
        let tables = table_specs(&config, 5, &[], &[], 0);
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].name, "prompt_positions");
        assert_eq!(tables[0].shape.dims(), &[3, 5]);
        assert_eq!(tables[1].name, "vision_destinations");
        assert!(tables[1].shape.numel() == 0);

        let ids = [7, 8, 9, 10, 11];
        let layout = EncoderLayout {
            rows: 5,
            tokens: 5,
            images: Vec::new(),
        };
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("a text prompt");

        // The positions are the reference's own, axis-major: axis `a`'s coordinates start at
        // `a * tokens`.
        let coordinates = positions::rope_index(
            &ids,
            &[],
            &[],
            config.image_token_id,
            config.video_token_id,
            config.vision.spatial_merge_size,
        )
        .expect("positions");
        for (token, coordinate) in coordinates.iter().enumerate() {
            for (axis, value) in coordinate.iter().enumerate() {
                assert_eq!(i32_at(&out[0], axis * 5 + token), *value as i32);
            }
        }
    }

    #[test]
    fn an_images_tables_are_one_time_steps_samples_repeated_over_its_grid() {
        let config = config();
        let merge = config.vision.spatial_merge_size;
        let side = BlockRows::position_grid(&config.vision).expect("a square table");
        let head_dim = BlockRows::head_dim(&config.vision).expect("a head width");
        // One 4x6 image block: 24 patches, 6 merged rows, after three prompt tokens.
        let ids = [7, 8, 9, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 10];
        let layout = EncoderLayout {
            rows: 10,
            tokens: 10,
            images: vec![ImageShape { grid: [1, 4, 6] }],
        };
        let tables = table_specs(&config, 10, &[24], &[1], 6);
        assert_eq!(tables.len(), 7);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("an image prompt");

        // The six merged rows take the six placeholders, in order.
        let destinations = (3..9)
            .map(|row| i32_at(&out[1], row - 3))
            .collect::<Vec<_>>();
        assert_eq!(destinations, vec![3, 4, 5, 6, 7, 8]);

        // The resample's taps are one time step's, in merge-block patch order.
        let (index, weight) = positions::position_samples(side, 4, 6, merge).expect("samples");
        for patch in 0..24 {
            for tap in 0..4 {
                assert_eq!(i32_at(&out[2], patch * 4 + tap), index[patch][tap]);
                assert_eq!(f32_at(&out[3], patch * 4 + tap), weight[patch][tap]);
            }
        }
        // The rotary factors are the 2-D rotation of each patch's own coordinates.
        let coordinates = positions::rotary_coordinates(4, 6, merge).expect("coordinates");
        let frequencies = vision::spectrum(&config.vision)
            .and_then(|spectrum| spectrum.inverse_frequencies())
            .expect("frequencies");
        let rotary = positions::VisionRotary::new(head_dim).expect("a rotary head");
        let columns = (head_dim / 2) as usize;
        let mut cosine = vec![0.0_f32; coordinates.len() * columns];
        let mut sine = vec![0.0_f32; coordinates.len() * columns];
        rotary
            .factors(&coordinates, &frequencies, &mut cosine, &mut sine)
            .expect("factors");
        for element in 0..cosine.len() {
            assert_eq!(f32_at(&out[4], element), cosine[element]);
            assert_eq!(f32_at(&out[5], element), sine[element]);
        }
        // One time step is one attention sequence.
        assert_eq!(i32_at(&out[6], 0), 0);
        assert_eq!(i32_at(&out[6], 1), 24);

        // The prompt's positions start the block at the running origin.
        let coordinates = positions::rope_index(
            &ids,
            &[[1, 4, 6]],
            &[],
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("positions");
        assert_eq!(i32_at(&out[0], 3), coordinates[3][0] as i32);
        assert_eq!(i32_at(&out[0], 10 + 3), coordinates[3][1] as i32);
    }

    #[test]
    fn a_video_is_one_run_per_temporal_patch_and_two_time_steps() {
        let config = config();
        let merge = config.vision.spatial_merge_size;
        // Two temporal patches of a 4x6 video: six placeholders each, behind a timestamp.
        let ids = [
            7, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, 8, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO,
            VIDEO, 9,
        ];
        let layout = EncoderLayout {
            rows: 15,
            tokens: 15,
            images: vec![ImageShape { grid: [2, 4, 6] }],
        };
        let tables = table_specs(&config, 15, &[48], &[2], 12);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("a video prompt");

        // 48 patches merge to 12 rows, on the twelve placeholders.
        let destinations = (0..12).map(|row| i32_at(&out[1], row)).collect::<Vec<_>>();
        assert_eq!(destinations, vec![1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]);
        // Two time steps, each one sequence of 24 patches.
        assert_eq!(i32_at(&out[6], 0), 0);
        assert_eq!(i32_at(&out[6], 1), 24);
        assert_eq!(i32_at(&out[6], 2), 48);

        // The reference reads a video as one single-step grid per temporal patch, and each run
        // advances the origin.
        let (images, videos) = positions::run_grids(
            &ids,
            &[[2, 4, 6]],
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("runs");
        assert!(images.is_empty());
        assert_eq!(videos, vec![[1, 4, 6], [1, 4, 6]]);
        let coordinates = positions::rope_index(
            &ids,
            &images,
            &videos,
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("positions");
        // One clock serves all three axes: a coordinate is the origin plus the patch's merged row
        // and column. The prompt's first token takes origin 0, the first run's patches sit at
        // `(1 + i, 1 + j)` because that token left origin 1, the run advances it by
        // `max(4, 6) / 2 = 3` to 4, the timestamp takes 4 and leaves 5, and the second run starts
        // there.
        assert_eq!(coordinates[0], [0, 0, 0]);
        assert_eq!(coordinates[1], [1, 1, 1]);
        assert_eq!(coordinates[6], [1, 2, 3]);
        assert_eq!(coordinates[7], [4, 4, 4]);
        assert_eq!(coordinates[8], [5, 5, 5]);
    }

    #[test]
    fn a_prompt_with_an_image_and_a_video_puts_both_in_one_table_run() {
        let config = config();
        let merge = config.vision.spatial_merge_size;
        // An image block of 4x4 after two text tokens, a comma, then a two-step 4x4 video: two
        // runs of four placeholders, behind its own text tokens.
        let ids = [
            7, 8, IMAGE, IMAGE, IMAGE, IMAGE, 9, VIDEO, VIDEO, VIDEO, VIDEO, 10, VIDEO, VIDEO,
            VIDEO, VIDEO, 11,
        ];
        let tokens = ids.len() as u32;
        let layout = EncoderLayout {
            rows: tokens,
            tokens,
            images: vec![
                ImageShape { grid: [1, 4, 4] },
                ImageShape { grid: [2, 4, 4] },
            ],
        };
        // The tables are one run over the prompt's blocks: sixteen patches of image, then the
        // video's thirty-two, in that order.
        let tables = table_specs(&config, tokens, &[16, 32], &[1, 2], 12);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("a prompt with an image and a video");

        // Four placeholders of an image, then four per temporal patch of the video: twelve merged
        // rows on twelve placeholders, in prompt order.
        let destinations = (0..12).map(|row| i32_at(&out[1], row)).collect::<Vec<_>>();
        assert_eq!(destinations, vec![2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]);
        // Both calls' rows are in the same tables: forty-eight patches of taps, factors and
        // sequence lengths, the image's one single-step sequence first. The factors carry the
        // tower's own head width, not the language model's.
        let tower_head = BlockRows::head_dim(&config.vision).expect("a head width");
        assert_eq!(out[2].len(), 48 * positions::POSITION_TAPS as usize);
        assert_eq!(out[4].len(), 48 * (tower_head / 2) as usize);
        let lengths = (0..5).map(|row| i32_at(&out[6], row)).collect::<Vec<_>>();
        assert_eq!(lengths, vec![0, 16, 0, 16, 32]);

        // The blocks are consumed in prompt order, so the video's two runs come after the image's
        // and each block advances the origin: the image leaves 2, its timestamp takes 3, the video
        // leaves 5, and its own timestamp leaves 6 for the second run.
        let (images, videos) = positions::run_grids(
            &ids,
            &[[1, 4, 4], [2, 4, 4]],
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("runs");
        assert_eq!(images, vec![[1, 4, 4]]);
        assert_eq!(videos, vec![[1, 4, 4], [1, 4, 4]]);
        let coordinates = positions::rope_index(
            &ids,
            &images,
            &videos,
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("positions");
        // The clock again: two text tokens leave origin 2, the image's four merged rows sit at
        // `(2 + i, 2 + j)`, the block advances by `max(4, 4) / 2 = 2` to 4, its timestamp takes 4
        // and leaves 5, the video's first run sits there, advances to 7, its timestamp leaves 8,
        // and the second run starts at 8.
        assert_eq!(coordinates[1], [1, 1, 1]);
        assert_eq!(coordinates[2], [2, 2, 2]);
        assert_eq!(coordinates[5], [2, 3, 3]);
        assert_eq!(coordinates[6], [4, 4, 4]);
        assert_eq!(coordinates[7], [5, 5, 5]);
        assert_eq!(coordinates[12], [8, 8, 8]);
    }

    #[test]
    fn a_buffer_of_the_wrong_size_is_refused() {
        let config = config();
        let layout = EncoderLayout {
            rows: 3,
            tokens: 3,
            images: Vec::new(),
        };
        let tables = table_specs(&config, 3, &[], &[], 0);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        out[0] = TableValues::I32(vec![0; 2]);
        assert!(write_state(&config, &[1, 2, 3], &layout, &mut out).is_err());
        // A layout that reserves rows the prompt does not present is refused too.
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        let wide = EncoderLayout {
            rows: 4,
            tokens: 3,
            images: Vec::new(),
        };
        assert!(write_state(&config, &[1, 2, 3], &wide, &mut out).is_err());
    }

    #[test]
    fn the_plan_counts_the_prompt_and_its_images() {
        let config = config();
        let plan = Plan::new(&config, 4096, &[[1, 24, 24], [4, 16, 16]])
            .expect("a prompt with two images");
        let merge = config.vision.spatial_merge_size;
        // The prompt's rows are the capacity the caller asked for; each image's patch rows are its
        // grid's patches, and its merged rows are the 2 by 2 blocks of each time step.
        assert_eq!(plan.rows, 4096);
        assert_eq!(plan.images, [24 * 24, 4 * 16 * 16]);
        assert_eq!(
            plan.merged,
            (24 / merge) * (24 / merge) + 4 * (16 / merge) * (16 / merge)
        );
        // One DeepStack tap per block the tower taps, and the width the text config states rather
        // than a number this file owns.
        assert_eq!(
            plan.deepstack,
            config.vision.deepstack_visual_indexes.len() as u32
        );
        assert_eq!(plan.hidden, config.text.hidden_size);
    }

    #[test]
    fn the_plan_of_a_text_prompt_has_no_vision_rows() {
        let config = config();
        let plan = Plan::new(&config, 128, &[]).expect("one prompt");
        assert!(plan.images.is_empty());
        assert_eq!(plan.merged, 0);
        assert_eq!(plan.rows, 128);
        assert!(Plan::new(&config, 0, &[]).is_err());
    }
}
