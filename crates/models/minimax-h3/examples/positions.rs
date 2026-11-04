//! The rotary coordinates one case's layout addresses, written for comparison with the reference.
//!
//! `tables.position_ids` is the one table every rotation in this model reads: one
//! `(time, height, width)` triple per packed row. The reference computes the same triples
//! independently (`uniserve/models/minimax_h3/packing.py`, "position_ids: [padded_tokens, 3] FP64
//! rotary coordinates"), and until now nothing compared the two: a swapped height and width, or a
//! coordinate that is the pixel index where the latent index belongs, produces factors of the right
//! shape, type and count, so every host check agrees while every rotation turns the wrong way.
//!
//! The comparison needs no device, because both sides are host arithmetic:
//!
//! ```text
//! cargo run -q -p minimax-h3 --example positions -- --width 256 --height 256 --frames 22 \
//!     --text-tokens 60 --text-rows 128 --condition-rows 192 --out artifacts/h3-positions
//! .venv/bin/python uniserve/models/minimax_h3/tools/... # the reference side (dump_tables.py)
//! ```
//!
//! Our file is `positions.h3t` in the container the other dumps use, holding the **operator's own
//! layout** - axis-major, `[3, rows]` int32, which is what `ops::rope` documents ("A sectioned
//! rotation's positions are `[axes, tokens]`"). The reference's is `[rows, 3]`; the comparator
//! transposes, and says which convention it read, rather than either side hiding the difference.

use std::path::{Path, PathBuf};

use minimax_h3::{Canvas, Error, packing};

/// The container's dtype codes, shared with every other dump this crate writes.
const I32: u8 = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = Arguments::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--width" => arguments.width = value()?.parse()?,
            "--height" => arguments.height = value()?.parse()?,
            "--frames" => arguments.frames = value()?.parse()?,
            "--text-tokens" => arguments.text_tokens = value()?.parse()?,
            "--text-rows" => arguments.text_rows = value()?.parse()?,
            "--condition-rows" => arguments.condition_rows = value()?.parse()?,
            "--shard" => arguments.shard = value()?.parse()?,
            "--out" => arguments.out = PathBuf::from(value()?),
            other => return Err(format!("{other} is not a flag this probe takes").into()),
        }
    }
    arguments.run().map_err(Into::into)
}

/// One case: the pixel canvas, the frame count, and the layout's two capacities.
struct Arguments {
    width: u32,
    height: u32,
    frames: u32,
    text_tokens: u32,
    text_rows: u32,
    condition_rows: u32,
    shard: u32,
    out: PathBuf,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            frames: 22,
            text_tokens: 60,
            text_rows: 128,
            condition_rows: 192,
            shard: 1,
            out: PathBuf::from("h3-positions"),
        }
    }
}

impl Arguments {
    fn run(&self) -> Result<(), Error> {
        if self.width == 0 || self.height == 0 || self.frames == 0 {
            return Err(Error::Config(
                "a case needs a positive canvas and frame count".to_owned(),
            ));
        }
        if self.text_tokens == 0 || self.text_tokens > self.text_rows {
            return Err(Error::Config(format!(
                "{} prompt rows do not fit a text capacity of {}",
                self.text_tokens, self.text_rows
            )));
        }
        // The canvas's resolution class is the short edge, and its ratio the raster's own: a case
        // names a raster, and the rule that names canvases is not this probe's business.
        let canvas = Canvas {
            short_edge: self.width.min(self.height),
            aspect: (self.width, self.height),
            width: self.width,
            height: self.height,
        };
        // The deployment's own shard, which is the packing's token alignment: a layout at one rank
        // and the same layout under Ulysses are different row counts, and a table is compared with
        // the layout it was built for.
        let packing = packing::dense_packing(
            self.frames,
            &canvas,
            self.text_rows,
            self.condition_rows,
            self.shard,
        )?;
        // No conditions: this probe compares the layout's coordinates, and a segment's own rows are
        // the conditioning path's, which `encode_rows` covers.
        let tables =
            packing::dense_tables(&packing, self.frames, &canvas, self.text_tokens, &[], &[])?;
        let rows = tables.position_ids.len();
        if rows != packing.padded_tokens as usize {
            return Err(Error::Config(format!(
                "{rows} coordinate rows for {} padded rows",
                packing.padded_tokens
            )));
        }
        // Axis-major, the operator's layout: every row's time, then every row's height, then every
        // row's width. The comparator transposes to the reference's `[rows, 3]`.
        let mut payload = Vec::with_capacity(rows * 3 * 4);
        for axis in 0..3 {
            for point in &tables.position_ids {
                payload.extend_from_slice(&(point[axis] as i32).to_le_bytes());
            }
        }
        std::fs::create_dir_all(&self.out)
            .map_err(|error| Error::Config(format!("{}: {error}", self.out.display())))?;
        write_container(
            &self.out.join("positions.h3t"),
            &[3, rows as u64],
            I32,
            &payload,
        )?;
        println!(
            "case: {}x{}, {} frames, prompt {} of {} rows, conditions {} of {} rows, shard {}",
            self.width,
            self.height,
            self.frames,
            self.text_tokens,
            self.text_rows,
            self.condition_rows,
            packing.condition_rows,
            self.shard
        );
        println!(
            "layout: {} packed rows = {} video + {} audio + {} text + {} condition",
            rows, packing.video_rows, packing.audio_rows, packing.text_rows, packing.condition_rows
        );
        println!(
            "wrote {}: [3, {rows}] int32, axis-major - the operator's own layout",
            self.out.join("positions.h3t").display()
        );
        println!(
            "rows 0..{} hold the prefix, so a swapped axis shows there first",
            self.text_rows + self.condition_rows + 1
        );
        Ok(())
    }
}

/// Write one tensor in the container the other dumps use: `H3D1`, a dtype code, a rank, the extents,
/// then the payload little-endian.
fn write_container(path: &Path, shape: &[u64], dtype: u8, payload: &[u8]) -> Result<(), Error> {
    let mut bytes = Vec::with_capacity(6 + shape.len() * 8 + payload.len());
    bytes.extend_from_slice(b"H3D1");
    bytes.push(dtype);
    bytes.push(u8::try_from(shape.len()).unwrap_or(0));
    for extent in shape {
        bytes.extend_from_slice(&extent.to_le_bytes());
    }
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes)
        .map_err(|error| Error::Config(format!("{}: {error}", path.display())))
}
