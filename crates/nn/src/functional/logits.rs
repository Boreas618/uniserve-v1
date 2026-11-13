//! The logits of a vocabulary-sharded head: the un-padding of the ranks' gathered shards.
//!
//! A tensor-parallel vocabulary head projects into each rank's shard of the vocabulary and the
//! ranks all-gather the shards; [`vocab_unpad`] reads each vocabulary column from the shard that
//! holds it and widens the logits into the type the sampler reads. Selecting the rows whose
//! logits a batch asks for is a row gather, which a caller writes as `index_select`.

use ops::logits::{UnpadArgs, UnpadDescriptor, VocabUnpad};
use tensor::{DType, Tensor};

use super::{PDL, on_cpu};
use crate::error::{Error, Result};

/// The element type of the logits a head returns: the sampler reads FP32.
pub const LOGITS: DType = DType::F32;

/// Elements one `vocab_unpad` kernel handles.
const BLOCK: u32 = 1024;

/// The `[rows, vocabulary]` logits, in [`LOGITS`], of `gathered`: the shards of `ranks` ranks as
/// a collective gather lays them, one `[rows, width]` block per rank in rank order, stated as one
/// `[rows, width · ranks]` tensor. Shard `r` holds the vocabulary ids from
/// `r · ceil(vocabulary / ranks)` on, so its columns past the ids it holds are padding and are
/// dropped. An unsharded head's product is one shard.
///
/// # Errors
///
/// When `gathered` is not a matrix of `ranks` equal shards that hold the vocabulary, or the
/// operator refuses its arguments.
pub fn vocab_unpad(gathered: &Tensor, vocabulary: u32, ranks: u32) -> Result<Tensor> {
    const OWNER: &str = "vocab_unpad";

    let &[rows, padded] = gathered.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "gathered logits are [rows, width · ranks], not {:?}",
                gathered.dims()
            ),
        ));
    };
    let shards = ranks as usize;
    let held = (vocabulary as usize).div_ceil(shards.max(1));
    if shards == 0 || !padded.is_multiple_of(shards) || held > padded / shards {
        return Err(Error::shape(
            OWNER,
            format!(
                "{ranks} shards of a {padded}-column gather do not hold a vocabulary of \
                 {vocabulary}"
            ),
        ));
    }

    if on_cpu(gathered) {
        let width = padded / shards;
        let blocks = gathered.reshape((shards, rows, width))?;

        // Shard `r` holds `held` ids from `r · held` on, the last shard the remainder.
        let mut columns = Vec::with_capacity(shards);
        for shard in 0..shards {
            let first = (shard * held).min(vocabulary as usize);
            let count = held.min(vocabulary as usize - first);
            columns.push(blocks.get(shard)?.narrow(1, 0, count)?);
        }
        return Ok(Tensor::cat(&columns, 1)?.to_dtype(LOGITS)?);
    }

    let out = ops::empty((rows, vocabulary as usize), LOGITS)?;
    ops::launch::<VocabUnpad>(
        // The gathered width is a launch value that follows the deployment's rank count; the
        // two element types are separate because the head's product is in the model's type and
        // the sampler reads FP32: the logits are widened here, once.
        &UnpadDescriptor {
            dtype: gathered.dtype(),
            output: LOGITS,
            vocabulary,
            padded: u32::try_from(padded)
                .map_err(|_| Error::shape(OWNER, "a gathered width above u32::MAX"))?,
            block: BLOCK,
            pdl: PDL,
        },
        UnpadArgs {
            gathered: ops::operand(gathered)?,
            out: ops::operand(&out)?,
            ranks,
        },
    )?;
    Ok(out)
}
