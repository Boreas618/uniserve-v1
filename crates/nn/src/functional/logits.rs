//! The logits of a vocabulary-sharded head: the un-padding of the ranks' gathered shards.
//!
//! A tensor-parallel vocabulary head projects into each rank's shard of the vocabulary and the
//! ranks all-gather the shards; [`vocab_unpad`] reads each vocabulary column from the shard that
//! holds it and widens the logits into the type the sampler reads. Selecting the rows whose
//! logits a batch asks for is a row gather, which a caller writes as `index_select`.

use ops::logits::{UnpadArgs, UnpadDescriptor, VocabUnpad};
use tensor::{DType, Device, Tensor};

use super::{Fused, compute};
use crate::error::{Error, Result};

/// The element type of the logits a head returns: the sampler reads FP32.
pub const LOGITS: DType = DType::F32;

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
    const OWNER: &str = VocabUnpadCall::OWNER;

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

    compute(VocabUnpadCall {
        gathered,
        vocabulary,
        ranks,
        rows,
        padded,
        held,
    })
}

/// A checked call of [`vocab_unpad`]: `gathered` is `[rows, padded]`, and each of its shards holds
/// `held` vocabulary ids.
struct VocabUnpadCall<'t> {
    gathered: &'t Tensor,
    vocabulary: u32,
    ranks: u32,
    rows: usize,
    padded: usize,
    held: usize,
}

impl VocabUnpadCall<'_> {
    const OWNER: &'static str = "vocab_unpad";
}

impl Fused for VocabUnpadCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.gathered.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            gathered,
            vocabulary,
            rows,
            padded,
            held,
            ..
        } = self;
        let shards = self.ranks as usize;
        let width = padded / shards;
        let blocks = gathered.reshape((shards, rows, width))?;

        // Shard `r` holds `held` ids from `r · held` on, the last shard the remainder.
        let mut columns = Vec::with_capacity(shards);
        for shard in 0..shards {
            let first = (shard * held).min(vocabulary as usize);
            let count = held.min(vocabulary as usize - first);
            columns.push(blocks.get(shard)?.narrow(1, 0, count)?);
        }
        Ok(Tensor::cat(&columns, 1)?.to_dtype(LOGITS)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            gathered,
            vocabulary,
            rows,
            padded,
            ..
        } = self;
        let out = ops::empty((rows, vocabulary as usize), LOGITS)?;
        ops::launch::<VocabUnpad>(
            // The gathered width is a launch value that follows the deployment's rank count; the
            // two element types are separate because the head's product is in the model's type
            // and the sampler reads FP32: the logits are widened here, once.
            &UnpadDescriptor::new(
                gathered.dtype(),
                LOGITS,
                vocabulary,
                u32::try_from(padded)
                    .map_err(|_| Error::shape(Self::OWNER, "a gathered width above u32::MAX"))?,
            ),
            UnpadArgs {
                gathered: ops::operand(gathered)?,
                out: ops::operand(&out)?,
                ranks: self.ranks,
            },
        )?;
        Ok(out)
    }
}
