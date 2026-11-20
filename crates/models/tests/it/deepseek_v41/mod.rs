//! DeepSeek-V4.1-Flash's tests, each against the outputs of its own reference generator:
//!
//! - [`reference`]: its n-gram hashing against the checkpoint's reference code
//!   (`generate_engram.py`, `data/engram.json`), and the hyper-connected residual and the n-gram
//!   memory of its layer 1, over the release's own parameters, against direct ports of the
//!   reference formulas;
//! - [`attention`]: the shared sparse latent attention modules over the checkpoint's parameters
//!   against its reference formulas (`generate_attention.py`);
//! - [`profile`]: the family's profile runs over the release's headers, whose kernel keys the
//!   tree's packages declare.

mod attention;
mod profile;
mod reference;
mod shards;
