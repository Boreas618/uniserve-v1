//! The model families, one module each, as vLLM's or candle's model files hold them: the
//! configuration read from a checkpoint, a module tree whose `forward`s compose tensor operations
//! and [`nn`] modules, the checkpoint's names, and the family values execution loads and request
//! processing serves.
//!
//! - [`qwen3`]: the Qwen3 decoder, Qwen3-MoE and the Qwen3.5-MoE language model with its drafter,
//!   and the Qwen3-VL encoder and vision tower;
//! - [`qwen4_exp`]: Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`), which shares the
//!   vision tower and the configuration readers of [`qwen3`];
//! - [`minimax_h3`]: MiniMax-H3 and FastH3, whose text encoder is [`qwen3`]'s Qwen3-VL model.
//!
//! The development feature `fixtures` adds the module `fixtures`, the paths of the tiny
//! checkpoints this crate's tests hold, for the tests of the crates that serve them.
//!
//! A family owns no device, stream, buffer, scheduler or request state (`AGENTS.md`, Model
//! Library Boundary): the crate depends on `nn`, the tensor library and `processing` only.

#[cfg(feature = "fixtures")]
pub mod fixtures;
pub mod minimax_h3;
pub mod qwen3;
pub mod qwen4_exp;

/// Every model family this library holds, the one registration of them: the engine executable
/// loads checkpoints through these (`nn::family::Registry`), and each family carries the request
/// processing side the daemon serves of it ([`nn::family::Family::request`]). A family is added
/// with its module here and its entry in this list.
pub const FAMILIES: &[nn::family::Family] = &[
    qwen3::family::FAMILY,
    qwen3::family::HYBRID_FAMILY,
    qwen4_exp::family::FAMILY,
    minimax_h3::family::FAMILY,
];

#[cfg(test)]
mod tests {
    /// The families the library registers claim distinct classes, so the registry the engine
    /// executable starts from accepts them and resolves each family's own class to it.
    #[test]
    fn the_registered_families_form_one_registry() {
        let registry = nn::family::Registry::new(super::FAMILIES.iter().copied())
            .expect("no two registered families claim one class");
        for family in super::FAMILIES {
            for class in family.request.classes {
                let resolved = registry
                    .by_class(class)
                    .expect("a registered class resolves");
                assert_eq!(resolved.id(), family.id(), "{class}");
            }
        }
    }
}
