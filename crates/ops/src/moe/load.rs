//! The expert load: execution's counters of the tokens each routed layer's experts take.
//!
//! A serving deployment counts, for every routed layer, how many tokens each of its experts
//! processed, which is what a redistribution of the experts balances. Execution owns the counters:
//! one `[experts]` I64 tensor of device memory per layer, which it allocates, zeroes, reads back
//! and releases. It lends them to the forwards of a launching context ([`Context::set_expert_load`])
//! by the module path each layer is named by among the model's routed layers, and a routed block
//! hands its layer's counters to its top-k ([`TopkArgs::load`](super::TopkArgs::load)), which adds
//! each live token's routes in place. The counters are launch values, so a graph captured with them
//! adds into them on every replay; their addresses stay the same for as long as the context lends
//! them.
//!
//! A context lends no load unless execution set one: a meta context, a rehearsal and a forward on
//! CPU tensors count nothing.

use std::collections::BTreeMap;
use std::sync::Arc;

use tensor::Tensor;

use crate::context::{self, Context};

/// The counters a context lends its forwards' routed layers, by the layer's module path.
#[derive(Clone, Default)]
pub struct ExpertLoad {
    /// Per layer, its `[experts]` I64 counters, indexed by the model's own expert id.
    layers: Arc<BTreeMap<String, Tensor>>,
}

impl ExpertLoad {
    /// The load over `layers`: each routed layer's module path with its `[experts]` I64 counters,
    /// external tensors over memory the caller keeps valid, at their addresses, for as long as a
    /// context lends them or a graph captured under it replays.
    pub fn new(layers: BTreeMap<String, Tensor>) -> Self {
        Self {
            layers: Arc::new(layers),
        }
    }

    /// The counters of the routed layer at module path `module`, when the load holds them.
    pub fn layer(&self, module: &str) -> Option<&Tensor> {
        self.layers.get(module)
    }
}

impl Context {
    /// The expert load this context lends its forwards' routed layers, or `None` to count
    /// nothing. A context beside this one, made afterwards, lends the same counters
    /// ([`Context::beside`]): a forward captured there replays here.
    pub fn set_expert_load(&self, load: Option<ExpertLoad>) {
        self.state().load = load;
    }

    /// The expert load this context lends, as [`Context::set_expert_load`] set it.
    pub fn expert_load(&self) -> Option<ExpertLoad> {
        self.state().load.clone()
    }
}

/// The counters of the routed layer at module path `module` that the context installed on this
/// thread lends; `None` with no context installed, or one that lends no load for the layer.
pub fn layer_load(module: &str) -> Option<Tensor> {
    let context = context::installed()?;
    let state = context.state();

    state.load.as_ref()?.layer(module).cloned()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tensor::DType;

    use super::{ExpertLoad, layer_load};
    use crate::Context;

    #[test]
    fn a_layer_reads_the_counters_the_installed_context_lends_it() {
        let counters = crate::meta_tensor(8, DType::I64).unwrap();
        let context = Context::meta(None);
        context.set_expert_load(Some(ExpertLoad::new(BTreeMap::from([(
            "layers.1.mlp.experts".to_owned(),
            counters,
        )]))));

        assert!(layer_load("layers.1.mlp.experts").is_none());
        let _guard = context.enter();
        let lent = layer_load("layers.1.mlp.experts").unwrap();
        assert_eq!((lent.dims(), lent.dtype()), (&[8][..], DType::I64));
        assert!(layer_load("layers.0.mlp.experts").is_none());

        context.set_expert_load(None);
        assert!(layer_load("layers.1.mlp.experts").is_none());
    }
}
