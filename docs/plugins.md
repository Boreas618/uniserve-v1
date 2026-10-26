# Plugins

A plugin is a Python package that adds kernel providers, model packages, worker hooks, communication buffers, weight sources or a CUDA graph policy to UniServe without changing its sources, as vLLM's `vllm.general_plugins`.

## Registering

A plugin package names a function in the entry point group `uniserve.plugins` of its packaging metadata:

```toml
[project.entry-points."uniserve.plugins"]
example = "example_plugin:register"
```

Every process that resolves a provider or a model architecture calls each such function once, first, with a `uniserve.plugins.Registry`:

```python
def register(registry):
    registry.provider("moe", "example", ExampleExperts)        # a moe Backend factory
    registry.provider("attention", "example", ExampleAttention)
    registry.model("ExampleForCausalLM", "example_plugin.model")
```

| Method | Adds |
| --- | --- |
| `provider(family, name, factory)` | A provider of the kernel family `attention`, `moe` or `matmul`: `factory()` returns the family's `Backend` (`uniserve/backends/<family>`), under `name` |
| `model(architecture, package)` | A model package (as those under `uniserve/models`) for checkpoints that declare `architecture` |
| `hook(name, function)` | A worker hook the admin API runs on every worker (`docs/admin.md`); `load_lora_adapter` and `unload_lora_adapter` make the plugin a LoRA plugin, whose kernels read each token's adapter slot through `uniserve.adapters.tokens()` |
| `buffer(name, factory)` | A communication buffer of the plugin's kernels (symmetric memory, multicast objects): every GPU worker calls `factory(models, device)` once its model is loaded and before it sizes the KV cache, keeps the returned object while it serves, and sizes the cache from the memory left, the buffer's memory that NVML attributes to no process included |
| `weight_source(scheme, factory)` | A source of a checkpoint's tensors named by URIs `scheme://...` (a tensor stream, CUDA IPC handles): `factory(uri)` returns an object with `identity()` (what it streams, which keys the retained weights), `tensors(source)` (each tensor's name, shape and element type) and `read(source, name)`. `load_config.weights` and `/admin/update_weights` name such a URI (`docs/admin.md`); the tensors load through the same layout conversion as the checkpoint's files |
| `graph_policy(function)` | The CUDA graphs of every worker: `function(models, configured)` receives the configured `mode` (`auto`, `full`, `off`) and the buckets `decode` (batch sizes), `prefill` (token counts) and `flow_shapes` (`[height, width]`), and returns the entries it changes; the worker checks them as the daemon checks `cuda_graph_config` and captures them. One plugin at most chooses |

A name may not shadow a built-in provider or architecture, and two plugins may not add the same one.

## Cache layer types

A model package's module owns a cache layer when it has a `cache_name` and a `cache_state(dtype)` method returning the layer's `uniserve.prefix_cache.StateConfig`; the decoder collects every such layer into the model's cache configuration (`TransformerDecoder.cache_config`). The `StateConfig` subclass is the cache layer's type, and states:

| Aspect | How the type states it |
| --- | --- |
| Shape | `buffers(num_blocks, block_size, dtype, quantizer)` declares the fields of `num_blocks` blocks; `bind(...)` borrows them as a `State` |
| Allocation unit | `indexing = "tokens"`: pages of `block_size` tokens in the unit pool; the type subclasses `mha.Config`, whose rows are `[heads, head_dim]`. `indexing = "states"`: one block per request slot, `[slots + 1, ...]` banks the worker allocates beside the unit pool and counts against the KV cache budget; slot 0 is the padding slot |
| Reuse and host KV cache rules | Pages follow their `window` (`None` keeps the whole history, a token count only the window before each reader). A deployment with a `"states"` layer reuses no prefix and copies no page through the host KV cache: a page does not hold the state a request reached |
| Transfer | `State.transfer_views` and `State.copy_blocks` borrow and copy whole blocks; a page transfer splits `heads` between tensor-parallel ranks |

In a forward the owning module reads its layer through `uniserve.prefix_cache.bound(module)`: the `State` and, for pages, the block table its entry of the attention batch is keyed by. A `"states"` layer reads the request slot of each row of the causal LM forward from `uniserve.prefix_cache.slots.rows()`; a row whose prefix is empty starts its request's state afresh. A paged layer owned by a module other than attention answers `reads_pages(pages, dtype)` for the page shapes the worker plans. The stub model's `TokenCount` (`uniserve/models/stub`) is the test implementation of a `"states"` type.

## Selecting

A plugin's provider serves where the configuration names a provider of its family, as it names a built-in one (for example `attention_backend = "example"`); automatic selection (`auto`) never picks a plugin's provider. A checkpoint of a plugin's architecture loads through the plugin's package.

## Updating

The installed plugins' distributions and versions enter the weights' fingerprint, since a provider declares the layout of the weights it serves. Installing, removing or upgrading a plugin is therefore an engine update (`POST /admin/update`, `specs/live-updates.md`): the standby engine imports the new plugin, loads the weights the plugin's providers read in their own layout, captures its graphs, and takes over without a restart.
