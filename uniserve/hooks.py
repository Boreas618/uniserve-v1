"""Worker hooks: named operations the admin API runs on every worker.

The daemon's admin API (``/admin/load_lora_adapter``,
``/admin/unload_lora_adapter``, ``/admin/hooks/{name}``) has the engine run
the hook of that name on every worker of the deployment, at a step boundary:
no forward runs while a hook does. A plugin implements a hook
(``uniserve.plugins``, ``Registry.hook``) as a function of the request's JSON
payload and of the models the worker loaded (:func:`models`), returning a
JSON value. A hook no plugin implements fails, and the admin API answers so.

These are extension points: UniServe implements none of them itself. Hooks
named ``uniserve.*`` are the engine's own (sleep, wake, the KV cache's
storage), which the worker runs without this module.
"""

from __future__ import annotations

import json
import weakref
from typing import Any

import torch

from uniserve import plugins

# The models this worker loaded, as ``uniserve.loading.rank.load_model``
# returned them.
_models: list[weakref.ref] = []


def register_model(model: Any) -> None:
    """Record a model the worker loaded, for the hooks to reach."""
    _models.append(weakref.ref(model))


def models() -> list[Any]:
    """The models the worker loaded that are still alive."""
    return [model for model in (ref() for ref in _models) if model is not None]


def call(name: str, payload: str) -> str:
    """Run the hook ``name`` with the JSON ``payload``; its JSON result.

    The hook's device work completes before this returns, since the next
    forward may read what it wrote.

    Raises:
        LookupError: No plugin implements ``name``.
    """
    hook = plugins.registry().hooks.get(name)
    if hook is None:
        raise LookupError(
            f"no plugin implements the worker hook {name!r} "
            "(uniserve.plugins, Registry.hook)"
        )
    result = hook(json.loads(payload) if payload else None, models())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return json.dumps(result)
