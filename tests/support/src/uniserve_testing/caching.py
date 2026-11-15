"""What a deployment's prefix cache reused for the requests it answered.

A chat completion states, as its usage's
``prompt_tokens_details.cached_tokens``, the prompt tokens its admission
found cached and did not compute again, in every tier together: the device,
the host KV cache, the peers' host KV caches and a remote store
(``proto::request::Event::Scheduled``). It is never the whole prompt, whose
last token is computed; for a prompt the deployment holds whole, such as the
same prompt asked again on the same deployment, it is the reuse limit: the
prompt less its last token, in whole blocks. A case reads both what a
request reused and how much a prompt can reuse from the answers themselves,
rather than from the scheduler's statistics or its block arithmetic.
"""

from __future__ import annotations

from typing import Any


def cached_tokens(answer: dict[str, Any]) -> int:
    """The prompt tokens the admission of `answer`'s request found cached
    in any tier (``usage.prompt_tokens_details.cached_tokens``).
    """  # noqa: D205
    return int(answer["usage"]["prompt_tokens_details"]["cached_tokens"])
