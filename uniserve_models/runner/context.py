"""The execution context a media runner constructs for its module."""

from __future__ import annotations

from uniserve_models.bindings.context import ExecutionContext, Scratch


def execution_context(
    module, stream: int, attention: str, scratch: Scratch
) -> ExecutionContext:
    """An execution context of ``module`` on a caller's stream.

    Args:
        module: The module whose operators the context prepares.
        stream: The CUDA stream handle the context enqueues on (0 for the
            current stream).
        attention: The attention backend the model loaded with.
        scratch: The transient work areas of the stream, which every
            context serialized on it borrows. Contexts are prepared largest
            first, so captured graphs keep addressing one backing.
    """
    return ExecutionContext(
        module,
        stream=int(stream) if stream else None,
        attention=attention,
        vsa="auto",
        matmul="auto",
        moe="auto",
        derive_host_lengths=False,
        scratch=scratch,
    )
