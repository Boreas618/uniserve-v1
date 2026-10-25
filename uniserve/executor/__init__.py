"""The numerical half of a worker rank: :class:`Executor`.

The worker (Rust) owns the control plane and the data plane: requests,
batches, lanes and their streams, product storage and transfers, the KV and
latent memory and the graph pools' device memory. The executor owns every
numerical decision and computation of the rank: the components' runners,
their buffers, layouts and buckets, graph capture and replay, sampling and
the media work of host components.

The worker calls the executor at a few points only, each from a job of the
lane whose stream the call works on (or at startup), with JSON for
structured values and tensors for storage:

- :meth:`Executor.manifest`: what capacity planning and the handshake read
  of the rank, after loading;
- :meth:`Executor.bind`: the storage capacity planning granted (request
  slots, lane streams, graph pools, latent banks) and the pipe host tasks
  report through; the executor constructs its runners;
- :meth:`Executor.warmup`: prepare, warm and capture before serving;
- :meth:`Executor.start` and :meth:`Executor.release`: the requests a batch
  starts, which the executor sizes and may refuse, and the slots the worker
  retires;
- :meth:`Executor.plan` and :meth:`Executor.execute`: one batch's reads and
  writes, then its computation on the tensors the worker leased and bound;
- :meth:`Executor.kernels` and :meth:`Executor.close`.

Within a call the executor never calls back into the worker.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import torch

__all__ = ["Executor"]


class Executor:
    """The executor of one worker rank.

    Args:
        loaded: The loaded model.
        config: JSON of the rank's deployment: ``rank``, ``device``,
            ``graphs`` (whether CUDA graphs serve), ``attention``,
            ``model_dtype``, ``max_model_len``, the placed ``components``
            (each with its ``name``, whether it is a ``host`` component, its
            declared ``calls`` (``capability`` and ``path``), the call
            ``kinds`` it serves, its ``ranks``, whether its media units are
            ``distributed`` and its ``units_per_rank``) and, for a video
            model, its ``video`` settings.
    """

    def __init__(self, loaded, config: str) -> None:
        settings = json.loads(str(config))
        self._video = None
        if settings.get("video") is not None:
            from uniserve.executor.video import VideoExecutor

            self._video = VideoExecutor(loaded, settings)

    def _served(self):
        if self._video is None:
            raise NotImplementedError(
                "the executor serves no component of this model"
            )
        return self._video

    def manifest(self) -> str:
        """JSON of what capacity planning and the handshake read."""
        return json.dumps({"video": self._served().manifest()})

    def bind(self, memory: str, *values: Any) -> None:
        """Construct the runners on the storage the worker granted.

        ``memory`` is JSON: ``slots``, the ``lanes`` (call kind to
        ``[stream handle, CUDA ordinal]``), the capabilities whose graph
        ``pools`` follow in ``values`` in that order, then one ``Scratch``
        per lane in the order of ``lanes``, whether the two latent ``banks``
        follow them, the ``reports`` pipe descriptor, the ``ffmpeg``
        executable and the ``transient_layouts`` a denoiser keeps.
        """
        settings = json.loads(str(memory))
        lanes = list(settings["lanes"].items())
        streams = {
            str(kind): torch.cuda.ExternalStream(
                int(handle), device=torch.device("cuda", int(ordinal))
            )
            for kind, (handle, ordinal) in lanes
        }
        names = [str(name) for name in settings["pools"]]
        rest = list(values)
        pools = dict(zip(names, rest[: len(names)], strict=True))
        del rest[: len(names)]
        scratches = {
            str(kind): scratch
            for (kind, _), scratch in zip(
                lanes, rest[: len(lanes)], strict=True
            )
        }
        del rest[: len(lanes)]
        banks = None
        if settings["banks"]:
            first, second = rest
            banks = (first, second)
        self._served().bind(settings, streams, pools, scratches, banks)

    def warmup(self) -> None:
        """Prepare, warm and capture every runner before serving."""
        self._served().warmup()

    def start(self, requests: str) -> str:
        """Admit the requests a batch started (JSON ``NewRequest`` list);
        returns JSON ``[[request_key, reason], ...]`` of those refused.
        """  # noqa: D205
        refused = self._served().start(json.loads(str(requests)))
        return json.dumps(
            [
                [
                    {
                        "engine_id": key[0],
                        "request_id": key[1],
                        "request_epoch": key[2],
                    },
                    reason,
                ]
                for key, reason in refused
            ]
        )

    def release(self, slots: Sequence[int]) -> None:
        """Forget the retired request slots."""
        self._served().release([int(slot) for slot in slots])

    def plan(self, step: str) -> str:
        """JSON of what each call of a batch reads and writes."""
        return json.dumps(self._served().plan_batch(json.loads(str(step))))

    def execute(self, step: str, *tensors: torch.Tensor) -> list[int]:
        """Run a batch on the tensors the worker leased and bound; returns
        how each call's forward ran (0 eager, 1 replay, 2 capture).
        """  # noqa: D205
        return [
            int(mode)
            for mode in self._served().execute(json.loads(str(step)), tensors)
        ]

    def kernels(self) -> str:
        """JSON ``[[label, [kernel record JSON, ...]], ...]``."""
        return json.dumps(self._served().kernels())

    def close(self, aborted: bool) -> None:
        """Release every runner; the worker drained the streams."""
        if self._video is not None:
            self._video.close(bool(aborted))
