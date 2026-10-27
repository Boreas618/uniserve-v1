"""Root-mean-square normalizations as ordinary numerical modules."""

import torch
from torch import nn
from torch.nn import functional as F

from . import functional


class RMSNorm(nn.Module):
    """Normalize the final feature axis with FP32 variance accumulation.

    With ``elementwise_affine`` (the default) the normalized rows multiply a
    learned ``weight`` Parameter. Without it the rows are only normalized;
    ``weight`` is then a non-persistent unit buffer, so both forms share one
    normalization primitive and multiplying by one leaves every value exact.
    """

    function = "nn.norm.RMSNorm/1"

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or eps <= 0:
            raise ValueError(
                "RMS normalization requires positive width and epsilon"
            )
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            # The unit scale is a derived constant, not a checkpoint value:
            # it stays a real FP32 tensor under meta construction, and loading
            # moves it with its module.
            actual = torch.device("cpu" if device is None else device)
            if actual.type == "meta":
                actual = torch.device("cpu")
            self.register_buffer(
                "weight",
                torch.ones(hidden_size, device=actual, dtype=torch.float32),
                persistent=False,
            )
        self.eps = eps

    def forward(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return functional.rms_norm(x, self.weight, self.eps, out=out)


class GemmaRMSNorm(nn.Module):
    """RMS normalization whose weight is stored zero-centered.

    Rows are normalized in FP32 and scaled by ``1 + weight``, also in FP32,
    then rounded once to the input dtype: the norm of Gemma, Qwen3-Next and
    Qwen3.5 checkpoints, whose ``weight`` is the checkpoint's offset from
    one.
    """

    function = "nn.norm.GemmaRMSNorm/1"

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or eps <= 0:
            raise ValueError(
                "RMS normalization requires positive width and epsilon"
            )
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.eps = eps

    @property
    def scale(self) -> torch.Tensor:
        """The FP32 factor each normalized row is multiplied by."""
        return 1.0 + self.weight.float()

    def forward(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return functional.rms_norm(x, self.scale, self.eps, out=out)


class GatedRMSNorm(nn.Module):
    """RMS normalization of heads followed by a SiLU gate.

    ``forward(x, gate)`` normalizes the last axis of ``x`` in FP32, rounds
    it to ``x``'s dtype, multiplies by ``weight`` in that dtype and then by
    ``silu(gate)`` in FP32, rounding once more: the output norm of gated
    delta attention (``FusedRMSNormGated`` of the flash-linear-attention
    library and Transformers' ``Qwen3_5RMSNormGated``).
    """

    function = "nn.norm.GatedRMSNorm/1"

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or eps <= 0:
            raise ValueError(
                "RMS normalization requires positive width and epsilon"
            )
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        values = x.float()
        normalized = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.eps
        )
        weighted = self.weight * normalized.to(x.dtype)
        return (weighted * F.silu(gate.float())).to(x.dtype)
