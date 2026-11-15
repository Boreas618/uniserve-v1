"""The command line and the determinism contract every generator shares.

A generator's fixtures are a function of its arguments and of the installed
packages alone:

- `--out` names the directory the fixtures are written under; nothing is
  written anywhere else.
- `--device` names the torch device the reference runs on. Weights and inputs
  are drawn on the CPU and moved there, so the draws do not depend on it.
- `--seed`, for a generator that draws, seeds every CPU `torch.Generator` it
  draws from; each draw takes the seed plus its own fixed offset.
- `torch.use_deterministic_algorithms(True)` holds for the whole run, so an
  operation with no deterministic implementation on the device raises instead
  of writing run-dependent values.

So a rerun with the same arguments in the locked environment writes the same
bytes.
"""

import argparse
from pathlib import Path

import torch


def options(description: str, *, seed: int | None = None) -> argparse.Namespace:
    """Parses a generator's command line and enters the determinism contract.

    Args:
        description: The generator's one-line description, for `--help`.
        seed: The generator's default base seed, or `None` for a generator
            that draws nothing and so takes no `--seed`.

    Returns:
        The arguments: `out` (a `Path`), `device` (a `torch.device`) and, when
        `seed` is given, `seed`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory the fixtures are written under",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="the torch device the reference runs on (default: cpu)",
    )
    if seed is not None:
        parser.add_argument(
            "--seed",
            type=int,
            default=seed,
            help=f"the base seed of every draw (default: {seed})",
        )

    arguments = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    return arguments
