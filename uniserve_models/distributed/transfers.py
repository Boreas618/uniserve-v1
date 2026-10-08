"""Tensor copies around a module placed on another device.

A module placed on a device other than its caller's receives its tensor
arguments on its own device and returns its results to the caller's device,
or into caller-provided output storage. These copies are the numerical part
of cross-device module placement; the worker owns streams and ordering.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import torch

__all__ = ["copy_inputs", "copy_outputs"]


def _map(value, function):
    """Apply ``function`` to every tensor inside a nested argument structure."""
    if isinstance(value, torch.Tensor):
        return function(value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _map(getattr(value, field.name), function)
                for field in fields(value)
                if field.init
            },
        )
    if isinstance(value, Mapping):
        return {key: _map(member, function) for key, member in value.items()}
    if isinstance(value, tuple):
        mapped = tuple(_map(member, function) for member in value)
        return type(value)(*mapped) if hasattr(value, "_fields") else mapped
    if isinstance(value, list):
        return [_map(member, function) for member in value]
    return value


def _copy_out(value, out):
    """Return caller output storage after delivery from a remote module."""
    if isinstance(out, torch.Tensor):
        return out.copy_(value, non_blocking=out.device.type != "cpu")
    if isinstance(out, Mapping) and isinstance(value, Mapping):
        for name, destination in out.items():
            _copy_out(value[name], destination)
        return out
    raise TypeError(
        "cross-device output storage must be a tensor or named tensors"
    )


def copy_inputs(args, kwargs, device):
    """Move tensor arguments and return their caller's first device."""
    target = None

    def copy(value):
        nonlocal target
        if target is None:
            target = value.device
        return value.to(device, non_blocking=device.type != "cpu")

    return _map((args, kwargs), copy), target


def copy_outputs(value, target, out):
    """Copy module results into caller storage or back to its input device."""
    if out is not None:
        return _copy_out(value, out)
    return _map(
        value,
        lambda tensor: tensor.to(target, non_blocking=target.type != "cpu"),
    )
