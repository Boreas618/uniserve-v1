"""The shared sources of UniServe's kernel packages.

A kernel package (``crates/ops/kernels/<op>/<backend>``) compiles its own
sources and imports what several packages compile from here:

- :mod:`uniserve_kernels.dependent_launch`: the programmatic dependent launch
  prologue every Triton kernel that may launch dependently runs first;
- :mod:`uniserve_kernels.strided`: the address arithmetic of the tensor
  primitives' strided operands, and the launch words it takes.

The Build Server's environment installs this library and digests its files into
every module's build identity (``uniserve_build_server.toolchain``).
"""
