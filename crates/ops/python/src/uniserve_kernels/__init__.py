"""The shared sources of UniServe's kernel packages.

A kernel package (``crates/ops/kernels/<op>/<backend>``) compiles its own
sources and imports what several packages compile from here:

- :mod:`uniserve_kernels.dependent_launch`: the programmatic dependent launch
  prologue every Triton kernel that may launch dependently runs first;
- :mod:`uniserve_kernels.strided`: the address arithmetic of the tensor
  primitives' strided operands, and the launch words it takes;
- :mod:`uniserve_kernels.grouped_gemm`: the CuTe DSL computation the routed
  experts' grouped GEMMs (``gather_gemm`` and ``route_gemm``) share, which
  calls FlashInfer's pipelines and device functions;
- ``include/``, C++ headers a native package's units include:
  ``uniserve_cute_dsl.h``, what every CuTe DSL module's entry shares, and
  ``uniserve_trtllm/cubins.h``, the cubin loader and load entry of the modules
  that launch trtllm-gen kernels.

The Build Server's environment installs this library and digests its files into
every module's build identity (``uniserve_build_server.toolchain``).
"""
