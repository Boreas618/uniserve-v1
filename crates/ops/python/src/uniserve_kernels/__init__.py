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
- ``include/``, C++ headers a native package's units include: the C ABI
  headers, each operator's call structure (``uniserve_<operator>.h``) and what
  they share (``uniserve_abi.h``), generated from the operator crate's
  definitions (``ops::abi`` in ``crates/ops``) and compared with them by its
  tests; ``uniserve_cute_dsl.h``, what every CuTe DSL module's entry shares;
  ``uniserve_trtllm/cubins.h``, the cubin loader and load entry of the modules
  that launch trtllm-gen kernels; and ``uniserve_trtllm/routed_gemm.h``, the
  entry of the routed NVFP4 products, whose packages state what differs.

The Build Server's environment installs this library, digests its files into
every module's build identity and passes ``include/`` as the include directory
of every native compilation (``uniserve_build_server.toolchain``).
"""
