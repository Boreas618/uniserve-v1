"""What the Build Server's Python environment adds to its compilers.

Python appears in the system only as the compiler of kernel packages (design
section 6.2). The environment under ``crates/build-server/python`` locks the
compilers and the external packages a kernel package imports; a package's own
``src/build.py`` is its compile script, run by the Build Server with this
interpreter, and states what is particular to the package: which kernel one
descriptor compiles, and how.

This package holds what the compile scripts share:

- :mod:`uniserve_build_server.triton`: the Triton compile driver, which turns
  a package's ``spec`` into ``module.cubin`` and ``metadata.json``;
- :mod:`uniserve_build_server.cute_dsl`: the CuTe DSL compile driver, which
  turns a package's ``spec`` into a native ``module.so``;
- :mod:`uniserve_build_server.package`: a build's arguments, the descriptor's
  fields, the checked nvcc and a native module's exports;
- :mod:`uniserve_build_server.fetch`: the pinned third-party sources
  (trtllm-gen's kernel libraries, DeepEP, NCCL's headers, FlashInfer's MegaMoE),
  fetched, verified and cached by one content-addressed pattern;
- :mod:`uniserve_build_server.trtllm`: the build of the packages that launch
  trtllm-gen kernels.
"""
