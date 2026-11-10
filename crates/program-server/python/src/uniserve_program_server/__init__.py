"""What the Program Server's Python environment adds to its compilers.

Python appears in the system only as the compiler of kernel programs (design
section 6.2). The environment under ``crates/program-server/python`` locks the
compilers and the external packages a kernel program imports; a package's own
``src/build.py`` is its compile script, run by the Program Server with this
interpreter, and states what is particular to the package: which kernel one
descriptor compiles, and how.

This package holds what the compile scripts share:

- :mod:`uniserve_program_server.triton`: the Triton compile driver, which turns
  a package's ``spec`` into ``module.cubin`` and ``metadata.json``;
- :mod:`uniserve_program_server.cute_dsl`: the CuTe DSL compile driver, which
  turns a package's ``spec`` into a native ``module.so``;
- :mod:`uniserve_program_server.package`: a build's arguments, the descriptor's
  fields, the checked nvcc and a native module's exports;
- :mod:`uniserve_program_server.fetch`: the pinned third-party sources
  (trtllm-gen's kernel libraries, DeepEP, NCCL's headers, FlashInfer's MegaMoE),
  fetched, verified and cached by one content-addressed pattern;
- :mod:`uniserve_program_server.trtllm`: the build of the packages that launch
  trtllm-gen kernels.
"""
