"""What the Program Server's Python environment adds to its compilers.

Python appears in the system only as the compiler of kernel programs (design
section 6.2). The environment under ``crates/program-server/python`` locks the
compilers and the external packages a kernel program imports; a package's own
``src/build.py`` is the compile driver, run by the Program Server with this
interpreter.

Besides the environment, this package holds what the compile drivers share:

- :mod:`uniserve_program_server.package`: a build's arguments, the descriptor's
  fields, the checked nvcc and a native module's exports;
- :mod:`uniserve_program_server.fetch`: the pinned third-party sources
  (trtllm-gen's kernel libraries, DeepEP, NCCL's headers, FlashInfer's MegaMoE),
  fetched, verified and cached by one content-addressed pattern;
- :mod:`uniserve_program_server.trtllm`: the build of the packages that launch
  trtllm-gen kernels.
"""
