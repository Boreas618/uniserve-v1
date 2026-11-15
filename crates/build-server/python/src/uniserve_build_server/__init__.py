"""What the Build Server's Python environment adds to its compilers.

Python appears in the system only as the compiler of kernel packages (design
section 6.2). The environment under ``crates/build-server/python`` locks the
compilers and the external packages a kernel package imports; the Build Server
reads its toolchain description once and runs every compilation through it,
and a package's build module states what is particular to the package: which
kernel one descriptor compiles, and how.

- :mod:`uniserve_build_server.__main__`: the ``uniserve-build-server`` program
  (``toolchain --json``, ``compile``, ``provision``);
- :mod:`uniserve_build_server.toolchain`: the toolchain description;
- :mod:`uniserve_build_server.request`: the compile request and the target's
  hardware facts (``targets.toml``);
- :mod:`uniserve_build_server.triton`: the Triton compile driver, which turns
  a package's ``spec`` into ``module.cubin`` and ``metadata.json``;
- :mod:`uniserve_build_server.cute_dsl`: the CuTe DSL compile driver, which
  turns a package's ``spec`` into a native ``module.so``;
- :mod:`uniserve_build_server.fetch`: the pinned third-party sources
  (trtllm-gen's kernel libraries, DeepEP, NCCL's headers, FlashInfer's MegaMoE),
  one data file each under ``pins/``, fetched, verified and cached by one
  content-addressed pattern;
- :mod:`uniserve_build_server.trtllm`: the build of the packages that launch
  trtllm-gen kernels.
"""
