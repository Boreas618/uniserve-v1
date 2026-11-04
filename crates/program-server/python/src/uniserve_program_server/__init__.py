"""What the Program Server's Python environment adds to its compilers.

Python appears in the system only as the compiler of kernel programs (design
section 6.2). The environment under ``crates/program-server/python`` locks the
compilers and the external packages a kernel program imports; a package's own
``src/build.py`` is the compile driver, run by the Program Server with this
interpreter.

The one thing here besides the environment is :mod:`uniserve_program_server.
trtllm_gen`, which fetches, verifies and stages the pinned trtllm-gen FMHA
artifact that the attention package compiles and loads against (design
section 9).
"""
