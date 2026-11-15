"""A CUDA-kind build module that records what its compilation was handed.

The compile request tests (``tests/test_compile.py``) run it as the Build
Server runs a package's build module: its module is the file ``handed.json``,
the request's typed values, so a test reads what the protocol delivered.
"""

import json

from uniserve_build_server.request import Request


def compile(request: Request) -> None:
    """Write the request's typed values as the module's only file."""
    handed = {
        "descriptor": request.descriptor,
        "fields": request.fields,
        "parameters": request.parameters,
        "target": str(request.target),
        "capability": request.target.capability,
        "families": list(request.target.families),
        "jobs": request.jobs,
        "nvcc": request.toolchain.require_cuda().nvcc,
    }
    (request.output / "handed.json").write_text(json.dumps(handed))
