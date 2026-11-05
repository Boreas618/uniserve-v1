"""The rope kernels' stub: same rule as the package's - import host arithmetic, never compute."""


def _refuse(*args, **kwargs):
    raise RuntimeError("uniserve_kernels.rope is a stub on this host: a diagnosis needs the build")


def __getattr__(name):
    return _refuse
