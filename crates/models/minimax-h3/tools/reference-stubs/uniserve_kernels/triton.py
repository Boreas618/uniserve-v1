"""Every kernel entry point refuses: this stub exists to import host arithmetic, not to compute."""


def _refuse(*args, **kwargs):
    raise RuntimeError(
        "uniserve_kernels.triton is a stub on this host: the caller reached a kernel, and a "
        "diagnosis that needs one must run where the kernels are built"
    )


require_kernel = _refuse


def __getattr__(name):
    return _refuse
