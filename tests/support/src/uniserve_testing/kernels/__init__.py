"""What the kernel checks (``tests/python/kernels``) share.

:mod:`~uniserve_testing.kernels.packages` reaches the kernel packages outside
a build (a package's kernel source, the launch its table states, a built
module and the generated ABI headers), and
:mod:`~uniserve_testing.kernels.chain` runs an operator as a programmatic
dependent behind a late producer. The
subpackages, one per family of checks, hold the reference arithmetic and the
operands that several checks of the family share.
"""
