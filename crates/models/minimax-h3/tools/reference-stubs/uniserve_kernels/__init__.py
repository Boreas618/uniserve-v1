"""A stub for the compiled kernel package, so host-only reference arithmetic can be imported.

The diagnosis scripts read the reference's *table* arithmetic, which is Python; the kernel package
is required at import time by the activation module and is never called on this path. Anything that
does call it must fail loudly rather than silently return a number.
"""
