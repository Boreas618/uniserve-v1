"""Profiling sessions' files: a reader and a converter to a Perfetto trace.

A session (``POST /admin/profile``) leaves one SQLite file per worker in
UniServe's own schema (``docs/profiling.md``).
:class:`~uniserve_eval.profile.reader.ProfileFile` reads one;
:func:`~uniserve_eval.profile.perfetto.convert` turns a session's files into
one trace. ``python -m uniserve_eval.profile summary FILE...`` prints what
files hold, and ``python -m uniserve_eval.profile perfetto FILE... -o
TRACE.json`` writes the trace.
"""

from .perfetto import convert, write
from .reader import SCHEMA_VERSION, ProfileError, ProfileFile

__all__ = [
    "SCHEMA_VERSION",
    "ProfileError",
    "ProfileFile",
    "convert",
    "write",
]
