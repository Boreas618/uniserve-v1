"""Stand-in programs the tests start in place of UniServe's processes.

Each module is a program of its own, run as
``[sys.executable, "-m", "uniserve_testing.standins.<name>", ...]`` by a test
or by the code under test (an evaluator profile's server command), and states
its arguments, its behaviour and what it reports in its documentation. A test
may also import a stand-in module to read the values it serves (the tokens of
:mod:`uniserve_testing.standins.server`).
"""
