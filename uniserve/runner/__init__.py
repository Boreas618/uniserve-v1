"""Runners: each capability's forward, captured into a graph or run eagerly.

A runner is the forward itself. The executor (``uniserve.executor``) prepares
a forward's inputs, picks its graph and applies its outputs; a Python caller
uses a runner directly.
"""
