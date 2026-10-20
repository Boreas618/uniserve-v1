"""Runners: each capability's forward, captured into a graph or run eagerly.

A runner has two halves with the same name. The half here is the forward
itself; the half in the worker runtime (``crates/worker/src/runner``) prepares
a forward's inputs, picks its graph and applies its outputs. A Python caller
uses this half directly.
"""
