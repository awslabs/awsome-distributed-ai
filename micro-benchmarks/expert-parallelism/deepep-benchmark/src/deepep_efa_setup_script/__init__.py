"""Packaging shim for setup_deepep_efa.sh.

The build-and-test flow for DeepEP on EFA lives in the shell script
``setup_deepep_efa.sh`` at the root of this package. This module ships that
script as package data so consumers (e.g. ElasticCollectivesNightlyTests) can
resolve it from the installed package instead of vendoring their own copy:

    from importlib.resources import files
    script = files("deepep_efa_setup_script") / "setup_deepep_efa.sh"
"""
import os


def script_path() -> str:
    """Absolute path to the bundled setup_deepep_efa.sh."""
    return os.path.join(os.path.dirname(__file__), "setup_deepep_efa.sh")
