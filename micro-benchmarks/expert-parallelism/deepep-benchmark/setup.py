import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

# Single source of truth: the script lives at the repo root (the reference
# runner and the code.amazon.com blob URL both read it from there). Copy it into
# the importable package at build time so it ships as package data and consumers
# can resolve it from the installed package.
_ROOT_SCRIPT = os.path.join(os.path.dirname(__file__), "setup_deepep_efa.sh")
_PKG_SCRIPT = os.path.join(os.path.dirname(__file__), "src", "deepep_efa_setup_script", "setup_deepep_efa.sh")


class BundleScript(_build_py):
    def run(self):
        shutil.copyfile(_ROOT_SCRIPT, _PKG_SCRIPT)
        os.chmod(_PKG_SCRIPT, 0o755)
        super().run()


setup(
    name="deepep_efa_setup_script",
    version="1.0.0",
    package_dir={"": "src"},
    packages=["deepep_efa_setup_script"],
    package_data={"deepep_efa_setup_script": ["setup_deepep_efa.sh"]},
    cmdclass={"build_py": BundleScript},
)
