"""Setup script for the geonav package."""

from setuptools import setup

from mypyc.build import mypycify


setup(
    name="geonav",
    packages=["geonav", "ctest"],
    ext_modules=mypycify(paths=["--disallow-untyped-defs", "./src/ctest/fib.py"]),
)
