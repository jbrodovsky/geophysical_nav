"""Setup script for the geonav package."""

from mypyc.build import mypycify
from setuptools import setup

setup(
    name="geonav",
    packages=["geonav", "ctest"],
    ext_modules=mypycify(paths=[
        #"./src/data_managment/dbmgr.py",
        #"./src/data_managment/m77t.py",
        "./src/ctest/fib.py"]),
)
