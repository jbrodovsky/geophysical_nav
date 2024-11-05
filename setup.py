"""Setup script for the geonav package."""

# from mypyc.build import mypycify
from setuptools import setup, find_packages, Extension

navtoolbox: Extension = Extension(
    name="navtoolbox",
    sources=["src/navtoolbox/navtoolbox.cpp"],
    language="c++",
)

setup(
    name="geonav",
    packages=find_packages(),
    ext_modules=[navtoolbox],
    #    ext_modules=mypycify(paths=[
    #        #"./src/data_managment/dbmgr.py",
    #        #"./src/data_managment/m77t.py",
    #        "./src/ctest/fib.py"]),
)
