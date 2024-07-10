from setuptools import setup

from mypyc.build import mypycify

setup(
    name='geonav',
    packagages=['geonav'],
    ext_modules=mypycify(['geonav/geonav.py']),
)