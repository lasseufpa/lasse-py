#!/usr/bin/env python
# will delete
from setuptools import find_packages, setup

setup(
    name="lasse-py",
    version="0.0.1",
    description="Python code for DSP, ML and telecom",
    author="LASSE",
    author_email="lasse",
    packages=find_packages(exclude=["tests"]),
    url="https://github.com/lasseufpa/lasse-py",
    install_requires=["numpy(>=1.14)"],
)
