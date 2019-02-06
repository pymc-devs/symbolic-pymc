#!/usr/bin/env python
from setuptools import find_packages, setup

classifiers = [
    "Development Status :: 4 - Beta",
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Operating System :: OS Independent'
]

setup(
    name="symbolic-pymc",
    version="0.0.1",
    install_requires=[
        'theano',
        'pymc3',
        'kanren',
        'multipledispatch',
        'unification',
        'sympy',
        'toolz',
    ],
    packages=find_packages(exclude=['tests']),
    tests_require=[
        'pytest'
    ],
    author="Brandon T. Willard",
    author_email="brandonwillard+symbolic_pymc@gmail.com",
    long_description="""Symbolic mathematics extensions for PyMC""",
    license="Apache License, Version 2.0",
    url="https://github.com/pymc-devs/symbolic-pymc",
    platforms=['any'],
    python_requires='>=3.6',
    classifiers=classifiers,
)
