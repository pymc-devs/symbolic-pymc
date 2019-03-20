#!/usr/bin/env python
import codecs
import re
from pathlib import Path

from setuptools import setup, find_packages


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
README_FILE = PROJECT_ROOT / "README.md"
VERSION_FILE = PROJECT_ROOT / "symbolic-pymc" / "__init__.py"

NAME = "Symbolic PyMC"
DESCRIPTION = "Symbolic mathematics for PyMC"
AUTHOR = "PyMC Developers"
AUTHOR_EMAIL = "pymc.devs@gmail.com"
URL = ("https://github.com/pymc-devs/symbolic-pymc",)


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_version():
    lines = open(VERSION_FILE, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (VERSION_FILE,))


if __name__ == "__main__":
    setup(
        name=NAME,
        version=get_version(),
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        install_requires=get_requirements(),
        packages=find_packages(exclude=['tests']),
        tests_require=[
            'pytest'
        ],
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        include_package_data=True,
    )
