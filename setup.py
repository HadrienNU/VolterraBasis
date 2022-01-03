#!/usr/bin/env python

import os
import setuptools

from distutils.core import setup, Extension

DISTNAME = "VolterraBasis"
DESCRIPTION = "Python module for the extraction of position dependant memory kernels from time series"
AUTHOR = "Hadrien Vroylandt"
AUTHOR_EMAIL = "hadrien.vroylandt@sorbonne-universite.fr"
URL = "https://github.com/HadrienNU/"
DOWNLOAD_URL = "https://github.com/HadrienNU/"
LICENSE = "new BSD"

# get __version__ from _version.py
ver_file = os.path.join("VolterraBasis", "_version.py")
with open(ver_file) as f:
    exec(f.read())

VERSION = __version__


INSTALL_REQUIRES = ["numpy>=1.15", "pandas>=0.23", "scipy>=1.1"]
EXTRAS_REQUIRE = {"docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"]}
ext_modules = [Extension(name="VolterraBasis.fkernel", sources=["VolterraBasis/fkernel.f90"], libraries=["lapack"])]


CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    ext_modules=ext_modules,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    packages=setuptools.find_packages(),
    package_data={"": ["*.f90"]},
)
