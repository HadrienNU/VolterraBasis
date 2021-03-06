[![Documentation Status](https://readthedocs.org/projects/volterrabasis/badge/?version=latest)](https://volterrabasis.readthedocs.io/en/latest/?badge=latest)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Python 3 tool suite for the computation of position dependent memory kernels from time series

Please read (and cite) the following reference, where the details of the algorithm are explained.

Hadrien Vroylandt and Pierre Monmarché. *Position-dependent memory kernel in generalized Langevin equations: theory and numerical estimation* [arXiv:2201.02457](https://arxiv.org/abs/2201.02457)

Run

    pip install .

to install, and see `examples/` to get started.

To compile the documentation (the code should be installed first)


    cd doc/
    make html

And the documentation will be available in


    doc/_build/html/index.html

This package is based on the [memtools](https://github.com/lucastepper/memtools) tool suite.
