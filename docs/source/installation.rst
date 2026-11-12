Installation
============

There are two main ways to install FluidGym: via PyPI or from the source code on
GitHub. Regardless of the installation method, it is recommended to set up a
dedicated Python virtual environment using tools like `venv` or `conda` to avoid
dependency conflicts.

FluidGym requires Linux, an NVIDIA GPU with a driver for CUDA 12.8 (≥ 570), and
Python 3.11–3.13.

1. Using PyPI
-------------

This is the simplest way to install FluidGym:

.. code-block:: bash

    pip install fluidgym

This also installs the GPU-accelerated solver
`phiPICT <https://github.com/upb-sciml/phiPICT>`_ together with
the PyTorch version it was built for (PyTorch 2.10 with CUDA 12.8). There is no
need to install PyTorch first: phiPICT's compiled extension only works with the
PyTorch version it was built against, so pip replaces any other version.

To use a different PyTorch or CUDA version, first install the matching phiPICT
wheel from the
`phiPICT wheel index <https://upb-sciml.github.io/phiPICT/docs/installation.html#other-pytorch-versions-wheel-index>`_
(or build phiPICT from source) and then FluidGym, e.g. for PyTorch 2.14 with
CUDA 12.6:

.. code-block:: bash

    pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu126
    pip install phipict -f https://upb-sciml.github.io/phiPICT/whl/torch-2.14+cu12.html
    pip install fluidgym

2. From GitHub
--------------

This is the best way to install the latest version of FluidGym. FluidGym itself
is pure Python; the solver still comes with phiPICT from PyPI. Clone the
repository and install the package:

.. code-block:: bash

    git clone https://github.com/upb-sciml/fluidgym.git
    cd fluidgym
    pip install .

To develop FluidGym, install it in editable mode together with the development
tools instead (needs pip ≥ 25.1):

.. code-block:: bash

    make install-dev
