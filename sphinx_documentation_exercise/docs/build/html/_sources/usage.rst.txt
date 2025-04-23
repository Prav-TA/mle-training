Usage Guide
============
This section provides a brief overview of how to use the code in this repository.

.. admonition:: Helpful Tip

It is preferrable to create the conda environment using the env.yml file provided in the repository.
This will ensure that all dependencies are installed correctly.

Go to the directory `sphinx_documentation_exercise` using the following command:

.. code-block:: bash

    cd sphinx_documentation_exercise

Then, run the following command to **create the conda environment**:

.. code-block:: bash

    conda env create -f env.yml

This will create a conda environment named `mle-dev` with all the required dependencies.

To **activate the environment**, run the following command:

.. code-block:: bash

    conda activate mle-dev

Finally to **run the script**, use the following command:

.. code-block:: bash

    python sphinx_lib/nonstandardcode.py

.. warning::

The script will download *dataset from the internet* and save it to the current directory.
Please ensure your device is *connected to the internet*