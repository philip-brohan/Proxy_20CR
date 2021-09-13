How to reproduce and extend this work
=====================================

This project is designed to be easy to reproduce and extend. Everything involved is kept under version control in a `git repository <https://en.wikipedia.org/wiki/Git>`_. The repository is hosted on `GitHub <https://github.com/>`_ (and the documentation made with `GitHub Pages <https://pages.github.com/>`_). The repository is `<https://github.com/philip-brohan/Proxy_20CR>`_; it contains everything you need to reproduce or extend this work.

If you are familiar with GitHub, you already know what to do (fork or clone `the repository <https://github.com/philip-brohan/Proxy_20CR>`_): If you'd prefer not to bother with that, you can download the whole thing as `a zip file <https://github.com/philip-brohan/Proxy_20CR/archive/master.zip>`_.


As well as downloading the software, some setup is necessary to run it successfully:

These scripts need to know where to put their output files. They rely on an environment variable ``SCRATCH`` - set this variable to a directory with plenty of free disc space.

These scripts will only work in a environment with the appropriate software and libraries available. I use `conda <https://docs.conda.io/en/latest/>`_ to manage the required environment - which is specified in a yaml file:

.. literalinclude:: ../environments/ProxyR.yml

Install `anaconda or miniconda <https://docs.conda.io/en/latest/>`_, `create and activate the environment in that yaml file <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs>`_, and all the scripts in this repository should run successfully.

The project documentation (these web pages) are included in the repository (in the `docs directory <https://github.com/philip-brohan/Proxy_20CR/tree/main/docs>`_). The documentation is in `reStructuredText <https://en.wikipedia.org/wiki/ReStructuredText>`_ format, and uses the `Sphinx <https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html>`_ documentation generator.


