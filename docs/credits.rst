Authors and acknowledgements
----------------------------

This document is currently maintained by `Philip Brohan <https://brohan.org>`_ (philip.brohan @ metoffice.gov.uk). All criticism should be directed to him - put please **don't** send email, `raise an issue <https://github.com/philip-brohan/Proxy_20CR/issues/new>`_ instead.

|

All blame should go to the maintainer; credit is more widely distributed:

* This document was written by `Philip Brohan  <https://brohan.org>`_ (Met Office). He was supported by the Met Office Hadley Centre Climate Programme funded by BEIS and Defra, and by the UK-China Research & Innovation Partnership Fund through the Met Office Climate Science for Service Partnership (CSSP) China as part of the Newton Fund.
  
* This work follows on from `previous work on weather modelling with ML <https://brohan.org/ML_GCM/>`_.
 
* The `TensorFlow <https://www.tensorflow.org/>`_ library is used throughout.
  
* Training data used came from the `ERA5 reanalysis <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_ the `20th Century Reanalysis (version 2c) <https://www.esrl.noaa.gov/psd/data/20thC_Rean/>`_ and `HadUK-Grid <https://www.metoffice.gov.uk/research/climate/maps-and-data/data/haduk-grid/haduk-grid>`_.
    
* This work used the Isambard UK National Tier-2 HPC Service operated by GW4 and the UK Met Office, and funded by EPSRC (EP/P020224/1). This turned out to be overkill - all these models can be trained and run on a modern laptop - they don't need a GPU, but Isambard did provide a valuable speed-up.

* The calculations here make extensive use of `GNU Parallel <https://www.gnu.org/software/parallel/>`_ (`Tange 2011 <https://www.usenix.org/publications/login/february-2011-volume-36-number-1/gnu-parallel-command-line-power-tool>`_).
 
* This software is written in `python <https://www.python.org/>`_, in an environment configured with `conda <https://docs.conda.io/en/latest/>`_.

* The code and documentation use `git <https://git-scm.com/>`_ and `GitHub <https://github.com/>`_. The documentation is written with `sphinx <https://www.sphinx-doc.org/en/master/index.html>`_.

Note that appearance on this list does not mean that the person or organisation named endorses this work, agrees with any of it, or even knows of its existence.
