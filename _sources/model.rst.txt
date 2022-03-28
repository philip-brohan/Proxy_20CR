Building a generative model
===========================

We will use a `Variational AutoEncoder (VAE) <https://en.wikipedia.org/wiki/Variational_autoencoder>`_ as a generator factory. The VAE is based on an example in the `Tensorflow documentation <https://www.tensorflow.org/tutorials/generative/cvae>`_, kept simple, for speed: the generator and encoder are 5-layer convolutional neural nets.

.. figure:: figures/DCVAE_structure.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   The structure of the VAE used to train the generator

We train the VAE on 40-years of daily temperature anomalies from `ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_.


.. toctree::
   :titlesonly:
   :maxdepth: 1

   Get the training data <models/ERA5-T2m/get_data>
   Convert the training data to tf tensors <models/ERA5-T2m/to_tensor>
   Package the tensors into a tf Dataset <models/ERA5-T2m/make_dataset>
   Specify the VAE <models/ERA5-T2m/VAE>
   Train the VAE <models/ERA5-T2m/training>
   Validate the trained VAE <models/ERA5-T2m/validation>

