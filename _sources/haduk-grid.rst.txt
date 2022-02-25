Working with haduk-grid
=======================

A weakness of this ML-reanalysis approach is that we have been using samples from a preexisting reanalysis to train the VAE. This makes the process somewhat circular - it would be more powerful if we could generate the VAE some other way.

One possibility for this is to train the VAE on pure observations, and the easy place to start is with an existing observational dataset - here we are using `HadUK-Grid <https://www.metoffice.gov.uk/research/climate/maps-and-data/data/haduk-grid/haduk-grid>`_, specifically the daily maximum air temperature.

The process is the same as :doc:`with ERA5 global T2m data <../model>` except that the data format is slightly different, and we are only using 20 dimensions in the latent space (an arbitary decision, but there should be fewer degrees of freedom in the UK near-surface temperatures than the global ones).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Get the training data <models/haduk-grid/get_data>
   Convert the training data to tf tensors <models/haduk-grid/to_tensor>
   Package the tensors into a tf Dataset <models/haduk-grid/make_dataset>
   Specify the VAE <models/haduk-grid/VAE>
   Train the VAE <models/haduk-grid/training>
   Validate the trained VAE <models/haduk-grid/validation>

.. figure:: figures/haduk-grid/comparison.jpg
   :width: 65%
   :align: center
   :figwidth: 65%

   VAE validation: top left - original field, top right - generator output, bottom left - difference, bottom right - scatter original::output. The point anomalies in the original, at the locations of some of the stations used, are an artefact of the simple spatial covariance model used in the dataset gridding process. That the generator does not reproduce them `might` be an advantage. 

Assimilation is done exactly as :doc:`with ERA5 global T2m data <../optimiser>`.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Details: DA by optimisation in latent space <models/haduk-grid/optimiser>

.. figure:: figures/haduk-grid/fit_full.jpg
   :width: 65%
   :align: center
   :figwidth: 65%

   Assimilation validation: top left - original field, top right - generator output at the latent-space location that maximises fit to pseudo-observations at the station locations, bottom left - difference, bottom right - scatter original::output. This uses the locations of the 310 stations used in making the original field.

The DA method in this case gives a method for making a gridded field from the observations - but we already have such a method (that's how we made the HadUK-grid fields in the first place). That does not make the DA useless, however. It provides a method for making gridded fields (with uncertainty estimates) from many fewer observations - so it will be useful for extending the gridded fields back in time.


.. figure:: figures/haduk-grid/fit_decimated.jpg
   :width: 65%
   :align: center
   :figwidth: 65%

   Assimilation validation with decimated observations: top left - original field, top right - generator output at the latent-space location that maximises fit to pseudo-observations at the station locations, bottom left - difference, bottom right - scatter original::output. This uses the locations of only 31 of the stations used in making the original field (10%).
