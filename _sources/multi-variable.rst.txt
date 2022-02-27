Finding mean-sea-level pressure by assimilating wind observations
=================================================================

What if we don't have observations of the variable we want? We ought to be able to generate fields of several variables from one location in latent space - one latent space location corresponds to the total weather state at one point in time. So we should be able to constrain the latent space with one variable, and then make the associated field of some other variable by using the appropriate generator. We're going to try this with pressure and wind, again using data from the `20th Century Reanalysis version 2c <https://psl.noaa.gov/data/gridded/data.20thC_ReanV2c.html>`_. We are going to reconstruct a mslp field using only wind observations.

The process is the same as :doc:`with just mslp <../mslp>` except that the VAE will take a set of three fields (mslp, u-wind at 10m, and v-wind at 10m). So we will have a single point in latent space (this time with 150 dimensions), but a generator that makes three fields from that point. 

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Get the training data <models/multi-variable/get_data>
   Convert the training data to tf tensors <models/multi-variable/to_tensor>
   Package the tensors into a tf Dataset <models/multi-variable/make_dataset>
   Specify the VAE <models/multi-variable/VAE>
   Train the VAE <models/multi-variable/training>
   Validate the trained VAE <models/multi-variable/validation>


.. figure:: figures/multi-variable/comparison.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   VAE validation: Red contours are from the input dataset, blue contours show generator output. The three rows are the variables: mslp - top, u10m - middle, v10m - bottom. The left hand column shows comparisons of a case from the training dataset, the middle column a case from the test dataset, and the right hand column an example of generator output. 

Assimilation is done exactly as :doc:`with mslp <../mslp>`, except we are assimilating u10m and v10m observations instead of pressure observations. We find the point in latent space that minimises the error in the wind obs. Then we use the generator to make the mslp field from that latent space location.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Details: DA by optimisation in latent space <models/multi-variable/optimiser>

.. figure:: figures/multi-variable/fit_U+V.jpg
   :width: 85%
   :align: center
   :figwidth: 85%

   Assimilation validation - fields of mslp: Red contours are from the input dataset, blue contours show generator output - for each of 25 samples with a different starting point in latent space. Black dots mark the locations of (wind) observations assimilated.

This works exactly as hoped - assimilating wind observations allow us to reconstruct the mslp field, where we have many observations the reconstruction is precise, where we have few observations it is very uncertain.

