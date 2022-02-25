ERA5 to HadUK-Grid - specify a Variational AutoEncoder
======================================================

Matches the VAE for :doc:`HadUK-Grid <../../haduk-grid>`. This makes it possible to use the same weights (and so the same generator) in both cases. This is not strictly necessary, but it is nice.

.. literalinclude:: ../../../models/DCVAE_single_ERA5_to_HUKG_Tmax/autoencoderModel.py

