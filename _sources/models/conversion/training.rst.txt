ERA5 to HadUK-Grid - train the Variational AutoEncoder
======================================================

Relies on already having a VAE for :doc:`HadUK-Grid <../../haduk-grid>`. It loads the weighs for that and freezes the weights for the generator. This forces the output to be compatible with making HadUK-Grid fields by assimilating observations (same generator in both cases => statistically compatible output). This is not necessary - we could just retrain the whole thing from scratch and it would still produce a good conversion, just one that was less consistent with the assimilation product.

.. literalinclude:: ../../../models/DCVAE_single_ERA5_to_HUKG_Tmax/autoencoder.py

