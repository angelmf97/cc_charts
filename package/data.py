"""Factory of Signatures.

This internal class is a factory of different signatures.
It is convenient because it allows initialization of different classes from
input string ``cctype``.

Also feature a method to "signaturize" an enternal matrix.
"""
import os
import h5py
import numpy as np
from chemicalchecker.util import logged


@logged
class DataFactory():
    """DataFactory class."""

    @staticmethod
    def make_data(cctype, *args, **kwargs):
        """Initialize *any* type of Signature.

        Args:
            cctype(str): the signature type: 'sign0-3', 'clus0-3', 'neig0-3'
                'proj0-3'.
            args: passed to signature constructor
            kwargs: passed to signature constructor
        """
        from chemicalchecker.core.sign0 import sign0
        from chemicalchecker.core.sign1 import sign1
        from chemicalchecker.core.sign2 import sign2
        from chemicalchecker.core.sign3 import sign3
        from chemicalchecker.core.sign4 import sign4

        from chemicalchecker.core.clus import clus
        from chemicalchecker.core.neig import neig  # nearest neighbour class
        from chemicalchecker.core.proj import proj
        from .char import char

        # DataFactory.__log.debug("initializing object %s", cctype)
        if cctype[:4] in ['clus', 'neig', 'proj', 'diag', 'char']:
            # NS, will return an instance of neig or of sign0 etc
            return eval(cctype[:4])(*args, **kwargs)
        else:
            return eval(cctype)(*args, **kwargs)
