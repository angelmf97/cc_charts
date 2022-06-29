from chemicalchecker import ChemicalChecker
from .data import DataFactory

class ChemicalChecker(ChemicalChecker):
    """ChemicalChecker class."""
    def get_signature(self, cctype, molset, dataset_code, *args, **kwargs):
        """Return the signature for the given dataset code.

        Args:
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*).
            molset(str): The molecule set name.
            dataset_code(str): The dataset code of the Chemical Checker.
            params(dict): Optional. The set of parameters to initialize and
                compute the signature. If the signature is already initialized
                this argument will be ignored.
        Returns:
            data(Signature): A `Signature` object, the specific type depends
                on the cctype passed.
        """
        signature_path = self.get_signature_path(cctype, molset, dataset_code)

        # the factory will return the signature with the right class
        data = DataFactory.make_data(
            cctype, signature_path, dataset_code, *args, **kwargs)
        return data
