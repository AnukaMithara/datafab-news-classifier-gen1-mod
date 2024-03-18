import os
import shutil
import tempfile
from pathlib import Path
import joblib


class Classifier:

    """
    Classifier class to load the model and transformer and transform the data
    """

    def __init__(self, file_path):
        """ Constructor to initialize the Classifier class

           Parameters
           ----------
               file_path : str
                    The path to the model file

           Returns
           -------

           Author
           ------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech

           Developers
           ----------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech
        """
        self.model = {}
        self.load(file_path)

    def load(self, path):
        """ Load the model and transformer from the given path

           Parameters
           ----------
                 path : str
                      The path to the model file


           Returns
           -------
                 self : object
                      Returns the instance itself.

           Author
           ------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech

           Developers
           ----------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech
        """

        path = Path(path).absolute()
        file_name = path.name

        with tempfile.TemporaryDirectory() as tmp_dir_path:
            tmp_zip_path = Path(tmp_dir_path) / (file_name + '.zip')
            tmp_zip_path = tmp_zip_path.absolute()
            shutil.copyfile(path, tmp_zip_path)
            shutil.unpack_archive(tmp_zip_path, extract_dir=tmp_dir_path)

            tmp_model_dir_path = Path(tmp_dir_path)

            model_file_name = 'model.pkl'
            model_file_path = os.path.join(tmp_model_dir_path, model_file_name)
            self.model['model'] = joblib.load(model_file_path)

            transformer_file_name = 'transformer.pkl'
            transformer_file_path = os.path.join(tmp_model_dir_path, transformer_file_name)
            self.model['transformer'] = joblib.load(transformer_file_path)

        return self

    def transform_preprocessing(self, data):
        """ Transform the given data using the transformer

           Parameters
           ----------
                data : str
                    The data to transform

           Returns
           -------
                preprocessed_data : object
                    Returns the transformed data

           Author
           ------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech

           Developers
           ----------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech
        """

        preprocessed_data = self.model['transformer'].transform(data)
        return preprocessed_data
