import os
import urllib.request
import requests
import zipfile
import torch

class DataExtractor:
    _xtract_paths = dict()
    @classmethod
    def extract(cls, extract_from: str , extract_to: str):
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)

        files = [f for f in os.listdir(extract_from) if f.endswith('.zip')]

        for file in files:
            zip_path = os.path.join(extract_from, file)
            save_path = os.path.join(extract_to, os.path.splitext(file)[0])
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                top_level = {m.split('/')[0] for m in members}
                if len(top_level) == 1:
                    zip_ref.extractall(extract_to)
                    cls._xtract_paths[os.path.splitext(file)[0]] = os.path.join(extract_to, list(top_level).pop())

                else:
                    zip_ref.extractall(save_path)
                    cls._xtract_paths[os.path.splitext(file)[0]] = save_path    
                
        return cls._xtract_paths        

    @classmethod
    def get_internal_paths(cls, directory: str):
        x = [os.path.join(directory, file) for file in os.listdir(directory)]
        return {str(file).split('\\')[-1]: os.path.join(directory, file) for file in x}


