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
                zip_ref.extractall(save_path)
                cls._xtract_paths[os.path.splitext(file)[0]] = save_path
                
        return cls._xtract_paths        
