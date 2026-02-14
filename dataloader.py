import os
import urllib.request
import requests
import zipfile
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

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
        return {str(file).split('\\')[-1]: file for file in x}

    @classmethod
    def get_internal_image_ids(cls, directory: str):
        x = [f for f in os.listdir(directory)]
        return sorted(x, key = lambda x: int(x.split('_')[-1].split('.')[0]))


class DataViewer:
    _methods : ['inspect_16b_8b', 'view_csv']
    @classmethod
    def view_csv(cls, path: str):
        if not os.path.exists(path):
            return f"Error: File not found at {path}"
    
        # Load and immediately sort by image_id
        df = pd.read_csv(path)
        if 'image_id' in df.columns:
            df = df.sort_values(by='image_id').reset_index(drop=True)
    
        print(f"Dataset Shape: {df.shape}")
        print("-" * 30)
        print("Column Types:")
        print(df.dtypes)
        print("-" * 30)
    
        return df.head()


    @classmethod
    def inspect_16b_8b(loader):
        batch = next(iter(loader))
        # Un-normalize RGB for display
        rgb = batch['rgb'][0].permute(1, 2, 0).numpy()
        rgb = (rgb * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    
        depth = batch['depth'][0].squeeze().numpy()
    
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(np.clip(rgb, 0, 1)); plt.title("RGB Input")
        plt.subplot(1, 2, 2); plt.imshow(depth, cmap='magma'); plt.title("Corrected Depth Map")
        plt.show()
    
        print(f"Tabular Inputs (Common): {batch['tab_in'][0]}")
        print(f"Targets (Aux + Label): {batch['aux_targets'][0]} | {batch['label'][0]}")

    @classmethod
    def inspect_sample(cls, sample, sample_idx=0):
        # 1. RGB (High-Fidelity 70% Res)
        rgb = sample['rgb'].permute(1, 2, 0).numpy()
        rgb = (rgb * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        
        # 2. Advanced Local Contrast Enhancement (CLAHE)
        depth_raw = sample['depth'].squeeze().numpy().astype(np.float32)
        
        # Step A: Normalize to 0-255 for OpenCV's CLAHE
        v_min, v_max = np.percentile(depth_raw, [2, 98])
        depth_norm = np.clip(depth_raw, v_min, v_max)
        depth_norm = ((depth_norm - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        
        # Step B: Create CLAHE object (clipLimit=3.0, tileGridSize=(8,8))
        # This amplifies the tiny height differences on the leaf surface
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        depth_enhanced = clahe.apply(depth_norm)
        
        # Step C: Edge Sharpening (Unsharp Mask)
        blurred = cv2.GaussianBlur(depth_enhanced, (5, 5), 0)
        sharpened = cv2.addWeighted(depth_enhanced, 1.5, blurred, -0.5, 0)

        # 3. Final Visualization with the 'Turbo' colormap (very high edge visibility)
        plt.figure(figsize=(16, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(rgb, 0, 1))
        plt.title(f"Row {sample_idx}: RGB")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # 'turbo' is much more aggressive for finding edges than 'plasma'
        plt.imshow(sharpened, cmap='turbo') 
        plt.title("Final Local Contrast Enhancement (CLAHE)")
        plt.axis('off')
        plt.show()

        print(f"Features: {sample['tab_in'].tolist()} | DryWeight: {sample['label'].item():.4f}")
           

