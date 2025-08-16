import os
import shutil

import kagglehub


def retrieve_dataset_from_kaggle(dataset_id: str, target_path: str) -> None:
    if os.path.isfile(target_path):
        print(f"{target_path} already exists. Dataset will not be downloaded.")
        return

    print(f"Downloading {dataset_id}...")
    path = kagglehub.dataset_download(dataset_id)
    shutil.move(path, target_path)

    print(f"Dataset downloaded to {target_path}.")
