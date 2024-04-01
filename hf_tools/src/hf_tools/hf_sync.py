from datasets import load_dataset
import os
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import shutil



def upload_dataset(dataset_path="testing.csv", 
                   dataset_type="csv", 
                   dataset_repo="ARG-NCTU/uav-usv-traj"):
    
    if os.path.isfile(dataset_path):
        print("File exists")
    else:
        print("File does not exist")
        return
    
    dataset = load_dataset(dataset_type, dataset_path)
    dataset.push_to_hub(dataset_repo)

def download_dataset(dataset_path="data/testing.csv", 
                     dataset_type="csv", 
                     dataset_repo="ARG-NCTU/uav-usv-traj"):
    
    dataset = load_dataset(dataset_type, dataset_repo)
    dataset.save_to_disk(dataset_path)

def upload_file(file_path='testing.csv', 
                repo_id="ARG-NCTU/uav-usv-traj", 
                repo_type="dataset", 
                hf_path=''):
    try:
        os.path.isfile(file_path)
        print("File exists")
    except FileNotFoundError:
        print("File does not exist")
        return
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.join(hf_path, os.path.basename(file_path)),
        repo_id=repo_id,
        repo_type=repo_type,
    )

def download_file(file_path='data/', 
                  repo_id="ARG-NCTU/uav-usv-traj", 
                  repo_type="dataset", 
                  hf_path='testing.csv'):
    
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    hf_hub_download(repo_id=repo_id, 
                    subfolder=os.path.dirname(hf_path), 
                    filename=os.path.basename(hf_path), 
                    repo_type=repo_type,
                    local_dir=os.path.dirname(file_path)
                    )

if __name__ == "__main__":
    download_file()
