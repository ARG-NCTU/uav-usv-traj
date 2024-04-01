import numpy as np
import os
import add_path
from hf_tools.hf_sync import *


def test_download_file():
    download_file(file_path='../data/', 
                  repo_id="ARG-NCTU/uav-usv-traj", 
                  repo_type="dataset", 
                  hf_path='testing.csv')
    assert os.path.isfile("../data/testing.csv")
    os.remove("../data/testing.csv")