# USV and UAV Trajectory Visualization
## This repo will use **movingpandas** for plotting the USV and UAV's trajaectory.

### 1. Prepare dataset
Make the data directory to prepare dataset.
```
cd ~/uav-usv-traj
mkdir data
cd data
```

Run this command to download raw kml files of 6897 UAV's trajaectories dataset from NAS.
```
wget ftp://140.113.148.83/arg-projectfile-download/uav-usv-traj/raw_kml.zip
```

Or, collect your own trajaectory data. This [link](https://docs.google.com/document/d/1mWLEjzz1vDetMLI1GxP4AXXHkgWak5V9Ur3kOQ9WKCw/edit?usp=sharing) guides you to get kml files from ardupilot px4. Converted result will be like this [link](http://gofile.me/773h8/XKIs8EA2K). 


### 2. Clone the repo
```
git clone git@github.com:ARG-NCTU/uav-usv-traj.git
```

### 3. Usage
#### 3.1. Preprocessing data (Optional): 
run python script to get base timestamps with method 1 (kml files from ardupilot px4):
```
python3 get-basetimestamps.py --kml_dir ./data/ardupilot-logs/kmls --method 1 --csv ./data/ardupilot-logs-timestamps.txt
```
run python script to get base timestamps with method 2 (6897 UAV's trajaectories dataset):
```
python3 get-basetimestamps.py --kml_dir ./data/raw_kml --method 2 --csv ./data/uav-dataset-timestamps.txt
```

#### 3.2. Run the notebooks 
run pyivp's docker 
```
source docker_run.sh
```
you will run the argnctu/pyivp:latest docker image and the current directory will be mounted to /workspace.

**loging hugging face with token**: 
```
source hf_login.sh
```

run the jupyter notebook
```
source jupyter.sh
```

#### 3.3. Chat with GPT API
run pyivp's docker 
```
source docker_run.sh
```

run streamlit api to select gpt model, upload csv file and ask question
```
streamlit run langchain_csv.py
```

Example question:
What is the average of all of all distance of trajectories? Hint: Use for loop to sum all distance of trajectories and division by number of trajectory. DO NOT use sum function of pandas.