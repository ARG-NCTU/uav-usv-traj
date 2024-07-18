# USV and UAV Trajectory Visualization
## This repo will use **movingpandas** for plotting the USV and UAV's trajaectory.

### 1. Prepare dataset
Download raw kml files of 6897 UAV's trajaectories dataset from [NAS](http://gofile.me/773h8/lDHkwDMWv).
Or, collect your own trajaectory data. This [link](https://docs.google.com/document/d/1mWLEjzz1vDetMLI1GxP4AXXHkgWak5V9Ur3kOQ9WKCw/edit?usp=sharing) guides you to get kml files from ardupilot px4. Converted result will be like this [link](http://gofile.me/773h8/XKIs8EA2K). 

### 2. Clone the repo
```
git clone git@github.com:ARG-NCTU/uav-usv-traj.git
```

### 3. Usage
#### 3-1. Preprocessing data (Optional): 
run python script to get base timestamps with method 1 (kml files from ardupilot px4):
```
python3 get-basetimestamps.py --kml_dir ./data/ardupilot-logs/kmls --method 1 --csv ./data/ardupilot-logs-timestamps.txt
```
run python script to get base timestamps with method 2 (6897 UAV's trajaectories dataset):
```
python3 get-basetimestamps.py --kml_dir ./data/raw_kml --method 2 --csv ./data/uav-dataset-timestamps.txt
```

#### 3-2. Run the notebooks 
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