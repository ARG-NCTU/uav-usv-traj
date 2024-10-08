# USV and UAV Trajectory Visualization
## This repo will use **movingpandas** for plotting the USV and UAV's trajaectory.

### 1. Clone the repo
```
git clone git@github.com:ARG-NCTU/uav-usv-traj.git
```

### 2. Prepare dataset
Make the data directory to prepare dataset.
```
cd ~/uav-usv-traj
mkdir data
cd data
```

Run this command to download raw kml files of 6897 UAV's trajaectories dataset from NAS.
```
wget ftp://140.113.148.83/arg-projectfile-download/uav-usv-traj/raw_kml.zip
unzip raw_kml.zip
```

Or, collect your own trajaectory data. This [link](https://docs.google.com/document/d/1mWLEjzz1vDetMLI1GxP4AXXHkgWak5V9Ur3kOQ9WKCw/edit?usp=sharing) guides you to get kml files from ardupilot px4. Converted result will be like this [link](http://gofile.me/773h8/XKIs8EA2K). 

### 3. Usage
#### 3.1. Preprocessing data
Run python script to get base timestamps (6897 UAV's trajaectories dataset):
```
python3 get-basetimestamps.py --kml_dir ./data/raw_kml --method 2 --csv ./data/uav-dataset-timestamps.txt
```
Or, if you collect your own trajaectory data. Run python script to get base timestamps (kml files from ardupilot px4):
```
python3 get-basetimestamps.py --kml_dir ./data/ardupilot-logs/kmls --method 1 --csv ./data/ardupilot-logs-timestamps.txt
```

#### 3.2. Run the notebooks 
Run pyivp's docker 
```
source docker_run.sh
```

Run the jupyter notebook
```
source jupyter.sh
```

#### 3.3. Chat with GPT API and tarjectory analysis
Run pyivp's docker 
```
source docker_run.sh
```

Run streamlit api to select gpt model and ask question
```
streamlit run langchain_csv.py
```

Example question & answer:

Which trajectory in the period from 2023/07/16 9:30AM to 2023/07/16 10:00AM is the shortest?
```
The shortest trajectory in the period from 2023/07/16 9:30AM to 2023/07/16 10:00AM is the one with the file path "./data/uav-csv/飛行軌跡_20230716093620_R0116350320.csv".
```

#### 3.4. Click the location on map and time analysis
Run pyivp's docker 
```
source docker_run.sh
```

Run streamlit api to analysis trajectories
```
streamlit run folium_time_analysis.py
```

#### 3.5. Click the location on map and tarjectory analysis
Run pyivp's docker 
```
source docker_run.sh
```

Run streamlit api to analysis trajectories
```
streamlit run folium_traj_analysis.py
```

#### 3.6. Visualize time stats of tarjectory analysis
Run pyivp's docker 
```
source docker_run.sh
```

Run streamlit api to analysis trajectories
```
python3 time_stats.py
```