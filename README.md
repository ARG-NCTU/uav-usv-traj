# USV and UAV Trajectory Visualization
## This repo will use **movingpandas** for plotting the USV and UAV's trajaectory.

clone the repo
```
git clone git@github.com:ARG-NCTU/uav-usv-traj.git
```

### In terminal 1 : 
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
source colab_jupyter.sh
```


