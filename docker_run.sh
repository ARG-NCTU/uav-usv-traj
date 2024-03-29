#!/usr/bin/env bash

ARGS=("$@")

# Host home path
HOSTHOME=/home/$USER/uav-usv-traj
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    HOSTHOME=/Users/$USER/uav-usv-traj
fi

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
    echo "[$XAUTH] was not properly created. Exiting..."
    exit 1
fi

BASH_OPTION=bash

docker run \
    -it \
    --rm \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "$XAUTH:$XAUTH" \
    -v "$HOSTHOME:/home/arg/uav-usv-traj" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -w "/home/arg/uav-usv-traj" \
    --user "root:root" \
    --name uav_usv_traj \
    --network host \
    --privileged \
    --security-opt seccomp=unconfined \
    argnctu/pyivp:latest \
    $BASH_OPTION
