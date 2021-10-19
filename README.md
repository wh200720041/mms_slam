# MMS-SLAM
## Multi-modal semantic SLAM in dynamic environments (Intel Realsense L515 as an example)

This code is modified from [SSL_SLAM](https://github.com/wh200720041/ssl_slam) 

**Modifier:** [Wang Han](http://wanghan.pro), Nanyang Technological University, Singapore

## 0. AGV dataset is available online! (optional)
### 0.1
training
```
roscd mms_slam
cd train
```
training
```
./dist_train.sh train_param.py 1
```
if you have multiple gpu (say 4 gpus), you can change '1' to your GPU number



## 1. Solid-State Lidar Sensor Example
### 1.1 Scene reconstruction
<p align='center'>
<a href="https://youtu.be/Ox7yDx6JslQ">
<img width="65%" src="/img/3D_reconstruction.gif"/>
</a>
</p>

### 1.2 SFM building example
<p align='center'>
<img width="65%" src="/img/3D_reconstruction.png"/>
</p>

### 1.3 Localization and Mapping in dynamic environments
<p align='center'>
<a href="https://youtu.be/tmWCrredJGI">
<img width="65%" src="/img/3D_SLAM.gif"/>
</a>
</p>

## 2. Prerequisites
### 2.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 18.04.

ROS Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 2.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 2.3. **PCL**
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

Tested with 1.8.1

### 2.4 **OctoMap**
Follow [OctoMap Installation](http://wiki.ros.org/octomap).

```bash
$ sudo apt install ros-melodic-octomap*
```

### 2.5. **Trajectory visualization**
For visualization purpose, this package uses hector trajectory sever, you may install the package by 
```
sudo apt-get install ros-melodic-hector-trajectory-server
```
Alternatively, you may remove the hector trajectory server node if trajectory visualization is not needed


## 3. Build 
### 3.1 Clone repository:
```
    cd ~/catkin_ws/src
    git clone https://github.com/wh200720041/mms_slam.git
    cd ..
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```
chmod python file 
```
roscd mms_slam
cd src
chmod +x solo_node.py
```

### 3.2 install mmdetection 
create conda environment (you need to install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) first) 
```
conda create -n solo python=3.7 -y
conda activate solo
```

install PyTorch and torchvision following the [official instruction](https://pytorch.org/get-started/previous-versions/) (find your cuda version)
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install -c conda-forge addict rospkg pycocotools
```
install mmdet 2.0
```
roscd mms_slam 
cd dependencies/mmdet
python setup.py install
```
it takes a while (a few minutes to install)

### 3.3 Download test rosbag and model
You may download our [trained model](https://drive.google.com/file/d/10ZwHyT7Ql1DYofe4p1jCEAj4rEfg499J/view?usp=sharing) and [recorded data](https://drive.google.com/file/d/1XX4M-aB5aFtj7gPMJKVAeRICdEEAT-EG/view?usp=sharing) if you dont have realsense L515, and by defult the file should be under /home/username/Downloads

put model under mms_slam/config/  
```
cp ~/Downloads/trained_model.pth ~/catkin_ws/src/MMS_SLAM/config/
```
unzip rosbag file under Download folder
```
cd ~/Downloads
unzip ~/Downloads/dynamic_warehouse.zip
```

### 3.4 Launch ROS
if you would like to create the map at the same time, you can run 
```
    roslaunch mms_slam mms_slam_mapping.launch
```

if only localization is required, you may refer to run
```
    roslaunch mms_slam mms_slam.launch
```

if you would like to test instance segmentation results only , you can run
```
    roslaunch mms_slam mms_slam_detection.launch
```

if ModuleNotFoundError: No module named 'alfred', install alfrey-py from pip install
```
pip install alfred-py
```

## 4. Sensor Setup
If you have new Realsense L515 sensor, you may follow the below setup instructions

### 4.1 L515
<p align='center'>
<img width="35%" src="/img/realsense_L515.jpg"/>
</p>

### 4.2 Librealsense
Follow [Librealsense Installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)

### 4.3 Realsense_ros
Copy [realsense_ros](https://github.com/IntelRealSense/realsense-ros) package to your catkin folder
```
    cd ~/catkin_ws/src
    git clone https://github.com/IntelRealSense/realsense-ros.git
    cd ..
    catkin_make
```

### 4.4 Launch ROS with live L515 camera data
In you launch file, uncomment realsense node like this 
```
    <include file="$(find realsense2_camera)/launch/rs_camera.launch">
        <arg name="color_width" value="1280" />
        <arg name="color_height" value="720" />
        <arg name="filters" value="pointcloud" />
    </include>
```
and comment rosbag play like this 
```
<!-- rosbag
    <node name="bag" pkg="rosbag" type="play" args="- -clock -r 0.4 -d 5 $(env HOME)/Downloads/dynamic_warehouse.bag" />
    <param name="/use_sim_time" value="true" />  
-->
```

## 5 Acknowlegement
Thanks for [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) and [LOAM](https://github.com/laboshinl/loam_velodyne) and [LOAM_NOTED](https://github.com/cuitaixiang/LOAM_NOTED) and [MMDetection](https://github.com/open-mmlab/mmdetection) and [SOLO](https://github.com/WXinlong/SOLO).

