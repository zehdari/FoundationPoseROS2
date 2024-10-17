# FoundationPoseROS2 Multi-Object Pose Estimation and Tracking of Novel Objects using ROS2 and RealSense2

<p align="center">
  <img src="assets/demo.gif" alt="Demo Video" width="330">
  <img src="assets/demo_robot.gif" alt="Robot Demo Video" width="434"><br>
</p>

FoundationPoseROS2 is a ROS2-integrated system for 6D object pose estimation and tracking, based on the FoundationPose architecture. It uses RealSense2 with the Segment Anything Model 2 (SAM2) framework for end-to-end, model-based, real-time pose estimation and tracking of novel objects.

It is built on top of [FoundationPose](https://github.com/NVlabs/FoundationPose) and [live-pose](https://github.com/Kaivalya192/live-pose).

The main advantages to the previous repositories and [isaac_ros_foundationpose](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_foundationpose):
1. ROS2-based real-time framework that works with 8GiB GPU, unlike Isaac Ros FoundationPose which requires more than 64GiB GPU.
2. SAM2-based automatic segmentation of the objects
3. Multi-object pose estimation and tracking
4. End-to-end assignment of object models with the segmented masks

Furthermore, it provides an interactive GUI for object model-to-mask assignment for end-to-end multi-pose estimation and tracking.

## Tutorial

[<video width="640" height="360" controls>
  <source src="[assets/tutorial.mp4](https://github.com/ammar-n-abbas/FoundationPoseROS2/blob/a73eb8412972296c8cf5c55fde940eed4e8d159a/assets/tutorial.mp4)" type="video/mp4">
  Your browser does not support the video tag.
</video>](https://github.com/ammar-n-abbas/FoundationPoseROS2/blob/a73eb8412972296c8cf5c55fde940eed4e8d159a/assets/tutorial.mp4)

## Env setup: conda 

```bash
# Create conda environment
conda create -n foundationpose_ros python=3.10

# Activate conda environment
conda activate foundationpose_ros

# Install dependencies
python -m pip install -r requirements.txt
```

## Features

- **Object Selection GUI**: Choose and reorder object files (.obj, .stl) using a simple Tkinter GUI.
- **Segmentation and Tracking**: SAM2 is used for object segmentation in real-time colour and depth images from a camera.
- **Pose Estimation**: Calculates and publishes the pose of detected objects based on camera images.
- **3D Visualization**: Visualize the objects’ pose with bounding boxes and axes.

## Pipeline

<p align="center">
    <img src="assets/pipeline.svg" alt="Algorithm Pipeline" style="width: 30%; height: auto;"/>
</p>
