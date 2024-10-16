# FoundationPoseROS2 Multi-Object Pose Estimation and Tracking using ROS2 and RealSense2

<p align="center">
    <img src="assets/demo.gif" alt="Demo Video" style="width: 100%; height: auto;"/>
</p>

FoundationPoseROS2 is a ROS2-integrated system for 6D object pose estimation and tracking, based on the FoundationPose architecture. It uses RealSense2 with the Segment Anything Model 2 (SAM2) framework for end-to-end, model-based, real-time pose estimation and tracking of novel objects.

It is built on top of https://github.com/NVlabs/FoundationPose and https://github.com/Kaivalya192/live-pose.

The main advantages to the previous repositories and isaac_ros_foundationpose:
1. ROS2-based real-time framework that works with 8GiB GPU, unlike Isaac Ros FoundationPose which requires more than 64GiB GPU.
2. SAM2-based automatic segmentation of the objects
3. Multi-object pose estimation and tracking
4. End-to-end assignment of object models with the segmented masks

Furthermore, it provides an interactive GUI for object selection and reordering.

## Features

- **Object Selection GUI**: Choose and reorder object files (.obj, .stl) using a simple Tkinter GUI.
- **Segmentation and Tracking**: Uses SAM2 for object segmentation in real-time color and depth images from a camera.
- **Pose Estimation**: Calculates and publishes the pose of detected objects based on camera images.
- **3D Visualization**: Visualize the objects’ pose with bounding boxes and axes.

## Pipeline

<p align="center">
    <img src="assets/pipeline.svg" alt="Algorithm Pipeline" style="width: 30%; height: auto;"/>
</p>
