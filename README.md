# Intro

This is the official repository for the paper *Learning Human Utility from Video Demonstrations for Deductive Planning in Robotics* (CoRL 2017). We are cleaning up the code and documentation. Please stay tuned.

If you use this code, please cite:

    @inproceedings{shukla17,
      title={Learning Human Utility from Video Demonstrations for Deductive Planning in Robotics},
      author={Shukla, Nishant and He, Yunzhong and Chen, Frank and Zhu, Song-Chun},
      booktitle={The Conference on Robot Learning (CoRL)},
      year={2017}
    }

# Overview of system

There are 2 nodes: 

1. `fluent_extractor`: Subscribes to `vision_buffer_pcl` and extracts pointcloud features
2. `vision_buffer`: Publishes cloth pointcloud to `vision_buffer_pcl` 

# Setup

1. Install dependencies

  Install `pcl_ros` as well as `pcl`
  - Install `pcl_ros` in Kinetic: `sudo apt-get install ros-kinetic-pcl-ros`
  - Install `pcl` in Ubuntu 16.04: https://larrylisky.com/2016/11/03/point-cloud-library-on-ubuntu-16-04-lts/
  - Install `cv_bridge` by `sudo apt-get install ros-kinetic-cv-bridge`

2. Follow steps 1-3 http://sdk.rethinkrobotics.com/wiki/Workstation_Setup

  Install the appropriate ROS version. Kinectic works on Ubuntu 16.04, whereas Indigo works on Ubuntu 14.04.

3. Put this `fluent_extractor` code in `ros_ws/src/`. Then run the following (and resolve build errors by installing required libraries):

        $ catkin_make

4. Edit the `config.json` file located `ros_ws/src/fluent_extractor`.

5. Start the `fluent_extractor`

        $ rosrun fluent_extractor fluent_extractor

6. Run the `vision_buffer`

        $ rosrun fluent_extractor vision_buffer src/fluent_extractor/config.json
        
