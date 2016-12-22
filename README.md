# Info

There are 2 nodes: 

1. `fluent_extractor`: Subscribes to `vision_buffer_pcl` and extracts pointcloud features
2. `vision_buffer`: Publishes cloth pointcloud to `vision_buffer_pcl` 


# Setup

0. Install dependencies

Install `pcl_ros` as well as `pcl`
  - How to install `pcl_ros` in Kinetic `sudo apt-get install ros-kinetic-pcl-ros`
  - How to install `pcl` in Ubuntu 16.04: https://larrylisky.com/2016/11/03/point-cloud-library-on-ubuntu-16-04-lts/

1. Follow steps 1-3 http://sdk.rethinkrobotics.com/wiki/Workstation_Setup

Install the appropriate ROS version. Kinectic works on Ubuntu 16.04, whereas Indigo works on Ubuntu 14.04.

2. Put this `fluent_extractor` code in `ros_ws/src/`. Then run the following (and resolve build errors by installing required libraries):

        $ catkin_make

3. Edit the `config.json` file located `ros_ws/src/fluent_extractor`.

4. Start the `fluent_extractor`

        $ rosrun fluent_extractor fluent_extractor

3. Run the `vision_buffer`

        $ rosrun fluent_extractor vision_buffer src/fluent_extractor/config.json
        

        
