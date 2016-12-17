# Info

There are 2 nodes: 

1. `fluent_extractor`: Subscribes to `vision_buffer_pcl` and extracts pointcloud features
2. `vision_buffer`: Publishes cloth pointcloud to `vision_buffer_pcl` 


# Setup

1. Follow steps 1-3 http://sdk.rethinkrobotics.com/wiki/Workstation_Setup

2. Put this `fluent_extractor` code in `ros_ws/src/`. Then run the following (and resolve build errors by installing required libraries):

        $ catkin_make

3. Edit the `config.json` file located `ros_ws/src/fluent_extractor`.

4. Start the `fluent_extractor`

        $ rosrun fluent_extractor fluent_extractor

3. Run the `vision_buffer`

        $ rosrun fluent_extractor vision_buffer src/fluent_extractor/config.json
        

        
